#!/usr/bin/env python3
"""Cut paused tickets the same way Albedo eval does.

No model. No training. Reads parquet (or a fake row) and writes prefix jsonl.

Cut rule (copied from albedo, do not "improve"):
  sample_id is shard:row:turn_idx
  turn_idx = 1 or 2                  if cold
           = first_edit - 2          if pre_edit (floor 1)
           = first_edit              if at_edit
  prefix  = all messages BEFORE the assistant turn at that index
            (see albedo_eval_service.remote.dataset._prompt_from_row)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterator

# Same regex the eval manifest uses to find the first source edit.
# Source: albedo/scripts/build_manifest.py
_EDIT_RE = re.compile(
    r"sed\s+-i|tee\s+[\w./-]|cat\s*>|str_replace|git apply|patch\s+-p|applypatch|"
    r"cp\s+[\w./-]|mv\s+[\w./-]|(?<![-\d&])>>?\s*(?!/dev/)[\w.][\w./-]*"
)

# Same leaks the eval drops. Source: albedo/scripts/prepare_datasets.py
_LEAKS = {
    "mini-coder": frozenset(
        ("modin-project__modin.8c7799fd.pr_7434", "scrapy__scrapy.35212ec5.pr_6671")
    ),
    "open-swe-traces": frozenset(
        (
            "agronholm__anyio-935",
            "astropy__ccdproc-901",
            "beeware__briefcase-2302",
            "beeware__briefcase-2401",
            "conan-io__conan-18327",
            "conan-io__conan-18444",
            "ethereum__web3.py-3690",
            "geopandas__geopandas-3591",
            "matthewwithanm__python-markdownify-230",
            "pdm-project__pdm-3575",
            "pydata__sparse-870",
        )
    ),
}

SOURCES = ("mini-coder", "open-swe-traces", "swe-hero", "mini-coder-rs")
MAX_PREFIX_CHARS = 54_000

# v125 leftover: verify + action still ~0.5. Stay close to live draw, nudge near-edit.
# See 07-gpu-dataset.md. Eval shares stay in --pack eval.
OVERFIT_PHASE = [("cold", 55), ("pre_edit", 25), ("at_edit", 20)]
OVERFIT_SOURCE = [
    ("mini-coder", 57),
    ("open-swe-traces", 37),
    ("swe-hero", 4),
    ("mini-coder-rs", 2),
]
EVAL_PHASE = [("cold", 65), ("pre_edit", 15), ("at_edit", 20)]


def family_of(instance_id: str, source: str) -> str:
    if source in ("open-swe-traces", "swe-hero"):
        return "pr"
    if "." not in instance_id:
        return "pr"
    tail = instance_id.rsplit(".", 1)[-1]
    for prefix, family in (("pr_", "pr"), ("lm_", "lm"), ("combine", "combine")):
        if tail.startswith(prefix):
            return family
    return "mechanical"


def language_of(source: str, row: dict[str, Any] | None = None) -> str:
    if row and row.get("language"):
        return str(row["language"])
    if source == "mini-coder-rs":
        return "rust"
    if source in ("mini-coder", "swe-hero"):
        return "python"
    return "mixed"


def role_of(turn: Any) -> str:
    if not isinstance(turn, dict):
        return ""
    return str(turn.get("role") or turn.get("speaker") or "").lower()


def content_of(turn: Any) -> str:
    if not isinstance(turn, dict):
        return str(turn) if turn is not None else ""
    return str(turn.get("content") or turn.get("text") or "")


def first_edit_index(messages: list[Any]) -> int:
    """1-based assistant index of the first source edit. 0 if none."""
    asst = 0
    for turn in messages:
        if role_of(turn) != "assistant":
            continue
        asst += 1
        if _EDIT_RE.search(content_of(turn)):
            return asst
    return 0


def assistant_count(messages: list[Any]) -> int:
    return sum(1 for t in messages if role_of(t) == "assistant")


def turn_idx_for(phase: str, first_edit: int, *, cold_pick: int) -> int:
    """Same as sampling._turn_idx. cold_pick is 1 or 2."""
    if phase == "cold":
        return 1 if cold_pick not in (1, 2) else cold_pick
    if first_edit <= 0:
        return 1
    return max(1, first_edit - 2) if phase == "pre_edit" else first_edit


def cut_prefix(messages: list[Any], turn_idx: int) -> list[dict[str, str]] | None:
    """Prefix is everything BEFORE assistant_turns[turn_idx].

    Need strictly more assistant turns than turn_idx (eval skips otherwise).
    """
    assistant_at = [i for i, t in enumerate(messages) if role_of(t) == "assistant"]
    if turn_idx >= len(assistant_at):
        return None
    head = messages[: assistant_at[turn_idx]]
    out: list[dict[str, str]] = []
    for turn in head:
        text = content_of(turn)
        if not text:
            continue
        role = role_of(turn)
        if role not in {"assistant", "system", "user"}:
            role = "user"
        out.append({"role": role, "content": text})
    return out


def prefix_phase(messages: list[dict[str, str]]) -> str:
    """How eval will label this prefix (sample_phase)."""
    asst = [m["content"] for m in messages if m.get("role") == "assistant"]
    if any(_EDIT_RE.search(t or "") for t in asst):
        return "at_edit"
    return "cold" if len(asst) <= 2 else "pre_edit"


def cold_pick(instance_id: str) -> int:
    return 1 if int(hashlib.sha256(instance_id.encode()).hexdigest(), 16) % 2 == 0 else 2


def iter_parquet_rows(path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    import pyarrow.parquet as pq

    parquet = pq.ParquetFile(path)
    seen = 0
    cols = [c for c in parquet.schema_arrow.names if c in ("instance_id", "messages", "repo", "language", "family", "first_edit")]
    for batch in parquet.iter_batches(batch_size=256, columns=cols or None):
        for local, row in enumerate(batch.to_pylist()):
            yield seen + local, row
        seen += batch.num_rows


def scan_shard(path: Path, source: str) -> Iterator[dict[str, Any]]:
    leaks = _LEAKS.get(source, frozenset())
    shard_name = f"{source}/data/{path.name}"
    for row_idx, row in iter_parquet_rows(path):
        iid = str(row.get("instance_id") or "")
        if not iid or iid in leaks:
            continue
        messages = row.get("messages") or []
        if not isinstance(messages, list):
            continue
        asst = assistant_count(messages)
        given = row.get("first_edit")
        first_edit = int(given) if given is not None else first_edit_index(messages)
        yield {
            "source": source,
            "shard": shard_name,
            "row": row_idx,
            "instance_id": iid,
            "repo": str(row.get("repo") or iid.split(".")[0]),
            "language": language_of(source, row),
            "family": str(row.get("family") or family_of(iid, source)),
            "asst": asst,
            "first_edit": first_edit,
            "messages": messages,
        }


def can_cut(entry: dict[str, Any], phase: str, turn_idx: int) -> bool:
    if entry["asst"] <= turn_idx:
        return False
    prefix = cut_prefix(entry["messages"], turn_idx)
    if prefix is None:
        return False
    n_chars = sum(len(m["content"]) for m in prefix)
    if phase != "cold" and n_chars > MAX_PREFIX_CHARS:
        return False
    return True


def make_record(entry: dict[str, Any], phase: str, turn_idx: int, horizon: int) -> dict[str, Any]:
    prefix = cut_prefix(entry["messages"], turn_idx)
    assert prefix is not None
    labeled = prefix_phase(prefix)
    return {
        "sample_id": f"{entry['shard']}:{entry['row']}:{turn_idx}",
        "phase_wanted": phase,
        "phase_labeled": labeled,
        "turn_idx": turn_idx,
        "horizon": horizon,
        "source": entry["source"],
        "instance_id": entry["instance_id"],
        "repo": entry["repo"],
        "language": entry["language"],
        "family": entry["family"],
        "first_edit": entry["first_edit"],
        "asst_in_row": entry["asst"],
        "asst_in_prefix": sum(1 for m in prefix if m["role"] == "assistant"),
        "prefix_chars": sum(len(m["content"]) for m in prefix),
        "messages": prefix,
    }


def _apportion(spec: list[tuple[str, int]], count: int) -> dict[str, int]:
    total = sum(s for _, s in spec)
    exact = [(n, count * s / total) for n, s in spec]
    out = {n: int(v // 1) for n, v in exact}
    rem = count - sum(out.values())
    order = sorted(enumerate(exact), key=lambda it: (-(it[1][1] % 1), it[0]))
    for _, (n, _) in order[:rem]:
        out[n] += 1
    return out


def pick_entries(
    pool: list[dict[str, Any]],
    *,
    count: int,
    pack: str,
    seed: str,
    repo_cap: int,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    rng.shuffle(pool)
    phase_spec = EVAL_PHASE if pack == "eval" else OVERFIT_PHASE
    # overfit: source mix first, then assign a phase each instance can support
    source_want = (
        {s: count for s in SOURCES}
        if pack == "eval"
        else _apportion(OVERFIT_SOURCE, count)
    )
    phase_want = _apportion(phase_spec, count)
    leftover = count - sum(phase_want.values())
    if leftover:
        phase_want["cold"] = phase_want.get("cold", 0) + leftover

    selected: list[dict[str, Any]] = []
    repo_counts: dict[str, int] = defaultdict(int)
    source_got: Counter[str] = Counter()
    phase_got: Counter[str] = Counter()

    def try_add(entry: dict[str, Any], phase: str) -> bool:
        if phase_got[phase] >= phase_want.get(phase, 0):
            return False
        if source_got[entry["source"]] >= source_want.get(entry["source"], count):
            return False
        if repo_counts[entry["repo"]] >= repo_cap:
            return False
        pick = cold_pick(entry["instance_id"]) if phase == "cold" else 0
        turn_idx = turn_idx_for(phase, entry["first_edit"], cold_pick=pick)
        if not can_cut(entry, phase, turn_idx):
            return False
        horizon = 12 if (len(selected) % 2 == 0) else 16
        rec = make_record(entry, phase, turn_idx, horizon)
        selected.append(rec)
        repo_counts[entry["repo"]] += 1
        source_got[entry["source"]] += 1
        phase_got[phase] += 1
        return True

    # Prefer the phase this row is "naturally" good for, then fall back.
    for entry in pool:
        if len(selected) >= count:
            break
        preferred = []
        if entry["first_edit"] >= 4:
            preferred.append("pre_edit")
        if entry["first_edit"] >= 1:
            preferred.append("at_edit")
        preferred.append("cold")
        # honor remaining phase demand: try the hungriest needed phase first
        hungry = sorted(phase_want, key=lambda p: phase_want[p] - phase_got[p], reverse=True)
        order = [p for p in hungry if p in preferred] + [p for p in preferred if p not in hungry]
        for phase in order:
            if try_add(entry, phase):
                break
    return selected


# ---------------------------------------------------------------------------
# Fake conversation for --self-test (always works, no download)
# ---------------------------------------------------------------------------

def demo_prefix(phase: str = "cold") -> dict:
    """One paused fake ticket, same shape as prefixes.jsonl."""
    messages = fake_row()
    first = first_edit_index(messages)
    pick = 2 if phase == "cold" else 1
    turn_idx = turn_idx_for(phase, first, cold_pick=pick)
    prefix = cut_prefix(messages, turn_idx)
    if prefix is None:
        raise RuntimeError(f"cannot cut demo phase={phase}")
    return make_record(
        {
            "source": "demo",
            "shard": "demo/data/train-00000.parquet",
            "row": 0,
            "instance_id": "demo.add.pr_1",
            "repo": "demo",
            "language": "python",
            "family": "pr",
            "asst": assistant_count(messages),
            "first_edit": first,
            "messages": messages,
        },
        phase,
        turn_idx,
        horizon=4,
    )


def fake_row() -> list[dict[str, str]]:
    """8-turn ticket. First source edit is assistant #5."""
    steps = [
        ("user", "Bug: add(1, 2) returns 0. Fix src/math.py."),
        ("assistant", "THOUGHT: look at the file\n\n```bash\ncat src/math.py\n```"),
        ("user", "<returncode>0</returncode>\n<output>\ndef add(a, b):\n    return 0\n</output>"),
        ("assistant", "THOUGHT: confirm the fail\n\n```bash\npython -c 'from src.math import add; print(add(1,2))'\n```"),
        ("user", "<returncode>0</returncode>\n<output>\n0\n</output>"),
        ("assistant", "THOUGHT: still reading\n\n```bash\ngrep -n add src/math.py\n```"),
        ("user", "<returncode>0</returncode>\n<output>\n1:def add(a, b):\n</output>"),
        ("assistant", "THOUGHT: one more look\n\n```bash\nsed -n '1,5p' src/math.py\n```"),
        ("user", "<returncode>0</returncode>\n<output>\ndef add(a, b):\n    return 0\n</output>"),
        (
            "assistant",
            "THOUGHT: fix it\n\n```bash\nsed -i 's/return 0/return a + b/' src/math.py\n```",
        ),
        ("user", "<returncode>0</returncode>\n<output>\n</output>"),
        ("assistant", "THOUGHT: check\n\n```bash\npython -c 'from src.math import add; print(add(1,2))'\n```"),
        ("user", "<returncode>0</returncode>\n<output>\n3\n</output>"),
    ]
    return [{"role": r, "content": c} for r, c in steps]


def run_self_test() -> int:
    messages = fake_row()
    first = first_edit_index(messages)
    print(f"fake row: {assistant_count(messages)} assistant turns, first_edit={first}")
    print("expected first_edit=5 (the sed -i turn)\n")
    if first != 5:
        print(f"FAIL first_edit={first} wanted 5")
        return 1
    ok = True
    for phase, pick in (("cold", 1), ("cold", 2), ("pre_edit", 0), ("at_edit", 0)):
        idx = turn_idx_for(phase, first, cold_pick=pick or 1)
        prefix = cut_prefix(messages, idx)
        labeled = prefix_phase(prefix or [])
        n_asst = sum(1 for m in (prefix or []) if m["role"] == "assistant")
        print(
            f"  {phase:9} cold_pick={pick or '-'}  turn_idx={idx}  "
            f"asst_in_prefix={n_asst}  labeled={labeled}"
        )
        if prefix is None:
            print("    FAIL: cut returned None")
            ok = False
            continue
        if phase == "cold" and labeled != "cold":
            print("    FAIL: cold prefix not labeled cold")
            ok = False
        if phase == "pre_edit" and labeled != "pre_edit":
            print(f"    FAIL: pre_edit labeled {labeled}")
            ok = False
        if phase == "at_edit" and labeled != "at_edit":
            print(f"    FAIL: at_edit labeled {labeled}")
            ok = False
        last = next((m["content"][:60] for m in reversed(prefix) if m["role"] == "assistant"), "")
        print(f"    last assistant in prefix: {last!r}")
    print("\nPASS" if ok else "\nFAIL")
    return 0 if ok else 1


def load_pool(data_root: Path, sources: list[str], max_rows: int | None) -> list[dict[str, Any]]:
    pool: list[dict[str, Any]] = []
    for source in sources:
        data_dir = data_root / source / "data"
        shards = sorted(data_dir.glob("train-*.parquet"))
        if not shards:
            print(f"skip {source}: no shards under {data_dir}", file=sys.stderr)
            continue
        for shard in shards:
            for entry in scan_shard(shard, source):
                pool.append(entry)
                if max_rows is not None and len(pool) >= max_rows:
                    return pool
    return pool


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true", help="cut the fake ticket, no download")
    parser.add_argument("--data-root", type=Path, default=Path(__file__).resolve().parent / "data")
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "out" / "prefixes.jsonl")
    parser.add_argument("--sources", default=",".join(SOURCES))
    parser.add_argument("--count", type=int, default=200, help="how many prefixes to write")
    parser.add_argument("--pack", choices=("overfit", "eval"), default="overfit")
    parser.add_argument("--seed", default="albedo-training-02")
    parser.add_argument("--repo-cap", type=int, default=0, help="0 = 2 for eval pack, 8 for overfit")
    parser.add_argument("--max-scan", type=int, default=None, help="stop after this many raw rows")
    args = parser.parse_args()

    if args.self_test:
        return run_self_test()

    sources = [s.strip() for s in args.sources.split(",") if s.strip()]
    repo_cap = args.repo_cap or (2 if args.pack == "eval" else 8)
    pool = load_pool(args.data_root, sources, args.max_scan)
    if not pool:
        print(
            "no rows. download the corpora into --data-root/<source>/data/ "
            "or run: python cut_prefixes.py --self-test",
            file=sys.stderr,
        )
        return 2

    selected = pick_entries(
        pool, count=args.count, pack=args.pack, seed=args.seed, repo_cap=repo_cap
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as fh:
        for rec in selected:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"scanned {len(pool)} rows, wrote {len(selected)} prefixes -> {args.out}")
    print("phase", dict(Counter(r["phase_wanted"] for r in selected)))
    print("source", dict(Counter(r["source"] for r in selected)))
    print("family", dict(Counter(r["family"] for r in selected)))
    print("labeled", dict(Counter(r["phase_labeled"] for r in selected)))
    mismatch = sum(1 for r in selected if r["phase_wanted"] != r["phase_labeled"])
    print(f"wanted≠labeled {mismatch}/{len(selected)}")
    if len(selected) < args.count:
        print(f"short: wanted {args.count}, got {len(selected)}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
