#!/usr/bin/env python3
"""Turn keep/drop siblings into SFT and DPO rows. Step 05 — files only, no train.

SFT  = every keep (teach the habit).
DPO  = same ticket: keep continuation vs drop continuation.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from cut_prefixes import demo_prefix
from filter_rollouts import judge
from roll_king import fake_king_skip_verify, fake_king_verify, roll_one, stub_observe


def prefix_of(rollout: dict[str, Any]) -> list[dict[str, str]]:
    full = rollout.get("messages") or []
    cont = rollout.get("continuation") or []
    if full and cont:
        return full[: len(full) - len(cont)]
    return []


def sft_row(keep: dict[str, Any]) -> dict[str, Any]:
    prefix = prefix_of(keep)
    return {
        "kind": "sft",
        "sample_id": keep["sample_id"],
        "phase_wanted": keep.get("phase_wanted"),
        "n_prefix": len(prefix),
        "messages": keep["messages"],
    }


def dpo_row(keep: dict[str, Any], drop: dict[str, Any]) -> dict[str, Any]:
    return {
        "kind": "dpo",
        "sample_id": keep["sample_id"],
        "phase_wanted": keep.get("phase_wanted"),
        "prompt": prefix_of(keep),
        "chosen": keep.get("continuation") or [],
        "rejected": drop.get("continuation") or [],
        "chosen_rollout": keep.get("rollout"),
        "rejected_rollout": drop.get("rollout"),
        "rejected_reasons": drop.get("reasons") or [],
    }


def pack(rollouts: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_id: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rollouts:
        by_id[str(r["sample_id"])].append(r)

    sft: list[dict[str, Any]] = []
    dpo: list[dict[str, Any]] = []
    for sample_id, group in by_id.items():
        keeps = [r for r in group if r.get("keep")]
        drops = [r for r in group if not r.get("keep")]
        for k in keeps:
            sft.append(sft_row(k))
        for k, d in zip(keeps, drops):
            dpo.append(dpo_row(k, d))
    return sft, dpo


def demo_judged() -> list[dict[str, Any]]:
    prefix = demo_prefix("cold")
    return [
        judge(roll_one(prefix, rollout=1, n=2, king=fake_king_verify, observe=stub_observe, horizon=4)),
        judge(
            roll_one(
                prefix, rollout=2, n=2, king=fake_king_skip_verify, observe=stub_observe, horizon=4
            )
        ),
    ]


def run_self_test() -> int:
    sft, dpo = pack(demo_judged())
    print(f"sft {len(sft)}  dpo {len(dpo)}")
    if not sft or not dpo:
        print("FAIL: expected 1 sft and 1 dpo")
        return 1
    chosen_txt = json.dumps(dpo[0]["chosen"])
    rejected_txt = json.dumps(dpo[0]["rejected"])
    print("chosen has python -c:", "python -c" in chosen_txt)
    print("rejected has grep:", "grep" in rejected_txt)
    print("sft n_prefix:", sft[0]["n_prefix"], "total msgs:", len(sft[0]["messages"]))
    ok = (
        len(sft) == 1
        and len(dpo) == 1
        and "python -c" in chosen_txt
        and "grep" in rejected_txt
        and sft[0]["n_prefix"] < len(sft[0]["messages"])
        and dpo[0]["sample_id"] == sft[0]["sample_id"]
    )
    print("\nPASS" if ok else "\nFAIL")
    return 0 if ok else 1


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--keep", type=Path, help="out/keep.jsonl")
    parser.add_argument("--drop", type=Path, help="out/drop.jsonl")
    parser.add_argument("--rollouts", type=Path, help="already-judged jsonl (keep+drop mixed)")
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent / "out")
    args = parser.parse_args()
    if args.self_test:
        return run_self_test()

    rows: list[dict[str, Any]] = []
    if args.rollouts:
        rows = load_jsonl(args.rollouts)
    else:
        if not args.keep or not args.drop:
            raise SystemExit("need --keep and --drop, or --rollouts, or --self-test")
        rows = load_jsonl(args.keep) + load_jsonl(args.drop)

    sft, dpo = pack(rows)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    sft_p = args.out_dir / "sft.jsonl"
    dpo_p = args.out_dir / "dpo.jsonl"
    with sft_p.open("w") as fh:
        for r in sft:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    with dpo_p.open("w") as fh:
        for r in dpo:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    unpaired_keep = sum(1 for r in rows if r.get("keep")) - len(dpo)
    print(f"sft {len(sft)} -> {sft_p}")
    print(f"dpo {len(dpo)} -> {dpo_p}")
    if unpaired_keep:
        print(f"keeps without a drop sibling: {unpaired_keep} (SFT only)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
