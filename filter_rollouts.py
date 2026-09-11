#!/usr/bin/env python3
"""Keep or drop a rollout. Step 04 — rules only, no training.

A keep is a continuation that would earn the tags the king leaks.
A drop is the sibling that skipped the habit. Step 05 will pair them.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from cut_prefixes import _EDIT_RE, demo_prefix
from format_turn import first_bash, is_submit
from roll_king import fake_king_skip_verify, fake_king_verify, roll_one, stub_observe

# Live eval short-circuit (we taught 4 / 0.5; code today is 5 / 0.61 — stay on the stricter teach).
MAX_RUN = 4
DUP_RATIO = 0.5

_VERIFY_RE = re.compile(
    r"\b(pytest|python\s+-c|python3\s+-c|cargo\s+test|go\s+test|npm\s+test|"
    r"unittest|nosetests|rake\s+test)\b",
    re.I,
)
_CLAIMS_RE = re.compile(
    r"\b(pytest|python\s+-c|python3\s+-c|cargo\s+test|go\s+test|npm\s+test)\b",
    re.I,
)
# Repo-wide search: no path from the ticket, just "search everything".
_WIDE_SEARCH_RE = re.compile(
    r"^\s*(find\s+\.|find\s+/\s|grep\s+-[a-zA-Z]*r[a-zA-Z]*\s|rg\s+.*\s+\.|grep\s+-rn\s)",
    re.I,
)


def _asst_cmds(messages: list[dict[str, str]]) -> list[str]:
    out = []
    for m in messages:
        if m.get("role") != "assistant":
            continue
        cmd = first_bash(m.get("content") or "")
        if cmd:
            out.append(cmd)
    return out


def _is_edit(command: str) -> bool:
    return bool(_EDIT_RE.search(command or ""))


def _is_verify(command: str) -> bool:
    return bool(_VERIFY_RE.search(command or "")) and not is_submit(command)


def _is_claims(command: str) -> bool:
    return bool(_CLAIMS_RE.search(command or ""))


def _loop_reason(commands: list[str]) -> str:
    work = [c for c in commands if not is_submit(c)]
    if not work:
        return ""
    run = max_run = 1
    for prev, cur in zip(work, work[1:]):
        run = run + 1 if cur == prev else 1
        max_run = max(max_run, run)
    dup = 1 - len(set(work)) / len(work)
    if max_run >= MAX_RUN:
        return f"loop:{max_run}x same command"
    if dup >= DUP_RATIO:
        return f"loop:dup_ratio={dup:.2f}"
    return ""


def judge(rollout: dict[str, Any]) -> dict[str, Any]:
    """Return the same rollout plus keep/drop and reasons."""
    phase = rollout.get("phase_wanted") or "cold"
    continuation = rollout.get("continuation") or []
    prefix = []
    # messages = prefix + continuation; recover prefix by length
    full = rollout.get("messages") or []
    if full and continuation:
        prefix = full[: len(full) - len(continuation)]
    cont_cmds = _asst_cmds(continuation)
    prefix_cmds = _asst_cmds(prefix)
    all_cmds = prefix_cmds + cont_cmds
    reasons: list[str] = []

    stopped = str(rollout.get("stopped") or "")
    if stopped.startswith("bad_turn"):
        reasons.append(stopped)

    loop = _loop_reason(cont_cmds)
    if loop:
        reasons.append(loop)

    cont_edits = [i for i, c in enumerate(cont_cmds) if _is_edit(c)]
    prefix_has_edit = any(_is_edit(c) for c in prefix_cmds)

    if phase in {"cold", "pre_edit"} and not cont_edits:
        reasons.append("no_source_edit")

    if phase == "pre_edit" and cont_cmds and _WIDE_SEARCH_RE.search(cont_cmds[0]):
        reasons.append("restart_wide_search")

    if cont_edits:
        last_edit = cont_edits[-1]
        after = cont_cmds[last_edit + 1 :]
        if not any(_is_verify(c) for c in after):
            reasons.append("no_verify_after_edit")
        earlier = prefix_cmds + cont_cmds[: last_edit]
        if phase == "cold" and not any(_is_claims(c) for c in earlier):
            reasons.append("edit_before_repro")
    elif prefix_has_edit and phase == "at_edit":
        if not any(_is_verify(c) for c in cont_cmds):
            reasons.append("no_verify_after_edit")

    if any(is_submit(c) for c in cont_cmds) and "no_verify_after_edit" in reasons:
        reasons.append("submit_without_verify")

    keep = not reasons
    return {
        **rollout,
        "keep": keep,
        "drop": not keep,
        "reasons": reasons,
    }


def run_self_test() -> int:
    prefix = demo_prefix("cold")
    good = judge(
        roll_one(prefix, rollout=1, n=2, king=fake_king_verify, observe=stub_observe, horizon=4)
    )
    bad = judge(
        roll_one(
            prefix, rollout=2, n=2, king=fake_king_skip_verify, observe=stub_observe, horizon=4
        )
    )
    print(f"verify  keep={good['keep']} reasons={good['reasons']} cmds={good['commands']}")
    print(f"skip    keep={bad['keep']} reasons={bad['reasons']} cmds={bad['commands']}")
    ok = good["keep"] is True and bad["keep"] is False and "no_verify_after_edit" in bad["reasons"]
    print("\nPASS" if ok else "\nFAIL")
    return 0 if ok else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--rollouts", type=Path)
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent / "out")
    args = parser.parse_args()
    if args.self_test:
        return run_self_test()
    if not args.rollouts or not args.rollouts.exists():
        raise SystemExit("need --rollouts out/rollouts.jsonl (or --self-test)")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    keep_p = args.out_dir / "keep.jsonl"
    drop_p = args.out_dir / "drop.jsonl"
    n_keep = n_drop = 0
    reason_counts: dict[str, int] = {}
    with args.rollouts.open() as inp, keep_p.open("w") as kh, drop_p.open("w") as dh:
        for line in inp:
            if not line.strip():
                continue
            rec = judge(json.loads(line))
            dest = kh if rec["keep"] else dh
            dest.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if rec["keep"]:
                n_keep += 1
            else:
                n_drop += 1
                for r in rec["reasons"]:
                    reason_counts[r] = reason_counts.get(r, 0) + 1
    print(f"keep {n_keep} -> {keep_p}")
    print(f"drop {n_drop} -> {drop_p}")
    print("reasons", reason_counts)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
