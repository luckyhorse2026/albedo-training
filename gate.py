#!/usr/bin/env python3
"""Local gate after a train. Script only — does not roll a model unless --run.

Meters we must see after training (held-out prefixes × 2 rollouts):

  cold_verify_rate     up
  pre_edit_no_wide     up
  pre_edit_edit_rate   up
  cold_edit_rate       flat (if this drops, revert)
  loop_rate            ~0
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from cut_prefixes import _EDIT_RE
from filter_rollouts import _WIDE_SEARCH_RE, _is_verify, judge


METERS = (
    "cold_verify_rate",
    "pre_edit_no_wide",
    "pre_edit_edit_rate",
    "cold_edit_rate",
    "loop_rate",
)


def score_rollouts(rollouts: list[dict]) -> dict:
    """Proxy meters from already-judged rollouts. No model call."""
    n = {k: 0 for k in ("cold", "pre_edit", "all")}
    hit = {k: 0 for k in METERS}

    for raw in rollouts:
        rec = raw if "keep" in raw else judge(raw)
        phase = rec.get("phase_wanted") or "cold"
        n["all"] += 1
        n[phase] = n.get(phase, 0) + 1
        cmds = rec.get("commands") or []
        reasons = rec.get("reasons") or []
        if any(r.startswith("loop:") for r in reasons):
            hit["loop_rate"] += 1
        if phase == "cold":
            if any(_EDIT_RE.search(c or "") for c in cmds):
                hit["cold_edit_rate"] += 1
            if "no_verify_after_edit" not in reasons and any(_is_verify(c) for c in cmds):
                hit["cold_verify_rate"] += 1
        if phase == "pre_edit":
            first = cmds[0] if cmds else ""
            if first and not _WIDE_SEARCH_RE.search(first):
                hit["pre_edit_no_wide"] += 1
            if any(_EDIT_RE.search(c or "") for c in cmds):
                hit["pre_edit_edit_rate"] += 1

    def rate(num_key: str, den_phase: str) -> float | None:
        den = n.get(den_phase, 0)
        return (hit[num_key] / den) if den else None

    return {
        "n": n,
        "cold_verify_rate": rate("cold_verify_rate", "cold"),
        "cold_edit_rate": rate("cold_edit_rate", "cold"),
        "pre_edit_no_wide": rate("pre_edit_no_wide", "pre_edit"),
        "pre_edit_edit_rate": rate("pre_edit_edit_rate", "pre_edit"),
        "loop_rate": rate("loop_rate", "all"),
        "pass_if": {
            "cold_verify_rate": "up vs baseline",
            "cold_edit_rate": "flat (must not drop)",
            "pre_edit_no_wide": "up",
            "pre_edit_edit_rate": "up",
            "loop_rate": "near 0",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollouts", type=Path, help="judged or raw rollouts jsonl")
    parser.add_argument("--out", type=Path, default=Path("out/gate.json"))
    parser.add_argument("--run", action="store_true", help="roll a live model — not used here")
    args = parser.parse_args()

    if args.run:
        raise SystemExit("refused: this folder only prepares scripts. gate on a GPU box later.")

    if args.rollouts and args.rollouts.exists():
        rows = [json.loads(l) for l in args.rollouts.read_text().splitlines() if l.strip()]
        report = score_rollouts(rows)
    else:
        report = {
            "n": {},
            "note": "no --rollouts file; recipe only",
            "meters": list(METERS),
            "pass_if": {
                "cold_verify_rate": "up vs baseline",
                "cold_edit_rate": "flat (must not drop)",
                "pre_edit_no_wide": "up",
                "pre_edit_edit_rate": "up",
                "loop_rate": "near 0",
            },
        }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {args.out}")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
