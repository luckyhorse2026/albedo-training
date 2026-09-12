#!/usr/bin/env python3
"""Write the train recipe. Does not train unless you pass --run (not used here).

SFT then DPO on v125. Small LR. Loss only after n_prefix.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def recipe(*, king_path: str, sft_path: str, dpo_path: str, out_dir: str) -> dict:
    return {
        "base": king_path,
        "order": ["sft", "dpo"],
        "sft": {
            "data": sft_path,
            "mask": "messages[0:n_prefix] — no loss",
            "loss": "assistant turns after n_prefix",
            "lora": {"targets": "attn + shared expert", "not": "all 256 experts"},
            "lr": 5e-6,
            "epochs": 1,
            "notes": "one-third epoch if cold edit rate starts falling",
        },
        "dpo": {
            "data": dpo_path,
            "prompt": "prefix",
            "chosen": "keep continuation",
            "rejected": "drop continuation",
            "beta": 0.2,
            "lr": 5e-6,
            "epochs": 1,
        },
        "do_not": [
            "train embed_tokens / lm_head / layernorms",
            "edit genesis tokenizer or chat_template",
            "SFT the whole mini-coder dump",
        ],
        "out_dir": out_dir,
        "adapter_dir": str(Path(out_dir) / "adapter"),
    }


def commands(r: dict) -> list[str]:
    """Commands a later GPU box would run. Not executed by this script."""
    return [
        f"# 1. SFT  (TRL / your trainer)  base={r['base']}  data={r['sft']['data']}",
        f"# 2. DPO  base=sft-merged       data={r['dpo']['data']}  beta={r['dpo']['beta']}",
        f"# 3. merge LoRA -> {r['adapter_dir']}",
        "# 4. uv run python gate.py --prefixes out/heldout.jsonl --base-url $KING_URL",
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--king", default="", help="path to v125 weights (required for --run)")
    parser.add_argument("--sft", default="out/sft.jsonl")
    parser.add_argument("--dpo", default="out/dpo.jsonl")
    parser.add_argument("--out-dir", default="out/train")
    parser.add_argument("--run", action="store_true", help="actually train — do not use in this repo")
    args = parser.parse_args()

    r = recipe(king_path=args.king or "<v125-repo>", sft_path=args.sft, dpo_path=args.dpo, out_dir=args.out_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "recipe.json"
    path.write_text(json.dumps(r, indent=2) + "\n")
    print(f"wrote {path}")
    for line in commands(r):
        print(line)
    if args.run:
        raise SystemExit("refused: this folder only prepares scripts. train on a GPU box later.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
