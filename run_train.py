#!/usr/bin/env python3
"""SFT then DPO LoRA on v125. Loss only on assistant turns after n_prefix.

This is the trainer 07-gpu-dataset.md points at. train.py only writes recipe.json.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from torch.utils.data import Dataset
from transformers import (
    AutoModelForImageTextToText,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)


# language_model only: full attn + linear-attn projs + shared expert. Not routed experts, not vision.
LORA_TARGET_REGEX = (
    r".*language_model.*(?:"
    r"q_proj|k_proj|v_proj|o_proj|"
    r"in_proj_qkv|in_proj_z|in_proj_a|in_proj_b|out_proj|"
    r"shared_expert\.(?:gate_proj|up_proj|down_proj)"
    r")$"
)


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def _has_user_query(messages: list[dict[str, str]]) -> bool:
    for msg in messages:
        if msg.get("role") != "user":
            continue
        content = str(msg.get("content") or "").strip()
        if content.startswith("<tool_response>") and content.endswith("</tool_response>"):
            continue
        return True
    return False


def _rendered_blocks_before(messages: list[dict[str, str]], n_prefix: int) -> int:
    n = 0
    for i, msg in enumerate(messages[:n_prefix]):
        role = msg.get("role")
        if role == "system" and i == 0:
            n += 1
        elif role in ("user", "assistant"):
            n += 1
    return n


def sft_example(row: dict[str, Any], tokenizer: Any, max_len: int) -> dict[str, list[int]] | None:
    """Tokenize the full ticket once. Qwen's template requires a user query, so
    we never render a system-only prefix. Loss = assistant tokens after n_prefix."""
    import re

    messages = row["messages"]
    n_prefix = int(row["n_prefix"])
    if not _has_user_query(messages):
        return None
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    starts = [m.start() for m in re.finditer(r"<\|im_start\|>", text)]
    cut_blocks = _rendered_blocks_before(messages, n_prefix)
    char_cut = starts[cut_blocks] if cut_blocks < len(starts) else len(text)
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    ids = list(encoded["input_ids"])
    offsets = encoded["offset_mapping"]
    prefix_len = next((i for i, (s, _e) in enumerate(offsets) if s >= char_cut), len(ids))
    asst_chars: set[int] = set()
    for match in re.finditer(r"<\|im_start\|>assistant\n", text):
        if match.start() < char_cut:
            continue
        end = text.find("<|im_start|>", match.end())
        if end < 0:
            end = len(text)
        asst_chars.update(range(match.start(), end))
    labels: list[int] = []
    for i, (tid, (s, e)) in enumerate(zip(ids, offsets)):
        if i < prefix_len or e <= s or not any(c in asst_chars for c in range(s, e)):
            labels.append(-100)
        else:
            labels.append(tid)
    if len(ids) < 2 or all(x == -100 for x in labels):
        return None
    ids = ids[:max_len]
    labels = labels[:max_len]
    return {
        "input_ids": ids,
        "attention_mask": [1] * len(ids),
        "labels": labels,
    }


class JsonlSftDataset(Dataset):
    def __init__(self, rows: list[dict[str, Any]], tokenizer: Any, max_len: int):
        self.examples: list[dict[str, list[int]]] = []
        skipped = 0
        for row in rows:
            ex = sft_example(row, tokenizer, max_len)
            if ex is None:
                skipped += 1
                continue
            self.examples.append(ex)
        if not self.examples:
            raise SystemExit("no usable SFT rows after masking/truncation")
        print(f"sft examples {len(self.examples)} (skipped {skipped})")

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, list[int]]:
        return self.examples[idx]


class PadCollator:
    def __init__(self, pad_id: int):
        self.pad_id = pad_id

    def __call__(self, features: list[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
        longest = max(len(f["input_ids"]) for f in features)
        batch = {"input_ids": [], "attention_mask": [], "labels": []}
        for f in features:
            pad = longest - len(f["input_ids"])
            batch["input_ids"].append(f["input_ids"] + [self.pad_id] * pad)
            batch["attention_mask"].append(f["attention_mask"] + [0] * pad)
            batch["labels"].append(f["labels"] + [-100] * pad)
        return {k: torch.tensor(v, dtype=torch.long) for k, v in batch.items()}


def load_base(model_path: str):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForImageTextToText.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="sdpa",
    )
    model.config.use_cache = False
    model.enable_input_require_grads()
    return tokenizer, model


def attach_lora(model: Any) -> Any:
    cfg = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=LORA_TARGET_REGEX,
    )
    model = get_peft_model(model, cfg)
    model.print_trainable_parameters()
    return model


def sft_args(*, out_dir: str, lr: float, epochs: float, batch: int, grad_accum: int) -> TrainingArguments:
    return TrainingArguments(
        output_dir=out_dir,
        per_device_train_batch_size=batch,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        num_train_epochs=epochs,
        logging_steps=5,
        save_strategy="epoch",
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        ddp_find_unused_parameters=False,
        remove_unused_columns=False,
        report_to=[],
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
    )


def run_sft(args: argparse.Namespace) -> None:
    tokenizer, model = load_base(args.base)
    model = attach_lora(model)
    data = JsonlSftDataset(load_jsonl(args.data), tokenizer, args.max_seq_len)
    trainer = Trainer(
        model=model,
        args=sft_args(
            out_dir=args.out_dir,
            lr=args.lr,
            epochs=args.epochs,
            batch=args.per_device_train_batch_size,
            grad_accum=args.grad_accum,
        ),
        train_dataset=data,
        data_collator=PadCollator(tokenizer.pad_token_id),
    )
    trainer.train()
    trainer.save_model(args.out_dir)
    tokenizer.save_pretrained(args.out_dir)


class SharedRefLogpsDPOTrainer:
    """TRL writes precomputed ref log-probs next to the Dataset cache.

    `Dataset.from_list` uses a unique `/tmp/hf_datasets-*` dir per rank, so
    rank 0's arrow file is invisible to the others. Pin the cache under
    output_dir so every rank reads the same path.
    """

    def _precompute_ref_logps(self, dataset, name: str, batch_size: int):
        import os

        cache_dir = os.path.join(os.path.abspath(self.args.output_dir), "_ref_logps")
        os.makedirs(cache_dir, exist_ok=True)
        orig = dataset._get_cache_file_path
        dataset._get_cache_file_path = lambda fingerprint: os.path.join(
            cache_dir, f"{name}-{fingerprint}.arrow"
        )
        try:
            return super()._precompute_ref_logps(dataset, name, batch_size)
        finally:
            dataset._get_cache_file_path = orig


def run_dpo(args: argparse.Namespace) -> None:
    import os

    from trl import DPOConfig, DPOTrainer

    class _DPOTrainer(SharedRefLogpsDPOTrainer, DPOTrainer):
        pass

    # DPO does chosen+rejected (+ ref). 8k OOM'd a 140GB H200 after SFT-sized weights.
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    tokenizer, model = load_base(args.base)
    model = attach_lora(model)
    tokenizer.padding_side = "left"
    rows = []
    for row in load_jsonl(args.data):
        prompt = row.get("prompt") or []
        chosen = row.get("chosen") or []
        rejected = row.get("rejected") or []
        if not prompt or not chosen or not rejected:
            continue
        rows.append(
            {
                "prompt": prompt,
                "chosen": chosen,
                "rejected": rejected,
            }
        )
    if not rows:
        raise SystemExit("no usable DPO rows")
    from datasets import Dataset

    print(f"dpo examples {len(rows)} max_seq_len={args.max_seq_len}")
    rows = Dataset.from_list(rows)
    cfg = DPOConfig(
        output_dir=args.out_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        beta=args.beta,
        logging_steps=5,
        save_strategy="epoch",
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        ddp_find_unused_parameters=False,
        remove_unused_columns=False,
        report_to=[],
        max_length=args.max_seq_len,
        # one frozen LoRA copy as ref, computed once — no second 35B and no 4-way live fwd
        precompute_ref_log_probs=True,
        precompute_ref_batch_size=1,
    )
    trainer = _DPOTrainer(
        model=model,
        ref_model=None,
        args=cfg,
        train_dataset=rows,
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model(args.out_dir)
    tokenizer.save_pretrained(args.out_dir)


def run_merge(args: argparse.Namespace) -> None:
    tokenizer, model = load_base(args.base)
    model = PeftModel.from_pretrained(model, args.adapter)
    merged = model.merge_and_unload()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(out)
    tokenizer.save_pretrained(out)
    # subnet rejects a leftover adapter_config next to merged weights
    leftover = out / "adapter_config.json"
    if leftover.exists():
        leftover.unlink()
    print(f"merged -> {out}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    sft = sub.add_parser("sft")
    sft.add_argument("--base", default="/data/kings/v125")
    sft.add_argument("--data", default="out/sft.jsonl")
    sft.add_argument("--out-dir", default="out/train/sft-adapter")
    sft.add_argument("--lr", type=float, default=5e-6)
    sft.add_argument("--epochs", type=float, default=1.0)
    sft.add_argument("--per-device-train-batch-size", type=int, default=1)
    sft.add_argument("--grad-accum", type=int, default=1)
    sft.add_argument("--max-seq-len", type=int, default=8192)

    dpo = sub.add_parser("dpo")
    dpo.add_argument("--base", default="out/train/sft-merged")
    dpo.add_argument("--data", default="out/dpo.jsonl")
    dpo.add_argument("--out-dir", default="out/train/dpo-adapter")
    dpo.add_argument("--lr", type=float, default=5e-6)
    dpo.add_argument("--epochs", type=float, default=1.0)
    dpo.add_argument("--beta", type=float, default=0.2)
    dpo.add_argument("--per-device-train-batch-size", type=int, default=1)
    dpo.add_argument("--grad-accum", type=int, default=1)
    dpo.add_argument("--max-seq-len", type=int, default=4096)

    merge = sub.add_parser("merge")
    merge.add_argument("--base", required=True)
    merge.add_argument("--adapter", required=True)
    merge.add_argument("--out-dir", required=True)

    args = parser.parse_args()
    if args.cmd == "sft":
        run_sft(args)
    elif args.cmd == "dpo":
        run_dpo(args)
    else:
        run_merge(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
