# albedo-training

Scripts for beating the sitting king on SN97.

**Sitting king = v125** ([CXXV](https://huggingface.co/dendriteholdings/albedo-qwen3.6-35b-king-CXXV)).
Train from that, not v124.

On a 4×H200 box, start here: **[`07-gpu-dataset.md`](07-gpu-dataset.md)**.
Model steps should keep all 4 GPUs busy (4 vLLM servers for roll/gate, `torchrun` for train).

This folder prepares the pipeline. `data/` and `out/` are gitignored. `--run` on `train.py` / `gate.py` is refused. Real train is `run_train.py`.

The backend lives in `../albedo`.

## First run (2026-09-12)

Cut 327 prefixes (287 train / 40 held-out) → 1722 v125 rolls → 383 SFT / 259 DPO. SFT+DPO LoRA on v125, merged to `out/train/challenger`. Held-out gate (80 rolls each):

| meter | v125 | challenger |
|---|---|---|
| cold_verify_rate | 0.650 | 0.675 |
| cold_edit_rate | 0.388 | 0.400 |
| loop_rate | 0 | 0 |

DPO at 8k OOM’d; use `--max-seq-len 4096` and the shared ref-logps cache in `run_train.py`. Serve vLLM again after train before you roll — otherwise `gate.json` is all nulls. HF pack is `out/train/challenger-hf` (king metadata + trained shards); this box has no `HF_TOKEN` yet.

## How we work

```
01  the map            01-the-map.md
02  prefix cutter      cut_prefixes.py
03  king rollouts      roll_king.py
04  keep/drop          filter_rollouts.py
05  sft / dpo packs    pack_pairs.py
06  train + gate       train.py  gate.py   (recipe + meters only)
07  GPU runbook        07-gpu-dataset.md   ← execute this
```

`--self-test` on a script checks the fake ticket. It is not a training run.

## Pipeline (GPU box)

```
cut → roll v125 ×6 → keep/drop → sft.jsonl + dpo.jsonl
    → run_train.py sft → merge → run_train.py dpo → merge
    → serve v125 + serve challenger → gate.py
```

## Layout

```
albedo-training/
  01-the-map.md … 07-gpu-dataset.md
  cut_prefixes.py  roll_king.py  filter_rollouts.py
  pack_pairs.py    train.py      gate.py
  run_train.py     # real SFT / DPO / merge
  notes/
```
