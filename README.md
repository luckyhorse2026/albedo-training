# albedo-training

Scripts for beating the sitting king on SN97.

**Sitting king = v125** ([CXXV](https://huggingface.co/dendriteholdings/albedo-qwen3.6-35b-king-CXXV)).
Train from that, not v124.

On a 4×H200 box, start here: **[`07-gpu-dataset.md`](07-gpu-dataset.md)**.
Model steps should keep all 4 GPUs busy (4 vLLM servers for roll/gate, `torchrun` for train).

This folder prepares the pipeline. `data/` and `out/` are gitignored. `--run` on `train.py` / `gate.py` is refused.

The backend lives in `../albedo`.

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
cut → roll v125 ×6 → keep/drop → sft.jsonl + dpo.jsonl → SFT → DPO → gate meters
```

## Layout

```
albedo-training/
  01-the-map.md … 07-gpu-dataset.md
  cut_prefixes.py  roll_king.py  filter_rollouts.py
  pack_pairs.py    train.py      gate.py
  notes/
```
