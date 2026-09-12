# 06 — Train + gate (scripts only)

This folder **prepares** the train and the local gate. `train.py` does not run LoRA. The trainer is `run_train.py`. GPU steps and the 2026-09-12 box log: `07-gpu-dataset.md`.

---

## Train (`train.py`)

Writes `out/train/recipe.json` and prints the commands a GPU box would run.

- Base = v125
- SFT on `sft.jsonl` (loss only after `n_prefix`)
- DPO on `dpo.jsonl` (keep vs drop, same ticket)
- LoRA on attn + shared expert, LR `5e-6`, 1 epoch, DPO β `0.2`
- Do not train `embed` / `lm_head`; do not touch genesis tokenizer

`--run` is refused on purpose.

## Gate (`gate.py`)

Defines the meters. After a real train, someone would roll held-out prefixes × 2 and fill them.

| meter | must |
|---|---|
| cold verify-after-edit | up |
| pre_edit first cmd is not repo-wide search | up |
| pre_edit actually edits | up |
| cold edit rate | **flat** (if it drops, revert) |
| loop rate | ~0 |

`--run` is refused here too. Point `--rollouts` at a jsonl you already rolled.

First held-out pass (40 prefixes × 2): v125 `cold_verify_rate` 0.650 / `cold_edit_rate` 0.388; challenger 0.675 / 0.400; `loop_rate` 0. No `pre_edit` tickets in that 40. Serve vLLM after train — rolling against a dead port writes empty files and a null `gate.json`.

---

## Pipeline we prepared (execute on GPU via 07)

```
cut_prefixes.py     pause tickets
roll_king.py        6 continuations (needs v125 on a GPU box)
filter_rollouts.py  keep / drop
pack_pairs.py       sft.jsonl + dpo.jsonl
train.py            recipe only
gate.py             meters only
```
