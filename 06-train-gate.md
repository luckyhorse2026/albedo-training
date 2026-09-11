# 06 — Train + gate (scripts only)

This folder **prepares** the train and the local gate. It does not download data, serve v124, or run LoRA.

---

## Train (`train.py`)

Writes `out/train/recipe.json` and prints the commands a GPU box would run.

- Base = v124
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

`--run` is refused here too. You can point `--rollouts` at a jsonl later to compute meters from files.

---

## Pipeline we prepared (not executed)

```
cut_prefixes.py     pause tickets
roll_king.py        4 continuations (needs v124 + env on a GPU box)
filter_rollouts.py  keep / drop
pack_pairs.py       sft.jsonl + dpo.jsonl
train.py            recipe only
gate.py             meters only
```
