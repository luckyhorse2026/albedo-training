# 05 — SFT and DPO rows

This step only **writes training files**. It does not start a trainer.

Script: `pack_pairs.py`

---

## Two files, two jobs

| file | what it is | why |
|---|---|---|
| `sft.jsonl` | every **keep** | “look like the good day” |
| `dpo.jsonl` | same ticket: **keep vs drop** | “prefer check over grep” |

SFT without DPO can still help. DPO without a keep/drop **pair** on the same `sample_id` is skipped (that keep is SFT-only).

---

## What is in a row

**SFT**

- `messages` = prefix + continuation (the whole chat)
- `n_prefix` = how many messages were the frozen ticket  
  Later the trainer must **not** train on those. Loss only after the cut.

**DPO**

- `prompt` = the frozen ticket (same for both)
- `chosen` = keep continuation (edit → check → submit)
- `rejected` = drop continuation (edit → grep …)
- `rejected_reasons` = why step 04 dropped it

Same `sample_id` on both sides. That is the point: one ticket, two days.

---

## Run

```bash
cd ~/sn97-albedo/albedo-training
uv run python pack_pairs.py --self-test
```

Expect: `sft 1  dpo 1`, chosen has `python -c`, rejected has `grep`.

From step 04 files:

```bash
uv run python pack_pairs.py --keep out/keep.jsonl --drop out/drop.jsonl --out-dir out
```

---

## What you check

1. SFT is the keep. DPO is keep **vs** drop on the **same** ticket.
2. `n_prefix` exists so we do not train on the paused part.
3. We still have not run LoRA / TRL.

Next (06): a real train command + a tiny local gate. Not yet. We do not have v124 on GPU in this step.
