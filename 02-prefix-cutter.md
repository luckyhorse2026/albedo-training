# 02 — Prefix cutter

This step only **pauses** tickets. No model. No training.

Script: `cut_prefixes.py`

---

## What it does

Eval does not send a whole bug-fix log. It **cuts** the log and both models continue from the cut.

This script makes those cuts the same way.

```
full log (assistant 1, 2, 3, 4, 5=first edit, 6, …)
                │
    cold        cut before assistant 1 or 2
    pre_edit    cut before assistant (first_edit - 2)
    at_edit     cut before assistant (first_edit + 1)
                │  so the first edit IS inside an at_edit prefix
                ▼
         prefix (paused ticket)  →  prefixes.jsonl
```

`turn_idx` is the assistant index in the **sample id**, same as eval: `shard:row:turn_idx`.

The prefix is **everything before** that assistant turn.

---

## Fake ticket (self-test)

The script ships a tiny made-up log. First source edit is assistant **#5** (`sed -i`).

| phase | turn_idx | assistants left in the prefix | last thing they did |
|---|---|---|---|
| cold (pick 1) | 1 | 1 | `cat` the file |
| cold (pick 2) | 2 | 2 | ran the failing `python -c` |
| pre_edit | 3 | 3 | still reading, no edit yet |
| at_edit | 5 | 5 | **includes** the `sed -i` |

If self-test prints PASS, the cut math matches the map.

```bash
cd /root/sn97-albedo/albedo-training
uv sync --extra dev          # once
uv run python cut_prefixes.py --self-test
```

---

## Real data (not downloaded yet)

Parquet must live at:

```
albedo-training/data/<source>/data/train-*.parquet
```

Sources: `mini-coder`, `open-swe-traces`, `swe-hero`, `mini-coder-rs`.

When you have data:

```bash
uv run python cut_prefixes.py --pack overfit --count 200 --out out/prefixes.jsonl
```

`--pack overfit` uses the v125 mix (55/25/20 cold/pre_edit/at_edit; mostly mini-coder + open-swe). See `07-gpu-dataset.md`.
`--pack eval` copies the eval mix if you want to compare.

Each jsonl line is one paused ticket: `messages` plus `phase_wanted`, `source`, `family`, `horizon` (12 or 16).

---

## What you check

After self-test, you should be able to say:

1. A prefix is the log **up to a pause**, not the whole fix.
2. `at_edit` still **contains** the first `sed -i`. `pre_edit` does not.
3. We have **not** trained anything. We only cut.

Next step (03) will be: take these prefixes and run **v125** on them. Not yet.
