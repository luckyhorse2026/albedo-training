# 04 — Keep / drop

This step only **labels** rollouts. No training.

Script: `filter_rollouts.py`

---

## What it does

Each rollout gets `keep: true` or a list of `reasons`.

```
rollouts.jsonl
    ├─ keep.jsonl   ← did the missing habit
    └─ drop.jsonl   ← skipped it (sibling for DPO later)
```

---

## The rules (plain words)

A keep must pass **all** that apply.

| reason if it fails | meaning |
|---|---|
| `loop:…` | same command 4× in a row, or ≥ half the commands are repeats |
| `bad_turn:…` | a turn had no bash block |
| `no_source_edit` | on cold / pre_edit, the continuation never edited source |
| `restart_wide_search` | on pre_edit, first new command is `find .` / `grep -r` |
| `no_verify_after_edit` | after the last `sed -i` (etc.) there is no check (`pytest`, `python -c`, …) |
| `edit_before_repro` | on cold, first edit happened before any failing-test / `python -c` |
| `submit_without_verify` | submitted after an edit with no check |

These are **our** rules, not GLM. They target the leftover leak: edit, then no taste; or restart-search / pytest-install loops.

Also dropped: `search_spam` (≥3 repo-wide searches on pre_edit/at_edit) and `infra_thrash` (`pip install pytest` / `which pytest` twice).

`at_edit` prefixes already contain the first edit. Then we only require a verify in the continuation.

---

## Run

```bash
cd ~/sn97-albedo/albedo-training
uv run python filter_rollouts.py --self-test
```

Uses the fake pair from step 03:

| script | expect |
|---|---|
| edit → `python -c` → submit | **keep** |
| edit → `grep` → `grep` → `grep` | **drop** (`no_verify_after_edit`) |

On real rollouts:

```bash
uv run python filter_rollouts.py --rollouts out/rollouts.jsonl --out-dir out
```

---

## What you check

1. Keep = the habit we want. Drop = the leak.
2. The labeler does not call a judge model.
3. We still have not trained.

Next (05): turn keep/drop siblings into SFT + DPO rows. Not yet.
