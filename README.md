# albedo-training

Scripts for beating the sitting king on SN97.

This folder **prepares** the pipeline. It does not download corpora, serve v124, or train.

The backend lives in `../albedo`.

## How we work

One step at a time. You read the step so you can give the next instruction.

```
01  the map            done   01-the-map.md
02  prefix cutter      done   cut_prefixes.py
03  king rollouts      done   roll_king.py
04  keep/drop          done   filter_rollouts.py
05  sft / dpo packs    done   pack_pairs.py
06  train + gate       done   train.py  gate.py   (recipe + meters only)
```

`--self-test` on a script checks the fake ticket. It is not a training run.
`--run` on `train.py` / `gate.py` is refused.

## Pipeline (execute later, on a GPU box)

```
cut → roll v124 ×4 → keep/drop → sft.jsonl + dpo.jsonl → SFT → DPO → gate meters
```

## Layout

```
albedo-training/
  01-the-map.md … 06-train-gate.md
  cut_prefixes.py  roll_king.py  filter_rollouts.py
  pack_pairs.py    train.py      gate.py
  notes/
```
