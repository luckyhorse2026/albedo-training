# 03 — King rollouts

This step only **continues** a paused ticket. We do **not** keep or drop yet. That is step 04.

Script: `roll_king.py`

---

## What a rollout is

```
[prefix from step 02]
    assistant: THOUGHT + one bash command     ← the model
    user:      observation                    ← the environment
    assistant: …
    user:      …
    … up to horizon (12 or 16), or until submit
```

Eval does this **twice** per ticket and averages the two scores.
We will do it **four** times per ticket so step 04 can pick a good one and a bad sibling.

Same ticket, four tries, different temperatures later. Today the fake backend just plays two scripts:

| rollout | what it does | later (step 04) |
|---|---|---|
| odd (1, 3) | edit, then `python -c` check, then submit | keep |
| even (2, 4) | edit, then `grep` forever | drop |

That pair is the whole training idea: v124’s good day vs v124’s leak.

---

## What this step does **not** do

- It does not score tags.
- It does not talk to GLM.
- It does not run commands in a real repo. Observations are a **stub** (`(stub) ran: …` in RETURNCODE wrappers). Real grounding comes later if we need it.
- It does not start v124. `--backend fake` is for learning the file shape. `--backend openai` is for when you serve the king with vLLM.

---

## Run

```bash
cd ~/sn97-albedo/albedo-training
uv run python roll_king.py --self-test
```

You should see two rollouts on the fake **cold** prefix: one submits after a check, one keeps grepping. Then `PASS`.

When you have real prefixes:

```bash
uv run python roll_king.py --prefixes out/prefixes.jsonl --n 4 --backend fake --out out/rollouts.jsonl
```

When v124 is on a vLLM server:

```bash
uv run python roll_king.py --prefixes out/prefixes.jsonl --backend openai \
  --base-url http://127.0.0.1:8000/v1 --model v124 --out out/rollouts.jsonl
```

Each jsonl line is one rollout: `commands`, `continuation`, `stopped` (`horizon` / `submit` / `bad_turn:…`).

---

## What you check

1. A rollout is the prefix **plus new turns**, not a new ticket.
2. We store **all** tries. We do not filter here.
3. Fake backend is not the king. It only teaches the shape. Real weights need `--backend openai`.

Next (04): keep/drop using our rules (verify after edit, no restart-grep). Not yet.
