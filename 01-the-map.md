# 01 — The map

Read this so you can give instructions. Every later script should be checkable against this file.

---

## 1. You are not chasing a number

On SWE-bench you can say “I got 45%, I want 50%.” That number exists even if nobody else runs.

Albedo is not that. It is a **fight against this week’s king, on this week’s 100 tasks.**

- They pick ~100 half-finished coding jobs (different every submission).
- They run **you** and **the king** on those **same** jobs.
- They ask the **same** yes/no questions about both transcripts.
- Your score and the king’s score are just “what fraction of questions was yes.”
- You win only if you are at least **2.5 points higher** than the king on *that* set.
- Then you must win a **second** fight, new tasks, same rule.

So “I scored 0.71 last time” is useless. The only useful sentence is:

> On the same tasks, did I beat this king by enough?

That is all “no score to beat” meant. Sorry — that phrase was bad.

Our starting weights: **king v125** ([CXXV](https://huggingface.co/dendriteholdings/albedo-qwen3.6-35b-king-CXXV)). Not v124. Not genesis.

---

## 2. One job = a cut conversation

They take a real agent log and **pause it**. Both models must continue from that pause.

The pause point is **phase**. You already have this:

| name | pause point | how often |
|---|---|---|
| `cold` | almost the start (turn 1 or 2) | 65 of 100 |
| `pre_edit` | 2 turns **before** anyone edited source | 15 of 100 |
| `at_edit` | the turn of the **first** source edit | 20 of 100 |

Picture a ticket:

- **cold:** you just got the ticket. Nobody has touched the code.
- **pre_edit:** someone already looked around. The fix is about to happen. You must finish, not start over.
- **at_edit:** someone already started changing a file. You must finish and check, not re-read the whole repo.

Code: `../albedo/src/albedo_eval_service/shared/sampling.py`.

Each model continues **12 or 16** turns, and does that **twice** (two rollouts). The sample score is the **average of the two**. One brilliant run + one loop = a bad average.

The model must write, every turn:

```
THOUGHT: ...

```bash
exactly one command
```
```

---

## 3. The four tags (the yes/no questions)

After both sides finish, a judge asks many yes/no questions. Every question is one of four kinds. All count the same.

Think of fixing a bug in a kitchen:

| tag | kitchen version | in the transcript |
|---|---|---|
| **claims** | Did you taste the soup and see it is salty? | You **ran** the thing the ticket says is wrong (the failing test / the bad CLI) and saw the error. |
| **explore** | Did you find *which* ingredient made it salty? | You **read code** (`cat` / `grep` / `sed -n`) and the answer is in an observation, not only in your THOUGHT. |
| **action** | Did you actually change the recipe? | A command **edits a real source file** (`sed -i`, `cat >`, search-replace). Talking about an edit is not an edit. |
| **verification** | Did you taste the soup **after** changing it? | **After** the edit, you **ran a check again** (re-run the test / `python -c` the changed function). |

Important: the judge looks at **commands and their output**, not at plans.

- “I would run pytest” → **no**
- `pytest tests/test_foo.py` after the edit → **yes** for verification

Most questions are **explore** (lots of “did you read X”). We still do **not** train explore as the main thing. The king already reads. He loses on other tags.

### The 65% slice = cold

Most fights start at `cold` (ticket just arrived).

On that slice, the king’s **main leftover leak is still verification** (v125 ~0.45 yes-rate).

In one line: **the king changes the code and does not re-run the check.**

Challengers who almost beat him did the extra taste. That is the first habit we overfit.

---

## 4. Why the answers come from v125, not from GLM

Two different things go into a training row:

| piece | where it comes from | why |
|---|---|---|
| **prefix** (the paused ticket) | public datasets (mini-coder, SWE-hero, …) | same kind of pause the eval uses |
| **continuation** (what to do next) | **v125, run several times** | same voice, same command style as the king we must beat |
| **keep or drop** | our rules (did he verify? did he restart grep?) | this is the “teacher”, not GLM |

**GLM** is the strong model the *subnet* uses to write questions. If we SFT on GLM’s transcripts, we teach “move like GLM.” The questions are written so that **any correct route** can score. Copying GLM’s files and greps does not help, and can hurt.

**v125** already knows how to edit. Sometimes it also verifies. Sometimes it doesn’t.

So we do this (see `07-gpu-dataset.md`):

1. Give v125 the same paused ticket **6 times**.
2. Keep the run where he **did** taste the soup after changing it.
3. Train him to look like *that* run.
4. Contrast it with his other run on the **same** ticket where he skipped the taste.

We are not inventing a new player. We are making his **good day** into his **normal day**.

That is all “SFT from the king itself” means.

---

## 5. Two things we must not break

While we teach “always taste after you change it,” we can accidentally teach “stop changing anything and only run tests.” That would lose.

Do not break what the king already does:

v125 is **already good enough** at:

1. **`cold` + `action`** — from a fresh ticket he **edits** more than the people who almost beat him. Keep that.
2. **`at_edit` + `verification`** — when the pause is already on the first edit, he already checks about as well as they do. Don’t overwrite that.

If after training he edits less on cold starts, we failed, even if he tests more.

---

## 6. Our bet (still later)

Overfit three habits only:

1. Cold: edit, then **immediately run a check**.
2. Pre-edit: do **not** start with `find` / `grep -r` over the whole repo; use the prefix, then edit.
3. SWE-hero: same, but leave the OpenHands output format as-is.

We will not SFT the whole internet of traces. We will not copy a king and add noise.

---

## 7. Try again (only the ones that failed)

Question 2 you already got. Skip it.

Answer these in your own words, short:

**A.** After an eval, why can’t you say “I need to reach 0.72 next time”?

**B.** Name the four tags in kitchen words, then say which one the king fails on a *fresh ticket*.

**C.** In a training row, what comes from the public dataset, what comes from v125, and what decides keep vs drop?

**D.** What two skills of the king must stay as good as they are now?

Write answers here in chat, or in `notes/`. We do not start step 02 until A–D are solid.
