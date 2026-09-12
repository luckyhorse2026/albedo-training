# 07 — GPU runbook: dataset that beats v125 with v125

Open this file on the GPU box. Run the commands in order. Do not skip the “what good looks like” checks.

This is **not** a new teacher. You roll the sitting king (v125) several times on the same paused ticket, keep its good day, drop its leak, then LoRA on **v125**.

```
prefix (public parquet, cut like eval)
    × 6 continuations from v125
        keep  = edit → check (and no restart-search)
        drop  = the sibling that skipped that
            → sft.jsonl + dpo.jsonl
                → LoRA on v125 (attn + shared expert)
                    → gate on held-out prefixes
```

`train.py --run` and `gate.py --run` are still refused in this repo. After the jsonl files exist, train with your own TRL / axolotl command using `out/train/recipe.json`.

---

## 0. Box (4× H200)

Goal: **no idle GPU** on any step that loads the model. One step at a time. When a step finishes, stop it so the next step can take every card.

Cut / filter / pack are CPU — GPUs will be idle then. That is fine. Download is disk.

| step | keep all 4 busy |
|---|---|
| roll ×6, gate | one vLLM per GPU (4 servers) + one roller per server |
| SFT, DPO | `torchrun --nproc_per_node=4` |

CPython 3.12, this repo, `uv`.

```bash
git clone https://github.com/luckyhorse2026/albedo-training.git
cd albedo-training
uv sync --extra dev
uv run python cut_prefixes.py --self-test
uv run python roll_king.py --self-test
uv run python filter_rollouts.py --self-test
uv run python pack_pairs.py --self-test
```

All four must print `PASS`. If not, stop.

`data/` and `out/` are gitignored. They stay on the box.

---

## 1. Weights = v125, not v124

Public mirror:

- https://huggingface.co/dendriteholdings/albedo-qwen3.6-35b-king-CXXV

```bash
huggingface-cli download dendriteholdings/albedo-qwen3.6-35b-king-CXXV \
  --local-dir /data/kings/v125
```

Serve **four replicas**, one GPU each. Same weights, four ports. Name them `v125`.

```bash
# stop anything leftover
pkill -f 'vllm serve' || true

for i in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$i vllm serve /data/kings/v125 \
    --served-model-name v125 \
    --port $((8000 + i)) \
    --enable-prefix-caching \
    --gpu-memory-utilization 0.90 \
    --max-model-len 32768 \
    > /tmp/vllm-$i.log 2>&1 &
done

for i in 0 1 2 3; do
  until curl -sf http://127.0.0.1:$((8000 + i))/v1/models >/dev/null; do sleep 2; done
  echo "gpu $i up on $((8000 + i))"
done
```

`nvidia-smi` should show ~72GB+ on **every** card. If one card is 0, that replica died — read `/tmp/vllm-$i.log`.

Eval uses temp **1.0**, top_p **0.95**, thinking on. Rollouts must look like the king, so keep temp at 1.0. You need **both** a keep and a drop on many tickets; low temp collapses to one day.

---

## 2. Prefix corpora (cut only — never train the gold completions)

Eval sources and HF repos:

| folder under `data/` | Hugging Face repo |
|---|---|
| `mini-coder` | `ricdomolm/mini-coder-trajs-400k` |
| `open-swe-traces` | `nvidia/Open-SWE-Traces` |
| `swe-hero` | `nvidia/SWE-Hero-openhands-trajectories` |
| `mini-coder-rs` | `AlienKevin/SWE-smith-rs-minimax-m2.5-trajectories` |

Cutter expects:

```
data/<source>/data/train-*.parquet
```

You do **not** need every mini-coder shard. A few early shards plus the other three repos is enough for 340 prefixes.

```bash
# examples — include paths depend on how the HF repo is laid out
huggingface-cli download ricdomolm/mini-coder-trajs-400k \
  --include "data/train-0000{0,1,2,3,4,5}-of-00060.parquet" \
  --local-dir data/mini-coder

huggingface-cli download nvidia/Open-SWE-Traces \
  --include "data/train-*.parquet" \
  --local-dir data/open-swe-traces

huggingface-cli download nvidia/SWE-Hero-openhands-trajectories \
  --include "data/train-*.parquet" \
  --local-dir data/swe-hero

huggingface-cli download AlienKevin/SWE-smith-rs-minimax-m2.5-trajectories \
  --include "data/train-*.parquet" \
  --local-dir data/mini-coder-rs
```

If a repo uses a different prefix than `data/`, symlink so `data/<source>/data/train-*.parquet` exists.

Do **not** SFT the assistant turns that are already in those parquets. They are another agent. Using them is how you get a different voice and a possible similarity hit.

---

## 3. Cut prefixes

v125 leftover is still **verification** and **action** (yes-rate ~0.48 / 0.50 on the crown tape). Mix for this king:

| slice | share | why |
|---|---|---|
| cold | 55% | 65% of live eval; v125 still skips the check often |
| pre_edit | 25% | must not restart `find` / `grep -r`, then must edit |
| at_edit | 20% | prefix already edited; continuation must check |
| mini-coder | ~57% | most of the live draw |
| open-swe | ~37% | second most |
| swe-hero + rust | ~6% | tiny live n |

`--pack overfit` uses that mix. `--pack eval` is the live 65/15/20 if you want a comparison cut.

```bash
uv run python cut_prefixes.py \
  --pack overfit \
  --count 340 \
  --repo-cap 8 \
  --out out/all-prefixes.jsonl
```

Expect ~340 lines. If it prints `short: wanted 340`, download more shards or raise `--max-scan`.

Split 40 held-out tickets. **Never train on these.**

```bash
uv run python - <<'PY'
from pathlib import Path
lines = Path("out/all-prefixes.jsonl").read_text().splitlines()
assert len(lines) >= 80, len(lines)
Path("out").mkdir(exist_ok=True)
Path("out/heldout.jsonl").write_text("\n".join(lines[:40]) + "\n")
Path("out/prefixes.jsonl").write_text("\n".join(lines[40:]) + "\n")
print(f"train {len(lines)-40}  heldout 40")
PY
```

Check a few lines:

- `at_edit` prefix still contains a `sed -i` / `cat >` / `tee`
- `pre_edit` does not
- `cold` has 1 or 2 assistant turns

---

## 4. Roll v125 × 6

Observations are still a **stub** (`(stub) ran: …`). That is enough for a first pack: the judge mostly looks at whether the **command** happened. Real docker/repo grounding is an upgrade if the gate does not move.

Four roller processes, one per replica. Each takes every 4th prefix.

```bash
mkdir -p out
for i in 0 1 2 3; do
  uv run python roll_king.py \
    --prefixes out/prefixes.jsonl \
    --n 6 \
    --backend openai \
    --base-url http://127.0.0.1:$((8000 + i))/v1 \
    --model v125 \
    --temperature 1.0 \
    --shard $i/4 \
    --out out/rollouts.$i.jsonl &
done
wait
cat out/rollouts.{0,1,2,3}.jsonl > out/rollouts.jsonl
wc -l out/rollouts.jsonl
```

~300 prefixes × 6 ≈ 1800 lines. `--backend fake` is only for learning the file shape. It is not v125.

When this step is done: **`pkill -f 'vllm serve'`** before you train. Training needs all four cards.

If the server dies mid-way, do not mix two jsonl files with overlapping `sample_id` unless you know what you are doing. Restart clean or concatenate only complete prefixes.

---

## 5. Keep / drop

```bash
uv run python filter_rollouts.py --rollouts out/rollouts.jsonl --out-dir out
```

A keep must pass **all** that apply:

| reason if it fails | meaning |
|---|---|
| `loop:…` | same command 4×, or ≥ half the commands are repeats |
| `bad_turn:…` | a turn had no usable bash block |
| `no_source_edit` | cold / pre_edit continuation never edited source |
| `restart_wide_search` | pre_edit first new command is `find .` / `grep -r` |
| `no_verify_after_edit` | after the last source edit, no `pytest` / `python -c` / `cargo test` / … |
| `edit_before_repro` | cold: first edit before any failing-test / `python -c` |
| `submit_without_verify` | submitted after an edit with no check |
| `search_spam` | pre_edit / at_edit: ≥3 repo-wide `find` / `grep -r` after the cut |
| `infra_thrash` | ≥2 `pip install pytest` / `which pytest` style commands |

`at_edit` prefixes already contain the first edit. Then we only require a verify in the continuation.

**Ready bar**

- ≥ 250 keeps → `out/keep.jsonl`
- ≥ 150 tickets with **both** a keep and a drop (those become DPO)
- tickets that are 6/6 keep or 6/6 drop: weak. Re-roll those prefixes with `--n 8` or drop the ticket

Open 10 keeps and 10 drops before you pack:

- keep = edit (or finish the existing edit) → **the check the ticket cares about** → stop
- drop = edit then grep, or pre_edit starts with `find .`, or at_edit never checks

If keeps look like essays or extra `git log`, the filter is too loose — do not pack yet.

Do **not** keep “only tests, no edit” on cold. That kills `cold × action`.

Do **not** call GLM to label. Live questions change every fight. These rules are the teacher.

---

## 6. Pack SFT + DPO

```bash
uv run python pack_pairs.py \
  --keep out/keep.jsonl \
  --drop out/drop.jsonl \
  --out-dir out
```

- `out/sft.jsonl` = every keep. Loss later only after `n_prefix`.
- `out/dpo.jsonl` = same `sample_id`, keep continuation vs drop continuation.

A keep with no drop sibling is SFT-only. That is fine. DPO without a pair is skipped on purpose.

```bash
uv run python train.py --king /data/kings/v125 --sft out/sft.jsonl --dpo out/dpo.jsonl
```

Reads `out/train/recipe.json`. It does not train.

---

## 7. Train (your trainer, all 4 GPUs, this step only)

vLLM must be dead so training can use all 4. Then:

```bash
# example shape — plug into TRL / your trainer. train.py --run is refused.
torchrun --nproc_per_node=4 --standalone your_sft.py \
  --base /data/kings/v125 \
  --data out/sft.jsonl \
  --lora-targets attn,shared_expert \
  --lr 5e-6 --epochs 1 \
  --per_device_train_batch_size 1 \
  --gradient_checkpointing \
  --bf16

torchrun --nproc_per_node=4 --standalone your_dpo.py \
  --base out/train/sft-merged \
  --data out/dpo.jsonl \
  --beta 0.2 --lr 5e-6 --epochs 1 \
  --per_device_train_batch_size 1 \
  --gradient_checkpointing \
  --bf16
```

Effective batch = 4, all four cards working. Check `nvidia-smi`: none should be 0%.

Recipe we want (same class as v125←v124):

| knob | value |
|---|---|
| base | **v125** |
| LoRA | attention + **shared** expert only (not the 256 routed experts) |
| LR | `5e-6` |
| epochs | 1 (cut to ~⅓ if cold edit rate starts falling) |
| DPO β | `0.2` |
| loss | assistant turns **after** `n_prefix` only |
| never | `embed_tokens`, `lm_head`, layernorms, tokenizer, `chat_template` |

Order: SFT on `sft.jsonl`, then DPO on `dpo.jsonl` from the SFT merge.

Merge LoRA before any upload. The subnet rejects `adapter_config`.

---

## 8. Gate (held-out, before you submit)

Roll **held-out** prefixes × 2 with **raw v125** and with **your merge**. Same filter, no GLM.

Training done → kill it → serve **four replicas again** (same loop as §1, swap the weights).

Baseline (raw v125), then your merge as `challenger`. One after the other. All 4 GPUs both times.

```bash
# baseline — 4 replicas of v125 already up
for i in 0 1 2 3; do
  uv run python roll_king.py --prefixes out/heldout.jsonl --n 2 \
    --backend openai --base-url http://127.0.0.1:$((8000 + i))/v1 --model v125 \
    --temperature 1.0 --shard $i/4 --out out/heldout-v125.$i.jsonl &
done
wait
cat out/heldout-v125.{0,1,2,3}.jsonl > out/heldout-v125.jsonl

pkill -f 'vllm serve' || true
# start 4 replicas of the merge on ports 8000-8003, --served-model-name challenger
for i in 0 1 2 3; do
  uv run python roll_king.py --prefixes out/heldout.jsonl --n 2 \
    --backend openai --base-url http://127.0.0.1:$((8000 + i))/v1 --model challenger \
    --temperature 1.0 --shard $i/4 --out out/heldout-chal.$i.jsonl &
done
wait
cat out/heldout-chal.{0,1,2,3}.jsonl > out/heldout-chal.jsonl

uv run python gate.py --rollouts out/heldout-v125.jsonl
uv run python gate.py --rollouts out/heldout-chal.jsonl
```

| meter | vs raw v125 |
|---|---|
| `cold_verify_rate` | up |
| `pre_edit_no_wide` | up |
| `pre_edit_edit_rate` | up |
| **`cold_edit_rate`** | **flat** — if this drops, revert |
| `loop_rate` | ~0 |

If verify is up and cold edits held, this pack can beat v125 the same way v125 beat v124.

`gate.py --run` is refused. The command above scores files you already rolled.

---

## Do not

- Serve and train at the same time (they would steal cards from each other).
- Serve or train **v124**. Wrong king.
- SFT parquet gold completions or GLM dumps.
- Mix v124 rollouts with v125 rollouts.
- Overweight SWE-hero (live n is ~5).
- Train claims / longer THOUGHT.
- Touch tokenizer or `lm_head`.
- Submit an unmerged adapter.
- Treat `cut_prefixes.py --self-test` / `--backend fake` as a real dataset.

---

## Why this should beat v125

v125 is a LoRA on v124 (attn + shared expert). It won by verifying and editing more often than v124, but those tags are still ~half-yes. Same method, new base: keep v125’s good day, drop the day it restarts `grep` or skips the check.
