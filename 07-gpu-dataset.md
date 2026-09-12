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

`train.py --run` and `gate.py --run` are still refused. After the jsonl files exist, train with `run_train.py` (commands in §7). `train.py` only writes `out/train/recipe.json`.

---

## 0. Box (4× H200)

Goal: **no idle GPU** on any step that loads the model. One step at a time. When a step finishes, stop it so the next step can take every card.

Cut / filter / pack are CPU — GPUs will be idle then. That is fine. Download is disk.

| step | keep all 4 busy |
|---|---|
| roll ×6, gate | one vLLM per GPU (4 servers) + one roller per server |
| SFT, DPO | `torchrun --nproc_per_node=4` |

CPython 3.12, this repo, `uv`. `huggingface-cli` is dead — use `hf`.

```bash
# if `uv` is missing
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env

git clone https://github.com/luckyhorse2026/albedo-training.git
cd albedo-training
uv sync --extra dev
uv tool install huggingface_hub   # provides `hf`; do not use huggingface-cli
uv run python cut_prefixes.py --self-test
uv run python roll_king.py --self-test
uv run python filter_rollouts.py --self-test
uv run python pack_pairs.py --self-test
```

All four must print `PASS`. If not, stop.

`data/` and `out/` are gitignored. They stay on the box.

### This box (2026-09-12)

First full pass on 4× H200. Numbers to match if you rerun:

| artifact | n |
|---|---|
| `out/all-prefixes.jsonl` | 327 (wanted 340, short) |
| train / held-out | 287 / 40 |
| `out/rollouts.jsonl` | 1722 (287 × 6) |
| keep / drop | 383 / 1339 (`no_source_edit` 883, `bad_turn` 605) |
| tickets with both keep+drop | 142 (bar was ≥150) |
| `out/sft.jsonl` / `out/dpo.jsonl` | 383 / 259 |

SFT LoRA finished. DPO at `--max-seq-len 8192` OOM’d (~130 GB used, tried +15 GB). `--max-seq-len 4096` plus `precompute_ref_log_probs` fits (~77 GB). TRL’s default ref-logps cache is a per-rank `/tmp/hf_datasets-*` file — ranks 1–3 then `FileNotFoundError`. `run_train.py` pins that cache under `out/train/dpo-adapter/_ref_logps`.

DPO then finished: 43 steps, ~3.7 min, `train_loss` 2.247, `rewards/accuracies` 0.58. Merged to `out/train/challenger`.

Held-out ×2 (80 rolls each). **vLLM is dead after train — serve again before you roll** (the “already up” comment below is a lie if you just finished §7).

| meter | v125 | challenger |
|---|---|---|
| n | 80 cold | 80 cold |
| cold_verify_rate | 0.650 | 0.675 |
| cold_edit_rate | 0.388 | 0.400 |
| pre_edit_* | — | no pre_edit prefixes in this 40 |
| loop_rate | 0 | 0 |

Verify up, cold edits did not drop. Delta is small (~2 extra verifies). Scores: `out/gate-v125.json`, `out/gate-chal.json`.

---

## 1. Weights = v125, not v124

Public mirror:

- https://huggingface.co/dendriteholdings/albedo-qwen3.6-35b-king-CXXV

```bash
mkdir -p /data/kings
hf download dendriteholdings/albedo-qwen3.6-35b-king-CXXV \
  --local-dir /data/kings/v125
```

Install vLLM the way albedo pins it (`vllm==0.23.0`, `transformers==5.11.0`). The training venv does not include it.

```bash
uv venv /opt/vllm --python 3.12
uv pip install --python /opt/vllm/bin/python 'vllm==0.23.0'
uv pip install --python /opt/vllm/bin/python 'transformers==5.11.0'
ln -sfn /opt/vllm/bin/vllm "$HOME/.local/bin/vllm"
source $HOME/.local/bin/env
```

Serve **four replicas**, one GPU each. Same weights, four ports. Name them `v125`. Flags match albedo's `VllmServerGenerator` plus `--gdn-prefill-backend triton` so FlashInfer does not JIT-compile GDN kernels on first request (that path needs a matching `nvcc` and dies without it).

Stop leftovers **first**, in a separate command. `pkill -f 'vllm serve'` will also kill a shell whose command line still contains `vllm serve`.

```bash
# stop leftovers — run this by itself, then the loop
kill $(ps -eo pid,cmd | awk '/\/vllm serve/ && !/awk/ {print $1}') 2>/dev/null || true

export VLLM_USE_FLASHINFER_SAMPLER=0
for i in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$i /opt/vllm/bin/vllm serve /data/kings/v125 \
    --served-model-name v125 \
    --host 127.0.0.1 \
    --port $((8000 + i)) \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.90 \
    --kv-cache-dtype auto \
    --max-num-seqs 256 \
    --trust-remote-code \
    --generation-config vllm \
    --enable-prefix-caching \
    --limit-mm-per-prompt '{"image": 0, "video": 0}' \
    --max-model-len 32768 \
    --gdn-prefill-backend triton \
    > /tmp/vllm-$i.log 2>&1 &
done

for i in 0 1 2 3; do
  until curl -sf http://127.0.0.1:$((8000 + i))/v1/models >/dev/null; do sleep 2; done
  echo "gpu $i up on $((8000 + i))"
done
```

On 4× H200 this lands ~129GB per card. If one card is ~0, that replica died — read `/tmp/vllm-$i.log`. First boot can take a few minutes (compile cache). If one replica dies with a Triton `.so` race, restart **that** GPU only after another replica has compiled.

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

These are **datasets**. `hf` defaults to models and 401s without `--repo-type dataset`. Use a real glob (`[0-5]`), not bash braces inside quotes.

```bash
source $HOME/.local/bin/env   # so `hf` is the uv-tool one, not a leftover /opt/vllm/bin/hf

hf download ricdomolm/mini-coder-trajs-400k \
  --repo-type dataset \
  --include "data/train-0000[0-5]-of-00060.parquet" \
  --local-dir data/mini-coder

# Open-SWE is nested: data/minisweagent/<agent>/<split>/train-*.parquet
# `data/train-*.parquet` matches nothing. Pull a few qwen36 shards, then flatten.
hf download nvidia/Open-SWE-Traces \
  --repo-type dataset \
  --include "data/minisweagent/qwen36_27b/scale-swe/train-0000[0-2]-of-00017.parquet" \
  --local-dir data/open-swe-traces
mkdir -p data/open-swe-traces/data
for f in data/open-swe-traces/data/minisweagent/qwen36_27b/scale-swe/train-*.parquet; do
  ln -sfn "minisweagent/qwen36_27b/scale-swe/$(basename "$f")" \
    "data/open-swe-traces/data/$(basename "$f")"
done

hf download nvidia/SWE-Hero-openhands-trajectories \
  --repo-type dataset \
  --include "data/train-*.parquet" \
  --local-dir data/swe-hero

hf download AlienKevin/SWE-smith-rs-minimax-m2.5-trajectories \
  --repo-type dataset \
  --include "data/train-*.parquet" \
  --local-dir data/mini-coder-rs
```

Cutter only sees `data/<source>/data/train-*.parquet`. If a repo uses another prefix, symlink like Open-SWE above.

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

When this step is done, stop the replicas **before** you train (training needs all four cards):

```bash
kill $(ps -eo pid,cmd | awk '/\/vllm serve/ && !/awk/ {print $1}') 2>/dev/null || true
```

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

## 7. Train (all 4 GPUs, this step only)

`train.py` only writes `out/train/recipe.json`. `--run` is refused. The trainer is `run_train.py`.

vLLM must be dead so training can use all 4. Install train libs into the same venv that already has torch + transformers 5.11:

```bash
source $HOME/.local/bin/env
uv pip install --python /opt/vllm/bin/python peft accelerate trl datasets
```

Then:

```bash
# SFT LoRA on v125 (attn + shared expert, loss after n_prefix)
/opt/vllm/bin/torchrun --nproc_per_node=4 --standalone run_train.py sft \
  --base /data/kings/v125 \
  --data out/sft.jsonl \
  --out-dir out/train/sft-adapter \
  --lr 5e-6 --epochs 1 \
  --per-device-train-batch-size 1

# merge SFT adapter into a full folder (subnet rejects adapter_config)
/opt/vllm/bin/python run_train.py merge \
  --base /data/kings/v125 \
  --adapter out/train/sft-adapter \
  --out-dir out/train/sft-merged

# DPO LoRA on the SFT merge
/opt/vllm/bin/torchrun --nproc_per_node=4 --standalone run_train.py dpo \
  --base out/train/sft-merged \
  --data out/dpo.jsonl \
  --out-dir out/train/dpo-adapter \
  --beta 0.2 --lr 5e-6 --epochs 1 \
  --per-device-train-batch-size 1 \
  --max-seq-len 4096

# merge DPO adapter — this is the challenger weights
/opt/vllm/bin/python run_train.py merge \
  --base out/train/sft-merged \
  --adapter out/train/dpo-adapter \
  --out-dir out/train/challenger
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

Training done → kill train leftovers → serve **four replicas again** (same flags as §1). Do not roll until `/v1/models` answers. Connection refused = empty jsonl = a null `gate.json`.

Baseline (raw v125), then your merge as `challenger`. One after the other. All 4 GPUs both times.

```bash
# serve v125 again (copy of §1). Wait until all four /v1/models are up.
export VLLM_USE_FLASHINFER_SAMPLER=0
for i in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$i /opt/vllm/bin/vllm serve /data/kings/v125 \
    --served-model-name v125 --host 127.0.0.1 --port $((8000 + i)) \
    --tensor-parallel-size 1 --gpu-memory-utilization 0.90 \
    --kv-cache-dtype auto --max-num-seqs 256 --trust-remote-code \
    --generation-config vllm --enable-prefix-caching \
    --limit-mm-per-prompt '{"image": 0, "video": 0}' \
    --max-model-len 32768 --gdn-prefill-backend triton \
    > /tmp/vllm-$i.log 2>&1 &
done
for i in 0 1 2 3; do
  until curl -sf http://127.0.0.1:$((8000 + i))/v1/models >/dev/null; do sleep 2; done
  echo "gpu $i up on $((8000 + i))"
done

for i in 0 1 2 3; do
  uv run python roll_king.py --prefixes out/heldout.jsonl --n 2 \
    --backend openai --base-url http://127.0.0.1:$((8000 + i))/v1 --model v125 \
    --temperature 1.0 --shard $i/4 --out out/heldout-v125.$i.jsonl &
done
wait
cat out/heldout-v125.{0,1,2,3}.jsonl > out/heldout-v125.jsonl

kill $(ps -eo pid,cmd | awk '/\/vllm serve/ && !/awk/ {print $1}') 2>/dev/null || true
# wait until nvidia-smi memory is ~0, then serve the merge
export VLLM_USE_FLASHINFER_SAMPLER=0
for i in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$i /opt/vllm/bin/vllm serve out/train/challenger \
    --served-model-name challenger --host 127.0.0.1 --port $((8000 + i)) \
    --tensor-parallel-size 1 --gpu-memory-utilization 0.90 \
    --kv-cache-dtype auto --max-num-seqs 256 --trust-remote-code \
    --generation-config vllm --enable-prefix-caching \
    --limit-mm-per-prompt '{"image": 0, "video": 0}' \
    --max-model-len 32768 --gdn-prefill-backend triton \
    > /tmp/vllm-chal-$i.log 2>&1 &
done
for i in 0 1 2 3; do
  until curl -sf http://127.0.0.1:$((8000 + i))/v1/models >/dev/null; do sleep 2; done
  echo "challenger gpu $i up on $((8000 + i))"
done

for i in 0 1 2 3; do
  uv run python roll_king.py --prefixes out/heldout.jsonl --n 2 \
    --backend openai --base-url http://127.0.0.1:$((8000 + i))/v1 --model challenger \
    --temperature 1.0 --shard $i/4 --out out/heldout-chal.$i.jsonl &
done
wait
cat out/heldout-chal.{0,1,2,3}.jsonl > out/heldout-chal.jsonl

uv run python gate.py --rollouts out/heldout-v125.jsonl
cp out/gate.json out/gate-v125.json
uv run python gate.py --rollouts out/heldout-chal.jsonl
cp out/gate.json out/gate-chal.json
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

## 9. Hugging Face pack (not uploaded yet)

`run_train.py merge` rewrites `config.json` / tokenizer and drops the two preprocessor files. SN97 hashes those against genesis / the sitting king. Copy metadata from v125, keep only the trained shards:

```bash
SRC=out/train/challenger
KING=/data/kings/v125
DST=out/train/challenger-hf
mkdir -p "$DST"
for f in config.json generation_config.json tokenizer_config.json tokenizer.json \
         chat_template.jinja preprocessor_config.json video_preprocessor_config.json; do
  cp -a "$KING/$f" "$DST/$f"
done
ln "$SRC"/model-00001-of-00002.safetensors "$DST"/
ln "$SRC"/model-00002-of-00002.safetensors "$DST"/
cp -a "$SRC"/model.safetensors.index.json "$DST"/
# no adapter_config. README.md is allowed.
```

Repo name if you commit on-chain: `<namespace>/albedo-qwen3.6-35b-<suffix>` (lowercase namespace). Needs `HF_TOKEN` with write. This box is not logged in (`hf auth whoami`).

```bash
hf auth login --token "$HF_TOKEN"
hf repo create "$NS/albedo-qwen3.6-35b-chal-v1" --type model
hf upload "$NS/albedo-qwen3.6-35b-chal-v1" out/train/challenger-hf
```

`albedo publish` (wallet + on-chain reveal) is a later step in `../albedo`.

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
