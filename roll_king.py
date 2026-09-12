#!/usr/bin/env python3
"""Continue paused tickets. Step 03 — rollouts only, no keep/drop yet.

A rollout is: prefix + (assistant turn, observation) repeated until horizon or submit.

Backends:
  fake     — scripted king, so you can see the file shape with no GPU
  openai   — any OpenAI-compatible server (vLLM serving v125)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable

from cut_prefixes import demo_prefix
from format_turn import first_bash, is_submit, usable, wrap_returncode

KingFn = Callable[[list[dict[str, str]], int], str]


def stub_observe(command: str) -> str:
    """Placeholder env. Real grounding is a later step."""
    cmd = command.replace("\n", " ")[:160]
    return wrap_returncode(f"(stub) ran: {cmd}")


def fake_king_verify(messages: list[dict[str, str]], step: int) -> str:
    """Does the missing habit: edit, then check."""
    if step == 0:
        return (
            "THOUGHT: the function returns 0; change it.\n\n"
            "```bash\nsed -i 's/return 0/return a + b/' src/math.py\n```"
        )
    if step == 1:
        return (
            "THOUGHT: taste the soup after the change.\n\n"
            "```bash\npython -c 'from src.math import add; print(add(1,2))'\n```"
        )
    return (
        "THOUGHT: done and checked.\n\n"
        "```bash\necho COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\n```"
    )


def fake_king_skip_verify(messages: list[dict[str, str]], step: int) -> str:
    """Edits, then keeps grepping — the king's leak."""
    if step == 0:
        return (
            "THOUGHT: fix it.\n\n"
            "```bash\nsed -i 's/return 0/return a + b/' src/math.py\n```"
        )
    return (
        "THOUGHT: look around more.\n\n"
        f"```bash\ngrep -n add src/math.py\n```"
    )


class OpenAIKing:
    def __init__(self, base_url: str, model: str, temperature: float) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.temperature = temperature

    def __call__(self, messages: list[dict[str, str]], step: int) -> str:
        payload = json.dumps(
            {
                "model": self.model,
                "messages": messages,
                "temperature": self.temperature,
                "max_tokens": 1024,
            }
        ).encode()
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = json.loads(resp.read().decode())
        except urllib.error.URLError as exc:
            raise SystemExit(f"openai backend failed: {exc}") from exc
        return str(data["choices"][0]["message"]["content"] or "")


def roll_one(
    prefix: dict[str, Any],
    *,
    rollout: int,
    n: int,
    king: KingFn,
    observe: Callable[[str], str],
    horizon: int,
) -> dict[str, Any]:
    messages = [dict(m) for m in prefix["messages"]]
    continuation: list[dict[str, str]] = []
    commands: list[str] = []
    stopped = "horizon"
    for step in range(horizon):
        text = king(messages, step)
        why = usable(text)
        messages.append({"role": "assistant", "content": text})
        continuation.append({"role": "assistant", "content": text})
        if why:
            stopped = f"bad_turn:{why}"
            break
        command = first_bash(text)
        commands.append(command)
        if is_submit(command):
            stopped = "submit"
            break
        obs = observe(command)
        messages.append({"role": "user", "content": obs})
        continuation.append({"role": "user", "content": obs})
    return {
        "sample_id": prefix["sample_id"],
        "rollout": rollout,
        "n_rollouts": n,
        "phase_wanted": prefix.get("phase_wanted"),
        "source": prefix.get("source"),
        "instance_id": prefix.get("instance_id"),
        "horizon": horizon,
        "stopped": stopped,
        "commands": commands,
        "messages": messages,
        "continuation": continuation,
    }


def _parse_shard(spec: str) -> tuple[int, int]:
    left, _, right = spec.partition("/")
    try:
        shard_i, shard_n = int(left), int(right)
    except ValueError as exc:
        raise SystemExit(f"bad --shard {spec!r}, want i/n") from exc
    if shard_n < 1 or not (0 <= shard_i < shard_n):
        raise SystemExit(f"bad --shard {spec!r}, want i/n with 0 <= i < n")
    return shard_i, shard_n


def load_prefixes(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text().splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def run_self_test() -> int:
    prefix = demo_prefix("cold")
    print(
        f"prefix {prefix['sample_id']} phase={prefix['phase_wanted']} "
        f"asst_in_prefix={prefix['asst_in_prefix']}"
    )
    kings = (fake_king_verify, fake_king_skip_verify)
    out = []
    for i, king in enumerate(kings, start=1):
        rec = roll_one(prefix, rollout=i, n=2, king=king, observe=stub_observe, horizon=4)
        out.append(rec)
        print(f"\nrollout {i} stopped={rec['stopped']} commands={rec['commands']}")
    verify_has_check = any("python -c" in c for c in out[0]["commands"][1:])
    skip_has_only_grep = all("grep" in c or "sed -i" in c for c in out[1]["commands"])
    ok = verify_has_check and skip_has_only_grep and out[0]["stopped"] == "submit"
    print("\nPASS" if ok else "\nFAIL")
    return 0 if ok else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--prefixes", type=Path, help="prefixes.jsonl from step 02")
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "out" / "rollouts.jsonl")
    parser.add_argument("--n", type=int, default=4, help="rollouts per prefix")
    parser.add_argument("--backend", choices=("fake", "openai"), default="fake")
    parser.add_argument("--base-url", default=os.environ.get("KING_BASE_URL", "http://127.0.0.1:8000/v1"))
    parser.add_argument("--model", default=os.environ.get("KING_MODEL", "v125"))
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument(
        "--shard",
        default="0/1",
        help="i/n: this process takes prefixes where index %% n == i (4 GPUs: 0/4 … 3/4)",
    )
    args = parser.parse_args()

    if args.self_test:
        return run_self_test()
    if not args.prefixes or not args.prefixes.exists():
        print("need --prefixes out/prefixes.jsonl (or --self-test)", file=sys.stderr)
        return 2

    prefixes = load_prefixes(args.prefixes)
    shard_i, shard_n = _parse_shard(args.shard)
    prefixes = [p for i, p in enumerate(prefixes) if i % shard_n == shard_i]
    if args.backend == "fake":
        # even slots verify, odd slots skip — so you always have a pair for step 04
        def king_for(r: int) -> KingFn:
            return fake_king_verify if r % 2 == 1 else fake_king_skip_verify
    else:
        real = OpenAIKing(args.base_url, args.model, args.temperature)

        def king_for(r: int) -> KingFn:
            return real

    args.out.parent.mkdir(parents=True, exist_ok=True)
    n_wrote = 0
    with args.out.open("w") as fh:
        for prefix in prefixes:
            horizon = int(prefix.get("horizon") or 12)
            for r in range(1, args.n + 1):
                rec = roll_one(
                    prefix,
                    rollout=r,
                    n=args.n,
                    king=king_for(r),
                    observe=stub_observe,
                    horizon=horizon,
                )
                rec["backend"] = args.backend
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n_wrote += 1
    print(f"wrote {n_wrote} rollouts ({len(prefixes)} prefixes × {args.n}) -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
