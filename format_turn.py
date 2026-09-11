"""Turn shape the eval harness accepts. Keep this tiny and local."""

from __future__ import annotations

import re

_BASH = re.compile(r"```(?:bash|sh|shell)?[ \t]*\n(.*?)```", re.DOTALL)
_SUBMIT = re.compile(
    r"\b(?:COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT|SUBMIT_TASK_[0-9A-F]{8}"
    r"|FINALIZE_AND_SUBMIT_TASK_OUTPUT)\b"
)


def first_bash(text: str) -> str:
    match = _BASH.search(text or "")
    return (match.group(1) if match else "").strip()


def usable(text: str) -> str:
    """Why this turn is unusable, or empty if it is fine."""
    if not (text or "").strip():
        return "empty"
    if not _BASH.search(text):
        return "no bash block"
    return ""


def is_submit(command: str) -> bool:
    return bool(_SUBMIT.search(command or ""))


def wrap_returncode(body: str, returncode: int = 0) -> str:
    return f"<returncode>{returncode}</returncode>\n<output>\n{body}\n</output>"
