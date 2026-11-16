"""Sampling settings and prompt templates shared by every run (Req 10.4).

They live in ``harness/evalset/sampling.json`` and ``prompts.json`` so that they are fixed in the
repository and a fingerprint of them can be stored next to every result.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from harness.datasets import EVALSET_DIR


@dataclass(frozen=True)
class Settings:
    sampling: dict
    prompts: dict
    fingerprint: str  # sha256 of the canonical JSON of both, recorded with every result


def load_settings(directory: Path = EVALSET_DIR) -> Settings:
    sampling = json.loads((directory / "sampling.json").read_text(encoding="utf-8"))
    prompts = json.loads((directory / "prompts.json").read_text(encoding="utf-8"))
    canon = json.dumps(
        {"sampling": sampling, "prompts": prompts}, sort_keys=True, separators=(",", ":")
    )
    return Settings(sampling, prompts, hashlib.sha256(canon.encode("utf-8")).hexdigest())


def render_prompt(settings: Settings, kind: str, problem: dict) -> str:
    """The user message for one problem. ``kind`` is "coding" or "reasoning"."""
    if kind == "coding":
        return settings.prompts["coding"].format(prompt=problem["prompt"])
    if kind == "reasoning":
        return settings.prompts["reasoning"].format(question=problem["question"])
    raise ValueError(f"unknown kind {kind!r}")


def chat_request(settings: Settings, kind: str, user_message: str) -> dict:
    """An OpenAI-compatible /v1/chat/completions body with the frozen sampling settings."""
    s = settings.sampling
    return {
        "messages": [{"role": "user", "content": user_message}],
        "temperature": s["temperature"],
        "seed": s["seed"],
        "max_tokens": s["max_tokens"][kind],
        "stream": s["stream"],
        "chat_template_kwargs": s["chat_template_kwargs"],
    }
