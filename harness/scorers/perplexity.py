"""Perplexity: drive llama-perplexity and read its result."""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

_FINAL = re.compile(r"Final estimate: PPL = ([0-9.]+) \+/- ([0-9.]+)")


class PerplexityError(RuntimeError):
    pass


@dataclass(frozen=True)
class PerplexityResult:
    ppl: float
    stderr_estimate: float  # the "+/-" llama-perplexity prints
    command: list[str]


def parse(output: str) -> tuple[float, float]:
    """Read 'Final estimate: PPL = X +/- Y' from llama-perplexity output."""
    m = _FINAL.search(output)
    if not m:
        tail = " | ".join([ln for ln in output.splitlines() if ln.strip()][-4:])
        raise PerplexityError(f"no 'Final estimate' line in the output; last lines: {tail}")
    return float(m.group(1)), float(m.group(2))


def command(
    binary: Path, model: Path, text_file: Path, *, context: int, chunks: int, extra_args: list[str]
) -> list[str]:
    return [
        str(binary),
        "-m",
        str(model),
        "-f",
        str(text_file),
        "-c",
        str(context),
        "--chunks",
        str(chunks),
        *extra_args,
    ]


def run(
    binary: Path,
    model: Path,
    text_file: Path,
    *,
    context: int = 512,
    chunks: int = 64,
    extra_args: list[str] | None = None,
    timeout_s: float | None = None,
) -> PerplexityResult:
    """Run llama-perplexity on ``text_file`` (the frozen wikitext-2 test file) and return the result.

    ``extra_args`` carry engine placement such as ``["-ngl", "99", "-ncmoe", "48"]``.
    """
    cmd = command(
        binary, model, text_file, context=context, chunks=chunks, extra_args=extra_args or []
    )
    try:
        done = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
    except subprocess.TimeoutExpired as e:
        raise PerplexityError(f"llama-perplexity timed out after {timeout_s} s") from e
    # llama.cpp writes its log to stderr
    output = (done.stdout or "") + "\n" + (done.stderr or "")
    if done.returncode != 0:
        tail = " | ".join([ln for ln in output.splitlines() if ln.strip()][-4:])
        raise PerplexityError(f"llama-perplexity exited with {done.returncode}: {tail}")
    ppl, err = parse(output)
    return PerplexityResult(ppl, err, cmd)
