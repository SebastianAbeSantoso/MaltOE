"""GSM8K scoring: exact match on the final number."""

from __future__ import annotations

import re

_NUM = r"-?\d[\d,]*\.?\d*"


def final_number(text: str) -> float | None:
    """The answer in a response: the number after the last '####', else the last number in the text."""
    after_marker = re.findall(r"####\s*\$?\s*(" + _NUM + ")", text)
    candidate = after_marker[-1] if after_marker else (re.findall(_NUM, text) or [None])[-1]
    if candidate is None:
        return None
    try:
        return float(candidate.replace(",", "").rstrip("."))
    except ValueError:
        return None


def gold_answer(problem: dict) -> float:
    """GSM8K stores the worked solution; the number after '####' is the answer."""
    return float(problem["answer"].split("####")[-1].strip().replace(",", ""))


def is_correct(problem: dict, response: str) -> bool:
    value = final_number(response)
    return value is not None and abs(value - gold_answer(problem)) < 1e-6
