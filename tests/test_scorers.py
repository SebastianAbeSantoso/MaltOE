"""Scorers, on small synthetic problems, so they run offline and fast."""

import stat
import sys
import textwrap

import pytest

from harness.scorers import gsm8k, humaneval, perplexity
from harness.scorers.counts import SolvedCount, describe_difference, difference_is_noise

# ----- GSM8K -----


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("so we get 1,234 total.\n#### 1,234", 1234),
        ("The result is 18.", 18),
        ("#### -7", -7),
        ("#### $12.50", 12.5),
        ("first 5, then 6.\n#### 11\nthat is all", 11),
        ("wrong guess 4 ... but #### 9", 9),  # the marker wins over other numbers
        ("no numbers here", None),
        ("", None),
    ],
)
def test_final_number(text, expected):
    assert gsm8k.final_number(text) == expected


def test_gold_answer_and_correctness():
    problem = {"answer": "She has 3 + 4 = <<3+4=7>>7 apples.\n#### 7"}
    assert gsm8k.gold_answer(problem) == 7
    assert gsm8k.is_correct(problem, "#### 7")
    assert gsm8k.is_correct(problem, "It is 7.")
    assert not gsm8k.is_correct(problem, "#### 8")
    assert not gsm8k.is_correct(problem, "I cannot tell.")
    assert gsm8k.gold_answer({"answer": "x\n#### 1,500"}) == 1500


# ----- HumanEval -----

ADD = {
    "task_id": "Synthetic/0",
    "entry_point": "add",
    "prompt": 'from typing import List\n\n\ndef add(a: int, b: int) -> int:\n    """Add two numbers."""\n',
    "test": "def check(candidate):\n    assert candidate(1, 2) == 3\n    assert candidate(-1, 1) == 0\n",
}


def fenced(code: str) -> str:
    return f"Here you go:\n```python\n{code}\n```\nHope that helps."


def test_correct_answer_passes():
    assert humaneval.passes(ADD, fenced("def add(a: int, b: int) -> int:\n    return a + b"))


def test_wrong_answer_fails():
    assert not humaneval.passes(ADD, fenced("def add(a, b):\n    return a - b"))


def test_missing_function_fails():
    assert not humaneval.passes(ADD, fenced("def plus(a, b):\n    return a + b"))
    assert not humaneval.passes(ADD, "I think the answer is a + b.")


def test_unfenced_code_is_still_scored():
    assert humaneval.passes(ADD, "def add(a, b):\n    return a + b\n")


def test_the_last_function_block_wins():
    reply = (
        fenced("def add(a, b):\n    return 0")
        + "\nBetter:\n"
        + fenced("def add(a, b):\n    return a + b")
    )
    assert humaneval.passes(ADD, reply)


def test_imports_from_the_prompt_are_available_to_the_reply():
    # the reply uses List without importing it; the prompt did
    assert humaneval.passes(
        ADD,
        fenced("def add(a: int, b: int) -> int:\n    xs: List[int] = [a, b]\n    return sum(xs)"),
    )


def test_infinite_loop_times_out_and_fails():
    assert not humaneval.passes(
        ADD, fenced("def add(a, b):\n    while True:\n        pass"), timeout_s=1
    )


def test_exiting_early_does_not_count_as_passing():
    # exit code 0, but the problem's checks never ran
    assert not humaneval.passes(
        ADD, fenced("import sys\nsys.exit(0)\ndef add(a, b):\n    return a - b")
    )
    assert not humaneval.passes(
        ADD, fenced("def add(a, b):\n    return a - b\nimport os\nos._exit(0)")
    )


def test_printing_the_sentinel_does_not_count_as_passing():
    reply = fenced(
        f"print({humaneval.PASS_SENTINEL!r})\nimport sys\nsys.exit(0)\ndef add(a, b):\n    return a - b"
    )
    assert not humaneval.passes(ADD, reply)


def test_a_crashing_reply_fails():
    assert not humaneval.passes(ADD, fenced("def add(a, b):\n    raise RuntimeError('x')"))


# ----- solved counts -----


def test_count_formats_as_n_over_total():
    assert str(SolvedCount(14, 20)) == "14/20"


@pytest.mark.parametrize("bad", [(21, 20, 0), (-1, 20, 0), (3, 20, 21), (3, 20, -1)])
def test_impossible_counts_are_rejected(bad):
    with pytest.raises(ValueError):
        SolvedCount(*bad)


def test_one_or_two_problems_is_noise_three_is_not():
    a = SolvedCount(14, 20)
    assert difference_is_noise(a, SolvedCount(14, 20))
    assert difference_is_noise(a, SolvedCount(15, 20))
    assert difference_is_noise(a, SolvedCount(12, 20))
    assert not difference_is_noise(a, SolvedCount(11, 20))
    assert "within noise" in describe_difference(a, SolvedCount(13, 20))
    assert "3 problems apart" in describe_difference(a, SolvedCount(11, 20))


def test_counts_over_different_sizes_cannot_be_compared():
    with pytest.raises(ValueError):
        difference_is_noise(SolvedCount(1, 5), SolvedCount(1, 20))


# ----- perplexity -----

LOG = textwrap.dedent(
    """\
    llama_model_loader: loaded meta data
    perplexity: calculating perplexity over 64 chunks, n_ctx=512
    [1]7.1211,[2]8.0102,[64]9.4401,
    Final estimate: PPL = 9.4401 +/- 0.05213
    llama_perf_context_print: total time = 1234 ms
    """
)


def test_parse_reads_the_final_estimate():
    assert perplexity.parse(LOG) == (9.4401, 0.05213)


def test_parse_fails_loudly_without_a_result():
    with pytest.raises(perplexity.PerplexityError, match="no 'Final estimate'"):
        perplexity.parse("llama_model_loader: loaded\nerror: something broke")


def test_command_uses_the_frozen_window_settings(tmp_path):
    cmd = perplexity.command(
        tmp_path / "bin",
        tmp_path / "m.gguf",
        tmp_path / "w.raw",
        context=512,
        chunks=64,
        extra_args=["-ngl", "99"],
    )
    assert cmd[cmd.index("-c") + 1] == "512" and cmd[cmd.index("--chunks") + 1] == "64"
    assert cmd[-2:] == ["-ngl", "99"]


posix_only = pytest.mark.skipif(
    sys.platform == "win32", reason="uses a shell script as a fake binary"
)


def fake_binary(tmp_path, body: str):
    f = tmp_path / "llama-perplexity"
    f.write_text("#!/bin/sh\n" + body, encoding="utf-8")
    f.chmod(f.stat().st_mode | stat.S_IEXEC)
    return f


@posix_only
def test_run_reads_stderr_of_the_engine(tmp_path):
    binary = fake_binary(tmp_path, "echo 'Final estimate: PPL = 7.5 +/- 0.1' >&2\n")
    result = perplexity.run(binary, tmp_path / "m.gguf", tmp_path / "w.raw")
    assert (result.ppl, result.stderr_estimate) == (7.5, 0.1)


@posix_only
def test_run_reports_engine_failure(tmp_path):
    binary = fake_binary(tmp_path, "echo 'cudaMalloc failed: out of memory' >&2\nexit 1\n")
    with pytest.raises(perplexity.PerplexityError, match="out of memory"):
        perplexity.run(binary, tmp_path / "m.gguf", tmp_path / "w.raw")


@posix_only
def test_run_times_out(tmp_path):
    binary = fake_binary(tmp_path, "sleep 5\n")
    with pytest.raises(perplexity.PerplexityError, match="timed out"):
        perplexity.run(binary, tmp_path / "m.gguf", tmp_path / "w.raw", timeout_s=0.5)
