"""The frozen evaluation set (Req 10.4, 10.7)."""

import importlib.util
import json
import re
from pathlib import Path

import pytest

from harness import datasets
from harness.scorers import humaneval
from harness.settings import chat_request, load_settings, render_prompt
from tests.conftest import needs_network

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module", params=["coding", "reasoning"])
def frozen(request):
    return datasets.load_frozen(request.param)


def test_each_set_has_twenty_distinct_problems(frozen):
    assert len(frozen.problems) == datasets.DRAW_SIZE == 20
    assert len(set(frozen.indices)) == 20
    assert len({p["id"] for p in frozen.problems}) == 20
    assert all(0 <= i < frozen.population for i in frozen.indices)
    assert all(re.fullmatch(r"[0-9a-f]{64}", p["sha256"]) for p in frozen.problems)


def test_populations_match_the_public_datasets():
    assert datasets.load_frozen("coding").population == 164  # HumanEval
    assert datasets.load_frozen("reasoning").population == 1319  # GSM8K test


def test_the_draw_is_deterministic_and_the_two_draws_are_independent():
    a = datasets.draw_indices(164, salt="humaneval")
    assert a == datasets.draw_indices(164, salt="humaneval")
    assert a != datasets.draw_indices(164, salt="gsm8k")
    assert datasets.load_frozen("coding").indices == a


def test_the_task_1_smoke_subset_is_the_first_five_of_each_frozen_list():
    # The G3 width-quality gate used the first 5 + 5 problems. Freezing the same draw means the
    # full evaluation cannot have been chosen after seeing how those 10 went.
    assert datasets.load_frozen("coding").indices[:5] == [11, 5, 93, 83, 61]
    assert datasets.load_frozen("reasoning").indices[:5] == [22, 194, 197, 809, 760]


def test_frozen_draw_matches_the_spike_that_ran_in_task_1():
    path = ROOT / "spikes" / "width_emulation" / "smoke_eval.py"
    if not path.exists():
        pytest.skip("spike removed")
    spec = importlib.util.spec_from_file_location("smoke_eval_spike", path)
    spike = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(spike)
    assert spike.SEED == datasets.SEED
    assert spike.draw(164, salt="humaneval") == datasets.load_frozen("coding").indices
    assert spike.draw(1319, salt="gsm8k") == datasets.load_frozen("reasoning").indices


def test_sources_and_licences_are_recorded():
    for kind in ("coding", "reasoning"):
        raw = json.loads((datasets.EVALSET_DIR / f"{kind}.json").read_text(encoding="utf-8"))
        assert raw["source"]["url"].startswith("https://")
        assert raw["source"]["licence"] == "MIT"
        assert re.fullmatch(r"[0-9a-f]{64}", raw["source"]["sha256"])


def test_perplexity_set_matches_the_design():
    ppl = json.loads((datasets.EVALSET_DIR / "perplexity.json").read_text(encoding="utf-8"))
    assert ppl["context_tokens"] == datasets.PPL_CONTEXT == 512
    assert ppl["chunks"] == datasets.PPL_CHUNKS == 64
    assert ppl["llama_perplexity_args"] == ["-c", "512", "--chunks", "64"]
    assert "CC BY-SA" in ppl["source"]["licence"]


# ----- sampling and prompts -----


def test_sampling_is_greedy_seeded_and_thinking_is_off():
    s = load_settings().sampling
    assert s["temperature"] == 0
    assert isinstance(s["seed"], int)
    assert s["chat_template_kwargs"] == {"enable_thinking": False}
    assert s["stream"] is False


def test_chat_request_carries_the_frozen_settings():
    st = load_settings()
    body = chat_request(st, "coding", "hello")
    assert body["messages"] == [{"role": "user", "content": "hello"}]
    assert body["temperature"] == 0 and body["seed"] == st.sampling["seed"]
    assert body["max_tokens"] == st.sampling["max_tokens"]["coding"]
    assert (
        chat_request(st, "reasoning", "x")["max_tokens"] == st.sampling["max_tokens"]["reasoning"]
    )


def test_settings_fingerprint_changes_when_anything_changes(tmp_path):
    base = load_settings()
    for name in ("sampling.json", "prompts.json"):
        (tmp_path / name).write_text(
            (datasets.EVALSET_DIR / name).read_text(encoding="utf-8"), encoding="utf-8"
        )
    assert load_settings(tmp_path).fingerprint == base.fingerprint
    s = json.loads((tmp_path / "sampling.json").read_text(encoding="utf-8"))
    s["seed"] = 2
    (tmp_path / "sampling.json").write_text(json.dumps(s), encoding="utf-8")
    assert load_settings(tmp_path).fingerprint != base.fingerprint


def test_prompts_render_both_kinds():
    st = load_settings()
    assert "def f():" in render_prompt(st, "coding", {"prompt": "def f():\n    pass\n"})
    assert "2+2?" in render_prompt(st, "reasoning", {"question": "2+2?"})
    with pytest.raises(ValueError):
        render_prompt(st, "poetry", {})


# ----- against the real datasets (opt-in) -----


@needs_network
def test_cache_matches_the_frozen_files(dataset_cache):
    assert datasets.cmd_verify(dataset_cache) == 0


@needs_network
def test_freezing_again_reproduces_the_committed_files(dataset_cache):
    for kind in ("coding", "reasoning"):
        committed = json.loads((datasets.EVALSET_DIR / f"{kind}.json").read_text(encoding="utf-8"))
        assert datasets.build_frozen(kind, dataset_cache) == committed
    committed = json.loads((datasets.EVALSET_DIR / "perplexity.json").read_text(encoding="utf-8"))
    assert datasets.build_perplexity(dataset_cache) == committed


@needs_network
def test_scorer_accepts_every_reference_solution_and_rejects_a_stub(dataset_cache):
    """End-to-end check of the coding scorer on the 20 frozen problems."""
    problems, _ = datasets.load_humaneval(dataset_cache)
    chosen = datasets.select(datasets.load_frozen("coding"), problems)
    assert len(chosen) == 20
    for p in chosen:
        reference = f"```python\n{p['prompt']}{p['canonical_solution']}\n```"
        stub = f"```python\n{p['prompt']}    pass\n```"
        assert humaneval.passes(p, reference), f"{p['task_id']}: reference solution failed"
        assert not humaneval.passes(p, stub), f"{p['task_id']}: a stub passed"
