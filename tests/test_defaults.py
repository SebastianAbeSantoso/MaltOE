"""Guards on the tunables in molt/defaults.py, against the design.md they mirror."""

from pathlib import Path

from molt import defaults

DESIGN = Path(__file__).resolve().parent.parent / ".kiro" / "specs" / "molt" / "design.md"


def test_values_are_sane():
    assert defaults.CHUNKS >= 3  # Req 4.1: at least three nested widths
    assert defaults.WS_WINDOW_TOKENS > 0
    assert defaults.RAM_WINDOW_FACTOR >= 1
    assert 0 < defaults.THRASH_MISS_RATE < 1
    assert defaults.VRAM_SAMPLE_INTERVAL_MS > 0
    assert 0 <= defaults.VRAM_MARGIN_FRACTION < 1
    assert defaults.WIDEN_HOLD_SECONDS > 0
    assert 0 < defaults.FALLBACK_VRAM_BUDGET_FRACTION <= 1
    assert 0 < defaults.FALLBACK_RAM_BUDGET_FRACTION <= 1


def test_no_threshold_is_invented_before_the_baseline_report():
    # design.md: D_bal comes from the Baseline Report, so it must stay unset until then.
    assert defaults.BALANCED_DEADLINE_MS is None


def test_design_md_states_the_same_initial_values():
    text = DESIGN.read_text(encoding="utf-8")
    assert f"W = {defaults.WS_WINDOW_TOKENS}" in text
    assert f"{defaults.VRAM_SAMPLE_INTERVAL_MS} ms" in text
    assert f"T_hold = {defaults.WIDEN_HOLD_SECONDS:g} s" in text
    assert f"h = {round(defaults.WIDEN_HEADROOM * 100)}%" in text
