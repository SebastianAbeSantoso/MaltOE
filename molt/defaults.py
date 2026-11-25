"""Every tunable that design.md marks as an "initial value", in one place.

These are starting points chosen before any measurement exists. They are not results. Reset them
from the Baseline Report (Task 4) and the hardware runs, and say so in the commit that does it.
Nothing outside this module should hard-code one of these numbers.

Values that design.md says must come from the Baseline Report are None until that report exists.
"""

from typing import Final

# --- package layout (design.md, "Chunked expert layout") ---
# Three chunks give widths of 33%, 67%, and 100%. Four chunks are a fallback decided at gate G3
# and only work with 32-element block types such as Q4_0 or Q8_0.
CHUNKS: Final[int] = 3

# --- working set and residency (design.md, "Expert Pager") ---
WS_WINDOW_TOKENS: Final[int] = 32  # W, tokens in the working-set window
RAM_WINDOW_FACTOR: Final[int] = 8  # the RAM tier uses a window of RAM_WINDOW_FACTOR * W
THRASH_MISS_RATE: Final[float] = 0.25  # miss rate over W tokens that counts as thrashing

# --- sampling and monitoring (design.md, "Hardware Profiler", "Runtime Monitor") ---
VRAM_SAMPLE_INTERVAL_MS: Final[int] = 250  # at least this often, and at every Safe Boundary
MONITOR_FRAME_MS: Final[int] = 250  # telemetry frame length for controller and dashboard

# --- joint controller (design.md, "Joint Controller") ---
VRAM_MARGIN_FRACTION: Final[float] = 0.05  # safety margin, share of the budget
WIDEN_HEADROOM: Final[float] = 0.10  # h: extra free VRAM required before widening
WIDEN_HOLD_SECONDS: Final[float] = 5.0  # T_hold: headroom must persist this long
BALANCED_DEADLINE_MS: Final[float | None] = None  # D_bal: set from the Baseline Report

# --- profiler fallbacks when a value cannot be measured (design.md, "Hardware Profiler") ---
FALLBACK_VRAM_BUDGET_FRACTION: Final[float] = 0.5  # of total VRAM, shown as "assumed"
FALLBACK_RAM_BUDGET_FRACTION: Final[float] = 0.5  # of RAM available after the OS

# --- server ---
# Prompts never leave the machine, so the server only listens on the loopback interface.
SERVER_HOST: Final[str] = "127.0.0.1"
SERVER_PORT: Final[int] = 8000
