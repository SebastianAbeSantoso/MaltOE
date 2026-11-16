"""HumanEval scoring: pass@1 by running the problem's tests.

SECURITY: this runs model-written Python on the local machine. It uses a temporary directory, an
isolated interpreter (-I), a stripped environment, and a timeout. It is NOT a sandbox: a response
can still read files or use the network. Run it only on a machine where that is acceptable.
"""

from __future__ import annotations

import os
import re
import secrets
import subprocess
import sys
import tempfile
from pathlib import Path

DEFAULT_TIMEOUT_S = 20

# Printed after the problem's own checks. A response that exits early (for example by calling
# exit(0) at the top level) ends with return code 0 without ever running them, so the return code
# alone cannot be trusted. The sentinel is only printed when check() has run to the end, and each
# run adds a random suffix so a response cannot print it itself.
PASS_SENTINEL = "MOLT-HUMANEVAL-CHECK-PASSED"


def extract_code(response: str) -> str:
    """The code to run: the last fenced block that defines a function, else any last block, else the text."""
    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", response, flags=re.S)
    with_def = [b for b in blocks if "def " in b]
    return (with_def or blocks or [response])[-1]


def program_for(problem: dict, response: str, sentinel: str = PASS_SENTINEL) -> str | None:
    """The full program to run, or None when the response does not define the required function."""
    code = extract_code(response)
    if f"def {problem['entry_point']}" not in code:
        return None
    # The problem prompt may import helpers (for example typing.List) that the reply leaves out.
    header = "\n".join(
        line for line in problem["prompt"].splitlines() if line.startswith(("import ", "from "))
    )
    return (
        f"{header}\n{code}\n\n{problem['test']}\n\n"
        f"check({problem['entry_point']})\nprint({sentinel!r})\n"
    )


def passes(problem: dict, response: str, timeout_s: int = DEFAULT_TIMEOUT_S) -> bool:
    sentinel = f"{PASS_SENTINEL}-{secrets.token_hex(8)}"
    program = program_for(problem, response, sentinel)
    if program is None:
        return False
    env = {k: v for k, v in os.environ.items() if k in ("PATH", "SYSTEMROOT", "TEMP", "TMP")}
    env["PYTHONHASHSEED"] = "0"
    with tempfile.TemporaryDirectory() as d:
        script = Path(d) / "program.py"
        script.write_text(program, encoding="utf-8")
        try:
            done = subprocess.run(
                [sys.executable, "-I", str(script)],
                cwd=d,
                env=env,
                capture_output=True,
                timeout=timeout_s,
            )
        except subprocess.TimeoutExpired:
            return False
    return done.returncode == 0 and sentinel in done.stdout.decode("utf-8", errors="replace")
