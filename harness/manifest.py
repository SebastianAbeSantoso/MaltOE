"""The machine manifest recorded with every result (Req 2.7).

Results depend on the machine: `--n-cpu-moe` on a laptop CPU is not the same measurement as on a
desktop CPU, a laptop GPU has a lower power limit than the desktop card, and a RAM figure that
ignores what the OS already uses is misleading. So every run stores:

    GPU name, power limit and driver version, CPU model and thread count, power mode,
    total RAM and RAM available after OS overhead, SSD model and interface, OS version,
    and the engine version.

Anything that cannot be read is left out of its field and listed under "unavailable" with the
reason, so a missing value is visible instead of silently absent.

    uv run python -m harness.manifest --engine-version "llama.cpp a946eb54" --ssd-path D:\\models
"""

from __future__ import annotations

import argparse
import json
import platform
import re
import shutil
import subprocess
import sys
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import psutil

SCHEMA_VERSION = 1
REPO_ROOT = Path(__file__).resolve().parent.parent

# A runner takes a command and returns its standard output, or None when it is missing or fails.
Runner = Callable[[list[str]], str | None]


def run_command(cmd: list[str], timeout: float = 20) -> str | None:
    try:
        done = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return done.stdout if done.returncode == 0 else None


def powershell(script: str) -> list[str]:
    return ["powershell", "-NoProfile", "-NonInteractive", "-Command", script]


def ps_quote(value: str) -> str:
    """A PowerShell single-quoted string literal."""
    return "'" + value.replace("'", "''") + "'"


# ----- parsers (pure, tested against canned output) -----


def _num(text: str) -> float | None:
    text = text.strip()
    if not text or text.startswith("[") or text.upper() == "N/A":
        return None
    try:
        return float(text)
    except ValueError:
        return None


def parse_nvidia_smi(csv_text: str) -> list[dict]:
    """Rows of: name, driver_version, memory.total [MiB], power.limit [W], power.default_limit [W]."""
    gpus = []
    for line in csv_text.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 5:
            continue
        gpus.append(
            {
                "name": parts[0],
                "driver_version": parts[1],
                "memory_total_mib": _num(parts[2]),
                "power_limit_w": _num(parts[3]),
                "power_default_limit_w": _num(parts[4]),
            }
        )
    return gpus


def parse_power_scheme(text: str) -> str | None:
    """'Power Scheme GUID: 381b4222-... (Balanced)' -> 'Balanced'."""
    m = re.search(r"\(([^()]+)\)\s*$", text.strip().splitlines()[0]) if text.strip() else None
    return m.group(1) if m else None


def parse_disk_json(text: str) -> dict | None:
    try:
        d = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(d, dict) or not d.get("model"):
        return None
    return {"model": d["model"], "interface": d.get("bus"), "media_type": d.get("media")}


def disk_script(path: str) -> str:
    return (
        f"$d = (Get-Item -LiteralPath {ps_quote(path)}).PSDrive.Name; "
        "$disk = Get-Partition -DriveLetter $d | Get-Disk; "
        "$pd = Get-PhysicalDisk | Where-Object { $_.DeviceId -eq $disk.Number } | Select-Object -First 1; "
        "[pscustomobject]@{ model = $disk.FriendlyName; bus = [string]$disk.BusType; "
        "media = [string]$pd.MediaType } | ConvertTo-Json -Compress"
    )


# ----- collection -----


def ram_info() -> dict:
    vm = psutil.virtual_memory()
    return {"total_bytes": int(vm.total), "available_bytes": int(vm.available)}


def _cpu_model(system: str, run: Runner) -> str | None:
    if system == "Windows":
        out = run(powershell("(Get-CimInstance Win32_Processor | Select-Object -First 1).Name"))
        return out.strip() or None if out else None
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor() or None


def collect_manifest(
    *,
    engine_version: str | None = None,
    ssd_path: str | Path | None = None,
    model_path: str | Path | None = None,
    run: Runner = run_command,
    ram_fn: Callable[[], dict] = ram_info,
    system: str | None = None,
    now: datetime | None = None,
    repo_root: Path = REPO_ROOT,
) -> dict:
    system = system or platform.system()
    missing: list[dict] = []

    def need(field: str, value, reason: str):
        if value is None or value == [] or value == "":
            missing.append({"field": field, "reason": reason})
        return value

    # GPU
    smi = run(
        [
            "nvidia-smi",
            "--query-gpu=name,driver_version,memory.total,power.limit,power.default_limit",
            "--format=csv,noheader,nounits",
        ]
    )
    gpus = parse_nvidia_smi(smi) if smi else []
    need("gpu", gpus, "nvidia-smi not found or failed (AMD, or no NVIDIA driver)")

    # CPU
    cpu = {
        "model": need("cpu.model", _cpu_model(system, run), "could not read the CPU model"),
        "physical_cores": psutil.cpu_count(logical=False),
        "logical_threads": psutil.cpu_count(logical=True),
    }
    power_mode = None
    if system == "Windows":
        out = run(["powercfg", "/getactivescheme"])
        power_mode = parse_power_scheme(out) if out else None
    need("cpu.power_mode", power_mode, "only read on Windows through powercfg")
    cpu["power_mode"] = power_mode

    # Storage (the drive that holds the model package)
    path = str(ssd_path) if ssd_path is not None else None
    disk: dict | None = None
    free_bytes = None
    if path:
        try:
            free_bytes = shutil.disk_usage(path).free
        except OSError:
            missing.append(
                {"field": "storage.free_bytes", "reason": f"cannot read free space of {path}"}
            )
        if system == "Windows":
            out = run(powershell(disk_script(path)))
            disk = parse_disk_json(out) if out else None
        need("storage.model", disk, "SSD model and interface are only read on Windows")
    else:
        missing.append({"field": "storage", "reason": "no --ssd-path given"})

    engine_commit = None
    pin = repo_root / "engine" / "LLAMA_CPP_COMMIT"
    if pin.exists():
        engine_commit = pin.read_text(encoding="utf-8").strip()
    need("engine.version", engine_version, "pass --engine-version")

    model = None
    if model_path is not None:
        mp = Path(model_path)
        if mp.exists():
            st = mp.stat()
            model = {
                "path": str(mp),
                "size_bytes": st.st_size,
                "modified": datetime.fromtimestamp(st.st_mtime, UTC).isoformat(),
            }
        else:
            missing.append({"field": "model", "reason": f"{mp} does not exist"})

    return {
        "schema_version": SCHEMA_VERSION,
        "collected_at": (now or datetime.now(UTC)).isoformat(),
        "os": {
            "system": system,
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
        },
        "python": platform.python_version(),
        "gpus": gpus,
        "cpu": cpu,
        # "available" is what the OS reports as usable right now, after its own overhead
        "ram": ram_fn(),
        "storage": {"path": path, "free_bytes": free_bytes, "device": disk},
        "engine": {"version": engine_version, "llama_cpp_commit": engine_commit},
        "model": model,
        "unavailable": missing,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--engine-version")
    ap.add_argument("--ssd-path", help="a path on the drive that holds the model package")
    ap.add_argument("--model", help="the model file, to record its size")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args(argv)
    manifest = collect_manifest(
        engine_version=args.engine_version, ssd_path=args.ssd_path, model_path=args.model
    )
    text = json.dumps(manifest, indent=2)
    if args.out:
        args.out.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
