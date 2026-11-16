"""The machine manifest (Req 2.7), against canned command output."""

import json
from datetime import UTC, datetime
from pathlib import Path

from harness import manifest as m

NVIDIA_SMI = "NVIDIA GeForce RTX 4060 Laptop GPU, 560.94, 8188, 80.00, 80.00\n"
POWERCFG = "Power Scheme GUID: 381b4222-f694-41f0-9685-ff5bb260df2e  (Balanced)\n"
DISK_JSON = json.dumps({"model": "Samsung SSD 980 PRO 1TB", "bus": "NVMe", "media": "SSD"})
RAM = {"total_bytes": 16 * 2**30, "available_bytes": 9 * 2**30}
FIXED_NOW = datetime(2026, 11, 16, 12, 0, tzinfo=UTC)


def windows_runner(cmd: list[str]) -> str | None:
    if cmd[0] == "nvidia-smi":
        return NVIDIA_SMI
    if cmd[0] == "powercfg":
        return POWERCFG
    if cmd[0] == "powershell":
        script = cmd[-1]
        if "Win32_Processor" in script:
            return "13th Gen Intel(R) Core(TM) i9-14980HX\n"
        if "Get-PhysicalDisk" in script:
            return DISK_JSON
    return None


def collect(tmp_path: Path, runner=windows_runner, **kw):
    return m.collect_manifest(
        engine_version="llama.cpp a946eb54",
        ssd_path=tmp_path,
        run=runner,
        ram_fn=lambda: RAM,
        system="Windows",
        now=FIXED_NOW,
        **kw,
    )


# ----- parsers -----


def test_parse_nvidia_smi():
    [gpu] = m.parse_nvidia_smi(NVIDIA_SMI)
    assert gpu == {
        "name": "NVIDIA GeForce RTX 4060 Laptop GPU",
        "driver_version": "560.94",
        "memory_total_mib": 8188.0,
        "power_limit_w": 80.0,
        "power_default_limit_w": 80.0,
    }


def test_parse_nvidia_smi_tolerates_unsupported_power_queries():
    [gpu] = m.parse_nvidia_smi("NVIDIA GeForce RTX 4060, 551.23, 8188, [N/A], [Not Supported]\n")
    assert gpu["power_limit_w"] is None and gpu["power_default_limit_w"] is None
    assert m.parse_nvidia_smi("") == [] and m.parse_nvidia_smi("garbage") == []


def test_parse_power_scheme():
    assert m.parse_power_scheme(POWERCFG) == "Balanced"
    assert (
        m.parse_power_scheme(
            "Power Scheme GUID: 8c5e7fda-e8bf-4a96-9a85-a6e23a8c635c  (High performance)"
        )
        == "High performance"
    )
    assert (
        m.parse_power_scheme("Skema Daya GUID: abc  (Seimbang)") == "Seimbang"
    )  # localized names still parse
    assert m.parse_power_scheme("") is None and m.parse_power_scheme("no parentheses") is None


def test_parse_disk_json():
    assert m.parse_disk_json(DISK_JSON) == {
        "model": "Samsung SSD 980 PRO 1TB",
        "interface": "NVMe",
        "media_type": "SSD",
    }
    assert m.parse_disk_json("not json") is None
    assert m.parse_disk_json("[]") is None
    assert m.parse_disk_json(json.dumps({"model": "", "bus": "NVMe"})) is None


def test_powershell_quoting_survives_an_apostrophe():
    assert m.ps_quote(r"D:\models") == r"'D:\models'"
    assert m.ps_quote("D:\\it's here") == "'D:\\it''s here'"
    assert "'D:\\it''s'" in m.disk_script("D:\\it's")


# ----- collection -----


def test_manifest_has_everything_req_2_7_asks_for(tmp_path):
    out = collect(tmp_path)
    gpu = out["gpus"][0]
    assert gpu["name"] and gpu["power_limit_w"] == 80.0 and gpu["driver_version"] == "560.94"
    assert out["cpu"]["model"].startswith("13th Gen Intel") and out["cpu"]["logical_threads"] >= 1
    assert out["cpu"]["power_mode"] == "Balanced"
    assert out["ram"] == RAM and out["ram"]["available_bytes"] < out["ram"]["total_bytes"]
    assert out["storage"]["device"] == {
        "model": "Samsung SSD 980 PRO 1TB",
        "interface": "NVMe",
        "media_type": "SSD",
    }
    assert out["storage"]["free_bytes"] > 0
    assert out["os"]["system"] == "Windows" and out["os"]["version"]
    assert out["engine"]["version"] == "llama.cpp a946eb54"
    assert out["unavailable"] == []


def test_engine_commit_comes_from_the_pin_file(tmp_path):
    pin = Path(__file__).resolve().parent.parent / "engine" / "LLAMA_CPP_COMMIT"
    assert (
        collect(tmp_path)["engine"]["llama_cpp_commit"] == pin.read_text(encoding="utf-8").strip()
    )


def test_missing_values_are_listed_not_silently_dropped(tmp_path):
    out = collect(tmp_path, runner=lambda cmd: None)
    fields = {u["field"] for u in out["unavailable"]}
    assert {"gpu", "cpu.model", "cpu.power_mode", "storage.model"} <= fields
    assert out["gpus"] == [] and out["storage"]["device"] is None
    assert all(u["reason"] for u in out["unavailable"])


def test_a_missing_engine_version_is_reported(tmp_path):
    out = m.collect_manifest(
        ssd_path=tmp_path, run=windows_runner, ram_fn=lambda: RAM, system="Windows"
    )
    assert "engine.version" in {u["field"] for u in out["unavailable"]}


def test_model_file_size_is_recorded(tmp_path):
    model = tmp_path / "model.gguf"
    model.write_bytes(b"x" * 1234)
    assert collect(tmp_path, model_path=model)["model"]["size_bytes"] == 1234
    assert "model" in {
        u["field"] for u in collect(tmp_path, model_path=tmp_path / "nope.gguf")["unavailable"]
    }


def test_the_manifest_is_json_serialisable(tmp_path):
    json.dumps(collect(tmp_path))


def test_runs_on_this_machine_without_raising(tmp_path):
    out = m.collect_manifest(ssd_path=tmp_path)  # real runner, real machine, whatever OS
    assert out["ram"]["total_bytes"] > 0 and out["cpu"]["logical_threads"] >= 1
