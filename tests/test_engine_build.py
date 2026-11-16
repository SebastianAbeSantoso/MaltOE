"""Tests for engine/build.py. They never touch the network or build anything."""

import importlib.util
from pathlib import Path

import pytest

ENGINE = Path(__file__).resolve().parent.parent / "engine"


@pytest.fixture(scope="module")
def build():
    spec = importlib.util.spec_from_file_location("engine_build", ENGINE / "build.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_pin_is_a_full_commit_hash(build):
    sha = build.read_pin()
    assert len(sha) == 40 and set(sha) <= set("0123456789abcdef")


def test_pin_matches_the_feasibility_report(build):
    report = (ENGINE.parent / "docs" / "feasibility-report.md").read_text(encoding="utf-8")
    assert build.read_pin() in report


@pytest.mark.parametrize(
    "bad", ["", "a946eb54", "A946EB541A6D3051496BDCC451A0FA85772B8DFD", "z" * 40]
)
def test_bad_pin_is_rejected(build, tmp_path, bad):
    f = tmp_path / "pin"
    f.write_text(bad + "\n", encoding="utf-8")
    with pytest.raises(ValueError):
        build.read_pin(f)


def test_patches_are_applied_in_name_order(build, tmp_path):
    for name in ("0002-b.patch", "0001-a.patch", "0010-c.patch", "notes.txt"):
        (tmp_path / name).write_text("x", encoding="utf-8")
    assert [p.name for p in build.patch_files(tmp_path)] == [
        "0001-a.patch",
        "0002-b.patch",
        "0010-c.patch",
    ]


def test_cuda_configure_command(build):
    cmd = build.configure_command("cuda", Path("src"), Path("out"), "89", "Ninja")
    assert "-DGGML_CUDA=ON" in cmd and "-DCMAKE_CUDA_ARCHITECTURES=89" in cmd
    assert cmd[cmd.index("-G") + 1] == "Ninja"
    assert "-DLLAMA_OPENSSL=OFF" in cmd and "-DLLAMA_USE_PREBUILT_UI=OFF" in cmd


def test_cpu_configure_command_has_no_gpu_flags(build):
    cmd = build.configure_command("cpu", Path("src"), Path("out"), "89", None)
    assert not any("CUDA" in c or "VULKAN" in c for c in cmd)
    assert "-G" not in cmd


def test_build_command_lists_targets(build):
    cmd = build.build_command(Path("out"), ["llama-bench", "llama-server"], 4)
    assert cmd[cmd.index("--target") + 1 :] == ["llama-bench", "llama-server"]
    assert cmd[cmd.index("-j") + 1] == "4"


def test_destructive_git_commands_are_confined_to_the_cache(build, tmp_path):
    build.require_inside(build.SRC_DIR, build.CACHE_DIR)
    with pytest.raises(RuntimeError):
        build.require_inside(tmp_path / "somewhere" / "else", build.CACHE_DIR)
    with pytest.raises(RuntimeError):
        build.require_inside(ENGINE.parent, build.CACHE_DIR)  # the repository itself


def test_dry_run_prints_the_plan_and_changes_nothing(build, capsys):
    existed_before = (ENGINE / ".cache").exists()
    assert build.main(["--backend", "cuda", "--cuda-arch", "89", "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert build.read_pin() in out
    assert "git -C" in out and "fetch --depth 1 origin" in out
    assert "cmake --build" in out and "llama-server" in out
    assert (ENGINE / ".cache").exists() == existed_before  # a dry run creates nothing
