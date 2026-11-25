#!/usr/bin/env python3
"""Fetch llama.cpp at the pinned commit, apply the Molt patch series, and build it.

    python engine/build.py --backend cpu
    python engine/build.py --backend cuda --cuda-arch 89
    python engine/build.py --backend cpu --dry-run          # print the commands only

The checkout lives in engine/.cache/llama.cpp and is owned by this script: before every build it is
reset to the pinned commit and cleaned, so local edits there are discarded. Edit patches in
engine/patches/ and code in engine/molt/ instead. Output goes to engine/build/<backend>/.

Only the Python standard library is used, so this runs before any environment is set up.
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path

ENGINE_DIR = Path(__file__).resolve().parent
PIN_FILE = ENGINE_DIR / "LLAMA_CPP_COMMIT"
PATCH_DIR = ENGINE_DIR / "patches"
CACHE_DIR = ENGINE_DIR / ".cache"
SRC_DIR = CACHE_DIR / "llama.cpp"
BUILD_ROOT = ENGINE_DIR / "build"
REPO_URL = "https://github.com/ggml-org/llama.cpp.git"

BACKEND_FLAGS: dict[str, list[str]] = {
    "cpu": [],
    "cuda": ["-DGGML_CUDA=ON"],
    "vulkan": ["-DGGML_VULKAN=ON"],
}

# What Molt's scripts and baselines use. Building everything would take much longer.
DEFAULT_TARGETS = [
    "llama-server",
    "llama-bench",
    "llama-perplexity",
    "llama-imatrix",
    "llama-quantize",
    "llama-completion",
]

# Options that avoid needing things Molt does not use: OpenSSL (HTTPS model download) and the
# prebuilt web UI that the llama.cpp server would otherwise fetch from the network at build time.
COMMON_FLAGS = [
    "-DCMAKE_BUILD_TYPE=Release",
    "-DLLAMA_OPENSSL=OFF",
    "-DLLAMA_USE_PREBUILT_UI=OFF",
    "-DLLAMA_BUILD_TESTS=OFF",
    "-DLLAMA_BUILD_EXAMPLES=OFF",
]

SHA_RE = re.compile(r"^[0-9a-f]{40}$")


def read_pin(path: Path = PIN_FILE) -> str:
    sha = path.read_text(encoding="utf-8").strip()
    if not SHA_RE.fullmatch(sha):
        raise ValueError(
            f"{path} must contain one full 40-character lowercase commit hash, got {sha!r}"
        )
    return sha


def patch_files(patch_dir: Path = PATCH_DIR) -> list[Path]:
    return sorted(patch_dir.glob("*.patch"), key=lambda p: p.name)


def require_inside(path: Path, parent: Path) -> None:
    """Refuse to run destructive git commands anywhere but in our own cache directory."""
    if parent.resolve() not in path.resolve().parents:
        raise RuntimeError(f"refusing to modify {path}: not inside {parent}")


def configure_command(
    backend: str, src: Path, build: Path, cuda_arch: str | None, generator: str | None
) -> list[str]:
    cmd = ["cmake", "-S", str(src), "-B", str(build)]
    if generator:
        cmd += ["-G", generator]
    cmd += COMMON_FLAGS + BACKEND_FLAGS[backend]
    if backend == "cuda" and cuda_arch:
        cmd.append(f"-DCMAKE_CUDA_ARCHITECTURES={cuda_arch}")
    return cmd


def build_command(build: Path, targets: list[str], jobs: int | None) -> list[str]:
    cmd = ["cmake", "--build", str(build), "--config", "Release"]
    if jobs:
        cmd += ["-j", str(jobs)]
    return cmd + ["--target", *targets]


class Runner:
    def __init__(self, dry_run: bool) -> None:
        self.dry_run = dry_run

    def run(self, cmd: list[str], cwd: Path | None = None, capture: bool = False) -> str:
        print("+", " ".join(cmd), flush=True)
        if self.dry_run:
            return ""
        done = subprocess.run(cmd, cwd=cwd, check=True, text=True, capture_output=capture)
        return done.stdout.strip() if capture else ""


def prepare_source(r: Runner, sha: str) -> None:
    """Make engine/.cache/llama.cpp an exact, clean checkout of the pinned commit."""
    require_inside(SRC_DIR, CACHE_DIR)
    if not (SRC_DIR / ".git").exists():
        if not r.dry_run:
            SRC_DIR.mkdir(parents=True, exist_ok=True)
        r.run(["git", "init", "-q", str(SRC_DIR)])
        r.run(["git", "-C", str(SRC_DIR), "remote", "add", "origin", REPO_URL])
    have = ""
    if not r.dry_run:
        try:
            have = subprocess.run(
                ["git", "-C", str(SRC_DIR), "rev-parse", "--verify", f"{sha}^{{commit}}"],
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
        except subprocess.CalledProcessError:
            have = ""
    if have != sha:
        r.run(["git", "-C", str(SRC_DIR), "fetch", "--depth", "1", "origin", sha])
    r.run(["git", "-C", str(SRC_DIR), "checkout", "--force", "--detach", sha])
    r.run(["git", "-C", str(SRC_DIR), "reset", "--hard", sha])
    r.run(["git", "-C", str(SRC_DIR), "clean", "-fdxq"])
    if not r.dry_run:
        head = subprocess.run(
            ["git", "-C", str(SRC_DIR), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        if head != sha:
            raise RuntimeError(f"checkout is at {head}, expected pinned commit {sha}")


def apply_patches(r: Runner) -> list[Path]:
    patches = patch_files()
    for p in patches:
        r.run(["git", "-C", str(SRC_DIR), "apply", "--check", "--whitespace=nowarn", str(p)])
        r.run(["git", "-C", str(SRC_DIR), "apply", "--whitespace=nowarn", str(p)])
    print(f"applied {len(patches)} patch(es)", flush=True)
    return patches


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--backend", choices=sorted(BACKEND_FLAGS), default="cpu")
    ap.add_argument("--targets", nargs="+", default=DEFAULT_TARGETS, help="CMake targets to build")
    ap.add_argument("--cuda-arch", help="CMAKE_CUDA_ARCHITECTURES, for example 89 for an RTX 4060")
    ap.add_argument("--jobs", "-j", type=int, help="parallel build jobs")
    ap.add_argument(
        "--no-fetch",
        action="store_true",
        help="skip fetching, patching, and cleaning; build what is there",
    )
    ap.add_argument(
        "--dry-run", action="store_true", help="print the commands without running them"
    )
    args = ap.parse_args(argv)

    for tool in ("git", "cmake"):
        if not args.dry_run and shutil.which(tool) is None:
            print(f"error: {tool} not found on PATH", file=sys.stderr)
            return 2
    sha = read_pin()
    r = Runner(args.dry_run)
    build = BUILD_ROOT / args.backend
    generator = "Ninja" if shutil.which("ninja") else None

    print(f"llama.cpp pinned commit: {sha}", flush=True)
    if not args.no_fetch:
        prepare_source(r, sha)
        apply_patches(r)
    r.run(configure_command(args.backend, SRC_DIR, build, args.cuda_arch, generator))
    r.run(build_command(build, args.targets, args.jobs))
    print(f"done: binaries are under {build}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
