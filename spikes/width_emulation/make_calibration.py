#!/usr/bin/env python3
"""Build the calibration text for llama-imatrix, and fetch the wikitext-2 test file.

Calibration text = wikitext-2 TRAIN split + permissively licensed source code
(a llama.cpp checkout, MIT), interleaved in blocks so that every imatrix chunk
sees both kinds of text.

The evaluation set uses wikitext-2 TEST, HumanEval and GSM8K. None of those are
in the calibration text, so importance scores are not tuned on the evaluation data.

Usage:
    python make_calibration.py --llama-cpp C:\\src\\llama.cpp --out-dir work

Outputs (in --out-dir):
    calibration.txt     calibration text
    wiki.test.raw       wikitext-2 test split, for llama-perplexity
    calibration.json    sizes and sha256 hashes, for the Feasibility Report
"""
import argparse
import hashlib
import io
import json
import sys
import urllib.request
import zipfile
from pathlib import Path

WIKITEXT_URL = "https://huggingface.co/datasets/ggml-org/ci/resolve/main/wikitext-2-raw-v1.zip"
CODE_SUFFIXES = {".cpp", ".h", ".c", ".cu", ".cuh", ".py", ".hpp"}
CODE_DIRS = ["src", "common", "tools", "gguf-py"]
BLOCK_CHARS = 8000


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def fetch_wikitext(cache: Path) -> tuple[str, str]:
    zpath = cache / "wikitext-2-raw-v1.zip"
    if not zpath.exists():
        print(f"downloading {WIKITEXT_URL}")
        with urllib.request.urlopen(WIKITEXT_URL) as r, open(zpath, "wb") as f:
            f.write(r.read())
    with zipfile.ZipFile(io.BytesIO(zpath.read_bytes())) as z:
        names = {Path(n).name: n for n in z.namelist()}
        train = z.read(names["wiki.train.raw"]).decode("utf-8")
        test = z.read(names["wiki.test.raw"]).decode("utf-8")
    return train, test


def collect_code(llama_cpp: Path, max_chars: int) -> str:
    files: list[Path] = []
    for d in CODE_DIRS:
        root = llama_cpp / d
        if root.is_dir():
            files += [p for p in root.rglob("*") if p.is_file() and p.suffix in CODE_SUFFIXES]
    files.sort(key=lambda p: p.relative_to(llama_cpp).as_posix())
    out, total = [], 0
    for p in files:
        try:
            text = p.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        out.append(f"// file: {p.relative_to(llama_cpp).as_posix()}\n{text}\n")
        total += len(text)
        if total >= max_chars:
            break
    return "".join(out)[:max_chars]


def blocks(text: str, size: int):
    for i in range(0, len(text), size):
        yield text[i:i + size]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--llama-cpp", required=True, type=Path, help="llama.cpp checkout (MIT licensed code corpus)")
    ap.add_argument("--out-dir", default=Path("work"), type=Path)
    ap.add_argument("--wiki-chars", type=int, default=300_000)
    ap.add_argument("--code-chars", type=int, default=300_000)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    if not (args.llama_cpp / "src").is_dir():
        print(f"{args.llama_cpp} does not look like a llama.cpp checkout", file=sys.stderr)
        return 2

    train, test = fetch_wikitext(args.out_dir)
    wiki = train[: args.wiki_chars]
    code = collect_code(args.llama_cpp, args.code_chars)
    if len(code) < args.code_chars // 2:
        print("warning: little code found; check --llama-cpp", file=sys.stderr)

    mixed = []
    for w, c in zip(blocks(wiki, BLOCK_CHARS), blocks(code, BLOCK_CHARS)):
        mixed += [w, c]
    # whatever is left of the longer text goes at the end
    n = min(len(wiki), len(code)) // BLOCK_CHARS * BLOCK_CHARS
    mixed += [wiki[n:], code[n:]]
    calib = "".join(mixed)

    (args.out_dir / "calibration.txt").write_text(calib, encoding="utf-8")
    (args.out_dir / "wiki.test.raw").write_text(test, encoding="utf-8")
    info = {
        "wikitext_source": WIKITEXT_URL,
        "wikitext_train_chars_used": len(wiki),
        "code_chars_used": len(code),
        "calibration_chars": len(calib),
        "calibration_sha256": sha256(calib.encode("utf-8")),
        "wiki_test_sha256": sha256(test.encode("utf-8")),
        "approx_tokens": len(calib) // 4,
        "licences": {
            "wikitext-2": "CC BY-SA 3.0 in the original release; the ggml-org/ci mirror carries no licence metadata (verify)",
            "llama.cpp code": "MIT",
        },
    }
    (args.out_dir / "calibration.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(json.dumps(info, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
