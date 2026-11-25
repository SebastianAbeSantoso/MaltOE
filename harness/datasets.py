"""The frozen evaluation set (Req 10.4, 10.7): where the data comes from, how problems were drawn,
and how a later run proves it is using the same problems.

Run from the repository root:

    uv run python -m harness.datasets fetch            # download the three sources to the cache
    uv run python -m harness.datasets verify           # check the cache against the frozen files
    uv run python -m harness.datasets freeze           # rewrite harness/evalset/*.json (a deliberate act)

The problems are drawn once, with a fixed seed, before any Molt result exists. The ids are stored
in ``harness/evalset/``. Each id also stores a hash of the problem text, and each source file a hash
of the whole file, so a changed upstream dataset is detected instead of silently used.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import random
import sys
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

EVALSET_DIR = Path(__file__).resolve().parent / "evalset"
DEFAULT_CACHE = Path("harness-cache")

# One seed for every draw. It is the date Task 1 started; it is not tuned.
SEED = 20261116
DRAW_SIZE = 20

HUMANEVAL_URL = "https://github.com/openai/human-eval/raw/master/data/HumanEval.jsonl.gz"
GSM8K_URL = (
    "https://raw.githubusercontent.com/openai/grade-school-math/master/"
    "grade_school_math/data/test.jsonl"
)
WIKITEXT_URL = "https://huggingface.co/datasets/ggml-org/ci/resolve/main/wikitext-2-raw-v1.zip"

# Perplexity: the first N windows of 512 tokens of the wikitext-2 test split (design.md).
PPL_CONTEXT = 512
PPL_CHUNKS = 64


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def problem_hash(problem: dict) -> str:
    """Hash of one problem, independent of key order and whitespace."""
    canon = json.dumps(problem, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return sha256_bytes(canon.encode("utf-8"))


def draw_indices(
    population: int, size: int = DRAW_SIZE, *, salt: str, seed: int = SEED
) -> list[int]:
    """The seeded draw. ``salt`` keeps the coding and reasoning draws independent.

    The same algorithm is used by spikes/width_emulation/smoke_eval.py, so the 5 + 5 problems of the
    Task 1 smoke test are the first 5 of each frozen list. tests/test_evalset.py checks that.
    """
    return random.Random(f"{seed}{salt}").sample(range(population), size)


# ----- fetching -----


def fetch(url: str, cache_dir: Path) -> Path:
    """Download ``url`` into ``cache_dir`` once and return the file path."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    dest = cache_dir / url.rsplit("/", 1)[-1]
    if not dest.exists():
        with urllib.request.urlopen(url, timeout=120) as r:
            dest.write_bytes(r.read())
    return dest


def load_humaneval(cache_dir: Path = DEFAULT_CACHE) -> tuple[list[dict], Path]:
    path = fetch(HUMANEVAL_URL, cache_dir)
    lines = gzip.decompress(path.read_bytes()).decode("utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()], path


def load_gsm8k(cache_dir: Path = DEFAULT_CACHE) -> tuple[list[dict], Path]:
    path = fetch(GSM8K_URL, cache_dir)
    lines = path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()], path


def load_wikitext_test(cache_dir: Path = DEFAULT_CACHE) -> tuple[Path, Path]:
    """Return (path of wiki.test.raw, path of the downloaded zip)."""
    zpath = fetch(WIKITEXT_URL, cache_dir)
    out = cache_dir / "wiki.test.raw"
    if not out.exists():
        with zipfile.ZipFile(io.BytesIO(zpath.read_bytes())) as z:
            names = {Path(n).name: n for n in z.namelist()}
            out.write_bytes(z.read(names["wiki.test.raw"]))
    return out, zpath


# ----- the frozen files -----


@dataclass(frozen=True)
class FrozenSet:
    """A frozen list of problems for one task kind."""

    name: str
    kind: str  # "coding" or "reasoning"
    source_url: str
    source_sha256: str
    population: int
    problems: list[dict]  # each: index, id, sha256

    @property
    def indices(self) -> list[int]:
        return [p["index"] for p in self.problems]


def load_frozen(kind: str, directory: Path = EVALSET_DIR) -> FrozenSet:
    raw = json.loads(
        (directory / {"coding": "coding.json", "reasoning": "reasoning.json"}[kind]).read_text(
            encoding="utf-8"
        )
    )
    return FrozenSet(
        name=raw["name"],
        kind=kind,
        source_url=raw["source"]["url"],
        source_sha256=raw["source"]["sha256"],
        population=raw["draw"]["population"],
        problems=raw["problems"],
    )


def select(frozen: FrozenSet, dataset: list[dict]) -> list[dict]:
    """The frozen problems, in frozen order, taken from a loaded dataset."""
    return [dataset[i] for i in frozen.indices]


def verify(frozen: FrozenSet, dataset: list[dict], source_file: Path) -> list[str]:
    """Return a list of problems; an empty list means the cache matches the frozen set."""
    issues: list[str] = []
    if len(dataset) != frozen.population:
        issues.append(
            f"{frozen.name}: dataset has {len(dataset)} problems, frozen set expects {frozen.population}"
        )
        return issues
    got = sha256_file(source_file)
    if got != frozen.source_sha256:
        issues.append(
            f"{frozen.name}: source file hash {got} differs from frozen {frozen.source_sha256}"
        )
    for p in frozen.problems:
        h = problem_hash(dataset[p["index"]])
        if h != p["sha256"]:
            issues.append(
                f"{frozen.name}: problem {p['id']} (index {p['index']}) changed since it was frozen"
            )
    return issues


def _problem_id(kind: str, index: int, problem: dict) -> str:
    return problem["task_id"] if kind == "coding" else f"gsm8k-test-{index}"


def build_frozen(kind: str, cache_dir: Path) -> dict:
    """Build the JSON for one frozen set from the downloaded data. Used by ``freeze`` and by a test."""
    if kind == "coding":
        dataset, src = load_humaneval(cache_dir)
        name, url, salt, licence = "humaneval-20", HUMANEVAL_URL, "humaneval", "MIT"
        credit = "OpenAI HumanEval, https://github.com/openai/human-eval"
    else:
        dataset, src = load_gsm8k(cache_dir)
        name, url, salt, licence = "gsm8k-test-20", GSM8K_URL, "gsm8k", "MIT"
        credit = "OpenAI GSM8K, https://github.com/openai/grade-school-math"
    idx = draw_indices(len(dataset), salt=salt)
    return {
        "name": name,
        "source": {"url": url, "sha256": sha256_file(src), "licence": licence, "credit": credit},
        "draw": {"seed": SEED, "salt": salt, "size": DRAW_SIZE, "population": len(dataset)},
        "problems": [
            {"index": i, "id": _problem_id(kind, i, dataset[i]), "sha256": problem_hash(dataset[i])}
            for i in idx
        ],
    }


def build_perplexity(cache_dir: Path) -> dict:
    raw, zpath = load_wikitext_test(cache_dir)
    return {
        "name": "wikitext-2-test-first-64x512",
        "source": {
            "url": WIKITEXT_URL,
            "zip_sha256": sha256_file(zpath),
            "file": "wiki.test.raw",
            "file_sha256": sha256_file(raw),
            "licence": "CC BY-SA 3.0 (original WikiText release); the ggml-org/ci mirror carries no licence metadata",
            "credit": "Merity et al., WikiText-2 (raw), mirrored at huggingface.co/datasets/ggml-org/ci",
        },
        "context_tokens": PPL_CONTEXT,
        "chunks": PPL_CHUNKS,
        "llama_perplexity_args": ["-c", str(PPL_CONTEXT), "--chunks", str(PPL_CHUNKS)],
    }


def _write(path: Path, obj: dict) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def cmd_fetch(cache: Path) -> int:
    load_humaneval(cache)
    load_gsm8k(cache)
    load_wikitext_test(cache)
    print(f"downloaded to {cache.resolve()}")
    return 0


def cmd_verify(cache: Path) -> int:
    issues: list[str] = []
    for kind, loader in (("coding", load_humaneval), ("reasoning", load_gsm8k)):
        dataset, src = loader(cache)
        issues += verify(load_frozen(kind), dataset, src)
    ppl = json.loads((EVALSET_DIR / "perplexity.json").read_text(encoding="utf-8"))
    raw, zpath = load_wikitext_test(cache)
    if sha256_file(raw) != ppl["source"]["file_sha256"]:
        issues.append("wikitext-2 test file differs from the frozen hash")
    if sha256_file(zpath) != ppl["source"]["zip_sha256"]:
        issues.append("wikitext-2 zip differs from the frozen hash")
    for i in issues:
        print("MISMATCH:", i)
    print(
        "evaluation set matches the frozen files" if not issues else f"{len(issues)} mismatch(es)"
    )
    return 1 if issues else 0


def cmd_freeze(cache: Path) -> int:
    """Rewrite the frozen files. Do this only before any result exists, and say so in the commit."""
    EVALSET_DIR.mkdir(parents=True, exist_ok=True)
    _write(EVALSET_DIR / "coding.json", build_frozen("coding", cache))
    _write(EVALSET_DIR / "reasoning.json", build_frozen("reasoning", cache))
    _write(EVALSET_DIR / "perplexity.json", build_perplexity(cache))
    print(f"wrote {EVALSET_DIR}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("command", choices=["fetch", "verify", "freeze"])
    ap.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    args = ap.parse_args(argv)
    return {"fetch": cmd_fetch, "verify": cmd_verify, "freeze": cmd_freeze}[args.command](
        args.cache
    )


if __name__ == "__main__":
    sys.exit(main())
