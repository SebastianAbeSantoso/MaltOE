#!/usr/bin/env python3
"""Small, fixed, offline-after-download quality check against an OpenAI-compatible server
(llama-server). Used for the G3 width gate (Task 1.5). The frozen evaluation set is Task 3;
this script draws the same ids Task 3 will use, so nothing is picked after seeing results.

  * coding:    HumanEval (MIT), pass@1, greedy
  * reasoning: GSM8K test (MIT), exact match on the final number, greedy
  * ids:       random.Random(SEED).sample(...) once; the first --n-code / --n-reason are used.
               The full 20-id draws are written to the output JSON.

SECURITY: HumanEval scoring executes model-written Python on this machine, in a temporary
directory with a timeout. It is not a sandbox. Run it on a machine you are happy to run
untrusted model output on.

Usage:
    python smoke_eval.py --server http://127.0.0.1:8080 --out results.json
"""
import argparse
import gzip
import json
import random
import re
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

SEED = 20261116
DRAW_SIZE = 20
HUMANEVAL_URL = "https://github.com/openai/human-eval/raw/master/data/HumanEval.jsonl.gz"
GSM8K_URL = "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl"

CODE_PROMPT = ("Complete the following Python function. Reply with the complete function "
               "in a single ```python code block and nothing else.\n\n{prompt}")
REASON_PROMPT = "{question}\n\nSolve this step by step. Finish with a final line of the form '#### <number>'."


def fetch(url: str, cache: Path) -> bytes:
    cache.mkdir(parents=True, exist_ok=True)
    f = cache / url.rsplit("/", 1)[-1]
    if not f.exists():
        with urllib.request.urlopen(url, timeout=60) as r:
            f.write_bytes(r.read())
    return f.read_bytes()


def load_humaneval(cache: Path) -> list[dict]:
    raw = gzip.decompress(fetch(HUMANEVAL_URL, cache)).decode("utf-8")
    return [json.loads(line) for line in raw.splitlines() if line.strip()]


def load_gsm8k(cache: Path) -> list[dict]:
    raw = fetch(GSM8K_URL, cache).decode("utf-8")
    return [json.loads(line) for line in raw.splitlines() if line.strip()]


def draw(n_total: int, size: int = DRAW_SIZE, salt: str = "") -> list[int]:
    return random.Random(f"{SEED}{salt}").sample(range(n_total), size)


def chat(server: str, prompt: str, max_tokens: int, timeout: int) -> tuple[str, int, float]:
    body = {
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "seed": 1,
        "max_tokens": max_tokens,
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": False},  # Qwen3: thinking off (design.md)
    }
    req = urllib.request.Request(server.rstrip("/") + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = json.loads(r.read())
    text = data["choices"][0]["message"].get("content") or ""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    return text, int(data.get("usage", {}).get("completion_tokens", 0)), time.time() - t


def wait_healthy(server: str, wait_s: int) -> None:
    end = time.time() + wait_s
    while True:
        try:
            with urllib.request.urlopen(server.rstrip("/") + "/health", timeout=5) as r:
                if r.status == 200:
                    return
        except (urllib.error.URLError, ConnectionError, TimeoutError):
            pass
        if time.time() > end:
            raise SystemExit(f"server {server} not healthy after {wait_s}s")
        time.sleep(2)


# ----- scoring -----

def extract_code(text: str) -> str:
    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", text, flags=re.S)
    withdef = [b for b in blocks if "def " in b]
    return (withdef or blocks or [text])[-1]


def passes_humaneval(problem: dict, response: str, timeout_s: int = 20) -> bool:
    code = extract_code(response)
    if f"def {problem['entry_point']}" not in code:
        return False
    header = "\n".join(l for l in problem["prompt"].splitlines() if l.startswith(("import ", "from ")))
    program = f"{header}\n{code}\n\n{problem['test']}\n\ncheck({problem['entry_point']})\n"
    with tempfile.TemporaryDirectory() as d:
        f = Path(d) / "prog.py"
        f.write_text(program, encoding="utf-8")
        try:
            r = subprocess.run([sys.executable, "-I", str(f)], cwd=d, capture_output=True, timeout=timeout_s)
        except subprocess.TimeoutExpired:
            return False
    return r.returncode == 0


NUM = r"-?\d[\d,]*\.?\d*"


def final_number(text: str) -> float | None:
    m = re.findall(r"####\s*\$?\s*(" + NUM + ")", text)
    cand = m[-1] if m else (re.findall(NUM, text) or [None])[-1]
    if cand is None:
        return None
    try:
        return float(cand.replace(",", "").rstrip("."))
    except ValueError:
        return None


def gsm8k_gold(answer: str) -> float:
    return float(answer.split("####")[-1].strip().replace(",", ""))


def correct_gsm8k(problem: dict, response: str) -> bool:
    v = final_number(response)
    return v is not None and abs(v - gsm8k_gold(problem["answer"])) < 1e-6


# ----- main -----

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--server", default="http://127.0.0.1:8080")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--cache", type=Path, default=Path("work/datasets"))
    ap.add_argument("--n-code", type=int, default=5)
    ap.add_argument("--n-reason", type=int, default=5)
    ap.add_argument("--max-tokens-code", type=int, default=768)
    ap.add_argument("--max-tokens-reason", type=int, default=640)
    ap.add_argument("--timeout", type=int, default=900, help="seconds per request")
    ap.add_argument("--wait", type=int, default=900, help="seconds to wait for /health")
    ap.add_argument("--label", default="")
    args = ap.parse_args()

    he, gsm = load_humaneval(args.cache), load_gsm8k(args.cache)
    code_ids = draw(len(he), salt="humaneval")
    reason_ids = draw(len(gsm), salt="gsm8k")
    wait_healthy(args.server, args.wait)

    results = []
    for kind, ids, n, data in (("coding", code_ids, args.n_code, he), ("reasoning", reason_ids, args.n_reason, gsm)):
        for i in ids[:n]:
            p = data[i]
            prompt = CODE_PROMPT.format(prompt=p["prompt"]) if kind == "coding" else REASON_PROMPT.format(question=p["question"])
            try:
                text, ntok, secs = chat(args.server, prompt, args.max_tokens_code if kind == "coding" else args.max_tokens_reason, args.timeout)
                err = None
            except Exception as e:  # keep going; an error counts as unsolved and is reported
                text, ntok, secs, err = "", 0, 0.0, repr(e)
            ok = (passes_humaneval(p, text) if kind == "coding" else correct_gsm8k(p, text)) if not err else False
            results.append({"kind": kind, "index": i, "id": p.get("task_id", i), "ok": ok, "error": err,
                            "completion_tokens": ntok, "seconds": round(secs, 1), "response": text[:1500]})
            print(f"{kind:9s} {str(results[-1]['id']):>14s}  {'PASS' if ok else 'fail'}  {ntok} tok  {secs:.0f}s" + (f"  ERROR {err}" if err else ""))

    def count(kind):
        r = [x for x in results if x["kind"] == kind]
        return {"solved": sum(x["ok"] for x in r), "n": len(r), "errors": sum(bool(x["error"]) for x in r)}

    summary = {"label": args.label, "seed": SEED, "coding": count("coding"), "reasoning": count("reasoning"),
               "draw": {"humaneval_indices": code_ids, "gsm8k_indices": reason_ids}, "results": results}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    c, r = summary["coding"], summary["reasoning"]
    print(f"\n{args.label}: coding {c['solved']}/{c['n']}  reasoning {r['solved']}/{r['n']}  (errors: {c['errors'] + r['errors']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
