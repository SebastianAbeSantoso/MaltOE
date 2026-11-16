# Frozen evaluation set

Every configuration (A to D, and the baselines) is scored on exactly these problems with exactly these settings (Req 10.4, 10.7). The three scores are reported separately and never combined.

| File | What it freezes |
| --- | --- |
| `coding.json` | 20 HumanEval problems: ids, a hash of each problem, and a hash of the source file |
| `reasoning.json` | 20 GSM8K test problems, same fields |
| `perplexity.json` | the first 64 windows of 512 tokens of the wikitext-2 test split, and the `llama-perplexity` arguments that select them |
| `sampling.json` | greedy decoding, fixed seed, thinking off, `max_tokens` per kind |
| `prompts.json` | the two prompt templates |

## How the problems were chosen

One draw, made before any Molt result existed: `random.Random(f"{seed}{salt}").sample(range(population), 20)` with seed `20261116` (the date Task 1 started) and salt `humaneval` or `gsm8k`. Nobody looked at the problems first. `harness/datasets.py` holds the algorithm. The Task 1 smoke test used the first 5 of each list, which `tests/test_evalset.py` checks.

## Sources and licences

| Part | Source | Licence |
| --- | --- | --- |
| HumanEval | [openai/human-eval](https://github.com/openai/human-eval) | MIT |
| GSM8K (test split) | [openai/grade-school-math](https://github.com/openai/grade-school-math) | MIT |
| wikitext-2 (raw, test split) | Merity et al., mirrored at [ggml-org/ci](https://huggingface.co/datasets/ggml-org/ci) | CC BY-SA 3.0 in the original release; the mirror carries no licence metadata |

The datasets are downloaded, not stored in this repository.

## Using it

```sh
uv run python -m harness.datasets fetch     # download to ./harness-cache
uv run python -m harness.datasets verify    # prove the download matches the frozen hashes
```

`verify` fails if any source file or any chosen problem has changed upstream.

## Rules

- **Do not edit these files once a result exists.** Changing a problem, a prompt, or a sampling value makes every earlier number incomparable. If a different set is needed, add a new set with a new name and report the old and new results separately.
- Calibration text for the importance scores (Task 5) must not overlap this set. It uses the wikitext-2 train split and llama.cpp source code.
- Twenty problems per set is small. A difference of one or two solved problems is noise (`harness/scorers/counts.py`), and reports show counts such as 14/20, not only percentages.
