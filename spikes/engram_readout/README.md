# Engram readout spike (2026-11-19)

Reads DeepSeek-V4.1-Flash's Engram tables (layers 1 and 14, about 384M rows each, FP8) directly, with HTTP range requests against the Hugging Face CDN. No GPU and no model run.

- `killtest.py`: hashing from the model repo's `inference/engram.py`; checks compressed vocab = 99,092 and that the bucket primes sum to each table's row count; then two readouts on 400 wikitext-2 test positions.
- `probe.py`: nearest input embeddings for a phrase's Engram value vector, with a random-row control.
- `assoc_bench.py`: 30 phrases, hit@10 against hand-written related-word sets that exclude the phrase's own words.

Run: `uv run --no-project --with numpy --with requests --with sympy --with tokenizers python killtest.py 400` (needs `config.json` and `tokenizer.json` from the model repo in the working directory; caches `head.npy` and `embed.npy`, 2.6 GB each as float32).

Results are in `docs/ideas/2026-11-19-engram-killtest.md`.
