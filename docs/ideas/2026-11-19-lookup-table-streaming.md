# Streaming a model's lookup tables over the network (checked 2026-11-19)

Question from the builder: can a model run from a web server via HTTP range requests, fetching only what each token needs? For MoE experts, no (about 375 MB per token even with a cache; see chat). For models with large n-gram lookup tables, the physics works.

## Models with large lookup tables (open weights)

| Model | Total / active | Lookup table | Per-token reads |
| --- | --- | --- | --- |
| DeepSeek-V4.1-Flash | 284B / 13B | Engram, 2 layers × 384M rows, 202.7 GB (FP8) ([CP paper, arXiv 2610.21810](https://arxiv.org/html/2610.21810)) | 24 rows per layer, 48 per token. Estimate: 202.7 GB / 768M rows ≈ 264 B per row, so about **13 KB per token** |
| Qwen3.8-Flash-Next | 125B / 6B + 51B n-gram + 4B MTP ([model card](https://huggingface.co/unsloth/Qwen3.8-Flash-Next)) | 320M rows × 160 values, 102.4 GB BF16 ([Delgon](https://huggingface.co/Delgon/Qwen3.8-Flash-Next-PLE-GGUF)) | 16 hashed heads per token at layer 1 → roughly 16 × 160 × 2 B ≈ 5 KB per token (estimate) |
| LongCat-Flash-Lite (MIT) | 68.5B / ~3B, over 30B in n-gram embeddings ([model card](https://huggingface.co/meituan-longcat/LongCat-Flash-Lite)) | 62.8 GB (CP paper) | not computed |

At 100 Mbps, 13 KB is about 1 ms of transfer. Decode is bound by round-trip latency instead, because each new token's addresses depend on the token just produced: one round trip per token unless draft tokens (MTP) are used to prefetch.

## Prior art

- **Rows read on demand from SSD:** done. A llama.cpp branch adds `--ngram-load-mode read` ([discussion #27864](https://github.com/ggml-org/llama.cpp/discussions/27864): 36 vs 40 tok/s against resident mode). DwarfStar reads DeepSeek's Engram rows from disk ([antirez GGUF](https://huggingface.co/antirez/deepseek-v4.1-flash-gguf)). Many recipes keep the table on NVMe.
- **Rows fetched from another machine:** done over RDMA or TCP on a LAN, for DeepSeek-V4.1 ([dsv41-remote-engram-rdma](https://huggingface.co/ProprietaryLegal/dsv41-remote-engram-rdma), [findings](https://huggingface.co/ProprietaryLegal/dsv41-spark-offload-findings)).
- **Rows fetched from a public HTTP host (e.g. Hugging Face) over the internet:** not found. A small step from the TCP work above.
- **Editing knowledge in the table:** [llama.cpp-NLTM](https://github.com/ortegaalfredo/llama.cpp-NLTM) hot-swaps knowledge into Qwen3.8's n-gram table.

## Verdict

Physically feasible, but mostly done. The internet-HTTP step only saves the table's disk space (60–200 GB). The backbone (68–125B parameters) still has to run locally, which the builder's machines can't do well. Not pursued as the project; recorded because the per-token numbers are useful.
