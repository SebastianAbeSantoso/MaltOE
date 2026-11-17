# Feasibility Report: MaltOE Task 1

Status as of 2026-11-16: **research gates decided; hardware gates waiting for your runs.** Gates G0 and G1 and the candidate review are complete. G2 is half answered and its main finding changes the engine plan. G3 and G4 need the laptop (instructions in [`spikes/README.md`](../spikes/README.md)).

Evidence labels used below: **verified** means I read it in a primary source or ran it; **unverified** means I did not, and each such item says how it will be checked. Numbers marked *sandbox CPU* were measured on a Linux server CPU and say nothing about CUDA speed.

## 1. Gate summary

| Gate | Question | State | Result |
| --- | --- | --- | --- |
| G0 Novelty | Does existing work already adapt width and residency together, online, under a changing memory budget with a user objective? | **Decided** | Not found. Claim stands as approved. Three prior-art additions needed in the README (section 3). |
| G1 Model | Is Qwen3-30B-A3B usable? | **Decided** | Yes. Apache-2.0, config matches design.md exactly, official GGUFs exist. |
| G2 Engine | Can the llama.cpp fork run variable-width experts and dynamic residency? | **Half answered** | Chunked experts are correct on CPU. **Upstream llama.cpp already has an expert cache that does the VRAM paging.** CUDA timings and cache behaviour need your runs (Steps A, B). |
| G3 Width quality | Does a width-reduced Qwen3-30B-A3B stay useful? | **Waiting for hardware** | Scripts tested; Step D. |
| G4 Budget enforcement | Do the VRAM and RAM fillers create real pressure on Windows? | **Decided on the laptop** | An idle VRAM filler does not; an active one does, and the result is a silent slowdown, not an error. The RAM filler works. PC check pending. |

## 2. Findings that change the spec

These need your decision at the checkpoint (section 9). I have not edited `requirements.md` or `design.md`.

1. **llama.cpp already contains a working-set-style expert cache.** `--moe-cache-mib` (marked EXPERIMENTAL in `include/llama.h`) keeps frequently and recently used experts of host-resident layers in a VRAM buffer, uploads only the missing ones, and hooks into `build_moe_ffn`. This removes the biggest risk in design.md (hand-building the per-layer slot remap and host sync) but means MaltOE extends someone else's cache. It also means the strongest stock baseline is `--n-cpu-moe` **plus** `--moe-cache-mib`, which Req 2.1 and Req 10.8 do not yet name. Details in section 6.
2. **The RAM and SSD tiers would be the operating system's page cache**, not a MaltOE-managed cache, unless we build one. Upstream host experts are memory-mapped by default.
3. **The importance run should use the Q4_K_M file, not Q8_0.** The official Q8_0 GGUF is 32.5 GB, more than the laptop's RAM and a heavy download. The final conversion (Task 5) can use Q8_0 on the PC if it matters.
5. **Name collision (the reason for the rename).** This project's earlier working name was Molt. NVIDIA's NeMo team open-sourced an unrelated project called Molt on 2026-07-22 ([NVIDIA-NeMo/labs-molt](https://github.com/NVIDIA-NeMo/labs-molt), [paper, arXiv 2607.21653](https://arxiv.org/abs/2607.21653)). It is a PyTorch framework for training language models with agentic reinforcement learning (Ray, vLLM, FSDP2). It does not do inference on consumer GPUs, expert width, or expert residency, so it is **not prior art for MaltOE's claim**. It was older than this project and would have been found first by anyone searching the name. The project was renamed to MaltOE on 2026-11-16 (checks in 3.2c). I did not check whether NVIDIA holds a trademark on the name; that needs a lawyer, not a search.
4. **README prior art needs additions:** llama.cpp's MoE cache, ElastiLM, and a note that DyMoE was also evaluated on Qwen3-30B-A3B. Also two wording fixes (DATS link, Yao et al.). Section 3.

## 3. G0: novelty

### 3.1 What I read

- **CRISP** ([arXiv 2610.37442](https://arxiv.org/abs/2610.37442), MobiCom '26): the full HTML text (about 80,000 characters). I searched it for every term about online behaviour, budgets, adaptation, profiling and runtime, and read the design sections on tiered execution (3.3), the pruning-aware I/O scheduler (3.4), the implementation, and the experimental setup and memory-budget experiments. I did not read every paragraph line by line.
- **WiSP** ([arXiv 2606.21868](https://arxiv.org/abs/2606.21868), v2): abstract plus keyword reading of the full text for controller behaviour, quantization, and engine.
- **DyMoE** ([arXiv 2603.19172](https://arxiv.org/abs/2603.19172)): full text, keyword reading.
- **DATS**: the full PDF text (13 pages) was downloaded; I read the abstract and structure, not the proofs.
- **Yao et al.**, **MC-MoE**, **D2MoE**, **MoSE**, **HydraLLM**, **ElastiLM**: abstracts only.

### 3.2 Closest work

| Work | What it does | What it does not do (as read) | Evidence |
| --- | --- | --- | --- |
| CRISP | Importance-ordered neurons so one expert supports several widths; offline calibration sorts expert *combinations* into Hot / Warm / Cold tiers with different widths; a width-aware scheduler overlaps expert loading with compute. Mixtral-8x7B and DeepSeek-MoE-16B on a Jetson AGX Orin (32 GB unified) and an RTX 4090 with 64 GB host RAM. | No mechanism found that changes widths or tiers *while serving* in response to a changing memory budget. The expert-memory budget (10% to 40%) is an experiment parameter set per run. When the workload or model changes, the text says the profiling artifacts are refreshed *offline* and "the same online execution mechanism" is kept. No user-selected objective. Not tested at 8 GB. Code stated as "will be made publicly available" (not yet found). | verified, partial coverage |
| WiSP | Working-set expert pager for vLLM, byte-identical outputs; MV-WSA splits VRAM between resident experts and the KV cache, as a startup configurator or a live controller that resizes both pools at a drained barrier. Code: [nokia-applied-research/WiSP](https://github.com/nokia-applied-research/WiSP). | Never changes computation (by design). The reported live-controller result is "over a fixed offline split at the same VRAM budget", so the total budget is held fixed. Whether it reacts to an *external* VRAM change: I found no evidence either way (unverified). Runs on vLLM, which needs WSL2 on Windows. | verified |
| DyMoE | Runtime per-expert mixed precision (8-bit, 4-bit, and "0-bit" meaning skip), driven by token and gate importance, depth-adaptive scheduling, look-ahead prefetching, mixed-precision VRAM cache. **Also evaluated on Qwen3-30B-A3B**, on an RTX 3090 with a software cap of 12 to 24 GB. | Precision and expert skipping, not width. Driven by importance, not by a changing budget. | verified, keyword reading |
| ElastiLM ([arXiv 2409.09071](https://arxiv.org/abs/2409.09071)) | Dense LLMs on mobile: one-time neuron reordering gives elastic sub-models, switched per request to meet latency SLOs. | Dense models, no MoE, no expert residency. | abstract level only |
| Edge0 | SSD expert streaming with a trained routing predictor, fixed expert slots per layer. | No width control. | verified from README |
| DATS | Splits each LLM request into prefill, a validator-backed mandatory prefix, and an optional suffix, with deadline admission. Authors Haque and Saifullah, UT Dallas, accepted at RTSS 2026 per the author's page. | Tokens, not experts. Physical-I/O agents, not local chat. | verified from the PDF |
| Yao et al. ([arXiv 2011.01112](https://arxiv.org/abs/2011.01112)) | Real-time scheduling of deep learning services as imprecise computations: mandatory part plus optional parts, "depth assignment and stage scheduling", on machine vision. | Not LLMs, not MoE. | abstract and section headings |
| llama.cpp MoE cache | Engine-level VRAM cache of host-resident experts, frequency-and-recency policy, fixed size chosen at context creation. | No width, no runtime resize, no controller. | verified in source |

### 3.2b GitHub landscape (added 2026-11-16)

The first G0 pass searched papers and a few named systems. A GitHub search afterwards found a crowded field of open-source projects that run MoE models larger than VRAM on consumer machines. Star counts are from the GitHub API on 2026-11-16. "Read" means I fetched the README and searched it for width, precision, top-k, budget, quality and live-pressure language, not that I read every line or ran the code.

| Project | What it does (per its README) | Overlap with MaltOE |
| --- | --- | --- |
| [JustVugg/colibri](https://github.com/JustVugg/colibri) (Apache-2.0, 51.5k stars) | Hierarchical engine for frontier MoE models: hot experts on the GPU (CUDA, Vulkan), warm in RAM, cold on SSD. A hardware-aware planner detects RAM and VRAM budgets at setup and picks a build. Lossless or "quality-bounded" formats. | **Residency tiers and hardware detection.** No width control found. |
| [wici-ai/inferssd](https://github.com/wici-ai/inferssd) (Apache-2.0, 1.2k) | llama.cpp plus SSD expert streaming, browser chat, OpenAI-compatible API. | Residency. |
| [yalun753/moe-l2](https://github.com/yalun753/moe-l2) (Apache-2.0, 147) | Expert cache for low-VRAM cards (8 GB class). Changelog lists "soft_resize" and a "VRAM-adaptive main table top-k (12 GB or less = 75, more = 100)". The README does not say what those two mean (unverified). | Residency, possibly a live cache resize. |
| [christ-pher/bnk](https://github.com/christ-pher/bnk) | Tiered experts on a V100 plus host RAM. Re-reads the driver's free VRAM whenever it hands out memory, so allocations made outside the cache "come back out of the expert cache". Lends a 256K context's memory to the expert cache until needed. | **Live shrinking of the expert cache under VRAM pressure**, residency only. |
| [pmerolla/fomoe](https://github.com/pmerolla/fomoe) (MIT, 32) | Multi-tier caching with cache-aware routing: a cached expert is substituted for an uncached one. Reported for Qwen3.5-397B: 5.1 tok/s exact, 8.8 tok/s with moderate substitution, +3.5% perplexity (their numbers). A threshold chooses fidelity. | **Quality traded for speed under a memory limit**, on the expert-selection axis. |
| [Niko1221/Strata](https://github.com/Niko1221/Strata) and forks ([eddoursul/Lamina](https://github.com/eddoursul/Lamina), Edge0's edge0-strata) | MoE offload engine. Its default skips low-ranked experts ("tail skip"), faster but approximate, with an exact mode. Lamina is a fork tuned for NVIDIA GPUs plus RAM. | Residency, and a static lossy knob (skip tail experts). |
| [ewhacc/tandem.cpp](https://github.com/ewhacc/tandem.cpp) (78) | llama.cpp patch for MoE offload. `--fast` skips tail experts, approximate, quality "TODO". | Residency, static lossy knob. |
| [neerajdad123-byte/moe](https://github.com/neerajdad123-byte/moe) (MIT, 9) | Qwen3.6-35B-A3B at about 68 to 78 tok/s on an RTX 4050 Laptop 6 GB (their numbers). Free VRAM becomes expert slots. `--fast` is slightly approximate. | **The "35B on a small GPU" demo already exists.** |
| [LozzKappa/hotpin-llm](https://github.com/LozzKappa/hotpin-llm) (AGPL-3.0) | Routing-guided expert pinning, bit-identical to a full-RAM run. | Residency, lossless. |
| giannisanni/pulsar, yairpatch/flyweight, AlphaChat-dotcom/alphallama, Kelsidavis/hydranet, IbrahimSiddiqui007/moe-cache | Expert streaming and caching engines. | Residency. |

**What I did not find in any of them:** expert **width** (a prefix of importance-ordered neurons within an expert) as a runtime knob, mandatory and optional work scheduled by an objective, or width and residency adjusted together when the budget changes. That is the same conclusion as 3.3, now across more projects. The survey covers about 14 READMEs of many more, found by keyword. A project that uses other words could exist.

**What the survey changes:**

1. **Residency is not a contribution, and neither is live cache shrinking.** bnk and possibly moe-l2 do it. The README must not suggest otherwise.
2. **Lossy knobs already exist on a different axis:** skipping tail experts (Strata, tandem.cpp) and substituting cached experts (fomoe). A width-reduced expert is only interesting if it gives a better quality-versus-compute-and-bytes curve than these. **They need to be baselines in the 2x2 envelope** (proposed, not applied): Strata's tail-skip or tandem's `--fast` at matched compute.
3. **Demo beat 1 is weaker than the handoff assumed.** A 35B model on a 6 GB laptop GPU at chat speed is already published by others, and stock `llama.cpp --n-cpu-moe` is far from "crawls below 1 tok/s". The comparison must be against `--n-cpu-moe` plus `--moe-cache-mib`, not a naive load. Beat 2 (the live squeeze with a controlled quality trade) is the real differentiator.

### 3.2c Name checks (2026-11-16)

| Name | Finding |
| --- | --- |
| Molt (earlier working name) | NVIDIA's NeMo team released an agentic-RL training framework called Molt on 2026-07-22 (section 2, finding 5). **Reason for the rename.** |
| Lamina | [eddoursul/Lamina](https://github.com/eddoursul/Lamina): a fork of Strata that runs MoE models across NVIDIA GPUs and RAM. Also a 2024 paper, [arXiv 2405.01814](https://arxiv.org/abs/2405.01814), on attention offloading. **Taken, and in the same field.** |
| Instar | [instar.sh](https://instar.sh/): an AI-agent framework ("Claude Code, with a mind of its own"). **Taken.** |
| Ecdysis | [cuiyu-ai/Ecdysis](https://github.com/cuiyu-ai/Ecdysis): "Efficient and Effective Training of Runtime Harnesses for LLM Agents" (14 stars). **Taken in AI.** |
| Malt | Taken as a package name on PyPI (a MATLAB linter) and npm. [bnpr/Malt](https://github.com/bnpr/Malt) (Blender rendering framework, 1.2k stars, malt3d.com), [themetaschemer/malt](https://github.com/themetaschemer/malt) (an ML toolkit), and malt.com (a large freelance marketplace whose AI and LLM engineer profiles dominate search results). **Busy; not chosen.** |
| **MaltOE (chosen)** | No repository, package, or project of that name found on GitHub, PyPI, npm, or the web. `maltoe` is a personal GitHub account (since 2009, inactive). Malt + MoE, pronounced "Malt". Searches cover only those sources; trademarks were not checked. |

### 3.3 Verdict

I did not find work that changes expert width and expert residency together, during a live session, when the VRAM budget changes, steered by a user objective. **The narrowed claim in `product.md` and the README stands unchanged.** Residual risks: the CRISP source is unreleased, I did not read every CRISP paragraph, and I could not check WiSP's behaviour under an external VRAM change.

### 3.4 Required README changes

1. DATS: link the primary source (done in the same PR as this report).
2. Yao et al.: the README said "applied to layers and early exits". The abstract says mandatory and optional parts with depth assignment and stage scheduling; the words "early exit" do not appear in the paper. Reworded (same PR).
3. Add llama.cpp's MoE cache to prior art, since MaltOE builds on it (same PR).
4. Add ElastiLM (same PR). DyMoE row gains the Qwen3-30B-A3B note (same PR).

### 3.5 Handoff figures re-verified

| Figure in handoff | Source | Result |
| --- | --- | --- |
| CRISP: up to 2.1x throughput at the same memory budget, accuracy loss limited to 3.6% | [abstract](https://arxiv.org/abs/2610.37442) | **verified** (Mixtral-8x7B and DeepSeek-MoE-16B) |
| D2MoE: up to 53% lower peak memory | [abstract](https://arxiv.org/abs/2504.15299) | **verified**; also 1.39x throughput |
| MC-MoE: 76.6% compression at 2.54 bits, 3.8% accuracy loss; 15% fewer activated parameters with under 0.6% drop | [abstract](https://arxiv.org/abs/2410.06270) | **verified** (ICLR 2025) |
| WiSP: up to 2.0x decode throughput vs static offload on a 24 GiB RTX 3090; byte-identical; code public | [abstract](https://arxiv.org/abs/2606.21868) | **verified**. First posted 2026-06-20; v2 is 2026-08-30. The handoff said August 2026. |
| Edge0 "released November 4, 2026" | [README](https://github.com/Edge0-AI/edge0) | **partly**: the unified engine `edge0-infer` was released 2026-11-04. The framework and models date from 2026-09-08 and the paper from 2026-09-16. |
| Krasis, FloE, OD-MoE, DynamicInfer, AMD GPT-OSS 20B dynamic loading, DeepSeek V4.1 Flash and DeepInfer-Mac figures, CPU-ARC | none read | **unverified**, not used anywhere in the README. V4.1 figures are re-verified in Task 15.2 if that stretch is attempted. |

## 4. G1: model

Source: [Qwen/Qwen3-30B-A3B](https://huggingface.co/Qwen/Qwen3-30B-A3B), `config.json` fetched 2026-11-16.

| Field | design.md table | Actual | Match |
| --- | --- | --- | --- |
| Layers | 48, all MoE | 48 (`decoder_sparse_step` 1, `mlp_only_layers` empty) | yes |
| Hidden size | 2048 | 2048 | yes |
| Routed experts | 128 | `num_experts` 128 | yes |
| Top-K | 8 | `num_experts_per_tok` 8 | yes |
| Expert intermediate size | 768 | `moe_intermediate_size` 768 | yes |
| Shared experts | none | none in the config | yes |
| Attention | 32 query heads, 4 KV heads, head dim 128 | 32, 4, 128 | yes |

- **Licence:** Apache-2.0 (model card metadata). Not gated.
- **GGUF:** the official [Qwen/Qwen3-30B-A3B-GGUF](https://huggingface.co/Qwen/Qwen3-30B-A3B-GGUF) has Q4_K_M (18.56 GB), Q5_0, Q5_K_M, Q6_K and Q8_0 (32.48 GB). llama.cpp at the pin has the `qwen3moe` architecture.
- **Quality evidence:** none collected by me. Gate G3 measures it.
- **Other checkpoints:** a public repo named `Qwen/Qwen3.6-35B-A3B` exists. I recorded that and did not evaluate it (per your instruction, we do not plan around it). Edge0's 35B model is built on it.

## 5. Candidate review (Req 1.2 to 1.4)

| Candidate | Repo / licence | Checkpoints | Nested widths | Variable-width execution | Dynamic residency across VRAM, RAM, SSD | Verdict |
| --- | --- | --- | --- | --- | --- | --- |
| MoSE | [tnurbek/mose](https://github.com/tnurbek/mose); no licence file; README says "The code is coming soon." | none found (HF search found none) | trained from scratch, GPT-style, plus one DeepSeek model adapted by continued pretraining | no released code | no | **Not feasible now.** Fallback only if code and checkpoints appear. |
| HydraLLM | paper only, [arXiv 2610.23455](https://arxiv.org/abs/2610.23455); no code repository found by search (unverified) | none found | yes, but width is the number of attention heads in a dense MQA model | no | no | **Not feasible** for an MoE chat model. |
| Edge0 | [Edge0-AI/Edge0](https://github.com/Edge0-AI/edge0), Apache-2.0; checkpoint [Edge0-35B-A3B-preview](https://huggingface.co/Edge0/Edge0-35B-A3B-preview) Apache-2.0, ungated | released, 4-bit with LoRA adapters and routing predictor | no | no | yes: fixed expert slots per layer, SSD streaming, `--memory-budget`; GGML engine with CPU and Vulkan on Windows; CUDA only through the separate `edge0-strata` engine for 145B and 120B models | **Fallback engine (Route B) and baseline BL-3.** It only runs its own checkpoint, so adding width would also touch the adapters (my inference, not verified). |
| llama.cpp | [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp), MIT | official Qwen GGUFs | no (we add it) | no (we add it; section 6) | VRAM to host: yes, `--moe-cache-mib` (experimental). Host to SSD: the OS page cache via mmap. | **Route A, default engine.** |
| llama.cpp `--n-cpu-moe` | same | same | no | no | static: experts of the first N layers stay on the CPU | **Baseline BL-2**, together with the cache (BL-2b, proposed). |
| CRISP | [arXiv](https://arxiv.org/abs/2610.37442); code not released | none | yes (design reference) | research code only | host to GPU | **Design reference, not a base.** |

## 6. G2: engine

### 6.1 Pinned commit

`a946eb541a6d3051496bdcc451a0fa85772b8dfd`, committed 2026-11-16 ("server: answer reasoning_effort levels the template rejects with 400 (#32332)"). Licence MIT. GitHub lists `v1.0.0` as the latest release (published 2026-11-13). **Unverified:** that this commit builds with CUDA on Windows; Step 0 of the spikes tests that.

### 6.2 Facts verified in the source at the pin

| Question | Answer | Where |
| --- | --- | --- |
| Does `llama-imatrix` record per-expert, per-neuron statistics for `ffn_down_exps`? | **Yes.** For each `MUL_MAT_ID` it accumulates `x[j]^2` (and `x[j]`) per expert id and per input column, with a count per expert. For `ffn_down_exps` the input is the intermediate activation, so this is exactly mean(h^2) per neuron per expert. Saved as `<tensor>.in_sum2` shaped [n_expert, 768] and `<tensor>.counts`. | `tools/imatrix/imatrix.cpp` lines 596 to 660 |
| Is there an API to drop a partial KV position? | Yes, `llama_memory_seq_rm`. | `include/llama.h` line 770 |
| Is `qwen3moe` supported? | Yes. | `src/models/qwen3moe.cpp` |
| Static MoE offload? | `-cmoe`, `-ncmoe`, and tensor overrides (`-ot`). | `common/arg.cpp` line 2731 |
| Does a MoE cache exist? | **Yes.** `--moe-cache-mib N`, `--moe-cache-cpu-share F`. See 6.3. | `src/llama-moe-cache.{h,cpp}`, `docs/moe-cache.md` |
| Where does the expert FFN get built? | One lambda, `build_experts` in `build_moe_ffn`: gate, up, activation, down. This is where a chunk loop would go. | `src/llama-graph.cpp` lines 2445 to 2610 |
| Does the scheduler copy only the used experts when it offloads a `MUL_MAT_ID`? | Yes, through `sched_copy_experts`, a copy callback installed on the scheduler. | `src/llama-context.cpp` lines 782, 3159 |

### 6.3 The upstream MoE cache

What it does: for layers whose experts are host-resident, it keeps a device buffer of expert "slots". Each layer has a slot map in host memory; when the scheduler copies the map to the device, a callback uploads the missing experts. Policy: one LRU list per frequency class, with a combined recency-and-frequency score decayed with a 128-token half-life. Optionally the CPU computes a share of the misses instead of uploading them.

What it lacks for MaltOE:

| MaltOE needs | Upstream |
| --- | --- |
| Change the cache size while running | No. Size is a context parameter (`cparams.moe_cache_size`) fixed at context creation. |
| Hit and miss counters for the dashboard | Printed once to the log at teardown (`log_stats`). No API. |
| Width-aware pages | No. A page is a whole expert. |
| A "working set no longer fits" signal | No. |
| Policy | Recency plus frequency, not a strict window over recent tokens. |
| Active development | Several related pull requests are open (for example #32076 "run everything without syncs", #30594 "hybrid MoE expert cache with profile prefilling and zero-upload inference"). The code will keep moving, so the pin matters. |

### 6.4 Chunked experts (Task 1.6, partly done): *sandbox CPU* results

[`spikes/chunked_moe`](../spikes/chunked_moe/chunked_moe.cpp) builds one Qwen3-30B-A3B MoE layer (2048 x 768, top-8) with ggml's own `mul_mat_id` and compares:

- **Correct:** with three chunks, the summed output equals the fused expert to a relative RMS error of about 1e-7 for f32, f16, q8_0, and q4_K. Width k equals a plain f32 reference that uses only the first k chunks (f32 about 2e-7; quantized types about 1.2e-2 to 1.9e-2, which is the activation-quantization noise the full-width kernel has against the same reference). Width 1 and width 3 differ by about 81% relative RMS, so the test can tell widths apart. The build is clean under AddressSanitizer.
- **Cost:** on a server CPU with 32 experts (not 128) and q4_K weights, a decode call took about 0.40x, 0.81x, and 1.21x of the fused time at k = 1, 2, 3. Time scales with width. The 1.1x to 1.2x overhead at full width on CPU comes from extra operations; **CUDA may differ, which is what Step A measures.**
- **Block-size rule:** four chunks (192 neurons) works with q8_0 and is rejected for q4_K because 192 is not a multiple of 256. This confirms the note in design.md.

### 6.6a First CUDA run of the chunked-expert spike (RTX 4060 Laptop GPU, q4_K, 128 experts, top-8, one layer)

Run by the builder on 2026-11-16, llama.cpp `a946eb54`, CUDA 12.9, MSVC 19.39.

**Correctness, verified.** Summing three chunks equals the fused expert to 9.0e-8 (decode, 1 token) and 1.3e-7 (prefill, 512 tokens) relative RMS error, the same agreement as on CPU. Width k against the f32 prefix reference is 1.16e-2 to 1.19e-2 at every k, which is the quantization noise the full-width kernel also has. Width 1 and width 3 differ by 0.85 (decode) and 0.82 (prefill), so the test can tell widths apart. Four chunks with q8_0 (192 neurons each) also pass, at 1.1e-2 against the reference and 9.9e-8 against fused. The chunked layout is therefore **correct on CUDA**.

**Timing, second run with the fixed method** (GPU warmed for 1.5 s, 7 rounds, order rotated, median shown; the spread between rounds was 1% to 19%, so these numbers are usable). ms per call, one layer, q4_K, no CUDA graphs:

| | decode, 1 token | prefill, 512 tokens |
| --- | ---: | ---: |
| fused (one tensor, full width) | 0.0725 | 2.689 |
| chunked k = 1 (33%) | 0.0492 (0.68x of fused) | 1.088 (0.40x) |
| chunked k = 2 (67%) | 0.0936 (1.29x) | 2.543 (0.95x) |
| chunked k = 3 (100%) | 0.1429 (1.97x) | 3.991 (1.48x) |

The first run's prefill figure for the fused variant (9.6 ms) was an artifact of timing it first on a GPU that had not reached full clocks. With the fix it is 2.7 ms. Decode agreed between the two runs. The four-chunk q8_0 timings were not rerun and are not used.

**What the numbers say:**

1. **Narrower is cheaper in proportion, measured against the chunked full width.** k = 2 costs 0.65x of k = 3 in decode and 0.64x in prefill; k = 1 costs 0.34x and 0.27x. The ideal is 0.67x and 0.33x. So the width mechanism does reduce work.
2. **Chunking has a price at full width, measured against the stock fused expert:** 1.97x in decode and 1.48x in prefill. At k = 2 it is still 1.29x in decode, and only k = 1 beats the fused expert in decode (0.68x). Prefill is better: k = 2 is on par with fused and k = 1 is 2.5x faster.
3. **Why, not tested:** each chunk repeats the gate, up, down and activation steps, so one layer runs 3x as many kernels, and the input is quantized again for every matmul. At decode the calls are only 0.05 to 0.14 ms, which is close to launch cost. This spike does not use CUDA graphs, which llama.cpp does use for single-token decode, so the decode figure probably overstates the real engine's cost.
4. **Sizing (estimate from arithmetic, not measured):** the extra 0.07 ms per layer at full-width decode is about 3.4 ms per token across 48 layers, if every expert were resident on the GPU. One expert at q4_K is about 2.65 MB (768 x 2048 x 3 weights at about 4.5 bits). Over a PCIe link of about 14 GB/s (a guess for a laptop x8 link) that is roughly 0.19 ms to fetch, longer than the whole per-layer chunking cost. So the price matters when everything fits in VRAM, and much less when experts are being streamed, which is the case MaltOE exists for. WiSP also reports decode as link-bound in that regime.

**What it changes (for your decision, section 9):**

- **Speed objective.** The approved rationale was that narrower is faster in FLOPs, so capping at C - 1 is a latency trade-off even with VRAM to spare. In this spike that is true against the chunked full width, but not against the fused expert in decode: k = 2 is 1.29x slower than fused. The Speed cap should be re-checked in the real engine (CUDA graphs on) before its rationale is repeated in the write-up.
- **Configuration A in the 2x2.** "Full" must say whether it is the stock fused expert or the chunked layout at k = C. They differ by the price above. Comparing MaltOE's chunked full width against stock llama.cpp charges MaltOE for the chunking; comparing against chunked full width isolates the controller. The report should show both.
- **Reducing the price** (none attempted): merge gate and up into one tensor per chunk (llama.cpp already has a merged `gate_up` path), quantize the input once per layer, or add one op that handles k chunks in a single launch.

### 6.6b Avoiding the chunk price: prefix views

The price in 6.6a comes from running one expert as several smaller ones. CRISP does not do that: it stores each expert once, reordered by importance, so that "a width-k execution simply accesses the first k neurons" (CRISP, section on neuron reordering). In ggml terms a width-k expert is a *view* of the one fused tensor: for gate and up, the first k x 256 rows of each expert; for down, the first k x 256 entries of every row, with the original row stride. That is one matmul per projection at every width, the same kernel count as the stock expert, only smaller shapes. design.md ruled this out because "kernels do not support truncated quantized rows in general"; that was an untested assumption.

`spikes/chunked_moe --views` tests it. Sandbox CPU result (32 experts, top-8, 1 token): the views give the same numbers as the chunked layout (difference 0 to 2.4e-7, all PASS) for f32, q8_0, and q4_K, and under AddressSanitizer. Timing on that CPU, as multiples of the fused expert:

| q4_K, CPU, decode | k = 1 | k = 2 | k = 3 |
| --- | ---: | ---: | ---: |
| chunked | 0.44x | 0.88x | 1.32x |
| prefix view | 0.48x | 0.78x | 1.04x |

So on CPU the price at full width disappears (1.04x, within noise of the 1.00x it should be) and narrower widths still cost proportionally less.

**CUDA result (RTX 4060 Laptop GPU, 128 experts, top-8, q4_K, run by the builder 2026-11-17).** The CUDA backend accepts the views: they match the chunked layout to 8.6e-8 to 1.3e-7 and the f32 prefix reference to the usual 1.2e-2, all PASS, in decode and prefill. Timing, as multiples of the stock fused expert:

| | prefill, 512 tokens (spread 0% to 1%) | decode, 1 token (spread up to 33%, flagged as noisy) |
| --- | ---: | ---: |
| chunked k = 1 / 2 / 3 | 0.40x / 0.95x / 1.49x | 0.95x / 1.76x / 2.61x |
| prefix view k = 1 / 2 / 3 | **0.44x / 0.76x / 1.08x** | 1.26x / 1.40x / 1.71x |

- **Prefill: views remove most of the price.** Full width falls from 1.49x to 1.08x of stock, and two-thirds width from 0.95x to 0.76x, so two-thirds width is now cheaper than the stock expert. One-third width costs about the same either way.
- **Decode: this spike cannot settle it.** The stock expert's time was stable between runs (0.0725 and 0.0712 ms), but the other variants moved a lot: chunked full width was 1.97x of stock yesterday and 2.61x today. Each call is 0.07 to 0.19 ms, which is mostly launch cost, and the spike runs without CUDA graphs, which llama.cpp uses for single-token decode.
- **One result I cannot explain.** The full-width view has the same shape and data as the stock tensor and goes through the same `mul_mat_id` dispatch, yet took 1.71x as long in decode. I read the CUDA dispatch and its gate/up/activation fusion check and found no condition that treats a view differently. Finding the cause needs a profiler run (Nsight Systems) or the real engine, not more guessing.

**Conclusion for the design:** prefix views are the better layout for compute in prefill, measured. For decode, the deciding measurement is Task 6.8 in the real engine with CUDA graphs on, against A-stock.

Two limits that stay now that CUDA accepts it:

- **Paging.** For down, a width-k prefix is a piece of every row, not one contiguous block. Copying a width-k slice over PCIe as 2,048 short pieces would be slow. A layout that is contiguous for transfer (chunk-major, as in the package) and a layout that is a prefix view for compute (expert-major) are different. One way to reconcile them is to upload each chunk contiguously and scatter it into the expert-major slot on the GPU, which runs at VRAM speed. That is a design idea, not tested.
- **Quality is a separate question.** Nothing here says a width-reduced Qwen3-30B-A3B stays useful. That is gate G3 (Step D).

### 6.6c Step B: llama.cpp's MoE cache against static offload (first run, 2026-11-17)

RTX 4060 Laptop GPU, i9-14900HX, 16 GB RAM, Qwen3-30B-A3B Q4_K_M (18.6 GB file, larger than RAM), 128 generated tokens from a fixed prompt, greedy, two runs per configuration, llama.cpp `a946eb54` with CUDA 12.9.

| configuration | decode tok/s (run 1 / 2) | prompt tok/s | peak GPU MiB | disk read GiB per run | lowest free RAM MiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| static `-ncmoe 40` | 10.8 / 13.3 | 12.1 / 12.5 | 4,387 | 6.0 / 10.2 | 89 / 101 |
| static `-ncmoe 34` | 15.9 / 25.7 | 14.0 / 13.3 | 6,429 | 9.7 / 9.5 | 201 / 213 |
| static `-ncmoe 28` | 17.2 / 18.6 | 11.1 / 13.3 | 7,905 | 5.0 / 3.5 | 242 / 252 |
| cache 1,024 MiB, all experts on host | 5.2 / 5.4 | 10.8 / 19.9 | 2,435 | 5.6 / 7.6 | 251 / 277 |
| cache 2,048 MiB | 6.3 / 6.0 | 19.9 / 19.6 | 3,459 | 7.7 / 7.4 | 99 / 95 |
| cache 3,072 MiB | 6.8 / 6.9 | 20.0 / 20.4 | 4,483 | 6.4 / 7.4 | 297 / 296 |
| cache 4,096 MiB | 7.5 / 7.9 | 19.5 / 18.9 | 5,509 | 7.4 / 6.8 | 305 / 319 |

Disk read is the sum of one-second samples of the whole disk, so it is approximate. Two script problems: every row says "FAIL (exit )" only because Windows PowerShell 5.1 drops the exit code (fixed in `spikes/common.ps1`; the runs finished, since speeds were printed), and the cache hit rate column is empty because the hit-rate line was not in the logs. The likely cause is log verbosity: llama.cpp's own tuner (`scripts/moe-cache-tune.py`) runs the server with `-lv 4` before reading that line, so it is probably filtered out at the default level. That is an inference from the tuner's code, not tested; the probe now passes `-lv 4`.

**What it shows:**

1. **RAM, not VRAM, is the binding limit on this laptop.** Free RAM fell to 0.1 to 0.3 GB in every run, and each 30 to 50 s run read several GiB from disk. The SSD tier is exercised with no RAM filler, as expected for a model file larger than RAM.
2. **At matched VRAM, static offload was faster than the cache.** At about 4.4 GB of GPU memory, static `-ncmoe 40` decoded at 10.8 to 13.3 tok/s and the 3 GB cache at 6.8 to 6.9. Every cache size was slower than every static setting. A likely reason, not tested: llama.cpp's own documentation says host experts are not pinned when the model is memory-mapped (the default), so uploads to the GPU are slower, and it suggests `--moe-cache-cpu-share` for that case. Pinning all experts is not possible here, because the file does not fit in RAM. The laptop's strong CPU also favours computing experts on the CPU, as static offload does.
3. **Runs vary a lot.** `-ncmoe 34` gave 15.9 and 25.7 tok/s in two runs of the same command. With RAM this tight, the state of the OS page cache decides much of the speed. Later measurements need more repeats and a recorded free-RAM figure.

**What it changes:**

- **The strongest stock baseline on this laptop is static `-ncmoe`, not the cache (BL-2b).** Both stay in the plan; the report must not assume the cache is the harder opponent.
- **Approved change 2 still stands but is now riskier.** MaltOE needs a VRAM residency mechanism for width-aware paging either way, but the upstream cache, as configured here, lost to static offload. Before building on it, the next probe should try `--moe-cache-cpu-share` (llama.cpp ships a tuner, `scripts/moe-cache-tune.py`) and record hit rates.
- **A point in width's favour, to be measured, not claimed:** when the model streams from disk, a narrower width reads fewer bytes per expert, so it may help this laptop more through disk and RAM than through GPU compute.

### 6.6d Step B2: the cache with a CPU share, against static offload at the same VRAM (2026-11-17)

Same laptop, model, and prompt as 6.6c, now with `-lv 4` so the cache's statistics are logged. All rows use about 4.4 GB of GPU memory. "CPU share" is `--moe-cache-cpu-share`: the share of cache misses the CPU computes from host memory instead of uploading to the GPU. Two runs each; the cache statistics were identical in both runs (greedy decoding gives the same routing).

| configuration | decode tok/s (run 1 / 2) | prompt tok/s | cache hit rate | misses computed on CPU | uploaded to GPU |
| --- | ---: | ---: | ---: | ---: | ---: |
| static `-ncmoe 40` | 12.8 / 13.4 | 11.8 / 10.4 | | | |
| 3 GB cache, CPU share 0 | 5.1 / 5.1 | 16.0 / 15.7 | 65.7% | 0 | 45,981 MiB |
| 3 GB cache, CPU share 0.5 | 7.8 / 7.9 | 17.9 / 15.7 | 64.1% | 8,195 | 25,554 MiB |
| 3 GB cache, CPU share 0.95 | 11.9 / 12.3 | 16.6 / 15.7 | 58.7% | 18,058 | 5,701 MiB |

(The hit-rate and upload figures are for decode-sized batches, up to 8 tokens, as llama.cpp reports them.)

**What it shows:**

1. **Routing reuses experts.** A 3 GB cache holds roughly 1,100 of the model's 6,144 experts, about 18% (estimate: 3,072 MiB divided by about 2.7 MB per Q4_K expert from the config), and it served about 66% of expert uses during decode. That reuse is what a working-set policy relies on. It was measured on one prompt of 128 generated tokens, so how far it holds across conversations is open.
2. **Without a CPU share, the cache is limited by uploads.** It uploaded about 45 GB to the GPU for 128 generated tokens, roughly 350 MB per token, and decoded at 5 tok/s. Every miss copies a whole expert over PCIe from memory that is not pinned (the model is memory-mapped and larger than RAM).
3. **With a 0.95 CPU share, the cache nearly matches static offload** (12.1 against 13.1 tok/s on average), because uploads fall to 5.7 GB. Static offload is still slightly faster on this laptop, whose strong CPU suits computing experts in RAM.
4. **Prompt processing is faster through the cache** at every share (about 16 tok/s against 11).

**What it means for MaltOE (estimates to test, not results):**

- **Bytes per miss is the lever, and width is how MaltOE moves it.** A miss at one-third width would upload about one third of the bytes. At the share-0 miss rate, that is roughly 120 MB per token instead of 350 MB. Whether speed follows the bytes depends on PCIe and disk behaviour that this run did not isolate. The 2x2 (configurations C and D) measures it.
- **Residency alone does not beat static offload on this laptop.** A plain cache, tuned, roughly ties it. Any gain MaltOE shows here must come from width shrinking what moves, which is the joint-controller claim.
- **BL-2b is run with `--moe-cache-cpu-share 0.95`**, the best of the three shares tried, so that the stock cache is compared at its best.

### 6.6e Task 1.7b: how width chunks fit llama.cpp's MoE cache (code read, 2026-11-17)

I read `src/llama-moe-cache.cpp` (984 lines) at the pinned commit end to end.

**How it works.** The cache keeps, per group of layers with the same expert tensor layout, one VRAM "bank" per expert tensor (gate, up, down, or merged gate_up and down) with room for `n_slots` experts. A slot holds one expert's slice of each tensor: index `e` along the expert axis (`ne[2]`), `nb[2]` bytes, contiguous in the host tensor. After routing, the graph does a `get_rows` of a per-layer slot map (a host tensor), which makes the scheduler start a new split there and call the cache's copy callback. The callback reads the selected expert ids from the GPU (one synchronisation per MoE layer), plans hits and misses, uploads each missing expert as one contiguous copy, and writes the slot map, which then serves as the expert ids of the `mul_mat_id`. The policy is an LRU per frequency class over a combined recency-and-frequency score with a 128-token half-life. Layers are discovered from the model's `ffn_*_exps` tensors in host memory, and all per-layer tables are sized by the number of model layers.

**Chunks as extra layers (the task's original idea) would mean deep changes:** every per-layer table (clock, slot map, trace) is indexed by the model's layer number, and the callback assumes one lookup per model layer.

**Chunks as extra experts fit with almost no change.** Store each projection of a layer as one tensor of `C × n_expert` small experts ("pages"), with page `p = j·n_expert + e` holding chunk `j` of expert `e` (256 neurons): gate and up `[n_embd, 256, 384]`, down `[256, n_embd, 384]`. To the cache this is an ordinary layer with 384 experts. A width-k run turns the 8 routed expert ids into the `8·k` page ids of chunks `j < k` (one `get_rows` on a small I32 table, the op the cache itself already relies on), so only those pages are looked up, cached, and uploaded. Because the router weight of an expert is the same for all its chunks, one `mul_mat_id` per projection over the `8·k` page ids, followed by a sum over each expert's `k` partial outputs, computes the layer exactly.

This answers three open points at once:

1. **Kernel count:** one `mul_mat_id` per projection at every width, as in the stock expert, instead of the chunk loop's `k` (the suspected source of the full-width price in 6.6a).
2. **Paging:** each page is one contiguous block per tensor, so a miss is one copy. Prefix views (6.6b) could not offer that for the down projection.
3. **Cache reuse:** the cache needs no change to page by chunk.

**Measured so far, sandbox CPU only** (`chunked_moe --pages`): the pages layout equals the chunked layout exactly (difference 0 at every width, f32 and q4_K, 1 and 32 tokens), clean under AddressSanitizer, and its time scales with width like the others. CPU timing cannot show a launch-count saving; the CUDA run on the laptop decides it.

**What the cache still lacks for MaltOE (unchanged by this):**

| Needed | State at the pin | Size of the change (my judgement, not measured) |
| --- | --- | --- |
| Pages smaller than an expert | Solved by the pages layout | Loader and graph changes only |
| `n_expert_used` for sizing | Taken from the model (8); a width-k run looks up `8·k` pages per token | Small: pass the page count where the cache sizes zero slots and the minimum slot count |
| Resize while running | Not supported: banks and `n_slots` are fixed when the context is created | Largest item. Simplest version: recreate the cache at a new size at a Safe Boundary and re-upload the working set |
| Hit, miss and byte counters while running | Logged only at shutdown | Small: expose the existing counters |
| Working-set policy and overflow signal | Recency-and-frequency LRU, no signal | Moderate: a working-set victim choice inside the self-contained LRU struct, plus a counter |
| One GPU synchronisation per MoE layer | In the callback (open upstream PR #32076, "run everything without syncs") | Not MaltOE's to fix; it applies to every cache baseline equally |

**Proposed design change (not applied, for your decision):** store the Elastic Package as pages (chunk tensors concatenated along the expert axis) instead of separate `.c{j}` tensors per chunk, and compute width k as one `mul_mat_id` over `8·k` page ids. Decide after the CUDA `--pages` run.

### 6.5 Smallest engine change for Route A (Req 1.7)

llama.cpp has dynamic residency (VRAM to host) but not variable-width execution, so the change is to add width on top and extend the cache:

1. **Package loader:** read `maltoe.*` metadata and the chunk tensors for `qwen3moe`.
2. **Graph:** loop the `build_experts` lambda over chunks `j < k` and sum the down-projection outputs.
3. **Cache registration:** register each chunk's tensors with the MoE cache so chunks are cached and evicted independently. A narrower width then never uploads the wider chunks. *Largest uncertainty:* the cache is keyed by layer and by tensor layout, and I have not yet read it deeply enough to say whether chunk triples can be registered as extra logical layers without restructuring it.
4. **Runtime resize and counters:** add what upstream lacks (6.3).
5. **Control endpoint** and Safe Boundary application.

I am not giving an hours estimate; the real cost is item 3, and a short spike (read `llama-moe-cache.cpp` end to end, then register one chunked layer) would settle it. That is a proposed addition to Task 1, section 9.

### 6.6 Waiting for hardware

| Measurement | Step | Result |
| --- | --- | --- |
| Chunked experts on CUDA: correctness | A | **Passed** on the RTX 4060 Laptop GPU (see 6.6a) |
| Chunked experts on CUDA: ms per layer, k = 1..3 vs fused | A | **Measured** (see 6.6a): 1.97x decode and 1.48x prefill at full width; narrower widths scale down as expected |
| Upstream cache: decode tok/s, hit rate, bytes uploaded, disk read, for static `-ncmoe` vs cache sizes | B | **Done** (see 6.6c and 6.6d) |

## 7. G3: width quality (pending)

Method ([`spikes/width_emulation`](../spikes/width_emulation/make_variant.py)): width k keeps the k x 256 most important neurons of each expert and zeroes the gate and up rows of the rest. A zero gate row gives silu(0) = 0, so the neuron contributes nothing, which is exactly what running a narrower expert computes. No reordering of stored tensors is needed to measure quality, and the file stays valid for stock llama.cpp. Importance is sqrt(mean(h^2)) x the L2 norm of the neuron's down-projection column, using llama-imatrix statistics from a calibration text of wikitext-2 **train** and llama.cpp source code. The evaluation uses wikitext-2 **test**, HumanEval, and GSM8K, so calibration and evaluation do not overlap. Controls: random neurons at k = 1.

Deviations from tasks.md 1.5: (a) the base is Q4_K_M, not Q8_0 (finding 3); (b) no physical permutation is written, because emulation by zeroing is equivalent; (c) the smoke subset is the first 5 of a seeded 20-problem draw per set, and the same 20 will be Task 3's frozen set.

| Variant | wikitext-2 perplexity | Coding (n/5) | Reasoning (n/5) | Importance mass kept |
| --- | --- | --- | --- | --- |
| k = 3 (full) | *pending* | | | |
| k = 2 | *pending* | | | |
| k = 1 | *pending* | | | |
| k = 1, random neurons | *pending* | | | |

Five problems per set means each difference of one problem is 20 percentage points. The perplexity column is the more reliable signal; the problem counts only show gross failure.

## 8. G4 and Edge0 (pending)

| Measurement | Step | Result |
| --- | --- | --- |
| Ballast holds VRAM; behaviour when a run exceeds free VRAM (fail, or spill to shared memory), with sysmem fallback on and off | C | **Done on the laptop** (see 8.1, 8.2): no failure in any case; an active filler slows the benchmark 2.8x to 5x |
| RAM filler lowers available RAM on laptop and PC | C | **Laptop: yes** (9.88 to 6.92 GiB available with 3 GiB held, VirtualLock succeeded); PC pending |
| Edge0 builds and runs on the RTX 4060 laptop; backend; tok/s at several `expert_slots` | E | *pending* |

### 8.1 G4 on the laptop, idle filler (2026-11-17)

`llama-bench` with `-ngl 99 -ncmoe 36` (about 5.4 GB of VRAM, fixed placement; `llama-bench` does not fit itself to free memory unless `--fit-target` is given) ran next to the VRAM filler holding 0 to 4 GiB. Driver 610.62.

| filler | Driver Default: prompt / decode tok/s | Prefer No Sysmem Fallback: prompt / decode tok/s | peak GPU in use (both) |
| ---: | ---: | ---: | ---: |
| 0 GiB | 35.4 / 24.1 | 53.0 / 28.2 | 5,397 MiB |
| 1 GiB | 45.4 / 27.5 | 32.8 / 26.3 | 6,504 MiB |
| 2 GiB | 45.7 / 24.7 | 48.8 / 28.5 | 7,528 MiB |
| 3 GiB | 34.8 / 26.4 | 39.1 / 27.4 | 7,924 MiB |
| 4 GiB | 29.3 / 25.3 | 41.4 / 24.2 | 7,924 MiB |

With the 3 and 4 GiB filler the two programs asked for about 8.5 and 9.5 GB, more than the 8 GB card, yet nothing failed, decode speed did not drop, and total use stayed at about 7.9 GB (`nvidia-smi` during the overlap: 7,924 and 7,830 MiB). With the default setting, one reading showed the 4 GiB filler alone at 2,451 MiB. So up to about 1.6 GB of what was asked for was not in VRAM, and since the benchmark did not slow down, it was almost certainly the idle filler's memory that Windows moved out. That is inferred from the totals; Windows does not report which allocation it moved. The CUDA sysmem fallback setting made no difference, because it governs what CUDA does when an allocation fails, and no allocation failed. The "shared GPU memory" counter moved only 55 to 134 MiB, so it did not see the move either (it samples once a second).

**Consequences:**

1. **An idle filler does not create lasting pressure on Windows**, so it does not impose the 4 and 6 GB budgets on the baselines. A filler that keeps using its memory is being tested next (`ballast_probe ... touch_ms`). If Windows then moves the benchmark's memory instead, that is the realistic "another GPU app" case for the Live Squeeze.
2. **MaltOE's own accounting stays the primary limit** on MaltOE (design.md), and baselines will need either the active filler or a configuration that fits each budget (for example `-ncmoe` chosen per budget).
3. **For all later runs, keep "Prefer No Sysmem Fallback"** so that a CUDA allocation that cannot be met fails visibly. (Proposed; it made no difference here.)

### 8.2 G4 on the laptop, active filler (2026-11-17)

Same benchmark, "Prefer No Sysmem Fallback", the filler now rewriting all of its memory every 50 ms, as a game or another GPU program keeps its memory in use.

| active filler | prompt tok/s | decode tok/s | run time |
| ---: | ---: | ---: | ---: |
| 0 GiB | 46.9 | 27.9 | 28 s |
| 1 GiB | 48.2 | 29.1 | 32 s |
| 2 GiB | 45.7 | 29.0 | 28 s |
| 3 GiB | 16.0 | **10.1** | 69 s |
| 4 GiB | 6.4 | **5.6** | 121 s |

Once the two programs together needed more than the card holds, the benchmark slowed to 0.36x (3 GiB) and 0.20x (4 GiB) of its normal decode speed. Nothing failed: no out-of-memory error, even with the fallback setting that is supposed to prevent spilling, and total GPU use stayed at about 7.9 GB. Windows kept the active program's memory in VRAM and moved the benchmark's out, so the benchmark paged over PCIe.

**What this settles:**

1. **On Windows, running out of VRAM looks like a slowdown, not a crash.** A stock setup does not die when another program takes memory; it quietly becomes 3x to 5x slower. The handoff's Live Squeeze line ("a normal setup would die") is wrong for this machine; the honest version is "a normal setup slows to a crawl and does not know why". That is still a good demo beat, and it is measured.
2. **MaltOE cannot wait for an out-of-memory error.** The error does not come. The controller has to act on falling free VRAM, which is what design.md already specifies (sampling free VRAM at least every 250 ms; the effective budget is the smaller of the user's budget and measured free plus MaltOE's own use). Requirement 7.9 (out-of-memory recovery) still holds, but on Windows it is a rare path, not the main one.
3. **The filler must be active.** `maltoe-ballast --vram` (Task 9.1) keeps rewriting its memory; an idle filler gets moved out and imposes nothing. The same holds for any budget imposed on the baselines.
4. **A detail to measure later:** whether `cudaMemGetInfo` reports the lower free VRAM while Windows is paging, so the controller sees the pressure in time. The `nvidia-smi` totals (7,924 MiB in use) suggest yes; MaltOE's monitor should log both.

## 9. Recommendation and checkpoint

**Provisional route (valid unless G3 or G4 fail): Route A.** Convert Qwen3-30B-A3B without training (importance-ordered chunks), on a llama.cpp fork at the pinned commit, with `C = 3`.

**Approved by the builder on 2026-11-16, all of the following, and applied to requirements.md, design.md, and tasks.md:**

1. **Baselines:** add `llama.cpp --n-cpu-moe + --moe-cache-mib` (BL-2b) to Req 2.1 and Req 10.8, because it is the strongest stock setup.
2. **Pager design:** replace "write an expert pager" with "extend llama.cpp's MoE cache" (chunk-aware pages, runtime resize, counters, working-set signal). Tasks 7.1 to 7.8 change accordingly.
3. **RAM and SSD tiers for v1:** use the operating system's page cache through mmap, and measure bytes read with OS counters. Build MaltOE's own unbuffered reader only if Step B and Step C show the page cache cannot be controlled well enough. Req 6.1 (three tiers) still holds.
4. **Add a task 1.7b:** a short spike that reads `llama-moe-cache.cpp` end to end and registers one chunked layer, to settle the main uncertainty in 6.5.
5. **Importance run on Q4_K_M** (finding 3).
6. **Fallbacks:** if G3 fails at k = 1, try `C = 4` (q8_0-type chunks only) before anything else; MoSE only if code and checkpoints appear; Route B (Edge0) only if G2 on CUDA fails.
7. **Tail-skip baseline (BL-4)** (from the survey in 3.2b): compare width reduction with skipping the lowest-ranked routed experts at matching compute (Req 10.9, Task 4.6).
8. **Configuration A in two forms,** A-stock and A-chunked (Req 10.10), because they run at different speeds (6.6a).
9. **Speed objective re-check** in the real engine (Task 6.8).

**To discuss with the builder (not yet decided): demo impact.** A 30B-class MoE running on a cheap consumer GPU is already shown by others (section 3.2b; for example Qwen3.6-35B-A3B at 68 to 78 tok/s on a 6 GB laptop GPU, per that project's README), and stock `llama.cpp --n-cpu-moe` does not "crawl below 1 tok/s". Demo beat 1 as written in the handoff will not surprise a judge who knows the field. The live squeeze with a controlled quality trade (beat 2) is the part nobody else shows. Whatever replaces beat 1 should be chosen after the Step B and Step D numbers exist, so it rests on measured results.

Candidate from the builder (2026-11-16): open with the asterisk. "An 8 GB GPU can run DeepSeek V4.1*", then reveal the footnote ("* at a uselessly low tokens per second"), then "well, I won't claim that", and turn to what MaltOE does claim. It makes the honesty rules part of the pitch. Conditions: the footnote number must be a real measurement from Task 15.2 on the test laptop, or the slide must say plainly that it is a joke figure. V4.1 needs hundreds of GB of disk, and none of the handoff's V4.1 figures are verified yet.

Open items: the CRISP source code (not released) and WiSP's behaviour under an external VRAM change (unverified). The pinned commit builds on Windows with CUDA 12.9 and MSVC 19.39 (verified on the laptop, 2026-11-16).
