# Local MoE inference: landscape matrix

Built 2026-11-18 from READMEs, docs and paper abstracts found by web search. I have run none of these except llama.cpp. "Not found" means I didn't see it in what I read, not that the project lacks it. More detail on many rows is in [`feasibility-report.md`](feasibility-report.md), sections 3.2 and 3.2b. Numbers are the projects' own.

## How to read it

The columns follow MaltOE's claim, one part at a time:

- **Win:** runs natively on Windows (our platform). **Y** yes, **N** no (Linux or WSL2 only), **?** not checked.
- **Residency:** where the routed experts live and run. **CPU** = experts computed on the CPU from RAM, **cache** = a GPU cache of hot experts, **SSD** = streamed from disk.
- **Lossy knob:** what the engine can give up to go faster or fit. **none** = exact output only.
- **Knob at runtime:** whether the lossy knob changes during a session, and what drives it.
- **Live VRAM:** reacts during a session when other apps take or release VRAM.
- **Joint:** one controller that moves the lossy knob and residency together.

## Matrix

### General local engines

| Project | Win | Residency | Lossy knob | Knob at runtime | Live VRAM | Joint | Evidence |
| --- | --- | --- | --- | --- | --- | --- | --- |
| [llama.cpp](https://github.com/ggml-org/llama.cpp) (pinned `a946eb5`) | Y | CPU (`--n-cpu-moe`), cache (`--moe-cache-mib`), SSD via mmap | quantization (fixed per file) | no | no: cache size fixed at context creation | no | source read (report 6.5, 6.6e); measured on the laptop |
| [ik_llama.cpp](https://github.com/ikawrakow/ik_llama.cpp) | Y | CPU, tensor overrides | extra quant types | no | not found | no | README excerpts; one issue reports it slower than mainline on a Qwen3 MoE CPU setup |
| [Ollama](https://github.com/ollama/ollama) | Y | CPU split (automatic) | quantization | no | not found | no | ghost-infer's benchmark lists it at 26.9 tok/s on Qwen3-30B-A3B, RTX 2060 SUPER |
| [ExLlamaV3](https://github.com/turboderp-org/exllamav3) + TabbyAPI | Y | CPU MoE offload (experimental), `cpu_moe_split_experts` | EXL3 bits per weight (fixed per file) | no | not found | no | docs and user write-ups |
| [KTransformers](https://github.com/kvcache-ai/ktransformers) | ? | CPU experts with AMX/AVX kernels, GPU for the rest | quantization | no | not found | no | docs; AVX2 tutorial asks for a 24 GB+ GPU |
| [PowerInfer](https://github.com/Tiiny-AI/PowerInfer) | ? | **neuron level**: hot neurons on the GPU, cold on the CPU | none (exact) | no | not found | no | README; needs ReLU-family activations, so not Qwen3's SwiGLU |
| [vLLM](https://github.com/vllm-project/vllm) | N | expert offload with a GPU cache proposed ([issue 38256](https://github.com/vllm-project/vllm/issues/38256)) | quantization | no | not found | no | issue text |

### Consumer MoE offload engines

| Project | Win | Residency | Lossy knob | Knob at runtime | Live VRAM | Joint | Evidence |
| --- | --- | --- | --- | --- | --- | --- | --- |
| [ghost-infer](https://github.com/affifuddin-hazam/ghost-infer) | Y | cache + CPU in parallel, CUDA graph per layer | quantization; exact speculative decoding | no | not found (user setting for RAM use) | no | README. **68.7 tok/s on Qwen3-30B-A3B, 2.3x llama.cpp, RTX 2060 SUPER 8 GB** |
| [Krasis](https://github.com/brontoguana/krasis) | Y (installer) | hot/cold expert residency ("HCS") between VRAM and RAM | INT4/INT8 experts, KV formats (fixed per run) | no | **yes, partly:** "pressure detection, idle pressure drain", hard exit before OOM | no | README |
| [FreeToken](https://github.com/attentiondotnet/FreeToken) (Berkeley/MIT, [arXiv 2608.16157](https://arxiv.org/abs/2608.16157)) | Y (desktop app) | global LRU expert cache, bandwidth-adaptive CPU-GPU co-execution | quant formats (MXFP4, NVFP4, FP8) | no | **partly:** "runtime VRAM re-allocation between expert caches and KV memory without engine restarts". Whether it reacts to other apps: unverified | no | README and abstract. Runs on 8 GB laptop GPUs |
| [colibri](https://github.com/JustVugg/colibri) | ? | GPU / RAM / SSD tiers, planner at setup | "quality-bounded" formats | no | at setup only | no | README |
| [bnk](https://github.com/christ-pher/bnk) | ? | tiered, V100 + RAM | none | no | **yes:** re-reads free VRAM, shrinks the expert cache | no | README |
| [moe-l2](https://github.com/yalun753/moe-l2) | ? | expert cache for 8 GB cards | ? | ? | possibly ("soft_resize", unexplained) | no | changelog |
| [moe-autopilot](https://github.com/JigSawPT/moe-autopilot) | ? | static hot-expert copy in VRAM on top of `--n-cpu-moe` | none | no (fixed at load) | no | no | README; +26% decode on Qwen3-Coder-Next 80B, RTX 5090 |
| [fomoe](https://github.com/pmerolla/fomoe) | ? | multi-tier cache | **expert substitution** (cached expert replaces an uncached one) | threshold set by the user | not found | partly: the knob depends on what is cached | README |
| [Strata](https://github.com/Niko1221/Strata) / [Lamina](https://github.com/eddoursul/Lamina) / [tandem.cpp](https://github.com/ewhacc/tandem.cpp) / [neerajdad123-byte/moe](https://github.com/neerajdad123-byte/moe) | ? | offload + cache | **tail-expert skip** (`--fast`) | no (on/off) | not found | no | READMEs |
| [hotpin-llm](https://github.com/LozzKappa/hotpin-llm) | ? | routing-guided pinning | none | no | not found | no | README |
| [inferssd](https://github.com/wici-ai/inferssd), [pulsar](https://github.com/giannisanni/pulsar), [moe-ssd-streaming-windows](https://github.com/tonbistudio/moe-ssd-streaming-windows), [flash-moe](https://github.com/danveloper/flash-moe) (Mac), [MoE-Infinity](https://github.com/EfficientMoE/MoE-Infinity) | mixed | SSD streaming + cache | none or quantization | no | not found | no | READMEs |
| Edge0 | Y | SSD streaming, trained routing predictor, fixed slots | none | no | no | no | README (report 3.2) |

### Research systems with a runtime knob

| Work | Knob | Driven by | Residency | Live VRAM | Code | Evidence |
| --- | --- | --- | --- | --- | --- | --- |
| **[MatMoE](https://arxiv.org/abs/2611.01704)** (PKU, ICCAD 2026, 1 Nov 2026) | **nested ("Matryoshka") precision per expert**, chosen per token from routing scores | a **target average bit-width set to meet a changing latency limit** ("fluctuating system utilization") | fixed memory budget, set at build | not found | built on llama.cpp; release not checked | abstract and method section. RTX 4090, Jetson, Raspberry Pi 5 |
| [CRISP](https://arxiv.org/abs/2610.37442) | **importance-ordered expert width** | offline tiers of expert combinations | width-aware loading scheduler | no: budget is a per-run parameter | not released | full text, keyword reading |
| [MoSE](https://arxiv.org/abs/2602.06154) | **expert width** (trained slimmable experts) | router confidence, under a fixed budget | not found | no | not found | abstract |
| [DyMoE](https://arxiv.org/abs/2603.19172) | precision per expert (8/4/0 bits) | token and gate importance | mixed-precision VRAM cache | no | ? | full text, keyword reading. Tested on Qwen3-30B-A3B |
| [M-MoE](https://arxiv.org/abs/2509.26520) | number of active experts (trained) | chosen at inference | none | no | ? | abstract |
| [WiSP](https://arxiv.org/abs/2606.21868) | none (byte-identical) | n/a | working-set pager, live split between experts and KV | total budget held fixed; external change unverified | [yes](https://github.com/nokia-applied-research/WiSP) (vLLM) | full text, keyword reading |

### MaltOE (planned)

| Project | Win | Residency | Lossy knob | Knob at runtime | Live VRAM | Joint |
| --- | --- | --- | --- | --- | --- | --- |
| MaltOE | Y | llama.cpp MoE cache, extended with chunk pages and resize | **expert width without training** (importance-ordered thirds) | per session, and at Safe Boundaries under memory pressure | planned | **planned:** width and residency moved together, steered by a user Objective |

## What the matrix says

1. **Speed is taken.** ghost-infer, Krasis and FreeToken are purpose-built, Windows-ready and fast. ghost-infer reports 2.3x llama.cpp on our exact model. MaltOE, as a llama.cpp fork, should never be pitched on tok/s.
2. **Reacting to VRAM pressure is partly taken, on the residency side.** bnk shrinks its cache when free VRAM drops. Krasis detects pressure and drains. FreeToken moves VRAM between expert cache and KV at runtime. None of them trades quality to cope; they only move experts.
3. **A runtime lossy knob driven by a changing system condition is taken too, for precision and latency.** MatMoE (published 1 Nov 2026) does nested precision per expert on llama.cpp, tuned live to a latency target. That is the closest prior art found so far, and it is new since our novelty check. It also covers our "precision skin" stretch idea, which should be dropped or cited as MatMoE's.
4. **Width as a runtime knob without training has no public engine.** CRISP has it in a paper with no code; MoSE needs training.
5. **What remains for MaltOE:** a width knob without training, plus residency, moved together by one controller when *other apps* change the VRAM budget during a session, with the quality cost shown to the user. I found that combination nowhere. It is narrower than before, and the README must cite MatMoE, FreeToken, Krasis and bnk next to WiSP and CRISP.

## What to test before the pitch is fixed

- **ghost-infer, Krasis and FreeToken under the Step C active VRAM filler** on the laptop, with the same model and filler settings. If any of them stays fast without losing quality, the live-squeeze demo loses its point and the pitch must change. All three run on Windows, so this needs no new setup beyond installing them.
- **MatMoE's code:** whether it is released, and whether its knob responds to memory as well as latency. If it does, the claim narrows again.
