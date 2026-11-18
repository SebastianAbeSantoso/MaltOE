# Implementation Plan — MaltOE (codename ELASTIC)

## How to read this plan

- Tasks run in order. Task 1 is a gate: no runtime code (Tasks 3 onward) starts until the Feasibility Report is written and the builder approves the route.
- **[HW-L]** means the step runs on the laptop (RTX 4060 8 GB, i9-14980HX, 16 GB RAM, Windows). **[HW-PC]** means the PC (RX 6650 XT 8 GB, i3-12100F, 32 GB RAM). The Kiro sandbox has no GPU. For these steps Kiro prepares the script and exact commands, the builder runs them and pastes the output, and Kiro records the results.
- **Checkpoint** marks a stop for builder review.
- No numeric threshold appears in this plan. Thresholds come from the Baseline Report (Task 4) or from the builder's judgement at a checkpoint.
- Every third-party claim recorded in a task must link its primary source or be marked *unverified*.

## Pre-flight (admin, not coding; do early)

- [x] P1. Request the Kiroween Kiro credits before December 1, 2026, 12:00 PM PT (December 2, 04:00 WITA). _Requirements: 15.5_ (done; confirmed by the builder 2026-11-16)
- [x] P2. Make the GitHub repo `SebastianAbeSantoso/MaltOE` public, or confirm it will be public before submission. Confirm `.kiro/` is committed and not in `.gitignore`. _Requirements: 15.1_ (done 2026-11-17: GitHub API reports `visibility: public`; `.kiro/` hooks, specs and steering are tracked on `main` and on the open branches; `.gitignore` has no `.kiro` entry and `git check-ignore` matches nothing. `main` still holds the spec under `.kiro/specs/molt/` until PRs #2–#4 are merged.)
- [x] P3. Re-check the official Kiroween 2026 rules on Devpost. Record any difference from the handoff dates in `docs/kiroween-rules.md`. _Requirements: 15.4, 15.6_ (done 2026-11-17 from the Devpost overview page and the Official Rules, pasted by the builder: no date differences. Submission rules not yet in Req 15, and an open question about trademarks in the video, are listed in `docs/kiroween-rules.md`. The builder confirms their own eligibility.)

## Tasks

- [ ] 1. Model Feasibility Round (gates G0–G4)
  - Output: `docs/feasibility-report.md`, with one section per gate, a candidate matrix, a recommended route, and every unverified claim marked.
  - **Status (2026-11-16):** 1.1 to 1.4 are done. For 1.5 to 1.9 the scripts are written and tested in the sandbox; the hardware runs are yours, in the order given in `spikes/README.md`. 1.10 waits for those results. The report lists proposed spec changes that need your approval at the 1.10 checkpoint.
  - _Requirements: 1.1–1.9_

  - [x] 1.1 G0 Novelty: read the full CRISP paper and the related work
    - Read CRISP (arXiv 2610.37442) in full. Answer: does CRISP change width or loading schedule *online* when the memory budget changes during serving? Does it have a user-facing objective? Is it single-user consumer hardware or edge servers? Which models and hardware did it use?
    - Re-read WiSP (arXiv 2606.21868) on MV-WSA: what exactly does its live controller resize, and does it ever change computation?
    - Verify DyMoE (arXiv 2603.19172) and say what it adapts (precision, orchestration) and whether it is online.
    - Verify the handoff's figures for CRISP, D2MoE, and MC-MoE against their papers, and record links.
    - Write the resulting novelty claim. If any work already does the narrowed claim, narrow it again and stop for the builder.
    - _Requirements: 1.6, 1.8, 12.1–12.5_

  - [x] 1.2 G1 Model: verify Qwen3-30B-A3B
    - Record the Hugging Face repo, licence, and `config.json` values (layers, hidden size, experts, top-K, expert intermediate size, shared experts, attention heads). Compare them with the table in design.md.
    - Find an existing Q8_0 GGUF, or confirm that `convert_hf_to_gguf.py` at the pinned commit supports the `qwen3moe` arch. Record file sizes for Q8_0 and Q4_K_M.
    - Check whether "Qwen3.6-35B-A3B" exists as a public checkpoint. Record the result only; do not plan around it.
    - _Requirements: 1.3, 1.9_

  - [x] 1.3 Candidate matrix: MoSE, HydraLLM, Edge0, llama.cpp MoE offload
    - For each candidate, record the repo link, licence, checkpoint availability and quality evidence, and nested-width support (existing / needs conversion / not feasible).
    - For each engine, record whether it can run variable-width experts and whether it supports dynamic expert placement across VRAM, RAM, and SSD.
    - MoSE: are trained checkpoints released, at what sizes, and under what licence? This is the G3 fallback only.
    - Edge0: confirm the Windows backends (Vulkan/CPU), the licence, and the supported model list for `edge0-infer`.
    - _Requirements: 1.2–1.4, 1.6_

  - [x] 1.4 llama.cpp: pin a commit and answer the existing-feature questions (pinned `a946eb54…`; that it builds with CUDA on Windows is checked in Step 0 of `spikes/README.md`)
    - Pick a recent llama.cpp commit that builds with CUDA on Windows, and record it in the report. It becomes `engine/LLAMA_CPP_COMMIT` in Task 2.
    - Does the backend scheduler already copy only the *used* experts when host-resident MoE weights are offloaded for a GPU `MUL_MAT_ID`? Cite the code path or PR.
    - Does `llama-imatrix` record per-expert input-activation statistics for `ffn_down_exps`, in a form that gives per-neuron `mean(h²)` per expert?
    - Is `llama_memory_seq_rm` (or its equivalent at this commit) available for removing a partial KV position?
    - _Requirements: 1.4, 1.7_

  - [ ] 1.5 G3 Width quality, emulated without engine changes [HW-L] (scripts ready: `spikes/width_emulation/`, Step D) Importance statistics come from the Q4_K_M file (approved change 5).
    - Write `spikes/width_emulation.py` using `gguf-py`. It loads a Q8_0 GGUF, computes importance per expert neuron from the imatrix and down-projection column norms, and permutes neurons by importance. For each `k ∈ {1, 2}` it zeroes the gate and up rows of neurons beyond `k·256`, which exactly emulates width `k` because `silu(0)·x = 0`. It writes one GGUF per `k`.
    - Control: also write a `k = 1` model that keeps a *random* 256 neurons instead of the top 256. This shows whether importance ordering matters.
    - The builder runs stock `llama-perplexity` and a 5-coding + 5-reasoning smoke subset of the eval set at k = 3, 2, 1, and the random k = 1 control.
    - Record perplexity and solved counts. **Checkpoint:** the builder judges whether `k = 1` and `k = 2` are useful. If not, record the C = 4 fallback plan; do not run it unless asked.
    - _Requirements: 1.5, 4.2, 11.3_

  - [x] 1.6 G2a Engine spike: chunked `mul_mat_id` on CUDA [HW-L] (CPU run done in the sandbox: correct for f32, f16, q8_0, q4_K. CUDA run done by the builder 2026-11-16: correct on the RTX 4060; timed with the fixed method: full-width chunked is 1.97x the fused expert in decode and 1.48x in prefill; see report 6.6a)
    - Write a small standalone C++ test against the pinned ggml in `spikes/chunked_moe/`. It builds an expert FFN as 3 chunks (`[n_embd, 256, n_expert]` gate/up, `[256, n_embd, n_expert]` down) in Q4_K, runs `mul_mat_id` on CUDA for k = 1..3, and compares with an f32 reference.
    - Record correctness and per-layer time per `k` against one fused `mul_mat_id` at full width.
    - _Requirements: 1.4, 1.7_

  - [x] 1.6b Run `chunked_moe --views` on CUDA to see whether width k can be a prefix view of the fused tensor with no chunk loop. Done 2026-11-17: correct on CUDA; prefill 0.44x / 0.76x / 1.08x of stock at k = 1 / 2 / 3; decode not settled by the spike (report 6.6b).
  - [x] 1.7 G2b Engine spike: per-MoE-layer host split for slot remapping [HW-L] (**changed from plan**: llama.cpp at the pin already has `--moe-cache-mib`, so Step B measures that instead of a hand-written split; see report section 6) Done 2026-11-17: hit rate 66% with a 3 GB cache; without a CPU share the cache uploads about 350 MB per token and decodes at 5 tok/s; with share 0.95 it nearly ties static offload (report 6.6c, 6.6d).
    - Extend the spike. After top-K, force a CPU op that reads the expert ids, remaps them to slot ids in a small slot pool, and copies one missing page per layer from pinned host memory. Run it for 48 layers.
    - Measure the per-token overhead of the 48 splits and syncs with no misses, and the cost per missed page, at k = 1..3.
    - Compare with stock `--n-cpu-moe` decode tok/s on the real model at the same budget.
    - **Checkpoint:** the builder decides whether the overhead is acceptable. If not, evaluate Route B (Edge0-infer fork) in 1.9 before deciding.
    - _Requirements: 1.4, 1.7_

  - [x] 1.8 G4 Budget enforcement on Windows [HW-L] (done on the laptop 2026-11-17: idle filler is moved out, active filler slows the benchmark 2.8x to 5x with no error, RAM filler works; PC RAM check still open; report 8.1, 8.2) (probes ready: `spikes/ballast_probe/`, Step C)
    - Write `spikes/ballast_probe.cu`: allocate and touch N GB of VRAM, then hold it.
    - With the probe holding 2 and 4 GB, run stock llama.cpp at full offload. Record whether it fails, falls back to shared system memory, or the probe's memory gets demoted. Use Task Manager "Dedicated/Shared GPU memory" and NVML.
    - Record the NVIDIA "CUDA – Sysmem Fallback Policy" setting and its effect. Pick the setting used for all runs.
    - Check the RAM filler approach (`VirtualAlloc` + `VirtualLock` with a raised working-set limit) on the PC [HW-PC], and record whether it reduces available RAM as seen by `psutil`.
    - _Requirements: 2.4, 2.6, 7.8_

  - [ ] 1.9 Edge0 reference run, and Route B check if needed [HW-L] (steps: `spikes/edge0/README.md`, Step E)
    - Build `edge0-infer` for Windows and try to run `edge0-35b` on the laptop. Record the backend used, whether it ran, TTFT, tok/s, and the memory it reports.
    - Only if G2 failed: assess how width chunks would fit Edge0's slot design and its Recover-LoRA adapters, and estimate the effort.
    - _Requirements: 1.3, 2.1_

  - [x] 1.7b Read `src/llama-moe-cache.cpp` end to end and register one chunked layer with the cache, to settle whether chunks can be cached as extra logical layers (report section 6.5). Done 2026-11-17: chunks fit as extra experts ("pages") with almost no cache change, not as extra layers; remaining gaps are resize, live counters, and the working-set policy (report 6.6e).
  - [ ] 1.7c [HW-L] Run `chunked_moe --pages` on CUDA (decode and prefill) to see whether the pages layout removes the chunking price (report 6.6e).
  - [ ] 1.11 Theory checks: test the two papers' own predictions on our data (sources and labels: `docs/theory-sources.md`)
    - [ ] 1.11a Routing-trace tool: record the experts each token selects, per layer, through llama.cpp's eval callback, without patching the engine. Test on the CPU in the sandbox with a small MoE model.
    - [ ] 1.11b [HW-L] Record one trace of Qwen3-30B-A3B on the frozen coding and reasoning prompts (about 20 minutes).
    - [ ] 1.11c Denning P1, in the sandbox: working-set size ω(τ) in pages and bytes at k = 3, 2, 1. Check "monotonically increasing and concave downward".
    - [ ] 1.11d Denning P3 and eq. 7, in the sandbox: the reentry rate from the trace against measured upload bytes per token (report 6.6d), and e(m) = 1/(1 + mT) with the measured miss rate and page time against measured tok/s.
    - [ ] 1.11e Working-set eviction against llama.cpp's frequency-and-recency policy on the same trace, at matched cache sizes (Denning, FJCC 1968, on global policies).
    - [ ] 1.11f Liu monotone and error: perplexity against k and importance mass kept (from G3); whether the importance-ordered k = 1 beats random k = 1; the shape of the error curve (convex or concave).
    - Report each check as holds, fails, or not tested. A failed prediction is reported, not hidden.
    - _Requirements: 12.3, 12.5_
  - [ ] 1.10 Write the Feasibility Report and recommend a route
    - Give pass or fail per gate, with evidence. Name one route and one model. State the chosen engine (A or B) and why. Give the final novelty claim. List open risks.
    - **Checkpoint:** the builder approves the route before Task 2.
    - _Requirements: 1.1, 1.5, 1.7_

- [x] 2. Repository scaffolding
  - [x] 2.1 Create the layout from `structure.md`: `maltoe/`, `engine/maltoe/`, `engine/patches/`, `engine/LLAMA_CPP_COMMIT`, `tools/ballast/`, `convert/`, `harness/`, `web/`, `tests/`, `docs/`.
  - [x] 2.2 Set up the Python project with `uv`, Python 3.12, FastAPI, pydantic, pytest, hypothesis, ruff, and `pyproject.toml` with the ruff and pytest config.
  - [x] 2.3 Add an engine build script that fetches llama.cpp at the pinned commit, applies `engine/patches/*.patch`, adds `engine/maltoe/` to the build, and builds with CUDA (Windows) or CPU (CI). (done and verified by a real CPU build of the pinned commit. The step "adds `engine/maltoe/` to the build" is deferred to the first patch in Task 6: there is no C++ in `engine/maltoe/` yet to add.)
  - [x] 2.4 Set up the web app with Vite, React, TypeScript, vitest, and eslint. FastAPI serves the build output as static files.
  - [x] 2.5 Add GitHub Actions CI: ruff and pytest; a CPU-only engine build and CTest; web lint and vitest.
  - _Requirements: 15.1_

- [x] 3. Fixed evaluation set and machine manifest
  - [x] 3.1 Freeze the eval set: 20 HumanEval ids and 20 GSM8K test ids from a seeded draw, plus 64 windows of 512 tokens from the wikitext-2 test split. Store them in `harness/evalset/` with source links and licences.
  - [x] 3.2 Write the scorers: sandboxed HumanEval execution with a timeout (pass@1), GSM8K final-number exact match, and the perplexity driver. Report solved counts (n/20), not only percentages.
  - [x] 3.3 Fix the sampling settings: greedy, fixed `max_tokens`, fixed seed, Qwen3 chat template with thinking off. Store them in `harness/evalset/sampling.json`.
  - [x] 3.4 Write the machine manifest collector: GPU name and power limit, CPU model and threads, power mode, total and available RAM, SSD model and interface, OS and driver versions, engine version.
  - [x] 3.5 Unit tests for the scorers on fixed sample outputs.
  - _Requirements: 2.3, 2.7, 10.4, 10.7_

- [ ] 4. Baselines [HW-L, then HW-PC for BL-2 and BL-3 only]
  - [ ] 4.1 Write `StockLlamaCppAdapter` and `Edge0Adapter`. Both start a server process, stream generation, and expose perplexity where the engine supports it.
  - [ ] 4.2 Write the harness runner: config matrix (4/6/8 GB VRAM budget via the VRAM probe from 1.8, 2K/4K/8K context, RAM Budget only on the PC), per-run metrics, process I/O counters for SSD bytes, peak VRAM/RAM, minimum available RAM, OOM detection. Output CSV and JSON.
  - [ ] 4.3 Run BL-1 (full offload where it fits), BL-2 (`--n-cpu-moe` with the smallest N that fits), BL-2b (`--n-cpu-moe` plus `--moe-cache-mib` with `--moe-cache-cpu-share 0.95`, the best share found in Step B2; cache sized to each budget), and BL-3 (Edge0, reference point) on the laptop. Label BL-2 and BL-2b with the CPU.
  - [ ] 4.4 Write `docs/baseline-report.md`: performance envelope, OOM cases, the statement that 4/6 GB budgets simulate memory only, and machine manifests.
  - [ ] 4.5 **Checkpoint:** the builder sets the thresholds the later tasks need: the usefulness bar for Req 11.3 and `D_bal` for the Balanced objective. Record them in the Baseline Report.
  - [ ] 4.6 Tail-skip baseline (BL-4): choose how to run it, either an existing engine's mode (Strata tail-skip, `tandem.cpp --fast`) after checking its licence and that it runs this model, or a small llama.cpp change that computes only the top `m` routed experts. Run it at `m` matching each width's active expert parameters, on the frozen evaluation set.
  - _Requirements: 2.1–2.7, 10.8_

- [ ] 5. Elastic Package conversion pipeline
  - [ ] 5.1 `convert/importance.py`: per-expert neuron importance from imatrix statistics and down-column norms. Promote the logic from spike 1.5.
  - [ ] 5.2 `convert/build_package.py`: permute, split into C = 3 chunks (`blk.{L}.ffn_{gate,up,down}_exps.c{j}`), copy non-expert tensors, write `maltoe.*` metadata, and quantize with `llama-quantize` using the imatrix.
  - [ ] 5.3 `maltoe/package.py`: package validator that checks every metadata rule in design.md and names the bad key or tensor on failure.
  - [ ] 5.4 Validator tests on synthetic GGUF headers (missing chunk, wrong chunk size, top-K mismatch).
  - [ ] 5.5 Build the Qwen3-30B-A3B package [HW-L or a machine with enough disk]. Record size, calibration data hash, and conversion time.
  - _Requirements: 4.1–4.4_

- [ ] 6. maltoe-engine: width control (enables configs A and B)
  - [ ] 6.1 Patch the loader to recognise `maltoe.*` keys and chunk tensors for `qwen3moe`.
  - [ ] 6.2 Patch the graph builder to replace the fused MoE FFN with the chunk loop over `j < k`, applying router weights once after the chunk sum.
  - [ ] 6.3 Add the control endpoint (`/maltoe/config`, `/maltoe/state`, `/maltoe/telemetry`) and the `PendingConfig` atomic swap, applied at Safe Boundaries (between decode steps, between prefill micro-batches, prefill to decode). Emit `config_applied`.
  - [ ] 6.4 Add per-token telemetry: latency, VRAM used, k.
  - [ ] 6.5 Engine test: chunk-loop output equals the f32 prefix-width reference for k = 1..3, and k = 3 matches the original fused expert within quant tolerance (Property 10).
  - [ ] 6.6 Engine test: `k` changes only inside boundary-apply (Property 4).
  - [ ] 6.7 [HW-L] Check that full-width perplexity of the package matches the original model within noise, and run the perplexity smoke test at k = 1, 2.
  - [ ] 6.8 [HW-L] In the real engine with CUDA graphs on, time each width against A-stock in decode and prefill. If `k = 2` is not faster than A-stock in decode, revise the Speed objective's rationale in design.md before the write-up (Task 1 checkpoint).
  - _Requirements: 4.4, 5.1, 5.2, 5.4, 5.5, 7.3_

- [ ] 7. maltoe-engine: residency through llama.cpp's MoE cache (enables configs C and D)
  - [ ] 7.1 Extend llama.cpp's MoE cache to pages `(L, e, j)`: one slot per chunk, so a width-`k` run loads only chunks `j < k` (design: Expert Pager). Depends on the outcome of Task 1.7b.
  - [ ] 7.2 Runtime resize of the cache's slot banks, shrinking first so peak memory never rises; upstream fixes the size when the context is created.
  - [ ] 7.3 Working-set tracker: per-layer ring buffer over `W` tokens, `|WS_L|` at several windows, `ws_overflow` signal.
  - [ ] 7.4 Eviction: non-working-set pages first (LRU), working-set pages only after `ws_overflow`. Count thrash events.
  - [ ] 7.5 *(only if Steps B and C show it is needed)* MaltOE's own RAM cache of pinned host buffers, capped by the RAM Budget, with working-set-then-LRU over `W_ram`. In v1 the OS page cache is the RAM tier.
  - [ ] 7.6 *(only if Steps B and C show it is needed)* Unbuffered aligned reads from the package file (`FILE_FLAG_NO_BUFFERING`). In v1 disk traffic is measured with OS counters.
  - [ ] 7.7 Counters API: per-tier hits, misses, loads, evictions, bytes, and miss stall time, readable while running (upstream only logs them at shutdown).
  - [ ] 7.8 Prefetch toggle, off by default: next-layer router guess, lower priority than demand loads.
  - [ ] 7.9 OOM inside a step: abort the step, remove the partial KV position, emit `oom`, wait for a new config.
  - [ ] 7.10 Engine tests on CPU: eviction order (Property 3), working set kept while it fits (Property 2), at most `k` pages per miss (Property 9).
  - _Requirements: 6.1–6.9, 7.9_

- [ ] 8. maltoe-core: models, profiler, adapters, controller
  - [ ] 8.1 Data models from design.md (`EngineConfig`, `PendingConfig`, `TokenStats`, `Decision`, `Objective`). No token ids, text, or logits in any controller input type.
  - [ ] 8.2 Hardware profiler: engine `--probe`, NVML, `psutil` (total and available RAM), disk free, CPU info, conservative fallbacks marked "assumed".
  - [ ] 8.3 `SimEngineAdapter`: replays a recorded routing trace and models page sizes, slots, PCIe/SSD bandwidth, and per-chunk compute.
  - [ ] 8.4 `MaltOELlamaCppAdapter`: process start and stop, generate, post_config, telemetry, perplexity, health.
  - [ ] 8.5 Memory model: `need(k, slots)` from engine-reported allocations, `effective_budget`.
  - [ ] 8.6 Joint Controller, session start: startup probe at k = 1..3, Objective rules (Quality / Balanced with `D_bal` / Speed capped at C − 1), refusal when nothing fits.
  - [ ] 8.7 Joint Controller, mid-session: the narrowing sequence (shrink cold slots → narrow k → trim to minimum → stop and report), widening with `h` and `T_hold`, memory-pressure signals only.
  - [ ] 8.8 Runtime Monitor (250 ms frames) and JSON-lines decision log.
  - [ ] 8.10 Overload scheme setting (Req 7.11): Responsive (default) and Guaranteed (apply PendingConfig at the next request boundary; mark the reply over budget; fall back to Responsive on OOM). Log the scheme in every decision.
  - [ ] 8.9 Property tests (hypothesis) on `SimEngineAdapter` for Properties 1, 5, 6, 7, and 8.
  - _Requirements: 3.1–3.5, 5.1–5.7, 7.1–7.8, 7.10, 7.11_

- [ ] 9. Fillers: `maltoe-ballast`
  - [ ] 9.1 `--vram` (CUDA): allocate in 256 MB blocks, keep rewriting the memory every 50 ms (an idle filler is moved out of VRAM by Windows; report 8.1), resize on command over a local control port. Promote from spike 1.8.
  - [ ] 9.2 `--ram`: `VirtualAlloc` + `VirtualLock`, resize on command.
  - [ ] 9.3 Harness and API integration: set filler size per run and for the squeeze schedule. Mark runs invalid if the filler cannot reach its target.
  - _Requirements: 2.2, 2.6, 7.4_

- [ ] 10. OOM and crash recovery
  - [ ] 10.1 Conversation store in `maltoe-core`, independent of the engine process.
  - [ ] 10.2 Restart path: on engine exit, restart at the next narrower config, re-run prefill on history, resume streaming. If the narrowest fails, show the cause and keep history.
  - [ ] 10.3 Test on `SimEngineAdapter` with injected OOM and crash: history unchanged (Property 11).
  - _Requirements: 7.9, 8.6_

- [ ] 11. API and web UI
  - [ ] 11.1 FastAPI: `/api/chat` (SSE), `/api/session` (Objective, budgets, prefetch, thinking toggle default off), `/api/squeeze`, `/api/profile`, `/ws/telemetry`. Bind to 127.0.0.1 only.
  - [ ] 11.2 Chat panel: streaming, Objective selector, VRAM budget slider, thinking toggle.
  - [ ] 11.3 Dashboard: VRAM gauge with budget marker and external usage, 48 × 128 expert heatmap (tier, chunks resident, working-set outline), skin indicator, tok/s and TTFT tiles, decision feed in plain words, working-set curve, "stale" indicator.
  - [ ] 11.4 Offline check: no external assets, bundled fonts and scripts.
  - [ ] 11.5 Component tests (vitest) rendering from recorded telemetry.
  - _Requirements: 8.1–8.6, 9.1–9.6_

- [ ] 12. Evaluation: 2x2, baselines, squeeze [HW-L]
  - [ ] 12.1 Add configs A–D to the harness runner (A in two forms: A-stock, the unmodified fused expert, and A-chunked, `k = 3` with residency off; B: adaptive width, residency off; C: `k = 3`, residency on; D: both), plus BL-2b and the tail-skip baseline BL-4.
  - [ ] 12.2 Run the full matrix at 4/6/8 GB × 2K/4K/8K with the fixed eval set. Re-run the baselines at the same commit.
  - [ ] 12.3 Run prefetch on versus off for configs C and D.
  - [ ] 12.4 Run the squeeze scenario (8 → 6 → 4 → 8 GB) on a fixed conversation. Record survival, stall at each change, and tok/s and quality per phase.
  - [ ] 12.5 Report analytic active parameters and FLOPs per token per `k`, checked against measured expert time (Req 11.3).
  - [ ] 12.6 Write `docs/results/` with the performance envelope, solved counts (n/20), and rows where D does not beat B or C shown alongside the rest.
  - [ ] 12.7 **Checkpoint:** the builder reviews against Req 11 success criteria. If D does not beat B or C, the write-up says so and does not claim the advantage.
  - _Requirements: 10.1–10.8, 11.1–11.5_

- [ ] 13. Write-up
  - [ ] 13.1 README: the problem, the demo task, how to run it, and the narrowed novelty claim exactly as approved in Task 1.
  - [ ] 13.2 Prior art section citing WiSP, CRISP, Edge0, Krasis, MoSE, HydraLLM, D2MoE, MC-MoE, DyMoE, Yao et al., and DATS, with links. Every figure verified.
  - [ ] 13.3 "Old ideas in the code" section: link Liu et al. (1991) to the mandatory/optional scheduler and Denning (1968) to the working-set tracker, with code links.
  - [ ] 13.4 Limitations: memory-only tier simulation, small eval set variance, mixed-width KV after a width change, laptop GPU variant, CPU dependence of BL-2.
  - [ ] 13.5 Platform statement in the README: Windows 11, NVIDIA 8 GB, CUDA. List only tested platforms as supported (Req 15.7).
  - [ ] 13.6 Disclosure section in the README: pre-existing code and work with licenses (llama.cpp at the pinned commit, the model, eval datasets, main libraries, the handoff document) (Req 15.9).
  - [ ] 13.7 Finish the Kiro write-up from the running draft in `docs/submission/kiro-writeup.md` (Req 15.11). The draft is updated at every checkpoint from the start, not only here.
  - [ ] 13.8 Text description for the submission form: features and functionality, intended platform, and how to test (Req 15.7, 15.8).
  - _Requirements: 12.1–12.7, 15.7, 15.9, 15.11_

- [ ] 14. Stretch: AMD on the PC [HW-PC]
  - [ ] 14.1 Build the engine with the Vulkan backend and run the same package. Add a Vulkan path to `maltoe-ballast --vram`.
  - [ ] 14.2 Run the 2x2 at one budget with the RAM filler on, and record which requirements pass and fail. Do not block the submission on this.
  - _Requirements: 13.1–13.3_

- [ ] 15. Stretch: beyond v1
  - [ ] 15.1 Precision skins: assess nested quantization within the chunk layout. Design note only, unless time remains.
  - [ ] 15.2 DeepSeek V4.1 Flash: only after Req 11.1–11.4 pass. Re-verify all V4.1 figures, assess disk/RAM needs against the test machines, and run as a closing beat if feasible.
  - _Requirements: 4.5, 14.1–14.3_

- [ ] 16. Demo and submission
  - [ ] 16.1 Script the demo: Impossible Model (BL-1 fails or crawls, then MaltOE runs), Live Squeeze (8 → 6 → 4 GB mid-conversation), quick quality check. Open with the coding task, not a graph.
  - [ ] 16.2 Record the video on the RTX 4060 laptop, under 3 minutes, with no copyrighted music and no third-party logos. Follow the answer to the trademark question in `docs/kiroween-rules.md`. Upload to YouTube as public (Req 15.3, 15.10).
  - [ ] 16.3 Publish the test build: a GitHub Release with the Windows build and Quickstart. Test the install from the release on a clean folder on the laptop (Req 15.7, 15.8).
  - [ ] 16.4 Fill in the submission form: Resurrection, Best Student Project: Resurrection with the student details, any optional bonus URLs, the YouTube link, the repo and release URLs, the text description and the Kiro write-up (Req 15.12).
  - [ ] 16.5 Final compliance check against `docs/submission/deliverables.md`: public repo, MIT license detected, `.kiro/` committed on `main`, video link, release link, submission before December 4, 2026, 2:00 PM PT. Tag the submitted commit (Req 15.8).
  - [ ] 16.6 After submitting: keep the repo and release available until January 29, 2027, 1:00 PM PT, and answer any proof-of-enrollment request in time.
  - _Requirements: 15.1–15.4, 15.7–15.13_
