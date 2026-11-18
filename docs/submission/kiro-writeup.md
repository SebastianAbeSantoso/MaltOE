# How Kiro was used (running draft)

Running draft of the Kiro write-up (Req 15.11, deliverable D11). Kiro adds to it at every checkpoint and the builder edits it into the final version (Task 13.7). Only things that actually happened go here. Anything planned is marked **planned**. Unverified items are marked **check**.

Last updated: 2026-11-17.

## Kiro clients

- **Kiro Web, Vibe mode** for everything so far: spec writing, research, code, spikes and the feasibility report. Kiro worked in a Linux sandbox with no GPU. It wrote and tested scripts there, then the builder ran the GPU steps on the Windows laptop and pasted the output back. Kiro parsed it, recorded it in `docs/feasibility-report.md`, and pushed.
- Kiro Web vs Kiro Crew: early on, the builder asked Kiro which to use. Kiro answered from the Kiro docs (introspect sub-agent) and recommended Web. The reason was that one builder steering one agent fit a project that alternates between sandbox work and hardware runs.
- Kiro IDE and CLI: not used so far.

## Spec-driven development

- One spec in `.kiro/specs/maltoe/`: `requirements.md` (EARS-style acceptance criteria) → `design.md` → `tasks.md`.
- Requirements were settled in rounds of questions and answers before any design. The answers are recorded as "Resolved Decisions" in the spec, so later sessions don't re-open them.
- **The spec has a gate before any runtime code.** Task 1 is a Feasibility Round with gates G0–G4: model, engine, chunked-expert speed, width quality, and VRAM pressure. Tasks 4 and later wait until the route is approved at Task 1.10. The gates changed the plan:
  - The B2 cache measurements showed the llama.cpp MoE cache uploads ~350 MB per token with the default CPU share. Kiro proposed extending that cache instead of writing a pager from scratch, and the spec changed accordingly.
  - The Windows VRAM tests showed that running out of VRAM gives a silent 2.8–5x slowdown, not an error. That became a demo beat and a requirement for the controller.
  - The builder approved nine spec changes from the Task 1 checkpoint in one decision ("yes to all"). They were applied to requirements, design and tasks together, so the three files stay consistent.
- **Spec vs vibe coding:** *(builder to write: how this compared for you)*.

## Steering

`.kiro/steering/`:

| File | Inclusion | Purpose |
| --- | --- | --- |
| `product.md` | always | what MaltOE is, the narrowed novelty claim, vocabulary |
| `tech.md` | always | stack, pinned llama.cpp commit, toolchain (CUDA 12.9 with MSVC v14.39) |
| `structure.md` | always | repo layout |
| `research-integrity.md` | always | no invented numbers; cite WiSP and CRISP as prior art; claims only from measurements or cited sources |
| `workflow.md` | always | push at every checkpoint, commit as the builder, merge only on request. Added mid-build when the builder asked for it |
| `handoff.md` | manual | the original project handoff, pulled in only when needed |

The strategy that mattered most: `research-integrity.md`. *(builder to confirm or change.)* It is why the report says "unexplained" or "unsettled" where results were unclear, instead of rounding them into a story.

## Agent hooks

- `.kiro/hooks/checks-after-task.json`: on `PostTaskExec`, runs `ruff check`, `ruff format --check` and `pytest`. It finds the repo root itself, so it still works after the repo was renamed from Molt to MaltOE.
- `.kiro/hooks/checks-after-task-windows.json` + `scripts/checks.ps1`: the same checks in PowerShell for the Windows laptop.
- **check:** how often the hooks fired during task execution, and whether one caught a failure.

## Sub-agents and custom agents

- Built-in sub-agents used: **introspect** (Kiro questions, such as Web vs Crew) and **context-gatherer** (codebase reads). **check** which tasks used context-gatherer.
- Custom agents: none so far.

## Kiro powers

- **check:** the `github-cli` power was available in the session for git and GitHub REST work. Confirm whether it was loaded before claiming it.

## MCP

- No MCP servers added by the builder so far. *(Don't claim MCP unless one is added.)*

## Property-based testing

- **planned:** `hypothesis` is a dependency, and `design.md` lists correctness properties 1–11 (budget, safe boundaries, `k` changing only at boundary-apply, and others) for property tests on `SimEngineAdapter`. No property tests exist yet. Record here what they catch once they run.

## Bugs Kiro caught during the build

Each item has the evidence in the report or the commit history.

- A use-after-free in the chunked-expert spike, found by building it with AddressSanitizer in the sandbox.
- A shape bug in the width-emulation script, found before the builder's overnight run.
- The HumanEval scorer counted a program that called `exit(0)` as a pass. Fixed with a random sentinel.
- Windows PowerShell 5.1 lost child exit codes, so failed runs showed as `FAIL(exit )`. Fixed in `spikes/common.ps1`.
- Early CUDA timings were unreliable. Kiro changed the timing method and the rerun gave stable numbers.

## Where Kiro got it wrong

- The G3 width-quality script (`run_g3.ps1`) estimated "an hour to several hours". On the laptop it took well over 10 hours, because the model is larger than free RAM and every expert ran on the CPU. Kiro had already measured that RAM limit in Step B but didn't use it to size this run. The partial results were still recorded as they came in (report section 7).

## Research with Kiro

- Web search and fetch for prior art (WiSP, CRISP, MoSE, Edge0 and others), a survey of similar GitHub projects (feasibility report 3.2b), toolchain compatibility (CUDA 12.9 vs 13, MSVC versions), and the name check that led to the rename from Molt to MaltOE. NVIDIA already has a "Molt" project.
- After the builder pointed out ghost-infer, Kiro surveyed about 30 local MoE engines and papers into a matrix (`docs/landscape-matrix.md`). It found new prior art (MatMoE, FreeToken, Krasis) that narrowed the novelty claim again, two weeks before the deadline.
- Devpost blocks automated reads, so the builder pasted the rules and Kiro recorded them (`docs/kiroween-rules.md`).
