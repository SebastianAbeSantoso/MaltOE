# The two old ideas: what the sources actually say

Checked 2026-11-18 by reading the papers' text, not from memory. This file is the reference for every claim the README, design, write-up or video makes about Liu et al. or Denning. If a claim isn't here, check the source and add it here first (rule in `.kiro/steering/research-integrity.md`).

Each mapping to MaltOE is labelled:
- **match:** the paper's concept, used as the paper defines it;
- **adaptation:** the paper's concept carried over to a different setting, which we must say;
- **ours:** not in the paper.

## Sources read

| Source | Access | How read |
| --- | --- | --- |
| P. J. Denning, "The Working Set Model for Program Behavior", *Communications of the ACM* 11(5):323–333, May 1968 | [scan hosted by UMass](https://people.cs.umass.edu/~emery/classes/cmpsci691st/readings/OS/WSModel_1968.pdf) | full text (11 pages), sections 1–7 |
| P. J. Denning, "Thrashing: Its Causes and Prevention", *AFIPS Fall Joint Computer Conference* 1968, pp. 915–922 | [copy hosted by U. Waterloo](https://cs.uwaterloo.ca/~brecht/courses/702/Possible-Readings/vm-and-gc/thrashing-denning-afips-1968.pdf) | full text (8 pages) |
| J. W. S. Liu, K.-J. Lin, W.-K. Shih, A. C. Yu, J.-Y. Chung, W. Zhao, "Algorithms for Scheduling Imprecise Computations", *IEEE Computer* 24(5):58–68, May 1991 | paywalled; the builder supplied the PDF on 2026-11-18 (not committed: copyright) | **full text (11 pages)** |
| K.-J. Lin, S. Natarajan, J. W. S. Liu, "Imprecise Results: Utilizing Partial Computations in Real-Time Systems", report dated April 15, 1987 (NASA CR-180561) | [NASA NTRS](https://ntrs.nasa.gov/api/citations/19870017086/downloads/19870017086.pdf) | full text (23 pages), sections 1–2 closely |
| J.-Y. Chung, J. W. S. Liu, K.-J. Lin, "Scheduling Periodic Jobs Using Imprecise Results", UIUCDCS-R-87-1307, November 1987 | [NASA NTRS](https://ntrs.nasa.gov/api/citations/19890017396/downloads/19890017396.pdf) | sections 1–3 |
| W. Feng, J. W. S. Liu, "Algorithms for Scheduling Real-Time Tasks with Input Error and End-to-End Deadlines", *IEEE Trans. Software Engineering* 23(2):93–106, 1997 | [copy hosted by Virginia Tech](https://synergy.cs.vt.edu/pubs/papers/feng-ieeetose1997-schedulingrts.pdf) | section 2 (the standard model, as Liu's group states it) |

The 1991 paper was read in full on 2026-11-18. It is the primary source below; the 1987 reports and Feng and Liu (1997) are kept as supporting sources.

## Imprecise computation (Liu et al. 1991)

### What the 1991 paper says

- **Technique (p. 58):** prevents timing faults and degrades gracefully by giving the user an approximate result of acceptable quality when the exact result can't be produced in time.
- **Decomposition (p. 58):**
  - Programmers structure every time-critical task as a mandatory subtask and an optional subtask.
  - The mandatory subtask is required for an acceptable result and must complete before the deadline. The optional subtask refines the result and may be left unfinished.
  - A completed task gives the precise result, with zero error. A task cut off after its mandatory subtask gives a usable, *imprecise* result.
- **Monotone task (p. 59):** a task whose intermediate result's quality does not decrease as it runs longer.
- **Three ways to build imprecise tasks (p. 59):**
  - **Milestone:** record intermediate results and error indicators; if the task is cut off, the latest recorded values are returned.
  - **Sieve:** computation steps that can be skipped to save time. The paper's example skips a radar step that re-estimates the noise level and reuses the old estimate.
  - **Multiple version:** a primary version (precise, slower) and an alternate version (imprecise, faster). During a transient overload, the system runs the alternate.
- **The 0/1 constraint (p. 59):** finishing part of a sieve gives no benefit, so a sieve must run to completion or not at all. The same holds for multiple versions. For scheduling, the primary version is the mandatory subtask plus the whole optional subtask, and the alternate is the mandatory subtask alone. Scheduling with 0/1 constraints is harder than scheduling monotone tasks.
- **Running it (p. 59):**
  - Mandatory subtasks need bounded resource and time requirements, and enough processor time to finish by their deadlines.
  - Leftover time completes as many optional subtasks as possible.
  - The mandatory subtasks can use a conservative scheduler (rate-monotone), and the optional subtasks a more dynamic one (earliest deadline first).
  - For monotone tasks, the system can decide online, or nearly online, how much of each optional subtask to run.
- **Basic workload model (pp. 59–60):**
  - Each task T_i has a ready time, a deadline, a processing time τ_i and a weight w_i, its relative importance. It splits into M_i and O_i with m_i + o_i = τ_i.
  - O_i becomes ready only when M_i completes, and can be terminated at any time.
  - A valid schedule gives each task between m_i and τ_i of processor time.
- **Error (p. 60):**
  - ε_i = E_i(o_i − σ_i) (eq. 1), where σ_i is the time given to O_i, and E_i is monotone non-increasing in σ_i.
  - The simplest choice is ε_i = o_i − σ_i (eq. 2a). Total error is ε = Σ w_i ε_i (eq. 2b).
  - Other objectives: maximum error; the number of discarded optional tasks (under the 0/1 constraint; NP-complete in general, with a shortest-processing-time-first heuristic); the number of tardy tasks; and mean flow time under a total-error limit.
- **Algorithms (pp. 61–63):**
  - The ED algorithm (earliest deadline first, terminating each task at its deadline) tests feasibility on the mandatory set alone.
  - Algorithm F is optimal for identical weights, in O(n log n).
  - The largest-weight-first (LWF) algorithm is optimal for different weights, in O(n²).
- **Periodic jobs (pp. 63–64):**
  - Error-noncumulative jobs, such as image enhancement, use average error. Error-cumulative jobs, such as tracking, can't skip optional work for several periods in a row.
  - The heuristics for noncumulative jobs give all mandatory jobs priority over all optional jobs.
  - The best order for optional work depends on the shape of the error function:
    - least-attained-time suits convex error functions (error falls fast early);
    - first-come-first-serve suits concave ones.
- **Queuing formulation (pp. 66–67):**
  - **Two-level discipline:** each task runs at full or reduced level. The reduced processing time is a fraction γ of the full time. A threshold H on the number of tasks in the system decides when the load counts as heavy. G, the fraction of tasks served at full level, measures quality of service.
  - **Monotone systems:** each task is a mandatory task followed by K optional tasks O_i,j. When overloaded, the system discards x ≤ K of them. The expected fraction of optional tasks discarded is a rough measure of average error. The *imprecision probability* is the chance that any task is imprecise.
  - **Responsive service scheme:** whenever the queue length is H or more, discard x optional tasks *in the tasks being served*. It clears overloads quickly but can't guarantee quality: a task that arrived under light load can still be degraded.
  - **Guaranteed-service scheme:** tag only *newly arriving* tasks for reduced service while H or more are queued. Tasks already in the system finish at full quality. It responds more slowly and guarantees quality for tasks that arrived before the overload.

### Mapping to MaltOE

| 1991 paper | MaltOE | Label |
| --- | --- | --- |
| Mandatory subtask M_i | Chunk 1 of each routed expert, plus attention and routing | match |
| A mandatory task followed by K optional tasks O_i,j (queuing section) | Chunks 2 and 3: K = 2 optional tasks per routed expert | **match** |
| **Sieve:** a step that can be skipped to save time | Each optional chunk adds its neurons' contribution to the expert's partial output. Skipping it leaves a usable result. | **match** |
| **0/1 constraint:** a sieve runs fully or not at all | A chunk is computed whole or skipped; there is no half chunk | **match.** So MaltOE schedules under the 0/1 constraint, not the continuous monotone case. |
| Multiple version: alternate = M_i, primary = M_i + O_i | Width k = 1 as the alternate, k = 3 as the primary | match (an equivalent view) |
| Monotone: quality doesn't decrease with more work | Quality non-decreasing in k | match, **only if measured.** G3 tests it on average; it isn't guaranteed per token. |
| Error ε_i = E_i(o_i − σ_i), with E_i unknown in general; simplest choice o_i − σ_i | Simplest: optional chunks skipped, (3 − k) per routed expert. Our estimate of E_i: importance mass dropped | the simple form is a **match**. Using importance mass as E_i is **ours**, and we must check that it predicts quality. |
| Total weighted error Σ w_i ε_i | Weights per task. Candidates: router weight of the expert, or layer importance | the form is a **match**; the choice of weights is **ours** |
| Fraction of optional tasks discarded as a rough average error; imprecision probability | (3 − k)/2 at uniform width; fraction of tokens served below full width | **match** |
| Two-level discipline, reduced time = γ × full time, threshold H | Width 2/3 or 1/3 of full; threshold on memory pressure instead of queue length | **adaptation:** the paper's load signal is queue length, ours is VRAM. |
| **Responsive vs guaranteed-service schemes** | Narrow the current conversation at the next Safe Boundary (responsive), or apply the narrower width only to the next request (guaranteed) | **match.** This is a design choice the paper names and analyses. MaltOE currently does the responsive scheme (Req 7). The paper's trade-off (fast relief vs quality guarantee) should be stated, and the guaranteed scheme is a candidate setting. |
| Deadline (time) | VRAM budget (memory), plus the Objective | **adaptation.** The paper schedules processor time against deadlines. Applying the model to a memory limit is ours. |
| Error-noncumulative jobs | Each token's width affects only that token's computation, but errors do carry through the KV cache and later tokens | **open question:** probably cumulative in part. Say so; don't claim noncumulative. |
| Convex vs concave error functions decide the order of optional work | The G3 importance-mass curve (k = 1 keeps 46.5%, k = 2 keeps 76.5%) gives the shape of our estimate | **prediction to check:** if quality also falls fastest for the first chunks dropped, the paper's guidance applies. |

### Supporting sources (still valid)

- Chung, Liu, Lin (1987): the monotone-process definition and an error defined as 1 for unacceptable, 0 for precise, non-increasing in between.
- Lin, Natarajan, Liu (1987): the original milestone and sieve primitives. A sieve refines its inputs into more precise outputs of the same number and type.
- Feng and Liu (1997): the same task model, with the fraction of discarded work (o − σ)/o.

## Working set and thrashing (Denning 1968)

### What the sources say

**CACM paper:**
- **Definition (section 3, p. 326):** W(t, τ) is the information a process referenced during the process-time interval (t − τ, t). τ is the working-set parameter, and ω(t, τ) is the number of pages in W(t, τ). The elements are pages, though Denning notes they could be any other named units of information.
- **Properties (section 3, pp. 326–327):**
  - **P1 size:** ω(t, 0) = 0; ω is monotonically increasing in τ and concave downward. On average ω(t, 2τ) ≤ 2ω(t, τ).
  - **P2 prediction:** for small α, W(t, τ) predicts W(t + α, τ) well; it doesn't for α much larger than τ.
  - **P3 reentry rate:** λ(τ) = (1 − F_x(τ)) / x̄, where x is the time between references to the same page. Every reentry is a page recalled from auxiliary memory, so λ estimates page traffic.
  - **P4 τ-sensitivity:** σ(τ) = −dλ/dτ ≥ 0. Shrinking τ never lowers the reentry rate.
- **Choosing τ (p. 327–328):** a compromise between too much page traffic (τ too small) and wasted memory (τ too large). Denning recommends τ comparable to the memory traverse time T, for example τ ≈ 2T to limit the residency drop to 50%.
- **Memory and scheduling are one problem (section 4, p. 329):** memory management and process scheduling must be closely related activities; one cannot take place independently of the other. Denning also requires that an implementation report current working-set sizes.
- **Demand and balance (section 6, pp. 330–332):** memory demand m_i = min(ω_i/M, 1). The system is balanced when total demand equals chosen fractions (α, β) of processor and memory. A balance policy picks the next job that comes closest to restoring balance, memory first.
- **Thrashing (p. 332):** named as the result of overcommitting memory, when too many working sets occupy memory and displace each other's pages.

**FJCC paper:**
- **Definition:** thrashing is excessive overhead and severe degradation or collapse caused by too much paging.
- **Missing-page probability m(n, s, M):** while Σ ω_i(t, τ) ≤ M (eq. 2), programs don't interact and m is small. Past a critical number n₀ of programs, the working sets exceed M and m rises sharply. Under a working-set algorithm, m depends only on τ.
- **Efficiency (eq. 7):** e(m) = 1 / (1 + mT). Thrashing comes from e's extreme sensitivity to m, because the traverse time T is large.
- **Cures:**
  - keep m from fluctuating by keeping n < n₀, so every active process's working set fits (eq. 19);
  - reduce T.
  - Good locality makes favored-page sets small, which raises n₀.
- **Global LRU and FIFO:** under saturation, they let one program's demand take another's pages, and so are highly susceptible to thrashing.

### Mapping to MaltOE

| Paper | MaltOE | Label |
| --- | --- | --- |
| Page; "any other named units of information" | One chunk of one expert in one layer | match (Denning allows any named unit) |
| Process time; window τ | Generated tokens; τ in tokens | adaptation: virtual time counted in tokens |
| W(t, τ), ω(t, τ) | Set and count (and bytes) of chunk pages referenced in the last τ tokens | match |
| P1: ω increasing, concave downward in τ | Testable on a routing trace | **prediction to test** |
| P3: reentry rate λ(τ) = page traffic | Bytes uploaded per token from RAM | **prediction to test** against measured uploads (B2: ~350 MB/token at CPU share 0) |
| Memory management and scheduling must be joint | One controller sets width (the work scheduled) and residency together | **match: Denning argues for exactly this coupling** |
| Σ ω ≤ M, else thrashing | Bytes of the working set ≤ VRAM budget, else the silent slowdown we measured (Step C, 2.8–5x) | adaptation: one process, memory in bytes |
| e(m) = 1/(1 + mT) | m = cache miss rate per expert reference, T = time to bring one page to the GPU | **prediction to test:** do measured m and T predict measured tok/s? |
| Cure: run fewer processes (n < n₀) | Not available: one user, one model | doesn't apply |
| Cure: smaller favored sets raise n₀ | At width k, chunks above k are never referenced, so they leave W(t, τ) by definition and ω in bytes shrinks | **ours.** Consistent with Denning's inequality, but shrinking the set by dropping information is our step, not his. |
| Global LRU/FIFO thrash under saturation | llama.cpp's cache uses a global frequency-and-recency policy | **prediction to test:** working-set eviction vs llama.cpp's policy on the same trace |
| Report working-set sizes | Dashboard shows ω, the budget and the decision | match |

## Corrections to what Kiro said earlier (2026-11-18 chat)

Before reading these sources, Kiro said several things that don't hold:

1. **"Error = importance mass dropped" as Liu's definition.** Wrong. The papers define error as a non-increasing function of the work given. Importance mass is our estimate of it.
2. **"The deadline is the VRAM budget."** This is an adaptation, not the papers' model. Their deadlines are time.
3. **"Denning's remedy is load control."** The term "load control" appears in neither paper. The substance, keeping n < n₀ so the working sets fit, is in the FJCC paper.
4. **"Thrashing and load control are in his separate paper."** Partly wrong. The CACM paper names thrashing (p. 332); the FJCC paper analyses it.
5. **"Miss-rate knee" as Denning's prediction.** It's in neither paper. Drop it, or cite later work only after reading it.
6. **"The working-set curve rises quickly, then flattens."** The paper's wording is "monotonically increasing and concave downward" (P1). Use that.
7. ~~"Multiple versions" is not verified.~~ **Retracted after reading the 1991 paper:** the multiple-version method is on p. 59, next to milestone and sieve. The earlier draft was right and this correction was wrong.

## What this changes in the plan

- **Theory checks to run, all from one routing trace and our existing measurements:**
  - P1 curve: ω(τ) in pages and bytes, at k = 3, 2, 1.
  - P3: the reentry rate against the measured upload bytes per token.
  - e(m) = 1/(1 + mT) against the measured slowdown.
  - Working-set eviction against llama.cpp's global policy, on the same trace.
  - Monotone and sieve: perplexity and importance mass at k = 1, 2, 3, plus the random-neuron control.
- **README wording:** describe chunks as *sieves* run under the paper's *0/1 constraint*, say whether MaltOE uses the *responsive* or *guaranteed-service* scheme, and quote Denning on memory and scheduling being inseparable as the reason for one joint controller. Label the two adaptations: a memory budget in place of a deadline, and tokens as process time.
