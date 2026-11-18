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
| J. W. S. Liu, K.-J. Lin, W.-K. Shih, A. C. Yu, J.-Y. Chung, W. Zhao, "Algorithms for Scheduling Imprecise Computations", *IEEE Computer* 24(5), May 1991 | paywalled; ResearchGate and SIAM return 403 to me | **abstract only** |
| K.-J. Lin, S. Natarajan, J. W. S. Liu, "Imprecise Results: Utilizing Partial Computations in Real-Time Systems", report dated April 15, 1987 (NASA CR-180561) | [NASA NTRS](https://ntrs.nasa.gov/api/citations/19870017086/downloads/19870017086.pdf) | full text (23 pages), sections 1–2 closely |
| J.-Y. Chung, J. W. S. Liu, K.-J. Lin, "Scheduling Periodic Jobs Using Imprecise Results", UIUCDCS-R-87-1307, November 1987 | [NASA NTRS](https://ntrs.nasa.gov/api/citations/19890017396/downloads/19890017396.pdf) | sections 1–3 |
| W. Feng, J. W. S. Liu, "Algorithms for Scheduling Real-Time Tasks with Input Error and End-to-End Deadlines", *IEEE Trans. Software Engineering* 23(2):93–106, 1997 | [copy hosted by Virginia Tech](https://synergy.cs.vt.edu/pubs/papers/feng-ieeetose1997-schedulingrts.pdf) | section 2 (the standard model, as Liu's group states it) |

**The 1991 paper itself is not read.** The formal model below comes from the 1987 reports and from Feng and Liu (1997), which restate the model used in the group's earlier studies. Until someone reads the 1991 text, the README should cite 1991 for the idea and its abstract only, and cite the others for definitions. If you have university library access, the 1991 paper is IEEE Computer vol. 24 no. 5.

## Imprecise computation (Liu and colleagues)

### What the sources say

- **Goal (1991 abstract):** avoid timing faults and degrade gracefully, by giving the user an approximate result of acceptable quality when the exact result can't be produced in time.
- **Monotone process (Chung, Liu, Lin 1987, section 1):** a process whose intermediate result's accuracy is non-decreasing as it runs longer. Its result on normal termination is the precise one. A result from premature termination may still be usable and is called imprecise.
- **Task model (Feng and Liu 1997, section 2):**
  - Each task is split into a mandatory part followed by an optional part.
  - The mandatory part must complete for an acceptable, usable result. A task meets its deadline if its mandatory part completes by then.
  - The optional part runs only after the mandatory part, and may be cut off so that this task and others meet their deadlines.
  - If a task gets less than its full processing time, the last portion is discarded. The fraction of discarded work is (o − σ)/o, where o is the optional processing time and σ the optional time actually given.
  - Usual objectives: weighted average, sum or maximum of discarded fractions, or the number of discarded optional tasks.
- **Error (Chung, Liu, Lin 1987, section 2):** error is 1 if the result is unacceptable, 0 if precise, and a non-increasing function of the time given in between.
- **Two ways to build monotone tasks (Lin, Natarajan, Liu 1987, section 2):**
  - **Milestone:** record partial results along the way, and return the last one when the deadline hits.
  - **Sieve:** a section of code whose only job is to make an existing result more precise. It returns results of the same number and type as its inputs, and skipping it is not fatal.

### Mapping to MaltOE

| Paper | MaltOE | Label |
| --- | --- | --- |
| Mandatory part | Chunk 1 of each routed expert, plus attention and routing | match |
| Optional part, run only after the mandatory part, can be cut off | Chunks 2 and 3 | match |
| **Sieve** | Each optional chunk adds its neurons' contribution to the expert's partial output. Same output shape, more precise, and skipping it leaves a usable result. | **match: chunks are sieves in Lin, Natarajan and Liu's sense** |
| Monotone task | Assumes quality is non-decreasing in k | match, **only if measured.** G3 tests it on average (perplexity at k = 1, 2, 3). It isn't guaranteed per token. |
| Fraction of discarded work (o − σ)/o | (3 − k)/2 of a routed expert's optional chunks | match |
| Error as a non-increasing function of work given | Importance mass dropped, used as our *estimate* of the error | **ours.** The papers define error abstractly. Choosing importance mass, and checking that it predicts quality, is our step. |
| Deadline (time) | VRAM budget (memory), plus the Objective | **adaptation.** The papers schedule processor time against deadlines. Applying the model to a memory limit is ours, and the README must say so. |
| Minimize total or weighted error under the constraint | Controller picks k to minimize weighted discarded work within the budget | adaptation of the objective |

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
7. **"Multiple versions" as a way to build imprecise tasks** (from memory, in an earlier draft). Not found in what was read. Only milestone and sieve are verified.

## What this changes in the plan

- **Theory checks to run, all from one routing trace and our existing measurements:**
  - P1 curve: ω(τ) in pages and bytes, at k = 3, 2, 1.
  - P3: the reentry rate against the measured upload bytes per token.
  - e(m) = 1/(1 + mT) against the measured slowdown.
  - Working-set eviction against llama.cpp's global policy, on the same trace.
  - Monotone and sieve: perplexity and importance mass at k = 1, 2, 3, plus the random-neuron control.
- **README wording:** describe chunks as *sieves*, and quote Denning on memory and scheduling being inseparable as the reason for one joint controller. Label the two adaptations: a memory budget in place of a deadline, and tokens as process time.
