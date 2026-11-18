# Model-internals idea search, 2026-11-19

Aimed at how LLMs work inside, excluding inference engines. 34 candidates screened by a research sub-agent with about 55 searches. "No close prior art found" means none in those searches.

| # | Idea | Claim to test | Novelty | Runs where |
| --- | --- | --- | --- | --- |
| 1 | **One Number:** click one glowing weight in a field of billions and the model turns to gibberish; census of which 2026 models still have such a "super weight" | zeroing one scalar raises perplexity >100× ([Super Weight paper](https://arxiv.org/abs/2411.07191)); a [2026 follow-up](https://arxiv.org/html/2607.08733) says not in all models | adjacent | ≤3B on laptop in bf16; census needs rental |
| 2 | **Expert Organ:** OLMoE's 64 experts as organ pipes; the router plays chords; hold a key to disable an expert live | listeners can tell code, prose and Indonesian apart from routing audio alone | no close prior art found | OLMoE-1B-7B on the builder's hardware |
| 3 | **Shannon's Ghost:** read DeepSeek-V4.1-Flash's Engram tables directly, fetched by HTTP range, as an n-gram model, side by side with a 1948 Shannon-style generator | Engram rows alone either predict next tokens like a trigram model, or store entity facts instead | no close prior art found | rows fetched over HTTP; direct readout on CPU; tuned lens may need rental |
| 4 | **LLM Homunculus:** Penfield's distorted body drawn from how many MoE experts each domain owns vs its share of training data | OLMoE's expert allocation is not proportional to its data mix | adjacent | laptop |
| 5 | **Forgotten Facts:** facts Pythia knew mid-training and lost by the end, with the batches that taught them | a measurable share of facts go from correct to wrong across checkpoints | adjacent | 410M feasible; 1.4B needs cloud downloads |

Dropped as taken (examples): steering time sense ([Time Travel Engine](https://aclanthology.org/2026.findings-emnlp.417.pdf)), world-map edits, weight-DNA lineage ([MoTHer](https://arxiv.org/html/2405.18432)), SteerMoE expert steering, glitch-token hunts, ROME/EngramEdit fact editing, Neuronpedia-style steering boards, bbycroft 3D walkthroughs, circuit-tracer graphs, LensLapse checkpoint lenses, GateRQ Engram ablation.
