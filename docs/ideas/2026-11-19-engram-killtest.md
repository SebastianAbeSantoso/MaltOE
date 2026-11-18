# Kill test: reading DeepSeek-V4.1-Flash's Engram tables directly (2026-11-19)

Setup: hashing reproduced from `inference/engram.py` in [deepseek-ai/DeepSeek-V4.1-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash) (commit 314113f). Two self-checks passed: the compressed vocab is 99,092 as in the config, and the bucket primes sum exactly to each table's row count (384,006,168 and 384,016,682). Rows fetched by HTTP range from the HF CDN: 24 rows of 256 FP8 values plus 8 scales per layer per position. The value vector is `wkv` (value part, 5120 × 6144) applied to the 24 concatenated rows, exactly what the model adds to its residual stream (times a gate).

Scripts: `spikes/engram_readout/`.

## Version A: does the memory alone predict the next token? **No.**

400 positions of wikitext-2 test.

| Readout | top-1 | top-5 | top-10 |
| --- | --- | --- | --- |
| layer 1, value → final norm → output head | 0.000 | 0.000 | 0.000 |
| layer 14, same | 0.010 | 0.013 | 0.018 |
| layer 1, nearest input embedding = next token | 0.000 | 0.000 | 0.000 |
| layer 1, nearest input embedding = current token | 0.020 | 0.037 | 0.045 |
| random rows (all readouts) | 0.000 | 0.000 | 0.000 |
| unigram baseline | 0.048 | 0.172 | 0.265 |
| bigram (wikitext-2 train) | 0.210 | 0.407 | 0.470 |
| trigram (wikitext-2 train) | 0.228 | 0.372 | 0.430 |

The "Shannon's Ghost" framing (Engram as a 1948-style next-token n-gram model) fails under direct readout. A trained readout (tuned lens) was not tried.

## Version B: does the memory store concepts about the phrase? **Yes, clearly.**

`assoc_bench.py`: 30 phrases (landmarks, people, events, fields). For each, the 10 nearest input embeddings to the layer-1 value vector of the phrase's last position; a hit means one of them is in a hand-written set of related words that excludes the phrase's own words (e.g. "the Eiffel Tower" → Paris/France).

| Readout | hit@10 |
| --- | --- |
| **Engram layer 1** | **22/30** |
| random rows | 0/30 |
| plain input embedding of the last token | 8/30 |

Examples (layer 1, top of the list):
- " the Eiffel Tower" → Paris, 巴黎, France, Louvre
- " Mount Everest" → Himalayas, Nepal, Bhutan, Tibet
- " Søren Kierkegaard" → Nietzsche, 黑格尔 (Hegel), Hegel, Heidegger, 康德 (Kant), Wittgenstein
- " World War II" → WWII, 二战, Holocaust, Hitler, Nazi
- " Yogyakarta" → Jakarta, Indonesia, Bali
- " Star Wars" → Jedi, Disney, Lego, Batman

Layer 14 is mostly noise under this readout, with exceptions (" World War II" → WWII, Axis; " Søren Kierkegaard" → existential, Denmark, Danish).

Odd regularity: "Titanic" appears in the top 10 for many unrelated famous things, a kind of default "famous object" direction. Not investigated.

**Caveats:**
- n = 30, and the related-word sets are hand-written.
- About 9 of the 30 phrases were seen in an earlier probe before the sets were written, so those could be biased. The other 21 were not.
- The decoding uses nearest input embeddings, which assumes the value vector lives in embedding space. That is plausible at layer 1 and not proven.
- "Engram" stores more than these words show. This readout only sees what is near a token embedding.

## What this means

The original claim is dead; a better one replaced it. DeepSeek-V4.1-Flash's 189 GiB Engram tables can be read like a dictionary of associations, by hash address, without running the model: about 12 KB fetched per phrase per layer, no GPU. Prior art found earlier ([EngramEdit](https://arxiv.org/abs/2610.10533), [GateRQ](https://arxiv.org/pdf/2610.35985.pdf)) edits or ablates the tables. Reading them out directly was not found.

Possible framing, **checked 2026-11-19:**
- Richard Semon introduced the term "engram" for a memory trace in *Die Mneme* (1904) ([Josselyn, Köhler & Frankland, J. Neurosci. 2017, "Heroes of the Engram"](https://pmc.ncbi.nlm.nih.gov/articles/PMC6596490/)).
- Karl Lashley's "In Search of the Engram" (1950) summarised about 30 years of cortical lesion experiments in rats. He concluded that memories are not localized but distributed within functional areas of the cortex ([Bruce 2001, "Fifty years since Lashley's In search of the Engram"](https://pubmed.ncbi.nlm.nih.gov/11770197)). His closing remark that learning "just is not possible" is quoted in later reviews ([Springer chapter](https://link.springer.com/chapter/10.1007/978-1-4615-9843-5_1)).
- So the hook holds: in DeepSeek-V4.1 an "engram" has an exact address, computable from the text, and can be read without running the model.

## Other checks (2026-11-19)

- **Browser access:** the Hugging Face CDN answers range requests with `access-control-allow-origin: *` and allows the `Range` header in preflight. A static web page can fetch rows directly, with no server.
- **Prior art for reading the rows out:** a further search found compression (CP, GateRQ), architecture variants (tokenizer-agnostic Engram), editing (EngramEdit) and ablation, but no direct readout.
- **Open:** the video rule on third-party trademarks applies to showing "DeepSeek" by name (see `docs/kiroween-rules.md`).
