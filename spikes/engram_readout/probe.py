"""Probe: what do DeepSeek-V4.1-Flash's Engram rows associate with a phrase?

For the last token of each phrase, hash the n-grams ending there, fetch the 24 rows per Engram layer
over HTTP range requests, project them with wkv's value part, and list the nearest input embeddings.
A random-row control is printed for comparison.
"""
import sys
import numpy as np
from tokenizers import Tokenizer
import killtest as K

PHRASES = sys.argv[1:] or [
    " Barack Obama", " the Eiffel Tower", " Mount Everest", " the capital of Indonesia", " Toraja coffee",
    " World War II", " Albert Einstein", " the Great Wall", " Harry Potter", " machine learning",
    " Søren Kierkegaard", " Yogyakarta", " photosynthesis", " the Titanic", " Jakarta",
]


def main():
    tok = Tokenizer.from_file("tokenizer.json")
    tmap, vocab = K.token_map(tok)
    pad = int(tmap[K.CFG["engram_pad_token_id"]])
    primes, offs = K.layout()
    mult = K.multipliers(vocab)
    EMB = np.load("embed.npy", mmap_mode="r")
    En = EMB / np.linalg.norm(EMB, axis=1, keepdims=True)
    meta = {}
    for lid in K.LAYERS:
        base, h = K.header(K.SHARD[lid])
        w, s = h[f"layers.{lid}.engram.wkv.weight"], h[f"layers.{lid}.engram.wkv.scale"]
        n_in = w["shape"][1]
        r0 = K.HC * K.DIM
        raw = K.rng(K.SHARD[lid], base + w["data_offsets"][0] + r0 * n_in, K.DIM * n_in)
        sraw = K.rng(K.SHARD[lid], base + s["data_offsets"][0] + (r0 // 32) * (n_in // 32), (K.DIM // 32) * (n_in // 32))
        W = K.E4M3[np.frombuffer(raw, dtype=np.uint8)].reshape(K.DIM, n_in)
        W *= np.repeat(np.repeat(K.E8M0[np.frombuffer(sraw, dtype=np.uint8)].reshape(K.DIM // 32, n_in // 32), 32, 0), 32, 1)
        e, es = h[f"layers.{lid}.engram.embed.weight"], h[f"layers.{lid}.engram.embed.scale"]
        meta[lid] = (W, base + e["data_offsets"][0], base + es["data_offsets"][0])

    def row(lid, r):
        _, eo, so = meta[lid]
        w = K.rng(K.SHARD[lid], eo + int(r) * K.HD, K.HD)
        s = K.rng(K.SHARD[lid], so + int(r) * (K.HD // 32), K.HD // 32)
        v = K.E4M3[np.frombuffer(w, dtype=np.uint8)].reshape(K.HD // 32, 32) * K.E8M0[np.frombuffer(s, dtype=np.uint8)][:, None]
        return v.reshape(K.HD)

    def nearest(lid, rids, k=10):
        X = np.concatenate([row(lid, r) for r in rids])
        V = meta[lid][0] @ X
        cos = En @ (V / np.linalg.norm(V))
        return [tok.decode([int(t)]) for t in np.argsort(-cos)[:k]]

    rs = np.random.default_rng(1)
    for p in PHRASES:
        ids = np.array(tok.encode(p).ids, dtype=np.int64)
        print(f"\n{p!r}  (tokens: {[tok.decode([int(t)]) for t in ids]})")
        for li, lid in enumerate(K.LAYERS):
            hid = K.hash_ids(tmap[ids], pad, mult[li], primes[li], offs[li])[-1]
            print(f"  layer {lid:2d}: {nearest(lid, hid)}")
        print(f"  random  : {nearest(K.LAYERS[0], rs.integers(0, K.NUM_EMB[0], size=K.NCOLS))}")


if __name__ == "__main__":
    main()
