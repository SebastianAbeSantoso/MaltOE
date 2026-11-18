"""Kill test: can DeepSeek-V4.1-Flash's Engram rows, read alone, predict the next token?

Hashing follows inference/engram.py from the model repo. Rows are fetched with HTTP range
requests. Readout: value = wkv[value part] @ concat(rows); logits = head @ rmsnorm(value) * norm.
"""
import io, os, json, struct, sys, time, zipfile, collections, concurrent.futures as cf
import numpy as np
import requests
from sympy import isprime
from tokenizers import Tokenizer, Regex, normalizers

REPO = "https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/resolve/main/"
CFG = json.load(open("config.json"))["text_config"]
LAYERS = CFG["engram_layer_ids"]
NUM_EMB = CFG["engram_num_embeddings"]
MAXN, NH, HD = CFG["engram_max_ngram_size"], CFG["engram_n_heads"], CFG["engram_head_dim"]
DIM, HC = CFG["hidden_size"], CFG["hc_mult"]
EPS = CFG["rms_norm_eps"]
NCOLS = (MAXN - 1) * NH
SHARD = {1: "model-00047-of-00048.safetensors", 14: "model-00048-of-00048.safetensors"}
HEAD_SHARD = "model-00043-of-00048.safetensors"

S = requests.Session()
S.mount("https://", requests.adapters.HTTPAdapter(pool_connections=64, pool_maxsize=64))


_CDN = {}


def cdn(shard):
    # resolve once; the signed CDN URL avoids per-request hub redirects and their rate limit
    if shard not in _CDN:
        r = S.get(REPO + shard, headers={"Range": "bytes=0-0"}, allow_redirects=False, timeout=60)
        _CDN[shard] = r.headers.get("Location", REPO + shard)
    return _CDN[shard]


def rng(shard, a, n):
    for attempt in range(8):
        try:
            r = S.get(cdn(shard), headers={"Range": f"bytes={a}-{a + n - 1}"}, timeout=60)
            if r.status_code == 429:
                time.sleep(float(r.headers.get("Retry-After", 2 ** attempt)))
                continue
            r.raise_for_status()
            assert len(r.content) == n, (len(r.content), n)
            return r.content
        except Exception:
            if attempt == 7:
                raise
            time.sleep(1 + attempt)
    raise RuntimeError("rate limited")


def header(shard):
    n = struct.unpack("<Q", rng(shard, 0, 8))[0]
    return 8 + n, json.loads(rng(shard, 8, n))


# fp8 e4m3fn and e8m0 decode tables
def _e4m3(b):
    s = -1.0 if b & 0x80 else 1.0
    e, m = (b >> 3) & 0xF, b & 7
    if e == 15 and m == 7:
        return np.nan
    return s * (m / 8 * 2.0**-6 if e == 0 else (1 + m / 8) * 2.0 ** (e - 7))


E4M3 = np.array([_e4m3(b) for b in range(256)], dtype=np.float32)
E8M0 = np.array([2.0 ** (b - 127) if b < 255 else np.nan for b in range(256)], dtype=np.float32)


def bf16(raw, shape):
    u = np.frombuffer(raw, dtype=np.uint16).astype(np.uint32) << 16
    return u.view(np.float32).reshape(shape)


# ---- hashing (inference/engram.py) ----
def token_map(tok):
    sentinel = "\ue000"
    norm = normalizers.Sequence([
        normalizers.NFKC(), normalizers.NFD(), normalizers.StripAccents(), normalizers.Lowercase(),
        normalizers.Replace(Regex(r"[ \t\r\n]+"), " "), normalizers.Replace(Regex(r"^ $"), sentinel),
        normalizers.Strip(), normalizers.Replace(sentinel, " "),
    ])
    key_to_new, lookup = {}, []
    for tid in range(tok.get_vocab_size(with_added_tokens=True)):
        text = tok.decode([tid], skip_special_tokens=False)
        if "\ufffd" in text:
            key = tok.id_to_token(tid)
        else:
            nt = norm.normalize_str(text)
            key = nt if nt else text
        if key not in key_to_new:
            key_to_new[key] = len(key_to_new)
        lookup.append(key_to_new[key])
    return np.array(lookup, dtype=np.int64), len(key_to_new)


def layout():
    primes, seen = [], set()
    for _ in LAYERS:
        flat = []
        for _ in range(MAXN - 1):
            cur = CFG["engram_vocab_size"] - 1
            for _ in range(NH):
                cur += 1
                while not isprime(cur) or cur in seen:
                    cur += 1
                seen.add(cur)
                flat.append(cur)
        primes.append(np.array(flat, dtype=np.int64))
    offsets = [np.concatenate([[0], np.cumsum(p[:-1])]).astype(np.int64) for p in primes]
    return primes, offsets


def multipliers(vocab):
    bound = max(1, (np.iinfo(np.int64).max // vocab) // 2)
    return [np.random.default_rng(10007 * lid).integers(0, bound, size=(MAXN,), dtype=np.int64) * 2 + 1 for lid in LAYERS]


def hash_ids(ctoks, pad, mult, primes, offs):
    """ctoks: compressed ids of one sequence. Returns [L, NCOLS] row ids for one layer."""
    L = len(ctoks)
    out = np.empty((L, NCOLS), dtype=np.int64)
    for t in range(L):
        toks = [ctoks[t - s] if t - s >= 0 else pad for s in range(MAXN)]
        rolling = int(toks[0]) * int(mult[0])
        col = 0
        for i in range(1, MAXN):
            rolling ^= int(toks[i]) * int(mult[i])
            for h in range(NH):
                out[t, col] = rolling % int(primes[col]) + int(offs[col])
                col += 1
    return out


def main(n_positions=400, seed=0):
    t0 = time.time()
    tok = Tokenizer.from_file("tokenizer.json")
    tmap, vocab = token_map(tok)
    print(f"compressed vocab {vocab} (config says {CFG['engram_compressed_vocab_size']})")
    assert vocab == CFG["engram_compressed_vocab_size"]
    pad = int(tmap[CFG["engram_pad_token_id"]])
    primes, offs = layout()
    for i, lid in enumerate(LAYERS):
        tot = int(primes[i].sum())
        print(f"layer {lid}: sum of bucket primes {tot}, table rows {NUM_EMB[i]}")
        assert tot == NUM_EMB[i]
    mult = multipliers(vocab)

    # ---- weights for the readout ----
    hbase, hh = header(HEAD_SHARD)
    hw = hh["head.weight"]; nw = hh["norm.weight"]
    print("downloading head.weight (%.2f GB) ..." % ((hw["data_offsets"][1] - hw["data_offsets"][0]) / 1e9))
    import os
    if os.path.exists("head.npy"):
        head = np.load("head.npy", mmap_mode="r")
    else:
        head = bf16(rng(HEAD_SHARD, hbase + hw["data_offsets"][0], hw["data_offsets"][1] - hw["data_offsets"][0]), hw["shape"])
        np.save("head.npy", head)
    norm_w = bf16(rng(HEAD_SHARD, hbase + nw["data_offsets"][0], nw["data_offsets"][1] - nw["data_offsets"][0]), nw["shape"])
    layer_meta = {}
    for lid in LAYERS:
        base, h = header(SHARD[lid])
        w, s = h[f"layers.{lid}.engram.wkv.weight"], h[f"layers.{lid}.engram.wkv.scale"]
        n_out, n_in = w["shape"]
        r0 = HC * DIM  # value rows start after the hc keys
        raw = rng(SHARD[lid], base + w["data_offsets"][0] + r0 * n_in, DIM * n_in)
        sraw = rng(SHARD[lid], base + s["data_offsets"][0] + (r0 // 32) * (n_in // 32), (DIM // 32) * (n_in // 32))
        W = E4M3[np.frombuffer(raw, dtype=np.uint8)].reshape(DIM, n_in)
        sc = E8M0[np.frombuffer(sraw, dtype=np.uint8)].reshape(DIM // 32, n_in // 32)
        W *= np.repeat(np.repeat(sc, 32, 0), 32, 1)
        e, es = h[f"layers.{lid}.engram.embed.weight"], h[f"layers.{lid}.engram.embed.scale"]
        layer_meta[lid] = dict(Wv=W, emb=base + e["data_offsets"][0], embs=base + es["data_offsets"][0])
        print(f"layer {lid}: value projection {W.shape}, finite={np.isfinite(W).all()}")

    # ---- eval and baseline text: wikitext-2 ----
    z = zipfile.ZipFile(io.BytesIO(requests.get("https://huggingface.co/datasets/ggml-org/ci/resolve/main/wikitext-2-raw-v1.zip", timeout=120).content))
    names = {n.split("/")[-1]: n for n in z.namelist()}
    train = z.read(names["wiki.train.raw"]).decode()
    test = z.read(names["wiki.test.raw"]).decode()
    tr = np.array(tok.encode(train[:3_000_000]).ids, dtype=np.int64)
    uni = collections.Counter(tr.tolist())
    big = collections.defaultdict(collections.Counter)
    tri = collections.defaultdict(collections.Counter)
    for a, b in zip(tr[:-1], tr[1:]):
        big[a][b] += 1
    for a, b, c in zip(tr[:-2], tr[1:-1], tr[2:]):
        tri[(a, b)][c] += 1
    uni_top = [t for t, _ in uni.most_common(10)]

    # eval positions: contiguous text from the test split, skipping headings
    paras = [p.strip() for p in test.split("\n") if len(p.strip()) > 400 and not p.strip().startswith("=")]
    ids = []
    for p in paras:
        ids.extend(tok.encode(" " + p).ids)
        if len(ids) > n_positions + 1:
            break
    ids = np.array(ids[: n_positions + 1], dtype=np.int64)
    ctoks = tmap[ids[:-1]]
    targets = ids[1:]

    results = {}
    emb_samples = {}
    rs = np.random.default_rng(seed)
    if os.path.exists("embed.npy"):
        EMB = np.load("embed.npy", mmap_mode="r")
    else:
        eb, eh = header("model-00002-of-00048.safetensors")
        ew = eh["embed.weight"]
        EMB = bf16(rng("model-00002-of-00048.safetensors", eb + ew["data_offsets"][0], ew["data_offsets"][1] - ew["data_offsets"][0]), ew["shape"])
        np.save("embed.npy", EMB)
    for lid_i, lid in enumerate(LAYERS):
        m = layer_meta[lid]
        hid = hash_ids(ctoks, pad, mult[lid_i], primes[lid_i], offs[lid_i])
        rnd = rs.integers(0, NUM_EMB[lid_i], size=hid.shape)
        for label, rows_ids in (("engram", hid), ("random rows", rnd)):
            flat = rows_ids.reshape(-1)
            uniq = np.unique(flat)

            def fetch(r):
                w = rng(SHARD[lid], m["emb"] + int(r) * HD, HD)
                s = rng(SHARD[lid], m["embs"] + int(r) * (HD // 32), HD // 32)
                return r, w, s

            import os
            cache = f"rows-L{lid}-{label.replace(' ', '_')}-{len(targets)}.npy"
            got = {}
            if os.path.exists(cache):
                X = np.load(cache)
            else:
              with cf.ThreadPoolExecutor(16) as ex:
                for r, w, s in ex.map(fetch, uniq):
                    v = E4M3[np.frombuffer(w, dtype=np.uint8)].reshape(HD // 32, 32)
                    v = v * E8M0[np.frombuffer(s, dtype=np.uint8)][:, None]
                    got[int(r)] = v.reshape(HD)
              X = np.stack([np.concatenate([got[int(r)] for r in row]) for row in rows_ids])  # [N, 6144]
              np.save(cache, X)
            V = X @ m["Wv"].T  # [N, 5120]
            Vn = V / np.sqrt((V * V).mean(-1, keepdims=True) + EPS) * norm_w
            logits = Vn @ head.T
            order = np.argsort(-logits, axis=1)[:, :10]
            ranks = [(order[i] == targets[i]).nonzero()[0] for i in range(len(targets))]
            top1 = np.mean([len(r) and r[0] == 0 for r in ranks])
            top5 = np.mean([len(r) and r[0] < 5 for r in ranks])
            top10 = np.mean([len(r) > 0 for r in ranks])
            results[f"layer {lid} {label}"] = (top1, top5, top10, len(uniq))
            En = EMB / np.linalg.norm(EMB, axis=1, keepdims=True)
            cos = (V / np.linalg.norm(V, axis=1, keepdims=True)) @ En.T
            o2 = np.argsort(-cos, axis=1)[:, :10]
            cur = ids[:-1]
            def acc(o, tg):
                return tuple(float(np.mean([tg[i] in o[i, :k] for i in range(len(tg))])) for k in (1, 5, 10))
            results[f"layer {lid} {label} | embed-cos -> next"] = acc(o2, targets)
            results[f"layer {lid} {label} | embed-cos -> current"] = acc(o2, cur)
            if label == "engram":
                emb_samples[lid] = [(tok.decode(ids[max(0, i - 3): i + 1].tolist()), [tok.decode([int(t)]) for t in o2[i, :5]], tok.decode([int(targets[i])])) for i in range(20, 32)]
            if label == "engram" and lid == LAYERS[0]:
                sample = [(tok.decode(ids[max(0, i - 3): i + 1].tolist()), tok.decode([int(order[i, 0])]), tok.decode([int(targets[i])])) for i in range(20, 32)]

    def topk_acc(pred_fn):
        a1 = a5 = a10 = 0
        for i in range(len(targets)):
            p = pred_fn(i)
            t = targets[i]
            a1 += t in p[:1]; a5 += t in p[:5]; a10 += t in p[:10]
        n = len(targets)
        return a1 / n, a5 / n, a10 / n

    def bigram(i):
        c = big.get(ids[i])
        return [t for t, _ in c.most_common(10)] if c else uni_top

    def trigram(i):
        c = tri.get((ids[i - 1], ids[i])) if i >= 1 else None
        return [t for t, _ in c.most_common(10)] if c else bigram(i)

    results["unigram (wikitext-2 train)"] = topk_acc(lambda i: uni_top)
    results["bigram (wikitext-2 train)"] = topk_acc(bigram)
    results["trigram (wikitext-2 train)"] = topk_acc(trigram)

    print(f"\n{len(targets)} positions of wikitext-2 test, {time.time() - t0:.0f} s")
    print(f"{'readout':34s} top1   top5   top10")
    for k, v in results.items():
        print(f"{k:34s} {v[0]:.3f}  {v[1]:.3f}  {v[2]:.3f}")
    print("\nlayer 1 engram, sample (context -> top-1 guess | actual next):")
    for ctx, guess, actual in sample:
        print(f"  ...{ctx!r:40s} -> {guess!r:14s} | {actual!r}")
    for lid, smp in emb_samples.items():
        print(f"\nlayer {lid} engram, nearest input embeddings (context -> top-5 | actual next):")
        for ctx, near, actual in smp:
            print(f"  ...{ctx!r:36s} -> {near!r} | {actual!r}")
    json.dump({k: list(map(float, v[:3])) for k, v in results.items()}, open("killtest-results.json", "w"), indent=1)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 400)
