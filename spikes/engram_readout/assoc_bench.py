"""Association benchmark for Engram rows: does the decoded row point at a related concept that is NOT in the phrase?

hit@10: any of the 10 nearest input embeddings (stripped, lowercased) is in the accepted set.
Compared against random rows and against the plain input embedding of the phrase's last token.
"""
import numpy as np
from tokenizers import Tokenizer
import killtest as K

PAIRS = [
    (" the Eiffel Tower", {"paris", "巴黎", "france", "法国"}),
    (" Mount Everest", {"nepal", "himalayas", "himalaya", "himalayan", "tibet", "尼泊尔", "喜马拉雅"}),
    (" the Great Wall", {"china", "chinese", "中国", "beijing", "北京"}),
    (" the Taj Mahal", {"india", "indian", "印度", "agra", "mughal"}),
    (" Big Ben", {"london", "伦敦", "england", "britain", "british", "英国"}),
    (" the Statue of Liberty", {"america", "american", "york", "美国", "纽约", "usa"}),
    (" the Colosseum", {"rome", "roman", "罗马", "italy", "意大利"}),
    (" the Sydney Opera House", {"australia", "australian", "澳大利亚", "悉尼"}),
    (" Mount Fuji", {"japan", "japanese", "日本", "tokyo", "东京"}),
    (" the Amazon rainforest", {"brazil", "巴西", "rainforest", "jungle", "南美"}),
    (" Albert Einstein", {"relativity", "physics", "physicist", "相对论", "物理", "newton", "galileo"}),
    (" Isaac Newton", {"gravity", "physics", "newtonian", "引力", "物理", "einstein", "galileo"}),
    (" Charles Darwin", {"evolution", "进化论", "进化", "species", "natural"}),
    (" William Shakespeare", {"hamlet", "macbeth", "playwright", "莎士比亚", "戏剧", "poet"}),
    (" Barack Obama", {"president", "总统", "biden", "拜登", "trump", "特朗普", "roosevelt"}),
    (" Leonardo da Vinci", {"mona", "lisa", "painter", "renaissance", "达芬奇", "michelangelo", "蒙娜丽莎"}),
    (" Wolfgang Amadeus Mozart", {"beethoven", "bach", "composer", "symphony", "贝多芬", "作曲家", "音乐"}),
    (" Søren Kierkegaard", {"nietzsche", "hegel", "heidegger", "existential", "康德", "黑格尔", "尼采", "philosopher"}),
    (" World War II", {"hitler", "nazi", "holocaust", "wwii", "二战", "希特勒", "纳粹"}),
    (" the French Revolution", {"napoleon", "robespierre", "bastille", "拿破仑", "法国大革命", "revolution"}),
    (" machine learning", {"人工智能", "深度学习", "神经网络", "ai", "neural", "deep"}),
    (" quantum mechanics", {"physics", "量子", "物理", "schrödinger", "heisenberg", "relativity"}),
    (" the Pacific Ocean", {"atlantic", "hawaii", "太平洋", "ocean", "海洋"}),
    (" the Sahara Desert", {"africa", "african", "非洲", "egypt", "撒哈拉"}),
    (" Yogyakarta", {"jakarta", "indonesia", "indonesian", "印尼", "bali", "java"}),
    (" the Olympic Games", {"olympics", "奥运会", "奥运", "athletes", "medal"}),
    (" the Mona Lisa", {"leonardo", "vinci", "louvre", "达芬奇", "painting", "卢浮宫"}),
    (" Star Wars", {"jedi", "skywalker", "darth", "lucas", "星球大战", "star", "trek"}),
    (" the Roman Empire", {"rome", "roman", "罗马", "罗马帝国", "byzantine", "caesar"}),
    (" the Nile River", {"egypt", "egyptian", "埃及", "africa", "尼罗河"}),
]


def main():
    tok = Tokenizer.from_file("tokenizer.json")
    tmap, vocab = K.token_map(tok)
    pad = int(tmap[K.CFG["engram_pad_token_id"]])
    primes, offs = K.layout()
    mult = K.multipliers(vocab)
    EMB = np.load("embed.npy", mmap_mode="r")
    En = EMB / np.linalg.norm(EMB, axis=1, keepdims=True)
    lid, li = K.LAYERS[0], 0
    base, h = K.header(K.SHARD[lid])
    w, s = h[f"layers.{lid}.engram.wkv.weight"], h[f"layers.{lid}.engram.wkv.scale"]
    n_in = w["shape"][1]
    r0 = K.HC * K.DIM
    raw = K.rng(K.SHARD[lid], base + w["data_offsets"][0] + r0 * n_in, K.DIM * n_in)
    sraw = K.rng(K.SHARD[lid], base + s["data_offsets"][0] + (r0 // 32) * (n_in // 32), (K.DIM // 32) * (n_in // 32))
    W = K.E4M3[np.frombuffer(raw, dtype=np.uint8)].reshape(K.DIM, n_in)
    W *= np.repeat(np.repeat(K.E8M0[np.frombuffer(sraw, dtype=np.uint8)].reshape(K.DIM // 32, n_in // 32), 32, 0), 32, 1)
    e, es = h[f"layers.{lid}.engram.embed.weight"], h[f"layers.{lid}.engram.embed.scale"]
    eo, so = base + e["data_offsets"][0], base + es["data_offsets"][0]

    def row(r):
        wb = K.rng(K.SHARD[lid], eo + int(r) * K.HD, K.HD)
        sb = K.rng(K.SHARD[lid], so + int(r) * (K.HD // 32), K.HD // 32)
        return (K.E4M3[np.frombuffer(wb, dtype=np.uint8)].reshape(K.HD // 32, 32) * K.E8M0[np.frombuffer(sb, dtype=np.uint8)][:, None]).reshape(K.HD)

    def near_from_vec(v, k=10):
        cos = En @ (v / np.linalg.norm(v))
        return [tok.decode([int(t)]).strip().lower() for t in np.argsort(-cos)[:k]]

    rs = np.random.default_rng(3)
    hits = {"engram": 0, "random rows": 0, "last-token embedding": 0}
    for phrase, ok in PAIRS:
        ids = np.array(tok.encode(phrase).ids, dtype=np.int64)
        in_phrase = {tok.decode([int(t)]).strip().lower() for t in ids}
        accept = {a for a in ok if a not in in_phrase}
        hid = K.hash_ids(tmap[ids], pad, mult[li], primes[li], offs[li])[-1]
        res = {
            "engram": near_from_vec(W @ np.concatenate([row(r) for r in hid])),
            "random rows": near_from_vec(W @ np.concatenate([row(r) for r in rs.integers(0, K.NUM_EMB[li], size=K.NCOLS)])),
            "last-token embedding": near_from_vec(np.asarray(EMB[int(ids[-1])], dtype=np.float32)),
        }
        line = []
        for k, near in res.items():
            hit = any(n in accept for n in near)
            hits[k] += hit
            line.append(f"{k}={'Y' if hit else '.'}")
        print(f"{phrase!r:30s} {' '.join(line)}   engram top-10: {res['engram']}")
    n = len(PAIRS)
    print(f"\nhit@10 over {n} phrases (accepted words never in the phrase):")
    for k, v in hits.items():
        print(f"  {k:22s} {v}/{n}")


if __name__ == "__main__":
    main()
