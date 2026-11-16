#!/usr/bin/env python3
"""Make a width-emulation variant of a Qwen3-MoE GGUF (Task 1.5, gate G3).

Idea
----
A width-k expert runs only the k*chunk most important neurons of the expert.
For SwiGLU, neuron i contributes  down[:, i] * silu(gate_i . x) * (up_i . x).
If the gate row of neuron i is all zeros, silu(0) = 0 and the neuron contributes nothing.
So zeroing the gate (and up) rows of every neuron outside the kept set gives exactly the
output of the narrower expert. No reordering of the stored tensors is needed to measure
quality, and the file layout stays valid for stock llama.cpp.

Zero bytes dequantize to 0.0 for every ggml block type (the block scale is 0), and a row of
a gate/up tensor is a whole number of blocks, so rows can be zeroed in place.

Importance of neuron i of expert e (design.md, "Importance and reordering"):
    I[e, i] = sqrt(mean_t(h_i^2)) * ||W_down[e][:, i]||_2
where mean(h^2) comes from llama-imatrix (the input of ffn_down_exps is the activation h).

Usage
-----
    python make_variant.py --base Qwen3-30B-A3B-Q4_K_M.gguf --imatrix imatrix.gguf \\
        --out work/k1.gguf --k 1
    python make_variant.py ... --k 1 --select random --seed 7     # control

Needs gguf-py from the pinned llama.cpp checkout:
    uv run --with <llama.cpp>/gguf-py python make_variant.py ...
"""
import argparse
import json
import re
import shutil
import sys
import time
from pathlib import Path

import numpy as np
from gguf import GGUFReader
from gguf.constants import GGML_QUANT_SIZES
from gguf.quants import dequantize

GATE_RE = re.compile(r"^blk\.(\d+)\.ffn_gate_exps\.weight$")


def load_imatrix(path: Path) -> dict[str, np.ndarray]:
    """name -> float32 array. Keys look like 'blk.0.ffn_down_exps.weight.in_sum2' and '.counts'."""
    reader = GGUFReader(str(path))
    return {t.name: np.array(t.data, dtype=np.float32) for t in reader.tensors}


def ne(tensor) -> tuple[int, ...]:
    """ggml dimension order: ne[0] is the contiguous row length. (gguf-py's .shape is in this order;
    .data is the numpy view with the dimensions reversed.)"""
    return tuple(int(x) for x in tensor.shape)


def row_bytes(tensor) -> int:
    block, size = GGML_QUANT_SIZES[tensor.tensor_type]
    n_per_row = ne(tensor)[0]
    if n_per_row % block:
        raise ValueError(f"{tensor.name}: row of {n_per_row} is not a whole number of {block}-element blocks")
    return n_per_row // block * size


def importance(down, in_sum2: np.ndarray | None, counts: np.ndarray | None):
    """Return (importance[e, i], n_experts_without_stats)."""
    w = dequantize(down.data, down.tensor_type).astype(np.float32)  # [n_expert, n_embd, n_ff]
    col_norm = np.sqrt((w.astype(np.float64) ** 2).sum(axis=1))      # [n_expert, n_ff]
    if in_sum2 is None or counts is None:
        return col_norm, col_norm.shape[0]
    n = counts.reshape(-1)
    have = n > 0
    mean_h2 = np.zeros_like(in_sum2, dtype=np.float64)
    mean_h2[have] = in_sum2[have] / n[have, None]
    imp = np.sqrt(mean_h2) * col_norm
    # experts the calibration never routed to: fall back to weight norm alone
    imp[~have] = col_norm[~have]
    return imp, int((~have).sum())


def choose_keep(imp: np.ndarray, n_keep: int, select: str, rng: np.random.Generator) -> np.ndarray:
    """Boolean [n_expert, n_ff] mask of neurons kept."""
    n_expert, n_ff = imp.shape
    keep = np.zeros((n_expert, n_ff), dtype=bool)
    for e in range(n_expert):
        if select == "importance":
            idx = np.argpartition(-imp[e], n_keep - 1)[:n_keep]
        elif select == "random":
            idx = rng.choice(n_ff, size=n_keep, replace=False)
        elif select == "prefix":  # the first n_keep neurons in stored order: no importance information
            idx = np.arange(n_keep)
        else:
            raise ValueError(select)
        keep[e, idx] = True
    return keep


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", required=True, type=Path)
    ap.add_argument("--imatrix", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--k", required=True, type=int, help="chunks kept, 1..chunks")
    ap.add_argument("--chunks", type=int, default=3)
    ap.add_argument("--select", choices=["importance", "random", "prefix"], default="importance")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--force", action="store_true", help="overwrite --out")
    args = ap.parse_args()

    if not 1 <= args.k <= args.chunks:
        print("--k must be between 1 and --chunks", file=sys.stderr)
        return 2
    if args.out.exists() and not args.force:
        print(f"{args.out} exists; use --force", file=sys.stderr)
        return 2

    base = GGUFReader(str(args.base))
    gates = {}
    for t in base.tensors:
        m = GATE_RE.match(t.name)
        if m:
            gates[int(m.group(1))] = t
    if not gates:
        print("no blk.N.ffn_gate_exps.weight tensors found: is this a merged-expert MoE GGUF?", file=sys.stderr)
        return 1
    by_name = {t.name: t for t in base.tensors}

    imx = load_imatrix(args.imatrix)

    # shape check on the first layer
    t0 = gates[min(gates)]
    _, n_ff, n_expert = ne(t0)           # gate/up: ne = [n_embd, n_ff, n_expert]
    if n_ff % args.chunks:
        print(f"n_ff={n_ff} is not divisible by --chunks={args.chunks}", file=sys.stderr)
        return 2
    chunk = n_ff // args.chunks
    n_keep = chunk * args.k
    print(f"{len(gates)} MoE layers, {n_expert} experts, n_ff={n_ff}, chunk={chunk}; keep {n_keep}/{n_ff} neurons ({100*n_keep/n_ff:.0f}%) by '{args.select}'")

    # disk check, then copy
    need = args.base.stat().st_size
    free = shutil.disk_usage(args.out.parent if args.out.parent.exists() else ".").free
    if free < need * 1.05:
        print(f"not enough free disk: need {need/1e9:.1f} GB, have {free/1e9:.1f} GB", file=sys.stderr)
        return 1
    args.out.parent.mkdir(parents=True, exist_ok=True)
    t = time.time()
    print(f"copying {need/1e9:.1f} GB ...")
    shutil.copyfile(args.base, args.out)
    print(f"copied in {time.time()-t:.0f} s")

    rng = np.random.default_rng(args.seed)
    mm = np.memmap(args.out, dtype=np.uint8, mode="r+")
    stats = {"layers": len(gates), "experts": n_expert, "n_ff": n_ff, "chunk": chunk, "k": args.k,
             "select": args.select, "kept_neurons": n_keep, "experts_without_calibration_stats": 0,
             "min_expert_count": None, "importance_mass_kept": [], "layers_detail": {}}
    min_count = None

    for il in sorted(gates):
        gate = gates[il]
        up = by_name[f"blk.{il}.ffn_up_exps.weight"]
        down = by_name[f"blk.{il}.ffn_down_exps.weight"]
        key = f"blk.{il}.ffn_down_exps.weight"
        in_sum2 = imx.get(key + ".in_sum2")
        counts = imx.get(key + ".counts")
        if in_sum2 is not None and in_sum2.shape != (n_expert, n_ff):
            print(f"layer {il}: imatrix shape {in_sum2.shape} != ({n_expert}, {n_ff})", file=sys.stderr)
            return 1
        imp, missing = importance(down, in_sum2, counts)
        stats["experts_without_calibration_stats"] += missing
        if counts is not None:
            c = int(counts.min())
            min_count = c if min_count is None else min(min_count, c)

        keep = choose_keep(imp, n_keep, args.select, rng)
        mass = float((imp * keep).sum() / imp.sum())
        stats["importance_mass_kept"].append(mass)

        drop = ~keep
        for tens in (gate, up):
            rb = row_bytes(tens)
            if ne(tens)[1:] != (n_ff, n_expert):
                print(f"layer {il}: unexpected shape {ne(tens)} for {tens.name}", file=sys.stderr)
                return 1
            off = int(tens.data_offset)
            view = mm[off: off + n_expert * n_ff * rb].reshape(n_expert, n_ff, rb)
            view[drop] = 0
        print(f"layer {il:2d}: importance mass kept {100*mass:5.1f}%  experts without stats: {missing}")

    mm.flush()
    del mm
    stats["min_expert_count"] = min_count
    stats["importance_mass_kept_mean"] = float(np.mean(stats["importance_mass_kept"]))
    out_json = args.out.with_suffix(".variant.json")
    out_json.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(f"done: {args.out}  (summary in {out_json.name})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
