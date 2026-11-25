"""Self-test for make_variant.py on a tiny synthetic MoE GGUF.

Run:  uv run --with <llama.cpp>/gguf-py --with pytest pytest -q test_make_variant.py
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from gguf import GGUFReader, GGUFWriter
from gguf.constants import GGML_QUANT_SIZES, GGMLQuantizationType
from gguf.quants import dequantize, quantize

HERE = Path(__file__).parent
N_LAYER, N_EXPERT, N_FF, N_EMBD, CHUNKS = 2, 4, 96, 64, 3
CHUNK = N_FF // CHUNKS
QT = GGMLQuantizationType.Q8_0


def make_base(path: Path, rng: np.random.Generator):
    w = GGUFWriter(str(path), "qwen3moe")
    for il in range(N_LAYER):
        gate = rng.normal(0, 0.05, (N_EXPERT, N_FF, N_EMBD)).astype(np.float32)
        up = rng.normal(0, 0.05, (N_EXPERT, N_FF, N_EMBD)).astype(np.float32)
        down = rng.normal(0, 0.05, (N_EXPERT, N_EMBD, N_FF)).astype(np.float32)
        for name, arr in (("gate", gate), ("up", up), ("down", down)):
            w.add_tensor(f"blk.{il}.ffn_{name}_exps.weight", quantize(arr, QT), raw_dtype=QT)
        w.add_tensor(f"blk.{il}.attn_q.weight", rng.normal(0, 1, (8, 32)).astype(np.float32))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()


def make_imatrix(path: Path, rng: np.random.Generator, zero_count=(1, 2)):
    """in_sum2 [n_expert, n_ff] and counts [n_expert, 1] per layer. One expert has no calibration data."""
    w = GGUFWriter(str(path), "imatrix")
    for il in range(N_LAYER):
        in_sum2 = rng.uniform(0.1, 10.0, (N_EXPERT, N_FF)).astype(np.float32)
        counts = np.full((N_EXPERT, 1), 100.0, dtype=np.float32)
        if il == zero_count[0]:
            counts[zero_count[1]] = 0.0
        w.add_tensor(f"blk.{il}.ffn_down_exps.weight.in_sum2", in_sum2)
        w.add_tensor(f"blk.{il}.ffn_down_exps.weight.counts", counts)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()


def tensors(path: Path):
    return {t.name: t for t in GGUFReader(str(path)).tensors}


def expected_keep(base_tensors, imx_tensors, il, n_keep):
    down = base_tensors[f"blk.{il}.ffn_down_exps.weight"]
    wd = dequantize(down.data, down.tensor_type).astype(np.float64)
    col_norm = np.sqrt((wd ** 2).sum(axis=1))
    in_sum2 = np.array(imx_tensors[f"blk.{il}.ffn_down_exps.weight.in_sum2"].data, dtype=np.float64)
    counts = np.array(imx_tensors[f"blk.{il}.ffn_down_exps.weight.counts"].data, dtype=np.float64).reshape(-1)
    keep = np.zeros((N_EXPERT, N_FF), dtype=bool)
    for e in range(N_EXPERT):
        imp = col_norm[e] if counts[e] == 0 else np.sqrt(in_sum2[e] / counts[e]) * col_norm[e]
        keep[e, np.argsort(-imp)[:n_keep]] = True
    return keep


@pytest.fixture(scope="module")
def files(tmp_path_factory):
    d = tmp_path_factory.mktemp("g3")
    rng = np.random.default_rng(0)
    make_base(d / "base.gguf", rng)
    make_imatrix(d / "imx.gguf", rng)
    return d


def run(files, k, select="importance", name=None):
    out = files / (name or f"k{k}_{select}.gguf")
    cmd = [sys.executable, str(HERE / "make_variant.py"), "--base", str(files / "base.gguf"),
           "--imatrix", str(files / "imx.gguf"), "--out", str(out), "--k", str(k),
           "--chunks", str(CHUNKS), "--select", select, "--force"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    return out


def rows_zero(t):
    d = np.asarray(t.data).reshape(N_EXPERT, N_FF, -1)
    return (d == 0).all(axis=2)


@pytest.mark.parametrize("k", [1, 2])
def test_importance_selection_matches_independent_computation(files, k):
    out = run(files, k)
    base, imx, var = tensors(files / "base.gguf"), tensors(files / "imx.gguf"), tensors(out)
    for il in range(N_LAYER):
        keep = expected_keep(base, imx, il, CHUNK * k)
        for name in ("gate", "up"):
            nm = f"blk.{il}.ffn_{name}_exps.weight"
            zero = rows_zero(var[nm])
            assert (zero == ~keep).all(), f"{nm}: zeroed rows differ from expected drop set"
            # kept rows are byte-identical to the base
            b = np.asarray(base[nm].data).reshape(N_EXPERT, N_FF, -1)
            v = np.asarray(var[nm].data).reshape(N_EXPERT, N_FF, -1)
            assert (b[keep] == v[keep]).all()


def test_other_tensors_untouched(files):
    out = run(files, 1)
    base, var = tensors(files / "base.gguf"), tensors(out)
    for il in range(N_LAYER):
        for nm in (f"blk.{il}.ffn_down_exps.weight", f"blk.{il}.attn_q.weight"):
            assert (np.asarray(base[nm].data) == np.asarray(var[nm].data)).all(), nm


def test_zeroed_rows_dequantize_to_exact_prefix_width_output(files):
    """The variant's FFN output equals the output computed from only the kept neurons of the base."""
    out = run(files, 2)
    base, imx, var = tensors(files / "base.gguf"), tensors(files / "imx.gguf"), tensors(out)
    rng = np.random.default_rng(5)
    x = rng.normal(0, 1, N_EMBD)
    il = 0
    keep = expected_keep(base, imx, il, CHUNK * 2)

    def deq(ts, name):
        t = ts[f"blk.{il}.ffn_{name}_exps.weight"]
        return dequantize(t.data, t.tensor_type).astype(np.float64)

    gb, ub, db = deq(base, "gate"), deq(base, "up"), deq(base, "down")
    gv, uv, dv = deq(var, "gate"), deq(var, "up"), deq(var, "down")
    silu = lambda z: z / (1 + np.exp(-z))
    for e in range(N_EXPERT):
        h_var = silu(gv[e] @ x) * (uv[e] @ x)       # [n_ff], dropped neurons are exactly 0
        y_var = dv[e] @ h_var
        idx = np.where(keep[e])[0]
        h_ref = silu(gb[e][idx] @ x) * (ub[e][idx] @ x)
        y_ref = db[e][:, idx] @ h_ref
        assert np.allclose(y_var, y_ref, atol=1e-9), f"expert {e}"
        assert (h_var[~keep[e]] == 0).all()


def test_random_and_prefix_controls(files):
    rnd = run(files, 1, "random", "rand.gguf")
    pre = run(files, 1, "prefix", "pre.gguf")
    for out in (rnd, pre):
        v = tensors(out)
        for il in range(N_LAYER):
            zero = rows_zero(v[f"blk.{il}.ffn_gate_exps.weight"])
            assert ((~zero).sum(axis=1) == CHUNK).all()
    zero = rows_zero(tensors(pre)["blk.0.ffn_gate_exps.weight"])
    assert (~zero[:, :CHUNK]).all() and zero[:, CHUNK:].all()


def test_summary_reports_expert_without_stats(files):
    out = run(files, 1)
    s = json.loads(out.with_suffix(".variant.json").read_text())
    assert s["experts_without_calibration_stats"] == 1
    assert s["min_expert_count"] == 0
    assert s["kept_neurons"] == CHUNK
    assert 0.0 < s["importance_mass_kept_mean"] <= 1.0


def test_q4_k_gate_up_rows_are_zeroed_with_correct_row_size(tmp_path):
    """Qwen3 Q4_K_M files store gate/up as Q4_K (144 bytes per 256 weights). Check the row arithmetic
    on random Q4_K bytes: gate/up rows are n_embd=256 wide, so one row = 1 block = 144 bytes."""
    rng = np.random.default_rng(3)
    n_embd = 256
    raw_row = GGML_QUANT_SIZES[GGMLQuantizationType.Q4_K][1] * (n_embd // 256)
    assert raw_row == 144
    w = GGUFWriter(str(tmp_path / "base.gguf"), "qwen3moe")
    gate = rng.integers(1, 255, (N_EXPERT, N_FF, raw_row), dtype=np.uint8)   # never all-zero rows
    up = rng.integers(1, 255, (N_EXPERT, N_FF, raw_row), dtype=np.uint8)
    down = rng.normal(0, 0.05, (N_EXPERT, n_embd, N_FF)).astype(np.float32)
    w.add_tensor("blk.0.ffn_gate_exps.weight", gate, raw_dtype=GGMLQuantizationType.Q4_K)
    w.add_tensor("blk.0.ffn_up_exps.weight", up, raw_dtype=GGMLQuantizationType.Q4_K)
    w.add_tensor("blk.0.ffn_down_exps.weight", quantize(down, QT), raw_dtype=QT)
    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()

    im = GGUFWriter(str(tmp_path / "imx.gguf"), "imatrix")
    im.add_tensor("blk.0.ffn_down_exps.weight.in_sum2", rng.uniform(0.1, 5, (N_EXPERT, N_FF)).astype(np.float32))
    im.add_tensor("blk.0.ffn_down_exps.weight.counts", np.full((N_EXPERT, 1), 50.0, dtype=np.float32))
    im.write_header_to_file(); im.write_kv_data_to_file(); im.write_tensors_to_file(); im.close()

    out = tmp_path / "k1.gguf"
    r = subprocess.run([sys.executable, str(HERE / "make_variant.py"), "--base", str(tmp_path / "base.gguf"),
                        "--imatrix", str(tmp_path / "imx.gguf"), "--out", str(out), "--k", "1",
                        "--chunks", str(CHUNKS)], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    v = tensors(out)
    for nm, orig in (("gate", gate), ("up", up)):
        d = np.asarray(v[f"blk.0.ffn_{nm}_exps.weight"].data)
        assert d.shape == orig.shape
        zero = (d == 0).all(axis=2)
        assert (zero.sum(axis=1) == N_FF - CHUNK).all()          # exactly the dropped rows are zero
        assert (d[~zero] == orig[~zero]).all()                    # kept rows untouched
