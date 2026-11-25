// Task 1.6 (gate G2a): chunked expert FFN on a ggml backend.
//
// Question: if each expert's intermediate dimension is stored as C chunks, and a
// width-k run computes only chunks 0..k-1 and sums their down-projection outputs,
// (a) is the result correct, and (b) what does it cost per MoE layer on this backend?
//
// Checks
//   1. k == C chunked output equals the fused (single tensor) expert output.
//   2. k < C chunked output equals a plain f32 reference that uses only the first
//      k*S neurons of each routed expert (the "prefix width" definition).
//   3. Time per call for: fused, and chunked k = 1..C.
//
// The expert math follows llama.cpp's build_moe_ffn: gate/up via mul_mat_id,
// swiglu_split, down via mul_mat_id. Router weights are not part of this spike.
//
// Build: see CMakeLists.txt and README.md in this folder.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

struct options {
    std::string backend   = "CPU";   // ggml device name: CPU, CUDA0, Vulkan0, ...
    std::string type_name = "q4_K";  // f32 | f16 | q8_0 | q4_K
    int64_t n_embd   = 2048;
    int64_t n_ff     = 768;          // Qwen3-30B-A3B moe_intermediate_size
    int64_t n_expert = 128;
    int64_t n_used   = 8;
    int64_t n_tokens = 1;            // 1 = decode, larger = prefill
    int     n_chunks = 3;
    int     iters    = 50;
    int     warmup   = 5;
    uint32_t seed    = 1234;
    bool    list_devices = false;
};

static ggml_type parse_type(const std::string & s) {
    if (s == "f32")  return GGML_TYPE_F32;
    if (s == "f16")  return GGML_TYPE_F16;
    if (s == "q8_0") return GGML_TYPE_Q8_0;
    if (s == "q4_K") return GGML_TYPE_Q4_K;
    fprintf(stderr, "unknown --type %s (use f32, f16, q8_0, q4_K)\n", s.c_str());
    exit(2);
}

static bool parse_args(int argc, char ** argv, options & o) {
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&](const char * name) -> const char * {
            if (i + 1 >= argc) { fprintf(stderr, "missing value for %s\n", name); exit(2); }
            return argv[++i];
        };
        if      (a == "--backend")  o.backend   = next("--backend");
        else if (a == "--type")     o.type_name = next("--type");
        else if (a == "--n-embd")   o.n_embd    = atoll(next("--n-embd"));
        else if (a == "--n-ff")     o.n_ff      = atoll(next("--n-ff"));
        else if (a == "--n-expert") o.n_expert  = atoll(next("--n-expert"));
        else if (a == "--n-used")   o.n_used    = atoll(next("--n-used"));
        else if (a == "--n-tokens") o.n_tokens  = atoll(next("--n-tokens"));
        else if (a == "--chunks")   o.n_chunks  = atoi(next("--chunks"));
        else if (a == "--iters")    o.iters     = atoi(next("--iters"));
        else if (a == "--warmup")   o.warmup    = atoi(next("--warmup"));
        else if (a == "--seed")     o.seed      = (uint32_t) atoll(next("--seed"));
        else if (a == "--list-devices") o.list_devices = true;
        else { fprintf(stderr, "unknown argument %s\n", a.c_str()); return false; }
    }
    return true;
}

// ---------------------------------------------------------------------------
// host-side weights

// Row-major f32 weights in ggml order: ne0 is the contiguous dimension.
//   gate/up: ne = [n_embd, n_ff, n_expert]  -> row (e, i) = neuron i of expert e, length n_embd
//   down:    ne = [n_ff, n_embd, n_expert]  -> row (e, r) = output r of expert e, length n_ff
struct host_weights {
    std::vector<float> gate, up, down;
};

static void fill_gaussian(std::vector<float> & v, std::mt19937 & rng, float stddev) {
    std::normal_distribution<float> d(0.0f, stddev);
    for (auto & x : v) { x = d(rng); }
}

// Quantize rows of n_per_row elements into the given type. Returns the raw bytes.
static std::vector<uint8_t> to_type(const float * src, int64_t nrows, int64_t n_per_row, ggml_type type) {
    const size_t row_bytes = ggml_row_size(type, n_per_row);
    std::vector<uint8_t> out(row_bytes * nrows);
    if (type == GGML_TYPE_F32) {
        memcpy(out.data(), src, out.size());
    } else if (type == GGML_TYPE_F16) {
        ggml_fp32_to_fp16_row(src, (ggml_fp16_t *) out.data(), nrows * n_per_row);
    } else {
        const size_t n = ggml_quantize_chunk(type, src, out.data(), 0, nrows, n_per_row, nullptr);
        if (n != out.size()) { fprintf(stderr, "quantize size mismatch %zu vs %zu\n", n, out.size()); exit(1); }
    }
    return out;
}

// Dequantize a row-major tensor back to f32 so the reference sees the same values the kernels see.
static std::vector<float> from_type(const std::vector<uint8_t> & raw, int64_t nrows, int64_t n_per_row, ggml_type type) {
    std::vector<float> out(nrows * n_per_row);
    if (type == GGML_TYPE_F32) {
        memcpy(out.data(), raw.data(), raw.size());
    } else if (type == GGML_TYPE_F16) {
        ggml_fp16_to_fp32_row((const ggml_fp16_t *) raw.data(), out.data(), nrows * n_per_row);
    } else {
        const auto * tr = ggml_get_type_traits(type);
        const size_t row_bytes = ggml_row_size(type, n_per_row);
        for (int64_t r = 0; r < nrows; ++r) {
            tr->to_float(raw.data() + r * row_bytes, out.data() + r * n_per_row, n_per_row);
        }
    }
    return out;
}

// ---------------------------------------------------------------------------
// graph construction

struct expert_tensors {
    ggml_tensor * gate = nullptr;
    ggml_tensor * up   = nullptr;
    ggml_tensor * down = nullptr;
};

// cur: [n_embd, 1, n_tokens], ids: [n_used, n_tokens] -> [n_embd, n_used, n_tokens]
static ggml_tensor * build_expert(ggml_context * ctx, const expert_tensors & w, ggml_tensor * cur, ggml_tensor * ids) {
    ggml_tensor * g = ggml_mul_mat_id(ctx, w.gate, cur, ids);   // [ff, n_used, n_tokens]
    ggml_tensor * u = ggml_mul_mat_id(ctx, w.up,   cur, ids);
    ggml_tensor * h = ggml_swiglu_split(ctx, g, u);
    return ggml_mul_mat_id(ctx, w.down, h, ids);                // [n_embd, n_used, n_tokens]
}

// Each graph owns its allocator and is allocated once. Sharing one allocator between graphs
// would invalidate the buffers of every graph but the last one allocated.
struct graph_pack {
    ggml_context *  ctx    = nullptr;
    ggml_cgraph *   graph  = nullptr;
    ggml_tensor *   out    = nullptr;
    ggml_gallocr_t  galloc = nullptr;
    std::vector<uint8_t> buf;
};

static graph_pack make_graph(ggml_backend_t backend, const std::vector<expert_tensors> & chunks, int k,
                             ggml_tensor * x, ggml_tensor * ids) {
    graph_pack g;
    g.buf.resize(ggml_tensor_overhead() * 256 + ggml_graph_overhead() + 4096);
    ggml_init_params p = { g.buf.size(), g.buf.data(), true };
    g.ctx = ggml_init(p);
    g.graph = ggml_new_graph(g.ctx);

    ggml_tensor * out = nullptr;
    for (int j = 0; j < k; ++j) {
        ggml_tensor * d = build_expert(g.ctx, chunks[j], x, ids);
        out = out ? ggml_add(g.ctx, out, d) : d;
    }
    ggml_set_output(out);
    ggml_build_forward_expand(g.graph, out);
    g.out = out;

    g.galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    if (!ggml_gallocr_alloc_graph(g.galloc, g.graph)) {
        fprintf(stderr, "graph allocation failed\n");
        exit(1);
    }
    return g;
}

static void free_graph(graph_pack & g) {
    ggml_gallocr_free(g.galloc);
    ggml_free(g.ctx);
}

// ---------------------------------------------------------------------------
// reference: plain f32 expert FFN restricted to the first n_keep neurons

static void reference_prefix(const options & o, const std::vector<float> & gate, const std::vector<float> & up,
                             const std::vector<float> & down, const std::vector<float> & x,
                             const std::vector<int32_t> & ids, int64_t n_keep, std::vector<float> & out) {
    // out: [n_embd, n_used, n_tokens]
    out.assign(o.n_embd * o.n_used * o.n_tokens, 0.0f);
    std::vector<float> h(n_keep);
    for (int64_t t = 0; t < o.n_tokens; ++t) {
        const float * xt = x.data() + t * o.n_embd;
        for (int64_t s = 0; s < o.n_used; ++s) {
            const int64_t e = ids[t * o.n_used + s];
            for (int64_t i = 0; i < n_keep; ++i) {
                const float * gr = gate.data() + (e * o.n_ff + i) * o.n_embd;
                const float * ur = up.data()   + (e * o.n_ff + i) * o.n_embd;
                double gs = 0.0, us = 0.0;
                for (int64_t c = 0; c < o.n_embd; ++c) { gs += (double) gr[c] * xt[c]; us += (double) ur[c] * xt[c]; }
                const double silu = gs / (1.0 + std::exp(-gs));
                h[i] = (float) (silu * us);
            }
            float * dst = out.data() + (t * o.n_used + s) * o.n_embd;
            for (int64_t r = 0; r < o.n_embd; ++r) {
                const float * dr = down.data() + (e * o.n_embd + r) * o.n_ff;
                double acc = 0.0;
                for (int64_t i = 0; i < n_keep; ++i) { acc += (double) dr[i] * h[i]; }
                dst[r] = (float) acc;
            }
        }
    }
}

struct diff_stats { double max_abs = 0, max_ref = 0, rel_rms = 0; };

static diff_stats compare(const std::vector<float> & a, const std::vector<float> & b) {
    diff_stats s;
    double num = 0, den = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        s.max_abs = std::max(s.max_abs, (double) std::fabs(a[i] - b[i]));
        s.max_ref = std::max(s.max_ref, (double) std::fabs(b[i]));
        num += (double) (a[i] - b[i]) * (a[i] - b[i]);
        den += (double) b[i] * b[i];
    }
    s.rel_rms = den > 0 ? std::sqrt(num / den) : 0.0;
    return s;
}

static double time_graph(ggml_backend_t backend, ggml_cgraph * graph, int warmup, int iters) {
    for (int i = 0; i < warmup; ++i) { ggml_backend_graph_compute(backend, graph); }
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < iters; ++i) { ggml_backend_graph_compute(backend, graph); }
    const auto t1 = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(t1 - t0).count() / iters;
}

int main(int argc, char ** argv) {
    options o;
    if (!parse_args(argc, argv, o)) { return 2; }

    setvbuf(stdout, nullptr, _IONBF, 0); // show progress immediately, also when output is redirected

    ggml_backend_load_all();

    if (o.list_devices) {
        for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
            ggml_backend_dev_t d = ggml_backend_dev_get(i);
            size_t free_b = 0, total_b = 0;
            ggml_backend_dev_memory(d, &free_b, &total_b);
            printf("%s  (%s)  free %.2f GiB / total %.2f GiB\n", ggml_backend_dev_name(d), ggml_backend_dev_description(d),
                   free_b / 1073741824.0, total_b / 1073741824.0);
        }
        return 0;
    }

    const ggml_type type = parse_type(o.type_name);
    const int C = o.n_chunks;
    if (o.n_ff % C != 0) { fprintf(stderr, "n_ff %lld not divisible by chunks %d\n", (long long) o.n_ff, C); return 2; }
    const int64_t S = o.n_ff / C;
    // quantized row length must be a multiple of the block size, for both the fused and the chunked down tensors
    const int64_t blck = ggml_blck_size(type);
    if (o.n_embd % blck != 0 || S % blck != 0) {
        fprintf(stderr, "chunk size S=%lld or n_embd=%lld is not a multiple of block size %lld for type %s\n",
                (long long) S, (long long) o.n_embd, (long long) blck, o.type_name.c_str());
        return 2;
    }
    if (o.n_used > o.n_expert) { fprintf(stderr, "n_used > n_expert\n"); return 2; }

    ggml_backend_t backend = ggml_backend_init_by_name(o.backend.c_str(), nullptr);
    if (!backend) { fprintf(stderr, "could not init backend '%s'; try --list-devices\n", o.backend.c_str()); return 1; }

    printf("backend=%s type=%s n_embd=%lld n_ff=%lld n_expert=%lld n_used=%lld n_tokens=%lld chunks=%d (S=%lld)\n",
           ggml_backend_name(backend), o.type_name.c_str(), (long long) o.n_embd, (long long) o.n_ff,
           (long long) o.n_expert, (long long) o.n_used, (long long) o.n_tokens, C, (long long) S);

    // ----- host weights (f32), then converted to the test type -----
    std::mt19937 rng(o.seed);
    host_weights hw;
    hw.gate.resize(o.n_embd * o.n_ff * o.n_expert);
    hw.up.resize(hw.gate.size());
    hw.down.resize(hw.gate.size());
    fill_gaussian(hw.gate, rng, 0.03f);
    fill_gaussian(hw.up,   rng, 0.03f);
    fill_gaussian(hw.down, rng, 0.03f);

    std::vector<float> x(o.n_embd * o.n_tokens);
    fill_gaussian(x, rng, 1.0f);

    std::vector<int32_t> ids(o.n_used * o.n_tokens);
    for (int64_t t = 0; t < o.n_tokens; ++t) {
        std::vector<int32_t> perm(o.n_expert);
        for (int64_t e = 0; e < o.n_expert; ++e) { perm[e] = (int32_t) e; }
        std::shuffle(perm.begin(), perm.end(), rng);
        for (int64_t s = 0; s < o.n_used; ++s) { ids[t * o.n_used + s] = perm[s]; }
    }

    // fused tensors in the test type
    auto gate_q = to_type(hw.gate.data(), o.n_ff * o.n_expert,   o.n_embd, type);
    auto up_q   = to_type(hw.up.data(),   o.n_ff * o.n_expert,   o.n_embd, type);
    auto down_q = to_type(hw.down.data(), o.n_embd * o.n_expert, o.n_ff,   type);

    // chunk j: gate/up rows (e, j*S .. (j+1)*S) are contiguous per expert; down needs a column slice of every row.
    std::vector<std::vector<uint8_t>> gate_c(C), up_c(C), down_c(C);
    for (int j = 0; j < C; ++j) {
        std::vector<float> g(o.n_embd * S * o.n_expert), u(g.size()), d(S * o.n_embd * o.n_expert);
        for (int64_t e = 0; e < o.n_expert; ++e) {
            memcpy(g.data() + e * S * o.n_embd, hw.gate.data() + (e * o.n_ff + j * S) * o.n_embd, sizeof(float) * S * o.n_embd);
            memcpy(u.data() + e * S * o.n_embd, hw.up.data()   + (e * o.n_ff + j * S) * o.n_embd, sizeof(float) * S * o.n_embd);
            for (int64_t r = 0; r < o.n_embd; ++r) {
                memcpy(d.data() + (e * o.n_embd + r) * S, hw.down.data() + (e * o.n_embd + r) * o.n_ff + j * S, sizeof(float) * S);
            }
        }
        gate_c[j] = to_type(g.data(), S * o.n_expert,      o.n_embd, type);
        up_c[j]   = to_type(u.data(), S * o.n_expert,      o.n_embd, type);
        down_c[j] = to_type(d.data(), o.n_embd * o.n_expert, S,      type);
    }

    // ----- device tensors -----
    const size_t n_w = 3 + 3 * C + 2;
    std::vector<uint8_t> wbuf(ggml_tensor_overhead() * (n_w + 8));
    ggml_init_params wp = { wbuf.size(), wbuf.data(), true };
    ggml_context * wctx = ggml_init(wp);

    expert_tensors fused;
    fused.gate = ggml_new_tensor_3d(wctx, type, o.n_embd, o.n_ff,   o.n_expert);
    fused.up   = ggml_new_tensor_3d(wctx, type, o.n_embd, o.n_ff,   o.n_expert);
    fused.down = ggml_new_tensor_3d(wctx, type, o.n_ff,   o.n_embd, o.n_expert);
    std::vector<expert_tensors> chunks(C);
    for (int j = 0; j < C; ++j) {
        chunks[j].gate = ggml_new_tensor_3d(wctx, type, o.n_embd, S,        o.n_expert);
        chunks[j].up   = ggml_new_tensor_3d(wctx, type, o.n_embd, S,        o.n_expert);
        chunks[j].down = ggml_new_tensor_3d(wctx, type, S,        o.n_embd, o.n_expert);
    }
    ggml_tensor * tx   = ggml_new_tensor_3d(wctx, GGML_TYPE_F32, o.n_embd, 1, o.n_tokens);
    ggml_tensor * tids = ggml_new_tensor_2d(wctx, GGML_TYPE_I32, o.n_used, o.n_tokens);

    ggml_backend_buffer_t wbuffer = ggml_backend_alloc_ctx_tensors(wctx, backend);
    if (!wbuffer) { fprintf(stderr, "could not allocate weights on %s (out of memory?)\n", ggml_backend_name(backend)); return 1; }
    printf("weights on device: %.1f MiB (fused) + chunked copy, total buffer %.1f MiB\n",
           (gate_q.size() + up_q.size() + down_q.size()) / 1048576.0, ggml_backend_buffer_get_size(wbuffer) / 1048576.0);

    ggml_backend_tensor_set(fused.gate, gate_q.data(), 0, gate_q.size());
    ggml_backend_tensor_set(fused.up,   up_q.data(),   0, up_q.size());
    ggml_backend_tensor_set(fused.down, down_q.data(), 0, down_q.size());
    for (int j = 0; j < C; ++j) {
        ggml_backend_tensor_set(chunks[j].gate, gate_c[j].data(), 0, gate_c[j].size());
        ggml_backend_tensor_set(chunks[j].up,   up_c[j].data(),   0, up_c[j].size());
        ggml_backend_tensor_set(chunks[j].down, down_c[j].data(), 0, down_c[j].size());
    }
    ggml_backend_tensor_set(tx,   x.data(),   0, x.size() * sizeof(float));
    ggml_backend_tensor_set(tids, ids.data(), 0, ids.size() * sizeof(int32_t));

    // ----- graphs: fused, and chunked k = 1..C -----
    std::vector<expert_tensors> fused_as_chunks = { fused };
    graph_pack gf = make_graph(backend, fused_as_chunks, 1, tx, tids);
    std::vector<graph_pack> gk;
    gk.reserve(C);
    for (int k = 1; k <= C; ++k) { gk.push_back(make_graph(backend, chunks, k, tx, tids)); }

    auto run = [&](graph_pack & g, std::vector<float> & result) {
        ggml_backend_graph_compute(backend, g.graph);
        result.resize(ggml_nelements(g.out));
        ggml_backend_tensor_get(g.out, result.data(), 0, result.size() * sizeof(float));
    };

    // ----- correctness -----
    // f32 and f16 are tight. Quantized types quantize the activations inside mul_mat_id, so the
    // chunked-vs-fused numbers stay tight but the f32 reference differs by activation quantization noise.
    const bool tight = (type == GGML_TYPE_F32 || type == GGML_TYPE_F16);
    const double tol_chunk_vs_fused = tight ? 1e-3 : 2e-2;
    const double tol_vs_reference   = tight ? 2e-3 : 8e-2;

    std::vector<float> out_fused;
    run(gf, out_fused);

    // dequantized weights as the kernels see them, used by the reference
    auto gate_ref = from_type(gate_q, o.n_ff * o.n_expert,   o.n_embd, type);
    auto up_ref   = from_type(up_q,   o.n_ff * o.n_expert,   o.n_embd, type);
    auto down_ref = from_type(down_q, o.n_embd * o.n_expert, o.n_ff,   type);

    bool ok = true;
    printf("\ncorrectness (relative RMS error)\n");
    for (int k = 1; k <= C; ++k) {
        std::vector<float> out_k, ref_k;
        run(gk[k - 1], out_k);
        reference_prefix(o, gate_ref, up_ref, down_ref, x, ids, (int64_t) k * S, ref_k);
        const diff_stats vs_ref = compare(out_k, ref_k);
        printf("  k=%d (%3d%% width): vs f32 prefix reference %.3e", k, (int) (100.0 * k / C), vs_ref.rel_rms);
        bool pass = vs_ref.rel_rms <= tol_vs_reference;
        if (k == C) {
            const diff_stats vs_fused = compare(out_k, out_fused);
            printf(" | vs fused expert %.3e", vs_fused.rel_rms);
            pass = pass && vs_fused.rel_rms <= tol_chunk_vs_fused;
        }
        printf("  %s\n", pass ? "PASS" : "FAIL");
        ok = ok && pass;
    }

    // a narrower width must actually change the result, otherwise the test proves nothing
    {
        std::vector<float> a, b;
        run(gk[0], a);
        run(gk[C - 1], b);
        const diff_stats d = compare(a, b);
        printf("  sanity: width 1 vs width %d differ by %.3e (must be well above 0)\n", C, d.rel_rms);
        ok = ok && d.rel_rms > 0.05;
    }

    // ----- timing -----
    printf("\ntiming, ms per call (one MoE layer: gate+up+swiglu+down for %lld routed experts, %lld token(s))\n",
           (long long) o.n_used, (long long) o.n_tokens);
    const double t_fused = time_graph(backend, gf.graph, o.warmup, o.iters);
    printf("  fused (1 tensor, 100%% width): %8.4f ms\n", t_fused);
    for (int k = 1; k <= C; ++k) {
        const double t = time_graph(backend, gk[k - 1].graph, o.warmup, o.iters);
        printf("  chunked k=%d (%3d%% width):     %8.4f ms   (%.2fx of fused)\n", k, (int) (100.0 * k / C), t, t / t_fused);
    }
    printf("\nresult: %s\n", ok ? "PASS" : "FAIL");

    free_graph(gf);
    for (auto & g : gk) { free_graph(g); }
    ggml_backend_buffer_free(wbuffer);
    ggml_free(wctx);
    ggml_backend_free(backend);
    return ok ? 0 : 1;
}
