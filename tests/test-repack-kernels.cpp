// Numerical correctness tests and microbenchmarks for the 8x8 repack matmul
// kernels (Q2_K / Q3_K / Q5_K / IQ1_S / IQ1_M weights, Q8_K activations).
//
// Three comparisons per weight type:
//   (a) native x86 kernel vs generic scalar kernel on identical repacked data
//   (b) same for the gemm path across row counts covering the 16-row main
//       block, the 4-row tail block and the odd 8-column group paths
//   (c) gemv/gemm results vs the legacy ggml_vec_dot_*_q8_K per-row reference
//       (verifies that the repack math is equivalent to the original layout)
//
// A perf mode (--perf) times legacy vec_dot vs repack gemv/gemm.

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "repack.h"
#include "quants.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <thread>
#include <utility>
#include <random>
#include <string>
#include <vector>

#if defined(__linux__)
#include <sched.h>
#endif

namespace {

struct diff_stats {
    double max_abs = 0.0;
    double max_rel = 0.0;
    int    n_bad   = 0;
};

void diff_update(diff_stats & st, double a, double b, double tol) {
    const double d = std::fabs(a - b);
    const double r = d / std::max(1.0, std::fabs(b));
    st.max_abs = std::max(st.max_abs, d);
    st.max_rel = std::max(st.max_rel, r);
    if (d > tol * std::max(1.0, std::fabs(b))) {
        st.n_bad++;
    }
}

// Quantize f32 weights row by row and repack them through the real repack
// buffer path (set_tensor triggers the interleave), returning the repacked
// block_q*_Kx8 array plus the original quantized rows for legacy reference.
struct repacked_weights {
    ggml_context *           ctx    = nullptr;
    ggml_backend_buffer_t    buffer = nullptr;
    ggml_tensor *            tensor = nullptr;
    std::vector<char>        raw;      // original block_q*_K rows, nc rows
    const void *             data = nullptr; // repacked block_q*_Kx8 array
    int64_t                  nbytes = 0;
};

bool repack_supported(ggml_type type, int nc, int k) {
    struct ggml_init_params params = { 16 * 1024, nullptr, true };
    ggml_context * ctx = ggml_init(params);
    if (ctx == nullptr) {
        return false;
    }

    ggml_tensor * tensor = ggml_new_tensor_2d(ctx, type, k, nc);
    ggml_backend_buffer_t buffer = ggml_backend_buft_alloc_buffer(
        ggml_backend_cpu_repack_buffer_type(), ggml_nbytes(tensor));
    if (buffer == nullptr) {
        ggml_free(ctx);
        return false;
    }

    ggml_backend_tensor_alloc(buffer, tensor, ggml_backend_buffer_get_base(buffer));
    const bool supported = tensor->extra != nullptr;
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    return supported;
}

bool make_repacked(ggml_type type, const std::vector<float> & w, int nc, int k, repacked_weights & out) {
    const ggml_type_traits * traits = ggml_get_type_traits(type);
    ggml_quantize_init(type);

    const size_t row_bytes = ggml_row_size(type, k);
    out.raw.resize(row_bytes * nc);
    if (traits->from_float_ref) {
        for (int r = 0; r < nc; r++) {
            traits->from_float_ref(w.data() + (size_t) r * k, out.raw.data() + row_bytes * r, k);
        }
    } else {
        // Types without a from_float_ref row quantizer go through the initialized
        // chunk quantizer with a flat importance matrix.
        const std::vector<float> imatrix((size_t) k, 1.0f);
        ggml_quantize_chunk(type, w.data(), out.raw.data(), 0, nc, k, imatrix.data());
    }

    struct ggml_init_params params = { 1 * 1024 * 1024, nullptr, true };
    out.ctx = ggml_init(params);
    if (out.ctx == nullptr) {
        return false;
    }
    out.tensor = ggml_new_tensor_2d(out.ctx, type, k, nc);
    out.nbytes = ggml_nbytes(out.tensor);

    out.buffer = ggml_backend_buft_alloc_buffer(ggml_backend_cpu_repack_buffer_type(), out.nbytes);
    if (out.buffer == nullptr) {
        return false;
    }
    ggml_backend_tensor_alloc(out.buffer, out.tensor, ggml_backend_buffer_get_base(out.buffer));
    if (out.tensor->extra == nullptr) {
        ggml_backend_buffer_free(out.buffer);
        ggml_free(out.ctx);
        out = repacked_weights();
        return false;
    }
    ggml_backend_tensor_set(out.tensor, out.raw.data(), 0, out.raw.size());

    out.data = out.tensor->data;
    return true;
}

void free_repacked(repacked_weights & w) {
    if (w.buffer) ggml_backend_buffer_free(w.buffer);
    if (w.ctx)    ggml_free(w.ctx);
    w = repacked_weights();
}

std::vector<float> make_random_f32(int64_t n, uint32_t seed) {
    std::mt19937                    rng(seed);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    std::vector<float>              v(n);
    for (auto & x : v) x = dist(rng);
    return v;
}

// Quantize nr activation rows of k floats each into the block_q8_{K,0}x4
// interleaved layout used by the gemm kernels (nr must be a multiple of 4).
std::vector<char> quantize_acts_4x8(const std::vector<float> & x, int nr, int k, ggml_type act = GGML_TYPE_Q8_K) {
    const size_t row_size = ggml_row_size(act, k) * 4;
    std::vector<char> vy(row_size * (nr / 4));
    for (int g = 0; g < nr / 4; g++) {
        if (act == GGML_TYPE_Q8_K) {
            ggml_quantize_mat_q8_K_4x8(x.data() + (size_t) g * 4 * k, vy.data() + row_size * g, k);
        } else {
            ggml_quantize_mat_q8_0_4x8(x.data() + (size_t) g * 4 * k, vy.data() + row_size * g, k);
        }
    }
    return vy;
}

// Quantize a single activation row into plain block_q8_K / block_q8_0 blocks (gemv path).
std::vector<char> quantize_act_row(const std::vector<float> & x, int k, ggml_type act = GGML_TYPE_Q8_K) {
    std::vector<char> vy(ggml_row_size(act, k));
    if (act == GGML_TYPE_Q8_K) {
        quantize_row_q8_K(x.data(), vy.data(), k);
    } else {
        quantize_row_q8_0(x.data(), vy.data(), k);
    }
    return vy;
}

typedef void (*vec_dot_fn)(int, float *, size_t, const void *, size_t, const void *, size_t, int);

struct kernel_fns {
    ggml_type    type;
    const char * name;
    vec_dot_fn vec_dot;
    void (*gemv)(int, float *, size_t, const void *, const void *, int, int);
    void (*gemv_generic)(int, float *, size_t, const void *, const void *, int, int);
    void (*gemm)(int, float *, size_t, const void *, const void *, int, int);
    void (*gemm_generic)(int, float *, size_t, const void *, const void *, int, int);
    ggml_type    act_type = GGML_TYPE_Q8_K;
};

bool test_type(const kernel_fns & fn, int nc, int k, const std::vector<int> & gemm_nrs, bool verbose) {
    bool ok = true;

    std::vector<float> w = make_random_f32((int64_t) nc * k, 1234 + nc);

    repacked_weights rw;
    if (!make_repacked(fn.type, w, nc, k, rw)) {
        printf("  [%s] FAILED to build repacked weights\n", fn.name);
        return false;
    }

    const size_t row_bytes = ggml_row_size(fn.type, k);
    const double tol       = 1e-3;

    // ---- gemv: native vs generic vs legacy vec_dot ----
    {
        std::vector<float> x   = make_random_f32(k, 777);
        std::vector<char>  q8  = quantize_act_row(x, k, fn.act_type);
        std::vector<float> s_nat(nc, 0.0f), s_gen(nc, 0.0f);

        fn.gemv(k, s_nat.data(), nc, rw.data, q8.data(), 1, nc);
        if (fn.gemv_generic) fn.gemv_generic(k, s_gen.data(), nc, rw.data, q8.data(), 1, nc);
        const bool has_gen_gemv = fn.gemv_generic != nullptr;

        diff_stats st_ng, st_nl;
        for (int c = 0; c < nc; c++) {
            if (has_gen_gemv) diff_update(st_ng, s_nat[c], s_gen[c], tol);
            float ref = 0.0f;
            fn.vec_dot(k, &ref, 0, rw.raw.data() + row_bytes * c, 0, q8.data(), 0, 1);
            diff_update(st_nl, s_nat[c], ref, tol);
        }
        printf("  [%s] gemv nc=%-4d k=%-5d native-vs-generic: max_abs=%.3g max_rel=%.3g bad=%d | native-vs-legacy: max_abs=%.3g max_rel=%.3g bad=%d\n",
               fn.name, nc, k, st_ng.max_abs, st_ng.max_rel, st_ng.n_bad, st_nl.max_abs, st_nl.max_rel, st_nl.n_bad);
        ok = ok && (!has_gen_gemv || st_ng.n_bad == 0) && st_nl.n_bad == 0;
    }

    // ---- gemm: native vs generic vs legacy vec_dot ----
    for (int nr : gemm_nrs) {
        std::vector<float> x  = make_random_f32((int64_t) nr * k, 999 + nr);
        std::vector<char>  q8 = quantize_acts_4x8(x, nr, k, fn.act_type);
        std::vector<float> s_nat((size_t) nr * nc, 0.0f), s_gen((size_t) nr * nc, 0.0f);

        fn.gemm(k, s_nat.data(), nc, rw.data, q8.data(), nr, nc);
        if (fn.gemm_generic) fn.gemm_generic(k, s_gen.data(), nc, rw.data, q8.data(), nr, nc);
        const bool has_gen_gemm = fn.gemm_generic != nullptr;

        diff_stats st_ng, st_nl;
        for (int r = 0; r < nr; r++) {
            // legacy reference needs the plain block_q8_K layout for row r
            std::vector<char> q8row = quantize_act_row(std::vector<float>(x.begin() + (size_t) r * k, x.begin() + (size_t) (r + 1) * k), k, fn.act_type);
            for (int c = 0; c < nc; c++) {
                const double a = s_nat[(size_t) r * nc + c];
                if (has_gen_gemm) diff_update(st_ng, a, s_gen[(size_t) r * nc + c], tol);
                float ref = 0.0f;
                fn.vec_dot(k, &ref, 0, rw.raw.data() + row_bytes * c, 0, q8row.data(), 0, 1);
                diff_update(st_nl, a, ref, tol);
            }
        }
        printf("  [%s] gemm nc=%-4d k=%-5d nr=%-3d native-vs-generic: max_abs=%.3g max_rel=%.3g bad=%d | native-vs-legacy: max_abs=%.3g max_rel=%.3g bad=%d\n",
               fn.name, nc, k, nr, st_ng.max_abs, st_ng.max_rel, st_ng.n_bad, st_nl.max_abs, st_nl.max_rel, st_nl.n_bad);
        ok = ok && (!has_gen_gemm || st_ng.n_bad == 0) && st_nl.n_bad == 0;
    }

    GGML_UNUSED(verbose);
    free_repacked(rw);
    return ok;
}

void perf_type(const kernel_fns & fn, int nc, int k, int nr, int nthreads, int n_iter) {
    std::vector<float> w = make_random_f32((int64_t) nc * k, 42);

    repacked_weights rw;
    if (!make_repacked(fn.type, w, nc, k, rw)) {
        printf("  [%s] perf: FAILED to build repacked weights\n", fn.name);
        return;
    }

    const size_t row_bytes = ggml_row_size(fn.type, k);

    std::vector<float> x = make_random_f32((int64_t) nr * k, 43);

    // column slices aligned to 8 (kernels require nc % 8 == 0)
    std::vector<std::pair<int, int>> slices;
    {
        int c = 0;
        for (int t = 0; t < nthreads && c < nc; t++) {
            int n = ((nc - c) / (nthreads - t)) & ~7;
            if (n == 0) n = std::min(8, nc - c);
            slices.push_back({c, n});
            c += n;
        }
    }

    // times total wall time per iteration (columns partitioned across threads)
    auto time_it = [&](const char * label, auto && body) {
        body(slices[0].first, slices[0].second); // warmup
        const auto t0 = std::chrono::steady_clock::now();
        if ((int) slices.size() <= 1) {
            for (int i = 0; i < n_iter; i++) body(0, nc);
        } else {
            std::vector<std::thread> ths;
            ths.reserve(slices.size());
            for (const auto & sl : slices) {
                ths.emplace_back([&, sl] { for (int i = 0; i < n_iter; i++) body(sl.first, sl.second); });
            }
            for (auto & th : ths) th.join();
        }
        const auto t1 = std::chrono::steady_clock::now();
        const double us = std::chrono::duration<double, std::micro>(t1 - t0).count() / n_iter;
        printf("  [%s] %-28s %12.1f us\n", fn.name, label, us);
        return us;
    };

    // legacy: nr rows x nc cols via per-row vec_dot (dot only, acts pre-quantized)
    std::vector<std::vector<char>> q8rows(nr);
    for (int r = 0; r < nr; r++) {
        q8rows[r] = quantize_act_row(std::vector<float>(x.begin() + (size_t) r * k, x.begin() + (size_t) (r + 1) * k), k, fn.act_type);
    }
    std::vector<float> s((size_t) nr * nc);
    const double t_legacy = time_it("legacy vec_dot (nr rows)", [&](int c0, int ncols) {
        for (int r = 0; r < nr; r++) {
            for (int c = c0; c < c0 + ncols; c++) {
                fn.vec_dot(k, &s[(size_t) r * nc + c], 0, rw.raw.data() + row_bytes * c, 0, q8rows[r].data(), 0, 1);
            }
        }
    });

    if (nr == 1) {
        const double t = time_it("repack gemv", [&](int c0, int ncols) {
            fn.gemv(k, s.data() + c0, nc, (const char *) rw.data + row_bytes * c0, q8rows[0].data(), 1, ncols);
        });
        printf("  [%s] speedup gemv vs legacy:      %.2fx\n", fn.name, t_legacy / t);
    } else {
        std::vector<char> q8 = quantize_acts_4x8(x, nr, k, fn.act_type);
        const double t = time_it(("repack gemm nr=" + std::to_string(nr)).c_str(), [&](int c0, int ncols) {
            fn.gemm(k, s.data() + c0, nc, (const char *) rw.data + row_bytes * c0, q8.data(), nr, ncols);
        });
        printf("  [%s] speedup gemm nr=%-3d vs legacy: %.2fx\n", fn.name, nr, t_legacy / t);
    }

    free_repacked(rw);
}

// ---- mxfp4 unrepack (inverse transform) bit-exact tests and bandwidth bench ----

// faithful replicas of make_block_mxfp4x4/x8 (repack.cpp) used to build x4/x8
// interleaved inputs without going through the ISA-dependent buffer path;
// `in` points at column block x of the row group, rows are strided by nblocks
void fwd_mxfp4x4(const block_mxfp4 * in, block_mxfp4x4 * out, int64_t nblocks) {
    for (int i = 0; i < 4; i++) out->e[i] = in[i*nblocks].e;
    for (int i = 0; i < QK_MXFP4*2/4; i++) {
        memcpy(&out->qs[i*4], &in[(i%4)*nblocks].qs[(i/4)*4], sizeof(uint32_t));
    }
}

void fwd_mxfp4x8(const block_mxfp4 * in, block_mxfp4x8 * out, int64_t nblocks) {
    for (int i = 0; i < 8; i++) out->e[i] = in[i*nblocks].e;
    for (int i = 0; i < QK_MXFP4*4/8; i++) {
        memcpy(&out->qs[i*8], &in[(i%8)*nblocks].qs[(i/8)*8], sizeof(uint64_t));
    }
}

std::vector<char> make_random_bytes(int64_t n, uint32_t seed) {
    std::mt19937 rng(seed);
    std::vector<char> v(n);
    for (auto & x : v) x = (char) (rng() & 0xff);
    return v;
}

// repack (via the replica) then unrepack, compare against the original bytes
bool test_unrepack_layout(int interleave, int64_t nblocks, int64_t nrows, bool verbose) {
    const int64_t row_bytes = nblocks*(int64_t) sizeof(block_mxfp4);
    std::vector<char> orig = make_random_bytes(nrows*row_bytes, 20260806 + (uint32_t) interleave + (uint32_t) nrows);

    std::vector<char> packed(orig.size());
    if (interleave == 8) {
        for (int64_t g = 0; g < nrows/8; g++) {
            for (int64_t x = 0; x < nblocks; x++) {
                fwd_mxfp4x8((const block_mxfp4 *) (orig.data() + g*8*row_bytes) + x,
                            (block_mxfp4x8 *) (packed.data() + g*8*row_bytes) + x, nblocks);
            }
        }
    } else {
        for (int64_t g = 0; g < nrows/4; g++) {
            for (int64_t x = 0; x < nblocks; x++) {
                fwd_mxfp4x4((const block_mxfp4 *) (orig.data() + g*4*row_bytes) + x,
                            (block_mxfp4x4 *) (packed.data() + g*4*row_bytes) + x, nblocks);
            }
        }
    }

    std::vector<char> back(orig.size());
    ggml_repack_mxfp4_unrepack_rows(packed.data(), back.data(), nblocks, nrows, interleave);

    const bool ok = memcmp(back.data(), orig.data(), orig.size()) == 0;
    printf("  [MXFP4 x%d] unrepack nblocks=%-4d nrows=%-5d bit-exact: %s\n",
           interleave, (int) nblocks, (int) nrows, ok ? "OK" : "MISMATCH");
    GGML_UNUSED(verbose);
    return ok;
}

// multithreaded job over whole expert planes (+ EP-style rank half), bit-exact;
// ep_win = 0 uses flat claims, a 128-aligned value exercises the windowed
// local/steal claim path
bool test_unrepack_job(int interleave, int64_t nblocks, int64_t rpe, int64_t n_exp, int nth, int64_t ep_win) {
    const int64_t row_bytes = nblocks*(int64_t) sizeof(block_mxfp4);
    const int64_t expert_bytes = rpe*row_bytes;
    std::vector<char> orig = make_random_bytes(n_exp*expert_bytes, 777 + (uint32_t) interleave);

    std::vector<char> packed(orig.size());
    for (int64_t e = 0; e < n_exp; e++) {
        for (int64_t g = 0; g < rpe/interleave; g++) {
            for (int64_t x = 0; x < nblocks; x++) {
                if (interleave == 8) {
                    fwd_mxfp4x8((const block_mxfp4 *) (orig.data() + e*expert_bytes + g*8*row_bytes) + x,
                                (block_mxfp4x8 *) (packed.data() + e*expert_bytes + g*8*row_bytes) + x, nblocks);
                } else {
                    fwd_mxfp4x4((const block_mxfp4 *) (orig.data() + e*expert_bytes + g*4*row_bytes) + x,
                                (block_mxfp4x4 *) (packed.data() + e*expert_bytes + g*4*row_bytes) + x, nblocks);
                }
            }
        }
    }

    bool ok = true;
    // full tensor
    {
        std::vector<char> back(orig.size(), 0);
        ggml_repack_unrepack_job * job = ggml_repack_unrepack_job_new(
                packed.data(), back.data(), nblocks, rpe, n_exp, interleave, ep_win);
        std::vector<std::thread> ths;
        for (int ith = 0; ith < nth; ith++) ths.emplace_back([&, ith] { ggml_repack_unrepack_job_run(job, ith, nth); });
        for (auto & th : ths) th.join();
        ggml_repack_unrepack_job_free(job);
        const bool pass = memcmp(back.data(), orig.data(), orig.size()) == 0;
        printf("  [MXFP4 x%d] unrepack job rpe=%-5d E=%-3d threads=%-2d win=%-5d full: %s\n",
               interleave, (int) rpe, (int) n_exp, nth, (int) ep_win, pass ? "OK" : "MISMATCH");
        ok = ok && pass;
    }
    // EP-rank style: second half of the expert stack from a repacked row offset
    {
        const int64_t half = n_exp/2;
        std::vector<char> back((size_t) half*expert_bytes, 0);
        ggml_repack_unrepack_job * job = ggml_repack_unrepack_job_new(
                packed.data() + half*expert_bytes, back.data(), nblocks, rpe, half, interleave, ep_win);
        std::vector<std::thread> ths;
        for (int ith = 0; ith < nth; ith++) ths.emplace_back([&, ith] { ggml_repack_unrepack_job_run(job, ith, nth); });
        for (auto & th : ths) th.join();
        ggml_repack_unrepack_job_free(job);
        const bool pass = memcmp(back.data(), orig.data() + half*expert_bytes, back.size()) == 0;
        printf("  [MXFP4 x%d] unrepack job rpe=%-5d E=%-3d threads=%-2d win=%-5d rank-half: %s\n",
               interleave, (int) rpe, (int) half, nth, (int) ep_win, pass ? "OK" : "MISMATCH");
        ok = ok && pass;
    }
    return ok;
}

// the real CPU_REPACK buffer path (set_tensor performs the interleave chosen
// for this ISA) must be undone bit-exactly by ggml_repack_mxfp4_unrepack_rows
bool test_unrepack_buffer_path(int nc, int k) {
    std::vector<float> w = make_random_f32((int64_t) nc*k, 555);

    repacked_weights rw;
    if (!make_repacked(GGML_TYPE_MXFP4, w, nc, k, rw)) {
        printf("  [MXFP4] unrepack buffer path: SKIPPED (no CPU_REPACK kernel)\n");
        return true;
    }

    const int interleave = ggml_repack_mxfp4_interleave(rw.tensor);
    const int64_t nblocks = k/QK_MXFP4;
    bool ok = true;
    if (interleave == 0) {
        printf("  [MXFP4] unrepack buffer path: SKIPPED (tensor not repacked)\n");
    } else {
        std::vector<char> back(rw.nbytes);
        ggml_repack_mxfp4_unrepack_rows(rw.data, back.data(), nblocks, nc, interleave);
        ok = memcmp(back.data(), rw.raw.data(), rw.nbytes) == 0;
        printf("  [MXFP4] unrepack buffer path nc=%-4d k=%-5d interleave=x%d bit-exact: %s\n",
               nc, k, interleave, ok ? "OK" : "MISMATCH");
    }

    // the replica must produce the exact bytes the buffer path produced
    if (ok && interleave == 8 && nc % 8 == 0) {
        const int64_t row_bytes = ggml_row_size(GGML_TYPE_MXFP4, k);
        std::vector<char> replica(rw.nbytes);
        for (int64_t g = 0; g < nc/8; g++) {
            for (int64_t x = 0; x < nblocks; x++) {
                fwd_mxfp4x8((const block_mxfp4 *) (rw.raw.data() + g*8*row_bytes) + x,
                            (block_mxfp4x8 *) (replica.data() + g*8*row_bytes) + x, nblocks);
            }
        }
        const bool pass = memcmp(replica.data(), rw.data, rw.nbytes) == 0;
        printf("  [MXFP4] replica-vs-buffer interleave layout: %s\n", pass ? "OK" : "MISMATCH");
        ok = ok && pass;
    }

    free_repacked(rw);
    return ok;
}

#if defined(__linux__)
// CPUs of each NUMA node parsed from /sys, for benchmark thread pinning
std::vector<std::vector<int>> numa_node_cpus() {
    std::vector<std::vector<int>> out;
    for (int node = 0; ; node++) {
        char path[128];
        snprintf(path, sizeof(path), "/sys/devices/system/node/node%d/cpulist", node);
        FILE * f = fopen(path, "r");
        if (!f) break;
        char buf[1024] = { 0 };
        if (!fgets(buf, sizeof(buf), f)) { fclose(f); break; }
        fclose(f);
        std::vector<int> cpus;
        for (char * tok = strtok(buf, ",\n"); tok; tok = strtok(nullptr, ",\n")) {
            int a = 0, b = 0;
            if (sscanf(tok, "%d-%d", &a, &b) == 2) {
                for (int c = a; c <= b; c++) cpus.push_back(c);
            } else if (sscanf(tok, "%d", &a) == 1) {
                cpus.push_back(a);
            }
        }
        out.push_back(std::move(cpus));
    }
    return out;
}
#endif

// bandwidth benchmark of the inverse transform at one MoE layer's expert size
// (default ~3.43 GiB): flat claims vs NUMA-local window claims
void bench_unrepack(int nthreads, double gib) {
    ggml_numa_init(GGML_NUMA_STRATEGY_DISTRIBUTE);
    const int n_nodes = ggml_numa_node_count();

    const int64_t nblocks = 128;   // k = 4096
    const int64_t rpe = 2048;      // rows per expert
    const int64_t row_bytes = nblocks*(int64_t) sizeof(block_mxfp4);
    const int64_t expert_bytes = rpe*row_bytes;
    const int64_t n_exp = std::max<int64_t>(1, (int64_t) (gib*1073741824.0/expert_bytes));
    const int64_t total = n_exp*expert_bytes;

    printf("== unrepack bench: %.3f GiB (E=%lld rpe=%lld k=%lld) threads=%d nodes=%d ==\n",
           total/1073741824.0, (long long) n_exp, (long long) rpe, (long long) (nblocks*QK_MXFP4), nthreads, n_nodes);

    char * src = (char *) aligned_alloc(64, total);
    char * dst = (char *) aligned_alloc(64, total);
    if (!src || !dst) {
        printf("  allocation failed\n");
        free(src); free(dst);
        return;
    }
    {
        // parallel first touch spreads pages across nodes before any binding
        std::vector<std::thread> ths;
        for (int t = 0; t < nthreads; t++) {
            ths.emplace_back([&, t] {
                std::mt19937 rng(1234 + t);
                const int64_t chunk = total/nthreads;
                char * p = src + t*chunk;
                for (int64_t i = 0; i < chunk; i += 4096) p[i] = (char) (rng() & 0xff);
            });
        }
        for (auto & th : ths) th.join();
    }

    const int64_t ep_win = n_nodes > 1 ? ((rpe + n_nodes - 1)/n_nodes + 127)/128*128 : 0;

    // EP-style placement: per-expert row windows bound to their node, for both
    // src (as the CPU_REPACK buffer does) and dst (as pinned staging first-touched
    // by the NUMA-local unrepack workers would be)
    if (ep_win > 0) {
        for (int64_t e = 0; e < n_exp; e++) {
            for (int n = 0; n < n_nodes; n++) {
                const int64_t r0 = n*ep_win;
                const int64_t r1 = std::min(r0 + ep_win, rpe);
                if (r1 <= r0) continue;
                ggml_numa_bind(src + e*expert_bytes + r0*row_bytes, (size_t) (r1 - r0)*row_bytes, n);
                ggml_numa_bind(dst + e*expert_bytes + r0*row_bytes, (size_t) (r1 - r0)*row_bytes, n);
            }
        }
    }

#if defined(__linux__)
    const auto node_cpus = numa_node_cpus();
    auto pin = [&](int ith, int nth) {
        if ((int) node_cpus.size() < n_nodes) return;
        const int node = ggml_numa_node_for_thread(ith, nth);
        const int t_first = (int) (((int64_t) node*nth + n_nodes - 1)/n_nodes);
        const int t_next  = (int) (((int64_t) (node + 1)*nth + n_nodes - 1)/n_nodes);
        const auto & cpus = node_cpus[node];
        if (cpus.empty() || t_next - t_first <= 0) return;
        cpu_set_t set;
        CPU_ZERO(&set);
        CPU_SET(cpus[(size_t) (ith - t_first)%cpus.size()], &set);
        sched_setaffinity(0, sizeof(set), &set);
    };
#else
    auto pin = [](int, int) {};
#endif

    auto run = [&](const char * label, int64_t job_ep_win, int nth) {
        double best = 1e30;
        for (int iter = 0; iter < 3; iter++) {
            ggml_repack_unrepack_job * job = ggml_repack_unrepack_job_new(
                    src, dst, nblocks, rpe, n_exp, 8, job_ep_win);
            const auto t0 = std::chrono::steady_clock::now();
            std::vector<std::thread> ths;
            for (int ith = 0; ith < nth; ith++) {
                ths.emplace_back([&, ith] {
                    pin(ith, nth);
                    ggml_repack_unrepack_job_run(job, ith, nth);
                });
            }
            for (auto & th : ths) th.join();
            const auto t1 = std::chrono::steady_clock::now();
            ggml_repack_unrepack_job_free(job);
            best = std::min(best, std::chrono::duration<double, std::milli>(t1 - t0).count());
        }
        const double gbs = 2.0*total/1e9/(best/1e3); // read + write
        printf("  %-34s %10.2f ms  %8.1f GB/s (r+w)\n", label, best, gbs);
        return best;
    };

    run("single thread", 0, 1);
    run("flat claims", 0, nthreads);
    if (ep_win > 0) {
        char label[64];
        snprintf(label, sizeof(label), "numa-local claims (win=%lld rows)", (long long) ep_win);
        run(label, ep_win, nthreads);
    } else {
        printf("  numa-local claims: SKIPPED (single NUMA node)\n");
    }

    free(src);
    free(dst);
}

} // namespace

int main(int argc, char ** argv) {
    const bool perf = argc > 1 && std::string(argv[1]) == "--perf";

    if (argc > 1 && std::string(argv[1]) == "--unrepack-bench") {
        const int nthreads = argc > 2 ? atoi(argv[2]) : 16;
        const double gib = argc > 3 ? atof(argv[3]) : 3.43;
        bench_unrepack(nthreads, gib);
        return 0;
    }

    const std::vector<kernel_fns> types = {
        { GGML_TYPE_Q2_K, "Q2_K", ggml_vec_dot_q2_K_q8_K, ggml_gemv_q2_K_8x8_q8_K, ggml_gemv_q2_K_8x8_q8_K_generic,
          ggml_gemm_q2_K_8x8_q8_K, ggml_gemm_q2_K_8x8_q8_K_generic },
        { GGML_TYPE_Q3_K, "Q3_K", ggml_vec_dot_q3_K_q8_K, ggml_gemv_q3_K_8x8_q8_K, ggml_gemv_q3_K_8x8_q8_K_generic,
          ggml_gemm_q3_K_8x8_q8_K, ggml_gemm_q3_K_8x8_q8_K_generic },
        { GGML_TYPE_Q5_K, "Q5_K", ggml_vec_dot_q5_K_q8_K, ggml_gemv_q5_K_8x8_q8_K, ggml_gemv_q5_K_8x8_q8_K_generic,
          ggml_gemm_q5_K_8x8_q8_K, ggml_gemm_q5_K_8x8_q8_K_generic },
        { GGML_TYPE_IQ1_S, "IQ1_S", ggml_vec_dot_iq1_s_q8_K, ggml_gemv_iq1_s_8x8_q8_K, ggml_gemv_iq1_s_8x8_q8_K_generic,
          ggml_gemm_iq1_s_8x8_q8_K, ggml_gemm_iq1_s_8x8_q8_K_generic },
        { GGML_TYPE_IQ1_M, "IQ1_M", ggml_vec_dot_iq1_m_q8_K, ggml_gemv_iq1_m_8x8_q8_K, ggml_gemv_iq1_m_8x8_q8_K_generic,
          ggml_gemm_iq1_m_8x8_q8_K, ggml_gemm_iq1_m_8x8_q8_K_generic },
        { GGML_TYPE_IQ2_XS, "IQ2_XS", ggml_vec_dot_iq2_xs_q8_K, ggml_gemv_iq2_xs_8x8_q8_K, ggml_gemv_iq2_xs_8x8_q8_K_generic,
          ggml_gemm_iq2_xs_8x8_q8_K, ggml_gemm_iq2_xs_8x8_q8_K_generic },
        { GGML_TYPE_IQ3_XXS, "IQ3_XXS", ggml_vec_dot_iq3_xxs_q8_K, ggml_gemv_iq3_xxs_8x8_q8_K, ggml_gemv_iq3_xxs_8x8_q8_K_generic,
          ggml_gemm_iq3_xxs_8x8_q8_K, ggml_gemm_iq3_xxs_8x8_q8_K_generic },
        { GGML_TYPE_Q4_0, "Q4_0", ggml_vec_dot_q4_0_q8_0, ggml_gemv_q4_0_8x8_q8_0, ggml_gemv_q4_0_8x8_q8_0_generic,
          ggml_gemm_q4_0_8x8_q8_0, ggml_gemm_q4_0_8x8_q8_0_generic, GGML_TYPE_Q8_0 },
        { GGML_TYPE_Q4_K, "Q4_K", ggml_vec_dot_q4_K_q8_K, ggml_gemv_q4_K_8x8_q8_K, ggml_gemv_q4_K_8x8_q8_K_generic,
          ggml_gemm_q4_K_8x8_q8_K, ggml_gemm_q4_K_8x8_q8_K_generic },
        { GGML_TYPE_Q6_K, "Q6_K", ggml_vec_dot_q6_K_q8_K, ggml_gemv_q6_K_8x8_q8_K, ggml_gemv_q6_K_8x8_q8_K_generic,
          ggml_gemm_q6_K_8x8_q8_K, ggml_gemm_q6_K_8x8_q8_K_generic },
        { GGML_TYPE_IQ4_NL, "IQ4_NL", ggml_vec_dot_iq4_nl_q8_0, ggml_gemv_iq4_nl_8x8_q8_0, ggml_gemv_iq4_nl_8x8_q8_0_generic,
          ggml_gemm_iq4_nl_8x8_q8_0, ggml_gemm_iq4_nl_8x8_q8_0_generic, GGML_TYPE_Q8_0 },
        { GGML_TYPE_MXFP4, "MXFP4", ggml_vec_dot_mxfp4_q8_0, ggml_gemv_mxfp4_8x8_q8_0, ggml_gemv_mxfp4_8x8_q8_0_generic,
          ggml_gemm_mxfp4_8x8_q8_0, ggml_gemm_mxfp4_8x8_q8_0_generic, GGML_TYPE_Q8_0 },
        // Q8_0 has no generic 8x8 kernels; native-vs-generic checks are skipped for it
        { GGML_TYPE_Q8_0, "Q8_0", ggml_vec_dot_q8_0_q8_0, ggml_gemv_q8_0_8x8_q8_0, nullptr,
          ggml_gemm_q8_0_8x8_q8_0, nullptr, GGML_TYPE_Q8_0 },
    };

    if (perf) {
        const int nthreads = argc > 2 ? atoi(argv[2]) : 1;
        const std::string only = argc > 3 ? argv[3] : "";
        const struct { int nc, k; } shapes[] = { { 2048, 4096 }, { 16384, 8192 } };
        const int nrs[] = { 1, 4, 8, 16, 32 };
        for (const auto & sh : shapes) {
            printf("== shape nc=%d k=%d threads=%d ==\n", sh.nc, sh.k, nthreads);
            for (const auto & fn : types) {
                if (!only.empty() && only != fn.name) continue;
                if (!repack_supported(fn.type, sh.nc, sh.k)) {
                    printf("  [%s] SKIPPED: no CPU_REPACK kernel for this build\n", fn.name);
                    continue;
                }
                for (const int nr : nrs) {
                    const int64_t dots = (int64_t) sh.nc * nr;
                    const int n_iter = dots > 200000 ? 2 : (dots > 50000 ? 4 : 10);
                    perf_type(fn, sh.nc, sh.k, nr, nthreads, n_iter);
                }
            }
        }
        return 0;
    }

    if (!ggml_cpu_has_avx512_vnni() || !ggml_cpu_has_avx512_vbmi()) {
        printf("WARNING: AVX512/VNNI/VBMI not available - native kernels fall back to generic,\n");
        printf("         native-vs-generic comparisons are vacuous on this machine\n");
    }

    bool ok = true;
    const std::vector<int> gemm_nrs = { 4, 8, 16, 20, 32 };

    for (const auto & fn : types) {
        if (!repack_supported(fn.type, 512, 2048)) {
            printf("[%s] SKIPPED: no CPU_REPACK kernel for this build\n", fn.name);
            continue;
        }
        printf("[%s] nc=512 k=2048 (main paths)\n", fn.name);
        ok &= test_type(fn, 512, 2048, gemm_nrs, false);
        printf("[%s] nc=8 k=2048 (256-bit tail column group only)\n", fn.name);
        ok &= test_type(fn, 8, 2048, { 4, 8, 16 }, false);
        printf("[%s] nc=520 k=2048 (odd trailing 8-column group)\n", fn.name);
        ok &= test_type(fn, 520, 2048, { 4, 20 }, false);
        if (fn.type == GGML_TYPE_MXFP4) {
            printf("[%s] nc=64 k=1024 (small-batch model path)\n", fn.name);
            ok &= test_type(fn, 64, 1024, { 8 }, false);
        }
    }

    printf("[MXFP4] unrepack (inverse transform) bit-exact roundtrips\n");
    ok &= test_unrepack_layout(8, 128, 512, false);   // k=4096 gate/up row
    ok &= test_unrepack_layout(8,  64, 520, false);   // k=2048, non-128-multiple rows
    ok &= test_unrepack_layout(8, 128,   8, false);   // single row group
    ok &= test_unrepack_layout(4, 128, 512, false);
    ok &= test_unrepack_layout(4,  64, 520, false);
    ok &= test_unrepack_job(8, 128, 2048, 16, 8, 0);     // model-like expert planes
    ok &= test_unrepack_job(8,  64, 2056,  8, 4, 0);     // rpe not a multiple of 128
    ok &= test_unrepack_job(4, 128, 1024,  8, 8, 0);
    ok &= test_unrepack_job(8, 128, 2048, 16, 8, 1024);  // windowed local/steal claims
    ok &= test_unrepack_job(8,  64, 2056,  8, 4, 1152);  // windowed, tail block per expert
    ok &= test_unrepack_job(4, 128, 1024,  8, 8, 512);
    ok &= test_unrepack_buffer_path(512, 2048);
    ok &= test_unrepack_buffer_path(520, 2048);

    printf("%s\n", ok ? "ALL TESTS PASSED" : "TESTS FAILED");
    ggml_quantize_free();
    return ok ? 0 : 1;
}
