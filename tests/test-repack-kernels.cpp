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
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "repack.h"
#include "quants.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <atomic>
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
    if (type == GGML_TYPE_UDNL_MX) {
        // UDNL_MX rows of a 16-row panel share the mode word: quantize the
        // whole plane at once (panel DP) instead of row-wise from_float_ref.
        ggml_quantize_chunk(type, w.data(), out.raw.data(), 0, nc, k, nullptr);
    } else if (traits->from_float_ref) {
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

std::vector<float> make_random_f32(int64_t n, uint32_t seed);

bool make_expert_weights(
        ggml_type type,
        int nc,
        int k,
        int n_experts,
        bool use_repack,
        repacked_weights & out) {
    const ggml_type_traits * traits = ggml_get_type_traits(type);
    ggml_quantize_init(type);

    const std::vector<float> plane = make_random_f32((int64_t) nc * k, 31415);
    const size_t row_bytes = ggml_row_size(type, k);
    const size_t plane_bytes = row_bytes * nc;
    std::vector<char> raw_plane(plane_bytes);
    if (type == GGML_TYPE_UDNL_MX) {
        // UDNL_MX rows of a 16-row panel share the mode word: quantize the
        // whole plane at once (panel DP) instead of row-wise from_float_ref.
        ggml_quantize_chunk(type, plane.data(), raw_plane.data(), 0, nc, k, nullptr);
    } else if (traits->from_float_ref) {
        for (int r = 0; r < nc; ++r) {
            traits->from_float_ref(plane.data() + (size_t) r * k,
                                   raw_plane.data() + row_bytes * r, k);
        }
    } else {
        const std::vector<float> imatrix((size_t) k, 1.0f);
        ggml_quantize_chunk(type, plane.data(), raw_plane.data(), 0, nc, k, imatrix.data());
    }

    out.raw.resize(plane_bytes * n_experts);
    for (int expert = 0; expert < n_experts; ++expert) {
        memcpy(out.raw.data() + (size_t) expert * plane_bytes, raw_plane.data(), plane_bytes);
    }

    struct ggml_init_params params = {1 * 1024 * 1024, nullptr, true};
    out.ctx = ggml_init(params);
    if (out.ctx == nullptr) {
        return false;
    }
    out.tensor = ggml_new_tensor_3d(out.ctx, type, k, nc, n_experts);
    out.nbytes = ggml_nbytes(out.tensor);
    out.buffer = ggml_backend_buft_alloc_buffer(
        use_repack ? ggml_backend_cpu_repack_buffer_type() : ggml_backend_cpu_buffer_type(), out.nbytes);
    if (out.buffer == nullptr) {
        return false;
    }
    ggml_backend_tensor_alloc(out.buffer, out.tensor, ggml_backend_buffer_get_base(out.buffer));
    if (use_repack && out.tensor->extra == nullptr) {
        free_repacked(out);
        return false;
    }
    ggml_backend_tensor_set(out.tensor, out.raw.data(), 0, out.raw.size());
    out.data = out.tensor->data;
    return true;
}

// Build two expert tensors in one CPU_REPACK buffer, matching the EPD loader.
// Besides being more realistic, this avoids measuring two independent buffer
// mappings as if they were the gate/up pair of one worker.
bool make_expert_weight_pair(
        ggml_type type,
        int nc,
        int k,
        int n_experts,
        bool use_repack,
        repacked_weights & out,
        ggml_tensor *& second) {
    const ggml_type_traits * traits = ggml_get_type_traits(type);
    ggml_quantize_init(type);

    const std::vector<float> plane = make_random_f32((int64_t) nc * k, 31415);
    const size_t row_bytes = ggml_row_size(type, k);
    const size_t plane_bytes = row_bytes * nc;
    std::vector<char> raw_plane(plane_bytes);
    if (type == GGML_TYPE_UDNL_MX) {
        // UDNL_MX rows of a 16-row panel share the mode word: quantize the
        // whole plane at once (panel DP) instead of row-wise from_float_ref.
        ggml_quantize_chunk(type, plane.data(), raw_plane.data(), 0, nc, k, nullptr);
    } else if (traits->from_float_ref) {
        for (int r = 0; r < nc; ++r) {
            traits->from_float_ref(plane.data() + (size_t) r * k,
                                   raw_plane.data() + row_bytes * r, k);
        }
    } else {
        const std::vector<float> imatrix((size_t) k, 1.0f);
        ggml_quantize_chunk(type, plane.data(), raw_plane.data(), 0, nc, k, imatrix.data());
    }

    out.raw.resize(plane_bytes * n_experts);
    for (int expert = 0; expert < n_experts; ++expert) {
        memcpy(out.raw.data() + (size_t) expert * plane_bytes, raw_plane.data(), plane_bytes);
    }

    const ggml_init_params params = {1 * 1024 * 1024, nullptr, true};
    out.ctx = ggml_init(params);
    if (out.ctx == nullptr) return false;
    out.tensor = ggml_new_tensor_3d(out.ctx, type, k, nc, n_experts);
    second = ggml_new_tensor_3d(out.ctx, type, k, nc, n_experts);
    out.nbytes = ggml_nbytes(out.tensor);

    ggml_backend_buffer_type_t buft = use_repack ?
        ggml_backend_cpu_repack_buffer_type() : ggml_backend_cpu_buffer_type();
    const size_t first_alloc = ggml_backend_buft_get_alloc_size(buft, out.tensor);
    const size_t second_alloc = ggml_backend_buft_get_alloc_size(buft, second);
    out.buffer = ggml_backend_buft_alloc_buffer(buft, first_alloc + second_alloc);
    if (out.buffer == nullptr) {
        free_repacked(out);
        second = nullptr;
        return false;
    }

    char * base = (char *) ggml_backend_buffer_get_base(out.buffer);
    ggml_backend_tensor_alloc(out.buffer, out.tensor, base);
    ggml_backend_tensor_alloc(out.buffer, second, base + first_alloc);
    if (use_repack && (out.tensor->extra == nullptr || second->extra == nullptr)) {
        free_repacked(out);
        second = nullptr;
        return false;
    }
    ggml_backend_tensor_set(out.tensor, out.raw.data(), 0, out.raw.size());
    ggml_backend_tensor_set(second, out.raw.data(), 0, out.raw.size());
    out.data = out.tensor->data;
    return true;
}

std::vector<float> make_random_f32(int64_t n, uint32_t seed) {
    std::mt19937                    rng(seed);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    std::vector<float>              v(n);
    for (auto & x : v) x = dist(rng);
    return v;
}

bool test_udnl_mx_imatrix_contract() {
    constexpr int64_t nrows = 16;
    constexpr int64_t k = 512;
    const std::vector<float> source = make_random_f32(nrows * k, 8642);
    std::vector<float> imatrix(k);
    for (int64_t i = 0; i < k; ++i) {
        imatrix[i] = 0.25f + (float) ((i * 17) % 101) / 37.0f;
    }
    std::vector<float> extended(nrows * k);
    std::copy(imatrix.begin(), imatrix.end(), extended.begin());
    for (int64_t i = k; i < nrows * k; ++i) {
        extended[i] = 1000.0f + (float) ((i * 29) % 251);
    }

    const size_t output_size = ggml_row_size(GGML_TYPE_UDNL_MX, k) * nrows;
    std::vector<char> short_weights(output_size);
    std::vector<char> extended_weights(output_size);
    const size_t short_size = ggml_quantize_chunk(
        GGML_TYPE_UDNL_MX, source.data(), short_weights.data(), 0, nrows, k, imatrix.data());
    const size_t extended_size = ggml_quantize_chunk(
        GGML_TYPE_UDNL_MX, source.data(), extended_weights.data(), 0, nrows, k, extended.data());
    const bool exact = short_size == output_size && extended_size == output_size &&
        memcmp(short_weights.data(), extended_weights.data(), output_size) == 0;
    printf("[UDNL_MX] single-row imatrix contract and tail independence: %s\n", exact ? "bit-exact" : "FAILED");
    return exact;
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

// Quantize nr activation rows into plain row-major blocks. Q3_R is an
// INTER_SIZE == 1 identity layout, so its gemm takes this layout rather than
// the 4x8-interleaved block_q8_0x4 one.
std::vector<char> quantize_act_rows(const std::vector<float> & x, int nr, int k, ggml_type act = GGML_TYPE_Q8_K) {
    const size_t qrow = ggml_row_size(act, k);
    std::vector<char> vy(qrow * nr);
    for (int r = 0; r < nr; r++) {
        if (act == GGML_TYPE_Q8_K) {
            quantize_row_q8_K(x.data() + (size_t) r * k, vy.data() + qrow * r, k);
        } else {
            quantize_row_q8_0(x.data() + (size_t) r * k, vy.data() + qrow * r, k);
        }
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
    // Most repack kernels use the same activation format as the legacy
    // vec_dot path. IQ4_XS is different: repack uses Q8_0 while legacy uses
    // Q8_K, so only native-vs-generic is an apples-to-apples comparison.
    bool         compare_legacy = true;
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
            if (fn.compare_legacy) {
                float ref = 0.0f;
                fn.vec_dot(k, &ref, 0, rw.raw.data() + row_bytes * c, 0, q8.data(), 0, 1);
                diff_update(st_nl, s_nat[c], ref, tol);
            }
        }
        if (fn.compare_legacy && has_gen_gemv) {
            printf("  [%s] gemv nc=%-4d k=%-5d native-vs-generic: max_abs=%.3g max_rel=%.3g bad=%d | native-vs-legacy: max_abs=%.3g max_rel=%.3g bad=%d\n",
                   fn.name, nc, k, st_ng.max_abs, st_ng.max_rel, st_ng.n_bad, st_nl.max_abs, st_nl.max_rel, st_nl.n_bad);
        } else if (fn.compare_legacy) {
            printf("  [%s] gemv nc=%-4d k=%-5d native-vs-generic: SKIPPED (no separate generic kernel) | native-vs-legacy: max_abs=%.3g max_rel=%.3g bad=%d\n",
                   fn.name, nc, k, st_nl.max_abs, st_nl.max_rel, st_nl.n_bad);
        } else if (has_gen_gemv) {
            printf("  [%s] gemv nc=%-4d k=%-5d native-vs-generic: max_abs=%.3g max_rel=%.3g bad=%d | native-vs-legacy: SKIPPED (different activation format)\n",
                   fn.name, nc, k, st_ng.max_abs, st_ng.max_rel, st_ng.n_bad);
        } else {
            printf("  [%s] gemv nc=%-4d k=%-5d native-vs-generic: SKIPPED (no native override) | native-vs-legacy: SKIPPED (different activation format)\n",
                   fn.name, nc, k);
        }
        ok = ok && (!has_gen_gemv || st_ng.n_bad == 0) && (!fn.compare_legacy || st_nl.n_bad == 0);
    }

    if (fn.type == GGML_TYPE_Q3_R) {
        // The default gemv path is the integer-scale variant (maddubs+madd,
        // multi-accumulator), which reassociates the fp finalize and is only
        // rounding-level accurate vs legacy vec_dot — it is covered by the
        // 1e-3 tolerance above. The GGML_REPACK_Q3_R_GEMV_INT=0 fallback must
        // remain bit-exact; verify the switch and the fallback here.
        std::vector<float> x  = make_random_f32(k, 555);
        std::vector<char>  q8 = quantize_act_row(x, k, fn.act_type);
        std::vector<float> s_fb(nc, 0.0f);
        setenv("GGML_REPACK_Q3_R_GEMV_INT", "0", 1);
        fn.gemv(k, s_fb.data(), nc, rw.data, q8.data(), 1, nc);
        unsetenv("GGML_REPACK_Q3_R_GEMV_INT");
        int n_diff = 0;
        for (int c = 0; c < nc; c++) {
            float ref = 0.0f;
            fn.vec_dot(k, &ref, 0, rw.raw.data() + row_bytes * c, 0, q8.data(), 0, 1);
            if (memcmp(&s_fb[c], &ref, sizeof(float)) != 0) {
                n_diff++;
            }
        }
        printf("  [%s] gemv int-variant fallback (GGML_REPACK_Q3_R_GEMV_INT=0) vs legacy: %s\n",
               fn.name, n_diff == 0 ? "bit-exact" : "FAILED");
        ok = ok && n_diff == 0;
    }

    if (fn.type == GGML_TYPE_MXFP4) {
        for (int nr = 2; nr <= 8; ++nr) {
            const std::vector<float> x = make_random_f32((int64_t) nr * k, 780 + nr);
            const size_t qrow_bytes = ggml_row_size(GGML_TYPE_Q8_0, k);
            std::vector<char> q8(qrow_bytes * nr);
            for (int r = 0; r < nr; ++r) {
                quantize_row_q8_0(x.data() + (size_t) r * k, q8.data() + (size_t) r * qrow_bytes, k);
            }
            std::vector<float> batched((size_t) nr * nc, 0.0f);
            std::vector<float> separate((size_t) nr * nc, 0.0f);
            fn.gemv(k, batched.data(), nc, rw.data, q8.data(), nr, nc);
            for (int r = 0; r < nr; ++r) {
                fn.gemv(k, separate.data() + (size_t) r * nc, nc, rw.data,
                        q8.data() + (size_t) r * qrow_bytes, 1, nc);
            }
            const bool exact = memcmp(batched.data(), separate.data(), batched.size() * sizeof(float)) == 0;
            printf("  [MXFP4] gemv small rows nc=%-4d k=%-5d nr=%d batched-vs-separate: %s\n",
                   nc, k, nr, exact ? "bit-exact" : "FAILED");
            ok = ok && exact;
        }
    }

    // ---- gemm: native vs generic vs legacy vec_dot ----
    for (int nr : gemm_nrs) {
        std::vector<float> x  = make_random_f32((int64_t) nr * k, 999 + nr);
        std::vector<char>  q8 = fn.type == GGML_TYPE_Q3_R || fn.type == GGML_TYPE_UDNL_W4 || fn.type == GGML_TYPE_UDNL_MX || fn.type == GGML_TYPE_E4A
                              ? quantize_act_rows(x, nr, k, fn.act_type)
                              : quantize_acts_4x8(x, nr, k, fn.act_type);
        std::vector<float> s_nat((size_t) nr * nc, 0.0f), s_gen((size_t) nr * nc, 0.0f);

        fn.gemm(k, s_nat.data(), nc, rw.data, q8.data(), nr, nc);
        if (fn.gemm_generic) fn.gemm_generic(k, s_gen.data(), nc, rw.data, q8.data(), nr, nc);
        const bool has_gen_gemm = fn.gemm_generic != nullptr;

        diff_stats st_ng, st_nl;
        for (int r = 0; r < nr; r++) {
            // Legacy reference needs the plain activation layout for row r.
            std::vector<char> q8row;
            if (fn.compare_legacy) {
                q8row = quantize_act_row(std::vector<float>(x.begin() + (size_t) r * k, x.begin() + (size_t) (r + 1) * k), k, fn.act_type);
            }
            for (int c = 0; c < nc; c++) {
                const double a = s_nat[(size_t) r * nc + c];
                if (has_gen_gemm) diff_update(st_ng, a, s_gen[(size_t) r * nc + c], tol);
                if (fn.compare_legacy) {
                    float ref = 0.0f;
                    fn.vec_dot(k, &ref, 0, rw.raw.data() + row_bytes * c, 0, q8row.data(), 0, 1);
                    diff_update(st_nl, a, ref, tol);
                }
            }
        }
        if (fn.compare_legacy && has_gen_gemm) {
            printf("  [%s] gemm nc=%-4d k=%-5d nr=%-3d native-vs-generic: max_abs=%.3g max_rel=%.3g bad=%d | native-vs-legacy: max_abs=%.3g max_rel=%.3g bad=%d\n",
                   fn.name, nc, k, nr, st_ng.max_abs, st_ng.max_rel, st_ng.n_bad, st_nl.max_abs, st_nl.max_rel, st_nl.n_bad);
        } else if (fn.compare_legacy) {
            printf("  [%s] gemm nc=%-4d k=%-5d nr=%-3d native-vs-generic: SKIPPED (no separate generic kernel) | native-vs-legacy: max_abs=%.3g max_rel=%.3g bad=%d\n",
                   fn.name, nc, k, nr, st_nl.max_abs, st_nl.max_rel, st_nl.n_bad);
        } else if (has_gen_gemm) {
            printf("  [%s] gemm nc=%-4d k=%-5d nr=%-3d native-vs-generic: max_abs=%.3g max_rel=%.3g bad=%d | native-vs-legacy: SKIPPED (different activation format)\n",
                   fn.name, nc, k, nr, st_ng.max_abs, st_ng.max_rel, st_ng.n_bad);
        } else {
            printf("  [%s] gemm nc=%-4d k=%-5d nr=%-3d native-vs-generic: SKIPPED (no native override) | native-vs-legacy: SKIPPED (different activation format)\n",
                   fn.name, nc, k, nr);
        }
        ok = ok && (!has_gen_gemm || st_ng.n_bad == 0) && (!fn.compare_legacy || st_nl.n_bad == 0);
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

    // Column slices follow the physical panel width. The UDNL/E4A kernels
    // require every per-thread slice to contain complete 16-column panels.
    std::vector<std::pair<int, int>> slices;
    {
        const int col_align = fn.type == GGML_TYPE_UDNL_W4 || fn.type == GGML_TYPE_UDNL_MX || fn.type == GGML_TYPE_E4A ? 16 : 8;
        int c = 0;
        for (int t = 0; t < nthreads && c < nc; t++) {
            int n = ((nc - c) / (nthreads - t)) / col_align * col_align;
            if (n == 0) n = std::min(col_align, nc - c);
            slices.push_back({c, n});
            c += n;
        }
    }

    // Times kernel wall time per iteration with columns partitioned across a
    // persistent team. Thread creation occurs before the timed region.
    auto time_it = [&](const char * label, auto && body) {
        for (const auto & sl : slices) {
            body(sl.first, sl.second);
        }
        std::chrono::steady_clock::time_point t0;
        std::chrono::steady_clock::time_point t1;
        if ((int) slices.size() <= 1) {
            t0 = std::chrono::steady_clock::now();
            for (int i = 0; i < n_iter; i++) body(0, nc);
            t1 = std::chrono::steady_clock::now();
        } else {
            std::atomic<int> ready{0};
            std::atomic<bool> go{false};
            std::vector<std::thread> ths;
            ths.reserve(slices.size());
            for (const auto & sl : slices) {
                ths.emplace_back([&, sl] {
                    ready.fetch_add(1, std::memory_order_release);
                    while (!go.load(std::memory_order_acquire)) {
                        std::this_thread::yield();
                    }
                    for (int i = 0; i < n_iter; i++) {
                        body(sl.first, sl.second);
                    }
                });
            }
            while (ready.load(std::memory_order_acquire) != (int) slices.size()) {
                std::this_thread::yield();
            }
            t0 = std::chrono::steady_clock::now();
            go.store(true, std::memory_order_release);
            for (auto & th : ths) {
                th.join();
            }
            t1 = std::chrono::steady_clock::now();
        }
        const double us = std::chrono::duration<double, std::micro>(t1 - t0).count() / n_iter;
        printf("  [%s] %-28s %12.1f us\n", fn.name, label, us);
        return us;
    };

    auto report = [&](const char * label, double us, double matrix_reads) {
        const double seconds = us / 1e6;
        const double gbps = seconds > 0.0 ? matrix_reads * rw.nbytes / seconds / 1e9 : 0.0;
        const double gops = seconds > 0.0 ? 2.0 * nc * (double) k * nr / seconds / 1e9 : 0.0;
        printf("  [%s] %-28s %8.1f GB/s matrix  %8.1f GOP/s\n", fn.name, label, gbps, gops);
    };

    // legacy: nr rows x nc cols via per-row vec_dot (dot only, acts pre-quantized)
    std::vector<std::vector<char>> q8rows(nr);
    for (int r = 0; r < nr; r++) {
        q8rows[r] = quantize_act_row(std::vector<float>(x.begin() + (size_t) r * k, x.begin() + (size_t) (r + 1) * k), k, fn.act_type);
    }
    std::vector<float> s((size_t) nr * nc);
    double t_legacy = 0.0;
    if (fn.compare_legacy) {
        t_legacy = time_it("legacy vec_dot (nr rows)", [&](int c0, int ncols) {
            for (int r = 0; r < nr; r++) {
                for (int c = c0; c < c0 + ncols; c++) {
                    fn.vec_dot(k, &s[(size_t) r * nc + c], 0, rw.raw.data() + row_bytes * c, 0, q8rows[r].data(), 0, 1);
                }
            }
        });
        report("legacy useful rates", t_legacy, nr);
    }

    if (nr == 1) {
        const double t = time_it("repack gemv", [&](int c0, int ncols) {
            fn.gemv(k, s.data() + c0, nc, (const char *) rw.data + row_bytes * c0, q8rows[0].data(), 1, ncols);
        });
        report("repack gemv useful rates", t, 1.0);
        if (fn.compare_legacy) {
            printf("  [%s] speedup gemv vs legacy:      %.2fx\n", fn.name, t_legacy / t);
        }
    } else if (fn.type == GGML_TYPE_MXFP4 && nr <= 8) {
        const size_t qrow_bytes = ggml_row_size(GGML_TYPE_Q8_0, k);
        std::vector<char> q8_plain(qrow_bytes * nr);
        for (int r = 0; r < nr; ++r) {
            memcpy(q8_plain.data() + (size_t) r * qrow_bytes, q8rows[r].data(), qrow_bytes);
        }

        const double t_separate = time_it("repack gemv separate rows", [&](int c0, int ncols) {
            for (int r = 0; r < nr; ++r) {
                fn.gemv(k, s.data() + (size_t) r * nc + c0, nc,
                        (const char *) rw.data + row_bytes * c0,
                        q8_plain.data() + (size_t) r * qrow_bytes, 1, ncols);
            }
        });
        report("separate gemv useful rates", t_separate, nr);

        const double t_shared = time_it("repack gemv shared weight", [&](int c0, int ncols) {
            fn.gemv(k, s.data() + c0, nc, (const char *) rw.data + row_bytes * c0,
                    q8_plain.data(), nr, ncols);
        });
        report("shared gemv useful rates", t_shared, 1.0);
        printf("  [MXFP4] shared-weight speedup nr=%-3d: %.2fx\n", nr, t_separate / t_shared);
        if (nr > 4) {
            const double t_split4 = time_it("repack gemv shared chunks 4", [&](int c0, int ncols) {
                for (int r = 0; r < nr; r += 4) {
                    const int tr = std::min(4, nr - r);
                    fn.gemv(k, s.data() + (size_t) r * nc + c0, nc,
                            (const char *) rw.data + row_bytes * c0,
                            q8_plain.data() + (size_t) r * qrow_bytes, tr, ncols);
                }
            });
            report("shared chunks-4 rates", t_split4, (nr + 3) / 4);
            printf("  [MXFP4] one-call vs chunks-4 nr=%-3d: %.2fx\n", nr, t_split4 / t_shared);
        }
    } else {
        std::vector<char> q8 = fn.type == GGML_TYPE_Q3_R || fn.type == GGML_TYPE_UDNL_W4 || fn.type == GGML_TYPE_UDNL_MX || fn.type == GGML_TYPE_E4A
                             ? quantize_act_rows(x, nr, k, fn.act_type)
                             : quantize_acts_4x8(x, nr, k, fn.act_type);
        const double t = time_it(("repack gemm nr=" + std::to_string(nr)).c_str(), [&](int c0, int ncols) {
            fn.gemm(k, s.data() + c0, nc, (const char *) rw.data + row_bytes * c0, q8.data(), nr, ncols);
        });
        report("repack gemm useful rates", t, 1.0);
        if (fn.compare_legacy) {
            printf("  [%s] speedup gemm nr=%-3d vs legacy: %.2fx\n", fn.name, nr, t_legacy / t);
        }
    }

    free_repacked(rw);
}

bool perf_mul_mat_id(
        const kernel_fns & fn,
        int nc,
        int k,
        int n_experts,
        int rows_per_expert,
        int nthreads,
        int n_iter,
        bool use_repack,
        int n_active_experts = 0,
        bool rotate_experts = false,
        const std::vector<int> * active_rows = nullptr) {
    repacked_weights rw;
    if (!make_expert_weights(fn.type, nc, k, n_experts, use_repack, rw)) {
        fprintf(stderr, "[%s] failed to create %s expert tensor\n", fn.name, use_repack ? "repacked" : "generic");
        return false;
    }

    // Optional NUMA-EP replication (GGML_NUMA_EP=1): init NUMA and bind each
    // expert's row windows to their node, matching the model loader's EP
    // placement and the mmid claim path's per-node windows. UDNL_MX repack
    // stores 16-row panels contiguously and ep_win is a multiple of 128 rows,
    // so a row window maps to one contiguous byte range per expert. Without
    // the env the bench keeps its historical flat (unpinned, unbound) behavior.
    {
        const char * numa_env = getenv("GGML_NUMA_EP");
        if (numa_env && atoi(numa_env) != 0) {
            ggml_numa_init(GGML_NUMA_STRATEGY_DISTRIBUTE);
            const int n_nodes = ggml_numa_node_count();
            if (n_nodes > 1) {
                const int64_t ep_win = (((int64_t) nc + n_nodes - 1)/n_nodes + 127)/128*128;
                const int64_t row_stride = rw.tensor->nb[2]/nc;
                for (int e = 0; e < n_experts; ++e) {
                    for (int n = 0; n < n_nodes; ++n) {
                        const int64_t r0 = (int64_t) n*ep_win;
                        const int64_t r1 = std::min<int64_t>(r0 + ep_win, nc);
                        if (r1 <= r0) continue;
                        ggml_numa_bind((char *) rw.data + e*rw.tensor->nb[2] + r0*row_stride,
                                       (size_t) ((r1 - r0)*row_stride), n);
                    }
                }
            }
        }
    }

    const int n_active = n_active_experts > 0 ? n_active_experts : n_experts;
    int n_tokens = n_active * rows_per_expert;
    if (active_rows != nullptr) {
        GGML_ASSERT((int) active_rows->size() == n_active);
        n_tokens = 0;
        for (int count : *active_rows) {
            GGML_ASSERT(count > 0);
            n_tokens += count;
        }
    }
    ggml_tensor * hidden = ggml_new_tensor_3d(rw.ctx, GGML_TYPE_F32, k, 1, n_tokens);
    ggml_tensor * ids = ggml_new_tensor_2d(rw.ctx, GGML_TYPE_I32, 1, n_tokens);
    ggml_set_input(hidden);
    ggml_set_input(ids);
    ggml_tensor * result = ggml_mul_mat_id(rw.ctx, rw.tensor, hidden, ids);
    ggml_cgraph * graph = ggml_new_graph(rw.ctx);
    ggml_build_forward_expand(graph, result);

    ggml_backend_t backend = ggml_backend_cpu_init();
    ggml_gallocr_t gallocr = ggml_gallocr_new(ggml_backend_cpu_buffer_type());
    if (backend == nullptr || gallocr == nullptr || !ggml_gallocr_alloc_graph(gallocr, graph)) {
        fprintf(stderr, "[%s] failed to allocate MUL_MAT_ID graph\n", fn.name);
        if (gallocr) ggml_gallocr_free(gallocr);
        if (backend) ggml_backend_free(backend);
        free_repacked(rw);
        return false;
    }

    struct ggml_threadpool_params tpp = ggml_threadpool_params_default(nthreads);
    ggml_threadpool_t threadpool = ggml_threadpool_new(&tpp);
    if (threadpool == nullptr) {
        fprintf(stderr, "[%s] failed to create threadpool\n", fn.name);
        ggml_gallocr_free(gallocr);
        ggml_backend_free(backend);
        free_repacked(rw);
        return false;
    }
    ggml_backend_cpu_set_n_threads(backend, nthreads);
    ggml_backend_cpu_set_threadpool(backend, threadpool);

    std::vector<float> hidden_data = make_random_f32((int64_t) n_tokens * k, 2718);
    std::vector<int32_t> id_data((size_t) n_tokens);
    auto fill_ids = [&](int iter) {
        const int base = rotate_experts ? (iter * n_active) % n_experts : 0;
        if (active_rows != nullptr) {
            int token = 0;
            for (int active = 0; active < n_active; ++active) {
                for (int row = 0; row < (*active_rows)[(size_t) active]; ++row) {
                    id_data[(size_t) token++] = (base + active) % n_experts;
                }
            }
            GGML_ASSERT(token == n_tokens);
        } else {
            for (int token = 0; token < n_tokens; ++token) {
                id_data[(size_t) token] = (base + token % n_active) % n_experts;
            }
        }
    };
    fill_ids(0);
    ggml_backend_tensor_set(hidden, hidden_data.data(), 0, hidden_data.size() * sizeof(float));
    ggml_backend_tensor_set(ids, id_data.data(), 0, id_data.size() * sizeof(int32_t));

    if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "[%s] MUL_MAT_ID warmup failed\n", fn.name);
        ggml_backend_free(backend);
        ggml_threadpool_free(threadpool);
        ggml_gallocr_free(gallocr);
        free_repacked(rw);
        return false;
    }

    double elapsed_ms = 0.0;
    for (int iter = 0; iter < n_iter; ++iter) {
        // Graph inputs may share the gallocr arena with dead intermediates.
        // Real callers upload them for every request; do the same outside the
        // timed region so repeated microbench iterations remain valid.
        fill_ids(iter + 1);
        ggml_backend_tensor_set(hidden, hidden_data.data(), 0, hidden_data.size() * sizeof(float));
        ggml_backend_tensor_set(ids, id_data.data(), 0, id_data.size() * sizeof(int32_t));
        const auto start = std::chrono::steady_clock::now();
        if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) {
            fprintf(stderr, "[%s] MUL_MAT_ID compute failed\n", fn.name);
            ggml_backend_free(backend);
            ggml_threadpool_free(threadpool);
            ggml_gallocr_free(gallocr);
            free_repacked(rw);
            return false;
        }
        const auto end = std::chrono::steady_clock::now();
        elapsed_ms += std::chrono::duration<double, std::milli>(end - start).count();
    }
    const double ms = elapsed_ms / n_iter;
    const double seconds = ms / 1000.0;
    const double useful_weight_bytes = (double) rw.nbytes * n_active / n_experts;
    const double weight_gbps = seconds > 0.0 ? useful_weight_bytes / seconds / 1e9 : 0.0;
    const double gops = seconds > 0.0 ?
        2.0 * nc * (double) k * n_tokens / seconds / 1e9 : 0.0;
    std::vector<float> checksum_data(std::min<int64_t>(result->ne[0], 16));
    ggml_backend_tensor_get(result, checksum_data.data(), 0, checksum_data.size() * sizeof(float));
    double checksum = 0.0;
    for (float value : checksum_data) {
        checksum += value;
    }
    printf("MMID_RESULT layout=%s type=%s nc=%d k=%d experts=%d active_experts=%d rotate=%d "
           "rows_per_expert=%d tokens=%d threads=%d iterations=%d ms=%.3f weight_gbps=%.3f gops=%.3f checksum=%.9g\n",
           use_repack ? "repack" : "generic", fn.name, nc, k, n_experts, n_active, rotate_experts ? 1 : 0,
           rows_per_expert, n_tokens, nthreads,
           n_iter, ms, weight_gbps, gops, checksum);

    ggml_backend_free(backend);
    ggml_threadpool_free(threadpool);
    ggml_gallocr_free(gallocr);
    free_repacked(rw);
    return true;
}

bool test_mmid_prequantized_q8_k() {
    // GLM-5.2 gate/up dimensions: the earlier small k=2048 probe missed
    // stride/workspace errors that only appear at the real 6144-wide row.
    constexpr int nc = 2048;
    constexpr int k = 6144;
    constexpr int n_experts = 2;
    constexpr int n_tokens = 2;

    repacked_weights rw;
    ggml_tensor * second_weight = nullptr;
    if (!make_expert_weight_pair(GGML_TYPE_IQ2_XS, nc, k, n_experts, true, rw, second_weight)) {
        printf("[MMID shared Q8_K] SKIPPED: IQ2_XS repack unavailable\n");
        return true;
    }

    ggml_backend_t backend = ggml_backend_cpu_init();
    constexpr int nthreads = 36;
    ggml_threadpool_params tpp = ggml_threadpool_params_default(nthreads);
    ggml_threadpool_t threadpool = ggml_threadpool_new(&tpp);
    if (backend == nullptr || threadpool == nullptr) {
        printf("[MMID shared Q8_K] FAILED: backend allocation\n");
        if (threadpool) ggml_threadpool_free(threadpool);
        if (backend) ggml_backend_free(backend);
        free_repacked(rw);
        return false;
    }
    ggml_backend_cpu_set_n_threads(backend, nthreads);
    ggml_backend_cpu_set_threadpool(backend, threadpool);

    const std::vector<float> hidden_data = make_random_f32((int64_t) n_tokens * k, 4242);
    const int32_t id_data[n_tokens] = {0, 1};

    // Compute the ordinary F32-input result in an independent allocation, then
    // release it. This prevents the additional reference branch from changing
    // the allocator layout of the shared-Q8 graph and masking aliasing bugs.
    ggml_tensor * hidden_ref = ggml_new_tensor_3d(rw.ctx, GGML_TYPE_F32, k, 1, n_tokens);
    ggml_tensor * ids_ref = ggml_new_tensor_2d(rw.ctx, GGML_TYPE_I32, 1, n_tokens);
    ggml_set_input(hidden_ref);
    ggml_set_input(ids_ref);
    ggml_tensor * result_ref = ggml_mul_mat_id(rw.ctx, rw.tensor, hidden_ref, ids_ref);
    ggml_cgraph * graph_ref = ggml_new_graph(rw.ctx);
    ggml_build_forward_expand(graph_ref, result_ref);
    ggml_gallocr_t alloc_ref = ggml_gallocr_new(ggml_backend_cpu_buffer_type());
    bool ok = alloc_ref != nullptr && ggml_gallocr_alloc_graph(alloc_ref, graph_ref);
    std::vector<float> expected((size_t) nc * n_tokens);
    if (ok) {
        ggml_backend_tensor_set(hidden_ref, hidden_data.data(), 0, hidden_data.size() * sizeof(float));
        ggml_backend_tensor_set(ids_ref, id_data, 0, sizeof(id_data));
        ok = ggml_backend_graph_compute(backend, graph_ref) == GGML_STATUS_SUCCESS;
    }
    if (ok) {
        ggml_backend_tensor_get(result_ref, expected.data(), 0, expected.size() * sizeof(float));
    }
    if (alloc_ref) ggml_gallocr_free(alloc_ref);

    ggml_tensor * hidden = ggml_new_tensor_3d(rw.ctx, GGML_TYPE_F32, k, 1, n_tokens);
    ggml_tensor * ids = ggml_new_tensor_2d(rw.ctx, GGML_TYPE_I32, 1, n_tokens);
    ggml_set_input(hidden);
    ggml_set_input(ids);
    ggml_tensor * hidden_q8 = ggml_cast(rw.ctx, hidden, GGML_TYPE_Q8_K);
    ggml_tensor * result_q8_a = ggml_mul_mat_id(rw.ctx, rw.tensor, hidden_q8, ids);
    ggml_tensor * result_q8_b = ggml_mul_mat_id(rw.ctx, second_weight, hidden_q8, ids);
    ggml_tensor * combined = ggml_add(rw.ctx, result_q8_a, result_q8_b);
    ggml_cgraph * graph_q8 = ggml_new_graph(rw.ctx);
    ggml_build_forward_expand(graph_q8, combined);
    ggml_gallocr_t alloc_q8 = ggml_gallocr_new(ggml_backend_cpu_buffer_type());
    ok = ok && alloc_q8 != nullptr && ggml_gallocr_alloc_graph(alloc_q8, graph_q8);
    std::vector<float> actual((size_t) nc * n_tokens);
    if (ok) {
        ggml_backend_tensor_set(hidden, hidden_data.data(), 0, hidden_data.size() * sizeof(float));
        ggml_backend_tensor_set(ids, id_data, 0, sizeof(id_data));
        ok = ggml_backend_graph_compute(backend, graph_q8) == GGML_STATUS_SUCCESS;
    }
    if (ok) {
        ggml_backend_tensor_get(combined, actual.data(), 0, actual.size() * sizeof(float));
        for (size_t i = 0; i < actual.size(); ++i) {
            if (actual[i] != expected[i] + expected[i]) {
                ok = false;
                break;
            }
        }
    }
    if (alloc_q8) ggml_gallocr_free(alloc_q8);
    printf("[MMID shared Q8_K] F32-internal-vs-prequantized: %s\n", ok ? "bit-exact" : "FAILED");

    ggml_backend_free(backend);
    ggml_threadpool_free(threadpool);
    free_repacked(rw);
    return ok;
}

// E4A exponent edge cases on the panel kernels. Real DSV4 experts carry
// e==0/e==1 dead groups inside otherwise-live, routed rows
// (quant-sweep/scan-e4a-exponents.py: ~0.4% of all groups, 68% of rows in a
// gate tensor sample). The AVX512 panel kernel must use the full e8m0-half
// construction there: the bare (e-1)<<23 turns e==0 into 0xFF800000 = -inf,
// and the (exactly zero) group dot then NaNs the whole accumulator via
// 0 x -inf — this was the production NaN/garbage bug. 0xFF groups are crafted
// with zeroed payloads (raw==0), matching how they occur in the wild; kernel
// (2^127) and reference (0.0) both yield exactly 0 then.
bool test_e4a_exponent_edges() {
    constexpr int nc = 16;              // one full panel
    constexpr int k  = 2048;            // 8 super-blocks x 8 groups = 64 groups/row
    const size_t row_bytes = ggml_row_size(GGML_TYPE_E4A, k);   // nb*136
    const int nb = k / 256;

    std::mt19937 rng(777);
    std::uniform_int_distribution<int> byte_dist(0, 255);

    // raw row blocks, written without block_e4a (layout: per super-block
    // e[8] at [b*136 + g], payload at [b*136 + 8 + 16*g + v])
    std::vector<char> raw(row_bytes * nc);
    for (int r = 0; r < nc; ++r) {
        char * row = raw.data() + row_bytes * r;
        for (int b = 0; b < nb; ++b) {
            for (int g = 0; g < 8; ++g) {
                const int gi = b * 8 + g;
                uint8_t e = (uint8_t) (118 + (gi + r) % 7);      // normal range
                bool zero_qs = false;
                switch ((gi + 16 * r) % 16) {
                    case 3:  e = 0;                 break;       // dead group, random payload
                    case 7:  e = 1;                 break;       // denormal e, random payload
                    case 11: e = 0xff; zero_qs = true; break;    // OCP NaN encoding, zero payload
                    case 15: e = 0;    zero_qs = true; break;    // dead group, zero payload
                    default: break;
                }
                row[b * 136 + g] = (char) e;
                for (int v = 0; v < 16; ++v) {
                    row[b * 136 + 8 + 16 * g + v] = zero_qs ? 0 : (char) byte_dist(rng);
                }
            }
        }
    }

    // repack through the real buffer path (set_tensor runs the byte gather)
    repacked_weights rw;
    {
        struct ggml_init_params params = { 1 * 1024 * 1024, nullptr, true };
        rw.ctx = ggml_init(params);
        rw.raw = raw;
        rw.tensor = rw.ctx ? ggml_new_tensor_2d(rw.ctx, GGML_TYPE_E4A, k, nc) : nullptr;
        rw.nbytes = rw.tensor ? ggml_nbytes(rw.tensor) : 0;
        rw.buffer = rw.tensor ? ggml_backend_buft_alloc_buffer(
            ggml_backend_cpu_repack_buffer_type(), rw.nbytes) : nullptr;
        if (rw.buffer == nullptr) {
            printf("[E4A edges] FAILED: buffer allocation\n");
            free_repacked(rw);
            return false;
        }
        ggml_backend_tensor_alloc(rw.buffer, rw.tensor, ggml_backend_buffer_get_base(rw.buffer));
        if (rw.tensor->extra == nullptr) {
            printf("[E4A edges] SKIPPED: E4A repack unavailable\n");
            free_repacked(rw);
            return true;
        }
        ggml_backend_tensor_set(rw.tensor, rw.raw.data(), 0, rw.raw.size());
        rw.data = rw.tensor->data;
    }

    constexpr int nr = 4;
    const std::vector<float> x = make_random_f32((int64_t) nr * k, 888);
    const std::vector<char> q8 = quantize_act_rows(x, nr, k, GGML_TYPE_Q8_0);
    const size_t qrow = ggml_row_size(GGML_TYPE_Q8_0, k);

    // fp64 reference: dequantized E4A rows x dequantized q8_0 activations
    const ggml_type_traits * e4a_traits = ggml_get_type_traits(GGML_TYPE_E4A);
    std::vector<double> expected((size_t) nr * nc, 0.0);
    {
        std::vector<float> wrow(k);
        std::vector<double> a_deq((size_t) nr * k);
        for (int r = 0; r < nr; ++r) {
            const block_q8_0 * qb = (const block_q8_0 *) (q8.data() + r * qrow);
            for (int j = 0; j < k / QK8_0; ++j) {
                const float d = ggml_fp16_to_fp32(qb[j].d);
                for (int v = 0; v < QK8_0; ++v) {
                    a_deq[(size_t) r * k + j * QK8_0 + v] = d * qb[j].qs[v];
                }
            }
        }
        for (int c = 0; c < nc; ++c) {
            e4a_traits->to_float(raw.data() + row_bytes * c, wrow.data(), k);
            for (int r = 0; r < nr; ++r) {
                double acc = 0.0;
                for (int i = 0; i < k; ++i) {
                    acc += wrow[i] * a_deq[(size_t) r * k + i];
                }
                expected[(size_t) r * nc + c] = acc;
            }
        }
    }

    std::vector<float> s_nat((size_t) nr * nc, 0.0f), s_gen((size_t) nr * nc, 0.0f);
    ggml_gemv_e4a_1x16_q8_0(k, s_nat.data(), nc, rw.data, q8.data(), nr, nc);
    ggml_gemv_e4a_1x16_q8_0_generic(k, s_gen.data(), nc, rw.data, q8.data(), nr, nc);

    bool ok = true;
    diff_stats st_nat, st_gen, st_row;
    for (int r = 0; r < nr; ++r) {
        for (int c = 0; c < nc; ++c) {
            float ref = 0.0f;
            ggml_vec_dot_e4a_q8_0(k, &ref, 0, raw.data() + row_bytes * c, 0, q8.data() + r * qrow, 0, 1);
            const double b = expected[(size_t) r * nc + c];
            if (!std::isfinite(s_nat[(size_t) r * nc + c]) || !std::isfinite(s_gen[(size_t) r * nc + c]) ||
                !std::isfinite(ref)) {
                ok = false;
            }
            diff_update(st_nat, s_nat[(size_t) r * nc + c], b, 2e-3);
            diff_update(st_gen, s_gen[(size_t) r * nc + c], b, 2e-3);
            diff_update(st_row, ref, b, 2e-3);
        }
    }
    ok = ok && st_nat.n_bad == 0 && st_gen.n_bad == 0 && st_row.n_bad == 0;
    printf("[E4A edges] e=0/1/0xFF groups: native %s (max_abs=%.3g bad=%d) | generic %s (bad=%d) | scalar %s (bad=%d)\n",
           st_nat.n_bad == 0 ? "ok" : "FAILED", st_nat.max_abs, st_nat.n_bad,
           st_gen.n_bad == 0 ? "ok" : "FAILED", st_gen.n_bad,
           st_row.n_bad == 0 ? "ok" : "FAILED", st_row.n_bad);

    free_repacked(rw);
    return ok;
}

// E4A mul_mat_id correctness through the full repack CPU path. Row counts per
// expert cover the three production branches: 40 rows (gemm arec main loop),
// 5 rows (batched arec gemv tail), 1 row (per-row arec gemv), 2 rows. A second
// leg runs the same graph in a plain (non-repack) buffer, exercising the
// legacy ggml_vec_dot_e4a_q8_0 fallback. Both legs are compared against an
// fp64 reference: dequantized E4A rows dotted with q8_0-quantized activations.
// With GGML_TEST_E4A_EP=1 (and GGML_NUMA_EP=1 in the environment from process
// start, so the cached flag picks it up) the test initializes NUMA distribute
// mode first, routing mul_mat_id through the two-phase EP row-claim protocol
// (n_tokens > GGML_NUMA_EP_STEAL_MIN_TOKENS enables the steal phase).
bool test_mmid_e4a() {
    const bool want_ep = getenv("GGML_TEST_E4A_EP") != nullptr;
    if (want_ep) {
        ggml_numa_init(GGML_NUMA_STRATEGY_DISTRIBUTE);
    }
    constexpr int nc = 2048;
    constexpr int k = 6144;
    constexpr int n_experts = 4;
    constexpr int n_tokens = 48;
    const int32_t id_data[n_tokens] = {
        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0, // 40 rows: gemm arec
        1,1,1,1,1,                               // 5 rows: batched arec tail
        2,                                       // 1 row: per-row arec gemv
        3,3,                                     // 2 rows: batched arec tail
    };

    repacked_weights gate;
    ggml_tensor * up_tensor = nullptr;
    if (!make_expert_weight_pair(GGML_TYPE_E4A, nc, k, n_experts, true, gate, up_tensor)) {
        printf("[MMID E4A] SKIPPED: E4A repack unavailable\n");
        return true;
    }
    repacked_weights plain;
    if (!make_expert_weights(GGML_TYPE_E4A, nc, k, n_experts, false, plain)) {
        printf("[MMID E4A] FAILED: plain buffer allocation\n");
        free_repacked(gate);
        return false;
    }

    // fp64 reference from the raw quantized plane (all experts share it)
    const size_t row_bytes = ggml_row_size(GGML_TYPE_E4A, k);
    const ggml_type_traits * e4a_traits = ggml_get_type_traits(GGML_TYPE_E4A);
    std::vector<double> w_deq((size_t) nc * k);
    {
        std::vector<float> row(k);
        for (int c = 0; c < nc; ++c) {
            e4a_traits->to_float(gate.raw.data() + row_bytes * c, row.data(), k);
            for (int i = 0; i < k; ++i) {
                w_deq[(size_t) c * k + i] = row[i];
            }
        }
    }
    const std::vector<float> hidden_data = make_random_f32((int64_t) n_tokens * k, 4242);
    const size_t act_row = ggml_row_size(GGML_TYPE_Q8_0, k);
    std::vector<char> act_q8((size_t) n_tokens * act_row);
    std::vector<double> a_deq((size_t) n_tokens * k);
    for (int t = 0; t < n_tokens; ++t) {
        quantize_row_q8_0(hidden_data.data() + (size_t) t * k,
                          act_q8.data() + t * act_row, k);
        const block_q8_0 * qb = (const block_q8_0 *) (act_q8.data() + t * act_row);
        for (int j = 0; j < k / QK8_0; ++j) {
            const float d = ggml_fp16_to_fp32(qb[j].d);
            for (int v = 0; v < QK8_0; ++v) {
                a_deq[(size_t) t * k + j * QK8_0 + v] = d * qb[j].qs[v];
            }
        }
    }
    std::vector<double> expected((size_t) nc * n_tokens);
    for (int t = 0; t < n_tokens; ++t) {
        for (int c = 0; c < nc; ++c) {
            double acc = 0.0;
            for (int i = 0; i < k; ++i) {
                acc += w_deq[(size_t) c * k + i] * a_deq[(size_t) t * k + i];
            }
            expected[(size_t) t * nc + c] = acc;
        }
    }

    ggml_backend_t backend = ggml_backend_cpu_init();
    constexpr int nthreads = 8;
    ggml_threadpool_params tpp = ggml_threadpool_params_default(nthreads);
    ggml_threadpool_t threadpool = ggml_threadpool_new(&tpp);
    bool ok = backend != nullptr && threadpool != nullptr;
    if (ok) {
        ggml_backend_cpu_set_n_threads(backend, nthreads);
        ggml_backend_cpu_set_threadpool(backend, threadpool);
    }

    auto run_leg = [&](ggml_tensor * w0, ggml_tensor * w1, ggml_context * ctx,
                       const char * name, double scale, diff_stats & st, bool prequantized) {
        ggml_tensor * hidden = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, k, 1, n_tokens);
        ggml_tensor * ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 1, n_tokens);
        ggml_set_input(hidden);
        ggml_set_input(ids);
        // The classic (non-repack) mul_mat_id only accepts F32 src1; the
        // prequantized shared-Q8 input is a repack-path feature.
        ggml_tensor * act = prequantized ? ggml_cast(ctx, hidden, GGML_TYPE_Q8_0) : hidden;
        ggml_tensor * out = ggml_mul_mat_id(ctx, w0, act, ids);
        if (w1 != nullptr) {
            out = ggml_add(ctx, out, ggml_mul_mat_id(ctx, w1, act, ids));
        }
        ggml_cgraph * graph = ggml_new_graph(ctx);
        ggml_build_forward_expand(graph, out);
        ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_cpu_buffer_type());
        bool leg_ok = alloc != nullptr && ggml_gallocr_alloc_graph(alloc, graph);
        std::vector<float> actual((size_t) nc * n_tokens);
        if (leg_ok) {
            ggml_backend_tensor_set(hidden, hidden_data.data(), 0, hidden_data.size() * sizeof(float));
            ggml_backend_tensor_set(ids, id_data, 0, sizeof(id_data));
            leg_ok = ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS;
        }
        if (leg_ok) {
            ggml_backend_tensor_get(out, actual.data(), 0, actual.size() * sizeof(float));
            for (size_t i = 0; i < actual.size(); ++i) {
                if (!std::isfinite(actual[i])) {
                    leg_ok = false;
                }
                diff_update(st, actual[i], scale * expected[i], 2e-3);
            }
            leg_ok = leg_ok && st.n_bad == 0;
        }
        if (alloc) ggml_gallocr_free(alloc);
        printf("[MMID E4A] %s: %s (max_abs=%.3g max_rel=%.3g bad=%d)\n",
               name, leg_ok ? "ok" : "FAILED", st.max_abs, st.max_rel, st.n_bad);
        return leg_ok;
    };

    diff_stats st_repack, st_plain;
    // repack leg: gate+up pair (identical weights) -> scale 2
    printf("[MMID E4A] repack leg with NUMA EP %s (nodes=%d)\n",
           want_ep ? "requested" : "off", ggml_numa_node_count());
    ok &= run_leg(gate.tensor, up_tensor, gate.ctx, "repack gemm/arec path", 2.0, st_repack, true);
    // plain leg: legacy vec_dot fallback (F32 src1; classic mmid quantizes internally) -> scale 1
    ok &= run_leg(plain.tensor, nullptr, plain.ctx, "plain vec_dot fallback", 1.0, st_plain, false);

    if (threadpool) ggml_threadpool_free(threadpool);
    if (backend) ggml_backend_free(backend);
    free_repacked(gate);
    free_repacked(plain);
    return ok;
}

bool perf_mul_mat_id_pair(
        const kernel_fns & fn,
        int nc,
        int k,
        int n_experts,
        int rows_per_expert,
        int nthreads,
        int n_iter,
        bool use_repack,
        bool shared_q8) {
    if (fn.act_type != GGML_TYPE_Q8_0 && fn.act_type != GGML_TYPE_Q8_K) {
        fprintf(stderr, "[%s] unsupported shared activation type\n", fn.name);
        return false;
    }

    repacked_weights gate;
    ggml_tensor * up_tensor = nullptr;
    if (!make_expert_weight_pair(fn.type, nc, k, n_experts, use_repack, gate, up_tensor)) {
        fprintf(stderr, "[%s] failed to create paired expert tensors\n", fn.name);
        free_repacked(gate);
        return false;
    }

    const int n_tokens = n_experts * rows_per_expert;
    ggml_tensor * hidden = ggml_new_tensor_3d(gate.ctx, GGML_TYPE_F32, k, 1, n_tokens);
    ggml_tensor * ids = ggml_new_tensor_2d(gate.ctx, GGML_TYPE_I32, 1, n_tokens);
    ggml_set_input(hidden);
    ggml_set_input(ids);
    ggml_tensor * input = shared_q8 ? ggml_cast(gate.ctx, hidden, fn.act_type) : hidden;
    ggml_tensor * gate_out = ggml_mul_mat_id(gate.ctx, gate.tensor, input, ids);
    ggml_tensor * up_out = ggml_mul_mat_id(gate.ctx, up_tensor, input, ids);
    // Keep both projection results live through a common consumer, matching
    // the gate/up -> SwiGLU lifetime in the EPD graph.
    ggml_tensor * combined = ggml_add(gate.ctx, gate_out, up_out);
    ggml_cgraph * graph = ggml_new_graph(gate.ctx);
    ggml_build_forward_expand(graph, combined);

    ggml_backend_t backend = ggml_backend_cpu_init();
    ggml_gallocr_t gallocr = ggml_gallocr_new(ggml_backend_cpu_buffer_type());
    if (backend == nullptr || gallocr == nullptr || !ggml_gallocr_alloc_graph(gallocr, graph)) {
        fprintf(stderr, "[%s] failed to allocate paired MUL_MAT_ID graph\n", fn.name);
        if (gallocr) ggml_gallocr_free(gallocr);
        if (backend) ggml_backend_free(backend);
        free_repacked(gate);
        return false;
    }

    ggml_threadpool_params tpp = ggml_threadpool_params_default(nthreads);
    ggml_threadpool_t threadpool = ggml_threadpool_new(&tpp);
    if (threadpool == nullptr) {
        fprintf(stderr, "[%s] failed to create paired benchmark threadpool\n", fn.name);
        ggml_gallocr_free(gallocr);
        ggml_backend_free(backend);
        free_repacked(gate);
        return false;
    }
    ggml_backend_cpu_set_n_threads(backend, nthreads);
    ggml_backend_cpu_set_threadpool(backend, threadpool);

    const std::vector<float> hidden_data = make_random_f32((int64_t) n_tokens * k, 2718);
    std::vector<int32_t> id_data((size_t) n_tokens);
    for (int token = 0; token < n_tokens; ++token) {
        id_data[(size_t) token] = token % n_experts;
    }
    ggml_backend_tensor_set(hidden, hidden_data.data(), 0, hidden_data.size() * sizeof(float));
    ggml_backend_tensor_set(ids, id_data.data(), 0, id_data.size() * sizeof(int32_t));

    bool ok = ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS;
    double elapsed_ms = 0.0;
    for (int iter = 0; ok && iter < n_iter; ++iter) {
        ggml_backend_tensor_set(hidden, hidden_data.data(), 0, hidden_data.size() * sizeof(float));
        ggml_backend_tensor_set(ids, id_data.data(), 0, id_data.size() * sizeof(int32_t));
        const auto start = std::chrono::steady_clock::now();
        ok = ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS;
        const auto end = std::chrono::steady_clock::now();
        elapsed_ms += std::chrono::duration<double, std::milli>(end - start).count();
    }
    if (ok) {
        const double ms = elapsed_ms / n_iter;
        const double seconds = ms / 1000.0;
        const double weight_gbps = (2.0 * gate.nbytes) / seconds / 1e9;
        const double gops = 4.0 * nc * (double) k * n_tokens / seconds / 1e9;
        std::vector<float> checksum_data(std::min<int64_t>(combined->ne[0], 16));
        ggml_backend_tensor_get(combined, checksum_data.data(), 0, checksum_data.size() * sizeof(float));
        double checksum = 0.0;
        for (float value : checksum_data) checksum += value;
        printf("MMID_PAIR_RESULT layout=%s quant=%s type=%s nc=%d k=%d experts=%d rows_per_expert=%d tokens=%d "
               "threads=%d iterations=%d ms=%.3f weight_gbps=%.3f gops=%.3f checksum=%.9g\n",
               use_repack ? "repack" : "generic", shared_q8 ? "shared" : "internal",
               fn.name, nc, k, n_experts, rows_per_expert, n_tokens,
               nthreads, n_iter, ms, weight_gbps, gops, checksum);
    } else {
        fprintf(stderr, "[%s] paired MUL_MAT_ID compute failed\n", fn.name);
    }

    ggml_backend_free(backend);
    ggml_threadpool_free(threadpool);
    ggml_gallocr_free(gallocr);
    free_repacked(gate);
    return ok;
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
    if (argc > 1 && std::string(argv[1]) == "--q2-policy") {
        const char * value = getenv("GGML_REPACK_Q2_K");
        const bool expected = value ? atoi(value) != 0 : !ggml_cpu_has_avx512_vnni();
        const bool actual = repack_supported(GGML_TYPE_Q2_K, 2048, 4096);
        printf("Q2_POLICY avx512_vnni=%d expected_repack=%d actual_repack=%d\n",
               ggml_cpu_has_avx512_vnni(), expected, actual);
        return expected == actual ? 0 : 1;
    }

    if (argc > 1 && std::string(argv[1]) == "--iq4-xs-policy") {
        const char * value = getenv("GGML_REPACK_IQ4_XS");
        const bool expected = value && atoi(value) != 0;
        const bool actual = repack_supported(GGML_TYPE_IQ4_XS, 6144, 2048);
        printf("IQ4_XS_POLICY expected_repack=%d actual_repack=%d\n", expected, actual);
        return expected == actual ? 0 : 1;
    }

    if (argc > 1 && std::string(argv[1]) == "--udnl-mx-imatrix-contract") {
        return test_udnl_mx_imatrix_contract() ? 0 : 1;
    }

#if defined(_WIN32)
    _putenv_s("GGML_REPACK_Q2_K", "1");
#else
    setenv("GGML_REPACK_Q2_K", "1", 1);
#endif

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
        // IQ4_XS currently has only the arch-neutral LUT repack kernels. Its
        // legacy path also uses Q8_K rather than Q8_0, so neither comparison
        // is numerically equivalent; graph-level tests cover this type.
        { GGML_TYPE_IQ4_XS, "IQ4_XS", ggml_vec_dot_iq4_xs_q8_K, ggml_gemv_iq4_xs_8x8_q8_0, nullptr,
          ggml_gemm_iq4_xs_8x8_q8_0, nullptr, GGML_TYPE_Q8_0, false },
        { GGML_TYPE_MXFP4, "MXFP4", ggml_vec_dot_mxfp4_q8_0, ggml_gemv_mxfp4_8x8_q8_0, ggml_gemv_mxfp4_8x8_q8_0_generic,
          ggml_gemm_mxfp4_8x8_q8_0, ggml_gemm_mxfp4_8x8_q8_0_generic, GGML_TYPE_Q8_0 },
        // Q3_R is an INTER_SIZE == 1 identity repack: gemm activations are plain
        // row-major q8_0 rows (see quantize_act_rows); no column-count constraint
        { GGML_TYPE_Q3_R, "Q3_R", ggml_vec_dot_q3_r_q8_0, ggml_gemv_q3_r_1x1_q8_0, ggml_gemv_q3_r_1x1_q8_0_generic,
          ggml_gemm_q3_r_1x1_q8_0, ggml_gemm_q3_r_1x1_q8_0_generic, GGML_TYPE_Q8_0 },
        // UDNL_W4 is an INTER_SIZE == 1 NR16xK4 panel repack: gemm activations
        // are plain row-major q8_0 rows (see quantize_act_rows), like Q3_R.
        // nc must be a multiple of 16; the nc=8/nc=520 shapes are skipped below.
        { GGML_TYPE_UDNL_W4, "UDNL_W4", ggml_vec_dot_udnl_w4_q8_0, ggml_gemv_udnl_w4_1x16_q8_0, ggml_gemv_udnl_w4_1x16_q8_0_generic,
          ggml_gemm_udnl_w4_1x16_q8_0, ggml_gemm_udnl_w4_1x16_q8_0_generic, GGML_TYPE_Q8_0 },
        // UDNL_MX is an INTER_SIZE == 1 NR16 panel repack like UDNL_W4, with
        // a panel-shared per-group W2/W3/W4 mode word. nc must be a multiple
        // of 16; the nc=8/nc=520 shapes are skipped below.
        { GGML_TYPE_UDNL_MX, "UDNL_MX", ggml_vec_dot_udnl_mx_q8_0, ggml_gemv_udnl_mx_1x16_q8_0, ggml_gemv_udnl_mx_1x16_q8_0_generic,
          ggml_gemm_udnl_mx_1x16_q8_0, ggml_gemm_udnl_mx_1x16_q8_0_generic, GGML_TYPE_Q8_0 },
        // E4A is an INTER_SIZE == 1 NR16xK4 panel repack like UDNL_W4 (E2M1x2
        // grid + per-group E8M0 instead of NL grid + d/srel). nc must be a
        // multiple of 16; the nc=8/nc=520 shapes are skipped below.
        { GGML_TYPE_E4A, "E4A", ggml_vec_dot_e4a_q8_0, ggml_gemv_e4a_1x16_q8_0, ggml_gemv_e4a_1x16_q8_0_generic,
          ggml_gemm_e4a_1x16_q8_0, ggml_gemm_e4a_1x16_q8_0_generic, GGML_TYPE_Q8_0 },
        // Q8_0 has no generic 8x8 kernels; native-vs-generic checks are skipped for it
        { GGML_TYPE_Q8_0, "Q8_0", ggml_vec_dot_q8_0_q8_0, ggml_gemv_q8_0_8x8_q8_0, nullptr,
          ggml_gemm_q8_0_8x8_q8_0, nullptr, GGML_TYPE_Q8_0 },
    };

    if (argc > 1 && std::string(argv[1]) == "--mmid-perf") {
        if (argc < 8) {
            fprintf(stderr, "usage: %s --mmid-perf NC K EXPERTS ROWS_PER_EXPERT THREADS TYPE [ITERS] [repack|generic]\n", argv[0]);
            return 1;
        }
        const int nc = atoi(argv[2]);
        const int k = atoi(argv[3]);
        const int n_experts = atoi(argv[4]);
        const int rows_per_expert = atoi(argv[5]);
        const int nthreads = atoi(argv[6]);
        const std::string only = argv[7];
        const int n_iter = argc > 8 ? atoi(argv[8]) : 5;
        const std::string layout = argc > 9 ? argv[9] : "repack";
        if (layout != "repack" && layout != "generic") {
            fprintf(stderr, "MMID layout must be repack or generic\n");
            return 1;
        }
        if (nc < 8 || nc % 8 != 0 || k < 1 || n_experts < 1 || rows_per_expert < 1 ||
            nthreads < 1 || n_iter < 1) {
            fprintf(stderr, "invalid MMID benchmark dimensions\n");
            return 1;
        }
        for (const auto & fn : types) {
            if (only == fn.name) {
                return perf_mul_mat_id(fn, nc, k, n_experts, rows_per_expert, nthreads, n_iter, layout == "repack") ? 0 : 1;
            }
        }
        fprintf(stderr, "unknown repack type '%s'\n", only.c_str());
        return 1;
    }

    if (argc > 1 && std::string(argv[1]) == "--mmid-routed-perf") {
        if (argc < 9) {
            fprintf(stderr, "usage: %s --mmid-routed-perf NC K EXPERTS ACTIVE_EXPERTS ROWS_PER_EXPERT THREADS TYPE [ITERS] [repack|generic]\n", argv[0]);
            return 1;
        }
        const int nc = atoi(argv[2]);
        const int k = atoi(argv[3]);
        const int n_experts = atoi(argv[4]);
        const int n_active_experts = atoi(argv[5]);
        const int rows_per_expert = atoi(argv[6]);
        const int nthreads = atoi(argv[7]);
        const std::string only = argv[8];
        const int n_iter = argc > 9 ? atoi(argv[9]) : 64;
        const std::string layout = argc > 10 ? argv[10] : "repack";
        if (layout != "repack" && layout != "generic") {
            fprintf(stderr, "MMID layout must be repack or generic\n");
            return 1;
        }
        if (nc < 8 || nc % 8 != 0 || k < 1 || n_experts < 1 || n_active_experts < 1 ||
                n_active_experts > n_experts || rows_per_expert < 1 || nthreads < 1 || n_iter < 1) {
            fprintf(stderr, "invalid routed MMID benchmark dimensions\n");
            return 1;
        }
        for (const auto & fn : types) {
            if (only == fn.name) {
                return perf_mul_mat_id(fn, nc, k, n_experts, rows_per_expert, nthreads, n_iter,
                                       layout == "repack", n_active_experts, true) ? 0 : 1;
            }
        }
        fprintf(stderr, "unknown repack type '%s'\n", only.c_str());
        return 1;
    }

    if (argc > 1 && std::string(argv[1]) == "--mmid-skew-perf") {
        if (argc < 9) {
            fprintf(stderr, "usage: %s --mmid-skew-perf NC K EXPERTS COUNTS THREADS TYPE ITERS [repack|generic]\n", argv[0]);
            return 1;
        }
        const int nc = atoi(argv[2]);
        const int k = atoi(argv[3]);
        const int n_experts = atoi(argv[4]);
        std::vector<int> active_rows;
        {
            const std::string spec = argv[5];
            size_t off = 0;
            while (off <= spec.size()) {
                const size_t comma = spec.find(',', off);
                const std::string item = spec.substr(off, comma == std::string::npos ? comma : comma - off);
                const int count = atoi(item.c_str());
                if (count < 1 || count > 4) {
                    fprintf(stderr, "COUNTS entries must be in [1, 4]\n");
                    return 1;
                }
                active_rows.push_back(count);
                if (comma == std::string::npos) break;
                off = comma + 1;
            }
        }
        const int nthreads = atoi(argv[6]);
        const std::string only = argv[7];
        const int n_iter = atoi(argv[8]);
        const std::string layout = argc > 9 ? argv[9] : "repack";
        if ((layout != "repack" && layout != "generic") || nc < 8 || nc % 8 != 0 || k < 1 ||
                n_experts < 1 || active_rows.empty() || (int) active_rows.size() > n_experts ||
                nthreads < 1 || n_iter < 1) {
            fprintf(stderr, "invalid skewed MMID benchmark arguments\n");
            return 1;
        }
        for (const auto & fn : types) {
            if (only == fn.name) {
                return perf_mul_mat_id(fn, nc, k, n_experts, 1, nthreads, n_iter,
                                       layout == "repack", (int) active_rows.size(), true, &active_rows) ? 0 : 1;
            }
        }
        fprintf(stderr, "unknown repack type '%s'\n", only.c_str());
        return 1;
    }

    if (argc > 1 && std::string(argv[1]) == "--mmid-pair-perf") {
        if (argc < 10) {
            fprintf(stderr, "usage: %s --mmid-pair-perf NC K EXPERTS ROWS_PER_EXPERT THREADS TYPE ITERS shared|internal|generic\n", argv[0]);
            return 1;
        }
        const int nc = atoi(argv[2]);
        const int k = atoi(argv[3]);
        const int n_experts = atoi(argv[4]);
        const int rows_per_expert = atoi(argv[5]);
        const int nthreads = atoi(argv[6]);
        const std::string only = argv[7];
        const int n_iter = atoi(argv[8]);
        const std::string quant = argv[9];
        if (quant != "shared" && quant != "internal" && quant != "generic") {
            fprintf(stderr, "paired MMID mode must be shared, internal, or generic\n");
            return 1;
        }
        if (nc < 8 || nc % 8 != 0 || k < 1 || n_experts < 1 || rows_per_expert < 1 ||
                nthreads < 1 || n_iter < 1) {
            fprintf(stderr, "invalid paired MMID benchmark dimensions\n");
            return 1;
        }
        for (const auto & fn : types) {
            if (only == fn.name) {
                return perf_mul_mat_id_pair(fn, nc, k, n_experts, rows_per_expert, nthreads, n_iter,
                                            quant != "generic", quant == "shared") ? 0 : 1;
            }
        }
        fprintf(stderr, "unknown repack type '%s'\n", only.c_str());
        return 1;
    }

    if (argc > 1 && std::string(argv[1]) == "--perf-shape") {
        if (argc < 5) {
            fprintf(stderr, "usage: %s --perf-shape NC K THREADS [TYPE] [NR[,NR...]]\n", argv[0]);
            return 1;
        }
        const int nc = atoi(argv[2]);
        const int k = atoi(argv[3]);
        const int nthreads = atoi(argv[4]);
        const std::string only = argc > 5 ? argv[5] : "";
        std::vector<int> nrs = {1, 4, 8, 16, 32};
        if (argc > 6) {
            nrs.clear();
            const std::string spec = argv[6];
            size_t off = 0;
            while (off <= spec.size()) {
                const size_t comma = spec.find(',', off);
                const std::string item = spec.substr(off, comma == std::string::npos ? comma : comma - off);
                const int nr = atoi(item.c_str());
                if (nr != 1 && (nr < 4 || nr % 4 != 0)) {
                    fprintf(stderr, "NR must be 1 or a positive multiple of 4\n");
                    return 1;
                }
                nrs.push_back(nr);
                if (comma == std::string::npos) {
                    break;
                }
                off = comma + 1;
            }
        }
        if (nc < 8 || nc % 8 != 0 || k < 1 || nthreads < 1 || nrs.empty()) {
            fprintf(stderr, "NC must be a positive multiple of 8; K, THREADS, and NR must be positive\n");
            return 1;
        }
        printf("== shape nc=%d k=%d threads=%d ==\n", nc, k, nthreads);
        bool matched = false;
        for (const auto & fn : types) {
            if (!only.empty() && only != fn.name) {
                continue;
            }
            matched = true;
            if (!repack_supported(fn.type, nc, k)) {
                printf("  [%s] SKIPPED: no CPU_REPACK kernel for this build\n", fn.name);
                continue;
            }
            for (const int nr : nrs) {
                const int64_t dots = (int64_t) nc * nr;
                const int n_iter = dots > 200000 ? 4 : (dots > 50000 ? 8 : 20);
                perf_type(fn, nc, k, nr, nthreads, n_iter);
            }
        }
        if (!matched) {
            fprintf(stderr, "unknown repack type '%s'\n", only.c_str());
            return 1;
        }
        return 0;
    }

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
        if (fn.type == GGML_TYPE_UDNL_W4 || fn.type == GGML_TYPE_UDNL_MX || fn.type == GGML_TYPE_E4A) {
            // UDNL panels are 16 columns wide; nc=8 and nc=520 cannot be
            // repacked (the registration rejects them, row vec_dot covers them)
            printf("[%s] nc=8/nc=520 shapes skipped (require nc %% 16 == 0)\n", fn.name);
        } else {
            printf("[%s] nc=8 k=2048 (256-bit tail column group only)\n", fn.name);
            ok &= test_type(fn, 8, 2048, { 4, 8, 16 }, false);
            printf("[%s] nc=520 k=2048 (odd trailing 8-column group)\n", fn.name);
            ok &= test_type(fn, 520, 2048, { 4, 20 }, false);
        }
        if (fn.type == GGML_TYPE_MXFP4) {
            printf("[%s] nc=64 k=1024 (small-batch model path)\n", fn.name);
            ok &= test_type(fn, 64, 1024, { 8 }, false);
        }
    }

    ok &= test_mmid_prequantized_q8_k();
    ok &= test_e4a_exponent_edges();
    ok &= test_mmid_e4a();
    ok &= test_udnl_mx_imatrix_contract();

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
