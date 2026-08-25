// Correctness + micro-benchmark tool for the IQ2_XS / IQ3_XXS repack (8x8 interleaved) traits.
//
//   test mode (default): 1. generic gemv/gemm vs ggml_vec_dot_*_q8_K (integer-exact reference)
//                        2. native (AVX512) gemv/gemm vs generic
//                        3. end-to-end ggml graph: mul_mat / mul_mat_id with the weight in a
//                           CPU_REPACK buffer vs the plain vec_dot path
//   bench mode:          kernel-level timing: vec_dot vs repack gemv (1 row) and gemm (8 rows)
//
// The default correctness mode is registered in CTest. The benchmark modes
// remain manual so timing does not become a CI pass condition.

#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-backend.h"
#include "ggml-cpu/repack.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <random>
#include <string>
#include <vector>
#include <chrono>

extern "C" {
void ggml_vec_dot_iq2_xs_q8_K (int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc);
void ggml_vec_dot_iq3_xxs_q8_K(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc);
void quantize_row_q8_K(const float * x, void * y, int64_t k);
void quantize_row_q8_0(const float * x, void * y, int64_t k);
void ggml_gemm_mxfp4_8x8_q8_0(int n, float * s, size_t bs, const void * vx, const void * vy, int nr, int nc);
}

static std::mt19937 g_rng(1234);

static float frand(float lo = -1.0f, float hi = 1.0f) {
    return std::uniform_real_distribution<float>(lo, hi)(g_rng);
}

static uint32_t urand() {
    return g_rng();
}

// ---------------------------------------------------------------------------
// random weight blocks (any bit pattern is a valid codebook entry for these types)
// ---------------------------------------------------------------------------

static void fill_random_iq2_xs(block_iq2_xs * x, int64_t nblocks) {
    for (int64_t i = 0; i < nblocks; i++) {
        x[i].d = ggml_fp32_to_fp16(frand(0.05f, 0.3f) * (urand() & 1 ? 1 : -1));
        for (int k = 0; k < QK_K/8; k++) {
            x[i].qs[k] = (uint16_t) urand();
        }
        for (int k = 0; k < QK_K/32; k++) {
            x[i].scales[k] = (uint8_t) urand();
        }
    }
}

static void fill_random_iq3_xxs(block_iq3_xxs * x, int64_t nblocks) {
    for (int64_t i = 0; i < nblocks; i++) {
        x[i].d = ggml_fp32_to_fp16(frand(0.05f, 0.3f) * (urand() & 1 ? 1 : -1));
        for (int k = 0; k < 3*QK_K/8; k++) {
            x[i].qs[k] = (uint8_t) urand();
        }
    }
}

static void fill_random_q2_K(block_q2_K * x, int64_t nblocks) {
    for (int64_t i = 0; i < nblocks; i++) {
        // d/dmin sit in a trailing {d, dmin} pair (access path varies by GGML_COMMON_AGGR)
        ggml_half * hd = (ggml_half *) ((char *) (x + i + 1) - 2*sizeof(ggml_half));
        hd[0] = ggml_fp32_to_fp16(frand(0.05f, 0.3f) * (urand() & 1 ? 1 : -1));
        hd[1] = ggml_fp32_to_fp16(frand(0.01f, 0.1f));
        for (int k = 0; k < QK_K/16; k++) {
            x[i].scales[k] = (uint8_t) urand();
        }
        for (int k = 0; k < QK_K/4; k++) {
            x[i].qs[k] = (uint8_t) urand();
        }
    }
}

static void fill_random_mxfp4(block_mxfp4 * x, int64_t nblocks) {
    for (int64_t i = 0; i < nblocks; ++i) {
        x[i].e = (uint8_t) (118 + urand() % 10);
        for (int k = 0; k < QK_MXFP4/2; ++k) {
            x[i].qs[k] = (uint8_t) urand();
        }
    }
}

// ---------------------------------------------------------------------------
// diff helpers
// ---------------------------------------------------------------------------

struct diff_stats {
    double max_abs = 0;
    double max_rel = 0;
    int    n_bad   = 0;
};

static diff_stats compare(const char * what, const float * a, const float * b, int64_t n, double tol) {
    diff_stats st;
    for (int64_t i = 0; i < n; i++) {
        const double d = std::fabs((double) a[i] - (double) b[i]);
        const double r = d / std::max(1e-4, (double) std::fabs(b[i]));
        st.max_abs = std::max(st.max_abs, d);
        st.max_rel = std::max(st.max_rel, r);
        if (r > tol && d > 1e-4) {
            if (st.n_bad < 5) {
                printf("    MISMATCH %s[%lld]: got %.9g want %.9g\n", what, (long long) i, a[i], b[i]);
            }
            st.n_bad++;
        }
    }
    printf("    %-44s max_abs=%.3g max_rel=%.3g bad=%d %s\n", what, st.max_abs, st.max_rel, st.n_bad,
           st.n_bad == 0 ? "OK" : "FAIL");
    return st;
}

// ---------------------------------------------------------------------------
// repack the plain weight rows through the real CPU_REPACK buffer machinery
// ---------------------------------------------------------------------------

struct repacked_weights {
    ggml_context *             ctx = nullptr;
    ggml_backend_buffer_t      buf = nullptr;
    ggml_tensor *              t   = nullptr; // tensor->data points at the interleaved layout
};

template <typename BLOCK>
static repacked_weights repack_via_buffer(ggml_type type, const BLOCK * plain, int64_t ne00, int64_t ne01, int64_t ne02 = 1) {
    repacked_weights rw;
    rw.ctx = ggml_init({ 1024 * 1024, nullptr, true });
    rw.t   = ne02 == 1 ? ggml_new_tensor_2d(rw.ctx, type, ne00, ne01)
                       : ggml_new_tensor_3d(rw.ctx, type, ne00, ne01, ne02);
    rw.buf = ggml_backend_buft_alloc_buffer(ggml_backend_cpu_repack_buffer_type(), ggml_nbytes(rw.t));
    if (!rw.buf) {
        fprintf(stderr, "failed to allocate repack buffer\n");
        exit(1);
    }
    ggml_backend_tensor_alloc(rw.buf, rw.t, ggml_backend_buffer_get_base(rw.buf));
    if (rw.t->extra == nullptr) {
        fprintf(stderr, "repack is unavailable for %s\n", ggml_type_name(type));
        exit(1);
    }
    ggml_backend_tensor_set(rw.t, plain, 0, ggml_nbytes(rw.t));
    return rw;
}

static void free_repacked(repacked_weights & rw) {
    ggml_backend_buffer_free(rw.buf);
    ggml_free(rw.ctx);
    rw = {};
}

// ---------------------------------------------------------------------------
// kernel-level tests
// ---------------------------------------------------------------------------

template <typename BLOCK>
static int test_gemv(const char * name, ggml_type type,
                     void (*fill)(BLOCK *, int64_t),
                     void (*vec_dot)(int, float *, size_t, const void *, size_t, const void *, size_t, int),
                     void (*gemv)(int, float *, size_t, const void *, const void *, int, int),
                     void (*gemv_generic)(int, float *, size_t, const void *, const void *, int, int)) {
    printf("  %s gemv kernels\n", name);
    const int64_t ne00 = 1024;
    const int64_t nb   = ne00 / QK_K;
    const int64_t nc   = 64; // 8 interleave groups

    std::vector<BLOCK> plain(nc * nb);
    fill(plain.data(), (int64_t) plain.size());

    repacked_weights rw = repack_via_buffer<BLOCK>(type, plain.data(), ne00, nc);

    // q8 activation
    std::vector<float>        xf(ne00);
    std::vector<block_q8_K>   y(nb);
    for (auto & v : xf) v = frand();
    quantize_row_q8_K(xf.data(), y.data(), ne00);

    // reference: vec_dot per column on the plain blocks
    std::vector<float> ref(nc);
    for (int64_t j = 0; j < nc; j++) {
        vec_dot(ne00, &ref[j], 0, plain.data() + j * nb, 0, y.data(), 0, 1);
    }

    std::vector<float> out_native(nc, 0), out_generic(nc, 0);
    gemv        (ne00, out_native.data(),  0, rw.t->data, y.data(), 1, nc);
    gemv_generic(ne00, out_generic.data(), 0, rw.t->data, y.data(), 1, nc);

    int bad = 0;
    bad += compare("generic gemv vs vec_dot", out_generic.data(), ref.data(), nc, 1e-5).n_bad;
    bad += compare("native  gemv vs vec_dot", out_native.data(),  ref.data(), nc, 1e-5).n_bad;
    bad += compare("native  gemv vs generic", out_native.data(),  out_generic.data(), nc, 1e-5).n_bad;

    free_repacked(rw);
    return bad;
}

template <typename BLOCK>
static int test_gemm(const char * name, ggml_type type,
                     void (*fill)(BLOCK *, int64_t),
                     void (*vec_dot)(int, float *, size_t, const void *, size_t, const void *, size_t, int),
                     void (*gemm)(int, float *, size_t, const void *, const void *, int, int),
                     void (*gemm_generic)(int, float *, size_t, const void *, const void *, int, int)) {
    printf("  %s gemm kernels\n", name);
    const int64_t ne00 = 1024;
    const int64_t nb   = ne00 / QK_K;
    const int64_t nc   = 64;
    const int     nr   = 20; // 5 four-row tiles, exercises the >16 tail shape

    std::vector<BLOCK> plain(nc * nb);
    fill(plain.data(), (int64_t) plain.size());

    repacked_weights rw = repack_via_buffer<BLOCK>(type, plain.data(), ne00, nc);

    // q8_Kx4 activation tiles
    std::vector<float>       xf(nr * ne00);
    std::vector<block_q8_Kx4> y4((nr / 4) * nb);
    for (auto & v : xf) v = frand();
    for (int yy = 0; yy < nr / 4; yy++) {
        ggml_quantize_mat_q8_K_4x8(xf.data() + (int64_t) yy * 4 * ne00, y4.data() + (int64_t) yy * nb, ne00);
    }

    // reference: vec_dot with row m's q8 data extracted from the interleaved tile
    // (iq2_xs / iq3_xxs vec_dot do not use bsums, so they can be left zero)
    std::vector<float> ref(nr * nc);
    for (int m = 0; m < nr; m++) {
        std::vector<block_q8_K> ym(nb);
        for (int64_t i = 0; i < nb; i++) {
            const block_q8_Kx4 & t = y4[(m / 4) * nb + i];
            ym[i].d = t.d[m % 4];
            for (int k = 0; k < QK_K/8; k++) {
                memcpy(ym[i].qs + 8*k, t.qs + (k/2)*64 + (k%2)*32 + (m%4)*8, 8);
            }
            memset(ym[i].bsums, 0, sizeof(ym[i].bsums));
        }
        for (int64_t j = 0; j < nc; j++) {
            vec_dot(ne00, &ref[m * nc + j], 0, plain.data() + j * nb, 0, ym.data(), 0, 1);
        }
    }

    std::vector<float> out_native(nr * nc, 0), out_generic(nr * nc, 0);
    gemm        (ne00, out_native.data(),  nc, rw.t->data, y4.data(), nr, nc);
    gemm_generic(ne00, out_generic.data(), nc, rw.t->data, y4.data(), nr, nc);

    int bad = 0;
    bad += compare("generic gemm vs vec_dot", out_generic.data(), ref.data(), nr * nc, 1e-5).n_bad;
    bad += compare("native  gemm vs vec_dot", out_native.data(),  ref.data(), nr * nc, 1e-5).n_bad;
    bad += compare("native  gemm vs generic", out_native.data(),  out_generic.data(), nr * nc, 1e-5).n_bad;

    free_repacked(rw);
    return bad;
}

// ---------------------------------------------------------------------------
// end-to-end graph tests (repack buffer vs plain vec_dot path)
// ---------------------------------------------------------------------------

template <typename BLOCK>
static int test_graph(const char * name, ggml_type type, void (*fill)(BLOCK *, int64_t), ggml_backend_t backend,
                      int64_t ntok, bool test_prequantized = false,
                      int64_t ne00 = 1024, int64_t ne01 = 64, int64_t nexp = 8, int64_t nids = 3) {
    printf("  %s end-to-end (repack buffer vs vec_dot, k=%lld, m=%lld, experts=%lld, topk=%lld, ntok=%lld)\n",
           name, (long long) ne00, (long long) ne01, (long long) nexp,
           (long long) nids, (long long) ntok);
    const int64_t nb   = ne00 / ggml_blck_size(type);
    // ntok >= 4 -> repack gemm path for mul_mat; ntok > 8 -> mul_mat_id large-batch
    // branch (gemm tiles + gather/scatter); ntok <= 8 -> mul_mat_id small-batch gemv

    std::vector<BLOCK> plain(ne01 * nexp * nb);
    fill(plain.data(), (int64_t) plain.size());

    std::vector<float> xf(ne00 * ntok);
    for (auto & v : xf) v = frand();
    std::vector<block_q8_0> xq((ne00 / QK8_0) * ntok);
    for (int64_t i = 0; i < ntok; ++i) {
        quantize_row_q8_0(xf.data() + i * ne00, xq.data() + i * ne00 / QK8_0, ne00);
    }

    int quant_bad = 0;
    std::vector<block_q8_0x4> xq4;
    if (test_prequantized) {
        const int64_t q8_blocks = ne00 / QK8_0;
        xq4.resize(q8_blocks * (ntok / 4));
        for (int64_t group = 0; group < ntok / 4; ++group) {
            block_q8_0x4 * interleaved = xq4.data() + group * q8_blocks;
            ggml_quantize_mat_q8_0_4x8(xf.data() + group * 4 * ne00, interleaved, ne00);
            for (int64_t block = 0; block < q8_blocks; ++block) {
                for (int row = 0; row < 4; ++row) {
                    const block_q8_0 & plain_q = xq[(group * 4 + row) * q8_blocks + block];
                    quant_bad += interleaved[block].d[row] != plain_q.d;
                    for (int chunk = 0; chunk < QK8_0 / 8; ++chunk) {
                        quant_bad += memcmp(interleaved[block].qs + chunk * 32 + row * 8,
                                            plain_q.qs + chunk * 8, 8) != 0;
                    }
                }
            }
        }
        printf("    q8_0 row vs 4x8 interleave mismatches=%d %s\n",
               quant_bad, quant_bad == 0 ? "OK" : "FAIL");
    }

    std::vector<int32_t> ids(nids * ntok);
    for (auto & v : ids) v = (int32_t) (urand() % nexp);

    repacked_weights rw_mmid = repack_via_buffer<BLOCK>(type, plain.data(), ne00, ne01, nexp);
    repacked_weights rw_mm   = repack_via_buffer<BLOCK>(type, plain.data(), ne00, ne01);

    const auto run = [&](bool use_repack, bool is_mmid, bool prequantized, std::vector<float> & dst) {
        repacked_weights & rw = is_mmid ? rw_mmid : rw_mm;
        ggml_context * ctx = use_repack
            ? rw.ctx
            : ggml_init({ ggml_tensor_overhead() * 16 + ggml_graph_overhead(), nullptr, true });
        ggml_tensor * src0 = use_repack
            ? rw.t
            : (is_mmid ? ggml_new_tensor_3d(ctx, type, ne00, ne01, nexp)
                       : ggml_new_tensor_2d(ctx, type, ne00, ne01));
        const ggml_type src1_type = prequantized ? GGML_TYPE_Q8_0 : GGML_TYPE_F32;
        ggml_tensor * src1 = is_mmid ? ggml_new_tensor_3d(ctx, src1_type, ne00, 1, ntok)
                                     : ggml_new_tensor_2d(ctx, src1_type, ne00, ntok);
        ggml_tensor * out;
        ggml_tensor * idst = nullptr;
        if (is_mmid) {
            idst = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, nids, ntok);
            out = ggml_mul_mat_id(ctx, src0, src1, idst);
        } else {
            out = ggml_mul_mat(ctx, src0, src1);
        }
        ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, out);
        ggml_gallocr_t gallocr = ggml_gallocr_new(ggml_backend_cpu_buffer_type());
        if (gallocr == nullptr || !ggml_gallocr_alloc_graph(gallocr, gf)) {
            fprintf(stderr, "graph allocation failed\n");
            exit(1);
        }
        if (idst != nullptr) {
            ggml_backend_tensor_set(idst, ids.data(), 0, ids.size() * sizeof(int32_t));
        }
        if (prequantized) {
            ggml_backend_tensor_set(src1, xq.data(), 0, ggml_nbytes(src1));
        } else {
            ggml_backend_tensor_set(src1, xf.data(), 0, xf.size() * sizeof(float));
        }
        if (!use_repack) {
            ggml_backend_tensor_set(src0, plain.data(), 0, ggml_nbytes(src0));
        }
        if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
            fprintf(stderr, "graph compute failed\n");
            exit(1);
        }
        dst.resize(ggml_nelements(out));
        ggml_backend_tensor_get(out, dst.data(), 0, ggml_nbytes(out));
        ggml_gallocr_free(gallocr);
        if (!use_repack) {
            ggml_free(ctx);
        }
    };

    int bad = quant_bad;
    for (int is_mmid = 0; is_mmid <= 1; is_mmid++) {
        std::vector<float> a, b;
        run(true,  is_mmid, false, a);
        run(false, is_mmid, false, b);
        bad += compare(is_mmid ? "mul_mat_id repack vs vec_dot" : "mul_mat repack vs vec_dot",
                       a.data(), b.data(), (int64_t) a.size(), 1e-5).n_bad;
        if (test_prequantized && !is_mmid && ntok % 4 == 0) {
            std::vector<float> direct(ne01 * ntok);
            const size_t weight_row_size = ggml_row_size(type, ne00);
            for (int64_t col = 0; col < ne01; col += 8) {
                ggml_gemm_mxfp4_8x8_q8_0(ne00, direct.data() + col, ne01,
                                          (const char *) rw_mm.t->data + col * weight_row_size,
                                          xq4.data(), ntok, 8);
            }
            bad += compare("mul_mat direct kernel vs vec_dot",
                           direct.data(), b.data(), (int64_t) b.size(), 1e-5).n_bad;
        }
    }
    if (test_prequantized) {
        std::vector<float> f32, q8;
        run(true, true, false, f32);
        run(true, true, true, q8);
        bad += compare("mul_mat_id f32 vs prequantized q8_0",
                       q8.data(), f32.data(), (int64_t) f32.size(), 0.0).n_bad;
    }

    free_repacked(rw_mm);
    free_repacked(rw_mmid);
    return bad;
}

// ---------------------------------------------------------------------------
// micro benchmark
// ---------------------------------------------------------------------------

static double now_s() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

template <typename BLOCK>
static void bench(const char * name, ggml_type type,
                  void (*fill)(BLOCK *, int64_t),
                  void (*vec_dot)(int, float *, size_t, const void *, size_t, const void *, size_t, int),
                  void (*gemv)(int, float *, size_t, const void *, const void *, int, int),
                  void (*gemm)(int, float *, size_t, const void *, const void *, int, int)) {
    const int64_t ne00 = 4096;
    const int64_t nb   = ne00 / QK_K;
    const int64_t nc   = 2048;
    const int     nr   = 8;
    const int     reps = 30;

    std::vector<BLOCK> plain(nc * nb);
    fill(plain.data(), (int64_t) plain.size());

    repacked_weights rw = repack_via_buffer<BLOCK>(type, plain.data(), ne00, nc);

    std::vector<float> xf(nr * ne00);
    for (auto & v : xf) v = frand();

    std::vector<block_q8_K> y(nb);
    quantize_row_q8_K(xf.data(), y.data(), ne00);

    std::vector<block_q8_Kx4> y4((nr / 4) * nb);
    for (int yy = 0; yy < nr / 4; yy++) {
        ggml_quantize_mat_q8_K_4x8(xf.data() + (int64_t) yy * 4 * ne00, y4.data() + (int64_t) yy * nb, ne00);
    }

    std::vector<float> out(std::max(nc, nr * nc));
    volatile float sink = 0;

    // vec_dot, 1 row
    double t0 = now_s();
    for (int r = 0; r < reps; r++) {
        for (int64_t j = 0; j < nc; j++) {
            vec_dot(ne00, out.data(), 0, plain.data() + j * nb, 0, y.data(), 0, 1);
        }
    }
    const double t_vec_dot = (now_s() - t0) / reps;

    // repack gemv, 1 row
    t0 = now_s();
    for (int r = 0; r < reps; r++) {
        gemv(ne00, out.data(), 0, rw.t->data, y.data(), 1, nc);
    }
    const double t_gemv = (now_s() - t0) / reps;

    // vec_dot x nr rows
    t0 = now_s();
    for (int r = 0; r < reps; r++) {
        for (int m = 0; m < nr; m++) {
            for (int64_t j = 0; j < nc; j++) {
                vec_dot(ne00, out.data(), 0, plain.data() + j * nb, 0, y.data(), 0, 1);
            }
        }
    }
    const double t_vec_dot8 = (now_s() - t0) / reps;

    // repack gemv x nr rows
    t0 = now_s();
    for (int r = 0; r < reps; r++) {
        for (int m = 0; m < nr; m++) {
            gemv(ne00, out.data(), 0, rw.t->data, y.data(), 1, nc);
        }
    }
    const double t_gemv8 = (now_s() - t0) / reps;

    // repack gemm, nr rows
    t0 = now_s();
    for (int r = 0; r < reps; r++) {
        gemm(ne00, out.data(), nc, rw.t->data, y4.data(), nr, nc);
    }
    const double t_gemm = (now_s() - t0) / reps;

    sink += out[0];
    (void) sink;

    printf("  %s bench (ne00=%lld nc=%lld nr=%d)\n", name, (long long) ne00, (long long) nc, nr);
    printf("    1 row : vec_dot %8.3f ms | repack gemv %8.3f ms | speedup %5.2fx\n",
           t_vec_dot * 1e3, t_gemv * 1e3, t_vec_dot / t_gemv);
    printf("    %d rows: vec_dot %8.3f ms | repack gemv %8.3f ms | repack gemm %8.3f ms | gemm vs vec_dot %5.2fx | gemm vs gemv %5.2fx\n",
           nr, t_vec_dot8 * 1e3, t_gemv8 * 1e3, t_gemm * 1e3, t_vec_dot8 / t_gemm, t_gemv8 / t_gemm);

    free_repacked(rw);
}

template <typename BLOCK>
static void bench_mmid_graph(const char * name, ggml_type type, int qk,
                             void (*fill)(BLOCK *, int64_t), ggml_backend_t backend) {
    const int64_t ne00 = 2048;
    const int64_t ne01 = 2048;
    const int64_t nexp = 32;
    const int64_t nids = 8;
    const int64_t ntok = 512;
    const int n_warmup = 2;
    const int n_iter = 7;

    std::vector<BLOCK> plain(ne01*nexp*(ne00/qk));
    fill(plain.data(), (int64_t) plain.size());
    repacked_weights rw = repack_via_buffer<BLOCK>(type, plain.data(), ne00, ne01, nexp);

    std::vector<float> xf(ne00*ntok);
    for (auto & v : xf) {
        v = frand();
    }
    std::vector<int32_t> ids(nids*ntok);
    for (int64_t t = 0; t < ntok; ++t) {
        for (int64_t id = 0; id < nids; ++id) {
            ids[t*nids + id] = (int32_t) ((t*nids + id) % nexp);
        }
    }

    ggml_context * ctx = rw.ctx;
    ggml_tensor * src0 = rw.t;
    ggml_tensor * src1 = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, ne00, 1, ntok);
    ggml_tensor * idst = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, nids, ntok);
    ggml_tensor * out = ggml_mul_mat_id(ctx, src0, src1, idst);
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);
    ggml_gallocr_t gallocr = ggml_gallocr_new(ggml_backend_cpu_buffer_type());
    GGML_ASSERT(gallocr != nullptr && ggml_gallocr_alloc_graph(gallocr, gf));
    ggml_backend_tensor_set(src1, xf.data(), 0, xf.size()*sizeof(float));
    ggml_backend_tensor_set(idst, ids.data(), 0, ids.size()*sizeof(int32_t));
    for (int i = 0; i < n_warmup; ++i) {
        GGML_ASSERT(ggml_backend_graph_compute(backend, gf) == GGML_STATUS_SUCCESS);
    }

    std::vector<double> samples;
    samples.reserve(n_iter);
    for (int i = 0; i < n_iter; ++i) {
        const double t0 = now_s();
        GGML_ASSERT(ggml_backend_graph_compute(backend, gf) == GGML_STATUS_SUCCESS);
        samples.push_back((now_s() - t0)*1e3);
    }
    std::sort(samples.begin(), samples.end());
    printf("  %s MUL_MAT_ID graph: k=%lld m=%lld experts=%lld topk=%lld tokens=%lld median=%.3f ms min=%.3f ms\n",
           name, (long long) ne00, (long long) ne01, (long long) nexp, (long long) nids,
           (long long) ntok, samples[samples.size()/2], samples.front());

    ggml_gallocr_free(gallocr);
    free_repacked(rw);
}

// ---------------------------------------------------------------------------

int main(int argc, char ** argv) {
    const bool do_bench       = argc > 1 && std::string(argv[1]) == "bench";
    const bool do_numa        = argc > 1 && std::string(argv[1]) == "numa";
    const bool do_bench_mmid  = argc > 1 && std::string(argv[1]) == "bench-mmid";
    const bool do_model_shape = argc > 1 && std::string(argv[1]) == "model-shape";

    if (!do_bench_mmid &&
            (!ggml_cpu_has_avx512() || !ggml_cpu_has_avx512_vnni() || !ggml_cpu_has_avx512_vbmi())) {
        printf("test-repack-iq: SKIPPED (needs AVX512/VNNI/VBMI)\n");
        return 0;
    }

#if defined(_WIN32)
    _putenv_s("GGML_REPACK_Q2_K", "1");
#else
    setenv("GGML_REPACK_Q2_K", "1", 1);
#endif

    if (do_numa || do_model_shape) {
#if defined(_WIN32)
        _putenv_s("GGML_NUMA_EP", "1");
#else
        setenv("GGML_NUMA_EP", "1", 1);
#endif
        ggml_numa_init(GGML_NUMA_STRATEGY_DISTRIBUTE);
    }

    if (do_bench) {
        printf("== IQ2_XS ==\n");
        bench<block_iq2_xs>("iq2_xs", GGML_TYPE_IQ2_XS, fill_random_iq2_xs, ggml_vec_dot_iq2_xs_q8_K,
                            ggml_gemv_iq2_xs_8x8_q8_K, ggml_gemm_iq2_xs_8x8_q8_K);
        printf("== IQ3_XXS ==\n");
        bench<block_iq3_xxs>("iq3_xxs", GGML_TYPE_IQ3_XXS, fill_random_iq3_xxs, ggml_vec_dot_iq3_xxs_q8_K,
                             ggml_gemv_iq3_xxs_8x8_q8_K, ggml_gemm_iq3_xxs_8x8_q8_K);
        return 0;
    }

    if (do_bench_mmid) {
        if (getenv("GGML_NUMA_EP") != nullptr) {
            ggml_numa_init(GGML_NUMA_STRATEGY_DISTRIBUTE);
        }
        ggml_backend_register(ggml_backend_cpu_reg());
        ggml_backend_t backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
        if (backend == nullptr) {
            fprintf(stderr, "failed to init CPU backend\n");
            return 1;
        }
        const int n_threads = argc > 2 ? atoi(argv[2]) : 72;
        ggml_backend_cpu_set_n_threads(backend, n_threads);
        bench_mmid_graph<block_mxfp4>("mxfp4", GGML_TYPE_MXFP4, QK_MXFP4, fill_random_mxfp4, backend);
        ggml_backend_free(backend);
        return 0;
    }

    int bad = 0;

    printf("== IQ2_XS ==\n");
    bad += test_gemv<block_iq2_xs>("iq2_xs", GGML_TYPE_IQ2_XS, fill_random_iq2_xs, ggml_vec_dot_iq2_xs_q8_K,
                                   ggml_gemv_iq2_xs_8x8_q8_K, ggml_gemv_iq2_xs_8x8_q8_K_generic);
    bad += test_gemm<block_iq2_xs>("iq2_xs", GGML_TYPE_IQ2_XS, fill_random_iq2_xs, ggml_vec_dot_iq2_xs_q8_K,
                                   ggml_gemm_iq2_xs_8x8_q8_K, ggml_gemm_iq2_xs_8x8_q8_K_generic);

    printf("== IQ3_XXS ==\n");
    bad += test_gemv<block_iq3_xxs>("iq3_xxs", GGML_TYPE_IQ3_XXS, fill_random_iq3_xxs, ggml_vec_dot_iq3_xxs_q8_K,
                                    ggml_gemv_iq3_xxs_8x8_q8_K, ggml_gemv_iq3_xxs_8x8_q8_K_generic);
    bad += test_gemm<block_iq3_xxs>("iq3_xxs", GGML_TYPE_IQ3_XXS, fill_random_iq3_xxs, ggml_vec_dot_iq3_xxs_q8_K,
                                    ggml_gemm_iq3_xxs_8x8_q8_K, ggml_gemm_iq3_xxs_8x8_q8_K_generic);

    ggml_backend_register(ggml_backend_cpu_reg());
    ggml_backend_t backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    if (!backend) {
        fprintf(stderr, "failed to init CPU backend\n");
        return 1;
    }
    const char * test_threads = getenv("TEST_REPACK_THREADS");
    ggml_backend_cpu_set_n_threads(backend, test_threads ? std::max(1, atoi(test_threads)) : 8);

    if (do_model_shape) {
        // Keep the expert count bounded so this remains safe on development
        // machines, while matching the model's k/m/top-k and prefill shape.
        bad += test_graph<block_mxfp4>("mxfp4 model-shape", GGML_TYPE_MXFP4,
                fill_random_mxfp4, backend, 512, true, 4096, 2048, 8, 6);
        ggml_backend_free(backend);
        printf(bad == 0 ? "ALL OK\n" : "FAILURES: %d\n", bad);
        return bad == 0 ? 0 : 1;
    }

    printf("== graph tests ==\n");
    for (int64_t ntok : { 8, 33 }) {
        bad += test_graph<block_iq2_xs>("iq2_xs", GGML_TYPE_IQ2_XS, fill_random_iq2_xs, backend, ntok);
        bad += test_graph<block_iq3_xxs>("iq3_xxs", GGML_TYPE_IQ3_XXS, fill_random_iq3_xxs, backend, ntok);
        bad += test_graph<block_q2_K>("q2_K", GGML_TYPE_Q2_K, fill_random_q2_K, backend, ntok);
        bad += test_graph<block_mxfp4>("mxfp4", GGML_TYPE_MXFP4, fill_random_mxfp4, backend, ntok, true);
    }

    ggml_backend_free(backend);

    printf(bad == 0 ? "ALL OK\n" : "FAILURES: %d\n", bad);
    return bad == 0 ? 0 : 1;
}
