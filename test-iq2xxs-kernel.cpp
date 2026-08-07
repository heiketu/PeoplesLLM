// Correctness + microbenchmark for the new IQ2_XXS 8x8 repack kernels.
//
// Compares, on random-but-valid quantized data:
//   - ggml_gemv/gemm_iq2_xxs_8x8_q8_K (AVX512) vs their _generic twins (bitexact expected)
//   - gemv/gemm output vs row-wise ggml_vec_dot_iq2_xxs_q8_K_generic (end-to-end layout check)
//   - ggml_vec_dot_iq2_xxs_q8_K (existing AVX512 vec_dot) vs its _generic twin
// and times vec_dot generic/SIMD vs gemv generic/AVX512 per output row.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#include "ggml.h"
#include "ggml-common.h"
#include "repack.h"
#include "quants.h"
#include "ggml-cpu.h"

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e3 + ts.tv_nsec / 1e6;
}

static uint64_t rng_state = 0x123456789abcdef0ull;
static uint64_t rng_u64(void) {
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 7;
    rng_state ^= rng_state << 17;
    return rng_state;
}

// interleave 8 consecutive rows of block_iq2_xxs into one block_iq2_xxsx8
// (same 8-byte-chunk interleave as make_block_iq2_xxsx8 in repack.cpp)
static void repack8(const block_iq2_xxs * in, block_iq2_xxsx8 * out) {
    for (int i = 0; i < 8; i++) {
        out->d[i] = in[i].d;
    }
    const int end = 8 * (QK_K/4) / 8; // 64 chunks of 8 bytes
    for (int i = 0; i < end; ++i) {
        int src_id     = i % 8;
        int src_offset = (i / 8) * 8;
        int dst_offset = i * 8;
        memcpy((uint8_t *) out->qs + dst_offset, (const uint8_t *) in[src_id].qs + src_offset, 8);
    }
}

int main(void) {
    ggml_cpu_init(); // fp16<->fp32 lookup tables

    const int n  = 2048;              // row length (multiple of QK_K)
    const int nb = n / QK_K;
    const int nc = 64;                // weight rows (multiple of 8)
    const int nr = 8;                 // activation rows for gemm (multiple of 4)

    block_iq2_xxs   * x   = (block_iq2_xxs *) aligned_alloc(64, (size_t) nc * nb * sizeof(block_iq2_xxs));
    block_iq2_xxsx8 * xr  = (block_iq2_xxsx8 *) aligned_alloc(64, (size_t) (nc / 8) * nb * sizeof(block_iq2_xxsx8));
    block_q8_K      * y   = (block_q8_K *) aligned_alloc(64, (size_t) nb * sizeof(block_q8_K));
    block_q8_K      * yr  = (block_q8_K *) aligned_alloc(64, (size_t) nr * nb * sizeof(block_q8_K));
    block_q8_Kx4    * y4  = (block_q8_Kx4 *) aligned_alloc(64, (size_t) (nr / 4) * nb * sizeof(block_q8_Kx4));

    // random weights: every bit pattern is a valid iq2_xxs block (8-bit grid indices,
    // 7-bit sign fields, 4-bit scales); d must be a sane fp16
    for (int64_t i = 0; i < (int64_t) nc * nb; i++) {
        uint64_t * p = (uint64_t *) x[i].qs;
        for (size_t k = 0; k < sizeof(x[i].qs) / 8; k++) {
            p[k] = rng_u64();
        }
        x[i].d = ggml_fp32_to_fp16(0.01f * (float) ((int) (rng_u64() % 200) - 100));
    }
    for (int g = 0; g < nc / 8; g++) {
        for (int b = 0; b < nb; b++) {
            // 8 consecutive ROWS, same block index (matches repack_iq2_xxs_to_iq2_xxs_8_bl)
            block_iq2_xxs tmp[8];
            for (int i = 0; i < 8; i++) {
                tmp[i] = x[(size_t) (g * 8 + i) * nb + b];
            }
            repack8(tmp, &xr[(size_t) g * nb + b]);
        }
    }

    // random activations quantized to q8_K
    float * f = (float *) malloc(n * sizeof(float));
    for (int r = 0; r < nr; r++) {
        for (int i = 0; i < n; i++) {
            f[i] = 0.5f * ((float) (rng_u64() % 2000) / 1000.0f - 1.0f);
        }
        quantize_row_q8_K(f, &yr[(size_t) r * nb], n);
    }
    memcpy(y, yr, (size_t) nb * sizeof(block_q8_K));

    // build q8_Kx4 layout from the per-row q8_K rows:
    // d[m] per row; qs entry t (32 entries of 8 elems) of row m at offset t*32 + m*8
    for (int yy = 0; yy < nr / 4; yy++) {
        for (int b = 0; b < nb; b++) {
            block_q8_Kx4 * dst = &y4[(size_t) yy * nb + b];
            for (int m = 0; m < 4; m++) {
                const block_q8_K * src = &yr[(size_t) (yy * 4 + m) * nb + b];
                dst->d[m] = src->d;
                for (int t = 0; t < QK_K/8; t++) {
                    memcpy(dst->qs + t * 32 + m * 8, src->qs + t * 8, 8);
                }
            }
            memset(dst->bsums, 0, sizeof(dst->bsums)); // unused by the iq2_xxs kernels
        }
    }

    // ground truth: row-wise generic vec_dot
    float * ref    = (float *) malloc(sizeof(float) * nc);
    float * ref_gm = (float *) malloc(sizeof(float) * nr * nc);
    for (int c = 0; c < nc; c++) {
        ggml_vec_dot_iq2_xxs_q8_K_generic(n, &ref[c], 0, &x[(size_t) c * nb], 0, y, 0, 1);
    }
    for (int m = 0; m < nr; m++) {
        for (int c = 0; c < nc; c++) {
            ggml_vec_dot_iq2_xxs_q8_K_generic(n, &ref_gm[(size_t) m * nc + c], 0, &x[(size_t) c * nb], 0, &yr[(size_t) m * nb], 0, 1);
        }
    }

    int fails = 0;
    #define CHECK(name, got, want, count) do { \
        double maxrel = 0; int nnan = 0; \
        for (int i = 0; i < (count); i++) { \
            double a = (got)[i], b = (want)[i]; \
            if (!isfinite(a) || !isfinite(b)) { nnan++; continue; } \
            double rel = fabs(a - b) / (fabs(b) + 1e-6); \
            if (rel > maxrel) maxrel = rel; \
        } \
        printf("%-40s maxrel=%.3e nan=%d %s\n", name, maxrel, nnan, (maxrel < 1e-4 && nnan == 0) ? "OK" : "FAIL"); \
        if (maxrel >= 1e-4 || nnan != 0) fails++; \
    } while (0)

    // 1. existing SIMD vec_dot vs generic vec_dot
    float * vdot = (float *) malloc(sizeof(float) * nc);
    for (int c = 0; c < nc; c++) {
        ggml_vec_dot_iq2_xxs_q8_K(n, &vdot[c], 0, &x[(size_t) c * nb], 0, y, 0, 1);
    }
    CHECK("vec_dot avx512 vs generic", vdot, ref, nc);

    // 2. gemv generic vs vec_dot reference (validates repack layout + semantics)
    float * gv_g = (float *) malloc(sizeof(float) * nc);
    ggml_gemv_iq2_xxs_8x8_q8_K_generic(n, gv_g, nc, xr, y, 1, nc);
    CHECK("gemv generic vs vec_dot ref", gv_g, ref, nc);

    // 3. gemv AVX512 vs generic
    float * gv_a = (float *) malloc(sizeof(float) * nc);
    ggml_gemv_iq2_xxs_8x8_q8_K(n, gv_a, nc, xr, y, 1, nc);
    CHECK("gemv avx512 vs generic", gv_a, gv_g, nc);

    // 4. gemm generic vs vec_dot reference
    float * gm_g = (float *) malloc(sizeof(float) * nr * nc);
    ggml_gemm_iq2_xxs_8x8_q8_K_generic(n, gm_g, nc, xr, y4, nr, nc);
    CHECK("gemm generic vs vec_dot ref", gm_g, ref_gm, nr * nc);

    // 5. gemm AVX512 vs generic
    float * gm_a = (float *) malloc(sizeof(float) * nr * nc);
    ggml_gemm_iq2_xxs_8x8_q8_K(n, gm_a, nc, xr, y4, nr, nc);
    CHECK("gemm avx512 vs generic", gm_a, gm_g, nr * nc);

    // ---------------- microbenchmark ----------------
    const int rep = 2000;
    volatile float sink = 0;
    double t0;

    t0 = now_ms();
    for (int r = 0; r < rep; r++) {
        for (int c = 0; c < nc; c++) {
            ggml_vec_dot_iq2_xxs_q8_K_generic(n, (float *) &sink, 0, &x[(size_t) c * nb], 0, y, 0, 1);
        }
    }
    double t_vd_g = (now_ms() - t0) * 1e6 / ((double) rep * nc); // us per row

    t0 = now_ms();
    for (int r = 0; r < rep; r++) {
        for (int c = 0; c < nc; c++) {
            ggml_vec_dot_iq2_xxs_q8_K(n, (float *) &sink, 0, &x[(size_t) c * nb], 0, y, 0, 1);
        }
    }
    double t_vd_a = (now_ms() - t0) * 1e6 / ((double) rep * nc);

    const int rep2 = 200;
    t0 = now_ms();
    for (int r = 0; r < rep2; r++) {
        ggml_gemv_iq2_xxs_8x8_q8_K_generic(n, gv_g, nc, xr, y, 1, nc);
    }
    double t_gv_g = (now_ms() - t0) * 1e6 / ((double) rep2 * nc);

    t0 = now_ms();
    for (int r = 0; r < rep2; r++) {
        ggml_gemv_iq2_xxs_8x8_q8_K(n, gv_a, nc, xr, y, 1, nc);
    }
    double t_gv_a = (now_ms() - t0) * 1e6 / ((double) rep2 * nc);

    t0 = now_ms();
    for (int r = 0; r < rep2; r++) {
        ggml_gemm_iq2_xxs_8x8_q8_K(n, gm_a, nc, xr, y4, nr, nc);
    }
    double t_gm_a = (now_ms() - t0) * 1e6 / ((double) rep2 * nc * nr);

    printf("\nn=%d nc=%d nr=%d\n", n, nc, nr);
    printf("vec_dot generic : %8.3f ns/row\n", t_vd_g);
    printf("vec_dot avx512  : %8.3f ns/row  (%.2fx vs generic)\n", t_vd_a, t_vd_g / t_vd_a);
    printf("gemv    generic : %8.3f ns/row\n", t_gv_g);
    printf("gemv    avx512  : %8.3f ns/row  (%.2fx vs vec_dot generic, %.2fx vs vec_dot avx512)\n",
           t_gv_a, t_vd_g / t_gv_a, t_vd_a / t_gv_a);
    printf("gemm    avx512  : %8.3f ns/row/act (%.2fx vs vec_dot avx512)\n", t_gm_a, t_vd_a / t_gm_a);
    printf("sink=%f\n", (float) sink);

    free(f); free(ref); free(ref_gm); free(vdot); free(gv_g); free(gv_a); free(gm_g); free(gm_a);
    free(x); free(xr); free(y); free(yr); free(y4);
    return fails ? 1 : 0;
}
