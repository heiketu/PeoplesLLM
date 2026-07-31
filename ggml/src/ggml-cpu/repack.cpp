#define GGML_COMMON_IMPL_CPP
#define GGML_COMMON_DECL_CPP
#include "ggml-common.h"
#include "ggml-backend-impl.h"

#include "ggml-impl.h"
#include "ggml-cpu.h"
#include "ggml-cpu-impl.h"
#include "simd-mappings.h"
#include "traits.h"
#include <atomic>

#include "arch-fallback.h"

// cache line padding for the per-expert atomic row claim counters (NUMA EP work stealing)
#define GGML_EP_CACHE_LINE 64

#include <cmath>
#include <cstring>
#include <cassert>
#include <cstdio>  // for GGML_ASSERT

#include "repack.h"

#if defined(__GNUC__)
#pragma GCC diagnostic ignored "-Woverlength-strings"
#endif

#define UNUSED GGML_UNUSED

static inline int nearest_int(float fval) {
    assert(fabsf(fval) <= 4194303.f);
    float val = fval + 12582912.f;
    int i; memcpy(&i, &val, sizeof(int));
    return (i & 0x007fffff) - 0x00400000;
}

// Functions to create the interleaved data layout formats

// interleave 4 block_q4_0s in blocks of blck_size_interleave
// returns an interleaved block_q4_0x4
// in the interleaved block_q4_0x4, place deltas for 4 block_q4_0 blocks
// first, then interleave quants from 4 block_q4_0s in blocks of blck_size_interleave
//
// - in                  : an array of block_q4_0 pointers
// - blck_size_interleave : the block_q4_0 quants bytes are interleaved in blocks of
//                         blck_size_interleave bytes
// - xor_mask            : the mask to convert the nibbles in block_q4_0 quants bytes
//                         from bias offset form to pure sign form (this saves subtract
//                         operations durin unpacking)
//

extern "C" {

#if defined __riscv_zvfh
void ggml_quantize_mat_q8_0_4x1_generic(const float * GGML_RESTRICT x, void * GGML_RESTRICT vy, int64_t k) {
    assert(QK8_0 == 32);
    assert(k % QK8_0 == 0);
    const int nb = k / QK8_0;

    block_q8_0x4 * GGML_RESTRICT y = (block_q8_0x4 *) vy;

    // scalar
    const int blck_size_interleave = 1;
    float srcv[4][QK8_0];
    float id[4];

    for (int i = 0; i < nb; i++) {
        for (int row_iter = 0; row_iter < 4; row_iter++) {
            float amax = 0.0f; // absolute max

            for (int j = 0; j < QK8_0; j++) {
                srcv[row_iter][j] = x[row_iter * k + i * QK8_0 + j];
                amax = MAX(amax, fabsf(srcv[row_iter][j]));
            }

            const float d = amax / ((1 << 7) - 1);
            id[row_iter] = d ? 1.0f / d : 0.0f;

            y[i].d[row_iter] = GGML_CPU_FP32_TO_FP16(d);
        }

        for (int j = 0; j < QK8_0 * 4; j++) {
            int src_offset = (j / (4 * blck_size_interleave)) * blck_size_interleave;
            int src_id = (j % (4 * blck_size_interleave)) / blck_size_interleave;
            src_offset += (j % blck_size_interleave);

            float x0 = srcv[src_id][src_offset] * id[src_id];
            y[i].qs[j] = roundf(x0);
        }
    }
}

void ggml_quantize_mat_q8_K_4x1_generic(const float * GGML_RESTRICT x, void * GGML_RESTRICT vy, int64_t k) {
    assert(QK_K == 256);
    assert(k % QK_K == 0);
    const int nb = k / QK_K;

    block_q8_Kx4 * GGML_RESTRICT y = (block_q8_Kx4 *) vy;

    const int blck_size_interleave = 1;
    float srcv[4][QK_K];
    float iscale[4];

    for (int i = 0; i < nb; i++) {
        for (int row_iter = 0; row_iter < 4; row_iter++) {
            float amax = 0.0f; // absolute max
            float max = 0;

            for (int j = 0; j < QK_K; j++) {
                srcv[row_iter][j] = x[row_iter * k + i * QK_K + j];
                // Update the maximum value of the corresponding super block
                if(amax < fabsf(srcv[row_iter][j])) {
                    amax = fabsf(srcv[row_iter][j]);
                    max = srcv[row_iter][j];
                }
            }

            iscale[row_iter] = amax ? -127.f/max : 0;
            y[i].d[row_iter] = amax ? 1/iscale[row_iter] : 0;
        }

        for (int j = 0; j < QK_K / 4; j++) {
            y[i].bsums[j] = 0;
        }
        for (int j = 0; j < QK_K * 4; j++) {
            int src_id = j % 4;
            int src_offset = j / 4;
            int index = ((j >> 6) << 2) + (j & 3);

            float x0 = srcv[src_id][src_offset] * iscale[src_id];
            y[i].qs[j] = nearest_int(x0);
            y[i].bsums[index] += y[i].qs[j];
        }
    }
}
#endif

void ggml_quantize_mat_q8_0_4x4_generic(const float * GGML_RESTRICT x, void * GGML_RESTRICT vy, int64_t k) {
    assert(QK8_0 == 32);
    assert(k % QK8_0 == 0);
    const int nb = k / QK8_0;

    block_q8_0x4 * GGML_RESTRICT y = (block_q8_0x4 *) vy;

    // scalar
    const int blck_size_interleave = 4;
    float srcv[4][QK8_0];
    float id[4];

    for (int i = 0; i < nb; i++) {
        for (int row_iter = 0; row_iter < 4; row_iter++) {
            float amax = 0.0f; // absolute max

            for (int j = 0; j < QK8_0; j++) {
                srcv[row_iter][j] = x[row_iter * k + i * QK8_0 + j];
                amax = MAX(amax, fabsf(srcv[row_iter][j]));
            }

            const float d = amax / ((1 << 7) - 1);
            id[row_iter] = d ? 1.0f / d : 0.0f;

            y[i].d[row_iter] = GGML_CPU_FP32_TO_FP16(d);
        }

        for (int j = 0; j < QK8_0 * 4; j++) {
            int src_offset = (j / (4 * blck_size_interleave)) * blck_size_interleave;
            int src_id = (j % (4 * blck_size_interleave)) / blck_size_interleave;
            src_offset += (j % blck_size_interleave);

            float x0 = srcv[src_id][src_offset] * id[src_id];
            y[i].qs[j] = roundf(x0);
        }
    }
}

void ggml_quantize_mat_q8_0_4x8_generic(const float * GGML_RESTRICT x, void * GGML_RESTRICT vy, int64_t k) {
    assert(QK8_0 == 32);
    assert(k % QK8_0 == 0);
    const int nb = k / QK8_0;

    block_q8_0x4 * GGML_RESTRICT y = (block_q8_0x4 *) vy;

    // scalar
    const int blck_size_interleave = 8;
    float srcv[4][QK8_0];
    float id[4];

    for (int i = 0; i < nb; i++) {
        for (int row_iter = 0; row_iter < 4; row_iter++) {
            float amax = 0.0f; // absolute max

            for (int j = 0; j < QK8_0; j++) {
                srcv[row_iter][j] = x[row_iter * k + i * QK8_0 + j];
                amax = MAX(amax, fabsf(srcv[row_iter][j]));
            }

            const float d = amax / ((1 << 7) - 1);
            id[row_iter] = d ? 1.0f / d : 0.0f;

            y[i].d[row_iter] = GGML_CPU_FP32_TO_FP16(d);
        }

        for (int j = 0; j < QK8_0 * 4; j++) {
            int src_offset = (j / (4 * blck_size_interleave)) * blck_size_interleave;
            int src_id = (j % (4 * blck_size_interleave)) / blck_size_interleave;
            src_offset += (j % blck_size_interleave);

            float x0 = srcv[src_id][src_offset] * id[src_id];
            y[i].qs[j] = roundf(x0);
        }
    }
}

void ggml_quantize_mat_q8_K_4x4_generic(const float * GGML_RESTRICT x, void * GGML_RESTRICT vy, int64_t k) {
    assert(QK_K == 256);
    assert(k % QK_K == 0);
    const int nb = k / QK_K;

    block_q8_Kx4 * GGML_RESTRICT y = (block_q8_Kx4 *) vy;

    // scalar
    const int blck_size_interleave = 4;
    float srcv[4][QK_K];
    float iscale[4];

    for (int i = 0; i < nb; i++) {
        for (int row_iter = 0; row_iter < 4; row_iter++) {
            float amax = 0.0f; // absolute max
            float max = 0;

            for (int j = 0; j < QK_K; j++) {
                srcv[row_iter][j] = x[row_iter * k + i * QK_K + j];
                // Update the maximum value of the corresponding super block
                if(amax < fabsf(srcv[row_iter][j])) {
                    amax = fabsf(srcv[row_iter][j]);
                    max = srcv[row_iter][j];
                }
            }

            iscale[row_iter] = amax ? -127.f/max : 0;

            y[i].d[row_iter] = amax ? 1/iscale[row_iter] : 0;
        }

        for (int j = 0; j < QK_K / 4; j++) {
            y[i].bsums[j] = 0;
        }

        // Quants values are interleaved in sequence of four bytes from corresponding super blocks
        // Bsums values are interleaved in sequence of four bsums from each super block taken for interleaving
        // i.e first four bsums from the first super block, followed by first four bsums from second super block and so on
        for (int j = 0; j < QK_K * 4; j++) {
            int src_offset = (j / (4 * blck_size_interleave)) * blck_size_interleave;
            int src_id     = (j % (4 * blck_size_interleave)) / blck_size_interleave;
            src_offset += (j % blck_size_interleave);
            int index = (((j & 15) >> 2) << 2) + ((j >> 8) << 4) + ((j >> 6) & 3);

            float x0 = srcv[src_id][src_offset] * iscale[src_id];
            y[i].qs[j] = nearest_int(x0);
            y[i].bsums[index] += y[i].qs[j];
        }
    }
}

void ggml_quantize_mat_q8_K_4x8_generic(const float * GGML_RESTRICT x, void * GGML_RESTRICT vy, int64_t k) {
    assert(QK_K == 256);
    assert(k % QK_K == 0);
    const int nb = k / QK_K;

    block_q8_Kx4 * GGML_RESTRICT y = (block_q8_Kx4 *) vy;

    // scalar
    const int blck_size_interleave = 8;
    float srcv[4][QK_K];
    float iscale[4];

    for (int i = 0; i < nb; i++) {
        for (int row_iter = 0; row_iter < 4; row_iter++) {
            float amax = 0.0f; // absolute max
            float max = 0;

            for (int j = 0; j < QK_K; j++) {
                srcv[row_iter][j] = x[row_iter * k + i * QK_K + j];
                // Update the maximum value of the corresponding super block
                if(amax < fabsf(srcv[row_iter][j])) {
                    amax = fabsf(srcv[row_iter][j]);
                    max = srcv[row_iter][j];
                }
            }

            iscale[row_iter] = amax ? -127.f/max : 0;

            y[i].d[row_iter] = amax ? 1/iscale[row_iter] : 0;
        }

        for (int j = 0; j < QK_K / 4; j++) {
            y[i].bsums[j] = 0;
        }

        // Quants values are interleaved in sequence of eight bytes from corresponding super blocks
        // Bsums values are interleaved in sequence of four bsums from each super block taken for interleaving
        // i.e first four bsums from the first super block, followed by first four bsums from second super block and so on
        for (int j = 0; j < QK_K * 4; j++) {
            int src_offset = (j / (4 * blck_size_interleave)) * blck_size_interleave;
            int src_id     = (j % (4 * blck_size_interleave)) / blck_size_interleave;
            src_offset += (j % blck_size_interleave);
            int index = (((j & 31) >> 3) << 2) + ((j >> 8) << 4) + ((j >> 6) & 3);

            float x0 = srcv[src_id][src_offset] * iscale[src_id];
            y[i].qs[j] = nearest_int(x0);
            y[i].bsums[index] += y[i].qs[j];
        }
    }
}

} // extern "C"

template <int64_t INTER_SIZE, ggml_type PARAM_TYPE>
void ggml_quantize_mat_t(const float * GGML_RESTRICT x, void * GGML_RESTRICT vy, int64_t nrow, int64_t n_per_row);

template <> void ggml_quantize_mat_t<4, GGML_TYPE_Q8_0>(const float * GGML_RESTRICT x, void * GGML_RESTRICT vy, int64_t nrow, int64_t n_per_row) {
    assert(nrow == 4);
    UNUSED(nrow);
    ggml_quantize_mat_q8_0_4x4(x, vy, n_per_row);
}

template <> void ggml_quantize_mat_t<8, GGML_TYPE_Q8_0>(const float * GGML_RESTRICT x, void * GGML_RESTRICT vy, int64_t nrow, int64_t n_per_row) {
    assert(nrow == 4);
    UNUSED(nrow);
    ggml_quantize_mat_q8_0_4x8(x, vy, n_per_row);
}

template <> void ggml_quantize_mat_t<4, GGML_TYPE_Q8_K>(const float * GGML_RESTRICT x, void * GGML_RESTRICT vy, int64_t nrow, int64_t n_per_row) {
    assert(nrow == 4);
    UNUSED(nrow);
    ggml_quantize_mat_q8_K_4x4(x, vy, n_per_row);
}

template <> void ggml_quantize_mat_t<8, GGML_TYPE_Q8_K>(const float * GGML_RESTRICT x, void * GGML_RESTRICT vy, int64_t nrow, int64_t n_per_row) {
    assert(nrow == 4);
    UNUSED(nrow);
    ggml_quantize_mat_q8_K_4x8(x, vy, n_per_row);
}

#if defined __riscv_zvfh
template <> void ggml_quantize_mat_t<1, GGML_TYPE_Q8_0>(const float * GGML_RESTRICT x, void * GGML_RESTRICT vy, int64_t nrow, int64_t n_per_row) {
    assert(nrow == 4);
    UNUSED(nrow);
    ggml_quantize_mat_q8_0_4x1(x, vy, n_per_row);
}

template <> void ggml_quantize_mat_t<1, GGML_TYPE_Q8_K>(const float * GGML_RESTRICT x, void * GGML_RESTRICT vy, int64_t nrow, int64_t n_per_row) {
    assert(nrow == 4);
    UNUSED(nrow);
    ggml_quantize_mat_q8_K_4x1(x, vy, n_per_row);
}
#endif

template <int M, int N>
static void ggml_gemv_q6_K_NxM_q8_K_generic_impl(int                        n,
                                                 float * GGML_RESTRICT      s,
                                                 size_t                     bs,
                                                 const void * GGML_RESTRICT vx,
                                                 const void * GGML_RESTRICT vy,
                                                 int                        nr,
                                                 int                        nc) {
    constexpr int blocklen          = M;
    constexpr int ncols_interleaved = N;
    const int     qk                = QK_K;
    const int     nb                = n / qk;
    const int     blocks_per_half   = 64 / blocklen;

    assert(n % qk == 0);
    assert(nc % ncols_interleaved == 0);

    UNUSED(bs);
    UNUSED(nr);

    float sumf[8];

    const block_q8_K * a_ptr = (const block_q8_K *) vy;
    for (int x = 0; x < nc / ncols_interleaved; x++) {
        const block_q6_Kx8 * b_ptr = (const block_q6_Kx8 *) vx + (x * nb);

        for (int j = 0; j < ncols_interleaved; j++) {
            sumf[j] = 0.0f;
        }

        for (int l = 0; l < nb; l++) {
            for (int k = 0; k < (qk / (2 * blocklen)); k++) {
                const int base_l = (k / blocks_per_half) * 128 + (k % blocks_per_half) * blocklen;
                const int base_h = base_l + 64;

                const int scale_idx_l = base_l / 16;
                const int scale_idx_h = base_h / 16;

                const int qh_shift_l = ((base_l % 128) / 32) * 2;
                const int qh_shift_h = ((base_h % 128) / 32) * 2;

                const int qh_half_l = (base_l / 128) * 32;
                const int qh_half_h = (base_h / 128) * 32;

                for (int j = 0; j < ncols_interleaved; j++) {
                    const int8_t scale_l = b_ptr[l].scales[scale_idx_l * ncols_interleaved + j];
                    const int8_t scale_h = b_ptr[l].scales[scale_idx_h * ncols_interleaved + j];

                    int sumi_l = 0;
                    int sumi_h = 0;

                    for (int i = 0; i < blocklen; i++) {
                        const int ql_pos = k * ncols_interleaved * blocklen + j * blocklen + i;
                        const int l_4    = b_ptr[l].ql[ql_pos] & 0xF;
                        const int hi_4   = (b_ptr[l].ql[ql_pos] >> 4) & 0xF;

                        const int qh_idx_l    = qh_half_l + ((base_l + i) % 32);
                        const int qh_chunk_l  = qh_idx_l / blocklen;
                        const int qh_pos_l    = qh_idx_l % blocklen;
                        const int qh_offset_l = qh_chunk_l * (blocklen * ncols_interleaved) + j * blocklen + qh_pos_l;
                        const int hi_2_l      = (b_ptr[l].qh[qh_offset_l] >> qh_shift_l) & 0x3;

                        const int qh_idx_h    = qh_half_h + ((base_h + i) % 32);
                        const int qh_chunk_h  = qh_idx_h / blocklen;
                        const int qh_pos_h    = qh_idx_h % blocklen;
                        const int qh_offset_h = qh_chunk_h * (blocklen * ncols_interleaved) + j * blocklen + qh_pos_h;
                        const int hi_2_h      = (b_ptr[l].qh[qh_offset_h] >> qh_shift_h) & 0x3;

                        const int q_l = ((hi_2_l << 4) | l_4) - 32;
                        const int q_h = ((hi_2_h << 4) | hi_4) - 32;

                        const int8_t a_l = a_ptr[l].qs[base_l + i];
                        const int8_t a_h = a_ptr[l].qs[base_h + i];

                        sumi_l += q_l * a_l;
                        sumi_h += q_h * a_h;
                    }

                    sumf[j] +=
                        (sumi_l * scale_l + sumi_h * scale_h) * GGML_CPU_FP16_TO_FP32(b_ptr[l].d[j]) * a_ptr[l].d;
                }
            }
        }

        for (int j = 0; j < ncols_interleaved; j++) {
            s[x * ncols_interleaved + j] = sumf[j];
        }
    }
}

template <int M, int N>
static void ggml_gemm_q6_K_NxM_q8_K_generic_impl(int                        n,
                                                 float * GGML_RESTRICT      s,
                                                 size_t                     bs,
                                                 const void * GGML_RESTRICT vx,
                                                 const void * GGML_RESTRICT vy,
                                                 int                        nr,
                                                 int                        nc) {
    constexpr int blocklen          = M;
    constexpr int ncols_interleaved = N;
    const int     qk                = QK_K;
    const int     nb                = n / qk;
    const int     blocks_per_half   = 64 / blocklen;
    const int     q8_half_stride    = 512;
    const int     q8_low_high_step  = 256;

    assert(n % qk == 0);
    assert(nr % 4 == 0);
    assert(nc % ncols_interleaved == 0);

    UNUSED(bs);

    float sumf[4][8];

    for (int y = 0; y < nr / 4; y++) {
        const block_q8_Kx4 * a_ptr = (const block_q8_Kx4 *) vy + (y * nb);
        for (int x = 0; x < nc / ncols_interleaved; x++) {
            const block_q6_Kx8 * b_ptr = (const block_q6_Kx8 *) vx + (x * nb);

            for (int m = 0; m < 4; m++) {
                for (int j = 0; j < ncols_interleaved; j++) {
                    sumf[m][j] = 0.0f;
                }
            }

            for (int l = 0; l < nb; l++) {
                for (int k = 0; k < (qk / (2 * blocklen)); k++) {
                    const int base_l = (k / blocks_per_half) * 128 + (k % blocks_per_half) * blocklen;
                    const int base_h = base_l + 64;

                    const int scale_idx_l = base_l / 16;
                    const int scale_idx_h = base_h / 16;

                    const int qh_shift_l = ((base_l % 128) / 32) * 2;
                    const int qh_shift_h = ((base_h % 128) / 32) * 2;

                    const int qh_half_l = (base_l / 128) * 32;
                    const int qh_half_h = (base_h / 128) * 32;

                    const int q8_base = (k / blocks_per_half) * q8_half_stride + (k % blocks_per_half) * (blocklen * 4);

                    for (int m = 0; m < 4; m++) {
                        for (int j = 0; j < ncols_interleaved; j++) {
                            const int8_t scale_l = b_ptr[l].scales[scale_idx_l * ncols_interleaved + j];
                            const int8_t scale_h = b_ptr[l].scales[scale_idx_h * ncols_interleaved + j];

                            int sumi_l = 0;
                            int sumi_h = 0;

                            for (int i = 0; i < blocklen; i++) {
                                const int ql_pos = k * ncols_interleaved * blocklen + j * blocklen + i;
                                const int l_4    = b_ptr[l].ql[ql_pos] & 0xF;
                                const int hi_4   = (b_ptr[l].ql[ql_pos] >> 4) & 0xF;

                                const int qh_idx_l   = qh_half_l + ((base_l + i) % 32);
                                const int qh_chunk_l = qh_idx_l / blocklen;
                                const int qh_pos_l   = qh_idx_l % blocklen;
                                const int qh_offset_l =
                                    qh_chunk_l * (blocklen * ncols_interleaved) + j * blocklen + qh_pos_l;
                                const int hi_2_l = (b_ptr[l].qh[qh_offset_l] >> qh_shift_l) & 0x3;

                                const int qh_idx_h   = qh_half_h + ((base_h + i) % 32);
                                const int qh_chunk_h = qh_idx_h / blocklen;
                                const int qh_pos_h   = qh_idx_h % blocklen;
                                const int qh_offset_h =
                                    qh_chunk_h * (blocklen * ncols_interleaved) + j * blocklen + qh_pos_h;
                                const int hi_2_h = (b_ptr[l].qh[qh_offset_h] >> qh_shift_h) & 0x3;

                                const int q_l = ((hi_2_l << 4) | l_4) - 32;
                                const int q_h = ((hi_2_h << 4) | hi_4) - 32;

                                const int8_t q8_l = a_ptr[l].qs[q8_base + m * blocklen + i];
                                const int8_t q8_h = a_ptr[l].qs[q8_base + m * blocklen + i + q8_low_high_step];

                                sumi_l += q_l * q8_l;
                                sumi_h += q_h * q8_h;
                            }

                            sumf[m][j] += (sumi_l * scale_l + sumi_h * scale_h) * GGML_CPU_FP16_TO_FP32(b_ptr[l].d[j]) *
                                          a_ptr[l].d[m];
                        }
                    }
                }
            }

            for (int m = 0; m < 4; m++) {
                for (int j = 0; j < ncols_interleaved; j++) {
                    s[(y * 4 + m) * bs + x * ncols_interleaved + j] = sumf[m][j];
                }
            }
        }
    }
}

template <int M, int N>
static void ggml_gemv_q5_K_NxM_q8_K_generic_impl(int                        n,
                                                 float * GGML_RESTRICT      s,
                                                 size_t                     bs,
                                                 const void * GGML_RESTRICT vx,
                                                 const void * GGML_RESTRICT vy,
                                                 int                        nr,
                                                 int                        nc) {
    constexpr int         blocklen          = M;
    constexpr int         ncols_interleaved = N;
    const int             qk                = QK_K;
    const int             nb                = n / qk;
    static const uint32_t kmask1            = 0x3f3f3f3f;
    static const uint32_t kmask2            = 0x0f0f0f0f;
    static const uint32_t kmask3            = 0x03030303;

    assert(n % qk == 0);
    assert(nc % ncols_interleaved == 0);

    UNUSED(bs);
    UNUSED(nr);

    float    sumf[ncols_interleaved];
    float    sum_minf[ncols_interleaved];
    uint32_t utmp[32];
    int      sumi1;
    int      sumi2;
    int      sumi;

    const block_q8_K * a_ptr = (const block_q8_K *) vy;
    for (int x = 0; x < nc / ncols_interleaved; x++) {
        const block_q5_Kx8 * b_ptr = (const block_q5_Kx8 *) vx + (x * nb);

        for (int j = 0; j < ncols_interleaved; j++) {
            sumf[j]     = 0.0;
            sum_minf[j] = 0.0;
        }
        for (int l = 0; l < nb; l++) {
            for (int sb = 0; sb < 8; sb++) {
                memcpy(utmp + sb * 4, b_ptr[l].scales + sb * K_SCALE_SIZE, K_SCALE_SIZE);
                utmp[sb * 4 + 3]      = ((utmp[sb * 4 + 2] >> 4) & kmask2) | (((utmp[sb * 4 + 1] >> 6) & kmask3) << 4);
                const uint32_t uaux_0 = utmp[sb * 4 + 1] & kmask1;
                utmp[sb * 4 + 1]      = (utmp[sb * 4 + 2] & kmask2) | (((utmp[sb * 4 + 0] >> 6) & kmask3) << 4);
                utmp[sb * 4 + 2]      = uaux_0;
                utmp[sb * 4 + 0] &= kmask1;
            }
            for (int k = 0; k < (qk / (2 * blocklen)); k++) {
                constexpr int scale_stride = 32;
                uint8_t *     scales_0     = (uint8_t *) utmp + (k / (32 / blocklen)) * scale_stride;
                uint8_t *     scales_1     = (uint8_t *) utmp + (k / (32 / blocklen)) * scale_stride + 16;

                const int qh_shift = (k / (32 / blocklen)) * 2;
                for (int j = 0; j < ncols_interleaved; j++) {
                    sumi1 = 0;
                    sumi2 = 0;
                    sumi  = 0;
                    for (int i = 0; i < blocklen; ++i) {
                        const int b_qs_offset = k * ncols_interleaved * blocklen + j * blocklen + i;

                        const int qh_idx      = (k * blocklen + i) % 32;
                        const int qh_chunk    = qh_idx / blocklen;
                        const int qh_pos      = qh_idx % blocklen;
                        const int b_qh_offset = qh_chunk * (blocklen * ncols_interleaved) + j * blocklen + qh_pos;

                        const uint8_t qh_val = b_ptr[l].qh[b_qh_offset];
                        const uint8_t h0     = (qh_val >> qh_shift) & 1;
                        const uint8_t h1     = (qh_val >> (qh_shift + 1)) & 1;

                        const int v0 = (int8_t) ((b_ptr[l].qs[b_qs_offset] & 0xF) | (h0 << 4));
                        const int v1 = (int8_t) ((b_ptr[l].qs[b_qs_offset] >> 4) | (h1 << 4));

                        const int q8_offset = (k / (32 / blocklen)) * 64 + (k % (32 / blocklen)) * blocklen + i;

                        sumi1 = (v0 * a_ptr[l].qs[q8_offset]);
                        sumi2 = (v1 * a_ptr[l].qs[q8_offset + 32]);
                        sumi1 = sumi1 * scales_0[j];
                        sumi2 = sumi2 * scales_1[j];
                        sumi += sumi1 + sumi2;
                    }
                    sumf[j] += sumi * GGML_CPU_FP16_TO_FP32(b_ptr[l].d[j]) * a_ptr[l].d;
                }
            }
            for (int sb = 0; sb < 8; sb++) {
                uint8_t * mins = (uint8_t *) utmp + 8 + sb * 16;
                for (int j = 0; j < ncols_interleaved; j++) {
                    sum_minf[j] += mins[j] * (a_ptr[l].bsums[sb * 2] + a_ptr[l].bsums[sb * 2 + 1]) *
                                   GGML_CPU_FP16_TO_FP32(b_ptr[l].dmin[j]) * a_ptr[l].d;
                }
            }
        }
        for (int j = 0; j < ncols_interleaved; j++) {
            s[x * ncols_interleaved + j] = sumf[j] - sum_minf[j];
        }
    }
}

template <int M, int N>
static void ggml_gemm_q5_K_NxM_q8_K_generic_impl(int                        n,
                                                 float * GGML_RESTRICT      s,
                                                 size_t                     bs,
                                                 const void * GGML_RESTRICT vx,
                                                 const void * GGML_RESTRICT vy,
                                                 int                        nr,
                                                 int                        nc) {
    constexpr int         blocklen          = M;
    constexpr int         ncols_interleaved = N;
    const int             qk                = QK_K;
    const int             nb                = n / qk;
    static const uint32_t kmask1            = 0x3f3f3f3f;
    static const uint32_t kmask2            = 0x0f0f0f0f;
    static const uint32_t kmask3            = 0x03030303;

    assert(n % qk == 0);
    assert(nr % 4 == 0);
    assert(nc % ncols_interleaved == 0);

    float    sumf[4][ncols_interleaved];
    float    sum_minf[4][ncols_interleaved];
    uint32_t utmp[32];
    int      sumi1;
    int      sumi2;
    int      sumi;

    for (int y = 0; y < nr / 4; y++) {
        const block_q8_Kx4 * a_ptr = (const block_q8_Kx4 *) vy + (y * nb);
        for (int x = 0; x < nc / ncols_interleaved; x++) {
            const block_q5_Kx8 * b_ptr = (const block_q5_Kx8 *) vx + (x * nb);
            for (int m = 0; m < 4; m++) {
                for (int j = 0; j < ncols_interleaved; j++) {
                    sumf[m][j]     = 0.0;
                    sum_minf[m][j] = 0.0;
                }
            }
            for (int l = 0; l < nb; l++) {
                for (int sb = 0; sb < 8; sb++) {
                    memcpy(utmp + sb * 4, b_ptr[l].scales + sb * K_SCALE_SIZE, K_SCALE_SIZE);
                    utmp[sb * 4 + 3] = ((utmp[sb * 4 + 2] >> 4) & kmask2) | (((utmp[sb * 4 + 1] >> 6) & kmask3) << 4);
                    const uint32_t uaux_0 = utmp[sb * 4 + 1] & kmask1;
                    utmp[sb * 4 + 1]      = (utmp[sb * 4 + 2] & kmask2) | (((utmp[sb * 4 + 0] >> 6) & kmask3) << 4);
                    utmp[sb * 4 + 2]      = uaux_0;
                    utmp[sb * 4 + 0] &= kmask1;
                }
                for (int k = 0; k < (qk / (2 * blocklen)); k++) {
                    constexpr int scale_stride = 32;
                    uint8_t *     scales_0     = (uint8_t *) utmp + (k / (32 / blocklen)) * scale_stride;
                    uint8_t *     scales_1     = (uint8_t *) utmp + (k / (32 / blocklen)) * scale_stride + 16;

                    const int qh_shift = (k / (32 / blocklen)) * 2;
                    for (int m = 0; m < 4; m++) {
                        for (int j = 0; j < ncols_interleaved; j++) {
                            sumi1 = 0;
                            sumi2 = 0;
                            sumi  = 0;
                            for (int i = 0; i < blocklen; ++i) {
                                const int b_qs_offset = k * ncols_interleaved * blocklen + j * blocklen + i;

                                const int qh_idx   = (k * blocklen + i) % 32;
                                const int qh_chunk = qh_idx / blocklen;
                                const int qh_pos   = qh_idx % blocklen;
                                const int b_qh_offset =
                                    qh_chunk * (blocklen * ncols_interleaved) + j * blocklen + qh_pos;

                                const uint8_t qh_val = b_ptr[l].qh[b_qh_offset];
                                const uint8_t h0     = (qh_val >> qh_shift) & 1;
                                const uint8_t h1     = (qh_val >> (qh_shift + 1)) & 1;

                                const int v0 = (int8_t) ((b_ptr[l].qs[b_qs_offset] & 0xF) | (h0 << 4));
                                const int v1 = (int8_t) ((b_ptr[l].qs[b_qs_offset] >> 4) | (h1 << 4));

                                const int q8_offset = (k / (32 / blocklen)) * 256 +
                                                      (k % (32 / blocklen)) * 4 * blocklen + m * blocklen + i;

                                sumi1 = (v0 * a_ptr[l].qs[q8_offset]);
                                sumi2 = (v1 * a_ptr[l].qs[q8_offset + 128]);
                                sumi1 = sumi1 * scales_0[j];
                                sumi2 = sumi2 * scales_1[j];
                                sumi += sumi1 + sumi2;
                            }
                            sumf[m][j] += sumi * GGML_CPU_FP16_TO_FP32(b_ptr[l].d[j]) * a_ptr[l].d[m];
                        }
                    }
                }
                for (int sb = 0; sb < 8; sb++) {
                    uint8_t * mins = (uint8_t *) utmp + 8 + sb * 16;
                    for (int m = 0; m < 4; m++) {
                        const int16_t * bsums = a_ptr[l].bsums + (sb * 8) + (m * 4) - ((sb % 2) * 6);
                        for (int j = 0; j < ncols_interleaved; j++) {
                            sum_minf[m][j] += mins[j] * (bsums[0] + bsums[1]) *
                                              GGML_CPU_FP16_TO_FP32(b_ptr[l].dmin[j]) * a_ptr[l].d[m];
                        }
                    }
                }
            }
            for (int m = 0; m < 4; m++) {
                for (int j = 0; j < ncols_interleaved; j++) {
                    s[(y * 4 + m) * bs + x * ncols_interleaved + j] = sumf[m][j] - sum_minf[m][j];
                }
            }
        }
    }
}

extern "C" {

void ggml_gemv_q4_0_4x4_q8_0_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc) {
    const int qk = QK8_0;
    const int nb = n / qk;
    const int ncols_interleaved = 4;
    const int blocklen = 4;

    assert(nr == 1);
    assert(n % qk == 0);
    assert(nc % ncols_interleaved == 0);

    UNUSED(s);
    UNUSED(bs);
    UNUSED(vx);
    UNUSED(vy);
    UNUSED(nr);
    UNUSED(nc);
    UNUSED(nb);
    UNUSED(ncols_interleaved);
    UNUSED(blocklen);

    float sumf[4];
    int sumi;

    const block_q8_0 * a_ptr = (const block_q8_0 *) vy;
    for (int x = 0; x < nc / ncols_interleaved; x++) {
        const block_q4_0x4 * b_ptr = (const block_q4_0x4 *) vx + (x * nb);

        for (int j = 0; j < ncols_interleaved; j++) sumf[j] = 0.0;
        for (int l = 0; l < nb; l++) {
            for (int k = 0; k < (qk / (2 * blocklen)); k++) {
                for (int j = 0; j < ncols_interleaved; j++) {
                    sumi = 0;
                    for (int i = 0; i < blocklen; ++i) {
                        const int v0 = (int8_t) (b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] << 4);
                        const int v1 = (int8_t) (b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] & 0xF0);
                        sumi += ((v0 * a_ptr[l].qs[k * blocklen + i]) + (v1 * a_ptr[l].qs[k * blocklen + i + qk / 2])) >> 4;
                    }
                    sumf[j] += sumi * GGML_CPU_FP16_TO_FP32(b_ptr[l].d[j]) * GGML_CPU_FP16_TO_FP32(a_ptr[l].d);
                }
            }
        }
        for (int j = 0; j < ncols_interleaved; j++) s[x * ncols_interleaved + j] = sumf[j];
    }
}

void ggml_gemv_q4_0_4x8_q8_0_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc) {
    const int qk = QK8_0;
    const int nb = n / qk;
    const int ncols_interleaved = 4;
    const int blocklen = 8;

    assert (n % qk == 0);
    assert (nc % ncols_interleaved == 0);

    UNUSED(s);
    UNUSED(bs);
    UNUSED(vx);
    UNUSED(vy);
    UNUSED(nr);
    UNUSED(nc);
    UNUSED(nb);
    UNUSED(ncols_interleaved);
    UNUSED(blocklen);

    float sumf[4];
    int sumi;

    const block_q8_0 * a_ptr = (const block_q8_0 *) vy;
    for (int x = 0; x < nc / ncols_interleaved; x++) {
        const block_q4_0x4 * b_ptr = (const block_q4_0x4 *) vx + (x * nb);

        for (int j = 0; j < ncols_interleaved; j++) sumf[j] = 0.0;
        for (int l = 0; l < nb; l++) {
            for (int k = 0; k < (qk / (2 * blocklen)); k++) {
                for (int j = 0; j < ncols_interleaved; j++) {
                    sumi = 0;
                    for (int i = 0; i < blocklen; ++i) {
                        const int v0 = (int8_t) (b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] << 4);
                        const int v1 = (int8_t) (b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] & 0xF0);
                        sumi += ((v0 * a_ptr[l].qs[k * blocklen + i]) + (v1 * a_ptr[l].qs[k * blocklen + i + qk / 2])) >> 4;
                    }
                    sumf[j] += sumi * GGML_CPU_FP16_TO_FP32(b_ptr[l].d[j]) * GGML_CPU_FP16_TO_FP32(a_ptr[l].d);
                }
            }
        }
        for (int j = 0; j < ncols_interleaved; j++) s[x * ncols_interleaved + j] = sumf[j];
    }
}

void ggml_gemv_q4_0_8x8_q8_0_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc) {
    const int qk = QK8_0;
    const int nb = n / qk;
    const int ncols_interleaved = 8;
    const int blocklen = 8;

    assert (n % qk == 0);
    assert (nc % ncols_interleaved == 0);

    UNUSED(s);
    UNUSED(bs);
    UNUSED(vx);
    UNUSED(vy);
    UNUSED(nr);
    UNUSED(nc);
    UNUSED(nb);
    UNUSED(ncols_interleaved);
    UNUSED(blocklen);

    float sumf[8];
    int sumi;

    const block_q8_0 * a_ptr = (const block_q8_0 *) vy;
    for (int x = 0; x < nc / ncols_interleaved; x++) {
        const block_q4_0x8 * b_ptr = (const block_q4_0x8 *) vx + (x * nb);

        for (int j = 0; j < ncols_interleaved; j++) sumf[j] = 0.0;
        for (int l = 0; l < nb; l++) {
            for (int k = 0; k < (qk / (2 * blocklen)); k++) {
                for (int j = 0; j < ncols_interleaved; j++) {
                    sumi = 0;
                    for (int i = 0; i < blocklen; ++i) {
                        const int v0 = (int8_t) (b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] << 4);
                        const int v1 = (int8_t) (b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] & 0xF0);
                        sumi += ((v0 * a_ptr[l].qs[k * blocklen + i]) + (v1 * a_ptr[l].qs[k * blocklen + i + qk / 2])) >> 4;
                    }
                    sumf[j] += sumi * GGML_CPU_FP16_TO_FP32(b_ptr[l].d[j]) * GGML_CPU_FP16_TO_FP32(a_ptr[l].d);
                }
            }
        }
        for (int j = 0; j < ncols_interleaved; j++) s[x * ncols_interleaved + j] = sumf[j];
    }
}

void ggml_gemv_q4_K_8x4_q8_K_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc) {
    const int qk = QK_K;
    const int nb = n / qk;
    const int ncols_interleaved = 8;
    const int blocklen = 4;
    static const uint32_t kmask1 = 0x3f3f3f3f;
    static const uint32_t kmask2 = 0x0f0f0f0f;
    static const uint32_t kmask3 = 0x03030303;

    assert (n % qk == 0);
    assert (nc % ncols_interleaved == 0);

    UNUSED(bs);
    UNUSED(nr);

    float sumf[8];
    float sum_minf[8];
    uint32_t utmp[32];
    int sumi1;
    int sumi2;
    int sumi;

    const block_q8_K * a_ptr = (const block_q8_K *) vy;
    for (int x = 0; x < nc / ncols_interleaved; x++) {
        const block_q4_Kx8 * b_ptr = (const block_q4_Kx8 *) vx + (x * nb);

        for (int j = 0; j < ncols_interleaved; j++) {
            sumf[j] = 0.0;
            sum_minf[j] = 0.0;
        }
        for (int l = 0; l < nb; l++) {
            for (int sb = 0; sb < 8; sb++) {
                memcpy(utmp + sb * 4, b_ptr[l].scales + sb * 12, 12);
                utmp[sb * 4 + 3] = ((utmp[sb * 4 + 2] >> 4) & kmask2) | (((utmp[sb * 4 + 1] >> 6) & kmask3) << 4);
                const uint32_t uaux_0 = utmp[sb * 4 + 1] & kmask1;
                utmp[sb * 4 + 1] = (utmp[sb * 4 + 2] & kmask2) | (((utmp[sb * 4 + 0] >> 6) & kmask3) << 4);
                utmp[sb * 4 + 2] = uaux_0;
                utmp[sb * 4 + 0] &= kmask1;
            }
            for (int k = 0; k < (qk / (2 * blocklen)); k++) {
                uint8_t * scales_0 = (uint8_t *) utmp + (k / 8) * 32;
                uint8_t * scales_1 = (uint8_t *) utmp + (k / 8) * 32 + 16;
                for (int j = 0; j < ncols_interleaved; j++) {
                    sumi1 = 0;
                    sumi2 = 0;
                    sumi = 0;
                    for (int i = 0; i < blocklen; ++i) {
                        const int v0 = (int8_t) (b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] & 0xF);
                        const int v1 = (int8_t) (b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] >> 4);
                        sumi1 = (v0 * a_ptr[l].qs[(k / 8) * 64 + (k % 8) * blocklen + i]);
                        sumi2 = (v1 * a_ptr[l].qs[(k / 8) * 64 + (k % 8) * blocklen + i + 32]);
                        sumi1 = sumi1 * scales_0[j];
                        sumi2 = sumi2 * scales_1[j];
                        sumi += sumi1 + sumi2;
                    }
                    sumf[j] += sumi * GGML_CPU_FP16_TO_FP32(b_ptr[l].d[j]) * a_ptr[l].d;
                }
            }
            for (int sb = 0; sb < 8; sb++) {
                uint8_t * mins = (uint8_t *) utmp + 8 + sb * 16;
                for (int j = 0; j < ncols_interleaved; j++) {
                    sum_minf[j] += mins[j] * (a_ptr[l].bsums[sb * 2] + a_ptr[l].bsums[sb * 2 + 1]) * GGML_CPU_FP16_TO_FP32(b_ptr[l].dmin[j]) * a_ptr[l].d;
                }
            }
        }
        for (int j = 0; j < ncols_interleaved; j++) {
            s[x * ncols_interleaved + j] = sumf[j] - sum_minf[j];
        }
    }
}

void ggml_gemv_q4_K_8x8_q8_K_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc) {
    const int qk = QK_K;
    const int nb = n / qk;
    const int ncols_interleaved = 8;
    const int blocklen = 8;
    static const uint32_t kmask1 = 0x3f3f3f3f;
    static const uint32_t kmask2 = 0x0f0f0f0f;
    static const uint32_t kmask3 = 0x03030303;

    assert (n % qk == 0);
    assert (nc % ncols_interleaved == 0);

    UNUSED(bs);
    UNUSED(nr);

    float sumf[8];
    float sum_minf[8];
    uint32_t utmp[32];
    int sumi1;
    int sumi2;
    int sumi;

    const block_q8_K * a_ptr = (const block_q8_K *) vy;
    for (int x = 0; x < nc / ncols_interleaved; x++) {
        const block_q4_Kx8 * b_ptr = (const block_q4_Kx8 *) vx + (x * nb);

        for (int j = 0; j < ncols_interleaved; j++) {
            sumf[j] = 0.0;
            sum_minf[j] = 0.0;
        }
        for (int l = 0; l < nb; l++) {
            for (int sb = 0; sb < 8; sb++) {
                memcpy(utmp + sb * 4, b_ptr[l].scales + sb * 12, 12);
                utmp[sb * 4 + 3] = ((utmp[sb * 4 + 2] >> 4) & kmask2) | (((utmp[sb * 4 + 1] >> 6) & kmask3) << 4);
                const uint32_t uaux_0 = utmp[sb * 4 + 1] & kmask1;
                utmp[sb * 4 + 1] = (utmp[sb * 4 + 2] & kmask2) | (((utmp[sb * 4 + 0] >> 6) & kmask3) << 4);
                utmp[sb * 4 + 2] = uaux_0;
                utmp[sb * 4 + 0] &= kmask1;
            }
            for (int k = 0; k < (qk / (2 * blocklen)); k++) {
                uint8_t *scales_0 = (uint8_t*) utmp + (k / 4) * 32;
                uint8_t *scales_1 = (uint8_t*) utmp + (k / 4) * 32 + 16;
                for (int j = 0; j < ncols_interleaved; j++) {
                    sumi1 = 0;
                    sumi2 = 0;
                    sumi = 0;
                    for (int i = 0; i < blocklen; ++i) {
                        const int v0 = (int8_t) (b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] & 0xF);
                        const int v1 = (int8_t) (b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] >> 4);
                        sumi1 = (v0 * a_ptr[l].qs[(k >> 2) * 64 + (k % 4) * blocklen + i]);
                        sumi2 = (v1 * a_ptr[l].qs[(k >> 2) * 64 + (k % 4) * blocklen + i + 32]);
                        sumi1 = sumi1 * scales_0[j];
                        sumi2 = sumi2 * scales_1[j];
                        sumi += sumi1 + sumi2;
                    }
                    sumf[j] += sumi * GGML_CPU_FP16_TO_FP32(b_ptr[l].d[j]) * a_ptr[l].d;
                }
            }
            for (int sb = 0; sb < 8; sb++) {
                uint8_t *mins = (uint8_t*) utmp + 8 + sb * 16;
                for (int j = 0; j < ncols_interleaved; j++) {
                    sum_minf[j] += mins[j] * (a_ptr[l].bsums[sb * 2] + a_ptr[l].bsums[sb * 2 + 1]) * GGML_CPU_FP16_TO_FP32(b_ptr[l].dmin[j]) * a_ptr[l].d;
                }
            }
        }
        for (int j = 0; j < ncols_interleaved; j++) {
            s[x * ncols_interleaved + j] = sumf[j] - sum_minf[j];
        }
    }
}

void ggml_gemv_q2_K_8x8_q8_K_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc) {
    const int qk = QK_K;
    const int nb = n / qk;
    const int ncols_interleaved = 8;
    const int blocklen = 8;

    assert (n % qk == 0);
    assert (nc % ncols_interleaved == 0);

    UNUSED(s);
    UNUSED(bs);
    UNUSED(vx);
    UNUSED(vy);
    UNUSED(nr);
    UNUSED(nc);
    UNUSED(nb);
    UNUSED(ncols_interleaved);
    UNUSED(blocklen);

    float sumf[8];
    float sum_minf[8];
    int sumi1,sumi2,sumi3,sumi4;
    int sumi;

    const block_q8_K * a_ptr = (const block_q8_K *)vy;
    for(int x = 0; x < nc / ncols_interleaved; x++) {
        const block_q2_Kx8 * b_ptr = (const block_q2_Kx8 *) vx + (x * nb);
        for (int j = 0; j < ncols_interleaved; j++) {
            sumf[j] = 0.0;
            sum_minf[j] = 0.0;
        }
        for (int l = 0; l < nb; l++) {
            for (int k = 0; k < (qk / (4 * blocklen)); k++) {
                const uint8_t *scales_0 = b_ptr[l].scales + (k / 4) * 64 ;
                const uint8_t *scales_1 = b_ptr[l].scales + (k / 4) * 64 + 16;
                const uint8_t *scales_2 = b_ptr[l].scales + (k / 4) * 64 + 32;
                const uint8_t *scales_3 = b_ptr[l].scales + (k / 4) * 64 + 48;
                for (int j = 0; j < ncols_interleaved; j++) {
                    sumi1 = 0;
                    sumi2 = 0;
                    sumi3 = 0;
                    sumi4 = 0;
                    sumi = 0;
                    int offset = ((k / 2) % 2) + j * 2;
                    for (int i = 0; i < blocklen; ++i){
                        const int v0 = (int8_t) (b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] & 3);
                        const int v1 = (int8_t) ((b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] >> 2 ) & 3);
                        const int v2 = (int8_t) ((b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] >> 4 ) & 3);
                        const int v3 = (int8_t) ((b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] >> 6 ) & 3);
                        sumi1 = (v0 * a_ptr[l].qs[(k >> 2) * 128 + (k % 4) * blocklen + i]);
                        sumi2 = (v1 * a_ptr[l].qs[(k >> 2) * 128 + (k % 4) * blocklen + i + 32]);
                        sumi3 = (v2 * a_ptr[l].qs[(k >> 2) * 128 + (k % 4) * blocklen + i + 64]);
                        sumi4 = (v3 * a_ptr[l].qs[(k >> 2) * 128 + (k % 4) * blocklen + i + 96]);

                        sumi1 = sumi1 * (scales_0[offset] & 0xF);
                        sumi2 = sumi2 * (scales_1[offset] & 0xF);
                        sumi3 = sumi3 * (scales_2[offset] & 0xF);
                        sumi4 = sumi4 * (scales_3[offset] & 0xF);
                        sumi += sumi1 + sumi2 + sumi3 + sumi4;
                    }
                    sumf[j] += sumi * GGML_FP16_TO_FP32(b_ptr[l].d[j]) * a_ptr[l].d;
                }
            }
            for(int sb = 0; sb < 8; sb++) {
                const uint8_t *mins = b_ptr[l].scales + sb * 16;
                for(int j = 0; j < ncols_interleaved; j++){
                    sum_minf[j] += ((mins[j * 2] >> 4) * a_ptr[l].bsums[sb * 2] + (mins[(j * 2)+ 1] >> 4) * a_ptr[l].bsums[sb * 2 + 1]) * GGML_FP16_TO_FP32(b_ptr[l].dmin[j]) * a_ptr[l].d;
                }
            }
        }
        for (int j = 0; j < ncols_interleaved; j++) {
            s[x * ncols_interleaved + j] = sumf[j] - sum_minf[j];
        }
    }
}

void ggml_gemv_q3_K_8x8_q8_K_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc) {
    const int qk = QK_K;
    const int nb = n / qk;
    const int ncols_interleaved = 8;

    assert (n % qk == 0);
    assert (nc % ncols_interleaved == 0);

    UNUSED(bs);
    UNUSED(nr);

    float sumf[8];
    // Sub block scales of the 8 interleaved Q3_K structures, unpacked to 6-bit values: sc[sb][j]
    uint8_t sc[16][8];

    const block_q8_K * a_ptr = (const block_q8_K *)vy;
    for (int x = 0; x < nc / ncols_interleaved; x++) {
        const block_q3_Kx8 * b_ptr = (const block_q3_Kx8 *) vx + (x * nb);
        for (int j = 0; j < ncols_interleaved; j++) {
            sumf[j] = 0.0;
        }
        for (int l = 0; l < nb; l++) {
            // The packed scales are stored transposed (scales byte i of block j at scales[i*8 + j]),
            // unpack them with the regular Q3_K bit fiddling to one 6-bit value per sub block
            const uint8_t * scp = b_ptr[l].scales;
            for (int j = 0; j < ncols_interleaved; j++) {
                for (int k = 0; k < 4; k++) {
                    sc[k     ][j] = (scp[ k     * 8 + j] & 15) | (( scp[(8 + k) * 8 + j]       & 3) << 4);
                    sc[4 + k ][j] = (scp[(4 + k) * 8 + j] & 15) | (((scp[(8 + k) * 8 + j] >> 2) & 3) << 4);
                    sc[8 + k ][j] = ((scp[k      * 8 + j] >> 4) & 15) | (((scp[(8 + k) * 8 + j] >> 4) & 3) << 4);
                    sc[12 + k][j] = ((scp[(4 + k) * 8 + j] >> 4) & 15) | (((scp[(8 + k) * 8 + j] >> 6) & 3) << 4);
                }
            }
            for (int j = 0; j < ncols_interleaved; j++) {
                int32_t isum = 0;
                for (int sb = 0; sb < 16; sb++) {
                    // Sub block sb of block j: the 2-bit fields of its 16 elements are at shift
                    // 2*jj of 16 consecutive bytes, the high bits at bit h2*4+jj of 16 consecutive bytes
                    const int h2 = sb / 8, jj = (sb % 8) / 2, c = sb % 2;
                    const uint8_t * q2 = b_ptr[l].qs + (h2 * 4 + c * 2) * 64 + j * 8;
                    const uint8_t * hm = b_ptr[l].hmask + (c * 2) * 64 + j * 8;
                    const int8_t  * q8 = a_ptr[l].qs + sb * 16;
                    // The dequantized value is scale * (u - 4) with u = q2 + 4*h in [0, 7];
                    // the -4 part is folded in with the q8_K sub block sums
                    int32_t dot = -4 * (int32_t) a_ptr[l].bsums[sb];
                    for (int i = 0; i < 16; i++) {
                        const int u = ((q2[(i / 8) * 64 + (i % 8)] >> (2 * jj)) & 3)
                                    + (((hm[(i / 8) * 64 + (i % 8)] >> (h2 * 4 + jj)) & 1) << 2);
                        dot += u * q8[i];
                    }
                    isum += ((int) sc[sb][j] - 32) * dot;
                }
                sumf[j] += isum * GGML_FP16_TO_FP32(b_ptr[l].d[j]) * a_ptr[l].d;
            }
        }
        for (int j = 0; j < ncols_interleaved; j++) {
            s[x * ncols_interleaved + j] = sumf[j];
        }
    }
}

void ggml_gemv_q5_K_8x4_q8_K_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc) {
    ggml_gemv_q5_K_NxM_q8_K_generic_impl<4, 8>(n, s, bs, vx, vy, nr, nc);
}

void ggml_gemv_q5_K_8x8_q8_K_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc) {
    ggml_gemv_q5_K_NxM_q8_K_generic_impl<8, 8>(n, s, bs, vx, vy, nr, nc);
}


void ggml_gemv_q6_K_8x4_q8_K_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc) {
    ggml_gemv_q6_K_NxM_q8_K_generic_impl<4, 8>(n, s, bs, vx, vy, nr, nc);
}

void ggml_gemv_q6_K_8x8_q8_K_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc) {
    ggml_gemv_q6_K_NxM_q8_K_generic_impl<8, 8>(n, s, bs, vx, vy, nr, nc);
}

void ggml_gemv_iq1_s_8x8_q8_K_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc) {
    const int qk = QK_K;
    const int nb = n / qk;
    const int ncols_interleaved = 8;

    assert(nr == 1);
    assert(n % qk == 0);
    assert(nc % ncols_interleaved == 0);

    UNUSED(bs);
    UNUSED(nr);

    const block_iq1_sx8 * GGML_RESTRICT x = (const block_iq1_sx8 *) vx;
    const block_q8_K    * GGML_RESTRICT y = (const block_q8_K *) vy;

    float sumf[8];

    for (int cx = 0; cx < nc / ncols_interleaved; cx++) {
        const block_iq1_sx8 * b_ptr = x + (size_t) cx * nb;
        for (int j = 0; j < 8; j++) {
            sumf[j] = 0.0f;
        }
        for (int i = 0; i < nb; i++) {
            for (int j = 0; j < 8; j++) {
                const int8_t * q8 = y[i].qs;
                int sumi = 0, sumi1 = 0;
                for (int ib = 0; ib < QK_K/32; ++ib) {
                    const uint16_t h = *(const uint16_t *)(b_ptr[i].qh + ib * 16 + j * 2);
                    const int ls    = 2*((h >> 12) & 7) + 1;
                    const int delta = h & 0x8000 ? -1 : 1;
                    int lsum = 0;
                    for (int l = 0; l < 4; ++l) {
                        const int     t    = ib * 4 + l; // grid entry index within the block
                        const uint8_t q    = b_ptr[i].qs[(t / 8) * 64 + j * 8 + t % 8];
                        const int8_t * grid = (const int8_t *)(iq1s_grid + (q | (((h >> 3*l) & 7) << 8)));
                        for (int k = 0; k < 8; ++k) {
                            lsum += q8[k] * grid[k];
                        }
                        q8 += 8;
                    }
                    sumi  += ls * lsum;
                    sumi1 += ls * delta * (y[i].bsums[2*ib+0] + y[i].bsums[2*ib+1]);
                }
                sumf[j] += GGML_CPU_FP16_TO_FP32(b_ptr[i].d[j]) * y[i].d * (sumi + IQ1S_DELTA * sumi1);
            }
        }
        for (int j = 0; j < 8; j++) {
            s[cx * ncols_interleaved + j] = sumf[j];
        }
    }
}

void ggml_gemv_iq1_m_8x8_q8_K_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc) {
    const int qk = QK_K;
    const int nb = n / qk;
    const int ncols_interleaved = 8;

    assert(nr == 1);
    assert(n % qk == 0);
    assert(nc % ncols_interleaved == 0);

    UNUSED(bs);
    UNUSED(nr);

    const block_iq1_mx8 * GGML_RESTRICT x = (const block_iq1_mx8 *) vx;
    const block_q8_K    * GGML_RESTRICT y = (const block_q8_K *) vy;

    iq1m_scale_t scale;

    float sumf[8];

    for (int cx = 0; cx < nc / ncols_interleaved; cx++) {
        const block_iq1_mx8 * b_ptr = x + (size_t) cx * nb;
        for (int j = 0; j < 8; j++) {
            sumf[j] = 0.0f;
        }
        for (int i = 0; i < nb; i++) {
            for (int j = 0; j < 8; j++) {
                const uint16_t * scj = (const uint16_t *)(b_ptr[i].scales + j * 8);
                scale.u16 = (scj[0] >> 12) | ((scj[1] >> 8) & 0x00f0) | ((scj[2] >> 4) & 0x0f00) | (scj[3] & 0xf000);

                const int8_t * q8 = y[i].qs;
                int sumi1 = 0, sumi2 = 0;
                for (int ib = 0; ib < QK_K/32; ++ib) {
                    const uint8_t h0 = b_ptr[i].qh[ib * 16 + j * 2 + 0];
                    const uint8_t h1 = b_ptr[i].qh[ib * 16 + j * 2 + 1];
                    const int delta[4] = { h0 & 0x08 ? -1 : 1, h0 & 0x80 ? -1 : 1, h1 & 0x08 ? -1 : 1, h1 & 0x80 ? -1 : 1 };
                    int sum1[2] = { 0, 0 }, sum2[2] = { 0, 0 };
                    for (int l = 0; l < 4; ++l) {
                        const int     t    = ib * 4 + l;
                        const uint8_t q    = b_ptr[i].qs[(t / 8) * 64 + j * 8 + t % 8];
                        const uint8_t hq   = l < 2 ? h0 : h1;
                        const int8_t * grid = (const int8_t *)(iq1s_grid + (q | (((uint16_t)hq << (8 - 4*(l%2))) & 0x700)));
                        int lsum1 = 0, lsum2 = 0;
                        for (int k = 0; k < 8; ++k) {
                            lsum1 += q8[k] * grid[k];
                            lsum2 += q8[k];
                        }
                        q8 += 8;
                        sum1[l/2] += lsum1;
                        sum2[l/2] += lsum2*delta[l];
                    }
                    const int ls1 = 2*((scj[ib/2] >> (6*(ib%2)+0)) & 0x7) + 1;
                    const int ls2 = 2*((scj[ib/2] >> (6*(ib%2)+3)) & 0x7) + 1;
                    sumi1 += sum1[0] * ls1 + sum1[1] * ls2;
                    sumi2 += sum2[0] * ls1 + sum2[1] * ls2;
                }
                sumf[j] += GGML_CPU_FP16_TO_FP32(scale.f16) * y[i].d * (sumi1 + IQ1M_DELTA * sumi2);
            }
        }
        for (int j = 0; j < 8; j++) {
            s[cx * ncols_interleaved + j] = sumf[j];
        }
    }
}

void ggml_gemv_iq2_xs_8x8_q8_K_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc) {
    const int qk = QK_K;
    const int nb = n / qk;
    const int ncols_interleaved = 8;

    assert(nr == 1);
    assert(n % qk == 0);
    assert(nc % ncols_interleaved == 0);

    UNUSED(bs);
    UNUSED(nr);

    const block_iq2_xsx8 * GGML_RESTRICT x = (const block_iq2_xsx8 *) vx;
    const block_q8_K     * GGML_RESTRICT y = (const block_q8_K *) vy;

    float sumf[8];

    for (int cx = 0; cx < nc / ncols_interleaved; cx++) {
        const block_iq2_xsx8 * b_ptr = x + (size_t) cx * nb;
        for (int j = 0; j < 8; j++) {
            sumf[j] = 0.0f;
        }
        for (int i = 0; i < nb; i++) {
            for (int j = 0; j < 8; j++) {
                const int8_t * q8 = y[i].qs;
                int bsum = 0;
                for (int g = 0; g < QK_K/32; ++g) {
                    const uint8_t  sc  = b_ptr[i].scales[g*8 + j];
                    const int      ls1 = 1 + 2*(sc & 0xf);
                    const int      ls2 = 1 + 2*(sc >> 4);
                    const uint16_t * q2 = b_ptr[i].qs + g*32 + j*4;
                    int sumi1 = 0, sumi2 = 0;
                    for (int l = 0; l < 4; ++l) {
                        const uint8_t * grid  = (const uint8_t *)(iq2xs_grid + (q2[l] & 511));
                        const uint8_t   signs = ksigns_iq2xs[q2[l] >> 9];
                        int lsum = 0;
                        for (int k = 0; k < 8; ++k) {
                            lsum += grid[k] * q8[k] * (signs & kmask_iq2xs[k] ? -1 : 1);
                        }
                        if (l < 2) {
                            sumi1 += lsum;
                        } else {
                            sumi2 += lsum;
                        }
                        q8 += 8;
                    }
                    bsum += ls1 * sumi1 + ls2 * sumi2;
                }
                sumf[j] += GGML_CPU_FP16_TO_FP32(b_ptr[i].d[j]) * y[i].d * bsum;
            }
        }
        for (int j = 0; j < 8; j++) {
            s[cx * ncols_interleaved + j] = 0.125f * sumf[j];
        }
    }
}

void ggml_gemv_iq3_xxs_8x8_q8_K_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc) {
    const int qk = QK_K;
    const int nb = n / qk;
    const int ncols_interleaved = 8;

    assert(nr == 1);
    assert(n % qk == 0);
    assert(nc % ncols_interleaved == 0);

    UNUSED(bs);
    UNUSED(nr);

    const block_iq3_xxsx8 * GGML_RESTRICT x = (const block_iq3_xxsx8 *) vx;
    const block_q8_K      * GGML_RESTRICT y = (const block_q8_K *) vy;

    uint32_t aux32;

    float sumf[8];

    for (int cx = 0; cx < nc / ncols_interleaved; cx++) {
        const block_iq3_xxsx8 * b_ptr = x + (size_t) cx * nb;
        for (int j = 0; j < 8; j++) {
            sumf[j] = 0.0f;
        }
        for (int i = 0; i < nb; i++) {
            for (int j = 0; j < 8; j++) {
                const int8_t * q8 = y[i].qs;
                int bsum = 0;
                for (int g = 0; g < QK_K/32; ++g) {
                    memcpy(&aux32, b_ptr[i].gas + g*32 + j*4, sizeof(uint32_t));
                    const uint32_t ls = 2*(aux32 >> 28) + 1;
                    const uint8_t * q3 = b_ptr[i].qs + g*64 + j*8;
                    int sumi = 0;
                    for (int l = 0; l < 4; ++l) {
                        const uint8_t * grid1 = (const uint8_t *)(iq3xxs_grid + q3[2*l+0]);
                        const uint8_t * grid2 = (const uint8_t *)(iq3xxs_grid + q3[2*l+1]);
                        const uint8_t   signs = ksigns_iq2xs[(aux32 >> 7*l) & 127];
                        for (int k = 0; k < 4; ++k) {
                            sumi += grid1[k] * q8[k+0] * (signs & kmask_iq2xs[k+0] ? -1 : 1);
                            sumi += grid2[k] * q8[k+4] * (signs & kmask_iq2xs[k+4] ? -1 : 1);
                        }
                        q8 += 8;
                    }
                    bsum += sumi * ls;
                }
                sumf[j] += GGML_CPU_FP16_TO_FP32(b_ptr[i].d[j]) * y[i].d * bsum;
            }
        }
        for (int j = 0; j < 8; j++) {
            s[cx * ncols_interleaved + j] = 0.25f * sumf[j];
        }
    }
}

void ggml_gemv_iq4_nl_4x4_q8_0_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc) {
    const int qk = QK8_0;
    const int nb = n / qk;
    const int ncols_interleaved = 4;
    const int blocklen = 4;

    assert(nr == 1);
    assert(n % qk == 0);
    assert(nc % ncols_interleaved == 0);

    UNUSED(bs);
    UNUSED(nr);

    float sumf[4];
    int sumi;

    const block_q8_0 * a_ptr = (const block_q8_0 *) vy;
    for (int x = 0; x < nc / ncols_interleaved; x++) {
        const block_iq4_nlx4 * b_ptr = (const block_iq4_nlx4 *) vx + (x * nb);

        for (int j = 0; j < ncols_interleaved; j++) sumf[j] = 0.0;
        for (int l = 0; l < nb; l++) {
            for (int k = 0; k < (qk / (2 * blocklen)); k++) {
                for (int j = 0; j < ncols_interleaved; j++) {
                    sumi = 0;
                    for (int i = 0; i < blocklen; ++i) {
                        const int v0 = kvalues_iq4nl[b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] & 0x0F];
                        const int v1 = kvalues_iq4nl[b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] >> 4];
                        sumi += ((v0 * a_ptr[l].qs[k * blocklen + i]) + (v1 * a_ptr[l].qs[k * blocklen + i + qk / 2]));
                    }
                    sumf[j] += sumi * GGML_CPU_FP16_TO_FP32(b_ptr[l].d[j]) * GGML_CPU_FP16_TO_FP32(a_ptr[l].d);
                }
            }
        }
        for (int j = 0; j < ncols_interleaved; j++) s[x * ncols_interleaved + j] = sumf[j];
    }
}

void ggml_gemv_iq4_nl_8x8_q8_0_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc) {
    const int qk = QK8_0;
    const int nb = n / qk;
    const int ncols_interleaved = 8;
    const int blocklen = 8;

    assert(nr == 1);
    assert(n % qk == 0);
    assert(nc % ncols_interleaved == 0);

    UNUSED(bs);
    UNUSED(nr);

    float sumf[8];
    int sumi;

    const block_q8_0 * a_ptr = (const block_q8_0 *) vy;
    for (int x = 0; x < nc / ncols_interleaved; x++) {
        const block_iq4_nlx8 * b_ptr = (const block_iq4_nlx8 *) vx + (x * nb);

        for (int j = 0; j < ncols_interleaved; j++) sumf[j] = 0.0;
        for (int l = 0; l < nb; l++) {
            for (int k = 0; k < (qk / (2 * blocklen)); k++) {
                for (int j = 0; j < ncols_interleaved; j++) {
                    sumi = 0;
                    for (int i = 0; i < blocklen; ++i) {
                        const int v0 = kvalues_iq4nl[b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] & 0x0F];
                        const int v1 = kvalues_iq4nl[b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] >> 4];
                        sumi += ((v0 * a_ptr[l].qs[k * blocklen + i]) + (v1 * a_ptr[l].qs[k * blocklen + i + qk / 2]));
                    }
                    sumf[j] += sumi * GGML_CPU_FP16_TO_FP32(b_ptr[l].d[j]) * GGML_CPU_FP16_TO_FP32(a_ptr[l].d);
                }
            }
        }
        for (int j = 0; j < ncols_interleaved; j++) s[x * ncols_interleaved + j] = sumf[j];
    }
}

void ggml_gemv_mxfp4_4x4_q8_0_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc) {
    const int qk = QK8_0;
    const int nb = n / qk;
    const int ncols_interleaved = 4;
    const int blocklen = 4;

    assert(nr == 1);
    assert(n % qk == 0);
    assert(nc % ncols_interleaved == 0);

    UNUSED(bs);
    UNUSED(nr);

    float sumf[4];
    int sumi;

    const block_q8_0 * a_ptr = (const block_q8_0 *) vy;
    for (int x = 0; x < nc / ncols_interleaved; x++) {
        const block_mxfp4x4 * b_ptr = (const block_mxfp4x4 *) vx + (x * nb);

        for (int j = 0; j < ncols_interleaved; j++) sumf[j] = 0.0;
        for (int l = 0; l < nb; l++) {
            for (int k = 0; k < (qk / (2 * blocklen)); k++) {
                for (int j = 0; j < ncols_interleaved; j++) {
                    sumi = 0;
                    for (int i = 0; i < blocklen; ++i) {
                        const int v0 = kvalues_mxfp4[b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] & 0x0F];
                        const int v1 = kvalues_mxfp4[b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] >> 4];
                        sumi += ((v0 * a_ptr[l].qs[k * blocklen + i]) + (v1 * a_ptr[l].qs[k * blocklen + i + qk / 2]));
                    }
                    sumf[j] += sumi * GGML_CPU_E8M0_TO_FP32_HALF(b_ptr[l].e[j]) * GGML_CPU_FP16_TO_FP32(a_ptr[l].d);
                }
            }
        }
        for (int j = 0; j < ncols_interleaved; j++) s[x * ncols_interleaved + j] = sumf[j];
    }
}

void ggml_gemv_mxfp4_8x8_q8_0_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc) {
    const int qk = QK8_0;
    const int nb = n / qk;
    const int ncols_interleaved = 8;
    const int blocklen = 8;

    assert(nr == 1);
    assert(n % qk == 0);
    assert(nc % ncols_interleaved == 0);

    UNUSED(bs);
    UNUSED(nr);

    float sumf[8];
    int sumi;

    const block_q8_0 * a_ptr = (const block_q8_0 *) vy;
    for (int x = 0; x < nc / ncols_interleaved; x++) {
        const block_mxfp4x8 * b_ptr = (const block_mxfp4x8 *) vx + (x * nb);

        for (int j = 0; j < ncols_interleaved; j++) sumf[j] = 0.0;
        for (int l = 0; l < nb; l++) {
            for (int k = 0; k < (qk / (2 * blocklen)); k++) {
                for (int j = 0; j < ncols_interleaved; j++) {
                    sumi = 0;
                    for (int i = 0; i < blocklen; ++i) {
                        const int v0 = kvalues_mxfp4[b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] & 0x0F];
                        const int v1 = kvalues_mxfp4[b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] >> 4];
                        sumi += ((v0 * a_ptr[l].qs[k * blocklen + i]) + (v1 * a_ptr[l].qs[k * blocklen + i + qk / 2]));
                    }
                    sumf[j] += sumi * GGML_CPU_E8M0_TO_FP32_HALF(b_ptr[l].e[j]) * GGML_CPU_FP16_TO_FP32(a_ptr[l].d);
                }
            }
        }
        for (int j = 0; j < ncols_interleaved; j++) s[x * ncols_interleaved + j] = sumf[j];
    }
}

void ggml_gemv_q8_0_4x4_q8_0_generic(int                        n,
                                     float * GGML_RESTRICT      s,
                                     size_t                     bs,
                                     const void * GGML_RESTRICT vx,
                                     const void * GGML_RESTRICT vy,
                                     int                        nr,
                                     int                        nc) {
    const int qk                = QK8_0;
    const int nb                = n / qk;
    const int ncols_interleaved = 4;
    const int blocklen          = 4;

    assert(nr == 1);
    assert(n % qk == 0);
    assert(nc % ncols_interleaved == 0);

    UNUSED(bs);
    UNUSED(nr);

    float sumf[4];
    int   sumi;

    const block_q8_0 * a_ptr = (const block_q8_0 *) vy;
    for (int x = 0; x < nc / ncols_interleaved; x++) {
        const block_q8_0x4 * b_ptr = (const block_q8_0x4 *) vx + (x * nb);

        for (int j = 0; j < ncols_interleaved; j++) {
            sumf[j] = 0.0;
        }
        for (int l = 0; l < nb; l++) {
            for (int k = 0; k < (qk / blocklen); k++) {
                for (int j = 0; j < ncols_interleaved; j++) {
                    sumi = 0;
                    for (int i = 0; i < blocklen; ++i) {
                        const int v0 = b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i];
                        sumi += v0 * a_ptr[l].qs[k * blocklen + i];
                    }
                    sumf[j] += sumi * GGML_CPU_FP16_TO_FP32(b_ptr[l].d[j]) * GGML_CPU_FP16_TO_FP32(a_ptr[l].d);
                }
            }
        }
        for (int j = 0; j < ncols_interleaved; j++) {
            s[x * ncols_interleaved + j] = sumf[j];
        }
    }
}

void ggml_gemv_q8_0_4x8_q8_0_generic(int                        n,
                                     float * GGML_RESTRICT      s,
                                     size_t                     bs,
                                     const void * GGML_RESTRICT vx,
                                     const void * GGML_RESTRICT vy,
                                     int                        nr,
                                     int                        nc) {
    const int qk                = QK8_0;
    const int nb                = n / qk;
    const int ncols_interleaved = 4;
    const int blocklen          = 8;

    assert(nr == 1);
    assert(n % qk == 0);
    assert(nc % ncols_interleaved == 0);

    UNUSED(bs);
    UNUSED(nr);

    float sumf[4];
    int   sumi;

    const block_q8_0 * a_ptr = (const block_q8_0 *) vy;
    for (int x = 0; x < nc / ncols_interleaved; x++) {
        const block_q8_0x4 * b_ptr = (const block_q8_0x4 *) vx + (x * nb);

        for (int j = 0; j < ncols_interleaved; j++) {
            sumf[j] = 0.0;
        }
        for (int l = 0; l < nb; l++) {
            for (int k = 0; k < (qk / blocklen); k++) {
                for (int j = 0; j < ncols_interleaved; j++) {
                    sumi = 0;
                    for (int i = 0; i < blocklen; ++i) {
                        const int v0 = b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i];
                        sumi += v0 * a_ptr[l].qs[k * blocklen + i];
                    }
                    sumf[j] += sumi * GGML_CPU_FP16_TO_FP32(b_ptr[l].d[j]) * GGML_CPU_FP16_TO_FP32(a_ptr[l].d);
                }
            }
        }
        for (int j = 0; j < ncols_interleaved; j++) {
            s[x * ncols_interleaved + j] = sumf[j];
        }
    }
}

// Only enable these for RISC-V.
#if defined __riscv_zvfh
void ggml_gemv_q4_0_16x1_q8_0_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc) {
    const int qk = QK8_0;
    const int nb = n / qk;
    const int ncols_interleaved = 16;
    const int blocklen = 1;

    assert (n % qk == 0);
    assert (nc % ncols_interleaved == 0);

    UNUSED(s);
    UNUSED(bs);
    UNUSED(vx);
    UNUSED(vy);
    UNUSED(nr);
    UNUSED(nc);
    UNUSED(nb);
    UNUSED(ncols_interleaved);
    UNUSED(blocklen);

    float sumf[16];
    int sumi;

    const block_q8_0 * a_ptr = (const block_q8_0 *) vy;
    for (int x = 0; x < nc / ncols_interleaved; x++) {
        const block_q4_0x16 * b_ptr = (const block_q4_0x16 *) vx + (x * nb);

        for (int j = 0; j < ncols_interleaved; j++) sumf[j] = 0.0;
        for (int l = 0; l < nb; l++) {
            for (int k = 0; k < (qk / (2 * blocklen)); k++) {
                for (int j = 0; j < ncols_interleaved; j++) {
                    sumi = 0;
                    for (int i = 0; i < blocklen; ++i) {
                        const int v0 = (int8_t) (b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] << 4);
                        const int v1 = (int8_t) (b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] & 0xF0);
                        sumi += ((v0 * a_ptr[l].qs[k * blocklen + i]) + (v1 * a_ptr[l].qs[k * blocklen + i + qk / 2])) >> 4;
                    }
                    sumf[j] += sumi * GGML_CPU_FP16_TO_FP32(b_ptr[l].d[j]) * GGML_CPU_FP16_TO_FP32(a_ptr[l].d);
                }
            }
        }
        for (int j = 0; j < ncols_interleaved; j++) s[x * ncols_interleaved + j] = sumf[j];
    }
}

void ggml_gemv_q4_K_16x1_q8_K_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc) {
    const int qk = QK_K;
    const int nb = n / qk;
    const int ncols_interleaved = 16;
    const int blocklen = 1;
    assert (n % qk == 0);
    assert (nc % ncols_interleaved == 0);
    UNUSED(s);
    UNUSED(bs);
    UNUSED(vx);
    UNUSED(vy);
    UNUSED(nr);
    UNUSED(nc);
    UNUSED(nb);
    UNUSED(ncols_interleaved);
    UNUSED(blocklen);
    float sumf[16];
    float sum_minf[16];
    uint8_t scales[128];
    uint8_t mins[128];
    int sumi1;
    int sumi2;
    int sumi;
    const block_q8_K * a_ptr = (const block_q8_K *) vy;
    for (int x = 0; x < nc / ncols_interleaved; x++) {
        const block_q4_Kx16 * b_ptr = (const block_q4_Kx16 *) vx + (x * nb);
        for (int j = 0; j < ncols_interleaved; j++) {
            sumf[j] = 0.0f;
            sum_minf[j] = 0.0f;
        }
        for (int l = 0; l < nb; l++) {
            for (int i = 0; i < 128; i++) {
                scales[i] = b_ptr[l].scales[i] & 0x0F;
                mins[i] = b_ptr[l].scales[i] >> 4;
            }
            for (int i = 0; i < 64; i++) {
                scales[i] |= (b_ptr[l].scales[128 + i] & 0x03) << 4;
                mins[i] |= (b_ptr[l].scales[128 + i] & 0x0C) << 2;
                scales[i + 64] |= (b_ptr[l].scales[128 + i] & 0x30);
                mins[i + 64] |= (b_ptr[l].scales[128 + i] & 0xC0) >> 2;
            }
            for (int sb = 0; sb < 8; sb++) {
                uint8_t *min = &mins[sb * 16];
                for (int j = 0; j < ncols_interleaved; j++) {
                    sum_minf[j] += min[j] * (a_ptr[l].bsums[sb * 2] + a_ptr[l].bsums[sb * 2 + 1]) * GGML_CPU_FP16_TO_FP32(b_ptr[l].dmin[j]) * a_ptr[l].d;
                }
            }
            for (int sb = 0; sb < 8; sb += 2) {
                uint8_t *scales_0 = &scales[sb * 16];
                uint8_t *scales_1 = &scales[(sb + 1) * 16];
                for (int i = 0; i < QK4_0; i++) {
                    for (int j = 0; j < ncols_interleaved; j++) {
                        sumi1 = 0;
                        sumi2 = 0;
                        sumi = 0;
                        const int v0 = (int8_t) (b_ptr[l].qs[sb * 256 + i * 16 + j] & 0xF);
                        const int v1 = (int8_t) (b_ptr[l].qs[sb * 256 + i * 16 + j] >> 4);
                        sumi1 = (v0 * a_ptr[l].qs[sb * 32 + i]);
                        sumi2 = (v1 * a_ptr[l].qs[sb * 32 + 32 + i]);
                        sumi1 = sumi1 * scales_0[j];
                        sumi2 = sumi2 * scales_1[j];
                        sumi += sumi1 + sumi2;
                        sumf[j] += sumi * GGML_CPU_FP16_TO_FP32(b_ptr[l].d[j]) * a_ptr[l].d;
                    }
                }
            }
        }
        for (int j = 0; j < ncols_interleaved; j++) {
            s[x * ncols_interleaved + j] = sumf[j] - sum_minf[j];
        }
    }
}

void ggml_gemv_iq4_nl_16x1_q8_0_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc) {
    const int qk = QK8_0;
    const int nb = n / qk;
    const int ncols_interleaved = 16;
    const int blocklen = 1;

    assert(nr == 1);
    assert(n % qk == 0);
    assert(nc % ncols_interleaved == 0);

    UNUSED(bs);
    UNUSED(nr);

    float sumf[16];
    int sumi;

    const block_q8_0 * a_ptr = (const block_q8_0 *) vy;
    for (int x = 0; x < nc / ncols_interleaved; x++) {
        const block_iq4_nlx16 * b_ptr = (const block_iq4_nlx16 *) vx + (x * nb);

        for (int j = 0; j < ncols_interleaved; j++) sumf[j] = 0.0;
        for (int l = 0; l < nb; l++) {
            for (int k = 0; k < (qk / (2 * blocklen)); k++) {
                for (int j = 0; j < ncols_interleaved; j++) {
                    sumi = 0;
                    for (int i = 0; i < blocklen; ++i) {
                        const int v0 = kvalues_iq4nl[b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] & 0x0F];
                        const int v1 = kvalues_iq4nl[b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] >> 4];
                        sumi += ((v0 * a_ptr[l].qs[k * blocklen + i]) + (v1 * a_ptr[l].qs[k * blocklen + i + qk / 2]));
                    }
                    sumf[j] += sumi * GGML_CPU_FP16_TO_FP32(b_ptr[l].d[j]) * GGML_CPU_FP16_TO_FP32(a_ptr[l].d);
                }
            }
        }
        for (int j = 0; j < ncols_interleaved; j++) s[x * ncols_interleaved + j] = sumf[j];
    }
}

void ggml_gemv_q8_0_16x1_q8_0_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc) {
    const int qk                = QK8_0;
    const int nb                = n / qk;
    const int ncols_interleaved = 16;
    const int blocklen          = 1;

    assert(nr == 1);
    assert(n % qk == 0);
    assert(nc % ncols_interleaved == 0);

    UNUSED(bs);
    UNUSED(nr);

    float sumf[16];
    int   sumi;

    const block_q8_0 * a_ptr = (const block_q8_0 *) vy;
    for (int x = 0; x < nc / ncols_interleaved; x++) {
        const block_q8_0x16 * b_ptr = (const block_q8_0x16 *) vx + (x * nb);

        for (int j = 0; j < ncols_interleaved; j++) {
            sumf[j] = 0.0;
        }
        for (int l = 0; l < nb; l++) {
            for (int k = 0; k < (qk / blocklen); k++) {
                for (int j = 0; j < ncols_interleaved; j++) {
                    sumi = 0;
                    for (int i = 0; i < blocklen; ++i) {
                        const int v0 = b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i];
                        sumi += v0 * a_ptr[l].qs[k * blocklen + i];
                    }
                    sumf[j] += sumi * GGML_CPU_FP16_TO_FP32(b_ptr[l].d[j]) * GGML_CPU_FP16_TO_FP32(a_ptr[l].d);
                }
            }
        }
        for (int j = 0; j < ncols_interleaved; j++) {
            s[x * ncols_interleaved + j] = sumf[j];
        }
    }
}

void ggml_gemv_q2_K_16x1_q8_K_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc) {
    assert(n % QK_K == 0);
    assert(nr == 1);
    assert(nc % 16 == 0);

    UNUSED(bs);
    UNUSED(nr);

    const int nb = n / QK_K;
    const block_q2_Kx16 * x = (const block_q2_Kx16 *)vx;
    const block_q8_K    * y = (const block_q8_K *)vy;

    // Layout: Even-Low(0,2,4,6), Odd-Low(1,3,5,7), Even-High(8...), Odd-High(9...)
    const int sb_perm[16] = {
        0, 4, 1, 5, 2, 6, 3, 7,  // 0-7
        8, 12, 9, 13, 10, 14, 11, 15 // 8-15
    };

    for (int col_tile = 0; col_tile < nc; col_tile += 16) {
        const block_q2_Kx16 * x_ptr = x + (col_tile / 16) * nb;
        const block_q8_K    * y_ptr = y;

        float sumf[16] = {0};

        // Loop over K-blocks
        for (int k_block = 0; k_block < nb; ++k_block) {
            int32_t isum[16]  = {0};
            int32_t summs[16] = {0};

            const uint8_t * qs_rhs = x_ptr[k_block].qs;
            const uint8_t * sc_rhs = x_ptr[k_block].scales;
            const int8_t  * qs_lhs = y_ptr[k_block].qs;
            const int16_t * bs_lhs = y_ptr[k_block].bsums;

            // Iterate over sub-blocks 0..15
            for (int sb = 0; sb < 16; ++sb) {
                // Correction Term
                int16_t bsum = bs_lhs[sb];
                int scale_offset = sb_perm[sb] * 16;

                for (int col = 0; col < 16; ++col) {
                    uint8_t sc_val = sc_rhs[scale_offset + col];
                    summs[col] += bsum * (sc_val >> 4); // Min is high 4 bits
                }

                // Main Dot Product
                // Calculate base offsets for Q2 unpacking based on SB
                int byte_base;
                if (sb < 8) byte_base = (sb % 2 == 0) ? 0 : 16;
                else        byte_base = (sb % 2 == 0) ? 32 : 48;

                int shift = ((sb / 2) % 4) * 2;

                for (int col = 0; col < 16; ++col) {
                    uint8_t sc_val = sc_rhs[scale_offset + col];
                    int32_t d_sb = sc_val & 0xF; // Scale is low 4 bits

                    // Process 16 elements (l=0..15)
                    for (int l = 0; l < 16; ++l) {
                        // Q2: Interleaved by column. Byte `l` contains 4 k-values.
                        int qs_idx = (byte_base + l) * 16 + col;
                        uint8_t q2_val = (qs_rhs[qs_idx] >> shift) & 3;

                        // Q8: Linear access
                        int k = sb * 16 + l;
                        int8_t q8_val = qs_lhs[k];

                        isum[col] += q8_val * q2_val * d_sb;
                    }
                }
            }

            // Finalize K-Block
            for (int col = 0; col < 16; ++col) {
                float d_lhs = y_ptr[k_block].d;
                float d_rhs = GGML_FP16_TO_FP32(x_ptr[k_block].d[col]);
                float dm_rhs = GGML_FP16_TO_FP32(x_ptr[k_block].dmin[col]);

                float d_all = d_lhs * d_rhs;
                float d_min = d_lhs * dm_rhs;

                sumf[col] += (isum[col] * d_all) - (summs[col] * d_min);
            }
        }

        for (int col = 0; col < 16; ++col) {
            s[col_tile + col] = sumf[col];
        }
    }
}
#endif

void ggml_gemm_q4_0_4x4_q8_0_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc) {
    const int qk = QK8_0;
    const int nb = n / qk;
    const int ncols_interleaved = 4;
    const int blocklen = 4;

    assert (n % qk == 0);
    assert (nr % 4 == 0);
    assert (nc % ncols_interleaved == 0);

    UNUSED(s);
    UNUSED(bs);
    UNUSED(vx);
    UNUSED(vy);
    UNUSED(nr);
    UNUSED(nc);
    UNUSED(nb);
    UNUSED(ncols_interleaved);
    UNUSED(blocklen);

    {
        float sumf[4][4];
        int sumi;

        for (int y = 0; y < nr / 4; y++) {
            const block_q8_0x4 * a_ptr = (const block_q8_0x4 *) vy + (y * nb);
            for (int x = 0; x < nc / ncols_interleaved; x++) {
                const block_q4_0x4 * b_ptr = (const block_q4_0x4 *) vx + (x * nb);
                for (int m = 0; m < 4; m++) {
                    for (int j = 0; j < ncols_interleaved; j++) sumf[m][j] = 0.0;
                }
                for (int l = 0; l < nb; l++) {
                    for (int k = 0; k < (qk / (2 * blocklen)); k++) {
                        for (int m = 0; m < 4; m++) {
                            for (int j = 0; j < ncols_interleaved; j++) {
                                sumi = 0;
                                for (int i = 0; i < blocklen; ++i) {
                                    const int v0 = (int8_t) (b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] << 4);
                                    const int v1 = (int8_t) (b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] & 0xF0);
                                    sumi += ((v0 * a_ptr[l].qs[k * 4 * blocklen + m * blocklen + i]) +
                                            (v1 * a_ptr[l].qs[k * 4 * blocklen + m * blocklen + i + qk / 2 * 4])) >> 4;
                                }
                                sumf[m][j] += sumi * GGML_CPU_FP16_TO_FP32(b_ptr[l].d[j]) * GGML_CPU_FP16_TO_FP32(a_ptr[l].d[m]);
                            }
                        }
                    }
                }
                for (int m = 0; m < 4; m++) {
                    for (int j = 0; j < ncols_interleaved; j++)
                        s[(y * 4 + m) * bs + x * ncols_interleaved + j] = sumf[m][j];
                }
            }
        }
    }
}

void ggml_gemm_q4_0_4x8_q8_0_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc) {
    const int qk = QK8_0;
    const int nb = n / qk;
    const int ncols_interleaved = 4;
    const int blocklen = 8;

    assert (n % qk == 0);
    assert (nr % 4 == 0);
    assert (nc % ncols_interleaved == 0);

    UNUSED(s);
    UNUSED(bs);
    UNUSED(vx);
    UNUSED(vy);
    UNUSED(nr);
    UNUSED(nc);
    UNUSED(nb);
    UNUSED(ncols_interleaved);
    UNUSED(blocklen);

    float sumf[4][4];
    int sumi;

    for (int y = 0; y < nr / 4; y++) {
        const block_q8_0x4 * a_ptr = (const block_q8_0x4 *) vy + (y * nb);
        for (int x = 0; x < nc / ncols_interleaved; x++) {
            const block_q4_0x4 * b_ptr = (const block_q4_0x4 *) vx + (x * nb);
            for (int m = 0; m < 4; m++) {
                for (int j = 0; j < ncols_interleaved; j++) sumf[m][j] = 0.0;
            }
            for (int l = 0; l < nb; l++) {
                for (int k = 0; k < (qk / (2 * blocklen)); k++) {
                    for (int m = 0; m < 4; m++) {
                        for (int j = 0; j < ncols_interleaved; j++) {
                            sumi = 0;
                            for (int i = 0; i < blocklen; ++i) {
                                const int v0 = (int8_t) (b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] << 4);
                                const int v1 = (int8_t) (b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] & 0xF0);
                                sumi += ((v0 * a_ptr[l].qs[k * 4 * blocklen + m * blocklen + i]) +
                                        (v1 * a_ptr[l].qs[k * 4 * blocklen + m * blocklen + i + qk / 2 * 4])) >> 4;
                            }
                            sumf[m][j] += sumi * GGML_CPU_FP16_TO_FP32(b_ptr[l].d[j]) * GGML_CPU_FP16_TO_FP32(a_ptr[l].d[m]);
                        }
                    }
                }
            }
            for (int m = 0; m < 4; m++) {
                for (int j = 0; j < ncols_interleaved; j++)
                    s[(y * 4 + m) * bs + x * ncols_interleaved + j] = sumf[m][j];
            }
        }
    }
}

void ggml_gemm_q4_0_8x8_q8_0_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc) {
    const int qk = QK8_0;
    const int nb = n / qk;
    const int ncols_interleaved = 8;
    const int blocklen = 8;

    assert (n % qk == 0);
    assert (nr % 4 == 0);
    assert (nc % ncols_interleaved == 0);

    UNUSED(s);
    UNUSED(bs);
    UNUSED(vx);
    UNUSED(vy);
    UNUSED(nr);
    UNUSED(nc);
    UNUSED(nb);
    UNUSED(ncols_interleaved);
    UNUSED(blocklen);

    float sumf[4][8];
    int sumi;

    for (int y = 0; y < nr / 4; y++) {
        const block_q8_0x4 * a_ptr = (const block_q8_0x4 *) vy + (y * nb);
        for (int x = 0; x < nc / ncols_interleaved; x++) {
            const block_q4_0x8 * b_ptr = (const block_q4_0x8 *) vx + (x * nb);
            for (int m = 0; m < 4; m++) {
                for (int j = 0; j < ncols_interleaved; j++) sumf[m][j] = 0.0;
            }
            for (int l = 0; l < nb; l++) {
                for (int k = 0; k < (qk / (2 * blocklen)); k++) {
                    for (int m = 0; m < 4; m++) {
                        for (int j = 0; j < ncols_interleaved; j++) {
                            sumi = 0;
                            for (int i = 0; i < blocklen; ++i) {
                                const int v0 = (int8_t) (b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] << 4);
                                const int v1 = (int8_t) (b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] & 0xF0);
                                sumi += ((v0 * a_ptr[l].qs[k * 4 * blocklen + m * blocklen + i]) +
                                         (v1 * a_ptr[l].qs[k * 4 * blocklen + m * blocklen + i + qk / 2 * 4])) >> 4;
                            }
                            sumf[m][j] += sumi * GGML_CPU_FP16_TO_FP32(b_ptr[l].d[j]) * GGML_CPU_FP16_TO_FP32(a_ptr[l].d[m]);
                        }
                    }
                }
            }
            for (int m = 0; m < 4; m++) {
                for (int j = 0; j < ncols_interleaved; j++)
                    s[(y * 4 + m) * bs + x * ncols_interleaved + j] = sumf[m][j];
            }
        }
    }
}

void ggml_gemm_q4_K_8x4_q8_K_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc) {
    const int qk = QK_K;
    const int nb = n / qk;
    const int ncols_interleaved = 8;
    const int blocklen = 4;
    static const uint32_t kmask1 = 0x3f3f3f3f;
    static const uint32_t kmask2 = 0x0f0f0f0f;
    static const uint32_t kmask3 = 0x03030303;

    assert (n % qk == 0);
    assert (nr % 4 == 0);
    assert (nc % ncols_interleaved == 0);

    UNUSED(nb);
    UNUSED(ncols_interleaved);
    UNUSED(blocklen);

    float sumf[4][8];
    float sum_minf[4][8];
    uint32_t utmp[32];
    int sumi1;
    int sumi2;
    int sumi;

    for (int y = 0; y < nr / 4; y++) {
        const block_q8_Kx4 * a_ptr = (const block_q8_Kx4 *) vy + (y * nb);
        for (int x = 0; x < nc / ncols_interleaved; x++) {
            const block_q4_Kx8 * b_ptr = (const block_q4_Kx8 *) vx + (x * nb);
            for (int m = 0; m < 4; m++) {
                for (int j = 0; j < ncols_interleaved; j++) {
                    sumf[m][j] = 0.0;
                    sum_minf[m][j] = 0.0;
                }
            }
            for (int l = 0; l < nb; l++) {
                for (int sb = 0; sb < 8; sb++) {
                    memcpy(utmp + sb * 4, b_ptr[l].scales + sb * 12, 12);
                    utmp[sb * 4 + 3] = ((utmp[sb * 4 + 2] >> 4) & kmask2) | (((utmp[sb * 4 + 1] >> 6) & kmask3) << 4);
                    const uint32_t uaux_0 = utmp[sb * 4 + 1] & kmask1;
                    utmp[sb * 4 + 1] = (utmp[sb * 4 + 2] & kmask2) | (((utmp[sb * 4 + 0] >> 6) & kmask3) << 4);
                    utmp[sb * 4 + 2] = uaux_0;
                    utmp[sb * 4 + 0] &= kmask1;
                }
                for (int k = 0; k < (qk / (2 * blocklen)); k++) {
                    uint8_t * scales_0 = (uint8_t *) utmp + (k / 8) * 32;
                    uint8_t * scales_1 = (uint8_t *) utmp + (k / 8) * 32 + 16;
                    for (int m = 0; m < 4; m++) {
                        for (int j = 0; j < ncols_interleaved; j++) {
                            sumi1 = 0;
                            sumi2 = 0;
                            sumi = 0;
                            for (int i = 0; i < blocklen; ++i) {
                                const int v0 = (int8_t) (b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] & 0xF);
                                const int v1 = (int8_t) (b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] >> 4);
                                sumi1 = (v0 * a_ptr[l].qs[(k / 8) * 256 + (k % 8) * 4 * blocklen + m * blocklen + i]);
                                sumi2 = (v1 * a_ptr[l].qs[(k / 8) * 256 + (k % 8) * 4 * blocklen + m * blocklen + i + 128]);
                                sumi1 = sumi1 * scales_0[j];
                                sumi2 = sumi2 * scales_1[j];
                                sumi += sumi1 + sumi2;
                            }
                            sumf[m][j] += sumi * GGML_CPU_FP16_TO_FP32(b_ptr[l].d[j]) * a_ptr[l].d[m];
                        }
                    }
                }
                for (int sb = 0; sb < 8; sb++) {
                    uint8_t * mins = (uint8_t *) utmp + 8 + sb * 16;
                    for(int m = 0; m < 4; m++) {
                        const int16_t * bsums = a_ptr[l].bsums + (sb * 8) + (m * 4) - ((sb % 2) * 6);
                        for(int j = 0; j < ncols_interleaved; j++) {
                            sum_minf[m][j] += mins[j] * (bsums[0] + bsums[1]) * GGML_CPU_FP16_TO_FP32(b_ptr[l].dmin[j]) * a_ptr[l].d[m];
                        }
                    }
                }
            }
            for (int m = 0; m < 4; m++) {
                for (int j = 0; j < ncols_interleaved; j++) {
                    s[(y * 4 + m) * bs + x * ncols_interleaved + j] = sumf[m][j] - sum_minf[m][j];
                }
            }
        }
    }
}

void ggml_gemm_q4_K_8x8_q8_K_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc) {
    const int qk = QK_K;
    const int nb = n / qk;
    const int ncols_interleaved = 8;
    const int blocklen = 8;
    static const uint32_t kmask1 = 0x3f3f3f3f;
    static const uint32_t kmask2 = 0x0f0f0f0f;
    static const uint32_t kmask3 = 0x03030303;

    assert (n % qk == 0);
    assert (nr % 4 == 0);
    assert (nc % ncols_interleaved == 0);

    UNUSED(bs);

    float sumf[4][8];
    float sum_minf[4][8];
    uint32_t utmp[32];
    int sumi1;
    int sumi2;
    int sumi;

    for (int y = 0; y < nr / 4; y++) {
        const block_q8_Kx4 * a_ptr = (const block_q8_Kx4 *) vy + (y * nb);
        for (int x = 0; x < nc / ncols_interleaved; x++) {
            const block_q4_Kx8 * b_ptr = (const block_q4_Kx8 *) vx + (x * nb);
            for (int m = 0; m < 4; m++) {
                for (int j = 0; j < ncols_interleaved; j++) {
                    sumf[m][j] = 0.0;
                    sum_minf[m][j] = 0.0;
                }
            }
            for (int l = 0; l < nb; l++) {
                for (int sb = 0; sb < 8; sb++) {
                    memcpy(utmp + sb * 4, b_ptr[l].scales + sb * 12, 12);
                    utmp[sb * 4 + 3] = ((utmp[sb * 4 + 2] >> 4) & kmask2) | (((utmp[sb * 4 + 1] >> 6) & kmask3) << 4);
                    const uint32_t uaux_0 = utmp[sb * 4 + 1] & kmask1;
                    utmp[sb * 4 + 1] = (utmp[sb * 4 + 2] & kmask2) | (((utmp[sb * 4 + 0] >> 6) & kmask3) << 4);
                    utmp[sb * 4 + 2] = uaux_0;
                    utmp[sb * 4 + 0] &= kmask1;
                }
                for (int k = 0; k < (qk / (2 * blocklen)); k++) {
                    uint8_t *scales_0 = (uint8_t*) utmp + (k / 4) * 32;
                    uint8_t *scales_1 = (uint8_t*) utmp + (k / 4) * 32 + 16;
                    for (int m = 0; m < 4; m++) {
                        for (int j = 0; j < ncols_interleaved; j++) {
                            sumi1 = 0;
                            sumi2 = 0;
                            sumi = 0;
                            for (int i = 0; i < blocklen; ++i) {
                                const int v0 = (int8_t) (b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] & 0xF);
                                const int v1 = (int8_t) (b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] >> 4);
                                sumi1 = (v0 * a_ptr[l].qs[(k >> 2) * 256 + (k % 4) * 4 * blocklen + m * blocklen + i]);
                                sumi2 = (v1 * a_ptr[l].qs[(k >> 2) * 256 + (k % 4) * 4 * blocklen + m * blocklen + i + 128]);
                                sumi1 = sumi1 * scales_0[j];
                                sumi2 = sumi2 * scales_1[j];
                                sumi += sumi1 + sumi2;
                            }
                            sumf[m][j] += sumi * GGML_CPU_FP16_TO_FP32(b_ptr[l].d[j]) * a_ptr[l].d[m];
                        }
                    }
                }
                for (int sb = 0; sb < 8; sb++) {
                    uint8_t *mins = (uint8_t*) utmp + 8 + sb * 16;
                    for(int m = 0; m < 4; m++) {
                        const int16_t *bsums = a_ptr[l].bsums + (sb * 8) + (m * 4) - ((sb % 2) * 6);
                        for(int j = 0; j < ncols_interleaved; j++) {
                            sum_minf[m][j] += mins[j] * (bsums[0] + bsums[1]) * GGML_CPU_FP16_TO_FP32(b_ptr[l].dmin[j]) * a_ptr[l].d[m];
                        }
                    }
                }
            }
            for (int m = 0; m < 4; m++) {
                for (int j = 0; j < ncols_interleaved; j++) {
                    s[(y * 4 + m) * bs + x * ncols_interleaved + j] = sumf[m][j] - sum_minf[m][j];
                }
            }
        }
    }
}

void ggml_gemm_q2_K_8x8_q8_K_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc) {
    const int qk = QK_K;
    const int nb = n / qk;
    const int ncols_interleaved = 8;
    const int blocklen = 8;

    assert (n % qk == 0);
    assert (nr % 4 == 0);
    assert (nc % ncols_interleaved == 0);

    UNUSED(s);
    UNUSED(bs);
    UNUSED(vx);
    UNUSED(vy);
    UNUSED(nr);
    UNUSED(nc);
    UNUSED(nb);
    UNUSED(ncols_interleaved);
    UNUSED(blocklen);

    float sumf[4][8];
    float sum_minf[4][8];
    int sumi1, sumi2, sumi3, sumi4;
    int sumi;

    for (int y = 0; y < nr / 4; y++) {
        const block_q8_Kx4 * a_ptr = (const block_q8_Kx4 *) vy + (y * nb);
        for (int x = 0; x < nc / ncols_interleaved; x++) {
            const block_q2_Kx8 * b_ptr = (const block_q2_Kx8 *) vx + (x * nb);
            for (int m = 0; m < 4; m++) {
                for (int j = 0; j < ncols_interleaved; j++) {
                    sumf[m][j] = 0.0;
                    sum_minf[m][j] = 0.0;
                }
            }
            for (int l = 0; l < nb; l++) {
                for (int k = 0; k < (qk / (4 * blocklen)); k++) {

                    const uint8_t *scales_0 = b_ptr[l].scales + (k / 4) * 64 ;
                    const uint8_t *scales_1 = b_ptr[l].scales + (k / 4) * 64 + 16;
                    const uint8_t *scales_2 = b_ptr[l].scales + (k / 4) * 64 + 32;
                    const uint8_t *scales_3 = b_ptr[l].scales + (k / 4) * 64 + 48;
                    for (int m = 0; m < 4; m++) {
                        for (int j = 0; j < ncols_interleaved; j++) {
                            sumi1 = 0;
                            sumi2 = 0;
                            sumi3 = 0;
                            sumi4 = 0;
                            sumi = 0;
                            int offset = ((k / 2) % 2) + j * 2;
                            for (int i = 0; i < blocklen; ++i){
                                const int v0 = (int8_t) (b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] & 3);
                                const int v1 = (int8_t) ((b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] >> 2 ) & 3);
                                const int v2 = (int8_t) ((b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] >> 4 ) & 3);
                                const int v3 = (int8_t) ((b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] >> 6 ) & 3);
                                sumi1 = (v0 * a_ptr[l].qs[(k >> 2) * 512 + (k % 4) * 4 * blocklen + m * blocklen + i]);
                                sumi2 = (v1 * a_ptr[l].qs[(k >> 2) * 512  + (k % 4) * 4 * blocklen + m * blocklen + i + 128]);
                                sumi3 = (v2 * a_ptr[l].qs[(k >> 2) * 512  + (k % 4) * 4 * blocklen + m * blocklen + i + 256]);
                                sumi4 = (v3 * a_ptr[l].qs[(k >> 2) * 512  + (k % 4) * 4 * blocklen + m * blocklen + i + 384]);
                                sumi1 = sumi1 * (scales_0[offset] & 0xF);
                                sumi2 = sumi2 * (scales_1[offset] & 0xF);
                                sumi3 = sumi3 * (scales_2[offset] & 0xF);
                                sumi4 = sumi4 * (scales_3[offset] & 0xF);
                                sumi += sumi1 + sumi2 + sumi3 + sumi4;
                            }
                            sumf[m][j] += sumi * GGML_FP16_TO_FP32(b_ptr[l].d[j]) * a_ptr[l].d[m];
                        }
                    }
                }
                for(int sb = 0; sb < 8; sb++) {
                    const uint8_t *mins = b_ptr[l].scales + sb * 16;
                    for(int m = 0; m < 4; m++) {
                        const int16_t *bsums = a_ptr[l].bsums + (sb * 8) + (m * 4) - ((sb % 2) *  6);
                        for(int j = 0; j < ncols_interleaved; j++) {
                            int mins_prod = ((mins[j * 2] >> 4) * bsums[0] + (mins[(j * 2)+ 1] >> 4) * bsums[1]);
                            sum_minf[m][j] += (mins_prod) * GGML_FP16_TO_FP32(b_ptr[l].dmin[j]) * a_ptr[l].d[m];
                        }
                    }
                }
            }

            for (int m = 0; m < 4; m++) {
                for (int j = 0; j < ncols_interleaved; j++) {
                    s[(y * 4 + m) * bs + x * ncols_interleaved + j] = sumf[m][j] - sum_minf[m][j];
                }
            }
        }
    }
}

void ggml_gemm_q3_K_8x8_q8_K_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc) {
    const int qk = QK_K;
    const int nb = n / qk;
    const int ncols_interleaved = 8;

    assert (n % qk == 0);
    assert (nr % 4 == 0);
    assert (nc % ncols_interleaved == 0);

    UNUSED(bs);

    float sumf[4][8];
    // Sub block scales of the 8 interleaved Q3_K structures, unpacked to 6-bit values: sc[sb][j]
    uint8_t sc[16][8];

    for (int y = 0; y < nr / 4; y++) {
        const block_q8_Kx4 * a_ptr = (const block_q8_Kx4 *) vy + (y * nb);
        for (int x = 0; x < nc / ncols_interleaved; x++) {
            const block_q3_Kx8 * b_ptr = (const block_q3_Kx8 *) vx + (x * nb);
            for (int m = 0; m < 4; m++) {
                for (int j = 0; j < ncols_interleaved; j++) {
                    sumf[m][j] = 0.0;
                }
            }
            for (int l = 0; l < nb; l++) {
                // The packed scales are stored transposed (scales byte i of block j at scales[i*8 + j]),
                // unpack them with the regular Q3_K bit fiddling to one 6-bit value per sub block
                const uint8_t * scp = b_ptr[l].scales;
                for (int j = 0; j < ncols_interleaved; j++) {
                    for (int k = 0; k < 4; k++) {
                        sc[k     ][j] = (scp[ k     * 8 + j] & 15) | (( scp[(8 + k) * 8 + j]       & 3) << 4);
                        sc[4 + k ][j] = (scp[(4 + k) * 8 + j] & 15) | (((scp[(8 + k) * 8 + j] >> 2) & 3) << 4);
                        sc[8 + k ][j] = ((scp[k      * 8 + j] >> 4) & 15) | (((scp[(8 + k) * 8 + j] >> 4) & 3) << 4);
                        sc[12 + k][j] = ((scp[(4 + k) * 8 + j] >> 4) & 15) | (((scp[(8 + k) * 8 + j] >> 6) & 3) << 4);
                    }
                }
                for (int m = 0; m < 4; m++) {
                    for (int j = 0; j < ncols_interleaved; j++) {
                        int32_t isum = 0;
                        for (int sb = 0; sb < 16; sb++) {
                            // Sub block sb of block j: the 2-bit fields of its 16 elements are at shift
                            // 2*jj of 16 consecutive bytes, the high bits at bit h2*4+jj of 16 consecutive bytes
                            const int h2 = sb / 8, jj = (sb % 8) / 2, c = sb % 2;
                            const uint8_t * q2 = b_ptr[l].qs + (h2 * 4 + c * 2) * 64 + j * 8;
                            const uint8_t * hm = b_ptr[l].hmask + (c * 2) * 64 + j * 8;
                            // block_q8_Kx4 quants are interleaved in chunks of 8 bytes per row
                            const int8_t  * q8 = a_ptr[l].qs + sb * 64 + m * 8;
                            // The dequantized value is scale * (u - 4) with u = q2 + 4*h in [0, 7];
                            // the -4 part is folded in with the q8_K sub block sums
                            int32_t dot = -4 * (int32_t) a_ptr[l].bsums[(sb / 4) * 16 + m * 4 + (sb % 4)];
                            for (int i = 0; i < 16; i++) {
                                const int u = ((q2[(i / 8) * 64 + (i % 8)] >> (2 * jj)) & 3)
                                            + (((hm[(i / 8) * 64 + (i % 8)] >> (h2 * 4 + jj)) & 1) << 2);
                                dot += u * q8[(i / 8) * 32 + (i % 8)];
                            }
                            isum += ((int) sc[sb][j] - 32) * dot;
                        }
                        sumf[m][j] += isum * GGML_FP16_TO_FP32(b_ptr[l].d[j]) * a_ptr[l].d[m];
                    }
                }
            }
            for (int m = 0; m < 4; m++) {
                for (int j = 0; j < ncols_interleaved; j++) {
                    s[(y * 4 + m) * bs + x * ncols_interleaved + j] = sumf[m][j];
                }
            }
        }
    }
}

void ggml_gemm_q5_K_8x4_q8_K_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc) {
    ggml_gemm_q5_K_NxM_q8_K_generic_impl<4, 8>(n, s, bs, vx, vy, nr, nc);
}

void ggml_gemm_q5_K_8x8_q8_K_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc) {
    ggml_gemm_q5_K_NxM_q8_K_generic_impl<8, 8>(n, s, bs, vx, vy, nr, nc);
}

void ggml_gemm_q6_K_8x4_q8_K_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc) {
    ggml_gemm_q6_K_NxM_q8_K_generic_impl<4, 8>(n, s, bs, vx, vy, nr, nc);
}

void ggml_gemm_q6_K_8x8_q8_K_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc) {
   ggml_gemm_q6_K_NxM_q8_K_generic_impl<8, 8>(n, s, bs, vx, vy, nr, nc);
}

void ggml_gemm_iq1_s_8x8_q8_K_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc) {
    const int qk = QK_K;
    const int nb = n / qk;
    const int ncols_interleaved = 8;

    assert(n % qk == 0);
    assert(nr % 4 == 0);
    assert(nc % ncols_interleaved == 0);

    const block_iq1_sx8 * GGML_RESTRICT x = (const block_iq1_sx8 *) vx;
    const block_q8_Kx4  * GGML_RESTRICT y = (const block_q8_Kx4 *) vy;

    float sumf[4][8];

    for (int yy = 0; yy < nr / 4; yy++) {
        const block_q8_Kx4 * a_ptr = y + (size_t) yy * nb;
        for (int cx = 0; cx < nc / ncols_interleaved; cx++) {
            const block_iq1_sx8 * b_ptr = x + (size_t) cx * nb;
            for (int m = 0; m < 4; m++) {
                for (int j = 0; j < 8; j++) {
                    sumf[m][j] = 0.0f;
                }
            }
            for (int i = 0; i < nb; i++) {
                for (int m = 0; m < 4; m++) {
                    for (int j = 0; j < 8; j++) {
                        int sumi = 0, sumi1 = 0;
                        for (int ib = 0; ib < QK_K/32; ++ib) {
                            const uint16_t h = *(const uint16_t *)(b_ptr[i].qh + ib * 16 + j * 2);
                            const int ls    = 2*((h >> 12) & 7) + 1;
                            const int delta = h & 0x8000 ? -1 : 1;
                            int lsum = 0, lsum2 = 0;
                            for (int l = 0; l < 4; ++l) {
                                const int     t    = ib * 4 + l;
                                const uint8_t q    = b_ptr[i].qs[(t / 8) * 64 + j * 8 + t % 8];
                                const int8_t * grid = (const int8_t *)(iq1s_grid + (q | (((h >> 3*l) & 7) << 8)));
                                const int8_t * q8   = a_ptr[i].qs + (t / 2) * 64 + (t % 2) * 32 + m * 8;
                                for (int k = 0; k < 8; ++k) {
                                    lsum  += q8[k] * grid[k];
                                    lsum2 += q8[k];
                                }
                            }
                            sumi  += ls * lsum;
                            sumi1 += ls * delta * lsum2;
                        }
                        sumf[m][j] += GGML_CPU_FP16_TO_FP32(b_ptr[i].d[j]) * a_ptr[i].d[m] * (sumi + IQ1S_DELTA * sumi1);
                    }
                }
            }
            for (int m = 0; m < 4; m++) {
                for (int j = 0; j < 8; j++) {
                    s[(yy * 4 + m) * bs + cx * ncols_interleaved + j] = sumf[m][j];
                }
            }
        }
    }
}

void ggml_gemm_iq1_m_8x8_q8_K_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc) {
    const int qk = QK_K;
    const int nb = n / qk;
    const int ncols_interleaved = 8;

    assert(n % qk == 0);
    assert(nr % 4 == 0);
    assert(nc % ncols_interleaved == 0);

    const block_iq1_mx8 * GGML_RESTRICT x = (const block_iq1_mx8 *) vx;
    const block_q8_Kx4  * GGML_RESTRICT y = (const block_q8_Kx4 *) vy;

    iq1m_scale_t scale;

    float sumf[4][8];

    for (int yy = 0; yy < nr / 4; yy++) {
        const block_q8_Kx4 * a_ptr = y + (size_t) yy * nb;
        for (int cx = 0; cx < nc / ncols_interleaved; cx++) {
            const block_iq1_mx8 * b_ptr = x + (size_t) cx * nb;
            for (int m = 0; m < 4; m++) {
                for (int j = 0; j < 8; j++) {
                    sumf[m][j] = 0.0f;
                }
            }
            for (int i = 0; i < nb; i++) {
                for (int m = 0; m < 4; m++) {
                    for (int j = 0; j < 8; j++) {
                        const uint16_t * scj = (const uint16_t *)(b_ptr[i].scales + j * 8);
                        scale.u16 = (scj[0] >> 12) | ((scj[1] >> 8) & 0x00f0) | ((scj[2] >> 4) & 0x0f00) | (scj[3] & 0xf000);

                        int sumi1 = 0, sumi2 = 0;
                        for (int ib = 0; ib < QK_K/32; ++ib) {
                            const uint8_t h0 = b_ptr[i].qh[ib * 16 + j * 2 + 0];
                            const uint8_t h1 = b_ptr[i].qh[ib * 16 + j * 2 + 1];
                            const int delta[4] = { h0 & 0x08 ? -1 : 1, h0 & 0x80 ? -1 : 1, h1 & 0x08 ? -1 : 1, h1 & 0x80 ? -1 : 1 };
                            int sum1[2] = { 0, 0 }, sum2[2] = { 0, 0 };
                            for (int l = 0; l < 4; ++l) {
                                const int     t    = ib * 4 + l;
                                const uint8_t q    = b_ptr[i].qs[(t / 8) * 64 + j * 8 + t % 8];
                                const uint8_t hq   = l < 2 ? h0 : h1;
                                const int8_t * grid = (const int8_t *)(iq1s_grid + (q | (((uint16_t)hq << (8 - 4*(l%2))) & 0x700)));
                                const int8_t * q8   = a_ptr[i].qs + (t / 2) * 64 + (t % 2) * 32 + m * 8;
                                int lsum1 = 0, lsum2 = 0;
                                for (int k = 0; k < 8; ++k) {
                                    lsum1 += q8[k] * grid[k];
                                    lsum2 += q8[k];
                                }
                                sum1[l/2] += lsum1;
                                sum2[l/2] += lsum2*delta[l];
                            }
                            const int ls1 = 2*((scj[ib/2] >> (6*(ib%2)+0)) & 0x7) + 1;
                            const int ls2 = 2*((scj[ib/2] >> (6*(ib%2)+3)) & 0x7) + 1;
                            sumi1 += sum1[0] * ls1 + sum1[1] * ls2;
                            sumi2 += sum2[0] * ls1 + sum2[1] * ls2;
                        }
                        sumf[m][j] += GGML_CPU_FP16_TO_FP32(scale.f16) * a_ptr[i].d[m] * (sumi1 + IQ1M_DELTA * sumi2);
                    }
                }
            }
            for (int m = 0; m < 4; m++) {
                for (int j = 0; j < 8; j++) {
                    s[(yy * 4 + m) * bs + cx * ncols_interleaved + j] = sumf[m][j];
                }
            }
        }
    }
}

void ggml_gemm_iq2_xs_8x8_q8_K_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc) {
    const int qk = QK_K;
    const int nb = n / qk;
    const int ncols_interleaved = 8;

    assert(n % qk == 0);
    assert(nr % 4 == 0);
    assert(nc % ncols_interleaved == 0);

    const block_iq2_xsx8 * GGML_RESTRICT x = (const block_iq2_xsx8 *) vx;
    const block_q8_Kx4   * GGML_RESTRICT y = (const block_q8_Kx4 *) vy;

    float sumf[4][8];

    for (int yy = 0; yy < nr / 4; yy++) {
        const block_q8_Kx4 * a_ptr = y + (size_t) yy * nb;
        for (int cx = 0; cx < nc / ncols_interleaved; cx++) {
            const block_iq2_xsx8 * b_ptr = x + (size_t) cx * nb;
            for (int m = 0; m < 4; m++) {
                for (int j = 0; j < 8; j++) {
                    sumf[m][j] = 0.0f;
                }
            }
            for (int i = 0; i < nb; i++) {
                for (int m = 0; m < 4; m++) {
                    for (int j = 0; j < 8; j++) {
                        int bsum = 0;
                        for (int g = 0; g < QK_K/32; ++g) {
                            const uint8_t  sc  = b_ptr[i].scales[g*8 + j];
                            const int      ls1 = 1 + 2*(sc & 0xf);
                            const int      ls2 = 1 + 2*(sc >> 4);
                            const uint16_t * q2 = b_ptr[i].qs + g*32 + j*4;
                            int sumi1 = 0, sumi2 = 0;
                            for (int l = 0; l < 4; ++l) {
                                const int       t     = g*4 + l; // grid entry index within the block
                                const uint8_t * grid  = (const uint8_t *)(iq2xs_grid + (q2[l] & 511));
                                const uint8_t   signs = ksigns_iq2xs[q2[l] >> 9];
                                const int8_t  * q8    = a_ptr[i].qs + (t / 2) * 64 + (t % 2) * 32 + m * 8;
                                int lsum = 0;
                                for (int k = 0; k < 8; ++k) {
                                    lsum += grid[k] * q8[k] * (signs & kmask_iq2xs[k] ? -1 : 1);
                                }
                                if (l < 2) {
                                    sumi1 += lsum;
                                } else {
                                    sumi2 += lsum;
                                }
                            }
                            bsum += ls1 * sumi1 + ls2 * sumi2;
                        }
                        sumf[m][j] += GGML_CPU_FP16_TO_FP32(b_ptr[i].d[j]) * a_ptr[i].d[m] * bsum;
                    }
                }
            }
            for (int m = 0; m < 4; m++) {
                for (int j = 0; j < 8; j++) {
                    s[(yy * 4 + m) * bs + cx * ncols_interleaved + j] = 0.125f * sumf[m][j];
                }
            }
        }
    }
}

void ggml_gemm_iq3_xxs_8x8_q8_K_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc) {
    const int qk = QK_K;
    const int nb = n / qk;
    const int ncols_interleaved = 8;

    assert(n % qk == 0);
    assert(nr % 4 == 0);
    assert(nc % ncols_interleaved == 0);

    const block_iq3_xxsx8 * GGML_RESTRICT x = (const block_iq3_xxsx8 *) vx;
    const block_q8_Kx4    * GGML_RESTRICT y = (const block_q8_Kx4 *) vy;

    uint32_t aux32;

    float sumf[4][8];

    for (int yy = 0; yy < nr / 4; yy++) {
        const block_q8_Kx4 * a_ptr = y + (size_t) yy * nb;
        for (int cx = 0; cx < nc / ncols_interleaved; cx++) {
            const block_iq3_xxsx8 * b_ptr = x + (size_t) cx * nb;
            for (int m = 0; m < 4; m++) {
                for (int j = 0; j < 8; j++) {
                    sumf[m][j] = 0.0f;
                }
            }
            for (int i = 0; i < nb; i++) {
                for (int m = 0; m < 4; m++) {
                    for (int j = 0; j < 8; j++) {
                        int bsum = 0;
                        for (int g = 0; g < QK_K/32; ++g) {
                            memcpy(&aux32, b_ptr[i].gas + g*32 + j*4, sizeof(uint32_t));
                            const uint32_t ls = 2*(aux32 >> 28) + 1;
                            const uint8_t * q3 = b_ptr[i].qs + g*64 + j*8;
                            int sumi = 0;
                            for (int l = 0; l < 4; ++l) {
                                const uint8_t * grid1 = (const uint8_t *)(iq3xxs_grid + q3[2*l+0]);
                                const uint8_t * grid2 = (const uint8_t *)(iq3xxs_grid + q3[2*l+1]);
                                const uint8_t   signs = ksigns_iq2xs[(aux32 >> 7*l) & 127];
                                const int8_t  * q8    = a_ptr[i].qs + g*128 + l*32 + m*8;
                                for (int k = 0; k < 4; ++k) {
                                    sumi += grid1[k] * q8[k+0] * (signs & kmask_iq2xs[k+0] ? -1 : 1);
                                    sumi += grid2[k] * q8[k+4] * (signs & kmask_iq2xs[k+4] ? -1 : 1);
                                }
                            }
                            bsum += sumi * ls;
                        }
                        sumf[m][j] += GGML_CPU_FP16_TO_FP32(b_ptr[i].d[j]) * a_ptr[i].d[m] * bsum;
                    }
                }
            }
            for (int m = 0; m < 4; m++) {
                for (int j = 0; j < 8; j++) {
                    s[(yy * 4 + m) * bs + cx * ncols_interleaved + j] = 0.25f * sumf[m][j];
                }
            }
        }
    }
}

void ggml_gemm_iq4_nl_4x4_q8_0_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc) {
    const int qk = QK8_0;
    const int nb = n / qk;
    const int ncols_interleaved = 4;
    const int blocklen = 4;

    assert (n % qk == 0);
    assert (nr % 4 == 0);
    assert (nc % ncols_interleaved == 0);

    UNUSED(s);
    UNUSED(bs);
    UNUSED(vx);
    UNUSED(vy);
    UNUSED(nr);
    UNUSED(nc);
    UNUSED(nb);
    UNUSED(ncols_interleaved);
    UNUSED(blocklen);

    {
        float sumf[4][4];
        int sumi;

        for (int y = 0; y < nr / 4; y++) {
            const block_q8_0x4 * a_ptr = (const block_q8_0x4 *) vy + (y * nb);
            for (int x = 0; x < nc / ncols_interleaved; x++) {
                const block_iq4_nlx4 * b_ptr = (const block_iq4_nlx4 *) vx + (x * nb);
                for (int m = 0; m < 4; m++) {
                    for (int j = 0; j < ncols_interleaved; j++) sumf[m][j] = 0.0;
                }
                for (int l = 0; l < nb; l++) {
                    for (int k = 0; k < (qk / (2 * blocklen)); k++) {
                        for (int m = 0; m < 4; m++) {
                            for (int j = 0; j < ncols_interleaved; j++) {
                                sumi = 0;
                                for (int i = 0; i < blocklen; ++i) {
                                    const int v0 = kvalues_iq4nl[b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] & 0x0F];
                                    const int v1 = kvalues_iq4nl[b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] >> 4];
                                    sumi += ((v0 * a_ptr[l].qs[k * 4 * blocklen + m * blocklen + i]) +
                                            (v1 * a_ptr[l].qs[k * 4 * blocklen + m * blocklen + i + qk / 2 * 4]));
                                }
                                sumf[m][j] += sumi * GGML_CPU_FP16_TO_FP32(b_ptr[l].d[j]) * GGML_CPU_FP16_TO_FP32(a_ptr[l].d[m]);
                            }
                        }
                    }
                }
                for (int m = 0; m < 4; m++) {
                    for (int j = 0; j < ncols_interleaved; j++)
                        s[(y * 4 + m) * bs + x * ncols_interleaved + j] = sumf[m][j];
                }
            }
        }
    }
}

void ggml_gemm_iq4_nl_8x8_q8_0_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc) {
    const int qk = QK8_0;
    const int nb = n / qk;
    const int ncols_interleaved = 8;
    const int blocklen = 8;

    assert(n % qk == 0);
    assert(nr % 4 == 0);
    assert(nc % ncols_interleaved == 0);

    float sumf[4][8];
    int sumi;

    for (int y = 0; y < nr / 4; y++) {
        const block_q8_0x4 * a_ptr = (const block_q8_0x4 *) vy + (y * nb);
        for (int x = 0; x < nc / ncols_interleaved; x++) {
            const block_iq4_nlx8 * b_ptr = (const block_iq4_nlx8 *) vx + (x * nb);
            for (int m = 0; m < 4; m++) {
                for (int j = 0; j < ncols_interleaved; j++) sumf[m][j] = 0.0;
            }
            for (int l = 0; l < nb; l++) {
                for (int k = 0; k < (qk / (2 * blocklen)); k++) {
                    for (int m = 0; m < 4; m++) {
                        for (int j = 0; j < ncols_interleaved; j++) {
                            sumi = 0;
                            for (int i = 0; i < blocklen; ++i) {
                                const int v0 = kvalues_iq4nl[b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] & 0x0F];
                                const int v1 = kvalues_iq4nl[b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] >> 4];
                                sumi += ((v0 * a_ptr[l].qs[k * 4 * blocklen + m * blocklen + i]) +
                                         (v1 * a_ptr[l].qs[k * 4 * blocklen + m * blocklen + i + qk / 2 * 4]));
                            }
                            sumf[m][j] += sumi * GGML_CPU_FP16_TO_FP32(b_ptr[l].d[j]) * GGML_CPU_FP16_TO_FP32(a_ptr[l].d[m]);
                        }
                    }
                }
            }
            for (int m = 0; m < 4; m++) {
                for (int j = 0; j < ncols_interleaved; j++)
                    s[(y * 4 + m) * bs + x * ncols_interleaved + j] = sumf[m][j];
            }
        }
    }
}

void ggml_gemm_mxfp4_4x4_q8_0_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc) {
    const int qk = QK8_0;
    const int nb = n / qk;
    const int ncols_interleaved = 4;
    const int blocklen = 4;

    assert(n % qk == 0);
    assert(nr % 4 == 0);
    assert(nc % ncols_interleaved == 0);

    float sumf[4][4];
    int sumi;

    for (int y = 0; y < nr / 4; y++) {
        const block_q8_0x4 * a_ptr = (const block_q8_0x4 *) vy + (y * nb);
        for (int x = 0; x < nc / ncols_interleaved; x++) {
            const block_mxfp4x4 * b_ptr = (const block_mxfp4x4 *) vx + (x * nb);
            for (int m = 0; m < 4; m++) {
                for (int j = 0; j < ncols_interleaved; j++) sumf[m][j] = 0.0;
            }
            for (int l = 0; l < nb; l++) {
                for (int k = 0; k < (qk / (2 * blocklen)); k++) {
                    for (int m = 0; m < 4; m++) {
                        for (int j = 0; j < ncols_interleaved; j++) {
                            sumi = 0;
                            for (int i = 0; i < blocklen; ++i) {
                                const int v0 = kvalues_mxfp4[b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] & 0x0F];
                                const int v1 = kvalues_mxfp4[b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] >> 4];
                                sumi += ((v0 * a_ptr[l].qs[k * 4 * blocklen + m * blocklen + i]) +
                                         (v1 * a_ptr[l].qs[k * 4 * blocklen + m * blocklen + i + qk / 2 * 4]));
                            }
                            sumf[m][j] += sumi * GGML_CPU_E8M0_TO_FP32_HALF(b_ptr[l].e[j]) * GGML_CPU_FP16_TO_FP32(a_ptr[l].d[m]);
                        }
                    }
                }
            }
            for (int m = 0; m < 4; m++) {
                for (int j = 0; j < ncols_interleaved; j++)
                    s[(y * 4 + m) * bs + x * ncols_interleaved + j] = sumf[m][j];
            }
        }
    }
}

void ggml_gemm_mxfp4_8x8_q8_0_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc) {
    const int qk = QK8_0;
    const int nb = n / qk;
    const int ncols_interleaved = 8;
    const int blocklen = 8;

    assert(n % qk == 0);
    assert(nr % 4 == 0);
    assert(nc % ncols_interleaved == 0);

    float sumf[4][8];
    int sumi;

    for (int y = 0; y < nr / 4; y++) {
        const block_q8_0x4 * a_ptr = (const block_q8_0x4 *) vy + (y * nb);
        for (int x = 0; x < nc / ncols_interleaved; x++) {
            const block_mxfp4x8 * b_ptr = (const block_mxfp4x8 *) vx + (x * nb);
            for (int m = 0; m < 4; m++) {
                for (int j = 0; j < ncols_interleaved; j++) sumf[m][j] = 0.0;
            }
            for (int l = 0; l < nb; l++) {
                for (int k = 0; k < (qk / (2 * blocklen)); k++) {
                    for (int m = 0; m < 4; m++) {
                        for (int j = 0; j < ncols_interleaved; j++) {
                            sumi = 0;
                            for (int i = 0; i < blocklen; ++i) {
                                const int v0 = kvalues_mxfp4[b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] & 0x0F];
                                const int v1 = kvalues_mxfp4[b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] >> 4];
                                sumi += ((v0 * a_ptr[l].qs[k * 4 * blocklen + m * blocklen + i]) +
                                         (v1 * a_ptr[l].qs[k * 4 * blocklen + m * blocklen + i + qk / 2 * 4]));
                            }
                            sumf[m][j] += sumi * GGML_CPU_E8M0_TO_FP32_HALF(b_ptr[l].e[j]) * GGML_CPU_FP16_TO_FP32(a_ptr[l].d[m]);
                        }
                    }
                }
            }
            for (int m = 0; m < 4; m++) {
                for (int j = 0; j < ncols_interleaved; j++)
                    s[(y * 4 + m) * bs + x * ncols_interleaved + j] = sumf[m][j];
            }
        }
    }
}

void ggml_gemm_q8_0_4x4_q8_0_generic(int                        n,
                                     float * GGML_RESTRICT      s,
                                     size_t                     bs,
                                     const void * GGML_RESTRICT vx,
                                     const void * GGML_RESTRICT vy,
                                     int                        nr,
                                     int                        nc) {
    const int qk                = QK8_0;
    const int nb                = n / qk;
    const int ncols_interleaved = 4;
    const int blocklen          = 4;

    assert(n % qk == 0);
    assert(nr % 4 == 0);
    assert(nc % ncols_interleaved == 0);

    float sumf[4][4];
    int   sumi;

    for (int y = 0; y < nr / 4; y++) {
        const block_q8_0x4 * a_ptr = (const block_q8_0x4 *) vy + (y * nb);
        for (int x = 0; x < nc / ncols_interleaved; x++) {
            const block_q8_0x4 * b_ptr = (const block_q8_0x4 *) vx + (x * nb);
            for (int m = 0; m < 4; m++) {
                for (int j = 0; j < ncols_interleaved; j++) {
                    sumf[m][j] = 0.0;
                }
            }
            for (int l = 0; l < nb; l++) {
                for (int k = 0; k < (qk / blocklen); k++) {
                    for (int m = 0; m < 4; m++) {
                        for (int j = 0; j < ncols_interleaved; j++) {
                            sumi = 0;
                            for (int i = 0; i < blocklen; ++i) {
                                const int v0 = b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i];
                                sumi += v0 * a_ptr[l].qs[k * 4 * blocklen + m * blocklen + i];
                            }
                            sumf[m][j] +=
                                sumi * GGML_CPU_FP16_TO_FP32(b_ptr[l].d[j]) * GGML_CPU_FP16_TO_FP32(a_ptr[l].d[m]);
                        }
                    }
                }
            }
            for (int m = 0; m < 4; m++) {
                for (int j = 0; j < ncols_interleaved; j++) {
                    s[(y * 4 + m) * bs + x * ncols_interleaved + j] = sumf[m][j];
                }
            }
        }
    }
}



void ggml_gemm_q8_0_4x8_q8_0_generic(int                        n,
                                     float * GGML_RESTRICT      s,
                                     size_t                     bs,
                                     const void * GGML_RESTRICT vx,
                                     const void * GGML_RESTRICT vy,
                                     int                        nr,
                                     int                        nc) {
    const int qk                = QK8_0;
    const int nb                = n / qk;
    const int ncols_interleaved = 4;
    const int blocklen          = 8;

    assert(n % qk == 0);
    assert(nr % 4 == 0);
    assert(nc % ncols_interleaved == 0);

    float sumf[4][4];
    int   sumi;

    for (int y = 0; y < nr / 4; y++) {
        const block_q8_0x4 * a_ptr = (const block_q8_0x4 *) vy + (y * nb);
        for (int x = 0; x < nc / ncols_interleaved; x++) {
            const block_q8_0x4 * b_ptr = (const block_q8_0x4 *) vx + (x * nb);
            for (int m = 0; m < 4; m++) {
                for (int j = 0; j < ncols_interleaved; j++) {
                    sumf[m][j] = 0.0;
                }
            }
            for (int l = 0; l < nb; l++) {
                for (int k = 0; k < (qk / blocklen); k++) {
                    for (int m = 0; m < 4; m++) {
                        for (int j = 0; j < ncols_interleaved; j++) {
                            sumi = 0;
                            for (int i = 0; i < blocklen; ++i) {
                                const int v0 = b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i];
                                sumi += v0 * a_ptr[l].qs[k * 4 * blocklen + m * blocklen + i];
                            }
                            sumf[m][j] +=
                                sumi * GGML_CPU_FP16_TO_FP32(b_ptr[l].d[j]) * GGML_CPU_FP16_TO_FP32(a_ptr[l].d[m]);
                        }
                    }
                }
            }
            for (int m = 0; m < 4; m++) {
                for (int j = 0; j < ncols_interleaved; j++) {
                    s[(y * 4 + m) * bs + x * ncols_interleaved + j] = sumf[m][j];
                }
            }
        }
    }
}

// Only enable these for RISC-V.
#if defined __riscv_zvfh
void ggml_gemm_q4_0_16x1_q8_0_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc) {
    const int qk = QK8_0;
    const int nb = n / qk;
    const int ncols_interleaved = 16;
    const int blocklen = 1;

    assert (n % qk == 0);
    assert (nr % 4 == 0);
    assert (nc % ncols_interleaved == 0);

    UNUSED(s);
    UNUSED(bs);
    UNUSED(vx);
    UNUSED(vy);
    UNUSED(nr);
    UNUSED(nc);
    UNUSED(nb);
    UNUSED(ncols_interleaved);
    UNUSED(blocklen);

    float sumf[4][16];
    int sumi;

    for (int y = 0; y < nr / 4; y++) {
        const block_q8_0x4 * a_ptr = (const block_q8_0x4 *) vy + (y * nb);
        for (int x = 0; x < nc / ncols_interleaved; x++) {
            const block_q4_0x16 * b_ptr = (const block_q4_0x16 *) vx + (x * nb);
            for (int m = 0; m < 4; m++) {
                for (int j = 0; j < ncols_interleaved; j++) sumf[m][j] = 0.0;
            }
            for (int l = 0; l < nb; l++) {
                for (int k = 0; k < (qk / (2 * blocklen)); k++) {
                    for (int m = 0; m < 4; m++) {
                        for (int j = 0; j < ncols_interleaved; j++) {
                            sumi = 0;
                            for (int i = 0; i < blocklen; ++i) {
                                const int v0 = (int8_t) (b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] << 4);
                                const int v1 = (int8_t) (b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] & 0xF0);
                                sumi += ((v0 * a_ptr[l].qs[k * 4 * blocklen + m * blocklen + i]) +
                                         (v1 * a_ptr[l].qs[k * 4 * blocklen + m * blocklen + i + qk / 2 * 4])) >> 4;
                            }
                            sumf[m][j] += sumi * GGML_CPU_FP16_TO_FP32(b_ptr[l].d[j]) * GGML_CPU_FP16_TO_FP32(a_ptr[l].d[m]);
                        }
                    }
                }
            }
            for (int m = 0; m < 4; m++) {
                for (int j = 0; j < ncols_interleaved; j++)
                    s[(y * 4 + m) * bs + x * ncols_interleaved + j] = sumf[m][j];
            }
        }
    }
}

void ggml_gemm_q4_K_16x1_q8_K_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc) {
    const int qk = QK_K;
    const int nb = n / qk;
    const int ncols_interleaved = 16;
    const int blocklen = 1;

    assert (n % qk == 0);
    assert (nr % 4 == 0);
    assert (nc % ncols_interleaved == 0);

    UNUSED(s);
    UNUSED(bs);
    UNUSED(vx);
    UNUSED(vy);
    UNUSED(nr);
    UNUSED(nc);
    UNUSED(nb);
    UNUSED(ncols_interleaved);
    UNUSED(blocklen);

    float sumf[4][16];
    float sum_minf[4][16];
    uint8_t scales[128];
    uint8_t mins[128];
    int sumi1;
    int sumi2;
    int sumi;

    for (int y = 0; y < nr / 4; y++) {
        const block_q8_Kx4 * a_ptr = (const block_q8_Kx4 *) vy + (y * nb);
        for (int x = 0; x < nc / ncols_interleaved; x++) {
            const block_q4_Kx16 * b_ptr = (const block_q4_Kx16 *) vx + (x * nb);
            for (int m = 0; m < 4; m++) {
                for (int j = 0; j < ncols_interleaved; j++) {
                    sumf[m][j] = 0.0;
                    sum_minf[m][j] = 0.0;
                }
            }
            for (int l = 0; l < nb; l++) {
                for (int i = 0; i < 128; i++) {
                    scales[i] = b_ptr[l].scales[i] & 0x0F;
                    mins[i] = b_ptr[l].scales[i] >> 4;
                }
                for (int i = 0; i < 64; i++) {
                    scales[i] |= (b_ptr[l].scales[128 + i] & 0x03) << 4;
                    mins[i] |= (b_ptr[l].scales[128 + i] & 0x0C) << 2;
                    scales[i + 64] |= (b_ptr[l].scales[128 + i] & 0x30);
                    mins[i + 64] |= (b_ptr[l].scales[128 + i] & 0xC0) >> 2;
                }

                for (int sb = 0; sb < 8; sb++) {
                    uint8_t *min = &mins[sb * 16];
                    for(int m = 0; m < 4; m++) {
                        const int16_t bsums = a_ptr[l].bsums[sb * 8 + m] + a_ptr[l].bsums[sb * 8 + m + 4];
                        for(int j = 0; j < ncols_interleaved; j++) {
                            sum_minf[m][j] += min[j] * bsums * GGML_CPU_FP16_TO_FP32(b_ptr[l].dmin[j]) * a_ptr[l].d[m];
                        }
                    }
                }

                for (int sb = 0; sb < 8; sb += 2) {
                    uint8_t *scales_0 = &scales[sb * 16];
                    uint8_t *scales_1 = &scales[(sb + 1) * 16];

                    for (int i = 0; i < QK4_0; i++) {
                        for (int m = 0; m < 4; m++) {
                            for (int j = 0; j < ncols_interleaved; j++) {
                                sumi1 = 0;
                                sumi2 = 0;
                                sumi = 0;

                                const int v0 = (int8_t) (b_ptr[l].qs[sb * 256 + i * 16 + j] & 0xF);
                                const int v1 = (int8_t) (b_ptr[l].qs[sb * 256 + i * 16 + j] >> 4);
                                sumi1 = (v0 * a_ptr[l].qs[sb * 4 * 32 + i * 4 + m]);
                                sumi2 = (v1 * a_ptr[l].qs[sb * 4 * 32 + 32 * 4 + i * 4 + m]);
                                sumi1 = sumi1 * scales_0[j];
                                sumi2 = sumi2 * scales_1[j];
                                sumi += sumi1 + sumi2;

                                sumf[m][j] += sumi * GGML_CPU_FP16_TO_FP32(b_ptr[l].d[j]) * a_ptr[l].d[m];
                            }
                        }
                    }
                }
            }
            for (int m = 0; m < 4; m++) {
                for (int j = 0; j < ncols_interleaved; j++) {
                    s[(y * 4 + m) * bs + x * ncols_interleaved + j] = sumf[m][j] - sum_minf[m][j];
                }
            }
        }
    }
}

void ggml_gemm_iq4_nl_16x1_q8_0_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc) {
    const int qk = QK8_0;
    const int nb = n / qk;
    const int ncols_interleaved = 16;
    const int blocklen = 1;

    assert(n % qk == 0);
    assert(nr % 4 == 0);
    assert(nc % ncols_interleaved == 0);

    float sumf[4][16];
    int sumi;

    for (int y = 0; y < nr / 4; y++) {
        const block_q8_0x4 * a_ptr = (const block_q8_0x4 *) vy + (y * nb);
        for (int x = 0; x < nc / ncols_interleaved; x++) {
            const block_iq4_nlx16 * b_ptr = (const block_iq4_nlx16 *) vx + (x * nb);
            for (int m = 0; m < 4; m++) {
                for (int j = 0; j < ncols_interleaved; j++) sumf[m][j] = 0.0;
            }
            for (int l = 0; l < nb; l++) {
                for (int k = 0; k < (qk / (2 * blocklen)); k++) {
                    for (int m = 0; m < 4; m++) {
                        for (int j = 0; j < ncols_interleaved; j++) {
                            sumi = 0;
                            for (int i = 0; i < blocklen; ++i) {
                                const int v0 = kvalues_iq4nl[b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] & 0x0F];
                                const int v1 = kvalues_iq4nl[b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] >> 4];
                                sumi += ((v0 * a_ptr[l].qs[k * 4 * blocklen + m * blocklen + i]) +
                                         (v1 * a_ptr[l].qs[k * 4 * blocklen + m * blocklen + i + (qk / 2) * 4]));
                            }
                            sumf[m][j] += sumi * GGML_CPU_FP16_TO_FP32(b_ptr[l].d[j]) * GGML_CPU_FP16_TO_FP32(a_ptr[l].d[m]);
                        }
                    }
                }
            }
            for (int m = 0; m < 4; m++) {
                for (int j = 0; j < ncols_interleaved; j++)
                    s[(y * 4 + m) * bs + x * ncols_interleaved + j] = sumf[m][j];
            }
        }
    }
}

void ggml_gemm_q8_0_16x1_q8_0_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc) {
    const int qk                = QK8_0;
    const int nb                = n / qk;
    const int ncols_interleaved = 16;
    const int blocklen          = 1;

    assert(n % qk == 0);
    assert(nr % 4 == 0);
    assert(nc % ncols_interleaved == 0);

    float sumf[4][16];
    int   sumi;

    for (int y = 0; y < nr / 4; y++) {
        const block_q8_0x4 * a_ptr = (const block_q8_0x4 *) vy + (y * nb);
        for (int x = 0; x < nc / ncols_interleaved; x++) {
            const block_q8_0x16 * b_ptr = (const block_q8_0x16 *) vx + (x * nb);
            for (int m = 0; m < 4; m++) {
                for (int j = 0; j < ncols_interleaved; j++) {
                    sumf[m][j] = 0.0;
                }
            }
            for (int l = 0; l < nb; l++) {
                for (int k = 0; k < (qk / blocklen); k++) {
                    for (int m = 0; m < 4; m++) {
                        for (int j = 0; j < ncols_interleaved; j++) {
                            sumi = 0;
                            for (int i = 0; i < blocklen; ++i) {
                                const int v0 = b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i];
                                sumi += v0 * a_ptr[l].qs[k * 4 * blocklen + m * blocklen + i];
                            }
                            sumf[m][j] +=
                                sumi * GGML_CPU_FP16_TO_FP32(b_ptr[l].d[j]) * GGML_CPU_FP16_TO_FP32(a_ptr[l].d[m]);
                        }
                    }
                }
            }
            for (int m = 0; m < 4; m++) {
                for (int j = 0; j < ncols_interleaved; j++) {
                    s[(y * 4 + m) * bs + x * ncols_interleaved + j] = sumf[m][j];
                }
            }
        }
    }
}


void ggml_gemm_q2_K_16x1_q8_K_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc) {
    assert(n % QK_K == 0);
    assert(nr % 4 == 0);
    assert(nc % 16 == 0);
    const int nb = n / QK_K;
    const block_q2_Kx16 * x = (const block_q2_Kx16 *)vx;
    const block_q8_Kx4  * y = (const block_q8_Kx4 *)vy;

    const int sb_perm[16] = {
        0, 4, 1, 5, 2, 6, 3, 7,
        8, 12, 9, 13, 10, 14, 11, 15
    };

    // Iterate Rows in tiles of 4
    for (int row_tile = 0; row_tile < nr; row_tile += 4) {
        // Iterate Columns in tiles of 16
        for (int col_tile = 0; col_tile < nc; col_tile += 16) {

            const block_q2_Kx16 * x_ptr = x + (col_tile / 16) * nb;
            const block_q8_Kx4  * y_ptr = y + (row_tile / 4) * nb;

            float sumf[4][16];
            memset(sumf, 0, sizeof(sumf));

            for (int k_block = 0; k_block < nb; ++k_block) {
                int32_t isum[4][16];
                int32_t summs[4][16];
                memset(isum, 0, sizeof(isum));
                memset(summs, 0, sizeof(summs));

                const uint8_t * qs_rhs = x_ptr[k_block].qs;
                const uint8_t * sc_rhs = x_ptr[k_block].scales;
                const int8_t  * qs_lhs = y_ptr[k_block].qs;
                const int16_t * bs_lhs = y_ptr[k_block].bsums;

                for (int sb = 0; sb < 16; ++sb) {
                    int scale_offset = sb_perm[sb] * 16;

                    int byte_base;
                    if (sb < 8) byte_base = (sb % 2 == 0) ? 0 : 16;
                    else        byte_base = (sb % 2 == 0) ? 32 : 48;
                    int shift = ((sb / 2) % 4) * 2;

                    for (int col = 0; col < 16; ++col) {
                        uint8_t sc_val = sc_rhs[scale_offset + col];
                        int32_t d_sb = sc_val & 0xF;
                        int32_t m_sb = sc_val >> 4;

                        // Correction Term
                        for (int r = 0; r < 4; ++r) {
                            int bsum_idx = (sb / 4) * 16 + r * 4 + (sb % 4);
                            summs[r][col] += bs_lhs[bsum_idx] * m_sb;
                        }

                        // Main Dot Product
                        for (int l = 0; l < 16; ++l) {
                            int qs_idx = (byte_base + l) * 16 + col;
                            uint8_t q2_val = (qs_rhs[qs_idx] >> shift) & 3;

                            // Calculate Q8 index for this specific k and row
                            int k = sb * 16 + l;
                            int q8_idx = (k / 4) * 16 + (k % 4);

                            for (int r = 0; r < 4; ++r) {
                                // Add r*4 to jump to the correct row within the 4x4 chunk
                                int8_t q8_val = qs_lhs[q8_idx + r * 4];
                                isum[r][col] += q8_val * q2_val * d_sb;
                            }
                        }
                    }
                }

                // Finalize K-Block
                for (int col = 0; col < 16; ++col) {
                    float d_rhs = GGML_FP16_TO_FP32(x_ptr[k_block].d[col]);
                    float dm_rhs = GGML_FP16_TO_FP32(x_ptr[k_block].dmin[col]);

                    for (int r = 0; r < 4; ++r) {
                        float d_lhs = y_ptr[k_block].d[r];
                        float d_all = d_lhs * d_rhs;
                        float d_min = d_lhs * dm_rhs;
                        sumf[r][col] += (isum[r][col] * d_all) - (summs[r][col] * d_min);
                    }
                }
            }

            for (int r = 0; r < 4; ++r) {
                for (int col = 0; col < 16; ++col) {
                    s[(row_tile + r) * bs + (col_tile + col)] = sumf[r][col];
                }
            }
        }
    }
}
#endif

} // extern "C"

static block_q8_0x4 make_block_q8_0x4(block_q8_0 * in, unsigned int blck_size_interleave) {
    block_q8_0x4 out;

    for (int i = 0; i < 4; i++) {
        out.d[i] = in[i].d;
    }

    const int end = QK8_0 * 4 / blck_size_interleave;
    for (int i = 0; i < end; ++i) {
        int src_id     = i % 4;
        int src_offset = (i / 4) * blck_size_interleave;
        int dst_offset = i * blck_size_interleave;
        memcpy(&out.qs[dst_offset], &in[src_id].qs[src_offset], blck_size_interleave);
    }
    return out;
}

static block_q4_0x4 make_block_q4_0x4(block_q4_0 * in, unsigned int blck_size_interleave) {
    block_q4_0x4 out;

    for (int i = 0; i < 4; i++) {
        out.d[i] = in[i].d;
    }

    const int end = QK4_0 * 2 / blck_size_interleave;

    if (blck_size_interleave == 8) {
        const uint64_t xor_mask = 0x8888888888888888ULL;
        for (int i = 0; i < end; ++i) {
            int src_id = i % 4;
            int src_offset = (i / 4) * blck_size_interleave;
            int dst_offset = i * blck_size_interleave;

            uint64_t elems;
            // Using memcpy to avoid unaligned memory accesses
            memcpy(&elems, &in[src_id].qs[src_offset], sizeof(uint64_t));
            elems ^= xor_mask;
            memcpy(&out.qs[dst_offset], &elems, sizeof(uint64_t));
        }
    } else if (blck_size_interleave == 4) {
        const uint32_t xor_mask = 0x88888888;
        for (int i = 0; i < end; ++i) {
            int src_id = i % 4;
            int src_offset = (i / 4) * blck_size_interleave;
            int dst_offset = i * blck_size_interleave;

            uint32_t elems;
            memcpy(&elems, &in[src_id].qs[src_offset], sizeof(uint32_t));
            elems ^= xor_mask;
            memcpy(&out.qs[dst_offset], &elems, sizeof(uint32_t));
        }
    } else {
        GGML_ASSERT(false);
    }

    return out;
}

// interleave 8 block_q4_0s in blocks of blck_size_interleave
// returns an interleaved block_q4_0x8
// in the interleaved block_q4_0x8, place deltas for 8 block_q4_0 blocks
// first, then interleave quants from 8 block_q4_0s in blocks of blck_size_interleave
static block_q4_0x8 make_block_q4_0x8(block_q4_0 * in, unsigned int blck_size_interleave) {
    block_q4_0x8 out;

    for (int i = 0; i < 8; i++) {
        out.d[i] = in[i].d;
    }

    const int end = QK4_0 * 4 / blck_size_interleave;
    const uint64_t xor_mask = 0x8888888888888888ULL;

    for (int i = 0; i < end; ++i) {
        int src_id = i % 8;
        int src_offset = (i / 8) * blck_size_interleave;
        int dst_offset = i * blck_size_interleave;

        uint64_t elems;
        memcpy(&elems, &in[src_id].qs[src_offset], sizeof(uint64_t));
        elems ^= xor_mask;
        memcpy(&out.qs[dst_offset], &elems, sizeof(uint64_t));
    }

    return out;
}

static block_q4_0x16 make_block_q4_0x16(block_q4_0 * in, unsigned int blck_size_interleave) {
    block_q4_0x16 out;

    for (int i = 0; i < 16; i++) {
        out.d[i] = in[i].d;
    }

    const int end = QK4_0 * 8 / blck_size_interleave;

    if (blck_size_interleave == 1) {
        const uint8_t xor_mask = 0x88;
        for (int i = 0; i < end; ++i) {
            int src_id = i % 16;
            int src_offset = i / 16;
            int dst_offset = i;

            out.qs[dst_offset] = in[src_id].qs[src_offset] ^ xor_mask;
        }
    } else {
        GGML_ASSERT(false);
    }

    return out;
}

static block_q4_Kx8 make_block_q4_Kx8(block_q4_K * in, unsigned int blck_size_interleave) {
    block_q4_Kx8 out;
    //Delta(scale) and dmin values of the eight Q4_K structures are copied onto the output interleaved structure
    for (int i = 0; i < 8; i++) {
        out.d[i] = in[i].GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.d;
    }

    for (int i = 0; i < 8; i++) {
        out.dmin[i] = in[i].GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.dmin;
    }

    const int end = QK_K * 4 / blck_size_interleave;

    // Interleave Q4_K quants by taking 8 bytes at a time
    for (int i = 0; i < end; ++i) {
        int src_id = i % 8;
        int src_offset = (i / 8) * blck_size_interleave;
        int dst_offset = i * blck_size_interleave;

        // buffer large enough for the max interleave block size (8 bytes)
        uint64_t elems;
        memcpy(&elems, &in[src_id].qs[src_offset], blck_size_interleave);
        memcpy(&out.qs[dst_offset], &elems, blck_size_interleave);
    }

    // The below logic is designed so as to unpack and rearrange scales and mins values in Q4_K
    // Currently the Q4_K structure has 8 scales and 8 mins packed in 12 bytes ( 6 bits for each value)
    // The output Q4_Kx8 structure has 96 bytes
    // Every 12 byte is packed such that it contains scales and mins for corresponding sub blocks from Q4_K structure
    // For eg - First 12 bytes contains 8 scales and 8 mins - each of first sub block from different Q4_K structures
    uint8_t s[8], m[8];

    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 8; j++) {
            s[j] = in[j].scales[i] & 63;
            m[j] = in[j].scales[i + 4] & 63;
        }

        out.scales[i * 12]      = (s[0] & 63) + ((s[4] & 48) << 2);
        out.scales[i * 12 + 1]  = (s[1] & 63) + ((s[5] & 48) << 2);
        out.scales[i * 12 + 2]  = (s[2] & 63) + ((s[6] & 48) << 2);
        out.scales[i * 12 + 3]  = (s[3] & 63) + ((s[7] & 48) << 2);
        out.scales[i * 12 + 4]  = (m[0] & 63) + ((m[4] & 48) << 2);
        out.scales[i * 12 + 5]  = (m[1] & 63) + ((m[5] & 48) << 2);
        out.scales[i * 12 + 6]  = (m[2] & 63) + ((m[6] & 48) << 2);
        out.scales[i * 12 + 7]  = (m[3] & 63) + ((m[7] & 48) << 2);
        out.scales[i * 12 + 8]  = (s[4] & 15) + ((m[4] & 15) << 4);
        out.scales[i * 12 + 9]  = (s[5] & 15) + ((m[5] & 15) << 4);
        out.scales[i * 12 + 10] = (s[6] & 15) + ((m[6] & 15) << 4);
        out.scales[i * 12 + 11] = (s[7] & 15) + ((m[7] & 15) << 4);

    }

    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 8; j++) {
            s[j] = ((in[j].scales[i] & 192) >> 2) | (in[j].scales[i+8] & 15);
            m[j] = ((in[j].scales[i + 4] & 192) >> 2) | ((in[j].scales[i+8] & 240) >> 4);
        }

        out.scales[i * 12 + 48] = (s[0] & 63) + ((s[4] & 48) << 2);
        out.scales[i * 12 + 49] = (s[1] & 63) + ((s[5] & 48) << 2);
        out.scales[i * 12 + 50] = (s[2] & 63) + ((s[6] & 48) << 2);
        out.scales[i * 12 + 51] = (s[3] & 63) + ((s[7] & 48) << 2);
        out.scales[i * 12 + 52] = (m[0] & 63) + ((m[4] & 48) << 2);
        out.scales[i * 12 + 53] = (m[1] & 63) + ((m[5] & 48) << 2);
        out.scales[i * 12 + 54] = (m[2] & 63) + ((m[6] & 48) << 2);
        out.scales[i * 12 + 55] = (m[3] & 63) + ((m[7] & 48) << 2);
        out.scales[i * 12 + 56] = (s[4] & 15) + ((m[4] & 15) << 4);
        out.scales[i * 12 + 57] = (s[5] & 15) + ((m[5] & 15) << 4);
        out.scales[i * 12 + 58] = (s[6] & 15) + ((m[6] & 15) << 4);
        out.scales[i * 12 + 59] = (s[7] & 15) + ((m[7] & 15) << 4);

    }

    return out;
}

static block_q4_Kx16 make_block_q4_Kx16(block_q4_K * in, unsigned int blck_size_interleave) {
    block_q4_Kx16 out;
    //Delta(scale) and dmin values of the 16 Q4_K structures are copied onto the output interleaved structure
    for (int i = 0; i < 16; i++) {
        out.d[i] = in[i].GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.d;
    }

    for (int i = 0; i < 16; i++) {
        out.dmin[i] = in[i].GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.dmin;
    }

    const int end = QK_K * 8 / blck_size_interleave;

    if (blck_size_interleave == 1) {
        for (int i = 0; i < end; ++i) {
            int src_id = i % 16;
            int src_offset = i / 16;
            int dst_offset = i;

            out.qs[dst_offset] = in[src_id].qs[src_offset];
        }

        // RVV repacking.
        //
        // Extract sums and mins for all 8 sub-blocks for each block of Q4_K.
        uint8_t s[128], m[128];
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 16; j++) {
                s[i * 16 + j] = in[j].scales[i] & 63;
                m[i * 16 + j] = in[j].scales[i + 4] & 63;
            }
        }
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 16; j++) {
                s[64 + i * 16 + j] = ((in[j].scales[i] & 192) >> 2) | (in[j].scales[i+8] & 15);
                m[64 + i * 16 + j] = ((in[j].scales[i + 4] & 192) >> 2) | ((in[j].scales[i+8] & 240) >> 4);
            }
        }

        for (int i = 0; i < 128; i++) {
            out.scales[i] = (s[i] & 15) | ((m[i] & 15) << 4);
        }
        for (int i = 0; i < 64; i++) {
            out.scales[128 + i] = ((s[i] & 48) >> 4) | ((m[i] & 48) >> 2) | (s[64 + i] & 48) | ((m[64 + i] & 48) << 2);
        }
    } else {
        GGML_ASSERT(false);
    }

    return out;
}

static block_q2_Kx8 make_block_q2_Kx8(block_q2_K * in, unsigned int blck_size_interleave) {
    block_q2_Kx8 out;

    // Delta(scale) and dmin values of the eight Q2_K structures are copied onto the output interleaved structure
    for (int i = 0; i < 8; i++) {
        out.d[i] = in[i].GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.d;
    }

    for (int i = 0; i < 8; i++) {
        out.dmin[i] = in[i].GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.dmin;
    }

    const int end = QK_K * 2 / blck_size_interleave;

    // Interleave Q2_K quants by taking 8 bytes at a time
    for (int i = 0; i < end; ++i) {
        int src_id = i % 8;
        int src_offset = (i / 8) * blck_size_interleave;
        int dst_offset = i * blck_size_interleave;

        uint64_t elems;
        memcpy(&elems, &in[src_id].qs[src_offset], sizeof(uint64_t));
        memcpy(&out.qs[dst_offset], &elems, sizeof(uint64_t));
    }

    // The below logic is designed so as to unpack and rearrange scales and mins values in Q2_K
    // Currently the Q2_K structure has 16 scales and 16 mins packed in 16 bytes ( 4 bits for each value)
    // The output Q2_Kx8 structure has 128 bytes for storing scales and mins
    // Every 16 byte is packed such that it contains scales and mins for corresponding sub blocks from Q2_K structure
    // For eg - First 16 bytes contains 16 scales and 16 mins - each of first and second sub blocks from different Q2_K structures

    for (int i = 0; i < 128; i++) {
        // Index for selecting which q2k super block
        int src1 = (i % 16) / 2;
        // Index for selecting scale
        int src2 = ((i / 16) * 2) + (i % 2);

        out.scales[i] = in[src1].scales[src2];
    }
    return out;
}

static block_q3_Kx8 make_block_q3_Kx8(block_q3_K * in, unsigned int blck_size_interleave) {
    block_q3_Kx8 out;

    // Delta(scale) values of the eight Q3_K structures are copied onto the output interleaved structure
    for (int i = 0; i < 8; i++) {
        out.d[i] = in[i].d;
    }

    const int end = QK_K * 2 / blck_size_interleave;

    // Interleave Q3_K quants (low 2 bits) by taking 8 bytes at a time
    for (int i = 0; i < end; ++i) {
        int src_id = i % 8;
        int src_offset = (i / 8) * blck_size_interleave;
        int dst_offset = i * blck_size_interleave;

        uint64_t elems;
        memcpy(&elems, &in[src_id].qs[src_offset], sizeof(uint64_t));
        memcpy(&out.qs[dst_offset], &elems, sizeof(uint64_t));
    }

    // Interleave Q3_K high bits with the same chunk size, so that the high bits of
    // sub block sb stay at the same position within a chunk as its low 2 bits
    for (int i = 0; i < end / 2; ++i) {
        int src_id = i % 8;
        int src_offset = (i / 8) * blck_size_interleave;
        int dst_offset = i * blck_size_interleave;

        uint64_t elems;
        memcpy(&elems, &in[src_id].hmask[src_offset], sizeof(uint64_t));
        memcpy(&out.hmask[dst_offset], &elems, sizeof(uint64_t));
    }

    // The 16 sub block scales of a Q3_K structure are packed 6-bit wise in 12 bytes.
    // They are stored transposed in the output structure: scales[i*8 + j] holds byte i
    // of the packed scales of the j-th Q3_K structure, so the kernels can unpack the
    // scales of all eight structures with a single sequence of 64-bit operations
    for (int i = 0; i < 12; i++) {
        for (int j = 0; j < 8; j++) {
            out.scales[i * 8 + j] = in[j].scales[i];
        }
    }

    return out;
}

static block_q5_Kx8 make_block_q5_Kx8(block_q5_K * in, unsigned int blck_size_interleave) {
    block_q5_Kx8 out;
    //Delta(scale) and dmin values of the eight Q5_K structures are copied onto the output interleaved structure
    for (int i = 0; i < 8; i++) {
        out.d[i] = in[i].GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.d;
    }

    for (int i = 0; i < 8; i++) {
        out.dmin[i] = in[i].GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.dmin;
    }

    const int end = QK_K * 4 / blck_size_interleave;

    // Interleave Q5_K quants by taking blck_size_interleave bytes at a time
    for (int i = 0; i < end; ++i) {
        int src_id     = i % 8;
        int src_offset = (i / 8) * blck_size_interleave;
        int dst_offset = i * blck_size_interleave;

        memcpy(&out.qs[dst_offset], &in[src_id].qs[src_offset], blck_size_interleave);
    }

    // Repeat for high bits with the same chunk size, since
    // the high bits are interleaved in Q5_K and the index is
    // qh_idx = (qs_idx % 32);
    // qh_val = qh[qh_idx] >> (qs_idx / 32);
    for (int i = 0; i < end / 4; ++i) {
        int src_id     = i % 8;
        int src_offset = (i / 8) * blck_size_interleave;
        int dst_offset = i * blck_size_interleave;

        memcpy(&out.qh[dst_offset], &in[src_id].qh[src_offset], blck_size_interleave);
    }

    // The below logic is copied over from Q4_K
    // The point is to unpack all the scales and mins for each sub block every time we load 12 bytes.
    // Currently the Q5_K structure has 8 scales and 8 mins packed in 12 bytes ( 6 bits for each value)
    // The output Q5_Kx8 structure has 96 bytes
    // Every 12 byte is packed such that it contains scales and mins for corresponding sub blocks from Q5_K structure
    // For eg - First 12 bytes contains 8 scales and 8 mins - each of first sub block from different Q5_K structures
    uint8_t s[8], m[8];

    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 8; j++) {
            s[j] = in[j].scales[i] & 63;
            m[j] = in[j].scales[i + 4] & 63;
        }

        out.scales[i * 12]      = (s[0] & 63) + ((s[4] & 48) << 2);
        out.scales[i * 12 + 1]  = (s[1] & 63) + ((s[5] & 48) << 2);
        out.scales[i * 12 + 2]  = (s[2] & 63) + ((s[6] & 48) << 2);
        out.scales[i * 12 + 3]  = (s[3] & 63) + ((s[7] & 48) << 2);
        out.scales[i * 12 + 4]  = (m[0] & 63) + ((m[4] & 48) << 2);
        out.scales[i * 12 + 5]  = (m[1] & 63) + ((m[5] & 48) << 2);
        out.scales[i * 12 + 6]  = (m[2] & 63) + ((m[6] & 48) << 2);
        out.scales[i * 12 + 7]  = (m[3] & 63) + ((m[7] & 48) << 2);
        out.scales[i * 12 + 8]  = (s[4] & 15) + ((m[4] & 15) << 4);
        out.scales[i * 12 + 9]  = (s[5] & 15) + ((m[5] & 15) << 4);
        out.scales[i * 12 + 10] = (s[6] & 15) + ((m[6] & 15) << 4);
        out.scales[i * 12 + 11] = (s[7] & 15) + ((m[7] & 15) << 4);
    }

    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 8; j++) {
            s[j] = ((in[j].scales[i] & 192) >> 2) | (in[j].scales[i + 8] & 15);
            m[j] = ((in[j].scales[i + 4] & 192) >> 2) | ((in[j].scales[i + 8] & 240) >> 4);
        }

        out.scales[i * 12 + 48] = (s[0] & 63) + ((s[4] & 48) << 2);
        out.scales[i * 12 + 49] = (s[1] & 63) + ((s[5] & 48) << 2);
        out.scales[i * 12 + 50] = (s[2] & 63) + ((s[6] & 48) << 2);
        out.scales[i * 12 + 51] = (s[3] & 63) + ((s[7] & 48) << 2);
        out.scales[i * 12 + 52] = (m[0] & 63) + ((m[4] & 48) << 2);
        out.scales[i * 12 + 53] = (m[1] & 63) + ((m[5] & 48) << 2);
        out.scales[i * 12 + 54] = (m[2] & 63) + ((m[6] & 48) << 2);
        out.scales[i * 12 + 55] = (m[3] & 63) + ((m[7] & 48) << 2);
        out.scales[i * 12 + 56] = (s[4] & 15) + ((m[4] & 15) << 4);
        out.scales[i * 12 + 57] = (s[5] & 15) + ((m[5] & 15) << 4);
        out.scales[i * 12 + 58] = (s[6] & 15) + ((m[6] & 15) << 4);
        out.scales[i * 12 + 59] = (s[7] & 15) + ((m[7] & 15) << 4);
    }

    return out;
}

static block_iq1_sx8 make_block_iq1_sx8(block_iq1_s * in, unsigned int blck_size_interleave) {
    block_iq1_sx8 out;
    // Delta values of the eight IQ1_S structures are copied onto the output interleaved structure
    for (int i = 0; i < 8; i++) {
        out.d[i] = in[i].d;
    }

    // Interleave grid indices (low 8 bits) by taking blck_size_interleave bytes at a time
    const int end = QK_K / 8 / blck_size_interleave * 8;
    for (int i = 0; i < end; ++i) {
        int src_id     = i % 8;
        int src_offset = (i / 8) * blck_size_interleave;
        int dst_offset = i * blck_size_interleave;

        memcpy(&out.qs[dst_offset], &in[src_id].qs[src_offset], blck_size_interleave);
    }

    // Interleave the per-32-values uint16 (grid index high bits + scale + sign)
    // in chunks of 2 bytes
    const int end_h = QK_K / 16 / 2 * 8;
    for (int i = 0; i < end_h; ++i) {
        int src_id     = i % 8;
        int src_offset = (i / 8) * 2;
        int dst_offset = i * 2;

        memcpy(&out.qh[dst_offset], (const uint8_t *) in[src_id].qh + src_offset, 2);
    }

    return out;
}

static block_iq1_mx8 make_block_iq1_mx8(block_iq1_m * in, unsigned int blck_size_interleave) {
    block_iq1_mx8 out;

    // Interleave grid indices (low 8 bits) by taking blck_size_interleave bytes at a time
    const int end = QK_K / 8 / blck_size_interleave * 8;
    for (int i = 0; i < end; ++i) {
        int src_id     = i % 8;
        int src_offset = (i / 8) * blck_size_interleave;
        int dst_offset = i * blck_size_interleave;

        memcpy(&out.qs[dst_offset], &in[src_id].qs[src_offset], blck_size_interleave);
    }

    // Interleave the grid index high bits + sign bits (2 bytes per 32 values) in chunks of 2 bytes
    const int end_h = QK_K / 16 / 2 * 8;
    for (int i = 0; i < end_h; ++i) {
        int src_id     = i % 8;
        int src_offset = (i / 8) * 2;
        int dst_offset = i * 2;

        memcpy(&out.qh[dst_offset], (const uint8_t *) in[src_id].qh + src_offset, 2);
    }

    // The 12->8 byte packed scales stay per-block contiguous: the block fp16 scale is
    // assembled from nibbles spread across all 8 bytes, so there is no useful sub-chunking
    for (int i = 0; i < 8; i++) {
        memcpy(&out.scales[i * 8], in[i].scales, 8);
    }

    return out;
}

static block_iq2_xsx8 make_block_iq2_xsx8(block_iq2_xs * in, unsigned int blck_size_interleave) {
    block_iq2_xsx8 out;
    // Delta values of the eight IQ2_XS structures are copied onto the output interleaved structure
    for (int i = 0; i < 8; i++) {
        out.d[i] = in[i].d;
    }

    // Interleave the grid index + sign uint16 entries (64 bytes per block) by taking
    // blck_size_interleave bytes at a time; chunk g*8+j then holds block j's 4 entries of
    // sub block g contiguously
    const int end = 8 * (QK_K/4) / blck_size_interleave;
    for (int i = 0; i < end; ++i) {
        int src_id     = i % 8;
        int src_offset = (i / 8) * blck_size_interleave;
        int dst_offset = i * blck_size_interleave;

        memcpy((uint8_t *) out.qs + dst_offset, (const uint8_t *) in[src_id].qs + src_offset, blck_size_interleave);
    }

    // The 8 packed 4-bit scale bytes of each block are stored transposed: scales[g*8 + j]
    // holds sub block scale byte g of block j, so a single 8-byte load per sub block
    // covers all eight columns
    for (int g = 0; g < QK_K/32; g++) {
        for (int j = 0; j < 8; j++) {
            out.scales[g*8 + j] = in[j].scales[g];
        }
    }

    return out;
}

static block_iq3_xxsx8 make_block_iq3_xxsx8(block_iq3_xxs * in, unsigned int blck_size_interleave) {
    block_iq3_xxsx8 out;
    // Delta values of the eight IQ3_XXS structures are copied onto the output interleaved structure
    for (int i = 0; i < 8; i++) {
        out.d[i] = in[i].d;
    }

    // Interleave the grid index bytes (64 bytes per block) by taking blck_size_interleave
    // bytes at a time; chunk g*8+j then holds block j's 8 grid indices of sub block g
    const int end = 8 * (QK_K/4) / blck_size_interleave;
    for (int i = 0; i < end; ++i) {
        int src_id     = i % 8;
        int src_offset = (i / 8) * blck_size_interleave;
        int dst_offset = i * blck_size_interleave;

        memcpy(out.qs + dst_offset, in[src_id].qs + src_offset, blck_size_interleave);
    }

    // Interleave the gas words (signs + scale, one uint32 per 32 values) in chunks of 4
    // bytes, so that gas[g*32 + j*4] holds block j's gas word of sub block g
    const int end_gas = 8 * (QK_K/8) / 4;
    for (int i = 0; i < end_gas; ++i) {
        int src_id     = i % 8;
        int src_offset = (i / 8) * 4;
        int dst_offset = i * 4;

        memcpy(out.gas + dst_offset, in[src_id].qs + QK_K/4 + src_offset, 4);
    }

    return out;
}

static block_q6_Kx8 make_block_q6_Kx8(block_q6_K * in, unsigned int blck_size_interleave) {
    block_q6_Kx8  out;
    constexpr int n_blocks = 8;  // Kx8
    for (int i = 0; i < n_blocks; i++) {
        out.d[i] = in[i].d;
    }

    const int end_ls = QK_K * 4 / blck_size_interleave;
    // Interleave Q6_K quants by taking blck_size_interleave bytes at a time
    for (int i = 0; i < end_ls; ++i) {
        int src_id     = i % n_blocks;
        int src_offset = (i / n_blocks) * blck_size_interleave;
        int dst_offset = i * blck_size_interleave;

        uint64_t elem_ls;
        memcpy(&elem_ls, &in[src_id].ql[src_offset], blck_size_interleave);
        memcpy(&out.ql[dst_offset], &elem_ls, blck_size_interleave);
    }

    // Interleave high bits using same chunk size as low bits
    const int end_hs = end_ls / 2;
    for (int i = 0; i < end_hs; ++i) {
        int src_id     = i % n_blocks;
        int src_offset = (i / n_blocks) * blck_size_interleave;
        int dst_offset = i * blck_size_interleave;

        uint64_t elem_hs;
        memcpy(&elem_hs, &in[src_id].qh[src_offset], blck_size_interleave);
        memcpy(&out.qh[dst_offset], &elem_hs, blck_size_interleave);
    }

    // The below logic is designed so as to unpack and rearrange scales in Q6_K
    // The output Q6_Kx8 structure interleaves the 8 bit scales in the same fashion as the quants
    // Q6_K structure has an 8-bit scale per 16 elements -> 16 scales
    // scales: [0 bl0 0 bl1 ... 0 bl7][1 bl0 ... 1 bl7] ... [15 bl0 ... 15 bl7]  (bl = block)
    constexpr int n_scales = QK_K / 16;

    for (int i = 0; i < n_blocks; i++) {
        for (int j = 0; j < n_scales; j++) {
            out.scales[j * n_blocks + i] = in[i].scales[j];
        }
    }

    return out;
}

static block_q2_Kx16 make_block_q2_Kx16(const block_q2_K * in, unsigned int blck_size_interleave) {
    block_q2_Kx16 out;
    constexpr int N_COLS = 16;

    // 1. Copy Super-Scales (d) and Super-Mins (dmin)
    for (int i = 0; i < N_COLS; i++) {
        out.d[i]    = in[i].GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.d;
        out.dmin[i] = in[i].GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.dmin;
    }

    // 2. Interleave Q2_K Data
    const int bytes_per_col = 64;
    const int total_bytes = N_COLS * bytes_per_col;
    const int end = total_bytes / blck_size_interleave;

    for (int i = 0; i < end; ++i) {
        int src_col_id = i % N_COLS;
        int src_offset = (i / N_COLS) * blck_size_interleave;
        int dst_offset = i * blck_size_interleave;
        memcpy(&out.qs[dst_offset], &in[src_col_id].qs[src_offset], blck_size_interleave);
    }

    // 3. Repack Scales into the Optimized "Sequential-Parallel" Layout
    int out_idx = 0;

    // Arrays define the sub-block order for each group
    const int even_low_sbs[]  = {0, 2, 4, 6};
    const int odd_low_sbs[]   = {1, 3, 5, 7};
    const int even_high_sbs[] = {8, 10, 12, 14};
    const int odd_high_sbs[]  = {9, 11, 13, 15};

    // Pack Group 1: Even-Low
    for (int sb : even_low_sbs) {
        for (int col = 0; col < N_COLS; col++) {
            out.scales[out_idx++] = in[col].scales[sb];
        }
    }

    // Pack Group 2: Odd-Low
    for (int sb : odd_low_sbs) {
        for (int col = 0; col < N_COLS; col++) {
            out.scales[out_idx++] = in[col].scales[sb];
        }
    }

    // Pack Group 3: Even-High
    for (int sb : even_high_sbs) {
        for (int col = 0; col < N_COLS; col++) {
            out.scales[out_idx++] = in[col].scales[sb];
        }
    }

    // Pack Group 4: Odd-High
    for (int sb : odd_high_sbs) {
        for (int col = 0; col < N_COLS; col++) {
            out.scales[out_idx++] = in[col].scales[sb];
        }
    }

    return out;
}

static int repack_q4_0_to_q4_0_4_bl(struct ggml_tensor * t, int interleave_block, const void * GGML_RESTRICT data, size_t data_size) {
    GGML_ASSERT(t->type == GGML_TYPE_Q4_0);
    GGML_ASSERT(interleave_block == 4 || interleave_block == 8);
    constexpr int nrows_interleaved = 4;

    block_q4_0x4 * dst = (block_q4_0x4 *)t->data;
    const block_q4_0 * src = (const block_q4_0 *)data;
    block_q4_0 dst_tmp[4];
    int nrow = ggml_nrows(t);
    int nblocks = t->ne[0] / QK4_0;

    GGML_ASSERT(data_size == nrow * nblocks * sizeof(block_q4_0));

    if (t->ne[1] % nrows_interleaved != 0 || t->ne[0] % 8 != 0) {
        return -1;
    }

    for (int b = 0; b < nrow; b += nrows_interleaved) {
        for (int64_t x = 0; x < nblocks; x++) {
            for (int i = 0; i < nrows_interleaved; i++) {
                dst_tmp[i] = src[x + i * nblocks];
            }
            *dst++ = make_block_q4_0x4(dst_tmp, interleave_block);
        }
        src += nrows_interleaved * nblocks;
    }
    return 0;

    GGML_UNUSED(data_size);
}

static int repack_q4_K_to_q4_K_8_bl(struct ggml_tensor * t, int interleave_block, const void * GGML_RESTRICT data, size_t data_size) {
    GGML_ASSERT(t->type == GGML_TYPE_Q4_K);
    GGML_ASSERT(interleave_block == 8 || interleave_block == 4);
    constexpr int nrows_interleaved = 8;

    block_q4_Kx8 * dst = (block_q4_Kx8*)t->data;
    const block_q4_K * src = (const block_q4_K*) data;
    block_q4_K dst_tmp[8];
    int nrow = ggml_nrows(t);
    int nblocks = t->ne[0] / QK_K;

    GGML_ASSERT(data_size == nrow * nblocks * sizeof(block_q4_K));

    if (t->ne[1] % nrows_interleaved != 0 || t->ne[0] % 8 != 0) {
        return -1;
    }

    for (int b = 0; b < nrow; b += nrows_interleaved) {
        for (int64_t x = 0; x < nblocks; x++) {
            for (int i  = 0; i < nrows_interleaved; i++ ) {
                dst_tmp[i] = src[x + i * nblocks];
            }
            *dst++ = make_block_q4_Kx8(dst_tmp, interleave_block);
        }
        src += nrows_interleaved * nblocks;
    }
    return 0;

    GGML_UNUSED(data_size);
}

static int repack_q4_K_to_q4_K_16_bl(struct ggml_tensor * t, int interleave_block, const void * GGML_RESTRICT data, size_t data_size) {
    GGML_ASSERT(t->type == GGML_TYPE_Q4_K);
    constexpr int nrows_interleaved = 16;

    block_q4_Kx16 * dst = (block_q4_Kx16*)t->data;
    const block_q4_K * src = (const block_q4_K*) data;
    block_q4_K dst_tmp[16];
    int nrow = ggml_nrows(t);
    int nblocks = t->ne[0] / QK_K;

    GGML_ASSERT(data_size == nrow * nblocks * sizeof(block_q4_K));

    if (t->ne[1] % nrows_interleaved != 0 || t->ne[0] % 8 != 0) {
        return -1;
    }

    for (int b = 0; b < nrow; b += nrows_interleaved) {
        for (int64_t x = 0; x < nblocks; x++) {
            for (int i  = 0; i < nrows_interleaved; i++ ) {
                dst_tmp[i] = src[x + i * nblocks];
            }
            *dst++ = make_block_q4_Kx16(dst_tmp, interleave_block);
        }
        src += nrows_interleaved * nblocks;
    }
    return 0;

    GGML_UNUSED(data_size);
}

static int repack_q2_K_to_q2_K_8_bl(struct ggml_tensor * t, int interleave_block, const void * GGML_RESTRICT data, size_t data_size) {
    GGML_ASSERT(t->type == GGML_TYPE_Q2_K);
    GGML_ASSERT(interleave_block == 8);
    constexpr int nrows_interleaved = 8;

    block_q2_Kx8 * dst = (block_q2_Kx8*)t->data;
    const block_q2_K * src = (const block_q2_K*) data;
    block_q2_K dst_tmp[8];
    int nrow = ggml_nrows(t);
    int nblocks = t->ne[0] / QK_K;

    GGML_ASSERT(data_size == nrow * nblocks * sizeof(block_q2_K));

    if (t->ne[1] % nrows_interleaved != 0 || t->ne[0] % 8 != 0) {
        return -1;
    }

    for (int b = 0; b < nrow; b += nrows_interleaved) {
        for (int64_t x = 0; x < nblocks; x++) {
            for (int i = 0; i < nrows_interleaved; i++) {
                dst_tmp[i] = src[x + i * nblocks];
            }
            *dst++ = make_block_q2_Kx8(dst_tmp, interleave_block);
        }
        src += nrows_interleaved * nblocks;
    }
    return 0;

    GGML_UNUSED(data_size);
}

static int repack_q3_K_to_q3_K_8_bl(struct ggml_tensor * t, int interleave_block, const void * GGML_RESTRICT data, size_t data_size) {
    GGML_ASSERT(t->type == GGML_TYPE_Q3_K);
    GGML_ASSERT(interleave_block == 8);
    constexpr int nrows_interleaved = 8;

    block_q3_Kx8 * dst = (block_q3_Kx8*)t->data;
    const block_q3_K * src = (const block_q3_K*) data;
    block_q3_K dst_tmp[8];
    int nrow = ggml_nrows(t);
    int nblocks = t->ne[0] / QK_K;

    GGML_ASSERT(data_size == nrow * nblocks * sizeof(block_q3_K));

    if (t->ne[1] % nrows_interleaved != 0 || t->ne[0] % 8 != 0) {
        return -1;
    }

    for (int b = 0; b < nrow; b += nrows_interleaved) {
        for (int64_t x = 0; x < nblocks; x++) {
            for (int i = 0; i < nrows_interleaved; i++) {
                dst_tmp[i] = src[x + i * nblocks];
            }
            *dst++ = make_block_q3_Kx8(dst_tmp, interleave_block);
        }
        src += nrows_interleaved * nblocks;
    }
    return 0;

    GGML_UNUSED(data_size);
}

static int repack_q2_K_to_q2_K_16_bl(struct ggml_tensor * t, int interleave_block, const void * GGML_RESTRICT data, size_t data_size) {
    GGML_ASSERT(t->type == GGML_TYPE_Q2_K);
    constexpr int nrows_interleaved = 16;

    block_q2_Kx16 * dst = (block_q2_Kx16*)t->data;
    const block_q2_K * src = (const block_q2_K*) data;

    block_q2_K dst_tmp[nrows_interleaved];

    int nrow = ggml_nrows(t);
    int nblocks = t->ne[0] / QK_K;

    GGML_ASSERT(data_size == nrow * nblocks * sizeof(block_q2_K));

    if (t->ne[1] % nrows_interleaved != 0 || t->ne[0] % 8 != 0) {
        return -1;
    }

    for (int b = 0; b < nrow; b += nrows_interleaved) {
        for (int64_t x = 0; x < nblocks; x++) {
            // This loop gathers 16 separate blocks (one from each column)
            // that correspond to the same K-dimension chunk.
            for (int i  = 0; i < nrows_interleaved; i++ ) {
                dst_tmp[i] = src[x + i * nblocks];
            }

            *dst++ = make_block_q2_Kx16(dst_tmp, interleave_block);
        }
        src += nrows_interleaved * nblocks;
    }
    return 0;

    GGML_UNUSED(data_size);
}

static int repack_q4_0_to_q4_0_16_bl(struct ggml_tensor * t, int interleave_block, const void * GGML_RESTRICT data, size_t data_size) {
    GGML_ASSERT(t->type == GGML_TYPE_Q4_0);
    constexpr int nrows_interleaved = 16;

    block_q4_0x16 * dst = (block_q4_0x16*)t->data;
    const block_q4_0 * src = (const block_q4_0*) data;
    block_q4_0 dst_tmp[16];
    int nrow = ggml_nrows(t);
    int nblocks = t->ne[0] / QK4_0;

    GGML_ASSERT(data_size == nrow * nblocks * sizeof(block_q4_0));

    if (t->ne[1] % nrows_interleaved != 0 || t->ne[0] % 8 != 0) {
        return -1;
    }

    for (int b = 0; b < nrow; b += nrows_interleaved) {
        for (int64_t x = 0; x < nblocks; x++) {
            for (int i  = 0; i < nrows_interleaved; i++ ) {
                dst_tmp[i] = src[x + i * nblocks];
            }
            *dst++ = make_block_q4_0x16(dst_tmp, interleave_block);
        }
        src += nrows_interleaved * nblocks;
    }
    return 0;

    GGML_UNUSED(data_size);
}

static int repack_q5_K_to_q5_K_8_bl(struct ggml_tensor *       t,
                                    int                        interleave_block,
                                    const void * GGML_RESTRICT data,
                                    size_t                     data_size) {
    GGML_ASSERT(t->type == GGML_TYPE_Q5_K);
    GGML_ASSERT(interleave_block == 4 || interleave_block == 8);
    constexpr int nrows_interleaved = 8;

    block_q5_Kx8 *     dst = (block_q5_Kx8 *) t->data;
    const block_q5_K * src = (const block_q5_K *) data;
    block_q5_K         dst_tmp[8];
    int                nrow    = ggml_nrows(t);
    int                nblocks = t->ne[0] / QK_K;

    GGML_ASSERT(data_size == nrow * nblocks * sizeof(block_q5_K));

    if (t->ne[1] % nrows_interleaved != 0 || t->ne[0] % 8 != 0) {
        return -1;
    }

    for (int b = 0; b < nrow; b += nrows_interleaved) {
        for (int64_t x = 0; x < nblocks; x++) {
            for (int i = 0; i < nrows_interleaved; i++) {
                dst_tmp[i] = src[x + i * nblocks];
            }
            *dst++ = make_block_q5_Kx8(dst_tmp, interleave_block);
        }
        src += nrows_interleaved * nblocks;
    }
    return 0;
}

static int repack_iq1_s_to_iq1_s_8_bl(struct ggml_tensor *       t,
                                      int                        interleave_block,
                                      const void * GGML_RESTRICT data,
                                      size_t                     data_size) {
    GGML_ASSERT(t->type == GGML_TYPE_IQ1_S);
    GGML_ASSERT(interleave_block == 8);
    constexpr int nrows_interleaved = 8;

    block_iq1_sx8 *     dst = (block_iq1_sx8 *) t->data;
    const block_iq1_s * src = (const block_iq1_s *) data;
    block_iq1_s         dst_tmp[8];
    int                 nrow    = ggml_nrows(t);
    int                 nblocks = t->ne[0] / QK_K;

    GGML_ASSERT(data_size == nrow * nblocks * sizeof(block_iq1_s));

    if (t->ne[1] % nrows_interleaved != 0 || t->ne[0] % 8 != 0) {
        return -1;
    }

    for (int b = 0; b < nrow; b += nrows_interleaved) {
        for (int64_t x = 0; x < nblocks; x++) {
            for (int i = 0; i < nrows_interleaved; i++) {
                dst_tmp[i] = src[x + i * nblocks];
            }
            *dst++ = make_block_iq1_sx8(dst_tmp, interleave_block);
        }
        src += nrows_interleaved * nblocks;
    }
    return 0;
}

static int repack_iq1_m_to_iq1_m_8_bl(struct ggml_tensor *       t,
                                      int                        interleave_block,
                                      const void * GGML_RESTRICT data,
                                      size_t                     data_size) {
    GGML_ASSERT(t->type == GGML_TYPE_IQ1_M);
    GGML_ASSERT(interleave_block == 8);
    constexpr int nrows_interleaved = 8;

    block_iq1_mx8 *     dst = (block_iq1_mx8 *) t->data;
    const block_iq1_m * src = (const block_iq1_m *) data;
    block_iq1_m         dst_tmp[8];
    int                 nrow    = ggml_nrows(t);
    int                 nblocks = t->ne[0] / QK_K;

    GGML_ASSERT(data_size == nrow * nblocks * sizeof(block_iq1_m));

    if (t->ne[1] % nrows_interleaved != 0 || t->ne[0] % 8 != 0) {
        return -1;
    }

    for (int b = 0; b < nrow; b += nrows_interleaved) {
        for (int64_t x = 0; x < nblocks; x++) {
            for (int i = 0; i < nrows_interleaved; i++) {
                dst_tmp[i] = src[x + i * nblocks];
            }
            *dst++ = make_block_iq1_mx8(dst_tmp, interleave_block);
        }
        src += nrows_interleaved * nblocks;
    }
    return 0;
}

static int repack_iq2_xs_to_iq2_xs_8_bl(struct ggml_tensor *       t,
                                        int                        interleave_block,
                                        const void * GGML_RESTRICT data,
                                        size_t                     data_size) {
    GGML_ASSERT(t->type == GGML_TYPE_IQ2_XS);
    GGML_ASSERT(interleave_block == 8);
    constexpr int nrows_interleaved = 8;

    block_iq2_xsx8 *     dst = (block_iq2_xsx8 *) t->data;
    const block_iq2_xs * src = (const block_iq2_xs *) data;
    block_iq2_xs         dst_tmp[8];
    int                  nrow    = ggml_nrows(t);
    int                  nblocks = t->ne[0] / QK_K;

    GGML_ASSERT(data_size == nrow * nblocks * sizeof(block_iq2_xs));

    if (t->ne[1] % nrows_interleaved != 0 || t->ne[0] % 8 != 0) {
        return -1;
    }

    for (int b = 0; b < nrow; b += nrows_interleaved) {
        for (int64_t x = 0; x < nblocks; x++) {
            for (int i = 0; i < nrows_interleaved; i++) {
                dst_tmp[i] = src[x + i * nblocks];
            }
            *dst++ = make_block_iq2_xsx8(dst_tmp, interleave_block);
        }
        src += nrows_interleaved * nblocks;
    }
    return 0;
}

static int repack_iq3_xxs_to_iq3_xxs_8_bl(struct ggml_tensor *       t,
                                          int                        interleave_block,
                                          const void * GGML_RESTRICT data,
                                          size_t                     data_size) {
    GGML_ASSERT(t->type == GGML_TYPE_IQ3_XXS);
    GGML_ASSERT(interleave_block == 8);
    constexpr int nrows_interleaved = 8;

    block_iq3_xxsx8 *     dst = (block_iq3_xxsx8 *) t->data;
    const block_iq3_xxs * src = (const block_iq3_xxs *) data;
    block_iq3_xxs         dst_tmp[8];
    int                   nrow    = ggml_nrows(t);
    int                   nblocks = t->ne[0] / QK_K;

    GGML_ASSERT(data_size == nrow * nblocks * sizeof(block_iq3_xxs));

    if (t->ne[1] % nrows_interleaved != 0 || t->ne[0] % 8 != 0) {
        return -1;
    }

    for (int b = 0; b < nrow; b += nrows_interleaved) {
        for (int64_t x = 0; x < nblocks; x++) {
            for (int i = 0; i < nrows_interleaved; i++) {
                dst_tmp[i] = src[x + i * nblocks];
            }
            *dst++ = make_block_iq3_xxsx8(dst_tmp, interleave_block);
        }
        src += nrows_interleaved * nblocks;
    }
    return 0;
}

static int repack_q6_K_to_q6_K_8_bl(struct ggml_tensor * t, int interleave_block, const void * GGML_RESTRICT data, size_t data_size) {
    GGML_ASSERT(t->type == GGML_TYPE_Q6_K);
    GGML_ASSERT(interleave_block == 4 || interleave_block == 8);
    constexpr int nrows_interleaved = 8;

    block_q6_Kx8 * dst = (block_q6_Kx8 *)t->data;
    const block_q6_K * src = (const block_q6_K *) data;
    block_q6_K dst_tmp[8];
    int nrow = ggml_nrows(t);
    int nblocks = t->ne[0] / QK_K;

    GGML_ASSERT(data_size == nrow * nblocks * sizeof(block_q6_K));

    if (t->ne[1] % nrows_interleaved != 0 || t->ne[0] % 8 != 0) {
        return -1;
    }

    for (int b = 0; b < nrow; b += nrows_interleaved) {
        for (int64_t x = 0; x < nblocks; x++) {
            for (int i = 0; i < nrows_interleaved; i++) {
                dst_tmp[i] = src[x + i * nblocks];
            }
            *dst++ = make_block_q6_Kx8(dst_tmp, interleave_block);
        }
        src += nrows_interleaved * nblocks;
    }
    return 0;
}

static int repack_q4_0_to_q4_0_8_bl(struct ggml_tensor * t, int interleave_block, const void * GGML_RESTRICT data, size_t data_size) {
    GGML_ASSERT(t->type == GGML_TYPE_Q4_0);
    GGML_ASSERT(interleave_block == 8);
    constexpr int nrows_interleaved = 8;

    block_q4_0x8 * dst = (block_q4_0x8*)t->data;
    const block_q4_0 * src = (const block_q4_0*) data;
    block_q4_0 dst_tmp[8];
    int nrow = ggml_nrows(t);
    int nblocks = t->ne[0] / QK4_0;

    GGML_ASSERT(data_size == nrow * nblocks * sizeof(block_q4_0));

    if (t->ne[1] % nrows_interleaved != 0 || t->ne[0] % 8 != 0) {
        return -1;
    }

    for (int b = 0; b < nrow; b += nrows_interleaved) {
        for (int64_t x = 0; x < nblocks; x++) {
            for (int i  = 0; i < nrows_interleaved; i++ ) {
                dst_tmp[i] = src[x + i * nblocks];
            }
            *dst++ = make_block_q4_0x8(dst_tmp, interleave_block);
        }
        src += nrows_interleaved * nblocks;
    }
    return 0;

    GGML_UNUSED(data_size);
}

static int repack_q8_0_to_q8_0_4_bl(struct ggml_tensor *       t,
                                    int                        interleave_block,
                                    const void * GGML_RESTRICT data,
                                    size_t                     data_size) {
    GGML_ASSERT(t->type == GGML_TYPE_Q8_0);
    GGML_ASSERT(interleave_block == 4 || interleave_block == 8);
    constexpr int nrows_interleaved = 4;

    block_q8_0x4 *     dst = (block_q8_0x4 *) t->data;
    const block_q8_0 * src = (const block_q8_0 *) data;
    block_q8_0         dst_tmp[4];
    int                nrow    = ggml_nrows(t);
    int                nblocks = t->ne[0] / QK8_0;

    GGML_ASSERT(data_size == nrow * nblocks * sizeof(block_q8_0));

    if (t->ne[1] % nrows_interleaved != 0 || t->ne[0] % 8 != 0) {
        return -1;
    }

    for (int b = 0; b < nrow; b += nrows_interleaved) {
        for (int64_t x = 0; x < nblocks; x++) {
            for (int i = 0; i < nrows_interleaved; i++) {
                dst_tmp[i] = src[x + i * nblocks];
            }
            *dst++ = make_block_q8_0x4(dst_tmp, interleave_block);
        }
        src += nrows_interleaved * nblocks;
    }
    return 0;
}

static block_q8_0x8 make_block_q8_0x8(block_q8_0 * in, unsigned int blck_size_interleave) {
    block_q8_0x8 out;

    for (int i = 0; i < 8; i++) {
        out.d[i] = in[i].d;
    }

    const int end = QK8_0 * 8 / blck_size_interleave;
    for (int i = 0; i < end; ++i) {
        int src_id     = i % 8;
        int src_offset = (i / 8) * blck_size_interleave;
        int dst_offset = i * blck_size_interleave;
        memcpy(&out.qs[dst_offset], &in[src_id].qs[src_offset], blck_size_interleave);
    }
    return out;
}

static int repack_q8_0_to_q8_0_8_bl(struct ggml_tensor *       t,
                                    int                        interleave_block,
                                    const void * GGML_RESTRICT data,
                                    size_t                     data_size) {
    GGML_ASSERT(t->type == GGML_TYPE_Q8_0);
    GGML_ASSERT(interleave_block == 8);
    constexpr int nrows_interleaved = 8;

    block_q8_0x8 *     dst = (block_q8_0x8 *) t->data;
    const block_q8_0 * src = (const block_q8_0 *) data;
    block_q8_0         dst_tmp[8];
    int                nrow    = ggml_nrows(t);
    int                nblocks = t->ne[0] / QK8_0;

    GGML_ASSERT(data_size == nrow * nblocks * sizeof(block_q8_0));

    if (t->ne[1] % nrows_interleaved != 0 || t->ne[0] % 8 != 0) {
        return -1;
    }

    for (int b = 0; b < nrow; b += nrows_interleaved) {
        for (int64_t x = 0; x < nblocks; x++) {
            for (int i = 0; i < nrows_interleaved; i++) {
                dst_tmp[i] = src[x + i * nblocks];
            }
            *dst++ = make_block_q8_0x8(dst_tmp, interleave_block);
        }
        src += nrows_interleaved * nblocks;
    }
    return 0;
}

static block_q8_0x16 make_block_q8_0x16(block_q8_0 * in, unsigned int blck_size_interleave) {
    block_q8_0x16 out;

    for (int i = 0; i < 16; i++) {
        out.d[i] = in[i].d;
    }

    const int end = QK8_0 * 16 / blck_size_interleave;

    if (blck_size_interleave == 1) {
        for (int i = 0; i < end; ++i) {
            int src_id     = i % 16;
            int src_offset = i / 16;
            int dst_offset = i;
            out.qs[dst_offset] = in[src_id].qs[src_offset];
        }
    } else {
        GGML_ASSERT(false);
    }

    return out;
}

static int repack_q8_0_to_q8_0_16_bl(struct ggml_tensor *       t,
                                    int                        interleave_block,
                                    const void * GGML_RESTRICT data,
                                    size_t                     data_size) {
    GGML_ASSERT(t->type == GGML_TYPE_Q8_0);
    constexpr int nrows_interleaved = 16;

    block_q8_0x16 *     dst = (block_q8_0x16 *) t->data;
    const block_q8_0 * src = (const block_q8_0 *) data;
    block_q8_0         dst_tmp[16];
    int                nrow    = ggml_nrows(t);
    int                nblocks = t->ne[0] / QK8_0;

    GGML_ASSERT(data_size == nrow * nblocks * sizeof(block_q8_0));

    if (t->ne[1] % nrows_interleaved != 0 || t->ne[0] % 8 != 0) {
        return -1;
    }

    for (int b = 0; b < nrow; b += nrows_interleaved) {
        for (int64_t x = 0; x < nblocks; x++) {
            for (int i = 0; i < nrows_interleaved; i++) {
                dst_tmp[i] = src[x + i * nblocks];
            }
            *dst++ = make_block_q8_0x16(dst_tmp, interleave_block);
        }
        src += nrows_interleaved * nblocks;
    }
    return 0;
}

static block_iq4_nlx4 make_block_iq4_nlx4(block_iq4_nl * in, unsigned int blck_size_interleave) {
    block_iq4_nlx4 out;

    for (int i = 0; i < 4; i++) {
        out.d[i] = in[i].d;
    }

    const int end = QK4_NL * 2 / blck_size_interleave;

    // TODO: this branch seems wrong
    //if (blck_size_interleave == 8) {
    //    for (int i = 0; i < end; ++i) {
    //        int src_id = i % 4;
    //        int src_offset = (i / 4) * blck_size_interleave;
    //        int dst_offset = i * blck_size_interleave;

    //        // Using memcpy to avoid unaligned memory accesses
    //        memcpy(&out.qs[dst_offset], &in[src_id].qs[src_offset], sizeof(uint64_t));
    //    }
    //} else
    if (blck_size_interleave == 4) {
        for (int i = 0; i < end; ++i) {
            int src_id = i % 4;
            int src_offset = (i / 4) * blck_size_interleave;
            int dst_offset = i * blck_size_interleave;

            memcpy(&out.qs[dst_offset], &in[src_id].qs[src_offset], sizeof(uint32_t));
        }
    } else {
        GGML_ASSERT(false);
    }

    return out;
}

static int repack_iq4_nl_to_iq4_nl_4_bl(struct ggml_tensor * t, int interleave_block, const void * GGML_RESTRICT data, size_t data_size) {
    GGML_ASSERT(t->type == GGML_TYPE_IQ4_NL);
    GGML_ASSERT(interleave_block == 4);

    const block_iq4_nl   * src = (const block_iq4_nl   *)data;
          block_iq4_nlx4 * dst = (      block_iq4_nlx4 *)t->data;

    block_iq4_nl dst_tmp[4];

    int nrow = ggml_nrows(t);
    int nrows_interleaved = 4;
    int nblocks = t->ne[0] / QK4_NL;

    GGML_ASSERT(data_size == nrow * nblocks * sizeof(block_iq4_nl));

    if (t->ne[1] % nrows_interleaved != 0 || t->ne[0] % 8 != 0) {
        return -1;
    }

    for (int b = 0; b < nrow; b += nrows_interleaved) {
        for (int64_t x = 0; x < nblocks; x++) {
            for (int i = 0; i < nrows_interleaved; i++) {
                dst_tmp[i] = src[x + i * nblocks];
            }
            *dst++ = make_block_iq4_nlx4(dst_tmp, interleave_block);
        }
        src += nrows_interleaved * nblocks;
    }
    return 0;

    GGML_UNUSED(data_size);
}

static block_iq4_nlx8 make_block_iq4_nlx8(block_iq4_nl * in, unsigned int blck_size_interleave) {
    block_iq4_nlx8 out;

    for (int i = 0; i < 8; i++) {
        out.d[i] = in[i].d;
    }

    const int end = QK4_NL * 4 / blck_size_interleave;

    if (blck_size_interleave == 8) {
        for (int i = 0; i < end; ++i) {
            int src_id = i % 8;
            int src_offset = (i / 8) * blck_size_interleave;
            int dst_offset = i * blck_size_interleave;

            memcpy(&out.qs[dst_offset], &in[src_id].qs[src_offset], sizeof(uint64_t));
        }
    } else {
        GGML_ASSERT(false);
    }

    return out;
}

static int repack_iq4_nl_to_iq4_nl_8_bl(struct ggml_tensor * t, int interleave_block, const void * GGML_RESTRICT data, size_t data_size) {
    GGML_ASSERT(t->type == GGML_TYPE_IQ4_NL);
    GGML_ASSERT(interleave_block == 8);

    const block_iq4_nl   * src = (const block_iq4_nl   *)data;
          block_iq4_nlx8 * dst = (      block_iq4_nlx8 *)t->data;

    block_iq4_nl dst_tmp[8];

    int nrow = ggml_nrows(t);
    int nrows_interleaved = 8;
    int nblocks = t->ne[0] / QK4_NL;

    GGML_ASSERT(data_size == nrow * nblocks * sizeof(block_iq4_nl));

    if (t->ne[1] % nrows_interleaved != 0) {
        return -1;
    }

    for (int b = 0; b < nrow; b += nrows_interleaved) {
        for (int64_t x = 0; x < nblocks; x++) {
            for (int i = 0; i < nrows_interleaved; i++) {
                dst_tmp[i] = src[x + i * nblocks];
            }
            *dst++ = make_block_iq4_nlx8(dst_tmp, interleave_block);
        }
        src += nrows_interleaved * nblocks;
    }
    return 0;

    GGML_UNUSED(data_size);
}

static block_iq4_nlx16 make_block_iq4_nlx16(block_iq4_nl * in, unsigned int blck_size_interleave) {
    block_iq4_nlx16 out;

    for (int i = 0; i < 16; i++) {
        out.d[i] = in[i].d;
    }

    const int end = QK4_NL * 8 / blck_size_interleave;

    if (blck_size_interleave == 1) {
        for (int i = 0; i < end; ++i) {
            int src_id = i % 16;
            int src_offset = i / 16;
            int dst_offset = i;

            out.qs[dst_offset] = in[src_id].qs[src_offset];
        }
    } else {
        GGML_ASSERT(false);
    }

    return out;
}

static int repack_iq4_nl_to_iq4_nl_16_bl(struct ggml_tensor * t, int interleave_block, const void * GGML_RESTRICT data, size_t data_size) {
    GGML_ASSERT(t->type == GGML_TYPE_IQ4_NL);
    GGML_ASSERT(interleave_block == 1);

    const block_iq4_nl    * src = (const block_iq4_nl   *)data;
          block_iq4_nlx16 * dst = (      block_iq4_nlx16 *)t->data;

    block_iq4_nl dst_tmp[16];

    int nrow = ggml_nrows(t);
    int nrows_interleaved = 16;
    int nblocks = t->ne[0] / QK4_NL;

    GGML_ASSERT(data_size == nrow * nblocks * sizeof(block_iq4_nl));

    if (t->ne[1] % nrows_interleaved != 0) {
        return -1;
    }

    for (int b = 0; b < nrow; b += nrows_interleaved) {
        for (int64_t x = 0; x < nblocks; x++) {
            for (int i = 0; i < nrows_interleaved; i++) {
                dst_tmp[i] = src[x + i * nblocks];
            }
            *dst++ = make_block_iq4_nlx16(dst_tmp, interleave_block);
        }
        src += nrows_interleaved * nblocks;
    }
    return 0;

    GGML_UNUSED(data_size);
}

static block_mxfp4x4 make_block_mxfp4x4(block_mxfp4 * in, unsigned int blck_size_interleave) {
    block_mxfp4x4 out;

    for (int i = 0; i < 4; i++) {
        out.e[i] = in[i].e;
    }

    const int end = QK_MXFP4 * 2 / blck_size_interleave;

    if (blck_size_interleave == 4) {
        for (int i = 0; i < end; ++i) {
            int src_id = i % 4;
            int src_offset = (i / 4) * blck_size_interleave;
            int dst_offset = i * blck_size_interleave;

            memcpy(&out.qs[dst_offset], &in[src_id].qs[src_offset], sizeof(uint32_t));
        }
    } else {
        GGML_ASSERT(false);
    }

    return out;
}

static int repack_mxfp4_to_mxfp4_4_bl(struct ggml_tensor * t, int interleave_block, const void * GGML_RESTRICT data, size_t data_size) {
    GGML_ASSERT(t->type == GGML_TYPE_MXFP4);
    GGML_ASSERT(interleave_block == 4);

    const block_mxfp4   * src = (const block_mxfp4   *)data;
          block_mxfp4x4 * dst = (      block_mxfp4x4 *)t->data;

    block_mxfp4 dst_tmp[4];

    int nrow = ggml_nrows(t);
    int nrows_interleaved = 4;
    int nblocks = t->ne[0] / QK_MXFP4;

    GGML_ASSERT(data_size == nrow * nblocks * sizeof(block_mxfp4));

    if (t->ne[1] % nrows_interleaved != 0 || t->ne[0] % 8 != 0) {
        return -1;
    }

    for (int b = 0; b < nrow; b += nrows_interleaved) {
        for (int64_t x = 0; x < nblocks; x++) {
            for (int i = 0; i < nrows_interleaved; i++) {
                dst_tmp[i] = src[x + i * nblocks];
            }
            *dst++ = make_block_mxfp4x4(dst_tmp, interleave_block);
        }
        src += nrows_interleaved * nblocks;
    }
    return 0;

    GGML_UNUSED(data_size);
}

static block_mxfp4x8 make_block_mxfp4x8(block_mxfp4 * in, unsigned int blck_size_interleave) {
    block_mxfp4x8 out;

    for (int i = 0; i < 8; i++) {
        out.e[i] = in[i].e;
    }

    const int end = QK_MXFP4 * 4 / blck_size_interleave;

    if (blck_size_interleave == 8) {
        for (int i = 0; i < end; ++i) {
            int src_id = i % 8;
            int src_offset = (i / 8) * blck_size_interleave;
            int dst_offset = i * blck_size_interleave;

            memcpy(&out.qs[dst_offset], &in[src_id].qs[src_offset], sizeof(uint64_t));
        }
    } else {
        GGML_ASSERT(false);
    }

    return out;
}

static int repack_mxfp4_to_mxfp4_8_bl(struct ggml_tensor * t, int interleave_block, const void * GGML_RESTRICT data, size_t data_size) {
    GGML_ASSERT(t->type == GGML_TYPE_MXFP4);
    GGML_ASSERT(interleave_block == 8);

    const block_mxfp4   * src = (const block_mxfp4   *)data;
          block_mxfp4x8 * dst = (      block_mxfp4x8 *)t->data;

    block_mxfp4 dst_tmp[8];

    int nrow = ggml_nrows(t);
    int nrows_interleaved = 8;
    int nblocks = t->ne[0] / QK_MXFP4;

    GGML_ASSERT(data_size == nrow * nblocks * sizeof(block_mxfp4));

    if (t->ne[1] % nrows_interleaved != 0) {
        return -1;
    }

    for (int b = 0; b < nrow; b += nrows_interleaved) {
        for (int64_t x = 0; x < nblocks; x++) {
            for (int i = 0; i < nrows_interleaved; i++) {
                dst_tmp[i] = src[x + i * nblocks];
            }
            *dst++ = make_block_mxfp4x8(dst_tmp, interleave_block);
        }
        src += nrows_interleaved * nblocks;
    }
    return 0;

    GGML_UNUSED(data_size);
}

namespace ggml::cpu::repack {
// repack
template <typename BLOC_TYPE, int64_t INTER_SIZE, int64_t NB_COLS>
int repack(struct ggml_tensor *, const void *, size_t);

// TODO: generalise.
template <> int repack<block_q4_0, 4, 4>(struct ggml_tensor * t, const void * data, size_t data_size) {
    return repack_q4_0_to_q4_0_4_bl(t, 4, data, data_size);
}

template <> int repack<block_q4_0, 8, 4>(struct ggml_tensor * t, const void * data, size_t data_size) {
    return repack_q4_0_to_q4_0_4_bl(t, 8, data, data_size);
}

template <> int repack<block_q4_0, 8, 8>(struct ggml_tensor * t, const void * data, size_t data_size) {
    return repack_q4_0_to_q4_0_8_bl(t, 8, data, data_size);
}

template <> int repack<block_q4_K, 8, 8>(struct ggml_tensor * t, const void * data, size_t data_size) {
    return repack_q4_K_to_q4_K_8_bl(t, 8, data, data_size);
}

template <> int repack<block_q4_K, 4, 8>(struct ggml_tensor * t, const void * data, size_t data_size) {
    return repack_q4_K_to_q4_K_8_bl(t, 4, data, data_size);
}

template <> int repack<block_q2_K, 8, 8>(struct ggml_tensor * t, const void * data, size_t data_size) {
    return repack_q2_K_to_q2_K_8_bl(t, 8, data, data_size);
}

template <> int repack<block_q3_K, 8, 8>(struct ggml_tensor * t, const void * data, size_t data_size) {
    return repack_q3_K_to_q3_K_8_bl(t, 8, data, data_size);
}

template <> int repack<block_q5_K, 4, 8>(struct ggml_tensor * t, const void * data, size_t data_size) {
    return repack_q5_K_to_q5_K_8_bl(t, 4, data, data_size);
}

template <> int repack<block_q5_K, 8, 8>(struct ggml_tensor * t, const void * data, size_t data_size) {
    return repack_q5_K_to_q5_K_8_bl(t, 8, data, data_size);
}

template <> int repack<block_q6_K, 4, 8>(struct ggml_tensor * t, const void * data, size_t data_size) {
    return repack_q6_K_to_q6_K_8_bl(t, 4, data, data_size);
}

template <> int repack<block_q6_K, 8, 8>(struct ggml_tensor * t, const void * data, size_t data_size) {
    return repack_q6_K_to_q6_K_8_bl(t, 8, data, data_size);
}

template <> int repack<block_iq4_nl, 4, 4>(struct ggml_tensor * t, const void * data, size_t data_size) {
    return repack_iq4_nl_to_iq4_nl_4_bl(t, 4, data, data_size);
}

// TODO: needs to be revisited
//template <> int repack<block_iq4_nl, 8, 4>(struct ggml_tensor * t, const void * data, size_t data_size) {
//    return repack_iq4_nl_to_iq4_nl_4_bl(t, 8, data, data_size);
//}

template <> int repack<block_iq4_nl, 8, 8>(struct ggml_tensor * t, const void * data, size_t data_size) {
    return repack_iq4_nl_to_iq4_nl_8_bl(t, 8, data, data_size);
}

template <> int repack<block_iq1_s, 8, 8>(struct ggml_tensor * t, const void * data, size_t data_size) {
    return repack_iq1_s_to_iq1_s_8_bl(t, 8, data, data_size);
}

template <> int repack<block_iq1_m, 8, 8>(struct ggml_tensor * t, const void * data, size_t data_size) {
    return repack_iq1_m_to_iq1_m_8_bl(t, 8, data, data_size);
}

template <> int repack<block_iq2_xs, 8, 8>(struct ggml_tensor * t, const void * data, size_t data_size) {
    return repack_iq2_xs_to_iq2_xs_8_bl(t, 8, data, data_size);
}

template <> int repack<block_iq3_xxs, 8, 8>(struct ggml_tensor * t, const void * data, size_t data_size) {
    return repack_iq3_xxs_to_iq3_xxs_8_bl(t, 8, data, data_size);
}

template <> int repack<block_mxfp4, 4, 4>(struct ggml_tensor * t, const void * data, size_t data_size) {
    return repack_mxfp4_to_mxfp4_4_bl(t, 4, data, data_size);
}

template <> int repack<block_mxfp4, 8, 8>(struct ggml_tensor * t, const void * data, size_t data_size) {
    return repack_mxfp4_to_mxfp4_8_bl(t, 8, data, data_size);
}

template <> int repack<block_q8_0, 4, 4>(struct ggml_tensor * t, const void * data, size_t data_size) {
    return repack_q8_0_to_q8_0_4_bl(t, 4, data, data_size);
}

template <> int repack<block_q8_0, 8, 4>(struct ggml_tensor * t, const void * data, size_t data_size) {
    return repack_q8_0_to_q8_0_4_bl(t, 8, data, data_size);
}

template <> int repack<block_q8_0, 8, 8>(struct ggml_tensor * t, const void * data, size_t data_size) {
    return repack_q8_0_to_q8_0_8_bl(t, 8, data, data_size);
}

#if defined __riscv_zvfh
template <> int repack<block_q4_0, 1, 16>(struct ggml_tensor * t, const void * data, size_t data_size) {
    return repack_q4_0_to_q4_0_16_bl(t, 1, data, data_size);
}

template <> int repack<block_q4_K, 1, 16>(struct ggml_tensor * t, const void * data, size_t data_size) {
    return repack_q4_K_to_q4_K_16_bl(t, 1, data, data_size);
}

template <> int repack<block_iq4_nl, 1, 16>(struct ggml_tensor * t, const void * data, size_t data_size) {
    return repack_iq4_nl_to_iq4_nl_16_bl(t, 1, data, data_size);
}

template <> int repack<block_q8_0, 1, 16>(struct ggml_tensor * t, const void * data, size_t data_size) {
    return repack_q8_0_to_q8_0_16_bl(t, 1, data, data_size);
}

template <> int repack<block_q2_K, 1, 16>(struct ggml_tensor * t, const void * data, size_t data_size) {
    return repack_q2_K_to_q2_K_16_bl(t, 1, data, data_size);
}
#endif

// gemv
template <typename BLOC_TYPE, int64_t INTER_SIZE, int64_t NB_COLS, ggml_type PARAM_TYPE>
void gemv(int, float *, size_t, const void *, const void *, int, int);

template <> void gemv<block_q4_0, 4, 4, GGML_TYPE_Q8_0>(int n, float * s, size_t bs, const void * vx, const void * vy, int nr, int nc) {
    ggml_gemv_q4_0_4x4_q8_0(n, s, bs, vx, vy, nr, nc);
}

template <> void gemv<block_q4_0, 8, 4, GGML_TYPE_Q8_0>(int n, float * s, size_t bs, const void * vx, const void * vy, int nr, int nc) {
    ggml_gemv_q4_0_4x8_q8_0(n, s, bs, vx, vy, nr, nc);
}

template <> void gemv<block_q4_0, 8, 8, GGML_TYPE_Q8_0>(int n, float * s, size_t bs, const void * vx, const void * vy, int nr, int nc) {
    ggml_gemv_q4_0_8x8_q8_0(n, s, bs, vx, vy, nr, nc);
}

template <> void gemv<block_q8_0, 8, 8, GGML_TYPE_Q8_0>(int n, float * s, size_t bs, const void * vx, const void * vy, int nr, int nc) {
    ggml_gemv_q8_0_8x8_q8_0(n, s, bs, vx, vy, nr, nc);
}

template <>
void gemv<block_q2_K, 8, 8, GGML_TYPE_Q8_K>(int          n,
                                            float *      s,
                                            size_t       bs,
                                            const void * vx,
                                            const void * vy,
                                            int          nr,
                                            int          nc) {
    ggml_gemv_q2_K_8x8_q8_K(n, s, bs, vx, vy, nr, nc);
}

template <> void gemv<block_q3_K, 8, 8, GGML_TYPE_Q8_K>(int n, float * s, size_t bs, const void * vx, const void * vy, int nr, int nc) {
    ggml_gemv_q3_K_8x8_q8_K(n, s, bs, vx, vy, nr, nc);
}

template <> void gemv<block_q4_K, 4, 8, GGML_TYPE_Q8_K>(int n, float * s, size_t bs, const void * vx, const void * vy, int nr, int nc) {
    ggml_gemv_q4_K_8x4_q8_K(n, s, bs, vx, vy, nr, nc);
}

template <> void gemv<block_q4_K, 8, 8, GGML_TYPE_Q8_K>(int n, float * s, size_t bs, const void * vx, const void * vy, int nr, int nc) {
    ggml_gemv_q4_K_8x8_q8_K(n, s, bs, vx, vy, nr, nc);
}

template <> void gemv<block_q5_K, 4, 8, GGML_TYPE_Q8_K>(int n, float * s, size_t bs, const void * vx, const void * vy, int nr, int nc) {
    ggml_gemv_q5_K_8x4_q8_K(n, s, bs, vx, vy, nr, nc);
}

template <> void gemv<block_q5_K, 8, 8, GGML_TYPE_Q8_K>(int n, float * s, size_t bs, const void * vx, const void * vy, int nr, int nc) {
    ggml_gemv_q5_K_8x8_q8_K(n, s, bs, vx, vy, nr, nc);
}

template <> void gemv<block_q6_K, 4, 8, GGML_TYPE_Q8_K>(int n, float * s, size_t bs, const void * vx, const void * vy, int nr, int nc) {
    ggml_gemv_q6_K_8x4_q8_K(n, s, bs, vx, vy, nr, nc);
}

template <> void gemv<block_q6_K, 8, 8, GGML_TYPE_Q8_K>(int n, float * s, size_t bs, const void * vx, const void * vy, int nr, int nc) {
    ggml_gemv_q6_K_8x8_q8_K(n, s, bs, vx, vy, nr, nc);
}

template <> void gemv<block_iq4_nl, 4, 4, GGML_TYPE_Q8_0>(int n, float * s, size_t bs, const void * vx, const void * vy, int nr, int nc) {
    ggml_gemv_iq4_nl_4x4_q8_0(n, s, bs, vx, vy, nr, nc);
}

template <> void gemv<block_iq4_nl, 8, 8, GGML_TYPE_Q8_0>(int n, float * s, size_t bs, const void * vx, const void * vy, int nr, int nc) {
    ggml_gemv_iq4_nl_8x8_q8_0(n, s, bs, vx, vy, nr, nc);
}

template <> void gemv<block_iq1_s, 8, 8, GGML_TYPE_Q8_K>(int n, float * s, size_t bs, const void * vx, const void * vy, int nr, int nc) {
    ggml_gemv_iq1_s_8x8_q8_K(n, s, bs, vx, vy, nr, nc);
}

template <> void gemv<block_iq1_m, 8, 8, GGML_TYPE_Q8_K>(int n, float * s, size_t bs, const void * vx, const void * vy, int nr, int nc) {
    ggml_gemv_iq1_m_8x8_q8_K(n, s, bs, vx, vy, nr, nc);
}

template <> void gemv<block_iq2_xs, 8, 8, GGML_TYPE_Q8_K>(int n, float * s, size_t bs, const void * vx, const void * vy, int nr, int nc) {
    ggml_gemv_iq2_xs_8x8_q8_K(n, s, bs, vx, vy, nr, nc);
}

template <> void gemv<block_iq3_xxs, 8, 8, GGML_TYPE_Q8_K>(int n, float * s, size_t bs, const void * vx, const void * vy, int nr, int nc) {
    ggml_gemv_iq3_xxs_8x8_q8_K(n, s, bs, vx, vy, nr, nc);
}

template <> void gemv<block_mxfp4, 4, 4, GGML_TYPE_Q8_0>(int n, float * s, size_t bs, const void * vx, const void * vy, int nr, int nc) {
    ggml_gemv_mxfp4_4x4_q8_0(n, s, bs, vx, vy, nr, nc);
}

template <> void gemv<block_mxfp4, 8, 8, GGML_TYPE_Q8_0>(int n, float * s, size_t bs, const void * vx, const void * vy, int nr, int nc) {
    ggml_gemv_mxfp4_8x8_q8_0(n, s, bs, vx, vy, nr, nc);
}

template <> void gemv<block_q8_0, 4, 4, GGML_TYPE_Q8_0>(int n, float * s, size_t bs, const void * vx, const void * vy, int nr, int nc) {
    ggml_gemv_q8_0_4x4_q8_0(n, s, bs, vx, vy, nr, nc);
}

template <> void gemv<block_q8_0, 8, 4, GGML_TYPE_Q8_0>(int n, float * s, size_t bs, const void * vx, const void * vy, int nr, int nc) {
    ggml_gemv_q8_0_4x8_q8_0(n, s, bs, vx, vy, nr, nc);
}

#if defined __riscv_zvfh
template <> void gemv<block_q4_0, 1, 16, GGML_TYPE_Q8_0>(int n, float * s, size_t bs, const void * vx, const void * vy, int nr, int nc) {
    ggml_gemv_q4_0_16x1_q8_0(n, s, bs, vx, vy, nr, nc);
}

template <> void gemv<block_q4_K, 1, 16, GGML_TYPE_Q8_K>(int n, float * s, size_t bs, const void * vx, const void * vy, int nr, int nc) {
    ggml_gemv_q4_K_16x1_q8_K(n, s, bs, vx, vy, nr, nc);
}

template <> void gemv<block_iq4_nl, 1, 16, GGML_TYPE_Q8_0>(int n, float * s, size_t bs, const void * vx, const void * vy, int nr, int nc) {
    ggml_gemv_iq4_nl_16x1_q8_0(n, s, bs, vx, vy, nr, nc);
}

template <> void gemv<block_q8_0, 1, 16, GGML_TYPE_Q8_0>(int n, float * s, size_t bs, const void * vx, const void * vy, int nr, int nc) {
    ggml_gemv_q8_0_16x1_q8_0(n, s, bs, vx, vy, nr, nc);
}

template <> void gemv<block_q2_K, 1, 16, GGML_TYPE_Q8_K>(int n, float * s, size_t bs, const void * vx, const void * vy, int nr, int nc) {
    ggml_gemv_q2_K_16x1_q8_K(n, s, bs, vx, vy, nr, nc);
}
#endif

// gemm
template <typename BLOC_TYPE, int64_t INTER_SIZE, int64_t NB_COLS, ggml_type PARAM_TYPE>
void gemm(int, float *, size_t, const void *, const void *, int, int);

template <> void gemm<block_q4_0, 4, 4, GGML_TYPE_Q8_0>(int n, float * s, size_t bs, const void * vx, const void * vy, int nr, int nc) {
    ggml_gemm_q4_0_4x4_q8_0(n, s, bs, vx, vy, nr, nc);
}

template <> void gemm<block_q4_0, 8, 4, GGML_TYPE_Q8_0>(int n, float * s, size_t bs, const void * vx, const void * vy, int nr, int nc) {
    ggml_gemm_q4_0_4x8_q8_0(n, s, bs, vx, vy, nr, nc);
}

template <>
void gemm<block_q4_0, 8, 8, GGML_TYPE_Q8_0>(int          n,
                                            float *      s,
                                            size_t       bs,
                                            const void * vx,
                                            const void * vy,
                                            int          nr,
                                            int          nc) {
    ggml_gemm_q4_0_8x8_q8_0(n, s, bs, vx, vy, nr, nc);
}

template <> void gemm<block_q8_0, 8, 8, GGML_TYPE_Q8_0>(int n, float * s, size_t bs, const void * vx, const void * vy, int nr, int nc) {
    ggml_gemm_q8_0_8x8_q8_0(n, s, bs, vx, vy, nr, nc);
}

template <> void gemm<block_q2_K, 8, 8, GGML_TYPE_Q8_K>(int n, float * s, size_t bs, const void * vx, const void * vy, int nr, int nc) {
    ggml_gemm_q2_K_8x8_q8_K(n, s, bs, vx, vy, nr, nc);
}

template <> void gemm<block_q3_K, 8, 8, GGML_TYPE_Q8_K>(int n, float * s, size_t bs, const void * vx, const void * vy, int nr, int nc) {
    ggml_gemm_q3_K_8x8_q8_K(n, s, bs, vx, vy, nr, nc);
}

template <> void gemm<block_q4_K, 4, 8, GGML_TYPE_Q8_K>(int n, float * s, size_t bs, const void * vx, const void * vy, int nr, int nc) {
    ggml_gemm_q4_K_8x4_q8_K(n, s, bs, vx, vy, nr, nc);
}

template <> void gemm<block_q4_K, 8, 8, GGML_TYPE_Q8_K>(int n, float * s, size_t bs, const void * vx, const void * vy, int nr, int nc) {
    ggml_gemm_q4_K_8x8_q8_K(n, s, bs, vx, vy, nr, nc);
}

template <> void gemm<block_q5_K, 4, 8, GGML_TYPE_Q8_K>(int n, float * s, size_t bs, const void * vx, const void * vy, int nr, int nc) {
    ggml_gemm_q5_K_8x4_q8_K(n, s, bs, vx, vy, nr, nc);
}

template <> void gemm<block_q5_K, 8, 8, GGML_TYPE_Q8_K>(int n, float * s, size_t bs, const void * vx, const void * vy, int nr, int nc) {
    ggml_gemm_q5_K_8x8_q8_K(n, s, bs, vx, vy, nr, nc);
}

template <> void gemm<block_q6_K, 4, 8, GGML_TYPE_Q8_K>(int n, float * s, size_t bs, const void * vx, const void * vy, int nr, int nc) {
    ggml_gemm_q6_K_8x4_q8_K(n, s, bs, vx, vy, nr, nc);
}

template <> void gemm<block_q6_K, 8, 8, GGML_TYPE_Q8_K>(int n, float * s, size_t bs, const void * vx, const void * vy, int nr, int nc) {
    ggml_gemm_q6_K_8x8_q8_K(n, s, bs, vx, vy, nr, nc);
}

template <> void gemm<block_iq4_nl, 4, 4, GGML_TYPE_Q8_0>(int n, float * s, size_t bs, const void * vx, const void * vy, int nr, int nc) {
    ggml_gemm_iq4_nl_4x4_q8_0(n, s, bs, vx, vy, nr, nc);
}

template <> void gemm<block_iq4_nl, 8, 8, GGML_TYPE_Q8_0>(int n, float * s, size_t bs, const void * vx, const void * vy, int nr, int nc) {
    ggml_gemm_iq4_nl_8x8_q8_0(n, s, bs, vx, vy, nr, nc);
}

template <> void gemm<block_iq1_s, 8, 8, GGML_TYPE_Q8_K>(int n, float * s, size_t bs, const void * vx, const void * vy, int nr, int nc) {
    ggml_gemm_iq1_s_8x8_q8_K(n, s, bs, vx, vy, nr, nc);
}

template <> void gemm<block_iq1_m, 8, 8, GGML_TYPE_Q8_K>(int n, float * s, size_t bs, const void * vx, const void * vy, int nr, int nc) {
    ggml_gemm_iq1_m_8x8_q8_K(n, s, bs, vx, vy, nr, nc);
}

template <> void gemm<block_iq2_xs, 8, 8, GGML_TYPE_Q8_K>(int n, float * s, size_t bs, const void * vx, const void * vy, int nr, int nc) {
    ggml_gemm_iq2_xs_8x8_q8_K(n, s, bs, vx, vy, nr, nc);
}

template <> void gemm<block_iq3_xxs, 8, 8, GGML_TYPE_Q8_K>(int n, float * s, size_t bs, const void * vx, const void * vy, int nr, int nc) {
    ggml_gemm_iq3_xxs_8x8_q8_K(n, s, bs, vx, vy, nr, nc);
}

template <> void gemm<block_mxfp4, 4, 4, GGML_TYPE_Q8_0>(int n, float * s, size_t bs, const void * vx, const void * vy, int nr, int nc) {
    ggml_gemm_mxfp4_4x4_q8_0(n, s, bs, vx, vy, nr, nc);
}

template <> void gemm<block_mxfp4, 8, 8, GGML_TYPE_Q8_0>(int n, float * s, size_t bs, const void * vx, const void * vy, int nr, int nc) {
    ggml_gemm_mxfp4_8x8_q8_0(n, s, bs, vx, vy, nr, nc);
}

template <> void gemm<block_q8_0, 4, 4, GGML_TYPE_Q8_0>(int n, float * s, size_t bs, const void * vx, const void * vy, int nr, int nc) {
    ggml_gemm_q8_0_4x4_q8_0(n, s, bs, vx, vy, nr, nc);
}

template <> void gemm<block_q8_0, 8, 4, GGML_TYPE_Q8_0>(int n, float * s, size_t bs, const void * vx, const void * vy, int nr, int nc) {
    ggml_gemm_q8_0_4x8_q8_0(n, s, bs, vx, vy, nr, nc);
}

#if defined __riscv_zvfh
template <> void gemm<block_q4_0, 1, 16, GGML_TYPE_Q8_0>(int n, float * s, size_t bs, const void * vx, const void * vy, int nr, int nc) {
    ggml_gemm_q4_0_16x1_q8_0(n, s, bs, vx, vy, nr, nc);
}

template <> void gemm<block_q4_K, 1, 16, GGML_TYPE_Q8_K>(int n, float * s, size_t bs, const void * vx, const void * vy, int nr, int nc) {
    ggml_gemm_q4_K_16x1_q8_K(n, s, bs, vx, vy, nr, nc);
}

template <> void gemm<block_iq4_nl, 1, 16, GGML_TYPE_Q8_0>(int n, float * s, size_t bs, const void * vx, const void * vy, int nr, int nc) {
    ggml_gemm_iq4_nl_16x1_q8_0(n, s, bs, vx, vy, nr, nc);
}

template <> void gemm<block_q8_0, 1, 16, GGML_TYPE_Q8_0>(int n, float * s, size_t bs, const void * vx, const void * vy, int nr, int nc) {
    ggml_gemm_q8_0_16x1_q8_0(n, s, bs, vx, vy, nr, nc);
}

template <> void gemm<block_q2_K, 1, 16, GGML_TYPE_Q8_K>(int n, float * s, size_t bs, const void * vx, const void * vy, int nr, int nc) {
    ggml_gemm_q2_K_16x1_q8_K(n, s, bs, vx, vy, nr, nc);
}
#endif

// Minimum number of src1 rows routed to gemm in forward_mul_mat_one_chunk.
// Kernels whose vectorized tile loop handles any nr % 4 == 0 (currently
// q2_K/q3_K/q5_K/iq1_s/iq1_m 8x8) can take small batches directly; the others need
// full 16-row tiles or they fall back to the ~60x slower scalar generic path for
// all rows below 16.
template <typename BLOC_TYPE, int64_t INTER_SIZE, int64_t NB_COLS, ggml_type PARAM_TYPE>
struct gemm_min_nrows { static constexpr int64_t value = 16; };

template <> struct gemm_min_nrows<block_q2_K, 8, 8, GGML_TYPE_Q8_K> { static constexpr int64_t value = 4; };
template <> struct gemm_min_nrows<block_q3_K, 8, 8, GGML_TYPE_Q8_K> { static constexpr int64_t value = 4; };
template <> struct gemm_min_nrows<block_q5_K, 8, 8, GGML_TYPE_Q8_K> { static constexpr int64_t value = 4; };
template <> struct gemm_min_nrows<block_iq1_s, 8, 8, GGML_TYPE_Q8_K> { static constexpr int64_t value = 4; };
template <> struct gemm_min_nrows<block_iq1_m, 8, 8, GGML_TYPE_Q8_K> { static constexpr int64_t value = 4; };
template <> struct gemm_min_nrows<block_iq2_xs, 8, 8, GGML_TYPE_Q8_K> { static constexpr int64_t value = 4; };
template <> struct gemm_min_nrows<block_iq3_xxs, 8, 8, GGML_TYPE_Q8_K> { static constexpr int64_t value = 4; };

// Tile of src1 rows gathered per gemm call in forward_mul_mat_id (multiple of 4).
static constexpr int64_t MMID_GEMM_TILE = 32;

class tensor_traits_base : public ggml::cpu::tensor_traits {
  public:
    virtual int repack(struct ggml_tensor * t, const void * data, size_t data_size) = 0;
};

template <typename BLOC_TYPE, int64_t INTER_SIZE, int64_t NB_COLS, ggml_type PARAM_TYPE> class tensor_traits : public tensor_traits_base {

    bool work_size(int n_threads, const struct ggml_tensor * op, size_t & size) override {
        // not realy a GGML_TYPE_Q8_0 but same size.
        switch (op->op) {
            case GGML_OP_MUL_MAT:
                {
                    size = ggml_row_size(PARAM_TYPE, ggml_nelements(op->src[1]));
                    return true;
                }
            case GGML_OP_MUL_MAT_ID:
                {
                    size = ggml_row_size(PARAM_TYPE, ggml_nelements(op->src[1]));
                    size = GGML_PAD(size, sizeof(int64_t)); // + padding for next block.

                    const int64_t ne02 = op->src[0]->ne[2]; // n_as, n_expert
                    const int64_t ne12 = op->src[1]->ne[2]; // n_tokens

                    const size_t sizeof_mmid_row_mapping = sizeof(int64_t);

                    size += sizeof_mmid_row_mapping*ne02*(ne12 + 1);

                    // per-expert per-node-window atomic row claim counters for NUMA EP
                    // work stealing (one cache line per expert per NUMA node)
                    size  = GGML_PAD(size, GGML_EP_CACHE_LINE);
                    size += GGML_EP_CACHE_LINE*ne02*GGML_NUMA_MAX_NODES;

                    // per-thread gather/scatter tiles for the mul_mat_id gemm path:
                    // an expert's selected src1 rows are scattered (slot, token) pairs,
                    // so tiles of MMID_GEMM_TILE f32 rows are gathered (4 at a time)
                    // into contiguous scratch, quantized to the interleaved layout the
                    // gemm kernels consume, multiplied, and scattered back. Column
                    // ranges are bounded by the per-thread row split (else branch) and
                    // the NUMA EP claim quantum of 16 rows.
                    const int64_t nbw1 = ggml_row_size(PARAM_TYPE, op->src[1]->ne[0]);
                    const int64_t ne01 = op->src[0]->ne[1];
                    const int64_t cols = MAX((int64_t) 16, (ne01 + n_threads - 1)/n_threads + NB_COLS);
                    size += n_threads*(MMID_GEMM_TILE*nbw1 + MMID_GEMM_TILE*cols*sizeof(float) +
                                     4*op->src[1]->ne[0]*sizeof(float));

                    return true;
                }
            default:
                // GGML_ABORT("fatal error");
                break;
        }
        return false;
    }

    bool compute_forward(struct ggml_compute_params * params, struct ggml_tensor * op) override {
        switch (op->op) {
            case GGML_OP_MUL_MAT:
                forward_mul_mat(params, op);
                return true;
            case GGML_OP_MUL_MAT_ID:
                forward_mul_mat_id(params, op);
                return true;
            default:
                // GGML_ABORT("fatal error");
                break;
        }
        return false;
    }

    void forward_mul_mat_one_chunk(ggml_compute_params * params,
                                   ggml_tensor *         op,
                                   int64_t               src0_start,
                                   int64_t               src0_end,
                                   int64_t               src1_start,
                                   int64_t               src1_end) {
        const ggml_tensor * src0 = op->src[0];
        const ggml_tensor * src1 = op->src[1];
        ggml_tensor *       dst  = op;

        GGML_TENSOR_BINARY_OP_LOCALS

        const size_t src1_col_stride = ggml_row_size(PARAM_TYPE, ne10);

        GGML_ASSERT(ne03 == 1 && ne13 == 1);
        GGML_ASSERT(ne12 % ne02 == 0);
        const int64_t r2 = ne12 / ne02;

        const int64_t i12 = src1_start / ne1;
        const int64_t i11 = src1_start - i12 * ne1;

        // Determine batch index
        const int64_t i02 = i12 / r2;

        const int64_t i1 = i11;
        const int64_t i2 = i12;

        const char * src0_ptr = (const char *) ggml_numa_tensor_data(src0, ggml_numa_node_for_thread(params->ith, params->nth)) + i02 * nb02;
        const char * src1_ptr = (const char *) params->wdata + (i11 + i12 * ne11) * src1_col_stride;
        char *       dst_ptr  = ((char *) dst->data + (i1 * nb1 + i2 * nb2));

        const int64_t nrows = src1_end - src1_start;
        const int64_t ncols = src0_end - src0_start;

        GGML_ASSERT(src1_ptr + src1_col_stride * nrows <= (const char *) params->wdata + params->wsize);

        // Use gemm for the src1 rows the type's vectorized kernel can take natively
        // (multiples of 4 down to gemm_min_nrows); the rest go through per-row gemv.
        // Kernels limited to 16-row tiles would otherwise run their leftover (or all)
        // rows on a scalar generic path that is ~60x slower than re-reading weights
        // once per gemv row.
        const int64_t nrows_gemm = nrows >= gemm_min_nrows<BLOC_TYPE, INTER_SIZE, NB_COLS, PARAM_TYPE>::value
                                 ? nrows - (nrows % 4) : 0;
        if (nrows_gemm > 0) {
            gemm<BLOC_TYPE, INTER_SIZE, NB_COLS, PARAM_TYPE>(ne00, (float *) (dst_ptr) + src0_start, nb1 / nb0,
                                                             src0_ptr + src0_start * nb01, src1_ptr,
                                                             nrows_gemm, ncols);
        }
        for (int iter = nrows_gemm; iter < nrows; iter++) {
            gemv<BLOC_TYPE, INTER_SIZE, NB_COLS, PARAM_TYPE>(ne00, (float *) (dst_ptr + (iter * nb1)) + src0_start,
                                                             ne01, src0_ptr + src0_start * nb01,
                                                             src1_ptr + (src1_col_stride * iter), 1 /* nrows */, ncols);
        }
    }

    void forward_mul_mat(ggml_compute_params * params, ggml_tensor * op) {
        const ggml_tensor * src0 = op->src[0];
        const ggml_tensor * src1 = op->src[1];
        ggml_tensor *       dst  = op;

        GGML_TENSOR_BINARY_OP_LOCALS

        const int ith = params->ith;
        const int nth = params->nth;

        GGML_ASSERT(ne0 == ne01);
        GGML_ASSERT(ne1 == ne11);
        GGML_ASSERT(ne2 == ne12);
        GGML_ASSERT(ne3 == ne13);

        // dst cannot be transposed or permuted
        GGML_ASSERT(nb0 == sizeof(float));
        GGML_ASSERT(nb0 <= nb1);
        GGML_ASSERT(nb1 <= nb2);
        GGML_ASSERT(nb2 <= nb3);

        // TODO: General batched mul mat for 4D tensors
        // Currently only supports 3D tensors
        GGML_ASSERT(ne03 == 1);
        GGML_ASSERT(ne13 == 1);
        GGML_ASSERT(ne3 == 1);

        GGML_ASSERT(src1->type == GGML_TYPE_F32);

        // src0 may be a contiguous 3D view of a 2D weight (e.g. MLA attention reshapes a
        // lora weight into groups). The rows of each ne02 plane stay contiguous in the
        // repacked layout and are processed plane by plane, driven by the src1 batching
        // below (ne12 % ne02 == 0 is asserted in forward_mul_mat_one_chunk). Each plane
        // must start on an interleave group boundary of the repacked rows.
        if (ggml_n_dims(src0) == 3) {
            GGML_ASSERT(ggml_is_contiguous(src0));
            GGML_ASSERT(ne01 % NB_COLS == 0);
        } else {
            GGML_ASSERT(ggml_n_dims(src0) == 2);
        }

        char *       wdata = static_cast<char *>(params->wdata);
        const size_t nbw1  = ggml_row_size(PARAM_TYPE, ne10);
        const size_t nbw2  = nbw1 * ne11;

        assert(params->wsize >= nbw2 * ne12);

        const ggml_from_float_t from_float = ggml_get_type_traits_cpu(PARAM_TYPE)->from_float;

        const size_t nbw0 = ggml_type_size(PARAM_TYPE);

        // RMS_NORM(+MUL) absorption prologue (see ggml-cpu.c): if the norm feeding src1
        // was elided from the graph, materialize it into src1->data before quantizing it
        {
            const struct ggml_cpu_absorb_entry * ab = ggml_cpu_absorb_find(params->threadpool, dst);
            if (ab != NULL) {
                ggml_cpu_absorb_materialize(params, ab, ggml_blck_size(PARAM_TYPE), true);
            }
        }

        // INFO: Quantization is done in planes to avoid extra complexity in chunking.
        // Flattening dimensions not multiple of INTER_SIZE would require extra handling depending on how
        // the planes are broadcast.
        // debug: per-phase timing for slow mul_mats (env GGML_MM_PHASE=1, thread 0 only)
        static int mm_phase_prof_r = -1;
        if (mm_phase_prof_r < 0) { const char * e = getenv("GGML_MM_PHASE"); mm_phase_prof_r = (e && atoi(e)) ? 1 : 0; }
        const int64_t mmp_t0 = (mm_phase_prof_r == 1 && ith == 0) ? ggml_time_us() : 0;
        for (int64_t i12 = 0; i12 < ne12; i12++) {
            char * data_ptr  = (char *) src1->data + i12 * nb12;
            char * wdata_ptr = wdata + i12 * nbw2;

            for (int64_t i11 = ith * 4; i11 < ne11 - ne11 % 4; i11 += nth * 4) {
                ggml_quantize_mat_t<INTER_SIZE, PARAM_TYPE>((float *) (data_ptr + i11 * nb11),
                                                            (void *) (wdata_ptr + i11 * nbw1), 4, ne10);
            }

            // Rows not covered by the 4-row groups (always the case when decoding a single
            // token) are quantized by slicing each row across all threads
            const int64_t blck = ggml_blck_size(PARAM_TYPE);
            const int64_t block_start = (ith * ne10/blck) / nth;
            const int64_t block_end   = ((ith + 1) * ne10/blck) / nth;
            for (int64_t i11 = ne11 - ne11 % 4; i11 < ne11; i11++) {
                from_float((float *) (data_ptr + i11 * nb11 + block_start * blck * nb10),
                           (void *) (wdata_ptr + i11 * nbw1 + block_start * nbw0),
                           (block_end - block_start) * blck);
            }
        }

        // disable for NUMA
        const bool disable_chunking = ggml_is_numa();

        // 4x chunks per thread
        // (rows of a single src0 plane; for a 3D src0 view the planes are batched like src1)
        const int64_t nr0 = src0->ne[1];

        int     nth_scaled  = nth * 4;
        int64_t chunk_size0 = (nr0 + nth_scaled - 1) / nth_scaled;
        int64_t nchunk0     = (nr0 + chunk_size0 - 1) / chunk_size0;

        // src1 is chunked only by full planes.
        // When we flatten we need to address dimensions not multiple of the q8 INTER_SIZE
        // to route them thorugh GEMV.
        // nchunk1 = ne12 also avoids messing the chunking for models with no 3d tensors
        // to avoid affecting their performance
        int64_t nchunk1 = ne12;

        // Ensure minimum chunk size to avoid alignment issues with high thread counts
        // Minimum chunk size should be at least NB_COLS to prevent overlapping chunks after alignment
        const int64_t min_chunk_size = NB_COLS;
        if (nchunk0 > 0 && (nr0 / nchunk0) < min_chunk_size && nr0 >= min_chunk_size) {
            nchunk0 = (nr0 + min_chunk_size - 1) / min_chunk_size;
        }

        int64_t dr0 = (nr0 + nchunk0 - 1) / nchunk0;
        // Only increase nchunk0 to nth if it won't make chunks too small
        if (nth == 1 || ((nchunk0 < nth || disable_chunking) && (nr0 + nth - 1) / nth >= min_chunk_size)) {
            nchunk0 = nth;
            dr0 = (nr0 + nchunk0 - 1) / nchunk0;
        }

        // Ensure nchunk doesn't exceed the number of rows divided by minimum chunk size
        // This prevents creating too many tiny chunks that could overlap after alignment
        const int64_t max_nchunk = (nr0 + min_chunk_size - 1) / min_chunk_size;
        nchunk0                  = MIN(nchunk0, max_nchunk);

        if (ith == 0) {
            // Every thread starts at ith, so the first unprocessed chunk is nth.  This save a bit of coordination right at the start.
            ggml_threadpool_chunk_set(params->threadpool, nth);
        }

        const int64_t mmp_t1 = mmp_t0 != 0 ? ggml_time_us() : 0; // after src1 quantization

        ggml_barrier(params->threadpool);

        const int64_t mmp_t2 = mmp_t0 != 0 ? ggml_time_us() : 0; // after internal barrier

        // The first chunk comes from our thread_id, the rest will get auto-assigned.
        int current_chunk = ith;

        while (current_chunk < nchunk0 * nchunk1) {
            const int64_t ith0 = current_chunk % nchunk0;
            const int64_t ith1 = current_chunk / nchunk0;

            int64_t src0_start = dr0 * ith0;
            int64_t src0_end   = MIN(src0_start + dr0, nr0);

            // full-plane range for src1
            int64_t src1_start = ith1 * ne11;
            int64_t src1_end = (ith1 + 1) * ne11;

            // Align boundaries to NB_COLS - round up to ensure all data is included
            // The chunk size limiting above ensures chunks are large enough to prevent overlaps
            src0_start = (src0_start % NB_COLS) ? src0_start + NB_COLS - (src0_start % NB_COLS) : src0_start;
            src0_end   = (src0_end % NB_COLS) ? src0_end + NB_COLS - (src0_end % NB_COLS) : src0_end;
            src0_end   = MIN(src0_end, nr0);

            // Make sure current plane is the last one before exiting
            if (src0_start >= src0_end) {
                current_chunk = ggml_threadpool_chunk_add(params->threadpool, 1);
                continue;
            }

            forward_mul_mat_one_chunk(params, dst, src0_start, src0_end, src1_start, src1_end);

            current_chunk = ggml_threadpool_chunk_add(params->threadpool, 1);
        }

        if (mmp_t0 != 0) {
            const int64_t mmp_t3 = ggml_time_us();
            if (mmp_t3 - mmp_t0 > 2000) {
                fprintf(stderr, "[mm-phase] %s repack ne0=%lld ne1=%lld quant=%.2f barr=%.2f gemm=%.2f ms\n",
                        op->name, (long long) src0->ne[1], (long long) ne1,
                        (mmp_t1 - mmp_t0) / 1000.0, (mmp_t2 - mmp_t1) / 1000.0, (mmp_t3 - mmp_t2) / 1000.0);
            }
        }
    }

    // --- NUMA EP steal diagnostics (GGML_NUMA_EP_DEBUG=1), temporary ---
    struct ep_debug_stats {
        std::atomic<int64_t> t0{0};            // start of current call (set by thread 0)
        std::atomic<int64_t> done{0};          // threads finished in current call
        std::atomic<int64_t> first_end{INT64_MAX};
        std::atomic<int64_t> last_end{0};
        std::atomic<int64_t> calls{0};
        std::atomic<int64_t> p0_rows[2]{{0},{0}};  // rows claimed in local phase, per node
        std::atomic<int64_t> p1_rows[2]{{0},{0}};  // rows claimed in steal phase, per node
        std::atomic<int64_t> p1_entries{0};        // thread-entries into phase 1
        std::atomic<int64_t> p1_remaining{0};      // unclaimed remote rows visible at phase-1 entry (sum)
        std::atomic<int64_t> p1_idle_entries{0};   // phase-1 entries that claimed 0 rows
        std::atomic<int64_t> busy_sum{0};          // sum of per-thread op durations (us)
        std::atomic<int64_t> p1_sum{0};            // sum of per-thread phase-1 durations (us)
        std::atomic<int64_t> spread_sum{0};        // sum of (last_end - first_end) per call (us)
        std::atomic<int64_t> wall_sum{0};          // sum of last_end per call (us)
    };
    static ep_debug_stats & ep_dbg() { static ep_debug_stats s; return s; }
    static bool ep_dbg_on() {
        static const bool on = []() {
            const char * e = getenv("GGML_NUMA_EP_DEBUG");
            return e && atoi(e) != 0;
        }();
        return on;
    }

    static void ep_dbg_finish(int64_t t_start, int64_t t_p0_end, int64_t p1_claimed, int nth, int ne01) {
        const int64_t t_end = ggml_time_us();
        ep_dbg().busy_sum.fetch_add(t_end - t_start, std::memory_order_relaxed);
        if (t_p0_end > 0) {
            ep_dbg().p1_sum.fetch_add(t_end - t_p0_end, std::memory_order_relaxed);
        }
        if (p1_claimed == 0 && t_p0_end > 0) {
            ep_dbg().p1_idle_entries.fetch_add(1, std::memory_order_relaxed);
        }
        const int64_t rel = t_end - ep_dbg().t0.load(std::memory_order_relaxed);
        int64_t cur = ep_dbg().first_end.load(std::memory_order_relaxed);
        while (rel < cur && !ep_dbg().first_end.compare_exchange_weak(cur, rel, std::memory_order_relaxed)) {}
        cur = ep_dbg().last_end.load(std::memory_order_relaxed);
        while (rel > cur && !ep_dbg().last_end.compare_exchange_weak(cur, rel, std::memory_order_relaxed)) {}
        if (ep_dbg().done.fetch_add(1, std::memory_order_acq_rel) == nth - 1) {
            // last thread of the call: fold per-call stats
            const int64_t last = ep_dbg().last_end.exchange(0, std::memory_order_relaxed);
            const int64_t first = ep_dbg().first_end.exchange(INT64_MAX, std::memory_order_relaxed);
            ep_dbg().done.store(0, std::memory_order_relaxed);
            ep_dbg().spread_sum.fetch_add(last - first, std::memory_order_relaxed);
            ep_dbg().wall_sum.fetch_add(last, std::memory_order_relaxed);
            const int64_t c = ep_dbg().calls.fetch_add(1, std::memory_order_relaxed) + 1;
            if ((c & 0xFF) == 0) {
                fprintf(stderr,
                    "[ep-dbg] calls=%lld ne01=%d wall=%lld spread=%lld busy=%lld p1time=%lld "
                    "p0rows0=%lld p0rows1=%lld p1rows0=%lld p1rows1=%lld p1_entries=%lld p1_idle=%lld p1_remaining_avg=%lld\n",
                    (long long) c, ne01,
                    (long long) (ep_dbg().wall_sum.load() / c),
                    (long long) (ep_dbg().spread_sum.load() / c),
                    (long long) (ep_dbg().busy_sum.load() / (c * nth)),
                    (long long) (ep_dbg().p1_sum.load() / (c * nth)),
                    (long long) (ep_dbg().p0_rows[0].load() / c),
                    (long long) (ep_dbg().p0_rows[1].load() / c),
                    (long long) (ep_dbg().p1_rows[0].load() / c),
                    (long long) (ep_dbg().p1_rows[1].load() / c),
                    (long long) (ep_dbg().p1_entries.load() / c),
                    (long long) ep_dbg().p1_idle_entries.load(),
                    (long long) (ep_dbg().p1_entries.load() ? ep_dbg().p1_remaining.load() / ep_dbg().p1_entries.load() : 0));
            }
        }
    }

    void forward_mul_mat_id(ggml_compute_params * params, ggml_tensor * op) {
        const ggml_tensor * src0 = op->src[0];
        const ggml_tensor * src1 = op->src[1];
        const ggml_tensor * ids  = op->src[2];
        ggml_tensor *       dst  = op;

        GGML_TENSOR_BINARY_OP_LOCALS

        const int ith = params->ith;
        const int nth = params->nth;

        const ggml_from_float_t from_float = ggml_get_type_traits_cpu(PARAM_TYPE)->from_float;

        // we don't support permuted src0 or src1
        GGML_ASSERT(nb00 == ggml_type_size(src0->type));
        GGML_ASSERT(nb10 == ggml_type_size(src1->type));

        // dst cannot be transposed or permuted
        GGML_ASSERT(nb0 == sizeof(float));
        GGML_ASSERT(nb0 <= nb1);
        GGML_ASSERT(nb1 <= nb2);
        GGML_ASSERT(nb2 <= nb3);

        GGML_ASSERT(ne03 == 1);
        GGML_ASSERT(ne13 == 1);
        GGML_ASSERT(ne3  == 1);

        GGML_ASSERT(src1->type == GGML_TYPE_F32);

        // row groups
        const int n_ids = ids->ne[0]; // n_expert_used
        const int n_as  = ne02;       // n_expert

        const size_t nbw1 = ggml_row_size(PARAM_TYPE, ne10);
        const size_t nbw2 = nbw1*ne11;
        const size_t nbw3 = nbw2*ne12;

        struct mmid_row_mapping {
            int32_t i1;
            int32_t i2;
        };

        // per-thread gather/scatter tiles for the gemm path (laid out after row_claim,
        // sized in work_size): interleaved src1 tile + dense dst tile + 4-row f32
        // gather buffer. Column ranges are bounded by the per-thread row split
        // (large-batch branch) and the NUMA EP claim quantum of 16 rows.
        const int64_t mmid_max_cols = MAX((int64_t) 16, (ne01 + nth - 1)/nth + NB_COLS);
        const size_t  mmid_scratch_stride = MMID_GEMM_TILE*nbw1 + MMID_GEMM_TILE*mmid_max_cols*sizeof(float) +
                                            4*ne00*sizeof(float);

        GGML_ASSERT(params->wsize >=
                (GGML_PAD(nbw3, sizeof(int64_t)) +
                 GGML_PAD(n_as*(ne12 + 1)*sizeof(mmid_row_mapping), GGML_EP_CACHE_LINE) +
                 n_as*GGML_NUMA_MAX_NODES*GGML_EP_CACHE_LINE +
                 nth*mmid_scratch_stride));

        auto * wdata          = (char *)params->wdata;
        auto * wdata_src1_end = (char *)wdata + GGML_PAD(nbw3, sizeof(int64_t));

        // total of [n_as][ne12 + 1] elements of type mmid_row_mapping (2*int32_t = int64_t)
        auto * matrix_row_counts = (int64_t *) (wdata_src1_end);                                        // [n_as]
        struct mmid_row_mapping * matrix_rows = (struct mmid_row_mapping *) (matrix_row_counts + n_as); // [n_as][ne12]

        // per-expert per-node-window atomic row claim counters for NUMA EP work stealing
        // (laid out as [n_as][GGML_NUMA_MAX_NODES], one cache line each)
        auto * row_claim = (std::atomic_int *) ((char *) wdata_src1_end +
            GGML_PAD(n_as*(ne12 + 1)*sizeof(mmid_row_mapping), GGML_EP_CACHE_LINE));

        char * mmid_scratch = (char *) row_claim + n_as*GGML_NUMA_MAX_NODES*GGML_EP_CACHE_LINE;

        // src1: float32 => param type
        for (int64_t i12 = 0; i12 < ne12; ++i12) {
            for (int64_t i11 = ith; i11 < ne11; i11 += nth) {
                from_float((float *)((char *) src1->data + i12 * nb12 + i11 * nb11),
                           (void *)               (wdata + i12 * nbw2 + i11 * nbw1),
                           ne10);
            }
        }

#define MMID_MATRIX_ROW(row_id, i1) matrix_rows[(row_id) * ne12 + (i1)]

        if (ith == 0) {
            // initialize matrix_row_counts
            memset(matrix_row_counts, 0, n_as * sizeof(int64_t));

            // group rows by src0 matrix
            for (int32_t iid1 = 0; iid1 < ids->ne[1]; ++iid1) {
                for (int32_t id = 0; id < n_ids; ++id) {
                    const int32_t i02 =
                        *(const int32_t *) ((const char *) ids->data + iid1 * ids->nb[1] + id * ids->nb[0]);

                    GGML_ASSERT(i02 >= 0 && i02 < n_as);

                    MMID_MATRIX_ROW(i02, matrix_row_counts[i02]) = { id, iid1 };
                    matrix_row_counts[i02] += 1;
                }
            }
        }

        // NUMA expert parallelism (GGML_NUMA_EP): every expert's rows are split into
        // per-node windows ([w*ep_win, min((w+1)*ep_win, ne01)) belongs to node w),
        // placed on node w by llama_model::numa_ep_place_experts. Local phase: each
        // thread claims rows of its own node's window of EVERY selected expert, so all
        // sockets stream every selected expert from local memory in parallel — balanced
        // by construction no matter how the router distributes experts across nodes.
        // (Expert-granular placement left the tail bounded by a single socket's DRAM
        // bandwidth; cross-UPI steals cannot add bandwidth to data that lives on one
        // node.) Steal phase (only when the token count exceeds
        // ggml_cpu_numa_ep_steal_min_tokens): threads then claim remaining rows of the
        // other nodes' windows (remote reads) to smooth out stragglers.
        // Rows are claimed in NB_COLS-aligned blocks, each dst row has a single writer,
        // and no barriers are needed inside the loops.
        const bool ep = ggml_cpu_numa_ep_active();
        const int ep_nodes = ep ? ggml_numa_node_count() : 1;
        const int ep_node = ep ? ggml_numa_node_for_thread(ith, nth) : 0;
        const int ep_phases = ep && ne12 > ggml_cpu_numa_ep_steal_min_tokens() ? 2 : 1;

        // rows are claimed in blocks of ep_chunk (a multiple of NB_COLS); the per-node
        // window size must match llama_model::numa_ep_place_experts. Smaller blocks
        // shrink the per-thread tail quantum at batch=1 (one block ≈ one tail unit).
        const int64_t ep_chunk = 16;
        const int64_t ep_win = ep ? (((ne01 + ep_nodes - 1) / ep_nodes + ep_chunk - 1) / ep_chunk) * ep_chunk : 0;

        if (ep) {
            for (int cur_a = ith; cur_a < n_as; cur_a += nth) {
                for (int w = 0; w < ep_nodes; ++w) {
                    std::atomic_store_explicit(row_claim + cur_a*GGML_NUMA_MAX_NODES + w, 0, std::memory_order_relaxed);
                }
            }
        }

        ggml_barrier(params->threadpool);

        // Compute dst columns [ws, we) of expert cur_a for all its selected src1 rows.
        // The selected rows are scattered (slot, token) pairs, so tiles of
        // MMID_GEMM_TILE rows are gathered from src1 (4 f32 rows at a time), quantized
        // into the interleaved layout the gemm kernels consume (same scheme as
        // forward_mul_mat's src1 quantization), multiplied with gemm, and scattered
        // back; the tail rows (< 4, or column ranges not NB_COLS-aligned) go through
        // per-row gemv as before.
        auto mmid_rows_range = [&](int64_t cur_a, const char * src0_cur, int64_t ws, int64_t we) {
            const int64_t cne1  = matrix_row_counts[cur_a];
            const int64_t ncols = we - ws;

            int64_t nrows_gemm = 0;
            if (cne1 >= gemm_min_nrows<BLOC_TYPE, INTER_SIZE, NB_COLS, PARAM_TYPE>::value && ncols % NB_COLS == 0) {
                nrows_gemm = cne1 - (cne1 % 4);
            }

            char * vy_tile  = mmid_scratch + ith*mmid_scratch_stride;
            auto * dst_tile = (float *) (vy_tile + MMID_GEMM_TILE*nbw1);
            auto * x4       = dst_tile + MMID_GEMM_TILE*mmid_max_cols; // 4*ne00 floats

            for (int64_t base = 0; base < nrows_gemm; base += MMID_GEMM_TILE) {
                const int64_t tr = MIN(MMID_GEMM_TILE, nrows_gemm - base);
                for (int64_t k = 0; k < tr; k += 4) {
                    for (int64_t m = 0; m < 4; m++) {
                        const struct mmid_row_mapping rm = MMID_MATRIX_ROW(cur_a, base + k + m);
                        memcpy(x4 + m*ne00, (const char *) src1->data + rm.i2*nb12 + (rm.i1 % ne11)*nb11,
                               ne00*sizeof(float));
                    }
                    ggml_quantize_mat_t<INTER_SIZE, PARAM_TYPE>(x4, vy_tile + k*nbw1, 4, ne00);
                }
                gemm<BLOC_TYPE, INTER_SIZE, NB_COLS, PARAM_TYPE>(ne00, dst_tile, ncols,
                                                                 src0_cur + ws*nb01, vy_tile, tr, ncols);
                for (int64_t k = 0; k < tr; k++) {
                    const struct mmid_row_mapping rm = MMID_MATRIX_ROW(cur_a, base + k);
                    memcpy((char *) dst->data + rm.i1*nb1 + rm.i2*nb2 + ws*sizeof(float),
                           dst_tile + k*ncols, ncols*sizeof(float));
                }
            }

            for (int64_t ir1 = nrows_gemm; ir1 < cne1; ir1++) {
                struct mmid_row_mapping row_mapping = MMID_MATRIX_ROW(cur_a, ir1);

                const int id = row_mapping.i1;  // selected expert index

                const int64_t i11 = id % ne11;
                const int64_t i12 = row_mapping.i2;  // row index in src1

                const auto * src1_col = (const char *) wdata + (i11 * nbw1 + i12 * nbw2);

                gemv<BLOC_TYPE, INTER_SIZE, NB_COLS, PARAM_TYPE>(
                    ne00, (float *) ((char *) dst->data + (id * nb1 + i12 * nb2)) + ws, ne01,
                    src0_cur + ws*nb01, src1_col, 1, ncols);
            }
        };

        const int64_t dbg_t_start = (ep && ep_dbg_on()) ? ggml_time_us() : 0;
        if (ep && ep_dbg_on() && ith == 0) {
            ep_dbg().t0.store(dbg_t_start, std::memory_order_relaxed);
        }

        if (ep) {
            int64_t dbg_t_p0_end = 0;
            int64_t dbg_p1_claimed = 0;

            for (int phase = 0; phase < ep_phases; phase++) {
                if (ep_dbg_on() && phase == 1) {
                    // how much remote work is still unclaimed as this thread enters the steal phase
                    int64_t remaining = 0;
                    for (int cur_a = 0; cur_a < n_as; ++cur_a) {
                        if (matrix_row_counts[cur_a] == 0) {
                            continue;
                        }
                        for (int w = 0; w < ep_nodes; ++w) {
                            if (w == ep_node) {
                                continue;
                            }
                            const int64_t w_lo = (int64_t) w*ep_win;
                            const int64_t w_hi = MIN(w_lo + ep_win, ne01);
                            if (w_lo >= w_hi) {
                                continue;
                            }
                            const int64_t claimed = std::atomic_load_explicit(row_claim + cur_a*GGML_NUMA_MAX_NODES + w, std::memory_order_relaxed);
                            if (claimed < w_hi - w_lo) {
                                remaining += w_hi - w_lo - claimed;
                            }
                        }
                    }
                    ep_dbg().p1_entries.fetch_add(1, std::memory_order_relaxed);
                    ep_dbg().p1_remaining.fetch_add(remaining, std::memory_order_relaxed);
                }
                for (int cur_a = 0; cur_a < n_as; ++cur_a) {
                    const int64_t cne1 = matrix_row_counts[cur_a];

                    if (cne1 == 0) {
                        continue;
                    }

                    const auto * src0_cur = (const char *) ggml_numa_tensor_data(src0, ggml_numa_node_for_thread(ith, nth)) + cur_a*nb02;

                    for (int w = 0; w < ep_nodes; ++w) {
                        if (phase == 0 ? (w != ep_node) : (w == ep_node)) {
                            continue; // handled in the other phase
                        }

                        const int64_t w_lo = (int64_t) w*ep_win;
                        const int64_t w_hi = MIN(w_lo + ep_win, ne01);
                        if (w_lo >= w_hi) {
                            continue;
                        }

                        std::atomic_int * ctr = row_claim + cur_a*GGML_NUMA_MAX_NODES + w;

                        for (;;) {
                            const int64_t r0 = w_lo + std::atomic_fetch_add_explicit(ctr, (int) ep_chunk, std::memory_order_relaxed);
                            if (r0 >= w_hi) {
                                break;
                            }
                            const int64_t r1 = MIN(r0 + ep_chunk, w_hi);

                            if (ep_dbg_on()) {
                                if (phase == 0) {
                                    ep_dbg().p0_rows[ep_node].fetch_add(r1 - r0, std::memory_order_relaxed);
                                } else {
                                    ep_dbg().p1_rows[ep_node].fetch_add(r1 - r0, std::memory_order_relaxed);
                                    dbg_p1_claimed += r1 - r0;
                                }
                            }

                            mmid_rows_range(cur_a, src0_cur, r0, r1);
                        }
                    }
                }
                if (ep_dbg_on() && phase == 0) {
                    dbg_t_p0_end = ggml_time_us();
                }
            }
            if (ep_dbg_on()) {
                ep_dbg_finish(dbg_t_start, ep_phases > 1 ? dbg_t_p0_end : 0, dbg_p1_claimed, nth, ne01);
            }
        } else if (ne12 <= 8) {
            // small-batch (TG) path: expert-first partitioning. assign the selected
            // experts to disjoint thread groups and split rows only within a group,
            // so each thread streams a long contiguous row range of a single expert
            // instead of a thin slice of every expert (pipeline-friendly). the math
            // is unchanged: every src0 row of every selected expert is computed
            // exactly once, by exactly one thread.
            int64_t n_sel = 0;
            for (int cur_a = 0; cur_a < n_as; ++cur_a) {
                n_sel += matrix_row_counts[cur_a] > 0 ? 1 : 0;
            }

            if (n_sel > 0) {
                const int64_t n_groups   = MIN(n_sel, (int64_t) nth);
                const int64_t group_base = nth / n_groups; // min threads per group
                const int64_t group_rem  = nth % n_groups; // first group_rem groups get one extra thread

                // locate this thread's group and its rank within the group
                const int64_t big_threads = group_rem * (group_base + 1);
                int64_t group, rank, group_size;
                if (ith < big_threads) {
                    group      = ith / (group_base + 1);
                    rank       = ith % (group_base + 1);
                    group_size = group_base + 1;
                } else {
                    group      = group_rem + (ith - big_threads) / group_base;
                    rank       = (ith - big_threads) % group_base;
                    group_size = group_base;
                }

                // compacted index range of the selected experts handled by this group
                const int64_t sel_begin = (group * n_sel) / n_groups;
                const int64_t sel_end   = ((group + 1) * n_sel) / n_groups;

                int64_t sel = 0;
                for (int cur_a = 0; cur_a < n_as; ++cur_a) {
                    const int64_t cne1 = matrix_row_counts[cur_a];

                    if (cne1 == 0) {
                        continue;
                    }

                    const bool mine = sel >= sel_begin && sel < sel_end;
                    sel++;
                    if (!mine) {
                        continue;
                    }

                    const auto * src0_cur = (const char *) ggml_numa_tensor_data(src0, ggml_numa_node_for_thread(ith, nth)) + cur_a*nb02;

                    //const int64_t nr0 = ne01; // src0 rows
                    const int64_t nr1 = cne1; // src1 rows

                    int64_t src0_cur_start = (rank * ne01) / group_size;
                    int64_t src0_cur_end   = ((rank + 1) * ne01) / group_size;

                    // Align boundaries to NB_COLS - round up to ensure all data is included
                    src0_cur_start = (src0_cur_start % NB_COLS) ? src0_cur_start + NB_COLS - (src0_cur_start % NB_COLS) : src0_cur_start;
                    src0_cur_end   = (src0_cur_end   % NB_COLS) ? src0_cur_end   + NB_COLS - (src0_cur_end   % NB_COLS) : src0_cur_end;
                    if (src0_cur_end > ne01) {
                        src0_cur_end = ne01;
                    }

                    if (src0_cur_start >= src0_cur_end) {
                        continue;
                    }

                    for (int ir1 = 0; ir1 < nr1; ir1++) {
                        struct mmid_row_mapping row_mapping = MMID_MATRIX_ROW(cur_a, ir1);

                        const int id = row_mapping.i1;  // selected expert index

                        const int64_t i11 = id % ne11;
                        const int64_t i12 = row_mapping.i2;  // row index in src1

                        const int64_t i1 = id;               // selected expert index
                        const int64_t i2 = i12;              // row

                        const auto * src1_col = (const char *) wdata + (i11 * nbw1 + i12 * nbw2);

                        gemv<BLOC_TYPE, INTER_SIZE, NB_COLS, PARAM_TYPE>(
                            ne00, (float *) ((char *) dst->data + (i1 * nb1 + i2 * nb2)) + src0_cur_start, ne01,
                            src0_cur + src0_cur_start * nb01, src1_col, 1, src0_cur_end - src0_cur_start);
                    }
                }
            }
        } else {
            // compute each matrix multiplication in sequence
            for (int cur_a = 0; cur_a < n_as; ++cur_a) {
                const int64_t cne1 = matrix_row_counts[cur_a];

                if (cne1 == 0) {
                    continue;
                }

                const auto * src0_cur = (const char *) ggml_numa_tensor_data(src0, ggml_numa_node_for_thread(ith, nth)) + cur_a*nb02;

                int64_t src0_cur_start = (ith * ne01) / nth;
                int64_t src0_cur_end   = ((ith + 1) * ne01) / nth;

                // Align boundaries to NB_COLS - round up to ensure all data is included
                src0_cur_start = (src0_cur_start % NB_COLS) ? src0_cur_start + NB_COLS - (src0_cur_start % NB_COLS) : src0_cur_start;
                src0_cur_end   = (src0_cur_end   % NB_COLS) ? src0_cur_end   + NB_COLS - (src0_cur_end   % NB_COLS) : src0_cur_end;
                if (src0_cur_end > ne01) {
                    src0_cur_end = ne01;
                }

                if (src0_cur_start >= src0_cur_end) {
                    continue;
                }

                mmid_rows_range(cur_a, src0_cur, src0_cur_start, src0_cur_end);
            }
        }
#undef MMID_MATRIX_ROW
    }

    int repack(struct ggml_tensor * t, const void * data, size_t data_size) override {
        GGML_LOG_DEBUG("%s: repack tensor %s with %s_%dx%d\n", __func__, t->name, ggml_type_name(t->type),
                       (int) NB_COLS, (int) INTER_SIZE);
        return ggml::cpu::repack::repack<BLOC_TYPE, INTER_SIZE, NB_COLS>(t, data, data_size);
    }
};

}  // namespace ggml::cpu::repack

static const ggml::cpu::tensor_traits * ggml_repack_get_optimal_repack_type(const struct ggml_tensor * cur) {
    // instance for Q4
    static const ggml::cpu::repack::tensor_traits<block_q4_0, 4, 4, GGML_TYPE_Q8_0> q4_0_4x4_q8_0;
    static const ggml::cpu::repack::tensor_traits<block_q4_0, 8, 4, GGML_TYPE_Q8_0> q4_0_4x8_q8_0;
    static const ggml::cpu::repack::tensor_traits<block_q4_0, 8, 8, GGML_TYPE_Q8_0> q4_0_8x8_q8_0;

    // instance for Q4_K
    static const ggml::cpu::repack::tensor_traits<block_q4_K, 4, 8, GGML_TYPE_Q8_K> q4_K_8x4_q8_K;
    static const ggml::cpu::repack::tensor_traits<block_q4_K, 8, 8, GGML_TYPE_Q8_K> q4_K_8x8_q8_K;

    // instance for Q5_K
    static const ggml::cpu::repack::tensor_traits<block_q5_K, 4, 8, GGML_TYPE_Q8_K> q5_K_8x4_q8_K;
    static const ggml::cpu::repack::tensor_traits<block_q5_K, 8, 8, GGML_TYPE_Q8_K> q5_K_8x8_q8_K;

    // instance for Q6_K
    static const ggml::cpu::repack::tensor_traits<block_q6_K, 4, 8, GGML_TYPE_Q8_K> q6_K_8x4_q8_K;
    static const ggml::cpu::repack::tensor_traits<block_q6_K, 8, 8, GGML_TYPE_Q8_K> q6_K_8x8_q8_K;

    // instance for Q2
    static const ggml::cpu::repack::tensor_traits<block_q2_K, 8, 8, GGML_TYPE_Q8_K> q2_K_8x8_q8_K;

    // instance for Q3
    static const ggml::cpu::repack::tensor_traits<block_q3_K, 8, 8, GGML_TYPE_Q8_K> q3_K_8x8_q8_K;

    // instance for IQ4
    static const ggml::cpu::repack::tensor_traits<block_iq4_nl, 4, 4, GGML_TYPE_Q8_0> iq4_nl_4x4_q8_0;
    static const ggml::cpu::repack::tensor_traits<block_iq4_nl, 8, 8, GGML_TYPE_Q8_0> iq4_nl_8x8_q8_0;

    // instances for IQ1
    static const ggml::cpu::repack::tensor_traits<block_iq1_s, 8, 8, GGML_TYPE_Q8_K> iq1_s_8x8_q8_K;
    static const ggml::cpu::repack::tensor_traits<block_iq1_m, 8, 8, GGML_TYPE_Q8_K> iq1_m_8x8_q8_K;

    // instances for IQ2/IQ3
    static const ggml::cpu::repack::tensor_traits<block_iq2_xs, 8, 8, GGML_TYPE_Q8_K>  iq2_xs_8x8_q8_K;
    static const ggml::cpu::repack::tensor_traits<block_iq3_xxs, 8, 8, GGML_TYPE_Q8_K> iq3_xxs_8x8_q8_K;

    // instance for MXFP4
    static const ggml::cpu::repack::tensor_traits<block_mxfp4, 4, 4, GGML_TYPE_Q8_0> mxfp4_4x4_q8_0;
    static const ggml::cpu::repack::tensor_traits<block_mxfp4, 8, 8, GGML_TYPE_Q8_0> mxfp4_8x8_q8_0;

    // instance for Q8_0
    static const ggml::cpu::repack::tensor_traits<block_q8_0, 4, 4, GGML_TYPE_Q8_0> q8_0_4x4_q8_0;
    static const ggml::cpu::repack::tensor_traits<block_q8_0, 8, 4, GGML_TYPE_Q8_0> q8_0_4x8_q8_0;
    static const ggml::cpu::repack::tensor_traits<block_q8_0, 8, 8, GGML_TYPE_Q8_0> q8_0_8x8_q8_0;

    // instances for RISC-V
    //
    // These implement outer-product style matrix multiplication kernels with
    // an interleave of 1.
#if defined __riscv_zvfh
    static const ggml::cpu::repack::tensor_traits<block_q4_0, 1, 16, GGML_TYPE_Q8_0> q4_0_16x1_q8_0;
    static const ggml::cpu::repack::tensor_traits<block_q4_K, 1, 16, GGML_TYPE_Q8_K> q4_K_16x1_q8_K;
    static const ggml::cpu::repack::tensor_traits<block_iq4_nl, 1, 16, GGML_TYPE_Q8_0> iq4_nl_16x1_q8_0;
    static const ggml::cpu::repack::tensor_traits<block_q8_0, 1, 16, GGML_TYPE_Q8_0> q8_0_16x1_q8_0;
    static const ggml::cpu::repack::tensor_traits<block_q2_K, 1, 16, GGML_TYPE_Q8_K> q2_K_16x1_q8_K;
#endif

    if (cur->type == GGML_TYPE_Q4_0) {
        if (ggml_cpu_has_avx2() || (ggml_cpu_has_sve() && ggml_cpu_has_matmul_int8() && ggml_cpu_get_sve_cnt() == QK8_0)) {
            if (cur->ne[1] % 8 == 0) {
                return &q4_0_8x8_q8_0;
            }
        }
        if (ggml_cpu_has_neon() && ggml_cpu_has_matmul_int8()) {
            if (cur->ne[1] % 4 == 0) {
                return &q4_0_4x8_q8_0;
            }
        }
        if (ggml_cpu_has_neon() && ggml_cpu_has_dotprod()) {
            if (cur->ne[1] % 4 == 0) {
                return &q4_0_4x4_q8_0;
            }
        }
        if (ggml_cpu_has_riscv_v()) {
            #if defined __riscv_zvfh
            switch (__riscv_vlenb() * 8) {
                case 128:  { break; } // TODO
                case 256:  { if (cur->ne[1] % 16 == 0) { return &q4_0_16x1_q8_0; } break; }
                case 512:  { break; } // TODO
                case 1024: { break; } // TODO
                default:   { return nullptr; }
            }
            #endif
        }
    } else if (cur->type == GGML_TYPE_Q4_K) {
        if (ggml_cpu_has_avx2()) {
            if (cur->ne[1] % 8 == 0) {
                return &q4_K_8x8_q8_K;
            }
        }
        if (ggml_cpu_has_neon() && ggml_cpu_has_matmul_int8()) {
            if (cur->ne[1] % 8 == 0) {
                return &q4_K_8x8_q8_K;
            }
        }
        if (ggml_cpu_has_neon() && ggml_cpu_has_dotprod()) {
            if (cur->ne[1] % 8 == 0) {
                return &q4_K_8x4_q8_K;
            }
        }
        if (ggml_cpu_has_riscv_v()) {
            #if defined __riscv_zvfh
            switch (__riscv_vlenb() * 8) {
                case 128:  { break; } // TODO
                case 256:  { if (cur->ne[1] % 16 == 0) { return &q4_K_16x1_q8_K; } break; }
                case 512:  { break; } // TODO
                case 1024: { break; } // TODO
                default:   { return nullptr; }
            }
            #endif
        }
    } else if (cur->type == GGML_TYPE_Q2_K) {
        if (ggml_cpu_has_avx512()) {
            if (cur->ne[1] % 8 == 0) {
                return &q2_K_8x8_q8_K;
            }
        }
        if (ggml_cpu_has_riscv_v()) {
            #if defined __riscv_zvfh
            switch (__riscv_vlenb() * 8) {
                case 128:  { break; } // TODO
                case 256:  { if (cur->ne[1] % 16 == 0) { return &q2_K_16x1_q8_K; } break; }
                case 512:  { break; } // TODO
                case 1024: { break; } // TODO
                default:   { return nullptr; }
            }
            #endif
        }
    } else if (cur->type == GGML_TYPE_Q3_K) {
        if (ggml_cpu_has_avx512()) {
            if (cur->ne[1] % 8 == 0) {
                return &q3_K_8x8_q8_K;
            }
        }
    } else if (cur->type == GGML_TYPE_Q5_K) {
        if (ggml_cpu_has_avx512()) {
            if (cur->ne[1] % 8 == 0) {
                return &q5_K_8x8_q8_K;
            }
        }
        if (ggml_cpu_has_neon() && ggml_cpu_has_matmul_int8()) {
            if (cur->ne[1] % 8 == 0) {
                return &q5_K_8x8_q8_K;
            }
        }
        if (ggml_cpu_has_neon() && ggml_cpu_has_dotprod()) {
            if (cur->ne[1] % 8 == 0) {
                return &q5_K_8x4_q8_K;
            }
        }
    } else if (cur->type == GGML_TYPE_Q6_K) {
        if (ggml_cpu_has_avx512()) {
            if (cur->ne[1] % 8 == 0) {
                return &q6_K_8x8_q8_K;
            }
        }
        if (ggml_cpu_has_neon() && ggml_cpu_has_matmul_int8()) {
            if (cur->ne[1] % 8 == 0) {
                return &q6_K_8x8_q8_K;
            }
        }
        if (ggml_cpu_has_neon() && ggml_cpu_has_dotprod()) {
            if (cur->ne[1] % 8 == 0) {
                return &q6_K_8x4_q8_K;
            }
        }
    } else if (cur->type == GGML_TYPE_IQ1_S) {
        if (ggml_cpu_has_avx512()) {
            if (cur->ne[1] % 8 == 0) {
                return &iq1_s_8x8_q8_K;
            }
        }
    } else if (cur->type == GGML_TYPE_IQ1_M) {
        if (ggml_cpu_has_avx512()) {
            if (cur->ne[1] % 8 == 0) {
                return &iq1_m_8x8_q8_K;
            }
        }
    } else if (cur->type == GGML_TYPE_IQ2_XS) {
        // the 8x8 kernels are only implemented for AVX512 with VNNI (vpdpbusd/vpdpwssd)
        // and VBMI (vpermb sign expansion); on anything older the scalar generic kernels
        // are ~60x slower than the plain vec_dot path, so do not repack there
        if (ggml_cpu_has_avx512() && ggml_cpu_has_avx512_vnni() && ggml_cpu_has_avx512_vbmi()) {
            if (cur->ne[1] % 8 == 0) {
                return &iq2_xs_8x8_q8_K;
            }
        }
    } else if (cur->type == GGML_TYPE_IQ3_XXS) {
        // same VNNI + VBMI requirement as IQ2_XS above
        if (ggml_cpu_has_avx512() && ggml_cpu_has_avx512_vnni() && ggml_cpu_has_avx512_vbmi()) {
            if (cur->ne[1] % 8 == 0) {
                return &iq3_xxs_8x8_q8_K;
            }
        }
    } else if (cur->type == GGML_TYPE_IQ4_NL) {
        if (ggml_cpu_has_avx2()) {
            if (cur->ne[1] % 8 == 0) {
                return &iq4_nl_8x8_q8_0;
            }
        }
        if (ggml_cpu_has_neon() && ggml_cpu_has_dotprod()) {
            if (cur->ne[1] % 4 == 0) {
                return &iq4_nl_4x4_q8_0;
            }
        }
        if (ggml_cpu_has_riscv_v()) {
            #if defined __riscv_zvfh
            switch (__riscv_vlenb() * 8) {
                case 128:  { break; } // TODO
                case 256:  { if (cur->ne[1] % 16 == 0) { return &iq4_nl_16x1_q8_0; } break; }
                case 512:  { break; } // TODO
                case 1024: { break; } // TODO
                default:   { return nullptr; }
            }
            #endif
        }
    } else if (cur->type == GGML_TYPE_MXFP4) {
        if (ggml_cpu_has_avx2()) {
            if (cur->ne[1] % 8 == 0) {
                return &mxfp4_8x8_q8_0;
            }
        }
        if (ggml_cpu_has_neon() && ggml_cpu_has_dotprod()) {
            if (cur->ne[1] % 4 == 0) {
                return &mxfp4_4x4_q8_0;
            }
        }
    } else if (cur->type == GGML_TYPE_Q8_0) {
        if (ggml_cpu_has_avx512()) {
            if (cur->ne[1] % 8 == 0) {
                return &q8_0_8x8_q8_0;
            }
        }
        if (ggml_cpu_has_neon() && ggml_cpu_has_matmul_int8()) {
            if (cur->ne[1] % 4 == 0) {
                return &q8_0_4x8_q8_0;
            }
        }
        if (ggml_cpu_has_neon() && ggml_cpu_has_dotprod()) {
            if (cur->ne[1] % 4 == 0) {
                return &q8_0_4x4_q8_0;
            }
        }
        if (ggml_cpu_has_riscv_v()) {
            #if defined __riscv_zvfh
            switch (__riscv_vlenb() * 8) {
                case 128:  { break; } // TODO
                case 256:  { if (cur->ne[1] % 16 == 0) { return &q8_0_16x1_q8_0; } break; }
                case 512:  { break; } // TODO
                case 1024: { break; } // TODO
                default:   { return nullptr; }
            }
            #endif
        }
    }

    return nullptr;
}

static enum ggml_status ggml_backend_cpu_repack_buffer_init_tensor(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor) {
    tensor->extra = (void *) const_cast<ggml::cpu::tensor_traits *>(ggml_repack_get_optimal_repack_type(tensor));

    GGML_UNUSED(buffer);
    return GGML_STATUS_SUCCESS;
}

static void ggml_backend_cpu_repack_buffer_set_tensor(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor,
                                                       const void * data, size_t offset, size_t size) {
    GGML_ASSERT(offset == 0);
    GGML_ASSERT(size == ggml_nbytes(tensor));

    auto tensor_traits = (ggml::cpu::repack::tensor_traits_base *) tensor->extra;
    auto OK            = tensor_traits->repack(tensor, data, size);

    GGML_ASSERT(OK == 0);
    GGML_UNUSED(buffer);
}

static const char * ggml_backend_cpu_repack_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
    return "CPU_REPACK";

    GGML_UNUSED(buft);
}

static ggml_backend_buffer_t ggml_backend_cpu_repack_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    ggml_backend_buffer_t buffer = ggml_backend_buft_alloc_buffer(ggml_backend_cpu_buffer_type(), size);

    if (buffer == nullptr) {
        return nullptr;
    }

    buffer->buft              = buft;
    buffer->iface.init_tensor = ggml_backend_cpu_repack_buffer_init_tensor;
    buffer->iface.set_tensor  = ggml_backend_cpu_repack_buffer_set_tensor;
    buffer->iface.get_tensor  = nullptr;
    buffer->iface.cpy_tensor  = nullptr;
    return buffer;
}

static size_t ggml_backend_cpu_repack_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    return TENSOR_ALIGNMENT;

    GGML_UNUSED(buft);
}

static bool ggml_backend_cpu_repack_buffer_type_is_host(ggml_backend_buffer_type_t buft) {
    // the repack buffer is allocated through the regular CPU buffer type, so it is host
    // memory; reporting this lets NUMA mirroring duplicate repacked weights per node
    return true;

    GGML_UNUSED(buft);
}

namespace ggml::cpu::repack {
class extra_buffer_type : ggml::cpu::extra_buffer_type {
    bool supports_op(ggml_backend_dev_t, const struct ggml_tensor * op) override {
        if (    op->op == GGML_OP_MUL_MAT &&
                op->src[0]->buffer &&
                (ggml_n_dims(op->src[0]) == 2) &&
                op->src[0]->buffer->buft == ggml_backend_cpu_repack_buffer_type() &&
                ggml_repack_get_optimal_repack_type(op->src[0])
                ) {
            if (op->src[1]->buffer && !ggml_backend_buft_is_host(op->src[1]->buffer->buft)) {
                return false;
            }
            if (op->src[1]->type == GGML_TYPE_F32) {
                return true;
            }
            //if (op->src[1]->type == GGML_TYPE_Q8_0) {
            //    return true;
            //}
            // may be possible if Q8_0 packed...
        } else if (op->op == GGML_OP_MUL_MAT_ID
                && op->src[0]->buffer
                && (ggml_n_dims(op->src[0]) == 3)
                && op->src[0]->buffer->buft == ggml_backend_cpu_repack_buffer_type()
                && ggml_repack_get_optimal_repack_type(op->src[0])
                ) {
            if (op->src[1]->buffer && !ggml_backend_buft_is_host(op->src[1]->buffer->buft)) {
                return false;
            }
            if (op->src[1]->type == GGML_TYPE_F32) {
                return true;
            }
            //if (op->src[1]->type == GGML_TYPE_Q8_0) {
            //    return true;
            //}
        }
        return false;
    }

    ggml::cpu::tensor_traits * get_tensor_traits(const struct ggml_tensor * op) override {
        if (op->op == GGML_OP_MUL_MAT || op->op == GGML_OP_MUL_MAT_ID) {
            if (op->src[0]->buffer && op->src[0]->buffer->buft == ggml_backend_cpu_repack_buffer_type()) {
                return (ggml::cpu::tensor_traits *) op->src[0]->extra;
            }
        }
        return nullptr;
    }
};
}  // namespace ggml::cpu::repack

ggml_backend_buffer_type_t ggml_backend_cpu_repack_buffer_type(void) {
    static struct ggml_backend_buffer_type ggml_backend_cpu_buffer_type_repack = {
        /* .iface    = */ {
                           /* .get_name         = */ ggml_backend_cpu_repack_buffer_type_get_name,
                           /* .alloc_buffer     = */ ggml_backend_cpu_repack_buffer_type_alloc_buffer,
                           /* .get_alignment    = */ ggml_backend_cpu_repack_buffer_type_get_alignment,
                           /* .get_max_size     = */ nullptr,  // defaults to SIZE_MAX
                           /* .get_alloc_size   = */ nullptr,  // defaults to ggml_nbytes
                           /* .is_host          = */ ggml_backend_cpu_repack_buffer_type_is_host,
                           },
        /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_cpu_reg(), 0),
        /* .context = */ new ggml::cpu::repack::extra_buffer_type(),
    };

    return &ggml_backend_cpu_buffer_type_repack;
}
