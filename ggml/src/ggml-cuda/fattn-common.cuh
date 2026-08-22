#pragma once

#include "common.cuh"
#include "convert.cuh"
#include "vecdotq.cuh"

#include <cstdint>
#include <cstdlib>

#define FATTN_KQ_STRIDE       256
#define HALF_MAX_HALF         __float2half(65504.0f/2) // Use neg. of this instead of -INFINITY to initialize KQ max vals to avoid NaN upon subtraction.
#define SOFTMAX_FTZ_THRESHOLD -20.0f                   // Softmax exp. of values smaller than this are flushed to zero to avoid NaNs.

// log(2) = 0.6931, by adding this to the KQ maximum used for the softmax the numerical range representable
//     by the VKQ accumulators is effectively being shifted up by a factor of 2.
// This reduces issues with numerical overflow but also causes larger values to be flushed to zero.
// However, as the output from FlashAttention will usually be used as an input for a matrix multiplication this should be negligible.
// Still, the value range should be shifted as much as necessary but as little as possible.
// The macro on the following line shifts it by a factor of 2**3=8, as was needed to fix https://github.com/ggml-org/llama.cpp/issues/18606 .
#define FATTN_KQ_MAX_OFFSET (3.0f*0.6931f)

// Flags passed to the flash-attention kernels, set via environment variables:
// FATTN_FLAGS_PV_Q8_0: opt-in (GGML_CUDA_FA_PV_Q8=1) alternative P*V path for q8_0 V in the vec kernel.
//     Reads the raw q8_0 data and folds the block scale into the softmax probability
//     (VKQ += qs * (d*P) instead of VKQ += (d*qs) * P), skipping the per-element dequantization
//     multiply. A full q8 x q8 DP4A P*V is not profitable in the vec kernel: the P*V reduction runs
//     along the token axis while q8_0 V packs 4 consecutive bytes along head_dim, so DP4A would need
//     a per-4-token byte transpose (3 PRMT/token), and the q8_0 block scales vary along the token
//     axis, forcing the V scales to be folded into P and re-quantized for every 128-token tile
//     (~11 instr/token/column) with only ncols <= 2 columns to amortize over. Instruction accounting
//     on sm_86 comes out a wash at best for ncols=1 and clearly negative for ncols=2.
#define FATTN_FLAGS_PV_Q8_0 1

typedef void (* fattn_kernel_t)(
        const char * __restrict__ Q,
        const char * __restrict__ K,
        const char * __restrict__ V,
        const char * __restrict__ mask,
        const char * __restrict__ sinks,
        const int  * __restrict__ KV_max,
        const int32_t * __restrict__ sparse_idx,
        const int32_t sparse_n_raw,
        const int32_t sparse_nb1,
        const int64_t sparse_nb3,
        const char * __restrict__ sparse_mask,
        const int32_t sparse_mask_nb1,
        const int64_t sparse_mask_nb3,
        float      * __restrict__ dst,
        float2     * __restrict__ dst_meta,
        const float scale,
        const float max_bias,
        const float m0,
        const float m1,
        const uint32_t n_head_log2,
        const float logit_softcap,
        const int32_t ne00, const uint3   ne01, const int32_t ne02, const int32_t ne03,
                            const int32_t nb01, const int32_t nb02, const int32_t nb03,
        const int32_t ne10, const int32_t ne11, const int32_t ne12, const int32_t ne13,
                            const int32_t nb11, const int32_t nb12, const int64_t nb13,
                            const int32_t nb21, const int32_t nb22, const int64_t nb23,
                            const int32_t ne31, const int32_t ne32, const int32_t ne33,
                            const int32_t nb31, const int32_t nb32, const int64_t nb33,
        const int32_t fattn_flags);

typedef float (*vec_dot_KQ_t)(
    const char * __restrict__ K_c, const void * __restrict__ Q_v, const int * __restrict__ Q_q8 , const void * __restrict__ Q_ds);

struct ggml_cuda_flash_attn_ext_f16_extra_data {
    uintptr_t K;
    uintptr_t V;
    uintptr_t end;
};

static inline ggml_cuda_flash_attn_ext_f16_extra_data ggml_cuda_flash_attn_ext_get_f16_extra_data(
        const ggml_tensor * dst, const bool need_f16_K, const bool need_f16_V) {
    GGML_ASSERT(dst->op == GGML_OP_FLASH_ATTN_EXT);

    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];

    GGML_ASSERT(K != nullptr);
    GGML_ASSERT(V != nullptr);

    const bool V_is_K_view = V->view_src && (V->view_src == K || (V->view_src == K->view_src && V->view_offs == K->view_offs));

    // Sparse kernels address the full physical K/V backing through the row map,
    // so a quantized sparse K/V must be converted across the whole backing
    // tensor, not just the compact logical view (see ggml_cuda_get_best_fattn_kernel).
    const bool sparse = dst->src[5] && !dst->src[6];
    const ggml_tensor * K_cvt = sparse && K->view_src ? K->view_src : K;
    const ggml_tensor * V_cvt = sparse && V->view_src ? V->view_src : V;

    ggml_cuda_flash_attn_ext_f16_extra_data data = {};
    data.end = (uintptr_t) dst->data + ggml_nbytes(dst);

    if (need_f16_K && K->type != GGML_TYPE_F16) {
        data.end = GGML_PAD(data.end, 128);
        data.K   = data.end;
        data.end += ggml_nelements(K_cvt)*ggml_type_size(GGML_TYPE_F16);
    }

    if (need_f16_V && V->type != GGML_TYPE_F16) {
        if (V_is_K_view) {
            data.V = data.K;
        } else {
            data.end = GGML_PAD(data.end, 128);
            data.V   = data.end;
            data.end += ggml_nelements(V_cvt)*ggml_type_size(GGML_TYPE_F16);
        }
    }

    return data;
}

template <int D, int nthreads>
static __device__ __forceinline__ float vec_dot_fattn_vec_KQ_f16(
    const char * __restrict__ K_c, const void * __restrict__ Q_v, const int * __restrict__ Q_q8 , const void * __restrict__ Q_ds_v) {

    const half2 * K_h2 = (const half2 *) K_c;
    GGML_UNUSED(Q_q8);
    GGML_UNUSED(Q_ds_v);

    constexpr int cpy_nb = ggml_cuda_get_max_cpy_bytes();
    constexpr int cpy_ne = cpy_nb / 4;

    float sum = 0.0f;

#pragma unroll
    for (int k_KQ_0 = 0; k_KQ_0 < D/2; k_KQ_0 += nthreads*cpy_ne) {
        __align__(16) half2 tmp[cpy_ne];
        ggml_cuda_memcpy_1<sizeof(tmp)>(tmp, K_h2 + k_KQ_0 + (threadIdx.x % nthreads)*cpy_ne);
#pragma unroll
        for (int k_KQ_1 = 0; k_KQ_1 < cpy_ne; ++k_KQ_1) {
#ifdef V_DOT2_F32_F16_AVAILABLE
            ggml_cuda_mad(sum,                tmp[k_KQ_1] , ((const half2  *) Q_v)[k_KQ_0/nthreads + k_KQ_1]);
#else
            ggml_cuda_mad(sum, __half22float2(tmp[k_KQ_1]), ((const float2 *) Q_v)[k_KQ_0/nthreads + k_KQ_1]);
#endif // V_DOT2_F32_F16_AVAILABLE
        }
    }

    return sum;
}

template <int D, int nthreads>
static __device__ __forceinline__ float vec_dot_fattn_vec_KQ_bf16(
    const char * __restrict__ K_c, const void * __restrict__ Q_v, const int * __restrict__ Q_q8 , const void * __restrict__ Q_ds_v) {

    const nv_bfloat162 * K_bf16 = (const nv_bfloat162 *) K_c;
    GGML_UNUSED(Q_q8);
    GGML_UNUSED(Q_ds_v);

    constexpr int cpy_nb = ggml_cuda_get_max_cpy_bytes();
    constexpr int cpy_ne = cpy_nb / 4;

    float sum = 0.0f;

#pragma unroll
    for (int k_KQ_0 = 0; k_KQ_0 < D/2; k_KQ_0 += nthreads*cpy_ne) {
        __align__(16) nv_bfloat162 tmp[cpy_ne];
        ggml_cuda_memcpy_1<sizeof(tmp)>(tmp, K_bf16 + k_KQ_0 + (threadIdx.x % nthreads)*cpy_ne);
#pragma unroll
        for (int k_KQ_1 = 0; k_KQ_1 < cpy_ne; ++k_KQ_1) {
#ifdef V_DOT2_F32_F16_AVAILABLE
            // FIXME replace macros in vector FA kernel with templating and use FP32 for BF16
            ggml_cuda_mad(sum, ggml_cuda_cast<float2>(tmp[k_KQ_1]), __half22float2(((const half2 *) Q_v)[k_KQ_0/nthreads + k_KQ_1]));
#else
            ggml_cuda_mad(sum, ggml_cuda_cast<float2>(tmp[k_KQ_1]), ((const float2 *) Q_v)[k_KQ_0/nthreads + k_KQ_1]);
#endif // V_DOT2_F32_F16_AVAILABLE
        }
    }

    return sum;
}

template<int D, int nthreads>
static __device__ __forceinline__ float vec_dot_fattn_vec_KQ_q4_0(
    const char * __restrict__ K_c, const void * __restrict__ Q_v, const int * __restrict__ Q_q8, const void * __restrict__ Q_ds_v) {

    const block_q4_0 * K_q4_0 = (const block_q4_0 *) K_c;
    GGML_UNUSED(Q_v);

    float sum = 0.0f;

#pragma unroll
    for (int k_KQ_0 = 0; k_KQ_0 < int(D/sizeof(int)); k_KQ_0 += nthreads) {
        const int k_KQ = k_KQ_0 + (nthreads == WARP_SIZE ? threadIdx.x : threadIdx.x % nthreads);

        const int ib    = k_KQ /  QI8_1;
        const int iqs4  = k_KQ %  QI4_0;
        const int shift = k_KQ & (QI8_1/2);

        int v;
        ggml_cuda_memcpy_1<sizeof(int), 2>(&v, K_q4_0[ib].qs + sizeof(int)*iqs4);
        v = (v >> shift) & 0x0F0F0F0F;
        const int u = Q_q8[k_KQ_0/nthreads];

        const int sumi = ggml_cuda_dp4a(v, u, 0);

        const float2 Q_ds = ((const float2 *) Q_ds_v)[k_KQ_0/nthreads];
        sum += __half2float(K_q4_0[ib].d) * (sumi*Q_ds.x - (8/QI8_1)*Q_ds.y);
    }

    return sum;
}

template<int D, int nthreads>
static __device__ __forceinline__ float vec_dot_fattn_vec_KQ_q4_1(
    const char * __restrict__ K_c, const void * __restrict__ Q_v, const int * __restrict__ Q_q8, const void * __restrict__ Q_ds_v) {

    const block_q4_1 * K_q4_1 = (const block_q4_1 *) K_c;
    GGML_UNUSED(Q_v);

    float sum = 0.0f;

#pragma unroll
    for (int k_KQ_0 = 0; k_KQ_0 < int(D/sizeof(int)); k_KQ_0 += nthreads) {
        const int k_KQ = k_KQ_0 + (nthreads == WARP_SIZE ? threadIdx.x : threadIdx.x % nthreads);

        const int ib    = k_KQ /  QI8_1;
        const int iqs4  = k_KQ %  QI4_1;
        const int shift = k_KQ & (QI8_1/2);

        int v;
        ggml_cuda_memcpy_1<sizeof(int)>(&v, K_q4_1[ib].qs + sizeof(int)*iqs4);
        v = (v >> shift) & 0x0F0F0F0F;
        const int u = Q_q8[k_KQ_0/nthreads];

        const int sumi = ggml_cuda_dp4a(v, u, 0);

        const float2 K_dm = __half22float2(K_q4_1[ib].dm);
        const float2 Q_ds = ((const float2 *) Q_ds_v)[k_KQ_0/nthreads];

        sum += K_dm.x*Q_ds.x*sumi + K_dm.y*Q_ds.y/QI8_1;
    }

    return sum;
}

template<int D, int nthreads>
static __device__ __forceinline__ float vec_dot_fattn_vec_KQ_q5_0(
    const char * __restrict__ K_c, const void * __restrict__ Q_v, const int * __restrict__ Q_q8, const void * __restrict__ Q_ds_v) {

    const block_q5_0 * K_q5_0 = (const block_q5_0 *) K_c;
    GGML_UNUSED(Q_v);

    float sum = 0.0f;

#pragma unroll
    for (int k_KQ_0 = 0; k_KQ_0 < int(D/sizeof(int)); k_KQ_0 += nthreads) {
        const int k_KQ = k_KQ_0 + (nthreads == WARP_SIZE ? threadIdx.x : threadIdx.x % nthreads);

        const int ib    = k_KQ /  QI8_1;
        const int iqs4  = k_KQ %  QI5_0;
        const int iqs8  = k_KQ %  QI8_1;
        const int shift = k_KQ & (QI8_1/2);

        int v;
        ggml_cuda_memcpy_1<sizeof(int), 2>(&v, K_q5_0[ib].qs + sizeof(int)*iqs4);
        v = (v >> shift) & 0x0F0F0F0F;

        {
            int vh;
            ggml_cuda_memcpy_1<sizeof(int), 2>(&vh, K_q5_0[ib].qh);
            vh >>= iqs8 * QI5_0;

            v |= (vh <<  4) & 0x00000010; // 0 ->  4
            v |= (vh << 11) & 0x00001000; // 1 -> 12
            v |= (vh << 18) & 0x00100000; // 2 -> 20
            v |= (vh << 25) & 0x10000000; // 3 -> 28
        }

        const int u = Q_q8[k_KQ_0/nthreads];

        const int sumi = ggml_cuda_dp4a(v, u, 0);

        const float2 Q_ds = ((const float2 *) Q_ds_v)[k_KQ_0/nthreads];

        sum += __half2float(K_q5_0[ib].d) * (sumi*Q_ds.x - (16/QI8_1)*Q_ds.y);
    }

    return sum;
}

template<int D, int nthreads>
static __device__ __forceinline__ float vec_dot_fattn_vec_KQ_q5_1(
    const char * __restrict__ K_c, const void * __restrict__ Q_v, const int * __restrict__ Q_q8, const void * __restrict__ Q_ds_v) {

    const block_q5_1 * K_q5_1 = (const block_q5_1 *) K_c;
    GGML_UNUSED(Q_v);

    float sum = 0.0f;

#pragma unroll
    for (int k_KQ_0 = 0; k_KQ_0 < int(D/sizeof(int)); k_KQ_0 += nthreads) {
        const int k_KQ = k_KQ_0 + (nthreads == WARP_SIZE ? threadIdx.x : threadIdx.x % nthreads);

        const int ib    = k_KQ /  QI8_1;
        const int iqs4  = k_KQ %  QI5_1;
        const int iqs8  = k_KQ %  QI8_1;
        const int shift = k_KQ & (QI8_1/2);

        int v;
        ggml_cuda_memcpy_1<sizeof(int)>(&v, K_q5_1[ib].qs + sizeof(int)*iqs4);
        v = (v >> shift) & 0x0F0F0F0F;

        {
            int vh;
            ggml_cuda_memcpy_1<sizeof(int)>(&vh, K_q5_1[ib].qh);
            vh >>= iqs8 * QI5_0;

            v |= (vh <<  4) & 0x00000010; // 0 ->  4
            v |= (vh << 11) & 0x00001000; // 1 -> 12
            v |= (vh << 18) & 0x00100000; // 2 -> 20
            v |= (vh << 25) & 0x10000000; // 3 -> 28
        }

        const int u = Q_q8[k_KQ_0/nthreads];

        const int sumi = ggml_cuda_dp4a(v, u, 0);

        const float2 K_dm = __half22float2(K_q5_1[ib].dm);
        const float2 Q_ds = ((const float2 *) Q_ds_v)[k_KQ_0/nthreads];

        sum += K_dm.x*Q_ds.x*sumi + K_dm.y*Q_ds.y/QI8_1;
    }

    return sum;
}

template <int D, int nthreads>
static __device__ __forceinline__ float vec_dot_fattn_vec_KQ_q8_0(
    const char * __restrict__ K_c, const void * __restrict__ Q_v, const int * __restrict__ Q_q8, const void * __restrict__ Q_ds_v) {

    const block_q8_0 * K_q8_0 = (const block_q8_0 *) K_c;
    GGML_UNUSED(Q_v);

    float sum = 0.0f;

#pragma unroll
    for (int k_KQ_0 = 0; k_KQ_0 < int(D/sizeof(int)); k_KQ_0 += nthreads) {
        const int k_KQ = k_KQ_0 + (nthreads == WARP_SIZE ? threadIdx.x : threadIdx.x % nthreads);

        const int ib  = k_KQ / QI8_0;
        const int iqs = k_KQ % QI8_0;

        int v;
        ggml_cuda_memcpy_1<sizeof(v), 2>(&v, K_q8_0[ib].qs + 4*iqs);

        const float2 * Q_ds = (const float2 *) Q_ds_v;
        const float Q_d = Q_ds[k_KQ_0/nthreads].x;

        sum += vec_dot_q8_0_q8_1_impl<float, 1>(&v, &Q_q8[k_KQ_0/nthreads], K_q8_0[ib].d, Q_d);
    }

    return sum;
}

template <typename Tds, int ni>
static __device__ __forceinline__ void quantize_q8_1_to_shared(
    const float * __restrict__ x, const float scale, int * __restrict__ yq32, void * __restrict__ yds) {

    float vals[sizeof(int)] = {0.0f};
#pragma unroll
    for (int l = 0; l < int(sizeof(int)); ++l) {
        vals[l] = (ni == WARP_SIZE || threadIdx.x < ni) ? scale * x[4*threadIdx.x + l] : 0.0f;
    }

    float amax = fabsf(vals[0]);
    float sum  = vals[0];
#pragma unroll
    for (int l = 1; l < int(sizeof(int)); ++l) {
        amax = fmaxf(amax, fabsf(vals[l]));
        sum += vals[l];
    }
#pragma unroll
    for (int mask = QI8_1/2; mask > 0; mask >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFF, amax, mask, 32));
        sum +=             __shfl_xor_sync(0xFFFFFFFF, sum,  mask, 32);
    }

    const float d = amax / 127;
    int q32 = 0;
    int8_t * q8 = (int8_t *) &q32;

    if (d != 0.0f) {
#pragma unroll
        for (int l = 0; l < int(sizeof(int)); ++l) {
            q8[l] = roundf(vals[l] / d);
        }
    }

    yq32[threadIdx.x] = q32;
    if (threadIdx.x % QI8_1 == 0 && (ni == WARP_SIZE || threadIdx.x < ni)) {
        if (std::is_same<Tds, half2>::value) {
            ((half2  *) yds)[threadIdx.x/QI8_1] =  make_half2(d, sum);
        } else {
            ((float2 *) yds)[threadIdx.x/QI8_1] = make_float2(d, sum);
        }
    }
}

typedef void (*dequantize_V_t)(const void *, void *, const int64_t);

template <typename T, int ne>
static __device__ __forceinline__ void dequantize_V_f16(const void * __restrict__ vx, void * __restrict__ dst, const int64_t i0) {
    if constexpr (std::is_same_v<T, half>) {
        ggml_cuda_memcpy_1<ne*sizeof(half)>(dst, (const half *) vx + i0);
    } else if constexpr (std::is_same_v<T, float>) {
        static_assert(ne % 2 == 0, "bad ne");
        __align__(16) half2 tmp[ne/2];
        ggml_cuda_memcpy_1<ne*sizeof(half)>(tmp, (const half *) vx + i0);
        float2 * dst_f2 = (float2 *) dst;
#pragma unroll
        for (int l = 0; l < ne/2; ++l) {
            dst_f2[l] = __half22float2(tmp[l]);
        }
    } else {
        static_assert(std::is_same_v<T, void>, "unsupported type");
    }
}

template <typename T, int ne>
static __device__ __forceinline__ void dequantize_V_bf16(const void * __restrict__ vx, void * __restrict__ dst, const int64_t i0) {
    static_assert(std::is_same_v<T, float>, "BF16 V dequantization only supports float output");
    static_assert(ne % 2 == 0, "bad ne");
    __align__(16) nv_bfloat162 tmp[ne/2];
    ggml_cuda_memcpy_1<ne*sizeof(nv_bfloat16)>(tmp, (const nv_bfloat16 *) vx + i0);
    float2 * dst_f2 = (float2 *) dst;
#pragma unroll
    for (int l = 0; l < ne/2; ++l) {
        dst_f2[l] = ggml_cuda_cast<float2>(tmp[l]);
    }
}

template <typename T, int ne>
static __device__ __forceinline__ void dequantize_V_q4_0(const void * __restrict__ vx, void * __restrict__ dst, const int64_t i0) {
    const block_q4_0 * x = (const block_q4_0 *) vx;

    const int64_t ib    =  i0          /  QK4_0;
    const int     iqs   =  i0          % (QK4_0/2);
    const int     shift = (i0 % QK4_0) / (QK4_0/2);

    int q;
    static_assert(ne == 2 || ne == 4, "bad ne");
    ggml_cuda_memcpy_1<ne, 2>(&q, x[ib].qs + iqs);
    q >>= 4*shift;
    q &= 0x0F0F0F0F;
    q = __vsubss4(q, 0x08080808);

    const int8_t * q8 = (const int8_t *) &q;

#ifdef FP16_AVAILABLE
    if constexpr (std::is_same_v<T, half>) {
        const half2 d = __half2half2(x[ib].d);

#pragma unroll
        for (int l0 = 0; l0 < ne; l0 += 2) {
            ((half2 *) dst)[l0/2] = d * make_half2(q8[l0 + 0], q8[l0 + 1]);
        }
    } else
#endif // FP16_AVAILABLE
    if constexpr (std::is_same_v<T, float>) {
        const float d = x[ib].d;

#pragma unroll
        for (int l = 0; l < ne; ++l) {
            ((float *) dst)[l] = d * q8[l];
        }
    } else {
        static_assert(std::is_same_v<T, void>, "bad type");
    }
}

template <typename T, int ne>
static __device__ __forceinline__ void dequantize_V_q4_1(const void * __restrict__ vx, void * __restrict__ dst, const int64_t i0) {
    const block_q4_1 * x = (const block_q4_1 *) vx;

    const int64_t ib    =  i0          /  QK4_1;
    const int     iqs   =  i0          % (QK4_1/2);
    const int     shift = (i0 % QK4_1) / (QK4_1/2);

    int q;
    static_assert(ne == 2 || ne == 4, "bad ne");
    ggml_cuda_memcpy_1<ne>(&q, x[ib].qs + iqs);
    q >>= 4*shift;
    q &= 0x0F0F0F0F;

    const int8_t * q8 = (const int8_t *) &q;

#ifdef FP16_AVAILABLE
    if constexpr (std::is_same_v<T, half>) {
        const half2 dm = x[ib].dm;
        const half2 d  = __half2half2( __low2half(dm));
        const half2 m  = __half2half2(__high2half(dm));

#pragma unroll
        for (int l0 = 0; l0 < ne; l0 += 2) {
            ((half2 *) dst)[l0/2] = d * make_half2(q8[l0 + 0], q8[l0 + 1]) + m;
        }
    } else
#endif // FP16_AVAILABLE
    if constexpr (std::is_same_v<T, float>) {
        const float2 dm = __half22float2(x[ib].dm);

#pragma unroll
        for (int l = 0; l < ne; ++l) {
            ((float *) dst)[l] = dm.x * q8[l] + dm.y;
        }
    } else {
        static_assert(std::is_same_v<T, void>, "bad type");
    }
}

template <typename T, int ne>
static __device__ __forceinline__ void dequantize_V_q5_0(const void * __restrict__ vx, void * __restrict__ dst, const int64_t i0) {
    const block_q5_0 * x = (const block_q5_0 *) vx;

    const int64_t ib    =  i0          /  QK5_0;
    const int     idq   =  i0          %  QK5_0;
    const int     iqs   =  i0          % (QK5_0/2);
    const int     shift = (i0 % QK5_0) / (QK5_0/2);

    int q;
    static_assert(ne == 2 || ne == 4, "bad ne");
    ggml_cuda_memcpy_1<ne, 2>(&q, x[ib].qs + iqs);
    q >>= 4*shift;
    q &= 0x0F0F0F0F;

    {
        int qh;
        ggml_cuda_memcpy_1<ne, 2>(&qh, x[ib].qh);
#pragma unroll
        for (int l = 0; l < ne; ++l) {
            q |= ((qh >> (idq + l)) & 0x00000001) << (8*l + 4);
        }
    }

    q = __vsubss4(q, 0x10101010);

    const int8_t * q8 = (const int8_t *) &q;

#ifdef FP16_AVAILABLE
    if constexpr (std::is_same_v<T, half>) {
        const half2 d = __half2half2(x[ib].d);

#pragma unroll
        for (int l0 = 0; l0 < ne; l0 += 2) {
            ((half2 *) dst)[l0/2] = d * make_half2(q8[l0 + 0], q8[l0 + 1]);
        }
    } else
#endif // FP16_AVAILABLE
    if constexpr (std::is_same_v<T, float>) {
        const float d = x[ib].d;

#pragma unroll
        for (int l = 0; l < ne; ++l) {
            ((float *) dst)[l] = d * q8[l];
        }
    } else {
        static_assert(std::is_same_v<T, void>, "bad type");
    }
}

template <typename T, int ne>
static __device__ __forceinline__ void dequantize_V_q5_1(const void * __restrict__ vx, void * __restrict__ dst, const int64_t i0) {
    const block_q5_1 * x = (const block_q5_1 *) vx;

    const int64_t ib    =  i0          /  QK5_1;
    const int     idq   =  i0          %  QK5_1;
    const int     iqs   =  i0          % (QK5_1/2);
    const int     shift = (i0 % QK5_1) / (QK5_1/2);

    int q;
    static_assert(ne == 2 || ne == 4, "bad ne");
    ggml_cuda_memcpy_1<ne>(&q, x[ib].qs + iqs);
    q >>= 4*shift;
    q &= 0x0F0F0F0F;

    {
        int qh;
        ggml_cuda_memcpy_1<ne>(&qh, x[ib].qh);
#pragma unroll
        for (int l = 0; l < ne; ++l) {
            q |= ((qh >> (idq + l)) & 0x00000001) << (8*l + 4);
        }
    }

    const int8_t * q8 = (const int8_t *) &q;

#ifdef FP16_AVAILABLE
    if constexpr (std::is_same_v<T, half>) {
        const half2 dm = x[ib].dm;
        const half2 d  = __half2half2( __low2half(dm));
        const half2 m  = __half2half2(__high2half(dm));

#pragma unroll
        for (int l0 = 0; l0 < ne; l0 += 2) {
            ((half2 *) dst)[l0/2] = d * make_half2(q8[l0 + 0], q8[l0 + 1]) + m;
        }
    } else
#endif // FP16_AVAILABLE
    if constexpr (std::is_same_v<T, float>) {
        const float2 dm = __half22float2(x[ib].dm);

#pragma unroll
        for (int l = 0; l < ne; ++l) {
            ((float *) dst)[l] = dm.x * q8[l] + dm.y;
        }
    } else {
        static_assert(std::is_same_v<T, void>, "bad type");
    }
}

template <typename T, int ne>
static __device__ __forceinline__ void dequantize_V_q8_0(const void * __restrict__ vx, void * __restrict__ dst, const int64_t i0) {
    const block_q8_0 * x = (const block_q8_0 *) vx;

    const int64_t ib  = i0 / QK8_0;
    const int     iqs = i0 % QK8_0;

    static_assert(ne % 2 == 0, "bad ne");
    int8_t qs[ne];
    ggml_cuda_memcpy_1<ne, 2>(qs, x[ib].qs + iqs);

#ifdef FP16_AVAILABLE
    if constexpr (std::is_same<T, half>::value) {
        const half2 d = __half2half2(x[ib].d);

#pragma unroll
        for (int l0 = 0; l0 < ne; l0 += 2) {
            ((half2 *) dst)[l0/2] = d * make_half2(qs[l0 + 0], qs[l0 + 1]);
        }
    } else
#endif // FP16_AVAILABLE
    if constexpr (std::is_same<T, float>::value) {
        const float d = x[ib].d;

#pragma unroll
        for (int l = 0; l < ne; ++l) {
            ((float *) dst)[l] = d * qs[l];
        }
    } else {
        static_assert(std::is_same_v<T, void>, "unsupported type");
    }
}

template <ggml_type type_K, int D, int nthreads>
constexpr __device__ vec_dot_KQ_t get_vec_dot_KQ() {
    if constexpr (type_K == GGML_TYPE_F16) {
        return vec_dot_fattn_vec_KQ_f16<D, nthreads>;
    } else if constexpr (type_K == GGML_TYPE_Q4_0) {
        return vec_dot_fattn_vec_KQ_q4_0<D, nthreads>;
    } else if constexpr (type_K == GGML_TYPE_Q4_1) {
        return vec_dot_fattn_vec_KQ_q4_1<D, nthreads>;
    } else if constexpr (type_K == GGML_TYPE_Q5_0) {
        return vec_dot_fattn_vec_KQ_q5_0<D, nthreads>;
    } else if constexpr (type_K == GGML_TYPE_Q5_1) {
        return vec_dot_fattn_vec_KQ_q5_1<D, nthreads>;
    } else if constexpr (type_K == GGML_TYPE_Q8_0) {
        return vec_dot_fattn_vec_KQ_q8_0<D, nthreads>;
    } else if constexpr (type_K == GGML_TYPE_BF16) {
        return vec_dot_fattn_vec_KQ_bf16<D, nthreads>;
    } else {
        static_assert(type_K == -1, "bad type");
        return nullptr;
    }
}

template <ggml_type type_V, typename T, int ne>
constexpr __device__ dequantize_V_t get_dequantize_V() {
    if constexpr (type_V == GGML_TYPE_F16) {
        return dequantize_V_f16<T, ne>;
    } else if constexpr (type_V == GGML_TYPE_Q4_0) {
        return dequantize_V_q4_0<T, ne>;
    } else if constexpr (type_V == GGML_TYPE_Q4_1) {
        return dequantize_V_q4_1<T, ne>;
    } else if constexpr (type_V == GGML_TYPE_Q5_0) {
        return dequantize_V_q5_0<T, ne>;
    } else if constexpr (type_V == GGML_TYPE_Q5_1) {
        return dequantize_V_q5_1<T, ne>;
    } else if constexpr (type_V == GGML_TYPE_Q8_0) {
        return dequantize_V_q8_0<T, ne>;
    } else if constexpr (type_V == GGML_TYPE_BF16) {
        return dequantize_V_bf16<float, ne>;
    } else {
        static_assert(type_V == -1, "bad type");
        return nullptr;
    }
}

template <int ncols1>
__launch_bounds__(FATTN_KQ_STRIDE/2, 1)
static __global__ void flash_attn_mask_to_KV_max(
        const half2 * mask_ptr, int * KV_max_ptr, const int ne30, const int64_t s31, const int64_t s33) {
    const half2 * GGML_CUDA_RESTRICT mask   = mask_ptr;
    int         * GGML_CUDA_RESTRICT KV_max = KV_max_ptr;

    const int ne31     = gridDim.x;
    const int tid      = threadIdx.x;
    const int sequence = blockIdx.y;
    const int jt       = blockIdx.x;

    mask += sequence*s33 + jt*ncols1*s31;

    __shared__ int buf_iw[WARP_SIZE];
    if (tid < WARP_SIZE) {
        buf_iw[tid] = 1;
    }
    ggml_cuda_pdl_sync();
    __syncthreads();

    int KV_max_sj = (ne30 - 1) * FATTN_KQ_STRIDE;
    for (; KV_max_sj >= 0; KV_max_sj -= FATTN_KQ_STRIDE) {
        int all_inf = 1;

#pragma unroll
        for (int j = 0; j < ncols1; ++j) {
            const float2 tmp = __half22float2(mask[j*s31 + KV_max_sj/2 + tid]);
            all_inf = all_inf && int(isinf(tmp.x)) && int(isinf(tmp.y));
        }

        all_inf = warp_reduce_all(all_inf);
        if (tid % WARP_SIZE == 0) {
            buf_iw[tid / WARP_SIZE] = all_inf;
        }
        __syncthreads();
        all_inf = buf_iw[tid % WARP_SIZE];
        __syncthreads();
        all_inf = warp_reduce_all(all_inf);

        if (!all_inf) {
            break;
        }
    }

    // If the break in the loop was not triggered, KV_max_sj is now -FATTN_KQ_STRIDE.
    // If the break was triggered it's the lower edge of the tile with the first non-masked values.
    // In either case, walk back the decrementation by FATTN_KQ_STRIDE.
    KV_max_sj += FATTN_KQ_STRIDE;

    // Also record the first tile with any finite mask value. The max bounds
    // occupy the first gridDim.x*gridDim.y entries and the min bounds the
    // second half. Kernels that do not support lower-bound skipping simply
    // continue reading the first half as before.
    int KV_min_sj = 0;
    for (; KV_min_sj < ne30 * FATTN_KQ_STRIDE; KV_min_sj += FATTN_KQ_STRIDE) {
        int all_inf = 1;

#pragma unroll
        for (int j = 0; j < ncols1; ++j) {
            const float2 tmp = __half22float2(mask[j*s31 + KV_min_sj/2 + tid]);
            all_inf = all_inf && int(isinf(tmp.x)) && int(isinf(tmp.y));
        }

        all_inf = warp_reduce_all(all_inf);
        if (tid % WARP_SIZE == 0) {
            buf_iw[tid / WARP_SIZE] = all_inf;
        }
        __syncthreads();
        all_inf = buf_iw[tid % WARP_SIZE];
        __syncthreads();
        all_inf = warp_reduce_all(all_inf);

        if (!all_inf) {
            break;
        }
    }

    if (threadIdx.x != 0) {
        return;
    }

    const int bounds_index = sequence*ne31 + jt;
    const int bounds_count = gridDim.x*gridDim.y;
    KV_max[bounds_index] = KV_max_sj;
    KV_max[bounds_count + bounds_index] = KV_min_sj;
}

template<int D, int ncols1, int ncols2> // D == head size
__launch_bounds__(D, 1)
static __global__ void flash_attn_stream_k_fixup_uniform(
        float * dst_ptr,
        const float2 * dst_fixup_ptr,
        const int ne01, const int ne02,
        const int ne12, const int nblocks_stream_k,
        const int gqa_ratio,
        const int blocks_per_tile,
        const uint3 fd_iter_j_z_ne12,
        const uint3 fd_iter_j_z,
        const uint3 fd_iter_j) {
    constexpr int ncols = ncols1*ncols2;
    ggml_cuda_pdl_lc();
    float        * GGML_CUDA_RESTRICT dst       = dst_ptr;
    const float2 * GGML_CUDA_RESTRICT dst_fixup = dst_fixup_ptr;

    const int tile_idx = blockIdx.x; // One block per output tile.
    const int j        = blockIdx.y;
    const int c        = blockIdx.z;
    const int jc       = j*ncols2 + c;
    const int tid      = threadIdx.x;

    // nblocks_stream_k is a multiple of ntiles_dst (== gridDim.x), so each tile gets the same number of blocks.
    const int b_first = tile_idx * blocks_per_tile;
    const int b_last  = b_first + blocks_per_tile - 1;

    const float * dst_fixup_data = ((const float *) dst_fixup) + nblocks_stream_k*(2*2*ncols);

    // z_KV == K/V head index, zt_gqa = Q head start index per K/V head, jt = token position start index
    const uint2 dm0 = fast_div_modulo(tile_idx, fd_iter_j_z_ne12);
    const uint2 dm1 = fast_div_modulo(dm0.y,    fd_iter_j_z);
    const uint2 dm2 = fast_div_modulo(dm1.y,    fd_iter_j);

    const int sequence = dm0.x;
    const int z_KV     = dm1.x;
    const int zt_gqa   = dm2.x;
    const int jt       = dm2.y;

    const int zt_Q = z_KV*gqa_ratio + zt_gqa*ncols2; // Global Q head start index.

    if (jt*ncols1 + j >= ne01 || zt_gqa*ncols2 + c >= gqa_ratio) {
        return;
    }

    dst += sequence*ne02*ne01*D + jt*ne02*(ncols1*D) + zt_Q*D + (j*ne02 + c)*D + tid;

    ggml_cuda_pdl_sync();
    // Load the partial result that needs a fixup
    float dst_val = *dst;
    float max_val;
    float rowsum;
    {
        const float2 tmp = dst_fixup[b_last*ncols + jc];
        max_val = tmp.x;
        rowsum  = tmp.y;
    }

    // Combine with all previous blocks in this tile.
    for (int bidx = b_last - 1; bidx >= b_first; --bidx) {
        const float dst_add = dst_fixup_data[bidx*ncols*D + jc*D + tid];

        const float2 tmp = dst_fixup[(nblocks_stream_k + bidx)*ncols + jc];

        const float max_val_new = fmaxf(max_val, tmp.x);

        const float diff_val = max_val - max_val_new;
        const float diff_add = tmp.x   - max_val_new;

        const float scale_val = diff_val >= SOFTMAX_FTZ_THRESHOLD ? expf(diff_val) : 0.0f;
        const float scale_add = diff_add >= SOFTMAX_FTZ_THRESHOLD ? expf(diff_add) : 0.0f;

        dst_val = scale_val*dst_val + scale_add*dst_add;
        rowsum  = scale_val*rowsum  + scale_add*tmp.y;

        max_val = max_val_new;
    }

    // Write back final result:
    *dst = dst_val / rowsum;
}

// General fixup kernel for the case where the number of blocks per tile is not uniform across tiles
// (blocks_num.x not a multiple of ntiles_dst)
template <int D, int ncols1, int ncols2> // D == head size
__launch_bounds__(D, 1)
static __global__ void flash_attn_stream_k_fixup_general(
        float * dst_ptr,
        const float2 * dst_fixup_ptr,
        const int ne01, const int ne02,
        const int gqa_ratio,
        const int total_work,
        const uint3 fd_iter_k_j_z_ne12,
        const uint3 fd_iter_k_j_z,
        const uint3 fd_iter_k_j,
        const uint3 fd_iter_k) {
    float        * GGML_CUDA_RESTRICT dst       = dst_ptr;
    const float2 * GGML_CUDA_RESTRICT dst_fixup = dst_fixup_ptr;
    constexpr int ncols = ncols1*ncols2;

    const int bidx0 = blockIdx.x;
    const int j     = blockIdx.y;
    const int c     = blockIdx.z;
    const int jc    = j*ncols2 + c;
    const int tid   = threadIdx.x;

    const float * dst_fixup_data = ((const float *) dst_fixup) + gridDim.x*(2*2*ncols);

    const int kbc0      = int64_t(bidx0 + 0)*total_work / gridDim.x;
    const int kbc0_stop = int64_t(bidx0 + 1)*total_work / gridDim.x;

    const bool did_not_have_any_data   = kbc0 == kbc0_stop;
    const bool wrote_beginning_of_tile = fastmodulo(kbc0, fd_iter_k) == 0;
    const bool did_not_write_last      = fastdiv(kbc0, fd_iter_k) == fastdiv(kbc0_stop, fd_iter_k) && fastmodulo(kbc0_stop, fd_iter_k) != 0;
    if (did_not_have_any_data || wrote_beginning_of_tile || did_not_write_last) {
        return;
    }

    // z_KV == K/V head index, zt_gqa = Q head start index per K/V head, jt = token position start index
    const uint2 dm0 = fast_div_modulo(kbc0, fd_iter_k_j_z_ne12);
    const uint2 dm1 = fast_div_modulo(dm0.y, fd_iter_k_j_z);
    const uint2 dm2 = fast_div_modulo(dm1.y, fd_iter_k_j);
    const uint2 dm3 = fast_div_modulo(dm2.y, fd_iter_k);

    const int sequence = dm0.x;
    const int z_KV     = dm1.x;
    const int zt_gqa   = dm2.x;
    const int jt       = dm3.x;

    const int zt_Q = z_KV*gqa_ratio + zt_gqa*ncols2; // Global Q head start index.

    if (jt*ncols1 + j >= ne01 || zt_gqa*ncols2 + c >= gqa_ratio) {
        return;
    }

    dst += sequence*ne02*ne01*D + jt*ne02*(ncols1*D) + zt_Q*D + (j*ne02 + c)*D + tid;

    // Load the partial result that needs a fixup:
    float dst_val = 0.0f;
    float max_val = 0.0f;
    float rowsum  = 0.0f;
    ggml_cuda_pdl_sync();
    {
        dst_val = *dst;

        const float2 tmp = dst_fixup[bidx0*ncols + jc];
        max_val = tmp.x;
        rowsum  = tmp.y;
    }

    // Iterate over previous blocks and compute the combined results.
    // All CUDA blocks that get here must have a previous block that needs a fixup.
    const int tile_kbc0 = fastdiv(kbc0, fd_iter_k);
    int bidx = bidx0 - 1;
    int kbc_stop = kbc0;
    while(true) {
        const int kbc = int64_t(bidx)*total_work / gridDim.x;
        if (kbc == kbc_stop) { // Did not have any data.
            bidx--;
            kbc_stop = kbc;
            continue;
        }

        const float dst_add = dst_fixup_data[bidx*ncols*D + jc*D + tid];

        const float2 tmp = dst_fixup[(gridDim.x + bidx)*ncols + jc];

        // Scale the current and new value accumulators depending on the max. values.
        const float max_val_new = fmaxf(max_val, tmp.x);

        const float diff_val = max_val - max_val_new;
        const float diff_add = tmp.x   - max_val_new;

        const float scale_val = diff_val >= SOFTMAX_FTZ_THRESHOLD ? expf(diff_val) : 0.0f;
        const float scale_add = diff_add >= SOFTMAX_FTZ_THRESHOLD ? expf(diff_add) : 0.0f;

        dst_val = scale_val*dst_val + scale_add*dst_add;
        rowsum  = scale_val*rowsum  + scale_add*tmp.y;

        max_val = max_val_new;

        // If this block started in a previous tile we are done and don't need to combine additional partial results.
        if (fastmodulo(kbc, fd_iter_k) == 0 || fastdiv(kbc, fd_iter_k) < tile_kbc0) {
            break;
        }
        bidx--;
        kbc_stop = kbc;
    }

    // Write back final result:
    *dst = dst_val / rowsum;
}

template<int D> // D == head size
__launch_bounds__(D, 1)
static __global__ void flash_attn_combine_results(
        const float  * VKQ_parts_ptr,
        const float2 * VKQ_meta_ptr,
        float * dst_ptr,
        const int parallel_blocks) {
    ggml_cuda_pdl_lc();
    const float  * GGML_CUDA_RESTRICT VKQ_parts = VKQ_parts_ptr;
    const float2 * GGML_CUDA_RESTRICT VKQ_meta  = VKQ_meta_ptr;
    float        * GGML_CUDA_RESTRICT dst       = dst_ptr;
    // Dimension 0: threadIdx.x
    // Dimension 1: blockIdx.x
    // Dimension 2: blockIdx.y
    // Dimension 3: blockIdx.z
    // Memory layout is permuted with [0, 2, 1, 3]

    const int ne01 = gridDim.x;
    const int ne02 = gridDim.y;

    const int col      = blockIdx.x;
    const int head     = blockIdx.y;
    const int sequence = blockIdx.z;

    const int j_dst_unrolled = (sequence*ne01 + col)*ne02 + head;

    VKQ_parts += j_dst_unrolled * parallel_blocks*D;
    VKQ_meta  += j_dst_unrolled * parallel_blocks;
    dst       += j_dst_unrolled *                 D;

    const int tid = threadIdx.x;
    __builtin_assume(tid < D);

    extern __shared__ float2 meta[];
    ggml_cuda_pdl_sync();
    for (int i = tid; i < 2*parallel_blocks; i += D) {
        ((float *) meta)[i] = ((const float *)VKQ_meta) [i];
    }

    __syncthreads();

    float kqmax = meta[0].x;
    for (int l = 1; l < parallel_blocks; ++l) {
        kqmax = max(kqmax, meta[l].x);
    }

    float VKQ_numerator   = 0.0f;
    float VKQ_denominator = 0.0f;
    for (int l = 0; l < parallel_blocks; ++l) {
        const float KQ_max_scale = expf(meta[l].x - kqmax);

        VKQ_numerator   += KQ_max_scale * VKQ_parts[l*D + tid];
        VKQ_denominator += KQ_max_scale * meta[l].y;
    }

    dst[tid] = VKQ_numerator / VKQ_denominator;
}

struct alignas(16) fattn_sparse_2kv_meta {
    uint64_t k2;
    uint64_t v2;
    int64_t  nb_k2_2;
    int64_t  nb_k2_3;
    int64_t  nb_v2_2;
    int64_t  nb_v2_3;
    int32_t  stride_k2;
    int32_t  stride_v2;
};

static_assert(sizeof(fattn_sparse_2kv_meta) % sizeof(int32_t) == 0, "bad sparse 2KV metadata size");
static constexpr int FATTN_SPARSE_2KV_META_I32 = sizeof(fattn_sparse_2kv_meta)/sizeof(int32_t);

template <int ncols1>
static __global__ void flash_attn_sparse_group_prepare(
        const int32_t * __restrict__ top_k,
        const half    * __restrict__ raw_mask,
        const half    * __restrict__ compressed_mask,
        int32_t       * __restrict__ group_rows,
        half          * __restrict__ group_mask,
        int           * __restrict__ kv_bounds,
        const char    * __restrict__ k2,
        const char    * __restrict__ v2,
        const int64_t nb_k2_2,
        const int64_t nb_k2_3,
        const int64_t nb_v2_2,
        const int64_t nb_v2_3,
        const int stride_k2,
        const int stride_v2,
        const int n_query,
        const int n_compressed,
        const int k,
        const int max_union,
        const int raw_capacity,
        const int group_capacity,
        const int top_k_nb1,
        const int64_t top_k_nb3,
        const int raw_mask_nb1,
        const int64_t raw_mask_nb3,
        const int compressed_mask_nb1,
        const int64_t compressed_mask_nb3,
        const int n_raw,
        const int bounds_count,
        const bool initialize_min_bound,
        const bool compact_raw) {
    static_assert(ncols1 <= 8, "sparse query group mask uses one byte");

    if (threadIdx.x != 0) {
        return;
    }

    if (blockIdx.x == 0 && blockIdx.y == 0) {
        fattn_sparse_2kv_meta * meta = reinterpret_cast<fattn_sparse_2kv_meta *>(
            group_rows - FATTN_SPARSE_2KV_META_I32);
        meta->k2 = reinterpret_cast<uint64_t>(k2);
        meta->v2 = reinterpret_cast<uint64_t>(v2);
        meta->nb_k2_2 = nb_k2_2;
        meta->nb_k2_3 = nb_k2_3;
        meta->nb_v2_2 = nb_v2_2;
        meta->nb_v2_3 = nb_v2_3;
        meta->stride_k2 = stride_k2;
        meta->stride_v2 = stride_v2;
    }

    const int group = blockIdx.x;
    const int sequence = blockIdx.y;
    const int query0 = group*ncols1;
    const int n_group_query = min(ncols1, n_query - query0);
    const int group_stride = gridDim.x*(group_capacity + 2);
    const int mask_group_stride = ncols1*group_capacity;

    const int32_t * top_k_sequence = top_k + int64_t(sequence)*top_k_nb3;
    const half * raw_mask_sequence = raw_mask + int64_t(sequence)*raw_mask_nb3;
    const half * compressed_mask_sequence = compressed_mask + int64_t(sequence)*compressed_mask_nb3;
    int32_t * rows_out = group_rows + int64_t(sequence)*group_stride + int64_t(group)*(group_capacity + 2);
    half * mask_out = group_mask +
        (int64_t(sequence)*gridDim.x + group)*mask_group_stride;

    const int bounds_index = sequence*gridDim.x + group;
    const int raw_begin = min(n_raw, max(0, kv_bounds[bounds_count + bounds_index]));
    const int raw_end   = min(n_raw, max(0, kv_bounds[bounds_index]));
    const int raw_span  = max(0, raw_end - raw_begin);
    const bool use_compact_raw = compact_raw && raw_span <= raw_capacity;
    rows_out[0] = use_compact_raw ? 1 : 0;
    rows_out += 2;

    int n_group_rows = 0;
    if (use_compact_raw) {
        for (int i = 0; i < raw_span; ++i) {
            const int physical_row = raw_begin + i;
            rows_out[n_group_rows] = physical_row;
#pragma unroll
            for (int j = 0; j < ncols1; ++j) {
                mask_out[int64_t(j)*group_capacity + n_group_rows] = j < n_group_query ?
                    raw_mask_sequence[int64_t(query0 + j)*raw_mask_nb1 + physical_row] :
                    __float2half(-INFINITY);
            }
            ++n_group_rows;
        }
    }

    int cursor[ncols1] = {};
    int n_union = 0;

    // The DSV4 producer marks top-k indices as sorted. Merge the adjacent
    // query rows so each physical K/V row is loaded once by the ncols1=8 FA
    // tile, while a compact mask preserves each query's exact selection.
    while (n_union < max_union) {
        int physical_row = INT_MAX;
#pragma unroll
        for (int j = 0; j < ncols1; ++j) {
            if (j < n_group_query && cursor[j] < k) {
                physical_row = min(physical_row,
                    top_k_sequence[int64_t(query0 + j)*top_k_nb1 + cursor[j]]);
            }
        }
        if (physical_row == INT_MAX) {
            break;
        }

        const int compressed_row = physical_row >= 0 && physical_row < n_compressed ? physical_row : 0;
        rows_out[n_group_rows + n_union] = use_compact_raw ? n_raw + compressed_row : compressed_row;
#pragma unroll
        for (int j = 0; j < ncols1; ++j) {
            bool selected = false;
            if (j < n_group_query && cursor[j] < k) {
                const int32_t current = top_k_sequence[int64_t(query0 + j)*top_k_nb1 + cursor[j]];
                selected = current == physical_row && physical_row >= 0 && physical_row < n_compressed;
                if (current == physical_row) {
                    ++cursor[j];
                }
            }
            mask_out[int64_t(j)*group_capacity + n_group_rows + n_union] = selected ?
                compressed_mask_sequence[int64_t(query0 + j)*compressed_mask_nb1 + physical_row] :
                __float2half(-INFINITY);
        }
        ++n_union;
    }

    // The last MMA tile can extend past the group union. Valid row zero plus
    // an all-masked value keeps those padded lanes harmless.
    for (int i = n_group_rows + n_union; i < group_capacity; ++i) {
        rows_out[i] = 0;
#pragma unroll
        for (int j = 0; j < ncols1; ++j) {
            mask_out[int64_t(j)*group_capacity + i] = __float2half(-INFINITY);
        }
    }

    const int n_logical_rows = use_compact_raw ? n_group_rows + n_union : n_raw + n_union;
    rows_out[-1] = n_logical_rows;
    kv_bounds[bounds_index] = n_logical_rows;
    if (use_compact_raw || initialize_min_bound) {
        kv_bounds[bounds_count + bounds_index] = 0;
    }
}

static __global__ void flash_attn_sparse_query_prepare(
        const int32_t * __restrict__ top_k,
        const half    * __restrict__ raw_mask,
        const half    * __restrict__ compressed_mask,
        int32_t       * __restrict__ group_rows,
        half          * __restrict__ group_mask,
        int           * __restrict__ kv_bounds,
        const char    * __restrict__ k2,
        const char    * __restrict__ v2,
        const int64_t nb_k2_2,
        const int64_t nb_k2_3,
        const int64_t nb_v2_2,
        const int64_t nb_v2_3,
        const int stride_k2,
        const int stride_v2,
        const int n_compressed,
        const int k,
        const int max_union,
        const int raw_capacity,
        const int group_capacity,
        const int top_k_nb1,
        const int64_t top_k_nb3,
        const int raw_mask_nb1,
        const int64_t raw_mask_nb3,
        const int compressed_mask_nb1,
        const int64_t compressed_mask_nb3,
        const int n_raw,
        const int bounds_count,
        const bool compact_raw) {
    const int query = blockIdx.x;
    const int sequence = blockIdx.y;
    const int group_stride = gridDim.x*(group_capacity + 2);

    if (query == 0 && sequence == 0 && threadIdx.x == 0) {
        fattn_sparse_2kv_meta * meta = reinterpret_cast<fattn_sparse_2kv_meta *>(
            group_rows - FATTN_SPARSE_2KV_META_I32);
        meta->k2 = reinterpret_cast<uint64_t>(k2);
        meta->v2 = reinterpret_cast<uint64_t>(v2);
        meta->nb_k2_2 = nb_k2_2;
        meta->nb_k2_3 = nb_k2_3;
        meta->nb_v2_2 = nb_v2_2;
        meta->nb_v2_3 = nb_v2_3;
        meta->stride_k2 = stride_k2;
        meta->stride_v2 = stride_v2;
    }

    const int32_t * top_k_query = top_k + int64_t(sequence)*top_k_nb3 + int64_t(query)*top_k_nb1;
    const half * raw_mask_query = raw_mask + int64_t(sequence)*raw_mask_nb3 + int64_t(query)*raw_mask_nb1;
    const half * compressed_mask_query =
        compressed_mask + int64_t(sequence)*compressed_mask_nb3 + int64_t(query)*compressed_mask_nb1;
    int32_t * rows_header = group_rows + int64_t(sequence)*group_stride + int64_t(query)*(group_capacity + 2);
    int32_t * rows_out = rows_header + 2;
    half * mask_out = group_mask + (int64_t(sequence)*gridDim.x + query)*group_capacity;

    __shared__ int raw_bounds[2];
    if (compact_raw) {
        if (threadIdx.x == 0) {
            raw_bounds[0] = n_raw;
            raw_bounds[1] = -1;
        }
        __syncthreads();

        int local_begin = n_raw;
        int local_end = -1;
        for (int i = threadIdx.x; i < n_raw; i += blockDim.x) {
            if (!isinf(__half2float(raw_mask_query[i]))) {
                local_begin = min(local_begin, i);
                local_end = max(local_end, i);
            }
        }
        if (local_begin < n_raw) {
            atomicMin(raw_bounds + 0, local_begin);
            atomicMax(raw_bounds + 1, local_end);
        }
        __syncthreads();
    }

    const int bounds_index = sequence*gridDim.x + query;
    const int raw_begin = compact_raw && raw_bounds[1] >= 0 ?
        (raw_bounds[0]/FATTN_KQ_STRIDE)*FATTN_KQ_STRIDE : n_raw;
    const int raw_end = compact_raw && raw_bounds[1] >= 0 ?
        min(n_raw, ((raw_bounds[1] + 1 + FATTN_KQ_STRIDE - 1)/FATTN_KQ_STRIDE)*FATTN_KQ_STRIDE) : 0;
    const int raw_span = max(0, raw_end - raw_begin);
    const bool use_compact_raw = compact_raw && raw_span <= raw_capacity;
    const int raw_rows = use_compact_raw ? raw_span : 0;

    for (int i = threadIdx.x; i < raw_rows; i += blockDim.x) {
        const int physical_row = raw_begin + i;
        rows_out[i] = physical_row;
        mask_out[i] = raw_mask_query[physical_row];
    }

    for (int i = threadIdx.x; i < max_union; i += blockDim.x) {
        const bool selected = i < k;
        const int physical_row = selected ? top_k_query[i] : -1;
        const bool valid = physical_row >= 0 && physical_row < n_compressed;
        const int compressed_row = valid ? physical_row : 0;
        rows_out[raw_rows + i] = use_compact_raw ? n_raw + compressed_row : compressed_row;
        mask_out[raw_rows + i] = selected && valid ?
            compressed_mask_query[physical_row] : __float2half(-INFINITY);
    }

    for (int i = threadIdx.x + raw_rows + max_union; i < group_capacity; i += blockDim.x) {
        rows_out[i] = 0;
        mask_out[i] = __float2half(-INFINITY);
    }

    if (threadIdx.x == 0) {
        const int n_logical_rows = use_compact_raw ? raw_rows + max_union : n_raw + max_union;
        rows_header[0] = use_compact_raw ? 1 : 0;
        rows_header[1] = n_logical_rows;
        kv_bounds[bounds_index] = n_logical_rows;
        kv_bounds[bounds_count + bounds_index] = 0;
    }
}

template <int DV, int ncols1, int ncols2>
void launch_fattn(
    ggml_backend_cuda_context & ctx, ggml_tensor * dst, fattn_kernel_t fattn_kernel, const int nwarps, const size_t nbytes_shared,
    const int nbatch_fa, const bool need_f16_K, const bool need_f16_V, const bool stream_k, const int warp_size = WARP_SIZE
) {
    constexpr int ncols = ncols1 * ncols2;

    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];

    const bool V_is_K_view = V->view_src && (V->view_src == K || (V->view_src == K->view_src && V->view_offs == K->view_offs));

    const ggml_tensor * mask  = dst->src[3];
    const ggml_tensor * sinks = dst->src[4];
    const ggml_tensor * sparse_idx = dst->src[5] && !dst->src[6] ? dst->src[5] : nullptr;
    const ggml_tensor * sparse_mask = sparse_idx ? dst->src[7] : nullptr;
    const ggml_tensor * K2 = sparse_idx ? dst->src[8] : nullptr;
    const ggml_tensor * V2 = sparse_idx ? dst->src[9] : nullptr;
    GGML_ASSERT((K2 == nullptr) == (V2 == nullptr));

    ggml_tensor * KQV = dst;

    GGML_ASSERT(Q->type == GGML_TYPE_F32);
    GGML_ASSERT(KQV->type == GGML_TYPE_F32);

    GGML_ASSERT(Q->nb[0] == ggml_element_size(Q));
    GGML_ASSERT(K->nb[0] == ggml_element_size(K));
    GGML_ASSERT(V->nb[0] == ggml_element_size(V));

    GGML_ASSERT(!mask || mask->type == GGML_TYPE_F16);

    ggml_cuda_pool & pool = ctx.pool();
    cudaStream_t main_stream = ctx.stream();
    const int id  = ggml_cuda_get_device();
    const int cc  = ggml_cuda_info().devices[id].cc;
    const int nsm = ggml_cuda_info().devices[id].nsm;

    const ggml_cuda_flash_attn_ext_f16_extra_data f16_extra =
        ggml_cuda_flash_attn_ext_get_f16_extra_data(KQV, need_f16_K, need_f16_V);

    // The VMM pool frees in strict reverse allocation order. Sparse staging is
    // allocated before any optional stream-k fixup buffers below.
    ggml_cuda_pool_alloc<int>     KV_max(pool);
    ggml_cuda_pool_alloc<int32_t> sparse_group_rows(pool);
    ggml_cuda_pool_alloc<half>    sparse_group_mask(pool);
    ggml_cuda_pool_alloc<float>   dst_tmp(pool);
    ggml_cuda_pool_alloc<float2>  dst_tmp_meta(pool);

    const char * K_data = (const char *) K->data;
    size_t nb11 = K->nb[1];
    size_t nb12 = K->nb[2];
    size_t nb13 = K->nb[3];

    const char * V_data = (const char *) V->data;
    size_t nb21 = V->nb[1];
    size_t nb22 = V->nb[2];
    size_t nb23 = V->nb[3];

    const char * K2_data = K2 ? (const char *) K2->data : nullptr;
    const char * V2_data = V2 ? (const char *) V2->data : nullptr;
    const size_t nbK2_1 = K2 ? K2->nb[1] : 0;
    const size_t nbK2_2 = K2 ? K2->nb[2] : 0;
    const size_t nbK2_3 = K2 ? K2->nb[3] : 0;
    const size_t nbV2_1 = V2 ? V2->nb[1] : 0;
    const size_t nbV2_2 = V2 ? V2->nb[2] : 0;
    const size_t nbV2_3 = V2 ? V2->nb[3] : 0;

    // Sparse kernels address the full physical backing through the row map, so
    // convert the backing tensor rather than the compact logical view.
    const ggml_tensor * K_cvt = (sparse_idx && K->view_src) ? K->view_src : K;
    const ggml_tensor * V_cvt = (sparse_idx && V->view_src) ? V->view_src : V;

    if (need_f16_K && K->type != GGML_TYPE_F16) {
        const size_t bs = ggml_blck_size(K_cvt->type);
        const size_t ts = ggml_type_size(K_cvt->type);

        GGML_ASSERT(f16_extra.K != 0);
        half * K_f16 = (half *) f16_extra.K;
        if (ggml_is_contiguously_allocated(K_cvt)) {
            to_fp16_cuda_t to_fp16 = ggml_get_to_fp16_cuda(K_cvt->type);
            to_fp16((const char *) K_cvt->data, K_f16, ggml_nelements(K_cvt), main_stream);

            nb11 = K_cvt->nb[1]*bs*sizeof(half)/ts;
            nb12 = K_cvt->nb[2]*bs*sizeof(half)/ts;
            nb13 = K_cvt->nb[3]*bs*sizeof(half)/ts;
        } else {
            GGML_ASSERT(K_cvt->nb[0] == ts);
            to_fp16_nc_cuda_t to_fp16 = ggml_get_to_fp16_nc_cuda(K_cvt->type);
            const int64_t s01 = K_cvt->nb[1] / ts;
            const int64_t s02 = K_cvt->nb[2] / ts;
            const int64_t s03 = K_cvt->nb[3] / ts;
            to_fp16((const char *) K_cvt->data, K_f16, K_cvt->ne[0], K_cvt->ne[1], K_cvt->ne[2], K_cvt->ne[3], s01, s02, s03, main_stream);

            nb11 = K_cvt->ne[0] * sizeof(half);
            nb12 = K_cvt->ne[1] * nb11;
            nb13 = K_cvt->ne[2] * nb12;
        }
        K_data = (char *) K_f16;
    }

    if (need_f16_V && V->type != GGML_TYPE_F16) {
        if (V_is_K_view) {
            V_data = K_data;
            nb21   = nb11;
            nb22   = nb12;
            nb23   = nb13;
        } else {
            const size_t bs = ggml_blck_size(V_cvt->type);
            const size_t ts = ggml_type_size(V_cvt->type);

            GGML_ASSERT(f16_extra.V != 0);
            half * V_f16 = (half *) f16_extra.V;
            if (ggml_is_contiguously_allocated(V_cvt)) {
                to_fp16_cuda_t to_fp16 = ggml_get_to_fp16_cuda(V_cvt->type);
                to_fp16((const char *) V_cvt->data, V_f16, ggml_nelements(V_cvt), main_stream);
                V_data = (char *) V_f16;

                nb21 = V_cvt->nb[1]*bs*sizeof(half)/ts;
                nb22 = V_cvt->nb[2]*bs*sizeof(half)/ts;
                nb23 = V_cvt->nb[3]*bs*sizeof(half)/ts;
            } else {
                GGML_ASSERT(V_cvt->nb[0] == ts);
                to_fp16_nc_cuda_t to_fp16 = ggml_get_to_fp16_nc_cuda(V_cvt->type);
                const int64_t s01 = V_cvt->nb[1] / ts;
                const int64_t s02 = V_cvt->nb[2] / ts;
                const int64_t s03 = V_cvt->nb[3] / ts;
                to_fp16((const char *) V_cvt->data, V_f16, V_cvt->ne[0], V_cvt->ne[1], V_cvt->ne[2], V_cvt->ne[3], s01, s02, s03, main_stream);

                nb21 = V_cvt->ne[0] * sizeof(half);
                nb22 = V_cvt->ne[1] * nb21;
                nb23 = V_cvt->ne[2] * nb22;
            }
            V_data = (char *) V_f16;
        }
    }

    // Sparse grouping addresses the full K/V backing through a compact logical
    // view. Non-F16 K/V is converted across the whole backing above (non-2kv
    // only); the 2kv variant has no scratch for the second segment and keeps
    // the F16 requirement.
    GGML_ASSERT(!sparse_idx || (K->type == GGML_TYPE_F16 && V->type == GGML_TYPE_F16) ||
        (!K2 && K->type == V->type && K->view_src && V->view_src));
    GGML_ASSERT(!K2 || (K2->type == GGML_TYPE_F16 && V2->type == GGML_TYPE_F16));

    const int ntiles_x     = ((Q->ne[1] + ncols1 - 1) / ncols1);
    const int gqa_ratio    = Q->ne[2] / K->ne[2];
    const int ntiles_z_gqa = ((gqa_ratio + ncols2 - 1) / ncols2);
    const int ntiles_dst   = ntiles_x * ntiles_z_gqa * K->ne[2] * Q->ne[3];

    // Optional optimization where the mask is scanned to determine whether part of the calculation can be skipped.
    // Only worth the overhead if there is at lease one FATTN_KQ_STRIDE x FATTN_KQ_STRIDE square to be skipped or
    //     multiple sequences of possibly different lengths.
    // q1 single-sequence decode over a long masked KV range (e.g. a raw SWA cache exposed at full width, where
    //     fully masked tiles dominate and are skipped entirely) qualifies as well.
    const int64_t n_kv_masked = sparse_idx ? mask->ne[0] : K->ne[1];
    const bool scan_mask_bounds = mask && n_kv_masked % FATTN_KQ_STRIDE == 0 &&
        (sparse_idx || Q->ne[1] >= 1024 || Q->ne[3] > 1 ||
         (Q->ne[1] == 1 && Q->ne[3] == 1 && n_kv_masked >= 2048));
    const bool query_prepare_bounds = sparse_idx && ncols1 == 1;
    const int ne_KV_max = ntiles_x*Q->ne[3];
    if (scan_mask_bounds || sparse_idx) {
        KV_max.alloc(2*ne_KV_max);
    }
    if (scan_mask_bounds && !query_prepare_bounds) {
        const int64_t s31 = mask->nb[1] / sizeof(half2);
        const int64_t s33 = mask->nb[3] / sizeof(half2);

        const dim3 blocks_num_KV_max(ntiles_x, Q->ne[3], 1);
        const dim3 block_dim_KV_max(FATTN_KQ_STRIDE/2, 1, 1);

        const int iter_k = n_kv_masked / FATTN_KQ_STRIDE;

        ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(blocks_num_KV_max, block_dim_KV_max, 0, main_stream);
        ggml_cuda_kernel_launch(flash_attn_mask_to_KV_max<ncols1>, launch_params,
            (const half2 *) mask->data, KV_max.ptr, iter_k, s31, s33);
        CUDA_CHECK(cudaGetLastError());
    }

    int64_t ne11_effective = K->ne[1];
    const int32_t * sparse_idx_data = nullptr;
    const half * sparse_mask_data = nullptr;
    int32_t sparse_idx_nb1 = 0;
    int64_t sparse_idx_nb3 = 0;
    int32_t sparse_mask_nb1 = 0;
    int64_t sparse_mask_nb3 = 0;

    if (sparse_idx) {
        if constexpr (ncols1 != 1 && ncols1 != 8) {
            GGML_ABORT("sparse flash attention requires 1- or 8-query groups");
        } else {
        GGML_ASSERT(sparse_mask);
        GGML_ASSERT(sparse_idx->type == GGML_TYPE_I32 && sparse_mask->type == GGML_TYPE_F16);
        GGML_ASSERT(sparse_idx->ne[1] == Q->ne[1] && sparse_idx->ne[3] == Q->ne[3]);
        GGML_ASSERT(sparse_mask->ne[1] == Q->ne[1] && sparse_mask->ne[3] == Q->ne[3]);

        const int n_raw = mask->ne[0];
        const int max_union = std::min<int64_t>(sparse_mask->ne[0], sparse_idx->ne[0]*ncols1);
        GGML_ASSERT(max_union >= sparse_idx->ne[0]);

        static const int sparse_raw_compact = []() {
            const char * value = getenv("GGML_CUDA_DSV4_SPARSE_RAW_COMPACT");
            return value ? atoi(value) : 0;
        }();
        // Value 2 forces compact staging for small unit-test shapes and the
        // opt-in decode experiment. Value 1 remains prefill-only here.
        const bool sparse_raw_compact_active = sparse_raw_compact > 0 &&
            (Q->ne[1] >= FATTN_KQ_STRIDE || sparse_raw_compact > 1);
        constexpr int raw_capacity = 2*FATTN_KQ_STRIDE;
        const int group_capacity = max_union + (sparse_raw_compact_active ? raw_capacity : 0);

        const size_t n_group_rows = FATTN_SPARSE_2KV_META_I32 +
            (size_t) Q->ne[3]*ntiles_x*(group_capacity + 2);
        const size_t n_group_mask = (size_t) Q->ne[3]*ntiles_x*group_capacity*ncols1;
        sparse_group_rows.alloc(n_group_rows);
        sparse_group_mask.alloc(n_group_mask);
        int32_t * const sparse_group_rows_data = sparse_group_rows.ptr + FATTN_SPARSE_2KV_META_I32;

        const dim3 blocks_num_sparse(ntiles_x, Q->ne[3], 1);
        const dim3 block_dim_sparse(ncols1 == 1 ? 256 : 1, 1, 1);
        ggml_cuda_kernel_launch_params launch_params =
            ggml_cuda_kernel_launch_params(blocks_num_sparse, block_dim_sparse, 0, main_stream);
        if constexpr (ncols1 == 1) {
            ggml_cuda_kernel_launch(flash_attn_sparse_query_prepare, launch_params,
                (const int32_t *) sparse_idx->data,
                (const half *) mask->data,
                (const half *) sparse_mask->data,
                sparse_group_rows_data,
                sparse_group_mask.ptr,
                KV_max.ptr,
                K2_data,
                V2_data,
                (int64_t) nbK2_2,
                (int64_t) nbK2_3,
                (int64_t) nbV2_2,
                (int64_t) nbV2_3,
                K2 ? (int) (nbK2_1/sizeof(half2)) : 0,
                V2 ? (int) (nbV2_1/sizeof(half2)) : 0,
                (int) sparse_mask->ne[0],
                (int) sparse_idx->ne[0],
                max_union,
                sparse_raw_compact_active ? raw_capacity : 0,
                group_capacity,
                (int) (sparse_idx->nb[1]/sizeof(int32_t)),
                (int64_t) (sparse_idx->nb[3]/sizeof(int32_t)),
                (int) (mask->nb[1]/sizeof(half)),
                (int64_t) (mask->nb[3]/sizeof(half)),
                (int) (sparse_mask->nb[1]/sizeof(half)),
                (int64_t) (sparse_mask->nb[3]/sizeof(half)),
                n_raw,
                ne_KV_max,
                sparse_raw_compact_active && scan_mask_bounds);
        } else {
            ggml_cuda_kernel_launch(flash_attn_sparse_group_prepare<ncols1>, launch_params,
                (const int32_t *) sparse_idx->data,
                (const half *) mask->data,
                (const half *) sparse_mask->data,
                sparse_group_rows_data,
                sparse_group_mask.ptr,
                KV_max.ptr,
                K2_data,
                V2_data,
                (int64_t) nbK2_2,
                (int64_t) nbK2_3,
                (int64_t) nbV2_2,
                (int64_t) nbV2_3,
                K2 ? (int) (nbK2_1/sizeof(half2)) : 0,
                V2 ? (int) (nbV2_1/sizeof(half2)) : 0,
                (int) Q->ne[1],
                (int) sparse_mask->ne[0],
                (int) sparse_idx->ne[0],
                max_union,
                sparse_raw_compact_active ? raw_capacity : 0,
                group_capacity,
                (int) (sparse_idx->nb[1]/sizeof(int32_t)),
                (int64_t) (sparse_idx->nb[3]/sizeof(int32_t)),
                (int) (mask->nb[1]/sizeof(half)),
                (int64_t) (mask->nb[3]/sizeof(half)),
                (int) (sparse_mask->nb[1]/sizeof(half)),
                (int64_t) (sparse_mask->nb[3]/sizeof(half)),
                n_raw,
                ne_KV_max,
                !scan_mask_bounds,
                sparse_raw_compact_active && scan_mask_bounds);
        }
        CUDA_CHECK(cudaGetLastError());

        ne11_effective = n_raw + max_union;
        sparse_idx_data = sparse_group_rows_data;
        sparse_mask_data = sparse_group_mask.ptr;
        sparse_idx_nb1 = group_capacity + 2;
        sparse_idx_nb3 = int64_t(ntiles_x)*(group_capacity + 2);
        sparse_mask_nb1 = group_capacity;
        sparse_mask_nb3 = int64_t(ntiles_x)*ncols1*group_capacity;
        }
    }

    const dim3 block_dim(warp_size, nwarps, 1);
    int max_blocks_per_sm = 1; // Max. number of active blocks limited by occupancy.
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_blocks_per_sm, fattn_kernel, block_dim.x * block_dim.y * block_dim.z, nbytes_shared));
    GGML_ASSERT(max_blocks_per_sm > 0);
    int parallel_blocks = max_blocks_per_sm;

    const int ntiles_KV = (ne11_effective + nbatch_fa - 1) / nbatch_fa; // Max. number of parallel blocks limited by KV cache length.

    dim3 blocks_num;
    if (stream_k) {
        // For short contexts it can be faster to have the SMs work on whole tiles because this lets us skip the fixup.
        const int max_blocks = max_blocks_per_sm*nsm;
        const int tiles_nwaves = (ntiles_dst + max_blocks - 1) / max_blocks;
        const int tiles_efficiency_percent = 100 * ntiles_dst / (max_blocks*tiles_nwaves);

        const bool use_stream_k = cc >= GGML_CUDA_CC_ADA_LOVELACE || amd_wmma_available(cc) || tiles_efficiency_percent < 75;

        blocks_num.x = ntiles_dst;
        blocks_num.y = 1;
        blocks_num.z = 1;

        if(use_stream_k) {
            const int nblocks_stream_k_raw = std::min(max_blocks, ntiles_KV*ntiles_dst);
            // Round down to a multiple of ntiles_dst so that each output tile gets the same number of blocks (avoids fixup).
            // Only do this if the occupancy loss from rounding is acceptable.
            const int nblocks_stream_k_rounded = (nblocks_stream_k_raw / ntiles_dst) * ntiles_dst;
            const int max_efficiency_loss_percent = 5;
            const int efficiency_loss_percent = nblocks_stream_k_rounded > 0
                ? 100 * (nblocks_stream_k_raw - nblocks_stream_k_rounded) / nblocks_stream_k_raw
                : 100;
            const int nblocks_stream_k = efficiency_loss_percent <= max_efficiency_loss_percent
                ? nblocks_stream_k_rounded
                : nblocks_stream_k_raw;

            blocks_num.x = nblocks_stream_k;
        }

        if (ntiles_dst % blocks_num.x != 0) { // Fixup is only needed if the SMs work on fractional tiles.
            dst_tmp_meta.alloc((size_t(blocks_num.x) * ncols * (2 + DV/2)));
        }
    } else {
        // parallel_blocks must not be larger than what the tensor size allows:
        parallel_blocks = std::min(parallel_blocks, ntiles_KV);

        // If ntiles_total % blocks_per_wave != 0 then some efficiency is lost due to tail effects.
        // Test whether parallel_blocks can be set to a higher value for better efficiency.
        const int blocks_per_wave = nsm * max_blocks_per_sm;
        int nwaves_best = 0;
        int efficiency_percent_best = 0;
        for (int parallel_blocks_test = parallel_blocks; parallel_blocks_test <= ntiles_KV; ++parallel_blocks_test) {
            const int nblocks_total = ntiles_dst * parallel_blocks_test;
            const int nwaves = (nblocks_total + blocks_per_wave - 1) / blocks_per_wave;
            const int efficiency_percent = 100 * nblocks_total / (nwaves*blocks_per_wave);

            // Stop trying configurations with more waves if we already have good efficiency to avoid excessive overhead.
            if (efficiency_percent_best >= 95 && nwaves > nwaves_best) {
                break;
            }

            if (efficiency_percent > efficiency_percent_best) {
                nwaves_best = nwaves;
                efficiency_percent_best = efficiency_percent;
                parallel_blocks = parallel_blocks_test;
            }
        }

        blocks_num.x = ntiles_x;
        blocks_num.y = parallel_blocks;
        blocks_num.z = ntiles_z_gqa*K->ne[2]*Q->ne[3];

        if (parallel_blocks > 1) {
            dst_tmp.alloc(parallel_blocks*ggml_nelements(KQV));
            dst_tmp_meta.alloc(parallel_blocks*ggml_nrows(KQV));
        }
    }

    float scale         = 1.0f;
    float max_bias      = 0.0f;
    float logit_softcap = 0.0f;

    memcpy(&scale,         (const float *) KQV->op_params + 0, sizeof(float));
    memcpy(&max_bias,      (const float *) KQV->op_params + 1, sizeof(float));
    memcpy(&logit_softcap, (const float *) KQV->op_params + 2, sizeof(float));

    if (logit_softcap != 0.0f) {
        scale /= logit_softcap;
    }

    // Opt-in kernel features via environment, read once:
    static const int32_t fattn_flags = []() {
        const char * env = getenv("GGML_CUDA_FA_PV_Q8");
        return env != nullptr && atoi(env) != 0 ? FATTN_FLAGS_PV_Q8_0 : 0;
    }();

    const uint32_t n_head      = Q->ne[2];
    const uint32_t n_head_log2 = 1u << uint32_t(floorf(log2f(float(n_head))));

    const float m0 = powf(2.0f, -(max_bias       ) / n_head_log2);
    const float m1 = powf(2.0f, -(max_bias / 2.0f) / n_head_log2);

    // TODO other tensor dimensions after removal of WMMA kernel:
    const uint3 ne01 = init_fastdiv_values(Q->ne[1]);

    GGML_ASSERT(block_dim.x % warp_size == 0);

        ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(blocks_num, block_dim, nbytes_shared, main_stream);
        ggml_cuda_kernel_launch(fattn_kernel, launch_params,
        (const char *) Q->data,
        K_data,
        V_data,
        mask ? ((const char *) mask->data) : nullptr,
        sinks ? ((const char *) sinks->data) : nullptr,
        KV_max.ptr,
        sparse_idx_data,
        sparse_idx ? (int32_t) mask->ne[0] : 0,
        sparse_idx_nb1,
        sparse_idx_nb3,
        sparse_mask_data ? (const char *) sparse_mask_data : nullptr,
        sparse_mask_nb1,
        sparse_mask_nb3,
        !stream_k && parallel_blocks > 1 ? dst_tmp.ptr : (float *) KQV->data, dst_tmp_meta.ptr,
        scale, max_bias, m0, m1, n_head_log2, logit_softcap,
        Q->ne[0], ne01,     Q->ne[2], Q->ne[3], Q->nb[1], Q->nb[2], Q->nb[3],
        K->ne[0], ne11_effective, K->ne[2], K->ne[3], nb11, nb12, nb13,
        nb21, nb22, nb23,
        mask ? mask->ne[1] : 0, mask ? mask->ne[2] : 0, mask ? mask->ne[3] : 0,
        mask ? mask->nb[1] : 0, mask ? mask->nb[2] : 0, mask ? mask->nb[3] : 0,
        fattn_flags
    );
    CUDA_CHECK(cudaGetLastError());

    if (stream_k) {
        if ((int)blocks_num.x % ntiles_dst == 0 && (int)blocks_num.x > ntiles_dst) {
            // Optimized fixup: nblocks_stream_k is a multiple of ntiles_dst, launch one block per tile.
            const int nblocks_sk  = (int)blocks_num.x;
            const int bpt         = nblocks_sk / ntiles_dst;

            const uint3 fd0 = init_fastdiv_values(ntiles_x * ntiles_z_gqa * K->ne[2]);
            const uint3 fd1 = init_fastdiv_values(ntiles_x * ntiles_z_gqa);
            const uint3 fd2 = init_fastdiv_values(ntiles_x);

            const dim3 block_dim_combine(DV, 1, 1);
            const dim3 blocks_num_combine = {(unsigned)ntiles_dst, ncols1, ncols2};

            const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(blocks_num_combine, block_dim_combine, 0, main_stream);
            ggml_cuda_kernel_launch(flash_attn_stream_k_fixup_uniform<DV, ncols1, ncols2>, launch_params,
                (float *) KQV->data, dst_tmp_meta.ptr,
                 Q->ne[1], Q->ne[2], K->ne[2], nblocks_sk,
                 gqa_ratio, bpt, fd0, fd1, fd2);
        } else if (ntiles_dst % blocks_num.x != 0) {
            // General fixup for the cases where nblocks_stream_k < ntiles_dst.
            const int total_work = ntiles_KV * ntiles_dst;

            const uint3 fd_k_j_z_ne12 = init_fastdiv_values(ntiles_KV * ntiles_x * ntiles_z_gqa * K->ne[2]);
            const uint3 fd_k_j_z      = init_fastdiv_values(ntiles_KV * ntiles_x * ntiles_z_gqa);
            const uint3 fd_k_j        = init_fastdiv_values(ntiles_KV * ntiles_x);
            const uint3 fd_k          = init_fastdiv_values(ntiles_KV);

            const dim3 block_dim_combine(DV, 1, 1);
            const dim3 blocks_num_combine = {blocks_num.x, ncols1, ncols2};

            const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(blocks_num_combine, block_dim_combine, 0, main_stream);
            ggml_cuda_kernel_launch(flash_attn_stream_k_fixup_general<DV, ncols1, ncols2>, launch_params,
                (float *) KQV->data, dst_tmp_meta.ptr,
                 Q->ne[1], Q->ne[2], gqa_ratio, total_work,
                 fd_k_j_z_ne12, fd_k_j_z, fd_k_j, fd_k);
        }
    } else if (parallel_blocks > 1) {
        const dim3 block_dim_combine(DV, 1, 1);
        const dim3 blocks_num_combine(Q->ne[1], Q->ne[2], Q->ne[3]);
        const size_t nbytes_shared_combine = parallel_blocks*sizeof(float2);

        const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(blocks_num_combine, block_dim_combine, nbytes_shared_combine, main_stream);
        ggml_cuda_kernel_launch(flash_attn_combine_results<DV>, launch_params,
            dst_tmp.ptr, dst_tmp_meta.ptr, (float *) KQV->data, parallel_blocks);
    }
    CUDA_CHECK(cudaGetLastError());
}
