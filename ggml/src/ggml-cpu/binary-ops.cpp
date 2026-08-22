#include "binary-ops.h"

#define GGML_COMMON_DECL_CPP
#include "ggml-common.h"

#if defined(GGML_USE_ACCELERATE)
#include <Accelerate/Accelerate.h>

using vDSP_fn_t = void (*)(const float *, vDSP_Stride, const float *, vDSP_Stride, float *, vDSP_Stride, vDSP_Length);
#endif

// ---- q8_0 intermediate-activation kernels (GGML_CPU_INT8_INTERMEDIATE path) ----
// Row-wise, block-aware binary ops on q8_0 activations. Rows are contiguous in
// blocks (nb0 == sizeof(block_q8_0)); row strides are arbitrary (view support).

#if defined(__AVX512F__)
// one q8_0 block = 32 values = two 16-lane f32 vectors
static inline void q8_0_block_to_ps2(const block_q8_0 * b, __m512 & v0, __m512 & v1) {
    const __m256i q8 = _mm256_loadu_si256((const __m256i *) b->qs);
    const __m512   d  = _mm512_set1_ps(GGML_CPU_FP16_TO_FP32(b->d));
    v0 = _mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(_mm256_castsi256_si128(q8))),    d);
    v1 = _mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(_mm256_extracti128_si256(q8, 1))), d);
}

static inline void ps2_to_q8_0_block(__m512 v0, __m512 v1, block_q8_0 * y) {
    const __m512 sign_bit = _mm512_set1_ps(-0.f);
    const __m512 vamax = _mm512_max_ps(_mm512_andnot_ps(sign_bit, v0), _mm512_andnot_ps(sign_bit, v1));
    const float amax = _mm512_reduce_max_ps(vamax);
    const float d  = amax / 127.f;
    const float id = amax ? 127.f / amax : 0.f;
    y->d = GGML_CPU_FP32_TO_FP16(d);
    const __m512 vid = _mm512_set1_ps(id);
    const __m512i q0 = _mm512_cvtps_epi32(_mm512_mul_ps(v0, vid)); // RNE
    const __m512i q1 = _mm512_cvtps_epi32(_mm512_mul_ps(v1, vid));
    _mm_storeu_si128((__m128i *) (y->qs +  0), _mm512_cvtepi32_epi8(q0));
    _mm_storeu_si128((__m128i *) (y->qs + 16), _mm512_cvtepi32_epi8(q1));
}
#endif

// z = x + y per block: dequant both, add in f32, requantize (amax + RNE)
static void add_row_q8_0(const block_q8_0 * GGML_RESTRICT x, const block_q8_0 * GGML_RESTRICT y,
                         block_q8_0 * GGML_RESTRICT z, int64_t n) {
    const int64_t nb = n / QK8_0;
#if defined(__AVX512F__)
    for (int64_t b = 0; b < nb; ++b) {
        __m512 x0, x1, y0, y1;
        q8_0_block_to_ps2(x + b, x0, x1);
        q8_0_block_to_ps2(y + b, y0, y1);
        ps2_to_q8_0_block(_mm512_add_ps(x0, y0), _mm512_add_ps(x1, y1), z + b);
    }
#else
    for (int64_t b = 0; b < nb; ++b) {
        const float dx = GGML_CPU_FP16_TO_FP32(x[b].d);
        const float dy = GGML_CPU_FP16_TO_FP32(y[b].d);
        float v[QK8_0];
        float amax = 0.0f;
        for (int j = 0; j < QK8_0; ++j) {
            v[j] = dx*x[b].qs[j] + dy*y[b].qs[j];
            amax = MAX(amax, fabsf(v[j]));
        }
        const float d  = amax / 127.f;
        const float id = amax ? 127.f / amax : 0.f;
        z[b].d = GGML_CPU_FP32_TO_FP16(d);
        for (int j = 0; j < QK8_0; ++j) {
            z[b].qs[j] = (int8_t) rintf(v[j] * id); // RNE
        }
    }
#endif
}

// z = x * w with a per-row scalar w: q8_0 stays exact by folding w into the scale
static void mul_row_q8_0_f32(const block_q8_0 * GGML_RESTRICT x, block_q8_0 * GGML_RESTRICT z, int64_t n, float w) {
    const int64_t nb = n / QK8_0;
    for (int64_t b = 0; b < nb; ++b) {
        z[b].d = GGML_CPU_FP32_TO_FP16(GGML_CPU_FP16_TO_FP32(x[b].d) * w);
        memcpy(z[b].qs, x[b].qs, QK8_0);
    }
}

// add(q8_0, q8_0) -> q8_0, same shape, block-contiguous rows with arbitrary strides
void ggml_compute_forward_add_q8_0_q8_0(const ggml_compute_params * params, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(ggml_are_same_shape(src0, src1) && ggml_are_same_shape(src0, dst));
    GGML_ASSERT(src0->type == GGML_TYPE_Q8_0 && src1->type == GGML_TYPE_Q8_0 && dst->type == GGML_TYPE_Q8_0);

    GGML_TENSOR_BINARY_OP_LOCALS

    // rows must be contiguous in blocks; higher-dim strides are arbitrary
    GGML_ASSERT(nb00 == (int64_t) sizeof(block_q8_0));
    GGML_ASSERT(nb10 == (int64_t) sizeof(block_q8_0));
    GGML_ASSERT(nb0  == (int64_t) sizeof(block_q8_0));
    GGML_ASSERT(ne00 % QK8_0 == 0);

    const int ith = params->ith;
    const int nth = params->nth;

    const int nr = (int) ggml_nrows(dst);
    const int dr = (nr + nth - 1)/nth;
    const int ir0 = dr*ith;
    const int ir1 = MIN(ir0 + dr, nr);

    for (int ir = ir0; ir < ir1; ++ir) {
        const int64_t i03 = ir/(ne02*ne01);
        const int64_t i02 = (ir - i03*ne02*ne01)/ne01;
        const int64_t i01 = (ir - i03*ne02*ne01 - i02*ne01);

        const block_q8_0 * x = (const block_q8_0 *) ((const char *) src0->data + i01*nb01 + i02*nb02 + i03*nb03);
        const block_q8_0 * y = (const block_q8_0 *) ((const char *) src1->data + i01*nb11 + i02*nb12 + i03*nb13);
        block_q8_0       * z = (block_q8_0 *)       ((char *)       dst->data + i01*nb1  + i02*nb2  + i03*nb3 );

        add_row_q8_0(x, y, z, ne00);
    }
}

// mul(q8_0, f32) -> q8_0 with f32 broadcast as a per-row scalar (src1->ne[0] == 1);
// exact: the scalar folds into each block's scale, qs are copied unchanged
void ggml_compute_forward_mul_q8_0_f32(const ggml_compute_params * params, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(ggml_can_repeat(src1, src0) && ggml_are_same_shape(src0, dst));
    GGML_ASSERT(src0->type == GGML_TYPE_Q8_0 && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_Q8_0);

    GGML_TENSOR_BINARY_OP_LOCALS

    GGML_ASSERT(ne10 == 1); // per-row scalar broadcast only
    GGML_ASSERT(nb00 == (int64_t) sizeof(block_q8_0));
    GGML_ASSERT(nb10 == (int64_t) sizeof(float));
    GGML_ASSERT(nb0  == (int64_t) sizeof(block_q8_0));
    GGML_ASSERT(ne00 % QK8_0 == 0);

    const int ith = params->ith;
    const int nth = params->nth;

    const int nr = (int) ggml_nrows(dst);
    const int dr = (nr + nth - 1)/nth;
    const int ir0 = dr*ith;
    const int ir1 = MIN(ir0 + dr, nr);

    for (int ir = ir0; ir < ir1; ++ir) {
        const int64_t i03 = ir/(ne02*ne01);
        const int64_t i02 = (ir - i03*ne02*ne01)/ne01;
        const int64_t i01 = (ir - i03*ne02*ne01 - i02*ne01);

        const int64_t i13 = i03 % ne13;
        const int64_t i12 = i02 % ne12;
        const int64_t i11 = i01 % ne11;

        const block_q8_0 * x = (const block_q8_0 *) ((const char *) src0->data + i01*nb01 + i02*nb02 + i03*nb03);
        const float        w = *(const float *)     ((const char *) src1->data + i11*nb11 + i12*nb12 + i13*nb13);
        block_q8_0       * z = (block_q8_0 *)       ((char *)       dst->data + i01*nb1  + i02*nb2  + i03*nb3 );

        mul_row_q8_0_f32(x, z, ne00, w);
    }
}

static inline float op_add(float a, float b) {
    return a + b;
}

static inline float op_sub(float a, float b) {
    return a - b;
}

static inline float op_mul(float a, float b) {
    return a * b;
}

static inline float op_div(float a, float b) {
    return a / b;
}

template <float (*op)(float, float), typename src0_t, typename src1_t, typename dst_t>
static inline void vec_binary_op_contiguous(const int64_t n, dst_t * z, const src0_t * x, const src1_t * y) {
    constexpr auto src0_to_f32 = type_conversion_table<src0_t>::to_f32;
    constexpr auto src1_to_f32 = type_conversion_table<src1_t>::to_f32;
    constexpr auto f32_to_dst  = type_conversion_table<dst_t >::from_f32;

    for (int i = 0; i < n; i++) {
        z[i] = f32_to_dst(op(src0_to_f32(x[i]), src1_to_f32(y[i])));
    }
}

template <float (*op)(float, float), typename src0_t, typename src1_t, typename dst_t>
static inline void vec_binary_op_non_contiguous(const int64_t n, const int64_t ne10, const int64_t nb10, dst_t * z, const src0_t * x, const src1_t * y) {
    constexpr auto src0_to_f32 = type_conversion_table<src0_t>::to_f32;
    constexpr auto src1_to_f32 = type_conversion_table<src1_t>::to_f32;
    constexpr auto f32_to_dst  = type_conversion_table<dst_t >::from_f32;

    for (int i = 0; i < n; i++) {
        int i10 = i % ne10;
        const src1_t * y_ptr = (const src1_t *)((const char *)y + i10*nb10);
        z[i] = f32_to_dst(op(src0_to_f32(x[i]), src1_to_f32(*y_ptr)));
    }
}

template <float (*op)(float, float), typename src0_t, typename src1_t, typename dst_t>
static void apply_binary_op(const ggml_compute_params * params, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(ggml_can_repeat(src1, src0) && ggml_are_same_shape(src0, dst));

    GGML_TENSOR_BINARY_OP_LOCALS

    GGML_ASSERT( nb0 == sizeof(dst_t));
    GGML_ASSERT(nb00 == sizeof(src0_t));

    const auto [ir0, ir1] = get_thread_range(params, src0);
    const bool is_src1_contiguous_rows = ggml_is_contiguous_rows(src1);

#ifdef GGML_USE_ACCELERATE
    vDSP_fn_t vDSP_op = nullptr;
    // TODO - avoid the f32-only check using type 'trait' lookup tables and row-based src-to-float conversion functions
    if (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
        if (op == op_add) {
            vDSP_op = vDSP_vadd;
        } else if (op == op_sub) {
            vDSP_op = vDSP_vsub;
        } else if (op == op_mul) {
            vDSP_op = vDSP_vmul;
        } else if (op == op_div) {
            vDSP_op = vDSP_vdiv;
        }
    }
#endif

    for (int64_t ir = ir0; ir < ir1; ++ir) {
        const int64_t i03 = ir/(ne02*ne01);
        const int64_t i02 = (ir - i03*ne02*ne01)/ne01;
        const int64_t i01 = (ir - i03*ne02*ne01 - i02*ne01);

        const int64_t i13 = i03 % ne13;
        const int64_t i12 = i02 % ne12;
        const int64_t i11 = i01 % ne11;

        dst_t        * dst_ptr  = (dst_t  *)       ((char *)       dst->data  + i03*nb3  + i02*nb2  + i01*nb1 );
        const src0_t * src0_ptr = (const src0_t *) ((const char *) src0->data + i03*nb03 + i02*nb02 + i01*nb01);
        const src1_t * src1_ptr = (const src1_t *) ((const char *) src1->data + i13*nb13 + i12*nb12 + i11*nb11);

        if (is_src1_contiguous_rows) {
            // src1 is broadcastable across src0 and dst in i1, i2, i3
            const int64_t nr0 = ne00 / ne10;

            for (int64_t r = 0; r < nr0; ++r) {
#ifdef GGML_USE_ACCELERATE
                if constexpr (std::is_same_v<src0_t, float> && std::is_same_v<src1_t, float> && std::is_same_v<dst_t, float>) {
                    if (vDSP_op != nullptr) {
                        vDSP_op(src1_ptr, 1, src0_ptr + r*ne10, 1, dst_ptr + r*ne10, 1, ne10);
                        continue;
                    }
                }
#endif
                vec_binary_op_contiguous<op>(ne10, dst_ptr + r*ne10, src0_ptr + r*ne10, src1_ptr);
            }
        } else {
            vec_binary_op_non_contiguous<op>(ne0, ne10, nb10, dst_ptr, src0_ptr, src1_ptr);
        }
    }
}

// TODO: Use the 'traits' lookup table (for type conversion fns), instead of a mass of 'if' conditions with long templates
template <float (*op)(float, float)>
static void binary_op(const ggml_compute_params * params, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    /*  */ if (src0->type == GGML_TYPE_F32  && src1->type == GGML_TYPE_F32  && dst->type == GGML_TYPE_F32) { // all f32
        apply_binary_op<op, float, float, float>(params, dst);
    } else if (src0->type == GGML_TYPE_F16  && src1->type == GGML_TYPE_F16  && dst->type == GGML_TYPE_F16) { // all f16
        apply_binary_op<op, ggml_fp16_t, ggml_fp16_t, ggml_fp16_t>(params, dst);
    } else if (src0->type == GGML_TYPE_BF16 && src1->type == GGML_TYPE_BF16 && dst->type == GGML_TYPE_BF16) { // all bf16
        apply_binary_op<op, ggml_bf16_t, ggml_bf16_t, ggml_bf16_t>(params, dst);
    } else if (src0->type == GGML_TYPE_BF16 && src1->type == GGML_TYPE_F32  && dst->type == GGML_TYPE_BF16) {
        apply_binary_op<op, ggml_bf16_t, float, ggml_bf16_t>(params, dst);
    } else if (src0->type == GGML_TYPE_BF16 && src1->type == GGML_TYPE_F32  && dst->type == GGML_TYPE_F32) {
        apply_binary_op<op, ggml_bf16_t, float, float>(params, dst);
    } else if (src0->type == GGML_TYPE_F16  && src1->type == GGML_TYPE_F32  && dst->type == GGML_TYPE_F16) {
        apply_binary_op<op, ggml_fp16_t, float, ggml_fp16_t>(params, dst);
    } else if (src0->type == GGML_TYPE_F16  && src1->type == GGML_TYPE_F32  && dst->type == GGML_TYPE_F32) {
        apply_binary_op<op, ggml_fp16_t, float, float>(params, dst);
    } else {
        GGML_ABORT("%s: unsupported types: dst: %s, src0: %s, src1: %s\n", __func__,
            ggml_type_name(dst->type), ggml_type_name(src0->type), ggml_type_name(src1->type));
    }
}

void ggml_compute_forward_add_non_quantized(const ggml_compute_params * params, ggml_tensor * dst) {
    binary_op<op_add>(params, dst);
}

void ggml_compute_forward_sub(const ggml_compute_params * params, ggml_tensor * dst) {
    binary_op<op_sub>(params, dst);
}

void ggml_compute_forward_mul(const ggml_compute_params * params, ggml_tensor * dst) {
    if (dst->type == GGML_TYPE_Q8_0 && dst->src[0]->type == GGML_TYPE_Q8_0 && dst->src[1]->type == GGML_TYPE_F32) {
        ggml_compute_forward_mul_q8_0_f32(params, dst);
        return;
    }
    binary_op<op_mul>(params, dst);
}

void ggml_compute_forward_div(const ggml_compute_params * params, ggml_tensor * dst) {
    binary_op<op_div>(params, dst);
}
