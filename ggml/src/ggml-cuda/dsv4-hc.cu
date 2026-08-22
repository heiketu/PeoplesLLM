#include "common.cuh"
#include "dsv4-hc.cuh"


static constexpr int DSV4_HC = 4;

static __device__ __forceinline__ float dsv4_sigmoid_f32(float x) {
    // identical to op_sigmoid in unary.cu
    return 1.0f / (1.0f + expf(-x));
}


static __device__ void dsv4_hc_comb_norm_cols(float * comb, float eps) {
    for (int idst = 0; idst < DSV4_HC; ++idst) {
        float sum = eps;
        for (int isrc = 0; isrc < DSV4_HC; ++isrc) {
            sum += comb[idst + DSV4_HC*isrc];
        }

        const float inv_sum = 1.0f / sum;
        for (int isrc = 0; isrc < DSV4_HC; ++isrc) {
            comb[idst + DSV4_HC*isrc] *= inv_sum;
        }
    }
}

static __device__ void dsv4_hc_comb_norm_rows(float * comb, float eps) {
    for (int isrc = 0; isrc < DSV4_HC; ++isrc) {
        float sum = eps;
        for (int idst = 0; idst < DSV4_HC; ++idst) {
            sum += comb[idst + DSV4_HC*isrc];
        }

        const float inv_sum = 1.0f / sum;
        for (int idst = 0; idst < DSV4_HC; ++idst) {
            comb[idst + DSV4_HC*isrc] *= inv_sum;
        }
    }
}

static __global__ void dsv4_hc_comb_f32(
        const float * mixes,
        const float * scale,
        const float * base,
        float * dst,
        int64_t n_tokens,
        int64_t sm0,
        int64_t sm1,
        int64_t ss0,
        int64_t sb0,
        int64_t sd0,
        int64_t sd1,
        int64_t sd2,
        float eps,
        int32_t n_iter) {
    constexpr int comb_offset = 2*DSV4_HC;

    ggml_cuda_pdl_lc();
    const int64_t it = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;

    if (it >= n_tokens) {
        return;
    }

    ggml_cuda_pdl_sync();

    const float scale_comb = scale[2*ss0];
    float comb[DSV4_HC*DSV4_HC];

    for (int isrc = 0; isrc < DSV4_HC; ++isrc) {
        float max = -INFINITY;
        for (int idst = 0; idst < DSV4_HC; ++idst) {
            const int idx = idst + DSV4_HC*isrc;
            const float v = mixes[(comb_offset + idx)*sm0 + it*sm1] * scale_comb + base[(comb_offset + idx)*sb0];
            comb[idx] = v;
            max = fmaxf(max, v);
        }

        float sum = 0.0f;
        for (int idst = 0; idst < DSV4_HC; ++idst) {
            const int idx = idst + DSV4_HC*isrc;
            const float v = expf(comb[idx] - max);
            comb[idx] = v;
            sum += v;
        }

        const float inv_sum = 1.0f / sum;
        for (int idst = 0; idst < DSV4_HC; ++idst) {
            const int idx = idst + DSV4_HC*isrc;
            comb[idx] = comb[idx] * inv_sum + eps;
        }
    }

    dsv4_hc_comb_norm_cols(comb, eps);
    for (int32_t i = 1; i < n_iter; ++i) {
        dsv4_hc_comb_norm_rows(comb, eps);
        dsv4_hc_comb_norm_cols(comb, eps);
    }

    for (int isrc = 0; isrc < DSV4_HC; ++isrc) {
        for (int idst = 0; idst < DSV4_HC; ++idst) {
            const int idx = idst + DSV4_HC*isrc;
            dst[idst*sd0 + isrc*sd1 + it*sd2] = comb[idx];
        }
    }
}

static __global__ void dsv4_hc_pre_f32(
        const float * x,
        const float * weights,
        float * dst,
        int64_t n_embd,
        int64_t hc,
        int64_t n_tokens,
        int64_t sx0,
        int64_t sx1,
        int64_t sx2,
        int64_t sw0,
        int64_t sw1,
        int64_t sd0,
        int64_t sd1) {
    ggml_cuda_pdl_lc();
    const int64_t ir = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t nr = n_embd * n_tokens;

    if (ir >= nr) {
        return;
    }

    ggml_cuda_pdl_sync();

    const int64_t i0 = ir % n_embd;
    const int64_t it = ir / n_embd;

    float sum = x[i0*sx0 + it*sx2] * weights[it*sw1];
    for (int64_t ih = 1; ih < hc; ++ih) {
        const float xv = x[i0*sx0 + ih*sx1 + it*sx2];
        const float wv = weights[ih*sw0 + it*sw1];
        sum += xv * wv;
    }

    dst[i0*sd0 + it*sd1] = sum;
}

static __global__ void dsv4_hc_post_f32(
        const float * x,
        const float * residual,
        const float * post,
        const float * comb,
        float * dst,
        int64_t n_embd,
        int64_t hc,
        int64_t n_tokens,
        int64_t sx0,
        int64_t sx1,
        int64_t sr0,
        int64_t sr1,
        int64_t sr2,
        int64_t sp0,
        int64_t sp1,
        int64_t sc0,
        int64_t sc1,
        int64_t sc2,
        int64_t sd0,
        int64_t sd1,
        int64_t sd2) {
    ggml_cuda_pdl_lc();
    const int64_t ir = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t nr = n_embd * hc * n_tokens;

    if (ir >= nr) {
        return;
    }

    ggml_cuda_pdl_sync();

    const int64_t i0   = ir % n_embd;
    const int64_t idst = (ir / n_embd) % hc;
    const int64_t it   = ir / (n_embd * hc);

    float sum = x[i0*sx0 + it*sx1] * post[idst*sp0 + it*sp1];
    for (int64_t isrc = 0; isrc < hc; ++isrc) {
        sum += residual[i0*sr0 + isrc*sr1 + it*sr2] * comb[idst*sc0 + isrc*sc1 + it*sc2];
    }

    dst[i0*sd0 + idst*sd1 + it*sd2] = sum;
}

void ggml_cuda_op_dsv4_hc_comb(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * mixes = dst->src[0];
    const ggml_tensor * scale = dst->src[1];
    const ggml_tensor * base  = dst->src[2];

    GGML_ASSERT(mixes->type == GGML_TYPE_F32);
    GGML_ASSERT(scale->type == GGML_TYPE_F32);
    GGML_ASSERT(base->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    constexpr int64_t hc_mix_dim = (2 + DSV4_HC)*DSV4_HC;

    GGML_ASSERT(mixes->ne[0] == hc_mix_dim);
    GGML_ASSERT(dst->ne[0] == DSV4_HC);
    GGML_ASSERT(dst->ne[1] == DSV4_HC);
    GGML_ASSERT(dst->ne[2] == mixes->ne[1]);
    GGML_ASSERT(scale->ne[0] >= 3);
    GGML_ASSERT(base->ne[0] == hc_mix_dim);

    GGML_TENSOR_LOCALS(size_t, nbm, mixes, nb);
    GGML_TENSOR_LOCALS(size_t, nbs, scale, nb);
    GGML_TENSOR_LOCALS(size_t, nbb, base,  nb);
    GGML_TENSOR_LOCALS(size_t, nbd, dst,   nb);

    const int64_t n_tokens = mixes->ne[1];
    const float eps = ggml_get_op_params_f32(dst, 0);
    const int32_t n_iter = ggml_get_op_params_i32(dst, 1);

    const int block_size = 256;
    const dim3 block_dims(block_size, 1, 1);
    const dim3 grid_dims((n_tokens + block_size - 1) / block_size, 1, 1);
    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(grid_dims, block_dims, 0, ctx.stream());

    ggml_cuda_kernel_launch(dsv4_hc_comb_f32, launch_params,
            (const float *) mixes->data, (const float *) scale->data, (const float *) base->data, (float *) dst->data,
            n_tokens,
            nbm0 / sizeof(float), nbm1 / sizeof(float),
            nbs0 / sizeof(float),
            nbb0 / sizeof(float),
            nbd0 / sizeof(float), nbd1 / sizeof(float), nbd2 / sizeof(float),
            eps, n_iter);
}

void ggml_cuda_op_dsv4_hc_pre(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * x       = dst->src[0];
    const ggml_tensor * weights = dst->src[1];

    GGML_ASSERT(x->type == GGML_TYPE_F32);
    GGML_ASSERT(weights->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    GGML_TENSOR_LOCALS(size_t, nbx, x,       nb);
    GGML_TENSOR_LOCALS(size_t, nbw, weights, nb);
    GGML_TENSOR_LOCALS(size_t, nbd, dst,     nb);

    const int64_t n_embd   = x->ne[0];
    const int64_t hc       = x->ne[1];
    const int64_t n_tokens = x->ne[2];

    const int block_size = 256;
    const int64_t nr = n_embd * n_tokens;
    const dim3 block_dims(block_size, 1, 1);
    const dim3 grid_dims((nr + block_size - 1) / block_size, 1, 1);
    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(grid_dims, block_dims, 0, ctx.stream());

    ggml_cuda_kernel_launch(dsv4_hc_pre_f32, launch_params,
            (const float *) x->data, (const float *) weights->data, (float *) dst->data,
            n_embd, hc, n_tokens,
            nbx0 / sizeof(float), nbx1 / sizeof(float), nbx2 / sizeof(float),
            nbw0 / sizeof(float), nbw1 / sizeof(float),
            nbd0 / sizeof(float), nbd1 / sizeof(float));
}

void ggml_cuda_op_dsv4_hc_post(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * x        = dst->src[0];
    const ggml_tensor * residual = dst->src[1];
    const ggml_tensor * post     = dst->src[2];
    const ggml_tensor * comb     = dst->src[3];

    GGML_ASSERT(x->type == GGML_TYPE_F32);
    GGML_ASSERT(residual->type == GGML_TYPE_F32);
    GGML_ASSERT(post->type == GGML_TYPE_F32);
    GGML_ASSERT(comb->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    GGML_TENSOR_LOCALS(size_t, nbx, x,        nb);
    GGML_TENSOR_LOCALS(size_t, nbr, residual, nb);
    GGML_TENSOR_LOCALS(size_t, nbp, post,     nb);
    GGML_TENSOR_LOCALS(size_t, nbc, comb,     nb);
    GGML_TENSOR_LOCALS(size_t, nbd, dst,      nb);

    const int64_t n_embd   = x->ne[0];
    const int64_t n_tokens = x->ne[1];
    const int64_t hc       = residual->ne[1];

    const int block_size = 256;
    const int64_t nr = n_embd * hc * n_tokens;
    const dim3 block_dims(block_size, 1, 1);
    const dim3 grid_dims((nr + block_size - 1) / block_size, 1, 1);
    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(grid_dims, block_dims, 0, ctx.stream());

    ggml_cuda_kernel_launch(dsv4_hc_post_f32, launch_params,
            (const float *) x->data, (const float *) residual->data,
            (const float *) post->data, (const float *) comb->data, (float *) dst->data,
            n_embd, hc, n_tokens,
            nbx0 / sizeof(float), nbx1 / sizeof(float),
            nbr0 / sizeof(float), nbr1 / sizeof(float), nbr2 / sizeof(float),
            nbp0 / sizeof(float), nbp1 / sizeof(float),
            nbc0 / sizeof(float), nbc1 / sizeof(float), nbc2 / sizeof(float),
            nbd0 / sizeof(float), nbd1 / sizeof(float), nbd2 / sizeof(float));
}

// fused pre + post + comb: one launch produces all three tensors packed into
// dst [(2 + hc)*hc, n_tokens]: rows [0, hc) pre, [hc, 2*hc) post, [2*hc, ...) comb.
static __global__ void dsv4_hc_prep_f32(
        const float * mixes,
        const float * scale,
        const float * base,
        float * dst,
        int64_t n_tokens,
        int64_t sm0,
        int64_t sm1,
        int64_t ss0,
        int64_t sb0,
        int64_t sd0,
        int64_t sd1,
        float eps,
        int32_t n_iter) {
    constexpr int comb_offset = 2*DSV4_HC;

    ggml_cuda_pdl_lc();
    const int64_t it = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;

    if (it >= n_tokens) {
        return;
    }

    ggml_cuda_pdl_sync();

    const float scale_pre  = scale[0*ss0];
    const float scale_post = scale[1*ss0];
    const float scale_comb = scale[2*ss0];

    // pre/post: affine (rounded mul, then rounded add) + sigmoid + scale,
    // bit-identical to the unfused mul + add + sigmoid + scale_bias/scale chain
    for (int r = 0; r < 2*DSV4_HC; ++r) {
        const float xv = mixes[r*sm0 + it*sm1];
        const float bv = base[r*sb0];
        const float sv = r < DSV4_HC ? scale_pre : scale_post;
        float v = __fmul_rn(xv, sv);
        v = __fadd_rn(v, bv);
        const float sg = dsv4_sigmoid_f32(v);
        const float o  = r < DSV4_HC ? __fadd_rn(sg, eps) : __fmul_rn(sg, 2.0f);
        dst[r*sd0 + it*sd1] = o;
    }

    // comb: identical to dsv4_hc_comb_f32
    float comb[DSV4_HC*DSV4_HC];

    for (int isrc = 0; isrc < DSV4_HC; ++isrc) {
        float max = -INFINITY;
        for (int idst = 0; idst < DSV4_HC; ++idst) {
            const int idx = idst + DSV4_HC*isrc;
            const float v = mixes[(comb_offset + idx)*sm0 + it*sm1] * scale_comb + base[(comb_offset + idx)*sb0];
            comb[idx] = v;
            max = fmaxf(max, v);
        }

        float sum = 0.0f;
        for (int idst = 0; idst < DSV4_HC; ++idst) {
            const int idx = idst + DSV4_HC*isrc;
            const float v = expf(comb[idx] - max);
            comb[idx] = v;
            sum += v;
        }

        const float inv_sum = 1.0f / sum;
        for (int idst = 0; idst < DSV4_HC; ++idst) {
            const int idx = idst + DSV4_HC*isrc;
            comb[idx] = comb[idx] * inv_sum + eps;
        }
    }

    dsv4_hc_comb_norm_cols(comb, eps);
    for (int32_t i = 1; i < n_iter; ++i) {
        dsv4_hc_comb_norm_rows(comb, eps);
        dsv4_hc_comb_norm_cols(comb, eps);
    }

    for (int isrc = 0; isrc < DSV4_HC; ++isrc) {
        for (int idst = 0; idst < DSV4_HC; ++idst) {
            const int idx = idst + DSV4_HC*isrc;
            dst[(comb_offset + idx)*sd0 + it*sd1] = comb[idx];
        }
    }
}

// fused sigmoid + bias for the MoE router, packed [2*n_expert, n_tokens]:
// rows [0, n_expert) probs = sigmoid(logits), rows [n_expert, ...) probs + bias.
static __global__ void dsv4_moe_probs_f32(
        const float * logits,
        const float * bias,
        float * dst,
        int64_t n_expert,
        int64_t n_tokens,
        int64_t sl0,
        int64_t sl1,
        int64_t sb0,
        int64_t sd0,
        int64_t sd1) {
    ggml_cuda_pdl_lc();
    const int64_t ir = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t nr = n_expert * n_tokens;

    if (ir >= nr) {
        return;
    }

    ggml_cuda_pdl_sync();

    const int64_t ie = ir % n_expert;
    const int64_t it = ir / n_expert;

    const float p = dsv4_sigmoid_f32(logits[ie*sl0 + it*sl1]);

    dst[ie*sd0 + it*sd1] = p;
    dst[(n_expert + ie)*sd0 + it*sd1] = __fadd_rn(p, bias[ie*sb0]);
}

// fused group-mask + top-k expert selection, one thread per token.
// selection order matches ggml_argsort_top_k (descending, ties keep the lower index).
static __global__ void dsv4_moe_topk_f32(
        const float * probs,
        int32_t * dst,
        int64_t n_expert,
        int64_t n_tokens,
        int64_t sp0,
        int64_t sp1,
        int32_t n_groups,
        int32_t n_group_used,
        int32_t k) {
    ggml_cuda_pdl_lc();
    const int64_t it = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;

    if (it >= n_tokens) {
        return;
    }

    ggml_cuda_pdl_sync();

    const int64_t n_exp_per_group = n_expert / n_groups;

    const float * sp = probs + it*sp1;

    // group scores: sum of the top-2 selection probs within each group
    float group_scores[64];
    for (int32_t g = 0; g < n_groups; ++g) {
        const float * gp = sp + (int64_t) g*n_exp_per_group*sp0;

        float top1 = -INFINITY;
        float top2 = -INFINITY;
        for (int64_t e = 0; e < n_exp_per_group; ++e) {
            const float v = gp[e*sp0];
            if (v > top1) {
                top2 = top1;
                top1 = v;
            } else if (v > top2) {
                top2 = v;
            }
        }
        group_scores[g] = top1 + top2;
    }

    // select the top n_group_used groups (descending, ties keep the lower index)
    bool group_used[64];
    for (int32_t g = 0; g < n_groups; ++g) {
        group_used[g] = false;
    }
    for (int32_t i = 0; i < n_group_used; ++i) {
        float best = -INFINITY;
        int32_t best_g = -1;
        for (int32_t g = 0; g < n_groups; ++g) {
            if (!group_used[g] && group_scores[g] > best) {
                best = group_scores[g];
                best_g = g;
            }
        }
        group_used[best_g] = true;
    }

    // select the top k experts among the used groups (descending, ties keep the lower index)
    int32_t sel[32];
    for (int32_t i = 0; i < k; ++i) {
        float best = -INFINITY;
        int64_t best_e = -1;
        for (int64_t e = 0; e < n_expert; ++e) {
            if (!group_used[e / n_exp_per_group]) {
                continue;
            }
            bool taken = false;
            for (int32_t j = 0; j < i; ++j) {
                taken |= sel[j] == (int32_t) e;
            }
            if (taken) {
                continue;
            }
            const float v = sp[e*sp0];
            if (v > best) {
                best = v;
                best_e = e;
            }
        }
        sel[i] = (int32_t) best_e;
        dst[i + it*k] = (int32_t) best_e;
    }
}

// fused gather + optional normalize + optional scale of expert weights.
// the row sum replicates the reduce_rows_f32 (sum_rows) reduction order for
// ncols <= 32: 32 zero-padded lanes, shuffle-xor tree with offsets 16..1.
static __global__ void dsv4_moe_weights_f32(
        const float * probs,
        const int32_t * selected,
        float * dst,
        int64_t n_tokens,
        int64_t sp0,
        int64_t sp1,
        int64_t ss0,
        int64_t ss1,
        int32_t k,
        float scale,
        int32_t normalize) {
    ggml_cuda_pdl_lc();
    const int64_t it = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;

    if (it >= n_tokens) {
        return;
    }

    ggml_cuda_pdl_sync();

    float w[32];
    for (int32_t i = 0; i < k; ++i) {
        const int32_t idx = selected[i*ss0 + it*ss1];
        w[i] = probs[idx*sp0 + it*sp1];
    }

    if (normalize) {
        float lane[32];
#pragma unroll
        for (int i = 0; i < 32; ++i) {
            lane[i] = i < k ? w[i] : 0.0f;
        }
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            float next[32];
#pragma unroll
            for (int i = 0; i < 32; ++i) {
                next[i] = __fadd_rn(lane[i], lane[i ^ off]);
            }
#pragma unroll
            for (int i = 0; i < 32; ++i) {
                lane[i] = next[i];
            }
        }

        // identical to ggml_clamp(x, 6.103515625e-5, +inf)
        const float sum = fminf(fmaxf(lane[0], 6.103515625e-5f), INFINITY);
        for (int32_t i = 0; i < k; ++i) {
            w[i] = __fdiv_rn(w[i], sum);
        }
    }

    if (scale != 0.0f && scale != 1.0f) {
        for (int32_t i = 0; i < k; ++i) {
            w[i] = __fmul_rn(w[i], scale);
        }
    }

    for (int32_t i = 0; i < k; ++i) {
        dst[i + it*k] = w[i];
    }
}

void ggml_cuda_op_dsv4_hc_prep(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * mixes = dst->src[0];
    const ggml_tensor * scale = dst->src[1];
    const ggml_tensor * base  = dst->src[2];

    GGML_ASSERT(mixes->type == GGML_TYPE_F32);
    GGML_ASSERT(scale->type == GGML_TYPE_F32);
    GGML_ASSERT(base->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    constexpr int64_t hc_mix_dim = (2 + DSV4_HC)*DSV4_HC;

    GGML_ASSERT(mixes->ne[0] == hc_mix_dim);
    GGML_ASSERT(dst->ne[0] == hc_mix_dim);
    GGML_ASSERT(dst->ne[1] == mixes->ne[1]);
    GGML_ASSERT(scale->ne[0] >= 3);
    GGML_ASSERT(base->ne[0] == hc_mix_dim);

    GGML_TENSOR_LOCALS(size_t, nbm, mixes, nb);
    GGML_TENSOR_LOCALS(size_t, nbs, scale, nb);
    GGML_TENSOR_LOCALS(size_t, nbb, base,  nb);
    GGML_TENSOR_LOCALS(size_t, nbd, dst,   nb);

    const int64_t n_tokens = mixes->ne[1];
    const float eps = ggml_get_op_params_f32(dst, 0);
    const int32_t n_iter = ggml_get_op_params_i32(dst, 1);

    const int block_size = 256;
    const dim3 block_dims(block_size, 1, 1);
    const dim3 grid_dims((n_tokens + block_size - 1) / block_size, 1, 1);
    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(grid_dims, block_dims, 0, ctx.stream());

    ggml_cuda_kernel_launch(dsv4_hc_prep_f32, launch_params,
            (const float *) mixes->data, (const float *) scale->data, (const float *) base->data, (float *) dst->data,
            n_tokens,
            nbm0 / sizeof(float), nbm1 / sizeof(float),
            nbs0 / sizeof(float),
            nbb0 / sizeof(float),
            nbd0 / sizeof(float), nbd1 / sizeof(float),
            eps, n_iter);
}

void ggml_cuda_op_dsv4_moe_probs(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * logits = dst->src[0];
    const ggml_tensor * bias   = dst->src[1];

    GGML_ASSERT(logits->type == GGML_TYPE_F32);
    GGML_ASSERT(bias->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    GGML_TENSOR_LOCALS(size_t, nbl, logits, nb);
    GGML_TENSOR_LOCALS(size_t, nbb, bias,   nb);
    GGML_TENSOR_LOCALS(size_t, nbd, dst,    nb);

    const int64_t n_expert = logits->ne[0];
    const int64_t n_tokens = logits->ne[1];

    GGML_ASSERT(bias->ne[0] == n_expert);
    GGML_ASSERT(dst->ne[0] == 2*n_expert);
    GGML_ASSERT(dst->ne[1] == n_tokens);

    const int block_size = 256;
    const int64_t nr = n_expert * n_tokens;
    const dim3 block_dims(block_size, 1, 1);
    const dim3 grid_dims((nr + block_size - 1) / block_size, 1, 1);
    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(grid_dims, block_dims, 0, ctx.stream());

    ggml_cuda_kernel_launch(dsv4_moe_probs_f32, launch_params,
            (const float *) logits->data, (const float *) bias->data, (float *) dst->data,
            n_expert, n_tokens,
            nbl0 / sizeof(float), nbl1 / sizeof(float),
            nbb0 / sizeof(float),
            nbd0 / sizeof(float), nbd1 / sizeof(float));
}

void ggml_cuda_op_dsv4_moe_topk(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * probs = dst->src[0];

    GGML_ASSERT(probs->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_I32);

    GGML_TENSOR_LOCALS(size_t, nbp, probs, nb);
    GGML_TENSOR_LOCALS(size_t, nbd, dst,   nb);

    const int64_t n_expert = probs->ne[0];
    const int64_t n_tokens = probs->ne[1];

    const int32_t n_groups     = ggml_get_op_params_i32(dst, 0);
    const int32_t n_group_used = ggml_get_op_params_i32(dst, 1);
    const int32_t k            = ggml_get_op_params_i32(dst, 2);

    GGML_ASSERT(n_expert % n_groups == 0);
    GGML_ASSERT(n_groups <= 64);
    GGML_ASSERT(k <= 32);
    GGML_ASSERT(dst->ne[0] == k);
    GGML_ASSERT(dst->ne[1] == n_tokens);
    GGML_ASSERT((int64_t) n_group_used*(n_expert/n_groups) >= k);
    GGML_ASSERT(nbd0 == sizeof(int32_t)); // dst rows are contiguous

    const int block_size = 256;
    const dim3 block_dims(block_size, 1, 1);
    const dim3 grid_dims((n_tokens + block_size - 1) / block_size, 1, 1);
    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(grid_dims, block_dims, 0, ctx.stream());

    ggml_cuda_kernel_launch(dsv4_moe_topk_f32, launch_params,
            (const float *) probs->data, (int32_t *) dst->data,
            n_expert, n_tokens,
            nbp0 / sizeof(float), nbp1 / sizeof(float),
            n_groups, n_group_used, k);
}

// fused DeepSeek-V4 router for the n_expert_groups == 1 path:
// softplus -> sqrt -> add(bias) -> argsort(desc) -> view -> dsv4_moe_weights.
// one warp per token computes probs = sqrt(softplus(logits)) and writes them to the
// (otherwise elided) SQRT node output, then selects the top-k experts on probs + bias
// (descending, ties keep the lower index, matching ggml_argsort_top_k) and writes them
// to the first k entries of the ARGSORT node output. DSV4_MOE_WEIGHTS runs unfused
// on top of these two outputs.
template <int n_experts>
__launch_bounds__(4 * WARP_SIZE, 1) __global__ void dsv4_moe_router_f32(
        const float * logits,
        const float * bias,
        float * probs,
        int32_t * ids,
        int64_t n_tokens,
        int64_t sl0,
        int64_t sl1,
        int64_t sb0,
        int64_t sp0,
        int64_t sp1,
        int64_t si0,
        int64_t si1,
        int32_t k) {
    constexpr int experts_per_thread = n_experts / WARP_SIZE;

    const int64_t it = (int64_t) blockIdx.x * blockDim.y + threadIdx.y;
    if (it >= n_tokens) {
        return;
    }

    ggml_cuda_pdl_sync();

    float wt[experts_per_thread];

    // identical to op_softplus followed by op_sqrt in unary.cu
#pragma unroll
    for (int i = 0; i < experts_per_thread; ++i) {
        const int   e = threadIdx.x + i*WARP_SIZE;
        const float x = logits[e*sl0 + it*sl1];
        wt[i] = sqrtf(x > 20.0f ? x : logf(1.0f + expf(x)));
        probs[e*sp0 + it*sp1] = wt[i];
    }

    // selection scores, identical to the elided ADD of the bias
    float sel[experts_per_thread];
#pragma unroll
    for (int i = 0; i < experts_per_thread; ++i) {
        const int e = threadIdx.x + i*WARP_SIZE;
        sel[i] = __fadd_rn(wt[i], bias[e*sb0]);
        // keep the argmax total even for NaN scores (same as topk_moe_cuda)
        if (__isnanf(sel[i])) {
            sel[i] = -FLT_MAX;
        }
    }

    for (int32_t i = 0; i < k; ++i) {
        float best_v = sel[0];
        int   best_e = threadIdx.x;

#pragma unroll
        for (int j = 1; j < experts_per_thread; ++j) {
            if (sel[j] > best_v) {
                best_v = sel[j];
                best_e = threadIdx.x + j*WARP_SIZE;
            }
        }

#pragma unroll
        for (int mask = WARP_SIZE/2; mask > 0; mask >>= 1) {
            const float v = __shfl_xor_sync(0xffffffff, best_v, mask, WARP_SIZE);
            const int   e = __shfl_xor_sync(0xffffffff, best_e, mask, WARP_SIZE);
            if (v > best_v || (v == best_v && e < best_e)) {
                best_v = v;
                best_e = e;
            }
        }

        if ((best_e & (WARP_SIZE - 1)) == threadIdx.x) {
            sel[best_e / WARP_SIZE] = -INFINITY;
            ids[i*si0 + it*si1] = best_e;
        }
    }
}

void ggml_cuda_op_dsv4_moe_router(ggml_backend_cuda_context & ctx,
                                  const ggml_tensor *         logits,
                                  const ggml_tensor *         bias,
                                  ggml_tensor *               probs,
                                  ggml_tensor *               ids,
                                  int32_t                     k) {
    GGML_ASSERT(logits->type == GGML_TYPE_F32);
    GGML_ASSERT(bias->type   == GGML_TYPE_F32);
    GGML_ASSERT(probs->type  == GGML_TYPE_F32);
    GGML_ASSERT(ids->type    == GGML_TYPE_I32);

    GGML_TENSOR_LOCALS(size_t, nbl, logits, nb);
    GGML_TENSOR_LOCALS(size_t, nbb, bias,   nb);
    GGML_TENSOR_LOCALS(size_t, nbp, probs,  nb);
    GGML_TENSOR_LOCALS(size_t, nbi, ids,    nb);

    const int64_t n_expert = logits->ne[0];
    const int64_t n_tokens = logits->ne[1];

    GGML_ASSERT(ggml_is_contiguous(logits));
    GGML_ASSERT(ggml_is_contiguous(probs));
    GGML_ASSERT(ggml_is_contiguous(ids));
    GGML_ASSERT(bias->ne[0] == n_expert);
    GGML_ASSERT(probs->ne[0] == n_expert && probs->ne[1] == n_tokens);
    GGML_ASSERT(ids->ne[0] == n_expert && ids->ne[1] == n_tokens);
    GGML_ASSERT(k > 0 && k <= 32 && k <= n_expert);

    const int  rows_per_block = 4;
    const dim3 grid_dims((n_tokens + rows_per_block - 1) / rows_per_block, 1, 1);
    const dim3 block_dims(WARP_SIZE, rows_per_block, 1);
    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(grid_dims, block_dims, 0, ctx.stream());

#define DSV4_MOE_ROUTER_CASE(N) \
    case N: \
        ggml_cuda_kernel_launch(dsv4_moe_router_f32<N>, launch_params, \
                (const float *) logits->data, (const float *) bias->data, \
                (float *) probs->data, (int32_t *) ids->data, \
                n_tokens, \
                nbl0 / sizeof(float), nbl1 / sizeof(float), \
                nbb0 / sizeof(float), \
                nbp0 / sizeof(float), nbp1 / sizeof(float), \
                nbi0 / sizeof(int32_t), nbi1 / sizeof(int32_t), \
                k); \
        break

    switch (n_expert) {
        DSV4_MOE_ROUTER_CASE(32);
        DSV4_MOE_ROUTER_CASE(64);
        DSV4_MOE_ROUTER_CASE(128);
        DSV4_MOE_ROUTER_CASE(256);
        default:
            GGML_ASSERT(false && "unsupported n_expert");
            break;
    }
#undef DSV4_MOE_ROUTER_CASE
}

void ggml_cuda_op_dsv4_moe_weights(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * probs    = dst->src[0];
    const ggml_tensor * selected = dst->src[1];

    GGML_ASSERT(probs->type == GGML_TYPE_F32);
    GGML_ASSERT(selected->type == GGML_TYPE_I32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    GGML_TENSOR_LOCALS(size_t, nbp, probs,    nb);
    GGML_TENSOR_LOCALS(size_t, nbs, selected, nb);
    GGML_TENSOR_LOCALS(size_t, nbd, dst,      nb);

    const int64_t n_tokens = probs->ne[1];
    const int32_t k        = selected->ne[0];

    GGML_ASSERT(selected->ne[1] == n_tokens);
    GGML_ASSERT(dst->ne[0] == k);
    GGML_ASSERT(dst->ne[1] == n_tokens);
    GGML_ASSERT(k <= 32);
    GGML_ASSERT(nbd0 == sizeof(float)); // dst rows are contiguous

    const float   scale     = ggml_get_op_params_f32(dst, 0);
    const int32_t normalize = ggml_get_op_params_i32(dst, 1);

    const int block_size = 256;
    const dim3 block_dims(block_size, 1, 1);
    const dim3 grid_dims((n_tokens + block_size - 1) / block_size, 1, 1);
    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(grid_dims, block_dims, 0, ctx.stream());

    ggml_cuda_kernel_launch(dsv4_moe_weights_f32, launch_params,
            (const float *) probs->data, (const int32_t *) selected->data, (float *) dst->data,
            n_tokens,
            nbp0 / sizeof(float), nbp1 / sizeof(float),
            nbs0 / sizeof(int32_t), nbs1 / sizeof(int32_t),
            k, scale, normalize);
}
