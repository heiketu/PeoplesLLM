#include "common.cuh"
#include "mmid.cuh"

// To reduce shared memory use, store "it" and "iex_used" with 22/10 bits each.
struct mm_ids_helper_store {
    uint32_t data;

    __device__ mm_ids_helper_store(const uint32_t it, const uint32_t iex_used) {
        data = (it & 0x003FFFFF) | (iex_used << 22);
    }

    __device__ uint32_t it() const {
        return data & 0x003FFFFF;
    }

    __device__ uint32_t iex_used() const {
        return data >> 22;
    }
};
static_assert(sizeof(mm_ids_helper_store) == 4, "unexpected size for mm_ids_helper_store");

// Helper function for mul_mat_id, converts ids to a more convenient format.
// ids_src1 describes how to permute the flattened column indices of src1 in order to get a compact src1 tensor sorted by expert.
// ids_dst describes the same mapping but for the dst tensor.
// The upper and lower bounds for the ith expert in the compact src1 tensor are stored in expert_bounds[i:i+1].
template <int n_expert_used_template>
__launch_bounds__(ggml_cuda_get_physical_warp_size(), 1)
static __global__ void mm_ids_helper(
        const int32_t * __restrict__ ids, int32_t * __restrict__ ids_src1, int32_t * __restrict__ ids_dst, int32_t * __restrict__ expert_bounds,
        const int n_tokens, const int n_expert_used_var, const int nchannels_y, const int si1, const int sis1,
        const int expert_id_offset, const bool write_inverse) {
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    const int n_expert_used = n_expert_used_template == 0 ? n_expert_used_var : n_expert_used_template;
    const int expert = blockIdx.x;
    const int expert_global = expert + expert_id_offset;

    extern __shared__ char data_mm_ids_helper[];
    mm_ids_helper_store * store = (mm_ids_helper_store *) data_mm_ids_helper;

    int nex_prev   = 0; // Number of columns for experts with a lower index.
    int it_compact = 0; // Running index for the compact slice of this expert.

    if constexpr (n_expert_used_template == 0) {
        // Generic implementation:
        for (int it = 0; it < n_tokens; ++it) {
            int iex_used = -1; // The index at which the expert is used, if any.
            for (int iex = threadIdx.x; iex < n_expert_used; iex += warp_size) {
                const int expert_used = ids[it*si1 + iex];
                nex_prev += expert_used >= expert_id_offset && expert_used < expert_global;
                if (expert_used == expert_global) {
                    iex_used = iex;
                }
            }

            if (iex_used != -1) {
                store[it_compact] = mm_ids_helper_store(it, iex_used);
            }

            if (warp_reduce_any<warp_size>(iex_used != -1)) {
                it_compact++;
            }
        }
    } else {
        // Implementation optimized for specific numbers of experts used:
        static_assert(n_expert_used == 6 || warp_size % n_expert_used == 0, "bad n_expert_used");
        const int neu_padded = n_expert_used == 6 ? 8 : n_expert_used; // Padded to next higher power of 2.
        for (int it0 = 0; it0 < n_tokens; it0 += warp_size/neu_padded) {
            const int it = it0 + threadIdx.x / neu_padded;

            const int iex = threadIdx.x % neu_padded; // The index at which the expert is used, if any.
            const int expert_used = (neu_padded == n_expert_used || iex < n_expert_used) && it < n_tokens ?
                ids[it*si1 + iex] : INT_MAX;
            const int iex_used = expert_used == expert_global ? iex : -1;
            nex_prev += expert_used >= expert_id_offset && expert_used < expert_global;

            // Whether the threads at this token position have used the expert:
            const int it_compact_add_self = warp_reduce_any<neu_padded>(iex_used != -1);

            // Do a scan over threads at lower token positions in warp to get the correct index for writing data:
            int it_compact_add_lower = 0;
#pragma unroll
            for (int offset = neu_padded; offset < warp_size; offset += neu_padded) {
                const int tmp = __shfl_up_sync(0xFFFFFFFF, it_compact_add_self, offset, warp_size);
                if (threadIdx.x >= static_cast<unsigned int>(offset)) {
                    it_compact_add_lower += tmp;
                }
            }

            if (iex_used != -1) {
                store[it_compact + it_compact_add_lower] = mm_ids_helper_store(it, iex_used);
            }

            // The thread with the highest index in the warp always has the sum over the whole warp, use it to increment all threads:
            it_compact += __shfl_sync(0xFFFFFFFF, it_compact_add_lower + it_compact_add_self, warp_size - 1, warp_size);
        }
    }
    nex_prev = warp_reduce_sum<warp_size>(nex_prev);

    for (int itc = threadIdx.x; itc < it_compact; itc += warp_size) {
        const mm_ids_helper_store store_it = store[itc];
        const int it       = store_it.it();
        const int iex_used = store_it.iex_used();
        ids_dst[nex_prev + itc] = it*n_expert_used + iex_used;
        // ids_src1 holds the forward map, or the inverse map (token slot -> compact row) for quant dedup
        if (write_inverse) {
            ids_src1[it*n_expert_used + iex_used] = nex_prev + itc;
        } else {
            ids_src1[nex_prev + itc] = it*sis1 + iex_used % nchannels_y;
        }
    }

    if (threadIdx.x != 0) {
        return;
    }

    expert_bounds[expert] = nex_prev;

    if (expert < static_cast<int>(gridDim.x) - 1) {
        return;
    }

    expert_bounds[gridDim.x] = nex_prev + it_compact;
}

template <int n_expert_used_template>
static void launch_mm_ids_helper(
        const int32_t * __restrict__ ids, int32_t * __restrict__ ids_src1, int32_t * __restrict__ ids_dst, int32_t * __restrict__ expert_bounds,
        const int n_experts, const int n_tokens, const int n_expert_used_var, const int nchannels_y, const int si1, const int sis1,
        const int expert_id_offset, const bool write_inverse, cudaStream_t stream) {
    GGML_ASSERT(n_tokens          < (1 << 22) && "too few bits in mm_ids_helper_store");
    GGML_ASSERT(n_expert_used_var < (1 << 10) && "too few bits in mm_ids_helper_store");

    const int id = ggml_cuda_get_device();
    const int warp_size = ggml_cuda_info().devices[id].warp_size;
    const size_t smpbo = ggml_cuda_info().devices[id].smpbo;
    CUDA_SET_SHARED_MEMORY_LIMIT(mm_ids_helper<n_expert_used_template>, smpbo);

    const dim3 num_blocks(n_experts, 1, 1);
    const dim3 block_size(warp_size, 1, 1);
    const size_t nbytes_shared = n_tokens*sizeof(mm_ids_helper_store);
    GGML_ASSERT(nbytes_shared <= smpbo);
    mm_ids_helper<n_expert_used_template><<<num_blocks, block_size, nbytes_shared, stream>>>
        (ids, ids_src1, ids_dst, expert_bounds, n_tokens, n_expert_used_var, nchannels_y, si1, sis1,
         expert_id_offset, write_inverse);
}

void ggml_cuda_launch_mm_ids_helper(
        const int32_t * __restrict__ ids, int32_t * __restrict__ ids_src1, int32_t * __restrict__ ids_dst, int32_t * __restrict__ expert_bounds,
        const int n_experts, const int n_tokens, const int n_expert_used, const int nchannels_y, const int si1, const int sis1,
        const int expert_id_offset, const bool write_inverse, cudaStream_t stream) {
    switch (n_expert_used) {
        case  2:
            launch_mm_ids_helper< 2>(ids, ids_src1, ids_dst, expert_bounds, n_experts, n_tokens, n_expert_used, nchannels_y, si1, sis1, expert_id_offset, write_inverse, stream);
            break;
        case  4:
            launch_mm_ids_helper< 4>(ids, ids_src1, ids_dst, expert_bounds, n_experts, n_tokens, n_expert_used, nchannels_y, si1, sis1, expert_id_offset, write_inverse, stream);
            break;
        case  6:
            launch_mm_ids_helper< 6>(ids, ids_src1, ids_dst, expert_bounds, n_experts, n_tokens, n_expert_used, nchannels_y, si1, sis1, expert_id_offset, write_inverse, stream);
            break;
        case  8:
            launch_mm_ids_helper< 8>(ids, ids_src1, ids_dst, expert_bounds, n_experts, n_tokens, n_expert_used, nchannels_y, si1, sis1, expert_id_offset, write_inverse, stream);
            break;
        case 16:
            launch_mm_ids_helper<16>(ids, ids_src1, ids_dst, expert_bounds, n_experts, n_tokens, n_expert_used, nchannels_y, si1, sis1, expert_id_offset, write_inverse, stream);
            break;
        case 32:
            launch_mm_ids_helper<32>(ids, ids_src1, ids_dst, expert_bounds, n_experts, n_tokens, n_expert_used, nchannels_y, si1, sis1, expert_id_offset, write_inverse, stream);
            break;
        default:
            launch_mm_ids_helper< 0>(ids, ids_src1, ids_dst, expert_bounds, n_experts, n_tokens, n_expert_used, nchannels_y, si1, sis1, expert_id_offset, write_inverse, stream);
            break;
    }
}

static __global__ void mm_id_zero_invalid(
        const int32_t * __restrict__ ids, float * __restrict__ dst, int64_t nrows, int64_t n_expert_used,
        int64_t nelements, int64_t ids_stride_token, int64_t stride_slot, int64_t stride_token,
        int expert_id_offset, int expert_id_count) {
    for (int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
            i < nelements; i += (int64_t) blockDim.x * gridDim.x) {
        const int64_t row = i % nrows;
        const int64_t slot_token = i / nrows;
        const int64_t slot = slot_token % n_expert_used;
        const int64_t token = slot_token / n_expert_used;
        const int32_t expert = ids[token * ids_stride_token + slot];
        if (expert < expert_id_offset || expert >= expert_id_offset + expert_id_count) {
            dst[token * stride_token + slot * stride_slot + row] = 0.0f;
        }
    }
}

void ggml_cuda_launch_mm_id_zero_invalid(
        const int32_t * ids, float * dst, int64_t nrows, int64_t n_expert_used, int64_t n_tokens,
        int64_t ids_stride_token, int64_t stride_slot, int64_t stride_token,
        int expert_id_offset, int expert_id_count, cudaStream_t stream) {
    const int64_t nelements = nrows * n_expert_used * n_tokens;
    const int block_size = 256;
    const int num_blocks = (int) std::min<int64_t>((nelements + block_size - 1) / block_size, 65535);
    mm_id_zero_invalid<<<num_blocks, block_size, 0, stream>>>(
        ids, dst, nrows, n_expert_used, nelements, ids_stride_token, stride_slot, stride_token,
        expert_id_offset, expert_id_count);
}

// Weighted slot reduction of expert-sliced MUL_MAT_ID outputs. For each
// (row, token) accumulate the n_expert_used slots in ascending slot order,
// keeping only slots whose expert id lies in the rank's expert range. Slots
// outside the range contribute nothing and are never read.
static __global__ void moe_wreduce(
        const float * __restrict__ experts, const float * __restrict__ weights, const int32_t * __restrict__ ids,
        float * __restrict__ dst, int64_t nrows, int64_t n_expert_used, int64_t n_tokens,
        int64_t experts_stride_slot, int64_t experts_stride_token,
        int64_t weights_stride_slot, int64_t weights_stride_token,
        int64_t ids_stride_token, int64_t dst_stride_token,
        int expert_id_offset, int expert_id_count) {
    const int64_t nelements = nrows * n_tokens;
    for (int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
            i < nelements; i += (int64_t) blockDim.x * gridDim.x) {
        const int64_t row   = i % nrows;
        const int64_t token = i / nrows;

        const float * __restrict__ experts_row = experts + row + token * experts_stride_token;
        const float * __restrict__ weights_tok = weights + token * weights_stride_token;
        const int32_t * __restrict__ ids_tok   = ids     + token * ids_stride_token;

        float acc = 0.0f;
        for (int64_t slot = 0; slot < n_expert_used; ++slot) {
            const int32_t expert = ids_tok[slot];
            if (expert >= expert_id_offset && expert < expert_id_offset + expert_id_count) {
                acc += experts_row[slot * experts_stride_slot] * weights_tok[slot * weights_stride_slot];
            }
        }
        dst[token * dst_stride_token + row] = acc;
    }
}

void ggml_cuda_moe_wreduce(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * experts = dst->src[0];
    const ggml_tensor * weights = dst->src[1];
    const ggml_tensor * ids     = dst->src[2];

    GGML_ASSERT(experts->type == GGML_TYPE_F32);
    GGML_ASSERT(weights->type == GGML_TYPE_F32);
    GGML_ASSERT(ids->type     == GGML_TYPE_I32);
    GGML_ASSERT(experts->nb[0] == sizeof(float));
    GGML_ASSERT(weights->nb[0] == sizeof(float));
    GGML_ASSERT(ids->nb[0]     == sizeof(int32_t));
    GGML_ASSERT(dst->nb[0]     == sizeof(float));

    const int64_t nrows         = experts->ne[0];
    const int64_t n_expert_used = experts->ne[1];
    const int64_t n_tokens      = experts->ne[2];

    const int expert_id_offset = ggml_get_op_params_i32(dst, 0);
    const int expert_id_count  = ggml_get_op_params_i32(dst, 1);

    const size_t weights_stride_token = weights->ne[2] == 1 ? weights->nb[3] : weights->nb[2];

    cudaStream_t stream = ctx.stream();

    const int64_t nelements = nrows * n_tokens;
    const int block_size = 256;
    const int num_blocks = (int) std::min<int64_t>((nelements + block_size - 1) / block_size, 65535);
    moe_wreduce<<<num_blocks, block_size, 0, stream>>>(
        (const float *) experts->data, (const float *) weights->data, (const int32_t *) ids->data,
        (float *) dst->data, nrows, n_expert_used, n_tokens,
        experts->nb[1] / sizeof(float), experts->nb[2] / sizeof(float),
        weights->nb[1] / sizeof(float), weights_stride_token / sizeof(float),
        ids->nb[1] / sizeof(int32_t), dst->nb[1] / sizeof(float),
        expert_id_offset, expert_id_count);
    CUDA_CHECK(cudaGetLastError());
}
