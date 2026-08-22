#include "argsort.cuh"
#include "top-k.cuh"

#ifdef GGML_CUDA_USE_CUB
#    include <cub/cub.cuh>
#    if (CCCL_MAJOR_VERSION >= 3 && CCCL_MINOR_VERSION >= 2)
#        define CUB_TOP_K_AVAILABLE
#        include <cuda/iterator>
using namespace cub;
#    endif  // CCCL_MAJOR_VERSION >= 3 && CCCL_MINOR_VERSION >= 2
#endif      // GGML_CUDA_USE_CUB

#ifdef CUB_TOP_K_AVAILABLE

static void top_k_cub(ggml_cuda_pool & pool,
                      const float *    src,
                      int *            dst,
                      const int        ncols,
                      const int        k,
                      cudaStream_t     stream) {
    auto requirements = cuda::execution::require(cuda::execution::determinism::not_guaranteed,
                                                 cuda::execution::output_ordering::unsorted);
    auto stream_env   = cuda::stream_ref{ stream };
    auto env          = cuda::std::execution::env{ stream_env, requirements };

    auto indexes_in = cuda::make_counting_iterator(0);

    size_t temp_storage_bytes = 0;
    CUDA_CHECK(DeviceTopK::MaxPairs(nullptr, temp_storage_bytes, src, cuda::discard_iterator(), indexes_in, dst, ncols, k,
                         env));

    ggml_cuda_pool_alloc<uint8_t> temp_storage_alloc(pool, temp_storage_bytes);
    void *                        d_temp_storage = temp_storage_alloc.get();

    CUDA_CHECK(DeviceTopK::MaxPairs(d_temp_storage, temp_storage_bytes, src, cuda::discard_iterator(), indexes_in, dst,
                         ncols, k, env));
}

#elif defined(GGML_CUDA_USE_CUB)  // CUB_TOP_K_AVAILABLE

static int next_power_of_2(int x) {
    int n = 1;
    while (n < x) {
        n *= 2;
    }
    return n;
}

#endif                            // CUB_TOP_K_AVAILABLE

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)

static __device__ __forceinline__ uint32_t top_k_float_key(const float value) {
    if (isnan(value)) {
        return 0;
    }

    const uint32_t bits = __float_as_uint(value);
    return bits ^ ((bits & 0x80000000U) ? 0xffffffffU : 0x80000000U);
}

// Selects the k largest values of each row without allocating temporary
// storage. Output ordering is unspecified, matching the CUB DeviceTopK path.
// The eight radix passes find the k-th largest key; two final passes emit all
// strictly larger keys and enough equal keys to fill the result.
static __launch_bounds__(256) __global__ void top_k_batched_radix_f32_i32(
        const float * src, int * dst, const int ncols, const int k) {
    constexpr int block_size = 256;
    constexpr int n_warps    = block_size / WARP_SIZE;

    __shared__ uint32_t hist[n_warps][16];
    __shared__ uint32_t prefix;
    __shared__ uint32_t prefix_mask;
    __shared__ uint32_t rank;
    __shared__ uint32_t threshold;
    __shared__ int      out_count;
    __shared__ int      equal_base;

    const int row = blockIdx.x;
    src += (size_t) row * ncols;
    dst += (size_t) row * k;

    if (threadIdx.x == 0) {
        prefix      = 0;
        prefix_mask = 0;
        rank        = k;
    }
    __syncthreads();

#pragma unroll
    for (int shift = 28; shift >= 0; shift -= 4) {
        for (int i = threadIdx.x; i < n_warps * 16; i += blockDim.x) {
            hist[i / 16][i % 16] = 0;
        }
        __syncthreads();

        const int warp = threadIdx.x / WARP_SIZE;
        for (int col = threadIdx.x; col < ncols; col += blockDim.x) {
            const uint32_t key = top_k_float_key(src[col]);
            if ((key & prefix_mask) == prefix) {
                atomicAdd(&hist[warp][(key >> shift) & 0x0fU], 1U);
            }
        }
        __syncthreads();

        if (threadIdx.x == 0) {
            uint32_t remaining = rank;
            for (int bucket = 15; bucket >= 0; --bucket) {
                uint32_t count = 0;
#pragma unroll
                for (int w = 0; w < n_warps; ++w) {
                    count += hist[w][bucket];
                }
                if (remaining > count) {
                    remaining -= count;
                } else {
                    prefix      |= (uint32_t) bucket << shift;
                    prefix_mask |= 0x0fU << shift;
                    rank         = remaining;
                    break;
                }
            }
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        threshold = prefix;
        out_count = 0;
    }
    __syncthreads();

    for (int col = threadIdx.x; col < ncols; col += blockDim.x) {
        if (top_k_float_key(src[col]) > threshold) {
            const int slot = atomicAdd(&out_count, 1);
            if (slot < k) {
                dst[slot] = col;
            }
        }
    }
    __syncthreads();

    // Resolve boundary ties deterministically by column index. CUB's unsorted
    // DeviceTopK may choose a different subset of equal values from run to run;
    // a block-wide prefix per 256-column tile keeps the selected mask stable.
    if (threadIdx.x == 0) {
        equal_base = out_count;
    }
    __syncthreads();

    const int warp = threadIdx.x / WARP_SIZE;
    const int lane = threadIdx.x % WARP_SIZE;
    for (int col_base = 0; col_base < ncols; col_base += blockDim.x) {
        const int col = col_base + threadIdx.x;
        const bool equal = col < ncols && top_k_float_key(src[col]) == threshold;
        const uint32_t equal_mask = __ballot_sync(0xffffffffU, equal);

        if (lane == 0) {
            hist[warp][0] = __popc(equal_mask);
        }
        __syncthreads();

        if (threadIdx.x == 0) {
            int next = equal_base;
#pragma unroll
            for (int w = 0; w < n_warps; ++w) {
                const int count = hist[w][0];
                hist[w][1] = next;
                next += count;
            }
            equal_base = next;
        }
        __syncthreads();

        if (equal) {
            const uint32_t lane_mask = lane == 0 ? 0U : ((1U << lane) - 1U);
            const int slot = hist[warp][1] + __popc(equal_mask & lane_mask);
            if (slot < k) {
                dst[slot] = col;
            }
        }
        __syncthreads();

        if (equal_base >= k) {
            break;
        }
    }
}

static bool top_k_use_batched_radix(const int64_t ncols, const int64_t nrows, const int64_t k) {
    static const bool enabled = []() {
        const char * value = getenv("GGML_CUDA_BATCHED_TOPK");
        return value == nullptr || std::atoi(value) != 0; // default ON (k==512 gate below still applies)
    }();

    // small-row case (e.g. DSV4 lightning indexer at tg: nrows==1, ncols==context
    // length, k==min(ncols, 512)): a single radix block per row beats the
    // multi-launch CUB DeviceTopK loop as long as the per-row scan stays short.
    const bool small_rows = nrows < 32 && k <= 1024 && ncols <= 8192;

    return enabled && ncols >= k && ncols <= INT_MAX && nrows <= INT_MAX &&
           ((k == 512 && nrows >= 32) || small_rows);
}

#endif // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)

static __global__ void top_k_sort_indices_i32(int * dst, const int k, const int k_pad) {
    extern __shared__ int values[];
    int * row = dst + (size_t) blockIdx.x*k;

    for (int i = threadIdx.x; i < k_pad; i += blockDim.x) {
        values[i] = i < k ? row[i] : INT_MAX;
    }
    __syncthreads();

    for (int size = 2; size <= k_pad; size *= 2) {
        for (int stride = size/2; stride > 0; stride /= 2) {
            for (int i = threadIdx.x; i < k_pad; i += blockDim.x) {
                const int peer = i ^ stride;
                if (peer > i) {
                    const bool ascending = (i & size) == 0;
                    const int a = values[i];
                    const int b = values[peer];
                    if ((a > b) == ascending) {
                        values[i] = b;
                        values[peer] = a;
                    }
                }
            }
            __syncthreads();
        }
    }

    for (int i = threadIdx.x; i < k; i += blockDim.x) {
        row[i] = values[i];
    }
}

static void top_k_sort_indices(int * dst, const int64_t nrows, const int64_t k, cudaStream_t stream) {
    GGML_ASSERT(k > 0 && k <= 1024);
    int k_pad = 1;
    while (k_pad < k) {
        k_pad *= 2;
    }
    top_k_sort_indices_i32<<<nrows, 256, k_pad*sizeof(int), stream>>>(dst, k, k_pad);
    CUDA_CHECK(cudaGetLastError());
}

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)

// Diagnostic only: sample the first, middle and last groups of adjacent query
// rows and report how many distinct physical KV rows their sorted top-k sets
// contain. A high reuse factor justifies a query-group sparse attention kernel;
// the default path never launches this kernel.
static __global__ void top_k_profile_group_overlap_i32(
        const int * dst, const int nrows, const int ncols, const int k, const int group_rows) {
    extern __shared__ uint32_t selected[];
    __shared__ int n_union;

    const int nwords = (ncols + 31)/32;
    const int sample = blockIdx.x;
    const int max_row0 = max(0, nrows - group_rows);
    const int row0 = sample == 0 ? 0 :
        (sample == 1 ? min(max_row0, (nrows/2/group_rows)*group_rows) : max_row0);
    const int rows = min(group_rows, nrows - row0);

    for (int i = threadIdx.x; i < nwords; i += blockDim.x) {
        selected[i] = 0;
    }
    if (threadIdx.x == 0) {
        n_union = 0;
    }
    __syncthreads();

    for (int linear = threadIdx.x; linear < rows*k; linear += blockDim.x) {
        const int row = row0 + linear/k;
        const int col = dst[(size_t) row*k + linear%k];
        if (col >= 0 && col < ncols) {
            atomicOr(&selected[col/32], 1U << (col%32));
        }
    }
    __syncthreads();

    int local = 0;
    for (int i = threadIdx.x; i < nwords; i += blockDim.x) {
        local += __popc(selected[i]);
    }
    if (local != 0) {
        atomicAdd(&n_union, local);
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        printf("[topk-overlap] rows=%d cols=%d k=%d sample=%d row0=%d group=%d union=%d reuse=%.3f\n",
            nrows, ncols, k, sample, row0, rows, n_union,
            n_union > 0 ? double(rows*k)/double(n_union) : 0.0);
    }
}

static bool top_k_profile_group_overlap_enabled() {
    static const bool enabled = []() {
        const char * value = getenv("GGML_CUDA_TOPK_OVERLAP_PROFILE");
        return value != nullptr && std::atoi(value) != 0;
    }();
    return enabled;
}

static void top_k_profile_group_overlap(
        const int * dst, const int64_t nrows, const int64_t ncols, const int64_t k, cudaStream_t stream) {
    constexpr int group_rows = 8;
    if (!top_k_profile_group_overlap_enabled() || nrows < group_rows ||
            ncols < k || ncols > 65536 || k > INT_MAX) {
        return;
    }

    const size_t shared_bytes = ((ncols + 31)/32)*sizeof(uint32_t);
    top_k_profile_group_overlap_i32<<<3, 256, shared_bytes, stream>>>(
        dst, nrows, ncols, k, group_rows);
    CUDA_CHECK(cudaGetLastError());
}

#endif // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)

void ggml_cuda_op_top_k(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0   = dst->src[0];
    const float *       src0_d = (const float *) src0->data;
    int *               dst_d  = (int *) dst->data;
    int *               dst_base = dst_d;
    cudaStream_t        stream = ctx.stream();

    // are these asserts truly necessary?
    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_I32);
    GGML_ASSERT(ggml_is_contiguous(src0));

    const int64_t    ncols = src0->ne[0];
    const int64_t    nrows = ggml_nrows(src0);
    const int64_t    k     = dst->ne[0];
    ggml_cuda_pool & pool  = ctx.pool();
#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
    if (top_k_use_batched_radix(ncols, nrows, k)) {
        top_k_batched_radix_f32_i32<<<nrows, 256, 0, stream>>>(src0_d, dst_d, ncols, k);
        CUDA_CHECK(cudaGetLastError());
        if (ggml_get_op_params_i32(dst, 0) != 0) {
            top_k_sort_indices(dst_base, nrows, k, stream);
        }
#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
        top_k_profile_group_overlap(dst_base, nrows, ncols, k, stream);
#endif
        return;
    }
#endif // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
#ifdef CUB_TOP_K_AVAILABLE
    // TODO: Switch to `DeviceSegmentedTopK` for multi-row TopK once implemented
    // https://github.com/NVIDIA/cccl/issues/6391
    // TODO: investigate if there exists a point where parallelized argsort is faster than sequential top-k
    for (int i = 0; i < nrows; i++) {
        top_k_cub(pool, src0_d + i * ncols, dst_d + i * k, ncols, k, stream);
    }
#elif defined(GGML_CUDA_USE_CUB)  // CUB_TOP_K_AVAILABLE
    // Fall back to argsort + copy
    const int    ncols_pad      = next_power_of_2(ncols);
    const size_t shared_mem     = ncols_pad * sizeof(int);
    const size_t max_shared_mem = ggml_cuda_info().devices[ggml_cuda_get_device()].smpb;
    const bool   use_bitonic    = shared_mem <= max_shared_mem && ncols <= 1024;
    const int    chunk_nrows    = argsort_f32_i32_cuda_cub_chunk_nrows(src0->nb[1], nrows);

    ggml_cuda_pool_alloc<int> temp_dst_alloc(pool, ncols * chunk_nrows);
    int *                     tmp_dst = temp_dst_alloc.get();

    for (int64_t i = 0; i < nrows; i += chunk_nrows) {
        int iter_nrows = std::min((int64_t) chunk_nrows, nrows - i);

        if (use_bitonic) {
            argsort_f32_i32_cuda_bitonic(src0_d, tmp_dst, ncols, iter_nrows, GGML_SORT_ORDER_DESC, stream);
        } else {
            argsort_f32_i32_cuda_cub(pool, src0_d, tmp_dst, ncols, iter_nrows, GGML_SORT_ORDER_DESC, stream);
        }
        CUDA_CHECK(cudaMemcpy2DAsync(dst_d, k * sizeof(int), tmp_dst, ncols * sizeof(int), k * sizeof(int), iter_nrows,
                                     cudaMemcpyDeviceToDevice, stream));

        src0_d += ncols * iter_nrows;
        dst_d  += k     * iter_nrows;
    }
#else                             // GGML_CUDA_USE_CUB
    ggml_cuda_pool_alloc<int> temp_dst_alloc(pool, ncols * nrows);
    int *                     tmp_dst = temp_dst_alloc.get();
    argsort_f32_i32_cuda_bitonic(src0_d, tmp_dst, ncols, nrows, GGML_SORT_ORDER_DESC, stream);
    CUDA_CHECK(cudaMemcpy2DAsync(dst_d, k * sizeof(int), tmp_dst, ncols * sizeof(int), k * sizeof(int), nrows,
                                 cudaMemcpyDeviceToDevice, stream));
#endif

    if (ggml_get_op_params_i32(dst, 0) != 0) {
        top_k_sort_indices(dst_base, nrows, k, stream);
    }
#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
    top_k_profile_group_overlap(dst_base, nrows, ncols, k, stream);
#endif
}
