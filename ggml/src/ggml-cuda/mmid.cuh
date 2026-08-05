#pragma once

void ggml_cuda_launch_mm_ids_helper(
        const int32_t * ids, int32_t * ids_src1, int32_t * ids_dst, int32_t * expert_bounds,
        int n_experts, int n_tokens, int n_expert_used, int nchannels_y, int si1, int sis1,
        int expert_id_offset, bool write_inverse, cudaStream_t stream);

void ggml_cuda_launch_mm_id_zero_invalid(
        const int32_t * ids, float * dst, int64_t nrows, int64_t n_expert_used, int64_t n_tokens,
        int64_t ids_stride_token, int64_t stride_slot, int64_t stride_token,
        int expert_id_offset, int expert_id_count, cudaStream_t stream);
