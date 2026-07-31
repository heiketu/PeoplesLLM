#pragma once
#include "ggml-backend-impl.h"
#include "ggml-cpu-impl.h"
#include "ggml.h"

#ifdef __cplusplus
#    include <vector>
extern "C" {
#endif

// return true if op part of extra "accelerator"
bool ggml_cpu_extra_compute_forward(struct ggml_compute_params * params, struct ggml_tensor * op);
bool ggml_cpu_extra_work_size(int n_threads, const struct ggml_tensor * op, size_t * size);

// NUMA expert parallelism: true when GGML_NUMA_EP=1 and more than one NUMA node is
// active; every expert's rows are split into per-node windows (node n owns rows
// [n*win, min((n+1)*win, ne01)), win aligned to 128 rows) — see repack.cpp
// forward_mul_mat_id and llama_model::numa_ep_place_experts
bool ggml_cpu_numa_ep_active(void);

// NUMA EP work stealing threshold in tokens (GGML_NUMA_EP_STEAL_MIN_TOKENS, default 32)
int ggml_cpu_numa_ep_steal_min_tokens(void);

// RMS_NORM(+MUL) absorption into mul_mat: decode-graph rms_norm nodes whose only
// consumers are matmuls are elided from the graph; the first consuming mul_mat
// materializes the norm result into src1->data in its prologue (see ggml-cpu.c).
struct ggml_cpu_absorb_entry {
    struct ggml_tensor       * dst;     // first consuming mul_mat dst (lookup key)
    struct ggml_tensor       * r_final; // norm output tensor; its data is written by the prologue
    const struct ggml_tensor * r_src;   // input of the rms_norm node
    const struct ggml_tensor * weight;  // optional rms_norm weight (NULL for plain rms_norm)
    float                      eps;
};

// lookup the absorb entry for a mul_mat dst (NULL when not absorbing)
const struct ggml_cpu_absorb_entry * ggml_cpu_absorb_find(const struct ggml_threadpool * tp, const struct ggml_tensor * dst);

// materialize the elided norm into r_final->data; blck is the block size of the
// quantization slicing used by the caller (sliced mode writes exactly that slice);
// allow_sliced=false forces the serial + barrier mode (for readers with full-row
// visibility ahead of the next barrier, e.g. the llamafile sgemm path)
void ggml_cpu_absorb_materialize(const struct ggml_compute_params * params, const struct ggml_cpu_absorb_entry * ab, int64_t blck, bool allow_sliced);

#ifdef __cplusplus
}

namespace ggml::cpu {
// register in tensor->extra
class tensor_traits {
  public:
    virtual ~tensor_traits();
    virtual bool work_size(int n_threads, const struct ggml_tensor * op, size_t & size)        = 0;
    virtual bool compute_forward(struct ggml_compute_params * params, struct ggml_tensor * op) = 0;
};

class extra_buffer_type {
  public:
    virtual ~extra_buffer_type();
    virtual bool            supports_op(ggml_backend_dev_t dev, const struct ggml_tensor * op) = 0;
    virtual tensor_traits * get_tensor_traits(const struct ggml_tensor * op)                   = 0;
};
}  // namespace ggml::cpu

// implemented in ggml-cpu.cpp.
std::vector<ggml_backend_buffer_type_t> & ggml_backend_cpu_get_extra_buffer_types();

#endif
