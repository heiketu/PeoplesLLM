#pragma once

// Slice 12: hot-expert GPU offload (GGML_HOT_EXPERT=1, default off).
//
// A per-layer top-K expert set (from GGML_MOE_HOT_STATS TSV) is kept resident
// on a GPU device as compact MXFP4 tensors. During decode (n_tokens small) the
// MoE block forks inside the graph: a custom op remaps the router ids into the
// compact slot space (cold slots are remapped to slot 0 with their routing
// weight zeroed) and launches the hot expert FFN on the GPU asynchronously;
// the CPU chain computes only the cold slots (sentinel ids are skipped by the
// repack mmid, so hot expert weights are never read from DRAM); a merge op
// waits on the backend event and combines the hot/cold slots. No scheduler
// splits, no host sync on the critical path beyond one event wait per layer.
//
// Numerics: the default slot-order merge restores the baseline router-slot
// reduction order. Hot expert FFNs are still not bit-identical to the CPU
// repack kernels, the same class of difference as -ot GPU execution.

#include <cstddef>
#include <cstdint>
#include <string>

struct ggml_tensor;

// env: GGML_HOT_EXPERT=1            master switch (default off)
//      GGML_HOT_EXPERT_TABLE        path to expert-hot TSV (required when on)
//      GGML_HOT_EXPERT_GGUF         path to the model GGUF (required when on)
//      GGML_HOT_EXPERT_K=16         experts pinned per layer
//      GGML_HOT_EXPERT_DEV=CUDA1    device that hosts the hot experts
//      GGML_HOT_EXPERT_LAYERS=all   comma list / ranges, e.g. "42" or "30-42"
//      GGML_HOT_EXPERT_MAX_TOKENS=1 max n_tokens for the GPU fork (1-4)
//      GGML_HOT_EXPERT_PACKED_IO=1  coalesce x/ids/weights into one H2D (default on)
//      GGML_HOT_EXPERT_SLOT_ORDER=1 return per-slot GPU output and restore the baseline fold (default on)
//      GGML_HOT_EXPERT_SLOT_MERGE_AVX512=1 use AVX512 strict slot merge when supported (default on)
//      GGML_HOT_EXPERT_REMOTE_EP=1  let strict pure remote EP dispatch only cold slots (default off)
//      GGML_HOT_EXPERT_REMOTE_EP_SHADOW=1 also compute hot slots on CPU for vector comparison (default off)
//      GGML_HOT_EXPERT_SHADOW_FILE   optional per-layer/expert comparison CSV
//      GGML_CUDA_HOT_MXFP4_CPU_Q8=1  match CPU Q8_0 activation codes in hot MXFP4 MMVQ (default off)

bool llama_hot_expert_enabled();

// release all GPU/host resources (called from llama_model teardown, before the
// ggml backend registry is torn down); no-op when the feature is off
void llama_hot_expert_shutdown();

// call once after model weights are loaded; returns true if the offload is active
bool llama_hot_expert_init(const float * swiglu_clamp_exp, int n_layer, int64_t n_expert_used);

// layer offload active (init succeeded and layer selected)
bool llama_hot_expert_layer_active(int il);

// max n_tokens handled by the GPU fork (GGML_HOT_EXPERT_MAX_TOKENS)
int64_t llama_hot_expert_max_tokens();

// use the strict router-slot merge; false selects the legacy two-partial merge
bool llama_hot_expert_slot_order_enabled();

// opt-in bridge for small-batch strict pure remote EP
bool llama_hot_expert_remote_ep_enabled();
bool llama_hot_expert_remote_ep_shadow_enabled();
bool llama_hot_expert_upe_verify_cpu_enabled();
bool llama_hot_expert_markers_enabled();

// userdata for the custom op callbacks of layer il (nullptr if inactive)
void * llama_hot_expert_userdata(int il);
void llama_hot_expert_remote_ep_cancel(void * userdata);

// ggml custom-op callbacks (see llama-graph.cpp build_moe_ffn)
void llama_hot_expert_mask_ids_cb(struct ggml_tensor * dst, const struct ggml_tensor * ids, int ith, int nth, void * userdata);
void llama_hot_expert_send_cb(struct ggml_tensor * dst, const struct ggml_tensor * x, const struct ggml_tensor * ids, const struct ggml_tensor * w, int ith, int nth, void * userdata);
void llama_hot_expert_merge_cb(struct ggml_tensor * dst, const struct ggml_tensor * send, const struct ggml_tensor * cold, int ith, int nth, void * userdata);

// Called after send_cb. Copies the runtime slot split used by the GPU graph.
// cold_active is 1 for slots that remote EP must dispatch.
bool llama_hot_expert_remote_ep_split(
        void * userdata, uint8_t * cold_active, uint8_t * hot_mask, int64_t n_tokens, int64_t n_slots);

// Waits for the GPU branch and left-folds hot slots with already-weighted
// remote cold slots. cold has one n_embd-stride vector per router slot.
void llama_hot_expert_remote_ep_merge(
        void * userdata, float * dst, const float * const * cold, const float * weights,
        const uint8_t * cpu_override,
        int64_t n_embd, int64_t n_tokens, int64_t n_slots);

// Strict merge for contiguous [n_tokens,n_slots,n_embd] buffers.
void llama_hot_expert_merge_tokens_f32(
        float * dst, const float * cold, const float * hot, const float * weights,
        const uint8_t * hot_mask, int64_t n_embd, int64_t n_tokens, int64_t n_slots);

// Strict merge for raw remote slot pointers and contiguous GPU hot slots.
void llama_hot_expert_merge_raw_tokens_f32(
        float * dst, const float * const * cold, const float * hot, const float * weights,
        const uint8_t * hot_mask, int64_t n_embd, int64_t n_tokens, int64_t n_slots);

// Strict F32 slot fold shared by the callback and its model-independent test.
// Strides are in float elements. Cold is already weighted; hot is multiplied
// by weights before each add.
void llama_hot_expert_merge_slots_f32(
        float * dst, const float * cold, size_t cold_slot_stride,
        const float * hot, size_t hot_slot_stride, const float * weights,
        const uint8_t * hot_mask, int64_t n_embd, int64_t n_slots);

// Explicit paths for bit-exact tests and microbenchmarks. The AVX512 function
// returns false without writing dst when the current build/CPU cannot run it.
void llama_hot_expert_merge_slots_f32_scalar(
        float * dst, const float * cold, size_t cold_slot_stride,
        const float * hot, size_t hot_slot_stride, const float * weights,
        const uint8_t * hot_mask, int64_t n_embd, int64_t n_slots);
bool llama_hot_expert_merge_slots_f32_avx512(
        float * dst, const float * cold, size_t cold_slot_stride,
        const float * hot, size_t hot_slot_stride, const float * weights,
        const uint8_t * hot_mask, int64_t n_embd, int64_t n_slots);
bool llama_hot_expert_slot_merge_avx512_supported();
bool llama_hot_expert_slot_merge_avx512_enabled();
