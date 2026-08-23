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
// waits on the backend event and adds the GPU partial. No scheduler splits,
// no host sync on the critical path beyond one event wait per layer.
//
// Numerics: the cold (CPU) partial is bit-identical to the baseline chain
// (skipped slots contribute exact +0.0f in the same ascending slot order).
// The hot partial is computed by CUDA kernels and is not bit-identical to the
// CPU repack kernels (different accumulation), same class of difference as
// running those experts with -ot on GPU.

#include <string>

struct ggml_tensor;

// env: GGML_HOT_EXPERT=1            master switch (default off)
//      GGML_HOT_EXPERT_TABLE        path to expert-hot TSV (required when on)
//      GGML_HOT_EXPERT_GGUF         path to the model GGUF (required when on)
//      GGML_HOT_EXPERT_K=16         experts pinned per layer
//      GGML_HOT_EXPERT_DEV=CUDA1    device that hosts the hot experts
//      GGML_HOT_EXPERT_LAYERS=all   comma list / ranges, e.g. "42" or "30-42"
//      GGML_HOT_EXPERT_MAX_TOKENS=1 max n_tokens for the GPU fork (tg path)

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

// userdata for the custom op callbacks of layer il (nullptr if inactive)
void * llama_hot_expert_userdata(int il);

// ggml custom-op callbacks (see llama-graph.cpp build_moe_ffn)
void llama_hot_expert_mask_ids_cb(struct ggml_tensor * dst, const struct ggml_tensor * ids, int ith, int nth, void * userdata);
void llama_hot_expert_send_cb(struct ggml_tensor * dst, const struct ggml_tensor * x, const struct ggml_tensor * ids, const struct ggml_tensor * w, int ith, int nth, void * userdata);
void llama_hot_expert_merge_cb(struct ggml_tensor * dst, const struct ggml_tensor * send, const struct ggml_tensor * cold, int ith, int nth, void * userdata);
