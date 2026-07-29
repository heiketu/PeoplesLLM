#pragma once

// llama-remote-ep: master side of the expert-parallel (EP) MoE dispatch.
//
// When enabled via env, build_moe_ffn (src/llama-graph.cpp) ships the hidden
// states + router decisions of a MoE layer to a remote EPD worker
// (tools/epd/llama-epd) over the LEP1 protocol and blocks on the merged
// expert output, which is written back into the graph as a custom op.
//
// Configuration (all env, default = disabled, mirrors the GGML_NUMA_EP style):
//   GGML_REMOTE_EP=1            enable remote expert dispatch
//   GGML_REMOTE_EP_HOST=ADDR    worker host (default 127.0.0.1)
//   GGML_REMOTE_EP_PORT=N       worker port (default 29200)
//   GGML_REMOTE_EP_LAYERS=A-B   layer range to dispatch remotely (default: all)
//
// Only the routed-expert FFN is remote; attention, router and the weighted
// sum semantics are unchanged. Layers outside the range (and warmup graphs)
// keep the local path.

#include <cstdint>
#include <string>

struct ggml_tensor;

// true if remote EP is enabled and layer il is in the configured remote range
bool llama_remote_ep_enabled_for_layer(int il);

// blocking round-trip: sends ids/weights/hidden for layer il, fills out.
// layouts (all contiguous, column-major ggml order = token-major on the wire):
//   ids     [n_ids, n_tokens] i32   (ids[t*n_ids + k])
//   weights [1, n_ids, n_tokens] f32
//   hidden  [n_embd, 1, n_tokens] f32 (hidden[t*n_embd + e])
//   out     [n_embd, 1, n_tokens] f32
bool llama_remote_ep_moe_ffn(
        int             il,
        int64_t         n_tokens,
        int64_t         n_ids,
        int64_t         n_embd,
        const int32_t * ids,
        const float   * weights,
        const float   * hidden,
        float         * out,
        std::string   & err);

// ggml_custom3_op_t glue used by build_moe_ffn; userdata = (void *)(intptr_t) il
void llama_remote_ep_graph_cb(
        ggml_tensor       * dst,
        const ggml_tensor * a, // hidden  [n_embd, 1, n_tokens] f32
        const ggml_tensor * b, // ids     [n_ids, n_tokens] i32
        const ggml_tensor * c, // weights [1, n_ids, n_tokens] f32
        int ith, int nth, void * userdata);
