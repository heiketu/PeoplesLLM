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
//   GGML_REMOTE_EP_PIPELINE=1   pipelined chunk dispatch for multi-token batches:
//                               split a layer's tokens into chunks (default 256
//                               tokens, capped to ~3 MiB of hidden over TCP / 1.5 MiB
//                               over RDMA, tunable via GGML_REMOTE_EP_PIPELINE_CHUNK)
//                               and send them with a W=1 sliding window so the
//                               worker's compute overlaps the master's transfers.
//                               Off by default; decode (1 token) is unaffected.
//   GGML_REMOTE_EP_MIRROR=1     layer mirroring + expert-slot split: the master
//                               also loads the remote layers' expert weights and
//                               each MoE layer is split along the expert-slot
//                               dimension — slots [0,k_r) go to the worker
//                               (send-only op), slots [k_r,k) run locally on the
//                               master, then a wait op merges partial_r + local
//                               slots in ascending slot order (same association
//                               order as the all-remote baseline). Default off.
//   GGML_REMOTE_EP_MIRROR_LAYERS=A-B
//                               subset of the remote range to mirror (default:
//                               the whole GGML_REMOTE_EP_LAYERS range).
//   GGML_REMOTE_EP_MIRROR_KREMOTE=N
//                               slots sent remotely (default n_expert_used/2,
//                               clamped to [1, n_expert_used-1]).
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

// layer mirroring (GGML_REMOTE_EP_MIRROR=1):
//
// true if the loader should skip layer il's expert weights: the layer is
// dispatched remotely and is NOT mirrored (mirrored layers keep a local copy)
bool llama_remote_ep_skip_weights_for_layer(int il);

// number of expert slots dispatched remotely for a mirrored layer; 0 when
// mirroring does not apply to layer il (then the layer uses the plain path).
// otherwise the result k_r satisfies 1 <= k_r < n_expert_used
int llama_remote_ep_mirror_kremote(int il, int n_expert_used);

// mirror custom-op pair used by build_moe_ffn; userdata = (void *)(intptr_t) il.
// send: fires the REQ for slots [0,k_r) (send only, no wait); dst is unused.
// wait: blocks on the RESP (partial_r [n_embd,1,n_tokens] f32) and writes it
// into dst; a = the send op's output and b = local experts_l only create the
// ordering dependencies (send -> local chain -> wait). on transport failure
// the request is resent once from the pending slot, then GGML_ABORT.
//
// note: over RDMA the wait op is the first time the master drains the RESP,
// so mirroring is only applied when the RESP fits the pre-posted receive
// ring; larger batches fall back to the classic path (llama_remote_ep_mirror_fits)
void llama_remote_ep_mirror_send_cb(
        ggml_tensor       * dst,
        const ggml_tensor * a, // hidden  [n_embd, 1, n_tokens] f32 (contiguous)
        const ggml_tensor * b, // ids     [n_ids, n_tokens] i32 (contiguous)
        const ggml_tensor * c, // weights [1, n_ids, n_tokens] f32 (contiguous)
        int ith, int nth, void * userdata);

void llama_remote_ep_mirror_wait_cb(
        ggml_tensor       * dst, // partial_r [n_embd, 1, n_tokens] f32
        const ggml_tensor * a,   // send-op output (ordering dependency only)
        const ggml_tensor * b,   // experts_l (ordering dependency only)
        int ith, int nth, void * userdata);

// conservative in-flight limit for the mirror path: over RDMA the master only
// starts draining the RESP in the wait op (after the local chain), so the RESP
// must fit the pre-posted receive ring (8 x 256 KiB). TCP is unaffected
// (blocking send just applies backpressure). false => use the classic path
bool llama_remote_ep_mirror_fits(int64_t n_tokens, int64_t n_embd);
