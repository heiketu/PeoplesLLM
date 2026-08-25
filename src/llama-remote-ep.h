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
//   GGML_REMOTE_EP_SCHED=1      expert-level dynamic scheduling (SCHEDULER-
//                               DESIGN.md): the router's top-k slots are dealt
//                               per token between the master (m* local slots,
//                               GGML_REMOTE_EP_SCHED_KLOCAL, default 2) and the
//                               slave endpoints of GGML_REMOTE_EP_SCHED_ENDPOINTS
//                               ("host:port,host:port"; default HOST:PORT) via
//                               REQ2; a merge op accumulates all per-slot vectors
//                               in ascending global slot order — bit-identical to
//                               the local baseline. mutually exclusive with
//                               MIRROR (SCHED wins). With KLOCAL>0, a failed CAP
//                               handshake can use the retained local expert path.
//                               KLOCAL=0 is strict pure EP: the
//                               master skips routed-expert weights and endpoint
//                               ranges must form an exact, non-overlapping cover
//                               of every expert; negotiation failure is fatal for
//                               those layers. decode-only unless
//                               GGML_REMOTE_EP_SCHED_PP=1. in SCHED mode the
//                               master keeps expert weights only when KLOCAL>0.
//   GGML_REMOTE_EP_SCHED_MAX_EFFORT=1
//                               with KLOCAL=0, permit endpoint expert ranges to
//                               overlap (they must still fully cover all experts).
//                               The dealer dynamically sends replicated experts
//                               to the least-loaded eligible worker per token.
//   GGML_REMOTE_EP_PIPE=1       async pipelined dispatch (protocol v3): implies
//                               SCHED, but requests travel as REQ3/RESP3 with a
//                               master-assigned req_id and a background receiver
//                               thread per endpoint collects completions into a
//                               registry; the merge op waits on the registry
//                               instead of draining the socket. multi-token
//                               batches are always allowed. foundation for the
//                               cross-slot pipeline scheduler (several requests
//                               may be in flight at once). requires a worker with
//                               the REQ3 capability (CAP). A missing capability is
//                               fatal in pure KLOCAL=0 mode.
//   GGML_REMOTE_EP_PIPE_MAX_MIB=N
//                               aggregate hidden/request/response staging credit
//                               across in-flight streams (default 512 MiB).
//   GGML_REMOTE_EP_PIPE_MAX_REQUESTS=N
//                               aggregate endpoint request credit (default 256).
//   GGML_REMOTE_EP_FREQ=1       expert activation frequency accounting: counts
//                               router selections per (layer, expert) in every
//                               SCHED/PIPE-dispatched graph and dumps a per-layer
//                               summary (tokens/active/min/max/max-over-mean) at
//                               process exit. with GGML_REMOTE_EP_FREQ_FILE=path
//                               the full (layer,expert,count) CSV goes to path.
//   GGML_REMOTE_EP_RECONNECT_TIMEOUT_MS=N
//                               after a scheduled endpoint transport failure,
//                               wait up to N ms for its worker to restart before
//                               the final fatal error (default 0, max 300000).
//                               The staged request is resent after reconnect.
//   GGML_REMOTE_EP_UPE_STRICT=1
//                               require every scheduled worker to return the
//                               negotiated Unified Precision Engine contract;
//                               reject unknown/mismatched data epochs and any
//                               contract change on reconnect. Default off for
//                               compatibility with old workers.
//   GGML_EP_DATA_EPOCH=STRING   immutable deployment/model version shared by
//                               master and every worker. It is hashed on the
//                               wire; strict UPE requires a non-empty value.
//   GGML_REMOTE_EP_UPE_ACTIVATION_TRACE=1|2
//                               default-off diagnostic: inspect each real F32
//                               expert input in 32-value blocks and count CPU
//                               versus nominal-CUDA code differences plus the
//                               distance to half-step quantization boundaries;
//                               1 samples only boundary blocks, 2 records all
//                               blocks for direct CUDA replay.
//   GGML_REMOTE_EP_UPE_ACTIVATION_TRACE_FILE=PATH
//                               write the per-layer CSV at process exit.
//   GGML_HOT_EXPERT_UPE_BOUNDARY_FALLBACK=1
//                               experimental verify-strict policy for the
//                               GPU-hot + CPU-remote bridge: if a token has an
//                               activation within the configured half-step
//                               distance, recompute its hot slots remotely and
//                               merge the CPU result instead of the GPU result.
//   GGML_HOT_EXPERT_UPE_BOUNDARY_THRESHOLD=FLOAT
//                               raw/TG normalized Q8 half-step distance (default
//                               1e-6). Nominal CPU/CUDA code disagreement always
//                               triggers independently of this distance.
//   GGML_HOT_EXPERT_UPE_VERIFY_THRESHOLD=FLOAT
//                               multi-token verify distance (default 0: only
//                               nominal CPU/CUDA code disagreement triggers).
//                               1.6e-5 covers all current trace mismatches but
//                               its remote fanout cost failed the first A/B.
//   GGML_HOT_EXPERT_UPE_VERIFY_CPU=1
//                               phase selector: for n_tokens>1, skip GPU-hot
//                               submission entirely and use ordinary CPU remote
//                               EP; single-token raw/TG remains GPU-hot. This
//                               avoids the duplicate GPU work of boundary fallback.
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
// dispatched remotely without a local expert path (classic remote EP, or strict
// scheduled pure EP with GGML_REMOTE_EP_SCHED_KLOCAL=0).
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

// expert-level dynamic scheduling (GGML_REMOTE_EP_SCHED=1):
//
// true when layer il is configured for strict scheduled pure EP (KLOCAL=0).
bool llama_remote_ep_sched_pure_for_layer(int il);

// true when the scheduled remote path can combine one-token CUDA hot slots
// with strict KLOCAL=0 CPU workers. The first implementation requires sync
// REQ4 so both domains are weighted and folded on the master.
bool llama_remote_ep_sched_hot_compatible(int il);

// number of expert slots computed locally per token for a scheduled layer.
// 0 is a valid strict-pure-EP result; -1 means scheduling does not apply or CAP /
// topology negotiation failed. The first call negotiates protocol v2 with every
// endpoint. In strict pure mode their ranges must exactly cover [0,n_expert);
// MAX_EFFORT permits overlap but still requires complete coverage. At most 63
// worker endpoints are supported by the holder bitmask.
int llama_remote_ep_sched_klocal(int il, int n_expert_used, int n_expert);

// gate on the batch shape: scheduling is decode-only unless
// GGML_REMOTE_EP_SCHED_PP=1; over RDMA the RESP2 (one vector per slot) must
// fit the pre-posted receive ring (same reasoning as llama_remote_ep_mirror_fits)
bool llama_remote_ep_sched_fits(int64_t n_tokens, int64_t n_ids, int64_t n_embd);

// scheduled custom-op pair used by build_moe_ffn (ggml_custom_4d, srcs in
// dst->src); userdata = llama_remote_ep_userdata(stream, il).
// send: dst = local ids [max(1,m*), n_tokens] i32 (written by the dealer when
// m*>0; the row is only a dependency token when m*=0), srcs =
// {hidden, ids, weights}; fires one REQ2 per endpoint (send only, no wait)
// and stashes the plan + request bytes in the pending slot for the merge op.
// merge: dst = moe_out [n_embd,1,n_tokens] f32, srcs = {send out, experts_l,
// weights}, or {send out, weights} when m*=0; receives the RESP2s in fixed
// endpoint order, then accumulates all
// contributions in ascending global slot order (local slots are multiplied by
// their router weight here — the same scalar multiply the baseline does in
// ggml_mul, keeping the merge bit-identical). on transport failure the
// request is resent from the pending slot after reconnect.  With
// GGML_REMOTE_EP_RECONNECT_TIMEOUT_MS>0 the reconnect waits for a restarting
// worker up to that deadline; exhaustion remains fatal because pure EP has no
// local expert result with which to finish the layer.
void llama_remote_ep_sched_send_cb(ggml_tensor * dst, int ith, int nth, void * userdata);
void llama_remote_ep_sched_merge_cb(ggml_tensor * dst, int ith, int nth, void * userdata);

// stream ids let several llama_contexts compute concurrently through the
// dispatcher (cross-slot pipeline): each context claims one id at init and
// encodes it into the op userdata together with the layer index.
// returns 0 for every caller when the pipe mode is off (single shared stream)
int llama_remote_ep_new_stream();

// Explicitly flush GGML_REMOTE_EP_FREQ counters to stderr / the configured CSV.
// Normal applications get the same dump from the dispatcher destructor; this
// hook is for harnesses that intentionally use _Exit to avoid waiting on live
// endpoint receiver threads.
void llama_remote_ep_dump_freq();

// userdata packing for the scheduled op pair: low 16 bits = layer, high bits
// = stream id
inline void * llama_remote_ep_userdata(int stream, int il) {
    return (void *) ((((intptr_t) stream) << 16) | (il & 0xffff));
}
inline int llama_remote_ep_userdata_il(void * userdata) {
    return (int) (((intptr_t) userdata) & 0xffff);
}
inline int llama_remote_ep_userdata_stream(void * userdata) {
    return (int) (((intptr_t) userdata) >> 16);
}
