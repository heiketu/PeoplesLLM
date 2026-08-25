#pragma once

// llama-ep-transport: message framing for MoE expert-parallel activation dispatch.
//
// Wire format (all integers little-endian):
//   frame header: { u32 magic, u32 type, u64 payload_len }
//   frame payload: type-specific
//
//   REQ  (type 1): { i32 layer, i32 n_tokens, i32 n_ids, i32 n_embd }
//                  + i32  expert_ids[n_tokens*n_ids]   (ids[t*n_ids + k] = k-th expert of token t)
//                  + fp32 weights[n_tokens*n_ids]      (router weight, same layout)
//                  + fp32 hidden[n_tokens*n_embd]      (row-major per token)
//   RESP (type 2): { i32 n_tokens, i32 n_embd }
//                  + fp32 out[n_tokens*n_embd]         (sum_k w[t,k] * expert_{ids[t,k]}(hidden[t]))
//   ERR  (type 3): { i32 code } + char msg[payload_len - 4]
//
// protocol v2 (expert-level dynamic scheduling, SCHEDULER-DESIGN.md §4.4):
//   CAP   (type 4): first frame after connect, master->worker:
//                   { u32 proto_ver=2, u32 flags }
//                   worker->master reply:
//                   { u32 proto_ver, u32 caps, i32 layer_first, i32 layer_last,
//                     i32 expert_first, i32 expert_last, u32 kernel_id }
//                   + optional u8 expert_bitmap[ceil((expert_last-expert_first)/8)]
//                   when LLAMA_EP_CAP_EXPERT_BITMAP is set. Bitmap bit e-first
//                   reports arbitrary sparse ownership; the fixed prefix stays
//                   wire-compatible with range-only protocol-v2 workers.
//                   A master that sets LLAMA_EP_CAP_MASTER_WANT_PRECISION asks a
//                   current worker to append llama_ep_precision_contract before
//                   the optional bitmap. Old masters send flags=0 and therefore
//                   continue to receive the old payload exactly.
//                   (kernel_id: ggml build + ISA fingerprint, for the §7.3
//                   homogeneity check). an old LEP1-only worker answers ERR to
//                   CAP. The caller may use a retained local path; a pure sharded
//                   master must fail because it has no expert weights to use.
//   REQ2  (type 5): { i32 layer, i32 n_tokens, i32 n_sel, i32 n_embd }
//                   + i32 token_idx[n_sel]   (which token this assignment belongs to)
//                   + i32 slot_idx[n_sel]    (global slot index; master merges by it)
//                   + i32 expert_id[n_sel]
//                   + fp32 weight[n_sel]
//                   + fp32 hidden[n_tokens*n_embd]
//   RESP2 (type 6): { i32 n_tokens, i32 n_sel, i32 n_embd }
//                   + fp32 out[n_sel*n_embd] (one vector per assignment, already
//                   multiplied by the router weight, NOT summed; same order as REQ2)
//
// protocol v3 (async pipelined dispatch, cross-slot pipeline scheduler):
//   REQ3  (type 7): { i32 layer, i32 n_tokens, i32 n_sel, i32 n_embd, u64 req_id }
//                   + same ragged arrays + hidden as REQ2. the worker answers
//                   with RESP3 carrying the same req_id; several REQ3s may be in
//                   flight on one connection and responses may be consumed in any
//                   order by the master (the worker itself still computes FIFO).
//   RESP3 (type 8): { u64 req_id, i32 n_sel, i32 n_embd }
//                   + fp32 out[n_sel*n_embd] (same semantics as RESP2)
//
// protocol v2 optional master-weighted sync dispatch:
//   REQ4  (type 9): payload is byte-identical to REQ2.
//   RESP4 (type 10): header/shape are byte-identical to RESP2, but each expert
//                    vector is unweighted. The master applies the router weight
//                    in its existing global slot-order merge. This removes one
//                    worker graph op/barrier without changing model math.
//
// The transport itself is message-agnostic: a function table (send_all/recv_all/close)
// isolates the framing from the byte mover, so the TCP backend below can be swapped
// for an RDMA verbs backend without touching protocol code.

#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

// Payload arrays are sent without byte swapping. Make the wire contract honest:
// unsupported hosts fail at build time instead of silently corrupting a mixed-
// endian cluster.
#if defined(__BYTE_ORDER__) && defined(__ORDER_LITTLE_ENDIAN__)
static_assert(__BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__,
        "llama EP protocol currently requires a little-endian host");
#endif
static_assert(sizeof(float) == 4 && std::numeric_limits<float>::is_iec559,
        "llama EP protocol requires IEEE-754 binary32 floats");

#define LLAMA_EP_MAGIC 0x4C455031u // "LEP1"

// Keep allocations and gathered sends bounded on both sides of the wire.
constexpr uint64_t LLAMA_EP_MAX_FRAME_BYTES = (uint64_t) 1 << 30;

// protocol version offered in the CAP handshake; LEP1 peers never send CAP
#define LLAMA_EP_PROTO_VER 2u

enum llama_ep_msg_type : uint32_t {
    LLAMA_EP_MSG_REQ  = 1,
    LLAMA_EP_MSG_RESP = 2,
    LLAMA_EP_MSG_ERR  = 3,
    LLAMA_EP_MSG_CAP  = 4,
    LLAMA_EP_MSG_REQ2 = 5,
    LLAMA_EP_MSG_RESP2 = 6,
    LLAMA_EP_MSG_REQ3 = 7,
    LLAMA_EP_MSG_RESP3 = 8,
    LLAMA_EP_MSG_REQ4 = 9,
    LLAMA_EP_MSG_RESP4 = 10,
};

// worker CAP caps bits
enum llama_ep_cap_bits : uint32_t {
    LLAMA_EP_CAP_REQ2 = 1u << 0, // ragged per-slot dispatch (REQ2/RESP2)
    LLAMA_EP_CAP_REQ3 = 1u << 1, // async req_id dispatch (REQ3/RESP3)
    LLAMA_EP_CAP_EXPERT_BITMAP = 1u << 2, // CAP appends sparse ownership bitmap
    LLAMA_EP_CAP_REQ4 = 1u << 3, // sync ragged dispatch, master applies weights
    LLAMA_EP_CAP_PRECISION_CONTRACT = 1u << 4, // CAP carries llama_ep_precision_contract
};

enum llama_ep_cap_master_flags : uint32_t {
    LLAMA_EP_CAP_MASTER_WANT_PRECISION = 1u << 0,
};

enum llama_ep_precision_profile_bits : uint32_t {
    LLAMA_EP_PRECISION_DETERMINISTIC_ENDPOINT = 1u << 0,
    LLAMA_EP_PRECISION_RAW_SLOT_F32            = 1u << 1,
    LLAMA_EP_PRECISION_WEIGHTED_SLOT_F32       = 1u << 2,
};

enum llama_ep_precision_ids : uint32_t {
    LLAMA_EP_PRECISION_ACTIVATION_Q8_0_BLOCK32_FP16_NATIVE = 1,
    LLAMA_EP_PRECISION_DOT_I8_I8_I32_BLOCK32_F32           = 1,
    LLAMA_EP_PRECISION_FFN_MODEL_SCHEMA_E8M0_LEGACY_FINITE = 3,
    LLAMA_EP_PRECISION_MERGE_PER_SLOT_F32                  = 1,
};

enum llama_ep_err_code : int32_t {
    LLAMA_EP_ERR_GENERIC       = 1,
    LLAMA_EP_ERR_BAD_LAYER     = 2, // layer not owned by this worker (or not MoE)
    LLAMA_EP_ERR_BAD_EXPERT    = 3, // expert id outside this worker's range
    LLAMA_EP_ERR_BAD_SHAPE     = 4, // n_embd / n_tokens / n_ids mismatch
    LLAMA_EP_ERR_COMPUTE       = 5,
};

#pragma pack(push, 1)
struct llama_ep_frame_header {
    uint32_t magic;
    uint32_t type;
    uint64_t payload_len;
};

struct llama_ep_req_header {
    int32_t layer;
    int32_t n_tokens;
    int32_t n_ids;   // experts per token (top-k)
    int32_t n_embd;
};

struct llama_ep_resp_header {
    int32_t n_tokens;
    int32_t n_embd;
};

struct llama_ep_cap_master {
    uint32_t proto_ver;
    uint32_t flags;
};

struct llama_ep_cap_worker {
    uint32_t proto_ver;
    uint32_t caps;
    int32_t  layer_first;
    int32_t  layer_last;
    int32_t  expert_first; // half-open [first, last), clamped to actual n_expert
    int32_t  expert_last;
    uint32_t kernel_id;
};

// Optional UPE extension negotiated by LLAMA_EP_CAP_MASTER_WANT_PRECISION.
// IDs describe observable execution semantics, not merely a formula name.
// model_schema_id covers architecture, layer shapes, weight types and FFN
// nonlinear/clamp metadata. data_epoch_id is an operator-supplied immutable
// model-version hash; zero explicitly means unknown and is rejected by strict
// UPE mode. contract_id fingerprints every preceding field plus kernel_id.
struct llama_ep_precision_contract {
    uint32_t version;
    uint32_t profile_flags;
    uint32_t activation_id;
    uint32_t dot_id;
    uint32_t ffn_id;
    uint32_t merge_id;
    uint64_t model_schema_id;
    uint64_t data_epoch_id;
    uint64_t contract_id;
};

struct llama_ep_req2_header {
    int32_t layer;
    int32_t n_tokens;
    int32_t n_sel;  // ragged assignment count (not n_tokens*k)
    int32_t n_embd;
};

struct llama_ep_resp2_header {
    int32_t n_tokens;
    int32_t n_sel;
    int32_t n_embd;
};

struct llama_ep_req3_header {
    int32_t layer;
    int32_t n_tokens;
    int32_t n_sel;
    int32_t n_embd;
    uint64_t req_id;
};

struct llama_ep_resp3_header {
    uint64_t req_id;
    int32_t n_sel;
    int32_t n_embd;
};
#pragma pack(pop)

// ggml build + ISA fingerprint (fnv1a over compile-time feature macros); two
// nodes with identical builds produce identical ids. used by the CAP handshake
// for the SCHEDULER-DESIGN §7.3 homogeneity check (bit-exact merge only holds
// between same-kernel nodes).
uint32_t llama_ep_kernel_id();

// transport function table: swap the TCP implementation for RDMA later
struct llama_ep_transport_ops {
    bool (*send_all)(void * ctx, const void * data, size_t len); // false on error/EOF
    bool (*recv_all)(void * ctx, void * data, size_t len);       // false on error/EOF
    void (*shutdown)(void * ctx); // wake blocked I/O without freeing ctx
    void (*close)(void * ctx);
    // Optional gathered byte-stream send. The framing layer uses this to keep
    // one logical frame in the minimum number of backend messages without
    // first concatenating it in an unregistered temporary buffer.
    bool (*sendv_all)(void * ctx, const void * const * parts, const size_t * part_lens, size_t n_parts);
};

struct llama_ep_transport {
    void * ctx;
    llama_ep_transport_ops ops;
};

// listening endpoint (server side)
struct llama_ep_listener_ops {
    // accept one connection; returns false on error. fills *out (caller frees with ops.close).
    bool (*accept)(void * ctx, llama_ep_transport * out);
    void (*close)(void * ctx);
};

struct llama_ep_listener {
    void * ctx;
    llama_ep_listener_ops ops;
};

// TCP backend (blocking, TCP_NODELAY). host may be NULL/"0.0.0.0" for listen.
// Returns nullptr on failure and fills err (if non-null).
llama_ep_transport * llama_ep_tcp_connect(const char * host, int port, std::string * err = nullptr);
llama_ep_listener  * llama_ep_tcp_listen(const char * host, int port, std::string * err = nullptr);

// port the listener is actually bound to (useful with port 0)
int llama_ep_tcp_listener_port(const llama_ep_listener * l);

// RDMA (RoCEv2) backend via rdma_cm, same blocking byte-stream semantics as TCP.
// Built only when CMake detects libibverbs + librdmacm (LLAMA_EP_HAVE_RDMA);
// callers must be compiled with that define and fall back to TCP on nullptr.
llama_ep_transport * llama_ep_rdma_connect(const char * host, int port, std::string * err = nullptr);
llama_ep_listener  * llama_ep_rdma_listen(const char * host, int port, std::string * err = nullptr);
int llama_ep_rdma_listener_port(const llama_ep_listener * l);

// true when GGML_REMOTE_EP_RDMA is set (always available, no RDMA build required)
bool llama_ep_rdma_requested();

// framing helpers built on top of send_all/recv_all (backend-agnostic)
bool llama_ep_send_frame(llama_ep_transport * t, uint32_t type, const void * payload, size_t payload_len);
// gathered send: parts are concatenated into the payload (avoids one big copy for large hiddens)
bool llama_ep_send_framev(llama_ep_transport * t, uint32_t type, const void ** parts, const size_t * part_lens, size_t n_parts);
// receive one frame; payload buffer is resized to fit. returns false on EOF/error.
bool llama_ep_recv_frame(llama_ep_transport * t, uint32_t & type, std::vector<uint8_t> & payload);
