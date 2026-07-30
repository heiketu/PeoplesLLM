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
//                   (kernel_id: ggml build + ISA fingerprint, for the §7.3
//                   homogeneity check). an old LEP1-only worker answers ERR to
//                   CAP — the master then falls back to the classic/mirror path.
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
// The transport itself is message-agnostic: a function table (send_all/recv_all/close)
// isolates the framing from the byte mover, so the TCP backend below can be swapped
// for an RDMA verbs backend without touching protocol code.

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#define LLAMA_EP_MAGIC 0x4C455031u // "LEP1"

// protocol version offered in the CAP handshake; LEP1 peers never send CAP
#define LLAMA_EP_PROTO_VER 2u

enum llama_ep_msg_type : uint32_t {
    LLAMA_EP_MSG_REQ  = 1,
    LLAMA_EP_MSG_RESP = 2,
    LLAMA_EP_MSG_ERR  = 3,
    LLAMA_EP_MSG_CAP  = 4,
    LLAMA_EP_MSG_REQ2 = 5,
    LLAMA_EP_MSG_RESP2 = 6,
};

// worker CAP caps bits
enum llama_ep_cap_bits : uint32_t {
    LLAMA_EP_CAP_REQ2 = 1u << 0, // ragged per-slot dispatch (REQ2/RESP2)
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
    void (*close)(void * ctx);
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
