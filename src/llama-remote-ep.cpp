#include "llama-remote-ep.h"

#include "llama-impl.h"

#include "../tools/epd/llama-ep-transport.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <mutex>
#include <vector>

#include "ggml.h"

namespace {

// env-gated per-call RPC timing (GGML_REMOTE_EP_DEBUG=1), default off
bool remote_ep_debug_enabled() {
    static const bool v = []() {
        const char * e = getenv("GGML_REMOTE_EP_DEBUG");
        return e && e[0] != '\0' && strcmp(e, "0") != 0;
    }();
    return v;
}

// env-gated pipelined chunk dispatch (GGML_REMOTE_EP_PIPELINE=1), default off.
// Instead of one blocking round-trip per layer, the token dimension is split into
// chunks that are sent with a W=1 sliding window (send chunk i, then wait for the
// response of chunk i-1): the worker starts computing after the first chunk and the
// bulk of the send/recv transfer overlaps worker compute. Protocol unchanged (each
// chunk is a regular REQ frame with fewer tokens), numerics unchanged (the expert
// FFN is per-token independent). GGML_REMOTE_EP_PIPELINE_CHUNK sets the chunk size
// in tokens (default 256), capped so one chunk's hidden stays within what the W=1
// window can buffer in flight: 3 MiB over TCP (the transport installs 4 MiB socket
// buffers), 1.5 MiB over RDMA (the pre-posted receive ring holds 8 x 256 KiB).
// Staying inside those limits is what makes the window deadlock-free.
bool remote_ep_pipeline_enabled() {
    static const bool v = []() {
        const char * e = getenv("GGML_REMOTE_EP_PIPELINE");
        return e && e[0] != '\0' && strcmp(e, "0") != 0;
    }();
    return v;
}

int remote_ep_pipeline_chunk(int64_t n_embd, bool is_rdma) {
    static const int cfg = []() {
        const char * e = getenv("GGML_REMOTE_EP_PIPELINE_CHUNK");
        const int v = e ? atoi(e) : 0;
        return v > 0 ? v : 256;
    }();
    const int64_t bytes_cap = is_rdma ? (3 << 20) / 2 : (3 << 20);
    const int64_t cap = bytes_cap / (n_embd * (int64_t) sizeof(float));
    int64_t chunk = std::min<int64_t>(cfg, std::max<int64_t>(cap, 16));
    return (int) chunk;
}

struct remote_ep_state {
    bool        enabled     = false;
    std::string host        = "127.0.0.1";
    int         port        = 29200;
    int         layer_first = 0;
    int         layer_last  = 1 << 30;

    // layer mirroring + expert-slot split (GGML_REMOTE_EP_MIRROR=1)
    bool        mirror            = false;
    int         mirror_layer_first = 0;
    int         mirror_layer_last  = 1 << 30;
    int         mirror_kremote_cfg = 0; // 0 = default n_expert_used/2

    std::mutex           mtx;
    llama_ep_transport * conn = nullptr; // lazy, persistent across decode steps
    bool                 is_rdma = false;

    // in-flight mirror request (send op -> wait op). the compute thread is
    // serial between the two ops of a layer, so a single slot suffices
    struct mirror_pending {
        bool                 active      = false;
        bool                 send_failed = false;
        int                  il          = -1;
        int64_t              n_tokens    = 0;
        int64_t              n_embd      = 0;
        int64_t              k_r         = 0;
        std::vector<int32_t> ids;     // staging, ids[t*k_r + j]
        std::vector<float>   weights; // staging, same layout
        const float        * hidden = nullptr; // graph buffer, valid until the wait op
    } pend;

    ~remote_ep_state() {
        if (conn) {
            conn->ops.close(conn->ctx);
            delete conn;
        }
    }

    void parse_env() {
        const char * env = getenv("GGML_REMOTE_EP");
        if (!env || env[0] == '\0' || strcmp(env, "0") == 0) {
            return;
        }
        enabled = true;

        if (const char * h = getenv("GGML_REMOTE_EP_HOST")) {
            host = h;
        }
        if (const char * p = getenv("GGML_REMOTE_EP_PORT")) {
            port = atoi(p);
        }
        if (const char * l = getenv("GGML_REMOTE_EP_LAYERS")) {
            int a = 0, b = 0;
            if (sscanf(l, "%d-%d", &a, &b) == 2 && a <= b) {
                layer_first = a; layer_last = b;
            } else if (sscanf(l, "%d", &a) == 1) {
                layer_first = a; layer_last = a;
            } else {
                LLAMA_LOG_WARN("%s: ignoring malformed GGML_REMOTE_EP_LAYERS='%s'\n", __func__, l);
            }
        }

        // layer mirroring: master keeps the expert weights of the mirrored layers
        // and splits each MoE along the expert-slot dimension (see llama-remote-ep.h)
        if (const char * m = getenv("GGML_REMOTE_EP_MIRROR")) {
            mirror = m[0] != '\0' && strcmp(m, "0") != 0;
        }
        mirror_layer_first = layer_first;
        mirror_layer_last  = layer_last;
        if (const char * l = getenv("GGML_REMOTE_EP_MIRROR_LAYERS")) {
            int a = 0, b = 0;
            if (sscanf(l, "%d-%d", &a, &b) == 2 && a <= b) {
                mirror_layer_first = a; mirror_layer_last = b;
            } else if (sscanf(l, "%d", &a) == 1) {
                mirror_layer_first = a; mirror_layer_last = a;
            } else {
                LLAMA_LOG_WARN("%s: ignoring malformed GGML_REMOTE_EP_MIRROR_LAYERS='%s'\n", __func__, l);
            }
        }
        if (const char * k = getenv("GGML_REMOTE_EP_MIRROR_KREMOTE")) {
            mirror_kremote_cfg = atoi(k);
            if (mirror_kremote_cfg < 0) {
                mirror_kremote_cfg = 0;
            }
        }

        LLAMA_LOG_INFO("%s: GGML_REMOTE_EP=1: layers %d-%d -> %s:%d\n", __func__,
                layer_first, layer_last == (1 << 30) ? -1 : layer_last, host.c_str(), port);
        if (mirror) {
            LLAMA_LOG_INFO("%s: GGML_REMOTE_EP_MIRROR=1: mirror layers %d-%d, k_remote=%s\n", __func__,
                    mirror_layer_first, mirror_layer_last == (1 << 30) ? -1 : mirror_layer_last,
                    mirror_kremote_cfg > 0 ? std::to_string(mirror_kremote_cfg).c_str() : "n_expert_used/2");
        }
    }

    bool ensure_conn(std::string & err) {
        if (conn) {
            return true;
        }
        is_rdma = false;
#ifdef LLAMA_EP_HAVE_RDMA
        if (llama_ep_rdma_requested()) {
            conn = llama_ep_rdma_connect(host.c_str(), port, &err);
            if (conn) {
                is_rdma = true;
                LLAMA_LOG_INFO("%s: connected to %s:%d via RDMA (RoCEv2)\n", __func__, host.c_str(), port);
            } else {
                LLAMA_LOG_WARN("%s: RDMA connect failed (%s), falling back to TCP\n", __func__, err.c_str());
                err.clear();
            }
        }
#endif
        if (!conn) {
            conn = llama_ep_tcp_connect(host.c_str(), port, &err);
        }
        return conn != nullptr;
    }

    void drop_conn() {
        if (conn) {
            conn->ops.close(conn->ctx);
            delete conn;
            conn = nullptr;
        }
    }
};

remote_ep_state & remote_ep_get() {
    static remote_ep_state st;
    static std::once_flag once;
    std::call_once(once, [&]() { st.parse_env(); });
    return st;
}

// one round-trip over st.conn; returns false on any transport/protocol error
bool remote_ep_roundtrip(
        remote_ep_state & st,
        int               il,
        int64_t           n_tokens,
        int64_t           n_ids,
        int64_t           n_embd,
        const int32_t   * ids,
        const float     * weights,
        const float     * hidden,
        float           * out,
        std::string     & err) {

    llama_ep_req_header hdr = {il, (int32_t) n_tokens, (int32_t) n_ids, (int32_t) n_embd};
    const size_t n_sel = (size_t) n_tokens * n_ids;

    const void * parts[4] = {&hdr, ids, weights, hidden};
    const size_t lens[4]  = {
        sizeof(hdr),
        n_sel * sizeof(int32_t),
        n_sel * sizeof(float),
        (size_t) n_tokens * n_embd * sizeof(float),
    };
    const bool dbg = remote_ep_debug_enabled();
    const int64_t t0 = dbg ? ggml_time_us() : 0;
    if (!llama_ep_send_framev(st.conn, LLAMA_EP_MSG_REQ, parts, lens, 4)) {
        err = "send REQ failed";
        return false;
    }
    const int64_t t_sent = dbg ? ggml_time_us() : 0;

    std::vector<uint8_t> payload;
    uint32_t type = 0;
    if (!llama_ep_recv_frame(st.conn, type, payload)) {
        err = "recv RESP failed";
        return false;
    }
    if (dbg) {
        fprintf(stderr, "GGML_REMOTE_EP: [ep-debug] layer %d n_tokens=%lld send %.3f ms wait %.3f ms\n",
                il, (long long) n_tokens, (t_sent - t0) / 1000.0, (ggml_time_us() - t_sent) / 1000.0);
    }
    if (type == LLAMA_EP_MSG_ERR) {
        int32_t code = 0;
        if (payload.size() >= sizeof(code)) {
            memcpy(&code, payload.data(), sizeof(code));
        }
        err = "worker ERR " + std::to_string(code) + ": " +
              std::string((const char *) payload.data() + sizeof(code),
                          payload.size() > sizeof(code) ? payload.size() - sizeof(code) : 0);
        return false;
    }
    if (type != LLAMA_EP_MSG_RESP ||
        payload.size() != sizeof(llama_ep_resp_header) + (size_t) n_tokens * n_embd * sizeof(float)) {
        err = "bad RESP";
        return false;
    }
    memcpy(out, payload.data() + sizeof(llama_ep_resp_header), (size_t) n_tokens * n_embd * sizeof(float));
    return true;
}

// pipelined variant: send one REQ per chunk of `chunk` tokens, sliding window W=1
// (send chunk i, then receive the RESP of chunk i-1). The worker (unchanged)
// processes the chunk REQs back to back, so its compute overlaps the master's
// send/recv of the neighbouring chunks. RESP payloads are received straight into
// the matching `out` slice. Returns false on any transport/protocol error; err is
// prefixed "worker ERR" when the worker answered with an ERR frame.
bool remote_ep_send_req_chunk(
        remote_ep_state & st,
        int               il,
        int64_t           t0,
        int64_t           n_tokens,
        int64_t           n_ids,
        int64_t           n_embd,
        const int32_t   * ids,
        const float     * weights,
        const float     * hidden,
        std::string     & err) {

    llama_ep_req_header hdr = {il, (int32_t) n_tokens, (int32_t) n_ids, (int32_t) n_embd};
    const size_t n_sel = (size_t) n_tokens * n_ids;

    const void * parts[4] = {&hdr, ids + t0 * n_ids, weights + t0 * n_ids, hidden + t0 * n_embd};
    const size_t lens[4]  = {
        sizeof(hdr),
        n_sel * sizeof(int32_t),
        n_sel * sizeof(float),
        (size_t) n_tokens * n_embd * sizeof(float),
    };
    if (!llama_ep_send_framev(st.conn, LLAMA_EP_MSG_REQ, parts, lens, 4)) {
        err = "send REQ failed";
        return false;
    }
    return true;
}

bool remote_ep_recv_resp_chunk(
        remote_ep_state & st,
        int64_t           t0,
        int64_t           n_tokens,
        int64_t           n_embd,
        float           * out,
        std::string     & err) {

    llama_ep_frame_header fh;
    if (!st.conn->ops.recv_all(st.conn->ctx, &fh, sizeof(fh)) ||
        fh.magic != LLAMA_EP_MAGIC || fh.payload_len > (uint64_t) 1 << 30) {
        err = "recv RESP failed";
        return false;
    }
    if (fh.type == LLAMA_EP_MSG_ERR) {
        std::vector<uint8_t> payload((size_t) fh.payload_len);
        if (fh.payload_len > 0 && !st.conn->ops.recv_all(st.conn->ctx, payload.data(), payload.size())) {
            err = "recv ERR payload failed";
            return false;
        }
        int32_t code = 0;
        if (payload.size() >= sizeof(code)) {
            memcpy(&code, payload.data(), sizeof(code));
        }
        err = "worker ERR " + std::to_string(code) + ": " +
              std::string((const char *) payload.data() + sizeof(code),
                          payload.size() > sizeof(code) ? payload.size() - sizeof(code) : 0);
        return false;
    }
    const size_t out_bytes = (size_t) n_tokens * n_embd * sizeof(float);
    if (fh.type != LLAMA_EP_MSG_RESP || fh.payload_len != sizeof(llama_ep_resp_header) + out_bytes) {
        err = "bad RESP";
        return false;
    }
    llama_ep_resp_header rhdr;
    if (!st.conn->ops.recv_all(st.conn->ctx, &rhdr, sizeof(rhdr)) ||
        rhdr.n_tokens != (int32_t) n_tokens || rhdr.n_embd != (int32_t) n_embd ||
        !st.conn->ops.recv_all(st.conn->ctx, out + t0 * n_embd, out_bytes)) {
        err = "recv RESP payload failed";
        return false;
    }
    return true;
}

bool remote_ep_roundtrip_pipelined(
        remote_ep_state & st,
        int               il,
        int64_t           n_tokens,
        int64_t           n_ids,
        int64_t           n_embd,
        const int32_t   * ids,
        const float     * weights,
        const float     * hidden,
        float           * out,
        int64_t           chunk,
        std::string     & err) {

    const int64_t n_chunks = (n_tokens + chunk - 1) / chunk;
    const bool dbg = remote_ep_debug_enabled();
    const int64_t t_begin = dbg ? ggml_time_us() : 0;
    int64_t t_send = 0;

    for (int64_t i = 0; i < n_chunks; ++i) {
        const int64_t t0   = i * chunk;
        const int64_t ntok = std::min(chunk, n_tokens - t0);

        const int64_t ts = dbg ? ggml_time_us() : 0;
        if (!remote_ep_send_req_chunk(st, il, t0, ntok, n_ids, n_embd, ids, weights, hidden, err)) {
            return false;
        }
        t_send += dbg ? ggml_time_us() - ts : 0;

        if (i > 0 && !remote_ep_recv_resp_chunk(st, t0 - chunk, chunk, n_embd, out, err)) {
            return false;
        }
    }
    const int64_t t_last = (n_chunks - 1) * chunk;
    if (!remote_ep_recv_resp_chunk(st, t_last, n_tokens - t_last, n_embd, out, err)) {
        return false;
    }
    if (dbg) {
        fprintf(stderr, "GGML_REMOTE_EP: [ep-debug] layer %d n_tokens=%lld chunks=%lld send %.3f ms wait %.3f ms\n",
                il, (long long) n_tokens, (long long) n_chunks,
                t_send / 1000.0, (ggml_time_us() - t_begin - t_send) / 1000.0);
    }
    return true;
}

} // namespace

// send the pending mirror REQ (slots [0,k_r)) without waiting for the RESP
static bool remote_ep_mirror_send_req(remote_ep_state & st, std::string & err) {
    const auto & p = st.pend;

    llama_ep_req_header hdr = {p.il, (int32_t) p.n_tokens, (int32_t) p.k_r, (int32_t) p.n_embd};

    const void * parts[4] = {&hdr, p.ids.data(), p.weights.data(), p.hidden};
    const size_t lens[4]  = {
        sizeof(hdr),
        (size_t) p.n_tokens * p.k_r * sizeof(int32_t),
        (size_t) p.n_tokens * p.k_r * sizeof(float),
        (size_t) p.n_tokens * p.n_embd * sizeof(float),
    };
    if (!llama_ep_send_framev(st.conn, LLAMA_EP_MSG_REQ, parts, lens, 4)) {
        err = "send REQ failed";
        return false;
    }
    return true;
}

static bool llama_remote_ep_mirror_layer(int il) {
    remote_ep_state & st = remote_ep_get();
    return st.enabled && st.mirror &&
           il >= st.layer_first        && il <= st.layer_last &&
           il >= st.mirror_layer_first && il <= st.mirror_layer_last;
}

bool llama_remote_ep_skip_weights_for_layer(int il) {
    return llama_remote_ep_enabled_for_layer(il) && !llama_remote_ep_mirror_layer(il);
}

int llama_remote_ep_mirror_kremote(int il, int n_expert_used) {
    remote_ep_state & st = remote_ep_get();
    if (!llama_remote_ep_mirror_layer(il) || n_expert_used < 2) {
        return 0;
    }
    const int k_r = st.mirror_kremote_cfg > 0 ? st.mirror_kremote_cfg : n_expert_used / 2;
    return std::max(1, std::min(k_r, n_expert_used - 1));
}

bool llama_remote_ep_enabled_for_layer(int il) {
    remote_ep_state & st = remote_ep_get();
    return st.enabled && il >= st.layer_first && il <= st.layer_last;
}

bool llama_remote_ep_moe_ffn(
        int             il,
        int64_t         n_tokens,
        int64_t         n_ids,
        int64_t         n_embd,
        const int32_t * ids,
        const float   * weights,
        const float   * hidden,
        float         * out,
        std::string   & err) {

    remote_ep_state & st = remote_ep_get();
    std::lock_guard<std::mutex> lock(st.mtx);

    // one reconnect+retry on transport failure; protocol errors are fatal.
    // a failed pipelined attempt is retried via the plain round-trip on the fresh
    // connection (the worker is stateless per REQ, so a full-layer resend is safe)
    bool allow_pipeline = remote_ep_pipeline_enabled();
    for (int attempt = 0; attempt < 2; ++attempt) {
        if (!st.ensure_conn(err)) {
            if (attempt == 0) {
                continue;
            }
            return false;
        }
        // pipelined chunk dispatch (GGML_REMOTE_EP_PIPELINE=1) only kicks in when the
        // layer actually splits into >= 2 chunks; a single chunk keeps the legacy path.
        // the chunk cap depends on the transport, so it is decided after connect
        bool pipelined = false;
        int64_t chunk  = 0;
        if (allow_pipeline) {
            chunk = remote_ep_pipeline_chunk(n_embd, st.is_rdma);
            pipelined = n_tokens > chunk;
        }
        const bool ok = pipelined
            ? remote_ep_roundtrip_pipelined(st, il, n_tokens, n_ids, n_embd, ids, weights, hidden, out, chunk, err)
            : remote_ep_roundtrip(st, il, n_tokens, n_ids, n_embd, ids, weights, hidden, out, err);
        if (ok) {
            return true;
        }
        st.drop_conn();
        if (err.rfind("worker ERR", 0) == 0) {
            return false; // the worker answered; reconnecting won't help
        }
        allow_pipeline = false;
    }
    return false;
}

void llama_remote_ep_graph_cb(
        ggml_tensor       * dst,
        const ggml_tensor * a,
        const ggml_tensor * b,
        const ggml_tensor * c,
        int ith, int nth, void * userdata) {
    if (ith != 0) {
        return;
    }
    (void) nth;

    const int il = (int) (intptr_t) userdata;

    const int64_t n_embd   = a->ne[0];
    const int64_t n_tokens = a->ne[2];
    const int64_t n_ids    = b->ne[0];

    const bool ok_shapes =
        a->type == GGML_TYPE_F32 && ggml_is_contiguous(a) &&
        b->type == GGML_TYPE_I32 && ggml_is_contiguous(b) && b->ne[1] == n_tokens &&
        c->type == GGML_TYPE_F32 && ggml_is_contiguous(c) && c->ne[0] == 1 && c->ne[1] == n_ids && c->ne[2] == n_tokens &&
        dst->type == GGML_TYPE_F32 && ggml_is_contiguous(dst);

    if (!ok_shapes) {
        GGML_ABORT("%s: unexpected tensor shapes/types for layer %d", __func__, il);
    }

    std::string err;
    if (!llama_remote_ep_moe_ffn(il, n_tokens, n_ids, n_embd,
            (const int32_t *) b->data, (const float *) c->data, (const float *) a->data,
            (float *) dst->data, err)) {
        GGML_ABORT("%s: layer %d: %s", __func__, il, err.c_str());
    }
}

void llama_remote_ep_mirror_send_cb(
        ggml_tensor       * dst,
        const ggml_tensor * a,
        const ggml_tensor * b,
        const ggml_tensor * c,
        int ith, int nth, void * userdata) {
    if (ith != 0) {
        return;
    }
    (void) nth;
    (void) dst; // unused; the op only fires the REQ

    const int il = (int) (intptr_t) userdata;

    const int64_t n_embd   = a->ne[0];
    const int64_t n_tokens = a->ne[2];
    const int64_t n_ids    = b->ne[0];

    const bool ok_shapes =
        a->type == GGML_TYPE_F32 && ggml_is_contiguous(a) &&
        b->type == GGML_TYPE_I32 && ggml_is_contiguous(b) && b->ne[1] == n_tokens &&
        c->type == GGML_TYPE_F32 && ggml_is_contiguous(c) && c->ne[0] == 1 && c->ne[1] == n_ids && c->ne[2] == n_tokens;

    if (!ok_shapes) {
        GGML_ABORT("%s: unexpected tensor shapes/types for layer %d", __func__, il);
    }

    remote_ep_state & st = remote_ep_get();

    const int64_t k_r = llama_remote_ep_mirror_kremote(il, n_ids);
    if (k_r <= 0 || k_r >= n_ids) {
        GGML_ABORT("%s: layer %d is not mirrored (k_r=%lld of %lld)", __func__, il, (long long) k_r, (long long) n_ids);
    }

    const bool dbg = remote_ep_debug_enabled();
    const int64_t t0 = dbg ? ggml_time_us() : 0;

    std::lock_guard<std::mutex> lock(st.mtx);

    auto & p = st.pend;
    if (p.active) {
        GGML_ABORT("%s: layer %d: previous mirror request (layer %d) was never consumed", __func__, il, p.il);
    }

    // gather slots [0,k_r) per token into contiguous staging (wire layout
    // ids[t*k_r + j]); the source is strided (k slots per token)
    const int32_t * ids = (const int32_t *) b->data;
    const float   * w   = (const float   *) c->data;
    p.ids.resize((size_t) n_tokens * k_r);
    p.weights.resize((size_t) n_tokens * k_r);
    for (int64_t t = 0; t < n_tokens; ++t) {
        memcpy(p.ids.data()     + t * k_r, ids + t * n_ids, (size_t) k_r * sizeof(int32_t));
        memcpy(p.weights.data() + t * k_r, w   + t * n_ids, (size_t) k_r * sizeof(float));
    }

    p.il          = il;
    p.n_tokens    = n_tokens;
    p.n_embd      = n_embd;
    p.k_r         = k_r;
    p.hidden      = (const float *) a->data;
    p.send_failed = false;
    p.active      = true;

    std::string err;
    if (!st.ensure_conn(err) || !remote_ep_mirror_send_req(st, err)) {
        // leave the pending slot in place: the wait op reconnects and resends once
        LLAMA_LOG_WARN("%s: layer %d: %s (will retry in wait op)\n", __func__, il, err.c_str());
        p.send_failed = true;
        st.drop_conn();
    }

    if (dbg) {
        fprintf(stderr, "GGML_REMOTE_EP: [ep-debug] layer %d mirror k_r=%lld/%lld n_tokens=%lld send %.3f ms (t=%lld us)\n",
                il, (long long) k_r, (long long) n_ids, (long long) n_tokens,
                (ggml_time_us() - t0) / 1000.0, (long long) t0);
    }
}

void llama_remote_ep_mirror_wait_cb(
        ggml_tensor       * dst,
        const ggml_tensor * a,
        const ggml_tensor * b,
        int ith, int nth, void * userdata) {
    if (ith != 0) {
        return;
    }
    (void) nth;
    (void) a;
    (void) b; // experts_l: ordering dependency only

    const int il = (int) (intptr_t) userdata;

    remote_ep_state & st = remote_ep_get();

    const bool dbg = remote_ep_debug_enabled();
    const int64_t t0 = dbg ? ggml_time_us() : 0;

    std::lock_guard<std::mutex> lock(st.mtx);

    auto & p = st.pend;
    if (!p.active || p.il != il) {
        GGML_ABORT("%s: layer %d: no pending mirror request (active=%d, pending layer %d)",
                __func__, il, (int) p.active, p.il);
    }

    if (dst->type != GGML_TYPE_F32 || !ggml_is_contiguous(dst) ||
        dst->ne[0] != p.n_embd || dst->ne[1] != 1 || dst->ne[2] != p.n_tokens) {
        GGML_ABORT("%s: unexpected dst shape for layer %d", __func__, il);
    }

    std::string err;
    bool ok = !p.send_failed && st.conn != nullptr &&
              remote_ep_recv_resp_chunk(st, 0, p.n_tokens, p.n_embd, (float *) dst->data, err);

    if (!ok) {
        // one reconnect + resend (from the pending request bytes) + receive
        LLAMA_LOG_WARN("%s: layer %d: %s — reconnecting and resending once\n", __func__, il, err.c_str());
        st.drop_conn();
        err.clear();
        ok = st.ensure_conn(err) &&
             remote_ep_mirror_send_req(st, err) &&
             remote_ep_recv_resp_chunk(st, 0, p.n_tokens, p.n_embd, (float *) dst->data, err);
    }
    if (!ok) {
        GGML_ABORT("%s: layer %d: %s", __func__, il, err.c_str());
    }

    p.active = false;
    p.hidden = nullptr;

    if (dbg) {
        fprintf(stderr, "GGML_REMOTE_EP: [ep-debug] layer %d mirror n_tokens=%lld wait %.3f ms (t=%lld us)\n",
                il, (long long) p.n_tokens, (ggml_time_us() - t0) / 1000.0, (long long) t0);
    }
}

bool llama_remote_ep_mirror_fits(int64_t n_tokens, int64_t n_embd) {
#ifdef LLAMA_EP_HAVE_RDMA
    // over RDMA the worker's RESP send lands in the master's pre-posted receive
    // ring (8 x 256 KiB); the master only drains it in the wait op, after the
    // local chain. keep the RESP inside the ring so the worker never stalls
    // (or worse) on a send nobody is receiving yet.
    if (llama_ep_rdma_requested()) {
        return (uint64_t) n_tokens * (uint64_t) n_embd * sizeof(float) <= (uint64_t) (3 << 20) / 2;
    }
#endif
    (void) n_tokens;
    (void) n_embd;
    return true;
}
