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

    std::mutex           mtx;
    llama_ep_transport * conn = nullptr; // lazy, persistent across decode steps
    bool                 is_rdma = false;

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

        LLAMA_LOG_INFO("%s: GGML_REMOTE_EP=1: layers %d-%d -> %s:%d\n", __func__,
                layer_first, layer_last == (1 << 30) ? -1 : layer_last, host.c_str(), port);
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
