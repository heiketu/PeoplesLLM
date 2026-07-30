#include "llama-remote-ep.h"

#include "llama-impl.h"

#include "../tools/epd/llama-ep-dealer.h"
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
    // comma-separated range list ("3-7,22-42"); empty = use [layer_first, layer_last]
    std::vector<std::pair<int, int>> layer_ranges;

    // layer mirroring + expert-slot split (GGML_REMOTE_EP_MIRROR=1)
    bool        mirror            = false;
    int         mirror_layer_first = 0;
    int         mirror_layer_last  = 1 << 30;
    int         mirror_kremote_cfg = 0; // 0 = default n_expert_used/2

    // expert-level dynamic scheduling (GGML_REMOTE_EP_SCHED=1)
    bool        sched           = false;
    int         sched_klocal    = 2;    // m*: local slots per token
    bool        sched_pp        = false; // allow n_tokens > 1 (P4; default decode-only)
    int         sched_negotiated = 0;   // 0 = not yet, 1 = ready, -1 = failed (fallback)
    struct sched_ep {
        std::string        host;
        int                port     = 29200;
        llama_ep_transport * conn    = nullptr;
        bool               is_rdma = false;
        int32_t            expert_first = 0; // from CAP
        int32_t            expert_last  = 0;
        uint32_t           kernel_id  = 0;
    };
    std::vector<sched_ep> sched_eps;

    // in-flight scheduled layer (partition+send op -> wait+merge op). the
    // compute thread is serial between the two ops of a layer, one slot suffices
    struct sched_pend_ep {
        bool                 send_failed = false;
        std::vector<int32_t> token_idx; // REQ2 staging, kept for one resend
        std::vector<int32_t> slot_idx;
        std::vector<int32_t> expert_id;
        std::vector<float>   weight;
        std::vector<float>   resp;      // [n_sel*n_embd], filled by the merge op
    };
    struct sched_pend {
        bool                 active   = false;
        int                  il       = -1;
        int64_t              n_tokens = 0;
        int64_t              n_embd   = 0;
        int                  k        = 0;
        int                  m_local  = 0;
        const float        * hidden  = nullptr;  // graph buffers, valid until the merge op
        const float        * weights = nullptr;
        std::vector<uint8_t> owner;   // [n_tokens*k]: 0 = local, 1+i = endpoint i
        std::vector<sched_pend_ep> eps;
    } spend;

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
        for (auto & ep : sched_eps) {
            if (ep.conn) {
                ep.conn->ops.close(ep.conn->ctx);
                delete ep.conn;
            }
        }
    }

    // "A-B" or "A" or a comma-separated list of those ("3-7,22-42")
    static bool parse_ranges(const char * s, std::vector<std::pair<int, int>> & out) {
        out.clear();
        size_t i = 0;
        while (s[i] != '\0') {
            int a = 0, b = 0, n = 0;
            if (sscanf(s + i, "%d-%d%n", &a, &b, &n) == 2 && a <= b) {
                out.emplace_back(a, b);
            } else if (sscanf(s + i, "%d%n", &a, &n) == 1) {
                out.emplace_back(a, a);
            } else {
                return false;
            }
            i += (size_t) n;
            if (s[i] == ',') {
                ++i;
            } else if (s[i] != '\0') {
                return false;
            }
        }
        return !out.empty();
    }

    bool in_ranges(int il) const {
        if (layer_ranges.empty()) {
            return il >= layer_first && il <= layer_last;
        }
        for (const auto & r : layer_ranges) {
            if (il >= r.first && il <= r.second) {
                return true;
            }
        }
        return false;
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
            std::vector<std::pair<int, int>> rs;
            if (parse_ranges(l, rs)) {
                layer_ranges = rs;
                layer_first = rs.front().first;
                layer_last  = rs.back().second;
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

        // expert-level dynamic scheduling (GGML_REMOTE_EP_SCHED=1): mutually
        // exclusive with MIRROR (SCHED wins); master keeps a full replica
        if (const char * s = getenv("GGML_REMOTE_EP_SCHED")) {
            sched = s[0] != '\0' && strcmp(s, "0") != 0;
        }
        if (sched) {
            if (const char * e = getenv("GGML_REMOTE_EP_SCHED_ENDPOINTS")) {
                std::string v = e;
                size_t off = 0;
                while (off <= v.size()) {
                    const size_t comma = v.find(',', off);
                    const std::string item = v.substr(off, comma == std::string::npos ? comma : comma - off);
                    const size_t colon = item.rfind(':');
                    if (colon == std::string::npos || colon == 0) {
                        LLAMA_LOG_WARN("%s: ignoring malformed endpoint '%s' (want host:port)\n", __func__, item.c_str());
                    } else {
                        sched_ep ep;
                        ep.host = item.substr(0, colon);
                        ep.port = atoi(item.substr(colon + 1).c_str());
                        sched_eps.push_back(ep);
                    }
                    if (comma == std::string::npos) {
                        break;
                    }
                    off = comma + 1;
                }
            }
            if (sched_eps.empty()) {
                sched_ep ep;
                ep.host = host;
                ep.port = port;
                sched_eps.push_back(ep);
            }
            if (const char * k = getenv("GGML_REMOTE_EP_SCHED_KLOCAL")) {
                sched_klocal = atoi(k);
                if (sched_klocal < 1) {
                    sched_klocal = 1;
                }
            }
            if (const char * p = getenv("GGML_REMOTE_EP_SCHED_PP")) {
                sched_pp = p[0] != '\0' && strcmp(p, "0") != 0;
            }
            if (const char * d = getenv("GGML_REMOTE_EP_SCHED_DEAL")) {
                if (strcmp(d, "static") != 0 && strcmp(d, "balance") != 0) {
                    LLAMA_LOG_WARN("%s: ignoring unknown GGML_REMOTE_EP_SCHED_DEAL='%s'\n", __func__, d);
                }
                // P0: both modes use the same deterministic pure-function dealer
            }
            if (mirror) {
                LLAMA_LOG_WARN("%s: GGML_REMOTE_EP_SCHED=1 takes precedence over MIRROR; mirror disabled\n", __func__);
            }
            LLAMA_LOG_INFO("%s: GGML_REMOTE_EP_SCHED=1: %zu endpoint(s), k_local=%d, pp=%d\n",
                    __func__, sched_eps.size(), sched_klocal, (int) sched_pp);
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
    return st.enabled && st.mirror && !st.sched &&
           st.in_ranges(il) &&
           il >= st.mirror_layer_first && il <= st.mirror_layer_last;
}

bool llama_remote_ep_skip_weights_for_layer(int il) {
    // SCHED mode: the master keeps a full expert replica (never skip)
    return llama_remote_ep_enabled_for_layer(il) && !llama_remote_ep_mirror_layer(il) &&
           !remote_ep_get().sched;
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
    return st.enabled && st.in_ranges(il);
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

// ---------------------------------------------------------------------------
// expert-level dynamic scheduling (GGML_REMOTE_EP_SCHED=1, SCHEDULER-DESIGN.md)
// ---------------------------------------------------------------------------

static bool remote_ep_sched_ep_connect(remote_ep_state::sched_ep & ep, std::string & err) {
    if (ep.conn) {
        return true;
    }
    ep.is_rdma = false;
#ifdef LLAMA_EP_HAVE_RDMA
    if (llama_ep_rdma_requested()) {
        ep.conn = llama_ep_rdma_connect(ep.host.c_str(), ep.port, &err);
        if (ep.conn) {
            ep.is_rdma = true;
        } else {
            LLAMA_LOG_WARN("%s: RDMA connect to %s:%d failed (%s), falling back to TCP\n",
                    __func__, ep.host.c_str(), ep.port, err.c_str());
            err.clear();
        }
    }
#endif
    if (!ep.conn) {
        ep.conn = llama_ep_tcp_connect(ep.host.c_str(), ep.port, &err);
    }
    return ep.conn != nullptr;
}

static void remote_ep_sched_ep_drop(remote_ep_state::sched_ep & ep) {
    if (ep.conn) {
        ep.conn->ops.close(ep.conn->ctx);
        delete ep.conn;
        ep.conn = nullptr;
    }
}

// CAP handshake with one endpoint (protocol v2). an old LEP1-only worker
// answers ERR to the unknown CAP type — treat any non-CAP reply as unsupported
static bool remote_ep_sched_ep_cap(remote_ep_state::sched_ep & ep, std::string & err) {
    llama_ep_cap_master mcap = {LLAMA_EP_PROTO_VER, 0};
    if (!llama_ep_send_frame(ep.conn, LLAMA_EP_MSG_CAP, &mcap, sizeof(mcap))) {
        err = "send CAP failed";
        return false;
    }
    uint32_t type = 0;
    std::vector<uint8_t> payload;
    if (!llama_ep_recv_frame(ep.conn, type, payload)) {
        err = "recv CAP failed";
        return false;
    }
    if (type != LLAMA_EP_MSG_CAP || payload.size() < sizeof(llama_ep_cap_worker)) {
        err = "worker does not speak protocol v2 (old LEP1 worker?)";
        return false;
    }
    llama_ep_cap_worker wcap;
    memcpy(&wcap, payload.data(), sizeof(wcap));
    if (wcap.proto_ver < LLAMA_EP_PROTO_VER || !(wcap.caps & LLAMA_EP_CAP_REQ2)) {
        err = "worker CAP lacks REQ2";
        return false;
    }
    if (wcap.kernel_id != llama_ep_kernel_id()) {
        err = "kernel_id mismatch (heterogeneous ggml build — SCHEDULER-DESIGN §7.3)";
        return false;
    }
    ep.expert_first = wcap.expert_first;
    ep.expert_last  = wcap.expert_last;
    ep.kernel_id    = wcap.kernel_id;
    return true;
}

// one-time negotiation with every endpoint; caches the outcome in
// st.sched_negotiated. caller holds st.mtx.
static bool remote_ep_sched_negotiate(remote_ep_state & st) {
    if (st.sched_negotiated != 0) {
        return st.sched_negotiated > 0;
    }
    bool ok = true;
    std::string err;
    for (auto & ep : st.sched_eps) {
        if (!remote_ep_sched_ep_connect(ep, err) || !remote_ep_sched_ep_cap(ep, err)) {
            LLAMA_LOG_WARN("%s: sched endpoint %s:%d: %s — scheduling disabled, layers fall back to classic/mirror\n",
                    __func__, ep.host.c_str(), ep.port, err.c_str());
            ok = false;
            break;
        }
        LLAMA_LOG_INFO("%s: sched endpoint %s:%d: protocol v2, experts [%d, %d), kernel_id %08x%s\n",
                __func__, ep.host.c_str(), ep.port, ep.expert_first, ep.expert_last, ep.kernel_id,
                ep.is_rdma ? " [rdma]" : "");
    }
    if (!ok) {
        for (auto & ep : st.sched_eps) {
            remote_ep_sched_ep_drop(ep);
        }
        st.sched_negotiated = -1;
        return false;
    }
    st.sched_negotiated = 1;
    return true;
}

int llama_remote_ep_sched_klocal(int il, int n_expert_used) {
    remote_ep_state & st = remote_ep_get();
    if (!st.enabled || !st.sched || !st.in_ranges(il) || st.sched_eps.empty() || n_expert_used < 1) {
        return 0;
    }
    std::lock_guard<std::mutex> lock(st.mtx);
    if (!remote_ep_sched_negotiate(st)) {
        return 0;
    }
    return std::min(st.sched_klocal, n_expert_used);
}

bool llama_remote_ep_sched_fits(int64_t n_tokens, int64_t n_ids, int64_t n_embd) {
    remote_ep_state & st = remote_ep_get();
    if (!st.sched_pp && n_tokens != 1) {
        return false; // decode-only by default (P4 lifts this)
    }
#ifdef LLAMA_EP_HAVE_RDMA
    // worst-case RESP2 (every slot remote) must fit the pre-posted receive ring
    if (llama_ep_rdma_requested()) {
        return (uint64_t) n_tokens * (uint64_t) n_ids * (uint64_t) n_embd * sizeof(float) <= (uint64_t) (3 << 20) / 2;
    }
#endif
    (void) n_ids;
    (void) n_embd;
    return true;
}

// fire the staged REQ2 of one endpoint (send only, no wait)
static bool remote_ep_sched_send_req2(remote_ep_state & st, int iep, std::string & err) {
    const auto & p  = st.spend;
    const auto & pe = p.eps[(size_t) iep];
    auto & ep = st.sched_eps[(size_t) iep];

    llama_ep_req2_header hdr = {p.il, (int32_t) p.n_tokens, (int32_t) pe.token_idx.size(), (int32_t) p.n_embd};

    const void * parts[6] = {&hdr, pe.token_idx.data(), pe.slot_idx.data(), pe.expert_id.data(), pe.weight.data(), p.hidden};
    const size_t lens[6]  = {
        sizeof(hdr),
        pe.token_idx.size() * sizeof(int32_t),
        pe.slot_idx.size()  * sizeof(int32_t),
        pe.expert_id.size() * sizeof(int32_t),
        pe.weight.size()    * sizeof(float),
        (size_t) p.n_tokens * p.n_embd * sizeof(float),
    };
    if (!llama_ep_send_framev(ep.conn, LLAMA_EP_MSG_REQ2, parts, lens, 6)) {
        err = "send REQ2 failed";
        return false;
    }
    return true;
}

// receive one RESP2 into the endpoint's resp staging
static bool remote_ep_sched_recv_resp2(remote_ep_state & st, int iep, std::string & err) {
    auto & p  = st.spend;
    auto & pe = p.eps[(size_t) iep];
    auto & ep = st.sched_eps[(size_t) iep];

    const size_t n_sel  = pe.token_idx.size();
    const size_t out_bytes = n_sel * (size_t) p.n_embd * sizeof(float);

    llama_ep_frame_header fh;
    if (!ep.conn->ops.recv_all(ep.conn->ctx, &fh, sizeof(fh)) ||
        fh.magic != LLAMA_EP_MAGIC || fh.payload_len > (uint64_t) 1 << 30) {
        err = "recv RESP2 failed";
        return false;
    }
    if (fh.type == LLAMA_EP_MSG_ERR) {
        std::vector<uint8_t> payload((size_t) fh.payload_len);
        if (fh.payload_len > 0 && !ep.conn->ops.recv_all(ep.conn->ctx, payload.data(), payload.size())) {
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
    if (fh.type != LLAMA_EP_MSG_RESP2 || fh.payload_len != sizeof(llama_ep_resp2_header) + out_bytes) {
        err = "bad RESP2";
        return false;
    }
    llama_ep_resp2_header rhdr;
    pe.resp.resize(n_sel * (size_t) p.n_embd);
    if (!ep.conn->ops.recv_all(ep.conn->ctx, &rhdr, sizeof(rhdr)) ||
        rhdr.n_sel != (int32_t) n_sel || rhdr.n_embd != (int32_t) p.n_embd ||
        (out_bytes > 0 && !ep.conn->ops.recv_all(ep.conn->ctx, pe.resp.data(), out_bytes))) {
        err = "recv RESP2 payload failed";
        return false;
    }
    return true;
}

void llama_remote_ep_sched_send_cb(ggml_tensor * dst, int ith, int nth, void * userdata) {
    if (ith != 0) {
        return;
    }
    (void) nth;

    const int il = (int) (intptr_t) userdata;

    const ggml_tensor * a = dst->src[0]; // hidden  [n_embd, 1, n_tokens] f32 (contiguous)
    const ggml_tensor * b = dst->src[1]; // ids     [k, n_tokens] i32 (contiguous)
    const ggml_tensor * c = dst->src[2]; // weights [1, k, n_tokens] f32 (contiguous)

    const int64_t n_embd   = a->ne[0];
    const int64_t n_tokens = a->ne[2];
    const int64_t k        = b->ne[0];

    const bool ok_shapes =
        a->type == GGML_TYPE_F32 && ggml_is_contiguous(a) &&
        b->type == GGML_TYPE_I32 && ggml_is_contiguous(b) && b->ne[1] == n_tokens &&
        c->type == GGML_TYPE_F32 && ggml_is_contiguous(c) && c->ne[0] == 1 && c->ne[1] == k && c->ne[2] == n_tokens &&
        dst->type == GGML_TYPE_I32 && ggml_is_contiguous(dst) && dst->ne[1] == n_tokens;

    if (!ok_shapes) {
        GGML_ABORT("%s: unexpected tensor shapes/types for layer %d", __func__, il);
    }

    remote_ep_state & st = remote_ep_get();

    const bool dbg = remote_ep_debug_enabled();
    const int64_t t0 = dbg ? ggml_time_us() : 0;

    std::lock_guard<std::mutex> lock(st.mtx);

    if (st.sched_negotiated != 1) {
        GGML_ABORT("%s: layer %d: sched endpoints were not negotiated at graph build", __func__, il);
    }

    auto & p = st.spend;
    if (p.active) {
        GGML_ABORT("%s: layer %d: previous sched request (layer %d) was never consumed", __func__, il, p.il);
    }

    const int m_local = (int) dst->ne[0];
    const int32_t * ids = (const int32_t *) b->data;
    const float   * w   = (const float   *) c->data;

    // holder bitmask: bit0 = master (full replica), bit(1+i) = endpoint i (CAP range)
    int32_t e_max = 0;
    for (int64_t i = 0; i < n_tokens * k; ++i) {
        e_max = std::max(e_max, ids[i]);
    }
    std::vector<uint64_t> holders((size_t) e_max + 1, 1u);
    for (size_t i = 0; i < st.sched_eps.size(); ++i) {
        const int64_t lo = st.sched_eps[i].expert_first;
        const int64_t hi = std::min<int64_t>(st.sched_eps[i].expert_last, (int64_t) e_max + 1);
        for (int64_t e = lo; e < hi; ++e) {
            holders[(size_t) e] |= 2ull << i;
        }
    }

    llama_ep_dealer_input din;
    din.n_tokens    = (int) n_tokens;
    din.k           = (int) k;
    din.n_endpoints = (int) st.sched_eps.size();
    din.m_star      = m_local;
    din.ids         = ids;
    din.holders     = holders.data();

    llama_ep_dealer_plan plan;
    if (!llama_ep_dealer_plan_build(din, plan) || plan.m_local != m_local) {
        GGML_ABORT("%s: layer %d: dealer failed (infeasible holder table)", __func__, il);
    }

    // local ids -> dst ([m*, n_tokens], ascending global slot order per token)
    memcpy(dst->data, plan.local_ids.data(), (size_t) n_tokens * m_local * sizeof(int32_t));

    // stage the plan + per-endpoint REQ2 bytes (kept for one resend in the merge op)
    p.il       = il;
    p.n_tokens = n_tokens;
    p.n_embd   = n_embd;
    p.k        = (int) k;
    p.m_local  = m_local;
    p.hidden   = (const float *) a->data;
    p.weights  = w;
    p.owner    = std::move(plan.owner);
    p.eps.clear();
    p.eps.resize(st.sched_eps.size());
    for (size_t i = 0; i < st.sched_eps.size(); ++i) {
        const auto & src = plan.eps[i];
        auto & pe = p.eps[i];
        pe.token_idx = src.token;
        pe.slot_idx  = src.slot;
        pe.expert_id = src.expert;
        pe.weight.resize(src.token.size());
        for (size_t j = 0; j < src.token.size(); ++j) {
            pe.weight[j] = w[(size_t) src.token[j] * k + src.slot[j]];
        }
        pe.resp.clear();
        pe.send_failed = false;
    }
    p.active = true;

    // fire one REQ2 per endpoint (send only, no wait); the local chain that
    // follows overlaps the workers' compute (same mechanism as the mirror op pair)
    for (size_t i = 0; i < st.sched_eps.size(); ++i) {
        auto & pe = p.eps[i];
        if (pe.token_idx.empty()) {
            continue;
        }
        std::string err;
        if (!remote_ep_sched_ep_connect(st.sched_eps[i], err) || !remote_ep_sched_send_req2(st, (int) i, err)) {
            // leave the staging in place: the merge op reconnects and resends once
            LLAMA_LOG_WARN("%s: layer %d: endpoint %zu: %s (will retry in merge op)\n", __func__, il, i, err.c_str());
            pe.send_failed = true;
            remote_ep_sched_ep_drop(st.sched_eps[i]);
        }
    }

    if (dbg) {
        fprintf(stderr, "GGML_REMOTE_EP: [ep-debug] layer %d sched k=%lld m*=%d n_tokens=%lld remote=[",
                il, (long long) k, m_local, (long long) n_tokens);
        for (size_t i = 0; i < p.eps.size(); ++i) {
            fprintf(stderr, "%s%zu", i ? "," : "", p.eps[i].token_idx.size());
        }
        fprintf(stderr, "] deal+send %.3f ms (t=%lld us)\n", (ggml_time_us() - t0) / 1000.0, (long long) t0);
    }
}

void llama_remote_ep_sched_merge_cb(ggml_tensor * dst, int ith, int nth, void * userdata) {
    if (ith != 0) {
        return;
    }
    (void) nth;

    const int il = (int) (intptr_t) userdata;

    const ggml_tensor * experts_l_t = dst->src[1]; // [n_embd, m*, n_tokens] f32 (NOT weighted)
    const ggml_tensor * weights_t   = dst->src[2]; // [1, k, n_tokens] f32

    remote_ep_state & st = remote_ep_get();

    const bool dbg = remote_ep_debug_enabled();
    const int64_t t0 = dbg ? ggml_time_us() : 0;

    std::lock_guard<std::mutex> lock(st.mtx);

    auto & p = st.spend;
    if (!p.active || p.il != il) {
        GGML_ABORT("%s: layer %d: no pending sched request (active=%d, pending layer %d)",
                __func__, il, (int) p.active, p.il);
    }

    if (dst->type != GGML_TYPE_F32 || !ggml_is_contiguous(dst) ||
        dst->ne[0] != p.n_embd || dst->ne[1] != 1 || dst->ne[2] != p.n_tokens ||
        experts_l_t->type != GGML_TYPE_F32 || !ggml_is_contiguous(experts_l_t) ||
        experts_l_t->ne[0] != p.n_embd || experts_l_t->ne[1] != (int64_t) p.m_local ||
        weights_t->type != GGML_TYPE_F32 || !ggml_is_contiguous(weights_t)) {
        GGML_ABORT("%s: unexpected dst/src shapes for layer %d", __func__, il);
    }

    // receive the RESP2s in fixed endpoint order; one reconnect+resend each
    for (size_t i = 0; i < st.sched_eps.size(); ++i) {
        auto & pe = p.eps[i];
        if (pe.token_idx.empty()) {
            continue;
        }
        std::string err;
        bool ok = !pe.send_failed && st.sched_eps[i].conn != nullptr &&
                  remote_ep_sched_recv_resp2(st, (int) i, err);
        if (!ok) {
            LLAMA_LOG_WARN("%s: layer %d: endpoint %zu: %s — reconnecting and resending once\n",
                    __func__, il, i, err.c_str());
            remote_ep_sched_ep_drop(st.sched_eps[i]);
            err.clear();
            ok = remote_ep_sched_ep_connect(st.sched_eps[i], err) &&
                 remote_ep_sched_ep_cap(st.sched_eps[i], err) &&
                 remote_ep_sched_send_req2(st, (int) i, err) &&
                 remote_ep_sched_recv_resp2(st, (int) i, err);
        }
        if (!ok) {
            GGML_ABORT("%s: layer %d: endpoint %zu: %s", __func__, il, i, err.c_str());
        }
    }

    const int64_t t_resp = dbg ? ggml_time_us() : 0;

    // merge in ascending global slot order, left-associated — the same scalar
    // operation sequence as the baseline ggml_mul + ggml_add chain (§4.5):
    // local contributions are multiplied by their router weight here (ggml_mul
    // is an elementwise f32 multiply; doing it scalar in the merge is the same
    // rounding), remote contributions arrive already weighted in the RESP2.
    const float * experts_l = (const float *) experts_l_t->data;
    const float * w         = (const float *) weights_t->data;
    float       * out       = (float       *) dst->data;

    const int64_t k       = p.k;
    const int64_t m_local = p.m_local;
    const int64_t n_embd  = p.n_embd;

    std::vector<size_t> cur_ep(p.eps.size(), 0);
    for (int64_t t = 0; t < p.n_tokens; ++t) {
        float * acc = out + t * n_embd;
        int64_t lp = 0; // local slots are ascending per token
        for (int64_t j = 0; j < k; ++j) {
            const uint8_t o = p.owner[(size_t) t * k + j];
            if (o == 0) {
                const float * v = experts_l + ((size_t) t * m_local + lp) * n_embd;
                const float wj = w[(size_t) t * k + j];
                ++lp;
                if (j == 0) {
                    for (int64_t e = 0; e < n_embd; ++e) {
                        acc[e] = v[e] * wj;
                    }
                } else {
                    for (int64_t e = 0; e < n_embd; ++e) {
                        acc[e] += v[e] * wj;
                    }
                }
            } else {
                auto & pe = p.eps[(size_t) o - 1];
                const size_t idx = cur_ep[(size_t) o - 1]++;
                const float * v = pe.resp.data() + idx * (size_t) n_embd;
                if (j == 0) {
                    memcpy(acc, v, (size_t) n_embd * sizeof(float));
                } else {
                    for (int64_t e = 0; e < n_embd; ++e) {
                        acc[e] += v[e];
                    }
                }
            }
        }
    }

    p.active  = false;
    p.hidden  = nullptr;
    p.weights = nullptr;

    if (dbg) {
        fprintf(stderr, "GGML_REMOTE_EP: [ep-debug] layer %d sched n_tokens=%lld wait %.3f ms merge %.3f ms (t=%lld us)\n",
                il, (long long) p.n_tokens, (t_resp - t0) / 1000.0,
                (ggml_time_us() - t_resp) / 1000.0, (long long) t0);
    }
}
