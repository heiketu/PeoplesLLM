#include "llama-remote-ep.h"

#include "llama-impl.h"

#include "../tools/epd/llama-ep-transport.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <vector>

#include "ggml.h"

namespace {

struct remote_ep_state {
    bool        enabled     = false;
    std::string host        = "127.0.0.1";
    int         port        = 29200;
    int         layer_first = 0;
    int         layer_last  = 1 << 30;

    std::mutex           mtx;
    llama_ep_transport * conn = nullptr; // lazy, persistent across decode steps

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
        conn = llama_ep_tcp_connect(host.c_str(), port, &err);
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
    if (!llama_ep_send_framev(st.conn, LLAMA_EP_MSG_REQ, parts, lens, 4)) {
        err = "send REQ failed";
        return false;
    }

    std::vector<uint8_t> payload;
    uint32_t type = 0;
    if (!llama_ep_recv_frame(st.conn, type, payload)) {
        err = "recv RESP failed";
        return false;
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

    // one reconnect+retry on transport failure; protocol errors are fatal
    for (int attempt = 0; attempt < 2; ++attempt) {
        if (!st.ensure_conn(err)) {
            if (attempt == 0) {
                continue;
            }
            return false;
        }
        if (remote_ep_roundtrip(st, il, n_tokens, n_ids, n_embd, ids, weights, hidden, out, err)) {
            return true;
        }
        st.drop_conn();
        if (err.rfind("worker ERR", 0) == 0) {
            return false; // the worker answered; reconnecting won't help
        }
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
