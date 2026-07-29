// llama-epd: MoE expert-parallel worker daemon (stage 1: TCP loopback, single layer verification).
//
// Loads a GGUF model via mmap (read-only, shared page cache), owns a range of layers
// and experts, listens on a TCP port for dispatch requests. For each REQ it computes
// the full MoE FFN (gate->silu->*up->down, mirroring build_moe_ffn in src/llama-graph.cpp)
// for the requested experts only, applies the router weights, and returns the merged
// output. No attention, no router: the master sends expert ids + weights.
//
// Modes:
//   llama-epd -m model.gguf --port 29200 --layers 3-42 [--experts 0-255] [--threads N]
//   llama-epd -m model.gguf --selftest [--selftest-layer N]   # local vs loopback diff

#include "llama-ep-transport.h"

#include "llama.h"

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "gguf.h"

#include <atomic>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#define LOG(...) fprintf(stderr, __VA_ARGS__)

// env-gated per-REQ compute timing (GGML_REMOTE_EP_DEBUG=1), default off
static bool ep_debug_enabled() {
    static const bool v = []() {
        const char * e = getenv("GGML_REMOTE_EP_DEBUG");
        return e && e[0] != '\0' && strcmp(e, "0") != 0;
    }();
    return v;
}

// env-gated startup weight prefault (GGML_EP_PREFAULT=1), default off
static bool ep_prefault_enabled() {
    static const bool v = []() {
        const char * e = getenv("GGML_EP_PREFAULT");
        return e && e[0] != '\0' && strcmp(e, "0") != 0;
    }();
    return v;
}

// ---------------------------------------------------------------------------
// model (gguf metadata + read-only mmap, tensors point into the mapping)
//
// multi-split GGUF: each split's metadata lists only its own tensors (the first
// split may have none at all). every split is opened + mmap'd; tensors are
// registered in a global name -> tensor map with data pointing into the split's
// mapping. pages are only faulted in for the experts actually requested.
// ---------------------------------------------------------------------------

struct ep_layer {
    int il = -1;

    // separate gate/up layout:
    ggml_tensor * gate = nullptr; // [n_embd, n_ff, n_expert]
    ggml_tensor * up   = nullptr; // [n_embd, n_ff, n_expert]
    // merged layout (mutually exclusive with gate/up):
    ggml_tensor * gate_up = nullptr; // [n_embd, 2*n_ff, n_expert]

    ggml_tensor * down = nullptr; // [n_ff, n_embd, n_expert]

    int64_t n_embd   = 0;
    int64_t n_ff     = 0;
    int64_t n_expert = 0;

    float clamp = 0.0f; // swiglu_clamp_exp[il], 0 = disabled
};

struct ep_shard {
    gguf_context * gguf = nullptr;
    ggml_context * ctx  = nullptr; // weight tensors (data -> this shard's mmap); freed by gguf_free

    int    fd        = -1;
    void * mmap_base = nullptr;
    size_t mmap_size = 0;

    const char * data_base = nullptr;

    ~ep_shard() {
        if (mmap_base && mmap_base != MAP_FAILED) {
            munmap(mmap_base, mmap_size);
        }
        if (fd >= 0) {
            ::close(fd);
        }
        if (gguf) {
            gguf_free(gguf); // also frees ctx
        }
    }
};

struct ep_model {
    std::vector<std::unique_ptr<ep_shard>> shards;

    std::map<std::string, ggml_tensor *> tensors; // name -> tensor (any shard)

    std::string arch;
    int n_layer = 0;

    std::map<int, ep_layer> layers; // owned MoE layers
};

static bool gguf_get_str(gguf_context * g, const char * key, std::string & out) {
    int64_t id = gguf_find_key(g, key);
    if (id < 0) {
        return false;
    }
    out = gguf_get_val_str(g, id);
    return true;
}

// open + mmap one split and register all of its tensors
static bool ep_shard_load(ep_model & m, const char * path) {
    std::unique_ptr<ep_shard> sh(new ep_shard);

    gguf_init_params iparams = {/*.no_alloc =*/ true, /*.ctx =*/ &sh->ctx};
    sh->gguf = gguf_init_from_file(path, iparams);
    if (!sh->gguf) {
        LOG("llama-epd: failed to read gguf header from %s\n", path);
        return false;
    }

    sh->fd = ::open(path, O_RDONLY);
    if (sh->fd < 0) {
        LOG("llama-epd: open %s: %s\n", path, strerror(errno));
        return false;
    }
    struct stat st;
    if (fstat(sh->fd, &st) != 0) {
        LOG("llama-epd: fstat: %s\n", strerror(errno));
        return false;
    }
    sh->mmap_size = (size_t) st.st_size;
    sh->mmap_base = mmap(nullptr, sh->mmap_size, PROT_READ, MAP_SHARED, sh->fd, 0);
    if (sh->mmap_base == MAP_FAILED) {
        LOG("llama-epd: mmap: %s\n", strerror(errno));
        sh->mmap_base = nullptr;
        return false;
    }
    // lazy page-in: the worker only touches the experts it is asked for
    sh->data_base = (const char *) sh->mmap_base + gguf_get_data_offset(sh->gguf);

    const int64_t n_tensors = gguf_get_n_tensors(sh->gguf);
    for (int64_t tid = 0; tid < n_tensors; ++tid) {
        const char * name = gguf_get_tensor_name(sh->gguf, tid);
        ggml_tensor * t = ggml_get_tensor(sh->ctx, name);
        if (!t) {
            LOG("llama-epd: tensor %s in gguf but not in ctx (%s)\n", name, path);
            return false;
        }
        t->data = const_cast<char *>(sh->data_base + gguf_get_tensor_offset(sh->gguf, tid));
        if (!m.tensors.emplace(name, t).second) {
            LOG("llama-epd: duplicate tensor name %s across splits\n", name);
            return false;
        }
    }
    LOG("llama-epd: split %s: %lld tensors, %.2f GiB mapped\n",
        path, (long long) n_tensors, sh->mmap_size / 1073741824.0);

    m.shards.push_back(std::move(sh));
    return true;
}

// probe one tensor name in the global map; returns nullptr if absent
static ggml_tensor * ep_get_tensor(ep_model & m, const char * name) {
    auto it = m.tensors.find(name);
    return it == m.tensors.end() ? nullptr : it->second;
}

static bool ep_model_load(ep_model & m, const char * path, int layer_first, int layer_last) {
    // first split (or the only file)
    if (!ep_shard_load(m, path)) {
        return false;
    }

    gguf_context * g0 = m.shards[0]->gguf;

    // discover additional splits via split.count / split.no (must load from the first split)
    {
        int64_t id = gguf_find_key(g0, "split.count");
        const uint16_t n_split = id >= 0 ? gguf_get_val_u16(g0, id) : 1;
        if (n_split > 1) {
            id = gguf_find_key(g0, "split.no");
            const uint16_t split_no = id >= 0 ? gguf_get_val_u16(g0, id) : 0;
            if (split_no != 0) {
                LOG("llama-epd: model must be loaded with the first split (got split.no = %d)\n", split_no);
                return false;
            }
            char prefix[4096];
            if (llama_split_prefix(prefix, sizeof(prefix), path, split_no, n_split) <= 0) {
                LOG("llama-epd: invalid split file name: %s\n", path);
                return false;
            }
            for (int idx = 1; idx < n_split; ++idx) {
                char spath[4096];
                if (llama_split_path(spath, sizeof(spath), prefix, idx, n_split) <= 0) {
                    LOG("llama-epd: failed to build split path %d/%d\n", idx, n_split);
                    return false;
                }
                if (!ep_shard_load(m, spath)) {
                    return false;
                }
            }
        }
    }

    if (!gguf_get_str(g0, "general.architecture", m.arch)) {
        LOG("llama-epd: missing general.architecture\n");
        return false;
    }

    {
        std::string key = m.arch + ".block_count";
        int64_t id = gguf_find_key(g0, key.c_str());
        if (id < 0) {
            LOG("llama-epd: missing %s\n", key.c_str());
            return false;
        }
        m.n_layer = (int) gguf_get_val_u32(g0, id);
    }

    // optional per-layer swiglu clamp (deepseek4)
    std::vector<float> clamps((size_t) m.n_layer, 0.0f);
    {
        std::string key = m.arch + ".swiglu_clamp_exp";
        int64_t id = gguf_find_key(g0, key.c_str());
        if (id >= 0) {
            if (gguf_get_arr_type(g0, id) != GGUF_TYPE_FLOAT32) {
                LOG("llama-epd: %s has unexpected type\n", key.c_str());
                return false;
            }
            size_t n = gguf_get_arr_n(g0, id);
            const float * v = (const float *) gguf_get_arr_data(g0, id);
            for (size_t i = 0; i < n && i < clamps.size(); ++i) {
                clamps[i] = v[i];
            }
        }
    }

    char name[128];
    for (int il = layer_first; il <= layer_last && il < m.n_layer; ++il) {
        ep_layer L;
        L.il = il;

        snprintf(name, sizeof(name), "blk.%d.ffn_gate_up_exps.weight", il);
        L.gate_up = ep_get_tensor(m, name);
        if (!L.gate_up) {
            snprintf(name, sizeof(name), "blk.%d.ffn_gate_exps.weight", il);
            L.gate = ep_get_tensor(m, name);
            snprintf(name, sizeof(name), "blk.%d.ffn_up_exps.weight", il);
            L.up = ep_get_tensor(m, name);
        }
        snprintf(name, sizeof(name), "blk.%d.ffn_down_exps.weight", il);
        L.down = ep_get_tensor(m, name);

        const bool has_ffn = L.gate_up || (L.gate && L.up);
        if (!has_ffn && !L.down) {
            continue; // dense layer, not MoE
        }
        if (!has_ffn || !L.down) {
            LOG("llama-epd: layer %d: incomplete MoE tensor set\n", il);
            return false;
        }

        ggml_tensor * ref = L.gate_up ? L.gate_up : L.gate;
        L.n_embd   = ref->ne[0];
        L.n_ff     = L.gate_up ? ref->ne[1] / 2 : ref->ne[1];
        L.n_expert = ref->ne[2];
        L.clamp    = clamps[(size_t) il];

        if (L.down->ne[0] != L.n_ff || L.down->ne[1] != L.n_embd || L.down->ne[2] != L.n_expert) {
            LOG("llama-epd: layer %d: down shape mismatch\n", il);
            return false;
        }
        if (L.gate && (L.gate->ne[1] != L.n_ff || L.up->ne[1] != L.n_ff)) {
            LOG("llama-epd: layer %d: gate/up shape mismatch\n", il);
            return false;
        }

        // expert biases are not expected for supported archs; refuse rather than compute wrong math
        snprintf(name, sizeof(name), "blk.%d.ffn_down_exps.bias", il);
        if (ep_get_tensor(m, name) != nullptr) {
            LOG("llama-epd: layer %d: expert biases not supported\n", il);
            return false;
        }

        m.layers[il] = L;
        LOG("llama-epd: layer %3d: MoE n_embd=%lld n_ff=%lld n_expert=%lld layout=%s clamp=%g types(g/u/d)=%s/%s/%s\n",
            il, (long long) L.n_embd, (long long) L.n_ff, (long long) L.n_expert,
            L.gate_up ? "merged" : "separate", (double) L.clamp,
            ggml_type_name(L.gate_up ? L.gate_up->type : L.gate->type),
            ggml_type_name(L.gate_up ? L.gate_up->type : L.up->type),
            ggml_type_name(L.down->type));
    }

    if (m.layers.empty()) {
        LOG("llama-epd: no MoE layers in range %d-%d\n", layer_first, layer_last);
        return false;
    }
    return true;
}

// fault in every page of the owned layers' expert weights at startup, so the
// first request that hits a cold expert does not stall on mmap page-in
// (observed as multi-ms compute spikes, worst for rarely-routed experts)
static void ep_prefault_weights(const ep_model & m) {
    std::vector<std::pair<const char *, size_t>> ranges;
    ranges.reserve(m.layers.size() * 3);
    for (const auto & kv : m.layers) {
        const ep_layer & L = kv.second;
        for (const ggml_tensor * t : {L.gate_up, L.gate, L.up, L.down}) {
            if (t && t->data) {
                ranges.emplace_back((const char *) t->data, ggml_nbytes(t));
            }
        }
    }

    size_t total = 0;
    for (const auto & r : ranges) {
        total += r.second;
    }

    int nth = 16;
    if (const char * e = getenv("GGML_EP_PREFAULT_THREADS")) {
        const int v = atoi(e);
        if (v > 0) {
            nth = v;
        }
    }

    LOG("llama-epd: prefault: touching %.2f GiB of expert weights (%zu tensors) with %d threads\n",
        total / 1073741824.0, ranges.size(), nth);
    const int64_t t0 = ggml_time_us();

    std::atomic<size_t> next{0};
    std::atomic<uint64_t> sink{0};
    std::vector<std::thread> pool;
    pool.reserve((size_t) nth);
    for (int i = 0; i < nth; ++i) {
        pool.emplace_back([&]() {
            uint64_t acc = 0;
            for (;;) {
                const size_t idx = next.fetch_add(1);
                if (idx >= ranges.size()) {
                    break;
                }
                const char * p = ranges[idx].first;
                const size_t n = ranges[idx].second;
                for (size_t off = 0; off < n; off += 4096) {
                    acc += *(volatile const uint8_t *) (p + off);
                }
                acc += *(volatile const uint8_t *) (p + n - 1);
            }
            sink += acc;
        });
    }
    for (auto & th : pool) {
        th.join();
    }

    LOG("llama-epd: prefault: done in %.1f s (sink %llu)\n",
        (ggml_time_us() - t0) / 1e6, (unsigned long long) sink.load());
}

// ---------------------------------------------------------------------------
// MoE FFN compute (mirrors build_moe_ffn math in src/llama-graph.cpp:1816-2175)
// ---------------------------------------------------------------------------

static bool ep_moe_ffn(
        ggml_backend_t    backend,
        const ep_layer  & L,
        int               n_tokens,
        int               n_ids,          // experts per token
        const int32_t   * ids,            // [n_tokens*n_ids]
        const float     * weights,        // [n_tokens*n_ids]
        const float     * hidden,         // [n_tokens*n_embd]
        float           * out,            // [n_tokens*n_embd]
        std::string     & err) {

    const int64_t n_embd = L.n_embd;
    const int64_t n_ff   = L.n_ff;

    ggml_init_params iparams;
    iparams.mem_size   = ggml_tensor_overhead() * 128 + ggml_graph_overhead();
    iparams.mem_buffer = nullptr;
    iparams.no_alloc   = true;

    ggml_context * ctx = ggml_init(iparams);
    if (!ctx) {
        err = "ggml_init failed";
        return false;
    }

    ggml_tensor * hidden_t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, n_tokens);
    ggml_tensor * ids_t    = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, n_ids, n_tokens);
    ggml_tensor * w_t      = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1, n_ids, n_tokens);

    ggml_tensor * cur = ggml_reshape_3d(ctx, hidden_t, n_embd, 1, n_tokens);

    ggml_tensor * gate = nullptr;
    ggml_tensor * up   = nullptr;

    if (L.gate_up) {
        // merged gate_up path (llama-graph.cpp:2023-2041)
        ggml_tensor * gu = ggml_mul_mat_id(ctx, L.gate_up, cur, ids_t); // [2*n_ff, n_ids, n_tokens]
        gate = ggml_view_3d(ctx, gu, n_ff, n_ids, n_tokens, gu->nb[1], gu->nb[2], 0);
        up   = ggml_view_3d(ctx, gu, n_ff, n_ids, n_tokens, gu->nb[1], gu->nb[2], n_ff * gu->nb[0]);
    } else {
        // separate path (llama-graph.cpp:2042-2071)
        up   = ggml_mul_mat_id(ctx, L.up,   cur, ids_t); // [n_ff, n_ids, n_tokens]
        gate = ggml_mul_mat_id(ctx, L.gate, cur, ids_t);
    }

    // deepseek4 clamped swiglu (llama-graph.cpp:2078-2098)
    if (L.clamp > 1e-6f) {
        up   = ggml_clamp(ctx, up, -L.clamp, L.clamp);
        gate = ggml_clamp(ctx, gate, -INFINITY, L.clamp);
    }

    cur = ggml_swiglu_split(ctx, gate, up); // silu(gate) * up -> [n_ff, n_ids, n_tokens]

    ggml_tensor * experts = ggml_mul_mat_id(ctx, L.down, cur, ids_t); // [n_embd, n_ids, n_tokens]
    experts = ggml_mul(ctx, experts, w_t);

    // sum over the n_ids (expert slot) dimension via views + adds (llama-graph.cpp:2165-2183)
    ggml_tensor * sum = ggml_view_2d(ctx, experts, n_embd, n_tokens, experts->nb[2], 0);
    for (int i = 1; i < n_ids; ++i) {
        ggml_tensor * v = ggml_view_2d(ctx, experts, n_embd, n_tokens, experts->nb[2], i * experts->nb[1]);
        sum = ggml_add(ctx, sum, v);
    }

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, sum);

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buf) {
        err = "failed to allocate compute tensors";
        ggml_free(ctx);
        return false;
    }

    ggml_backend_tensor_set(hidden_t, hidden,  0, (size_t) n_tokens * n_embd * sizeof(float));
    ggml_backend_tensor_set(ids_t,    ids,     0, (size_t) n_tokens * n_ids  * sizeof(int32_t));
    ggml_backend_tensor_set(w_t,      weights, 0, (size_t) n_tokens * n_ids  * sizeof(float));

    if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
        err = "graph compute failed";
        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
        return false;
    }

    ggml_backend_tensor_get(sum, out, 0, (size_t) n_tokens * n_embd * sizeof(float));

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return true;
}

// ---------------------------------------------------------------------------
// server
// ---------------------------------------------------------------------------

struct ep_config {
    int layer_first  = 0;
    int layer_last   = 1 << 30;
    int expert_first = 0; // half-open [first, last)
    int expert_last  = 1 << 30;
    int n_threads    = 8;
};

static bool ep_send_err(llama_ep_transport * t, int32_t code, const std::string & msg) {
    std::vector<uint8_t> payload(sizeof(int32_t) + msg.size());
    memcpy(payload.data(), &code, sizeof(code));
    memcpy(payload.data() + sizeof(code), msg.data(), msg.size());
    return llama_ep_send_frame(t, LLAMA_EP_MSG_ERR, payload.data(), payload.size());
}

static bool ep_handle_req(
        llama_ep_transport * t,
        ggml_backend_t       backend,
        const ep_model     & m,
        const ep_config    & cfg,
        const uint8_t      * payload,
        size_t               payload_len) {

    if (payload_len < sizeof(llama_ep_req_header)) {
        return ep_send_err(t, LLAMA_EP_ERR_BAD_SHAPE, "short REQ");
    }

    llama_ep_req_header hdr;
    memcpy(&hdr, payload, sizeof(hdr));

    const int64_t n_tokens = hdr.n_tokens;
    const int64_t n_ids    = hdr.n_ids;

    auto it = m.layers.find(hdr.layer);
    if (it == m.layers.end()) {
        return ep_send_err(t, LLAMA_EP_ERR_BAD_LAYER, "layer " + std::to_string(hdr.layer) + " not owned by this worker");
    }
    const ep_layer & L = it->second;

    if (hdr.n_embd != (int32_t) L.n_embd) {
        return ep_send_err(t, LLAMA_EP_ERR_BAD_SHAPE, "n_embd mismatch");
    }
    if (n_tokens < 1 || n_tokens > 65536 || n_ids < 1 || n_ids > L.n_expert) {
        return ep_send_err(t, LLAMA_EP_ERR_BAD_SHAPE, "bad n_tokens/n_ids");
    }

    const size_t n_sel   = (size_t) n_tokens * n_ids;
    const size_t need    = sizeof(hdr) + n_sel * sizeof(int32_t) + n_sel * sizeof(float)
                         + (size_t) n_tokens * L.n_embd * sizeof(float);
    if (payload_len != need) {
        return ep_send_err(t, LLAMA_EP_ERR_BAD_SHAPE, "REQ payload length mismatch");
    }

    const int32_t * ids     = (const int32_t *) (payload + sizeof(hdr));
    const float   * weights = (const float *) (ids + n_sel);
    const float   * hidden  = weights + n_sel;

    const int e_first = cfg.expert_first;
    const int e_last  = std::min<int64_t>(cfg.expert_last, L.n_expert);
    for (size_t i = 0; i < n_sel; ++i) {
        if (ids[i] < e_first || ids[i] >= e_last) {
            return ep_send_err(t, LLAMA_EP_ERR_BAD_EXPERT,
                "expert " + std::to_string(ids[i]) + " outside [" + std::to_string(e_first) +
                ", " + std::to_string(e_last) + ")");
        }
    }

    std::vector<float> out((size_t) n_tokens * L.n_embd);
    std::string err;
    const int64_t t0 = ep_debug_enabled() ? ggml_time_us() : 0;
    if (!ep_moe_ffn(backend, L, (int) n_tokens, (int) n_ids, ids, weights, hidden, out.data(), err)) {
        return ep_send_err(t, LLAMA_EP_ERR_COMPUTE, err);
    }
    if (ep_debug_enabled()) {
        LOG("llama-epd: [ep-debug] layer %d n_tokens=%d compute %.3f ms\n",
            hdr.layer, (int) n_tokens, (ggml_time_us() - t0) / 1000.0);
    }

    llama_ep_resp_header rhdr;
    rhdr.n_tokens = hdr.n_tokens;
    rhdr.n_embd   = hdr.n_embd;

    const void * parts[2] = {&rhdr, out.data()};
    const size_t lens[2]  = {sizeof(rhdr), out.size() * sizeof(float)};
    return llama_ep_send_framev(t, LLAMA_EP_MSG_RESP, parts, lens, 2);
}

// serve frames on one connection until EOF
static void ep_serve_connection(
        llama_ep_transport * t,
        ggml_backend_t       backend,
        const ep_model     & m,
        const ep_config    & cfg) {
    std::vector<uint8_t> payload;
    for (;;) {
        uint32_t type = 0;
        if (!llama_ep_recv_frame(t, type, payload)) {
            break;
        }
        if (type != LLAMA_EP_MSG_REQ) {
            ep_send_err(t, LLAMA_EP_ERR_GENERIC, "expected REQ");
            continue;
        }
        if (!ep_handle_req(t, backend, m, cfg, payload.data(), payload.size())) {
            LOG("llama-epd: failed to handle REQ\n");
            break;
        }
    }
}

// ---------------------------------------------------------------------------
// client helper (used by --selftest; master side will use the same code path)
// ---------------------------------------------------------------------------

static bool ep_client_moe_ffn(
        llama_ep_transport * t,
        int                  layer,
        int                  n_tokens,
        int                  n_ids,
        int                  n_embd,
        const int32_t      * ids,
        const float        * weights,
        const float        * hidden,
        float              * out,
        std::string        & err) {

    llama_ep_req_header hdr = {layer, n_tokens, n_ids, n_embd};
    const size_t n_sel = (size_t) n_tokens * n_ids;

    const void * parts[4] = {&hdr, ids, weights, hidden};
    const size_t lens[4]  = {
        sizeof(hdr),
        n_sel * sizeof(int32_t),
        n_sel * sizeof(float),
        (size_t) n_tokens * n_embd * sizeof(float),
    };
    if (!llama_ep_send_framev(t, LLAMA_EP_MSG_REQ, parts, lens, 4)) {
        err = "send REQ failed";
        return false;
    }

    std::vector<uint8_t> payload;
    uint32_t type = 0;
    if (!llama_ep_recv_frame(t, type, payload)) {
        err = "recv RESP failed";
        return false;
    }
    if (type == LLAMA_EP_MSG_ERR) {
        int32_t code = 0;
        if (payload.size() >= sizeof(code)) {
            memcpy(&code, payload.data(), sizeof(code));
        }
        err = "worker ERR " + std::to_string(code) + ": " +
              std::string((const char *) payload.data() + sizeof(code), payload.size() - sizeof(code));
        return false;
    }
    if (type != LLAMA_EP_MSG_RESP || payload.size() != sizeof(llama_ep_resp_header) + (size_t) n_tokens * n_embd * sizeof(float)) {
        err = "bad RESP";
        return false;
    }
    memcpy(out, payload.data() + sizeof(llama_ep_resp_header), (size_t) n_tokens * n_embd * sizeof(float));
    return true;
}

// ---------------------------------------------------------------------------
// selftest: (a) local direct compute vs (b) loopback TCP worker
// ---------------------------------------------------------------------------

// deterministic rng (xorshift64*), no <random> implementation dependence
struct ep_rng {
    uint64_t s;
    explicit ep_rng(uint64_t seed) : s(seed ? seed : 1) {}
    uint64_t next_u64() {
        s ^= s >> 12; s ^= s << 25; s ^= s >> 27;
        return s * 0x2545F4914F6CDD1DULL;
    }
    float next_f() { // (-1, 1)
        return 2.0f * ((next_u64() >> 40) / 16777216.0f) - 1.0f;
    }
    uint32_t next_below(uint32_t n) { return (uint32_t) (next_u64() % n); }
};

static int ep_selftest(ep_model & m, const ep_config & cfg, int layer, int n_tokens, int n_ids) {
    if (layer < 0) {
        layer = m.layers.begin()->first;
    }
    auto it = m.layers.find(layer);
    if (it == m.layers.end()) {
        LOG("selftest: layer %d is not an owned MoE layer\n", layer);
        return 1;
    }
    const ep_layer & L = it->second;

    if (n_ids > L.n_expert) {
        n_ids = (int) L.n_expert;
    }

    const int n_embd = (int) L.n_embd;

    ggml_backend_t backend = ggml_backend_cpu_init();
    if (!backend) {
        LOG("selftest: failed to init CPU backend\n");
        return 1;
    }
    ggml_backend_cpu_set_n_threads(backend, cfg.n_threads);

    // fixed-seed inputs
    ep_rng rng(0xC0FFEE);
    std::vector<int32_t> ids((size_t) n_tokens * n_ids);
    std::vector<float>   weights((size_t) n_tokens * n_ids);
    std::vector<float>   hidden((size_t) n_tokens * n_embd);

    const int e_first = cfg.expert_first;
    const int e_last  = (int) std::min<int64_t>(cfg.expert_last, L.n_expert);
    for (int t = 0; t < n_tokens; ++t) {
        // sample n_ids distinct experts from [e_first, e_last)
        std::vector<int> pool;
        for (int e = e_first; e < e_last; ++e) {
            pool.push_back(e);
        }
        for (int k = 0; k < n_ids; ++k) {
            int j = k + (int) rng.next_below((uint32_t) (pool.size() - k));
            std::swap(pool[k], pool[j]);
            ids[(size_t) t * n_ids + k] = pool[k];
        }
        float sum = 0.0f;
        for (int k = 0; k < n_ids; ++k) {
            float w = 0.05f + 0.95f * (rng.next_f() + 1.0f) * 0.5f;
            weights[(size_t) t * n_ids + k] = w;
            sum += w;
        }
        for (int k = 0; k < n_ids; ++k) {
            weights[(size_t) t * n_ids + k] /= sum;
        }
        for (int e = 0; e < n_embd; ++e) {
            hidden[(size_t) t * n_embd + e] = rng.next_f();
        }
    }

    // (a) local direct compute
    std::vector<float> out_a((size_t) n_tokens * n_embd);
    {
        std::string err;
        if (!ep_moe_ffn(backend, L, n_tokens, n_ids, ids.data(), weights.data(), hidden.data(), out_a.data(), err)) {
            LOG("selftest: local compute failed: %s\n", err.c_str());
            return 1;
        }
    }

    // (b) loopback: server thread + client
    std::string lerr;
    llama_ep_listener * listener = llama_ep_tcp_listen("127.0.0.1", 0, &lerr);
    if (!listener) {
        LOG("selftest: listen failed: %s\n", lerr.c_str());
        return 1;
    }
    const int port = llama_ep_tcp_listener_port(listener);

    std::thread server_thread([&]() {
        llama_ep_transport conn;
        if (listener->ops.accept(listener->ctx, &conn)) {
            ep_serve_connection(&conn, backend, m, cfg);
            conn.ops.close(conn.ctx);
        }
        listener->ops.close(listener->ctx);
    });

    std::vector<float> out_b((size_t) n_tokens * n_embd);
    {
        std::string err;
        llama_ep_transport * cli = llama_ep_tcp_connect("127.0.0.1", port, &err);
        if (!cli) {
            LOG("selftest: connect failed: %s\n", err.c_str());
            server_thread.join();
            return 1;
        }
        bool ok = ep_client_moe_ffn(cli, layer, n_tokens, n_ids, n_embd,
                ids.data(), weights.data(), hidden.data(), out_b.data(), err);
        cli->ops.close(cli->ctx);
        delete cli;
        if (!ok) {
            LOG("selftest: loopback request failed: %s\n", err.c_str());
            server_thread.join();
            return 1;
        }
    }
    server_thread.join();

    // compare
    double max_abs = 0.0, max_rel = 0.0, norm = 0.0;
    for (size_t i = 0; i < out_a.size(); ++i) {
        double d = fabs((double) out_a[i] - (double) out_b[i]);
        max_abs = std::max(max_abs, d);
        max_rel = std::max(max_rel, d / (fabs((double) out_a[i]) + 1e-12));
        norm += (double) out_a[i] * out_a[i];
    }
    norm = sqrt(norm);

    LOG("selftest: arch=%s layer=%d n_tokens=%d n_ids=%d n_embd=%d n_expert=%lld layout=%s clamp=%g\n",
        m.arch.c_str(), layer, n_tokens, n_ids, n_embd, (long long) L.n_expert,
        L.gate_up ? "merged" : "separate", (double) L.clamp);
    LOG("selftest: |out|=%.6f  max_abs_diff=%.6g  max_rel_diff=%.6g\n", norm, max_abs, max_rel);

    const bool pass = max_abs < 1e-5;
    LOG("selftest: %s\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

static void ep_usage(const char * argv0) {
    fprintf(stderr,
        "usage: %s -m model.gguf [options]\n"
        "\n"
        "  -m, --model PATH       gguf model file (required)\n"
        "  --port N               listen port (default 29200)\n"
        "  --layers A-B           owned layer range (default: all)\n"
        "  --experts A-B          owned expert range, half-open [A,B) (default: all)\n"
        "  -t, --threads N        compute threads (default 8)\n"
        "  --selftest             local vs loopback numerical check, then exit\n"
        "  --selftest-layer N     layer for selftest (default: first owned MoE layer)\n"
        "  --selftest-tokens N    tokens for selftest (default 4)\n"
        "\n",
        argv0);
}

static bool parse_range(const char * s, int & a, int & b) {
    int x, y;
    if (sscanf(s, "%d-%d", &x, &y) == 2) {
        a = x; b = y;
        return x <= y;
    }
    if (sscanf(s, "%d", &x) == 1) {
        a = x; b = x;
        return true;
    }
    return false;
}

int main(int argc, char ** argv) {
    std::string model_path;
    ep_config cfg;
    int port = 29200;
    bool selftest = false;
    int selftest_layer = -1;
    int selftest_tokens = 4;
    int layer_first = 0, layer_last = 1 << 30;
    bool have_layers = false;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&](const char * what) -> const char * {
            if (++i >= argc) {
                LOG("llama-epd: missing value for %s\n", what);
                exit(1);
            }
            return argv[i];
        };
        if (a == "-m" || a == "--model") {
            model_path = next(a.c_str());
        } else if (a == "--port") {
            port = atoi(next(a.c_str()));
        } else if (a == "--layers") {
            if (!parse_range(next(a.c_str()), layer_first, layer_last)) {
                LOG("llama-epd: bad --layers range\n");
                return 1;
            }
            have_layers = true;
        } else if (a == "--experts") {
            int e0, e1;
            if (!parse_range(next(a.c_str()), e0, e1)) {
                LOG("llama-epd: bad --experts range\n");
                return 1;
            }
            cfg.expert_first = e0;
            cfg.expert_last = e1 + 1; // CLI is inclusive, config is half-open
        } else if (a == "-t" || a == "--threads") {
            cfg.n_threads = atoi(next(a.c_str()));
        } else if (a == "--selftest") {
            selftest = true;
        } else if (a == "--selftest-layer") {
            selftest_layer = atoi(next(a.c_str()));
        } else if (a == "--selftest-tokens") {
            selftest_tokens = atoi(next(a.c_str()));
        } else if (a == "-h" || a == "--help") {
            ep_usage(argv[0]);
            return 0;
        } else {
            LOG("llama-epd: unknown option %s\n", a.c_str());
            ep_usage(argv[0]);
            return 1;
        }
    }

    if (model_path.empty()) {
        ep_usage(argv[0]);
        return 1;
    }
    (void) have_layers;

    cfg.layer_first = layer_first;
    cfg.layer_last  = layer_last;

    ggml_backend_load_all(); // no-op for static builds, keeps dl builds working

    ep_model m;
    if (!ep_model_load(m, model_path.c_str(), cfg.layer_first, cfg.layer_last)) {
        return 1;
    }
    LOG("llama-epd: arch=%s n_layer=%d, owning %zu MoE layers, experts [%d, %s)\n",
        m.arch.c_str(), m.n_layer, m.layers.size(), cfg.expert_first,
        cfg.expert_last == (1 << 30) ? "n_expert" : std::to_string(cfg.expert_last).c_str());

    if (selftest) {
        return ep_selftest(m, cfg, selftest_layer, selftest_tokens, 6);
    }

    if (ep_prefault_enabled()) {
        ep_prefault_weights(m);
    }

    ggml_backend_t backend = ggml_backend_cpu_init();
    if (!backend) {
        LOG("llama-epd: failed to init CPU backend\n");
        return 1;
    }
    ggml_backend_cpu_set_n_threads(backend, cfg.n_threads);

    std::string err;
    llama_ep_listener * listener = nullptr;
    bool rdma_mode = false;
#ifdef LLAMA_EP_HAVE_RDMA
    if (llama_ep_rdma_requested()) {
        listener = llama_ep_rdma_listen(nullptr, port, &err);
        if (listener) {
            rdma_mode = true;
        } else {
            LOG("llama-epd: RDMA listen failed (%s), falling back to TCP\n", err.c_str());
            err.clear();
        }
    }
#endif
    if (!listener) {
        listener = llama_ep_tcp_listen(nullptr, port, &err);
    }
    if (!listener) {
        LOG("llama-epd: listen failed: %s\n", err.c_str());
        return 1;
    }
    int bound_port = 0;
    if (rdma_mode) {
#ifdef LLAMA_EP_HAVE_RDMA
        bound_port = llama_ep_rdma_listener_port(listener);
#endif
    } else {
        bound_port = llama_ep_tcp_listener_port(listener);
    }
    LOG("llama-epd: listening on port %d (%d threads)%s\n",
        bound_port, cfg.n_threads, rdma_mode ? " [rdma]" : "");

    for (;;) {
        llama_ep_transport conn;
        if (!listener->ops.accept(listener->ctx, &conn)) {
            LOG("llama-epd: accept failed\n");
            continue;
        }
        LOG("llama-epd: client connected\n");
        ep_serve_connection(&conn, backend, m, cfg);
        conn.ops.close(conn.ctx);
        LOG("llama-epd: client disconnected\n");
    }
    return 0;
}
