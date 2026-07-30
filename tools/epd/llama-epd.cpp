// llama-epd: MoE expert-parallel worker daemon (stage 1: TCP loopback, single layer verification).
//
// Loads a GGUF model via mmap (read-only, shared page cache), owns a range of layers
// and experts, listens on a TCP port for dispatch requests. For each REQ it computes
// the full MoE FFN (gate->silu->*up->down, mirroring build_moe_ffn in src/llama-graph.cpp)
// for the requested experts only, applies the router weights, and returns the merged
// output. No attention, no router: the master sends expert ids + weights.
//
// Modes:
//   llama-epd -m model.gguf --port 29200 --layers 3-42 [--experts 0-255] [--threads N] [--no-autotune] [--no-mmap]
//   llama-epd -m model.gguf --selftest [--selftest-layer N]   # local vs loopback diff
//
// Without -t the worker autotunes the compute thread count at startup (after the
// model is mapped and layers claimed, before listen): it times the expert FFN on
// representative owned layers over a {16,24,32,48,physical cores} ladder and picks
// the knee point (smallest count with < 3% marginal gain). Disable with
// --no-autotune or GGML_EPD_AUTOTUNE=0.
//
// --no-mmap replaces the default read-only MAP_SHARED mapping with a one-time
// sequential pread of the owned layers' expert weights into anonymous memory:
// slow start (full read), RSS = owned weights permanently resident, zero page-in
// stalls, immune to page-cache eviction (slow-disk setups where mmap page-in
// dominates). GGML_EP_PREFAULT is skipped (meaningless) when --no-mmap is on.

#include "llama-ep-transport.h"

#include "llama.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "gguf.h"

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iterator>
#include <map>
#include <memory>
#include <numeric>
#include <set>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <fcntl.h>
#include <pthread.h>
#include <sched.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/syscall.h>
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

    // --no-mmap: per-tensor anonymous buffers holding this shard's owned weights
    std::vector<void *> load_bufs;

    ~ep_shard() {
        for (void * p : load_bufs) {
            free(p);
        }
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

    // --no-mmap: name -> (shard, absolute file offset of tensor data)
    std::map<std::string, std::pair<ep_shard *, uint64_t>> tensor_src;

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

// open one split and register all of its tensors; mmap mode points tensor data
// into a read-only shared mapping, no-mmap mode records file offsets for the
// post-claiming bulk read (ep_nommap_load_weights)
static bool ep_shard_load(ep_model & m, const char * path, bool no_mmap) {
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

    if (!no_mmap) {
        sh->mmap_base = mmap(nullptr, sh->mmap_size, PROT_READ, MAP_SHARED, sh->fd, 0);
        if (sh->mmap_base == MAP_FAILED) {
            LOG("llama-epd: mmap: %s\n", strerror(errno));
            sh->mmap_base = nullptr;
            return false;
        }
        // lazy page-in: the worker only touches the experts it is asked for
        sh->data_base = (const char *) sh->mmap_base + gguf_get_data_offset(sh->gguf);
    }

    const uint64_t data_off = gguf_get_data_offset(sh->gguf);
    const int64_t n_tensors = gguf_get_n_tensors(sh->gguf);
    for (int64_t tid = 0; tid < n_tensors; ++tid) {
        const char * name = gguf_get_tensor_name(sh->gguf, tid);
        ggml_tensor * t = ggml_get_tensor(sh->ctx, name);
        if (!t) {
            LOG("llama-epd: tensor %s in gguf but not in ctx (%s)\n", name, path);
            return false;
        }
        if (no_mmap) {
            // data stays nullptr until the owned tensors are read in after layer claiming
            m.tensor_src.emplace(name, std::make_pair(sh.get(), data_off + gguf_get_tensor_offset(sh->gguf, tid)));
        } else {
            t->data = const_cast<char *>(sh->data_base + gguf_get_tensor_offset(sh->gguf, tid));
        }
        if (!m.tensors.emplace(name, t).second) {
            LOG("llama-epd: duplicate tensor name %s across splits\n", name);
            return false;
        }
    }
    LOG("llama-epd: split %s: %lld tensors, %.2f GiB %s\n",
        path, (long long) n_tensors, sh->mmap_size / 1073741824.0,
        no_mmap ? "to be read (--no-mmap)" : "mapped");

    m.shards.push_back(std::move(sh));
    return true;
}

// --no-mmap: read every owned tensor's byte range into an anonymous buffer
// (sequential pread per tensor; tensors are stored in file order, so the sweep
// is near-sequential). Unlike mmap the pages cannot be evicted or re-faulted.
static bool ep_nommap_load_weights(ep_model & m) {
    size_t total = 0;
    for (const auto & kv : m.layers) {
        for (const ggml_tensor * t : {kv.second.gate_up, kv.second.gate, kv.second.up, kv.second.down}) {
            if (t) {
                total += ggml_nbytes(t);
            }
        }
    }
    LOG("llama-epd: no-mmap: reading %.2f GiB of owned expert weights into anonymous memory\n",
        total / 1073741824.0);
    const int64_t t0 = ggml_time_us();

    for (const auto & kv : m.layers) {
        for (ggml_tensor * t : {kv.second.gate_up, kv.second.gate, kv.second.up, kv.second.down}) {
            if (!t) {
                continue;
            }
            auto it = m.tensor_src.find(t->name);
            if (it == m.tensor_src.end()) {
                LOG("llama-epd: no-mmap: no source for tensor %s\n", t->name);
                return false;
            }
            ep_shard * sh = it->second.first;
            const uint64_t off = it->second.second;
            if (off % 32 != 0) {
                LOG("llama-epd: no-mmap: tensor %s file offset %llu not 32-byte aligned\n",
                    t->name, (unsigned long long) off);
                return false;
            }
            const size_t n = ggml_nbytes(t);
            void * buf = nullptr;
            if (posix_memalign(&buf, 64, n) != 0) {
                LOG("llama-epd: no-mmap: posix_memalign %.2f GiB failed\n", n / 1073741824.0);
                return false;
            }
            size_t done = 0;
            while (done < n) {
                const ssize_t r = pread(sh->fd, (char *) buf + done, std::min<size_t>(n - done, 32 << 20), off + done);
                if (r <= 0) {
                    LOG("llama-epd: no-mmap: pread %s: %s\n", t->name, r == 0 ? "unexpected EOF" : strerror(errno));
                    free(buf);
                    return false;
                }
                done += (size_t) r;
            }
            t->data = buf;
            sh->load_bufs.push_back(buf);
        }
    }

    LOG("llama-epd: no-mmap: done in %.1f s (%.2f GB/s)\n",
        (ggml_time_us() - t0) / 1e6, total / 1073741824.0 / ((ggml_time_us() - t0) / 1e6));
    return true;
}

// probe one tensor name in the global map; returns nullptr if absent
static ggml_tensor * ep_get_tensor(ep_model & m, const char * name) {
    auto it = m.tensors.find(name);
    return it == m.tensors.end() ? nullptr : it->second;
}

static bool ep_model_load(ep_model & m, const char * path, int layer_first, int layer_last, bool no_mmap) {
    // first split (or the only file)
    if (!ep_shard_load(m, path, no_mmap)) {
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
                if (!ep_shard_load(m, spath, no_mmap)) {
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
    if (no_mmap && !ep_nommap_load_weights(m)) {
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
        ggml_gallocr_t    gallocr,
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

    // the gallocr keeps one grow-only buffer across requests: no per-REQ
    // alloc/free and no repeated first-touch page faults on the intermediates
    if (!ggml_gallocr_alloc_graph(gallocr, gf)) {
        err = "failed to allocate compute tensors";
        ggml_free(ctx);
        return false;
    }

    ggml_backend_tensor_set(hidden_t, hidden,  0, (size_t) n_tokens * n_embd * sizeof(float));
    ggml_backend_tensor_set(ids_t,    ids,     0, (size_t) n_tokens * n_ids  * sizeof(int32_t));
    ggml_backend_tensor_set(w_t,      weights, 0, (size_t) n_tokens * n_ids  * sizeof(float));

    if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
        err = "graph compute failed";
        ggml_free(ctx);
        return false;
    }

    ggml_backend_tensor_get(sum, out, 0, (size_t) n_tokens * n_embd * sizeof(float));

    ggml_free(ctx);
    return true;
}

// ---------------------------------------------------------------------------
// startup thread autotune (only when -t was not given explicitly)
//
// The expert FFN is memory-bandwidth bound: beyond the saturation point extra
// threads only add barrier overhead (measured: 32 threads already saturate the
// slave's ~174 GB/s; going past the physical core count wrecks compute via
// ggml barrier contention). The optimum shifts with machine / layer count, so
// probe a small candidate ladder at startup on real owned layers and pick the
// knee point: the smallest thread count whose next step gains < 3%.
// ---------------------------------------------------------------------------

struct ep_config {
    int layer_first  = 0;
    int layer_last   = 1 << 30;
    int expert_first = 0; // half-open [first, last)
    int expert_last  = 1 << 30;
    int n_threads    = 8;
};

// env-gated autotune (GGML_EPD_AUTOTUNE=0 to disable), default on
static bool ep_autotune_enabled() {
    static const bool v = []() {
        const char * e = getenv("GGML_EPD_AUTOTUNE");
        return !(e && e[0] != '\0' && strcmp(e, "0") == 0);
    }();
    return v;
}

// count unique (physical_package_id, core_id) pairs from sysfs topology so
// hyper-thread siblings are not counted twice; fall back to online cpu count
static int ep_physical_cores() {
    std::set<std::pair<int, int>> cores;
    for (int cpu = 0; cpu < 4096; ++cpu) {
        char path[128];
        snprintf(path, sizeof(path), "/sys/devices/system/cpu/cpu%d/topology/core_id", cpu);
        FILE * f = fopen(path, "r");
        if (!f) {
            if (cpu > 0) {
                break; // cpu numbering is contiguous
            }
            continue;
        }
        int core_id = -1;
        if (fscanf(f, "%d", &core_id) != 1) {
            fclose(f);
            continue;
        }
        fclose(f);
        int pkg_id = 0;
        snprintf(path, sizeof(path), "/sys/devices/system/cpu/cpu%d/topology/physical_package_id", cpu);
        f = fopen(path, "r");
        if (f) {
            if (fscanf(f, "%d", &pkg_id) != 1) {
                pkg_id = 0;
            }
            fclose(f);
        }
        cores.emplace(pkg_id, core_id);
    }
    if (!cores.empty()) {
        return (int) cores.size();
    }
    const long n = sysconf(_SC_NPROCESSORS_ONLN);
    return n > 0 ? (int) n : 8;
}

static double ep_median_ms(std::vector<double> & v) {
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

// returns the tuned thread count, or -1 on failure (caller keeps its default)
static int ep_autotune_threads(ggml_backend_t backend, ggml_gallocr_t gallocr, const ep_model & m, const ep_config & cfg) {
    const int n_phys = ep_physical_cores();

    // candidate ladder + physical core count, clamped to [1, n_phys]
    std::set<int> uniq;
    for (int c : {16, 24, 32, 48, n_phys}) {
        uniq.insert(std::max(1, std::min(c, n_phys)));
    }
    std::vector<int> cands(uniq.begin(), uniq.end());
    if (cands.size() == 1) {
        return cands[0];
    }

    // representative layers: first + middle owned
    std::vector<const ep_layer *> layers;
    layers.push_back(&m.layers.begin()->second);
    {
        auto mid = m.layers.begin();
        std::advance(mid, m.layers.size() / 2);
        if (mid->second.il != layers[0]->il) {
            layers.push_back(&mid->second);
        }
    }

    // router top-k from gguf metadata (decode shape: n_tokens = 1)
    int n_ids = 6;
    {
        const std::string key = m.arch + ".expert_used_count";
        const int64_t id = gguf_find_key(m.shards[0]->gguf, key.c_str());
        if (id >= 0) {
            n_ids = (int) gguf_get_val_u32(m.shards[0]->gguf, id);
        }
    }

    const int warmup = 2;
    const int iters  = 5;

    LOG("llama-epd: autotune: %d physical cores, %zu probe layers, n_ids=%d, candidates:",
        n_phys, layers.size(), n_ids);
    for (int c : cands) {
        LOG(" %d", c);
    }
    LOG("\n");

    std::vector<double> med(cands.size());
    const int64_t t_all = ggml_time_us();
    for (size_t ci = 0; ci < cands.size(); ++ci) {
        ggml_backend_cpu_set_n_threads(backend, cands[ci]); // same path as serving
        std::vector<double> times;
        for (int it = 0; it < warmup + iters; ++it) {
            const int64_t t0 = ggml_time_us();
            for (const ep_layer * L : layers) {
                // dummy decode-shaped inputs: 1 token, n_ids distinct owned experts
                const int e_first = cfg.expert_first;
                const int e_last  = (int) std::min<int64_t>(cfg.expert_last, L->n_expert);
                const int k = std::max(1, std::min(n_ids, e_last - e_first));
                std::vector<int32_t> ids((size_t) k);
                for (int j = 0; j < k; ++j) {
                    ids[(size_t) j] = e_first + (int) (((int64_t) j * (e_last - e_first)) / k);
                }
                std::vector<float> weights((size_t) k, 1.0f / k);
                std::vector<float> hidden((size_t) L->n_embd, 1e-3f);
                std::vector<float> out((size_t) L->n_embd);
                std::string err;
                if (!ep_moe_ffn(backend, gallocr, *L, 1, k, ids.data(), weights.data(), hidden.data(), out.data(), err)) {
                    LOG("llama-epd: autotune: compute failed: %s\n", err.c_str());
                    return -1;
                }
            }
            if (it >= warmup) {
                times.push_back((ggml_time_us() - t0) / 1000.0);
            }
        }
        med[ci] = ep_median_ms(times);
        LOG("llama-epd: autotune: threads=%3d  %.3f ms/iter (%zu layers)\n",
            cands[ci], med[ci], layers.size());
    }

    // knee point: smallest thread count whose next candidate gains < 3%
    size_t best = 0;
    while (best + 1 < cands.size() && (med[best] - med[best + 1]) / med[best] >= 0.03) {
        ++best;
    }
    LOG("llama-epd: autotune: selected threads=%d (%.3f ms/iter, probed in %.1f s)\n",
        cands[best], med[best], (ggml_time_us() - t_all) / 1e6);
    return cands[best];
}

// ---------------------------------------------------------------------------
// NUMA placement policy (GGML_EPD_NUMA=off|interleave|weighted, default off)
//
// The expert FFN is memory-bandwidth bound, so which NUMA nodes the weight
// pages land on matters. off keeps the historical behavior: mmap mode scatters
// page-cache pages by first-touch, --no-mmap faults every anonymous page on
// the loading thread's node. interleave applies MPOL_INTERLEAVE over all
// online nodes; weighted applies MPOL_WEIGHTED_INTERLEAVE (kernel >= 6.9)
// with per-node weights from, in priority order:
//   1. GGML_EPD_NUMA_WEIGHT (e.g. "2:3" or "2,3", one value per online node)
//   2. a startup bandwidth probe (per-node pinned streaming read, ~150 ms/node)
//   3. the weights already in /sys/kernel/mm/mempolicy/weighted_interleave/
//      (used as-is when the sysfs write is not permitted — no root needed if
//      the administrator preconfigured them)
// The policy is set before any weight allocation/first-touch, so both the
// --no-mmap pread buffers and later mmap page-ins follow it. All failures
// degrade to a warning + fallback, never a refused start.
// ---------------------------------------------------------------------------

#ifndef MPOL_BIND
#define MPOL_BIND 2
#endif
#ifndef MPOL_INTERLEAVE
#define MPOL_INTERLEAVE 3
#endif
#ifndef MPOL_WEIGHTED_INTERLEAVE
#define MPOL_WEIGHTED_INTERLEAVE 6
#endif

// parse a sysfs id list ("0-3,8-11") into individual ids
static std::vector<int> ep_parse_id_list(const std::string & s) {
    std::vector<int> out;
    size_t i = 0;
    while (i < s.size()) {
        int a = 0, b = 0, n = 0;
        if (sscanf(s.c_str() + i, "%d-%d%n", &a, &b, &n) == 2 && a <= b) {
            for (int v = a; v <= b; ++v) {
                out.push_back(v);
            }
        } else if (sscanf(s.c_str() + i, "%d%n", &a, &n) == 1) {
            out.push_back(a);
        } else {
            break;
        }
        i += (size_t) n;
        if (i < s.size() && s[i] == ',') {
            ++i;
        }
    }
    return out;
}

static bool ep_read_first_line(const char * path, std::string & out) {
    FILE * f = fopen(path, "r");
    if (!f) {
        return false;
    }
    char buf[256];
    const bool ok = fgets(buf, sizeof(buf), f) != nullptr;
    fclose(f);
    if (ok) {
        out = buf;
        while (!out.empty() && (out.back() == '\n' || out.back() == ' ')) {
            out.pop_back();
        }
    }
    return ok;
}

static std::vector<int> ep_numa_online_nodes() {
    std::string s;
    if (ep_read_first_line("/sys/devices/system/node/online", s)) {
        std::vector<int> v = ep_parse_id_list(s);
        if (!v.empty()) {
            return v;
        }
    }
    return {0}; // UMA or unreadable sysfs: single node, policy is a no-op
}

static std::vector<int> ep_numa_node_cpus(int node) {
    char path[128];
    snprintf(path, sizeof(path), "/sys/devices/system/node/node%d/cpulist", node);
    std::string s;
    if (!ep_read_first_line(path, s)) {
        return {};
    }
    return ep_parse_id_list(s);
}

// streaming-read bandwidth probe for one node: a 1 GiB anonymous buffer is
// mbind(MPOL_BIND)'d to the node, faulted, then read by T = min(8, n_cpus)
// threads pinned to the node's cpus for ~150 ms. returns GB/s, 0 on failure.
static double ep_numa_probe_bw_gbps(int node, const std::vector<int> & cpus) {
    if (cpus.empty()) {
        return 0.0;
    }
    const size_t len = (size_t) 1 << 30;
    char * buf = (char *) mmap(nullptr, len, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (buf == MAP_FAILED) {
        return 0.0;
    }
    unsigned long mask[16] = {0};
    mask[node / 64] |= 1UL << (node % 64);
    if (syscall(SYS_mbind, buf, len, MPOL_BIND, mask, 1024, 0) != 0) {
        LOG("llama-epd: numa: mbind node %d failed: %s\n", node, strerror(errno));
        munmap(buf, len);
        return 0.0;
    }
    memset(buf, 0x5a, len); // first touch after mbind: pages land on this node

    const int T = std::min(8, (int) cpus.size());
    const size_t slice = len / (size_t) T / 64 * 64; // 64-byte multiple
    std::atomic<bool> go{false}, halt{false};
    std::vector<std::atomic<uint64_t>> bytes((size_t) T);
    std::vector<std::thread> ths;
    for (int i = 0; i < T; ++i) {
        ths.emplace_back([&, i]() {
            cpu_set_t set;
            CPU_ZERO(&set);
            CPU_SET(cpus[(size_t) i % cpus.size()], &set);
            pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
            const uint64_t * p = (const uint64_t *) (buf + (size_t) i * slice);
            const size_t n = slice / 8;
            while (!go.load(std::memory_order_acquire)) {
            }
            uint64_t done = 0;
            volatile uint64_t sink = 0;
            while (!halt.load(std::memory_order_relaxed)) {
                uint64_t s = 0;
                for (size_t j = 0; j < n; ++j) {
                    s += p[j];
                }
                sink = s;
                done += slice;
            }
            (void) sink;
            bytes[(size_t) i].store(done, std::memory_order_relaxed);
        });
    }
    go.store(true, std::memory_order_release);
    const int64_t t0 = ggml_time_us();
    usleep(150000);
    const int64_t t1 = ggml_time_us();
    halt.store(true, std::memory_order_relaxed);
    uint64_t total = 0;
    for (auto & b : bytes) {
        total += b.load(std::memory_order_relaxed);
    }
    for (auto & t : ths) {
        t.join();
    }
    munmap(buf, len);
    const double sec = (t1 - t0) / 1e6;
    return sec > 0 ? (double) total / sec / 1e9 : 0.0;
}

// scale bandwidths to small integer weights (max 16), reduced by their gcd
static std::vector<int> ep_numa_weights_from_bw(const std::vector<double> & bw) {
    double mx = 0.0;
    for (double b : bw) {
        mx = std::max(mx, b);
    }
    std::vector<int> w(bw.size(), 1);
    if (mx <= 0.0) {
        return w;
    }
    int g = 0;
    for (size_t i = 0; i < bw.size(); ++i) {
        w[i] = std::max(1, (int) lround(16.0 * bw[i] / mx));
        g = (i == 0) ? w[i] : std::gcd(g, w[i]);
    }
    if (g > 1) {
        for (int & v : w) {
            v /= g;
        }
    }
    return w;
}

static bool ep_numa_write_weight(int node, int w) {
    char path[160];
    snprintf(path, sizeof(path), "/sys/kernel/mm/mempolicy/weighted_interleave/node%d", node);
    FILE * f = fopen(path, "w");
    if (!f) {
        return false;
    }
    const bool ok = fprintf(f, "%d", w) > 0;
    fclose(f);
    return ok;
}

static int ep_numa_read_weight(int node) {
    char path[160];
    snprintf(path, sizeof(path), "/sys/kernel/mm/mempolicy/weighted_interleave/node%d", node);
    std::string s;
    int w = 1;
    if (ep_read_first_line(path, s)) {
        sscanf(s.c_str(), "%d", &w);
    }
    return w;
}

// apply the env-configured NUMA policy; called once from main() before the
// model is loaded so every weight page (pread buffer or mmap fault) follows it
static void ep_numa_apply_policy() {
    const char * e = getenv("GGML_EPD_NUMA");
    if (!e || e[0] == '\0' || strcmp(e, "off") == 0 || strcmp(e, "0") == 0) {
        return;
    }
    int mode = 0;
    if (strcmp(e, "interleave") == 0) {
        mode = MPOL_INTERLEAVE;
    } else if (strcmp(e, "weighted") == 0) {
        mode = MPOL_WEIGHTED_INTERLEAVE;
    } else {
        LOG("llama-epd: numa: unknown GGML_EPD_NUMA='%s' (want off|interleave|weighted), ignoring\n", e);
        return;
    }

    const std::vector<int> nodes = ep_numa_online_nodes();
    if (nodes.size() < 2) {
        LOG("llama-epd: numa: only %zu online node(s), policy not applied\n", nodes.size());
        return;
    }

    std::vector<int> weights(nodes.size(), 1);
    if (mode == MPOL_WEIGHTED_INTERLEAVE) {
        bool have = false;
        // 1) explicit GGML_EPD_NUMA_WEIGHT ("a:b" or "a,b", one per online node)
        if (const char * w = getenv("GGML_EPD_NUMA_WEIGHT")) {
            std::string s = w;
            for (char & c : s) {
                if (c == ':' || c == ',') {
                    c = ' ';
                }
            }
            std::vector<int> v;
            int x = 0, n = 0;
            size_t off = 0;
            while (sscanf(s.c_str() + off, "%d%n", &x, &n) == 1) {
                v.push_back(x);
                off += (size_t) n;
            }
            if (v.size() == nodes.size()) {
                weights = v;
                have = true;
                LOG("llama-epd: numa: weights from GGML_EPD_NUMA_WEIGHT\n");
            } else {
                LOG("llama-epd: numa: GGML_EPD_NUMA_WEIGHT='%s' has %zu values, want %zu — ignoring\n",
                    w, v.size(), nodes.size());
            }
        }
        // 2) startup bandwidth probe (layout-agnostic: adapts to whatever DIMM
        //    config the machine currently has)
        if (!have) {
            std::vector<double> bw(nodes.size(), 0.0);
            for (size_t i = 0; i < nodes.size(); ++i) {
                bw[i] = ep_numa_probe_bw_gbps(nodes[i], ep_numa_node_cpus(nodes[i]));
            }
            bool ok = true;
            for (double b : bw) {
                ok = ok && b > 0.0;
            }
            if (ok) {
                weights = ep_numa_weights_from_bw(bw);
                have = true;
                LOG("llama-epd: numa: bandwidth probe:");
                for (size_t i = 0; i < nodes.size(); ++i) {
                    LOG(" node%d=%.1f GB/s", nodes[i], bw[i]);
                }
                LOG("\n");
            } else {
                LOG("llama-epd: numa: bandwidth probe failed, keeping sysfs weights\n");
            }
        }
        // write the weights; without permission fall back to whatever the
        // administrator already configured (never refuse to start)
        if (have) {
            bool wrote = true;
            for (size_t i = 0; i < nodes.size(); ++i) {
                wrote = ep_numa_write_weight(nodes[i], weights[i]) && wrote;
            }
            if (!wrote) {
                LOG("llama-epd: numa: cannot write sysfs weights (need root); using current values. to apply:");
                for (size_t i = 0; i < nodes.size(); ++i) {
                    LOG(" echo %d | sudo tee /sys/kernel/mm/mempolicy/weighted_interleave/node%d;", weights[i], nodes[i]);
                }
                LOG("\n");
            }
        }
        for (size_t i = 0; i < nodes.size(); ++i) {
            weights[i] = ep_numa_read_weight(nodes[i]);
        }
    }

    unsigned long mask[16] = {0};
    for (int nd : nodes) {
        mask[nd / 64] |= 1UL << (nd % 64);
    }
    if (syscall(SYS_set_mempolicy, mode, mask, 1024) != 0) {
        if (mode == MPOL_WEIGHTED_INTERLEAVE && errno == EINVAL) {
            LOG("llama-epd: numa: MPOL_WEIGHTED_INTERLEAVE unsupported (kernel < 6.9?), falling back to plain interleave\n");
            mode = MPOL_INTERLEAVE;
            if (syscall(SYS_set_mempolicy, mode, mask, 1024) != 0) {
                LOG("llama-epd: numa: set_mempolicy(interleave) failed: %s — continuing without policy\n", strerror(errno));
                return;
            }
        } else {
            LOG("llama-epd: numa: set_mempolicy failed: %s — continuing without policy\n", strerror(errno));
            return;
        }
    }
    LOG("llama-epd: numa: policy %s over nodes", mode == MPOL_WEIGHTED_INTERLEAVE ? "weighted-interleave" : "interleave");
    for (size_t i = 0; i < nodes.size(); ++i) {
        LOG(" %d%s", nodes[i], mode == MPOL_WEIGHTED_INTERLEAVE ?
            (std::string("(w=") + std::to_string(weights[i]) + ")").c_str() : "");
    }
    LOG("\n");
}

// ---------------------------------------------------------------------------
// server
// ---------------------------------------------------------------------------

static bool ep_send_err(llama_ep_transport * t, int32_t code, const std::string & msg) {
    std::vector<uint8_t> payload(sizeof(int32_t) + msg.size());
    memcpy(payload.data(), &code, sizeof(code));
    memcpy(payload.data() + sizeof(code), msg.data(), msg.size());
    return llama_ep_send_frame(t, LLAMA_EP_MSG_ERR, payload.data(), payload.size());
}

static bool ep_handle_req(
        llama_ep_transport * t,
        ggml_backend_t       backend,
        ggml_gallocr_t       gallocr,
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
    if (!ep_moe_ffn(backend, gallocr, L, (int) n_tokens, (int) n_ids, ids, weights, hidden, out.data(), err)) {
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
        ggml_gallocr_t       gallocr,
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
        if (!ep_handle_req(t, backend, gallocr, m, cfg, payload.data(), payload.size())) {
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

    ggml_gallocr_t gallocr = ggml_gallocr_new(ggml_backend_cpu_buffer_type());

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
        if (!ep_moe_ffn(backend, gallocr, L, n_tokens, n_ids, ids.data(), weights.data(), hidden.data(), out_a.data(), err)) {
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
            ep_serve_connection(&conn, backend, gallocr, m, cfg);
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
        "  -t, --threads N        compute threads (default: startup autotune; 8 if disabled)\n"
        "  --no-autotune          disable startup thread autotune (also GGML_EPD_AUTOTUNE=0)\n"
        "  --no-mmap              read owned expert weights into anonymous memory at startup\n"
        "                         (slow start, RSS = owned weights resident, no page-in ever)\n"
        "  --selftest             local vs loopback numerical check, then exit\n"
        "  --selftest-layer N     layer for selftest (default: first owned MoE layer)\n"
        "  --selftest-tokens N    tokens for selftest (default 4)\n"
        "\n"
        "env: GGML_EPD_NUMA=off|interleave|weighted (default off) — weight page NUMA\n"
        "     placement; weighted uses GGML_EPD_NUMA_WEIGHT=a:b or a startup bandwidth\n"
        "     probe (MPOL_WEIGHTED_INTERLEAVE, kernel >= 6.9)\n"
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
    bool threads_set = false;
    bool autotune = ep_autotune_enabled();
    bool no_mmap = false;

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
            threads_set = true;
        } else if (a == "--no-autotune") {
            autotune = false;
        } else if (a == "--no-mmap") {
            no_mmap = true;
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

    // NUMA placement policy (GGML_EPD_NUMA): must run before any weight
    // allocation/first-touch (both --no-mmap pread buffers and mmap page-ins
    // follow the process mempolicy)
    ep_numa_apply_policy();

    ggml_backend_load_all(); // no-op for static builds, keeps dl builds working

    ep_model m;
    if (!ep_model_load(m, model_path.c_str(), cfg.layer_first, cfg.layer_last, no_mmap)) {
        return 1;
    }
    LOG("llama-epd: arch=%s n_layer=%d, owning %zu MoE layers, experts [%d, %s)\n",
        m.arch.c_str(), m.n_layer, m.layers.size(), cfg.expert_first,
        cfg.expert_last == (1 << 30) ? "n_expert" : std::to_string(cfg.expert_last).c_str());

    if (selftest) {
        return ep_selftest(m, cfg, selftest_layer, selftest_tokens, 6);
    }

    if (ep_prefault_enabled()) {
        if (no_mmap) {
            LOG("llama-epd: prefault skipped (--no-mmap: weights already fully resident)\n");
        } else {
            ep_prefault_weights(m);
        }
    }

    ggml_backend_t backend = ggml_backend_cpu_init();
    if (!backend) {
        LOG("llama-epd: failed to init CPU backend\n");
        return 1;
    }
    ggml_gallocr_t gallocr = ggml_gallocr_new(ggml_backend_cpu_buffer_type());
    if (!threads_set && autotune) {
        const int tuned = ep_autotune_threads(backend, gallocr, m, cfg);
        if (tuned > 0) {
            cfg.n_threads = tuned;
        }
    } else if (!threads_set) {
        LOG("llama-epd: autotune disabled, using %d threads\n", cfg.n_threads);
    }
    ggml_backend_cpu_set_n_threads(backend, cfg.n_threads);

    // persistent threadpool: without it every ggml_graph_compute spawns and joins a
    // disposable pool (measured: ~7 ms of the per-REQ fixed cost at 70 threads).
    // attached after autotune so the ladder still probes with disposable pools
    struct ggml_threadpool_params tpp = ggml_threadpool_params_default(cfg.n_threads);
    ggml_threadpool_t threadpool = ggml_threadpool_new(&tpp);
    if (threadpool) {
        ggml_backend_cpu_set_threadpool(backend, threadpool);
    } else {
        LOG("llama-epd: WARNING: threadpool creation failed, falling back to per-compute pools\n");
    }

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
        ep_serve_connection(&conn, backend, gallocr, m, cfg);
        conn.ops.close(conn.ctx);
        LOG("llama-epd: client disconnected\n");
    }
    return 0;
}
