#include "llama-remote-ep.h"

#include "llama-impl.h"

#include "../tools/epd/llama-ep-capability.h"
#include "../tools/epd/llama-ep-dealer.h"
#include "../tools/epd/llama-ep-credit.h"
#include "../tools/epd/llama-ep-parallel.h"
#include "../tools/epd/llama-ep-service.h"
#include "../tools/epd/llama-ep-topology.h"
#include "../tools/epd/llama-ep-protocol.h"
#include "../tools/epd/llama-ep-transport.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <limits>
#include <memory>
#include <mutex>
#include <thread>
#include <unordered_map>
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

// The scheduled merge preserves slot order within each token, but different
// tokens are independent.  PP uses a small thread team by default; decode and
// short prompts stay single-threaded at the call site.  The environment knob
// can force a different team size (1 restores the original behavior).
int remote_ep_merge_threads() {
    static const int v = []() {
        const char * e = getenv("GGML_REMOTE_EP_MERGE_THREADS");
        const int n = e ? atoi(e) : 8;
        return std::max(1, std::min(n, 64));
    }();
    return v;
}

// Send requests to, and receive responses from, independent EP endpoints in
// parallel.  The response buffers are still merged later in ascending global
// slot order, so this only changes transport timing, not floating-point order.
// Keep the first deployment opt-in until all supported transports have been
// exercised; GGML_REMOTE_EP_PARALLEL_IO=1 enables it.
bool remote_ep_parallel_io_enabled() {
    static const bool v = []() {
        const char * e = getenv("GGML_REMOTE_EP_PARALLEL_IO");
        return e && e[0] != '\0' && strcmp(e, "0") != 0;
    }();
    return v;
}

// A pure-EP master cannot finish a layer while one of its workers is
// unavailable.  By default the historical behavior is preserved: reconnect
// once and fail immediately.  Serving deployments can instead give a worker
// process time to reload its shard and start listening again.  The value is
// cached and only consulted after a transport failure, so the healthy hot path
// is unchanged.
int remote_ep_reconnect_timeout_ms() {
    static const int v = []() {
        const char * e = getenv("GGML_REMOTE_EP_RECONNECT_TIMEOUT_MS");
        const long n = e ? strtol(e, nullptr, 10) : 0;
        return (int) std::max<long>(0, std::min<long>(n, 300000));
    }();
    return v;
}

// Diagnostic-only router trace used to design small-batch replica maps.  It is
// intentionally nested under GGML_REMOTE_EP_DEBUG at the call site so normal
// serving never formats or writes expert-id vectors.
bool remote_ep_trace_router_enabled() {
    static const bool v = []() {
        const char * e = getenv("GGML_REMOTE_EP_TRACE_ROUTER");
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
// buffers), 12 MiB over RDMA (the pre-posted receive ring holds 64 x 256 KiB).
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
    const int64_t bytes_cap = is_rdma ? (12 << 20) : (3 << 20);
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
    bool        sched_max_effort = false; // KLOCAL=0: allow replicated worker experts
    bool        sched_pp        = false; // allow n_tokens > 1 (P4; default decode-only)
    bool        sched_weight_on_master = false; // REQ4: worker skips output weighting op
    int64_t     sched_tg_activation_cost = 0; // 1000 = one median assignment
    int         sched_tg_repeat_cost = 250; // per mille of a new expert weight stream
    // Prefill normally balances every routed row independently. Values below
    // 1000 keep rows of the same expert on one replica when the resulting
    // virtual makespan stays competitive, enabling worker-side shared-weight
    // GEMV/GEMM. Default 1000 preserves the established PP policy.
    int         sched_pp_repeat_cost = 1000;
    // Experimental multi-stream queue model. Single-slot ABBA currently favors
    // the legacy assignment-count feedback, so keep this opt-in until a real
    // multi-slot sweep demonstrates a net gain.
    bool        sched_repeat_accounting = false;
    int         sched_negotiated = 0;   // 0 = not yet, 1 = ready, -1 = failed
    int         sched_n_expert   = 0;   // exact-cover size validated for KLOCAL=0
    // Dealer-format remote holder masks (bit 0 is reserved for the master).
    // Pure EP validates and builds this table once at graph construction, then
    // every MoE layer can point the dealer at it without rescanning CAP maps.
    std::vector<uint64_t> sched_remote_holders;

    // async pipelined dispatch (GGML_REMOTE_EP_PIPE=1): SCHED semantics over
    // REQ3/RESP3 with a background receiver thread per endpoint. completions
    // land in a req_id-keyed registry that the merge op polls/waits on, so
    // several requests may be in flight at once (cross-slot pipeline)
    bool        pipe            = false;
    struct pipe_request {
        int         iep    = -1;  // owning endpoint index
        int         state  = 0;   // 0 pending, 1 ready, 2 failed
        std::string err;
        // Own the validated RESP3 frame payload.  Keeping the wire buffer avoids
        // copying every expert vector into a second float vector on the receiver
        // thread; the merge reads the values after the fixed-size header.
        std::vector<uint8_t> resp_payload;
        int32_t     n_sel    = 0;
        int64_t     n_embd   = 0;
        int64_t     accounted_work = 0;
        int64_t     service_work_milli = 0; // 1000 = one new expert weight stream
        int64_t     dispatch_us = 0;
        int32_t     service_class = 0; // 0 decode, 1 prefill
        bool        measure_service = false;
        // request staging (owned copy, kept for one resend)
        int         il       = -1;
        int32_t     n_tokens = 0;
        std::vector<int32_t> token_idx, slot_idx, expert_id;
        std::vector<float>   weight;
        std::shared_ptr<std::vector<float>> hidden;
    };
    std::unordered_map<uint64_t, std::shared_ptr<pipe_request>> pipe_map;
    std::mutex              pipe_mtx;
    std::condition_variable pipe_cv;
    uint64_t                pipe_next_id = 1;
    llama_ep_credit_pool    pipe_credits{{(size_t) 512 << 20, 256}};

    // expert activation frequency (GGML_REMOTE_EP_FREQ=1): router selection
    // counts per (layer, expert) across all dispatched graphs, dumped at exit
    // (stderr summary; full CSV to GGML_REMOTE_EP_FREQ_FILE when set)
    bool                             ep_freq_on = false;
    std::string                      ep_freq_file;
    std::mutex                       ep_freq_mtx;
    std::vector<std::vector<uint64_t>> ep_freq;        // [layer][expert]
    std::vector<uint64_t>              ep_freq_tokens; // [layer] tokens seen

    struct sched_ep {
        std::string        host;
        int                port     = 29200;
        llama_ep_transport * conn    = nullptr;
        bool               is_rdma = false;
        int32_t            expert_first = 0; // from CAP
        int32_t            expert_last  = 0;
        std::vector<uint8_t> expert_bitmap; // optional CAP sparse ownership bits
        uint32_t           kernel_id  = 0;
        uint32_t           caps       = 0;
        std::unique_ptr<std::mutex> send_mtx = std::unique_ptr<std::mutex>(new std::mutex); // serializes frame writes on the conn
        std::thread        recv_thread; // pipe mode background receiver
        std::unique_ptr<std::atomic<bool>> recv_live =
            std::unique_ptr<std::atomic<bool>>(new std::atomic<bool>(false));
        // Queued/running virtual work on this worker. Service cost is learned
        // independently for decode and prefill; pointers keep sched_ep movable.
        std::unique_ptr<std::atomic<int64_t>> outstanding_work =
            std::unique_ptr<std::atomic<int64_t>>(new std::atomic<int64_t>(0));
        std::unique_ptr<std::atomic<int64_t>> decode_us_per_assignment =
            std::unique_ptr<std::atomic<int64_t>>(new std::atomic<int64_t>(0));
        std::unique_ptr<std::atomic<int64_t>> prefill_us_per_assignment =
            std::unique_ptr<std::atomic<int64_t>>(new std::atomic<int64_t>(0));
    };
    std::vector<sched_ep> sched_eps;

    // in-flight scheduled layer (partition+send op -> wait+merge op). the
    // compute thread is serial between the two ops of a layer, one slot suffices
    struct sched_pend_ep {
        bool                 send_failed = false;
        std::vector<float>   weight;
        std::vector<float>   resp;      // [n_sel*n_embd], filled by the merge op
        // pipe mode: registry entry carrying the staging + completion
        uint64_t             req_id = 0;
        std::shared_ptr<pipe_request> preq;
        int64_t              accounted_work = 0;
        int64_t              service_work_milli = 0;
        int64_t              dispatch_us = 0;
        int32_t              service_class = 0; // 0 decode, 1 prefill
        bool                 measure_service = false;
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
        // Persistent plan and scratch are retained per EP stream. Decode calls
        // this path for every MoE layer; retaining capacity removes allocator
        // traffic without carrying load or routing state across layers.
        llama_ep_dealer_plan      plan;
        llama_ep_dealer_workspace dealer_workspace;
        std::vector<uint64_t> holders;
        std::vector<int64_t> initial_load;
        std::vector<int64_t> observed_cost;
        std::vector<int64_t> assignment_cost;
        std::vector<int64_t> activation_penalty;
        std::vector<int64_t> repeat_cost;
        std::vector<int64_t> reserved_work;
        std::vector<int64_t> service_work_milli;
        std::vector<uint8_t> measure_service;
        std::vector<uint8_t> accounting_seen;
        std::vector<int> active_eps;
        std::vector<uint8_t> recv_ok;
        std::vector<std::string> recv_err;
        std::vector<size_t> ep_token_base;
        std::vector<size_t> merge_cursors;
        std::vector<sched_pend_ep> eps;
        llama_ep_credit_amount pipe_reserved;
    };
    // one reusable pending descriptor per EP stream (= concurrently computing
    // context); nodes are retained after merge so their planner/network vector
    // capacities survive into the next layer.
    std::unordered_map<int, sched_pend> spends;
    std::mutex                          spends_mtx;
    // Serializes load snapshot + plan + reservation across EP streams. It is
    // never held during network I/O or worker compute.
    std::mutex                          dealer_mtx;
    std::unordered_map<int, std::unique_ptr<llama_ep_parallel_for>> merge_pools;
    std::mutex                          merge_pools_mtx;
    // Sync REQ2 mode holds mtx across each send/merge callback, so one shared
    // persistent endpoint-I/O team is sufficient and cannot see overlapping
    // jobs.  It avoids two rounds of thread creation per MoE layer.
    std::unique_ptr<llama_ep_parallel_for> io_pool;

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

    // defined out-of-line after remote_ep_sched_ep_shutdown (which needs the
    // nested sched_ep type, and the destructor calls it)
    ~remote_ep_state();

    // per-(layer, expert) router activation counts: stderr summary line per
    // layer + optional full CSV (layer,expert,count rows)
    void dump_ep_freq() {
        FILE * f = stderr;
        std::string path = ep_freq_file;
        if (!path.empty()) {
            FILE * ff = fopen(path.c_str(), "w");
            if (ff != nullptr) {
                f = ff;
            } else {
                fprintf(stderr, "[ep-freq] cannot open %s, dumping to stderr\n", path.c_str());
            }
        }
        const bool csv = !path.empty() && f != stderr;
        if (csv) {
            fprintf(f, "layer,expert,count\n");
        }
        for (size_t il = 0; il < ep_freq.size(); ++il) {
            const auto & v = ep_freq[il];
            const uint64_t toks = il < ep_freq_tokens.size() ? ep_freq_tokens[il] : 0;
            if (toks == 0) {
                continue;
            }
            uint64_t sum = 0, mn = UINT64_MAX, mx = 0;
            size_t active = 0, argmn = 0, argmx = 0;
            for (size_t e = 0; e < v.size(); ++e) {
                sum += v[e];
                if (v[e] > 0) {
                    ++active;
                }
                if (v[e] < mn) {
                    mn = v[e];
                    argmn = e;
                }
                if (v[e] > mx) {
                    mx = v[e];
                    argmx = e;
                }
            }
            const double mean = v.empty() ? 0.0 : (double) sum / v.size();
            fprintf(stderr,
                    "[ep-freq] layer %zu: tokens=%llu experts=%zu active=%zu sum=%llu "
                    "mean=%.2f min=%llu(e%zu) max=%llu(e%zu) max/mean=%.2f\n",
                    il, (unsigned long long) toks, v.size(), active, (unsigned long long) sum,
                    mean, (unsigned long long) mn, argmn, (unsigned long long) mx, argmx,
                    mean > 0 ? mx / mean : 0.0);
            if (csv) {
                for (size_t e = 0; e < v.size(); ++e) {
                    fprintf(f, "%zu,%zu,%llu\n", il, e, (unsigned long long) v[e]);
                }
            }
        }
        if (f != stderr) {
            fclose(f);
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
        // expert activation frequency accounting (independent of SCHED): counts
        // router selections per (layer, expert) in every dispatched graph
        if (const char * f = getenv("GGML_REMOTE_EP_FREQ")) {
            ep_freq_on = f[0] != '\0' && strcmp(f, "0") != 0;
        }
        if (const char * f = getenv("GGML_REMOTE_EP_FREQ_FILE")) {
            ep_freq_file = f;
        }
        // async pipelined dispatch (GGML_REMOTE_EP_PIPE=1): implies SCHED with
        // the REQ3/RESP3 async transport; any multi-token batch is allowed
        if (const char * s = getenv("GGML_REMOTE_EP_PIPE")) {
            pipe = s[0] != '\0' && strcmp(s, "0") != 0;
            if (pipe) {
                sched = true;
            }
        }
        if (const char * value = getenv("GGML_REMOTE_EP_PIPE_MAX_MIB")) {
            char * end = nullptr;
            const unsigned long long mib = strtoull(value, &end, 10);
            if (end != value && *end == '\0' && mib <= (unsigned long long) (SIZE_MAX >> 20)) {
                auto limit = pipe_credits.limit();
                limit.bytes = (size_t) mib << 20;
                pipe_credits.set_limit(limit);
            } else {
                LLAMA_LOG_WARN("%s: ignoring malformed GGML_REMOTE_EP_PIPE_MAX_MIB='%s'\n", __func__, value);
            }
        }
        if (const char * value = getenv("GGML_REMOTE_EP_PIPE_MAX_REQUESTS")) {
            char * end = nullptr;
            const unsigned long long parsed = strtoull(value, &end, 10);
            if (end != value && *end == '\0' && parsed > 0 && parsed <= SIZE_MAX) {
                auto limit = pipe_credits.limit();
                limit.requests = (size_t) parsed;
                pipe_credits.set_limit(limit);
            } else {
                LLAMA_LOG_WARN("%s: ignoring malformed GGML_REMOTE_EP_PIPE_MAX_REQUESTS='%s'\n", __func__, value);
            }
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
                        sched_eps.push_back(std::move(ep));
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
                sched_eps.push_back(std::move(ep));
            }
            if (const char * k = getenv("GGML_REMOTE_EP_SCHED_KLOCAL")) {
                sched_klocal = atoi(k);
                if (sched_klocal < 0) {
                    LLAMA_LOG_WARN("%s: clamping negative GGML_REMOTE_EP_SCHED_KLOCAL=%d to 0\n",
                            __func__, sched_klocal);
                    sched_klocal = 0;
                }
            }
            if (const char * e = getenv("GGML_REMOTE_EP_SCHED_MAX_EFFORT")) {
                sched_max_effort = e[0] != '\0' && strcmp(e, "0") != 0;
            }
            if (sched_max_effort && sched_klocal != 0) {
                LLAMA_LOG_WARN("%s: GGML_REMOTE_EP_SCHED_MAX_EFFORT requires KLOCAL=0; disabling max-effort mode\n",
                        __func__);
                sched_max_effort = false;
            }
            if (const char * p = getenv("GGML_REMOTE_EP_SCHED_PP")) {
                sched_pp = p[0] != '\0' && strcmp(p, "0") != 0;
            }
            if (const char * p = getenv("GGML_REMOTE_EP_WEIGHT_ON_MASTER")) {
                sched_weight_on_master = p[0] != '\0' && strcmp(p, "0") != 0;
            }
            if (sched_weight_on_master && pipe) {
                LLAMA_LOG_WARN("%s: GGML_REMOTE_EP_WEIGHT_ON_MASTER is sync-only; disabling it in pipe mode\n", __func__);
                sched_weight_on_master = false;
            }
            if (const char * value = getenv("GGML_REMOTE_EP_SCHED_TG_ACTIVATION_COST")) {
                char * end = nullptr;
                const long long parsed = strtoll(value, &end, 10);
                if (end != value && *end == '\0' && parsed >= 0) {
                    sched_tg_activation_cost = parsed;
                } else {
                    LLAMA_LOG_WARN("%s: ignoring malformed GGML_REMOTE_EP_SCHED_TG_ACTIVATION_COST='%s'\n",
                            __func__, value);
                }
            }
            if (const char * value = getenv("GGML_REMOTE_EP_SCHED_TG_REPEAT_COST")) {
                char * end = nullptr;
                const long parsed = strtol(value, &end, 10);
                if (end != value && *end == '\0' && parsed >= 0 && parsed <= 1000) {
                    sched_tg_repeat_cost = (int) parsed;
                } else {
                    LLAMA_LOG_WARN("%s: ignoring malformed GGML_REMOTE_EP_SCHED_TG_REPEAT_COST='%s'\n",
                            __func__, value);
                }
            }
            if (const char * value = getenv("GGML_REMOTE_EP_SCHED_PP_REPEAT_COST")) {
                char * end = nullptr;
                const long parsed = strtol(value, &end, 10);
                if (end != value && *end == '\0' && parsed >= 0 && parsed <= 1000) {
                    sched_pp_repeat_cost = (int) parsed;
                } else {
                    LLAMA_LOG_WARN("%s: ignoring malformed GGML_REMOTE_EP_SCHED_PP_REPEAT_COST='%s'\n",
                            __func__, value);
                }
            }
            if (const char * value = getenv("GGML_REMOTE_EP_SCHED_REPEAT_ACCOUNTING")) {
                sched_repeat_accounting = value[0] == '\0' || strcmp(value, "0") != 0;
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
            LLAMA_LOG_INFO("%s: GGML_REMOTE_EP_SCHED=1: %zu endpoint(s), k_local=%d, pp=%d%s%s,"
                    " tg activation cost=%lld, TG repeat cost=%d/1000, PP repeat cost=%d/1000,"
                    " repeat accounting=%d\n",
                    __func__, sched_eps.size(), sched_klocal, (int) sched_pp,
                    pipe ? ", pipe(async REQ3)" : "",
                    sched_max_effort ? ", max-effort replicas" : "",
                    (long long) sched_tg_activation_cost, sched_tg_repeat_cost, sched_pp_repeat_cost,
                    (int) sched_repeat_accounting);
            if (pipe) {
                LLAMA_LOG_INFO("%s: pipe credits: %.1f MiB, %zu endpoint requests\n", __func__,
                        pipe_credits.limit().bytes / 1048576.0, pipe_credits.limit().requests);
            }
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

void llama_remote_ep_dump_freq() {
    remote_ep_state & st = remote_ep_get();
    if (!st.ep_freq_on) {
        return;
    }
    std::lock_guard<std::mutex> lock(st.ep_freq_mtx);
    st.dump_ep_freq();
}

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
    remote_ep_state & st = remote_ep_get();
    // Scheduled KLOCAL>0 needs the full local expert tensors. KLOCAL=0 is a
    // router/merge-only master and therefore skips them just like classic EP.
    return st.enabled && st.in_ranges(il) && !llama_remote_ep_mirror_layer(il) &&
           (!st.sched || st.sched_klocal == 0);
}

bool llama_remote_ep_sched_pure_for_layer(int il) {
    remote_ep_state & st = remote_ep_get();
    return st.enabled && st.sched && st.sched_klocal == 0 && st.in_ranges(il);
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

int llama_remote_ep_new_stream() {
    remote_ep_state & st = remote_ep_get();
    if (!st.pipe) {
        return 0; // single shared stream; the classic/sched paths serialize anyway
    }
    static std::atomic<int> next_stream{1};
    return next_stream.fetch_add(1);
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
    // ring (64 x 256 KiB); the master only drains it in the wait op, after the
    // local chain. keep the RESP inside the ring so the worker never stalls
    // (or worse) on a send nobody is receiving yet.
    if (llama_ep_rdma_requested()) {
        return (uint64_t) n_tokens * (uint64_t) n_embd * sizeof(float) <= (uint64_t) (12 << 20);
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

// Rebuild a scheduled endpoint connection, optionally waiting for a worker
// restart.  Callers have already dropped the failed transport.  A fixed 1 s
// cadence avoids connection storms and keeps the timeout behavior predictable;
// independent endpoints call this concurrently when PARALLEL_IO is enabled.
static bool remote_ep_sched_ep_connect_with_grace(remote_ep_state::sched_ep & ep, std::string & err) {
    const int timeout_ms = remote_ep_reconnect_timeout_ms();
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
    bool announced = false;

    for (;;) {
        err.clear();
        if (remote_ep_sched_ep_connect(ep, err)) {
            return true;
        }
        if (timeout_ms == 0 || std::chrono::steady_clock::now() >= deadline) {
            return false;
        }
        if (!announced) {
            LLAMA_LOG_WARN("%s: endpoint %s:%d unavailable (%s); waiting up to %d ms for restart\n",
                    __func__, ep.host.c_str(), ep.port, err.c_str(), timeout_ms);
            announced = true;
        }
        const auto remaining = std::chrono::duration_cast<std::chrono::milliseconds>(
                deadline - std::chrono::steady_clock::now());
        if (remaining.count() <= 0) {
            return false;
        }
        std::this_thread::sleep_for(std::min(remaining, std::chrono::milliseconds(1000)));
    }
}

static void remote_ep_sched_ep_drop(remote_ep_state::sched_ep & ep) {
    if (ep.conn) {
        ep.conn->ops.close(ep.conn->ctx);
        delete ep.conn;
        ep.conn = nullptr;
    }
}

// Stop a pipe receiver without racing transport destruction. shutdown wakes a
// blocking TCP recv or RDMA CM/CQ poll but keeps ctx valid until join returns.
static void remote_ep_sched_ep_shutdown(remote_ep_state::sched_ep & ep) {
    if (ep.conn) {
        ep.conn->ops.shutdown(ep.conn->ctx);
    }
    if (ep.recv_thread.joinable()) {
        ep.recv_thread.join();
    }
    if (ep.conn) {
        ep.conn->ops.close(ep.conn->ctx);
        delete ep.conn;
    }
    ep.recv_live->store(false, std::memory_order_release);
    ep.conn = nullptr;
}

remote_ep_state::~remote_ep_state() {
    if (ep_freq_on) {
        dump_ep_freq();
    }
    if (conn) {
        conn->ops.close(conn->ctx);
        delete conn;
    }
    for (auto & ep : sched_eps) {
        // shutdown -> join -> close/delete: keep the transport alive until a
        // blocked receiver has observed shutdown and returned.
        remote_ep_sched_ep_shutdown(ep);
    }
}

// CAP handshake with one endpoint (protocol v2). an old LEP1-only worker
// answers ERR to the unknown CAP type — treat any non-CAP reply as unsupported
static bool remote_ep_sched_ep_cap(
        remote_ep_state::sched_ep & ep,
        std::string               & err,
        bool                        require_stable = false) {
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
    if (type != LLAMA_EP_MSG_CAP) {
        err = "worker does not speak protocol v2 (old LEP1 worker?)";
        return false;
    }
    llama_ep_worker_capability capability;
    if (!llama_ep_parse_worker_capability(payload.data(), payload.size(), capability, err)) {
        return false;
    }
    const llama_ep_cap_worker & wcap = capability.wire;
    if (wcap.proto_ver < LLAMA_EP_PROTO_VER || !(wcap.caps & LLAMA_EP_CAP_REQ2)) {
        err = "worker CAP lacks REQ2";
        return false;
    }
    if (remote_ep_get().pipe && !(wcap.caps & LLAMA_EP_CAP_REQ3)) {
        err = "worker CAP lacks REQ3 (pipe mode needs the async protocol)";
        return false;
    }
    if (wcap.kernel_id != llama_ep_kernel_id()) {
        err = "kernel_id mismatch (heterogeneous ggml build — SCHEDULER-DESIGN §7.3)";
        return false;
    }
    if (require_stable && (ep.expert_first != wcap.expert_first ||
            ep.expert_last != wcap.expert_last || ep.expert_bitmap != capability.expert_bitmap ||
            ep.kernel_id != wcap.kernel_id || ep.caps != wcap.caps)) {
        err = "worker capabilities changed after reconnect";
        return false;
    }
    ep.expert_first  = wcap.expert_first;
    ep.expert_last   = wcap.expert_last;
    ep.expert_bitmap = std::move(capability.expert_bitmap);
    ep.kernel_id     = wcap.kernel_id;
    ep.caps          = wcap.caps;
    return true;
}

static bool remote_ep_sched_ep_holds(const remote_ep_state::sched_ep & ep, int expert) {
    const llama_ep_expert_ownership ownership{
        ep.expert_first,
        ep.expert_last,
        ep.expert_bitmap.empty() ? nullptr : ep.expert_bitmap.data(),
        ep.expert_bitmap.size(),
    };
    return ownership.holds(expert);
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
            if (st.sched_klocal == 0) {
                LLAMA_LOG_ERROR("%s: pure EP endpoint %s:%d: %s; no local expert fallback exists\n",
                        __func__, ep.host.c_str(), ep.port, err.c_str());
            } else {
                LLAMA_LOG_WARN("%s: sched endpoint %s:%d: %s; using retained local experts\n",
                        __func__, ep.host.c_str(), ep.port, err.c_str());
            }
            ok = false;
            break;
        }
        size_t n_owned = 0;
        if (!ep.expert_bitmap.empty()) {
            for (int e = ep.expert_first; e < ep.expert_last; ++e) {
                n_owned += remote_ep_sched_ep_holds(ep, e);
            }
        }
        const std::string owned_desc = ep.expert_bitmap.empty()
            ? std::string()
            : " sparse-owned=" + std::to_string(n_owned);
        LLAMA_LOG_INFO("%s: sched endpoint %s:%d: protocol v2, experts [%d, %d)%s, kernel_id %08x%s\n",
                __func__, ep.host.c_str(), ep.port, ep.expert_first, ep.expert_last,
                owned_desc.c_str(),
                ep.kernel_id, ep.is_rdma ? " [rdma]" : "");
    }
    if (!ok) {
        for (auto & ep : st.sched_eps) {
            remote_ep_sched_ep_drop(ep);
        }
        st.sched_negotiated = -1;
        return false;
    }
    if (st.sched_weight_on_master) {
        const auto unsupported = std::find_if(st.sched_eps.begin(), st.sched_eps.end(),
                [](const remote_ep_state::sched_ep & ep) { return !(ep.caps & LLAMA_EP_CAP_REQ4); });
        if (unsupported != st.sched_eps.end()) {
            LLAMA_LOG_WARN("%s: endpoint %s:%d lacks REQ4; keeping worker-side router weighting\n",
                    __func__, unsupported->host.c_str(), unsupported->port);
            st.sched_weight_on_master = false;
        } else {
            LLAMA_LOG_INFO("%s: REQ4 enabled: router weights are applied in the master slot-order merge\n", __func__);
        }
    }
    st.sched_negotiated = 1;
    return true;
}

static void remote_ep_sched_complete_work(
        remote_ep_state & st,
        int               iep,
        int64_t         & accounted_work,
        bool              measure_service,
        int               service_class,
        int64_t           dispatch_us,
        int64_t           assignments,
        int64_t           service_work_milli) {
    if (accounted_work <= 0) {
        return;
    }
    GGML_ASSERT(iep >= 0 && (size_t) iep < st.sched_eps.size());
    const int64_t released = accounted_work;
    accounted_work = 0;
    auto & ep = st.sched_eps[(size_t) iep];
    const int64_t previous = ep.outstanding_work->fetch_sub(
            released, std::memory_order_relaxed);
    GGML_ASSERT(previous >= released);

    // Only a request that saw an idle endpoint when reserved is a clean
    // service-rate sample. Queued requests include wait time already captured
    // by outstanding_work, so feeding their latency back would double-count it.
    if (!measure_service || dispatch_us <= 0 || assignments <= 0 || service_work_milli <= 0 ||
            (service_class != 0 && service_class != 1)) {
        return;
    }
    const int64_t elapsed_us = ggml_time_us() - dispatch_us;
    if (elapsed_us <= 0) {
        return;
    }
    // Convert elapsed request time to microseconds per new-expert-equivalent.
    // Quotient/remainder arithmetic avoids overflowing elapsed_us*1000 for a
    // malformed or extremely large PP request.
    const int64_t quotient = elapsed_us / service_work_milli;
    const int64_t remainder = elapsed_us % service_work_milli;
    const int64_t remainder_scaled = remainder <= std::numeric_limits<int64_t>::max() / 1000
        ? remainder * 1000 / service_work_milli
        : std::numeric_limits<int64_t>::max();
    const int64_t sample = quotient <= (std::numeric_limits<int64_t>::max() - remainder_scaled) / 1000
        ? std::max<int64_t>(1, quotient * 1000 + remainder_scaled)
        : std::numeric_limits<int64_t>::max();
    std::atomic<int64_t> & ewma = service_class == 0
        ? *ep.decode_us_per_assignment
        : *ep.prefill_us_per_assignment;
    int64_t old = ewma.load(std::memory_order_relaxed);
    for (;;) {
        const int64_t next = llama_ep_service_ewma(old, sample);
        if (ewma.compare_exchange_weak(old, next, std::memory_order_relaxed)) {
            if (remote_ep_debug_enabled()) {
                fprintf(stderr,
                        "GGML_REMOTE_EP: [ep-debug] endpoint %d %s service sample=%lld us/new-expert-equivalent"
                        " work=%lld/1000 assignments=%lld ewma=%lld\n",
                        iep, service_class == 0 ? "TG" : "PP",
                        (long long) sample, (long long) service_work_milli,
                        (long long) assignments, (long long) next);
            }
            break;
        }
    }
}

// ---------------------------------------------------------------------------
// async pipelined dispatch (GGML_REMOTE_EP_PIPE=1, protocol v3 REQ3/RESP3)
// ---------------------------------------------------------------------------

static void remote_ep_pipe_fail_pending(remote_ep_state & st, int iep, const std::string & err) {
    for (auto & kv : st.pipe_map) {
        auto & r = *kv.second;
        if (r.state == 0 && r.iep == iep) {
            r.err   = err;
            r.state = 2;
        }
    }
    st.pipe_cv.notify_all();
}

// background receiver: drains RESP3/ERR frames off one endpoint connection and
// marks the matching registry entries ready. exits when the transport dies
static void remote_ep_pipe_recv_loop(remote_ep_state * stp, size_t iep, llama_ep_transport * conn) {
    remote_ep_state & st = *stp;
    for (;;) {
        uint32_t type = 0;
        std::vector<uint8_t> payload;
        if (!llama_ep_recv_frame(conn, type, payload)) {
            break;
        }
        std::lock_guard<std::mutex> lock(st.pipe_mtx);
        if (type == LLAMA_EP_MSG_RESP3) {
            llama_ep_resp3_view response;
            std::string parse_err;
            if (!llama_ep_parse_resp3(payload.data(), payload.size(), response, parse_err)) {
                remote_ep_pipe_fail_pending(st, (int) iep, parse_err);
            } else {
                auto it = st.pipe_map.find(response.req_id);
                if (it != st.pipe_map.end()) {
                    auto & r = *it->second;
                    if (r.iep != (int) iep) {
                        remote_ep_pipe_fail_pending(st, (int) iep, "RESP3 request id belongs to another endpoint");
                    } else if (r.state == 0) {
                        if (response.n_sel == r.n_sel && response.n_embd == r.n_embd) {
                            r.resp_payload.swap(payload);
                            r.state = 1;
                            // The worker has completed this compute. Do not
                            // wait for the owning stream's merge callback to
                            // make the endpoint available to other streams.
                            remote_ep_sched_complete_work(*stp, (int) iep,
                                    r.accounted_work, r.measure_service, r.service_class,
                                    r.dispatch_us, r.n_sel, r.service_work_milli);
                        } else {
                            r.err   = "bad RESP3 shape";
                            r.state = 2;
                        }
                    }
                }
            }
        } else if (type == LLAMA_EP_MSG_ERR) {
            // ERR frames carry no req_id: fail every pending request of this
            // endpoint; the merge op resends once from the kept staging
            int32_t code = 0;
            if (payload.size() >= sizeof(code)) {
                memcpy(&code, payload.data(), sizeof(code));
            }
            remote_ep_pipe_fail_pending(st, (int) iep, "worker ERR " + std::to_string(code));
        } else {
            remote_ep_pipe_fail_pending(st, (int) iep, "unexpected frame type " + std::to_string(type));
        }
        st.pipe_cv.notify_all();
    }
    {
        std::lock_guard<std::mutex> lock(st.pipe_mtx);
        remote_ep_pipe_fail_pending(st, (int) iep, "connection lost");
    }
    st.sched_eps[iep].recv_live->store(false, std::memory_order_release);
}

// start the background receiver for an endpoint. caller holds st.mtx
static void remote_ep_pipe_recv_start(remote_ep_state & st, size_t iep) {
    auto & ep = st.sched_eps[iep];
    if (ep.recv_live->load(std::memory_order_acquire)) {
        return;
    }
    if (ep.recv_thread.joinable()) {
        ep.recv_thread.join();
    }
    ep.recv_live->store(true, std::memory_order_release);
    ep.recv_thread = std::thread(remote_ep_pipe_recv_loop, &st, iep, ep.conn);
}

// fire the staged REQ3 of one endpoint (send only, no wait). the staging in
// the registry entry owns the payload, so the send is safe to retry later
static bool remote_ep_pipe_send_req3(remote_ep_state & st, remote_ep_state::sched_pend & p, int iep, std::string & err) {
    auto & pe = p.eps[(size_t) iep];
    auto & ep = st.sched_eps[(size_t) iep];
    auto & r  = *pe.preq;

    llama_ep_req3_header hdr = {r.il, r.n_tokens, r.n_sel, (int32_t) r.n_embd, pe.req_id};

    const void * parts[6] = {&hdr, r.token_idx.data(), r.slot_idx.data(), r.expert_id.data(), r.weight.data(), r.hidden->data()};
    const size_t lens[6]  = {
        sizeof(hdr),
        r.token_idx.size() * sizeof(int32_t),
        r.slot_idx.size()  * sizeof(int32_t),
        r.expert_id.size() * sizeof(int32_t),
        r.weight.size()    * sizeof(float),
        r.hidden->size()   * sizeof(float),
    };
    std::lock_guard<std::mutex> slock(*ep.send_mtx);
    const int64_t dispatch_us = ggml_time_us();
    if (!llama_ep_send_framev(ep.conn, LLAMA_EP_MSG_REQ3, parts, lens, 6)) {
        err = "send REQ3 failed";
        return false;
    }
    r.dispatch_us = dispatch_us;
    return true;
}

// wait for one endpoint's pipe request to complete; on failure attempt one
// resend over a freshly rebuilt connection. returns false when the second
// wait still fails — the caller aborts
static bool remote_ep_pipe_wait_ep(remote_ep_state & st, remote_ep_state::sched_pend & p, int iep, const char * ctx) {
    auto & pe = p.eps[(size_t) iep];
    auto & r  = *pe.preq;

    auto wait_done = [&]() {
        std::unique_lock<std::mutex> lock(st.pipe_mtx);
        return st.pipe_cv.wait_for(lock, std::chrono::seconds(15),
                [&]() { return r.state != 0; });
    };

    if (!wait_done()) {
        LLAMA_LOG_WARN("%s: %s: endpoint %d: wait timeout, resending\n", __func__, ctx, iep);
        {
            std::lock_guard<std::mutex> lock(st.pipe_mtx);
            if (r.state == 0) {
                r.err   = "wait timeout";
                r.state = 2;
            }
        }
    }

    if (r.state == 1) {
        return true;
    }

    // one resend. rebuild the connection unconditionally: the failure may be a
    // dead transport whose recv thread has not flagged itself yet, and a clean
    // re-handshake is cheap compared to a wedged layer
    LLAMA_LOG_WARN("%s: %s: endpoint %d: %s — rebuilding connection and resending once\n",
            __func__, ctx, iep, r.err.c_str());
    {
        std::lock_guard<std::mutex> lock(st.mtx);
        auto & ep = st.sched_eps[(size_t) iep];
        // close -> join -> delete: close unblocks the receiver's recv_all, join
        // then returns, then the transport is freed (avoids deleting a conn a
        // live receiver is still inside) before reconnecting
        remote_ep_sched_ep_shutdown(ep);
        std::string cerr;
        if (!remote_ep_sched_ep_connect_with_grace(ep, cerr) || !remote_ep_sched_ep_cap(ep, cerr, true)) {
            return false;
        }
        remote_ep_pipe_recv_start(st, (size_t) iep);
        {
            std::lock_guard<std::mutex> plock(st.pipe_mtx);
            r.state = 0;
            r.err.clear();
            r.resp_payload.clear();
        }
        std::string serr;
        if (!remote_ep_pipe_send_req3(st, p, iep, serr)) {
            return false;
        }
    }
    if (!wait_done() || r.state != 1) {
        return false;
    }
    return true;
}

int llama_remote_ep_sched_klocal(int il, int n_expert_used, int n_expert) {
    remote_ep_state & st = remote_ep_get();
    if (!st.enabled || !st.sched || !st.in_ranges(il) || st.sched_eps.empty() ||
            n_expert_used < 1 || n_expert < 1) {
        return -1;
    }
    std::lock_guard<std::mutex> lock(st.mtx);
    if (!remote_ep_sched_negotiate(st)) {
        return -1;
    }
    if (st.sched_eps.size() > 63) {
        LLAMA_LOG_ERROR("%s: %zu scheduled endpoints exceed the 63-worker holder-mask limit\n",
                __func__, st.sched_eps.size());
        st.sched_negotiated = -1;
        return -1;
    }
    if (st.sched_klocal == 0 && st.sched_n_expert != n_expert) {
        std::vector<uint64_t> holders((size_t) n_expert, 0);
        bool ranges_ok = true;
        for (size_t i = 0; i < st.sched_eps.size(); ++i) {
            const auto & ep = st.sched_eps[i];
            ranges_ok = ranges_ok && ep.expert_first >= 0 && ep.expert_last <= n_expert;
            for (int e = 0; e < n_expert; ++e) {
                if (remote_ep_sched_ep_holds(ep, e)) {
                    holders[(size_t) e] |= 1ull << i;
                }
            }
        }
        const bool cover_ok = ranges_ok && llama_ep_holder_cover(
                n_expert, (int) st.sched_eps.size(), holders.data(), !st.sched_max_effort);
        if (!cover_ok) {
            LLAMA_LOG_ERROR("%s: KLOCAL=0 topology rejected: %zu endpoint ranges must %scover experts [0, %d)\n",
                    __func__, st.sched_eps.size(), st.sched_max_effort ? "fully " : "exactly and uniquely ", n_expert);
            st.sched_negotiated = -1;
            return -1;
        }
        st.sched_remote_holders.resize((size_t) n_expert);
        for (int e = 0; e < n_expert; ++e) {
            // Topology validation uses endpoint bit i. The dealer reserves bit
            // 0 for local compute and therefore addresses endpoint i at bit i+1.
            st.sched_remote_holders[(size_t) e] = holders[(size_t) e] << 1;
        }
        st.sched_n_expert = n_expert;
        LLAMA_LOG_INFO("%s: KLOCAL=0 pure EP topology ready: %zu worker(s), experts [0, %d), master expert replica disabled%s\n",
                __func__, st.sched_eps.size(), n_expert,
                st.sched_max_effort ? ", overlapping worker replicas enabled for dynamic balance" : ", strict non-overlap");
    }
    if (st.pipe) {
        for (size_t i = 0; i < st.sched_eps.size(); ++i) {
            remote_ep_pipe_recv_start(st, i);
        }
    }
    return std::min(st.sched_klocal, n_expert_used);
}

bool llama_remote_ep_sched_fits(int64_t n_tokens, int64_t n_ids, int64_t n_embd) {
    remote_ep_state & st = remote_ep_get();
    if (!st.sched_pp && !st.pipe && n_tokens != 1) {
        return false; // decode-only by default (P4 lifts this; pipe always lifts it)
    }
    if (n_tokens < 1 || n_ids < 1 || n_embd < 1) {
        return false;
    }
    const size_t tokens = (size_t) n_tokens;
    const size_t ids = (size_t) n_ids;
    if (tokens > std::numeric_limits<size_t>::max() / ids) {
        return false;
    }
    llama_ep_credit_amount amount;
    if (!llama_ep_credit_estimate(tokens, tokens * ids, (size_t) n_embd, 1, amount)) {
        return false;
    }
    if (st.pipe) {
        return st.pipe_credits.fits(amount);
    }

    // RDMA receive slots form a streaming byte ring: a frame may be larger
    // than the ring as long as the consumer keeps reposting slots.  The former
    // 12 MiB gate therefore tied graph construction to transport buffering and
    // forced PP to use tiny ubatches.  Bound the conservative aggregate staging
    // estimate by the protocol frame limit instead.  This admits DSV4 ubatch
    // 8192 while still rejecting shapes that cannot be represented on the wire.
    return amount.bytes <= LLAMA_EP_MAX_FRAME_BYTES;
}

// fire the staged REQ2 of one endpoint (send only, no wait)
static bool remote_ep_sched_send_req2(remote_ep_state & st, remote_ep_state::sched_pend & p, int iep, std::string & err) {
    auto & pe = p.eps[(size_t) iep];
    auto & ep = st.sched_eps[(size_t) iep];
    const llama_ep_dealer_ep & assignment = p.plan.eps[(size_t) iep];

    llama_ep_req2_header hdr = {p.il, (int32_t) p.n_tokens, (int32_t) assignment.token.size(), (int32_t) p.n_embd};

    const void * parts[6] = {&hdr, assignment.token.data(), assignment.slot.data(), assignment.expert.data(), pe.weight.data(), p.hidden};
    const size_t lens[6]  = {
        sizeof(hdr),
        assignment.token.size()  * sizeof(int32_t),
        assignment.slot.size()   * sizeof(int32_t),
        assignment.expert.size() * sizeof(int32_t),
        pe.weight.size()    * sizeof(float),
        (size_t) p.n_tokens * p.n_embd * sizeof(float),
    };
    const int64_t dispatch_us = ggml_time_us();
    const uint32_t type = st.sched_weight_on_master ? LLAMA_EP_MSG_REQ4 : LLAMA_EP_MSG_REQ2;
    if (!llama_ep_send_framev(ep.conn, type, parts, lens, 6)) {
        err = st.sched_weight_on_master ? "send REQ4 failed" : "send REQ2 failed";
        return false;
    }
    pe.dispatch_us = dispatch_us;
    return true;
}

// receive one RESP2 into the endpoint's resp staging
static bool remote_ep_sched_recv_resp2(remote_ep_state & st, remote_ep_state::sched_pend & p, int iep, std::string & err) {
    auto & pe = p.eps[(size_t) iep];
    auto & ep = st.sched_eps[(size_t) iep];

    const size_t n_sel  = p.plan.eps[(size_t) iep].token.size();
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
    const uint32_t expected_type = st.sched_weight_on_master ? LLAMA_EP_MSG_RESP4 : LLAMA_EP_MSG_RESP2;
    if (fh.type != expected_type || fh.payload_len != sizeof(llama_ep_resp2_header) + out_bytes) {
        err = st.sched_weight_on_master ? "bad RESP4" : "bad RESP2";
        return false;
    }
    llama_ep_resp2_header rhdr;
    pe.resp.resize(n_sel * (size_t) p.n_embd);
    if (!ep.conn->ops.recv_all(ep.conn->ctx, &rhdr, sizeof(rhdr)) ||
        rhdr.n_sel != (int32_t) n_sel || rhdr.n_embd != (int32_t) p.n_embd ||
        (out_bytes > 0 && !ep.conn->ops.recv_all(ep.conn->ctx, pe.resp.data(), out_bytes))) {
        err = st.sched_weight_on_master ? "recv RESP4 payload failed" : "recv RESP2 payload failed";
        return false;
    }
    return true;
}

void llama_remote_ep_sched_send_cb(ggml_tensor * dst, int ith, int nth, void * userdata) {
    if (ith != 0) {
        return;
    }
    (void) nth;

    const int il     = llama_remote_ep_userdata_il(userdata);
    const int stream = llama_remote_ep_userdata_stream(userdata);

    const ggml_tensor * a = dst->src[0]; // hidden  [n_embd, 1, n_tokens] f32 (contiguous)
    const ggml_tensor * b = dst->src[1]; // ids     [k, n_tokens] i32 (contiguous)
    const ggml_tensor * c = dst->src[2]; // weights [1, k, n_tokens] f32 (contiguous)

    const int64_t n_embd   = a->ne[0];
    const int64_t n_tokens = a->ne[2];
    const int64_t k        = b->ne[0];

    remote_ep_state & st = remote_ep_get();
    const int m_local = std::min(st.sched_klocal, (int) k);
    const int64_t dst_rows = std::max(1, m_local); // one dummy dependency row in pure EP

    const bool ok_shapes =
        a->type == GGML_TYPE_F32 && ggml_is_contiguous(a) &&
        b->type == GGML_TYPE_I32 && ggml_is_contiguous(b) && b->ne[1] == n_tokens &&
        c->type == GGML_TYPE_F32 && ggml_is_contiguous(c) && c->ne[0] == 1 && c->ne[1] == k && c->ne[2] == n_tokens &&
        dst->type == GGML_TYPE_I32 && ggml_is_contiguous(dst) &&
        dst->ne[0] == dst_rows && dst->ne[1] == n_tokens;

    if (!ok_shapes) {
        GGML_ABORT("%s: unexpected tensor shapes/types for layer %d", __func__, il);
    }

    const bool dbg = remote_ep_debug_enabled();
    const int64_t t0 = dbg ? ggml_time_us() : 0;

    // pipe mode: spend is compute-thread exclusive and the transport state is
    // guarded by pipe_mtx / per-endpoint send_mtx, so the global lock is skipped
    std::unique_lock<std::mutex> lock(st.mtx, std::defer_lock);
    if (!st.pipe) {
        lock.lock();
    }

    if (st.sched_negotiated != 1) {
        GGML_ABORT("%s: layer %d: sched endpoints were not negotiated at graph build", __func__, il);
    }

    remote_ep_state::sched_pend * pending = nullptr;
    {
        std::lock_guard<std::mutex> spends_lock(st.spends_mtx);
        pending = &st.spends[stream];
    }
    auto & p = *pending;
    if (p.active) {
        GGML_ABORT("%s: layer %d: previous sched request (layer %d, stream %d) was never consumed", __func__, il, p.il, stream);
    }

    const int32_t * ids = (const int32_t *) b->data;
    const float   * w   = (const float   *) c->data;

    // holder bitmask: bit0 = master only when a local chain exists;
    // bit(1+i) = endpoint i (CAP range). Pure KLOCAL=0 intentionally starts
    // with no master holder bits.
    int32_t e_max = 0;
    for (int64_t i = 0; i < n_tokens * k; ++i) {
        if (ids[i] < 0 || (st.sched_n_expert > 0 && ids[i] >= st.sched_n_expert)) {
            GGML_ABORT("%s: layer %d: router returned invalid expert id %d", __func__, il, ids[i]);
        }
        e_max = std::max(e_max, ids[i]);
    }
    const uint64_t * holder_data = nullptr;
    std::vector<uint64_t> & holders = p.holders;
    if (m_local == 0 && (size_t) e_max < st.sched_remote_holders.size()) {
        holder_data = st.sched_remote_holders.data();
    } else {
        holders.assign((size_t) e_max + 1, m_local > 0 ? 1u : 0u);
        for (size_t i = 0; i < st.sched_eps.size(); ++i) {
            for (int e = 0; e <= e_max; ++e) {
                if (remote_ep_sched_ep_holds(st.sched_eps[i], e)) {
                    holders[(size_t) e] |= 2ull << i;
                }
            }
        }
        holder_data = holders.data();
    }

    if (st.ep_freq_on) {
        std::lock_guard<std::mutex> flock(st.ep_freq_mtx);
        if ((int) st.ep_freq.size() <= il) {
            st.ep_freq.resize((size_t) il + 1);
            st.ep_freq_tokens.resize((size_t) il + 1, 0);
        }
        auto & v = st.ep_freq[(size_t) il];
        const int n_freq_expert = st.sched_n_expert > 0 ? st.sched_n_expert : e_max + 1;
        if ((int) v.size() < n_freq_expert) {
            v.resize((size_t) n_freq_expert, 0);
        }
        for (int64_t i = 0; i < n_tokens * k; ++i) {
            ++v[(size_t) ids[i]];
        }
        st.ep_freq_tokens[(size_t) il] += (uint64_t) n_tokens;
    }

    llama_ep_dealer_input din;
    din.n_tokens    = (int) n_tokens;
    din.k           = (int) k;
    din.n_endpoints = (int) st.sched_eps.size();
    din.m_star      = m_local;
    din.ids         = ids;
    din.holders     = holder_data;

    llama_ep_dealer_plan & plan = p.plan;
    const int service_class = n_tokens <= 4 ? 0 : 1;
    std::vector<int64_t> & observed_cost = p.observed_cost;
    std::vector<int64_t> & assignment_cost = p.assignment_cost;
    std::vector<int64_t> & activation_penalty = p.activation_penalty;
    std::vector<int64_t> & repeat_cost = p.repeat_cost;
    std::vector<int64_t> & reserved_work = p.reserved_work;
    std::vector<int64_t> & service_work_milli = p.service_work_milli;
    std::vector<uint8_t> & measure_service = p.measure_service;
    observed_cost.assign(st.sched_eps.size(), 0);
    assignment_cost.assign(st.sched_eps.size(), 1000);
    activation_penalty.assign(st.sched_eps.size(), 0);
    repeat_cost.assign(st.sched_eps.size(), 0);
    reserved_work.assign(st.sched_eps.size(), 0);
    service_work_milli.assign(st.sched_eps.size(), 0);
    measure_service.assign(st.sched_eps.size(), 0);
    {
        // Snapshot and reserve endpoint work atomically with respect to other
        // EP streams. The counters are released as soon as each worker response
        // arrives, rather than after the final merge.
        std::lock_guard<std::mutex> dealer_lock(st.dealer_mtx);
        std::vector<int64_t> & initial_load = p.initial_load;
        initial_load.assign(st.sched_eps.size(), 0);
        for (size_t i = 0; i < st.sched_eps.size(); ++i) {
            const auto & ep = st.sched_eps[i];
            initial_load[i] = ep.outstanding_work->load(std::memory_order_relaxed);
            observed_cost[i] = (service_class == 0
                    ? ep.decode_us_per_assignment
                    : ep.prefill_us_per_assignment)->load(std::memory_order_relaxed);
        }
        if (!llama_ep_service_normalize_costs(observed_cost.data(), (int) observed_cost.size(),
                1000, assignment_cost.data())) {
            GGML_ABORT("%s: invalid endpoint service cost", __func__);
        }
        din.initial_remote_load = initial_load.data();
        din.remote_assignment_cost = assignment_cost.data();
        if (service_class == 0 && st.sched_tg_activation_cost > 0) {
            std::fill(activation_penalty.begin(), activation_penalty.end(), st.sched_tg_activation_cost);
            din.remote_activation_penalty = activation_penalty.data();
        }
        const int repeat_permille = service_class == 0
            ? st.sched_tg_repeat_cost
            : st.sched_pp_repeat_cost;
        const bool use_repeat_affinity = repeat_permille < 1000;
        if (use_repeat_affinity) {
            for (size_t i = 0; i < st.sched_eps.size(); ++i) {
                const int64_t full = assignment_cost[i];
                const int64_t scaled = (full / 1000) * repeat_permille +
                    (full % 1000) * repeat_permille / 1000;
                repeat_cost[i] = std::max<int64_t>(1, scaled);
            }
            din.remote_repeat_assignment_cost = repeat_cost.data();
        }
        if (!llama_ep_dealer_plan_build(din, plan, p.dealer_workspace) || plan.m_local != m_local) {
            GGML_ABORT("%s: layer %d: dealer failed (infeasible holder table)", __func__, il);
        }
        // PP repeat affinity changes the dealer's unit from routed rows to
        // new-expert-equivalents, so its queue reservation must use the same
        // unit. TG retains the opt-in accounting switch because the older
        // single-slot A/B favored assignment-count feedback there.
        const bool account_repeats = use_repeat_affinity &&
            (service_class == 1 || st.sched_repeat_accounting);
        const size_t accounting_stride = (size_t) (st.sched_n_expert > 0
            ? st.sched_n_expert
            : e_max + 1);
        std::vector<uint8_t> & accounting_seen = p.accounting_seen;
        accounting_seen.assign(account_repeats
                ? st.sched_eps.size() * accounting_stride
                : 0, 0);
        for (size_t i = 0; i < st.sched_eps.size(); ++i) {
            const int64_t assignments = (int64_t) plan.eps[i].token.size();
            int64_t unique_experts = assignments;
            int64_t repeated_rows = 0;
            if (account_repeats) {
                unique_experts = 0;
                const auto & experts = plan.eps[i].expert;
                uint8_t * seen = accounting_seen.data() + i * accounting_stride;
                for (const int32_t expert : experts) {
                    GGML_ASSERT(expert >= 0 && (size_t) expert < accounting_stride);
                    if (!seen[(size_t) expert]) {
                        seen[(size_t) expert] = 1;
                        ++unique_experts;
                    }
                }
                repeated_rows = assignments - unique_experts;
            }
            if (!llama_ep_service_split_work_units(
                        unique_experts, repeated_rows,
                        assignment_cost[i], repeated_rows > 0 ? repeat_cost[i] : 0,
                        reserved_work[i]) ||
                    !llama_ep_service_split_work_units(
                        unique_experts, repeated_rows, 1000,
                        account_repeats ? repeat_permille : 1000,
                        service_work_milli[i])) {
                GGML_ABORT("%s: layer %d: endpoint work estimate overflow", __func__, il);
            }
            measure_service[i] = assignments > 0 && initial_load[i] == 0;
            st.sched_eps[i].outstanding_work->fetch_add(reserved_work[i], std::memory_order_relaxed);
        }
    }

    // local ids -> dst ([m*, n_tokens], ascending global slot order per token).
    // In pure EP dst is only a one-row dependency token and remains untouched.
    if (m_local > 0) {
        memcpy(dst->data, plan.local_ids.data(), (size_t) n_tokens * m_local * sizeof(int32_t));
    }

    // stage the plan + per-endpoint REQ2 bytes (kept for one resend in the merge op)
    p.il       = il;
    p.n_tokens = n_tokens;
    p.n_embd   = n_embd;
    p.k        = (int) k;
    p.m_local  = m_local;
    p.hidden   = (const float *) a->data;
    p.weights  = w;
    p.pipe_reserved = {};
    if (p.eps.size() != st.sched_eps.size()) {
        p.eps.resize(st.sched_eps.size());
    }
    for (size_t i = 0; i < st.sched_eps.size(); ++i) {
        const auto & src = plan.eps[i];
        auto & pe = p.eps[i];
        pe.weight.resize(src.token.size());
        for (size_t j = 0; j < src.token.size(); ++j) {
            pe.weight[j] = w[(size_t) src.token[j] * k + src.slot[j]];
        }
        pe.resp.clear();
        pe.send_failed = false;
        pe.req_id = 0;
        pe.preq   = nullptr;
        pe.accounted_work = reserved_work[i];
        pe.service_work_milli = service_work_milli[i];
        pe.dispatch_us = 0;
        pe.service_class = service_class;
        pe.measure_service = measure_service[i] != 0;
    }
    p.active = true;
    p.active_eps.clear();
    p.active_eps.reserve(st.sched_eps.size());
    for (size_t i = 0; i < p.plan.eps.size(); ++i) {
        if (!p.plan.eps[i].token.empty()) {
            p.active_eps.push_back((int) i);
        }
    }

    if (st.pipe) {
        // register one async request per endpoint in the completion registry;
        // all endpoint requests share one owned hidden copy. This keeps resend
        // safety without multiplying the largest payload by the worker count.
        size_t remote_assignments = 0;
        size_t remote_requests = 0;
        for (const auto & assignment : p.plan.eps) {
            if (!assignment.token.empty()) {
                remote_assignments += assignment.token.size();
                remote_requests += 1;
            }
        }
        llama_ep_credit_amount reserve;
        if (!llama_ep_credit_estimate((size_t) n_tokens, remote_assignments, (size_t) n_embd,
                remote_requests, reserve) || !st.pipe_credits.fits(reserve)) {
            GGML_ABORT("%s: layer %d: pipe request exceeds configured credits", __func__, il);
        }

        std::unique_lock<std::mutex> plock(st.pipe_mtx);
        st.pipe_cv.wait(plock, [&]() {
            return st.pipe_credits.can_reserve(reserve);
        });
        GGML_ASSERT(st.pipe_credits.try_reserve(reserve));
        p.pipe_reserved = reserve;
        if (dbg) {
            fprintf(stderr, "GGML_REMOTE_EP: [ep-debug] pipe credits %.1f/%.1f MiB requests %zu/%zu\n",
                    st.pipe_credits.current().bytes / 1048576.0, st.pipe_credits.limit().bytes / 1048576.0,
                    st.pipe_credits.current().requests, st.pipe_credits.limit().requests);
        }
        std::shared_ptr<std::vector<float>> hidden_copy;
        for (int iep : p.active_eps) {
            const size_t i = (size_t) iep;
            auto & pe = p.eps[i];
            const auto & assignment = p.plan.eps[i];
            if (!hidden_copy) {
                hidden_copy = std::make_shared<std::vector<float>>(
                    p.hidden, p.hidden + (size_t) n_tokens * n_embd);
            }
            auto r = std::make_shared<remote_ep_state::pipe_request>();
            r->iep       = (int) i;
            r->n_sel     = (int32_t) assignment.token.size();
            r->n_embd    = n_embd;
            r->accounted_work = pe.accounted_work;
            pe.accounted_work = 0; // ownership moves to the async request
            r->service_work_milli = pe.service_work_milli;
            r->dispatch_us = 0;
            r->service_class = pe.service_class;
            r->measure_service = pe.measure_service;
            r->il        = il;
            r->n_tokens  = (int32_t) n_tokens;
            r->token_idx = assignment.token;
            r->slot_idx  = assignment.slot;
            r->expert_id = assignment.expert;
            r->weight    = pe.weight;
            r->hidden     = hidden_copy;
            uint64_t id = 0;
            do {
                id = st.pipe_next_id++;
            } while (id == 0 || st.pipe_map.find(id) != st.pipe_map.end());
            st.pipe_map[id] = r;
            pe.req_id = id;
            pe.preq   = r;
        }
    }

    // Fire one request per endpoint (send only, no wait). With KLOCAL>0 the
    // following local chain overlaps worker compute; with KLOCAL=0 the merge op
    // immediately waits while all selected workers run concurrently.
    if (st.pipe) {
        for (int iep : p.active_eps) {
            const size_t i = (size_t) iep;
            auto & pe = p.eps[i];
            std::lock_guard<std::mutex> clock(st.mtx);
            std::string err;
            if (!remote_ep_sched_ep_connect(st.sched_eps[i], err) || !remote_ep_pipe_send_req3(st, p, (int) i, err)) {
                // the merge op reconnects and resends once from the registry staging
                LLAMA_LOG_WARN("%s: layer %d: endpoint %zu: %s (will retry in merge op)\n", __func__, il, i, err.c_str());
                std::lock_guard<std::mutex> plock(st.pipe_mtx);
                pe.preq->err   = err;
                pe.preq->state = 2;
            }
        }
    } else {
        auto send_one = [&](int iep) {
            auto & pe = p.eps[(size_t) iep];
            std::string err;
            if (!remote_ep_sched_ep_connect(st.sched_eps[(size_t) iep], err) ||
                    !remote_ep_sched_send_req2(st, p, iep, err)) {
                // Leave staging in place: merge reconnects and resends once.
                LLAMA_LOG_WARN("%s: layer %d: endpoint %d: %s (will retry in merge op)\n",
                        __func__, il, iep, err.c_str());
                pe.send_failed = true;
                remote_ep_sched_ep_drop(st.sched_eps[(size_t) iep]);
            }
        };
        if (remote_ep_parallel_io_enabled() && p.active_eps.size() > 1) {
            if (!st.io_pool || st.io_pool->max_tasks() < (int) p.active_eps.size()) {
                st.io_pool = std::make_unique<llama_ep_parallel_for>((int) p.active_eps.size());
            }
            st.io_pool->run((int) p.active_eps.size(), [&](int task, int) {
                send_one(p.active_eps[(size_t) task]);
            });
        } else {
            for (int iep : p.active_eps) {
                send_one(iep);
            }
        }
    }

    if (dbg) {
        fprintf(stderr, "GGML_REMOTE_EP: [ep-debug] layer %d sched k=%lld m*=%d n_tokens=%lld remote=[",
                il, (long long) k, m_local, (long long) n_tokens);
        for (size_t i = 0; i < p.eps.size(); ++i) {
            fprintf(stderr, "%s%zu", i ? "," : "", p.plan.eps[i].token.size());
        }
        fprintf(stderr, "]");
        if (remote_ep_trace_router_enabled()) {
            fprintf(stderr, " ids=[");
            for (int64_t i = 0; i < n_tokens * k; ++i) {
                fprintf(stderr, "%s%d", i ? "," : "", ids[i]);
            }
            fprintf(stderr, "]");
        }
        fprintf(stderr, " deal+send %.3f ms (t=%lld us)\n", (ggml_time_us() - t0) / 1000.0, (long long) t0);
    }
}

void llama_remote_ep_sched_merge_cb(ggml_tensor * dst, int ith, int nth, void * userdata) {
    if (ith != 0) {
        return;
    }
    (void) nth;

    const int il     = llama_remote_ep_userdata_il(userdata);
    const int stream = llama_remote_ep_userdata_stream(userdata);

    remote_ep_state & st = remote_ep_get();

    const bool dbg = remote_ep_debug_enabled();
    const int64_t t0 = dbg ? ggml_time_us() : 0;

    std::unique_lock<std::mutex> lock(st.mtx, std::defer_lock);
    if (!st.pipe) {
        lock.lock();
    }

    // the map node stays put while this op runs; other streams may come and go
    // concurrently, so do not hold spends_mtx across the (potentially long)
    // endpoint waits below
    remote_ep_state::sched_pend * pp = nullptr;
    {
        std::lock_guard<std::mutex> spends_lock(st.spends_mtx);
        auto it = st.spends.find(stream);
        if (it == st.spends.end()) {
            GGML_ABORT("%s: layer %d: no pending sched request for stream %d", __func__, il, stream);
        }
        pp = &it->second;
    }
    auto & p = *pp;
    if (!p.active || p.il != il) {
        GGML_ABORT("%s: layer %d: no pending sched request (active=%d, pending layer %d, stream %d)",
                __func__, il, (int) p.active, p.il, stream);
    }

    // KLOCAL>0 graph: {send, local expert output, weights}.
    // KLOCAL=0 graph: {send, weights}; there is no local expert tensor at all.
    const bool pure = p.m_local == 0;
    const ggml_tensor * experts_l_t = pure ? nullptr     : dst->src[1];
    const ggml_tensor * weights_t   = pure ? dst->src[1] : dst->src[2];
    const bool local_shape_ok = pure || (experts_l_t != nullptr &&
        experts_l_t->type == GGML_TYPE_F32 && ggml_is_contiguous(experts_l_t) &&
        experts_l_t->ne[0] == p.n_embd && experts_l_t->ne[1] == (int64_t) p.m_local &&
        experts_l_t->ne[2] == p.n_tokens);

    if (dst->type != GGML_TYPE_F32 || !ggml_is_contiguous(dst) ||
        dst->ne[0] != p.n_embd || dst->ne[1] != 1 || dst->ne[2] != p.n_tokens ||
        !local_shape_ok || weights_t == nullptr || weights_t->type != GGML_TYPE_F32 ||
        !ggml_is_contiguous(weights_t) || weights_t->ne[0] != 1 ||
        weights_t->ne[1] != p.k || weights_t->ne[2] != p.n_tokens) {
        GGML_ABORT("%s: unexpected dst/src shapes for layer %d", __func__, il);
    }

    if (st.pipe) {
        // wait for the background receiver to deliver every endpoint's RESP3;
        // one resend per endpoint on failure (see remote_ep_pipe_wait_ep)
        for (int iep : p.active_eps) {
            const size_t i = (size_t) iep;
            auto & pe = p.eps[i];
            const std::string ctx = "layer " + std::to_string(il);
            if (!remote_ep_pipe_wait_ep(st, p, (int) i, ctx.c_str())) {
                GGML_ABORT("%s: layer %d: endpoint %zu: %s", __func__, il, i, pe.preq->err.c_str());
            }
            {
                std::lock_guard<std::mutex> plock(st.pipe_mtx);
                st.pipe_map.erase(pe.req_id);
            }
        }
    } else {
        // Each endpoint owns an independent transport and response buffer. Read
        // them concurrently, then perform the unchanged fixed slot-order merge.
        p.recv_ok.assign(st.sched_eps.size(), 0);
        p.recv_err.resize(st.sched_eps.size());
        for (std::string & err : p.recv_err) {
            err.clear();
        }
        auto recv_one = [&](int iep) {
            auto & pe = p.eps[(size_t) iep];
            auto & err = p.recv_err[(size_t) iep];
            bool ok = !pe.send_failed && st.sched_eps[(size_t) iep].conn != nullptr &&
                      remote_ep_sched_recv_resp2(st, p, iep, err);
            if (!ok) {
                LLAMA_LOG_WARN("%s: layer %d: endpoint %d: %s — reconnecting and resending once\n",
                        __func__, il, iep, err.c_str());
                remote_ep_sched_ep_drop(st.sched_eps[(size_t) iep]);
                err.clear();
                ok = remote_ep_sched_ep_connect_with_grace(st.sched_eps[(size_t) iep], err) &&
                     remote_ep_sched_ep_cap(st.sched_eps[(size_t) iep], err, true) &&
                     remote_ep_sched_send_req2(st, p, iep, err) &&
                     remote_ep_sched_recv_resp2(st, p, iep, err);
            }
            p.recv_ok[(size_t) iep] = ok;
            if (ok) {
                remote_ep_sched_complete_work(st, iep,
                        pe.accounted_work, pe.measure_service, pe.service_class,
                        pe.dispatch_us, (int64_t) p.plan.eps[(size_t) iep].token.size(),
                        pe.service_work_milli);
            }
        };
        if (remote_ep_parallel_io_enabled() && p.active_eps.size() > 1) {
            GGML_ASSERT(st.io_pool && st.io_pool->max_tasks() >= (int) p.active_eps.size());
            st.io_pool->run((int) p.active_eps.size(), [&](int task, int) {
                recv_one(p.active_eps[(size_t) task]);
            });
        } else {
            for (int iep : p.active_eps) {
                recv_one(iep);
            }
        }
        for (int iep : p.active_eps) {
            if (!p.recv_ok[(size_t) iep]) {
                GGML_ABORT("%s: layer %d: endpoint %d: %s",
                        __func__, il, iep, p.recv_err[(size_t) iep].c_str());
            }
        }
    }

    const int64_t t_resp = dbg ? ggml_time_us() : 0;

    // merge in ascending global slot order, left-associated — the same scalar
    // operation sequence as the baseline ggml_mul + ggml_add chain (§4.5).
    // REQ2 responses arrive weighted. REQ4 deliberately returns raw vectors,
    // so the same scalar merge applies their router weight and removes the
    // worker's standalone ggml_mul op/barrier.
    const float * experts_l = pure ? nullptr : (const float *) experts_l_t->data;
    const float * w         = (const float *) weights_t->data;
    float       * out       = (float       *) dst->data;

    const int64_t k       = p.k;
    const int64_t m_local = p.m_local;
    const int64_t n_embd  = p.n_embd;

    // Response vectors are endpoint-grouped in (token, slot) order. Build a
    // token prefix for each endpoint so token ranges can merge independently
    // without sharing the old sequential endpoint cursors.
    const size_t token_stride = (size_t) p.n_tokens + 1;
    p.ep_token_base.assign(p.eps.size() * token_stride, 0);
    for (size_t i = 0; i < p.eps.size(); ++i) {
        size_t * base = p.ep_token_base.data() + i * token_stride;
        for (int32_t token : p.plan.eps[i].token) {
            ++base[(size_t) token + 1];
        }
        for (int64_t t = 0; t < p.n_tokens; ++t) {
            base[(size_t) t + 1] += base[(size_t) t];
        }
    }

    const int n_merge_threads = std::min<int64_t>(remote_ep_merge_threads(), p.n_tokens);
    p.merge_cursors.assign((size_t) n_merge_threads * p.eps.size(), 0);
    auto merge_range = [&](int task, int64_t t_first, int64_t t_last) {
        size_t * cur_ep = p.merge_cursors.data() + (size_t) task * p.eps.size();
        for (int64_t t = t_first; t < t_last; ++t) {
            for (size_t i = 0; i < p.eps.size(); ++i) {
                cur_ep[i] = p.ep_token_base[i * token_stride + (size_t) t];
            }
            float * acc = out + t * n_embd;
            int64_t lp = 0; // local slots are ascending per token
            for (int64_t j = 0; j < k; ++j) {
                const uint8_t o = p.plan.owner[(size_t) t * k + j];
                if (o == 0) {
                    if (experts_l == nullptr) {
                        GGML_ABORT("%s: layer %d: pure EP dealer assigned slot %lld to the master",
                                __func__, il, (long long) j);
                    }
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
                    const float * response = st.pipe
                        ? reinterpret_cast<const float *>(
                              pe.preq->resp_payload.data() + sizeof(llama_ep_resp3_header))
                        : pe.resp.data();
                    const float * v = response + idx * (size_t) n_embd;
                    if (st.sched_weight_on_master) {
                        const float wj = w[(size_t) t * k + j];
                        if (j == 0) {
                            for (int64_t e = 0; e < n_embd; ++e) {
                                acc[e] = v[e] * wj;
                            }
                        } else {
                            for (int64_t e = 0; e < n_embd; ++e) {
                                acc[e] += v[e] * wj;
                            }
                        }
                    } else if (j == 0) {
                        memcpy(acc, v, (size_t) n_embd * sizeof(float));
                    } else {
                        for (int64_t e = 0; e < n_embd; ++e) {
                            acc[e] += v[e];
                        }
                    }
                }
            }
        }
    };

    if (n_merge_threads == 1 || p.n_tokens < 64) {
        merge_range(0, 0, p.n_tokens);
    } else {
        llama_ep_parallel_for * merge_pool = nullptr;
        {
            std::lock_guard<std::mutex> pools_lock(st.merge_pools_mtx);
            auto & pool = st.merge_pools[stream];
            if (!pool) {
                pool = std::make_unique<llama_ep_parallel_for>(remote_ep_merge_threads());
            }
            merge_pool = pool.get();
        }
        merge_pool->run(n_merge_threads, [&](int task, int n_tasks) {
            const int64_t first = p.n_tokens * task       / n_tasks;
            const int64_t last  = p.n_tokens * (task + 1) / n_tasks;
            merge_range(task, first, last);
        });
    }

    const int64_t done_n_tokens = p.n_tokens;
    if (st.pipe) {
        for (auto & pe : p.eps) {
            pe.preq.reset();
            pe.req_id = 0;
        }
    }
    p.active  = false;
    p.hidden  = nullptr;
    p.weights = nullptr;
    if (st.pipe && (p.pipe_reserved.bytes > 0 || p.pipe_reserved.requests > 0)) {
        {
            std::lock_guard<std::mutex> pipe_lock(st.pipe_mtx);
            GGML_ASSERT(st.pipe_credits.release(p.pipe_reserved));
            p.pipe_reserved = {};
        }
        st.pipe_cv.notify_all();
    }
    // Keep the per-stream descriptor in the map. Its active flag is the
    // lifetime guard; retaining it preserves dealer and transport staging
    // capacity for the next layer and bounds allocations by stream count.

    if (dbg) {
        fprintf(stderr, "GGML_REMOTE_EP: [ep-debug] layer %d sched n_tokens=%lld wait %.3f ms merge %.3f ms (t=%lld us)\n",
                il, (long long) done_n_tokens, (t_resp - t0) / 1000.0,
                (ggml_time_us() - t_resp) / 1000.0, (long long) t0);
    }
}
