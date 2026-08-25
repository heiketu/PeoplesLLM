// llama-epd: MoE expert-parallel worker daemon (stage 1: TCP loopback, single layer verification).
//
// Loads a GGUF model via mmap (read-only, shared page cache), owns a range of layers
// and either a contiguous or sparse set of experts, and listens on a TCP port. For
// each REQ it computes
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
// representative owned layers over a {16,24,32,36,physical cores} ladder and picks
// the smallest count within 3% of the global best. Disable with
// --no-autotune or GGML_EPD_AUTOTUNE=0.
//
// --no-mmap replaces the default read-only MAP_SHARED mapping with a one-time
// sequential pread of the owned layers' expert weights into anonymous memory:
// slow start (full read), RSS = owned weights permanently resident, zero page-in
// stalls, immune to page-cache eviction (slow-disk setups where mmap page-in
// dominates). GGML_EP_PREFAULT is skipped (meaningless) when --no-mmap is on.
//
// After loading, the owned expert weights are converted into the CPU_REPACK
// (interleaved) layout so mul_mat_id dispatches to the repack gemv/gemm kernels
// instead of generic vec_dot (one-time cost at startup; the original mapping is
// no longer referenced, --no-mmap anonymous copies are freed). Tensors without
// matching repack traits keep the raw layout. GGML_EPD_REPACK=0 disables.

#include "llama-ep-transport.h"
#include "llama-ep-capability.h"
#include "llama-ep-protocol.h"
#include "llama-ep-session-manager.h"
#include "llama-ep-expert-map.h"
#include "llama-ep-fusion.h"
#include "llama-ep-autotune.h"
#include "llama-epd-runtime.h"

#include "llama.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-shard-plan.h"
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
#include <unordered_map>
#include <mutex>
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

// Request-side ragged hidden gather. For small TG batches, a serial memcpy is
// cheaper than scheduling GET_ROWS across the full worker thread pool and
// paying another graph barrier. Large batches stay on graph-side GET_ROWS.
// GGML_EPD_CPP_GATHER=0 disables the TG fast path for diagnostics.
static bool ep_cpp_gather_enabled() {
    static const bool value = []() {
        const char * env = getenv("GGML_EPD_CPP_GATHER");
        return !(env && env[0] != '\0' && strcmp(env, "0") == 0);
    }();
    return value;
}

// Periodically export the CPU backend's per-op timing table while a long-lived
// worker is running. The backend normally dumps it only during orderly process
// exit, but network transports may keep a worker inside a blocking receive until
// systemd's stop timeout. Keep this diagnostic fully opt-in and serialize it with
// graph execution through ep_compute_runtime::compute_mutex.
static void ep_op_timing_maybe_dump() {
    static const uint64_t every = []() -> uint64_t {
        const char * value = getenv("GGML_EPD_OP_TIMING_EVERY");
        if (!value || value[0] == '\0') {
            return 0;
        }
        char * end = nullptr;
        const unsigned long long parsed = strtoull(value, &end, 10);
        return end != value && *end == '\0' ? (uint64_t) parsed : 0;
    }();
    if (every == 0) {
        return;
    }
    static uint64_t calls = 0;
    if (++calls % every == 0) {
        ggml_backend_cpu_op_timing_dump();
    }
}

// env-gated weight repacking into the CPU_REPACK layout (GGML_EPD_REPACK=0 disables), default on
static bool ep_repack_enabled() {
    static const bool v = []() {
        const char * e = getenv("GGML_EPD_REPACK");
        return !(e && e[0] != '\0' && strcmp(e, "0") == 0);
    }();
    return v;
}

// Merge separate gate/up expert tensors into one repacked tensor at load time.
// This preserves the total weight bytes but removes one MMID dispatch and one
// activation quantization from every MoE layer. Disable for A/B or debugging.
static bool ep_fuse_gate_up_enabled() {
    static const bool v = []() {
        const char * e = getenv("GGML_EPD_FUSE_GATE_UP");
        return !(e && e[0] != '\0' && strcmp(e, "0") == 0);
    }();
    return v;
}

static bool ep_fuse_clamp_swiglu_enabled() {
    static const bool v = []() {
        const char * e = getenv("GGML_EPD_FUSE_CLAMP_SWIGLU");
        return !(e && e[0] != '\0' && strcmp(e, "0") == 0);
    }();
    return v;
}

// CPU_REPACK expert buffers are tens of GiB and are streamed on every MoE
// request.  On systems whose transparent-hugepage policy is "madvise", a
// regular backend allocation otherwise remains entirely in 4 KiB pages.  The
// hint is allocation-local, changes neither contents nor lifetime, and is safe
// to ignore when the kernel or allocator cannot honor it.
static bool ep_hugepages_enabled() {
    static const bool v = []() {
        const char * e = getenv("GGML_EPD_HUGEPAGES");
        return e && e[0] != '\0' && strcmp(e, "0") != 0;
    }();
    return v;
}

static void ep_advise_hugepages(void * base, size_t size) {
#if defined(MADV_HUGEPAGE)
    if (!ep_hugepages_enabled() || base == nullptr || size == 0) {
        return;
    }

    const long page_size = sysconf(_SC_PAGESIZE);
    if (page_size <= 0) {
        return;
    }

    const uintptr_t begin = (uintptr_t) base;
    const uintptr_t end = begin + size;
    const uintptr_t aligned_begin = (begin + (uintptr_t) page_size - 1) & ~((uintptr_t) page_size - 1);
    const uintptr_t aligned_end = end & ~((uintptr_t) page_size - 1);
    if (aligned_begin >= aligned_end) {
        return;
    }

    if (madvise((void *) aligned_begin, aligned_end - aligned_begin, MADV_HUGEPAGE) != 0) {
        LOG("llama-epd: repack: MADV_HUGEPAGE failed for %.2f GiB: %s\n",
            size / 1073741824.0, strerror(errno));
    } else {
        LOG("llama-epd: repack: advised huge pages for %.2f GiB CPU_REPACK buffer\n",
            size / 1073741824.0);
    }
#else
    GGML_UNUSED(base);
    GGML_UNUSED(size);
#endif
}

// env-gated startup weight prefault (GGML_EP_PREFAULT=1), default off
static bool ep_prefault_enabled() {
    static const bool v = []() {
        const char * e = getenv("GGML_EP_PREFAULT");
        return e && e[0] != '\0' && strcmp(e, "0") != 0;
    }();
    return v;
}

// A remote-only NUMA node benefits from hybrid polling. A worker colocated
// with the master often needs poll=0 so idle compute threads immediately yield
// the CPUs used by the master's intervening attention/local-MoE work.
static uint32_t ep_poll_level() {
    const char * e = getenv("GGML_EPD_POLL");
    if (!e || e[0] == '\0') {
        return 50;
    }
    char * end = nullptr;
    const long value = strtol(e, &end, 10);
    if (end == e || *end != '\0' || value < 0 || value > 100) {
        LOG("llama-epd: invalid GGML_EPD_POLL='%s' (want 0..100), using 50\n", e);
        return 50;
    }
    return (uint32_t) value;
}

// ---------------------------------------------------------------------------
// model (gguf metadata + read-only mmap, tensors point into the mapping)
//
// multi-split GGUF: each split's metadata lists only its own tensors (the first
// split may have none at all). every split is opened + mmap'd; tensors are
// registered in a global name -> tensor map with data pointing into the split's
// mapping. After ownership is resolved, only owned expert planes are kept and
// repacked; sparse mmap ownership is compacted once into anonymous memory.
// ---------------------------------------------------------------------------

struct ep_config {
    int layer_first  = 0;
    int layer_last   = 1 << 30;
    int expert_first = 0; // contiguous half-open [first, last), default mode
    int expert_last  = 1 << 30;
    int expert_mod_r = 0; // sparse mode: global id % expert_mod_n == expert_mod_r
    int expert_mod_n = 0;
    bool expert_list_set = false;
    std::vector<int32_t> expert_list; // sparse explicit global ids
    int n_threads    = 8;
};

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
    // full (pre-sharding) expert count from the gguf; n_expert is the owned
    // slice after --experts sharding, validation of incoming global expert ids
    // still uses the full range
    int64_t n_expert_full = 0;

    float clamp = 0.0f; // swiglu_clamp_exp[il], 0 = disabled
};

struct ep_shard {
    gguf_context * gguf = nullptr;
    ggml_context * ctx  = nullptr; // weight tensors (data -> this shard's mmap); freed by gguf_free

    int    fd        = -1;
    void * mmap_base = nullptr;
    size_t mmap_size = 0;

    const char * data_base = nullptr;

    // Anonymous compact source buffers (--no-mmap and sparse mmap), released
    // once their tensors have been converted to CPU_REPACK.
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

    // name -> (shard, absolute file offset of tensor data), used by no-mmap
    // plane reads and to associate sparse mmap compact buffers with a shard.
    std::map<std::string, std::pair<ep_shard *, uint64_t>> tensor_src;

    // CPU_REPACK buffers holding the owned expert weights in interleaved
    // layout (ep_repack_weights); empty when repack is unavailable/disabled
    std::vector<ggml_backend_buffer_t> repack_bufs;
    ggml_context * fused_ctx = nullptr; // metadata for load-time gate/up fusion

    ~ep_model() {
        for (ggml_backend_buffer_t b : repack_bufs) {
            ggml_backend_buffer_free(b);
        }
        if (fused_ctx) {
            ggml_free(fused_ctx);
        }
    }

    std::string arch;
    int n_layer = 0;

    llama_ep_expert_map expert_map;

    std::map<int, ep_layer> layers; // owned MoE layers
};

// Remove page-table residency for raw file-backed weight pages after their
// CPU_REPACK copy is complete. MADV_DONTNEED on a MAP_SHARED file mapping does
// not invalidate the file or another process's mapping; it only makes these
// clean pages immediately reclaimable for this worker. Boundary pages are kept
// because adjacent GGUF tensors may share them.
static size_t ep_drop_raw_mmap_pages(const ep_model & m, const void * data, size_t size) {
    if (data == nullptr || size == 0) {
        return 0;
    }
    const long page_long = sysconf(_SC_PAGESIZE);
    if (page_long <= 0) {
        return 0;
    }
    const uintptr_t page = (uintptr_t) page_long;
    const uintptr_t raw_begin = (uintptr_t) data;
    if (size > UINTPTR_MAX - raw_begin) {
        return 0;
    }
    const uintptr_t raw_end = raw_begin + size;

    for (const auto & shard : m.shards) {
        if (shard->mmap_base == nullptr || shard->mmap_base == MAP_FAILED) {
            continue;
        }
        const uintptr_t map_begin = (uintptr_t) shard->mmap_base;
        if (shard->mmap_size > UINTPTR_MAX - map_begin) {
            continue;
        }
        const uintptr_t map_end = map_begin + shard->mmap_size;
        if (raw_begin < map_begin || raw_end > map_end) {
            continue;
        }

        const uintptr_t begin_remainder = raw_begin % page;
        const uintptr_t begin_padding = begin_remainder == 0 ? 0 : page - begin_remainder;
        if (begin_padding > UINTPTR_MAX - raw_begin) {
            return 0;
        }
        const uintptr_t begin = raw_begin + begin_padding;
        const uintptr_t end = (raw_end / page) * page;
        if (end <= begin) {
            return 0;
        }
        const size_t bytes = (size_t) (end - begin);
        if (madvise((void *) begin, bytes, MADV_DONTNEED) != 0) {
            LOG("llama-epd: repack: madvise raw mmap pages failed: %s\n", strerror(errno));
            return 0;
        }
        return bytes;
    }
    return 0;
}

static bool gguf_get_str(gguf_context * g, const char * key, std::string & out) {
    int64_t id = gguf_find_key(g, key);
    if (id < 0) {
        return false;
    }
    out = gguf_get_val_str(g, id);
    return true;
}

// Open one split and register all of its tensors. mmap mode points tensor data
// into a read-only shared mapping; both modes record file offsets for later
// ownership compaction and no-mmap plane reads.
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
        m.tensor_src.emplace(name, std::make_pair(sh.get(), data_off + gguf_get_tensor_offset(sh->gguf, tid)));
        if (no_mmap) {
            // data stays nullptr until the owned tensors are read in after layer claiming
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

// --no-mmap: read every owned global expert plane into a compact anonymous
// buffer. Contiguous ownership remains a near-sequential sweep; sparse maps do
// one pread per selected plane. Unlike mmap these pages cannot be evicted.
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
            // The file holds the full tensor. Compact the configured global
            // expert planes into local_to_global order.
            const size_t plane = ggml_row_size(t->type, t->ne[0]) * (size_t) t->ne[1];
            void * buf = nullptr;
            if (posix_memalign(&buf, 64, n) != 0) {
                LOG("llama-epd: no-mmap: posix_memalign %.2f GiB failed\n", n / 1073741824.0);
                return false;
            }
            for (size_t local = 0; local < m.expert_map.local_to_global.size(); ++local) {
                const int32_t global = m.expert_map.local_to_global[local];
                const uint64_t read_off = off + (uint64_t) global * plane;
                size_t done = 0;
                while (done < plane) {
                    const ssize_t r = pread(sh->fd, (char *) buf + local * plane + done,
                            std::min<size_t>(plane - done, 32 << 20), read_off + done);
                    if (r <= 0) {
                        LOG("llama-epd: no-mmap: pread %s expert %d: %s\n", t->name, global,
                                r == 0 ? "unexpected EOF" : strerror(errno));
                        free(buf);
                        return false;
                    }
                    done += (size_t) r;
                }
            }
            t->data = buf;
            sh->load_bufs.push_back(buf);
        }
    }

    LOG("llama-epd: no-mmap: done in %.1f s (%.2f GB/s)\n",
        (ggml_time_us() - t0) / 1e6, total / 1073741824.0 / ((ggml_time_us() - t0) / 1e6));
    return true;
}

// ---------------------------------------------------------------------------
// CPU_REPACK weight layout
//
// The worker's MoE graph runs mul_mat_id with the expert weights as src0. In a
// plain CPU buffer that op falls back to the generic vec_dot path; the same op
// with src0 in a CPU_REPACK buffer (tensor->extra set by the buffer's
// init_tensor) dispatches to the repack gemv/gemm kernels (8x8 interleaved,
// AVX512). Selection is type-specific: formats/shapes that benchmark worse in
// repack form stay on the mature row-major vec_dot path. The conversion is a
// one-time load-time cost. Tensors whose type/shape has no selected repack trait
// keep their original storage; when CPU_REPACK is unavailable the step is
// skipped entirely.
// ---------------------------------------------------------------------------
static void ep_repack_weights(ep_model & m) {
    if (!ep_repack_enabled()) {
        LOG("llama-epd: repack: disabled (GGML_EPD_REPACK=0), keeping raw weight layout\n");
        return;
    }

    // locate the CPU_REPACK buffer type via the CPU device's extra bufts
    ggml_backend_buffer_type_t repack_buft = nullptr;
    if (ggml_backend_dev_t cpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU)) {
        ggml_backend_reg_t cpu_reg = ggml_backend_dev_backend_reg(cpu_dev);
        auto get_extra_bufts = (ggml_backend_dev_get_extra_bufts_t)
            ggml_backend_reg_get_proc_address(cpu_reg, "ggml_backend_dev_get_extra_bufts");
        if (get_extra_bufts) {
            for (ggml_backend_buffer_type_t * extra = get_extra_bufts(cpu_dev); extra && *extra; ++extra) {
                if (strcmp(ggml_backend_buft_name(*extra), "CPU_REPACK") == 0) {
                    repack_buft = *extra;
                    break;
                }
            }
        }
    }
    if (!repack_buft) {
        LOG("llama-epd: repack: CPU_REPACK buffer type not available, keeping raw weight layout\n");
        return;
    }

    // Probe traits before allocating the destination. A mixed-quant model may
    // deliberately keep one type row-major (for example Q2_K on AVX512-VNNI)
    // while repacking MXFP4. Reserving space for unsupported tensors would keep
    // their raw storage and also strand an equally large hole in CPU_REPACK.
    // A zero-sized ggml allocation is a generic dummy buffer with an empty
    // interface; it deliberately bypasses buft->alloc_buffer. Such a buffer
    // never calls CPU_REPACK's init_tensor and would classify every tensor as
    // unsupported. Allocate one aligned unit so the real buffer callbacks are
    // installed without reserving weight-sized storage.
    const size_t probe_size = std::max<size_t>(1, ggml_backend_buft_get_alignment(repack_buft));
    ggml_backend_buffer_t probe = ggml_backend_buft_alloc_buffer(repack_buft, probe_size);
    if (!probe) {
        LOG("llama-epd: repack: trait probe buffer allocation failed, keeping raw weight layout\n");
        return;
    }

    auto probe_supported = [probe](ggml_tensor * t) {
        ggml_backend_buffer_t saved_buffer = t->buffer;
        void * saved_data = t->data;
        void * saved_extra = t->extra;
        t->buffer = probe;
        t->extra = nullptr;
        ggml_backend_buffer_init_tensor(probe, t);
        const bool supported = t->extra != nullptr;
        t->buffer = saved_buffer;
        t->data = saved_data;
        t->extra = saved_extra;
        return supported;
    };

    struct fused_gate_up_source {
        ep_layer * layer = nullptr;
        ggml_tensor * fused = nullptr;
        ggml_tensor * gate = nullptr;
        ggml_tensor * up = nullptr;
        bool selected = false;
    };
    std::vector<fused_gate_up_source> fusions;
    std::map<ep_layer *, size_t> fusion_by_layer;

    if (ep_fuse_gate_up_enabled()) {
        size_t eligible = 0;
        for (auto & kv : m.layers) {
            const ep_layer & L = kv.second;
            eligible += L.gate_up == nullptr && L.gate != nullptr && L.up != nullptr &&
                L.gate->type == L.up->type && ggml_are_same_shape(L.gate, L.up) ? 1 : 0;
        }
        if (eligible > 0) {
            ggml_init_params params;
            params.mem_size = ggml_tensor_overhead() * (eligible + 1);
            params.mem_buffer = nullptr;
            params.no_alloc = true;
            m.fused_ctx = ggml_init(params);
            if (!m.fused_ctx) {
                LOG("llama-epd: repack: gate/up fusion metadata allocation failed, keeping separate tensors\n");
            } else {
                fusions.reserve(eligible);
                for (auto & kv : m.layers) {
                    ep_layer & L = kv.second;
                    if (L.gate_up != nullptr || L.gate == nullptr || L.up == nullptr ||
                            L.gate->type != L.up->type || !ggml_are_same_shape(L.gate, L.up)) {
                        continue;
                    }
                    ggml_tensor * fused = ggml_new_tensor_3d(
                        m.fused_ctx, L.gate->type, L.n_embd, 2 * L.n_ff, L.n_expert);
                    char fused_name[GGML_MAX_NAME];
                    snprintf(fused_name, sizeof(fused_name), "blk.%d.ffn_gate_up_exps.weight (epd-fused)", L.il);
                    ggml_set_name(fused, fused_name);
                    fusion_by_layer[&L] = fusions.size();
                    fusions.push_back({&L, fused, L.gate, L.up, probe_supported(fused)});
                }
            }
        }
    }

    std::vector<ggml_tensor *> candidates;
    std::map<ggml_tensor *, fused_gate_up_source *> fused_sources;
    for (auto & kv : m.layers) {
        ep_layer & L = kv.second;
        const auto fit = fusion_by_layer.find(&L);
        if (fit != fusion_by_layer.end() && fusions[fit->second].selected) {
            fused_gate_up_source & fusion = fusions[fit->second];
            candidates.push_back(fusion.fused);
            fused_sources[fusion.fused] = &fusion;
        } else {
            for (ggml_tensor * t : {L.gate_up, L.gate, L.up}) {
                if (t) candidates.push_back(t);
            }
        }
        if (L.down) candidates.push_back(L.down);
    }

    std::vector<ggml_tensor *> wts;
    wts.reserve(candidates.size());
    std::map<ggml_type, std::pair<size_t, size_t>> raw_by_type;
    for (ggml_tensor * t : candidates) {
        if (probe_supported(t)) {
            wts.push_back(t);
        } else {
            auto & stats = raw_by_type[t->type];
            stats.first++;
            stats.second += ggml_nbytes(t);
        }
    }
    ggml_backend_buffer_free(probe);

    for (const auto & item : raw_by_type) {
        LOG("llama-epd: repack: %zu %s tensor(s), %.2f GiB keep raw layout by format policy\n",
            item.second.first, ggml_type_name(item.first), item.second.second / 1073741824.0);
    }

    if (wts.empty()) {
        LOG("llama-epd: repack: no tensor has matching repack traits, keeping raw weight layout\n");
        return;
    }

    size_t total = 0;
    for (const ggml_tensor * t : wts) {
        total += ggml_backend_buft_get_alloc_size(repack_buft, t);
    }
    LOG("llama-epd: repack: converting %.2f GiB of expert weights (%zu/%zu tensors) to CPU_REPACK layout\n",
        total / 1073741824.0, wts.size(), candidates.size());
    const int64_t t0 = ggml_time_us();

    ggml_backend_buffer_t buf = ggml_backend_buft_alloc_buffer(repack_buft, total);
    if (!buf) {
        LOG("llama-epd: repack: buffer alloc failed, keeping raw weight layout\n");
        return;
    }

    // place every tensor in the repack buffer; init_tensor sets tensor->extra
    // to the repack traits, or leaves it nullptr when the type/shape has none
    char * base = (char *) ggml_backend_buffer_get_base(buf);
    size_t off = 0;
    struct repack_job {
        ggml_tensor * tensor;
        const void *  src[2] = {nullptr, nullptr};
        size_t        src_size[2] = {0, 0};
        int           n_src = 0;
        fused_gate_up_source * fusion = nullptr;
        ep_shard *    source_shard[2] = {nullptr, nullptr};
        size_t        source_index[2] = {0, 0};
    };
    std::vector<repack_job> jobs;
    for (ggml_tensor * t : wts) {
        repack_job job;
        job.tensor = t;
        const auto fused_it = fused_sources.find(t);
        if (fused_it != fused_sources.end()) {
            job.fusion = fused_it->second;
            job.src[0] = job.fusion->gate->data;
            job.src[1] = job.fusion->up->data;
            job.src_size[0] = ggml_nbytes(job.fusion->gate);
            job.src_size[1] = ggml_nbytes(job.fusion->up);
            job.n_src = 2;
        } else {
            job.src[0] = t->data;
            job.src_size[0] = ggml_nbytes(t);
            job.n_src = 1;
        }
        t->buffer = buf;
        t->data   = base + off;
        off += ggml_backend_buffer_get_alloc_size(buf, t);
        ggml_backend_buffer_init_tensor(buf, t);
        if (t->extra) {
            // Sparse ownership and --no-mmap store every source tensor in its
            // own anonymous allocation. Remember the slot so the conversion
            // worker can release it immediately instead of retaining the
            // complete raw copy until all tensors have been repacked. This
            // keeps startup RSS near one weight copy for very large models.
            for (int source = 0; source < job.n_src; ++source) {
                for (auto & sh : m.shards) {
                    for (size_t i = 0; i < sh->load_bufs.size(); ++i) {
                        if (sh->load_bufs[i] == job.src[source]) {
                            job.source_shard[source] = sh.get();
                            job.source_index[source] = i;
                            break;
                        }
                    }
                    if (job.source_shard[source]) {
                        break;
                    }
                }
            }
            jobs.push_back(job);
        } else {
            GGML_ABORT("%s: repack traits changed between probe and allocation for %s", __func__, t->name);
        }
    }

    if (jobs.empty()) {
        LOG("llama-epd: repack: no tensor has matching repack traits, keeping raw weight layout\n");
        ggml_backend_buffer_free(buf);
        return;
    }

    // set_tensor runs the actual layout conversion. Fused jobs stage one
    // layer's gate+up bytes temporarily; four converters saturate memory
    // bandwidth while bounding peak startup RSS.
    const int nth = (int) std::min<size_t>(4, jobs.size());
    std::atomic<size_t> next{0};
    std::atomic<size_t> dropped_mmap_bytes{0};
    std::vector<std::thread> pool;
    pool.reserve((size_t) nth);
    for (int i = 0; i < nth; ++i) {
        pool.emplace_back([&]() {
            for (;;) {
                const size_t idx = next.fetch_add(1);
                if (idx >= jobs.size()) {
                    break;
                }
                repack_job & job = jobs[idx];
                if (job.fusion) {
                    const size_t merged_bytes = ggml_nbytes(job.tensor);
                    void * merged = nullptr;
                    if (posix_memalign(&merged, 64, merged_bytes) != 0) {
                        GGML_ABORT("%s: gate/up fusion staging allocation failed for %s", __func__, job.tensor->name);
                    }
                    if (!llama_ep_merge_gate_up_raw(
                            merged, merged_bytes,
                            job.src[0], ggml_nbytes(job.fusion->gate),
                            job.src[1], ggml_nbytes(job.fusion->up),
                            (size_t) job.tensor->ne[2])) {
                        free(merged);
                        GGML_ABORT("%s: invalid gate/up fusion layout for %s", __func__, job.tensor->name);
                    }
                    ggml_backend_tensor_set(job.tensor, merged, 0, merged_bytes);
                    free(merged);
                } else {
                    ggml_backend_tensor_set(job.tensor, job.src[0], 0, ggml_nbytes(job.tensor));
                }
                for (int source = 0; source < job.n_src; ++source) {
                    if (job.source_shard[source]) {
                        free(const_cast<void *>(job.src[source]));
                        // Jobs refer to distinct tensor allocations, so each
                        // thread writes a distinct pre-existing vector element.
                        job.source_shard[source]->load_bufs[job.source_index[source]] = nullptr;
                    } else {
                        // File-backed sources can be discarded as soon as this
                        // tensor has converted. Waiting for every model layer
                        // would transiently retain raw + repacked full-model
                        // RSS and can exceed a 128-GiB NUMA node.
                        dropped_mmap_bytes.fetch_add(
                            ep_drop_raw_mmap_pages(m, job.src[source], job.src_size[source]),
                            std::memory_order_relaxed);
                    }
                }
            }
        });
    }
    for (auto & th : pool) {
        th.join();
    }

    const size_t dropped = dropped_mmap_bytes.load(std::memory_order_relaxed);
    if (dropped > 0) {
        LOG("llama-epd: repack: dropped %.2f GiB of converted raw mmap pages from worker RSS\n",
            dropped / 1073741824.0);
    }

    // Remove slots already released by the conversion workers. mmap-backed
    // sources have no load_buf entry and remain owned by their shard mapping.
    for (auto & sh : m.shards) {
        sh->load_bufs.erase(
                std::remove(sh->load_bufs.begin(), sh->load_bufs.end(), nullptr),
                sh->load_bufs.end());
    }

    // Advise only after conversion released the raw source allocations.  On a
    // large sparse worker, advising the untouched destination before repack can
    // make first-touch huge-page allocation overlap the complete raw copy and
    // create a large transient memory-pressure spike.
    ep_advise_hugepages(base, total);
    m.repack_bufs.push_back(buf);

    size_t fused_count = 0;
    for (fused_gate_up_source & fusion : fusions) {
        if (!fusion.selected) {
            continue;
        }
        fusion.layer->gate_up = fusion.fused;
        fusion.layer->gate = nullptr;
        fusion.layer->up = nullptr;
        fused_count++;
    }

    LOG("llama-epd: repack: %zu/%zu tensors converted, %zu gate/up pairs fused in %.1f s\n",
        jobs.size(), wts.size(), fused_count, (ggml_time_us() - t0) / 1e6);
}

// GGML_NUMA_EP=1 per-node row-window placement. Mirrors
// llama_model::numa_ep_place_experts (src/llama-model.cpp) for the epd loader:
// within every repacked expert tensor, each NUMA node's row window (the rows
// the compute side claims for that node, in 128-row blocks) is mbind()'d with
// MPOL_MF_MOVE onto that node, so the repack gemv/gemm kernels read node-local
// memory instead of interleaved/first-touch placement. Must run after
// ep_repack_weights (t->data must point into the CPU_REPACK buffer); tensors
// that kept the raw layout have no per-node claim path and are skipped.
static void ep_numa_tp_place(ep_model & m) {
    const char * env = getenv("GGML_NUMA_EP");
    if (!env || atoi(env) == 0) {
        return;
    }
    const int n_nodes = ggml_numa_node_count();
    if (n_nodes < 2) {
        LOG("llama-epd: numa-ep: GGML_NUMA_EP set but ggml reports %d node(s); placement skipped\n", n_nodes);
        return;
    }
    long pg = sysconf(_SC_PAGESIZE);
    if (pg <= 0) {
        pg = 4096;
    }

    const int64_t t0 = ggml_time_us();
    int n_placed = 0;
    size_t placed_bytes = 0;
    int n_raw = 0;
    for (auto & kv : m.layers) {
        const ep_layer & L = kv.second;
        for (ggml_tensor * t : {L.gate_up, L.gate, L.up, L.down}) {
            if (t == nullptr || t->data == nullptr || t->ne[2] <= 1) {
                continue;
            }
            if (t->extra == nullptr) {
                ++n_raw; // raw layout: the generic mul_mat_id path claims no per-node rows
                continue;
            }
            const int64_t n_expert = t->ne[2];
            const size_t eb = t->nb[2]; // bytes per expert plane (contiguous along ne[2])
            // per-node row window inside each expert plane; must match the compute side
            // in repack.cpp (rows claimed in blocks of ep_chunk, windows aligned to 128
            // rows so any ep_chunk <= 128 divides them evenly)
            const int64_t ne1 = t->ne[1];
            const size_t rb = t->nb[1];
            char * base = (char *) t->data;
            for (int64_t e = 0; e < n_expert; ++e) {
                char * ebase = base + e * eb;
                for (int n = 0; n < n_nodes; ++n) {
                    ggml_shard_window window;
                    if (!ggml_shard_window_equal(ne1, (size_t) n_nodes, (size_t) n, 128, window)) {
                        continue;
                    }
                    const int64_t r0 = window.begin;
                    const int64_t r1 = window.end;
                    if (r1 <= r0) {
                        continue;
                    }
                    char * p = ebase + r0 * rb;
                    size_t sz = (size_t) (r1 - r0) * rb;
                    // keep the boundary page on the earlier node: only bind whole pages
                    const uintptr_t up = ((uintptr_t) p + pg - 1) & ~((uintptr_t) pg - 1);
                    if (n > 0) {
                        if ((size_t) (up - (uintptr_t) p) >= sz) {
                            continue;
                        }
                        sz -= up - (uintptr_t) p;
                        p = (char *) up;
                    }
                    ggml_numa_bind(p, sz, n);
                }
            }
            ++n_placed;
            placed_bytes += ggml_nbytes(t);
        }
    }
    LOG("llama-epd: numa-ep: placed %.2f GiB of expert weights across %d nodes (%d tensors, %d raw-layout tensors skipped) in %.1f s\n",
        placed_bytes / 1073741824.0, n_nodes, n_placed, n_raw, (ggml_time_us() - t0) / 1e6);
}

// probe one tensor name in the global map; returns nullptr if absent
static ggml_tensor * ep_get_tensor(ep_model & m, const char * name) {
    auto it = m.tensors.find(name);
    return it == m.tensors.end() ? nullptr : it->second;
}

static bool ep_model_load(ep_model & m, const char * path, int layer_first, int layer_last, bool no_mmap,
                          const ep_config & cfg) {
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

        // Resolve global expert ownership once. Every supported MoE layer must
        // expose the same full expert count; local tensor plane i then always
        // corresponds to expert_map.local_to_global[i].
        L.n_expert_full = L.n_expert;
        {
            const int64_t n_full = L.n_expert;
            if (m.expert_map.n_expert == 0) {
                bool map_ok = false;
                if (cfg.expert_list_set) {
                    map_ok = m.expert_map.init_ids((int32_t) n_full, cfg.expert_list);
                } else if (cfg.expert_mod_n > 0) {
                    map_ok = m.expert_map.init_mod((int32_t) n_full, cfg.expert_mod_r, cfg.expert_mod_n);
                } else {
                    map_ok = m.expert_map.init_range((int32_t) n_full, cfg.expert_first, cfg.expert_last);
                }
                if (!map_ok) {
                    LOG("llama-epd: invalid or empty expert ownership map for %lld experts\n",
                            (long long) n_full);
                    return false;
                }
            } else if (m.expert_map.n_expert != n_full) {
                LOG("llama-epd: layer %d expert count %lld differs from prior layers (%d)\n",
                        il, (long long) n_full, m.expert_map.n_expert);
                return false;
            }

            auto slice_t = [&](ggml_tensor * t) -> bool {
                if (!t) {
                    return true;
                }
                const size_t plane = ggml_nbytes(t) / (size_t) n_full;
                if (!no_mmap && t->data) {
                    if (m.expert_map.contiguous) {
                        t->data = (char *) t->data + (size_t) m.expert_map.first * plane;
                    } else {
                        const size_t n = plane * m.expert_map.local_to_global.size();
                        void * buf = nullptr;
                        if (posix_memalign(&buf, 64, n) != 0) {
                            LOG("llama-epd: sparse mmap copy alloc %.2f GiB failed\n", n / 1073741824.0);
                            return false;
                        }
                        const char * src = (const char *) t->data;
                        for (size_t local = 0; local < m.expert_map.local_to_global.size(); ++local) {
                            memcpy((char *) buf + local * plane,
                                    src + (size_t) m.expert_map.local_to_global[local] * plane, plane);
                        }
                        t->data = buf;
                        auto sit = m.tensor_src.find(t->name);
                        if (sit == m.tensor_src.end()) {
                            free(buf);
                            LOG("llama-epd: sparse mmap copy has no owner for %s\n", t->name);
                            return false;
                        }
                        sit->second.first->load_bufs.push_back(buf);
                    }
                }
                t->ne[2] = (int64_t) m.expert_map.local_to_global.size();
                return true;
            };
            if (!slice_t(L.gate_up) || !slice_t(L.gate) || !slice_t(L.up) || !slice_t(L.down)) {
                return false;
            }
            L.n_expert = (int64_t) m.expert_map.local_to_global.size();
        }

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
    ep_repack_weights(m);
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

// Per-shape cached MoE FFN graphs: building the ggml graph (context, tensors,
// build_forward_expand) costs ~0.2 ms per request - dominant at decode shapes
// where the actual compute is ~0.1 ms.  Cache small graphs and reuse them,
// swapping only the input contents.  Large PP shapes deliberately use the
// shared grow-only allocator instead: ragged routing produces many distinct
// row counts, and permanently caching every large graph can consume tens of
// GiB. Callers serialize compute on their worker runtime's compute mutex.
static size_t ep_graph_cache_budget() {
    static const size_t bytes = []() {
        const char * e = getenv("GGML_EPD_GRAPH_CACHE_MIB");
        if (!e || e[0] == '\0') {
            return (size_t) 512 << 20;
        }
        char * end = nullptr;
        const unsigned long long mib = strtoull(e, &end, 10);
        if (end == e || *end != '\0') {
            return (size_t) 512 << 20;
        }
        if (mib > (unsigned long long) (SIZE_MAX >> 20)) {
            return SIZE_MAX;
        }
        return (size_t) mib << 20;
    }();
    return bytes;
}

// Repack MUL_MAT_ID accepts a prequantized activation when it matches the
// kernel's dot-product type. Separate gate/up projections can therefore share
// one activation quantization instead of each filling its own workspace.
static ggml_type ep_repack_activation_type(const ggml_tensor * t) {
    if (t == nullptr || t->buffer == nullptr || t->extra == nullptr ||
            strcmp(ggml_backend_buffer_name(t->buffer), "CPU_REPACK") != 0) {
        return GGML_TYPE_COUNT;
    }
    switch (t->type) {
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_IQ4_NL:
        case GGML_TYPE_IQ4_XS:
        case GGML_TYPE_MXFP4:
        case GGML_TYPE_Q8_0:
            return GGML_TYPE_Q8_0;
        case GGML_TYPE_Q2_K:
        case GGML_TYPE_Q3_K:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_Q6_K:
        case GGML_TYPE_IQ1_S:
        case GGML_TYPE_IQ1_M:
        case GGML_TYPE_IQ2_XXS:
        case GGML_TYPE_IQ2_XS:
        case GGML_TYPE_IQ3_XXS:
            return GGML_TYPE_Q8_K;
        default:
            return GGML_TYPE_COUNT;
    }
}

static int64_t ep_shared_q8_min_tokens() {
    static const int64_t value = []() -> int64_t {
        const char * e = getenv("GGML_EPD_SHARED_Q8_MIN_TOKENS");
        if (e == nullptr || e[0] == '\0') {
            return (int64_t) 2;
        }
        char * end = nullptr;
        const long long parsed = strtoll(e, &end, 10);
        return end != e && *end == '\0' ? std::max(0LL, parsed) : 2LL;
    }();
    return value;
}

static void ep_graph_cache_make_room(ep_compute_runtime & rt, size_t incoming) {
    const size_t budget = ep_graph_cache_budget();
    while (!rt.graph_cache.empty() &&
           (rt.graph_cache_bytes > budget || incoming > budget - rt.graph_cache_bytes)) {
        auto oldest = rt.graph_cache.begin();
        for (auto it = std::next(rt.graph_cache.begin()); it != rt.graph_cache.end(); ++it) {
            if (it->second->last_use < oldest->second->last_use) {
                oldest = it;
            }
        }
        if (ep_debug_enabled()) {
            LOG("llama-epd: [ep-debug] graph cache evict layer=%p rows=%d ids=%d bytes=%.1f MiB\n",
                oldest->first.layer, oldest->first.n_tokens, oldest->first.n_ids,
                oldest->second->bytes / 1048576.0);
        }
        rt.graph_cache_bytes -= oldest->second->bytes;
        rt.graph_cache.erase(oldest);
    }
}

static bool ep_moe_ffn(
        ep_compute_runtime & rt,
        const ep_layer  & L,
        int               n_tokens,
        int               n_ids,          // experts per token
        const int32_t   * ids,            // [n_tokens*n_ids]
        const float     * weights,        // [n_tokens*n_ids]
        int               n_hidden_tokens,
        const int32_t   * hidden_token_idx, // optional [n_tokens], gather from hidden
        const float     * hidden,         // [n_hidden_tokens*n_embd]
        float           * out,            // [n_tokens*n_embd] (no_sum=false) or
                                          // [n_embd*n_ids*n_tokens] (no_sum=true)
        bool              no_sum,         // REQ2: return per-slot weighted vectors
        bool              apply_weights,  // false: master applies router weights (REQ4)
        std::string     & err) {

    const int64_t n_embd = L.n_embd;
    const int64_t n_ff   = L.n_ff;

    // Graph cache is safe by default now that each cached graph has its own
    // gallocr (buffer never moves, no stale pointers). Disable with
    // GGML_EPD_NO_GRAPH_CACHE=1.
    static const bool no_graph_cache = []() {
        const char * e = getenv("GGML_EPD_NO_GRAPH_CACHE");
        return e && e[0] != '\0' && strcmp(e, "0") != 0;
    }();
    static const int64_t graph_cache_max_rows = []() -> int64_t {
        const char * e = getenv("GGML_EPD_GRAPH_CACHE_MAX_ROWS");
        if (!e || e[0] == '\0') {
            return (int64_t) 64;
        }
        char * end = nullptr;
        const long long v = strtoll(e, &end, 10);
        return end != e && *end == '\0' ? std::max(0LL, v) : 64LL;
    }();
    const bool use_graph_cache = !no_graph_cache && ep_graph_cache_budget() > 0 &&
        (int64_t) n_tokens * n_ids <= graph_cache_max_rows;
    if (n_hidden_tokens < 1 || (hidden_token_idx == nullptr && n_hidden_tokens != n_tokens)) {
        err = "invalid hidden row mapping";
        return false;
    }
    const bool indexed_hidden = hidden_token_idx != nullptr;
    const ep_graph_cache_key cache_key = {
        (const void *) &L, n_tokens, n_ids, n_hidden_tokens, indexed_hidden, no_sum, apply_weights,
    };
    std::unique_ptr<ep_graph_cache_entry> local_entry;
    ep_graph_cache_entry * ce = nullptr;
    bool cache_hit = false;
    if (use_graph_cache) {
        const auto it = rt.graph_cache.find(cache_key);
        if (it != rt.graph_cache.end()) {
            ce = it->second.get();
            ce->last_use = ++rt.graph_cache_clock;
            cache_hit = true;
        }
    }
    if (ce == nullptr) {
        local_entry.reset(new ep_graph_cache_entry);
        ce = local_entry.get();
    }

    if (ce->gf == nullptr) {
        ggml_init_params iparams;
        iparams.mem_size   = ggml_tensor_overhead() * 128 + ggml_graph_overhead();
        iparams.mem_buffer = nullptr;
        iparams.no_alloc   = true;

        ggml_context * ctx = ggml_init(iparams);
        if (!ctx) {
            err = "ggml_init failed";
            return false;
        }

        ggml_tensor * hidden_t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, n_hidden_tokens);
        ggml_tensor * ids_t    = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, n_ids, n_tokens);
        ggml_tensor * w_t      = apply_weights
            ? ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1, n_ids, n_tokens)
            : nullptr;
        ggml_set_input(hidden_t);
        ggml_set_input(ids_t);
        if (w_t != nullptr) {
            ggml_set_input(w_t);
        }

        ggml_tensor * hidden_idx_t = nullptr;
        ggml_tensor * selected_hidden = hidden_t;
        if (indexed_hidden) {
            hidden_idx_t = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_tokens);
            ggml_set_input(hidden_idx_t);
            selected_hidden = ggml_get_rows(ctx, hidden_t, hidden_idx_t);
        }
        ggml_tensor * cur = ggml_reshape_3d(ctx, selected_hidden, n_embd, 1, n_tokens);

        ggml_tensor * gate = nullptr;
        ggml_tensor * up   = nullptr;

        if (L.gate_up) {
            // merged gate_up path (llama-graph.cpp:2023-2041)
            ggml_tensor * gu = ggml_mul_mat_id(ctx, L.gate_up, cur, ids_t); // [2*n_ff, n_ids, n_tokens]
            gate = ggml_view_3d(ctx, gu, n_ff, n_ids, n_tokens, gu->nb[1], gu->nb[2], 0);
            up   = ggml_view_3d(ctx, gu, n_ff, n_ids, n_tokens, gu->nb[1], gu->nb[2], n_ff * gu->nb[0]);
        } else {
            // separate path (llama-graph.cpp:2042-2071)
            ggml_tensor * gate_up_input = cur;
            const ggml_type gate_act = ep_repack_activation_type(L.gate);
            const ggml_type up_act   = ep_repack_activation_type(L.up);
            const int64_t shared_q8_min_tokens = ep_shared_q8_min_tokens();
            if (shared_q8_min_tokens > 0 && n_tokens >= shared_q8_min_tokens && gate_act == up_act &&
                    (gate_act == GGML_TYPE_Q8_0 || gate_act == GGML_TYPE_Q8_K)) {
                gate_up_input = ggml_cast(ctx, cur, gate_act);
            }
            up   = ggml_mul_mat_id(ctx, L.up,   gate_up_input, ids_t); // [n_ff, n_ids, n_tokens]
            gate = ggml_mul_mat_id(ctx, L.gate, gate_up_input, ids_t);
        }

        // DeepSeek4 clamped SWIGLU. The CPU kernel interprets op-param slot 2
        // as a positive clamp limit and fuses both clamps into its SWIGLU
        // vector pass. This worker runtime is CPU-only, so no other backend can
        // observe the private convention. GGML_EPD_FUSE_CLAMP_SWIGLU=0 keeps
        // the three-node reference graph for A/B and emergency fallback.
        if (L.clamp > 1e-6f && ep_fuse_clamp_swiglu_enabled()) {
            cur = ggml_swiglu_split_clamped(ctx, gate, up, L.clamp);
        } else {
            if (L.clamp > 1e-6f) {
                up   = ggml_clamp(ctx, up, -L.clamp, L.clamp);
                gate = ggml_clamp(ctx, gate, -INFINITY, L.clamp);
            }
            cur = ggml_swiglu_split(ctx, gate, up);
        }

        ggml_tensor * experts = ggml_mul_mat_id(ctx, L.down, cur, ids_t); // [n_embd, n_ids, n_tokens]
        if (apply_weights) {
            experts = ggml_mul(ctx, experts, w_t);
        }

        // REQ2 (no_sum): skip the sum chain and ship the per-slot weighted vectors;
        // the master merges them in ascending global slot order (SCHEDULER-DESIGN §4.5)
        ggml_tensor * result = experts;
        if (!no_sum) {
            // sum over the n_ids (expert slot) dimension via views + adds (llama-graph.cpp:2165-2183)
            ggml_tensor * sum = ggml_view_2d(ctx, experts, n_embd, n_tokens, experts->nb[2], 0);
            for (int i = 1; i < n_ids; ++i) {
                ggml_tensor * v = ggml_view_2d(ctx, experts, n_embd, n_tokens, experts->nb[2], i * experts->nb[1]);
                sum = ggml_add(ctx, sum, v);
            }
            result = sum;
        }

        ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, result);

        ce->ctx      = ctx;
        ce->gf       = gf;
        ce->hidden_t = hidden_t;
        ce->hidden_idx_t = hidden_idx_t;
        ce->ids_t    = ids_t;
        ce->w_t      = w_t;
        ce->result   = result;
        if (use_graph_cache) {
            // Dedicated allocator so this cached graph's buffer never moves.
            ce->gallocr = ggml_gallocr_new(ggml_backend_cpu_buffer_type());
            if (!ce->gallocr) {
                err = "ggml_gallocr_new failed";
                return false;
            }
        }
    }

    // cached graphs use their own allocator (buffer never moves -> pointers stay
    // valid); the no-cache path rebuilds every request and uses the shared one
    ggml_gallocr_t allocr = use_graph_cache ? ce->gallocr : rt.gallocr;
    if (!ggml_gallocr_alloc_graph(allocr, ce->gf)) {
        err = "failed to allocate compute tensors";
        return false;
    }

    if (use_graph_cache && !cache_hit) {
        ce->bytes = ggml_gallocr_get_buffer_size(ce->gallocr, 0);
        if (ce->bytes <= ep_graph_cache_budget()) {
            ep_graph_cache_make_room(rt, ce->bytes);
            ce->last_use = ++rt.graph_cache_clock;
            rt.graph_cache_bytes += ce->bytes;
            rt.graph_cache.emplace(cache_key, std::move(local_entry));
        } else if (ep_debug_enabled()) {
            LOG("llama-epd: [ep-debug] graph cache bypass rows=%d ids=%d bytes=%.1f MiB budget=%.1f MiB\n",
                n_tokens, n_ids, ce->bytes / 1048576.0, ep_graph_cache_budget() / 1048576.0);
        }
    }

    ggml_backend_tensor_set(ce->hidden_t, hidden,  0, (size_t) n_hidden_tokens * n_embd * sizeof(float));
    if (ce->hidden_idx_t != nullptr) {
        ggml_backend_tensor_set(ce->hidden_idx_t, hidden_token_idx, 0, (size_t) n_tokens * sizeof(int32_t));
    }
    ggml_backend_tensor_set(ce->ids_t,    ids,     0, (size_t) n_tokens * n_ids  * sizeof(int32_t));
    if (ce->w_t != nullptr) {
        ggml_backend_tensor_set(ce->w_t, weights, 0, (size_t) n_tokens * n_ids * sizeof(float));
    }

    if (ggml_backend_graph_compute(rt.backend, ce->gf) != GGML_STATUS_SUCCESS) {
        err = "graph compute failed";
        return false;
    }

    ggml_backend_tensor_get(ce->result, out, 0,
            (size_t) n_tokens * n_embd * (no_sum ? (size_t) n_ids : 1) * sizeof(float));

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
// smallest thread count within 3% of the global best.
// ---------------------------------------------------------------------------

// env-gated autotune (GGML_EPD_AUTOTUNE=0 to disable), default on
static bool ep_autotune_enabled() {
    static const bool v = []() {
        const char * e = getenv("GGML_EPD_AUTOTUNE");
        return !(e && e[0] != '\0' && strcmp(e, "0") == 0);
    }();
    return v;
}

// Count unique (physical_package_id, core_id) pairs inside this process's CPU
// affinity. A one-NUMA worker must not autotune against cores from the other
// socket merely because they are online.
static int ep_physical_cores() {
    cpu_set_t allowed;
    CPU_ZERO(&allowed);
    const bool have_affinity = sched_getaffinity(0, sizeof(allowed), &allowed) == 0;
    std::set<std::pair<int, int>> cores;
    for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
        if (have_affinity && !CPU_ISSET(cpu, &allowed)) {
            continue;
        }
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
    if (have_affinity) {
        const int n = CPU_COUNT(&allowed);
        if (n > 0) {
            return n;
        }
    }
    const long n = sysconf(_SC_NPROCESSORS_ONLN);
    return n > 0 ? (int) n : 8;
}

static double ep_median_ms(std::vector<double> & v) {
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

static int ep_autotune_rows_override() {
    static const int value = []() {
        const char * e = getenv("GGML_EPD_AUTOTUNE_ROWS");
        if (e == nullptr || e[0] == '\0') {
            return 0;
        }
        char * end = nullptr;
        const long parsed = strtol(e, &end, 10);
        if (end == e || *end != '\0' || parsed < 1 || parsed > (1 << 20)) {
            LOG("llama-epd: invalid GGML_EPD_AUTOTUNE_ROWS='%s', using ownership estimate\n", e);
            return 0;
        }
        return (int) parsed;
    }();
    return value;
}

// returns the tuned thread count, or -1 on failure (caller keeps its default)
static int ep_autotune_threads(ep_compute_runtime & rt, const ep_model & m) {
    const int n_phys = ep_physical_cores();

    // candidate ladder + physical core count, clamped to [1, n_phys]
    std::set<int> uniq;
    for (int c : {16, 24, 32, 36, n_phys}) {
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

    // Router top-k from GGUF metadata. The worker sees only the fraction held
    // by this NUMA endpoint, and modern SCHED sends those assignments as a
    // compact REQ2 graph (n_ids=1), not as one full top-k classic request.
    int top_k = 6;
    {
        const std::string key = m.arch + ".expert_used_count";
        const int64_t id = gguf_find_key(m.shards[0]->gguf, key.c_str());
        if (id >= 0) {
            top_k = (int) gguf_get_val_u32(m.shards[0]->gguf, id);
        }
    }

    struct autotune_probe {
        const ep_layer * layer = nullptr;
        int rows = 0;
        std::vector<int32_t> ids;
        std::vector<int32_t> token_idx;
        std::vector<float> weights;
        std::vector<float> hidden;
        std::vector<float> out;
    };
    std::vector<autotune_probe> probes;
    probes.reserve(layers.size());
    const int rows_override = ep_autotune_rows_override();
    for (const ep_layer * L : layers) {
        if (L->n_expert > (1 << 30) || L->n_expert_full > (1 << 30)) {
            LOG("llama-epd: autotune: expert count is too large\n");
            return -1;
        }
        autotune_probe probe;
        probe.layer = L;
        probe.rows = llama_ep_autotune_compact_rows(
            top_k, (int) L->n_expert, (int) L->n_expert_full, rows_override);
        if (probe.rows < 1) {
            LOG("llama-epd: autotune: invalid ownership %lld/%lld with top-k %d\n",
                (long long) L->n_expert, (long long) L->n_expert_full, top_k);
            return -1;
        }
        probe.ids.resize((size_t) probe.rows);
        probe.token_idx.assign(probe.rows > 1 ? (size_t) probe.rows : 0, 0);
        probe.weights.assign((size_t) probe.rows, 1.0f / top_k);
        probe.hidden.assign((size_t) L->n_embd, 1e-3f);
        probe.out.resize((size_t) probe.rows * L->n_embd);
        for (int row = 0; row < probe.rows; ++row) {
            probe.ids[(size_t) row] = (int) (((int64_t) row * L->n_expert) / probe.rows);
        }
        probes.push_back(std::move(probe));
    }

    // A single forward candidate sweep was only a few tens of milliseconds and
    // proved sensitive to boost/cache drift. Repeat in alternating order so
    // every candidate samples both the early and late thermal state.
    const int passes = 4;
    const int warmup = 1;
    const int iters  = 7;

    LOG("llama-epd: autotune: %d physical cores, %zu probe layers, top-k=%d, compact_rows=%d%s, candidates:",
        n_phys, probes.size(), top_k, probes[0].rows, rows_override > 0 ? " (override)" : "");
    for (int c : cands) {
        LOG(" %d", c);
    }
    LOG("\n");

    std::vector<std::vector<double>> samples(cands.size());
    for (auto & values : samples) {
        values.reserve((size_t) passes * iters);
    }
    const int64_t t_all = ggml_time_us();
    for (int pass = 0; pass < passes; ++pass) {
        for (size_t pos = 0; pos < cands.size(); ++pos) {
            const size_t ci = pass % 2 == 0 ? pos : cands.size() - 1 - pos;
            ggml_backend_cpu_set_n_threads(rt.backend, cands[ci]); // same path as serving
            for (int it = 0; it < warmup + iters; ++it) {
                const int64_t t0 = ggml_time_us();
                for (autotune_probe & probe : probes) {
                    std::string err;
                    if (!ep_moe_ffn(rt, *probe.layer,
                            probe.rows, 1, probe.ids.data(), probe.weights.data(),
                            1, probe.token_idx.empty() ? nullptr : probe.token_idx.data(),
                            probe.hidden.data(), probe.out.data(), true, true, err)) {
                        LOG("llama-epd: autotune: compute failed: %s\n", err.c_str());
                        return -1;
                    }
                }
                if (it >= warmup) {
                    samples[ci].push_back((ggml_time_us() - t0) / 1000.0);
                }
            }
        }
    }

    std::vector<double> med(cands.size());
    for (size_t ci = 0; ci < cands.size(); ++ci) {
        med[ci] = ep_median_ms(samples[ci]);
        LOG("llama-epd: autotune: threads=%3d  %.3f ms/iter (%zu layers)\n",
            cands[ci], med[ci], layers.size());
    }

    const int selected = llama_ep_autotune_select(
            cands.data(), med.data(), (int) cands.size(), 0.03);
    if (selected == 0) {
        LOG("llama-epd: autotune: invalid timing result\n");
        return -1;
    }
    const auto best = std::find(cands.begin(), cands.end(), selected);
    LOG("llama-epd: autotune: selected threads=%d (%.3f ms/iter, probed in %.1f s)\n",
        selected, med[(size_t) (best - cands.begin())], (ggml_time_us() - t_all) / 1e6);
    return selected;
}

// ---------------------------------------------------------------------------
// NUMA placement policy
// (GGML_EPD_NUMA=off|local|interleave|weighted, default auto-local)
//
// The expert FFN is memory-bandwidth bound, so which NUMA nodes the weight
// pages land on matters. When process CPU affinity covers exactly one NUMA
// node, the default binds allocations to that node. This matches the supported
// one-worker-per-NUMA topology and prevents silent remote-page spill. `off`
// restores first-touch behavior. `local` requests the same single-affinity-node
// bind explicitly. interleave applies MPOL_INTERLEAVE over all online nodes;
// weighted applies MPOL_WEIGHTED_INTERLEAVE (kernel >= 6.9)
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

static std::vector<int> ep_numa_affinity_nodes(const std::vector<int> & online_nodes) {
    cpu_set_t allowed;
    CPU_ZERO(&allowed);
    if (sched_getaffinity(0, sizeof(allowed), &allowed) != 0) {
        return {};
    }
    std::vector<int> result;
    for (int node : online_nodes) {
        const std::vector<int> cpus = ep_numa_node_cpus(node);
        for (int cpu : cpus) {
            if (cpu >= 0 && cpu < CPU_SETSIZE && CPU_ISSET(cpu, &allowed)) {
                result.push_back(node);
                break;
            }
        }
    }
    return result;
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
    for (auto & t : ths) {
        t.join(); // threads store their byte counts after seeing halt; join before reading
    }
    uint64_t total = 0;
    for (auto & b : bytes) {
        total += b.load(std::memory_order_relaxed);
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
    const bool explicit_policy = e != nullptr && e[0] != '\0';
    if (explicit_policy && (strcmp(e, "off") == 0 || strcmp(e, "0") == 0)) {
        return;
    }
    const std::vector<int> online_nodes = ep_numa_online_nodes();
    const std::vector<int> affinity_nodes = ep_numa_affinity_nodes(online_nodes);

    int mode = 0;
    std::vector<int> nodes;
    if (!explicit_policy) {
        if (affinity_nodes.size() != 1) {
            return; // unrestricted/multi-node process: preserve first-touch default
        }
        mode = MPOL_BIND;
        nodes = affinity_nodes;
        LOG("llama-epd: numa: single-node CPU affinity detected; auto-binding weight pages to node %d\n", nodes[0]);
    } else if (strcmp(e, "local") == 0) {
        if (affinity_nodes.size() != 1) {
            LOG("llama-epd: numa: local requires CPU affinity to exactly one NUMA node (found %zu); policy not applied\n",
                    affinity_nodes.size());
            return;
        }
        mode = MPOL_BIND;
        nodes = affinity_nodes;
    } else if (strcmp(e, "interleave") == 0) {
        mode = MPOL_INTERLEAVE;
        nodes = online_nodes;
    } else if (strcmp(e, "weighted") == 0) {
        mode = MPOL_WEIGHTED_INTERLEAVE;
        nodes = online_nodes;
    } else {
        LOG("llama-epd: numa: unknown GGML_EPD_NUMA='%s' (want off|local|interleave|weighted), ignoring\n", e);
        return;
    }

    if (mode != MPOL_BIND && nodes.size() < 2) {
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
    const char * policy_name = mode == MPOL_BIND ? "bind" :
        (mode == MPOL_WEIGHTED_INTERLEAVE ? "weighted-interleave" : "interleave");
    LOG("llama-epd: numa: policy %s over nodes", policy_name);
    for (size_t i = 0; i < nodes.size(); ++i) {
        LOG(" %d%s", nodes[i], mode == MPOL_WEIGHTED_INTERLEAVE ?
            (std::string("(w=") + std::to_string(weights[i]) + ")").c_str() : "");
    }
    LOG("\n");
}

// ---------------------------------------------------------------------------
// server
// ---------------------------------------------------------------------------

// The runtime mutex serializes graph alloc + compute across concurrently served connections:
// the SCHED master holds a persistent connection per endpoint, and a second
// connection may arrive for the classic REQ fallback (PP/warmup batches) —
// the accept loop serves every connection on its own thread, but the shared
// backend/gallocr may only run one graph at a time
static bool ep_send_err(llama_ep_transport * t, int32_t code, const std::string & msg) {
    std::vector<uint8_t> payload(sizeof(int32_t) + msg.size());
    memcpy(payload.data(), &code, sizeof(code));
    memcpy(payload.data() + sizeof(code), msg.data(), msg.size());
    return llama_ep_send_frame(t, LLAMA_EP_MSG_ERR, payload.data(), payload.size());
}

static bool ep_handle_req(
        llama_ep_transport * t,
        ep_compute_runtime & rt,
        const ep_model     & m,
        const ep_config    & cfg,
        const uint8_t      * payload,
        size_t               payload_len) {

    (void) cfg;

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

    std::vector<int32_t> local_ids(n_sel);
    for (size_t i = 0; i < n_sel; ++i) {
        local_ids[i] = m.expert_map.local(ids[i]);
        if (local_ids[i] < 0) {
            return ep_send_err(t, LLAMA_EP_ERR_BAD_EXPERT,
                "expert " + std::to_string(ids[i]) + " is not owned by this worker");
        }
    }

    std::vector<float> out((size_t) n_tokens * L.n_embd);
    std::string err;
    const int64_t t0 = ep_debug_enabled() ? ggml_time_us() : 0;
    {
        std::lock_guard<std::mutex> lock(rt.compute_mutex);
        if (!ep_moe_ffn(rt, L, (int) n_tokens, (int) n_ids, local_ids.data(), weights,
                (int) n_tokens, nullptr, hidden, out.data(), false, true, err)) {
            return ep_send_err(t, LLAMA_EP_ERR_COMPUTE, err);
        }
        ep_op_timing_maybe_dump();
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

static uint64_t ep_model_precision_schema_id(const ep_model & m) {
    uint64_t hash = UINT64_C(1469598103934665603);
    hash = llama_ep_fnv1a64_update(hash, m.arch.data(), m.arch.size());
    hash = llama_ep_fnv1a64_update(hash, &m.n_layer, sizeof(m.n_layer));
    const char * fp16_env = getenv("GGML_CPU_FP16_INTERMEDIATE");
    const uint64_t execution_modes[5] = {
        ep_repack_enabled(),
        ep_fuse_gate_up_enabled(),
        ep_fuse_clamp_swiglu_enabled(),
        (uint64_t) ep_shared_q8_min_tokens(),
        fp16_env != nullptr && atoi(fp16_env) != 0,
    };
    hash = llama_ep_fnv1a64_update(hash, execution_modes, sizeof(execution_modes));
    for (const auto & item : m.layers) {
        const ep_layer & layer = item.second;
        const int32_t types[3] = {
            (int32_t) (layer.gate_up ? layer.gate_up->type : layer.gate->type),
            (int32_t) (layer.gate_up ? layer.gate_up->type : layer.up->type),
            (int32_t) layer.down->type,
        };
        const int64_t shape[4] = {layer.il, layer.n_embd, layer.n_ff, layer.n_expert_full};
        const uint32_t fused_gate_up = layer.gate_up != nullptr;
        hash = llama_ep_fnv1a64_update(hash, shape, sizeof(shape));
        hash = llama_ep_fnv1a64_update(hash, types, sizeof(types));
        hash = llama_ep_fnv1a64_update(hash, &fused_gate_up, sizeof(fused_gate_up));
        hash = llama_ep_fnv1a64_update(hash, &layer.clamp, sizeof(layer.clamp));
    }
    return hash == 0 ? UINT64_C(1) : hash;
}

static bool ep_model_has_cpu_repack_precision_contract(const ep_model & m) {
    if (!ep_repack_enabled() || m.layers.empty()) {
        return false;
    }
    for (const auto & item : m.layers) {
        const ep_layer & layer = item.second;
        if (ep_repack_activation_type(layer.gate_up ? layer.gate_up : layer.gate) == GGML_TYPE_COUNT ||
                (!layer.gate_up && ep_repack_activation_type(layer.up) == GGML_TYPE_COUNT) ||
                ep_repack_activation_type(layer.down) == GGML_TYPE_COUNT) {
            return false;
        }
    }
    return true;
}

static uint64_t ep_data_epoch_id() {
    const char * epoch = getenv("GGML_EP_DATA_EPOCH");
    if (epoch == nullptr || epoch[0] == '\0') {
        return 0;
    }
    uint64_t hash = llama_ep_fnv1a64_update(
        UINT64_C(1469598103934665603), epoch, strlen(epoch));
    return hash == 0 ? UINT64_C(1) : hash;
}

// CAP handshake (protocol v2): answer with this worker's capabilities and owned
// ranges. Precision fields are appended only when requested by the master, so
// old flags=0 masters retain their byte-identical CAP reply.
static bool ep_handle_cap(
        llama_ep_transport * t,
        const ep_model     & m,
        const ep_config    & cfg,
        const uint8_t      * payload,
        size_t               payload_len) {
    if (payload == nullptr || payload_len != sizeof(llama_ep_cap_master)) {
        return ep_send_err(t, LLAMA_EP_ERR_BAD_SHAPE, "CAP: invalid master payload");
    }
    llama_ep_cap_master master = {};
    memcpy(&master, payload, sizeof(master));
    const bool want_precision = (master.flags & LLAMA_EP_CAP_MASTER_WANT_PRECISION) != 0 &&
        ep_model_has_cpu_repack_precision_contract(m);

    llama_ep_cap_worker cap;
    cap.proto_ver = LLAMA_EP_PROTO_VER;
    cap.caps      = LLAMA_EP_CAP_REQ2 | LLAMA_EP_CAP_REQ3 | LLAMA_EP_CAP_REQ4;
    cap.layer_first = m.layers.empty() ? cfg.layer_first : m.layers.begin()->first;
    cap.layer_last  = m.layers.empty() ? cfg.layer_first : m.layers.rbegin()->first;
    cap.expert_first = m.expert_map.contiguous ? m.expert_map.first : 0;
    cap.expert_last  = m.expert_map.contiguous ? m.expert_map.last  : m.expert_map.n_expert;
    cap.kernel_id = llama_ep_kernel_id();
    llama_ep_precision_contract precision = {};
    if (want_precision) {
        cap.caps |= LLAMA_EP_CAP_PRECISION_CONTRACT;
        precision = llama_ep_make_cpu_repack_precision_contract(
            cap.kernel_id, ep_model_precision_schema_id(m), ep_data_epoch_id());
    }

    if (m.expert_map.contiguous && !want_precision) {
        return llama_ep_send_frame(t, LLAMA_EP_MSG_CAP, &cap, sizeof(cap));
    }
    std::vector<uint8_t> bitmap;
    if (!m.expert_map.contiguous) {
        cap.caps |= LLAMA_EP_CAP_EXPERT_BITMAP;
        bitmap = m.expert_map.bitmap();
    }
    if (!want_precision) {
        const void * parts[2] = {&cap, bitmap.data()};
        const size_t lens[2]  = {sizeof(cap), bitmap.size()};
        return llama_ep_send_framev(t, LLAMA_EP_MSG_CAP, parts, lens, 2);
    }
    const void * parts[3] = {&cap, &precision, bitmap.data()};
    const size_t lens[3]  = {sizeof(cap), want_precision ? sizeof(precision) : 0, bitmap.size()};
    const size_t n_parts = !bitmap.empty() ? 3 : 2;
    return llama_ep_send_framev(t, LLAMA_EP_MSG_CAP, parts, lens, n_parts);
}

// REQ2 (SCHEDULER-DESIGN §4.4): ragged per-slot dispatch. each assignment is a
// (token, slot, expert, weight) tuple; the reply carries one weighted expert
// vector per assignment (NOT summed) in REQ2 order, so the master can merge
// them in ascending global slot order bit-identically to the local baseline.
//
// Shared compute path for REQ2/RESP2, REQ3/RESP3, and REQ4/RESP4. The payload
// arrays are already validated by the caller. REQ4 returns raw expert vectors;
// the master applies weights during its existing ordered merge.
static bool ep_handle_req23(
        llama_ep_transport * t,
        ep_compute_runtime & rt,
        const ep_model     & m,
        const ep_config    & cfg,
        int32_t              layer,
        int32_t              n_tokens,
        int32_t              n_sel,
        int32_t              n_embd_hdr,
        const int32_t      * token_idx,
        const int32_t      * slot_idx,
        const int32_t      * expert_id,
        const float        * weights,
        const float        * hidden,
        uint64_t             req_id,
        bool                 unweighted) {

    (void) cfg;

    auto it = m.layers.find(layer);
    if (it == m.layers.end()) {
        return ep_send_err(t, LLAMA_EP_ERR_BAD_LAYER, "layer " + std::to_string(layer) + " not owned by this worker");
    }
    const ep_layer & L = it->second;

    if (n_embd_hdr != (int32_t) L.n_embd) {
        return ep_send_err(t, LLAMA_EP_ERR_BAD_SHAPE, "n_embd mismatch");
    }
    if (n_tokens < 1 || n_tokens > 65536 || n_sel < 0 || n_sel > (int64_t) 1 << 22) {
        return ep_send_err(t, LLAMA_EP_ERR_BAD_SHAPE, "bad n_tokens/n_sel");
    }

    // A connection is handled by one persistent session thread.  Retaining
    // these request buffers avoids allocating and zero-filling them for every
    // MoE layer while keeping concurrent sessions isolated.
    thread_local std::vector<int32_t> local_expert;
    thread_local std::vector<float> out;
    thread_local std::vector<float> gathered_hidden;
    local_expert.resize((size_t) n_sel);
    for (size_t i = 0; i < (size_t) n_sel; ++i) {
        local_expert[i] = m.expert_map.local(expert_id[i]);
        if (local_expert[i] < 0) {
            return ep_send_err(t, LLAMA_EP_ERR_BAD_EXPERT,
                "expert " + std::to_string(expert_id[i]) + " is not owned by this worker");
        }
        if (token_idx[i] < 0 || token_idx[i] >= n_tokens) {
            return ep_send_err(t, LLAMA_EP_ERR_BAD_SHAPE, "token_idx out of range");
        }
    }
    (void) slot_idx; // the worker answers in request order; slot order is the master's concern

    if (n_sel == 0) {
        if (req_id == 0) {
            llama_ep_resp2_header rhdr = {n_tokens, 0, n_embd_hdr};
            return llama_ep_send_frame(t,
                    unweighted ? LLAMA_EP_MSG_RESP4 : LLAMA_EP_MSG_RESP2,
                    &rhdr, sizeof(rhdr));
        }
        llama_ep_resp3_header rhdr = {req_id, 0, n_embd_hdr};
        return llama_ep_send_frame(t, LLAMA_EP_MSG_RESP3, &rhdr, sizeof(rhdr));
    }

    // One graph row per real assignment, preserving request order. Feed the
    // original hidden matrix plus token_idx to a graph-side get_rows instead
    // of expanding hidden in C++ and then copying that expanded matrix into
    // the backend a second time. This removes one n_sel*n_embd host copy.
    const bool identity_hidden = n_sel == n_tokens && [&]() {
        for (int32_t i = 0; i < n_sel; ++i) {
            if (token_idx[i] != i) {
                return false;
            }
        }
        return true;
    }();

    const bool cpp_gather = ep_cpp_gather_enabled() && n_tokens <= 8 && n_sel <= 64 && !identity_hidden;
    if (cpp_gather) {
        gathered_hidden.resize((size_t) n_sel * L.n_embd);
        for (int32_t i = 0; i < n_sel; ++i) {
            memcpy(gathered_hidden.data() + (size_t) i * L.n_embd,
                   hidden + (size_t) token_idx[i] * L.n_embd,
                   (size_t) L.n_embd * sizeof(float));
        }
    }

    out.resize((size_t) n_sel * L.n_embd);
    std::string err;
    const int64_t t0 = ep_debug_enabled() ? ggml_time_us() : 0;
    {
        std::lock_guard<std::mutex> lock(rt.compute_mutex);
        if (!ep_moe_ffn(rt, L, n_sel, 1, local_expert.data(), weights,
                cpp_gather ? n_sel : n_tokens,
                cpp_gather || identity_hidden ? nullptr : token_idx,
                cpp_gather ? gathered_hidden.data() : hidden,
                out.data(), true, !unweighted, err)) {
            return ep_send_err(t, LLAMA_EP_ERR_COMPUTE, err);
        }
        ep_op_timing_maybe_dump();
    }
    if (ep_debug_enabled()) {
        LOG("llama-epd: [ep-debug] layer %d n_tokens=%d compact_rows=%d compute %.3f ms\n",
            layer, (int) n_tokens, (int) n_sel, (ggml_time_us() - t0) / 1000.0);
    }

    if (req_id == 0) {
        llama_ep_resp2_header rhdr = {n_tokens, n_sel, n_embd_hdr};
        const void * parts[2] = {&rhdr, out.data()};
        const size_t lens[2]  = {sizeof(rhdr), out.size() * sizeof(float)};
        return llama_ep_send_framev(t,
                unweighted ? LLAMA_EP_MSG_RESP4 : LLAMA_EP_MSG_RESP2,
                parts, lens, 2);
    }
    llama_ep_resp3_header rhdr = {req_id, n_sel, n_embd_hdr};
    const void * parts[2] = {&rhdr, out.data()};
    const size_t lens[2]  = {sizeof(rhdr), out.size() * sizeof(float)};
    return llama_ep_send_framev(t, LLAMA_EP_MSG_RESP3, parts, lens, 2);
}

static bool ep_handle_req2(
        llama_ep_transport * t,
        ep_compute_runtime & rt,
        const ep_model     & m,
        const ep_config    & cfg,
        const uint8_t      * payload,
        size_t               payload_len) {

    llama_ep_ragged_request_view request;
    std::string err;
    if (!llama_ep_parse_req2(payload, payload_len, request, err)) {
        return ep_send_err(t, LLAMA_EP_ERR_BAD_SHAPE, "REQ2: " + err);
    }

    return ep_handle_req23(t, rt, m, cfg,
            request.layer, request.n_tokens, request.n_sel, request.n_embd,
            request.token_idx, request.slot_idx, request.expert_id,
            request.weights, request.hidden, 0, false);
}

// REQ4 is the explicit, capability-gated sync request for unweighted expert
// vectors. Its payload layout is intentionally identical to REQ2.
static bool ep_handle_req4(
        llama_ep_transport * t,
        ep_compute_runtime & rt,
        const ep_model     & m,
        const ep_config    & cfg,
        const uint8_t      * payload,
        size_t               payload_len) {
    llama_ep_ragged_request_view request;
    std::string err;
    if (!llama_ep_parse_req2(payload, payload_len, request, err)) {
        return ep_send_err(t, LLAMA_EP_ERR_BAD_SHAPE, "REQ4: " + err);
    }
    return ep_handle_req23(t, rt, m, cfg,
            request.layer, request.n_tokens, request.n_sel, request.n_embd,
            request.token_idx, request.slot_idx, request.expert_id,
            request.weights, request.hidden, 0, true);
}

// REQ3 (protocol v3): same ragged dispatch as REQ2 with a master-assigned
// req_id echoed in the RESP3, so the master can keep several requests in
// flight and consume completions out of order (cross-slot pipeline scheduler)
static bool ep_handle_req3(
        llama_ep_transport * t,
        ep_compute_runtime & rt,
        const ep_model     & m,
        const ep_config    & cfg,
        const uint8_t      * payload,
        size_t               payload_len) {

    llama_ep_ragged_request_view request;
    std::string err;
    if (!llama_ep_parse_req3(payload, payload_len, request, err)) {
        return ep_send_err(t, LLAMA_EP_ERR_BAD_SHAPE, "REQ3: " + err);
    }

    return ep_handle_req23(t, rt, m, cfg,
            request.layer, request.n_tokens, request.n_sel, request.n_embd,
            request.token_idx, request.slot_idx, request.expert_id,
            request.weights, request.hidden, request.req_id, false);
}

// serve frames on one connection until EOF
static void ep_serve_connection(
        llama_ep_transport * t,
        ep_compute_runtime & rt,
        const ep_model     & m,
        const ep_config    & cfg) {
    std::vector<uint8_t> payload;
    for (;;) {
        uint32_t type = 0;
        if (!llama_ep_recv_frame(t, type, payload)) {
            break;
        }
        if (type == LLAMA_EP_MSG_CAP) {
            if (!ep_handle_cap(t, m, cfg, payload.data(), payload.size())) {
                LOG("llama-epd: failed to answer CAP\n");
                break;
            }
            continue;
        }
        if (type == LLAMA_EP_MSG_REQ2) {
            if (!ep_handle_req2(t, rt, m, cfg, payload.data(), payload.size())) {
                LOG("llama-epd: failed to handle REQ2\n");
                break;
            }
            continue;
        }
        if (type == LLAMA_EP_MSG_REQ3) {
            if (!ep_handle_req3(t, rt, m, cfg, payload.data(), payload.size())) {
                LOG("llama-epd: failed to handle REQ3\n");
                break;
            }
            continue;
        }
        if (type == LLAMA_EP_MSG_REQ4) {
            if (!ep_handle_req4(t, rt, m, cfg, payload.data(), payload.size())) {
                LOG("llama-epd: failed to handle REQ4\n");
                break;
            }
            continue;
        }
        if (type != LLAMA_EP_MSG_REQ) {
            ep_send_err(t, LLAMA_EP_ERR_GENERIC, "expected REQ");
            continue;
        }
        if (!ep_handle_req(t, rt, m, cfg, payload.data(), payload.size())) {
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

// REQ2 variant: ragged per-slot dispatch, per-assignment weighted vectors back
static bool ep_client_moe_ffn2(
        llama_ep_transport * t,
        int                  layer,
        int                  n_tokens,
        int                  n_sel,
        int                  n_embd,
        const int32_t      * token_idx,
        const int32_t      * slot_idx,
        const int32_t      * expert_id,
        const float        * weights,
        const float        * hidden,
        float              * out, // [n_sel*n_embd]
        std::string        & err) {

    llama_ep_req2_header hdr = {layer, n_tokens, n_sel, n_embd};

    const void * parts[6] = {&hdr, token_idx, slot_idx, expert_id, weights, hidden};
    const size_t lens[6]  = {
        sizeof(hdr),
        (size_t) n_sel * sizeof(int32_t),
        (size_t) n_sel * sizeof(int32_t),
        (size_t) n_sel * sizeof(int32_t),
        (size_t) n_sel * sizeof(float),
        (size_t) n_tokens * n_embd * sizeof(float),
    };
    if (!llama_ep_send_framev(t, LLAMA_EP_MSG_REQ2, parts, lens, 6)) {
        err = "send REQ2 failed";
        return false;
    }

    std::vector<uint8_t> payload;
    uint32_t type = 0;
    if (!llama_ep_recv_frame(t, type, payload)) {
        err = "recv RESP2 failed";
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
    if (type != LLAMA_EP_MSG_RESP2 || payload.size() != sizeof(llama_ep_resp2_header) + (size_t) n_sel * n_embd * sizeof(float)) {
        err = "bad RESP2";
        return false;
    }
    memcpy(out, payload.data() + sizeof(llama_ep_resp2_header), (size_t) n_sel * n_embd * sizeof(float));
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

    ep_compute_runtime rt;
    rt.backend = ggml_backend_cpu_init();
    if (!rt.backend) {
        LOG("selftest: failed to init CPU backend\n");
        return 1;
    }
    ggml_backend_cpu_set_n_threads(rt.backend, cfg.n_threads);
    rt.gallocr = ggml_gallocr_new(ggml_backend_cpu_buffer_type());
    if (!rt.gallocr) {
        LOG("selftest: failed to init graph allocator\n");
        return 1;
    }
    struct ggml_threadpool_params selftest_tpp = ggml_threadpool_params_default(cfg.n_threads);
    selftest_tpp.poll = ep_poll_level();
    rt.threadpool = ggml_threadpool_new(&selftest_tpp);
    if (!rt.threadpool) {
        LOG("selftest: failed to init persistent threadpool\n");
        return 1;
    }
    ggml_backend_cpu_set_threadpool(rt.backend, rt.threadpool);

    // fixed-seed inputs
    ep_rng rng(0xC0FFEE);
    std::vector<int32_t> ids((size_t) n_tokens * n_ids);
    std::vector<int32_t> local_ids((size_t) n_tokens * n_ids);
    std::vector<float>   weights((size_t) n_tokens * n_ids);
    std::vector<float>   hidden((size_t) n_tokens * n_embd);

    for (int t = 0; t < n_tokens; ++t) {
        // sample distinct global experts from this worker's ownership map
        std::vector<int32_t> pool = m.expert_map.local_to_global;
        for (int k = 0; k < n_ids; ++k) {
            int j = k + (int) rng.next_below((uint32_t) (pool.size() - k));
            std::swap(pool[k], pool[j]);
            ids[(size_t) t * n_ids + k] = pool[k];
            local_ids[(size_t) t * n_ids + k] = m.expert_map.local(pool[k]);
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
        if (!ep_moe_ffn(rt, L, n_tokens, n_ids, local_ids.data(), weights.data(),
                n_tokens, nullptr, hidden.data(), out_a.data(), false, true, err)) {
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
            ep_serve_connection(&conn, rt, m, cfg);
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

    // (c) REQ2 loopback: CAP handshake + ragged per-slot dispatch. derive a
    // ragged assignment set from the same inputs (token t keeps its first
    // k_t slots, k_t in [1, n_ids]) and verify that the per-assignment
    // vectors summed in ascending slot order reproduce the summed local
    // compute bit-for-bit (max_abs_diff must be exactly 0)
    double req2_max_abs = 0.0;
    bool req2_ok = true;
    std::string req2_err;
    {
        // build the ragged assignment list, (token, slot) ascending
        std::vector<int32_t> a_tok, a_slot, a_exp;
        std::vector<float>   a_w;
        std::vector<int>     k_per((size_t) n_tokens);
        for (int t = 0; t < n_tokens; ++t) {
            const int k_t = 1 + (int) rng.next_below((uint32_t) n_ids);
            k_per[(size_t) t] = k_t;
            for (int j = 0; j < k_t; ++j) {
                a_tok.push_back(t);
                a_slot.push_back(j);
                a_exp.push_back(ids[(size_t) t * n_ids + j]);
                a_w.push_back(weights[(size_t) t * n_ids + j]);
            }
        }
        const int n_sel = (int) a_tok.size();

        std::string lerr2;
        llama_ep_listener * listener2 = llama_ep_tcp_listen("127.0.0.1", 0, &lerr2);
        if (!listener2) {
            LOG("selftest-req2: listen failed: %s\n", lerr2.c_str());
            return 1;
        }
        const int port2 = llama_ep_tcp_listener_port(listener2);
        std::thread server_thread2([&]() {
            llama_ep_transport conn;
            if (listener2->ops.accept(listener2->ctx, &conn)) {
                ep_serve_connection(&conn, rt, m, cfg);
                conn.ops.close(conn.ctx);
            }
            listener2->ops.close(listener2->ctx);
        });

        std::vector<float> out_c((size_t) n_sel * n_embd);
        llama_ep_transport * cli2 = llama_ep_tcp_connect("127.0.0.1", port2, &req2_err);
        if (cli2) {
            // CAP handshake first, as the master would
            auto request_cap = [&](uint32_t flags, llama_ep_worker_capability & capability,
                                   std::vector<uint8_t> & raw) {
                llama_ep_cap_master mcap = {LLAMA_EP_PROTO_VER, flags};
                uint32_t ctype = 0;
                return llama_ep_send_frame(cli2, LLAMA_EP_MSG_CAP, &mcap, sizeof(mcap)) &&
                    llama_ep_recv_frame(cli2, ctype, raw) && ctype == LLAMA_EP_MSG_CAP &&
                    llama_ep_parse_worker_capability(raw.data(), raw.size(), capability, req2_err);
            };

            // First prove backward compatibility: a flags=0 master must receive
            // exactly the old prefix plus optional ownership bitmap.
            llama_ep_worker_capability legacy_capability;
            std::vector<uint8_t> legacy_payload;
            if (!request_cap(0, legacy_capability, legacy_payload) || legacy_capability.has_precision()) {
                req2_ok = false;
                req2_err = "legacy CAP layout negotiation failed";
            } else {
                const size_t legacy_size = sizeof(llama_ep_cap_worker) +
                    (m.expert_map.contiguous ? 0 : m.expert_map.bitmap().size());
                if (legacy_payload.size() != legacy_size) {
                    req2_ok = false;
                    req2_err = "legacy CAP byte layout changed";
                }
            }

            llama_ep_worker_capability worker_capability;
            std::vector<uint8_t> cpayload;
            if (req2_ok && request_cap(
                    LLAMA_EP_CAP_MASTER_WANT_PRECISION, worker_capability, cpayload)) {
                const llama_ep_cap_worker & wcap = worker_capability.wire;
                if (wcap.proto_ver < LLAMA_EP_PROTO_VER || !(wcap.caps & LLAMA_EP_CAP_REQ2)) {
                    req2_ok = false;
                    req2_err = "worker CAP lacks REQ2";
                }
                if (req2_ok && ep_model_has_cpu_repack_precision_contract(m)) {
                    const llama_ep_precision_contract expected = llama_ep_make_cpu_repack_precision_contract(
                        wcap.kernel_id, ep_model_precision_schema_id(m), ep_data_epoch_id());
                    if (!worker_capability.has_precision() ||
                            !llama_ep_precision_contract_equal(worker_capability.precision, expected)) {
                        req2_ok = false;
                        req2_err = "worker CAP UPE precision contract mismatch";
                    }
                }
                if (req2_ok && m.expert_map.contiguous) {
                    if ((wcap.caps & LLAMA_EP_CAP_EXPERT_BITMAP) ||
                            wcap.expert_first != m.expert_map.first ||
                            wcap.expert_last != m.expert_map.last) {
                        req2_ok = false;
                        req2_err = "worker CAP contiguous ownership mismatch";
                    }
                } else if (req2_ok) {
                    const std::vector<uint8_t> expected = m.expert_map.bitmap();
                    if (!(wcap.caps & LLAMA_EP_CAP_EXPERT_BITMAP) ||
                            wcap.expert_first != 0 || wcap.expert_last != m.expert_map.n_expert ||
                            worker_capability.expert_bitmap != expected) {
                        req2_ok = false;
                        req2_err = "worker CAP sparse ownership bitmap mismatch";
                    }
                }
                if (req2_ok && worker_capability.has_precision()) {
                    LOG("selftest-req2: UPE contract=%016llx schema=%016llx epoch=%016llx\n",
                        (unsigned long long) worker_capability.precision.contract_id,
                        (unsigned long long) worker_capability.precision.model_schema_id,
                        (unsigned long long) worker_capability.precision.data_epoch_id);
                }
            } else if (req2_ok) {
                req2_ok = false;
                req2_err = "CAP handshake failed";
            }
            if (req2_ok) {
                // Send the same request twice on one session. The second call
                // exercises the cached graph/input re-upload path; it must be
                // bit-identical to the cold graph-build call.
                std::vector<float> first((size_t) n_sel * n_embd);
                for (int repeat = 0; repeat < 2 && req2_ok; ++repeat) {
                    req2_ok = ep_client_moe_ffn2(cli2, layer, n_tokens, n_sel, n_embd,
                            a_tok.data(), a_slot.data(), a_exp.data(), a_w.data(),
                            hidden.data(), out_c.data(), req2_err);
                    if (!req2_ok) {
                        break;
                    }
                    if (repeat == 0) {
                        first = out_c;
                    } else if (memcmp(first.data(), out_c.data(), first.size() * sizeof(float)) != 0) {
                        req2_ok = false;
                        req2_err = "cached REQ2 output differs from first compute";
                    }
                }
            }
            cli2->ops.close(cli2->ctx);
            delete cli2;
        }
        server_thread2.join();

        if (req2_ok) {
            // per-token reference: summed local compute on token t's k_t slots
            size_t off = 0;
            for (int t = 0; t < n_tokens && req2_ok; ++t) {
                const int k_t = k_per[(size_t) t];
                std::vector<float> ref((size_t) n_embd);
                std::string rerr;
                if (!ep_moe_ffn(rt, L, 1, k_t,
                        local_ids.data() + (size_t) t * n_ids, weights.data() + (size_t) t * n_ids,
                        1, nullptr, hidden.data() + (size_t) t * n_embd, ref.data(), false, true, rerr)) {
                    req2_ok = false;
                    req2_err = "reference compute failed: " + rerr;
                    break;
                }
                // ascending slot order, left-associated — the merge order of §4.5;
                // float accumulation matches the ggml_add chain bit-for-bit
                for (int e = 0; e < n_embd; ++e) {
                    float acc = 0.0f;
                    for (int j = 0; j < k_t; ++j) {
                        acc += out_c[(off + (size_t) j) * n_embd + e];
                    }
                    req2_max_abs = std::max(req2_max_abs, (double) fabsf(acc - ref[(size_t) e]));
                }
                off += (size_t) k_t;
            }
        }
    }
    if (!req2_ok) {
        LOG("selftest-req2: FAIL: %s\n", req2_err.c_str());
        return 1;
    }
    LOG("selftest-req2: n_sel ragged OK, max_abs_diff=%.6g  %s\n",
        req2_max_abs, req2_max_abs == 0.0 ? "PASS" : "FAIL");
    if (req2_max_abs != 0.0) {
        return 1;
    }

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
        "  --experts A-B          owned contiguous expert range, inclusive (default: all)\n"
        "  --expert-mod R/N       owned sparse experts e where e %% N == R (for 4n+r EP)\n"
        "  --expert-list SPEC     owned sparse ids/ranges, e.g. 0,4,8-11,19\n"
        "  -t, --threads N        compute threads (default: startup autotune; 8 if disabled)\n"
        "  --no-autotune          disable startup thread autotune (also GGML_EPD_AUTOTUNE=0)\n"
        "  --no-mmap              read owned expert weights into anonymous memory at startup\n"
        "                         (slow start, RSS = owned weights resident, no page-in ever)\n"
        "  --selftest             local vs loopback numerical check, then exit\n"
        "  --selftest-layer N     layer for selftest (default: first owned MoE layer)\n"
        "  --selftest-tokens N    tokens for selftest (default 4)\n"
        "\n"
        "env: GGML_EPD_NUMA=off|local|interleave|weighted — weight page NUMA placement;\n"
        "     default auto-binds when CPU affinity selects one node; weighted uses\n"
        "     GGML_EPD_NUMA_WEIGHT=a:b or a startup bandwidth\n"
        "     probe (MPOL_WEIGHTED_INTERLEAVE, kernel >= 6.9)\n"
        "     GGML_EPD_REPACK=0 — keep raw GGUF weight layout (default: convert to\n"
        "     CPU_REPACK interleaved layout for the repack gemv/gemm kernels)\n"
        "     GGML_EPD_AUTOTUNE_ROWS=N — override the estimated compact decode\n"
        "     assignments used by startup thread autotune\n"
        "     GGML_EPD_FUSE_GATE_UP=0 — keep separate gate/up tensors (default:\n"
        "     fuse compatible repacked pairs at load time)\n"
        "     GGML_EPD_SHARED_Q8_MIN_TOKENS=N — share gate/up Q8 activation on\n"
        "     the separate fallback path from N tokens (default 2; 0 disables)\n"
        "     GGML_EPD_POLL=0..100 — persistent compute-thread polling (default 50;\n"
        "     use 0 for workers sharing CPUs with the master)\n"
        "     GGML_EPD_MAX_SESSIONS=N - maximum owned client sessions (default 64)\n"
        "\n",
        argv0);
}

static size_t ep_max_sessions() {
    const char * value = getenv("GGML_EPD_MAX_SESSIONS");
    if (!value || value[0] == '\0') {
        return 64;
    }
    char * end = nullptr;
    const long parsed = strtol(value, &end, 10);
    if (end == value || *end != '\0') {
        LOG("llama-epd: ignoring malformed GGML_EPD_MAX_SESSIONS='%s'\n", value);
        return 64;
    }
    return (size_t) std::max(1L, std::min(parsed, 4096L));
}

static bool parse_range(const char * s, int & a, int & b) {
    int x, y;
    char tail = 0;
    if (sscanf(s, "%d-%d%c", &x, &y, &tail) == 2) {
        a = x; b = y;
        return x <= y;
    }
    if (sscanf(s, "%d%c", &x, &tail) == 1) {
        a = x; b = x;
        return true;
    }
    return false;
}

static bool parse_expert_mod(const char * s, int & r, int & n) {
    char tail = 0;
    return sscanf(s, "%d/%d%c", &r, &n, &tail) == 2 && n > 0 && r >= 0 && r < n;
}

static bool parse_expert_list(const char * s, std::vector<int32_t> & out) {
    out.clear();
    const std::string spec = s;
    size_t off = 0;
    while (off <= spec.size()) {
        const size_t comma = spec.find(',', off);
        const std::string item = spec.substr(off, comma == std::string::npos ? comma : comma - off);
        int first = 0, last = -1;
        if (item.empty() || !parse_range(item.c_str(), first, last) || first < 0) {
            return false;
        }
        if ((int64_t) last - first + 1 > (1 << 20)) {
            return false;
        }
        for (int64_t e = first; e <= last; ++e) {
            out.push_back((int32_t) e);
        }
        if (comma == std::string::npos) {
            break;
        }
        off = comma + 1;
    }
    return !out.empty();
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
    int expert_selector = 0; // 0 default range, 1 --experts, 2 --expert-mod, 3 --expert-list

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
            if (expert_selector != 0) {
                LOG("llama-epd: expert selectors are mutually exclusive\n");
                return 1;
            }
            int e0, e1;
            if (!parse_range(next(a.c_str()), e0, e1)) {
                LOG("llama-epd: bad --experts range\n");
                return 1;
            }
            cfg.expert_first = e0;
            cfg.expert_last = e1 + 1; // CLI is inclusive, config is half-open
            expert_selector = 1;
        } else if (a == "--expert-mod") {
            if (expert_selector != 0 || !parse_expert_mod(next(a.c_str()), cfg.expert_mod_r, cfg.expert_mod_n)) {
                LOG("llama-epd: bad or conflicting --expert-mod (want R/N, 0 <= R < N)\n");
                return 1;
            }
            expert_selector = 2;
        } else if (a == "--expert-list") {
            if (expert_selector != 0 || !parse_expert_list(next(a.c_str()), cfg.expert_list)) {
                LOG("llama-epd: bad or conflicting --expert-list\n");
                return 1;
            }
            cfg.expert_list_set = true;
            expert_selector = 3;
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
    //
    // GGML_NUMA_EP=1 switches to per-node expert row-window placement (NUMA TP
    // across this worker's sockets): ggml NUMA must be initialized before any
    // weight loading so thread affinity and the repack per-node row claims see
    // the real node count. A process-wide GGML_EPD_NUMA policy would fight the
    // per-window mbind placement, so it is skipped unless explicitly set.
    const char * numa_ep_env = getenv("GGML_NUMA_EP");
    const bool numa_ep = numa_ep_env && atoi(numa_ep_env) != 0;
    if (numa_ep) {
        ggml_numa_init(GGML_NUMA_STRATEGY_DISTRIBUTE);
        if (ggml_numa_node_count() < 2) {
            LOG("llama-epd: numa-ep: GGML_NUMA_EP set but only %d NUMA node(s) detected; running without placement\n",
                    ggml_numa_node_count());
        } else if (getenv("GGML_EPD_NUMA") == nullptr) {
            LOG("llama-epd: numa-ep: %d NUMA nodes; per-node expert row windows after repack (GGML_EPD_NUMA process policy skipped)\n",
                    ggml_numa_node_count());
        } else {
            LOG("llama-epd: numa-ep: WARNING: GGML_EPD_NUMA is also set; process mempolicy may conflict with row-window placement\n");
            ep_numa_apply_policy();
        }
    } else {
        ep_numa_apply_policy();
    }

    ggml_backend_load_all(); // no-op for static builds, keeps dl builds working

    ep_model m;
    if (!ep_model_load(m, model_path.c_str(), cfg.layer_first, cfg.layer_last, no_mmap, cfg)) {
        return 1;
    }
    if (m.expert_map.contiguous) {
        LOG("llama-epd: arch=%s n_layer=%d, owning %zu MoE layers, experts [%d, %d)\n",
                m.arch.c_str(), m.n_layer, m.layers.size(), m.expert_map.first, m.expert_map.last);
    } else {
        LOG("llama-epd: arch=%s n_layer=%d, owning %zu MoE layers, sparse experts %zu/%d (global ids %d..%d)\n",
                m.arch.c_str(), m.n_layer, m.layers.size(), m.expert_map.local_to_global.size(),
                m.expert_map.n_expert, m.expert_map.first, m.expert_map.last - 1);
    }

    if (numa_ep) {
        ep_numa_tp_place(m);
    }

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

    ep_compute_runtime rt;
    rt.backend = ggml_backend_cpu_init();
    if (!rt.backend) {
        LOG("llama-epd: failed to init CPU backend\n");
        return 1;
    }
    rt.gallocr = ggml_gallocr_new(ggml_backend_cpu_buffer_type());
    if (!rt.gallocr) {
        LOG("llama-epd: failed to init graph allocator\n");
        return 1;
    }
    if (!threads_set && autotune) {
        const int tuned = ep_autotune_threads(rt, m);
        if (tuned > 0) {
            cfg.n_threads = tuned;
        }
    } else if (!threads_set) {
        LOG("llama-epd: autotune disabled, using %d threads\n", cfg.n_threads);
    }
    ggml_backend_cpu_set_n_threads(rt.backend, cfg.n_threads);

    // persistent threadpool: without it every ggml_graph_compute spawns and joins a
    // disposable pool (measured: ~7 ms of the per-REQ fixed cost at 70 threads).
    // attached after autotune so the ladder still probes with disposable pools
    struct ggml_threadpool_params tpp = ggml_threadpool_params_default(cfg.n_threads);
    tpp.poll = ep_poll_level();
    rt.threadpool = ggml_threadpool_new(&tpp);
    if (rt.threadpool) {
        ggml_backend_cpu_set_threadpool(rt.backend, rt.threadpool);
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
    LOG("llama-epd: listening on port %d (%d threads, poll=%u)%s\n",
        bound_port, cfg.n_threads, tpp.poll, rdma_mode ? " [rdma]" : "");

    llama_ep_session_manager sessions(ep_max_sessions(),
        [&rt, &m, &cfg](llama_ep_transport * conn) {
            ep_serve_connection(conn, rt, m, cfg);
            LOG("llama-epd: client disconnected\n");
        });

    for (;;) {
        llama_ep_transport conn;
        if (!listener->ops.accept(listener->ctx, &conn)) {
            LOG("llama-epd: accept failed\n");
            continue;
        }
        LOG("llama-epd: client connected\n");
        // Sessions own their threads and transports. Compute remains serialized
        // by rt.compute_mutex until the worker microbatch scheduler replaces it.
        if (!sessions.start(conn)) {
            ep_send_err(&conn, LLAMA_EP_ERR_GENERIC, "worker session limit reached");
            conn.ops.close(conn.ctx);
            LOG("llama-epd: rejected client at session limit %zu\n", ep_max_sessions());
        }
    }
    return 0;
}
