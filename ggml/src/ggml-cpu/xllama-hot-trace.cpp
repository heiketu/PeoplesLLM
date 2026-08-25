#include "xllama-hot-trace.h"

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <mutex>
#include <new>
#include <string>
#include <vector>

namespace xllama {
namespace {

constexpr size_t default_max_bytes = 64u * 1024u * 1024u;
constexpr size_t min_max_bytes     = 1u  * 1024u * 1024u;
constexpr size_t max_max_bytes     = 4ull * 1024u * 1024u * 1024u;
constexpr int    max_layers        = 128;
constexpr int    max_experts       = 1024;

struct trace_record_header {
    uint64_t step;
    int32_t  layer;
    uint32_t n_ids;
};

static_assert(sizeof(trace_record_header) == 16, "unexpected trace header size");

struct trace_state {
    bool                  enabled = false;
    bool                  dirty = false;
    bool                  dumped = false;
    size_t                max_bytes = default_max_bytes;
    std::string           path;
    std::mutex            mutex;
    std::vector<uint8_t>  data;
    std::atomic<uint32_t> active{0};
    uint64_t              next_step = 0;
    uint64_t              records = 0;
    uint64_t              dropped_capacity = 0;
    uint64_t              dropped_invalid = 0;
    uint64_t              concurrent_overlaps = 0;

    trace_state() {
        const char * env = std::getenv("GGML_MOE_HOT_TRACE");
        enabled = env != nullptr && std::atoi(env) != 0;
        if (!enabled) {
            return;
        }

        const char * hot_expert = std::getenv("GGML_HOT_EXPERT");
        if (hot_expert != nullptr && std::atoi(hot_expert) != 0) {
            std::fprintf(stderr,
                "xllama hot trace: GGML_HOT_EXPERT masks routed ids; disable hot-expert while tracing\n");
            enabled = false;
            return;
        }

        const char * path_env = std::getenv("GGML_MOE_HOT_TRACE_PATH");
        path = path_env != nullptr && path_env[0] != '\0' ? path_env : "/tmp/expert-hot-trace.tsv";

        const char * max_mb_env = std::getenv("GGML_MOE_HOT_TRACE_MAX_MB");
        if (max_mb_env != nullptr && max_mb_env[0] != '\0') {
            errno = 0;
            char * end = nullptr;
            const unsigned long long max_mb = std::strtoull(max_mb_env, &end, 10);
            if (errno == 0 && end != max_mb_env && *end == '\0' && max_mb >= 1 && max_mb <= 4096) {
                max_bytes = static_cast<size_t>(max_mb) * 1024u * 1024u;
            }
        }
        max_bytes = std::max(min_max_bytes, std::min(max_bytes, max_max_bytes));

        try {
            data.reserve(max_bytes);
        } catch (const std::bad_alloc &) {
            std::fprintf(stderr, "xllama hot trace: failed to reserve %.1f MiB; trace disabled\n", max_bytes / 1048576.0);
            enabled = false;
        }
    }
};

trace_state & get_state() {
    static trace_state state;
    return state;
}

bool dump_locked(trace_state & state) {
    if (state.dumped && !state.dirty) {
        return true;
    }

    FILE * file = std::fopen(state.path.c_str(), "w");
    if (file == nullptr) {
        return false;
    }

    bool ok = true;
    ok = ok && std::fprintf(file, "# xllama-moe-hot-trace v1\n") >= 0;
    ok = ok && std::fprintf(file, "# columns: step layer expert...\n") >= 0;
    ok = ok && std::fprintf(file, "# step_semantics=capture_ordinal_per_token_row\n") >= 0;

    size_t offset = 0;
    while (ok && offset < state.data.size()) {
        trace_record_header header;
        std::memcpy(&header, state.data.data() + offset, sizeof(header));
        offset += sizeof(header);
        ok = ok && std::fprintf(file, "%llu\t%d", static_cast<unsigned long long>(header.step), header.layer) >= 0;
        for (uint32_t i = 0; ok && i < header.n_ids; ++i) {
            int32_t expert;
            std::memcpy(&expert, state.data.data() + offset, sizeof(expert));
            offset += sizeof(expert);
            ok = ok && std::fprintf(file, "\t%d", expert) >= 0;
        }
        ok = ok && std::fputc('\n', file) != EOF;
    }

    ok = ok && std::fprintf(file, "# records=%llu\n", static_cast<unsigned long long>(state.records)) >= 0;
    ok = ok && std::fprintf(file, "# dropped_capacity=%llu\n", static_cast<unsigned long long>(state.dropped_capacity)) >= 0;
    ok = ok && std::fprintf(file, "# dropped_invalid=%llu\n", static_cast<unsigned long long>(state.dropped_invalid)) >= 0;
    ok = ok && std::fprintf(file, "# concurrent_overlaps=%llu\n", static_cast<unsigned long long>(state.concurrent_overlaps)) >= 0;
    ok = std::fclose(file) == 0 && ok;

    if (ok) {
        state.dirty = false;
        state.dumped = true;
    }
    return ok;
}

void dump_at_exit() {
    if (!moe_hot_trace_flush()) {
        std::fprintf(stderr, "xllama hot trace: failed to write output\n");
    }
}

struct active_guard {
    std::atomic<uint32_t> & active;

    ~active_guard() {
        active.fetch_sub(1, std::memory_order_release);
    }
};

} // namespace

bool moe_hot_trace_enabled() {
    trace_state & state = get_state();
    static const bool registered = [&state]() {
        if (state.enabled && std::atexit(dump_at_exit) != 0) {
            std::fprintf(stderr, "xllama hot trace: failed to register exit flush; trace disabled\n");
            state.enabled = false;
        }
        return true;
    }();
    (void) registered;
    return state.enabled;
}

void moe_hot_trace_record(
        int          layer,
        const void * ids,
        int64_t      n_tokens,
        int          n_ids,
        size_t       token_stride,
        size_t       id_stride,
        int          n_experts) {
    if (!moe_hot_trace_enabled()) {
        return;
    }

    trace_state & state = get_state();
    const uint32_t active_before = state.active.fetch_add(1, std::memory_order_acquire);
    active_guard guard{state.active};
    std::lock_guard<std::mutex> lock(state.mutex);

    if (active_before > 0) {
        ++state.concurrent_overlaps;
        state.dirty = true;
    }

    if (ids == nullptr || layer < 0 || layer >= max_layers || n_tokens <= 0 || n_ids <= 0 ||
            n_ids > max_experts || n_experts <= 0 || n_experts > max_experts) {
        state.dropped_invalid += n_tokens > 0 ? static_cast<uint64_t>(n_tokens) : 1;
        state.dirty = true;
        return;
    }

    if (static_cast<size_t>(n_ids) > (std::numeric_limits<size_t>::max() - sizeof(trace_record_header)) / sizeof(int32_t)) {
        state.dropped_invalid += static_cast<uint64_t>(n_tokens);
        state.dirty = true;
        return;
    }
    const size_t record_bytes = sizeof(trace_record_header) + static_cast<size_t>(n_ids) * sizeof(int32_t);

    for (int64_t token = 0; token < n_tokens; ++token) {
        const char * token_ids = static_cast<const char *>(ids) + static_cast<size_t>(token) * token_stride;
        bool valid = true;
        for (int id = 0; id < n_ids; ++id) {
            int32_t expert;
            std::memcpy(&expert, token_ids + static_cast<size_t>(id) * id_stride, sizeof(expert));
            valid = valid && expert >= 0 && expert < n_experts;
        }
        if (!valid) {
            ++state.dropped_invalid;
            state.dirty = true;
            continue;
        }

        if (record_bytes > state.max_bytes - state.data.size()) {
            state.dropped_capacity += static_cast<uint64_t>(n_tokens - token);
            state.dirty = true;
            break;
        }

        const trace_record_header header{state.next_step++, layer, static_cast<uint32_t>(n_ids)};
        const size_t offset = state.data.size();
        state.data.resize(offset + record_bytes);
        std::memcpy(state.data.data() + offset, &header, sizeof(header));
        char * output_ids = reinterpret_cast<char *>(state.data.data() + offset + sizeof(header));
        for (int id = 0; id < n_ids; ++id) {
            std::memcpy(output_ids + static_cast<size_t>(id) * sizeof(int32_t),
                        token_ids + static_cast<size_t>(id) * id_stride, sizeof(int32_t));
        }
        ++state.records;
        state.dirty = true;
    }
}

bool moe_hot_trace_flush() {
    trace_state & state = get_state();
    if (!state.enabled) {
        return false;
    }
    std::lock_guard<std::mutex> lock(state.mutex);
    return dump_locked(state);
}

} // namespace xllama
