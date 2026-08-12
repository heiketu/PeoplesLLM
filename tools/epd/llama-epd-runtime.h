#pragma once

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <unordered_map>

// Mutable CPU execution state owned by one llama-epd worker. This module is
// intentionally independent of model/config types: the current serialized
// executor and a future queued/microbatched executor can share the same clear
// lifetime boundary without reintroducing process-global backend state.

struct ep_graph_cache_entry {
    ggml_context  * ctx          = nullptr;
    ggml_cgraph   * gf           = nullptr;
    ggml_tensor   * hidden_t     = nullptr;
    ggml_tensor   * hidden_idx_t = nullptr;
    ggml_tensor   * ids_t        = nullptr;
    ggml_tensor   * w_t          = nullptr;
    ggml_tensor   * result       = nullptr;
    ggml_gallocr_t gallocr       = nullptr;
    size_t         bytes         = 0;
    uint64_t       last_use      = 0;

    ep_graph_cache_entry() = default;
    ep_graph_cache_entry(const ep_graph_cache_entry &) = delete;
    ep_graph_cache_entry(ep_graph_cache_entry &&) = delete;
    ep_graph_cache_entry & operator=(const ep_graph_cache_entry &) = delete;
    ep_graph_cache_entry & operator=(ep_graph_cache_entry &&) = delete;

    ~ep_graph_cache_entry() {
        if (gallocr != nullptr) {
            ggml_gallocr_free(gallocr);
        }
        if (ctx != nullptr) {
            ggml_free(ctx);
        }
    }
};

struct ep_graph_cache_key {
    const void * layer = nullptr;
    int n_tokens = 0;
    int n_ids = 0;
    int n_hidden_tokens = 0;
    bool indexed_hidden = false;
    bool no_sum = false;
    bool apply_weights = true;

    bool operator==(const ep_graph_cache_key & other) const {
        return layer == other.layer && n_tokens == other.n_tokens &&
               n_ids == other.n_ids && n_hidden_tokens == other.n_hidden_tokens &&
               indexed_hidden == other.indexed_hidden && no_sum == other.no_sum &&
               apply_weights == other.apply_weights;
    }
};

struct ep_graph_cache_key_hash {
    size_t operator()(const ep_graph_cache_key & key) const {
        size_t h = (size_t) key.layer;
        h ^= (size_t) (uint32_t) key.n_tokens + 0x9e3779b9u + (h << 6) + (h >> 2);
        h ^= (size_t) (uint32_t) key.n_ids + 0x9e3779b9u + (h << 6) + (h >> 2);
        h ^= (size_t) (uint32_t) key.n_hidden_tokens + 0x9e3779b9u + (h << 6) + (h >> 2);
        h ^= (size_t) key.indexed_hidden + 0x9e3779b9u + (h << 6) + (h >> 2);
        h ^= (size_t) key.no_sum + 0x9e3779b9u + (h << 6) + (h >> 2);
        h ^= (size_t) key.apply_weights + 0x9e3779b9u + (h << 6) + (h >> 2);
        return h;
    }
};

struct ep_compute_runtime {
    ggml_backend_t    backend    = nullptr;
    ggml_gallocr_t    gallocr    = nullptr;
    ggml_threadpool_t threadpool = nullptr;

    std::mutex compute_mutex;
    std::unordered_map<ep_graph_cache_key, std::unique_ptr<ep_graph_cache_entry>, ep_graph_cache_key_hash>
        graph_cache;
    size_t   graph_cache_bytes = 0;
    uint64_t graph_cache_clock = 0;
    ep_compute_runtime() = default;
    ep_compute_runtime(const ep_compute_runtime &) = delete;
    ep_compute_runtime & operator=(const ep_compute_runtime &) = delete;

    ~ep_compute_runtime() {
        graph_cache.clear();
        if (threadpool != nullptr) {
            if (backend != nullptr) {
                ggml_backend_cpu_set_threadpool(backend, nullptr);
            }
            ggml_threadpool_free(threadpool);
        }
        if (gallocr != nullptr) {
            ggml_gallocr_free(gallocr);
        }
        if (backend != nullptr) {
            ggml_backend_free(backend);
        }
    }
};
