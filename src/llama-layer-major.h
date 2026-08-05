#pragma once

#include "ggml-backend.h"
#include "ggml-cpp.h"

#include <cstddef>
#include <cstdint>
#include <vector>

struct llama_layer_major_ubatch_profile {
    int64_t memory_apply_us = 0;
    int64_t graph_prepare_us = 0;
    int64_t set_inputs_us = 0;
    int64_t submit_us = 0;
};

struct llama_layer_major_graph_input {
    ggml_backend_t hc_backend = nullptr;
    const ggml_tensor * hc_tensor = nullptr;
    ggml_backend_t raw_kq_mask_backend = nullptr;
    const ggml_tensor * raw_kq_mask = nullptr;
    llama_layer_major_ubatch_profile * profile = nullptr;
};

ggml_backend_t llama_layer_major_buffer_backend(
        const std::vector<ggml_backend_t> & backends,
        const ggml_tensor * tensor);

class llama_layer_major_hc_state {
public:
    bool init(
            size_t n_tokens,
            size_t hc_dim,
            size_t n_ubatch,
            ggml_backend_sched_t sched,
            const std::vector<ggml_backend_t> & backends);

    bool is_device_resident() const;
    ggml_backend_t device_backend() const;
    bool select_backend(ggml_backend_t backend);

    float * host_tile(size_t token_offset);
    const ggml_tensor * device_tile(size_t tile_index) const;

    bool store_tile(
            ggml_backend_sched_t sched,
            const std::vector<ggml_backend_t> & backends,
            ggml_tensor * src,
            size_t tile_index,
            size_t token_offset);

private:
    struct device_storage {
        ggml_backend_t backend = nullptr;
        ggml_context_ptr ctx;
        ggml_backend_buffer_ptr buf;
        std::vector<ggml_tensor *> tiles;
    };

    size_t hc_dim_ = 0;

    std::vector<float> host_;
    std::vector<device_storage> device_storage_;
    size_t active_storage_ = SIZE_MAX;
    bool migrate_between_backends_ = false;
};
