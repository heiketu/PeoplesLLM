#pragma once

#include <cstddef>
#include <functional>

struct ggml_tensor;
struct llama_file;

enum class llama_repack_stream_bake_result {
    unsupported,
    complete,
    cancelled,
};

struct llama_repack_stream_bake_config {
    size_t chunk_bytes = 1024 * 1024;
    size_t progress_interval_bytes = 64 * 1024 * 1024;
    bool validate = false;
};

bool llama_repack_stream_bake_env_enabled();
size_t llama_repack_stream_bake_env_chunk_bytes();

llama_repack_stream_bake_result llama_repack_stream_bake_load(
        llama_file & file,
        size_t file_offset,
        ggml_tensor * tensor,
        const llama_repack_stream_bake_config & config,
        const std::function<bool(size_t, size_t)> & progress);
