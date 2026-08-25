#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"
#include "gguf.h"
#include "llama-repack-stream-bake.h"
#include "llama-mmap.h"
#include "llama-model-loader.h"
#include "repack.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#if !defined(_WIN32)
#include <unistd.h>
#endif

namespace {

struct repacked_tensor {
    ggml_context * ctx = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    ggml_tensor * tensor = nullptr;

    repacked_tensor() = default;
    repacked_tensor(const repacked_tensor &) = delete;
    repacked_tensor & operator=(const repacked_tensor &) = delete;
    repacked_tensor(repacked_tensor && other) noexcept
        : ctx(other.ctx), buffer(other.buffer), tensor(other.tensor) {
        other.ctx = nullptr;
        other.buffer = nullptr;
        other.tensor = nullptr;
    }
    repacked_tensor & operator=(repacked_tensor &&) = delete;

    ~repacked_tensor() {
        if (buffer != nullptr) {
            ggml_backend_buffer_free(buffer);
        }
        if (ctx != nullptr) {
            ggml_free(ctx);
        }
    }
};

repacked_tensor make_tensor(int64_t ne0, int64_t nrows, ggml_type type = GGML_TYPE_E4A) {
    repacked_tensor result;
    ggml_init_params params = {1024 * 1024, nullptr, true};
    result.ctx = ggml_init(params);
    if (result.ctx == nullptr) {
        throw std::runtime_error("ggml_init failed");
    }
    result.tensor = ggml_new_tensor_2d(result.ctx, type, ne0, nrows);
    ggml_set_name(result.tensor, "synthetic.e4a");
    result.buffer = ggml_backend_buft_alloc_buffer(
        ggml_backend_cpu_repack_buffer_type(), ggml_nbytes(result.tensor));
    if (result.buffer == nullptr) {
        throw std::runtime_error("CPU_REPACK allocation failed");
    }
    ggml_backend_tensor_alloc(result.buffer, result.tensor, ggml_backend_buffer_get_base(result.buffer));
    return result;
}

bool same(const repacked_tensor & a, const repacked_tensor & b) {
    const size_t size = ggml_nbytes(a.tensor);
    return size == ggml_nbytes(b.tensor) && std::memcmp(a.tensor->data, b.tensor->data, size) == 0;
}

uint64_t fnv1a(const void * data, size_t size) {
    const uint8_t * bytes = static_cast<const uint8_t *>(data);
    uint64_t hash = 1469598103934665603ULL;
    for (size_t i = 0; i < size; ++i) {
        hash ^= bytes[i];
        hash *= 1099511628211ULL;
    }
    return hash;
}

void write_full(int fd, const void * data, size_t size) {
    const uint8_t * src = static_cast<const uint8_t *>(data);
    size_t done = 0;
    while (done < size) {
        const ssize_t written = ::write(fd, src + done, size - done);
        if (written <= 0) {
            throw std::runtime_error("temporary file write failed");
        }
        done += (size_t) written;
    }
}

struct temp_file {
    std::string path;

    explicit temp_file(const std::vector<uint8_t> & data, size_t prefix, bool truncate_last = false) {
        char name[] = "/tmp/xllama-e4a-stream-XXXXXX";
        const int fd = ::mkstemp(name);
        if (fd < 0) {
            throw std::runtime_error("mkstemp failed");
        }
        path = name;
        std::vector<uint8_t> padding(prefix, 0x5a);
        write_full(fd, padding.data(), padding.size());
        const size_t size = truncate_last ? data.size() - 1 : data.size();
        write_full(fd, data.data(), size);
        ::close(fd);
    }

    ~temp_file() {
        if (!path.empty()) {
            std::remove(path.c_str());
        }
    }
};

struct temp_path {
    std::string path;

    temp_path() {
        char name[] = "/tmp/xllama-e4a-gguf-XXXXXX";
        const int fd = ::mkstemp(name);
        if (fd < 0) {
            throw std::runtime_error("mkstemp failed");
        }
        ::close(fd);
        path = name;
    }

    ~temp_path() {
        if (!path.empty()) {
            std::remove(path.c_str());
        }
    }
};

} // namespace

int main(int argc, char ** argv) {
#if defined(_WIN32)
    std::printf("test-repack-stream-bake: SKIPPED on Windows\n");
    return 0;
#else
    ggml_cpu_init();
    if (argc > 1) {
        const bool stream_mode = std::strcmp(argv[1], "--real") == 0;
        const bool default_mode = std::strcmp(argv[1], "--real-default") == 0;
        if ((argc != 5 && argc != 6) || (!stream_mode && !default_mode) || (argc == 6 && !stream_mode)) {
            std::fprintf(stderr, "usage: %s [--real|--real-default model.gguf tensor expected-fnv1a64 [chunk-mib]]\n", argv[0]);
            return 2;
        }
        if ((stream_mode && (::setenv("GGML_STREAM_BAKE", "1", 1) != 0 ||
                ::setenv("GGML_STREAM_BAKE_CHUNK_MIB", argc == 6 ? argv[5] : "1", 1) != 0)) ||
                (default_mode && (::unsetenv("GGML_STREAM_BAKE") != 0 ||
                                  ::unsetenv("GGML_E4A_STREAM_BAKE") != 0))) {
            throw std::runtime_error("setenv failed");
        }

        std::vector<std::string> splits;
        llama_model_loader loader(
            nullptr, nullptr, nullptr, argv[2], splits, nullptr,
            LLAMA_LOAD_MODE_NONE, false, false, false, nullptr, nullptr);
        ggml_tensor * meta = loader.require_tensor_meta(argv[3]);
        ggml_init_params params = {ggml_tensor_overhead() * 2, nullptr, true};
        ggml_context * ctx = ggml_init(params);
        if (ctx == nullptr) {
            throw std::runtime_error("real smoke ggml_init failed");
        }
        ggml_tensor * tensor = ggml_dup_tensor(ctx, meta);
        ggml_set_name(tensor, argv[3]);
        ggml_backend_buffer_t buffer = ggml_backend_buft_alloc_buffer(
            ggml_backend_cpu_repack_buffer_type(), ggml_nbytes(tensor));
        if (buffer == nullptr) {
            ggml_free(ctx);
            throw std::runtime_error("real smoke CPU_REPACK allocation failed");
        }
        ggml_backend_tensor_alloc(buffer, tensor, ggml_backend_buffer_get_base(buffer));
        if (ggml_backend_cpu_repack_chunk_alignment(tensor) == 0) {
            ggml_backend_buffer_free(buffer);
            ggml_free(ctx);
            throw std::runtime_error("real smoke stream chunk loading unavailable");
        }

        loader.init_mappings(false);
        llama_buf_map bufs;
        const bool loaded = loader.load_all_data(ctx, bufs, nullptr, nullptr, nullptr);
        const uint64_t actual = loaded ? fnv1a(tensor->data, ggml_nbytes(tensor)) : 0;
        char * end = nullptr;
        const uint64_t expected = std::strtoull(argv[4], &end, 16);
        const bool hash_ok = end != argv[4] && *end == '\0' && actual == expected;
        std::printf("test-repack-stream-bake: real mode=%s tensor=%s bytes=%zu hash=%016llx expected=%016llx %s\n",
            stream_mode ? "stream" : "default", argv[3], ggml_nbytes(tensor),
            (unsigned long long) actual, (unsigned long long) expected,
            loaded && hash_ok ? "PASS" : "FAIL");
        ggml_backend_buffer_free(buffer);
        ggml_free(ctx);
        return loaded && hash_ok ? 0 : 1;
    }

    constexpr int64_t ne0 = 2048;
    constexpr int64_t nrows = 64;

    repacked_tensor reference = make_tensor(ne0, nrows);
    const size_t alignment = ggml_backend_cpu_repack_chunk_alignment(reference.tensor);
    if (alignment == 0) {
        std::printf("test-repack-stream-bake: SKIPPED, E4A CPU_REPACK unavailable\n");
        return 0;
    }
    const size_t size = ggml_nbytes(reference.tensor);
    std::vector<uint8_t> raw(size);
    for (size_t i = 0; i < raw.size(); ++i) {
        raw[i] = (uint8_t) ((i * 131 + i / 17 + 29) & 0xff);
    }
    ggml_backend_tensor_set(reference.tensor, raw.data(), 0, raw.size());

    repacked_tensor partial = make_tensor(ne0, nrows);
    GGML_ASSERT(ggml_backend_cpu_repack_chunk_alignment(partial.tensor) == alignment);
    bool ok = true;
    size_t offset = 0;
    for (size_t panels : {2UL, 1UL, 1UL}) {
        const size_t chunk = panels * alignment;
        ok &= ggml_backend_cpu_repack_write_chunk(partial.tensor, raw.data() + offset, offset, chunk);
        offset += chunk;
    }
    ok &= offset == size && same(reference, partial);

    std::vector<uint8_t> snapshot(size);
    std::memcpy(snapshot.data(), partial.tensor->data, size);
    ok &= !ggml_backend_cpu_repack_write_chunk(partial.tensor, raw.data(), 1, alignment);
    ok &= std::memcmp(snapshot.data(), partial.tensor->data, size) == 0;

    bool generic_ok = true;
    for (ggml_type type : {GGML_TYPE_Q2_K, GGML_TYPE_Q3_K, GGML_TYPE_Q4_K, GGML_TYPE_IQ2_XXS,
                           GGML_TYPE_IQ2_XS, GGML_TYPE_IQ3_XXS, GGML_TYPE_MXFP4}) {
        repacked_tensor generic_partial = make_tensor(ne0, nrows, type);
        const size_t generic_alignment = ggml_backend_cpu_repack_chunk_alignment(generic_partial.tensor);
        if (generic_alignment == 0) {
            generic_ok &= type == GGML_TYPE_Q2_K;
            continue;
        }
        repacked_tensor generic_reference = make_tensor(ne0, nrows, type);
        const size_t generic_size = ggml_nbytes(generic_reference.tensor);
        std::vector<uint8_t> generic_raw(generic_size);
        for (size_t i = 0; i < generic_size; ++i) {
            generic_raw[i] = (uint8_t) ((i * 193 + i / 29 + 11 * (size_t) type) & 0xff);
        }
        ggml_backend_tensor_set(generic_reference.tensor, generic_raw.data(), 0, generic_size);

        generic_ok &= generic_alignment > 0 && generic_size % generic_alignment == 0;
        for (size_t generic_offset = 0; generic_alignment > 0 && generic_offset < generic_size;
                generic_offset += generic_alignment) {
            generic_ok &= ggml_backend_cpu_repack_write_chunk(
                generic_partial.tensor, generic_raw.data() + generic_offset, generic_offset, generic_alignment);
        }
        generic_ok &= same(generic_reference, generic_partial);
    }
    ok &= generic_ok;

    repacked_tensor wrong_shape = make_tensor(ne0, nrows - 1);
    repacked_tensor wrong_type = make_tensor(ne0, nrows, GGML_TYPE_Q8_0);
    ok &= ggml_backend_cpu_repack_chunk_alignment(wrong_shape.tensor) == 0;
    ok &= ggml_backend_cpu_repack_chunk_alignment(wrong_type.tensor) == 0;

    constexpr size_t prefix = 257;
    temp_file complete_file(raw, prefix);
    llama_file fallback_file_a(complete_file.path.c_str(), "rb", false);
    llama_file fallback_file_b(complete_file.path.c_str(), "rb", false);
    llama_repack_stream_bake_config config;
    config.chunk_bytes = alignment;
    config.progress_interval_bytes = alignment;
    config.validate = true;
    const auto fallback_shape = llama_repack_stream_bake_load(
        fallback_file_a, prefix, wrong_shape.tensor, config, {});
    const auto fallback_type = llama_repack_stream_bake_load(
        fallback_file_b, prefix, wrong_type.tensor, config, {});
    ok &= fallback_shape == llama_repack_stream_bake_result::unsupported;
    ok &= fallback_type == llama_repack_stream_bake_result::unsupported;

    llama_file file(complete_file.path.c_str(), "rb", false);
    repacked_tensor streamed = make_tensor(ne0, nrows);
    size_t last_progress = 0;
    const auto result = llama_repack_stream_bake_load(
        file, prefix, streamed.tensor, config,
        [&](size_t done, size_t total) {
            ok &= done >= last_progress && done <= total;
            last_progress = done;
            return true;
        });
    ok &= result == llama_repack_stream_bake_result::complete;
    ok &= last_progress == size && same(reference, streamed);

    llama_file cancel_file(complete_file.path.c_str(), "rb", false);
    repacked_tensor cancelled = make_tensor(ne0, nrows);
    size_t callbacks = 0;
    const auto cancel_result = llama_repack_stream_bake_load(
        cancel_file, prefix, cancelled.tensor, config,
        [&](size_t, size_t) {
            callbacks++;
            return false;
        });
    ok &= cancel_result == llama_repack_stream_bake_result::cancelled && callbacks == 1;

    bool short_read_threw = false;
    temp_file short_file(raw, prefix, true);
    llama_file truncated(short_file.path.c_str(), "rb", false);
    repacked_tensor short_target = make_tensor(ne0, nrows);
    try {
        (void) llama_repack_stream_bake_load(truncated, prefix, short_target.tensor, config, {});
    } catch (const std::runtime_error &) {
        short_read_threw = true;
    }
    ok &= short_read_threw;

    if (::setenv("GGML_E4A_STREAM_BAKE", "1", 1) != 0 ||
            ::setenv("GGML_E4A_STREAM_BAKE_CHUNK_MIB", "1", 1) != 0) {
        throw std::runtime_error("setenv failed");
    }
    temp_path synthetic_path;
    gguf_context * gguf = gguf_init_empty();
    gguf_set_val_str(gguf, "general.architecture", "llama");
    ggml_init_params meta_params = {ggml_tensor_overhead() * 2, nullptr, true};
    ggml_context * meta_ctx = ggml_init(meta_params);
    ggml_tensor * meta_tensor = ggml_new_tensor_2d(meta_ctx, GGML_TYPE_E4A, ne0, nrows);
    ggml_set_name(meta_tensor, "synthetic.e4a");
    gguf_add_tensor(gguf, meta_tensor);
    gguf_set_tensor_data(gguf, "synthetic.e4a", raw.data());
    if (!gguf_write_to_file(gguf, synthetic_path.path.c_str(), false)) {
        throw std::runtime_error("synthetic GGUF write failed");
    }
    gguf_free(gguf);
    ggml_free(meta_ctx);

    std::vector<std::string> synthetic_splits;
    llama_model_loader synthetic_loader(
        nullptr, nullptr, nullptr, synthetic_path.path, synthetic_splits, nullptr,
        LLAMA_LOAD_MODE_NONE, false, false, false, nullptr, nullptr);
    repacked_tensor loader_target = make_tensor(ne0, nrows);
    ggml_set_name(loader_target.tensor, "synthetic.e4a");
    synthetic_loader.init_mappings(false);
    llama_buf_map synthetic_bufs;
    const bool loader_ok = synthetic_loader.load_all_data(
        loader_target.ctx, synthetic_bufs, nullptr, nullptr, nullptr);
    ok &= loader_ok && same(reference, loader_target);

    std::printf("test-repack-stream-bake: partial=%s generic=%s fallback=%s stream=%s cancel=%s short-read=%s loader=%s\n",
        same(reference, partial) ? "PASS" : "FAIL",
        generic_ok ? "PASS" : "FAIL",
        ggml_backend_cpu_repack_chunk_alignment(wrong_shape.tensor) == 0 &&
                ggml_backend_cpu_repack_chunk_alignment(wrong_type.tensor) == 0 &&
                fallback_shape == llama_repack_stream_bake_result::unsupported &&
                fallback_type == llama_repack_stream_bake_result::unsupported ? "PASS" : "FAIL",
        same(reference, streamed) ? "PASS" : "FAIL",
        cancel_result == llama_repack_stream_bake_result::cancelled ? "PASS" : "FAIL",
        short_read_threw ? "PASS" : "FAIL",
        loader_ok && same(reference, loader_target) ? "PASS" : "FAIL");
    return ok ? 0 : 1;
#endif
}
