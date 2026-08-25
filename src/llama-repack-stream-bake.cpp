#include "llama-repack-stream-bake.h"

#include "ggml-cpu.h"
#include "ggml.h"
#include "llama-impl.h"
#include "llama-mmap.h"

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <vector>

#if !defined(_WIN32)
#include <unistd.h>
#endif

namespace {

struct env_config {
    bool enabled = false;
    size_t chunk_bytes = 1024 * 1024;
};

const env_config & get_env_config() {
    static const env_config config = []() {
        env_config result;
        const char * enabled = std::getenv("GGML_STREAM_BAKE");
        if (enabled == nullptr) {
            enabled = std::getenv("GGML_E4A_STREAM_BAKE");
        }
        result.enabled = enabled != nullptr && std::atoi(enabled) != 0;
        if (!result.enabled) {
            return result;
        }

        const char * chunk = std::getenv("GGML_STREAM_BAKE_CHUNK_MIB");
        if (chunk == nullptr) {
            chunk = std::getenv("GGML_E4A_STREAM_BAKE_CHUNK_MIB");
        }
        if (chunk == nullptr) {
            return result;
        }
        char * end = nullptr;
        errno = 0;
        const unsigned long long mib = std::strtoull(chunk, &end, 10);
        if (errno != 0 || end == chunk || *end != '\0' || mib < 1 || mib > 1024) {
            LLAMA_LOG_WARN("invalid stream-bake chunk MiB '%s'; disabling stream bake\n", chunk);
            result.enabled = false;
            return result;
        }
        result.chunk_bytes = (size_t) mib * 1024 * 1024;
        return result;
    }();
    return config;
}

#if !defined(_WIN32)
void read_at_exact(int fd, void * data, size_t size, size_t offset) {
    uint8_t * dst = static_cast<uint8_t *>(data);
    size_t done = 0;
    while (done < size) {
        const ssize_t nread = ::pread(fd, dst + done, size - done, (off_t) (offset + done));
        if (nread < 0) {
            if (errno == EINTR) {
                continue;
            }
            throw std::runtime_error(format("CPU repack stream bake read failed: %s", std::strerror(errno)));
        }
        if (nread == 0) {
            throw std::runtime_error("CPU repack stream bake unexpectedly reached end of file");
        }
        done += (size_t) nread;
    }
}
#endif

struct stream_slot {
    explicit stream_slot(size_t size) : data(size) {}

    std::vector<uint8_t> data;
    size_t size = 0;
    size_t tensor_offset = 0;
    bool full = false;
    std::mutex mutex;
    std::condition_variable full_cv;
    std::condition_variable empty_cv;
};

} // namespace

bool llama_repack_stream_bake_env_enabled() {
    return get_env_config().enabled;
}

size_t llama_repack_stream_bake_env_chunk_bytes() {
    return get_env_config().chunk_bytes;
}

llama_repack_stream_bake_result llama_repack_stream_bake_load(
        llama_file & file,
        size_t file_offset,
        ggml_tensor * tensor,
        const llama_repack_stream_bake_config & config,
        const std::function<bool(size_t, size_t)> & progress) {
#if defined(_WIN32)
    (void) file;
    (void) file_offset;
    (void) tensor;
    (void) config;
    (void) progress;
    return llama_repack_stream_bake_result::unsupported;
#else
    if (tensor == nullptr || config.chunk_bytes == 0 || file.has_direct_io()) {
        return llama_repack_stream_bake_result::unsupported;
    }

    const size_t alignment = ggml_backend_cpu_repack_chunk_alignment(tensor);
    const size_t tensor_size = ggml_nbytes(tensor);
    if (alignment == 0 || tensor_size == 0) {
        return llama_repack_stream_bake_result::unsupported;
    }
    if (file_offset > file.size() || tensor_size > file.size() - file_offset) {
        throw std::runtime_error(format("CPU repack stream bake tensor '%s' is outside file bounds", tensor->name));
    }

    const size_t chunk_panels = std::min(
        tensor_size / alignment, std::max<size_t>(1, config.chunk_bytes / alignment));
    const size_t chunk_bytes = chunk_panels * alignment;
    const size_t n_chunks = (tensor_size + chunk_bytes - 1) / chunk_bytes;

    stream_slot slots[2] = {stream_slot(chunk_bytes), stream_slot(chunk_bytes)};
    std::atomic<bool> stop{false};
    std::atomic<bool> reader_done{false};
    std::exception_ptr reader_error;
    std::mutex error_mutex;

    auto notify_slots = [&]() {
        for (stream_slot & slot : slots) {
            slot.full_cv.notify_all();
            slot.empty_cv.notify_all();
        }
    };

    std::thread reader([&]() {
        try {
            for (size_t i = 0; i < n_chunks && !stop.load(std::memory_order_acquire); ++i) {
                stream_slot & slot = slots[i % 2];
                {
                    std::unique_lock<std::mutex> lock(slot.mutex);
                    slot.empty_cv.wait(lock, [&]() {
                        return !slot.full || stop.load(std::memory_order_acquire);
                    });
                    if (stop.load(std::memory_order_acquire)) {
                        break;
                    }
                }

                const size_t offset = i * chunk_bytes;
                const size_t size = std::min(chunk_bytes, tensor_size - offset);
                read_at_exact(file.file_id(), slot.data.data(), size, file_offset + offset);

                {
                    std::lock_guard<std::mutex> lock(slot.mutex);
                    slot.size = size;
                    slot.tensor_offset = offset;
                    slot.full = true;
                }
                slot.full_cv.notify_one();
            }
        } catch (...) {
            {
                std::lock_guard<std::mutex> lock(error_mutex);
                reader_error = std::current_exception();
            }
            stop.store(true, std::memory_order_release);
        }
        reader_done.store(true, std::memory_order_release);
        notify_slots();
    });

    bool cancelled = false;
    size_t completed = 0;
    size_t next_progress = std::max(config.progress_interval_bytes, chunk_bytes);
    try {
        for (size_t i = 0; i < n_chunks; ++i) {
            stream_slot & slot = slots[i % 2];
            size_t size = 0;
            size_t offset = 0;
            {
                std::unique_lock<std::mutex> lock(slot.mutex);
                slot.full_cv.wait(lock, [&]() {
                    return slot.full || stop.load(std::memory_order_acquire) || reader_done.load(std::memory_order_acquire);
                });
                if (!slot.full) {
                    break;
                }
                size = slot.size;
                offset = slot.tensor_offset;
            }

            if (config.validate && !ggml_validate_row_data(tensor->type, slot.data.data(), size)) {
                throw std::runtime_error(format("CPU repack stream bake validation failed for tensor '%s'", tensor->name));
            }
            if (!ggml_backend_cpu_repack_write_chunk(tensor, slot.data.data(), offset, size)) {
                throw std::runtime_error(format("CPU repack stream bake rejected an aligned chunk for tensor '%s'", tensor->name));
            }
            completed += size;

            {
                std::lock_guard<std::mutex> lock(slot.mutex);
                slot.full = false;
            }
            slot.empty_cv.notify_one();

            if (progress && (completed >= next_progress || completed == tensor_size)) {
                if (!progress(completed, tensor_size)) {
                    cancelled = true;
                    stop.store(true, std::memory_order_release);
                    notify_slots();
                    break;
                }
                next_progress = completed + std::max(config.progress_interval_bytes, chunk_bytes);
            }
        }
    } catch (...) {
        stop.store(true, std::memory_order_release);
        notify_slots();
        reader.join();
        throw;
    }

    stop.store(true, std::memory_order_release);
    notify_slots();
    reader.join();

    {
        std::lock_guard<std::mutex> lock(error_mutex);
        if (reader_error) {
            std::rethrow_exception(reader_error);
        }
    }
    if (cancelled) {
        return llama_repack_stream_bake_result::cancelled;
    }
    if (completed != tensor_size) {
        throw std::runtime_error(format(
            "CPU repack stream bake stopped after %zu of %zu bytes for tensor '%s'", completed, tensor_size, tensor->name));
    }
    return llama_repack_stream_bake_result::complete;
#endif
}
