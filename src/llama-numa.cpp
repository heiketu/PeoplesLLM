#include "llama-numa.h"

// llama.h declares llama_numa_set_mirror() inside extern "C"; include it here so this
// definition gets C linkage too and common/ can link against it.
#include "llama.h"

#include "ggml-backend.h"

#include <cstdio>

size_t llama_get_available_ram_bytes() {
#if defined(__linux__)
    FILE * f = fopen("/proc/meminfo", "r");
    if (!f) return 0;
    char line[256];
    size_t kb = 0;
    while (fgets(line, sizeof(line), f)) {
        if (sscanf(line, "MemAvailable: %zu kB", &kb) == 1) break;
    }
    fclose(f);
    return kb * 1024;
#else
    return 0;
#endif
}

// resolve a ggml-cpu symbol through the CPU backend registry.
// returns nullptr when the CPU backend is not loaded or the symbol is missing.
static void * llama_numa_resolve(const char * name) {
    auto * dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    if (dev == nullptr) {
        return nullptr;
    }
    auto * reg = ggml_backend_dev_backend_reg(dev);
    if (reg == nullptr) {
        return nullptr;
    }
    return ggml_backend_reg_get_proc_address(reg, name);
}

// cached resolver, but re-tries while unresolved: the first call may happen before the CPU
// backend is registered (e.g. frontends applying --numa before llama_backend_init()).
#define LLAMA_NUMA_PROC(var, name) \
    static void * var = nullptr; \
    if (var == nullptr) { var = llama_numa_resolve(name); }

bool llama_numa_mirror_active() {
    LLAMA_NUMA_PROC(fn, "ggml_backend_cpu_numa_mirror_active");
    return fn ? ((bool (*)(void)) fn)() : false;
}

int llama_numa_node_count() {
    LLAMA_NUMA_PROC(fn, "ggml_backend_cpu_numa_node_count");
    return fn ? ((int (*)(void)) fn)() : 1;
}

uint32_t llama_numa_get_mirror() {
    LLAMA_NUMA_PROC(fn, "ggml_backend_cpu_numa_get_mirror");
    return fn ? ((uint32_t (*)(void)) fn)() : 0;
}

void llama_numa_set_mirror(uint32_t flags) {
    LLAMA_NUMA_PROC(fn, "ggml_backend_cpu_numa_set_mirror");
    if (fn) {
        ((void (*)(uint32_t)) fn)(flags);
    }
}

void * llama_numa_alloc(size_t size, int node) {
    LLAMA_NUMA_PROC(fn, "ggml_backend_cpu_numa_alloc");
    return fn ? ((void * (*)(size_t, int)) fn)(size, node) : nullptr;
}

void llama_numa_free(void * ptr, size_t size) {
    LLAMA_NUMA_PROC(fn, "ggml_backend_cpu_numa_free");
    if (fn) {
        ((void (*)(void *, size_t)) fn)(ptr, size);
    }
}

void llama_numa_bind(void * ptr, size_t size, int node) {
    LLAMA_NUMA_PROC(fn, "ggml_backend_cpu_numa_bind");
    if (fn) {
        ((void (*)(void *, size_t, int)) fn)(ptr, size, node);
    }
}

void llama_numa_bind_policy(void * ptr, size_t size, int node) {
    LLAMA_NUMA_PROC(fn, "ggml_backend_cpu_numa_bind_policy");
    if (fn) {
        ((void (*)(void *, size_t, int)) fn)(ptr, size, node);
    }
}

void llama_numa_tensor_set_mirror(ggml_tensor * tensor, void * const * node_data) {
    LLAMA_NUMA_PROC(fn, "ggml_backend_cpu_numa_tensor_set_mirror");
    if (fn) {
        ((void (*)(ggml_tensor *, void * const *)) fn)(tensor, node_data);
    }
}

void llama_numa_tensor_clear_mirror(ggml_tensor * tensor) {
    LLAMA_NUMA_PROC(fn, "ggml_backend_cpu_numa_tensor_clear_mirror");
    if (fn) {
        ((void (*)(ggml_tensor *)) fn)(tensor);
    }
}

void llama_numa_tensor_resync(ggml_tensor * tensor) {
    LLAMA_NUMA_PROC(fn, "ggml_backend_cpu_numa_tensor_resync");
    if (fn) {
        ((void (*)(ggml_tensor *)) fn)(tensor);
    }
}
