#pragma once

// NUMA mirror helpers (GGML_NUMA_STRATEGY_MIRROR).
//
// Thin wrappers around the ggml-cpu NUMA mirror API. All ggml-cpu symbols are resolved through
// the CPU backend registry (ggml_backend_reg_get_proc_address) instead of being linked directly,
// so this also works with GGML_BACKEND_DL builds where libggml-cpu is dlopen'ed.

#include "ggml-cpu.h" // GGML_NUMA_MAX_NODES, enum ggml_numa_mirror_flags

#include <cstddef>
#include <cstdint>

struct ggml_tensor;

// true if the NUMA strategy is MIRROR and the system has more than one NUMA node
bool llama_numa_mirror_active();

// number of NUMA nodes detected (>= 1)
int llama_numa_node_count();

// which ggml_numa_mirror_flags to duplicate per node (default: GGML_NUMA_MIRROR_ALL)
uint32_t llama_numa_get_mirror();

// allocate/free memory bound to a specific NUMA node
void * llama_numa_alloc(size_t size, int node);
void   llama_numa_free(void * ptr, size_t size);

// best-effort migrate an already-populated range onto a node
void llama_numa_bind(void * ptr, size_t size, int node);
void llama_numa_bind_policy(void * ptr, size_t size, int node);

// attach/detach per-node copies to a tensor. node_data must have llama_numa_node_count() entries.
void llama_numa_tensor_set_mirror(ggml_tensor * tensor, void * const * node_data);
void llama_numa_tensor_clear_mirror(ggml_tensor * tensor);

// re-sync node copies after a write that bypassed graph-CPY replication
void llama_numa_tensor_resync(ggml_tensor * tensor);

// MemAvailable from /proc/meminfo (0 when unknown / non-Linux); used for mirror RAM sanity checks
size_t llama_get_available_ram_bytes();
