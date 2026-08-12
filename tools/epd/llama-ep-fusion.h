#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>

// Merge raw [K, N, E] gate and up tensors into [K, 2*N, E]. Each expert
// plane remains contiguous and gate precedes up, matching the gate_up view
// contract used by the EPD graph. The caller owns all buffers.
inline bool llama_ep_merge_gate_up_raw(
        void * dst,
        size_t dst_size,
        const void * gate,
        size_t gate_size,
        const void * up,
        size_t up_size,
        size_t n_expert) {
    if (dst == nullptr || gate == nullptr || up == nullptr || n_expert == 0 ||
            gate_size % n_expert != 0 || up_size % n_expert != 0 ||
            gate_size > SIZE_MAX - up_size || dst_size != gate_size + up_size) {
        return false;
    }

    const size_t gate_plane = gate_size / n_expert;
    const size_t up_plane = up_size / n_expert;
    for (size_t expert = 0; expert < n_expert; ++expert) {
        char * dst_plane = (char *) dst + expert * (gate_plane + up_plane);
        memcpy(dst_plane, (const char *) gate + expert * gate_plane, gate_plane);
        memcpy(dst_plane + gate_plane, (const char *) up + expert * up_plane, up_plane);
    }
    return true;
}
