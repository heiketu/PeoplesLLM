#pragma once

#include "llama-ep-topology.h"
#include "llama-ep-transport.h"

#include <cstring>
#include <string>
#include <utility>
#include <vector>

struct llama_ep_worker_capability {
    llama_ep_cap_worker wire = {};
    std::vector<uint8_t> expert_bitmap;

    llama_ep_expert_ownership ownership() const {
        return {
            wire.expert_first,
            wire.expert_last,
            expert_bitmap.empty() ? nullptr : expert_bitmap.data(),
            expert_bitmap.size(),
        };
    }
};

// Parse the variable-length worker CAP payload before the master publishes any
// range or bitmap pointers. Protocol feature requirements and kernel identity
// are deployment policy and remain the caller's responsibility.
inline bool llama_ep_parse_worker_capability(
        const uint8_t              * payload,
        size_t                       payload_size,
        llama_ep_worker_capability & result,
        std::string                & error) {
    result = {};
    if (payload == nullptr || payload_size < sizeof(llama_ep_cap_worker)) {
        error = "worker CAP payload is truncated";
        return false;
    }

    llama_ep_worker_capability parsed;
    std::memcpy(&parsed.wire, payload, sizeof(parsed.wire));
    if (parsed.wire.expert_first < 0 || parsed.wire.expert_last <= parsed.wire.expert_first) {
        error = "worker CAP has invalid expert range";
        return false;
    }
    if (parsed.wire.caps & LLAMA_EP_CAP_EXPERT_BITMAP) {
        const size_t n_bits = (size_t) (parsed.wire.expert_last - parsed.wire.expert_first);
        const size_t n_bytes = (n_bits + 7) / 8;
        if (payload_size != sizeof(parsed.wire) + n_bytes) {
            error = "worker CAP expert bitmap length mismatch";
            return false;
        }
        parsed.expert_bitmap.assign(payload + sizeof(parsed.wire), payload + payload_size);
    } else if (payload_size != sizeof(parsed.wire)) {
        error = "worker CAP has unexpected trailing payload";
        return false;
    }
    if (!parsed.ownership().valid()) {
        error = "worker CAP has invalid expert ownership bitmap";
        return false;
    }

    result = std::move(parsed);
    error.clear();
    return true;
}
