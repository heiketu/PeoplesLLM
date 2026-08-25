#pragma once

#include "llama-ep-topology.h"
#include "llama-ep-transport.h"

#include <cstring>
#include <string>
#include <utility>
#include <vector>

struct llama_ep_worker_capability {
    llama_ep_cap_worker wire = {};
    llama_ep_precision_contract precision = {};
    std::vector<uint8_t> expert_bitmap;

    bool has_precision() const {
        return (wire.caps & LLAMA_EP_CAP_PRECISION_CONTRACT) != 0;
    }

    llama_ep_expert_ownership ownership() const {
        return {
            wire.expert_first,
            wire.expert_last,
            expert_bitmap.empty() ? nullptr : expert_bitmap.data(),
            expert_bitmap.size(),
        };
    }
};

inline uint64_t llama_ep_fnv1a64_update(uint64_t hash, const void * data, size_t size) {
    const uint8_t * bytes = (const uint8_t *) data;
    for (size_t i = 0; i < size; ++i) {
        hash ^= bytes[i];
        hash *= UINT64_C(1099511628211);
    }
    return hash;
}

inline uint64_t llama_ep_precision_contract_id(
        const llama_ep_precision_contract & contract,
        uint32_t                            kernel_id) {
    uint64_t hash = UINT64_C(1469598103934665603);
    hash = llama_ep_fnv1a64_update(hash, &contract, offsetof(llama_ep_precision_contract, contract_id));
    hash = llama_ep_fnv1a64_update(hash, &kernel_id, sizeof(kernel_id));
    return hash;
}

inline llama_ep_precision_contract llama_ep_make_cpu_repack_precision_contract(
        uint32_t kernel_id,
        uint64_t model_schema_id,
        uint64_t data_epoch_id) {
    llama_ep_precision_contract contract = {};
    contract.version = 1;
    contract.profile_flags = LLAMA_EP_PRECISION_DETERMINISTIC_ENDPOINT |
        LLAMA_EP_PRECISION_RAW_SLOT_F32 | LLAMA_EP_PRECISION_WEIGHTED_SLOT_F32;
    contract.activation_id = LLAMA_EP_PRECISION_ACTIVATION_Q8_0_BLOCK32_FP16_NATIVE;
    contract.dot_id = LLAMA_EP_PRECISION_DOT_I8_I8_I32_BLOCK32_F32;
    contract.ffn_id = LLAMA_EP_PRECISION_FFN_MODEL_SCHEMA_E8M0_LEGACY_FINITE;
    contract.merge_id = LLAMA_EP_PRECISION_MERGE_PER_SLOT_F32;
    contract.model_schema_id = model_schema_id;
    contract.data_epoch_id = data_epoch_id;
    contract.contract_id = llama_ep_precision_contract_id(contract, kernel_id);
    return contract;
}

inline bool llama_ep_precision_contract_valid(
        const llama_ep_precision_contract & contract,
        uint32_t                            kernel_id) {
    const uint32_t required_profiles = LLAMA_EP_PRECISION_DETERMINISTIC_ENDPOINT |
        LLAMA_EP_PRECISION_RAW_SLOT_F32 | LLAMA_EP_PRECISION_WEIGHTED_SLOT_F32;
    return contract.version == 1 &&
        (contract.profile_flags & required_profiles) == required_profiles &&
        contract.activation_id == LLAMA_EP_PRECISION_ACTIVATION_Q8_0_BLOCK32_FP16_NATIVE &&
        contract.dot_id == LLAMA_EP_PRECISION_DOT_I8_I8_I32_BLOCK32_F32 &&
        contract.ffn_id == LLAMA_EP_PRECISION_FFN_MODEL_SCHEMA_E8M0_LEGACY_FINITE &&
        contract.merge_id == LLAMA_EP_PRECISION_MERGE_PER_SLOT_F32 &&
        contract.model_schema_id != 0 &&
        contract.contract_id == llama_ep_precision_contract_id(contract, kernel_id);
}

inline bool llama_ep_precision_contract_equal(
        const llama_ep_precision_contract & lhs,
        const llama_ep_precision_contract & rhs) {
    return std::memcmp(&lhs, &rhs, sizeof(lhs)) == 0;
}

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
    size_t offset = sizeof(parsed.wire);
    if (parsed.wire.caps & LLAMA_EP_CAP_PRECISION_CONTRACT) {
        if (payload_size < offset + sizeof(parsed.precision)) {
            error = "worker CAP precision contract is truncated";
            return false;
        }
        std::memcpy(&parsed.precision, payload + offset, sizeof(parsed.precision));
        offset += sizeof(parsed.precision);
        if (!llama_ep_precision_contract_valid(parsed.precision, parsed.wire.kernel_id)) {
            error = "worker CAP has invalid precision contract";
            return false;
        }
    }
    if (parsed.wire.caps & LLAMA_EP_CAP_EXPERT_BITMAP) {
        const size_t n_bits = (size_t) (parsed.wire.expert_last - parsed.wire.expert_first);
        const size_t n_bytes = (n_bits + 7) / 8;
        if (payload_size != offset + n_bytes) {
            error = "worker CAP expert bitmap length mismatch";
            return false;
        }
        parsed.expert_bitmap.assign(payload + offset, payload + payload_size);
    } else if (payload_size != offset) {
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
