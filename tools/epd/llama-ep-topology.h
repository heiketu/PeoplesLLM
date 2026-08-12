#pragma once

// Shared EP ownership and topology validation.  Keep capability parsing and
// coverage rules out of the dealer: the dealer consumes an already validated
// holder table and should remain a pure assignment planner.

#include <cstddef>
#include <cstdint>
#include <vector>

struct llama_ep_expert_ownership {
    int32_t first = 0;
    int32_t last  = 0;
    const uint8_t * bitmap = nullptr;
    size_t bitmap_size = 0;

    size_t expected_bitmap_size() const {
        return last > first ? ((size_t) (last - first) + 7) / 8 : 0;
    }

    bool valid() const {
        if (first < 0 || last <= first) {
            return false;
        }
        if (bitmap == nullptr) {
            return bitmap_size == 0;
        }
        if (bitmap_size != expected_bitmap_size()) {
            return false;
        }
        const unsigned used_tail_bits = (unsigned) (last - first) & 7u;
        if (used_tail_bits != 0) {
            const uint8_t padding_mask = (uint8_t) ~((1u << used_tail_bits) - 1u);
            if ((bitmap[bitmap_size - 1] & padding_mask) != 0) {
                return false;
            }
        }
        return true;
    }

    bool holds(int32_t expert) const {
        if (expert < first || expert >= last) {
            return false;
        }
        if (bitmap == nullptr) {
            return true;
        }
        const size_t bit = (size_t) (expert - first);
        return bit / 8 < bitmap_size &&
            (bitmap[bit >> 3] & (uint8_t) (1u << (bit & 7))) != 0;
    }

    size_t count() const {
        if (!valid()) {
            return 0;
        }
        if (bitmap == nullptr) {
            return (size_t) (last - first);
        }
        size_t result = 0;
        for (size_t i = 0; i < bitmap_size; ++i) {
            uint8_t value = bitmap[i];
            while (value != 0) {
                value &= (uint8_t) (value - 1);
                ++result;
            }
        }
        return result;
    }
};

inline bool llama_ep_shard_cover_impl(
        int             n_expert,
        int             n_endpoints,
        const int32_t * expert_first,
        const int32_t * expert_last,
        bool            exact) {
    if (n_expert < 1 || n_endpoints < 1 || n_endpoints > 63 ||
            expert_first == nullptr || expert_last == nullptr) {
        return false;
    }

    std::vector<uint8_t> cover((size_t) n_expert, 0);
    for (int i = 0; i < n_endpoints; ++i) {
        const int32_t first = expert_first[i];
        const int32_t last  = expert_last[i];
        if (first < 0 || last <= first || last > n_expert) {
            return false;
        }
        for (int32_t e = first; e < last; ++e) {
            if (++cover[(size_t) e] != 1 && exact) {
                return false;
            }
        }
    }
    for (uint8_t n : cover) {
        if (n == 0 || (exact && n != 1)) {
            return false;
        }
    }
    return true;
}

inline bool llama_ep_exact_shard_cover(
        int n_expert, int n_endpoints,
        const int32_t * expert_first, const int32_t * expert_last) {
    return llama_ep_shard_cover_impl(n_expert, n_endpoints, expert_first, expert_last, true);
}

inline bool llama_ep_full_shard_cover(
        int n_expert, int n_endpoints,
        const int32_t * expert_first, const int32_t * expert_last) {
    return llama_ep_shard_cover_impl(n_expert, n_endpoints, expert_first, expert_last, false);
}

// Here bit i is endpoint i; there is no master bit in this topology-only table.
inline bool llama_ep_holder_cover(
        int n_expert, int n_endpoints, const uint64_t * holders, bool exact) {
    if (n_expert < 1 || n_endpoints < 1 || n_endpoints > 63 || holders == nullptr) {
        return false;
    }
    const uint64_t allowed = (1ull << n_endpoints) - 1;
    for (int e = 0; e < n_expert; ++e) {
        const uint64_t mask = holders[e];
        if (mask == 0 || (mask & ~allowed) != 0) {
            return false;
        }
        if (exact && (mask & (mask - 1)) != 0) {
            return false;
        }
    }
    return true;
}
