#pragma once

// Sparse expert ownership map shared by the EPD loader and its unit test.
// local_to_global defines the physical expert-plane order in the worker's
// compact tensors; global_to_local makes request validation/remapping O(1).

#include <algorithm>
#include <cstdint>
#include <vector>

struct llama_ep_expert_map {
    int32_t n_expert = 0;
    int32_t first    = 0;
    int32_t last     = 0;
    bool    contiguous = false;

    std::vector<int32_t> local_to_global;
    std::vector<int32_t> global_to_local;

    bool init_ids(int32_t n_full, std::vector<int32_t> ids) {
        if (n_full < 1 || ids.empty()) {
            return false;
        }
        std::sort(ids.begin(), ids.end());
        if (ids.front() < 0 || ids.back() >= n_full ||
                std::adjacent_find(ids.begin(), ids.end()) != ids.end()) {
            return false;
        }

        n_expert = n_full;
        first = ids.front();
        last  = ids.back() + 1;
        local_to_global = std::move(ids);
        global_to_local.assign((size_t) n_full, -1);
        contiguous = true;
        for (size_t i = 0; i < local_to_global.size(); ++i) {
            const int32_t e = local_to_global[i];
            global_to_local[(size_t) e] = (int32_t) i;
            contiguous = contiguous && e == first + (int32_t) i;
        }
        return true;
    }

    bool init_range(int32_t n_full, int32_t range_first, int32_t range_last) {
        const int32_t lo = std::max<int32_t>(0, range_first);
        const int32_t hi = std::min<int32_t>(n_full, range_last);
        if (lo >= hi) {
            return false;
        }
        std::vector<int32_t> ids((size_t) (hi - lo));
        for (int32_t e = lo; e < hi; ++e) {
            ids[(size_t) (e - lo)] = e;
        }
        return init_ids(n_full, std::move(ids));
    }

    bool init_mod(int32_t n_full, int32_t remainder, int32_t modulus) {
        if (modulus < 1 || remainder < 0 || remainder >= modulus) {
            return false;
        }
        std::vector<int32_t> ids;
        for (int32_t e = remainder; e < n_full; e += modulus) {
            ids.push_back(e);
        }
        return init_ids(n_full, std::move(ids));
    }

    int32_t local(int32_t global) const {
        return global >= 0 && global < (int32_t) global_to_local.size()
            ? global_to_local[(size_t) global] : -1;
    }

    std::vector<uint8_t> bitmap() const {
        std::vector<uint8_t> bits(((size_t) n_expert + 7) / 8, 0);
        for (int32_t e : local_to_global) {
            bits[(size_t) e >> 3] |= (uint8_t) (1u << (e & 7));
        }
        return bits;
    }
};

