#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <utility>
#include <vector>

enum class ggml_shard_domain_kind {
    GPU,
    NUMA,
    REMOTE,
};

enum class ggml_shard_placement {
    OWNER,
    MIRRORED,
    SPLIT,
    PARTIAL,
};

enum class ggml_shard_collective {
    NONE,
    BROADCAST,
    ALL_GATHER,
    ALL_REDUCE,
    REDUCE_SCATTER,
};

constexpr size_t GGML_SHARD_NO_DOMAIN = std::numeric_limits<size_t>::max();

struct ggml_shard_domain {
    size_t id = 0;
    ggml_shard_domain_kind kind = ggml_shard_domain_kind::GPU;
    float capacity = 1.0f;
};

struct ggml_shard_segment {
    int64_t extent = 0;
    uint32_t repetitions = 1;
    int64_t granularity = 1;
};

struct ggml_shard_plan_input {
    ggml_shard_placement placement = ggml_shard_placement::MIRRORED;
    int axis = -1;
    size_t owner = 0;
    size_t rotation = 0;
    ggml_shard_collective collective_before = ggml_shard_collective::NONE;
    ggml_shard_collective collective_after = ggml_shard_collective::NONE;
    std::vector<ggml_shard_domain> domains;
    std::vector<ggml_shard_segment> segments;
};

struct ggml_shard_plan {
    ggml_shard_placement placement = ggml_shard_placement::MIRRORED;
    int axis = -1;
    size_t owner = 0;
    size_t rotation = 0;
    ggml_shard_collective collective_before = ggml_shard_collective::NONE;
    ggml_shard_collective collective_after = ggml_shard_collective::NONE;
    std::vector<ggml_shard_domain> domains;
    std::vector<ggml_shard_segment> segments;
    std::vector<int64_t> offsets;
    std::vector<int64_t> extents;
};

struct ggml_shard_execution_phase {
    ggml_shard_collective reduction_after = ggml_shard_collective::NONE;
    size_t broadcast_root = GGML_SHARD_NO_DOMAIN;
    std::vector<int> broadcast_values;
};

struct ggml_shard_window {
    int64_t begin = 0;
    int64_t end = 0;
    int64_t stride = 0;
};

inline bool ggml_shard_window_equal(
        int64_t extent,
        size_t n_domains,
        size_t domain,
        int64_t granularity,
        ggml_shard_window & output) {
    if (extent < 0 || n_domains == 0 || n_domains > static_cast<size_t>(std::numeric_limits<int64_t>::max()) ||
            domain >= n_domains || granularity <= 0) {
        return false;
    }

    const int64_t n = static_cast<int64_t>(n_domains);
    const int64_t unaligned = extent / n + (extent % n != 0);
    if (unaligned > std::numeric_limits<int64_t>::max() - (granularity - 1)) {
        return false;
    }
    const int64_t width = (unaligned + granularity - 1) / granularity * granularity;
    const int64_t begin = width == 0 || domain <= static_cast<size_t>(extent / width) ?
        std::min<int64_t>(static_cast<int64_t>(domain) * width, extent) : extent;
    output.begin = begin;
    output.end = std::min<int64_t>(begin + std::min<int64_t>(width, extent - begin), extent);
    output.stride = width;
    return true;
}

inline bool ggml_shard_plan_build(
        const ggml_shard_plan_input & input,
        ggml_shard_plan & output,
        std::string * error = nullptr) {
    auto fail = [&](const char * message) {
        if (error != nullptr) {
            *error = message;
        }
        return false;
    };

    const size_t n_domains = input.domains.size();
    if (n_domains == 0 || n_domains > 16) {
        return fail("domain count must be between 1 and 16");
    }
    if (input.segments.empty() || input.segments.size() > 16) {
        return fail("segment count must be between 1 and 16");
    }
    if (input.owner >= n_domains) {
        return fail("owner is outside the domain set");
    }
    if (input.placement == ggml_shard_placement::SPLIT && (input.axis < 0 || input.axis > 3)) {
        return fail("split placement requires an axis between 0 and 3");
    }

    double capacity_sum = 0.0;
    for (size_t i = 0; i < n_domains; ++i) {
        const ggml_shard_domain & domain = input.domains[i];
        if (!std::isfinite(domain.capacity) || domain.capacity < 0.0f) {
            return fail("domain capacity must be finite and non-negative");
        }
        capacity_sum += domain.capacity;
        for (size_t j = 0; j < i; ++j) {
            if (input.domains[j].id == domain.id) {
                return fail("domain ids must be unique");
            }
        }
    }
    if (!std::isfinite(capacity_sum)) {
        return fail("domain capacity sum must be finite");
    }

    for (const ggml_shard_segment & segment : input.segments) {
        if (segment.extent < 0 || segment.repetitions == 0 || segment.granularity <= 0) {
            return fail("segment extent, repetitions, or granularity is invalid");
        }
    }

    ggml_shard_plan result;
    result.placement = input.placement;
    result.axis = input.axis;
    result.owner = input.owner;
    result.rotation = input.rotation % n_domains;
    result.collective_before = input.collective_before;
    result.collective_after = input.collective_after;
    result.domains = input.domains;
    result.segments = input.segments;
    result.offsets.assign(input.segments.size() * n_domains, 0);
    result.extents.assign(input.segments.size() * n_domains, 0);

    const size_t rotation = result.rotation;
    for (size_t is = 0; is < input.segments.size(); ++is) {
        const ggml_shard_segment & segment = input.segments[is];

        if (input.placement == ggml_shard_placement::OWNER) {
            result.extents[is * n_domains + input.owner] = segment.extent;
            continue;
        }
        if (input.placement != ggml_shard_placement::SPLIT) {
            for (size_t i = 0; i < n_domains; ++i) {
                result.extents[is * n_domains + i] = segment.extent;
            }
            continue;
        }

        int64_t low = 0;
        double capacity_scan = 0.0;
        for (size_t j = 0; j + 1 < n_domains; ++j) {
            const size_t domain = (j + rotation) % n_domains;
            capacity_scan += input.domains[domain].capacity;
            int64_t high;
            if (capacity_sum == 0.0) {
                const int64_t quotient = segment.extent / static_cast<int64_t>(n_domains);
                const int64_t remainder = segment.extent % static_cast<int64_t>(n_domains);
                high = quotient * static_cast<int64_t>(j + 1) +
                    remainder * static_cast<int64_t>(j + 1) / static_cast<int64_t>(n_domains);
            } else {
                high = static_cast<int64_t>(static_cast<long double>(segment.extent) * capacity_scan / capacity_sum);
            }
            high = std::max<int64_t>(low, std::min<int64_t>(high, segment.extent));
            high -= high % segment.granularity;
            result.offsets[is * n_domains + domain] = low;
            result.extents[is * n_domains + domain] = high - low;
            low = high;
        }
        const size_t last = (n_domains - 1 + rotation) % n_domains;
        result.offsets[is * n_domains + last] = low;
        result.extents[is * n_domains + last] = segment.extent - low;
    }

    output = std::move(result);
    if (error != nullptr) {
        error->clear();
    }
    return true;
}
