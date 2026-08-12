#pragma once

#include <algorithm>
#include <cstdint>
#include <limits>

// Pure policy helpers for service-rate-aware replicated-expert placement.
// Cost units are currently microseconds per expert-token assignment, separated
// by the caller into decode and prefill classes.

static inline int64_t llama_ep_service_ewma(int64_t old_cost, int64_t sample_cost) {
    constexpr int64_t max_cost = 1000000;
    int64_t sample = std::max<int64_t>(1, std::min<int64_t>(max_cost, sample_cost));
    if (old_cost <= 0) {
        return sample;
    }
    const int64_t old = std::min<int64_t>(max_cost, old_cost);
    sample = std::max<int64_t>(std::max<int64_t>(1, old / 4), sample);
    sample = std::min<int64_t>(std::min<int64_t>(max_cost, old * 4), sample);
    return (old * 7 + sample + 4) / 8;
}

// Fill unknown (zero) endpoint costs with the mean of known peers. This avoids
// favoring or penalizing a worker before it has produced its first clean sample.
static inline bool llama_ep_service_normalize_costs(
        const int64_t * observed, int n, int64_t fallback, int64_t * out) {
    if (observed == nullptr || out == nullptr || n < 0 || fallback <= 0) {
        return false;
    }
    int64_t sum = 0;
    int64_t count = 0;
    for (int i = 0; i < n; ++i) {
        if (observed[i] < 0) {
            return false;
        }
        if (observed[i] > 0) {
            if (sum > std::numeric_limits<int64_t>::max() - observed[i]) {
                return false;
            }
            sum += observed[i];
            ++count;
        }
    }
    const int64_t replacement = count > 0 ? std::max<int64_t>(1, sum / count) : fallback;
    for (int i = 0; i < n; ++i) {
        out[i] = observed[i] > 0 ? observed[i] : replacement;
    }
    return true;
}

static inline bool llama_ep_service_work_units(int64_t assignments, int64_t cost, int64_t & out) {
    if (assignments < 0 || cost <= 0 ||
            (assignments > 0 && cost > std::numeric_limits<int64_t>::max() / assignments)) {
        return false;
    }
    out = assignments * cost;
    return true;
}

// Repeat-aware work estimate used by both dealer queue accounting and service
// sampling. A new expert row streams the full weight plane; another row of an
// expert already present in the request pays only the configured marginal
// cost. Keeping one definition prevents the learned endpoint rate from using
// a different unit than the dealer objective.
static inline bool llama_ep_service_split_work_units(
        int64_t unique_experts,
        int64_t repeated_rows,
        int64_t new_expert_cost,
        int64_t repeated_row_cost,
        int64_t & out) {
    if (unique_experts < 0 || repeated_rows < 0 || new_expert_cost <= 0 || repeated_row_cost < 0 ||
            (unique_experts > 0 && new_expert_cost > std::numeric_limits<int64_t>::max() / unique_experts) ||
            (repeated_rows > 0 && repeated_row_cost > std::numeric_limits<int64_t>::max() / repeated_rows)) {
        return false;
    }
    const int64_t unique_work = unique_experts * new_expert_cost;
    const int64_t repeated_work = repeated_rows * repeated_row_cost;
    if (unique_work > std::numeric_limits<int64_t>::max() - repeated_work) {
        return false;
    }
    out = unique_work + repeated_work;
    return true;
}
