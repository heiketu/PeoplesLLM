#pragma once

#include <cstdint>

// Estimate the number of compact routed rows a NUMA worker receives for one
// decode token. Strict N-way sharding gets top_k/N rows on average; hotspot
// replicas naturally raise the estimate with their larger owned fraction.
// An explicit override is useful for measured skew or nonstandard dealers.
static inline int llama_ep_autotune_compact_rows(
        int top_k, int n_owned, int n_full, int override_rows = 0) {
    if (top_k < 1 || n_owned < 1 || n_full < 1 || n_owned > n_full || override_rows < 0) {
        return 0;
    }
    const int max_rows = top_k < n_owned ? top_k : n_owned;
    if (override_rows > 0) {
        return override_rows < max_rows ? override_rows : max_rows;
    }
    const int64_t scaled = (int64_t) top_k * n_owned;
    int rows = (int) ((scaled + n_full - 1) / n_full);
    if (rows < 1) {
        rows = 1;
    }
    return rows < max_rows ? rows : max_rows;
}

// Pick the smallest candidate within tolerance of the global best. Candidate
// order is expected to be ascending so equivalent bandwidth picks the smaller
// thread team and avoids needless barrier overhead.
static inline int llama_ep_autotune_select(
        const int * candidates, const double * median_ms, int n, double tolerance) {
    if (candidates == nullptr || median_ms == nullptr || n < 1 || tolerance < 0.0) {
        return 0;
    }
    double best = median_ms[0];
    for (int i = 0; i < n; ++i) {
        if (candidates[i] < 1 || median_ms[i] <= 0.0) {
            return 0;
        }
        if (median_ms[i] < best) {
            best = median_ms[i];
        }
    }
    const double limit = best * (1.0 + tolerance);
    for (int i = 0; i < n; ++i) {
        if (median_ms[i] <= limit) {
            return candidates[i];
        }
    }
    return 0;
}
