#include "llama-hot-expert.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

static float rounded_product(float a, float b) {
    const volatile float result = a*b;
    return result;
}

static float rounded_add(float a, float b) {
    const volatile float result = a + b;
    return result;
}

static double mean(const std::vector<double> & values) {
    double sum = 0.0;
    for (double value : values) {
        sum += value;
    }
    return sum/values.size();
}

static double median(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    const size_t middle = values.size()/2;
    return values.size() % 2 ? values[middle] : (values[middle - 1] + values[middle])*0.5;
}

int main() {
    constexpr int64_t n_embd = 4096;
    constexpr int64_t n_slots = 6;

    const char * slot_env = std::getenv("GGML_HOT_EXPERT_SLOT_ORDER");
    const bool expected_slot_order = slot_env ? std::atoi(slot_env) != 0 : true;
    if (llama_hot_expert_slot_order_enabled() != expected_slot_order) {
        std::fprintf(stderr, "slot-order default/fallback selection failed\n");
        return 1;
    }

    const bool avx512_supported = llama_hot_expert_slot_merge_avx512_supported();
    const char * avx512_env = std::getenv("GGML_HOT_EXPERT_SLOT_MERGE_AVX512");
    const bool avx512_requested = avx512_env ? std::atoi(avx512_env) != 0 : true;
    if (llama_hot_expert_slot_merge_avx512_enabled() != (avx512_requested && avx512_supported)) {
        std::fprintf(stderr, "slot-merge AVX512 feature selection failed\n");
        return 1;
    }

    const float weights[n_slots] = { 0.31f, 0.23f, 0.17f, 0.13f, 0.09f, 0.07f };
    const float target[n_slots] = { 1.0e20f, 1.0f, -1.0e20f, 1.0f, 1.0f, 1.0f };
    std::vector<float> raw(n_slots*n_embd);
    for (int64_t slot = 0; slot < n_slots; ++slot) {
        raw[slot*n_embd + 0] = target[slot]/weights[slot];
        raw[slot*n_embd + 1] = std::ldexp((float) (slot - 2), (int) (slot*3 - 8));
        raw[slot*n_embd + 2] = (slot & 1 ? -1.0f : 1.0f)*(0.001f + 0.17f*(float) slot);
        raw[slot*n_embd + 3] = (float) (slot + 1)/7.0f;
        for (int64_t row = 4; row < n_embd; ++row) {
            const float wave = std::sin((float) ((row + 1)*(slot + 3))*0.0017f);
            raw[slot*n_embd + row] = std::ldexp(wave, (int) ((row + 5*slot) % 13) - 6);
        }
    }

    std::vector<float> products(n_slots*n_embd);
    std::vector<float> expected(n_embd);
    for (int64_t slot = 0; slot < n_slots; ++slot) {
        for (int64_t row = 0; row < n_embd; ++row) {
            products[slot*n_embd + row] = rounded_product(raw[slot*n_embd + row], weights[slot]);
        }
    }
    for (int64_t row = 0; row < n_embd; ++row) {
        float acc = products[row];
        for (int64_t slot = 1; slot < n_slots; ++slot) {
            acc = rounded_add(acc, products[slot*n_embd + row]);
        }
        expected[row] = acc;
    }

    std::vector<float> cold(n_slots*n_embd);
    std::vector<float> hot(n_slots*n_embd);
    std::vector<float> scalar(n_embd);
    std::vector<float> automatic(n_embd);
    std::vector<float> avx512(n_embd);
    std::vector<float> grouped(n_embd);
    int grouped_mismatches = 0;

    for (uint32_t mask_bits = 0; mask_bits < (1u << n_slots); ++mask_bits) {
        std::fill(cold.begin(), cold.end(), std::numeric_limits<float>::quiet_NaN());
        std::fill(hot.begin(), hot.end(), std::numeric_limits<float>::quiet_NaN());
        uint8_t hot_mask[n_slots];

        for (int64_t slot = 0; slot < n_slots; ++slot) {
            hot_mask[slot] = (mask_bits >> slot) & 1u;
            for (int64_t row = 0; row < n_embd; ++row) {
                if (hot_mask[slot]) {
                    hot[slot*n_embd + row] = raw[slot*n_embd + row];
                } else {
                    cold[slot*n_embd + row] = products[slot*n_embd + row];
                }
            }
        }

        llama_hot_expert_merge_slots_f32_scalar(
            scalar.data(), cold.data(), n_embd, hot.data(), n_embd,
            weights, hot_mask, n_embd, n_slots);
        llama_hot_expert_merge_slots_f32(
            automatic.data(), cold.data(), n_embd, hot.data(), n_embd,
            weights, hot_mask, n_embd, n_slots);
        if (std::memcmp(scalar.data(), expected.data(), n_embd*sizeof(float)) != 0 ||
            std::memcmp(automatic.data(), expected.data(), n_embd*sizeof(float)) != 0) {
            std::fprintf(stderr, "scalar/auto slot-order mismatch for hot mask 0x%02x\n", mask_bits);
            return 1;
        }

        if (avx512_supported) {
            if (!llama_hot_expert_merge_slots_f32_avx512(
                    avx512.data(), cold.data(), n_embd, hot.data(), n_embd,
                    weights, hot_mask, n_embd, n_slots) ||
                std::memcmp(avx512.data(), expected.data(), n_embd*sizeof(float)) != 0) {
                std::fprintf(stderr, "AVX512 slot-order mismatch for hot mask 0x%02x\n", mask_bits);
                return 1;
            }
        }

        for (int64_t row = 0; row < n_embd; ++row) {
            float cold_partial = hot_mask[0] ? 0.0f : products[row];
            for (int64_t slot = 1; slot < n_slots; ++slot) {
                cold_partial = rounded_add(cold_partial, hot_mask[slot] ? 0.0f : products[slot*n_embd + row]);
            }
            float hot_partial = 0.0f;
            for (int64_t slot = 0; slot < n_slots; ++slot) {
                if (hot_mask[slot]) {
                    hot_partial = rounded_add(hot_partial, products[slot*n_embd + row]);
                }
            }
            grouped[row] = rounded_add(cold_partial, hot_partial);
            grouped_mismatches += std::memcmp(&grouped[row], &expected[row], sizeof(float)) != 0;
        }
    }

    if (grouped_mismatches == 0) {
        std::fprintf(stderr, "adversarial inputs did not detect legacy partial reassociation\n");
        return 1;
    }

    const uint32_t bench_mask_bits = 0x2a;
    uint8_t bench_mask[n_slots];
    for (int64_t slot = 0; slot < n_slots; ++slot) {
        bench_mask[slot] = (bench_mask_bits >> slot) & 1u;
        for (int64_t row = 0; row < n_embd; ++row) {
            cold[slot*n_embd + row] = bench_mask[slot] ? std::numeric_limits<float>::quiet_NaN() : products[slot*n_embd + row];
            hot[slot*n_embd + row] = bench_mask[slot] ? raw[slot*n_embd + row] : std::numeric_limits<float>::quiet_NaN();
        }
    }

    constexpr int iterations = 2000;
    auto time_path = [&](bool use_avx512) {
        const auto begin = std::chrono::steady_clock::now();
        for (int iteration = 0; iteration < iterations; ++iteration) {
            if (use_avx512) {
                if (!llama_hot_expert_merge_slots_f32_avx512(
                        avx512.data(), cold.data(), n_embd, hot.data(), n_embd,
                        weights, bench_mask, n_embd, n_slots)) {
                    std::abort();
                }
            } else {
                llama_hot_expert_merge_slots_f32_scalar(
                    scalar.data(), cold.data(), n_embd, hot.data(), n_embd,
                    weights, bench_mask, n_embd, n_slots);
            }
        }
        const auto end = std::chrono::steady_clock::now();
        const double elapsed_ns = std::chrono::duration<double, std::nano>(end - begin).count();
        return elapsed_ns/iterations;
    };

    std::vector<double> scalar_ns;
    std::vector<double> avx512_ns;
    if (avx512_supported) {
        for (int warmup = 0; warmup < 32; ++warmup) {
            (void) llama_hot_expert_merge_slots_f32_avx512(
                avx512.data(), cold.data(), n_embd, hot.data(), n_embd,
                weights, bench_mask, n_embd, n_slots);
            llama_hot_expert_merge_slots_f32_scalar(
                scalar.data(), cold.data(), n_embd, hot.data(), n_embd,
                weights, bench_mask, n_embd, n_slots);
        }
        for (int cycle = 0; cycle < 4; ++cycle) {
            scalar_ns.push_back(time_path(false));
            avx512_ns.push_back(time_path(true));
            avx512_ns.push_back(time_path(true));
            scalar_ns.push_back(time_path(false));
        }
        if (std::memcmp(scalar.data(), avx512.data(), n_embd*sizeof(float)) != 0) {
            std::fprintf(stderr, "microbenchmark scalar/AVX512 output mismatch\n");
            return 1;
        }
        const double scalar_mean = mean(scalar_ns);
        const double avx512_mean = mean(avx512_ns);
        std::printf("merge4096x6 scalar mean/median=%.1f/%.1f ns layer43=%.4f ms | "
                    "avx512 mean/median=%.1f/%.1f ns layer43=%.4f ms speedup=%.2fx\n",
                    scalar_mean, median(scalar_ns), scalar_mean*43.0/1.0e6,
                    avx512_mean, median(avx512_ns), avx512_mean*43.0/1.0e6,
                    scalar_mean/avx512_mean);
    } else {
        std::printf("merge4096x6 AVX512 unsupported; scalar-only correctness path\n");
    }

    std::printf("hot-expert slot-order: enabled=%d avx512=%d masks=64 legacy_mismatches=%d PASS\n",
                llama_hot_expert_slot_order_enabled(), llama_hot_expert_slot_merge_avx512_enabled(), grouped_mismatches);
    return 0;
}
