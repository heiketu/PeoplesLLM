#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-impl.h"
#include "ggml.h"
#include "quants.h"
#include "repack.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

bool env_enabled(const char * name) {
    const char * value = std::getenv(name);
    return value != nullptr && value[0] != '\0' && std::strcmp(value, "0") != 0;
}

bool cpu_q8_compat_enabled(bool hot_tensor) {
    return hot_tensor && env_enabled("GGML_CUDA_HOT_MXFP4_CPU_Q8");
}

struct output_stats {
    double max_abs = 0.0;
    double mae = 0.0;
    double rmse = 0.0;
    double relative_rmse = 0.0;
    double cosine = 0.0;
    size_t nonfinite_cpu = 0;
    size_t nonfinite_gpu = 0;
};

struct q8_formula_stats {
    size_t cuda_default_mismatches = 0;
    size_t cpu_compat_mismatches = 0;
    size_t values = 0;
};

q8_formula_stats compare_cpu_q8_formulas(const std::vector<float> & input) {
    GGML_ASSERT(input.size() % QK8_0 == 0);
    std::vector<block_q8_0> cpu(input.size() / QK8_0);
    quantize_row_q8_0(input.data(), cpu.data(), input.size());

    q8_formula_stats stats;
    stats.values = input.size();
    for (size_t ib = 0; ib < cpu.size(); ++ib) {
        const float * x = input.data() + ib * QK8_0;
        float amax = 0.0f;
        for (int lane = 0; lane < QK8_0; ++lane) {
            amax = std::max(amax, std::abs(x[lane]));
        }
        const float d = amax / 127.0f;
        const float id = amax == 0.0f ? 0.0f : 127.0f / amax;
        for (int lane = 0; lane < QK8_0; ++lane) {
            const int8_t cuda_default = amax == 0.0f ? 0 : (int8_t) std::round(x[lane] / d);
            const int8_t cpu_compat = (int8_t) std::nearbyint(x[lane] * id);
            stats.cuda_default_mismatches += cuda_default != cpu[ib].qs[lane];
            stats.cpu_compat_mismatches += cpu_compat != cpu[ib].qs[lane];
        }
    }
    return stats;
}

uint64_t fnv1a(const std::vector<float> & data) {
    const uint8_t * bytes = (const uint8_t *) data.data();
    const size_t size = data.size() * sizeof(float);
    uint64_t hash = 1469598103934665603ULL;
    for (size_t i = 0; i < size; ++i) {
        hash ^= bytes[i];
        hash *= 1099511628211ULL;
    }
    return hash;
}

output_stats compare(const std::vector<float> & cpu, const std::vector<float> & gpu) {
    output_stats stats;
    double sum_abs = 0.0;
    double sum_sq_err = 0.0;
    double sum_sq_cpu = 0.0;
    double sum_sq_gpu = 0.0;
    double dot = 0.0;
    for (size_t i = 0; i < cpu.size(); ++i) {
        const double cv = cpu[i];
        const double gv = gpu[i];
        stats.nonfinite_cpu += !std::isfinite(cv);
        stats.nonfinite_gpu += !std::isfinite(gv);
        if (!std::isfinite(cv) || !std::isfinite(gv)) {
            continue;
        }
        const double diff = gv - cv;
        stats.max_abs = std::max(stats.max_abs, std::abs(diff));
        sum_abs += std::abs(diff);
        sum_sq_err += diff * diff;
        sum_sq_cpu += cv * cv;
        sum_sq_gpu += gv * gv;
        dot += cv * gv;
    }
    stats.mae = sum_abs / cpu.size();
    stats.rmse = std::sqrt(sum_sq_err / cpu.size());
    stats.relative_rmse = sum_sq_cpu > 0.0 ? std::sqrt(sum_sq_err / sum_sq_cpu) : 0.0;
    stats.cosine = sum_sq_cpu > 0.0 && sum_sq_gpu > 0.0 ? dot / std::sqrt(sum_sq_cpu * sum_sq_gpu) : 1.0;
    return stats;
}

std::vector<uint8_t> make_weights(size_t bytes, int e8m0_edge = 0) {
    std::vector<uint8_t> data(bytes);
    for (size_t block = 0; block < bytes / 17; ++block) {
        uint8_t * b = data.data() + block * 17;
        if ((e8m0_edge & 1) && block % 128 == 0) {
            b[0] = 0xff;
        } else if ((e8m0_edge & 2) && block % 128 == 1) {
            b[0] = 250;
        } else {
            b[0] = (uint8_t) (120 + block % 12);
        }
        for (size_t q = 0; q < 16; ++q) {
            b[1 + q] = (uint8_t) (block * 131 + q * 29 + 17);
        }
    }
    return data;
}

std::vector<float> make_input(int64_t ncols, int64_t n_tokens, bool adversarial) {
    std::vector<float> input((size_t) ncols * n_tokens);
    if (!adversarial) {
        for (size_t i = 0; i < input.size(); ++i) {
            input[i] = std::sin((float) (i + 3) * 0.0017f) * 0.125f;
        }
        return input;
    }

    for (int64_t token = 0; token < n_tokens; ++token) {
        for (int64_t block = 0; block < ncols / 32; ++block) {
            float * row = input.data() + token * ncols + block * 32;
            const float amax = 31.0f + (float) ((token * 17 + block) % 97);
            row[0] = amax;
            for (int lane = 1; lane < 32; ++lane) {
                const float q_half = (float) ((lane % 15) - 7) + 0.5f;
                row[lane] = q_half * amax / 127.0f;
            }
        }
    }
    return input;
}

std::vector<int32_t> make_ids(int64_t n_used, int64_t n_tokens, int64_t n_experts) {
    std::vector<int32_t> ids((size_t) n_used * n_tokens);
    for (int64_t token = 0; token < n_tokens; ++token) {
        for (int64_t slot = 0; slot < n_used; ++slot) {
            ids[(size_t) token * n_used + slot] = (int32_t) ((token * 3 + slot) % n_experts);
        }
    }
    return ids;
}

std::vector<float> run_q8_reference(
        const std::vector<uint8_t> & weights, const std::vector<float> & input,
        const std::vector<int32_t> & ids, int64_t ncols, int64_t nrows,
        int64_t n_experts, int64_t n_used, int64_t n_tokens, bool cuda_default) {
    static constexpr int8_t mxfp4_values[16] = {
        0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12,
    };
    const int64_t blocks_per_row = ncols / QK_MXFP4;
    std::vector<block_q8_0> q8((size_t) n_tokens * blocks_per_row);
    for (int64_t token = 0; token < n_tokens; ++token) {
        if (!cuda_default) {
            quantize_row_q8_0(input.data() + token * ncols,
                q8.data() + token * blocks_per_row, ncols);
            continue;
        }
        for (int64_t block = 0; block < blocks_per_row; ++block) {
            const float * x = input.data() + token * ncols + block * QK8_0;
            float amax = 0.0f;
            for (int lane = 0; lane < QK8_0; ++lane) {
                amax = std::max(amax, std::abs(x[lane]));
            }
            const float d = amax / 127.0f;
            q8[token * blocks_per_row + block].d = ggml_fp32_to_fp16(d);
            for (int lane = 0; lane < QK8_0; ++lane) {
                q8[token * blocks_per_row + block].qs[lane] =
                    amax == 0.0f ? 0 : (int8_t) std::round(x[lane] / d);
            }
        }
    }

    const block_mxfp4 * all_weights = (const block_mxfp4 *) weights.data();
    std::vector<float> result((size_t) n_tokens * n_used * nrows);
    for (int64_t token = 0; token < n_tokens; ++token) {
        const block_q8_0 * activation = q8.data() + token * blocks_per_row;
        for (int64_t slot = 0; slot < n_used; ++slot) {
            const int64_t expert = ids[token * n_used + slot];
            GGML_ASSERT(expert >= 0 && expert < n_experts);
            for (int64_t row = 0; row < nrows; ++row) {
                const block_mxfp4 * weight = all_weights + (expert * nrows + row) * blocks_per_row;
                float sum = 0.0f;
                for (int64_t block = 0; block < blocks_per_row; ++block) {
                    int32_t isum = 0;
                    for (int lane = 0; lane < QK_MXFP4 / 2; ++lane) {
                        const uint8_t packed = weight[block].qs[lane];
                        isum += activation[block].qs[lane] * mxfp4_values[packed & 0x0f];
                        isum += activation[block].qs[lane + QK_MXFP4 / 2] * mxfp4_values[packed >> 4];
                    }
                    const float scale = GGML_E8M0_TO_FP32_HALF(weight[block].e) *
                        ggml_fp16_to_fp32(activation[block].d);
                    sum = std::fma((float) isum, scale, sum);
                }
                result[((size_t) token * n_used + slot) * nrows + row] = sum;
            }
        }
    }
    return result;
}

std::vector<float> run_cpu(
        ggml_backend_t backend, const std::vector<uint8_t> & weights,
        const std::vector<float> & input, const std::vector<int32_t> & ids,
        int64_t ncols, int64_t nrows, int64_t n_experts, int64_t n_used, int64_t n_tokens) {
    ggml_context * ctx = ggml_init({4 * 1024 * 1024, nullptr, true});
    ggml_tensor * w = ggml_new_tensor_3d(ctx, GGML_TYPE_MXFP4, ncols, nrows, n_experts);
    ggml_backend_buffer_t wbuf = ggml_backend_buft_alloc_buffer(
        ggml_backend_cpu_repack_buffer_type(), ggml_nbytes(w));
    if (wbuf == nullptr) {
        std::fprintf(stderr, "CPU_REPACK allocation failed\n");
        std::exit(1);
    }
    ggml_backend_tensor_alloc(wbuf, w, ggml_backend_buffer_get_base(wbuf));
    if (w->extra == nullptr) {
        std::fprintf(stderr, "MXFP4 CPU_REPACK trait unavailable\n");
        std::exit(1);
    }
    ggml_backend_tensor_set(w, weights.data(), 0, weights.size());

    ggml_tensor * x = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, ncols, 1, n_tokens);
    ggml_tensor * id = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, n_used, n_tokens);
    ggml_tensor * out = ggml_mul_mat_id(ctx, w, x, id);
    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, out);
    ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_cpu_buffer_type());
    if (alloc == nullptr || !ggml_gallocr_alloc_graph(alloc, graph)) {
        std::fprintf(stderr, "CPU graph allocation failed\n");
        std::exit(1);
    }
    ggml_backend_tensor_set(x, input.data(), 0, input.size() * sizeof(float));
    ggml_backend_tensor_set(id, ids.data(), 0, ids.size() * sizeof(int32_t));
    if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "CPU graph compute failed\n");
        std::exit(1);
    }
    std::vector<float> result((size_t) nrows * n_used * n_tokens);
    ggml_backend_tensor_get(out, result.data(), 0, result.size() * sizeof(float));
    ggml_gallocr_free(alloc);
    ggml_backend_buffer_free(wbuf);
    ggml_free(ctx);
    return result;
}

std::vector<float> run_cuda(
        ggml_backend_t backend, const std::vector<uint8_t> & weights,
        const std::vector<float> & input, const std::vector<int32_t> & ids,
        int64_t ncols, int64_t nrows, int64_t n_experts, int64_t n_used, int64_t n_tokens,
        bool hot_tensor) {
    ggml_context * ctx = ggml_init({4 * 1024 * 1024, nullptr, true});
    ggml_tensor * w = ggml_new_tensor_3d(ctx, GGML_TYPE_MXFP4, ncols, nrows, n_experts);
    ggml_set_name(w, hot_tensor ? "hot_expert.test" : "regular.test");
    ggml_tensor * x = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, ncols, 1, n_tokens);
    ggml_tensor * id = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, n_used, n_tokens);
    ggml_tensor * out = ggml_mul_mat_id(ctx, w, x, id);
    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, out);
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (buffer == nullptr) {
        std::fprintf(stderr, "CUDA graph allocation failed\n");
        std::exit(1);
    }
    ggml_backend_tensor_set(w, weights.data(), 0, weights.size());
    ggml_backend_tensor_set(x, input.data(), 0, input.size() * sizeof(float));
    ggml_backend_tensor_set(id, ids.data(), 0, ids.size() * sizeof(int32_t));
    if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "CUDA graph compute failed\n");
        std::exit(1);
    }
    std::vector<float> result((size_t) nrows * n_used * n_tokens);
    ggml_backend_tensor_get(out, result.data(), 0, result.size() * sizeof(float));
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    return result;
}

bool run_case(
        ggml_backend_t cpu, ggml_backend_t cuda, int64_t n_tokens, bool adversarial,
        int e8m0_edge = 0, bool hot_tensor = false) {
    constexpr int64_t ncols = 4096;
    constexpr int64_t nrows = 256;
    constexpr int64_t n_experts = 4;
    constexpr int64_t n_used = 6;
    const size_t weight_bytes = (size_t) ncols * nrows * n_experts * 17 / 32;
    const std::vector<uint8_t> weights = make_weights(weight_bytes, e8m0_edge);
    const std::vector<float> input = make_input(ncols, n_tokens, adversarial);
    const std::vector<int32_t> ids = make_ids(n_used, n_tokens, n_experts);
    const q8_formula_stats q8_stats = compare_cpu_q8_formulas(input);

    const std::vector<float> cpu_out = run_cpu(
        cpu, weights, input, ids, ncols, nrows, n_experts, n_used, n_tokens);
    const std::vector<float> gpu_out = run_cuda(
        cuda, weights, input, ids, ncols, nrows, n_experts, n_used, n_tokens, hot_tensor);
    const std::vector<float> cpu_q8_reference = run_q8_reference(
        weights, input, ids, ncols, nrows, n_experts, n_used, n_tokens, false);
    const std::vector<float> cuda_q8_reference = run_q8_reference(
        weights, input, ids, ncols, nrows, n_experts, n_used, n_tokens, !cpu_q8_compat_enabled(hot_tensor));
    const output_stats stats = compare(cpu_out, gpu_out);
    const output_stats cpu_reference = compare(cpu_q8_reference, cpu_out);
    const output_stats gpu_reference = compare(cuda_q8_reference, gpu_out);
    const output_stats reference_gap = compare(cpu_q8_reference, cuda_q8_reference);
    const bool default_ok = e8m0_edge != 0 ?
        (stats.nonfinite_cpu == 0 && stats.nonfinite_gpu == 0 && stats.max_abs == 0.0) :
        (std::isfinite(stats.relative_rmse) && std::isfinite(stats.cosine) &&
         stats.relative_rmse < 0.10 && stats.cosine > 0.99 && cpu_reference.relative_rmse < 1.0e-6);
    const bool compat_ok = !cpu_q8_compat_enabled(hot_tensor) || e8m0_edge != 0 ||
        (stats.relative_rmse < 1.0e-5 && stats.cosine > 0.999999 && reference_gap.relative_rmse == 0.0);
    const bool ok = default_ok && compat_ok;
    std::printf("CPU_REPACK-vs-CUDA MXFP4 cpu_compat=%d tokens=%lld input=%s weights=%s max_abs=%.9g mae=%.9g rmse=%.9g rel_rmse=%.9g cosine=%.9g nonfinite_cpu=%zu nonfinite_gpu=%zu cpu_hash=%016llx gpu_hash=%016llx %s\n",
        cpu_q8_compat_enabled(hot_tensor),
        (long long) n_tokens, adversarial ? "q8-half" : "smooth",
        e8m0_edge == 1 ? "e8m0-nan" : e8m0_edge == 2 ? "e8m0-high" : e8m0_edge == 3 ? "e8m0-both" : "normal",
        stats.max_abs, stats.mae, stats.rmse, stats.relative_rmse, stats.cosine,
        stats.nonfinite_cpu, stats.nonfinite_gpu,
        (unsigned long long) fnv1a(cpu_out), (unsigned long long) fnv1a(gpu_out),
        ok ? "PASS" : "FAIL");
    std::printf("CPU-Q8 formula tokens=%lld input=%s values=%zu cuda-default-mismatch=%zu cpu-compat-mismatch=%zu\n",
        (long long) n_tokens, adversarial ? "q8-half" : "smooth", q8_stats.values,
        q8_stats.cuda_default_mismatches, q8_stats.cpu_compat_mismatches);
    std::printf("Q8 references tokens=%lld input=%s cpu-vs-cpu-ref=%.9g gpu-vs-cuda-ref=%.9g ref-gap=%.9g cpu-ref-cosine=%.9g gpu-ref-cosine=%.9g\n",
        (long long) n_tokens, adversarial ? "q8-half" : "smooth",
        cpu_reference.relative_rmse, gpu_reference.relative_rmse,
        reference_gap.relative_rmse, cpu_reference.cosine, gpu_reference.cosine);
    return ok;
}

} // namespace

int main() {
    ggml_cpu_init();
    ggml_backend_t cpu = ggml_backend_cpu_init();
    ggml_backend_dev_t cuda_dev = ggml_backend_dev_by_name("CUDA0");
    if (cpu == nullptr || cuda_dev == nullptr) {
        std::printf("test-cpu-cuda-mxfp4-parity: SKIPPED, required backend unavailable\n");
        if (cpu != nullptr) {
            ggml_backend_free(cpu);
        }
        return 0;
    }
    ggml_backend_t cuda = ggml_backend_dev_init(cuda_dev, nullptr);
    if (cuda == nullptr) {
        ggml_backend_free(cpu);
        return 1;
    }
    bool ok = true;
    const bool hot_tensor = env_enabled("GGML_CUDA_HOT_MXFP4_CPU_Q8");
    for (int64_t tokens : {1LL, 2LL, 4LL}) {
        ok &= run_case(cpu, cuda, tokens, false, 0, hot_tensor);
        ok &= run_case(cpu, cuda, tokens, true, 0, hot_tensor);
    }
    ok &= run_case(cpu, cuda, 1, false, 1, hot_tensor);
    ok &= run_case(cpu, cuda, 1, false, 2, hot_tensor);
    ok &= run_case(cpu, cuda, 1, false, 3, hot_tensor);
    ggml_backend_free(cuda);
    ggml_backend_free(cpu);
    return ok ? 0 : 1;
}
