#include "ggml.h"
#include "quants.h"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <vector>

struct cuda_block_q8_1 {
    uint32_t ds;
    int8_t qs[QK8_0];
};
static_assert(sizeof(cuda_block_q8_1) == 2 * sizeof(ggml_fp16_t) + QK8_0);

struct upe_activation_sample {
    int32_t layer;
    float values[QK8_0];
};
static_assert(sizeof(upe_activation_sample) == sizeof(int32_t) + QK8_0 * sizeof(float));

void quantize_row_q8_1_cuda(
        const float * x, const int32_t * ids, void * vy,
        ggml_type type_src0, int64_t ne00, int64_t s01, int64_t s02, int64_t s03,
        int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3, bool cpu_compat, cudaStream_t stream);

namespace {

bool cpu_q8_compat_enabled() {
    const char * value = std::getenv("GGML_CUDA_MXFP4_CPU_Q8");
    return value != nullptr && value[0] != '\0' && std::strcmp(value, "0") != 0;
}

void cuda_check(cudaError_t status, const char * what) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(status));
        std::exit(1);
    }
}

uint64_t fnv1a64(const int8_t * data, size_t size) {
    uint64_t hash = UINT64_C(1469598103934665603);
    for (size_t i = 0; i < size; ++i) {
        hash ^= (uint8_t) data[i];
        hash *= UINT64_C(1099511628211);
    }
    return hash;
}

std::vector<float> make_input(int64_t ncols, bool adversarial) {
    std::vector<float> input((size_t) ncols);
    if (!adversarial) {
        for (size_t i = 0; i < input.size(); ++i) {
            input[i] = std::sin((float) (i + 3) * 0.0017f) * 0.125f;
        }
        return input;
    }
    for (int64_t block = 0; block < ncols / QK8_0; ++block) {
        float * row = input.data() + block * QK8_0;
        const float amax = 31.0f + (float) (block % 97);
        row[0] = amax;
        for (int lane = 1; lane < QK8_0; ++lane) {
            const float q_half = (float) ((lane % 15) - 7) + 0.5f;
            row[lane] = q_half * amax / 127.0f;
        }
    }
    return input;
}

bool run_case(bool adversarial) {
    constexpr int64_t ncols = 4096;
    constexpr int64_t nblocks = ncols / QK8_0;
    const std::vector<float> input = make_input(ncols, adversarial);

    std::vector<block_q8_0> cpu((size_t) nblocks);
    quantize_row_q8_0(input.data(), cpu.data(), ncols);

    float * device_input = nullptr;
    cuda_block_q8_1 * device_q8 = nullptr;
    cuda_check(cudaMalloc((void **) &device_input, input.size() * sizeof(float)), "cudaMalloc input");
    cuda_check(cudaMalloc((void **) &device_q8, (size_t) nblocks * sizeof(cuda_block_q8_1)), "cudaMalloc q8");
    cuda_check(cudaMemcpy(device_input, input.data(), input.size() * sizeof(float), cudaMemcpyHostToDevice),
        "copy input");
    quantize_row_q8_1_cuda(device_input, nullptr, device_q8, GGML_TYPE_MXFP4,
        ncols, ncols, ncols, ncols, ncols, 1, 1, 1, cpu_q8_compat_enabled(), nullptr);
    cuda_check(cudaGetLastError(), "quantize_row_q8_1_cuda launch");
    cuda_check(cudaDeviceSynchronize(), "quantize_row_q8_1_cuda sync");

    std::vector<cuda_block_q8_1> gpu((size_t) nblocks);
    cuda_check(cudaMemcpy(gpu.data(), device_q8, gpu.size() * sizeof(cuda_block_q8_1), cudaMemcpyDeviceToHost),
        "copy q8");
    cudaFree(device_q8);
    cudaFree(device_input);

    size_t cpu_gpu_code_mismatch = 0;
    size_t gpu_default_formula_mismatch = 0;
    size_t gpu_cpu_formula_mismatch = 0;
    size_t scale_mismatch = 0;
    double max_sum_error = 0.0;
    std::vector<int8_t> cpu_codes((size_t) ncols);
    std::vector<int8_t> gpu_codes((size_t) ncols);

    for (int64_t block = 0; block < nblocks; ++block) {
        const float * x = input.data() + block * QK8_0;
        float amax = 0.0f;
        float sum = 0.0f;
        for (int lane = 0; lane < QK8_0; ++lane) {
            amax = std::max(amax, std::abs(x[lane]));
            sum += x[lane];
        }
        const float d = amax / 127.0f;
        const float id = amax == 0.0f ? 0.0f : 127.0f / amax;
        const float gpu_d = ggml_fp16_to_fp32((ggml_fp16_t) (gpu[(size_t) block].ds & 0xffffu));
        const float gpu_sum = ggml_fp16_to_fp32((ggml_fp16_t) (gpu[(size_t) block].ds >> 16));
        const float cpu_d = ggml_fp16_to_fp32(cpu[(size_t) block].d);
        scale_mismatch += gpu_d != cpu_d;
        max_sum_error = std::max(max_sum_error, std::abs((double) gpu_sum - sum));

        for (int lane = 0; lane < QK8_0; ++lane) {
            const size_t index = (size_t) block * QK8_0 + lane;
            const int8_t gpu_code = gpu[(size_t) block].qs[lane];
            const int8_t cpu_code = cpu[(size_t) block].qs[lane];
            const int8_t default_formula = amax == 0.0f ? 0 : (int8_t) std::round(x[lane] / d);
            const int8_t cpu_formula = (int8_t) std::nearbyint(x[lane] * id);
            cpu_codes[index] = cpu_code;
            gpu_codes[index] = gpu_code;
            cpu_gpu_code_mismatch += cpu_code != gpu_code;
            gpu_default_formula_mismatch += default_formula != gpu_code;
            gpu_cpu_formula_mismatch += cpu_formula != gpu_code;
        }
    }

    std::printf(
        "CUDA-Q8 contract input=%s cpu_compat=%d values=%lld cpu-gpu-code-mismatch=%zu "
        "gpu-default-formula-mismatch=%zu gpu-cpu-formula-mismatch=%zu "
        "scale-mismatch=%zu max-sum-error=%.9g cpu-hash=%016llx gpu-hash=%016llx\n",
        adversarial ? "q8-half" : "smooth", cpu_q8_compat_enabled(), (long long) ncols,
        cpu_gpu_code_mismatch, gpu_default_formula_mismatch, gpu_cpu_formula_mismatch,
        scale_mismatch, max_sum_error,
        (unsigned long long) fnv1a64(cpu_codes.data(), cpu_codes.size()),
        (unsigned long long) fnv1a64(gpu_codes.data(), gpu_codes.size()));

    // Smooth inputs must already agree exactly. Half-step inputs intentionally
    // expose the discrete contract boundary and are reported, not normalized
    // into a hardware-specific expected mismatch count.
    return scale_mismatch == 0 && std::isfinite(max_sum_error) &&
        (cpu_q8_compat_enabled() ? cpu_gpu_code_mismatch == 0 : adversarial || cpu_gpu_code_mismatch == 0);
}

bool run_sample_file(const char * path) {
    FILE * file = std::fopen(path, "rb");
    if (file == nullptr) {
        std::fprintf(stderr, "cannot open UPE activation sample file: %s\n", path);
        return false;
    }
    uint32_t header[4] = {};
    if (std::fread(header, sizeof(header), 1, file) != 1 ||
            header[0] != UINT32_C(0x31504555) || header[1] != 1 || header[2] != QK8_0 ||
            header[3] == 0 || header[3] > 2097152) {
        std::fprintf(stderr, "invalid UPE activation sample header: %s\n", path);
        std::fclose(file);
        return false;
    }
    std::vector<upe_activation_sample> samples(header[3]);
    const bool read_ok = std::fread(samples.data(), sizeof(samples[0]), samples.size(), file) == samples.size();
    const int trailing = std::fgetc(file);
    std::fclose(file);
    if (!read_ok || trailing != EOF) {
        std::fprintf(stderr, "invalid UPE activation sample payload: %s\n", path);
        return false;
    }

    std::vector<float> input(samples.size() * QK8_0);
    for (size_t i = 0; i < samples.size(); ++i) {
        std::memcpy(input.data() + i * QK8_0, samples[i].values, QK8_0 * sizeof(float));
    }
    std::vector<block_q8_0> cpu(samples.size());
    quantize_row_q8_0(input.data(), cpu.data(), input.size());

    constexpr size_t cuda_rows_per_launch = 65535;
    const size_t device_rows = std::min(cuda_rows_per_launch, samples.size());
    float * device_input = nullptr;
    cuda_block_q8_1 * device_q8 = nullptr;
    cuda_check(cudaMalloc((void **) &device_input, device_rows * QK8_0 * sizeof(float)),
        "sample cudaMalloc input");
    cuda_check(cudaMalloc((void **) &device_q8, device_rows * sizeof(cuda_block_q8_1)),
        "sample cudaMalloc q8");
    std::vector<cuda_block_q8_1> gpu(samples.size());
    for (size_t offset = 0; offset < samples.size(); offset += cuda_rows_per_launch) {
        const size_t rows = std::min(cuda_rows_per_launch, samples.size() - offset);
        cuda_check(cudaMemcpy(device_input, input.data() + offset * QK8_0,
            rows * QK8_0 * sizeof(float), cudaMemcpyHostToDevice), "sample copy input");
        quantize_row_q8_1_cuda(device_input, nullptr, device_q8, GGML_TYPE_MXFP4,
            QK8_0, QK8_0, (int64_t) rows * QK8_0, (int64_t) rows * QK8_0,
            QK8_0, rows, 1, 1, cpu_q8_compat_enabled(), nullptr);
        cuda_check(cudaGetLastError(), "sample quantize launch");
        cuda_check(cudaDeviceSynchronize(), "sample quantize sync");
        cuda_check(cudaMemcpy(gpu.data() + offset, device_q8, rows * sizeof(gpu[0]), cudaMemcpyDeviceToHost),
            "sample copy q8");
    }
    cudaFree(device_q8);
    cudaFree(device_input);

    size_t code_mismatch = 0;
    size_t nominal_mismatch = 0;
    size_t mismatch_intersection = 0;
    size_t actual_only = 0;
    size_t nominal_only = 0;
    size_t scale_mismatch = 0;
    double max_mismatch_half_distance = 0.0;
    static constexpr double thresholds[] = {1.0e-6, 2.0e-6, 4.0e-6, 8.0e-6, 1.6e-5};
    size_t predicate_true[sizeof(thresholds) / sizeof(thresholds[0])] = {};
    size_t predicate_false_positive[sizeof(thresholds) / sizeof(thresholds[0])] = {};
    size_t predicate_false_negative[sizeof(thresholds) / sizeof(thresholds[0])] = {};
    std::map<int32_t, std::pair<size_t, size_t>> per_layer;
    for (size_t block = 0; block < samples.size(); ++block) {
        float amax = 0.0f;
        for (int lane = 0; lane < QK8_0; ++lane) {
            amax = std::max(amax, std::abs(samples[block].values[lane]));
        }
        const float id = amax == 0.0f ? 0.0f : 127.0f / amax;
        const float d = amax / 127.0f;
        const float gpu_d = ggml_fp16_to_fp32((ggml_fp16_t) (gpu[block].ds & 0xffffu));
        const float cpu_d = ggml_fp16_to_fp32(cpu[block].d);
        scale_mismatch += gpu_d != cpu_d;
        auto & counts = per_layer[samples[block].layer];
        counts.second += QK8_0;
        for (int lane = 0; lane < QK8_0; ++lane) {
            const bool mismatch = cpu[block].qs[lane] != gpu[block].qs[lane];
            const int8_t nominal_code = amax == 0.0f ? 0 :
                (int8_t) std::round(samples[block].values[lane] / d);
            const bool nominal = cpu[block].qs[lane] != nominal_code;
            code_mismatch += mismatch;
            nominal_mismatch += nominal;
            mismatch_intersection += mismatch && nominal;
            actual_only += mismatch && !nominal;
            nominal_only += !mismatch && nominal;
            counts.first += mismatch;
            if (mismatch) {
                const float normalized = samples[block].values[lane] * id;
                const float fraction = normalized - std::floor(normalized);
                max_mismatch_half_distance = std::max(
                    max_mismatch_half_distance, (double) std::abs(fraction - 0.5f));
            }
            const float normalized = samples[block].values[lane] * id;
            const float fraction = normalized - std::floor(normalized);
            const double half_distance = std::abs((double) fraction - 0.5);
            for (size_t threshold = 0; threshold < sizeof(thresholds) / sizeof(thresholds[0]); ++threshold) {
                const bool predicted = nominal || half_distance <= thresholds[threshold];
                predicate_true[threshold] += predicted;
                predicate_false_positive[threshold] += predicted && !mismatch;
                predicate_false_negative[threshold] += !predicted && mismatch;
            }
        }
    }
    std::printf("CUDA-Q8 real-samples cpu_compat=%d blocks=%zu values=%zu cpu-gpu-code-mismatch=%zu rate=%.9g "
                "nominal-mismatch=%zu intersection=%zu actual-only=%zu nominal-only=%zu "
                "scale-mismatch=%zu max-mismatch-half-distance=%.9g\n",
        cpu_q8_compat_enabled(), samples.size(), input.size(), code_mismatch,
        (double) code_mismatch / input.size(), nominal_mismatch, mismatch_intersection,
        actual_only, nominal_only, scale_mismatch, max_mismatch_half_distance);
    for (const auto & item : per_layer) {
        if (item.second.first != 0) {
            std::printf("CUDA-Q8 real-layer=%d code-mismatch=%zu values=%zu rate=%.9g\n",
                item.first, item.second.first, item.second.second,
                (double) item.second.first / item.second.second);
        }
    }
    for (size_t threshold = 0; threshold < sizeof(thresholds) / sizeof(thresholds[0]); ++threshold) {
        std::printf("CUDA-Q8 boundary-predicate threshold=%.9g predicted=%zu false-positive=%zu false-negative=%zu\n",
            thresholds[threshold], predicate_true[threshold], predicate_false_positive[threshold],
            predicate_false_negative[threshold]);
    }
    return scale_mismatch == 0 && (!cpu_q8_compat_enabled() || code_mismatch == 0);
}

} // namespace

int main(int argc, char ** argv) {
    int devices = 0;
    cudaError_t status = cudaGetDeviceCount(&devices);
    if (status != cudaSuccess || devices == 0) {
        std::puts("test-cuda-q8-contract: SKIPPED, CUDA device unavailable");
        return 0;
    }
    cuda_check(cudaSetDevice(0), "cudaSetDevice");
    bool ok = run_case(false) && run_case(true);
    if (argc == 2) {
        ok = run_sample_file(argv[1]) && ok;
    } else if (argc != 1) {
        std::fprintf(stderr, "usage: %s [upe-activation.blocks.bin]\n", argv[0]);
        return 2;
    }
    return ok ? 0 : 1;
}
