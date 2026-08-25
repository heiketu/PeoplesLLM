#include "ggml-backend.h"
#include "ggml.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace {

uint64_t fnv1a(const void * data, size_t size) {
    const uint8_t * bytes = (const uint8_t *) data;
    uint64_t hash = 1469598103934665603ULL;
    for (size_t i = 0; i < size; ++i) {
        hash ^= bytes[i];
        hash *= 1099511628211ULL;
    }
    return hash;
}

std::vector<uint8_t> make_mxfp4(size_t bytes, uint32_t salt) {
    std::vector<uint8_t> data(bytes);
    for (size_t block = 0; block < bytes / 17; ++block) {
        uint8_t * b = data.data() + block * 17;
        b[0] = (uint8_t) (120 + (block + salt) % 12);
        for (size_t q = 0; q < 16; ++q) {
            b[1 + q] = (uint8_t) (block * 131 + q * 29 + salt);
        }
    }
    return data;
}

bool run_case(ggml_backend_t backend, int64_t n_tokens, bool benchmark) {
    constexpr int64_t n_embd = 4096;
    constexpr int64_t n_ff = 2048;
    constexpr int64_t n_used = 6;
    const int64_t n_experts = benchmark ? 24 : 4;

    ggml_context * ctx = ggml_init({32 * 1024 * 1024, nullptr, true});
    if (ctx == nullptr) {
        return false;
    }
    ggml_tensor * wg = ggml_new_tensor_3d(ctx, GGML_TYPE_MXFP4, n_embd, n_ff, n_experts);
    ggml_tensor * wu = ggml_new_tensor_3d(ctx, GGML_TYPE_MXFP4, n_embd, n_ff, n_experts);
    ggml_tensor * wd = ggml_new_tensor_3d(ctx, GGML_TYPE_MXFP4, n_ff, n_embd, n_experts);
    ggml_tensor * input = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_embd, 1, n_tokens);
    ggml_tensor * ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, n_used, n_tokens);

    ggml_tensor * gate = ggml_mul_mat_id(ctx, wg, input, ids);
    ggml_tensor * up = ggml_mul_mat_id(ctx, wu, input, ids);
    up = ggml_clamp(ctx, up, -10.0f, 10.0f);
    gate = ggml_clamp(ctx, gate, -INFINITY, 10.0f);
    ggml_tensor * act = ggml_swiglu_split(ctx, gate, up);
    ggml_tensor * output = ggml_mul_mat_id(ctx, wd, act, ids);

    ggml_cgraph * graph = ggml_new_graph_custom(ctx, 32, false);
    ggml_build_forward_expand(graph, output);
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (buffer == nullptr) {
        ggml_free(ctx);
        return false;
    }

    std::vector<uint8_t> gu = make_mxfp4(ggml_nbytes(wg), 17);
    std::vector<uint8_t> down = make_mxfp4(ggml_nbytes(wd), 91);
    ggml_backend_tensor_set(wg, gu.data(), 0, gu.size());
    ggml_backend_tensor_set(wu, gu.data(), 0, gu.size());
    ggml_backend_tensor_set(wd, down.data(), 0, down.size());

    std::vector<float> input_values((size_t) n_embd * n_tokens);
    for (size_t i = 0; i < input_values.size(); ++i) {
        input_values[i] = std::sin((float) (i + 3) * 0.0017f) * 0.125f;
    }
    std::vector<int32_t> id_values((size_t) n_used * n_tokens);
    for (int64_t token = 0; token < n_tokens; ++token) {
        for (int64_t slot = 0; slot < n_used; ++slot) {
            id_values[(size_t) token * n_used + slot] = (int32_t) ((token * 3 + slot) % n_experts);
        }
    }
    ggml_backend_tensor_set(input, input_values.data(), 0, input_values.size() * sizeof(float));
    ggml_backend_tensor_set(ids, id_values.data(), 0, id_values.size() * sizeof(int32_t));

    bool ok = ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS;
    std::vector<float> result((size_t) n_embd * n_used * n_tokens);
    ggml_backend_tensor_get(output, result.data(), 0, result.size() * sizeof(float));
    const uint64_t hash = fnv1a(result.data(), result.size() * sizeof(float));
    uint64_t expected_hash = 0;
    if (!benchmark) {
        switch (n_tokens) {
            case 1: expected_hash = 0xb04343b491a8a082ULL; break;
            case 2: expected_hash = 0x83d033b3e6e37d24ULL; break;
            case 4: expected_hash = 0xa07953927f9c340fULL; break;
            default: break;
        }
        ok &= hash == expected_hash;
    }

    double usec = 0.0;
    if (ok && benchmark) {
        constexpr int warmup = 12;
        constexpr int iterations = 80;
        for (int i = 0; i < warmup; ++i) {
            ok &= ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS;
        }
        const auto begin = std::chrono::steady_clock::now();
        for (int i = 0; i < iterations; ++i) {
            ok &= ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS;
        }
        ggml_backend_synchronize(backend);
        usec = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - begin).count() / iterations;
    }

    std::printf("MXFP4 clamped hot FFN tokens=%lld experts=%lld hash=%016llx expected=%016llx time_us=%.3f %s\n",
        (long long) n_tokens, (long long) n_experts, (unsigned long long) hash,
        (unsigned long long) expected_hash, usec, ok ? "PASS" : "FAIL");
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    return ok;
}

} // namespace

int main(int argc, char ** argv) {
    const bool benchmark = argc == 2 && std::string(argv[1]) == "--bench";
    if (argc > 1 && !benchmark) {
        std::fprintf(stderr, "usage: %s [--bench]\n", argv[0]);
        return 2;
    }
    ggml_backend_dev_t device = ggml_backend_dev_by_name("CUDA0");
    if (device == nullptr) {
        std::printf("test-cuda-mxfp4-fused-clamp: SKIPPED, CUDA0 unavailable\n");
        return 0;
    }
    ggml_backend_t backend = ggml_backend_dev_init(device, nullptr);
    if (backend == nullptr) {
        return 1;
    }
    bool ok = run_case(backend, 1, benchmark);
    ok &= run_case(backend, 2, benchmark);
    ok &= run_case(backend, 4, benchmark);
    ggml_backend_free(backend);
    return ok ? 0 : 1;
}
