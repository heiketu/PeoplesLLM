// Correctness harness for the DSV4 q1 decode compact-gather path with
// quantized KV (LLAMA_DSV4_Q8_SPARSE_FA, default on).
//
// The gather path runs ggml_get_rows on the quantized compressed KV segment
// so only the selected rows are dequantized, instead of dequantizing the full
// cache and selecting afterwards. This test verifies on the CPU backend that
// both orders produce bit-identical results:
//
//   A: cast(get_rows(q8, idx), F32)      (gather, then dequant selected rows)
//   B: get_rows(cast(q8, F32), idx)      (dequant everything, then gather)
//
// usage: test-dsv4-gather-dequant

#include "ggml.h"
#include "ggml-cpu.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

static bool run_case(int64_t ne0, int64_t ne1, int64_t k, uint32_t seed) {
    std::mt19937 rng(seed);

    std::vector<float> src(ne0*ne1);
    {
        std::normal_distribution<float> dist(0.0f, 1.0f);
        for (auto & x : src) {
            x = dist(rng);
        }
    }

    std::vector<uint8_t> q8(ggml_row_size(GGML_TYPE_Q8_0, ne0)*ne1);
    ggml_quantize_chunk(GGML_TYPE_Q8_0, src.data(), q8.data(), 0, ne1, ne0, nullptr);

    std::vector<int32_t> indices(k);
    {
        std::uniform_int_distribution<int32_t> dist(0, ne1 - 1);
        for (auto & i : indices) {
            i = dist(rng);
        }
    }

    struct ggml_init_params params = {
        /*.mem_size   =*/ 64*1024*1024,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ false,
    };
    ggml_context * ctx = ggml_init(params);

    ggml_tensor * t_q8 = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, ne0, ne1);
    memcpy(t_q8->data, q8.data(), q8.size());

    ggml_tensor * idx = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, k);
    memcpy(idx->data, indices.data(), k*sizeof(int32_t));

    // A: gather quantized rows, then dequant the selection
    ggml_tensor * sel_q8 = ggml_get_rows(ctx, t_q8, idx);
    ggml_tensor * out_a  = ggml_cast(ctx, sel_q8, GGML_TYPE_F32);

    // B: dequant everything, then gather
    ggml_tensor * full_f32 = ggml_cast(ctx, t_q8, GGML_TYPE_F32);
    ggml_tensor * out_b    = ggml_get_rows(ctx, full_f32, idx);

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out_a);
    ggml_build_forward_expand(gf, out_b);
    ggml_graph_compute_with_ctx(ctx, gf, 4);

    const size_t n = (size_t) ne0*k;
    const float * a = (const float *) out_a->data;
    const float * b = (const float *) out_b->data;

    size_t mismatches = 0;
    for (size_t i = 0; i < n; ++i) {
        if (memcmp(&a[i], &b[i], sizeof(float)) != 0) {
            if (mismatches < 8) {
                printf("  bit mismatch at %zu: A=%a B=%a\n", i, a[i], b[i]);
            }
            ++mismatches;
        }
    }

    const bool ok = mismatches == 0;
    printf("case ne0=%4lld ne1=%5lld k=%4lld seed=%u: %s (bit mismatches %zu/%zu)\n",
            (long long) ne0, (long long) ne1, (long long) k, seed,
            ok ? "PASS" : "FAIL", mismatches, n);

    ggml_free(ctx);
    return ok;
}

int main() {
    bool ok = true;
    ok &= run_case(512, 1024, 128, 42);   // DSV4 compressed KV shape
    ok &= run_case(512, 4096, 512, 43);   // full indexer top-k width
    ok &= run_case(576,  300,  37, 44);   // non-power-of-two rows/cols
    if (!ok) {
        printf("test-dsv4-gather-dequant: FAIL\n");
        return 1;
    }
    printf("test-dsv4-gather-dequant: PASS\n");
    return 0;
}
