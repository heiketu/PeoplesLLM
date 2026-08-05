// Correctness harness for GGML_CUDA_RESIDENT_BF16 (opt-in resident f16/bf16
// weight materialization in the CUDA device buffer).
//
// Builds a small mul_mat graph with a quantized 2D weight, runs it on the CPU
// backend (reference) and on the first GPU backend, and reports the max
// absolute/relative difference plus the effective on-device weight type.
//
// Run twice - without and with GGML_CUDA_RESIDENT_BF16=1 - and compare the two
// output dumps (see the project's resident-bf16 notes); each process also
// prints GPU-vs-CPU-reference error metrics directly.
//
// usage: test-resident-bf16 [out.bin]

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

static void fill_random(std::vector<float> & v, uint32_t seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    for (auto & x : v) {
        x = dist(rng);
    }
}

// build a fresh context holding one quantized weight + one f32 activation,
// upload through the backend's buffer (this is where the resident-bf16 hook
// lives), run mul_mat, and download the result
static bool run_mul_mat(
        ggml_backend_t           backend,
        enum ggml_type           wtype,
        const std::vector<uint8_t> & wq,
        const std::vector<float> & x,
        int64_t                  ne0,
        int64_t                  ne1,
        int64_t                  m,
        std::vector<float>     & out,
        enum ggml_type         & wtype_after) {
    struct ggml_init_params params = {
        /*.mem_size   =*/ 16*1024*1024,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context * ctx = ggml_init(params);

    ggml_tensor * w = ggml_new_tensor_2d(ctx, wtype, ne0, ne1);
    ggml_set_name(w, "test.attn_q.weight");
    ggml_tensor * a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ne0, m);
    ggml_set_name(a, "test.act");

    ggml_tensor * cur = ggml_mul_mat(ctx, w, a);

    if (!ggml_backend_alloc_ctx_tensors(ctx, backend)) {
        fprintf(stderr, "%s: failed to allocate tensors\n", __func__);
        ggml_free(ctx);
        return false;
    }

    ggml_backend_tensor_set(w, wq.data(), 0, wq.size());
    ggml_backend_tensor_set(a, x.data(), 0, x.size()*sizeof(float));

    wtype_after = w->type; // the resident-bf16 hook flips this after the upload

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, cur);

    if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "%s: graph compute failed\n", __func__);
        ggml_free(ctx);
        return false;
    }

    out.resize(ne1*m);
    ggml_backend_tensor_get(cur, out.data(), 0, out.size()*sizeof(float));

    ggml_free(ctx);
    return true;
}

int main(int argc, char ** argv) {
    const char * out_path = argc > 1 ? argv[1] : nullptr;

    ggml_backend_load_all();

    const int64_t ne0 = 4096;
    const int64_t ne1 = 2048;

    const enum ggml_type wtype = GGML_TYPE_Q3_K;

    std::vector<float> w_f32(ne0*ne1);
    fill_random(w_f32, 1234);

    std::vector<uint8_t> wq(ggml_row_size(wtype, ne0)*ne1);
    const size_t nq = ggml_quantize_chunk(wtype, w_f32.data(), wq.data(), 0, ne1, ne0, nullptr);
    if (nq != wq.size()) {
        fprintf(stderr, "quantize size mismatch: %zu != %zu\n", nq, wq.size());
        return 1;
    }

    // exact reference weight: dequantize back to f32 (same values the GEMM should see)
    const ggml_type_traits * qtraits = ggml_get_type_traits(wtype);
    std::vector<float> w_deq(ne0*ne1);
    for (int64_t r = 0; r < ne1; ++r) {
        qtraits->to_float(wq.data() + r*ggml_row_size(wtype, ne0), w_deq.data() + r*ne0, ne0);
    }

    ggml_backend_t cpu = ggml_backend_cpu_init();
    if (!cpu) {
        fprintf(stderr, "no CPU backend\n");
        return 1;
    }

    ggml_backend_dev_t gpu_dev = nullptr;
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_GPU) {
            gpu_dev = dev;
            break;
        }
    }
    if (!gpu_dev) {
        fprintf(stderr, "no GPU backend\n");
        return 1;
    }
    ggml_backend_t gpu = ggml_backend_dev_init(gpu_dev, nullptr);

    printf("weight qtype        = %s\n", ggml_type_name(wtype));
    printf("weight bytes quant  = %zu\n", wq.size());
    printf("weight bytes f16eq  = %zu\n", ggml_row_size(GGML_TYPE_F16, ne0)*ne1);

    bool ok = true;
    enum ggml_type wtype_gpu = GGML_TYPE_COUNT;
    for (const int64_t m : {int64_t(512), int64_t(1)}) {
        std::vector<float> x(ne0*m);
        fill_random(x, 5678 + (uint32_t) m);

        // exact f32 reference (CPU, dequantized f32 weight x f32 activation)
        std::vector<float> w_deq_bytes(w_deq);
        std::vector<uint8_t> w_f32_raw(w_deq_bytes.size()*sizeof(float));
        memcpy(w_f32_raw.data(), w_deq_bytes.data(), w_f32_raw.size());

        std::vector<float> out_ref, out_gpu;
        enum ggml_type wt_cpu = GGML_TYPE_COUNT, wt_gpu = GGML_TYPE_COUNT;
        ok = ok && run_mul_mat(cpu, GGML_TYPE_F32, w_f32_raw, x, ne0, ne1, m, out_ref, wt_cpu);
        ok = ok && run_mul_mat(gpu, wtype,        wq,        x, ne0, ne1, m, out_gpu, wt_gpu);
        if (!ok) {
            break;
        }
        wtype_gpu = wt_gpu;

        double max_abs = 0.0, sum_sq_ref = 0.0, sum_sq_diff = 0.0;
        for (size_t i = 0; i < out_ref.size(); ++i) {
            const double d = std::fabs((double) out_gpu[i] - out_ref[i]);
            max_abs = std::max(max_abs, d);
            sum_sq_ref  += (double) out_ref[i]*out_ref[i];
            sum_sq_diff += d*d;
        }
        const double rel_l2 = std::sqrt(sum_sq_diff/(sum_sq_ref + 1e-30));
        printf("m=%4ld  GPU vs f32 ref: max_abs = %.6e  rel_l2 = %.6e\n", (long) m, max_abs, rel_l2);

        if (out_path && m == 512) {
            FILE * f = fopen(out_path, "wb");
            if (!f) {
                fprintf(stderr, "failed to open %s\n", out_path);
                return 1;
            }
            fwrite(out_gpu.data(), sizeof(float), out_gpu.size(), f);
            fclose(f);
            printf("wrote %s\n", out_path);
        }
    }
    if (!ok) {
        return 1;
    }

    printf("weight type on GPU  = %s (after upload)\n", ggml_type_name(wtype_gpu));

    ggml_backend_free(gpu);
    ggml_backend_free(cpu);

    const bool converted = wtype_gpu != wtype;
    const bool expect_converted = getenv("GGML_CUDA_RESIDENT_BF16") && atoi(getenv("GGML_CUDA_RESIDENT_BF16")) != 0;
    if (converted != expect_converted) {
        fprintf(stderr, "FAIL: conversion state %d does not match env expectation %d\n", (int) converted, (int) expect_converted);
        return 1;
    }

    printf("PASS\n");
    return 0;
}
