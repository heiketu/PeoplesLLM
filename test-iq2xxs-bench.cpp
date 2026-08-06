// Op-level A/B benchmark: MUL_MAT_ID (and MUL_MAT) on IQ2_XXS through the plain CPU
// buffer path (per-row AVX512 vec_dot) vs the CPU_REPACK path (new 8x8 gemv/gemm),
// through the real ggml CPU backend dispatch with threads.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <vector>

#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-backend.h"
#include "ggml-common.h"
#include "repack.h"

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e3 + ts.tv_nsec / 1e6;
}

static uint64_t st = 0x9e3779b97f4a7c15ull;
static uint64_t ru(void){ st^=st<<13; st^=st>>7; st^=st<<17; return st; }

struct bench_result { double ms_plain, ms_repack; };

static bench_result bench_mmid(ggml_backend_t backend, int k, int m, int n_mats, int n_used, int n_tokens, int iters) {
    std::vector<block_iq2_xxs> w((size_t) n_mats * m * (k / QK_K));
    for (auto & b : w) {
        uint64_t * p = (uint64_t *) b.qs;
        for (size_t i = 0; i < sizeof(b.qs) / 8; i++) p[i] = ru();
        b.d = ggml_fp32_to_fp16(0.01f * (float) (ru() % 200 - 100));
    }
    std::vector<float> act((size_t) k * n_tokens);
    for (auto & v : act) v = 0.5f * ((float) (ru() % 2000) / 1000.0f - 1.0f);
    std::vector<int32_t> ids_data(n_used * n_tokens);
    for (int t = 0; t < n_tokens; t++)
        for (int u = 0; u < n_used; u++)
            ids_data[t * n_used + u] = (int) (ru() % n_mats);

    ggml_backend_buffer_type_t cpu_buft    = ggml_backend_cpu_buffer_type();
    ggml_backend_buffer_type_t repack_buft = ggml_backend_cpu_repack_buffer_type();

    bench_result res = {0, 0};
    for (int phase = 0; phase < 2; phase++) {
        struct ggml_init_params ip = { 64 * 1024 * 1024, NULL, true };
        struct ggml_context * ctx = ggml_init(ip);

        struct ggml_tensor * a  = ggml_new_tensor_3d(ctx, GGML_TYPE_IQ2_XXS, k, m, n_mats);
        struct ggml_tensor * b  = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, k, 1, n_tokens);
        struct ggml_tensor * id = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, n_used, n_tokens);
        struct ggml_tensor * d  = ggml_mul_mat_id(ctx, a, b, id);
        struct ggml_cgraph * g  = ggml_new_graph(ctx);
        ggml_build_forward_expand(g, d);

        ggml_backend_buffer_type_t buft = phase == 0 ? cpu_buft : repack_buft;
        ggml_backend_buffer_t buf_a = ggml_backend_buft_alloc_buffer(buft, ggml_nbytes(a));
        ggml_backend_tensor_alloc(buf_a, a, ggml_backend_buffer_get_base(buf_a));
        ggml_backend_tensor_set(a, w.data(), 0, ggml_nbytes(a));

        ggml_backend_buffer_t buf_aux = ggml_backend_buft_alloc_buffer(cpu_buft, (size_t) 64 << 20);
        ggml_backend_tensor_alloc(buf_aux, b, ggml_backend_buffer_get_base(buf_aux));
        ggml_backend_tensor_alloc(buf_aux, id, (char *) ggml_backend_buffer_get_base(buf_aux) + (16 << 20));
        ggml_backend_tensor_alloc(buf_aux, d, (char *) ggml_backend_buffer_get_base(buf_aux) + (32 << 20));
        ggml_backend_tensor_set(b, act.data(), 0, ggml_nbytes(b));
        ggml_backend_tensor_set(id, ids_data.data(), 0, ggml_nbytes(id));

        // warmup
        for (int i = 0; i < 3; i++) ggml_backend_graph_compute(backend, g);
        double t0 = now_ms();
        for (int i = 0; i < iters; i++) ggml_backend_graph_compute(backend, g);
        double dt = (now_ms() - t0) / iters;
        if (phase == 0) res.ms_plain = dt; else res.ms_repack = dt;

        ggml_backend_buffer_free(buf_a);
        ggml_backend_buffer_free(buf_aux);
        ggml_free(ctx);
    }
    return res;
}

int main(void) {
    ggml_cpu_init();
    ggml_backend_t backend = ggml_backend_cpu_init();
    const int n_threads = 32;
    ggml_backend_cpu_set_n_threads(backend, n_threads);

    printf("threads=%d\n\n", n_threads);
    printf("%-44s %10s %10s %8s\n", "case", "plain(ms)", "repack(ms)", "speedup");

    struct { int k, m, n_mats, n_used, n_tokens, iters; } cases[] = {
        // TG-like: 1 token, DSV4-Flash-ish expert sizes
        { 2048, 2048, 16, 6,  1, 300 },
        // small batch (4 tokens)
        { 2048, 2048, 16, 6,  4, 300 },
        // medium batch (32 tokens)
        { 2048, 2048, 16, 6, 32, 200 },
        // bigger experts, TG
        { 7168, 2048, 16, 6,  1, 100 },
    };
    for (auto & c : cases) {
        bench_result r = bench_mmid(backend, c.k, c.m, c.n_mats, c.n_used, c.n_tokens, c.iters);
        char name[128];
        snprintf(name, sizeof name, "MMID k=%d m=%d mats=%d used=%d tok=%d", c.k, c.m, c.n_mats, c.n_used, c.n_tokens);
        printf("%-44s %10.4f %10.4f %7.2fx\n", name, r.ms_plain, r.ms_repack, r.ms_plain / r.ms_repack);
    }
    return 0;
}
