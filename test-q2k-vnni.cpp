// Regression reference + microbenchmark for ggml_gemm_q2_K_8x8_q8_K
// (maddubs+add_epi16 chain -> vpdpbusd VNNI conversion, bitexact).
// Exercises the real ggml CPU backend MUL_MAT dispatch with the repack
// buffer type (which routes big-nr tiles through ggml_gemm_q2_K_8x8_q8_K).
// Usage:
//   test-q2k-vnni dump [threads]  -> write repack-path outputs to /tmp/q2k-ref.bin
//   test-q2k-vnni check [threads] -> compare against /tmp/q2k-ref.bin (must be bitexact)
//   test-q2k-vnni bench [threads] -> skip correctness, timing only
// Always prints ms/graph for plain (CPU buffer) vs repack path.

#include <math.h>
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

struct run_result { double ms; std::vector<float> out; };

// one MUL_MAT: a[k x m] (Q2_K) * b[k x n_tokens] (F32) -> d[m x n_tokens]
static run_result run_mm(ggml_backend_t backend, ggml_backend_buffer_type_t buft,
                         const std::vector<block_q2_K> & w,
                         const std::vector<float> & act,
                         int k, int m, int n_tokens, int iters) {
    ggml_backend_buffer_type_t cpu_buft = ggml_backend_cpu_buffer_type();

    struct ggml_init_params ip = { 64 * 1024 * 1024, NULL, true };
    struct ggml_context * ctx = ggml_init(ip);

    struct ggml_tensor * a = ggml_new_tensor_2d(ctx, GGML_TYPE_Q2_K, k, m);
    struct ggml_tensor * b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k, n_tokens);
    struct ggml_tensor * d = ggml_mul_mat(ctx, a, b);
    struct ggml_cgraph * g = ggml_new_graph(ctx);
    ggml_build_forward_expand(g, d);

    ggml_backend_buffer_t buf_a = ggml_backend_buft_alloc_buffer(buft, ggml_nbytes(a));
    ggml_backend_tensor_alloc(buf_a, a, ggml_backend_buffer_get_base(buf_a));
    ggml_backend_tensor_set(a, w.data(), 0, ggml_nbytes(a));

    ggml_backend_buffer_t buf_aux = ggml_backend_buft_alloc_buffer(cpu_buft, (size_t) 64 << 20);
    ggml_backend_tensor_alloc(buf_aux, b, ggml_backend_buffer_get_base(buf_aux));
    ggml_backend_tensor_alloc(buf_aux, d, (char *) ggml_backend_buffer_get_base(buf_aux) + (16 << 20));
    ggml_backend_tensor_set(b, act.data(), 0, ggml_nbytes(b));

    // warmup + capture output
    for (int i = 0; i < 3; i++) ggml_backend_graph_compute(backend, g);
    run_result res;
    res.out.resize((size_t) m * n_tokens);
    ggml_backend_tensor_get(d, res.out.data(), 0, ggml_nbytes(d));

    double t0 = now_ms();
    for (int i = 0; i < iters; i++) ggml_backend_graph_compute(backend, g);
    res.ms = (now_ms() - t0) / iters;

    ggml_backend_buffer_free(buf_a);
    ggml_backend_buffer_free(buf_aux);
    ggml_free(ctx);
    return res;
}

int main(int argc, char ** argv) {
    const char * mode = argc > 1 ? argv[1] : "check";
    const int n_threads = argc > 2 ? atoi(argv[2]) : 32;
    const bool do_correct = strcmp(mode, "bench") != 0;

    ggml_cpu_init();
    ggml_backend_t backend = ggml_backend_cpu_init();
    ggml_backend_cpu_set_n_threads(backend, n_threads);

    // gemm shapes: nr (= n_tokens) >= 16 routes through ggml_gemm_q2_K_8x8_q8_K
    struct { int k, m, n_tokens, iters; } cases[] = {
        { 7168, 4096, 16, 100 },
        { 7168, 4096, 64, 50 },
        { 2048, 2048, 32, 200 },
        // covers all four gemm loops: 16r x 16c, 4r x 16c (tok%16=4), 16r x 8c
        // and 4r x 8c tail columns (m%16=8)
        { 2048, 1032, 20, 200 },
    };

    ggml_backend_buffer_type_t cpu_buft    = ggml_backend_cpu_buffer_type();
    ggml_backend_buffer_type_t repack_buft = ggml_backend_cpu_repack_buffer_type();

    printf("threads=%d mode=%s\n\n", n_threads, mode);
    printf("%-36s %10s %10s %8s\n", "case", "plain(ms)", "repack(ms)", "speedup");

    std::vector<float> all_out;
    for (auto & c : cases) {
        std::vector<block_q2_K> w((size_t) c.m * (c.k / QK_K));
        for (auto & blk : w) {
            uint8_t * p = (uint8_t *) blk.scales;
            for (size_t i = 0; i < sizeof(blk.scales); i++) p[i] = (uint8_t) ru();
            p = (uint8_t *) blk.qs;
            for (size_t i = 0; i < sizeof(blk.qs); i++) p[i] = (uint8_t) ru();
            blk.data.data.d    = ggml_fp32_to_fp16(0.01f * (float) ((int) (ru() % 200) - 100));
            blk.data.data.dmin = ggml_fp32_to_fp16(0.01f * (float) ((int) (ru() % 200) - 100));
        }
        std::vector<float> act((size_t) c.k * c.n_tokens);
        for (auto & v : act) v = 0.5f * ((float) (ru() % 2000) / 1000.0f - 1.0f);

        run_result r_plain  = run_mm(backend, cpu_buft,    w, act, c.k, c.m, c.n_tokens, c.iters);
        run_result r_repack = run_mm(backend, repack_buft, w, act, c.k, c.m, c.n_tokens, c.iters);

        char name[128];
        snprintf(name, sizeof name, "MM k=%d m=%d tok=%d", c.k, c.m, c.n_tokens);
        printf("%-36s %10.4f %10.4f %7.2fx\n", name, r_plain.ms, r_repack.ms, r_plain.ms / r_repack.ms);

        if (do_correct) {
            // sanity: plain vs repack must be numerically close (different summation order)
            double max_diff = 0;
            for (size_t i = 0; i < r_plain.out.size(); i++) {
                double d = fabs((double) r_plain.out[i] - (double) r_repack.out[i]);
                if (d > max_diff) max_diff = d;
            }
            printf("  plain-vs-repack max abs diff: %g\n", max_diff);
            size_t off = all_out.size();
            all_out.resize(off + r_repack.out.size());
            memcpy(all_out.data() + off, r_repack.out.data(), r_repack.out.size() * sizeof(float));
        }
    }

    if (!do_correct) return 0;

    const char * ref_path = "/tmp/q2k-ref.bin";
    if (strcmp(mode, "dump") == 0) {
        FILE * f = fopen(ref_path, "wb");
        fwrite(all_out.data(), sizeof(float), all_out.size(), f);
        fclose(f);
        printf("\ndumped reference (%zu floats)\n", all_out.size());
        return 0;
    }

    FILE * f = fopen(ref_path, "rb");
    if (!f) { printf("no reference file; run with 'dump' first\n"); return 1; }
    std::vector<float> ref(all_out.size());
    if (fread(ref.data(), sizeof(float), ref.size(), f) != ref.size()) return 1;
    fclose(f);
    int bad = 0, nan = 0;
    for (size_t i = 0; i < all_out.size(); i++) {
        if (all_out[i] != all_out[i]) { nan++; continue; }
        if (memcmp(&all_out[i], &ref[i], 4) != 0) {
            if (bad < 5) printf("mismatch at %zu: %g vs %g\n", i, all_out[i], ref[i]);
            bad++;
        }
    }
    printf("\nbitexact vs reference: %s (%d mismatches, %d nan, %zu values)\n",
           bad ? "FAIL" : "OK", bad, nan, all_out.size());
    return bad ? 1 : 0;
}
