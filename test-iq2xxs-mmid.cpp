// MUL_MAT_ID dispatch + A/B test for the IQ2_XXS 8x8 repack kernels at
// production-like MoE shapes: decode (1 token, gemv tail path) and PP (64
// tokens, gemm tile path). Compares the CPU_REPACK path against the plain CPU
// buffer path (ggml-cpu.c mul_mat_id, vec_dot per row) for correctness and time.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <vector>

#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-backend.h"
#include "ggml-common.h"
#include "repack.h"
#include "quants.h"

static uint64_t st = 0x9e3779b97f4a7c15ull;
static uint64_t ru(void){ st^=st<<13; st^=st>>7; st^=st<<17; return st; }

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e3 + ts.tv_nsec / 1e6;
}

struct mmid_case {
    const char * name;
    int n_tokens;
    int iters;
};

int main(void) {
    ggml_cpu_init();

    const int k      = 2048;  // row length
    const int m      = 512;   // rows per expert (multiple of 8)
    const int n_as   = 32;    // experts
    const int n_used = 8;     // experts per token
    const int n_threads = 8;

    ggml_backend_t backend = ggml_backend_cpu_init();
    ggml_backend_cpu_set_n_threads(backend, n_threads);

    ggml_backend_buffer_type_t repack_buft = ggml_backend_cpu_repack_buffer_type();
    ggml_backend_buffer_type_t cpu_buft    = ggml_backend_cpu_buffer_type();

    std::vector<block_iq2_xxs> w((size_t) n_as * m * (k / QK_K));
    for (auto & b : w) {
        uint64_t * p = (uint64_t *) b.qs;
        for (size_t i = 0; i < sizeof(b.qs) / 8; i++) p[i] = ru();
        b.d = ggml_fp32_to_fp16(0.01f * (float) ((int) (ru() % 200) - 100));
    }

    const mmid_case cases[] = {
        { "decode tok=1 ", 1, 50 },
        { "pp     tok=64", 64, 10 },
    };

    int fails = 0;

    for (const auto & c : cases) {
        const int n_tokens = c.n_tokens;

        std::vector<float> act((size_t) k * n_tokens);
        for (auto & v : act) v = 0.5f * ((float) (ru() % 2000) / 1000.0f - 1.0f);

        std::vector<int32_t> ids_data((size_t) n_used * n_tokens);
        for (int t = 0; t < n_tokens; t++)
            for (int u = 0; u < n_used; u++)
                ids_data[(size_t) t * n_used + u] = (int) (ru() % n_as);

        struct ggml_init_params ip = { 256 * 1024 * 1024, NULL, true };
        struct ggml_context * ctx = ggml_init(ip);

        // ---- reference: plain CPU buffer (production non-repack path) ----
        struct ggml_tensor * a_ref = ggml_new_tensor_3d(ctx, GGML_TYPE_IQ2_XXS, k, m, n_as);
        struct ggml_tensor * b_ref = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, k, 1, n_tokens);
        struct ggml_tensor * ids   = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, n_used, n_tokens);
        struct ggml_tensor * d_ref = ggml_mul_mat_id(ctx, a_ref, b_ref, ids);
        struct ggml_cgraph * g_ref = ggml_new_graph(ctx);
        ggml_build_forward_expand(g_ref, d_ref);

        ggml_backend_buffer_t buf_ref = ggml_backend_alloc_ctx_tensors_from_buft(ctx, cpu_buft);
        ggml_backend_tensor_set(a_ref, w.data(), 0, ggml_nbytes(a_ref));
        ggml_backend_tensor_set(b_ref, act.data(), 0, ggml_nbytes(b_ref));
        ggml_backend_tensor_set(ids, ids_data.data(), 0, ggml_nbytes(ids));

        if (ggml_backend_graph_compute(backend, g_ref) != GGML_STATUS_SUCCESS) {
            printf("%s ref compute failed\n", c.name); return 1;
        }
        {
            std::vector<float> dbg(ggml_nelements(d_ref));
            ggml_backend_tensor_get(d_ref, dbg.data(), 0, ggml_nbytes(d_ref));
            int nf = 0;
            for (size_t i = 0; i < dbg.size(); i++) if (isfinite(dbg[i])) nf++;
            printf("%s ref finite=%zu/%zu first8:", c.name, (size_t) nf, dbg.size());
            for (int i = 0; i < 8; i++) printf(" %g", dbg[i]);
            printf("\n");
        }

        // ---- test: CPU_REPACK buffer ----
        struct ggml_tensor * a_rp = ggml_new_tensor_3d(ctx, GGML_TYPE_IQ2_XXS, k, m, n_as);
        struct ggml_tensor * b_rp = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, k, 1, n_tokens);
        struct ggml_tensor * ids2 = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, n_used, n_tokens);
        struct ggml_tensor * d_rp = ggml_mul_mat_id(ctx, a_rp, b_rp, ids2);
        struct ggml_cgraph * g_rp = ggml_new_graph(ctx);
        ggml_build_forward_expand(g_rp, d_rp);

        ggml_backend_buffer_t buf_rp = ggml_backend_buft_alloc_buffer(repack_buft, ggml_nbytes(a_rp));
        ggml_backend_tensor_alloc(buf_rp, a_rp, ggml_backend_buffer_get_base(buf_rp));
        ggml_backend_tensor_set(a_rp, w.data(), 0, ggml_nbytes(a_rp)); // triggers repack()

        if (a_rp->extra == NULL) { printf("%s FAIL: no repack traits attached\n", c.name); return 1; }

        ggml_backend_buffer_t buf_aux = cpu_buft->iface.alloc_buffer(cpu_buft, 64 << 20);
        ggml_backend_tensor_alloc(buf_aux, b_rp, ggml_backend_buffer_get_base(buf_aux));
        ggml_backend_tensor_alloc(buf_aux, ids2, (char *) ggml_backend_buffer_get_base(buf_aux) + (32 << 20));
        ggml_backend_tensor_alloc(buf_aux, d_rp, (char *) ggml_backend_buffer_get_base(buf_aux) + (40 << 20));
        ggml_backend_tensor_set(b_rp, act.data(), 0, ggml_nbytes(b_rp));
        ggml_backend_tensor_set(ids2, ids_data.data(), 0, ggml_nbytes(ids2));

        if (ggml_backend_graph_compute(backend, g_rp) != GGML_STATUS_SUCCESS) {
            printf("%s repack compute failed\n", c.name); return 1;
        }

        std::vector<float> r0(ggml_nelements(d_ref)), r1(ggml_nelements(d_rp));
        ggml_backend_tensor_get(d_ref, r0.data(), 0, ggml_nbytes(d_ref));
        ggml_backend_tensor_get(d_rp, r1.data(), 0, ggml_nbytes(d_rp));
        double maxrel = 0, maxabs = 0, rms = 0;
        int nnan0 = 0, nnan1 = 0;
        for (size_t i = 0; i < r0.size(); i++) {
            if (!isfinite(r0[i])) { nnan0++; continue; }
            if (!isfinite(r1[i])) { nnan1++; continue; }
            rms += (double) r0[i] * r0[i];
            double ad = fabs(r0[i] - r1[i]);
            double rel = ad / (fabs(r0[i]) + 1e-6);
            if (rel > maxrel) maxrel = rel;
            if (ad > maxabs) maxabs = ad;
        }
        rms = sqrt(rms / (r0.size() - nnan0 - nnan1));

        // spot-check one row of the repack result against direct vec_dot
        {
            const int nb = k / QK_K;
            block_q8_K * q8 = (block_q8_K *) malloc(sizeof(block_q8_K) * nb);
            double worst = 0; int nnan = 0;
            for (int t = 0; t < n_tokens; t += (n_tokens > 4 ? n_tokens/4 : 1)) {
                quantize_row_q8_K(act.data() + (size_t) t * k, q8, k);
                for (int u = 0; u < n_used; u++) {
                    const int e = ids_data[(size_t) t * n_used + u];
                    for (int row = 0; row < m; row += m/4) {
                        float ref;
                        ggml_vec_dot_iq2_xxs_q8_K(k, &ref, 0, &w[((size_t) e * m + row) * nb], 0, q8, 0, 1);
                        // dst layout: [m, n_used, n_tokens]
                        float got = r1[(size_t) t * n_used * m + (size_t) u * m + row];
                        if (!isfinite(ref) || !isfinite(got)) { nnan++; continue; }
                        double rel = fabs(ref - got) / (fabs(ref) + 1e-6);
                        if (rel > worst) worst = rel;
                    }
                }
            }
            free(q8);
            printf("%s repack vs vec_dot spot-check: maxrel=%.3e nan=%d %s\n", c.name, worst, nnan, (worst < 1e-4 && nnan == 0) ? "OK" : "FAIL");
            if (worst >= 1e-4 || nnan != 0) fails++;
        }

        // timing: interleave the two graphs, take the best-of
        double t_ref = 1e30, t_rp = 1e30;
        for (int it = 0; it < c.iters; it++) {
            double t0 = now_ms();
            ggml_backend_graph_compute(backend, g_ref);
            double t1 = now_ms();
            ggml_backend_graph_compute(backend, g_rp);
            double t2 = now_ms();
            if (t1 - t0 < t_ref) t_ref = t1 - t0;
            if (t2 - t1 < t_rp)  t_rp  = t2 - t1;
        }

        // FP reassociation noise shows up as large *relative* error on near-zero
        // outputs; judge by absolute error relative to the output magnitude
        const double noise = maxabs / rms;
        printf("%s dst rms=%.3f maxabs=%.3e (maxabs/rms=%.1e) maxrel=%.3e nan_ref=%d nan_rp=%d\n",
               c.name, rms, maxabs, noise, maxrel, nnan0, nnan1);
        if (nnan0 != 0 || nnan1 != 0) fails++;

        printf("%s repack vs plain path:      noise=%.1e %s | plain=%.3f ms repack=%.3f ms speedup=%.2fx\n",
               c.name, noise, noise < 1e-5 ? "OK" : "FAIL", t_ref, t_rp, t_ref / t_rp);
        if (noise >= 1e-5) fails++;

        ggml_backend_buffer_free(buf_ref);
        ggml_backend_buffer_free(buf_rp);
        ggml_backend_buffer_free(buf_aux);
        ggml_free(ctx);
    }

    ggml_backend_free(backend);
    return fails ? 1 : 0;
}
