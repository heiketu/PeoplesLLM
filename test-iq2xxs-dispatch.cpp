// Dispatch-level test: prove that MUL_MAT / MUL_MAT_ID on IQ2_XXS weights placed in the
// CPU_REPACK extra buffer route to the new 8x8 kernels through the CPU backend's
// supports_op/compute_forward path, and that results match the plain vec_dot path.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-backend.h"
#include "ggml-common.h"
#include "repack.h"
#include "quants.h"

static uint64_t st = 0x9e3779b97f4a7c15ull;
static uint64_t ru(void){ st^=st<<13; st^=st>>7; st^=st<<17; return st; }

int main(void) {
    ggml_cpu_init();

    const int k = 2048;        // row length
    const int m = 256;         // rows per expert (multiple of 8)
    const int n_mats = 8;      // experts
    const int n_used = 3;      // experts per token
    const int n_tokens = 2;    // id rows

    ggml_backend_t backend = ggml_backend_cpu_init();
    ggml_backend_cpu_set_n_threads(backend, 8);

    ggml_backend_buffer_type_t repack_buft = ggml_backend_cpu_repack_buffer_type();
    ggml_backend_buffer_type_t cpu_buft    = ggml_backend_cpu_buffer_type();

    // source weights (original iq2_xxs layout), filled with random valid blocks
    std::vector<block_iq2_xxs> w((size_t) n_mats * m * (k / QK_K));
    for (auto & b : w) {
        uint64_t * p = (uint64_t *) b.qs;
        for (size_t i = 0; i < sizeof(b.qs) / 8; i++) p[i] = ru();
        b.d = ggml_fp32_to_fp16(0.01f * (float) ((int) (ru() % 200) - 100));
    }
    std::vector<float> act(k * n_tokens);
    for (auto & v : act) v = 0.5f * ((float) (ru() % 2000) / 1000.0f - 1.0f);

    int32_t ids_data[n_used * n_tokens];
    for (int t = 0; t < n_tokens; t++)
        for (int u = 0; u < n_used; u++)
            ids_data[t * n_used + u] = (int) (ru() % n_mats);

    struct ggml_init_params ip = { 64 * 1024 * 1024, NULL, true };
    struct ggml_context * ctx = ggml_init(ip);

    // ---- reference graph: weights in a plain CPU buffer ----
    struct ggml_tensor * a_ref = ggml_new_tensor_3d(ctx, GGML_TYPE_IQ2_XXS, k, m, n_mats);
    struct ggml_tensor * b_ref = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, k, 1, n_tokens);
    struct ggml_tensor * ids   = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, n_used, n_tokens);
    struct ggml_tensor * d_ref = ggml_mul_mat_id(ctx, a_ref, b_ref, ids);

    struct ggml_cgraph * g_ref = ggml_new_graph(ctx);
    ggml_build_forward_expand(g_ref, d_ref);

    ggml_backend_buffer_t buf_ref = ggml_backend_alloc_ctx_tensors_from_buft(ctx, cpu_buft);
    if (!buf_ref) { printf("alloc ref failed\n"); return 1; }
    ggml_backend_tensor_set(a_ref, w.data(), 0, ggml_nbytes(a_ref));
    ggml_backend_tensor_set(b_ref, act.data(), 0, ggml_nbytes(b_ref));
    ggml_backend_tensor_set(ids, ids_data, 0, ggml_nbytes(ids));

    if (ggml_backend_graph_compute(backend, g_ref) != GGML_STATUS_SUCCESS) {
        printf("ref compute failed\n"); return 1;
    }
    printf("ref path tensor->extra=%p (plain cpu buffer)\n", a_ref->extra);

    // ---- test graph: weights in the CPU_REPACK buffer ----
    struct ggml_tensor * a_rp = ggml_new_tensor_3d(ctx, GGML_TYPE_IQ2_XXS, k, m, n_mats);
    struct ggml_tensor * b_rp = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, k, 1, n_tokens);
    struct ggml_tensor * ids2 = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, n_used, n_tokens);
    struct ggml_tensor * d_rp = ggml_mul_mat_id(ctx, a_rp, b_rp, ids2);

    struct ggml_cgraph * g_rp = ggml_new_graph(ctx);
    ggml_build_forward_expand(g_rp, d_rp);

    ggml_backend_buffer_t buf_rp = ggml_backend_buft_alloc_buffer(repack_buft, ggml_nbytes(a_rp));
    if (!buf_rp) { printf("alloc repack failed\n"); return 1; }
    ggml_backend_tensor_alloc(buf_rp, a_rp, ggml_backend_buffer_get_base(buf_rp));
    ggml_backend_tensor_set(a_rp, w.data(), 0, ggml_nbytes(a_rp)); // triggers repack()
    printf("repack path tensor->extra=%p (CPU_REPACK buffer)\n", a_rp->extra);
    if (a_rp->extra == NULL) { printf("FAIL: no repack traits attached for IQ2_XXS\n"); return 1; }

    // b_rp/ids2/d_rp need a regular buffer
    struct ggml_context * ctx2 = ggml_init(ip);
    // put aux tensors in their own context/buffer but same graph nodes? simpler: allocate
    // remaining tensors of ctx into cpu buffer (a_rp already has buffer set)
    ggml_backend_buffer_t buf_aux = ggml_backend_cpu_buffer_type()->iface.alloc_buffer(cpu_buft, 1 << 20);
    ggml_backend_tensor_alloc(buf_aux, b_rp, ggml_backend_buffer_get_base(buf_aux));
    ggml_backend_tensor_alloc(buf_aux, ids2, (char *) ggml_backend_buffer_get_base(buf_aux) + 65536);
    ggml_backend_tensor_alloc(buf_aux, d_rp, (char *) ggml_backend_buffer_get_base(buf_aux) + 131072);
    ggml_backend_tensor_set(b_rp, act.data(), 0, ggml_nbytes(b_rp));
    ggml_backend_tensor_set(ids2, ids_data, 0, ggml_nbytes(ids2));

    if (ggml_backend_graph_compute(backend, g_rp) != GGML_STATUS_SUCCESS) {
        printf("repack compute failed\n"); return 1;
    }

    // compare
    std::vector<float> r0(ggml_nelements(d_ref)), r1(ggml_nelements(d_rp));
    ggml_backend_tensor_get(d_ref, r0.data(), 0, ggml_nbytes(d_ref));
    ggml_backend_tensor_get(d_rp, r1.data(), 0, ggml_nbytes(d_rp));
    double maxrel = 0, maxabs = 0, rms0 = 0; int nnan = 0;
    for (size_t i = 0; i < r0.size(); i++) {
        if (!isfinite(r0[i]) || !isfinite(r1[i])) { nnan++; continue; }
        rms0 += (double) r0[i] * r0[i];
        double ad = fabs(r0[i] - r1[i]);
        double rel = ad / (fabs(r0[i]) + 1e-6);
        if (rel > maxrel) maxrel = rel;
        if (ad > maxabs) maxabs = ad;
    }
    rms0 = sqrt(rms0 / (r0.size() - nnan));
    double noise0 = maxabs / rms0;
    printf("MUL_MAT_ID repack vs vec_dot path: maxrel=%.3e maxabs/rms=%.1e nan=%d %s\n", maxrel, noise0, nnan, (noise0 < 1e-5 && nnan == 0) ? "OK" : "FAIL");
    if (nnan || noise0 >= 1e-5) return 1;

    // ---- MUL_MAT (2d) through the repack path ----
    struct ggml_tensor * a2 = ggml_new_tensor_2d(ctx2, GGML_TYPE_IQ2_XXS, k, m);
    struct ggml_tensor * b2 = ggml_new_tensor_2d(ctx2, GGML_TYPE_F32, k, n_tokens);
    struct ggml_tensor * d2 = ggml_mul_mat(ctx2, a2, b2);
    struct ggml_cgraph * g2 = ggml_new_graph(ctx2);
    ggml_build_forward_expand(g2, d2);
    ggml_backend_buffer_t buf2 = ggml_backend_buft_alloc_buffer(repack_buft, ggml_nbytes(a2));
    ggml_backend_tensor_alloc(buf2, a2, ggml_backend_buffer_get_base(buf2));
    ggml_backend_tensor_set(a2, w.data(), 0, ggml_nbytes(a2));
    ggml_backend_buffer_t buf2a = ggml_backend_cpu_buffer_type()->iface.alloc_buffer(cpu_buft, 1 << 20);
    ggml_backend_tensor_alloc(buf2a, b2, ggml_backend_buffer_get_base(buf2a));
    ggml_backend_tensor_alloc(buf2a, d2, (char *) ggml_backend_buffer_get_base(buf2a) + 65536);
    ggml_backend_tensor_set(b2, act.data(), 0, ggml_nbytes(b2));
    printf("MUL_MAT tensor->extra=%p\n", a2->extra);
    if (ggml_backend_graph_compute(backend, g2) != GGML_STATUS_SUCCESS) {
        printf("mul_mat repack compute failed\n"); return 1;
    }
    std::vector<float> r2(ggml_nelements(d2));
    ggml_backend_tensor_get(d2, r2.data(), 0, ggml_nbytes(d2));
    // reference for mul_mat: reuse vec_dot directly
    const int nb = k / QK_K;
    block_q8_K * q8 = (block_q8_K *) malloc(sizeof(block_q8_K) * nb * n_tokens);
    double maxrel2 = 0, maxabs2 = 0, rms2 = 0; int nnan2 = 0;
    for (int t = 0; t < n_tokens; t++) {
        quantize_row_q8_K(act.data() + (size_t) t * k, q8 + (size_t) t * nb, k);
        for (int row = 0; row < m; row++) {
            float ref;
            ggml_vec_dot_iq2_xxs_q8_K(k, &ref, 0, &w[(size_t) row * nb], 0, q8 + (size_t) t * nb, 0, 1);
            float got = r2[(size_t) t * m + row];
            if (!isfinite(ref) || !isfinite(got)) { nnan2++; continue; }
            rms2 += (double) ref * ref;
            double ad = fabs(ref - got);
            double rel = ad / (fabs(ref) + 1e-6);
            if (rel > maxrel2) maxrel2 = rel;
            if (ad > maxabs2) maxabs2 = ad;
        }
    }
    rms2 = sqrt(rms2 / (m * n_tokens - nnan2));
    double noise2 = maxabs2 / rms2;
    printf("MUL_MAT repack vs vec_dot path:      maxrel=%.3e maxabs/rms=%.1e nan=%d %s\n", maxrel2, noise2, nnan2, (noise2 < 1e-5 && nnan2 == 0) ? "OK" : "FAIL");

    return (noise0 < 1e-5 && noise2 < 1e-5 && nnan2 == 0) ? 0 : 1;
}
