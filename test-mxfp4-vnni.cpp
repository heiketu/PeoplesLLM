// Microbenchmark + regression reference for ggml_vec_dot_mxfp4_q8_0 / ggml_vec_dot_nvfp4_q8_0
// (maddubs+madd -> vpdpbusd VNNI conversion). Usage:
//   test-mxfp4-vnni dump   -> write reference outputs to /tmp/mxfp4-ref.bin
//   test-mxfp4-vnni check  -> compare against /tmp/mxfp4-ref.bin (must be bitexact)
// Always prints ns/row timings.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-common.h"
#include "quants.h"

static uint64_t st = 0x243f6a8885a308d3ull;
static uint64_t ru(void){ st^=st<<13; st^=st>>7; st^=st<<17; return st; }
static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e3 + ts.tv_nsec / 1e6;
}

int main(int argc, char ** argv) {
    ggml_cpu_init();
    const int n = 7168;           // row length (multiple of 32 and 64)
    const int rows = 1024;
    const int nbm = n / QK_MXFP4; // 224
    const int nbn = n / QK_NVFP4; // 112
    const int nb8 = n / QK8_0;    // 224

    block_mxfp4 * wx = (block_mxfp4 *) aligned_alloc(64, (size_t) rows * nbm * sizeof(block_mxfp4));
    block_nvfp4 * wn = (block_nvfp4 *) aligned_alloc(64, (size_t) rows * nbn * sizeof(block_nvfp4));
    block_q8_0  * y  = (block_q8_0 *)  aligned_alloc(64, (size_t) nb8 * sizeof(block_q8_0));

    for (int64_t i = 0; i < (int64_t) rows * nbm; i++) {
        uint8_t * p = (uint8_t *) wx[i].qs;
        for (size_t k = 0; k < sizeof(wx[i].qs); k++) p[k] = (uint8_t) ru();
        wx[i].e = (uint8_t) (127 + (int) (ru() % 9) - 4); // e8m0 around 1.0
    }
    for (int64_t i = 0; i < (int64_t) rows * nbn; i++) {
        uint8_t * p = (uint8_t *) wn[i].qs;
        for (size_t k = 0; k < sizeof(wn[i].qs); k++) p[k] = (uint8_t) ru();
        for (int k = 0; k < QK_NVFP4 / QK_NVFP4_SUB; k++) wn[i].d[k] = (uint8_t) (ru() % 100 + 28); // ue4m3 small-ish
    }
    for (int i = 0; i < nb8; i++) {
        for (int k = 0; k < QK8_0; k++) y[i].qs[k] = (int8_t) (ru() % 255) - 127;
        y[i].d = ggml_fp32_to_fp16(0.01f * (float) ((int) (ru() % 200) - 100));
    }

    float * out_mx = (float *) malloc(rows * sizeof(float));
    float * out_nv = (float *) malloc(rows * sizeof(float));

    // correctness pass
    for (int r = 0; r < rows; r++) {
        ggml_vec_dot_mxfp4_q8_0(n, &out_mx[r], 0, wx + (size_t) r * nbm, 0, y, 0, 1);
        ggml_vec_dot_nvfp4_q8_0(n, &out_nv[r], 0, wn + (size_t) r * nbn, 0, y, 0, 1);
    }

    const char * mode = argc > 1 ? argv[1] : "check";
    if (strcmp(mode, "dump") == 0) {
        FILE * f = fopen("/tmp/mxfp4-ref.bin", "wb");
        fwrite(out_mx, sizeof(float), rows, f);
        fwrite(out_nv, sizeof(float), rows, f);
        fclose(f);
        printf("dumped reference (%d rows x2)\n", rows);
    } else {
        FILE * f = fopen("/tmp/mxfp4-ref.bin", "rb");
        if (!f) { printf("no reference file; run with 'dump' first\n"); return 1; }
        float * ref_mx = (float *) malloc(rows * sizeof(float));
        float * ref_nv = (float *) malloc(rows * sizeof(float));
        if (fread(ref_mx, sizeof(float), rows, f) != (size_t) rows) return 1;
        if (fread(ref_nv, sizeof(float), rows, f) != (size_t) rows) return 1;
        fclose(f);
        int bad = 0;
        for (int r = 0; r < rows; r++) {
            if (memcmp(&out_mx[r], &ref_mx[r], 4) != 0) { if (bad < 3) printf("mx row %d: %g vs %g\n", r, out_mx[r], ref_mx[r]); bad++; }
            if (memcmp(&out_nv[r], &ref_nv[r], 4) != 0) { if (bad < 3) printf("nv row %d: %g vs %g\n", r, out_nv[r], ref_nv[r]); bad++; }
        }
        printf("bitexact vs reference: %s (%d mismatches)\n", bad ? "FAIL" : "OK", bad);
        if (bad) return 1;
    }

    // timing
    const int rep = 300;
    volatile float sink = 0;
    double t0 = now_ms();
    for (int it = 0; it < rep; it++)
        for (int r = 0; r < rows; r++)
            ggml_vec_dot_mxfp4_q8_0(n, (float *) &sink, 0, wx + (size_t) r * nbm, 0, y, 0, 1);
    double t_mx = (now_ms() - t0) * 1e6 / ((double) rep * rows);

    t0 = now_ms();
    for (int it = 0; it < rep; it++)
        for (int r = 0; r < rows; r++)
            ggml_vec_dot_nvfp4_q8_0(n, (float *) &sink, 0, wn + (size_t) r * nbn, 0, y, 0, 1);
    double t_nv = (now_ms() - t0) * 1e6 / ((double) rep * rows);

    printf("mxfp4 vec_dot: %8.1f ns/row (n=%d)\n", t_mx, n);
    printf("nvfp4 vec_dot: %8.1f ns/row (n=%d)\n", t_nv, n);
    printf("sink=%g\n", (float) sink);
    return 0;
}
