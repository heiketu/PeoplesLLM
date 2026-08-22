// Slice-0 microbenchmark for the UD-NL (UDNL) weight-packing proposal.
//
// Part A (--ceiling): single-core decode-throughput ceiling of the existing
//   formats, L2-hot (small weight footprint, no DRAM streaming): MXFP4, Q3_R,
//   Q4_K, IQ4_NL, Q2_K, IQ2_XXS — both the legacy per-row vec_dot and, where
//   the CPU_REPACK path exists, the repacked gemv with nr=1 (decode shape).
//   Reported as weight GB/s, GOP/s and cycles/weight (calibrated core clock).
//
// Part B (--w4nl): W4-NL prototype in the v2 NR16 x K4 layout: one ZMM
//   VPDPBUSD directly produces 16 output-row i32 lanes; the 16-entry biased-U8
//   codebook is decoded with VPSHUFB (lane-replicated LUT); activations are
//   q8_0 and the +128 bias correction uses a per-KQ activation sum precomputed
//   once per token (as the v1/v2 docs specify). Compared against the MXFP4 and
//   Q3_R repacked GEMV kernels on the same logical matrix.
//
// No default code paths are touched; everything lives in this file. The AVX512
// kernel uses a function-level target attribute so the test builds with the
// default flags and self-skips on machines without AVX512F/BW/DQ/VL/VNNI.

#include "ggml.h"
#include "ggml-impl.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "repack.h"
#include "quants.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <random>
#include <string>
#include <vector>

#if defined(__x86_64__) && defined(__GNUC__)
#include <immintrin.h>
#include <x86intrin.h>
#define UDNL_X86 1
#else
#define UDNL_X86 0
#endif

namespace {

std::vector<float> make_random_f32(int64_t n, uint32_t seed) {
    std::mt19937                    rng(seed);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    std::vector<float>              v(n);
    for (auto & x : v) x = dist(rng);
    return v;
}

// ---------------------------------------------------------------------------
// repacked-weight helper (same pattern as tests/test-repack-kernels.cpp)
// ---------------------------------------------------------------------------

struct repacked_weights {
    ggml_context *        ctx    = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    ggml_tensor *         tensor = nullptr;
    std::vector<char>     raw;     // original per-row quantized blocks
    const void *          data   = nullptr; // repacked array
    int64_t               nbytes = 0;
};

bool repack_supported(ggml_type type, int nc, int k) {
    struct ggml_init_params params = { 16 * 1024, nullptr, true };
    ggml_context * ctx = ggml_init(params);
    if (ctx == nullptr) return false;
    ggml_tensor * tensor = ggml_new_tensor_2d(ctx, type, k, nc);
    ggml_backend_buffer_t buffer = ggml_backend_buft_alloc_buffer(
        ggml_backend_cpu_repack_buffer_type(), ggml_nbytes(tensor));
    if (buffer == nullptr) { ggml_free(ctx); return false; }
    ggml_backend_tensor_alloc(buffer, tensor, ggml_backend_buffer_get_base(buffer));
    const bool supported = tensor->extra != nullptr;
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    return supported;
}

bool make_repacked(ggml_type type, const std::vector<float> & w, int nc, int k, repacked_weights & out) {
    const ggml_type_traits * traits = ggml_get_type_traits(type);
    ggml_quantize_init(type);

    const size_t row_bytes = ggml_row_size(type, k);
    out.raw.resize(row_bytes * nc);
    if (traits->from_float_ref) {
        for (int r = 0; r < nc; r++) {
            traits->from_float_ref(w.data() + (size_t) r * k, out.raw.data() + row_bytes * r, k);
        }
    } else {
        const std::vector<float> imatrix((size_t) k, 1.0f);
        ggml_quantize_chunk(type, w.data(), out.raw.data(), 0, nc, k, imatrix.data());
    }

    struct ggml_init_params params = { 1 * 1024 * 1024, nullptr, true };
    out.ctx = ggml_init(params);
    if (out.ctx == nullptr) return false;
    out.tensor = ggml_new_tensor_2d(out.ctx, type, k, nc);
    out.nbytes = ggml_nbytes(out.tensor);
    out.buffer = ggml_backend_buft_alloc_buffer(ggml_backend_cpu_repack_buffer_type(), out.nbytes);
    if (out.buffer == nullptr) return false;
    ggml_backend_tensor_alloc(out.buffer, out.tensor, ggml_backend_buffer_get_base(out.buffer));
    if (out.tensor->extra == nullptr) {
        ggml_backend_buffer_free(out.buffer);
        ggml_free(out.ctx);
        out = repacked_weights();
        return false;
    }
    ggml_backend_tensor_set(out.tensor, out.raw.data(), 0, out.raw.size());
    out.data = out.tensor->data;
    return true;
}

void free_repacked(repacked_weights & w) {
    if (w.buffer) ggml_backend_buffer_free(w.buffer);
    if (w.ctx)    ggml_free(w.ctx);
    w = repacked_weights();
}

// ---------------------------------------------------------------------------
// timing + frequency calibration
// ---------------------------------------------------------------------------

using clock_type = std::chrono::steady_clock;

double elapsed_s(clock_type::time_point a, clock_type::time_point b) {
    return std::chrono::duration<double>(b - a).count();
}

// Effective core frequency via a 1-cycle-latency dependent add chain. Printed
// for context only; cycles/weight below are reported in TSC cycles (invariant
// 2.3 GHz on this box) because the achieved core clock drifts ±20% under the
// powersave governor, and perf counters are unavailable (perf_event_paranoid=4).
double calibrate_ghz() {
    uint64_t x = 0x9e3779b97f4a7c15ull;
    const int64_t iters = 200000000;
    double best = 1e30;
    for (int rep = 0; rep < 5; rep++) {
        const auto t0 = clock_type::now();
        for (int64_t i = 0; i < iters; i++) {
            __asm__ volatile("add $1, %0" : "+r"(x));
        }
        const auto t1 = clock_type::now();
        best = std::min(best, elapsed_s(t0, t1));
    }
    volatile uint64_t sink = x;
    (void) sink;
    return (double) iters / best / 1e9;
}

#if UDNL_X86
// Invariant TSC frequency (rdtsc against CLOCK_MONOTONIC).
double measure_tsc_ghz() {
    double best = 1e30;
    for (int rep = 0; rep < 2; rep++) {
        struct timespec a, b, req = { 0, 200000000 };
        clock_gettime(CLOCK_MONOTONIC, &a);
        const unsigned long long t0 = __rdtsc();
        nanosleep(&req, nullptr);
        clock_gettime(CLOCK_MONOTONIC, &b);
        const unsigned long long t1 = __rdtsc();
        const double s = (b.tv_sec - a.tv_sec) + 1e-9 * (b.tv_nsec - a.tv_nsec);
        best = std::min(best, (t1 - t0) / s / 1e9);
    }
    return best;
}
#else
double measure_tsc_ghz() { return 1.0; }
#endif

// Min-over-reps timing with iteration-count auto-calibration.
template <typename F>
double time_best(F && body, double target_s = 0.15, int reps = 3) {
    body(); // warmup
    const auto t0 = clock_type::now();
    body();
    const auto t1 = clock_type::now();
    const double est = std::max(elapsed_s(t0, t1), 1e-9);
    const int iters = std::max(1, std::min(200000, (int) (target_s / est)));
    double best = 1e30;
    for (int r = 0; r < reps; r++) {
        const auto a = clock_type::now();
        for (int i = 0; i < iters; i++) {
            body();
            __asm__ volatile("" ::: "memory");
        }
        const auto b = clock_type::now();
        best = std::min(best, elapsed_s(a, b) / iters);
    }
    return best; // seconds per call
}

void report(const char * tag, const char * name, const char * kernel,
            int64_t weight_bytes, int64_t weights, double seconds, double tsc_ghz) {
    const double gbps = weight_bytes / seconds / 1e9;
    const double gops = 2.0 * weights / seconds / 1e9;
    const double cpw  = seconds * tsc_ghz * 1e9 / weights; // TSC cycles (2.3 GHz base)
    printf("[%-12s] %-8s %-18s %8.3f MB  %9.1f us  %8.2f GB/s  %8.2f GOP/s  %7.4f tsc_cyc/weight\n",
           tag, name, kernel, weight_bytes / 1048576.0, seconds * 1e6, gbps, gops, cpw);
}

// ---------------------------------------------------------------------------
// W4-NL NR16 x K4 prototype (v2 layout)
// ---------------------------------------------------------------------------

// 16-entry signed NL codebook (same grid as IQ4_NL); biased U8 LUT = C4[i]+128.
static const int8_t w4nl_codebook[16] = {
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113
};

// biased (+128) copy of the codebook for VPDPBUSD's U8xS8 form
static uint8_t w4nl_codebook_u8[16];

void init_codebook_u8() {
    for (int i = 0; i < 16; i++) w4nl_codebook_u8[i] = (uint8_t) (w4nl_codebook[i] + 128);
}

// Physical layout (all indexes derived so the AVX512 decode needs only
// and/shift/unpacklo/unpackhi + 2 VPSHUFB per 64 payload bytes):
//   panel p  = rows 16p..16p+15, one panel per 16 rows
//   KQ g     = 32 K columns; per panel per KQ: 256 B payload + 16 B srel (u8)
//   payload chunk s (0..3) covers k = 32g+8s .. 32g+8s+7, 64 B
//   within a chunk, 16-byte lane l covers rows 4l..4l+3; with i = row%4:
//     byte 16l+2i    = idx(k0) | idx(k1)<<4
//     byte 16l+2i+1  = idx(k2) | idx(k3)<<4
//     byte 16l+8+2i  = idx(k4) | idx(k5)<<4
//     byte 16l+8+2i+1= idx(k6) | idx(k7)<<4
// so lo/hi nibble split + unpacklo -> rows 0..15 x k0..3 (16 i32 lanes = 16
// rows), unpackhi -> rows 0..15 x k4..7.
struct w4nl_pack {
    std::vector<uint8_t> payload; // (n/16)*(k/32)*256
    std::vector<uint8_t> srel;    // (n/16)*(k/32)*16, 0 = zero group
    std::vector<float>   drow;    // n, includes the /255 of srel
    int64_t              nbytes = 0;
};

w4nl_pack pack_w4nl(const std::vector<float> & w, int n, int k) {
    const int ng = k / 32, np = n / 16;
    w4nl_pack pk;
    pk.payload.assign((size_t) np * ng * 256, 0);
    pk.srel.assign((size_t) np * ng * 16, 0);
    pk.drow.assign(n, 0.0f);

    std::vector<uint8_t> idx(32);
    for (int r = 0; r < n; r++) {
        const float * wr = w.data() + (size_t) r * k;
        std::vector<float> sg(ng);
        float dmax = 0.0f;
        for (int g = 0; g < ng; g++) {
            float amax = 0.0f;
            for (int j = 0; j < 32; j++) amax = std::max(amax, std::fabs(wr[32 * g + j]));
            sg[g] = amax / 127.0f;
            dmax = std::max(dmax, sg[g]);
        }
        pk.drow[r] = dmax > 0.0f ? dmax / 255.0f : 0.0f;
        const int p = r / 16, l = (r % 16) / 4, i = r % 4;
        for (int g = 0; g < ng; g++) {
            const int srel = sg[g] > 0.0f ? std::max(1, (int) lroundf(255.0f * sg[g] / dmax)) : 0;
            pk.srel[((size_t) p * ng + g) * 16 + (r % 16)] = (uint8_t) srel;
            const float seff = pk.drow[r] * srel;
            for (int j = 0; j < 32; j++) {
                int best = 0;
                if (srel > 0) {
                    const float t = wr[32 * g + j] / seff;
                    float bd = 1e30f;
                    for (int c = 0; c < 16; c++) {
                        const float d = std::fabs(t - w4nl_codebook[c]);
                        if (d < bd) { bd = d; best = c; }
                    }
                }
                idx[j] = (uint8_t) best;
            }
            uint8_t * pl = pk.payload.data() + ((size_t) p * ng + g) * 256;
            for (int j = 0; j < 32; j++) {
                const int s = j / 8, jj = j % 8;
                const int off = 64 * s + 16 * l + (jj < 4 ? 2 * i + jj / 2 : 8 + 2 * i + (jj - 4) / 2);
                if (jj % 2 == 0) pl[off] = (uint8_t) (pl[off] | idx[j]);
                else             pl[off] = (uint8_t) (pl[off] | (idx[j] << 4));
            }
        }
    }
    pk.nbytes = (int64_t) pk.payload.size() + pk.srel.size() + (int64_t) pk.drow.size() * 4;
    return pk;
}

// scalar reference: decode through the same packing map, int-exact raw dots
void w4nl_gemv_ref(int n, int k, float * out, const w4nl_pack & pk, const block_q8_0 * a) {
    const int ng = k / 32;
    for (int r = 0; r < n; r++) {
        const int p = r / 16, l = (r % 16) / 4, i = r % 4;
        double acc = 0.0;
        for (int g = 0; g < ng; g++) {
            const uint8_t * pl = pk.payload.data() + ((size_t) p * ng + g) * 256;
            const int srel = pk.srel[((size_t) p * ng + g) * 16 + (r % 16)];
            int64_t raw = 0;
            for (int j = 0; j < 32; j++) {
                const int s = j / 8, jj = j % 8;
                const int off = 64 * s + 16 * l + (jj < 4 ? 2 * i + jj / 2 : 8 + 2 * i + (jj - 4) / 2);
                const int id = jj % 2 == 0 ? pl[off] & 0xF : pl[off] >> 4;
                raw += (int) w4nl_codebook[id] * (int) a[g].qs[j];
            }
            const float dy = GGML_COMPUTE_FP16_TO_FP32(a[g].d);
            acc += (double) raw * (double) srel * (double) pk.drow[r] * dy;
        }
        out[r] = (float) acc;
    }
}

#if UDNL_X86
typedef int32_t unaligned_i32 __attribute__((aligned(1), may_alias));

__attribute__((target("avx512f,avx512bw,avx512dq,avx512vl,avx512vnni")))
void w4nl_gemv_nr16_avx512(int n, int k, float * GGML_RESTRICT out,
                           const uint8_t * GGML_RESTRICT payload,
                           const uint8_t * GGML_RESTRICT srel,
                           const float * GGML_RESTRICT drow,
                           const block_q8_0 * GGML_RESTRICT a,
                           const int32_t * GGML_RESTRICT asum /* per KQ, precomputed per token */) {
    const int ng = k / 32;
    const __m512i m4  = _mm512_set1_epi8(0x0F);
    const __m512i lut = _mm512_broadcast_i32x4(_mm_loadu_si128((const __m128i *) w4nl_codebook_u8));
    for (int p = 0; p < n / 16; p++) {
        const uint8_t * pl = payload + (size_t) p * ng * 256;
        const uint8_t * sr = srel + (size_t) p * ng * 16;
        __m512 accf = _mm512_setzero_ps();
        for (int g = 0; g < ng; g++, pl += 256, sr += 16) {
            const block_q8_0 * ab = a + g;
            __m512i accA = _mm512_setzero_si512();
            __m512i accB = _mm512_setzero_si512();
            for (int s = 0; s < 4; s++) {
                const __m512i pkq = _mm512_loadu_si512(pl + 64 * s);
                const __m512i lo = _mm512_and_si512(pkq, m4);
                const __m512i hi = _mm512_and_si512(_mm512_srli_epi16(pkq, 4), m4);
                const __m512i iA = _mm512_unpacklo_epi8(lo, hi); // rows 0..15 x k0..3
                const __m512i iB = _mm512_unpackhi_epi8(lo, hi); // rows 0..15 x k4..7
                const __m512i wA = _mm512_shuffle_epi8(lut, iA);
                const __m512i wB = _mm512_shuffle_epi8(lut, iB);
                accA = _mm512_dpbusd_epi32(accA, wA,
                    _mm512_set1_epi32(*(const unaligned_i32 *) (ab->qs + 8 * s)));
                accB = _mm512_dpbusd_epi32(accB, wB,
                    _mm512_set1_epi32(*(const unaligned_i32 *) (ab->qs + 8 * s + 4)));
            }
            // undo the +128 codebook bias: raw = acc - 128*sum(a[KQ])
            __m512i raw = _mm512_add_epi32(accA, accB);
            raw = _mm512_sub_epi32(raw, _mm512_set1_epi32(128 * asum[g]));
            const __m512 sv = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i *) sr)));
            const float  dy = GGML_COMPUTE_FP16_TO_FP32(ab->d);
            accf = _mm512_fmadd_ps(_mm512_cvtepi32_ps(raw), _mm512_mul_ps(sv, _mm512_set1_ps(dy)), accf);
        }
        accf = _mm512_mul_ps(accf, _mm512_loadu_ps(drow + 16 * p));
        _mm512_storeu_ps(out + 16 * p, accf);
    }
}

// L2-hot pure-load roofline: sequential 64 B loads + xor, no decode work.
__attribute__((target("avx512f")))
uint64_t pure_load_u64(const uint8_t * p, int64_t bytes) {
    __m512i acc0 = _mm512_setzero_si512();
    __m512i acc1 = _mm512_setzero_si512();
    for (int64_t i = 0; i + 128 <= bytes; i += 128) {
        acc0 = _mm512_xor_si512(acc0, _mm512_loadu_si512(p + i));
        acc1 = _mm512_xor_si512(acc1, _mm512_loadu_si512(p + i + 64));
    }
    return (uint64_t) _mm512_reduce_add_epi32(acc0) + _mm512_reduce_add_epi32(acc1);
}
#endif // UDNL_X86

// ---------------------------------------------------------------------------
// Part A: per-format single-core decode ceiling
// ---------------------------------------------------------------------------

typedef void (*vec_dot_fn_t)(int, float *, size_t, const void *, size_t, const void *, size_t, int);

struct ceil_type {
    ggml_type    type;
    const char * name;
    vec_dot_fn_t vec_dot;
    void (*gemv)(int, float *, size_t, const void *, const void *, int, int); // may be null
    ggml_type    act_type;
};

void run_ceiling(int nc, int k, double ghz) {
    const std::vector<ceil_type> types = {
        { GGML_TYPE_MXFP4,   "MXFP4",   ggml_vec_dot_mxfp4_q8_0,   ggml_gemv_mxfp4_8x8_q8_0,   GGML_TYPE_Q8_0 },
        { GGML_TYPE_Q3_R,    "Q3_R",    ggml_vec_dot_q3_r_q8_0,    ggml_gemv_q3_r_1x1_q8_0,    GGML_TYPE_Q8_0 },
        { GGML_TYPE_Q4_K,    "Q4_K",    ggml_vec_dot_q4_K_q8_K,    ggml_gemv_q4_K_8x8_q8_K,    GGML_TYPE_Q8_K },
        { GGML_TYPE_IQ4_NL,  "IQ4_NL",  ggml_vec_dot_iq4_nl_q8_0,  ggml_gemv_iq4_nl_8x8_q8_0,  GGML_TYPE_Q8_0 },
        { GGML_TYPE_Q2_K,    "Q2_K",    ggml_vec_dot_q2_K_q8_K,    ggml_gemv_q2_K_8x8_q8_K,    GGML_TYPE_Q8_K },
        { GGML_TYPE_IQ2_XXS, "IQ2_XXS", ggml_vec_dot_iq2_xxs_q8_K, nullptr,                    GGML_TYPE_Q8_K },
    };

    printf("== ceiling: nc=%d k=%d (single core, L2-hot) ==\n", nc, k);
    const std::vector<float> w = make_random_f32((int64_t) nc * k, 4242);
    const std::vector<float> x = make_random_f32(k, 777);

    for (const auto & fn : types) {
        std::vector<char> q8(ggml_row_size(fn.act_type, k));
        if (fn.act_type == GGML_TYPE_Q8_K) quantize_row_q8_K(x.data(), q8.data(), k);
        else                               quantize_row_q8_0(x.data(), q8.data(), k);

        repacked_weights rw;
        const bool has_repack = fn.gemv != nullptr && make_repacked(fn.type, w, nc, k, rw);
        if (fn.gemv != nullptr && !has_repack) {
            // quantization itself may be what failed; build raw rows alone for vec_dot
            printf("  [%s] note: no CPU_REPACK kernel, vec_dot only\n", fn.name);
        }
        if (!has_repack) {
            // still need raw rows for the legacy path
            const ggml_type_traits * traits = ggml_get_type_traits(fn.type);
            ggml_quantize_init(fn.type);
            const size_t row_bytes = ggml_row_size(fn.type, k);
            rw.raw.resize(row_bytes * nc);
            if (traits->from_float_ref) {
                for (int r = 0; r < nc; r++) traits->from_float_ref(w.data() + (size_t) r * k, rw.raw.data() + row_bytes * r, k);
            } else {
                const std::vector<float> imatrix((size_t) k, 1.0f);
                ggml_quantize_chunk(fn.type, w.data(), rw.raw.data(), 0, nc, k, imatrix.data());
            }
            rw.nbytes = (int64_t) rw.raw.size();
        }

        const size_t row_bytes = ggml_row_size(fn.type, k);
        std::vector<float> s(nc, 0.0f);

        const double t_dot = time_best([&] {
            for (int r = 0; r < nc; r++) {
                fn.vec_dot(k, &s[r], 0, rw.raw.data() + row_bytes * r, 0, q8.data(), 0, 1);
            }
        });
        report("ceiling", fn.name, "vec_dot-legacy", rw.nbytes, (int64_t) nc * k, t_dot, ghz);

        if (has_repack) {
            const double t_gemv = time_best([&] {
                fn.gemv(k, s.data(), nc, rw.data, q8.data(), 1, nc);
            });
            report("ceiling", fn.name, "gemv-repack-nr1", rw.nbytes, (int64_t) nc * k, t_gemv, ghz);
        }
        free_repacked(rw);
    }
}

// ---------------------------------------------------------------------------
// Part B: W4-NL NR16 x K4 vs MXFP4 gemv vs Q3_R gemv
// ---------------------------------------------------------------------------

void run_w4nl_bench(int nc, int k, double ghz) {
#if UDNL_X86
    printf("== w4nl: nc=%d k=%d (single core, L2-hot) ==\n", nc, k);
    const std::vector<float> w = make_random_f32((int64_t) nc * k, 1234);
    const std::vector<float> x = make_random_f32(k, 999);

    std::vector<char> q8(ggml_row_size(GGML_TYPE_Q8_0, k));
    quantize_row_q8_0(x.data(), q8.data(), k);
    const block_q8_0 * a = (const block_q8_0 *) q8.data();
    const int ng = k / 32;
    std::vector<int32_t> asum(ng);
    for (int g = 0; g < ng; g++) {
        int32_t s = 0;
        for (int j = 0; j < 32; j++) s += a[g].qs[j];
        asum[g] = s;
    }

    w4nl_pack pk = pack_w4nl(w, nc, k);

    // correctness on the full matrix (scalar reference is slow but the matrix is small)
    {
        std::vector<float> o_ref(nc), o_avx(nc);
        w4nl_gemv_ref(nc, k, o_ref.data(), pk, a);
        w4nl_gemv_nr16_avx512(nc, k, o_avx.data(), pk.payload.data(), pk.srel.data(), pk.drow.data(), a, asum.data());
        double max_abs = 0.0, max_rel = 0.0;
        int n_bad = 0;
        for (int r = 0; r < nc; r++) {
            const double d = std::fabs((double) o_avx[r] - o_ref[r]);
            const double rel = d / std::max(1.0, (double) std::fabs(o_ref[r]));
            max_abs = std::max(max_abs, d);
            max_rel = std::max(max_rel, rel);
            if (d > 1e-3 * std::max(1.0, (double) std::fabs(o_ref[r]))) n_bad++;
        }
        printf("  [W4-NL] avx512-vs-scalar: max_abs=%.3g max_rel=%.3g bad=%d -> %s\n",
               max_abs, max_rel, n_bad, n_bad == 0 ? "OK" : "MISMATCH");
    }

    std::vector<float> s(nc, 0.0f);
    const double t_w4nl = time_best([&] {
        w4nl_gemv_nr16_avx512(nc, k, s.data(), pk.payload.data(), pk.srel.data(), pk.drow.data(), a, asum.data());
    });
    report("w4nl", "W4-NL", "gemv-nr16-proto", pk.nbytes, (int64_t) nc * k, t_w4nl, ghz);

    // comparison kernels on the same logical matrix
    repacked_weights rw;
    if (make_repacked(GGML_TYPE_MXFP4, w, nc, k, rw)) {
        const double t = time_best([&] {
            ggml_gemv_mxfp4_8x8_q8_0(k, s.data(), nc, rw.data, q8.data(), 1, nc);
        });
        report("w4nl", "MXFP4", "gemv-repack-nr1", rw.nbytes, (int64_t) nc * k, t, ghz);
        free_repacked(rw);
    }
    if (make_repacked(GGML_TYPE_Q3_R, w, nc, k, rw)) {
        const double t = time_best([&] {
            ggml_gemv_q3_r_1x1_q8_0(k, s.data(), nc, rw.data, q8.data(), 1, nc);
        });
        report("w4nl", "Q3_R", "gemv-repack-nr1", rw.nbytes, (int64_t) nc * k, t, ghz);
        free_repacked(rw);
    }

    // L2-hot pure-load roofline over a same-sized buffer
    {
        std::vector<uint8_t> buf(pk.payload.size());
        uint64_t sink = 0;
        const double t = time_best([&] { sink ^= pure_load_u64(buf.data(), (int64_t) buf.size()); });
        report("w4nl", "ROOFLINE", "pure-load-64B", (int64_t) buf.size(), (int64_t) nc * k, t, ghz);
        if (sink == 42) printf("!\n");
    }

    // bits-per-weight summary for the table
    const double bpw_w4nl  = 8.0 * pk.nbytes / ((double) nc * k);
    printf("  [W4-NL] effective bpw (payload+srel+drow): %.4f\n", bpw_w4nl);
#else
    (void) nc; (void) k; (void) ghz;
    printf("== w4nl: SKIPPED (not an x86-64 GCC build) ==\n");
#endif
}

bool has_isa() {
    // ggml-cpu only exposes these three; F implies BW/DQ/VL in practice on the
    // CPUs this bench targets (ICX-SP), and the kernel self-guards at build time.
    return ggml_cpu_has_avx512() && ggml_cpu_has_avx512_vnni() && ggml_cpu_has_avx512_vbmi();
}

} // namespace

int main(int argc, char ** argv) {
    setenv("GGML_REPACK_Q2_K", "1", 1); // same opt-in as test-repack-kernels

    if (!has_isa()) {
        printf("test-udnl-w4nl: SKIPPED (needs AVX512F/BW/DQ/VL/VNNI/VBMI)\n");
        return 0;
    }
    init_codebook_u8();

    const double tsc_ghz = measure_tsc_ghz();
    const double core_ghz = calibrate_ghz();
    printf("clock: TSC %.3f GHz (cycles below are TSC cycles), achieved core %.3f GHz (powersave governor)\n",
           tsc_ghz, core_ghz);

    const std::string mode = argc > 1 ? argv[1] : "--all";
    const int nc = argc > 2 ? atoi(argv[2]) : 1024;
    const int k  = argc > 3 ? atoi(argv[3]) : 2048;
    if (nc % 16 != 0 || k % 256 != 0) {
        fprintf(stderr, "nc must be a multiple of 16, k a multiple of 256\n");
        return 1;
    }

    if (mode == "--ceiling" || mode == "--all") {
        run_ceiling(nc, k, tsc_ghz);
    }
    if (mode == "--w4nl" || mode == "--all") {
        run_w4nl_bench(nc, k, tsc_ghz);
    }
    return 0;
}
