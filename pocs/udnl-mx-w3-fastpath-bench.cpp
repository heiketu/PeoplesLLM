// udnl-mx-w3-fastpath-bench: controlled microkernel PoC for the UDNL_MX
// all-W3 (mw == 0xAAAA) specialization. Duplicates the runtime panel kernel
// (mixed) and a fast variant with the per-group mode extraction/branch
// removed; verifies bit-exact equality on all-W3 panels and measures both.
// Gate per E4A-UDNL-OPT-DIRECTIONS.md: >=2% kernel speedup to justify runtime.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

#include <immintrin.h>

using clock_type = std::chrono::steady_clock;

#define GGML_RESTRICT __restrict

static const int8_t kvalues_iq4nl[16] = {
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
};
static const int8_t kvalues_udnl3[8] = {
    -127, -83, -49, -22, 1, 25, 53, 89,
};
static const int8_t kvalues_udnl2[4] = {
    -127, -35, 25, 113,
};

static constexpr int UDNL_MX_PB = 1728; // 16 rows x 108B row-block
static constexpr int QK8_0 = 32;

struct block_q8_0 {
    uint16_t d;
    int8_t qs[QK8_0];
};

struct udnl_w4_arec {
    int32_t asum128;
    float dy;
};

static inline float fp16_to_fp32(uint16_t h) {
    return _cvtsh_ss(h);
}

static inline uint16_t fp32_to_fp16(float f) {
    return _cvtss_sh(f, 0);
}

static inline __m512i udnl_w4_lut_biased(void) {
    uint8_t lut8[16];
    for (int i = 0; i < 16; ++i) lut8[i] = (uint8_t) (kvalues_iq4nl[i] + 128);
    return _mm512_broadcast_i32x4(_mm_loadu_si128((const __m128i *) lut8));
}

static inline __m512i udnl_mx_lut_biased(const int8_t * cb, int ncb) {
    uint8_t lut8[16] = {};
    for (int i = 0; i < ncb; ++i) lut8[i] = (uint8_t) (cb[i] + 128);
    return _mm512_broadcast_i32x4(_mm_loadu_si128((const __m128i *) lut8));
}

static inline int32_t udnl_w4_load_i32(const int8_t * p) {
    int32_t v;
    memcpy(&v, p, sizeof(v));
    return v;
}

// ---- reference: verbatim copy of the runtime mixed kernel ----
template <int NR>
static void panel_mixed(
        int nb, const uint8_t * GGML_RESTRICT pw,
        const block_q8_0 * GGML_RESTRICT a, int64_t anb,
        const udnl_w4_arec * GGML_RESTRICT rec,
        float * GGML_RESTRICT s_p, size_t bs) {
    const __m512i LUT4 = udnl_w4_lut_biased();
    const __m512i LUT3 = udnl_mx_lut_biased(kvalues_udnl3, 8);
    const __m512i LUT2 = udnl_mx_lut_biased(kvalues_udnl2, 4);
    const __m512i m0F  = _mm512_set1_epi8(0x0F);
    const __m512i m03  = _mm512_set1_epi8(0x03);
    const __m512i m01  = _mm512_set1_epi8(0x01);
    const __m512i msA2 = _mm512_set1_epi64(0x2624222006040200ULL);
    const __m512i msB2 = _mm512_set1_epi64(0x2E2C2A280E0C0A08ULL);
    const __m512i msAH = _mm512_set1_epi64(0x2322212003020100ULL);
    const __m512i msBH = _mm512_set1_epi64(0x2726252407060504ULL);

    __m512 accf[NR];
    for (int y = 0; y < NR; ++y) accf[y] = _mm512_setzero_ps();

    for (int b = 0; b < nb; ++b) {
        const uint8_t * GGML_RESTRICT pb = pw + (int64_t) b*UDNL_MX_PB;
        const uint16_t mw = ((const uint16_t *) (pb + 1696))[0];
        const __m512 dv = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *) (pb + 1664)));
        __m512 accq[NR];
        for (int y = 0; y < NR; ++y) accq[y] = _mm512_setzero_ps();

        const uint8_t * GGML_RESTRICT pl = pb;
        for (int g = 0; g < 8; ++g) {
            const int mode = (mw >> 2*g) & 3;
            __m512i wA[4], wB[4];
            if (mode == 3) {
                for (int s4 = 0; s4 < 4; ++s4) {
                    const __m512i c  = _mm512_loadu_si512(pl + 64*s4);
                    const __m512i lo = _mm512_and_si512(c, m0F);
                    const __m512i hi = _mm512_and_si512(_mm512_srli_epi16(c, 4), m0F);
                    wA[s4] = _mm512_shuffle_epi8(LUT4, _mm512_unpacklo_epi8(lo, hi));
                    wB[s4] = _mm512_shuffle_epi8(LUT4, _mm512_unpackhi_epi8(lo, hi));
                }
                pl += 256;
            } else if (mode == 2) {
                for (int s4 = 0; s4 < 4; ++s4) {
                    const __m512i cl = _mm512_cvtepu16_epi32(_mm256_loadu_si256((const __m256i *) (pl + 48*s4)));
                    const __m512i ch = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i *) (pl + 48*s4 + 32)));
                    const __m512i uA = _mm512_and_si512(_mm512_multishift_epi64_epi8(msA2, cl), m03);
                    const __m512i uB = _mm512_and_si512(_mm512_multishift_epi64_epi8(msB2, cl), m03);
                    const __m512i hA = _mm512_and_si512(_mm512_multishift_epi64_epi8(msAH, ch), m01);
                    const __m512i hB = _mm512_and_si512(_mm512_multishift_epi64_epi8(msBH, ch), m01);
                    wA[s4] = _mm512_shuffle_epi8(LUT3, _mm512_or_si512(uA, _mm512_slli_epi32(hA, 2)));
                    wB[s4] = _mm512_shuffle_epi8(LUT3, _mm512_or_si512(uB, _mm512_slli_epi32(hB, 2)));
                }
                pl += 192;
            } else {
                for (int s4 = 0; s4 < 4; ++s4) {
                    const __m512i cl = _mm512_cvtepu16_epi32(_mm256_loadu_si256((const __m256i *) (pl + 32*s4)));
                    const __m512i uA = _mm512_and_si512(_mm512_multishift_epi64_epi8(msA2, cl), m03);
                    const __m512i uB = _mm512_and_si512(_mm512_multishift_epi64_epi8(msB2, cl), m03);
                    wA[s4] = _mm512_shuffle_epi8(LUT2, uA);
                    wB[s4] = _mm512_shuffle_epi8(LUT2, uB);
                }
                pl += 128;
            }
            const __m512 svf = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(
                    _mm_loadu_si128((const __m128i *) pl)));
            pl += 16;
            for (int y = 0; y < NR; ++y) {
                const block_q8_0 * GGML_RESTRICT ab = a + (int64_t) y*anb + 8*b + g;
                const udnl_w4_arec * GGML_RESTRICT rg = rec + (int64_t) y*anb + 8*b + g;
                __m512i accA = _mm512_setzero_si512();
                __m512i accB = _mm512_setzero_si512();
                for (int s4 = 0; s4 < 4; ++s4) {
                    accA = _mm512_dpbusd_epi32(accA, wA[s4], _mm512_set1_epi32(udnl_w4_load_i32(ab->qs + 8*s4)));
                    accB = _mm512_dpbusd_epi32(accB, wB[s4], _mm512_set1_epi32(udnl_w4_load_i32(ab->qs + 8*s4 + 4)));
                }
                const __m512i raw = _mm512_sub_epi32(_mm512_add_epi32(accA, accB),
                                                     _mm512_set1_epi32(rg->asum128));
                const __m512 part = _mm512_mul_ps(_mm512_cvtepi32_ps(raw), svf);
                accq[y] = _mm512_fmadd_ps(part, _mm512_set1_ps(rg->dy), accq[y]);
            }
        }
        for (int y = 0; y < NR; ++y) {
            accf[y] = _mm512_fmadd_ps(accq[y], dv, accf[y]);
        }
    }
    for (int y = 0; y < NR; ++y) {
        _mm512_storeu_ps(s_p + y*bs, accf[y]);
    }
}

// ---- fast variant: mw == 0xAAAA known, fixed W3 decode, no mode branches ----
template <int NR>
static void panel_w3_fast(
        int nb, const uint8_t * GGML_RESTRICT pw,
        const block_q8_0 * GGML_RESTRICT a, int64_t anb,
        const udnl_w4_arec * GGML_RESTRICT rec,
        float * GGML_RESTRICT s_p, size_t bs) {
    const __m512i LUT3 = udnl_mx_lut_biased(kvalues_udnl3, 8);
    const __m512i m03  = _mm512_set1_epi8(0x03);
    const __m512i m01  = _mm512_set1_epi8(0x01);
    const __m512i msA2 = _mm512_set1_epi64(0x2624222006040200ULL);
    const __m512i msB2 = _mm512_set1_epi64(0x2E2C2A280E0C0A08ULL);
    const __m512i msAH = _mm512_set1_epi64(0x2322212003020100ULL);
    const __m512i msBH = _mm512_set1_epi64(0x2726252407060504ULL);

    __m512 accf[NR];
    for (int y = 0; y < NR; ++y) accf[y] = _mm512_setzero_ps();

    for (int b = 0; b < nb; ++b) {
        const uint8_t * GGML_RESTRICT pb = pw + (int64_t) b*UDNL_MX_PB;
        const __m512 dv = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *) (pb + 1664)));
        __m512 accq[NR];
        for (int y = 0; y < NR; ++y) accq[y] = _mm512_setzero_ps();

        const uint8_t * GGML_RESTRICT pl = pb;
        for (int g = 0; g < 8; ++g) {
            __m512i wA[4], wB[4];
            for (int s4 = 0; s4 < 4; ++s4) {
                const __m512i cl = _mm512_cvtepu16_epi32(_mm256_loadu_si256((const __m256i *) (pl + 48*s4)));
                const __m512i ch = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i *) (pl + 48*s4 + 32)));
                const __m512i uA = _mm512_and_si512(_mm512_multishift_epi64_epi8(msA2, cl), m03);
                const __m512i uB = _mm512_and_si512(_mm512_multishift_epi64_epi8(msB2, cl), m03);
                const __m512i hA = _mm512_and_si512(_mm512_multishift_epi64_epi8(msAH, ch), m01);
                const __m512i hB = _mm512_and_si512(_mm512_multishift_epi64_epi8(msBH, ch), m01);
                wA[s4] = _mm512_shuffle_epi8(LUT3, _mm512_or_si512(uA, _mm512_slli_epi32(hA, 2)));
                wB[s4] = _mm512_shuffle_epi8(LUT3, _mm512_or_si512(uB, _mm512_slli_epi32(hB, 2)));
            }
            pl += 192;
            const __m512 svf = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(
                    _mm_loadu_si128((const __m128i *) pl)));
            pl += 16;
            for (int y = 0; y < NR; ++y) {
                const block_q8_0 * GGML_RESTRICT ab = a + (int64_t) y*anb + 8*b + g;
                const udnl_w4_arec * GGML_RESTRICT rg = rec + (int64_t) y*anb + 8*b + g;
                __m512i accA = _mm512_setzero_si512();
                __m512i accB = _mm512_setzero_si512();
                for (int s4 = 0; s4 < 4; ++s4) {
                    accA = _mm512_dpbusd_epi32(accA, wA[s4], _mm512_set1_epi32(udnl_w4_load_i32(ab->qs + 8*s4)));
                    accB = _mm512_dpbusd_epi32(accB, wB[s4], _mm512_set1_epi32(udnl_w4_load_i32(ab->qs + 8*s4 + 4)));
                }
                const __m512i raw = _mm512_sub_epi32(_mm512_add_epi32(accA, accB),
                                                     _mm512_set1_epi32(rg->asum128));
                const __m512 part = _mm512_mul_ps(_mm512_cvtepi32_ps(raw), svf);
                accq[y] = _mm512_fmadd_ps(part, _mm512_set1_ps(rg->dy), accq[y]);
            }
        }
        for (int y = 0; y < NR; ++y) {
            accf[y] = _mm512_fmadd_ps(accq[y], dv, accf[y]);
        }
    }
    for (int y = 0; y < NR; ++y) {
        _mm512_storeu_ps(s_p + y*bs, accf[y]);
    }
}

static void make_w3_panel(uint8_t * pb, int nb, std::mt19937 & rng) {
    std::uniform_int_distribution<int> byte(0, 255);
    std::uniform_real_distribution<float> scale(0.5f, 2.0f);
    for (int b = 0; b < nb; ++b) {
        uint8_t * p = pb + (int64_t) b*UDNL_MX_PB;
        for (int i = 0; i < 1664; ++i) p[i] = (uint8_t) byte(rng);
        for (int r = 0; r < 16; ++r) {
            ((uint16_t *) (p + 1664))[r] = fp32_to_fp16(scale(rng));
        }
        ((uint16_t *) (p + 1696))[0] = 0xAAAA;
    }
}

template <typename F>
static double best_time(F && fn, int repeats) {
    double best = 1e30;
    for (int sample = 0; sample < 7; ++sample) {
        const auto begin = clock_type::now();
        for (int repeat = 0; repeat < repeats; ++repeat) fn();
        const auto end = clock_type::now();
        best = std::min(best, std::chrono::duration<double>(end - begin).count() / repeats);
    }
    return best;
}

template <int NR>
static void run_case(int nb, int panels) {
    std::mt19937 rng(12345);
    const int nc = panels*16;
    std::vector<uint8_t> w((size_t) panels*nb*UDNL_MX_PB);
    make_w3_panel(w.data(), panels*nb, rng);

    const int anb = 8*nb;
    std::vector<block_q8_0> act((size_t) anb*NR);
    std::uniform_int_distribution<int> code(-127, 127);
    std::uniform_real_distribution<float> dscale(0.01f, 0.05f);
    for (size_t i = 0; i < act.size(); ++i) {
        act[i].d = fp32_to_fp16(dscale(rng));
        for (int v = 0; v < 32; ++v) act[i].qs[v] = (int8_t) code(rng);
    }
    std::vector<udnl_w4_arec> rec((size_t) anb*NR);
    for (int y = 0; y < NR; ++y) {
        for (int j = 0; j < anb; ++j) {
            int asum = 0;
            for (int v = 0; v < QK8_0; ++v) asum += act[(size_t) y*anb + j].qs[v];
            rec[(size_t) y*anb + j].asum128 = 128*asum;
            rec[(size_t) y*anb + j].dy = fp16_to_fp32(act[(size_t) y*anb + j].d);
        }
    }

    std::vector<float> s_mix((size_t) NR*nc, -1.0f), s_fast((size_t) NR*nc, 1.0f);
    for (int p = 0; p < panels; ++p) {
        panel_mixed<NR>(nb, w.data() + (size_t) p*nb*UDNL_MX_PB, act.data(), anb, rec.data(),
                        s_mix.data() + 16*p, nc);
        panel_w3_fast<NR>(nb, w.data() + (size_t) p*nb*UDNL_MX_PB, act.data(), anb, rec.data(),
                          s_fast.data() + 16*p, nc);
    }
    uint64_t mismatch = 0;
    for (size_t i = 0; i < s_mix.size(); ++i) {
        uint32_t a, b;
        memcpy(&a, &s_mix[i], 4);
        memcpy(&b, &s_fast[i], 4);
        if (a != b) ++mismatch;
    }

    const int repeats = 20000 / panels;
    auto mixed_fn = [&]() {
        for (int p = 0; p < panels; ++p) {
            panel_mixed<NR>(nb, w.data() + (size_t) p*nb*UDNL_MX_PB, act.data(), anb, rec.data(),
                            s_mix.data() + 16*p, nc);
        }
    };
    auto fast_fn = [&]() {
        for (int p = 0; p < panels; ++p) {
            panel_w3_fast<NR>(nb, w.data() + (size_t) p*nb*UDNL_MX_PB, act.data(), anb, rec.data(),
                              s_fast.data() + 16*p, nc);
        }
    };
    const double t_mix = best_time(mixed_fn, repeats);
    const double t_fast = best_time(fast_fn, repeats);
    printf("NR=%d nb=%d panels=%d bitexact=%s mismatches=%llu mixed=%.3f us fast=%.3f us speedup=%.4fx (%+.2f%%)\n",
           NR, nb, panels, mismatch == 0 ? "PASS" : "FAIL",
           (unsigned long long) mismatch, t_mix*1e6, t_fast*1e6,
           t_mix / t_fast, 100.0*(t_mix / t_fast - 1.0));
}

int main() {
    run_case<1>(16, 64);
    run_case<1>(16, 256);
    run_case<4>(16, 64);
    run_case<8>(16, 64);
    run_case<1>(64, 64);
    return 0;
}
