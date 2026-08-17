#if defined(__GNUC__)
#pragma GCC diagnostic ignored "-Wpedantic"
#pragma GCC diagnostic ignored "-Wunused-local-typedefs"
#endif

#include "amx.h"
#include "mmq.h"
#include "ggml-impl.h"
#include "ggml-cpu-impl.h"
#include "simd-mappings.h"
#include "quants.h"
#include "ggml-quants.h"

// grid/sign LUTs (iq2xxs_grid, iq2xs_grid, iq3xxs_grid, ksigns_iq2xs, kmask_iq2xs)
// for the pack-time decode of the lookup-based quants (same mechanism as repack.cpp)
#define GGML_COMMON_IMPL_CPP
#include "ggml-common.h"

#include <algorithm>
#include <type_traits>

#if defined(__gnu_linux__)
#include <sys/syscall.h>
#include <unistd.h>
#endif

#if (defined(_WIN32) || defined(_WIN64))
#define RESTRICT __restrict
#else
#define RESTRICT __restrict__
#endif

#if (defined(_WIN32) || defined(_WIN64))
#define ALWAYS_INLINE __forceinline
#elif __has_attribute(always_inline) || defined(__GNUC__)
#define ALWAYS_INLINE __attribute__((__always_inline__)) inline
#else
#define ALWAYS_INLINE inline
#endif

#if defined(__AMX_INT8__) && defined(__AVX512VNNI__)

namespace {

// Forced unrolling
template <int n>
struct Unroll {
    template <typename Func, typename... Args>
    ALWAYS_INLINE void operator()(const Func& f, Args... args) const {
        Unroll<n - 1>{}(f, args...);
        f(std::integral_constant<int, n - 1>{}, args...);
    }
};

template <>
struct Unroll<1> {
    template <typename Func, typename... Args>
    ALWAYS_INLINE void operator()(const Func& f, Args... args) const {
        f(std::integral_constant<int, 0>{}, args...);
    }
};

// type traits
template <typename T> struct PackedTypes {};
template <> struct PackedTypes<block_q4_0> { using type = int8_t; };
template <> struct PackedTypes<block_q4_1> { using type = uint8_t; };
template <> struct PackedTypes<block_q8_0> { using type = int8_t; };
template <> struct PackedTypes<block_mxfp4> { using type = int8_t; };
template <> struct PackedTypes<block_q2_K> { using type = int8_t; };
template <> struct PackedTypes<block_iq2_xxs> { using type = int8_t; };
template <> struct PackedTypes<block_iq2_xs> { using type = int8_t; };
template <> struct PackedTypes<block_iq3_xxs> { using type = int8_t; };
template <typename T> using packed_B_type = typename PackedTypes<T>::type;

template <typename T>
struct do_compensate : std::integral_constant<bool,
    std::is_same<T, block_q8_0>::value> {};

template <typename T>
struct do_unpack : std::integral_constant<bool,
    std::is_same<T, block_q4_0>::value ||
    std::is_same<T, block_q4_1>::value ||
    std::is_same<T, block_mxfp4>::value> {};

template <typename T>
struct is_type_qkk : std::integral_constant<bool,
    std::is_same<T, block_q4_K>::value ||
    std::is_same<T, block_q5_K>::value ||
    std::is_same<T, block_q6_K>::value ||
    std::is_same<T, block_iq4_xs>::value ||
    std::is_same<T, block_q2_K>::value ||
    std::is_same<T, block_iq2_xxs>::value ||
    std::is_same<T, block_iq2_xs>::value ||
    std::is_same<T, block_iq3_xxs>::value> {};

// qkk kernel granularity: these types have one scale per 16 values
// (the others have one scale per 32 values)
template <typename T>
struct qkk_group_16 : std::integral_constant<bool,
    std::is_same<T, block_q6_K>::value ||
    std::is_same<T, block_q2_K>::value ||
    std::is_same<T, block_iq2_xs>::value> {};

#define GGML_DISPATCH_FLOATING_TYPES(TYPE, ...)                                        \
    [&] {                                                                              \
        switch (TYPE) {                                                                \
            case GGML_TYPE_F16: {                                                      \
                using type = ggml_fp16_t;                                              \
                constexpr int blck_size = 16;                                          \
                return __VA_ARGS__();                                                  \
            }                                                                          \
            case GGML_TYPE_BF16: {                                                     \
                using type = ggml_bf16_t;                                              \
                constexpr int blck_size = 32;                                          \
                return __VA_ARGS__();                                                  \
            }                                                                          \
            default:                                                                   \
                fprintf(stderr, "Unsupported floating data type\n");                   \
        }                                                                              \
    }()

#define GGML_DISPATCH_QTYPES(QT, ...)                                                  \
    [&] {                                                                              \
        switch (QT) {                                                                  \
            case GGML_TYPE_Q4_0: {                                                     \
                using type = block_q4_0;                                               \
                using vec_dot_type = block_q8_0;                                       \
                constexpr int blck_size = QK4_0;                                       \
                return __VA_ARGS__();                                                  \
            }                                                                          \
            case GGML_TYPE_Q4_1: {                                                     \
                using type = block_q4_1;                                               \
                using vec_dot_type = block_q8_1;                                       \
                constexpr int blck_size = QK4_1;                                       \
                return __VA_ARGS__();                                                  \
            }                                                                          \
            case GGML_TYPE_Q8_0: {                                                     \
                using type = block_q8_0;                                               \
                using vec_dot_type = block_q8_0;                                       \
                constexpr int blck_size = QK8_0;                                       \
                return __VA_ARGS__();                                                  \
            }                                                                          \
            case GGML_TYPE_MXFP4: {                                                    \
                using type = block_mxfp4;                                              \
                using vec_dot_type = block_q8_0;                                       \
                constexpr int blck_size = QK_MXFP4;                                    \
                return __VA_ARGS__();                                                  \
            }                                                                          \
            case GGML_TYPE_Q4_K: {                                                     \
                using type = block_q4_K;                                               \
                using vec_dot_type = block_q8_K;                                       \
                constexpr int blck_size = QK_K;                                        \
                return __VA_ARGS__();                                                  \
            }                                                                          \
            case GGML_TYPE_Q5_K: {                                                     \
                using type = block_q5_K;                                               \
                using vec_dot_type = block_q8_K;                                       \
                constexpr int blck_size = QK_K;                                        \
                return __VA_ARGS__();                                                  \
            }                                                                          \
            case GGML_TYPE_Q6_K: {                                                     \
                using type = block_q6_K;                                               \
                using vec_dot_type = block_q8_K;                                       \
                constexpr int blck_size = QK_K;                                        \
                return __VA_ARGS__();                                                  \
            }                                                                          \
            case GGML_TYPE_IQ4_XS: {                                                   \
                using type = block_iq4_xs;                                             \
                using vec_dot_type = block_q8_K;                                       \
                constexpr int blck_size = QK_K;                                        \
                return __VA_ARGS__();                                                  \
            }                                                                          \
            case GGML_TYPE_Q2_K: {                                                     \
                using type = block_q2_K;                                               \
                using vec_dot_type = block_q8_K;                                       \
                constexpr int blck_size = QK_K;                                        \
                return __VA_ARGS__();                                                  \
            }                                                                          \
            case GGML_TYPE_IQ2_XXS: {                                                  \
                using type = block_iq2_xxs;                                            \
                using vec_dot_type = block_q8_K;                                       \
                constexpr int blck_size = QK_K;                                        \
                return __VA_ARGS__();                                                  \
            }                                                                          \
            case GGML_TYPE_IQ2_XS: {                                                   \
                using type = block_iq2_xs;                                             \
                using vec_dot_type = block_q8_K;                                       \
                constexpr int blck_size = QK_K;                                        \
                return __VA_ARGS__();                                                  \
            }                                                                          \
            case GGML_TYPE_IQ3_XXS: {                                                  \
                using type = block_iq3_xxs;                                            \
                using vec_dot_type = block_q8_K;                                       \
                constexpr int blck_size = QK_K;                                        \
                return __VA_ARGS__();                                                  \
            }                                                                          \
            default:                                                                   \
                fprintf(stderr, "Unsupported quantized data type: %d\n", int(TYPE));   \
        }                                                                              \
    }()

#define GGML_DISPATCH_BOOL(BOOL_V, BOOL_NAME, ...)                                     \
    [&] {                                                                              \
        if (BOOL_V) {                                                                  \
            constexpr bool BOOL_NAME = true;                                           \
            return __VA_ARGS__();                                                      \
        } else {                                                                       \
            constexpr bool BOOL_NAME = false;                                          \
            return __VA_ARGS__();                                                      \
        }                                                                              \
    }()

// define amx tile config data structure
struct tile_config_t{
    uint8_t palette_id = 0;
    uint8_t start_row = 0;
    uint8_t reserved_0[14] = {0};
    uint16_t colsb[16] = {0};
    uint8_t rows[16] = {0};
};

// Notes: amx tile config
//
// Typically, TMUL calculates A and B of size 16 x 64 containing INT8 values,
// and accumulate the result to a 16 x 16 matrix C containing INT32 values,
//
// As many GGUF quantized types as `block_size` of 32, so a 16-16-32 config is used
// instead of the normally used 16-16-64 config.
//
//    Block A: {16, 32}, dtype = int8_t
//    Block B: {16, 32}, dtype = uint8_t/int8_t
//    Block C: {16, 16}, dtype = int32_t
//
// Block B needs to be prepacked to vnni format before feeding into  TMUL:
//    packed_B: from {n, k} to {k/vnni_blk, n, vnni_blck}, viewed in 2d, we get {8, 64}
//
// Therefore, we get tileconfig:
//             A    B    C
//    rows    16    8   16
//    colsb   32   64   16
//
// For tile distribution, follow a 2-2-4 pattern, e.g. A used TMM2-TMM3, B used TMM0-TMM1,
// C used TMM4-TMM7:
//            B TMM0  B TMM1
//    A TMM2  C TMM4  C TMM6
//    A TMM3  C TMM5  C TMM7
//
// Each `amx` kernel handles 4 blocks at a time: 2MB * 2NB, when m < 2 * BLOCK_M, unpack A
// will be needed.
//
// Here another commonly used pattern 1-3-3 is skipped, as it is mostly used when m <=16;
// and the single batch gemm (m=1) has a special fast path with `avx512-vnni`.
//
// ref: https://www.intel.com/content/www/us/en/developer/articles/code-sample/
//    advanced-matrix-extensions-intrinsics-functions.html
//

inline void ggml_tile_config_init(void) {
    static thread_local bool done = false;

    if (done) {
        return;
    }

    alignas(64) tile_config_t tc = {};
    tc.palette_id = 1;
    tc.start_row = 0;
    tc.rows[0] = 8;   tc.colsb[0] = 64;
    tc.rows[1] = 8;   tc.colsb[1] = 64;
    tc.rows[2] = 16;  tc.colsb[2] = 32;
    tc.rows[3] = 16;  tc.colsb[3] = 32;
    tc.rows[4] = 16;  tc.colsb[4] = 64;
    tc.rows[5] = 16;  tc.colsb[5] = 64;
    tc.rows[6] = 16;  tc.colsb[6] = 64;
    tc.rows[7] = 16;  tc.colsb[7] = 64;

    _tile_loadconfig(&tc);
    done = true;
}

// we need an extra 16 * 4B (TILE_N * int32_t) for each NB/KB block for compensation.
// See the notes `s8s8 igemm compensation in avx512-vnni` for detail.
//
// The lookup-based types (IQ2_XXS/IQ2_XS/IQ3_XXS) and Q2_K decode quants to 8-bit
// at pack time (see pack_B below), so their tile size is a fixed custom layout
// rather than sizeof(TB) + extras:
//   Q2_K:     quants {QK_K} int8 + scales {16} int8 + mins {16} int8 + d/dmin fp16
//   IQ2_XS:   quants {QK_K} int8 + scales {16} int8 + d fp16
//   IQ2_XXS / IQ3_XXS: quants {QK_K} int8 + scales {8} int8 + d fp16
template <typename TB>
int get_tile_size() {
    if (std::is_same<TB, block_q2_K>::value) {
        return TILE_N * (QK_K + 16 + 16 + 2 + 2);
    }
    if (std::is_same<TB, block_iq2_xs>::value) {
        return TILE_N * (QK_K + 16 + 2);
    }
    if (std::is_same<TB, block_iq2_xxs>::value ||
        std::is_same<TB, block_iq3_xxs>::value) {
        return TILE_N * (QK_K + 8 + 2);
    }
    int tile_size = TILE_N * sizeof(TB);
    if (do_compensate<TB>::value) {
        tile_size += TILE_N * sizeof(int32_t);
    }
    if (std::is_same<TB, block_q4_K>::value ||
        std::is_same<TB, block_q5_K>::value) {
        tile_size += TILE_N * 4;
    }
    if (std::is_same<TB, block_iq4_xs>::value) {
        tile_size += TILE_N * 2;
    }
    return tile_size;
}

template <typename TB, int BLOCK_K>
int get_row_size(int K) {
    int KB = K / BLOCK_K;
    if (std::is_same<TB, block_q2_K>::value) {
        return KB * (QK_K + 16 + 16 + 2 + 2);
    }
    if (std::is_same<TB, block_iq2_xs>::value) {
        return KB * (QK_K + 16 + 2);
    }
    if (std::is_same<TB, block_iq2_xxs>::value ||
        std::is_same<TB, block_iq3_xxs>::value) {
        return KB * (QK_K + 8 + 2);
    }
    int row_size = KB * sizeof(TB);
    if (do_compensate<TB>::value) {
        row_size += KB * sizeof(int32_t);
    }
    if (std::is_same<TB, block_q4_K>::value ||
        std::is_same<TB, block_q5_K>::value) {
        row_size += KB * 4;
    }
    if (std::is_same<TB, block_iq4_xs>::value) {
        row_size += KB * 2;
    }
    return row_size;
}

// transpose utils
#define SHUFFLE_EPI32(a, b, mask) \
    _mm256_castps_si256(_mm256_shuffle_ps(_mm256_castsi256_ps(a), _mm256_castsi256_ps(b), mask))
inline void transpose_8x8_32bit(__m256i * v, __m256i * v1) {
    // unpacking and 32-bit elements
    v1[0] = _mm256_unpacklo_epi32(v[0], v[1]);
    v1[1] = _mm256_unpackhi_epi32(v[0], v[1]);
    v1[2] = _mm256_unpacklo_epi32(v[2], v[3]);
    v1[3] = _mm256_unpackhi_epi32(v[2], v[3]);
    v1[4] = _mm256_unpacklo_epi32(v[4], v[5]);
    v1[5] = _mm256_unpackhi_epi32(v[4], v[5]);
    v1[6] = _mm256_unpacklo_epi32(v[6], v[7]);
    v1[7] = _mm256_unpackhi_epi32(v[6], v[7]);

    // shuffling the 32-bit elements
    v[0] = SHUFFLE_EPI32(v1[0], v1[2], 0x44);
    v[1] = SHUFFLE_EPI32(v1[0], v1[2], 0xee);
    v[2] = SHUFFLE_EPI32(v1[4], v1[6], 0x44);
    v[3] = SHUFFLE_EPI32(v1[4], v1[6], 0xee);
    v[4] = SHUFFLE_EPI32(v1[1], v1[3], 0x44);
    v[5] = SHUFFLE_EPI32(v1[1], v1[3], 0xee);
    v[6] = SHUFFLE_EPI32(v1[5], v1[7], 0x44);
    v[7] = SHUFFLE_EPI32(v1[5], v1[7], 0xee);

    // shuffling 128-bit elements
    v1[0] = _mm256_permute2f128_si256(v[2], v[0], 0x02);
    v1[1] = _mm256_permute2f128_si256(v[3], v[1], 0x02);
    v1[2] = _mm256_permute2f128_si256(v[6], v[4], 0x02);
    v1[3] = _mm256_permute2f128_si256(v[7], v[5], 0x02);
    v1[4] = _mm256_permute2f128_si256(v[2], v[0], 0x13);
    v1[5] = _mm256_permute2f128_si256(v[3], v[1], 0x13);
    v1[6] = _mm256_permute2f128_si256(v[6], v[4], 0x13);
    v1[7] = _mm256_permute2f128_si256(v[7], v[5], 0x13);
}

inline void transpose_16x4_32bit(__m512i * r, __m512i * d) {

    static const __m512i index1 = _mm512_set_epi32(
        0x0f, 0x0b, 0x07, 0x03,
        0x0e, 0x0a, 0x06, 0x02,
        0x0d, 0x09, 0x05, 0x01,
        0x0c, 0x08, 0x04, 0x00);

    d[0] = _mm512_permutexvar_epi32(index1, r[0]);
    d[1] = _mm512_permutexvar_epi32(index1, r[1]);
    d[2] = _mm512_permutexvar_epi32(index1, r[2]);
    d[3] = _mm512_permutexvar_epi32(index1, r[3]);

    r[0] = _mm512_shuffle_i32x4(d[0], d[1], 0x44);
    r[1] = _mm512_shuffle_i32x4(d[0], d[1], 0xee);
    r[2] = _mm512_shuffle_i32x4(d[2], d[3], 0x44);
    r[3] = _mm512_shuffle_i32x4(d[2], d[3], 0xee);

    d[0] = _mm512_shuffle_i32x4(r[0], r[2], 0x88);
    d[1] = _mm512_shuffle_i32x4(r[0], r[2], 0xdd);
    d[2] = _mm512_shuffle_i32x4(r[1], r[3], 0x88);
    d[3] = _mm512_shuffle_i32x4(r[1], r[3], 0xdd);
}

inline void transpose_16x16_32bit(__m512i * v) {
    __m512i v1[16];
    v1[0] = _mm512_unpacklo_epi32(v[0], v[1]);
    v1[1] = _mm512_unpackhi_epi32(v[0], v[1]);
    v1[2] = _mm512_unpacklo_epi32(v[2], v[3]);
    v1[3] = _mm512_unpackhi_epi32(v[2], v[3]);
    v1[4] = _mm512_unpacklo_epi32(v[4], v[5]);
    v1[5] = _mm512_unpackhi_epi32(v[4], v[5]);
    v1[6] = _mm512_unpacklo_epi32(v[6], v[7]);
    v1[7] = _mm512_unpackhi_epi32(v[6], v[7]);
    v1[8] = _mm512_unpacklo_epi32(v[8], v[9]);
    v1[9] = _mm512_unpackhi_epi32(v[8], v[9]);
    v1[10] = _mm512_unpacklo_epi32(v[10], v[11]);
    v1[11] = _mm512_unpackhi_epi32(v[10], v[11]);
    v1[12] = _mm512_unpacklo_epi32(v[12], v[13]);
    v1[13] = _mm512_unpackhi_epi32(v[12], v[13]);
    v1[14] = _mm512_unpacklo_epi32(v[14], v[15]);
    v1[15] = _mm512_unpackhi_epi32(v[14], v[15]);

    v[0] = _mm512_unpacklo_epi64(v1[0], v1[2]);
    v[1] = _mm512_unpackhi_epi64(v1[0], v1[2]);
    v[2] = _mm512_unpacklo_epi64(v1[1], v1[3]);
    v[3] = _mm512_unpackhi_epi64(v1[1], v1[3]);
    v[4] = _mm512_unpacklo_epi64(v1[4], v1[6]);
    v[5] = _mm512_unpackhi_epi64(v1[4], v1[6]);
    v[6] = _mm512_unpacklo_epi64(v1[5], v1[7]);
    v[7] = _mm512_unpackhi_epi64(v1[5], v1[7]);
    v[8] = _mm512_unpacklo_epi64(v1[8], v1[10]);
    v[9] = _mm512_unpackhi_epi64(v1[8], v1[10]);
    v[10] = _mm512_unpacklo_epi64(v1[9], v1[11]);
    v[11] = _mm512_unpackhi_epi64(v1[9], v1[11]);
    v[12] = _mm512_unpacklo_epi64(v1[12], v1[14]);
    v[13] = _mm512_unpackhi_epi64(v1[12], v1[14]);
    v[14] = _mm512_unpacklo_epi64(v1[13], v1[15]);
    v[15] = _mm512_unpackhi_epi64(v1[13], v1[15]);

    v1[0] = _mm512_shuffle_i32x4(v[0], v[4], 0x88);
    v1[1] = _mm512_shuffle_i32x4(v[1], v[5], 0x88);
    v1[2] = _mm512_shuffle_i32x4(v[2], v[6], 0x88);
    v1[3] = _mm512_shuffle_i32x4(v[3], v[7], 0x88);
    v1[4] = _mm512_shuffle_i32x4(v[0], v[4], 0xdd);
    v1[5] = _mm512_shuffle_i32x4(v[1], v[5], 0xdd);
    v1[6] = _mm512_shuffle_i32x4(v[2], v[6], 0xdd);
    v1[7] = _mm512_shuffle_i32x4(v[3], v[7], 0xdd);
    v1[8] = _mm512_shuffle_i32x4(v[8], v[12], 0x88);
    v1[9] = _mm512_shuffle_i32x4(v[9], v[13], 0x88);
    v1[10] = _mm512_shuffle_i32x4(v[10], v[14], 0x88);
    v1[11] = _mm512_shuffle_i32x4(v[11], v[15], 0x88);
    v1[12] = _mm512_shuffle_i32x4(v[8], v[12], 0xdd);
    v1[13] = _mm512_shuffle_i32x4(v[9], v[13], 0xdd);
    v1[14] = _mm512_shuffle_i32x4(v[10], v[14], 0xdd);
    v1[15] = _mm512_shuffle_i32x4(v[11], v[15], 0xdd);

    v[0] = _mm512_shuffle_i32x4(v1[0], v1[8], 0x88);
    v[1] = _mm512_shuffle_i32x4(v1[1], v1[9], 0x88);
    v[2] = _mm512_shuffle_i32x4(v1[2], v1[10], 0x88);
    v[3] = _mm512_shuffle_i32x4(v1[3], v1[11], 0x88);
    v[4] = _mm512_shuffle_i32x4(v1[4], v1[12], 0x88);
    v[5] = _mm512_shuffle_i32x4(v1[5], v1[13], 0x88);
    v[6] = _mm512_shuffle_i32x4(v1[6], v1[14], 0x88);
    v[7] = _mm512_shuffle_i32x4(v1[7], v1[15], 0x88);
    v[8] = _mm512_shuffle_i32x4(v1[0], v1[8], 0xdd);
    v[9] = _mm512_shuffle_i32x4(v1[1], v1[9], 0xdd);
    v[10] = _mm512_shuffle_i32x4(v1[2], v1[10], 0xdd);
    v[11] = _mm512_shuffle_i32x4(v1[3], v1[11], 0xdd);
    v[12] = _mm512_shuffle_i32x4(v1[4], v1[12], 0xdd);
    v[13] = _mm512_shuffle_i32x4(v1[5], v1[13], 0xdd);
    v[14] = _mm512_shuffle_i32x4(v1[6], v1[14], 0xdd);
    v[15] = _mm512_shuffle_i32x4(v1[7], v1[15], 0xdd);
}

void quantize_row_q8_K_vnni(const float * RESTRICT x, void * RESTRICT vy, int64_t k) {
    assert(k % QK_K == 0);
    const int KB = k / QK_K;
    constexpr int kVecs = QK_K / 16;

    block_q8_K * y = reinterpret_cast<block_q8_K *>(vy);

    // hold 16 float vecs from x
    __m512  v[kVecs];

    // hold the quants vecs
    __m512i vq[kVecs / 4];

    // hold the packed quants vecs
    __m512i vq_packed[kVecs / 4];

    const __m512 signBit = _mm512_set1_ps(-0.f);

    for (int i = 0; i < KB; ++i) {
        // Compute max(abs(e)) for the block
        __m512 vamax = _mm512_set1_ps(0.f);
        for (int j = 0; j < kVecs; ++j) {
            v[j] = _mm512_loadu_ps(x); x += 16;
            vamax = _mm512_max_ps(vamax, _mm512_andnot_ps(signBit, v[j]));
        }
        const float amax = _mm512_reduce_max_ps(vamax);

        // Quantize these floats
        const float iscale = 127.f / amax;
        y[i].d = GGML_CPU_FP32_TO_FP16(1 / iscale);
        const float id = ( amax != 0.0f ) ? iscale : 0.f;
        const __m512 vscale = _mm512_set1_ps(id);

        // Apply multiplier and round to nearest integer
        for (int j = 0; j < kVecs; ++j) {
            v[j] = _mm512_mul_ps(v[j], vscale);
            v[j] = _mm512_roundscale_ps(v[j], (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
        }

        // Pack to epi8 vecs
        for (int j = 0; j < kVecs / 4; ++j) {
            __m128i q8_0 = _mm512_cvtepi32_epi8(_mm512_cvtps_epi32(v[j * 4 + 0]));
            __m128i q8_1 = _mm512_cvtepi32_epi8(_mm512_cvtps_epi32(v[j * 4 + 1]));
            __m128i q8_2 = _mm512_cvtepi32_epi8(_mm512_cvtps_epi32(v[j * 4 + 2]));
            __m128i q8_3 = _mm512_cvtepi32_epi8(_mm512_cvtps_epi32(v[j * 4 + 3]));

            __m256i q8_01 = _mm256_insertf128_si256(_mm256_castsi128_si256(q8_0), (q8_1), 1);
            __m256i q8_23 = _mm256_insertf128_si256(_mm256_castsi128_si256(q8_2), (q8_3), 1);

            vq[j] = _mm512_inserti32x8(_mm512_castsi256_si512(q8_01), q8_23, 1);
            _mm512_storeu_si512((__m512i *)(y[i].qs + j * 64), vq[j]);
        }

        // Compute the bsums with vnni
        transpose_16x4_32bit(vq, vq_packed);

        const __m512i one = _mm512_set1_epi8(1);
        __m512i sum = _mm512_setzero_si512();
        for (int k = 0; k < 4; ++k) {
            sum = _mm512_dpbusd_epi32(sum, one, vq_packed[k]);
        }
        _mm256_storeu_si256((__m256i *)(y[i].bsums), _mm512_cvtepi32_epi16(sum));
    }
}

// quantize A from float to `vec_dot_type`
template <typename T>
inline void from_float(const float * x, char * vy, int64_t k);

template <>
inline void from_float<block_q8_0>(const float * x, char * vy, int64_t k) {
    quantize_row_q8_0(x, (block_q8_0 *)vy, k);
}

template <>
inline void from_float<block_q8_1>(const float * x, char * vy, int64_t k) {
    quantize_row_q8_1(x, (block_q8_1 *)vy, k);
}

template <>
inline void from_float<block_q8_K>(const float * x, char * vy, int64_t k) {
#if 1
    // TODO: this is reference impl!
    quantize_row_q8_K_ref(x, (block_q8_K *)vy, k);
#else
    quantize_row_q8_K_vnni(x, vy, k);
#endif
}

// load A from memory to array when nrows can not fill in whole tile
void unpack_A(int8_t * RESTRICT tile, const block_q8_0 * RESTRICT A, int lda, int nr) {
    assert(nr != TILE_M);
    for (int m = 0; m < nr; ++m) {
        const __m256i v = _mm256_loadu_si256((const __m256i *)(A[m * lda].qs));
        _mm256_storeu_si256((__m256i *)(tile + m * TILE_K), v);
    }
}

void unpack_A(int8_t * RESTRICT tile, const block_q8_1 * RESTRICT A, int lda, int nr) {
    assert(nr != TILE_M);
    for (int m = 0; m < nr; ++m) {
        const __m256i v = _mm256_loadu_si256((const __m256i *)(A[m * lda].qs));
        _mm256_storeu_si256((__m256i *)(tile + m * TILE_K), v);
    }
}

template <typename TB>
void unpack_A(int8_t * RESTRICT tile, const block_q8_K * RESTRICT A, int lda, int k, int nr) {
    assert(nr <= TILE_M);
    for (int m = 0; m < nr; ++m) {
        const __m256i v = _mm256_loadu_si256((const __m256i *)(A[m * lda].qs + k * 32));
        _mm256_storeu_si256((__m256i *)(tile + m * TILE_K), v);
    }
}

template <>
void unpack_A<block_q6_K>(int8_t * RESTRICT tile, const block_q8_K * RESTRICT A, int lda, int k, int nr) {
    assert(nr <= TILE_M);
    // zero padding k from 16 to 32, so that we don't have to re-config amx
    const __m128i zero = _mm_setzero_si128();
    for (int m = 0; m < nr; ++m) {
        const __m128i v = _mm_loadu_si128((const __m128i *)(A[m * lda].qs + k * 16));
        const __m256i r = _mm256_insertf128_si256(_mm256_castsi128_si256(v), zero, 1);
        _mm256_storeu_si256((__m256i *)(tile + m * TILE_K), r);
    }
}

template <>
void unpack_A<block_q2_K>(int8_t * RESTRICT tile, const block_q8_K * RESTRICT A, int lda, int k, int nr) {
    unpack_A<block_q6_K>(tile, A, lda, k, nr);
}

template <>
void unpack_A<block_iq2_xs>(int8_t * RESTRICT tile, const block_q8_K * RESTRICT A, int lda, int k, int nr) {
    unpack_A<block_q6_K>(tile, A, lda, k, nr);
}

#define MM256_SET_M128I(a, b) _mm256_insertf128_si256(_mm256_castsi128_si256(b), (a), 1)
inline __m256i bytes_from_nibbles_32(const uint8_t * rsi) {
    const __m128i tmp = _mm_loadu_si128((const __m128i *)rsi);
    const __m256i bytes = MM256_SET_M128I(_mm_srli_epi16(tmp, 4), tmp);
    const __m256i lowMask = _mm256_set1_epi8(0xF);
    return _mm256_and_si256(lowMask, bytes);
}

// used for block_q4_K
inline __m512i bytes_from_nibbles_64(const uint8_t * rsi) {
    const __m256i tmp = _mm256_loadu_si256((const __m256i *)rsi);
    const __m256i lowMask = _mm256_set1_epi8(0xF);
    const __m256i q4l = _mm256_and_si256(tmp, lowMask);
    const __m256i q4h = _mm256_and_si256(_mm256_srli_epi16(tmp, 4), lowMask);
    return _mm512_inserti32x8(_mm512_castsi256_si512(q4l), q4h, 1);
}

// used for block_q5_K
inline __m512i bytes_from_nibbles_64(const uint8_t * qs, const uint8_t * qh, int k) {
    const __m256i lowMask = _mm256_set1_epi8(0xF);
    __m256i hmask = _mm256_set1_epi8(1);
    hmask = _mm256_slli_epi16(hmask, k);

    const __m256i q5bits = _mm256_loadu_si256((const __m256i *)qs);
    const __m256i hbits = _mm256_loadu_si256((const __m256i *)qh);

    const __m256i q5l_0 = _mm256_and_si256(q5bits, lowMask);
    const __m256i q5h_0 = _mm256_slli_epi16(_mm256_srli_epi16(_mm256_and_si256(hbits, hmask), k + 0), 4);
    const __m256i q5_0  = _mm256_add_epi8(q5l_0, q5h_0);
    hmask = _mm256_slli_epi16(hmask, 1);

    const __m256i q5l_1 = _mm256_and_si256(_mm256_srli_epi16(q5bits, 4), lowMask);
    const __m256i q5h_1 = _mm256_slli_epi16(_mm256_srli_epi16(_mm256_and_si256(hbits, hmask), k + 1), 4);
    const __m256i q5_1  = _mm256_add_epi8(q5l_1, q5h_1);

    return _mm512_inserti32x8(_mm512_castsi256_si512(q5_0), q5_1, 1);
}

// used for block_q6_K
inline void bytes_from_nibbles_128(__m512i& r0, __m512i& r1, const uint8_t * qs, const uint8_t * qh) {
    const __m256i m4 = _mm256_set1_epi8(0xF);
    const __m256i m2 = _mm256_set1_epi8(0x3);

    const __m256i q6bits1 = _mm256_loadu_si256((const __m256i *)qs);
    const __m256i q6bits2 = _mm256_loadu_si256((const __m256i *)(qs + 32));
    const __m256i q6bitsH = _mm256_loadu_si256((const __m256i *)qh);

    const __m256i q6h_0 = _mm256_slli_epi16(_mm256_and_si256(                  q6bitsH,     m2), 4);
    const __m256i q6h_1 = _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(q6bitsH, 2), m2), 4);
    const __m256i q6h_2 = _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(q6bitsH, 4), m2), 4);
    const __m256i q6h_3 = _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(q6bitsH, 6), m2), 4);

    const __m256i q6_0 = _mm256_or_si256(_mm256_and_si256(q6bits1, m4), q6h_0);
    const __m256i q6_1 = _mm256_or_si256(_mm256_and_si256(q6bits2, m4), q6h_1);
    const __m256i q6_2 = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(q6bits1, 4), m4), q6h_2);
    const __m256i q6_3 = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(q6bits2, 4), m4), q6h_3);

    r0 = _mm512_inserti32x8(_mm512_castsi256_si512(q6_0), q6_1, 1);
    r1 = _mm512_inserti32x8(_mm512_castsi256_si512(q6_2), q6_3, 1);
}

inline __m512i packNibbles(__m512i r0, __m512i r1) {
    return _mm512_or_si512(r0, _mm512_slli_epi16(r1, 4));
}

template <typename TB>
inline void pack_qs(void * RESTRICT packed_B, const TB * RESTRICT B, int KB) {
    int8_t tmp[8 * 64];
    __m256i v[8], v2[8];
    for (int n = 0; n < 8; ++n) {
        v[n] = bytes_from_nibbles_32(B[n * KB].qs);
    }
    transpose_8x8_32bit(v, v2);
    for (int n = 0; n < 8; ++n) {
        _mm256_storeu_si256((__m256i *)(tmp + n * 64), v2[n]);
    }
    for (int n = 0; n < 8; ++n) {
        v[n] = bytes_from_nibbles_32(B[(n + 8) * KB].qs);
    }
    transpose_8x8_32bit(v, v2);
    for (int n = 0; n < 8; ++n) {
        _mm256_storeu_si256((__m256i *)(tmp + n * 64 + 32), v2[n]);
    }

    // pack again with 128 to fully utilize vector length
    for (int n = 0; n < 8; n += 2) {
        __m512i r0 = _mm512_loadu_si512((const __m512i *)(tmp + n * 64));
        __m512i r1 = _mm512_loadu_si512((const __m512i *)(tmp + n * 64 + 64));
        __m512i r1r0 = packNibbles(r0, r1);
        _mm512_storeu_si512((__m512i *)((char *)packed_B + n * 32), r1r0);
    }
}

template <>
inline void pack_qs<block_q8_0>(void * RESTRICT packed_B, const block_q8_0 * RESTRICT B, int KB) {
    __m256i v[8], v2[8];
    for (int n = 0; n < 8; ++n) {
        v[n] = _mm256_loadu_si256((const __m256i *)(B[n * KB].qs));
    }
    transpose_8x8_32bit(v, v2);
    for (int n = 0; n < 8; ++n) {
        _mm256_storeu_si256((__m256i *)((char *)packed_B + n * 64), v2[n]);
    }
    for (int n = 0; n < 8; ++n) {
        v[n] = _mm256_loadu_si256((const __m256i *)(B[(n + 8) * KB].qs));
    }
    transpose_8x8_32bit(v, v2);
    for (int n = 0; n < 8; ++n) {
        _mm256_storeu_si256((__m256i *)((char *)packed_B + n * 64 + 32), v2[n]);
    }
}

template <>
inline void pack_qs<block_q4_K>(void * RESTRICT packed_B, const block_q4_K * RESTRICT B, int KB) {
    __m512i v[16];
    // QK_K 256 with 8 groups, handle 2 groups at a time
    char * pb = (char *)packed_B;
    for (int k = 0; k < QK_K / 64; ++k) {
        // pack 2 groups { n, g,  k} to {g, k/4, 4n}
        //          e.g. {16, 2, 32} to {2,   8, 64}
        for (int n = 0; n < TILE_N; ++n) {
            v[n] = bytes_from_nibbles_64(B[n * KB].qs + k * 32);
        }

        transpose_16x16_32bit(v);

        // pack again with 128 to fully utilize vector length
        for (int n = 0; n < TILE_N; n += 2) {
            _mm512_storeu_si512((__m512i *)pb, packNibbles(v[n], v[n + 1]));
            pb += 64;
        }
    }
}

template <>
inline void pack_qs<block_q5_K>(void * RESTRICT packed_B, const block_q5_K * RESTRICT B, int KB) {
    __m512i v[16];
    const __m512i lowMask = _mm512_set1_epi8(0xF);
    // QK_K 256 with 8 groups, handle 2 groups at a time
    char * pb = (char *)packed_B;
    char * ph = (char *)packed_B + (QK_K / 2) * TILE_N;
    for (int k = 0; k < QK_K / 64; ++k) {
        // pack 2 groups { n, g,  k} to {g, k/4, 4n}
        //          e.g. {16, 2, 32} to {2,   8, 64}
        for (int n = 0; n < TILE_N; ++n) {
            v[n] = bytes_from_nibbles_64(B[n * KB].qs + k * 32, B[n * KB].qh, /* group */2 * k);
        }

        transpose_16x16_32bit(v);

        // 1. pack lower 4bits with 2 groups
        for (int n = 0; n < TILE_N; n += 2) {
            // get lower 4 bits
            const __m512i r0 = _mm512_and_si512(v[n], lowMask);
            const __m512i r1 = _mm512_and_si512(v[n + 1], lowMask);
            _mm512_storeu_si512((__m512i *)pb, packNibbles(r0, r1)); pb += 64;
        }

        // 2. pack higher 1bit with 2 groups
        const __m512i hmask = _mm512_set1_epi8(0x10);
        for (int g = 0; g < 2; ++g) {
            __m512i hbits = _mm512_setzero_si512();
            hbits = _mm512_add_epi8(hbits, _mm512_srli_epi16(_mm512_and_si512(v[g * 8 + 0], hmask), 4));
            hbits = _mm512_add_epi8(hbits, _mm512_srli_epi16(_mm512_and_si512(v[g * 8 + 1], hmask), 3));
            hbits = _mm512_add_epi8(hbits, _mm512_srli_epi16(_mm512_and_si512(v[g * 8 + 2], hmask), 2));
            hbits = _mm512_add_epi8(hbits, _mm512_srli_epi16(_mm512_and_si512(v[g * 8 + 3], hmask), 1));
            hbits = _mm512_add_epi8(hbits,                   _mm512_and_si512(v[g * 8 + 4], hmask)    );
            hbits = _mm512_add_epi8(hbits, _mm512_slli_epi16(_mm512_and_si512(v[g * 8 + 5], hmask), 1));
            hbits = _mm512_add_epi8(hbits, _mm512_slli_epi16(_mm512_and_si512(v[g * 8 + 6], hmask), 2));
            hbits = _mm512_add_epi8(hbits, _mm512_slli_epi16(_mm512_and_si512(v[g * 8 + 7], hmask), 3));
            _mm512_storeu_si512((__m512i *)ph, hbits); ph += 64;
        }
    }
}

template <>
inline void pack_qs<block_q6_K>(void * RESTRICT packed_B, const block_q6_K * RESTRICT B, int KB) {
    __m512i v[32];
    const __m512i lowMask = _mm512_set1_epi8(0xF);
    // QK_K 256 with 8 groups, handle 4 groups at a time
    char * pb = (char *)packed_B;
    char * ph = (char *)packed_B + (QK_K / 2) * TILE_N;
    for (int k = 0; k < QK_K / 128; ++k) {
        for (int n = 0; n < TILE_N; ++n) {
            bytes_from_nibbles_128(v[n], v[n + 16], B[n * KB].ql + k * 64, B[n * KB].qh + k * 32);
        }

        // top half: group 0,1 or 4,5; bottom half: group 2,3 or 6,7
        transpose_16x16_32bit(v);
        transpose_16x16_32bit(v + 16);

        // 1. pack lower 4bits with 4 groups
        for (int n = 0; n < 32; n += 2) {
            const __m512i r0 = _mm512_and_si512(v[n], lowMask);
            const __m512i r1 = _mm512_and_si512(v[n + 1], lowMask);
            _mm512_storeu_si512((__m512i *)pb, packNibbles(r0, r1)); pb += 64;
        }

        // 2. pack higher 2bit with 4 groups
        const __m512i hmask = _mm512_set1_epi8(0x30);
        for (int g = 0; g < 8; ++g) {
            __m512i hbits = _mm512_setzero_si512();
            hbits = _mm512_add_epi8(hbits, _mm512_srli_epi16(_mm512_and_si512(v[g * 4 + 0], hmask), 4));
            hbits = _mm512_add_epi8(hbits, _mm512_srli_epi16(_mm512_and_si512(v[g * 4 + 1], hmask), 2));
            hbits = _mm512_add_epi8(hbits,                   _mm512_and_si512(v[g * 4 + 2], hmask)    );
            hbits = _mm512_add_epi8(hbits, _mm512_slli_epi16(_mm512_and_si512(v[g * 4 + 3], hmask), 2));
            _mm512_storeu_si512((__m512i *)ph, hbits); ph += 64;
        }
    }
}

template <>
inline void pack_qs<block_iq4_xs>(void * RESTRICT packed_B, const block_iq4_xs * RESTRICT B, int KB) {
    __m512i v[16];
    char * pb = (char *)packed_B;
    for (int k = 0; k < QK_K / 64; ++k) {
        for (int n = 0; n < TILE_N; ++n) {
            __m256i r0 = bytes_from_nibbles_32(B[n * KB].qs + k * 32 +  0);
            __m256i r1 = bytes_from_nibbles_32(B[n * KB].qs + k * 32 + 16);
            v[n] = _mm512_inserti32x8(_mm512_castsi256_si512(r0), r1, 1);
        }

        transpose_16x16_32bit(v);

        // pack again with 128 to fully utilize vector length
        for (int n = 0; n < TILE_N; n += 2) {
            _mm512_storeu_si512((__m512i *)pb, packNibbles(v[n], v[n + 1]));
            pb += 64;
        }
    }
}

// ---------------------------------------------------------------------------
// pack-time decoders for the lookup-based types and Q2_K.
//
// The grid-LUT sign/index decode of IQ2_XXS/IQ2_XS/IQ3_XXS is too irregular to
// redo on every tile load, so we decode once at pack time into plain int8
// quants (values fit easily: |grid| <= 43) plus one int8 scale per group.
// Q2_K likewise unpacks its 2-bit quants to uint8 (0..3). The decoded quants
// are stored directly in the vnni tile layout, so unpack_B degenerates to a
// memcpy and the qkk AMX/VNNI kernels work unchanged.
// ---------------------------------------------------------------------------

// pack 16 rows x QK_K decoded int8 quants (row-major {TILE_N, QK_K}) into the
// vnni tile layout {8 k-groups of 32, 512B each} (same layout pack_qs<block_q8_0>
// produces per 32-value group)
inline void pack_qs_decoded(void * RESTRICT packed_B, const int8_t * RESTRICT dec) {
    char * pb = (char *)packed_B;
    for (int g = 0; g < QK_K / 32; ++g) {
        __m256i v[8], v2[8];
        for (int n = 0; n < 8; ++n) {
            v[n] = _mm256_loadu_si256((const __m256i *)(dec + n * QK_K + g * 32));
        }
        transpose_8x8_32bit(v, v2);
        for (int n = 0; n < 8; ++n) {
            _mm256_storeu_si256((__m256i *)(pb + n * 64), v2[n]);
        }
        for (int n = 0; n < 8; ++n) {
            v[n] = _mm256_loadu_si256((const __m256i *)(dec + (n + 8) * QK_K + g * 32));
        }
        transpose_8x8_32bit(v, v2);
        for (int n = 0; n < 8; ++n) {
            _mm256_storeu_si256((__m256i *)(pb + n * 64 + 32), v2[n]);
        }
        pb += 512;
    }
}

// one row: 8 groups of 32, one scale per group (matches ggml_vec_dot_iq2_xxs_q8_K_generic)
inline void decode_iq2_xxs_row(const block_iq2_xxs * RESTRICT x, int8_t * RESTRICT q8, int8_t * RESTRICT scales) {
    const uint16_t * q2 = x->qs;
    for (int ib32 = 0; ib32 < QK_K / 32; ++ib32) {
        uint32_t aux32[2];
        memcpy(aux32, q2, 2 * sizeof(uint32_t));
        q2 += 4;
        const uint8_t * aux8 = (const uint8_t *)aux32;
        scales[ib32] = 2 * (aux32[1] >> 28) + 1;
        int8_t * out = q8 + ib32 * 32;
        for (int l = 0; l < 4; ++l) {
            const uint8_t * grid = (const uint8_t *)(iq2xxs_grid + aux8[l]);
            const uint8_t  signs = ksigns_iq2xs[(aux32[1] >> 7 * l) & 127];
            for (int j = 0; j < 8; ++j) {
                out[l * 8 + j] = signs & kmask_iq2xs[j] ? -(int8_t)grid[j] : (int8_t)grid[j];
            }
        }
    }
}

// one row: 8 groups of 32, but one scale per 16 values (matches ggml_vec_dot_iq2_xs_q8_K_generic)
inline void decode_iq2_xs_row(const block_iq2_xs * RESTRICT x, int8_t * RESTRICT q8, int8_t * RESTRICT scales) {
    const uint16_t * q2 = x->qs;
    const uint8_t  * sc = x->scales;
    for (int ib32 = 0; ib32 < QK_K / 32; ++ib32) {
        scales[2 * ib32 + 0] = 2 * (sc[ib32] & 0xf) + 1;
        scales[2 * ib32 + 1] = 2 * (sc[ib32] >>  4) + 1;
        int8_t * out = q8 + ib32 * 32;
        for (int l = 0; l < 4; ++l) {
            const uint8_t * grid = (const uint8_t *)(iq2xs_grid + (q2[l] & 511));
            const uint8_t  signs = ksigns_iq2xs[q2[l] >> 9];
            for (int j = 0; j < 8; ++j) {
                out[l * 8 + j] = signs & kmask_iq2xs[j] ? -(int8_t)grid[j] : (int8_t)grid[j];
            }
        }
        q2 += 4;
    }
}

// one row: 8 groups of 32, one scale per group (matches ggml_vec_dot_iq3_xxs_q8_K_generic)
inline void decode_iq3_xxs_row(const block_iq3_xxs * RESTRICT x, int8_t * RESTRICT q8, int8_t * RESTRICT scales) {
    const uint8_t * q3  = x->qs;
    const uint8_t * gas = x->qs + QK_K / 4;
    for (int ib32 = 0; ib32 < QK_K / 32; ++ib32) {
        uint32_t aux32;
        memcpy(&aux32, gas, sizeof(uint32_t));
        gas += sizeof(uint32_t);
        scales[ib32] = 2 * (aux32 >> 28) + 1;
        int8_t * out = q8 + ib32 * 32;
        for (int l = 0; l < 4; ++l) {
            const uint8_t * grid1 = (const uint8_t *)(iq3xxs_grid + q3[2 * l + 0]);
            const uint8_t * grid2 = (const uint8_t *)(iq3xxs_grid + q3[2 * l + 1]);
            const uint8_t  signs = ksigns_iq2xs[(aux32 >> 7 * l) & 127];
            for (int j = 0; j < 4; ++j) {
                out[l * 8 + j + 0] = signs & kmask_iq2xs[j + 0] ? -(int8_t)grid1[j] : (int8_t)grid1[j];
                out[l * 8 + j + 4] = signs & kmask_iq2xs[j + 4] ? -(int8_t)grid2[j] : (int8_t)grid2[j];
            }
        }
        q3 += 8;
    }
}

// one row: 16 groups of 16, 4-bit scales and mins (matches ggml_vec_dot_q2_K_q8_K_generic)
inline void decode_q2_K_row(const block_q2_K * RESTRICT x, int8_t * RESTRICT q8,
                            int8_t * RESTRICT scales, int8_t * RESTRICT mins) {
    const uint8_t * q2 = x->qs;
    const uint8_t * sc = x->scales;
    for (int g = 0; g < QK_K / 16; ++g) {
        const int shift = 2 * ((g % 8) / 2);
        const uint8_t * p = q2 + (g / 8) * 32 + (g % 2) * 16;
        int8_t * out = q8 + g * 16;
        for (int l = 0; l < 16; ++l) {
            out[l] = (p[l] >> shift) & 3;
        }
        scales[g] = sc[g] & 0xF;
        mins[g]   = sc[g] >> 4;
    }
}

// pack B to vnni formats in 4bits or 8 bits
void pack_B(void * RESTRICT packed_B, const block_q4_0 * RESTRICT B, int KB) {
    pack_qs(packed_B, B, KB);
    ggml_half * d0 = reinterpret_cast<ggml_half *>((char *)packed_B + TILE_N * TILE_K / 2);
    for (int n = 0; n < TILE_N; ++n) {
        d0[n] = B[n * KB].d;
    }
}

void pack_B(void * RESTRICT packed_B, const block_q4_1 * RESTRICT B, int KB) {
    pack_qs(packed_B, B, KB);
    ggml_half * d0 = reinterpret_cast<ggml_half *>((char *)packed_B + TILE_N * TILE_K / 2);
    ggml_half * m0 = d0 + TILE_N;
    for (int n = 0; n < TILE_N; ++n) {
        d0[n] = B[n * KB].d;
        m0[n] = B[n * KB].m;
    }
}

// packed_B layout (same nibble order as block_q4_0, scales are raw E8M0 bytes):
//   quants {TILE_N, TILE_K/2}  uint8
//   e      {TILE_N}            uint8 (E8M0 scales)
void pack_B(void * RESTRICT packed_B, const block_mxfp4 * RESTRICT B, int KB) {
    pack_qs(packed_B, B, KB);
    uint8_t * e0 = reinterpret_cast<uint8_t *>((char *)packed_B + TILE_N * TILE_K / 2);
    for (int n = 0; n < TILE_N; ++n) {
        e0[n] = B[n * KB].e;
    }
}

// convert 16 raw E8M0 bytes to 16 fp32 values, matching ggml_e8m0_to_fp32_half:
//   e < 2  -> denormal pattern 0x00200000 << e  (2^-128, 2^-127)
//   e >= 2 -> 2^(e-128) = bits (e-1)<<23
// (e == 255 / NaN is not handled, same as the reference)
inline __m512 e8m0_half_to_fp32_16(const uint8_t * e) {
    const __m512i ve = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i *)e));
    __m512i bits = _mm512_slli_epi32(_mm512_sub_epi32(ve, _mm512_set1_epi32(1)), 23);
    const __mmask16 is_denorm = _mm512_cmplt_epu32_mask(ve, _mm512_set1_epi32(2));
    const __m512i denorm_bits = _mm512_sllv_epi32(_mm512_set1_epi32(0x00200000), ve);
    bits = _mm512_mask_mov_epi32(bits, is_denorm, denorm_bits);
    return _mm512_castsi512_ps(bits);
}

inline void s8s8_compensation(void * RESTRICT packed_B) {
    // packed_B layout:
    //   quants {TILE_N, TILEK}  int8_t
    //   d0     {TILE_N}      ggml_half
    //   comp   {TILE_N}        int32_t
    const int offset = TILE_N * TILE_K + TILE_N * sizeof(ggml_half);
    __m512i vcomp = _mm512_setzero_si512();
    const __m512i off = _mm512_set1_epi8(static_cast<char>(0x80));
    for (int k = 0; k < 8; ++k) {
        __m512i vb = _mm512_loadu_si512((const __m512i *)((const char *)packed_B + k * 64));
        vcomp = _mm512_dpbusd_epi32(vcomp, off, vb);
    }
    _mm512_storeu_si512((__m512i *)((char *)(packed_B) + offset), vcomp);
}

void pack_B(void * RESTRICT packed_B, const block_q8_0 * RESTRICT B, int KB) {
    pack_qs(packed_B, B, KB);
    ggml_half * d0 = reinterpret_cast<ggml_half *>((char *)packed_B + TILE_N * TILE_K);
    for (int n = 0; n < TILE_N; ++n) {
        d0[n] = B[n * KB].d;
    }
    s8s8_compensation(packed_B);
}

// convert 8 * {min, scale} from int6 to int8
inline void unpack_mins_and_scales(const uint8_t * scales, uint32_t * utmp) {
    const uint32_t kmask1 = 0x3f3f3f3f;
    const uint32_t kmask2 = 0x0f0f0f0f;
    const uint32_t kmask3 = 0x03030303;

    memcpy(utmp, scales, 12);
    utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
    const uint32_t uaux = utmp[1] & kmask1;
    utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
    utmp[2] = uaux;
    utmp[0] &= kmask1;
}

// packed_B layout:
//   quants {8, TILE_N, 16}  uint8
//   scales {8, TILE_N}      uint8
//   mins   {8, TILE_N}      uint8
//   d      {TILE_N}     ggml_half
//   dmin   {TILE_N}     ggml_half
void pack_B(void * RESTRICT packed_B, const block_q4_K * RESTRICT B, int KB) {
    pack_qs(packed_B, B, KB);

    uint8_t * scales = reinterpret_cast<uint8_t *>((char *)packed_B + (QK_K / 2) * TILE_N);
    uint8_t * mins = scales + 8 * TILE_N;
    ggml_half * d = reinterpret_cast<ggml_half *>(mins + 8 * TILE_N);
    ggml_half * dmin = d + TILE_N;

    union {
        uint32_t u32[4];
        uint8_t  u8[16];
    } s;

    for (int n = 0; n < TILE_N; ++n) {
        unpack_mins_and_scales(B[n * KB].scales, s.u32);
        for (int k = 0; k < 8; ++k) {
            scales[k * TILE_N + n] = s.u8[k];
            mins[(k >> 1) * TILE_N * 2 + n * 2 + (k & 0x1)] = s.u8[k + 8];
        }
        d[n] = B[n * KB].d;
        dmin[n] = B[n * KB].dmin;
    }
}

// packed_B layout:
//   quants {8, TILE_N, 16}  uint8
//   qh     {8, TILE_N,  4}  uint8
//   scales {8, TILE_N}      uint8
//   mins   {8, TILE_N}      uint8
//   d      {TILE_N}     ggml_half
//   dmin   {TILE_N}     ggml_half
void pack_B(void * RESTRICT packed_B, const block_q5_K * RESTRICT B, int KB) {
    pack_qs(packed_B, B, KB);

    uint8_t * scales = reinterpret_cast<uint8_t *>((char *)packed_B + (QK_K / 2) * TILE_N + (QK_K / 8) * TILE_N);
    uint8_t * mins = scales + 8 * TILE_N;
    ggml_half * d = reinterpret_cast<ggml_half *>(mins + 8 * TILE_N);
    ggml_half * dmin = d + TILE_N;

    union {
        uint32_t u32[4];
        uint8_t  u8[16];
    } s;

    for (int n = 0; n < TILE_N; ++n) {
        unpack_mins_and_scales(B[n * KB].scales, s.u32);
        for (int k = 0; k < 8; ++k) {
            scales[k * TILE_N + n] = s.u8[k];
            mins[(k >> 1) * TILE_N * 2 + n * 2 + (k & 0x1)] = s.u8[k + 8];
        }
        d[n] = B[n * KB].d;
        dmin[n] = B[n * KB].dmin;
    }
}

// packed_B layout:
//   quants {16, TILE_N, 8}  uint8
//   qh     {16, TILE_N, 4}  uint8
//   scales {16, TILE_N}      uint8
//   d      {TILE_N}     ggml_half
void pack_B(void * RESTRICT packed_B, const block_q6_K * RESTRICT B, int KB) {
    pack_qs(packed_B, B, KB);

    uint8_t * scales = reinterpret_cast<uint8_t *>((char *)packed_B + (QK_K / 2) * TILE_N + (QK_K / 4) * TILE_N);
    ggml_half * d = reinterpret_cast<ggml_half *>(scales + 16 * TILE_N);
    for (int n = 0; n < TILE_N; ++n) {
        const int8_t * ps = B[n * KB].scales;
        for (int k = 0; k < 16; ++k) {
            scales[k * TILE_N + n] = ps[k];
        }
        d[n] = B[n * KB].d;
    }
}

// packed_B layout:
//   quants {8, TILE_N, 16}  uint8
//   scales {8, TILE_N}       int8
//   d      {TILE_N}     ggml_half
void pack_B(void * RESTRICT packed_B, const block_iq4_xs * RESTRICT B, int KB) {
    pack_qs(packed_B, B, KB);

    int8_t * scales = reinterpret_cast<int8_t *>((char *)packed_B + (QK_K / 2) * TILE_N);
    ggml_half * d = reinterpret_cast<ggml_half *>(scales + 8 * TILE_N);

    // pack the scales
    for (int n = 0; n < TILE_N; ++n) {
        uint16_t sh = B[n * KB].scales_h;
        for (int k = 0; k < 8; k += 2) {
            const int16_t ls1 = ((B[n * KB].scales_l[k / 2] & 0xf) | ((sh << 4) & 0x30)) - 32;
            const int16_t ls2 = ((B[n * KB].scales_l[k / 2] >>  4) | ((sh << 2) & 0x30)) - 32;
            scales[(k + 0) * TILE_N + n] = ls1;
            scales[(k + 1) * TILE_N + n] = ls2;
            sh >>= 4;
        }
        d[n] = B[n * KB].d;
    }
}

// packed_B layout (decode-at-pack, see decoders above):
//   quants {8, 8, TILE_N, 4}  int8  (vnni tiles, 512B per 32-value group)
//   scales {8, TILE_N}        int8
//   d      {TILE_N}           ggml_half
void pack_B(void * RESTRICT packed_B, const block_iq2_xxs * RESTRICT B, int KB) {
    int8_t dec[TILE_N * QK_K];
    int8_t * scales = reinterpret_cast<int8_t *>((char *)packed_B + QK_K * TILE_N);
    ggml_half * d = reinterpret_cast<ggml_half *>(scales + 8 * TILE_N);
    for (int n = 0; n < TILE_N; ++n) {
        int8_t sc[8];
        decode_iq2_xxs_row(&B[n * KB], dec + n * QK_K, sc);
        for (int g = 0; g < 8; ++g) {
            scales[g * TILE_N + n] = sc[g];
        }
        d[n] = B[n * KB].d;
    }
    pack_qs_decoded(packed_B, dec);
}

// packed_B layout: same as IQ2_XXS
void pack_B(void * RESTRICT packed_B, const block_iq3_xxs * RESTRICT B, int KB) {
    int8_t dec[TILE_N * QK_K];
    int8_t * scales = reinterpret_cast<int8_t *>((char *)packed_B + QK_K * TILE_N);
    ggml_half * d = reinterpret_cast<ggml_half *>(scales + 8 * TILE_N);
    for (int n = 0; n < TILE_N; ++n) {
        int8_t sc[8];
        decode_iq3_xxs_row(&B[n * KB], dec + n * QK_K, sc);
        for (int g = 0; g < 8; ++g) {
            scales[g * TILE_N + n] = sc[g];
        }
        d[n] = B[n * KB].d;
    }
    pack_qs_decoded(packed_B, dec);
}

// packed_B layout:
//   quants {8, 8, TILE_N, 4}  int8  (vnni tiles, 512B per 32-value group)
//   scales {16, TILE_N}       int8  (one scale per 16 values)
//   d      {TILE_N}           ggml_half
void pack_B(void * RESTRICT packed_B, const block_iq2_xs * RESTRICT B, int KB) {
    int8_t dec[TILE_N * QK_K];
    int8_t * scales = reinterpret_cast<int8_t *>((char *)packed_B + QK_K * TILE_N);
    ggml_half * d = reinterpret_cast<ggml_half *>(scales + 16 * TILE_N);
    for (int n = 0; n < TILE_N; ++n) {
        int8_t sc[16];
        decode_iq2_xs_row(&B[n * KB], dec + n * QK_K, sc);
        for (int g = 0; g < 16; ++g) {
            scales[g * TILE_N + n] = sc[g];
        }
        d[n] = B[n * KB].d;
    }
    pack_qs_decoded(packed_B, dec);
}

// packed_B layout:
//   quants {8, 8, TILE_N, 4}  uint8 (vnni tiles, 512B per 32-value group)
//   scales {16, TILE_N}       int8  (one scale per 16 values)
//   mins   {8, TILE_N, 2}     int8  (pairwise interleaved, same as Q4_K)
//   d      {TILE_N}           ggml_half
//   dmin   {TILE_N}           ggml_half
void pack_B(void * RESTRICT packed_B, const block_q2_K * RESTRICT B, int KB) {
    int8_t dec[TILE_N * QK_K];
    int8_t * scales = reinterpret_cast<int8_t *>((char *)packed_B + QK_K * TILE_N);
    int8_t * mins = scales + 16 * TILE_N;
    ggml_half * d = reinterpret_cast<ggml_half *>(mins + 16 * TILE_N);
    ggml_half * dmin = d + TILE_N;
    for (int n = 0; n < TILE_N; ++n) {
        int8_t sc[16], mn[16];
        decode_q2_K_row(&B[n * KB], dec + n * QK_K, sc, mn);
        for (int g = 0; g < 16; ++g) {
            scales[g * TILE_N + n] = sc[g];
            mins[(g >> 1) * TILE_N * 2 + n * 2 + (g & 0x1)] = mn[g];
        }
        d[n] = B[n * KB].d;
        dmin[n] = B[n * KB].dmin;
    }
    pack_qs_decoded(packed_B, dec);
}

template<typename TB, typename packed_B_t = packed_B_type<TB>>
void unpack_B(packed_B_t * RESTRICT tile, const void * RESTRICT packed_B) {
    GGML_UNUSED(tile);
    GGML_UNUSED(packed_B);
}

template <>
void unpack_B<block_q4_0>(int8_t * RESTRICT tile, const void * RESTRICT packed_B) {
  const __m512i off = _mm512_set1_epi8(8);
  const __m512i lowMask = _mm512_set1_epi8(0xF);
  for (int n = 0; n < 8; n += 2) {
    __m512i bytes = _mm512_loadu_si512((const __m512i *)((const char *)packed_B + n * 32));
    const __m512i r0 = _mm512_sub_epi8(_mm512_and_si512(bytes, lowMask), off);
    const __m512i r1 = _mm512_sub_epi8(_mm512_and_si512(_mm512_srli_epi16(bytes, 4), lowMask), off);
    _mm512_storeu_si512((__m512i *)(tile + n * 64 +  0), r0);
    _mm512_storeu_si512((__m512i *)(tile + n * 64 + 64), r1);
  }
}

template <>
void unpack_B<block_q4_1>(uint8_t * RESTRICT tile, const void * RESTRICT packed_B) {
    const __m512i lowMask = _mm512_set1_epi8(0xF);
    for (int n = 0; n < 8; n += 2) {
        __m512i bytes = _mm512_loadu_si512((const __m512i *)((const char *)packed_B + n * 32));
        const __m512i r0 = _mm512_and_si512(bytes, lowMask);
        const __m512i r1 = _mm512_and_si512(_mm512_srli_epi16(bytes, 4), lowMask);
        _mm512_storeu_si512((__m512i *)(tile + n * 64 +  0), r0);
        _mm512_storeu_si512((__m512i *)(tile + n * 64 + 64), r1);
    }
}

// kvalues_fp4 LUT (e2m1 values, doubled), replicated in each 128-bit lane for _mm512_shuffle_epi8
inline __m512i kvalues_fp4_128() {
    return _mm512_set_epi8(
        -12, -8, -6, -4, -3, -2, -1, 0, 12, 8, 6, 4, 3, 2, 1, 0,
        -12, -8, -6, -4, -3, -2, -1, 0, 12, 8, 6, 4, 3, 2, 1, 0,
        -12, -8, -6, -4, -3, -2, -1, 0, 12, 8, 6, 4, 3, 2, 1, 0,
        -12, -8, -6, -4, -3, -2, -1, 0, 12, 8, 6, 4, 3, 2, 1, 0
    );
}

template <>
void unpack_B<block_mxfp4>(int8_t * RESTRICT tile, const void * RESTRICT packed_B) {
    const __m512i values128 = kvalues_fp4_128();
    const __m512i lowMask = _mm512_set1_epi8(0xF);
    for (int n = 0; n < 8; n += 2) {
        __m512i bytes = _mm512_loadu_si512((const __m512i *)((const char *)packed_B + n * 32));
        const __m512i r0 = _mm512_shuffle_epi8(values128, _mm512_and_si512(bytes, lowMask));
        const __m512i r1 = _mm512_shuffle_epi8(values128, _mm512_and_si512(_mm512_srli_epi16(bytes, 4), lowMask));
        _mm512_storeu_si512((__m512i *)(tile + n * 64 +  0), r0);
        _mm512_storeu_si512((__m512i *)(tile + n * 64 + 64), r1);
    }
}

// packed_B_t for QKK is int8_t
template <typename TB>
void unpack_B(int8_t * RESTRICT tile, const void * RESTRICT packed_B, int k) {
    const int packed_B_group_size = QK_K / 2 * TILE_N / 8;
    const char * packed_B_group = (const char *)packed_B + k * packed_B_group_size;
    const __m512i lowMask = _mm512_set1_epi8(0xF);
    for (int n = 0; n < 8; n += 2) {
        __m512i bytes = _mm512_loadu_si512(packed_B_group + n * 32);
        const __m512i r0 = _mm512_and_si512(bytes, lowMask);
        const __m512i r1 = _mm512_and_si512(_mm512_srli_epi16(bytes, 4), lowMask);
        _mm512_storeu_si512((__m512i *)(tile + n * 64 +  0), r0);
        _mm512_storeu_si512((__m512i *)(tile + n * 64 + 64), r1);
    }
}

template <>
void unpack_B<block_q5_K>(int8_t * RESTRICT tile, const void * RESTRICT packed_B, int k) {
    // lower 4bits, stride 256 bytes
    const int packed_l4_group_size = QK_K / 2 * TILE_N / 8;
    const char * pb = (const char *)packed_B + k * packed_l4_group_size;

    // higher 1bit, stride 64 bytes
    const int packed_h1_group_size = QK_K / 8 * TILE_N / 8;
    const char * ph = (const char *)packed_B + (QK_K / 2) * TILE_N + k * packed_h1_group_size;
    const __m512i hbits = _mm512_loadu_si512(ph);

    const __m512i lowMask = _mm512_set1_epi8(0xF);
    __m512i hmask0 = _mm512_set1_epi8(0x1);
    __m512i hmask1 = _mm512_set1_epi8(0x2);

    for (int n = 0; n < 8; n += 2) {
        __m512i bytes = _mm512_loadu_si512(pb + n * 32);
        __m512i r0 = _mm512_and_si512(bytes, lowMask);
        __m512i r1 = _mm512_and_si512(_mm512_srli_epi16(bytes, 4), lowMask);
        __m512i h0 = _mm512_slli_epi16(_mm512_srli_epi16(_mm512_and_si512(hbits, hmask0), n), 4);
        __m512i h1 = _mm512_slli_epi16(_mm512_srli_epi16(_mm512_and_si512(hbits, hmask1), n + 1), 4);

        hmask0 = _mm512_slli_epi16(hmask0, 2);
        hmask1 = _mm512_slli_epi16(hmask1, 2);
        r0 = _mm512_add_epi8(r0, h0);
        r1 = _mm512_add_epi8(r1, h1);
        _mm512_storeu_si512((__m512i *)(tile + n * 64 +  0), r0);
        _mm512_storeu_si512((__m512i *)(tile + n * 64 + 64), r1);
    }
}

template <>
void unpack_B<block_q6_K>(int8_t * RESTRICT tile, const void * RESTRICT packed_B, int k) {
    // lower 4bits, stride 128 bytes
    const int packed_l4_group_size = QK_K / 2 * TILE_N / 16;
    const char * pb = (const char *)packed_B + k * packed_l4_group_size;

    // higher 2bits, stride 64 bytes
    const int packed_h2_group_size = QK_K / 4 * TILE_N / 16;
    const char * ph = (const char *)packed_B + (QK_K / 2) * TILE_N + k * packed_h2_group_size;
    const __m512i hbits = _mm512_loadu_si512(ph);

    const __m512i off = _mm512_set1_epi8(32);
    const __m512i lowMask = _mm512_set1_epi8(0xF);
    __m512i hmask0 = _mm512_set1_epi8(0x3); // 0011
    __m512i hmask1 = _mm512_set1_epi8(0xC); // 1100

    // notes: skip zero padding from row4 to row7 as we have done so in `unpack_A`
    __m512i bytes = _mm512_loadu_si512(pb);
    __m512i r0 = _mm512_and_si512(bytes, lowMask);
    __m512i r1 = _mm512_and_si512(_mm512_srli_epi16(bytes, 4), lowMask);
    __m512i h0 = _mm512_slli_epi16(_mm512_and_si512(hbits, hmask0), 4);
    __m512i h1 = _mm512_slli_epi16(_mm512_and_si512(hbits, hmask1), 2);
    _mm512_storeu_si512((__m512i *)(tile +  0), _mm512_sub_epi8(_mm512_add_epi8(r0, h0), off));
    _mm512_storeu_si512((__m512i *)(tile + 64), _mm512_sub_epi8(_mm512_add_epi8(r1, h1), off));

    hmask0 = _mm512_slli_epi16(hmask0, 4);
    hmask1 = _mm512_slli_epi16(hmask1, 4);

    bytes = _mm512_loadu_si512(pb + 64);
    r0 = _mm512_and_si512(bytes, lowMask);
    r1 = _mm512_and_si512(_mm512_srli_epi16(bytes, 4), lowMask);
    h0 =                   _mm512_and_si512(hbits, hmask0);
    h1 = _mm512_srli_epi16(_mm512_and_si512(hbits, hmask1), 2);
    _mm512_storeu_si512((__m512i *)(tile + 128), _mm512_sub_epi8(_mm512_add_epi8(r0, h0), off));
    _mm512_storeu_si512((__m512i *)(tile + 192), _mm512_sub_epi8(_mm512_add_epi8(r1, h1), off));
}

template <>
void unpack_B<block_iq4_xs>(int8_t * RESTRICT tile, const void * RESTRICT packed_B, int k) {
    static const __m512i values128 = _mm512_set_epi8(
        113, 89, 69, 53, 38, 25, 13, 1, -10, -22, -35, -49, -65, -83, -104, -127,
        113, 89, 69, 53, 38, 25, 13, 1, -10, -22, -35, -49, -65, -83, -104, -127,
        113, 89, 69, 53, 38, 25, 13, 1, -10, -22, -35, -49, -65, -83, -104, -127,
        113, 89, 69, 53, 38, 25, 13, 1, -10, -22, -35, -49, -65, -83, -104, -127
    );

    const int packed_B_group_size = QK_K / 2 * TILE_N / 8;
    const char * pb = (const char *)packed_B + k * packed_B_group_size;
    const __m512i lowMask = _mm512_set1_epi8(0xF);

    for (int n = 0; n < 8; n += 2) {
        __m512i bytes = _mm512_loadu_si512(pb + n * 32);
        const __m512i r0 = _mm512_shuffle_epi8(values128, _mm512_and_si512(bytes, lowMask));
        const __m512i r1 = _mm512_shuffle_epi8(values128, _mm512_and_si512(_mm512_srli_epi16(bytes, 4), lowMask));
        _mm512_storeu_si512((__m512i *)(tile + n * 64 +  0), r0);
        _mm512_storeu_si512((__m512i *)(tile + n * 64 + 64), r1);
    }
}

// IQ2_XXS / IQ3_XXS store decoded 8-bit quants in the vnni tile layout:
// unpack is a plain copy of the 512B k-group slice.
template <>
void unpack_B<block_iq2_xxs>(int8_t * RESTRICT tile, const void * RESTRICT packed_B, int k) {
    memcpy(tile, (const char *)packed_B + k * (QK_K / 8) * TILE_N, 32 * TILE_N);
}

template <>
void unpack_B<block_iq3_xxs>(int8_t * RESTRICT tile, const void * RESTRICT packed_B, int k) {
    memcpy(tile, (const char *)packed_B + k * (QK_K / 8) * TILE_N, 32 * TILE_N);
}

// Q2_K / IQ2_XS have one scale per 16 values; k indexes 16-value groups, which are
// the lower/upper half (256B) of the containing 32-value vnni tile. The upper half
// of the tile needs no zeroing: unpack_A zero-pads A from 16 to 32 (as for Q6_K).
template <>
void unpack_B<block_q2_K>(int8_t * RESTRICT tile, const void * RESTRICT packed_B, int k) {
    memcpy(tile, (const char *)packed_B + (k >> 1) * (QK_K / 8) * TILE_N + (k & 1) * 16 * TILE_N, 16 * TILE_N);
}

template <>
void unpack_B<block_iq2_xs>(int8_t * RESTRICT tile, const void * RESTRICT packed_B, int k) {
    memcpy(tile, (const char *)packed_B + (k >> 1) * (QK_K / 8) * TILE_N + (k & 1) * 16 * TILE_N, 16 * TILE_N);
}

template <typename TA, typename TB, bool is_acc>
struct acc_C {};

template <bool is_acc>
struct acc_C<block_q8_0, block_q4_0, is_acc> {
    static void apply(float * RESTRICT C, int ldc, const int32_t * RESTRICT tile, const block_q8_0 * A, int lda, const void * packed_B, int nr) {
        const int offset = TILE_N * TILE_K / 2;
        const __m512 vd0 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *)((const char *)packed_B + offset)));

        for (int m = 0; m < nr; ++m) {
            const __m512 vd1 = _mm512_set1_ps(GGML_CPU_FP16_TO_FP32(A[m * lda].d));
            const __m512 vtile = _mm512_cvtepi32_ps(_mm512_loadu_si512(tile + m * TILE_N));

            __m512 vsum;
            if (is_acc) {
                vsum = _mm512_loadu_ps(C + m * ldc);
            } else {
                vsum = _mm512_set1_ps(0.f);
            }
            vsum = _mm512_fmadd_ps(vtile, _mm512_mul_ps(vd0, vd1), vsum);
            _mm512_storeu_ps(C + m * ldc, vsum);
        }
    }
};

template <bool is_acc>
struct acc_C<block_q8_0, block_mxfp4, is_acc> {
    static void apply(float * RESTRICT C, int ldc, const int32_t * RESTRICT tile, const block_q8_0 * A, int lda, const void * packed_B, int nr) {
        const int offset = TILE_N * TILE_K / 2;
        const __m512 vd0 = e8m0_half_to_fp32_16((const uint8_t *)packed_B + offset);

        for (int m = 0; m < nr; ++m) {
            const __m512 vd1 = _mm512_set1_ps(GGML_CPU_FP16_TO_FP32(A[m * lda].d));
            const __m512 vtile = _mm512_cvtepi32_ps(_mm512_loadu_si512(tile + m * TILE_N));

            __m512 vsum;
            if (is_acc) {
                vsum = _mm512_loadu_ps(C + m * ldc);
            } else {
                vsum = _mm512_set1_ps(0.f);
            }
            vsum = _mm512_fmadd_ps(vtile, _mm512_mul_ps(vd0, vd1), vsum);
            _mm512_storeu_ps(C + m * ldc, vsum);
        }
    }
};

template <bool is_acc>
struct acc_C<block_q8_1, block_q4_1, is_acc> {
    static void apply(float * RESTRICT C, int ldc, const int32_t * RESTRICT tile, const block_q8_1 * A, int lda, const void * packed_B, int nr) {
        const int offset = TILE_N * TILE_K / 2;
        const __m512 vd0 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *)((const char *)packed_B + offset)));
        const __m512 vm0 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *)((const char *)packed_B + offset + TILE_N * sizeof(ggml_half))));

        for (int m = 0; m < nr; ++m) {
            const __m512 vd1 = _mm512_set1_ps(GGML_CPU_FP16_TO_FP32(A[m * lda].d));
            const __m512 vs1 = _mm512_set1_ps(GGML_CPU_FP16_TO_FP32(A[m * lda].s));
            const __m512 vtile = _mm512_cvtepi32_ps(_mm512_loadu_si512(tile + m * TILE_N));

            __m512 vsum;
            if (is_acc) {
                vsum = _mm512_loadu_ps(C + m * ldc);
            } else {
                vsum = _mm512_set1_ps(0.f);
            }
            vsum = _mm512_fmadd_ps(vtile, _mm512_mul_ps(vd0, vd1), vsum);
            vsum = _mm512_fmadd_ps(vm0, vs1, vsum);
            _mm512_storeu_ps(C + m * ldc, vsum);
        }
    }
};

template <bool is_acc>
struct acc_C<block_q8_0, block_q8_0, is_acc> {
    static void apply(float * RESTRICT C, int ldc, const int32_t * RESTRICT tile, const block_q8_0 * A, int lda, const void * packed_B, int nr) {
        const int offset = TILE_N * TILE_K;
        const __m512 vd0 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *)((const char *)packed_B + offset)));

        for (int m = 0; m < nr; ++m) {
            const __m512 vd1 = _mm512_set1_ps(GGML_CPU_FP16_TO_FP32(A[m * lda].d));
            const __m512 vtile = _mm512_cvtepi32_ps(_mm512_loadu_si512(tile + m * TILE_N));

            __m512 vsum;
            if (is_acc) {
                vsum = _mm512_loadu_ps(C + m * ldc);
            } else {
                vsum = _mm512_set1_ps(0.f);
            }
            vsum = _mm512_fmadd_ps(vtile, _mm512_mul_ps(vd0, vd1), vsum);
            _mm512_storeu_ps(C + m * ldc, vsum);
        }
    }
};

template <bool is_acc>
struct acc_C<block_q8_K, block_q4_K, is_acc> {
    static void apply(float * RESTRICT C, int ldc, const int32_t * RESTRICT tile, const block_q8_K * A, int lda, const void * packed_B, int nr) {
        const uint8_t * scales = reinterpret_cast<const uint8_t *>((const char *)packed_B + (QK_K / 2) * TILE_N);
        const uint8_t * mins = scales + 8 * TILE_N;
        const ggml_half * d0 = reinterpret_cast<const ggml_half *>(mins + 8 * TILE_N);
        const ggml_half * dmin = d0 + TILE_N;

        const __m512 vd0 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *)d0));
        const __m512 vdmin = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *)dmin));

        for (int m = 0; m < nr; ++m) {
            const float d1 = A[m * lda].d;
            const __m512 vd = _mm512_mul_ps(_mm512_set1_ps(d1), vd0);
            const __m512 vdm = _mm512_mul_ps(_mm512_set1_ps(-d1), vdmin);
            const __m512 vtile = _mm512_cvtepi32_ps(_mm512_loadu_si512(tile + m * TILE_N));

            __m512 vsum;
            if (is_acc) {
                vsum = _mm512_loadu_ps(C + m * ldc);
            } else {
                vsum = _mm512_set1_ps(0.f);
            }

            const __m256i q8sums = _mm256_loadu_si256((const __m256i *)A[m * lda].bsums);
            const __m128i q8s = _mm_hadd_epi16(_mm256_extracti128_si256(q8sums, 0), _mm256_extracti128_si256(q8sums, 1));

            __m512i acc_m = _mm512_setzero_si512();
            for (int k = 0; k < 4; ++k) {
                __m512i vmask = _mm512_set1_epi32(k);
                __m512i va = _mm512_permutexvar_epi32(vmask, _mm512_castsi128_si512(q8s));
                __m512i vb = _mm512_cvtepi8_epi16(_mm256_loadu_si256((const __m256i *)(mins + k * 32)));
                acc_m = _mm512_dpwssds_epi32(acc_m, va, vb);
            }

            vsum = _mm512_fmadd_ps(vtile, vd, vsum);
            vsum = _mm512_fmadd_ps(_mm512_cvtepi32_ps(acc_m), vdm, vsum);
            _mm512_storeu_ps(C + m * ldc, vsum);
        }
    }
};

template <bool is_acc>
struct acc_C<block_q8_K, block_q5_K, is_acc> {
    static void apply(float * RESTRICT C, int ldc, const int32_t * RESTRICT tile, const block_q8_K * A, int lda, const void * packed_B, int nr) {
        const uint8_t * scales = reinterpret_cast<const uint8_t *>((const char *)packed_B + (QK_K / 2) * TILE_N + (QK_K / 8) * TILE_N);
        const uint8_t * mins = scales + 8 * TILE_N;
        const ggml_half * d0 = reinterpret_cast<const ggml_half *>(mins + 8 * TILE_N);
        const ggml_half * dmin = d0 + TILE_N;

        const __m512 vd0 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *)d0));
        const __m512 vdmin = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *)dmin));

        for (int m = 0; m < nr; ++m) {
            const float d1 = A[m * lda].d;
            const __m512 vd = _mm512_mul_ps(_mm512_set1_ps(d1), vd0);
            const __m512 vdm = _mm512_mul_ps(_mm512_set1_ps(-d1), vdmin);
            const __m512 vtile = _mm512_cvtepi32_ps(_mm512_loadu_si512(tile + m * TILE_N));

            __m512 vsum;
            if (is_acc) {
                vsum = _mm512_loadu_ps(C + m * ldc);
            } else {
                vsum = _mm512_set1_ps(0.f);
            }

            const __m256i q8sums = _mm256_loadu_si256((const __m256i *)A[m * lda].bsums);
            const __m128i q8s = _mm_hadd_epi16(_mm256_extracti128_si256(q8sums, 0), _mm256_extracti128_si256(q8sums, 1));

            __m512i acc_m = _mm512_setzero_si512();
            for (int k = 0; k < 4; ++k) {
                __m512i vmask = _mm512_set1_epi32(k);
                __m512i va = _mm512_permutexvar_epi32(vmask, _mm512_castsi128_si512(q8s));
                __m512i vb = _mm512_cvtepi8_epi16(_mm256_loadu_si256((const __m256i *)(mins + k * 32)));
                acc_m = _mm512_dpwssds_epi32(acc_m, va, vb);
            }

            vsum = _mm512_fmadd_ps(vtile, vd, vsum);
            vsum = _mm512_fmadd_ps(_mm512_cvtepi32_ps(acc_m), vdm, vsum);
            _mm512_storeu_ps(C + m * ldc, vsum);
        }
    }
};

template <bool is_acc>
struct acc_C<block_q8_K, block_q6_K, is_acc> {
    static void apply(float * RESTRICT C, int ldc, const int32_t * RESTRICT tile, const block_q8_K * A, int lda, const void * packed_B, int nr) {
        const uint8_t * scales = reinterpret_cast<const uint8_t *>((const char *)packed_B + (QK_K / 2) * TILE_N + (QK_K / 4) * TILE_N);
        const ggml_half * d0 = reinterpret_cast<const ggml_half *>(scales + 16 * TILE_N);

        const __m512 vd0 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *)d0));

        for (int m = 0; m < nr; ++m) {
            const float d1 = A[m * lda].d;
            const __m512 vd = _mm512_mul_ps(_mm512_set1_ps(d1), vd0);
            const __m512 vtile = _mm512_cvtepi32_ps(_mm512_loadu_si512(tile + m * TILE_N));

            __m512 vsum;
            if (is_acc) {
                vsum = _mm512_loadu_ps(C + m * ldc);
            } else {
                vsum = _mm512_set1_ps(0.f);
            }

            vsum = _mm512_fmadd_ps(vtile, vd, vsum);
            _mm512_storeu_ps(C + m * ldc, vsum);
        }
    }
};

template <bool is_acc>
struct acc_C<block_q8_K, block_iq4_xs, is_acc> {
    static void apply(float * RESTRICT C, int ldc, const int32_t * RESTRICT tile, const block_q8_K * A, int lda, const void * packed_B, int nr) {
        const int8_t * scales = reinterpret_cast<const int8_t *>((const char *)packed_B + (QK_K / 2) * TILE_N);
        const ggml_half * d0 = reinterpret_cast<const ggml_half *>(scales + 8 * TILE_N);

        const __m512 vd0 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *)d0));

        for (int m = 0; m < nr; ++m) {
            const float d1 = A[m * lda].d;
            const __m512 vd = _mm512_mul_ps(_mm512_set1_ps(d1), vd0);
            const __m512 vtile = _mm512_cvtepi32_ps(_mm512_loadu_si512(tile + m * TILE_N));

            __m512 vsum;
            if (is_acc) {
                vsum = _mm512_loadu_ps(C + m * ldc);
            } else {
                vsum = _mm512_set1_ps(0.f);
            }

            vsum = _mm512_fmadd_ps(vtile, vd, vsum);
            _mm512_storeu_ps(C + m * ldc, vsum);
        }
    }
};

// IQ2_XXS / IQ2_XS / IQ3_XXS: tile already contains scale-weighted integer sums
// (scale_C applied the decoded int8 group scales); here we only multiply by the
// super-block scales and the type's global factor (0.125f / 0.125f / 0.25f,
// matching the generic dot products).
template <bool is_acc>
struct acc_C<block_q8_K, block_iq2_xxs, is_acc> {
    static void apply(float * RESTRICT C, int ldc, const int32_t * RESTRICT tile, const block_q8_K * A, int lda, const void * packed_B, int nr) {
        const ggml_half * d0 = reinterpret_cast<const ggml_half *>((const char *)packed_B + QK_K * TILE_N + 8 * TILE_N);

        const __m512 vd0 = _mm512_mul_ps(_mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *)d0)), _mm512_set1_ps(0.125f));

        for (int m = 0; m < nr; ++m) {
            const float d1 = A[m * lda].d;
            const __m512 vd = _mm512_mul_ps(_mm512_set1_ps(d1), vd0);
            const __m512 vtile = _mm512_cvtepi32_ps(_mm512_loadu_si512(tile + m * TILE_N));

            __m512 vsum;
            if (is_acc) {
                vsum = _mm512_loadu_ps(C + m * ldc);
            } else {
                vsum = _mm512_set1_ps(0.f);
            }

            vsum = _mm512_fmadd_ps(vtile, vd, vsum);
            _mm512_storeu_ps(C + m * ldc, vsum);
        }
    }
};

template <bool is_acc>
struct acc_C<block_q8_K, block_iq3_xxs, is_acc> {
    static void apply(float * RESTRICT C, int ldc, const int32_t * RESTRICT tile, const block_q8_K * A, int lda, const void * packed_B, int nr) {
        const ggml_half * d0 = reinterpret_cast<const ggml_half *>((const char *)packed_B + QK_K * TILE_N + 8 * TILE_N);

        const __m512 vd0 = _mm512_mul_ps(_mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *)d0)), _mm512_set1_ps(0.25f));

        for (int m = 0; m < nr; ++m) {
            const float d1 = A[m * lda].d;
            const __m512 vd = _mm512_mul_ps(_mm512_set1_ps(d1), vd0);
            const __m512 vtile = _mm512_cvtepi32_ps(_mm512_loadu_si512(tile + m * TILE_N));

            __m512 vsum;
            if (is_acc) {
                vsum = _mm512_loadu_ps(C + m * ldc);
            } else {
                vsum = _mm512_set1_ps(0.f);
            }

            vsum = _mm512_fmadd_ps(vtile, vd, vsum);
            _mm512_storeu_ps(C + m * ldc, vsum);
        }
    }
};

template <bool is_acc>
struct acc_C<block_q8_K, block_iq2_xs, is_acc> {
    static void apply(float * RESTRICT C, int ldc, const int32_t * RESTRICT tile, const block_q8_K * A, int lda, const void * packed_B, int nr) {
        const ggml_half * d0 = reinterpret_cast<const ggml_half *>((const char *)packed_B + QK_K * TILE_N + 16 * TILE_N);

        const __m512 vd0 = _mm512_mul_ps(_mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *)d0)), _mm512_set1_ps(0.125f));

        for (int m = 0; m < nr; ++m) {
            const float d1 = A[m * lda].d;
            const __m512 vd = _mm512_mul_ps(_mm512_set1_ps(d1), vd0);
            const __m512 vtile = _mm512_cvtepi32_ps(_mm512_loadu_si512(tile + m * TILE_N));

            __m512 vsum;
            if (is_acc) {
                vsum = _mm512_loadu_ps(C + m * ldc);
            } else {
                vsum = _mm512_set1_ps(0.f);
            }

            vsum = _mm512_fmadd_ps(vtile, vd, vsum);
            _mm512_storeu_ps(C + m * ldc, vsum);
        }
    }
};

// Q2_K: same structure as Q4_K (scales already applied by scale_C; here the mins
// term is accumulated against the per-16-group bsums), but with 16 groups of 16
// instead of 8 groups of 32, so the mins loop runs 8 iterations over the raw
// bsums dwords (no hadd pairing needed).
template <bool is_acc>
struct acc_C<block_q8_K, block_q2_K, is_acc> {
    static void apply(float * RESTRICT C, int ldc, const int32_t * RESTRICT tile, const block_q8_K * A, int lda, const void * packed_B, int nr) {
        const int8_t * mins = reinterpret_cast<const int8_t *>((const char *)packed_B + QK_K * TILE_N + 16 * TILE_N);
        const ggml_half * d0 = reinterpret_cast<const ggml_half *>(mins + 16 * TILE_N);
        const ggml_half * dmin = d0 + TILE_N;

        const __m512 vd0 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *)d0));
        const __m512 vdmin = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *)dmin));

        for (int m = 0; m < nr; ++m) {
            const float d1 = A[m * lda].d;
            const __m512 vd = _mm512_mul_ps(_mm512_set1_ps(d1), vd0);
            const __m512 vdm = _mm512_mul_ps(_mm512_set1_ps(-d1), vdmin);
            const __m512 vtile = _mm512_cvtepi32_ps(_mm512_loadu_si512(tile + m * TILE_N));

            __m512 vsum;
            if (is_acc) {
                vsum = _mm512_loadu_ps(C + m * ldc);
            } else {
                vsum = _mm512_set1_ps(0.f);
            }

            // 16 int16 bsums; dword j holds the (bsums[2j], bsums[2j+1]) pair
            const __m512i q8sums = _mm512_castsi256_si512(_mm256_loadu_si256((const __m256i *)A[m * lda].bsums));

            __m512i acc_m = _mm512_setzero_si512();
            for (int k = 0; k < 8; ++k) {
                __m512i va = _mm512_permutexvar_epi32(_mm512_set1_epi32(k), q8sums);
                __m512i vb = _mm512_cvtepi8_epi16(_mm256_loadu_si256((const __m256i *)(mins + k * 32)));
                acc_m = _mm512_dpwssds_epi32(acc_m, va, vb);
            }

            vsum = _mm512_fmadd_ps(vtile, vd, vsum);
            vsum = _mm512_fmadd_ps(_mm512_cvtepi32_ps(acc_m), vdm, vsum);
            _mm512_storeu_ps(C + m * ldc, vsum);
        }
    }
};

template <typename TB> constexpr int get_quants_size();
template <> constexpr int get_quants_size<block_q4_K>() { return (QK_K / 2) * TILE_N; }
template <> constexpr int get_quants_size<block_q5_K>() { return (QK_K / 2) * TILE_N + (QK_K / 8) * TILE_N; }
template <> constexpr int get_quants_size<block_q6_K>() { return (QK_K / 2) * TILE_N + (QK_K / 4) * TILE_N; }
template <> constexpr int get_quants_size<block_iq4_xs>() { return (QK_K / 2) * TILE_N; }
template <> constexpr int get_quants_size<block_q2_K>() { return QK_K * TILE_N; }
template <> constexpr int get_quants_size<block_iq2_xxs>() { return QK_K * TILE_N; }
template <> constexpr int get_quants_size<block_iq2_xs>() { return QK_K * TILE_N; }
template <> constexpr int get_quants_size<block_iq3_xxs>() { return QK_K * TILE_N; }

// used for QKK format
template <typename TB, bool is_acc,
          typename std::enable_if<is_type_qkk<TB>::value, int>::type = 0>
inline void scale_C(const int32_t * RESTRICT tile, int32_t * RESTRICT sumi, const void * packed_B, int k, int nr) {
    const uint8_t * scales = reinterpret_cast<const uint8_t *>((const char *)packed_B + get_quants_size<TB>());
    const __m512i vscale = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i *)(scales + k * TILE_N)));

    for (int m = 0; m < nr; ++m) {
        __m512i vsumi;
        if (is_acc) {
            vsumi = _mm512_loadu_si512(sumi + m * TILE_N);
        } else {
            vsumi = _mm512_setzero_si512();
        }
        __m512i vtile = _mm512_loadu_si512(tile + m * TILE_N);
        vsumi = _mm512_add_epi32(vsumi, _mm512_mullo_epi32(vtile, vscale));
        _mm512_storeu_si512((__m512i *)(sumi + m * TILE_N), vsumi);
    }
}

template <typename TA, typename TB, typename TC, int BLOCK_M, int BLOCK_N, int BLOCK_K>
struct tinygemm_kernel_avx {
    static void apply(int K, const TA * RESTRICT A, const TB * RESTRICT B, TC * RESTRICT C, int ldc) {
        GGML_UNUSED(K);
        GGML_UNUSED(A);
        GGML_UNUSED(B);
        GGML_UNUSED(C);
        GGML_UNUSED(ldc);
    }
};

template <int BLOCK_M, int BLOCK_N, int BLOCK_K>
struct tinygemm_kernel_avx<float, ggml_fp16_t, float, BLOCK_M, BLOCK_N, BLOCK_K> {
    static void apply(int K, const float * RESTRICT A, const ggml_fp16_t * RESTRICT B, float * RESTRICT C, int ldc) {
        constexpr int ROWS = BLOCK_M;
        constexpr int COLS = BLOCK_N;
        assert(BLOCK_K == 16);

        __m512 va;
        __m512 vb[COLS];
        __m512 vc[ROWS * COLS];

        auto loadc = [&](auto idx) {
            vc[idx] = _mm512_setzero_ps();
        };
        Unroll<ROWS * COLS>{}(loadc);

        auto compute = [&](auto idx, auto k) {
            constexpr int row = idx / COLS;
            constexpr int col = idx % COLS;

            if constexpr (col == 0) {
                va = _mm512_loadu_ps(A + row * K + k);
            }
            if constexpr (row == 0) {
                vb[col] =  _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *)(B + col * K + k)));
            }
            vc[idx] = _mm512_fmadd_ps(va, vb[col], vc[idx]);
        };

        for (int k = 0; k < K; k += 16) {
            Unroll<ROWS * COLS>{}(compute, k);
        }

        auto storec = [&](auto idx) {
            constexpr int row = idx / COLS;
            constexpr int col = idx % COLS;
            C[row * ldc + col] = _mm512_reduce_add_ps(vc[idx]);
        };
        Unroll<ROWS * COLS>{}(storec);
    }
};

#define LAUNCH_TINYGEMM_KERNEL_AVX(MB_SIZE, NB_SIZE)                                \
    tinygemm_kernel_avx<float, type, float, MB_SIZE, NB_SIZE, blck_size>::apply(    \
        K, (const float *)src1->data + src1_offset + mb_start * K,                  \
        (const type *)src0->data + src0_offset + nb_start * K,                      \
        (float *)dst->data + dst_offset + mb_start * ldc + nb_start, ldc)


// re-organize in the format {NB, KB, TILE_SIZE}:
#define PACKED_INDEX(n, k, KB, tile_size) (n * KB + k) * tile_size

template<typename TB, int BLOCK_K>
void convert_B_packed_format(void * RESTRICT packed_B, const TB * RESTRICT B, int N, int K) {
    const int NB = N / TILE_N;
    const int KB = K / BLOCK_K;
    const int TILE_SIZE = get_tile_size<TB>();

    // parallel on NB should be enough
    parallel_for(NB, [&](int begin, int end) {
        for (int n = begin; n < end; ++n) {
            for (int k = 0; k < KB; ++k) {
                int n0 = n * TILE_N;
                pack_B((char *)packed_B + PACKED_INDEX(n, k, KB, TILE_SIZE), &B[n0 * KB + k], KB);
            }
        }
    });
}

template <typename TA, typename TB, typename TC, int BLOCK_M, int BLOCK_N, int BLOCK_K>
struct tinygemm_kernel_vnni {};

template <int BLOCK_M, int BLOCK_N, int BLOCK_K>
struct tinygemm_kernel_vnni<block_q8_0, block_q4_0, float, BLOCK_M, BLOCK_N, BLOCK_K> {
    static void apply(int KB, const void * RESTRICT _A, const void * RESTRICT _B, float * RESTRICT C, int ldc) {

        constexpr int COLS = BLOCK_N / 16;
        const int TILE_SIZE = TILE_N * sizeof(block_q4_0);

        const block_q8_0 * RESTRICT A = static_cast<const block_q8_0 *>(_A);
        const char * RESTRICT B = static_cast<const char *>(_B);

        __m512i va[8];
        __m512 vc[COLS];
        __m512 vd1;

        // sum of offsets, shared across COLS
        //
        // avx512-vnni does not have `_mm512_dpbssd_epi32`,
        // need to transform ss to us:
        //   a * (b - 8) is equivalent to b * a - 8 * a
        //   s    u   u                   u   s   u   s
        //
        __m512i vcomp;

        const __m512i off = _mm512_set1_epi8(8);
        const __m512i lowMask = _mm512_set1_epi8(0xF);

        auto loadc = [&](auto col) {
            vc[col] = _mm512_setzero_ps();
        };
        Unroll<COLS>{}(loadc);

        auto compute = [&](auto col, auto i) {
            // load a and compute compensation
            if constexpr (col == 0) {
                const int32_t * a_ptr = reinterpret_cast<const int32_t *>(A[0 * KB + i].qs);
                vcomp = _mm512_setzero_si512();
                for (int k = 0; k < 8; ++k) {
                    va[k] = _mm512_set1_epi32(a_ptr[k]);
                    vcomp = _mm512_dpbusd_epi32(vcomp, off, va[k]);
                }
                vd1 = _mm512_set1_ps(GGML_CPU_FP16_TO_FP32(A[0 * KB + i].d));
            }

            // load b
            __m512i vsum = _mm512_setzero_si512();
            const char * b_ptr = B + PACKED_INDEX(col, i, KB, TILE_SIZE);
            for (int k = 0; k < 8; k += 2) {
                __m512i bytes = _mm512_loadu_si512((const __m512i *)(b_ptr + k * 32));
                __m512i vb0 = _mm512_and_si512(bytes, lowMask);
                vsum = _mm512_dpbusd_epi32(vsum, vb0, va[k + 0]);
                __m512i vb1 = _mm512_and_si512(_mm512_srli_epi16(bytes, 4), lowMask);
                vsum = _mm512_dpbusd_epi32(vsum, vb1, va[k + 1]);
            }
            const int offset = TILE_N * TILE_K / 2;
            const __m512 vd0 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *)(b_ptr + offset)));
            vsum = _mm512_sub_epi32(vsum, vcomp);

            vc[col] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(vsum), _mm512_mul_ps(vd0, vd1), vc[col]);
        };

        for (int i = 0; i < KB; ++i) {
            Unroll<COLS>{}(compute, i);
        }

        //store to C
        auto storec = [&](auto col) {
            _mm512_storeu_ps((__m512i*)(C + 0 * ldc + col * 16), vc[col]);
        };
        Unroll<COLS>{}(storec);
    }
};

template <int BLOCK_M, int BLOCK_N, int BLOCK_K>
struct tinygemm_kernel_vnni<block_q8_0, block_mxfp4, float, BLOCK_M, BLOCK_N, BLOCK_K> {
    static void apply(int KB, const void * RESTRICT _A, const void * RESTRICT _B, float * RESTRICT C, int ldc) {

        constexpr int COLS = BLOCK_N / 16;
        const int TILE_SIZE = TILE_N * sizeof(block_mxfp4);

        const block_q8_0 * RESTRICT A = static_cast<const block_q8_0 *>(_A);
        const char * RESTRICT B = static_cast<const char *>(_B);

        __m512i va[8];
        __m512 vc[COLS];
        __m512 vd1;

        // sum of offsets, shared across COLS
        //
        // kvalues_fp4 are signed and avx512-vnni does not have `_mm512_dpbssd_epi32`,
        // need to transform ss to us:
        //   a * b is equivalent to a * (b + 128) - 128 * a
        //   s   s                   s    u          u    s
        //
        __m512i vcomp;

        const __m512i off = _mm512_set1_epi8(static_cast<char>(0x80));
        const __m512i lowMask = _mm512_set1_epi8(0xF);
        const __m512i values256 = _mm512_add_epi8(kvalues_fp4_128(), off);

        auto loadc = [&](auto col) {
            vc[col] = _mm512_setzero_ps();
        };
        Unroll<COLS>{}(loadc);

        auto compute = [&](auto col, auto i) {
            // load a and compute compensation
            if constexpr (col == 0) {
                const int32_t * a_ptr = reinterpret_cast<const int32_t *>(A[0 * KB + i].qs);
                vcomp = _mm512_setzero_si512();
                for (int k = 0; k < 8; ++k) {
                    va[k] = _mm512_set1_epi32(a_ptr[k]);
                    vcomp = _mm512_dpbusd_epi32(vcomp, off, va[k]);
                }
                vd1 = _mm512_set1_ps(GGML_CPU_FP16_TO_FP32(A[0 * KB + i].d));
            }

            // load b, decode nibbles to (kvalues_fp4 + 128) via LUT
            __m512i vsum = _mm512_setzero_si512();
            const char * b_ptr = B + PACKED_INDEX(col, i, KB, TILE_SIZE);
            for (int k = 0; k < 8; k += 2) {
                __m512i bytes = _mm512_loadu_si512((const __m512i *)(b_ptr + k * 32));
                __m512i vb0 = _mm512_shuffle_epi8(values256, _mm512_and_si512(bytes, lowMask));
                vsum = _mm512_dpbusd_epi32(vsum, vb0, va[k + 0]);
                __m512i vb1 = _mm512_shuffle_epi8(values256, _mm512_and_si512(_mm512_srli_epi16(bytes, 4), lowMask));
                vsum = _mm512_dpbusd_epi32(vsum, vb1, va[k + 1]);
            }
            const int offset = TILE_N * TILE_K / 2;
            const __m512 vd0 = e8m0_half_to_fp32_16((const uint8_t *)(b_ptr + offset));
            vsum = _mm512_sub_epi32(vsum, vcomp);

            vc[col] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(vsum), _mm512_mul_ps(vd0, vd1), vc[col]);
        };

        for (int i = 0; i < KB; ++i) {
            Unroll<COLS>{}(compute, i);
        }

        //store to C
        auto storec = [&](auto col) {
            _mm512_storeu_ps((__m512i*)(C + 0 * ldc + col * 16), vc[col]);
        };
        Unroll<COLS>{}(storec);
    }
};

template <int BLOCK_N, int BLOCK_K>
struct tinygemm_kernel_vnni<block_q8_1, block_q4_1, float, 1, BLOCK_N, BLOCK_K> {
    static void apply(int KB, const void * RESTRICT _A, const void * RESTRICT _B, float * RESTRICT C, int ldc) {

        constexpr int COLS = BLOCK_N / 16;
        const int TILE_SIZE = TILE_N * sizeof(block_q4_1);

        const block_q8_1 * RESTRICT A = static_cast<const block_q8_1 *>(_A);
        const char * RESTRICT B = static_cast<const char *>(_B);

        __m512i va[8];
        __m512i vb[8];
        __m512 vc[COLS];
        __m512 vd1, vs1;

        const __m512i lowMask = _mm512_set1_epi8(0xF);

        auto loadc = [&](auto col) {
            vc[col] = _mm512_setzero_ps();
        };
        Unroll<COLS>{}(loadc);

        auto compute = [&](auto col, auto i) {
            // load a
            if constexpr (col == 0) {
                const int32_t * a_ptr = reinterpret_cast<const int32_t *>(A[0 * KB + i].qs);
                for (int k = 0; k < 8; ++k) {
                    va[k] = _mm512_set1_epi32(a_ptr[k]);
                }
                vd1 = _mm512_set1_ps(GGML_CPU_FP16_TO_FP32(A[0 * KB + i].d));
                vs1 = _mm512_set1_ps(GGML_CPU_FP16_TO_FP32(A[0 * KB + i].s));
            }

            // load b
            const char * b_ptr = B + PACKED_INDEX(col, i, KB, TILE_SIZE);
            for (int k = 0; k < 8; k += 2) {
                __m512i bytes = _mm512_loadu_si512((const __m512i *)(b_ptr + k * 32));
                vb[k + 0] = _mm512_and_si512(bytes, lowMask);
                vb[k + 1] = _mm512_and_si512(_mm512_srli_epi16(bytes, 4), lowMask);
            }
            const int offset = TILE_N * TILE_K / 2;
            const __m512 vd0 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *)(b_ptr + offset)));
            const __m512 vm0 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *)(b_ptr + offset + TILE_N * sizeof(ggml_half))));

            __m512i vsum = _mm512_setzero_si512();
            for (int k = 0; k < 8; ++k) {
                vsum = _mm512_dpbusd_epi32(vsum, vb[k], va[k]);
            }

            vc[col] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(vsum), _mm512_mul_ps(vd0, vd1), vc[col]);
            vc[col] = _mm512_fmadd_ps(vm0, vs1, vc[col]);
        };

        for (int i = 0; i < KB; ++i) {
            Unroll<COLS>{}(compute, i);
        }

        //store to C
        auto storec = [&](auto col) {
            _mm512_storeu_ps((__m512i*)(C + 0 * ldc + col * 16), vc[col]);
        };
        Unroll<COLS>{}(storec);
    }
};

template <int BLOCK_M, int BLOCK_N, int BLOCK_K>
struct tinygemm_kernel_vnni<block_q8_0, block_q8_0, float, BLOCK_M, BLOCK_N, BLOCK_K> {
    static void apply(int KB, const void * RESTRICT _A, const void * RESTRICT _B, float * RESTRICT C, int ldc) {

        constexpr int COLS = BLOCK_N / 16;
        const int TILE_SIZE = TILE_N * sizeof(block_q8_0) + TILE_N * sizeof(int32_t);

        const block_q8_0 * RESTRICT A = static_cast<const block_q8_0 *>(_A);
        const char * RESTRICT B = static_cast<const char *>(_B);

        __m512i va[8];
        __m512i vb[8];
        __m512 vc[COLS];
        __m512 vd1;

        // Notes: s8s8 igemm compensation in avx512-vnni
        // change s8s8 to u8s8 with compensate
        //   a * b = (a + 128) * b - 128 * b
        //   s   s       u       s    u    s
        //
        // (128 * b is pre-computed when packing B to vnni formats)
        //
        const __m512i off = _mm512_set1_epi8(static_cast<char>(0x80));

        auto loadc = [&](auto col) {
            vc[col] = _mm512_setzero_ps();
        };
        Unroll<COLS>{}(loadc);

        auto compute = [&](auto col, auto i) {
            // load a and add offset 128
            if constexpr (col == 0) {
                const int32_t * a_ptr = reinterpret_cast<const int32_t *>(A[0 * KB + i].qs);
                for (int k = 0; k < 8; ++k) {
                    va[k] = _mm512_set1_epi32(a_ptr[k]);
                    va[k] = _mm512_add_epi8(va[k], off);
                }
                vd1 = _mm512_set1_ps(GGML_CPU_FP16_TO_FP32(A[0 * KB + i].d));
            }

            // load b
            const char * b_ptr = B + PACKED_INDEX(col, i, KB, TILE_SIZE);
            for (int k = 0; k < 8; ++k) {
                vb[k] = _mm512_loadu_si512((const __m512i *)(b_ptr + k * 64));
            }
            const int offset = TILE_N * TILE_K;
            const __m512 vd0 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *)(b_ptr + offset)));
            const int offset2 = TILE_N * TILE_K + TILE_N * sizeof(ggml_half);
            const __m512i vcomp = _mm512_loadu_si512((const __m512i *)(b_ptr + offset2));

            __m512i vsum = _mm512_setzero_si512();
            for (int k = 0; k < 8; ++k) {
                vsum = _mm512_dpbusd_epi32(vsum, va[k], vb[k]);
            }
            vsum = _mm512_sub_epi32(vsum, vcomp);

            vc[col] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(vsum), _mm512_mul_ps(vd0, vd1), vc[col]);
        };

        for (int i = 0; i < KB; ++i) {
            Unroll<COLS>{}(compute, i);
        }

        //store to C
        auto storec = [&](auto col) {
            _mm512_storeu_ps((__m512i*)(C + 0 * ldc + col * 16), vc[col]);
        };
        Unroll<COLS>{}(storec);
    }
};

template <int BLOCK_M, int BLOCK_N, int BLOCK_K>
struct tinygemm_kernel_vnni<block_q8_K, block_q4_K, float, BLOCK_M, BLOCK_N, BLOCK_K> {
    static void apply(int KB, const void * RESTRICT _A, const void * RESTRICT _B, float * RESTRICT C, int ldc) {

        constexpr int COLS = BLOCK_N / 16;
        const int TILE_SIZE = TILE_N * sizeof(block_q4_K) + TILE_N * 4;

        const block_q8_K * RESTRICT A = static_cast<const block_q8_K *>(_A);
        const char * RESTRICT B = static_cast<const char *>(_B);

        // a.qs:   8 groups, 32 bytes each group (m256i)
        __m512i va[8];
        // a.bsum: 8 groups,  2 bytes each group (m128i)
        __m512i va_bsum;
        __m512 vc[COLS];
        __m512 vd1;

        // packed_B:
        const int offset_scales = (QK_K / 2) * TILE_N;
        const int offset_mins   = (QK_K / 2) * TILE_N +  8 * TILE_N;
        const int offset_d0     = (QK_K / 2) * TILE_N + 16 * TILE_N;
        const int offset_dmin   = (QK_K / 2) * TILE_N + 16 * TILE_N + TILE_N * sizeof(ggml_half);

        const __m512i lowMask = _mm512_set1_epi8(0xF);

        auto loadc = [&](auto col) {
            vc[col] = _mm512_setzero_ps();
        };
        Unroll<COLS>{}(loadc);

        // Notes: vnni formats in QK_K
        //   a) quants vnni format
        //     int8  {k/4, n, 4}, viewed as 2d {k/4, 4n}, k = 32
        //     from {16, 32} to {8, 64}
        //
        //   b) min vnni format
        //     int16 {k/2, n, 2}, viewed as 2d {k/2, 2n}, k = 8
        //     from {16,  8} to {4, 32}
        //
        auto compute = [&](auto col, auto i) {
            // load a
            if constexpr (col == 0) {
                for (int k_group = 0; k_group < QK_K / 32; ++k_group) {
                    va[k_group] = _mm512_castsi256_si512(_mm256_loadu_si256((const __m256i *)(A[0 * KB + i].qs + k_group * 32)));
                }
                const __m256i q8sums = _mm256_loadu_si256((const __m256i *)A[0 * KB + i].bsums);
                const __m128i q8s = _mm_hadd_epi16(_mm256_extracti128_si256(q8sums, 0), _mm256_extracti128_si256(q8sums, 1));
                va_bsum = _mm512_castsi128_si512(q8s);
                vd1 = _mm512_set1_ps(A[0 * KB + i].d);
            }

            // step 1: accumultate the quants
            __m512i acc = _mm512_setzero_si512();
            const char * b_ptr = B + PACKED_INDEX(col, i, KB, TILE_SIZE);
            const char * b_qs  = b_ptr;
            for (int k_group = 0; k_group < QK_K / 32; ++k_group) {
                __m512i vsum = _mm512_setzero_si512();
                for (int k = 0; k < 8; k += 2) {
                    __m512i va0 = _mm512_permutexvar_epi32(_mm512_set1_epi32(k + 0), va[k_group]);
                    __m512i va1 = _mm512_permutexvar_epi32(_mm512_set1_epi32(k + 1), va[k_group]);

                    __m512i bytes = _mm512_loadu_si512((const __m512i *)b_qs);
                    __m512i vb0 = _mm512_and_si512(bytes, lowMask);
                    vsum = _mm512_dpbusd_epi32(vsum, vb0, va0);
                    __m512i vb1 = _mm512_and_si512(_mm512_srli_epi16(bytes, 4), lowMask);
                    vsum = _mm512_dpbusd_epi32(vsum, vb1, va1);

                    b_qs += 64;
                }
                // vacc += scale * (q8 @ q4)
                const __m512i vscale = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i *)(b_ptr + offset_scales + k_group * TILE_N)));
                acc = _mm512_add_epi32(acc, _mm512_mullo_epi32(vsum, vscale));
            }
            const __m512 vd0 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *)(b_ptr + offset_d0)));
            vc[col] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(acc), _mm512_mul_ps(vd0, vd1), vc[col]);

            // step 2: accumulate the mins
            __m512i acc_m = _mm512_setzero_si512();
            for (int k = 0; k < 4; ++k) {
                __m512i vmask = _mm512_set1_epi32(k);
                __m512i va = _mm512_permutexvar_epi32(vmask, va_bsum);
                __m512i vb = _mm512_cvtepi8_epi16(_mm256_loadu_si256((const __m256i *)(b_ptr + offset_mins + k * 32)));
                acc_m = _mm512_dpwssds_epi32(acc_m, va, vb);
            }
            const __m512 vdmin = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *)(b_ptr + offset_dmin)));
            vc[col] = _mm512_fnmadd_ps(_mm512_cvtepi32_ps(acc_m), _mm512_mul_ps(vdmin, vd1), vc[col]);
        };

        for (int i = 0; i < KB; ++i) {
            Unroll<COLS>{}(compute, i);
        }

        //store to C
        auto storec = [&](auto col) {
            _mm512_storeu_ps((__m512i*)(C + 0 * ldc + col * 16), vc[col]);
        };
        Unroll<COLS>{}(storec);
    }
};

template <int BLOCK_M, int BLOCK_N, int BLOCK_K>
struct tinygemm_kernel_vnni<block_q8_K, block_q5_K, float, BLOCK_M, BLOCK_N, BLOCK_K> {
    static void apply(int KB, const void * RESTRICT _A, const void * RESTRICT _B, float * RESTRICT C, int ldc) {

        constexpr int COLS = BLOCK_N / 16;
        const int TILE_SIZE = TILE_N * sizeof(block_q5_K) + TILE_N * 4;

        const block_q8_K * RESTRICT A = static_cast<const block_q8_K *>(_A);
        const char * RESTRICT B = static_cast<const char *>(_B);

        // a.qs:   8 groups, 32 bytes each group (m256i)
        __m512i va[8];
        // a.bsum: 8 groups,  2 bytes each group (m128i)
        __m512i va_bsum;
        __m512 vc[COLS];
        __m512 vd1;

        // packed_B:
        const int offset_qh     = (QK_K / 2) * TILE_N;
        const int offset_scales = (QK_K / 2) * TILE_N + (QK_K / 8) * TILE_N;
        const int offset_mins   = (QK_K / 2) * TILE_N + (QK_K / 8) * TILE_N +  8 * TILE_N;
        const int offset_d0     = (QK_K / 2) * TILE_N + (QK_K / 8) * TILE_N + 16 * TILE_N;
        const int offset_dmin   = (QK_K / 2) * TILE_N + (QK_K / 8) * TILE_N + 16 * TILE_N + TILE_N * sizeof(ggml_half);

        const __m512i lowMask = _mm512_set1_epi8(0xF);

        auto loadc = [&](auto col) {
            vc[col] = _mm512_setzero_ps();
        };
        Unroll<COLS>{}(loadc);

        // Q5_K and Q4_K shares the same vnni formats, refer to notes above.
        auto compute = [&](auto col, auto i) {
            // load a
            if constexpr (col == 0) {
                for (int k_group = 0; k_group < QK_K / 32; ++k_group) {
                    va[k_group] = _mm512_castsi256_si512(_mm256_loadu_si256((const __m256i *)(A[0 * KB + i].qs + k_group * 32)));
                }
                const __m256i q8sums = _mm256_loadu_si256((const __m256i *)A[0 * KB + i].bsums);
                const __m128i q8s = _mm_hadd_epi16(_mm256_extracti128_si256(q8sums, 0), _mm256_extracti128_si256(q8sums, 1));
                va_bsum = _mm512_castsi128_si512(q8s);
                vd1 = _mm512_set1_ps(A[0 * KB + i].d);
            }

            // step 1: accumultate the quants
            __m512i acc = _mm512_setzero_si512();
            const char * b_ptr = B + PACKED_INDEX(col, i, KB, TILE_SIZE);
            const char * b_qs  = b_ptr;
            const char * b_qh  = b_ptr + offset_qh;
            for (int k_group = 0; k_group < QK_K / 32; ++k_group) {
                __m512i vsum = _mm512_setzero_si512();
                __m512i hmask0 = _mm512_set1_epi8(0x1);
                __m512i hmask1 = _mm512_set1_epi8(0x2);
                __m512i hbits = _mm512_loadu_si512((const __m512i *)(b_qh + k_group * 64));
                for (int k = 0; k < 8; k += 2) {
                    __m512i va0 = _mm512_permutexvar_epi32(_mm512_set1_epi32(k + 0), va[k_group]);
                    __m512i va1 = _mm512_permutexvar_epi32(_mm512_set1_epi32(k + 1), va[k_group]);

                    __m512i bytes = _mm512_loadu_si512((const __m512i *)b_qs);
                    __m512i vb0 = _mm512_and_si512(bytes, lowMask);
                    __m512i vb1 = _mm512_and_si512(_mm512_srli_epi16(bytes, 4), lowMask);

                    __m512i vh0 = _mm512_slli_epi16(_mm512_srli_epi16(_mm512_and_si512(hbits, hmask0), k), 4);
                    __m512i vh1 = _mm512_slli_epi16(_mm512_srli_epi16(_mm512_and_si512(hbits, hmask1), k + 1), 4);

                    hmask0 = _mm512_slli_epi16(hmask0, 2);
                    hmask1 = _mm512_slli_epi16(hmask1, 2);
                    vb0 = _mm512_add_epi8(vb0, vh0);
                    vb1 = _mm512_add_epi8(vb1, vh1);

                    vsum = _mm512_dpbusd_epi32(vsum, vb0, va0);
                    vsum = _mm512_dpbusd_epi32(vsum, vb1, va1);

                    b_qs += 64;
                }
                // vacc += scale * (q8 @ q5)
                const __m512i vscale = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i *)(b_ptr + offset_scales + k_group * TILE_N)));
                acc = _mm512_add_epi32(acc, _mm512_mullo_epi32(vsum, vscale));
            }
            const __m512 vd0 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *)(b_ptr + offset_d0)));
            vc[col] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(acc), _mm512_mul_ps(vd0, vd1), vc[col]);

            // step 2: accumulate the mins
            __m512i acc_m = _mm512_setzero_si512();
            for (int k = 0; k < 4; ++k) {
                __m512i vmask = _mm512_set1_epi32(k);
                __m512i va = _mm512_permutexvar_epi32(vmask, va_bsum);
                __m512i vb = _mm512_cvtepi8_epi16(_mm256_loadu_si256((const __m256i *)(b_ptr + offset_mins + k * 32)));
                acc_m = _mm512_dpwssds_epi32(acc_m, va, vb);
            }
            const __m512 vdmin = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *)(b_ptr + offset_dmin)));
            vc[col] = _mm512_fnmadd_ps(_mm512_cvtepi32_ps(acc_m), _mm512_mul_ps(vdmin, vd1), vc[col]);
        };

        for (int i = 0; i < KB; ++i) {
            Unroll<COLS>{}(compute, i);
        }

        //store to C
        auto storec = [&](auto col) {
            _mm512_storeu_ps((__m512i*)(C + 0 * ldc + col * 16), vc[col]);
        };
        Unroll<COLS>{}(storec);
    }
};

template <int BLOCK_M, int BLOCK_N, int BLOCK_K>
struct tinygemm_kernel_vnni<block_q8_K, block_q6_K, float, BLOCK_M, BLOCK_N, BLOCK_K> {
    static void apply(int KB, const void * RESTRICT _A, const void * RESTRICT _B, float * RESTRICT C, int ldc) {

        constexpr int COLS = BLOCK_N / 16;
        const int TILE_SIZE = TILE_N * sizeof(block_q6_K);

        const block_q8_K * RESTRICT A = static_cast<const block_q8_K *>(_A);
        const char * RESTRICT B = static_cast<const char *>(_B);

        // load the 256 bytes from A to 4 avx512 vectors
        __m512i va[4];
        __m512 vc[COLS];
        __m512 vd1;

        // packed_B:
        const int offset_qh     = (QK_K / 2) * TILE_N;
        const int offset_scales = (QK_K / 2) * TILE_N + (QK_K / 4) * TILE_N;
        const int offset_d0     = (QK_K / 2) * TILE_N + (QK_K / 4) * TILE_N + 16 * TILE_N;

        // compensation
        __m512i vcomp;

        const __m512i m32s = _mm512_set1_epi32(32);
        const __m512i lowMask = _mm512_set1_epi8(0xF);

        auto loadc = [&](auto col) {
            vc[col] = _mm512_setzero_ps();
        };
        Unroll<COLS>{}(loadc);

        auto compute = [&](auto col, auto i) {
            if constexpr (col == 0) {
                // load a
                va[0] = _mm512_loadu_si512((const __m512i *)(A[0 * KB + i].qs +   0));
                va[1] = _mm512_loadu_si512((const __m512i *)(A[0 * KB + i].qs +  64));
                va[2] = _mm512_loadu_si512((const __m512i *)(A[0 * KB + i].qs + 128));
                va[3] = _mm512_loadu_si512((const __m512i *)(A[0 * KB + i].qs + 192));

                const __m256i q8sums = _mm256_loadu_si256((const __m256i *)A[0 * KB + i].bsums);
                vcomp = _mm512_mullo_epi32(_mm512_cvtepi16_epi32(q8sums), m32s);
                vd1 = _mm512_set1_ps(A[0 * KB + i].d);
            }

            // accmulate the quants
            __m512i acc = _mm512_setzero_si512();
            const char * b_ptr = B + PACKED_INDEX(col, i, KB, TILE_SIZE);
            const char * b_qs = b_ptr;
            const char * b_qh = b_ptr + offset_qh;
            int mask = 0;
            for (int k_group = 0; k_group < QK_K / 16; ++k_group) {
                int r = k_group >> 2;
                __m512i va0 = _mm512_permutexvar_epi32(_mm512_set1_epi32(mask++), va[r]);
                __m512i va1 = _mm512_permutexvar_epi32(_mm512_set1_epi32(mask++), va[r]);

                __m512i vsum = _mm512_setzero_si512();
                __m512i hmask = _mm512_set1_epi8(0x3);

                __m512i bytes = _mm512_loadu_si512(b_qs);
                __m512i hbits = _mm512_loadu_si512(b_qh);
                __m512i vb0 = _mm512_and_si512(bytes, lowMask);
                __m512i vb1 = _mm512_and_si512(_mm512_srli_epi16(bytes, 4), lowMask);
                __m512i vh0 = _mm512_slli_epi16(_mm512_and_si512(hbits, hmask), 4);
                __m512i vh1 = _mm512_slli_epi16(_mm512_and_si512(hbits, _mm512_slli_epi16(hmask, 2)), 2);

                vb0 = _mm512_add_epi8(vb0, vh0);
                vb1 = _mm512_add_epi8(vb1, vh1);
                vsum = _mm512_dpbusd_epi32(vsum, vb0, va0);
                vsum = _mm512_dpbusd_epi32(vsum, vb1, va1);
                b_qs += 64;

                va0 = _mm512_permutexvar_epi32(_mm512_set1_epi32(mask++), va[r]);
                va1 = _mm512_permutexvar_epi32(_mm512_set1_epi32(mask++), va[r]);

                bytes = _mm512_loadu_si512(b_qs);
                vb0 = _mm512_and_si512(bytes, lowMask);
                vb1 = _mm512_and_si512(_mm512_srli_epi16(bytes, 4), lowMask);
                vh0 =                   _mm512_and_si512(hbits, _mm512_slli_epi16(hmask, 4));
                vh1 = _mm512_srli_epi16(_mm512_and_si512(hbits, _mm512_slli_epi16(hmask, 6)), 2);
                vb0 = _mm512_add_epi8(vb0, vh0);
                vb1 = _mm512_add_epi8(vb1, vh1);
                vsum = _mm512_dpbusd_epi32(vsum, vb0, va0);
                vsum = _mm512_dpbusd_epi32(vsum, vb1, va1);
                b_qs += 64;
                b_qh += 64;

                // B * A - 32 * A
                __m512i vmask = _mm512_set1_epi32(k_group);
                vsum = _mm512_sub_epi32(vsum, _mm512_permutexvar_epi32(vmask, vcomp));

                // vacc += scale * (q8 @ q6)
                const __m512i vscale = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i *)(b_ptr + offset_scales + k_group * TILE_N)));
                acc = _mm512_add_epi32(acc, _mm512_mullo_epi32(vsum, vscale));
            }
            const __m512 vd0 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *)(b_ptr + offset_d0)));
            vc[col] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(acc), _mm512_mul_ps(vd0, vd1), vc[col]);
        };

        for (int i = 0; i < KB; ++i) {
            Unroll<COLS>{}(compute, i);
        }

        //store to C
        auto storec = [&](int col) {
            _mm512_storeu_ps((__m512i*)(C + 0 * ldc + col * 16), vc[col]);
        };
        Unroll<COLS>{}(storec);
    }
};

template <int BLOCK_M, int BLOCK_N, int BLOCK_K>
struct tinygemm_kernel_vnni<block_q8_K, block_iq4_xs, float, BLOCK_M, BLOCK_N, BLOCK_K> {
    static void apply(int KB, const void * RESTRICT _A, const void * RESTRICT _B, float * RESTRICT C, int ldc) {

        constexpr int COLS = BLOCK_N / 16;
        const int TILE_SIZE = TILE_N * sizeof(block_iq4_xs) + TILE_N * 2;

        const block_q8_K * RESTRICT A = static_cast<const block_q8_K *>(_A);
        const char * RESTRICT B = static_cast<const char *>(_B);

        // load the 256 bytes from A to 4 avx512 vectors
        __m512i va[4];
        __m512 vc[COLS];
        __m512 vd1;

        // packed_B:
        const int offset_scales = (QK_K / 2) * TILE_N ;
        const int offset_d0     = (QK_K / 2) * TILE_N + 8 * TILE_N;

        // compensation
        __m512i vcomp;

        const __m256i m128s = _mm256_set1_epi16(128);
        const __m512i lowMask = _mm512_set1_epi8(0xF);

        const __m512i values128 = _mm512_set_epi8(
            113, 89, 69, 53, 38, 25, 13, 1, -10, -22, -35, -49, -65, -83, -104, -127,
            113, 89, 69, 53, 38, 25, 13, 1, -10, -22, -35, -49, -65, -83, -104, -127,
            113, 89, 69, 53, 38, 25, 13, 1, -10, -22, -35, -49, -65, -83, -104, -127,
            113, 89, 69, 53, 38, 25, 13, 1, -10, -22, -35, -49, -65, -83, -104, -127
        );
        const __m512i off = _mm512_set1_epi8(static_cast<char>(0x80));
        const __m512i values256 = _mm512_add_epi8(values128, off);

        auto loadc = [&](auto col) {
            vc[col] = _mm512_setzero_ps();
        };
        Unroll<COLS>{}(loadc);

        auto compute = [&](auto col, auto i) {
            if constexpr (col == 0) {
                // load a
                va[0] = _mm512_loadu_si512((const __m512i *)(A[0 * KB + i].qs +   0));
                va[1] = _mm512_loadu_si512((const __m512i *)(A[0 * KB + i].qs +  64));
                va[2] = _mm512_loadu_si512((const __m512i *)(A[0 * KB + i].qs + 128));
                va[3] = _mm512_loadu_si512((const __m512i *)(A[0 * KB + i].qs + 192));

                // compensation: 128 * A
                const __m256i q8sums = _mm256_loadu_si256((const __m256i *)A[0 * KB + i].bsums);
                vcomp = _mm512_castsi256_si512(_mm256_madd_epi16(q8sums, m128s));
                vd1 = _mm512_set1_ps(A[0 * KB + i].d);
            }

            // accmulate the quants
            __m512i acc = _mm512_setzero_si512();
            const char * b_ptr = B + PACKED_INDEX(col, i, KB, TILE_SIZE);
            const char * b_qs = b_ptr;
            int mask = 0;
            for (int k_group = 0; k_group < QK_K / 32; ++k_group) {
                int r = k_group >> 1;
                __m512i vmask = _mm512_set1_epi32(k_group);
                __m512i vsum = _mm512_setzero_si512();
                for (int k = 0; k < 8; k += 2) {
                    __m512i va0 = _mm512_permutexvar_epi32(_mm512_set1_epi32(mask++), va[r]);
                    __m512i va1 = _mm512_permutexvar_epi32(_mm512_set1_epi32(mask++), va[r]);

                    __m512i bytes = _mm512_loadu_si512(b_qs);
                    __m512i vb0 = _mm512_shuffle_epi8(values256, _mm512_and_si512(bytes, lowMask));
                    __m512i vb1 = _mm512_shuffle_epi8(values256, _mm512_and_si512(_mm512_srli_epi16(bytes, 4), lowMask));

                    vsum = _mm512_dpbusd_epi32(vsum, vb0, va0);
                    vsum = _mm512_dpbusd_epi32(vsum, vb1, va1);
                    b_qs += 64;
                }
                // (B + 128) * A - 128 * A
                vsum = _mm512_sub_epi32(vsum, _mm512_permutexvar_epi32(vmask, vcomp));

                // vacc += scale * (q8 @ q4)
                const __m512i vscale = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i *)(b_ptr + offset_scales + k_group * TILE_N)));
                acc = _mm512_add_epi32(acc, _mm512_mullo_epi32(vsum, vscale));
            }
            const __m512 vd0 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *)(b_ptr + offset_d0)));
            vc[col] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(acc), _mm512_mul_ps(vd0, vd1), vc[col]);
        };

        for (int i = 0; i < KB; ++i) {
            Unroll<COLS>{}(compute, i);
        }

        //store to C
        auto storec = [&](auto col) {
            _mm512_storeu_ps((__m512i*)(C + 0 * ldc + col * 16), vc[col]);
        };
        Unroll<COLS>{}(storec);
    }
};

// IQ2_XXS / IQ3_XXS M==1: quants are decoded 8-bit in vnni layout (512B per
// 32-value group). Same compensation trick as IQ4_XS: bias the signed B values
// by +128 to make them the unsigned operand of dpbusd, then subtract
// 128 * (per-group bsum of A).
template <int BLOCK_M, int BLOCK_N, int BLOCK_K>
struct tinygemm_kernel_vnni<block_q8_K, block_iq2_xxs, float, BLOCK_M, BLOCK_N, BLOCK_K> {
    static void apply(int KB, const void * RESTRICT _A, const void * RESTRICT _B, float * RESTRICT C, int ldc) {

        constexpr int COLS = BLOCK_N / 16;
        const int TILE_SIZE = TILE_N * (QK_K + 8 + 2);

        const block_q8_K * RESTRICT A = static_cast<const block_q8_K *>(_A);
        const char * RESTRICT B = static_cast<const char *>(_B);

        __m512i va[4];
        __m512 vc[COLS];
        __m512 vd1;

        // packed_B:
        const int offset_scales = QK_K * TILE_N;
        const int offset_d0     = QK_K * TILE_N + 8 * TILE_N;

        // compensation
        __m512i vcomp;

        const __m256i m128s = _mm256_set1_epi16(128);
        const __m512i off = _mm512_set1_epi8(static_cast<char>(0x80));

        auto loadc = [&](auto col) {
            vc[col] = _mm512_setzero_ps();
        };
        Unroll<COLS>{}(loadc);

        auto compute = [&](auto col, auto i) {
            if constexpr (col == 0) {
                // load a
                va[0] = _mm512_loadu_si512((const __m512i *)(A[0 * KB + i].qs +   0));
                va[1] = _mm512_loadu_si512((const __m512i *)(A[0 * KB + i].qs +  64));
                va[2] = _mm512_loadu_si512((const __m512i *)(A[0 * KB + i].qs + 128));
                va[3] = _mm512_loadu_si512((const __m512i *)(A[0 * KB + i].qs + 192));

                // compensation: 128 * A (one int32 per 32-value group)
                const __m256i q8sums = _mm256_loadu_si256((const __m256i *)A[0 * KB + i].bsums);
                vcomp = _mm512_castsi256_si512(_mm256_madd_epi16(q8sums, m128s));
                vd1 = _mm512_set1_ps(A[0 * KB + i].d);
            }

            // accumulate the quants
            __m512i acc = _mm512_setzero_si512();
            const char * b_ptr = B + PACKED_INDEX(col, i, KB, TILE_SIZE);
            const char * b_qs = b_ptr;
            int mask = 0;
            for (int k_group = 0; k_group < QK_K / 32; ++k_group) {
                int r = k_group >> 1;
                __m512i vmask = _mm512_set1_epi32(k_group);
                __m512i vsum = _mm512_setzero_si512();
                for (int k = 0; k < 8; ++k) {
                    __m512i va0 = _mm512_permutexvar_epi32(_mm512_set1_epi32(mask++), va[r]);
                    __m512i vb = _mm512_add_epi8(_mm512_loadu_si512(b_qs), off);
                    vsum = _mm512_dpbusd_epi32(vsum, vb, va0);
                    b_qs += 64;
                }
                // (B + 128) * A - 128 * A
                vsum = _mm512_sub_epi32(vsum, _mm512_permutexvar_epi32(vmask, vcomp));

                // vacc += scale * (q8 @ q2)
                const __m512i vscale = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i *)(b_ptr + offset_scales + k_group * TILE_N)));
                acc = _mm512_add_epi32(acc, _mm512_mullo_epi32(vsum, vscale));
            }
            const __m512 vd0 = _mm512_mul_ps(
                _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *)(b_ptr + offset_d0))), _mm512_set1_ps(0.125f));
            vc[col] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(acc), _mm512_mul_ps(vd0, vd1), vc[col]);
        };

        for (int i = 0; i < KB; ++i) {
            Unroll<COLS>{}(compute, i);
        }

        //store to C
        auto storec = [&](auto col) {
            _mm512_storeu_ps((__m512i*)(C + 0 * ldc + col * 16), vc[col]);
        };
        Unroll<COLS>{}(storec);
    }
};

// IQ3_XXS M==1: identical to IQ2_XXS except the global factor is 0.25f
template <int BLOCK_M, int BLOCK_N, int BLOCK_K>
struct tinygemm_kernel_vnni<block_q8_K, block_iq3_xxs, float, BLOCK_M, BLOCK_N, BLOCK_K> {
    static void apply(int KB, const void * RESTRICT _A, const void * RESTRICT _B, float * RESTRICT C, int ldc) {

        constexpr int COLS = BLOCK_N / 16;
        const int TILE_SIZE = TILE_N * (QK_K + 8 + 2);

        const block_q8_K * RESTRICT A = static_cast<const block_q8_K *>(_A);
        const char * RESTRICT B = static_cast<const char *>(_B);

        __m512i va[4];
        __m512 vc[COLS];
        __m512 vd1;

        // packed_B:
        const int offset_scales = QK_K * TILE_N;
        const int offset_d0     = QK_K * TILE_N + 8 * TILE_N;

        // compensation
        __m512i vcomp;

        const __m256i m128s = _mm256_set1_epi16(128);
        const __m512i off = _mm512_set1_epi8(static_cast<char>(0x80));

        auto loadc = [&](auto col) {
            vc[col] = _mm512_setzero_ps();
        };
        Unroll<COLS>{}(loadc);

        auto compute = [&](auto col, auto i) {
            if constexpr (col == 0) {
                // load a
                va[0] = _mm512_loadu_si512((const __m512i *)(A[0 * KB + i].qs +   0));
                va[1] = _mm512_loadu_si512((const __m512i *)(A[0 * KB + i].qs +  64));
                va[2] = _mm512_loadu_si512((const __m512i *)(A[0 * KB + i].qs + 128));
                va[3] = _mm512_loadu_si512((const __m512i *)(A[0 * KB + i].qs + 192));

                // compensation: 128 * A (one int32 per 32-value group)
                const __m256i q8sums = _mm256_loadu_si256((const __m256i *)A[0 * KB + i].bsums);
                vcomp = _mm512_castsi256_si512(_mm256_madd_epi16(q8sums, m128s));
                vd1 = _mm512_set1_ps(A[0 * KB + i].d);
            }

            // accumulate the quants
            __m512i acc = _mm512_setzero_si512();
            const char * b_ptr = B + PACKED_INDEX(col, i, KB, TILE_SIZE);
            const char * b_qs = b_ptr;
            int mask = 0;
            for (int k_group = 0; k_group < QK_K / 32; ++k_group) {
                int r = k_group >> 1;
                __m512i vmask = _mm512_set1_epi32(k_group);
                __m512i vsum = _mm512_setzero_si512();
                for (int k = 0; k < 8; ++k) {
                    __m512i va0 = _mm512_permutexvar_epi32(_mm512_set1_epi32(mask++), va[r]);
                    __m512i vb = _mm512_add_epi8(_mm512_loadu_si512(b_qs), off);
                    vsum = _mm512_dpbusd_epi32(vsum, vb, va0);
                    b_qs += 64;
                }
                // (B + 128) * A - 128 * A
                vsum = _mm512_sub_epi32(vsum, _mm512_permutexvar_epi32(vmask, vcomp));

                // vacc += scale * (q8 @ q3)
                const __m512i vscale = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i *)(b_ptr + offset_scales + k_group * TILE_N)));
                acc = _mm512_add_epi32(acc, _mm512_mullo_epi32(vsum, vscale));
            }
            const __m512 vd0 = _mm512_mul_ps(
                _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *)(b_ptr + offset_d0))), _mm512_set1_ps(0.25f));
            vc[col] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(acc), _mm512_mul_ps(vd0, vd1), vc[col]);
        };

        for (int i = 0; i < KB; ++i) {
            Unroll<COLS>{}(compute, i);
        }

        //store to C
        auto storec = [&](auto col) {
            _mm512_storeu_ps((__m512i*)(C + 0 * ldc + col * 16), vc[col]);
        };
        Unroll<COLS>{}(storec);
    }
};

// IQ2_XS M==1: 16-value group granularity (like Q6_K), decoded signed quants,
// +128 bias compensation per 16-value group.
template <int BLOCK_M, int BLOCK_N, int BLOCK_K>
struct tinygemm_kernel_vnni<block_q8_K, block_iq2_xs, float, BLOCK_M, BLOCK_N, BLOCK_K> {
    static void apply(int KB, const void * RESTRICT _A, const void * RESTRICT _B, float * RESTRICT C, int ldc) {

        constexpr int COLS = BLOCK_N / 16;
        const int TILE_SIZE = TILE_N * (QK_K + 16 + 2);

        const block_q8_K * RESTRICT A = static_cast<const block_q8_K *>(_A);
        const char * RESTRICT B = static_cast<const char *>(_B);

        __m512i va[4];
        __m512 vc[COLS];
        __m512 vd1;

        // packed_B:
        const int offset_scales = QK_K * TILE_N;
        const int offset_d0     = QK_K * TILE_N + 16 * TILE_N;

        // compensation
        __m512i vcomp;

        const __m512i m128 = _mm512_set1_epi32(128);
        const __m512i off = _mm512_set1_epi8(static_cast<char>(0x80));

        auto loadc = [&](auto col) {
            vc[col] = _mm512_setzero_ps();
        };
        Unroll<COLS>{}(loadc);

        auto compute = [&](auto col, auto i) {
            if constexpr (col == 0) {
                // load a
                va[0] = _mm512_loadu_si512((const __m512i *)(A[0 * KB + i].qs +   0));
                va[1] = _mm512_loadu_si512((const __m512i *)(A[0 * KB + i].qs +  64));
                va[2] = _mm512_loadu_si512((const __m512i *)(A[0 * KB + i].qs + 128));
                va[3] = _mm512_loadu_si512((const __m512i *)(A[0 * KB + i].qs + 192));

                // compensation: 128 * A (one int32 per 16-value group)
                const __m256i q8sums = _mm256_loadu_si256((const __m256i *)A[0 * KB + i].bsums);
                vcomp = _mm512_mullo_epi32(_mm512_cvtepi16_epi32(q8sums), m128);
                vd1 = _mm512_set1_ps(A[0 * KB + i].d);
            }

            // accumulate the quants
            __m512i acc = _mm512_setzero_si512();
            const char * b_ptr = B + PACKED_INDEX(col, i, KB, TILE_SIZE);
            int mask = 0;
            for (int k_group = 0; k_group < QK_K / 16; ++k_group) {
                int r = k_group >> 2;
                const char * b_qs = b_ptr + k_group * (16 * TILE_N);
                __m512i vsum = _mm512_setzero_si512();
                for (int k = 0; k < 4; ++k) {
                    __m512i va0 = _mm512_permutexvar_epi32(_mm512_set1_epi32(mask++), va[r]);
                    __m512i vb = _mm512_add_epi8(_mm512_loadu_si512(b_qs), off);
                    vsum = _mm512_dpbusd_epi32(vsum, vb, va0);
                    b_qs += 64;
                }
                // (B + 128) * A - 128 * A
                vsum = _mm512_sub_epi32(vsum, _mm512_permutexvar_epi32(_mm512_set1_epi32(k_group), vcomp));

                // vacc += scale * (q8 @ q2)
                const __m512i vscale = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i *)(b_ptr + offset_scales + k_group * TILE_N)));
                acc = _mm512_add_epi32(acc, _mm512_mullo_epi32(vsum, vscale));
            }
            const __m512 vd0 = _mm512_mul_ps(
                _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *)(b_ptr + offset_d0))), _mm512_set1_ps(0.125f));
            vc[col] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(acc), _mm512_mul_ps(vd0, vd1), vc[col]);
        };

        for (int i = 0; i < KB; ++i) {
            Unroll<COLS>{}(compute, i);
        }

        //store to C
        auto storec = [&](auto col) {
            _mm512_storeu_ps((__m512i*)(C + 0 * ldc + col * 16), vc[col]);
        };
        Unroll<COLS>{}(storec);
    }
};

// Q2_K M==1: 16-value groups, decoded quants are unsigned (0..3) so dpbusd
// needs no compensation; mins term mirrors the Q4_K kernel but runs over 16
// groups (8 dpwssds iterations over the raw bsums dwords).
template <int BLOCK_M, int BLOCK_N, int BLOCK_K>
struct tinygemm_kernel_vnni<block_q8_K, block_q2_K, float, BLOCK_M, BLOCK_N, BLOCK_K> {
    static void apply(int KB, const void * RESTRICT _A, const void * RESTRICT _B, float * RESTRICT C, int ldc) {

        constexpr int COLS = BLOCK_N / 16;
        const int TILE_SIZE = TILE_N * (QK_K + 16 + 16 + 2 + 2);

        const block_q8_K * RESTRICT A = static_cast<const block_q8_K *>(_A);
        const char * RESTRICT B = static_cast<const char *>(_B);

        __m512i va[4];
        // a.bsum: 16 groups, 2 bytes each group (m256i)
        __m512i va_bsum;
        __m512 vc[COLS];
        __m512 vd1;

        // packed_B:
        const int offset_scales = QK_K * TILE_N;
        const int offset_mins   = QK_K * TILE_N + 16 * TILE_N;
        const int offset_d0     = QK_K * TILE_N + 32 * TILE_N;
        const int offset_dmin   = QK_K * TILE_N + 32 * TILE_N + TILE_N * sizeof(ggml_half);

        auto loadc = [&](auto col) {
            vc[col] = _mm512_setzero_ps();
        };
        Unroll<COLS>{}(loadc);

        auto compute = [&](auto col, auto i) {
            if constexpr (col == 0) {
                // load a
                va[0] = _mm512_loadu_si512((const __m512i *)(A[0 * KB + i].qs +   0));
                va[1] = _mm512_loadu_si512((const __m512i *)(A[0 * KB + i].qs +  64));
                va[2] = _mm512_loadu_si512((const __m512i *)(A[0 * KB + i].qs + 128));
                va[3] = _mm512_loadu_si512((const __m512i *)(A[0 * KB + i].qs + 192));

                va_bsum = _mm512_castsi256_si512(_mm256_loadu_si256((const __m256i *)A[0 * KB + i].bsums));
                vd1 = _mm512_set1_ps(A[0 * KB + i].d);
            }

            // step 1: accumulate the quants
            __m512i acc = _mm512_setzero_si512();
            const char * b_ptr = B + PACKED_INDEX(col, i, KB, TILE_SIZE);
            int mask = 0;
            for (int k_group = 0; k_group < QK_K / 16; ++k_group) {
                int r = k_group >> 2;
                const char * b_qs = b_ptr + k_group * (16 * TILE_N);
                __m512i vsum = _mm512_setzero_si512();
                for (int k = 0; k < 4; ++k) {
                    __m512i va0 = _mm512_permutexvar_epi32(_mm512_set1_epi32(mask++), va[r]);
                    __m512i vb = _mm512_loadu_si512(b_qs);
                    vsum = _mm512_dpbusd_epi32(vsum, vb, va0);
                    b_qs += 64;
                }
                // vacc += scale * (q8 @ q2)
                const __m512i vscale = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i *)(b_ptr + offset_scales + k_group * TILE_N)));
                acc = _mm512_add_epi32(acc, _mm512_mullo_epi32(vsum, vscale));
            }
            const __m512 vd0 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *)(b_ptr + offset_d0)));
            vc[col] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(acc), _mm512_mul_ps(vd0, vd1), vc[col]);

            // step 2: accumulate the mins
            __m512i acc_m = _mm512_setzero_si512();
            for (int k = 0; k < 8; ++k) {
                __m512i va = _mm512_permutexvar_epi32(_mm512_set1_epi32(k), va_bsum);
                __m512i vb = _mm512_cvtepi8_epi16(_mm256_loadu_si256((const __m256i *)(b_ptr + offset_mins + k * 32)));
                acc_m = _mm512_dpwssds_epi32(acc_m, va, vb);
            }
            const __m512 vdmin = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *)(b_ptr + offset_dmin)));
            vc[col] = _mm512_fnmadd_ps(_mm512_cvtepi32_ps(acc_m), _mm512_mul_ps(vdmin, vd1), vc[col]);
        };

        for (int i = 0; i < KB; ++i) {
            Unroll<COLS>{}(compute, i);
        }

        //store to C
        auto storec = [&](auto col) {
            _mm512_storeu_ps((__m512i*)(C + 0 * ldc + col * 16), vc[col]);
        };
        Unroll<COLS>{}(storec);
    }
};

#define LAUNCH_TINYGEMM_KERNEL_VNNI(NB_SIZE)                                                   \
    tinygemm_kernel_vnni<vec_dot_type, type, float, 1, NB_SIZE, blck_size>::apply(             \
        KB, wdata_batch,                                                                       \
        (const char *)src0->data + src0_offset + PACKED_INDEX(nb * kTilesN, 0, KB, TILE_SIZE), \
        (float *) dst->data + dst_offset + nb_start, ldc)

template <typename TA, typename TB, typename TC, int BLOCK_K,
          typename std::enable_if<!is_type_qkk<TB>::value, int>::type = 0>
void tinygemm_kernel_amx(int M, int N, int KB, const void * RESTRICT _A, const void * RESTRICT _B, TC * RESTRICT C, int ldc) {
    using packed_B_t = packed_B_type<TB>;
    const int TILE_SIZE = get_tile_size<TB>();
    const bool need_unpack = do_unpack<TB>::value;

    GGML_ASSERT(M <= 2 * TILE_M && N == 2 * TILE_N);
    const TA * RESTRICT A = static_cast<const TA *>(_A);
    const char * RESTRICT B = static_cast<const char *>(_B);

    const int m0 = std::min(M, TILE_M);
    const int m1 = std::max(M - TILE_M, 0);
    const int lda = KB * sizeof(TA);
    //const int ldb = KB * sizeof(TB);

    alignas(64) static thread_local packed_B_t Tile0[TILE_N * TILE_K];
    alignas(64) static thread_local packed_B_t Tile1[TILE_N * TILE_K];
    alignas(64) static thread_local int8_t Tile23[TILE_M * TILE_K];

    alignas(64) static thread_local int32_t TileC0[TILE_M * TILE_N * 4];
    alignas(64) static thread_local int32_t TileC1[TILE_M * TILE_N * 4];

    // double buffering C to interleave avx512 and amx
    int32_t * C_cur = TileC0;
    int32_t * C_pre = TileC1;

    auto Tile4 = [&](int32_t * base) { return base; };
    auto Tile5 = [&](int32_t * base) { return base + TILE_M * TILE_N; };
    auto Tile6 = [&](int32_t * base) { return base + 2 * TILE_M * TILE_N; };
    auto Tile7 = [&](int32_t * base) { return base + 3 * TILE_M * TILE_N; };

    if (M == 2 * TILE_M) {
        // i = 0
        const char * B_blk0 = B + PACKED_INDEX(0, 0, KB, TILE_SIZE);
        const char * B_blk1 = B + PACKED_INDEX(1, 0, KB, TILE_SIZE);
        if (need_unpack) {
            unpack_B<TB>(Tile0, B_blk0);
            _tile_loadd(TMM0, Tile0, TILE_N * VNNI_BLK);
        } else {
            _tile_loadd(TMM0, B_blk0, TILE_N * VNNI_BLK);
        }

        _tile_zero(TMM4);
        _tile_loadd(TMM2, A[0].qs, lda);
        _tile_dpbssd(TMM4, TMM2, TMM0);
        _tile_stored(TMM4, Tile4(C_pre), TILE_N * sizeof(int32_t));

        _tile_zero(TMM5);
        _tile_loadd(TMM3, A[TILE_M * KB + 0].qs, lda);
        _tile_dpbssd(TMM5, TMM3, TMM0);
        _tile_stored(TMM5, Tile5(C_pre), TILE_N * sizeof(int32_t));

        if (need_unpack) {
            unpack_B<TB>(Tile1, B_blk1);
            _tile_loadd(TMM1, Tile1, TILE_N * VNNI_BLK);
        } else {
            _tile_loadd(TMM1, B_blk1, TILE_N * VNNI_BLK);
        }

        _tile_zero(TMM6);
        _tile_dpbssd(TMM6, TMM2, TMM1);
        _tile_stored(TMM6, Tile6(C_pre), TILE_N * sizeof(int32_t));

        _tile_zero(TMM7);
        _tile_dpbssd(TMM7, TMM3, TMM1);
        _tile_stored(TMM7, Tile7(C_pre), TILE_N * sizeof(int32_t));

        for (int i = 1; i < KB; ++i) {
            // index of previous iter
            const int ii = i - 1;
            const char * B_blk0 = B + PACKED_INDEX(0, i, KB, TILE_SIZE);
            const char * B_blk1 = B + PACKED_INDEX(1, i, KB, TILE_SIZE);
            GGML_DISPATCH_BOOL(ii > 0, is_acc, [&] {
                if (need_unpack) {
                    unpack_B<TB>(Tile0, B_blk0);
                    _tile_loadd(TMM0, Tile0, TILE_N * VNNI_BLK);
                } else {
                    _tile_loadd(TMM0, B_blk0, TILE_N * VNNI_BLK);
                }
                _tile_zero(TMM4);
                _tile_loadd(TMM2, A[i].qs, lda);
                acc_C<TA, TB, is_acc>::apply(C, ldc, Tile4(C_pre), &A[ii], KB, B + PACKED_INDEX(0, ii, KB, TILE_SIZE), TILE_M);

                _tile_dpbssd(TMM4, TMM2, TMM0);
                _tile_stored(TMM4, Tile4(C_cur), TILE_N * sizeof(int32_t));

                _tile_zero(TMM5);
                _tile_loadd(TMM3, A[TILE_M * KB + i].qs, lda);
                acc_C<TA, TB, is_acc>::apply(C + TILE_M * ldc, ldc, Tile5(C_pre), &A[TILE_M * KB + ii], KB, B + PACKED_INDEX(0, ii, KB, TILE_SIZE), TILE_M);

                _tile_dpbssd(TMM5, TMM3, TMM0);
                _tile_stored(TMM5, Tile5(C_cur), TILE_N * sizeof(int32_t));

                if (need_unpack) {
                    unpack_B<TB>(Tile1, B_blk1);
                    _tile_loadd(TMM1, Tile1, TILE_N * VNNI_BLK);
                } else {
                    _tile_loadd(TMM1, B_blk1, TILE_N * VNNI_BLK);
                }
                _tile_zero(TMM6);
                acc_C<TA, TB, is_acc>::apply(C + TILE_N, ldc, Tile6(C_pre), &A[ii], KB, B + PACKED_INDEX(1, ii, KB, TILE_SIZE), TILE_M);

                _tile_dpbssd(TMM6, TMM2, TMM1);
                _tile_stored(TMM6, Tile6(C_cur), TILE_N * sizeof(int32_t));

                _tile_zero(TMM7);
                acc_C<TA, TB, is_acc>::apply(C + TILE_M * ldc + TILE_N, ldc, Tile7(C_pre), &A[TILE_M * KB + ii], KB, B + PACKED_INDEX(1, ii, KB, TILE_SIZE), TILE_M);

                _tile_dpbssd(TMM7, TMM3, TMM1);
                _tile_stored(TMM7, Tile7(C_cur), TILE_N * sizeof(int32_t));

                std::swap(C_cur, C_pre);
            });
        }
        // final accumulation
        {
            int ii = KB - 1;
            acc_C<TA, TB, true>::apply(C, ldc, Tile4(C_pre), &A[ii], KB, B + PACKED_INDEX(0, ii, KB, TILE_SIZE), TILE_M);
            acc_C<TA, TB, true>::apply(C + TILE_M * ldc, ldc, Tile5(C_pre), &A[TILE_M * KB + ii], KB, B + PACKED_INDEX(0, ii, KB, TILE_SIZE), TILE_M);
            acc_C<TA, TB, true>::apply(C + TILE_N, ldc, Tile6(C_pre), &A[ii], KB, B + PACKED_INDEX(1, ii, KB, TILE_SIZE), TILE_M);
            acc_C<TA, TB, true>::apply(C + TILE_M * ldc + TILE_N, ldc, Tile7(C_pre), &A[TILE_M * KB + ii], KB, B + PACKED_INDEX(1, ii, KB, TILE_SIZE), TILE_M);
        }
    } else {
        for (int i = 0; i < KB; ++i) {
            _tile_zero(TMM4);
            _tile_zero(TMM6);
            if (m1 != 0) {
                _tile_zero(TMM5);
                _tile_zero(TMM7);
            }

            const char * B_blk0 = B + PACKED_INDEX(0, i, KB, TILE_SIZE);
            const char * B_blk1 = B + PACKED_INDEX(1, i, KB, TILE_SIZE);
            if (need_unpack) {
                unpack_B<TB>(Tile0, B_blk0);
                _tile_loadd(TMM0, Tile0, TILE_N * VNNI_BLK);
            } else {
                _tile_loadd(TMM0, B_blk0, TILE_N * VNNI_BLK);
            }

            if (need_unpack) {
                unpack_B<TB>(Tile1, B_blk1);
                _tile_loadd(TMM1, Tile1, TILE_N * VNNI_BLK);
            } else {
                _tile_loadd(TMM1, B_blk1, TILE_N * VNNI_BLK);
            }

            if (m0 == TILE_M) {
                _tile_loadd(TMM2, A[i].qs, lda);
            } else {
                unpack_A(Tile23, &A[i], KB, m0);
                _tile_loadd(TMM2, Tile23, TILE_K);
            }

            _tile_dpbssd(TMM4, TMM2, TMM0);
            _tile_dpbssd(TMM6, TMM2, TMM1);

            _tile_stored(TMM4, Tile4(C_cur), TILE_N * sizeof(int32_t));
            _tile_stored(TMM6, Tile6(C_cur), TILE_N * sizeof(int32_t));

            GGML_DISPATCH_BOOL(i > 0, is_acc, [&] {
                acc_C<TA, TB, is_acc>::apply(C,          ldc, Tile4(C_cur), &A[i], KB, B + PACKED_INDEX(0, i, KB, TILE_SIZE), m0);
                acc_C<TA, TB, is_acc>::apply(C + TILE_N, ldc, Tile6(C_cur), &A[i], KB, B + PACKED_INDEX(1, i, KB, TILE_SIZE), m0);
            });

            if (m1 != 0) {
                unpack_A(Tile23, &A[TILE_M * KB + i], KB, m1);
                _tile_loadd(TMM3, Tile23, TILE_K);

                _tile_dpbssd(TMM5, TMM3, TMM0);
                _tile_dpbssd(TMM7, TMM3, TMM1);
                _tile_stored(TMM5, Tile5(C_cur), TILE_N * sizeof(int32_t));
                _tile_stored(TMM7, Tile7(C_cur), TILE_N * sizeof(int32_t));
                GGML_DISPATCH_BOOL(i > 0, is_acc, [&] {
                    acc_C<TA, TB, is_acc>::apply(C + TILE_M * ldc,          ldc, Tile5(C_cur), &A[TILE_M * KB + i], KB, B + PACKED_INDEX(0, i, KB, TILE_SIZE), m1);
                    acc_C<TA, TB, is_acc>::apply(C + TILE_M * ldc + TILE_N, ldc, Tile7(C_cur), &A[TILE_M * KB + i], KB, B + PACKED_INDEX(1, i, KB, TILE_SIZE), m1);
                });
            }
        }
    }
    return;
}

template <typename TA, typename TB, typename TC, int BLOCK_K,
          typename std::enable_if<is_type_qkk<TB>::value, int>::type = 0>
void tinygemm_kernel_amx(int M, int N, int KB, const void * RESTRICT _A, const void * RESTRICT _B, float * RESTRICT C, int ldc) {
    static_assert(std::is_same<TA, block_q8_K>::value);
    const int TILE_SIZE = get_tile_size<TB>();

    GGML_ASSERT(M <= 2 * TILE_M && N == 2 * TILE_N);
    const TA * RESTRICT A = static_cast<const TA *>(_A);
    const char * RESTRICT B = static_cast<const char *>(_B);

    const int m0 = std::min(M, TILE_M);
    const int m1 = std::max(M - TILE_M, 0);
    //const int lda = KB * sizeof(TA);

    alignas(64) static thread_local int8_t Tile0[TILE_N * TILE_K];
    alignas(64) static thread_local int8_t Tile1[TILE_N * TILE_K];
    alignas(64) static thread_local int8_t Tile23[TILE_M * TILE_K];

    // mat mul result for each group
    alignas(64) static thread_local int32_t Tile4[TILE_M * TILE_N];
    alignas(64) static thread_local int32_t Tile5[TILE_M * TILE_N];
    alignas(64) static thread_local int32_t Tile6[TILE_M * TILE_N];
    alignas(64) static thread_local int32_t Tile7[TILE_M * TILE_N];

    // sum of each QK_K block, contains 8 groups, int32
    alignas(64) static thread_local int32_t Sumi4[TILE_M * TILE_N];
    alignas(64) static thread_local int32_t Sumi5[TILE_M * TILE_N];
    alignas(64) static thread_local int32_t Sumi6[TILE_M * TILE_N];
    alignas(64) static thread_local int32_t Sumi7[TILE_M * TILE_N];

    const int k_group_size = qkk_group_16<TB>::value ? 16 : 32;
    for (int i = 0; i < KB; ++i) {
        // step 1: accumulate the quants across 8 groups, each group with 32
        for (int k = 0; k < QK_K / k_group_size; ++k) {
            GGML_DISPATCH_BOOL(k > 0, is_acc, [&] {
                _tile_zero(TMM4);
                _tile_zero(TMM6);

                unpack_B<TB>(Tile0, B + PACKED_INDEX(0, i, KB, TILE_SIZE), k);
                _tile_loadd(TMM0, Tile0, TILE_N * VNNI_BLK);

                unpack_B<TB>(Tile1, B + PACKED_INDEX(1, i, KB, TILE_SIZE), k);
                _tile_loadd(TMM1, Tile1, TILE_N * VNNI_BLK);

                unpack_A<TB>(Tile23, &A[i], KB, k, m0);
                _tile_loadd(TMM2, Tile23, TILE_K);

                _tile_dpbssd(TMM4, TMM2, TMM0);
                _tile_dpbssd(TMM6, TMM2, TMM1);

                _tile_stored(TMM4, Tile4, TILE_N * sizeof(int32_t));
                _tile_stored(TMM6, Tile6, TILE_N * sizeof(int32_t));

                scale_C<TB, is_acc>(Tile4, Sumi4, B + PACKED_INDEX(0, i, KB, TILE_SIZE), k, m0);
                scale_C<TB, is_acc>(Tile6, Sumi6, B + PACKED_INDEX(1, i, KB, TILE_SIZE), k, m0);

                if (m1 != 0) {
                    _tile_zero(TMM5);
                    _tile_zero(TMM7);

                    unpack_A<TB>(Tile23, &A[TILE_M * KB + i], KB, k, m1);
                    _tile_loadd(TMM3, Tile23, TILE_K);

                    _tile_dpbssd(TMM5, TMM3, TMM0);
                    _tile_dpbssd(TMM7, TMM3, TMM1);

                    _tile_stored(TMM5, Tile5, TILE_N * sizeof(int32_t));
                    _tile_stored(TMM7, Tile7, TILE_N * sizeof(int32_t));

                    scale_C<TB, is_acc>(Tile5, Sumi5, B + PACKED_INDEX(0, i, KB, TILE_SIZE), k, m1);
                    scale_C<TB, is_acc>(Tile7, Sumi7, B + PACKED_INDEX(1, i, KB, TILE_SIZE), k, m1);
                }
            });
        }

        // step 2: accmulate the mins
        GGML_DISPATCH_BOOL(i > 0, is_acc, [&] {
            acc_C<TA, TB, is_acc>::apply(C,          ldc, Sumi4, &A[i], KB, B + PACKED_INDEX(0, i, KB, TILE_SIZE), m0);
            acc_C<TA, TB, is_acc>::apply(C + TILE_N, ldc, Sumi6, &A[i], KB, B + PACKED_INDEX(1, i, KB, TILE_SIZE), m0);
            if (m1 != 0) {
                acc_C<TA, TB, is_acc>::apply(C + TILE_M * ldc,          ldc, Sumi5, &A[TILE_M * KB + i], KB, B + PACKED_INDEX(0, i, KB, TILE_SIZE), m1);
                acc_C<TA, TB, is_acc>::apply(C + TILE_M * ldc + TILE_N, ldc, Sumi7, &A[TILE_M * KB + i], KB, B + PACKED_INDEX(1, i, KB, TILE_SIZE), m1);
            }
        });
    }
    return;
}

} // anonymous namespace

// get the packed tensor size for quantized weights
size_t ggml_backend_amx_get_alloc_size(const struct ggml_tensor * tensor) {
    const enum ggml_type TYPE = tensor->type;

    const int K = tensor->ne[0]; // ne0: in_features
    const int N = tensor->ne[1]; // ne1: out_features

    // > 1 only for MUL_MAT_ID expert stacks: [K, N, n_expert]
    const int64_t n_matrices = (int64_t) tensor->ne[2] * tensor->ne[3];

    auto get_tensor_size = [&] {
        size_t row_size_B{0};
        GGML_DISPATCH_QTYPES(TYPE, [&] {
            row_size_B = get_row_size<type, blck_size>(K);
        });
        return (size_t) n_matrices * N * row_size_B;
    };

    if (qtype_has_amx_kernels(TYPE)) {
        return get_tensor_size();
    } else {
        // for f16, bf16 we don't do packing
        return ggml_nbytes(tensor);
    }
}

// pack weight to vnni format
void ggml_backend_amx_convert_weight(struct ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    GGML_ASSERT(offset == 0 && size == ggml_nbytes(tensor)); // only full tensor conversion is supported for now

    const enum ggml_type TYPE = tensor->type;

    const int K = tensor->ne[0]; // ne0: in_features
    const int N = tensor->ne[1]; // ne1: out_features

    // > 1 only for MUL_MAT_ID expert stacks: [K, N, n_expert]; each expert matrix is
    // packed independently into its own contiguous {NB, KB, TILE_SIZE} run so that
    // expert e's tile (n, k) stays at e * (N/TILE_N) * KB * TILE_SIZE + PACKED_INDEX(n, k).
    const int64_t n_matrices = (int64_t) tensor->ne[2] * tensor->ne[3];

    GGML_DISPATCH_QTYPES(TYPE, [&] {
        const size_t src_stride = ggml_row_size(TYPE, K) * N;            // one unpacked expert
        const size_t dst_stride = (size_t) get_row_size<type, blck_size>(K) * N; // one packed expert
        for (int64_t e = 0; e < n_matrices; ++e) {
            convert_B_packed_format<type, blck_size>((char *)tensor->data + offset + e * dst_stride,
                                                     (const type *)((const char *)data + e * src_stride), N, K);
        }
    });
}

// ne2 is passed explicitly to help compiler optimize repeated calls
inline int64_t ggml_batch_offset(const ggml_tensor * t, int64_t batch_idx, int64_t ne2) {
    const int64_t i2 = batch_idx % ne2;
    const int64_t i3 = batch_idx / ne2;
    return i3 * t->nb[3] + i2 * t->nb[2];
}

size_t ggml_backend_amx_desired_wsize(const struct ggml_tensor * dst) {
    struct ggml_tensor * src0 = dst->src[0];

    const enum ggml_type TYPE = src0->type;

    const bool is_floating_type = TYPE == GGML_TYPE_F16;
    if (is_floating_type) {
        return 0;
    }

    const int M = dst->ne[1];
    const int K = src0->ne[0];
    const int64_t n_batch = dst->ne[2] * dst->ne[3];

    size_t desired_wsize = 0;

    GGML_DISPATCH_QTYPES(TYPE, [&] {
        const size_t row_size_A = K / blck_size * sizeof(vec_dot_type);
        desired_wsize = n_batch * M * row_size_A;
    });

    return desired_wsize;
}

// NB: mixed dtype gemm with Advanced Matrix Extensions (Intel AMX)
//
// src0: weight in shape of {N, K}, quantized
// src1: input  in shape of {M, K}, float32
// dst:  output in shape of {M, N}, float32
//
// the function performs: dst = src1 @ src0.T for each batch
//
void ggml_backend_amx_mul_mat(const ggml_compute_params * params, struct ggml_tensor * dst) {
    struct ggml_tensor * src0 = dst->src[0];
    struct ggml_tensor * src1 = dst->src[1];

    const enum ggml_type TYPE = src0->type;

    // f16 only has avx512 kernels for now,
    // amx kernels will be added once 6th gen xeon is released.
    const bool is_floating_type = TYPE == GGML_TYPE_F16;

    const int M = dst->ne[1];
    const int N = dst->ne[0];
    const int K = src0->ne[0];
    const int ldc = dst->nb[1] / dst->nb[0];

    const int64_t ne2 = dst->ne[2];
    const int64_t n_batch = ne2 * dst->ne[3];

    if (is_floating_type) {
        constexpr int BLOCK_M = 4;
        constexpr int BLOCK_N = 6;
        const int MB = div_up(M, BLOCK_M);
        const int NB = div_up(N, BLOCK_N);

        parallel_for_ggml(params, n_batch * MB * NB, [&](int begin, int end) {
            GGML_DISPATCH_FLOATING_TYPES(TYPE, [&] {
                for (int i = begin; i < end; ++i) {
                    int batch_idx = i / (MB * NB);
                    int remaining = i % (MB * NB);
                    int mb = remaining / NB;
                    int nb = remaining % NB;

                    int64_t src0_offset = ggml_batch_offset(src0, batch_idx, ne2);
                    int64_t src1_offset = ggml_batch_offset(src1, batch_idx, ne2);
                    int64_t dst_offset  = ggml_batch_offset(dst,  batch_idx, ne2);

                    int mb_start = mb * BLOCK_M;
                    int mb_size = std::min(BLOCK_M, M - mb_start);
                    int nb_start = nb * BLOCK_N;
                    int nb_size = std::min(BLOCK_N, N - nb_start);

                    switch (mb_size << 4 | nb_size) {
                        case 0x12: LAUNCH_TINYGEMM_KERNEL_AVX(1, 2); break;
                        case 0x14: LAUNCH_TINYGEMM_KERNEL_AVX(1, 4); break;
                        case 0x16: LAUNCH_TINYGEMM_KERNEL_AVX(1, 6); break;
                        case 0x22: LAUNCH_TINYGEMM_KERNEL_AVX(2, 2); break;
                        case 0x24: LAUNCH_TINYGEMM_KERNEL_AVX(2, 4); break;
                        case 0x26: LAUNCH_TINYGEMM_KERNEL_AVX(2, 6); break;
                        case 0x32: LAUNCH_TINYGEMM_KERNEL_AVX(3, 2); break;
                        case 0x34: LAUNCH_TINYGEMM_KERNEL_AVX(3, 4); break;
                        case 0x36: LAUNCH_TINYGEMM_KERNEL_AVX(3, 6); break;
                        case 0x42: LAUNCH_TINYGEMM_KERNEL_AVX(4, 2); break;
                        case 0x44: LAUNCH_TINYGEMM_KERNEL_AVX(4, 4); break;
                        case 0x46: LAUNCH_TINYGEMM_KERNEL_AVX(4, 6); break;
                        default: fprintf(stderr, "Unexpected block size!\n");
                    }
                }
            });
        });
        return;
    }

    // pointer to work space, used convert A from float to quantized type
    void * wdata = params->wdata;

    //TODO: performance improvement: merge quant A
 // if (params->ith == 0) {
        GGML_DISPATCH_QTYPES(TYPE, [&] {
            const size_t row_size_A = K / blck_size * sizeof(vec_dot_type);
            const size_t desired_wsize = n_batch * M * row_size_A;
            if (params->wsize < desired_wsize) {
                GGML_ABORT("insufficient work space size");
            }

            // Q4_0, Q4_1, Q8_0 handles 1 TILE_K per blck_size
            // Q4_K, Q5_K, Q6_K, IQ4_XS handles 8 TILE_K per blck_size
            GGML_ASSERT(TILE_K == blck_size || TILE_K * 8 == blck_size);

            parallel_for_ggml(params, n_batch * M, [&](int begin, int end) {
                for (int idx = begin; idx < end; ++idx) {
                    int batch_idx = idx / M;
                    int m         = idx % M;
                    int64_t src1_offset = ggml_batch_offset(src1, batch_idx, ne2);
                    const float * A_data = (const float *)((const char *)src1->data + src1_offset);
                    char * wdata_batch = (char *)wdata + batch_idx * M * row_size_A;
                    from_float<vec_dot_type>(A_data + m * K, wdata_batch + m * row_size_A, K);
                }
            });
        });
 // }

    ggml_barrier(params->threadpool);

    if (M == 1) {
        // MB = 1 and handle 8 tiles in each block
        constexpr int kTilesN = 4;
        constexpr int BLOCK_N = TILE_N * kTilesN;
        const int NB = div_up(N, BLOCK_N);

        parallel_for_ggml(params, n_batch * NB, [&](int begin, int end) {
            GGML_DISPATCH_QTYPES(TYPE, [&] {
                const int KB = K / blck_size;
                const int TILE_SIZE = get_tile_size<type>();
                const int row_size_A = KB * sizeof(vec_dot_type);
                for (int i = begin; i < end; ++i) {
                    int batch_idx = i / NB;
                    int nb = i % NB;

                    int64_t src0_offset = ggml_batch_offset(src0, batch_idx, ne2);
                    int64_t dst_offset  = ggml_batch_offset(dst,  batch_idx, ne2);
                    const char * wdata_batch = (const char *)wdata + batch_idx * row_size_A;

                    int nb_start = nb * BLOCK_N;
                    int nb_size = std::min(BLOCK_N, N - nb_start); // 32, 64, 96

                    switch (nb_size) {
                        //case 160: LAUNCH_TINYGEMM_KERNEL_VNNI(160); break;
                        case 128: LAUNCH_TINYGEMM_KERNEL_VNNI(128); break;
                        case 96: LAUNCH_TINYGEMM_KERNEL_VNNI(96); break;
                        case 64: LAUNCH_TINYGEMM_KERNEL_VNNI(64); break;
                        case 32: LAUNCH_TINYGEMM_KERNEL_VNNI(32); break;
                        default: fprintf(stderr, "Unexpected n block size!\n");
                    }
                }
            });
        });
        return;
    }

    // handle 4 tiles at a tile
    constexpr int BLOCK_M = TILE_M * 2;
    constexpr int BLOCK_N = TILE_N * 2;
    const int MB = div_up(M, BLOCK_M);
    const int NB = div_up(N, BLOCK_N);

    parallel_for_ggml(params, n_batch * MB * NB, [&](int begin, int end) {
        // init tile config for each thread
        ggml_tile_config_init();

        GGML_DISPATCH_QTYPES(TYPE, [&] {
            const int KB = K / blck_size;
            const int TILE_SIZE = get_tile_size<type>();
            const int row_size_A = KB * sizeof(vec_dot_type);

            for (int i = begin; i < end; ++i) {
                int batch_idx = i / (MB * NB);
                int remaining = i % (MB * NB);
                int mb = remaining / NB;
                int nb = remaining % NB;

                int64_t src0_offset = ggml_batch_offset(src0, batch_idx, ne2);
                int64_t dst_offset  = ggml_batch_offset(dst,  batch_idx, ne2);
                const char * wdata_batch = (const char *)wdata + batch_idx * M * row_size_A;

                int mb_start = mb * BLOCK_M;
                int mb_size = std::min(BLOCK_M, M - mb_start);
                int nb_start = nb * BLOCK_N;
                int nb_size = BLOCK_N;

                tinygemm_kernel_amx<vec_dot_type, type, float, blck_size>(
                    mb_size, nb_size, KB,
                    wdata_batch + mb_start * row_size_A,
                    (const char *)src0->data + src0_offset + PACKED_INDEX(nb * 2, 0, KB, TILE_SIZE),
                    (float *) dst->data + dst_offset + mb_start * N + nb_start, ldc);
            }
        });
    });
}

// NB: MoE expert-routed matmul (MUL_MAT_ID) with Advanced Matrix Extensions (Intel AMX)
//
// src0: weights in shape of {K, N, n_expert}, quantized, AMX packed (one {NB, KB, TILE_SIZE}
//       run per expert, see ggml_backend_amx_convert_weight)
// src1: input  in shape of {K, ne11, n_tokens}, float32
// ids:  expert ids in shape of {n_expert_used, n_tokens}, int32
// dst:  output in shape of {N, n_expert_used, n_tokens}, float32
//
// the function performs: dst[:, e, t] = src0[:, :, ids[e, t]].T @ src1[:, e % ne11, t]
//
// The row mapping (expert -> compact list of (slot, token) pairs) mirrors the repack
// forward_mul_mat_id construction, but the AMX A operand is a plain per-row
// vec_dot_type array, so the gather degenerates to a per-row memcpy into a
// contiguous per-thread scratch (no 4-row interleave) and partial M blocks are
// handled inside the tinygemm kernel (unpack_A path), same as MUL_MAT.
//
// TODO(EP): the NUMA EP row-window claim loop (GGML_NUMA_EP) is only wired into the
// repack forward_mul_mat_id; when the AMX buft is integrated with the EPD worker,
// this is where the per-node window / row_claim claiming should hook in.

#define AMX_MMID_BLOCK_M (2 * TILE_M)

struct amx_mmid_row_mapping {
    int32_t i1; // selected expert slot (dst dim 1)
    int32_t i2; // token (src1/dst dim 2)
};

// workspace layout (must match ggml_backend_amx_mul_mat_id):
//   [0]                        quantized src1 rows, n_src1_rows * row_size_A
//   [qact_size]                expert row counts/offsets/cursors, (3*n_as + 1) int64
//   [qact_size + counts_size]  compact (slot, token) mapping, n_selected entries
//   [...]                      per-thread scratch: AMX_MMID_BLOCK_M gathered activation
//                              rows + AMX_MMID_BLOCK_M x 2*TILE_N f32 tile
static void ggml_backend_amx_mmid_wsize_layout(const struct ggml_tensor * dst, int n_threads,
                                               size_t & row_size_A_out, size_t & qact_size,
                                               size_t & counts_size, size_t & map_size,
                                               size_t & scratch_stride, size_t & total) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    const ggml_tensor * ids  = dst->src[2];

    const enum ggml_type TYPE = src0->type;

    const int K    = src0->ne[0];
    const int n_as = src0->ne[2];

    const int64_t n_src1_rows = (int64_t) src1->ne[1] * src1->ne[2];
    const int64_t n_selected  = (int64_t) ids->ne[0] * ids->ne[1];

    total = 0;
    GGML_DISPATCH_QTYPES(TYPE, [&] {
        const size_t row_size_A = (size_t) K / blck_size * sizeof(vec_dot_type);
        row_size_A_out = row_size_A;
        qact_size      = GGML_PAD(n_src1_rows * row_size_A, 64);
        counts_size    = GGML_PAD((3 * (size_t) n_as + 1) * sizeof(int64_t), 64);
        map_size       = GGML_PAD((size_t) n_selected * sizeof(amx_mmid_row_mapping), 64);
        scratch_stride = GGML_PAD(AMX_MMID_BLOCK_M * row_size_A +
                                  AMX_MMID_BLOCK_M * 2 * TILE_N * sizeof(float), 64);
        total = qact_size + counts_size + map_size + (size_t) n_threads * scratch_stride;
    });
}

size_t ggml_backend_amx_mmid_desired_wsize(int n_threads, const struct ggml_tensor * dst) {
    size_t row_size_A, qact_size, counts_size, map_size, scratch_stride, total;
    ggml_backend_amx_mmid_wsize_layout(dst, n_threads, row_size_A, qact_size, counts_size,
                                       map_size, scratch_stride, total);
    return total;
}

void ggml_backend_amx_mul_mat_id(const ggml_compute_params * params, struct ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    const ggml_tensor * ids  = dst->src[2];

    const enum ggml_type TYPE = src0->type;

    const int K     = src0->ne[0];
    const int N     = src0->ne[1];
    const int n_as  = src0->ne[2]; // n_expert
    const int n_ids = ids->ne[0];  // n_expert_used

    const int64_t n_tokens    = ids->ne[1];
    const int64_t ne11        = src1->ne[1]; // src1 slots (broadcast: n_ids % ne11 == 0)
    const int64_t n_src1_rows = ne11 * src1->ne[2];
    const int64_t n_selected  = (int64_t) n_ids * n_tokens;

    // we don't support permuted inputs; dst row stride must be dense floats
    GGML_ASSERT(ggml_is_contiguous(src0) && ggml_is_contiguous(src1));
    GGML_ASSERT(src1->type == GGML_TYPE_F32 && ids->type == GGML_TYPE_I32);
    GGML_ASSERT(dst->nb[0] == sizeof(float));
    GGML_ASSERT(dst->ne[0] == N && dst->ne[1] == n_ids && dst->ne[2] == n_tokens);
    GGML_ASSERT(N % (2 * TILE_N) == 0); // gated by supports_op

    size_t row_size_A, qact_size, counts_size, map_size, scratch_stride, wsize_needed;
    ggml_backend_amx_mmid_wsize_layout(dst, params->nth, row_size_A, qact_size, counts_size,
                                       map_size, scratch_stride, wsize_needed);
    if (params->wsize < wsize_needed) {
        GGML_ABORT("insufficient work space size");
    }

    char * wdata = (char *) params->wdata;

    char * qact = wdata; // quantized src1 rows, row r = i11 + i12 * ne11
    int64_t * matrix_row_counts  = (int64_t *) (wdata + qact_size);      // [n_as]
    int64_t * matrix_row_offsets = matrix_row_counts + n_as;             // [n_as + 1]
    int64_t * matrix_row_cursors = matrix_row_offsets + n_as + 1;        // [n_as]
    amx_mmid_row_mapping * matrix_rows =
        (amx_mmid_row_mapping *) (wdata + qact_size + counts_size);      // [n_selected]
    char * scratch_base = wdata + qact_size + counts_size + map_size;    // per-thread

    GGML_DISPATCH_QTYPES(TYPE, [&] {
        const int KB = K / blck_size;
        const int TILE_SIZE = get_tile_size<type>();
        const size_t expert_stride = (size_t) (N / TILE_N) * KB * TILE_SIZE; // one packed expert

        GGML_ASSERT(TILE_K == blck_size || TILE_K * 8 == blck_size);

        // quantize all src1 rows once (row-major: slot-fastest, then token)
        parallel_for_ggml(params, (int) n_src1_rows, [&](int begin, int end) {
            for (int r = begin; r < end; ++r) {
                const int64_t i12 = r / ne11;
                const int64_t i11 = r % ne11;
                from_float<vec_dot_type>(
                    (const float *) ((const char *) src1->data + i12 * src1->nb[2] + i11 * src1->nb[1]),
                    qact + (size_t) r * row_size_A, K);
            }
        });

        // build the compact expert -> (slot, token) row mapping (same construction as
        // repack.cpp forward_mul_mat_id: count, prefix-sum, fill)
        if (params->ith == 0) {
            memset(matrix_row_counts, 0, n_as * sizeof(int64_t));

            for (int32_t iid1 = 0; iid1 < ids->ne[1]; ++iid1) {
                for (int32_t id = 0; id < n_ids; ++id) {
                    const int32_t i02 =
                        *(const int32_t *) ((const char *) ids->data + iid1 * ids->nb[1] + id * ids->nb[0]);
                    GGML_ASSERT(i02 >= 0 && i02 < n_as);
                    matrix_row_counts[i02] += 1;
                }
            }

            matrix_row_offsets[0] = 0;
            for (int cur_a = 0; cur_a < n_as; ++cur_a) {
                matrix_row_offsets[cur_a + 1] = matrix_row_offsets[cur_a] + matrix_row_counts[cur_a];
                matrix_row_cursors[cur_a] = 0;
            }
            GGML_ASSERT(matrix_row_offsets[n_as] == n_selected);

            for (int32_t iid1 = 0; iid1 < ids->ne[1]; ++iid1) {
                for (int32_t id = 0; id < n_ids; ++id) {
                    const int32_t i02 =
                        *(const int32_t *) ((const char *) ids->data + iid1 * ids->nb[1] + id * ids->nb[0]);
                    matrix_rows[matrix_row_offsets[i02] + matrix_row_cursors[i02]++] = { id, iid1 };
                }
            }
        }

        ggml_barrier(params->threadpool);

        // compute: work items are (expert, 32-column block) pairs; each thread gathers
        // the expert's selected activation rows in chunks of AMX_MMID_BLOCK_M, runs the
        // tinygemm kernel on the chunk and scatters the rows back to (slot, token).
        // Every dst element has exactly one writer, so the result is run-to-run
        // bit-exact for a fixed thread count.
        const int NB32 = N / (2 * TILE_N);

        parallel_for_ggml(params, n_as * NB32, [&](int begin, int end) {
            ggml_tile_config_init();

            char * gather = scratch_base + (size_t) params->ith * scratch_stride;
            float * tile  = (float *) (gather + AMX_MMID_BLOCK_M * row_size_A);

            for (int i = begin; i < end; ++i) {
                const int cur_a = i / NB32;
                const int nb    = i % NB32;

                const int64_t cne1 = matrix_row_counts[cur_a];
                if (cne1 == 0) {
                    continue;
                }

                const amx_mmid_row_mapping * rows = matrix_rows + matrix_row_offsets[cur_a];
                const char * B = (const char *) src0->data + (size_t) cur_a * expert_stride +
                                 PACKED_INDEX(nb * 2, 0, KB, TILE_SIZE);

                for (int64_t base = 0; base < cne1; base += AMX_MMID_BLOCK_M) {
                    const int m = (int) std::min<int64_t>(AMX_MMID_BLOCK_M, cne1 - base);

                    // gather: plain per-row copy of the quantized activations
                    for (int r = 0; r < m; ++r) {
                        const int64_t src1_row = (rows[base + r].i1 % ne11) + (int64_t) rows[base + r].i2 * ne11;
                        memcpy(gather + (size_t) r * row_size_A, qact + (size_t) src1_row * row_size_A, row_size_A);
                    }

                    tinygemm_kernel_amx<vec_dot_type, type, float, blck_size>(
                        m, 2 * TILE_N, KB, gather, B, tile, 2 * TILE_N);

                    // scatter rows back to their original (slot, token) positions
                    for (int r = 0; r < m; ++r) {
                        memcpy((char *) dst->data + rows[base + r].i1 * dst->nb[1] +
                               rows[base + r].i2 * dst->nb[2] + (size_t) nb * 2 * TILE_N * sizeof(float),
                               tile + (size_t) r * 2 * TILE_N, 2 * TILE_N * sizeof(float));
                    }
                }
            }
        });
    });
}

#endif // if defined(__AMX_INT8__) && defined(__AVX512VNNI__)
