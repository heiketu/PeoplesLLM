#include "llama-hot-expert.h"

#include "llama-impl.h"

#include "ggml.h"
#include "ggml-backend.h"
#include "gguf.h"

#include <algorithm>
#include <atomic>
#include <cinttypes>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#ifndef _WIN32
#include <stdlib.h> // on_exit
#include <unistd.h> // _exit
#endif
#include <map>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

#if (defined(__x86_64__) || defined(_M_X64)) && (defined(__GNUC__) || defined(__clang__)) && !defined(_MSC_VER)
#include <immintrin.h>
#define LLAMA_HOT_EXPERT_HAVE_TARGET_AVX512 1
#else
#define LLAMA_HOT_EXPERT_HAVE_TARGET_AVX512 0
#endif

namespace {

struct hot_layer {
    int il = -1;

    int64_t n_expert = 0;
    int64_t n_embd   = 0;
    int64_t n_ff     = 0;
    int     K        = 0;

    // remap[global expert] -> compact slot, or -1 if the expert is cold
    std::vector<int32_t> remap;

    float clamp_limit = 0.0f;

    // device-side objects
    ggml_tensor * x_gpu       = nullptr; // [n_embd, 1, 1]  f32
    ggml_tensor * ids_gpu     = nullptr; // [K_used, 1]     i32 (remapped, cold slots point at slot 0)
    ggml_tensor * w_gpu       = nullptr; // [1, K_used, 1]  f32 (cold slots zeroed)
    ggml_tensor * input_gpu   = nullptr; // byte view over the packed x/ids/weights region
    ggml_tensor * output_gpu  = nullptr; // partial [n_embd, 1] or slots [n_embd, K_used, 1]

    ggml_cgraph * graph = nullptr;
    ggml_context * gctx = nullptr;

    ggml_backend_buffer_t wbuf  = nullptr; // compact hot weights
    ggml_backend_buffer_t iobuf = nullptr; // device io tensors
    ggml_backend_event_t  event = nullptr;

    // pinned host staging (single buffer): x | ids | weights | GPU output
    ggml_backend_buffer_t hbuf     = nullptr;
    void *                h_base   = nullptr;
    float *               h_x      = nullptr;
    int32_t *             h_ids    = nullptr;
    float *               h_w      = nullptr;
    float *               h_output = nullptr;
    std::vector<uint8_t>   h_hot_mask;
    std::unique_ptr<std::atomic<bool>> remote_inflight;

    size_t input_ids_offset = 0;
    size_t input_w_offset   = 0;
    size_t input_bytes      = 0;
    size_t output_bytes     = 0;
};

struct hot_state {
    bool enabled = false;
    bool markers = false;
    bool remote_ep = false;

    std::string table_path;
    std::string gguf_path;
    std::string dev_name = "CUDA1";
    int K = 16;
    int64_t max_tokens = 1;
    std::string layers_spec = "all";

    ggml_backend_dev_t dev = nullptr;
    ggml_backend_t     backend = nullptr;

    int64_t n_expert_used = 0; // slots per token (top-k of the router)

    std::map<int, hot_layer> layers;
};

hot_state g_hot;

bool env_flag(const char * name, bool dflt) {
    const char * e = getenv(name);
    return e ? atoi(e) != 0 : dflt;
}

std::string env_str(const char * name, const std::string & dflt) {
    const char * e = getenv(name);
    return e ? e : dflt;
}

int env_int(const char * name, int dflt) {
    const char * e = getenv(name);
    return e ? atoi(e) : dflt;
}

size_t align_256(size_t value) {
    return (value + 255) & ~size_t(255);
}

bool layer_selected(const std::string & spec, int il) {
    if (spec == "all") {
        return true;
    }
    size_t pos = 0;
    while (pos <= spec.size()) {
        size_t comma = spec.find(',', pos);
        std::string tok = spec.substr(pos, comma == std::string::npos ? std::string::npos : comma - pos);
        size_t dash = tok.find('-');
        if (dash != std::string::npos) {
            int lo = atoi(tok.substr(0, dash).c_str());
            int hi = atoi(tok.substr(dash + 1).c_str());
            if (il >= lo && il <= hi) {
                return true;
            }
        } else if (!tok.empty() && atoi(tok.c_str()) == il) {
            return true;
        }
        if (comma == std::string::npos) {
            break;
        }
        pos = comma + 1;
    }
    return false;
}

// read "layer\texpert\thits" TSV into per-layer hit vectors
bool load_table(const std::string & path, std::map<int, std::map<int, int64_t>> & out) {
    FILE * f = fopen(path.c_str(), "r");
    if (!f) {
        return false;
    }
    int layer, expert;
    int64_t hits;
    while (fscanf(f, "%d%d%" SCNd64, &layer, &expert, &hits) == 3) {
        out[layer][expert] += hits;
    }
    fclose(f);
    return !out.empty();
}

struct gguf_tensor_info {
    int64_t offset; // absolute file offset of the tensor data
    int64_t nbytes;
    int64_t ne[3];
    ggml_type type;
};

bool find_expert_tensor(struct gguf_context * gctx, size_t data_offset, const char * name, gguf_tensor_info & out) {
    const int64_t id = gguf_find_tensor(gctx, name);
    if (id < 0) {
        return false;
    }
    out.offset = (int64_t) (data_offset + gguf_get_tensor_offset(gctx, id));
    const int64_t * ne = gguf_get_tensor_ne(gctx, id);
    // expert weight tensors are 3D {rows, cols, n_expert}
    for (int i = 0; i < 3; i++) {
        out.ne[i] = ne[i];
    }
    out.type = (ggml_type) gguf_get_tensor_type(gctx, id);
    out.nbytes = (int64_t) ggml_row_size(out.type, out.ne[0] * out.ne[1]) * out.ne[2];
    return true;
}

// upload the top-K expert slices of one projection from the GGUF file into the
// compact device tensor (expert slices of a contiguous 3D tensor are self-contained)
bool upload_hot_slices(FILE * f, const gguf_tensor_info & info, const hot_layer & L,
                       ggml_tensor * dst, const std::vector<int> & slot_expert) {
    const int64_t slice_bytes = info.nbytes / info.ne[2];
    std::vector<uint8_t> staging(slice_bytes);
    for (int s = 0; s < L.K; s++) {
        const int64_t expert = slot_expert[s];
        const int64_t off = info.offset + expert * slice_bytes;
        if (fseeko(f, off, SEEK_SET) != 0 || fread(staging.data(), 1, slice_bytes, f) != (size_t) slice_bytes) {
            return false;
        }
        ggml_backend_tensor_set(dst, staging.data(), s * slice_bytes, slice_bytes);
    }
    return true;
}

bool build_layer(hot_layer & L, FILE * gguf_file, struct gguf_context * gctx, size_t data_offset,
                 const std::map<int, int64_t> & hits, float clamp_limit) {
    char name[64];

    gguf_tensor_info tg, tu, td;
    snprintf(name, sizeof(name), "blk.%d.ffn_gate_exps.weight", L.il);
    if (!find_expert_tensor(gctx, data_offset, name, tg)) return false;
    snprintf(name, sizeof(name), "blk.%d.ffn_up_exps.weight", L.il);
    if (!find_expert_tensor(gctx, data_offset, name, tu)) return false;
    snprintf(name, sizeof(name), "blk.%d.ffn_down_exps.weight", L.il);
    if (!find_expert_tensor(gctx, data_offset, name, td)) return false;

    if (tg.type != GGML_TYPE_MXFP4 || tu.type != GGML_TYPE_MXFP4 || td.type != GGML_TYPE_MXFP4) {
        LLAMA_LOG_WARN("%s: layer %d expert weights are not MXFP4, skipping\n", __func__, L.il);
        return false;
    }

    L.n_embd   = tg.ne[0];
    L.n_ff     = tg.ne[1];
    L.n_expert = tg.ne[2];
    L.clamp_limit = clamp_limit;

    if (td.ne[0] != L.n_ff || td.ne[1] != L.n_embd || td.ne[2] != L.n_expert ||
        tu.ne[0] != L.n_embd || tu.ne[1] != L.n_ff  || tu.ne[2] != L.n_expert) {
        LLAMA_LOG_WARN("%s: layer %d unexpected expert tensor shapes, skipping\n", __func__, L.il);
        return false;
    }

    // top-K by hits; ranked holds (hits, expert) — do NOT construct the vector
    // directly from the hits map iterators: its value_type is (expert, hits)
    // and the silent pair conversion would swap the fields ( Slice 12 bug:
    // expert index 375 OOB on remap, and the "hot" set was just the highest
    // expert ids instead of the hottest ones )
    std::vector<std::pair<int64_t, int>> ranked;
    ranked.reserve(hits.size());
    for (const auto & [expert, h] : hits) {
        if (expert >= 0 && expert < L.n_expert) {
            ranked.emplace_back(h, expert);
        }
    }
    std::sort(ranked.begin(), ranked.end(), [](const auto & a, const auto & b) {
        return a.first > b.first || (a.first == b.first && a.second < b.second);
    });
    const int K = std::min<int>(g_hot.K, ranked.size());

    std::vector<int> slot_expert(K);
    L.remap.assign(L.n_expert, -1);
    for (int s = 0; s < K; s++) {
        slot_expert[s] = ranked[s].second;
        L.remap[ranked[s].second] = s;
    }
    L.K = K;

    ggml_backend_buffer_type_t buft = ggml_backend_dev_buffer_type(g_hot.dev);

    // --- weights buffer (compact [n_embd, n_ff, K] / [n_ff, n_embd, K] per projection)
    const int64_t slice_g = tg.nbytes / tg.ne[2];
    const int64_t slice_d = td.nbytes / td.ne[2];
    if (getenv("GGML_HOT_EXPERT_DEBUG")) {
        fprintf(stderr, "[hotdbg] layer %d: n_embd=%lld n_ff=%lld n_expert=%lld K=%d slice_g=%lld slice_d=%lld wbuf=%lld\n",
            L.il, (long long)L.n_embd, (long long)L.n_ff, (long long)L.n_expert, K,
            (long long)slice_g, (long long)slice_d, (long long)((2 * slice_g + slice_d) * K + 3 * 256));
    }
    L.wbuf = ggml_backend_buft_alloc_buffer(buft, (2 * slice_g + slice_d) * K + 3 * 256);
    if (getenv("GGML_HOT_EXPERT_DEBUG")) {
        fprintf(stderr, "[hotdbg] layer %d: wbuf -> %p\n", L.il, (void *) L.wbuf);
    }
    if (!L.wbuf) {
        LLAMA_LOG_WARN("%s: layer %d failed to allocate %.1f MiB on %s\n", __func__, L.il,
            ((2 * slice_g + slice_d) * K) / 1048576.0, g_hot.dev_name.c_str());
        return false;
    }
    uint8_t * wbase = (uint8_t *) ggml_backend_buffer_get_base(L.wbuf);

    // --- graph context (no_alloc; data assigned manually into the buffers)
    struct ggml_init_params ip = { /*.mem_size =*/ 1 << 20, /*.mem_buffer =*/ nullptr, /*.no_alloc =*/ true };
    L.gctx = ggml_init(ip);

    ggml_tensor * Wg = ggml_new_tensor_3d(L.gctx, GGML_TYPE_MXFP4, L.n_embd, L.n_ff, K);
    ggml_tensor * Wu = ggml_new_tensor_3d(L.gctx, GGML_TYPE_MXFP4, L.n_embd, L.n_ff, K);
    ggml_tensor * Wd = ggml_new_tensor_3d(L.gctx, GGML_TYPE_MXFP4, L.n_ff, L.n_embd, K);

    const int64_t k_used = g_hot.n_expert_used;
    L.x_gpu   = ggml_new_tensor_3d(L.gctx, GGML_TYPE_F32, L.n_embd, 1, 1);
    L.ids_gpu = ggml_new_tensor_2d(L.gctx, GGML_TYPE_I32, k_used, 1);
    L.w_gpu   = ggml_new_tensor_3d(L.gctx, GGML_TYPE_F32, 1, k_used, 1);

    L.input_ids_offset = align_256(ggml_nbytes(L.x_gpu));
    L.input_w_offset   = align_256(L.input_ids_offset + ggml_nbytes(L.ids_gpu));
    L.input_bytes      = align_256(L.input_w_offset + ggml_nbytes(L.w_gpu));
    L.input_gpu        = ggml_new_tensor_1d(L.gctx, GGML_TYPE_I8, L.input_bytes);

    ggml_tensor * gate = ggml_mul_mat_id(L.gctx, Wg, L.x_gpu, L.ids_gpu); // [n_ff, k_used, 1]
    ggml_tensor * up   = ggml_mul_mat_id(L.gctx, Wu, L.x_gpu, L.ids_gpu); // [n_ff, k_used, 1]

    // replicate the DSV4 clamped swiglu of the classic path
    ggml_tensor * act = nullptr;
    if (L.clamp_limit > 1e-6f) {
        up   = ggml_clamp(L.gctx, up, -L.clamp_limit, L.clamp_limit);
        gate = ggml_clamp(L.gctx, gate, -INFINITY, L.clamp_limit);
        act  = ggml_swiglu_split(L.gctx, gate, up);
    } else {
        act  = ggml_swiglu_split(L.gctx, gate, up);
    }

    ggml_tensor * exps = ggml_mul_mat_id(L.gctx, Wd, act, L.ids_gpu);     // [n_embd, k_used, 1]

    if (llama_hot_expert_slot_order_enabled()) {
        L.output_gpu = exps;
    } else {
        // legacy path: reduce hot slots independently, then add the CPU partial
        L.output_gpu = ggml_moe_wreduce(L.gctx, exps, L.w_gpu, L.ids_gpu, 0, K); // [n_embd, 1]
    }
    GGML_ASSERT(ggml_is_contiguous(L.output_gpu));
    L.output_bytes = ggml_nbytes(L.output_gpu);

    // --- io buffer: every intermediate of the graph, bump-allocated
    size_t io_size = L.input_bytes;
    for (ggml_tensor * t = ggml_get_first_tensor(L.gctx); t; t = ggml_get_next_tensor(L.gctx, t)) {
        if (t == Wg || t == Wu || t == Wd || t == L.input_gpu ||
            t == L.x_gpu || t == L.ids_gpu || t == L.w_gpu) {
            continue;
        }
        io_size += ggml_nbytes(t) + 256;
    }
    L.iobuf = ggml_backend_buft_alloc_buffer(buft, io_size + 256);
    if (getenv("GGML_HOT_EXPERT_DEBUG")) {
        fprintf(stderr, "[hotdbg] layer %d: iobuf(%zu) -> %p\n", L.il, io_size + 256, (void *) L.iobuf);
    }
    if (!L.iobuf) {
        LLAMA_LOG_WARN("%s: layer %d failed to allocate io buffer\n", __func__, L.il);
        return false;
    }
    uint8_t * iobase = (uint8_t *) ggml_backend_buffer_get_base(L.iobuf);

    size_t off = L.input_bytes;
    for (ggml_tensor * t = ggml_get_first_tensor(L.gctx); t; t = ggml_get_next_tensor(L.gctx, t)) {
        if (t == Wg || t == Wu || t == Wd || t == L.input_gpu ||
            t == L.x_gpu || t == L.ids_gpu || t == L.w_gpu) {
            continue;
        }
        t->buffer = L.iobuf;
        t->data   = iobase + off;
        off += ggml_nbytes(t) + 256;
        off &= ~size_t(255);
    }
    // weights live in the dedicated weights buffer
    Wg->buffer = L.wbuf; Wg->data = wbase;
    Wu->buffer = L.wbuf; Wu->data = wbase + K * slice_g;
    Wd->buffer = L.wbuf; Wd->data = wbase + K * slice_g * 2;
    L.input_gpu->buffer = L.iobuf;
    L.input_gpu->data   = iobase;
    L.x_gpu->buffer     = L.iobuf;
    L.x_gpu->data       = iobase;
    L.ids_gpu->buffer   = L.iobuf;
    L.ids_gpu->data     = iobase + L.input_ids_offset;
    L.w_gpu->buffer     = L.iobuf;
    L.w_gpu->data       = iobase + L.input_w_offset;

    GGML_ASSERT((uint8_t *) L.x_gpu->data == iobase);
    GGML_ASSERT((uint8_t *) L.ids_gpu->data == iobase + L.input_ids_offset);
    GGML_ASSERT((uint8_t *) L.w_gpu->data == iobase + L.input_w_offset);

    L.graph = ggml_new_graph_custom(L.gctx, 64, /*grads =*/ false);
    ggml_build_forward_expand(L.graph, L.output_gpu);

    // --- pinned host staging
    ggml_backend_buffer_type_t host_buft = ggml_backend_dev_host_buffer_type(g_hot.dev);
    const size_t h_need = L.input_bytes + L.output_bytes;
    L.hbuf = ggml_backend_buft_alloc_buffer(host_buft, h_need + 4 * 256);
    if (getenv("GGML_HOT_EXPERT_DEBUG")) {
        fprintf(stderr, "[hotdbg] layer %d: hbuf(%zu) -> %p\n", L.il, h_need + 4 * 256, (void *) L.hbuf);
    }
    if (!L.hbuf) {
        LLAMA_LOG_WARN("%s: layer %d failed to allocate pinned staging\n", __func__, L.il);
        return false;
    }
    L.h_base = ggml_backend_buffer_get_base(L.hbuf);
    L.h_x       = (float *)   L.h_base;
    L.h_ids     = (int32_t *) ((uint8_t *) L.h_base + L.input_ids_offset);
    L.h_w       = (float *)   ((uint8_t *) L.h_base + L.input_w_offset);
    L.h_output  = (float *)   ((uint8_t *) L.h_base + L.input_bytes);
    L.h_hot_mask.resize(k_used);
    L.remote_inflight.reset(new std::atomic<bool>(false));

    L.event = ggml_backend_event_new(g_hot.dev);
    if (getenv("GGML_HOT_EXPERT_DEBUG")) {
        fprintf(stderr, "[hotdbg] layer %d: event -> %p\n", L.il, (void *) L.event);
    }
    if (!L.event) {
        LLAMA_LOG_WARN("%s: layer %d failed to create event\n", __func__, L.il);
        return false;
    }

    // --- upload hot slices
    if (getenv("GGML_HOT_EXPERT_DEBUG")) {
        fprintf(stderr, "[hotdbg] layer %d: uploading slices\n", L.il);
    }
    if (!upload_hot_slices(gguf_file, tg, L, Wg, slot_expert) ||
        !upload_hot_slices(gguf_file, tu, L, Wu, slot_expert) ||
        !upload_hot_slices(gguf_file, td, L, Wd, slot_expert)) {
        LLAMA_LOG_WARN("%s: layer %d weight upload failed\n", __func__, L.il);
        return false;
    }
    if (getenv("GGML_HOT_EXPERT_DEBUG")) {
        fprintf(stderr, "[hotdbg] layer %d: done\n", L.il);
    }

    return true;
}

void merge_slots_f32_validate(
        float * dst, const float * cold, const float * hot, const float * weights,
        const uint8_t * hot_mask, int64_t n_embd, int64_t n_slots) {
    GGML_ASSERT(dst != nullptr && cold != nullptr && hot != nullptr && weights != nullptr && hot_mask != nullptr);
    GGML_ASSERT(n_embd > 0 && n_slots > 0);
}

void merge_slots_f32_scalar_rows(
        float * dst, const float * cold, size_t cold_slot_stride,
        const float * hot, size_t hot_slot_stride, const float * weights,
        const uint8_t * hot_mask, int64_t row_begin, int64_t row_end, int64_t n_slots) {
    for (int64_t row = row_begin; row < row_end; ++row) {
        float acc = 0.0f;
        for (int64_t slot = 0; slot < n_slots; ++slot) {
            float term;
            if (hot_mask[slot]) {
                const volatile float product = hot[slot*hot_slot_stride + row]*weights[slot];
                term = product;
            } else {
                term = cold[slot*cold_slot_stride + row];
            }
            if (slot == 0) {
                acc = term;
            } else {
                const volatile float sum = acc + term;
                acc = sum;
            }
        }
        dst[row] = acc;
    }
}

#if LLAMA_HOT_EXPERT_HAVE_TARGET_AVX512
__attribute__((target("avx512f")))
void merge_slots_f32_avx512_impl(
        float * dst, const float * cold, size_t cold_slot_stride,
        const float * hot, size_t hot_slot_stride, const float * weights,
        const uint8_t * hot_mask, int64_t n_embd, int64_t n_slots) {
    const int64_t row_vec_end = n_embd & ~int64_t(15);
    for (int64_t row = 0; row < row_vec_end; row += 16) {
        __m512 acc;
        if (hot_mask[0]) {
            acc = _mm512_mul_ps(
                _mm512_loadu_ps(hot + row),
                _mm512_set1_ps(weights[0]));
            __asm__ volatile("" : "+v"(acc));
        } else {
            acc = _mm512_loadu_ps(cold + row);
        }

        for (int64_t slot = 1; slot < n_slots; ++slot) {
            __m512 term;
            if (hot_mask[slot]) {
                term = _mm512_mul_ps(
                    _mm512_loadu_ps(hot + slot*hot_slot_stride + row),
                    _mm512_set1_ps(weights[slot]));
                // Keep product rounding separate from the left-fold add.
                __asm__ volatile("" : "+v"(term));
            } else {
                term = _mm512_loadu_ps(cold + slot*cold_slot_stride + row);
            }
            acc = _mm512_add_ps(acc, term);
            // Keep the slot-to-slot dependency visible to the optimizer.
            __asm__ volatile("" : "+v"(acc));
        }
        _mm512_storeu_ps(dst + row, acc);
    }
    merge_slots_f32_scalar_rows(
        dst, cold, cold_slot_stride, hot, hot_slot_stride, weights,
        hot_mask, row_vec_end, n_embd, n_slots);
}
#endif

} // namespace

void llama_hot_expert_merge_slots_f32_scalar(
        float * dst, const float * cold, size_t cold_slot_stride,
        const float * hot, size_t hot_slot_stride, const float * weights,
        const uint8_t * hot_mask, int64_t n_embd, int64_t n_slots) {
    merge_slots_f32_validate(dst, cold, hot, weights, hot_mask, n_embd, n_slots);
    merge_slots_f32_scalar_rows(
        dst, cold, cold_slot_stride, hot, hot_slot_stride, weights,
        hot_mask, 0, n_embd, n_slots);
}

bool llama_hot_expert_slot_merge_avx512_supported() {
#if LLAMA_HOT_EXPERT_HAVE_TARGET_AVX512
    static const bool supported = []() {
        __builtin_cpu_init();
        return __builtin_cpu_supports("avx512f");
    }();
    return supported;
#else
    return false;
#endif
}

bool llama_hot_expert_slot_merge_avx512_enabled() {
    static const bool requested = env_flag("GGML_HOT_EXPERT_SLOT_MERGE_AVX512", true);
    return requested && llama_hot_expert_slot_merge_avx512_supported();
}

bool llama_hot_expert_merge_slots_f32_avx512(
        float * dst, const float * cold, size_t cold_slot_stride,
        const float * hot, size_t hot_slot_stride, const float * weights,
        const uint8_t * hot_mask, int64_t n_embd, int64_t n_slots) {
    merge_slots_f32_validate(dst, cold, hot, weights, hot_mask, n_embd, n_slots);
#if LLAMA_HOT_EXPERT_HAVE_TARGET_AVX512
    if (llama_hot_expert_slot_merge_avx512_supported()) {
        merge_slots_f32_avx512_impl(
            dst, cold, cold_slot_stride, hot, hot_slot_stride,
            weights, hot_mask, n_embd, n_slots);
        return true;
    }
#endif
    return false;
}

void llama_hot_expert_merge_slots_f32(
        float * dst, const float * cold, size_t cold_slot_stride,
        const float * hot, size_t hot_slot_stride, const float * weights,
        const uint8_t * hot_mask, int64_t n_embd, int64_t n_slots) {
    if (llama_hot_expert_slot_merge_avx512_enabled()) {
        const bool ok = llama_hot_expert_merge_slots_f32_avx512(
            dst, cold, cold_slot_stride, hot, hot_slot_stride,
            weights, hot_mask, n_embd, n_slots);
        GGML_ASSERT(ok);
        return;
    }
    llama_hot_expert_merge_slots_f32_scalar(
        dst, cold, cold_slot_stride, hot, hot_slot_stride,
        weights, hot_mask, n_embd, n_slots);
}

bool llama_hot_expert_enabled() {
    static const bool enabled = env_flag("GGML_HOT_EXPERT", false);
    return enabled;
}

bool llama_hot_expert_slot_order_enabled() {
    static const bool enabled = env_flag("GGML_HOT_EXPERT_SLOT_ORDER", true);
    return enabled;
}

bool llama_hot_expert_remote_ep_enabled() {
    return g_hot.enabled && g_hot.remote_ep && llama_hot_expert_slot_order_enabled();
}

bool llama_hot_expert_markers_enabled() {
    return g_hot.markers;
}

int64_t llama_hot_expert_max_tokens() {
    return g_hot.enabled ? g_hot.max_tokens : 0;
}

bool llama_hot_expert_layer_active(int il) {
    return g_hot.enabled && g_hot.layers.count(il) > 0;
}

void * llama_hot_expert_userdata(int il) {
    auto it = g_hot.layers.find(il);
    return it == g_hot.layers.end() ? nullptr : (void *) &it->second;
}

bool llama_hot_expert_init(const float * swiglu_clamp_exp, int n_layer, int64_t n_expert_used) {
    if (!llama_hot_expert_enabled()) {
        return false;
    }
    if (g_hot.enabled) {
        // already initialized for the main model; secondary models (e.g. a
        // speculative draft) reuse the main model's hot tables
        return true;
    }

    g_hot.table_path  = env_str("GGML_HOT_EXPERT_TABLE", "");
    g_hot.gguf_path   = env_str("GGML_HOT_EXPERT_GGUF", "");
    g_hot.dev_name    = env_str("GGML_HOT_EXPERT_DEV", "CUDA1");
    g_hot.K           = env_int("GGML_HOT_EXPERT_K", 16);
    g_hot.max_tokens  = env_int("GGML_HOT_EXPERT_MAX_TOKENS", 1);
    g_hot.layers_spec = env_str("GGML_HOT_EXPERT_LAYERS", "all");
    g_hot.n_expert_used = n_expert_used;
    g_hot.markers       = env_flag("GGML_HOT_EXPERT_MARKERS", false);
    g_hot.remote_ep     = env_flag("GGML_HOT_EXPERT_REMOTE_EP", false);

    if (g_hot.max_tokens > 1) {
        LLAMA_LOG_WARN("%s: hot-expert graph supports max_tokens=1; clamping %" PRId64 " to 1\n", __func__, g_hot.max_tokens);
        g_hot.max_tokens = 1;
    }

    if (g_hot.table_path.empty() || g_hot.gguf_path.empty()) {
        LLAMA_LOG_ERROR("%s: GGML_HOT_EXPERT=1 requires GGML_HOT_EXPERT_TABLE and GGML_HOT_EXPERT_GGUF\n", __func__);
        return false;
    }
    if (n_expert_used <= 0) {
        LLAMA_LOG_ERROR("%s: invalid n_expert_used=%" PRId64 "\n", __func__, n_expert_used);
        return false;
    }

    std::map<int, std::map<int, int64_t>> table;
    if (!load_table(g_hot.table_path, table)) {
        LLAMA_LOG_ERROR("%s: failed to read hot table %s\n", __func__, g_hot.table_path.c_str());
        return false;
    }

    g_hot.dev = ggml_backend_dev_by_name(g_hot.dev_name.c_str());
    if (!g_hot.dev) {
        LLAMA_LOG_ERROR("%s: device %s not found\n", __func__, g_hot.dev_name.c_str());
        return false;
    }
    g_hot.backend = ggml_backend_dev_init(g_hot.dev, nullptr);
    if (!g_hot.backend) {
        LLAMA_LOG_ERROR("%s: failed to init backend on %s\n", __func__, g_hot.dev_name.c_str());
        return false;
    }

    struct gguf_init_params gip = { /*.no_alloc =*/ true, /*.ctx =*/ nullptr };
    struct gguf_context * gctx = gguf_init_from_file(g_hot.gguf_path.c_str(), gip);
    if (!gctx) {
        LLAMA_LOG_ERROR("%s: failed to open %s\n", __func__, g_hot.gguf_path.c_str());
        return false;
    }
    const size_t data_offset = gguf_get_data_offset(gctx);

    FILE * gf = fopen(g_hot.gguf_path.c_str(), "rb");
    if (!gf) {
        LLAMA_LOG_ERROR("%s: failed to read %s\n", __func__, g_hot.gguf_path.c_str());
        gguf_free(gctx);
        return false;
    }

    int n_built = 0;
    double mib = 0.0;
    for (const auto & [il, hits] : table) {
        if (il < 0 || il >= n_layer || !layer_selected(g_hot.layers_spec, il)) {
            continue;
        }
        hot_layer L;
        L.il = il;
        const float limit = swiglu_clamp_exp ? swiglu_clamp_exp[il] : 0.0f;
        if (build_layer(L, gf, gctx, data_offset, hits, limit)) {
            mib += 3.0 * L.n_embd * L.n_ff * L.K * 17.0 / 32 / 1048576.0;
            g_hot.layers.emplace(il, std::move(L));
            n_built++;
        }
    }

    fclose(gf);
    gguf_free(gctx);

    if (n_built == 0) {
        LLAMA_LOG_WARN("%s: no layers offloaded\n", __func__);
        return false;
    }

    g_hot.enabled = true;

#ifndef _WIN32
    // Workaround: libcuda's exit-time finalizer intermittently SIGSEGVs (bad
    // free inside its __cxa_finalize, driven by _dl_fini after main returns)
    // once extra CUDA contexts were created after model load. exit() runs
    // on_exit/atexit handlers before _dl_fini, so bailing out here with
    // _exit() preempts the buggy finalizer; the kernel reclaims all driver
    // state at process exit anyway. on_exit (unlike atexit) forwards the
    // original exit status so it is preserved. abort()/fatal signals bypass
    // handlers entirely, so those paths are unaffected. Handlers run before
    // glibc's own stdio teardown, so flush explicitly — otherwise the tail of
    // buffered stdout (timings line, "Exiting...") is lost.
    on_exit([](int status, void *) { fflush(NULL); _exit(status); }, nullptr);
#endif

    LLAMA_LOG_INFO("%s: hot-expert offload active: %d layers, top-%d/layer on %s, ~%.1f MiB weights, slot_order=%d, slot_avx512=%d\n",
        __func__, n_built, g_hot.K, g_hot.dev_name.c_str(), mib,
        llama_hot_expert_slot_order_enabled(), llama_hot_expert_slot_merge_avx512_enabled());
    if (g_hot.markers) {
        fprintf(stderr, "[hotmarker] init layers=%d K=%d dev=%s weight_mib=%.1f slot_order=%d slot_avx512=%d\n",
            n_built, g_hot.K, g_hot.dev_name.c_str(), mib,
            llama_hot_expert_slot_order_enabled(), llama_hot_expert_slot_merge_avx512_enabled());
    }
    return true;
}

void llama_hot_expert_shutdown() {
    if (!g_hot.enabled) {
        return;
    }
    // NOTE: intentionally leak everything (backend, buffers, events, contexts).
    // This runs inside ~llama_model while the CUDA driver and the model's own
    // backends are about to be torn down; issuing any CUDA free here races
    // with that teardown and intermittently SIGSEGVs at process exit (observed
    // with buffer/backend frees; gdb perturbs timing and hides it). The process
    // is exiting anyway, so the OS/driver reclaims all resources. We only
    // detach the layer table so no further graph-build touches it.
    g_hot.layers.clear();
    g_hot.backend = nullptr;
    g_hot.enabled = false;
}

void llama_hot_expert_mask_ids_cb(struct ggml_tensor * dst, const struct ggml_tensor * ids, int ith, int nth, void * userdata) {
    GGML_UNUSED(nth);
    if (ith != 0) {
        return;
    }
    const hot_layer & L = *(const hot_layer *) userdata;

    const int64_t k  = ids->ne[0];
    const int64_t nt = ids->ne[1];
    GGML_ASSERT(dst->type == GGML_TYPE_I32 && dst->ne[0] == k && dst->ne[1] == nt);

    for (int64_t t = 0; t < nt; t++) {
        for (int64_t s = 0; s < k; s++) {
            const int32_t e = *(const int32_t *) ((const char *) ids->data + s * ids->nb[0] + t * ids->nb[1]);
            const int32_t slot = (e >= 0 && e < L.n_expert) ? L.remap[e] : -1;
            // sentinel n_expert: the repack mmid skips out-of-range ids and
            // zeroes the slot (GGML_HOT_EXPERT path), so hot experts are never
            // read from DRAM and contribute exactly 0 to the cold partial
            *(int32_t *) ((char *) dst->data + s * dst->nb[0] + t * dst->nb[1]) = slot >= 0 ? (int32_t) L.n_expert : e;
        }
    }
}

void llama_hot_expert_send_cb(struct ggml_tensor * dst, const struct ggml_tensor * x, const struct ggml_tensor * ids, const struct ggml_tensor * w,
                              int ith, int nth, void * userdata) {
    GGML_UNUSED(nth);
    if (ith != 0) {
        return;
    }
    hot_layer & L = *(hot_layer *) userdata;

    if (g_hot.remote_ep) {
        bool expected = false;
        if (!L.remote_inflight->compare_exchange_strong(expected, true, std::memory_order_acq_rel)) {
            GGML_ABORT("%s: layer %d hot/remote staging is already in flight", __func__, L.il);
        }
    }

    static std::map<int, bool> reported;
    if (!reported.count(L.il)) {
        reported[L.il] = true;
        LLAMA_LOG_INFO("%s: layer %d hot fork active (K=%d on %s)\n", __func__, L.il, L.K, g_hot.dev_name.c_str());
        if (g_hot.markers) {
            fprintf(stderr, "[hotmarker] fork layer=%d K=%d dev=%s\n", L.il, L.K, g_hot.dev_name.c_str());
        }
    }

    GGML_ASSERT(x->type == GGML_TYPE_F32 && x->ne[0] == L.n_embd);
    GGML_ASSERT(ggml_is_contiguous(x));

    const int64_t k = ids->ne[0];
    GGML_ASSERT(ids->type == GGML_TYPE_I32 && ids->ne[1] == 1 && k == g_hot.n_expert_used);
    GGML_ASSERT(w->type == GGML_TYPE_F32 && w->ne[1] == k);

    // stage x
    memcpy(L.h_x, x->data, L.n_embd * sizeof(float));

    // remap ids into compact slots; cold slots point at slot 0 with weight 0
    for (int64_t s = 0; s < k; s++) {
        const int32_t e = *(const int32_t *) ((const char *) ids->data + s * ids->nb[0]);
        const int32_t slot = (e >= 0 && e < L.n_expert) ? L.remap[e] : -1;
        const float wt = *(const float *) ((const char *) w->data + s * w->nb[1]);
        L.h_ids[s] = slot >= 0 ? slot : 0;
        L.h_w[s]   = slot >= 0 ? wt : 0.0f;
        L.h_hot_mask[s] = slot >= 0;
    }

    // all stream-ordered on the executor backend: H2D -> graph -> D2H -> event
    static const bool packed_io = env_flag("GGML_HOT_EXPERT_PACKED_IO", true);
    if (packed_io) {
        ggml_backend_tensor_set_async(g_hot.backend, L.input_gpu, L.h_base, 0, L.input_bytes);
    } else {
        ggml_backend_tensor_set_async(g_hot.backend, L.x_gpu,   L.h_x,   0, L.n_embd * sizeof(float));
        ggml_backend_tensor_set_async(g_hot.backend, L.ids_gpu, L.h_ids, 0, k * sizeof(int32_t));
        ggml_backend_tensor_set_async(g_hot.backend, L.w_gpu,   L.h_w,   0, k * sizeof(float));
    }

    ggml_status status = ggml_backend_graph_compute_async(g_hot.backend, L.graph);
    if (status != GGML_STATUS_SUCCESS) {
        if (g_hot.remote_ep) {
            L.remote_inflight->store(false, std::memory_order_release);
        }
        GGML_ABORT("%s: layer %d GPU graph submission failed: %s", __func__, L.il, ggml_status_to_string(status));
    }

    ggml_backend_tensor_get_async(g_hot.backend, L.output_gpu, L.h_output, 0, L.output_bytes);
    ggml_backend_event_record(L.event, g_hot.backend);

    // dst content is never read (merge only uses it as a shape/ordering token)
    memset(dst->data, 0, ggml_nbytes(dst));
}

void llama_hot_expert_merge_cb(struct ggml_tensor * dst, const struct ggml_tensor * send, const struct ggml_tensor * cold,
                               int ith, int nth, void * userdata) {
    GGML_UNUSED(send);
    GGML_UNUSED(nth);
    if (ith != 0) {
        return;
    }
    hot_layer & L = *(hot_layer *) userdata;

    ggml_backend_event_synchronize(L.event);

    GGML_ASSERT(dst->type == GGML_TYPE_F32 && dst->ne[0] == L.n_embd && dst->ne[1] == 1 && dst->nb[0] == sizeof(float));
    GGML_ASSERT(cold->type == GGML_TYPE_F32 && cold->ne[0] == L.n_embd);
    float * out = (float *) dst->data;
    if (llama_hot_expert_slot_order_enabled()) {
        GGML_ASSERT(cold->nb[0] == sizeof(float));
        GGML_ASSERT(cold->nb[1] % sizeof(float) == 0);
        GGML_ASSERT(cold->ne[1] == g_hot.n_expert_used && cold->ne[2] == 1 && cold->ne[3] == 1);
        GGML_ASSERT(L.output_bytes == (size_t) L.n_embd*g_hot.n_expert_used*sizeof(float));

        // send and merge are serialized graph nodes, and each layer owns its
        // staging state, so the mask and weights cannot be overwritten here.
        llama_hot_expert_merge_slots_f32(
            out, (const float *) cold->data, cold->nb[1]/sizeof(float),
            L.h_output, L.n_embd, L.h_w, L.h_hot_mask.data(), L.n_embd, g_hot.n_expert_used);
    } else {
        GGML_ASSERT(cold->ne[1] == 1 && L.output_bytes == (size_t) L.n_embd*sizeof(float));
        for (int64_t i = 0; i < L.n_embd; i++) {
            const float c = *(const float *) ((const char *) cold->data + i*cold->nb[0]);
            out[i] = c + L.h_output[i];
        }
    }
    if (g_hot.remote_ep) {
        L.remote_inflight->store(false, std::memory_order_release);
    }
}

bool llama_hot_expert_remote_ep_split(
        void * userdata, uint8_t * cold_active, uint8_t * hot_mask, int64_t n_slots) {
    hot_layer & L = *(hot_layer *) userdata;
    if (!llama_hot_expert_remote_ep_enabled() || cold_active == nullptr || hot_mask == nullptr ||
            n_slots != g_hot.n_expert_used || (int64_t) L.h_hot_mask.size() != n_slots ||
            !L.remote_inflight->load(std::memory_order_acquire)) {
        return false;
    }
    for (int64_t slot = 0; slot < n_slots; ++slot) {
        hot_mask[slot] = L.h_hot_mask[(size_t) slot] != 0;
        cold_active[slot] = hot_mask[slot] ? 0 : 1;
    }
    return true;
}

void llama_hot_expert_remote_ep_merge(
        void * userdata, float * dst, const float * cold, size_t cold_slot_stride,
        int64_t n_embd, int64_t n_slots) {
    hot_layer & L = *(hot_layer *) userdata;
    GGML_ASSERT(llama_hot_expert_remote_ep_enabled());
    GGML_ASSERT(dst != nullptr && cold != nullptr && n_embd == L.n_embd);
    GGML_ASSERT(n_slots == g_hot.n_expert_used && (int64_t) L.h_hot_mask.size() == n_slots);
    GGML_ASSERT(L.output_bytes == (size_t) n_embd * n_slots * sizeof(float));
    ggml_backend_event_synchronize(L.event);
    llama_hot_expert_merge_slots_f32(
        dst, cold, cold_slot_stride, L.h_output, (size_t) n_embd,
        L.h_w, L.h_hot_mask.data(), n_embd, n_slots);
    L.remote_inflight->store(false, std::memory_order_release);
}
