#include "llama-layer-major.h"

#include "llama-batch.h"
#include "llama-context.h"
#include "llama-graph.h"
#include "llama-impl.h"
#include "llama-kv-cache-dsv4.h"
#include "llama-memory.h"
#include "llama-model.h"

#include <algorithm>
#include <cstring>
#include <cstdlib>
#include <iterator>
#include <limits>
#include <memory>
#include <new>

namespace {

ggml_backend_t backend_for_device(
        const std::vector<ggml_backend_t> & backends,
        ggml_backend_dev_t device) {
    for (ggml_backend_t backend : backends) {
        if (ggml_backend_get_device(backend) == device) {
            return backend;
        }
    }
    return nullptr;
}

class device_raw_kq_mask_cache {
public:
    bool record_layout(const ggml_tensor * tensor, size_t tile_index) {
        if (!tensor || !ggml_is_contiguous(tensor)) {
            return false;
        }
        if (layouts_.size() <= tile_index) {
            layouts_.resize(tile_index + 1);
        }

        auto & layout = layouts_[tile_index];
        layout.type = tensor->type;
        std::copy(std::begin(tensor->ne), std::end(tensor->ne), std::begin(layout.ne));
        layout.valid = true;
        return true;
    }

    void initialize(
            ggml_backend_sched_t sched,
            const std::vector<ggml_backend_t> & backends,
            const llama_kv_cache_dsv4_raw_context & raw_ctx) {
        if (layouts_.empty()) {
            return;
        }

        size_t cache_bytes = 0;
        for (size_t i = 0; i < layouts_.size(); ++i) {
            const auto * data = raw_ctx.get_input_replay_mask(i);
            if (!layouts_[i].valid || !data || data->size() > SIZE_MAX - cache_bytes) {
                return;
            }
            cache_bytes += data->size();
        }

        for (ggml_backend_t backend : backends) {
            if (ggml_backend_dev_type(ggml_backend_get_device(backend)) != GGML_BACKEND_DEVICE_TYPE_GPU) {
                continue;
            }

            size_t memory_free = 0;
            size_t memory_total = 0;
            ggml_backend_dev_memory(ggml_backend_get_device(backend), &memory_free, &memory_total);
            const size_t reserve_bytes = std::max<size_t>(4ull*1024*1024*1024, memory_total/5);
            if (cache_bytes > memory_free || reserve_bytes > memory_free - cache_bytes ||
                    layouts_.size() > (std::numeric_limits<size_t>::max() - 1024)/ggml_tensor_overhead() - 1) {
                LLAMA_LOG_WARN("%s: insufficient memory on %s for raw KQ replay cache\n",
                        __func__, ggml_backend_name(backend));
                continue;
            }

            backend_storage storage;
            storage.backend = backend;

            ggml_init_params params = {
                /*.mem_size   =*/ (layouts_.size() + 1)*ggml_tensor_overhead() + 1024,
                /*.mem_buffer =*/ nullptr,
                /*.no_alloc   =*/ true,
            };
            storage.ctx.reset(ggml_init(params));
            if (!storage.ctx) {
                continue;
            }

            storage.tiles.reserve(layouts_.size());
            for (size_t i = 0; i < layouts_.size(); ++i) {
                const auto & layout = layouts_[i];
                ggml_tensor * tile = ggml_new_tensor_4d(
                        storage.ctx.get(), layout.type,
                        layout.ne[0], layout.ne[1], layout.ne[2], layout.ne[3]);
                ggml_format_name(tile, "layer_major_raw_kq_mask_%zu", i);
                storage.tiles.push_back(tile);
            }

            ggml_backend_buffer_type_t buft = ggml_backend_sched_get_buffer_type(sched, backend);
            storage.buf.reset(ggml_backend_alloc_ctx_tensors_from_buft(storage.ctx.get(), buft));
            if (!storage.buf) {
                continue;
            }

            for (size_t i = 0; i < storage.tiles.size(); ++i) {
                const auto * data = raw_ctx.get_input_replay_mask(i);
                GGML_ASSERT(data && data->size() == ggml_nbytes(storage.tiles[i]));
                ggml_backend_tensor_set_async(
                        backend, storage.tiles[i], data->data(), 0, data->size());
            }

            LLAMA_LOG_INFO("%s: using %s raw KQ replay cache, size = %.2f MiB\n",
                    __func__, ggml_backend_name(backend), cache_bytes/(1024.0*1024.0));
            storage_.push_back(std::move(storage));
        }
    }

    const ggml_tensor * tile(ggml_backend_t backend, size_t tile_index) const {
        for (const auto & storage : storage_) {
            if (storage.backend == backend) {
                return tile_index < storage.tiles.size() ? storage.tiles[tile_index] : nullptr;
            }
        }
        return nullptr;
    }

private:
    struct tensor_layout {
        ggml_type type = GGML_TYPE_F32;
        int64_t ne[GGML_MAX_DIMS] = { 0, 0, 0, 0 };
        bool valid = false;
    };

    struct backend_storage {
        ggml_backend_t backend = nullptr;
        ggml_context_ptr ctx;
        ggml_backend_buffer_ptr buf;
        std::vector<ggml_tensor *> tiles;
    };

    std::vector<tensor_layout> layouts_;
    std::vector<backend_storage> storage_;
};

} // namespace

namespace {

// Staging hint for the bounded MoE copy-stream pipeline
// [GGML_CUDA_MOE_PP_PIPE]: stage the next layer's expert views into the
// per-GPU prefetch slots so their H2D overlaps the current layer's compute.
class moe_pipe_hint {
public:
    bool init(const std::vector<ggml_backend_t> & backends, int64_t n_tokens) {
        static const bool enabled = []() {
            const char * value = getenv("GGML_CUDA_MOE_PP_PIPE");
            return value && atoi(value) > 0;
        }();
        if (!enabled) {
            return false;
        }
        static const int64_t pp_min = []() {
            const char * value = getenv("GGML_CUDA_MOE_PP_MIN_TOKENS");
            return value ? (int64_t) atoll(value) : (int64_t) 0;
        }();
        static const int64_t ep_min = []() {
            const char * value = getenv("GGML_CUDA_MOE_PP_EP_MIN_TOKENS");
            return value ? std::max<int64_t>(0, atoll(value)) : (int64_t) 0;
        }();
        static const bool ep_env = []() {
            const char * value = getenv("GGML_CUDA_MOE_PP_EP");
            return value && atoi(value) > 0;
        }();
        ep_ = ep_env && pp_min > 0 && n_tokens >= std::max(ep_min, pp_min);
        // mirror of the scheduler's candidate gate: below this token count no
        // MoE weight is staged or committed, so hinting would only burn PCIe
        active_ = pp_min > 0 && n_tokens >= pp_min;
        if (!active_) {
            return false;
        }

        for (ggml_backend_t backend : backends) {
            if (ggml_backend_dev_type(ggml_backend_get_device(backend)) != GGML_BACKEND_DEVICE_TYPE_GPU) {
                continue;
            }
            ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(backend));
            auto fn = (moe_prefetch_auto_t)
                ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_moe_prefetch_auto");
            if (fn != nullptr) {
                backends_.push_back(backend);
                fns_.push_back(fn);
                // src variant: takes the weight tensor and un-repacks
                // CPU_REPACK buffers on the CPU before the H2D copy
                fns_src_.push_back((moe_prefetch_auto_src_t)
                    ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_moe_prefetch_auto_src"));
            }
        }
        return !backends_.empty();
    }

    // Mirror of the scheduler's staged size for the EP rank views of w.
    static size_t rank_view_nbytes(const ggml_tensor * w, int64_t half) {
        const size_t blck_size = ggml_blck_size(w->type);
        size_t nbytes = blck_size == 1 ?
            (size_t) ggml_type_size(w->type) + (w->ne[0] - 1)*w->nb[0] :
            (size_t) w->ne[0]*w->nb[0]/blck_size;
        nbytes += (w->ne[1] - 1)*w->nb[1] + (half - 1)*w->nb[2];
        return nbytes;
    }

    void hint(const llama_model & model, const std::vector<ggml_backend_t> & backends, int32_t il) const {
        if (backends_.empty()) {
            return;
        }
        const llama_layer & layer = model.layers[il];
        const bool ep = ep_ && backends_.size() >= 2 &&
            layer.ffn_gate_exps && layer.ffn_up_exps && layer.ffn_down_exps &&
            !layer.ffn_gate_up_exps &&
            layer.ffn_gate_exps->type == GGML_TYPE_MXFP4 &&
            layer.ffn_up_exps->type   == GGML_TYPE_MXFP4 &&
            layer.ffn_down_exps->type == GGML_TYPE_MXFP4;
        const ggml_tensor * weights[] = { layer.ffn_up_exps, layer.ffn_gate_exps, layer.ffn_down_exps };
        for (const ggml_tensor * w : weights) {
            if (w == nullptr || w->data == nullptr) {
                continue;
            }
            ggml_backend_buffer_t buf = w->view_src ? w->view_src->buffer : w->buffer;
            if (buf == nullptr || !ggml_backend_buffer_is_host(buf) ||
                    ggml_backend_buffer_get_usage(buf) != GGML_BACKEND_BUFFER_USAGE_WEIGHTS) {
                continue;
            }
            // CPU_REPACK weights need the src variant (CPU-side inverse
            // transform before H2D). Without EP the MoE op is not pinned to
            // the GPU for host weights, so hinting would only burn bandwidth.
            const bool repack = strcmp(ggml_backend_buffer_name(buf), "CPU_REPACK") == 0;
            if (repack && !ep) {
                continue;
            }
            if (ep && w->ne[2] % 2 == 0) {
                const int64_t half = w->ne[2]/2;
                for (int rank = 0; rank < 2; ++rank) {
                    const char * data = (const char *) w->data + rank*half*w->nb[2];
                    const size_t size = rank_view_nbytes(w, half);
                    if (fns_src_[rank] != nullptr) {
                        fns_src_[rank](backends_[rank], w, data, size);
                    } else if (!repack) {
                        fns_[rank](backends_[rank], data, size);
                    }
                }
            } else {
                ggml_backend_t backend = backend_for_device(backends, model.dev_layer(il));
                for (size_t i = 0; i < backends_.size(); ++i) {
                    if (backends_[i] == backend) {
                        if (fns_src_[i] != nullptr) {
                            fns_src_[i](backend, w, w->data, ggml_nbytes(w));
                        } else if (!repack) {
                            fns_[i](backend, w->data, ggml_nbytes(w));
                        }
                        break;
                    }
                }
            }
        }
    }

private:
    typedef int (*moe_prefetch_auto_t)(ggml_backend_t, const void *, size_t);
    typedef int (*moe_prefetch_auto_src_t)(ggml_backend_t, const ggml_tensor *, const void *, size_t);

    bool active_ = false;
    bool ep_ = false;
    std::vector<ggml_backend_t> backends_;
    std::vector<moe_prefetch_auto_t> fns_;
    std::vector<moe_prefetch_auto_src_t> fns_src_;
};

} // namespace

namespace {

// Preflight VRAM budget for the streaming MoE prefetch slots: production must
// never discover the shortfall mid-prefill (the device slot buffers allocate
// lazily on first use). Estimates GGML_CUDA_MOE_PP_MAX_PREFETCH slots of the
// largest staged expert view per GPU plus margin and compares against the
// free memory of every GPU backend.
bool llama_layer_major_vram_budget_ok(
        const llama_model & model,
        const std::vector<ggml_backend_t> & backends,
        int64_t n_tokens) {
    static const int64_t pp_min = []() {
        const char * value = getenv("GGML_CUDA_MOE_PP_MIN_TOKENS");
        return value ? (int64_t) atoll(value) : (int64_t) 0;
    }();
    static const int depth = []() {
        const char * value = getenv("GGML_CUDA_MOE_PP_PREFETCH");
        return value ? std::max(0, std::min(4, atoi(value))) : 0;
    }();
    static const bool ep_env = []() {
        const char * value = getenv("GGML_CUDA_MOE_PP_EP");
        return value && atoi(value) > 0;
    }();
    if (pp_min <= 0 || depth <= 0 || n_tokens < pp_min) {
        return true; // the scheduler will not stage any MoE weight
    }

    size_t max_bytes = 0;
    for (const auto & layer : model.layers) {
        const ggml_tensor * weights[] = {
            layer.ffn_up_exps, layer.ffn_gate_exps, layer.ffn_down_exps, layer.ffn_gate_up_exps,
        };
        for (const ggml_tensor * w : weights) {
            if (w == nullptr || w->data == nullptr) {
                continue;
            }
            ggml_backend_buffer_t buf = w->view_src ? w->view_src->buffer : w->buffer;
            if (buf == nullptr || !ggml_backend_buffer_is_host(buf) ||
                    ggml_backend_buffer_get_usage(buf) != GGML_BACKEND_BUFFER_USAGE_WEIGHTS) {
                continue;
            }
            max_bytes = std::max(max_bytes, ggml_nbytes(w));
        }
    }
    if (max_bytes == 0) {
        return true; // no host MoE weight, nothing is staged
    }

    // EP mode stages per-rank halves of each expert tensor on every GPU
    const size_t staged = ep_env && backends.size() >= 2 ? max_bytes/2 : max_bytes;

    constexpr int n_slots = 4; // GGML_CUDA_MOE_PP_MAX_PREFETCH
    const size_t margin   = (size_t) 1*1024*1024*1024;
    const size_t required = (size_t) n_slots*staged + margin;

    typedef bool (*mem_get_info_t)(ggml_backend_t, size_t *, size_t *);
    for (ggml_backend_t backend : backends) {
        if (ggml_backend_dev_type(ggml_backend_get_device(backend)) != GGML_BACKEND_DEVICE_TYPE_GPU) {
            continue;
        }
        ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(backend));
        auto fn = reg != nullptr ? (mem_get_info_t)
            ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_mem_get_info") : nullptr;
        if (fn == nullptr) {
            continue; // cannot query: do not block the fast path on it
        }
        size_t free_mem = 0;
        size_t total_mem = 0;
        if (!fn(backend, &free_mem, &total_mem)) {
            continue;
        }
        if (free_mem < required) {
            LLAMA_LOG_WARN("%s: VRAM budget insufficient on %s: free %.2f of %.2f GiB < required %.2f GiB "
                    "(%d prefetch slots x %.2f GiB staged + %.2f GiB margin)\n",
                    __func__, ggml_backend_name(backend),
                    free_mem/1073741824.0, total_mem/1073741824.0, required/1073741824.0,
                    n_slots, staged/1073741824.0, margin/1073741824.0);
            return false;
        }
    }
    return true;
}

} // namespace

ggml_backend_t llama_layer_major_buffer_backend(
        const std::vector<ggml_backend_t> & backends,
        const ggml_tensor * tensor) {
    if (!tensor) {
        return nullptr;
    }

    ggml_backend_buffer_t buffer = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;
    ggml_backend_dev_t device = buffer ?
        ggml_backend_buft_get_device(ggml_backend_buffer_get_type(buffer)) : nullptr;

    for (ggml_backend_t backend : backends) {
        if (ggml_backend_get_device(backend) == device) {
            return backend;
        }
    }
    return nullptr;
}

bool llama_layer_major_hc_state::init(
        size_t n_tokens,
        size_t hc_dim,
        size_t n_ubatch,
        ggml_backend_sched_t sched,
        const std::vector<ggml_backend_t> & backends) {
    hc_dim_ = hc_dim;

    try {
        host_.resize(n_tokens*hc_dim);
    } catch (const std::bad_alloc &) {
        return false;
    }

    const char * env = getenv("LLAMA_LAYER_MAJOR_DEVICE_HC");
    const int device_mode = env ? atoi(env) : 0;
    if (device_mode <= 0) {
        return true;
    }
    migrate_between_backends_ = device_mode >= 2;

    const size_t state_bytes = n_tokens*hc_dim*sizeof(float);
    const size_t n_tiles = (n_tokens + n_ubatch - 1)/n_ubatch;
    if (n_tiles > (std::numeric_limits<size_t>::max() - 1024)/ggml_tensor_overhead() - 1) {
        return true;
    }

    for (ggml_backend_t backend : backends) {
        if (ggml_backend_dev_type(ggml_backend_get_device(backend)) != GGML_BACKEND_DEVICE_TYPE_GPU) {
            continue;
        }

        size_t memory_free = 0;
        size_t memory_total = 0;
        ggml_backend_dev_memory(ggml_backend_get_device(backend), &memory_free, &memory_total);
        const size_t reserve_bytes = std::max<size_t>(4ull*1024*1024*1024, memory_total/5);
        if (state_bytes > memory_free || reserve_bytes > memory_free - state_bytes) {
            LLAMA_LOG_WARN("%s: insufficient memory on %s for device HC state\n",
                    __func__, ggml_backend_name(backend));
            continue;
        }

        device_storage storage;
        storage.backend = backend;

        ggml_init_params params = {
            /*.mem_size   =*/ (n_tiles + 1)*ggml_tensor_overhead() + 1024,
            /*.mem_buffer =*/ nullptr,
            /*.no_alloc   =*/ true,
        };
        storage.ctx.reset(ggml_init(params));
        if (!storage.ctx) {
            continue;
        }

        ggml_tensor * state = ggml_new_tensor_2d(storage.ctx.get(), GGML_TYPE_F32, hc_dim, n_tokens);
        ggml_format_name(state, "layer_major_hc_%s", ggml_backend_name(backend));

        storage.tiles.reserve(n_tiles);
        for (size_t i = 0, token_offset = 0; i < n_tiles; ++i, token_offset += n_ubatch) {
            const size_t tile_tokens = std::min(n_ubatch, n_tokens - token_offset);
            ggml_tensor * tile = ggml_view_2d(
                    storage.ctx.get(), state, hc_dim, tile_tokens,
                    hc_dim*sizeof(float), token_offset*hc_dim*sizeof(float));
            ggml_format_name(tile, "layer_major_hc_%s_%zu", ggml_backend_name(backend), i);
            storage.tiles.push_back(tile);
        }

        ggml_backend_buffer_type_t buft = ggml_backend_sched_get_buffer_type(sched, backend);
        storage.buf.reset(ggml_backend_alloc_ctx_tensors_from_buft(storage.ctx.get(), buft));
        if (!storage.buf) {
            LLAMA_LOG_WARN("%s: device HC allocation failed on %s\n",
                    __func__, ggml_backend_name(backend));
            continue;
        }

        LLAMA_LOG_INFO("%s: using %s HC state, size = %.2f MiB\n",
                __func__, ggml_backend_name(backend), state_bytes/(1024.0*1024.0));
        device_storage_.push_back(std::move(storage));

        // Mode 1 preserves the proven first-GPU residency policy. Mode 2 is
        // the explicit NVLink/P2P experiment with one state copy per GPU.
        if (!migrate_between_backends_) {
            break;
        }
    }

    if (device_storage_.empty()) {
        LLAMA_LOG_WARN("%s: no device HC state is available, using host state\n", __func__);
        return true;
    }

    active_storage_ = 0;
    std::vector<float>().swap(host_);
    return true;
}

bool llama_layer_major_hc_state::is_device_resident() const {
    return active_storage_ < device_storage_.size();
}

ggml_backend_t llama_layer_major_hc_state::device_backend() const {
    return is_device_resident() ? device_storage_[active_storage_].backend : nullptr;
}

bool llama_layer_major_hc_state::select_backend(ggml_backend_t backend) {
    if (!backend || !is_device_resident()) {
        return false;
    }
    if (!migrate_between_backends_) {
        return true;
    }

    size_t next_storage = 0;
    for (; next_storage < device_storage_.size(); ++next_storage) {
        if (device_storage_[next_storage].backend == backend) {
            break;
        }
    }
    if (next_storage == device_storage_.size()) {
        // The original resident backend remains a valid compatibility path;
        // the graph scheduler will perform the required peer copies.
        return true;
    }
    if (next_storage == active_storage_) {
        return true;
    }

    const auto & src = device_storage_[active_storage_];
    const auto & dst = device_storage_[next_storage];
    if (src.tiles.size() != dst.tiles.size()) {
        return false;
    }
    for (size_t i = 0; i < src.tiles.size(); ++i) {
        if (!ggml_are_same_shape(src.tiles[i], dst.tiles[i]) ||
                !ggml_are_same_stride(src.tiles[i], dst.tiles[i])) {
            return false;
        }
        ggml_backend_tensor_copy_async(src.backend, dst.backend, src.tiles[i], dst.tiles[i]);
    }

    active_storage_ = next_storage;
    return true;
}

float * llama_layer_major_hc_state::host_tile(size_t token_offset) {
    if (is_device_resident() || hc_dim_ == 0 || token_offset >= host_.size()/hc_dim_) {
        return nullptr;
    }
    return host_.data() + token_offset*hc_dim_;
}

const ggml_tensor * llama_layer_major_hc_state::device_tile(size_t tile_index) const {
    if (!is_device_resident()) {
        return nullptr;
    }
    const auto & storage = device_storage_[active_storage_];
    return tile_index < storage.tiles.size() ? storage.tiles[tile_index] : nullptr;
}

bool llama_layer_major_hc_state::store_tile(
        ggml_backend_sched_t sched,
        const std::vector<ggml_backend_t> & backends,
        ggml_tensor * src,
        size_t tile_index,
        size_t token_offset) {
    if (!is_device_resident()) {
        ggml_backend_t backend = ggml_backend_sched_get_tensor_backend(sched, src);
        float * dst = host_tile(token_offset);
        if (!backend || !dst) {
            return false;
        }
        ggml_backend_tensor_get_async(backend, src, dst, 0, ggml_nbytes(src));
        return true;
    }

    auto & storage = device_storage_[active_storage_];
    ggml_tensor * dst = tile_index < storage.tiles.size() ? storage.tiles[tile_index] : nullptr;
    ggml_backend_t backend = llama_layer_major_buffer_backend(backends, src);
    if (!dst || !backend || src->type != dst->type ||
            !ggml_are_same_shape(src, dst) || !ggml_are_same_stride(src, dst)) {
        return false;
    }

    ggml_backend_tensor_copy_async(backend, storage.backend, src, dst);
    return true;
}

int llama_context::decode_layer_major(const llama_batch & batch_inp, uint32_t n_ubatch) {
    // Keep the initial implementation narrow so unsupported requests retain
    // the regular llama_decode() path and its full compatibility surface.
    if (model.arch != LLM_ARCH_DEEPSEEK4 ||
            cparams.ctx_type != LLAMA_CONTEXT_TYPE_DEFAULT ||
            !cparams.causal_attn || cparams.embeddings ||
            cparams.embeddings_nextn || cparams.embeddings_pre_norm ||
            cparams.pooling_type != LLAMA_POOLING_TYPE_NONE ||
            !sampling.samplers.empty() || !loras->empty() ||
            batch_inp.n_tokens <= 0 || !batch_inp.token || batch_inp.embd ||
            n_ubatch == 0 || n_ubatch > cparams.n_ubatch) {
        LLAMA_LOG_INFO("%s: ineligible: arch=%d ctx_type=%d causal=%d embd=%d nextn=%d prenorm=%d pooling=%d samplers=%zu loras=%zu n_tokens=%d token=%p embd=%p n_ubatch=%u cparams_ubatch=%u\n",
                __func__, (int) model.arch, (int) cparams.ctx_type, (int) cparams.causal_attn,
                (int) cparams.embeddings, (int) cparams.embeddings_nextn, (int) cparams.embeddings_pre_norm,
                (int) cparams.pooling_type, sampling.samplers.size(), loras->size(),
                batch_inp.n_tokens, (void *) batch_inp.token, (void *) batch_inp.embd,
                n_ubatch, cparams.n_ubatch);
        return -1;
    }

    if (!dynamic_cast<llama_kv_cache_dsv4 *>(memory.get())) {
        LLAMA_LOG_INFO("%s: ineligible: KV memory is not llama_kv_cache_dsv4\n", __func__);
        return -1;
    }

    const auto & vocab   = model.vocab;
    const auto & hparams = model.hparams;
    const uint32_t n_seq_max = cparams.kv_unified ? LLAMA_MAX_SEQ : cparams.n_seq_max;

    try {
        if (!balloc->init(batch_inp, vocab, memory.get(), hparams.n_embd_inp(), n_seq_max, false)) {
            return -1;
        }
    } catch (const std::bad_alloc &) {
        return -2;
    }

    const llama_batch & batch = balloc->get_batch();
    const uint32_t n_tokens_all = balloc->get_n_tokens();
    const uint32_t n_outputs_all = balloc->get_n_outputs();

    // Initial-prompt, one-sequence sessions give DSV4 a reliable rollback:
    // seq_rm(seq, 0, -1) clears raw KV, compressed KV and compressor state.
    if (n_outputs_all != 1 || batch.logits[n_tokens_all - 1] == 0) {
        LLAMA_LOG_INFO("%s: ineligible: n_outputs=%u last_logits=%d\n",
                __func__, n_outputs_all, (int) batch.logits[n_tokens_all - 1]);
        return -1;
    }

    const llama_seq_id seq_id = batch.seq_id[0][0];
    for (uint32_t i = 0; i < n_tokens_all; ++i) {
        if (batch.n_seq_id[i] != 1 || batch.seq_id[i][0] != seq_id ||
                batch.pos[i] != (llama_pos) i ||
                batch.logits[i] != (int8_t) (i + 1 == n_tokens_all)) {
            LLAMA_LOG_INFO("%s: ineligible: token %u n_seq_id=%d seq=%d pos=%d logits=%d\n",
                    __func__, i, (int) batch.n_seq_id[i], (int) batch.seq_id[i][0],
                    (int) batch.pos[i], (int) batch.logits[i]);
            return -1;
        }
    }
    if (memory->seq_pos_max(seq_id) >= 0) {
        LLAMA_LOG_INFO("%s: ineligible: seq %d has cached pos %d\n",
                __func__, (int) seq_id, (int) memory->seq_pos_max(seq_id));
        return -1;
    }

    // preflight the prefetch-slot VRAM budget before anything allocates;
    // a shortfall here still has the untouched chunked path as fallback
    if (!llama_layer_major_vram_budget_ok(model, backend_ptrs, n_tokens_all)) {
        return 1;
    }

    const size_t hc_dim = hparams.n_embd_h();
    if (hc_dim == 0 || n_tokens_all > SIZE_MAX/hc_dim ||
            (size_t) n_tokens_all*hc_dim > SIZE_MAX/sizeof(float)) {
        return -2;
    }

    if (t_compute_start_us == 0) {
        t_compute_start_us = ggml_time_us();
    }
    n_queued_tokens += n_tokens_all;
    embd_seq.clear();
    output_swaps.clear();

    sched_reserve();

    // Allocate the layer boundary before applying cache slots so allocation
    // failure remains safe for caller fallback.
    llama_layer_major_hc_state hc_state;
    if (!hc_state.init(n_tokens_all, hc_dim, n_ubatch, sched.get(), backend_ptrs)) {
        return -2;
    }

    memory_update(false);

    auto mctx_base = memory->init_batch(*balloc, n_ubatch, false);
    if (!mctx_base) {
        return -2;
    }
    switch (mctx_base->get_status()) {
        case LLAMA_MEMORY_STATUS_SUCCESS: break;
        case LLAMA_MEMORY_STATUS_FAILED_PREPARE: return 1;
        case LLAMA_MEMORY_STATUS_NO_UPDATE:
        case LLAMA_MEMORY_STATUS_FAILED_COMPUTE: return -2;
    }

    auto * mctx = dynamic_cast<llama_kv_cache_dsv4_context *>(mctx_base.get());
    if (!mctx) {
        return -1;
    }
    mctx->enable_input_replay_cache();

    if (output_reserve(1) < 1) {
        return -2;
    }
    if (output_ids.size() < n_tokens_all) {
        output_ids.resize(n_tokens_all, -1);
    }
    std::fill(output_ids.begin(), output_ids.end(), -1);

    const int32_t n_layer = hparams.n_layer();
    const int64_t n_vocab = vocab.n_tokens();
    bool cache_touched = false;
    device_raw_kq_mask_cache raw_kq_mask_cache;

    const char * profile_env = getenv("LLAMA_LAYER_MAJOR_PROFILE");
    const bool profile_enabled = profile_env && atoi(profile_env) > 0;
    llama_layer_major_ubatch_profile ubatch_profile;
    int64_t store_us = 0;
    int64_t sync_us = 0;
    int64_t rewind_us = 0;
    size_t tile_count = 0;
    const int64_t total_start_us = profile_enabled ? ggml_time_us() : 0;

    auto rollback = [&](int rc) {
        ggml_backend_sched_synchronize(sched.get());
        if (cache_touched) {
            memory->seq_rm(seq_id, 0, -1);
        }
        return rc;
    };

    moe_pipe_hint pipe_hint;
    if (pipe_hint.init(backend_ptrs, n_tokens_all)) {
        pipe_hint.hint(model, backend_ptrs, 0);
    }

    for (int32_t il = 0; il < n_layer; ++il) {
        ggml_backend_t layer_backend = backend_for_device(backend_ptrs, model.dev_layer(il));
        if (hc_state.is_device_resident() && layer_backend &&
                ggml_backend_dev_type(ggml_backend_get_device(layer_backend)) == GGML_BACKEND_DEVICE_TYPE_GPU &&
                !hc_state.select_backend(layer_backend)) {
            LLAMA_LOG_WARN("%s: no device-local HC state is available for layer %d on %s\n",
                    __func__, il, ggml_backend_name(layer_backend));
        }

        size_t token_offset = 0;
        size_t tile_index = 0;

        do {
            const llama_ubatch & prepared = mctx->get_ubatch();
            llama_ubatch ubatch = prepared;

            if (il > 0) {
                ubatch.embd = hc_state.host_tile(token_offset);
            }

            const ggml_tensor * h_src = il > 0 ? hc_state.device_tile(tile_index) : nullptr;
            if (hc_state.is_device_resident() && il > 0 && !h_src) {
                return rollback(-3);
            }

            const ggml_tensor * raw_kq_mask = il > 0 ? raw_kq_mask_cache.tile(layer_backend, tile_index) : nullptr;

            n_outputs = il + 1 == n_layer ?
                std::count(ubatch.output, ubatch.output + ubatch.n_tokens, (int8_t) 1) : 0;

            llama_layer_major_graph_input graph_input = {
                /*.hc_backend         =*/ hc_state.device_backend(),
                /*.hc_tensor          =*/ h_src,
                /*.raw_kq_mask_backend =*/ raw_kq_mask ? layer_backend : nullptr,
                /*.raw_kq_mask         =*/ raw_kq_mask,
                /*.profile             =*/ profile_enabled ? &ubatch_profile : nullptr,
            };

            ggml_status status = GGML_STATUS_SUCCESS;
            const llm_graph_result * res = process_ubatch(
                    ubatch, LLM_GRAPH_TYPE_DEFAULT, mctx, status, il, il + 1,
                    hc_state.is_device_resident() || raw_kq_mask || profile_enabled ? &graph_input : nullptr);
            cache_touched = true;

            if (!res) {
                switch (status) {
                    case GGML_STATUS_ABORTED:      return rollback( 2);
                    case GGML_STATUS_ALLOC_FAILED: return rollback(-2);
                    case GGML_STATUS_FAILED:       return rollback(-3);
                    case GGML_STATUS_SUCCESS:      GGML_ABORT("should not happen");
                }
            }

            if (il == 1 && hc_state.is_device_resident() &&
                    !raw_kq_mask_cache.record_layout(res->get_dsv4_raw_kq_mask(), tile_index)) {
                LLAMA_LOG_WARN("%s: raw KQ replay cache layout is unavailable\n", __func__);
            }

            const int64_t store_start_us = profile_enabled ? ggml_time_us() : 0;
            if (il + 1 < n_layer) {
                ggml_tensor * h = res->get_h_pre_norm();
                if (!h || ggml_nelements(h) != (int64_t) ubatch.n_tokens*(int64_t) hc_dim) {
                    return rollback(-3);
                }
                if (!hc_state.store_tile(sched.get(), backend_ptrs, h, tile_index, token_offset)) {
                    return rollback(-3);
                }
            } else if (n_outputs > 0) {
                ggml_tensor * t_logits = res->get_logits();
                if (!t_logits || !logits.data) {
                    return rollback(-3);
                }
                ggml_backend_t backend_logits = ggml_backend_sched_get_tensor_backend(sched.get(), t_logits);
                if (!backend_logits) {
                    return rollback(-3);
                }
                ggml_backend_tensor_get_async(backend_logits, t_logits, logits.data, 0,
                        (size_t) n_outputs*n_vocab*sizeof(float));
            }
            if (profile_enabled) {
                store_us += ggml_time_us() - store_start_us;
            }

            token_offset += ubatch.n_tokens;
            ++tile_index;
            ++tile_count;
        } while (mctx->next());

        // stage the next layer's expert weights before the boundary sync so
        // the slot H2D overlaps this layer's remaining compute
        if (il + 1 < n_layer) {
            pipe_hint.hint(model, backend_ptrs, il + 1);
        }

        const int64_t sync_start_us = profile_enabled ? ggml_time_us() : 0;
        ggml_backend_sched_synchronize(sched.get());
        if (profile_enabled) {
            sync_us += ggml_time_us() - sync_start_us;
        }

        // diagnostic: per-layer hc checksum to localize nondeterministic drift
        // (LLAMA_LAYER_MAJOR_DEBUG_SUM=1); the hc state holds this layer's
        // output right after the boundary sync
        static const bool debug_sum = []() {
            const char * value = getenv("LLAMA_LAYER_MAJOR_DEBUG_SUM");
            return value && atoi(value) > 0;
        }();
        if (debug_sum && il + 1 < n_layer) {
            std::vector<float> state((size_t) n_tokens_all*hc_dim);
            bool got = false;
            if (hc_state.is_device_resident()) {
                const ggml_tensor * t0 = hc_state.device_tile(0);
                const ggml_tensor * full = t0 != nullptr && t0->view_src != nullptr ? t0->view_src : t0;
                if (full != nullptr && hc_state.device_backend() != nullptr) {
                    ggml_backend_tensor_get(full, state.data(), 0, state.size()*sizeof(float));
                    got = true;
                }
            } else if (const float * h = hc_state.host_tile(0)) {
                memcpy(state.data(), h, state.size()*sizeof(float));
                got = true;
            }
            if (got) {
                double sum = 0.0;
                for (const float v : state) {
                    sum += v;
                }
                LLAMA_LOG_INFO("LMDBG layer %d hc sum %.9e\n", il, sum);
            }
        }

        if (token_offset != n_tokens_all) {
            return rollback(-3);
        }
        if (il == 1 && hc_state.is_device_resident()) {
            raw_kq_mask_cache.initialize(sched.get(), backend_ptrs, *mctx->get_raw());
        }
        if (il + 1 < n_layer) {
            const int64_t rewind_start_us = profile_enabled ? ggml_time_us() : 0;
            mctx->rewind(true);
            if (profile_enabled) {
                rewind_us += ggml_time_us() - rewind_start_us;
            }
        }
    }

    if (profile_enabled) {
        const int64_t total_us = ggml_time_us() - total_start_us;
        LLAMA_LOG_INFO(
                "%s: total %.3f ms, tiles %zu, apply %.3f ms, graph %.3f ms, "
                "inputs %.3f ms, submit %.3f ms, store %.3f ms, sync %.3f ms, rewind %.3f ms\n",
                __func__, total_us/1000.0, tile_count,
                ubatch_profile.memory_apply_us/1000.0,
                ubatch_profile.graph_prepare_us/1000.0,
                ubatch_profile.set_inputs_us/1000.0,
                ubatch_profile.submit_us/1000.0,
                store_us/1000.0, sync_us/1000.0, rewind_us/1000.0);
    }

    // On by default: after a large layer-major prefill, pack the raw SWA
    // window into a 256-cell decode ring so q1 decode graph width stays
    // decoupled from the prompt length. LLAMA_DSV4_COMPACT_DECODE_SWA=0
    // restores the full-cache behavior.
    static const bool compact_decode_swa = []() {
        const char * value = getenv("LLAMA_DSV4_COMPACT_DECODE_SWA");
        return value ? atoi(value) > 0 : true;
    }();
    if (compact_decode_swa && hparams.n_swa > 0 && n_tokens_all > hparams.n_swa) {
        auto * dsv4_memory = dynamic_cast<llama_kv_cache_dsv4 *>(memory.get());
        llama_kv_cache * raw_swa = dsv4_memory ? dsv4_memory->get_raw()->get_swa() : nullptr;
        // Multi-slot contexts (n_seq_max > 1) keep one raw SWA stream per
        // sequence; a single ring bound cannot isolate them, so they keep
        // full-cache semantics. compact_decode_window re-checks this together
        // with per-cell exclusive ownership before moving any data.
        if (raw_swa && raw_swa->get_n_stream() == 1) {
            const uint32_t n_keep = hparams.n_swa;
            const uint32_t capacity = GGML_PAD(2*n_keep, 256);
            if (!raw_swa->compact_decode_window(seq_id, n_keep, capacity)) {
                LLAMA_LOG_WARN("%s: unable to compact the raw SWA cache for decode; using the full cache\n", __func__);
            }
        }
    }

    n_outputs = 1;
    output_ids[n_tokens_all - 1] = 0;
    return 0;
}
