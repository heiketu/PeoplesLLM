#include "llama-model.h"

#include "llama-arch.h"
#include "llama-ext.h"
#include "llama-hparams.h"
#include "llama-impl.h"
#include "llama-mmap.h"
#include "llama-cparams.h"
#include "llama-model-loader.h"

#include "llama-kv-cache.h"
#include "llama-kv-cache-iswa.h"
#include "llama-kv-cache-dsa.h"
#include "llama-kv-cache-msa.h"
#include "llama-kv-cache-dsv4.h"
#include "llama-layer-major.h"
#include "llama-memory-hybrid.h"
#include "llama-memory-hybrid-iswa.h"
#include "llama-memory-recurrent.h"
#include "llama-numa.h"
#include "ggml-shard-plan.h"

#include "llama.h"
#include "models/models.h"

#include "ggml.h"
#include "ggml-cpp.h"

#include <algorithm>
#include <cassert>
#include <cfloat>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <functional>
#include <map>
#include <numeric>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#if defined(__gnu_linux__)
#include <cerrno>
#include <sys/mman.h>
#ifndef MADV_COLLAPSE
#define MADV_COLLAPSE 25 // linux >= 6.1
#endif
#endif

static llama_model * llama_model_mapping(llm_arch arch, const llama_model_params & params) {
    switch (arch) {
        case LLM_ARCH_CLIP:
            return new llama_model_clip(params);
        case LLM_ARCH_LLAMA:
            return new llama_model_llama(params);
        case LLM_ARCH_LLAMA4:
            return new llama_model_llama4(params);
        case LLM_ARCH_LLAMA_EMBED:
            return new llama_model_llama_embed(params);
        case LLM_ARCH_MAINCODER:
            return new llama_model_maincoder(params);
        case LLM_ARCH_TALKIE:
            return new llama_model_talkie(params);
        case LLM_ARCH_DECI:
            return new llama_model_deci(params);
        case LLM_ARCH_BAICHUAN:
            return new llama_model_baichuan(params);
        case LLM_ARCH_FALCON:
            return new llama_model_falcon(params);
        case LLM_ARCH_GROK:
            return new llama_model_grok(params);
        case LLM_ARCH_STARCODER:
            return new llama_model_starcoder(params);
        case LLM_ARCH_REFACT:
            return new llama_model_refact(params);
        case LLM_ARCH_BERT:
            return new llama_model_bert(params);
        case LLM_ARCH_JINA_BERT_V2:
            return new llama_model_jina_bert_v2(params);
        case LLM_ARCH_JINA_BERT_V3:
            return new llama_model_jina_bert_v3(params);
        case LLM_ARCH_NOMIC_BERT:
            return new llama_model_nomic_bert(params);
        case LLM_ARCH_NOMIC_BERT_MOE:
            return new llama_model_nomic_bert_moe(params);
        case LLM_ARCH_MODERN_BERT:
            return new llama_model_modern_bert(params);
        case LLM_ARCH_NEO_BERT:
            return new llama_model_neo_bert(params);
        case LLM_ARCH_EUROBERT:
            return new llama_model_eurobert(params);
        case LLM_ARCH_BLOOM:
            return new llama_model_bloom(params);
        case LLM_ARCH_MPT:
            return new llama_model_mpt(params);
        case LLM_ARCH_STABLELM:
            return new llama_model_stablelm(params);
        case LLM_ARCH_MELLUM:
            return new llama_model_mellum(params);
        case LLM_ARCH_NANBEIGE:
            return new llama_model_nanbeige(params);
        case LLM_ARCH_QWEN:
            return new llama_model_qwen(params);
        case LLM_ARCH_QWEN2:
            return new llama_model_qwen2(params);
        case LLM_ARCH_DREAM:
            return new llama_model_dream(params);
        case LLM_ARCH_LLADA:
            return new llama_model_llada(params);
        case LLM_ARCH_LLADA_MOE:
            return new llama_model_llada_moe(params);
        case LLM_ARCH_RND1:
            return new llama_model_rnd1(params);
        case LLM_ARCH_QWEN2VL:
            return new llama_model_qwen2vl(params);
        case LLM_ARCH_QWEN2MOE:
            return new llama_model_qwen2moe(params);
        case LLM_ARCH_QWEN3:
            return new llama_model_qwen3(params);
        case LLM_ARCH_QWEN3MOE:
            return new llama_model_qwen3moe(params);
        case LLM_ARCH_QWEN3VL:
            return new llama_model_qwen3vl(params);
        case LLM_ARCH_QWEN3VLMOE:
            return new llama_model_qwen3vlmoe(params);
        case LLM_ARCH_QWEN3TTS:
            return new llama_model_qwen3tts(params);
        case LLM_ARCH_POCKETTTS:
            return new llama_model_pockettts(params);
        case LLM_ARCH_PHI2:
            return new llama_model_phi2(params);
        case LLM_ARCH_PHI3:
            return new llama_model_phi3(params);
        case LLM_ARCH_PHIMOE:
            return new llama_model_phimoe(params);
        case LLM_ARCH_PLAMO:
            return new llama_model_plamo(params);
        case LLM_ARCH_PLAMO2:
            return new llama_model_plamo2(params);
        case LLM_ARCH_PLAMO3:
            return new llama_model_plamo3(params);
        case LLM_ARCH_GPT2:
            return new llama_model_gpt2(params);
        case LLM_ARCH_CODESHELL:
            return new llama_model_codeshell(params);
        case LLM_ARCH_ORION:
            return new llama_model_orion(params);
        case LLM_ARCH_INTERNLM2:
            return new llama_model_internlm2(params);
        case LLM_ARCH_MINICPM3:
            return new llama_model_minicpm3(params);
        case LLM_ARCH_GEMMA:
            return new llama_model_gemma(params);
        case LLM_ARCH_GEMMA2:
            return new llama_model_gemma2(params);
        case LLM_ARCH_GEMMA3:
            return new llama_model_gemma3(params);
        case LLM_ARCH_GEMMA3N:
            return new llama_model_gemma3n(params);
        case LLM_ARCH_GEMMA4:
            return new llama_model_gemma4(params);
        case LLM_ARCH_GEMMA4_ASSISTANT:
            return new llama_model_gemma4_assistant(params);
        case LLM_ARCH_GEMMA_EMBEDDING:
            return new llama_model_gemma_embedding(params);
        case LLM_ARCH_STARCODER2:
            return new llama_model_starcoder2(params);
        case LLM_ARCH_MAMBA:
            return new llama_model_mamba(params);
        case LLM_ARCH_MAMBA2:
            return new llama_model_mamba2(params);
        case LLM_ARCH_JAMBA:
            return new llama_model_jamba(params);
        case LLM_ARCH_XVERSE:
            return new llama_model_xverse(params);
        case LLM_ARCH_COMMAND_R:
            return new llama_model_command_r(params);
        case LLM_ARCH_COHERE2:
            return new llama_model_cohere2(params);
        case LLM_ARCH_COHERE2MOE:
            return new llama_model_cohere2moe(params);
        case LLM_ARCH_DBRX:
            return new llama_model_dbrx(params);
        case LLM_ARCH_OLMO:
            return new llama_model_olmo(params);
        case LLM_ARCH_OLMO2:
            return new llama_model_olmo2(params);
        case LLM_ARCH_OLMOE:
            return new llama_model_olmoe(params);
        case LLM_ARCH_MUSE_GLIMMER:
            return new llama_model_muse_glimmer(params);
        case LLM_ARCH_OPENELM:
            return new llama_model_openelm(params);
        case LLM_ARCH_GPTNEOX:
            return new llama_model_gptneox(params);
        case LLM_ARCH_ARCTIC:
            return new llama_model_arctic(params);
        case LLM_ARCH_DEEPSEEK:
            return new llama_model_deepseek(params);
        case LLM_ARCH_DEEPSEEK2:
            return new llama_model_deepseek2(params);
        case LLM_ARCH_DEEPSEEK2OCR:
            return new llama_model_deepseek2ocr(params);
        case LLM_ARCH_DEEPSEEK32:
            return new llama_model_deepseek32(params);
        case LLM_ARCH_DEEPSEEK4:
            return new llama_model_deepseek4(params);
        case LLM_ARCH_GLM_DSA:
            return new llama_model_glm_dsa(params);
        case LLM_ARCH_MISTRAL4:
            return new llama_model_mistral4(params);
        case LLM_ARCH_CHATGLM:
            return new llama_model_chatglm(params);
        case LLM_ARCH_GLM4:
            return new llama_model_glm4(params);
        case LLM_ARCH_GLM4_MOE:
            return new llama_model_glm4_moe(params);
        case LLM_ARCH_BITNET:
            return new llama_model_bitnet(params);
        case LLM_ARCH_T5:
            return new llama_model_t5(params);
        case LLM_ARCH_T5ENCODER:
            return new llama_model_t5encoder(params);
        case LLM_ARCH_JAIS:
            return new llama_model_jais(params);
        case LLM_ARCH_JAIS2:
            return new llama_model_jais2(params);
        case LLM_ARCH_NEMOTRON:
            return new llama_model_nemotron(params);
        case LLM_ARCH_NEMOTRON_H:
            return new llama_model_nemotron_h(params);
        case LLM_ARCH_NEMOTRON_H_MOE:
            return new llama_model_nemotron_h_moe(params);
        case LLM_ARCH_EXAONE:
            return new llama_model_exaone(params);
        case LLM_ARCH_EXAONE4:
            return new llama_model_exaone4(params);
        case LLM_ARCH_EXAONE_MOE:
            return new llama_model_exaone_moe(params);
        case LLM_ARCH_RWKV6:
            return new llama_model_rwkv6(params);
        case LLM_ARCH_RWKV6QWEN2:
            return new llama_model_rwkv6qwen2(params);
        case LLM_ARCH_RWKV7:
            return new llama_model_rwkv7(params);
        case LLM_ARCH_ARWKV7:
            return new llama_model_arwkv7(params);
        case LLM_ARCH_GRANITE:
            return new llama_model_granite(params);
        case LLM_ARCH_GRANITE_MOE:
            return new llama_model_granite_moe(params);
        case LLM_ARCH_GRANITE_SWITCH:
            return new llama_model_granite_switch(params);
        case LLM_ARCH_MINICPM:
            return new llama_model_minicpm(params);
        case LLM_ARCH_GRANITE_HYBRID:
            return new llama_model_granite_hybrid(params);
        case LLM_ARCH_GRANITE_SWA:
            return new llama_model_granite_swa(params);
        case LLM_ARCH_CHAMELEON:
            return new llama_model_chameleon(params);
        case LLM_ARCH_WAVTOKENIZER_DEC:
            return new llama_model_wavtokenizer_dec(params);
        case LLM_ARCH_PLM:
            return new llama_model_plm(params);
        case LLM_ARCH_BAILINGMOE:
            return new llama_model_bailingmoe(params);
        case LLM_ARCH_BAILINGMOE2:
            return new llama_model_bailingmoe2(params);
        case LLM_ARCH_BAILINGMOE3:
            return new llama_model_bailingmoe3(params);
        case LLM_ARCH_SEED_OSS:
            return new llama_model_seed_oss(params);
        case LLM_ARCH_DOTS1:
            return new llama_model_dots1(params);
        case LLM_ARCH_ARCEE:
            return new llama_model_arcee(params);
        case LLM_ARCH_AFMOE:
            return new llama_model_afmoe(params);
        case LLM_ARCH_LAGUNA:
            return new llama_model_laguna(params);
        case LLM_ARCH_ERNIE4_5:
            return new llama_model_ernie4_5(params);
        case LLM_ARCH_ERNIE4_5_MOE:
            return new llama_model_ernie4_5_moe(params);
        case LLM_ARCH_PADDLEOCR:
            return new llama_model_paddleocr(params);
        case LLM_ARCH_HUNYUAN_MOE:
            return new llama_model_hunyuan_moe(params);
        case LLM_ARCH_HUNYUAN_VL:
            return new llama_model_hunyuan_vl(params);
        case LLM_ARCH_HUNYUAN_DENSE:
            return new llama_model_hunyuan_dense(params);
        case LLM_ARCH_HY_V3:
            return new llama_model_hy_v3(params);
        case LLM_ARCH_SMOLLM3:
            return new llama_model_smollm3(params);
        case LLM_ARCH_OPENAI_MOE:
            return new llama_model_openai_moe(params);
        case LLM_ARCH_FALCON_H1:
            return new llama_model_falcon_h1(params);
        case LLM_ARCH_LFM2:
            return new llama_model_lfm2(params);
        case LLM_ARCH_LFM2MOE:
            return new llama_model_lfm2moe(params);
        case LLM_ARCH_SMALLTHINKER:
            return new llama_model_smallthinker(params);
        case LLM_ARCH_GROVEMOE:
            return new llama_model_grovemoe(params);
        case LLM_ARCH_APERTUS:
            return new llama_model_apertus(params);
        case LLM_ARCH_MINIMAX_01:
            return new llama_model_minimax_01(params);
        case LLM_ARCH_MINIMAX_M2:
            return new llama_model_minimax_m2(params);
        case LLM_ARCH_MINIMAX_M3:
            return new llama_model_minimax_m3(params);
        case LLM_ARCH_COGVLM:
            return new llama_model_cogvlm(params);
        case LLM_ARCH_PANGU_EMBED:
            return new llama_model_pangu_embed(params);
        case LLM_ARCH_QWEN3NEXT:
            return new llama_model_qwen3next(params);
        case LLM_ARCH_QWEN35:
            return new llama_model_qwen35(params);
        case LLM_ARCH_QWEN35MOE:
            return new llama_model_qwen35moe(params);
        case LLM_ARCH_MISTRAL3:
            return new llama_model_mistral3(params);
        case LLM_ARCH_EAGLE3:
            return new llama_model_eagle3(params);
        case LLM_ARCH_DFLASH:
            return new llama_model_dflash(params);
        case LLM_ARCH_MIMO2:
            return new llama_model_mimo2(params);
        case LLM_ARCH_KIMI_LINEAR:
            return new llama_model_kimi_linear(params);
        case LLM_ARCH_KIMI_K3:
            return new llama_model_kimi_k3(params);
        case LLM_ARCH_STEP35:
            return new llama_model_step35(params);
        default:
            throw std::runtime_error(std::string("unsupported model architecture: '") + llm_arch_name(arch) + "'");
    }

}

llama_model * llama_model_create(llm_arch arch, const llama_model_params & params) {
    llama_model * model = llama_model_mapping(arch, params);

    if (model != nullptr) {
        model->arch = arch;
        if (params.split_mode == LLAMA_SPLIT_MODE_TENSOR && !llm_arch_supports_sm_tensor(arch)) {
            throw std::runtime_error(std::string("LLAMA_SPLIT_MODE_TENSOR not implemented for architecture '") + llm_arch_name(arch) + "'");
        }
    }

    return model;
}

llama_model * llama_model_create(llama_model_loader & ml, const llama_model_params & params) {
    llm_arch arch = ml.get_arch();
    if (arch == LLM_ARCH_UNKNOWN) {
        throw std::runtime_error("unknown model architecture: '" + ml.get_arch_name() + "'");
    }

    return llama_model_create(arch, params);
}

struct ggml_backend_meta_split_state llama_meta_device_get_split_state(const struct ggml_tensor * tensor, void * userdata) {
    const llama_meta_device_get_split_state_userdata * ud = (const llama_meta_device_get_split_state_userdata *) userdata;
    const llama_hparams & hparams = ud->model->hparams;
    const std::string tensor_name = tensor->name;

    static const std::regex pattern_q_weight        ("blk\\.\\d*\\.attn_q.weight");
    static const std::regex pattern_kv_weight       ("blk\\.\\d*\\.attn_(k|v).weight");
    static const std::regex pattern_qkv_weight      ("blk\\.\\d*\\.attn_qkv.weight");
    static const std::regex pattern_q_bias          ("blk\\.\\d*\\.attn_q\\.bias");
    static const std::regex pattern_kv_bias         ("blk\\.\\d*\\.attn_(k|v)\\.bias");
    static const std::regex pattern_qkv_bias        ("blk\\.\\d*\\.attn_qkv.bias");
    static const std::regex pattern_qk_norm         ("blk\\.\\d*\\.attn_(q|k)_norm\\.weight");
    static const std::regex pattern_kv_cache        ("cache_(k|v)_l\\d*");
    static const std::regex pattern_attn_sinks      ("blk\\.\\d*\\.attn_sinks.weight");
    static const std::regex pattern_attn_out_weight ("blk\\.\\d*\\.attn_output.weight");
    static const std::regex pattern_attn_out_bias   ("blk\\.\\d*\\.attn_output.bias");
    static const std::regex pattern_attn_gate_weight("blk\\.\\d*\\.attn_gate.weight");

    // DeepSeek-V4 MLA: low-rank Q up-projection and grouped o_proj LoRA
    static const std::regex pattern_ds4_attn_q_b_weight  ("blk\\.\\d*\\.attn_q_b.weight");
    static const std::regex pattern_ds4_attn_out_a_weight("blk\\.\\d*\\.attn_output_a.weight");
    static const std::regex pattern_ds4_attn_out_b_weight("blk\\.\\d*\\.attn_output_b.weight");

    static const std::regex pattern_ssm_dt          ("blk\\.\\d*\\.ssm_dt.bias");
    static const std::regex pattern_ssm_a           ("blk\\.\\d*\\.ssm_a");
    static const std::regex pattern_ssm_alpha       ("blk\\.\\d*\\.ssm_alpha.weight");
    static const std::regex pattern_ssm_beta        ("blk\\.\\d*\\.ssm_beta.weight");
    static const std::regex pattern_ssm_beta_alpha  ("blk\\.\\d*\\.ssm_ba.weight");
    static const std::regex pattern_r_cache         ("cache_r_l\\d*");
    static const std::regex pattern_s_cache         ("cache_s_l\\d*");
    static const std::regex pattern_ssm_conv1d      ("blk\\.\\d*\\.ssm_conv1d.weight");
    static const std::regex pattern_ssm_out_weight  ("blk\\.\\d*\\.ssm_out.weight");

    static const std::regex pattern_ffn_up_weight     ("blk\\.\\d*\\.ffn_up(_exps)?.weight");
    static const std::regex pattern_ffn_up_bias       ("blk\\.\\d*\\.ffn_up(_exps)?.bias");
    static const std::regex pattern_ffn_gate_weight   ("blk\\.\\d*\\.ffn_gate(_exps)?.weight");
    static const std::regex pattern_ffn_gate_bias     ("blk\\.\\d*\\.ffn_gate(_exps)?.bias");
    static const std::regex pattern_ffn_gate_up_weight("blk\\.\\d*\\.ffn_gate_up(_exps)?.weight");
    static const std::regex pattern_ffn_down_weight   ("blk\\.\\d*\\.ffn_down(_exps)?.weight");
    static const std::regex pattern_ffn_down_bias     ("blk\\.\\d*\\.ffn_down.bias");
    static const std::regex pattern_ffn_down_exps_bias("blk\\.\\d*\\.ffn_down_exps.bias");

    static const std::regex pattern_output_weight("output\\.weight");
    static const std::regex pattern_output_bias  ("output\\.bias");

    struct tensor_config {
        ggml_backend_meta_split_axis axis;

        const ggml_tensor * tensor_axis_0;

        uint32_t il;
        size_t   rotation; // when assigning tensor slices, rotate how the rounding is done for more even allocation
    };

    auto get_tensor_config_impl = [&](
                const ggml_backend_meta_split_axis axis, const std::string & suffix = "", const std::string & suffix_fallback = "") -> tensor_config {
        // the layers in a tensor can be inhomogeneous, if the pattern is cleanly divided by the number of GPUs there can be aliasing effects,
        //     count only the same type of previous layers to avoid this
        auto get_il_eff = [&](const size_t il){
            size_t ret = 0;
            const bool il_is_recr = hparams.is_recr(il);
            const bool il_is_swa  = hparams.is_swa(il);
            for (size_t il_prev = 0; il_prev < il; il_prev++) {
                ret += hparams.is_recr(il_prev) == il_is_recr && hparams.is_swa(il_prev) == il_is_swa;
            }
            return ret;
        };

        uint32_t il;
        std::string prefix;
        size_t rotation;
        if (tensor_name.substr(0, 4) == "blk.") {
            const size_t length_prefix = tensor_name.find('.', 4);
            GGML_ASSERT(length_prefix != std::string::npos);
            prefix = tensor_name.substr(0, length_prefix + 1);
            il = std::stoull(tensor_name.substr(4, length_prefix));
            rotation = get_il_eff(il) % ud->n_devices;
        } else if (tensor_name.substr(0, 6) == "cache_") {
            const size_t layer_index_start = tensor_name.find("_l", 6);
            GGML_ASSERT(layer_index_start != std::string::npos);
            il = std::stoull(tensor_name.substr(layer_index_start + 2));
            prefix = "blk." + std::to_string(il) + ".";
            rotation = get_il_eff(il) % ud->n_devices;
        } else {
            il = 0;
            rotation = hparams.n_layer() % ud->n_devices;
        }
        const ggml_tensor * tensor_axis_0 = suffix.empty() ? tensor : ud->model->get_tensor((prefix + suffix).c_str());
        if (tensor_axis_0 == nullptr) {
            GGML_ASSERT(!suffix_fallback.empty());
            tensor_axis_0 = ud->model->get_tensor((prefix + suffix_fallback).c_str());
        }
        GGML_ASSERT(tensor_axis_0 != nullptr);
        return {axis, tensor_axis_0, il, rotation};
    };

    auto get_tensor_config = [&]() -> tensor_config {
        // DeepSeek-V4: MLA with a single latent KV head shared by all Q heads, plus a grouped
        // o_proj LoRA (attn_output_a: {n_head*n_embd_head/o_groups, o_lora_rank*o_groups} applied as a
        // batched per-group matmul, then attn_output_b: {o_lora_rank*o_groups, n_embd}).
        // Split the Q heads (axis 1 of attn_q_b) and the matching o_proj groups (axis 2 of the
        // reshaped attn_output_a) across devices; attn_output_b is split along its input dim (axis 0) so the
        // partial sums are combined with an allreduce. The latent KV (attn_kv) and the KV cache are
        // mirrored because every Q head attends to the full latent KV, as are the DSA
        // compressor/indexer, hyper-connection, and routing tensors (default below).
        if (ud->model->arch == LLM_ARCH_DEEPSEEK4) {
            if (std::regex_match(tensor_name, pattern_ds4_attn_q_b_weight)) {
                return get_tensor_config_impl(GGML_BACKEND_SPLIT_AXIS_1, "attn_output_a.weight");
            }
            if (std::regex_match(tensor_name, pattern_ds4_attn_out_a_weight)) {
                return get_tensor_config_impl(GGML_BACKEND_SPLIT_AXIS_2, "attn_output_b.weight");
            }
            if (std::regex_match(tensor_name, pattern_ds4_attn_out_b_weight)) {
                return get_tensor_config_impl(GGML_BACKEND_SPLIT_AXIS_0, "attn_output_a.weight");
            }
            if (std::regex_match(tensor_name, pattern_attn_sinks)) {
                return get_tensor_config_impl(GGML_BACKEND_SPLIT_AXIS_0);
            }
            if (std::regex_match(tensor_name, pattern_kv_cache)) {
                return get_tensor_config_impl(GGML_BACKEND_SPLIT_AXIS_MIRRORED);
            }
        }

        // standard attention
        if (std::regex_match(tensor_name, pattern_q_weight) || std::regex_match(tensor_name, pattern_kv_weight)) {
            return get_tensor_config_impl(GGML_BACKEND_SPLIT_AXIS_1, "attn_output.weight", "ssm_out.weight");
        }
        if (std::regex_match(tensor_name, pattern_q_bias) || std::regex_match(tensor_name, pattern_kv_bias)) {
            return get_tensor_config_impl(GGML_BACKEND_SPLIT_AXIS_0, "attn_output.weight", "ssm_out.weight");
        }
        if (std::regex_match(tensor_name, pattern_qkv_weight)) {
            return get_tensor_config_impl(GGML_BACKEND_SPLIT_AXIS_1, "attn_output.weight", "ssm_out.weight");
        }
        if ( std::regex_match(tensor_name, pattern_qkv_bias)) {
            return get_tensor_config_impl(GGML_BACKEND_SPLIT_AXIS_0, "attn_output.weight", "ssm_out.weight");
        }
        if (std::regex_match(tensor_name, pattern_qk_norm)) {
            return get_tensor_config_impl(tensor->ne[1] == 1 ? GGML_BACKEND_SPLIT_AXIS_MIRRORED : GGML_BACKEND_SPLIT_AXIS_1, "attn_output.weight");
        }
        if (std::regex_match(tensor_name, pattern_kv_cache) || std::regex_match(tensor_name, pattern_attn_sinks)) {
            return get_tensor_config_impl(GGML_BACKEND_SPLIT_AXIS_0, "attn_output.weight");
        }
        if (std::regex_match(tensor_name, pattern_attn_out_weight)) {
            return get_tensor_config_impl(GGML_BACKEND_SPLIT_AXIS_0);
        }
        if (std::regex_match(tensor_name, pattern_attn_out_bias)) {
            return get_tensor_config_impl(GGML_BACKEND_SPLIT_AXIS_MIRRORED);
        }

        if (std::regex_match(tensor_name, pattern_attn_gate_weight)) {
            return get_tensor_config_impl(GGML_BACKEND_SPLIT_AXIS_1, "attn_output.weight", "ssm_out.weight");
        }
        if (std::regex_match(tensor_name, pattern_ssm_dt) || std::regex_match(tensor_name, pattern_ssm_a)) {
            return get_tensor_config_impl(GGML_BACKEND_SPLIT_AXIS_0, "ssm_out.weight");
        }
        if (std::regex_match(tensor_name, pattern_ssm_alpha) || std::regex_match(tensor_name, pattern_ssm_beta) ||
                std::regex_match(tensor_name, pattern_ssm_beta_alpha)) {
            return get_tensor_config_impl(GGML_BACKEND_SPLIT_AXIS_1, "ssm_out.weight");
        }
        if (std::regex_match(tensor_name, pattern_r_cache) || std::regex_match(tensor_name, pattern_s_cache)) {
            return get_tensor_config_impl(GGML_BACKEND_SPLIT_AXIS_0, "ssm_out.weight");
        }
        if (std::regex_match(tensor_name, pattern_ssm_conv1d)) {
            return get_tensor_config_impl(GGML_BACKEND_SPLIT_AXIS_1, "ssm_out.weight");
        }
        if (std::regex_match(tensor_name, pattern_ssm_out_weight)) {
            return get_tensor_config_impl(GGML_BACKEND_SPLIT_AXIS_0);
        }

        // FFN
        if (std::regex_match(tensor_name, pattern_ffn_up_weight) || std::regex_match(tensor_name, pattern_ffn_gate_weight)) {
            return get_tensor_config_impl(GGML_BACKEND_SPLIT_AXIS_1, "ffn_down.weight", "ffn_down_exps.weight");
        }
        if (std::regex_match(tensor_name, pattern_ffn_up_bias) || std::regex_match(tensor_name, pattern_ffn_gate_bias)) {
            return get_tensor_config_impl(GGML_BACKEND_SPLIT_AXIS_0, "ffn_down.weight", "ffn_down_exps.weight");
        }
        if (std::regex_match(tensor_name, pattern_ffn_gate_up_weight)) {
            return get_tensor_config_impl(GGML_BACKEND_SPLIT_AXIS_1, "ffn_down.weight", "ffn_down_exps.weight");
        }
        if (std::regex_match(tensor_name, pattern_ffn_down_weight)) {
            return get_tensor_config_impl(GGML_BACKEND_SPLIT_AXIS_0, "ffn_down.weight", "ffn_down_exps.weight");
        }
        if (std::regex_match(tensor_name, pattern_ffn_down_bias)) {
            return get_tensor_config_impl(GGML_BACKEND_SPLIT_AXIS_MIRRORED);
        }
        if (std::regex_match(tensor_name, pattern_ffn_down_exps_bias)) {
            return get_tensor_config_impl(GGML_BACKEND_SPLIT_AXIS_PARTIAL);
        }

        // output
        if (std::regex_match(tensor_name, pattern_output_weight)) {
            return get_tensor_config_impl(GGML_BACKEND_SPLIT_AXIS_1);
        }
        if (std::regex_match(tensor_name, pattern_output_bias)) {
            const ggml_tensor * output_weight = ud->model->get_tensor("output.weight");
            GGML_ASSERT(output_weight != nullptr);
            return get_tensor_config_impl(GGML_BACKEND_SPLIT_AXIS_0);
        }

        // everything else
        return get_tensor_config_impl(GGML_BACKEND_SPLIT_AXIS_MIRRORED);
    };

    auto get_split_segments = [&](int axis, uint32_t il) -> std::vector<std::pair<int64_t, uint32_t>> {
        if (ud->model->arch == LLM_ARCH_QWEN3NEXT || ud->model->arch == LLM_ARCH_QWEN35 || ud->model->arch == LLM_ARCH_QWEN35MOE) {
            const int64_t head_k_dim = hparams.ssm_d_state;
            const int64_t head_v_dim = hparams.ssm_d_state;
            const int64_t n_k_heads  = hparams.ssm_n_group;
            const int64_t n_v_heads  = hparams.ssm_dt_rank;
            const int64_t key_dim    = head_k_dim * n_k_heads;
            const int64_t value_dim  = head_v_dim * n_v_heads;

            // both Qwen 3 Next and Qwen 3.5 support n_v_heads > n_k_heads but the broadcasting pattern is different:
            //   - Qwen 3 Next: [k0_v0, k0_v1, k1_v2, k1_v3] (this is the default split pattern)
            //   - Qwen 3.5:    [k0_v0, k1_v1, k0_v2, k1_v3] (needs segmenting of V on the scale of K to get the correct pattern)
            if (ud->model->arch == LLM_ARCH_QWEN3NEXT) {
                if (std::regex_match(tensor_name, pattern_qkv_weight) || std::regex_match(tensor_name, pattern_ssm_conv1d)) {
                    GGML_ASSERT(tensor->ne[axis] == 2*key_dim + value_dim);
                    return {{key_dim, 2}, {value_dim, 1}};
                }
            } else {
                const int64_t head_ratio = n_v_heads / n_k_heads;
                if (std::regex_match(tensor_name, pattern_qkv_weight) || std::regex_match(tensor_name, pattern_ssm_conv1d)) {
                    GGML_ASSERT(tensor->ne[axis] == 2*key_dim + value_dim);
                    return {{key_dim, 2 + head_ratio}};
                }
                if (std::regex_match(tensor_name, pattern_attn_gate_weight) || std::regex_match(tensor_name, pattern_ssm_out_weight)) {
                    return {{key_dim, head_ratio}};
                }
                if (std::regex_match(tensor_name, pattern_ssm_dt) || std::regex_match(tensor_name, pattern_ssm_a) ||
                        std::regex_match(tensor_name, pattern_ssm_alpha) || std::regex_match(tensor_name, pattern_ssm_beta)) {
                    return {{n_k_heads, head_ratio}};
                }
                if (std::regex_match(tensor_name, pattern_r_cache)) {
                    return {{key_dim * (hparams.ssm_d_conv - 1), 2 + head_ratio}};
                }
                if (std::regex_match(tensor_name, pattern_s_cache)) {
                    return {{n_k_heads * head_v_dim * head_v_dim, head_ratio}};
                }
            }

            // the FFN is the same for Qwen 3 Next and Qwen 3.5:
            if (std::regex_match(tensor_name, pattern_ffn_gate_up_weight)) {
                const int64_t n_ff_exp = hparams.n_ff_exp;
                GGML_ASSERT(tensor->ne[axis] == 2*n_ff_exp);
                return {{n_ff_exp, 2}};
            }
            return {{tensor->ne[axis], 1}};
        }

        if (std::regex_match(tensor_name, pattern_qkv_weight) || std::regex_match(tensor_name, pattern_qkv_bias)) {
            const int64_t n_embd      = hparams.n_embd;
            const int64_t n_embd_gqa  = hparams.n_embd_v_gqa(il);
            GGML_ASSERT(hparams.n_embd_k_gqa() == n_embd_gqa);
            GGML_ASSERT(tensor->ne[axis] == n_embd + 2*n_embd_gqa);
            return {{n_embd, 1}, {n_embd_gqa, 2}};
        }
        if (std::regex_match(tensor_name, pattern_ffn_up_weight) || std::regex_match(tensor_name, pattern_ffn_up_bias)) {
            const int64_t n_ff = hparams.n_ff(il);
            // some models such as Phi 3 have fused up + gate tensors named "up" tensors, which need to be segmented
            if (tensor->ne[axis] == 2*n_ff) {
                return {{n_ff, 2}};
            }
            return {{tensor->ne[axis], 1}};
        }
        if (std::regex_match(tensor_name, pattern_ffn_gate_up_weight)) {
            const int64_t n_ff_exp = hparams.n_ff_exp;
            GGML_ASSERT(tensor->ne[axis] == 2*n_ff_exp);
            return {{n_ff_exp, 2}};
        }
        return {{tensor->ne[axis], 1}};
    };

    auto get_split_granularity = [&](int64_t blck_size, uint32_t il, const std::vector<std::pair<int64_t, uint32_t>> & segments) -> std::vector<int64_t> {
        if (ud->model->arch == LLM_ARCH_DEEPSEEK4) {
            // keep the Q head split aligned with the o_proj groups: each device must own whole groups
            const int64_t n_heads_group = hparams.n_head(il)/hparams.dsv4_o_group_count;
            const int64_t o_group_dim   = n_heads_group*hparams.n_embd_head_k(il);
            if (std::regex_match(tensor_name, pattern_ds4_attn_q_b_weight)) {
                GGML_ASSERT(segments.size() == 1);
                return {std::lcm(blck_size, o_group_dim)};
            }
            if (std::regex_match(tensor_name, pattern_ds4_attn_out_a_weight)) {
                GGML_ASSERT(segments.size() == 1);
                return {1};
            }
            if (std::regex_match(tensor_name, pattern_ds4_attn_out_b_weight)) {
                GGML_ASSERT(segments.size() == 1);
                return {std::lcm(blck_size, int64_t(hparams.dsv4_o_lora_rank))};
            }
            if (std::regex_match(tensor_name, pattern_attn_sinks)) {
                GGML_ASSERT(segments.size() == 1);
                return {n_heads_group};
            }
        }

        // for better performance it may make sense to round up blck_size to a higher power of 2 so that more efficient kernels can be used
        if (hparams.is_recr(il)) {
            // linear attention
            const int64_t head_dim        = hparams.ssm_d_state;
            const int64_t blck_size_perf  = std::lcm(blck_size, 128);
            const int64_t granularity_qkv = std::lcm(blck_size_perf, head_dim);
            if (std::regex_match(tensor_name, pattern_qkv_weight) || std::regex_match(tensor_name, pattern_attn_gate_weight) ||
                    std::regex_match(tensor_name, pattern_ssm_conv1d) || std::regex_match(tensor_name, pattern_ssm_out_weight)) {
                return std::vector<int64_t>(segments.size(), granularity_qkv);
            }
            if (std::regex_match(tensor_name, pattern_ssm_dt) || std::regex_match(tensor_name, pattern_ssm_a) ||
                    std::regex_match(tensor_name, pattern_ssm_alpha) || std::regex_match(tensor_name, pattern_ssm_beta)) {
                return std::vector<int64_t>(segments.size(), granularity_qkv / head_dim);
            }
            if (std::regex_match(tensor_name, pattern_ssm_beta_alpha)) {
                return std::vector<int64_t>(segments.size(), 2 * (granularity_qkv / head_dim));
            }
            if (std::regex_match(tensor_name, pattern_r_cache)) {
                return std::vector<int64_t>(segments.size(), granularity_qkv * (hparams.ssm_d_conv - 1));
            }
            if (std::regex_match(tensor_name, pattern_s_cache)) {
                return std::vector<int64_t>(segments.size(), granularity_qkv * head_dim);
            }
        } else {
            // regular attention
            const uint32_t n_gqa    = hparams.n_gqa(il);
            const uint32_t n_embd_q = n_gqa * hparams.n_embd_head_k(il);

            // to handle head sizes like 80, only increase granularity while it doesn't cause underutilization
            int64_t blck_size_perf = blck_size;
            while (blck_size_perf < 128 && blck_size_perf*ud->n_devices < n_embd_q) {
                blck_size_perf *= 2;
            }

            if (std::regex_match(tensor_name, pattern_attn_sinks)) {
                GGML_ASSERT(segments.size() == 1);
                return {std::lcm(n_embd_q, blck_size_perf)/n_embd_q * n_gqa};
            }

            const int64_t granularity_q = std::lcm(n_embd_q, blck_size_perf);
            if (std::regex_match(tensor_name, pattern_q_weight) || std::regex_match(tensor_name, pattern_q_bias)) {
                GGML_ASSERT(segments.size() == 1);
                // some models have Q gate tensors, for those cases the granularity needs to be doubled:
                if (ud->model->arch == LLM_ARCH_QWEN3NEXT || ud->model->arch == LLM_ARCH_QWEN35 || ud->model->arch == LLM_ARCH_QWEN35MOE) {
                    return {std::lcm(2*n_embd_q, blck_size_perf)};
                }
                return {granularity_q};
            }
            if (std::regex_match(tensor_name, pattern_attn_out_weight)) {
                GGML_ASSERT(segments.size() == 1);
                return {granularity_q};
            }

            const int64_t granularity_kv = granularity_q / n_gqa;
            if (std::regex_match(tensor_name, pattern_kv_weight) ||
                std::regex_match(tensor_name, pattern_kv_bias) ||
                std::regex_match(tensor_name, pattern_kv_cache)) {
                GGML_ASSERT(segments.size() == 1);
                return {granularity_kv};
            }
            if (std::regex_match(tensor_name, pattern_qkv_weight) || std::regex_match(tensor_name, pattern_qkv_bias)) {
                GGML_ASSERT(segments.size() == 2);
                return {granularity_q, granularity_kv};
            }
        }

        // FFN
        if (std::regex_match(tensor_name, pattern_ffn_up_weight) || std::regex_match(tensor_name, pattern_ffn_up_bias) ||
                std::regex_match(tensor_name, pattern_ffn_gate_weight) || std::regex_match(tensor_name, pattern_ffn_gate_bias) ||
                std::regex_match(tensor_name, pattern_ffn_gate_up_weight) || std::regex_match(tensor_name, pattern_ffn_down_weight)) {
            const int64_t blck_size_perf = std::lcm(blck_size, 128);
            GGML_ASSERT(segments.size() == 1);
            return {blck_size_perf};
        }

        // everything else
        GGML_ASSERT(segments.size() == 1);
        return {1};
    };

    ggml_backend_meta_split_state split_state;
    memset(&split_state, 0, sizeof(split_state));
    tensor_config tc = get_tensor_config();
    split_state.axis = tc.axis;
    if (split_state.axis >= 0 && split_state.axis < GGML_MAX_DIMS) {
        const int64_t blck_size = ggml_blck_size(tc.tensor_axis_0->type);
        const float * tensor_split = ud->model->tensor_split();
        const std::vector<std::pair<int64_t, uint32_t>> segments = get_split_segments(split_state.axis, tc.il);
        const std::vector<int64_t> granularity = get_split_granularity(blck_size, tc.il, segments);
        ggml_shard_plan_input plan_input;
        plan_input.placement = ggml_shard_placement::SPLIT;
        plan_input.axis = split_state.axis;
        plan_input.rotation = tc.rotation;
        plan_input.domains.resize(ud->n_devices);
        plan_input.segments.resize(segments.size());
        for (size_t i = 0; i < ud->n_devices; ++i) {
            plan_input.domains[i].id = i;
            plan_input.domains[i].capacity = tensor_split == nullptr ? 0.0 : tensor_split[i];
        }
        for (size_t is = 0; is < segments.size(); is++) {
            plan_input.segments[is] = {segments[is].first, segments[is].second, granularity[is]};
        }
        ggml_shard_plan plan;
        std::string plan_error;
        GGML_ASSERT(ggml_shard_plan_build(plan_input, plan, &plan_error));
        for (size_t is = 0; is < segments.size(); ++is) {
            for (size_t i = 0; i < ud->n_devices; ++i) {
                split_state.ne[is * ud->n_devices + i] = plan.extents[is * ud->n_devices + i];
            }
            split_state.nr[is] = plan.segments[is].repetitions;
        }
        split_state.n_segments = segments.size();
    } else {
        memset(split_state.ne, 0, sizeof(split_state.ne));
        split_state.nr[0] = 1;
        split_state.n_segments = 1;
    }
    return split_state;
    GGML_UNUSED(userdata);
}

const char * llm_type_name(llm_type type) {
    switch (type) {
        case LLM_TYPE_14M:           return "14M";
        case LLM_TYPE_17M:           return "17M";
        case LLM_TYPE_22M:           return "22M";
        case LLM_TYPE_33M:           return "33M";
        case LLM_TYPE_47M:           return "47M";
        case LLM_TYPE_60M:           return "60M";
        case LLM_TYPE_70M:           return "70M";
        case LLM_TYPE_80M:           return "80M";
        case LLM_TYPE_109M:          return "109M";
        case LLM_TYPE_137M:          return "137M";
        case LLM_TYPE_140M:          return "140M";
        case LLM_TYPE_149M:          return "149M";
        case LLM_TYPE_160M:          return "160M";
        case LLM_TYPE_190M:          return "190M";
        case LLM_TYPE_220M:          return "220M";
        case LLM_TYPE_230M:          return "230M";
        case LLM_TYPE_250M:          return "250M";
        case LLM_TYPE_256M:          return "256M";
        case LLM_TYPE_270M:          return "270M";
        case LLM_TYPE_335M:          return "335M";
        case LLM_TYPE_350M:          return "350M";
        case LLM_TYPE_360M:          return "360M";
        case LLM_TYPE_395M:          return "395M";
        case LLM_TYPE_410M:          return "410M";
        case LLM_TYPE_450M:          return "450M";
        case LLM_TYPE_475M:          return "475M";
        case LLM_TYPE_558M:          return "558M";
        case LLM_TYPE_700M:          return "700M";
        case LLM_TYPE_770M:          return "770M";
        case LLM_TYPE_780M:          return "780M";
        case LLM_TYPE_950M:          return "950M";
        case LLM_TYPE_0_3B:          return "0.3B";
        case LLM_TYPE_0_5B:          return "0.5B";
        case LLM_TYPE_0_6B:          return "0.6B";
        case LLM_TYPE_0_8B:          return "0.8B";
        case LLM_TYPE_1B:            return "1B";
        case LLM_TYPE_1_2B:          return "1.2B";
        case LLM_TYPE_1_3B:          return "1.3B";
        case LLM_TYPE_1_4B:          return "1.4B";
        case LLM_TYPE_1_5B:          return "1.5B";
        case LLM_TYPE_1_6B:          return "1.6B";
        case LLM_TYPE_1_7B:          return "1.7B";
        case LLM_TYPE_1_8B:          return "1.8B";
        case LLM_TYPE_2B:            return "2B";
        case LLM_TYPE_2_6B:          return "2.6B";
        case LLM_TYPE_2_8B:          return "2.8B";
        case LLM_TYPE_2_9B:          return "2.9B";
        case LLM_TYPE_3B:            return "3B";
        case LLM_TYPE_4B:            return "4B";
        case LLM_TYPE_6B:            return "6B";
        case LLM_TYPE_6_9B:          return "6.9B";
        case LLM_TYPE_7B:            return "7B";
        case LLM_TYPE_8B:            return "8B";
        case LLM_TYPE_9B:            return "9B";
        case LLM_TYPE_11B:           return "11B";
        case LLM_TYPE_12B:           return "12B";
        case LLM_TYPE_13B:           return "13B";
        case LLM_TYPE_14B:           return "14B";
        case LLM_TYPE_15B:           return "15B";
        case LLM_TYPE_16B:           return "16B";
        case LLM_TYPE_20B:           return "20B";
        case LLM_TYPE_26B:           return "26B";
        case LLM_TYPE_27B:           return "27B";
        case LLM_TYPE_30B:           return "30B";
        case LLM_TYPE_31B:           return "31B";
        case LLM_TYPE_32B:           return "32B";
        case LLM_TYPE_34B:           return "34B";
        case LLM_TYPE_35B:           return "35B";
        case LLM_TYPE_36B:           return "36B";
        case LLM_TYPE_40B:           return "40B";
        case LLM_TYPE_65B:           return "65B";
        case LLM_TYPE_70B:           return "70B";
        case LLM_TYPE_120B:          return "120B";
        case LLM_TYPE_142B:          return "142B";
        case LLM_TYPE_236B:          return "236B";
        case LLM_TYPE_290B:          return "290B";
        case LLM_TYPE_314B:          return "314B";
        case LLM_TYPE_405B:          return "405B";
        case LLM_TYPE_456B:          return "456B";
        case LLM_TYPE_671B:          return "671B";
        case LLM_TYPE_SMALL:         return "0.1B";
        case LLM_TYPE_MEDIUM:        return "0.4B";
        case LLM_TYPE_LARGE:         return "0.8B";
        case LLM_TYPE_XL:            return "1.5B";
        case LLM_TYPE_A1_7B:         return "A1.7B";
        case LLM_TYPE_A2_7B:         return "A2.7B";
        case LLM_TYPE_8x7B:          return "8x7B";
        case LLM_TYPE_8x22B:         return "8x22B";
        case LLM_TYPE_16x12B:        return "16x12B";
        case LLM_TYPE_16x3_8B:       return "16x3.8B";
        case LLM_TYPE_10B_128x3_66B: return "10B+128x3.66B";
        case LLM_TYPE_57B_A14B:      return "57B.A14B";
        case LLM_TYPE_17B_16E:       return "17Bx16E (Scout)";
        case LLM_TYPE_17B_128E:      return "17Bx128E (Maverick)";
        case LLM_TYPE_A13B:          return "A13B";
        case LLM_TYPE_7B_A1B:        return "7B.A1B";
        case LLM_TYPE_8B_A1B:        return "8B.A1B";
        case LLM_TYPE_7_9B_A1_3B:    return "7.9B.A1.3B";
        case LLM_TYPE_12B_A2_5B:     return "12B.A2.5B";
        case LLM_TYPE_16B_A1B:       return "16B.A1B";
        case LLM_TYPE_21B_A3B:       return "21B.A3B";
        case LLM_TYPE_24B_A2B:       return "24B.A2B";
        case LLM_TYPE_26B_A4B:       return "26B.A4B";
        case LLM_TYPE_30B_A3B:       return "30B.A3B";
        case LLM_TYPE_31B_A3_5B:     return "31B.A3.5B";
        case LLM_TYPE_35B_A3B:       return "35B.A3B";
        case LLM_TYPE_48B_A3B:       return "48B.A3B";
        case LLM_TYPE_80B_A3B:       return "80B.A3B";
        case LLM_TYPE_100B_A6B:      return "100B.A6B";
        case LLM_TYPE_102B_A12B:     return "102B.A12B";
        case LLM_TYPE_106B_A12B:     return "106B.A12B";
        case LLM_TYPE_118B_A8B:      return "118B.A8B";
        case LLM_TYPE_120B_A12B:     return "120B.A12B";
        case LLM_TYPE_122B_A10B:     return "122B.A10B";
        case LLM_TYPE_124B_A5_1B:    return "124B.A5.1B";
        case LLM_TYPE_196B_A11B:     return "196B.A11B";
        case LLM_TYPE_230B_A10B:     return "230B.A10B";
        case LLM_TYPE_428B_A23B:     return "428B.A23B";
        case LLM_TYPE_235B_A22B:     return "235B.A22B";
        case LLM_TYPE_300B_A47B:     return "300B.A47B";
        case LLM_TYPE_310B_A15B:     return "310B.A15B";
        case LLM_TYPE_355B_A32B:     return "355B.A32B";
        case LLM_TYPE_397B_A17B:     return "397B.A17B";
        case LLM_TYPE_685B_A37B:     return "685B.A37B";
        case LLM_TYPE_744B_A40B:     return "744B.A40B";
        case LLM_TYPE_2_8T_A50B:     return "2.8T.A50B";
        case LLM_TYPE_E2B:           return "E2B";
        case LLM_TYPE_E4B:           return "E4B";
        default:                     return "?B";
    }
}

static const char * llama_expert_gating_func_name(llama_expert_gating_func_type type) {
    switch (type) {
        case LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX: return "softmax";
        case LLAMA_EXPERT_GATING_FUNC_TYPE_SIGMOID: return "sigmoid";
        case LLAMA_EXPERT_GATING_FUNC_TYPE_SQRT_SOFTPLUS: return "sqrtsoftplus";
        default:                                    return "unknown";
    }
}

static const std::map<llama_rope_scaling_type, const char *> LLAMA_ROPE_SCALING_TYPES = {
    { LLAMA_ROPE_SCALING_TYPE_NONE,       "none"       },
    { LLAMA_ROPE_SCALING_TYPE_LINEAR,     "linear"     },
    { LLAMA_ROPE_SCALING_TYPE_YARN,       "yarn"       },
    { LLAMA_ROPE_SCALING_TYPE_LONGROPE,   "longrope"   },
};

std::string llama_rope_scaling_type_name(llama_rope_scaling_type rope_scaling_type) {
    return LLAMA_ROPE_SCALING_TYPES.at(rope_scaling_type);
}

static llama_rope_scaling_type llama_rope_scaling_type_from_string(const std::string & name) {
    for (const auto & kv : LLAMA_ROPE_SCALING_TYPES) {
        if (kv.second == name) {
            return (llama_rope_scaling_type) kv.first;
        }
    }

    return LLAMA_ROPE_SCALING_TYPE_UNSPECIFIED;
}

// Maps the GGUF `<arch>.hidden_activation` string to the FFN op type used by the
// graph builders. Only gated activations that map cleanly to llm_ffn_op_type are
// listed; unrecognized values fall back to GeGLU, which matches the historical
// default for ModernBert-style architectures.
static const std::map<std::string, llm_ffn_op_type> LLM_FFN_OP_TYPES_FROM_STRING = {
    { "gelu",   LLM_FFN_GEGLU  },
    { "geglu",  LLM_FFN_GEGLU  },
    { "silu",   LLM_FFN_SWIGLU },
    { "swish",  LLM_FFN_SWIGLU },
    { "swiglu", LLM_FFN_SWIGLU },
    { "relu",   LLM_FFN_RELU   },
    { "reglu",  LLM_FFN_REGLU  },
};

llm_ffn_op_type llm_ffn_op_type_from_string(const std::string & name, llm_ffn_op_type fallback) {
    const auto it = LLM_FFN_OP_TYPES_FROM_STRING.find(name);
    if (it != LLM_FFN_OP_TYPES_FROM_STRING.end()) {
        return it->second;
    }
    return fallback;
}

// CPU: ACCEL -> GPU host -> CPU extra -> CPU
static buft_list_t make_cpu_buft_list(const std::vector<llama_device> & devices, bool use_extra_bufts, bool no_host) {
    buft_list_t buft_list;

    // add ACCEL buffer types
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_ACCEL) {
            auto * buft = ggml_backend_dev_buffer_type(dev);
            // skip
            if (buft != ggml_backend_cpu_buffer_type()) {
                buft_list.emplace_back(dev, buft);
            }
        }
    }

    // add extra buffer types
    // NOTE: these must come BEFORE the host buffer types below. Extra bufts
    // (e.g. CPU_REPACK) store weights in a compute-optimized layout for the CPU
    // backend, while the (pinned) host buft stores them verbatim. Any weight
    // that lands in a host buffer is always computed by the CPU backend (GPU
    // only ever computes weights in its own device buffers), so a matmul-eligible
    // weight in the host buft would run the slow generic vec_dot path for no
    // benefit. With CUDA builds this ordering is the difference between the
    // repack gemv kernels and a ~2x slower CPU fallback.
    if (use_extra_bufts) {
        auto * cpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
        if (cpu_dev == nullptr) {
            throw std::runtime_error(format("%s: no CPU backend found", __func__));
        }

        auto * cpu_reg = ggml_backend_dev_backend_reg(cpu_dev);
        auto ggml_backend_dev_get_extra_bufts_fn = (ggml_backend_dev_get_extra_bufts_t)
            ggml_backend_reg_get_proc_address(cpu_reg, "ggml_backend_dev_get_extra_bufts");
        if (ggml_backend_dev_get_extra_bufts_fn) {
            ggml_backend_buffer_type_t * extra_bufts = ggml_backend_dev_get_extra_bufts_fn(cpu_dev);
            while (extra_bufts && *extra_bufts) {
                buft_list.emplace_back(cpu_dev, *extra_bufts);
                ++extra_bufts;
            }
        }
    }

    // add a host buffer type
    // storing the tensors in a host buffer is useful when the processing of large batches
    // is offloaded to a GPU device, since it reduces the time spent on data transfers
    // generally, this will be done using the first device in the list
    // a better approach would be to handle this on a weight-by-weight basis using the offload_op
    // function of the device to determine if it would benefit from being stored in a host buffer
    if (!no_host) {
        for (const auto & dev : devices) {
            ggml_backend_buffer_type_t buft = ggml_backend_dev_host_buffer_type(dev.dev);
            if (buft) {
                buft_list.emplace_back(dev.dev, buft);
                break;
            }
        }
    }

    // add the CPU buffer type
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_CPU) {
            buft_list.emplace_back(dev, ggml_backend_dev_buffer_type(dev));
        }
    }

    return buft_list;
}

// GPU: split if LLAMA_SPLIT_MODE_ROW -> GPU
static buft_list_t make_gpu_buft_list(ggml_backend_dev_t dev, llama_split_mode split_mode, const float * tensor_split) {
    buft_list_t buft_list;

    // add the device split buffer type if requested and available
    if (split_mode == LLAMA_SPLIT_MODE_ROW) {
        ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
        auto ggml_backend_split_buffer_type_fn = (ggml_backend_split_buffer_type_t)
            ggml_backend_reg_get_proc_address(reg, "ggml_backend_split_buffer_type");
        if (ggml_backend_split_buffer_type_fn) {
            size_t dev_index = [&]() {
                auto * reg = ggml_backend_dev_backend_reg(dev);
                for (size_t i = 0; i < ggml_backend_reg_dev_count(reg); ++i) {
                    if (ggml_backend_reg_dev_get(reg, i) == dev) {
                        return i;
                    }
                }
                throw std::runtime_error(format("device %s not found in its backend reg", ggml_backend_dev_name(dev)));
            }();
            auto * buft = ggml_backend_split_buffer_type_fn(dev_index, tensor_split);
            if (buft != nullptr) {
                buft_list.emplace_back(dev, buft);
            }
        } else {
            throw std::runtime_error(format("device %s does not support split buffers", ggml_backend_dev_name(dev)));
        }
    }

    // add the device default buffer type
    buft_list.emplace_back(dev, ggml_backend_dev_buffer_type(dev));

    // add the device extra buffer type (if any)
    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
    if (reg) {
        auto ggml_backend_dev_get_extra_bufts_fn = (ggml_backend_dev_get_extra_bufts_t)
            ggml_backend_reg_get_proc_address(reg, "ggml_backend_dev_get_extra_bufts");

        if (ggml_backend_dev_get_extra_bufts_fn) {
            ggml_backend_buffer_type_t * extra_bufts = ggml_backend_dev_get_extra_bufts_fn(dev);
            while (extra_bufts && *extra_bufts) {
                buft_list.emplace_back(dev, *extra_bufts);
                ++extra_bufts;
            }
        }
    }

    return buft_list;
}

struct llama_model::impl {
    impl() = default;
    ~impl();

    uint64_t n_elements = 0;

    size_t n_bytes = 0;

    std::string desc_str;

    llama_ftype ftype = LLAMA_FTYPE_ALL_F32;

    // model memory mapped files
    llama_mmaps mappings;

    // objects representing data potentially being locked in memory
    llama_mlocks mlock_bufs;
    llama_mlocks mlock_mmaps;

    // contexts where the model tensors metadata is stored as well as the corresponding buffers:
    std::vector<std::pair<ggml_context_ptr, std::vector<ggml_backend_buffer_ptr>>> ctxs_bufs;

    // NUMA mirror: per-(host weight buffer) node-local copies. node_base[0] aliases the original
    // buffer; node_base[1..n-1] are extra copies allocated with llama_numa_alloc and freed here.
    struct numa_mirror_buffer {
        ggml_backend_buffer_t buf;
        size_t size;
        void * node_base[GGML_NUMA_MAX_NODES];
    };
    std::vector<numa_mirror_buffer> numa_mirror_bufs;

    buft_list_t cpu_buft_list;
    std::map<ggml_backend_dev_t, buft_list_t> gpu_buft_list;

    struct layer_dev {
        ggml_backend_dev_t dev;
        buft_list_t * buft_list;
    };

    layer_dev dev_input = {};
    layer_dev dev_output = {};
    std::vector<layer_dev> dev_layer;

    bool has_tensor_overrides;

    std::vector<float> tensor_split_owned;
};

llama_model::llama_model(const llama_model_params & params) : params(params), pimpl(std::make_unique<impl>()) {
    if (params.tensor_split != nullptr) {
        // llama_model_params stores tensor_split as a borrowed pointer, but the model
        // may need it later for tensor-parallel KV-cache split metadata.
        pimpl->tensor_split_owned.assign(params.tensor_split, params.tensor_split + llama_max_devices());
        this->params.tensor_split = pimpl->tensor_split_owned.data();
    }
    pimpl->has_tensor_overrides = params.tensor_buft_overrides && params.tensor_buft_overrides[0].pattern;
}

llama_model::~llama_model() {
    for (auto * lora : loras) {
        delete lora;
    }
}

llama_model::impl::~impl() {
    // free NUMA mirror per-tensor pointer tables and the extra node-local weight copies
    for (auto & [ctx, bufs] : ctxs_bufs) {
        for (ggml_tensor * t = ggml_get_first_tensor(ctx.get()); t != nullptr; t = ggml_get_next_tensor(ctx.get(), t)) {
            llama_numa_tensor_clear_mirror(t);
        }
    }
    for (auto & mb : numa_mirror_bufs) {
        for (int n = 1; n < llama_numa_node_count(); ++n) { // node_base[0] aliases the original buffer
            if (mb.node_base[n]) {
                llama_numa_free(mb.node_base[n], mb.size);
            }
        }
    }
    numa_mirror_bufs.clear();
}

// Partial mirror for models too big to duplicate in full (e.g. 215 GiB on a 256 GiB box):
// duplicate only *hot* tensors — everything except routed-expert stacks (read for only
// 8/256 experts per token) and the token embedding table (gather, not mul_mat) — as many as
// fit in the per-node free RAM, largest first (every hot byte is read once per token, so the
// value per byte is equal). The original pages of each mirrored tensor are migrated onto
// node 0 (they were interleaved by first-touch); per-node arena copies serve the other nodes.
// Routed experts stay interleaved, which is the best placement for them anyway since every
// node reads a random subset each token.

#if defined(__linux__)
#include <sys/syscall.h>
#include <unistd.h>
#ifndef MPOL_MF_MOVE
#define MPOL_MF_MOVE (1 << 1)
#endif
// migrate up to max_bytes of resident pages that currently sit on from_node within
// [base, base+size) to to_node; returns bytes actually migrated. Own-process
// move_pages(MPOL_MF_MOVE) needs no privilege.
static size_t llama_numa_migrate_node_pages(void * base, size_t size, int from_node, int to_node, size_t max_bytes) {
    const long ps = sysconf(_SC_PAGESIZE);
    if (ps <= 0) return 0;
    uintptr_t beg = (uintptr_t) base & ~(uintptr_t) (ps - 1);
    uintptr_t end = ((uintptr_t) base + size + ps - 1) & ~(uintptr_t) (ps - 1);
    const size_t chunk = 1024;
    std::vector<void *> addrs(chunk);
    std::vector<int>    status(chunk);
    std::vector<int>    nodes(chunk);
    size_t moved = 0;
    for (uintptr_t p = beg; p < end && moved < max_bytes; ) {
        size_t n = 0;
        for (; n < chunk && p < end; ++n, p += ps) {
            addrs[n] = (void *) p;
        }
        if (syscall(SYS_move_pages, 0, n, addrs.data(), nullptr, status.data(), 0) != 0) {
            break;
        }
        size_t m = 0;
        for (size_t i = 0; i < n; ++i) {
            if (status[i] == from_node) {
                addrs[m] = addrs[i];
                nodes[m] = to_node;
                ++m;
            }
        }
        if (m > 0 && syscall(SYS_move_pages, 0, m, addrs.data(), nodes.data(), status.data(), MPOL_MF_MOVE) == 0) {
            for (size_t i = 0; i < m; ++i) {
                if (status[i] == to_node) {
                    moved += (size_t) ps;
                }
            }
        }
    }
    return moved;
}
#else
static size_t llama_numa_migrate_node_pages(void * base, size_t size, int from_node, int to_node, size_t max_bytes) {
    GGML_UNUSED(base); GGML_UNUSED(size); GGML_UNUSED(from_node); GGML_UNUSED(to_node); GGML_UNUSED(max_bytes);
    return 0;
}
#endif

// per-node reclaimable RAM ("Node N MemFree" + inactive file cache) in bytes; 0 when unknown
static size_t llama_node_free_ram_bytes(int node) {
#if defined(__linux__)
    char path[128];
    snprintf(path, sizeof(path), "/sys/devices/system/node/node%d/meminfo", node);
    FILE * f = fopen(path, "r");
    if (!f) return 0;
    char line[256];
    size_t free_kb = 0, inact_file_kb = 0;
    while (fgets(line, sizeof(line), f)) {
        if (sscanf(line, "Node %*d MemFree: %zu kB", &free_kb) == 1) continue;
        if (sscanf(line, "Node %*d Inactive(file): %zu kB", &inact_file_kb) == 1) continue;
    }
    fclose(f);
    // anonymous weight pages can only grow by reclaiming clean file cache on the same node
    return (free_kb + inact_file_kb) * 1024;
#else
    GGML_UNUSED(node);
    return 0;
#endif
}

// Parallel memcpy used by the NUMA weight-mirroring paths below. A single-threaded memcpy of
// a multi-GiB MPOL_BIND destination spends most of its time in the kernel (first-touch page
// faults, THP allocation, direct reclaim when RAM is tight), so splitting the range across
// worker threads speeds the mirror up roughly until memory bandwidth saturates. Chunk
// boundaries are aligned to 2 MiB so each thread faults disjoint huge pages. The destination
// VMA carries MPOL_BIND from llama_numa_alloc, so pages land on the target node no matter
// which thread faults them; the source has already been bound to node 0 by llama_numa_bind.
static void llama_numa_mirror_memcpy(void * dst, const void * src, size_t size) {
    const size_t align = 2ull << 20;
    unsigned n_threads = std::thread::hardware_concurrency();
    if (const char * env = getenv("GGML_NUMA_MIRROR_THREADS")) {
        const int v = atoi(env);
        if (v > 0) {
            n_threads = (unsigned) v;
        }
    }
    n_threads = std::min(n_threads, 32u);
    if (n_threads < 2 || size < (256ull << 20)) {
        memcpy(dst, src, size);
        return;
    }
    size_t chunk = ((size / n_threads) + align - 1) & ~(align - 1);
    if (chunk == 0) {
        chunk = align;
    }
    std::vector<std::thread> workers;
    workers.reserve(n_threads - 1);
    size_t pos = 0;
    for (unsigned i = 0; i < n_threads && pos < size; ++i) {
        const size_t end = std::min(pos + chunk, size);
        char       * d = (char *)       dst + pos;
        const char * s = (const char *) src + pos;
        const size_t n = end - pos;
        if (i + 1 == n_threads || end == size) {
            memcpy(d, s, n); // tail on the calling thread
            break;
        }
        workers.emplace_back([d, s, n]() { memcpy(d, s, n); });
        pos = end;
    }
    for (auto & w : workers) {
        w.join();
    }

#if defined(__gnu_linux__)
    // env GGML_NUMA_THP=collapse: synchronously collapse the just-populated mirror into
    // 2 MiB pages (MADV_COLLAPSE, best-effort). The VMA already carries MADV_HUGEPAGE from
    // ggml_numa_alloc, but under memory fragmentation fault-time THP allocation mostly falls
    // back to 4 KiB pages; collapse triggers direct compaction instead. One-time cost at
    // model load; ignored on kernels < 6.1 or when the range cannot collapse.
    static const bool thp_collapse = []() {
        const char * e = getenv("GGML_NUMA_THP");
        return e && strcmp(e, "collapse") == 0;
    }();
    if (thp_collapse) {
        const int64_t t0 = ggml_time_us();
        if (madvise(dst, size, MADV_COLLAPSE) == 0) {
            LLAMA_LOG_INFO("%s: MADV_COLLAPSE %.2f GiB in %.1f s\n", __func__,
                    size / 1073741824.0, (ggml_time_us() - t0) / 1.0e6);
        } else {
            LLAMA_LOG_WARN("%s: MADV_COLLAPSE failed (errno=%d); continuing with 4K pages\n", __func__, errno);
        }
    }
#endif
}

static bool llama_buffer_is_pinned_host(ggml_backend_buffer_t buffer) {
    return strstr(ggml_backend_buft_name(ggml_backend_buffer_get_type(buffer)), "_Host") != nullptr;
}

void llama_model::numa_mirror_weights_partial() {
    const int n_nodes = llama_numa_node_count();

    // budget: tightest node's free RAM minus a reserve for KV cache, compute buffers and the OS
    const size_t reserve = 6ull << 30;
    // hard cap: keep well clear of the node capacity (strict MPOL_BIND/mbind migration can
    // OOM-kill before reclaim keeps up); env override for experimentation
    const char * cap_env = getenv("GGML_NUMA_MIRROR_BUDGET_GB");
    const size_t cap = cap_env ? ((size_t) atoi(cap_env) << 30) : (12ull << 30);

    std::vector<size_t> node_free(n_nodes, 0);
    for (int n = 0; n < n_nodes; ++n) {
        node_free[n] = llama_node_free_ram_bytes(n);
    }
    const auto compute_budget = [&]() {
        size_t b = SIZE_MAX;
        for (int n = 0; n < n_nodes; ++n) {
            if (node_free[n] > 0) {
                b = std::min(b, node_free[n] > reserve ? node_free[n] - reserve : (size_t) 0);
            }
        }
        if (b == SIZE_MAX) {
            b = cap; // per-node free unknown: conservative default
        }
        return std::min(b, cap);
    };
    size_t budget = compute_budget();

    // Rebalance: one over-full node (e.g. numa_balancing drift during the long no-mmap
    // load) caps the budget for every node. Migrate pages of routed-expert tensors (they
    // stay effectively interleaved either way, since every node reads a random subset of
    // experts per token) from the tightest node to the roomiest one to even the budget out.
    // Skipped when NUMA EP is enabled: expert pages are placed per node by
    // numa_ep_place_experts instead.
    const char * ep_env = getenv("GGML_NUMA_EP");
    const bool ep = ep_env && atoi(ep_env) != 0;
    if (budget < cap && n_nodes >= 2) {
        int tight = 0, roomy = 0;
        for (int n = 0; n < n_nodes; ++n) {
            if (node_free[n] < node_free[tight]) tight = n;
            if (node_free[n] > node_free[roomy]) roomy = n;
        }
        const size_t want = std::min(cap, node_free[roomy] > reserve ? node_free[roomy] - reserve : (size_t) 0);
        if (tight != roomy && want > budget) {
            const size_t deficit = want - budget + reserve; // slack so the reserve survives
            std::vector<ggml_tensor *> exps;
            for (auto & [ctx, bufs] : pimpl->ctxs_bufs) {
                for (ggml_tensor * t = ggml_get_first_tensor(ctx.get()); t != nullptr; t = ggml_get_next_tensor(ctx.get(), t)) {
                    if (!t->data || t->view_src || !t->buffer || !ggml_backend_buffer_is_host(t->buffer) ||
                            llama_buffer_is_pinned_host(t->buffer)) {
                        continue;
                    }
                    const bool is_exps = strstr(t->name, "_exps") != nullptr;
                    // with NUMA EP, expert pages are placed per node by numa_ep_place_experts;
                    // migrating them would break that locality, so only non-expert pages
                    // participate in balancing (EP=0 keeps the historical experts-only set)
                    if (ep ? is_exps : !is_exps) {
                        continue;
                    }
                    exps.push_back(t);
                }
            }
            std::sort(exps.begin(), exps.end(), [](const ggml_tensor * a, const ggml_tensor * b) {
                return ggml_nbytes(a) > ggml_nbytes(b);
            });
            size_t moved = 0;
            for (ggml_tensor * t : exps) {
                if (moved >= deficit) {
                    break;
                }
                moved += llama_numa_migrate_node_pages(t->data, ggml_nbytes(t), tight, roomy, deficit - moved);
            }
            for (int n = 0; n < n_nodes; ++n) {
                node_free[n] = llama_node_free_ram_bytes(n);
            }
            const size_t new_budget = compute_budget();
            LLAMA_LOG_INFO("%s: NUMA partial mirror: rebalanced %.2f GiB node%d->node%d, budget %.2f -> %.2f GiB/node\n",
                    __func__, moved/1073741824.0, tight, roomy, budget/1073741824.0, new_budget/1073741824.0);
            budget = new_budget;
        }
    }

    std::vector<ggml_tensor *> hot;
    size_t hot_bytes = 0;
    std::vector<ggml_tensor *> cand;
    for (auto & [ctx, bufs] : pimpl->ctxs_bufs) {
        for (ggml_tensor * t = ggml_get_first_tensor(ctx.get()); t != nullptr; t = ggml_get_next_tensor(ctx.get(), t)) {
            if (!t->data || t->view_src || !t->buffer || !ggml_backend_buffer_is_host(t->buffer) ||
                    llama_buffer_is_pinned_host(t->buffer)) {
                continue;
            }
            const char * name = t->name;
            if (strstr(name, "_exps") || strstr(name, "token_embd")) {
                continue;
            }
            cand.push_back(t);
        }
    }
    // largest first: fewer mbind/memcpy calls per mirrored GiB
    std::sort(cand.begin(), cand.end(), [](const ggml_tensor * a, const ggml_tensor * b) {
        return ggml_nbytes(a) > ggml_nbytes(b);
    });
    for (ggml_tensor * t : cand) {
        const size_t padded = (ggml_nbytes(t) + 63) & ~(size_t) 63;
        if (hot_bytes + padded > budget) {
            continue; // keep going: smaller tensors may still fit
        }
        hot.push_back(t);
        hot_bytes += padded;
    }
    if (hot.empty()) {
        LLAMA_LOG_WARN("%s: NUMA partial mirror: no room for any hot tensor (budget %.2f GiB/node); "
                "continuing WITHOUT weight mirroring\n", __func__, budget/1073741824.0);
        return;
    }

    const size_t extra_needed = hot_bytes * (size_t) (n_nodes - 1);
    const size_t avail = llama_get_available_ram_bytes();
    LLAMA_LOG_INFO("%s: NUMA partial mirror: %zu hot tensors %.2f GiB; need +%.2f GiB (%.2f GiB available)\n",
            __func__, hot.size(), hot_bytes/1073741824.0, extra_needed/1073741824.0, avail/1073741824.0);
    if (avail > 0 && extra_needed + (4ull << 30) > avail) {
        LLAMA_LOG_WARN("%s: NUMA partial mirror: insufficient free RAM; continuing WITHOUT weight mirroring\n", __func__);
        return;
    }

    // one arena per extra node; packed 64-byte aligned tensor copies
    for (int n = 1; n < n_nodes; ++n) {
        void * arena = llama_numa_alloc(hot_bytes, n);
        if (!arena) {
            for (auto & mb : pimpl->numa_mirror_bufs) {
                for (int j = 1; j < n_nodes; ++j) {
                    if (mb.node_base[j]) {
                        llama_numa_free(mb.node_base[j], mb.size);
                    }
                }
            }
            pimpl->numa_mirror_bufs.clear();
            LLAMA_LOG_WARN("%s: NUMA partial mirror: allocation failed; continuing WITHOUT weight mirroring\n", __func__);
            return;
        }
        impl::numa_mirror_buffer mb;
        mb.buf  = nullptr;
        mb.size = hot_bytes;
        for (int j = 0; j < GGML_NUMA_MAX_NODES; ++j) {
            mb.node_base[j] = nullptr;
        }
        mb.node_base[n] = arena;
        pimpl->numa_mirror_bufs.push_back(mb);
    }

    int n_mirrored = 0;
    size_t off = 0;
    for (ggml_tensor * t : hot) {
        const size_t nbytes = ggml_nbytes(t);
        const size_t padded = (nbytes + 63) & ~(size_t) 63;
        // node 0 reuses the original data, migrated node-0-local (best effort)
        llama_numa_bind(t->data, nbytes, 0);
        for (int n = 1; n < n_nodes; ++n) {
            char * arena = (char *) pimpl->numa_mirror_bufs[n - 1].node_base[n];
            llama_numa_mirror_memcpy(arena + off, t->data, nbytes); // MPOL_BIND places these faulted pages on node n
        }
        void * node_data[GGML_NUMA_MAX_NODES];
        node_data[0] = t->data;
        for (int n = 1; n < n_nodes; ++n) {
            node_data[n] = (char *) pimpl->numa_mirror_bufs[n - 1].node_base[n] + off;
        }
        llama_numa_tensor_set_mirror(t, node_data);
        ++n_mirrored;
        off += padded;
    }
    LLAMA_LOG_INFO("%s: NUMA partial mirror: duplicated %d hot tensors across %d nodes\n",
            __func__, n_mirrored, n_nodes);
}

// NUMA expert placement serves two different EP layouts:
//
//   - CPU GGML_NUMA_EP splits the rows inside every expert across NUMA nodes.
//   - CUDA streaming GGML_CUDA_MOE_PP_EP splits the expert dimension in half,
//     so the pinned host pages for experts 0..N/2 and N/2..N are placed next
//     to GPU 0 and GPU 1 respectively.
//
// Keeping these layouts separate matters: row-striped placement is ideal for
// CPU EP, but makes every GPU H2D copy cross both sockets. Boundary pages stay
// on the lower-numbered node.
void llama_model::numa_ep_place_experts(bool used_mmap) {
    const char * cpu_env = getenv("GGML_NUMA_EP");
    const char * gpu_env = getenv("GGML_CUDA_MOE_PP_EP");
    const bool cpu_ep = cpu_env && atoi(cpu_env) != 0;
    const bool gpu_ep = gpu_env && atoi(gpu_env) != 0;
    if (!cpu_ep && !gpu_ep) {
        return;
    }
    const int n_nodes = llama_numa_node_count();
    if (n_nodes < 2) {
        return;
    }
    if (used_mmap && cpu_ep) {
        const char * env_mmap = getenv("GGML_NUMA_EP_MMAP");
        if (!env_mmap || atoi(env_mmap) == 0) {
            LLAMA_LOG_INFO("%s: GGML_NUMA_EP set but the model was loaded with mmap; skipping expert placement (set GGML_NUMA_EP_MMAP=1 for policy-only placement)\n", __func__);
            return;
        }
        LLAMA_LOG_INFO("%s: GGML_NUMA_EP_MMAP=1: policy-only expert placement on the file mapping (no page migration)\n", __func__);
    }

    long pg = sysconf(_SC_PAGESIZE);
    if (pg <= 0) {
        pg = 4096;
    }

    int n_placed = 0;
    size_t placed_bytes = 0;
    int n_gpu_placed = 0;
    size_t gpu_placed_bytes = 0;
    std::unordered_map<const ggml_tensor *, int> gpu_owner_rank;
    if (gpu_ep) {
        for (size_t il = 0; il < layers.size(); ++il) {
            int owner_rank = -1;
            const ggml_backend_dev_t owner_dev = dev_layer((int) il);
            int rank = 0;
            for (const llama_device & device : devices) {
                if (device.is_meta) {
                    continue;
                }
                if (rank < 2 && device.dev == owner_dev) {
                    owner_rank = rank;
                }
                rank++;
            }
            if (owner_rank < 0 || owner_rank > 1) {
                continue;
            }
            const llama_layer & layer = layers[il];
            for (const ggml_tensor * w : {
                    layer.ffn_up_exps, layer.ffn_gate_exps, layer.ffn_down_exps, layer.ffn_gate_up_exps }) {
                if (w != nullptr) {
                    gpu_owner_rank[w] = owner_rank;
                }
            }
        }
    }
#if defined(__gnu_linux__)
    // env GGML_NUMA_THP=collapse: after binding each expert range to its node (pages are
    // already populated by the weight read + mbind migration), synchronously collapse it
    // into 2 MiB pages (MADV_COLLAPSE, best-effort). This is where the bulk of CPU-side
    // weight bytes live under GGML_NUMA_EP, so it is the main TLB-miss lever for TG.
    static const bool thp_collapse = []() {
        const char * e = getenv("GGML_NUMA_THP");
        return e && strcmp(e, "collapse") == 0;
    }();
    size_t collapsed_bytes = 0;
#endif
    for (auto & [ctx, bufs] : pimpl->ctxs_bufs) {
        GGML_UNUSED(bufs);
        for (ggml_tensor * t = ggml_get_first_tensor(ctx.get()); t != nullptr; t = ggml_get_next_tensor(ctx.get(), t)) {
            if (!t->data || t->view_src || !t->buffer || !ggml_backend_buffer_is_host(t->buffer)) {
                continue;
            }
            if (t->ne[2] <= 1 || strstr(t->name, "_exps") == nullptr) {
                continue;
            }
            const bool pinned = llama_buffer_is_pinned_host(t->buffer);
            if (pinned) {
                if (!gpu_ep) {
                    continue;
                }

                const int n_ranks = std::min(n_nodes, 2);
                const int64_t n_expert = t->ne[2];
                const size_t expert_bytes = t->nb[2];
                char * base = (char *) t->data;
                const llama_moe_pp_ep_plan plan = llama_moe_pp_ep_plan_make(n_expert);
                const auto owner_it = gpu_owner_rank.find(t);
                const int owner_rank = owner_it != gpu_owner_rank.end() ? owner_it->second : -1;
                for (int logical_rank = 0; logical_rank < n_ranks; ++logical_rank) {
                    const int rank = llama_moe_pp_ep_device_rank(plan, logical_rank, owner_rank);
                    const int64_t e0 = plan.expert_begin[logical_rank];
                    const int64_t e1 = e0 + plan.expert_count[logical_rank];
                    uintptr_t begin = (uintptr_t) (base + e0 * expert_bytes);
                    uintptr_t end   = (uintptr_t) (base + e1 * expert_bytes);
                    if (logical_rank > 0) {
                        begin = (begin + pg - 1) & ~((uintptr_t) pg - 1);
                    }
                    if (logical_rank + 1 < n_ranks) {
                        end = (end + pg - 1) & ~((uintptr_t) pg - 1);
                    }
                    if (end > begin) {
                        llama_numa_bind((void *) begin, end - begin, rank);
                    }
                }
                ++n_gpu_placed;
                gpu_placed_bytes += ggml_nbytes(t);
                continue;
            }
            if (!cpu_ep) {
                continue;
            }
            const int64_t n_expert = t->ne[2];
            const size_t eb = t->nb[2]; // bytes per expert plane (contiguous along ne[2])
            // per-node row window inside each expert plane; must match the compute side
            // in repack.cpp (rows claimed in blocks of ep_chunk, windows aligned to 128
            // rows so any ep_chunk <= 128 divides them evenly)
            const int64_t ne1 = t->ne[1];
            const size_t rb = t->nb[1];
            char * base = (char *) t->data;
            for (int64_t e = 0; e < n_expert; ++e) {
                char * ebase = base + e * eb;
                for (int n = 0; n < n_nodes; ++n) {
                    ggml_shard_window window;
                    GGML_ASSERT(ggml_shard_window_equal(ne1, n_nodes, n, 128, window));
                    const int64_t r0 = window.begin;
                    const int64_t r1 = window.end;
                    if (r1 <= r0) {
                        continue;
                    }
                    char * p = ebase + r0 * rb;
                    size_t sz = (size_t) (r1 - r0) * rb;
                    // keep the boundary page on the earlier node: only bind whole pages
                    const uintptr_t up = ((uintptr_t) p + pg - 1) & ~((uintptr_t) pg - 1);
                    if (n > 0) {
                        if ((size_t) (up - (uintptr_t) p) >= sz) {
                            continue;
                        }
                        sz -= up - (uintptr_t) p;
                        p = (char *) up;
                    }
                    if (used_mmap) {
                        llama_numa_bind_policy(p, sz, n);
                    } else {
                        llama_numa_bind(p, sz, n);
                    }
#if defined(__gnu_linux__)
                    if (thp_collapse) {
                        const uintptr_t ua = ((uintptr_t) p + pg - 1) & ~((uintptr_t) pg - 1);
                        if ((size_t) (ua - (uintptr_t) p) < sz) {
                            const size_t csz = sz - (ua - (uintptr_t) p);
                            // the weight buffer is plain posix_memalign memory: make sure the
                            // VMA is THP-eligible before asking for a synchronous collapse
                            madvise((void *) ua, csz, MADV_HUGEPAGE);
                            if (madvise((void *) ua, csz, MADV_COLLAPSE) == 0) {
                                collapsed_bytes += csz;
                            } else {
                                LLAMA_LOG_WARN("%s: MADV_COLLAPSE failed on %.2f MiB (errno=%d); continuing\n",
                                        __func__, csz/1048576.0, errno);
                            }
                        }
                    }
#endif
                }
            }
            ++n_placed;
            placed_bytes += ggml_nbytes(t);
        }
    }
#if defined(__gnu_linux__)
    if (thp_collapse) {
        LLAMA_LOG_INFO("%s: MADV_COLLAPSE applied to %.2f GiB of expert weights\n",
                __func__, collapsed_bytes/1073741824.0);
    }
#endif
    LLAMA_LOG_INFO("%s: NUMA EP: placed %d expert tensors (%.2f GiB) across %d nodes\n",
            __func__, n_placed, placed_bytes/1073741824.0, n_nodes);
    if (n_gpu_placed > 0) {
        LLAMA_LOG_INFO("%s: CUDA MoE PP EP: placed %d pinned expert tensors (%.2f GiB) across %d GPU-local NUMA nodes\n",
                __func__, n_gpu_placed, gpu_placed_bytes/1073741824.0, std::min(n_nodes, 2));
    }
}

void llama_model::numa_mirror_weights() {
    if (!llama_numa_mirror_active() || !(llama_numa_get_mirror() & GGML_NUMA_MIRROR_WEIGHTS)) {
        return;
    }
    if (!pimpl->numa_mirror_bufs.empty()) {
        return; // already mirrored (e.g. a second context created on the same model)
    }
    const int n_nodes = llama_numa_node_count();
    if (n_nodes < 2) {
        return;
    }

    // NUMA expert parallelism: routed experts stay single-copy and are placed per node by
    // numa_ep_place_experts instead of being duplicated
    const char * ep_env = getenv("GGML_NUMA_EP");
    const bool ep = ep_env && atoi(ep_env) != 0;

    size_t total_host = 0;
    size_t total_ep = 0;
    for (auto & [ctx, bufs] : pimpl->ctxs_bufs) {
        for (auto & buf : bufs) {
            if (buf && ggml_backend_buffer_is_host(buf.get()) && !llama_buffer_is_pinned_host(buf.get())) {
                total_host += ggml_backend_buffer_get_size(buf.get());
            }
        }
        if (ep) {
            for (ggml_tensor * t = ggml_get_first_tensor(ctx.get()); t != nullptr; t = ggml_get_next_tensor(ctx.get(), t)) {
                if (t->data && !t->view_src && t->buffer && ggml_backend_buffer_is_host(t->buffer) &&
                    !llama_buffer_is_pinned_host(t->buffer) &&
                    t->ne[2] > 1 && strstr(t->name, "_exps")) {
                    total_ep += ggml_nbytes(t);
                }
            }
        }
    }
    if (total_host == 0) {
        return;
    }

    const size_t extra_needed = (total_host - total_ep) * (size_t) (n_nodes - 1); // node 0 reuses the original
    const size_t avail = llama_get_available_ram_bytes();
    LLAMA_LOG_INFO("%s: NUMA mirror: %d nodes, host weights %.2f GiB; need +%.2f GiB for mirrors (%.2f GiB available)\n",
            __func__, n_nodes, total_host/1073741824.0, extra_needed/1073741824.0, avail/1073741824.0);
    if (avail > 0 && extra_needed + (2ull << 30) > avail) {
        LLAMA_LOG_WARN("%s: NUMA mirror: insufficient free RAM to duplicate weights across %d nodes; "
                "falling back to PARTIAL mirror (hot tensors only)\n", __func__, n_nodes);
        numa_mirror_weights_partial();
        return;
    }
    if (getenv("GGML_NUMA_MIRROR_PARTIAL")) {
        LLAMA_LOG_INFO("%s: NUMA mirror: GGML_NUMA_MIRROR_PARTIAL set, mirroring hot tensors only\n", __func__);
        numa_mirror_weights_partial();
        return;
    }

    // MOE-only mirror: only duplicate buffers containing routed expert weights (_exps tensors).
    // Useful in hybrid CPU+GPU mode where attention is on GPU and only MoE experts are on CPU.
    const bool mirror_moe_only = getenv("GGML_NUMA_MIRROR_MOE") != nullptr;
    if (mirror_moe_only) {
        size_t mirrored_bytes = 0;
        for (auto & [ctx, bufs] : pimpl->ctxs_bufs) {
            for (auto & buf : bufs) {
                if (!buf || !ggml_backend_buffer_is_host(buf.get()) || llama_buffer_is_pinned_host(buf.get())) {
                    continue;
                }
                // Check if this buffer contains any _exps tensors
                bool has_exps = false;
                for (ggml_tensor * t = ggml_get_first_tensor(ctx.get()); t != nullptr; t = ggml_get_next_tensor(ctx.get(), t)) {
                    if (t->data && !t->view_src && t->buffer == buf.get() && strstr(t->name, "_exps")) {
                        has_exps = true;
                        break;
                    }
                }
                if (!has_exps) {
                    continue; // skip non-expert buffers
                }

                impl::numa_mirror_buffer mb;
                mb.buf  = buf.get();
                mb.size = ggml_backend_buffer_get_size(buf.get());
                void * base = ggml_backend_buffer_get_base(buf.get());
                for (int n = 0; n < GGML_NUMA_MAX_NODES; ++n) {
                    mb.node_base[n] = nullptr;
                }
                mb.node_base[0] = base;

                llama_numa_bind(base, mb.size, 0);
                for (int n = 1; n < n_nodes; ++n) {
                    void * copy = llama_numa_alloc(mb.size, n);
                    if (!copy) {
                        LLAMA_LOG_WARN("%s: NUMA mirror-moe: alloc failed on node %d\n", __func__, n);
                        for (int k = 1; k < n; ++k) { llama_numa_free(mb.node_base[k], mb.size); mb.node_base[k] = nullptr; }
                        goto skip_buf_moe;
                    }
                    llama_numa_mirror_memcpy(copy, base, mb.size);
                    mb.node_base[n] = copy;
                }
                mirrored_bytes += mb.size * (n_nodes - 1);
                pimpl->numa_mirror_bufs.push_back(mb);
                skip_buf_moe:;
            }
        }
        LLAMA_LOG_INFO("%s: NUMA mirror-moe: mirrored %.2f GiB of expert weights across %d nodes\n",
                __func__, mirrored_bytes / 1073741824.0, n_nodes);
        return;
    }

    // allocate per-node copies of each host weight buffer
    for (auto & [ctx, bufs] : pimpl->ctxs_bufs) {
        for (auto & buf : bufs) {
            if (!buf || !ggml_backend_buffer_is_host(buf.get()) || llama_buffer_is_pinned_host(buf.get())) {
                continue;
            }
            impl::numa_mirror_buffer mb;
            mb.buf  = buf.get();
            mb.size = ggml_backend_buffer_get_size(buf.get());
            void * base = ggml_backend_buffer_get_base(buf.get());
            for (int n = 0; n < GGML_NUMA_MAX_NODES; ++n) {
                mb.node_base[n] = nullptr;
            }
            mb.node_base[0] = base;                 // node 0 aliases the original buffer ...

            // with NUMA EP the *_exps tensors stay single-copy: their byte ranges are
            // neither bound to node 0 nor duplicated to the other nodes
            std::vector<std::pair<size_t, size_t>> excl;
            if (ep) {
                for (ggml_tensor * t = ggml_get_first_tensor(ctx.get()); t != nullptr; t = ggml_get_next_tensor(ctx.get(), t)) {
                    if (t->data && !t->view_src && t->buffer == buf.get() &&
                        t->ne[2] > 1 && strstr(t->name, "_exps")) {
                        const size_t off = (size_t) ((const char *) t->data - (const char *) base);
                        excl.emplace_back(off, off + ggml_nbytes(t));
                    }
                }
                std::sort(excl.begin(), excl.end());
            }
            {
                size_t pos = 0;
                const auto bind_gaps = [&](int node) {
                    for (const auto & [s, e] : excl) {
                        if (s > pos) { llama_numa_bind((char *) base + pos, s - pos, node); }
                        pos = e;
                    }
                    if (mb.size > pos) { llama_numa_bind((char *) base + pos, mb.size - pos, node); }
                };
                if (excl.empty()) {
                    llama_numa_bind(base, mb.size, 0);  // ... migrated onto node 0 (best effort)
                } else {
                    bind_gaps(0);
                }
            }
            bool ok = true;
            for (int n = 1; n < n_nodes; ++n) {
                void * p = llama_numa_alloc(mb.size, n);
                if (!p) { ok = false; break; }
                if (excl.empty()) {
                    llama_numa_mirror_memcpy(p, base, mb.size); // MPOL_BIND places these faulted pages on node n
                } else {
                    // duplicate only the non-expert segments
                    size_t pos = 0;
                    for (const auto & [s, e] : excl) {
                        if (s > pos) { llama_numa_mirror_memcpy((char *) p + pos, (const char *) base + pos, s - pos); }
                        pos = e;
                    }
                    if (mb.size > pos) { llama_numa_mirror_memcpy((char *) p + pos, (const char *) base + pos, mb.size - pos); }
                }
                mb.node_base[n] = p;
            }
            if (!ok) {
                for (int n = 1; n < n_nodes; ++n) {
                    llama_numa_free(mb.node_base[n], mb.size);
                }
                for (auto & done : pimpl->numa_mirror_bufs) {
                    for (int n = 1; n < n_nodes; ++n) {
                        llama_numa_free(done.node_base[n], done.size);
                    }
                }
                pimpl->numa_mirror_bufs.clear();
                LLAMA_LOG_WARN("%s: NUMA mirror: allocation failed; continuing WITHOUT weight mirroring\n", __func__);
                return;
            }
            pimpl->numa_mirror_bufs.push_back(mb);
        }
    }

    // point every (non-view) host weight tensor at its per-node copies
    int n_mirrored = 0;
    for (auto & [ctx, bufs] : pimpl->ctxs_bufs) {
        for (ggml_tensor * t = ggml_get_first_tensor(ctx.get()); t != nullptr; t = ggml_get_next_tensor(ctx.get(), t)) {
            if (!t->data || t->view_src || !t->buffer || !ggml_backend_buffer_is_host(t->buffer) ||
                    llama_buffer_is_pinned_host(t->buffer)) {
                continue;
            }
            if (ep && t->ne[2] > 1 && strstr(t->name, "_exps")) {
                continue; // routed experts stay single-copy (NUMA EP placement)
            }
            const impl::numa_mirror_buffer * mb = nullptr;
            for (auto & cand : pimpl->numa_mirror_bufs) {
                if (cand.buf == t->buffer) { mb = &cand; break; }
            }
            if (!mb) {
                continue;
            }
            const size_t offset = (const char *) t->data - (const char *) mb->node_base[0];
            void * node_data[GGML_NUMA_MAX_NODES];
            for (int n = 0; n < n_nodes; ++n) {
                node_data[n] = (char *) mb->node_base[n] + offset;
            }
            llama_numa_tensor_set_mirror(t, node_data);
            ++n_mirrored;
        }
    }
    LLAMA_LOG_INFO("%s: NUMA mirror: duplicated %d weight tensors across %d nodes\n",
            __func__, n_mirrored, n_nodes);
}

void llama_model_base::load_stats(llama_model_loader & ml) {
    pimpl->n_elements = ml.n_elements;
    pimpl->n_bytes = ml.n_bytes;
}

void llama_model_base::load_hparams(llama_model_loader & ml) {
    const gguf_context * ctx = ml.metadata;

    // get metadata as string
    for (int i = 0; i < gguf_get_n_kv(ctx); i++) {
        gguf_type type = gguf_get_kv_type(ctx, i);
        if (type == GGUF_TYPE_ARRAY) {
            continue;
        }
        const char * name = gguf_get_key(ctx, i);
        const std::string value = gguf_kv_to_str(ctx, i);
        gguf_kv.emplace(name, value);
    }

    // get general kv
    ml.get_key(LLM_KV_GENERAL_NAME, name, false);

    // everything past this point is not vocab-related
    // for CLIP models, we only need to load tensors, no hparams
    if (hparams.vocab_only || ml.get_arch() == LLM_ARCH_CLIP) {
        return;
    }

    ml.get_key(LLM_KV_CONTEXT_LENGTH,          hparams.n_ctx_train);
    ml.get_key(LLM_KV_EMBEDDING_LENGTH,        hparams.n_embd);
    ml.get_key(LLM_KV_EMBEDDING_LENGTH_OUT,    hparams.n_embd_out_impl, false);
    ml.get_key(LLM_KV_ATTENTION_CAUSAL,        hparams.causal_attn,     false);
    ml.get_key(LLM_KV_POOLING_TYPE,            hparams.pooling_type,    false);
    ml.get_key(LLM_KV_BLOCK_COUNT,             hparams.n_layer_all);
    GGML_ASSERT(hparams.n_layer_all > 0 && hparams.n_layer_all <= LLAMA_MAX_LAYERS);
    ml.get_key(LLM_KV_EXPERT_COUNT,            hparams.n_expert,        false);
    ml.get_key(LLM_KV_EXPERT_USED_COUNT,       hparams.n_expert_used,   false);
    ml.get_key(LLM_KV_EXPERT_GROUP_COUNT,      hparams.n_expert_groups, false);
    ml.get_key(LLM_KV_EXPERT_GROUP_USED_COUNT, hparams.n_group_used,    false);

    if (arch == LLM_ARCH_HUNYUAN_VL || arch == LLM_ARCH_HUNYUAN_DENSE) {
        if (hparams.n_expert <= 1) {
            hparams.n_expert      = 0;
            hparams.n_expert_used = 0;
        }
    }

    if (arch == LLM_ARCH_WAVTOKENIZER_DEC) {
        ml.get_key(LLM_KV_FEATURES_LENGTH,  hparams.n_embd);
        ml.get_key(LLM_KV_EMBEDDING_LENGTH, hparams.n_embd_out_impl);

        ml.get_key(LLM_KV_POSNET_EMBEDDING_LENGTH, hparams.posnet.n_embd);
        ml.get_key(LLM_KV_POSNET_BLOCK_COUNT,      hparams.posnet.n_layer);

        ml.get_key(LLM_KV_CONVNEXT_EMBEDDING_LENGTH, hparams.convnext.n_embd);
        ml.get_key(LLM_KV_CONVNEXT_BLOCK_COUNT,      hparams.convnext.n_layer);

        GGML_ASSERT(hparams.posnet.n_layer   <= hparams.n_layer_all);
        GGML_ASSERT(hparams.convnext.n_layer <= hparams.n_layer_all);
    }

    GGML_ASSERT(hparams.n_expert <= LLAMA_MAX_EXPERTS);
    GGML_ASSERT(hparams.n_expert_used <= hparams.n_expert);
    if (hparams.n_expert > 0) {
        GGML_ASSERT(hparams.n_expert_used > 0);
        GGML_ASSERT(hparams.n_expert_groups < hparams.n_expert);
        if (hparams.n_expert_groups > 1) {
            GGML_ASSERT(hparams.n_expert % hparams.n_expert_groups == 0);
            GGML_ASSERT(hparams.n_group_used > 0);
            GGML_ASSERT(hparams.n_group_used < hparams.n_expert_groups);
        }
    } else {
        GGML_ASSERT(hparams.n_expert_used == 0);
        GGML_ASSERT(hparams.n_expert_groups == 0);
    }

    std::fill(hparams.n_head_arr.begin(),    hparams.n_head_arr.end(),    0);
    std::fill(hparams.n_head_kv_arr.begin(), hparams.n_head_kv_arr.end(), 0);
    std::fill(hparams.n_ff_arr.begin(),      hparams.n_ff_arr.end(),      0);

    std::fill(hparams.rope_sections.begin(), hparams.rope_sections.end(), 0);
    std::fill(hparams.rope_pattern.begin(),  hparams.rope_pattern.end(), 1);
    std::fill(hparams.is_swa_impl.begin(),   hparams.is_swa_impl.end(), 0);
    std::fill(hparams.is_recr_impl.begin(),  hparams.is_recr_impl.end(),  llm_arch_is_recurrent(ml.get_arch()) ? 1 : 0);
    std::fill(hparams.is_indexer_full_impl.begin(), hparams.is_indexer_full_impl.end(), 0);

    std::fill(hparams.xielu_alpha_n.begin(), hparams.xielu_alpha_n.end(), 0.0f);
    std::fill(hparams.xielu_alpha_p.begin(), hparams.xielu_alpha_p.end(), 0.0f);
    std::fill(hparams.xielu_beta.begin(),    hparams.xielu_beta.end(), 0.0f);
    std::fill(hparams.xielu_eps.begin(),     hparams.xielu_eps.end(), 0.0f);

    std::fill(hparams.swiglu_clamp_exp.begin(),   hparams.swiglu_clamp_exp.end(),   0.0f);
    std::fill(hparams.swiglu_clamp_shexp.begin(), hparams.swiglu_clamp_shexp.end(), 0.0f);

    ml.get_key_or_arr(LLM_KV_FEED_FORWARD_LENGTH,  hparams.n_ff_arr,   hparams.n_layer(), false);
    ml.get_key_or_arr(LLM_KV_ATTENTION_HEAD_COUNT, hparams.n_head_arr, hparams.n_layer(), false);

    // Populate deepstack_mapping_arr - initialized to -1 (no deepstack)
    std::fill(hparams.deepstack_mapping_arr.begin(), hparams.deepstack_mapping_arr.end(), -1);

    // n_head_kv is optional, default to n_head
    hparams.n_head_kv_arr = hparams.n_head_arr;

    ml.get_key_or_arr(LLM_KV_ATTENTION_HEAD_COUNT_KV, hparams.n_head_kv_arr, hparams.n_layer(), false);

    bool rope_finetuned = false;
    ml.get_key(LLM_KV_ROPE_SCALING_FINETUNED, rope_finetuned, false);
    hparams.rope_finetuned = rope_finetuned;

    hparams.n_ctx_orig_yarn = hparams.n_ctx_train;
    ml.get_key(LLM_KV_ROPE_SCALING_ORIG_CTX_LEN, hparams.n_ctx_orig_yarn, false);

    // rope_freq_base (optional)
    hparams.rope_freq_base_train = 10000.0f;
    ml.get_key(LLM_KV_ROPE_FREQ_BASE, hparams.rope_freq_base_train, false);

    std::string rope_scaling("linear");
    ml.get_key(LLM_KV_ROPE_SCALING_TYPE, rope_scaling, false);
    hparams.rope_scaling_type_train = llama_rope_scaling_type_from_string(rope_scaling);
    GGML_ASSERT(hparams.rope_scaling_type_train != LLAMA_ROPE_SCALING_TYPE_UNSPECIFIED);

    // TODO: Handle SWA metadata similarly when models start implementing it
    // rope_freq_scale (inverse of the kv) is optional
    float ropescale = 0.0f;
    if (!ml.get_key(LLM_KV_ROPE_SCALING_FACTOR, ropescale, false)) {
        // try the old key name
        ml.get_key(LLM_KV_ROPE_SCALE_LINEAR, ropescale, false);
    }
    hparams.rope_freq_scale_train = ropescale == 0.0f ? 1.0f : 1.0f/ropescale;

    ml.get_key(LLM_KV_ROPE_SCALING_ATTN_FACTOR, hparams.rope_attn_factor, false);
    ml.get_key(LLM_KV_ROPE_SCALING_ALPHA,       hparams.rope_scaling_alpha, false);

    // non-transformer models do not have attention heads
    if (hparams.n_head() > 0) {
        // gpt-neox n_rot = rotary_pct * (n_embd / n_head)
        // gpt-j n_rot = rotary_dim

        hparams.n_embd_head_k_full = hparams.n_embd / hparams.n_head();
        ml.get_key(LLM_KV_ATTENTION_KEY_LENGTH, hparams.n_embd_head_k_full, false);

        hparams.n_embd_head_v_full = hparams.n_embd / hparams.n_head();
        ml.get_key(LLM_KV_ATTENTION_VALUE_LENGTH, hparams.n_embd_head_v_full, false);

        // sanity check for n_rot (optional)
        hparams.n_rot_full = hparams.n_embd_head_k_full;

        ml.get_key(LLM_KV_ROPE_DIMENSION_COUNT, hparams.n_rot_full, false);

        if (arch == LLM_ARCH_LLAMA || arch == LLM_ARCH_DECI || arch == LLM_ARCH_FALCON || arch == LLM_ARCH_LLAMA_EMBED) {
            if (hparams.n_rot_full != hparams.n_embd_head_k_full) {
                throw std::runtime_error(format("invalid n_rot: %u, expected %u", hparams.n_rot_full, hparams.n_embd_head_k_full));
            }
        }
    } else {
        hparams.n_rot_full = 0;
        hparams.n_embd_head_k_full = 0;
        hparams.n_embd_head_v_full = 0;
    }

    // head size and n_rot for SWA layers
    {
        hparams.n_embd_head_k_swa = hparams.n_embd_head_k_full;
        hparams.n_embd_head_v_swa = hparams.n_embd_head_v_full;
        ml.get_key(LLM_KV_ATTENTION_KEY_LENGTH_SWA, hparams.n_embd_head_k_swa, false);
        ml.get_key(LLM_KV_ATTENTION_VALUE_LENGTH_SWA, hparams.n_embd_head_v_swa, false);

        hparams.n_rot_swa = hparams.n_rot_full;
        ml.get_key(LLM_KV_ROPE_DIMENSION_COUNT_SWA, hparams.n_rot_swa, false);
    }

    // for classifier models
    ml.get_arr(LLM_KV_CLASSIFIER_OUTPUT_LABELS, classifier_labels, false);
    if (!classifier_labels.empty()) {
        hparams.n_cls_out = classifier_labels.size();
    }

    // per-arch hparams
    load_arch_hparams(ml);

    pimpl->n_bytes = ml.n_bytes;

    pimpl->desc_str = arch_name() + " " + type_name() + " " + ml.ftype_name();

    pimpl->ftype = ml.ftype;

    if (hparams.f_max_alibi_bias > 0.0f) {
        hparams.use_alibi = true;
    }

    hparams.rope_type = llama_model_rope_type(this);
}

void llama_model_base::load_vocab(llama_model_loader & ml) {
    const auto kv = LLM_KV(arch);

    vocab.load(ml, kv);
}

bool llama_model_base::load_tensors(llama_model_loader & ml) {
    const auto & split_mode   = params.split_mode;
    const bool use_mlock      = params.load_mode == LLAMA_LOAD_MODE_MLOCK || params.load_mode == LLAMA_LOAD_MODE_MMAP_MLOCK;
    const auto & tensor_split = params.tensor_split;

    const int n_layer_all = hparams.n_layer_all;
    const int n_gpu_layers = this->n_gpu_layers();

    const bool use_mmap_buffer = true;

    this->ml = &ml; // to be used by create_tensor() and load_arch_tensors()

    if (ml.use_mmap && params.load_mode == LLAMA_LOAD_MODE_AUTO) {
        for (const auto & dev : devices) {
            ggml_backend_dev_props props;
            ggml_backend_dev_get_props(dev.dev, &props);
            if (!props.caps.mmap_support) {
                ml.use_mmap = false;
                break;
            }
        }
    }

    const char * load_mode_name = params.load_mode == LLAMA_LOAD_MODE_AUTO
        ? llama_load_mode_name(ml.use_mmap ? LLAMA_LOAD_MODE_MMAP : LLAMA_LOAD_MODE_NONE)
        : llama_load_mode_name(params.load_mode);

    LLAMA_LOG_INFO("%s: loading model tensors, this can take a while... (load_mode = %s)\n",
        __func__, load_mode_name);

    // build a list of buffer types for the CPU and GPU devices
    pimpl->cpu_buft_list = make_cpu_buft_list(devices, params.use_extra_bufts, params.no_host);
    for (const auto & dev : devices) {
        buft_list_t buft_list = make_gpu_buft_list(dev.dev, split_mode, tensor_split);
        // add CPU buffer types as a fallback
        buft_list.insert(buft_list.end(), pimpl->cpu_buft_list.begin(), pimpl->cpu_buft_list.end());
        pimpl->gpu_buft_list.emplace(dev.dev, std::move(buft_list));
    }

    ggml_backend_dev_t cpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    if (cpu_dev == nullptr) {
        throw std::runtime_error(format("%s: no CPU backend found", __func__));
    }

    // calculate the split points
    bool all_zero = tensor_split == nullptr || std::all_of(tensor_split, tensor_split + n_devices(), [](float x) { return x == 0.0f; });
    std::vector<float> splits(n_devices());
    if (all_zero) {
        // default split, by free memory
        for (size_t i = 0; i < n_devices(); ++i) {
            ggml_backend_dev_t dev = devices[i].dev;
            size_t total;
            size_t free;
            ggml_backend_dev_memory(dev, &free, &total);

            // devices can return 0 bytes for free and total memory if they do not
            // have any to report. in this case, we will use the host memory as a fallback
            // fixes: https://github.com/ggml-org/llama.cpp/issues/18577
            if (free == 0 && total == 0) {
                ggml_backend_dev_memory(cpu_dev, &free, &total);
            }
            splits[i] = free;
        }
    } else {
        std::copy(tensor_split, tensor_split + n_devices(), splits.begin());
    }

    // sum and normalize the splits to get the split points
    float split_sum = 0.0f;
    for (size_t i = 0; i < n_devices(); ++i) {
        split_sum += splits[i];
        splits[i] = split_sum;
    }
    for (size_t i = 0; i < n_devices(); ++i) {
        splits[i] /= split_sum;
    }

    const int i_gpu_start = std::max(n_layer_all + 1 - n_gpu_layers, 0);
    const int act_gpu_layers = devices.empty() ? 0 : std::min(n_gpu_layers, n_layer_all + 1);
    auto get_layer_buft_list = [&](int il) -> llama_model::impl::layer_dev {
        const bool is_swa = il < n_layer_all && hparams.is_swa(il);
        if (il < i_gpu_start || (il - i_gpu_start) >= act_gpu_layers) {
            LLAMA_LOG_DEBUG("load_tensors: layer %3d assigned to device %s, is_swa = %d\n", il, ggml_backend_dev_name(cpu_dev), is_swa);
            return {cpu_dev, &pimpl->cpu_buft_list};
        }
        const int layer_gpu = std::upper_bound(splits.begin(), splits.begin() + n_devices(), float(il - i_gpu_start)/act_gpu_layers) - splits.begin();
        auto * dev = devices.at(layer_gpu).dev;
        LLAMA_LOG_DEBUG("load_tensors: layer %3d assigned to device %s, is_swa = %d\n", il, ggml_backend_dev_name(dev), is_swa);
        return {dev, &pimpl->gpu_buft_list.at(dev)};
    };

    // assign the input layer
    // there is very little benefit to offloading the input layer, so always keep it on the CPU
    pimpl->dev_input = { cpu_dev, &pimpl->cpu_buft_list };

    // assign the repeating layers to the devices according to the splits
    pimpl->dev_layer.resize(n_layer_all);
    for (int il = 0; il < n_layer_all; ++il) {
        pimpl->dev_layer[il] = get_layer_buft_list(il);
    }

    // assign the output layer
    pimpl->dev_output = get_layer_buft_list(n_layer_all);

    const auto TENSOR_NOT_REQUIRED = llama_model_loader::TENSOR_NOT_REQUIRED;

    // create tensors for the weights
    {
        // TODO: move to a separate function
        const auto tn = LLM_TN(arch);

        const int64_t n_expert      = hparams.n_expert;
        const int64_t n_expert_used = hparams.n_expert_used;

        if (n_expert > 0 && n_expert_used == 0) {
            throw std::runtime_error("model has expert layers but no expert layers are used");
        }

        layers.resize(n_layer_all);

        // call the per-model loading function
        load_arch_tensors(ml);

        // generic pass: load optional per-tensor/per-expert ".scale" tensors (e.g. NVFP4 scale2)
        // this avoids having to add scale loading to every architecture
        for (int i = 0; i < n_layer_all; ++i) {
            auto & layer = layers[i];

            // attention weight scales (per-tensor, shape {1})
            if (!layer.wq_s && layer.wq) {
                layer.wq_s = create_tensor(tn(LLM_TENSOR_ATTN_Q,   "scale", i), {1}, TENSOR_NOT_REQUIRED);
            }
            if (!layer.wk_s && layer.wk) {
                layer.wk_s = create_tensor(tn(LLM_TENSOR_ATTN_K,   "scale", i), {1}, TENSOR_NOT_REQUIRED);
            }
            if (!layer.wv_s && layer.wv) {
                layer.wv_s = create_tensor(tn(LLM_TENSOR_ATTN_V,   "scale", i), {1}, TENSOR_NOT_REQUIRED);
            }
            if (!layer.wo_s && layer.wo) {
                layer.wo_s = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "scale", i), {1}, TENSOR_NOT_REQUIRED);
            }
            if (!layer.wqkv_s && layer.wqkv) {
                layer.wqkv_s = create_tensor(tn(LLM_TENSOR_ATTN_QKV, "scale", i), {1}, TENSOR_NOT_REQUIRED);
            }
            if (!layer.wqkv_gate_s && layer.wqkv_gate) {
                layer.wqkv_gate_s = create_tensor(tn(LLM_TENSOR_ATTN_GATE, "scale", i), {1}, TENSOR_NOT_REQUIRED);
            }

            // dense FFN weight scales (per-tensor, shape {1})
            if (!layer.ffn_gate_s && layer.ffn_gate) {
                layer.ffn_gate_s = create_tensor(tn(LLM_TENSOR_FFN_GATE, "scale", i), {1}, TENSOR_NOT_REQUIRED);
            }
            if (!layer.ffn_down_s && layer.ffn_down) {
                layer.ffn_down_s = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "scale", i), {1}, TENSOR_NOT_REQUIRED);
            }
            if (!layer.ffn_up_s && layer.ffn_up) {
                layer.ffn_up_s = create_tensor(tn(LLM_TENSOR_FFN_UP, "scale", i), {1}, TENSOR_NOT_REQUIRED);
            }
            if (!layer.ffn_gate_shexp_s && layer.ffn_gate_shexp) {
                layer.ffn_gate_shexp_s = create_tensor(tn(LLM_TENSOR_FFN_GATE_SHEXP, "scale", i), {1}, TENSOR_NOT_REQUIRED);
            }
            if (!layer.ffn_down_shexp_s && layer.ffn_down_shexp) {
                layer.ffn_down_shexp_s = create_tensor(tn(LLM_TENSOR_FFN_DOWN_SHEXP, "scale", i), {1}, TENSOR_NOT_REQUIRED);
            }
            if (!layer.ffn_up_shexp_s && layer.ffn_up_shexp) {
                layer.ffn_up_shexp_s = create_tensor(tn(LLM_TENSOR_FFN_UP_SHEXP, "scale", i), {1}, TENSOR_NOT_REQUIRED);
            }

            // MoE expert weight scales (per-expert, shape {n_expert})
            if (!layer.ffn_gate_exps_s && layer.ffn_gate_exps) {
                layer.ffn_gate_exps_s = create_tensor(tn(LLM_TENSOR_FFN_GATE_EXPS, "scale", i), {n_expert}, TENSOR_NOT_REQUIRED);
            }
            if (!layer.ffn_down_exps_s && layer.ffn_down_exps) {
                layer.ffn_down_exps_s = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS, "scale", i), {n_expert}, TENSOR_NOT_REQUIRED);
            }
            if (!layer.ffn_up_exps_s && layer.ffn_up_exps) {
                layer.ffn_up_exps_s = create_tensor(tn(LLM_TENSOR_FFN_UP_EXPS, "scale", i), {n_expert}, TENSOR_NOT_REQUIRED);
            }

            // recurrent / linear-attention weight scales (per-tensor, shape {1})
            if (!layer.ssm_in_s && layer.ssm_in) {
                layer.ssm_in_s = create_tensor(tn(LLM_TENSOR_SSM_IN, "scale", i), {1}, TENSOR_NOT_REQUIRED);
            }
            if (!layer.ssm_out_s && layer.ssm_out) {
                layer.ssm_out_s = create_tensor(tn(LLM_TENSOR_SSM_OUT, "scale", i), {1}, TENSOR_NOT_REQUIRED);
            }
            if (!layer.ssm_alpha_s && layer.ssm_alpha) {
                layer.ssm_alpha_s = create_tensor(tn(LLM_TENSOR_SSM_ALPHA, "scale", i), {1}, TENSOR_NOT_REQUIRED);
            }
            if (!layer.ssm_beta_s && layer.ssm_beta) {
                layer.ssm_beta_s = create_tensor(tn(LLM_TENSOR_SSM_BETA, "scale", i), {1}, TENSOR_NOT_REQUIRED);
            }
            if (!layer.nextn.eh_proj_s && layer.nextn.eh_proj) {
                layer.nextn.eh_proj_s = create_tensor(tn(LLM_TENSOR_NEXTN_EH_PROJ, "scale", i), {1}, TENSOR_NOT_REQUIRED);
            }
            if (!layer.nextn.shared_head_head_s && layer.nextn.shared_head_head) {
                layer.nextn.shared_head_head_s = create_tensor(tn(LLM_TENSOR_NEXTN_SHARED_HEAD_HEAD, "scale", i), {1}, TENSOR_NOT_REQUIRED);
            }

            // input scales
            if (!layer.wq_in_s && layer.wq) {
                layer.wq_in_s = create_tensor(tn(LLM_TENSOR_ATTN_Q,   "input_scale", i), {1}, TENSOR_NOT_REQUIRED);
            }
            if (!layer.wk_in_s && layer.wk) {
                layer.wk_in_s = create_tensor(tn(LLM_TENSOR_ATTN_K,   "input_scale", i), {1}, TENSOR_NOT_REQUIRED);
            }
            if (!layer.wv_in_s && layer.wv) {
                layer.wv_in_s = create_tensor(tn(LLM_TENSOR_ATTN_V,   "input_scale", i), {1}, TENSOR_NOT_REQUIRED);
            }
            if (!layer.wo_in_s && layer.wo) {
                layer.wo_in_s = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "input_scale", i), {1}, TENSOR_NOT_REQUIRED);
            }
            if (!layer.wqkv_in_s && layer.wqkv) {
                layer.wqkv_in_s = create_tensor(tn(LLM_TENSOR_ATTN_QKV, "input_scale", i), {1}, TENSOR_NOT_REQUIRED);
            }
            if (!layer.wqkv_gate_in_s && layer.wqkv_gate) {
                layer.wqkv_gate_in_s = create_tensor(tn(LLM_TENSOR_ATTN_GATE, "input_scale", i), {1}, TENSOR_NOT_REQUIRED);
            }
            if (!layer.ffn_gate_in_s && layer.ffn_gate) {
                layer.ffn_gate_in_s = create_tensor(tn(LLM_TENSOR_FFN_GATE, "input_scale", i), {1}, TENSOR_NOT_REQUIRED);
            }
            if (!layer.ffn_down_in_s && layer.ffn_down) {
                layer.ffn_down_in_s = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "input_scale", i), {1}, TENSOR_NOT_REQUIRED);
            }
            if (!layer.ffn_up_in_s && layer.ffn_up) {
                layer.ffn_up_in_s = create_tensor(tn(LLM_TENSOR_FFN_UP, "input_scale", i), {1}, TENSOR_NOT_REQUIRED);
            }
            if (!layer.ffn_gate_exps_in_s && layer.ffn_gate_exps) {
                layer.ffn_gate_exps_in_s = create_tensor(tn(LLM_TENSOR_FFN_GATE_EXPS, "input_scale", i), {n_expert}, TENSOR_NOT_REQUIRED);
            }
            if (!layer.ffn_down_exps_in_s && layer.ffn_down_exps) {
                layer.ffn_down_exps_in_s = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS, "input_scale", i), {n_expert}, TENSOR_NOT_REQUIRED);
            }
            if (!layer.ffn_up_exps_in_s && layer.ffn_up_exps) {
                layer.ffn_up_exps_in_s = create_tensor(tn(LLM_TENSOR_FFN_UP_EXPS, "input_scale", i), {n_expert}, TENSOR_NOT_REQUIRED);
            }
            if (!layer.ffn_gate_shexp_in_s && layer.ffn_gate_shexp) {
                layer.ffn_gate_shexp_in_s = create_tensor(tn(LLM_TENSOR_FFN_GATE_SHEXP, "input_scale", i), {1}, TENSOR_NOT_REQUIRED);
            }
            if (!layer.ffn_down_shexp_in_s && layer.ffn_down_shexp) {
                layer.ffn_down_shexp_in_s = create_tensor(tn(LLM_TENSOR_FFN_DOWN_SHEXP, "input_scale", i), {1}, TENSOR_NOT_REQUIRED);
            }
            if (!layer.ffn_up_shexp_in_s && layer.ffn_up_shexp) {
                layer.ffn_up_shexp_in_s = create_tensor(tn(LLM_TENSOR_FFN_UP_SHEXP, "input_scale", i), {1}, TENSOR_NOT_REQUIRED);
            }
            if (!layer.ssm_in_in_s && layer.ssm_in) {
                layer.ssm_in_in_s = create_tensor(tn(LLM_TENSOR_SSM_IN, "input_scale", i), {1}, TENSOR_NOT_REQUIRED);
            }
            if (!layer.ssm_out_in_s && layer.ssm_out) {
                layer.ssm_out_in_s = create_tensor(tn(LLM_TENSOR_SSM_OUT, "input_scale", i), {1}, TENSOR_NOT_REQUIRED);
            }
            if (!layer.ssm_alpha_in_s && layer.ssm_alpha) {
                layer.ssm_alpha_in_s = create_tensor(tn(LLM_TENSOR_SSM_ALPHA, "input_scale", i), {1}, TENSOR_NOT_REQUIRED);
            }
            if (!layer.ssm_beta_in_s && layer.ssm_beta) {
                layer.ssm_beta_in_s = create_tensor(tn(LLM_TENSOR_SSM_BETA, "input_scale", i), {1}, TENSOR_NOT_REQUIRED);
            }
            if (!layer.nextn.eh_proj_in_s && layer.nextn.eh_proj) {
                layer.nextn.eh_proj_in_s = create_tensor(tn(LLM_TENSOR_NEXTN_EH_PROJ, "input_scale", i), {1}, TENSOR_NOT_REQUIRED);
            }
            if (!layer.nextn.shared_head_head_in_s && layer.nextn.shared_head_head) {
                layer.nextn.shared_head_head_in_s = create_tensor(tn(LLM_TENSOR_NEXTN_SHARED_HEAD_HEAD, "input_scale", i), {1}, TENSOR_NOT_REQUIRED);
            }
        }
        // output scales
        if (output && output->type == GGML_TYPE_NVFP4) {
            // weight scale
            if (!output_s) {
                output_s = create_tensor(tn(LLM_TENSOR_OUTPUT, "scale"), {1}, TENSOR_NOT_REQUIRED);
            }
            // input scale
            if (!output_in_s) {
                output_in_s = create_tensor(tn(LLM_TENSOR_OUTPUT, "input_scale"), {1}, TENSOR_NOT_REQUIRED);
            }
        }
    }
    ml.done_getting_tensors();

    // Tied NVFP4 output is valid when no separate LM-head scale tensors are present.
    // If sidecar scales exist, the output weight must be an actual output tensor.
    GGML_ASSERT(!(output && tok_embd &&
            strcmp(output->name, tok_embd->name) == 0 &&
            output->type == GGML_TYPE_NVFP4 &&
            (output_s || output_in_s)));
    // populate tensors_by_name
    for (auto & [_, ctx_ptr] : ml.ctx_map) {
        for (auto * cur = ggml_get_first_tensor(ctx_ptr.get()); cur != NULL; cur = ggml_get_next_tensor(ctx_ptr.get(), cur)) {
            tensors_by_name.emplace_back(ggml_get_name(cur), cur);
        }
    }

    ml.init_mappings(true, use_mlock ? &pimpl->mlock_mmaps : nullptr);
    pimpl->mappings.reserve(ml.mappings.size());

    // create the backend buffers
    std::vector<std::pair<ggml_context *, llama_buf_map>> ctx_buf_maps;
    ctx_buf_maps.reserve(ml.ctx_map.size());

    // Ensure we have enough capacity for the maximum backend buffer we will potentially create
    const size_t n_max_backend_buffer = ml.ctx_map.size() * ml.files.size();
    pimpl->ctxs_bufs.reserve(n_max_backend_buffer);

    for (auto & [buft, ctx_ptr] : ml.ctx_map) {
        ggml_context * ctx = ctx_ptr.get();

        // skip contexts without tensors
        if (ggml_get_first_tensor(ctx) == nullptr) {
            continue;
        }

        llama_buf_map buf_map;
        buf_map.reserve(n_max_backend_buffer);

        // check if it is possible to use buffer_from_host_ptr with this buffer type
        ggml_backend_dev_t dev = ggml_backend_buft_get_device(buft);
        if (!dev) {
            // FIXME: workaround for CPU backend buft having a NULL device
            dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
            if (!dev) {
                throw std::runtime_error(format("%s: no CPU backend found", __func__));
            }
        }
        ggml_backend_dev_props props;
        ggml_backend_dev_get_props(dev, &props);
        bool buffer_from_host_ptr_supported = props.caps.buffer_from_host_ptr;
        bool is_default_buft = buft == ggml_backend_dev_buffer_type(dev);

        std::vector<ggml_backend_buffer_ptr> bufs;
        if (ml.use_mmap && use_mmap_buffer && buffer_from_host_ptr_supported && is_default_buft) {
            GGML_ASSERT(!ml.no_alloc);
            for (uint32_t idx = 0; idx < ml.files.size(); idx++) {
                // only the mmap region containing the tensors in the model is mapped to the backend buffer
                // this is important for metal with apple silicon: if the entire model could be mapped to a metal buffer,
                //     then we could just use metal for all layers
                // this allows using partial offloading when the model size exceeds the metal buffer size, but not the RAM size
                void * addr = nullptr;
                size_t first, last; // NOLINT
                ml.get_mapping_range(&first, &last, &addr, idx, ctx);
                if (first >= last) {
                    continue;
                }
                const size_t max_size = ggml_get_max_tensor_size(ctx);
                ggml_backend_buffer_t buf = ggml_backend_dev_buffer_from_host_ptr(dev, (char *) addr + first, last - first, max_size);
                if (buf == nullptr) {
                    throw std::runtime_error(format("unable to allocate %s buffer", ggml_backend_buft_name(buft)));
                }
                bufs.emplace_back(buf);
                buf_map.emplace(idx, buf);
            }
        } else {
            ggml_backend_buffer_t buf;
            if (ml.no_alloc) {
                buf = ggml_backend_buft_alloc_buffer(buft, /*size =*/ 0); // dummy buffer
                for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != nullptr; t = ggml_get_next_tensor(ctx, t)) {
                    t->buffer = buf; // set dummy buffer for weights so that the backend scheduler won't try to allocate them
                }
            } else {
                buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft); // real buffer
            }
            if (buf == nullptr) {
                throw std::runtime_error(format("unable to allocate %s buffer", ggml_backend_buft_name(buft)));
            }
            if (use_mlock && ggml_backend_buffer_is_host(buf)) {
                pimpl->mlock_bufs.emplace_back(new llama_mlock);
                auto & mlock_buf = pimpl->mlock_bufs.back();
                mlock_buf->init   (ggml_backend_buffer_get_base(buf));
                mlock_buf->grow_to(ggml_backend_buffer_get_size(buf));
            }
            bufs.emplace_back(buf);
            for (uint32_t idx = 0; idx < ml.files.size(); idx++) {
                buf_map.emplace(idx, buf);
            }
        }

        for (auto & buf : bufs) {
            // indicate that this buffer contains weights
            // this is used by ggml_backend_sched to improve op scheduling: ops that use a weight are preferably scheduled to the backend that contains the weight
            ggml_backend_buffer_set_usage(buf.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        }

        pimpl->ctxs_bufs.emplace_back(std::move(ctx_ptr), std::move(bufs));

        ctx_buf_maps.emplace_back(ctx, buf_map);
    }

    if (llama_supports_gpu_offload()) {
        const int n_gpu = std::min(n_gpu_layers, n_layer_all);

        int n_repeating = n_gpu;
        if (n_repeating > 0) {
            LLAMA_LOG_INFO("%s: offloading output layer to GPU\n", __func__);
            n_repeating--;
        }
        LLAMA_LOG_INFO("%s: offloading %d repeating layers to GPU\n", __func__, n_repeating);

        const int max_backend_supported_layers = n_layer_all + 1;
        const int max_offloadable_layers       = n_layer_all + 1;

        LLAMA_LOG_INFO("%s: offloaded %d/%d layers to GPU\n", __func__, std::min(n_gpu_layers, max_offloadable_layers), max_backend_supported_layers);
    }

    // print memory requirements per buffer type
    for (auto & [_, bufs] : pimpl->ctxs_bufs) {
        for (auto & buf: bufs) {
            LLAMA_LOG_INFO("%s: %12s model buffer size = %8.2f MiB\n",
                __func__, ggml_backend_buffer_name(buf.get()), ggml_backend_buffer_get_size(buf.get()) / 1024.0 / 1024.0);
        }
    }

    if (ml.no_alloc) {
        return true;
    }

    // load tensor data
    for (auto & [ctx, buf_map] : ctx_buf_maps) {
        if (!ml.load_all_data(ctx, buf_map, use_mlock ? &pimpl->mlock_mmaps : NULL, params.progress_callback, params.progress_callback_user_data)) {
            return false;
        }
    }

    if (use_mmap_buffer) {
        for (auto & mapping : ml.mappings) {
            pimpl->mappings.emplace_back(std::move(mapping));
        }
    }

    return true;
}

ggml_tensor * llama_model_base::create_tensor(llama_model_loader & ml, const LLM_TN_IMPL & tn, const std::initializer_list<int64_t> & ne, int flags) {
    const buft_list_t * buft_list_layer = tn.bid == -1 ? nullptr : pimpl->dev_layer.at(tn.bid).buft_list;
    return ml.create_tensor(
        hparams, &pimpl->cpu_buft_list, pimpl->dev_input.buft_list, pimpl->dev_output.buft_list, buft_list_layer,
        tn, ne, flags);
}

std::string llama_model::arch_name() const {
    return llm_arch_name(arch);
}

std::string llama_model::type_name() const {
    return llm_type_name(type);
}

std::string llama_model::desc() const {
    return pimpl->desc_str;
}

llama_ftype llama_model::ftype() const {
    return pimpl->ftype;
}

size_t llama_model::size() const {
    return pimpl->n_bytes;
}

size_t llama_model::n_tensors() const {
    return tensors_by_name.size();
}

size_t llama_model::n_devices() const {
    return devices.size();
}

const float * llama_model::tensor_split() const {
    return params.tensor_split;
}

uint32_t llama_model::n_gpu_layers() const {
    // note: plus 1 for the "output" layer
    return params.n_gpu_layers >= 0 ? params.n_gpu_layers : hparams.n_layer_all + 1;
}

llama_split_mode llama_model::split_mode() const {
    return params.split_mode;
}

std::map<ggml_backend_buffer_type_t, size_t> llama_model::memory_breakdown() const {
    std::map<ggml_backend_buffer_type_t, size_t> ret;
    for (const auto & [ctx, bufs] : pimpl->ctxs_bufs) {
        if (hparams.no_alloc) {
            GGML_ASSERT(bufs.size() == 1);
            ggml_backend_buffer_t buf = bufs[0].get();
            GGML_ASSERT(ggml_backend_buffer_get_base(buf) == nullptr);
            ggml_backend_buffer_type_t buft = ggml_backend_buffer_get_type(buf);
            ret[buft] += ggml_backend_alloc_ctx_tensors_from_buft_size(ctx.get(), buft);
        } else {
            for (const auto & buf : bufs) {
                // GGML_ASSERT(ggml_backend_buffer_get_base(buf.get()) != nullptr); // multi_buffer does not have a defined base
                ret[ggml_backend_buffer_get_type(buf.get())] += ggml_backend_buffer_get_size(buf.get());
            }
        }
    }
    return ret;
}

uint64_t llama_model::n_elements() const {
    return pimpl->n_elements;
}

void llama_model::print_info() const {
    const std::string rope_scaling_type = llama_rope_scaling_type_name(hparams.rope_scaling_type_train);

    auto print_f = [](const std::function<int32_t(uint32_t)> & f, uint32_t n) {
        bool is_var = false;

        std::vector<int32_t> v;
        for (uint32_t i = 0; i < n; ++i) {
            v.push_back(f(i));
            if (v[i] != v[0]) {
                is_var = true;
            }
        }

        std::stringstream ss;

        if (is_var) {
            ss << "[";
            for (uint32_t i = 0; i < n; ++i) {
                ss << v[i];
                if (i < n - 1) {
                    ss << ", ";
                }
            }
            ss << "]";
        } else {
            ss << v[0];
        }

        return ss.str();
    };

    // hparams
    LLAMA_LOG_INFO("%s: arch                  = %s\n",     __func__, arch_name().c_str());
    LLAMA_LOG_INFO("%s: vocab_only            = %d\n",     __func__, hparams.vocab_only);
    LLAMA_LOG_INFO("%s: no_alloc              = %d\n",     __func__, hparams.no_alloc);

    if (!hparams.vocab_only) {
        LLAMA_LOG_INFO("%s: n_ctx_train           = %u\n",     __func__, hparams.n_ctx_train);
        LLAMA_LOG_INFO("%s: n_embd_inp            = %u\n",     __func__, hparams.n_embd_inp());
        LLAMA_LOG_INFO("%s: n_embd                = %u\n",     __func__, hparams.n_embd);
        LLAMA_LOG_INFO("%s: n_embd_out            = %u\n",     __func__, hparams.n_embd_out());
        LLAMA_LOG_INFO("%s: n_layer               = %u\n",     __func__, hparams.n_layer());
        LLAMA_LOG_INFO("%s: n_layer_all           = %u\n",     __func__, hparams.n_layer_all);
        LLAMA_LOG_INFO("%s: n_head                = %s\n",     __func__, print_f([&](uint32_t il) { return hparams.n_head(il);    }, hparams.n_layer_all).c_str());
        LLAMA_LOG_INFO("%s: n_head_kv             = %s\n",     __func__, print_f([&](uint32_t il) { return hparams.n_head_kv(il); }, hparams.n_layer_all).c_str());
        LLAMA_LOG_INFO("%s: n_rot                 = %u\n",     __func__, hparams.n_rot_full);
        LLAMA_LOG_INFO("%s: n_swa                 = %u\n",     __func__, hparams.n_swa);
        LLAMA_LOG_INFO("%s: is_swa_any            = %u\n",     __func__, hparams.is_swa_any());
        LLAMA_LOG_INFO("%s: n_embd_head_k         = %u\n",     __func__, hparams.n_embd_head_k_full);
        LLAMA_LOG_INFO("%s: n_embd_head_v         = %u\n",     __func__, hparams.n_embd_head_v_full);
        LLAMA_LOG_INFO("%s: n_gqa                 = %s\n",     __func__, print_f([&](uint32_t il) { return hparams.n_gqa(il);        }, hparams.n_layer_all).c_str());
        LLAMA_LOG_INFO("%s: n_embd_k_gqa          = %s\n",     __func__, print_f([&](uint32_t il) { return hparams.n_embd_k_gqa(il); }, hparams.n_layer_all).c_str());
        LLAMA_LOG_INFO("%s: n_embd_v_gqa          = %s\n",     __func__, print_f([&](uint32_t il) { return hparams.n_embd_v_gqa(il); }, hparams.n_layer_all).c_str());
        LLAMA_LOG_INFO("%s: f_norm_eps            = %.1e\n",   __func__, hparams.f_norm_eps);
        LLAMA_LOG_INFO("%s: f_norm_rms_eps        = %.1e\n",   __func__, hparams.f_norm_rms_eps);
        LLAMA_LOG_INFO("%s: f_clamp_kqv           = %.1e\n",   __func__, hparams.f_clamp_kqv);
        LLAMA_LOG_INFO("%s: f_max_alibi_bias      = %.1e\n",   __func__, hparams.f_max_alibi_bias);
        LLAMA_LOG_INFO("%s: f_logit_scale         = %.1e\n",   __func__, hparams.f_logit_scale);
        LLAMA_LOG_INFO("%s: f_attn_scale          = %.1e\n",   __func__, hparams.f_attention_scale);
        LLAMA_LOG_INFO("%s: f_attn_value_scale    = %.4f\n",   __func__, hparams.f_attn_value_scale);
        LLAMA_LOG_INFO("%s: n_ff                  = %s\n",     __func__, print_f([&](uint32_t il) { return hparams.n_ff(il); }, hparams.n_layer_all).c_str());
        LLAMA_LOG_INFO("%s: n_expert              = %u\n",     __func__, hparams.n_expert);
        LLAMA_LOG_INFO("%s: n_expert_used         = %u\n",     __func__, hparams.n_expert_used);
        LLAMA_LOG_INFO("%s: n_expert_groups       = %d\n",     __func__, hparams.n_expert_groups);
        LLAMA_LOG_INFO("%s: n_group_used          = %d\n",     __func__, hparams.n_group_used);
        LLAMA_LOG_INFO("%s: causal attn           = %d\n",     __func__, hparams.causal_attn);
        LLAMA_LOG_INFO("%s: pooling type          = %d\n",     __func__, hparams.pooling_type);
        LLAMA_LOG_INFO("%s: rope type             = %d\n",     __func__, hparams.rope_type);
        LLAMA_LOG_INFO("%s: rope scaling          = %s\n",     __func__, rope_scaling_type.c_str());
        LLAMA_LOG_INFO("%s: freq_base_train       = %.1f\n",   __func__, hparams.rope_freq_base_train);
        LLAMA_LOG_INFO("%s: freq_scale_train      = %g\n",     __func__, hparams.rope_freq_scale_train);
        if (hparams.swa_type != LLAMA_SWA_TYPE_NONE) {
            LLAMA_LOG_INFO("%s: freq_base_swa         = %.1f\n",   __func__, hparams.rope_freq_base_train_swa);
            LLAMA_LOG_INFO("%s: freq_scale_swa        = %g\n",     __func__, hparams.rope_freq_scale_train_swa);
            LLAMA_LOG_INFO("%s: n_embd_head_k_swa     = %u\n",     __func__, hparams.n_embd_head_k_swa);
            LLAMA_LOG_INFO("%s: n_embd_head_v_swa     = %u\n",     __func__, hparams.n_embd_head_v_swa);
            LLAMA_LOG_INFO("%s: n_rot_swa             = %u\n",     __func__, hparams.n_rot_swa);
        }
        LLAMA_LOG_INFO("%s: n_ctx_orig_yarn       = %u\n",     __func__, hparams.n_ctx_orig_yarn);
        LLAMA_LOG_INFO("%s: rope_yarn_log_mul     = %.4f\n",   __func__, hparams.rope_yarn_log_mul);
        LLAMA_LOG_INFO("%s: rope_finetuned        = %s\n",     __func__, hparams.rope_finetuned ? "yes" : "unknown");
        if (arch == LLM_ARCH_GRANITE &&
            std::any_of(hparams.deepstack_mapping_arr.begin(),
                        hparams.deepstack_mapping_arr.end(),
                        [](const auto & entry) { return entry >= 0; })) {
            LLAMA_LOG_INFO("%s: deepstack_mapping_arr = %s\n", __func__,
                           print_f([&](uint32_t il) { return hparams.deepstack_mapping_arr[il]; },
                           hparams.n_layer_all).c_str());
        }
        // MRoPE (Multi-axis Rotary Position Embedding) sections
        if (const auto & s = hparams.rope_sections; s[0] || s[1] || s[2] || s[3]) {
            LLAMA_LOG_INFO("%s: mrope sections        = [%d, %d, %d, %d]\n", __func__, s[0], s[1], s[2], s[3]);
        }
        if (!classifier_labels.empty()) {
            LLAMA_LOG_INFO("%s: n_cls_out             = %u\n", __func__, hparams.n_cls_out);

            size_t i = 0;
            for (const auto & label : classifier_labels) {
                LLAMA_LOG_INFO("%s: cls_label[%2zu]         = %s\n", __func__, i++, label.c_str());
            }
        }

        if (arch == LLM_ARCH_MAMBA ||
                arch == LLM_ARCH_MAMBA2 ||
                arch == LLM_ARCH_JAMBA ||
                arch == LLM_ARCH_FALCON_H1 ||
                arch == LLM_ARCH_PLAMO2 ||
                arch == LLM_ARCH_GRANITE_HYBRID ||
                arch == LLM_ARCH_QWEN3NEXT ||
                arch == LLM_ARCH_QWEN35 ||
                arch == LLM_ARCH_QWEN35MOE ||
                arch == LLM_ARCH_NEMOTRON_H ||
                arch == LLM_ARCH_NEMOTRON_H_MOE) {
            LLAMA_LOG_INFO("%s: ssm_d_conv            = %u\n",     __func__, hparams.ssm_d_conv);
            LLAMA_LOG_INFO("%s: ssm_d_inner           = %u\n",     __func__, hparams.ssm_d_inner);
            LLAMA_LOG_INFO("%s: ssm_d_state           = %u\n",     __func__, hparams.ssm_d_state);
            LLAMA_LOG_INFO("%s: ssm_dt_rank           = %u\n",     __func__, hparams.ssm_dt_rank);
            LLAMA_LOG_INFO("%s: ssm_n_group           = %u\n",     __func__, hparams.ssm_n_group);
            LLAMA_LOG_INFO("%s: ssm_dt_b_c_rms        = %d\n",     __func__, hparams.ssm_dt_b_c_rms);
        }

        LLAMA_LOG_INFO("%s: model type            = %s\n",     __func__, type_name().c_str());
        if (pimpl->n_elements >= 1e12) {
            LLAMA_LOG_INFO("%s: model params          = %.2f T\n", __func__, pimpl->n_elements*1e-12);
        } else if (pimpl->n_elements >= 1e9) {
            LLAMA_LOG_INFO("%s: model params          = %.2f B\n", __func__, pimpl->n_elements*1e-9);
        } else if (pimpl->n_elements >= 1e6) {
            LLAMA_LOG_INFO("%s: model params          = %.2f M\n", __func__, pimpl->n_elements*1e-6);
        } else {
            LLAMA_LOG_INFO("%s: model params          = %.2f K\n", __func__, pimpl->n_elements*1e-3);
        }

        // general kv
        LLAMA_LOG_INFO("%s: general.name          = %s\n",    __func__, name.c_str());

        if (arch == LLM_ARCH_DEEPSEEK) {
            LLAMA_LOG_INFO("%s: n_layer_dense_lead    = %d\n",     __func__, hparams.n_layer_dense_lead);
            LLAMA_LOG_INFO("%s: n_ff_exp              = %d\n",     __func__, hparams.n_ff_exp);
            LLAMA_LOG_INFO("%s: n_expert_shared       = %d\n",     __func__, hparams.n_expert_shared);
            LLAMA_LOG_INFO("%s: expert_weights_scale  = %.1f\n",   __func__, hparams.expert_weights_scale);
        }

        if (arch == LLM_ARCH_DEEPSEEK2 || arch == LLM_ARCH_DEEPSEEK2OCR || arch == LLM_ARCH_DEEPSEEK32 || arch == LLM_ARCH_GLM_DSA || arch == LLM_ARCH_MISTRAL4) {
            LLAMA_LOG_INFO("%s: n_layer_dense_lead    = %d\n",     __func__, hparams.n_layer_dense_lead);
            LLAMA_LOG_INFO("%s: n_lora_q              = %d\n",     __func__, hparams.n_lora_q);
            LLAMA_LOG_INFO("%s: n_lora_kv             = %d\n",     __func__, hparams.n_lora_kv);
            LLAMA_LOG_INFO("%s: n_embd_head_k_mla     = %d\n",     __func__, hparams.n_embd_head_k_mla());
            LLAMA_LOG_INFO("%s: n_embd_head_v_mla     = %d\n",     __func__, hparams.n_embd_head_v_mla());
            LLAMA_LOG_INFO("%s: n_ff_exp              = %d\n",     __func__, hparams.n_ff_exp);
            LLAMA_LOG_INFO("%s: n_expert_shared       = %d\n",     __func__, hparams.n_expert_shared);
            LLAMA_LOG_INFO("%s: expert_weights_scale  = %.1f\n",   __func__, hparams.expert_weights_scale);
            LLAMA_LOG_INFO("%s: expert_weights_norm   = %d\n",     __func__, hparams.expert_weights_norm);
            LLAMA_LOG_INFO("%s: expert_gating_func    = %s\n",     __func__, llama_expert_gating_func_name((llama_expert_gating_func_type) hparams.expert_gating_func));
        }

        if (arch == LLM_ARCH_QWEN2MOE) {
            LLAMA_LOG_INFO("%s: n_ff_exp              = %d\n",     __func__, hparams.n_ff_exp);
            LLAMA_LOG_INFO("%s: n_ff_shexp            = %d\n",     __func__, hparams.n_ff_shexp);
        }

        if (arch == LLM_ARCH_MELLUM ||
                arch == LLM_ARCH_COHERE2MOE ||
                arch == LLM_ARCH_QWEN3MOE ||
                arch == LLM_ARCH_OPENAI_MOE ||
                arch == LLM_ARCH_QWEN3VLMOE ||
                arch == LLM_ARCH_RND1) {
            LLAMA_LOG_INFO("%s: n_ff_exp              = %d\n",     __func__, hparams.n_ff_exp);
        }

        if (arch == LLM_ARCH_MINICPM ||
                arch == LLM_ARCH_GRANITE ||
                arch == LLM_ARCH_GRANITE_MOE ||
                arch == LLM_ARCH_GRANITE_HYBRID ||
                arch == LLM_ARCH_GRANITE_SWITCH ||
                arch == LLM_ARCH_NEMOTRON_H_MOE) {
            LLAMA_LOG_INFO("%s: f_embedding_scale     = %f\n", __func__, hparams.f_embedding_scale);
            LLAMA_LOG_INFO("%s: f_residual_scale      = %f\n", __func__, hparams.f_residual_scale);
            LLAMA_LOG_INFO("%s: f_attention_scale     = %f\n", __func__, hparams.f_attention_scale);
            LLAMA_LOG_INFO("%s: n_ff_shexp            = %d\n", __func__, hparams.n_ff_shexp);
        }

        if (arch == LLM_ARCH_BAILINGMOE) {
            LLAMA_LOG_INFO("%s: n_layer_dense_lead    = %d\n",     __func__, hparams.n_layer_dense_lead);
            LLAMA_LOG_INFO("%s: n_ff_exp              = %d\n",     __func__, hparams.n_ff_exp);
            LLAMA_LOG_INFO("%s: n_expert_shared       = %d\n",     __func__, hparams.n_expert_shared);
            LLAMA_LOG_INFO("%s: expert_weights_scale  = %.1f\n",   __func__, hparams.expert_weights_scale);
            LLAMA_LOG_INFO("%s: expert_weights_norm   = %d\n",     __func__, hparams.expert_weights_norm);
        }

        if (arch == LLM_ARCH_BAILINGMOE2 || arch == LLM_ARCH_BAILINGMOE3) {
            LLAMA_LOG_INFO("%s: n_layer_dense_lead    = %d\n",     __func__, hparams.n_layer_dense_lead);
            LLAMA_LOG_INFO("%s: n_ff_exp              = %d\n",     __func__, hparams.n_ff_exp);
            LLAMA_LOG_INFO("%s: n_ff_shexp            = %d\n",     __func__, hparams.n_ff_shexp);
            LLAMA_LOG_INFO("%s: n_expert_shared       = %d\n",     __func__, hparams.n_expert_shared);
            LLAMA_LOG_INFO("%s: expert_weights_scale  = %.1f\n",   __func__, hparams.expert_weights_scale);
            LLAMA_LOG_INFO("%s: expert_weights_norm   = %d\n",     __func__, hparams.expert_weights_norm);
            LLAMA_LOG_INFO("%s: expert_gating_func    = %s\n",     __func__, llama_expert_gating_func_name((llama_expert_gating_func_type) hparams.expert_gating_func));
            LLAMA_LOG_INFO("%s: n_layer_nextn         = %d\n",     __func__, hparams.n_layer_nextn);
        }

        if (arch == LLM_ARCH_SMALLTHINKER || arch == LLM_ARCH_LFM2MOE) {
            LLAMA_LOG_INFO("%s: n_ff_exp              = %d\n",     __func__, hparams.n_ff_exp);
            LLAMA_LOG_INFO("%s: expert_gating_func    = %s\n",     __func__, llama_expert_gating_func_name((llama_expert_gating_func_type) hparams.expert_gating_func));
        }

        if (arch == LLM_ARCH_GROVEMOE) {
            LLAMA_LOG_INFO("%s: n_ff_exp              = %d\n",     __func__, hparams.n_ff_exp);
            LLAMA_LOG_INFO("%s: n_ff_chexp            = %d\n",     __func__, hparams.n_ff_chexp);
            LLAMA_LOG_INFO("%s: n_group_experts       = %d\n",     __func__, hparams.n_group_experts);
            LLAMA_LOG_INFO("%s: expert_group_scale    = %.2f\n",   __func__, hparams.expert_group_scale);
        }
    }

    vocab.print_info();
}

ggml_backend_dev_t llama_model::dev_layer(int il) const {
    return pimpl->dev_layer.at(il).dev;
}

ggml_backend_dev_t llama_model::dev_output() const {
    return pimpl->dev_output.dev;
}

template<typename F>
static bool buft_supported(ggml_backend_buffer_type_t buft, ggml_backend_dev_t dev, F & fn) {
    ggml_init_params params = {
        /*.mem_size   =*/ ggml_tensor_overhead()*8,
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true,
    };

    ggml_context_ptr ctx { ggml_init(params) };
    if (!ctx) {
        throw std::runtime_error(format("failed to create ggml context"));
    }

    ggml_backend_buffer_ptr buf { ggml_backend_buft_alloc_buffer(buft, 0) };
    ggml_tensor * op_tensor = fn(ctx.get());
    for (int i = 0; i < GGML_MAX_SRC; i++) {
        if (op_tensor->src[i] != nullptr) {
            assert(op_tensor->src[i]->buffer == nullptr);
            op_tensor->src[i]->buffer = buf.get();
        }
    }

    bool op_supported = ggml_backend_dev_supports_op(dev, op_tensor);

    return op_supported;
}

template<typename F>
static ggml_backend_buffer_type_t select_buft(const buft_list_t & buft_list, const F & fn) {
    for (const auto & cur : buft_list) {
        ggml_backend_dev_t cur_dev = cur.first;
        ggml_backend_buffer_type_t cur_buft = cur.second;
        if (buft_supported(cur_buft, cur_dev, fn)) {
            return cur_buft;
        }
    }

    throw std::runtime_error(format("no suitable buffer type found"));
}

ggml_backend_buffer_type_t llama_model::select_buft(int il) const {
    return ::select_buft(
            *pimpl->dev_layer.at(il).buft_list,
            [&](ggml_context * ctx) {
                ggml_tensor * cur = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hparams.n_embd);
                ggml_tensor * layer_dir = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hparams.n_embd);
                return ggml_add(ctx, cur, layer_dir);
            });
}

bool llama_model::has_tensor_overrides() const {
    return pimpl->has_tensor_overrides;
}

const ggml_tensor * llama_model::get_tensor(const char * name) const {
    auto it = std::find_if(tensors_by_name.begin(), tensors_by_name.end(),
            [name](const std::pair<std::string, ggml_tensor *> & it) {
                return it.first == name;
            });
    if (it == tensors_by_name.end()) {
        return nullptr;
    }

    return it->second;
}

float llama_model::get_rope_freq_base (const llama_cparams & cparams, int il) const {
    return hparams.is_swa(il) ? hparams.rope_freq_base_train_swa : cparams.rope_freq_base;
}

float llama_model::get_rope_freq_scale(const llama_cparams & cparams, int il) const {
    return hparams.is_swa(il) ? hparams.rope_freq_scale_train_swa : cparams.rope_freq_scale;
}

ggml_tensor * llama_model::get_rope_factors(const llama_cparams & cparams, int il) const {
    const uint32_t n_ctx_seq = cparams.n_ctx_seq;

    // choose long/short freq factors based on the context size
    if (layers[il].rope_freqs != nullptr) {
        return layers[il].rope_freqs;
    }

    if (n_ctx_seq > hparams.n_ctx_orig_yarn) {
        return layers[il].rope_long;
    }

    return layers[il].rope_short;
}

llama_memory_i * llama_model::create_memory(const llama_memory_params & params, const llama_cparams & cparams) const {
    llama_memory_i * res;

    switch (arch) {
        // Models that need specific instantiation should be handled in the
        // switch statement
        case LLM_ARCH_BERT:
        case LLM_ARCH_JINA_BERT_V2:
        case LLM_ARCH_JINA_BERT_V3:
        case LLM_ARCH_NOMIC_BERT:
        case LLM_ARCH_NOMIC_BERT_MOE:
        case LLM_ARCH_NEO_BERT:
        case LLM_ARCH_EUROBERT:
        case LLM_ARCH_WAVTOKENIZER_DEC:
        case LLM_ARCH_MODERN_BERT:
        case LLM_ARCH_GEMMA_EMBEDDING:
        case LLM_ARCH_DREAM:
        case LLM_ARCH_LLADA:
        case LLM_ARCH_LLADA_MOE:
        case LLM_ARCH_RND1:
            {
                res = nullptr;
            } break;
        case LLM_ARCH_MINIMAX_M3:
            {
                // sparse (MSA) layers carry an indexer key cache, but leading dense layers do not
                llama_kv_cache::layer_filter_cb filter_idx =
                    [&](int32_t il) { return (uint32_t) il >= hparams.n_layer_dense_lead; };

                res = new llama_kv_cache_msa(
                        *this,
                        params.type_k,
                        params.type_v,
                        !cparams.flash_attn,
                        cparams.offload_kqv,
                        cparams.kv_unified,
                        cparams.n_ctx_seq,
                        cparams.n_seq_max,
                        1,
                        hparams.n_swa,
                        hparams.swa_type,
                        nullptr,
                        filter_idx,
                        nullptr);
            } break;
        case LLM_ARCH_GLM_DSA:
        case LLM_ARCH_DEEPSEEK32:
            {
                if (params.ctx_type == LLAMA_CONTEXT_TYPE_MTP && hparams.n_layer_nextn > 0) {
                    // The NextN/MTP draft head runs dense MLA (no DSA indexer), so the
                    // MTP context uses a plain attention KV cache holding only the
                    // nextn layer(s) - same pattern as the hybrid Qwen3.5 MTP context.
                    llama_kv_cache::layer_filter_cb filter =
                        [&](uint32_t il) { return il >= hparams.n_layer(); };

                    res = new llama_kv_cache(
                            *this,
                            hparams,
                            params.type_k,
                            params.type_v,
                            !cparams.flash_attn,
                            cparams.offload_kqv,
                            cparams.kv_unified,
                            cparams.n_ctx_seq,
                            cparams.n_seq_max,
                            1,
                            hparams.n_swa,
                            hparams.swa_type,
                            nullptr,
                            filter,
                            nullptr,
                            nullptr);
                } else {
                    // Main context: DSA cache for the trunk layers only - the nextn
                    // layer(s) are never attended by the trunk graph.
                    llama_kv_cache::layer_filter_cb filter_mla = nullptr;
                    if (hparams.n_layer_nextn > 0) {
                        filter_mla = [&](uint32_t il) { return il < hparams.n_layer(); };
                    }
                    llama_kv_cache::layer_filter_cb filter_lid = [&](uint32_t il) { return il < hparams.n_layer() && (arch != LLM_ARCH_GLM_DSA || hparams.is_indexer_full(il)); };

                    res = new llama_kv_cache_dsa(
                            *this,
                            params.type_k,
                            params.type_v,
                            !cparams.flash_attn,
                            cparams.offload_kqv,
                            cparams.kv_unified,
                            cparams.n_ctx_seq,
                            cparams.n_seq_max,
                            1,
                            hparams.n_swa,
                            hparams.swa_type,
                            filter_mla,
                            filter_lid,
                            nullptr);
                }
            } break;
        case LLM_ARCH_DEEPSEEK4:
            {
                GGML_ASSERT(hparams.swa_type != LLAMA_SWA_TYPE_NONE);

                if (params.ctx_type == LLAMA_CONTEXT_TYPE_MTP) {
                    const llama_memory_i::layer_filter_cb filter_mtp = [&](int32_t il) {
                        return il >= (int32_t) hparams.n_layer();
                    };

                    res = new llama_kv_cache_iswa(
                            *this,
                            params.type_k,
                            params.type_v,
                            !cparams.flash_attn,
                            cparams.offload_kqv,
                            params.swa_full,
                            cparams.kv_unified,
                            cparams.n_ctx_seq,
                            cparams.n_seq_max,
                            cparams.n_ubatch,
                            1,
                            nullptr,
                            filter_mtp,
                            nullptr,
                            nullptr);
                } else {
                    res = new llama_kv_cache_dsv4(
                            *this,
                            params.type_k,
                            params.type_v,
                            !cparams.flash_attn,
                            cparams.offload_kqv,
                            params.swa_full,
                            cparams.kv_unified,
                            cparams.n_ctx_seq,
                            cparams.n_seq_max,
                            cparams.n_ubatch,
                            1,
                            cparams.n_rs_seq,
                            nullptr,
                            nullptr);
                }
            } break;
        case LLM_ARCH_DFLASH:
            {
                // DSV4 DSpark stages store a single MLA-style K per position (window = the draft ring)
                if (hparams.dsv4_hc_mult > 0) {
                    GGML_ASSERT(hparams.swa_type != LLAMA_SWA_TYPE_NONE);

                    res = new llama_kv_cache_iswa(
                            *this,
                            params.type_k,
                            params.type_v,
                            !cparams.flash_attn,
                            cparams.offload_kqv,
                            params.swa_full,
                            cparams.kv_unified,
                            cparams.n_ctx_seq,
                            cparams.n_seq_max,
                            cparams.n_ubatch,
                            1,
                            nullptr,
                            nullptr,
                            nullptr,
                            nullptr);
                    break;
                }
            }
            [[fallthrough]];
        // Models that need standard caching should rely on recurrent/hybrid
        // checks
        default:
            {
                // Dense MTP heads use a plain attention KV cache instead of the hybrid wrapper.
                const bool mtp_on_hybrid_qwen =
                    params.ctx_type == LLAMA_CONTEXT_TYPE_MTP &&
                    (arch == LLM_ARCH_QWEN3NEXT || arch == LLM_ARCH_QWEN35 || arch == LLM_ARCH_QWEN35MOE ||
                     arch == LLM_ARCH_BAILINGMOE3);

                const bool mtp_on_hybrid_nemotron =
                    params.ctx_type == LLAMA_CONTEXT_TYPE_MTP && arch == LLM_ARCH_NEMOTRON_H_MOE;

                if (llm_arch_is_recurrent(arch)) {
                    res = new llama_memory_recurrent(
                            *this,
                            GGML_TYPE_F32,
                            GGML_TYPE_F32,
                            cparams.offload_kqv,
                            std::max((uint32_t) 1, cparams.n_seq_max),
                            cparams.n_seq_max,
                            cparams.n_rs_seq,
                            nullptr);
                } else if (llm_arch_is_hybrid(arch) && !mtp_on_hybrid_qwen && !mtp_on_hybrid_nemotron) {
                    // The main difference between hybrid architectures is the
                    // layer filters, so pick the right one here
                    llama_memory_hybrid::layer_filter_cb filter_attn = nullptr;
                    llama_memory_hybrid::layer_filter_cb filter_recr = nullptr;
                    if (arch == LLM_ARCH_FALCON_H1) {
                        filter_attn = [&](uint32_t) { return true; };
                        filter_recr = [&](uint32_t) { return true; };
                    } else if (arch == LLM_ARCH_NEMOTRON_H || arch == LLM_ARCH_NEMOTRON_H_MOE) {
                        filter_attn = [&](uint32_t il) {
                            return !hparams.is_recr(il) && hparams.n_ff(il) == 0;
                        };
                        filter_recr = [&](uint32_t il) {
                            return hparams.is_recr(il) && hparams.n_ff(il) == 0;
                        };
                    } else if (arch == LLM_ARCH_QWEN3NEXT || arch == LLM_ARCH_QWEN35 || arch == LLM_ARCH_QWEN35MOE || arch == LLM_ARCH_MINIMAX_01) {
                        filter_attn = [&](uint32_t il) {
                            return il < hparams.n_layer() && !hparams.is_recr(il);
                        };
                        filter_recr = [&](uint32_t il) {
                            return il < hparams.n_layer() && hparams.is_recr(il);
                        };
                    }

                    if (hparams.swa_type != LLAMA_SWA_TYPE_NONE) {
                        // Use hybrid-iswa for hybrid models with SWA
                        res = new llama_memory_hybrid_iswa(
                            /* model             */ *this,
                            /* attn_type_k       */ params.type_k,
                            /* attn_type_v       */ params.type_v,
                            /* attn_v_trans      */ !cparams.flash_attn,
                            /* attn_swa_full     */ params.swa_full,
                            /* attn_kv_size      */ cparams.n_ctx_seq,
                            /* attn_n_ubatch     */ cparams.n_ubatch,
                            /* attn_n_pad        */ 1,
                            /* recurrent_type_r  */ GGML_TYPE_F32,
                            /* recurrent_type_s  */ GGML_TYPE_F32,
                            /* recurrent_rs_size */ std::max((uint32_t) 1, cparams.n_seq_max),
                            /* n_seq_max         */ cparams.n_seq_max,
                            /* n_rs_seq          */ cparams.n_rs_seq,
                            /* offload           */ cparams.offload_kqv,
                            /* unified           */ cparams.kv_unified,
                            /* filter_attn       */ std::move(filter_attn),
                            /* filter_recr       */ std::move(filter_recr));
                    } else {
                        res = new llama_memory_hybrid(
                            /* model             */ *this,
                            /* attn_type_k       */ params.type_k,
                            /* attn_type_v       */ params.type_v,
                            /* attn_v_trans      */ !cparams.flash_attn,
                            /* attn_kv_size      */ cparams.n_ctx_seq,
                            /* attn_n_pad        */ 1,
                            /* attn_n_swa        */ hparams.n_swa,
                            /* attn_swa_type     */ hparams.swa_type,
                            /* recurrent_type_k  */ GGML_TYPE_F32,
                            /* recurrent_type_v  */ GGML_TYPE_F32,
                            /* recurrent_kv_size */ std::max((uint32_t) 1, cparams.n_seq_max),
                            /* n_seq_max         */ cparams.n_seq_max,
                            /* n_rs_seq          */ cparams.n_rs_seq,
                            /* offload           */ cparams.offload_kqv,
                            /* unified           */ cparams.kv_unified,
                            /* filter_attn       */ std::move(filter_attn),
                            /* filter_recr       */ std::move(filter_recr));
                    }
                } else {
                    llama_kv_cache::layer_filter_cb filter = nullptr;
                    llama_memory_i::layer_reuse_cb reuse = nullptr;
                    llama_kv_cache::layer_share_cb share = nullptr;

                    if (arch == LLM_ARCH_GEMMA3N || arch == LLM_ARCH_GEMMA4) {
                        reuse = [&](uint32_t il) {
                            GGML_ASSERT(hparams.n_layer_kv_from_start >= 2);

                            if (il >= (uint32_t)hparams.n_layer_kv_from_start) {
                                return hparams.n_layer_kv_from_start - (hparams.is_swa(il) ? 2 : 1);
                            }

                            return -1;
                        };
                    }

                    if (mtp_on_hybrid_qwen || mtp_on_hybrid_nemotron) {
                        filter = [&](uint32_t il) { return il >= hparams.n_layer(); };
                    }

                    if ((arch == LLM_ARCH_STEP35 || arch == LLM_ARCH_HY_V3 || arch == LLM_ARCH_GLM_DSA ||
                            arch == LLM_ARCH_MIMO2 || arch == LLM_ARCH_DEEPSEEK32) &&
                            hparams.n_layer_nextn > 0) {
                        if (params.ctx_type == LLAMA_CONTEXT_TYPE_MTP) {
                            filter = [&](uint32_t il) { return il >= hparams.n_layer(); };
                        } else {
                            filter = [&](uint32_t il) { return il <  hparams.n_layer(); };
                        }
                    }

                    if (hparams.swa_type != LLAMA_SWA_TYPE_NONE) {
                        GGML_ASSERT(hparams.is_swa_any());

                        if (arch == LLM_ARCH_GEMMA4_ASSISTANT) {
                            llama_memory_t mem_other = llama_get_memory(cparams.ctx_other);

                            share = [&](int32_t il) {
                                const llama_model * model_other = llama_get_model(cparams.ctx_other);

                                if (hparams.is_swa(il)) {
                                    return llama_model_n_layer(model_other) - 2;
                                }

                                return llama_model_n_layer(model_other) - 1;
                            };

                            res = new llama_kv_cache_iswa(
                                    *this,
                                    params.type_k,
                                    params.type_v,
                                    !cparams.flash_attn,
                                    cparams.offload_kqv,
                                    params.swa_full,
                                    cparams.kv_unified,
                                    cparams.n_ctx_seq,
                                    cparams.n_seq_max,
                                    cparams.n_ubatch,
                                    1,
                                    mem_other,
                                    filter,
                                    reuse,
                                    share);
                        } else {
                            res = new llama_kv_cache_iswa(
                                    *this,
                                    params.type_k,
                                    params.type_v,
                                    !cparams.flash_attn,
                                    cparams.offload_kqv,
                                    params.swa_full,
                                    cparams.kv_unified,
                                    cparams.n_ctx_seq,
                                    cparams.n_seq_max,
                                    cparams.n_ubatch,
                                    1,
                                    nullptr,
                                    filter,
                                    reuse,
                                    share);
                        }
                    } else {
                        GGML_ASSERT(!hparams.is_swa_any());

                        res = new llama_kv_cache(
                                *this,
                                hparams,
                                params.type_k,
                                params.type_v,
                                !cparams.flash_attn,
                                cparams.offload_kqv,
                                cparams.kv_unified,
                                cparams.n_ctx_seq,
                                cparams.n_seq_max,
                                1,
                                hparams.n_swa,
                                hparams.swa_type,
                                nullptr,
                                filter,
                                nullptr,
                                nullptr);
                    }
                }
            }
    }

    return res;
}

ggml_cgraph * llama_model::build_graph(const llm_graph_params & params) const {
    std::unique_ptr<llm_graph_context> llm = build_arch_graph(params);

    // add on pooling layer
    llm->build_pooling(cls, cls_b, cls_out, cls_out_b, cls_norm);

    // add backend sampling layers (if any)
    llm->build_sampling();

    // if the gguf model was converted with --sentence-transformers-dense-modules
    // there will be two additional dense projection layers
    // dense linear projections are applied after pooling
    // TODO: move reranking logic here and generalize
    llm->build_dense_out(dense_2_out_layers, dense_2_out_layers_b, dense_3_out_layers);

    llm->res->set_outputs(params);

    return llm->res->get_gf();
}


//
// interface implementation
//

llama_model_params llama_model_default_params() {
    llama_model_params result = {
        /*.devices                     =*/ nullptr,
        /*.tensor_buft_overrides       =*/ nullptr,
        /*.n_gpu_layers                =*/ -1,
        /*.split_mode                  =*/ LLAMA_SPLIT_MODE_LAYER,
        /*.load_mode                   =*/ LLAMA_LOAD_MODE_AUTO,
        /*.main_gpu                    =*/ 0,
        /*.tensor_split                =*/ nullptr,
        /*.progress_callback           =*/ nullptr,
        /*.progress_callback_user_data =*/ nullptr,
        /*.kv_overrides                =*/ nullptr,
        /*.vocab_only                  =*/ false,
        /*.check_tensors               =*/ false,
        /*.use_extra_bufts             =*/ true,
        /*.no_host                     =*/ false,
        /*.no_alloc                    =*/ false,
        /*.load_mtp                    =*/ false,
    };

    return result;
}

const llama_vocab * llama_model_get_vocab(const llama_model * model) {
    return &model->vocab;
}

void llama_free_model(llama_model * model) {
    llama_model_free(model);
}

void llama_model_free(llama_model * model) {
    delete model;
}

int32_t llama_model_n_ctx_train(const llama_model * model) {
    return model->hparams.n_ctx_train;
}

int32_t llama_model_n_embd(const llama_model * model) {
    return model->hparams.n_embd;
}

int32_t llama_model_n_embd_inp(const llama_model * model) {
    return model->hparams.n_embd_inp();
}

int32_t llama_model_n_embd_out(const llama_model * model) {
    return model->hparams.n_embd_out();
}

int32_t llama_model_n_embd_h(const llama_model * model) {
    return model->hparams.n_embd_h();
}

int32_t llama_model_n_layer(const llama_model * model) {
    return model->hparams.n_layer();
}

int32_t llama_model_n_layer_nextn(const llama_model * model) {
    return model->hparams.n_layer_nextn;
}

int32_t llama_model_n_head(const llama_model * model) {
    return model->hparams.n_head();
}

int32_t llama_model_n_head_kv(const llama_model * model) {
    return model->hparams.n_head_kv();
}

int32_t llama_model_n_swa(const llama_model * model) {
    // dsv4 kv-cache has SWA but it cannot be used as a rollback because of
    // other compression ratios, so we return 0 here
    if (model->arch == LLM_ARCH_DEEPSEEK4) {
        return 0;
    }
    return model->hparams.n_swa;
}


uint32_t llama_model_n_cls_out(const struct llama_model * model) {
    return model->hparams.n_cls_out;
}

const char * llama_model_cls_label(const struct llama_model * model, uint32_t i) {
    if (i < model->classifier_labels.size()) {
        return model->classifier_labels[i].c_str();
    }

    return nullptr;
}

// deprecated
int32_t llama_n_ctx_train(const llama_model * model) {
    return llama_model_n_ctx_train(model);
}

// deprecated
int32_t llama_n_embd(const llama_model * model) {
    return llama_model_n_embd(model);
}

// deprecated
int32_t llama_n_layer(const llama_model * model) {
    return llama_model_n_layer(model);
}

// deprecated
int32_t llama_n_head(const llama_model * model) {
    return llama_model_n_head(model);
}

llama_rope_type llama_model_rope_type(const llama_model * model) {
    switch (model->arch) {
        // these models do not use RoPE
        case LLM_ARCH_CLIP:
        case LLM_ARCH_GPT2:
        case LLM_ARCH_GPTJ:
        case LLM_ARCH_MPT:
        case LLM_ARCH_REFACT:
        case LLM_ARCH_BLOOM:
        case LLM_ARCH_MAMBA:
        case LLM_ARCH_MAMBA2:
        case LLM_ARCH_JAMBA:
        case LLM_ARCH_JINA_BERT_V2:
        case LLM_ARCH_T5:
        case LLM_ARCH_T5ENCODER:
        case LLM_ARCH_JAIS:
        case LLM_ARCH_RWKV6:
        case LLM_ARCH_RWKV6QWEN2:
        case LLM_ARCH_RWKV7:
        case LLM_ARCH_ARWKV7:
        case LLM_ARCH_WAVTOKENIZER_DEC:
        case LLM_ARCH_NEMOTRON_H:
        case LLM_ARCH_NEMOTRON_H_MOE:
        case LLM_ARCH_KIMI_LINEAR:
        case LLM_ARCH_KIMI_K3:
            return LLAMA_ROPE_TYPE_NONE;

        // use what we call a normal RoPE, operating on pairs of consecutive head values
        case LLM_ARCH_LLAMA:
        case LLM_ARCH_LLADA:
        case LLM_ARCH_LLAMA4:
        case LLM_ARCH_DECI:
        case LLM_ARCH_BAICHUAN:
        case LLM_ARCH_STARCODER:
        case LLM_ARCH_INTERNLM2:
        case LLM_ARCH_MINICPM:
        case LLM_ARCH_XVERSE:
        case LLM_ARCH_COMMAND_R:
        case LLM_ARCH_COHERE2:
        case LLM_ARCH_COHERE2MOE:
        case LLM_ARCH_OLMO:
        case LLM_ARCH_ARCTIC:
        case LLM_ARCH_DEEPSEEK:
        case LLM_ARCH_DEEPSEEK2:
        case LLM_ARCH_DEEPSEEK2OCR:
        case LLM_ARCH_DEEPSEEK32:
        case LLM_ARCH_DEEPSEEK4:
        case LLM_ARCH_MUSE_GLIMMER:
        case LLM_ARCH_PLM:
        case LLM_ARCH_CHATGLM:
        case LLM_ARCH_GRANITE:
        case LLM_ARCH_GRANITE_MOE:
        case LLM_ARCH_GRANITE_HYBRID:
        case LLM_ARCH_GRANITE_SWITCH:
        case LLM_ARCH_GRANITE_SWA:
        case LLM_ARCH_CHAMELEON:
        case LLM_ARCH_BAILINGMOE:
        case LLM_ARCH_BAILINGMOE3:
        case LLM_ARCH_NEO_BERT:
        case LLM_ARCH_SMOLLM3:
        case LLM_ARCH_ARCEE:
        case LLM_ARCH_ERNIE4_5:
        case LLM_ARCH_ERNIE4_5_MOE:
        case LLM_ARCH_MISTRAL3:
        case LLM_ARCH_EAGLE3:
        case LLM_ARCH_MISTRAL4:
        case LLM_ARCH_LLAMA_EMBED:
        case LLM_ARCH_MAINCODER:
        case LLM_ARCH_GLM_DSA:
        case LLM_ARCH_NANBEIGE:
        case LLM_ARCH_POCKETTTS:
            return LLAMA_ROPE_TYPE_NORM;

        // the pairs of head values are offset by n_rot/2
        case LLM_ARCH_FALCON:
        case LLM_ARCH_FALCON_H1:
        case LLM_ARCH_GROK:
        case LLM_ARCH_DBRX:
        case LLM_ARCH_BERT:
        case LLM_ARCH_JINA_BERT_V3:
        case LLM_ARCH_MODERN_BERT:
        case LLM_ARCH_NOMIC_BERT:
        case LLM_ARCH_NOMIC_BERT_MOE:
        case LLM_ARCH_EUROBERT:
        case LLM_ARCH_STABLELM:
        case LLM_ARCH_BITNET:
        case LLM_ARCH_QWEN:
        case LLM_ARCH_QWEN2:
        case LLM_ARCH_DREAM:
        case LLM_ARCH_QWEN2MOE:
        case LLM_ARCH_QWEN3:
        case LLM_ARCH_QWEN3MOE:
        case LLM_ARCH_LLADA_MOE:
        case LLM_ARCH_RND1:
        case LLM_ARCH_OLMO2:
        case LLM_ARCH_OLMOE:
        case LLM_ARCH_PHI2:
        case LLM_ARCH_PHI3:
        case LLM_ARCH_PHIMOE:
        case LLM_ARCH_PLAMO:
        case LLM_ARCH_PLAMO2:
        case LLM_ARCH_PLAMO3:
        case LLM_ARCH_GEMMA:
        case LLM_ARCH_GEMMA2:
        case LLM_ARCH_GEMMA3:
        case LLM_ARCH_GEMMA3N:
        case LLM_ARCH_GEMMA4:
        case LLM_ARCH_GEMMA4_ASSISTANT:
        case LLM_ARCH_GEMMA_EMBEDDING:
        case LLM_ARCH_STARCODER2:
        case LLM_ARCH_OPENELM:
        case LLM_ARCH_GPTNEOX:
        case LLM_ARCH_CODESHELL:
        case LLM_ARCH_ORION:
        case LLM_ARCH_NEMOTRON:
        case LLM_ARCH_EXAONE:
        case LLM_ARCH_EXAONE4:
        case LLM_ARCH_EXAONE_MOE:
        case LLM_ARCH_MINICPM3:
        case LLM_ARCH_BAILINGMOE2:
        case LLM_ARCH_DOTS1:
        case LLM_ARCH_HUNYUAN_MOE:
        case LLM_ARCH_JAIS2:
        case LLM_ARCH_OPENAI_MOE:
        case LLM_ARCH_HUNYUAN_DENSE:
        case LLM_ARCH_HY_V3:
        case LLM_ARCH_LFM2:
        case LLM_ARCH_LFM2MOE:
        case LLM_ARCH_SMALLTHINKER:
        case LLM_ARCH_SEED_OSS:
        case LLM_ARCH_GROVEMOE:
        case LLM_ARCH_APERTUS:
        case LLM_ARCH_MINIMAX_01:
        case LLM_ARCH_MINIMAX_M2:
        case LLM_ARCH_MINIMAX_M3:
        case LLM_ARCH_COGVLM:
        case LLM_ARCH_PANGU_EMBED:
        case LLM_ARCH_AFMOE:
        case LLM_ARCH_LAGUNA:
        case LLM_ARCH_QWEN3NEXT:
        case LLM_ARCH_MIMO2:
        case LLM_ARCH_STEP35:
        case LLM_ARCH_TALKIE:
        case LLM_ARCH_MELLUM:
            return LLAMA_ROPE_TYPE_NEOX;

        case LLM_ARCH_DFLASH:
            // DSV4 DSpark drafters use DeepSeek-V4's normal RoPE; legacy DFlash backbones are NeoX
            return model->hparams.dsv4_hc_mult > 0 ? LLAMA_ROPE_TYPE_NORM : LLAMA_ROPE_TYPE_NEOX;

        case LLM_ARCH_QWEN2VL:
        case LLM_ARCH_PADDLEOCR:
            return LLAMA_ROPE_TYPE_MROPE;
        case LLM_ARCH_QWEN3VL:
        case LLM_ARCH_QWEN3VLMOE:
        case LLM_ARCH_QWEN35:
        case LLM_ARCH_QWEN35MOE:
        case LLM_ARCH_QWEN3TTS:
            return LLAMA_ROPE_TYPE_IMROPE;

        case LLM_ARCH_GLM4:
            return model->hparams.use_mrope() ? LLAMA_ROPE_TYPE_MROPE : LLAMA_ROPE_TYPE_NORM;
        case LLM_ARCH_GLM4_MOE:
            return model->hparams.use_mrope() ? LLAMA_ROPE_TYPE_MROPE : LLAMA_ROPE_TYPE_NEOX;

        case LLM_ARCH_HUNYUAN_VL:
            return model->hparams.use_mrope() ? LLAMA_ROPE_TYPE_MROPE : LLAMA_ROPE_TYPE_NEOX;

        // all model arches should be listed explicitly here
        case LLM_ARCH_UNKNOWN:
            GGML_ABORT("unknown architecture");
    }

    return LLAMA_ROPE_TYPE_NONE;
}

float llama_model_rope_freq_scale_train(const llama_model * model) {
    return model->hparams.rope_freq_scale_train;
}

int32_t llama_model_meta_val_str(const llama_model * model, const char * key, char * buf, size_t buf_size) {
    const auto & it = model->gguf_kv.find(key);
    if (it == model->gguf_kv.end()) {
        if (buf_size > 0) {
            buf[0] = '\0';
        }
        return -1;
    }
    return snprintf(buf, buf_size, "%s", it->second.c_str());
}

int32_t llama_model_meta_count(const llama_model * model) {
    return (int)model->gguf_kv.size();
}

const char * llama_model_meta_key_str(llama_model_meta_key key) {
    switch (key) {
        case LLAMA_MODEL_META_KEY_SAMPLING_SEQUENCE:        return "general.sampling.sequence";
        case LLAMA_MODEL_META_KEY_SAMPLING_TOP_K:           return "general.sampling.top_k";
        case LLAMA_MODEL_META_KEY_SAMPLING_TOP_P:           return "general.sampling.top_p";
        case LLAMA_MODEL_META_KEY_SAMPLING_MIN_P:           return "general.sampling.min_p";
        case LLAMA_MODEL_META_KEY_SAMPLING_XTC_PROBABILITY: return "general.sampling.xtc_probability";
        case LLAMA_MODEL_META_KEY_SAMPLING_XTC_THRESHOLD:   return "general.sampling.xtc_threshold";
        case LLAMA_MODEL_META_KEY_SAMPLING_TEMP:            return "general.sampling.temp";
        case LLAMA_MODEL_META_KEY_SAMPLING_PENALTY_LAST_N:  return "general.sampling.penalty_last_n";
        case LLAMA_MODEL_META_KEY_SAMPLING_PENALTY_REPEAT:  return "general.sampling.penalty_repeat";
        case LLAMA_MODEL_META_KEY_SAMPLING_MIROSTAT:        return "general.sampling.mirostat";
        case LLAMA_MODEL_META_KEY_SAMPLING_MIROSTAT_TAU:    return "general.sampling.mirostat_tau";
        case LLAMA_MODEL_META_KEY_SAMPLING_MIROSTAT_ETA:    return "general.sampling.mirostat_eta";
        default:                                            return nullptr;
    }
}

int32_t llama_model_meta_key_by_index(const llama_model * model, int i, char * buf, size_t buf_size) {
    if (i < 0 || i >= (int)model->gguf_kv.size()) {
        if (buf_size > 0) {
            buf[0] = '\0';
        }
        return -1;
    }
    auto it = model->gguf_kv.begin();
    std::advance(it, i);
    return snprintf(buf, buf_size, "%s", it->first.c_str());
}

int32_t llama_model_meta_val_str_by_index(const llama_model * model, int32_t i, char * buf, size_t buf_size) {
    if (i < 0 || i >= (int)model->gguf_kv.size()) {
        if (buf_size > 0) {
            buf[0] = '\0';
        }
        return -1;
    }
    auto it = model->gguf_kv.begin();
    std::advance(it, i);
    return snprintf(buf, buf_size, "%s", it->second.c_str());
}

int32_t llama_model_desc(const llama_model * model, char * buf, size_t buf_size) {
    return snprintf(buf, buf_size, "%s", model->desc().c_str());
}

llama_ftype llama_model_ftype(const llama_model * model) {
    return model->ftype();
}

uint64_t llama_model_size(const llama_model * model) {
    return model->size();
}

const char * llama_model_chat_template(const llama_model * model, const char * name) {
    const auto key = name ? LLM_KV(model->arch, name)(LLM_KV_TOKENIZER_CHAT_TEMPLATE)
        : LLM_KV(model->arch)(LLM_KV_TOKENIZER_CHAT_TEMPLATE);
    const auto & it = model->gguf_kv.find(key);
    if (it == model->gguf_kv.end()) {
        // one-off fix for very popular models (so we are not flooded with issues)
        // do not extend this list unless absolutely necessary
        // Mistral-Small-2503 does not have built-in chat template
        llama_vocab_pre_type pre_type = model->vocab.get_pre_type();
        if (!name && pre_type == LLAMA_VOCAB_PRE_TYPE_TEKKEN && model->layers.size() == 40) {
            return "mistral-v7-tekken";
        }

        return nullptr;
    }

    return it->second.c_str();
}

uint64_t llama_model_n_params(const llama_model * model) {
    return model->n_elements();
}

bool llama_model_has_encoder(const llama_model * model) {
    switch (model->arch) {
        case LLM_ARCH_T5:
        case LLM_ARCH_T5ENCODER:
        case LLM_ARCH_EAGLE3:
        case LLM_ARCH_DFLASH:    return true;
        default:                 return false;
    }
}

bool llama_model_has_decoder(const llama_model * model) {
    switch (model->arch) {
        case LLM_ARCH_T5ENCODER: return false;
        default:                 return true;
    }
}

llama_token llama_model_decoder_start_token(const llama_model * model) {
    return model->hparams.dec_start_token_id;
}

bool llama_model_is_recurrent(const llama_model * model) {
    return llm_arch_is_recurrent(model->arch);
}

bool llama_model_is_hybrid(const llama_model * model) {
    return llm_arch_is_hybrid(model->arch);
}

bool llama_model_is_diffusion(const llama_model * model) {
    return llm_arch_is_diffusion(model->arch);
}

const std::vector<std::pair<std::string, ggml_tensor *>> & llama_internal_get_tensor_map(const llama_model * model) {
    return model->tensors_by_name;
}

int32_t llama_model_n_expert(const struct llama_model * model) {
    return model->hparams.n_expert;
}

int32_t llama_model_n_devices(const struct llama_model * model) {
    return (int32_t)model->devices.size();
}

ggml_backend_dev_t llama_model_get_device(const struct llama_model * model, int i) {
    if (i < 0 || i >= (int)model->devices.size()) {
        return nullptr;
    }
    return model->devices[i].dev;
}

//
// llama_model_base
//

llama_model_base::llama_model_base(const struct llama_model_params & params) : llama_model(params), model(this), tn(model->arch),
    TENSOR_DUPLICATED     (llama_model_loader::TENSOR_DUPLICATED),
    TENSOR_NOT_REQUIRED   (llama_model_loader::TENSOR_NOT_REQUIRED),
    TENSOR_SKIP           (llama_model_loader::TENSOR_SKIP),
    TENSOR_SKIP_IF_VIRTUAL(llama_model_loader::TENSOR_SKIP_IF_VIRTUAL),
    TENSOR_ALLOW_RESHAPE  (llama_model_loader::TENSOR_ALLOW_RESHAPE) {}

ggml_tensor * llama_model_base::create_tensor(const LLM_TN_IMPL & tn, const std::initializer_list<int64_t> & ne, int flags) {
    GGML_ASSERT(ml != nullptr);
    return create_tensor(*ml, tn, ne, flags);
}

void llama_model_base::create_tensor_gate_up_exps(llama_layer & layer, int bid, int64_t n_embd_, int64_t n_ff_, int64_t n_expert_, int flags) {
    layer.ffn_gate_up_exps = create_tensor(tn(LLM_TENSOR_FFN_GATE_UP_EXPS, "weight", bid), {n_embd_, n_ff_ * 2, n_expert_}, TENSOR_NOT_REQUIRED);
    if (layer.ffn_gate_up_exps == nullptr) {
        layer.ffn_gate_exps = create_tensor(tn(LLM_TENSOR_FFN_GATE_EXPS, "weight", bid), {n_embd_, n_ff_, n_expert_}, flags);
        layer.ffn_up_exps   = create_tensor(tn(LLM_TENSOR_FFN_UP_EXPS,   "weight", bid), {n_embd_, n_ff_, n_expert_}, flags);
    }
}

void llama_model_base::create_tensor_qkv(llama_layer & layer, int bid,
        int64_t n_embd_, int64_t n_embd_q_, int64_t n_embd_k_, int64_t n_embd_v_,
        int flags) {
    const int64_t n_embd_qkv = n_embd_q_ + n_embd_k_ + n_embd_v_;

    if (flags & TENSOR_SKIP) {
        const int skip = TENSOR_NOT_REQUIRED | TENSOR_SKIP;

        create_tensor(tn(LLM_TENSOR_ATTN_QKV, "weight", bid), {n_embd_, n_embd_qkv}, skip | TENSOR_SKIP_IF_VIRTUAL);
        create_tensor(tn(LLM_TENSOR_ATTN_QKV, "bias",   bid), {n_embd_qkv},          skip | TENSOR_SKIP_IF_VIRTUAL);
        create_tensor(tn(LLM_TENSOR_ATTN_Q,   "weight", bid), {n_embd_, n_embd_q_},  skip);
        create_tensor(tn(LLM_TENSOR_ATTN_K,   "weight", bid), {n_embd_, n_embd_k_},  skip);
        create_tensor(tn(LLM_TENSOR_ATTN_V,   "weight", bid), {n_embd_, n_embd_v_},  skip);
        create_tensor(tn(LLM_TENSOR_ATTN_Q,   "bias",   bid), {n_embd_q_},           skip);
        create_tensor(tn(LLM_TENSOR_ATTN_K,   "bias",   bid), {n_embd_k_},           skip);
        create_tensor(tn(LLM_TENSOR_ATTN_V,   "bias",   bid), {n_embd_v_},           skip);
        return;
    }

    layer.wqkv = create_tensor(tn(LLM_TENSOR_ATTN_QKV, "weight", bid), {n_embd_, n_embd_qkv}, TENSOR_NOT_REQUIRED | TENSOR_SKIP_IF_VIRTUAL);
    if (layer.wqkv) {
        layer.wqkv_b = create_tensor(tn(LLM_TENSOR_ATTN_QKV, "bias", bid), {n_embd_qkv}, TENSOR_NOT_REQUIRED | TENSOR_SKIP_IF_VIRTUAL);
    } else {
        layer.wq = create_tensor(tn(LLM_TENSOR_ATTN_Q, "weight", bid), {n_embd_, n_embd_q_}, flags);
        layer.wk = create_tensor(tn(LLM_TENSOR_ATTN_K, "weight", bid), {n_embd_, n_embd_k_}, flags);
        layer.wv = create_tensor(tn(LLM_TENSOR_ATTN_V, "weight", bid), {n_embd_, n_embd_v_}, flags);
        layer.wq_b = create_tensor(tn(LLM_TENSOR_ATTN_Q, "bias", bid), {n_embd_q_}, TENSOR_NOT_REQUIRED);
        layer.wk_b = create_tensor(tn(LLM_TENSOR_ATTN_K, "bias", bid), {n_embd_k_}, TENSOR_NOT_REQUIRED);
        layer.wv_b = create_tensor(tn(LLM_TENSOR_ATTN_V, "bias", bid), {n_embd_v_}, TENSOR_NOT_REQUIRED);
    }
}

const int32_t * llama_model_target_layer_ids(const struct llama_model * model) {
    const auto & v = model->target_layer_ids;
    return v.empty() ? nullptr : v.data();
}

uint32_t llama_model_target_layer_ids_n(const struct llama_model * model) {
    return (uint32_t) model->target_layer_ids.size();
}

uint32_t llama_model_get_tok_embd(const struct llama_model * model, float * out) {
    if (model->vocab.n_tokens() == 0 || model->tok_embd == nullptr) {
        return 0;
    }

    const ggml_tensor * tensor = model->tok_embd;
    const size_t nelements = ggml_nelements(tensor);
    GGML_ASSERT(nelements <= UINT32_MAX); // for the return type

    if (out == nullptr) {
        return (uint32_t) nelements;
    }

    if (tensor->type == GGML_TYPE_F32) {
        ggml_backend_tensor_get(tensor, out, 0, nelements * sizeof(float));
        return (uint32_t) nelements;
    }

    std::vector<uint8_t> buf(ggml_nbytes(tensor));
    ggml_backend_tensor_get(tensor, buf.data(), 0, buf.size());

    const ggml_type_traits * traits = ggml_get_type_traits(tensor->type);
    if (tensor->type == GGML_TYPE_F16) {
        ggml_fp16_to_fp32_row((const ggml_fp16_t *) buf.data(), out, nelements);
    } else if (tensor->type == GGML_TYPE_BF16) {
        ggml_bf16_to_fp32_row((const ggml_bf16_t *) buf.data(), out, nelements);
    } else if (ggml_is_quantized(tensor->type) && traits->to_float != nullptr) {
        traits->to_float(buf.data(), out, nelements);
    } else {
        GGML_ABORT("unsupported tensor type for dequantization: %s", ggml_type_name(tensor->type));
    }

    return (uint32_t) nelements;
}
