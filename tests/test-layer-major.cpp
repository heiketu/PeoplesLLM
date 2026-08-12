#include "llama.h"
#include "../src/llama-ext.h"
#include "../src/llama-layer-major.h"
#include "ggml-backend.h"
#include "repack.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

#ifdef GGML_USE_CUDA
#include <cuda_profiler_api.h>
#endif

static std::vector<float> copy_logits(llama_context * ctx, int32_t n_vocab) {
    const float * data = llama_get_logits_ith(ctx, -1);
    return std::vector<float>(data, data + n_vocab);
}

static int32_t top_one(const std::vector<float> & logits) {
    return std::max_element(logits.begin(), logits.end()) - logits.begin();
}

static bool compare_logits(const char * label, const std::vector<float> & a, const std::vector<float> & b) {
    double max_abs = 0.0;
    double max_rel = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        const double abs = std::fabs((double) a[i] - b[i]);
        max_abs = std::max(max_abs, abs);
        max_rel = std::max(max_rel, abs/std::max(1.0, std::fabs((double) a[i])));
    }

    const int32_t top_a = top_one(a);
    const int32_t top_b = top_one(b);
    printf("%s: top-1=%d/%d max_abs=%.8g max_rel=%.8g\n", label, top_a, top_b, max_abs, max_rel);
    return top_a == top_b && (max_abs <= 1e-4 || max_rel <= 1e-5);
}

int main(int argc, char ** argv) {
    if (argc == 2 && std::strcmp(argv[1], "--ep-plan-test") == 0) {
#if defined(_WIN32)
        _putenv_s("GGML_CUDA_MOE_PP_EP_OWNER_EXPERTS", "");
#else
        unsetenv("GGML_CUDA_MOE_PP_EP_OWNER_EXPERTS");
#endif
        const llama_moe_pp_ep_plan equal = llama_moe_pp_ep_plan_make(256);
        bool ok = !equal.asymmetric && equal.expert_begin[0] == 0 && equal.expert_count[0] == 128 &&
            equal.expert_begin[1] == 128 && equal.expert_count[1] == 128 &&
            llama_moe_pp_ep_device_rank(equal, 0, 1) == 0 &&
            llama_moe_pp_ep_device_rank(equal, 1, 1) == 1;
#if defined(_WIN32)
        _putenv_s("GGML_CUDA_MOE_PP_EP_OWNER_EXPERTS", "96");
#else
        setenv("GGML_CUDA_MOE_PP_EP_OWNER_EXPERTS", "96", 1);
#endif
        const llama_moe_pp_ep_plan asymmetric = llama_moe_pp_ep_plan_make(256);
        ok = ok && asymmetric.asymmetric && asymmetric.expert_begin[0] == 0 && asymmetric.expert_count[0] == 96 &&
            asymmetric.expert_begin[1] == 96 && asymmetric.expert_count[1] == 160 &&
            llama_moe_pp_ep_device_rank(asymmetric, 0, 1) == 1 &&
            llama_moe_pp_ep_device_rank(asymmetric, 1, 1) == 0;
        printf("MoE PP EP plan test: %s\n", ok ? "PASS" : "FAIL");
        return ok ? 0 : 1;
    }

    if (argc < 2) {
        printf("usage: %s MODEL [N_TOKENS=8] [TILE=4] [N_GPU_LAYERS=999] "
               "[BENCH_CPU_REPACK|BENCH_GPU_STREAM|BENCH_GPU_STREAM_REPACK|CHECK_CPU_REPACK] [N_THREADS=72] [N_GEN=0]\n", argv[0]);
        return 0;
    }

    const int32_t n_tokens = argc > 2 ? std::atoi(argv[2]) : 8;
    const int32_t n_ubatch = argc > 3 ? std::atoi(argv[3]) : 4;
    const int32_t n_gpu_layers = argc > 4 ? std::atoi(argv[4]) : 999;
    if (n_tokens < 2 || n_ubatch < 1 || n_ubatch > n_tokens) {
        fprintf(stderr, "invalid token/tile counts\n");
        return 1;
    }

    const bool bench_cpu_repack = argc > 5 && std::strcmp(argv[5], "BENCH_CPU_REPACK") == 0;
    const bool bench_gpu_stream = argc > 5 && std::strcmp(argv[5], "BENCH_GPU_STREAM") == 0;
    const bool bench_gpu_stream_repack = argc > 5 && std::strcmp(argv[5], "BENCH_GPU_STREAM_REPACK") == 0;
    const bool check_cpu_repack = argc > 5 && std::strcmp(argv[5], "CHECK_CPU_REPACK") == 0;
    const bool use_cpu_repack = bench_cpu_repack || check_cpu_repack || bench_gpu_stream_repack;
    const bool benchmark = bench_cpu_repack || bench_gpu_stream || bench_gpu_stream_repack;
    const int32_t n_threads = argc > 6 ? std::atoi(argv[6]) : 72;
    const int32_t n_gen = argc > 7 ? std::atoi(argv[7]) : 0;
    const bool fixed_tg = getenv("LLAMA_BENCH_FIXED_TG") != nullptr;
    const bool bench_numa_distribute = []() {
        const char * value = getenv("LLAMA_BENCH_NUMA_DISTRIBUTE");
        return value != nullptr && std::atoi(value) != 0;
    }();
    if (n_threads < 1) {
        fprintf(stderr, "invalid thread count\n");
        return 1;
    }
    if (n_gen < 0) {
        fprintf(stderr, "invalid generation token count\n");
        return 1;
    }

    llama_backend_init();
    if (use_cpu_repack || bench_numa_distribute) {
        llama_numa_init(GGML_NUMA_STRATEGY_DISTRIBUTE);
    }

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = n_gpu_layers;
    if (use_cpu_repack || bench_gpu_stream) {
        static llama_model_tensor_buft_override overrides[] = {
            { "\\.ffn_(up|down|gate|gate_up)_(ch|)exps", nullptr },
            { nullptr, nullptr },
        };
        if (bench_gpu_stream_repack) {
            // MoE weights stay in CPU_REPACK host buffers; the GPU streams
            // them through the unrepack prefetch path (moe_prefetch_src)
            overrides[0].buft = ggml_backend_cpu_repack_buffer_type();
        } else if (use_cpu_repack) {
            // A CPU override still lets the model loader select CPU extra buffer
            // types, including CPU_REPACK, when the tensor shape and ISA support it.
            overrides[0].buft = ggml_backend_cpu_buffer_type();
        } else {
            ggml_backend_dev_t gpu = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
            if (gpu == nullptr) {
                fprintf(stderr, "BENCH_GPU_STREAM requires a GPU backend\n");
                llama_backend_free();
                return 1;
            }
            overrides[0].buft = ggml_backend_dev_host_buffer_type(gpu);
        }
        mparams.tensor_buft_overrides = overrides;
        mparams.load_mode = LLAMA_LOAD_MODE_DIRECT_IO;
        mparams.use_extra_bufts = true;
    }
    llama_model * model = llama_model_load_from_file(argv[1], mparams);
    if (!model) {
        fprintf(stderr, "failed to load model\n");
        llama_backend_free();
        return 1;
    }

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = std::max(128, n_tokens + n_gen + 2);
    cparams.n_batch = n_ubatch;
    cparams.n_ubatch = n_ubatch;
    if (const char * env = getenv("LLAMA_BENCH_N_CTX")) {
        cparams.n_ctx = std::max((int32_t) std::atoi(env), n_tokens + n_gen + 2);
    }
    if (const char * env = getenv("LLAMA_BENCH_N_BATCH")) {
        cparams.n_batch = std::max(std::atoi(env), n_ubatch);
    }
    if (getenv("LLAMA_BENCH_KV_Q8")) {
        cparams.type_k = GGML_TYPE_Q8_0;
        cparams.type_v = GGML_TYPE_Q8_0;
    }
    if (const char * env = getenv("LLAMA_BENCH_SWA_FULL")) {
        // default is the llama_context_default_params() value (true);
        // LLAMA_BENCH_SWA_FULL=0 reproduces the server's compact SWA cache
        cparams.swa_full = std::atoi(env) != 0;
    }
    cparams.n_threads = n_threads;
    cparams.n_threads_batch = n_threads;
    cparams.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;

    // quantized-KV A/B for the q1 sparse decode paths (e.g. LLAMA_BENCH_KV_TYPE=q8_0)
    if (const char * kv_type = getenv("LLAMA_BENCH_KV_TYPE")) {
        if (std::strcmp(kv_type, "q8_0") == 0) {
            cparams.type_k = GGML_TYPE_Q8_0;
            cparams.type_v = GGML_TYPE_Q8_0;
        } else if (std::strcmp(kv_type, "f16") != 0) {
            fprintf(stderr, "unknown LLAMA_BENCH_KV_TYPE '%s' (expected f16 or q8_0)\n", kv_type);
            llama_model_free(model);
            llama_backend_free();
            return 1;
        }
    }

    if (benchmark) {
        // Run a token/tile sweep without reloading the (potentially 140+ GiB)
        // model. Example: LLAMA_BENCH_CASES=8192:8192,16384:8192,32768:8192.
        // Each case gets a fresh context so KV/cache state cannot leak between
        // measurements; the immutable model and pinned expert weights stay.
        std::vector<std::pair<int32_t, int32_t>> benchmark_cases = {{n_tokens, n_ubatch}};
        if (const char * env = getenv("LLAMA_BENCH_CASES")) {
            benchmark_cases.clear();
            const std::string cases(env);
            size_t begin = 0;
            while (begin < cases.size()) {
                const size_t end = cases.find(',', begin);
                const std::string item = cases.substr(begin, end == std::string::npos ? end : end - begin);
                int32_t case_tokens = 0;
                int32_t case_ubatch = 0;
                char trailing = '\0';
                if (sscanf(item.c_str(), "%d:%d%c", &case_tokens, &case_ubatch, &trailing) != 2 ||
                        case_tokens < 2 || case_ubatch < 1 || case_ubatch > case_tokens) {
                    fprintf(stderr, "invalid LLAMA_BENCH_CASES item '%s' (expected TOKENS:UBATCH)\n", item.c_str());
                    llama_model_free(model);
                    llama_backend_free();
                    return 1;
                }
                benchmark_cases.emplace_back(case_tokens, case_ubatch);
                if (end == std::string::npos) {
                    break;
                }
                begin = end + 1;
            }
            if (benchmark_cases.empty()) {
                fprintf(stderr, "LLAMA_BENCH_CASES contains no cases\n");
                llama_model_free(model);
                llama_backend_free();
                return 1;
            }
        }

        bool benchmark_ok = true;
        for (const auto & benchmark_case : benchmark_cases) {
        const int32_t n_tokens = benchmark_case.first;
        const int32_t n_ubatch = benchmark_case.second;
        llama_context_params case_cparams = cparams;
        case_cparams.n_ctx = std::max(case_cparams.n_ctx, (uint32_t) (n_tokens + n_gen + 2));
        case_cparams.n_batch = std::max(case_cparams.n_batch, (uint32_t) n_ubatch);
        case_cparams.n_ubatch = n_ubatch;

        llama_context * ctx = llama_init_from_model(model, case_cparams);
        if (!ctx) {
            fprintf(stderr, "failed to create benchmark context\n");
            llama_model_free(model);
            llama_backend_free();
            return 1;
        }

        using threadpool_new_t  = ggml_threadpool_t (*)(struct ggml_threadpool_params *);
        using threadpool_free_t = void (*)(ggml_threadpool_t);

        ggml_backend_dev_t cpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
        ggml_backend_reg_t cpu_reg = cpu_dev ? ggml_backend_dev_backend_reg(cpu_dev) : nullptr;
        auto threadpool_new_fn = cpu_reg ? (threadpool_new_t)
            ggml_backend_reg_get_proc_address(cpu_reg, "ggml_threadpool_new") : nullptr;
        auto threadpool_free_fn = cpu_reg ? (threadpool_free_t)
            ggml_backend_reg_get_proc_address(cpu_reg, "ggml_threadpool_free") : nullptr;
        if (!threadpool_new_fn || !threadpool_free_fn) {
            fprintf(stderr, "CPU threadpool API is unavailable\n");
            llama_free(ctx);
            llama_model_free(model);
            llama_backend_free();
            return 1;
        }

        struct ggml_threadpool_params tpp = ggml_threadpool_params_default(n_threads);
        ggml_threadpool_t threadpool = threadpool_new_fn(&tpp);
        if (!threadpool) {
            fprintf(stderr, "failed to create CPU threadpool\n");
            llama_free(ctx);
            llama_model_free(model);
            llama_backend_free();
            return 1;
        }
        llama_attach_threadpool(ctx, threadpool, nullptr);

        const int32_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
        llama_batch full = llama_batch_init(n_tokens, 0, 1);
        for (int32_t i = 0; i < n_tokens; ++i) {
            full.token[i] = (llama_token) ((i + 1) % n_vocab);
            full.pos[i] = i;
            full.n_seq_id[i] = 1;
            full.seq_id[i][0] = 0;
            full.logits[i] = i + 1 == n_tokens;
        }
        full.n_tokens = n_tokens;

        const int64_t t0 = llama_time_us();
        const int32_t rc = llama_decode_layer_major(ctx, full, n_ubatch);
        llama_synchronize(ctx);
        const double seconds = (llama_time_us() - t0) / 1e6;
        llama_batch_free(full);
        bool generation_ok = true;
        FILE * logits_trace = nullptr;

        if (rc == 0) {
            const std::vector<float> logits = copy_logits(ctx, n_vocab);
            double sum = 0.0;
            double sum_sq = 0.0;
            float min_value = logits[0];
            float max_value = logits[0];
            for (float value : logits) {
                sum += value;
                sum_sq += (double) value * value;
                min_value = std::min(min_value, value);
                max_value = std::max(max_value, value);
            }
            // Keep RESULT for existing parsers and add an explicit phase label
            // for the combined PP/TG benchmark output.
            printf("RESULT rc=0 tokens=%d tile=%d gpu_layers=%d seconds=%.6f tokens_per_second=%.6f\n",
                   n_tokens, n_ubatch, n_gpu_layers, seconds, n_tokens / seconds);
            printf("PP_RESULT rc=0 tokens=%d tile=%d gpu_layers=%d seconds=%.6f tokens_per_second=%.6f\n",
                   n_tokens, n_ubatch, n_gpu_layers, seconds, n_tokens / seconds);
            printf("LOGITS sum=%.12e sum_sq=%.12e min=%.9g max=%.9g top1=%d\n",
                   sum, sum_sq, min_value, max_value, top_one(logits));

            const char * logits_trace_path = getenv("LLAMA_BENCH_LOGITS_TRACE");
            if (logits_trace_path) {
                logits_trace = fopen(logits_trace_path, "wb");
                const uint32_t header[] = { 0x584c4d54u, (uint32_t) n_vocab, (uint32_t) n_gen + 1 };
                if (!logits_trace ||
                        fwrite(header, sizeof(header), 1, logits_trace) != 1 ||
                        fwrite(logits.data(), sizeof(float), logits.size(), logits_trace) != logits.size()) {
                    fprintf(stderr, "failed to initialize logits trace: %s\n", logits_trace_path);
                    if (logits_trace) {
                        fclose(logits_trace);
                        logits_trace = nullptr;
                    }
                }
            }

            if (n_gen > 0) {
                llama_token token = (llama_token) top_one(logits);
                const llama_token first = token;
                llama_token last = token;
                uint64_t token_hash = 1469598103934665603ULL;
                llama_batch gen = llama_batch_init(1, 0, 1);
                int32_t gen_rc = 0;
                const char * fixed_tg_base_env = getenv("LLAMA_BENCH_FIXED_TG_BASE");
                const int64_t fixed_tg_base = fixed_tg_base_env ? atoll(fixed_tg_base_env) : n_tokens;
#ifdef GGML_USE_CUDA
                const bool cuda_profile = getenv("LLAMA_BENCH_CUDA_PROFILE") != nullptr;
                if (cuda_profile) {
                    cudaProfilerStart();
                }
#endif
                const int64_t tg0 = llama_time_us();
                for (int32_t i = 0; i < n_gen; ++i) {
                    gen.token[0] = fixed_tg && i > 0 ?
                        (llama_token) ((fixed_tg_base + i) % n_vocab) : token;
                    gen.pos[0] = n_tokens + i;
                    gen.n_seq_id[0] = 1;
                    gen.seq_id[0][0] = 0;
                    gen.logits[0] = true;
                    gen.n_tokens = 1;
                    gen_rc = llama_decode(ctx, gen);
                    if (gen_rc != 0) {
                        break;
                    }
                    llama_synchronize(ctx);
                    const float * gen_logits = llama_get_logits_ith(ctx, -1);
                    if (logits_trace &&
                            fwrite(gen_logits, sizeof(float), n_vocab, logits_trace) != (size_t) n_vocab) {
                        fprintf(stderr, "failed to append logits trace\n");
                        fclose(logits_trace);
                        logits_trace = nullptr;
                    }
                    last = (llama_token) (std::max_element(gen_logits, gen_logits + n_vocab) - gen_logits);
                    token = last;
                    token_hash ^= (uint64_t) last;
                    token_hash *= 1099511628211ULL;
                }
#ifdef GGML_USE_CUDA
                if (cuda_profile) {
                    cudaProfilerStop();
                }
#endif
                const double tg_seconds = (llama_time_us() - tg0) / 1e6;
                printf("TG_RESULT rc=%d context=%d tokens=%d seconds=%.6f tokens_per_second=%.6f "
                       "fixed=%d first=%d last=%d hash=%016llx\n",
                       gen_rc, n_tokens, n_gen, tg_seconds, n_gen / tg_seconds, fixed_tg ? 1 : 0, first, last,
                       (unsigned long long) token_hash);
                generation_ok = gen_rc == 0;
                llama_batch_free(gen);
            }
        } else {
            fprintf(stderr, "layer-major benchmark failed: %d\n", rc);
        }

        if (logits_trace) {
            fclose(logits_trace);
        }

        llama_free(ctx);
        threadpool_free_fn(threadpool);
        benchmark_ok = benchmark_ok && rc == 0 && generation_ok;
        }
        llama_model_free(model);
        llama_backend_free();
        return benchmark_ok ? 0 : 1;
    }

    llama_context * ctx_ref = llama_init_from_model(model, cparams);
    if (!ctx_ref) {
        fprintf(stderr, "failed to create reference context\n");
        llama_free(ctx_ref);
        llama_model_free(model);
        llama_backend_free();
        return 1;
    }

    const int32_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    std::vector<llama_token> tokens(n_tokens);
    for (int32_t i = 0; i < n_tokens; ++i) {
        tokens[i] = (llama_token) ((i + 1) % n_vocab);
    }

    // optionally replace the synthetic token sequence with real prompt token
    // ids (JSON array, e.g. saved from llama-server /tokenize)
    if (const char * env = getenv("LLAMA_BENCH_TOKENS_FILE")) {
        FILE * f = fopen(env, "rb");
        if (!f) {
            fprintf(stderr, "failed to open tokens file: %s\n", env);
            return 1;
        }
        int64_t value = -1;
        int32_t n_loaded = 0;
        int ch = 0;
        while (n_loaded < n_tokens && (ch = fgetc(f)) != EOF) {
            if (ch >= '0' && ch <= '9') {
                value = std::max<int64_t>(0, value)*10 + (ch - '0');
            } else if (value >= 0) {
                tokens[n_loaded++] = (llama_token) value;
                value = -1;
            }
        }
        if (n_loaded < n_tokens && value >= 0) {
            tokens[n_loaded++] = (llama_token) value;
        }
        fclose(f);
        if (n_loaded != n_tokens) {
            fprintf(stderr, "tokens file has %d ids, expected %d\n", n_loaded, n_tokens);
            return 1;
        }
    }

    for (int32_t off = 0; off < n_tokens; off += n_ubatch) {
        const int32_t count = std::min(n_ubatch, n_tokens - off);
        if (llama_decode(ctx_ref, llama_batch_get_one(tokens.data() + off, count)) != 0) {
            fprintf(stderr, "reference decode failed at offset %d\n", off);
            return 1;
        }
    }

    // capture the reference logits (prefill + one decode step) and release the
    // reference context before creating the layer-major one, so that large
    // n_ctx KV caches of the two contexts never coexist
    const std::vector<float> logits_ref = copy_logits(ctx_ref, n_vocab);
    const llama_token next = (llama_token) top_one(logits_ref);
    std::vector<float> logits_ref_next;
    if (llama_decode(ctx_ref, llama_batch_get_one(const_cast<llama_token *>(&next), 1)) != 0) {
        fprintf(stderr, "reference post-prefill decode failed\n");
        return 1;
    }
    logits_ref_next = copy_logits(ctx_ref, n_vocab);
    llama_free(ctx_ref);
    ctx_ref = nullptr;

    llama_context * ctx_lm  = llama_init_from_model(model, cparams);
    if (!ctx_lm) {
        fprintf(stderr, "failed to create layer-major context\n");
        llama_model_free(model);
        llama_backend_free();
        return 1;
    }

    // mirror of common_init_ warmup (llama-server runs this on the shared
    // context before serving): a tiny decode followed by a memory clear
    if (getenv("LLAMA_BENCH_WARMUP")) {
        const llama_vocab * vocab_w = llama_model_get_vocab(model);
        std::vector<llama_token> tmp;
        const llama_token bos = llama_vocab_bos(vocab_w);
        const llama_token eos = llama_vocab_eos(vocab_w);
        if (bos != LLAMA_TOKEN_NULL) {
            tmp.push_back(bos);
        }
        if (eos != LLAMA_TOKEN_NULL) {
            tmp.push_back(eos);
        }
        if (tmp.empty()) {
            tmp.push_back(0);
        }
        llama_decode(ctx_lm, llama_batch_get_one(tmp.data(), tmp.size()));
        llama_memory_clear(llama_get_memory(ctx_lm), true);
        llama_synchronize(ctx_lm);
        // llama-server also trims the slot's sequence before the prompt
        // (common_context_seq_rm(ctx, slot.id, p0=0, -1)); mirror it for seq 0
        llama_memory_seq_rm(llama_get_memory(ctx_lm), 0, 0, -1);
    }

    llama_batch full = llama_batch_init(n_tokens, 0, 1);
    for (int32_t i = 0; i < n_tokens; ++i) {
        full.token[i] = tokens[i];
        full.pos[i] = i;
        full.n_seq_id[i] = 1;
        full.seq_id[i][0] = 0;
        full.logits[i] = i + 1 == n_tokens;
    }
    full.n_tokens = n_tokens;

    const int32_t rc = llama_decode_layer_major(ctx_lm, full, n_ubatch);
    llama_batch_free(full);
    if (rc != 0) {
        fprintf(stderr, "layer-major decode failed: %d\n", rc);
        return 1;
    }

    bool ok = compare_logits("prefill", logits_ref, copy_logits(ctx_lm, n_vocab));

    if (llama_decode(ctx_lm, llama_batch_get_one(const_cast<llama_token *>(&next), 1)) != 0) {
        fprintf(stderr, "post-prefill decode failed\n");
        return 1;
    }
    ok = compare_logits("next-token KV", logits_ref_next, copy_logits(ctx_lm, n_vocab)) && ok;

    llama_free(ctx_lm);
    llama_model_free(model);
    llama_backend_free();

    return ok ? 0 : 1;
}
