// moe-pp-prefetch-bench: ggml-level microbench for the GPU MoE prefill
// streaming path (GGML_CUDA_MOE_PP_MIN_TOKENS / GGML_CUDA_MOE_PP_PREFETCH /
// GGML_CUDA_MOE_PP_PIPE / GGML_CUDA_MOE_PP_DEFER_PREFETCH).
//
// Builds a synthetic MoE ubatch graph: PP_BENCH_LAYERS layers x 3 chained
// mul_mat_id ops, expert weights in a pinned host buffer (usage=WEIGHTS,
// MXFP4, plain GGUF layout), F32 activations. Runs through
// ggml_backend_sched with {CUDA, CPU} so the offload + streaming + prefetch
// scheduler paths execute exactly as in llama.cpp prefill — no model load.
//
// Env knobs (all optional):
//   PP_BENCH_DEVICE   GPU device index              (default 0)
//   PP_BENCH_LAYERS   MoE layers                    (default 8)
//   PP_BENCH_EXPERTS  experts per matrix            (default 32)
//   PP_BENCH_K        expert rows/cols (square)     (default 2048)
//   PP_BENCH_TOKENS   ubatch tokens                 (default 1024)
//   PP_BENCH_USED     experts used per token        (default 8)
//   PP_BENCH_ITERS    timed iterations              (default 5)
//   PP_BENCH_CPU_ONLY=1  single-CPU-backend sched (reference dump)
//   PP_BENCH_SAVE     write final output floats to this file
//   PP_BENCH_REF      load reference floats and report NMSE vs this run
//
// Example A/B:
//   GGML_CUDA_MOE_PP_MIN_TOKENS=512 ./moe-pp-prefetch-bench            # v1 serial
//   GGML_CUDA_MOE_PP_MIN_TOKENS=512 GGML_CUDA_MOE_PP_PREFETCH=2 \
//     GGML_CUDA_MOE_PP_PIPE=1 ./moe-pp-prefetch-bench                  # v3 pipeline

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static int env_int(const char * name, int def) {
    const char * v = getenv(name);
    return v ? atoi(v) : def;
}

int main() {
    const int dev_idx  = env_int("PP_BENCH_DEVICE", 0);
    const int n_layers = env_int("PP_BENCH_LAYERS", 8);
    const int n_expert = env_int("PP_BENCH_EXPERTS", 32);
    const int k        = env_int("PP_BENCH_K", 2048);
    const int tokens   = env_int("PP_BENCH_TOKENS", 1024);
    const int n_used   = env_int("PP_BENCH_USED", 8);
    const int iters    = env_int("PP_BENCH_ITERS", 5);
    const bool cpu_only = env_int("PP_BENCH_CPU_ONLY", 0) != 0;
    const char * save_path = getenv("PP_BENCH_SAVE");
    const char * ref_path  = getenv("PP_BENCH_REF");
    const bool check   = ref_path != nullptr;

    ggml_backend_load_all();

    ggml_backend_dev_t gpu_dev = nullptr;
    ggml_backend_dev_t cpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    {
        int seen = 0;
        const size_t ndev = ggml_backend_dev_count();
        for (size_t i = 0; i < ndev; ++i) {
            ggml_backend_dev_t d = ggml_backend_dev_get(i);
            if (ggml_backend_dev_type(d) == GGML_BACKEND_DEVICE_TYPE_GPU) {
                if (seen == dev_idx) {
                    gpu_dev = d;
                }
                ++seen;
            }
        }
    }
    if (gpu_dev == nullptr || cpu_dev == nullptr) {
        fprintf(stderr, "no GPU/CPU device found\n");
        return 1;
    }
    fprintf(stderr, "gpu: %s\n", ggml_backend_dev_name(gpu_dev));

    ggml_backend_t gpu_backend = ggml_backend_dev_init(gpu_dev, nullptr);
    ggml_backend_t cpu_backend = ggml_backend_dev_init(cpu_dev, nullptr);
    if (gpu_backend == nullptr || cpu_backend == nullptr) {
        fprintf(stderr, "backend init failed\n");
        return 1;
    }

    // ---- weights: pinned host buffer, usage=WEIGHTS (same as the loader path)
    ggml_backend_buffer_type_t pinned_buft = ggml_backend_dev_host_buffer_type(gpu_dev);
    if (pinned_buft == nullptr) {
        pinned_buft = ggml_backend_cpu_buffer_type();
    }
    fprintf(stderr, "weight buft: %s\n", ggml_backend_buft_name(pinned_buft));

    const int n_mats = n_layers * 3;
    struct ggml_init_params wparams = {
        /*.mem_size   =*/ ggml_tensor_overhead() * (size_t) n_mats,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context * wctx = ggml_init(wparams);

    std::vector<ggml_tensor *> weights(n_mats);
    for (int i = 0; i < n_mats; ++i) {
        weights[i] = ggml_new_tensor_3d(wctx, GGML_TYPE_MXFP4, k, k, n_expert);
        char name[64];
        snprintf(name, sizeof(name), "blk.%d.ffn_m%d_exps.weight", i / 3, i % 3);
        ggml_set_name(weights[i], name);
    }
    ggml_backend_buffer_t wbuf = ggml_backend_alloc_ctx_tensors_from_buft(wctx, pinned_buft);
    if (wbuf == nullptr) {
        fprintf(stderr, "weight buffer alloc failed\n");
        return 1;
    }
    ggml_backend_buffer_set_usage(wbuf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

    // fill MXFP4: byte 0 of each 17-byte block is the E8M0 scale; keep it
    // small (126) so values stay finite, nibbles pseudo-random
    {
        const size_t blck = ggml_blck_size(GGML_TYPE_MXFP4);          // 32
        const size_t tsz  = ggml_type_size(GGML_TYPE_MXFP4);          // 17
        uint64_t rng = 0x9e3779b97f4a7c15ull;
        for (auto * w : weights) {
            uint8_t * p = (uint8_t *) w->data;
            const int64_t nvals = ggml_nelements(w);
            for (int64_t b = 0; b < nvals / (int64_t) blck; ++b) {
                uint8_t * blk = p + b * tsz;
                blk[0] = 126;
                for (size_t j = 1; j < tsz; ++j) {
                    rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
                    blk[j] = (uint8_t) rng;
                }
            }
        }
    }

    // ---- graph inputs: activation + ids on a plain CPU buffer
    struct ggml_init_params iparams = {
        /*.mem_size   =*/ ggml_tensor_overhead() * 2,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context * ictx = ggml_init(iparams);
    ggml_tensor * x0  = ggml_new_tensor_3d(ictx, GGML_TYPE_F32, k, n_used, tokens);
    ggml_tensor * ids = ggml_new_tensor_2d(ictx, GGML_TYPE_I32, n_used, tokens);
    ggml_set_name(x0, "inp_x");
    ggml_set_name(ids, "inp_ids");
    x0->flags  |= GGML_TENSOR_FLAG_INPUT;
    ids->flags |= GGML_TENSOR_FLAG_INPUT;
    ggml_backend_buffer_t ibuf = ggml_backend_alloc_ctx_tensors_from_buft(ictx, ggml_backend_cpu_buffer_type());
    if (ibuf == nullptr) {
        fprintf(stderr, "input buffer alloc failed\n");
        return 1;
    }
    {
        std::vector<float>   xd(ggml_nelements(x0));
        std::vector<int32_t> idd(ggml_nelements(ids));
        uint64_t rng = 12345;
        for (auto & v : xd) {
            rng = rng * 6364136223846793005ull + 1442695040888963407ull;
            v = ((rng >> 33) & 0xffff) / 65536.0f - 0.5f;
        }
        for (auto & v : idd) {
            rng = rng * 6364136223846793005ull + 1442695040888963407ull;
            v = (int32_t) ((rng >> 33) % (uint64_t) n_expert);
        }
        ggml_backend_tensor_set(x0, xd.data(), 0, ggml_nbytes(x0));
        ggml_backend_tensor_set(ids, idd.data(), 0, ggml_nbytes(ids));
    }

    // ---- graph: n_layers x 3 chained mul_mat_id (square dims so out feeds next)
    const int n_nodes = n_mats;
    struct ggml_init_params gparams = {
        /*.mem_size   =*/ ggml_tensor_overhead() * (size_t) (n_nodes + 8) + ggml_graph_overhead(),
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context * gctx = ggml_init(gparams);
    ggml_cgraph * gf = ggml_new_graph_custom(gctx, GGML_DEFAULT_GRAPH_SIZE, false);
    ggml_tensor * cur = x0;
    for (int i = 0; i < n_mats; ++i) {
        cur = ggml_mul_mat_id(gctx, weights[i], cur, ids);
        char name[64];
        snprintf(name, sizeof(name), "moe.%d", i);
        ggml_set_name(cur, name);
        ggml_build_forward_expand(gf, cur);
    }
    ggml_tensor * result = cur;

    // ---- reference IO helpers
    auto save_result = [&](const char * path) {
        std::vector<float> out(ggml_nelements(result));
        ggml_backend_tensor_get(result, out.data(), 0, ggml_nbytes(result));
        FILE * f = fopen(path, "wb");
        if (f == nullptr) {
            fprintf(stderr, "cannot open %s for writing\n", path);
            return false;
        }
        fwrite(out.data(), sizeof(float), out.size(), f);
        fclose(f);
        return true;
    };

    // ---- sched selection: PP_BENCH_CPU_ONLY=1 runs a single-CPU-backend
    // sched (reference dump); otherwise {CUDA, CPU} with op_offload enabled
    // (mirrors llama.cpp prefill)
    ggml_backend_sched_t sched = nullptr;
    if (cpu_only) {
        ggml_backend_t only_cpu[1] = { cpu_backend };
        sched = ggml_backend_sched_new(only_cpu, nullptr, 1, GGML_DEFAULT_GRAPH_SIZE, false, false);
    } else {
        ggml_backend_t backends[2] = { gpu_backend, cpu_backend };
        sched = ggml_backend_sched_new(backends, nullptr, 2, GGML_DEFAULT_GRAPH_SIZE, false, true);
    }

    auto run_once = [&]() -> bool {
        return ggml_backend_sched_graph_compute(sched, gf) == GGML_STATUS_SUCCESS;
    };

    if (!run_once()) { // warmup + alloc
        fprintf(stderr, "compute failed\n");
        return 1;
    }

    std::vector<double> times(iters);
    for (int it = 0; it < iters; ++it) {
        const int64_t t0 = ggml_time_us();
        if (!run_once()) {
            fprintf(stderr, "compute failed at iter %d\n", it);
            return 1;
        }
        times[it] = (ggml_time_us() - t0) / 1000.0;
        if (getenv("PP_BENCH_VERBOSE") != nullptr) {
            fprintf(stderr, "iter %d: %.2f ms\n", it, times[it]);
        }
    }

    const double weight_mib = (double) ggml_backend_buffer_get_size(wbuf) / (1024.0 * 1024.0);
    std::sort(times.begin(), times.end());
    fprintf(stdout, "RESULT layers=%d experts=%d k=%d tokens=%d used=%d weights=%.1fMiB "
            "median=%.2fms min=%.2fms max=%.2fms\n",
            n_layers, n_expert, k, tokens, n_used, weight_mib,
            times[iters / 2], times.front(), times.back());

    if (check) {
        std::vector<float> got(ggml_nelements(result));
        ggml_backend_tensor_get(result, got.data(), 0, ggml_nbytes(result));

        std::vector<float> ref(ggml_nelements(result));
        FILE * f = fopen(ref_path, "rb");
        if (f == nullptr) {
            fprintf(stderr, "cannot open reference %s\n", ref_path);
            return 1;
        }
        const size_t nread = fread(ref.data(), sizeof(float), ref.size(), f);
        fclose(f);
        if (nread != ref.size()) {
            fprintf(stderr, "reference %s has %zu floats, expected %zu\n", ref_path, nread, ref.size());
            return 1;
        }

        // NMSE
        double num = 0.0, den = 0.0;
        double max_abs = 0.0;
        for (size_t i = 0; i < ref.size(); ++i) {
            const double d = (double) got[i] - ref[i];
            num += d * d;
            den += (double) ref[i] * ref[i];
            max_abs = std::max(max_abs, std::fabs(d));
        }
        const double nmse = den > 0 ? num / den : 0.0;
        fprintf(stdout, "CHECK nmse=%.3e max_abs=%.3e %s\n", nmse, max_abs,
                nmse < 5e-4 ? "OK" : "FAIL");
        if (nmse >= 5e-4) {
            return 2;
        }
    }
    if (save_path != nullptr && !save_result(save_path)) {
        return 1;
    }

    ggml_backend_sched_free(sched);
    ggml_backend_buffer_free(wbuf);
    ggml_backend_buffer_free(ibuf);
    ggml_free(gctx);
    ggml_free(wctx);
    ggml_free(ictx);
    ggml_backend_free(gpu_backend);
    ggml_backend_free(cpu_backend);
    return 0;
}
