// ep-crossslot-bench: cross-slot pipelined EP proof-of-concept harness.
//
// Loads the model once and runs N independent llama_contexts on N threads,
// each decoding its own sequence. With GGML_REMOTE_EP_PIPE=1 every context
// owns its own dispatch stream, so the remote MoE requests of the N slots are
// in flight concurrently: while one slot's merge op waits for the worker, the
// other slots advance through attention/dense/local-MoE — the cross-slot
// pipeline that the per-layer blocking RPC never allowed.
//
// usage:
//   llama-ep-crossslot -m model.gguf [-n 4] [--tokens 512] [-t 72]
//                      [-c 2048] [--ngl 99] [--ncmoe 99] [--fa 1]
//                      [--check ref.txt] [--save out.txt]
//
// --save writes slot 0's sampled token text; --check compares every slot's
// text against a reference file produced with -n 1 (bit-exact EP merge must
// reproduce the single-stream output per slot).

#include "llama.h"
#include "common.h"
#include "llama-remote-ep.h"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <list>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

static const char * g_prompt =
    "Explain the economic impact of distributed computing on small businesses "
    "in the 21st century. Cover infrastructure costs, scalability, and the role of cloud providers.";

struct slot_result {
    std::string text;
    int         n_tokens = 0;
    double      tps      = 0.0;
    bool        ok       = false;
};

int main(int argc, char ** argv) {
    int         n_slots   = 4;
    int         n_gen     = 512;
    int         n_threads = 72;
    int         n_ctx     = 2048;
    int         n_gpu     = 99;
    int         n_cpu_moe = 99;
    int         fa        = 1;
    std::string model_path;
    std::string check_file;
    std::string save_file;

    for (int i = 1; i < argc; ++i) {
        auto need = [&](const char * name) -> const char * {
            if (i + 1 >= argc) {
                fprintf(stderr, "missing value for %s\n", name);
                exit(1);
            }
            return argv[++i];
        };
        const std::string a = argv[i];
        if      (a == "-m")        model_path = need("-m");
        else if (a == "-n")        n_slots    = atoi(need("-n"));
        else if (a == "--tokens")  n_gen      = atoi(need("--tokens"));
        else if (a == "-t")        n_threads  = atoi(need("-t"));
        else if (a == "-c")        n_ctx      = atoi(need("-c"));
        else if (a == "--ngl")     n_gpu      = atoi(need("--ngl"));
        else if (a == "--ncmoe")   n_cpu_moe  = atoi(need("--ncmoe"));
        else if (a == "--fa")      fa         = atoi(need("--fa"));
        else if (a == "--check")   check_file = need("--check");
        else if (a == "--save")    save_file  = need("--save");
        else {
            fprintf(stderr, "unknown arg: %s\n", a.c_str());
            return 1;
        }
    }
    if (model_path.empty() || n_slots < 1) {
        fprintf(stderr, "usage: llama-ep-crossslot -m model.gguf [-n slots] [--tokens N]\n");
        return 1;
    }

    llama_backend_init();

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = n_gpu;
    mparams.load_mode    = LLAMA_LOAD_MODE_NONE;

    // --ncmoe N: keep the routed experts of the first N layers on CPU, same
    // override list -ncmoe builds in common/arg.cpp (terminated entry at end)
    static std::list<std::string> exps_patterns;
    std::vector<llama_model_tensor_buft_override> buft_overrides;
    for (int i = 0; i < n_cpu_moe; ++i) {
        exps_patterns.push_back(llm_ffn_exps_block_regex(i));
        buft_overrides.push_back({exps_patterns.back().c_str(), ggml_backend_cpu_buffer_type()});
    }
    buft_overrides.push_back({nullptr, nullptr});
    if (n_cpu_moe > 0) {
        mparams.tensor_buft_overrides = buft_overrides.data();
    }

    llama_model * model = llama_model_load_from_file(model_path.c_str(), mparams);
    if (!model) {
        fprintf(stderr, "failed to load model\n");
        return 1;
    }
    const llama_vocab * vocab = llama_model_get_vocab(model);

    std::vector<slot_result> results((size_t) n_slots);
    std::atomic<int> n_ready{0};
    std::mutex mtx;

    auto worker = [&](int slot) {
        // any early exit must still release the start barrier
        struct ready_guard {
            std::atomic<int> * n;
            bool done = false;
            ~ready_guard() {
                if (!done) {
                    n->fetch_add(1);
                }
            }
        } guard{&n_ready};

        llama_context_params cparams = llama_context_default_params();
        cparams.n_ctx            = n_ctx;
        cparams.n_batch          = 512;
        cparams.n_ubatch         = 512;
        cparams.n_seq_max        = std::max(1, n_slots);
        cparams.n_threads        = std::max(1, n_threads / n_slots);
        cparams.n_threads_batch  = cparams.n_threads;
        cparams.flash_attn_type  = fa != 0 ? LLAMA_FLASH_ATTN_TYPE_ENABLED : LLAMA_FLASH_ATTN_TYPE_DISABLED;
        cparams.type_k           = GGML_TYPE_Q8_0;
        cparams.type_v           = GGML_TYPE_Q8_0;
        cparams.no_perf          = true;

        llama_context * ctx = llama_init_from_model(model, cparams);
        if (!ctx) {
            fprintf(stderr, "slot %d: llama_init_from_model failed\n", slot);
            return;
        }

        // tokenize the shared prompt
        std::vector<llama_token> tokens(1024);
        int n_prompt = llama_tokenize(vocab, g_prompt, strlen(g_prompt), tokens.data(), (int32_t) tokens.size(), true, true);
        if (n_prompt < 0) {
            tokens.resize((size_t) -n_prompt);
            n_prompt = llama_tokenize(vocab, g_prompt, strlen(g_prompt), tokens.data(), (int32_t) tokens.size(), true, true);
        }
        if (n_prompt < 0) {
            fprintf(stderr, "slot %d: tokenization failed\n", slot);
            llama_free(ctx);
            return;
        }
        tokens.resize((size_t) n_prompt);

        llama_sampler * smpl = llama_sampler_chain_init(llama_sampler_chain_default_params());
        llama_sampler_chain_add(smpl, llama_sampler_init_greedy());

        // decode the prompt with logits on the last token only
        llama_batch batch = llama_batch_init(512, 0, 1);
        for (int i = 0; i < n_prompt; ++i) {
            common_batch_add(batch, tokens[(size_t) i], i, {slot}, i + 1 == n_prompt);
        }
        if (llama_decode(ctx, batch) != 0) {
            fprintf(stderr, "slot %d: prompt decode failed\n", slot);
            llama_sampler_free(smpl);
            llama_batch_free(batch);
            llama_free(ctx);
            return;
        }

        // barrier: start all decode loops together (bounded so a failed peer
        // cannot wedge the survivors)
        guard.done = true;
        n_ready.fetch_add(1);
        {
            const auto b0 = std::chrono::steady_clock::now();
            while (n_ready.load() < n_slots) {
                if (std::chrono::duration<double>(std::chrono::steady_clock::now() - b0).count() > 120.0) {
                    fprintf(stderr, "slot %d: start barrier timeout, a peer failed to initialize\n", slot);
                    return;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
        }
        const auto t0 = std::chrono::steady_clock::now();

        std::string text;
        llama_token tok = llama_sampler_sample(smpl, ctx, -1);
        int n_decoded = 0;
        while (n_decoded < n_gen) {
            if (llama_vocab_is_eog(vocab, tok)) {
                tok = llama_sampler_sample(smpl, ctx, -1); // ignore EOG: keep the window full
            }
            {
                char buf[256];
                const int n = llama_token_to_piece(vocab, tok, buf, sizeof(buf), 0, false);
                if (n > 0) {
                    text.append(buf, (size_t) n);
                }
            }
            common_batch_clear(batch);
            common_batch_add(batch, tok, n_prompt + n_decoded, {slot}, true);
            const int dret = llama_decode(ctx, batch);
            if (dret != 0) {
                fprintf(stderr, "slot %d: decode failed at token %d (ret=%d)\n", slot, n_decoded, dret);
                break;
            }
            ++n_decoded;
            tok = llama_sampler_sample(smpl, ctx, -1);
        }
        const auto t1 = std::chrono::steady_clock::now();

        const double secs = std::chrono::duration<double>(t1 - t0).count();
        {
            std::lock_guard<std::mutex> lock(mtx);
            auto & r = results[(size_t) slot];
            r.text     = std::move(text);
            r.n_tokens = n_decoded;
            r.tps      = secs > 0 ? n_decoded / secs : 0.0;
            r.ok       = n_decoded == n_gen;
        }

        llama_sampler_free(smpl);
        llama_batch_free(batch);
        // skip llama_free(ctx): with remote-EP endpoints the context teardown
        // joins the per-endpoint receiver threads, which block in recv and can
        // wedge the whole exit. the OS reclaims everything at process exit.
    };

    std::vector<std::thread> threads;
    const auto wall0 = std::chrono::steady_clock::now();
    for (int s = 0; s < n_slots; ++s) {
        threads.emplace_back(worker, s);
    }
    for (auto & t : threads) {
        t.join();
    }
    const double wall = std::chrono::duration<double>(std::chrono::steady_clock::now() - wall0).count();

    int n_ok = 0;
    int total_tok = 0;
    std::ostringstream per;
    for (int s = 0; s < n_slots; ++s) {
        const auto & r = results[(size_t) s];
        if (r.ok) {
            ++n_ok;
        }
        total_tok += r.n_tokens;
        if (s) {
            per << " ";
        }
        per << "req" << s << ":" << r.tps << "t/s/" << r.n_tokens << "tok";
    }
    printf("SLOTS n=%d ok=%d wall=%.1fs total_tok=%d aggregate=%.2f tok/s\n",
            n_slots, n_ok, wall, total_tok, wall > 0 ? total_tok / wall : 0.0);
    printf("  per-req: %s\n", per.str().c_str());

    if (!save_file.empty()) {
        for (int s = 0; s < n_slots; ++s) {
            std::ofstream f(save_file + ".s" + std::to_string(s));
            f << results[(size_t) s].text;
        }
    }

    int rc = 0;
    if (!check_file.empty()) {
        for (int s = 0; s < n_slots; ++s) {
            std::ifstream f(check_file + ".s" + std::to_string(s));
            std::stringstream ss;
            ss << f.rdbuf();
            const std::string ref = ss.str();
            if (!ref.empty() && results[(size_t) s].text == ref) {
                printf("slot %d: MATCH reference\n", s);
            } else {
                printf("slot %d: DIFF vs reference (ref_bytes=%zu got_bytes=%zu)\n",
                        s, ref.size(), results[(size_t) s].text.size());
                rc = 2;
            }
        }
    }

    // skip llama_model_free/llama_backend_free and _exit: the remote_ep_state
    // static destructor joins the per-endpoint receiver threads, which block in
    // recv (RDMA poll / TCP) and wedge process exit once any endpoint was used.
    // Flush frequency counters explicitly because _Exit skips static destructors.
    llama_remote_ep_dump_freq();
    ggml_backend_cpu_op_timing_dump();
    (void) model;
    fflush(stdout);
    fflush(stderr);
    _Exit(rc);
}
