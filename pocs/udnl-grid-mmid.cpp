// Standalone scheduling prototype for the "32-core grid matrix engine" idea:
// same UDNL_MX arec panel kernels as the mul_mat_id pp path, but the per-expert
// GEMM (M tokens x N out-channels x K) is partitioned over a 2D grid of cores
// (g_m row-groups x g_n column-groups) instead of the current 1D N-only claim.
// Compares schedule variants on identical data and reports GOP/s + checksums.
//
// Build (against build-udnl):
//   g++ -O2 -std=c++17 -I ggml/include pocs/udnl-grid-mmid.cpp -o build-udnl/bin/udnl-grid-mmid \
//       -L build-udnl/bin -lggml-cpu -lggml-base -lggml -Wl,-rpath,$PWD/build-udnl/bin -lpthread
//
// Usage:
//   udnl-grid-mmid [M=64] [N=2048] [K=4096] [EXPERTS=256] [ITERS=6]
// Schedules are swept internally per thread count.

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include <pthread.h>
#include <sched.h>

// ---- layout constants (mirrors ggml-common.h UDNL_MX repack) ----
// block_udnl_mx = 108 B per row per 256-K block; repacked panel of 16 rows x
// 256 K = 1728 B: 8 x [payload_g | srel 16B] + d[16] fp16 @1664 + modes u16
// @1696 + 30 B pad. Valid mode mix must satisfy #W3 + 2*#W4 == 8.
static constexpr int QK_MX = 256;
static constexpr int MX_PB = 1728;        // bytes per (panel, 256-K block)
static constexpr int Q8_B = 34;           // sizeof(block_q8_0): fp16 d + 32 x int8

struct udnl_w4_arec {
    int32_t asum128;
    float   dy;
};

extern "C" void ggml_gemm_udnl_mx_1x16_q8_0_arec(
        int n, float * s, size_t bs, const void * vx, const void * vy,
        const udnl_w4_arec * arec, int nr, int nc);
extern "C" void ggml_gemm_udnl_w4_1x16_q8_0_arec(
        int n, float * s, size_t bs, const void * vx, const void * vy,
        const udnl_w4_arec * arec, int nr, int nc);

static void pin_thread(int cpu) {
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(cpu, &set);
    pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
}

// simple sense-reversing barrier
struct spin_barrier {
    std::atomic<int> count{0};
    std::atomic<int> sense{0};
    int n;
    explicit spin_barrier(int n_) : n(n_) {}
    void arrive() {
        int s = sense.load(std::memory_order_relaxed);
        if (count.fetch_add(1, std::memory_order_acq_rel) == n - 1) {
            count.store(0, std::memory_order_relaxed);
            sense.store(!s, std::memory_order_release);
        } else {
            while (sense.load(std::memory_order_acquire) == s) {}
        }
    }
};

struct prob {
    int M, N, K, E;
    int nb;   // 256-K blocks per row = K/256
    int anb;  // q8_0 blocks per row = K/32
    int row_bytes;      // nb * MX_PB (one weight row... stored panel-major)
    int a_row_bytes;    // anb * Q8_B
    std::vector<uint8_t> w;      // [E][N/16 panels][nb][MX_PB]
    std::vector<uint8_t> act;    // [M][anb][Q8_B]
    std::vector<udnl_w4_arec> arec; // [M][anb]
    std::vector<float> dst;      // [E][M][N] — one output slab per expert (the
                                 // real mmid scatters experts to distinct rows;
                                 // sharing one slab across experts would make the
                                 // last-writer racy in the claim schedule)
};

static void fill_problem(prob & p, uint64_t seed) {
    std::mt19937_64 rng(seed);
    const int64_t wbytes = (int64_t) p.E * (p.N/16) * p.nb * MX_PB;
    p.w.resize((size_t) wbytes);
    for (int64_t i = 0; i < wbytes; ++i) p.w[(size_t) i] = (uint8_t) rng();
    // fix up per (panel, block): d[16] fp16 @1664 = 2^-10, modes @1696 =
    // mixed W4(g0,g4)/W3(g1,g3,g5,g7)/W2(g2,g6): 3|2<<2|1<<4|2<<6|3<<8|2<<10|1<<12|2<<14
    const uint16_t modes = (uint16_t) (3 | (2<<2) | (1<<4) | (2<<6) | (3<<8) | (2<<10) | (1<<12) | (2<<14));
    const int64_t nblocks = (int64_t) p.E * (p.N/16) * p.nb;
    for (int64_t b = 0; b < nblocks; ++b) {
        uint8_t * pb = p.w.data() + b*MX_PB;
        for (int r = 0; r < 16; ++r) { pb[1664 + 2*r] = 0x00; pb[1664 + 2*r + 1] = 0x14; }
        memcpy(pb + 1696, &modes, 2);
    }
    // activations: q8_0 rows, d = 2^-10
    p.act.resize((size_t) p.M * p.anb * Q8_B);
    p.arec.resize((size_t) p.M * p.anb);
    const float dy = 0.0009765625f; // 2^-10
    for (int m = 0; m < p.M; ++m) {
        for (int j = 0; j < p.anb; ++j) {
            uint8_t * blk = p.act.data() + ((size_t) m*p.anb + j)*Q8_B;
            blk[0] = 0x00; blk[1] = 0x14;
            int32_t asum = 0;
            for (int v = 0; v < 32; ++v) {
                int8_t q = (int8_t) rng();
                blk[2 + v] = (uint8_t) q;
                asum += q;
            }
            p.arec[(size_t) m*p.anb + j] = { 128*asum, dy };
        }
    }
    p.dst.assign((size_t) p.E * p.M * p.N, 0.0f);
}

// aligned contiguous column split of [0, N) into ns slices (16-col aligned),
// mirrors the repack.cpp thread split (boundaries rounded up to 16).
static void col_split(int N, int ns, int j, int & c0, int & c1) {
    auto align_up = [](int64_t v) { return (v % 16) ? v + 16 - (v % 16) : v; };
    int64_t s = align_up(((int64_t) j     * N) / ns);
    int64_t e = align_up(((int64_t) (j+1) * N) / ns);
    c0 = (int) s;
    c1 = (int) (e > N ? N : e);
}

struct run_cfg {
    int nth;      // threads
    int gm, gn;   // grid dims; gm == 1 => current 1D N-split
    int m_tile;   // rows per gemm call (32 matches MMID_GEMM_TILE_MAX)
    int iters;
    int claim;    // > 0: dynamic 1D column claims of `claim` cols per expert
                  // (mirrors the GGML_NUMA_EP claim loop, minus NUMA windows)
    const char * name;
};

static double run_schedule(prob & p, const run_cfg & cfg, double & checksum) {
    std::fill(p.dst.begin(), p.dst.end(), 0.0f);
    spin_barrier bar{ cfg.nth + 1 }; // workers + main
    spin_barrier wbar{ cfg.nth };    // workers only (per-iter drain in claim mode)
    std::atomic<bool> go{false};
    std::atomic<int64_t> claim_ctr{0};
    const int nchunks = cfg.claim > 0 ? p.N / cfg.claim : 0;
    std::vector<std::thread> ths;
    ths.reserve(cfg.nth);
    auto worker = [&](int t) {
        pin_thread(t); // cpus 0..nth-1 are physical cores (0-37 socket0, 38-75 socket1)
        // grid coords: same row-group => consecutive cpu ids (socket-local sharing
        // of the activation slice when gn <= 38)
        const int gi = t / cfg.gn;  // row group
        const int gj = t % cfg.gn;  // col group
        int r0 = 0, r1 = p.M, c0 = 0, c1 = p.N;
        if (cfg.gm > 1) {
            r0 = (int) (((int64_t) gi     * p.M) / cfg.gm);
            r1 = (int) (((int64_t) (gi+1) * p.M) / cfg.gm);
        }
        col_split(p.N, cfg.gn, gj, c0, c1);
        const int ncols = c1 - c0;
        bar.arrive();
        while (!go.load(std::memory_order_acquire)) {}
        const int nb = p.nb;
        if (cfg.claim > 0) {
            // dynamic (expert, col-chunk) claims: one global work queue, threads
            // pull the next chunk of the next expert — the balanced 1D baseline
            // matching the production EP claim path.
            for (int it = 0; it < cfg.iters; ++it) {
                for (;;) {
                    const int64_t item = claim_ctr.fetch_add(1, std::memory_order_relaxed);
                    const int e = (int) (item / nchunks);
                    if (e >= p.E) break;
                    const int c = (int) (item % nchunks) * cfg.claim;
                    const uint8_t * we = p.w.data() + (int64_t) e * (p.N/16) * nb * MX_PB
                                       + (int64_t) (c/16) * nb * MX_PB;
                    for (int base = 0; base < p.M; base += cfg.m_tile) {
                        const int tr = std::min(cfg.m_tile, p.M - base);
                        const uint8_t * ay = p.act.data() + (size_t) base * p.anb * Q8_B;
                        const udnl_w4_arec * rec = p.arec.data() + (size_t) base * p.anb;
                        float * s = p.dst.data() + ((size_t) e * p.M + base) * p.N + c;
                        ggml_gemm_udnl_mx_1x16_q8_0_arec(p.K, s, p.N, we, ay, rec, tr, cfg.claim);
                    }
                }
                wbar.arrive(); // wait for all threads to drain this iter
                if (t == 0) claim_ctr.store(0, std::memory_order_relaxed);
                wbar.arrive();
            }
        } else if (ncols > 0 && r0 < r1) {
            for (int it = 0; it < cfg.iters; ++it) {
                for (int e = 0; e < p.E; ++e) {
                    const uint8_t * we = p.w.data() + (int64_t) e * (p.N/16) * nb * MX_PB
                                       + (int64_t) (c0/16) * nb * MX_PB;
                    for (int base = r0; base < r1; base += cfg.m_tile) {
                        const int tr = std::min(cfg.m_tile, r1 - base);
                        const uint8_t * ay = p.act.data() + (size_t) base * p.anb * Q8_B;
                        const udnl_w4_arec * rec = p.arec.data() + (size_t) base * p.anb;
                        float * s = p.dst.data() + ((size_t) e * p.M + base) * p.N + c0;
                        ggml_gemm_udnl_mx_1x16_q8_0_arec(p.K, s, p.N, we, ay, rec, tr, ncols);
                    }
                }
            }
        }
        bar.arrive();
    };
    for (int t = 0; t < cfg.nth; ++t) ths.emplace_back(worker, t);
    bar.arrive(); // all workers spawned and pinned
    const auto start = std::chrono::steady_clock::now();
    go.store(true, std::memory_order_release);
    bar.arrive(); // all workers done
    const auto end = std::chrono::steady_clock::now();
    for (auto & th : ths) th.join();
    const double sec = std::chrono::duration<double>(end - start).count();

    double sum = 0.0;
    for (float v : p.dst) sum += v;
    checksum = sum;
    const double flops = 2.0 * p.M * p.N * (double) p.K * p.E * cfg.iters;
    return flops / sec / 1e9;
}

int main(int argc, char ** argv) {
    prob p;
    p.M = argc > 1 ? atoi(argv[1]) : 64;
    p.N = argc > 2 ? atoi(argv[2]) : 2048;
    p.K = argc > 3 ? atoi(argv[3]) : 4096;
    p.E = argc > 4 ? atoi(argv[4]) : 256;
    const int iters = argc > 5 ? atoi(argv[5]) : 6;
    p.nb = p.K / QK_MX;
    p.anb = p.K / 32;
    p.row_bytes = p.nb * MX_PB;
    p.a_row_bytes = p.anb * Q8_B;
    if (p.K % QK_MX || p.N % 16 || p.M % 8) {
        fprintf(stderr, "bad dims\n");
        return 1;
    }
    fprintf(stderr, "filling %lld MB of weights...\n",
            (long long) ((int64_t) p.E * (p.N/16) * p.nb * MX_PB >> 20));
    fill_problem(p, 1234);

    struct sweep { int nth, gm, gn, mtile, claim; const char * label; };
    const std::vector<sweep> cases = {
        {  1, 1,  1, 32,   0, "flat-1d (ceiling)" },
        { 72, 1, 72, 32,   0, "flat-1d static" },
        { 72, 1, 72, 32,  32, "claim-32 (prod EP)" },
        { 72, 1, 72, 32,  64, "claim-64 (prod EP)" },
        { 72, 1, 72, 64,   0, "flat-1d mtile64" },
        { 72, 1, 72, 64,  32, "claim-32 mtile64" },
        { 72, 2, 36, 32,   0, "grid-2x36" },
        { 72, 4, 18, 32,   0, "grid-4x18" },
        { 72, 8,  9, 32,   0, "grid-8x9" },
        { 64, 1, 64, 32,   0, "flat-1d static" },
        { 64, 2, 32, 32,   0, "grid-2x32" },
        { 64, 4, 16, 32,   0, "grid-4x16" },
        { 32, 1, 32, 32,   0, "flat-1d static" },
        { 32, 2, 16, 32,   0, "grid-2x16" },
        { 32, 4,  8, 32,   0, "grid-4x8" },
    };

    printf("%-20s %6s %9s %14s\n", "schedule", "nth", "gops", "checksum");
    double ref = 0.0;
    bool have_ref = false;
    std::vector<float> dst_ref; // full reference output from the first (T=1) run
    for (int rep = 0; rep < 2; ++rep) {
        for (const auto & c : cases) {
            if (p.M % c.gm || (c.gm > 1 && (p.M / c.gm) % 4)) {
                continue;
            }
            if (c.claim > 0 && p.N % c.claim) {
                continue;
            }
            run_cfg cfg{ c.nth, c.gm, c.gn, c.mtile, iters, c.claim, c.label };
            double checksum = 0.0;
            const double gops = run_schedule(p, cfg, checksum);
            if (!have_ref) { ref = checksum; have_ref = true; dst_ref = p.dst; }
            // elementwise diff against the T=1 reference: counts missed/extra
            // tiles (a bare checksum can hide cancellation)
            int ndiff = 0;
            double maxdiff = 0.0;
            for (size_t i = 0; i < p.dst.size(); ++i) {
                const double d = std::abs((double) p.dst[i] - dst_ref[i]);
                if (d != 0.0) { ++ndiff; maxdiff = std::max(maxdiff, d); }
            }
            const bool exact = (checksum == ref) && ndiff == 0;
            printf("%-20s %6d %9.1f %14.6f %s%s\n", c.label, c.nth, gops, checksum,
                   rep ? "(rep1)" : "",
                   exact ? "" : ndiff ? "  <-- MISMATCH" : "  <-- cksum-only-diff");
            if (ndiff) {
                printf("    ndiff=%d maxdiff=%g\n", ndiff, maxdiff);
            }
            fflush(stdout);
        }
    }
    return 0;
}
