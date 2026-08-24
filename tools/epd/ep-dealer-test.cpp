// ep-dealer-test: unit tests for the pure-function slot dealer
// (tools/epd/llama-ep-dealer.h, SCHEDULER-DESIGN.md §4.2)

#include "llama-ep-dealer.h"

#include <cstdio>
#include <cstring>

static int failures = 0;

#define CHECK(cond, ...) do { \
    if (!(cond)) { \
        ++failures; \
        printf("FAIL %s:%d: ", __func__, __LINE__); \
        printf(__VA_ARGS__); \
        printf("\n"); \
    } \
} while (0)

// full replica everywhere, one endpoint: master takes the first m* slots
static void test_full_replica_1ep() {
    const int k = 8, n_exp = 256;
    int32_t ids[8] = {5, 9, 33, 64, 100, 128, 200, 255};
    uint64_t holders[n_exp];
    for (int e = 0; e < n_exp; ++e) {
        holders[e] = 0x3; // master + ep0
    }
    llama_ep_dealer_input in;
    in.n_tokens = 1; in.k = k; in.n_endpoints = 1; in.m_star = 2;
    in.ids = ids; in.holders = holders;
    llama_ep_dealer_plan p;
    CHECK(llama_ep_dealer_plan_build(in, p), "plan should succeed");
    CHECK(p.m_local == 2, "m_local=%d", p.m_local);
    CHECK(p.local_ids.size() == 2 && p.local_ids[0] == 5 && p.local_ids[1] == 9,
          "locals should be slots 0,1 (got %d,%d)", p.local_ids[0], p.local_ids[1]);
    CHECK(p.eps.size() == 1 && p.eps[0].slot.size() == 6, "ep0 should get 6 slots");
    for (size_t i = 0; i < p.eps[0].slot.size(); ++i) {
        CHECK(p.eps[0].slot[i] == (int32_t) i + 2, "ep0 slot %zu = %d", i, p.eps[0].slot[i]);
        CHECK(p.eps[0].expert[i] == ids[i + 2], "ep0 expert %zu", i);
    }
}

// partitioned holders, two endpoints: forced remote + balance
static void test_partitioned_2ep() {
    const int k = 8, n_exp = 256;
    // slots: 4 experts held by ep0 only, 4 by ep1 only; master holds all
    int32_t ids[8] = {10, 20, 30, 40, 130, 140, 150, 160};
    uint64_t holders[n_exp];
    for (int e = 0; e < n_exp; ++e) {
        holders[e] = 0x1 | (e < 128 ? 0x2 : 0x4); // master always; ep0 low half, ep1 high half
    }
    llama_ep_dealer_input in;
    in.n_tokens = 1; in.k = k; in.n_endpoints = 2; in.m_star = 2;
    in.ids = ids; in.holders = holders;
    llama_ep_dealer_plan p;
    CHECK(llama_ep_dealer_plan_build(in, p), "plan should succeed");
    int n_ep0 = 0, n_ep1 = 0, n_local = 0;
    for (int j = 0; j < k; ++j) {
        const int o = p.owner[(size_t) j];
        if (o == 0) { ++n_local; }
        if (o == 1) { ++n_ep0; CHECK(ids[j] < 128, "ep0 got expert %d", ids[j]); }
        if (o == 2) { ++n_ep1; CHECK(ids[j] >= 128, "ep1 got expert %d", ids[j]); }
    }
    CHECK(n_local == 2, "n_local=%d", n_local);
    CHECK(n_ep0 + n_ep1 == 6, "remote count %d+%d", n_ep0, n_ep1);
    // master should relieve the most-pressured side: both sides start with 4
    // exclusive slots, so master takes one from each → 3/3
    CHECK(n_ep0 == 3 && n_ep1 == 3, "balance 3/3, got %d/%d", n_ep0, n_ep1);
    // locals: slot 0 (first tie-break) then slot 4 (ep1 side, highest demand)
    CHECK(p.local_ids[0] == ids[0] && p.local_ids[1] == ids[4],
          "locals %d,%d", p.local_ids[0], p.local_ids[1]);
}

// expert with no holder must fail
static void test_no_holder() {
    const int k = 2, n_exp = 8;
    int32_t ids[2] = {1, 2};
    uint64_t holders[n_exp];
    memset(holders, 0, sizeof(holders));
    holders[1] = 0x3;
    llama_ep_dealer_input in;
    in.n_tokens = 1; in.k = k; in.n_endpoints = 1; in.m_star = 1;
    in.ids = ids; in.holders = holders;
    llama_ep_dealer_plan p;
    CHECK(!llama_ep_dealer_plan_build(in, p), "no-holder expert must fail");
}

// m* >= k: everything local, nothing remote
static void test_all_local() {
    const int k = 4, n_exp = 8;
    int32_t ids[4] = {1, 2, 3, 4};
    uint64_t holders[n_exp];
    for (int e = 0; e < n_exp; ++e) {
        holders[e] = 0x3;
    }
    llama_ep_dealer_input in;
    in.n_tokens = 1; in.k = k; in.n_endpoints = 1; in.m_star = 8;
    in.ids = ids; in.holders = holders;
    llama_ep_dealer_plan p;
    CHECK(llama_ep_dealer_plan_build(in, p), "plan should succeed");
    CHECK(p.m_local == k, "m_local=%d", p.m_local);
    CHECK(p.eps[0].slot.empty(), "nothing remote");
    for (int j = 0; j < k; ++j) {
        CHECK(p.local_ids[(size_t) j] == ids[j], "local %d", j);
    }
}

// two tokens are dealt independently; m* smaller than forced-remote count is fine
static void test_two_tokens() {
    const int k = 4, n_exp = 8;
    int32_t ids[8] = {1, 5, 2, 6, 3, 7, 4, 0}; // token0: 1,5,2,6; token1: 3,7,4,0
    uint64_t holders[n_exp];
    for (int e = 0; e < n_exp; ++e) {
        holders[e] = e < 4 ? 0x3 : 0x2; // 0-3: master+ep0; 4-7: ep0 only
    }
    llama_ep_dealer_input in;
    in.n_tokens = 2; in.k = k; in.n_endpoints = 1; in.m_star = 2;
    in.ids = ids; in.holders = holders;
    llama_ep_dealer_plan p;
    CHECK(llama_ep_dealer_plan_build(in, p), "plan should succeed");
    // token0: slots 1(5),3(6) forced remote; locals = slots 0(1),2(2)
    CHECK(p.owner[0] == 0 && p.owner[1] == 1 && p.owner[2] == 0 && p.owner[3] == 1,
          "token0 owners %d%d%d%d", p.owner[0], p.owner[1], p.owner[2], p.owner[3]);
    CHECK(p.local_ids[0] == 1 && p.local_ids[1] == 2, "token0 locals %d,%d",
          p.local_ids[0], p.local_ids[1]);
    // token1: slots 1(7) forced remote; master takes 2 of {0(3),2(4),3(0)}
    CHECK(p.owner[5] == 1, "token1 slot1 forced remote");
    int nl = 0;
    for (int j = 0; j < k; ++j) {
        if (p.owner[(size_t) k + j] == 0) { ++nl; }
    }
    CHECK(nl == 2, "token1 n_local=%d", nl);
    // ep list must be (token, slot) ascending
    for (size_t i = 1; i < p.eps[0].slot.size(); ++i) {
        const int prev = p.eps[0].token[i - 1] * 100 + p.eps[0].slot[i - 1];
        const int cur  = p.eps[0].token[i] * 100 + p.eps[0].slot[i];
        CHECK(prev < cur, "ep order at %zu", i);
    }
}

// infeasible: fewer master-capable slots than m*
static void test_infeasible_local() {
    const int k = 3, n_exp = 8;
    int32_t ids[3] = {1, 5, 6};
    uint64_t holders[n_exp];
    for (int e = 0; e < n_exp; ++e) {
        holders[e] = e < 4 ? 0x3 : 0x2;
    }
    llama_ep_dealer_input in;
    in.n_tokens = 1; in.k = k; in.n_endpoints = 1; in.m_star = 2; // only slot 0 master-capable
    in.ids = ids; in.holders = holders;
    llama_ep_dealer_plan p;
    CHECK(!llama_ep_dealer_plan_build(in, p), "m*>master-capable must fail");
}

// determinism: same input twice → identical plan
static void test_determinism() {
    const int k = 8, n_exp = 256;
    int32_t ids[8] = {5, 9, 33, 64, 100, 128, 200, 255};
    uint64_t holders[n_exp];
    for (int e = 0; e < n_exp; ++e) {
        holders[e] = 0x7; // master + both eps
    }
    llama_ep_dealer_input in;
    in.n_tokens = 1; in.k = k; in.n_endpoints = 2; in.m_star = 2;
    in.ids = ids; in.holders = holders;
    llama_ep_dealer_plan a, b;
    CHECK(llama_ep_dealer_plan_build(in, a) && llama_ep_dealer_plan_build(in, b), "plans");
    CHECK(a.owner == b.owner && a.local_ids == b.local_ids, "determinism");
    // 6 remote slots split over 2 endpoints: 3/3 by least-load
    CHECK(a.eps[0].slot.size() == 3 && a.eps[1].slot.size() == 3,
          "split %zu/%zu", a.eps[0].slot.size(), a.eps[1].slot.size());
}

static void test_invalid_endpoint_count() {
    int32_t ids[1] = {0};
    uint64_t holders[1] = {1};
    llama_ep_dealer_input in;
    in.n_tokens = 1; in.k = 1; in.n_endpoints = 64; in.m_star = 1;
    in.ids = ids; in.holders = holders;
    llama_ep_dealer_plan p;
    CHECK(!llama_ep_dealer_plan_build(in, p), "more than 63 endpoints must fail");
    in.n_endpoints = -1;
    CHECK(!llama_ep_dealer_plan_build(in, p), "negative endpoint count must fail");
}

// KLOCAL=0 pure EP: four disjoint shards own every slot; the master owns none.
static void test_pure_ep_4workers() {
    const int k = 8, n_tokens = 2, n_exp = 16, n_ep = 4;
    int32_t ids[n_tokens * k] = {
        0, 4, 8, 12, 3, 7, 11, 15,
        14, 10, 6, 2, 13, 9, 5, 1,
    };
    uint64_t holders[n_exp];
    for (int e = 0; e < n_exp; ++e) {
        holders[e] = 2ull << (e / 4); // one endpoint only; no master bit
    }

    llama_ep_dealer_input in;
    in.n_tokens = n_tokens; in.k = k; in.n_endpoints = n_ep; in.m_star = 0;
    in.ids = ids; in.holders = holders;
    llama_ep_dealer_plan a, b;
    CHECK(llama_ep_dealer_plan_build(in, a) && llama_ep_dealer_plan_build(in, b),
          "pure four-worker plans should succeed");
    CHECK(a.m_local == 0 && a.local_ids.empty(),
          "master must have no local expert slots");
    CHECK(a.owner == b.owner, "pure four-worker plan must be deterministic");
    for (int i = 0; i < n_tokens * k; ++i) {
        CHECK(a.owner[(size_t) i] == (uint8_t) (1 + ids[i] / 4),
              "slot %d expert %d owner=%u", i, ids[i], (unsigned) a.owner[(size_t) i]);
    }
    for (int i = 0; i < n_ep; ++i) {
        CHECK(a.eps[(size_t) i].slot.size() == 4,
              "endpoint %d selections=%zu", i, a.eps[(size_t) i].slot.size());
    }
}

static void test_exact_shard_cover() {
    const int32_t valid_first[4] = {0, 4, 8, 12};
    const int32_t valid_last [4] = {4, 8, 12, 16};
    CHECK(llama_ep_exact_shard_cover(16, 4, valid_first, valid_last),
          "four disjoint shards should exactly cover all experts");

    const int32_t gap_first[4] = {0, 4, 9, 12};
    CHECK(!llama_ep_exact_shard_cover(16, 4, gap_first, valid_last),
          "a gap must fail strict KLOCAL=0 topology validation");

    const int32_t overlap_last[4] = {5, 8, 12, 16};
    CHECK(!llama_ep_exact_shard_cover(16, 4, valid_first, overlap_last),
          "an overlap must fail strict KLOCAL=0 topology validation");
    CHECK(llama_ep_full_shard_cover(16, 4, valid_first, overlap_last),
          "max-effort topology should permit overlap when coverage is complete");

    const int32_t out_of_range_last[4] = {4, 8, 12, 17};
    CHECK(!llama_ep_exact_shard_cover(16, 4, valid_first, out_of_range_last),
          "an out-of-range shard must fail topology validation");
}

static void test_sparse_holder_cover() {
    uint64_t holders[16];
    for (int e = 0; e < 16; ++e) {
        holders[e] = 1ull << (e % 4); // 4n+r sparse exact cover
    }
    CHECK(llama_ep_holder_cover(16, 4, holders, true),
          "modulo sparse ownership should be an exact cover");
    holders[3] |= 1ull << 2;
    CHECK(!llama_ep_holder_cover(16, 4, holders, true),
          "sparse overlap must fail strict cover");
    CHECK(llama_ep_holder_cover(16, 4, holders, false),
          "sparse overlap is valid max-effort coverage");
    holders[7] = 0;
    CHECK(!llama_ep_holder_cover(16, 4, holders, false),
          "sparse gap must fail max-effort coverage");
    holders[7] = 1ull << 5;
    CHECK(!llama_ep_holder_cover(16, 4, holders, false),
          "out-of-endpoint holder bit must fail coverage");
}

static void test_pure_ep_overlap_balance() {
    const int k = 8, n_exp = 8, n_ep = 4;
    int32_t ids[k] = {0, 1, 2, 3, 4, 5, 6, 7};
    uint64_t holders[n_exp];
    for (int e = 0; e < n_exp; ++e) {
        holders[e] = 0x1e; // all four endpoints, no master
    }
    llama_ep_dealer_input in;
    in.n_tokens = 1; in.k = k; in.n_endpoints = n_ep; in.m_star = 0;
    in.ids = ids; in.holders = holders;
    llama_ep_dealer_plan p;
    CHECK(llama_ep_dealer_plan_build(in, p), "replicated pure-EP plan should succeed");
    for (int i = 0; i < n_ep; ++i) {
        CHECK(p.eps[(size_t) i].slot.size() == 2,
              "dynamic dealer should balance 8 slots as 2/2/2/2 (ep%d=%zu)",
              i, p.eps[(size_t) i].slot.size());
    }
}

static void test_replicated_tie_rotation() {
    uint64_t holders[256];
    for (int e = 0; e < 256; ++e) {
        holders[e] = 0x1e; // all four endpoints, no master
    }
    int totals[4] = {0, 0, 0, 0};
    for (int e = 0; e < 256; ++e) {
        int32_t id = e;
        llama_ep_dealer_input in;
        in.n_tokens = 1; in.k = 1; in.n_endpoints = 4; in.m_star = 0;
        in.ids = &id; in.holders = holders;
        llama_ep_dealer_plan p;
        CHECK(llama_ep_dealer_plan_build(in, p), "rotated plan for expert %d", e);
        if (!p.owner.empty() && p.owner[0] >= 1 && p.owner[0] <= 4) {
            ++totals[p.owner[0] - 1];
        }
    }
    CHECK(totals[0] == 64 && totals[1] == 64 && totals[2] == 64 && totals[3] == 64,
          "rotated ties should be unbiased, got %d/%d/%d/%d",
          totals[0], totals[1], totals[2], totals[3]);
}

// Replica balancing must span the complete PP batch. With a per-token reset,
// these overlapping holder masks produce 3/0/1/2 assignments; cumulative load
// produces the optimal 2/1/1/2 split.
static void test_batch_wide_replica_balance() {
    const int n_tokens = 2, k = 3, n_exp = 3, n_ep = 4;
    int32_t ids[n_tokens * k] = {0, 1, 2, 0, 1, 2};
    uint64_t holders[n_exp] = {
        (uint64_t) 0x5 << 1, // ep0 + ep2
        (uint64_t) 0xa << 1, // ep1 + ep3
        (uint64_t) 0x9 << 1, // ep0 + ep3
    };
    llama_ep_dealer_input in;
    in.n_tokens = n_tokens; in.k = k; in.n_endpoints = n_ep; in.m_star = 0;
    in.ids = ids; in.holders = holders;
    llama_ep_dealer_plan p;
    CHECK(llama_ep_dealer_plan_build(in, p), "batch-wide replica plan should succeed");
    const size_t expected[n_ep] = {2, 1, 1, 2};
    for (int i = 0; i < n_ep; ++i) {
        CHECK(p.eps[(size_t) i].slot.size() == expected[i],
              "batch endpoint %d expected %zu assignments, got %zu",
              i, expected[i], p.eps[(size_t) i].slot.size());
    }
}

static void test_inflight_load_avoidance() {
    const int k = 4, n_exp = 4, n_ep = 4;
    int32_t ids[k] = {0, 1, 2, 3};
    uint64_t holders[n_exp] = {0x1e, 0x1e, 0x1e, 0x1e};
    int64_t initial[n_ep] = {100, 0, 0, 0};
    llama_ep_dealer_input in;
    in.n_tokens = 1; in.k = k; in.n_endpoints = n_ep; in.m_star = 0;
    in.ids = ids; in.holders = holders; in.initial_remote_load = initial;
    llama_ep_dealer_plan p;
    CHECK(llama_ep_dealer_plan_build(in, p), "in-flight-aware plan should succeed");
    CHECK(p.eps[0].slot.empty(), "busy endpoint must receive no new work");
    CHECK(p.eps[1].slot.size() + p.eps[2].slot.size() + p.eps[3].slot.size() == k,
          "all assignments should go to idle endpoints");

    initial[0] = -1;
    CHECK(!llama_ep_dealer_plan_build(in, p), "negative initial load must fail");
}

static void test_service_rate_weighting() {
    const int k = 8, n_exp = 8, n_ep = 2;
    int32_t ids[k] = {0, 1, 2, 3, 4, 5, 6, 7};
    uint64_t holders[n_exp];
    for (int e = 0; e < n_exp; ++e) {
        holders[e] = 0x6; // both remote endpoints, no master
    }
    const int64_t cost[n_ep] = {4, 1}; // endpoint 0 takes 4x work per assignment
    llama_ep_dealer_input in;
    in.n_tokens = 1; in.k = k; in.n_endpoints = n_ep; in.m_star = 0;
    in.ids = ids; in.holders = holders; in.remote_assignment_cost = cost;
    llama_ep_dealer_plan p;
    CHECK(llama_ep_dealer_plan_build(in, p), "service-weighted plan should succeed");
    CHECK(p.eps[1].slot.size() > p.eps[0].slot.size(),
          "faster endpoint should receive more replicas, got %zu/%zu",
          p.eps[0].slot.size(), p.eps[1].slot.size());

    const int64_t invalid[n_ep] = {0, 1};
    in.remote_assignment_cost = invalid;
    CHECK(!llama_ep_dealer_plan_build(in, p), "zero assignment cost must fail");
}

static void test_decode_activation_penalty() {
    const int k = 8, n_exp = 8, n_ep = 4;
    int32_t ids[k] = {0, 1, 2, 3, 4, 5, 6, 7};
    uint64_t holders[n_exp];
    for (int e = 0; e < n_exp; ++e) {
        holders[e] = 0x1e; // all four remote endpoints, no master
    }
    const int64_t cost[n_ep] = {1000, 1000, 1000, 1000};
    const int64_t penalty[n_ep] = {4000, 4000, 4000, 4000};
    llama_ep_dealer_input in;
    in.n_tokens = 1; in.k = k; in.n_endpoints = n_ep; in.m_star = 0;
    in.ids = ids; in.holders = holders;
    in.remote_assignment_cost = cost;
    in.remote_activation_penalty = penalty;
    llama_ep_dealer_plan p;
    CHECK(llama_ep_dealer_plan_build(in, p), "activation-aware plan should succeed");
    int fanout = 0;
    for (const auto & ep : p.eps) {
        fanout += !ep.slot.empty();
    }
    CHECK(fanout == 2, "activation penalty should reduce fanout to two, got %d", fanout);

    const int64_t invalid[n_ep] = {-1, 0, 0, 0};
    in.remote_activation_penalty = invalid;
    CHECK(!llama_ep_dealer_plan_build(in, p), "negative activation penalty must fail");
}

static void test_repeat_expert_affinity() {
    const int n_tokens = 2, k = 2, n_exp = 2, n_ep = 2;
    int32_t ids[n_tokens * k] = {0, 1, 0, 1};
    uint64_t holders[n_exp] = {0x6, 0x6};
    const int64_t cost[n_ep] = {1000, 1000};
    const int64_t repeat[n_ep] = {100, 100};
    llama_ep_dealer_input in;
    in.n_tokens = n_tokens; in.k = k; in.n_endpoints = n_ep; in.m_star = 0;
    in.ids = ids; in.holders = holders;
    in.remote_assignment_cost = cost;
    in.remote_repeat_assignment_cost = repeat;
    llama_ep_dealer_plan p;
    CHECK(llama_ep_dealer_plan_build(in, p), "repeat-aware plan should succeed");
    CHECK(p.owner[0] == p.owner[2] && p.owner[1] == p.owner[3],
          "repeated experts should keep endpoint affinity, owners=%d/%d/%d/%d",
          p.owner[0], p.owner[1], p.owner[2], p.owner[3]);

    const int64_t invalid[n_ep] = {1001, 100};
    in.remote_repeat_assignment_cost = invalid;
    CHECK(!llama_ep_dealer_plan_build(in, p), "repeat cost above full cost must fail");
}

// Prefill uses the same repeat-aware greedy policy without the small-TG local
// refinement pass. Rows of two hot experts should stay on their respective
// replicas across a multi-token batch, halving duplicate weight streams while
// preserving an even assignment count.
static void test_prefill_repeat_affinity() {
    const int n_tokens = 16, k = 2, n_exp = 2, n_ep = 2;
    int32_t ids[n_tokens * k];
    for (int t = 0; t < n_tokens; ++t) {
        ids[t * k + 0] = 0;
        ids[t * k + 1] = 1;
    }
    uint64_t holders[n_exp] = {0x6, 0x6};
    const int64_t cost[n_ep] = {1000, 1000};
    const int64_t repeat[n_ep] = {250, 250};
    llama_ep_dealer_input in;
    in.n_tokens = n_tokens; in.k = k; in.n_endpoints = n_ep; in.m_star = 0;
    in.ids = ids; in.holders = holders;
    in.remote_assignment_cost = cost;
    in.remote_repeat_assignment_cost = repeat;
    llama_ep_dealer_plan p;
    CHECK(llama_ep_dealer_plan_build(in, p), "prefill repeat-aware plan should succeed");
    CHECK(p.eps[0].slot.size() == n_tokens && p.eps[1].slot.size() == n_tokens,
          "prefill affinity should retain even rows, got %zu/%zu",
          p.eps[0].slot.size(), p.eps[1].slot.size());
    for (int expert = 0; expert < n_exp; ++expert) {
        int holders_used = 0;
        for (int endpoint = 0; endpoint < n_ep; ++endpoint) {
            bool found = false;
            for (const int32_t assigned : p.eps[(size_t) endpoint].expert) {
                found = found || assigned == expert;
            }
            holders_used += found;
        }
        CHECK(holders_used == 1, "prefill expert %d should use one replica, got %d", expert, holders_used);
    }
}

static void test_decode_global_replica_refinement() {
    const int n_tokens = 2, k = 6, n_exp = 225, n_ep = 4;
    int32_t ids[n_tokens * k] = {
        64, 182, 178, 110, 50, 34,
        64, 182, 178, 170, 110, 224,
    };
    uint64_t holders[n_exp] = {};
    holders[64]  = 0x1a; // endpoints 0, 2, 3
    holders[182] = 0x0a; // endpoints 0, 2
    holders[178] = 0x10; // endpoint 3
    holders[110] = 0x14; // endpoints 1, 3
    holders[50]  = 0x18; // endpoints 2, 3
    holders[34]  = 0x10; // endpoint 3
    holders[170] = 0x18; // endpoints 2, 3
    holders[224] = 0x08; // endpoint 2
    const int64_t cost[n_ep] = {1000, 1000, 1000, 1000};
    const int64_t repeat[n_ep] = {250, 250, 250, 250};
    llama_ep_dealer_input in;
    in.n_tokens = n_tokens; in.k = k; in.n_endpoints = n_ep; in.m_star = 0;
    in.ids = ids; in.holders = holders;
    in.remote_assignment_cost = cost;
    in.remote_repeat_assignment_cost = repeat;
    llama_ep_dealer_plan p;
    CHECK(llama_ep_dealer_plan_build(in, p), "global replica refinement plan should succeed");

    int32_t counts[n_ep][n_exp] = {};
    for (int pos = 0; pos < n_tokens * k; ++pos) {
        const int endpoint = (int) p.owner[(size_t) pos] - 1;
        CHECK(endpoint >= 0 && endpoint < n_ep, "pure EP slot must remain remote");
        CHECK(holders[ids[pos]] & (2ull << endpoint), "refined owner must hold its expert");
        ++counts[endpoint][ids[pos]];
    }
    int64_t critical = 0;
    for (int endpoint = 0; endpoint < n_ep; ++endpoint) {
        int64_t load = 0;
        for (int expert = 0; expert < n_exp; ++expert) {
            if (counts[endpoint][expert] > 0) {
                load += cost[endpoint] + (counts[endpoint][expert] - 1) * repeat[endpoint];
            }
        }
        if (load > critical) {
            critical = load;
        }
    }
    CHECK(critical == 3000, "refinement should reduce critical work to 3000, got %lld",
          (long long) critical);
}

static void test_reusable_workspace_is_transparent() {
    const int n_tokens = 2, k = 8, n_exp = 16, n_ep = 4;
    int32_t ids[n_tokens * k] = {
        0, 1, 2, 3, 4, 5, 6, 7,
        0, 1, 8, 9, 10, 11, 12, 13,
    };
    uint64_t holders[n_exp];
    for (int e = 0; e < n_exp; ++e) {
        holders[e] = 0x1e;
    }
    const int64_t cost[n_ep] = {1000, 1000, 1000, 1000};
    const int64_t repeat[n_ep] = {250, 250, 250, 250};

    llama_ep_dealer_input in;
    in.n_tokens = n_tokens; in.k = k; in.n_endpoints = n_ep; in.m_star = 0;
    in.ids = ids; in.holders = holders;
    in.remote_assignment_cost = cost;
    in.remote_repeat_assignment_cost = repeat;

    llama_ep_dealer_plan expected;
    CHECK(llama_ep_dealer_plan_build(in, expected), "convenience plan should succeed");

    llama_ep_dealer_plan plan;
    llama_ep_dealer_workspace workspace;
    CHECK(llama_ep_dealer_plan_build(in, plan, workspace), "workspace plan should succeed");
    CHECK(plan.owner == expected.owner && plan.local_ids == expected.local_ids,
          "workspace API must preserve owner/local plan");
    for (int i = 0; i < n_ep; ++i) {
        CHECK(plan.eps[(size_t) i].token == expected.eps[(size_t) i].token &&
              plan.eps[(size_t) i].slot == expected.eps[(size_t) i].slot &&
              plan.eps[(size_t) i].expert == expected.eps[(size_t) i].expert,
              "workspace API endpoint %d differs", i);
    }

    const uint8_t * owner_data = plan.owner.data();
    const int32_t * ep0_data = plan.eps[0].token.data();
    const int64_t * load_data = workspace.load.data();
    const int32_t * counts_data = workspace.counts.data();
    for (int iteration = 0; iteration < 64; ++iteration) {
        CHECK(llama_ep_dealer_plan_build(in, plan, workspace),
              "workspace rebuild %d should succeed", iteration);
        CHECK(plan.owner == expected.owner, "workspace rebuild %d changed owners", iteration);
        CHECK(plan.owner.data() == owner_data && plan.eps[0].token.data() == ep0_data &&
              workspace.load.data() == load_data && workspace.counts.data() == counts_data,
              "workspace rebuild %d unexpectedly reallocated stable buffers", iteration);
    }
}

static size_t assignment_count(const llama_ep_dealer_plan & plan) {
    size_t total = 0;
    for (int32_t count : plan.local_count) {
        total += (size_t) count;
    }
    for (const auto & ep : plan.eps) {
        total += ep.slot.size();
    }
    return total;
}

static bool same_endpoint_plan(const llama_ep_dealer_plan & a, const llama_ep_dealer_plan & b) {
    if (a.eps.size() != b.eps.size()) return false;
    for (size_t i = 0; i < a.eps.size(); ++i) {
        if (a.eps[i].token != b.eps[i].token || a.eps[i].slot != b.eps[i].slot || a.eps[i].expert != b.eps[i].expert) {
            return false;
        }
    }
    return true;
}

static void test_active_mask_null_is_transparent() {
    const int n_tokens = 2, k = 6, n_exp = 16, n_ep = 3;
    int32_t ids[n_tokens*k] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
    uint64_t holders[n_exp];
    for (int expert = 0; expert < n_exp; ++expert) holders[expert] = 0xf;
    uint8_t all_active[n_tokens*k];
    memset(all_active, 1, sizeof(all_active));
    llama_ep_dealer_input in;
    in.n_tokens = n_tokens; in.k = k; in.n_endpoints = n_ep; in.m_star = 2;
    in.ids = ids; in.holders = holders;
    llama_ep_dealer_plan legacy, masked;
    CHECK(llama_ep_dealer_plan_build(in, legacy), "legacy active plan should succeed");
    in.active_mask = all_active;
    CHECK(llama_ep_dealer_plan_build(in, masked), "all-active plan should succeed");
    CHECK(legacy.owner == masked.owner && legacy.local_ids == masked.local_ids &&
          legacy.local_count == masked.local_count && same_endpoint_plan(legacy, masked),
          "nullptr and all-active mask must be bit-identical");
    CHECK(assignment_count(masked) == (size_t) n_tokens*k, "all-active assignment count=%zu", assignment_count(masked));
}

static void test_active_mask_ragged_pure_ep() {
    const int n_tokens = 3, k = 5, n_exp = 16, n_ep = 2;
    int32_t ids[n_tokens*k] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14};
    uint8_t active[n_tokens*k] = {1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 1, 1, 1, 0};
    uint64_t holders[n_exp];
    for (int expert = 0; expert < n_exp; ++expert) holders[expert] = 2ull << (expert & 1);
    llama_ep_dealer_input in;
    in.n_tokens = n_tokens; in.k = k; in.n_endpoints = n_ep; in.m_star = 0;
    in.ids = ids; in.holders = holders; in.active_mask = active;
    llama_ep_dealer_plan plan;
    CHECK(llama_ep_dealer_plan_build(in, plan), "ragged pure-EP active plan should succeed");
    size_t active_count = 0;
    for (int pos = 0; pos < n_tokens*k; ++pos) {
        active_count += active[pos] != 0;
        if (!active[pos]) {
            CHECK(plan.owner[(size_t) pos] == 0xff, "inactive position %d owner=%u", pos, (unsigned) plan.owner[(size_t) pos]);
        } else {
            const int endpoint = (int) plan.owner[(size_t) pos] - 1;
            CHECK(endpoint >= 0 && endpoint < n_ep, "active position %d must be remote", pos);
            CHECK(holders[ids[pos]] & (2ull << endpoint), "active position %d assigned to non-holder", pos);
        }
    }
    CHECK(assignment_count(plan) == active_count, "ragged assignments=%zu active=%zu", assignment_count(plan), active_count);
}

static void test_active_mask_ragged_local_and_all_inactive() {
    const int n_tokens = 2, k = 4, n_exp = 8;
    int32_t ids[n_tokens*k] = {1, 2, 3, 4, 4, 5, 6, 7};
    uint8_t active[n_tokens*k] = {0, 1, 0, 0, 1, 1, 1, 0};
    uint64_t holders[n_exp];
    for (int expert = 0; expert < n_exp; ++expert) holders[expert] = 0x3;
    llama_ep_dealer_input in;
    in.n_tokens = n_tokens; in.k = k; in.n_endpoints = 1; in.m_star = 2;
    in.ids = ids; in.holders = holders; in.active_mask = active;
    llama_ep_dealer_plan plan;
    CHECK(llama_ep_dealer_plan_build(in, plan), "ragged local plan should succeed");
    CHECK(plan.local_count == std::vector<int32_t>({1, 2}), "local counts should be 1/2");
    CHECK(plan.local_ids[0] == 2 && plan.local_ids[1] == -1, "token0 local row must preserve inactive tail");
    CHECK(assignment_count(plan) == 4, "ragged local assignment count=%zu", assignment_count(plan));

    memset(active, 0, sizeof(active));
    for (int & id : ids) id = -1; // inactive ids must never index holders
    CHECK(llama_ep_dealer_plan_build(in, plan), "all-inactive plan should ignore ids/holders");
    CHECK(assignment_count(plan) == 0, "all-inactive assignments=%zu", assignment_count(plan));
    for (uint8_t owner : plan.owner) CHECK(owner == 0xff, "all-inactive owner=%u", (unsigned) owner);
}

static void test_active_mask_holder_feasibility() {
    int32_t ids[2] = {1, 2};
    uint8_t active[2] = {1, 0};
    uint64_t holders[3] = {0, 0x2, 0};
    llama_ep_dealer_input in;
    in.n_tokens = 1; in.k = 2; in.n_endpoints = 1; in.m_star = 0;
    in.ids = ids; in.holders = holders; in.active_mask = active;
    llama_ep_dealer_plan plan;
    CHECK(llama_ep_dealer_plan_build(in, plan), "inactive no-holder slot must be ignored");
    CHECK(assignment_count(plan) == 1 && plan.owner[1] == 0xff, "partial-active assignment count");
    active[1] = 1;
    CHECK(!llama_ep_dealer_plan_build(in, plan), "active no-holder slot must fail");
}

int main() {
    test_full_replica_1ep();
    test_partitioned_2ep();
    test_no_holder();
    test_all_local();
    test_two_tokens();
    test_infeasible_local();
    test_determinism();
    test_invalid_endpoint_count();
    test_pure_ep_4workers();
    test_exact_shard_cover();
    test_sparse_holder_cover();
    test_pure_ep_overlap_balance();
    test_replicated_tie_rotation();
    test_batch_wide_replica_balance();
    test_inflight_load_avoidance();
    test_service_rate_weighting();
    test_decode_activation_penalty();
    test_decode_global_replica_refinement();
    test_repeat_expert_affinity();
    test_prefill_repeat_affinity();
    test_reusable_workspace_is_transparent();
    test_active_mask_null_is_transparent();
    test_active_mask_ragged_pure_ep();
    test_active_mask_ragged_local_and_all_inactive();
    test_active_mask_holder_feasibility();
    if (failures == 0) {
        printf("ep-dealer-test: PASS (all cases)\n");
        return 0;
    }
    printf("ep-dealer-test: FAIL (%d checks)\n", failures);
    return 1;
}
