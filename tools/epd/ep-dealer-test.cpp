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

int main() {
    test_full_replica_1ep();
    test_partitioned_2ep();
    test_no_holder();
    test_all_local();
    test_two_tokens();
    test_infeasible_local();
    test_determinism();
    test_invalid_endpoint_count();
    if (failures == 0) {
        printf("ep-dealer-test: PASS (all cases)\n");
        return 0;
    }
    printf("ep-dealer-test: FAIL (%d checks)\n", failures);
    return 1;
}
