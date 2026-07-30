#pragma once

// llama-ep-dealer: pure-function slot dealer for expert-level dynamic
// scheduling (SCHEDULER-DESIGN.md §4.2).
//
// Each token arrives with k top-k expert slots from the router. the dealer
// assigns every slot to exactly one compute node — node 0 is the master
// (local chain), nodes 1..N are the slave endpoints — subject to:
//   - slot s may only go to a holder of expert ids[s] (holder bitmask);
//   - exactly m_star slots per token go local (static graph shape, §4.3);
//   - the result is a pure function of (ids, holders, m_star): two runs with
//     the same inputs produce the same plan, and any plan yields the same
//     output bits (§4.5), so load-aware variants stay word-identical.
//
// header-only so both the master (src/llama-remote-ep.cpp) and the standalone
// unit test (tools/epd/ep-dealer-test.cpp) share one implementation.

#include <cstdint>
#include <vector>

struct llama_ep_dealer_input {
    int n_tokens = 0;
    int k        = 0;  // slots (top-k) per token
    int n_endpoints = 0;  // slave endpoints (nodes 1..n_endpoints)
    int m_star   = 0;  // target local slots per token
    const int32_t  * ids     = nullptr;  // [k, n_tokens]: ids[t*k + j]
    const uint64_t * holders = nullptr;  // [n_expert] bitmask: bit0=master, bit(1+i)=endpoint i
};

struct llama_ep_dealer_ep {
    // assignments for one endpoint, ordered by (token, slot) ascending
    std::vector<int32_t> token;
    std::vector<int32_t> slot;
    std::vector<int32_t> expert;
};

struct llama_ep_dealer_plan {
    int m_local = 0;                    // == clamped m_star (uniform per token)
    std::vector<uint8_t> owner;         // [n_tokens*k]: 0=master, 1+i=endpoint i
    std::vector<int32_t> local_ids;     // [m_local, n_tokens]: expert ids in
                                        // ascending global slot order per token
    std::vector<llama_ep_dealer_ep> eps;  // [n_endpoints]
};

// fills out; returns false when the input is infeasible (an expert with no
// holder, or fewer master-capable slots than m_star on some token)
static inline bool llama_ep_dealer_plan_build(
        const llama_ep_dealer_input & in,
        llama_ep_dealer_plan        & out) {

    if (in.n_tokens < 1 || in.k < 1 || in.m_star < 0 || in.ids == nullptr || in.holders == nullptr) {
        return false;
    }
    const int m_local = in.m_star > in.k ? in.k : in.m_star;

    out.m_local = m_local;
    out.owner.assign((size_t) in.n_tokens * in.k, 0xff);
    out.local_ids.assign((size_t) in.n_tokens * m_local, -1);
    out.eps.clear();
    out.eps.resize((size_t) in.n_endpoints);

    for (int t = 0; t < in.n_tokens; ++t) {
        const int32_t * ids = in.ids + (size_t) t * in.k;

        // per-token endpoint load counters (balance within the token)
        std::vector<int> load((size_t) in.n_endpoints, 0);

        // pass 1: forced assignments — slots whose expert the master does not
        // hold must go to a slave holder (least loaded, lowest index on ties)
        for (int j = 0; j < in.k; ++j) {
            const uint64_t mask = in.holders[ids[j]];
            if (mask == 0) {
                return false; // nobody holds this expert
            }
            if (mask & 1u) {
                continue; // master-capable: decided in pass 2/3
            }
            int best = -1;
            for (int i = 0; i < in.n_endpoints; ++i) {
                if (mask & (2ull << i)) {
                    if (best < 0 || load[(size_t) i] < load[(size_t) best]) {
                        best = i;
                    }
                }
            }
            out.owner[(size_t) t * in.k + j] = (uint8_t) (1 + best);
            ++load[(size_t) best];
        }

        // pass 2: pick exactly m_local local slots among the master-capable
        // ones. forced-local (master is the only holder) first, then greedily
        // relieve the most-pressured side: for each endpoint, demand = the
        // number of remaining slots that have it as their ONLY slave holder;
        // master takes a slot from the highest-demand side (§4.2.2), ties by
        // ascending slot so the plan stays deterministic.
        std::vector<int> cand;
        for (int j = 0; j < in.k; ++j) {
            if (out.owner[(size_t) t * in.k + j] == 0xff) {
                cand.push_back(j);
            }
        }
        if ((int) cand.size() < m_local) {
            return false; // cannot fill the static local shape
        }
        std::vector<char> is_local((size_t) in.k, 0);
        int n_picked = 0;
        for (int j : cand) {
            if (in.holders[ids[j]] == 1u) {
                // forced-local; more of these than m_local is infeasible (the
                // static shape would overflow) — cannot happen when the master
                // holds a full replica (every mask has bit0)
                if (n_picked == m_local) {
                    return false;
                }
                is_local[(size_t) j] = 1;
                ++n_picked;
            }
        }
        while (n_picked < m_local) {
            std::vector<int> demand((size_t) in.n_endpoints, 0);
            for (int j : cand) {
                if (is_local[(size_t) j]) {
                    continue;
                }
                const uint64_t slave = in.holders[ids[j]] >> 1;
                if (slave != 0 && (slave & (slave - 1)) == 0) { // exactly one slave holder
                    int i = 0;
                    while (!((slave >> i) & 1u)) { ++i; }
                    ++demand[(size_t) i];
                }
            }
            int best = -1, best_score = -2;
            for (int j : cand) { // ascending: strict > keeps the lowest slot on ties
                if (is_local[(size_t) j]) {
                    continue;
                }
                const uint64_t slave = in.holders[ids[j]] >> 1;
                int score = -1; // multi-holder slots stay remote-able (keep flexibility)
                if (slave != 0 && (slave & (slave - 1)) == 0) {
                    int i = 0;
                    while (!((slave >> i) & 1u)) { ++i; }
                    score = demand[(size_t) i];
                }
                if (score > best_score) {
                    best_score = score;
                    best = j;
                }
            }
            is_local[(size_t) best] = 1;
            ++n_picked;
        }
        for (int j = 0, p = 0; j < in.k; ++j) {
            if (is_local[(size_t) j]) {
                out.owner[(size_t) t * in.k + j] = 0;
                out.local_ids[(size_t) t * m_local + p++] = ids[j];
            }
        }

        // pass 3: remaining slots go to their least-loaded slave holder
        for (int j = 0; j < in.k; ++j) {
            if (out.owner[(size_t) t * in.k + j] != 0xff) {
                continue;
            }
            const uint64_t mask = in.holders[ids[j]];
            int best = -1;
            for (int i = 0; i < in.n_endpoints; ++i) {
                if (mask & (2ull << i)) {
                    if (best < 0 || load[(size_t) i] < load[(size_t) best]) {
                        best = i;
                    }
                }
            }
            if (best < 0) {
                return false; // master-capable but master quota full and no slave holder
            }
            out.owner[(size_t) t * in.k + j] = (uint8_t) (1 + best);
            ++load[(size_t) best];
        }
    }

    // per-endpoint assignment lists, (token, slot) ascending by construction
    for (int t = 0; t < in.n_tokens; ++t) {
        for (int j = 0; j < in.k; ++j) {
            const uint8_t o = out.owner[(size_t) t * in.k + j];
            if (o == 0 || o == 0xff) {
                continue;
            }
            llama_ep_dealer_ep & ep = out.eps[(size_t) o - 1];
            ep.token.push_back(t);
            ep.slot.push_back(j);
            ep.expert.push_back(in.ids[(size_t) t * in.k + j]);
        }
    }
    return true;
}
