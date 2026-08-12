#pragma once

#include "llama-ep-topology.h"

// llama-ep-dealer: pure-function slot dealer for expert-level dynamic
// scheduling (SCHEDULER-DESIGN.md §4.2).
//
// Each token arrives with k top-k expert slots from the router. the dealer
// assigns every slot to exactly one compute node — node 0 is the master
// (local chain), nodes 1..N are the slave endpoints — subject to:
//   - slot s may only go to a holder of expert ids[s] (holder bitmask);
//   - exactly m_star slots per token go local (static graph shape, §4.3);
//   - the result is a pure function of (ids, holders, m_star, initial load):
//     two runs with the same inputs produce the same plan, and any valid plan
//     yields the same output bits (§4.5), so load-aware placement remains
//     numerically transparent.
//
// header-only so both the master (src/llama-remote-ep.cpp) and the standalone
// unit test (tools/epd/ep-dealer-test.cpp) share one implementation.

#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <limits>
#include <vector>

struct llama_ep_dealer_input {
    int n_tokens = 0;
    int k        = 0;  // slots (top-k) per token
    int n_endpoints = 0;  // slave endpoints (nodes 1..n_endpoints)
    int m_star   = 0;  // target local slots per token
    const int32_t  * ids     = nullptr;  // [k, n_tokens]: ids[t*k + j]
    const uint64_t * holders = nullptr;  // [n_expert] bitmask: bit0=master, bit(1+i)=endpoint i
    // Optional work already queued/running on each endpoint, in common
    // scheduler work units. The dealer adds the current batch on top so
    // concurrent EP streams avoid piling onto an already busy worker.
    const int64_t * initial_remote_load = nullptr;  // [n_endpoints], non-negative
    // Optional positive work units charged for one new assignment on each
    // endpoint. This makes replicated-expert placement service-rate aware:
    // slower NUMA workers accumulate virtual load faster. Null means unit cost.
    const int64_t * remote_assignment_cost = nullptr; // [n_endpoints], positive
    // Optional non-negative cost added only while an endpoint has no slot in
    // the current plan. This is a fanout regularizer, not endpoint work: it
    // models the master-side fixed cost of opening another request. Null means
    // no penalty and preserves pure least-load placement.
    const int64_t * remote_activation_penalty = nullptr; // [n_endpoints], non-negative
    // Optional marginal work for another row of an expert already assigned to
    // the same endpoint in this batch. Null charges the normal assignment cost.
    const int64_t * remote_repeat_assignment_cost = nullptr; // [n_endpoints], positive
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

// Reusable scratch for the hot per-layer scheduling path.  A decode step runs
// the dealer once for every MoE layer; keeping these buffers per EP stream
// avoids rebuilding a large set of small vectors thousands of times per
// second.  The workspace contains no persistent scheduling state, so reusing it
// cannot change a plan for identical inputs.
struct llama_ep_dealer_workspace {
    std::vector<int64_t> load;
    std::vector<int64_t> cost;
    std::vector<int64_t> activation_penalty;
    std::vector<int64_t> repeat_cost;
    std::vector<int64_t> next_cost;
    std::vector<int64_t> demand;
    std::vector<int64_t> refined_load;
    std::vector<int64_t> best_load;
    std::vector<int64_t> candidate_load;
    std::vector<uint8_t> active;
    std::vector<uint8_t> seen;
    std::vector<int32_t> counts;
    std::vector<int>     candidates;
    std::vector<char>    is_local;
};

// Least-load placement with a deterministic rotated tie-break. Always choosing
// the lowest endpoint makes replicated KLOCAL=0 topologies accumulate every
// odd-slot remainder on the same workers across decode tokens. Rotating from a
// stable (expert,slot,token) hash preserves reproducibility without that bias.
static inline int llama_ep_dealer_pick_remote(
        uint64_t mask,
        const std::vector<int64_t> & load,
        const std::vector<int64_t> & activation_penalty,
        const std::vector<uint8_t> & active,
        const std::vector<int64_t> * next_cost,
        int n_endpoints,
        uint32_t salt) {
    if (n_endpoints < 1) {
        return -1;
    }
    const int start = (int) (salt % (uint32_t) n_endpoints);
    int best = -1;
    int64_t best_score = 0;
    for (int step = 0; step < n_endpoints; ++step) {
        const int i = (start + step) % n_endpoints;
        if (mask & (2ull << i)) {
            const int64_t penalty = active[(size_t) i] ? 0 : activation_penalty[(size_t) i];
            int64_t score = load[(size_t) i] > std::numeric_limits<int64_t>::max() - penalty
                ? std::numeric_limits<int64_t>::max()
                : load[(size_t) i] + penalty;
            if (next_cost != nullptr) {
                const int64_t increment = (*next_cost)[(size_t) i];
                score = score > std::numeric_limits<int64_t>::max() - increment
                    ? std::numeric_limits<int64_t>::max()
                    : score + increment;
            }
            if (best < 0 || score < best_score) {
                best = i;
                best_score = score;
            }
        }
    }
    return best;
}

// fills out; returns false when the input is infeasible (an expert with no
// holder, or fewer master-capable slots than m_star on some token)
static inline bool llama_ep_dealer_plan_build(
        const llama_ep_dealer_input & in,
        llama_ep_dealer_plan        & out,
        llama_ep_dealer_workspace   & workspace) {

    if (in.n_tokens < 1 || in.k < 1 || in.n_endpoints < 0 || in.n_endpoints > 63 ||
            in.m_star < 0 || in.ids == nullptr || in.holders == nullptr) {
        return false;
    }
    const int m_local = in.m_star > in.k ? in.k : in.m_star;

    out.m_local = m_local;
    out.owner.assign((size_t) in.n_tokens * in.k, 0xff);
    out.local_ids.assign((size_t) in.n_tokens * m_local, -1);
    if (out.eps.size() != (size_t) in.n_endpoints) {
        out.eps.resize((size_t) in.n_endpoints);
    }
    for (llama_ep_dealer_ep & ep : out.eps) {
        ep.token.clear();
        ep.slot.clear();
        ep.expert.clear();
    }

    // Keep one load vector for the whole batch. Resetting it per token only
    // balances each top-k group in isolation and can leave a PP request badly
    // skewed when replica holder masks overlap unevenly.
    std::vector<int64_t> & load = workspace.load;
    std::vector<int64_t> & cost = workspace.cost;
    std::vector<int64_t> & activation_penalty = workspace.activation_penalty;
    std::vector<uint8_t> & active = workspace.active;
    load.assign((size_t) in.n_endpoints, 0);
    cost.assign((size_t) in.n_endpoints, 1);
    activation_penalty.assign((size_t) in.n_endpoints, 0);
    active.assign((size_t) in.n_endpoints, 0);
    if (in.initial_remote_load != nullptr) {
        for (int i = 0; i < in.n_endpoints; ++i) {
            if (in.initial_remote_load[i] < 0) {
                return false;
            }
            load[(size_t) i] = in.initial_remote_load[i];
        }
    }
    if (in.remote_assignment_cost != nullptr) {
        for (int i = 0; i < in.n_endpoints; ++i) {
            if (in.remote_assignment_cost[i] <= 0) {
                return false;
            }
            cost[(size_t) i] = in.remote_assignment_cost[i];
        }
    }
    if (in.remote_activation_penalty != nullptr) {
        for (int i = 0; i < in.n_endpoints; ++i) {
            if (in.remote_activation_penalty[i] < 0) {
                return false;
            }
            activation_penalty[(size_t) i] = in.remote_activation_penalty[i];
        }
    }
    const bool repeat_aware = in.remote_repeat_assignment_cost != nullptr;
    std::vector<int64_t> & repeat_cost = workspace.repeat_cost;
    repeat_cost.assign((size_t) in.n_endpoints, 0);
    if (repeat_aware) {
        for (int i = 0; i < in.n_endpoints; ++i) {
            if (in.remote_repeat_assignment_cost[i] <= 0 ||
                    in.remote_repeat_assignment_cost[i] > cost[(size_t) i]) {
                return false;
            }
            repeat_cost[(size_t) i] = in.remote_repeat_assignment_cost[i];
        }
    }

    int32_t max_expert = 0;
    if (repeat_aware) {
        for (int i = 0; i < in.n_tokens * in.k; ++i) {
            max_expert = std::max(max_expert, in.ids[i]);
        }
    }
    const size_t expert_stride = repeat_aware ? (size_t) max_expert + 1 : 0;
    std::vector<uint8_t> & seen = workspace.seen;
    seen.assign(repeat_aware ? (size_t) in.n_endpoints * expert_stride : 0, 0);
    std::vector<int64_t> & next_cost = workspace.next_cost;
    next_cost.assign((size_t) in.n_endpoints, 0);

    auto was_seen = [&](int endpoint, int32_t expert) -> uint8_t & {
        return seen[(size_t) endpoint * expert_stride + (size_t) expert];
    };

    auto prepare_next_cost = [&](int32_t expert) -> const std::vector<int64_t> * {
        if (!repeat_aware) {
            return nullptr;
        }
        for (int i = 0; i < in.n_endpoints; ++i) {
            next_cost[(size_t) i] = was_seen(i, expert)
                ? repeat_cost[(size_t) i]
                : cost[(size_t) i];
        }
        return &next_cost;
    };

    auto add_assignment = [&](int endpoint, int32_t expert) {
        const int64_t increment = repeat_aware && was_seen(endpoint, expert)
            ? repeat_cost[(size_t) endpoint]
            : cost[(size_t) endpoint];
        if (load[(size_t) endpoint] > std::numeric_limits<int64_t>::max() - increment) {
            return false;
        }
        load[(size_t) endpoint] += increment;
        if (repeat_aware) {
            was_seen(endpoint, expert) = 1;
        }
        active[(size_t) endpoint] = 1;
        return true;
    };

    for (int t = 0; t < in.n_tokens; ++t) {
        const int32_t * ids = in.ids + (size_t) t * in.k;

        // pass 1: forced assignments — slots whose expert the master does not
        // hold must go to a slave holder (least loaded, rotated stable tie)
        for (int j = 0; j < in.k; ++j) {
            const uint64_t mask = in.holders[ids[j]];
            if (mask == 0) {
                return false; // nobody holds this expert
            }
            if (mask & 1u) {
                continue; // master-capable: decided in pass 2/3
            }
            const uint32_t salt = (uint32_t) ids[j] * 0x9e3779b1u ^
                                  (uint32_t) j      * 0x85ebca6bu ^
                                  (uint32_t) t      * 0xc2b2ae35u;
            const int best = llama_ep_dealer_pick_remote(
                    mask, load, activation_penalty, active, prepare_next_cost(ids[j]), in.n_endpoints, salt);
            if (best < 0) {
                return false;
            }
            out.owner[(size_t) t * in.k + j] = (uint8_t) (1 + best);
            if (!add_assignment(best, ids[j])) {
                return false;
            }
        }

        // pass 2: pick exactly m_local local slots among the master-capable
        // ones. forced-local (master is the only holder) first, then greedily
        // relieve the most-pressured side: for each endpoint, demand = the
        // number of remaining slots that have it as their ONLY slave holder;
        // master takes a slot from the highest-demand side (§4.2.2), ties by
        // ascending slot so the plan stays deterministic.
        std::vector<int> & cand = workspace.candidates;
        cand.clear();
        for (int j = 0; j < in.k; ++j) {
            if (out.owner[(size_t) t * in.k + j] == 0xff) {
                cand.push_back(j);
            }
        }
        if ((int) cand.size() < m_local) {
            return false; // cannot fill the static local shape
        }
        std::vector<char> & is_local = workspace.is_local;
        is_local.assign((size_t) in.k, 0);
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
            std::vector<int64_t> & demand = workspace.demand;
            demand.assign((size_t) in.n_endpoints, 0);
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
            int best = -1;
            int64_t best_score = std::numeric_limits<int64_t>::min();
            for (int j : cand) { // ascending: strict > keeps the lowest slot on ties
                if (is_local[(size_t) j]) {
                    continue;
                }
                const uint64_t slave = in.holders[ids[j]] >> 1;
                int64_t score = -1; // multi-holder slots stay remote-able (keep flexibility)
                if (slave != 0 && (slave & (slave - 1)) == 0) {
                    int i = 0;
                    while (!((slave >> i) & 1u)) { ++i; }
                    // Relieve the endpoint that would be most loaded after
                    // this token, including earlier tokens/in-flight work.
                    const int64_t room = std::numeric_limits<int64_t>::max() - load[(size_t) i];
                    score = demand[(size_t) i] > room / cost[(size_t) i]
                        ? std::numeric_limits<int64_t>::max()
                        : load[(size_t) i] + demand[(size_t) i] * cost[(size_t) i];
                }
                if (score > best_score) {
                    best_score = score;
                    best = j;
                }
            }
            if (best < 0) {
                return false;
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
            const uint32_t salt = (uint32_t) ids[j] * 0x9e3779b1u ^
                                  (uint32_t) j      * 0x85ebca6bu ^
                                  (uint32_t) t      * 0xc2b2ae35u;
            const int best = llama_ep_dealer_pick_remote(
                    mask, load, activation_penalty, active, prepare_next_cost(ids[j]), in.n_endpoints, salt);
            if (best < 0) {
                return false; // master-capable but master quota full and no slave holder
            }
            out.owner[(size_t) t * in.k + j] = (uint8_t) (1 + best);
            if (!add_assignment(best, ids[j])) {
                return false;
            }
        }
    }

    // Small pure-EP decode batches are latency-bound by the slowest worker.
    // The streaming greedy pass above is intentionally cheap, but replica
    // constraints can make an early tie choice leave a later worker with an
    // avoidable extra weight stream.  Refine that plan with deterministic
    // single-slot moves.  This is restricted to the repeat-aware TG path;
    // PP batches keep the O(n) planner and activation-penalty experiments keep
    // their requested fanout policy.
    if (m_local == 0 && repeat_aware && in.remote_activation_penalty == nullptr && in.n_tokens <= 4) {
        const size_t n_assignments = (size_t) in.n_tokens * in.k;
        std::vector<int32_t> & counts = workspace.counts;
        std::vector<int64_t> & refined_load = workspace.refined_load;
        std::vector<int64_t> & best_load = workspace.best_load;
        std::vector<int64_t> & candidate = workspace.candidate_load;
        counts.assign((size_t) in.n_endpoints * expert_stride, 0);
        refined_load.assign((size_t) in.n_endpoints, 0);
        best_load.resize((size_t) in.n_endpoints);
        candidate.resize((size_t) in.n_endpoints);
        if (in.initial_remote_load != nullptr) {
            for (int i = 0; i < in.n_endpoints; ++i) {
                refined_load[(size_t) i] = in.initial_remote_load[i];
            }
        }
        for (size_t pos = 0; pos < n_assignments; ++pos) {
            const int endpoint = (int) out.owner[pos] - 1;
            if (endpoint < 0 || endpoint >= in.n_endpoints) {
                return false;
            }
            ++counts[(size_t) endpoint * expert_stride + (size_t) in.ids[pos]];
        }
        for (int endpoint = 0; endpoint < in.n_endpoints; ++endpoint) {
            for (int32_t expert = 0; expert <= max_expert; ++expert) {
                const int32_t n = counts[(size_t) endpoint * expert_stride + (size_t) expert];
                if (n == 0) {
                    continue;
                }
                const int64_t repeated = (int64_t) (n - 1) * repeat_cost[(size_t) endpoint];
                if (refined_load[(size_t) endpoint] >
                        std::numeric_limits<int64_t>::max() - cost[(size_t) endpoint] - repeated) {
                    return false;
                }
                refined_load[(size_t) endpoint] += cost[(size_t) endpoint] + repeated;
            }
        }

        // Lexicographically compare the descending endpoint loads.  This is
        // an integer-only makespan objective: minimize the critical worker,
        // then the second slowest, and so on, without overflow-prone squares.
        auto load_is_better = [&](const std::vector<int64_t> & lhs, const std::vector<int64_t> & rhs) {
            assert(lhs.size() == rhs.size() && lhs.size() <= 63);
            std::array<int64_t, 63> lhs_sorted{};
            std::array<int64_t, 63> rhs_sorted{};
            std::copy(lhs.begin(), lhs.end(), lhs_sorted.begin());
            std::copy(rhs.begin(), rhs.end(), rhs_sorted.begin());
            for (size_t rank = 0; rank < lhs.size(); ++rank) {
                size_t lhs_max = rank;
                size_t rhs_max = rank;
                for (size_t i = rank + 1; i < lhs.size(); ++i) {
                    if (lhs_sorted[i] > lhs_sorted[lhs_max]) {
                        lhs_max = i;
                    }
                    if (rhs_sorted[i] > rhs_sorted[rhs_max]) {
                        rhs_max = i;
                    }
                }
                const int64_t lhs_value = lhs_sorted[lhs_max];
                const int64_t rhs_value = rhs_sorted[rhs_max];
                lhs_sorted[lhs_max] = lhs_sorted[rank];
                rhs_sorted[rhs_max] = rhs_sorted[rank];
                lhs_sorted[rank] = lhs_value;
                rhs_sorted[rank] = rhs_value;
                if (lhs_value != rhs_value) {
                    return lhs_value < rhs_value;
                }
            }
            return false;
        };

        for (;;) {
            size_t best_pos = n_assignments;
            int best_endpoint = -1;
            best_load = refined_load;
            for (size_t pos = 0; pos < n_assignments; ++pos) {
                const int32_t expert = in.ids[pos];
                const int src = (int) out.owner[pos] - 1;
                const int32_t src_count = counts[(size_t) src * expert_stride + (size_t) expert];
                const int64_t remove_cost = src_count > 1
                    ? repeat_cost[(size_t) src]
                    : cost[(size_t) src];
                const uint64_t mask = in.holders[expert];
                for (int dst = 0; dst < in.n_endpoints; ++dst) {
                    if (dst == src || !(mask & (2ull << dst))) {
                        continue;
                    }
                    const int32_t dst_count = counts[(size_t) dst * expert_stride + (size_t) expert];
                    const int64_t add_cost = dst_count > 0
                        ? repeat_cost[(size_t) dst]
                        : cost[(size_t) dst];
                    if (best_load[(size_t) dst] > std::numeric_limits<int64_t>::max() - add_cost) {
                        continue;
                    }
                    candidate = refined_load;
                    candidate[(size_t) src] -= remove_cost;
                    candidate[(size_t) dst] += add_cost;
                    if (load_is_better(candidate, best_load)) {
                        best_pos = pos;
                        best_endpoint = dst;
                        best_load = candidate;
                    }
                }
            }
            if (best_endpoint < 0) {
                break;
            }
            const int32_t expert = in.ids[best_pos];
            const int src = (int) out.owner[best_pos] - 1;
            --counts[(size_t) src * expert_stride + (size_t) expert];
            ++counts[(size_t) best_endpoint * expert_stride + (size_t) expert];
            out.owner[best_pos] = (uint8_t) (1 + best_endpoint);
            refined_load = best_load;
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

// Convenience API for callers that do not run in a hot loop.
static inline bool llama_ep_dealer_plan_build(
        const llama_ep_dealer_input & in,
        llama_ep_dealer_plan        & out) {
    llama_ep_dealer_workspace workspace;
    return llama_ep_dealer_plan_build(in, out, workspace);
}
