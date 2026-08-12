#pragma once

#include <cstddef>
#include <limits>

struct llama_ep_credit_amount {
    size_t bytes = 0;
    size_t requests = 0;
};

inline bool llama_ep_credit_estimate(
        size_t n_tokens,
        size_t n_assignments,
        size_t n_embd,
        size_t n_requests,
        llama_ep_credit_amount & amount) {
    amount = {};
    if (n_requests == 0) {
        return n_assignments == 0;
    }
    if (n_tokens == 0 || n_assignments == 0 || n_embd == 0 ||
        n_embd > std::numeric_limits<size_t>::max() / sizeof(float)) {
        return false;
    }
    const size_t row_bytes = n_embd * sizeof(float);
    if (n_tokens > std::numeric_limits<size_t>::max() / row_bytes ||
        n_assignments > std::numeric_limits<size_t>::max() / row_bytes ||
        n_assignments > std::numeric_limits<size_t>::max() / 32) {
        return false;
    }
    const size_t hidden = n_tokens * row_bytes;
    const size_t response = n_assignments * row_bytes;
    const size_t metadata = n_assignments * 32;
    if (hidden > std::numeric_limits<size_t>::max() - response ||
        metadata > std::numeric_limits<size_t>::max() - hidden - response) {
        return false;
    }
    amount.bytes = hidden + response + metadata;
    amount.requests = n_requests;
    return true;
}

// Thread-compatible credit accounting. The owner supplies synchronization so
// a condition variable can use can_reserve() as its wait predicate.
class llama_ep_credit_pool {
public:
    explicit llama_ep_credit_pool(llama_ep_credit_amount limit = {}) : limit_(limit) {
    }

    void set_limit(llama_ep_credit_amount limit) {
        limit_ = limit;
    }

    bool fits(const llama_ep_credit_amount & amount) const {
        return amount.bytes <= limit_.bytes && amount.requests <= limit_.requests;
    }

    bool can_reserve(const llama_ep_credit_amount & amount) const {
        return fits(amount) && amount.bytes <= limit_.bytes - current_.bytes &&
               amount.requests <= limit_.requests - current_.requests;
    }

    bool try_reserve(const llama_ep_credit_amount & amount) {
        if (!can_reserve(amount)) {
            return false;
        }
        current_.bytes += amount.bytes;
        current_.requests += amount.requests;
        if (current_.bytes > high_water_.bytes) {
            high_water_.bytes = current_.bytes;
        }
        if (current_.requests > high_water_.requests) {
            high_water_.requests = current_.requests;
        }
        return true;
    }

    bool release(const llama_ep_credit_amount & amount) {
        if (amount.bytes > current_.bytes || amount.requests > current_.requests) {
            return false;
        }
        current_.bytes -= amount.bytes;
        current_.requests -= amount.requests;
        return true;
    }

    const llama_ep_credit_amount & limit() const {
        return limit_;
    }

    const llama_ep_credit_amount & current() const {
        return current_;
    }

    const llama_ep_credit_amount & high_water() const {
        return high_water_;
    }

private:
    llama_ep_credit_amount limit_;
    llama_ep_credit_amount current_;
    llama_ep_credit_amount high_water_;
};
