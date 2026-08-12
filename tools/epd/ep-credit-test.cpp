#include "llama-ep-credit.h"

#include <cstdio>
#include <limits>

int main() {
    llama_ep_credit_amount first;
    if (!llama_ep_credit_estimate(128, 1024, 4096, 4, first) ||
        first.bytes != (size_t) 128 * 4096 * 4 + (size_t) 1024 * 4096 * 4 + (size_t) 1024 * 32 ||
        first.requests != 4) {
        fprintf(stderr, "FAIL: valid credit estimate\n");
        return 1;
    }

    llama_ep_credit_amount invalid;
    if (llama_ep_credit_estimate(1, 1, std::numeric_limits<size_t>::max(), 1, invalid)) {
        fprintf(stderr, "FAIL: overflowing credit estimate\n");
        return 1;
    }
    if (!llama_ep_credit_estimate(0, 0, 0, 0, invalid) || invalid.bytes != 0) {
        fprintf(stderr, "FAIL: empty credit estimate\n");
        return 1;
    }

    llama_ep_credit_pool pool({first.bytes * 2, 8});
    if (!pool.try_reserve(first) || !pool.try_reserve(first) || pool.try_reserve(first)) {
        fprintf(stderr, "FAIL: credit admission limit\n");
        return 1;
    }
    if (pool.high_water().bytes != first.bytes * 2 || pool.high_water().requests != 8) {
        fprintf(stderr, "FAIL: credit high water\n");
        return 1;
    }
    if (!pool.release(first) || !pool.can_reserve(first) || !pool.release(first) ||
        pool.current().bytes != 0 || pool.current().requests != 0 || pool.release(first)) {
        fprintf(stderr, "FAIL: credit release\n");
        return 1;
    }

    printf("EP credit tests passed\n");
    return 0;
}
