#include "llama-ep-autotune.h"

#include <cstdio>

static int failures = 0;

static void expect_rows(int top_k, int owned, int full, int override_rows, int expected) {
    const int actual = llama_ep_autotune_compact_rows(top_k, owned, full, override_rows);
    if (actual != expected) {
        fprintf(stderr, "rows(top_k=%d owned=%d full=%d override=%d): got %d, want %d\n",
                top_k, owned, full, override_rows, actual, expected);
        ++failures;
    }
}

int main() {
    expect_rows(8,  64, 256, 0, 2); // strict four-way EP
    expect_rows(8,  80, 256, 0, 3); // hotspot replicas
    expect_rows(8, 256, 256, 0, 8); // one full worker
    expect_rows(8,   1, 256, 0, 1);
    expect_rows(8,  64, 256, 4, 4); // measured override
    expect_rows(8,   2, 256, 8, 2); // cannot exceed owned experts
    expect_rows(0,  64, 256, 0, 0);
    expect_rows(8, 257, 256, 0, 0);
    expect_rows(8,  64, 256, -1, 0);

    const int candidates[4] = {16, 24, 32, 36};
    const double clear_best[4] = {0.42, 0.31, 0.20, 0.22};
    const double near_best [4] = {0.42, 0.31, 0.205, 0.20};
    if (llama_ep_autotune_select(candidates, clear_best, 4, 0.03) != 32) {
        fprintf(stderr, "autotune should select the global best team\n");
        ++failures;
    }
    if (llama_ep_autotune_select(candidates, near_best, 4, 0.03) != 32) {
        fprintf(stderr, "autotune should select the smaller team within tolerance\n");
        ++failures;
    }
    if (llama_ep_autotune_select(candidates, near_best, 0, 0.03) != 0 ||
            llama_ep_autotune_select(candidates, near_best, 4, -0.01) != 0) {
        fprintf(stderr, "invalid autotune selection input must fail\n");
        ++failures;
    }

    if (failures == 0) {
        printf("ep-autotune-test: PASS\n");
        return 0;
    }
    printf("ep-autotune-test: FAIL (%d checks)\n", failures);
    return 1;
}
