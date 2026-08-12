#include "llama-ep-service.h"

#include <cstdio>
#include <limits>

static int failures = 0;

#define CHECK(cond, ...) do { \
    if (!(cond)) { \
        ++failures; \
        std::printf("FAIL %s:%d: ", __func__, __LINE__); \
        std::printf(__VA_ARGS__); \
        std::printf("\n"); \
    } \
} while (0)

static void test_ewma() {
    CHECK(llama_ep_service_ewma(0, 80) == 80, "first clean sample should bootstrap directly");
    CHECK(llama_ep_service_ewma(100, 1000) == 138, "upward outlier should be capped at 4x");
    CHECK(llama_ep_service_ewma(100, 1) == 91, "downward outlier should be capped at 1/4x");
    CHECK(llama_ep_service_ewma(100, 108) == 101, "ordinary sample should use 1/8 EWMA");
}

static void test_normalize() {
    const int64_t observed[4] = {80, 0, 120, 0};
    int64_t out[4] = {};
    CHECK(llama_ep_service_normalize_costs(observed, 4, 1000, out), "normalize should succeed");
    CHECK(out[0] == 80 && out[1] == 100 && out[2] == 120 && out[3] == 100,
          "unknown endpoints should receive peer mean, got %lld/%lld/%lld/%lld",
          (long long) out[0], (long long) out[1], (long long) out[2], (long long) out[3]);

    const int64_t unknown[2] = {0, 0};
    CHECK(llama_ep_service_normalize_costs(unknown, 2, 1000, out), "fallback normalize");
    CHECK(out[0] == 1000 && out[1] == 1000, "all-unknown endpoints should use fallback");
}

static void test_work_units() {
    int64_t work = 0;
    CHECK(llama_ep_service_work_units(7, 80, work) && work == 560, "work units");
    CHECK(!llama_ep_service_work_units(-1, 80, work), "negative assignments must fail");
    CHECK(!llama_ep_service_work_units(2, std::numeric_limits<int64_t>::max(), work),
          "overflow must fail");
    CHECK(llama_ep_service_split_work_units(4, 3, 1000, 250, work) && work == 4750,
          "repeat-aware work units");
    CHECK(llama_ep_service_split_work_units(1, 3, 1000, 0, work) && work == 1000,
          "zero-cost repeats are valid");
    CHECK(!llama_ep_service_split_work_units(-1, 0, 1000, 250, work),
          "negative unique count must fail");
    CHECK(!llama_ep_service_split_work_units(2, 1,
                  std::numeric_limits<int64_t>::max(), 1, work),
          "split work multiplication overflow");
}

int main() {
    test_ewma();
    test_normalize();
    test_work_units();
    if (failures == 0) {
        std::printf("ep-service-test: PASS (all cases)\n");
        return 0;
    }
    std::printf("ep-service-test: FAIL (%d checks)\n", failures);
    return 1;
}
