#include "llama-ep-expert-map.h"

#include <cstdio>

static int failures = 0;

#define CHECK(cond, ...) do { \
    if (!(cond)) { \
        ++failures; \
        printf("FAIL %s:%d: ", __func__, __LINE__); \
        printf(__VA_ARGS__); \
        printf("\n"); \
    } \
} while (0)

static void test_range() {
    llama_ep_expert_map m;
    CHECK(m.init_range(16, 4, 8), "range init");
    CHECK(m.contiguous && m.first == 4 && m.last == 8, "range metadata");
    CHECK(m.local_to_global == std::vector<int32_t>({4, 5, 6, 7}), "range planes");
    CHECK(m.local(3) == -1 && m.local(4) == 0 && m.local(7) == 3 && m.local(8) == -1,
          "range lookup");
}

static void test_mod4() {
    llama_ep_expert_map m;
    CHECK(m.init_mod(16, 2, 4), "mod init");
    CHECK(!m.contiguous, "mod map must be sparse");
    CHECK(m.local_to_global == std::vector<int32_t>({2, 6, 10, 14}), "4n+2 planes");
    CHECK(m.local(2) == 0 && m.local(10) == 2 && m.local(11) == -1, "mod lookup");
    const auto bits = m.bitmap();
    CHECK(bits.size() == 2 && bits[0] == 0x44 && bits[1] == 0x44,
          "mod bitmap %02x/%02x", bits[0], bits[1]);
}

static void test_explicit_ids() {
    llama_ep_expert_map m;
    CHECK(m.init_ids(16, {15, 0, 7, 3}), "explicit init");
    CHECK(m.local_to_global == std::vector<int32_t>({0, 3, 7, 15}), "ids sorted");
    CHECK(m.local(7) == 2 && m.local(8) == -1, "explicit lookup");
    CHECK(!m.init_ids(16, {1, 1}), "duplicates rejected");
    CHECK(!m.init_ids(16, {-1, 2}), "negative rejected");
    CHECK(!m.init_ids(16, {16}), "out of range rejected");
}

int main() {
    test_range();
    test_mod4();
    test_explicit_ids();
    if (failures == 0) {
        printf("ep-expert-map-test: PASS (all cases)\n");
        return 0;
    }
    printf("ep-expert-map-test: FAIL (%d checks)\n", failures);
    return 1;
}

