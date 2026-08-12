#include "server-prefill-scheduler.h"

#include <cstdio>

#define CHECK(condition) do { \
    if (!(condition)) { \
        std::fprintf(stderr, "check failed at line %d: %s\n", __LINE__, #condition); \
        return 1; \
    } \
} while (0)

int main() {
    server_prefill_scheduler scheduler;

    {
        const auto round = scheduler.begin_round(2048, 512, 0, 1, 4);
        CHECK(round.grant(4096, true, 0) == 2048);
        CHECK(round.allows_exclusive_executor(2048, 0));
        CHECK(round.allows_exclusive_executor(16384, 0));
        CHECK(!round.allows_exclusive_executor(16384, 1));
    }

    {
        const auto round = scheduler.begin_round(2048, 512, 4, 1, 4);
        CHECK(round.grant(4096, true, 4) == 512);
        CHECK(!round.allows_exclusive_executor(512, 4));
    }

    {
        const auto round = scheduler.begin_round(2048, 512, 0, 2, 4);
        CHECK(round.grant(4096, true, 0) == 1024);
        CHECK(round.grant(4096, true, 1024) == 1024);
    }

    {
        scheduler.set_chunk_size(256);
        const auto round = scheduler.begin_round(2048, 512, 0, 1, 4);
        CHECK(round.grant(4096, true, 0) == 256);
        CHECK(!round.allows_exclusive_executor(4096, 0));
        CHECK(round.grant(300, false, 1800) == 0);
        CHECK(round.grant(200, false, 1800) == 200);
    }

    {
        scheduler.set_chunk_size(-1);
        CHECK(scheduler.chunk_size() == 0);
        scheduler.on_scheduled(2, 4, 128);
        const auto round = scheduler.begin_round(2048, 512, 0, 1, 4);
        CHECK(round.start_slot == 3);
        CHECK(scheduler.slot_at(round, 0, 4) == 3);
        CHECK(scheduler.slot_at(round, 1, 4) == 0);
        scheduler.on_scheduled(1, 4, 0);
        CHECK(scheduler.begin_round(2048, 512, 0, 1, 4).start_slot == 3);
    }

    {
        const auto round = scheduler.begin_round(-1, -1, -1, -1, 0);
        CHECK(round.grant(1, true, 0) == 0);
        CHECK(scheduler.slot_at(round, 0, 0) == 0);
    }

    std::puts("server prefill scheduler tests passed");
    return 0;
}
