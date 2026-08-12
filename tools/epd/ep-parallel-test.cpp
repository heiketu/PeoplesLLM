#include "llama-ep-parallel.h"

#include <atomic>
#include <cstdio>
#include <stdexcept>
#include <vector>

int main() {
    llama_ep_parallel_for team(8);
    if (team.max_tasks() != 8) {
        fprintf(stderr, "FAIL: max task count\n");
        return 1;
    }

    for (int round = 0; round < 100; ++round) {
        const int n_tasks = round % 8 + 1;
        std::vector<int> seen((size_t) n_tasks, 0);
        team.run(n_tasks, [&](int task, int actual_tasks) {
            if (actual_tasks != n_tasks || task < 0 || task >= n_tasks) {
                throw std::runtime_error("invalid task index");
            }
            seen[(size_t) task]++;
        });
        for (int count : seen) {
            if (count != 1) {
                fprintf(stderr, "FAIL: task executed %d times\n", count);
                return 1;
            }
        }
    }

    std::atomic<int> completed{0};
    bool caught = false;
    try {
        team.run(4, [&](int task, int) {
            if (task == 2) {
                throw std::runtime_error("expected");
            }
            completed.fetch_add(1, std::memory_order_relaxed);
        });
    } catch (const std::runtime_error &) {
        caught = true;
    }
    if (!caught || completed.load(std::memory_order_relaxed) != 3) {
        fprintf(stderr, "FAIL: exception propagation\n");
        return 1;
    }

    team.run(2, [&](int, int) { completed.fetch_add(1, std::memory_order_relaxed); });
    if (completed.load(std::memory_order_relaxed) != 5) {
        fprintf(stderr, "FAIL: reuse after exception\n");
        return 1;
    }

    printf("EP parallel team tests passed\n");
    return 0;
}
