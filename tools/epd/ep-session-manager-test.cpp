#include "llama-ep-session-manager.h"

#include <atomic>
#include <condition_variable>
#include <cstdio>
#include <mutex>

namespace {

struct fake_context {
    std::mutex mutex;
    std::condition_variable cv;
    bool stopped = false;
    std::atomic<int> closes{0};
};

bool fake_send(void *, const void *, size_t) {
    return true;
}

bool fake_recv(void *, void *, size_t) {
    return false;
}

void fake_shutdown(void * opaque) {
    fake_context & ctx = *static_cast<fake_context *>(opaque);
    {
        std::lock_guard<std::mutex> lock(ctx.mutex);
        ctx.stopped = true;
    }
    ctx.cv.notify_all();
}

void fake_close(void * opaque) {
    static_cast<fake_context *>(opaque)->closes.fetch_add(1, std::memory_order_relaxed);
}

llama_ep_transport make_transport(fake_context & ctx) {
    return {
        &ctx,
        {fake_send, fake_recv, fake_shutdown, fake_close, nullptr},
    };
}

void wait_for_shutdown(llama_ep_transport * transport) {
    fake_context & ctx = *static_cast<fake_context *>(transport->ctx);
    std::unique_lock<std::mutex> lock(ctx.mutex);
    ctx.cv.wait(lock, [&ctx]() { return ctx.stopped; });
}

} // namespace

int main() {
    fake_context first;
    fake_context rejected;
    {
        llama_ep_session_manager manager(1, wait_for_shutdown);
        llama_ep_transport first_transport = make_transport(first);
        if (!manager.start(first_transport) || first_transport.ctx != nullptr) {
            fprintf(stderr, "FAIL: first session was not transferred to manager\n");
            return 1;
        }
        llama_ep_transport second_transport = make_transport(rejected);
        if (manager.start(second_transport) || second_transport.ctx == nullptr) {
            fprintf(stderr, "FAIL: session limit did not preserve rejected transport ownership\n");
            return 1;
        }
        second_transport.ops.close(second_transport.ctx);
        manager.shutdown();
        if (manager.active() != 0 || manager.high_water() != 1) {
            fprintf(stderr, "FAIL: shutdown did not drain the session registry\n");
            return 1;
        }
    }
    if (first.closes.load(std::memory_order_relaxed) != 1 ||
        rejected.closes.load(std::memory_order_relaxed) != 1) {
        fprintf(stderr, "FAIL: transports were not closed exactly once\n");
        return 1;
    }
    printf("EP session manager tests passed\n");
    return 0;
}
