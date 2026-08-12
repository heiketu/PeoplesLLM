#include "llama-ep-session-manager.h"

#include <atomic>
#include <thread>
#include <utility>
#include <vector>

struct llama_ep_session_manager::impl {
    struct session {
        llama_ep_transport transport = {};
        std::atomic<bool> finished{false};
        std::thread thread;
    };

    size_t max_sessions = 0;
    size_t high_water = 0;
    handler session_handler;
    std::vector<std::unique_ptr<session>> sessions;

    void close(session & item) {
        if (item.transport.ctx != nullptr && item.transport.ops.close != nullptr) {
            item.transport.ops.close(item.transport.ctx);
        }
        item.transport = {};
    }

    void reap_finished() {
        for (auto it = sessions.begin(); it != sessions.end();) {
            session & item = **it;
            if (!item.finished.load(std::memory_order_acquire)) {
                ++it;
                continue;
            }
            if (item.thread.joinable()) {
                item.thread.join();
            }
            close(item);
            it = sessions.erase(it);
        }
    }

    void shutdown() {
        for (auto & item : sessions) {
            if (item->transport.ctx != nullptr && item->transport.ops.shutdown != nullptr) {
                item->transport.ops.shutdown(item->transport.ctx);
            }
        }
        for (auto & item : sessions) {
            if (item->thread.joinable()) {
                item->thread.join();
            }
            close(*item);
        }
        sessions.clear();
    }
};

llama_ep_session_manager::llama_ep_session_manager(size_t max_sessions, handler session_handler)
    : pimpl(new impl) {
    pimpl->max_sessions = max_sessions > 0 ? max_sessions : 1;
    pimpl->session_handler = std::move(session_handler);
}

llama_ep_session_manager::~llama_ep_session_manager() {
    shutdown();
}

bool llama_ep_session_manager::start(llama_ep_transport & transport) {
    pimpl->reap_finished();
    if (transport.ctx == nullptr || transport.ops.close == nullptr ||
        !pimpl->session_handler || pimpl->sessions.size() >= pimpl->max_sessions) {
        return false;
    }

    std::unique_ptr<impl::session> owned(new impl::session);
    owned->transport = transport;
    impl::session * item = owned.get();
    pimpl->sessions.push_back(std::move(owned));
    try {
        item->thread = std::thread([this, item]() {
            try {
                pimpl->session_handler(&item->transport);
            } catch (...) {
            }
            item->finished.store(true, std::memory_order_release);
        });
    } catch (...) {
        pimpl->sessions.pop_back();
        return false;
    }

    transport = {};
    if (pimpl->sessions.size() > pimpl->high_water) {
        pimpl->high_water = pimpl->sessions.size();
    }
    return true;
}

void llama_ep_session_manager::reap_finished() {
    pimpl->reap_finished();
}

void llama_ep_session_manager::shutdown() {
    if (pimpl) {
        pimpl->shutdown();
    }
}

size_t llama_ep_session_manager::active() const {
    return pimpl->sessions.size();
}

size_t llama_ep_session_manager::high_water() const {
    return pimpl->high_water;
}
