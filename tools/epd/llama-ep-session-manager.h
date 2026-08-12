#pragma once

#include "llama-ep-transport.h"

#include <cstddef>
#include <functional>
#include <memory>

// Owns worker connection threads and transports. The caller retains ownership
// when start() returns false; a successful start clears the caller's transport.
class llama_ep_session_manager {
public:
    using handler = std::function<void(llama_ep_transport *)>;

    llama_ep_session_manager(size_t max_sessions, handler session_handler);
    ~llama_ep_session_manager();

    llama_ep_session_manager(const llama_ep_session_manager &) = delete;
    llama_ep_session_manager & operator=(const llama_ep_session_manager &) = delete;

    bool start(llama_ep_transport & transport);
    void reap_finished();
    void shutdown();

    size_t active() const;
    size_t high_water() const;

private:
    struct impl;
    std::unique_ptr<impl> pimpl;
};
