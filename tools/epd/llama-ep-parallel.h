#pragma once

#include <algorithm>
#include <condition_variable>
#include <cstdint>
#include <exception>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

// Small persistent thread team for host-side EP work such as response merging.
// One instance is owned by each logical EP stream, so independent streams do
// not serialize while repeated layers avoid creating and joining OS threads.
class llama_ep_parallel_for {
public:
    explicit llama_ep_parallel_for(int max_tasks) : max_tasks_(std::max(1, max_tasks)) {
        workers_.reserve((size_t) max_tasks_ - 1);
        for (int task = 1; task < max_tasks_; ++task) {
            workers_.emplace_back([this, task]() { worker_loop(task); });
        }
    }

    ~llama_ep_parallel_for() {
        {
            std::lock_guard<std::mutex> lock(job_mtx_);
            stopping_ = true;
            generation_++;
        }
        start_cv_.notify_all();
        for (auto & worker : workers_) {
            worker.join();
        }
    }

    llama_ep_parallel_for(const llama_ep_parallel_for &) = delete;
    llama_ep_parallel_for & operator=(const llama_ep_parallel_for &) = delete;

    int max_tasks() const {
        return max_tasks_;
    }

    void run(int n_tasks, const std::function<void(int, int)> & fn) {
        const int tasks = std::max(1, std::min(n_tasks, max_tasks_));
        if (tasks == 1) {
            fn(0, 1);
            return;
        }

        // A logical stream executes one graph at a time. Keep this guard anyway
        // so misuse cannot replace a job while its workers still reference it.
        std::lock_guard<std::mutex> run_lock(run_mtx_);
        {
            std::lock_guard<std::mutex> lock(job_mtx_);
            job_ = fn;
            job_tasks_ = tasks;
            remaining_ = tasks - 1;
            error_ = nullptr;
            generation_++;
        }
        start_cv_.notify_all();

        try {
            fn(0, tasks);
        } catch (...) {
            std::lock_guard<std::mutex> lock(job_mtx_);
            if (!error_) {
                error_ = std::current_exception();
            }
        }

        std::unique_lock<std::mutex> lock(job_mtx_);
        done_cv_.wait(lock, [this]() { return remaining_ == 0; });
        auto error = error_;
        job_ = {};
        lock.unlock();

        if (error) {
            std::rethrow_exception(error);
        }
    }

private:
    void worker_loop(int task) {
        uint64_t seen_generation = 0;
        for (;;) {
            std::function<void(int, int)> fn;
            int tasks = 0;
            {
                std::unique_lock<std::mutex> lock(job_mtx_);
                start_cv_.wait(lock, [&]() { return stopping_ || generation_ != seen_generation; });
                if (stopping_) {
                    return;
                }
                seen_generation = generation_;
                tasks = job_tasks_;
                if (task >= tasks) {
                    continue;
                }
                fn = job_;
            }

            try {
                fn(task, tasks);
            } catch (...) {
                std::lock_guard<std::mutex> lock(job_mtx_);
                if (!error_) {
                    error_ = std::current_exception();
                }
            }

            {
                std::lock_guard<std::mutex> lock(job_mtx_);
                remaining_--;
                if (remaining_ == 0) {
                    done_cv_.notify_one();
                }
            }
        }
    }

    const int max_tasks_;
    std::vector<std::thread> workers_;

    std::mutex run_mtx_;
    std::mutex job_mtx_;
    std::condition_variable start_cv_;
    std::condition_variable done_cv_;
    std::function<void(int, int)> job_;
    std::exception_ptr error_;
    uint64_t generation_ = 0;
    int job_tasks_ = 0;
    int remaining_ = 0;
    bool stopping_ = false;
};
