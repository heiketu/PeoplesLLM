#include "server-prefill-scheduler.h"

#include <algorithm>
#include <limits>

int32_t server_prefill_round::remaining(int32_t batch_tokens) const {
    if (token_budget <= 0 || batch_tokens >= token_budget) {
        return 0;
    }

    return token_budget - std::max<int32_t>(0, batch_tokens);
}

int32_t server_prefill_round::grant(int32_t requested, bool can_split, int32_t batch_tokens) const {
    if (requested <= 0) {
        return 0;
    }

    const int32_t available = remaining(batch_tokens);
    if (available <= 0) {
        return 0;
    }

    if (!can_split) {
        return requested <= available ? requested : 0;
    }

    return std::min(requested, std::min(available, request_limit));
}

bool server_prefill_round::allows_exclusive_executor(int32_t requested, int32_t batch_tokens) const {
    if (decode_tokens != 0 || pending_prefills != 1 || requested <= 0 || batch_tokens != 0) {
        return false;
    }

    const int32_t available = remaining(0);
    if (available <= 0) {
        return false;
    }

    // An exclusive executor performs its own physical microbatching. Under
    // the adaptive, uncontended policy request_limit == available, so do not
    // confuse the logical prompt length with the ordinary server batch token
    // budget. A fixed chunk limit remains authoritative.
    return requested <= available || request_limit == available;
}

void server_prefill_scheduler::set_chunk_size(int32_t chunk_size) {
    chunk_size_ = std::max<int32_t>(0, chunk_size);
}

int32_t server_prefill_scheduler::chunk_size() const {
    return chunk_size_;
}

server_prefill_round server_prefill_scheduler::begin_round(
        int32_t token_budget,
        int32_t physical_batch,
        int32_t decode_tokens,
        int32_t pending_prefills,
        size_t  n_slots) const {
    server_prefill_round result;
    result.start_slot       = n_slots > 0 ? next_slot_ % n_slots : 0;
    result.token_budget     = std::max<int32_t>(0, token_budget);
    result.decode_tokens    = std::max<int32_t>(0, decode_tokens);
    result.pending_prefills = std::max<int32_t>(0, pending_prefills);

    const int32_t available = result.remaining(result.decode_tokens);
    if (available <= 0) {
        return result;
    }

    if (chunk_size_ > 0) {
        result.request_limit = std::min(chunk_size_, available);
        return result;
    }

    // Preserve maximum PP throughput for an uncontended prompt. If decode work
    // is active, cap the prefill tail to one physical batch to bound ITL. With
    // multiple prefills, divide the logical budget fairly while keeping chunks
    // large enough for an efficient physical batch.
    if (result.decode_tokens > 0) {
        result.request_limit = std::min(available, std::max<int32_t>(1, physical_batch));
    } else if (result.pending_prefills > 1) {
        const int32_t fair_share = (available + result.pending_prefills - 1) / result.pending_prefills;
        result.request_limit = std::min(available, std::max(std::max<int32_t>(1, physical_batch), fair_share));
    } else {
        result.request_limit = available;
    }

    return result;
}

size_t server_prefill_scheduler::slot_at(
        const server_prefill_round & round,
        size_t offset,
        size_t n_slots) const {
    if (n_slots == 0) {
        return 0;
    }

    return (round.start_slot + offset) % n_slots;
}

void server_prefill_scheduler::on_scheduled(size_t slot, size_t n_slots, int32_t n_tokens) {
    if (n_slots > 0 && n_tokens > 0) {
        next_slot_ = (slot + 1) % n_slots;
    }
}
