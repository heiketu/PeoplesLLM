#pragma once

#include <cstddef>
#include <cstdint>

// Token-budget policy for continuous batching. The executor remains responsible
// for KV-cache admission and physical ubatching; this class only decides how
// much logical prefill work may enter the next server batch.
struct server_prefill_round {
    size_t  start_slot        = 0;
    int32_t token_budget      = 0;
    int32_t decode_tokens     = 0;
    int32_t request_limit     = 0;
    int32_t pending_prefills  = 0;

    int32_t remaining(int32_t batch_tokens) const;
    int32_t grant(int32_t requested, bool can_split, int32_t batch_tokens) const;

    // An executor that processes work outside the normal llama_batch may run
    // only when it is the sole request. Under the adaptive policy it performs
    // its own physical microbatching and may therefore exceed token_budget;
    // an explicit fixed chunk limit remains authoritative.
    bool allows_exclusive_executor(int32_t requested, int32_t batch_tokens) const;
};

class server_prefill_scheduler {
public:
    // Zero selects the adaptive policy. A positive value is a hard per-request
    // chunk cap for splittable prompts.
    void set_chunk_size(int32_t chunk_size);
    int32_t chunk_size() const;

    server_prefill_round begin_round(
            int32_t token_budget,
            int32_t physical_batch,
            int32_t decode_tokens,
            int32_t pending_prefills,
            size_t  n_slots) const;

    size_t slot_at(const server_prefill_round & round, size_t offset, size_t n_slots) const;
    void on_scheduled(size_t slot, size_t n_slots, int32_t n_tokens);

private:
    int32_t chunk_size_ = 0;
    size_t  next_slot_  = 0;
};
