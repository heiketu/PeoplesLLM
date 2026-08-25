#pragma once

#include <cstddef>
#include <cstdint>

namespace xllama {

bool moe_hot_trace_enabled();

void moe_hot_trace_record(
        int          layer,
        const void * ids,
        int64_t      n_tokens,
        int          n_ids,
        size_t       token_stride,
        size_t       id_stride,
        int          n_experts);

bool moe_hot_trace_flush();

} // namespace xllama
