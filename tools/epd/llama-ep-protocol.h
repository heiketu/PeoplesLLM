#pragma once

#include "llama-ep-transport.h"

#include <cstddef>
#include <cstdint>
#include <string>

// A validated, non-owning view of a REQ2 or REQ3 payload. The view remains
// valid only while the payload buffer passed to the parser remains alive.
struct llama_ep_ragged_request_view {
    int32_t layer    = 0;
    int32_t n_tokens = 0;
    int32_t n_sel    = 0;
    int32_t n_embd   = 0;
    uint64_t req_id  = 0;

    const int32_t * token_idx = nullptr;
    const int32_t * slot_idx  = nullptr;
    const int32_t * expert_id = nullptr;
    const float   * weights   = nullptr;
    const float   * hidden    = nullptr;
};

struct llama_ep_resp3_view {
    uint64_t req_id = 0;
    int32_t  n_sel  = 0;
    int32_t  n_embd = 0;
    const float * out = nullptr;
};

bool llama_ep_parse_req2(
        const uint8_t * payload,
        size_t payload_len,
        llama_ep_ragged_request_view & view,
        std::string & err);

bool llama_ep_parse_req3(
        const uint8_t * payload,
        size_t payload_len,
        llama_ep_ragged_request_view & view,
        std::string & err);

bool llama_ep_parse_resp3(
        const uint8_t * payload,
        size_t payload_len,
        llama_ep_resp3_view & view,
        std::string & err);
