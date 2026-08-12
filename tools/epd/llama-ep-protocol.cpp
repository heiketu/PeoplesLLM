#include "llama-ep-protocol.h"

#include <cstring>
#include <limits>

namespace {

bool checked_add(size_t a, size_t b, size_t & out) {
    if (b > std::numeric_limits<size_t>::max() - a) {
        return false;
    }
    out = a + b;
    return true;
}

bool checked_mul(size_t a, size_t b, size_t & out) {
    if (a != 0 && b > std::numeric_limits<size_t>::max() / a) {
        return false;
    }
    out = a * b;
    return true;
}

bool parse_ragged_arrays(
        const uint8_t * payload,
        size_t payload_len,
        size_t header_len,
        llama_ep_ragged_request_view & view,
        std::string & err) {
    if (view.n_tokens < 1 || view.n_tokens > 65536) {
        err = "n_tokens is outside the supported range";
        return false;
    }
    if (view.n_sel < 0 || view.n_sel > (int64_t) 1 << 22) {
        err = "n_sel is outside the supported range";
        return false;
    }
    if (view.n_embd < 1) {
        err = "n_embd must be positive";
        return false;
    }

    size_t assignment_bytes = 0;
    size_t hidden_values = 0;
    size_t hidden_bytes = 0;
    size_t expected = header_len;
    if (!checked_mul((size_t) view.n_sel, 3 * sizeof(int32_t) + sizeof(float), assignment_bytes) ||
        !checked_mul((size_t) view.n_tokens, (size_t) view.n_embd, hidden_values) ||
        !checked_mul(hidden_values, sizeof(float), hidden_bytes) ||
        !checked_add(expected, assignment_bytes, expected) ||
        !checked_add(expected, hidden_bytes, expected)) {
        err = "request payload size overflows size_t";
        return false;
    }
    if (expected > LLAMA_EP_MAX_FRAME_BYTES) {
        err = "request payload exceeds the frame limit";
        return false;
    }
    if (payload_len != expected) {
        err = "request payload length mismatch";
        return false;
    }

    const uint8_t * cursor = payload + header_len;
    view.token_idx = reinterpret_cast<const int32_t *>(cursor);
    cursor += (size_t) view.n_sel * sizeof(int32_t);
    view.slot_idx = reinterpret_cast<const int32_t *>(cursor);
    cursor += (size_t) view.n_sel * sizeof(int32_t);
    view.expert_id = reinterpret_cast<const int32_t *>(cursor);
    cursor += (size_t) view.n_sel * sizeof(int32_t);
    view.weights = reinterpret_cast<const float *>(cursor);
    cursor += (size_t) view.n_sel * sizeof(float);
    view.hidden = reinterpret_cast<const float *>(cursor);
    return true;
}

} // namespace

bool llama_ep_parse_req2(
        const uint8_t * payload,
        size_t payload_len,
        llama_ep_ragged_request_view & view,
        std::string & err) {
    view = {};
    err.clear();
    if (payload == nullptr || payload_len < sizeof(llama_ep_req2_header)) {
        err = "short REQ2";
        return false;
    }

    llama_ep_req2_header header;
    memcpy(&header, payload, sizeof(header));
    view.layer = header.layer;
    view.n_tokens = header.n_tokens;
    view.n_sel = header.n_sel;
    view.n_embd = header.n_embd;
    return parse_ragged_arrays(payload, payload_len, sizeof(header), view, err);
}

bool llama_ep_parse_req3(
        const uint8_t * payload,
        size_t payload_len,
        llama_ep_ragged_request_view & view,
        std::string & err) {
    view = {};
    err.clear();
    if (payload == nullptr || payload_len < sizeof(llama_ep_req3_header)) {
        err = "short REQ3";
        return false;
    }

    llama_ep_req3_header header;
    memcpy(&header, payload, sizeof(header));
    view.layer = header.layer;
    view.n_tokens = header.n_tokens;
    view.n_sel = header.n_sel;
    view.n_embd = header.n_embd;
    view.req_id = header.req_id;
    return parse_ragged_arrays(payload, payload_len, sizeof(header), view, err);
}

bool llama_ep_parse_resp3(
        const uint8_t * payload,
        size_t payload_len,
        llama_ep_resp3_view & view,
        std::string & err) {
    view = {};
    err.clear();
    if (payload == nullptr || payload_len < sizeof(llama_ep_resp3_header)) {
        err = "short RESP3";
        return false;
    }

    llama_ep_resp3_header header;
    memcpy(&header, payload, sizeof(header));
    if (header.n_sel < 0 || header.n_sel > (int64_t) 1 << 22 || header.n_embd < 1) {
        err = "RESP3 shape is outside the supported range";
        return false;
    }

    size_t values = 0;
    size_t bytes = 0;
    size_t expected = sizeof(header);
    if (!checked_mul((size_t) header.n_sel, (size_t) header.n_embd, values) ||
        !checked_mul(values, sizeof(float), bytes) ||
        !checked_add(expected, bytes, expected)) {
        err = "RESP3 payload size overflows size_t";
        return false;
    }
    if (expected > LLAMA_EP_MAX_FRAME_BYTES) {
        err = "RESP3 payload exceeds the frame limit";
        return false;
    }
    if (payload_len != expected) {
        err = "RESP3 payload length mismatch";
        return false;
    }

    view.req_id = header.req_id;
    view.n_sel = header.n_sel;
    view.n_embd = header.n_embd;
    view.out = reinterpret_cast<const float *>(payload + sizeof(header));
    return true;
}
