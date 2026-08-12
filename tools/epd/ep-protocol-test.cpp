#include "llama-ep-protocol.h"

#include <cstdio>
#include <cstring>
#include <vector>

namespace {

int failures = 0;

void expect(bool condition, const char * message) {
    if (!condition) {
        fprintf(stderr, "FAIL: %s\n", message);
        ++failures;
    }
}

template <typename Header>
std::vector<uint8_t> make_payload(const Header & header, int32_t n_tokens, int32_t n_sel, int32_t n_embd) {
    const size_t size = sizeof(header) + (size_t) n_sel * (3 * sizeof(int32_t) + sizeof(float))
                      + (size_t) n_tokens * n_embd * sizeof(float);
    std::vector<uint8_t> payload(size, 0);
    memcpy(payload.data(), &header, sizeof(header));
    return payload;
}

void test_req2_valid() {
    const llama_ep_req2_header header = {7, 3, 2, 16};
    std::vector<uint8_t> payload = make_payload(header, header.n_tokens, header.n_sel, header.n_embd);
    llama_ep_ragged_request_view view;
    std::string err;
    expect(llama_ep_parse_req2(payload.data(), payload.size(), view, err), "valid REQ2 parses");
    expect(view.layer == 7 && view.n_tokens == 3 && view.n_sel == 2 && view.n_embd == 16,
           "REQ2 header fields are preserved");
    expect(view.token_idx != nullptr && view.hidden != nullptr, "REQ2 array views are populated");
}

void test_req3_valid() {
    const llama_ep_req3_header header = {9, 1, 0, 32, 0x123456789abcdef0ULL};
    std::vector<uint8_t> payload = make_payload(header, header.n_tokens, header.n_sel, header.n_embd);
    llama_ep_ragged_request_view view;
    std::string err;
    expect(llama_ep_parse_req3(payload.data(), payload.size(), view, err), "valid zero-selection REQ3 parses");
    expect(view.req_id == header.req_id, "REQ3 request id is preserved");
}

void test_resp3_validation() {
    const llama_ep_resp3_header header = {42, 3, 16};
    std::vector<uint8_t> payload(sizeof(header) + (size_t) header.n_sel * header.n_embd * sizeof(float), 0);
    memcpy(payload.data(), &header, sizeof(header));
    llama_ep_resp3_view view;
    std::string err;
    expect(llama_ep_parse_resp3(payload.data(), payload.size(), view, err), "valid RESP3 parses");
    expect(view.req_id == 42 && view.n_sel == 3 && view.n_embd == 16, "RESP3 fields are preserved");

    llama_ep_resp3_header invalid = {42, -1, 16};
    payload.resize(sizeof(invalid));
    memcpy(payload.data(), &invalid, sizeof(invalid));
    expect(!llama_ep_parse_resp3(payload.data(), payload.size(), view, err), "negative RESP3 n_sel is rejected");

    invalid = {42, 65536, 65536};
    memcpy(payload.data(), &invalid, sizeof(invalid));
    expect(!llama_ep_parse_resp3(payload.data(), payload.size(), view, err), "oversized RESP3 is rejected");
}

void test_invalid_shapes() {
    llama_ep_ragged_request_view view;
    std::string err;

    llama_ep_req2_header header = {0, 1, -1, 16};
    std::vector<uint8_t> payload(sizeof(header), 0);
    memcpy(payload.data(), &header, sizeof(header));
    expect(!llama_ep_parse_req2(payload.data(), payload.size(), view, err), "negative n_sel is rejected");

    header = {0, 0, 0, 16};
    memcpy(payload.data(), &header, sizeof(header));
    expect(!llama_ep_parse_req2(payload.data(), payload.size(), view, err), "zero n_tokens is rejected");

    header = {0, 1, 0, -16};
    memcpy(payload.data(), &header, sizeof(header));
    expect(!llama_ep_parse_req2(payload.data(), payload.size(), view, err), "negative n_embd is rejected");

    header = {0, 65536, 0, 65536};
    memcpy(payload.data(), &header, sizeof(header));
    expect(!llama_ep_parse_req2(payload.data(), payload.size(), view, err), "payload above frame limit is rejected");
}

void test_invalid_lengths() {
    const llama_ep_req2_header header = {0, 2, 1, 8};
    std::vector<uint8_t> payload = make_payload(header, header.n_tokens, header.n_sel, header.n_embd);
    llama_ep_ragged_request_view view;
    std::string err;

    expect(!llama_ep_parse_req2(nullptr, 0, view, err), "null payload is rejected");
    expect(!llama_ep_parse_req2(payload.data(), payload.size() - 1, view, err), "truncated payload is rejected");
    payload.push_back(0);
    expect(!llama_ep_parse_req2(payload.data(), payload.size(), view, err), "trailing bytes are rejected");
}

struct fake_transport_ctx {
    size_t bytes_sent = 0;
    size_t sendv_calls = 0;
};

bool fake_send(void * opaque, const void *, size_t len) {
    static_cast<fake_transport_ctx *>(opaque)->bytes_sent += len;
    return true;
}

bool fake_recv(void *, void *, size_t) {
    return false;
}

bool fake_sendv(void * opaque, const void * const *, const size_t * lens, size_t n_parts) {
    auto * ctx = static_cast<fake_transport_ctx *>(opaque);
    ctx->sendv_calls++;
    for (size_t i = 0; i < n_parts; ++i) {
        ctx->bytes_sent += lens[i];
    }
    return true;
}

void fake_shutdown(void *) {
}

void fake_close(void *) {
}

void test_frame_send_limits() {
    fake_transport_ctx ctx;
    llama_ep_transport transport = {
        &ctx,
        {fake_send, fake_recv, fake_shutdown, fake_close, nullptr},
    };
    const uint8_t byte = 0;
    const void * parts[1] = {&byte};
    const size_t too_large[1] = {(size_t) LLAMA_EP_MAX_FRAME_BYTES + 1};
    expect(!llama_ep_send_framev(&transport, LLAMA_EP_MSG_REQ2, parts, too_large, 1),
           "oversized gathered send is rejected");
    expect(ctx.bytes_sent == 0, "rejected send writes no frame header");
    expect(!llama_ep_send_frame(&transport, LLAMA_EP_MSG_REQ2, nullptr, 1),
           "non-empty null payload is rejected");

    const uint8_t first[3] = {1, 2, 3};
    const uint8_t second[2] = {4, 5};
    const void * valid_parts[2] = {first, second};
    const size_t valid_lens[2] = {sizeof(first), sizeof(second)};
    transport.ops.sendv_all = fake_sendv;
    expect(llama_ep_send_framev(&transport, LLAMA_EP_MSG_REQ4, valid_parts, valid_lens, 2),
           "gather-capable transport accepts a valid frame");
    expect(ctx.sendv_calls == 1, "one logical frame uses one gathered transport call");
    expect(ctx.bytes_sent == sizeof(llama_ep_frame_header) + sizeof(first) + sizeof(second),
           "gathered transport sees the complete frame byte stream");
}

} // namespace

int main() {
    test_req2_valid();
    test_req3_valid();
    test_resp3_validation();
    test_invalid_shapes();
    test_invalid_lengths();
    test_frame_send_limits();
    if (failures != 0) {
        fprintf(stderr, "%d protocol test(s) failed\n", failures);
        return 1;
    }
    printf("EP protocol tests passed\n");
    return 0;
}
