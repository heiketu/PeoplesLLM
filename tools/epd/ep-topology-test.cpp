#include "llama-ep-capability.h"
#include "llama-ep-topology.h"

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#define CHECK(expr) do { \
    if (!(expr)) { \
        std::fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #expr); \
        return 1; \
    } \
} while (0)

int main() {
    const llama_ep_expert_ownership contiguous{4, 12, nullptr, 0};
    CHECK(contiguous.valid());
    CHECK(contiguous.count() == 8);
    CHECK(!contiguous.holds(3));
    CHECK(contiguous.holds(4));
    CHECK(contiguous.holds(11));
    CHECK(!contiguous.holds(12));

    const uint8_t sparse_bits[] = {0b01001001, 0b00000010};
    const llama_ep_expert_ownership sparse{10, 20, sparse_bits, sizeof(sparse_bits)};
    CHECK(sparse.valid());
    CHECK(sparse.count() == 4);
    CHECK(sparse.holds(10));
    CHECK(sparse.holds(13));
    CHECK(sparse.holds(16));
    CHECK(sparse.holds(19));
    CHECK(!sparse.holds(18));

    const llama_ep_expert_ownership short_bitmap{10, 20, sparse_bits, 1};
    CHECK(!short_bitmap.valid());
    const uint8_t bad_padding[] = {0, 0b10000000};
    const llama_ep_expert_ownership padded{10, 20, bad_padding, sizeof(bad_padding)};
    CHECK(!padded.valid());

    const int32_t first[] = {0, 4, 8, 12};
    const int32_t last[]  = {4, 8, 12, 16};
    CHECK(llama_ep_exact_shard_cover(16, 4, first, last));
    CHECK(llama_ep_full_shard_cover(16, 4, first, last));

    const uint64_t exact[] = {1, 2, 4, 8};
    const uint64_t overlap[] = {1, 3, 4, 8};
    CHECK(llama_ep_holder_cover(4, 4, exact, true));
    CHECK(!llama_ep_holder_cover(4, 4, overlap, true));
    CHECK(llama_ep_holder_cover(4, 4, overlap, false));

    llama_ep_cap_worker cap = {};
    cap.proto_ver = LLAMA_EP_PROTO_VER;
    cap.caps = LLAMA_EP_CAP_REQ2 | LLAMA_EP_CAP_EXPERT_BITMAP;
    cap.expert_first = 10;
    cap.expert_last = 20;
    std::vector<uint8_t> payload(sizeof(cap) + sizeof(sparse_bits));
    std::memcpy(payload.data(), &cap, sizeof(cap));
    std::memcpy(payload.data() + sizeof(cap), sparse_bits, sizeof(sparse_bits));
    llama_ep_worker_capability parsed;
    std::string error;
    CHECK(llama_ep_parse_worker_capability(payload.data(), payload.size(), parsed, error));
    CHECK(parsed.ownership().count() == 4);
    CHECK(parsed.ownership().holds(19));

    CHECK(!llama_ep_parse_worker_capability(payload.data(), payload.size() - 1, parsed, error));
    payload.back() = 0b10000010;
    CHECK(!llama_ep_parse_worker_capability(payload.data(), payload.size(), parsed, error));

    cap.caps = LLAMA_EP_CAP_REQ2;
    std::memcpy(payload.data(), &cap, sizeof(cap));
    CHECK(!llama_ep_parse_worker_capability(payload.data(), payload.size(), parsed, error));
    CHECK(llama_ep_parse_worker_capability(payload.data(), sizeof(cap), parsed, error));

    cap.kernel_id = 0x1234abcd;
    cap.caps = LLAMA_EP_CAP_REQ2 | LLAMA_EP_CAP_PRECISION_CONTRACT;
    const llama_ep_precision_contract precision =
        llama_ep_make_cpu_repack_precision_contract(cap.kernel_id, UINT64_C(0x11223344), UINT64_C(0x55667788));
    std::vector<uint8_t> precision_payload(sizeof(cap) + sizeof(precision));
    std::memcpy(precision_payload.data(), &cap, sizeof(cap));
    std::memcpy(precision_payload.data() + sizeof(cap), &precision, sizeof(precision));
    CHECK(llama_ep_parse_worker_capability(
        precision_payload.data(), precision_payload.size(), parsed, error));
    CHECK(parsed.has_precision());
    CHECK(llama_ep_precision_contract_equal(parsed.precision, precision));
    CHECK(!llama_ep_parse_worker_capability(
        precision_payload.data(), precision_payload.size() - 1, parsed, error));

    llama_ep_precision_contract bad_precision = precision;
    bad_precision.contract_id ^= 1;
    std::memcpy(precision_payload.data() + sizeof(cap), &bad_precision, sizeof(bad_precision));
    CHECK(!llama_ep_parse_worker_capability(
        precision_payload.data(), precision_payload.size(), parsed, error));

    cap.caps |= LLAMA_EP_CAP_EXPERT_BITMAP;
    std::vector<uint8_t> full_payload(sizeof(cap) + sizeof(precision) + sizeof(sparse_bits));
    std::memcpy(full_payload.data(), &cap, sizeof(cap));
    std::memcpy(full_payload.data() + sizeof(cap), &precision, sizeof(precision));
    std::memcpy(full_payload.data() + sizeof(cap) + sizeof(precision), sparse_bits, sizeof(sparse_bits));
    CHECK(llama_ep_parse_worker_capability(full_payload.data(), full_payload.size(), parsed, error));
    CHECK(parsed.has_precision());
    CHECK(parsed.ownership().count() == 4);

    std::puts("EP topology tests passed");
    return 0;
}
