#include "llama-ep-fusion.h"

#include <cstdint>
#include <cstdio>
#include <vector>

int main() {
    constexpr size_t n_expert = 3;
    constexpr size_t gate_plane = 5;
    constexpr size_t up_plane = 7;

    std::vector<uint8_t> gate(n_expert * gate_plane);
    std::vector<uint8_t> up(n_expert * up_plane);
    for (size_t i = 0; i < gate.size(); ++i) gate[i] = (uint8_t) (0x10 + i);
    for (size_t i = 0; i < up.size(); ++i) up[i] = (uint8_t) (0x80 + i);

    std::vector<uint8_t> merged(gate.size() + up.size(), 0);
    bool ok = llama_ep_merge_gate_up_raw(
        merged.data(), merged.size(), gate.data(), gate.size(), up.data(), up.size(), n_expert);
    for (size_t expert = 0; ok && expert < n_expert; ++expert) {
        const size_t dst = expert * (gate_plane + up_plane);
        for (size_t i = 0; i < gate_plane; ++i) {
            ok = ok && merged[dst + i] == gate[expert * gate_plane + i];
        }
        for (size_t i = 0; i < up_plane; ++i) {
            ok = ok && merged[dst + gate_plane + i] == up[expert * up_plane + i];
        }
    }

    ok = ok && !llama_ep_merge_gate_up_raw(
        merged.data(), merged.size() - 1, gate.data(), gate.size(), up.data(), up.size(), n_expert);
    ok = ok && !llama_ep_merge_gate_up_raw(
        merged.data(), merged.size(), gate.data(), gate.size(), up.data(), up.size(), 0);
    ok = ok && !llama_ep_merge_gate_up_raw(
        merged.data(), merged.size(), gate.data(), gate.size() - 1, up.data(), up.size(), n_expert);

    printf("ep fusion tests: %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
