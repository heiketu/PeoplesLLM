#include "ggml-shard-plan.h"

#include <cassert>
#include <limits>
#include <string>

static ggml_shard_plan make_plan(
        ggml_shard_placement placement,
        const std::vector<float> & capacities,
        const std::vector<ggml_shard_segment> & segments,
        size_t rotation = 0,
        size_t owner = 0) {
    ggml_shard_plan_input input;
    input.placement = placement;
    input.axis = placement == ggml_shard_placement::SPLIT ? 1 : -1;
    input.rotation = rotation;
    input.owner = owner;
    input.segments = segments;
    for (size_t i = 0; i < capacities.size(); ++i) {
        input.domains.push_back({i, ggml_shard_domain_kind::GPU, capacities[i]});
    }
    ggml_shard_plan output;
    std::string error;
    assert(ggml_shard_plan_build(input, output, &error));
    assert(error.empty());
    return output;
}

int main() {
    {
        const ggml_shard_plan plan = make_plan(ggml_shard_placement::SPLIT, {0, 0, 0, 0}, {{1024, 1, 128}});
        assert((plan.extents == std::vector<int64_t>{256, 256, 256, 256}));
    }
    {
        const ggml_shard_plan plan = make_plan(ggml_shard_placement::SPLIT, {1, 2, 1, 0}, {{1024, 1, 128}});
        assert((plan.extents == std::vector<int64_t>{256, 512, 256, 0}));
        assert((plan.offsets == std::vector<int64_t>{0, 256, 768, 1024}));
    }
    {
        const ggml_shard_plan plan = make_plan(ggml_shard_placement::SPLIT, {1, 1, 1, 1}, {{1000, 2, 128}}, 1);
        assert((plan.extents == std::vector<int64_t>{360, 128, 256, 256}));
        assert((plan.offsets == std::vector<int64_t>{640, 0, 128, 384}));
        assert(plan.rotation == 1);
        assert(plan.domains.size() == 4 && plan.domains[3].id == 3);
        assert(plan.segments[0].repetitions == 2);
    }
    {
        const ggml_shard_plan plan = make_plan(ggml_shard_placement::SPLIT, {1, 1}, {{384, 2, 128}, {256, 1, 64}}, 1);
        assert((plan.extents == std::vector<int64_t>{256, 128, 128, 128}));
    }
    {
        const ggml_shard_plan owner = make_plan(ggml_shard_placement::OWNER, {1, 1}, {{128, 1, 1}}, 0, 1);
        const ggml_shard_plan mirrored = make_plan(ggml_shard_placement::MIRRORED, {1, 1}, {{128, 1, 1}});
        const ggml_shard_plan partial = make_plan(ggml_shard_placement::PARTIAL, {1, 1}, {{128, 1, 1}});
        assert((owner.extents == std::vector<int64_t>{0, 128}));
        assert((mirrored.extents == std::vector<int64_t>{128, 128}));
        assert((partial.extents == std::vector<int64_t>{128, 128}));
    }
    {
        ggml_shard_plan_input input;
        input.placement = ggml_shard_placement::SPLIT;
        input.axis = 0;
        input.domains = {
            {10, ggml_shard_domain_kind::NUMA, 1.0},
            {42, ggml_shard_domain_kind::REMOTE, 1.0},
        };
        input.segments = {{256, 1, 64}};
        ggml_shard_plan output;
        assert(ggml_shard_plan_build(input, output));
        assert((output.extents == std::vector<int64_t>{128, 128}));
    }
    {
        ggml_shard_plan_input input;
        input.placement = ggml_shard_placement::SPLIT;
        input.axis = 4;
        input.domains = {{99, ggml_shard_domain_kind::NUMA, 1.0}};
        input.segments = {{128, 1, 1}};
        ggml_shard_plan output;
        std::string error;
        assert(!ggml_shard_plan_build(input, output, &error));
        assert(!error.empty());
    }
    {
        ggml_shard_plan_input input;
        input.placement = ggml_shard_placement::SPLIT;
        input.axis = 0;
        input.domains = {
            {7, ggml_shard_domain_kind::NUMA, 1.0f},
            {7, ggml_shard_domain_kind::NUMA, 1.0f},
        };
        input.segments = {{128, 1, 1}};
        ggml_shard_plan output;
        output.axis = 3;
        std::string error;
        assert(!ggml_shard_plan_build(input, output, &error));
        assert(output.axis == 3);
        assert(!error.empty());
    }
    {
        ggml_shard_execution_phase phase;
        phase.reduction_after = ggml_shard_collective::ALL_REDUCE;
        phase.broadcast_root = 1;
        phase.broadcast_values = {2, 7};
        assert(phase.reduction_after == ggml_shard_collective::ALL_REDUCE);
        assert(phase.broadcast_root == 1 && phase.broadcast_values.size() == 2);
    }
    {
        ggml_shard_window window;
        assert(ggml_shard_window_equal(1000, 4, 0, 128, window));
        assert(window.begin == 0 && window.end == 256 && window.stride == 256);
        assert(ggml_shard_window_equal(1000, 4, 3, 128, window));
        assert(window.begin == 768 && window.end == 1000);
        assert(ggml_shard_window_equal(1025, 4, 3, 128, window));
        assert(window.begin == 1025 && window.end == 1025);
        assert(!ggml_shard_window_equal(-1, 4, 0, 128, window));
        assert(!ggml_shard_window_equal(128, 0, 0, 128, window));
        assert(!ggml_shard_window_equal(std::numeric_limits<int64_t>::max(), 1, 0, 128, window));
    }
    return 0;
}
