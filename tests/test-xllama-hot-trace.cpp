#include "xllama-hot-trace.h"

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

static void set_env(const char * name, const char * value) {
#if defined(_WIN32)
    _putenv_s(name, value);
#else
    setenv(name, value, 1);
#endif
}

static void unset_env(const char * name) {
#if defined(_WIN32)
    _putenv_s(name, "");
#else
    unsetenv(name);
#endif
}

static uint64_t metadata_value(const std::string & line, const char * key) {
    const std::string prefix = std::string("# ") + key + "=";
    if (line.compare(0, prefix.size(), prefix) != 0) {
        return UINT64_MAX;
    }
    return std::strtoull(line.c_str() + prefix.size(), nullptr, 10);
}

int main(int argc, char ** argv) {
    if (argc > 1 && std::string(argv[1]) == "disabled") {
        unset_env("GGML_MOE_HOT_TRACE");
        return !xllama::moe_hot_trace_enabled() && !xllama::moe_hot_trace_flush() ? 0 : 1;
    }
    if (argc > 1 && std::string(argv[1]) == "hot-conflict") {
        set_env("GGML_MOE_HOT_TRACE", "1");
        set_env("GGML_HOT_EXPERT", "1");
        return !xllama::moe_hot_trace_enabled() && !xllama::moe_hot_trace_flush() ? 0 : 1;
    }

    const bool capacity_mode = argc > 1 && std::string(argv[1]) == "capacity";
    const bool keep_output = argc > 2;
    const std::string path = keep_output ? argv[2] :
        (capacity_mode ? "xllama-hot-trace-test-capacity.tsv" : "xllama-hot-trace-test-exact.tsv");
    if (!keep_output) {
        std::remove(path.c_str());
    }

    set_env("GGML_MOE_HOT_TRACE", "1");
    set_env("GGML_MOE_HOT_TRACE_PATH", path.c_str());
    set_env("GGML_MOE_HOT_TRACE_MAX_MB", "1");

    if (!xllama::moe_hot_trace_enabled()) {
        std::fprintf(stderr, "trace did not enable\n");
        return 1;
    }

    const int32_t ids[2][3] = {{3, 7, 9}, {4, 8, 10}};
    xllama::moe_hot_trace_record(12, ids, 2, 3, sizeof(ids[0]), sizeof(ids[0][0]), 16);
    xllama::moe_hot_trace_record(13, ids, 1, 3, sizeof(ids[0]), sizeof(ids[0][0]), 16);

    if (capacity_mode) {
        const int32_t invalid_ids[3] = {-1, 0, 1};
        xllama::moe_hot_trace_record(14, invalid_ids, 1, 3, sizeof(invalid_ids), sizeof(invalid_ids[0]), 1024);
        std::vector<int32_t> all_experts(1024);
        for (int i = 0; i < 1024; ++i) {
            all_experts[i] = i;
        }
        for (int i = 0; i < 300; ++i) {
            xllama::moe_hot_trace_record(14, all_experts.data(), 1, 1024,
                    all_experts.size() * sizeof(int32_t), sizeof(int32_t), 1024);
        }
    }

    if (!xllama::moe_hot_trace_flush()) {
        std::fprintf(stderr, "trace flush failed\n");
        return 1;
    }

    std::ifstream input(path);
    if (!input) {
        std::fprintf(stderr, "trace output is missing\n");
        return 1;
    }

    std::vector<std::string> records;
    uint64_t metadata_records = UINT64_MAX;
    uint64_t dropped_capacity = UINT64_MAX;
    uint64_t dropped_invalid = UINT64_MAX;
    uint64_t concurrent_overlaps = UINT64_MAX;
    std::string line;
    while (std::getline(input, line)) {
        if (line.empty()) {
            continue;
        }
        if (line[0] != '#') {
            records.push_back(line);
            continue;
        }
        const uint64_t records_value = metadata_value(line, "records");
        const uint64_t capacity_value = metadata_value(line, "dropped_capacity");
        const uint64_t invalid_value = metadata_value(line, "dropped_invalid");
        const uint64_t overlaps_value = metadata_value(line, "concurrent_overlaps");
        if (records_value != UINT64_MAX) {
            metadata_records = records_value;
        }
        if (capacity_value != UINT64_MAX) {
            dropped_capacity = capacity_value;
        }
        if (invalid_value != UINT64_MAX) {
            dropped_invalid = invalid_value;
        }
        if (overlaps_value != UINT64_MAX) {
            concurrent_overlaps = overlaps_value;
        }
    }

    bool ok = records.size() >= 3;
    ok = ok && records[0] == "0\t12\t3\t7\t9";
    ok = ok && records[1] == "1\t12\t4\t8\t10";
    ok = ok && records[2] == "2\t13\t3\t7\t9";
    ok = ok && metadata_records == records.size();
    ok = ok && dropped_invalid == (capacity_mode ? 1 : 0);
    ok = ok && concurrent_overlaps == 0;
    ok = ok && (capacity_mode ? dropped_capacity > 0 : dropped_capacity == 0);

    for (size_t i = 0; ok && i < records.size(); ++i) {
        std::istringstream fields(records[i]);
        uint64_t step = UINT64_MAX;
        int layer = -1;
        int expert = -1;
        fields >> step >> layer >> expert;
        ok = fields.good() && step == i && layer >= 0 && expert >= 0;
    }

    if (!keep_output) {
        std::remove(path.c_str());
    }
    if (!ok) {
        std::fprintf(stderr, "trace content mismatch\n");
        return 1;
    }
    return 0;
}
