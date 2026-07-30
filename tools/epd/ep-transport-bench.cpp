// ep-transport-bench: TCP vs RDMA round-trip microbenchmark over the EP
// transport function table (the exact code path used by remote EP dispatch).
//
//   echo server:  ep-transport-bench echo <tcp|rdma> <port>
//   ping client:  ep-transport-bench ping <tcp|rdma> <host> <port>
//
// Protocol: client sends [u32 len][payload], server echoes the payload back.
// Reports RTT median/p90 per size plus client-side thread CPU time per op
// (CLOCK_THREAD_CPUTIME_ID), which is the metric this backend targets.

#include "llama-ep-transport.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <time.h>
#include <vector>

static double now_s() {
    timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static double cpu_s() {
    timespec ts;
    clock_gettime(CLOCK_THREAD_CPUTIME_ID, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static llama_ep_transport * connect(const char * kind, const char * host, int port) {
    std::string err;
    llama_ep_transport * t = nullptr;
    if (strcmp(kind, "tcp") == 0) {
        t = llama_ep_tcp_connect(host, port, &err);
    } else if (strcmp(kind, "rdma") == 0) {
#ifdef LLAMA_EP_HAVE_RDMA
        t = llama_ep_rdma_connect(host, port, &err);
#else
        err = "built without LLAMA_EP_HAVE_RDMA";
#endif
    } else {
        err = std::string("unknown transport '") + kind + "'";
    }
    if (!t) {
        fprintf(stderr, "connect(%s %s:%d) failed: %s\n", kind, host, port, err.c_str());
    }
    return t;
}

static llama_ep_listener * listen(const char * kind, int port) {
    std::string err;
    llama_ep_listener * l = nullptr;
    if (strcmp(kind, "tcp") == 0) {
        l = llama_ep_tcp_listen(nullptr, port, &err);
    } else if (strcmp(kind, "rdma") == 0) {
#ifdef LLAMA_EP_HAVE_RDMA
        l = llama_ep_rdma_listen(nullptr, port, &err);
#else
        err = "built without LLAMA_EP_HAVE_RDMA";
#endif
    } else {
        err = std::string("unknown transport '") + kind + "'";
    }
    if (!l) {
        fprintf(stderr, "listen(%s :%d) failed: %s\n", kind, port, err.c_str());
    }
    return l;
}

static int cmd_echo(const char * kind, int port) {
    llama_ep_listener * l = listen(kind, port);
    if (!l) {
        return 1;
    }
    fprintf(stderr, "echo server (%s) on :%d\n", kind, port);
    std::vector<uint8_t> buf(64 << 20);
    for (;;) {
        llama_ep_transport conn;
        if (!l->ops.accept(l->ctx, &conn)) {
            continue;
        }
        fprintf(stderr, "client connected\n");
        for (;;) {
            uint32_t len = 0;
            if (!conn.ops.recv_all(conn.ctx, &len, sizeof(len))) {
                break;
            }
            if (len > buf.size()) {
                break;
            }
            if (!conn.ops.recv_all(conn.ctx, buf.data(), len)) {
                break;
            }
            if (!conn.ops.send_all(conn.ctx, buf.data(), len)) {
                break;
            }
        }
        conn.ops.close(conn.ctx);
        fprintf(stderr, "client disconnected\n");
    }
    return 0;
}

static int cmd_ping(const char * kind, const char * host, int port) {
    llama_ep_transport * t = connect(kind, host, port);
    if (!t) {
        return 1;
    }

    static const size_t sizes[]   = {64, 4096, 65536, 1048576, 4194304, 16777216};
    static const char * names[]   = {"ping64B", "act4K", "act64K", "act1M", "act4M", "act16M"};
    static const int    samples[] = {2000, 2000, 1000, 200, 100, 50};
    const int n_sizes = 6;

    std::vector<uint8_t> buf(16 << 20, 0x5a);
    std::vector<double>  us;

    printf("{\"transport\":\"%s\",\"host\":\"%s\",\"results\":[\n", kind, host);
    for (int s = 0; s < n_sizes; ++s) {
        const size_t len = sizes[s];
        const int    n   = samples[s];
        us.resize(n);
        const double c0 = cpu_s();
        const double w0 = now_s();
        for (int i = 0; i < n; ++i) {
            const uint32_t hdr = (uint32_t) len;
            const double t0 = now_s();
            if (!t->ops.send_all(t->ctx, &hdr, sizeof(hdr)) ||
                !t->ops.send_all(t->ctx, buf.data(), len) ||
                !t->ops.recv_all(t->ctx, buf.data(), len)) {
                fprintf(stderr, "io error at size %zu iter %d\n", len, i);
                return 1;
            }
            us[i] = (now_s() - t0) * 1e6;
        }
        const double wall = now_s() - w0;
        const double cpu  = cpu_s() - c0;
        std::sort(us.begin(), us.end());
        const double med = us[n / 2];
        const double p90 = us[(int) (0.90 * (n - 1))];
        const double p99 = us[(int) (0.99 * (n - 1))];
        printf("  {\"name\":\"%s\",\"bytes\":%zu,\"samples\":%d,"
               "\"rtt_us_median\":%.1f,\"rtt_us_p90\":%.1f,\"rtt_us_p99\":%.1f,"
               "\"cpu_us_per_op\":%.1f,\"cpu_frac\":%.3f,\"mb_per_s\":%.1f}%s\n",
               names[s], len, n, med, p90, p99,
               cpu * 1e6 / n, wall > 0 ? cpu / wall : 0.0,
               wall > 0 ? (double) len * n / wall / 1e6 : 0.0,
               s == n_sizes - 1 ? "" : ",");
        fprintf(stderr, "  %-7s %8zu B x%-5d rtt med %7.1f us  p90 %7.1f us  p99 %8.1f us  cpu %6.1f us/op (%4.1f%%)  %8.1f MB/s\n",
                names[s], len, n, med, p90, p99, cpu * 1e6 / n, 100.0 * cpu / wall,
                wall > 0 ? (double) len * n / wall / 1e6 : 0.0);
    }
    printf("]}\n");

    t->ops.close(t->ctx);
    delete t;
    return 0;
}

int main(int argc, char ** argv) {
    if (argc >= 4 && strcmp(argv[1], "echo") == 0) {
        return cmd_echo(argv[2], atoi(argv[3]));
    }
    if (argc >= 5 && strcmp(argv[1], "ping") == 0) {
        return cmd_ping(argv[2], argv[3], atoi(argv[4]));
    }
    fprintf(stderr,
        "usage:\n"
        "  %s echo <tcp|rdma> <port>\n"
        "  %s ping <tcp|rdma> <host> <port>\n", argv[0], argv[0]);
    return 2;
}
