// ep-topo-probe.c — NUMA 拓扑探测器（EPD 配套）
//
// 子命令（JSON 片段输出到 stdout，进度到 stderr）：
//   membw  <node> [MiB] [iters]          节点本地读带宽（线程绑 node，首触分配）
//   pingpong <nodeA> <nodeB> [rounds] [samples]
//                                        缓存线 ping-pong 往返延迟（ns，median/p90）
//   tcpecho <port> <node>                TCP 回显服务端（供 tcping 对端）
//   tcping <host> <port> <node>          TCP 小消息 ping + 激活载荷往返（us）
//
// 构建：gcc -O2 -o ep-topo-probe ep-topo-probe.c -lnuma -lpthread
#define _GNU_SOURCE
#include <errno.h>
#include <fcntl.h>
#include <math.h>
#include <numa.h>
#include <pthread.h>
#include <sched.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <arpa/inet.h>
#include <netdb.h>

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static int cmp_d(const void *a, const void *b) {
    double x = *(const double *) a, y = *(const double *) b;
    return (x > y) - (x < y);
}

// median / p90 of unsorted array (sorted in place)
static void med_p90(double *v, int n, double *med, double *p90) {
    qsort(v, n, sizeof(double), cmp_d);
    *med = v[n / 2];
    int i90 = (int) (0.90 * (n - 1));
    *p90 = v[i90];
}

// pin current thread to the nth cpu of numa node (n wraps around node cpu count)
static int pin_node_nth_cpu(int node, int n) {
    struct bitmask *bm = numa_allocate_cpumask();
    if (numa_node_to_cpus(node, bm) != 0) { perror("numa_node_to_cpus"); return -1; }
    int seen = -1;
    unsigned long cpu = bm->size;
    for (unsigned long c = 0; c < bm->size; c++)
        if (numa_bitmask_isbitset(bm, c)) {
            seen++;
            if (seen == n % (int) numa_bitmask_weight(bm)) { cpu = c; break; }
        }
    if (cpu >= bm->size) { fprintf(stderr, "node %d has no cpus\n", node); return -1; }
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(cpu, &set);
    int rc = sched_setaffinity(0, sizeof(set), &set);
    numa_free_cpumask(bm);
    return rc;
}

static int pin_node_first_cpu(int node) { return pin_node_nth_cpu(node, 0); }

// ---------------------------------------------------------------- membw

typedef struct {
    int      node;
    double  *buf;      // this thread's slice
    size_t   n;        // doubles in slice
    double   best;     // best read time over iters
    uint64_t checksum;
} membw_arg_t;

// membw worker needs iters; use a global to keep the struct simple
static int g_iters = 10;

static void *membw_worker(void *p) {
    membw_arg_t *a = (membw_arg_t *) p;
    struct bitmask *bm = numa_allocate_cpumask();
    numa_node_to_cpus(a->node, bm);
    cpu_set_t set;
    CPU_ZERO(&set);
    for (unsigned long c = 0; c < bm->size && c < CPU_SETSIZE; c++)
        if (numa_bitmask_isbitset(bm, c)) CPU_SET(c, &set);
    sched_setaffinity(0, sizeof(set), &set);
    numa_free_cpumask(bm);

    double *b = a->buf;
    for (size_t i = 0; i < a->n; i++) b[i] = (double) i * 1e-9;

    a->best = 1e30;
    for (int it = 0; it < g_iters; it++) {
        double t0 = now_s();
        double s = 0.0;
        for (size_t i = 0; i < a->n; i++) s += b[i];
        double dt = now_s() - t0;
        if (dt < a->best) a->best = dt;
        a->checksum = (uint64_t) s;
    }
    return NULL;
}

static int cmd_membw(int node, size_t mib, int iters, int threads) {
    if (node >= 0 && node > numa_max_node()) {
        fprintf(stderr, "node %d out of range (max %d)\n", node, numa_max_node());
        return 1;
    }
    if (threads <= 0) {
        // 默认 = 节点 cpu 数
        struct bitmask *bm = numa_allocate_cpumask();
        numa_node_to_cpus(node, bm);
        threads = 0;
        for (unsigned long c = 0; c < bm->size; c++)
            if (numa_bitmask_isbitset(bm, c)) threads++;
        numa_free_cpumask(bm);
    }
    g_iters = iters;
    size_t total = (mib << 20) / sizeof(double);
    size_t per   = total / threads;
    double *base = aligned_alloc(64, per * threads * sizeof(double));
    if (!base) { perror("alloc"); return 1; }

    pthread_t  *tid = malloc(threads * sizeof(pthread_t));
    membw_arg_t *arg = calloc(threads, sizeof(membw_arg_t));
    for (int t = 0; t < threads; t++) {
        arg[t].node = node;
        arg[t].buf  = base + t * per;
        arg[t].n    = per;
        pthread_create(&tid[t], NULL, membw_worker, &arg[t]);
    }
    double t0 = now_s();
    for (int t = 0; t < threads; t++) pthread_join(tid[t], NULL);
    double wall = now_s() - t0;
    (void) wall;

    // 聚合带宽：总字节 / max(best)（各线程并发，瓶颈为内存控制器，近似取最慢线程）
    double slowest = 0;
    uint64_t cs = 0;
    for (int t = 0; t < threads; t++) {
        if (arg[t].best > slowest) slowest = arg[t].best;
        cs ^= arg[t].checksum;
    }
    double gbps = (double) per * threads * sizeof(double) / slowest / 1e9;
    printf("{\"kind\":\"membw\",\"node\":%d,\"threads\":%d,\"mib\":%zu,"
           "\"read_gbps\":%.1f,\"best_s\":%.4f,\"checksum\":%llu}\n",
           node, threads, mib, gbps, slowest, (unsigned long long) cs);
    free(base); free(tid); free(arg);
    return 0;
}

// ---------------------------------------------------------------- pingpong

typedef struct {
    volatile uint64_t seq;
    char pad[64 - sizeof(uint64_t)];
} cacheline_t;

typedef struct {
    cacheline_t *a2b, *b2a;
    int node, role;   // role 0 = initiator, 1 = responder
    int samples;
    double *rtt;      // initiator fills
} pp_arg_t;

static void *pp_thread(void *p) {
    pp_arg_t *a = (pp_arg_t *) p;
    if (pin_node_nth_cpu(a->node, a->role) != 0) return (void *) 1;
    if (a->role == 0) {
        for (int i = 0; i < a->samples; i++) {
            double t0 = now_s();
            __atomic_store_n(&a->a2b->seq, (uint64_t) i + 1, __ATOMIC_RELEASE);
            uint64_t v;
            while ((v = __atomic_load_n(&a->b2a->seq, __ATOMIC_ACQUIRE)) != (uint64_t) i + 1)
                ;
            (void) v;
            a->rtt[i] = (now_s() - t0) * 1e9;   // ns
        }
    } else {
        for (int i = 0; i < a->samples; i++) {
            uint64_t v;
            while ((v = __atomic_load_n(&a->a2b->seq, __ATOMIC_ACQUIRE)) != (uint64_t) i + 1)
                ;
            (void) v;
            __atomic_store_n(&a->b2a->seq, (uint64_t) i + 1, __ATOMIC_RELEASE);
        }
    }
    return NULL;
}

static int cmd_pingpong(int na, int nb, int rounds, int samples) {
    void *mem = aligned_alloc(64, 2 * sizeof(cacheline_t));
    memset(mem, 0, 2 * sizeof(cacheline_t));
    double *rtt = malloc(samples * sizeof(double));

    double med_sum = 0, p90_sum = 0;
    for (int r = 0; r < rounds; r++) {
        pp_arg_t aa = { .a2b = (cacheline_t *) mem, .b2a = (cacheline_t *) mem + 1,
                        .node = na, .role = 0, .samples = samples, .rtt = rtt };
        pp_arg_t ab = aa; ab.node = nb; ab.role = 1;
        pthread_t ta, tb;
        // warmup: 先各跑一轮不计时
        pthread_create(&tb, NULL, pp_thread, &ab);
        pthread_create(&ta, NULL, pp_thread, &aa);
        if (pthread_join(ta, NULL) || 0) {}
        void *rcb = NULL;
        pthread_join(tb, &rcb);
        if (rcb) { fprintf(stderr, "pin failed on node %d\n", nb); return 1; }
        double med, p90;
        med_p90(rtt, samples, &med, &p90);
        med_sum += med; p90_sum += p90;
        fprintf(stderr, "  round %d: median %.1f ns  p90 %.1f ns\n", r, med, p90);
    }
    printf("{\"kind\":\"pingpong\",\"node_a\":%d,\"node_b\":%d,\"rounds\":%d,"
           "\"samples\":%d,\"rtt_ns_median\":%.1f,\"rtt_ns_p90\":%.1f,"
           "\"oneway_ns_median\":%.1f}\n",
           na, nb, rounds, samples, med_sum / rounds, p90_sum / rounds,
           med_sum / rounds / 2.0);
    free(mem); free(rtt);
    return 0;
}

// ---------------------------------------------------------------- TCP

static int write_full(int fd, const void *buf, size_t n) {
    const char *p = buf;
    while (n) {
        ssize_t w = write(fd, p, n);
        if (w <= 0) return -1;
        p += w; n -= (size_t) w;
    }
    return 0;
}

static int read_full(int fd, void *buf, size_t n) {
    char *p = buf;
    while (n) {
        ssize_t r = read(fd, p, n);
        if (r <= 0) return -1;
        p += r; n -= (size_t) r;
    }
    return 0;
}

static int cmd_tcpecho(int port, int node) {
    if (node >= 0) {
        if (pin_node_first_cpu(node) != 0) return 1;
        numa_run_on_node(node);
    }
    int ls = socket(AF_INET, SOCK_STREAM, 0);
    int one = 1;
    setsockopt(ls, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
    struct sockaddr_in sa = { .sin_family = AF_INET, .sin_port = htons((uint16_t) port),
                              .sin_addr.s_addr = INADDR_ANY };
    if (bind(ls, (struct sockaddr *) &sa, sizeof(sa)) || listen(ls, 1)) {
        perror("bind/listen"); return 1;
    }
    fprintf(stderr, "tcpecho listening :%d node %d\n", port, node);
    size_t cap = 1 << 20;
    char *buf = malloc(cap);
    for (;;) {
        int c = accept(ls, NULL, NULL);
        if (c < 0) continue;
        setsockopt(c, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
        for (;;) {
            uint32_t len;
            if (read_full(c, &len, sizeof(len))) break;
            len = ntohl(len);
            if (len > cap) break;
            if (read_full(c, buf, len)) break;
            if (write_full(c, buf, len)) break;
        }
        close(c);
    }
    return 0;
}

static int cmd_tcping(const char *host, int port, int node) {
    if (node >= 0) {
        if (pin_node_first_cpu(node) != 0) return 1;
        numa_run_on_node(node);
    }
    char portstr[16];
    snprintf(portstr, sizeof(portstr), "%d", port);
    struct addrinfo hints = { .ai_family = AF_INET, .ai_socktype = SOCK_STREAM }, *ai = NULL;
    if (getaddrinfo(host, portstr, &hints, &ai) != 0) { perror("getaddrinfo"); return 1; }
    int fd = socket(ai->ai_family, ai->ai_socktype, ai->ai_protocol);
    if (connect(fd, ai->ai_addr, ai->ai_addrlen) != 0) { perror("connect"); return 1; }
    freeaddrinfo(ai);
    int one = 1;
    setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));

    static const size_t sizes[] = { 64, 8192, 32768, 131072, 524288 };
    static const char *names[]  = { "ping64B", "act8K", "act32K", "act128K", "act512K" };
    size_t cap = 1 << 20;
    char *buf = malloc(cap);
    memset(buf, 0x5a, cap);
    double *us = malloc(4096 * sizeof(double));

    printf("{\"kind\":\"tcping\",\"host\":\"%s\",\"node\":%d,\"results\":[", host, node);
    for (int s = 0; s < 5; s++) {
        size_t len = sizes[s];
        int samples = len <= 32768 ? 1000 : 200;
        for (int i = 0; i < samples; i++) {
            uint32_t hdr = htonl((uint32_t) len);
            double t0 = now_s();
            if (write_full(fd, &hdr, sizeof(hdr)) || write_full(fd, buf, len)) return 1;
            if (read_full(fd, buf, len)) return 1;
            us[i] = (now_s() - t0) * 1e6;
        }
        double med, p90;
        med_p90(us, samples, &med, &p90);
        // 有效单向带宽：载荷字节 / (RTT/2)
        double gbps = len / (med * 1e-6) / 2.0 / 1e9;
        printf("%s{\"name\":\"%s\",\"bytes\":%zu,\"samples\":%d,"
               "\"rtt_us_median\":%.1f,\"rtt_us_p90\":%.1f,\"oneway_gbps\":%.2f}",
               s ? "," : "", names[s], len, samples, med, p90, gbps);
        fprintf(stderr, "  %-8s %7zu B  x%d  rtt median %.1f us  p90 %.1f us\n",
                names[s], len, samples, med, p90);
    }
    printf("]}\n");
    close(fd);
    free(buf); free(us);
    return 0;
}

// ---------------------------------------------------------------- main

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr,
            "usage:\n"
            "  %s membw <node> [MiB=512] [iters=10] [threads=node cpus]\n"
            "  %s pingpong <nodeA> <nodeB> [rounds=5] [samples=2000]\n"
            "  %s tcpecho <port> <node|-1>\n"
            "  %s tcping <host> <port> <node|-1>\n", argv[0], argv[0], argv[0], argv[0]);
        return 2;
    }
    if (!strcmp(argv[1], "membw") && argc >= 3) {
        int node = atoi(argv[2]);
        size_t mib = argc > 3 ? (size_t) atoi(argv[3]) : 512;
        int iters  = argc > 4 ? atoi(argv[4]) : 10;
        int thr    = argc > 5 ? atoi(argv[5]) : 0;
        return cmd_membw(node, mib, iters, thr);
    }
    if (!strcmp(argv[1], "pingpong") && argc >= 4) {
        int rounds  = argc > 4 ? atoi(argv[4]) : 5;
        int samples = argc > 5 ? atoi(argv[5]) : 2000;
        return cmd_pingpong(atoi(argv[2]), atoi(argv[3]), rounds, samples);
    }
    if (!strcmp(argv[1], "tcpecho") && argc >= 4)
        return cmd_tcpecho(atoi(argv[2]), atoi(argv[3]));
    if (!strcmp(argv[1], "tcping") && argc >= 5)
        return cmd_tcping(argv[2], atoi(argv[3]), atoi(argv[4]));
    fprintf(stderr, "bad args\n");
    return 2;
}
