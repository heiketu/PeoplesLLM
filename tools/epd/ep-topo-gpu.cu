// ep-topo-gpu.cu — GPU 传输拓扑探测（CUDA events 计时）
//
// 输出 JSON 片段：
//   h2d/d2h：每 GPU × 每 host NUMA 节点（numa_alloc_onnode + cudaHostRegister），
//            大块（64MB）带宽 + 激活级小载荷（8K/32K/128K/512K）延迟
//   p2p：GPU0<->GPU1 cudaMemcpyPeer，大块带宽 + 小载荷延迟
//
// 构建：nvcc -O2 -o ep-topo-gpu ep-topo-gpu.cu -lnuma
#include <cuda_runtime.h>
#include <numa.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); \
    exit(1); } } while (0)

static float event_ms(cudaEvent_t a, cudaEvent_t b) {
    float ms = 0;
    CK(cudaEventElapsedTime(&ms, a, b));
    return ms;
}

// median of collected samples (ms)
static float median(float *v, int n) {
    std::sort(v, v + n);
    return v[n / 2];
}

// 测一次拷贝延迟/带宽：warmup + iters 次 events 计时，取中位
static float bench_copy(void *dst, void *src, size_t bytes, cudaMemcpyKind kind,
                        int iters, float *out_med_ms) {
    cudaEvent_t ev0, ev1;
    CK(cudaEventCreate(&ev0));
    CK(cudaEventCreate(&ev1));
    float *ms = (float *) malloc(iters * sizeof(float));
    for (int i = 0; i < 3; i++) CK(cudaMemcpyAsync(dst, src, bytes, kind, 0));
    CK(cudaStreamSynchronize(0));
    for (int i = 0; i < iters; i++) {
        CK(cudaEventRecord(ev0));
        CK(cudaMemcpyAsync(dst, src, bytes, kind, 0));
        CK(cudaEventRecord(ev1));
        CK(cudaEventSynchronize(ev1));
        ms[i] = event_ms(ev0, ev1);
    }
    *out_med_ms = median(ms, iters);
    cudaEventDestroy(ev0); cudaEventDestroy(ev1);
    free(ms);
    return *out_med_ms;
}

int main() {
    int ndev = 0;
    CK(cudaGetDeviceCount(&ndev));
    if (ndev < 1) { fprintf(stderr, "no CUDA devices\n"); return 1; }
    if (ndev > 2) ndev = 2;
    int nnode = numa_max_node() + 1;

    const size_t BIG = 64u << 20;
    const size_t act_sizes[] = { 8192, 32768, 131072, 524288 };
    const char  *act_names[] = { "act8K", "act32K", "act128K", "act512K" };

    printf("{\"kind\":\"gpu\",\"devices\":%d,\"transfer\":[", ndev);
    int first = 1;
    for (int d = 0; d < ndev; d++) {
        CK(cudaSetDevice(d));
        void *dbuf;
        CK(cudaMalloc(&dbuf, BIG));
        for (int n = 0; n < nnode; n++) {
            char *hbuf = (char *) numa_alloc_onnode(BIG, n);
            memset(hbuf, 0x11, BIG);           // first touch on node n
            CK(cudaHostRegister(hbuf, BIG, cudaHostRegisterDefault));

            float ms;
            bench_copy(dbuf, hbuf, BIG, cudaMemcpyHostToDevice, 10, &ms);
            float h2d_gbps = BIG / (ms * 1e-3f) / 1e9f;
            bench_copy(hbuf, dbuf, BIG, cudaMemcpyDeviceToHost, 10, &ms);
            float d2h_gbps = BIG / (ms * 1e-3f) / 1e9f;

            printf("%s{\"dev\":%d,\"host_node\":%d,\"h2d_gbps\":%.2f,\"d2h_gbps\":%.2f,"
                   "\"act_us\":[", first ? "" : ",", d, n, h2d_gbps, d2h_gbps);
            first = 0;
            for (int s = 0; s < 4; s++) {
                bench_copy(dbuf, hbuf, act_sizes[s], cudaMemcpyHostToDevice, 100, &ms);
                float h2d_us = ms * 1e3f;
                bench_copy(hbuf, dbuf, act_sizes[s], cudaMemcpyDeviceToHost, 100, &ms);
                printf("%s{\"name\":\"%s\",\"h2d_us\":%.1f,\"d2h_us\":%.1f}",
                       s ? "," : "", act_names[s], h2d_us, ms * 1e3f);
            }
            printf("]}");
            fflush(stdout);
            CK(cudaHostUnregister(hbuf));
            numa_free(hbuf, BIG);
        }
        CK(cudaFree(dbuf));
    }
    printf("]}\n");

    // P2P 单独一行 JSON
    if (ndev == 2) {
        int can01 = 0, can10 = 0;
        CK(cudaDeviceCanAccessPeer(&can01, 0, 1));
        CK(cudaDeviceCanAccessPeer(&can10, 1, 0));
        void *buf0, *buf1;
        CK(cudaSetDevice(0)); CK(cudaMalloc(&buf0, BIG));
        CK(cudaSetDevice(1)); CK(cudaMalloc(&buf1, BIG));
        if (can01) { CK(cudaSetDevice(0)); CK(cudaDeviceEnablePeerAccess(1, 0)); }
        if (can10) { CK(cudaSetDevice(1)); CK(cudaDeviceEnablePeerAccess(0, 0)); }
        float ms, bw01 = 0, bw10 = 0, lat01 = 0, lat10 = 0;
        if (can01) {
            CK(cudaSetDevice(0));
            bench_copy(buf1, buf0, BIG, cudaMemcpyDeviceToDevice, 10, &ms);
            bw01 = BIG / (ms * 1e-3f) / 1e9f;
            bench_copy(buf1, buf0, 8192, cudaMemcpyDeviceToDevice, 100, &ms);
            lat01 = ms * 1e3f;
        }
        if (can10) {
            CK(cudaSetDevice(1));
            bench_copy(buf0, buf1, BIG, cudaMemcpyDeviceToDevice, 10, &ms);
            bw10 = BIG / (ms * 1e-3f) / 1e9f;
            bench_copy(buf0, buf1, 8192, cudaMemcpyDeviceToDevice, 100, &ms);
            lat10 = ms * 1e3f;
        }
        printf("{\"kind\":\"gpu-p2p\",\"can_0_to_1\":%d,\"can_1_to_0\":%d,"
               "\"bw_0to1_gbps\":%.2f,\"bw_1to0_gbps\":%.2f,"
               "\"lat8k_0to1_us\":%.1f,\"lat8k_1to0_us\":%.1f}\n",
               can01, can10, bw01, bw10, lat01, lat10);
    }
    return 0;
}
