#!/usr/bin/env python3
# ep-plan.py — NUMA 感知 EP 层分配规划器（EPD 配套）
#
# 输入：拓扑 profile JSON（ep-topo-run.sh 产出）+ 模型结构参数（内置 DSV4 默认）
# 输出：关键路径最短的层分配方案（GPU / master node0/1 / slave node0/1 层数、
#       worker 线程建议、预期 TG）
#
# 关键路径模型（decode 每 token）：
#   T = max(Tm, Ts) + (1-β)·min(Tm, Ts) + C
#   Tm = Lm · t_master        （master 本地 CPU MoE，双节点 mirror 口径）
#   Ts = Ls · (t_slave + t_net)（slave 计算 + 每层网络开销）
#   C  = GPU 段 + barrier 常数（由基线锚点拟合）
#   β  = 本地/远端路径重叠度（0=完全串行，1=完全重叠），由锚点校准
#
# 均衡约束：Ts = Tm（slave 计算+网络 = master 计算）时关键路径最短。
#
# 校准：--calibrate 用三组实测锚点（单机 / 双机12层 / 双机8层 TG512）拟合 β，
#       报告预测 vs 实测误差。
import argparse
import json
import math
import sys

# ---- DSV4 默认结构参数（dsv4-recipeB-v2，43层/256专家/top-6，Q3_K） ----
DSV4 = dict(
    name="DSV4",
    total_layers=43,
    moe_layers=list(range(3, 43)),      # 0-2 稠密，3-42 共 40 MoE 层
    gpu_layers=list(range(8, 22)),      # GPU 固定卸载 14 层
    experts=256,
    top_k=6,
    layer_weight_gb=2.0,                # ~16G/8层 ≈ 2.0G/层（Q3_K 实测 RSS 口径）
    remote_from="tail",                 # slave 认领尾部 MoE 层（35-42）
)

# ---- GLM-5.2 结构参数（glm-dsa，79层/160专家/top-8，UD-Q2_K_MXFP4 7分卷 ≈236G） ----
GLM = dict(
    name="GLM-5.2",
    total_layers=79,
    moe_layers=list(range(3, 79)),      # 0-2 稠密，3-78 共 76 MoE 层
    gpu_layers=[29, 30, 31, 32, 58, 59, 60, 61, 62],  # GPU 专家卸载 8 层
    experts=160,
    top_k=8,
    layer_weight_gb=2.9,                # 43.5G/15层 ≈ 2.9G/层（实测 RSS 口径）
    remote_from="head",                 # slave 认领前部 MoE 层（3 起）
)

MODELS = {"dsv4": DSV4, "glm": GLM}

# ---- 实测每层耗时（GGML_REMOTE_EP_DEBUG，见 tools/epd/README.md） ----
# DSV4：master 0.37 / slave 0.73 实测。GLM：由 TG512 三锚点（9.86/10.71/10.42）
# 反解 t_m=1.33、t_s=2.60（β=0.71 两锚点自洽，误差 <0.1%）；均为 slave 旧带宽
# （双节点合计 174 GB/s）口径，slave 换内存后按带宽比缩放 t_s 即可。
T_MASTER = {"dsv4": 0.37, "glm": 1.33}   # ms/层，master 本地（双节点 mirror）
T_SLAVE  = {"dsv4": 0.73, "glm": 2.60}   # ms/层，slave（旧带宽 174 GB/s 合计）
T_NET    = 0.13   # ms/层，网络（EDR 100Gb，激活 8KB 量级 RTT + 结果回传）

# ---- 节点带宽（membw 干净环境实测，用于非均衡分配比例） ----
# 2026-07-30 新硬件：master 重启后 t76 实测 156.9/153.8；slave 内存整改后
# 双节点 IMC 对称各 ~157 GB/s（总 ~315，旧 174 的 1.8×），旧的 94/80 作废。
BW_MASTER = [156.9, 153.8]   # master node0 / node1 GB/s
BW_SLAVE  = [157.0, 157.0]   # slave node0 / node1 GB/s（对称）

# ---- 线程模型（2026-07-30 实测标定） ----
# TG 是内存带宽敏感型：最优线程数 ≈ min(物理核数, 带宽饱和所需线程数)，
# 超物理核（用满 HT）ggml barrier 严重劣化且非单调（slave -t128/136 TG512 崩到 4-7 t/s）。
# PP 是计算密集型：可以物理核−少量保留核跑满。
SLAVE_PHYSICAL_CORES = 72    # slave 物理核（=2×36，HT 共 144）
PER_CORE_BW = 2.5            # GB/s/核，标定使 slave 推荐 ≈ 实测最优 72
SLAVE_PP_RESERVE = 4         # PP 保留给系统/后台的核数

# ---- 校准锚点（TG512 t/s → ms/token；Ls=远端层数，Lm=可分配-Ls） ----
# 来源：tools/epd/README.md，同会话 A/B 反序复测
ANCHORS = {
    "dsv4": [
        dict(ls=0,  tg=24.88, note="单机 mirror 基线（Lm=26）"),
        dict(ls=12, tg=24.18, note="双机 31-42 远端 12 层（复测）"),
        dict(ls=8,  tg=25.49, note="双机 35-42 远端 8 层（当前最优）"),
    ],
    "glm": [
        dict(ls=0,  tg=9.86,  note="单机基线（Lm=68，旧 mmap 口径）"),
        dict(ls=15, tg=10.71, note="双机 3-17 远端 15 层（t70/no-mmap）"),
        dict(ls=25, tg=10.42, note="双机 25 层（全面变差，已否定）"),
    ],
}


def split_by_bw(n, bw):
    """按带宽比例把 n 层分到两节点（非均衡：慢节点少分），返回 (n0, n1)。"""
    n0 = round(n * bw[0] / (bw[0] + bw[1]))
    n0 = max(0, min(n, n0))
    return n0, n - n0


def token_ms(lm, ls, beta, c, t_m=T_MASTER, t_s=T_SLAVE, t_net=T_NET):
    tm = lm * t_m
    ts = ls * (t_s + t_net)
    return max(tm, ts) + (1.0 - beta) * min(tm, ts) + c


def calibrate(anchors, n_cpu, t_m, t_s, t_net):
    """C 由基线锚点精确确定，β 对其余锚点最小二乘（网格搜索）。"""
    base = anchors[0]
    c = 1000.0 / base["tg"] - (n_cpu - base["ls"]) * t_m \
        - base["ls"] * (t_s + t_net)
    best = None
    for i in range(0, 1001):
        beta = i / 1000.0
        err = 0.0
        for a in anchors[1:]:
            lm = n_cpu - a["ls"]
            pred = token_ms(lm, a["ls"], beta, c, t_m, t_s, t_net)
            err += (1000.0 / pred - a["tg"]) ** 2
        if best is None or err < best[0]:
            best = (err, beta)
    return c, best[1]


def plan(model, beta, c, t_m, t_s, t_net, max_remote, bw_master, bw_slave):
    rows = []
    n_cpu = len(model["moe_layers"]) - len(model["gpu_layers"])
    for ls in range(0, max_remote + 1):
        lm = n_cpu - ls
        if lm < 0:
            break
        ms = token_ms(lm, ls, beta, c, t_m, t_s, t_net)
        rows.append((ls, lm, ms, 1000.0 / ms))
    best = min(rows, key=lambda r: r[2])
    return rows, best


def main():
    ap = argparse.ArgumentParser(description="NUMA 感知 EP 层分配规划器")
    ap.add_argument("--model", choices=sorted(MODELS), default="dsv4", help="模型结构预设")
    ap.add_argument("--profile", default=None, help="ep-topo-profile.json 路径（展示网络实测）")
    ap.add_argument("--t-master", type=float, default=None, help="master ms/层（缺省=预设值）")
    ap.add_argument("--t-slave", type=float, default=None, help="slave ms/层（缺省=预设值）")
    ap.add_argument("--t-net", type=float, default=T_NET, help="网络 ms/层")
    ap.add_argument("--beta", type=float, default=None, help="重叠度（缺省=锚点校准值）")
    ap.add_argument("--const", type=float, default=None, help="GPU段+barrier 常数 ms（缺省=校准值）")
    ap.add_argument("--max-remote", type=int, default=16, help="扫描的最大远端层数")
    ap.add_argument("--bw-master", type=float, nargs=2, default=BW_MASTER)
    ap.add_argument("--bw-slave", type=float, nargs=2, default=BW_SLAVE)
    ap.add_argument("--slave-physical-cores", type=int, default=SLAVE_PHYSICAL_CORES,
                    help="slave 物理核数（不是 HT 线程数）")
    ap.add_argument("--slave-pp-reserve", type=int, default=SLAVE_PP_RESERVE,
                    help="PP 预留系统核（PP=计算密集，可跑满物理核−保留核）")
    ap.add_argument("--per-core-bw", type=float, default=PER_CORE_BW,
                    help="单核内存带宽 GB/s（TG 带宽饱和估算）")
    ap.add_argument("--calibrate", action="store_true", help="只做校准验证并报告误差")
    ap.add_argument("--json", action="store_true", help="输出机器可读 JSON 方案")
    args = ap.parse_args()

    model = MODELS[args.model]
    key = args.model
    anchors = ANCHORS[key]
    n_cpu = len(model["moe_layers"]) - len(model["gpu_layers"])
    t_m = args.t_master if args.t_master is not None else T_MASTER[key]
    t_s = args.t_slave if args.t_slave is not None else T_SLAVE[key]
    t_net = args.t_net
    c_fit, beta_fit = calibrate(anchors, n_cpu, t_m, t_s, t_net)
    beta = args.beta if args.beta is not None else beta_fit
    c = args.const if args.const is not None else c_fit

    if args.profile:
        try:
            prof = json.load(open(args.profile))
            print(f"[profile] {args.profile} ({prof.get('timestamp','?')})")
            for k in sorted(prof.get("tcping", {})):
                r = prof["tcping"][k]["results"]
                print(f"  tcping {k}: 64B RTT {r[0]['rtt_us_median']:.0f}us "
                      f"p90 {r[0]['rtt_us_p90']:.0f}us, 8K RTT {r[1]['rtt_us_median']:.0f}us, "
                      f"512K 双向 {r[4].get('bidir_gbps',0):.2f} GB/s")
        except Exception as e:
            print(f"[profile] 读取失败（忽略，用内置实测常数）: {e}", file=sys.stderr)

    # ---- 校准验证 ----
    print("== 校准（锚点：实测 TG512） ==")
    print(f"  拟合参数: C = {c_fit:.2f} ms（GPU段+barrier），β = {beta_fit:.3f}（重叠度）")
    print(f"  {'配置':<28}{'Lm':>4}{'Ls':>4}{'预测ms':>9}{'预测TG':>9}{'实测TG':>9}{'误差':>8}")
    for a in anchors:
        lm = n_cpu - a["ls"]
        ms = token_ms(lm, a["ls"], beta_fit, c_fit, t_m, t_s, t_net)
        tg = 1000.0 / ms
        err = (tg - a["tg"]) / a["tg"] * 100
        print(f"  {a['note']:<28}{lm:>4}{a['ls']:>4}{ms:>9.2f}{tg:>9.2f}{a['tg']:>9.2f}{err:>7.1f}%")
    if args.calibrate:
        return 0

    # ---- 规划 ----
    rows, (ls, lm, ms, tg) = plan(model, beta, c, t_m, t_s, t_net,
                                  args.max_remote, args.bw_master, args.bw_slave)
    moe = model["moe_layers"]
    gpu = model["gpu_layers"]
    if model.get("remote_from") == "head":
        remote_layers = moe[:ls]
    else:
        remote_layers = moe[-ls:] if ls else []
    local_layers = [l for l in moe if l not in gpu and l not in remote_layers]

    m0, m1 = split_by_bw(lm, args.bw_master)
    s0, s1 = split_by_bw(ls, args.bw_slave)
    bw_sat = math.ceil(sum(args.bw_slave) / args.per_core_bw)  # 带宽饱和所需线程
    slave_threads = min(args.slave_physical_cores, bw_sat)     # TG：物理核与带宽饱和取小
    slave_pp = args.slave_physical_cores - args.slave_pp_reserve  # PP：物理核−保留核
    st0 = round(slave_threads * s0 / ls) if ls else 0
    st1 = slave_threads - st0 if ls else 0

    print(f"\n== 扫描（t_m={t_m} t_s={t_s} t_net={t_net} β={beta:.3f} C={c:.2f}） ==")
    print(f"  {'Ls':>3}{'Lm':>4}{'Tm(ms)':>9}{'Ts(ms)':>9}{'T(ms)':>9}{'TG':>8}  标记")
    anchor_ls = {a["ls"] for a in anchors}
    for r_ls, r_lm, r_ms, r_tg in rows:
        tm = r_lm * t_m
        ts = r_ls * (t_s + t_net)
        mark = " <-- 最优" if r_ls == ls else (" (实测锚点)" if r_ls in anchor_ls else "")
        print(f"  {r_ls:>3}{r_lm:>4}{tm:>9.2f}{ts:>9.2f}{r_ms:>9.2f}{r_tg:>8.2f}{mark}")

    print(f"\n== 推荐配置（{model['name']}） ==")
    print(f"  GPU 固定卸载: {len(gpu)} 层")
    print(f"  master 本地: {lm} 层 ", end="")
    if local_layers:
        # 压缩成区间表示
        rngs = []
        start = prev = local_layers[0]
        for l in local_layers[1:]:
            if l == prev + 1:
                prev = l
            else:
                rngs.append((start, prev)); start = prev = l
        rngs.append((start, prev))
        print(",".join(f"{a}-{b}" if a != b else f"{a}" for a, b in rngs))
    else:
        print("(无)")
    print(f"    node0: {m0} 层 / node1: {m1} 层  (带宽比 {args.bw_master[0]}:{args.bw_master[1]})")
    if ls:
        print(f"  slave 远端: {ls} 层 ({remote_layers[0]}-{remote_layers[-1]})"
              f"  → GGML_REMOTE_EP_LAYERS={remote_layers[0]}-{remote_layers[-1]}")
        print(f"    snode0: {s0} 层 × ~{st0} 线程 / snode1: {s1} 层 × ~{st1} 线程"
              f"  (带宽比 {args.bw_slave[0]}:{args.bw_slave[1]}，慢节点少分)")
        print(f"  slave 权重: ~{ls * model['layer_weight_gb']:.0f}G，master RSS 省"
              f" ~{ls * model['layer_weight_gb']:.0f}G")
    else:
        print("  slave 远端: 无（单机最优）")
    print(f"  worker 线程: TG -t {slave_threads}（min(物理核 {args.slave_physical_cores},"
          f" 带宽饱和 {bw_sat})，实测最优 72，超物理核崩溃切勿用 HT 数）")
    print(f"               PP --threads-batch {slave_pp}（物理核−保留 {args.slave_pp_reserve}）")
    print(f"  预期: {ms:.2f} ms/token → TG ≈ {tg:.2f} t/s")
    base_ms = rows[0][2]
    print(f"  对比单机基线: {base_ms:.2f} ms ({1000/base_ms:.2f} t/s)，"
          f"差 {1000*(1/tg - 1/(1000/base_ms)):+.2f} ms/token")

    # 均衡约束说明
    n_eq = (len(moe) - len(gpu)) * t_m / (t_m + t_s + t_net)
    print(f"\n  均衡点（Ts=Tm）: N* = {n_eq:.2f} 层 → 取整 {round(n_eq)}；"
          f"关键路径 min-max 最优 Ls = {ls}")

    if args.json:
        out = dict(gpu_layers=[gpu[0], gpu[-1]],
                   master=dict(layers=lm, node0=m0, node1=m1),
                   slave=dict(layers=ls,
                              range=[remote_layers[0], remote_layers[-1]] if ls else None,
                              snode0=s0, snode1=s1, threads=slave_threads,
                              pp_threads=slave_pp,
                              snode0_threads=st0, snode1_threads=st1),
                   predicted_ms=round(ms, 2), predicted_tg=round(tg, 2),
                   params=dict(t_m=t_m, t_s=t_s, t_net=t_net, beta=beta, c=c))
        print("\n" + json.dumps(out, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
