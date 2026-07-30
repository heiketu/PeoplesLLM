#!/usr/bin/env python3
# glm-bench.py — GLM 测速客户端：TG96/TG512 + PP 摊销曲线 + gen48 对拍采样
# 用法:
#   python3 glm-bench.py wait                     # 等 server 健康
#   python3 glm-bench.py tg512                    # 单点 TG512
#   python3 glm-bench.py full <tag>               # TG96 + TG512x2 + PP 5/63/254/1020 + gen48
#   python3 glm-bench.py abba <tag>               # TG512x2 + PP63 + PP1020 + gen48（反序复测轮）
import json
import sys
import urllib.request

BASE = "http://127.0.0.1:18121"
FILLER = ("The quick brown fox jumps over the lazy dog and runs through the "
          "forest while the sun shines brightly over the green hills. ")


def post(path, payload, timeout=1800):
    req = urllib.request.Request(
        BASE + path, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def wait_health(timeout=900):
    import time
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            with urllib.request.urlopen(BASE + "/health", timeout=5) as r:
                if json.loads(r.read()).get("status") == "ok":
                    print(f"healthy after {time.time()-t0:.0f}s")
                    return True
        except Exception:
            pass
        time.sleep(5)
    print("TIMEOUT waiting for health")
    return False


def n_tokens(text):
    return len(post("/tokenize", {"content": text})["tokens"])


def make_prompt(target):
    text = FILLER * (target // 8 + 2)
    lo, hi = 0, len(text)
    # 二分字符前缀长度使 token 数 == target
    for _ in range(24):
        mid = (lo + hi) // 2
        n = n_tokens(text[:mid])
        if n == target:
            return text[:mid]
        if n < target:
            lo = mid
        else:
            hi = mid
        if hi - lo <= 1:
            break
    return text[:lo]


def run(prompt, n_predict, tag, save=None, ignore_eos=True):
    r = post("/completion", {
        "prompt": prompt, "n_predict": n_predict,
        "temperature": 0.0, "seed": 42, "cache_prompt": False,
        "stream": False, "ignore_eos": ignore_eos,
    })
    t = r["timings"]
    pp = t["prompt_n"] / (t["prompt_ms"] / 1000.0) if t["prompt_ms"] > 0 else 0
    tg = t["predicted_n"] / (t["predicted_ms"] / 1000.0) if t["predicted_ms"] > 0 else 0
    print(f"[{tag}] prompt_n={t['prompt_n']} prompt_ms={t['prompt_ms']:.0f} -> PP {pp:.2f} t/s | "
          f"pred_n={t['predicted_n']} pred_ms={t['predicted_ms']:.0f} -> TG {tg:.2f} t/s",
          flush=True)
    if save:
        with open(save, "w") as f:
            f.write(r["content"])
    return pp, tg


SHORT = "The capital of France is"


def main():
    cmd = sys.argv[1]
    if cmd == "wait":
        sys.exit(0 if wait_health() else 1)
    if cmd == "tg512":
        run(SHORT, 512, "TG512")
        return
    tag = sys.argv[2]
    if cmd == "full":
        run(SHORT, 48, "gen48", save=f"/tmp/gen48-{tag}.txt", ignore_eos=False)
        run(SHORT, 96, "TG96")
        run(SHORT, 512, "TG512-1")
        run(SHORT, 512, "TG512-2")
        for target in (5, 63, 254, 1020):
            p = make_prompt(target)
            run(p, 1, f"PP{target}")
    elif cmd == "abba":
        run(SHORT, 512, "TG512-1")
        run(SHORT, 512, "TG512-2")
        p63 = make_prompt(63)
        run(p63, 1, "PP63")
        p1020 = make_prompt(1020)
        run(p1020, 1, "PP1020")
        run(SHORT, 48, "gen48", save=f"/tmp/gen48-{tag}.txt", ignore_eos=False)
    else:
        print("unknown cmd")
        sys.exit(2)


main()
