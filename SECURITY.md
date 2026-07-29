# 安全政策 / Security Policy

## 中文版

### 支持范围

仅 `main` 分支接受安全修复。本项目为个人调优分支，作者只维护自己用到的配置。

### 报告安全问题

- **敏感问题**（如可被远程利用的漏洞）：请使用 GitHub 的 [Private Vulnerability Reporting](https://github.com/heiketu/PeoplesLLM/security/advisories/new) 私下报告，**不要**先开公开 issue。
- 一般问题：直接开 issue 即可。

### 部署注意事项（重要）

- `llama-server` 默认**没有认证**。本项目作者的启动脚本将服务绑定到 `0.0.0.0`——请勿将端口直接暴露到公网或不受信任的网络，建议仅绑定 `127.0.0.1` 或放在防火墙/反向代理之后。
- GGUF 模型文件请从可信来源获取；加载来历不明或被第三方修改（字节级补丁）的模型文件存在风险。
- 本项目包含实验性的跨机 EP 组件（`tools/epd`），其传输层**未做加密与认证**，仅可在受信内网/点对点直连环境使用。

---

## English

### Scope

Only the `main` branch receives security fixes. This is a personal tuning fork; the author only maintains the configurations they actually use.

### Reporting a vulnerability

- **Sensitive issues** (e.g. remotely exploitable bugs): please use GitHub [Private Vulnerability Reporting](https://github.com/heiketu/PeoplesLLM/security/advisories/new) — do **not** open a public issue first.
- Everything else: a regular issue is fine.

### Deployment notes (important)

- `llama-server` has **no authentication** by default. The author's own launch scripts bind to `0.0.0.0` — do not expose the port to the public internet or untrusted networks; bind to `127.0.0.1` or put it behind a firewall/reverse proxy.
- Obtain GGUF model files from trusted sources; loading untrusted or third-party-modified (byte-patched) model files carries risk.
- This fork includes experimental cross-machine EP components (`tools/epd`) whose transport has **no encryption or authentication** — use only on trusted private networks or direct point-to-point links.
