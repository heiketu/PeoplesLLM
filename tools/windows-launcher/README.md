# PeoplesLLM Windows launcher

This is a small Windows GUI for starting this fork's `llama-server`. It uses Windows PowerShell 5.1, WinForms and the .NET libraries already included with Windows. It does not use Python, Node.js, Electron, package managers or a shell command evaluator.

## Start

Build or download a Windows `llama-server.exe`, then run:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\tools\windows-launcher\PeoplesLLM-Launcher.ps1
```

You can also double-click `Launch.cmd`.

Relative paths are resolved from the current working directory. `Launch.cmd` changes to the repository root first; the documented PowerShell command should also be run from the repository root. Paths saved by the GUI are normally absolute because the file pickers return absolute paths.

If local policy permits signed or local scripts, omit `-ExecutionPolicy Bypass`. The launcher changes policy only for this process; it does not modify the machine policy.

Choose the server executable and main GGUF. A DSpark GGUF is optional. Press **Refresh preview** to inspect the exact environment and argv, then **Start server**. **Stop server** terminates the directly launched server process. Closing the window also stops it.

The status line polls `GET /health`. The chat panel sends non-streaming `POST /v1/chat/completions` requests and keeps an in-memory conversation until **Reset chat** is pressed.

## Profiles and dry-run

Profiles are ordinary JSON. **Load profile** and **Save profile** round-trip all visible settings. Newer fields receive safe defaults when an older profile omits them.

To print the exact command without opening WinForms or starting a process:

```powershell
powershell.exe -NoProfile -File .\tools\windows-launcher\PeoplesLLM-Launcher.ps1 `
  -Profile .\tools\windows-launcher\sample-profile.json -DryRun
```

The preview uses the same argv builder and Windows quoting routine as process launch. `ENV` and `COMMAND` are descriptive prefixes; the preview text itself is never executed. Paths are never interpolated into `cmd.exe`, `Invoke-Expression` or a PowerShell command string. **Extra arguments** accepts one complete argv token per line; do not add shell quotes. For example, enter `--rope-freq-base` on one line and `1000000` on the next. A flag such as `--no-context-shift` needs only one line. **Extra environment** accepts one `KEY=VALUE` per line and rejects malformed or duplicate generated keys.

Run the PowerShell selftest on Windows:

```powershell
powershell.exe -NoProfile -File .\tools\windows-launcher\launcher-selftest.ps1
```

## Parameter mapping

| GUI field | llama-server setting |
|---|---|
| Main model, alias | `--model`, `--alias` |
| Host, port | `--host`, `--port` |
| Context, slots | `--ctx-size`, `--parallel` |
| Threads, batch threads | `--threads`, `--threads-batch` |
| Batch, ubatch | `--batch-size`, `--ubatch-size` |
| Device, GPU layers | `--device`, `--n-gpu-layers` |
| CPU MoE layers | `--n-cpu-moe` |
| Split mode, tensor split, main GPU | `--split-mode`, `--tensor-split`, `--main-gpu` |
| NUMA, mirror components | `--numa`, `--numa-mirror` |
| Load and KV types | `--load-mode`, `--cache-type-k`, `--cache-type-v` |
| DSpark path | `--model-draft PATH --spec-type draft-dspark` |
| DSpark tuning | `--spec-draft-n-max`, `--spec-draft-p-min`, `--spec-draft-device`, `--spec-draft-ngl` |
| CPU NUMA EP | `GGML_NUMA_EP=1`, optionally `GGML_NUMA_HIER_BARRIER=1` and `GGML_NUMA_EP_GATE_UP_PARALLEL=1` |
| Scheduled remote EP | `GGML_REMOTE_EP*` and `GGML_REMOTE_EP_SCHED*` environment variables |

`--ctx-size` is the server context setting. The effective `n_ctx_slot` is printed by `llama-server` during startup; verify that line when configuring several large slots because KV allocation depends on the binary, unified-KV mode and slot count.

## EP and platform boundaries

The launcher configures only the master `llama-server`. It does not start `llama-epd` workers. Start workers first and verify that their model, layers, expert ownership, protocol and transport match the master.

NUMA EP, remote EP and RDMA availability depend on how the Windows binary was built and on backend/platform support. In particular, `numactl` is a Linux strategy and RDMA requires the fork's RDMA transport and driver stack. A GUI checkbox cannot make an unsupported build feature available. Read `docs/QUICKSTART.md` and `docs/PARAMETERS.md` before enabling pure EP (`K local = 0`) or max-effort replicas.

Binding to `0.0.0.0` exposes the HTTP API on all interfaces. The launcher uses `127.0.0.1` only for its own health/chat requests in that case. Configure Windows Firewall and API authentication or a trusted reverse proxy before exposing the port outside a trusted network.

## Validation status

The repository includes a dependency-free Linux-side structural test:

```bash
python3 tools/windows-launcher/launcher-static-test.py
```

It checks JSON shape, balanced PowerShell delimiters, required parameter mappings, process redirection/cleanup markers and the absence of shell evaluation. This implementation is not Windows-tested in the current Linux workspace. Run `launcher-selftest.ps1`, inspect the dry-run output, and do a local small-model start/stop/chat smoke test on Windows before treating it as production-ready.
