#!/usr/bin/env python3
"""Linux-side structural checks for the dependency-free Windows launcher."""

from __future__ import annotations

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def strip_powershell(text: str) -> str:
    output: list[str] = []
    state = "code"
    index = 0
    while index < len(text):
        char = text[index]
        if state == "comment":
            if char == "\n":
                state = "code"
                output.append(char)
            else:
                output.append(" ")
        elif state == "single":
            if char == "'" and index + 1 < len(text) and text[index + 1] == "'":
                output.extend((" ", " "))
                index += 1
            elif char == "'":
                state = "code"
                output.append(" ")
            else:
                output.append("\n" if char == "\n" else " ")
        elif state == "double":
            if char == "`" and index + 1 < len(text):
                output.extend((" ", " "))
                index += 1
            elif char == '"':
                state = "code"
                output.append(" ")
            else:
                output.append("\n" if char == "\n" else " ")
        elif char == "#":
            state = "comment"
            output.append(" ")
        elif char == "'":
            state = "single"
            output.append(" ")
        elif char == '"':
            state = "double"
            output.append(" ")
        else:
            output.append(char)
        index += 1
    if state in {"single", "double"}:
        raise AssertionError(f"unterminated PowerShell {state}-quoted string")
    return "".join(output)


def balanced(text: str) -> None:
    clean = strip_powershell(text)
    pairs = {')': '(', ']': '[', '}': '{'}
    stack: list[tuple[str, int]] = []
    for index, char in enumerate(clean):
        if char in '([{':
            stack.append((char, index))
        elif char in pairs:
            if not stack or stack[-1][0] != pairs[char]:
                raise AssertionError(f"delimiter mismatch at byte {index}: {char}")
            stack.pop()
    if stack:
        raise AssertionError(f"unclosed delimiter: {stack[-1]}")


def main() -> None:
    main_text = (ROOT / "PeoplesLLM-Launcher.ps1").read_text()
    module_text = (ROOT / "PeoplesLLM.Launcher.psm1").read_text()
    readme = (ROOT / "README.md").read_text()
    profile = json.loads((ROOT / "sample-profile.json").read_text())
    balanced(main_text)
    balanced(module_text)
    assert profile["Version"] == 1
    assert 1 <= profile["Port"] <= 65535
    assert profile["Slots"] >= 1 and profile["ContextSize"] >= 0
    assert profile["SplitMode"] in {"none", "layer", "row", "tensor"}
    assert profile["Numa"] in {"none", "distribute", "isolate", "numactl", "mirror"}
    assert isinstance(profile["ExtraArguments"], list)
    combined = main_text + module_text
    for forbidden in ("Invoke-Expression", "cmd.exe /c", "UseShellExecute = $true"):
        assert forbidden not in combined
    for marker in (
        "Quote-WindowsArgument", "RedirectStandardOutput", "RedirectStandardError",
        "--model-draft", "draft-dspark", "--ctx-size", "--parallel", "--device",
        "--n-gpu-layers", "--n-cpu-moe", "--numa", "GGML_NUMA_EP",
        "GGML_REMOTE_EP_SCHED_ENDPOINTS", "/health", "/v1/chat/completions",
        "ConvertTo-Json", "ConvertFrom-Json", "DryRun", "Stop-ServerProcess",
    ):
        assert marker in combined, marker
    for marker in ("Windows PowerShell 5.1", "dry-run", "one complete argv token per line", "not Windows-tested"):
        assert marker in readme, marker
    print("windows launcher static test: PASS")


if __name__ == "__main__":
    main()
