#!/usr/bin/env python3
"""Audit documented environment variables owned by the x-llama.cpp fork."""

from __future__ import annotations

import argparse
import dataclasses
import json
import re
import sys
import tempfile
from pathlib import Path


SOURCE_GLOBS = (
    "ggml/src/ggml-cpu/ggml-cpu.c",
    "ggml/src/ggml-cpu/repack.cpp",
    "ggml/src/ggml-cpu/xllama-*.cpp",
    "ggml/src/ggml-cpu/arch/x86/repack.cpp",
    "ggml/src/ggml-cpu/ops.cpp",
    "ggml/src/ggml-cpu/binary-ops.cpp",
    "ggml/src/ggml-backend.cpp",
    "ggml/src/ggml-backend-meta.cpp",
    "ggml/src/ggml-cuda/**/*.cu",
    "ggml/src/ggml-cuda/**/*.cuh",
    "ggml/src/ggml-cuda/**/*.cpp",
    "src/llama-context.cpp",
    "src/llama-graph.cpp",
    "src/llama-model.cpp",
    "src/llama-repack-stream-bake.cpp",
    "src/llama-remote-ep.cpp",
    "src/llama-layer-major.cpp",
    "src/llama-hot-expert.cpp",
    "src/llama-kv-cache-dsv4.cpp",
    "src/models/deepseek4.cpp",
    "common/speculative.cpp",
    "tools/epd/*.cpp",
    "tools/epd/*.h",
    "tools/server/server-context.cpp",
)

AUDITED_PREFIXES = (
    "GGML_NUMA_",
    "GGML_REPACK_",
    "GGML_CPU_",
    "GGML_MOE_",
    "GGML_REMOTE_EP_",
    "GGML_EP_",
    "GGML_EPD_",
    "GGML_E4A_",
    "GGML_STREAM_BAKE",
    "GGML_HOT_EXPERT",
    "GGML_SCHED_",
    "GGML_CUDA_MOE_",
    "GGML_CUDA_HOT_",
    "GGML_CUDA_MXFP4_",
    "GGML_CUDA_DSV4_",
    "GGML_CUDA_AR_",
    "GGML_CUDA_GRAPH_",
    "LLAMA_LAYER_MAJOR_",
    "LLAMA_GPU_PREFILL_",
    "LLAMA_DSV4_",
    "LLAMA_DSPARK_",
    "LLAMA_SPC_",
    "LLAMA_SERVER_",
)

AUDITED_EXACT = frozenset(
    {
        "GGML_OP_TIMING",
        "GGML_MM_PHASE",
        "GGML_COPY_TRACE",
        "GGML_OFFLOAD_TRACE",
        "GGML_OP_OFFLOAD_MIN_BATCH",
        "GGML_CUDA_ALLREDUCE",
        "GGML_CUDA_P2P",
        "GGML_CUDA_BATCHED_TOPK",
        "GGML_CUDA_TOPK_OVERLAP_PROFILE",
        "GGML_CUDA_MMQ_MOE_J",
        "GGML_CUDA_FA_PV_Q8",
        "LLAMA_DECODE_TIMING",
        "LLAMA_NAN_DEBUG",
        "LLAMA_TRACE",
    }
)

# These are upstream/common parser interfaces documented for deployment context.
# Their reads are dynamic or outside the fork module boundary above.
DOCUMENTED_EXCLUSIONS = {
    "GGML_EPD_RDMA": "negative documentation for a removed/nonexistent alias",
    "GGML_RPC_DEBUG": "upstream RPC backend",
}
DOCUMENTED_EXCLUDED_PREFIXES = {
    "LLAMA_ARG_": "common CLI env mapping is generated dynamically",
}

ENV_NAME_RE = re.compile(r"(?:GGML|LLAMA)_[A-Z0-9_]+")
CALL_RE = re.compile(
    r"\b(?P<function>[A-Za-z_]\w*(?:::\w+)*)\s*\(\s*\"(?P<name>(?:GGML|LLAMA)_[A-Z0-9_]+)\""
)
TABLE_RE = re.compile(r"\{\s*\"(?P<name>(?:GGML|LLAMA)_[A-Z0-9_]+)\"\s*,")
DYNAMIC_RE = re.compile(r"\b(?:(?:std::)?getenv)\s*\(\s*(?!\")(?P<argument>[^)\n]+)\)")
BACKTICK_RE = re.compile(r"`([^`]+)`")


@dataclasses.dataclass(frozen=True, order=True)
class ReadPoint:
    path: str
    line: int
    column: int


@dataclasses.dataclass(frozen=True, order=True)
class ExcludedDynamic:
    path: str
    line: int
    expression: str


def is_audited(name: str) -> bool:
    return name in AUDITED_EXACT or any(name.startswith(prefix) for prefix in AUDITED_PREFIXES)


def documented_exclusion(name: str) -> str | None:
    if name in DOCUMENTED_EXCLUSIONS:
        return DOCUMENTED_EXCLUSIONS[name]
    for prefix, reason in DOCUMENTED_EXCLUDED_PREFIXES.items():
        if name.startswith(prefix):
            return reason
    return None


def line_column(text: str, offset: int) -> tuple[int, int]:
    line = text.count("\n", 0, offset) + 1
    line_start = text.rfind("\n", 0, offset) + 1
    return line, offset - line_start + 1


def extract_reads(path: str, text: str) -> tuple[dict[str, list[ReadPoint]], list[ExcludedDynamic]]:
    reads: dict[str, list[ReadPoint]] = {}
    seen: set[tuple[str, int, int]] = set()
    for pattern in (CALL_RE, TABLE_RE):
        for match in pattern.finditer(text):
            if pattern is CALL_RE and "env" not in match.group("function").lower():
                continue
            name = match.group("name")
            if not is_audited(name):
                continue
            line, column = line_column(text, match.start("name"))
            key = (name, line, column)
            if key in seen:
                continue
            seen.add(key)
            reads.setdefault(name, []).append(ReadPoint(path, line, column))

    dynamic: list[ExcludedDynamic] = []
    for match in DYNAMIC_RE.finditer(text):
        line, _ = line_column(text, match.start())
        dynamic.append(ExcludedDynamic(path, line, match.group("argument").strip()))
    return reads, dynamic


def extract_documented(text: str) -> set[str]:
    documented: set[str] = set()
    for line in text.splitlines():
        if not line.lstrip().startswith("|"):
            continue
        for token in BACKTICK_RE.findall(line):
            if ENV_NAME_RE.fullmatch(token) and (is_audited(token) or documented_exclusion(token) is not None):
                documented.add(token)
    return documented


def resolve_sources(root: Path) -> list[Path]:
    files: set[Path] = set()
    for pattern in SOURCE_GLOBS:
        files.update(path for path in root.glob(pattern) if path.is_file())
    return sorted(files)


def audit_texts(source_texts: dict[str, str], docs_text: str) -> dict:
    reads: dict[str, list[ReadPoint]] = {}
    excluded_dynamic: list[ExcludedDynamic] = []
    for path, text in sorted(source_texts.items()):
        file_reads, file_dynamic = extract_reads(path, text)
        for name, points in file_reads.items():
            reads.setdefault(name, []).extend(points)
        excluded_dynamic.extend(file_dynamic)

    documented = extract_documented(docs_text)
    source_names = set(reads)
    ignored_documented = {
        name: reason for name in documented if (reason := documented_exclusion(name)) is not None
    }
    comparable_documented = documented - set(ignored_documented)

    missing_doc = sorted(source_names - comparable_documented)
    documented_but_no_read = sorted(comparable_documented - source_names)
    duplicate_read_points = {
        name: sorted(points) for name, points in reads.items() if len(set(points)) > 1
    }
    return {
        "source_env_count": len(source_names),
        "documented_env_count": len(comparable_documented),
        "missing_doc": missing_doc,
        "documented_but_no_read": documented_but_no_read,
        "duplicate_read_points": duplicate_read_points,
        "read_points": {name: sorted(points) for name, points in sorted(reads.items())},
        "excluded_dynamic": sorted(excluded_dynamic),
        "ignored_documented": ignored_documented,
    }


def serializable(report: dict, root: Path, source_files: list[Path]) -> dict:
    def point(point: ReadPoint) -> dict:
        return dataclasses.asdict(point)

    result = {
        "version": 1,
        "root": str(root),
        "source_files": [str(path.relative_to(root)) for path in source_files],
        "source_env_count": report["source_env_count"],
        "documented_env_count": report["documented_env_count"],
        "missing_doc": [
            {"name": name, "read_points": [point(item) for item in report["read_points"][name]]}
            for name in report["missing_doc"]
        ],
        "documented_but_no_read": report["documented_but_no_read"],
        "duplicate_read_points": [
            {"name": name, "read_points": [point(item) for item in points]}
            for name, points in sorted(report["duplicate_read_points"].items())
        ],
        "excluded_dynamic": [dataclasses.asdict(item) for item in report["excluded_dynamic"]],
        "ignored_documented": report["ignored_documented"],
    }
    result["ok"] = not result["missing_doc"] and not result["documented_but_no_read"]
    return result


def print_text(report: dict, verbose: bool) -> None:
    print(
        f"fork env audit: source={report['source_env_count']} documented={report['documented_env_count']} "
        f"missing={len(report['missing_doc'])} stale={len(report['documented_but_no_read'])} "
        f"duplicates={len(report['duplicate_read_points'])}"
    )
    for item in report["missing_doc"]:
        locations = ", ".join(f"{point['path']}:{point['line']}" for point in item["read_points"])
        print(f"MISSING_DOC {item['name']} {locations}")
    for name in report["documented_but_no_read"]:
        print(f"NO_READ {name}")
    for item in report["duplicate_read_points"]:
        locations = ", ".join(f"{point['path']}:{point['line']}" for point in item["read_points"])
        print(f"DUPLICATE_READ {item['name']} {locations}")
    if verbose:
        for item in report["excluded_dynamic"]:
            print(f"EXCLUDED_DYNAMIC {item['path']}:{item['line']} {item['expression']}")
        for name, reason in sorted(report["ignored_documented"].items()):
            print(f"IGNORED_DOC {name} {reason}")


def selftest() -> None:
    sources = {
        "fixture.cpp": """
            getenv("GGML_NUMA_ONE");
            env_int("GGML_NUMA_DUP", 0);
            std::getenv("GGML_NUMA_DUP");
            const row vars[] = {{"LLAMA_DSV4_TABLE", 1}};
            getenv(dynamic_name);
            getenv("CUDA_VISIBLE_DEVICES");
        """,
    }
    docs = """
        | Variable | Default |
        |---|---|
        | `GGML_NUMA_ONE` | off |
        | `GGML_NUMA_DUP` | off |
        | `LLAMA_DSV4_STALE` | off |
        | `GGML_RPC_DEBUG` | upstream |
    """
    report = audit_texts(sources, docs)
    assert report["missing_doc"] == ["LLAMA_DSV4_TABLE"]
    assert report["documented_but_no_read"] == ["LLAMA_DSV4_STALE"]
    assert sorted(report["duplicate_read_points"]) == ["GGML_NUMA_DUP"]
    assert len(report["excluded_dynamic"]) == 1
    assert report["ignored_documented"] == {"GGML_RPC_DEBUG": "upstream RPC backend"}

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        output = serializable(report, root, [])
        json.dumps(output, sort_keys=True)
        assert not output["ok"]
    print("fork-env-audit selftest: PASS")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--docs", type=Path)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--fail-duplicates", action="store_true")
    parser.add_argument("--selftest", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.selftest:
        selftest()
        return 0

    root = args.root.resolve()
    docs = args.docs.resolve() if args.docs else root / "docs/PARAMETERS.md"
    source_files = resolve_sources(root)
    if not source_files:
        print("fork env audit: no source files matched the fork scope", file=sys.stderr)
        return 2
    if not docs.is_file():
        print(f"fork env audit: documentation not found: {docs}", file=sys.stderr)
        return 2

    source_texts = {
        str(path.relative_to(root)): path.read_text(encoding="utf-8", errors="replace") for path in source_files
    }
    report = serializable(audit_texts(source_texts, docs.read_text(encoding="utf-8")), root, source_files)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print_text(report, args.verbose)
    failed = not report["ok"] or (args.fail_duplicates and bool(report["duplicate_read_points"]))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
