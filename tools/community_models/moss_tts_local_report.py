#!/usr/bin/env python3
"""Build a PR-ready MOSS-TTS-Local report from a C++ result dir and a Python
reference result dir.

Both dirs must follow the ``audiocpp_cli`` path-test layout (per-case subdirs
with ``outputs/<request_id>.wav`` and a ``stdout.log`` carrying ``[TIMING]``
lines). The C++ dir comes from ``run_audiocpp_cli_path_tests.py``; the Python
dir from ``moss_tts_local_reference.py``.

Reuses the similarity metrics (cosine, log-mel) from
``compare_audiocpp_cli_path_results.py`` so the numbers match that tool exactly.

Usage:
    python moss_tts_local_report.py CPP_DIR PY_DIR [--title "..."] [--out report.md]
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Optional

import compare_audiocpp_cli_path_results as cmp

TIMING_RE = re.compile(r"^\[TIMING[^\]]*\]\s+(\S+)\s+([-+0-9.eE]+)\s*$")
LOG_MEL_RE = re.compile(r"log_mel_cos=([0-9.]+|n/a)")
SR_RE = re.compile(r"sr=(\d+)/(\d+)")


def case_dirs(root: Path) -> dict[str, Path]:
    return {p.name: p for p in sorted(root.iterdir()) if p.is_dir() and (p / "command.json").exists()}


def timings(case_dir: Path) -> dict[str, float]:
    out: dict[str, float] = {}
    log = case_dir / "stdout.log"
    if not log.exists():
        return out
    for line in log.read_text(encoding="utf-8", errors="replace").splitlines():
        m = TIMING_RE.match(line)
        if m:
            out[m.group(1)] = float(m.group(2))
    return out


def request_ids(case: dict) -> list[str]:
    return [r["id"] for r in case.get("requests", [])]


def load_case_meta(cases_files: list[Path]) -> dict[str, dict]:
    meta: dict[str, dict] = {}
    for path in cases_files:
        if not path.exists():
            continue
        for case in json.loads(path.read_text(encoding="utf-8")).get("cases", []):
            meta[case["id"]] = case
    return meta


def parse_detail(detail: str) -> tuple[Optional[str], Optional[str]]:
    log_mel = LOG_MEL_RE.search(detail)
    sr = SR_RE.search(detail)
    sr_txt = f"{sr.group(1)}/{sr.group(2)}" if sr else "?"
    return (log_mel.group(1) if log_mel else None), sr_txt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("cpp_dir", type=Path)
    parser.add_argument("py_dir", type=Path)
    parser.add_argument("--title", default="MOSS-TTS-Local: audio.cpp vs Python reference")
    parser.add_argument("--cases", type=Path, action="append", default=[])
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[2]
    cases_files = args.cases or [
        repo / "tools" / "audiocpp_cli" / "audiocpp_cli_path_cases.json",
        repo / "tools" / "audiocpp_cli" / "audiocpp_cli_longform_tts_clone_cases.json",
    ]
    meta = load_case_meta(cases_files)

    cpp = case_dirs(args.cpp_dir)
    py = case_dirs(args.py_dir)
    shared = [cid for cid in sorted(set(cpp) & set(py))]

    lines: list[str] = [f"# {args.title}", ""]
    lines.append(f"- C++ results: `{args.cpp_dir}`")
    lines.append(f"- Python reference: `{args.py_dir}`")
    lines.append("")

    sim_rows: list[str] = []
    perf_rows: list[str] = []
    mem_rows: list[str] = []

    for cid in shared:
        cpp_dir, py_dir = cpp[cid], py[cid]
        cpp_t, py_t = timings(cpp_dir), timings(py_dir)
        py_mem = {}
        mem_path = py_dir / "memory.json"
        if mem_path.exists():
            py_mem = {r["request_id"]: r for r in json.loads(mem_path.read_text(encoding="utf-8")).get("requests", [])}

        ids = request_ids(meta.get(cid, {})) or sorted(
            p.stem for p in (cpp_dir / "outputs").glob("*.wav")
        )
        for rid in ids:
            cpp_wav = cpp_dir / "outputs" / f"{rid}.wav"
            py_wav = py_dir / "outputs" / f"{rid}.wav"
            if not cpp_wav.exists() or not py_wav.exists():
                sim_rows.append(f"| {cid} | {rid} | missing wav | | |")
                continue
            wav_cos, detail = cmp.wav_similarity_detail(cpp_wav, py_wav)
            log_mel, sr = parse_detail(detail)
            sim_rows.append(f"| {cid} | {rid} | {wav_cos:.6f} | {log_mel} | {sr} |")

            cpp_ms = cpp_t.get(f"request.{rid}.wall_ms")
            py_ms = py_t.get(f"request.{rid}.wall_ms")
            if cpp_ms and py_ms:
                # Fair comparison uses RTF (time per second of audio): greedy can
                # stop at different frame counts per backend, so raw ms is
                # apples-to-oranges. Duration comes straight from each wav.
                cpp_sr, cpp_audio = cmp.read_wav_f32(cpp_wav)
                py_sr, py_audio = cmp.read_wav_f32(py_wav)
                cpp_s = cpp_audio.shape[0] / cpp_sr if cpp_sr else 0.0
                py_s = py_audio.shape[0] / py_sr if py_sr else 0.0
                cpp_rtf = (cpp_ms / 1000.0) / cpp_s if cpp_s else 0.0
                py_rtf = (py_ms / 1000.0) / py_s if py_s else 0.0
                rtf_speedup = py_rtf / cpp_rtf if cpp_rtf else 0.0
                perf_rows.append(
                    f"| {cid} | {rid} | {cpp_s:.2f} | {py_s:.2f} | {cpp_ms:.0f} | {py_ms:.0f} | "
                    f"{cpp_rtf:.3f} | {py_rtf:.3f} | {rtf_speedup:.2f}x |"
                )

            m = py_mem.get(rid)
            if m:
                mem_rows.append(
                    f"| {cid} | {rid} | {m.get('rss_mb', 0):.0f} | "
                    f"{m.get('cuda_alloc_mb', 0):.0f} | {m.get('cuda_peak_mb', 0):.0f} |"
                )

    lines += ["## Similarity (C++ vs Python reference)", "",
              "| case | request | wav_cos | log_mel_cos | sr Hz |",
              "| --- | --- | ---: | ---: | ---: |", *sim_rows, ""]
    lines += ["## Performance (per request, one loaded session; excludes model load)", "",
              "_Greedy can stop at different frame counts per backend, so compare RTF "
              "(wall seconds per second of generated audio), not raw ms. "
              "RTF speedup = Python RTF / C++ RTF (>1 means C++ is faster per audio-second)._",
              "",
              "| case | request | C++ audio s | Py audio s | C++ ms | Py ms | C++ RTF | Py RTF | RTF speedup |",
              "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |", *perf_rows, ""]
    if mem_rows:
        lines += ["## Python reference memory (per request; stable => no growth after warmup)", "",
                  "| case | request | RSS MB | CUDA alloc MB | CUDA peak MB |",
                  "| --- | --- | ---: | ---: | ---: |", *mem_rows, ""]

    report = "\n".join(lines) + "\n"
    if args.out:
        args.out.write_text(report, encoding="utf-8")
        print(f"wrote {args.out}")
    else:
        print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
