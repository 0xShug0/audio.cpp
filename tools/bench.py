#!/usr/bin/env python3
import os
import sys
import argparse
import subprocess
import json
from pathlib import Path

FAMILIES = [
    "ace_step", "chatterbox", "citrinet_asr", "heartmula", "higgs_audio_stt", "higgs_tts",
    "htdemucs", "hviske_asr", "index_tts2", "irodori_tts", "kokoro_tts", "marblenet_vad",
    "mel_band_roformer", "miocodec", "miotts", "moss_tts_local", "moss_tts_nano", "nemotron_asr",
    "omnivoice", "parakeet_tdt", "pocket_tts", "qwen3_asr", "qwen3_forced_aligner", "qwen3_tts",
    "seed_vc", "silero_vad", "sortformer_diar", "stable_audio", "supertonic", "vevo2", "vibevoice",
    "vibevoice_asr", "voxcpm2", "voxtral_realtime"
]

BACKENDS = ["vulkan", "cuda", "cpu", "metal"]
PLATFORMS = ["linux", "metal", "windows"]

def get_cli_bin(platform: str, backend: str) -> str:
    exe = "audiocpp_cli.exe" if platform == "windows" else "audiocpp_cli"
    if platform == "linux":
        return f"./build/linux-{backend}-release/bin/{exe}"
    elif platform == "metal":
        return f"./build/macos-metal-release/bin/{exe}"
    elif platform == "windows":
        return f"./build/windows-{backend}-release/bin/Release/{exe}" # Visual Studio usually adds Release/
    return f"./build/{exe}"

def generate_markdown_report(summary_path: str):
    path = Path(summary_path)
    if not path.exists():
        return
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    scenarios = data.get("scenarios", [])
    if not scenarios:
        return
    for scenario in scenarios:
        family = scenario.get("family", "unknown")
        backend = scenario.get("backend", "unknown")
        baseline = scenario.get("baseline", {})
        cpp = scenario.get("cpp", {})
        if not baseline or not cpp:
            continue
        print(f"\n# 📊 Benchmark Report: `{family}` ({backend})")
        print(f"\n### 🧪 Validation Status")
        print("| Test | Status | Details |")
        print("|---|---|---|")
        baseline_valid = baseline.get("valid", {}).get("ok", False)
        cpp_valid = cpp.get("valid", {}).get("ok", False)
        print(f"| Baseline Output Valid | {'✅ Pass' if baseline_valid else '❌ Fail'} | |")
        print(f"| New Output Valid | {'✅ Pass' if cpp_valid else '❌ Fail'} | |")
        
        for p in scenario.get("parity", []):
            req_idx = p.get("request_index", "?")
            p_ok = p.get("ok", False)
            reason = p.get("reason", "")
            icon = '✅ Pass' if p_ok else '❌ Fail'
            print(f"| Parity (Request {req_idx}) | {icon} | {reason} |")

        print(f"\n### ⏱️ Performance Metrics")
        print("| Metric | Baseline | New | Diff | % Change |")
        print("|---|---|---|---|---|")
        
        baseline_metrics = baseline.get("parsed", {}).get("metrics", {})
        cpp_metrics = cpp.get("parsed", {}).get("metrics", {})
        
        for key in sorted(set(baseline_metrics.keys()) | set(cpp_metrics.keys())):
            base_val = baseline_metrics.get(key)
            cpp_val = cpp_metrics.get(key)
            base_str = f"{base_val:.2f}" if base_val is not None else "N/A"
            cpp_str = f"{cpp_val:.2f}" if cpp_val is not None else "N/A"
            diff_str = "N/A"
            pct_str = "N/A"
            if base_val is not None and cpp_val is not None and base_val != 0:
                diff = cpp_val - base_val
                pct = (diff / base_val) * 100
                diff_str = f"{diff:+.2f}"
                pct_str = f"{pct:+.2f}%"
                
                # Highlight in green if improved (decreased), red if worsened
                if diff < 0:
                    pct_str = f"🟢 {pct_str}"
                elif diff > 0:
                    pct_str = f"🔴 {pct_str}"
                    
            print(f"| `{key}` | {base_str} | {cpp_str} | {diff_str} | {pct_str} |")
        print("\n")

def run_setup(args):
    script_ext = ".ps1" if args.platform == "windows" else ".sh"
    cmd = [f"scripts/build_{args.platform}{script_ext}"]
    if args.platform in ("linux", "windows"):
        cmd.extend(["--backend", args.backend])
    
    # Optimize build speed by setting --jobs automatically
    cpu_cores = os.cpu_count() or 4
    optimal_jobs = max(1, cpu_cores - 1)
    cmd.extend(["--with-warmbench", "--jobs", str(optimal_jobs)])

    print(f"🔧 Compiling warmbench binaries for {args.platform} ({args.backend})...")
    print(f"🚀 Running: {' '.join(cmd)}")
    if args.platform == "windows":
        subprocess.run(["powershell", "-ExecutionPolicy", "Bypass", "-File"] + cmd, check=True)
    else:
        subprocess.run(["bash"] + cmd, check=True)
    print("✅ Compilation complete!")

def run_path(args):
    cli_bin = get_cli_bin(args.platform, args.backend)
    cmd = [
        "python3", "tools/audiocpp_cli/run_audiocpp_cli_path_tests.py",
        "--models-root", "models",
        "--backend", args.backend,
        "--out", "out",
        "--log",
        "--audiocpp-cli-bin", cli_bin
    ]
    if args.family:
        cmd.extend(["--family", args.family])
    
    print(f"🚀 Running Path Tests: {' '.join(cmd)}")
    if not Path(cli_bin).exists():
        print(f"⚠️  WARNING: {cli_bin} not found. Did you run 'tools/bench.py setup' first?")
    subprocess.run(cmd, check=False)

def run_warmbench(args):
    cli_bin = get_cli_bin(args.platform, args.backend)
    cmd = [
        "python3", "tests/warmbench.py",
        "--family", args.family,
        "--backend", args.backend,
        "--output-dir", "out",
        "--audiocpp-cli-bin", cli_bin,
    ]
    if args.baseline_bin:
        cmd.extend(["--baseline-audiocpp-cli-bin", args.baseline_bin])
    else:
        cmd.append("--skip-python")
    print(f"🚀 Running Warmbench: {' '.join(cmd)}")
    if not Path(cli_bin).exists():
        print(f"⚠️  WARNING: {cli_bin} not found. Did you run 'tools/bench.py setup' first?")
    subprocess.run(cmd, check=False)
    if args.baseline_bin:
        generate_markdown_report("out/summary.json")

def run_report(args):
    generate_markdown_report(args.summary_json)

def main():
    parser = argparse.ArgumentParser(description="audio.cpp benchmarking CLI wrapper")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Setup command
    setup_parser = subparsers.add_parser("setup", help="Build the warmbench testing binaries for your platform")
    setup_parser.add_argument("--platform", choices=PLATFORMS, default="linux", help="Target OS platform")
    setup_parser.add_argument("--backend", choices=BACKENDS, default="vulkan", help="Backend to compile for")

    # Path tests command
    path_parser = subparsers.add_parser("path", help="Run Path Tests (CLI)")
    path_parser.add_argument("--family", choices=FAMILIES, help="Specific model family to test")
    path_parser.add_argument("--platform", choices=PLATFORMS, default="linux", help="Target OS platform")
    path_parser.add_argument("--backend", choices=BACKENDS, default="vulkan", help="Backend to use")

    # Warmbench command
    warm_parser = subparsers.add_parser("warmbench", help="Run Throughput Benchmarks (Warmbench)")
    warm_parser.add_argument("--family", choices=FAMILIES, required=True, help="Specific model family to test")
    warm_parser.add_argument("--platform", choices=PLATFORMS, default="linux", help="Target OS platform")
    warm_parser.add_argument("--backend", choices=BACKENDS, default="vulkan", help="Backend to use")
    warm_parser.add_argument("--baseline-bin", default="", help="Optional path to a baseline audiocpp_cli binary for A/B testing")

    # Report command
    report_parser = subparsers.add_parser("report", help="Generate a Markdown report from a summary.json file")
    report_parser.add_argument("--summary-json", default="out/summary.json", help="Path to the summary.json file")

    args = parser.parse_args()

    if args.command == "setup":
        run_setup(args)
    elif args.command == "path":
        run_path(args)
    elif args.command == "warmbench":
        run_warmbench(args)
    elif args.command == "report":
        run_report(args)

if __name__ == "__main__":
    main()
