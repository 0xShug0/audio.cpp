#!/usr/bin/env python3
"""Quantize XTTS v2 GPT GGUF to Q8_0 or Q4_0 using llama.cpp's llama-quantize.

The vocoder GGUF is small (~49MB F16) and is kept as-is. Only the GPT GGUF
(~1.7GB F16/F32) is quantized, achieving 3-6x size reduction.

Usage:
    python quantize_xtts_v2.py --model-dir /path/to/gguf --output-dir /path/to/output [--type q8_0|q4_0]

Example:
    python quantize_xtts_v2.py --model-dir /data4/work/TTS/weights/gguf --output-dir /kv/quantized
    python quantize_xtts_v2.py --model-dir /data4/work/TTS/weights/gguf --output-dir /kv/quantized --type q4_0

Size reduction:
    F16/F32: 1.7GB GPT + 49MB vocoder = ~1.75GB
    Q8_0:    469MB GPT + 49MB vocoder = ~518MB  (3.4x smaller)
    Q4_0:    277MB GPT + 49MB vocoder = ~326MB  (5.4x smaller)
"""

import argparse, sys, os, subprocess, shutil


def find_quantize():
    """Find llama-quantize binary."""
    candidates = [
        "llama-quantize",
        "/home/jani/llama.cpp/build/bin/llama-quantize",
        "/home/jani/lcpp-glm/llama-quantize",
    ]
    for c in candidates:
        if os.path.isfile(c):
            return c
    # Search in common build dirs
    for root in ["/home/jani", "/opt"]:
        for dp, dns, fns in os.walk(root):
            if "llama.cpp" in dp or "llama" in dp:
                if "llama-quantize" in fns:
                    return os.path.join(dp, "llama-quantize")
    return None


def quantize_gpt(gpt_path, output_path, target_type, nthread=16):
    """Quantize GPT GGUF using llama-quantize with architecture override."""
    print(f"\n=== Quantizing GPT: {os.path.basename(gpt_path)} -> {os.path.basename(output_path)} ({target_type}) ===")

    quantize = find_quantize()
    if not quantize:
        sys.exit("ERROR: llama-quantize not found. Build llama.cpp or set the path.")

    # Override architecture to gpt2 and add required metadata fields
    # so llama-quantize recognizes the model
    kv_overrides = [
        "--override-kv", "general.architecture=str:gpt2",
        "--override-kv", "gpt2.context_length=int:2048",
        "--override-kv", "gpt2.embedding_length=int:1024",
        "--override-kv", "gpt2.block_count=int:30",
        "--override-kv", "gpt2.feed_forward_length=int:4096",
        "--override-kv", "gpt2.attention.head_count=int:16",
        "--override-kv", "gpt2.attention.layer_norm_epsilon=float:0.00001",
    ]

    cmd = [quantize] + kv_overrides + [gpt_path, output_path, target_type, str(nthread)]
    print(f"  Command: {' '.join(cmd)}")

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"  ERROR: quantization failed")
        print(result.stderr[-1000:])
        sys.exit(1)

    # Print summary
    for line in result.stderr.split('\n'):
        if 'model size' in line or 'quant size' in line or 'WARNING' in line:
            print(f"  {line.strip()}")

    # Verify output
    if not os.path.exists(output_path):
        sys.exit(f"ERROR: output file not created: {output_path}")

    orig_size = os.path.getsize(gpt_path) / 1e6
    new_size = os.path.getsize(output_path) / 1e6
    print(f"  Size: {orig_size:.0f} MB -> {new_size:.0f} MB ({new_size/orig_size*100:.0f}%)")


def copy_vocoder(voc_path, output_dir):
    """Copy vocoder GGUF as-is (F16, ~49MB)."""
    if not voc_path or not os.path.exists(voc_path):
        print(f"\n  Vocoder not found at {voc_path}, skipping.")
        return
    dest = os.path.join(output_dir, os.path.basename(voc_path))
    shutil.copy2(voc_path, dest)
    sz = os.path.getsize(dest) / 1e6
    print(f"\n=== Copied vocoder: {os.path.basename(voc_path)} ({sz:.0f} MB, kept F16) ===")


def main():
    ap = argparse.ArgumentParser(description="Quantize XTTS v2 GPT GGUF")
    ap.add_argument("--model-dir", required=True, help="Directory containing GGUF files")
    ap.add_argument("--output-dir", required=True, help="Directory to write quantized GGUF files")
    ap.add_argument("--type", default="q8_0", choices=["q8_0", "q4_0"],
                    help="Quantization type (default: q8_0)")
    ap.add_argument("--threads", type=int, default=16, help="Number of threads (default: 16)")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Find GGUF files
    gpt_path = os.path.join(args.model_dir, "xtts-v2-gpt.gguf")
    voc_path = os.path.join(args.model_dir, "xtts-v2-vocoder-f16.gguf")
    if not os.path.exists(voc_path):
        voc_path = os.path.join(args.model_dir, "xtts-v2-vocoder.gguf")

    if not os.path.exists(gpt_path):
        sys.exit(f"ERROR: GPT GGUF not found: {gpt_path}")

    # Quantize GPT
    gpt_out = os.path.join(args.output_dir, f"xtts-v2-gpt-{args.type}.gguf")
    quantize_gpt(gpt_path, gpt_out, args.type, args.threads)

    # Copy vocoder (kept as F16)
    copy_vocoder(voc_path, args.output_dir)

    # Copy mel_norms if present
    mel_path = os.path.join(args.model_dir, "mel_norms.bin")
    if os.path.exists(mel_path):
        shutil.copy2(mel_path, args.output_dir)
        print(f"\n  Copied mel_norms.bin")

    print("\nDone.")
    print(f"  Output directory: {args.output_dir}")
    total = sum(os.path.getsize(os.path.join(args.output_dir, f)) for f in os.listdir(args.output_dir)) / 1e6
    print(f"  Total size: {total:.0f} MB")


if __name__ == "__main__":
    main()
