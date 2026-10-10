#!/usr/bin/env python3
"""Compare a native Phonon-2 GGUF with the pinned official F32 reference.

Optional real-weight test: numpy, safetensors, gguf, zstandard, torch,
transformers (with ParakeetForTDT), and soundfile. Audio must be mono 16 kHz
WAV. No model downloads are performed. Pass paths to the built CLI and parity
dumper, --source-dir (official HF download), --reference-dir (Parakeet
sidecars), --staging-dir (converted F32 package), and --model (GGUF).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("convert_phonon2", REPO / "tools/community_models/convert_phonon2.py")
converter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(converter)
REFERENCE_FILES = {
    "fermion_container.py": "5cf172e8cf6b313e951c4ef588ca913c7285b0aa3b92de413c3d7ec2b939c0e4",
    "reference_transformers.py": "8481cfc11537e6dc1e40a8296c93aa15f9b7437b1514564caf647f8773c0dcb3",
}
WORD_FORMATTER_SHA256 = "cbd6b08883682167b7ee6cec3a1e770d778243e31807880ea239ef9fdc3d4929"


def load_word_formatter(path: Path):
    # Independent published policy, not a copy of our native formatter.
    if converter.sha256(path) != WORD_FORMATTER_SHA256:
        raise ValueError("expected fermion-research 0.2.11 _speech/segment.py")
    spec = importlib.util.spec_from_file_location("phonon_reference_segment", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.words_from_tokens


def reference_words(generated, vocab: list[str], formatter, audio_seconds: float) -> list:
    if not hasattr(generated, "durations"):
        raise ValueError("Transformers generate() must return actual TDT durations to check timestamps")
    frame = 0
    pieces = []
    for i, (tid, duration) in enumerate(zip(generated.sequences[0].tolist(), generated.durations[0].tolist())):
        if i > 0 and 0 <= tid < len(vocab):
            piece = vocab[tid]
            if not (piece.startswith("<|") and piece.endswith("|>")) and piece not in ("<unk>", "<pad>"):
                pieces.append((piece.replace("\u2581", " "), frame * .08, duration * .08))
        frame += duration
    return formatter(pieces, limit=audio_seconds)


def compare_words(actual: list, expected: list) -> dict:
    segments_match = len(actual) == len(expected) and all(
        a["word"] == b["text"] for a, b in zip(actual, expected))
    shift = max((abs(a[key + "_sample"] / 16000 - b[key])
                 for a, b in zip(actual, expected) for key in ("start", "end")), default=0) if segments_match else None
    return {"word_segmentation_match": segments_match, "max_shift_ms": shift * 1000 if shift is not None else None,
            # Original JSON rounds to milliseconds; allow its print resolution,
            # never a whole 80 ms model frame or an inferred forced alignment.
            "timestamps_match": segments_match and shift <= .001 + 1e-9}


def audit_weights(source: Path, staging: Path, model: Path) -> dict:
    from safetensors.numpy import load_file
    from fermion_container import read_container
    from gguf import GGUFReader, GGMLQuantizationType
    packed, index = read_container(str(source))
    tensors = load_file(str(staging / "model.safetensors"))
    expected_names = set()
    for name, value in packed.items():
        if name.endswith(".num_batches_tracked"):
            continue
        expected_names.add(name)
        expected = value.astype(np.float32)
        if expected.ndim == 2 and re.search(r"\.conv\.pointwise_conv[12]\.weight$", name):
            expected = expected[:, :, None]
        if name not in tensors or not np.array_equal(expected, tensors[name]):
            raise ValueError(f"official/staging weight mismatch: {name}")
    if expected_names != set(tensors):
        raise ValueError("unexpected or missing staged tensors")
    reader = GGUFReader(str(model))
    if set(t.name for t in reader.tensors) != expected_names:
        raise ValueError("unexpected or missing GGUF tensors")
    for tensor in reader.tensors:
        if tensor.tensor_type != GGMLQuantizationType.F32 or not np.array_equal(
                tensor.data.ravel(), tensors[tensor.name].ravel()):
            raise ValueError(f"staging/GGUF weight mismatch: {tensor.name}")
    field = reader.fields["audiocpp.model_spec.json"]
    package_spec = json.loads(bytes(field.parts[field.data[0]]).decode("utf-8"))
    if package_spec["languages"] != ["en"] or package_spec["modes"] != ["offline"]:
        raise ValueError("GGUF does not declare English-only offline support")
    return {"official_records": len(index), "inference_tensors": len(tensors),
            "official_to_staging_exact": True, "staging_to_gguf_exact": True}


def run_native(command: list[str], log: Path) -> str:
    kwargs = {"creationflags": subprocess.CREATE_NO_WINDOW} if sys.platform == "win32" else {}
    result = subprocess.run(command, cwd=REPO, capture_output=True, text=True,
                            encoding="utf-8", errors="replace", **kwargs)
    log.write_text(result.stdout + "\nSTDERR:\n" + result.stderr, encoding="utf-8")
    if result.returncode:
        raise RuntimeError(f"native command failed ({result.returncode}); see {log}")
    return result.stdout


def main() -> int:
    import soundfile as sf
    import torch
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ("source-dir", "reference-dir", "staging-dir", "model", "cli", "parity-dumper", "output-dir"):
        ap.add_argument("--" + name, type=Path, required=True)
    ap.add_argument("--backend", choices=["cpu", "cuda", "vulkan"], required=True)
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--audio", nargs="+", type=Path, required=True)
    ap.add_argument("--reference-formatter", type=Path,
                    help="fermion-research 0.2.11 _speech/segment.py; also compare original word timestamps")
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, expected_hash in REFERENCE_FILES.items():
        if converter.sha256(args.source_dir / name) != expected_hash:
            raise ValueError(f"reference code differs from pinned official revision: {name}")
    sys.path.insert(0, str(args.source_dir.resolve()))
    from reference_transformers import load_model
    torch.set_num_threads(args.threads)
    formatter = load_word_formatter(args.reference_formatter) if args.reference_formatter else None
    report = {"backend": args.backend, "device": args.device,
              "model_sha256": converter.sha256(args.model), "rows": [],
              "word_timestamps_checked": formatter is not None,
              "validation_scope": "weights, text, encoder and word timestamps" if formatter else "weights, text and encoder only"}
    if formatter:
        report["word_formatter_sha256"] = converter.sha256(args.reference_formatter)
    with tempfile.TemporaryDirectory(prefix="phonon2-reference-") as tmp:
        tmp = Path(tmp)
        converter.extract_archive(args.source_dir, tmp)
        vocab = json.loads((tmp / "config.json").read_text(encoding="utf-8"))["joint"]["vocabulary"]
        report["weights"] = audit_weights(tmp / "model.fermion", args.staging_dir, args.model)
        model, processor, report["reference_load"] = load_model(
            str(tmp / "model.fermion"), str(args.reference_dir.resolve()), dtype=torch.float32)
        for i, audio in enumerate(args.audio):
            wave, sr = sf.read(audio, dtype="float32", always_2d=True)
            if sr != 16000 or wave.shape[1] != 1:
                raise ValueError(f"expected mono 16 kHz WAV: {audio}")
            inputs = processor([wave[:, 0]], sampling_rate=sr, return_tensors="pt", padding=True)
            with torch.inference_mode():
                generated = model.generate(**inputs)
            sequences = getattr(generated, "sequences", generated)
            text = processor.batch_decode(sequences, skip_special_tokens=True)[0].strip()
            stdout = run_native([str(args.cli.resolve()), "--model", str(args.model.resolve()),
                "--family", "parakeet_tdt", "--task", "asr", "--audio", str(audio.resolve()),
                "--backend", args.backend, "--device", str(args.device), "--threads", str(args.threads)],
                args.output_dir / f"cli-{i}.log")
            native = next((line[len("text_output="):] for line in stdout.splitlines()
                           if line.startswith("text_output=")), "")
            row = {"audio": str(audio.resolve()), "reference_text": text, "native_text": native,
                   "exact_text_match": native == text}
            if formatter:
                expected = reference_words(generated, vocab, formatter, wave.shape[0] / sr)
                words_json = next((line[len("word_timestamps="):] for line in stdout.splitlines()
                                   if line.startswith("word_timestamps=")), None)
                # Silence may omit this line; speech must never silently pass
                # the timestamp check with no words.
                actual = json.loads(words_json) if words_json else []
                row.update(reference_words=expected, native_words=actual, **compare_words(actual, expected))
            report["rows"].append(row)
            print(json.dumps(row, ensure_ascii=True), flush=True)
            # One encoder comparison on the first supplied speech clip keeps
            # this optional validation bounded; every clip checks decoded text.
            if i == 0:
                with torch.inference_mode():
                    reference_encoder = model.encoder(**inputs).last_hidden_state[0].numpy()
                dump = args.output_dir / "encoder"
                run_native([str(args.parity_dumper.resolve()), "--model", str(args.staging_dir.resolve()),
                    "--audio", str(audio.resolve()), "--output-dir", str(dump.resolve()),
                    "--backend", args.backend, "--device", str(args.device), "--encoder-only", "1"],
                    args.output_dir / "encoder.log")
                actual = np.load(dump / "enc_out.npy")
                if actual.shape != reference_encoder.shape:
                    raise ValueError("reference/native encoder shapes differ")
                error = actual.astype(np.float64) - reference_encoder.astype(np.float64)
                relative_l2 = float(np.linalg.norm(error) / max(np.linalg.norm(reference_encoder), 1e-20))
                # GPU kernels may round inputs despite F32 weight storage and
                # accumulation. These are explicit numerical smoke-test bounds,
                # accompanied by exact transcript checks, not WER certification.
                bound = {"cpu": 1e-4, "cuda": 1e-3, "vulkan": 5e-3}[args.backend]
                report["encoder"] = {"relative_l2": relative_l2, "relative_l2_limit": bound,
                    "max_abs": float(np.max(np.abs(error))), "finite": bool(np.isfinite(actual).all()),
                    "pass": bool(np.isfinite(actual).all()) and relative_l2 <= bound}
                np.save(args.output_dir / "reference_encoder.npy", reference_encoder)
                np.save(args.output_dir / "reference_mel.npy", inputs["input_features"][0].numpy())
    report["pass"] = all(r["exact_text_match"] for r in report["rows"]) and report["encoder"]["pass"]
    if formatter:
        report["pass"] = report["pass"] and all(r["timestamps_match"] for r in report["rows"])
    (args.output_dir / "validation.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("PASS" if report["pass"] else "FAIL")
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
