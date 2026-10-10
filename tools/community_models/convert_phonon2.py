#!/usr/bin/env python3
"""Convert the official Phonon-2 container to an exact, dense Parakeet package.

Requires numpy, safetensors, and zstandard (no PyTorch or NeMo). The packed
archive and the original Parakeet config/tokenizer must already be downloaded.
F32 is intentional: the int6 row-scale products cannot all be represented in
F16. This is not a second quantization of Phonon's trained five-value weights.
The format follows Fermion Research's Apache-2.0 fermion_container.py:
https://huggingface.co/FermionResearch/Phonon-2/blob/main/fermion_container.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shutil
import subprocess
import tarfile
import tempfile
from pathlib import Path

import numpy as np

SOURCE_REPO = "FermionResearch/Phonon-2"
SOURCE_REVISION = "7e153e4054c0a10db6d47d6fff5c062d4c77c154"
BASE_REPO = "nvidia/parakeet-tdt-0.6b-v3"
BASE_REVISION = "541d1f99c6b0c3cd0b11a95167540bb8edefd82b"
ARCHIVE = "phonon-2.bps.tar.zst"
ARCHIVE_SHA256 = "98125795b6dda72f5c6eee9ba33d19815df65dcb18b50a357bf9f73c9935309e"
CONTAINER_SHA256 = "4b6bfa3a12cc3c4e0a54f2ab3ec4ca7a842b09e5c7ecfc8e7ca0ac6cc8c11468"
FORMAT = "fermion-five-value-parakeet-v1"
SIDECARS = ("config.json", "processor_config.json", "tokenizer.json",
            "tokenizer_config.json", "generation_config.json")
LICENSES = ("NOTICE", "LICENSE-WEIGHTS-CC-BY-4.0.txt", "LICENSE-CODE-Apache-2.0.txt")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_exact(f, count: int) -> bytes:
    data = f.read(count)
    if len(data) != count:
        raise ValueError(f"truncated container: wanted {count} bytes, got {len(data)}")
    return data


def decode_record(kind: str, shape: tuple[int, ...], blob: bytes) -> np.ndarray:
    if (not shape and kind != "fp16") or len(shape) > 4 or any(type(d) is not int or d <= 0 for d in shape):
        raise ValueError(f"invalid tensor shape: {shape}")
    total = math.prod(shape)
    if kind == "fp16":
        if len(blob) != total * 2:
            raise ValueError("invalid fp16 record size")
        return np.frombuffer(blob, dtype="<f2").reshape(shape).astype(np.float32)
    o = shape[0]
    if kind == "five_value":
        if len(shape) != 2:
            raise ValueError("five_value records must be matrices")
        i = shape[1]
        trit_bytes = o * ((i + 4) // 5)
        if len(blob) < trit_bytes + 4 * o:
            raise ValueError("truncated five_value record")
        packed = np.frombuffer(blob, dtype=np.uint8, count=trit_bytes)
        if np.any(packed >= 243):
            raise ValueError("invalid base-three byte")
        packed = packed.reshape(o, -1).astype(np.uint16)
        codes = np.stack([(packed // (3 ** k)) % 3 for k in range(5)], axis=-1)
        codes = codes.reshape(o, -1)[:, :i]
        nonzero = codes != 1
        nnz = int(nonzero.sum())
        bit_bytes = (nnz + 7) // 8
        offset = trit_bytes + bit_bytes
        if len(blob) != offset + 4 * o:
            raise ValueError("invalid five_value record size")
        levels = np.zeros(shape, dtype=bool)
        levels[nonzero] = np.unpackbits(
            np.frombuffer(blob[trit_bytes:offset], dtype=np.uint8), bitorder="little")[:nnz]
        lo = np.frombuffer(blob, dtype="<f2", count=o, offset=offset).astype(np.float32)
        hi = np.frombuffer(blob, dtype="<f2", count=o, offset=offset + 2 * o).astype(np.float32)
        return (codes.astype(np.float32) - 1) * np.where(levels, hi[:, None], lo[:, None])
    if kind not in ("int6", "int8"):
        raise ValueError(f"unsupported record kind: {kind}")
    body_size = ((total + 3) // 4) * 3 if kind == "int6" else total
    if len(blob) != body_size + 2 * o:
        raise ValueError(f"invalid {kind} record size")
    scales = np.frombuffer(blob, dtype="<f2", count=o, offset=body_size).astype(np.float32)
    if kind == "int8":
        q = np.frombuffer(blob, dtype=np.int8, count=total).astype(np.float32)
    else:
        b = np.frombuffer(blob, dtype=np.uint8, count=body_size).reshape(-1, 3).astype(np.uint32)
        packed = b[:, 0] | (b[:, 1] << 8) | (b[:, 2] << 16)
        q = np.stack([(packed >> shift) & 63 for shift in (0, 6, 12, 18)], axis=1)
        q = q.ravel()[:total].astype(np.float32) - 32
    return (q.reshape(o, -1) * scales[:, None]).reshape(shape)


def read_container(path: Path) -> tuple[dict[str, np.ndarray], list[dict]]:
    tensors = {}
    size = path.stat().st_size
    with path.open("rb") as f:
        header_size = int.from_bytes(read_exact(f, 8), "little")
        if header_size > min(16 * 1024 * 1024, size - 8):
            raise ValueError("invalid container header size")
        header = json.loads(read_exact(f, header_size))
        if header.get("format") != FORMAT:
            raise ValueError("unsupported container format")
        index = header["index"]
        if sum(e["b"] for e in index) != size - 8 - header_size:
            raise ValueError("container index size does not match file")
        for e in index:
            if type(e["b"]) is not int or e["b"] < 0:
                raise ValueError("invalid record byte count")
            name = e["n"] + (".weight" if e["k"] == "five_value" else "")
            if name in tensors:
                raise ValueError(f"duplicate tensor: {name}")
            value = decode_record(e["k"], tuple(e["shape"]), read_exact(f, e["b"]))
            tensors[name] = value
        if f.read(1):
            raise ValueError("trailing container bytes")
    return tensors, index


def extract_archive(source: Path, destination: Path) -> None:
    import zstandard
    archive = source / ARCHIVE
    if sha256(archive) != ARCHIVE_SHA256:
        raise ValueError("archive SHA256 differs from the pinned official Phonon-2 artifact")
    with archive.open("rb") as f, zstandard.ZstdDecompressor().stream_reader(f) as stream:
        with tarfile.open(fileobj=stream, mode="r|") as tar:
            seen = set()
            for member in tar:
                # Only the two inputs required here are accepted; no extractall,
                # links, paths outside the staging directory, or executable code.
                if member.name not in ("config.json", "model.fermion"):
                    continue
                if not member.isfile() or member.name in seen:
                    raise ValueError("invalid or duplicate archive member")
                seen.add(member.name)
                with tar.extractfile(member) as inp, (destination / member.name).open("wb") as out:
                    shutil.copyfileobj(inp, out)
            if seen != {"config.json", "model.fermion"}:
                raise ValueError("archive lacks config.json or model.fermion")
    if sha256(destination / "model.fermion") != CONTAINER_SHA256:
        raise ValueError("container SHA256 differs from the official artifact")


def validate_sidecars(reference: Path, source_config: dict) -> None:
    def load(name):
        return json.loads((reference / name).read_text(encoding="utf-8"))
    cfg = load("config.json")
    expected = {"model_type": "parakeet_tdt", "vocab_size": 8193, "blank_token_id": 8192,
                "decoder_hidden_size": 640, "num_decoder_layers": 2,
                "durations": [0, 1, 2, 3, 4], "max_symbols_per_step": 10}
    for key, value in expected.items():
        if cfg.get(key) != value:
            raise ValueError(f"incompatible reference config: {key}")
    encoder = cfg["encoder_config"]
    for key, value in {"hidden_size": 1024, "intermediate_size": 4096,
                       "num_hidden_layers": 24, "num_attention_heads": 8,
                       "num_mel_bins": 128, "conv_kernel_size": 9,
                       "subsampling_factor": 8, "subsampling_conv_channels": 256,
                       "subsampling_conv_kernel_size": 3, "subsampling_conv_stride": 2,
                       "attention_bias": False, "convolution_bias": False,
                       "scale_input": False, "hidden_act": "silu"}.items():
        if encoder.get(key) != value:
            raise ValueError(f"incompatible encoder config: {key}")
    feature = load("processor_config.json")["feature_extractor"]
    for key, value in {"sampling_rate": 16000, "feature_size": 128, "n_fft": 512,
                       "win_length": 400, "hop_length": 160, "preemphasis": 0.97}.items():
        if feature.get(key) != value:
            raise ValueError(f"incompatible frontend config: {key}")
    vocabulary = load("tokenizer.json")["model"]["vocab"]
    if not isinstance(vocabulary, dict):
        raise ValueError("expected the original Parakeet BPE tokenizer")
    source_vocab = source_config["joint"]["vocabulary"]
    if len(source_vocab) != 8192 or vocabulary != {token: i for i, token in enumerate(source_vocab)}:
        raise ValueError("tokenizer vocabulary differs from the Phonon-2 archive")
    for name in SIDECARS:
        if not (reference / name).is_file():
            raise ValueError(f"missing reference sidecar: {name}")


def configure_timestamps(config: dict, tokenizer_path: Path) -> None:
    vocabulary = json.loads(tokenizer_path.read_text(encoding="utf-8"))["model"]["vocab"]
    config["audiocpp_word_timestamp_mode"] = "token_duration"
    # Match Phonon's Unicode-aware rule: punctuation has no letter/digit.
    config["audiocpp_punctuation_token_ids"] = sorted(
        i for piece, i in vocabulary.items() if not any(ch.isalnum() for ch in piece))


def convert(source: Path, reference: Path, output: Path) -> Path:
    from safetensors.numpy import save_file
    if output.exists():
        raise ValueError(f"output directory already exists; use a new directory: {output}")
    for name in LICENSES:
        if not (source / name).is_file():
            raise ValueError(f"missing required attribution file: {name}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="phonon2-", dir=output.parent) as temp:
        temp = Path(temp)
        extracted = temp / "source"
        extracted.mkdir()
        extract_archive(source, extracted)
        source_config = json.loads((extracted / "config.json").read_text(encoding="utf-8"))
        validate_sidecars(reference, source_config)
        tensors, records = read_container(extracted / "model.fermion")
        if len(records) != 723:
            raise ValueError("unexpected tensor inventory in the pinned artifact")
        manifest = []
        counters = []
        for name in list(tensors):
            arr = tensors[name]
            if name.endswith(".num_batches_tracked"):
                # Training-only BatchNorm counters may be stored as infinity.
                # Neither audio.cpp nor the reference eval graph reads them.
                counters.append(name)
                del tensors[name]
                continue
            if not np.isfinite(arr).all():
                raise ValueError(f"non-finite inference tensor: {name}")
            if arr.ndim == 2 and re.search(r"\.conv\.pointwise_conv[12]\.weight$", name):
                arr = arr[:, :, None]
            tensors[name] = np.ascontiguousarray(arr, dtype=np.float32)
            manifest.append({"name": name, "shape": list(arr.shape), "dtype": "F32",
                             "sha256": hashlib.sha256(tensors[name].tobytes()).hexdigest()})
        staged = temp / "package"
        staged.mkdir()
        save_file(tensors, str(staged / "model.safetensors"), metadata={
            "format": "pt", "source_repo": SOURCE_REPO, "source_revision": SOURCE_REVISION,
            "conversion": "exact F32 reconstruction of the trained five-value/int6/fp16 records"})
        for name in SIDECARS:
            shutil.copyfile(reference / name, staged / name)
        # The existing Parakeet loader floors BN variance at epsilon before
        # adding epsilon. Phonon has two trained variances below that floor;
        # its reference uses sqrt(variance + epsilon). Opt in via this package
        # only, without changing either its tensors or other Parakeet variants.
        runtime_config = json.loads((staged / "config.json").read_text(encoding="utf-8"))
        runtime_config["encoder_config"]["batch_norm_variance_floor"] = 0.0
        # The HF reference's encoder_config.scale_input is false. Keep the
        # established NeMo scaling as the loader default for other packages.
        runtime_config["encoder_config"]["subsampling_input_scale"] = 1.0
        runtime_config["audiocpp_matmul_precision"] = "f32"
        configure_timestamps(runtime_config, staged / "tokenizer.json")
        (staged / "config.json").write_text(json.dumps(runtime_config, indent=2) + "\n", encoding="utf-8")
        for name in LICENSES:
            shutil.copyfile(source / name, staged / name)
        spec_path = Path(__file__).resolve().parents[2] / "model_specs" / "parakeet_tdt.json"
        spec = json.loads(spec_path.read_text(encoding="utf-8"))
        spec.update(display_name="Phonon-2 (dense F32)", status="experimental", languages=["en"],
                    modes=["offline", "streaming"], packages=[],
                    description="Fermion Research Phonon-2 English ASR. Exact F32 weight reconstruction; "
                                "uses the native Parakeet TDT runtime, without packed five-value kernels.")
        spec.pop("package_defaults", None)
        spec["ui"].pop("recommended_package", None)
        spec["ui"]["docs"] = ["docs/community_models/phonon2.md"]
        spec["runtime"]["tags"] = ["gguf", "stream"]
        spec["capabilities"]["asr"] = ["word_timestamps", "partial_results"]
        (staged / "model_spec.json").write_text(json.dumps(spec, indent=2) + "\n", encoding="utf-8")
        provenance = {"source_repo": SOURCE_REPO, "source_revision": SOURCE_REVISION,
                      "archive_sha256": ARCHIVE_SHA256, "container_sha256": CONTAINER_SHA256,
                      "base_repo": BASE_REPO, "base_revision": BASE_REVISION,
                      "weights_license": "CC-BY-4.0", "family": "parakeet_tdt", "language": "en",
                      "storage": "F32", "records": len(records), "tensors": len(tensors),
                      "runtime_config_overrides": {"encoder_config.batch_norm_variance_floor": 0.0,
                                                   "encoder_config.subsampling_input_scale": 1.0,
                                                   "audiocpp_matmul_precision": "f32",
                                                   "audiocpp_word_timestamp_mode": "token_duration"},
                      "dropped_training_counters": counters,
                      "sidecar_sha256": {name: sha256(reference / name) for name in SIDECARS},
                      "output_sidecar_sha256": {name: sha256(staged / name) for name in SIDECARS}}
        (staged / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
        (staged / "tensor_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        staged.rename(output)
    return output


def write_gguf(staged: Path, converter: Path, output: Path, storage: str) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    # Reduced precision is a rounded derivative of exact F32 staging. Do not
    # embed its F32 tensor hashes or describe the derivative as lossless.
    with tempfile.TemporaryDirectory(prefix="phonon2-gguf-") as temp:
        metadata = Path(temp)
        for name in SIDECARS + LICENSES + ("model_spec.json", "provenance.json"):
            shutil.copyfile(staged / name, metadata / name)
        config = json.loads((metadata / "config.json").read_text(encoding="utf-8"))
        configure_timestamps(config, metadata / "tokenizer.json")
        if storage == "q8_0":
            config["audiocpp_cpu_matmul_weight_type"] = "f32"
        (metadata / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
        if storage == "orig":
            shutil.copyfile(staged / "tensor_manifest.json", metadata / "tensor_manifest.json")
        if storage != "orig":
            label = storage.upper()
            spec = json.loads((staged / "model_spec.json").read_text(encoding="utf-8"))
            spec["display_name"] = f"Phonon-2 (dense {label})"
            spec["description"] = (f"Fermion Research Phonon-2 English ASR. {label}-rounded derivative "
                                   "of the exact F32 reconstruction; uses the native Parakeet TDT "
                                   "runtime, without packed five-value kernels.")
            (metadata / "model_spec.json").write_text(json.dumps(spec, indent=2) + "\n", encoding="utf-8")
            provenance = json.loads((staged / "provenance.json").read_text(encoding="utf-8"))
            provenance.update(storage=label, exact_official_weights=False,
                              source_f32_safetensors_sha256=sha256(staged / "model.safetensors"),
                              conversion=f"audiocpp_gguf --type {storage}; weights rounded from exact F32 staging")
            (metadata / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
        provenance = json.loads((metadata / "provenance.json").read_text(encoding="utf-8"))
        provenance["runtime_config_overrides"]["audiocpp_word_timestamp_mode"] = "token_duration"
        if storage == "q8_0":
            provenance["runtime_config_overrides"]["audiocpp_cpu_matmul_weight_type"] = "f32"
            provenance["preserved_f32_tensors"] = ["joint.head.weight", "joint.head.bias"]
            provenance["conversion"] += "; --keep-type joint.head.*=orig (duration-sensitive output head)"
        provenance["output_sidecar_sha256"]["config.json"] = sha256(metadata / "config.json")
        (metadata / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
        command = [str(converter.resolve()), "--input", str(staged / "model.safetensors"),
                   "--root", str(metadata), "--family", "parakeet_tdt", "--model-spec",
                   str(metadata / "model_spec.json"), "--type", storage, "--output", str(output.resolve())]
        if storage == "q8_0":
            # The joint head predicts both text and duration. Preserve its
            # original scale products: quantization can change duration ends
            # even when text and subsequent word starts remain identical.
            command += ["--keep-type", "joint.head.*=orig"]
        subprocess.run(command, check=True, **({"creationflags": subprocess.CREATE_NO_WINDOW}
                                              if hasattr(subprocess, "CREATE_NO_WINDOW") else {}))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True, help="official archive and license/NOTICE files")
    parser.add_argument("--reference-dir", type=Path, required=True, help="original Parakeet config/tokenizer sidecars")
    parser.add_argument("--output-dir", type=Path, required=True, help="new dense staging directory")
    parser.add_argument("--converter", type=Path, help="optional path to audiocpp_gguf")
    parser.add_argument("--gguf-output", type=Path, help="optional standalone GGUF output")
    parser.add_argument("--gguf-type", choices=("orig", "bf16", "q8_0"), default="orig",
                        help="orig preserves exact F32 weights; bf16/q8_0 reduce storage with rounding")
    args = parser.parse_args()
    if bool(args.converter) != bool(args.gguf_output):
        parser.error("--converter and --gguf-output must be provided together")
    if args.gguf_type != "orig" and not args.converter:
        parser.error("--gguf-type requires --converter and --gguf-output")
    if args.converter and not args.converter.is_file():
        parser.error(f"audiocpp_gguf not found: {args.converter}")
    if args.gguf_output and args.gguf_output.exists():
        parser.error(f"GGUF output already exists; use a new path: {args.gguf_output}")
    staged = convert(args.source_dir.resolve(), args.reference_dir.resolve(), args.output_dir.resolve())
    print(f"Wrote lossless F32 package: {staged}")
    if args.converter:
        write_gguf(staged, args.converter, args.gguf_output, args.gguf_type)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
