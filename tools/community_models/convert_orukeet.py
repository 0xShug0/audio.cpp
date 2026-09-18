#!/usr/bin/env python3
"""Stage Orukeet r3 weights for audio.cpp (Parakeet TDT family).

This converter reads the official Oruk ``orukeet-v0.1.0.nemo`` checkpoint, not
a GGUF round-trip, and stages an HF-layout package that the existing
``parakeet_tdt`` audio.cpp loader consumes unchanged:

    model.safetensors       HF/audio.cpp tensor names (``encoder.*``/``decoder.*``/...)
    config.json             copied from the stock Parakeet-TDT-0.6B-v3 install
    processor_config.json   copied from the stock install, verified against .nemo
    tokenizer.json          copied from the stock install (tokenizer is unchanged)
    tensor_manifest.json    source/destination names, shapes, and dtypes
    provenance.json         source revision, hashes, and license information

Architecture verdict: Orukeet r3 is a drop-in weight replacement for
``nvidia/parakeet-tdt-0.6b-v3``. It keeps 24 FastConformer layers, 1024 hidden,
9-tap temporal depthwise convolutions, 8x subsampling, 128 mel bins, a 2x640
LSTM prediction network, 8192-token vocabulary plus blank, and TDT durations
[0..4]. The 12,288 fitted Gabor kernels are materialized as ordinary Conv1d
weights, so no new operators, tensor shapes, or tokenizer changes are needed.
Evidence: the r3 export audit (651/651 parameter tensors changed, 74 fixed
buffers, tokenizer assets byte-equal, tensor shapes/dtypes equal) and the ONNX
export notes (same frontend, subsampling, vocabulary, predictor, durations).
The NeMo-side key inventory below was derived from that audit; the HF-side
names are the contract in ``src/community_models/parakeet_tdt/weights.cpp``.

The NeMo ``joint.pred`` projection (640->640) maps to audio.cpp's
``decoder.decoder_projector``; ``joint.enc`` maps to ``encoder_projector`` and
``joint.joint_net.{1,2}`` (index 2 when joint dropout is enabled, else 1) maps
to ``joint.head``. Batch-norm running statistics are fixed buffers in the
checkpoint and are staged as ``conv.norm.running_mean/var`` because the
audio.cpp loader folds them into the depthwise weights at load time.

Sidecars are intentionally NOT regenerated: the audit certifies the tokenizer
is unchanged and this script verifies the preprocessor section of the .nemo
config against the reference install before staging. Pass
``--reference-dir`` pointing at an existing
``models/parakeet-tdt-0.6b-v3`` directory (``model_manager_v2.py install
parakeet_tdt``).

This script never downloads anything. It verifies the input hash against the
pinned r3 checkpoint and refuses to convert anything else unless
``--skip-hash-check`` is given.

Dependencies are intentionally small and work without installing NeMo:

    uv run --python 3.12 --with torch --with pyyaml --with safetensors \
        --with sentencepiece --with numpy \
      tools/community_models/convert_orukeet.py \
      --checkpoint path/to/orukeet-v0.1.0.nemo \
      --converter build/windows-cuda-release/bin/audiocpp_gguf.exe \
      --gguf-output models/Orukeet-GGUF/orukeet-q8_0.gguf \
      --type q8_0

To inspect a checkpoint without converting:

    uv run --python 3.12 --with torch --with pyyaml --with sentencepiece \
      tools/community_models/convert_orukeet.py \
      --checkpoint path/to/orukeet-v0.1.0.nemo --dump-keys
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import tarfile
import tempfile
from pathlib import Path
from typing import Any

SOURCE_REPO = "oruk/orukeet"
SOURCE_REVISION = "555136b50265a132d4cea0d35560c26fc4f657ab"
SOURCE_FILE = "orukeet-v0.1.0.nemo"
SOURCE_SHA256 = "031c8ddab4845aeced904a7cde8e8aa57993b2e344716cf83a545b079c473b56"
SOURCE_LICENSE = "CC-BY-SA-4.0 (Orukeet r3 weights and fitted kernels)"
BASE_LICENSE = "CC-BY-4.0 (NVIDIA parakeet-tdt-0.6b-v3 foundation weights)"
SOURCE_LICENSE_URL = "https://huggingface.co/oruk/orukeet/blob/main/LICENSE-WEIGHTS"
ATTRIBUTION_URL = "https://huggingface.co/oruk/orukeet/blob/main/NOTICE.md"
MAPPING_REFERENCE = "oruk/orukeet evidence/librispeech-ft-20260908/r3/export-audit.json"
MAPPING_REFERENCE_COMMIT = "e0c97b367f31a49ae8f43e6d5ec48b6300d1126b"

EXPECTED_LAYERS = 24
EXPECTED_HIDDEN = 1024
EXPECTED_VOCAB = 8193
EXPECTED_BLANK = 8192
EXPECTED_DURATIONS = [0, 1, 2, 3, 4]
EXPECTED_JOINT_OUT = EXPECTED_VOCAB + len(EXPECTED_DURATIONS)

SUBSAMPLING_TABLE = [
    ("conv.0", "layers.0"),
    ("conv.2", "layers.2"),
    ("conv.3", "layers.3"),
    ("conv.5", "layers.5"),
    ("conv.6", "layers.6"),
]

# NeMo conformer-layer suffix -> HF/audio.cpp suffix, relative to the layer
# prefix. Batch-norm statistics ride along as fixed buffers.
LAYER_TABLE = [
    ("norm_feed_forward1.weight", "norm_feed_forward1.weight"),
    ("norm_feed_forward1.bias", "norm_feed_forward1.bias"),
    ("feed_forward1.linear1.weight", "feed_forward1.linear1.weight"),
    ("feed_forward1.linear2.weight", "feed_forward1.linear2.weight"),
    ("norm_self_att.weight", "norm_self_att.weight"),
    ("norm_self_att.bias", "norm_self_att.bias"),
    ("self_attn.linear_q.weight", "self_attn.q_proj.weight"),
    ("self_attn.linear_k.weight", "self_attn.k_proj.weight"),
    ("self_attn.linear_v.weight", "self_attn.v_proj.weight"),
    ("self_attn.linear_out.weight", "self_attn.o_proj.weight"),
    ("self_attn.linear_pos.weight", "self_attn.relative_k_proj.weight"),
    ("self_attn.pos_bias_u", "self_attn.bias_u"),
    ("self_attn.pos_bias_v", "self_attn.bias_v"),
    ("norm_conv.weight", "norm_conv.weight"),
    ("norm_conv.bias", "norm_conv.bias"),
    ("conv.pointwise_conv1.weight", "conv.pointwise_conv1.weight"),
    ("conv.depthwise_conv.weight", "conv.depthwise_conv.weight"),
    ("conv.batch_norm.weight", "conv.norm.weight"),
    ("conv.batch_norm.bias", "conv.norm.bias"),
    ("conv.batch_norm.running_mean", "conv.norm.running_mean"),
    ("conv.batch_norm.running_var", "conv.norm.running_var"),
    ("conv.pointwise_conv2.weight", "conv.pointwise_conv2.weight"),
    ("norm_feed_forward2.weight", "norm_feed_forward2.weight"),
    ("norm_feed_forward2.bias", "norm_feed_forward2.bias"),
    ("feed_forward2.linear1.weight", "feed_forward2.linear1.weight"),
    ("feed_forward2.linear2.weight", "feed_forward2.linear2.weight"),
    ("norm_out.weight", "norm_out.weight"),
    ("norm_out.bias", "norm_out.bias"),
]

SKIP_PREFIXES = ("preprocessor.",)
SKIP_SUFFIXES = (".num_batches_tracked",)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_checkpoint(source: Path) -> tuple[dict[str, Any], dict[str, Any], list[str]]:
    """Return the NeMo YAML config, state dict, and .nemo member names."""
    import torch
    import yaml

    if source.is_dir():
        config_path = source / "model_config.yaml"
        checkpoint_path = source / "model_weights.ckpt"
        if not config_path.exists() or not checkpoint_path.exists():
            raise SystemExit(
                f"checkpoint directory needs model_config.yaml and model_weights.ckpt: {source}"
            )
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        members = [config_path.name, checkpoint_path.name]
        return config, state, members

    if not source.exists():
        raise SystemExit(f"checkpoint not found: {source}")
    def member(archive: tarfile.TarFile, name: str):
        for candidate in (name, f"./{name}"):
            try:
                found = archive.extractfile(candidate)
            except KeyError:
                continue
            if found is not None:
                return found
        return None

    with tarfile.open(source, mode="r:*") as archive:
        names = archive.getnames()
        config_member = member(archive, "model_config.yaml")
        weights_member = member(archive, "model_weights.ckpt")
        if config_member is None or weights_member is None:
            raise SystemExit(
                f"NeMo archive is missing model_config.yaml or model_weights.ckpt: {source}"
            )
        config_text = config_member.read().decode("utf-8")
        weights = weights_member.read()

    with tempfile.NamedTemporaryFile(suffix=".ckpt", delete=False) as handle:
        handle.write(weights)
        checkpoint_path = Path(handle.name)
    config = yaml.safe_load(config_text)
    try:
        import torch

        state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    finally:
        checkpoint_path.unlink(missing_ok=True)
    return config, state, names


def need(config: dict[str, Any], dotted: str) -> Any:
    node: Any = config
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            top = sorted(config.keys()) if isinstance(config, dict) else []
            raise ValueError(
                f"NeMo config has no '{dotted}' (top-level keys: {top})"
            )
        node = node[part]
    return node


def assert_model_shape(config: dict[str, Any]) -> None:
    checks = {
        "encoder.n_layers": (need(config, "encoder.n_layers"), EXPECTED_LAYERS),
        "encoder.d_model": (need(config, "encoder.d_model"), EXPECTED_HIDDEN),
    }
    mismatches = [
        f"{name}={actual!r}, expected {expected!r}"
        for name, (actual, expected) in checks.items()
        if actual != expected
    ]
    if mismatches:
        raise ValueError(
            "not the expected Parakeet-TDT-0.6B-v3 architecture: " + "; ".join(mismatches)
        )


def verify_preprocessor(config: dict[str, Any], reference: Path) -> None:
    """Fail unless the .nemo preprocessor matches the reference install."""
    ref = json.loads((reference / "processor_config.json").read_text(encoding="utf-8"))
    feat = ref.get("feature_extractor", ref)
    pre = config.get("preprocessor", {})
    expected = {
        "sampling_rate": int(feat.get("sampling_rate", 16000)),
        "features": int(feat.get("feature_size", 128)),
        "n_fft": int(feat.get("n_fft", 512)),
    }
    actual = {
        "sampling_rate": int(pre.get("sample_rate", -1)),
        "features": int(pre.get("features", -1)),
        "n_fft": int(pre.get("n_fft", -1)),
    }
    if actual != expected:
        raise ValueError(
            f".nemo preprocessor {actual} does not match reference {expected}; "
            "refusing to reuse the stock sidecars"
        )


def verify_tokenizer_vocab(checkpoint: Path, expected: int) -> None:
    """Fail unless the .nemo sentencepiece model has the expected piece count.

    Note: the ``*_vocab.txt`` member is a training word list, not the BPE
    vocabulary, so the check runs against ``*_tokenizer.model`` instead.
    """
    import sentencepiece as spm

    if checkpoint.is_dir():
        candidates = list(checkpoint.glob("*_tokenizer.model"))
        if not candidates:
            raise SystemExit(f"checkpoint directory has no *_tokenizer.model: {checkpoint}")
        processor = spm.SentencePieceProcessor(model_file=str(candidates[0]))
    else:
        with tarfile.open(checkpoint, mode="r:*") as archive:
            names = [n for n in archive.getnames() if n.endswith("_tokenizer.model")]
            if not names:
                raise SystemExit(f"NeMo archive has no *_tokenizer.model: {checkpoint}")
            member = archive.extractfile(names[0])
            assert member is not None
            blob = member.read()
        with tempfile.NamedTemporaryFile(suffix=".model", delete=False) as handle:
            handle.write(blob)
            model_path = Path(handle.name)
        try:
            processor = spm.SentencePieceProcessor(model_file=str(model_path))
        finally:
            model_path.unlink(missing_ok=True)
    pieces = processor.get_piece_size()
    if pieces != expected:
        raise ValueError(
            f".nemo tokenizer has {pieces} pieces, expected {expected}; "
            "refusing to reuse the stock tokenizer"
        )


def dump_keys(checkpoint: Path) -> None:
    _, state, _ = load_checkpoint(checkpoint)
    for key in sorted(state.keys()):
        tensor = state[key]
        shape = list(tensor.shape) if hasattr(tensor, "shape") else type(tensor).__name__
        print(f"{key} {shape}")


def convert(checkpoint: Path, reference: Path, output_dir: Path, source_hash: str) -> None:
    import torch
    from safetensors.torch import save_file

    for name in ("config.json", "processor_config.json", "tokenizer.json"):
        if not (reference / name).exists():
            raise SystemExit(f"reference install is missing {name}: {reference}")
    ref_config = json.loads((reference / "config.json").read_text(encoding="utf-8"))
    if int(ref_config.get("vocab_size", -1)) != EXPECTED_VOCAB:
        raise SystemExit(
            f"reference vocab_size={ref_config.get('vocab_size')}, expected {EXPECTED_VOCAB}"
        )

    config, state, _ = load_checkpoint(checkpoint)
    if not isinstance(state, dict) or not state:
        raise ValueError("checkpoint did not contain a non-empty state dictionary")
    assert_model_shape(config)
    verify_preprocessor(config, reference)
    verify_tokenizer_vocab(checkpoint, 8192)
    output_dir.mkdir(parents=True, exist_ok=True)

    out: dict[str, torch.Tensor] = {}
    manifest: list[dict[str, Any]] = []
    used: set[str] = set()

    def emit(src: str, dst: str, shape: list[int] | None = None) -> None:
        if src not in state:
            raise KeyError(f"missing expected tensor: {src}")
        if dst in out:
            raise KeyError(f"duplicate destination tensor: {dst}")
        tensor = state[src]
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"expected tensor for {src}, got {type(tensor).__name__}")
        tensor = tensor.detach().cpu().contiguous()
        if shape is not None and list(tensor.shape) != shape:
            raise ValueError(
                f"shape mismatch for {src}: got {list(tensor.shape)}, expected {shape}"
            )
        out[dst] = tensor
        used.add(src)
        manifest.append(
            {"source": src, "destination": dst, "shape": list(tensor.shape),
             "dtype": str(tensor.dtype)}
        )

    for src_mid, dst_mid in SUBSAMPLING_TABLE:
        emit(f"encoder.pre_encode.{src_mid}.weight",
             f"encoder.subsampling.{dst_mid}.weight")
        emit(f"encoder.pre_encode.{src_mid}.bias",
             f"encoder.subsampling.{dst_mid}.bias")
    emit("encoder.pre_encode.out.weight", "encoder.subsampling.linear.weight")
    emit("encoder.pre_encode.out.bias", "encoder.subsampling.linear.bias")

    for i in range(EXPECTED_LAYERS):
        for src_suffix, dst_suffix in LAYER_TABLE:
            emit(f"encoder.layers.{i}.{src_suffix}", f"encoder.layers.{i}.{dst_suffix}")

    emit("decoder.prediction.embed.weight", "decoder.embedding.weight",
         [EXPECTED_VOCAB, 640])
    for layer in range(2):
        for kind in ("weight_ih", "weight_hh", "bias_ih", "bias_hh"):
            emit(f"decoder.prediction.dec_rnn.lstm.{kind}_l{layer}",
                 f"decoder.lstm.{kind}_l{layer}", [2560, 640] if "weight" in kind else [2560])
    # NeMo's joint prediction projection is filed under the audio.cpp decoder.
    emit("joint.pred.weight", "decoder.decoder_projector.weight", [640, 640])
    emit("joint.pred.bias", "decoder.decoder_projector.bias", [640])
    emit("joint.enc.weight", "encoder_projector.weight", [640, EXPECTED_HIDDEN])
    emit("joint.enc.bias", "encoder_projector.bias", [640])
    joint_head = None
    for candidate in ("joint.joint_net.2", "joint.joint_net.1"):
        if f"{candidate}.weight" in state:
            joint_head = candidate
            break
    if joint_head is None:
        raise KeyError("missing expected tensor: joint.joint_net.{1,2}.weight")
    emit(f"{joint_head}.weight", "joint.head.weight", [EXPECTED_JOINT_OUT, 640])
    emit(f"{joint_head}.bias", "joint.head.bias", [EXPECTED_JOINT_OUT])

    unexpected = [
        key for key in set(state) - used
        if not key.startswith(SKIP_PREFIXES) and not key.endswith(SKIP_SUFFIXES)
    ]
    if unexpected:
        raise ValueError(
            f"{len(unexpected)} unmapped state_dict tensors, e.g. {sorted(unexpected)[:8]}"
        )

    save_file(out, str(output_dir / "model.safetensors"), metadata={
        "format": "orukeet-r3-audiocpp-staging",
        "tensor_namespace": "encoder/decoder/joint",
        "source_repo": SOURCE_REPO,
        "source_revision": SOURCE_REVISION,
    })
    for name in ("config.json", "processor_config.json", "tokenizer.json"):
        shutil.copyfile(reference / name, output_dir / name)
    (output_dir / "tensor_manifest.json").write_text(json.dumps({
        "source": SOURCE_REPO,
        "revision": SOURCE_REVISION,
        "source_sha256": source_hash,
        "joint_head_source": joint_head,
        "source_tensor_count": len(state),
        "emitted_tensor_count": len(out),
        "skipped_tensor_count": len(state) - len(out),
        "tensors": manifest,
    }, indent=2) + "\n", encoding="utf-8")
    (output_dir / "provenance.json").write_text(json.dumps({
        "source_repo": SOURCE_REPO,
        "source_revision": SOURCE_REVISION,
        "source_file": SOURCE_FILE,
        "source_license": SOURCE_LICENSE,
        "base_license": BASE_LICENSE,
        "source_license_url": SOURCE_LICENSE_URL,
        "attribution": ATTRIBUTION_URL,
        "source_sha256": source_hash,
        "mapping_reference": MAPPING_REFERENCE,
        "mapping_reference_commit": MAPPING_REFERENCE_COMMIT,
        "tensor_namespace": "encoder/decoder/joint",
        "sidecars": "reused from stock parakeet-tdt-0.6b-v3 install; "
                    "tokenizer certified unchanged by the r3 export audit",
    }, indent=2) + "\n", encoding="utf-8")
    print(f"source tensors: {len(state)}")
    print(f"emitted tensors: {len(out)}")
    print(f"skipped tensors: {len(state) - len(out)} (preprocessor + counters)")
    print(f"wrote staging package: {output_dir}")


def package_with_audiocpp(args: argparse.Namespace, staging: Path) -> None:
    if args.converter is None or args.gguf_output is None:
        return
    converter = args.converter.resolve()
    if not converter.exists():
        raise SystemExit(f"audiocpp_gguf not found: {converter}")
    spec = Path(__file__).resolve().parents[2] / "model_specs" / "parakeet_tdt.json"
    if not spec.exists():
        raise SystemExit(f"model spec not found: {spec}")
    output = args.gguf_output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        str(converter),
        "--input", str(staging / "model.safetensors"),
        "--root", str(staging),
        "--family", "parakeet_tdt",
        "--model-spec", str(spec),
        "--type", args.type,
        "--output", str(output),
        "--overwrite",
    ]
    print("+", " ".join(command))
    subprocess.run(command, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="official .nemo file or extracted checkpoint directory")
    parser.add_argument("--reference-dir", type=Path, default=None,
                        help="stock parakeet-tdt-0.6b-v3 install "
                             "(default: <repo>/models/parakeet-tdt-0.6b-v3)")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="keep the staging package here instead of a temp dir")
    parser.add_argument("--converter", type=Path, default=None,
                        help="audiocpp_gguf executable for the GGUF step")
    parser.add_argument("--gguf-output", type=Path, default=None,
                        help="final GGUF output path (requires --converter)")
    parser.add_argument("--type", default="orig",
                        choices=["orig", "f32", "f16", "bf16", "q8_0"],
                        help="GGUF storage type; use orig for the F32 gate")
    parser.add_argument("--dump-keys", action="store_true",
                        help="list checkpoint keys and shapes, then exit")
    parser.add_argument("--skip-hash-check", action="store_true",
                        help="allow a checkpoint whose SHA-256 is not the pinned r3")
    args = parser.parse_args()

    checkpoint = args.checkpoint.resolve()
    if args.dump_keys:
        dump_keys(checkpoint)
        return 0
    if args.reference_dir is None:
        args.reference_dir = Path(__file__).resolve().parents[2] / "models" / "parakeet-tdt-0.6b-v3"
    if (args.gguf_output is not None) != (args.converter is not None):
        raise SystemExit("--converter and --gguf-output must be given together")
    if args.output_dir is None and args.converter is None:
        raise SystemExit("nothing to keep: give --output-dir, or --converter with --gguf-output")

    source_hash = sha256(checkpoint) if checkpoint.is_file() else None
    if source_hash is None:
        import torch

        sibling = next(checkpoint.glob("*.ckpt"), None)
        source_hash = sha256(sibling) if sibling else "extracted-directory"
    if not args.skip_hash_check and source_hash != SOURCE_SHA256:
        raise SystemExit(
            f"checkpoint SHA-256 {source_hash} is not the pinned r3 {SOURCE_SHA256}; "
            "pass --skip-hash-check to override"
        )

    if args.output_dir is not None:
        staging = args.output_dir.resolve()
        convert(checkpoint, args.reference_dir.resolve(), staging, source_hash)
    else:
        with tempfile.TemporaryDirectory(prefix="orukeet-convert-") as tmp:
            staging = Path(tmp) / "staging"
            convert(checkpoint, args.reference_dir.resolve(), staging, source_hash)
            package_with_audiocpp(args, staging)
            return 0
    package_with_audiocpp(args, staging)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
