#!/usr/bin/env python3
"""Export MuLaCover YourMT3 and five chord checkpoints to native safetensors.

Run in conda qwen3-tts. Preserves inference.py parameter names and floating
point values, including separate embedding and LM head tensors. Output can be
passed to the existing GGUF conversion primitive by the package converter.
These are conversion intermediates; MuLaCover runtime loading is GGUF-only.
No architecture is executed.
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file


class _LegacyMetadata:
    """Inert replacement for historical tokenizer metadata, never executed."""


def convert(checkpoint_path: Path, output: Path, overwrite: bool = False) -> None:
    if output.exists() and not overwrite:
        raise FileExistsError(output)
    # Same restricted allowlist as upstream yourmt3/checkpoint.py. The aliases
    # discard historical training/tokenizer objects without importing them.
    aliases = [
        (_LegacyMetadata, name)
        for name in (
            "utils.task_manager.TaskManager",
            "utils.tokenizer.NoteEventTokenizer",
            "utils.note_event_dataclasses.EventRange",
            "utils.event_codec.FastCodec",
            "utils.note_event_dataclasses.Event",
        )
    ]
    core = np._core if hasattr(np, "_core") else np.core
    aliases.extend([
        core.multiarray._reconstruct,
        core.multiarray.scalar,
        (core.multiarray._reconstruct, "numpy.core.multiarray._reconstruct"),
        (core.multiarray.scalar, "numpy.core.multiarray.scalar"),
        np.ndarray, np.dtype,
        type(np.dtype("int64")), type(np.dtype("float64")),
        type(np.dtype("int32")), type(np.dtype("float32")),
    ])
    with torch.serialization.safe_globals(aliases):
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict):
        raise ValueError("YourMT3 checkpoint must be a dictionary")
    state = checkpoint.get("state_dict", checkpoint)
    if not isinstance(state, dict) or not state:
        raise ValueError("YourMT3 checkpoint must contain a nonempty state_dict")
    if not all(isinstance(k, str) and isinstance(v, torch.Tensor) for k, v in state.items()):
        raise ValueError("YourMT3 state_dict must contain named tensors only")
    # Only pitchshift is absent from the upstream inference architecture. Keep
    # BatchNorm statistics/counters and every duplicated rotary parameter name.
    tensors = {
        k: v.detach().cpu().contiguous().clone()
        for k, v in state.items() if not k.startswith("pitchshift.")
    }
    expected = {
        "embed_tokens.weight": (596, 512),
        "lm_head.lm_head.weight": (596, 512),
        "pre_decoder.0.linear.weight": (512, 256),
        "encoder.latent_array.latents": (26, 128),
        "decoder.decoder.additive_pe.weight": (1034, 512),
    }
    for name, shape in expected.items():
        if name not in tensors or tuple(tensors[name].shape) != shape:
            raise ValueError(f"Expected fixed YourMT3 tensor {name} with shape {shape}")
    # The safetensors writer is the existing conversion primitive. Clone above
    # preserves duplicated state_dict names even when tensors alias storage.
    output.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(output), metadata={
        "architecture": "yourmt3-perceiver-tf-top2moe-mc13",
        "sample_rate": "16000",
        "segment_samples": "32767",
    })
    logging.info("Exported YourMT3: %s (%d tensors)", output.name, len(tensors))


def convert_chords(root: Path, overwrite: bool = False) -> None:
    """Keep every tensor in each checkpoint's net state, with no name rewriting."""
    for fold in range(5):
        checkpoint_path = root / (
            f"joint_chord_net_ismir_naive_v1.0_reweight(0.0,10.0)_s{fold}.best.sdict"
        )
        output = root / f"fold{fold}.safetensors"
        if output.exists() and not overwrite:
            raise FileExistsError(output)
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("net"), dict):
            raise ValueError(f"Chord fold {fold} must contain a net state dictionary")
        state = checkpoint["net"]
        if not state or not all(isinstance(k, str) and isinstance(v, torch.Tensor) for k, v in state.items()):
            raise ValueError(f"Chord fold {fold} net must contain named tensors only")
        tensors = {k: v.detach().cpu().contiguous().clone() for k, v in state.items()}
        save_file(tensors, str(output), metadata={"architecture": "chordnet", "fold": str(fold)})
        logging.info("Exported chord fold %d: %s (%d tensors)", fold, output.name, len(tensors))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--chord-root", type=Path,
                        help="Directory with five raw .sdict checkpoints; writes fold0..4.safetensors there")
    parser.add_argument("--log", action="store_true")
    parser.add_argument("--log-file", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if bool(args.checkpoint) != bool(args.output) or not (args.checkpoint or args.chord_root):
        parser.error("provide --checkpoint and --output, --chord-root, or both")
    if args.log and not args.log_file:
        parser.error("--log requires --log-file")
    if args.log_file:
        args.log_file.parent.mkdir(parents=True, exist_ok=True)
        logging.basicConfig(filename=args.log_file, level=logging.INFO, format="%(message)s")
    if args.checkpoint:
        convert(args.checkpoint, args.output, args.overwrite)
    if args.chord_root:
        convert_chords(args.chord_root, args.overwrite)


if __name__ == "__main__":
    main()
