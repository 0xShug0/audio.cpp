#!/usr/bin/env python3
"""Convert a sherpa-onnx streaming Zipformer keyword spotter to GGUF."""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import onnx
from safetensors.numpy import save_file


REPO = Path(__file__).resolve().parents[3]


def load_zipformer_converter():
    path = REPO / "tools/community_models/convert_kroko_onnx.py"
    spec = importlib.util.spec_from_file_location("audiocpp_zipformer_converter", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def metadata(model_path: Path, converter) -> tuple[dict[str, object], list[str]]:
    model = onnx.load(model_path, load_external_data=False)
    values = {item.key: item.value for item in model.metadata_props}

    def ints(key: str) -> list[int]:
        return [int(part) for part in values[key].split(",")]

    layers = ints("num_encoder_layers")
    prefixes: list[str] = []
    for stack, count in enumerate(layers):
        base = f"encoder.encoders.{stack}"
        if stack != 0:
            base += ".encoder"
        prefixes.extend(f"{base}.layers.{layer}" for layer in range(count))
    left = ints("left_context_len")
    base_left = left[0]
    if any(value <= 0 or base_left % value != 0 for value in left):
        raise ValueError("invalid Zipformer left-context metadata")
    config = {
        "audiocpp_family": "sherpa_kws",
        "model_type": values.get("model_type", "zipformer2"),
        "variant": "sherpa-onnx-streaming-kws",
        "sample_rate": 16000,
        "feature_dim": 80,
        "chunk_size": int(values["T"]),
        "chunk_shift": int(values["decode_chunk_len"]),
        "subsampling_factor": 4,
        "encoder_dims": ints("encoder_dims"),
        "query_head_dims": ints("query_head_dims"),
        "value_head_dims": ints("value_head_dims"),
        "num_heads": ints("num_heads"),
        "num_encoder_layers": layers,
        "cnn_module_kernels": ints("cnn_module_kernels"),
        "left_context_len": left,
        "downsampling_factors": [base_left // value for value in left],
    }
    return config, prefixes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="directory containing encoder, decoder, joiner and tokens files")
    parser.add_argument("output", type=Path, help="output GGUF path")
    parser.add_argument("--encoder", default="encoder-epoch-13-avg-2-chunk-16-left-64.int8.onnx")
    parser.add_argument("--decoder", default="decoder-epoch-13-avg-2-chunk-16-left-64.onnx")
    parser.add_argument("--joiner", default="joiner-epoch-13-avg-2-chunk-16-left-64.int8.onnx")
    parser.add_argument("--keywords", type=Path, help="default tokenized keyword file")
    parser.add_argument("--audiocpp-gguf", type=Path, default=REPO / "build/debug/bin/audiocpp_gguf")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    root = args.input.resolve()
    output = args.output.resolve()
    paths = {name: root / getattr(args, name) for name in ("encoder", "decoder", "joiner")}
    paths["tokens"] = root / "tokens.txt"
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise SystemExit("missing sherpa resources: " + ", ".join(missing))
    if output.exists() and not args.overwrite:
        raise SystemExit(f"output exists (pass --overwrite): {output}")

    converter = load_zipformer_converter()
    config, prefixes = metadata(paths["encoder"], converter)
    converter.LAYER_PREFIXES = prefixes
    tensors = converter.convert_graph(paths["encoder"], "encoder", linear_pos_prefixes=prefixes)
    tensors.update(converter.convert_graph(paths["decoder"], "decoder"))
    tensors.update(converter.convert_graph(paths["joiner"], "joiner"))
    decoder_meta = converter.decoder_metadata(paths["decoder"])
    config.update(decoder_meta)

    keywords = args.keywords.resolve() if args.keywords else root / "test_wavs/keywords.txt"
    if not keywords.is_file():
        raise SystemExit(f"default tokenized keyword file not found: {keywords}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="audiocpp-sherpa-kws-") as temporary:
        staging = Path(temporary)
        save_file(tensors, str(staging / "model.safetensors"), metadata={
            "format": "pt", "source": "sherpa-onnx", "audiocpp_family": "sherpa_kws"})
        (staging / "config.json").write_text(json.dumps(config, indent=2) + "\n")
        shutil.copyfile(paths["tokens"], staging / "tokens.txt")
        shutil.copyfile(keywords, staging / "keywords.txt")
        command = [str(args.audiocpp_gguf.resolve()), "--input", f"weights={staging / 'model.safetensors'}",
                   "--output", str(output), "--type", "orig", "--family", "sherpa_kws",
                   "--model-spec", str(REPO / "model_specs/sherpa_kws.json"), "--root", str(staging)]
        if args.overwrite:
            command.append("--overwrite")
        print("+", " ".join(command))
        subprocess.run(command, check=True)
        for name in ("config.json", "tokens.txt", "keywords.txt"):
            shutil.copyfile(staging / name, output.parent / name)
    print(f"wrote {output} ({len(tensors)} tensors)")


if __name__ == "__main__":
    main()
