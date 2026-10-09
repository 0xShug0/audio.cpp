#!/usr/bin/env python3
"""Package an official mispeech CED checkpoint with complete AudioSet labels."""

import argparse
import csv
import json
from pathlib import Path
import shutil
import subprocess
import tempfile

import torchaudio
from safetensors.torch import load_file, save_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--labels-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--converter", type=Path, default=Path("build/debug/bin/audiocpp_gguf"))
    parser.add_argument("--type", default="orig")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    source = args.source.resolve()
    config = json.loads((source / "config.json").read_text())
    with args.labels_csv.open(newline="", encoding="utf-8") as file:
        labels = {row["index"]: row["display_name"] for row in csv.DictReader(file)}
    if set(labels) != {str(i) for i in range(config["outputdim"])}:
        raise ValueError("AudioSet labels must cover every classifier output")
    # HF's CedConfig replaces the serialized labels at load time. Some of the
    # serialized names are truncated at commas; preserve the actual CSV labels.
    config["id2label"] = labels
    config["label2id"] = {label: int(index) for index, label in labels.items()}
    processor = json.loads((source / "preprocessor_config.json").read_text())
    frontend = torchaudio.transforms.MelSpectrogram(
        sample_rate=processor["sampling_rate"], n_fft=processor["n_fft"],
        win_length=processor["win_size"], hop_length=processor["hop_size"],
        f_min=processor["f_min"], f_max=processor["f_max"],
        n_mels=processor["feature_size"], center=processor["center"],
    )
    tensors = load_file(source / "model.safetensors")
    tensors["frontend.window"] = frontend.spectrogram.window.contiguous()
    tensors["frontend.mel_filterbank"] = frontend.mel_scale.fb.T.contiguous()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="ced-gguf-") as directory:
        root = Path(directory)
        (root / "config.json").write_text(json.dumps(config, indent=2) + "\n")
        shutil.copyfile(source / "preprocessor_config.json", root / "preprocessor_config.json")
        save_file(tensors, root / "model.safetensors")
        command = [
            str(args.converter.resolve()), "--input", str(root / "model.safetensors"),
            "--root", str(root), "--family", "ced", "--model-spec",
            str(Path(__file__).resolve().parents[3] / "model_specs/ced.json"),
            "--type", args.type, "--output", str(args.output.resolve()),
        ]
        if args.overwrite:
            command.append("--overwrite")
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
