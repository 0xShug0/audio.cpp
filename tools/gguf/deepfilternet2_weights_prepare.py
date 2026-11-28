#!/usr/bin/env python3
"""Convert official DeepFilterNet2 checkpoint weights to framework safetensors."""

from __future__ import annotations

import argparse
import configparser
import json
import tempfile
import zipfile
from pathlib import Path

import torch
from safetensors.torch import save_file


SAMPLE_RATE = 48000
FFT_SIZE = 960
HOP_SIZE = 480
NB_ERB = 32
NB_DF = 96
DF_ORDER = 5


def read_config(config_path: Path) -> tuple[str, dict[str, object]]:
    text = config_path.read_text(encoding="utf-8")
    parser = configparser.ConfigParser()
    parser.read_string(text)
    model = parser["train"].get("model")
    if model != "deepfilternet2":
        raise RuntimeError(f"DeepFilterNet2 converter received model={model!r}")
    df = parser["df"]
    deepfilter = parser["deepfilternet"]
    expected = {
        "df.sr": SAMPLE_RATE,
        "df.fft_size": FFT_SIZE,
        "df.hop_size": HOP_SIZE,
        "df.nb_erb": NB_ERB,
        "df.nb_df": NB_DF,
        "deepfilternet.df_order": DF_ORDER,
        "deepfilternet.df_lookahead": 2,
        "deepfilternet.conv_ch": 64,
        "deepfilternet.emb_hidden_dim": 256,
        "deepfilternet.emb_num_layers": 3,
        "deepfilternet.df_hidden_dim": 256,
        "deepfilternet.df_num_layers": 2,
        "deepfilternet.gru_type": "squeeze",
        "deepfilternet.linear_groups": 8,
        "deepfilternet.enc_concat": True,
        "deepfilternet.df_output_layer": "groupedlinear",
        "deepfilternet.dfop_method": "df",
        "deepfilternet.conv_kernel": "1,3",
        "deepfilternet.conv_kernel_inp": "3,3",
        "deepfilternet.df_pathway_kernel_size_t": 5,
        "deepfilternet.df_n_iter": 1,
    }
    actual: dict[str, object] = {
        "df.sr": df.getint("sr"),
        "df.fft_size": df.getint("fft_size"),
        "df.hop_size": df.getint("hop_size"),
        "df.nb_erb": df.getint("nb_erb"),
        "df.nb_df": df.getint("nb_df"),
        "deepfilternet.df_order": deepfilter.getint("df_order"),
        "deepfilternet.df_lookahead": deepfilter.getint("df_lookahead"),
        "deepfilternet.conv_ch": deepfilter.getint("conv_ch"),
        "deepfilternet.emb_hidden_dim": deepfilter.getint("emb_hidden_dim"),
        "deepfilternet.emb_num_layers": deepfilter.getint("emb_num_layers"),
        "deepfilternet.df_hidden_dim": deepfilter.getint("df_hidden_dim"),
        "deepfilternet.df_num_layers": deepfilter.getint("df_num_layers"),
        "deepfilternet.gru_type": deepfilter.get("gru_type"),
        "deepfilternet.linear_groups": deepfilter.getint("linear_groups"),
        "deepfilternet.enc_concat": deepfilter.getboolean("enc_concat"),
        "deepfilternet.df_output_layer": deepfilter.get("df_output_layer"),
        "deepfilternet.dfop_method": deepfilter.get("dfop_method"),
        "deepfilternet.conv_kernel": deepfilter.get("conv_kernel"),
        "deepfilternet.conv_kernel_inp": deepfilter.get("conv_kernel_inp"),
        "deepfilternet.df_pathway_kernel_size_t": deepfilter.getint("df_pathway_kernel_size_t"),
        "deepfilternet.df_n_iter": deepfilter.getint("df_n_iter"),
    }
    for key, expected_value in expected.items():
        if actual[key] != expected_value:
            raise RuntimeError(f"unsupported DeepFilterNet2 config {key}: {actual[key]!r}")
    manifest = {
        "architecture": "DeepFilterNet2",
        "sample_rate": SAMPLE_RATE,
        "fft_size": FFT_SIZE,
        "hop_size": HOP_SIZE,
        "erb_bands": NB_ERB,
        "df_bins": NB_DF,
        "df_order": DF_ORDER,
        "df_lookahead": 2,
        "channels": 64,
        "hidden_size": 256,
        "emb_layers": 3,
        "df_layers": 2,
        "gru_type": "squeeze",
        "linear_groups": 8,
        "enc_concat": True,
        "df_output_layer": "groupedlinear",
        "dfop_method": "df",
    }
    return text, manifest


def extract_model_archive(archive: Path, output_root: Path) -> Path:
    if not archive.is_file():
        raise RuntimeError(f"DeepFilterNet2 archive does not exist: {archive}")
    with zipfile.ZipFile(archive) as zip_file:
        zip_file.extractall(output_root)
    model_dir = output_root / "DeepFilterNet2"
    checkpoint = model_dir / "checkpoints" / "model_96.ckpt.best"
    config = model_dir / "config.ini"
    if not checkpoint.is_file() or not config.is_file():
        raise RuntimeError("DeepFilterNet2 archive is missing checkpoint or config.ini")
    return model_dir


def load_named_weights(checkpoint_path: Path) -> dict[str, torch.Tensor]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise RuntimeError("DeepFilterNet2 checkpoint is not a state_dict")
    tensors: dict[str, torch.Tensor] = {}
    for name, tensor in checkpoint.items():
        if not torch.is_tensor(tensor):
            raise RuntimeError(f"DeepFilterNet2 checkpoint entry is not a tensor: {name}")
        if tensor.dtype == torch.float32:
            tensors[name] = tensor.detach().cpu().contiguous().clone()
        elif tensor.dtype == torch.int64 and name.endswith(".num_batches_tracked"):
            continue
        else:
            raise RuntimeError(f"unsupported DeepFilterNet2 tensor dtype for {name}: {tensor.dtype}")
    required = (
        "enc.erb_conv0.1.weight",
        "enc.df_conv0.1.weight",
        "enc.emb_gru.gru.weight_ih_l0",
        "erb_dec.convt2.0.weight",
        "df_dec.df_convp.1.weight",
        "df_dec.df_out.0.weight",
        "mask.erb_inv_fb",
    )
    missing = [name for name in required if name not in tensors]
    if missing:
        raise RuntimeError("DeepFilterNet2 checkpoint is missing required tensors: " + ", ".join(missing))
    return tensors


def convert(checkpoint_archive: Path, output_dir: Path) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        model_dir = extract_model_archive(checkpoint_archive, Path(tmp))
        config_text, manifest = read_config(model_dir / "config.ini")
        tensors = load_named_weights(model_dir / "checkpoints" / "model_96.ckpt.best")

    output_dir.mkdir(parents=True, exist_ok=True)
    save_file(
        tensors,
        str(output_dir / "deepfilternet2.safetensors"),
        metadata={
            "source": "https://github.com/Rikorose/DeepFilterNet",
            "license": "MIT OR Apache-2.0",
            "architecture": "DeepFilterNet2",
            "config_ini": config_text,
            "architecture_json": json.dumps(manifest, separators=(",", ":"), sort_keys=True),
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-archive", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("assets/framework/audio_utilities/deepfilternet2"))
    args = parser.parse_args()
    convert(args.checkpoint_archive, args.output_dir)


if __name__ == "__main__":
    main()
