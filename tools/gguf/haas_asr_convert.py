#!/usr/bin/env python3
"""Convert Haas's NeMo archive for the Haas family sharing Parakeet-TDT.

The FastConformer and joint tensors retain the standard Parakeet layout. Only
the registered Haas family selects its KDA prediction network. No NeMo/fla installation
is needed for conversion: PyTorch, PyYAML, SentencePiece, tokenizers,
transformers and safetensors suffice.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile
import tempfile

SOURCE_REPO = "third-intelligence/Haas-ja-v1.0"
SOURCE_REVISION = "660728d39019e5e065046154beca44ac9a66a540"


def prepare(checkpoint: Path, output: Path) -> None:
    import numpy as np
    import torch
    import yaml
    import sentencepiece as spm
    from safetensors.numpy import save_file
    from tokenizers import AddedToken, Tokenizer, decoders, models, normalizers, pre_tokenizers
    from sentencepiece import sentencepiece_model_pb2
    from transformers.tokenization_utils_base import generate_merges

    if (output / "model.safetensors").exists():
        raise FileExistsError("staging already contains model.safetensors")
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="haas-convert-") as work:
        work = Path(work)
        with tarfile.open(checkpoint, "r:*") as archive:
            archive.extractall(work, filter="data")
        config = yaml.safe_load((work / "model_config.yaml").read_text())
        decoder, encoder, pre = config["decoder"], config["encoder"], config["preprocessor"]
        pred = decoder["prednet"]
        if decoder["_target_"] != "haas_asr.prednet.KDATransducerDecoder" or pred["kda_expand_v"] != 1:
            raise ValueError("expected Haas KDA predictor with equal key/value head dimensions")
        if encoder["self_attention_model"] != "rel_pos" or encoder["use_bias"] or not encoder["xscaling"]:
            raise ValueError("encoder does not match the Parakeet FastConformer contract")
        if pre["normalize"] != "per_feature" or pre["sample_rate"] != 16000:
            raise ValueError("unsupported Parakeet frontend configuration")
        state = torch.load(work / "model_weights.ckpt", map_location="cpu", weights_only=True)
        tensors = {}
        for name, tensor in state.items():
            if name.startswith("preprocessor.") or name.endswith(".num_batches_tracked"):
                continue
            target = name
            if name.startswith("encoder.pre_encode."):
                target = name.replace("encoder.pre_encode.conv.", "encoder.subsampling.layers.").replace(
                    "encoder.pre_encode.out.", "encoder.subsampling.linear.")
            elif name.startswith("encoder.layers."):
                for a, b in [("linear_q", "q_proj"), ("linear_k", "k_proj"), ("linear_v", "v_proj"),
                             ("linear_out", "o_proj"), ("linear_pos", "relative_k_proj"),
                             ("pos_bias_u", "bias_u"), ("pos_bias_v", "bias_v")]:
                    target = target.replace(".self_attn." + a, ".self_attn." + b)
                target = target.replace(".conv.batch_norm.", ".conv.norm.")
            elif name == "decoder.prediction.embed.weight":
                target = "decoder.embedding.weight"
            elif name.startswith("decoder.prediction.dec_rnn."):
                target = name.replace("decoder.prediction.dec_rnn.", "decoder.kda.")
            elif name.startswith("joint.pred."):
                target = name.replace("joint.pred.", "decoder.decoder_projector.")
            elif name.startswith("joint.enc."):
                target = name.replace("joint.enc.", "encoder_projector.")
            elif name.startswith("joint.joint_net.2."):
                target = name.replace("joint.joint_net.2.", "joint.head.")
            else:
                raise ValueError("unmapped inference tensor: " + name)
            if target in tensors:
                raise ValueError("duplicate mapped tensor: " + target)
            tensors[target] = np.ascontiguousarray(tensor.float().numpy())
        # The existing Parakeet loader clamps variance before adding epsilon.
        # Fold its inverse correction into gamma to preserve NeMo's var+eps.
        for i in range(encoder["n_layers"]):
            prefix = f"encoder.layers.{i}.conv.norm."
            v = tensors[prefix + "running_var"]
            tensors[prefix + "weight"] *= np.sqrt((np.maximum(v, 1e-5) + 1e-5) / (v + 1e-5))
        spm_path = next(work.glob("*_tokenizer.model"))
        processor = spm.SentencePieceProcessor(model_file=str(spm_path))
        vocab_size = processor.get_piece_size() + 1
        # Reuse Transformers' SentencePiece BPE merge extraction, including the
        # trained normalization map and source piece scores.
        proto = sentencepiece_model_pb2.ModelProto()
        proto.ParseFromString(spm_path.read_bytes())
        vocab = {p.piece: i for i, p in enumerate(proto.pieces)}
        scores = {p.piece: p.score for p in proto.pieces}
        tokenizer = Tokenizer(models.BPE(vocab, generate_merges(vocab, scores),
            unk_token=proto.trainer_spec.unk_piece, byte_fallback=True, fuse_unk=True))
        tokenizer.normalizer = normalizers.Precompiled(proto.normalizer_spec.precompiled_charsmap)
        tokenizer.pre_tokenizer = pre_tokenizers.Metaspace(replacement="▁", prepend_scheme="always", split=False)
        tokenizer.decoder = decoders.Sequence([decoders.ByteFallback(), decoders.Metaspace(replacement="▁", prepend_scheme="always")])
        tokenizer.add_special_tokens([AddedToken("<blank>", special=True)])
        if tokenizer.get_vocab_size() != vocab_size:
            raise ValueError("converted tokenizer does not match the predictor embedding")
        tokenizer.save(str(output / "tokenizer.json"))
        runtime_config = {
            "model_type": "parakeet_tdt", "variant": "haas_ja_v1", "predictor_type": "kda",
            "vocab_size": vocab_size, "blank_token_id": vocab_size - 1, "pad_token_id": vocab_size - 1,
            "decoder_hidden_size": pred["pred_hidden"], "num_decoder_layers": pred["pred_rnn_layers"],
            "max_symbols_per_step": config["decoding"]["greedy"]["max_symbols"],
            "durations": config["decoding"]["durations"],
            "kda_config": {"num_heads": pred["kda_num_heads"], "head_dim": pred["kda_head_dim"],
                "conv_size": pred["kda_conv_size"], "final_norm": pred["kda_final_norm"]},
            "encoder_config": {"hidden_size": encoder["d_model"],
                "intermediate_size": encoder["d_model"] * encoder["ff_expansion_factor"],
                "num_hidden_layers": encoder["n_layers"], "num_attention_heads": encoder["n_heads"],
                "conv_kernel_size": encoder["conv_kernel_size"], "subsampling_factor": encoder["subsampling_factor"],
                "subsampling_conv_channels": encoder["subsampling_conv_channels"],
                "subsampling_conv_kernel_size": 3, "subsampling_conv_stride": 2,
                "max_position_embeddings": encoder["pos_emb_max_len"]}}
        (output / "config.json").write_text(json.dumps(runtime_config, indent=2) + "\n")
        processor_config = {"feature_extractor": {"sampling_rate": pre["sample_rate"], "feature_size": pre["features"],
            "n_fft": pre["n_fft"], "win_length": round(pre["window_size"] * pre["sample_rate"]),
            "hop_length": round(pre["window_stride"] * pre["sample_rate"]), "preemphasis": pre.get("preemph", 0.97)}}
        (output / "processor_config.json").write_text(json.dumps(processor_config, indent=2) + "\n")
        repo = Path(__file__).resolve().parents[2]
        spec = json.loads((repo / "model_specs/haas_asr.json").read_text())
        spec.update(display_name="Haas-ja-v1.0", languages=["ja"], description="Japanese FastConformer-TDT ASR with a Kimi Delta Attention predictor; uses the shared Parakeet-TDT implementation.")
        spec["ui"]["docs"] = ["docs/models/haas.md"]
        spec["packages"] = [p for p in spec["packages"] if p["id"].startswith("haas_ja_v1_")]
        for package in spec["packages"]:
            package["default"] = package["id"] == "haas_ja_v1_q8_0"
        spec["ui"]["recommended_package"] = "haas_ja_v1_q8_0"
        spec["sources"] = [s for s in spec["sources"] if s["format"] == "gguf"]
        (output / "model_spec.json").write_text(json.dumps(spec, ensure_ascii=False, indent=2) + "\n")
        with checkpoint.open("rb") as source:
            digest = hashlib.file_digest(source, "sha256").hexdigest()
        (output / "conversion.json").write_text(json.dumps({"source": SOURCE_REPO, "revision": SOURCE_REVISION,
            "checkpoint_sha256": digest, "batch_norm_gamma_correction": "sqrt((max(var,eps)+eps)/(var+eps))",
            "inference_tensors": len(tensors)}, indent=2) + "\n")
        save_file(tensors, output / "model.safetensors")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, help="Official Haas .nemo archive; required when preparing")
    parser.add_argument("--staging", required=True, type=Path)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--pack-only", action="store_true")
    parser.add_argument("--converter", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--type", choices=["f32", "f16", "q8_0"], default="q8_0")
    args = parser.parse_args()
    if args.prepare_only and args.pack_only:
        parser.error("--prepare-only and --pack-only are mutually exclusive")
    if not args.pack_only and args.source is None:
        parser.error("preparing requires --source")
    if not args.prepare_only and (args.converter is None or args.output is None):
        parser.error("packing requires --converter and --output")
    if not args.pack_only:
        prepare(args.source.resolve(), args.staging.resolve())
    if args.prepare_only:
        return
    stage = args.staging.resolve()
    command = [str(args.converter.resolve()), "--input", str(stage / "model.safetensors"), "--root", str(stage),
        "--family", "haas_asr", "--model-spec", str(stage / "model_spec.json"), "--type", args.type,
        "--keep-type", "*conv.norm.weight=f32", "--output", str(args.output.resolve())]
    for name in ["config.json", "processor_config.json", "tokenizer.json", "conversion.json"]:
        command += ["--sidecar", f"{stage / name}={name}"]
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
