"""Convert official SAM Audio checkpoint and T5 into a self-contained GGUF."""

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import torch
import gguf
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import AutoTokenizer, T5EncoderModel
from huggingface_hub import hf_hub_download


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gguf-output", type=Path, required=True)
    repo = Path(__file__).resolve().parents[2]
    parser.add_argument("--converter", type=Path, default=repo / "build/debug/bin/audiocpp_gguf")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    config = json.loads((args.model / "config.json").read_text())
    state = torch.load(args.model / "checkpoint.pt", map_location="cpu", weights_only=True, mmap=True)
    save_file(state, str(args.output / "sam_audio.safetensors"))
    names = {name: "sam_audio.safetensors" for name in state}
    text_model = config["text_encoder"]["name"]
    encoder = T5EncoderModel.from_pretrained(text_model)
    text_state = {"text_encoder.model." + name: tensor.contiguous().clone()
                  for name, tensor in encoder.state_dict().items()}
    save_file(text_state, str(args.output / "t5.safetensors"))
    names.update({name: "t5.safetensors" for name in text_state})
    (args.output / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": names}, indent=2) + "\n")
    tokenizer = AutoTokenizer.from_pretrained(text_model)
    tokenizer.save_pretrained(args.output / "tokenizer")
    shutil.copyfile(hf_hub_download(text_model, "spiece.model"), args.output / "tokenizer/spiece.model")
    encoder.config.to_json_file(args.output / "t5_config.json")
    shutil.copyfile(args.model / "config.json", args.output / "config.json")
    print(f"Prepared {len(state)} checkpoint tensors and {len(text_state)} T5 tensors", flush=True)
    args.gguf_output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([
        str(args.converter), "--input", str(args.output / "model.safetensors.index.json"),
        "--root", str(args.output), "--output", str(args.gguf_output),
        "--type", "f32", "--family", "sam_audio", "--model-spec", str(repo / "model_specs/sam_audio.json"),
    ], check=True)
    reader = gguf.GGUFReader(str(args.gguf_output))
    ranks = reader.fields["audiocpp.tensor_ranks"].contents()
    shapes = reader.fields["audiocpp.tensor_shapes"].contents()
    logical_names = reader.fields["audiocpp.tensor_names"].contents()
    if len(reader.tensors) != len(names):
        raise RuntimeError("GGUF tensor count differs from source")
    cursor = 0
    for i, tensor in enumerate(reader.tensors):
        name = logical_names[i]
        with safe_open(args.output / names[name], framework="np") as source:
            original = source.get_tensor(name)
        if list(original.shape) != shapes[cursor:cursor + ranks[i]]:
            raise RuntimeError(f"GGUF shape differs: {name}")
        if hashlib.sha256(original.tobytes()).digest() != hashlib.sha256(tensor.data.tobytes()).digest():
            raise RuntimeError(f"GGUF tensor bytes differ: {name}")
        cursor += ranks[i]
    print(f"Verified all {len(reader.tensors)} GGUF tensors byte-identical", flush=True)


if __name__ == "__main__":
    main()
