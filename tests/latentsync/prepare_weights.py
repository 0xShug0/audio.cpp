"""Prepare LatentSync's original checkpoints for the native GGUF converter."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import torch
import cv2
import numpy as np
from omegaconf import OmegaConf
from safetensors import safe_open
from safetensors.torch import load_file, save_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--vae-dir", type=Path, required=True)
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--face-model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    index = {}

    checkpoint = torch.load(args.checkpoint_dir / "latentsync_unet.pt",
                            map_location="cpu", weights_only=True, mmap=True)
    unet = {"unet." + key: value.contiguous() for key, value in checkpoint["state_dict"].items()}
    save_file(unet, args.output_dir / "unet.safetensors")
    index.update({key: "unet.safetensors" for key in unet})
    del unet, checkpoint

    checkpoint = torch.load(args.checkpoint_dir / "whisper/tiny.pt",
                            map_location="cpu", weights_only=True)
    whisper_config = checkpoint["dims"]
    whisper = {"whisper." + key: value.contiguous()
               for key, value in checkpoint["model_state_dict"].items()
               if key.startswith("encoder.")}
    save_file(whisper, args.output_dir / "whisper.safetensors")
    index.update({key: "whisper.safetensors" for key in whisper})
    del whisper, checkpoint

    vae = {"vae." + key: value.contiguous()
           for key, value in load_file(args.vae_dir / "diffusion_pytorch_model.safetensors").items()}
    save_file(vae, args.output_dir / "vae.safetensors")
    index.update({key: "vae.safetensors" for key in vae})
    del vae

    for script, checkpoint, directory, config_name in [
        ("prepare_scrfd_weights.py", "det_10g.onnx", "detector", "detector_config.json"),
        ("prepare_landmark_weights.py", "2d106det.onnx", "landmarks", "landmark_config.json"),
    ]:
        destination = args.output_dir / directory
        subprocess.run([sys.executable, str(Path(__file__).with_name(script)),
                        "--checkpoint", str(args.face_model_dir / checkpoint),
                        "--output", str(destination)], check=True)
        (args.output_dir / config_name).write_bytes((destination / "config.json").read_bytes())
        shard = f"{directory}.safetensors"
        (destination / "weights.safetensors").replace(args.output_dir / shard)
        with safe_open(args.output_dir / shard, framework="pt") as weights:
            index.update({key: shard for key in weights.keys()})

    mask = cv2.imread(str(args.reference_dir / "latentsync/utils/mask.png"))
    if mask is None:
        raise ValueError("Cannot read LatentSync's face mask")
    mask = cv2.cvtColor(cv2.resize(mask, (512, 512), interpolation=cv2.INTER_LANCZOS4), cv2.COLOR_BGR2RGB)
    mask = np.ascontiguousarray(mask.transpose(2, 0, 1), dtype=np.float32) / 255
    save_file({"face.mask": torch.from_numpy(mask)}, args.output_dir / "mask.safetensors")
    index["face.mask"] = "mask.safetensors"

    config = {
        "model_type": "latentsync",
        "unet": OmegaConf.to_container(OmegaConf.load(args.reference_dir / "configs/unet/stage2_512.yaml"))["model"],
        "whisper": whisper_config,
        "vae": json.loads((args.vae_dir / "config.json").read_text()),
        "scheduler": json.loads((args.reference_dir / "configs/scheduler_config.json").read_text()),
    }
    (args.output_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    (args.output_dir / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": index}, indent=2) + "\n")
    print(f"Prepared {len(index)} tensors without dtype conversion")


if __name__ == "__main__":
    main()
