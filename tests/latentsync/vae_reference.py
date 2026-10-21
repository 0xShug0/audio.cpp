"""Export deterministic SD VAE encoder/decoder references from the official demo."""

import argparse
import shutil
from pathlib import Path

import cv2
import torch
from diffusers import AutoencoderKL
from safetensors.torch import save_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vae-dir", type=Path, required=True)
    parser.add_argument("--native-weights", type=Path, required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    capture = cv2.VideoCapture(str(args.video))
    success, frame = capture.read()
    capture.release()
    if not success:
        raise RuntimeError("Could not read the reference video")
    frame = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), (512, 512))
    image = torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0).float().cuda() / 127.5 - 1
    vae = AutoencoderKL.from_pretrained(str(args.vae_dir), torch_dtype=torch.float32).cuda().eval()
    with torch.inference_mode():
        moments = vae.encode(image).latent_dist.parameters
        latent = moments[:, :4].contiguous()
        decoded = vae.decode(latent).sample
    for component, input_value, output_value in (
        ("vae-encoder", image, moments), ("vae-decoder", latent, decoded)
    ):
        directory = args.output / component
        directory.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(args.vae_dir / "config.json", directory / "config.json")
        link = directory / "weights.safetensors"
        if not link.exists():
            link.symlink_to(args.native_weights.resolve())
        save_file({"input": input_value.cpu().contiguous(), "output": output_value.cpu().contiguous()},
                  directory / "reference.safetensors")
        print(f"{component}: {tuple(input_value.shape)} -> {tuple(output_value.shape)}")


if __name__ == "__main__":
    main()
