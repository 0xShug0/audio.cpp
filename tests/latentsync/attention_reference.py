"""Checkpoint-backed spatial and temporal U-Net block parity fixtures."""

import argparse
import json
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=3)
    args = parser.parse_args()
    sys.path.insert(0, str(args.reference_dir.resolve()))
    from latentsync.models.attention import Transformer3DModel
    from latentsync.models.motion_module import TemporalTransformer3DModel

    torch.manual_seed(42)
    with safe_open(args.weights, framework="pt", device="cpu") as source:
        for level, channels in enumerate((320, 640, 1280)):
            image = torch.randn(2, channels, args.frames, 4, 4, device="cuda")
            audio = torch.randn(2, args.frames, 50, 384, device="cuda")
            for component in ("spatial", "temporal"):
                if component == "spatial":
                    prefix = f"unet.down_blocks.{level}.attentions.0"
                    model = Transformer3DModel(
                        num_attention_heads=8, attention_head_dim=channels // 8,
                        in_channels=channels, cross_attention_dim=384, add_audio_layer=True)
                else:
                    prefix = f"unet.down_blocks.{level}.motion_modules.0.temporal_transformer"
                    model = TemporalTransformer3DModel(
                        in_channels=channels, num_attention_heads=8, attention_head_dim=channels // 8,
                        num_layers=1, temporal_position_encoding=True)
                state = {key[len(prefix) + 1:]: source.get_tensor(key).float()
                         for key in source.keys() if key.startswith(prefix + ".")}
                model.load_state_dict(state, strict=True)
                model = model.cuda().eval()
                with torch.inference_mode():
                    result = model(image, audio).sample if component == "spatial" else model(image)
                directory = args.output / f"{component}-{channels}"
                directory.mkdir(parents=True, exist_ok=True)
                (directory / "config.json").write_text(json.dumps({"prefix": prefix, "frames": args.frames}))
                link = directory / "weights.safetensors"
                if not link.exists():
                    link.symlink_to(args.weights.resolve())
                # Native spatial convolutions operate on folded batch/frame axes.
                packed_input = image.permute(0, 2, 1, 3, 4).flatten(0, 1).contiguous().cpu()
                packed_output = result.permute(0, 2, 1, 3, 4).flatten(0, 1).contiguous().cpu()
                save_file({"input": packed_input, "output": packed_output,
                           "audio": audio.flatten(0, 1).contiguous().cpu()}, directory / "reference.safetensors")
                print(f"{component}-{channels}: {tuple(packed_output.shape)}", flush=True)
                del model, result


if __name__ == "__main__":
    main()
