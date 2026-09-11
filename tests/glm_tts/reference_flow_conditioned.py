#!/usr/bin/env python3

import argparse
import json
import math
import pathlib
import sys
from unittest.mock import patch

import torch

REFERENCE_ROOT = (
    pathlib.Path(__file__).resolve().parents[3] / "_reference" / "GLM-TTS"
)
sys.path.insert(0, str(REFERENCE_ROOT))

from flow.flow import Flow


REFERENCE_TOKENS = [
    29252, 4906, 833, 15564, 12971, 13808, 8457, 20137,
    7313, 30766, 5123, 2098, 891, 2464, 16935, 26908,
    20338, 286, 32542, 10092, 13792, 18489, 16200, 9895,
    3221, 29590, 5542, 29513, 25025, 9777, 29029, 21176,
    9781, 15068, 30891, 5324, 28699, 28591, 20197, 4599,
    11932, 4759, 19157, 10647, 4938, 8754, 30267, 31575,
    4731, 6326, 30991, 5040, 17687, 17687, 19635, 10780,
    21045, 5387, 11503, 13228, 13228, 6164, 12240,
]
GENERATED_TOKENS = [
    25568, 26211, 20216, 29473, 21992, 30971, 14699,
    25619, 20563, 20602, 22277, 9540, 25669, 21373,
    22055, 13228, 6387, 16809, 13228, 4503, 15102, 32001,
    31164, 26617, 6852, 32401, 27451, 5997, 5590, 18014,
    780, 26030, 17460, 15143, 18324, 5413, 23684, 28226,
    16826, 31440, 19490, 15302, 3693, 17870, 515, 32307,
    12991, 23343, 10209, 1118, 10442, 9769, 6294, 12860,
    10055, 3886, 7692,
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", type=pathlib.Path)
    parser.add_argument("--steps", type=int, default=10)
    args = parser.parse_args()

    model = Flow(
        spkr_emb_adaLN=True,
        speech_token_cfg=False,
        remove_spkr_concat_condition=True,
        mel_dim=80,
        mel_framerate=50,
        input_frame_rate=25,
    )
    checkpoint = torch.load(
        args.model_path / "flow" / "flow.pt",
        map_location="cpu",
        weights_only=True,
    )
    model.load_state_dict(
        checkpoint["model"] if "model" in checkpoint else checkpoint
    )
    model.eval()

    prompt_frames = 126
    prompt = torch.tensor(
        [
            math.sin((index + 1) * 0.011) * 0.5
            for index in range(prompt_frames * 80)
        ],
        dtype=torch.float32,
    ).reshape(1, prompt_frames, 80)
    speaker = torch.tensor(
        [[
            math.sin((index + 1) * 0.03125)
            for index in range(192)
        ]],
        dtype=torch.float32,
    )
    full_frames = int(
        (len(REFERENCE_TOKENS) + len(GENERATED_TOKENS)) / 25 * 50
    )
    noise = torch.tensor(
        [
            math.sin((index + 1) * 0.017)
            for index in range(full_frames * 80)
        ],
        dtype=torch.float32,
    ).reshape(1, full_frames, 80)

    original_randn_like = torch.randn_like

    def fixed_randn_like(value, *args, **kwargs):
        del args, kwargs
        if tuple(value.shape) == tuple(noise.shape):
            return noise.clone()
        return original_randn_like(value)

    with torch.inference_mode(), patch(
        "torch.randn_like", side_effect=fixed_randn_like
    ):
        output, _ = model.inference_with_cache(
            token=torch.tensor([GENERATED_TOKENS], dtype=torch.long),
            prompt_token=torch.tensor(
                [REFERENCE_TOKENS], dtype=torch.long
            ),
            prompt_feat=prompt,
            embedding=speaker,
            n_timesteps=args.steps,
        )
    frame_major = output.permute(0, 2, 1).contiguous().view(-1)
    print(
        json.dumps(
            {
                "frames": int(output.shape[-1]),
                "values": frame_major.tolist(),
            },
            separators=(",", ":"),
        )
    )


if __name__ == "__main__":
    main()
