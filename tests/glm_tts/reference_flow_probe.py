import json
import math
import pathlib
import sys

import torch

REFERENCE_ROOT = pathlib.Path(__file__).resolve().parents[3] / "_reference" / "GLM-TTS"
sys.path.insert(0, str(REFERENCE_ROOT))

from flow.flow import Flow


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: reference_flow_probe.py <model-path>")
    model_path = pathlib.Path(sys.argv[1])
    model = Flow(
        spkr_emb_adaLN=True,
        speech_token_cfg=False,
        remove_spkr_concat_condition=True,
        mel_dim=80,
        mel_framerate=50,
        input_frame_rate=25,
    )
    checkpoint = torch.load(
        model_path / "flow" / "flow.pt", map_location="cpu", weights_only=True
    )
    model.load_state_dict(checkpoint["model"] if "model" in checkpoint else checkpoint)
    model.eval()

    tokens = torch.tensor([[29252, 4906, 833, 15564, 12971]], dtype=torch.long)
    frames = int(tokens.shape[1] / model.input_frame_rate * model.mel_framerate)
    noise = torch.tensor(
        [math.sin((index + 1) * 0.017) for index in range(frames * 80)],
        dtype=torch.float32,
    ).reshape(1, frames, 80)
    speaker = torch.tensor(
        [[math.sin((index + 1) * 0.03125) for index in range(192)]],
        dtype=torch.float32,
    )
    speaker = torch.nn.functional.normalize(speaker, dim=1)
    condition = torch.zeros_like(noise)
    mask = torch.ones((1, frames), dtype=torch.bool)
    debug = {}

    def capture(name):
        def hook(_module, _args, output):
            value = output[0] if isinstance(output, tuple) else output
            if name not in debug:
                debug[name] = value.detach().flatten()[:8].tolist()

        return hook

    def capture_input(name):
        def hook(_module, args):
            if name not in debug:
                debug[name] = args[0].detach().flatten()[:8].tolist()

        return hook

    hooks = [
        model.estimator.time_embed.time_mlp[0].register_forward_hook(
            capture("time_mlp1")
        ),
        model.estimator.time_embed.register_forward_hook(capture("time")),
        model.estimator.text_emb_layer.register_forward_hook(capture("text")),
        model.estimator.emb_concator.proj.register_forward_hook(
            capture("projection")
        ),
        model.estimator.emb_concator.conv_pos_embed.register_forward_hook(
            capture("position")
        ),
        model.estimator.emb_concator.conv_pos_embed.conv1d[
            1
        ].register_forward_hook(capture("position_first")),
        model.estimator.emb_concator.register_forward_hook(capture("input")),
        model.estimator.transformer_blocks[0].register_forward_hook(
            capture("block0")
        ),
        model.estimator.transformer_blocks[0].attn_norm.register_forward_hook(
            capture("block0_norm")
        ),
        model.estimator.transformer_blocks[0].attn.register_forward_hook(
            capture("block0_attention")
        ),
        model.estimator.transformer_blocks[0].ff_norm.register_forward_pre_hook(
            capture_input("block0_after_attention")
        ),
        model.estimator.transformer_blocks[0].ff.register_forward_hook(
            capture("block0_ff")
        ),
        model.estimator.norm_out.register_forward_hook(capture("final_norm")),
    ]
    with torch.inference_mode():
        conditional = model.estimator(
            middle_point_btd=noise,
            condition_btd=condition,
            text=tokens,
            time_step_1d=torch.zeros(1),
            padding_mask_bt=mask,
            spkr_emb_bd=speaker,
        )
        unconditional = model.estimator(
            middle_point_btd=noise,
            condition_btd=condition,
            text=tokens,
            time_step_1d=torch.zeros(1),
            padding_mask_bt=mask,
            spkr_emb_bd=torch.zeros_like(speaker),
        )
        output = noise + 1.0 * (1.7 * conditional - 0.7 * unconditional)
    for hook in hooks:
        hook.remove()
    for name, values in debug.items():
        print(f"glm_flow_debug.{name}={values}", file=sys.stderr)
    print(
        json.dumps(
            {
                "frames": frames,
                "values": output.flatten()[:16].tolist(),
            },
            separators=(",", ":"),
        )
    )


if __name__ == "__main__":
    main()
