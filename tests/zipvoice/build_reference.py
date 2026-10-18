#!/usr/bin/env python3
"""Generate parity references from an upstream ZipVoice checkout and fixed inputs.

No downloads or text frontend are needed. Run with the upstream Python environment:
  python build_reference.py --repo /path/to/ZipVoice --model-dir /path/to/zipvoice_distill \
      --fixture moonlight_distill_case.npz --output reference.npz [--vocos model.safetensors]

Hooks capture actual masked module outputs, never a hand-recomputed approximation.
The supplied fixture's text_condition/x1 are retained as independent golden outputs.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--vocos", type=Path)
    args = parser.parse_args()
    sys.path.insert(0, str(args.repo.resolve()))
    from zipvoice.models.zipvoice_distill import ZipVoiceDistill
    from zipvoice.utils.feature import VocosFbank

    torch.set_num_threads(4)
    config = json.loads((args.model_dir / "model.json").read_text())
    vocab = dict(line.rstrip("\n").split("\t") for line in
                 (args.model_dir / "tokens.txt").read_text().splitlines())
    model = ZipVoiceDistill(**config["model"], vocab_size=len(vocab), pad_id=int(vocab["_"]))
    state = torch.load(args.model_dir / "model.pt", map_location="cpu", weights_only=False)["model"]
    model.load_state_dict(state, strict=True)
    model.eval()
    with np.load(args.fixture) as fixture:
        outputs = {key: fixture[key] for key in fixture.files}

    def capture(name):
        def hook(module, inputs, output):
            outputs[name] = output.detach().clone().cpu().numpy().astype(np.float32)
        return hook

    hooks = []
    for i, layer in enumerate(model.text_encoder.encoders[0].layers):
        hooks.append(layer.register_forward_hook(capture(f"layer{i}")))
    first = model.text_encoder.encoders[0].layers[0]
    for name in ("self_attn_weights", "feed_forward1", "nonlin_attention", "self_attn1",
                 "conv_module1", "feed_forward2", "bypass_mid", "self_attn2",
                 "conv_module2", "feed_forward3", "norm"):
        hooks.append(dict(first.named_modules())[name].register_forward_hook(capture(name)))
    for name, key in (("nonlin_attention.in_proj", "na_h_ref"),
                      ("nonlin_attention.identity1", "na_gated_ref"),
                      ("nonlin_attention.identity2", "na_y2_ref"),
                      ("nonlin_attention.identity3", "na_mul_ref")):
        hooks.append(dict(first.named_modules())[name].register_forward_hook(capture(key)))
    with torch.inference_mode():
        tokens = outputs["tokens"].tolist()
        prompt_tokens = outputs["prompt_tokens"].tolist()
        embed, _ = model.forward_text_embed([prompt_tokens[0] + tokens[0]])
        outputs["embed"] = embed.numpy()
        for hook in hooks:
            hook.remove()
        tc, mask = model.forward_text_inference_ratio_duration(
            tokens=tokens, prompt_tokens=prompt_tokens,
            prompt_features_lens=torch.from_numpy(outputs["prompt_features_lens"]), speed=1.0)
        np.testing.assert_allclose(tc.numpy(), outputs["text_condition"], atol=2e-6, rtol=2e-5)
        # Odd and even lengths exercise repeat-last padding and downsampling.
        for length in (17, 32, outputs["x0"].shape[1]):
            for step, time in enumerate((0.0, 0.6)):
                key = f"velocity_{length}_{step}"
                result = model.forward_fm_decoder(
                    t=torch.tensor(time), xt=torch.from_numpy(outputs["x0"][:, :length]),
                    text_condition=tc[:, :length],
                    speech_condition=torch.from_numpy(outputs["speech_condition"][:, :length]),
                    padding_mask=mask[:, :length], guidance_scale=torch.tensor(3.0))
                outputs[key] = result.numpy()
        # Different conditions in each batch catch accidental head-0/batch-0 reuse.
        length = 17
        batch_x = torch.from_numpy(outputs["x0"][:, :length]).repeat(2, 1, 1)
        batch_text = torch.cat([torch.zeros_like(tc[:, :length]), tc[:, :length]])
        batch_speech = torch.from_numpy(outputs["speech_condition"][:, :length]).repeat(2, 1, 1)
        outputs["batch_x"] = batch_x.numpy()
        outputs["batch_text"] = batch_text.numpy()
        outputs["batch_speech"] = batch_speech.numpy()
        outputs["batch_velocity"] = model.forward_fm_decoder(
            t=torch.tensor(0.6), xt=batch_x, text_condition=batch_text,
            speech_condition=batch_speech, padding_mask=mask[:, :length].repeat(2, 1),
            guidance_scale=torch.tensor(3.0)).numpy()
        # Broadband input keeps quiet high-frequency bins numerically well conditioned.
        wav = torch.from_numpy(np.random.default_rng(123).normal(0, 0.1, 4096).astype(np.float32))
        outputs["fbank_wav"] = wav.numpy()
        outputs["fbank_mel"] = VocosFbank().extract(wav, sampling_rate=24000).numpy()
        if args.vocos:
            from safetensors.torch import load_file
            from vocos.models import VocosBackbone
            from vocos.heads import ISTFTHead
            backbone = VocosBackbone(input_channels=100, dim=512, intermediate_dim=1536, num_layers=8).eval()
            head = ISTFTHead(dim=512, n_fft=1024, hop_length=256, padding="center").eval()
            weights = (load_file(str(args.vocos)) if args.vocos.suffix == ".safetensors"
                       else torch.load(args.vocos, map_location="cpu", weights_only=True))
            backbone.load_state_dict({k.removeprefix("backbone."): v for k, v in weights.items() if k.startswith("backbone.")})
            head.load_state_dict({k.removeprefix("head."): v for k, v in weights.items() if k.startswith("head.")})
            # Skip the leading silence in the reference recording.
            mel = torch.from_numpy(outputs["prompt_features"][:, 32:64]) / 0.1
            outputs["vocos_mel"] = mel.numpy()
            outputs["vocos_audio"] = head(backbone(mel.transpose(1, 2))).numpy()
            assert np.sqrt(np.mean(outputs["vocos_audio"] ** 2)) > 1e-3, "Vocos fixture must contain speech"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, **outputs)
    print(f"Wrote {args.output}: {len(outputs)} arrays")


if __name__ == "__main__":
    main()
