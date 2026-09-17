#!/usr/bin/env python3
"""Dump the YuE2 NAR reference tensors consumed by yue2_nar_parity_probe.

The probe feeds the engine the exact prefix, codec tokens and acoustic noise the
Python reference integrated, so any difference it reports is the port rather
than sampling RNG: ABC and semantic sampling both run greedy (top_k = 1).

Writes into --out-dir:
  prefix.i32        token prefix the acoustic flow conditions on
  codec.i32         codec tokens, one per acoustic frame
  noise.f32         [frames, 64] fp32 noise the reference integrated
  latents_ref.f32   [frames, 64] fp32 reference latents
  abc.txt           score the plan used (empty for cot=off and provided scores)
  metadata.json     seed / ode_steps / context / latent_dim / shapes

The noise draw mirrors yue2.nar.song_chunks (one CPU fp32 randn seeded with the
request seed), and steps/context come from the pipeline generation config, so
the probe can reproduce the same integration path.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch


def import_yue(reference_root: Path) -> None:
    sys.path.insert(0, str(reference_root / "src"))


def write_i32(path: Path, values) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.asarray(list(values), dtype="<i4").tofile(path)


def write_f32(path: Path, values) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.asarray(values, dtype="<f4").reshape(-1).tofile(path)


def resolve_vae(model: Path, vae: Path | None) -> Path:
    if vae is not None:
        return vae
    sibling = model.parent / "YuE2-Vae"
    if sibling.is_dir():
        return sibling
    raise SystemExit("pass --vae: no YuE2-Vae directory next to --model")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-root", type=Path, default=Path("reference/YuE"))
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--vae", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--style", default="Upbeat electronic pop with a driving bass line.")
    parser.add_argument("--lyrics", default="[verse]\nCity lights are calling out\n")
    parser.add_argument("--abc-file", type=Path, default=None,
                        help="External ABC score; skips ABC generation entirely.")
    parser.add_argument("--cot", default="full", choices=("off", "melody", "full"))
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--semantic-max-tokens", type=int, default=9000,
                        help="Cap codec frames: the reference NAR attention is quadratic in frames.")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    import_yue(args.reference_root.resolve())
    from yue2 import YuE2Pipeline
    from yue2.protocol import Sampling

    vae = resolve_vae(args.model, args.vae)
    abc_text = args.abc_file.read_text(encoding="utf-8") if args.abc_file is not None else None
    # Greedy everywhere: token choice carries no RNG, so the probe only has to
    # reproduce the acoustic flow.
    greedy_abc = Sampling(0.7, 0.9, 1, 1.005, 100, 32, 4096)
    greedy_semantic = Sampling(1.0, 0.95, 1, 1.2, 50, 200, args.semantic_max_tokens)

    with YuE2Pipeline.from_pretrained(str(args.model), vae=str(vae), device=args.device,
                                      progress=False) as pipe:
        plan = pipe.plan(args.style, args.lyrics, cot=args.cot, seed=args.seed, abc=abc_text,
                         abc_sampling=greedy_abc)
        semantic = pipe.generate_semantic(plan, sampling=greedy_semantic)
        steps = int(pipe.generation_config.ode_steps)
        context = int(pipe.generation_config.context)
        generator = torch.Generator(device="cpu").manual_seed(int(args.seed))
        noise = torch.randn((len(semantic.tokens), 64), dtype=torch.float32, device="cpu",
                            generator=generator)
        latents = pipe.synthesize(semantic)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_i32(args.out_dir / "prefix.i32", plan.prefix)
    write_i32(args.out_dir / "codec.i32", semantic.tokens)
    write_f32(args.out_dir / "noise.f32", noise.numpy())
    write_f32(args.out_dir / "latents_ref.f32", latents)
    (args.out_dir / "abc.txt").write_text(plan.abc or "", encoding="utf-8")
    metadata = {
        "seed": args.seed,
        "cot": args.cot,
        "abc_source": "provided" if abc_text is not None else ("none" if args.cot == "off" else "generated"),
        "ode_steps": steps,
        "context": context,
        "latent_dim": int(np.asarray(latents).shape[1]),
        "codec_frames": int(len(semantic.tokens)),
        "latent_frames": int(np.asarray(latents).shape[0]),
        "style": args.style,
        "lyrics": args.lyrics,
        "model": str(args.model),
        "vae": str(vae),
    }
    (args.out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
