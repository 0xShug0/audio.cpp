#!/usr/bin/env python3
"""Official component generation and final-waveform reference for the C++ port."""

import argparse
import logging
from pathlib import Path
import time

import flax.traverse_util as traversal
import jax
import jax.numpy as jnp
import numpy as np
from safetensors import safe_open
from safetensors.numpy import save_file
import soundfile as sf

from magenta_rt import musiccoca
from magenta_rt.jax import model, spectrostream
import sequence_layers.jax as sl


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--variant", choices=("small", "base"), default="small")
    parser.add_argument("--prompt", default="disco funk")
    parser.add_argument("--frames", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=1.3)
    parser.add_argument("--compute-dtype", choices=("bf16", "f32"), default="bf16")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, handlers=[logging.FileHandler(args.log), logging.StreamHandler()], force=True)
    args.output.mkdir(parents=True, exist_ok=True)
    style = musiccoca.MusicCoCa(args.source / "resources/musiccoca")
    embedding = style.embed(args.prompt, use_mapper=True, seed=0)
    style_codes = style.tokenize(embedding).reshape(-1).astype(np.int32)
    config = model.get_model_class("mrt2_" + args.variant)()
    if args.compute_dtype == "f32":
        config.compute_dtype = jnp.float32
    decoder = config.depthformer_config().make()
    codec = spectrostream.stft_spectrostream_40ms_generic_48khz_stereo_config(rvq_truncation_level=12).make()
    checkpoint = args.source / "checkpoints" / f"mrt2_{args.variant}.safetensors"
    with safe_open(checkpoint, framework="numpy") as f:
        flat = {tuple(name.split("/")): jnp.asarray(f.get_tensor(name)) for name in f.keys()}
    parameters = traversal.unflatten_dict(flat)["params"]
    decoder_params = {"params": parameters["depthformer"]}
    codec_params = {"params": parameters["soundstream"]}
    controls = np.array([-1] * 129 + [20, 10, 2], dtype=np.int32)
    conditioning = np.concatenate((style_codes, controls)) + 7
    block = sl.Sequence.from_values(jnp.asarray(conditioning.reshape(1, 1, -1)))
    constants = {"temperature": jnp.array([args.temperature]), "top_k": jnp.array([40], dtype=jnp.int32)}
    rngs = {"params": jax.random.PRNGKey(42), "random": jax.random.PRNGKey(0)}

    @jax.jit
    def decoder_init(params):
        return decoder.apply(params, rngs=rngs, method=lambda module: module.get_sampler_sequence_layer().get_initial_state(
            1, block.channel_spec, training=False, constants={}))

    @jax.jit
    def decoder_step(params, state):
        return decoder.apply(params, rngs=rngs, method=lambda module: module.get_sampler_sequence_layer().step(
            block, state, training=False, constants=constants))

    @jax.jit
    def codec_init(params):
        return codec.apply(params, method=lambda module: module.get_codes_to_waveform_layer().get_initial_state(
            1, jax.ShapeDtypeStruct((12,), jnp.int32), training=False, constants={}))

    @jax.jit
    def codec_step(params, codes, state):
        return codec.apply(params, method=lambda module: module.get_codes_to_waveform_layer().step(
            sl.Sequence.from_values(codes), state, training=False, constants={}))

    start = time.perf_counter()
    decoder_state = decoder_init(decoder_params)
    codec_state = codec_init(codec_params)
    output_codes = []
    output_audio = []
    for _ in range(args.frames):
        unique_codes, decoder_state = decoder_step(decoder_params, decoder_state)
        codes = (unique_codes.values - 6) % 1024
        waveform, codec_state = codec_step(codec_params, codes, codec_state)
        output_codes.append(codes)
        output_audio.append(waveform.values)
    waveform = np.asarray(jnp.concatenate(output_audio, axis=1))[0]
    elapsed = time.perf_counter() - start
    codes = np.asarray(jnp.concatenate(output_codes, axis=1))[0].astype(np.int32)
    save_file({"codes": codes, "waveform": waveform, "style_codes": style_codes,
               "control_tokens": controls, "style_embedding": embedding.reshape(-1)}, args.output / "reference.safetensors")
    sf.write(args.output / "python.wav", np.clip(waveform * .5, -1, 1), 48000, subtype="PCM_16")
    logging.info("session.wall_ms=%.3f audio.duration_sec=%.3f", elapsed*1000, len(waveform)/48000)


if __name__ == "__main__":
    main()
