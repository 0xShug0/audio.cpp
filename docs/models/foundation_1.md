# Foundation-1

[Foundation-1](https://huggingface.co/RoyalCities/Foundation-1) generates stereo
instrument samples and musical loops from structured text prompts. This is an
experimental, independent family. It uses the framework T5 encoder, DiT and
Oobleck VAE runtimes, with DDIM sampling. Inference needs no Python installation.

## Convert

Download `Foundation_1.safetensors`, `model_config.json` and `LICENSE.md` from the
original model repository into `models/Foundation-1`. Download T5-base's
`model.safetensors`, `config.json`, `tokenizer.json` and `spiece.model` into
`models/t5-base`. Build `audiocpp_gguf`, then run:

```bash
python tools/gguf/foundation_1_convert.py \
  --source models/Foundation-1 --text-encoder models/t5-base \
  --type f16 --output models/Foundation-1-GGUF/foundation-1-f16.gguf \
  --log convert-foundation-1.log
```

Use `--type q8_0` for the smaller transformer package. VAE and numeric
conditioning parameters retain F32 storage. The GGUF embeds the configuration,
tokenizer and schema-v1 contract; the unused T5 decoder is excluded.

## Run

```bash
audiocpp_cli --task gen --family foundation_1 \
  --model models/Foundation-1-GGUF/foundation-1-f16.gguf \
  --backend cuda --threads 8 \
  --text "Keys, Grand Piano, Warm, Clean, gentle melody, C major, 100 BPM, 4 bars" \
  --request-option duration_sec=10 --request-option num_steps=100 \
  --request-option guidance_scale=7 --seed 42 --out piano.wav --log
```

Request options are `duration_sec` (0.1–20, default 10), `num_steps` (default 100),
`guidance_scale` (default 7), `negative_prompt` and `seed` (default 42).
Guidance 1 runs a single conditional branch. The initial port provides offline
text-to-audio generation, with stereo 44.1 kHz output. Requested duration is
rounded up internally to a whole VAE frame and trimmed on output.

The session retains compatible graphs and weights across requests and replaces
shape-dependent graphs when needed. Numerical differences from Python are
expected; matching seeds do not guarantee identical waveforms across backends
or package precisions.

## License

The checkpoint is governed by its upstream Stability AI Community License;
read [the model's license](https://huggingface.co/RoyalCities/Foundation-1/blob/main/LICENSE.md)
and its referenced base-model terms before use or redistribution. T5-base is
Apache-2.0 licensed. These weight licenses are separate from audio.cpp's code license.
