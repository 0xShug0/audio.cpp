# Coqui XTTS v2

XTTS v2 is a multilingual, zero-shot voice-cloning model. The audio.cpp port
runs the conditioning encoder, Perceiver resampler, 30-layer GPT-2 acoustic
token generator, ResNet speaker encoder, and speaker-conditioned HiFiGAN
decoder natively through ggml. Python is needed only to convert the original
PyTorch checkpoint.

The model supports `en`, `es`, `fr`, `de`, `it`, `pt`, `pl`, `tr`, `ru`, `nl`,
`cs`, `ar`, `zh-cn`, `ja`, `hu`, `ko`, and `hi`. Output is mono 24 kHz audio.
A clean reference recording of at least three seconds is required.

```sh
audio.cpp -m models/XTTS-v2-GGUF/xtts-v2-q8_0.gguf \
  --task clone \
  --voice-ref reference.wav \
  --language en \
  -p "This voice was synthesized locally by audio.cpp." \
  --seed 42 \
  -o xtts-v2.wav
```

The default generation settings match Coqui's published XTTS v2 configuration:
temperature `0.75`, top-k `50`, top-p `0.85`, and repetition penalty `5.0`.
Use `--speed` to adjust speaking rate without changing pitch.

## Conversion

Download the official `coqui/XTTS-v2` snapshot, then run:

```sh
python tools/community_models/convert_xtts_v2.py \
  --model-dir /path/to/XTTS-v2 \
  --output-dir /tmp/xtts-v2-staging \
  --run-converter build/bin/audiocpp_gguf \
  --type q8_0
```

The converter excludes optimizer, scaler, and other training-only state. It
also materializes the effective weights of the PyTorch weight-normalized
HiFiGAN layers. Convolution tensors, token embeddings, and the sampling head
remain F16 in the mixed Q8 package to protect synthesis quality.

## License

The XTTS v2 weights and their outputs are licensed under the Coqui Public Model
License 1.0.0 for non-commercial use only. The converted package includes the
original `LICENSE.txt`; downstream redistribution must keep the license or its
URL with the model and its outputs. The audio.cpp source changes retain the
repository's source-code license.

