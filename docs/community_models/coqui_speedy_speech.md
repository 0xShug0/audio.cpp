# Coqui SpeedySpeech

`coqui_speedy_speech` is a native port of Coqui's Apache-2.0 LJSpeech
SpeedySpeech checkpoint and its HiFi-GAN v2 vocoder. It performs feed-forward
English text-to-speech at 22.05 kHz without Python at runtime.

Native audio.cpp output samples: [default rate](../assets/coqui-speedy-speech-demo.mp4)
and [1.2x speaking rate](../assets/coqui-speedy-speech-demo-fast.mp4).

```bash
audiocpp_cli --task tts --family coqui_speedy_speech \
  --model ./Coqui-SpeedySpeech-LJSpeech-GGUF \
  --text "SpeedySpeech is running natively in audio.cpp." \
  --out speedy.wav
```

`--speaking-rate 1.2` speaks faster; values from `0.5` through `2.0`
are accepted. eSpeak-ng is required for English IPA phonemization.

## Conversion

Download Coqui's `tts_models/en/ljspeech/speedy-speech` and
`vocoder_models/en/ljspeech/hifigan_v2` checkpoints, then run:

```bash
uv run --with torch --with numpy --with safetensors \
  python tools/community_models/convert_coqui_speedy_speech.py \
  --acoustic-checkpoint speedy/model_file.pth \
  --acoustic-config speedy/config.json \
  --vocoder-checkpoint hifigan/model_file.pth \
  --vocoder-config hifigan/config.json \
  --output converted/model.safetensors \
  --output-config converted/config.json

audiocpp_gguf --input weights=converted/model.safetensors \
  --output converted/coqui-speedy-speech-ljspeech-f32.gguf \
  --type orig --family coqui_speedy_speech \
  --model-spec model_specs/coqui_speedy_speech.json --root converted
```

The converter removes training-only tensors, materializes BatchNorm inference
affines, and folds HiFi-GAN weight normalization. The initial port is English,
offline-only, and supports the exact released LJSpeech architecture.
