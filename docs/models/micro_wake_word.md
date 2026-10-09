# microWakeWord

This runtime executes trained microWakeWord MixedNet TFLite models after conversion to GGUF. It uses the same fixed-point TensorFlow Lite Micro speech frontend as the training project: 16 kHz mono PCM, a 30 ms window, a 10 ms step, and 40 PCAN-normalized filterbank channels.

The default package is `MicroWakeWord-GGUF/micro-wake-word-okay-nabu-f32.gguf`
under `audio-cpp/audio.cpp-gguf`. It is self-contained, including the model
configuration and spec. Only F32 is packaged.

Convert a model with its published manifest:

```bash
conda run -n qwen3-tts python tools/models/micro_wake_word/convert_tflite_to_gguf.py \
  model.tflite model.gguf --manifest model.json
```

The converter accepts the stateful MixedNet operator set emitted by microWakeWord. It rejects different TFLite graphs explicitly. Converted weights use F32, while the runtime preserves the TFLite activation quantization boundaries and streaming state.

Run an audio file:

```bash
build/debug/bin/audiocpp_cli --family micro_wake_word --task wake \
  --model model.gguf --audio input.wav --log --log-file logs/micro_wake_word/run.log
```

Detections are returned as speech segments. `text` contains the wake phrase, `confidence` contains the smoothed probability, and the sample span identifies the feature block that triggered the detection. Request options `threshold` and `sliding_window_size` override the values copied from the model manifest.

Use `--mode streaming` for incremental audio.
