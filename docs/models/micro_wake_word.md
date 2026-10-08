# microWakeWord

This runtime executes trained microWakeWord MixedNet TFLite models after conversion to GGUF. It uses the same fixed-point TensorFlow Lite Micro speech frontend as the training project: 16 kHz mono PCM, a 30 ms window, a 10 ms step, and 40 PCAN-normalized filterbank channels.

The default package is `MicroWakeWord-GGUF/micro-wake-word-okay-nabu-f32.gguf`
under `audio-cpp/audio.cpp-gguf`. It is self-contained, including the model
configuration and spec. Only F32 is packaged.

The default [Okay Nabu checkpoint](https://github.com/esphome/micro-wake-word-models/blob/main/models/v2/okay_nabu.json)
is Apache-2.0 licensed. Upstream explicitly includes the trained weights and
manifests in its [license scope](https://github.com/esphome/micro-wake-word-models#license).
Retain its LICENSE and [NOTICE](https://github.com/esphome/micro-wake-word-models/blob/main/NOTICE)
when redistributing; other converted models may have different licenses.

Convert a model with its published manifest:

```bash
conda run -n qwen3-tts python tools/models/convert_micro_wake_word_gguf.py \
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

## Optional NEON network

The default network remains the existing GGML implementation on the session
backend. On AArch64, build with `-DENGINE_ENABLE_NEON=ON` and select the native
stateful CNN with:

```bash
--session-option 'micro_wake_word.exec_plan={"network":{"impl":"neon"}}'
```

The NEON network runs on CPU, uses the session thread count, and works in both
offline and streaming modes. The fixed-point frontend, input quantization,
activation quantization boundaries, recurrent-state updates, probability
smoothing and detection thresholds retain their existing behavior. The native
network supports batch-one MixedNet manifests with unit spatial width and
depthwise multiplier one.

`impl: "ggml"` or `impl: "default"` retains the original path. The component
accepts no artifacts or params. Existing GGUFs can use the new session option
without reconversion, including when loading their embedded spec.
