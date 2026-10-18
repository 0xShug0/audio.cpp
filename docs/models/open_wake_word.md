# OpenWakeWord

Experimental wake-word detection using the official OpenWakeWord speech embedding
CNN and Alexa classifier. Offline and streaming inference use the same 80 ms
audio cadence. Input is mono PCM audio at 16 kHz.

## Conversion

Download `melspectrogram.onnx`, `embedding_model.onnx` and `alexa_v0.1.onnx` from
the [official model release](https://github.com/dscripka/openWakeWord/releases/tag/v0.5.1).
The converter also requires an OpenWakeWord source checkout and Python packages
`numpy`, `onnx`, `onnxruntime`, `scipy`, `scikit-learn`, and `gguf`.

```bash
python tests/open_wake_word/convert_gguf.py \
  --source-repo <openWakeWord-checkout> --models <onnx-model-directory> \
  --output <model-directory>/open-wake-word-alexa-f32.gguf
```

The GGUF contains weights, the v1 model spec, configuration, frontend coefficients
and fixed initial embedding history. The initial history is the official
initializer's output with seed zero, cached during conversion. This gives graph
and native sessions identical initial state without repeated startup inference.

## Inference

```bash
audiocpp_cli --task wake --family open_wake_word \
  --model <model-directory>/open-wake-word-alexa-f32.gguf \
  --backend cpu --audio <audio.wav> --segments-out detections.json --log --metrics
```

Add `--mode streaming` for streaming input. Omit `--metrics` in streaming mode.
`threshold` is a request option in `[0,1]`, default `0.5`. A rising probability
above the threshold produces a speech segment labelled `Alexa`. The first five
80 ms predictions are suppressed, matching the upstream initialization policy.

## ARM CPU

Build with `-DENGINE_ENABLE_NEON=ON` (default OFF) to enable the native path.
Add `--session-option 'open_wake_word.exec_plan={"embedding":{"impl":"neon"}}'` to run the complete embedding
CNN with the NEON library. The classifier keeps its existing graph. Omission,
an empty plan, or `embedding.impl=default/ggml` preserves the graph path. Explicit
NEON requires an AArch64 CPU session. No artifacts or params are accepted.
Frontend and detection policy remain unchanged.

The shared workspace is reused across all convolution layers. No per-request
weight packing or per-layer worker creation is needed. Numerical and performance
probes are under `tests/open_wake_word`; they use the production runtime.
