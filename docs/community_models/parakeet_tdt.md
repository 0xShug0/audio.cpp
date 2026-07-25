# Parakeet-TDT 0.6B v3

FastConformer-TDT ASR port of NVIDIA's [`nvidia/parakeet-tdt-0.6b-v3`](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3)
(0.6B params, 25 European languages with auto language detection), loaded directly
from the repo's Transformers-compatible `model.safetensors` checkpoint (not the
`.nemo` archive).

## Status

Offline (full-context) transcription only — see [Known limitations](#known-limitations)
for streaming. Verified correct end to end on CPU and CUDA against the real NeMo
reference model, both by final transcription and by numerical (per-layer
activation) comparison; see [Validation](#validation).

## Architecture

```
Frontend: 16kHz -> 128 mel bins, preemphasis=0.97, NeMo per-feature normalization
Encoder:  3-stage Conv2D subsampling (8x) -> 24-layer FastConformer -> 1024-dim
Decoder:  2-layer LSTM predictor + joint network (TDT: 5 duration classes [0..4])
```

## Build and run

```bash
cmake --build build/<preset> --target parakeet_warm_bench

build/<preset>/bin/parakeet_warm_bench \
    --model models/parakeet-tdt-0.6b-v3 \
    --audio tests/parakeet_tdt/assets/2086-149220-0033.wav \
    --backend cpu   # or: --backend cuda
```

Install the model with:

```bash
python3 tools/model_manager.py install parakeet_tdt --models-root models
```

## Validation

- **Golden-transcription regression test** (`ctest -R parakeet_golden_transcription_test`,
  wired into the normal `ENGINE_BUILD_TESTS` suite, skips cleanly when the model
  isn't downloaded): runs the offline pipeline against a checked-in LibriSpeech
  test-clean clip (`tests/parakeet_tdt/assets/2086-149220-0033.wav`) and asserts the
  decoded text matches the real NeMo model's transcription for that clip exactly:
  *"Well, I don't wish to see it any more, observed Phoebe, turning away her eyes.
  It is certainly very like the old portrait."*
- **Numerical parity harness** (`tests/parakeet_tdt/parity/`, manual/pre-release
  check, not a CI gate — needs a NeMo install): compares mel features, a single
  isolated encoder layer, and the full encoder output directly against real NeMo
  activations via forward hooks. See `tests/parakeet_tdt/parity/README.md` for
  exact setup and run commands. This is what actually caught the encoder
  conv-module bug below — the golden-transcription test alone did not, since
  greedy decoding happened to be robust enough to it on that one clip.
- Backends tested: CPU and CUDA (GTX 1650), both producing the exact same
  transcription.

Two real bugs were found and fixed getting to this state, worth knowing about if
you're touching this code:

1. The frontend was missing NeMo's per-feature mel normalization (subtract the
   per-mel-bin mean, divide by the per-mel-bin unbiased std over valid frames).
   Without it the encoder saw wildly out-of-distribution input and produced
   garbage (~500x too large in magnitude by the end of the conv subsampling
   stack).
2. The FastConformer encoder layer's conv module used causal (left-only)
   padding, and separately built the depthwise conv with `use_bias=false`,
   silently dropping the folded batch-norm bias term. The padding mode is
   wrong for this model's full-context (non-streaming) configuration — NeMo's
   `CausalConv1D` is only actually causal when constructed with `padding=None`;
   this model constructs it with an explicit symmetric `padding=(kernel-1)//2`.
   The dropped bias corrupted every layer's output by a per-channel constant
   offset; because greedy/argmax decoding is somewhat robust to numerical
   drift, this second bug did not visibly break transcription on the one test
   clip and was only caught by the numerical parity harness comparing actual
   activations against NeMo, not just final decoded text.

## Known limitations

- **No streaming support yet.** NeMo's cache-aware streaming FastConformer
  (causal conv with a cross-chunk `cache_last_time` cache, chunked-limited
  attention with a `cache_last_channel` KV cache) is not implemented; only
  `RunMode::Offline` is supported. This is tracked as follow-up work.
- Validated against a single test clip end to end; the numerical parity
  harness has not been run across a broader validation set.
- The frontend does not implement NeMo's preprocessor `dither` (small random
  waveform noise, standard to disable at inference time in most ASR
  pipelines) — no observed effect on transcription correctness.
