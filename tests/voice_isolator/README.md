# Voice Isolator validation

Use the checkpoints from
[inworld/voice-isolator](https://huggingface.co/inworld/voice-isolator/tree/9b0285b3fe8403aa8e2f1dbb2165a80a4294c970)
at that revision:

| File | SHA-256 |
|---|---|
| `denoiser.safetensors` | `647335a6aeb4dc706b895532c209d51e0e84a4849d17b339f00d4392a050ae53` |
| `denoiser_realtime.safetensors` | `76598f05523c0669e76562a8b2f31c9d395fca316e80c907ee11775e7497ae0f` |

```bash
hf download inworld/voice-isolator denoiser.safetensors denoiser_realtime.safetensors \
  --revision 9b0285b3fe8403aa8e2f1dbb2165a80a4294c970 --local-dir models/voice-isolator
```

Model-specific tests are built with `ENGINE_BUILD_MODEL_TESTS=ON` when `voice_isolator`
is among the linked models. `voice_isolator_domain_test` needs no checkpoint and runs
under ctest: config parsing, band layout, STFT round trips, the attention window and
the offline window plan.

## Parity with the Python reference

Reference tensors are not stored in this repository. `make_fixtures.py` writes them
from `inworld.voice_isolation` 1.0.0 on a fixed synthetic signal that the C++ test
rebuilds bit for bit, so no audio file is involved. Keep the output directory outside
the repository:

```bash
pip install inworld-voice-isolation==1.0.0 safetensors numpy torch
python tests/voice_isolator/make_fixtures.py \
  --checkpoint-dir models/voice-isolator --output-dir /path/to/fixtures

build/debug/bin/test_voice_isolator_reference \
  --model models/voice-isolator --fixtures /path/to/fixtures --backend cpu
```

The test compares the public offline and streaming outputs with the reference for
uneven push sizes, the three look-aheads, a 20 s stream past the attention window,
stream reset, and offline windows; it also checks that rebasing the rotary positions
leaves a stream unchanged. Limits are 2e-3 on the largest sample difference and 2e-4
on the mean; with F32 weights and cache the CPU differences are about 2e-7.
`make_fixtures.py --check` regenerates into a temporary directory and compares with
an existing fixture directory.

## GGUF

```bash
python tests/voice_isolator/convert_gguf.py \
  --source models/voice-isolator/denoiser_realtime.safetensors \
  --output models/voice-isolator-gguf/voice-isolator-realtime-f32.gguf
```

The converter checks the checkpoint's SHA-256 and embeds the model spec and the
architecture JSON the checkpoint carries as metadata.

## Timing

`voice_isolator_warm_bench` (`ENGINE_BUILD_WARMBENCH=ON`) reports the real-time factor
of an offline render or of a stream, with per-push latency percentiles:

```bash
build/debug/bin/voice_isolator_warm_bench --model models/voice-isolator/denoiser_realtime.safetensors \
  --backend cpu --threads 16 --seconds 20 --push 480
build/debug/bin/voice_isolator_warm_bench --model models/voice-isolator/denoiser.safetensors \
  --offline --backend cuda --seconds 30
```

`cli_cases.json` supplies offline, streaming and request-reuse cases for the existing
CLI path runner.
