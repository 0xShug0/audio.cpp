# MOSS-SoundEffect v2 parity

Manual CUDA comparison of native tokenization, Qwen3 text encoding, the full
DiT diffusion trajectory, and DAC waveform against the official reference.
Run the Python and C++ stages sequentially in the `qwen3-tts` environment.

```bash
conda run --no-capture-output -n qwen3-tts python tests/moss_soundeffect/generate_reference.py \
  --model-dir path/to/MOSS-SoundEffect-v2.0-upstream \
  --output-dir build/logs/moss_soundeffect/parity \
  --log build/logs/moss_soundeffect/parity/reference.log

conda run --no-capture-output -n qwen3-tts cmake --build build/debug \
  --target moss_soundeffect_parity -j 8

conda run --no-capture-output -n qwen3-tts build/debug/bin/moss_soundeffect_parity \
  --model path/to/moss-soundeffect-v2-bf16.gguf \
  --model-spec-override model_specs/moss_soundeffect.json \
  --fixture-dir build/logs/moss_soundeffect/parity \
  --threads 8 \
  --log build/logs/moss_soundeffect/parity/native.log
```

The generator requires the official package's Python dependencies and local
source at `reference/MOSS-TTS` (`--reference` overrides that source root).
It writes `reference.safetensors` and `manifest.json` and refuses to overwrite
existing fixtures or logs.

The single prompt is “Steady rain pattering on a metal roof with distant rolling
thunder.” with ` duration: 10.0s`. Default settings are batch one, mono 48 kHz,
the checkpoint's full 30-second latent window, 100 Euler steps, CFG 4, sigma shift
5, seed 0, CPU-generated BF16 noise, and eight CPU threads. `--steps` and `--seed`
overrides are recorded in `manifest.json` alongside source hashes.

The default reference uses upstream BF16 text encoding and CUDA autocast,
F32 guidance accumulation, and F32 DAC decoding. Saved floating tensors use F32
containers without changing their values. The official compiled model function
is unwrapped for reproducible validation. TF32 is disabled.

The Python generator also supports `--dtype f32` for controlled mathematical
comparisons. That control is separate from production inference; the native
replay uses BF16 fixtures and follows upstream mixed precision regardless of
weight storage.

| Fixture tensor | Layout |
|---|---|
| `context.positive`, `context.negative` | F32 `[1,512,2048]`, zero-padded official Qwen hidden states |
| `text.positive.ids`, `text.negative.ids` | I32 `[1,512]` |
| `text.positive.mask`, `text.negative.mask` | I32 `[1,512]`, valid prefix entries equal 1 |
| `text.positive.valid_count`, `text.negative.valid_count` | I32 `[1]` |
| `initial_noise`, `final_latent` | F32 `[1,128,frames]` |
| `schedule.timesteps` | F32 `[steps]`, official DiT inputs `1000*sigma` |
| `schedule.sigmas` | F32 `[steps+1]`, including terminal zero |
| `waveform` | F32 `[1,1,full_samples]`, full DAC output |

The validator reads token IDs, masks, and valid counts from the same fixture.
It tokenizes each manifest prompt natively, truncates to 512, checks the exact
valid token prefix/count, and encodes those native IDs with the existing Qwen
runtime. Its zero-padded hidden states are compared against both official
contexts and then supplied to DiT. An additional oracle-context trajectory isolates text-encoder numerical drift;
the primary comparison always uses native contexts. Only initial noise and the official schedule are replayed;
seeded native RNG and schedule construction are outside this comparison.

Use the BF16 package for the production replay. The text encoder uses DecoderHiddenModule with AllSteps and session-owned
length-bucketed graphs. It does not allocate an autoregressive decode cache. Runtimes own and reuse their
graphs throughout the validation. Euler updates use sigma-space `t` and `t_next`;
DiT prediction receives the saved `1000*sigma` values.

Each numerical comparison must have cosine similarity **strictly greater than
0.99999**: both Qwen contexts, final diffusion latent, full waveform, its returned
10-second crop, and DAC decoding of the reference final latent. Token IDs/counts
must match exactly. Non-finite tensors and shape/rate mismatches fail explicitly.
Zero reference and native contexts compare equal when both are entirely zero.
There is no time alignment, gain normalization, resampling, or quiet-region
removal.

Logs report cosine, maximum absolute error, RMSE, and allclose failure counts.
`--atol` and `--rtol` (both default 0.001) affect only those diagnostic allclose
counts, never the pass/fail gate. `--log <path>` is required and
`session.wall_ms` is recorded. Exit 0 means all gates passed, 1 means validation
failed, and 2 means invalid invocation. The test is manual and is not a benchmark
or automatically registered CTest.

## Device sampler and bounded decoder

Add `--sampling-check true` to the native command to compare backend-resident
Euler/CFG against the host reference, switch CFG modes in one runtime, and
check text-context reuse across short/full-length prompts and full and cropped
DAC decoding. This check uses the saved fixture noise
and final latent; it does not use waveform similarity as a performance gate.

`benchmark_python.py` measures the upstream compiled pipeline after warmup.
Its default `--attention sdpa` selects native BF16 attention, excluding optional
quantized SageAttention. Use `--attention auto` to retain upstream optional
attention-package selection.
