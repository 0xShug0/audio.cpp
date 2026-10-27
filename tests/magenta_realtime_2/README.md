# Magenta RealTime 2 validation

Build the model test targets in a non-deployment Debug build:

```bash
cmake --build build/debug --target magenta_musiccoca_parity \
  magenta_codec_parity magenta_greedy_parity magenta_validation -j8
```

The Python reference scripts need the official `magenta-realtime` package with
its JAX and TFLite dependencies. Use a separate reference environment.

| Test | Coverage |
|---|---|
| `musiccoca_reference.py` / `magenta_musiccoca_parity` | Official text embeddings, mapper with aligned noise, RVQ codes, audio frontend, and audio embeddings. |
| `reference.py` / `magenta_codec_parity` | Teacher-forced official AR codes through the C++ codec; final-waveform cosine must exceed 0.99999. Tests one/four-frame blocks, padded final blocks, and reset/reuse. |
| `reference.py` / `magenta_greedy_parity` | Unforced greedy AR generation with aligned conditioning and F32 precision, then final-waveform comparison. |
| `magenta_validation` | Normal generation, repeated requests after different lengths/prompts, and exact concatenation of streaming events. |
| `benchmark.py` | Official Python system using the same prompts, durations, and seed as `cases.json`. |

For greedy parity, convert the selected generator with `--type orig` and generate
a matching F32 reference. Storage/compute differences can change greedy
choices and then the subsequent sequence; use the teacher-forced waveform test
to validate codec decoding independently.

```bash
python tests/magenta_realtime_2/reference.py --source checkpoint-dir \
  --variant small --frames 17 --temperature 0 --compute-dtype f32 \
  --output reference-small --log reference-small.log

build/debug/bin/magenta_greedy_parity gguf-dir/magenta-rt2-small-orig.gguf \
  reference-small/reference.safetensors cuda --log greedy-small.log

build/debug/bin/magenta_codec_parity gguf-dir/magenta-rt2-small-f16.gguf \
  reference-small/reference.safetensors cuda --log codec-small.log
```

Repeat with `--variant base` and the Base GGUF for the other transformer.
For runtime validation, execute outside the checkout using absolute paths to
the binary, case file, and output directory. Do not pass a model-spec override;
confirm `runtime.model.contract_spec` selects `@gguf/magenta_realtime_2.json`.

```text
magenta_validation MODEL.gguf cpu|cuda|vulkan offline|streaming CASES.json OUTPUT_DIR --log FILE
```

`cases.json` uses one session: warmup, a request and repeat, different short and
long prompts, then the original request again. Keep timing runs sequential.
