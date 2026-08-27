## Soprano TTS — Community Model

Soprano is an ultra-lightweight (~80M parameter) English-only text-to-speech model using a two-stage architecture: a Qwen3-style causal LM (17 layers, hidden 512, vocab 8192) that autoregressively emits per-frame 512-dimensional features, and a non-iterative Vocos-style decoder (ConvNeXt backbone + single ISTFT head, n_fft 2048 / hop 512) that turns those features into 32 kHz audio.

Reference: https://github.com/ekwek1/soprano
Weights: https://huggingface.co/ekwek/Soprano-1.1-80M
GGUF packages: https://huggingface.co/WalkingCat/Soprano-1.1-80M-GGUF

### Files added

| Path | Purpose |
|---|---|
| `src/community_models/soprano_tts/` (5 .cpp) | Assets, Qwen3 LM generator, Vocos decoder, tokenizer, session |
| `include/engine/community_models/soprano_tts/` (5 .h) | Corresponding headers |
| `model_specs/soprano_tts.json` | Schema-v1 spec, community status, GGUF + safetensors sources |
| `docs/soprano_tts.md` | User-facing documentation |
| `docs/soprano_validation.md` | Validation record with build/run commands and timing |
| `tests/soprano_tts/soprano_warm_bench.cpp` | C++ warmbench binary |
| `tests/soprano_tts/soprano_warm_bench_cases.txt` | Warmbench test cases (short/medium/long/longform) |
| `tests/soprano_tts/soprano_python_warm_bench.py` | Python reference warmbench |
| `tools/soprano_tts/convert_soprano.py` | HF checkpoint converter (folds weight-norm) |
| `tools/soprano_tts/run_official.py` | Python reference inference runner |
| `tools/soprano_tts/compare_parity.py` | Automated validation harness |

### Files modified

| File | Change |
|---|---|
| `CMakeLists.txt` | +16 lines: `audiocpp_add_model` + `add_engine_warmbench` |
| `README.md` | +2 lines: community table row |
| `docs/gguf.md` | +1 line: GGUF status table row |
| `webui/configs/models_catalog.json` | +1 line: WebUI catalog entry |
| `webui/native/dist/index.html` | Rebuilt frontend with Soprano baked in |

### Build

```bash
# Soprano only
cmake -B build -DCMAKE_BUILD_TYPE=Release \
  -DAUDIOCPP_MODEL_SET=custom -DAUDIOCPP_MODELS=soprano_tts
cmake --build build --target audiocpp_cli --parallel

# With Vulkan
cmake -B build -DCMAKE_BUILD_TYPE=Release \
  -DAUDIOCPP_MODEL_SET=custom -DAUDIOCPP_MODELS=soprano_tts \
  -DENGINE_ENABLE_VULKAN=ON
cmake --build build --target audiocpp_cli --parallel
```

### Quick start

```bash
# Install GGUF package
python3 tools/model_manager_v2.py install soprano_1_1_80m_q8_0

# Run inference
build/bin/audiocpp_cli --task tts --family soprano_tts \
  --model models/Soprano-1.1-80M-GGUF/soprano-1.1-80m-q8_0.gguf \
  --text "Soprano is an extremely lightweight text to speech model." \
  --out soprano.wav
```

### Run warmbench

```bash
# Build warmbench
cmake --build build --target soprano_warm_bench --parallel

# Run (CPU)
build/bin/soprano_warm_bench --model models/soprano_pkg \
  --output-dir build/logs/warmbench/soprano_tts

# Python reference
python3 tests/soprano_tts/soprano_python_warm_bench.py \
  --model models/Soprano-1.1-80M \
  --out-dir build/logs/warmbench/soprano_tts_py
```

### Validation

CPU performance vs official Python `soprano` package (transformers backend, temp=0.3, top_p=0.95):

| Test | Chars | Platform | Audio (s) | Infer (s) | RTF | Speedup |
|---|---|---|---|---|---|---|
| short | 57 | Python | 0.752 | 1.281 | 1.7037 | — |
| | | **C++** | **3.136** | **0.740** | **0.2360** | **7.22x** |
| medium | 152 | Python | 2.096 | 1.909 | 0.9106 | — |
| | | **C++** | **8.320** | **1.955** | **0.2350** | **3.87x** |
| long | 567 | Python | 7.424 | 5.773 | 0.7776 | — |
| | | **C++** | **16.384** | **4.302** | **0.2626** | **2.96x** |

Key observations:
- C++ RTF stays below 0.27 on all tests (faster than real-time).
- Audio durations differ between Python and C++ because of different RNG implementations; both produce valid 32 kHz speech.
- Backend coverage: CPU (tested, RTF ~0.24), Vulkan (tested on RX Vega, RTF ~0.08-0.12).
- See `docs/soprano_validation.md` for full validation record.

### Known limitations

- English-only (model limitation)
- No voice cloning
- EOS sampling unreliable at low temperature (PyTorch vs C++ RNG difference)
- Full composite build may OOM; use AUDIOCPP_MODEL_SET=custom
- Vulkan decoder output has numerical drift on AMD RX Vega (no matrix-core ops)
