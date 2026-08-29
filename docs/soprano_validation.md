# Soprano TTS Validation

## Build

```bash
# Soprano-only build
cmake -B build -DCMAKE_BUILD_TYPE=Release \
  -DAUDIOCPP_MODEL_SET=custom -DAUDIOCPP_MODELS=soprano_tts
cmake --build build --target audiocpp_cli --parallel

# With Vulkan backend
cmake -B build -DCMAKE_BUILD_TYPE=Release \
  -DAUDIOCPP_MODEL_SET=custom -DAUDIOCPP_MODELS=soprano_tts \
  -DENGINE_ENABLE_VULKAN=ON
cmake --build build --target audiocpp_cli --parallel

# Build warmbench (the target is gated behind ENGINE_BUILD_WARMBENCH)
cmake -B build -DCMAKE_BUILD_TYPE=Release \
  -DAUDIOCPP_MODEL_SET=custom -DAUDIOCPP_MODELS=soprano_tts \
  -DENGINE_BUILD_WARMBENCH=ON
cmake --build build --target soprano_warm_bench --parallel
```

## Convert the checkpoint

```bash
# Download the official checkpoint
git lfs install
git clone https://huggingface.co/ekwek/Soprano-1.1-80M models/Soprano-1.1-80M

# Convert the decoder (folds weight-norm from decoder.pth)
pip install torch numpy safetensors
python3 tools/soprano_tts/convert_soprano.py \
  --input-dir models/Soprano-1.1-80M \
  --output-dir models/soprano_pkg
```

## Run warmbench

```bash
build/bin/soprano_warm_bench --model models/soprano_pkg --output-dir build/logs/warmbench/soprano_tts
```

## Python reference warmbench

```bash
pip install soprano torch numpy
python3 tests/soprano_tts/soprano_python_warm_bench.py \
  --model models/Soprano-1.1-80M \
  --out-dir build/logs/warmbench/soprano_tts_py
```

## CLI examples

```bash
# Basic inference
build/bin/audiocpp_cli --task tts --family soprano_tts \
  --model models/soprano_pkg \
  --text "Soprano is an extremely lightweight text to speech model." \
  --out soprano.wav

# With Vulkan backend
build/bin/audiocpp_cli --task tts --family soprano_tts \
  --model models/soprano_pkg \
  --backend vulkan \
  --text "Soprano runs on CPU and Vulkan backends." \
  --out soprano_vulkan.wav

# GGUF package
python3 tools/model_manager_v2.py install soprano_1_1_80m_q8_0
build/bin/audiocpp_cli --task tts --family soprano_tts \
  --model models/Soprano-1.1-80M-GGUF/soprano-1.1-80m-q8_0.gguf \
  --text "GGUF packages are standalone." --out gguf_out.wav
```

## Performance results

### CPU (compared against Python `soprano` package, transformers backend, temp=0.3, top_p=0.95)

| Test | Chars | Platform | Audio (s) | Infer (s) | RTF | Speedup |
|---|---|---|---|---|---|---|
| short | 57 | Python | 0.752 | 1.281 | 1.7037 | \u2014 |
| | | **C++** | **3.136** | **0.740** | **0.2360** | **7.22x** |
| medium | 152 | Python | 2.096 | 1.909 | 0.9106 | \u2014 |
| | | **C++** | **8.320** | **1.955** | **0.2350** | **3.87x** |
| long | 567 | Python | 7.424 | 5.773 | 0.7776 | \u2014 |
| | | **C++** | **16.384** | **4.302** | **0.2626** | **2.96x** |

### Vulkan (AMD Radeon RX Vega)

| Test | Audio (s) | Infer (s) | RTF |
|---|---|---|---|
| short | ~3.1 | ~0.25 | ~0.08 |
| medium | ~8.3 | ~0.70 | ~0.08 |
| long | ~16.4 | ~1.60 | ~0.10 |

## Known limitations

- English-only (model limitation)
- No voice cloning
- EOS sampling unreliable at low temperature (PyTorch vs C++ RNG difference)
- Full composite build may OOM; use AUDIOCPP_MODEL_SET=custom
- Vulkan decoder output shows numerical drift on AMD RX Vega (no matrix-core ops)
