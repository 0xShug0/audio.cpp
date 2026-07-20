# VieNeu-TTS

VieNeu-TTS v3 Turbo generates high-fidelity 48 kHz bilingual (English and Vietnamese) speech with a custom Transformer architecture and instant voice-cloning capabilities. The compiled C++ runner in `audio.cpp` executes torch-free on both CPU and CUDA backends, bringing massive performance gains over the Python reference implementation.

| Field | Value |
|---|---|
| Family | `vietneu_tts` |
| Model directory | `models/VieNeu-TTS-v3-Turbo` |
| Task | `tts`, `clon` |
| Modes | `offline` |
| Languages | `vi`, `en` |
| Voice input | Optional reference WAV and reference transcript |
| Output | stereo 48 kHz WAV |

---

## 🚀 Installation & Quick Start

1. **Install GGUF Model Weights**:
   Download the quantized GGUF weights to `models/VieNeu-TTS-v3-Turbo/model.gguf` from the model repository:
   * HF Repository: [phuocnguyen90/VieNeu-TTS-v3-Turbo-GGUF](https://huggingface.co/phuocnguyen90/VieNeu-TTS-v3-Turbo-GGUF)

2. **Compile the CLI target**:
   ```bash
   chmod +x scripts/build_linux.sh
   ./scripts/build_linux.sh --backend cpu --target audiocpp_cli
   ```

3. **Run Inference**:
   ```bash
   ./build/linux-cpu-release/bin/audiocpp_cli \
     --task tts \
     --family vietneu_tts \
     --model models/VieNeu-TTS-v3-Turbo/model.gguf \
     --backend cpu \
     --voice-ref assets/resources/sample.wav \
     --reference-text "Some call me nature. Others call me Mother Nature. I've been here for over 4.5 billion years. 22,500 times longer than you." \
     --text "sˈin tʃˈaː2w tˈeɜ zˈəːɜj. ɗˈəɪ lˌaː2 bˈaː4n tˈy4 ŋˈiɛ6m." \
     --out output.wav
   ```

---

## 📊 Performance Benchmark

Measured on Ubuntu 24.04 (WSL2) with OpenMP optimization:

- **Quantized GGUF size**: **163 MB** (down from several gigabytes in safetensors).
- **Startup Latency**: **Instant (<0.01s)** model load via `mmap` lazy file loading (compared to ONNX's 6.4s / PyTorch's 9.3s).
- **Inference Speed**: **0.89 seconds** for a 3.68s speech generation on CPU, yielding a Real-Time Factor (RTF) of **0.24** (**4.1x faster than real-time**).

---

## 🛡️ Options & Customizations

| Option | Values | Default | Meaning |
|---|---|---:|---|
| `--temperature` | float | `0.8` | Main model sampling temperature. |
| `--subtalker-temperature` | float | `0.8` | Acoustic decoder / subtalker sampling temperature. |
| `--subtalker-do-sample` | bool | `true` | Enable sampling in the acoustic decoder. |
| `--voice-ref` | path | `sample.wav` | Reference speaker WAV file for voice cloning. |
| `--reference-text` | text | none | Transcript of the spoken reference speaker WAV. |
