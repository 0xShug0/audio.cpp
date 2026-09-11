# Docker CLI

Run the audio.cpp CLI tool inside a Docker container for text-to-speech.

## PocketTTS

### 1. Download the PocketTTS model

Get the English q8 model from
[audio-cpp/audio.cpp-gguf](https://huggingface.co/audio-cpp/audio.cpp-gguf/tree/main/PocketTTS-GGUF)
on Hugging Face.
Place the files from `english/` into `../models/PocketTTS-GGUF/english/`:

The directory should look like:
```
../models/PocketTTS-GGUF/english/
├── pocket-tts-english-q8_0.gguf
└── embeddings/
    ├── alba.safetensors
    └── ...
```

### 2. Run

Run one of:

```bash
./pocket-tts-cuda12.sh
./pocket-tts-cuda13.sh
./pocket-tts-cpu.sh
```

See the scripts for details. The scripts reference the published docker image
and save the generated speech to `output/speech.wav`.

## Qwen3-TTS (with voice cloning)

### 1. Download the Qwen3-TTS model

Get the Qwen3-TTS-12Hz-1.7B-Base-GGUF q8 model from
[audio-cpp/audio.cpp-gguf](https://huggingface.co/audio-cpp/audio.cpp-gguf/tree/main/Qwen3-TTS-12Hz-1.7B-Base-GGUF)
on Hugging Face.
Place the file into `../models/Qwen3-TTS-12Hz-1.7B-Base-GGUF/`:

The directory should look like:
```
../models/Qwen3-TTS-12Hz-1.7B-Base-GGUF/
└── qwen3-tts-12hz-1.7b-base-q8_0_v2.gguf
```

### 2. Add reference audio and transcription

Put a `ref_audio.wav` and `ref_text.txt` in `../references/`.

The directory should look like:
```
../references/
├── ref_audio.wav
└── ref_text.txt
```

### 3. Run

Run one of:

```bash
./qwen3-tts-cuda12.sh
./qwen3-tts-cuda13.sh
./qwen3-tts-cpu.sh
```

See the scripts for details. The scripts reference the published docker image
and save the generated speech based on the provided reference audio in `output/speech.wav`.
