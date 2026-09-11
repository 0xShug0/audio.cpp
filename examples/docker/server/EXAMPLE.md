# Docker Compose Server

Run the audio.cpp TTS server as configured in the server JSON files 
with Docker Compose.

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

### 2. Start the server

Start **one** of:

```bash
docker compose -f pocket-tts-cuda12.yml up
docker compose -f pocket-tts-cuda13.yml up
docker compose -f pocket-tts-cpu.yml up
```

### 3. Optionally: Wait for server to be ready

```bash
./wait-for-server.sh
```

### 4. Generate speech

```bash
./pocket-tts.sh
```

This sends a request to `http://localhost:8080/v1/audio/speech` and saves
the result to `output/speech.wav`.

## Qwen3-TTS (with voice cloning)

### 1. Download the Qwen3-TTS model

Get the q8 model from
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

### 3. Start the server

Start **one** of:

```bash
docker compose -f qwen3-tts-cuda12.yml up
docker compose -f qwen3-tts-cuda13.yml up
docker compose -f qwen3-tts-cpu.yml up
```

### 4. Optionally: Wait for server to be ready

```bash
./wait-for-server.sh
```

### 5. Generate speech

```bash
./qwen3-tts.sh
```

This sends a request to `http://localhost:8080/v1/audio/speech` with voice
cloning parameters and saves the result to `output/speech.wav`.