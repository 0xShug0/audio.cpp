# Docker CLI

Run the audio.cpp CLI tool inside a Docker container for text-to-speech.

## Setup

**1. Download the PocketTTS model**

Get the English model from [kyutai/pocket-tts](https://huggingface.co/kyutai/pocket-tts/) on Hugging Face.
Place the files from `languages/english/` into `../models/pocket-tts/languages/english/`:

The directory should look like:
```
../models/pocket-tts/languages/english/
├── model.safetensors
├── tokenizer.model
└── embeddings/
    ├── alba.safetensors
    └── ...
```

**2. Build the image**

Run from repository root:

CPU:

```bash
docker build -f .devops/cpu.Dockerfile -t local/audiocpp:full-cpu .
```

GPU (CUDA):

```bash
docker build -f .devops/cuda.Dockerfile -t local/audiocpp:full-cuda .
```

## Usage

Run one of:

```bash
./cpu-tts.sh
./cuda-tts.sh
```

Speech is saved to `output/speech.wav`.

