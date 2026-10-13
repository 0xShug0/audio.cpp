# IBM Granite Speech 5.0 470M TurboCTC in audio.cpp

IBM Granite Speech 5.0 TurboCTC is a compact 470-million-parameter English Automatic Speech Recognition (ASR) model delivering state-of-the-art transcription accuracy with ultra-low latency.

## Architecture

- **Audio Frontend**: 16 kHz, 80-bin HTK mel spectrogram with 8.0 dB dynamic flooring, first-order deltas, and 2x frame stacking (320-dim features).
- **Acoustic Conformer**: 16 layers ($d_{model}=1024, d_{ff}=4096, d_{conv}=2048, heads=8, head\_dim=128$) with block self-attention ($context\_size=128$), Shaw relative positional embeddings, depthwise convolution with folded batch-norm, and mid-layer CTC self-conditioning at layer 8.
- **Decoder**: Non-autoregressive Connectionist Temporal Classification (CTC) with greedy decoding and 16,384 BPE vocabulary via fast HuggingFace tokenizer.

## CLI Usage

### Offline Transcription

```bash
# Transcribe audio using native safetensors directory
audiocpp_cli --task asr --family granite5asr --model path/to/checkpoint --audio sample.wav

```

### Quantized GGUF Loading

```bash
# Transcribe using a converted Q8_0 GGUF package
audiocpp_cli --task asr --family granite5asr --model models/granite-speech-5.0-470m-turboctc-gguf --audio sample.wav
```

### Audio Chunking & VAD

```bash
# Long audio with VAD segmentation using a valid Silero VAD model file
audiocpp_cli --task asr --family granite5asr --model path/to/checkpoint --audio long_audio.wav --request-option audio_chunk_mode=auto --session-option granite5asr.vad_model_path=/path/to/silero-vad.gguf

# Fixed 30-second chunking
audiocpp_cli --task asr --family granite5asr --model path/to/checkpoint --audio long_audio.wav --request-option audio_chunk_mode=fixed --request-option audio_chunk_duration_sec=30
```

VAD segmentation requires `granite5asr.vad_model_path` to point to an existing
model file. The default directory path does not enable VAD; without a valid
file, or if VAD fails, transcription uses fixed chunks.

Streaming mode buffers incoming audio and transcribes it when the input closes.
It returns a final transcript rather than incremental text.
