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
audiocpp_cli asr --model granite5asr --input sample.wav

# Or explicitly pass the family
audiocpp_cli asr --family granite5asr --model path/to/checkpoint --input sample.wav
```

### Quantized GGUF Loading

```bash
# Transcribe using a converted Q8_0 GGUF package
audiocpp_cli asr --model models/granite-speech-5.0-470m-turboctc-gguf --input sample.wav
```

### Audio Chunking & VAD

```bash
# Long audio with automatic VAD segmentation
audiocpp_cli asr --model granite5asr --input long_audio.wav --option audio_chunk_mode=auto

# Fixed 30-second chunking
audiocpp_cli asr --model granite5asr --input long_audio.wav --option audio_chunk_mode=fixed --option audio_chunk_duration_sec=30
```
