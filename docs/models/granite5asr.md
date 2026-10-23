# IBM Granite Speech 5.0 470M TurboCTC in audio.cpp

IBM Granite Speech 5.0 TurboCTC is a compact 470-million-parameter Automatic Speech Recognition (ASR) model. The current checkpoint transcribes English, French, German, Spanish, Portuguese, Italian, and Japanese, and it writes punctuation and capitals itself. There is no language input: a request language is accepted and ignored. A result is labelled `en` unless the decoder emits a trained language token such as `<|fr|>`. An older English-only checkpoint still loads and is still labelled `en`.

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

## Word timestamps

Word timestamps are produced. `supports_timestamps` is true.

The grid is 80 ms (1280 samples at the model's 16 kHz rate). Indexes are samples of the audio that was submitted, at that audio's sample rate. Offline and streaming both use this clock, so input at 8 kHz or 48 kHz reports the same times in seconds as 16 kHz input. Streaming resamples internally to 16 kHz and converts the word indexes back.

A token ends at the next different label, including blank. A word runs from its first subword piece through its last letter or digit, so blanks between those pieces are inside the word. A trailing punctuation piece is written onto that word and does not move its end or its confidence. A leading symbol, such as a currency mark or an opening quote, stays on the following word. Blanks between words are in neither word.

`confidence` is the geometric mean of the greedy-class probability on the non-blank frames of that word.

Streaming attaches the words to the final result.

Limits:

- The frame is 80 ms, so word boundaries are only as precise as one frame at any input rate.
- Confidence is the greedy-class probability, not a marginal CTC probability.
- A repeated non-blank label occupies every frame of that run.
- Blanks between subword pieces of one word are inside the word's time span and outside its confidence.
- Streaming words arrive on the final result only.
