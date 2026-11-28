# Real-time RE-USE

`realtime_reuse` ports NVIDIA's causal speech-enhancement checkpoint, including offline restoration and incremental waveform streaming. It is separate from `reuse`: the realtime checkpoint uses causal time-Mamba, channel LayerNorm, a 20 ms STFT hop, and trained decoders for selectable exit layers. Existing RE-USE packages and behavior are unchanged.

Upstream: [nvidia/Real-time_RE-USE, pinned revision](https://huggingface.co/nvidia/Real-time_RE-USE/tree/4c0c186272ca853cd18d832fc5c28777b62f32d1).
GGUF: [audio-cpp/RE-USE-GGUF](https://huggingface.co/audio-cpp/RE-USE-GGUF).
The F32 package preserves checkpoint tensors exactly. Mamba state and waveform processing remain F32.

The checkpoint is distributed under the upstream NVIDIA One-Way Noncommercial License.

## Run

The model accepts mono audio at 8–48 kHz and returns the same sample rate and sample count. Select the `s2s` task. Input must be longer than the centered STFT reflection pad (approximately 20 ms).

```bash
python tools/model_manager_v2.py install realtime_reuse_f32 --models-root models

build/debug/bin/audiocpp_cli --task s2s --family realtime_reuse \
  --model models/RE-USE-GGUF/realtime-reuse-f32.gguf \
  --backend cuda --threads 8 --audio noisy.wav --out restored.wav --log
```

For incremental processing, add `--mode streaming`:

```bash
build/debug/bin/audiocpp_cli --task s2s --mode streaming --family realtime_reuse \
  --model models/RE-USE-GGUF/realtime-reuse-f32.gguf \
  --backend cuda --threads 8 --audio noisy.wav --out restored.wav --log
```

Live PCM input uses the standard CLI interface:

```bash
cat microphone.s16le | build/debug/bin/audiocpp_cli \
  --task s2s --mode streaming --family realtime_reuse \
  --model models/RE-USE-GGUF/realtime-reuse-f32.gguf \
  --backend cuda --audio - --input-format s16le --input-rate 16000 \
  --input-channels 1 --out restored.wav --log
```

## Session options

| Option | Default | Meaning |
|---|---|---|
| `realtime_reuse.exit_layer` | `8` | Use 3–12 time-frequency blocks and the corresponding trained decoder. |
| `realtime_reuse.lookahead_frames` | `0` | Use 0–2 encoder lookahead frames, approximately 20 ms each. |
| `realtime_reuse.weight_type` | `native` | Preserve the GGUF types, or explicitly select another supported storage type. |

For example, append `--session-option realtime_reuse.exit_layer=3 --session-option realtime_reuse.lookahead_frames=2` to choose a different quality/latency trade-off. These options select a model configuration for the session; they are not per-request overrides. Together they cover all 30 upstream latency configurations.

## Streaming and memory

The streaming frontend performs incremental centered STFT, feature extraction, and overlap-add ISTFT using framework FFT plans. It retains bounded input waveform, lookahead, convolution, and Mamba processing state. The session also collects emitted PCM for the standard final audio result. It flushes lookahead at the end and preserves the upstream ISTFT length and final epsilon padding. Changing input chunk boundaries does not reset the network. Streaming processes one frame at a time until the first audio is emitted, then grows to 2, 4, and 8 frames per graph execution. This preserves first-audio latency while amortizing GPU submissions. Eight frames represent approximately 160 ms of live input, so later audio is emitted in larger batches. Each graph size is cached for reuse, including across requests. The final partial batch is padded internally; only its valid output frames are emitted.

Offline inference carries causal state across bounded frame blocks, rather than allocating a whole-recording network graph. Graphs belong to the session and are reused. State is cleared between requests while cached graphs remain available. Only the selected encoder branch, exit layers, and decoder pair are uploaded; the GGUF retains the complete checkpoint so any supported configuration can be selected in a new session.

The upstream `online_inference.py` runs the network frame by frame but computes STFT and ISTFT over the complete recording. Its algorithm is the streaming-network reference; it is not a live waveform frontend.
