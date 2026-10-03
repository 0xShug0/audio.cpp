# TF-GridNet

TF-GridNet separates overlapping speech into one waveform per speaker. The
published WSJ0-2mix package uses 8 kHz audio and produces two speaker tracks.
The runtime reads the speaker count and architecture from the GGUF; checkpoints
trained for other speaker counts do not require a separate model family.

This is offline source separation, not diarization or target-speaker extraction.
Speaker track numbers do not identify people. Chunked inference matches tracks
across overlaps, but cannot guarantee identity through long silent gaps.

## Usage

```bash
audiocpp_cli --task sep --family tf_gridnet \
  --model /path/to/TF-GridNet-GGUF/tf-gridnet-wsj0-2mix-f32.gguf \
  --backend cuda --audio mixture.wav --out-dir separated --log
```

Outputs are `speaker_1.wav`, `speaker_2.wav`, and further numbered tracks when
the checkpoint supports more speakers. Audio is resampled to the checkpoint's
sample rate. Single-microphone checkpoints downmix input channels; multi-microphone
checkpoints require the configured channel count and microphone arrangement.

### Common Options

| Option | Description |
| --- | --- |
| `--audio` | Mixed speech recording. |
| `--out-dir` | Directory for the separated speaker WAVs. |
| `--backend` | Execution backend. |
| `--threads` | CPU thread count. |
| `--out-format` | `pcm16`, `pcm24`, or `float32`. |

### Request Options (Use With `--request-option`)

| Option | Default | Description |
| --- | --- | --- |
| `audio_chunk_duration_sec` | `2.4` | Window length; `0` processes the whole recording in one graph shape. |
| `audio_chunk_overlap_sec` | `1.6` | Overlap for speaker matching and averaging; positive and smaller than the window. |
| `normalize_output` | `true` | Normalize each final track to a peak of `0.9`. |

### Session Options (Use With `--session-option`)

| Option | Default | Description |
| --- | --- | --- |
| `tf_gridnet.weight_type` | `native` | Weight storage type. |

## Long Recordings

The default 2.4-second windows and 0.8-second hop follow ESPnet's separation
example. Windows reuse the same session-owned graphs. SI-SNR matching determines
the speaker permutation before overlapping waveforms are averaged.

For whole-file inference, set `audio_chunk_duration_sec=0`. This retains global
context but increases memory use with recording length. To retain the model's raw
output scale, also set `normalize_output=false` and use `--out-format float32`;
the unnormalized output can exceed the PCM range.

## Conversion

Download and extract the checkpoint archive linked in the
[ESPnet separation example](https://kan-bayashi.github.io/espnet/notebook/SpeechEnhancement_CMU_11492_692_Spring2023%28Assignment7%29.html#model-selection).
Use its matching `config.yaml` and `98epoch.pth`:

```bash
python tests/tf_gridnet/convert_gguf.py \
  --checkpoint /path/to/98epoch.pth \
  --config /path/to/config.yaml \
  --output /path/to/TF-GridNet-GGUF/tf-gridnet-wsj0-2mix-f32.gguf \
  --log conversion.log
```

The converter packages the original tensors, STFT window, configuration, and
model spec, then verifies that every tensor is byte-identical to its source.
Changing the configured speaker count requires matching trained output weights;
it is not an inference-time speaker-count switch.

## References

- [TF-GridNet paper](https://arxiv.org/abs/2211.12433)
- [ESPnet implementation](https://github.com/espnet/espnet/blob/master/espnet2/enh/separator/tfgridnet_separator.py)
- [Checkpoint archive](https://drive.google.com/file/d/1TasZxZSnbSPsk_Wf7ZDhBAigS6zN8G9G/view)

The downloaded checkpoint archive does not state a weight license. ESPnet's
source-code license should not be assumed to cover the checkpoint; see
[model licenses](../model_licenses.md).
