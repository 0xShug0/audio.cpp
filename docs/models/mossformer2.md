# MossFormer2

MossFormer2 separates overlapping speech into two 16 kHz mono speaker tracks.
This family uses the ClearVoice `MossFormer2_SS_16K` checkpoint, not its speech
enhancement, super-resolution, or audio-visual extraction models.

## Usage

```bash
audiocpp_cli --task sep --family mossformer2 \
  --model /path/to/MossFormer2-GGUF/mossformer2-ss-16k-f32.gguf \
  --backend cuda --audio mixture.wav --out-dir separated --log
```

The outputs are `speaker_1.wav` and `speaker_2.wav`. Stereo input is downmixed
and audio is resampled to the checkpoint sample rate.

### Common Options

| Option | Description |
| --- | --- |
| `--audio` | Mixed speech recording. |
| `--out-dir` | Directory for separated speaker WAVs. |
| `--backend` | Execution backend. |
| `--threads` | CPU thread count. |
| `--out-format` | `pcm16`, `pcm24`, or `float32`. |

### Request Options (Use With `--request-option`)

| Option | Default | Description |
| --- | --- | --- |
| `audio_chunk_duration_sec` | `2` | Separation window duration; `0` processes the whole recording. |
| `audio_chunk_overlap_sec` | `0.5` | Overlap in seconds; recommended 25% of the chunk duration, matching official Python settings. Must be non-negative and smaller than the chunk duration. Overlapping edges are trimmed rather than averaged. Ignored for whole-file inference. |
| `normalize_output` | `true` | Match each trimmed track's RMS to the input, following ClearVoice's NumPy interface. |

## Long Recordings

The default path uses two-second windows with 0.5-second overlap (a 1.5-second hop) and trims window
edges as in ClearVoice. A session reuses its graph for repeated windows; changing
the window size replaces that graph rather than retaining multiple graph sizes.

Speaker numbers identify output lanes, not persistent speaker identities.
The upstream windowing procedure does not match speaker identities across
windows, so tracks can exchange speakers. Whole-file inference avoids window
boundaries but increases memory use with recording length.

## Conversion

Download the original checkpoint from
[alibabasglab/MossFormer2_SS_16K](https://huggingface.co/alibabasglab/MossFormer2_SS_16K)
and use the matching `MossFormer2_SS_16K.yaml` from
[ClearerVoice-Studio](https://github.com/modelscope/ClearerVoice-Studio).

```bash
python tests/mossformer2/convert_gguf.py \
  --checkpoint /path/to/last_best_checkpoint.pt \
  --config /path/to/MossFormer2_SS_16K.yaml \
  --output /path/to/MossFormer2-GGUF/mossformer2-ss-16k-f32.gguf \
  --log conversion.log
```

The converter discards training optimizer state, embeds the configuration and
model spec, and verifies every F32 tensor byte against the checkpoint.
Use `--type f16` and an `-f16.gguf` output name for the smaller package. Small
affine and positional tensors remain F32 for the elementwise operations; the
converter verifies the remaining tensors against the F16-rounded source.
