# MossFormer2

MossFormer2 supports three ClearVoice checkpoints:

| Variant | Task | Output | GGUF |
|---|---|---|---|
| SS 16K | `sep` | Two 16 kHz mono speaker tracks | F32 or F16 |
| SE 48K | `s2s` | Enhanced 48 kHz mono speech | F32 |
| SR 48K | `s2s` | Speech with restored high frequencies, 48 kHz mono | F16 |

All packages are under `MossFormer2-GGUF` in
[audio-cpp/audio.cpp-gguf](https://huggingface.co/audio-cpp/audio.cpp-gguf/tree/main/MossFormer2-GGUF).
Choose the task matching the checkpoint. Audio-visual target-speaker extraction
is not included.

## Usage

```bash
audiocpp_cli --task sep --family mossformer2 \
  --model /path/to/MossFormer2-GGUF/mossformer2-ss-16k-f32.gguf \
  --backend cuda --audio mixture.wav --out-dir separated --log
```

The outputs are `speaker_1.wav` and `speaker_2.wav`. Stereo input is downmixed
and audio is resampled to the checkpoint sample rate.

For enhancement or super-resolution, select the corresponding checkpoint and
write a single WAV with `--out`:

```bash
audiocpp_cli --task s2s --family mossformer2 \
  --model MossFormer2-GGUF/mossformer2-se-48k-f32.gguf \
  --backend cuda --audio noisy.wav --out enhanced.wav --log

audiocpp_cli --task s2s --family mossformer2 \
  --model MossFormer2-GGUF/mossformer2-sr-48k-f16.gguf \
  --backend cuda --audio narrowband.wav --out restored.wav --log
```

SE predicts a spectral mask from Kaldi filterbanks and their first/second-order
deltas. SR uses a mel-feature MossFormer2 network and a Snake-activated vocoder,
then preserves the input's low-frequency band as in ClearVoice. SR can trim less
than one vocoder hop (256 samples) from whole-file output.
Silent inputs and inputs whose estimated bandwidth reaches Nyquist pass through
unchanged, without running SR inference.

### Common Options

| Option | Description |
| --- | --- |
| `--audio` | Input speech recording. |
| `--out-dir` | Directory for separated speaker WAVs. |
| `--out` | Enhanced or super-resolved WAV. |
| `--backend` | Execution backend. |
| `--threads` | CPU thread count. |
| `--out-format` | `pcm16`, `pcm24`, or `float32`. |

### Request Options (Use With `--request-option`)

| Option | Default | Description |
| --- | --- | --- |
| `audio_chunk_duration_sec` | SS: `2`; SE/SR: automatic | `0` processes the whole recording. SE/SR use whole-file inference up to 20 seconds and four-second windows above that threshold when unset. An explicit positive value enables windowing; SR rounds windows up to a 256-sample hop. |
| `audio_chunk_overlap_sec` | SS: `0.5`; SE/SR: 25% of the window | Must be non-negative and smaller than the window. Overlapping edges are trimmed rather than averaged. Ignored for whole-file inference. |
| `normalize_output` | SS: `true`; SE/SR: `false` | Match output RMS to the input. Defaults follow the corresponding ClearVoice inference path. |

The spec leaves variant-dependent defaults unset; the selected session supplies
the defaults above. SE uses stochastic Kaldi dithering, matching the upstream
frontend; repeated outputs can differ slightly.

## Long Recordings

The SS default path uses two-second windows with 0.5-second overlap (a 1.5-second hop) and trims
shared edges between windows, preserving the final window's tail. A session reuses its graph for repeated windows; changing
the window size replaces that graph rather than retaining multiple graph sizes.
SE/SR use the same session-owned graph reuse pattern with their own analysis
frontends. Their default long-recording window is four seconds with one-second
overlap.

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

SE uses `--variant se`, the [original checkpoint](https://huggingface.co/alibabasglab/MossFormer2_SE_48K), and `MossFormer2_SE_48K.yaml`:

```bash
python tests/mossformer2/convert_gguf.py --variant se \
  --checkpoint MossFormer2_SE_48K/last_best_checkpoint.pt \
  --config ClearerVoice-Studio/clearvoice/clearvoice/config/inference/MossFormer2_SE_48K.yaml \
  --output MossFormer2-GGUF/mossformer2-se-48k-f32.gguf --log conversion-se.log
```

SR packs its [feature network and vocoder](https://huggingface.co/alibabasglab/MossFormer2_SR_48K) into one GGUF. Weight normalization is
folded during conversion using PyTorch's weight-normalization operation:

```bash
python tests/mossformer2/convert_gguf.py --variant sr \
  --checkpoint MossFormer2_SR_48K/last_best_checkpoint_m.pt \
  --vocoder-checkpoint MossFormer2_SR_48K/last_best_checkpoint_g.pt \
  --config ClearerVoice-Studio/clearvoice/clearvoice/config/inference/MossFormer2_SR_48K.yaml \
  --vocoder-config ClearerVoice-Studio/clearvoice/clearvoice/config/inference/MossFormer2_SR_48K.json \
  --output MossFormer2-GGUF/mossformer2-sr-48k-f16.gguf --log conversion-sr.log
```

SR conversion defaults to F16; affine, positional, and Snake parameters remain
F32. SS still defaults to F32 and supports F16, while SE uses F32. Precision
follows the selected GGUF; the runtime does not replace F32 weights with F16
weights.
