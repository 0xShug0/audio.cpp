# MuLaCover

MuLaCover generates cover songs from reference audio or melody/chord MIDI,
with new lyrics and musical style. It uses a Llama-derived causal backbone with gated
symbolic cross attention, a Qwen3 style encoder and HeartCodec.

> [!IMPORTANT]
> Upstream restricts model weights and generated outputs to noncommercial use.
> See the [upstream model license](https://github.com/HeartMuLa/MuLaCover/blob/e27fa7fa0da7c9817879cc6210cc5693f9327946/MODEL_LICENSE).

## Install

Convert the official checkpoints with
[convert_mulacover.py](../../tools/models/convert_mulacover.py), or use
a complete [MuLaCover GGUF package](https://huggingface.co/audio-cpp/audio.cpp-gguf/tree/main/MuLaCover-GGUF).
Runtime loading is GGUF-only; source checkpoints are used only for conversion.
FP16 and Q8 packages contain the generator,
style encoder, codec, YourMT3 and all five ChordNet folds, with embedded
configuration, tokenizers and chord vocabulary.

The upstream reference is
[HeartMuLa/MuLaCover](https://github.com/HeartMuLa/MuLaCover/tree/e27fa7fa0da7c9817879cc6210cc5693f9327946),
with [HeartMuLa/MuLaCover weights](https://huggingface.co/HeartMuLa/MuLaCover/tree/bbbaef2b31835c2ef5172ff528dbe325230d46fb),
[Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B),
and [HeartCodec](https://huggingface.co/HeartMuLa/HeartCodec-oss-20260123).
The upstream generation guide links the YourMT3 and ChordNet checkpoints.

To convert locally, arrange the downloaded checkpoints as `MuLaCover/`,
`Qwen3-Embedding-0.6B/`, `HeartCodec-oss/`, and `SymbolicTranscriptor/` under
one source directory. First prepare the transcription checkpoints:

```bash
python tools/models/mulacover/convert_transcriber.py \
  --checkpoint models/MuLaCover-source/SymbolicTranscriptor/yourmt3/last.ckpt \
  --output models/MuLaCover-source/SymbolicTranscriptor/yourmt3/model.safetensors \
  --chord-root models/MuLaCover-source/SymbolicTranscriptor/chord \
  --log --log-file convert-mulacover-transcriber.log
```

Then write either package (`--type f16` or `--type q8_0`):

```bash
python tools/models/convert_mulacover.py \
  --model-dir models/MuLaCover-source --type q8_0 \
  --output models/MuLaCover-GGUF/mulacover-q8_0.gguf \
  --log-file convert-mulacover-q8.log
```

The converter uses `build/debug/bin/audiocpp_gguf` by default; use `--converter`
to select another built binary. It includes the component license notices.

## Run

Generate from MIDI:

```bash
audiocpp_cli --task gen --family mulacover \
  --model models/MuLaCover-GGUF/mulacover-q8_0.gguf --backend cuda \
  --text '[Verse] A morning light across the sea. [Chorus] Keep a little hope for me.' \
  --request-option 'tags=topic:[hope]; genre:[folk]; instrument:[piano]; mood:[warm]' \
  --request-option melody_file=melody.mid \
  --request-option chord_file=chord.mid \
  --request-option seed=1234 --out cover.wav --log
```

For reference-audio conditioning, replace the MIDI options with
`--audio reference.wav`. Tempo is estimated automatically; set
`--request-option bpm=120` to provide the reference tempo explicitly.
Reference audio and MIDI inputs are mutually exclusive.

Upstream selects the transcribed lead-singing track as melody; audio without
a detected melody is rejected. Reference-audio preprocessing requires SOXR.
MIDI inputs use musical ticks, so changing MIDI tempo metadata does not impose
a target generation tempo.

## Model

| Field | Value |
|---|---|
| Family | `mulacover` |
| Task | `music` (`--task gen`) |
| Mode | Offline |
| Input | Lyrics, style, and reference audio or melody/chord MIDI |
| Output | 48 kHz stereo audio |
| Packages | FP16, Q8_0 |

## Common Options (use directly)

| Option | Description |
|---|---|
| `--model` | Complete GGUF file or directory containing the package. |
| `--backend` | Inference backend. |
| `--threads` | CPU thread count. |
| `--text` | Lyrics, including song-section markers. |
| `--audio` | Reference audio; mutually exclusive with MIDI conditioning. |
| `--log` | Enable runtime logging. |

## Request Options (use with `--request-option`)

| Option | Default | Description |
|---|---|---|
| `lyrics` | `--text` | Lyrics override. |
| `tags` | Empty | Musical style description. |
| `melody_file` | Unset | Melody MIDI, required for the MIDI route. |
| `chord_file` | Unset | Chord MIDI, required for the MIDI route. |
| `drum_file` | Unset | Optional drum MIDI. |
| `bpm` | Automatic | Positive reference tempo, for audio input only. |
| `duration_sec` | `300` | Maximum generation budget; the model may stop earlier at EOS. |
| `temperature` | `1` | Positive sampling temperature. |
| `top_k` | `250` | Number of candidate audio tokens. |
| `guidance_scale` | `1.5` | Positive generator CFG scale; values above one enable both branches. |
| `seed` | `1234` | Generation seed. |
| `num_inference_steps` | `10` | Codec flow solver steps. |
| `codec_guidance_scale` | `1.25` | Positive codec CFG scale. |

CUDA uses BF16 activation rounding and BF16 KV storage, matching the upstream
default generator precision. Vulkan uses F16 activations and KV storage; CPU
uses F32 activations and KV storage. These policies do not convert quantized
package weights to F32. The codec and reference-audio preprocessing retain their
own precision policies.

The session owns the backbone and depth graphs. Backbone KV capacity covers the
requested prompt and generation budget, bounded by the model context; depth KV
is limited to eight codebook positions and reset for each audio frame. Cached
decode graphs attend to the active prefix in 128-position buckets and share one
scratch allocator. Backbone prefill, symbolic-conditioning, transcription-encoder,
chord-fold and codec workspaces are released after use. ChordNet retains only
the current reference-length graph for each fold. Codebook heads use views of
the stored weights and share one scratch buffer.

The prompt and generation budget must fit the 8,192-token context. Symbolic
conditioning is limited to 5,000 sixteenth notes. Offline generation uses
12.5 audio-code frames per second; frame progress is not streamed waveform output.
