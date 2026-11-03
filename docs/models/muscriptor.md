# MuScriptor

MuScriptor is an audio-to-symbolic tool that converts music audio into
note-event JSON or a MIDI file. The default download is the standalone F32
GGUF package; the original safetensors layout is still supported for local
development.

## Model Sizes

The official variants share the same model architecture and runtime. Width,
head count, layer count and vocabulary size are read from the embedded config.
Small remains the default package. All three sizes use F32 GGUFs.

| Variant | Parameters | Width | Heads | Layers | Vocabulary (`card`) | Original precision |
|---|---:|---:|---:|---:|---:|---|
| [Small](https://huggingface.co/MuScriptor/muscriptor-small) | 103M | 768 | 12 | 14 | 1393 | F32 |
| [Medium](https://huggingface.co/MuScriptor/muscriptor-medium) | 307M | 1024 | 16 | 24 | 1395 | F32 |
| [Large](https://huggingface.co/MuScriptor/muscriptor-large) | 1.366B | 1536 | 24 | 48 | 1395 | F32 |

## Download And Run

```bash
python3 tools/model_manager_v2.py install muscriptor
```

Select another size explicitly:

```bash
python3 tools/model_manager_v2.py install muscriptor_medium_f32
python3 tools/model_manager_v2.py install muscriptor_large_f32
```

All three packages share `models/MuScriptor-Small-GGUF/` for compatibility with
the existing download directory. Pass the specific GGUF filename to `--model`
when that directory contains multiple variants.

```bash
audiocpp_cli --task midi --family muscriptor \
  --model models/MuScriptor-Small-GGUF/muscriptor-small-f32.gguf \
  --backend cuda \
  --audio song.wav \
  --request-option instruments=drums,electric_bass \
  --out result.mid
```

Streaming mode accepts audio chunks and returns the final MIDI/event result:

```bash
audiocpp_cli --task midi --mode streaming --family muscriptor \
  --model models/MuScriptor-Small-GGUF/muscriptor-small-f32.gguf \
  --backend cuda \
  --audio song.wav \
  --request-option instruments=drums,electric_bass \
  --out result.mid
```

## Convert Original Checkpoints

Accept the license on the chosen upstream model page, then download its
`config.json` and `model.safetensors` into one directory. Install the Python
conversion dependencies:

```bash
python3 -m pip install numpy safetensors
```

Build the converter and CLI (Metal on macOS; select your backend on other
platforms):

```bash
cmake -S . -B build/muscriptor -DCMAKE_BUILD_TYPE=Release \
  -DAUDIOCPP_MODEL_SET=custom -DAUDIOCPP_MODELS=muscriptor \
  -DENGINE_ENABLE_METAL=ON
cmake --build build/muscriptor --config Release \
  --target audiocpp_gguf audiocpp_cli -j
```

Convert the downloaded checkpoint without changing precision or model math:

```bash
python3 tests/muscriptor/convert_gguf.py \
  --source <checkpoint-directory> \
  --output models/MuScriptor-Small-GGUF/muscriptor-medium-f32.gguf \
  --converter build/muscriptor/bin/audiocpp_gguf
```

For Large, use its checkpoint directory and an output such as
`models/MuScriptor-Small-GGUF/muscriptor-large-f32.gguf`. The same script also
accepts Small. Each generated GGUF is self-contained; no sidecar files are
needed for inference.

```bash
build/muscriptor/bin/audiocpp_cli --task midi --family muscriptor \
  --model models/MuScriptor-Small-GGUF/muscriptor-medium-f32.gguf \
  --backend metal --audio <music.wav> --out result.mid --log
```

## Common Options (use directly)

| Option | Values | Default | Meaning |
|---|---|---:|---|
| `--out` | path | not set | Write the selected primary output. Use `.mid` for MIDI or `.json` for event JSON. |
| `--text-out` | path | not set | Optionally also write generated note-event JSON. |
| `--max-tokens` | integer | `2000` | Maximum generated MIDI-event tokens per chunk. |
| `--temperature` | float | `1.0` | Sampling temperature. `0` selects deterministic argmax. |
| `--guidance-scale` | float | `1.0` | Classifier-free guidance coefficient; `1` disables CFG. |
| `--num-beams` | integer | `1` | Beam-search width; 1 disables beam search, default 1. Beam search requires guidance_scale=1, do_sample=false, and an actual graph batch of one. |
| `--seed` | integer | `0` | Sampling seed; default 0. Parsed as int64 and converted to uint64; negative values wrap. |

## Request Options (use with `--request-option`)

| Option | Values | Default | Meaning |
|---|---|---:|---|
| `instruments` | comma-separated names listed below | empty | Constrain generated MIDI events to instrument groups. |
| `output_format` | `midi`, `json` | `midi` | Select the primary output serialization written by `--out`. |
| `max_tokens` | integer | `2000` | Maximum generated MIDI-event tokens per chunk. |
| `do_sample` | bool | `false` | Use sampling instead of greedy token selection. |
| `guidance_scale` | float | `1.0` | Classifier-free guidance coefficient; `1` disables CFG. |
| `num_beams` | integer | `1` | Beam-search width; 1 disables beam search, default 1. Beam search requires guidance_scale=1, do_sample=false, and an actual graph batch of one. |
| `batch_size` | integer | `1` | Number of chunks decoded per batch when `prelude_forcing=false`. |
| `prelude_forcing` | bool | `true` | Force open-note prelude tokens between sequential chunks. |

`instruments` accepts `acoustic_piano`, `electric_piano`,
`chromatic_percussion`, `organ`, `acoustic_guitar`,
`clean_electric_guitar`, `distorted_electric_guitar`, `acoustic_bass`,
`electric_bass`, `violin`, `viola`, `cello`, `contrabass`,
`orchestral_harp`, `timpani`, `string_ensemble`, `synth_strings`, `voice`,
`orchestra_hit`, `trumpet`, `trombone`, `tuba`, `french_horn`,
`brass_section`, `soprano_and_alto_sax`, `tenor_sax`, `baritone_sax`, `oboe`,
`english_horn`, `bassoon`, `clarinet`, `flutes`, `synth_lead`, `synth_pad`, and
`drums`.

## Session Options (use with `--session-option`)

| Option | Values | Default | Meaning |
|---|---|---:|---|
| `muscriptor.weight_type` | `native`, `f32`, `f16`, `bf16`, `q8_0` | `native` | Transformer weight storage type. |
| `muscriptor.perf_mode` | `off`, `flash_attention` | `flash_attention` | Decoder attention mode. |
| `muscriptor.weight_context_mb` | integer MiB | `512` | Weight context arena size. |
| `muscriptor.conditioning_graph_arena_mb` | integer MiB | `128` | Condition graph arena size. |
| `muscriptor.decoder_prefill_graph_arena_mb` | integer MiB | `768` | Decoder prefill graph arena size. |
| `muscriptor.decoder_decode_graph_arena_mb` | integer MiB | `512` | Decoder cached-step graph arena size. |
