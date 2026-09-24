# MuScriptor

MuScriptor is an audio-to-symbolic tool that converts music audio into
note-event JSON or a MIDI file. The default download is the standalone F32
GGUF package; the original safetensors layout is still supported for local
development.

```bash
python3 tools/model_manager_v2.py install muscriptor
```

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

## Common Options (use directly)

| Option | Values | Default | Meaning |
|---|---|---:|---|
| `--out` | path | not set | Write the selected primary output. Use `.mid` for MIDI or `.json` for event JSON. |
| `--text-out` | path | not set | Optionally also write generated note-event JSON. |
| `--max-tokens` | integer | `2000` | Maximum generated MIDI-event tokens per chunk. |
| `--temperature` | float | `1.0` | Sampling temperature. `0` selects deterministic argmax. |
| `--guidance-scale` | float | `1.0` | Classifier-free guidance coefficient; `1` disables CFG. |
| `--num-beams` | integer | `1` | Beam-search width; `1` disables beam search. |
| `--seed` | integer | `0` | Sampling seed. |

## Request Options (use with `--request-option`)

| Option | Values | Default | Meaning |
|---|---|---:|---|
| `instruments` | comma-separated names listed below | empty | Constrain generated MIDI events to instrument groups. |
| `output_format` | `midi`, `json` | `midi` | Select the primary output serialization written by `--out`. |
| `max_tokens` | integer | `2000` | Maximum generated MIDI-event tokens per chunk. |
| `do_sample` | bool | `false` | Use sampling instead of greedy token selection. |
| `guidance_scale` | float | `1.0` | Classifier-free guidance coefficient; `1` disables CFG. |
| `num_beams` | integer | `1` | Beam-search width; `1` disables beam search. |
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
