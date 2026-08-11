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

| Option | Values | Default | Meaning |
|---|---|---:|---|
| `--request-option instruments=<csv>` | instrument names | empty | Constrain generated MIDI events to instrument groups. |
| `--request-option output_format=midi\|json` | `midi`, `json` | `midi` | Select the primary output serialization written by `--out`. |
| `--out <file>` | path | not set | Write the selected primary output. Use `.mid` for MIDI or `.json` for event JSON. |
| `--text-out <json>` | path | not set | Optionally also write generated note-event JSON. |
| `--max-tokens` / `--request-option max_tokens=<n>` | integer | `2000` | Maximum generated MIDI-event tokens per chunk. |
| `--request-option do_sample=true\|false` | bool | `false` | Use sampling instead of greedy token selection. |
| `--temperature` | float | `1.0` | Sampling temperature. `0` selects deterministic argmax. |
| `--guidance-scale` / `--request-option guidance_scale=<f>` | float | `1.0` | Classifier-free guidance coefficient; `1` disables CFG. |
| `--num-beams` / `--request-option num_beams=<n>` | integer | `1` | Beam-search width; `1` disables beam search. |
| `--request-option batch_size=<n>` | integer | `1` | Number of chunks decoded per batch when `prelude_forcing=false`. |
| `--request-option prelude_forcing=true\|false` | bool | `true` | Force open-note prelude tokens between sequential chunks. |
| `--seed` | integer | `0` | Sampling seed. |
| `--session-option muscriptor.weight_type=<type>` | `native`, `f32`, `f16`, `bf16`, `q8_0` | `native` | Transformer weight storage type. |
| `--session-option muscriptor.perf_mode=<mode>` | `off`, `flash_attention` | `flash_attention` | Decoder attention mode. |
| `--session-option muscriptor.weight_context_mb=<mb>` | integer MiB | `512` | Weight context arena size. |
| `--session-option muscriptor.conditioning_graph_arena_mb=<mb>` | integer MiB | `128` | Condition graph arena size. |
| `--session-option muscriptor.decoder_prefill_graph_arena_mb=<mb>` | integer MiB | `768` | Decoder prefill graph arena size. |
| `--session-option muscriptor.decoder_decode_graph_arena_mb=<mb>` | integer MiB | `512` | Decoder cached-step graph arena size. |
