# YuE2

YuE2 is wired as `--family yue2 --task gen`. It generates music from lyrics and
a style prompt, with optional symbolic ABC conditioning.

## Quick Start

Default packaged GGUF layout:

```bash
./build/debug/bin/audiocpp_cli \
  --task gen \
  --family yue2 \
  --model models/Yue2-3B-GGUF \
  --backend cuda \
  --threads 8 \
  --lyrics "[Verse]
Soft morning light is touching the window.
I hear the city waking below.
[Chorus]
Stay with the rhythm, let it carry us home.
Sing with the sunrise, we are never alone." \
  --request-option style="English, indie pop, bright acoustic guitar, soft drums, warm lead vocal, polished demo mix" \
  --request-option cot=off \
  --seed 1234 \
  --out yue2.wav \
  --log
```

The default session loads `yue2-3b-q8_0.gguf` for the main AR/NAR model and
`yue2-vae-f16.gguf` for the VAE from the model root.

## Model

| Field | Value |
|---|---|
| Family | `yue2` |
| Model directory | `models/Yue2-3B-GGUF` |
| Task | `gen` |
| Main GGUF default | `yue2-3b-q8_0.gguf` |
| VAE GGUF default | `yue2-vae-f16.gguf` |
| Required sidecars | `sidecars/yue2-model-config.json`, `sidecars/yue2-generation-config.json`, `sidecars/yue2-qwen.tiktoken`, `sidecars/yue2-vae-config.json` |
| Lyrics input | `--lyrics`; `--text` is accepted as a fallback |
| Style input | `--request-option style=<prompt>` |

## Component Selection

Select the BF16 main model:

```bash
./build/debug/bin/audiocpp_cli \
  --task gen \
  --family yue2 \
  --model models/Yue2-3B-GGUF \
  --backend cuda \
  --threads 8 \
  --session-option yue2.model_gguf=yue2-3b-bf16.gguf \
  --session-option yue2.vae_gguf=yue2-vae-f16.gguf \
  --lyrics "[Verse]
Soft morning light is touching the window.
[Chorus]
Stay with the rhythm, let it carry us home." \
  --request-option style="English, pop rock, bright guitars, clean drums, warm vocal" \
  --request-option cot=off \
  --seed 1234 \
  --out yue2-bf16.wav \
  --log
```

Select the F32 VAE:

```bash
--session-option yue2.vae_gguf=yue2-vae-f32.gguf
```

The component paths are relative to `--model`; absolute paths are rejected.

## AR and NAR LoRA

Use `yue2.ar_lora` for score/semantic planning and `yue2.nar_lora` for acoustic
rendering. Either adapter can be used alone or both together. Use unfused
SafeTensors files, not the `_comfyui` layouts. Changing adapters requires a new
session; merged weights are cached within that session, not merged per request.

For example, add the acoustic adapter alongside the AR adapter:

```bash
--session-option yue2.nar_lora=/path/to/nar_lora_joint_v4.safetensors \
--session-option yue2.nar_lora_scale=1.0
```

The [NAR adapter](https://huggingface.co/Mothersuperior/yue2-mothersuperior-realaudio-tokenizer-v4)
includes full `vae2llm` and `llm2vae` projection replacements in addition to LoRA
deltas. Its scale applies only to deltas; replacements are loaded at full strength.
A scale of `0` disables the entire adapter, including replacements. No separate
audio-tokenizer head is needed for text-to-music generation.

Load an unfused YuE2 AR adapter with session options:

```bash
--session-option yue2.ar_lora=/path/to/ar_lora_inst_v3abc.safetensors \
--session-option yue2.ar_lora_scale=1.0 \
--request-option cot=full
```

Both FP32 and BF16 adapter files are supported. Use the standard A/B files from
[the instrumental adapter repository](https://huggingface.co/Mothersuperior/YuE2-instrumental-cot-full-loras),
not its ComfyUI fused file. NAR adapters are not supported by this option.
Relative adapter paths resolve under the model root. Reload the session after
changing the adapter or scale; `0` disables it. With no adapter, weight loading
and generation are unchanged.

The adapter modifies AR attention and MLP projections at load time, using
`BF16(W + BF16(scale * (B @ A)))`. There is no extra alpha/rank scaling or
per-token adapter computation. NAR and VAE weights are unchanged.
Merged AR tensors are cached in CPU memory in their upload dtype for the session,
so later requests skip merging. Only one dtype per adapted tensor is retained;
unloading the session frees the cache. AR weights are still released from VRAM
before VAE decoding.
Use the BF16 main GGUF for the closest match to the original weights. Loading
onto Q8/Q4 bases merges into dequantized weights and requantizes the result;
it is not equivalent to merging into the original BF16 model first.

For this instrumental adapter, use `cot=full`, instrument/style tags in `style`,
and `[instrumental]` or section tags in `lyrics`. Its license is CC BY-NC 4.0.

## ABC Conditioning

Use `cot=melody` or `cot=full` to run the symbolic route. External ABC requires
one of those modes:

```bash
./build/debug/bin/audiocpp_cli \
  --task gen \
  --family yue2 \
  --model models/Yue2-3B-GGUF \
  --backend cuda \
  --threads 8 \
  --lyrics "[Verse]
Write the melody over this score." \
  --request-option style="English, folk pop, acoustic guitar, steady drums" \
  --request-option cot=melody \
  --request-option abc_file=/path/to/score.abc \
  --seed 1234 \
  --out yue2-abc.wav \
  --log
```

Inline ABC can be passed with `--request-option abc=<abc text>`.

## Generated ABC Export

When the model generates its own plan (`cot=melody` or `cot=full` with no
external `abc` / `abc_file`), the decoded ABC score is attached to the result
as a `score` artifact (`text/vnd.abc`). The WebUI shows a Save ABC download in
the result panel, and the CLI writes `score.abc` when `--out-dir` is set:

```bash
./build/debug/bin/audiocpp_cli \
  --task gen \
  --family yue2 \
  --model models/Yue2-3B-GGUF \
  --backend cuda \
  --threads 8 \
  --lyrics "..." \
  --request-option style="English, folk pop" \
  --request-option cot=full \
  --seed 1234 \
  --out yue2.wav \
  --out-dir yue2_out \
  --log
# -> yue2_out/score.abc
```

## Semantic Token Export

Set `export_semantic=true` to attach the semantic stage output as a `semantic`
artifact (`application/vnd.yue2.semantic+json`). The payload is a flat JSON
array of codec indices, one integer per semantic frame (25 frames per second)
in `[0, 32768)`, without the stop token. The CLI writes `semantic.json` when
`--out-dir` is set:

```bash
./build/debug/bin/audiocpp_cli \
  --task gen \
  --family yue2 \
  --model models/Yue2-3B-GGUF \
  --backend cuda \
  --threads 8 \
  --lyrics "..." \
  --request-option style="English, folk pop" \
  --request-option export_semantic=true \
  --seed 1234 \
  --out yue2.wav \
  --out-dir yue2_out \
  --log
# -> yue2_out/semantic.json
```

The artifact meta carries `frames` (the number of indices) and `truncated`
(`true` when the stage stopped on `semantic_max_tokens` instead of the stop
token).

The indices are the same values the reference YuE2 Python pipeline stores in its
`semantic.npy` (1-D `int32`). To convert:

```bash
python3 -c "import json, numpy; numpy.save('semantic.npy', numpy.array(json.load(open('semantic.json')), dtype=numpy.int32))"
```

## Stopping Early

`stop_after` ends the run before the remaining stages:

| Value | Runs | Produces |
|---|---|---|
| `abc` | ABC planner | `score` artifact |
| `semantic` | ABC planner, semantic AR | `score` and `semantic` artifacts |
| `audio` | everything | audio, plus `score`; `semantic` with `export_semantic=true` |

`stop_after=abc` requires `cot=melody` or `cot=full` and no external
`abc` / `abc_file`, because there would be nothing to generate.
`stop_after=semantic` implies `export_semantic=true`, since the token stream is
all that stage produces.

A result from `abc` or `semantic` carries **no audio**. Use `--out-dir` on the
CLI to collect the artifacts (`--out` writes nothing), and `/v1/tasks/run` on
the server — `/v1/audio/speech` requires an audio output.

```bash
./build/debug/bin/audiocpp_cli \
  --task gen \
  --family yue2 \
  --model models/Yue2-3B-GGUF \
  --backend cuda \
  --threads 8 \
  --lyrics "..." \
  --request-option style="English, folk pop" \
  --request-option stop_after=semantic \
  --seed 1234 \
  --out-dir yue2_out \
  --log
# -> yue2_out/score.abc, yue2_out/semantic.json
```

## Continuing From Semantic Tokens

`semantic_prefix` takes a JSON array of semantic codec indices, one per frame at
25 frames per second, each in `[0,32768)`. The frames become forced history: the
AR stage prefills them behind the prompt and samples the rest of the song from
frame `N`. `semantic_prefix_file` reads the same text from a file and is ignored
when `semantic_prefix` is set.

The returned stream, and therefore the NAR stage and the rendered audio, includes
the forced frames. `semantic_min_tokens`, `semantic_max_tokens` and the
repetition penalty window all count the total stream, so `semantic_max_tokens`
must be at least `N`.

```bash
./build/debug/bin/audiocpp_cli \
  --task gen \
  --family yue2 \
  --model models/Yue2-3B-GGUF \
  --backend cuda \
  --threads 8 \
  --lyrics "..." \
  --request-option style="English, folk pop" \
  --request-option cot=off \
  --request-option semantic_prefix_file=/path/to/semantic.json \
  --request-option semantic_max_tokens=1200 \
  --seed 1234 \
  --out yue2-continue.wav \
  --log
```

With `cot=melody` or `cot=full` a prefix also requires `abc` or `abc_file`:
without a score the run would plan a new one that the forced frames do not
belong to.

To render exactly the given frames and sample nothing, set both token bounds to
`N`:

```bash
  --request-option semantic_min_tokens=640 \
  --request-option semantic_max_tokens=640
```

That run skips the AR stage: the given frames are the stream, and with
`stop_after=semantic` they are returned as they came. `stop_after=audio` still
prefills them once for the NAR conditioning.

Results are deterministic for a given request, but a prefix does not reproduce
the draws of an uninterrupted run: the sampler RNG starts at the first sampled
frame, and prefilled K/V differ numerically from step-decoded K/V.

With a guidance scale other than `1.0` the forced frames are appended to both the
positive and the negative prefix, and that path prefills through host K/V, so
memory grows with `N`. The default `cot=full` route uses scale `1.0`, which runs a
single stream on the device.

A prefix does not have to come from the run it continues. Splicing the tokens of
two renders of one score changes a song's style part-way through;
[examples/yue2_style_change](../../examples/yue2_style_change/) is a complete
script for it.

## Common Options (use directly)

| Option | Values | Default | Meaning |
|---|---|---:|---|
| `--lyrics` | text | required | Song lyrics. |
| `--text` | text | empty | Fallback lyrics source when `--lyrics` is not supplied. |
| `--seed` | integer in `[0, 2^63)` | `1234` | Generation seed. |

## Request Options (use with `--request-option`)

| Option | Values | Default | Meaning |
|---|---|---:|---|
| `style` | text | required | Music style prompt. |
| `cot` | `off`, `melody`, `full` | `full` | Symbolic planning route. |
| `stop_after` | `abc`, `semantic`, `audio` | `audio` | Last stage to run. See "Stopping Early". |
| `abc` | ABC text | empty | Inline ABC score; requires `cot=melody` or `cot=full`. |
| `abc_file` | path | empty | ABC score file; requires `cot=melody` or `cot=full`. |
| `nar_noise_file` | raw float32 file | empty | Provide a noise file for NAR generation, shaped `[frames,64]`. |
| `export_semantic` | `true`, `false` | `false` | Attach the semantic token stream as a `semantic` artifact. |
| `semantic_prefix` | JSON array of codec indices | empty | Inline semantic frames to force at the start of the music stream. |
| `semantic_prefix_file` | path | empty | File holding the same JSON array; ignored when `semantic_prefix` is set. |
| `guidance_scale` | `0..20` | `1.01` for `cot=off`, otherwise `1.0` | Semantic classifier-free guidance scale. Legacy alias: `cfg_scale`. |
| `num_inference_steps` | integer > 0 | `8` | NAR midpoint ODE steps. |
| `seed` | integer in `[0, 2^63)` | `1234` | Generation seed. Equivalent to `--seed <n>`. |

## Sampling Options (use with `--request-option`)

| Option | Values | Default | Meaning |
|---|---|---:|---|
| `abc_temperature` | `0..5` | `0.7` | ABC planner sampling temperature. |
| `abc_top_p` | `0..1` | `0.9` | ABC planner nucleus sampling probability. |
| `abc_top_k` | integer >= 1 | `30` | ABC planner top-k limit. |
| `abc_repetition_penalty` | float > 0 | `1.005` | ABC planner repetition penalty. |
| `abc_penalty_window` | integer >= 1 | `100` | ABC planner repetition penalty window. |
| `abc_min_tokens` | integer >= 0 | `32` | Minimum ABC planner tokens before EOS is accepted. |
| `abc_max_tokens` | integer >= `abc_min_tokens` | `4096` | Maximum ABC planner tokens. |
| `semantic_temperature` | `0..5` | `1.0` | Semantic codec sampling temperature. |
| `semantic_top_p` | `0..1` | `0.95` | Semantic codec nucleus sampling probability. |
| `semantic_top_k` | integer >= 1 | `100` | Semantic codec top-k limit. |
| `semantic_repetition_penalty` | float > 0 | `1.2` | Semantic codec repetition penalty. |
| `semantic_penalty_window` | integer >= 1 | `50` | Semantic codec repetition penalty window. |
| `semantic_min_tokens` | integer >= 0 | `200` | Minimum semantic tokens before EOS is accepted. |
| `semantic_max_tokens` | integer >= `semantic_min_tokens` | `9000` | Maximum semantic codec tokens. |

## Session Options (use with `--session-option`)

| Option | Values | Default | Meaning |
|---|---|---:|---|
| `yue2.model_gguf` | relative GGUF path | `yue2-3b-q8_0.gguf` | Main AR/NAR component. |
| `yue2.vae_gguf` | relative GGUF path | `yue2-vae-f16.gguf` | VAE component. |
| `yue2.ar_lora` | safetensors path | none | Unfused AR adapter; relative paths use the model root. |
| `yue2.ar_lora_scale` | finite float | `1.0` | Adapter delta scale; `0` disables it. |
| `yue2.nar_lora` | safetensors path | none | Unfused NAR adapter; relative paths use the model root. |
| `yue2.nar_lora_scale` | finite float | `1.0` | NAR delta scale; replacements stay at full strength. `0` disables the entire adapter. |
| `yue2.weight_type` | `native`, `f32`, `f16`, `bf16`, `q8_0`, `q4_0`, `q4_k` | `native` | Shared weight storage fallback for the main model and VAE. |
| `yue2.model_weight_type` | `native`, `f32`, `f16`, `bf16`, `q8_0`, `q4_0`, `q4_k` | `native` | Main model weight storage override. |
| `yue2.vae_weight_type` | `native`, `f32`, `f16`, `bf16`, `q8_0`, `q4_0`, `q4_k` | `native` | VAE weight storage override. |
| `yue2.model_weight_context_mb` | MiB integer >= 1 | `6144` | Main model weight context size. |
| `yue2.vae_weight_context_mb` | MiB integer >= 1 | `1536` | VAE weight context size. |
| `yue2.ar_prefill_graph_arena_mb` | MiB integer >= 1 | `4096` | AR prefill graph arena size. |
| `yue2.ar_decode_graph_arena_mb` | MiB integer >= 1 | `1536` | AR one-token decode graph arena size. |
| `yue2.nar_graph_arena_mb` | MiB integer >= 1 | `6144` | NAR acoustic flow graph arena size. |
| `yue2.vae_graph_arena_mb` | MiB integer >= 1 | `1536` | VAE decode graph arena size. |
| `yue2.attention` | `auto`, `flash`, `eager` | `auto` | NAR acoustic-flow attention kernel. `auto` uses flash, except on Volta/Turing CUDA GPUs (missing MMA kernels) and Intel Vulkan GPUs (eager measured 2.2x faster) where it uses eager; explicit `flash` / `eager` override the probe. The AR decode path always uses flash. |
| `yue2.attention_tile_rows` | integer >= 0 | `0` | Query rows per tile in the eager NAR attention. The eager lowering holds the whole score matrix, which grows with the square of the song length; `0` splits the query rows into as few equal tiles as keep one tile's scores under 3 GiB, which is what long songs need on drivers that cap a single buffer at 4 GiB. A song whose scores already fit runs as one tile. Ignored by the flash kernel. |

## Parity Probes

The parity probes compare the port against tensors dumped from the Python
reference without adding reference-only inputs to the request surface. Build
them with `-DENGINE_BUILD_TESTS=ON` (or `-DENGINE_BUILD_EXTENDED_TESTS=ON` /
`-DENGINE_BUILD_MODEL_TESTS=ON`).

| Probe | Reference dump | Compares |
|---|---|---|
| `yue2_vae_parity_probe` | `tests/yue2/yue2_vae_reference_dump.py` | VAE encode/decode planar tensors. |
| `yue2_nar_parity_probe` | `tests/yue2/yue2_nar_reference_dump.py` | NAR acoustic-flow latents. |

`yue2_nar_reference_dump.py` runs ABC and semantic sampling greedily, draws the
same fp32 noise `yue2.nar.song_chunks` uses, and writes `prefix.i32`,
`codec.i32`, `noise.f32`, `latents_ref.f32` and `metadata.json`.
`yue2_nar_parity_probe` feeds exactly those tensors through `Yue2ArRuntime` /
`Yue2NarRuntime` and reports `max_abs`, `rmse` and `cosine` against
`latents_ref.f32`:

```bash
python tests/yue2/yue2_nar_reference_dump.py --reference-root reference/YuE \
    --model models/YuE2-3B --vae models/YuE2-Vae --out-dir /tmp/yue2nar
build/bin/yue2_nar_parity_probe --model models/Yue2-3B-GGUF \
    --reference /tmp/yue2nar --backend cuda
```

Measured cosine is `>= 0.9999` with the bf16 package and `>= 0.995` with the
shipped q8_0 package, which accumulates more error over long prefixes. Use
`--min-cosine` / `--max-rmse` for a stricter gate, and `--model-gguf` /
`--weight-type` to match the deployment under test.
