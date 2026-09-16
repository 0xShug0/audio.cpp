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

## AR LoRA

Load an unfused YuE2 AR adapter with session options:

```bash
--session-option yue2.lora=/path/to/ar_lora_inst_v3abc.safetensors \
--session-option yue2.lora_scale=1.0 \
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

## Request Options

| Option | Values | Default | Meaning |
|---|---|---:|---|
| `--lyrics` | text | required | Song lyrics. |
| `--text` | text | empty | Fallback lyrics source when `--lyrics` is not supplied. |
| `--request-option style=<text>` | text | required | Music style prompt. |
| `--request-option cot=<mode>` | `off`, `melody`, `full` | `full` | Symbolic planning route. |
| `--request-option abc=<text>` | ABC text | empty | Inline ABC score; requires `cot=melody` or `cot=full`. |
| `--request-option abc_file=<path>` | path | empty | ABC score file; requires `cot=melody` or `cot=full`. |
| `--request-option nar_noise_file=<path>` | raw float32 file | empty | Provide a noise file for NAR generation, shaped `[frames,64]`. |
| `--request-option guidance_scale=<f>` | `0..20` | `1.01` for `cot=off`, otherwise `1.0` | Semantic classifier-free guidance scale. Legacy alias: `cfg_scale`. |
| `--request-option num_inference_steps=<n>` | integer > 0 | `8` | NAR midpoint ODE steps. |
| `--seed <n>` | integer in `[0, 2^63)` | `1234` | Generation seed. Equivalent to `--request-option seed=<n>`. |

## Sampling Options

| Option | Values | Default | Meaning |
|---|---|---:|---|
| `--request-option abc_temperature=<f>` | `0..5` | `0.7` | ABC planner sampling temperature. |
| `--request-option abc_top_p=<f>` | `0..1` | `0.9` | ABC planner nucleus sampling probability. |
| `--request-option abc_top_k=<n>` | integer >= 1 | `30` | ABC planner top-k limit. |
| `--request-option abc_repetition_penalty=<f>` | float > 0 | `1.005` | ABC planner repetition penalty. |
| `--request-option abc_penalty_window=<n>` | integer >= 1 | `100` | ABC planner repetition penalty window. |
| `--request-option abc_min_tokens=<n>` | integer >= 0 | `32` | Minimum ABC planner tokens before EOS is accepted. |
| `--request-option abc_max_tokens=<n>` | integer >= `abc_min_tokens` | `4096` | Maximum ABC planner tokens. |
| `--request-option semantic_temperature=<f>` | `0..5` | `1.0` | Semantic codec sampling temperature. |
| `--request-option semantic_top_p=<f>` | `0..1` | `0.95` | Semantic codec nucleus sampling probability. |
| `--request-option semantic_top_k=<n>` | integer >= 1 | `100` | Semantic codec top-k limit. |
| `--request-option semantic_repetition_penalty=<f>` | float > 0 | `1.2` | Semantic codec repetition penalty. |
| `--request-option semantic_penalty_window=<n>` | integer >= 1 | `50` | Semantic codec repetition penalty window. |
| `--request-option semantic_min_tokens=<n>` | integer >= 0 | `200` | Minimum semantic tokens before EOS is accepted. |
| `--request-option semantic_max_tokens=<n>` | integer >= `semantic_min_tokens` | `9000` | Maximum semantic codec tokens. |

## Session Options

| Option | Values | Default | Meaning |
|---|---|---:|---|
| `--session-option yue2.model_gguf=<file>` | relative GGUF path | `yue2-3b-q8_0.gguf` | Main AR/NAR component. |
| `--session-option yue2.vae_gguf=<file>` | relative GGUF path | `yue2-vae-f16.gguf` | VAE component. |
| `--session-option yue2.lora=<file>` | safetensors path | none | Unfused AR adapter; relative paths use the model root. |
| `--session-option yue2.lora_scale=<float>` | finite float | `1.0` | Adapter delta scale; `0` disables it. |
| `--session-option yue2.weight_type=<type>` | `native`, `f32`, `f16`, `bf16`, `q8_0`, `q4_0`, `q4_k` | `native` | Shared weight storage fallback for the main model and VAE. |
| `--session-option yue2.model_weight_type=<type>` | `native`, `f32`, `f16`, `bf16`, `q8_0`, `q4_0`, `q4_k` | `native` | Main model weight storage override. |
| `--session-option yue2.vae_weight_type=<type>` | `native`, `f32`, `f16`, `bf16`, `q8_0`, `q4_0`, `q4_k` | `native` | VAE weight storage override. |
| `--session-option yue2.model_weight_context_mb=<n>` | MiB integer >= 1 | `6144` | Main model weight context size. |
| `--session-option yue2.vae_weight_context_mb=<n>` | MiB integer >= 1 | `1536` | VAE weight context size. |
| `--session-option yue2.ar_prefill_graph_arena_mb=<n>` | MiB integer >= 1 | `4096` | AR prefill graph arena size. |
| `--session-option yue2.ar_decode_graph_arena_mb=<n>` | MiB integer >= 1 | `1536` | AR one-token decode graph arena size. |
| `--session-option yue2.nar_graph_arena_mb=<n>` | MiB integer >= 1 | `6144` | NAR acoustic flow graph arena size. |
| `--session-option yue2.vae_graph_arena_mb=<n>` | MiB integer >= 1 | `1536` | VAE decode graph arena size. |
| `--session-option yue2.attention=<mode>` | `auto`, `flash`, `eager` | `auto` | NAR acoustic-flow attention kernel. `auto` uses flash, except on Volta/Turing CUDA GPUs (missing MMA kernels) and Intel Vulkan GPUs (eager measured 2.2x faster) where it uses eager; explicit `flash` / `eager` override the probe. The AR decode path always uses flash. |
