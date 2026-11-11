# Fish Audio Drama 3

`fish_drama3` is a community model port of
[fishaudio/drama-3](https://huggingface.co/fishaudio/drama-3). Inference runs in
native audio.cpp with GGML; Python is used to prepare the checkpoint.

Built with Fish Audio.

The base model includes the 40-layer hybrid attention/MoE backbone,
256 experts with eight selected per token, four shared enrichment layers,
nine residual codebook layers with persistent timeline caches, and the Fish DAC
encoder/decoder. It supports offline and streaming TTS, voice references,
speaker and event tags in text, long-form chunking, and continuation/edit history.

## Windows build

Install Visual Studio 2022 C++ build tools, a CUDA toolkit that supports your GPU,
and Git. Use the Drama 3 branch and initialize its public frontend submodule:

```powershell
git clone --branch drama-3-native https://github.com/JarodMica/audio.cpp.git
Set-Location audio.cpp
git -c submodule.external/audio.cpp-server-frontends.url=https://github.com/0xShug0/audio.cpp-server-frontends.git submodule update --init --recursive
```

The command-local HTTPS override lets the public frontend submodule download
without GitHub SSH credentials. It does not modify `.gitmodules` or global Git
configuration.

Run the build helper from the repository root. It selects the required model
dependencies and detects the CUDA architecture:

```powershell
.\scripts\build_windows.ps1 -Preset windows-cuda-release -ModelSet custom -Models fish_drama3 -Target audiocpp_cli -Jobs 4
.\scripts\build_windows.ps1 -Preset windows-cuda-release -ModelSet custom -Models fish_drama3 -Target audiocpp_server -Jobs 4
```

The executables are in `build\windows-cuda-release\bin`. A CPU runtime uses the
`windows-cpu-release` preset with the same model selection. To override automatic
CUDA detection, add `-CudaArchitectures` with the architecture for your GPU.

## Download a runnable GGUF

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then download
one package from [Jmica/Drama-3-GGUF](https://huggingface.co/Jmica/Drama-3-GGUF).
For the Q2 expert package:

```powershell
uv run --no-project --python 3.12 --with huggingface_hub hf download `
  Jmica/Drama-3-GGUF drama-3-backbone-8bit-experts-Q2_K.gguf `
  --local-dir models\Drama-3-GGUF
```

### Checkpoint versions

The default files in [Jmica/Drama-3-GGUF](https://huggingface.co/Jmica/Drama-3-GGUF)
use the October 29, 2026 upstream checkpoint,
[revision `78e5ab7`](https://huggingface.co/fishaudio/drama-3/tree/78e5ab79e49aa9a25abe8799eff2c4d73a1c3216),
labeled V2 by the conversion repository. The original V1 packages, based on
upstream revision `dd64a85`, remain available at
[revision `original-2026-10-13`](https://huggingface.co/Jmica/Drama-3-GGUF/tree/original-2026-10-13).
V1 and V2 are GGUF release labels, not official Fish Audio model-version names.

Both releases retain the same filenames. Select the revision explicitly and
use separate directories when keeping both. For example, to download V1:

```powershell
uv run --no-project --python 3.12 --with huggingface_hub hf download `
  Jmica/Drama-3-GGUF drama-3-backbone-8bit-experts-Q2_K.gguf `
  --revision original-2026-10-13 --local-dir models\Drama-3-GGUF-V1
```

The model manager lists the base and small packages:

```powershell
uv run --no-project --python 3.12 tools\model_manager_v2.py list | Select-String 'fish_drama3'
uv run --no-project --python 3.12 tools\model_manager_v2.py install fish_drama3_q2_experts_q8_backbone --models-root models
```

| File | Size | Expert banks | Other backbone matrices |
|---|---:|---|---|
| `drama-3-backbone-8bit-experts-Q4_K_S.gguf` | 22.91 GiB | Q4_K | Q8_0 |
| `drama-3-backbone-8bit-experts-Q3_K_S.gguf` | 18.93 GiB | Q3_K | Q8_0 |
| `drama-3-backbone-8bit-experts-Q2_K.gguf` | 15.88 GiB | Q2_K | Q8_0 |
| `drama-3-backbone-4bit-experts-Q4_K_S.gguf` | 21.77 GiB | Q4_K | Q4_K, with Q4_0 text embedding |
| `drama-3-backbone-8bit-F32-routers-experts-Q4_K_S.gguf` | 22.97 GiB | Q4_K | Q8_0, with FP32 routers/shared-expert gates |
| `drama-3-backbone-8bit-F32-routers-BF16-embed-head-experts-Q4_K_S.gguf` | 23.87 GiB | Q4_K | Q8_0, with FP32 routers/shared-expert gates and BF16 text embedding/output head |

These packages use standard GGML K-quants. The Q3_K_S and Q4_K_S file profiles
use pure expert banks: all 120 routed expert projection banks have uniform
Q3_K or Q4_K storage. Their storage-type selection corresponds to a standard
`--pure` K profile with explicit precision overrides for the dense backbone,
text embedding and audio components. Architecture-specific higher-bit
promotion heuristics are not applied; an unmodified default llama.cpp preset
can choose different types for some tensors. See the standard converter's
[pure and per-tensor options](https://github.com/ggml-org/llama.cpp/blob/master/tools/quantize/quantize.cpp).
The native converter takes `--expert-type q3_k` or `--expert-type q4_k`.
`backbone-8bit` means Q8_0 backbone matrices; `backbone-4bit` means Q4_K matrices
with a Q4_0 text embedding. Q2 experts use uniform Q2_K storage.

All six packages retain FP16 audio-tower matrices and waveform codec weights.
Backbone and audio-tower normalization vectors stay FP32; codec normalization
vectors use FP16 with the rest of the codec. The packages embed their tokenizer,
configuration and model contract. Stock audio.cpp releases without this port
cannot run Drama 3.

## Generate and stream

Host sampling is used by default. Set
`--session-option fish_drama3.device_sampling=true` to enable tower decode and
codebook sampling in one cached backend graph, or select the `cuda_perf` preset.
Graph probability arithmetic is F32, so sampled outputs may differ from host sampling.

`fish_drama3.perf_mode` defaults to `off`. The opt-in `cuda_perf` preset sets
the following defaults; explicit session options override the preset:

| Session option | `off` | `cuda_perf` | Effect |
|---|---|---|---|
| `fish_drama3.fused_moe` | `false` | `true` | Cached-decode routing and SwiGLU fusion. |
| `fish_drama3.kv_cache_type` | `f16` | `f16` | KV storage precision, not weight precision. Set `f32` to use F32 KV storage. |
| `fish_drama3.kv_cache_alignment` | `1` | `256` | Cached attention-prefix buckets; all history retained. |
| `fish_drama3.device_sampling` | `false` | `true` | Cached tower decode and device sampling. |
| `fish_drama3.codec_batch_frames` | `4` | `4` | Streaming codec batch ceiling; existing default retained. |
| `fish_drama3.codec_async_decode` | `true` | `true` | Streaming AR/codec overlap; existing default retained. |
| `fish_drama3.backbone_prefill_cache_mb` | `256` | `256` | Backbone prefill graph-cache budget in MiB; existing default retained. |
| `fish_drama3.tower_prefill_cache_mb` | `256` | `256` | Tower prefill graph-cache budget in MiB; existing default retained. |

`off` retains the defaults listed above. Explicit session options override either
preset, including `fish_drama3.device_sampling=false` with `cuda_perf`.

Fused arithmetic and attention shapes can change sampled audio. To enable the
combined CUDA path, add:

```text
--session-option fish_drama3.perf_mode=cuda_perf
```

Streaming defaults to a four-frame codec batch ceiling with AR/codec overlap.
To use one-frame synchronous decoding:

```text
--session-option fish_drama3.codec_batch_frames=1
--session-option fish_drama3.codec_async_decode=false
```

Batching starts with one frame, grows to the configured ceiling, and flushes the
final partial batch. Larger batches retain more codec workspace and can introduce
small waveform rounding differences without changing AR sampling. Overlap uses
the dedicated codec backend and completes all audio before the request returns.
Offline decoding is unchanged.

```powershell
.\build\windows-cuda-release\bin\audiocpp_cli.exe `
  --task tts --family fish_drama3 --backend cuda --mode streaming `
  --model models\Drama-3-GGUF\drama-3-backbone-8bit-experts-Q2_K.gguf `
  --text 'A quick test of Drama Three on Windows.' `
  --max-tokens 500 --seed 123 --threads 8 `
  --session-option fish_drama3.cpu_expert_layers=8 `
  --session-option fish_drama3.context_length=2048 `
  --out speech.wav
```

For the tested 16 GB offload configurations, use 8 CPU expert layers with Q2,
12 with Q3, 24 with Q4-experts, or 20 with Q4-backbone. For a 24 GB configuration,
use 4 with Q4-experts or 0 with the other packages. These are starting settings,
not enforced memory limits. They were measured on a 96 GB GPU; physical 16 GB
and 24 GB cards have not been tested. These counts control where expert weights and matmuls run.
The backbone's attention/dense weights and the audio tower use the selected
backend. `fish_drama3.codec_backend=cpu` moves codec computation to CPU as well.

If Vulkan voice cloning produces incomplete speech or reaches the token limit,
try `--session-option fish_drama3.codec_encoder_f32_accumulation=true`.

This opt-in control uses F32 accumulation in the Vulkan reference encoder while
keeping F16 storage. It defaults to false and does not change codec decoding or
other backends. It also works with existing GGUFs without replacing their specs.

On CUDA, BF16 reference-encoder arithmetic is enabled by default with F16/BF16
codec weights. Set `--session-option fish_drama3.codec_encoder_bf16_compute=false`
to restore the previous encoder arithmetic. This changes reference-conditioned
generation but does not change codec decoding or backbone inference.

To reduce reference-encoder convolution memory, set
`--session-option fish_drama3.codec_encoder_chunk_samples=65536`.
The value is measured in samples at the codec sample rate and must be zero
(the default, full encoding) or a positive multiple of 512. Chunking starts
when the padded reference exceeds the chunk size plus 16,384 context samples;
shorter references use the full graph. Overlapping convolution windows reuse
one cached graph; the transformer and quantizer still process the full feature
sequence, so total memory is not constant. Chunking can cost additional time
and produce small backend-dependent numerical differences. Existing GGUFs
accept the option without a spec override.

Sampling defaults to `top_k=4097`, which keeps every candidate for the semantic
token and each tower codebook, so only `top_p=0.9` filters. This matches Fish's
official Drama 3 Space, which dropped the top-k cap because `top_k=30` left takes
stuck in silence. With `--request-option top_k=30`, long chunked requests can
stall for about 9-21 s. The repetition-avoiding resample keeps its own
`top_k=30`, `top_p=0.95`.

Host memory is required for offloaded weights. Context capacity includes text,
reference audio frames, prior takes, and the requested generation budget.

## Model

| Field | Value |
|---|---|
| Family | `fish_drama3` |
| Tasks | `tts`, `clon` |
| Modes | `offline`, `streaming` |
| Languages | `auto` |
| Default package | `fish_drama3_q4_experts_q8_backbone` |
| Input | Text; voice conditioning as shown in the examples |
| Output | Generated speech audio |

“Model/session default” means no explicit override. The runtime uses the
checkpoint or session setting, or leaves the optional value unset as described.

## Common Options (use directly)

| Option | Meaning |
|---|---|
| `--model` | Model file or directory shown in the examples. |
| `--backend` | Select an available backend supported by this model. |
| `--threads` | CPU thread count. |
| `--log` | Enable runtime logging. |
| `--text` | Input text; see the examples for the expected prompt format. |
| `--voice-ref` | Reference audio for voice cloning. |
| `--out` | Output WAV file. |

## Request Options (use with `--request-option`)

| Option | Type / values | Default | Description |
|---|---|---|---|
| `encode_reference_only` | `bool` | `false` | Prepare reference VQ codes without generating speech. Use the task API to receive drama3_reference_v1 acoustic-token artifacts. |
| `reference_text` | `string` | Model/session default | Transcript of inline reference audio. |
| `multi_reference_cond` | `string` | Model/session default | Ordered references as JSON: [{text, audio_path or vq_codes}]. Speaker tags 0..15 select voices. |
| `references` | `string` | Model/session default | Alias of multi_reference_cond. |
| `previous_takes` | `string` | Model/session default | Continuation/edit history as a JSON array of {text, vq_codes or audio_path}; a string generates that earlier take first. |
| `previous_takes_file` | `string` | Model/session default | JSON history file containing a take object or an ordered array of earlier takes. Avoids the Windows command-line length limit for VQ grids. |
| `max_tokens` | `int` | `1024` | Maximum generated acoustic frames per text chunk (about 21.5 frames/second). Declared range: minimum 1. |
| `text_chunk_size` | `int` | `128` | Maximum characters per long-form chunk. Declared range: minimum 1. |
| `skip_empty_chunks` | `bool` | `false` | Log and skip chunks that emit EOS before producing audio. The request fails if all chunks are empty. |
| `text_chunk_mode` | `word_budget`, `tag_aware`, `japanese`, `endline` | `tag_aware` | Framework text chunking mode. |
| `temperature` | `float` | `1.0` | Main and residual sampling temperature. Zero makes the initial draws greedy; semantic repeat redraws retain the fixed Drama 3 RAS settings. Declared range: minimum 0. |
| `top_p` | `float` | `0.9` | Inclusive nucleus cutoff before top-k filtering, matching Drama 3. Declared range: minimum 0, maximum 1. |
| `top_k` | `int` | `4097` | Top-k candidate count. The default 4097 keeps every candidate (no cap), as Fish's Drama 3 Space does; 30 can stall in long silences. Declared range: minimum 1. |
| `seed` | `int` | Model/session default | Reproducible native sampling seed. Declared range: minimum 0. |

## Session Options (use with `--session-option`)

| Option | Type / values | Default | Description |
|---|---|---|---|
| `fish_drama3.perf_mode` | `off`, `cuda_perf` | `off` | AR decode preset. cuda_perf enables fused MoE, 256-row attention buckets and device sampling. F16 KV storage, codec batching/overlap and prefill-cache defaults remain unchanged in both modes. Explicit session options override the preset. |
| `fish_drama3.fused_moe` | `bool` | `false` | Fuse cached backbone MoE routing and SwiGLU activations. Fused arithmetic may change sampled output; prompt prefill is unchanged. |
| `fish_drama3.kv_cache_alignment` | `int` | `1` | Cached decode attention prefix bucket size. Values above 1 retain all history while skipping unused KV capacity; execution shapes may change numerical results. Declared range: minimum 1. |
| `fish_drama3.kv_cache_type` | `f32`, `f16` | `f16` | Storage precision for full-attention backbone KV caches. |
| `fish_drama3.backbone_prefill_cache_mb` | `int` | `256` | Bounded cache of backbone prefill graphs, in MiB. Reuses exact shapes and their CUDA captures. One active graph is retained even if larger than this budget. Zero retains only the active graph. Declared range: minimum 0, maximum 4096. |
| `fish_drama3.tower_prefill_cache_mb` | `int` | `256` | Bounded cache of audio-tower prefill graphs, in MiB. Reuses exact shapes and their CUDA captures. Zero disables it. Declared range: minimum 0, maximum 4096. |
| `fish_drama3.codec_batch_frames` | `int` | `4` | Streaming codec batch ceiling. Starts at one frame and grows to this size; partial final batches are flushed. Offline decoding is unchanged. Declared range: minimum 1. |
| `fish_drama3.codec_async_decode` | `bool` | `true` | Overlap streaming codec decoding with AR generation on the dedicated codec backend. Uses a bounded queue and waits for all audio before completing the request. |
| `fish_drama3.device_sampling` | `bool` | `false` | Enable tower decode and codebook sampling in one cached backend graph. Disabled by default; enabled by cuda_perf unless explicitly overridden. Uses F32 probabilities; sampled outputs may differ from the host sampler. |
| `fish_drama3.profile_first_audio` | `bool` | `false` | Emit first-audio stage timings to stderr for performance diagnosis. Disabled by default. |
| `fish_drama3.context_length` | `int` | `2048` | Cache capacity for prompt plus generated frames; includes references and history. Declared range: minimum 1. |
| `fish_drama3.cpu_expert_layers` | `int` | `4` | Number of the 40 expert banks kept and evaluated on CPU; use 4 for a 24 GB profile or 24 for a 16 GB profile. Dense layers and the tower use the selected backend. Declared range: minimum 0, maximum 40. |
| `fish_drama3.disk_experts` | `bool` | `false` | Read routed expert weights from GGUF on demand into a bounded cache instead of keeping full expert banks resident. Trades speed for memory; default false. |
| `fish_drama3.pin_experts` | `bool` | `false` | Keep full expert banks in pinned host RAM for direct cache-miss uploads. Requires disk_experts and pinned host-buffer support; default false. |
| `fish_drama3.expert_cache_size` | `int` | `32` | Resident expert slots per layer in disk_experts mode; 8 to 256. Larger caches trade memory for fewer disk reads; ignored in resident mode. Declared range: minimum 8, maximum 256. |
| `fish_drama3.disk_expert_read_threads` | `int` | `1` | Host workers for copying selected GGUF experts into staging when disk_experts is enabled and pin_experts is false. Default 1 preserves serial reads; values above 1 use OpenMP or Apple dispatch. Declared range: minimum 1, maximum 64. |
| `fish_drama3.codec_encoder_f32_accumulation` | `bool` | `false` | Use F32 accumulation for F16 reference-encoder GEMMs on Vulkan. Try this if voice cloning produces incomplete speech or reaches the token limit. Other backends and codec decoding are unchanged. |
| `fish_drama3.codec_encoder_bf16_compute` | `bool` | `true` | Use BF16 reference-encoder arithmetic on CUDA with F16/BF16 codec weights; default true. Decoding is unchanged. |
| `fish_drama3.codec_encoder_chunk_samples` | `int` | `0` | Reference-encoder convolution chunk size in samples at the codec sample rate. Use 0 for full encoding or a positive multiple of 512; default 0. Declared range: minimum 0. |
| `fish_drama3.codec_backend` | `same`, `cpu` | `same` | Run the codec on the selected backend or on CPU. |

## Variants

Both variants use `--family fish_drama3` and the same CLI and server routes.
The loader reads layer-specific expert counts and the output-head layout from
the checkpoint, so Drama3-small does not need a separate family.

| Variant | GGUF packages | Source |
|---|---|---|
| Drama3 base | Q2_K, Q3_K, and Q4_K expert packages | [Jmica/Drama-3-GGUF](https://huggingface.co/Jmica/Drama-3-GGUF) |
| Drama3-small | `fish_drama3_small_iq3_xxs`: `Drama3-small-GGUF/drama3-small-IQ3_XXS.gguf` | [pentacoxian-dev/drama3-small-GGUF](https://huggingface.co/pentacoxian-dev/drama3-small-GGUF) |

Install the small variant with:

```bash
python tools/model_manager_v2.py install fish_drama3_small_iq3_xxs --models-root models
```

Use `--model models/Drama3-small-GGUF/drama3-small-IQ3_XXS.gguf` with the
generation examples below. Base-model weight sizes and memory measurements
in this guide do not describe the small checkpoint.

## Prepare weights with uv

Conversion is optional when using a downloaded package. Build the native
converter if you want to prepare a different precision:

```powershell
.\scripts\build_windows.ps1 -Preset windows-cpu-release -ModelSet custom -Models fish_drama3 -Target audiocpp_gguf -Jobs 4
```

Download the complete official checkpoint into `models\drama-3-original`.
The example below pins the original V1 checkpoint used for the earlier port
checks. To prepare V2, select revision
`78e5ab79e49aa9a25abe8799eff2c4d73a1c3216` and use separate checkpoint and output
paths so the original files remain available:

```powershell
uv run --no-project --python 3.12 --with huggingface_hub hf download fishaudio/drama-3 --revision dd64a85cd82e9521f7202610ede97b367fc83946 --local-dir models\drama-3-original
```

Its original per-expert FP8 tensors have 128-by-128 inverse scales. The
preparation script applies those scales and packs each projection into an
expert bank.

```powershell
uv run --python 3.12 tools\community_models\convert_fish_drama3.py `
  --checkpoint models\drama-3-original `
  --output models\drama3-f16-local.gguf `
  --quantized-output models\drama-3-backbone-8bit-experts-Q4_K_S.gguf `
  --gguf-tool build\windows-cpu-release\bin\audiocpp_gguf.exe
```

The intermediate tensor archive is approximately 68.2 GiB. The runnable package
is approximately 22.9 GiB, with Q4_K routed experts, Q8_0 backbone matrices,
FP16 audio tower and codec weights, and embedded configuration, tokenizer and
license files. Codec weight normalization is folded during packaging. Allow
space for both files during conversion. The intermediate archive is retained.

An existing intermediate can be reused with `--input-gguf` in place of `--output`.
`--expert-type` selects the routed expert format; `--backbone-type` selects other
backbone matrices; `--embedding-type` selects the text embedding format. Keep the
embedding at a CUDA-supported lookup type such as Q8_0 or Q4_0. Conversion uses
CPU PyTorch dependencies managed by uv and does not load weights onto a GPU.

## Disk-backed experts

Set `fish_drama3.disk_experts=true` to keep the full routed expert banks in the
GGUF file and read only selected experts into a bounded native-format cache.
`fish_drama3.expert_cache_size` sets resident slots per layer (8 to 256; default
32). Dense backbone, tower, and codec weights remain resident. The existing
`cpu_expert_layers` setting still selects CPU execution for those layers, but
their expert caches are bounded too. No GGUF conversion is required.

`fish_drama3.disk_expert_read_threads` selects 1 to 64 host workers for copying
selected experts from the memory-mapped GGUF into staging. The default is 1
(serial); try 4 to enable parallel copies using OpenMP or Apple dispatch. This applies only
with `disk_experts=true` and `pin_experts=false`. Backend uploads and inference
are unchanged. The host reader is backend-independent; Apple builds do not
require OpenMP for parallel reads.

This is a memory/speed trade-off, not a faster MoE kernel. With
`cpu_expert_layers=0`, prompt prefill groups tokens by expert within each layer,
so one upload serves multiple tokens. Attention and shared/router projections
retain the resident batched-prefill path; cached experts retain their vector
projection path. Session-owned graphs share one activation workspace and are
reused for the same exact prompt length while retained. A different length
replaces that graph set rather than reusing a rounded capacity for stateful
attention. Offline disk mode releases backbone and tower prefill graphs before
waveform decoding, without clearing continuation state. Later full prefills
rebuild those workspaces; resident execution keeps its existing cache policy.
Mixed CPU/backend expert execution retains token-wise prefill. Storage speed
still matters, and resident mode remains the default. For the minimum cache:

```text
--session-option fish_drama3.disk_experts=true
--session-option fish_drama3.expert_cache_size=8
--session-option fish_drama3.cpu_expert_layers=0
```

When sufficient host RAM is available, add
`fish_drama3.pin_experts=true` and set `fish_drama3.cpu_expert_layers=0`.
This loads full expert banks into pinned host memory once, then uploads cache
misses directly without an intermediate host copy. Expert matmuls still run on
the selected backend, and GPU cache size is unchanged. This option is off by
default and requires pinned host-buffer support; allocation failure is an error,
not a fallback to a different memory policy.

Logs report `disk_experts.bytes_read`, `disk_experts.expert_loads`,
`disk_experts.cache_hits`, `disk_experts.buffer_mb`, and
`disk_experts.host_buffer_mb` when the session is destroyed. These describe
expert I/O and cache allocation, not total process peak VRAM. With pinned
experts, bytes read include the initial host-bank load instead of per-request
cache-miss reads.

## Serve over HTTP

Save the following as `drama3-server.json` in the repository root:

```json
{
  "host": "127.0.0.1",
  "port": 8035,
  "backend": "cuda",
  "threads": 8,
  "lazy_load": true,
  "max_loaded_models": 1,
  "idle_unload_ms": 300000,
  "ui": true,
  "models": [{
    "id": "drama-3",
    "family": "fish_drama3",
    "path": "models/Drama-3-GGUF/drama-3-backbone-8bit-experts-Q2_K.gguf",
    "task": "tts",
    "mode": "streaming",
    "session_options": {
      "fish_drama3.cpu_expert_layers": 8,
      "fish_drama3.context_length": 2048
    },
    "default_request_options": {"max_tokens": 500}
  }]
}
```

```powershell
.\build\windows-cuda-release\bin\audiocpp_server.exe --config drama3-server.json
```

The server exposes `POST /v1/audio/speech` and the task APIs, including streaming
events and continuation artifacts. See the [server guide](../../app/server/README.md).

## Voice references

For cloning, add `--voice-ref reference.wav` and
`--request-option 'reference_text=What the reference says.'`. Use `--task clon`
to require a reference. Ordered references can also be supplied through
`multi_reference_cond` or `references` as a JSON string:

```json
[{"text":"<|speaker:0|> First reference.","audio_path":"first.wav"},
 {"text":"<|speaker:1|> Second reference.","audio_path":"second.wav"}]
```

Descriptors accept `vq_codes` instead of `audio_path`: a nonempty integer grid
with ten rows, one per codebook. Text accepts `<|speaker:0|>` through
`<|speaker:15|>` and event tags such as `[sighs]` and `[laughs]`.

## Continuation and editing

`previous_takes` is a JSON string containing ordered earlier takes. Each object
has `text` and either `vq_codes` or `audio_path`. Earlier text strings can also
be given; the runtime generates their codes before the requested take.
`previous_takes_file` accepts the same data from a JSON file, including a single
take object. Use it for VQ grids that exceed the Windows command-line limit.

```powershell
--request-option 'previous_takes=["The report is due on Wednesday."]'
--text 'The report is due on Thursday.'
```

The library returns an `AcousticTokens` artifact named `drama3_take`, with
`family=fish_drama3` and `schema=drama3_take_v1`. Its UTF-8 JSON payload contains
the text and VQ grid. Keep this artifact and pass it in the next request's
`input_artifacts` to reuse the original codes for an edit. Multiple artifacts
carry chained history. The current take's audio is returned separately.

Streaming sessions use `IStreamingVoiceTaskSession`. `start_stream()` runs
synchronously and calls the event sink while audio is generated; `finalize()`
returns the complete audio and the reusable take artifact. An application can
invoke generation on its worker thread to keep its UI responsive.

Long-form text uses the framework chunker, with `text_chunk_size=128` and
`text_chunk_mode=tag_aware` by default. Each new chunk includes the first generated
chunk as an anchor and the latest chunk as recent context, without duplicating
the first chunk. If that prompt does not fit, automatic history is reduced to the first chunk only, then removed.
References and caller-supplied history remain ordered and are never dropped.

`skip_empty_chunks=false` preserves the error when a chunk emits EOS before any
audio frame. Set `--request-option skip_empty_chunks=true` to skip such chunks
and continue in offline or streaming mode. Skipped chunks do not enter generated
voice history, matching upstream. If all chunks are empty, the request fails with
`Drama 3 produced no audio: all text chunks were empty`. Other errors are not
suppressed.

`max_tokens` is an upper bound: generation is capped to the rows remaining after
the selected prompt within `fish_drama3.context_length`. Reductions are logged,
and `reached_limit` reports reaching this effective budget. A required prompt
that leaves no generation room is still rejected. These rules apply to both
offline and streaming generation.

## Validation limits

### V1/V2 checkpoint comparison

A [community evaluation](https://github.com/0xShug0/audio.cpp/discussions/891#discussioncomment-19427812)
replayed the same 5,000 requests across English, Japanese, Chinese, French and
Spanish, with 1,000 requests per language. Seeds, reference voices, runtime and
generation settings stayed unchanged between checkpoints.

The comparison used Q4_K experts, FP32 routers and BF16 text embedding/full
output head, with all experts on GPU. The native runtime was pinned to
`4faef969850e2dd485e717fe2159875b91cfa3dc`. Settings were temperature `1.0`,
`top_p=0.9`, `top_k=4097`, `text_chunk_size=128`, context `8192` and a maximum of
`1024` acoustic frames per text chunk. These results describe this precision
profile, not every quantization or Drama3-small.

| Automatic check | V1 | V2 |
|---|---:|---:|
| Trailing quiet >=2 seconds | 25 | 0 |
| Internal quiet >=2 seconds | 35 | 0 |
| Clips reaching a chunk's frame cap before EOS | 9 | 14 |

The updated checkpoint improved silent-continuation behavior in this test.
However, repetitive speech and prolonged nonverbal reactions remain possible,
sometimes continuing until the generation limit. An ASR-assisted pass over the
V2 outputs flagged 29 potential speech-repetition cases. All five selected cases
checked by listening contained repetition. Review generated audio for looping
speech even when no long quiet sections are detected.

These results describe a fixed benchmark with 5,000 text strings representing
1,000 translated meaning families, not a general failure rate or a guarantee
that trailing silence is eliminated. The full V1 corpus was not evaluated with
the same repetition screen, so the repetition results do not establish a V1/V2
regression. ASR discrepancies are review flags, not a full assessment of spoken
content, voice similarity or acting quality.

### Port validation

Reproducible codec, tokenizer, quantization and HTTP smoke commands are in the
[Drama 3 test guide](../../tests/community_models/fish_drama3/README.md).

The port was exercised on native Windows with an RTX PRO 6000 Blackwell 96 GB,
CUDA 12.8 and eight CPU worker threads. English speech, voice references,
streaming audio and continuation/edit artifacts were checked. Streaming emits
44.1 kHz audio, with streamed chunks matching the complete output in the tests.

Independent checks cover hybrid backbone prefill and cached decode, actual Q2_K,
Q3_K and Q4_K expert bank types and layouts against dequantized PyTorch weights,
CPU/GPU expert splits, audio tower
prefill/cached decode, and exact prompt rows against the official builder.
Repeated seeded generation and offline/streaming generation produced identical
codes within each profile. English transcript checks recovered the requested
speech, including Wednesday-to-Thursday and noon-to-five edits.

This is an experimental port. Some emotion and nonverbal prompts produce prolonged
reactions, omit dialogue or reach the generation cap. This occurs with both the
original Q4-expert package and smaller formats. A heavier Q2 CPU offload trial
also reached the long-form cap; the documented eight-layer configuration completed
that case. These checks do not establish full perceptual parity with the original
checkpoint. Multilingual pronunciation, speaker similarity and event delivery
have not received a broad perceptual evaluation. A native
seed reproduces a take within the same execution configuration; changing
offload placement can change floating-point results and the sampled take.

The weights are subject to the Fish Audio Community License. See the model
repository's [LICENSE](https://huggingface.co/Jmica/Drama-3-GGUF/blob/main/LICENSE)
and [NOTICE](https://huggingface.co/Jmica/Drama-3-GGUF/blob/main/NOTICE).
