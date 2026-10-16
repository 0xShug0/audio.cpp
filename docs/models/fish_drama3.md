# Fish Audio Drama 3

`fish_drama3` is a community model port of
[fishaudio/drama-3](https://huggingface.co/fishaudio/drama-3). Inference runs in
native audio.cpp with GGML; Python is used to prepare the checkpoint.

Built with Fish Audio.

The implementation includes the 40-layer hybrid attention/MoE backbone,
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

The model manager also lists all four packages:

```powershell
uv run --no-project --python 3.12 tools\model_manager_v2.py list | Select-String 'fish_drama3'
uv run --no-project --python 3.12 tools\model_manager_v2.py install fish_drama3_q2_experts_q8_backbone --models-root models
```

| File | Size | Expert banks | Other backbone matrices |
|---|---:|---|---|
| `drama-3-backbone-8bit-experts-Q4_K_S.gguf` | 22.91 GiB | Q4_K | Q8_0 |
| `drama-3-backbone-8bit-experts-Q3_K_S.gguf` | 18.93 GiB | Q3_K | Q8_0 |
| `drama-3-backbone-8bit-experts-Q2_K.gguf` | 15.88 GiB | Q2_K | Q8_0 |
| `drama-3-backbone-4bit-experts-Q4_K_S.gguf` | 21.78 GiB | Q4_K | Q4_K, with Q4_0 text embedding |

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

All four packages retain FP16 audio-tower matrices and waveform codec weights.
Backbone and audio-tower normalization vectors stay FP32; codec normalization
vectors use FP16 with the rest of the codec. The packages embed their tokenizer,
configuration and model contract. Stock audio.cpp releases without this port
cannot run Drama 3.

## Prepare weights with uv

Conversion is optional when using a downloaded package. Build the native
converter if you want to prepare a different precision:

```powershell
.\scripts\build_windows.ps1 -Preset windows-cpu-release -ModelSet custom -Models fish_drama3 -Target audiocpp_gguf -Jobs 4
```

Download the complete official checkpoint into `models\drama-3-original`.
The revision below is the checkpoint used for validation:

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

## Generate and stream

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

Host memory is required for offloaded weights. Context capacity includes text,
reference audio frames, prior takes, and the requested generation budget.

### Disk-backed experts

Set `fish_drama3.disk_experts=true` to keep the full routed expert banks in the
GGUF file and read only selected experts into a bounded native-format cache.
`fish_drama3.expert_cache_size` sets resident slots per layer (8 to 256; default
32). Dense backbone, tower, and codec weights remain resident. The existing
`cpu_expert_layers` setting still selects CPU execution for those layers, but
their expert caches are bounded too. No GGUF conversion is required.

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

Long-form text uses the framework chunker, with `text_chunk_size=200` and
`text_chunk_mode=tag_aware` by default. Each new chunk includes the previous
chunk as context. Caller-supplied history remains ordered in every chunk.

## Validation limits

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
