# Fish Audio Drama 3

`fish_drama3` is a community model port of
[fishaudio/drama-3](https://huggingface.co/fishaudio/drama-3). Inference runs in
native audio.cpp with GGML; Python is used to prepare the checkpoint.

The implementation includes the 40-layer hybrid attention/MoE backbone,
256 experts with eight selected per token, four shared enrichment layers,
nine residual codebook layers with persistent timeline caches, and the Fish DAC
encoder/decoder. It supports offline and streaming TTS, voice references,
speaker and event tags in text, long-form chunking, and continuation/edit history.

## Windows build

Install Visual Studio 2022 C++ build tools, a CUDA toolkit that supports your GPU,
and Git. Use the Drama 3 branch and initialize its public frontend submodule:

```powershell
git clone --branch drama-3-native --recurse-submodules https://github.com/JarodMica/audio.cpp.git
Set-Location audio.cpp
```

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
  Jmica/Drama-3-GGUF drama-3-experts-q2_k-backbone-q8_0.gguf `
  --local-dir models\Drama-3-GGUF
```

The model manager also lists all four packages:

```powershell
uv run --no-project --python 3.12 tools\model_manager_v2.py list | Select-String 'fish_drama3'
uv run --no-project --python 3.12 tools\model_manager_v2.py install fish_drama3_q2_experts_q8_backbone --models-dir models
```

| File | Size | Expert banks | Other backbone matrices |
|---|---:|---|---|
| `drama-3-experts-q4_k-backbone-q8_0.gguf` | 22.91 GiB | Q4_K | Q8_0 |
| `drama-3-experts-q3_k-backbone-q8_0.gguf` | 18.93 GiB | Q3_K | Q8_0 |
| `drama-3-experts-q2_k-backbone-q8_0.gguf` | 15.88 GiB | Q2_K | Q8_0 |
| `drama-3-experts-q4_k-backbone-q4_k.gguf` | 21.78 GiB | Q4_K | Q4_K, with Q4_0 text embedding |

All four packages retain FP16 audio tower and waveform codec weights and FP32
normalization vectors. They embed their tokenizer, configuration and model
contract. Stock audio.cpp releases without this port cannot run Drama 3.

## Prepare weights with uv

Conversion is optional when using a downloaded package. Build the native
converter if you want to prepare a different precision:

```powershell
.\scripts\build_windows.ps1 -Preset windows-cpu-release -ModelSet custom -Models fish_drama3 -Target audiocpp_gguf -Jobs 4
```

Download the complete official checkpoint into `models\drama-3-original`. Its
original per-expert FP8 tensors have 128-by-128 inverse scales. The preparation
script applies those scales and packs each projection into an expert bank.

```powershell
uv run --python 3.12 tools\community_models\convert_fish_drama3.py `
  --checkpoint models\drama-3-original `
  --output models\drama3-f16-local.gguf `
  --quantized-output models\drama-3-experts-q4_k-backbone-q8_0.gguf `
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
  --model models\Drama-3-GGUF\drama-3-experts-q2_k-backbone-q8_0.gguf `
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
Host memory is required for offloaded weights. Context capacity includes text,
reference audio frames, prior takes, and the requested generation budget.

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
    "path": "models/Drama-3-GGUF/drama-3-experts-q2_k-backbone-q8_0.gguf",
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
