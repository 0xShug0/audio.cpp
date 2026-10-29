# AudioCPP Command Usage

Use `audiocpp_cli` for direct model inference.

```bash
audiocpp_cli --task <task> --family <family> --model <model-dir> --backend <backend> [inputs] [outputs]
```

> **Always specify `--family` for model commands, including model-specific help.** Treat it as required; omission is supported only for backward compatibility. Automatic loader discovery without it can substantially slow startup and `--model ... --help`.

## Common Options

| Option | Values | Default | Meaning |
|---|---|---:|---|
| `--task` | `gen`, `tts`, `clon`, `vc`, `svc`, `s2s`, `asr`, `align`, `vad`, `diar`, `sep`, `vdes`, `midi`, `wake`, `cls` | required | User task. |
| `--family` | model family name | required | Selects the model implementation. Must match a registered loader (`audiocpp_cli --list-loaders`). |
| `--model` | local model directory | required | Path to local model assets. |
| `--backend` | `cpu`, `cuda`, `vulkan`, `metal`, `best` | `cpu` | Inference backend. |
| `--mode` | `offline`, `streaming` | `offline` | Run mode. Most models are offline. |
| `--device` | integer | `0` | Backend device index. |
| `--list-devices` | flag | off | List available backend devices and exit; combine with `--backend`/`--device` to select one. Add `--json` for [machine-readable output](#listing-devices) with PCI ids and memory. |
| `--threads` | integer | `4` | Backend/OpenMP worker threads. |
| `--log` | flag | off | Print progress and timing logs to stdout. |
| `--log-file` | path | not set | Stream progress and timing logs to a file. |
| `--metrics` | flag | off | Print compact offline wall time, audio duration, RTF, realtime speed, sample rate, and channel metrics. |
| `--resample-kernel-cache-mb` | integer MiB | `32` | Memory for cached resampling kernels, evicting the least recently used first; `0` disables the cache. A kernel larger than the limit, such as the 84.5 MiB one for 44056 Hz to 16 kHz, is rebuilt on every call. |

## Common Inputs And Outputs

| Option | Used by | Meaning |
|---|---|---|
| `--text` | generation, TTS, ASR context, alignment transcript | Input text. |
| `--audio` | generation/editing, ASR, classification, wake word, VAD, diarization, separation, conversion, alignment | Input WAV, or `-` to stream raw PCM from stdin (requires `--mode streaming`). |
| `--input-format` | streaming ASR with `--audio -` | Raw PCM sample format, `s16le` or `f32le`. Default `s16le`. |
| `--input-rate` | streaming ASR with `--audio -` | Raw PCM sample rate in Hz. Default `16000`. |
| `--input-channels` | streaming ASR with `--audio -` | Raw PCM channel count. Default `1`. |
| `--voice-ref` | voice clone / voice design / some VC paths | Reference voice WAV. |
| `--language` | language-aware models | Language code. |
| `--out` | single-primary-output models | Output file path, such as WAV for audio tasks or MIDI/JSON for MuScriptor. |
| `--out-dir` | multi-output or batch models | Output directory. |
| `--out-format` | audio outputs written by `--out` / `--out-dir` | WAV sample format: `pcm16` (default), `pcm24`, or `float32`. `float32` keeps samples above full scale instead of clipping them. |
| `--segments-out` | VAD | Speech segments JSON. |
| `--vad-chunks-out` | offline VAD | VAD-based audio chunk windows JSON. |
| `--turns-out` | diarization | Speaker turns JSON. |
| `--words-out` | ASR/alignment | Word timestamps JSON. Sets `return_timestamps`. For `kokoro_tts` the entries are phoneme groups, not written words — read [its page](models/kokoro_tts.md#phoneme-group-timings) before joining them to text. |
| `--audio-chunk-seconds` | ASR | Split long audio before model inference, where supported. |
| `--audio-chunk-mode` | ASR/alignment | `auto`, `fixed`, `vad`, or `none`, where supported. |

## Common Generation Options

Omit these unless you need explicit control. Seed defaults are model-specific:
some models choose a random seed when omitted, while others use a fixed default
(for example, AudioSR uses 42 and MiDashengLM uses 0). Neither `0` nor `-1` is a
universal random-seed sentinel; check the model's usage page.

| Option | Values | Meaning |
|---|---|---|
| `--seed` | integer | Reproducible random seed. |
| `--max-tokens` | integer | Maximum generated tokens for AR/LLM-style models. |
| `--max-steps` | integer | Maximum diffusion or generation steps for models that expose it. |
| `--temperature` | float | Sampling temperature. |
| `--top-k` | integer | Top-k sampling limit. |
| `--top-p` | float in `(0, 1]` | Nucleus sampling limit. |
| `--repetition-penalty` | float | Penalize repeated tokens. |
| `--do-sample` | `true`, `false` | Enable sampling instead of greedy decode. |
| `--guidance-scale` | float | Classifier-free guidance scale. |
| `--num-inference-steps` | integer | Diffusion/flow denoising steps. |
| `--text-chunk-size` | integer chars | Split long text where supported, including TTS text. |
| `--text-chunk-mode` | `default`, `tag_aware`, `japanese`, `endline` | Select text chunking mode where supported. |

## Batch Inputs

| Option | Meaning |
|---|---|
| `--request-sequence <json>` | Run multiple JSON requests through one offline model session. |
| `--batch-text-file <txt>` | One request per non-empty text line. |
| `--batch-text-dir <dir>` | One request per `.txt`, `.md`, or `.json` file; each file is normalized into a single paragraph. |
| `--batch-audio-dir <dir>` | One request per `.wav` file. |
| `--batch-audio-role audio\|voice_ref\|source_audio\|target_voice\|prosody_ref\|style_ref` | How to use each batch WAV. |
| `--batch-merge-audio none\|concat` | Keep outputs separate or concatenate generated audio. |
| `--batch-manifest-out <json>` | Write a batch output manifest. |

`--batch-text-dir` reads `.txt` and `.md` files as plain text. For `.json`, use either a JSON string root or an object with a string `input` or `text` field.

Use `--request-sequence` when you want to send multiple requests in one long-lived offline session:

```bash
audiocpp_cli --task tts --family pocket_tts \
  --model models/PocketTTS-GGUF/english/pocket-tts-english-q8_0.gguf \
  --backend cuda \
  --request-sequence requests.json \
  --out-dir outputs \
  --metrics
```

The JSON may be either an array or an object with a `requests` array. Each item is parsed like one CLI request. Use `id` as the request name; with `--out-dir`, primary audio is written as `<out-dir>/<id>.wav`. There is no per-request `out` field.

```json
{
  "requests": [
    {
      "id": "speaker1_output",
      "text": "First text",
      "voice_ref": "voices/speaker1.wav",
      "reference_text": "Reference transcript 1",
      "seed": 1234
    },
    {
      "id": "speaker2_output",
      "text": "Second text",
      "voice_ref": "voices/speaker2.wav",
      "reference_text": "Reference transcript 2",
      "seed": 1234
    }
  ]
}
```

For each request id, `--metrics` prints `metrics[<id>].wall_ms`, `audio_duration_ms`, `rtf`, `x_realtime`, `sample_rate`, and `channels`.

A request can also pass input artifacts in an `artifacts` array, in the shape the server's [`/v1/tasks/run`](../app/server/README.md#post-v1tasksrun) takes: `id`, `kind`, `payload` (Base64) or `path` (a file holding the bytes, relative to the JSON file), and optional `meta`. Workflow requests take the same field, with paths relative to the workflow file.

## Listing Devices

`--list-devices` prints every backend device with the `--backend`/`--device` pair that selects it. Add `--json` to print one JSON document on stdout instead, with PCI ids and memory. `audiocpp_server --list-devices --json` prints the same document. ggml and driver start-up messages go to stderr. Builds from before `--json` was added ignore it and print the text listing, so a caller can check that stdout starts with `{`.

```bash
audiocpp_cli --list-devices --json | python3 -m json.tool
```

Abridged output from a CUDA and Vulkan build; the real document is one line:

```json
{
  "schema_version": 1,
  "devices": [
    {
      "backend": "cuda",
      "registry": "CUDA",
      "index": 0,
      "name": "CUDA0",
      "description": "NVIDIA GeForce RTX 5090",
      "type": "gpu",
      "pci_bus_id": "0000:01:00.0",
      "memory": {
        "free_bytes": 29778771968,
        "total_bytes": 33743044608
      },
      "caps": {
        "async": true,
        "host_buffer": true,
        "buffer_from_host_ptr": false,
        "events": true
      }
    },
    {
      "backend": "cpu",
      "registry": "CPU",
      "index": 0,
      "name": "CPU",
      "description": "AMD Ryzen 9 9950X3D 16-Core Processor",
      "type": "cpu",
      "pci_bus_id": null,
      "memory": {
        "free_bytes": 17498980352,
        "total_bytes": 32635379712
      },
      "caps": {
        "async": false,
        "host_buffer": false,
        "buffer_from_host_ptr": true,
        "events": false
      }
    }
  ]
}
```

| Field | Meaning |
|---|---|
| `schema_version` | `1`. Later versions may add fields; a change that breaks readers bumps it. |
| `backend` | The `--backend` value that selects the device: `cpu`, `cuda`, `hip`, `vulkan` or `metal`. `null` when no `--backend` value selects it, as for BLAS. |
| `registry` | ggml registry name, the prefix in the text listing: `CUDA` (`MUSA` on MUSA builds), `ROCm`, `Vulkan`, `MTL`, `BLAS` or `CPU`. |
| `index` | The `--device` value, and the server's `device`. It counts within the registry, after `CUDA_VISIBLE_DEVICES`, `HIP_VISIBLE_DEVICES` or `GGML_VK_VISIBLE_DEVICES` filtering. `--backend cpu` ignores `--device`. |
| `name` | ggml device name, such as `CUDA0` or `Vulkan1`. |
| `description` | Device name from the driver, such as `NVIDIA GeForce RTX 5090`. |
| `type` | `cpu`, `gpu`, `igpu`, `accel` or `meta`. |
| `pci_bus_id` | Lower-case `domain:bus:device.function`, such as `0000:01:00.0`. `nvidia-smi` prints the same id with an 8-digit domain (`00000000:01:00.0`). A GPU that both CUDA and Vulkan list has the same id in both. `null` for Metal, CPU and BLAS, and on Vulkan drivers without `VK_EXT_pci_bus_info`. |
| `memory` | `free_bytes` and `total_bytes` as exact integers. `null` when the device reports no total, as BLAS does, or a free figure above its total. |
| `caps` | ggml device capabilities: `async`, `host_buffer`, `buffer_from_host_ptr` and `events`. |

A value the backend does not report is `null`. ggml itself falls back to an estimate in two places, both in the table below: for the CPU, where the listing reports the host's available memory instead, and for Vulkan without `VK_EXT_memory_budget`, where free equals total.

`free_bytes` is a snapshot taken while the listing runs. Except on Metal, it covers the whole device, including memory other processes hold. Backends measure it differently:

| Backend | `total_bytes` | `free_bytes` |
|---|---|---|
| CUDA, HIP | `cudaMemGetInfo` / `hipMemGetInfo`, which leaves out memory the driver reserves: on NVIDIA, `nvidia-smi`'s `memory.total` minus `memory.reserved` | Free memory from the same call. The listing reads it after creating its own context on the device, so the figure already excludes one context. On Linux, integrated GPUs, and every device while `GGML_CUDA_ENABLE_UNIFIED_MEMORY` is set (to any value, `0` included), report `MemAvailable` from `/proc/meminfo` instead, against the device's own total. |
| Vulkan | Device-local heaps (on NVIDIA, `nvidia-smi`'s `memory.total`); every heap on an integrated GPU | Heap budget minus heap usage, from `VK_EXT_memory_budget`. Without that extension, ggml reports the heap size, so free equals total whatever is in use. |
| Metal | `recommendedMaxWorkingSetSize` | That minus `currentAllocatedSize`, which counts only this process's allocations. |
| CPU | Physical RAM | Memory the host can still allocate: `MemAvailable` on Linux, available physical memory on Windows, and free + inactive + purgeable pages on macOS. The server's `--min-free-memory-mb` guard checks the same figure. |

`--json` reads each CUDA or HIP device's PCI id and memory through ggml's device properties, which query memory and so create the device's primary context; the plain listing creates none. On an RTX 5090 with CUDA 13.4, the context held 498 MiB of VRAM until the command exited and added about 0.1 s. On Vulkan, `--json` added no measurable time or VRAM.

If a CUDA or HIP device cannot give the process a context, for example a GPU in exclusive-process mode that another process holds, or one too full for a new context, ggml stops the process: `--json` exits with a non-zero status, prints nothing on stdout and reports the CUDA error on stderr. The plain `--list-devices` still works in that state.

## Model Docs

| Need | Doc |
|---|---|
| Speech, voice clone, long-form TTS | [tts.md](tts.md) |
| Music and sound generation | [music_generation.md](music_generation.md) |
| Video lip synchronization | [LatentSync](models/latentsync.md) |
| OmniVoice TTS, voice cloning, voice design, and streaming | [models/omnivoice.md](models/omnivoice.md) |
| ASR models | [asr.md](asr.md) |
| Classification, wake word, VAD, and diarization | [speech_analysis.md](speech_analysis.md) |
| Audio tools, voice conversion, codec, and source separation | [audio_tools.md](audio_tools.md) |
