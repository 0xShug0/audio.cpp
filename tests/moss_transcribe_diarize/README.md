# MOSS-Transcribe-Diarize Validation

The native family uses the framework Whisper component and Qwen AR runtime.
Validation is CUDA-only; CPU, Vulkan, and Metal performance is not claimed.

## Regenerated Packages and Quantization

Original checkpoints and conversion intermediates live outside the local
`audio.cpp-gguf` repository. All variants are generated from the pinned original
safetensors, with the lossless Whisper name mapping, not by requantizing GGUFs.
The converter accepts `--type orig|q8_0|q4_k|q4_0`; original mode retains its
byte-exact checks. Quantized mode checks shapes, retained BF16 bytes, finite
dequantized values and reports quantization error, not a claim of output parity.

Before regeneration, run `tools/check_loader_catalog_sync.py --self-test`,
`tools/check_loader_catalog_sync.py`, and the typed spec checks. Sync alone checks
catalog drift, not schema correctness. Supported specs require package entries.

Sequential RTX 5090 CUDA Debug comparison, eight threads, default 5120-token
budget, one cold plus three warm requests per recording and precision:

| Precision | 28 s warm median / RTF | 327.6 s warm median / RTF |
| --- | ---: | ---: |
| BF16 | 626.398 ms / 0.02237 | 8845.980 ms / 0.02700 |
| Q8_0 | 587.946 ms / 0.02100 | 8402.070 ms / 0.02565 |
| Q4_K | 589.642 ms / 0.02106 | 8318.250 ms / 0.02539 |
| Q4_0 | 896.753 ms / 0.03203 | 8215.950 ms / 0.02508 |

All repeated transcripts are identical within each precision. Timings exclude
load/upload and are end-to-end, not an equal-generated-token kernel comparison.
The comparison used the corrected spec override for every candidate. All three
final packages subsequently passed standalone server loading from `/tmp`, with
no spec override, and short/long inference using the embedded `instruct` spec.

Q8_0 passes the unchanged Python FP32 gate on both primary fixtures: zero word
or speaker errors, no segment reflow, at most the pre-existing 30 ms timestamp
drift. Q4_K passes the short gate but fails the long gate: words/speakers match,
but segment boundaries change and matched timestamps differ by up to 680 ms.
Q4_0 fails both gates: the short result has 14 segments instead of six, and the
long result changes `phrase` to `frays` (WER 0.0011013) with segment reflow.
No thresholds were relaxed. Q4_K is the only retained Q4 variant, based on its
better short-input latency and word preservation; its long latency is about
1.2% slower than Q4_0. Q4_K must not be described as parity-safe.

Evidence: `quant_*_{short,long}_{0,1,2,3}.json`, `quant_*_parity.json`,
`quant_comparison_summary.json`, and `server_quant_compare.log` under
`build/logs/moss_transcribe_diarize/`. Quantized tensors use 342 matrices;
339 tensors remain BF16 and two lookup tables become F16. Both Q4 candidates
have the same file size. The original-mode tensor checks remain byte-exact.

### Final Package Regression Checks

The final BF16, Q8_0 and Q4_K conversions ran once each from the original
safetensors after the `instruct` rename. No further regeneration was needed.
All six final server results exactly match their corresponding precision's
saved pre-regeneration text, segments, speaker turns and sample rate.
BF16 and Q8_0 pass both Python gates; Q4_K retains the documented long-input
failure above. Regeneration parity does not imply quantization parity.

The standalone BF16 CLI accepts `--instruct` in streaming mode. Its short output
passes the Python gate. The long server SSE output passes the same gate and
the delta/final-text and completion protocol checks. This is text-output
streaming after complete input processing, not streaming audio input.

Evidence under `build/logs/moss_transcribe_diarize/`:
`instruct-*-conversion.{log,stdout}`, `final_*_load.json`,
`final_*_{short,long}.json`, `final_regeneration_identity.jsonl`,
`final_instruct_cli.{log,stdout}`, `final_bf16_long.sse`,
`final_{bf16,q8_0,q4_k,stream}_parity.{log,json}` and
`server_final_packages.log`. The temporary validation server was stopped.

Pending Apollo and UniverSR regression checks also completed: Apollo music
and held-out Jingle Bells, and UniverSR music, speech and held-out Jingle Bells
are byte-identical to their saved native WAVs. Apollo music and all three
UniverSR cases pass their Python waveform gates. Apollo's held-out check uses
the saved native reference; its previously documented strict Python drift is
unchanged, not claimed as a Python pass. See `regenerated-final-*` logs/WAVs
under `build/logs/{apollo-family,universr-family}/`.

## Pinned Inputs

- Weights: `OpenMOSS-Team/MOSS-Transcribe-Diarize`, revision
  `704aa4a9c304e8520be88901e0d1960158ef5b15`.
- Reference code: `OpenMOSS/MOSS-Transcribe-Diarize`, revision
  `61bc29cd4120be7b5d3b761b64cd5dff57263642`.
- Original checkpoint: 683 tensors, 908,513,280 BF16 parameters, Apache-2.0.
- Local upstream directories: `/media/leo/Share/models/MOSS-Transcribe-Diarize`
  and `/media/leo/Share/models/MOSS-Transcribe-Diarize-upstream`.
- Logs: `build/logs/moss_transcribe_diarize/`.

## Reference

Run in `conda qwen3-tts`, sequentially. The runner uses upstream
`generate_transcription`, SDPA, greedy decoding, eight CPU threads, and disables
TF32. It includes audio loading/preprocessing and decoding in the measured time,
but excludes model loading. Trial zero is cold; the median uses three warm trials.
It rejects truncated generations and inconsistent repeated transcripts.

```bash
conda run --no-capture-output -n qwen3-tts python tests/moss_transcribe_diarize/reference.py \
  --model /media/leo/Share/models/MOSS-Transcribe-Diarize \
  --upstream /media/leo/Share/models/MOSS-Transcribe-Diarize-upstream \
  --audio assets/resources/four_speaker_short.wav assets/resources/qwen3_tts_longform_asr_input.wav \
  --max-tokens 4096 \
  --output build/logs/moss_transcribe_diarize/python_bf16.json \
  --log build/logs/moss_transcribe_diarize/python_bf16.log
```

RTX 5090, PyTorch 2.11.0+cu128, Transformers 5.12.1:

| Input | Seconds | Warm inference | RTF | Peak allocated | Peak reserved |
| --- | ---: | ---: | ---: | ---: | ---: |
| Four speakers | 28.0 | 1.960 s | 0.0700 | 1,918,658,560 B | 2,000,683,008 B |
| Long narration | 327.6 | 25.466 s | 0.0777 | 2,638,412,800 B | 2,921,332,736 B |

Weights occupy 1,817,026,560 bytes. PyTorch's peak allocated memory is 1.06x
and 1.45x that size respectively. These are allocator statistics, not total
process VRAM including CUDA context/library overhead. Native validation must
also measure process peak VRAM and account for weights, KV cache, and graph
workspace. Native memory use must not be justified solely by fitting the GPU.

## GGUF

`convert_gguf.py` calls the existing native converter with `--type orig`, embeds
the model spec and tokenizer/configuration sidecars, and verifies all tensor
shapes, BF16 types, and bytes. Only Whisper names change, to the existing OpenAI
Whisper loader layout. No weights are quantized, folded, or numerically changed.

```bash
conda run --no-capture-output -n qwen3-tts python tests/moss_transcribe_diarize/convert_gguf.py \
  --model /media/leo/Share/models/MOSS-Transcribe-Diarize \
  --output /media/leo/Share/models/audio.cpp-gguf/MOSS-Transcribe-Diarize-GGUF/moss-transcribe-diarize-bf16.gguf \
  --log build/logs/moss_transcribe_diarize/conversion.log
```

## Native Implementation Boundary

Reuse `WhisperFrontendComponent::load_openai_layout` with explicit 16-head,
24-layer configuration; `WhisperLogMelExtractor`; framework linear, activation,
normalization, and embedding modules; `LlamaBpeTokenizer` with Qwen2
pretokenization; and `QwenCausalDecodeRuntime` with managed KV cache. Model code
owns audio chunk assembly, four-frame merging, prompt time anchors, the adaptor,
and output parsing. Large/frequent graphs must be session-owned and reused.

Remaining framework candidates (not implemented):

- HF Whisper loading currently hardcodes 12 heads and accepts no explicit
  encoder config. An explicit config/prefix binding API would remove the need
  for conversion-time Whisper renaming.
- Whisper's reusable component exposes host-vector output, while the Qwen
  runtime accepts host-vector prefill embeddings. A device-resident handoff
  could remove encoder/adaptor/decoder transfers. Measure before proposing it.
- The original Qwen prefill API exports KV to host before decode. MOSS now
  uses the additive device-cache prefill API below, avoiding that transfer.
- A config-driven HF Qwen3 weight-binding API could replace the model-local
  layer binding loop. It currently uses framework norm/linear binding helpers;
  no public whole-Qwen3 binding helper was available in this checkout.

## Native Results

Original BF16 GGUF weights; F32 graph activations and F16 managed decode KV
storage. There is no quantized-weight path in this validation.

The checkpoint's processor config sets time markers every **five seconds**,
overriding the Python class default of two. Its integer marker stride is 62 audio
embeddings, not 62.5. Both details must be preserved. Native/Python prompt lengths
are 445 for the short fixture and 4,356 for the long fixture.

Final-output comparison against Python FP32 (`parity_f32.json`):

- Four speakers: exact raw transcript, including timestamps and speaker labels.
- Long narration: all words and all 77 segment texts/speakers match. One shared
  segment boundary differs by 30 ms (146.59 s versus 146.56 s); all others match.
- Gate: zero word errors, no segment reflow, no speaker attribution changes,
  at most 50 ms timestamp difference. The tolerance is not a claim of bit parity.
- Held-out 14.07-second `sample.wav`: identical words, segment texts, and speaker
  labels to Python FP32, with at most 30 ms timestamp drift. Held-out LibriSpeech
  `test_other_7902-96591-0001.wav`: exact raw transcript. See
  `parity_heldout_f32.json`; the same zero-word-error/50-ms gate passes both.

Against upstream BF16, the short fixture has identical words/speakers and up to
40 ms timestamp differences. The long fixture has one homophone substitution
(`frays` versus `phrase`) and a few segment reflows. Python FP32 makes the same
word and segmentation choices as native, establishing their precision
sensitivity rather than treating BF16 output as identical.

Warm server results (one cold plus three sequential warm requests, same 4,096
token limit, eight threads; model loading excluded):

| Input | Native median | Native RTF | Python BF16 median | Speedup |
| --- | ---: | ---: | ---: | ---: |
| Four speakers, 28 s | 0.96931 s | 0.03462 | 1.95993 s | 2.02x |
| Narration, 327.6 s | 9.69783 s | 0.02960 | 25.46615 s | 2.63x |

Every repeated native output was identical. The framework log shows one graph
build per input shape, reused by subsequent requests. Python timing includes
file loading/preprocessing; native server timing starts with decoded waveform
input and includes resampling/frontend/inference. HTTP upload time is excluded
from this table. Curl's default Expect/Continue wait must not be counted as
inference; use `-H 'Expect:'` for HTTP wall-time comparisons.

## Memory

`server_vram.csv` sampled NVML process memory every 50 ms for PID 237546. The
test server was stopped afterwards; unrelated running servers were untouched.

- BF16 weight payload: 1.69 GiB.
- Native short-case process peak: 3,242 MiB (3.17 GiB).
- Native long-case process peak: 5,144 MiB (5.02 GiB), stable over repeats.
- Returning to the short shape reduced residency to 3,718 MiB.

The memory overhead is explainable but not optimal: the 28-layer, eight-KV-head,
128-head-dimension decoder stores a separate F32 prefill state and F16 decode
cache. For the long case, the 4,356-token prefill KV is about 0.93 GiB and the
8,452-slot decode KV about 0.90 GiB, in addition to weights, encoder/decoder
workspaces, CUDA graphs, and driver/library allocations. This is reasonable for
5.5 minutes of retained context, not a minimal-memory implementation. It is
roughly 3x the weight payload, and higher than Python's 2.46 GiB allocator peak;
NVML process usage and PyTorch allocator-only usage are different measurements.

The detailed server endpoint `/v1/audio/transcriptions/details` was verified to
return six segments, six speaker turns, and the original input sample rate for
the four-speaker fixture. Those initial baseline results used no framework or
GGML changes; the authorized opt-in extensions are documented below.

The final standalone GGUF was run from `/tmp`, without a model-spec override,
on both primary fixtures. `standalone_short.txt` and `standalone_long.txt` are
byte-identical to the native outputs used for the parity reports. The final
conversion log verifies all 683 tensors again; package size is 1,833,016,992 bytes.
The packaged default of 5,120 generated tokens was also tested without a request
override; the short output remained byte-identical (`default_options.txt`).

## Output Streaming Follow-Up

The model implements the framework pull-event streaming interface. Complete
audio is still required before generation, matching the upstream generation
order. Text deltas end at ASCII transcript delimiters to avoid splitting UTF-8
characters. Both modes share one decoder path and retain session-owned graphs.

CUDA CLI streaming runs with eight threads, max_tokens=4096, and the local spec
override produced `stream_short` and `stream_long` logs/stdout/text under the
same log directory. Both final text files compare byte-identically to the
previous offline fixtures. Partial text events appear before the final result.
The GGUF was subsequently repackaged with the streaming-capable spec.
`conversion_streaming.log` verifies all 683 BF16 tensors byte-for-byte against
the original checkpoint. `stream_standalone` was run from `/tmp` without a
spec override; its final text is byte-identical to the short offline baseline.

The pinned Transformers runner uses full-context Qwen generation; its
ProgressStreamer counts output tokens, not incoming audio chunks. The original
native path retained a square prefill mask and separate prefill/decode KV.
The default chunked implementation below replaces that execution path,
without independent audio windows or sliding-window truncation.

The upstream-recommended SGLang-Omni serving implementation documents
[4096-token chunked prefill and incremental output](https://github.com/sgl-project/sglang-omni/blob/main/docs/cookbook/moss_transcribe_diarize.md).
This is an optimization reference distinct from the pinned Transformers runner.
Chunked prefill limits the query dimension of temporary attention work; it does
not eliminate full-context KV storage. Strict input-independent VRAM still
requires a fixed-capacity policy or KV offloading, not audio-window truncation.

## GPU-Resident Chunked Prefill

Chunked prefill is the only MOSS path; there is no memory-saver option.
The rejected CPU KV-offload runtime has been removed. MOSS uses explicit
`prefill_embeddings_into_cache`, `build_with_static_cache_block`, and
`FastKVSetRowsModule::build_block` APIs. The original row writer is unchanged;
the decoder shares its math implementation between the two entry points.

Prefill writes 128-token blocks directly into the same FP16 backend cache
used by token decoding. A session-owned graph and allocator are reused for
each block and subsequent requests with the same capacity and block size.
The final partial block writes padding only into unused future slots, which
remain masked during decode until overwritten. No history is exported to
CPU, no separate full-prompt KV is retained, and no square prompt mask is
allocated. Full-context KV and the block mask still grow with capacity;
this does not promise constant VRAM or unlimited context.

The additive `prefill_embeddings_into_cache` API is opt-in. Existing runtime
entry points retain their behavior. The special import optimization and its
configuration flag have been removed. The existing import implementation is
unchanged; the new path calls the additive `clear_on_backend` cache API.
GGML source, weights, tokenizer rules, and sampling math are unchanged.

The earlier A/B validation below used RTX 5090, Debug, eight threads, original BF16 weights,
sequential requests, and three warm requests after one cold request. The
multipart `max_tokens=4096` field in these HTTP commands is not consumed:
the effective model default is 5120, confirmed by cache capacities 5565
(short) and 9476 (long). Both sides use those same effective settings.

| Audio | Non-chunked median | Chunked median | Chunked RTF | Non-chunked peak | Chunked peak |
|---|---:|---:|---:|---:|---:|
| 28 s | 660.857 ms | 617.826 ms | 0.0221 | 3390 MiB | 3234 MiB |
| 327.6 s | 9405.42 ms | 8846.91 ms | 0.0270 | 5258 MiB | 3664 MiB |

Elapsed-time reductions are 6.5% and 5.9%; long-input peak VRAM falls 30.3%.
Peaks are per-process NVML measurements, not post-request residency.

Evidence under `build/logs/moss_transcribe_diarize/`:

- `server_device_{short,long}_{0,1,2,3}.json`: all eight final transcripts
  are byte-identical to the saved native non-chunked baselines.
- `parity_device.json`: zero word or speaker errors and no segment reflow
  against Python FP32; only the pre-existing 30 ms long-recording boundary
  difference remains.
- `server_device_prefill.log` and `server_device_prefill_vram.csv`
  (PID 288882): peak process VRAM sampled every 50 ms.
- `server_device_control_{short,long}_{0,1,2,3}.json`,
  `server_device_control.log`, and `server_device_control_vram.csv`
  (PID 289555): fresh non-chunked control in the same build and environment.
- `chunked_unit_{cuda,cpu}.log`: partial blocks, repeated requests, decoding,
  cache growth, capacity and reset tests. Legacy and backend-zero import produce
  identical logits and F32/F16/BF16 cache bytes, including unused tails.
- `conversion_device_prefill.log`: all 683 BF16 tensors remain byte-exact
  against the pinned safetensors after refreshing the embedded spec.
  That package was 1,833,017,184 bytes, before removing the mode option.
- `device_stream_{short,long}.sse`, `server_device_stream.log`, and
  `parity_device_stream.json`: both fixtures pass the SSE protocol and Python
  output gates. Final transcripts equal offline byte-for-byte. The server
  ran from `/tmp` without a spec override, using the refreshed GGUF.

### Default Path Validation

After removing the mode option and special import optimization, the refreshed
GGUF was loaded from `/tmp` with no spec override or session options.
RTX 5090, Debug, eight threads, default 5120 output-token budget; sequential
requests with one cold request followed by three warm requests:

| Audio | Warm median | RTF | Peak process VRAM |
|---|---:|---:|---:|
| 28 s | 603.660 ms | 0.0216 | 3234 MiB |
| 327.6 s | 8560.940 ms | 0.0261 | 3664 MiB |

All eight transcripts match the previous C++ results byte-for-byte.
`parity_default.json` passes both Python FP32 gates with zero word/speaker
errors and only the existing 30 ms long-recording boundary difference.
These are current default-path timings, not a fresh non-chunked A/B run.

- `server_default_{short,long}_{0,1,2,3}.json`,
  `server_default_chunked.log`, and `server_default_chunked_vram.csv`
  (PID 307440) contain timing and 50 ms NVML sampling evidence.
- `additive_unit_{cpu,cuda}.log` covers block boundaries, repeated prefill,
  cache growth, decode parity, and F32/F16/BF16 cache clearing.
- `conversion_default_chunked.log` verifies all 683 BF16 tensors byte-for-byte
  after removing the option from the embedded spec. GGUF size: 1,833,016,992 bytes.
- `default_stream_{short,long}.sse` and `parity_default_stream.json` pass
  both streaming protocol and Python output gates without session options.
- The original `FastKVSetRowsModule::build` implementation and cache-import
  implementation were compared with HEAD and are unchanged. Decoder math is
  shared by explicit single-token and block entry points, without duplication.

Historical `server_saver*` / `saver_*` evidence describes the removed CPU
offload experiment, not the current implementation.

```bash
conda run --no-capture-output -n qwen3-tts build/debug/bin/qwen_chunked_prefill_test \
  --backend cuda --log build/logs/moss_transcribe_diarize/chunked_unit_cuda.log
```

Recheck the final-output gate against the saved FP32 reference:

```bash
conda run --no-capture-output -n qwen3-tts python tests/moss_transcribe_diarize/test_parity.py \
  --upstream /media/leo/Share/models/MOSS-Transcribe-Diarize-upstream \
  --reference build/logs/moss_transcribe_diarize/python_f32.json \
  --actual four_speaker_short=build/logs/moss_transcribe_diarize/standalone_short.stdout \
    qwen3_tts_longform_asr_input=build/logs/moss_transcribe_diarize/standalone_long.stdout \
  --max-timestamp-drift 0.05 \
  --log build/logs/moss_transcribe_diarize/parity_f32.log \
  --report build/logs/moss_transcribe_diarize/parity_f32.json
```
