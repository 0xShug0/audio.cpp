# MiraTTS-local sparse output head

Validated on Windows, 2026-09-06, using the RTX 3090 for CUDA and Vulkan
(Vulkan device 1).

MiraTTS now creates its CPU sparse output-head view when loading its own
weights. The view shares the embedding storage without copying or converting
weights. Its metadata context lives with the model weights and outlives all
prefill/decode graphs. MiraTTS rebases only the compact logits readback indices;
prompt and generated token IDs still use the full embedding vocabulary.

The shared `qwen_causal_decode_runtime.h` and `.cpp` are restored to upstream
`origin/main` at `a8fccb4`, with no sparse-head field, slicing, or index rebasing.
GPU backends retain their full output projection. The model-local CPU
`AUDIOCPP_MIRA_TTS_SPARSE_HEAD=0` diagnostic remains available.

## Output preservation

All six WAV files are byte-for-byte identical to the corresponding samples
generated before this refactor: CPU, Vulkan, and CUDA, each with Q8 and BF16.
The comparison uses SHA-256 on the complete WAV files.

Text: `Hello, this is a native Mira TTS test.`
Reference: `../models_v3_test/MiraTTS-comparison/00-reference-voice.wav`.
Seed: 1234. Maximum tokens: 1024. Other sampling parameters: CLI defaults.

Build each backend with:

```powershell
cmake --build build/windows-cpu-release --target audiocpp_cli mira_tts_warm_bench -j 12
cmake --build build/windows-vulkan-release --target audiocpp_cli mira_tts_warm_bench -j 12
cmake --build build/windows-cuda-release --target audiocpp_cli mira_tts_warm_bench -j 12
```

Example (substitute backend and precision; Vulkan uses `--device 1`):

```powershell
.\build\windows-cpu-release\bin\audiocpp_cli.exe --task clon --family mira_tts --model ..\models_v3_test\MiraTTS-GGUF\mira-tts-q8.gguf --backend cpu --text "Hello, this is a native Mira TTS test." --voice-ref ..\models_v3_test\MiraTTS-comparison\00-reference-voice.wav --seed 1234 --max-tokens 1024 --out ..\outputs\MiraTTS-q8-CPU-local-head.wav --metrics
```

Local samples are named `../outputs/MiraTTS-{q8,bf16}-{cpu,vulkan,cuda}-local-head.wav`.
The previously generated files without `-local-head` are the comparison baseline.

## Repeated CPU requests

The existing five-request offline fixture exercises the new view through
repeated requests, a changed prompt, a longer request, and graph release/rebuild.
Both precisions use `assets/resources/b.wav` with eight CPU threads:

```powershell
.\build\windows-cpu-release\bin\mira_tts_warm_bench.exe --model ..\models_v3_test\MiraTTS-GGUF\mira-tts-q8.gguf --model-spec-override model_specs/mira_tts.json --backend cpu --voice-ref assets/resources/b.wav --request-file tests/mira_tts/offline_requests.json --audio-out-dir ../outputs/mira-local-head-q8-offline --summary-file ../outputs/mira-local-head-q8-offline.json --log-file ../outputs/mira-local-head-q8-offline.log
```

Repeat with `bf16` instead of `q8` for the second precision. The fixture's token
budgets cap outputs at 5.12 and 15.36 seconds; these requests test lifecycle and
determinism rather than full-prompt coverage.

All five requests completed for each precision. The cold, immediate-repeat,
and repeat-after-long-form audio hashes match within each precision:
Q8 `eac900c3f8b032d9`, BF16 `97ef7aca310a5692`.
