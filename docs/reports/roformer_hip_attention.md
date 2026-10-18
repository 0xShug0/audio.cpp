# RoFormer HIP FP32 layout optimization

Preferred candidate, September 16 2026. Supersedes the earlier HIP flash-attention
candidate. Based on upstream `048c9a0ce4151a590ced13074d71931808bafd1c`, tested on
Radeon 8060S gfx1151 / ROCm 10.1 nightly / BS-RoFormer ep368 Q8_0.

## Change and reason

Materialize the left operand of RoFormer's explicit attention matrix products
as contiguous on HIP. The right operand is already materialized. This enables
strided batched hipBLASLt GEMM instead of its per-batch-element launch fallback.
The existing FP32 tensor types, FP32 attention precision, scaling and softmax
remain in use. CUDA, Vulkan and other backend paths are unchanged. No global environment
setting is required.

The earlier flash candidate was fast but GGML internally converted keys/values
to FP16, despite the FP32 attention accumulation option. Accumulation order alone
was therefore an incomplete explanation of its differences. Packing FP32 changes
GEMM scheduling and can also affect rounding, so bitwise equivalence is not promised.

## Controlled observations

Same input/model, fresh CLI processes, sequential GPU work; seconds include process
startup. These are a small set of workstation measurements, not broad statistical
performance guarantees.

| Path | 3-second speech/music mixture | Maximum difference from first baseline | Vocal RMS difference |
|---|---:|---:|---:|
| Original upstream | 140.536 s | reference | reference |
| Original upstream repeated, unchanged binary/input | 136.273 s | 12 PCM16 steps | 3.962e-5 |
| Earlier HIP flash candidate | 4.722 s | 17 PCM16 steps | 4.853e-5 |
| Preferred FP32 layout fix | 5.925 s | 9 PCM16 steps | 3.785e-5 |

The FP32 candidate is about 24x faster than the first upstream run and its measured
RMS difference from that run is slightly below the unchanged upstream repeat's
difference. This does not prove equality or greater accuracy: the original itself
exhibits process-to-process numerical variation, whose exact source is not isolated.
A second candidate run took 5.824 s and differed from the first candidate by at most
9 PCM16 steps, with vocal RMS difference 3.777e-5.

On one-second speech, FP32 packing took 3.971 s versus original 4.421 s, maximum
error 3 PCM16 steps. On a twelve-second chirp probe, the original timed out at
180.030 s, the earlier flash path took 10.781 s and FP32 packing took 14.142 s.
No output-parity claim is made for that timed-out baseline. Other AMD architectures,
Mel-Band RoFormer, longer real recordings and listening quality need more coverage.

## Reproduction

Use the same installed model and 44.1 kHz input, with separate baseline/candidate
executables. The output directory must be new. Retain model/source revisions and
hashes alongside the results.

```sh
python tests/roformer/benchmark_hip_attention.py --binary /path/to/baseline \
  --model /path/to/BS-RoFormer-ep368-GGUF --audio /path/to/input.wav \
  --out /new/baseline --timeout 180
python tests/roformer/benchmark_hip_attention.py --binary /path/to/candidate \
  --model /path/to/BS-RoFormer-ep368-GGUF --audio /path/to/input.wav \
  --out /new/candidate --reference /new/baseline --timeout 180
```

Omit `--reference` if the baseline did not complete. The helper records command,
input/executable hashes, GPU environment overrides, elapsed time and PCM16 stem
differences. Repeat the baseline as well as the candidate before attributing all
numerical differences to a patch. No local sound playback is used.

## Implementation attribution

Implemented using OpenAI Codex. Measurements above were collected from local
HIP executions; output comparisons do not establish bitwise equivalence.
