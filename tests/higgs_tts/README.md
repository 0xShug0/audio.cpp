# Higgs Audio v3 TTS tests

The focused framework unit test validates that packed QKV/gate-up projections
match the separate projections, suffix causal masks are correct, F16 KV writes
preserve their values, and the decode graph exposes the intended CUDA paths:
grouped FlashAttention, packed SwiGLU, direct KV updates, and the
`ROPE -> VIEW -> SET_ROWS` fusion pattern.

```powershell
cmake --build build/windows-cuda-release --config Release --target qwen_decoder_packed_projection_test higgs_tts_warm_bench -j 8
ctest --test-dir build/windows-cuda-release -C Release -R qwen_decoder_packed_projection_test --output-on-failure
```

Run the fixed-seed, five-request CUDA benchmark and save every generated WAV:

```powershell
tests/higgs_tts/run_cuda_performance.ps1 `
  -Model ../models/higgs-audio-v3-tts-4b_Q8/higgs-audio-v3-tts-4b_Q8.gguf `
  -Label candidate
```

Compare a candidate run with a prior result directory request by request:

```powershell
tests/higgs_tts/run_cuda_performance.ps1 `
  -Model ../models/higgs-audio-v3-tts-4b_Q8/higgs-audio-v3-tts-4b_Q8.gguf `
  -Label candidate `
  -Baseline tests/higgs_tts/results/baseline
```

The comparison reports frame counts, wall time, RTF, speedup, waveform cosine,
and 80-band log-mel cosine. Result WAVs, logs, and JSON reports are written below
`tests/higgs_tts/results/`, which is intentionally ignored by Git.
The comparison helper requires Python 3 with NumPy.

Add `-RequireSameFrames` when comparing paths that are expected to be
deterministic and numerically identical. Sampled or mixed-precision paths still
report their frame drift and similarity metrics without hiding the results.
