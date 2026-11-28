# Higgs Audio v3 TTS tests

The focused framework unit test validates that packed QKV/gate-up projections
match the separate projections, suffix causal masks are correct, F16 KV writes
preserve their values, and the decode graph exposes the intended CUDA paths:
grouped FlashAttention, packed SwiGLU, direct KV updates, and the
`ROPE -> VIEW -> SET_ROWS` fusion pattern.

```powershell
cmake --build build/windows-cuda-release --config Release --target decoder_packed_projection_test -j 8
ctest --test-dir build/windows-cuda-release -C Release -R decoder_packed_projection_test --output-on-failure
```

The removed five-request benchmark driver, comparison options and output
conventions are preserved in the [pre-cleanup guide](https://github.com/0xShug0/audio.cpp/blob/checkpoint/before-tests-cleanup/tests/higgs_audio_tts/README.md).
The framework unit test above remains independent of that timing workflow.
