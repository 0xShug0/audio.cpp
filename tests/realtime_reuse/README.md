# Real-time RE-USE validation

Use the original checkpoint from
[nvidia/Real-time_RE-USE](https://huggingface.co/nvidia/Real-time_RE-USE/tree/4c0c186272ca853cd18d832fc5c28777b62f32d1).
The converter checks its SHA256 and verifies the complete tensor inventory, shapes,
and retained F32 tensor bytes after packaging.

```bash
python tests/realtime_reuse/convert_gguf.py --source SOURCE \
  --output models/Real-time-RE-USE-GGUF/realtime-reuse-f32.gguf \
  --log conversion.log
```

`run_python_reference.py --help` describes reference generation with the original
Python implementation. Use `--no-tf32` for the strict F32 waveform comparison;
leave that diagnostic switch off for ordinary Python performance measurements.
`--reference-file reference.safetensors` exports the waveform, frontend features,
and expected restored waveform together with the selected configuration.

```bash
build/debug/bin/test_realtime_reuse_reference \
  --model models/Real-time-RE-USE-GGUF/realtime-reuse-f32.gguf \
  --reference reference.safetensors --backend cuda --log --log-file parity.log

build/debug/bin/test_realtime_reuse_streaming \
  --model models/Real-time-RE-USE-GGUF/realtime-reuse-f32.gguf \
  --audio noisy.wav --backend cuda --sample-rate 8000 --all-configs \
  --log --log-file streaming.log
```

The reference test compares the final waveform after supplying the official
frontend features to the C++ network (cosine greater than 0.99999). It also logs
frontend differences separately. The streaming test covers all 30 exit/lookahead
combinations, checks early emission, input-fragmentation invariance, exact sample
counts, finite samples, long-short-long repeatability, and streaming/offline
waveform similarity through the 1→2→4→8 frame batch transitions and the padded
final batch. The first 40 ms are excluded from the last comparison because
upstream online inference deliberately skips the first-frame sweep filter.

`cli_cases.json` supplies offline, streaming, request-reuse, and long-form cases
for the existing CLI path runner. Run resource measurements sequentially.
