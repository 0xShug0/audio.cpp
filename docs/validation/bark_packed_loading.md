# Native compressed Bark validation

Standalone original compressed `.bark` loading for Standard and Plus, based on merged Pro support in main. It does not depend on the separate Standard/Plus GGUF ports. Validated on Windows 11, Ryzen 7950X3D, RTX 3090 24 GiB, MSVC 14.43, CUDA 12.4 and Vulkan SDK 1.4.350.0. No backend or GGML kernel changes.

Both checkpoints reproduce all 699 inference tensors bit for bit against the independent official/Python reconstruction (after the native loader compatibility transforms). The 24 training counters are read separately and excluded from inference.

54 short/repeated fixture comparisons passed (2 checkpoints x 3 backends x 9 requests). All texts and word timestamps match existing exact mixed-F16 GGUF runs. The five-minute recording also matches the original transcript and the GGUF word timestamps for every backend/checkpoint.

## Storage

| Checkpoint | Exact mixed-F16 GGUF | Original `.bark` | Package including sidecars | Reduction in weight file |
|---|---:|---:|---:|---:|
| Underdog-Bark-3.0 | 1301.6 MB | 163.4 MB | 164.6 MB | 87.45% |
| Underdog-Bark-3.0-Plus | 1301.6 MB | 179.5 MB | 180.7 MB | 86.21% |


## Five-minute recording

8 threads, RTX 3090, full context, original 299.994-second recording. Each pair ran serially in fresh processes using the same build, fixture and options. CPU: one full request after a short warm-up; GPU: median of the last two of three full requests after a short warm-up. Single paired measurements are not a broad throughput claim.

| Checkpoint | Backend | GGUF / packed inference (s) | GGUF / packed peak RSS (GiB) | GGUF / packed peak extra VRAM (GiB) |
|---|---|---:|---:|---:|
| standard | cpu | 33.612 / 33.396 | 4.56 / 4.56 | 0.00 / 0.00 |
| standard | cuda | 0.769 / 0.771 | 2.18 / 1.66 | 5.06 / 5.06 |
| standard | vulkan | 0.866 / 0.890 | 2.12 / 1.60 | 5.01 / 5.07 |
| plus | cpu | 33.484 / 33.562 | 4.56 / 4.56 | 0.00 / 0.00 |
| plus | cuda | 0.761 / 0.765 | 2.18 / 1.68 | 5.06 / 5.29 |
| plus | vulkan | 0.857 / 0.890 | 2.11 / 1.61 | 5.07 / 5.07 |

Memory is sampled every 50 ms and includes expanded model weights, graph buffers and backend pools. GPU numbers use total device memory above each process baseline, so desktop/WDDM activity can affect individual peaks. CPU peak is dominated by the full-context graph. CUDA timings are essentially unchanged here; Vulkan packed runs were about 3-4% slower in this pass.

## Tradeoffs and scope

- The original checkpoint remains compressed on disk, and inference creates no expanded file. Python is used only to prepare verified sidecars.
- Weights expand tensor by tensor for existing native operations. This is not 163/180 MB GPU inference. No shared GGML/CUDA/Vulkan kernels changed.
- Decoding adds about 5-6 seconds to fresh-process startup in these measurements. Long-lived servers pay that cost on model/session preparation, not each request.
- Pro distributes ordinary BF16 safetensors (~1.25 GB), not this compressed format; its existing GGUF path is retained.
- GGUF remains the faster-starting option. Metal/HIP are not runtime-validated.
- Raw commands, transcripts, per-tensor hashes and sampled measurements remain local under `outputs/bark-native/`. Published checkpoint revisions/hashes are in `tools/community_models/prepare_bark_packed.py` and the compressed-loading guide.

## Runtime and regression checks

- 129 HTTP/lifecycle checks passed (43 per backend), covering both package-directory and direct `.bark` paths, model eviction/reloading, invalid-request recovery, ten repeated requests per model, and buffered streaming.
- 12 CTest executions passed (6 targets in each CPU/GPU build), including the fixed compressed fixture and malformed-input bounds/checksum checks.
- After extraction: 18 Pro/Phonon converter regression tests and four standalone package-preparation tests pass. Catalog self-tests and synchronization pass.
- cpu / standard: repeated-request idle RSS range 0.02 MiB; device VRAM range 0.00 MiB.
- cpu / plus: repeated-request idle RSS range 0.05 MiB; device VRAM range 0.00 MiB.
- cuda / standard: repeated-request idle RSS range 0.03 MiB; device VRAM range 0.00 MiB.
- cuda / plus: repeated-request idle RSS range 0.10 MiB; device VRAM range 0.00 MiB.
- vulkan / standard: repeated-request idle RSS range 1.00 MiB; device VRAM range 0.00 MiB.
- vulkan / plus: repeated-request idle RSS range 0.68 MiB; device VRAM range 0.00 MiB.
- Final rebuilt public CLI passes on CPU, CUDA and Vulkan, including a direct `.bark` path. Existing Pro BF16 GGUF and stock Parakeet Q8 GGUF inference smoke regressions pass.

## Reproduce

Follow the [compressed-loading guide](../models/bark_packed_loading.md) to download
pinned source files and prepare the native package. Build commands match the
existing [Bark build guide](../models/underdog_bark3.md#convert): use the custom
`parakeet_tdt` model set, enable tests, and build `audiocpp_cli`,
`audiocpp_server`, `parakeet_warm_bench`, `bark_packed_probe` and
`bark_packed_test`. The CPU build disables both GPU backends; the combined GPU
build enables CUDA and Vulkan (CUDA architecture 86 on the test machine).

```bash
ctest --test-dir build/bark-packed -R '^(bark_packed_test|parakeet_bark3_session_test|parakeet_variant_config_test|phonon_hotwords_test|gguf_tensor_source_test|model_spec_system_test)$' --output-on-failure
python -m unittest discover -s tests/parakeet_tdt -p 'test_*conversion.py'
python -m unittest discover -s tests/parakeet_tdt -p test_bark_packed_package.py
python tools/check_loader_catalog_sync.py --self-test
python tools/check_loader_catalog_sync.py

# Compare independent original-reference manifests, in one reused native session.
python tests/parakeet_tdt/validate_bark3.py --bin-dir build/bark-packed/bin \
  --model models/Underdog-Bark-3.0-Native --backend cuda \
  --reference original-results.json --output-dir outputs/bark-native-parity

# Native full-recording measurement. Keep the same backend, threads, warm-up and
# input for the GGUF comparison; add --model-spec-override embedded for GGUF.
build/bark-packed/bin/parakeet_warm_bench \
  --model models/Underdog-Bark-3.0-Native --backend cuda --device 0 --threads 8 \
  --audio movie-full.wav --warmup-audio tests/parakeet_tdt/assets/2086-149220-0033.wav \
  --warmup 1 --iterations 3 --timing-file outputs/bark-native-timing.log
```

Use `--iterations 1` for the reported CPU measurement. Set `OMP_NUM_THREADS=8`
and `OMP_WAIT_POLICY=PASSIVE`. On the test machine the GPU build additionally
used `GGML_VK_VISIBLE_DEVICES=1` to hide the AMD iGPU before choosing device 0.
The original-reference manifests were captured with the independent official
Standard/Plus Python loaders during the model-port validation; the six fixtures are LibriSpeech
`2086-149220-0033.wav`, four excerpts from the user's video (start/middle/late/
30 seconds), and one second of silence. The full recording is the user's
299.99375-second video, decoded to 16 kHz mono PCM. The private recording is
not redistributed. These tests establish fixture parity, not corpus WER.

## Standalone extraction checks

The separate format PR starts from main after Pro (#1130) merged. Standard/Plus
GGUF conversion remains in their own model PRs. Its decoder and Parakeet runtime
source files are byte-for-byte identical to the validated combined implementation
(`2bcbca34`); only preparation tooling and documentation were separated.

Both checkpoints were prepared again with `prepare_bark_packed.py`. Their
published compressed SHA256 hashes and bytes are unchanged. All 54 short/reused
request comparisons pass again on CPU/CUDA/Vulkan, including exact word timestamps
against the pre-split runs. Public CLI and Pro/stock-Parakeet GGUF regressions
also pass after the split. No inference change was introduced, so the earlier
five-minute paired measurements above remain applicable.
