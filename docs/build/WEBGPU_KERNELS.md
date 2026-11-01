# WebGPU backend and kernel update

## Source provenance

WebGPU backend source, shader library, shader preprocessor, WGSL shaders, CMake configuration and public header imported from [ggml-org/ggml](https://github.com/ggml-org/ggml/tree/facdc44a67da17df4433adcc468b424f48e9ded2/src/ggml-webgpu), commit `facdc44a67da17df4433adcc468b424f48e9ded2`. The import uses upstream line endings. Backend-local compatibility adaptations omit references to upstream-only `Q2_0`, `SWIGLU_CLAMP`, mmap capability and multi-buffer interface members absent from the vendored GGML API; shared headers and other backends are not extended for these unrelated features. Other GGML backends and shared GGML operator math are not replaced.

## Scope

This change updates only the vendored GGML WebGPU backend, its public backend header, and the existing GGML operator tests. AudioCPP backend selection, framework, model, browser application, and root build integration are separate work. Native WebGPU uses Dawn; browser WebGPU uses Emscripten.

## Local extensions and fixes

- F16 `CONCAT` uses the existing concat shader's indexing and copy path with F16 storage. CPU-reference checks passed 54 cases across F32/I32/F16, including all axes, strided views and odd tails, plus eight additional F16 shared-input cases. F32/I32 indexing and arithmetic are unchanged.

- `ROUND_BF16` F32 unary added using the existing unary pipeline/cache and GGML's ties-to-even bit conversion, including quiet-NaN handling. CPU-reference operator checks passed (4/4 normal-value contiguous/strided cases).

- Unsupported operation dispatch now fails explicitly instead of silently skipping unimplemented nodes.

- `MEAN` reuses the row-sum reduction. Row-sum dispatch also bounds overprovisioned workgroups and writes the correct flattened row for two-dimensional dispatch. MEAN cases (8) and SUM_ROWS cases (11) passed CPU-reference checks, including 65,537 rows. A combined run hit an existing CPU SUM fixture assertion; see `logs/webgpu/reductions-reference.log`.

- F32 `CONV_TRANSPOSE_1D` added using a cached pipeline, independent output writes, and existing tensor binding/dispatch helpers. CPU-reference checks passed (116/116).

- `LEAKY_RELU` added through the existing unary shader and cache; 7/7 CPU-reference cases passed.

- `PAD_REFLECT_1D` reuses the existing pad shader and pipeline cache; 5/5 CPU-reference cases passed, including asymmetric and maximal valid padding.

- Dawn device creation is now stable across backend instances; registry lookups no longer replace live devices or reset the device count. Existing backend test extended with a live-buffer/pipeline lifetime case; passed on WebGPU, with CUDA/Vulkan selection checks also passing.

- F32 `DIAG_MASK_INF` added through the existing unary pipeline; 5/5 CPU-reference cases passed (no past, cached past, fully unmasked, multiple heads/batches).

- BF16 destination copies use packed 32-bit atomic updates to preserve neighboring BF16 elements; 32 combined BF16 copy cases passed, including scalar, odd-sized, permuted and transposed tensors.

- BF16/BF16 matrix-vector multiplication (up to four query columns) added to the existing scalar matvec shader, preserving BF16 values through the existing unpack helper; 3/3 CPU-reference cases passed, including an odd inner dimension.

- Imported F32/BF16/Q8 matrix paths narrowed F32 activations through FP16 staging, producing infinities on real FastConformer subsampling output (CUDA reference reached about 169,000). Register-tile F32 staging preserves that range. The F16-weight path retains its existing FP16 RHS conversion, matching the CPU reference. Wide-range tests therefore cover F32/BF16/Q8 weights, not F16 weights: inputs above 65,504 overflow the F16 CPU reference and cannot provide a valid finite comparison. Regular F16 tests remain unchanged.

## Validation evidence

After extraction, the Debug standalone GGML harness passed 9/9 focused matrix cases and 169/169 copy, concat, masking, mean, row-sum and reflection-padding cases against the CPU reference. Logs are under `logs/webgpu/kernel-pr/fixed-matmul.log` and `logs/webgpu/kernel-pr/added-ops-fixed.log`.

An initial full-suite attempt aborted in the CPU reference on the invalid added F16 wide-range fixture. That case was removed without changing tolerances or the valid F32/BF16/Q8 overflow checks. The subsequent broad run was stopped to rebuild the corrected fixture; a full-suite pass has not been established. Original failure logs are retained under `logs/webgpu/kernel-pr/`.

Other counts above refer to earlier checks on the full WebGPU development branch. The Dawn lifetime check lives in the AudioCPP integration tests and is not included in this kernel-only change. No new inference or performance measurements were run for the split; model performance results do not establish whole-model waveform parity.
