# Original Metal IM2COL outputs

`original_metal_sha256.txt` was captured on an Apple M4 Pro from the unmodified
Metal backend at audio.cpp commit `c8365f391307a97bfe133e0872e632fe572ecace`,
the parent of the tiled-kernel change. The baseline checkout was clean. Its
vendored GGML was built in Release mode with the Metal library embedded.

Each record contains:

```text
case_id ne0 ne1 ne2 ne3 output_bytes input_sha256 output_sha256_first output_sha256_reused
```

The output hashes cover every raw output byte, including F16 rounding and signed
zero. They are frozen outputs from the original GPU kernel, not hashes of the
scalar reference or the tiled kernel. Hashes keep the six full-size HiFT outputs
in the fixture without storing hundreds of megabytes of tensors.

Inputs use an integer-derived pattern with explicit boundary values. On the
second execution, the input is negated and uploaded again. The output buffer is
poisoned differently before each execution. Every output element is also checked
against a separate scalar indexing implementation.

Coverage includes 1D and 2D, F32/F16, time and patch-column tile boundaries,
stride, padding, dilation, odd/even kernels through 1024 taps, multiple channels,
batches through 257, contiguous views with nonzero offsets, the public
`ggml_conv_1d_fast_1d_im2col` lowering, and production HiFT shapes.

## Run

```sh
cmake -S . -B build -DENGINE_BUILD_EXTENDED_TESTS=ON -DENGINE_ENABLE_METAL=ON
cmake --build build --target metal_im2col_test
ctest --test-dir build -R '^metal_im2col_test$' --output-on-failure
```

This test is enabled by `ENGINE_BUILD_EXTENDED_TESTS`, independently
of `ENGINE_BUILD_TESTS`. Enabling unit tests alone does not build or register it.

`build/bin/metal_im2col_test --benchmark` additionally reports median compute time
over five synchronized executions of each production shape after the two
correctness executions. Timing has no pass/fail threshold. The test needs no
model weights and returns CTest's skip code 77 when no Metal device is available.

## Updating coverage

Capture new expected outputs only against the original revision above. Compile
`tests/unittests/test_metal_im2col.cpp` against that revision's GGML headers and
static Metal/base libraries, with `METAL_IM2COL_CAPTURE_BASELINE` defined and
`METAL_IM2COL_GOLDEN_PATH` defined to a quoted string. Run that baseline-only
executable with `--capture-baseline <output-file>` and review the fixture diff.
The capture build validates every element against the scalar reference before
writing its hashes. The normal CMake test build does not expose capture mode and
never updates fixtures.
