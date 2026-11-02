# WebGPU

WebGPU is disabled by default. Native builds use Dawn; browser builds use Emscripten's Emdawnwebgpu port and run the C++ CLI in a Web Worker. Browser inference requires no native inference server.

Run the following Bash commands from the repository root. Replace `/path/to/...` placeholders with your own installations or files.

## Native build

### Requirements

Install CMake, a C/C++ toolchain, and Dawn with its CMake package. Only the WebGPU backend requires C++20. `Dawn_DIR` must point to the directory containing `DawnConfig.cmake`.

### Configure and compile

```bash
cmake -S . -B build/debug \
  -DCMAKE_BUILD_TYPE=Debug \
  -DAUDIOCPP_MODEL_SET=custom -DAUDIOCPP_MODELS=pocket_tts \
  -DENGINE_ENABLE_WEBGPU=ON \
  -DENGINE_ENABLE_CUDA=OFF -DENGINE_ENABLE_HIP=OFF \
  -DENGINE_ENABLE_VULKAN=OFF -DENGINE_ENABLE_METAL=OFF \
  -DDawn_DIR=/path/to/dawn/lib/cmake/Dawn
cmake --build build/debug --target audiocpp_cli -j 8
```

Change `AUDIOCPP_MODELS` to the model families you need. The Linux helper is an alternative:

```bash
CMAKE_PREFIX_PATH=/path/to/dawn bash scripts/build_linux.sh \
  --backend webgpu --model-set custom --models pocket_tts \
  --build-type Debug --build-dir build/debug --target audiocpp_cli -j 8
```

Use `--backend webgpu`; if using individual backend switches instead, pass `--cuda off --webgpu on` so automatic CUDA detection cannot enable both.

### Verify and run

```bash
build/debug/bin/audiocpp_cli --version --log
build/debug/bin/audiocpp_cli --list-devices --log
build/debug/bin/audiocpp_cli \
  --task tts --family pocket_tts --model /path/to/model.gguf \
  --backend webgpu --threads 1 \
  --request-option voice_embedding_file=/path/to/voice.safetensors \
  --text "Hello from WebGPU." --out output.wav --log > run.log 2>&1
```

Use the options and sidecars required by your model. The server, C API backend string, and shared `exec_plan` parser also accept `webgpu`. `best` retains GGML's automatic selection and may select WebGPU when built in. GPU free/total memory is reported as unknown.

## Browser build

### Requirements and SDK setup

Install Emscripten and activate its SDK environment. Emscripten 6.0.13 and Chrome 150 were used for validation. The browser must support WebGPU, `shader-f16` and JavaScript Promise Integration (JSPI). Pthread builds also require cross-origin isolation.

```bash
source /path/to/emsdk/emsdk_env.sh
```

### Common browser configuration

These flags select PocketTTS for the example, disable native GPU backends, and enable pthreads for models that prefetch work asynchronously. Keep this Bash array in the same shell for the following configure commands.

```bash
browser_options=(
  -DCMAKE_BUILD_TYPE=Debug
  -DAUDIOCPP_MODEL_SET=custom -DAUDIOCPP_MODELS=pocket_tts
  -DAUDIOCPP_DEPLOYMENT_BUILD=ON
  -DENGINE_ENABLE_WEBGPU=ON -DGGML_WEBGPU_JSPI=ON
  -DENGINE_ENABLE_CUDA=OFF -DENGINE_ENABLE_HIP=OFF
  -DENGINE_ENABLE_VULKAN=OFF -DENGINE_ENABLE_METAL=OFF
  -DENGINE_ENABLE_OPENMP=OFF -DGGML_OPENMP=OFF
  -DENGINE_ENABLE_NATIVE_CPU=OFF -DENGINE_ENABLE_LLAMAFILE=OFF
  -DAUDIOCPP_BUILD_NATIVE_MODEL_MANAGER=OFF
  -DAUDIOCPP_BUILD_SERVER_FRONTENDS=OFF -DAUDIOCPP_BUILD_C_API=OFF
  -DENGINE_BUILD_TESTS=OFF
  '-DCMAKE_CXX_FLAGS_DEBUG=-O3 -g'
  '-DCMAKE_C_FLAGS_DEBUG=-O3 -g'
  '-DCMAKE_EXE_LINKER_FLAGS=-pthread -sPTHREAD_POOL_SIZE=navigator.hardwareConcurrency'
)
```

### WASM32

```bash
emcmake cmake -S . -B build/debug/browser "${browser_options[@]}" \
  '-DCMAKE_CXX_FLAGS=-fwasm-exceptions -msimd128 -pthread' \
  '-DCMAKE_C_FLAGS=-msimd128 -pthread'
cmake --build build/debug/browser --target audiocpp_cli -j 8
```

WASM32 has a growable 4 GiB address-space ceiling. Weights, graph arenas and file copies must fit within it; this is not a 4 GiB GPU-memory allowance.

### WASM64 for larger models

Use a separate build directory. Select the required model families through `AUDIOCPP_MODELS`.

```bash
emcmake cmake -S . -B build/debug/browser64 "${browser_options[@]}" \
  '-DCMAKE_CXX_FLAGS=-fwasm-exceptions -msimd128 -pthread -sMEMORY64=1' \
  '-DCMAKE_C_FLAGS=-msimd128 -pthread -sMEMORY64=1'
cmake --build build/debug/browser64 --target audiocpp_cli -j 8
```

This build permits growth up to a 16 GiB WASM address space, without allocating it all at startup. Files larger than 2 GiB use read-only WORKERFS mounts and require WASM64 in this example. Smaller files are copied into MEMFS in bounded chunks. Browser storage-buffer binding limits still apply independently.

### Serve and open the example

Emscripten's server supplies the isolation headers required by pthreads:

```bash
emrun --no-browser --no-emrun-detect --serve-after-close \
  --hostname 127.0.0.1 --port 8081 build/debug/browser/bin/index.html
```

Open `http://localhost:8081/index.html`. For WASM64, serve `build/debug/browser64/bin/index.html` instead. Use localhost or HTTPS.

Select your model GGUF, its required sidecars, and an input WAV when applicable. The GGUF is mounted as `/model.gguf`, the WAV as `/input.wav`, and sidecars as `/filename`. The resource-directory input preserves the selected directory's relative paths. Adapt the page's CLI arguments to those paths and your model. The PocketTTS example requires a compatible voice embedding sidecar and its matching `voice_embedding_file` path.

Run displays logs and offers generated WAVs/text for download. `--out-dir` collects separation outputs. Stop terminates the worker; each run starts a fresh session. External-process workflows report an error because browsers cannot spawn native programs. `--threads 1` limits model CPU threading; asynchronous prefetch may use additional workers.

### Linux Chrome adapter configuration

The tested Linux/NVIDIA setup needed these flags to expose WebGPU and `shader-f16`. Use a separate browser profile:

```bash
google-chrome --user-data-dir=/path/to/dedicated-browser-profile \
  --enable-gpu --use-angle=vulkan --enable-features=Vulkan \
  --enable-unsafe-webgpu \
  --enable-dawn-features=vulkan_enable_f16_on_nvidia \
  http://localhost:8081/index.html
```

Adapter support varies by browser and driver. A missing required feature produces an error; the example does not silently switch inference to CPU.

## Kokoro and eSpeak data

### Generate data with a native build

The browser cannot run the native eSpeak data-generation tools. Generate their package first:

```bash
cmake -S . -B build/debug \
  -DCMAKE_BUILD_TYPE=Debug \
  -DAUDIOCPP_MODEL_SET=custom -DAUDIOCPP_MODELS=kokoro_tts \
  -DAUDIOCPP_STATIC_ESPEAK=ON
cmake --build build/debug --target audiocpp_espeak_package -j 8
```

The package is written to `build/debug/_deps/espeak-static/espeak-ng-data.bin`.

### Configure the browser phonemizer

After setting up the SDK and `browser_options` above:

```bash
emcmake cmake -S . -B build/debug/browser "${browser_options[@]}" \
  -DAUDIOCPP_MODELS=kokoro_tts \
  -DAUDIOCPP_STATIC_ESPEAK=ON \
  -DAUDIOCPP_ESPEAK_DATA_PACKAGE="$PWD/build/debug/_deps/espeak-static/espeak-ng-data.bin" \
  '-DCMAKE_CXX_FLAGS=-fwasm-exceptions -msimd128 -pthread' \
  '-DCMAKE_C_FLAGS=-msimd128 -pthread'
cmake --build build/debug/browser --target audiocpp_cli -j 8
```

For WASM64, use `build/debug/browser64` and include `-sMEMORY64=1` in both compiler flag values, as in the WASM64 configuration above. The eSpeak child build inherits the same compilation and Debug flags.

Select `espeak-ng-data.bin` as a sidecar. Set the page's environment JSON to:

```json
{"AUDIOCPP_ESPEAK_DATA":"/espeak-ng-data.bin"}
```

### Run Kokoro within WASM32 memory limits

Kokoro's default host arenas can exhaust the WASM32 address space, including the default 384 MiB duration-graph arena. The successful WASM32 browser check used the explicit settings below; use this CLI argument array in the page after selecting a Kokoro GGUF and the eSpeak data sidecar:

```json
[
  "--task", "tts", "--family", "kokoro_tts",
  "--model", "/model.gguf", "--backend", "webgpu", "--threads", "1",
  "--voice-id", "af_heart",
  "--session-option", "kokoro_tts.weight_context_mb=32",
  "--session-option", "kokoro_tts.predictor_duration_graph_arena_mb=128",
  "--session-option", "kokoro_tts.predictor_text_graph_arena_mb=128",
  "--session-option", "kokoro_tts.predictor_tail_graph_arena_mb=256",
  "--text", "Hello from the browser WebGPU backend.",
  "--request-option", "seed=1234",
  "--out", "/output.wav", "--log"
]
```

These are explicit host-memory settings for the short browser example, not model-default changes or a guarantee that longer requests fit. Use WASM64 when the model and workload exceed WASM32's address space. The eSpeak library is cross-compiled; its native-generated data is reused without changing phonemization algorithms.

## Developer checks

Enable `ENGINE_BUILD_TESTS=ON` in the native WebGPU build, then run:

```bash
cmake -S . -B build/debug -DENGINE_BUILD_TESTS=ON
cmake --build build/debug \
  --target test-backend-ops backend_device_resolution_test lstm_sequence_test -j 8
build/debug/bin/test-backend-ops test -b WebGPU > backend-ops.log 2>&1
build/debug/bin/backend_device_resolution_test --log > backend-devices.log 2>&1
build/debug/bin/lstm_sequence_test --backend webgpu --log > lstm.log 2>&1
```

The GGML harness writes its own output and has no `--log` option. The optional `nemotron_asr_webgpu_rnnt_test` target requires both WebGPU and Nemotron in the selected native models; invoke it with `--model /path/to/nemotron.gguf --log`. It is a fixed-input model test, not automatic CTest coverage.

## Implementation and validation

The backend is based on [GGML commit facdc44a67da17df4433adcc468b424f48e9ded2](https://github.com/ggml-org/ggml/tree/facdc44a67da17df4433adcc468b424f48e9ded2/src/ggml-webgpu), adapted to the vendored GGML API. Model-specific integration reuses Irodori's padding/cropping path, permits WebGPU in VibeVoice, and protects Nemotron output-view backing storage. Other backend kernels and shared model math are unchanged.

Performance results, output checks and validation limits are in [the browser validation report](../reports/webgpu_browser_validation.md).
