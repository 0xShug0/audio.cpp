# Model Spec Demo

This demo exercises the production C++ `engine::model_spec` subsystem with a
toy Qwen3-ASR-shaped package. It does not migrate any real model spec.

Build and run the C++ check:

```bash
cmake --build build/debug --target model_spec_demo model_spec_download_demo --parallel $(nproc)
build/debug/bin/model_spec_demo \
  examples/model_spec_demo/specs/toy_qwen3_asr.json \
  examples/model_spec_demo/toy_package
```

Preview the package download metadata:

```bash
build/debug/bin/model_spec_download_demo \
  examples/model_spec_demo/specs/toy_qwen3_asr.json
build/debug/bin/model_spec_download_demo \
  examples/model_spec_demo/specs/toy_qwen3_asr.json \
  toy_qwen3_asr_gguf_q8
```

Serve the UI demo:

```bash
python3 -m http.server 8765 --directory examples/model_spec_demo
```

Open `http://127.0.0.1:8765`.
