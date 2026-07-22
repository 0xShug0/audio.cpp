# Model Spec Demo

This demo exercises the production C++ `engine::model_spec` subsystem with a
toy Qwen3-ASR-shaped package. It does not migrate any real model spec.

Build and run the C++ check:

```bash
cmake --build build/debug --target model_spec_demo --parallel $(nproc)
build/debug/bin/model_spec_demo \
  examples/model_spec_demo/specs/toy_qwen3_asr.json \
  examples/model_spec_demo/toy_package
```

Serve the UI demo:

```bash
python3 -m http.server 8765 --directory examples/model_spec_demo
```

Open `http://127.0.0.1:8765`.
