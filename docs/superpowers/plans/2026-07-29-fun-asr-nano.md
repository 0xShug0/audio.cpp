# Fun-ASR-Nano Native Runtime Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add complete offline Fun-ASR-Nano transcription to audio.cpp for official HF safetensors and audio.cpp GGUF on CPU and CUDA.

**Architecture:** Parse the official Transformers checkpoint into typed assets, run a shared Kaldi-LFR frontend and SAN-M encoder, project through the bidirectional adaptor, and reuse audio.cpp's Qwen causal decoder for generation. Register the completed family through the existing loader, model-spec, CLI, server, and GGUF systems.

**Tech Stack:** C++20, CMake, GGML, audio.cpp runtime/model-spec APIs, Python reference probes, safetensors, GGUF, Hugging Face Hub.

## Global Constraints

- Family id is exactly `fun_asr_nano`; first supported variant is `FunAudioLLM/Fun-ASR-Nano-2512-hf`.
- Support offline ASR only; do not advertise native streaming, timestamps, translation, diarization, or CTC decoding.
- CPU and CUDA are release gates; Metal remains supported through generic GGML operations but is not a first-release gate.
- Official safetensors and audio.cpp-owned F16, Q8_0, and Q4_K_M GGUF use one logical tensor-name layout.
- Preserve MIT attribution for math adapted from transcribe.cpp commit `c87109304c42707867f926fb7b9c378d9f46df8a`.
- Preserve the FunASR Model Open Source License Agreement v1.1 in model package metadata and docs.
- Use TDD for every task and keep `.mcp-tasks/` untracked and untouched.
- Back up the worktree with a git bundle and patch before every commit.

---

## File Map

- `include/engine/framework/audio/kaldi_fbank.h`, `src/framework/audio/kaldi_fbank.cpp`: parameterized Kaldi fbank and LFR frontend.
- `include/engine/framework/modules/speech_encoders/sanm.h`, `src/framework/modules/speech_encoders/sanm.cpp`: reusable SAN-M graph primitives.
- `include/engine/models/fun_asr_nano/assets.h`, `src/models/fun_asr_nano/assets.cpp`: typed config, resources, and tensor source.
- `include/engine/models/fun_asr_nano/types.h`: frontend, embedding, prompt, generation, and result value types.
- `include/engine/models/fun_asr_nano/frontend.h`, `src/models/fun_asr_nano/frontend.cpp`: fixed Fun-ASR frontend wrapper.
- `include/engine/models/fun_asr_nano/encoder.h`, `src/models/fun_asr_nano/encoder.cpp`: 70-block SAN-M runtime.
- `include/engine/models/fun_asr_nano/adaptor.h`, `src/models/fun_asr_nano/adaptor.cpp`: projector and two-layer bidirectional adaptor.
- `include/engine/models/fun_asr_nano/tokenizer_text.h`, `src/models/fun_asr_nano/tokenizer_text.cpp`: Qwen tokenizer wrapper.
- `include/engine/models/fun_asr_nano/prompt.h`, `src/models/fun_asr_nano/prompt.cpp`: prompt construction and ITN selection.
- `include/engine/models/fun_asr_nano/decoder.h`, `src/models/fun_asr_nano/decoder.cpp`: Qwen prefill/decode runtime.
- `include/engine/models/fun_asr_nano/session.h`, `src/models/fun_asr_nano/session.cpp`: offline task orchestration and timings.
- `include/engine/models/fun_asr_nano/loader.h`, `src/models/fun_asr_nano/loader.cpp`: runtime registration and capabilities.
- `model_specs/fun_asr_nano.json`: packages, sources, capabilities, and options.
- `tests/fun_asr_nano/`: config, frontend, stage parity, warm bench, and reference scripts.
- `docs/models/fun_asr_nano.md`, `docs/asr.md`, `docs/gguf.md`: user-facing install, CLI, server, and GGUF instructions.

### Task 1: Typed Assets And Model Spec

**Files:**
- Create: `include/engine/models/fun_asr_nano/assets.h`
- Create: `src/models/fun_asr_nano/assets.cpp`
- Create: `model_specs/fun_asr_nano.json`
- Create: `tests/unittests/test_fun_asr_nano_assets.cpp`
- Modify: `CMakeLists.txt`

**Interfaces:**
- Produces: `FunAsrNanoConfig`, `FunAsrNanoAssets`, and `load_fun_asr_nano_assets(path)`.
- Consumes: `ResourceBundle`, JSON helpers, `TensorSource`, and the existing model-spec loader.

- [ ] **Step 1: Write failing config and spec tests**

Create fixtures with nested current config and legacy flat adaptor fields. Assert the parser yields encoder `(560, 512, 50, 20, 4, 2048, 11)`, adaptor `(1024, 8, 256, 2)`, projector hidden size `2048`, and Qwen `(1024, 3072, 28, 16, 8, 128, 151936)`.

```cpp
engine::test::require(config.encoder.input_size == 560, "Fun-ASR input size");
engine::test::require(config.adaptor.ffn_dim == 256, "Fun-ASR adaptor FFN");
engine::test::require(config.text.rope_theta == 1000000.0F, "Fun-ASR RoPE");
```

- [ ] **Step 2: Run the focused test and confirm it fails**

Run: `cmake --build build/debug --target test_fun_asr_nano_assets -j 8 && build/debug/bin/test_fun_asr_nano_assets`

Expected: compilation fails because `FunAsrNanoConfig` is not defined.

- [ ] **Step 3: Implement strict config parsing and the typed spec**

Reject dimension mismatches, unsupported activation functions, untied embeddings, absent tokenizer resources, and a tensor source missing the stem or token embedding. Define the HF package and safetensors source in the typed schema.

- [ ] **Step 4: Run asset and model-spec checks**

Run: `cmake --build build/debug --target test_fun_asr_nano_assets model_spec_demo -j 8`

Run: `build/debug/bin/model_spec_demo model_specs/fun_asr_nano.json tests/fixtures/fun_asr_nano_package`

Expected: both commands pass and report family `fun_asr_nano`.

- [ ] **Step 5: Back up and commit**

```bash
git add CMakeLists.txt include/engine/models/fun_asr_nano/assets.h \
  src/models/fun_asr_nano/assets.cpp model_specs/fun_asr_nano.json \
  tests/unittests/test_fun_asr_nano_assets.cpp
git commit -s -m "Add Fun-ASR-Nano assets and model spec"
```

### Task 2: Shared Kaldi Fbank And LFR Frontend

**Files:**
- Create: `include/engine/framework/audio/kaldi_fbank.h`
- Create: `src/framework/audio/kaldi_fbank.cpp`
- Create: `tests/fun_asr_nano/reference_frontend.py`
- Create: `tests/fun_asr_nano/fun_asr_nano_frontend_probe.cpp`
- Modify: `CMakeLists.txt`

**Interfaces:**
- Produces: `KaldiFbankOptions`, `KaldiFbankFeatures`, and `extract_kaldi_fbank(audio, options)`.
- Output layout: time-major `frames * (num_mels * lfr_m)` float values and a valid-frame count.

- [ ] **Step 1: Generate pinned reference values**

The Python probe must use the Transformers `FunAsrNanoFeatureExtractor` from PR #46180 and print JSON summaries plus the first and last 32 values for silence, impulse, 440 Hz sine, and `assets/resources/sample_16k.wav`.

- [ ] **Step 2: Write the C++ probe assertions**

Assert frame counts, output dimension `560`, finite values, and per-fixture numerical parity at `atol=2e-4`, `rtol=2e-4`.

- [ ] **Step 3: Run and confirm the frontend probe fails**

Run: `cmake --build build/debug --target fun_asr_nano_frontend_probe -j 8`

Expected: compilation fails because `extract_kaldi_fbank` is not defined.

- [ ] **Step 4: Implement the frontend**

Implement pre-emphasis `0.97`, 25 ms Hamming frames, 10 ms shift, 80 HTK mel bins, natural log, centered LFR `7/6`, replicated edge frames, and optional CMVN. Keep Fun-ASR's CMVN flag off.

- [ ] **Step 5: Run parity and existing audio tests**

Run: `build/debug/bin/fun_asr_nano_frontend_probe tests/fun_asr_nano/frontend_reference.json`

Run: `ctest --test-dir build/debug -R "audio|frontend" --output-on-failure`

Expected: all pass.

- [ ] **Step 6: Back up and commit**

```bash
git add CMakeLists.txt include/engine/framework/audio/kaldi_fbank.h \
  src/framework/audio/kaldi_fbank.cpp tests/fun_asr_nano
git commit -s -m "Add Kaldi LFR audio frontend"
```

### Task 3: Shared SAN-M Graph Primitives

**Files:**
- Create: `include/engine/framework/modules/speech_encoders/sanm.h`
- Create: `src/framework/modules/speech_encoders/sanm.cpp`
- Create: `tests/fun_asr_nano/sanm_probe.cpp`
- Create: `tests/fun_asr_nano/reference_sanm.py`
- Modify: `CMakeLists.txt`

**Interfaces:**
- Produces: `SanmBlockWeightsView`, `SanmBlockConfig`, `sanm_projection_block`, `sanm_residual_block`, and `sanm_layer_norm`.
- Consumes: GGML graph context and non-owning weight tensors.

- [ ] **Step 1: Write one-block reference and failing probe**

Use deterministic small tensors with `d_input=12`, `d_model=8`, two heads, FFN 16, and kernel 3. Compare projection and residual variants separately.

- [ ] **Step 2: Run the probe and confirm missing symbols**

Run: `cmake --build build/debug --target fun_asr_nano_sanm_probe -j 8`

Expected: link or compilation failure for the SAN-M API.

- [ ] **Step 3: Implement SAN-M math**

Implement Q/K/V projection, scaled dot-product attention, depthwise FSMN over V, residual addition, affine layer norm, and ReLU FFN. Request F32 accumulation for F16 matrix multiplication.

- [ ] **Step 4: Run CPU and CUDA probe variants**

Run: `build/debug/bin/fun_asr_nano_sanm_probe --backend cpu`

Run: `build/debug/bin/fun_asr_nano_sanm_probe --backend cuda`

Expected: both pass their configured tolerances.

- [ ] **Step 5: Back up and commit**

```bash
git add CMakeLists.txt include/engine/framework/modules/speech_encoders/sanm.h \
  src/framework/modules/speech_encoders/sanm.cpp tests/fun_asr_nano
git commit -s -m "Add reusable SAN-M encoder blocks"
```

### Task 4: Fun-ASR Encoder Runtime

**Files:**
- Create: `include/engine/models/fun_asr_nano/types.h`
- Create: `include/engine/models/fun_asr_nano/frontend.h`
- Create: `src/models/fun_asr_nano/frontend.cpp`
- Create: `include/engine/models/fun_asr_nano/encoder.h`
- Create: `src/models/fun_asr_nano/encoder.cpp`
- Create: `tests/fun_asr_nano/fun_asr_nano_encoder_probe.cpp`
- Create: `tests/fun_asr_nano/reference_encoder.py`
- Modify: `CMakeLists.txt`

**Interfaces:**
- Produces: `FunAsrNanoAudioFeatures` and `FunAsrNanoEncoderRuntime::encode(features)` returning `[frames, 512]`.
- Consumes: Tasks 1-3.

- [ ] **Step 1: Write tensor catalog and stage-parity failures**

Assert required tensor names and shapes. Capture stem, main blocks 0/24/48, timestamp blocks 0/10/19, and final normalization from the HF model.

- [ ] **Step 2: Run the probe and confirm the runtime is absent**

Run: `cmake --build build/debug --target fun_asr_nano_encoder_probe -j 8`

Expected: compilation fails for `FunAsrNanoEncoderRuntime`.

- [ ] **Step 3: Implement weight loading and shape-keyed graph caching**

Load split Q/K/V tensors from the HF namespaces. Build `sqrt(512)` scaling, sinusoidal positions, one 560 -> 512 stem, 49 main blocks, normalization, 20 timestamp blocks, and final normalization.

- [ ] **Step 4: Verify stage parity on CPU and CUDA**

Run: `build/debug/bin/fun_asr_nano_encoder_probe --backend cpu --reference tests/fun_asr_nano/encoder_reference`

Run: `build/debug/bin/fun_asr_nano_encoder_probe --backend cuda --reference tests/fun_asr_nano/encoder_reference`

Expected: every captured stage passes `atol=2e-3`, `rtol=2e-3`.

- [ ] **Step 5: Back up and commit**

```bash
git add CMakeLists.txt include/engine/models/fun_asr_nano \
  src/models/fun_asr_nano tests/fun_asr_nano
git commit -s -m "Implement Fun-ASR-Nano SAN-M encoder"
```

### Task 5: Projector And Bidirectional Adaptor

**Files:**
- Create: `include/engine/models/fun_asr_nano/adaptor.h`
- Create: `src/models/fun_asr_nano/adaptor.cpp`
- Create: `tests/fun_asr_nano/fun_asr_nano_adaptor_probe.cpp`
- Create: `tests/fun_asr_nano/reference_adaptor.py`
- Modify: `CMakeLists.txt`

**Interfaces:**
- Produces: `FunAsrNanoAdaptorRuntime::adapt(encoder_embeddings, mask)` returning packed valid `[tokens, 1024]` audio embeddings.
- Consumes: `FunAsrNanoEncoderEmbeddings` and Task 1 config.

- [ ] **Step 1: Write projector and block parity tests**

Capture `linear_1`, `linear_2`, adaptor block 0, adaptor block 1, and packed valid embeddings from Transformers.

- [ ] **Step 2: Confirm the probe fails before implementation**

Run: `cmake --build build/debug --target fun_asr_nano_adaptor_probe -j 8`

- [ ] **Step 3: Implement projector, mask, and two pre-norm blocks**

Use 512 -> 2048 -> 1024 projector linears with ReLU, eight-head bidirectional attention, 256-wide ReLU FFN, residuals, and padding removal.

- [ ] **Step 4: Run CPU/CUDA parity and padding cases**

Expected: stage tensors pass `atol=2e-3`, `rtol=2e-3`; padded batches match individual inference.

- [ ] **Step 5: Back up and commit**

```bash
git add CMakeLists.txt include/engine/models/fun_asr_nano/adaptor.h \
  src/models/fun_asr_nano/adaptor.cpp tests/fun_asr_nano
git commit -s -m "Implement Fun-ASR-Nano audio adaptor"
```

### Task 6: Tokenizer, Prompt, And Qwen Decoder

**Files:**
- Create: `include/engine/models/fun_asr_nano/tokenizer_text.h`
- Create: `src/models/fun_asr_nano/tokenizer_text.cpp`
- Create: `include/engine/models/fun_asr_nano/prompt.h`
- Create: `src/models/fun_asr_nano/prompt.cpp`
- Create: `include/engine/models/fun_asr_nano/decoder.h`
- Create: `src/models/fun_asr_nano/decoder.cpp`
- Create: `tests/fun_asr_nano/fun_asr_nano_decoder_probe.cpp`
- Create: `tests/fun_asr_nano/reference_decoder.py`
- Modify: `CMakeLists.txt`

**Interfaces:**
- Produces: `FunAsrNanoPromptBuilder::build(request, audio_tokens)` and `FunAsrNanoDecoderRuntime::generate(prompt, audio_embeddings, options)`.
- Consumes: `HfTokenizerJson`, `QwenCausalDecoderModule`, generic HF sampler, and Task 5 embeddings.

- [ ] **Step 1: Write prompt token and logit parity tests**

Assert the exact system/user/assistant token sequence, audio token positions, ITN prompt variant, prefill logits, and first three greedy decode steps.

- [ ] **Step 2: Run and confirm missing decoder APIs**

Run: `cmake --build build/debug --target fun_asr_nano_decoder_probe -j 8`

- [ ] **Step 3: Implement tokenizer and prompt splice**

Resolve `<|im_start|>`, `<|im_end|>`, and audio token id `151646` from tokenizer assets. Insert one placeholder per valid audio embedding.

- [ ] **Step 4: Wrap the existing Qwen causal decoder**

Configure hidden size 1024, 28 layers, 16 query heads, 8 KV heads, head dim 128, FFN 3072, RoPE theta `1e6`, and tied embeddings. Do not duplicate Qwen layer math.

- [ ] **Step 5: Run token, prefill, decode, and KV-cache parity**

Expected: prompt ids are exact and logits pass `atol=3e-3`, `rtol=3e-3`.

- [ ] **Step 6: Back up and commit**

```bash
git add CMakeLists.txt include/engine/models/fun_asr_nano \
  src/models/fun_asr_nano tests/fun_asr_nano
git commit -s -m "Add Fun-ASR-Nano Qwen decoding"
```

### Task 7: Offline Session And Loader Registration

**Files:**
- Create: `include/engine/models/fun_asr_nano/session.h`
- Create: `src/models/fun_asr_nano/session.cpp`
- Create: `include/engine/models/fun_asr_nano/loader.h`
- Create: `src/models/fun_asr_nano/loader.cpp`
- Create: `tests/fun_asr_nano/fun_asr_nano_warm_bench.cpp`
- Create: `tests/fun_asr_nano/fun_asr_nano_warm_bench_cases.json`
- Modify: `CMakeLists.txt`

**Interfaces:**
- Produces: `make_fun_asr_nano_loader()` and an `IOfflineVoiceTaskSession` for task `asr`.
- Consumes: Tasks 1-6 and framework audio chunking.

- [ ] **Step 1: Write failing loader, option, and transcript tests**

Assert family/task/mode, rejection of streaming and timestamps, `fun_asr_nano.itn`, unknown-option errors, and deterministic transcripts for zh/en/ja fixtures.

- [ ] **Step 2: Confirm the family is unavailable**

Run: `build/debug/bin/audiocpp_cli --inspect --family fun_asr_nano --model tests/fixtures/fun_asr_nano_package`

Expected: loader registry reports unknown family.

- [ ] **Step 3: Implement session orchestration and timings**

Execute resample -> frontend -> encoder -> adaptor -> prompt -> decoder -> decode. Emit `fun_asr_nano.frontend_ms`, `encoder_ms`, `adaptor_ms`, `decoder_ms`, token count, audio frame count, and `session.wall_ms`.

- [ ] **Step 4: Register only the complete family**

Add an `audiocpp_add_model(fun_asr_nano ...)` entry with loader include and `make_fun_asr_nano_loader`.

- [ ] **Step 5: Run CLI and server smoke tests**

Run: `build/debug/bin/audiocpp_cli --task asr --family fun_asr_nano --model "$FUN_ASR_NANO_MODEL" --audio assets/resources/sample_16k.wav --text-out /tmp/fun_asr.txt`

Run the existing OpenAI transcription route test with family `fun_asr_nano` and assert the JSON transcript equals the CLI result.

- [ ] **Step 6: Back up and commit**

```bash
git add CMakeLists.txt include/engine/models/fun_asr_nano \
  src/models/fun_asr_nano tests/fun_asr_nano
git commit -s -m "Register Fun-ASR-Nano offline ASR"
```

### Task 8: GGUF Conversion And Package Validation

**Files:**
- Modify: `model_specs/fun_asr_nano.json`
- Create: `tests/fun_asr_nano/test_gguf_roundtrip.cpp`
- Create: `tests/fun_asr_nano/compare_transcripts.py`
- Modify: `tools/model_manager_deprecated.py`
- Modify: `docs/gguf.md`

**Interfaces:**
- Produces: installable F16, Q8_0, and Q4_K_M packages with embedded sidecars and spec.
- Consumes: generic `audiocpp_gguf` and the safetensors loader.

- [ ] **Step 1: Write a failing round-trip test**

Convert a tiny fixture, reopen it as a standalone GGUF, and assert exact config, tokenizer sidecars, tensor logical names, and family selection.

- [ ] **Step 2: Run the generic converter on the official checkpoint**

```bash
build/release/bin/audiocpp_gguf --family fun_asr_nano \
  --model "$FUN_ASR_NANO_MODEL" --type f16 --output fun-asr-nano-f16.gguf
```

Repeat with `q8_0` and `q4_k_m` after F16 parity passes.

- [ ] **Step 3: Validate transcript and logit drift**

Compare safetensors, F16, Q8_0, and Q4_K_M on the zh/en/ja fixture matrix. Require exact greedy transcripts for F16 and Q8_0; record any Q4_K_M text differences.

- [ ] **Step 4: Add package records after artifacts exist**

Use `audio-cpp/audio.cpp-gguf`, exact filenames, SHA-backed Hub revisions, model license metadata, and Q8_0 as the default package.

- [ ] **Step 5: Back up and commit**

```bash
git add model_specs/fun_asr_nano.json tools/model_manager_deprecated.py \
  tests/fun_asr_nano docs/gguf.md
git commit -s -m "Add Fun-ASR-Nano GGUF packages"
```

### Task 9: Documentation, Full Verification, And Draft PR

**Files:**
- Create: `docs/models/fun_asr_nano.md`
- Modify: `docs/asr.md`
- Modify: `docs/model_manager.md`
- Modify: `README.md`
- Modify: `tools/audiocpp_cli/audiocpp_cli_path_cases.json`

**Interfaces:**
- Produces: complete install, CLI, server, GGUF, limitations, licensing, and validation documentation.

- [ ] **Step 1: Add executable CLI path coverage**

Add `fun_asr_nano_offline` with `task=asr`, `mode=offline`, `family=fun_asr_nano`, a 16 kHz fixture, text output, and expected non-empty transcript.

- [ ] **Step 2: Document user workflows and limits**

Include model-manager install, safetensors and GGUF commands, OpenAI-compatible transcription endpoint, ITN option, CPU/CUDA selection, 30-second chunking, language coverage, and unsupported timestamps/streaming.

- [ ] **Step 3: Run repository verification**

```bash
cmake -S . -B build/release -DCMAKE_BUILD_TYPE=Release
cmake --build build/release --target audiocpp_cli audiocpp_server audiocpp_gguf \
  fun_asr_nano_frontend_probe fun_asr_nano_encoder_probe \
  fun_asr_nano_adaptor_probe fun_asr_nano_decoder_probe \
  fun_asr_nano_warm_bench -j 8
ctest --test-dir build/release --output-on-failure
python3 tools/check_loader_catalog_sync.py
python3 tools/audiocpp_cli/run_audiocpp_cli_path_tests.py \
  --binary build/release/bin/audiocpp_cli --case fun_asr_nano_offline
```

Expected: every command passes.

- [ ] **Step 4: Run release parity matrix**

Run zh/en/ja transcripts on CPU and CUDA for safetensors, F16, Q8_0, and Q4_K_M. Record timing, peak memory, transcript, and known limitations in `docs/reports/fun_asr_nano_validation.md`.

- [ ] **Step 5: Self-review scope and attribution**

Confirm no incomplete CTC/timestamp/streaming claims, no unrelated refactor, transcribe.cpp MIT attribution is present, and model license text is linked from every package.

- [ ] **Step 6: Back up, commit, push, and open a draft PR**

```bash
git add README.md docs tools tests
git commit -s -m "Document and validate Fun-ASR-Nano support"
git push -u origin codex/add-fun-asr-nano-audiocpp-20260729
```

Open a draft PR linked to modelscope/FunASR#3439 with exact local verification results and remaining release gates. Mark ready only after CPU/CUDA and GGUF parity are green.

## Self-Review

- Spec coverage: assets, official HF loading, frontend, SAN-M, adaptor, Qwen decoding, session, CLI/server, GGUF, CPU/CUDA, parity, docs, licensing, and rollback each map to a task.
- Placeholder scan: all steps name concrete files, interfaces, commands, and expected outcomes.
- Type consistency: frontend emits `FunAsrNanoAudioFeatures`; encoder emits 512-wide embeddings; adaptor emits packed 1024-wide embeddings; decoder consumes those embeddings and returns token ids; session returns a framework transcript.
