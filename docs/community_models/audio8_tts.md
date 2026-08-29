# Audio8 TTS

Audio8 TTS Preview 0.6B (Qwen backbone) and 0.1B (Falcon-H1 hybrid Mamba2+attention) are compact multilingual text-to-speech models with zero-shot voice cloning, ported natively into audio.cpp as the community family `audio8_tts`. They use a DualAR architecture derived from Fish Audio
S2 Pro: a slow semantic transformer generates speech semantics, a fast codebook transformer expands each semantic step into a full codec frame, and a neural codec renders 44.1 kHz audio. The native path executes all three
stages directly on ggml with no Python dependency.

> **Status 2026-08-29:** `0.6B` Qwen is fully native, CPU-validated via SenseVoice ASR round-trip (`The quick brown fox…`, `你好，欢迎使用audio8。`, `Artificial intelligence…`). `0.1B` Falcon-H1 is weight-complete and builds natively (GGUF `slow.embed_tokens` + `24× mamba/attention` + `semantic_output`), but the slow AR forward is a documented stub pending the Mamba2 port — see `docs/FALCON_H1_0.1B_PORT_PLAN.md` and `src/community_models/audio8_tts/ar.cpp:861` `TODO(Falcon-H1)`.

| Field | Value |
|---|---|
| Family | `audio8_tts` |
| Model directory | any directory holding the standalone GGUF (or a safetensors snapshot layout) |
| Task | `tts`, `clon` |
| Modes | `offline` |
| Languages | auto, yue, zh, nl, en, fr, de, it, ja, ko, pl, es |
| Voice input | optional reference WAV plus its exact transcript (clone) |
| Output | mono 44.1 kHz WAV |
| Backbones | `0.6B`: Qwen `slow_backbone=qwen` (24L, dim 896, 14H/2KV, RoPE 1e6), `0.1B`: Falcon-H1 `slow_backbone=falcon_h1` (dim 512, `d_inner 768=32×24`, `d_state 64`, `d_conv 4`, `dt_rank 24`, `GQA 8/2`, RoPE 1e11) |
| GGUF examples | `Audio8-TTS-Preview-0.6b-GGUF/audio8-tts-preview-0.6b-q8_0.gguf` (1.4G), `Audio8-TTS-Preview-0.1B-GGUF/audio8-tts-preview-0.1b-q8_0.gguf` (812M) |
| External ggml SSM | `external/ggml` already provides `ggml_ssm_conv/scan` for `cpu/cuda/metal/vulkan/opencl/sycl/cann` — no fork needed |

## Source

Upstream project and checkpoints:

- Project: <https://github.com/Audio8-AI/Audio8_TTS>
- Hugging Face organization: <https://huggingface.co/Audio8> (an official
  `Audio8/Audio8-TTS-Preview-0.6B-ONNX-INT4` runtime export is published;
  the PyTorch preview checkpoint ships the files below)
- License: Apache-2.0

The port targets the HF preview checkpoint snapshot containing:

```
config.json                     flat arktts configuration
model.safetensors               226 DualAR tensors (BF16)
codec.pth                       neural codec weights (PyTorch pickle)
tokenizer.json                  Qwen-style BPE tokenizer
tokenizer_config.json           tokenizer metadata
modeling_arktts.py              slow/fast AR reference implementation
modeling_arktts_codec.py        codec reference implementation
processing_arktts.py            prompt-format reference implementation
```

## What it does

Two tasks are exposed:

- **TTS** (`--task tts`): synthesize speech from text with automatic language
  handling across the eleven advertised languages.
- **Clone** (`--task clon`, or `tts` with a reference): condition generation
  on a speaker-reference WAV and its transcript to reproduce that voice.

Generation applies repetition-aware sampling (RAS): semantic logits are
restricted to the valid semantic range plus EOS, sampled at the request
temperature, and re-sampled with a high-stability fallback distribution when
the drawn token repeats within a sliding window, which suppresses the
looping artifacts typical of small AR speech models. Long inputs are split
into word-budget chunks (default 200 characters) and concatenated.

## Usage

Build with the family enabled and run against a converted GGUF:

```bash
cmake -S . -B build/linux-cpu-release -DCMAKE_BUILD_TYPE=Release \
      -DENGINE_ENABLE_CUDA=OFF -DENGINE_ENABLE_METAL=OFF \
      -DENGINE_ENABLE_VULKAN=OFF -DAUDIOCPP_MODEL_SET=custom \
      -DAUDIOCPP_MODELS=audio8_tts
cmake --build build/linux-cpu-release --target audiocpp_cli -j "$(nproc)"
```

Plain synthesis:

```bash
audiocpp_cli --task tts --family audio8_tts \
  --model models/Audio8-TTS-Preview-0.6B-GGUF/audio8-tts-preview-0.6b-q8_0.gguf \
  --text "The quick brown fox jumps over the lazy dog." \
  --seed 42 --metrics \
  --out tts.wav
```

Zero-shot cloning — provide a clean reference WAV and the exact words spoken
in it:

```bash
audiocpp_cli --task clon --family audio8_tts \
  --model models/Audio8-TTS-Preview-0.6B-GGUF/audio8-tts-preview-0.6b-q8_0.gguf \
  --voice-ref reference.wav \
  --reference-text "The exact words spoken in reference.wav." \
  --text "Hello from my cloned voice." \
  --seed 42 --out clone.wav
```

Multiple ordered references can be conditioned through one request option:

```bash
--request-option 'multi_reference_cond=[{"audio":"ref1.wav","text":"..."},{"audio":"ref2.wav","text":"..."}]'
```

### Controls

| Option | Default | Meaning |
|---|---:|---|
| `--voice-ref <wav>` | none | Speaker-reference audio for cloning. |
| `--reference-text <text>` | none | Exact transcript of the reference audio. |
| `--request-option temperature=<f>` | `0.7` | Semantic-token sampling temperature. |
| `--request-option top_p=<f>` | `0.9` | Nucleus threshold. |
| `--request-option top_k=<n>` | `50` | Top-k limit. |
| `--request-option seed=<n>` | random | Sampling seed for reproducible output. |
| `--request-option max_new_tokens=<n>` | `1024` | Maximum semantic steps per chunk. |
| `--request-option text_chunk_size=<n>` | `200` | Word-budget chunk size in characters. |
| `--session-option audio8_tts.weight_type=<mode>` | `native` | AR matmul weight storage type. |
| `--session-option audio8_tts.codec_weight_type=<mode>` | `native` | Codec weight storage type. |
| `--session-option audio8_tts.mem_saver=<bool>` | `false` | Release cached AR graphs after each request. |
| `--session-option audio8_tts.reference_cache_slots=<n>` | `1` | Prepared reference cache slots. |

## Architecture

Three ggml graphs mirror the Python reference exactly (0.6B Qwen path):

1. **Slow semantic AR** — 24-layer Llama-style decoder (dim 896, 14 heads +
   2 KV heads, head_dim 64, FFN 4864, RoPE base 1e6, packed QKV *with*
   bias). A single prefill graph ingests the whole prompt
   (`[1, steps, 896]`) and a step graph decodes one column per step against
   a static KV cache. Vocabulary is 155776 Qwen-style tokens; valid speech
   semantics span `[semantic_begin_id, semantic_end_id]` =
   `[151678, 155773]`. **0.1B Falcon-H1 variant:** dim 512, hybrid per-layer
   `input_layernorm → parallel mamba2 (in_proj 1688=768+896+24, conv1d 896×1×4,
   dt_bias/A_log/D, out_proj) + GQA attention (q 512/512, k/v 128/512,
   RoPE 1e11) → pre_ff_layernorm → FFN (768)**, plus `embedding_multiplier
   0.1088`/`lm_head 0.0781`/`ssm/attn` multipliers from `types.h`. Current
   `falcon_forward_stateless:861` is a **stub** (RMSNorm + in_proj split +
   conv bias SiLU + gated out_proj + FFN, `attn_out=0`, no
   `ggml_ssm_conv/B/C/dt/A/D/ggml_ssm_scan` nor recurrent `conv[3,896]/ssm[64,32,24]`
   state, full recompute `O(N²)`); full Mamba2 tracked in
   `docs/FALCON_H1_0.1B_PORT_PLAN.md` M2/M3 (reuses `external/ggml` SSM backends).
2. **Fast codebook AR** — 4-layer decoder (same width, no attention biases,
   untied output head over 4096 codes). Conditioned on the slow hidden
   state, it autoregressively expands one semantic token into a frame of
   10 codebook indices (10 × 4096-entry books). Shared by both backbones
   (0.1B uses compact vocab 1024+EOS → expanded 4097 with `semantic 65537…`).
3. **Neural codec** — window-transformer encoder/decoder (8 layers, 16
   heads, FFN 1216) with Snake1d residual units, ConvNeXt blocks, causal
   transposed upsampling, and a downsample quantizer holding one semantic +
   nine acoustic codebooks. Decoding turns each 10-code frame into 2048
   samples at 44.1 kHz (≈21.5 frames/s).

Prompting follows `ArkttsProcessor._prompt_segments`: a chat-template system
turn ("convert the provided text to speech", optionally with reference text
and reference codes placed at the semantic span) followed by the user turn
and an `<|voice|>` assistant anchor. On semantic begin/end positions the
input embedding is the **plain sum** of the token embedding and all ten
codebook embeddings — deliberately without fish_audio's `1/sqrt(n)`
scaling, which arktts does not use.

Sampling is Gumbel-max (`argmax(softmax(logits/T) − log u)`) with legacy
top-k/top-p filtering, RAS window 10, and EOS `151645`.

## Porting procedure

The port reused the proven `fish_audio` family instead of starting from
scratch, since arktts shares its DualAR design:

1. **Copy + rename**: duplicate `src/models/fish_audio/*` into
   `src/community_models/audio8_tts/*` (headers likewise), rename
   `FishAudio*` → `Audio8Tts*`, rewrite namespaces and include paths, add
   the `audiocpp_add_model(audio8_tts …)` CMake block. The renamed skeleton
   had to compile before any semantic change.
2. **Contract first**: `types.h` (flat `arktts` config parsing, RAS
   parameters, uniform 10×4096 codebooks), `assets.h/.cpp`, the spec-backed
   `make_audio8_tts_loader()` in `session.cpp`, and schema-v1
   `model_specs/audio8_tts.json`.
3. **Parallel adaptation**, each citing the mirrored Python line:
   prompt format (`prompt_builder.cpp`), embedding lookup + sampling
   (`ar.cpp`), codec graphs (`codec.cpp`), option plumbing
   (`session.cpp`, `generator.cpp`).
4. **Codec conversion tooling** (below), then GGUF packaging.
5. **End-to-end validation** by SenseVoice ASR round-trip: every generated
   clip must transcribe verbatim. This caught one real bug — a leftover
   fish_audio `semantic_scale` multiply in the embedding lookup produced
   stationary noise until removed, after which seed-fixed clips transcribe
   verbatim (e.g. *"The quick brown fox jumps over the lazy dog."*, 3.44 s,
   RMS 0.13).

## Created files

| File | Purpose |
|---|---|
| `include/engine/community_models/audio8_tts/types.h` | Flat `arktts` config structs (slow/fast AR, codec, RAS params). |
| `include/.../audio8_tts/assets.h` + `src/.../assets.cpp` | Checkpoint loading: config, tokenizer, safetensors/GGUF tensor binding into host/device weights. |
| `include/.../audio8_tts/session.h` + `src/.../session.cpp` | `Audio8TtsSession`, request/session options, reference handling, and the spec-backed `make_audio8_tts_loader()`. |
| `include/.../audio8_tts/prompt_builder.h` + `src/.../prompt_builder.cpp` | Chat-prompt assembly and reference-code placement (`_prompt_segments`). |
| `include/.../audio8_tts/tokenizer_text.h` + `src/.../tokenizer_text.h/.cpp` | Text normalization matching Python `clean()` (`" ".join(split())`). |
| `include/.../audio8_tts/ar.h` + `src/.../ar.cpp` | Slow AR prefill/step graphs, fast codebook AR, embeddings, RAS/top-k/top-p/Gumbel sampling. |
| `include/.../audio8_tts/generator.h` + `src/.../generator.cpp` | Generation loop orchestration across both AR runtimes. |
| `include/.../audio8_tts/codec.h` + `src/.../codec.cpp` | Codec decode graphs (codes → 44.1 kHz waveform) and reference-audio encode for cloning. |
| `model_specs/audio8_tts.json` | Schema-v1 contract: metadata, languages, options, packages, GGUF/safetensors sources. |
| `CMakeLists.txt` | `audiocpp_add_model(audio8_tts …)` registration block. |
| `tools/community_models/convert_audio8_tts_codec.py` | Torch-free `codec.pth` → `codec.safetensors` converter. |
| `tools/community_models/convert_audio8_tts.py` | Safetensors snapshot → standalone GGUF packager. |
| `docs/community_models/audio8_tts.md` | This document. |

## Converting to GGUF

Two offline steps turn a fresh HF snapshot into standalone GGUFs. Neither tool downloads anything or requires a Python torch install.

**Step 1 — convert the codec** (`codec.pth` is a PyTorch pickle, unreadable by the C++ loader). The converter reads it with a zipfile/pickle parser, fuses new-style parametrization and legacy weight-norm pairs into the plain `*.conv.weight` / `*.in_proj.weight` keys the C++ graphs bind, and asserts shape anchors against the checkpoint:

```bash
python3 tools/community_models/convert_audio8_tts_codec.py \
    /path/to/Audio8-TTS-Preview-0.6b/codec.pth \
    /path/to/Audio8-TTS-Preview-0.6b/codec.safetensors
```

**Step 2 — package one self-contained GGUF** (AR + codec tensors, embedded config/tokenizer/spec):

```bash
# 16-bit reference package (AR BF16, codec F32 -> BF16)
python3 tools/community_models/convert_audio8_tts.py \
    --model-dir /path/to/Audio8-TTS-Preview-0.6b \
    --converter build/bin/audiocpp_gguf --type bf16

# default Q8_0 package, like fish_audio ships
python3 tools/community_models/convert_audio8_tts.py \
    --model-dir /path/to/Audio8-TTS-Preview-0.6b \
    --converter build/bin/audiocpp_gguf --type q8_0
```

Each output embeds 681 tensors (226 AR + 455 codec) in two namespaces `model_weights.*`, `codec_weights.*`). If a Q8_0 package audibly drifts, keep the codec namespace at 16 bit via the tool's mixed-storage options —codec conv stacks are quantization-sensitive.

**Where to get the inputs**: the upstream checkpoint from the
[Audio8 Hugging Face organization](https://huggingface.co/Audio8) / [Audio8-AI GitHub](https://github.com/Audio8-AI/Audio8_TTS) (file list under *Source* above). 

## Limitations and TODO

Current limitations (2026-08-29):

- **0.6B**: validated on CPU via SenseVoice ASR (`The quick brown fox…` 3.02s RMS 0.15, `你好，欢迎使用audio8。` 2.32s, `Artificial intelligence…` 2.97s) with `--family audio8_tts` GGUF `q8_0`/`bf16`; CUDA/Vulkan/Metal SSM backends are built but not yet exercised for this family (ggml kernels already present).
- **0.1B**: weight-complete and `audiocpp_cli` builds, but slow AR is a documented stub (`ar.cpp:861 TODO(Falcon-H1)`, `attn_out=0`, no `ggml_ssm_scan`/state, full recompute) — STT currently `like.` vs target; no `/tmp` writes or Python dependency. Full hybrid Mamba2 port is planned, not blocked on ggml (see `docs/FALCON_H1_0.1B_PORT_PLAN.md`).
- Offline mode only — no streaming session path.
- Cloning uses the same DualAR loop (reference WAV + `reference_text` → prompt builder) — 0.6B cloning path is structurally identical to TTS but has not yet been ASR-evaluated against real reference voices beyond the TTS evidence above.
- No conversation-turn continuation (Python supports multi-turn prompting; the C++ v1 path is single-request).
- ASR round-trip verifies intelligibility, not speaker similarity; formal parity runs against the Python/ONNX reference are still outstanding.

TODO:

- [ ] Complete 0.1B Falcon-H1 Mamba2 port (`ggml_ssm_conv/B/C/dt/A/D/scan` + recurrent `conv/ssm` ring + hybrid attention `K=1`) — `docs/FALCON_H1_0.1B_PORT_PLAN.md` M2/M3 (3–3.5d), then SenseVoice verification of `out/*0.1b.wav`
- [ ] Support streaming
- [ ] Clone-task validation with real reference voices (0.6B first, then 0.1B after Mamba2)
- [ ] Exercise CUDA/Metal/Vulkan backends for both models (ggml SSM kernels already vendored)
