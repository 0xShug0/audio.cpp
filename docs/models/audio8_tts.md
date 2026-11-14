# Audio8 TTS

Audio8 TTS Realistic and Preview 0.6B (Qwen backbone) and 0.1B (Falcon-H1 hybrid Mamba2+attention) are compact multilingual text-to-speech models with zero-shot voice cloning, ported natively into audio.cpp as the community family `audio8_tts`. They use a DualAR architecture derived from Fish Audio
S2 Pro: a slow semantic transformer generates speech semantics, a fast codebook transformer expands each semantic step into a full codec frame, and a neural codec renders 44.1 kHz audio. The native path executes all three
stages directly on ggml with no Python dependency.

## Usage

The default downloadable package is [Audio8 TTS Realistic 0.6B](https://huggingface.co/audio-cpp/Audio8-TTS-Realistic-0.6b-GGUF), available in Q8_0 and BF16. It uses the existing `audio8_tts` family; the Preview package remains available for existing installations.

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
  --model models/Audio8-TTS-Realistic-0.6B-GGUF/audio8-tts-realistic-0.6b-q8_0.gguf \
  --text "The quick brown fox jumps over the lazy dog." \
  --seed 42 --metrics \
  --out tts.wav
```

Zero-shot cloning — provide a clean reference WAV and the exact words spoken
in it:

```bash
audiocpp_cli --task clon --family audio8_tts \
  --model models/Audio8-TTS-Realistic-0.6B-GGUF/audio8-tts-realistic-0.6b-q8_0.gguf \
  --voice-ref reference.wav \
  --reference-text "The exact words spoken in reference.wav." \
  --text "Hello from my cloned voice." \
  --seed 42 --out clone.wav
```

Multiple ordered references can be conditioned through one request option:

```bash
--request-option 'multi_reference_cond=[{"audio":"ref1.wav","text":"..."},{"audio":"ref2.wav","text":"..."}]'
```

## Model

| Field | Value |
|---|---|
| Family | `audio8_tts` |
| Model directory | any directory holding the standalone GGUF (or a safetensors snapshot layout) |
| Task | `tts`, `clon` |
| Modes | `offline`, `streaming` |
| Languages | auto, yue, zh, nl, en, fr, de, it, ja, ko, pl, es |
| Voice input | optional reference WAV plus its exact transcript (clone) |
| Output | mono 44.1 kHz WAV |
| Backbones | `0.6B`: Qwen `slow_backbone=qwen` (24L, dim 896, 14H/2KV, RoPE 1e6), `0.1B`: Falcon-H1 `slow_backbone=falcon_h1` (dim 512, `d_inner 768=32×24`, `d_state 64`, `d_conv 4`, `dt_rank 24`, `GQA 8/2`, RoPE 1e11) |
| Default GGUF | `Audio8-TTS-Realistic-0.6B-GGUF/audio8-tts-realistic-0.6b-q8_0.gguf` |
| Preview GGUF examples | `Audio8-TTS-Preview-0.6b-GGUF/audio8-tts-preview-0.6b-q8_0.gguf` (1.4G), `Audio8-TTS-Preview-0.1B-GGUF/audio8-tts-preview-0.1b-q8_0.gguf` (812M) |

“Model/session default” means no explicit override. The runtime uses the
checkpoint or session setting, or leaves the optional value unset as described.

## Common Options (use directly)

| Option | Meaning |
|---|---|
| `--model` | Model file or directory shown in the examples. |
| `--backend` | Select an available backend supported by this model. |
| `--threads` | CPU thread count. |
| `--log` | Enable runtime logging. |
| `--text` | Input text; see the examples for the expected prompt format. |
| `--voice-ref` | Reference audio for voice cloning. |
| `--out` | Output WAV file. |

## Request Options (use with `--request-option`)

| Option | Type / values | Default | Description |
|---|---|---|---|
| `reference_text` | `string` | Model/session default | Reference transcript used with speaker reference audio. |
| `multi_reference_cond` | `string` | Model/session default | Ordered Audio8 TTS reference conditioning pairs as a JSON array: [{"audio":"ref.wav","text":"transcript"}, ...]. |
| `max_tokens` | `int` | `1024` | Maximum semantic tokens to generate; default 1024. Zero selects the default. Declared range: minimum 0. |
| `text_chunk_size` | `int` | `200` | Maximum characters per generated text chunk; default 200. Declared range: minimum 1. |
| `text_chunk_mode` | `word_budget`, `tag_aware`, `japanese`, `endline` | `word_budget` | Framework text chunking mode. |
| `top_p` | `float` | `0.8` | Top-p sampling value in (0, 1]; default 0.8. Declared range: minimum 0.0, maximum 1.0. |
| `top_k` | `int` | `30` | Top-k sampling value; default 30. Declared range: minimum 1. |
| `temperature` | `float` | `0.8` | Semantic-token sampling temperature in (0, 2); default 0.8. Declared range: minimum 0.0, maximum 2.0. |
| `seed` | `int` | random | Sampling seed for reproducible output; omitted uses a random seed. Declared range: minimum 0. |
| `language` | `string` | Model/session default | Language hint for Chinese-script normalization; Cantonese/yue preserves Traditional characters. Inherits structured transcript or voice language when provided. |

## Session Options (use with `--session-option`)

| Option | Type / values | Default | Description |
|---|---|---|---|
| `audio8_tts.mem_saver` | `bool` | `false` | Release cached AR runtime graphs after each request; default false. |
| `audio8_tts.reference_cache_slots` | `int` | `1` | Prepared reference-audio cache slots; default 1. Declared range: minimum 0. |
| `audio8_tts.weight_type` | `native`, `f32`, `f16`, `bf16`, `q8_0` | `native` | AR matmul weight storage type; default native. |
| `audio8_tts.codec_weight_type` | `native`, `f32`, `f16`, `q8_0` | `native` | Codec conv/matmul weight storage type; default native. |
| `audio8_tts.ar_graph_arena_mb` | `int` | `512` | AR runtime graph arena size in MiB; default 512. Declared range: minimum 1. |
| `audio8_tts.ar_weight_context_mb` | `int` | `512` | AR weight context size in MiB; default 512. Declared range: minimum 1. |
| `audio8_tts.codec_graph_arena_mb` | `int` | `512` | Codec encode/decode graph arena size in MiB; default 512. Declared range: minimum 1. |
| `audio8_tts.codec_weight_context_mb` | `int` | `512` | Codec weight context size in MiB; default 512. Declared range: minimum 1. |

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
