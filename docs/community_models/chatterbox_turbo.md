# Chatterbox Turbo (community model)

[Chatterbox Turbo](https://huggingface.co/ResembleAI/chatterbox-turbo) is Resemble AI's
distilled 350M-parameter sibling of Chatterbox (see [the Chatterbox section in docs/tts.md](../tts.md#chatterbox)): a GPT2-style T3
backbone (vs. the base model's 0.5B Llama-style backbone), a GPT2 BPE tokenizer with 19 built-in
emotion/style tags (`[laugh]`, `[sigh]`, ...), and a 2-step meanflow-distilled S3Gen decoder (vs.
the base model's 10-step CFG decoder) for substantially faster generation. English-only.

**Status: testing.** The T3 backbone and the built-in default voice both load and generate
audio end to end. Custom voice cloning (a caller-supplied reference clip) is not implemented
yet — it depends on the checkpoint's speaker-encoder and S3-tokenizer sections, whose tensor
layout has not been validated. This family lives under `community_models` rather than the core
model tree because it does not yet have the CUDA/Vulkan/Metal runtime test coverage core models
carry.

## Packaging: audio.cpp-native, self-contained GGUF

Resemble AI has not published Chatterbox Turbo weights in a format audio.cpp can convert
directly. The only available conversion is a **third-party GGUF**,
[`cstr/chatterbox-turbo-GGUF`](https://huggingface.co/cstr/chatterbox-turbo-GGUF), published by
`cstr` for their own CrispASR project (MIT-relicensed) — not published by ResembleAI or
audio.cpp. It ships as two loose GGUF files (T3 and S3Gen) with a flat dot-separated tensor
namespace and abbreviated S3Gen tensor names that don't match this codebase's own naming.

Rather than teach the runtime a compatibility layer for that third-party layout,
[`tools/community_models/chatterbox_turbo/repack_chatterbox_turbo_gguf.py`](../../tools/community_models/chatterbox_turbo/repack_chatterbox_turbo_gguf.py)
repacks it offline into one self-contained, audio.cpp-native GGUF:

- T3 and built-in-conditional tensors are moved from the upstream flat `t3.`/`conds.` dot
  namespace into this project's own `/`-delimited packed-GGUF namespace convention.
- S3Gen tensors are renamed back to the exact names base Chatterbox's own S3Gen flow/HiFT-vocoder
  loader (`src/models/chatterbox/s3gen_flow.cpp`,
  `src/framework/modules/vocoders/hift_vocoder.cpp`) already expects, so that loader runs
  completely unmodified for Turbo — no tensor-name translation code exists in this family at
  runtime.
- The GPT2 BPE tokenizer (vocab, merges, and the trailing emotion/style special tokens) is
  extracted into plain `vocab.json`/`merges.txt`/`special_tokens.json` sidecar files instead of
  being read from raw GGUF metadata at load time.
- The result is fed through this project's own `audiocpp_gguf` converter, which quantizes,
  embeds the package spec (`model_specs/chatterbox_turbo.json`), and embeds the sidecar files —
  producing one file that loads with nothing else needed, like every other GGUF family here.

The `ve.*` (LSTM speaker-verification voice encoder) and `s3.se.*`/`s3.tok.*` (ResNet speaker
encoder / S3 speech tokenizer) sections of the upstream checkpoint are not repacked: nothing in
this codebase reads them yet (see Current limitations above).

### Repacking it yourself

```bash
# 1. Build the converter
cmake --build build/debug --parallel --target audiocpp_gguf

# 2. Get the upstream third-party GGUF pair (~1 GB for Q8_0)
huggingface-cli download cstr/chatterbox-turbo-GGUF \
    chatterbox-turbo-t3-q8_0.gguf chatterbox-turbo-s3gen-q8_0.gguf \
    --local-dir /tmp/chatterbox-turbo-src

# 3. Repack
pip install gguf numpy safetensors
python3 tools/community_models/chatterbox_turbo/repack_chatterbox_turbo_gguf.py \
    --t3-source /tmp/chatterbox-turbo-src/chatterbox-turbo-t3-q8_0.gguf \
    --s3gen-source /tmp/chatterbox-turbo-src/chatterbox-turbo-s3gen-q8_0.gguf \
    --output models/Chatterbox-Turbo-GGUF/chatterbox-turbo-q8_0.gguf \
    --type q8_0 --overwrite
```

Verify with `build/debug/bin/audiocpp_gguf --inspect models/Chatterbox-Turbo-GGUF/chatterbox-turbo-q8_0.gguf` (expect `embedded_sidecars=true`, `embedded_model_spec=true`, and `t3`/`conds`/`s3gen` namespaces).

## Usage

```bash
audiocpp_cli --task clon --family chatterbox_turbo \
    --model models/Chatterbox-Turbo-GGUF/chatterbox-turbo-q8_0.gguf \
    --backend cuda --text "Hello from Chatterbox Turbo." --out out.wav
```

See the [Chatterbox Turbo section in docs/tts.md](../tts.md#chatterbox-turbo) for the full option
table.

## Checkpoints

| Model | Source | License |
|---|---|---|
| Chatterbox Turbo (T3 + S3Gen) | `cstr/chatterbox-turbo-GGUF` (third-party repack of `ResembleAI/chatterbox-turbo`) | MIT |
