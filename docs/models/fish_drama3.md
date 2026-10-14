# Fish Audio Drama 3 (experimental Fish variant)

Drama 3 uses `--family fish_audio`, selected by its embedded `fish_omni`
configuration. S2 Pro keeps its existing runtime. Drama 3 adds a Qwen3.5 MoE
backbone with hybrid full/linear attention and a shared/residual codebook tower
whose attention caches persist over audio frames. No CUDA/Vulkan kernels change.
Its tokenizer uses Qwen3.5 splitting and NFC normalization through the pinned,
model-private `external/utf8proc` dependency; S2 Pro keeps its prior tokenizer.

The tested checkpoint is [fishaudio/drama-3](https://huggingface.co/fishaudio/drama-3/tree/e0a9218aa48888270f06ef1ed9ea67622c0e17d2).
The reference is [Fish Speech](https://github.com/fishaudio/fish-speech/tree/541566932abbaebcb0a40e562a1f846738383f38).
Weights retain the checkpoint's [Fish Audio Community License](https://huggingface.co/fishaudio/drama-3/blob/e0a9218aa48888270f06ef1ed9ea67622c0e17d2/LICENSE).

## Prepare and run

Download the official config/tokenizer sidecars, all eight safetensors shards,
and `codec.pth` from that pinned revision into `models/Drama-3-source`.
Build the conversion helper together with the CLI/server:

```bash
cmake -S . -B build -DENGINE_ENABLE_CUDA=ON -DENGINE_BUILD_MODEL_TESTS=ON \
  -DAUDIOCPP_MODEL_SET=custom -DAUDIOCPP_MODELS=fish_audio
cmake --build build --parallel --target audiocpp_cli audiocpp_server drama3_quantize

python tools/community_models/convert_fish_drama3.py \
  --source models/Drama-3-source \
  --source-revision e0a9218aa48888270f06ef1ed9ea67622c0e17d2 \
  --quantizer build/bin/drama3_quantize \
  --staging build/drama3-converted-tensors \
  --output models/Drama-3-GGUF/drama-3-q4_k.gguf

build/bin/audiocpp_cli --task tts --family fish_audio \
  --model models/Drama-3-GGUF/drama-3-q4_k.gguf --backend cuda \
  --text "Hello from Drama three." --seed 1234 --max-tokens 256 --out drama3.wav
```

On Windows, append `.exe` to executable paths. For Vulkan use
`ENGINE_ENABLE_VULKAN=ON` and select your GPU with `--device`; the RTX 3090 was
Vulkan device 1 on the tested Windows machine. CPU is also supported.

The self-contained Q4_K package is 20.32 GiB. Expert/dense matrices are Q4_K;
embeddings and speech heads remain BF16, sensitive routers/norms are F32,
codec linear weights are Q8_0, and codec convolutions are F32. Conversion folds
codec weight normalization and applies the original 128x128 FP8 block scales.
Only EOS and the 4096 speech rows of the text output head are retained.
The original checkpoint is unchanged. Staging is resumable and tied to the
source file identities; `--pack-only` rebuilds metadata from completed payloads.

The default converted context is 4096 rows, including prompt and generated
frames; `--context-size` changes it during conversion (128..8192). Requests that
exceed it are rejected. Use native weight storage; expanding this 35B model to
F32/F16 on a 24 GiB GPU will not fit. Q8/BF16 converter modes are available but
were not validated as full packages on this machine. No downloadable Drama 3
package is advertised in the model manager.

## Requests and state

Voice cloning uses the existing `--task clon --voice-ref ... --reference-text ...`
interface. `--mode streaming` emits audio frames while generating, through the
normal CLI and server streaming interfaces. Long text uses the framework chunker
and the preceding generated turn as context. Speaker tags must be canonical
`<|speaker:0|>` through `<|speaker:15|>`; explicitly tag reference transcripts when
conditioning different speakers. Drama 3 follows the original bracketed reference
template, rather than S2 Pro's automatic reference-index speaker assignment.

Backbone recurrent/KV state and all tower caches reset for each request/chunk.
Projection accumulation is explicitly F32. Vulkan prefill uses the same vector
kernels as decoding: batched half-precision projections/routed matmuls changed
close expert-routing decisions in the reference tests. This policy is local to
the new MoE runtime; the existing dense Qwen3.5 runtime is unchanged.
Vulkan offline decoding uses the existing causal codec one frame at a time to
bound temporary VRAM; it retains attention/convolution history across frames.
This avoids a multi-gigabyte whole-utterance decoder graph beside the 35B model.
S2 Pro and CUDA/CPU offline decoding retain their existing decoder paths.

Tower layers with known codes share a graph; generation passes hidden states
between residual layers on the backend. Teacher-forced prompt rows do not run
unused codebook heads. Vulkan queues up to eight vector prefill rows through
pinned host buffers, preserving causal order and the validated arithmetic.
The text embedding table stays in native host storage because the caller mixes
audio codes on the CPU; host row lookup preserves the backend's existing
conversion policy (native BF16 becomes F16 on Vulkan/Metal).

Vulkan repeats linear-attention heads through one interleaved repeat instead of
per-head concatenations. It can reuse the first 14 prefill rows when their input
embeddings match exactly, retaining the corresponding recurrent/KV state and
hidden values in a private cache (about 64 MiB). This covers the fixed no-reference
instruction/header across different texts. Changed prefixes are recomputed;
speech generation and sampling always run anew. Offline causal codec frames are
queued in groups of eight with the same one-frame arithmetic and history updates.
The existing streaming/CPU/CUDA decode entry points retain their behavior.

Memory saving is opt-in for CLI and server sessions:

```bash
--session-option fish_audio.mem_saver=true
```

The corresponding server model configuration is
`"session_options": {"fish_audio.mem_saver": "true"}`. Drama 3 releases decoder
graphs, recurrent/KV caches, and tower graphs/caches after generation. For
offline requests it releases AR buffers before codec decoding. Weights and the
host embedding table remain loaded. Rebuilding caches increases subsequent
request latency; leave the option off when generation speed is the priority.

For a long-lived, single-session server, save this as `drama3-server.json`
(use an absolute model path when launching from another directory):

```json
{
  "host": "127.0.0.1",
  "port": 8080,
  "backend": "cuda",
  "device": 0,
  "threads": 8,
  "lazy_load": false,
  "models": [{
    "id": "drama3",
    "family": "fish_audio",
    "path": "models/Drama-3-GGUF/drama-3-q4_k.gguf",
    "task": "tts",
    "mode": "offline"
  }]
}
```

```bash
build/bin/audiocpp_server --config drama3-server.json --no-ui
curl http://127.0.0.1:8080/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"model":"drama3","input":"Hello from Drama three.","response_format":"wav","seed":1234,"max_tokens":512}' \
  --output drama3-server.wav
```

For Vulkan, change `backend` to `vulkan` and choose the corresponding device.
The benchmark uses the first request as cold and subsequent serial requests as
warm; keep the same server process loaded. On Windows use `curl.exe` or an HTTP
client that sends the equivalent JSON. Multi-slot execution is not validated.

Sampling follows Drama 3's top-k/top-p and repetition-aware policy, with a native
seeded RNG. Exact PyTorch random draws and identical outputs across backends are
not promised. Tests validate reference mathematics, token formatting, request
repeatability, and real generated speech; they do not establish waveform or
perceptual equivalence to the original FP8 model after Q4 quantization.

## Reference tests

Install `torch`, `numpy`, `safetensors`, `tokenizers`, `loguru`, and `gguf` in a
Python environment. With the pinned Fish Speech checkout and official sidecars:

```bash
python tests/fish_audio/generate_drama3_reference.py \
  --reference-dir /path/to/fish-speech --config-dir models/Drama-3-source \
  --output build/drama3-reference
cmake -S . -B build -DENGINE_BUILD_MODEL_TESTS=ON \
  -DAUDIOCPP_DRAMA3_REFERENCE_DIR=/absolute/path/to/build/drama3-reference
cmake --build build --parallel --target drama3_parity
ctest --test-dir build -R fish_drama3 --output-on-failure
```

The seeded small architecture tests 129-token prefill, grouped/routed/shared MoE,
step-by-step continuation, prefix hit/invalidation and changed-suffix equality,
resets and release/recreation, partial RoPE, tower
history, device handoffs with fixed reference codes, and frame
embeddings against the original classes. Twenty-eight original prompt cases cover
text, Unicode whitespace, tags, references, history, and rejection of structural
tokens, including combining accents, Indic/Thai/Arabic/CJK/Cyrillic text,
numbers, and contractions. Maximum absolute tolerances are `1e-4` CPU and `1e-3` CUDA/Vulkan;
GPU attention follows the framework's F16 cache/flash-attention policy.
See the [validation record](../reports/fish_drama3_validation.md) for measurements
and the limits of the full-checkpoint comparison.

With a converted model available, validate the optional queued codec against
serial frame decoding (including mixed calls and the 128-frame attention window):

```bash
cmake --build build --parallel --target drama3_codec_queue
build/bin/drama3_codec_queue models/Drama-3-GGUF/drama-3-q4_k.gguf vulkan
```
