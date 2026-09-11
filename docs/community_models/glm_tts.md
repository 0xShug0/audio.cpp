# GLM-TTS

GLM-TTS is a zero-shot Chinese and English speech-synthesis model. The native
audio.cpp path executes its Llama speech-token generator, Whisper-VQ reference
encoder, Flow/DiT mel generator, CAMPPlus speaker encoder, and HiFT vocoder.
Both advertised routes are reference-conditioned: provide a clean WAV and the
exact words spoken in it.

GLM-TTS is v1-native. `model_specs/glm_tts.json` is the single source of truth
for metadata, packages, normalized options, and GGUF/safetensors resources.
The runtime uses the generic spec-backed loader; the legacy request aliases
listed below remain an internal compatibility layer for existing users.

| Field | Value |
|---|---|
| Family | `glm_tts` |
| Model directory | `models/GLM-TTS` |
| Task | `tts`, `clon` |
| Modes | `offline` |
| Languages | Chinese, English |
| Voice input | Required reference WAV plus its exact transcript |
| Output | mono 24 kHz WAV |

Install the default package:

```bash
python tools/model_manager_v2.py install glm_tts --models-root models
```

The default install is the standalone Q8 GGUF package.

The original safetensors preparation path remains available through the
deprecated manager:

```bash
python tools/model_manager_deprecated.py install glm_tts --models-dir models
```

That installer downloads `zai-org/GLM-TTS`, converts the official Flow and HiFT
PyTorch checkpoints to safetensors, prepares the ChatGLM tokenizer resources,
and installs the matching CAMPPlus safetensors weights. The latter are sourced
from `mlx-community/index-tts2-mlx` because the GLM-TTS repository publishes
CAMPPlus as ONNX only; the native output was checked directly against that
official ONNX graph.

Run the prepared safetensors package:

```bash
audiocpp_cli --task clon --family glm_tts \
  --model models/GLM-TTS --backend cuda \
  --voice-ref reference.wav \
  --reference-text "The exact words spoken in reference.wav." \
  --text "Hello from GLM-TTS." \
  --top-k 25 --top-p 0.8 --temperature 1.0 \
  --seed 0 --max-tokens 256 --out glm_tts.wav
```

The `tts` task accepts the same reference arguments. It is an alias for the
same zero-shot synthesis path rather than an unconditioned preset voice.

## Standalone GGUF

After installing the package, create a standalone mixed GGUF. The
autoregressive Llama group remains F16 for speech-token quality; the speech
tokenizer, Flow, HiFT, and CAMPPlus groups use Q8_0:

```bash
audiocpp_gguf \
  --input llama_weights=models/GLM-TTS/llm/model.safetensors.index.json \
  --input speech_tokenizer_weights=models/GLM-TTS/speech_tokenizer/model.safetensors \
  --input flow_weights=models/GLM-TTS/flow/model.safetensors \
  --input hift_weights=models/GLM-TTS/hift/model.safetensors \
  --input campplus_weights=models/GLM-TTS/frontend/campplus.safetensors \
  --root models/GLM-TTS \
  --family glm_tts \
  --model-spec model_specs/glm_tts.json \
  --type q8_0 \
  --keep-type "llama_weights/*=f16" \
  --overwrite \
  --output models/GLM-TTS-Q8/GLM-TTS_Q8.gguf
```

The resulting file embeds all five tensor groups, the package specification,
configs, and tokenizer sidecars. It runs from a directory containing only the
GGUF:

```bash
audiocpp_cli --task clon --family glm_tts \
  --model models/GLM-TTS-Q8/GLM-TTS_Q8.gguf --backend cuda \
  --voice-ref reference.wav \
  --reference-text "The exact words spoken in reference.wav." \
  --text "Hello from GLM-TTS." \
  --seed 0 --max-tokens 256 --out glm_tts_q8.wav
```

## Controls

| Option | Values | Default | Meaning |
|---|---|---:|---|
| `--reference-text` | text | required | Exact transcript of `--voice-ref`. |
| `--max-tokens` | integer | automatic | Maximum generated speech-token count. |
| `--temperature` | float | `1.0` | Speech-token sampling temperature. |
| `--top-k` | integer | `25` | Speech-token top-k limit. |
| `--top-p` | float | `0.8` | Speech-token nucleus threshold. |
| `--seed` | integer | `0` | Seed used by token sampling, Flow noise, and HiFT. |
| `--request-option num_inference_steps=<n>` | integer | `10` | Flow Euler integration steps. |
| `--request-option flow_guidance_scale=<float>` | float | `0.7` | Flow classifier-free guidance rate. |
| `--request-option flow_noise_path=<path>` | raw F32 path | none | Optional exact initial Flow noise for parity tests. |
| `--request-option hift_source_random_path=<path>` | raw F32 path | none | Optional exact HiFT phase-uniform and Gaussian values for parity tests. |
| `--request-option hift_prior_noise_count=<n>` | integer | `0` | Torch RNG offset before normal HiFT source generation. |

The legacy GLM-TTS request keys `flow_steps`, `cfg_rate`, `flow_noise_file`,
`hift_source_random_file`, and `hift_prior_noise_values` remain accepted for
backward compatibility.
| `--session-option glm_tts.weight_type=native\|f32\|f16\|bf16\|q8_0` | enum | `native` | Requested component weight storage type. |
| `--session-option glm_tts.mem_saver=true\|false` | bool | `false` | Release the reference-only Whisper-VQ and CAMPPlus runtimes after caching the voice, while keeping Llama, Flow, and HiFT warm. |
| `--session-option glm_tts.aggressive_mem_saver=true\|false` | bool | `false` | Also release Llama, Flow, and HiFT after each stage. This minimizes VRAM but reloads the generation path on every request. |
| `--session-option glm_tts.reference_cache_slots=<n>` | integer | `1` | Prepared reference-audio cache slots. Reusing a reference skips Whisper-VQ, mel, fbank, and CAMPPlus preparation; `0` disables it. |

Balanced mem-saver is intended for a server repeatedly using a cached
reference voice. On a reference-cache miss it first releases the warm
generation path, prepares and caches the new voice, then reconstructs the
generation path. This prevents the reference and generation weight groups
from overlapping in VRAM. Set `glm_tts.aggressive_mem_saver=true` only when
the lowest possible peak VRAM is more important than request latency; it
implies balanced mem-saver even when `glm_tts.mem_saver` is omitted.

The reference transcript must match the audio. A mismatched transcript changes
both semantic and speaker conditioning and can substantially reduce quality.
Q8 generation can select a slightly different speech-token sequence from the
native checkpoint, so waveform identity is not expected.

See [GLM-TTS validation](../reports/glm_tts_validation.md) for exact component
parity, path-test, timing, and output details.
