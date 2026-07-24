# GLM-TTS

GLM-TTS is a zero-shot Chinese and English speech-synthesis model. The native
audio.cpp path executes its Llama speech-token generator, Whisper-VQ reference
encoder, Flow/DiT mel generator, CAMPPlus speaker encoder, and HiFT vocoder.
Both advertised routes are reference-conditioned: provide a clean WAV and the
exact words spoken in it.

| Field | Value |
|---|---|
| Family | `glm_tts` |
| Model directory | `models/GLM-TTS` |
| Task | `tts`, `clon` |
| Modes | `offline` |
| Languages | Chinese, English |
| Voice input | Required reference WAV plus its exact transcript |
| Output | mono 24 kHz WAV |

Install and prepare the official checkpoint:

```bash
python tools/model_manager.py install glm_tts --models-dir models
```

The installer downloads `zai-org/GLM-TTS`, converts the official Flow and HiFT
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
| `--request-option flow_steps=<n>` | integer | `10` | Flow Euler integration steps. |
| `--request-option cfg_rate=<float>` | float | `0.7` | Flow classifier-free guidance rate. |
| `--request-option flow_noise_file=<path>` | raw F32 path | none | Optional exact initial Flow noise for parity tests. |
| `--request-option hift_source_random_file=<path>` | raw F32 path | none | Optional exact HiFT phase-uniform and Gaussian values for parity tests. |
| `--request-option hift_prior_noise_values=<n>` | integer | `0` | Torch RNG offset before normal HiFT source generation. |
| `--session-option glm_tts.weight_type=native|f32|f16|bf16|q8_0` | enum | `native` | Requested component weight storage type. |
| `--session-option glm_tts.mem_saver=true|false` | bool | `false` | Release each large component after use. Enable only when lower resident memory matters more than repeated-request speed. |
| `--session-option glm_tts.reference_cache_slots=<n>` | integer | `1` | Prepared reference-audio cache slots. Reusing a reference skips Whisper-VQ, mel, fbank, and CAMPPlus preparation; `0` disables it. |

The reference transcript must match the audio. A mismatched transcript changes
both semantic and speaker conditioning and can substantially reduce quality.
Q8 generation can select a slightly different speech-token sequence from the
native checkpoint, so waveform identity is not expected.

See [GLM-TTS validation](../reports/glm_tts_validation.md) for exact component
parity, path-test, timing, and output details.
