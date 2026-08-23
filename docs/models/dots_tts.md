# DotTTS

DotTTS is an experimental multilingual TTS, voice-cloning, and speech-editing
family with SOAR, MeanFlow, and Edit packages. The default download is the
standalone SOAR Q8 GGUF.

```bash
python3 tools/model_manager_v2.py install dots_tts_soar_q8_0
```

Use prompt audio plus `reference_text` for the best clone quality:

```bash
audiocpp_cli --task tts --family dots_tts \
  --model models/DotTTS-SOAR-GGUF/dots-tts-soar-q8_0.gguf \
  --backend cuda \
  --text "Our field team finished the morning inspection and prepared a concise update." \
  --voice-ref assets/resources/a.wav \
  --reference-text "This little work was finished in the year eighteen o three, and intended for immediate publication." \
  --request-option reference_duration_sec=5 \
  --out out.wav
```

MeanFlow uses the same runtime and request options with the MF package:

```bash
python3 tools/model_manager_v2.py install dots_tts_mf_q8_0

audiocpp_cli --task tts --family dots_tts \
  --model models/DotTTS-MF-GGUF/dots-tts-mf-q8_0.gguf \
  --backend cuda \
  --text "The release candidate passed the local checks, so the team prepared baseline audio." \
  --voice-ref assets/resources/b.wav \
  --reference-text "Some call me nature. Others call me Mother Nature." \
  --request-option reference_duration_sec=5 \
  --out mf.wav
```

DotTTS Edit uses a separate Edit GGUF package and `template_name=edit`. It
takes source audio plus a structured edit instruction. When `source_text` and
`target_text` are omitted, the runtime derives them from the tagged instruction.

```bash
python3 tools/model_manager_v2.py install dots_tts_edit_q8_0

audiocpp_cli --task tts --family dots_tts \
  --model models/DotTTS-Edit-GGUF/dots-tts-edit-q8_0.gguf \
  --backend cuda \
  --text 'Hello <sub targ="brave">small</sub> world.' \
  --request-option source_audio=source.wav \
  --request-option template_name=edit \
  --out edited.wav
```

Use `source_text` and `target_text` to override the derived transcripts.
`use_xvector=auto` follows the upstream behavior: speaker guidance is disabled
for pure `emo`, `bg`, or `enhance` edits and enabled for text, pitch, rate,
pause, speaker-transfer, or mixed edits.

Supported structural tags include `<del>`, `<ins>`, `<sub targ="replacement">`,
`<emo>`, `<pitch>`, `<rate>`, `<enhance>`, `<bg>`, `<pause/>`, and
`<spk_transfer/>`.

Streaming mode emits generated audio chunks and a final merged WAV:

```bash
audiocpp_cli --task tts --mode streaming --family dots_tts \
  --model models/DotTTS-SOAR-GGUF/dots-tts-soar-q8_0.gguf \
  --backend cuda \
  --text "During the workshop, Maya explained the prototype clearly while the group compared audio quality and latency." \
  --voice-ref assets/resources/b.wav \
  --reference-text "Some call me nature. Others call me Mother Nature." \
  --request-option reference_duration_sec=5 \
  --out stream.wav \
  --out-dir stream_chunks
```

DotTTS exposes several synthesis templates:

```bash
audiocpp_cli --task tts --family dots_tts \
  --model models/DotTTS-SOAR-GGUF/dots-tts-soar-q8_0.gguf \
  --backend cuda \
  --text "The museum guide welcomed visitors and invited questions." \
  --voice-ref assets/resources/a.wav \
  --reference-text "This little work was finished in the year eighteen o three, and intended for immediate publication." \
  --request-option template_name=instruction_tts \
  --out instruction.wav
```

For long-form text, DotTTS uses the framework chunker. The default
`tag_aware` mode preserves leading style or language tags across chunks.

```bash
audiocpp_cli --task tts --family dots_tts \
  --model models/DotTTS-SOAR-GGUF/dots-tts-soar-q8_0.gguf \
  --backend cuda \
  --text "This longer request checks that several generated chunks keep the same reference voice while the final audio remains clear." \
  --voice-ref assets/resources/b.wav \
  --reference-text "Some call me nature. Others call me Mother Nature." \
  --request-option text_chunk_size=160 \
  --request-option text_chunk_mode=tag_aware \
  --out longform.wav
```

| Option | Values | Default | Meaning |
|---|---|---:|---|
| `--reference-text` / `--request-option reference_text=<text>` | text | empty | Transcript for prompt audio. |
| `--request-option reference_duration_sec=<f>` | seconds | not set | Trim prompt audio before reference conditioning. |
| `--request-option template_name=<name>` | `tts`, `instruction_tts`, `text_to_audio`, `tts_interleave`, `edit` | `tts` | Synthesis template. Use `edit` with the DotTTS Edit package. |
| `--request-option source_audio=<path>` | WAV path | not set | Source audio for `template_name=edit`. |
| `--request-option instruction=<text>` / `--request-option instruct=<text>` | text | request text | Structured DotTTS Edit instruction. |
| `--request-option source_text=<text>` | text | derived | Optional source transcript override for DotTTS Edit. |
| `--request-option target_text=<text>` | text | derived | Optional target transcript override for DotTTS Edit. |
| `--request-option use_xvector=<mode>` | `auto`, `on`, `off` | `auto` | Speaker-guidance control for DotTTS Edit. |
| `--language` / `--request-option language=<code>` | language code or `none` | `none` | Optional language tag, such as `en` or `zh`. |
| `--request-option num_inference_steps=<n>` | integer | `10` | Flow-matching inference steps. |
| `--guidance-scale` / `--request-option guidance_scale=<f>` | float | `1.2` | Classifier-free guidance scale. |
| `--request-option speaker_scale=<f>` | float | `1.5` | Prompt speaker embedding scale. |
| `--request-option sampler_mode=<name>` | `euler`, `midpoint`, `rk4` | `euler` | Flow sampler mode. |
| `--max-tokens` / `--request-option max_tokens=<n>` | integer | `500` | Maximum generated audio patch count per segment. |
| `--text-chunk-size` / `--request-option text_chunk_size=<n>` | chars | `320` | Long-form chunk size. |
| `--text-chunk-mode` / `--request-option text_chunk_mode=<name>` | `default`, `tag_aware`, `japanese`, `endline` | `tag_aware` | Framework chunking mode. |
| `--request-option vocoder_merge_steps=<n>` | integer | `4` | Streaming vocoder latent merge size. |
| `--seed` / `--request-option seed=<n>` | integer | `42` | Prompt latent and flow noise seed. |
| `--session-option dots_tts.weight_type=<type>` | `native`, `f32`, `f16`, `bf16`, `q8_0` | `native` | Shared matmul weight storage type. |
| `--session-option dots_tts.speaker_encoder_weight_type=<type>` | `native`, `f32`, `f16`, `bf16`, `q8_0` | `weight_type` | Speaker encoder matmul weight storage type. |
| `--session-option dots_tts.codec_weight_type=<type>` | `native`, `f32`, `f16`, `bf16`, `q8_0` | `weight_type` | AudioVAE matmul and recurrent weight storage type. |
| `--session-option dots_tts.patch_encoder_weight_type=<type>` | `native`, `f32`, `f16`, `bf16`, `q8_0` | `weight_type` | Patch encoder matmul weight storage type. |
| `--session-option dots_tts.llm_weight_type=<type>` | `native`, `f32`, `f16`, `bf16`, `q8_0` | `weight_type` | LLM matmul weight storage type. |
| `--session-option dots_tts.flow_weight_type=<type>` | `native`, `f32`, `f16`, `bf16`, `q8_0` | `weight_type` | SOAR or MeanFlow DiT matmul weight storage type. |
| `--session-option dots_tts.codec_conv_weight_type=<type>` | `native`, `f32`, `f16` | `native` | AudioVAE convolution weight storage type. |
| `--session-option dots_tts.reference_cache_slots=<n>` | integer | `4` | Prepared reference-audio cache slots; use `0` to disable reuse. |
| `--session-option dots_tts.mem_saver=true\|false` | bool | `false` | Release request-phase components while keeping reference cache slots alive. |
