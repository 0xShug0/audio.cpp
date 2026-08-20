# IndexTTS

IndexTTS is wired as `--family index_tts2` for offline text to speech and voice
cloning. The same model family handles IndexTTS2 and IndexTTS2.5; the runtime
selects the variant from the model config.

## IndexTTS2

IndexTTS2 is a Chinese and English TTS model with voice cloning and expressive
emotion controls. It requires a speaker reference through the framework
`--voice-ref` path.

| Field | Value |
|---|---|
| Family | `index_tts2` |
| Model directory | `models/IndexTTS-2` |
| Task | `tts`, `clon` |
| Modes | `offline` |
| Languages | `zh`, `en` |
| Voice input | Required reference WAV through `--voice-ref` |
| Built-in voices | Not exposed |

Voice clone:

```bash
audiocpp_cli --task clon --family index_tts2 \
  --model /path/to/IndexTTS-2 \
  --backend cuda \
  --language en \
  --text "Hello from IndexTTS2." \
  --voice-ref /path/to/reference.wav \
  --out out.wav
```

Emotion text:

```bash
audiocpp_cli --task tts --family index_tts2 \
  --model /path/to/IndexTTS-2 \
  --backend cuda \
  --language zh \
  --text "今天的演示会更有情绪。" \
  --voice-ref /path/to/reference.wav \
  --emotion "你吓死我了！你是鬼吗？" \
  --request-option emotion_alpha=0.6 \
  --out out.wav
```

### IndexTTS2 Options

| Option | Values | Default | Meaning |
|---|---|---:|---|
| `--voice-ref` | WAV path | required | Reference speaker audio. |
| `--language` | `zh`, `en` | empty | Text language label. |
| `--emotion` | text | not set | Emotion-text conditioning through the framework style field. |
| `--request-option emotion_alpha=<float>` | float in `[0, 1]` | `1.0` | Blend strength for explicit emotion conditioning. |
| `--request-option emotion_vector=<v0,...,v7>` | 8 floats | not set | Explicit emotion vector. |
| `--request-option use_emotion_text=true\|false` | bool | `false` | Infer emotion from text. |
| `--request-option use_random_emotion=true\|false` | bool | `false` | Use random emotion weights in the emotion mixer. |
| `--request-option interval_silence_ms=<n>` | milliseconds | `200` | Silence inserted between generated text chunks. |
| `--request-option duration_factor=<float>` | positive float | `1.0` | Output duration multiplier for speech-rate control; `>1` slower, `<1` faster. Matches the official IndexTTS2.5 `duration_factor`; also accepted for the v2 variant. |
| `--text-chunk-size` | characters | not set | Optional framework outer text chunk size. When omitted, IndexTTS2 keeps its internal tokenizer segmentation. |
| `--text-chunk-mode` | `default`, `tag_aware`, `japanese`, `endline` | `default` | Framework chunking mode used only when `--text-chunk-size` is set. |
| `--max-tokens` | integer | model default | Maximum generated GPT mel tokens. |
| `--temperature` | float | model default | GPT sampling temperature. |
| `--top-p` | float | model default | GPT nucleus sampling limit. |
| `--top-k` | integer | model default | GPT top-k sampling limit. |
| `--repetition-penalty` | float | model default | GPT repetition penalty. |
| `--do-sample` | `true`, `false` | model default | Enable stochastic GPT sampling. |
| `--session-option index_tts2.mem_saver=true\|false` | bool | `false` | Release staged reference and conditioning graphs after request phases. |
| `--session-option index_tts2.weight_type=native\|f32\|f16\|bf16\|q8_0` | enum | `native` | Matmul weight storage type. |
| `--session-option index_tts2.conv_weight_type=native\|f32\|f16` | enum | `native` | Convolution weight storage type. |
| `--session-option index_tts2.speaker_cache_slots=<n>` | integer slots | `1` | Prepared speaker-reference cache slots; set `0` to disable reuse. |
| `--session-option index_tts2.emotion_cache_slots=<n>` | integer slots | `1` | Prepared emotion-reference cache slots; set `0` to disable reuse. |
| `--session-option index_tts2.emotion_text_cache_slots=<n>` | integer slots | `1` | Emotion-text weight cache slots; set `0` to disable reuse. |
| `--session-option index_tts2.gpt_graph_arena_mb=<n>` | MB | model default | GPT graph arena size. |
| `--session-option index_tts2.s2mel_graph_arena_mb=<n>` | MB | model default | S2Mel graph arena size. |
| `--session-option index_tts2.reference_graph_arena_mb=<n>` | MB | model default | Reference encoder and codec graph arena size. |
| `--session-option index_tts2.emotion_text_prefill_graph_arena_mb=<n>` | MB | model default | Emotion-text prefill graph arena size. |
| `--session-option index_tts2.emotion_text_decode_graph_arena_mb=<n>` | MB | model default | Emotion-text cached-step graph arena size. |
| `--session-option index_tts2.emotion_text_max_tokens=<n>` | tokens | `256` | Maximum generated tokens for emotion-text classification; old name `index_tts2.emotion_text_max_new_tokens` is still accepted. |
| `--session-option index_tts2.weight_context_mb=<n>` | MB | `32` | Shared ggml weight metadata context size. |

Text normalization note: audio.cpp reimplements the official IndexTTS text
front-end (wetext-style number/date/symbol rules, pinyin-tone and name
protection) as lightweight C++ rules instead of running the official OpenFst
grammars. Alignment is tracked by a golden-corpus parity test
(`index_tts_tn_corpus_test`, which replays 50+ tricky cases against outputs
generated by the official Python normalizer), but a rule-based port cannot be
exhaustive: for unusual inputs (letter-attached digits, rare date/unit formats,
emails, URLs) the normalized text can still differ from the official pipeline,
so the same prompt may be read differently. This is a text-frontend difference,
not a model-quality problem; the GPT/codec/S2Mel stack itself matches the
official pipeline token-for-token on the validated cases.

## IndexTTS2.5

IndexTTS2.5 is IndexTeam/bilibili's multilingual zero-shot TTS model (released
2026-07): a 0.8B GPT autoregressive model + DiT CFM + BigVGAN stack that keeps
IndexTTS2's timbre-emotion decoupling and adds Japanese, Spanish, and Arabic on
top of Chinese and English. It requires a speaker reference through the
framework `--voice-ref` path. Inline `<文字|发音>` pronunciation overrides
(pinyin, CMU phonemes, or kana) are supported. Upstream weights live at
[IndexTeam/IndexTTS-2.5](https://huggingface.co/IndexTeam/IndexTTS-2.5); the
reference implementation is [index-tts/index-tts](https://github.com/index-tts/index-tts)
branch `indextts-2.5`.

IndexTTS2.5 is implemented as a variant of the `index_tts2` family rather than
a separate family: both variants share the audio features, wav2vec2bert, Qwen
emotion, style encoder, BigVGAN vocoder, S2Mel, and GPT decode/cache code. The
tokenizer, GPT speaker conditioning, and semantic-codec decode path are selected
per variant from the model config `version` field (`"2.5"`). All
`index_tts2.*` session options apply to both variants.

| Field | Value |
|---|---|
| Family | `index_tts2` |
| Variant | `2.5`, selected from the model config `version` field |
| Model directory | `models/IndexTTS2.5-GGUF` |
| Default package | `index_tts2_5_q8_0` |
| Other packages | `index_tts2_5_f16`, `index_tts2_5_orig` |
| Task | `tts`, `clon` |
| Modes | `offline` |
| Languages | `zh`, `en`, `ja`, `es`, `ar` |
| Voice input | Required reference WAV through `--voice-ref` |
| Built-in voices | Not exposed |

Voice clone:

```bash
audiocpp_cli --task clon --family index_tts2 \
  --model /path/to/IndexTTS2.5-GGUF \
  --backend cuda \
  --text "Hello from IndexTTS2.5." \
  --voice-ref /path/to/reference.wav \
  --out out.wav
```

Emotion text:

```bash
audiocpp_cli --task tts --family index_tts2 \
  --model /path/to/IndexTTS2.5-GGUF \
  --backend cuda \
  --text "今天的演示会更有情绪。" \
  --voice-ref /path/to/reference.wav \
  --emotion "你吓死我了！你是鬼吗？" \
  --request-option emotion_alpha=0.6 \
  --out out.wav
```

The `language` request option selects the text language (`auto`, `zh`, `en`,
`ja`, `es`, `ar`, or any tokenizer language code). The default `auto` picks
`zh` when the text contains Han characters and `en` otherwise, so mixed
Japanese, Spanish, or Arabic text should set
`--request-option language=ja|es|ar` explicitly.

Emotion conditioning supports all three IndexTTS2 paths: an emotion reference
WAV through `--audio`, an explicit `emotion_vector`, and Qwen-based emotion-text
classification through `--emotion` / `use_emotion_text`. See the IndexTTS2
text-normalization note above for the zh/en caveat. Known limitations: the
Spanish NeMo text normalizer is not ported, so es text passes through without
upstream-style normalization. The official ja path has no NeMo grammar, so NeMo
TN is a no-op for Japanese upstream; the official ja wakachi/fugashi word
segmentation is also not ported, so ja text is tokenized unsegmented. It is
intelligible, but not token-identical to the official pipeline.

License: IndexTTS-2.5 weights are distributed under the bilibili Model Use
License, which is not OSI-approved. It requires separate commercial
authorization when monthly active users exceed 100 million or annual revenue
exceeds 1 billion RMB, and it forbids using model outputs to improve other AI
models. Check the upstream repository for the full terms before redistribution
or commercial use.

### IndexTTS2.5 Options

| Option | Values | Default | Meaning |
|---|---|---:|---|
| `--voice-ref` | WAV path | required | Reference speaker audio. |
| `--request-option language=<code>` | `auto`, `zh`, `en`, `ja`, `es`, `ar`, ... | `auto` | Text language hint; `auto` infers `zh` when the text contains Han characters, otherwise `en`. |
| `--emotion` | text | not set | Emotion-text conditioning through the framework style field. |
| `--request-option emotion_alpha=<float>` | float in `[0, 1]` | `1.0` | Blend strength for explicit emotion conditioning. |
| `--request-option emotion_vector=<v0,...,v7>` | 8 floats | not set | Explicit emotion vector. |
| `--request-option use_emotion_text=true\|false` | bool | `false` | Infer emotion from text. |
| `--request-option use_random_emotion=true\|false` | bool | `false` | Use random emotion weights in the emotion mixer. |
| `--request-option interval_silence_ms=<n>` | milliseconds | `200` | Silence inserted between generated text chunks. |
| `--request-option duration_factor=<float>` | positive float | `1.0` | Output duration multiplier for speech-rate control; `>1` slower, `<1` faster. Matches the official IndexTTS2.5 `duration_factor`. |
| `--text-chunk-size` | characters | not set | Optional framework outer text chunk size. When omitted, IndexTTS2.5 keeps its internal tokenizer segmentation. |
| `--text-chunk-mode` | `default`, `tag_aware`, `japanese`, `endline` | `default` | Framework chunking mode used only when `--text-chunk-size` is set. |
| `--max-tokens` | integer | `1500` | Maximum generated GPT mel tokens. |
| `--temperature` | float | `0.8` | GPT sampling temperature. |
| `--top-p` | float | `0.8` | GPT nucleus sampling limit. |
| `--top-k` | integer | `30` | GPT top-k sampling limit. |
| `--repetition-penalty` | float | `10.0` | GPT repetition penalty. |
| `--do-sample` | `true`, `false` | `true` | Enable stochastic GPT sampling. |
| `--request-option length_penalty=<float>` | float | `0.0` | GPT beam-search length penalty. |
| `--request-option num_beams=<n>` | integer | `3` | GPT beam count. |
| `--session-option index_tts2.mem_saver=true\|false` | bool | `false` | Release staged reference and conditioning graphs after request phases. |
| `--session-option index_tts2.weight_type=native\|f32\|f16\|bf16\|q8_0` | enum | `native` | Matmul weight storage type. |
| `--session-option index_tts2.conv_weight_type=native\|f32\|f16` | enum | `native` | Convolution weight storage type. |
| `--session-option index_tts2.speaker_cache_slots=<n>` | integer slots | `1` | Prepared speaker-reference cache slots; set `0` to disable reuse. |
| `--session-option index_tts2.emotion_cache_slots=<n>` | integer slots | `1` | Prepared emotion-reference cache slots; set `0` to disable reuse. |
| `--session-option index_tts2.emotion_text_cache_slots=<n>` | integer slots | `1` | Emotion-text weight cache slots; set `0` to disable reuse. |
| `--session-option index_tts2.gpt_graph_arena_mb=<n>` | MB | model default | GPT graph arena size. |
| `--session-option index_tts2.s2mel_graph_arena_mb=<n>` | MB | model default | S2Mel graph arena size. |
| `--session-option index_tts2.reference_graph_arena_mb=<n>` | MB | model default | Reference encoder and codec graph arena size. |
| `--session-option index_tts2.emotion_text_prefill_graph_arena_mb=<n>` | MB | model default | Emotion-text prefill graph arena size. |
| `--session-option index_tts2.emotion_text_decode_graph_arena_mb=<n>` | MB | model default | Emotion-text cached-step graph arena size. |
| `--session-option index_tts2.emotion_text_max_tokens=<n>` | tokens | `256` | Maximum generated tokens for emotion-text classification; old name `index_tts2.emotion_text_max_new_tokens` is still accepted. |
| `--session-option index_tts2.weight_context_mb=<n>` | MB | `32` | Shared ggml weight metadata context size. |

## Converting IndexTTS2.5 From Upstream Weights

`tools/convert_index_tts2_5.py` turns an official `IndexTeam/IndexTTS-2.5`
snapshot (the `.pth` checkpoints) into the Safetensors staging layout the
engine expects, and prints or runs the matching `audiocpp_gguf` command. The
w2v-bert-2.0, CAMPPlus, and BigVGAN checkpoints are auto-detected under
`<model-dir>/hf_cache/` after running official inference once to populate it,
and each has an explicit override flag:

```bash
python tools/convert_index_tts2_5.py \
    --model-dir /path/to/IndexTTS-2.5 \
    --output-dir /path/to/staging \
    --run-converter /path/to/audiocpp_gguf --type q8_0
```

Pass `--native-dir /path/to/IndexTTS-2.5-native` to also emit a directly
loadable native Safetensors model directory hardlinked from the staging files,
without GGUF conversion.

The script repackages the checkpoints the loader needs, unwraps the
`s2mel.pth`/`codec.pth` container keys, prefixes CAMPPlus tensors with
`speaker_encoder.`, strips BigVGAN's `generator.` prefix, wraps the
`feat1/feat2.pt` matrices as a single `tensor`, and assembles the sidecar
`root/` config/tokenizer resources that get embedded into the GGUF. The staged
`config.yaml` has its `version` field normalized to `"2.5"` because the
official snapshot ships `version: 2.0`; the engine uses that field to select the
IndexTTS2 family variant.
