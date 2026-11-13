# Ming 3.0 Audio Tiny

Ming 3.0 Audio combines a Whisper audio encoder, a hybrid KDA/MLA
mixture-of-experts Thinker, and an independent continuous-latent Talker with
an audio VAE. The Thinker produces text; the Talker synthesizes that text.

Upstream checkpoint: [inclusionAI/Ming-3.0-audio-tiny](https://huggingface.co/inclusionAI/Ming-3.0-audio-tiny/tree/25b96e30be3739849bf76c0f48ac11db4a262fb3).
Code: [inclusionAI/Ming-3.0-audio-flash](https://github.com/inclusionAI/Ming-3.0-audio-flash).
GGUF: [audio-cpp/Ming-3.0-Audio-Tiny-GGUF](https://huggingface.co/audio-cpp/Ming-3.0-Audio-Tiny-GGUF).

The standalone GGUF includes the Thinker, Talker, audio VAE, CAMPPlus speaker
encoder, tokenizers, and configuration. No separate Python runtime or codec
download is required. Speech output is mono 44.1 kHz. Only offline sessions
are exposed.

## Usage

### Transcription And Audio Understanding

```bash
audiocpp_cli --family ming3_audio --task asr --backend cuda \
  --model /path/to/ming-3.0-audio-tiny-bf16.gguf \
  --audio recording.wav --log
```

The default prompt returns a language label followed by a tab and the transcript.
Use `--request-option prompt="Describe the sounds in this recording."` for
audio understanding instead of transcription.

### Named Voice

```bash
audiocpp_cli --family ming3_audio --task tts --backend cuda \
  --model /path/to/ming-3.0-audio-tiny-bf16.gguf \
  --text "The sun warms the earth, and a soft breeze moves through the garden." \
  --request-option voice_id=US_James --request-option language=en \
  --out speech.wav --log
```

### Reference Voice

```bash
audiocpp_cli --family ming3_audio --task clon --backend cuda \
  --model /path/to/ming-3.0-audio-tiny-bf16.gguf \
  --voice-ref reference.wav --text "Please leave the lamp beside the window." \
  --request-option language=en --out clone.wav --log
```

Reference audio supplies speaker identity. Adding
`--request-option reference_text="Words spoken in the reference."` also supplies
an acoustic prefix. Named voices are used only when reference audio is absent.
Both `tts` and `clon` accept reference audio.

Prefer a reference in the target language: cross-language cloned pronunciation
can drift even when the generated text is correct.

### Spoken Response

Use `--task s2s --audio question.wav --out response.wav` with the same model
and backend. The Thinker answers the question in the recording, and the Talker
speaks its answer. `prompt` overrides the Thinker instruction; `voice_id` or
`--voice-ref` controls the response voice.

## Model

| Field | Value |
|---|---|
| Family | `ming3_audio` |
| Tasks | `asr`, `s2s`, `tts`, `clon` |
| Modes | `offline` |
| Languages | `zh`, `en` |
| Default package | `ming3_audio_tiny_bf16` |
| Input | Text and/or audio, depending on the selected task |
| Output | Transcript/response text or speech audio, depending on the task |

## Common Options (use directly)

| Option | Meaning |
|---|---|
| `--model` | Model file or directory shown in the examples. |
| `--backend` | Select an available backend supported by this model. |
| `--threads` | CPU thread count. |
| `--log` | Enable runtime logging. |
| `--text` | Input text; see the examples for the expected prompt format. |
| `--audio` | Input audio, where required by the selected task. |
| `--voice-ref` | Reference audio for voice cloning. |
| `--out` | Output WAV file. |

## Request Options (use with `--request-option`)

| Option | Type / values | Default | Description |
|---|---|---|---|
| `system_prompt` | `string` | Empty | Thinker system instruction. When supplied, the user prompt precedes audio as in the official chat examples. |
| `enable_thinking` | `bool` | `false` | Enable the Thinker chat template's detailed thinking mode. |
| `prompt` | `string` | Task-dependent | Thinker instruction accompanying the input audio; omitted ASR requests use the official transcription prompt. |
| `max_tokens` | `int` | `256` | Maximum generated Thinker text tokens. Declared range: minimum 1. |
| `max_audio_tokens` | `int` | `1000` | Maximum generated Talker acoustic patches per text fragment. Declared range: minimum 1. |
| `voice_id` | `US_James`, `US_Stella`, `灵小柔`, `灵楷`, `US_Emma`, `UK_George`, `UK_Charlotte`, `灵小安`, `灵语汐`, `灵小萌`, `灵小雅`, `花小呗`, `灵韵`, `妩媚妲己`, `灵婉`, `灵小月`, `灵朵儿`, `灵岩`, `灵小锋`, `灵小川`, `灵小甄`, `灵小知`, `灵小惠`, `灵小晴`, `灵小琪`, `灵梦`, `灵恬`, `灵小汐`, `灵小棠`, `灵小梨`, `灵舒`, `灵娇`, `灵小芸`, `灵锐`, `灵空`, `灵宇`, `灵朗`, `灵旭`, `灵骁`, `灵江`, `悲伤女声`, `皮六`, `说唱小哥`, `蜡笔小新`, `悟空`, `灵信远_生气`, `灵信远_厌恶`, `灵信远_惊讶`, `灵信远_高兴`, `灵信远_恐惧`, `灵信远_中性`, `灵信远_悲伤`, `灵思齐_高兴`, `灵思齐_恐惧`, `灵思齐_生气`, `灵思齐_厌恶`, `灵思齐_中性`, `灵思齐_惊讶`, `灵思齐_悲伤`, `灵叙柔_高兴`, `灵叙柔_恐惧`, `灵叙柔_惊讶`, `灵叙柔_悲伤`, `灵叙柔_生气`, `灵叙柔_中性`, `灵叙柔_厌恶`, `灵绮木_悲伤`, `灵绮木_厌恶`, `灵绮木_中性`, `灵绮木_恐惧`, `灵绮木_惊讶`, `灵绮木_高兴`, `灵绮木_生气`, `灵书悦_恐惧`, `灵书悦_厌恶`, `灵书悦_高兴`, `灵书悦_惊讶`, `灵书悦_悲伤`, `灵书悦_生气`, `灵书悦_中性`, `灵渊霆_厌恶`, `灵渊霆_中性`, `灵渊霆_恐惧`, `灵渊霆_惊讶`, `灵渊霆_悲伤`, `灵渊霆_生气`, `灵渊霆_高兴`, `灵素心_高兴`, `灵素心_惊讶`, `灵素心_恐惧`, `灵素心_中性`, `灵素心_生气`, `灵素心_悲伤`, `灵素心_厌恶`, `灵晓声_惊讶`, `灵晓声_厌恶`, `灵晓声_高兴`, `灵晓声_生气`, `灵晓声_中性`, `灵晓声_悲伤`, `灵风信_生气`, `灵风信_中性`, `灵风信_厌恶`, `灵风信_高兴`, `灵风信_惊讶`, `灵风信_悲伤`, `灵风信_恐惧`, `灵采薇_惊讶`, `灵采薇_厌恶`, `灵采薇_恐惧`, `灵采薇_生气`, `灵采薇_悲伤`, `灵采薇_中性`, `灵采薇_高兴`, `灵卓然_厌恶`, `灵卓然_中性`, `灵卓然_恐惧`, `灵卓然_惊讶`, `灵卓然_高兴`, `灵卓然_悲伤`, `灵卓然_生气`, `灵惜梦_中性`, `灵惜梦_厌恶`, `灵惜梦_高兴`, `灵惜梦_悲伤`, `灵惜梦_恐惧`, `灵惜梦_惊讶`, `灵惜梦_生气`, `灵浅忧_高兴`, `灵浅忧_生气`, `灵浅忧_厌恶`, `灵浅忧_悲伤`, `灵浅忧_惊讶`, `灵浅忧_恐惧`, `灵浅忧_中性`, `灵昭野_生气`, `灵昭野_高兴`, `灵昭野_悲伤`, `灵昭野_中性`, `灵昭野_厌恶`, `灵昭野_恐惧`, `灵昭野_惊讶`, `灵心澜_厌恶`, `灵心澜_惊讶`, `灵心澜_生气`, `灵心澜_中性`, `灵心澜_高兴`, `灵心澜_悲伤`, `灵心澜_恐惧`, `灵犀言_悲伤`, `灵犀言_厌恶`, `灵犀言_恐惧`, `灵犀言_中性`, `灵犀言_惊讶`, `灵犀言_高兴`, `灵犀言_生气`, `灵语冰_悲伤`, `灵语冰_厌恶`, `灵语冰_高兴`, `灵语冰_恐惧`, `灵语冰_惊讶`, `灵语冰_中性`, `灵语冰_生气`, `灵逸风_生气`, `灵逸风_悲伤`, `灵逸风_惊讶`, `灵逸风_中性`, `灵逸风_恐惧`, `灵逸风_高兴`, `灵逸风_厌恶`, `灵悦澜_中性`, `灵悦澜_高兴`, `灵悦澜_厌恶`, `灵悦澜_恐惧`, `灵悦澜_生气`, `灵悦澜_悲伤`, `灵悦澜_惊讶`, `灵惊鸿_厌恶`, `灵惊鸿_高兴`, `灵惊鸿_中性`, `灵惊鸿_惊讶`, `灵惊鸿_恐惧`, `灵惊鸿_生气`, `灵惊鸿_悲伤`, `灵御心_高兴`, `灵御心_中性`, `灵御心_生气`, `灵御心_恐惧`, `灵御心_厌恶`, `灵御心_惊讶`, `灵御心_悲伤`, `灵语恬_厌恶`, `灵语恬_高兴`, `灵语恬_恐惧`, `灵语恬_悲伤`, `灵语恬_惊讶`, `灵语恬_生气`, `灵语恬_中性`, `灵心恬_厌恶`, `灵心恬_惊讶`, `灵心恬_悲伤`, `灵心恬_中性`, `灵心恬_恐惧`, `灵心恬_高兴`, `灵心恬_生气`, `灵惜蕊_惊讶`, `灵惜蕊_厌恶`, `灵惜蕊_恐惧`, `灵惜蕊_高兴`, `灵惜蕊_中性`, `灵惜蕊_悲伤`, `灵惜蕊_生气`, `灵语萱_厌恶`, `灵语萱_高兴`, `灵语萱_惊讶`, `灵语萱_恐惧`, `灵语萱_生气`, `灵语萱_悲伤`, `灵语萱_中性`, `灵语烟_生气`, `灵语烟_恐惧`, `灵语烟_中性`, `灵语烟_厌恶`, `灵语烟_惊讶`, `灵语烟_高兴`, `灵语烟_悲伤`, `灵律雅_恐惧`, `灵律雅_生气`, `灵律雅_惊讶`, `灵律雅_高兴`, `灵律雅_厌恶`, `灵律雅_中性`, `灵律雅_悲伤`, `灵惊涛_厌恶`, `灵惊涛_恐惧`, `灵惊涛_高兴`, `灵惊涛_悲伤`, `灵惊涛_中性`, `灵惊涛_惊讶`, `灵惊涛_生气`, `灵寒陌_生气`, `灵寒陌_厌恶`, `灵寒陌_恐惧`, `灵寒陌_高兴`, `灵寒陌_中性`, `灵寒陌_惊讶`, `灵寒陌_悲伤`, `灵决冥_悲伤`, `灵决冥_厌恶`, `灵决冥_生气`, `灵决冥_中性`, `灵决冥_惊讶`, `灵决冥_恐惧`, `灵决冥_高兴`, `灵亦尘_中性`, `灵亦尘_厌恶`, `灵亦尘_高兴`, `灵亦尘_恐惧`, `灵亦尘_生气`, `灵亦尘_悲伤`, `灵亦尘_惊讶`, `灵语尘_生气`, `灵语尘_高兴`, `灵语尘_厌恶`, `灵语尘_惊讶`, `灵语尘_中性`, `灵语尘_悲伤`, `灵语尘_恐惧`, `灵慈萱_高兴`, `灵慈萱_厌恶`, `灵慈萱_惊讶`, `灵慈萱_生气`, `灵慈萱_中性`, `灵慈萱_悲伤`, `灵慈萱_恐惧`, `灵妙音_中性`, `灵妙音_悲伤`, `灵妙音_惊讶`, `灵妙音_厌恶`, `灵妙音_生气`, `灵妙音_高兴`, `灵妙音_恐惧`, `灵断尘_悲伤`, `灵断尘_高兴`, `灵断尘_惊讶`, `灵断尘_恐惧`, `灵断尘_中性`, `灵断尘_生气`, `灵断尘_厌恶`, `灵悦兮_高兴`, `灵悦兮_中性`, `灵悦兮_生气`, `灵悦兮_惊讶`, `灵悦兮_厌恶`, `灵悦兮_悲伤`, `灵悦兮_恐惧`, `灵若虚_恐惧`, `灵若虚_惊讶`, `灵若虚_中性`, `灵若虚_悲伤`, `灵若虚_高兴`, `灵若虚_厌恶`, `灵若虚_生气`, `灵熙婉_厌恶`, `灵熙婉_生气`, `灵熙婉_惊讶`, `灵熙婉_高兴`, `灵熙婉_中性`, `灵熙婉_恐惧`, `灵熙婉_悲伤`, `灵旭扬_中性`, `灵旭扬_生气`, `灵旭扬_恐惧`, `灵旭扬_高兴`, `灵旭扬_惊讶`, `灵旭扬_悲伤`, `灵旭扬_厌恶`, `灵阅川_生气`, `灵阅川_悲伤`, `灵阅川_高兴`, `灵阅川_厌恶`, `灵阅川_惊讶`, `灵阅川_中性`, `灵阅川_恐惧`, `灵未央_高兴`, `灵未央_中性`, `灵未央_惊讶`, `灵未央_悲伤`, `灵未央_厌恶`, `灵未央_恐惧`, `灵未央_生气`, `灵妩月_厌恶`, `灵妩月_生气`, `灵妩月_中性`, `灵妩月_惊讶`, `灵妩月_恐惧`, `灵妩月_高兴`, `灵妩月_悲伤`, `灵梦瑶_中性`, `灵梦瑶_恐惧`, `灵梦瑶_高兴`, `灵梦瑶_悲伤`, `灵梦瑶_惊讶`, `灵梦瑶_生气`, `灵妙声_中性`, `灵妙声_恐惧`, `灵妙声_高兴`, `灵妙声_悲伤`, `灵妙声_惊讶`, `灵妙声_生气`, `灵烈瑶_中性`, `灵烈瑶_厌恶`, `灵烈瑶_恐惧`, `灵烈瑶_悲伤`, `灵烈瑶_惊讶`, `灵烈瑶_生气`, `灵烈瑶_高兴`, `灵幽月_中性`, `灵幽月_厌恶`, `灵幽月_恐惧`, `灵幽月_悲伤`, `灵幽月_惊讶`, `灵幽月_生气`, `灵幽月_高兴`, `灵悦心_中性`, `灵悦心_厌恶`, `灵悦心_恐惧`, `灵悦心_悲伤`, `灵悦心_惊讶`, `灵悦心_生气`, `灵悦心_高兴`, `灵汐月_中性`, `灵汐月_厌恶`, `灵汐月_恐惧`, `灵汐月_悲伤`, `灵汐月_惊讶`, `灵汐月_生气`, `灵汐月_高兴`, `灵书意_中性`, `灵书意_厌恶`, `灵书意_恐惧`, `灵书意_悲伤`, `灵书意_惊讶`, `灵书意_生气`, `灵书意_高兴`, `灵知远_中性`, `灵知远_厌恶`, `灵知远_恐惧`, `灵知远_悲伤`, `灵知远_惊讶`, `灵知远_生气`, `灵知远_高兴`, `灵温言_中性`, `灵温言_厌恶`, `灵温言_恐惧`, `灵温言_悲伤`, `灵温言_惊讶`, `灵温言_生气`, `灵温言_高兴`, `灵傲尘_中性`, `灵傲尘_厌恶`, `灵傲尘_恐惧`, `灵傲尘_悲伤`, `灵傲尘_生气`, `灵傲尘_高兴`, `灵佑安_中性`, `灵佑安_厌恶`, `灵佑安_恐惧`, `灵佑安_悲伤`, `灵佑安_惊讶`, `灵佑安_生气`, `灵佑安_高兴`, `灵惜晨_中性`, `灵惜晨_厌恶`, `灵惜晨_恐惧`, `灵惜晨_悲伤`, `灵惜晨_惊讶`, `灵惜晨_生气`, `灵惜晨_高兴`, `灵宛溪_中性`, `灵宛溪_厌恶`, `灵宛溪_恐惧`, `灵宛溪_悲伤`, `灵宛溪_惊讶`, `灵宛溪_生气`, `灵宛溪_高兴`, `灵言澈_中性`, `灵言澈_厌恶`, `灵言澈_恐惧`, `灵言澈_悲伤`, `灵言澈_惊讶`, `灵言澈_生气`, `灵言澈_高兴`, `灵溯光_中性`, `灵溯光_厌恶`, `灵溯光_恐惧`, `灵溯光_悲伤`, `灵溯光_惊讶`, `灵溯光_生气`, `灵溯光_高兴`, `灵惊弦_中性`, `灵惊弦_厌恶`, `灵惊弦_恐惧`, `灵惊弦_悲伤`, `灵惊弦_惊讶`, `灵惊弦_生气`, `灵惊弦_高兴`, `灵嘉肴_中性`, `灵嘉肴_厌恶`, `灵嘉肴_恐惧`, `灵嘉肴_悲伤`, `灵嘉肴_惊讶`, `灵嘉肴_生气`, `灵嘉肴_高兴`, `灵黛嗔_中性`, `灵黛嗔_厌恶`, `灵黛嗔_恐惧`, `灵黛嗔_悲伤`, `灵黛嗔_惊讶`, `灵黛嗔_生气`, `灵黛嗔_高兴`, `灵孤影_中性`, `灵孤影_厌恶`, `灵孤影_恐惧`, `灵孤影_悲伤`, `灵孤影_惊讶`, `灵孤影_生气`, `灵孤影_高兴`, `灵雨汐_中性`, `灵雨汐_恐惧`, `灵雨汐_高兴`, `灵雨汐_悲伤`, `灵雨汐_惊讶`, `灵雨汐_生气`, `蜡笔小新_野原新之助 (小新)`, `蜡笔小新_野原美伢 (美伢)`, `小猪佩奇_佩奇`, `康熙王朝_康熙`, `西游记_唐僧`, `甄嬛传_四郎`, `潜伏_余则成`, `水浒传_武松`, `水浒传_潘金莲`, `雍正王朝_雍正` | `灵小柔` | Talker built-in voice name; ignored when reference audio is supplied. |
| `reference_text` | `string` | Empty | Reference transcript enabling acoustic-prefix continuation; without it reference audio supplies speaker conditioning only. |
| `language` | `string` | Unset | Talker output language; inferred from its input text when omitted. |
| `seed` | `int` | `42` | Talker flow and reference-posterior sampling seed. Thinker generation is greedy. Declared range: minimum 0. |
| `guidance_scale` | `float` | `2` | Talker conditioned prediction plus guidance_scale times its difference from the null prediction. |
| `sigma` | `float` | `0.25` | Talker flow SDE noise scale. Declared range: minimum 0. |
| `temperature` | `float` | `0` | Talker flow sampling temperature; zero still uses seeded initial noise. Declared range: minimum 0. |
| `raw_text` | `bool` | `false` | Synthesize text as one fragment without framework text chunking. |
| `trim_silence` | `bool` | `true` | Trim trailing low-energy 100-ms frames per Talker fragment, retaining a 300-ms tail. |
| `text_chunk_size` | `int` | `100` | Talker text chunk budget in Unicode codepoints. Declared range: minimum 1. |
| `text_chunk_mode` | `default`, `tag_aware`, `japanese`, `endline` | `tag_aware` | Framework Talker text chunking policy. |

## Session Options (use with `--session-option`)

Shared MoE controls apply to ASR and speech-to-speech sessions, which use the
Thinker. TTS and voice-cloning sessions use only the Talker and accept explicit
`moe_cpu_share=0` as disabled. They reject positive CPU shares and other shared
`moe_*` controls. Existing family-prefixed
expert controls remain accepted but inactive for those Talker-only tasks.

MoE defaults: resident banks, pinning on (off on Metal), eight cache slots,
one read thread, zero CPU MoE layers; CPU share is zero.
Use the [shared MoE controls](../usage.md#shared-moe-session-controls) via CLI
flags, `--session-option`, or the server model's `session_options`.
Legacy `ming3_audio.disk_experts`, `.pin_experts`, `.expert_cache_size`,
and `.disk_expert_read_threads` aliases remain accepted.
For CPU-layer placement, use the shared `moe_cpu_expert_layers` key.
Supplying both a canonical key and its alias is an error, even with equal values.
CPU counts select the first N MoE ordinals, skipping dense-only backbone layers.
CPU-selected layers override `moe_cpu_share` and retain bounded CPU caches in
staged mode. Pinning is accepted but inactive without staging.

| Option | Type / values | Default | Description |
|---|---|---|---|
| `moe_disk_experts` | `bool` | `false` | Keep Thinker expert weights outside GPU memory and stage routed experts into a bounded cache on the selected backend or CPU, according to layer placement. |
| `moe_pin_experts` | `bool` | `true` | With disk_experts, retain expert payloads in pinned host RAM instead of reading them from the GGUF on each cache miss. Defaults to false on Metal, which uses ordinary host staging. |
| `moe_expert_cache_size` | `int` | `8` | With disk_experts, number of cached expert slots per routed Thinker layer. Declared range: minimum 8, maximum 128. |
| `moe_disk_expert_read_threads` | `int` | `1` | Host workers for copying selected GGUF experts into staging when disk_experts is enabled and pin_experts is false. Default 1 preserves serial reads; values above 1 use OpenMP or Apple dispatch. Declared range: minimum 1, maximum 64. |
| `moe_cpu_expert_layers` | `int` | `0` | First N routed MoE layers on CPU; dense-only layers do not count. |
| `moe_cpu_share` | `float` | `0` | CPU share of eligible staged GPU misses, 0–1; requires `moe_disk_experts=true`. CPU-selected layers take precedence. |
| `ming3_audio.reference_cache` | `bool` | `true` | Reuse speaker features and seeded acoustic conditioning for the most recent reference waveform. |

## Usage Notes

Use `--request-option name=value`, or the server request's `options` map.

Session option `ming3_audio.reference_cache=true` caches prepared conditioning
for the most recent reference waveform. There are no model-specific load options.

`--session-option moe_disk_experts=true` keeps Thinker expert weights in
host memory and stages selected experts into bounded backend or CPU caches.
CPU placement is controlled separately by `moe_cpu_expert_layers`.
`moe_pin_experts=true` retains expert payloads in pinned RAM; set it to `false` for range reads from the GGUF instead.
It defaults to `false` on Metal, which uses ordinary host staging.
`moe_expert_cache_size` selects 8 to 128 slots per layer (default 8).
`moe_disk_expert_read_threads` selects 1 to 64 host workers for copying
selected GGUF experts into staging. The default is 1 (serial); try 4 using
OpenMP or Apple dispatch. This applies only with `moe_disk_experts=true` and `moe_pin_experts=false`.
The host reader is backend-independent; Metal performance has not been validated.
Spoken-response sessions release Thinker workspaces before allocating the speech
stage in this mode. The offloaded Thinker allocates its KV cache in growing
buckets rather than reserving the entire `max_tokens` allowance upfront.
Growth preserves the existing KV and recurrent state without changing precision
or reducing the token limit. Memory still grows with actual context length.
The normal resident path remains the default.

Thinker MLA attention uses F16 KV storage; KDA recurrent state and convolution
tails remain F32. CUDA decode reuses session-owned attention graphs sized to
the live prefix, retaining the full context. KDA input projections and shared
MLP gate/up projections are packed at load without changing weight precision.
These optimizations are enabled by default and do not change the Talker or VAE.

The native frontend uses the framework text chunker rather than Python's
word/character buffering and does not run the optional Python FST normalizer.
Spell out numbers and abbreviations when pronunciation matters. Long text is
synthesized as independent fragments. Reference conditioning is retained across
fragments; generated conversation history is not.

Long clones can omit words within a fragment. The default chunk budget is 100;
any positive integer remains supported. For example,
`--request-option text_chunk_size=300` selects the previous budget. On a
1,518-character English clone with the same reference audio and transcript,
reducing the budget from 300 to 100 lowered Whisper turbo WER from 4.35% to
0.40% with seed 42 and from 3.95% to 0% with seed 43. Whisper medium independently
confirmed the improvement. Smaller chunks reduce this risk but do not guarantee
word-perfect output.
Chunk boundaries and numerical execution can change generated lengths and the
random stream consumed by later fragments; a fixed seed does not guarantee
matching words across implementations. The tested BF16 and Q8_0 packages produce
identical clone WAVs, so this omission is not caused by Thinker quantization.

## Conversion

Build the Debug `audiocpp_gguf` target, then convert the complete official snapshot:

```bash
python tools/models/convert_ming3_audio_gguf.py \
  --source /path/to/Ming-3.0-audio-tiny \
  --output /path/to/ming-3.0-audio-tiny-bf16.gguf \
  --type orig --temporary-dir /path/to/conversion-work --log
```

The converter stacks MoE expert banks without changing their BF16 values and
keeps expert routers in F32. Training-only next-token prediction weights are
not included. Conversion requires PyTorch, NumPy, ONNX, Safetensors, and GGUF
Python packages; inference does not require them. The checkpoint is MIT-licensed.

`--type q8_0` quantizes only Thinker expert banks. `--type q4_k` uses Q4_K
experts plus a Q8_0 text output head to reduce GPU-resident weights during
expert offload. Both retain source precision for token embeddings, attention and
dense/shared MLP projections: broader projection quantization caused repetition
or prolonged generation on the full meeting examples. Both
preserve source precision for the audio encoder, Talker, VAE and speaker
encoder, and keep routers F32. TTS-only memory usage therefore does not shrink
with these Thinker quantizations.
