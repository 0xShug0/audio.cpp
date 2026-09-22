# MOSS-TTSD

Dialogue text-to-speech: a speaker-tagged conversation rendered in one take, with
per-speaker zero-shot voice cloning. 8B Qwen3 backbone, 16 codebooks, 24 kHz.

MOSS-TTSD is the dialogue member of the `moss_tts_delay` family, which it shares
with [MOSS-TTS-v1.5](moss_tts_v15.md) and [MOSS-VoiceGenerator](moss_voicegen.md):
the same backbone, heads, delay decoder, audio-tokenizer codec and prompt
assembly. What is its own is how its prompt block renders and how it clones.

| | |
|---|---|
| Family | `moss_ttsd` |
| Tasks | `tts`, `clon` |
| Codec | MOSS-Audio-Tokenizer v1, first 16 of its 32 RVQ layers |
| Sample rate | 24 kHz mono |

## Speaking a dialogue

The text is speaker-tagged. Tags are positional: `[S1]` is the first speaker,
`[S2]` the second, and they may alternate as often as the script does.

```bash
audiocpp_cli --task tts --family moss_ttsd --model /path/to/MOSS-TTSD-GGUF \
  --backend cuda --language English \
  --text "[S1] So what did you make of it? [S2] Honestly, I was not expecting that ending." \
  --out dialogue.wav
```

## Cloning the speakers

Pass one reference recording per speaker, comma separated and positional, and
the transcript of what those recordings say:

```bash
audiocpp_cli --task clon --family moss_ttsd --model /path/to/MOSS-TTSD-GGUF \
  --backend cuda --language English \
  --option voice_samples=/path/to/s1.wav,/path/to/s2.wav \
  --option "reference_text=[S1] This is the first voice. [S2] And this is the second." \
  --text "[S1] So what did you make of it? [S2] Honestly, I was not expecting that ending." \
  --out dialogue.wav
```

**Leave an entry empty to clone one speaker and invent the other.**
`voice_samples=/path/to/s1.wav,` gives `[S1]` the recording and lets the model
choose a voice for `[S2]`.

### Why `reference_text` matters

This model does not clone by showing the reference and then starting fresh. It
clones by **continuation**: the reference recording is handed to the model as
audio it has already produced, and the dialogue is generated as a continuation of
it. `reference_text` is what that audio says, and it is prepended to your text so
the words and the audio line up. Omitting it leaves the model continuing a
recording whose transcript it was never given, which is worth hearing before you
rely on it.

## Things worth knowing

- **No duration budget.** The family's `- Tokens:` field is baked to `None` in
  this checkpoint's own template, so unlike v1.5 there is nothing to request. The
  `max_frames` option is a ceiling against a run that never stops, not a target.
- **No `scene` control**, despite the field in the upstream message class: the
  reference renderer substitutes the literal `"None"` for it unconditionally, so
  no value has ever reached the model.
- **The text is not normalised**, unlike v1.5's. Numbers and abbreviations are
  spoken as the model reads them, and the speaker tags are passed through
  untouched.
- **One take, not chunked.** A long single-voice text can be split and rendered
  piece by piece; a dialogue cannot, because which speaker a fragment belongs to
  depends on the tags before it and each piece would restart the continuation.
  Very long scripts are bounded by the backbone's context rather than by chunking.
