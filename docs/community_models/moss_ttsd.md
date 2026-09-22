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
  --request-option voice_samples=/path/to/s1.wav,/path/to/s2.wav \
  --request-option "reference_text=[S1] This is the first voice. [S2] And this is the second." \
  --text "[S1] So what did you make of it? [S2] Honestly, I was not expecting that ending." \
  --out dialogue.wav
```

**The number of entries is the number of speakers**, and an entry with no path
is a speaker that is named but not cloned:

| `voice_samples=` | speakers |
|---|---|
| `s1.wav,s2.wav` | both cloned |
| `s1.wav,` | `[S1]` cloned, `[S2]` invented |
| `,s2.wav` | `[S1]` invented, `[S2]` cloned |
| `s1.wav` | one speaker only — no `[S2]` is implied |

So the trailing separator is meaningful, not a typo: `s1.wav,` and `s1.wav` are
different prompts.

### Why `reference_text` matters

This model does not clone by showing the reference and then starting fresh. It
clones by **continuation**: the reference recording is handed to the model as
audio it has already produced, and the dialogue is generated as a continuation of
it. `reference_text` is what that audio says, and it is prepended to your text so
the words and the audio line up. Omitting it leaves the model continuing a
recording whose transcript it was never given, which is worth hearing before you
rely on it.

## Packages and what fits

| package | size | clones on a 24 GiB card |
|---|---|---|
| `moss_ttsd_q8_0_codec_f16` (default) | 12.2 GB | yes |
| `moss_ttsd_q4_k_codec_f16` | 9.5 GB | yes |
| `moss_ttsd_bf16_codec_f16` | 18.9 GB | **no** |

⚠ **bf16 cannot clone on a 24 GiB card.** Cloning needs the codec *encoder* as
well as the decoder, and its weights want a further 3.5 GB on top of the
backbone; on an RTX 3090 that fails at `moss.audio_tokenizer.encoder`. Plain
generation in bf16 is fine. Use q8_0 to clone on 24 GiB, or run bf16 on CPU.

### q4_k needs its heads left alone

⚠ **A plain q4_k of this model is broken**, and quietly: it speaks the first turn
of a dialogue and stops.

```
q4_k, heads quantised   3.20s  "The train leaves at four in the afternoon. Ah, back, please."
q4_k, heads quantised   2.56s  "The train leaves at 4 in the afternoon."
q4_k, heads at f16     16.48s  full script
q8_0                   12.48s  full script
```

`lm_heads.0` is the text head, and it is what decides when the turn ends.
Quantised to four bits it ends the turn early, which reads as a model that
cannot hold a dialogue rather than as a bad quantisation. The published package
is therefore built with the heads kept at f16, which costs 0.95 GB:

```bash
audiocpp_gguf ... --type q4_k \
  --keep-type "audio_tokenizer_weights*=f16" \
  --keep-type "model_weights/lm_heads*=f16"
```

Note the namespace separator is `/`, not `.`. A pattern that does not match is
not an error -- it silently produces the broken package.

## How well does it clone?

**Not reliably, and this is the model rather than this port.** Measuring median
F0 per turn against two references 84 Hz apart, across three runs of the
reference implementation and one of ours:

```
run                S1 turn1  S1 turn2  S2 turn1  S2 turn2
reference run 1      200.0     110.3     154.8     131.9
reference run 2      203.4     102.8     179.1     160.0
reference run 3      208.8     104.8     161.1     152.5
ours (seed 7)        210.5     120.6     166.7     164.4
references:          S1 = 201.7 Hz, S2 = 117.6 Hz
```

The first speaker's first turn matches its reference closely. By that speaker's
second turn the pitch has fallen to roughly the *other* reference's, and the
second speaker sits between the two throughout. The reference implementation
does this in every run and ours reproduces the pattern at the same magnitude.

Median F0 is a crude stand-in for speaker identity and this is four runs of one
configuration with short references, so treat it as "identity drifts across
turns" rather than as a measured rate. Longer references may do better; the
model card's own example uses them.

Separately, one take in four ended with a short spurious utterance after the
script had finished. Too few takes to attribute it to anything.

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
