# VeVo2

VeVo2 supports speech, singing, voice conversion, singing conversion, and speech editing through explicit routes. The route matters: some routes start from text, some from source audio, and some require prosody or style references.

Common CLI shape:

```bash
audiocpp_cli --task <task> --family vevo2 --model models/Vevo2 --backend cuda --task-route <route> ...
```

## Model

| Field | Value |
|---|---|
| Family | `vevo2` |
| Model directory | `models/Vevo2` |
| Tasks | `tts`, `vc`, `s2s`, `svc` |
| Modes | `offline` |
| Voice input | Target timbre WAV through `--voice-ref` or `--target-voice` |
| Text input | `--text` or `--target-text`, depending on route |
| Audio input | `--source-audio`, `--prosody-ref`, and `--style-ref`, depending on route |

## Zero-Shot TTS

Generate speech from text using the target voice timbre. This is the default route for `--task tts`. Use a reference WAV and transcript that match each other; poor transcript/reference alignment can leak reference content because the upstream VeVo2 prompt path conditions on both reference transcript and target text.

> [!WARNING]
> The official Python zero-shot TTS route itself has this leakage/conditioning behavior, meaning the generated audio may include or echo part of the reference prompt instead of speaking only the target text.

| Field | Value |
|---|---|
| Task | `tts` |
| Route | `zero_shot_tts` |
| Required text | `--text` or `--target-text` |
| Required voice | `--voice-ref` or `--target-voice` |
| Optional style transcript | `--style-ref-text` or `--reference-text` |
| Prosody reference | Not required by default |
| Pitch shift default | `false` |

```bash
audiocpp_cli --task tts --family vevo2 --model models/Vevo2 --backend cuda --task-route zero_shot_tts --text "This is a VeVo2 voice." --voice-ref voice.wav --style-ref-text "Transcript of voice.wav." --out out.wav
```

## Text To Singing

Generate singing from lyrics/text and a target voice. Use a prosody/melody reference when you want explicit melody conditioning.

| Field | Value |
|---|---|
| Task | `tts` |
| Route | `text_to_singing` |
| Required text | `--target-text` or `--text` |
| Required voice | `--voice-ref` or `--target-voice` |
| Prosody reference | Optional unless `--use-prosody-code=true` is set |
| Pitch shift default | `false` |

```bash
audiocpp_cli --task tts --family vevo2 --model models/Vevo2 --backend cuda --task-route text_to_singing --target-text "We follow the light" --voice-ref target_singer.wav --prosody-ref melody.wav --out singing.wav
```

## Singing Voice Synthesis

Generate singing from lyrics/prosody using the target voice. This route is still a `tts` task because it synthesizes from text/prosody rather than converting an existing source vocal.

| Field | Value |
|---|---|
| Task | `tts` |
| Route | `svs` |
| Required text | `--target-text` or `--text` |
| Required voice | `--voice-ref` or `--target-voice` |
| Prosody reference | Optional unless `--use-prosody-code=true` is set |
| Pitch shift default | `false` |

```bash
audiocpp_cli --task tts --family vevo2 --model models/Vevo2 --backend cuda --task-route svs --target-text "We follow the light" --voice-ref target_singer.wav --prosody-ref melody.wav --out svs.wav
```

## Style-Preserved Voice Conversion

Convert source speech to the target voice while preserving the source speech style. This is the default route for `--task vc`.

| Field | Value |
|---|---|
| Task | `vc` |
| Route | `style_preserved_vc` |
| Required source | `--source-audio` or `--audio` |
| Required target voice | `--target-voice` or `--voice-ref` |
| Text | Not required |
| Prosody code default | `false` |
| Pitch shift default | `true` |

```bash
audiocpp_cli --task vc --family vevo2 --model models/Vevo2 --backend cuda --task-route style_preserved_vc --source-audio source.wav --target-voice assets/resources/b.wav --out converted.wav
```

## Style-Converted Voice Conversion

Convert speech with explicit text/prosody/style conditioning. The source audio is used as the prosody/style reference unless separate references are provided.

| Field | Value |
|---|---|
| Task | `vc` |
| Route | `style_converted_vc` |
| Required source | `--source-audio` or `--audio` |
| Required target voice | `--target-voice` or `--voice-ref` |
| Required text | `--target-text` or `--text` |
| Prosody code default | `true` |
| Pitch shift default | `false` |

```bash
audiocpp_cli --task vc --family vevo2 --model models/Vevo2 --backend cuda --task-route style_converted_vc --source-audio source.wav --target-voice assets/resources/b.wav --target-text "Convert this sentence with the target timbre." --out converted_style.wav
```

## Speech Editing

Edit source speech into new target text while using the target voice. This is the default route for `--task s2s`.

| Field | Value |
|---|---|
| Task | `s2s` |
| Route | `editing` |
| Required source | `--source-audio` or `--audio` |
| Required target voice | `--target-voice` or `--voice-ref` |
| Required text | `--target-text` or `--text` |
| Prosody code default | `true` |
| Pitch shift default | `false` |

```bash
audiocpp_cli --task s2s --family vevo2 --model models/Vevo2 --backend cuda --task-route editing --source-audio source.wav --target-voice assets/resources/b.wav --target-text "Replace this sentence." --out edited.wav
```

## Style-Preserved Singing Conversion

Convert a singing source to the target singer while preserving source singing style. This is the default route for `--task svc`.

| Field | Value |
|---|---|
| Task | `svc` |
| Route | `style_preserved_svc` |
| Required source | `--source-audio` or `--audio` |
| Required target voice | `--target-voice` or `--voice-ref` |
| Text | Not required |
| Prosody code default | `false` |
| Pitch shift default | `true` |

```bash
audiocpp_cli --task svc --family vevo2 --model models/Vevo2 --backend cuda --task-route style_preserved_svc --source-audio song.wav --target-voice target_singer.wav --out svc.wav
```

## Style-Converted Singing Conversion

Convert singing with explicit lyrics/prosody/style conditioning.

| Field | Value |
|---|---|
| Task | `svc` |
| Route | `style_converted_svc` |
| Required source | `--source-audio` or `--audio` |
| Required target voice | `--target-voice` or `--voice-ref` |
| Required text | `--target-text` or `--text` |
| Prosody code default | `true` |
| Pitch shift default | `true` |

```bash
audiocpp_cli --task svc --family vevo2 --model models/Vevo2 --backend cuda --task-route style_converted_svc --source-audio song.wav --target-voice target_singer.wav --target-text "New lyrics for the converted singing." --out svc_style.wav
```

## Singing Style Conversion

Convert singing style with explicit conditioning. The source audio is also used as a reference for the style/prosody path when separate references are not provided.

| Field | Value |
|---|---|
| Task | `svc` |
| Route | `singing_style_conversion` |
| Required source | `--source-audio` or `--audio` |
| Required target voice | `--target-voice` or `--voice-ref` |
| Required text | `--target-text` or `--text` |
| Prosody code default | `true` |
| Pitch shift default | `true` |

```bash
audiocpp_cli --task svc --family vevo2 --model models/Vevo2 --backend cuda --task-route singing_style_conversion --source-audio song.wav --target-voice target_singer.wav --target-text "Keep the song but change the style." --out singing_style.wav
```

## Humming Or Instrument To Singing

Generate singing from a humming or instrumental melody. The two route names share the same melody-control path; use the name that matches your input.

| Field | Value |
|---|---|
| Task | `svc` |
| Routes | `humming_to_singing`, `instrument_to_singing` |
| Required prosody | `--prosody-ref` or `--audio`, containing humming or instrumental melody |
| Required target voice | `--target-voice` or `--voice-ref` |
| Required text | `--target-text` or `--text` |
| Prosody code default | `true` |
| Pitch shift default | `true` |

```bash
audiocpp_cli --task svc --family vevo2 --model models/Vevo2 --backend cuda --task-route humming_to_singing --prosody-ref humming.wav --target-voice target_singer.wav --target-text "Lyrics to sing from the humming melody." --out humming_song.wav
```

```bash
audiocpp_cli --task svc --family vevo2 --model models/Vevo2 --backend cuda --task-route instrument_to_singing --prosody-ref melody.wav --target-voice target_singer.wav --target-text "Lyrics to sing from the instrumental melody." --out instrument_song.wav
```

## Shared Controls

| Option | Values | Default | Meaning |
|---|---|---:|---|
| `--task-route` | route names above | `zero_shot_tts` for `tts`, `style_preserved_vc` for `vc`, `editing` for `s2s`, `style_preserved_svc` for `svc` | Select VeVo2 route. |
| `--text` | text | empty string | Target text for TTS-style routes. |
| `--target-text` | text | `--text` value | Target text/lyrics when separate from main `--text`. |
| `--source-audio` | WAV path | not set | Source speech, song, humming, or instrument. Required by conversion/edit routes. |
| `--target-voice` / `--voice-ref` | WAV path | required | Target timbre reference. |
| `--prosody-ref` | WAV path | not set | Prosody/melody reference. Required when `--use-prosody-code=true` on text/prosody routes. |
| `--style-ref` | WAV path | not set | Style reference audio. |
| `--style-ref-text` / `--reference-text` | text | empty string | Transcript for style reference. |
| `--use-prosody-code` | `true`, `false` | route-dependent | Enable explicit prosody conditioning. Defaults to `true` for style-converted VC/SVC, editing, singing style conversion, humming-to-singing, and instrument-to-singing; otherwise `false`. |
| `--use-pitch-shift` | `true`, `false` | route-dependent | Pitch-align source/prosody/style references to the target voice. Defaults to `true` for style-preserved VC/SVC, style-converted SVC, singing style conversion, humming-to-singing, and instrument-to-singing; otherwise `false`. |
| `--source-shift-steps` | integer semitones | `0` | Manual source pitch shift. If `0` and pitch shift is enabled on source-audio routes, VeVo2 estimates it. |
| `--prosody-shift-steps` | integer semitones | `0` | Manual prosody pitch shift. If `0` and pitch shift is enabled with a prosody reference, VeVo2 estimates it. |
| `--style-shift-steps` | integer semitones | `0` | Manual style pitch shift. If `0` and pitch shift is enabled with a style reference, VeVo2 estimates it. |
| `--target-duration-seconds` | float | not set | Flow-matching target duration hint. |
| `--reference-duration-seconds` | float | not set | Trim target voice reference before conditioning. |
| `--temperature` | float | `1.0` | AR sampling temperature. |
| `--top-k` | integer | `25` | AR top-k. |
| `--top-p` | float | `0.8` | AR top-p. |
| `--repetition-penalty` | float | `1.1` | AR repetition penalty. |
| `--max-tokens` | integer | `500` | Maximum AR tokens. |
| `--num-inference-steps` | integer | `32` | Flow-matching steps. |
| `--seed` | integer | random if omitted | Request seed. |
| `--predict-target-prosody` | `true`, `false` | `false` | Parsed by the CLI, but `true` is not implemented in the current reference path. |

For backend weight-type controls, use `audiocpp_cli --inspect --model models/Vevo2 --family vevo2`.
