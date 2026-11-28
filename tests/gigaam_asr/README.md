# GigaAM ASR manifest inputs

`path_cases.json` uses `assets/resources/sample_16k.wav` for six repeated
requests in one session. The case is now `gigaam_shared_repeat`, replacing
`gigaam_official_repeat`; the input is shared English speech rather than the
external official example formerly at `build/logs/gigaam_asr/inputs/example.wav`.
This preserves repeated-request/session-reuse coverage only. It does not
establish official-example transcription quality or Russian-language accuracy.
Official-example or Russian-language acceptance requires a separate official
or licensed Russian fixture with its matching transcript.
