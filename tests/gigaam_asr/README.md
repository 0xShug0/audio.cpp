# GigaAM ASR manifest inputs

`path_cases.json` uses `assets/resources/sample_16k.wav` for six repeated
requests in one session. The case is now `gigaam_shared_repeat`, replacing
`gigaam_official_repeat`; the input is shared English speech rather than the
external official example formerly at `build/logs/gigaam_asr/inputs/example.wav`.
This preserves repeated-request/session-reuse coverage only. It does not
establish official-example transcription quality or Russian-language accuracy.
Official-example or Russian-language acceptance requires a separate official
or licensed Russian fixture with its matching transcript.

`gigaam_v4_e2e_repeat` and `gigaam_v4_raw_repeat` run the same shared input
through the GigaAM-v4 package (`gigaam-v4-rnnt-f16.gguf`, converted locally as
described in `docs/models/gigaam_asr.md`), once per head; the raw case selects
its head and tokenizer with `gigaam_asr.head=raw`.

`python_reference.py` also accepts a GigaAM-v4 snapshot (`model.safetensors`
and the checkpoint's `modeling_gigaam.py`) with `--head e2e|raw`. With F16 or
Q8_0 v4 packages on CPU, the CLI transcripts of `sample_16k.wav` and of the
official GigaAM Russian example matched this reference byte for byte for both
heads. The official example is not committed; download it to an ignored
directory such as `build/logs/gigaam_asr/inputs/`.
