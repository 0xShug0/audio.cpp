# Cohere ASR manifest inputs

CLI short-long-short and chunking cases use `assets/resources/sample_16k.wav`
and `assets/resources/qwen3_tts_longform_asr_input.wav`. The English LibriSpeech
cases use `assets/asr_validation/librispeech/librispeech_test_clean_6930-75918-0000.wav`.
Held-out and punctuation variants remain separate; request ordering and chunk
settings are unchanged.

## External multilingual inputs

`cases.json` requires three externally supplied mono 16 kHz WAV recordings:

| Language | Default external input | Cases |
| --- | --- | --- |
| German | `reference/cohere_asr/inputs/de-16k.wav` | `de-pnc`, `de-no-pnc` |
| French | `reference/cohere_asr/inputs/fr-16k.wav` | `fr-pnc`, `fr-no-pnc` |
| Spanish | `reference/cohere_asr/inputs/es-16k.wav` | `es-pnc`, `es-no-pnc` |

These are required inputs, not generated log artifacts or bundled assets. No
licensed DE/FR/ES replacements are currently provided by this repository.
Supply actual speech in each language and keep its exact transcript, source
revision/recording ID, license, and SHA-256 alongside the external fixture set.
Use identical bytes and transcripts when comparing implementations or revisions.
English speech does not preserve this language-quality coverage.

`reference.py` already accepts `--cases` and reads each `audio` path directly;
it does not expand environment variables in JSON. To use arbitrary external
paths, set `DE_WAV`, `FR_WAV`, and `ES_WAV` to absolute paths and `CASES_JSON`
to an output manifest outside the tracked test files. From the repository root,
materialize all six language cases without changing their IDs or language flags:

```bash
jq --arg de "${DE_WAV:?absolute German WAV path required}" \
   --arg fr "${FR_WAV:?absolute French WAV path required}" \
   --arg es "${ES_WAV:?absolute Spanish WAV path required}" \
   'map(if .language == "de" then .audio = $de
        elif .language == "fr" then .audio = $fr
        elif .language == "es" then .audio = $es
        else . end)' tests/cohere_asr/cases.json > "${CASES_JSON:?external manifest path required}"
```

Pass `--cases "$CASES_JSON"` to `tests/cohere_asr/reference.py` with its existing
required `--model`, `--output`, and `--log` arguments. Run in conda `qwen3-tts`
with CUDA, eight threads, and saved logs; coordinate runs sequentially.
This list-format manifest is for the reference harness; the CLI path runner
uses the separate object-format `cli_cases.json` catalog.

The reference harness records generated text but does not score it against a
ground-truth transcript. Retain all six cases and assess their full transcripts
against the matching external references; successful execution or agreement
between implementations alone does not establish language accuracy.
