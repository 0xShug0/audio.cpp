# Whistle five-language synthetic spot checks

Measured on 2026-10-06 using the current locally converted FP32 GGUF.

These clips test French, Spanish, Italian, Dutch, and Polish through the actual
Whistle CLI. They are synthetic prompt-agreement checks, not a general accuracy
benchmark or independently human-annotated recordings. No runtime code changes
are required for these checks.

## Inputs and observed transcripts

ElevenLabs CLI 1.4.0 generated raw `pcm_16000`. Python standard-library `wave`
wrapped it as signed 16-bit mono WAV without changing samples or resampling.
Only one voice and one short prompt per language are covered.

All ten CLI invocations exit 0. Automatic and forced-language outputs agree.
Unicode NFKC, case folding, and punctuation removal produce zero word edits
against each synthesis prompt. Accents remain significant. The CLI does not
expose the resulting transcript language field, so automatic language-selection
execution is verified, not its internal language tag.

### fr.

Prompt: Bonjour. Aujourd'hui, le ciel est clair et la lumière est douce.

Observed automatic and forced transcript: Bonjour, aujourd'hui le ciel est clair et la lumière est douce.

Input duration is 4.040250 seconds. WAV SHA-256 is
`245e4386e74a2e137c7cea0635d0e41084033354c8972ea5dc53a826643f2edf`.

Automatic inference is 182.696 ms, RTF 0.045219,
peak process working set 434 MiB.
Forced inference is 184.586 ms, RTF 0.0456867,
peak process working set 433 MiB.

### es.

Prompt: Hola. Hoy el cielo está despejado y la luz es suave.

Observed automatic and forced transcript: Hola, hoy el cielo está despejado y la luz es suave.

Input duration is 3.436562 seconds. WAV SHA-256 is
`0f7b3b1ffe169852dd3443e92e9ef36cbac210af70bc34d95d7591104d4908a9`.

Automatic inference is 168.729 ms, RTF 0.0490982,
peak process working set 433 MiB.
Forced inference is 159.688 ms, RTF 0.0464672,
peak process working set 433 MiB.

### it.

Prompt: Ciao. Oggi il cielo è sereno e la luce è delicata.

Observed automatic and forced transcript: Ciao, oggi il cielo è sereno e la luce è delicata.

Input duration is 3.483000 seconds. WAV SHA-256 is
`0fa7b9f1b8a582abae94ca23b7914de4dc66e275b448e57def683a7ccc7becfd`.

Automatic inference is 163.765 ms, RTF 0.0470183,
peak process working set 433 MiB.
Forced inference is 161.245 ms, RTF 0.0462949,
peak process working set 433 MiB.

### nl.

Prompt: Hallo. Vandaag is de lucht helder en het licht is zacht.

Observed automatic and forced transcript: Hallo, vandaag is de lucht helder en het licht is zacht.

Input duration is 3.297250 seconds. WAV SHA-256 is
`d2c0089973b8d62ef093d81b93c60d0c5c044345e7da7acb1479143976771abe`.

Automatic inference is 131.029 ms, RTF 0.0397388,
peak process working set 432 MiB.
Forced inference is 129.424 ms, RTF 0.0392522,
peak process working set 433 MiB.

### pl.

Prompt: Dzień dobry. Dzisiaj niebo jest bezchmurne, a światło jest łagodne.

Observed automatic and forced transcript: Dzień dobry, dzisiaj niebo jest bezchmurne, a światło jest łagodne.

Input duration is 4.226063 seconds. WAV SHA-256 is
`8356c1fb3ff392a0f099a5f95e3ce1a6ee21cbb430f945a8be25d536395bc94b`.

Automatic inference is 241.06 ms, RTF 0.0570413,
peak process working set 434 MiB.
Forced inference is 225.295 ms, RTF 0.0533109,
peak process working set 434 MiB.

## Quantized reference comparison

The cached official `needle.exe` reference ran on all five WAVs with automatic
language selection and four threads. All five invocations exit 0. It reports
`fr`, `es`, `it`, `nl`, and `pl` respectively. French, Spanish, Italian, and Dutch
reference text exactly matches the FP32 CLI text recorded above.

The Polish reference output differs:

```text
Dzień dobry, dzisiaj nie bo jest bezchmurne, a światło jest łagodne.
```

The FP32 output uses `niebo`, matching the synthesis prompt. The quantized
reference splits this as `nie bo`, producing two normalized word edits.
This discrepancy is retained, not described as exact transcript parity.
Neither transcript is an independent human annotation of the synthesized audio.

The reference engine source revision remains unknown. The `.cact` bytes match
the pinned revision and hash recorded in the validation report. These are transcript
comparisons against quantized official weights, not FP32 numerical parity.
Exact argument lists, exit codes, raw JSON outputs, and timing remain in
`reference-runs/results.json`. Each reference call uses this shape:

```powershell
& C:/Users/adityasharma/Projects/ansible-windows/.runtime/whistle/reference/needle.exe --model C:/Users/adityasharma/Projects/ansible-windows/.runtime/whistle/reference/whistle.cact --audio C:/Users/adityasharma/Projects/audio.cpp/build/whistle/pr-prep/multilingual/fr.wav --threads 4
```

## Synthesis settings and commands

Measured generation uses voice `JBFqnCBsd6RMkjVDRZzb` and model
`eleven_multilingual_v2`, with seed 42, stability 0.5, similarity boost 0.75,
style 0, speed 1, and speaker boost enabled. No `language_code` is passed.
The service model revision is not pinned; seeded regeneration is not guaranteed
byte-identical. Recorded audio hashes identify the actual tested artifacts.

Exact generation invocations are preserved in `generation-manifest.json`.
The same commands with resolved output paths are:

```powershell
elevenlabs text-to-speech convert --voice-id JBFqnCBsd6RMkjVDRZzb --model-id eleven_multilingual_v2 --output-format pcm_16000 --seed 42 --voice-settings.stability 0.5 --voice-settings.similarity-boost 0.75 --voice-settings.style 0 --voice-settings.speed 1 --voice-settings.use-speaker-boost true --text "Bonjour. Aujourd'hui, le ciel est clair et la lumière est douce." --output "build/whistle/pr-prep/multilingual/fr.pcm"
elevenlabs text-to-speech convert --voice-id JBFqnCBsd6RMkjVDRZzb --model-id eleven_multilingual_v2 --output-format pcm_16000 --seed 42 --voice-settings.stability 0.5 --voice-settings.similarity-boost 0.75 --voice-settings.style 0 --voice-settings.speed 1 --voice-settings.use-speaker-boost true --text "Hola. Hoy el cielo está despejado y la luz es suave." --output "build/whistle/pr-prep/multilingual/es.pcm"
elevenlabs text-to-speech convert --voice-id JBFqnCBsd6RMkjVDRZzb --model-id eleven_multilingual_v2 --output-format pcm_16000 --seed 42 --voice-settings.stability 0.5 --voice-settings.similarity-boost 0.75 --voice-settings.style 0 --voice-settings.speed 1 --voice-settings.use-speaker-boost true --text "Ciao. Oggi il cielo è sereno e la luce è delicata." --output "build/whistle/pr-prep/multilingual/it.pcm"
elevenlabs text-to-speech convert --voice-id JBFqnCBsd6RMkjVDRZzb --model-id eleven_multilingual_v2 --output-format pcm_16000 --seed 42 --voice-settings.stability 0.5 --voice-settings.similarity-boost 0.75 --voice-settings.style 0 --voice-settings.speed 1 --voice-settings.use-speaker-boost true --text "Hallo. Vandaag is de lucht helder en het licht is zacht." --output "build/whistle/pr-prep/multilingual/nl.pcm"
elevenlabs text-to-speech convert --voice-id JBFqnCBsd6RMkjVDRZzb --model-id eleven_multilingual_v2 --output-format pcm_16000 --seed 42 --voice-settings.stability 0.5 --voice-settings.similarity-boost 0.75 --voice-settings.style 0 --voice-settings.speed 1 --voice-settings.use-speaker-boost true --text "Dzień dobry. Dzisiaj niebo jest bezchmurne, a światło jest łagodne." --output "build/whistle/pr-prep/multilingual/pl.pcm"
```

An additional `Hello.` synthesis probe tested CLI binary output. It is retained
as `probe.pcm` and excluded from the five-language corpus. No corpus clips were
regenerated to hide recognition errors.

## Validation commands and artifacts

Commands run from the repository root:

```powershell
python build/whistle/pr-prep/multilingual/wrap_pcm.py
python build/whistle/pr-prep/multilingual/run_cli_multilingual.py
python -X utf8 build/whistle/pr-prep/multilingual/compare_transcripts.py > build/whistle/pr-prep/multilingual/root-comparison.json
```

The wrapping script refuses to overwrite existing WAVs. The validation script
preserves exact CLI argument lists and separate stdout/stderr logs in `runs3/`.
Each automatic call uses this shape; each forced call adds `--language <code>`:

```powershell
build/whistle/bin/audiocpp_cli.exe --task asr --family whistle_asr --backend cpu --threads 4 --model build/whistle/pr-prep/model/whistle-f32.gguf --audio build/whistle/pr-prep/multilingual/fr.wav --metrics --log
```

The FP32 weights use upstream revision
`b358ddadd89b7a713b5aa131f23032d3cca1b251`. Hardware, OS, build configuration,
and model setup are recorded in [the validation report](VALIDATION.md).
`reference-runs/` retains the five quantized reference runs.
CLI timing excludes model/session creation and input loading, but includes
request preparation. Whole-process elapsed time is recorded separately in
`runs3/results.json`; it is not used as inference latency.

All audio, scripts, hashes, and logs stay in ignored
`build/whistle/pr-prep/multilingual/`. `audio-inputs.json` records each WAV format
and hash. `multilingual-validation.json` records outputs, metrics, and comparisons.
Initial PowerShell process-monitoring attempts remain in `runs/` and `runs2/`.
The successful ten-case Python run is `runs3/`. No recordings or weights are
added to source changes.

## Post-review regression check

After the byte-fallback decoder fix, the rebuilt CLI repeated all ten requests.
All exit 0 and preserve the exact earlier transcripts. The successful command and
artifact paths are recorded in [the validation report](VALIDATION.md).
