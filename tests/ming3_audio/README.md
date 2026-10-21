# Ming3 Audio Tests

Use the Debug binaries in `build/debug`, conda `qwen3-tts`, and saved `--log`
output. Run measurements sequentially. Test CUDA before Vulkan; no CPU inference
is required by this port.

## Official Fixtures

Clone [Ming-3.0-audio-flash](https://github.com/inclusionAI/Ming-3.0-audio-flash)
into `reference/Ming-3.0-audio-flash`. Download the pinned
[Tiny checkpoint](https://huggingface.co/inclusionAI/Ming-3.0-audio-tiny/tree/25b96e30be3739849bf76c0f48ac11db4a262fb3)
and make it available at `reference/Ming-3.0-audio-tiny` (a symlink is sufficient).
The reference voice and exact transcript come from that checkpoint's
`voice_presets` directory.

The CLI consumes WAV, so decode the two complete MP3 fixtures once for both
implementations. Do not trim, resample, or shorten them:

```bash
ffmpeg -i reference/Ming-3.0-audio-flash/examples/audio/meeting.mp3 \
  -c:a pcm_f32le reference/Ming-3.0-audio-flash/examples/audio/meeting.wav
ffmpeg -i reference/Ming-3.0-audio-flash/examples/audio/music.mp3 \
  -c:a pcm_f32le reference/Ming-3.0-audio-flash/examples/audio/music.wav
```

| Upstream example | Case / Request |
| --- | --- |
| Short ASR | `official_asr`: `ant_group`, `ant_group_repeat` |
| Native Thinker + Talker, US_Stella | `official_spoken_qa`: `ip_us_stella` |
| Spoken QA with reference audio and transcript | `official_spoken_qa`: `reference_us_james` |
| Spoken QA with the checkpoint's Chinese reference preset | `official_spoken_qa`: `reference_lingkai` |
| Complete music understanding fixture | `official_understanding`: `music` |
| Complete meeting transcription and summary | `official_understanding`: `meeting` |
| Complete meeting, thinking enabled | `official_understanding`: `meeting_thinking` |
| Bundled tool-calling audio asset | `official_asr`: `tool_audio_transcription` (input/transcription coverage only, not a structured tool-call API) |

The short ASR case reproduces the literal prompt and 256-token limit. Upstream's
10-gram repetition filter cannot activate on its six-token result; this test
does not claim a native repetition-filter implementation. The vLLM/SGLang server
and streaming transports are separate from this offline model integration.

Q8_0 quantizes only Thinker expert banks. Q4_K also uses a Q8_0 text output
head, reducing GPU-resident weights during expert offload. Token embeddings,
attention and dense/shared MLP projections retain source precision.
Both keep routers F32 and
preserve the source precision of the audio encoder, Talker, VAE and speaker
encoder. Broader projection quantization caused repetition or prolonged
generation on the full meeting cases.

## CUDA Runs

Run the same cases independently for BF16, Q8_0, and Q4_K. Set `GGUF` to the
selected standalone file and use a distinct output directory for each:

```bash
conda run --no-capture-output -n qwen3-tts \
  python tools/audiocpp_cli/run_audiocpp_cli_path_tests.py \
  --cases tests/ming3_audio/cli_official_cases.json \
  --audiocpp-cli-bin build/debug/bin/audiocpp_cli --model-path "$GGUF" \
  --backend cuda --model-spec-override embedded --measure-resources \
  --out-root "$OUTPUT" --log
```

`--only official_asr`, `--only official_spoken_qa`, or
`--only official_understanding` selects an individual group. For expert offload,
append `--session-option ming3_audio.disk_experts=true`. The selected GGUF must
embed the current spec before using newly added options without a spec override.

Also run `cli_asr_cases.json`, `cli_audio_lifecycle_cases.json`,
`cli_reference_cases.json`, and `cli_long_tts_cases.json` for longer ASR,
complete music/short-speech alternation, changed reference voices, and chunked
long-short-long generation. Compare repeated WAVs and transcribe generated audio;
a successful exit alone does not prove valid output or cache isolation.
For offload cache growth, run `cli_context_lifecycle_cases.json` with
`ming3_audio.disk_experts=true` and `--resource-sample-ms 20`. It repeats the
complete official meeting around a short request, retaining the 32,768-token
generation allowance. Check that the cache grows, both meeting responses match,
and the short transcript is unchanged after the meeting.
Review complete `stdout.log` responses for multiline text: the runner's
`text_outputs.json` currently records only their first line. The CLI's
`--text-out` also saves complete text when running cases directly.

## Official Python Boundary

`python_reference.py` uses the upstream example's message builder and processing
function. Pass a runner-produced `requests.json` to exercise exactly the same
prompts, audio, token limits, and thinking flag. `--capture` saves the backbone's
actual merged prefill input and final output, not separately recomputed
embeddings or per-step logits. Do not use captured runs as performance measurements.

`ming3_audio_thinker_parity` accepts `--capacity` to hold context allocation fixed,
`--prefill-tokens` to feed that same fixture in smaller pieces, and
`--disk-experts` to test staged experts. These are component diagnostics, not
standalone-GGUF acceptance tests. Compare completed generated-token sequences;
do not use internal logits or shortened generation limits as a quality pass.

`python_talker_reference.py` captures the Talker input and its final latents and
waveform. `ming3_audio_talker_parity` teacher-forces those final latents into the
native VAE and retains the strict waveform gate. Record any explicitly accepted
BF16 exception separately rather than changing that threshold.

For reference-conditioned fixtures, pass `--reference-voice`, `--reference-text`,
`--text`, and `--language` to the same parity test. Capture includes prompt IDs,
speaker features, reference latents, and the initial CUDA RNG offset. Optional
`--generated-output` runs normal generation from those reference features;
`--official-inputs` selects the captured rather than native-prepared prompt
embeddings for that control. These controls do not replace end-to-end tests with
native reference encoding. Input mismatch and the strict waveform gate remain
test failures even when diagnostic WAVs are written.
