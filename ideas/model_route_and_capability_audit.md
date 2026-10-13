# Model Route and Capability Audit

## Scope

This audit separates two concepts that should not be treated as interchangeable:

- **Routes/tasks** provide intuitive entry names such as `tts`, `clon`, `vdes`, `asr`, or `align`. Multiple names may intentionally alias the same model implementation.
- **Capabilities** describe what a model can do within a route, such as speaker conditioning, word timestamps, long-form processing, or emotion control.

A capability does not automatically justify another route. For example, word timestamps do not make an ASR model a forced aligner, and speaker-tagged transcripts do not make an ASR model a standalone diarization model.

This is a source audit of the current specs, documentation, advertised capabilities, and task checks. It is not runtime validation of every package.

## Normalization Rules

1. Preserve existing working routes. Route normalization should be additive unless a route was advertised but never implemented.
2. Add an alias only when it gives users an intuitive way to select an operation already supported by the model. Do not create a new execution path solely for route normalization.
3. `tts`, `clon`, and `vdes` may enter the same synthesis implementation. Existing request inputs decide whether the operation uses a preset voice, speaker reference, or instruction.
4. Adding an alias must not introduce route-specific required fields, defaults, validation, or output changes. Existing model behavior remains the source of truth.
5. Validate an alias with the same request through the old and new route. Both must reach the same implementation and produce equivalent output.
6. A speech generator should retain or expose `tts`; a synthesis path that already accepts speaker references may also expose `clon`, and one that already accepts voice instructions may also expose `vdes`.
7. Use `vc` for speech-to-speech target-speaker conversion, `svc` for singing voice conversion, and `s2s` for other semantic audio-to-audio transformations.
8. Use `align` only when the request consumes a transcript and aligns it to audio. ASR-generated timestamps stay under `asr`.
9. Use `diar` only for standalone speaker segmentation. Speaker turns emitted as part of transcription stay under `asr`.
10. Keep public generation routes specific: `music`, `sfx`, and `edit`. Their shared internal `gen` task is an implementation detail.
11. For multi-checkpoint families, route availability must be package-aware. A family-level union must not make the UI offer a route unsupported by the selected package.

## Action Items

Checklist convention: `[ ]` is pending. Mark an item `[x]` only after its spec,
runtime, documentation, and UI projection agree. Route changes also require a
focused request using the new route and a regression request using every
preserved route.

| Done | Priority | Family | Action | Acceptance criteria |
| --- | --- | --- | --- | --- |
| [x] | High | `fish_audio` | Accept `clon` through the existing synthesis implementation. | Validated `tts` and `clon` with identical request data on CUDA; both routes produced byte-identical output. |
| [x] | High | `higgs_audio_tts` | Accept `clon` through the existing synthesis implementation. | Validated `tts` and `clon` with identical request data on CUDA in offline and streaming modes; each route pair produced byte-identical output. |
| [ ] | High | `confucius4_tts` | Accept `tts` as an alias to the existing synthesis implementation. | Identical `clon` and `tts` requests preserve existing validation and output behavior. |
| [ ] | High | `echo_tts` | Accept `tts` as an alias to the existing synthesis implementation. | Identical `clon` and `tts` requests preserve existing validation and output behavior. |
| [ ] | High | `miotts` | Accept `clon` as an alias to the existing synthesis implementation. | Identical `tts` and `clon` requests preserve existing validation and output behavior. |
| [ ] | High | `omnivoice` | Accept the already-advertised `clon` and `vdes` aliases through the existing synthesis implementation. | Identical requests preserve existing validation and output behavior across all accepted aliases. |
| [ ] | High | `pocket_tts` | Accept `clon` as an alias to the existing synthesis implementation. | Identical `tts` and `clon` requests preserve preset and reference-conditioned behavior. |
| [ ] | High | `vieneu_v3_turbo` | Accept `clon` as an alias for the Base variant. | Identical Base requests preserve existing validation and output behavior through both aliases. |
| [ ] | High | `voxcpm1` | Accept `clon` as an alias to the existing synthesis implementation. | Identical `tts` and `clon` requests preserve existing validation and output behavior. |
| [ ] | High | `voxcpm2` | Accept the already-advertised `clon` and `vdes` aliases through the existing synthesis implementation. | Identical requests preserve existing validation and output behavior across all accepted aliases. |
| [ ] | High | `qwen3_tts` | Add aliases only where the selected package already supports the corresponding inputs. | Alias selection does not change package behavior, validation, or output; package-aware UI projection remains accurate. |
| [ ] | High | `sopro_tts` | Preserve the requested alias in session identity instead of always reporting `tts`. | `tts` and `clon` continue to use the same existing synthesis implementation. |
| [ ] | High | `chatterbox` | Accept the advertised `tts` alias through the existing cloning implementation. | Identical `tts` and `clon` requests preserve existing validation and output behavior; `vc` remains unchanged. |
| [ ] | Medium | `auk` | Add `clon` as an alias to the existing TTS implementation. | Identical `tts` and `clon` requests preserve existing validation and output behavior; `edit` remains unchanged. |
| [ ] | Medium | `maya1` | Add `vdes` as an alias to the existing instruction-capable TTS implementation. | Identical `tts` and `vdes` requests preserve existing validation and output behavior. |
| [ ] | Medium | `fireredtts3` | Make route exposure package-aware. | Each checkpoint variant exposes only routes accepted by that variant. |
| [ ] | Medium | `vieneu_v3_turbo` | Make route exposure package-aware. | Base and any non-Base variant do not inherit unsupported routes from the family union. |
| [ ] | Low | `miocodec` | Remove the orphan capability-key mismatch. | `encode_decode` is described under `s2s`, or under a future generic artifact contract; no model-only route is introduced. |
| [ ] | Low | `moss_voicegen` | Rename capability key `vdes` to public spec label `design`. | Runtime task mapping remains `vdes`; only spec metadata changes. |
| [ ] | Cross-cutting | UI/spec projection | Consume package-aware routes and remove route-specific family exceptions made obsolete by the items above. | Selecting a package shows only its valid routes; capability badges do not create routes. |

## Per-Model Audit

This is the evidence index behind the checklist. “Keep” means the model audit is
complete and no route action is needed. Every other disposition has a pending
checkbox in **Action Items** above. “Align runtime” means the spec already
advertises the route but current task checks or session identity do not
consistently honor it.

### TTS and Speech Generation

| Family | Current routes | Relevant announced capabilities | Audit disposition |
| --- | --- | --- | --- |
| `audio8_tts` | `tts`, `clone` | speaker reference | Keep. |
| `auk` | `tts`, `edit` | speaker reference; prompt editing | Add `clone` as an alias through the existing TTS path without changing input validation. |
| `breeze_tts` | `tts`, `clone`, `design` | style, speaker reference, voice design | Keep. |
| `chatterbox` | `tts`, `clone`, `vc` | speaker reference | Align runtime so `tts` aliases the existing cloning implementation; keep `vc` unchanged. |
| `chatterbox_turbo` | `tts` | none | Keep. Do not infer cloning from the related Chatterbox family. |
| `confucius4_tts` | `clone` | speaker reference; long form | Add `tts` as an alias to the existing synthesis implementation without changing input validation. |
| `cosyvoice3` | `tts`, `clone` | speaker reference; style | Keep. |
| `dots_tts` | `tts`, `clone` | speaker reference; style; long form | Keep. |
| `dramabox` | `tts`, `clone` | speaker reference; style; long form | Keep. Do not expose the internal `gen` acceptance as another public route without a distinct request contract. |
| `echo_tts` | `clone` | speaker reference | Add `tts` as an alias to the existing synthesis implementation without changing input validation. |
| `f5_tts` | `tts`, `clone` | speaker reference | Keep. Both routes already enter the same synthesis implementation. |
| `firered_audio` | `asr`, `tts`, `clone`, `design`, `edit` | speaker reference; style; voice design; prompt editing | Keep; these are distinct user workflows. |
| `fireredtts3` | `tts`, `clone`, `design` | speaker reference; style; voice design | Keep family union, but project routes by checkpoint variant. |
| `fish_audio` | `tts`, `clone` | speaker reference | Align runtime to accept the advertised `clon` route; preserve `tts`. |
| `glm_tts` | `tts`, `clone` | speaker reference | Keep. |
| `higgs_audio_tts` | `tts`, `clone` | speaker reference | Align runtime to accept the advertised `clon` route; preserve `tts`. |
| `index_tts2` | `tts`, `clone` | speaker reference; emotion | Keep. Emotion remains a capability. |
| `inflect_v2` | `tts` | long form | Keep. |
| `irodori_tts` | `tts`, `clone`, `design` | speaker reference; voice design | Keep. |
| `kitten_tts` | `tts` | built-in voices; long form | Keep. |
| `kitten_tts2` | `tts`, `clone` | built-in voices; speaker reference; long form | Keep. |
| `kokoro_tts` | `tts` | built-in voices; long form | Keep. |
| `kugelaudio` | `tts` | long form | Keep. |
| `lfm2_audio` | `asr`, `tts`, `s2s` | VAD chunking; built-in voices; long form | Keep. The three routes expose distinct modalities. |
| `magpie_tts` | `tts` | long form | Keep. |
| `maya1` | `tts` | voice design; emotion; long form | Add `design` as an alias to the existing instruction-capable TTS path; preserve `tts`. Do not create a route for emotion control. |
| `minimax_h3` | `tts`, `music`, `sfx` | style; multi-speaker; instrumental; prompt generation | Keep. Shared implementation does not make the public operations redundant. |
| `miotts` | `tts`, `clone` | speaker reference | Align runtime so `clon` aliases the existing synthesis implementation; preserve `tts`. |
| `mira_tts` | `tts`, `clone` | speaker reference | Keep. |
| `moss_tts_local` | `tts`, `clone` | speaker reference | Keep. |
| `moss_tts_nano` | `tts`, `clone` | speaker reference | Keep. |
| `moss_tts_v15` | `tts`, `clone` | speaker reference; style; long form | Keep. |
| `moss_ttsd` | `tts`, `clone` | speaker reference; multi-speaker; long form | Keep. Multi-speaker is a capability, not another route. |
| `moss_voicegen` | `design` | style conditioning | Keep route; rename capability key from `vdes` to public spec label `design`. |
| `neutts` | `tts` | built-in voices; emotion; long form | Keep. |
| `omnivoice` | `tts`, `clone`, `design` | speaker reference; voice design | Align runtime so advertised `clon` and `vdes` alias the existing synthesis implementation; preserve `tts`. |
| `outetts` | `tts`, `clone` | speaker reference | Keep. |
| `personaplex` | `s2s` | speaker reference | Keep. It is speech-to-speech interaction, not a plain TTS route. |
| `piper_tts` | `tts` | long form | Keep. |
| `pocket_tts` | `tts`, `clone` | speaker reference | Align runtime so advertised `clon` aliases the existing synthesis implementation; preserve `tts`. |
| `qwen3_tts` | `tts`, `clone`, `design` | speaker reference; voice design | Add aliases only to packages whose existing synthesis path accepts the corresponding inputs; do not branch behavior on the route token. |
| `sanotts` | `tts` | long form | Keep. |
| `soprano_tts` | `tts` | long form | Keep. |
| `sopro_tts` | `tts`, `clone` | speaker reference | Preserve the requested alias in session identity while both routes use the same existing synthesis implementation. |
| `supertonic` | `tts` | built-in voices; long form | Keep. |
| `vibevoice` | `tts` | multi-speaker; long form | Keep. Multi-speaker synthesis remains TTS. |
| `vieneu_v3_turbo` | `tts`, `clone` | speaker reference | Align Base runtime so `clon` aliases the existing synthesis implementation; preserve `tts` and project variant-specific routes. |
| `voxcpm1` | `tts`, `clone` | speaker reference | Align runtime so advertised `clon` aliases the existing synthesis implementation; preserve `tts`. |
| `voxcpm2` | `tts`, `clone`, `design` | speaker reference; voice design | Align runtime so `clon` and `vdes` alias the existing synthesis implementation; preserve `tts`. |
| `voxtral_tts` | `tts` | built-in voices; long form | Keep. |
| `zipvoice` | `tts`, `clone` | speaker reference | Keep. |

### ASR, Alignment, Diarization, and Speech Analysis

| Family | Current routes | Relevant announced capabilities | Audit disposition |
| --- | --- | --- | --- |
| `ast_audioset` | `classify` | class scores | Keep. |
| `audio8_asr` | `asr` | none | Keep. |
| `audio_flamingo` | `asr` | none | Keep. Audio understanding currently shares the ASR text-output contract. |
| `canary_asr` | `asr` | none | Keep. |
| `ced` | `classify` | class scores | Keep. |
| `citrinet_asr` | `asr` | none | Keep. |
| `cohere_asr` | `asr` | none | Keep. |
| `confucius4_r2t2` | `asr` | partial results | Keep. |
| `crisperwhisper` | `asr`, `align` | word timestamps; partial results | Keep. It implements both recognition and transcript-conditioned alignment. |
| `firered_vad` | `vad` | speech segments | Keep. |
| `fun_asr_nano` | `asr` | none | Keep. |
| `gigaam_asr` | `asr` | word timestamps; VAD chunking | Keep. Do not add `align`. |
| `granite5asr` | `asr` | VAD chunking; partial results | Keep. |
| `higgs_audio_stt` | `asr` | none | Keep. |
| `hviske_asr` | `asr` | none | Keep. |
| `hviske_asr_v6` | `asr` | none | Keep. |
| `index_echo` | `asr` | none | Keep. |
| `kroko_asr` | `asr` | words; segments; partial results | Keep. Do not add `align`. |
| `mms_forced_aligner` | `align` | word timestamps | Keep. |
| `moonshine_asr` | `asr` | VAD chunking | Keep. |
| `moss_transcribe_diarize` | `asr` | segments; speaker turns | Keep. Speaker-attributed transcription does not imply standalone `diar`. |
| `nemotron_3_diar` | `diar` | speaker turns | Keep. |
| `nemotron_asr` | `asr` | none | Keep. |
| `niagara_asr` | `asr` | none | Keep. |
| `owsm` | `asr` | segments | Keep. |
| `owsm_ctc` | `asr` | none | Keep. |
| `parakeet_tdt` | `asr` | words; partial results; VAD chunking | Keep. Do not add `align`. |
| `qwen3_asr` | `asr` | words; VAD chunking; partial results | Keep. Do not add `align`. |
| `qwen3_forced_aligner` | `align` | word timestamps | Keep. |
| `samsone` | `asr` | none | Keep. |
| `sense_asr` | `asr` | VAD chunking; partial results | Keep. |
| `smart_turn` | `turn` | turn decision | Keep. |
| `sortformer_diar` | `diar` | speaker turns | Keep. |
| `sortformer_diar_v2` | `diar` | speaker turns | Keep. |
| `vibeasr` | `asr` | none | Keep. |
| `vibevoice_asr` | `asr` | segments; speaker turns; VAD chunking | Keep. Do not add `diar`. |
| `vibevoice_asr_streaming` | `asr` | speaker turns | Keep. Do not add `diar`. |
| `voxtral_realtime` | `asr` | partial results | Keep. |
| `whistle_asr` | `asr` | none | Keep. |

### Music and General Audio Generation

| Family | Current routes | Relevant announced capabilities | Audit disposition |
| --- | --- | --- | --- |
| `ace_step` | `music`, `edit` | lyrics; prompt editing | Keep. |
| `controlfoley` | `sfx` | prompt generation | Keep. |
| `heartmula` | `music` | lyrics; style | Keep. |
| `liveavatar` | `sfx` | prompt generation | Keep for compatibility; a future dedicated audio-video route should be framework-wide, not model-specific. |
| `midashenglm_gen` | `music` | instrumental; style | Keep. |
| `minimax_music3` | `music` | lyrics; style | Keep. |
| `stable_audio` | `music`, `sfx`, `edit` | lyrics; prompt generation; prompt editing | Keep. |
| `yue2` | `music` | lyrics; style | Keep. |

### Conversion, Enhancement, Separation, and Utilities

| Family | Current routes | Relevant announced capabilities | Audit disposition |
| --- | --- | --- | --- |
| `apollo` | `s2s` | audio enhancement | Keep. |
| `audiosr` | `s2s` | audio enhancement | Keep. |
| `bs_roformer` | `sep` | stems | Keep. |
| `builtin_audio_utils` | `s2s` | audio enhancement | Keep. |
| `htdemucs` | `sep` | stems | Keep. |
| `htdemucs_6stems` | `sep` | stems | Keep. |
| `meanvc2` | `vc` | speaker reference | Keep. |
| `mel_band_roformer` | `sep` | stems | Keep. |
| `micro_wake_word` | `wake_word` | speech segments; stream events | Keep. |
| `miocodec` | `vc`, `s2s` | encode/decode; speaker reference | Keep routes; move the orphan `codec` capability under `s2s` or a future generic artifact contract. |
| `mossformer2` | `sep` | stems | Keep. |
| `muscriptor` | `midi` | none | Keep. |
| `pulsevad` | `vad` | speech segments | Keep. |
| `reuse` | `s2s` | audio enhancement | Keep. |
| `rvc` | `vc` | none | Keep. |
| `sam_audio` | `s2s` | none | Keep. Its target/residual transformation is not plain TTS or VC. |
| `seed_vc` | `vc`, `svc` | speaker reference; singing | Keep. |
| `sheetsage2` | `midi` | MIDI artifact | Keep. |
| `sherpa_kws` | `wake_word` | speech segments; stream events | Keep. |
| `sidon` | `s2s` | audio enhancement | Keep. |
| `tf_gridnet` | `sep` | stems | Keep. |
| `tone_color_vc` | `vc` | speaker reference | Keep. |
| `universr` | `s2s` | audio enhancement | Keep. |
| `vevo2` | `tts`, `music`, `vc`, `edit`, `svc`, `s2s` | lyrics; speaker reference; singing; prompt editing | Keep. The broad route set corresponds to distinct pipelines; do not add `clone` solely because several routes accept speaker conditioning. |

## Implementation Order

- [ ] **Phase 1: metadata-only corrections.** Complete `moss_voicegen` and
  `miocodec` without changing runtime behavior.
- [ ] **Phase 2: synthesis-route aliases.** Complete Fish Audio, Higgs Audio
  TTS, Confucius4 TTS, Echo-TTS, MioTTS, OmniVoice, PocketTTS, Sopro TTS,
  VieNeu v3 Turbo, VoxCPM1, and VoxCPM2 without adding route-specific behavior.
- [ ] **Phase 3: package-aware projection.** Complete Qwen3 TTS,
  FireRedTTS3, and VieNeu v3 Turbo before the UI consumes family route unions.
- [ ] **Phase 4: additive routes.** Add only the two new high-value routes found
  by this audit: `clon` for AuK and `vdes` for Maya1.
- [ ] **Phase 5: UI and documentation.** Update grouping and examples after
  runtime route acceptance is correct. Do not use UI exceptions to mask a
  spec/runtime disagreement.
