| Model | Option / area | Before | After |
|---|---|---|---|
| All renamed options below | Compatibility | Existing spellings | Legacy spellings remain accepted; discovery uses canonical names. |
| Audio Flamingo, AuK, Maya1, MOSS-Transcribe-Diarize, SAMSONE | Instruction | `instruct` | `instruction`; task-specific meaning unchanged. |
| MOSS-TTS v1.5, MOSS-TTSD, MOSS-VoiceGen, DotTTS | Instruction discovery | Legacy or duplicate instruction names | Advertise only `instruction`. |
| Inflect v2 | Speaking speed | `speaking_rate` | `speed`; larger values remain faster. |
| sanoTTS | Speaking speed | `speaking_rate` multiplies duration | `speed` uses the reciprocal: `speed=2` equals legacy `speaking_rate=0.5`. Legacy behavior unchanged. |
| Chatterbox Turbo, VibeASR | Generated-token limit | `max_new_tokens` | `max_tokens` |
| Index Echo | Transcript-token limit | `max_text_tokens` | `max_tokens` |
| FireRed Audio | Understanding-token limit | `max_new_tokens` | `max_tokens` |
| FireRed Audio | Generation limits | `max_new_text_tokens`, `max_new_audio_steps`, `min_new_audio_steps` | `max_text_tokens`, `max_audio_steps`, `min_audio_steps`; independent limits retained. |
| OWSM | Beam width | `beam_size` | `num_beams` |
| ZipVoice | Language / timestep shift | `lang`, `t_shift` | `language`, `shift`; session shift remains family-prefixed. |
| ACE-Step | Duration / LM guidance / thinking | `duration_seconds`, `lm_cfg_scale`, `thinking` | `duration_sec`, `lm_guidance_scale`, `enable_thinking` |
| F5-TTS | Guidance discovery | Both `cfg_strength` and `guidance_scale` | Advertise only `guidance_scale`. |
| Chatterbox | Flow guidance | `s3gen_cfg_rate` | `flow_guidance_scale`; separate from T3 `guidance_scale`. |
| Chatterbox | Sampling discovery | Both `greedy` and `do_sample` | Advertise `do_sample`; legacy `greedy=true` still overrides sampling. |
| FunASR Nano, LFM2 Audio | ASR chunk duration | `audio_chunk_seconds` | `audio_chunk_duration_sec` |
| CLI | `--audio-chunk-seconds` request mapping | Emits `audio_chunk_seconds` | Emits `audio_chunk_duration_sec`; flag spelling unchanged |
| LFM2 Audio | Session one-pass audio limit | `lfm2_audio.max_pass_seconds` | `lfm2_audio.max_pass_duration_sec`; unchanged 120-second default. |
| Sopro TTS | Generation duration / reference crop | `min_seconds`, `max_seconds`, `ref_seconds` | `min_duration_sec`, `max_duration_sec`, `reference_duration_sec` |
| Vevo2 | Target duration / reference trim | `target_duration_seconds`, `reference_duration_seconds` | `duration_sec`, `reference_duration_sec` |
| Irodori TTS | Reference trim | `max_ref_sec` | `reference_duration_sec`; zero still selects the full reference. |
| Silero VAD | Speech duration limit | `max_speech_duration_s` | `max_speech_duration_sec` |
| MOSS-TTS v1.5, MOSS-TTSD, MOSS-VoiceGen | Request text temperature | Family-prefixed `text_temperature` | Unprefixed `text_temperature`; prefixed aliases retained. |
| LiveAvatar | Request memory policy | `memory_saver` | `mem_saver` |
| Stable Audio, MiniMax H3 | Sampler selection | `sampler` | `sampler_mode`; supported algorithms unchanged. |
| Qwen3 TTS | CustomVoice preset | `speaker` | `voice_id` |
| Canary ASR, Cohere ASR | Punctuation and casing | `pnc` | `punc`; still controls both punctuation and capitalization. |
| Hviske ASR | Punctuation and casing | `punctuation` | `punc`; still controls both punctuation and capitalization. |
| Hviske v6 | Punctuation | `punctuated` | `punc`; casing remains separate. Supply both `punc` and `cased` to override automatic formatting. |
| PocketTTS | Input tokens per text chunk | `max_tokens` | `text_chunk_tokens`; legacy `max_tokens` remains an input-token budget, not a generation limit. |
| PocketTTS | Generated audio-frame limit | `max_steps` | `max_frames`; zero still selects an estimated cap. |
| MOSS-TTS v1.5 | Requested duration in codec frames | `tokens` | `duration_frames`; separate from the `max_frames` ceiling. |
| IndexTTS2.5 | Duration multiplier | `duration_factor` | `duration_scale`; larger values still mean longer audio. |
| Seed-VC | Duration multiplier | `length_adjust` | `duration_scale`; not inverted into speed. |
| VieNeu v3 Turbo | Automatic frame cap / minimum text chunk | `frame_cap`, `text_chunk_min` | `auto_frame_cap`, `text_chunk_min_size`; the frame-cap switch remains boolean. |
| GigaAM | Session VAD location | Consumer read only `vad_model_path` | Also consume canonical `gigaam_asr.vad_model_path`. |
| Qwen3 ASR | Session VAD discovery | `qwen3_asr.vad_path` | `qwen3_asr.vad_model_path` |
| OuteTTS | Session aligner location | `outetts.aligner_path` | `outetts.forced_aligner_path` |
| Kokoro | Session predictor graph arenas | `kokoro_tts.predictor_{duration,text,tail}_graph_mb` | Corresponding `*_graph_arena_mb`; same host metadata allocations and defaults. |
| Vevo2 | Session graph arenas | `vevo2.{ar_prefill,ar_decode,whisper,tokenizer,fm,vocoder}_graph_context_mb` | Corresponding `*_graph_arena_mb`; sizes unchanged. |
| MiniMax Music3 | Session graph arena | `minimax_music3.graph_context_mb` | `minimax_music3.graph_arena_mb`; size unchanged. |
| PocketTTS | Session FlowLM/Mimi graph arenas | Component `*_graph_context_mb` | Corresponding `*_graph_arena_mb`; sizes unchanged. |
| VoxCPM1, VoxCPM2 | Session generation/AudioVAE graph arenas | Component `*_graph_context_mb` | Corresponding `*_graph_arena_mb`; sizes unchanged. |
| Seed-VC | Noise file discovery | `noise_path` | `noise_file` |
| GLM-TTS | Noise-file discovery | `flow_noise_path`, `hift_source_random_path` | `flow_noise_file`, `hift_source_random_file` |
| Irodori TTS | Speaker embedding file | `speaker_embedding_path` | `speaker_embedding_file` |
| PocketTTS | Voice-state files | `voice_embedding_path`, `export_voice_state_path` | `voice_embedding_file`, `export_voice_state_file` |
| RVC | Pitch / retrieval files | `pitch_path`, `retrieval_index_path` | `pitch_file`, `retrieval_index_file` |
| SAM Audio | Visual reference files | `reference_image_path`, `reference_video_path` | `reference_image_file`, `reference_video_file` |
| LiveAvatar | Reference image file | `reference_image_path` | `reference_image_file`; existing `ref_image` alias retained. |
| ControlFoley | Video conditioning file | `video` | `video_file` |
| VieNeu v3 Turbo | Session G2P dictionary file | `vieneu_v3_turbo.g2p_dict` / bare `g2p_dict` | `vieneu_v3_turbo.g2p_dict_file` |
| Soprano TTS | Session weight selection | Specs classified weights as load options; consumers read bare names | Consume `soprano_tts.backbone_weight_type` and `soprano_tts.decoder_weight_type`; classify as session options and retain bare aliases. |
| Soprano TTS | Weight defaults in specs | `native` despite F32 runtime defaults | Publish `f32`; runtime defaults unchanged. |
| Soprano TTS | Session text chunk size | Consumer read bare `text_chunk_size` | Also consume `soprano_tts.text_chunk_size` in offline and streaming paths. |
| MiraTTS | Session weight selection | Specs classified weights as load options; consumers read bare names | Consume `mira_tts.{backbone,linear,conv}_weight_type`; classify as session options, retaining bare aliases and Native/Native/F32 defaults. |
| FireRed Audio | Helper graph-arena discovery | Advertised `firered_audio.helper_graph_arena_mb` without a consumer | Remove from discovery; legacy setting remains accepted and ineffective. |
| CosyVoice3 | Convolution storage discovery | Advertised `cosyvoice3.conv_weight_type` without a consumer | Remove from discovery; legacy setting remains accepted and ineffective. Existing shared storage selection unchanged. |
| PocketTTS | Clone transcript discovery | Advertised `voice_clone_text`, unused by voice encoding | Remove from discovery; legacy input remains accepted with clone audio but does not condition the voice. |
| MiniMax Music3 | Depth-decoder default in specs | `rvq_depth_decoder_bf16.gguf` | `rvq_depth_decoder_q8_0.gguf`, matching the existing runtime default. |
| FireRed Audio | Template default / instruction description | Unconditional `tts_clone` metadata default; unclear task applicability | Document `asr` for ASR and `tts_clone` for generation. `instruction` conditions generation; ASR/understanding uses text input as its prompt. |
| Fish Audio, Voxtral Realtime | Token-limit help | `max_new_tokens` | Advertise existing canonical `max_tokens`. |
| Stable Audio | Duration and inpainting help | `duration_seconds`, `duration_padding_seconds`, `inpaint_mask_*_seconds` | Advertise existing corresponding `*_sec` names. |
| Supertonic | Speed help | Both `speaking_rate` and `speed` | Advertise only `speed`; legacy input retained. |
| MioTTS | Request selection help | `miotts.best_of_n`, `miotts.best_of_n_enabled`, `miotts.best_of_n_language` | Advertise unprefixed request names; session defaults remain family-prefixed. |
| VoxCPM2 | Request chunk-strategy help | `voxcpm2.chunk_strategy` | Advertise `chunk_strategy`; prefixed alias retained. |

## Numeric Metadata Correction Checklist

Specs and existing handwritten inspection entries were updated together where
those entries exist. Spec-backed inspection continues using spec metadata.
Useful local descriptions and additional options are retained; no new runtime
spec reads were introduced.

Numeric `min`/`max` boundaries are retained for discovery. Descriptions state
whether each endpoint is included: `> 0` excludes zero, `(0, 2)` excludes both
endpoints, and `(0, 1]` includes only the upper endpoint. Conditional descriptions
identify the mode in which a stricter bound applies; zero-valid greedy, disabled,
and whole-file behavior is preserved.

The current [CLI metadata](../../include/engine/framework/runtime/model.h#L43)
and [spec conversion](../../src/framework/model_spec/metadata.cpp#L169) only carry
inclusive scalar bounds. Those fields alone therefore do not encode the complete
acceptance predicate: clients must not treat a displayed minimum of 0 as a
promise that zero is accepted in every mode. Runtime checks remain authoritative.
No positive type, exclusive-bound fields, arbitrary epsilon, or conditional
schema extension was added. Integer bounds were corrected directly.

### Scalar Boundary Clarifications

Scope is request unless marked session. `> 0` below means finite and positive
where the consumer uses the positive-finite parser. Grouped options share the
stated issue, not necessarily their defaults or upper bounds.

| Done | Model / option | Previous metadata / spec | Runtime consumer / actual constraint | Fix |
|---|---|---|---|---|
| [x] | Qwen3 TTS `repetition_penalty` | Minimum 0 ([spec](../../model_specs/qwen3_tts.json#L104)) | Must be positive ([session](../../src/models/qwen3_tts/session.cpp#L76)). | Retained minimum 0 as a boundary; explicitly described `> 0` (zero excluded), with default unchanged. |
| [x] | Qwen3 TTS `temperature`, `subtalker_temperature` | Minimum 0; prose already describes sampling conditions ([spec](../../model_specs/qwen3_tts.json#L79), [subtalker](../../model_specs/qwen3_tts.json#L119)) | Positive when the corresponding `do_sample`/`subtalker_do_sample` is true ([session](../../src/models/qwen3_tts/session.cpp#L67)). | Retained minimum 0; named `do_sample` / `subtalker_do_sample` in the `> 0` conditions and documented greedy-mode behavior. |
| [x] | Qwen3 ASR `audio_chunk_duration_sec` | Minimum 0 ([spec](../../model_specs/qwen3_asr.json#L90)) | Positive in chunking and streaming paths ([offline](../../src/models/qwen3_asr/session.cpp#L675), [streaming](../../src/models/qwen3_asr/session.cpp#L558)). | Retained minimum 0; documented `> 0` when chunking/streaming, offline `none` behavior, and mode-specific defaults. |
| [x] | Audio8 TTS `top_p`, `temperature` | `[0,1]`, `[0,2]` ([spec](../../model_specs/audio8_tts.json#L81)) | `(0,1]`, `(0,2)` ([session](../../src/community_models/audio8_tts/session.cpp#L149)). | Retained numeric boundaries and existing exact `(0,1]` / `(0,2)` descriptions. No scalar spec change needed; zero and temperature 2 are explicitly excluded in prose. |
| [x] | Fish Audio `top_p`, `temperature` | `[0,1]`, `[0,2]` ([spec](../../model_specs/fish_audio.json#L67)) | `(0,1]`, `(0,2)` ([session](../../src/models/fish_audio/session.cpp#L136)). | Retained numeric boundaries and exact `(0,1]` / `(0,2)` descriptions; copied those intervals into local help. |
| [x] | BreezeTTS `temperature`, `depth_temperature`, `top_p` | Minimum 0 ([spec](../../model_specs/breeze_tts.json#L91)) | Positive-finite parser ([session](../../src/models/breeze_tts/session.cpp#L379)). | Retained bounds; documented temperatures `> 0` and top-p `(0,1]`. |
| [x] | Confucius4 TTS `temperature`, `top_p`, `repetition_penalty` | Minimum 0 ([spec](../../model_specs/confucius4_tts.json#L53)) | Temperature `> 0`, top-p `(0,1]` ([request](../../src/models/confucius4_tts/request.cpp#L62)); penalty `> 0` ([T2S](../../src/models/confucius4_tts/t2s.cpp#L415)). | Retained bounds; made temperature/penalty `> 0` explicit and retained top-p `(0,1]`. |
| [x] | ControlFoley `guidance_scale` | Minimum 0 ([spec](../../model_specs/controlfoley.json#L111)) | Positive parser ([pipeline](../../src/models/controlfoley/pipeline.cpp#L176)). | Retained bounds; documented guidance `> 0`. |
| [x] | DramaBox `reference_duration_sec`, `audio_chunk_threshold_sec`, `audio_chunk_duration_sec` | Minimum 0 ([spec](../../model_specs/dramabox.json#L90)) | All must be positive ([session](../../src/models/dramabox/session.cpp#L148)). | Retained bounds; documented all three as `> 0`, leaving crossfade unchanged. |
| [x] | FireRed Audio `guidance_scale` | Minimum 0 ([spec](../../model_specs/firered_audio.json#L87)) | Positive parser ([session](../../src/models/firered_audio/session.cpp#L121)). | Retained minimum 0; documented guidance `> 0`. |
| [x] | FireRedTTS3 `guidance_scale` | Minimum 0 ([spec](../../model_specs/fireredtts3.json#L146)) | Positive parser in base and instruct paths ([session](../../src/models/fireredtts3/session.cpp#L95)). | Retained minimum 0; documented guidance `> 0` for the shared option. |
| [x] | HeartMuLa `duration_sec`, `temperature`, `guidance_scale`, `codec_duration_sec`, `codec_guidance_scale` | Minimum 0 ([spec](../../model_specs/heartmula.json#L38)) | Positive parsers ([session](../../src/models/heartmula/session.cpp#L73)). | Retained bounds; explicitly described all five values as `> 0`. |
| [x] | Higgs Audio STT `audio_chunk_duration_sec` | Minimum 0 ([spec](../../model_specs/higgs_audio_stt.json#L56)) | Must be positive ([session](../../src/models/higgs_audio_stt/session.cpp#L340)). | Retained minimum 0; explicitly described duration `> 0`. |
| [x] | Irodori TTS `duration_scale`, `min_duration_sec`, `max_duration_sec` | Minimum 0 ([spec](../../model_specs/irodori_tts.json#L78)) | Scale and minimum duration positive; maximum at least minimum ([session](../../src/models/irodori_tts/session.cpp#L894)). | Retained bounds; explicitly described scale/minimum `> 0` and retained `max_duration_sec >= min_duration_sec`. Reference duration unchanged. |
| [x] | LiveAvatar `shift` | Minimum 0 ([spec](../../model_specs/liveavatar.json#L165)) | Must be positive ([pipeline state](../../src/community_models/liveavatar/pipeline_state.cpp#L419)). | Retained minimum 0; documented shift `> 0`. |
| [x] | Maya1 `temperature`, `top_p`, `repetition_penalty` | Minimum 0 ([spec](../../model_specs/maya1.json#L54)) | Positive; top-p also at most 1 ([session](../../src/models/maya1/session.cpp#L97)). | Retained bounds; documented temperature/penalty `> 0` and top-p `(0,1]`. |
| [x] | MiniMax Music3 `guidance_scale`, `ar_guidance_scale` | Minimum 0 ([spec](../../model_specs/minimax_music3.json#L53)) | Both must be positive ([pipeline](../../src/community_models/minimax_music3/pipeline.cpp#L157)). | Retained both minima; documented both guidance scales `> 0`. |
| [x] | NeuTTS `temperature` | Minimum 0 ([spec](../../model_specs/neutts.json#L84)) | Positive parser ([session](../../src/models/neutts/session.cpp#L99)). | Retained minimum 0; explicitly described temperature `> 0`. |
| [x] | OuteTTS `repetition_penalty` | Minimum 0 ([spec](../../model_specs/outetts.json#L85)) | Nonpositive values throw ([decoder](../../src/community_models/outetts/llama.cpp#L781)). | Retained minimum 0; explicitly described penalty `> 0`; greedy temperature unchanged. |
| [x] | PersonaPlex `temperature`, `text_temperature` | Minimum 0 ([spec](../../model_specs/personaplex.json#L65)) | Positive when `do_sample=true` ([request](../../src/models/personaplex/request.cpp#L106)). | Retained minimum 0; documented nonnegativity, sampling-only `> 0`, and greedy behavior. |
| [x] | Seed-VC `duration_scale` | Minimum 0 ([spec](../../model_specs/seed_vc.json#L60)) | Must be positive ([session](../../src/models/seed_vc/session.cpp#L332)). | Retained minimum 0; explicitly described scale `> 0`; V2 behavior unchanged. |
| [x] | Sortformer session `session_len_sec` | Minimum 0 ([spec](../../model_specs/sortformer_diar.json#L69)) | Must be positive ([postprocess](../../src/models/sortformer_diar/postprocess.cpp#L284)). | Retained minimum 0; explicitly described session length `> 0`; Sortformer v2 unchanged. |
| [x] | Soprano TTS `temperature` | Minimum 0; prose claims zero selects a default ([spec](../../model_specs/soprano_tts.json#L40)) | Explicitly rejects `<= 0` ([session](../../src/community_models/soprano_tts/session.cpp#L53)). | Retained minimum 0; removed the false zero/default claim and documented temperature `> 0`. |
| [x] | Voxtral Realtime `temperature`, `top_p` | Minimum 0 ([spec](../../model_specs/voxtral_realtime.json#L46)) | Temperature `> 0`, top-p `(0,1]` ([session](../../src/models/voxtral_realtime/session.cpp#L71)). | Retained bounds; explicitly described temperature `> 0` and retained top-p `(0,1]`, including local help. |
| [x] | MOSS-TTSD `temperature`, `repetition_penalty` | Minimum 0 ([spec](../../model_specs/moss_ttsd.json#L120)) | Passed to HF sampling ([decoder](../../src/framework/decoders/moss_tts_delay/delay_decoder.cpp#L102)); sampled temperature must be positive ([sampler](../../src/framework/sampling/hf_sampler.cpp#L381)), penalty must be positive when history is present ([sampler](../../src/framework/sampling/hf_sampler.cpp#L200)). | Retained bounds; documented temperature/penalty `> 0` and history-dependent penalty checking. |
| [x] | MiDashengLM `min_stop_step` | Minimum 0 ([spec](../../model_specs/midashenglm_gen.json#L59)) | Positive integer parser ([session](../../src/models/midashenglm_gen/session.cpp#L175)). | Changed integer minimum to 1. |

### Session Allocation Bounds

All entries below previously advertised minimum 0 but use `parse_size_mb_option`,
which rejects nonpositive sizes ([shared parser](../../src/framework/runtime/options.cpp#L272)).
Names omit their canonical session family prefix for readability.

| Done | Model | Options | Spec | Consumer | Fix |
|---|---|---|---|---|---|
| [x] | Audio8 TTS | `ar_graph_arena_mb`, `ar_weight_context_mb`, `codec_graph_arena_mb`, `codec_weight_context_mb` | [spec](../../model_specs/audio8_tts.json#L147) | [session](../../src/community_models/audio8_tts/session.cpp#L403) | Changed integer minimum to 1 MiB for every listed option; retained defaults. |
| [x] | Fish Audio | `ar_graph_arena_mb`, `ar_weight_context_mb`, `codec_graph_arena_mb`, `codec_weight_context_mb` | [spec](../../model_specs/fish_audio.json#L133) | [session](../../src/models/fish_audio/session.cpp#L325) | Changed integer minimum to 1 MiB for every listed option; retained defaults. |
| [x] | Higgs Audio STT | `audio_encoder_graph_arena_mb`, `text_decoder_prefill_graph_arena_mb`, `text_decoder_decode_graph_arena_mb`, `text_decoder_weight_context_mb` | [spec](../../model_specs/higgs_audio_stt.json#L89) | [session](../../src/models/higgs_audio_stt/session.cpp#L117) | Changed integer minimum to 1 MiB for every listed option; retained defaults. |
| [x] | Irodori TTS | `condition_graph_arena_mb`, `rf_graph_arena_mb`, `codec_graph_arena_mb`, `condition_weight_context_mb`, `rf_weight_context_mb`, `codec_weight_context_mb` | [arenas](../../model_specs/irodori_tts.json#L222), [weight contexts](../../model_specs/irodori_tts.json#L278) | [session](../../src/models/irodori_tts/session.cpp#L474) | Changed integer minimum to 1 MiB for every listed option; retained defaults. |

### Sentinels And Coupled Constraints

The first three rows are direct advertised-behavior mismatches. Remaining rows
are missing conditional, divisibility, or cross-option constraints, not evidence
that every advertised scalar zero is invalid.

| Done | Model / option | Previous metadata / spec | Actual consumer behavior | Fix |
|---|---|---|---|---|
| [x] | F5-TTS session `frame_budget` | Default/minimum 0; zero described as 2048 ([spec](../../model_specs/f5_tts.json#L110)) | Explicit values must be in `[256,8192]`; explicit zero throws ([session](../../src/community_models/f5_tts/session.cpp#L132)). | Set minimum 256, maximum 8192, and effective default 2048; documented that explicit zero is rejected. Internal omitted-value sentinel unchanged. |
| [x] | AudioSR `seed` | `-1` described as random ([spec](../../model_specs/audiosr.json#L76)) | Every negative value maps to fixed seed 42; zero is an ordinary seed ([pipeline](../../src/models/audiosr/pipeline.cpp#L35)). | Removed the false random claim and minimum -1; documented that all negative inputs and omission use fixed seed 42. Zero remains an ordinary seed. |
| [x] | MiDashengLM `seed` | `-1` described as random, minimum -1, maximum 2147483647 ([spec](../../model_specs/midashenglm_gen.json#L68)) | Unsigned 32-bit parsing rejects -1; omitted seed is 0 ([session](../../src/models/midashenglm_gen/session.cpp#L176)). | Set range `[0,4294967295]`; retained default 0 and documented fixed unsigned seed semantics. Downstream seed is uint32_t, widened to uint64_t for flow sampling. |
| [x] | LiveAvatar `height`, `width`, `num_frames` | Independent minimum values ([spec](../../model_specs/liveavatar.json#L50)) | Dimensions must be multiples of 16; blockwise frame count must be a multiple of 4 ([pipeline state](../../src/community_models/liveavatar/pipeline_state.cpp#L810)). | Documented multiples of 16 for dimensions and the multiple-of-4 frame constraint only for blockwise generation. |
| [x] | LiveAvatar `vae_decoder_tile_size` | Minimum 0, zero disables tiling ([spec](../../model_specs/liveavatar.json#L142)) | Enabled tiles must be multiples of 8 and larger than overlap ([VAE](../../src/community_models/liveavatar/pipeline_vae.cpp#L1143)). | Documented enabled tiles as multiples of 8 greater than the 64-pixel overlap; retained zero-disable. |
| [x] | MMS forced aligner session `emission_window_sec`, `emission_context_sec` | Missing upper/alignment constraints ([spec](../../model_specs/mms_forced_aligner.json#L71)) | Window positive, context nonnegative, both at most 3600 s; rounded 16 kHz sample counts must be divisible by 320 ([emissions](../../src/community_models/mms_forced_aligner/emissions.cpp#L176)). | Added maximum 3600 for both; documented rounded 16 kHz sample counts divisible by 320, preserving context zero and the context/window relation. |
| [x] | TFGridNet `audio_chunk_duration_sec`, `audio_chunk_overlap_sec` | Independent nonnegative bounds ([spec](../../model_specs/tf_gridnet.json#L30)) | Enabled chunk length must exceed `n_fft/2` samples; overlap must be positive and smaller than the chunk ([session](../../src/models/tf_gridnet/session.cpp#L119)). | Retained numeric bounds; documented nonnegativity, sample-based chunk/overlap constraints, and strictly positive overlap only when chunking. Whole-file duration zero retained. |
| [x] | Apollo `audio_chunk_duration_sec`, `audio_chunk_overlap_sec` | Independent nonnegative bounds; prose already states the half-chunk overlap limit ([spec](../../model_specs/apollo.json#L30)) | Enabled chunk length must exceed `n_fft/2` samples; overlap may not exceed half the chunk ([session](../../src/models/apollo/session.cpp#L67)). | Added the enabled chunk's sample-based minimum; retained existing half-chunk overlap prose and zero behavior. |
| [x] | RE-USE `audio_chunk_duration_sec`, `audio_chunk_overlap_sec` | Independent nonnegative bounds; prose already states overlap must be shorter ([spec](../../model_specs/reuse.json#L30)) | Enabled chunk length must exceed `sample_rate/50` samples; overlap smaller than chunk ([session](../../src/community_models/reuse/session.cpp#L63)). | Added the enabled chunk's sample-based minimum; retained existing overlap relation and zero behavior. |
| [x] | MossFormer2 `audio_chunk_duration_sec`, `audio_chunk_overlap_sec` | Independent nonnegative bounds ([spec](../../model_specs/mossformer2.json#L30)) | When chunking is enabled, overlap must be smaller than duration ([session](../../src/models/mossformer2/session.cpp#L46)). | Documented overlap smaller than duration when chunking; retained whole-file duration zero. |
| [x] | AudioSR chunk duration/overlap | Independent bounds ([spec](../../model_specs/audiosr.json#L67)) | Overlap must be smaller than duration, including after sample rounding ([pipeline](../../src/models/audiosr/pipeline.cpp#L456)). | Documented overlap smaller than duration, including after sample rounding. |
| [x] | Stable Audio Foundation `init_noise_level` | `[0,1]` ([spec](../../model_specs/stable_audio.json#L111)) | With init audio and k-diffusion, this becomes `sampler_sigma_max`, which must be positive ([assignment](../../src/models/stable_audio/foundation/session.cpp#L281), [check](../../src/models/stable_audio/foundation/session.cpp#L158)). | Documented the Foundation/init-audio/k-diffusion positivity condition; retained `[0,1]` for the shared option. |

Seed semantics remain model-specific. F5-TTS's default and the AudioSR/MiDashengLM
seed descriptions now reflect existing behavior; no new zero-default or random
sentinel was implemented. Any future runtime semantics change needs separate
authorization and validation; the equivalence results below do not cover it.

### Usage Documentation

- [x] Updated the usage tables for all 33 families in this numeric checklist, including shared [TTS](../tts.md), [music](../music_generation.md), and [speech analysis](../speech_analysis.md) guides: positive/exclusive bounds, conditional constraints, allocation minima, seed behavior, and the F5 frame-budget default.
- [x] Added missing Audio8/Fish allocation controls, MOSS-TTSD sampling controls, and LiveAvatar shift. Corrected [Music3 CFG semantics](../community_models/minimax_music3.md), [Seed-VC V2's duration no-op](../models/seed_vc.md), and the [general seed guidance](../usage.md).

### Remaining Discovery Corrections

| Done | Model / control | Fix |
|---|---|---|
| [x] | Echo-TTS text chunk size | Added canonical `text_chunk_size` to the spec with minimum 1 and default 300, allowing it through request-key validation to the existing consumer. Added a contract regression check and usage documentation. |
| [x] | MMS forced aligner audio chunking | Removed standalone audio-chunk flags from this model's alignment help; other alignment families retain their existing help. |
| [x] | MOSS-TTS-v1.5 / VoiceGenerator generation bounds | Added `min_frames` and `max_frames` to specs, local CLI tables, and usage docs. Documented automatic nonpositive values and the runtime-adjusted ceiling. |
| [x] | MiniMax-H3 cache window | Advertised 10%/95% progress controls; removed false literal sigma defaults and documented paired sigma overrides. Updated usage docs. |
| [x] | MOSS-TTS-v1.5 weight format | Corrected local help and the usage table to default native, preserving packaged quantization. |
| [x] | VoxCPM1 automatic retries | Removed the unconditional metadata default; documented true offline, false streaming, and streaming's warning/ignore behavior for explicit true. |
| [x] | Music3 ensemble takes | Added maximum 16 to local CLI metadata and documented the inclusive 1-16 range. |

## Runtime Validation Checklist

Checked rows state either a runtime option-equivalence pass or an exemption.
Existing GGUFs are used where available; supported safetensors and external-spec
packages are labeled explicitly. Unchecked rows retain missing coverage.
Output quality is separate from this refactor check: a pass means old/new names
reach the same consumers and behave consistently, not that generation quality
or pre-existing model behavior is correct.

| Done | Model | Runtime validation requirement |
|---|---|---|
| [x] | ACE-Step | CUDA Turbo Q8: all three old/new aliases produce identical 12-second WAVs under existing embedded/current specs. Thinking invokes planner code generation, guidance enables CFG, and disabling either changes output; duration yields 60 planner codes and 12 seconds. |
| [x] | Audio Flamingo | CUDA Flamingo 3 Q4_K: both instruction names produce identical summaries under embedded and updated specs; the default produces a transcription instead. |
| [x] | AuK | Flash Q8 aliases match byte-for-byte across old/current external specs; editing rejects both correctly. Unexpected opening phrase is recorded separately as an output-quality observation, not an alias failure. |
| [x] | Canary ASR | CUDA Q8: legacy `pnc` and canonical `punc` produce identical transcripts with the existing embedded spec and the updated spec. Disabling the control removes punctuation/capitalization; the default restores them. |
| [x] | Chatterbox | Clone flow aliases/greedy override match byte-for-byte across specs. VC aliases match at the same request position with reversed-order controls. Truncation and repeatability observations are separate from option equivalence. |
| [x] | Chatterbox Turbo | CUDA Q8: token-limit aliases produce identical complete speech under existing and updated specs. Reducing the cap from 512 to 64 shortens output from 6.00 to 2.68 s. |
| [x] | Cohere ASR | CUDA Q8: both punctuation aliases produce identical text with embedded and updated specs; the default restores punctuation/capitalization. |
| [x] | ControlFoley | CUDA existing Q8 GGUF, eight-second upstream skateboarding video: legacy video/canonical video_file produce identical WAVs under embedded/current specs. Video decoding and all visual conditioners execute; output differs from text-only control. |
| [x] | CosyVoice3 | CUDA Q8: legacy `conv_weight_type=f16` remains accepted and produces the same WAV as omitting it, under existing and updated specs. ASR recovers the complete requested text. |
| [x] | DotTTS | Edit Q8: old/new instructions perform the requested word replacement with existing/current specs. Reversed-order control gives exact waveform matches at the same request position. Streaming reaches the expected offline-only edit rejection. Repeatability caveat recorded. |
| [x] | F5-TTS | Not required: duplicate guidance discovery removed; parser unchanged. |
| [x] | FireRed Audio | CUDA Q8: ASR token aliases match under embedded/current specs and a four-token control truncates identically. Generation text/audio aliases produce identical complete speech. Removed helper-arena setting remains an accepted no-op. |
| [x] | Fish Audio | Not required: token-limit help only. |
| [x] | FunASR Nano | CUDA Q8: both duration spellings produce identical transcripts and two audio chunks with a three-second setting, under embedded and updated specs. |
| [x] | GigaAM | CUDA multilingual F16 GGUF: bare/prefixed VAD paths produce identical transcripts and two VAD-derived segment boundaries under existing and updated specs. |
| [x] | GLM-TTS | Valid Flow/HiFT noise files: old/new aliases produce identical full-sentence WAVs with the old embedded spec and current override. ASR recovers the full text; controls reach their consumers. |
| [x] | Hviske ASR | CUDA Q8, 16-second Danish sample: both punctuation aliases produce identical text under embedded and updated specs; default conditioning restores capitalization. |
| [x] | Hviske v6 | CUDA Q8, 16-second Danish sample: `punctuated=false` and `punc=false`, with `cased=false`, give identical text under embedded and updated specs. Enabling punctuation/casing changes formatting as expected. |
| [x] | Index Echo | CUDA 2B Q8: `max_text_tokens=96` and `max_tokens=96` give identical text/timestamps under existing and updated specs. An 8-token control truncates the cue and raises a malformed-subtitle error under both specs. |
| [x] | IndexTTS2.5 | CUDA Q8: `duration_factor=1.2` and `duration_scale=1.2` match under existing and updated specs; output grows from 6.548 to 7.848 s. ASR recovers the complete requested text. |
| [x] | Inflect v2 | CUDA original GGUF: `speaking_rate=1.25` and `speed=1.25` produce identical audio under existing and updated specs; 5.831 s versus default 6.492 s confirms the rate direction. ASR recovers the requested text. |
| [x] | Irodori TTS | Both alias pairs match byte-for-byte across old embedded/current specs; reference trimming and embedding loading execute. Embedding-conditioned extra tail is a separate output-quality observation. |
| [x] | Kokoro | CUDA Q8 offline: all three legacy/canonical arena names, with nondefault sizes, produce byte-identical WAVs under existing and updated specs. ASR recovers the requested text. |
| [x] | LFM2 Audio | CUDA upstream Q8 GGUFs require an explicit spec override (no embedded spec). With it, both chunk/one-pass aliases produce identical two-chunk transcripts; both enforce the limit and remain rejected on TTS/S2S where inapplicable. |
| [x] | LiveAvatar | Existing NVFP4/Q4/F16 package: all three image and memory-policy spellings produce byte-identical 45-frame videos. Nondefault false policy is logged in each run; sampled frames are valid and moving. Model-local parsing, not spec-backed validation. |
| [x] | Maya1 | CUDA Q8: both instruction names produce byte-identical WAVs under embedded and updated specs; ASR recovers the requested sentence. Missing instruction still raises the required-input error. |
| [x] | MiniMax H3 | CUDA existing Q4 component package: sampler=res_multistep and sampler_mode=res_multistep select the same nondefault sampler and produce identical 8.7-second audio. Canonical run also uses current spec override. GGUF has no embedded spec; no embedded-spec coverage claimed. |
| [x] | MiniMax Music3 | Existing Q4/Q8/Q4 package: old/new 48 MiB arena names produce identical 20-second WAVs and logged allocations. Lyrics mismatch is recorded separately. Spec overrides do not control this model-local loader. |
| [x] | MiraTTS | Native safetensors: bare/prefixed nondefault weights produce identical WAVs and buffers; canonical names also work with the pre-refactor external spec. Omitted defaults match explicit native/native/F32. No legacy-GGUF coverage claimed. Existing session.wall_ms logging gap recorded. |
| [x] | MioTTS | Not required: request-option help names only. |
| [x] | MOSS-Transcribe-Diarize | Instruction aliases match in offline and streaming with existing/current specs. Custom prompt omits timestamps under both names; default prompt produces a parsed speaker turn. |
| [ ] | MOSS-TTS v1.5 | Matching local package not located, nor found in audio-cpp's HF repositories. Configured package is third-party; not downloaded. Available MOSS-TTS-Local-v1.5 is a different family, not a substitute. |
| [x] | MOSS-TTSD | Instruction and text-temperature aliases produce byte-identical short dialogue across embedded/current specs; ASR recovers both sentences. |
| [x] | MOSS-VoiceGen | Instruction and text-temperature aliases produce byte-identical speech across embedded/current specs; ASR recovers the complete input. |
| [ ] | OuteTTS | Aligner-path aliases and file resolution remain unvalidated. No local package or audio-cpp-hosted package located; configured third-party package not downloaded. |
| [x] | OWSM | CUDA Q8: `beam_size=2` and `num_beams=2` produce identical transcripts offline and streaming, under embedded and updated specs. |
| [x] | PocketTTS | CUDA English Q8: token/frame limits, voice-state export/import, and five nondefault graph-arena aliases match exactly within offline/streaming modes. Offline old/new options also match across embedded/current specs. Two text chunks; an eight-frame control caps output at 0.64 s. ASR recovers both sentences with minor word differences. |
| [x] | Qwen3 ASR | CUDA 0.6B Q8: both VAD-path names load Silero and produce identical two-chunk transcripts under existing and updated specs. |
| [x] | Qwen3 TTS | CUDA CustomVoice Q8: `speaker=ryan` and `voice_id=ryan` override structured Aiden, matching the Ryan-only control and differing from Aiden. Offline results match across specs; streaming alias WAVs also match. Both modes transcribe to the requested text. |
| [x] | RVC | Existing F16 GGUF: valid pitch contour and real FAISS index, retrieval blend 0.25. Old/new names match exactly with packaged loading/current spec; pitch/retrieval consumers execute and ASR preserves the full source sentence. |
| [x] | SAM Audio | CUDA Small F32, six-second official office excerpt: image/video aliases produce byte-identical target and residual WAVs under old embedded/current specs. Vision executes and differs from text-only conditioning. |
| [x] | SAMSONE | CUDA Q8: old/new instruction names give identical audio descriptions under embedded and updated specs; the default prompt produces a different description. |
| [x] | sanoTTS | CUDA F32 Nano and PiperLite: legacy `speaking_rate=0.8` equals `speed=1.25`, byte-for-byte under existing and updated specs. Both shorten audio versus default and preserve the full transcript. |
| [x] | Seed-VC | V1/V2 aliases match at corresponding request positions across packaged/current specs and reversed alias order. Noise files reach their reader; V1 duration control works. V2 parses both duration names but retains the pre-existing no-op; metadata correction is separate from alias equivalence. |
| [x] | Silero VAD | CUDA bundled safetensors (no model spec/GGUF fixture): both maximum-duration names at 2 s produce identical three-segment boundaries; default produces two segments. |
| [x] | Soprano TTS | CUDA Q8: bare/prefixed weight and chunk settings produce byte-identical audio offline and streaming with embedded and updated specs. Nondefault F16 backbone/decoder settings change their weight buffers; chunk size 80 splits the input, unlike the default. |
| [x] | Sopro TTS | CUDA F16: all three duration/reference aliases produce identical offline WAVs under existing and updated specs. Legacy-existing versus canonical-updated streaming WAVs also match, emit 11 chunks, and transcribe to the requested text. |
| [ ] | Stable Audio | Shared Small-SFX parser passes old/new sampler aliases with packaged/current specs; Euler matches exactly and PingPong changes output. Foundation parser remains untested: no local checkpoint or audio-cpp-hosted Foundation package located. |
| [x] | Supertonic | Not required: duplicate speed help removed; parser unchanged. |
| [x] | Vevo2 | Six arena aliases and reference/prosody-duration aliases match exactly on Q8 with packaged loading/current override. Prosody explicitly enabled; 9/6 s controls produce 9.04/6.08 s, and ASR recovers the full text. |
| [x] | VibeASR | CPU upstream two-GGUF package: old/new token limits match using pre-refactor/current external specs. Full transcript at 96 tokens; four-token control truncates. No embedded-spec coverage claimed. |
| [x] | VieNeu v3 Turbo | Existing Q8 GGUF: dictionary, frame-cap, and minimum-chunk aliases give identical 5.2-second WAVs under packaged/current specs. Real SEA-G2P library/dictionary executes. Nondefault cap=false and minimum=60 reach the existing consumers; no frame-ceiling boundary claim. |
| [x] | VoxCPM1 | CUDA Q8: all seven arena aliases at nondefault sizes match byte-for-byte under existing/current specs. Reference encoding is exercised; ASR recovers the full input. |
| [x] | VoxCPM2 | CUDA Q8: all seven arena aliases at nondefault sizes match byte-for-byte under existing/current specs. Reference encoding is exercised; ASR recovers the full input. Chunk-strategy help is documentation-only. |
| [x] | Voxtral Realtime | Not required: token-limit help only. |
| [ ] | ZipVoice | Language and timestep-shift aliases remain unvalidated. No local package or audio-cpp-hosted package located; configured third-party package not downloaded. |
