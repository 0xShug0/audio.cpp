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

## Runtime Validation Checklist

Checked rows explicitly state either a runtime pass or an exemption. Unchecked
rows still require validation of legacy and canonical options with existing GGUFs,
including affected offline/streaming paths. A model remains unchecked if any of
its changes affect parsing or compatibility, even when other changes are docs-only.

| Done | Model | Runtime validation requirement |
|---|---|---|
| [ ] | ACE-Step | Duration, LM guidance, and thinking aliases. |
| [x] | Audio Flamingo | CUDA Flamingo 3 Q4_K: both instruction names produce identical summaries under embedded and updated specs; the default produces a transcription instead. |
| [ ] | AuK | Flash Q8 aliases match byte-for-byte across old/current external specs; editing rejects both correctly. ASR recovers the input but also reports an unexpected opening phrase. Output-quality follow-up remains open. |
| [x] | Canary ASR | CUDA Q8: legacy `pnc` and canonical `punc` produce identical transcripts with the existing embedded spec and the updated spec. Disabling the control removes punctuation/capitalization; the default restores them. |
| [ ] | Chatterbox | Clone flow aliases/greedy override match byte-for-byte, but ASR finds only the first sentence. VC aliases match at the same request position; unchanged repeated VC requests differ in waveform while preserving the transcript. Correctness follow-up remains open. |
| [x] | Chatterbox Turbo | CUDA Q8: token-limit aliases produce identical complete speech under existing and updated specs. Reducing the cap from 512 to 64 shortens output from 6.00 to 2.68 s. |
| [x] | Cohere ASR | CUDA Q8: both punctuation aliases produce identical text with embedded and updated specs; the default restores punctuation/capitalization. |
| [ ] | ControlFoley | Video-file aliases and embedded-spec compatibility. |
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
| [ ] | Irodori TTS | Both alias pairs match byte-for-byte across old embedded/current specs; reference trimming and embedding loading execute. Reference-conditioned ASR matches; embedding-conditioned ASR reports a short extra tail, so output-quality follow-up remains open. |
| [x] | Kokoro | CUDA Q8 offline: all three legacy/canonical arena names, with nondefault sizes, produce byte-identical WAVs under existing and updated specs. ASR recovers the requested text. |
| [x] | LFM2 Audio | CUDA upstream Q8 GGUFs require an explicit spec override (no embedded spec). With it, both chunk/one-pass aliases produce identical two-chunk transcripts; both enforce the limit and remain rejected on TTS/S2S where inapplicable. |
| [ ] | LiveAvatar | Memory-policy and reference-image file aliases. |
| [x] | Maya1 | CUDA Q8: both instruction names produce byte-identical WAVs under embedded and updated specs; ASR recovers the requested sentence. Missing instruction still raises the required-input error. |
| [ ] | MiniMax H3 | Sampler aliases. |
| [ ] | MiniMax Music3 | Graph-arena aliases; component default correction itself is metadata-only. |
| [ ] | MiraTTS | Prefixed/bare weights, unchanged defaults, and older embedded specs still need validation. Located native/upstream safetensors only; these cannot prove legacy-GGUF compatibility. |
| [x] | MioTTS | Not required: request-option help names only. |
| [x] | MOSS-Transcribe-Diarize | Instruction aliases match in offline and streaming with existing/current specs. Custom prompt omits timestamps under both names; default prompt produces a parsed speaker turn. |
| [ ] | MOSS-TTS v1.5 | Matching package not located. Available MOSS-TTS-Local-v1.5 GGUF embeds the different `moss_tts_local` family and is not a valid substitute. |
| [x] | MOSS-TTSD | Instruction and text-temperature aliases produce byte-identical short dialogue across embedded/current specs; ASR recovers both sentences. |
| [x] | MOSS-VoiceGen | Instruction and text-temperature aliases produce byte-identical speech across embedded/current specs; ASR recovers the complete input. |
| [ ] | OuteTTS | Aligner-path aliases and file resolution. |
| [x] | OWSM | CUDA Q8: `beam_size=2` and `num_beams=2` produce identical transcripts offline and streaming, under embedded and updated specs. |
| [ ] | PocketTTS | Input-token budget, frame cap, voice-state files, and graph-arena aliases; offline and streaming. |
| [x] | Qwen3 ASR | CUDA 0.6B Q8: both VAD-path names load Silero and produce identical two-chunk transcripts under existing and updated specs. |
| [x] | Qwen3 TTS | CUDA CustomVoice Q8: `speaker=ryan` and `voice_id=ryan` override structured Aiden, matching the Ryan-only control and differing from Aiden. Offline results match across specs; streaming alias WAVs also match. Both modes transcribe to the requested text. |
| [x] | RVC | Existing F16 GGUF: valid pitch contour and real FAISS index, retrieval blend 0.25. Old/new names match exactly with packaged loading/current spec; pitch/retrieval consumers execute and ASR preserves the full source sentence. |
| [ ] | SAM Audio | Reference image/video file aliases and validation. |
| [x] | SAMSONE | CUDA Q8: old/new instruction names give identical audio descriptions under embedded and updated specs; the default prompt produces a different description. |
| [x] | sanoTTS | CUDA F32 Nano and PiperLite: legacy `speaking_rate=0.8` equals `speed=1.25`, byte-for-byte under existing and updated specs. Both shorten audio versus default and preserve the full transcript. |
| [ ] | Seed-VC | V1/V2 aliases match at corresponding request positions across packaged/current specs and reversed alias order; ASR preserves the full sentence. V1 duration control works. V2 duration is a pre-existing no-op requiring metadata correction; repeated requests are not waveform-identical. |
| [x] | Silero VAD | CUDA bundled safetensors (no model spec/GGUF fixture): both maximum-duration names at 2 s produce identical three-segment boundaries; default produces two segments. |
| [x] | Soprano TTS | CUDA Q8: bare/prefixed weight and chunk settings produce byte-identical audio offline and streaming with embedded and updated specs. Nondefault F16 backbone/decoder settings change their weight buffers; chunk size 80 splits the input, unlike the default. |
| [x] | Sopro TTS | CUDA F16: all three duration/reference aliases produce identical offline WAVs under existing and updated specs. Legacy-existing versus canonical-updated streaming WAVs also match, emit 11 chunks, and transcribe to the requested text. |
| [ ] | Stable Audio | Shared Small-SFX parser passes old/new sampler aliases with packaged/current specs; Euler matches exactly and PingPong changes output. Foundation parser remains untested without its checkpoint. |
| [x] | Supertonic | Not required: duplicate speed help removed; parser unchanged. |
| [x] | Vevo2 | Six arena aliases and reference/prosody-duration aliases match exactly on Q8 with packaged loading/current override. Prosody explicitly enabled; 9/6 s controls produce 9.04/6.08 s, and ASR recovers the full text. |
| [x] | VibeASR | CPU upstream two-GGUF package: old/new token limits match using pre-refactor/current external specs. Full transcript at 96 tokens; four-token control truncates. No embedded-spec coverage claimed. |
| [ ] | VieNeu v3 Turbo | Automatic frame cap, minimum text chunk, and dictionary-file aliases remain untested; the required real G2P dictionary/library fixture was not located. |
| [x] | VoxCPM1 | CUDA Q8: all seven arena aliases at nondefault sizes match byte-for-byte under existing/current specs. Reference encoding is exercised; ASR recovers the full input. |
| [x] | VoxCPM2 | CUDA Q8: all seven arena aliases at nondefault sizes match byte-for-byte under existing/current specs. Reference encoding is exercised; ASR recovers the full input. Chunk-strategy help is documentation-only. |
| [x] | Voxtral Realtime | Not required: token-limit help only. |
