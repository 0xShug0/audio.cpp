# GGUF tools

This directory contains GGUF conversion, package preparation, repacking, requantization and their input-preparation/inspection tools.

Models with multiple scripts or supporting conversion files use `tools/gguf/<model>/`. Preparation scripts, converters, helpers, vocabularies and notices for that pipeline stay together. Single-script model tools remain directly in `tools/gguf/`.

Filenames start with the model/family, followed by any variant or component, and end with the actual stage (`convert`, `prepare`, `repack`, `quantize`, `export`, `extract`, `verify`, or `manifest`). Distinct stages remain separate tools; helper filenames stay descriptive.

Run from the repository root, for example:

```bash
python tools/gguf/audio8_tts/audio8_tts_codec_prepare.py --help
python tools/gguf/audio8_tts/audio8_tts_convert.py --help
python tools/gguf/m5_tts_convert.py --help
```

These are repository tools, not self-contained distributions. Depending on the tool, they use repository specs, retained notices, an upstream checkout, and the native `build/debug/bin/audiocpp_gguf` converter. See each script's arguments and the corresponding [model documentation](../../docs/models/) for prerequisites and commands. Existing arguments, storage modes and defaults are unchanged by this reorganization.

Reference runners, fixture generators and parity/lifecycle tests belong under `tests/<model>/`. General model-output utilities and source-code generators remain in [tools/models](../models/).

| File | Purpose |
|---|---|
| [ace_step_silence_latent_prepare.py](ace_step_silence_latent_prepare.py) | Prepare source weights or package assets |
| [apollo_convert.py](apollo_convert.py) | Model weights to GGUF |
| [ast_audioset_convert.py](ast_audioset_convert.py) | Model weights to GGUF |
| [audio8_asr_convert.py](audio8_asr_convert.py) | Model weights to GGUF |
| [audio8_tts/audio8_tts_codec_prepare.py](audio8_tts/audio8_tts_codec_prepare.py) | Prepare source weights or package assets |
| [audio8_tts/audio8_tts_convert.py](audio8_tts/audio8_tts_convert.py) | Model weights to GGUF |
| [auk_convert.py](auk_convert.py) | Model weights to GGUF |
| [bs_roformer_weights_prepare.py](bs_roformer_weights_prepare.py) | Prepare source weights or package assets |
| [canary_asr_weights_prepare.py](canary_asr_weights_prepare.py) | Prepare source weights or package assets |
| [ced_convert.py](ced_convert.py) | Model weights to GGUF |
| [chatterbox_turbo_repack.py](chatterbox_turbo_repack.py) | GGUF repacking |
| [confucius4_tts_convert.py](confucius4_tts_convert.py) | Model weights to GGUF |
| [controlfoley_convert.py](controlfoley_convert.py) | Model weights to GGUF |
| [deepfilternet2_weights_prepare.py](deepfilternet2_weights_prepare.py) | Prepare source weights or package assets |
| [demucs_weights_prepare.py](demucs_weights_prepare.py) | Prepare source weights or package assets |
| [dettato_convert.py](dettato_convert.py) | Model weights to GGUF |
| [dramabox_silence_latent_prepare.py](dramabox_silence_latent_prepare.py) | Prepare source weights or package assets |
| [earshot_vad_weights_prepare.py](earshot_vad_weights_prepare.py) | Prepare source weights or package assets |
| [echo_tts/echo_tts_convert.py](echo_tts/echo_tts_convert.py) | Model weights to GGUF |
| [echo_tts/echo_tts_gguf_verify.py](echo_tts/echo_tts_gguf_verify.py) | Check the converted GGUF |
| [echo_tts/echo_tts_manifest.py](echo_tts/echo_tts_manifest.py) | Inspect checkpoint tensors |
| [f5_tts_convert.py](f5_tts_convert.py) | Model weights to GGUF |
| [firered_vad_convert.py](firered_vad_convert.py) | Model weights to GGUF |
| [foundation_1_convert.py](foundation_1_convert.py) | Package Foundation-1 and T5-base as GGUF |
| [fish_drama3_convert.py](fish_drama3_convert.py) | Model weights to GGUF |
| [gigaam_asr_convert.py](gigaam_asr_convert.py) | Model weights to GGUF |
| [glm_tts_weights_prepare.py](glm_tts_weights_prepare.py) | Prepare source weights or package assets |
| [granite5asr_convert.py](granite5asr_convert.py) | Model weights to GGUF |
| [haas_asr_convert.py](haas_asr_convert.py) | Model weights to GGUF |
| [index_echo_s2tt_convert.py](index_echo_s2tt_convert.py) | Model weights to GGUF |
| [index_tts2_5_convert.py](index_tts2_5_convert.py) | Model weights to GGUF |
| [inflect_v2_weights_prepare.py](inflect_v2_weights_prepare.py) | Prepare source weights or package assets |
| [irodori_tts/irodori_tts_convert.py](irodori_tts/irodori_tts_convert.py) | Model weights to GGUF |
| [irodori_tts/irodori_tts_dacvae_prepare.py](irodori_tts/irodori_tts_dacvae_prepare.py) | Prepare source weights or package assets |
| [irodori_tts/irodori_tts_v4_1_anime_convert.py](irodori_tts/irodori_tts_v4_1_anime_convert.py) | Model weights to GGUF |
| [kitten_tts/kitten_tts_mini_weights_prepare.py](kitten_tts/kitten_tts_mini_weights_prepare.py) | Prepare source weights or package assets |
| [kitten_tts/kitten_tts_onnx_weights_prepare.py](kitten_tts/kitten_tts_onnx_weights_prepare.py) | Prepare source weights or package assets |
| [kitten_tts2/kitten_tts2_convert.py](kitten_tts2/kitten_tts2_convert.py) | Model weights to GGUF |
| [kitten_tts2/kitten_tts2_voices_prepare.cpp](kitten_tts2/kitten_tts2_voices_prepare.cpp) | Prepare source weights or package assets |
| [kitten_tts2/kitten_tts2_weights.py](kitten_tts2/kitten_tts2_weights.py) | Shared weight-packing helper |
| [kokoro_tts_convert.py](kokoro_tts_convert.py) | Model weights to GGUF |
| [kroko_asr/kroko_asr_offline_convert.py](kroko_asr/kroko_asr_offline_convert.py) | Model weights to GGUF |
| [kroko_asr/kroko_asr_onnx_extract.py](kroko_asr/kroko_asr_onnx_extract.py) | Extract conversion inputs |
| [kugelaudio/kugelaudio_convert.py](kugelaudio/kugelaudio_convert.py) | Model weights to GGUF |
| [kugelaudio/kugelaudio_reference_encoder_repack.py](kugelaudio/kugelaudio_reference_encoder_repack.py) | Repack GGUF with reference encoder weights |
| [latentsync/latentsync_landmark_weights_prepare.py](latentsync/latentsync_landmark_weights_prepare.py) | Prepare source weights or package assets |
| [latentsync/latentsync_scrfd_weights_prepare.py](latentsync/latentsync_scrfd_weights_prepare.py) | Prepare source weights or package assets |
| [latentsync/latentsync_weights_prepare.py](latentsync/latentsync_weights_prepare.py) | Prepare source weights or package assets |
| [liveavatar/liveavatar_components_convert.py](liveavatar/liveavatar_components_convert.py) | Model weights to GGUF |
| [liveavatar/liveavatar_denoiser_convert.py](liveavatar/liveavatar_denoiser_convert.py) | Model weights to GGUF |
| [m5_tts_convert.py](m5_tts_convert.py) | Model weights to GGUF |
| [magenta_realtime_2_convert.py](magenta_realtime_2_convert.py) | Model weights to GGUF |
| [magpie_tts_weights_prepare.py](magpie_tts_weights_prepare.py) | Prepare source weights or package assets |
| [maya1_convert.py](maya1_convert.py) | Model weights to GGUF |
| [mel_band_roformer_weights_prepare.py](mel_band_roformer_weights_prepare.py) | Prepare source weights or package assets |
| [micro_wake_word_convert.py](micro_wake_word_convert.py) | Model weights to GGUF |
| [ming3_audio_convert.py](ming3_audio_convert.py) | Model weights to GGUF |
| [ming3_tts_convert.py](ming3_tts_convert.py) | Model weights to GGUF |
| [minimax_h3/minimax_h3_audio_vae_convert.py](minimax_h3/minimax_h3_audio_vae_convert.py) | Model weights to GGUF |
| [minimax_h3/minimax_h3_dit_convert.py](minimax_h3/minimax_h3_dit_convert.py) | Model weights to GGUF |
| [minimax_music3/minimax_music3_convert.py](minimax_music3/minimax_music3_convert.py) | Model weights to GGUF |
| [minimax_music3/minimax_music3_lm_head_repack.py](minimax_music3/minimax_music3_lm_head_repack.py) | GGUF repacking |
| [mira_tts_weights_prepare.py](mira_tts_weights_prepare.py) | Prepare source weights or package assets |
| [moonshine_asr_convert.py](moonshine_asr_convert.py) | Model weights to GGUF |
| [moss_soundeffect_convert.py](moss_soundeffect_convert.py) | Model weights to GGUF |
| [moss_transcribe_diarize_convert.py](moss_transcribe_diarize_convert.py) | Model weights to GGUF |
| [mossformer2_convert.py](mossformer2_convert.py) | Model weights to GGUF |
| [mulacover/mulacover_chord_vocabulary.txt](mulacover/mulacover_chord_vocabulary.txt) | Conversion vocabulary |
| [mulacover/mulacover_convert.py](mulacover/mulacover_convert.py) | Model weights to GGUF |
| [mulacover/mulacover_transcriber_prepare.py](mulacover/mulacover_transcriber_prepare.py) | Prepare source weights or package assets |
| [muscriptor_convert.py](muscriptor_convert.py) | Model weights to GGUF |
| [nemotron_3_diar_convert.py](nemotron_3_diar_convert.py) | Model weights to GGUF |
| [nemotron_transcribe/nemotron_transcribe_bf16_export.py](nemotron_transcribe/nemotron_transcribe_bf16_export.py) | Export weights or package assets |
| [nemotron_transcribe/nemotron_transcribe_convert.py](nemotron_transcribe/nemotron_transcribe_convert.py) | Model weights to GGUF |
| [nemotron_transcribe/nemotron_transcribe_prepare.py](nemotron_transcribe/nemotron_transcribe_prepare.py) | Prepare source weights or package assets |
| [nemotron_transcribe/nemotron_transcribe_quantize.py](nemotron_transcribe/nemotron_transcribe_quantize.py) | GGUF requantization |
| [niagara_asr_convert.py](niagara_asr_convert.py) | Model weights to GGUF |
| [open_wake_word/open_wake_word_convert.py](open_wake_word/open_wake_word_convert.py) | Model weights to GGUF |
| [open_wake_word/open_wake_word_weights.py](open_wake_word/open_wake_word_weights.py) | Shared weight-packing helper |
| [outetts_dac_convert.py](outetts_dac_convert.py) | Model weights to GGUF |
| [owsm_convert.py](owsm_convert.py) | Model weights to GGUF |
| [owsm_ctc_convert.py](owsm_ctc_convert.py) | Model weights to GGUF |
| [parakeet_tdt_orukeet_convert.py](parakeet_tdt_orukeet_convert.py) | Model weights to GGUF |
| [personaplex_voice_prompts_prepare.py](personaplex_voice_prompts_prepare.py) | Prepare source weights or package assets |
| [phonon2_asr_convert.py](phonon2_asr_convert.py) | Model weights to GGUF |
| [piper_tts_onnx_weights_prepare.py](piper_tts_onnx_weights_prepare.py) | Prepare source weights or package assets |
| [plapre_convert.py](plapre_convert.py) | Model weights to GGUF |
| [pulsevad_weights_prepare.py](pulsevad_weights_prepare.py) | Prepare source weights or package assets |
| [qwen3_aligner_tokenizers_export.py](qwen3_aligner_tokenizers_export.py) | Export weights or package assets |
| [qwen3_asr_rumble2_flash_convert.py](qwen3_asr_rumble2_flash_convert.py) | Model weights to GGUF |
| [realtime_reuse_convert.py](realtime_reuse_convert.py) | Model weights to GGUF |
| [reuse_convert.py](reuse_convert.py) | Model weights to GGUF |
| [rosvot_convert.py](rosvot_convert.py) | Model weights to GGUF |
| [rvc_weights_prepare.py](rvc_weights_prepare.py) | Prepare source weights or package assets |
| [sam_audio_convert.py](sam_audio_convert.py) | Model weights to GGUF |
| [samsone_convert.py](samsone_convert.py) | Model weights to GGUF |
| [seed_vc_weights_prepare.py](seed_vc_weights_prepare.py) | Prepare source weights or package assets |
| [sheetsage2_convert.py](sheetsage2_convert.py) | Model weights to GGUF |
| [sherpa_kws_convert.py](sherpa_kws_convert.py) | Model weights to GGUF |
| [sidon_convert.py](sidon_convert.py) | Model weights to GGUF |
| [smart_turn_convert.py](smart_turn_convert.py) | Model weights to GGUF |
| [soprano_tts_convert.py](soprano_tts_convert.py) | Model weights to GGUF |
| [sortformer_diar_v2/sortformer_diar_v2_NOTICES.md](sortformer_diar_v2/sortformer_diar_v2_NOTICES.md) | Attribution / provenance |
| [sortformer_diar_v2/sortformer_diar_v2_convert.py](sortformer_diar_v2/sortformer_diar_v2_convert.py) | Model weights to GGUF |
| [soulx_flashhead_convert.py](soulx_flashhead_convert.py) | Model weights to GGUF |
| [soulx_singer_convert.py](soulx_singer_convert.py) | Model weights to GGUF |
| [supertonic_onnx_weights_prepare.py](supertonic_onnx_weights_prepare.py) | Prepare source weights or package assets |
| [tf_gridnet_convert.py](tf_gridnet_convert.py) | Model weights to GGUF |
| [tiny_voca_convert.py](tiny_voca_convert.py) | Model weights to GGUF |
| [tone_color_vc_convert.py](tone_color_vc_convert.py) | Model weights to GGUF |
| [underdog_bark3/underdog_bark3_convert.py](underdog_bark3/underdog_bark3_convert.py) | Model weights to GGUF |
| [underdog_bark3/underdog_bark3_plus_convert.py](underdog_bark3/underdog_bark3_plus_convert.py) | Model weights to GGUF |
| [underdog_bark3/underdog_bark3_standard_convert.py](underdog_bark3/underdog_bark3_standard_convert.py) | Model weights to GGUF |
| [universr_convert.py](universr_convert.py) | Model weights to GGUF |
| [vibeasr_convert.py](vibeasr_convert.py) | Model weights to GGUF |
| [vibevoice_asr_streaming_convert.py](vibevoice_asr_streaming_convert.py) | Model weights to GGUF |
| [voice_isolator_convert.py](voice_isolator_convert.py) | Model weights to GGUF |
| [voxcpm1_convert.py](voxcpm1_convert.py) | Model weights to GGUF |
| [voxcpm2_weights_prepare.py](voxcpm2_weights_prepare.py) | Prepare source weights or package assets |
| [voxmel_tts_convert.py](voxmel_tts_convert.py) | Model weights to GGUF |
| [voxtral_tts_convert.py](voxtral_tts_convert.py) | Model weights to GGUF |
| [whisper_medium_weights_prepare.py](whisper_medium_weights_prepare.py) | Prepare source weights or package assets |
| [whistle_asr/whistle_asr_PROVENANCE.md](whistle_asr/whistle_asr_PROVENANCE.md) | Attribution / provenance |
| [whistle_asr/whistle_asr_assets_extract.py](whistle_asr/whistle_asr_assets_extract.py) | Extract conversion inputs |
| [whistle_asr/whistle_asr_convert.py](whistle_asr/whistle_asr_convert.py) | Model weights to GGUF |
| [xtts_v2_convert.py](xtts_v2_convert.py) | Model weights to GGUF |
| [yue2_convert.py](yue2_convert.py) | Model weights to GGUF |
| [zipvoice/zipvoice_convert.py](zipvoice/zipvoice_convert.py) | Model weights to GGUF |
| [zipvoice/zipvoice_zh_dictionary_export.py](zipvoice/zipvoice_zh_dictionary_export.py) | Export weights or package assets |
