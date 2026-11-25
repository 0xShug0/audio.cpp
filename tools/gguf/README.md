# GGUF tools

GGUF conversion, package preparation, repacking and requantization scripts live in this flat directory.

Names start with the model/family, followed by any variant or component, and end with the stage: `_convert.py`, `_prepare.py`, `_repack.py` or `_quantize.py`. Separate stages remain separate tools.

Run from the repository root, for example:

```bash
python tools/gguf/m5_tts_convert.py --help
```

These are repository tools, not self-contained distributions. Depending on the tool, they use repository specs, retained helpers or notices, an upstream checkout, and the native `build/debug/bin/audiocpp_gguf` converter. See each script's arguments and the corresponding [model documentation](../../docs/models/) for prerequisites and commands. Existing arguments, storage modes and defaults are unchanged by this reorganization.

Tensor dumps, reference generators, verification scripts and source-to-safetensors-only preparation remain outside this directory.

| Script | Stage |
|---|---|
| [apollo_convert.py](apollo_convert.py) | Source/checkpoint or GGUF conversion |
| [ast_audioset_convert.py](ast_audioset_convert.py) | Source/checkpoint or GGUF conversion |
| [audio8_asr_convert.py](audio8_asr_convert.py) | Source/checkpoint or GGUF conversion |
| [audio8_tts_convert.py](audio8_tts_convert.py) | Source/checkpoint or GGUF conversion |
| [auk_convert.py](auk_convert.py) | Source/checkpoint or GGUF conversion |
| [ced_convert.py](ced_convert.py) | Source/checkpoint or GGUF conversion |
| [chatterbox_turbo_repack.py](chatterbox_turbo_repack.py) | GGUF repacking |
| [confucius4_tts_convert.py](confucius4_tts_convert.py) | Source/checkpoint or GGUF conversion |
| [controlfoley_convert.py](controlfoley_convert.py) | Source/checkpoint or GGUF conversion |
| [dettato_convert.py](dettato_convert.py) | Source/checkpoint or GGUF conversion |
| [echo_tts_convert.py](echo_tts_convert.py) | Source/checkpoint or GGUF conversion |
| [f5_tts_convert.py](f5_tts_convert.py) | Source/checkpoint or GGUF conversion |
| [firered_vad_convert.py](firered_vad_convert.py) | Source/checkpoint or GGUF conversion |
| [fish_drama3_convert.py](fish_drama3_convert.py) | Source/checkpoint or GGUF conversion |
| [gigaam_asr_convert.py](gigaam_asr_convert.py) | Source/checkpoint or GGUF conversion |
| [granite5asr_convert.py](granite5asr_convert.py) | Source/checkpoint or GGUF conversion |
| [haas_asr_convert.py](haas_asr_convert.py) | Source/checkpoint or GGUF conversion |
| [index_echo_s2tt_convert.py](index_echo_s2tt_convert.py) | Source/checkpoint or GGUF conversion |
| [index_tts2_5_convert.py](index_tts2_5_convert.py) | Source/checkpoint or GGUF conversion |
| [irodori_tts_convert.py](irodori_tts_convert.py) | Source/checkpoint or GGUF conversion |
| [irodori_tts_v4_1_anime_convert.py](irodori_tts_v4_1_anime_convert.py) | Source/checkpoint or GGUF conversion |
| [kitten_tts2_convert.py](kitten_tts2_convert.py) | Source/checkpoint or GGUF conversion |
| [kokoro_tts_convert.py](kokoro_tts_convert.py) | Source/checkpoint or GGUF conversion |
| [kroko_asr_offline_convert.py](kroko_asr_offline_convert.py) | Source/checkpoint or GGUF conversion |
| [kugelaudio_convert.py](kugelaudio_convert.py) | Source/checkpoint or GGUF conversion |
| [liveavatar_components_convert.py](liveavatar_components_convert.py) | Source/checkpoint or GGUF conversion |
| [liveavatar_denoiser_convert.py](liveavatar_denoiser_convert.py) | Source/checkpoint or GGUF conversion |
| [m5_tts_convert.py](m5_tts_convert.py) | Source/checkpoint or GGUF conversion |
| [magenta_realtime_2_convert.py](magenta_realtime_2_convert.py) | Source/checkpoint or GGUF conversion |
| [maya1_convert.py](maya1_convert.py) | Source/checkpoint or GGUF conversion |
| [micro_wake_word_convert.py](micro_wake_word_convert.py) | Source/checkpoint or GGUF conversion |
| [ming3_audio_convert.py](ming3_audio_convert.py) | Source/checkpoint or GGUF conversion |
| [ming3_tts_convert.py](ming3_tts_convert.py) | Source/checkpoint or GGUF conversion |
| [minimax_h3_audio_vae_convert.py](minimax_h3_audio_vae_convert.py) | Source/checkpoint or GGUF conversion |
| [minimax_h3_dit_convert.py](minimax_h3_dit_convert.py) | Source/checkpoint or GGUF conversion |
| [minimax_music3_convert.py](minimax_music3_convert.py) | Source/checkpoint or GGUF conversion |
| [minimax_music3_lm_head_repack.py](minimax_music3_lm_head_repack.py) | GGUF repacking |
| [moonshine_asr_convert.py](moonshine_asr_convert.py) | Source/checkpoint or GGUF conversion |
| [moss_soundeffect_convert.py](moss_soundeffect_convert.py) | Source/checkpoint or GGUF conversion |
| [moss_transcribe_diarize_convert.py](moss_transcribe_diarize_convert.py) | Source/checkpoint or GGUF conversion |
| [mossformer2_convert.py](mossformer2_convert.py) | Source/checkpoint or GGUF conversion |
| [mulacover_convert.py](mulacover_convert.py) | Source/checkpoint or GGUF conversion |
| [muscriptor_convert.py](muscriptor_convert.py) | Source/checkpoint or GGUF conversion |
| [nemotron_3_diar_convert.py](nemotron_3_diar_convert.py) | Source/checkpoint or GGUF conversion |
| [nemotron_transcribe_convert.py](nemotron_transcribe_convert.py) | Source/checkpoint or GGUF conversion |
| [nemotron_transcribe_prepare.py](nemotron_transcribe_prepare.py) | GGUF package preparation |
| [nemotron_transcribe_quantize.py](nemotron_transcribe_quantize.py) | GGUF requantization |
| [niagara_asr_convert.py](niagara_asr_convert.py) | Source/checkpoint or GGUF conversion |
| [open_wake_word_convert.py](open_wake_word_convert.py) | Source/checkpoint or GGUF conversion |
| [outetts_dac_convert.py](outetts_dac_convert.py) | Source/checkpoint or GGUF conversion |
| [owsm_convert.py](owsm_convert.py) | Source/checkpoint or GGUF conversion |
| [owsm_ctc_convert.py](owsm_ctc_convert.py) | Source/checkpoint or GGUF conversion |
| [parakeet_tdt_orukeet_convert.py](parakeet_tdt_orukeet_convert.py) | Source/checkpoint or GGUF conversion |
| [phonon2_asr_convert.py](phonon2_asr_convert.py) | Source/checkpoint or GGUF conversion |
| [plapre_convert.py](plapre_convert.py) | Source/checkpoint or GGUF conversion |
| [qwen3_asr_rumble2_flash_convert.py](qwen3_asr_rumble2_flash_convert.py) | Source/checkpoint or GGUF conversion |
| [realtime_reuse_convert.py](realtime_reuse_convert.py) | Source/checkpoint or GGUF conversion |
| [reuse_convert.py](reuse_convert.py) | Source/checkpoint or GGUF conversion |
| [rosvot_convert.py](rosvot_convert.py) | Source/checkpoint or GGUF conversion |
| [sam_audio_convert.py](sam_audio_convert.py) | Source/checkpoint or GGUF conversion |
| [samsone_convert.py](samsone_convert.py) | Source/checkpoint or GGUF conversion |
| [sheetsage2_convert.py](sheetsage2_convert.py) | Source/checkpoint or GGUF conversion |
| [sherpa_kws_convert.py](sherpa_kws_convert.py) | Source/checkpoint or GGUF conversion |
| [sidon_convert.py](sidon_convert.py) | Source/checkpoint or GGUF conversion |
| [smart_turn_convert.py](smart_turn_convert.py) | Source/checkpoint or GGUF conversion |
| [soprano_tts_convert.py](soprano_tts_convert.py) | Source/checkpoint or GGUF conversion |
| [sortformer_diar_v2_convert.py](sortformer_diar_v2_convert.py) | Source/checkpoint or GGUF conversion |
| [soulx_flashhead_convert.py](soulx_flashhead_convert.py) | Source/checkpoint or GGUF conversion |
| [soulx_singer_convert.py](soulx_singer_convert.py) | Source/checkpoint or GGUF conversion |
| [tf_gridnet_convert.py](tf_gridnet_convert.py) | Source/checkpoint or GGUF conversion |
| [tiny_voca_convert.py](tiny_voca_convert.py) | Source/checkpoint or GGUF conversion |
| [tone_color_vc_convert.py](tone_color_vc_convert.py) | Source/checkpoint or GGUF conversion |
| [underdog_bark3_convert.py](underdog_bark3_convert.py) | Source/checkpoint or GGUF conversion |
| [underdog_bark3_plus_convert.py](underdog_bark3_plus_convert.py) | Source/checkpoint or GGUF conversion |
| [underdog_bark3_standard_convert.py](underdog_bark3_standard_convert.py) | Source/checkpoint or GGUF conversion |
| [universr_convert.py](universr_convert.py) | Source/checkpoint or GGUF conversion |
| [vibeasr_convert.py](vibeasr_convert.py) | Source/checkpoint or GGUF conversion |
| [vibevoice_asr_streaming_convert.py](vibevoice_asr_streaming_convert.py) | Source/checkpoint or GGUF conversion |
| [voice_isolator_convert.py](voice_isolator_convert.py) | Source/checkpoint or GGUF conversion |
| [voxcpm1_convert.py](voxcpm1_convert.py) | Source/checkpoint or GGUF conversion |
| [voxmel_tts_convert.py](voxmel_tts_convert.py) | Source/checkpoint or GGUF conversion |
| [voxtral_tts_convert.py](voxtral_tts_convert.py) | Source/checkpoint or GGUF conversion |
| [whistle_asr_convert.py](whistle_asr_convert.py) | Source/checkpoint or GGUF conversion |
| [xtts_v2_convert.py](xtts_v2_convert.py) | Source/checkpoint or GGUF conversion |
| [yue2_convert.py](yue2_convert.py) | Source/checkpoint or GGUF conversion |
| [zipvoice_convert.py](zipvoice_convert.py) | Source/checkpoint or GGUF conversion |

`kitten_tts2_weights.py` is the shared offline weight-packing helper used by `kitten_tts2_convert.py`; it is not a standalone command.
