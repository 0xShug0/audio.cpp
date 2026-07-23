# Model Manager

`tools/model_manager.py` downloads or assembles supported model packages into the
framework's expected `models/` layout.

This tool is still useful for safetensors-based packages and a few composite model
layouts, but it is gradually becoming a legacy path as audio.cpp moves toward
standalone GGUF packages.

If a model has a ready-to-use GGUF package, prefer that route first.

## GGUF Downloads

Ready-to-use GGUF packages are published here:

- Core released models: [audio-cpp/audio.cpp-gguf](https://huggingface.co/audio-cpp/audio.cpp-gguf)
- Community OuteTTS package: [mirek190/audio.cpp](https://huggingface.co/mirek190/audio.cpp/tree/main/Text%20to%20audio%20(TTS))

For support status and tested precision coverage, see the [GGUF guide](gguf.md).
For measured 16-bit vs Q8 speed and peak-VRAM results, see the
[Q8 performance report](reports/gguf_q8_performance.md).

## Dependencies

- Python 3
- `torch`
- `safetensors`
- `PyYAML`
- Network access to the upstream model source

## Commands

- `list` shows the available package ids
- `list --json` prints a machine-readable package catalog
- `info` shows the target layout, required files, and install source for one package
- `info <package> --json` prints machine-readable package details
- `install` downloads or converts one package into a models root

The runtime loader catalog is also available from:

```bash
audiocpp_cli --list-loaders --json
```

## Quick Start

List installable packages:

```bash
python3 tools/model_manager.py list
```

Inspect one package:

```bash
python3 tools/model_manager.py info qwen3_tts_1_7b_base
```

Install into the default `models/` directory:

```bash
python3 tools/model_manager.py install qwen3_tts_1_7b_base
```

Install into a custom models root:

```bash
python3 tools/model_manager.py install vevo2 --models-root /path/to/models
```

Overwrite an existing install:

```bash
python3 tools/model_manager.py install pocket_tts --overwrite
```

Install a converter-style package that needs a source file:

```bash
python3 tools/model_manager.py info voxcpm2_audiovae
python3 tools/model_manager.py install voxcpm2_audiovae --source-file models/VoxCPM2/audiovae.pth --models-root models --overwrite
```

## Package Notes

For shared audio.cpp GGUF packages, the model manager installs the default `q8_0`
GGUF. Other precision variants can be downloaded directly from
[audio-cpp/audio.cpp-gguf](https://huggingface.co/audio-cpp/audio.cpp-gguf).

`Yes` means Hugging Face has a ready-to-use repo that the framework can download
as-is. `No` means the tool must assemble, convert, or post-process files before the
framework can use them.

Packages whose loaders are not registered in the current release tree are listed as
**Unavailable**; see [loader/catalog sync notes](maintainers/loader_and_catalog.md).

| Package id | Model | HF ready-to-use repo |
|---|---|---|
| `ace_step` | ACE-Step 1.5 Turbo/Base | No |
| `chatterbox` | Chatterbox | **Yes** |
| `citrinet_asr` | Citrinet ASR converted layout | No |
| `fish_audio_s2_pro` | Fish Audio S2 Pro GGUF Q8_0 | **Yes** |
| `heartmula` | HeartMuLa | No |
| `higgs_audio_stt` | Higgs Audio STT | No |
| `higgs_audio_v3_tts_4b` | Higgs Audio v3 TTS 4B GGUF Q8_0 | **Yes** |
| `htdemucs` | HTDemucs | No |
| `hviske_asr` | Hviske ASR | **Yes** |
| `irodori_tts_500m_v3` | Irodori-TTS 500M v3 | No |
| `irodori_tts_600m_v3_voice_design` | Irodori-TTS 600M v3 VoiceDesign | No |
| `index_tts2` | IndexTTS-2 | **Yes** |
| `mel_band_roformer` | Mel-Band RoFormer MLX | **Yes** |
| `miocodec_25hz_44k_v2` | MioCodec 25Hz 44.1kHz v2 | No |
| `miotts_1_7b` | MioTTS 1.7B | No |
| `moss_audio_tokenizer_nano` | MOSS Audio Tokenizer Nano | No |
| `moss_audio_tokenizer_v2` | MOSS Audio Tokenizer v2 | No |
| `moss_tts_nano_100m` | MOSS-TTS-Nano 100M | No |
| `moss_tts_nano_100m_model` | MOSS-TTS-Nano 100M model subcomponent | No |
| `moss_tts_local_v1_5` | MOSS-TTS-Local Transformer v1.5 | No |
| `nemotron_asr` | Nemotron ASR | **Yes** |
| `omnivoice` | OmniVoice | **Yes** |
| `outetts_1_0_1b` | OuteTTS 1.0 1B with IBM DAC codec and Qwen3-aligned voice cloning | No |
| `pocket_tts` | PocketTTS | **Yes** |
| `qwen3_asr_0_6b` | Qwen3 ASR 0.6B | **Yes** |
| `qwen3_asr_1_7b_hf` | Qwen3 ASR 1.7B HF | **Yes** |
| `qwen3_forced_aligner_0_6b` | Qwen3 Forced Aligner 0.6B | **Yes** |
| `qwen3_tts_0_6b_base` | Qwen3 TTS 12Hz 0.6B Base | **Yes** |
| `qwen3_tts_1_7b_base` | Qwen3 TTS 12Hz 1.7B Base | **Yes** |
| `qwen3_tts_1_7b_custom_voice` | Qwen3 TTS 12Hz 1.7B Custom Voice | **Yes** |
| `qwen3_tts_1_7b_voice_design` | Qwen3 TTS 12Hz 1.7B Voice Design | **Yes** |
| `seed_vc` | SeedVC-MLX | **Yes** |
| `sortformer_diar_4spk_v1` | Sortformer diarization 4 speaker v1 | **Yes** |
| `stable_audio_3_medium` | Stable Audio 3 Medium | **Yes** |
| `stable_audio_3_small_music` | Stable Audio 3 Small Music | **Yes** |
| `stable_audio_3_small_sfx` | Stable Audio 3 Small SFX | **Yes** |
| `supertonic_3` | Supertonic 3 | **Yes** |
| `vevo2` | VeVo2 | No |
| `vietneu_tts_v3_turbo` | VieNeu-TTS v3 Turbo | **Yes** |
| `vibevoice_1_5b` | VibeVoice 1.5B | **Yes** |
| `vibevoice_7b` | VibeVoice 7B | **Yes** |
| `vibevoice_asr` | VibeVoice ASR | **Yes** |
| `voxcpm2` | VoxCPM2 | No |
| `voxtral_realtime` | Voxtral Mini 4B Realtime GGUF Q8_0 | **Yes** |
