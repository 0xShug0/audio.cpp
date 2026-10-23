# VocalRender-Pro

VocalRender-Pro is a Mandarin score-conditioned singing model fine-tuned from
VoxCPM2. It uses the existing `voxcpm2` runtime family; it is not a separate
audio.cpp model family. The model renders a supplied monophonic melody and
lyrics rather than composing a melody from plain text.

| Field | Value |
|---|---|
| Runtime family | `voxcpm2` |
| Package | `voxcpm2_vocalrender_pro` |
| Model file | `models/VoxCPM2-GGUF/finetune/vocalrender-pro-orig.gguf` |
| Input | Mandarin lyrics, MIDI pitches, note values, and BPM |
| Reference audio | Required; use 2–8 seconds of clean, unaccompanied singing |
| Output | Mono, 48 kHz singing audio |

## Install

Download the pre-converted GGUF from the audio.cpp Hugging Face model
collection:

```bash
python3 tools/model_manager_v2.py install voxcpm2_vocalrender_pro --models-root models
```

The GGUF is about 9.5 GB. For lower VRAM use, the runtime can quantize the
generator and AudioVAE weights when loading:

```bash
--session-option voxcpm2.weight_type=q8_0 \
--session-option voxcpm2.audiovae_weight_type=q8_0
```

Quantized loading reduces GPU memory use. Leave enough free VRAM for the
generated audio and AudioVAE graph; long score inputs need more workspace than
short samples.

## Generate singing

VocalRender uses score tokens embedded in `--text`. `<BPM_n>` sets tempo,
`<P_n>` sets MIDI pitch, and `<NOTE_n>` sets note length. Associate each lyric
syllable or character with its pitch and note value. The example below is a
short Mandarin phrase; replace it with lyrics and note tokens aligned to your
own monophonic score.

```bash
audiocpp_cli \
  --task tts \
  --family voxcpm2 \
  --model models/VoxCPM2-GGUF/finetune/vocalrender-pro-orig.gguf \
  --backend hip \
  --session-option voxcpm2.weight_type=q8_0 \
  --session-option voxcpm2.audiovae_weight_type=q8_0 \
  --text '<BPM_64>我<P_65><NOTE_8>的<P_64><NOTE_8>孤<P_64><NOTE_4>独<P_65><NOTE_4>是<P_67><NOTE_4>真<P_65><NOTE_4>的<P_67><NOTE_2>' \
  --voice-ref path/to/clean-singing-reference.wav \
  --out vocalrender.wav
```

The converted GGUF declares its family as `voxcpm2`, so pass
`--family voxcpm2`. The VocalRender-Pro checkpoint currently targets Mandarin
singing. English and German are not validated language options. Use a single
vocal melody; simultaneous notes or accompaniment are not represented by this
score-token interface.

The audio.cpp package is hosted at
[`audio-cpp/audio.cpp-gguf`](https://huggingface.co/audio-cpp/audio.cpp-gguf),
under `VoxCPM2-GGUF/finetune/`. The upstream model and its score format are
documented in the [VocalRender model card](https://huggingface.co/pymaster/VocalRender).
