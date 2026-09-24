# Model Licenses

audio.cpp is licensed under Apache-2.0. That license does not cover the model
weights it loads: each model keeps the license of its original release, and a GGUF
conversion keeps the license of the model it was converted from.

This page lists, per model family, where the original weights are published, their
license, and whether that license allows commercial use. The license text at the source
is what counts. This page is a summary to help you find it, not legal advice.

Licenses change. If a row is wrong or out of date, please open a PR.

## Commercial use

| Value | Meaning |
|---|---|
| Yes | The license allows commercial use. Attribution, notice, share-alike or use-policy terms may apply. |
| Conditional | Commercial use is allowed only under a condition, such as a revenue or user limit, a territory, registration, or a notice in the product. Notes names the condition. |
| No | Research or non-commercial use only. |
| Unclear | The sources conflict, or no license is stated. Notes says why. |

## Families

| Family | Original weights | License | Commercial use | Notes | Checked |
|---|---|---|---|---|---|
| `ace_step` | [ACE-Step/Ace-Step1.5](https://huggingface.co/ACE-Step/Ace-Step1.5) | MIT | Yes | | 2026-09-21 |
| `apollo` | [JusperLee/Apollo](https://huggingface.co/JusperLee/Apollo) | CC-BY-SA-4.0 | Yes | | 2026-09-21 |
| `audio8_asr` | [Edge0/Audio8-ASR-0.1B](https://huggingface.co/Edge0/Audio8-ASR-0.1B) | CC-BY-NC-4.0 | No | | 2026-09-21 |
| `audio8_tts` | [Edge0/Audio8-TTS-Preview-0.6b](https://huggingface.co/Edge0/Audio8-TTS-Preview-0.6b) | Apache-2.0 | Yes | | 2026-09-21 |
| `audiosr` | [haoheliu/audiosr_basic](https://huggingface.co/haoheliu/audiosr_basic) | Apache-2.0 | Yes | | 2026-09-21 |
| `auk` | [tencent/AuK](https://huggingface.co/tencent/AuK)<br>[tencent/AuK-Flash](https://huggingface.co/tencent/AuK-Flash) | MIT | Yes | | 2026-09-21 |
| `breeze_tts` | [BreezeBlue/Breeze-TTS-2](https://huggingface.co/BreezeBlue/Breeze-TTS-2) | [BreezeBlue Research and Non-Commercial License](https://huggingface.co/BreezeBlue/Breeze-TTS-2/blob/main/LICENSE) | No | | 2026-09-21 |
| `bs_roformer` | Not linked | Apache-2.0, as listed in the audio.cpp GGUF repo | Unclear | The source of the ep368 checkpoint is not documented. | 2026-09-21 |
| `builtin_audio_utils` | [xiph/rnnoise](https://github.com/xiph/rnnoise)<br>[Rikorose/DeepFilterNet](https://github.com/Rikorose/DeepFilterNet) | BSD-3-Clause (RNNoise); MIT or Apache-2.0 (DeepFilterNet) | Yes | Weights are downloaded separately. | 2026-09-21 |
| `canary_asr` | [nvidia/canary-180m-flash](https://huggingface.co/nvidia/canary-180m-flash) | CC-BY-4.0 | Yes | | 2026-09-21 |
| `chatterbox` | [ResembleAI/chatterbox](https://huggingface.co/ResembleAI/chatterbox) | MIT | Yes | | 2026-09-21 |
| `chatterbox_turbo` | [ResembleAI/chatterbox-turbo](https://huggingface.co/ResembleAI/chatterbox-turbo) | MIT | Yes | | 2026-09-21 |
| `citrinet_asr` | [nvidia/stt_en_citrinet_256_ls](https://huggingface.co/nvidia/stt_en_citrinet_256_ls) | CC-BY-4.0 | Yes | | 2026-09-21 |
| `cohere_asr` | [CohereLabs/cohere-transcribe-03-2026](https://huggingface.co/CohereLabs/cohere-transcribe-03-2026) | Apache-2.0 | Yes | | 2026-09-21 |
| `confucius4_r2t2` | [netease-youdao/Confucius4-R2T2](https://huggingface.co/netease-youdao/Confucius4-R2T2) | [NetEase Youdao Model License](https://github.com/netease-youdao/Confucius4-R2T2/blob/master/MODEL_LICENSE) | Conditional | A separate license is needed above RMB 1 billion yearly revenue or 100 million monthly users. The model and its outputs may not be used to improve other commercial AI models. | 2026-09-21 |
| `confucius4_tts` | [netease-youdao/Confucius4-TTS](https://huggingface.co/netease-youdao/Confucius4-TTS) | Apache-2.0 | Yes | | 2026-09-21 |
| `controlfoley` | [YJX-Xiaomi/ControlFoley](https://huggingface.co/YJX-Xiaomi/ControlFoley) | CC-BY-NC-4.0 | No | | 2026-09-21 |
| `cosyvoice3` | [FunAudioLLM/Fun-CosyVoice3-0.5B-2512](https://huggingface.co/FunAudioLLM/Fun-CosyVoice3-0.5B-2512) | Apache-2.0 | Yes | | 2026-09-21 |
| `dots_tts` | [dots-studio/dots.tts-soar](https://huggingface.co/dots-studio/dots.tts-soar)<br>[dots-studio/dots.tts-mf](https://huggingface.co/dots-studio/dots.tts-mf)<br>[dots-studio/dots.tts.edit](https://huggingface.co/dots-studio/dots.tts.edit) | Apache-2.0 | Yes | The dots.tts.edit card defers to the [dots.tts project](https://github.com/studio-dots-ai/dots.tts), which is Apache-2.0. | 2026-09-21 |
| `dramabox` | [ResembleAI/Dramabox](https://huggingface.co/ResembleAI/Dramabox) | [LTX-2 Community License](https://huggingface.co/ResembleAI/Dramabox/blob/main/LICENSE) | Conditional | A paid license is needed from USD 10M yearly revenue. Generated content must be disclosed as machine-generated. Not for products that compete with Lightricks'. The Gemma 3 text encoder falls under the Gemma Terms of Use. | 2026-09-21 |
| `echo_tts` | [jordand/echo-tts-base](https://huggingface.co/jordand/echo-tts-base)<br>[jordand/fish-s1-dac-min](https://huggingface.co/jordand/fish-s1-dac-min) | CC-BY-NC-SA-4.0 | No | | 2026-09-21 |
| `f5_tts` | [SWivid/F5-TTS](https://huggingface.co/SWivid/F5-TTS)<br>[SWivid/Habibi-TTS](https://huggingface.co/SWivid/Habibi-TTS) | CC-BY-NC-4.0 (F5-TTS); CC-BY-NC-SA-4.0 (Habibi-TTS) | No | | 2026-09-21 |
| `firered_audio` | [FireRedTeam/FireRedAudio](https://huggingface.co/FireRedTeam/FireRedAudio) | Apache-2.0 | Yes | | 2026-09-21 |
| `fireredtts3` | [FireRedTeam/FireRedTTS3](https://huggingface.co/FireRedTeam/FireRedTTS3) | Apache-2.0 | Yes | | 2026-09-21 |
| `fish_audio` | [fishaudio/s2-pro](https://huggingface.co/fishaudio/s2-pro) | [Fish Audio Research License](https://huggingface.co/fishaudio/s2-pro/blob/main/LICENSE.md) | No | Commercial use needs a separate license from Fish Audio. | 2026-09-21 |
| `fun_asr_nano` | [FunAudioLLM/Fun-ASR-Nano-2512](https://huggingface.co/FunAudioLLM/Fun-ASR-Nano-2512) | Apache-2.0 | Yes | | 2026-09-21 |
| `glm_tts` | [zai-org/GLM-TTS](https://huggingface.co/zai-org/GLM-TTS) | MIT | Yes | | 2026-09-21 |
| `granite5asr` | [ibm-granite/granite-speech-5.0-470m-turboctc](https://huggingface.co/ibm-granite/granite-speech-5.0-470m-turboctc) | Apache-2.0 | Yes | | 2026-09-21 |
| `heartmula` | [HeartMuLa/HeartMuLa-oss-3B](https://huggingface.co/HeartMuLa/HeartMuLa-oss-3B)<br>[HeartMuLa/HeartCodec-oss-20260123](https://huggingface.co/HeartMuLa/HeartCodec-oss-20260123) | Apache-2.0 | Yes | | 2026-09-21 |
| `higgs_audio_stt` | [bosonai/higgs-audio-v3-stt](https://huggingface.co/bosonai/higgs-audio-v3-stt) | Apache-2.0 | Yes | | 2026-09-21 |
| `higgs_audio_tts` | [bosonai/higgs-tts-3-4b](https://huggingface.co/bosonai/higgs-tts-3-4b) | [Boson Higgs TTS 3 Research and Non-Commercial License](https://huggingface.co/bosonai/higgs-tts-3-4b/blob/main/LICENSE) | No | | 2026-09-21 |
| `htdemucs` | [facebookresearch/demucs](https://github.com/facebookresearch/demucs) | MIT | Yes | | 2026-09-21 |
| `hviske_asr` | [syvai/hviske-v5.3](https://huggingface.co/syvai/hviske-v5.3) | CC-BY-NC-4.0 | No | | 2026-09-21 |
| `index_tts2` | [IndexTeam/IndexTTS-2](https://huggingface.co/IndexTeam/IndexTTS-2)<br>[IndexTeam/IndexTTS-2.5](https://huggingface.co/IndexTeam/IndexTTS-2.5) | [bilibili Model Use License](https://huggingface.co/IndexTeam/IndexTTS-2.5/blob/main/LICENSE) | Conditional | A separate license is needed above RMB 1 billion yearly revenue or 100 million monthly users. The model and its outputs may not be used to improve other commercial AI models. | 2026-09-21 |
| `inflect_v2` | [owensong/Inflect-Micro-v2](https://huggingface.co/owensong/Inflect-Micro-v2)<br>[owensong/Inflect-Nano-v2](https://huggingface.co/owensong/Inflect-Nano-v2) | Apache-2.0 | Yes | | 2026-09-21 |
| `irodori_tts` | [Aratako/Irodori-TTS-500M-v3](https://huggingface.co/Aratako/Irodori-TTS-500M-v3)<br>[Aratako/Irodori-TTS-600M-v3-VoiceDesign](https://huggingface.co/Aratako/Irodori-TTS-600M-v3-VoiceDesign)<br>[Aratako/Irodori-TTS-v4.1-Small](https://huggingface.co/Aratako/Irodori-TTS-v4.1-Small) | MIT | Yes | | 2026-09-21 |
| `kitten_tts` | [KittenML/kitten-tts-mini-0.8](https://huggingface.co/KittenML/kitten-tts-mini-0.8) | Apache-2.0 | Yes | | 2026-09-21 |
| `kokoro_tts` | [hexgrad/Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) | Apache-2.0 | Yes | | 2026-09-21 |
| `kroko_asr` | [Banafo/Kroko-ASR](https://huggingface.co/Banafo/Kroko-ASR) | CC-BY-SA | Yes | Stated in the model card text only: the LICENSE file is empty and no version is named. Applies to the community models; Kroko's commercial models are licensed separately. | 2026-09-21 |
| `liveavatar` | [Quark-Vision/Live-Avatar](https://huggingface.co/Quark-Vision/Live-Avatar)<br>[Wan-AI/Wan2.2-S2V-14B](https://huggingface.co/Wan-AI/Wan2.2-S2V-14B) | Apache-2.0 | Yes | | 2026-09-21 |
| `magpie_tts` | [nvidia/magpie_tts_multilingual_357m](https://huggingface.co/nvidia/magpie_tts_multilingual_357m) | [NVIDIA Open Model License](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/) | Yes | | 2026-09-21 |
| `marblenet_vad` | Not linked | Not stated | Unclear | Bundled in `assets/framework/models`; the checkpoint's source is not documented. | 2026-09-21 |
| `meanvc2` | [ASLP-lab/MeanVC2](https://huggingface.co/ASLP-lab/MeanVC2) | Apache-2.0 | Yes | | 2026-09-21 |
| `mel_band_roformer` | [mlx-community/mel-roformer-mlx](https://huggingface.co/mlx-community/mel-roformer-mlx) | MIT | Yes | | 2026-09-21 |
| `midashenglm_gen` | [mispeech/midashenglm-gen](https://huggingface.co/mispeech/midashenglm-gen) | Apache-2.0 | Yes | | 2026-09-21 |
| `minimax_h3` | [MiniMaxAI/MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3) | [MiniMax H3 Community License](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE) | Conditional | Grants no rights in the EU, the UK, South Korea or the USA. Elsewhere, written authorization is needed above USD 20M yearly revenue, and "MiniMax H3" must be shown in the product's user interface. | 2026-09-21 |
| `minimax_music3` | [MiniMaxAI/MiniMax-Music3](https://huggingface.co/MiniMaxAI/MiniMax-Music3) | [MiniMax-Music3 Community License](https://huggingface.co/MiniMaxAI/MiniMax-Music3/blob/main/LICENSE) | Conditional | "MiniMax-Music3" must be shown in the product's user interface. Written authorization is needed above USD 20M yearly revenue. Content published in public must be disclosed as machine-generated. | 2026-09-21 |
| `miocodec` | [Aratako/MioCodec-25Hz-44.1kHz-v2](https://huggingface.co/Aratako/MioCodec-25Hz-44.1kHz-v2) | MIT | Yes | | 2026-09-21 |
| `miotts` | [Aratako/MioTTS-1.7B](https://huggingface.co/Aratako/MioTTS-1.7B) | Apache-2.0 | Yes | | 2026-09-21 |
| `mira_tts` | [YatharthS/MiraTTS](https://huggingface.co/YatharthS/MiraTTS) | CC-BY-NC-SA-4.0 | No | | 2026-09-21 |
| `mms_forced_aligner` | [MahmoudAshraf/mms-300m-1130-forced-aligner](https://huggingface.co/MahmoudAshraf/mms-300m-1130-forced-aligner) | CC-BY-NC-4.0 | No | | 2026-09-21 |
| `moonshine_asr` | [moonshine-ai/moonshine-streaming-tiny](https://huggingface.co/moonshine-ai/moonshine-streaming-tiny)<br>[moonshine-ai/moonshine-streaming-small](https://huggingface.co/moonshine-ai/moonshine-streaming-small)<br>[moonshine-ai/moonshine-streaming-medium](https://huggingface.co/moonshine-ai/moonshine-streaming-medium) | MIT | Yes | | 2026-09-21 |
| `moss_transcribe_diarize` | [OpenMOSS-Team/MOSS-Transcribe-Diarize](https://huggingface.co/OpenMOSS-Team/MOSS-Transcribe-Diarize) | Apache-2.0 | Yes | | 2026-09-21 |
| `moss_tts_local` | [OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5](https://huggingface.co/OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5) | Apache-2.0 | Yes | | 2026-09-21 |
| `moss_tts_nano` | [OpenMOSS-Team/MOSS-TTS-Nano-100M](https://huggingface.co/OpenMOSS-Team/MOSS-TTS-Nano-100M) | Apache-2.0 | Yes | | 2026-09-21 |
| `moss_tts_v15` | [OpenMOSS-Team/MOSS-TTS-v1.5](https://huggingface.co/OpenMOSS-Team/MOSS-TTS-v1.5) | Apache-2.0 | Yes | | 2026-09-22 |
| `moss_voicegen` | [OpenMOSS-Team/MOSS-VoiceGenerator](https://huggingface.co/OpenMOSS-Team/MOSS-VoiceGenerator) | Apache-2.0 | Yes | | 2026-09-21 |
| `muscriptor` | [MuScriptor/muscriptor-small](https://huggingface.co/MuScriptor/muscriptor-small) | CC-BY-NC-4.0 | No | | 2026-09-21 |
| `nemotron_3_diar` | [nvidia/Nemotron-3-Diarization](https://huggingface.co/nvidia/Nemotron-3-Diarization) | [OpenMDW-1.1](https://openmdw.ai/license/1-1/) | Yes | | 2026-09-24 |
| `nemotron_asr` | [nvidia/nemotron-3.5-asr-streaming-0.6b](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b) | [OpenMDW-1.1](https://openmdw.ai/license/1-1/) | Yes | | 2026-09-21 |
| `neutts` | [neuphonic/neutts-2e](https://huggingface.co/neuphonic/neutts-2e) | [NeuTTS Open License v1.0](https://huggingface.co/neuphonic/neutts-2e/blob/main/LICENSE) | Conditional | Commercial use is licensed only while yearly revenue stays under USD 5M. | 2026-09-21 |
| `niagara_asr` | [abr-ai/niagara-19m-batch.en](https://huggingface.co/abr-ai/niagara-19m-batch.en)<br>[abr-ai/niagara-38m-batch.en](https://huggingface.co/abr-ai/niagara-38m-batch.en) | [Applied Brain Research Open License v1.1](https://www.appliedbrainresearch.com/license) | Conditional | Any use while yearly gross revenue, including affiliates, stays under USD 1M; above that, non-commercial research only. | 2026-09-21 |
| `omnivoice` | [k2-fsa/OmniVoice](https://huggingface.co/k2-fsa/OmniVoice) | [CC-BY-NC](https://huggingface.co/k2-fsa/OmniVoice#license) | No | Changed from Apache-2.0 on 2026-07-03, "due to constraints from its training data (e.g., Emilia)". The code is Apache-2.0. The audio tokenizer, [bosonai/higgs-audio-v2-tokenizer](https://huggingface.co/bosonai/higgs-audio-v2-tokenizer), is under the Boson Higgs Audio 2 Community License. | 2026-09-21 |
| `outetts` | [OuteAI/Llama-OuteTTS-1.0-1B](https://huggingface.co/OuteAI/Llama-OuteTTS-1.0-1B) | CC-BY-NC-SA-4.0 | No | | 2026-09-21 |
| `parakeet_tdt` | [nvidia/parakeet-tdt-0.6b-v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) | CC-BY-4.0 | Yes | | 2026-09-21 |
| `parakeet_tdt` | [oruk/orukeet](https://huggingface.co/oruk/orukeet) | CC-BY-SA-4.0 | Yes | The Orukeet r3 weight variant. | 2026-09-21 |
| `personaplex` | [nvidia/personaplex-7b-v1](https://huggingface.co/nvidia/personaplex-7b-v1) | [NVIDIA Open Model License](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/) | Yes | | 2026-09-21 |
| `piper_tts` | [rhasspy/piper-voices](https://huggingface.co/rhasspy/piper-voices/tree/main/en/en_US/lessac/medium) (`en_US-lessac-medium`) | MIT | Unclear | The voice's model card points to the [Lessac Blizzard 2013 dataset license](https://www.cstr.ed.ac.uk/projects/blizzard/2013/lessac_blizzard2013/), which allows non-commercial use only. | 2026-09-21 |
| `pocket_tts` | [kyutai/pocket-tts](https://huggingface.co/kyutai/pocket-tts) | CC-BY-4.0 | Yes | | 2026-09-21 |
| `pulsevad` | [AydinAdnan/PulseVAD](https://github.com/AydinAdnan/PulseVAD) | MIT | Yes | | 2026-09-21 |
| `qwen3_asr` | [Qwen/Qwen3-ASR-0.6B](https://huggingface.co/Qwen/Qwen3-ASR-0.6B)<br>[Qwen/Qwen3-ASR-1.7B-hf](https://huggingface.co/Qwen/Qwen3-ASR-1.7B-hf) | Apache-2.0 | Yes | | 2026-09-21 |
| `qwen3_forced_aligner` | [Qwen/Qwen3-ForcedAligner-0.6B](https://huggingface.co/Qwen/Qwen3-ForcedAligner-0.6B) | Apache-2.0 | Yes | | 2026-09-21 |
| `qwen3_tts` | [Qwen/Qwen3-TTS-12Hz-0.6B-Base](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-Base)<br>[Qwen/Qwen3-TTS-12Hz-1.7B-Base](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base)<br>[Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice)<br>[Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign) | Apache-2.0 | Yes | | 2026-09-21 |
| `rvc` | [lj1995/VoiceConversionWebUI](https://huggingface.co/lj1995/VoiceConversionWebUI) | MIT | Unclear | The base models (HuBERT, RMVPE) are MIT. The packaged voices (`manthos`, `chocola`, `fraise`) have no documented source or license. | 2026-09-21 |
| `sanotts` | [ampixa/sanoTTS](https://huggingface.co/ampixa/sanoTTS) | GPL-3.0 | Yes | Copyleft. | 2026-09-21 |
| `seed_vc` | [Plachta/Seed-VC](https://huggingface.co/Plachta/Seed-VC)<br>[mlx-community/SeedVC-MLX](https://huggingface.co/mlx-community/SeedVC-MLX) | GPL-3.0 | Yes | Copyleft. | 2026-09-21 |
| `sense_asr` | [FunAudioLLM/SenseVoiceSmall](https://huggingface.co/FunAudioLLM/SenseVoiceSmall) | [FunASR Model Open Source License v1.1](https://github.com/modelscope/FunASR/blob/main/MODEL_LICENSE) | Yes | Credit the source and authors, and keep the model names. | 2026-09-21 |
| `sheetsage2` | [m-a-p/SheetSage2](https://huggingface.co/m-a-p/SheetSage2) | CC-BY-NC-4.0 | No | | 2026-09-21 |
| `silero_vad` | [snakers4/silero-vad](https://github.com/snakers4/silero-vad) | MIT | Yes | Bundled in `assets/framework/models`. | 2026-09-21 |
| `soprano_tts` | [ekwek/Soprano-1.1-80M](https://huggingface.co/ekwek/Soprano-1.1-80M) | Apache-2.0 | Yes | | 2026-09-21 |
| `sopro_tts` | [samuel-vitorino/sopro-v2-turbo](https://huggingface.co/samuel-vitorino/sopro-v2-turbo) | Apache-2.0 | Yes | | 2026-09-21 |
| `sortformer_diar` | [nvidia/diar_sortformer_4spk-v1](https://huggingface.co/nvidia/diar_sortformer_4spk-v1) | CC-BY-NC-4.0 | No | | 2026-09-21 |
| `sortformer_diar_v2` | [nvidia/diar_streaming_sortformer_4spk-v2.1](https://huggingface.co/nvidia/diar_streaming_sortformer_4spk-v2.1) | [NVIDIA Open Model License](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/) | Yes | | 2026-09-21 |
| `stable_audio` | [stabilityai/stable-audio-3-small-sfx](https://huggingface.co/stabilityai/stable-audio-3-small-sfx)<br>[stabilityai/stable-audio-3-small-music](https://huggingface.co/stabilityai/stable-audio-3-small-music)<br>[stabilityai/stable-audio-3-medium](https://huggingface.co/stabilityai/stable-audio-3-medium) | [Stability AI Community License](https://huggingface.co/stabilityai/stable-audio-3-small-sfx/blob/main/LICENSE.md) | Conditional | Free while the yearly revenue of you and your affiliates stays under USD 1M; above that an enterprise license is needed. Register commercial use with Stability AI, and display "Powered by Stability AI". The Gemma text encoder falls under the Gemma Terms of Use. | 2026-09-21 |
| `supertonic` | [Supertone/supertonic-3](https://huggingface.co/Supertone/supertonic-3) | [BigScience OpenRAIL-M](https://huggingface.co/Supertone/supertonic-3/blob/main/LICENSE) | Yes | Its use restrictions apply and must be passed on. | 2026-09-21 |
| `universr` | [woongzip1/universr-audio](https://huggingface.co/woongzip1/universr-audio)<br>[woongzip1/universr-speech](https://huggingface.co/woongzip1/universr-speech) | CC-BY-4.0 | Yes | | 2026-09-21 |
| `vevo2` | [RMSnow/Vevo2](https://huggingface.co/RMSnow/Vevo2) | CC-BY-NC-ND-4.0 | No | | 2026-09-21 |
| `vibeasr` | [microsoft/VibeVoice-ASR-BitNet](https://huggingface.co/microsoft/VibeVoice-ASR-BitNet) | MIT | Yes | | 2026-09-21 |
| `vibevoice` | [microsoft/VibeVoice-1.5B](https://huggingface.co/microsoft/VibeVoice-1.5B)<br>[vibevoice/VibeVoice-7B](https://huggingface.co/vibevoice/VibeVoice-7B) | MIT | Yes | | 2026-09-21 |
| `vibevoice_asr` | [microsoft/VibeVoice-ASR](https://huggingface.co/microsoft/VibeVoice-ASR) | MIT | Yes | | 2026-09-21 |
| `vibevoice_asr_streaming` | [microsoft/VibeVoice-ASR-Streaming-7B](https://huggingface.co/microsoft/VibeVoice-ASR-Streaming-7B)<br>[microsoft/VibeVoice-ASR-Streaming-1.5B](https://huggingface.co/microsoft/VibeVoice-ASR-Streaming-1.5B) | MIT | Yes | | 2026-09-21 |
| `vietneu_tts` | [pnnbao-ump/VieNeu-TTS-v3-Turbo](https://huggingface.co/pnnbao-ump/VieNeu-TTS-v3-Turbo) | Apache-2.0 | Yes | | 2026-09-21 |
| `voxcpm1` | [openbmb/VoxCPM-0.5B](https://huggingface.co/openbmb/VoxCPM-0.5B) | Apache-2.0 | Yes | | 2026-09-21 |
| `voxcpm2` | [openbmb/VoxCPM2](https://huggingface.co/openbmb/VoxCPM2) | Apache-2.0 | Yes | | 2026-09-21 |
| `voxtral_realtime` | [mistralai/Voxtral-Mini-4B-Realtime-2602](https://huggingface.co/mistralai/Voxtral-Mini-4B-Realtime-2602) | Apache-2.0 | Yes | | 2026-09-21 |
| `yue2` | [m-a-p/YuE2-3B](https://huggingface.co/m-a-p/YuE2-3B) | CC-BY-NC-4.0 | No | | 2026-09-21 |
| `zipvoice` | [k2-fsa/ZipVoice](https://huggingface.co/k2-fsa/ZipVoice) | Not stated | Unclear | The model card states no license; the [code](https://github.com/k2-fsa/ZipVoice) is Apache-2.0. | 2026-09-21 |

## Adding or updating a row

- Keep one row per family, in alphabetical order. Split a family into several rows only
  when its models carry different licenses.
- Link the original weights, not a GGUF conversion: the conversion keeps the original's
  license.
- Read the license on the original model card and in its LICENSE file. Use the SPDX
  identifier where one exists (`Apache-2.0`, `MIT`, `CC-BY-4.0`). Otherwise use the
  license's own name, linked to its text.
- Pick one of the four values under [Commercial use](#commercial-use). Put any
  condition in Notes, in one or two short sentences: the limit, what must be shown, or
  where the license does not apply.
- Name bundled parts under a different license, such as a codec, tokenizer or text
  encoder, in Notes.
- Set Checked to the date you read the license.
