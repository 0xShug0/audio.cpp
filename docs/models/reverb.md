# Reverb-TTS

`reverb` runs [Reverb-TTS](https://huggingface.co/software-mansion/Reverb-TTS) by
Software Mansion AI Labs: a 297M-parameter zero-shot voice-cloning TTS for English,
German, French, Spanish, Italian, Polish and Portuguese. Reverb is OmniVoice pruned
(24 of 28 layers, 6 KV heads, a 62k text vocabulary with a rank-256 factorized
embedding) and retrained, with its own sampler: the prefix (prompt text, then the
reference frames) never attends to the target, so it runs once and its keys and
values are kept, and the target is filled in 8 main passes (codebooks 0-1) and 2
polish passes (codebooks 2-7). There is no classifier-free guidance batch.

The family reuses OmniVoice's Qwen2 BPE text tokenizer and Higgs Audio v2 codec, so
it depends on the `omnivoice` model target. Everything runs in C++/GGML; Python is
only needed to prepare a package.

Status: experimental. Offline TTS with the 16 preset voices, voice cloning, and
long-form text are supported. Validated on CPU, CUDA and Vulkan (RTX 4090: a warm 4 to
6 s request takes about 95 ms on CUDA and 120 ms on Vulkan). Metal uses the same standard
GGML ops but has not been run yet. Vulkan builds need the SPIR-V headers (`spirv-headers`).

## Build

```sh
cmake -S . -B build -DAUDIOCPP_MODEL_SET=custom -DAUDIOCPP_MODELS=reverb
cmake --build build --target audiocpp_cli audiocpp_server audiocpp_gguf -j 8
```

`AUDIOCPP_MODELS=reverb` links `omnivoice` as a dependency. Add
`-DENGINE_ENABLE_CUDA=ON` or `-DENGINE_ENABLE_VULKAN=ON` for a GPU build. On Vulkan the
first request of each new shape compiles pipelines, so it is slower than the rest.

## Package

Download the release weights, the codec and the preset voices, lay them out as a
package, and convert:

```sh
hf download software-mansion/Reverb-TTS --revision v1.0.0 \
  --include "model/safetensors/fp16/*" --include "model/tokenizer*" --include "model/chat_template.jinja" --include "voices/*" \
  --local-dir Reverb-TTS
hf download eustlb/higgs-audio-v2-tokenizer --revision 0e3a9177c2991016b8e5e17d23aa474ee665bef2 \
  --local-dir higgs-audio-v2-tokenizer
python tools/gguf/reverb_convert.py --reverb Reverb-TTS --codec higgs-audio-v2-tokenizer \
  --output Reverb-TTS-package \
  --audiocpp-gguf build/bin/audiocpp_gguf --gguf Reverb-TTS-GGUF/reverb-tts-q8_0.gguf --type q8_0
```

`reverb_convert.py` needs only numpy and safetensors; it writes the package directory, then runs
`audiocpp_gguf` on it. `--variant int8` or `int4`
starts from the smaller upstream checkpoints and expands them to fp16. The package
directory also loads directly as a safetensors model.

| Package ([JRufer/Reverb-TTS-GGUF](https://huggingface.co/JRufer/Reverb-TTS-GGUF)) | Size | Notes |
|---|---|---|
| `reverb-tts-q8_0.gguf` | 875 MB | Generator Q8_0 (about 340 MB); codec at the converter's Q8 selection, as for OmniVoice |
| `reverb-tts-f16.gguf` | 1.0 GB | |

About 300 MB of the Q8_0 package is the generator; the rest is the Higgs Audio v2
codec (encoder for cloning, decoder for every request).

## Run

Preset voice:

```sh
audiocpp_cli --task tts --family reverb --model Reverb-TTS-GGUF/reverb-tts-q8_0.gguf \
  --text "Hello from Reverb." --request-option voice_id=en_male_narrator_1 --out out.wav
```

Voice cloning needs the reference's exact transcript. 3 to 10 seconds of clean
speech from one speaker works best; the recording's noise and room carry over.

```sh
audiocpp_cli --task clon --family reverb --model Reverb-TTS-GGUF/reverb-tts-q8_0.gguf \
  --text "The final render is ready for review." --voice-ref assets/resources/b.wav \
  --request-option "reference_text=Some call me nature. Others call me Mother Nature. I've been here for over four point five billion years. twenty-two thousand five hundred times longer than you." \
  --request-option language=en --out clone.wav
```

Server (`/v1/audio/speech`; `voice` selects a preset):

```json
{"host": "127.0.0.1", "port": 8080, "backend": "cpu",
 "models": [{"id": "reverb", "family": "reverb", "path": "Reverb-TTS-GGUF/reverb-tts-q8_0.gguf", "task": "tts"}]}
```

Preset voices: `de_female_standard`, `de_male_standard`, `en_female_narrator`
(default), `en_female_whisper`, `en_male_angry`, `en_male_narrator_1`,
`en_male_narrator_2`, `es_male_standard_1`, `es_male_standard_2`,
`fr_female_standard`, `fr_male_standard`, `it_male_expressive`, `it_male_standard`,
`pl_male_standard`, `pt_female_standard`, `pt_male_standard`. A voice can speak any
of the seven languages (`language=...`), most naturally its own.

Request options follow `reverb_tts.TTS.generate` and `SamplerConfig`: `seed`,
`speed`, `duration_sec`, `pace` (`speech` or `recording`), `trim`, and the sampler
settings `num_inference_steps`, `remasking`, `remask_ratio`, `shift`,
`codebook_penalty`, `position_temperature`, `polish_bias`, `polish_guidance`. A text
estimated over 30 s is cut at sentence ends into pieces of about 15 s, each with the
next seed. Session options: `reverb.weight_type` and `reverb.codec_weight_type`.

One deliberate difference from upstream: upstream cuts a fixed 0.2 s from the end
of every take to remove an occasional codec noise burst. That 0.2 s usually holds
the last syllable's fade and sometimes speech, so endings sound clipped. By
default (`end_cut=burst`) this port cuts only a burst that is there: within the
last 0.2 s, a 10 ms window at least 15 dB louder than the quietest one before it
and within 30 dB of the take's peak, cut at its onset. `end_cut=0.2` restores
upstream's cut (the parity check uses it), and any other number of seconds cuts
that much.

## Text normalization

Upstream passes text through [Verbalis](https://github.com/software-mansion-labs/Verbalis),
which reads numbers, dates, money, units and symbols out as words. audio.cpp does not
include it: brackets are dropped and whitespace collapsed, as upstream does after
Verbalis, but `4.5`, `$20` or `22,500` are passed through as written. Send
spoken-form text (and a spoken-form `reference_text` when cloning) for the best
results, the same requirement upstream documents for its ExecuTorch and ruNNtime
programs.

## Validation

See [tests/reverb](../../tests/reverb/README.md). In short, on CPU with fp32
weights the generated codes are identical to `reverb-tts` 1.0.1 for the same prompt
and seed, the text and audio helpers match upstream on randomized inputs, and
Whisper small.en transcribed every word of 12 preset and cloned takes with F16
and Q8_0 packages (0% WER, same as upstream).

## License

Code: Apache-2.0 (upstream) / audio.cpp's license (this port). The Reverb-TTS
weights are [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/):
non-commercial use, with attribution. The Higgs Audio v2 tokenizer keeps the Boson
Higgs Audio 2 Community License. Clone only voices you have the right to use, and
tell listeners the speech is synthetic.
