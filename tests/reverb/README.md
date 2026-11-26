# Reverb-TTS validation

Run from the audio.cpp repository root. Build with model tests:

```sh
cmake -S . -B build -DAUDIOCPP_MODEL_SET=custom -DAUDIOCPP_MODELS=reverb \
  -DENGINE_BUILD_TESTS=ON -DENGINE_BUILD_MODEL_TESTS=ON
cmake --build build --target audiocpp_cli audiocpp_gguf reverb_rules_test reverb_frontend_probe -j 8
ctest --test-dir build -R reverb_rules_test   # unit test (cmake/tests/unit.cmake)
python tools/check_loader_catalog_sync.py
```

The Python checks need the upstream package and its references:

```sh
pip install "reverb-tts @ git+https://github.com/software-mansion-labs/Reverb-TTS@2c72a7f9e9b7" torchaudio soundfile transformers
hf download software-mansion/Reverb-TTS --revision v1.0.0 \
  --include "model/safetensors/fp16/*" --include "model/tokenizer*" --include "model/chat_template.jinja" --include "voices/*" \
  --local-dir Reverb-TTS
hf download eustlb/higgs-audio-v2-tokenizer --revision 0e3a9177c2991016b8e5e17d23aa474ee665bef2 \
  --local-dir higgs-audio-v2-tokenizer
hf download openai/whisper-small.en --local-dir whisper-small.en
python tools/gguf/reverb_convert.py --reverb Reverb-TTS --codec higgs-audio-v2-tokenizer --output Reverb-TTS-package \
  --audiocpp-gguf build/bin/audiocpp_gguf --gguf Reverb-TTS-GGUF/reverb-tts-q8_0.gguf --type q8_0
```

| Check | Command | What it shows |
|---|---|---|
| Rules | `build/bin/reverb_rules_test` | Commit schedule bit-exact to PyTorch float32; length estimate, Python rounding, text preparation, splitting, trimming and loudness raise on fixed cases. No downloads. |
| Frontend parity | `python tests/reverb/check_frontend.py build/bin/reverb_frontend_probe` | 300 random multilingual texts: preparation, length estimate and splitting identical to `reverb_tts.text`. Kaiser resampling (5 rate pairs, max error ~2e-7), `raise_quiet`, `speech_length` and `trim_silence` against `reverb_tts.audio`. |
| End-to-end parity | `python tests/reverb/check_parity.py --cli build/bin/audiocpp_cli --package Reverb-TTS-package --reverb Reverb-TTS --codec higgs-audio-v2-tokenizer` | Same text, preset and seed through `reverb_tts.TTS.generate` and audiocpp_cli (`reverb.weight_type=f32`): equal length and waveform correlation 1.0, i.e. identical generated codes. |
| Intelligibility | `python tests/reverb/check_quality.py --cli build/bin/audiocpp_cli --whisper whisper-small.en --model <gguf> [--reverb Reverb-TTS --codec higgs-audio-v2-tokenizer]` | Whisper small.en WER over six sentences, preset and cloned voice, in one long-lived session. |

`reverb_rules_test` (unit category, `ENGINE_BUILD_TESTS`) needs no downloads; `reverb_frontend_probe`
(model category, `ENGINE_BUILD_MODEL_TESTS`) is a manual probe driven by `check_frontend.py`. The rules
test's constants come from PyTorch 2.x (`torch.linspace` + the t_shift warp, printed
with `float.hex`) and `reverb_tts.text.estimate_frames`. See [validation.md](validation.md) for
the recorded results.
