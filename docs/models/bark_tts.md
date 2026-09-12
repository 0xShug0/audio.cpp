# Bark Small

Bark is a multilingual, expressive text-to-audio model. The audio.cpp package
contains the complete Bark Small semantic, coarse, fine, and EnCodec pipeline,
plus all 261 upstream speaker histories.

```sh
python3 tools/model_manager_v2.py install bark_small_f16

audiocpp_cli --task tts --family bark_tts \
  --model Bark-Small-GGUF/bark-small-f16.gguf \
  --text "Hello from Bark. [laughs] This model can be quite expressive." \
  --option history_prompt=v2/en_speaker_6 --output bark.wav
```

Use `--option history_prompt=<preset>` to select another packaged history. Presets
follow upstream names such as `v2/de_speaker_3`, `v2/ja_speaker_0`, or
`announcer`. `temperature`, `top_k`, `max_tokens`, and `seed` are available as
request options. Bark supports non-speech cues written in brackets, although
the result remains probabilistic.

F16 is the recommended package. The hybrid Q8 package keeps both autoregressive
semantic/coarse transformers and the complete EnCodec path in F16, and only
quantizes the non-causal fine transformer's dense matrices. Quantizing Bark's
autoregressive stages causes token errors to compound and is not quality-safe.

Source model: [suno/bark-small](https://huggingface.co/suno/bark-small), pinned
to revision `1dbd7a128513b8ae4a4e2130fed57b7ac9da5bcd`. Bark is MIT licensed.
