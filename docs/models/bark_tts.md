# Bark Small

Bark is a multilingual, expressive text-to-audio model. The audio.cpp package
contains the complete Bark Small semantic, coarse, fine, and EnCodec pipeline,
plus all 261 upstream speaker histories.

```sh
python3 tools/model_manager_v2.py install bark_small_q8_0

audiocpp_cli --task tts --family bark_tts \
  --model Bark-Small-GGUF/bark-small-q8_0.gguf \
  --text "Hello from Bark. [laughs] This model can be quite expressive." \
  --option history_prompt=v2/en_speaker_6 --output bark.wav
```

Use `--option history_prompt=<preset>` to select another packaged history. Presets
follow upstream names such as `v2/de_speaker_3`, `v2/ja_speaker_0`, or
`announcer`. `temperature`, `top_k`, `max_tokens`, and `seed` are available as
request options. Bark supports non-speech cues written in brackets, although
the result remains probabilistic.

The Q8 package quantizes transformer matrix weights. Embeddings, normalization
parameters, EnCodec codebooks, recurrent weights, and convolution weights are
kept at 16-bit precision to protect synthesis quality.

Source model: [suno/bark-small](https://huggingface.co/suno/bark-small), pinned
to revision `1dbd7a128513b8ae4a4e2130fed57b7ac9da5bcd`. Bark is MIT licensed.
