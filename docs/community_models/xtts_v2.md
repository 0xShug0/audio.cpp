# XTTS v2

Coqui XTTS v2 provides offline multilingual text-to-speech with a speaker reference.

| Field | Value |
| --- | --- |
| Family | `xtts_v2` |
| Task / mode | `tts` / `offline` |
| Languages | en, es, fr, de, it, pt, pl, tr, ru, nl, cs, ar, zh-cn, hu, ko, ja, hi |
| Voice input | Required reference WAV; 6–30 seconds recommended |
| Output | Mono 24 kHz audio |

## Weights

Packages in [the model spec](../../model_specs/xtts_v2.json) download from
[onicubag/xTTSv2-GGUF](https://huggingface.co/onicubag/xTTSv2-GGUF).
Choose a package containing one GPT GGUF and `xtts-v2-vocoder-f16.gguf`. Q8_0 is the default; the original GPT file and Q4_K_M are also available.
`mel_norms.bin` is required only when neither GGUF embeds normalization
metadata or a `mel_stats` tensor.
Pass the directory containing these files, or its GPT GGUF, as `--model`.

## Usage

```bash
audiocpp_cli --task tts --family xtts_v2 --model models/XTTS-v2-Q8 \
    --backend cuda --threads 8 \
    --text "Hello, my name is Anna and this is a test." \
    --language en --voice-ref reference.wav \
    --out out.wav --log
```

Request options: `language`, `temperature`, `top_k`, `top_p`,
`repetition_penalty`, `max_mel_tokens`, `seed`, `text_chunk_size`, and
`text_chunk_mode`. Set them using
`--request-option name=value`. Defaults and limits are in the model spec.
The reference is mixed to mono and resampled internally. Repeated requests
within a session reuse prepared conditioning. Set
`--session-option xtts_v2.conditionals_cache_slots=2` to retain two voices;
the default is 1, and 0 disables caching. The least recently used voice is evicted
when the slots are full.

Use `--text-chunk-size N` and `--text-chunk-mode default|tag_aware|japanese|endline`
to control the framework text chunker. Defaults are 256 Unicode codepoints and
word-budget splitting. Chunking modes do not add model support for emotion or
other text tags.

## Validated scope

The local validation covers English offline CUDA, Vulkan and CPU generation, repeated requests,
seed reproducibility, long text, changed references, stereo reference mixdown,
and German UTF-8 tokenization. It does not establish full multilingual text
normalization parity: upstream's Chinese/Japanese/Korean romanization and
non-English number normalization are not implemented here.

## License

The model weights use the [Coqui Public Model License](https://huggingface.co/coqui/XTTS-v2#license).
