# AST AudioSet

AST AudioSet classifies an audio file into the 527 sound classes used by AudioSet. It uses the
[MIT `ast-finetuned-audioset-10-10-0.4593` checkpoint](https://huggingface.co/MIT/ast-finetuned-audioset-10-10-0.4593).
The default self-contained package is
`AST-AudioSet-GGUF/ast-audioset-f32.gguf` under
`audio-cpp/audio.cpp-gguf`. Only F32 is packaged.

```bash
build/debug/bin/audiocpp_cli \
  --task cls \
  --family ast_audioset \
  --model /path/to/AST-AudioSet-GGUF/ast-audioset-f32.gguf \
  --audio input.wav \
  --backend cuda \
  --threads 8 \
  --request-option top_k=10 \
  --log
```

## Request options (use with `--request-option`)

| Option | Values | Default | Description |
| --- | --- | --- | --- |
| `top_k` | `1` to `527` | `10` | Number of highest-scoring AudioSet classes to return. |

## Session options (use with `--session-option`)

| Option | Values | Default | Description |
| --- | --- | --- | --- |
| `ast_audioset.attention` | `auto`, `flash`, `eager` | `auto` | Attention lowering. |
| `ast_audioset.weight_type` | Supported tensor storage type | `native` | Weight storage override. |

The model accepts ordinary audio inputs and resamples them to 16 kHz mono. Audio beyond the model's fixed
10.24-second feature window is truncated, matching the official feature extractor.
