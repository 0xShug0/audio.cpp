# Kroko Community ASR

`kroko_asr` is a native audio.cpp port of the free Kroko Community
Zipformer2/RNN-T models. The Kaldi-compatible filterbank, streaming
Conv2dSubsampling/ConvNeXt encoder, 19-layer Zipformer2, stateless predictor,
joiner, and greedy decoder run without ONNX Runtime.

## Capabilities

| Field | Value |
|---|---|
| Task | `asr` |
| Modes | `offline`, native stateful `streaming` |
| Public free languages | German (`de`), English (`en`), Spanish (`es`), French (`fr`), Italian (`it`), Hebrew (`he`; package code `IW`), Dutch (`nl`), Portuguese (`pt`), Swedish (`sv`), Turkish (`tr`) |
| Input | WAV; audio.cpp converts to 16 kHz mono |
| Output | Transcript and word timestamps |
| Package variants | 64-L and 128-L streaming packages |
| Native layouts | Converted safetensors and standalone GGUF |

Each Kroko package recognizes one language. Select a package whose language
matches `--language`; `auto` uses the package language. The loader normalizes
the legacy Hebrew code `iw` to `he`.

Streaming keeps the subsampling cache, Zipformer layer states, RNN-T predictor
context, emitted tokens, and emission frames across chunks. For 16 kHz input,
already consumed waveform is compacted while retaining the filterbank boundary
overlap, so the streaming audio buffer remains bounded. Finalization adds the
same 660 ms zero tail as the original Kroko/sherpa runner to flush final
punctuation and tokens.

Word starts come from the RNN-T encoder frame where the first token of the word
is emitted. One encoder frame is 40 ms (`160` filterbank-hop samples times
subsampling factor `4`). `--words-out` writes audio.cpp sample spans.

## Source packages and conversion

Free packages are published in
[Banafo/Kroko-ASR](https://huggingface.co/Banafo/Kroko-ASR). Their `.data`
container holds a JSON header, quantized encoder/decoder/joiner ONNX graphs,
and `tokens.txt`. Commercial/encrypted packages are intentionally rejected;
use Kroko's licensed runtime for those models.

Install converter dependencies:

```powershell
python -m pip install numpy onnx safetensors
```

The model manager infers a language/size-specific target such as
`Kroko-DE-Community-64-L-Native`, so different languages do not overwrite one
another:

```powershell
python .\tools\model_manager.py install kroko_asr_community_converted `
  --source-file .\models\Kroko-ASR\Kroko-DE-Community-64-L-Streaming-001.data `
  --models-root .\models\Kroko-ASR `
  --overwrite
```

Use `--variant <directory-name>` to override the inferred target directory.
The converter can also be called directly:

```powershell
python .\tools\community_models\convert_kroko_onnx.py `
  .\models\Kroko-ASR\Kroko-SV-Community-64-L-Streaming-001.data `
  .\models\Kroko-ASR\Kroko-SV-Community-64-L-Native `
  --overwrite
```

The result contains:

```text
Kroko-SV-Community-64-L-Native/
|-- config.json
|-- model.safetensors
`-- tokens.txt
```

The package metadata follows the current split catalog:

- `model_specs/kroko_asr.json` contains only the runtime GGUF/safetensors
  resource layout used by loaders and embedded standalone GGUF metadata.
- `model_specs_v1/kroko_asr.json` contains the family catalog metadata,
  languages, capabilities, options, and converter-package description.

The converter supports both public chunk layouts (`141/128` feature frames for
64-L and `269/256` for 128-L). It dequantizes `MatMulInteger` tensors, recovers
folded Zipformer/downsampling constants and both exported forms of chunk-edge
scales, ignores k2 disambiguation symbols beyond the joiner vocabulary, and
writes semantic audio.cpp tensor names.

## CLI

Offline safetensors transcription with word timestamps:

```powershell
.\build\windows-cuda-release\bin\audiocpp_cli.exe `
  --task asr --family kroko_asr `
  --model .\models\Kroko-ASR\Kroko-EN-Community-128-L-Native `
  --backend cpu --audio .\SAMPLES\EN_3.wav --language en `
  --text-out .\outputs\kroko_en.txt `
  --words-out .\outputs\kroko_en_words.json --log
```

Native streaming:

```powershell
.\build\windows-cuda-release\bin\audiocpp_cli.exe `
  --task asr --mode streaming --family kroko_asr `
  --model .\models\Kroko-ASR\Kroko-SV-Community-64-L-Native `
  --backend cpu --audio .\speech_sv.wav --language sv `
  --text-out .\outputs\kroko_sv.txt `
  --words-out .\outputs\kroko_sv_words.json --log
```

## Standalone GGUF

```powershell
.\build\windows-cuda-release\bin\audiocpp_gguf.exe `
  --input .\models\Kroko-ASR\Kroko-SV-Community-64-L-Native\model.safetensors `
  --root .\models\Kroko-ASR\Kroko-SV-Community-64-L-Native `
  --family kroko_asr --type q8_0 `
  --output .\models\Kroko-ASR\Kroko-SV-Community-64-L-Native\Kroko-SV-Community-64-L-Q8.gguf `
  --overwrite
```

The GGUF embeds `config.json`, `tokens.txt`, and the `kroko_asr` package spec.
It can therefore be moved or renamed and passed directly to `--model`.

```powershell
.\build\windows-cuda-release\bin\audiocpp_cli.exe `
  --task asr --mode streaming --family kroko_asr `
  --model .\models\Kroko-ASR\Kroko-SV-Community-64-L-Native\Kroko-SV-Community-64-L-Q8.gguf `
  --backend cuda --audio .\speech_sv.wav --language sv `
  --words-out .\outputs\kroko_sv_q8_words.json
```

## Server

Configure either mode. This example exposes streaming SSE:

```json
{
  "host": "127.0.0.1",
  "port": 8080,
  "backend": "cuda",
  "models": [
    {
      "id": "kroko-sv-stream",
      "family": "kroko_asr",
      "path": "models/Kroko-SV-Community-64-L-Q8.gguf",
      "task": "asr",
      "mode": "streaming"
    }
  ]
}
```

```powershell
.\build\windows-cuda-release\bin\audiocpp_server.exe --config .\server.json --log

curl.exe -N http://127.0.0.1:8080/v1/audio/transcriptions `
  -F "file=@speech_sv.wav" -F "model=kroko-sv-stream" `
  -F "language=sv" -F "stream=true" -F "response_format=json"
```

The generic `/v1/tasks/run` and `/v1/tasks/stream` result schemas carry the
model's `word_timestamps`. The OpenAI-compatible transcription route currently
returns its normal text/delta schema.

## Validation

The complete reproducible commands, per-request multilingual ONNX comparison,
64-L and 128-L tensor-boundary parity, streaming/offline equality, standalone
GGUF path test, server results, timings, and memory notes are in
[Kroko ASR validation](../reports/kroko_asr_validation.md).

## Known limitations

- Only public packages with `free=true` are converted. Commercial/encrypted
  packages require Kroko's license and runtime.
- Each package is single-language; audio.cpp does not implement Kroko's
  multi-model language router.
- Greedy RNN-T decoding is implemented. Upstream modified beam search,
  hotwords, blank penalty, and automatic endpoint segmentation are not yet
  exposed.
- The 16 kHz streaming path compacts consumed waveform. Other input sample
  rates remain correct but retain the accumulated source waveform until
  finalization.
- The current CPU runtime is parity-focused and slower than the optimized
  ONNX Runtime reference; see the validation report.

Source references:

- [kroko-ai/kroko-onnx](https://github.com/kroko-ai/kroko-onnx)
- [Banafo/Kroko-ASR](https://huggingface.co/Banafo/Kroko-ASR)
- [Kroko API documentation](https://docs.kroko.ai/api/)
