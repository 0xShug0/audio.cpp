# sherpa-onnx Zipformer keyword spotting

The `sherpa_kws` family runs streaming Zipformer transducer keyword-spotting
exports. Convert a model directory with:

```bash
conda run -n qwen3-tts python tools/models/sherpa_kws/convert_onnx_to_gguf.py \
  /path/to/sherpa-onnx-kws-model /path/to/sherpa-kws-f32.gguf
```

The default `keywords.txt` uses sherpa-onnx tokenized keyword rows. A request
can replace it with `keywords=...`; separate rows with newlines or `/`. Each
row contains token symbols and can include `:score`, `#threshold`, and
`@label`, matching sherpa-onnx keyword files.
