# sherpa-onnx Zipformer keyword spotting

The `sherpa_kws` family runs streaming Zipformer transducer keyword-spotting
exports. Convert a model directory with:

```bash
conda run -n qwen3-tts python tools/models/sherpa_kws/convert_onnx_to_gguf.py \
  /path/to/sherpa-onnx-kws-model /path/to/sherpa-kws-f32.gguf
```

The converter defaults to the original F32 encoder, decoder, and joiner ONNX
files. The default self-contained package is
`Sherpa-KWS-GGUF/sherpa-kws-zipformer-zh-en-3m-f32.gguf` under
`audio-cpp/audio.cpp-gguf`; it embeds configuration, tokens, keywords, and the
model spec. Only F32 is packaged.

```bash
build/debug/bin/audiocpp_cli --family sherpa_kws --task wake \
  --model /path/to/sherpa-kws-zipformer-zh-en-3m-f32.gguf \
  --backend cpu --threads 4 --audio input.wav --segments-out detections.json --log
```

Use `--mode streaming` for incremental input. Detections are speech segments
with the keyword in `text`, its confidence, and absolute start/end sample positions.

The default `keywords.txt` uses sherpa-onnx tokenized keyword rows. A request
can replace it with `keywords=...`; separate rows with newlines or `/`. Each
row contains token symbols and can include `:score`, `#threshold`, and
`@label`, matching sherpa-onnx keyword files.

See [CPU/CUDA measurements and parity limits](../reports/wake_word_cpu_cuda.md).
