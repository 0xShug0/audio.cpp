# Haas-ja-v1.0

[Haas-ja-v1.0](https://huggingface.co/third-intelligence/Haas-ja-v1.0) is a
Japanese ASR model. It reuses Parakeet-TDT's FastConformer encoder and TDT joint,
replacing the LSTM prediction network with two Kimi Delta Attention blocks.
Select the `haas_asr` family. Its registration reuses the Parakeet runtime and
selects the KDA predictor; checkpoint variant labels do not select behavior.

```bash
./build/debug/bin/audiocpp_cli --task asr --family haas_asr \
  --model models/Haas-ja-v1.0-GGUF/haas-ja-v1.0-q8_0.gguf \
  --backend cuda --session-option haas_asr.matmul_weight_type=q8_0 \
  --audio japanese.wav --text-out transcript.txt --log
```

F16 and Q8_0 packages are available. The checkpoint is Japanese-only. The
encoder uses full-context attention; the family's buffered streaming interface
does not make it a native cache-aware streaming encoder. Long-form splitting
uses the existing Parakeet runtime policy rather than upstream's quiet-point
splitting policy.

To convert the official checkpoint, download `Haas-ja-v1.0.nemo` from the
upstream repository, then run:

```bash
python tools/community_models/convert_haas.py \
  --source Haas-ja-v1.0.nemo --staging haas-converted \
  --converter build/debug/bin/audiocpp_gguf --type q8_0 \
  --output haas-ja-v1.0-q8_0.gguf
```

Conversion requires PyTorch, PyYAML, NumPy, safetensors, SentencePiece,
tokenizers, protobuf and Transformers (tested with Transformers 5.12.1 and
protobuf 5.29.5). NeMo and flash-linear-attention are needed only
to run the official Python implementation, not for conversion or C++ inference.

The existing `haas_asr.matmul_weight_type=q8_0` session option also keeps
load-time packed and scaled projections in Q8_0. With `native`, projections
constructed from float32 values retain float32 storage. Omit the override when
using the F16 package if that precision is preferred.

Timestamp word boundaries follow the tokenizer’s word markers. Continuous
Japanese text can therefore occupy one phrase-sized span; this does not add
Japanese morphological word segmentation.
