# Sidon

Sidon v0.1 restores single-speaker speech and produces mono audio at 48 kHz.
This is the original Sidon model, not DialogueSidon.

## Usage

```bash
audiocpp_cli --task s2s --family sidon \
  --model /path/to/Sidon-GGUF/sidon-v0.1-f32.gguf \
  --audio input.wav --backend cuda --threads 8 --log --out restored.wav
```

Stereo input is mixed to mono. The frontend normalizes the peak, applies a
50 Hz high-pass filter, and resamples to 16 kHz. Output is trimmed to the
original recording's duration.

Long recordings follow the released demo's 96-second chunks with a one-frame
encoder carry between chunks. This is offline processing, not streaming.

## Conversion

Download `feature_extractor_cuda.pt` and `decoder_cuda.pt` from
[sarulab-speech/sidon-v0.1](https://huggingface.co/sarulab-speech/sidon-v0.1).
Build `audiocpp_gguf`, then run:

```bash
python tests/sidon/convert_gguf.py \
  --model-dir /path/to/sidon-v0.1 \
  --output /path/to/Sidon-GGUF/sidon-v0.1-f32.gguf
```

The converter extracts the frozen TorchScript constants, folds the encoder's
LoRA adapters, and embeds the model spec. The resulting GGUF does not require
the original TorchScript files at runtime.

## Upstream

- [Model checkpoint](https://huggingface.co/sarulab-speech/sidon-v0.1)
- [Source and license](https://github.com/sarulab-speech/Sidon)
