# Qwen3 Forced Aligner

For shared conversion guidance, see
[Qwen3 GGUF checkpoints](qwen3.md#gguf-checkpoints).

The forced aligner maps an exact transcript onto speech audio. It is not an ASR route: the transcript is required input.
Standalone forced alignment does not chunk audio because exact transcript/audio
chunk pairing cannot be inferred safely from a raw transcript. For long-audio
timestamping, use Qwen3 ASR with `--words-out` and
`qwen3_asr.forced_aligner_path`; ASR chunks the audio first, then aligns
each recognized transcript to its matching audio chunk.

| Field | Value |
|---|---|
| Family | `qwen3_forced_aligner` |
| GGUF model | `models/Qwen3-ForcedAligner-0.6B-GGUF/qwen3-forced-aligner-0.6b-q8_0.gguf` |
| Task | `align` |
| Modes | `offline` |
| Input | Speech WAV plus exact transcript |
| Output | Word timestamp JSON through `--words-out` |

```bash
audiocpp_cli --task align --family qwen3_forced_aligner --model models/Qwen3-ForcedAligner-0.6B-GGUF/qwen3-forced-aligner-0.6b-q8_0.gguf --backend cuda --audio assets/resources/sample_16k.wav --text "Some call me nature, others call me Mother Nature." --language English --words-out words.json
```

## Japanese and Korean word boundaries

Japanese alignment uses Nagisa's learned word segmenter, matching the official
Qwen implementation. Korean alignment uses the official `qwen-asr` dictionary
and its longest-prefix splitting rule. Both accept language codes (`ja`, `ko`)
and language labels (`Japanese`, `Korean`). Japanese segmentation runs on CPU;
the aligner can continue to use CUDA, Vulkan or CPU.

For a local checkpoint, export the tokenizer assets from the reference Python
environment before converting the model:

```bash
python tools/models/export_qwen3_aligner_tokenizers.py \
  --output-dir models/Qwen3-ForcedAligner-0.6B \
  --korean-dictionary /path/to/qwen_asr/inference/assets/korean_dict_jieba.dict
```

The exporter requires `nagisa` and `safetensors`. The reference `qwen-asr==0.0.6`
uses `nagisa==0.2.11`. It writes `nagisa.json`, `nagisa.safetensors` and
`korean_dict_jieba.dict`; the model spec declares these as optional resources.
The JSON and dictionary are embedded as sidecars; pass Nagisa weights as a
second tensor input to the existing converter:

```bash
audiocpp_gguf --family qwen3_forced_aligner \
  --input models/Qwen3-ForcedAligner-0.6B/model.safetensors \
  --input nagisa=models/Qwen3-ForcedAligner-0.6B/nagisa.safetensors \
  --root models/Qwen3-ForcedAligner-0.6B \
  --model-spec model_specs/qwen3_forced_aligner.json --type f16 \
  --keep-type "nagisa/*=f32" \
  --output qwen3-forced-aligner-0.6b-f16.gguf
```
To update an existing GGUF without changing any of its tensor types or payloads,
use that GGUF as the first `--input`, add the same `nagisa=...` input and use
`--type native` instead of `--type f16`. This preserves mixed Q8/BF16 packages;
do not build their replacements through an intermediate F16 package.
Older GGUFs still run using the previous Japanese/Korean word segmentation,
with a warning that timestamp accuracy is reduced. Download the updated GGUF
or repack it with these resources to use the upstream word boundaries.
Present but invalid tokenizer assets remain an error. Other languages do not
require these assets.

Nagisa's POS tagger is unnecessary for word boundaries and is not exported.
See the [Nagisa license](../../tests/qwen3_forced_aligner/LICENSE.nagisa) and
[Qwen ASR license](../../tests/qwen3_forced_aligner/LICENSE.qwen-asr).

## Common Options (use directly)

| Option | Values | Default | Meaning |
|---|---|---:|---|
| `--audio` | WAV path | required | Speech audio at any sample rate; timestamps are reported in that file's sample domain. |
| `--text` | exact transcript | required | Transcript to align. |
| `--language` | language code | required | Transcript language. |
| `--audio-chunk-mode` | `auto`, `none` | `auto` | Standalone forced alignment runs one audio/transcript pair. `fixed` and `vad` are rejected because transcript chunk boundaries would be ambiguous. |
| `--words-out` | JSON path | not set | Word timestamp output. |

## Request Options (use with `--request-option`)

| Option | Values | Default | Meaning |
|---|---|---:|---|
| `clamp_timestamps_to_audio` | bool | `false` | Opt-in guard for word timestamps: keep repaired spans inside the local audio input. Default preserves existing timestamp repair behavior. |

## Session Options (use with `--session-option`)

| Option | Values | Default | Meaning |
|---|---|---:|---|
| `qwen3_forced_aligner.weight_type` | `native`, `f32`, `f16`, `bf16`, `q8_0` | `native` | Aligner thinker weight type. |
