# d1-Omni-2B audio decisions

Family: `lfm_d1_omni`. Upstream: [LiquidAI/d1-Omni-2B](https://huggingface.co/LiquidAI/d1-Omni-2B).
Use Liquid AI's [published GGUFs](https://huggingface.co/LiquidAI/d1-Omni-2B-GGUF);
no audio.cpp conversion or repackaging is required.

This integration answers named questions about an audio recording. It reads the
recording and questions in a full-sequence forward pass and returns calibrated
answers as JSON. It does not generate speech or a transcript. Image, video, and
text-state inputs are not exposed by this integration.

## Download and run

Keep the main GGUF and its matching `mmproj-` file together. Their precision
suffixes must match. Q8_0, F16, and BF16 pairs are available upstream.

```bash
hf download LiquidAI/d1-Omni-2B-GGUF \
  d1-Omni-2B-Q8_0.gguf mmproj-d1-Omni-2B-Q8_0.gguf \
  --local-dir models/d1-Omni-2B-GGUF

build/debug/bin/audiocpp_cli \
  --family lfm_d1_omni --task cls --backend cuda \
  --model models/d1-Omni-2B-GGUF/d1-Omni-2B-Q8_0.gguf \
  --audio recording.wav \
  --request-option 'questions={"speech":{"type":"noul","instructions":"Does someone speak in the recording?"},"sound":{"type":"choice","instructions":"What is the main sound?","criteria":{"speech":"A person speaking","music":"Music playing","noise":"Background noise"}}}' \
  --log
```

The result is an `audio_decisions` custom-schema output containing `answers`
and `usage`. Answer names match the input question names. `output_tokens` is
zero because the model reads answer slots rather than generating tokens.

## Model

| Field | Value |
|---|---|
| Family | `lfm_d1_omni` |
| Tasks | `classify` |
| Modes | `offline` |
| Languages | `en`, `ar`, `zh`, `fr`, `de`, `it`, `ja`, `ko`, `pt`, `es`, `ru`, `pl` |
| Default package | `lfm_d1_omni_2b_q8_0` |
| Input | Audio recording and named questions |
| Output | JSON audio decisions |

## Common Options (use directly)

| Option | Meaning |
|---|---|
| `--model` | Model file or directory shown in the examples. |
| `--backend` | Select an available backend supported by this model. |
| `--threads` | CPU thread count. |
| `--log` | Enable runtime logging. |
| `--audio` | Input audio, where required by the selected task. |

## Request Options (use with `--request-option`)

| Option | Type / values | Default | Description |
|---|---|---|---|
| `questions` | `string` | Required | JSON object mapping question names to typed questions, in input order. |
| `set_size` | `int` | `16` | Questions packed together per forward pass. Larger sets may affect accuracy. Declared range: minimum 1. |
| `score_readout` | `median`, `mean`, `mode` | `median` | How ordered score probabilities are reduced to a score. |

## Session Options (use with `--session-option`)

| Option | Type / values | Default | Description |
|---|---|---|---|
| `lfm_d1_omni.attention` | `auto`, `flash`, `eager` | `auto` | Attention lowering for the audio encoder and decoder. |

## Questions

`questions` is a JSON object whose insertion order is preserved, including the
order of choice criteria. Instructions are written questions about the supplied
audio.

In request-sequence files and server requests, `options.questions` is a
JSON-encoded string. Question and criterion order form part of the model input.

| Type | Question fields | Answer fields |
| --- | --- | --- |
| `noul` | `instructions`; optional `criteria` with `true`/`false` descriptions | `noul`: probability of yes |
| `choice` | `instructions`, `criteria`: label-to-description object | `choice`, `confidence`, `probabilities` |
| `choice`, `multiple: true` | Same criteria object | `choices`, independent per-label `probabilities` |
| `score` | `instructions`, `criteria`: ordered array of 2–10 levels | `score`, `confidence`, `probabilities`, `legend` |
| `number` with `range` | `instructions`, `range: [low, high]`; optional `step` and `unit` | `number`, `interval`, `unit` |

Numeric estimates return the median and a p10–p90 interval. Text/date extraction,
boxes, and spoken question instructions are not currently exposed.

`set_size` defaults to the trained value of 16 questions per forward pass. A
larger value can change accuracy. `score_readout` selects `median` (default),
`mean`, or `mode` for ordered score questions. These are not sampling controls.
The model has no temperature or token-generation request options.

The `lfm_d1_omni.attention` session option accepts `auto` (default), `flash`, or
`eager`. The audio frontend resamples to 16 kHz and follows upstream's log-mel
and normalization recipe. It does not silently truncate long recordings.

## License

The weights use the [LFM Open License v1.0](https://huggingface.co/LiquidAI/d1-Omni-2B/blob/main/LICENSE).
The license is included alongside this integration. Review its commercial-use
conditions before deploying the model.
