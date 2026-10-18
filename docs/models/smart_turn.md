# Smart Turn v3.2

Smart Turn estimates whether a speaker has finished a conversational turn. It
uses the most recent eight seconds of 16 kHz mono audio, left-padding shorter
recordings with silence. The output is a completion probability and a Boolean
decision, not speech segments or a transcript.

The model uses the [official Smart Turn v3.2 checkpoint](https://huggingface.co/pipecat-ai/smart-turn-v3).
Only the F32 GGUF is provided; no quantized variants are packaged.

## CLI

```bash
audiocpp_cli --task turn --family smart_turn \
  --model /path/to/smart-turn-v3.2-f32.gguf \
  --backend cpu --audio input.wav --log
```

The CLI prints a `custom_schema_output` JSON object with schema `smart_turn.v1` and
`data.complete` / `data.probability`. The decision is complete when the
probability is strictly greater than the threshold.

## Options

### Request Options (`--request-option`)

| Name | Values | Default | Description |
|---|---|---|---|
| `threshold` | 0 to 1 | `0.5` | Completion probability threshold. |

## Server

Configure a model with `family: "smart_turn"`, `task: "turn"`, and
`mode: "offline"`. Use `/v1/tasks/run` to obtain `custom_schema_output.data.complete`
and `custom_schema_output.data.probability`. The input must be 16 kHz mono audio.

```bash
curl http://127.0.0.1:8080/v1/tasks/run \
  -H 'Content-Type: application/json' \
  -d '{"model":"smart-turn","request":{"audio":"input.wav","options":{"threshold":"0.5"}}}'
```

The response includes `custom_schema_output` in this shape:

```json
{"custom_schema_output":{"schema":"smart_turn.v1","data":{"complete":true,"probability":0.57}}}
```
