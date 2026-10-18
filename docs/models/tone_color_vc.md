# Tone Color VC

Tone Color VC is the standalone tone color converter from OpenVoice V2.
It converts source speech to the speaker in a reference recording.
The standalone converter includes its reference encoder, flow model, and waveform
decoder. It does not require a text-to-speech model or a transcript.

## Quick Start

```bash
audiocpp_cli --task vc --family tone_color_vc \
  --model models/Tone-Color-VC-GGUF/tone-color-vc-f32.gguf \
  --backend cuda --audio source.wav --voice-ref reference.wav \
  --out converted.wav
```

The output is mono, 22,050 Hz audio. Inputs are mixed down to mono. Other sample
rates require libsoxr; otherwise resample the input WAVs to 22,050 Hz first.
Conversion is offline. No separate watermark model is applied.

F32 and F16 packages are available. F16 keeps biases and normalization vectors
in F32. F32 remains the default; F16 is the smaller alternative.

## Common Options (use directly)

| Option | Values | Default | Meaning |
|---|---|---|---|
| `--audio` | WAV path | required | Source speech. |
| `--voice-ref` | WAV path | required | Target speaker reference. |
| `--seed` | integer >= -1 | `1234` | Sampling seed; `-1` chooses a random seed. |

## Request Options (use with `--request-option`)

| Option | Values | Default | Meaning |
|---|---|---|---|
| `seed` | integer >= -1 | `1234` | Sampling seed. |
| `temperature` | number >= 0 | `0.3` | Posterior noise scale, called `tau` upstream. Zero disables posterior sampling noise. |

There are no model-specific session or load options.

## Server

Add an offline VC entry to your server configuration:

```json
{
  "models": [
    {
      "id": "tone_color_vc",
      "family": "tone_color_vc",
      "path": "/path/to/tone-color-vc-f32.gguf",
      "task": "vc",
      "mode": "offline"
    }
  ]
}
```

Submit source and reference WAV paths accessible to the server:

```bash
curl http://localhost:8080/v1/tasks/run \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "tone_color_vc",
    "request": {
      "audio": "/path/to/source.wav",
      "voice_ref": "/path/to/reference.wav",
      "options": {"seed": "1234", "temperature": "0.3"}
    }
  }'
```

The JSON response contains a base64-encoded WAV in `audio`, plus `sample_rate`,
`channels`, and `timing`.

## Convert Weights

Download the `converter` directory from
[myshell-ai/OpenVoiceV2](https://huggingface.co/myshell-ai/OpenVoiceV2/tree/main/converter)
and obtain the [OpenVoice source](https://github.com/myshell-ai/OpenVoice).
Build `audiocpp_gguf`, then run:

```bash
python tests/tone_color_vc/convert_gguf.py \
  --upstream /path/to/OpenVoice \
  --checkpoint-dir /path/to/OpenVoiceV2/converter \
  --output models/Tone-Color-VC-GGUF/tone-color-vc-f32.gguf
```

Use `--type f16` and the corresponding output filename to convert the smaller
package.

The converter folds weight normalization and embeds the configuration and model
spec. The resulting GGUF does not need external checkpoint files.
