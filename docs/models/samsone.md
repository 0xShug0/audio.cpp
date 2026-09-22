# SAMSONE

SAMSONE is a compact English audio-language model for captioning, question answering, and sound, music, and speech understanding. The native port supports the upstream 99M, 134M, and 356M variants with audio clips up to 30 seconds.

## Usage

```bash
build/debug/bin/audiocpp_cli \
  --task asr \
  --family samsone \
  --model /path/to/samsone-99m-bf16.gguf \
  --backend cuda \
  --audio input.wav \
  --request-option "instruct=what can be heard in this audio?" \
  --log \
  --text-out response.txt
```

## Request Options

Use these with `--request-option key=value`.

| Option | Values | Default | Description |
| --- | --- | --- | --- |
| `instruct` | English text | `describe the audio` | Question or instruction about the audio. |
| `max_tokens` | `1` to `512` | `100` | Maximum response tokens. |

The upstream repository does not currently publish a license file. Review the upstream terms before redistributing converted weights.

## Upstream

- [SamsungLabs/samsone](https://github.com/SamsungLabs/samsone), release `v1.0.0`
