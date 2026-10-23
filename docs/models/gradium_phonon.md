# Gradium Phonon

`gradium_phonon` runs Gradium's Phonon text-to-speech models. Phonon is a small
flow-matching language model with a Mimi decoder, a descendant of Kyutai's PocketTTS,
so this family reuses the `pocket_tts` graphs and session. Output is mono 24 kHz.

The packaged model is Phonon 31m English
([gradium/phonon-31m-en](https://huggingface.co/gradium/phonon-31m-en), Apache-2.0), in
two packages:

| Package | Source | Weights |
|---|---|---|
| `gradium_phonon_31m_en_q8_0` (default) | [gradium/phonon-31m-en-audiocpp](https://huggingface.co/gradium/phonon-31m-en-audiocpp) | GGUF: flow transformer linears Q8_0, the rest as published |
| `gradium_phonon_31m_en_safetensors` | [gradium/phonon-31m-en](https://huggingface.co/gradium/phonon-31m-en), as published | Safetensors: BF16 flow model, F32 Mimi |

The 40m checkpoints (`gradium/phonon-40m-*`, CC-BY-NC-4.0,
gated) use the same code with nine layers and load the same way, but are not packaged.

## Usage

```bash
python3 tools/model_manager_v2.py install gradium_phonon_31m_en_q8_0 --models-root models

audiocpp_cli --task tts --family gradium_phonon \
  --model models/phonon-31m-en --backend metal \
  --text "Hello from Phonon." --out out.wav
```

The package's `default-voice.safetensors` is the `default` voice. `--mode streaming`
emits one audio event per generated frame. The safetensors package installs to
`models/phonon-31m-en-safetensors` and runs the same way.

## Voices

Phonon has no voice cloning from audio: the released checkpoints carry no Mimi
encoder. Voices are `speaker_wavs` latent files (one tensor `[1, 512, 125]`), which the
checkpoint projects into its voice prompt. Every Phonon checkpoint reads the same files.

More voices are in [gradium/phonon-voices](https://huggingface.co/gradium/phonon-voices),
listed with their names and languages in `voices.jsonl`. Pass one with
`--request-option voice_embedding_file=/path/to/voice.safetensors`, or copy it into the
model folder as `embeddings/<name>.safetensors`: `--voice-id <name>` and the server's
`"voice": "<name>"` then select it, and `GET /v1/audio/voices` lists it.

## Text

The text goes to the SentencePiece tokenizer after the PocketTTS clean-up: whitespace
collapsed, first letter capitalized, a final period added. There is no number or symbol
normalization, so write numbers, currencies and abbreviations out in words for the best
results. Letters the tokenizer has no piece for, such as `û` and `ï`, reach the model as
byte tokens.

## Controls

Request options (`--request-option name=value`). An option not listed here is an error:

| Option | Default | Meaning |
|---|---|---|
| `temperature` | `0.3` | Flow noise temperature. The reference runtime samples at 0.3 and does not read the config's `temp`. |
| `seed` | random | Noise seed. |
| `eos_threshold` | `4` | EOS logit threshold, from the checkpoint's `config.json`. |
| `frames_after_eos` | `-1` | Frames kept after EOS, counting the EOS frame. `-1` keeps 4 for texts of up to four words and 2 otherwise, as the reference does. |
| `max_frames` | `0` | Frame cap; zero estimates it from the text length. |
| `text_chunk_tokens` | `50` | Tokens per text chunk. |
| `text_chunk_size` | unset | Character chunk size instead of sentence/token budgeting. |
| `voice_embedding_file` | unset | A voice file outside the model folder. |
| `export_voice_state_file` | unset | Export the prepared voice state. |

Session options are those of PocketTTS under the `gradium_phonon.` prefix, for example
`--session-option gradium_phonon.weight_type=f32`, minus the Mimi encoder ones.

## Conversion

The Q8_0 package is built from the safetensors checkpoint with the second command
below; the first keeps the published types. `audiocpp_gguf` embeds `config.json` and
`tokenizer.model` in the GGUF; put `default-voice.safetensors` (or
`embeddings/default.safetensors`) beside it:

```bash
audiocpp_gguf --input models/phonon-31m-en-safetensors/model.safetensors \
  --root models/phonon-31m-en-safetensors \
  --family gradium_phonon --output phonon-31m-en-bf16.gguf --type orig \
  --exclude-prefix flow_lm.flow.w_s_t --overwrite

audiocpp_gguf --input models/phonon-31m-en-safetensors/model.safetensors \
  --root models/phonon-31m-en-safetensors \
  --family gradium_phonon --output phonon-31m-en-q8_0.gguf --type q8_0 \
  --exclude-prefix flow_lm.flow.w_s_t \
  --keep-type 'mimi.*=f32' --keep-type 'flow_lm.flow_net.*=bf16' \
  --keep-type 'flow_lm.condition_provider.*=bf16' \
  --keep-type 'flow_lm.input_linear.*=bf16' --keep-type 'flow_lm.out_eos.*=bf16' \
  --overwrite
```

`orig` keeps the checkpoint's types: BF16 for the flow model, F32 for Mimi. The Q8_0
recipe quantizes only the flow transformer's linear layers, as Gradium's own runtime
does; Mimi stays F32. `flow_lm.flow.w_s_t` is used only in training.

The loader reads the model sizes from `config.json`, uses the plain RMS time embedding
when it says `"time_rms_norm": "rms"`, and folds the summed conditioners
(`num_speakers` = 1, `duration_delta` = 0) into a bias on `input_linear`, which is what
the reference adds to every generated frame.

## Validation

Checked against Gradium's Rust runtime with its reference dumps, feeding the same
noise through `noise_file`: with F32 weights on the CPU, the token ids, the first EOS
frame and the frame count match exactly, and the per-frame latents agree to about 1e-5
relative. BF16 and Q8_0 weights and the Metal backend drift as usual for a lower
precision, with the same Whisper transcripts.
