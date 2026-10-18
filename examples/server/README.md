# Ready-to-edit server configs

Pick **one** config below. Each starts one server on `http://127.0.0.1:8080`.
The files are normal JSON. `_comment` fields explain what to edit; the server
ignores them. Do not add `//` comments or put `_comment` inside an option map.

| Config | What it demonstrates |
|---|---|
| [qwen3-tts.json](qwen3-tts.json) | Base voice cloning, CustomVoice built-in speakers, VoiceDesign instructions; separate GGUFs |
| [breeze-tts.json](breeze-tts.json) | One GGUF for instruction-only speech or voice cloning |
| [kokoro.json](kokoro.json) | One GGUF with built-in voices; no cloning reference |
| [pocket-tts.json](pocket-tts.json) | A language-specific GGUF, accompanying voice embeddings, and optional voice cloning |
| [yue2.json](yue2.json) | A model directory with separate main and VAE GGUFs plus sidecars |
| [minimax-music3.json](minimax-music3.json) | A model directory with five component GGUFs, configs, and tokenizer |
| [parakeet-asr.json](parakeet-asr.json) | Audio transcription and word timestamps |

## 1. Put your files somewhere permanent

Download the package described in the model's documentation. Keep its files
and folders together. These configs do not download weights for you.

Copy your chosen config to a writable location, for example `server.json`.
Replace **every** `/PATH/TO/...` value with a real **absolute path** on your
computer. Do not copy a path from somebody else's machine. On Windows, use
forward slashes, for example `C:/models/kokoro-82m-q8_0.gguf`.

You can install the packages with the repository's model manager. Run only the
commands for your chosen model, from the repository root:

```bash
# Qwen3: choose the variant(s) you need.
python3 tools/model_manager_v2.py install qwen3_tts_1_7b_base_q8_0 --models-root models
python3 tools/model_manager_v2.py install qwen3_tts_1_7b_customvoice_q8_0 --models-root models
python3 tools/model_manager_v2.py install qwen3_tts_1_7b_voicedesign_q8_0 --models-root models
python3 tools/model_manager_v2.py install breeze_tts_2_q8_0 --models-root models
python3 tools/model_manager_v2.py install kokoro_82m_q8_0 --models-root models
python3 tools/model_manager_v2.py install pocket_tts_english_q8_0 --models-root models
# YuE2 needs BOTH packages in the same model root.
python3 tools/model_manager_v2.py install yue2_main_q8_0 --models-root models
python3 tools/model_manager_v2.py install yue2_vae_f16 --models-root models
python3 tools/model_manager_v2.py install minimax_music3_q4_0 --models-root models
python3 tools/model_manager_v2.py install parakeet_tdt_q8_0 --models-root models
```

A path ending in `.gguf` names a file. A directory path names the folder
containing the complete package; do not replace it with just one component file.
Qwen3's three entries require three different checkpoints. Delete entries for
variants you did not download.

These examples select CUDA. Set `backend` to `cpu` for CPU execution, or use
another backend supported by your build/model. CPU is especially practical for
Kokoro, PocketTTS, and the ASR example. The large music models need substantially
more resources. `max_loaded_models: 1` unloads an idle model before loading another;
it does not reduce the memory required by one model.

### A voice library is a folder of your reference recordings

For Qwen3 Base or cloning with Breeze/Pocket, create this folder yourself:

```text
voice-library/
  narrator.wav
  prompt_text
```

Put a short, clean recording of **one person speaking** into `narrator.wav`.
Create a plain-text file named exactly `prompt_text` (no `.txt` extension).
Its contents should be:

```text
narrator|The exact words spoken in narrator.wav go here.
```

Replace the sentence after `|` with what the recording actually says. The word
before `|` must match the WAV's filename without `.wav`. For another voice, add
another WAV and another line. Set the config's `voice_dir` to this folder.

Send `"voice": "narrator"` to select the recording. The server reads its WAV and,
when supported by the model, its transcript. This is different from a **built-in
voice**, which is already supplied with the model. Configured `voice_presets`
take priority over same-named files in the library, so use distinct names.
PocketTTS uses reference audio without a reference transcript.

You can also send `"voice_ref": "/absolute/path/reference.wav"` and
`"reference_text": "The words in that recording."` directly. These paths refer
to the **server's computer**, not a remote client's computer. The speech endpoint
also accepts a base64 reference object; see [server API documentation](../../app/server/README.md).

## 2. Start the server

From the repository root, using your existing build:

```bash
build/debug/bin/audiocpp_server --config /absolute/path/server.json --log --log-file server.log
```

Open `http://127.0.0.1:8080` for the UI, or use the requests below. The configs
use lazy loading, so the first generation request loads the weights and takes
longer. `/health` confirms the server is running; it does **not** prove that a
model has loaded. `/v1/models` lists the configured IDs.

```bash
curl --fail-with-body http://127.0.0.1:8080/health
curl --fail-with-body http://127.0.0.1:8080/v1/models
```

The examples below use a POSIX shell. On Windows use `curl.exe` with a JSON
request file and `--data-binary @request.json` to avoid shell quoting problems.
If a command fails, inspect the response/error and `server.log` before trying to
play the output file.

## 3. Make a request

### Qwen3-TTS

Base: select your reference recording from the voice library.

```bash
curl --fail-with-body http://127.0.0.1:8080/v1/audio/speech -H 'Content-Type: application/json' \
  -d '{"model":"qwen3-clone","input":"Hello from my reference voice.","voice":"narrator"}' -o qwen-clone.wav
```

CustomVoice: the config defaults to Vivian; `ryan` selects the configured Ryan preset.
VoiceDesign: `instructions` describes the new voice.

```bash
curl --fail-with-body http://127.0.0.1:8080/v1/audio/speech -H 'Content-Type: application/json' \
  -d '{"model":"qwen3-custom","input":"Hello from Ryan.","voice":"ryan"}' -o qwen-custom.wav
curl --fail-with-body http://127.0.0.1:8080/v1/audio/speech -H 'Content-Type: application/json' \
  -d '{"model":"qwen3-design","input":"Welcome to our story.","instructions":"A calm adult narrator with a warm voice."}' -o qwen-design.wav
```

### BreezeTTS

Omit `voice` for instruction-only generation. Add `voice` for cloning.

```bash
curl --fail-with-body http://127.0.0.1:8080/v1/audio/speech -H 'Content-Type: application/json' \
  -d '{"model":"breeze","input":"Welcome to this demonstration.","options":{"instruct":"A warm female narrator, speaking clearly."}}' -o breeze.wav
curl --fail-with-body http://127.0.0.1:8080/v1/audio/speech -H 'Content-Type: application/json' \
  -d '{"model":"breeze","input":"This uses my reference recording.","voice":"narrator"}' -o breeze-clone.wav
```

### Kokoro

`heart` and `michael` are configured aliases for built-in voices. No WAV reference
is needed. Use a current published GGUF whose embedded spec declares `speed`.

```bash
curl --fail-with-body http://127.0.0.1:8080/v1/audio/speech -H 'Content-Type: application/json' \
  -d '{"model":"kokoro","input":"Hello from Kokoro.","voice":"heart","speed":1.0}' -o kokoro.wav
```

### PocketTTS

Keep `embeddings/alba.safetensors` beside the English GGUF. The config defaults
to Alba. To clone your own voice, select the library name instead.

```bash
curl --fail-with-body http://127.0.0.1:8080/v1/audio/speech -H 'Content-Type: application/json' \
  -d '{"model":"pocket","input":"Hello from Alba."}' -o pocket.wav
curl --fail-with-body http://127.0.0.1:8080/v1/audio/speech -H 'Content-Type: application/json' \
  -d '{"model":"pocket","input":"Hello in my reference voice.","voice":"narrator"}' -o pocket-clone.wav
```

### YuE2: main model plus VAE

Point `path` at the folder containing:

```text
Yue2-3B-GGUF/
  yue2-3b-q8_0.gguf
  yue2-vae-f16.gguf
  sidecars/
    yue2-model-config.json
    yue2-generation-config.json
    yue2-qwen.tiktoken
    yue2-vae-config.json
```

The `yue2.model_gguf` and `yue2.vae_gguf` session options choose filenames inside
this folder. Change these filenames if you downloaded another supported precision.
Do not pass absolute paths for these two options. Lyrics and style are different
inputs: put lyrics in `request.text` and style in `request.options.style`.

```bash
curl --fail-with-body http://127.0.0.1:8080/v1/tasks/run -H 'Content-Type: application/json' \
  -d '{"model":"yue2","request":{"text":"[Verse]\nMorning sunlight fills the sky.\n[Chorus]\nSing together, you and I.","options":{"style":"English, acoustic pop, warm vocals","cot":"off"}}}' -o yue2-result.json
```

This returns JSON with base64 WAV audio; see the decoding command below.
Optional AR/NAR LoRAs belong in `session_options`, using `yue2.ar_lora` or
`yue2.nar_lora` plus the corresponding `_scale`. They are not needed for this
example. See [YuE2](../../docs/models/yue2.md) before adding adapters.

### MiniMax Music3: five components

Point `path` at the package folder. Keep its layout intact:

```text
MiniMax-Music3-GGUF/
  config.json
  config/
    language_model.json
    rvq_depth_decoder.json
    condition_encoder.json
    transformer.json
    vocoder.json
  tokenizer/
    tokenizer.json
    tokenizer_config.json
  language_model_q4_0.gguf
  rvq_depth_decoder_q8_0.gguf
  condition_encoder.gguf
  transformer_q4_0.gguf
  vocoder.gguf
```

Here `request.text` is the musical description, and `request.options.lyrics`
contains the required lyrics. `duration_sec` is a generation budget, not a
promise of an exact output duration. Keep `mem_saver` enabled for this example.

```bash
curl --fail-with-body http://127.0.0.1:8080/v1/tasks/run -H 'Content-Type: application/json' \
  -d '{"model":"music3","request":{"text":"Warm acoustic pop with clear vocals and gentle drums.","options":{"lyrics":"[verse] Morning sunlight fills the sky. [chorus] Sing together, you and I.","duration_sec":20}}}' -o music3-result.json
```

Decode either music model's JSON output using Python (change the two filenames):

```bash
python -c 'import base64,json,pathlib; r=json.loads(pathlib.Path("music3-result.json").read_text()); pathlib.Path("music3.wav").write_bytes(base64.b64decode(r["audio"]))'
```

### Parakeet ASR: upload audio from your client

Replace `/PATH/TO/input.wav` with a WAV on the computer running `curl`.
The plain endpoint returns text; the details endpoint also preserves word
positions and the sample rate needed to interpret them.

```bash
curl --fail-with-body http://127.0.0.1:8080/v1/audio/transcriptions \
  -F model=parakeet -F file=@/PATH/TO/input.wav
curl --fail-with-body http://127.0.0.1:8080/v1/audio/transcriptions/details \
  -F model=parakeet -F file=@/PATH/TO/input.wav
```

## Streaming and option placement

The configs deliberately start in offline mode. For speech streaming, change
that model entry to `"mode": "streaming"`, restart the server, and add
`"stream_format": "sse"` to its speech request. SSE carries base64 **PCM chunks**,
not a single WAV file; use a streaming client instead of saving SSE as `.wav`.
YuE2 and MiniMax Music3 support offline generation only.

Check the package's actual capabilities before enabling streaming:

```bash
build/debug/bin/audiocpp_cli --family breeze_tts --model /PATH/TO/breeze-tts-2-q8_0.gguf --help
```

Older GGUFs can embed older specs, so help and server model discovery may omit
new modes/options even with a new binary. Update the package. If you deliberately
use an older GGUF with a current source checkout, you can explicitly add
`"model_spec_override": "/PATH/TO/audio.cpp/model_specs/breeze_tts.json"` inside
that model entry (use the matching family file). This changes which contract the
loader uses; it does not convert weights. The equivalent CLI flag is
`--model-spec-override /PATH/TO/audio.cpp/model_specs/breeze_tts.json`.
The offline examples above do not require this override.

- `load_options`: package selection needed when loading weights (Pocket language).
- `session_options`: component selection and session setup; restart/reload after changes.
- `default_request_options`: generation defaults. Request `options` override them.
- `voice_presets`: named built-in voices or fixed reference recordings.
- `voice_dir`: your reusable reference WAV library, shared by configured models.

Do not copy CLI flags literally into JSON. For example, `--session-option
 yue2.model_gguf=...` becomes `"session_options": {"yue2.model_gguf": "..."}`.
See the [server documentation](../../app/server/README.md) for the complete API.
