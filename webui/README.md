# audio.cpp WebUI Launcher Guide

> **语言 / Language:** **English** · [中文](README.zh.md)

The `webui/` directory holds the Python dependencies, launch scripts, and model-download wrappers needed to run the WebUI.
The launch scripts can be **double-clicked** or invoked from a command line / PowerShell.

| Script | Purpose | Typical command |
|---|---|---|
| `webui/run_webui.bat` | Gradio web interface (starts the server on demand) | `webui\run_webui.bat` |
| `webui/run_webui.sh` | Linux / macOS WebUI launcher | `./webui/run_webui.sh` |
| `webui/_env.bat` | WebUI environment detection (**do not run directly**) | `call`ed by `run_webui.bat` |

## Linux / macOS

On Linux / macOS, use `webui/run_webui.sh`:

```bash
python3 -m venv venv && ./venv/bin/pip install -r webui/requirements.txt
./webui/run_webui.sh                # UI -> http://127.0.0.1:7860
```

- The Python interpreter is probed in order: `$AUDIOCPP_PYTHON`, `venv/bin/python`, `.venv/bin/python`,
  `webui/venv/bin/python`, `webui/.venv/bin/python`, `python3`, `python`.
- The backend (cuda/cpu) is auto-detected: Windows checks `nvcuda.dll`, other platforms check `nvidia-smi`,
  then confirms that the matching server build exists; override with `AUDIOCPP_BACKEND=gpu|cpu`.
- The binaries can come from a portable bundle's `gpu/` or `cpu/` directories, or from a source build's
  `build/<os>-<backend>-<type>/bin` (such as `build/linux-cuda-release/bin`).
  A plain `cmake -B build` producing `build/bin` is also recognized — when the directory name carries no
  backend information, the backend is inferred from `GGML_CUDA` in `CMakeCache.txt`.
- The Windows-only packages in `webui/requirements.txt` (pywin32, plus the pythonnet/pywebview used by
  SpeakType) are marked with `sys_platform`, so they install cleanly on Linux too.

## UI language / 界面语言

The interface supports **English / 中文 / 中文繁體**, defaulting to English, and the language dropdown in the
top-right corner can switch at any time.
The choice is saved to `webui/configs/ui_language.json` and reused on the next launch.

The environment variable `AUDIOCPP_LANG` (`en` | `zh` | `zh-Hant`, and forms like `zh_TW`, `zh-CN` are also
accepted) only sets the default language **when no choice has been saved yet** — otherwise it would silently
override the user's explicit in-UI selection.
To hand control back to the environment variable, just delete `webui/configs/ui_language.json`.

> Traditional Chinese is a **glyph-level** conversion from Simplified performed by OpenCC
> (`opencc-python-reimplemented`); it does not substitute Taiwan/Hong Kong vocabulary
> (e.g. `软件` → `軟件`, not `軟體`).

> Long-text synthesis no longer needs a separate script (the old `run_tts_long.bat` has been removed): the
> WebUI's TTS tab automatically splits long text into chunks (600 chars/chunk for VibeVoice, 1000 chars/chunk
> for other models), synthesizes each chunk, and concatenates them into a single wav.
> The command-line equivalent is `audiocpp_cli`'s `--batch-text-file <txt> --batch-merge-audio concat`.
>
> Conversely, **VibeVoice produces garbled output for very short text (roughly <40 Chinese characters)** — a
> model characteristic, independent of voice / parameters / seed. The WebUI intercepts this directly and prompts
> you to lengthen the text or switch models; an over-short trailing chunk left by splitting is also merged back
> into the previous chunk automatically. For short-sentence tests, use `qwen3-tts` / `voxcpm2` / `pocket-tts`.

---

## General conventions

- **Models are specified by catalog id.** The WebUI uses the **id** from `webui/configs/models_catalog.json`
  to select a model, and automatically resolves its `family` / `task` / absolute path, so you never have to
  write those by hand.
  Currently installed ids: `qwen3-tts`, `qwen3-asr`, `vibevoice`, `omnivoice`, `pocket-tts`.
  An uninstalled id shows "not installed"; you can download it in the WebUI, or install it with
  `python tools/model_manager.py install <download_id>` (see `models_catalog.json`).
- **Automatic backend selection:** if CUDA (an NVIDIA driver) is detected, GPU is used; otherwise it falls back
  to CPU. To force a backend, set `AUDIOCPP_BACKEND=gpu` (= cuda) or `AUDIOCPP_BACKEND=cpu`.
  The CLI, server, and WebUI all follow this detection (a machine without an NVIDIA GPU automatically drops to
  the CPU build, which is slower and impractical for some large models).
- **Path base:** relative paths inside the scripts (such as `voice\demo_01_man.wav`, `output\xxx.wav`) are all
  relative to the `webui\` directory.
- **Executable source:** the bundle `..\audiocpp-portable` (containing `cpu\ gpu\ models\`) is located
  automatically, and the scripts self-locate even when copied into the bundle.

---

## `webui/_env.bat` (internal shared helper, do not run directly)

`call`ed by the other scripts; it sets up the common variables once (deliberately not using `setlocal` so the
variables carry back to the caller):

- `BUNDLE` — bundle root directory (contains `cpu\ gpu\ models\`)
- `HAS_CUDA` — whether CUDA was detected (`nvcuda.dll` or `nvidia-smi`)
- `BACKEND` / `CLI_EXE` — the selected backend (`cuda`/`cpu`) and its matching `audiocpp_cli.exe`
- `SERVER_EXE` — `audiocpp_server.exe` from `gpu\` or `cpu\` per `BACKEND` (falls back to the gpu build when the cpu one is missing)
- `PY` — the Python with dependencies (used by `webui/run_webui.bat`)

To change the detection logic, you only need to edit this one file.

---

## `webui/run_webui.bat` — graphical interface

Starts the Gradio web interface (`webui.py`); open **http://127.0.0.1:7860** in a browser.

- **On-demand loading:** you do not need to start `audiocpp_server` first — when you select a model and click
  "load" / "generate" in the UI, the WebUI automatically starts/switches the underlying `audiocpp_server`
  (one model in VRAM at a time; switching models restarts it).
- The UI lets you upload a reference voice, download uninstalled models, enter an HF token / proxy, and so on.
- **"⬇️ Download" asks before it downloads.** It does not start anything: it reports what the download will cost —
  size (read live from the Hugging Face repo listing, so it never goes stale), free space on `models/`, the
  estimated VRAM, and any warnings — then shows **✅ Confirm download** / **✖️ Cancel**. Nothing is fetched until
  you confirm, and changing the selected model withdraws the prompt. "📊 Progress" shows the same figures for a
  model that has not been downloaded yet, if you want to check without arming anything.
  The VRAM estimate is reported on the CPU backend too (labelled as such — the CPU backend runs in system RAM);
  only the low-VRAM *warning* stays CPU-quiet, since there is no VRAM to fall short of. A download the `models/`
  volume cannot hold is refused outright — no Confirm button is offered — and a fit that leaves under 2 GB free is
  flagged but still allowed.
  Sizes are unavailable for converter installs and for gated repos without a token; those download as before.
- **While it runs**, progress is shown against the package total ("Downloaded 5.00 GB of 17.00 GB (29%)"), and the
  low-VRAM and low-disk warnings are repeated on every refresh rather than scrolling away after the first one.
  Disk is judged against the bytes still to fetch, so a healthy download does not drift into a false alarm as free
  space drops. The VRAM warning stays on screen after the download finishes — it decides whether the model can be
  run at all, not whether it can be fetched.
- Backend is auto-detected (as above: GPU if CUDA is present, otherwise CPU); `AUDIOCPP_BACKEND=gpu|cpu` forces it.
  In CPU mode, the ggml thread count is set automatically from the **physical** core count — SMT/Hyper-Threading
  siblings are not counted, and one core is left free above 4 — so a long run leaves the rest of the machine
  usable. Filling every logical CPU is faster (~1.4x in a short CPU TTS run on an 8-core/16-thread 5800H); set
  `AUDIOCPP_THREADS=N` if you want that. The VRAM warning is not shown in CPU mode.

> The web interface (7860) is for humans; to use it as an **API for other programs**, start `audiocpp_server`
> directly, or once the WebUI is up, hit the port 8080 it manages directly.

---

## WebUI advanced parameters (widgets auto-generated per model)

The widgets under the TTS tab's "Synthesis settings → Advanced parameters" are driven by
**`configs/model_params.json`**: once you select a model, the WebUI **dynamically generates the matching
sliders / number boxes / toggles / text boxes** for its `family` (via `gr.render`), so you don't have to hand-write
JSON. Below the widgets there is also a collapsible "Other parameters (JSON)" fallback box for passing keys not
listed as widgets. General rules:

- **Only widget values you have changed are sent with the request** (untouched ones use the model's own defaults);
  the `options` merge order is: family defaults → generated widgets → JSON box (JSON overrides widgets).
- `seed` and `max_tokens` already have dedicated input boxes (synthesis settings) and are not duplicated here.
- The reference voice comes from "upload/record" or a "built-in reference voice"; the original transcript of the
  reference audio goes in the "reference text" box (equivalent to `reference_text`).
- Invalid values are usually ignored, or the server reports an error — errors are shown **below the output audio**
  (no popup card).
- **Chatterbox** cloning parameters are fixed at model-load time: after changing them you must click
  '📥 Load model' again for them to take effect (otherwise it reports "session config is fixed").

### Custom widgets (edit `configs/model_params.json`)

Grouped by `family`, one widget spec per entry:

```json
{"name": "guidance_scale", "type": "slider", "label": "guidance_scale",
 "default": 1.3, "minimum": 0.0, "maximum": 5.0, "step": 0.1, "info": "CFG guidance strength"}
```

- `name`: the key passed through to the request `options`. `type`: `slider` / `number` (`precision:0` for
  integers) / `bool` / `text` / `choice` (with `choices:[...]`).
- `default` should equal the model default (already checked against each `src/models/<family>/*.cpp`).
- After editing, click '🔄 Refresh list' in the UI to reload this file — no restart needed.
- File-path / parity-style rare parameters (such as `*_noise_file`) are not exposed as widgets; pass them through
  the "Other parameters (JSON)" box. Quantization keys (such as `vibevoice.weight_type`) are in the project-root
  `README.md`.

The table below lists the full set of usable keys each model's `session.cpp` **actually reads** (the widgets are a
curated common subset; other keys can still be passed via the JSON box):

| Model (family) | Usable keys (also passable via JSON box) | Example |
|---|---|---|
| **Qwen3-TTS** (qwen3_tts) 0.6B / 1.7B / CustomVoice | `do_sample` `temperature` `top_k` `top_p`; the CustomVoice variant also has `speaker` | `{"do_sample": true, "temperature": 0.8, "top_k": 40, "top_p": 0.9}`<br>CustomVoice built-in voice: `{"speaker": "<CustomVoice voice name>"}` |
| **VibeVoice** (vibevoice) 1.5B long-form/multi-speaker | `num_inference_steps` `guidance_scale` `max_length_times` `do_sample` `temperature` `top_k` `top_p`; multi-speaker `voice_samples` (comma-separated wavs, max 4, **cannot** be combined with a reference voice) | `{"num_inference_steps": 10, "guidance_scale": 1.3, "max_length_times": 2.0}`<br>Multi-speaker: `{"voice_samples": "D:/a.wav,D:/b.wav"}` |
| **VoxCPM2** (voxcpm2) | `num_inference_steps` `guidance_scale` `min_tokens` `retry_badcase` `retry_badcase_max_times` `retry_badcase_ratio_threshold`; reference transcript `prompt_text` | `{"num_inference_steps": 10, "guidance_scale": 2.0, "retry_badcase": true}` |
| **MioTTS** (miotts, needs MioCodec) | `temperature` `top_k` `top_p` `repetition_penalty` `presence_penalty` `frequency_penalty` `do_sample` `best_of_n` `best_of_n_enabled` `best_of_n_language` | `{"temperature": 0.9, "top_p": 0.9, "repetition_penalty": 1.1, "best_of_n": 3}` |
| **Chatterbox** (chatterbox, voice cloning) | `exaggeration` `guidance_scale` `temperature` `repetition_penalty` `min_p` `top_p` `s3gen_cfg_rate` `max_new_tokens` `do_sample` `greedy` `stop_on_eos` | `{"exaggeration": 0.5, "guidance_scale": 0.5, "temperature": 0.8, "repetition_penalty": 1.2}` |
| **OmniVoice** (omnivoice) | `instruct` (style/instruction text); `reference_text` (usually the "reference text" box is enough) | `{"instruct": "Read in a light, upbeat tone"}` |
| **Pocket TTS** (pocket_tts) | No dedicated advanced parameters (just a reference voice + language) | — |

> The key names come from the options each model's `src/models/<family>/session.cpp` actually reads; the same key
> may have a different range/meaning across models. Quantization-related keys (such as `vibevoice.weight_type`,
> `voxcpm2.*_weight_type`) are in the project-root `README.md` quantization section and are not universal defaults.

### Music generation / voice conversion parameters in detail

The hints on the page are condensed; the full explanation is collected here.

**ACE-Step (music generation/editing)**

- The prompt describes style/instruments/mood (English works best); lyrics are optional; a duration of `-1` means auto.
- `task_route` operation type: `text2music` = pure text-to-music (default, needs no source audio); `cover` = re-lyric
  cover (the main Remix route, used with the two cover sliders below); `cover-nofsq` = a cover variant (skips FSQ
  quantization); `remix` = fine flow-edit re-lyric; `complete` / `lego` / `extract` / `repaint` are other editing routes.
  **All except text2music require uploading source audio.**
- After uploading source audio, it is recommended to click '🔍 Analyze source audio' first: it back-infers the source
  song's description/lyrics/BPM/key and auto-fills the advanced parameters (especially recommended before remix/cover
  re-lyric; the first run needs '📥 Load model' first; ~1 minute of audio takes tens of seconds).
  The analysis is reproducible: analyzing the same audio yields the same result each time (VAE mean encoding; with
  seed=-1 the analysis always uses 1234 — pass a specific seed to re-sample the lyric transcription).
- Diffusion parameters: `num_inference_steps` caps at 20 for turbo, defaults to 16 for the remix route when unset and
  8 for other routes; `shift` (timestep warping) defaults to 3.0 to match the original turbo UI — dropping it back to
  1.0 noticeably degrades remix re-lyric articulation.
- The two cover-route sliders:
  - `audio_cover_strength` (Remix strength): what fraction of denoising steps reference the source structure; 1 = close
    to the original, 0 = free improvisation; the original Remix recommends 0.5. Only effective for cover/cover-nofsq.
  - `cover_noise_strength` (melody preservation): starts denoising from the noised point of the source; 0 = don't keep
    the melody, 0.1–0.25 = recommended range (keep the melody while changing lyrics/style), higher hugs the original
    more closely. Only effective for cover.
- remix (flow-edit) parameters:
  - `source_caption` / `source_lyrics`: source-side text conditions (the source song's own style description / original
    lyrics, with `[Verse]` `[Chorus]` tags); an empty caption uses the main prompt; '🔍 Analyze' can auto-fill them.
    **New lyrics go in the main 'Lyrics' box.**
  - `flow_edit_n_min` (edit start): the fraction of leading high-noise steps to skip; 0 = edit from the start; larger
    keeps more of the source but weakens re-lyric.
  - `flow_edit_n_max` (edit end): 1 = paired editing throughout; lowering it to 0.7–0.9 makes the tail denoise only
    toward the new lyrics — **tune this first when lyrics won't come out**.
  - `flow_edit_n_avg`: multiple samples averaged per step (remix defaults to 2, more stable), 1 = fastest.
    Note that the remix default of 16 steps × n_avg 2 ≈ 4× the old default's time (8 steps × 1); lower it manually for speed.
- Score parameters `bpm` / `keyscale` (such as `F major`, `c# minor`) / `timesignature` (such as `4`):
  0/blank = unspecified; auto-filled after '🔍 Analyze'.

**Stable Audio (music/SFX):** the prompt is **English only** and uses no lyrics; the music build generates music, the
sfx build generates sound effects.
Uploading source audio enables continuation/inpainting: `audio_input_kind` selects `init_audio` (with `init_noise_level`
strength) or `inpaint_audio`.

**HeartMuLa (lyrics+tags song generation):** the advanced parameter `tags` is required (comma-separated, e.g.
`pop,bright,drums,female vocals`), and 'Lyrics' holds the sung words. It is a 3B model; the official 120-second long song
measured a peak VRAM of ~25G (docs/memory_saver.md), so an 8G GPU can't run it; mem_saver is on by default, and long songs
can enable `infinite_mode`.

**Chatterbox VC (voice conversion):** the source speech provides content, the target voice reference provides speaker
identity, and the output is 24 kHz mono speech. `s3gen_cfg_rate` controls voice guidance strength, `num_inference_steps`
controls the number of generation steps; defaults are 0.7 and 10 respectively. This entry shares the same model files as
the Chatterbox voice cloning on the TTS page.

**Seed-VC (voice conversion):** source speech + target voice reference (a few seconds to tens of seconds of clean voice).
An empty `route` follows the task default (a vc entry → `v2_vc`, an svc entry → `v1_svc`); `v1_whisper_bigvgan_vc` /
`v1_xlsr_hift_vc` are the older v1 routes; `v1_svc` only pairs with an svc entry. `intelligibility_cfg_rate` /
`similarity_cfg_rate` are v2-only, `inference_cfg_rate` is v1-only.

**Vevo2 (voice conversion):** the three Vevo2 entries install the self-contained Q8_0 GGUF package
(`vevo2_gguf` → `models/Vevo2-GGUF`, ~3.2 GB), which needs no sibling `whisper-medium` download. An earlier
safetensors install in `models/Vevo2` is no longer what these entries point at; it still works from the CLI with
`--model models/Vevo2`, or reinstall it with `python3 tools/model_manager.py install vevo2`.
Vevo2 defaults to `route=style_preserved_vc` (preserves the source speech's speaking style,
changing only the voice). An empty `route` follows the entry's task default (vc → style_preserved_vc, svc →
style_preserved_svc, s2s → editing) and must match the selected entry's task; `style_converted_*` / `editing` need
`style_ref` (a server-local wav path) / `style_ref_text` / `target_text` added in the "Other parameters (JSON)" box.
`use_pitch_shift` (global transposition by the source/target median-pitch difference) follows the route default when
blank: on for style_preserved_* and singing routes, off for style_converted_vc / editing.
Long audio is adaptively chunked and concatenated by 'target voice duration + per-chunk source duration ≤ VRAM budget',
and a reference voice longer than ~10s is auto-truncated (an 8G VRAM limit).

### Input requirements per task page, in detail

- **VibeVoice:** each line of the multi-speaker script is `Speaker N: content` (N starts at 0); plain text is
  auto-wrapped as `Speaker 0: ...`. For different voices per role, use the advanced parameter `voice_samples`
  (comma-separated server-local wavs, ≤4); in that case, **do not** also upload a reference voice.
- **VoxCPM2 / Qwen3-TTS:** upload/select a clean single-speaker reference voice and put that audio's transcript in the
  'reference text' box, or output may be truncated early. Long text is auto-chunked, synthesized, and concatenated;
  VoxCPM2 defaults to q8_0 quantization on an 8G GPU.
- **Chatterbox:** language is limited to english / spanish / french / german / italian / portuguese / korean (no
  Chinese/Japanese/Russian, and no auto-detection); 'blank' = English.
- **Qwen3-ASR:** long audio is auto-split at silences into ≤60-second chunks, transcribed, and concatenated. The
  'context hint' takes names/terms/background (e.g.: a meeting discussing ggml quantization, attendees: Zhang Wei, Li Na)
  to help recognize proper nouns. Conversation mode (limited to 120s) first runs Sortformer speaker separation (≤4 people),
  then transcribes each segment into a conversation transcript with speakers and timestamps; the Sortformer model must be installed.
- **Audio analysis (VAD/separation/alignment):** WAV input is auto-converted to 16 kHz mono before going to the model,
  and result timelines are computed at 16 kHz. Qwen3 forced alignment caps a single audio at ~115 seconds.
- **Source separation:** HTDemucs outputs four tracks — drums/bass/other/vocals (long audio takes a while);
  Mel-Band RoFormer outputs a vocals track + an accompaniment track (mixture − vocals).
- **IndexTTS2** (new in 0.3): Chinese/English voice cloning; a reference voice is **required**. Emotion control is in the
  advanced parameters: `emotion_text` holds an emotion description (setting it auto-enables `use_emotion_text`) +
  `emotion_alpha` adjusts strength; or check `use_emotion_text` to infer it automatically from the read text;
  `emotion_vector` (8 floats) goes through the JSON fallback box.
- **Irodori-TTS** (new in 0.3, Japanese): the 500M generates directly without a reference by default; uploading a
  reference voice auto-switches to cloning (the UI sends `no_ref=false` for you); the 600M VoiceDesign describes the
  voice with a Japanese caption on the 'Voice design' page. The language dropdown only accepts japanese/blank.
- **MOSS-TTS** (new in 0.3): Local v1.5 generates from plain text; for cloning, a 'reference text' is recommended, and it
  outputs 48 kHz stereo; Nano 100M is lightweight — no reference = continuation-style generation (random voice), with a
  reference = cloning.
- **Supertonic 3** (new in 0.3): preset-voice multilingual TTS (English/Japanese/Korean/European languages, **no Chinese**);
  in the advanced parameters, choose `voice` (M1-M5 male / F1-F5 female) and `speaking_rate`; reference-audio cloning is not supported.
- **Model downloads** run in the background with auto-refreshing progress; you can also click "📊 Download progress" to check manually.

### GGUF conversion, inspection, and loading

Every task page's "Model management" card offers the same set of GGUF operations: pick a type (default `q8_0`) then click
"🧊 Convert GGUF", which writes the result as `model.gguf` in the selected model's directory; an existing file is not
overwritten. Clicking "🔎 Inspect GGUF" runs `audiocpp_gguf.exe --inspect` and shows the package metadata on the page.
For models with native GGUF support, a normal "📥 Load model" automatically prefers the GGUF in the model directory —
`model.gguf` if it is there, otherwise the single `*.gguf` in the directory, so a downloaded package keeps its release
name (`vevo2-q8_0.gguf`). A directory holding several GGUFs and no `model.gguf` is ambiguous and neither the page nor the
server picks one. Clicking "🗑️ Delete GGUF" removes conversion output (and any leftover `.tmp` of the same name) and the
next normal load restores the original weights; a GGUF that is itself the downloaded package is left alone — re-download
or remove it from the model list instead.

- The converter looks in order for the dev builds' `build\windows-cuda-release\bin` / `build\windows-cpu-release\bin`, and
  the bundle's `audiocpp-portable\gpu` / `audiocpp-portable\cpu`; you can also point `AUDIOCPP_GGUF` at a custom `audiocpp_gguf.exe`.
- The page only marks a model as "convertible" when it has native GGUF model-spec support and the conversion inputs can be
  unambiguously assembled; the presence of `.safetensors` does not mean the matching C++ backend supports GGUF. A model
  that supports conversion but isn't fully installed shows "convertible, but model not fully installed" up front, to help
  you decide before downloading; Stable Audio currently still uses original weights and is not marked convertible.
- The page automatically handles a supported single `model.safetensors`, sharded indexes, and Qwen3-TTS composite weights.
  Other composite models that need multiple named `--input namespace=...` should still use the command line, to avoid the
  UI guessing the wrong weight namespace.
- Only audio.cpp-native GGUF can be loaded; quantization compatibility varies by model and inference route. Even a
  successful conversion should be checked with a short sample first.

---

## Model id quick reference

The full list is in `configs\models_catalog.json` (each entry has `id` / `family` / `path` / `task` / `download_id`).
Common ones:

| id | family | task | Notes |
|---|---|---|---|
| `qwen3-tts` | qwen3_tts | tts | Qwen3-TTS 0.6B (voice cloning) |
| `qwen3-asr` | qwen3_asr | asr | Qwen3-ASR 0.6B |
| `vibevoice` | vibevoice | tts | VibeVoice 1.5B (long-form/multi-speaker, `Speaker N:` script) |
| `omnivoice` | omnivoice | tts | OmniVoice |
| `pocket-tts` | pocket_tts | tts | Pocket TTS (needs a reference voice) |
| `index-tts2` | index_tts2 | tts | IndexTTS2 (Chinese/English cloning + emotion, needs a reference voice) |
| `irodori-tts` | irodori_tts | tts | Irodori-TTS 500M (Japanese) |
| `irodori-tts-vdesign` | irodori_tts | vdes | Irodori-TTS 600M VoiceDesign (Japanese caption) |
| `moss-tts-local` | moss_tts_local | tts | MOSS-TTS-Local v1.5 (48 kHz stereo) |
| `moss-tts-nano` | moss_tts_nano | tts | MOSS-TTS-Nano 100M (lightweight) |
| `supertonic` | supertonic | tts | Supertonic 3 (preset voices, no Chinese) |

An uninstalled id prompts at runtime; you can click "download" in the WebUI, or run
`python tools\model_manager.py install <download_id> --models-root <bundle>\models`.

---

## Environment variables

| Variable | Purpose | Applies to |
|---|---|---|
| `AUDIOCPP_BACKEND` | `gpu`(=cuda) / `cpu` to force the backend | cli / server / webui |
| `AUDIOCPP_HOST` | server bind address (`0.0.0.0` opens it to the LAN) | server |
| `AUDIOCPP_BUNDLE` | manually specify the bundle root directory | all |
| `AUDIOCPP_SERVER` | make the WebUI connect to an already-running external server | webui |
| `AUDIOCPP_LOAD_TIMEOUT` | seconds the WebUI waits for a model to load (default 300) | webui |

---

## Troubleshooting

- **`.bat` flashes and closes on double-click / command syntax error:** these scripts must use **CRLF** line endings
  (LF makes cmd misparse them); keep CRLF after editing.
- **Port already in use:** the WebUI-managed `audiocpp_server` defaults to 8080. To also run an external server, change
  the port on one of them, or set `AUDIOCPP_SERVER` so the WebUI reuses the external server.
- **`model path does not exist` / not installed:** the model isn't installed. Use the model_manager command above or
  download it in the WebUI.
- **Out of VRAM:** on 8GB, when running two servers at once both models must fit; for 1.7B, running just one is recommended.
- **Voice cloning output is too short (ends at ~0.4s):** the `voice_ref` voice isn't clean or `reference_text` is missing;
  switch to a clean single-speaker reference audio with its matching text.

---

## API vs command-line performance

Same engine, same backend → **inference itself is identical**. The difference is mainly the **amortization of model
loading**:

- Calling `audiocpp_cli` directly **reloads the model into VRAM on every call** (a fixed few-second overhead each time).
- The `audiocpp_server` service **loads once and stays resident**, so each subsequent request only spends "inference + a
  tiny transfer". Local HTTP + a few-MB wav transfer ≈ milliseconds, negligible against multi-second inference (use the
  default binary wav; avoid the base64 of `response_format:"json"`, which adds about +33%).
- The web interface (7860) is one proxy hop further than hitting 8080 directly; other programs hitting 8080 directly skip that hop.

**Conclusion:** going through the API adds almost no per-generation cost — the one-time warmup is amortized by the
server. Except for "generate exactly once" cases, the API approach is usually **faster** than repeatedly calling the CLI.
