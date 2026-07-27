# Model Specs

`model_specs/*.json` is the target source of truth for model metadata, package
layout, downloads, UI hints, CLI options, runtime capabilities, and runtime
dependencies.

The only accepted spec shapes are the current source-layout specs already used
by production models, and the typed schema shown here for new metadata/catalog
work.

## Typed Schema

Top-level fields:

| Field | Meaning |
|---|---|
| `schema_version` | Must be `1`. |
| `family` | Runtime model family id. Must match the filename stem. |
| `display_name` | User-facing model family name. |
| `category` | Typed category such as `asr`, `tts`, `audio_generation`, or `community`. |
| `status` | Typed status: `supported`, `community`, `experimental`, `wip`, or `unsupported`. |
| `tasks` | Typed task tags such as `asr`, `tts`, `clone`, `vc`, or `align`. |
| `modes` | Supported run modes: `offline` and/or `streaming`. |
| `languages` | Family-level language scope, such as `en`, `zh`, `ja`, `multilingual`, or `language_agnostic`. |
| `runtime` | Runtime tags such as `gguf` or `stream`. |
| `capabilities` | Stable task-keyed capability tags. |
| `options` | Typed request/session/load options. |
| `package_defaults` | Optional shared package metadata, such as a common download source. |
| `packages` | Installable model packages and download metadata. |
| `layouts` | Typed resource/tensor layouts for future runtime package loading. |
| `dependencies` | Runtime peer models or bundled model assets needed for optional features. |
| `ui` | UI/catalog hints. |
| `sources` | Temporary runtime bridge for current package-spec loading. |

Shared request options must use canonical names such as `seed`, `language`,
`voice_ref`, `text_chunk_mode`, `text_chunk_size`, `max_tokens`,
`temperature`, `top_p`, `top_k`, and `return_timestamps`.

Model-specific request options can be local names in the model spec. Use a
`<family>.<name>` request key only when the runtime already exposes that exact
public option. Do not add one-off model options to framework option contracts.
`load` and `session` options are local names in the spec; the framework derives
their public keys as `<family>.<name>`.

`options` is split by runtime scope. Request options are per request, session
options are fixed when creating a long-lived model session, and load options are
used before the model is loaded. Each row has a stable name, typed value,
explicit `required` flag, and description. Optional rows should include
`default` when production behavior has a stable literal default. Numeric rows
should include `min` and/or `max` when the runtime enforces or documents a
range.
Shared option names carry framework-level contracts: for example `top_k` is an
integer top-k control, `top_p` is a float nucleus-sampling control, and `route`
must be an enum. Repeated enum domains should use a preset instead of copying
the same values into every model.

```json
{
  "options": {
    "request": [
      {
        "name": "text_chunk_mode",
        "type": "enum",
        "preset": "text_chunk_mode_full",
        "required": false,
        "default": "word_budget",
        "description": "Framework text chunking mode."
      }
    ],
    "session": [
      {
        "name": "perf_mode",
        "type": "enum",
        "preset": "perf_mode_flash_attention",
        "required": false,
        "default": "off",
        "description": "Q8_0-only attention performance mode. Public key: qwen3_tts.perf_mode."
      }
    ],
    "load": []
  }
}
```

Current enum presets:

| Preset | Values |
|---|---|
| `weight_type_full` | `native`, `f32`, `f16`, `bf16`, `q8_0` |
| `weight_type_conv` | `native`, `f32`, `f16` |
| `weight_type_codec_q8` | `native`, `f32`, `f16`, `q8_0` |
| `text_chunk_mode_full` | `word_budget`, `tag_aware`, `japanese`, `endline` |
| `perf_mode_flash_attention` | `off`, `flash_attention` |
| `best_of_n_language` | `auto`, `en`, `ja` |

Use structural list types when the option accepts a comma-separated value list:
`string_list`, `float_list`, `path_list`, or `audio_path_list`. Do not hide
structured values behind plain `string`.

`tasks` are the single typed operation vocabulary for the family. Keep model
implementation compatibility, such as serving voice cloning through an existing
TTS session internally, out of the spec.

`capabilities` is keyed by task, and omitted tasks mean no extra advertised
capability beyond the task itself. Keep capabilities typed and concrete:

```json
{
  "languages": ["zh", "en"],
  "capabilities": {
    "clone": ["speaker_reference"],
    "design": ["voice_design"]
  }
}
```

Packages are install targets, not runtime layouts. Each package owns its display
name, precision, target directory, and exact remote files. If several packages
come from the same repo, put the shared source in `package_defaults.download`
and keep package-level `download` only for overrides.

```json
{
  "package_defaults": {
    "download": {
      "kind": "huggingface_snapshot",
      "repo": "audio-cpp/audio.cpp-gguf",
      "revision": "main",
      "gated": false
    }
  },
  "packages": [
    {
      "id": "qwen3_asr_1_7b_q8_0",
      "display_name": "Qwen3-ASR 1.7B Q8_0 GGUF",
      "default": true,
      "format": "gguf",
      "precision": "q8_0",
      "target_directory": "Qwen3-ASR-1.7B-GGUF",
      "files": ["Qwen3-ASR-1.7B-GGUF/qwen3-asr-1.7b-q8_0.gguf"],
      "strip_prefix": "Qwen3-ASR-1.7B-GGUF"
    }
  ]
}
```

Dependencies describe extra model-level resources required by runtime features.
Use `kind: "model"` for another model family, and `kind: "bundled_model"` for an
in-repo bundled model asset. Do not use dependencies for sidecars or tensor
files that are already part of `sources`. The dependency `scope` says where the
dependency path is consumed (`load`, `session`, or `request`), and its public
runtime option key is derived as `<family>.<option>`.

Required dependencies are unconditional. Optional dependencies must declare
typed `required_when` rows. Each row is a condition over a public option key.
Common request keys such as `return_timestamps` stay unprefixed; model-specific
keys stay namespaced. The dependency is needed when any row matches.

```json
{
  "dependencies": [
    {
      "kind": "model",
      "family": "qwen3_forced_aligner",
      "scope": "session",
      "option": "forced_aligner_path",
      "required": false,
      "required_when": [
        {
          "scope": "request",
          "option_key": "return_timestamps",
          "equals": true
        }
      ]
    },
    {
      "kind": "bundled_model",
      "family": "silero_vad",
      "path": "assets/framework/models/silero_vad",
      "scope": "session",
      "option": "vad_path",
      "required": false,
      "required_when": [
        {
          "scope": "request",
          "option_key": "audio_chunk_mode",
          "equals": "vad"
        }
      ]
    }
  ]
}
```

Supported download kinds are `huggingface_snapshot`, `local_snapshot`,
`converter`, and `unsupported`. `tools/model_manager_v2.py` intentionally
installs only simple `huggingface_snapshot` packages; use the legacy manager for
composite or converter-driven installs.

The C++ `framework/model_spec` subsystem is the authoritative schema gate.
`audiocpp_cli`, `audiocpp_server`, and GGUF loading fail when a typed schema field
is invalid.

Run the toy C++ demo through the production subsystem:

```bash
cmake --build build/debug --target model_spec_demo --parallel $(nproc)
build/debug/bin/model_spec_demo \
  examples/model_spec_demo/specs/toy_qwen3_asr.json \
  examples/model_spec_demo/toy_package
```

Preview package download plans from the same validated spec:

```bash
cmake --build build/debug --target model_spec_download_demo --parallel $(nproc)
build/debug/bin/model_spec_download_demo \
  examples/model_spec_demo/specs/toy_qwen3_asr.json
```

The toy browser UI reads `examples/model_spec_demo/specs/toy_qwen3_asr.json`
directly, so there is no duplicated demo catalog.
