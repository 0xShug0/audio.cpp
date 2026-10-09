# Model Specs

`model_specs/*.json` is the target source of truth for model metadata, package
layout, downloads, UI hints, CLI options, runtime capabilities, and runtime
dependencies.

The only accepted spec shapes are the current source-layout specs already used
by production models, and the typed schema shown here for new metadata/catalog
work.

## Migration Trees

`model_specs/*.json` is the authoritative runtime tree. Specs with numeric
`schema_version: 1` or `schema_version: 2` in this directory are
active typed specs and participate in `model_contract()` validation, embedded
GGUF metadata, CLI/server inspection, and package discovery.

`model_specs_v1/*.json` is a migration reference tree only. It records the
expected typed shape for families that have not moved to the v1 runtime contract
yet. Do not treat it as an active runtime source, and do not assume changes there
affect loaders, CLI options, server validation, or package installs until the
family is migrated into `model_specs/` with a typed `schema_version` (`1` or
`2`).

During migration, keep public option keys aligned with the code path that will
actually consume them. If a live loader still accepts an older public key, either
migrate the runtime to the normalized key in the same change or keep the legacy
key out of the active spec until cutover. Avoid creating UI-only aliases.

## Typed Schema

Top-level fields:

| Field | Meaning | Regenerate standalone GGUF if touched? |
|---|---|---|
| `schema_version` | Numeric `1` for the original typed contract; numeric `2` for the integrator contract. | No |
| `family` | Runtime model family id. Must match the filename stem. | Yes, if changing an already published family id |
| `display_name` | User-facing model family name. | No |
| `category` | Typed category such as `asr`, `tts`, `audio_generation`, `audio_tools`, or `community`. | No |
| `status` | Typed status: `supported`, `community`, `experimental`, `wip`, or `unsupported`. | No |
| `tasks` | Typed task tags such as `asr`, `tts`, `clone`, `vc`, `midi`, or `align`. | No |
| `modes` | Supported run modes: `offline` and/or `streaming`. | No |
| `languages` | Family-level language scope, such as `en`, `zh`, `ja`, `multilingual`, or `language_agnostic`. | No |
| `runtime` | Runtime tags such as `gguf` or `stream`. | No |
| `capabilities` | Stable task-keyed capability tags. | No |
| `options` | Typed request/session/load options. | Yes, otherwise new code needs compatibility mapping for old embedded specs |
| `package_defaults` | Optional shared package metadata, such as a common download source. | No |
| `packages` | Installable model packages and download metadata. | No |
| `dependencies` | Runtime peer models or bundled model assets needed for optional features. | No, but loader/session behavior must support the dependency |
| `ui` | UI/catalog hints. | No |
| `sources` | Canonical runtime resource/tensor mappings. | Yes |

## Integrator Contract (Schema 2)

Schema 2 is the integrator contract. Existing schema 1 specs remain valid
without edits. A spec must declare numeric `schema_version: 2` before it uses
any of the fields in this section. A reader that only accepts schema 1 rejects
`2` instead of ignoring the new fields.

The contract is task-scoped because model categories are not routing rules.
Multi-task families can expose speech, conversion, generation, analysis, and
structured outputs from the same package. Integrators must not infer behavior
from the family name, category, option names, or capability tags.

### Operations

`task_operations` maps a task declared in `tasks` to one or more existing server
operations. `preferred_operation` is the normal choice:

| Operation | HTTP surface | Encoding |
|---|---|---|
| `tasks.run` | `POST /v1/tasks/run` | JSON |
| `speech.create` | `POST /v1/audio/speech` | JSON |
| `transcriptions.create` | `POST /v1/audio/transcriptions` | multipart |
| `alignments.create` | `POST /v1/audio/alignments` | multipart |

Method, path, encoding, and response type may be included for readability, but
validation requires them to match the operation. `audiocpp_cli --spec --json`
fills any of those fields the source spec omitted. `speech.create` responds
with `audio/wav` unless the body field `response_format` is `json` or
`b64_json`. `tasks.run`, `transcriptions.create`, and `alignments.create`
respond with `application/json`. Specs cannot create arbitrary routes.

```json
{
  "task_operations": {
    "tts": {
      "preferred_operation": "speech.create",
      "operations": ["speech.create"]
    },
    "vc": {
      "preferred_operation": "tasks.run"
    }
  }
}
```

A task without an entry has an unknown route. It is not unsupported.
An operation may set `"stream": true` only when `modes` includes `streaming`.

### Task-scoped options and bindings

Any request, session, or load option may add `tasks`. Omission preserves schema
1 behavior and means the row applies to every task in the family. An option may
also declare stable `aliases`.

`bindings` maps an operation id to the actual wire location. JSON operations
use `json_pointer`; multipart operations use `multipart_field`.

```json
{
  "name": "temperature",
  "type": "float",
  "required": false,
  "default": 0.7,
  "min": 0,
  "max": 2,
  "description": "Sampling temperature.",
  "tasks": ["tts"],
  "bindings": {
    "speech.create": {"json_pointer": "/temperature"}
  }
}
```

For `tasks.run`, request JSON pointers must start with `/request/`. Root inputs
bind under `/request/<field>`. Spec request options bind under
`/request/options/<public_key>`. A value may bind differently on different
operations. Literal `false` and `0` defaults are present defaults; an absent
default remains unknown.

`speech.create` pointers are limited to the body fields that route already
accepts: `/input`, `/voice`, `/voice_ref`, `/instructions`, `/language`,
`/speed`, `/speaking_rate`, `/seed`, `/temperature`, `/top_k`, `/top_p`,
`/max_tokens`, `/max_steps`, `/repetition_penalty`, `/guidance_scale`,
`/reference_text`, `/num_inference_steps`, and `/options/*`.

Multipart names are limited to the fields already documented for that route.
`transcriptions.create` accepts `file`, `model`, `language`, `prompt`, and
`stream`. `alignments.create` accepts `file`, `model`, `text`, and `language`.

Each binding operation must be declared by at least one task on the row, and
every task on the row must be covered by a binding whose operation that task
declares. An option used by both `speech.create` and `tasks.run` carries one
binding for each. A request binding sets exactly one of `json_pointer` or
`multipart_field`. An output binding sets `slot` instead.
Aliases must not collide with another option's public key in the same scope.
On schema 1, `task_operations`, `inputs`, `outputs`, and option `tasks`,
`bindings`, and `aliases` are errors. Other unknown keys on schema 1 stay
ignored. Omitted schema 2 sections mean the route or role is unknown, not that the
family rejects it.

### Inputs

`inputs` describes semantic roles independently of option names:

```json
{
  "inputs": [
    {
      "id": "instructions",
      "tasks": ["tts"],
      "schema": {
        "type": "enum",
        "values": ["calm", "energetic"]
      },
      "required": false,
      "scope": "request",
      "bindings": {
        "speech.create": {"json_pointer": "/instructions"}
      }
    },
    {
      "id": "audio",
      "tasks": ["vc"],
      "schema": {
        "type": "audio",
        "media_type": "audio/wav",
        "wire": ["path_string"]
      },
      "required": true,
      "scope": "request",
      "bindings": {
        "tasks.run": {"json_pointer": "/request/audio"}
      },
      "alternatives": ["reference_audio"]
    }
  ]
}
```

Input schema types are `string`, `enum`, `audio`, and `artifact`. Enum inputs
use either explicit `values` or an existing option `preset`. `tasks` is always
required. `required` may be omitted when the requirement is not known.
`scope` is `request` or `session`.

`instructions` and `voice` are reserved role ids, but they use the same shape as
family-defined ids such as `lyrics`, `style`, or `abc`. Instructions are
supported only for the tasks listed on an `instructions` role. A voice is
required only when a `voice` role says `required: true`. `ui.default_voice`
and `ui.builtin_voices` remain presentation/catalog metadata and do not create
a runtime requirement.

`alternatives` names other input ids or declared option keys that can satisfy
the role. The caller supplies at most one member of an alternatives set.
Optional `presentation` may contain `label`, `group`, and `advanced`; it is not
needed to serialize a valid request.

An audio input requires `media_type` `audio/wav` and `wire`. Speech `/voice_ref`
uses both `path_string` and `voice_ref_object`. Every other audio field uses
`path_string` only. The resolved spec expands those names to `wire_forms`.
`path_string` is a filesystem path to a WAV. `voice_ref_object` is a path
string or an object with `type` `path` or `base64`; decoded base64 WAV is at
most 5 MiB.

A string input with id `instructions`, or one bound to `/instructions`, requires
`schema.text`. `caller` is `plain`: the caller sends the string unchanged.
`engine_prefix` and `engine_suffix` are the affix the engine adds. They are an
alternative to `engine_boundary`, which has `token`, `prefix`, and `suffix` and
applies only when the string does not already contain `token`. The resolved
spec copies instruction inputs to `instruction_fields`. An empty array means
the family declares no instruction input. An enum `instructions` input is
reported with `caller` `enum` and its `values`.

### Outputs

`outputs` prevents clients from assuming every result is one waveform or one
transcript:

```json
{
  "outputs": [
    {
      "id": "stems",
      "tasks": ["sep"],
      "kind": "artifact",
      "bindings": {"tasks.run": {"slot": "artifact"}}
    },
    {
      "id": "transcript",
      "tasks": ["asr"],
      "kind": "text",
      "bindings": {"transcriptions.create": {"slot": "text"}}
    }
  ]
}
```

Output kinds are `audio`, `text`, `json`, and `artifact`. Every output binding
names a `slot` instead of a JSON pointer. The slot must be one the operation
defines: speech `audio`; `tasks.run` `audio`, `text`, or `artifact`;
transcriptions `text`; alignments `alignment`. The resolved operation's
`response_slots` entry is the body layout for that slot. Speech `audio` is raw
WAV, or base64 WAV at `/audio` with `/format` `wav` when `response_format` is
`json` or `b64_json`. `tasks.run` `audio` is base64 WAV at `/audio` with
`/sample_rate` and `/channels`.

`stream_response` is not written in the source spec. The resolved task operation
gains it when `stream` is true. Speech streaming is PCM, either SSE
`speech.audio.delta` / `speech.audio.done` or a raw PCM16 body. `tasks.run`
streaming is one JSON object with `events` and `result`.

### Startup

Schema 2 requires `default_task` (one of `tasks`), `default_mode` (one of
`modes`), and on every package `tasks` plus `default_task`. Package `tasks` is
a non-empty subset of the family tasks. The package marked `default: true` uses
the family `default_task`. A checkpoint that cannot run every family task lists
only the tasks it can run.

The resolved spec adds the objects an integrator would otherwise have to guess:

- `task_tokens` maps each spec task name to the runtime token. `clone` is
  `clon`, `design` is `vdes`, and `music` is `gen`. CLI `--task` and the server
  config field `task` take that token.
- `public_key` on every option. Request options use the local name. Session and
  load options use `<family>.<name>`. CLI flags assign `public_key=value`.
  Server `load_options` and `session_options` are objects keyed by `public_key`.
- `startup` names `default_task`, `default_task_token`, `default_mode`, the CLI
  flags (`--family`, `--model`, `--task`, `--mode`, `--load-option`,
  `--session-option`, `--request-option`), and the server config fields
  (`family`, `path`, `task`, `mode`, `load_options`, `session_options`).
- `wire_forms` on each audio input, `instruction_fields`, `response_slots`,
  and `field_aliases` on speech. `stream_response` appears only when the task
  sets `stream` to true.

`path` / `--model` is a directory, or a file in that directory when the directory
contains more than one weight. The server config also requires `id`, which the
caller chooses.
Schema 1 rejects `default_task`, `default_mode`, and package `tasks`.

### Installed-binary discovery

The control-plane walkthrough is [../integrators.md](../integrators.md).

`audiocpp_cli --family <id> --spec --json` prints the validated effective spec
on stdout without loading model weights or allocating a backend device.
`--spec` requires `--json`. Diagnostics go to stderr. The command resolves the
normal override, embedded-GGUF, workspace, and builtin-spec precedence, expands
enum presets, and merges `package_defaults.download` into each package
`download`. Package fields override the shared defaults.

`GET /v1/models?include_params=true` retains the existing `params` array and
adds the same resolved document as `spec` for each configured model. `"{}"`
means no contract was available. The experimental `--parallel-jobs` runtime
does not add `spec`.

`python3 tools/model_manager_v2.py list --json` adds `files`, `strip_prefix`,
the resolved `download` object, and `access_status` to each package row.
`access_status` is `public` when `download.gated` is false, `gated` when it is
true, and `unknown` when the flag is absent. That is the declared flag, not a
credential check. Human-readable `list` output is unchanged.

These interfaces let an integration configure a model, start it, and build
requests from a packaged binary without a source checkout or a family-specific
table. The conformance fixture is fictional and is not shipped as a model.

Five shipped families are filled in as samples: `kokoro_tts`, `chatterbox`,
`qwen3_tts`, `cosyvoice3`, and `fish_audio`. Every task those specs list has a
`task_operations` entry, and every package lists the tasks that checkpoint can
start. A task with no entry would still mean an unknown route.
Other families stay on numeric schema 1, or unversioned, until the same sections
are filled.

Some tasks share one route because the session already implemented them that way:

| Family | Tasks | Route | What differs |
|---|---|---|---|
| `kokoro_tts` | `tts` | `speech.create` | Preset `voice` string. No instruction input. `speed` aliases `speaking_rate`. |
| `chatterbox` | `tts`, `clone` | `speech.create` | Both require `/voice_ref` as a WAV path or a `voice_ref` object. `vc` uses `tasks.run` with WAV paths. `num_inference_steps` aliases `max_steps`. |
| `qwen3_tts` | `tts`, `clone` | `speech.create` | `clone` is the base checkpoint with reference audio. Custom-voice `speaker` stays on `tts`. `design` also uses `speech.create`. Instructions are plain text; the engine wraps them as a user turn. |
| `cosyvoice3` | `tts`, `clone` | `speech.create` | Reference audio is required. Instructions are plain text; the engine adds `<|endofprompt|>` when the string does not already contain it. |
| `fish_audio` | `tts`, `clone` | `speech.create` | `clone` sends optional reference audio on the same speech request. `max_tokens` aliases `max_new_tokens`. |

## Metadata vs Runtime Loading

The metadata fields above do not change tensor loading, sidecar lookup, graph
construction, or inference math.

Changing those fields does not require regenerating or reconverting a GGUF model.
If a GGUF already embeds an older spec, the old embedded metadata may still be
what a fully standalone package reports when no external `model_specs/` override
is available, but the model weights and runtime execution remain valid.

Fields that can affect runtime behavior are `sources`, `options`, and
`dependencies`. Changing `sources` changes where files or tensors are resolved
and requires refreshing any standalone GGUF package that relies on the embedded
spec. Changing `options` changes the typed runtime contract; either regenerate
standalone GGUF packages with the updated embedded spec, or add explicit
compatibility mapping in code for packages that still carry the old spec.
Changing `dependencies` must be reviewed with the loader/session behavior that
consumes it.

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

Packages are install targets, not runtime resource maps. Each package owns its
display name, precision, target directory, and exact remote files. If several
packages come from the same repo, put the shared source in
`package_defaults.download` and keep package-level `download` only for
overrides.

Experimental ports may use an empty `packages` array while conversion and
runtime validation are still local-only. In that case `ui.recommended_package`
is omitted, so model managers do not advertise a download that cannot yet be
loaded. Community and supported families must publish at least one package.

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

`kind: "modelscope_snapshot"` downloads the same way from a ModelScope
(modelscope.cn) repo. It takes the same fields (`repo` required, `revision`
optional); the only differences are that the default revision is `master`
(ModelScope's default branch) and the `gated` flag does not apply. The native
package manager resolves the endpoint through `AUDIOCPP_MS_BASE_URL`
(default `https://www.modelscope.cn`), mirroring `AUDIOCPP_HF_BASE_URL` for
Hugging Face. ModelScope requests authenticate with `AUDIOCPP_MS_TOKEN` only;
the Hugging Face token is never sent to a ModelScope endpoint.

Dependencies describe extra model-level resources required by runtime features.
Use `kind: "model"` for another model family, and `kind: "bundled_model"` for an
in-repo bundled model asset. Do not use dependencies for sidecars or tensor
files that are already part of `sources`. The dependency `scope` says where the
dependency path is consumed (`load`, `session`, or `request`), and its public
runtime option key is derived as `<family>.<option>`.

Required dependencies are unconditional. Optional dependencies must declare
typed `required_when` rows. Each row is a condition over a public option key.
Common request keys such as `return_timestamps` stay unprefixed; model-specific
session/load keys stay namespaced as `<family>.<name>`. Multiple rows are OR'd:
the dependency is needed when **any** row matches.

`dependencies[].option` must be a local name that already exists under
`options.<scope>` for the same dependency `scope`. Each
`required_when[].option_key` must refer to an option declared under
`options.<required_when.scope>` (using the public key form above).

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

Release download packages should use ready-to-run `huggingface_snapshot` GGUF
entries. Publish a GGUF package first, then expose it through
`tools/model_manager_v2.py`.

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
