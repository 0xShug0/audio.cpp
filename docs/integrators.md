# Integrator's Guide

This is the contract for a UI or control plane built against an installed
audio.cpp binary. Read the resolved model spec, the CLI help, and the server
config documented here. Family names, categories, capability tags, and option
names do not select a route or a task.

Schema authors edit `model_specs/*.json`. The field grammar for that work is
[docs/maintainers/model_specs.md](maintainers/model_specs.md). This page is the
reader's side of the same contract.

## Surfaces

Use these, in this order. Each one is available from a packaged build.

| Question | Command or request |
|---|---|
| Which families can this binary load, and which runtime task tokens do they advertise? | `audiocpp_cli --list-loaders --json` |
| Which packages can be installed, and where do the files land? | `python3 tools/model_manager_v2.py list --json` |
| What is the family contract, including startup and request bindings? | `audiocpp_cli --family <id> --spec --json` |
| The same contract for a model the server already has configured | `GET /v1/models?include_params=true` → `spec` |
| Which flag names and runtime tokens does the CLI accept? | `audiocpp_cli --help` |

`--list-loaders` answers a binary question: the loader is linked, and it
advertises runtime task tokens and modes. Its `api_endpoints` and
`instructions_policy` fields are capability-derived advertisements. A schema 2
spec replaces them. Take the route from `task_operations`, and take instruction
support from an `inputs` entry whose id is `instructions`.

## Read the resolved spec

```bash
audiocpp_cli --family qwen3_tts --spec --json
```

`--spec` requires `--json` and `--family`. It prints the validated spec on
stdout and does not load weights or allocate a device. Diagnostics go to
stderr and start with `audiocpp_cli failed:`. The command follows the normal
override, embedded-GGUF, workspace, and builtin-spec precedence, expands enum
presets to `values`, and merges `package_defaults.download` into each package.
A package's own `download` wins on overlapping keys.

`GET /v1/models?include_params=true` adds that same document as `spec` on each
configured model, next to the existing `params` array. `"{}"` means no
contract was available. The experimental `--parallel-jobs` runtime does not
add `spec`.

Every resolved spec adds:

- `task_tokens`, mapping each spec task name to the runtime token used by
  `--task` and by the server config field `task`. `clone` is `clon`, `design`
  is `vdes`, and `music` is `gen`. `--help` lists the tokens.
- `public_key` on every option. Request options use the local name. Session
  and load options use `<family>.<name>`.

A schema 2 spec also adds `startup`, fills each `task_operations` entry with
method, path, encoding, and response type, and lists `tasks` and
`default_task` on every package.

## Schema 2 and schema 1

`schema_version` is a number. `1` is the original typed contract. `2` is the
integrator contract. A spec with no `schema_version` is an older layout spec.

| You need | Schema 2 | Schema 1 or unversioned |
|---|---|---|
| Runtime token for a spec task name | `task_tokens` | `task_tokens` when the spec lists `tasks` |
| Option key for CLI and server config | `public_key` | `public_key` when the spec lists `options` |
| Which checkpoint runs which task | `packages[].tasks` and `packages[].default_task` | Unknown |
| Default task and mode | `default_task`, `default_mode`, and `startup` | Unknown |
| HTTP route for a task | `task_operations` | Unknown |
| Request fields | `inputs` and option `bindings` | Unknown |
| Result kind | `outputs` | Unknown beyond the route's response type, and the route itself is unknown |

An omitted schema 2 section means unknown. It does not mean the family rejects
that role. Five families ship as schema 2 samples: `kokoro_tts`, `chatterbox`,
`qwen3_tts`, `cosyvoice3`, and `fish_audio`. Other families stay on schema 1,
or unversioned, until the same sections are filled.

## Choose a package and a task

Schema 2 requires:

- `default_task`, one of `tasks`
- `default_mode`, one of `modes`
- on every package, a non-empty `tasks` subset of the family tasks, and a
  `default_task` that belongs to that subset

The package with `"default": true` uses the family `default_task`. A
checkpoint that cannot run every family task lists only its own tasks. Qwen3-TTS
is the example: base packages list `tts` and `clone`, custom-voice packages
list `tts`, and voice-design packages list `design`.

Pick the package, then pick a spec task from that package's `tasks`. The value
you pass to the runtime is `task_tokens[<spec task>]`. `startup.task_value` is
the string `task_tokens[<spec task>]`, which is that rule. `startup.default_task_token`
is the token for the family default.

`startup.model_path` says what `--model` and the server `path` field are: a
directory, or a file in that directory when the directory contains more than
one weight. After `model_manager_v2.py install`, the directory is
`<models-root>/<target_directory>`.

## Start the model

`startup.cli` names the CLI flags. `startup.option_assignment` is
`public_key=value`. `startup.server_option_key` is `public_key`.

```bash
audiocpp_cli \
  --family qwen3_tts \
  --model /path/to/models/Qwen3-TTS-12Hz-1.7B-CustomVoice-GGUF \
  --task tts \
  --mode offline \
  --session-option qwen3_tts.mem_saver=false
```

`--task` receives `tts` because `task_tokens.tts` is `tts`. A voice-design
package uses `--task vdes` because `task_tokens.design` is `vdes`. Load and
session values use the option's `public_key`, not its local `name`.

`startup.server_config` names the server.json fields. `startup.server_id` is
`caller-chosen`: `id` is the name later requests use, and the spec does not
assign it. The always-on way to start a model is the `models` array in
server.json. `POST /v1/models/load` exists only when the server is run with
`--ui-management`.

```json
{
  "id": "qwen-design",
  "family": "qwen3_tts",
  "path": "/path/to/models/Qwen3-TTS-12Hz-1.7B-VoiceDesign-GGUF",
  "task": "vdes",
  "mode": "offline",
  "load_options": {},
  "session_options": {
    "qwen3_tts.mem_saver": false
  }
}
```

`task` is the runtime token. `load_options` and `session_options` are objects
keyed by `public_key`.

## Build an HTTP request

1. Select the spec task the loaded model was started with. The task is fixed
   at startup. It is not a field of `POST /v1/audio/speech`.
2. Read `task_operations[<spec task>]`. Use `preferred_operation` unless you
   have a reason to use another id in `operations`.
3. The resolved entry has `method`, `path`, `encoding`, and
   `response_content_type`. When the route accepts more than one response
   body, it also has `response_formats` and `response_format_pointer`.

| Operation | Method and path | Encoding | Response |
|---|---|---|---|
| `speech.create` | `POST /v1/audio/speech` | JSON | `audio/wav`, unless `response_format` is `json` or `b64_json` |
| `tasks.run` | `POST /v1/tasks/run` | JSON | `application/json` |
| `transcriptions.create` | `POST /v1/audio/transcriptions` | multipart | `application/json` |
| `alignments.create` | `POST /v1/audio/alignments` | multipart | `application/json` |

Every JSON and multipart call names the loaded server entry. That field is
`model`, and its value is the caller-chosen `id` from server.json. It is not a
spec binding, because it selects a server entry rather than a model input.
Multipart routes put it in a part named `model`.

Then fill the body from `inputs` and from `options.request`:

- Keep rows whose `tasks` include the selected spec task. A request option
  that omits `tasks` applies to every task in the family.
- Honor `required`. An omitted `required` on an input means the requirement is
  unknown.
- `alternatives` names other input ids or option keys that can satisfy the
  same role. Supply at most one member of that set.
- `bindings[<operation>]` is the wire location. JSON operations use
  `json_pointer`. Multipart operations use `multipart_field`. A row used by
  two operations has one binding for each.
- `tasks.run` request pointers start with `/request/`. The `model` id stays
  beside that object. Speech pointers are body fields the route already
  accepts (`/input`, `/voice`, `/voice_ref`, `/instructions`, and the rest
  listed in the maintainer doc) or `/options/<public_key>`.
- Instructions exist only when an input with id `instructions` lists the
  selected task. A voice is required only when an input with id `voice` lists
  that task and says `required: true`. `ui.default_voice` and
  `ui.builtin_voices` are catalog labels.

`presentation` (`label`, `group`, `advanced`) is display metadata. A client
can build a valid request without it.

Option rows carry `type`, `description`, `required`, and, when the runtime has
a stable literal, `default`. Numeric rows include `min` and `max` when a range
is known. Types are `bool`, `int`, `float`, `string`, `enum`, `path`,
`audio_path`, `string_list`, `float_list`, `path_list`, and
`audio_path_list`. An enum uses `values` in the resolved spec. A literal
`false` or `0` default is a real default. A missing `default` means unknown.

Input schema types are `string`, `enum`, `audio`, and `artifact`.

For a Qwen custom-voice model started with spec task `tts`, `preferred_operation`
is `speech.create`. The text input binds to `/input`. The optional `voice`
input binds to `/voice`. The `speaker` request option binds to
`/options/speaker` and lists only `tts`, so a voice-design model does not show
it. The request is:

```bash
curl http://127.0.0.1:8080/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -o out.wav \
  -d '{
    "model": "qwen-custom",
    "input": "Schema 2 named this field.",
    "options": {"speaker": "ryan"}
  }'
```

`"model": "qwen-custom"` is the server id. To receive JSON instead of a WAV
body, set `response_format` to `json` or `b64_json`, which is
`response_format_pointer` on that operation. The JSON body is
`{"audio": "<base64 wav>", "format": "wav", ...}`.

A voice-design model is a different package and a different startup task
(`vdes`). Its `instructions` input lists `design` and binds to
`/instructions`. The same speech route is used because
`task_operations.design.preferred_operation` says so, not because the family
is a TTS model.

Chatterbox `tts` and `clone` also share `speech.create`. Both require
reference audio: the `voice` input lists those tasks, `required: true`, and
binds to `/voice_ref`. `tts` is not a preset-voice path. `vc` uses
`tasks.run`, and its inputs bind under `/request/`. `seed` carries both
bindings, `/seed` and `/request/seed`, because that option lists all three
tasks.

CLI text and audio flags (`--text`, `--audio`, `--voice-ref`, `--out`) are
documented by `audiocpp_cli --help` and [docs/usage.md](usage.md). Per-family
knobs on the CLI go through `--request-option <public_key>=<value>`.

## Read the result

Use `response_content_type` for the HTTP body. When `outputs` is present, each
row has `kind` (`audio`, `text`, `json`, or `artifact`) and the spec tasks it
belongs to. An output binding, when declared, locates that artifact in the
JSON body. A missing `outputs` array means the result roles are unknown; the
operation's response type is still known.

`stream: true` on an operation is legal only when `modes` includes
`streaming`. The spec does not describe streaming event payloads.

## Packages and downloads

`model_manager_v2.py list --json` rows include `family`, `id`, `display_name`,
`format`, `precision`, `default`, `target_directory`, `files`,
`strip_prefix`, `download`, and `access_status`. `access_status` is `public`
when `download.gated` is false, `gated` when it is true, and `unknown` when
the flag is absent. That is the declared flag, not a credential check.

```bash
python3 tools/model_manager_v2.py install qwen3_tts_1_7b_voicedesign_q8_0
```

Human-readable `list` output is unchanged. `info <id> --json` prints one
package. The native `audiocpp_model_manager` speaks the same catalog when the
binary was built with native model management.

`dependencies` names peer models or bundled assets. A `required_when` entry
is the condition: `scope`, `option_key`, and `equals`. Evaluate it against the
option values you send. An empty `dependencies` array means the family
declares none.

## Checklist

1. `--list-loaders --json` includes the family.
2. `list --json` selects a package id, and `access_status` tells you whether
   the download is declared public, gated, or unknown.
3. `--family <id> --spec --json` is schema 2, or you stop and treat routes,
   inputs, and package tasks as unknown.
4. The package's `tasks` includes the spec task you will run.
5. Startup uses `task_tokens`, `public_key`, and the flag names in `startup`.
6. The HTTP call uses `preferred_operation`'s method, path, and encoding, the
   server `id` in `model`, and the bindings for that operation.
7. The response is read as `response_content_type`, with `outputs` when the
   spec declares them.
