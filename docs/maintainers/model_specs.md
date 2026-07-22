# Model Specs

`model_specs/*.json` is the target source of truth for model metadata, package
layout, downloads, UI hints, CLI options, runtime capabilities, and companion
models.

Specs without `schema_version` are legacy source-layout specs. New or migrated
specs should use `schema_version: 1`.

## Schema V1

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
| `runtime` | Runtime tags and default package format. |
| `capabilities` | Stable capability booleans and language hints. |
| `options` | Typed request/session/load options. |
| `packages` | Installable model packages and download sources. |
| `layouts` | Resource/tensor layouts used by packages. |
| `companions` | Runtime peer models or external resources. |
| `ui` | UI/catalog hints. |
| `sources` | Temporary runtime bridge for current package-spec loading. |

Common options must use canonical names such as `seed`, `language`,
`voice_ref`, `text_chunk_mode`, `text_chunk_size`, `max_new_tokens`,
`temperature`, `top_p`, `top_k`, and `return_timestamps`.

Model-specific options must be namespaced as `<family>.<name>`.

Supported package source kinds are `huggingface_snapshot`, `local_snapshot`,
`composite_snapshot`, `converter`, and `unsupported`.

The C++ `framework/model_spec` subsystem is the authoritative schema gate.
`audiocpp_cli`, `audiocpp_server`, and GGUF loading fail when a typed v1 field
is invalid.

Run the toy C++ demo through the production subsystem:

```bash
cmake --build build/debug --target model_spec_demo --parallel $(nproc)
build/debug/bin/model_spec_demo \
  examples/model_spec_demo/specs/toy_qwen3_asr.json \
  examples/model_spec_demo/toy_package
```

The toy browser UI reads `examples/model_spec_demo/specs/toy_qwen3_asr.json`
directly, so there is no duplicated demo catalog.
