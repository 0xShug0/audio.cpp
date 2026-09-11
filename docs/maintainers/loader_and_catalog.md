# Maintaining Loaders and Model-Spec Packages

For release model downloads, `model_specs/*.json` is the source of truth and
both `tools/model_manager_v2.py` and the optional native
`audiocpp_model_manager` consume that package catalog.

Integrators treat two surfaces as authoritative:

1. **Runtime loaders** — `audiocpp_cli --list-loaders --json`
2. **Install packages** — `python3 tools/model_manager_v2.py list --json` and,
   when native model management is enabled,
   `audiocpp_model_manager list`

Those surfaces must stay in sync. A package that is installable from
`model_specs/*.json` but whose `family` is missing from `--list-loaders` looks
available to users and then fails at runtime or in search/install UIs.

## The Rule

For every installable release package:

| Field | Must match |
|---|---|
| `model_specs/<family>.json` `family` | The family id emitted by the runtime loader registry |
| `model_specs/<family>.json` `packages[]` | Ready-to-use GGUF package entries installable by the Python and native model managers |
| `CMakeLists.txt` `LOADERS` entry | `make_<family>_loader` or the family's actual factory name so the generated registry includes it |
| README supported-model table | Lists the released family and its tested runtime format |

New release packages should be standalone GGUF downloads. If a model is not
available as a ready-to-run GGUF package, keep it out of the release download
surface until that package exists.

## Adding A Model Family

1. Implement the family under `src/models/<family>/` or
   `src/community_models/<family>/`.
2. Register it in `CMakeLists.txt` with `audiocpp_add_model(... INCLUDES ...
   LOADERS ...)`.
3. Add or update `model_specs/<family>.json` with the release metadata,
   normalized runtime options, sources, and GGUF `packages[]`.
4. Make one GGUF package the default package unless there is a specific reason
   to require users to choose a variant.
5. Update the model guide and README supported-model table.
6. Check the install and runtime surfaces:

```bash
python3 tools/model_manager_v2.py list --json
python3 tools/model_manager_v2.py info <family-or-package-id>
audiocpp_model_manager list
audiocpp_model_manager info <family-or-package-id>
build/debug/bin/audiocpp_cli --list-loaders --json
```

Confirm the family appears in both the loader list and the v2 package list.

## Parking Or Removing A Family

If a loader is not ready for the release tree:

1. Do not include it in an enabled `audiocpp_add_model(... LOADERS ...)` entry.
2. Do not publish it as a default installable package in `model_specs/*.json`.
3. Update README and model docs so the family is not advertised as released.

Do not leave a visible GGUF package entry for a family that is not emitted into
the runtime registry.

## Family Id Consistency

Pick one runtime family id and keep it consistent across the surfaces that users
and tooling match:

- `IVoiceModelLoader::family()`, which is what `--list-loaders` emits
- `model_specs/<family>.json` `family`
- `CMakeLists.txt` `LOADERS` entry
- README supported-model family column

The factory name in CMake is a registration symbol, not the family id itself.
Use `make_<family>_loader()` when practical because it is easier to audit, but
the runtime contract comes from `loader->family()` matching the spec `family`.
Integrators match on that family string; aliases are not implied.
