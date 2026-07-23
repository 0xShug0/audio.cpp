# Maintaining Loaders and the Package Catalog

Integrators (CLI users, servers, and UIs such as Studio) treat two exports as
**authoritative**:

1. **Runtime loaders** — `audiocpp_cli --list-loaders --json`
2. **Install packages** — `python tools/model_manager.py list --json`

Those surfaces must stay in sync. A package that is installable in the catalog
but whose `family` is missing from `--list-loaders` looks available to users and
then fails at runtime or in search/install UIs.

## The rule

For every **installable, standalone** `ModelPackage`:

| Field | Must match |
|---|---|
| `ModelPackage.family` | The loader family string advertised by the C++ loader |
| `model_specs/<family>.json` | Present when the family uses package-spec loading |
| `registry.cpp` entry | Uncommented `make_<family>_loader()` (or the family's actual factory name) |
| README package table | Lists the package; use **Unavailable** when not installable |

Dependency / subcomponent packages (`standalone=False`, with
`parent_package_id`) do **not** need their own loader.

Registered loaders that ship as bundled assets (no downloadable package) are
allowed. List them in `BUNDLED_LOADERS_WITHOUT_PACKAGE` inside
`tools/check_loader_catalog_sync.py`.

If a loader is not ready for this release tree:

1. Keep it **commented out** in `src/framework/runtime/registry.cpp`, and
2. Mark matching catalog packages as `UnsupportedSource(reason=...)`, **or**
   remove them from `CATALOG`, and
3. Mark the README package row **Unavailable**.

Do **not** leave a live `SnapshotSource` for a commented-out loader.

Optional catalog↔registry family renames for parked stubs go in
`PARKED_FAMILY_ALIASES` in the sync check (collapse to one id when re-enabling).

## Runtime companions (peer models)

Some loaders take **another model path** at load/session time (forced aligner,
VAD, codec, ASR for best-of-N, etc.). Integrators need a machine-readable way to
discover those peers — not by hard-coding family graphs in UIs.

Advertise them from the loader via `advertised_companions()`. They appear under
each loader in `audiocpp_cli --list-loaders --json` as an optional `companions`
array (`schema_version` stays **1**; additive fields only).

| Field | Meaning |
|---|---|
| `id` | Stable integrator id (e.g. `forced_aligner`) |
| `config_key` | Option key that receives the peer path |
| `scope` | `session`, `load`, or `request` |
| `target_family` | Registered loader family to install/select; omit/empty for external dirs |
| `optional` | Whether the primary can run without the peer |
| `required_for` | Capability/feature keys that need this peer |
| `label` | Short UI label |
| `bundled_default` | Optional in-tree default path (e.g. Silero under `assets/`) |

Rules:

1. Prefer a real `target_family` when the peer is an audio.cpp loader.
2. Non-empty `target_family` must be an **active** registry family (or a bundled
   loader such as `silero_vad`).
3. Use empty `target_family` + a clear `label` only for external/non-loader peers
   (e.g. Vevo2 Whisper directory).
4. Do **not** invent companions for catalog `parent_package_id` install deps —
   those stay install-time only.
5. Keep companions in the loader that owns the path option; do not duplicate a
   Studio-owned seed graph.

## Checklist: adding a model family

1. Implement `include/engine/models/<family>/` (or `community_models/`) with a
   loader that overrides `advertised_capabilities()` so tasks/endpoints are
   explicit. Override `advertised_companions()` when the family takes peer paths.
2. Register it in `src/framework/runtime/registry.cpp` (include +
   `available_loaders` entry). Prefer the factory name
   `make_<family>_loader()` so the id matches the advertised family.
3. Add `model_specs/<family>.json` when the family needs package-spec discovery.
4. Add one or more `ModelPackage` entries in `tools/model_manager.py`:
   - Set `family="<family>"` explicitly when the package id does not strip cleanly
     to the loader id.
   - Set `tasks=(...)` when defaults would be ambiguous.
   - Use `standalone=False` + `parent_package_id` for tokenizers / subcomponents.
5. Update README supported-model / package tables.
6. Run:

```bash
python3 tools/check_loader_catalog_sync.py --self-test
python3 tools/check_loader_catalog_sync.py
# after building:
build/.../bin/audiocpp_cli --list-loaders --json
python3 tools/model_manager.py list --json
```

Confirm the new family appears in `--list-loaders` and that installable packages
for that family set `"family"` to the same string. Confirm any `companions[].target_family`
values resolve to registered loaders.

## Checklist: parking or removing a family

1. Comment out the include and `make_*_loader()` entry in `registry.cpp`.
2. Convert related **standalone** packages to `UnsupportedSource` with a reason
   that names the missing loader and points at this doc (or delete them).
3. Leave `family=` / `tasks=` on unsupported entries if useful for history.
4. Update README so the package row says **Unavailable**.
5. Run `python3 tools/check_loader_catalog_sync.py`.

## Family id consistency

Pick **one** family string and use it everywhere:

- C++ loader / `advertised_capabilities()`
- `make_<family>_loader()` naming (when practical)
- `model_specs/<family>.json`
- `ModelPackage.family`
- README “Supported Models” family column

Integrators match on the string; aliases are not implied unless listed in
`PARKED_FAMILY_ALIASES` for currently parked stubs.

## CI

`tools/check_loader_catalog_sync.py` runs in GitHub Actions on Linux/macOS/Windows
builds. It:

- Parses active vs commented `make_*_loader()` calls in `registry.cpp`
- Compares them to installable standalone packages from `model_manager.py`
- Cross-checks the README recommended package table
- Parses `advertised_companions()` initializers in `**/loader.cpp` and ensures
  non-empty `target_family` values are active registry families
- Does **not** require a compiled binary

```bash
python3 tools/check_loader_catalog_sync.py --self-test
python3 tools/check_loader_catalog_sync.py
```
