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
| `CMakeLists.txt` `LOADERS` entry | `make_<family>_loader` (or the family's actual factory name) so the generated registry includes it |
| `docs/model_manager.md` package table | Lists the package; use **Unavailable** when not installable |

Dependency / subcomponent packages (`standalone=False`, with
`parent_package_id`) do **not** need their own loader.

Registered loaders that ship as bundled assets (no downloadable package) are
allowed. List them in `BUNDLED_LOADERS_WITHOUT_PACKAGE` inside
`tools/check_loader_catalog_sync.py`.

If a loader is not ready for this release tree:

1. Do not include it in an enabled `audiocpp_add_model(... LOADERS ...)` entry, and
2. Mark matching catalog packages as `UnsupportedSource(reason=...)`, **or**
   remove them from `CATALOG`, and
3. Mark the `docs/model_manager.md` package row **Unavailable**.

Do **not** leave a live `SnapshotSource` for a loader that is not emitted into
the generated registry.

Optional catalog↔registry family renames for parked stubs go in
`PARKED_FAMILY_ALIASES` in the sync check (collapse to one id when re-enabling).

## Checklist: adding a model family

1. Implement `include/engine/models/<family>/` (or `community_models/`) with a
   loader that overrides `advertised_capabilities()` so tasks/endpoints are
   explicit.
2. Register it in `CMakeLists.txt` with `audiocpp_add_model(... INCLUDES ...
   LOADERS ...)`. Prefer the factory name `make_<family>_loader` so the id
   matches the advertised family.
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
for that family set `"family"` to the same string.

## Checklist: parking or removing a family

1. Remove the family from the enabled `LOADERS` list in `CMakeLists.txt`.
2. Convert related **standalone** packages to `UnsupportedSource` with a reason
   that names the missing loader and points at this doc (or delete them).
3. Leave `family=` / `tasks=` on unsupported entries if useful for history.
4. Update README so the package row says **Unavailable**.
5. Run `python3 tools/check_loader_catalog_sync.py`.

## Family id consistency

Pick **one** family string and use it everywhere:

- C++ loader / `advertised_capabilities()`
- `make_<family>_loader()` naming (when practical)
- `CMakeLists.txt` `LOADERS` entry
- `model_specs/<family>.json`
- `ModelPackage.family`
- README “Supported Models” family column

Integrators match on the string; aliases are not implied unless listed in
`PARKED_FAMILY_ALIASES` for currently parked stubs.

## CI

`tools/check_loader_catalog_sync.py` runs in GitHub Actions on Linux/macOS/Windows
builds. It:

- Parses active vs commented `make_*_loader()` calls in `registry.cpp`
- Parses generated-registry loader declarations from `CMakeLists.txt`
- Unions both sources because most model loaders are generated from CMake, while
  bundled loaders may still be listed directly in `registry.cpp`
- Compares them to installable standalone packages from `model_manager.py`
- Cross-checks the `docs/model_manager.md` recommended package table
- Does **not** require a compiled binary

```bash
python3 tools/check_loader_catalog_sync.py --self-test
python3 tools/check_loader_catalog_sync.py
```
