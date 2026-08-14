# Native UI Config Mode Tracking

Issue: https://github.com/0xShug0/audio.cpp/issues/234

This note tracks native UI gaps seen when `audiocpp_server` is launched with a
server config but without `--ui-management`.

## Bugs

- The Studio model picker is built from the embedded model catalog, not from the
  configured `/v1/models` response. In config mode this exposes models that the
  server was not configured to serve.
- The UI uses catalog IDs as request model IDs. Configured server model IDs are
  deployment-local, so a resident configured model may not be found if its ID
  differs from the catalog ID.
- Management-only endpoints such as path inspection are correctly forbidden
  when `ui_management=false`, but the UI still probes them and presents unclear
  state.
- Package choice defaults are precision-centric. This can hide installed BF16
  packages behind a Q8 default and cannot represent multi-axis variants such as
  ACE-Step base vs turbo.

## Intended Direction

- Add a config-mode UI path where selectable models come from `/v1/models`.
- Keep catalog/package-management behavior behind `--ui-management`.
- Make loaded/resident state use the actual server model ID.
- Represent package variants without assuming precision is the only choice axis.
