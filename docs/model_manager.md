# Model Manager

Use the **Models** tab in the WebUI or a command-line manager to download model
packages. Packages describe the files, precision, download source and install
layout in `model_specs/*.json`. Installing a package does not load it for inference.

## Choose a manager

| Interface | Requirements | Package selection | Install directory |
|---|---|---|---|
| WebUI **Models** tab | Server built with native model management; launch with `--ui-management` | Choose a model and one of its published packages | Models folder shown in the UI |
| `audiocpp_model_manager` | Native manager executable | Exact package ID from `list` | `--models-root`, default `models` relative to the working directory |
| `tools/model_manager_v2.py` | Python 3 and the source checkout; no third-party Python packages required | Exact package ID, or family with optional format/precision filters | `--models-root`, default `models` relative to the working directory |

The native CLI and WebUI share the C++ package manager. Normal browser downloads
and native CLI downloads do not invoke Python. The Python v2 tool is a separate
supported downloader using the same specs; its command names and flags differ.

## WebUI downloads

Start a native-manager-enabled server:

```bash
audiocpp_server --ui --ui-management --backend cuda --models-root /path/to/models
```

Open `http://127.0.0.1:8080`, select **Models**, and choose the model and package
you want to install. Use the same models folder when installing from a CLI.
Without an explicit `--models-root` or server configuration, the server defaults
to `models/` beside its executable, which can differ from the CLI's working-directory
`models/`. The Models page can change the folder during the server session.

Downloads run in the background. The UI supports progress, stopping downloads,
cleaning partial downloads, checking for updates and deleting installed packages.
An installed model can then be selected in Studio or Arena where supported.

If management is unavailable, use a build with
`AUDIOCPP_BUILD_NATIVE_MODEL_MANAGER=ON`; see [building the native manager](#building-the-native-manager).
A server without that feature can still serve its embedded UI and run models
from its configuration. See the [WebUI guide](../webui/README.md).

## Native CLI

These commands assume the binaries are on `PATH`; source builds place them in
`build/debug/bin/` (or the build directory you selected).

```bash
audiocpp_model_manager list
audiocpp_model_manager info qwen3_asr_0_6b_q8_0 --models-root models
audiocpp_model_manager install qwen3_asr_0_6b_q8_0 --models-root models
```

Use the **exact package ID**, not a family name. The native CLI has no
`--precision` or `--format` selector: select the corresponding package ID instead.
`list` and `info` print JSON. By default they inspect local presence only;
`--remote` also requests file sizes and update metadata from the provider:

```bash
audiocpp_model_manager info qwen3_asr_0_6b_q8_0 --models-root models --remote
```

| Command | Behavior |
|---|---|
| `list [--remote]` | List packages and local presence, optionally querying remote metadata |
| `info PACKAGE [--remote]` | Show the inventory entry for one exact package ID |
| `install PACKAGE [--overwrite]` | Download the package; `--overwrite` replaces its existing files |
| `clean PACKAGE` | Remove staging directories associated with the package's target directory |
| `remove PACKAGE` | Remove an installed package's managed files, retaining sidecars needed by another installed package |

All commands accept `--models-root PATH` and `--repository-root PATH`.
`--models-dir` remains accepted as a compatibility alias for `--models-root`.
The native manager embeds the catalog. If `--repository-root` (default: the
working directory) contains JSON files in `model_specs/`, those files
**replace the embedded catalog**. This is a development override; the deployed
binary does not require the repository checkout.

`install` reports progress on stderr and its completion message on stdout.
`clean` removes interrupted staging data, not the installed model. Stop active
downloads before cleaning their staging directories.

For a container image with the native executable installed:

```dockerfile
RUN audiocpp_model_manager install qwen3_asr_0_6b_q8_0 --models-root /models
```

## Python CLI

Run from the source checkout, or provide the script's full path. Its specs are
read from the checkout, independently of the working directory.

```bash
python3 tools/model_manager_v2.py list
python3 tools/model_manager_v2.py info qwen3_asr_0_6b_q8_0
python3 tools/model_manager_v2.py install qwen3_asr_0_6b_q8_0 --models-root models
```

A family selects its declared default package when unambiguous. Filters apply
only to family selection; use the precision spelling reported by `list`:

```bash
python3 tools/model_manager_v2.py info qwen3_asr --format gguf --precision q8_0
python3 tools/model_manager_v2.py install qwen3_asr --format gguf --precision q8_0 --models-root models --dry-run
```

If several matching packages have no unique default, selection fails and lists
the package IDs to choose from. An exact package ID cannot be combined with
`--format` or `--precision`.

| Command | Behavior |
|---|---|
| `list [--json]` | List package IDs, families, formats, precisions and defaults |
| `info PACKAGE_OR_FAMILY [--json]` | Show source, required files and target directory |
| `install PACKAGE_OR_FAMILY` | Download the selected snapshot package |
| `installed --models-root PATH` | Report local file presence for every package as JSON, without network access |
| `sizes [PACKAGE_ID ...] --models-root PATH` | Query remote sizes and compare local/remote version metadata; output is JSON |
| `uninstall PACKAGE_OR_FAMILY --models-root PATH` | Remove that package's files, preserving shared sidecars and sibling precisions |
| `clean-partial PACKAGE_OR_FAMILY --models-root PATH` | Remove staging directories associated with its target directory |

`install` supports `--overwrite`, `--dry-run` (show the plan without downloading)
and `--check` (check remote file metadata without downloading weights).
The global `--specs-dir PATH` override must precede the subcommand.
Python v2 downloads published snapshot files; it does **not** convert checkpoints.

### ModelScope override

Both managers can install packages whose specs declare a ModelScope source.
Only Python v2 also offers a per-command source override, for `install` and `sizes`:

```bash
python3 tools/model_manager_v2.py install qwen3_tts --source modelscope --source-repo HereIsMark/audio.cpp-gguf
```

The mirror must provide the files declared by the selected package. Omitting
`--source-repo` reuses the spec's repository name. An unset or `main` revision
maps to ModelScope's `master`; other revisions are retained. `--source-repo`
requires `--source modelscope`. Use the same source for later version checks,
because providers use different file identifiers.

## Packages, paths and updates

A **family** identifies an inference implementation, such as `qwen3_asr`. A
**package ID** identifies a downloadable set of files, such as
`qwen3_asr_0_6b_q8_0`. A package can contain one GGUF or several components;
installing it downloads all files declared by that package. Sibling precisions
can coexist in a shared target directory.

The install location is the models root plus the spec's `target_directory` and
its declared file paths, with `strip_prefix` removed where specified. This
location is not necessarily a single file named after the family. Use the model's
[usage guide](../README.md#supported-models) for the correct inference path and
`--family`; the installer does not choose a runtime family for an ambiguous folder.

Completed installs write `.audiocpp-package-<id>.json` beside the package files.
It records the source revision and per-file metadata used for freshness checks.
File presence alone is not an integrity check. Manually copied or older packages
can remain usable while showing **Version unknown**. Reinstall with `--overwrite`
to refresh the manifest; installing without it can simply report **Already installed**.
Update status compares available per-file identifiers, not just the repository's
latest commit.

Both managers download into staging directories before publishing files. An
interrupted process can leave staging data; use `clean` (native) or
`clean-partial` (Python) after stopping the download. Staging cleanup is grouped
by target directory, so it can include sibling precisions. Neither command
uninstalls completed weights. Publication is per file when merging into an
existing package directory; do not assume a whole multi-file replacement is
one atomic transaction.

For model repositories, start with [audio-cpp on Hugging Face](https://huggingface.co/audio-cpp)
and the individual model guides, which also link community-hosted packages.
Explicit package IDs let you install the available precision variants through
either manager. See the [GGUF guide](gguf.md) for format support and the
[Q8 report](reports/gguf_q8_performance.md) for measured precision trade-offs.

## Authentication and mirrors

Both managers read Hugging Face credentials from `HF_TOKEN`, then
`HUGGING_FACE_HUB_TOKEN`, then `~/.cache/huggingface/token` (on Windows, beneath
the user's profile directory). Accept any gated model's terms on Hugging Face
before installing it. ModelScope uses `AUDIOCPP_MS_TOKEN` when required.
Provider credentials are kept separate.

| Setting | Native manager | Python v2 |
|---|---|---|
| Hugging Face endpoint | `AUDIOCPP_HF_BASE_URL` | `HF_ENDPOINT` |
| ModelScope endpoint | `AUDIOCPP_MS_BASE_URL` | `AUDIOCPP_MS_BASE_URL` |
| Default endpoints | `https://huggingface.co`, `https://www.modelscope.cn` | Same |

Endpoint overrides change the host, not the package's file list or repository.
Configure credentials and mirrors in the environment of the CLI or server
process, not just in the browser.

### Native Xet transfers

The native downloader uses Hugging Face's Xet transport for eligible files,
with parallel range downloads and SHA-256 verification when the Hub supplies a
hash. It needs no additional installed dependency. Before publication, supported
Xet transport failures can fall back to HTTP and are reported in progress;
a completed reconstruction that fails size or hash checks is an error.
Python v2 uses HTTP, not the native Xet implementation.

| Native setting | Default | Effect |
|---|---|---|
| `AUDIOCPP_HF_XET` | enabled | `0` selects HTTP instead |
| `AUDIOCPP_HF_XET_CONCURRENCY` | `16` | Parallel Xet range requests per file, 1–256 |
| `AUDIOCPP_HF_XET_VERIFY` | enabled | `0` disables optional SHA-256 verification; size checks remain |

## Building the native manager

Native management is opt-in. It adds the standalone executable and server
package-management endpoints; ordinary builds without it do not fetch or link
its HTTP/TLS dependency.

```bash
cmake -S . -B build/debug -DCMAKE_BUILD_TYPE=Debug -DAUDIOCPP_BUILD_NATIVE_MODEL_MANAGER=ON
cmake --build build/debug --parallel --target audiocpp_model_manager audiocpp_server
```

Build helpers expose `--native-model-manager` on Linux/macOS and
`-NativeModelManager` on Windows; see [build instructions](../README.md#build-yourself).

The default HTTPS backend fetches a pinned BoringSSL archive at configure time
and links it statically. It needs no system OpenSSL development package,
libcurl, or separate TLS DLLs at runtime. For offline configuration, provide
`-DAUDIOCPP_BORINGSSL_ARCHIVE=/path/to/boringssl.tar.gz`. Packagers can instead
select `-DAUDIOCPP_USE_SYSTEM_OPENSSL=ON` and provide OpenSSL development files
and the required runtime libraries. Helpers also accept `--boringssl-archive`
and `--system-openssl` (Windows: `-BoringSslArchive`, `-SystemOpenSsl`).

## Legacy conversion

`tools/model_manager_deprecated.py` is a separate legacy download/conversion
catalog. It is not used for normal managed WebUI downloads, and its package IDs
need not match the current spec-backed catalog. Use it only when a model's guide
requires that conversion path. Dependencies can include PyTorch, safetensors,
PyYAML and model-specific inputs.

The server retains a legacy API branch for direct callers that supply conversion
fields such as `source_directory`, `source_file`, `output_file` or `variant`. That
branch invokes the deprecated Python helper and requires a source checkout or
portable bundle containing it; the normal WebUI does not send those fields.
