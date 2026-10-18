#!/usr/bin/env bash
#
# Regression check for the build helper scripts.
#
# `--target` and `--conda-env` are optional, so the `TARGETS` and `RUNNER`
# arrays are empty in the common case. Expanding an empty array as
# `"${arr[@]}"` is an error under `set -u` on bash 3.2 (stock macOS /bin/bash)
# and bash < 4.4, which used to make `scripts/build_metal.sh` fail right after a
# successful configure — the build never started.
#
# Both helper scripts are driven here with stub `cmake`/`xcrun` executables and
# no `--target`, under the stock bash, so the whole argument/expansion tail runs
# without configuring or building anything real.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
BASH_BIN="${CHECK_BASH_BIN:-/bin/bash}"

fail() {
    echo "FAIL: $*" >&2
    exit 1
}

work_dir="$(mktemp -d "${TMPDIR:-/tmp}/audiocpp-build-script-check.XXXXXX")"
cleanup() {
    /bin/rm -rf "$work_dir"
}
trap cleanup EXIT

script_count=0
for script in "$REPO_ROOT"/scripts/*.sh; do
    "$BASH_BIN" -n "$script" || fail "syntax error: $script"
    script_count=$((script_count + 1))
done
echo "ok: $BASH_BIN -n passed for $script_count scripts"

# Stub the tools the scripts call so their tail runs without doing real work.
stub_dir="$work_dir/bin"
mkdir -p "$stub_dir"

cat > "$stub_dir/cmake" <<'STUB'
#!/usr/bin/env bash
printf 'cmake %s\n' "$*" >> "$CHECK_STUB_LOG"
exit 0
STUB

cat > "$stub_dir/xcrun" <<'STUB'
#!/usr/bin/env bash
printf 'xcrun %s\n' "$*" >> "$CHECK_STUB_LOG"
exit 0
STUB

chmod +x "$stub_dir/cmake" "$stub_dir/xcrun"

run_script() {
    local label="$1"
    local script="$2"
    shift 2
    local log="$work_dir/$label.log"
    local stub_log="$work_dir/$label.stub.log"
    : > "$stub_log"

    if ! PATH="$stub_dir:$PATH" CHECK_STUB_LOG="$stub_log" "$BASH_BIN" "$script" "$@" > "$log" 2>&1; then
        echo "--- $label output (tail) ---" >&2
        tail -n 20 "$log" >&2
        fail "$label: $script exited non-zero under $BASH_BIN ($("$BASH_BIN" -c 'echo "$BASH_VERSION"'))"
    fi
    grep -q '^cmake .*-S ' "$stub_log" || fail "$label: cmake configure was never reached"
    grep -q '^cmake --build ' "$stub_log" || fail "$label: cmake --build was never reached"
    echo "ok: $label completed under bash $("$BASH_BIN" -c 'echo "$BASH_VERSION"')"
}

if [[ "$(uname -s)" == "Darwin" ]]; then
    run_script build_metal "$REPO_ROOT/scripts/build_metal.sh" \
        --build-dir "$work_dir/metal-build" \
        --model-set custom --models breeze_tts
else
    echo "skip: build_metal.sh (requires macOS)"
fi

# build_linux.sh parses the same way on any host, which is what this check cares
# about (the real build is Linux-only).
run_script build_linux "$REPO_ROOT/scripts/build_linux.sh" \
    --backend cpu --build-dir "$work_dir/linux-build"

echo "ok: build helper scripts"
