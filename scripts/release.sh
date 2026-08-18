#!/usr/bin/env bash
#
# Release audio.cpp prebuilt binaries.
#
# Builds every backend (macOS Metal, Linux CPU/Vulkan, Windows CPU/Vulkan/CUDA
# 12.4 + 13.3) on GitHub Actions, then publishes a GitHub Release (tag b<N>)
# with all artifacts when run on `main`.
#
# Usage:
#   ./scripts/release.sh             # build + publish a release
#   ./scripts/release.sh --dry-run   # build artifacts only, do not publish
#   ./scripts/release.sh --watch     # poll until done, then print Release + assets
#   ./scripts/release.sh --help
#
set -euo pipefail

REPO="${REPO:-drzsdrtfg/audio.cpp}"
BRANCH="${BRANCH:-main}"
PUBLISH="true"
WATCH="false"

# ---------------------------------------------------------------- argparse
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)  PUBLISH="false";  shift ;;
        --watch)    WATCH="true";     shift ;;
        --repo)     REPO="$2";        shift 2 ;;
        --branch)   BRANCH="$2";      shift 2 ;;
        --help|-h)  sed -n '2,12p' "$0"; exit 0 ;;
        *) echo "unknown option: $1" >&2; sed -n '2,12p' "$0"; exit 1 ;;
    esac
done

# ------------------------------------------------------------- preconditions
command -v gh >/dev/null 2>&1 || { echo "gh CLI is required (https://cli.github.com)" >&2; exit 1; }

if ! gh auth status >/dev/null 2>&1; then
    echo "Not authenticated with gh — run 'gh auth login' first." >&2
    exit 1
fi

if [[ "$BRANCH" == "main" ]]; then
    CURRENT="$(git branch --show-current 2>/dev/null || true)"
    if [[ "$CURRENT" != "main" ]]; then
        echo "WARNING: not on main (current branch: ${CURRENT:-n/a})." >&2
        if [[ "$PUBLISH" == "true" ]]; then
            echo "Refusing to publish from a non-main branch. Use --dry-run to build only." >&2
            exit 1
        fi
    fi
    if [[ -n "$(git status --porcelain)" ]]; then
        echo "Working tree is dirty; commit or stash changes before releasing." >&2
        exit 1
    fi
fi

# ----------------------------------------------------------------- dispatch
echo "Dispatching Release workflow on ${REPO} @ ${BRANCH} (publish=${PUBLISH})"
if [[ "$PUBLISH" == "true" ]]; then
    gh workflow run release.yml --repo "$REPO" --ref "$BRANCH" -f publish=true
else
    gh workflow run release.yml --repo "$REPO" --ref "$BRANCH" -f publish=false
fi

echo "Waiting for the run to be registered..."
for _ in $(seq 1 30); do
    RUN_ID="$(gh run list --repo "$REPO" --workflow release.yml --limit 1 --json databaseId -q '.[0].databaseId' 2>/dev/null || true)"
    [[ -n "$RUN_ID" ]] && break
    sleep 2
done
RUN_ID="${RUN_ID:-$(gh run list --repo "$REPO" --workflow release.yml --limit 1 --json databaseId -q '.[0].databaseId')}"
echo "Workflow run: https://github.com/${REPO}/actions/runs/${RUN_ID}"

if [[ "$WATCH" != "true" ]]; then
    echo "Triggered. Monitor at the URL above (or rerun with --watch)."
    exit 0
fi

# ---------------------------------------------------------------------- watch
echo "Watching run ${RUN_ID} (this can take ~1-2 h while CUDA builds)..."
gh run watch "$RUN_ID" --repo "$REPO" --exit-status || {
    echo "Release workflow failed — see $RUN_ID" >&2
    exit 1
}

TAG="$(gh run view "$RUN_ID" --repo "$REPO" --json jobs -q '.jobs[] | select(.name=="get-version" and .conclusion=="success") | .steps[0].outputs' 2>/dev/null || true)"
echo
echo "================================================================"
echo " ✅ Release finished:"
echo "    https://github.com/${REPO}/actions/runs/${RUN_ID}"
if [[ "$PUBLISH" == "true" ]]; then
    RELEASE_URL="$(gh api "repos/${REPO}/releases" -q '.[0].html_url' 2>/dev/null || true)"
    echo "    GitHub release: ${RELEASE_URL:-check the run above}"
    gh release view --repo "$REPO" --json assets -q '.assets[] | "    - \(.name) (\((.size/1048576)|round|tostring) MB)"' 2>/dev/null || echo "    (no release assets found)"
fi
echo "================================================================"