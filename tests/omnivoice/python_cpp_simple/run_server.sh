#!/usr/bin/env bash
set -e

# Starts audiocpp_server with the OmniVoice config in this folder.
# Keep this terminal open.
#
# Run from anywhere. The script resolves the audio.cpp repository root.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

cd "${REPO_ROOT}"

DEVICE=0

CUDA_VISIBLE_DEVICES=$DEVICE build/bin/audiocpp_server \
    --config tests/omnivoice/python_cpp_simple/server.json
