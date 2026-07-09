#!/usr/bin/env bash
# ============================================================
# audio.cpp Docker multiplexer
#
# Dispatches to the correct binary based on the first
# argument. All remaining arguments are forwarded verbatim.
# ============================================================
set -e

arg1="$1"
shift || true

if [[ "$arg1" == "cli" ]]; then
    exec ./audiocpp_cli "$@"
elif [[ "$arg1" == "server" ]]; then
    exec ./audiocpp_server "$@"
elif [[ "$arg1" == "perf" ]]; then
    exec ./model_perf "$@"
elif [[ "$arg1" == "parity" ]]; then
    exec ./miocodec_wavlm_parity "$@"
else
    echo "Unknown command: $arg1"
    echo ""
    echo "Available commands:"
    echo "  cli     Run audio tasks (TTS, ASR, VAD, VC, diar, etc.)"
    echo "  server  Run the HTTP server"
    echo "  perf    Run model performance benchmarks"
    echo "  parity  Run Miocodec WavLM parity tests"
    exit 1
fi

