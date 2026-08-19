# OmniVoice from Python over HTTP

Drives OmniVoice from Python while inference stays in C++, by sending
OpenAI-compatible requests to `audiocpp_server`. The client uses only the Python
standard library.

## Build

```bash
cmake -S . -B build -DENGINE_ENABLE_CUDA=ON
cmake --build build --parallel --target audiocpp_cli audiocpp_server
```

## Direct C++ baseline

```bash
./tests/omnivoice/python_cpp_simple/run_cpp.sh
```

Writes `tests/omnivoice/outputs/python_cpp_simple/cpp_output.wav`.

## Through the server

Terminal 1, and keep it open until the server reports it is ready:

```bash
./tests/omnivoice/python_cpp_simple/run_server.sh
```

Terminal 2:

```bash
python3 tests/omnivoice/python_cpp_simple/test_python.py
```

Writes `tests/omnivoice/outputs/python_cpp_simple/python_output.wav`.

## Files

```text
run_cpp.sh       audiocpp_cli baseline
run_server.sh    starts audiocpp_server with server.json
server.json      server config, paths are relative to this file
test_python.py   POST /v1/audio/speech client
```

Change weight types for the C++ baseline in `run_cpp.sh`, and for the server
path in the `session_options` block of `server.json` followed by a server
restart.
