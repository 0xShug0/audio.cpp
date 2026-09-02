# Server Frontends

Server frontends are optional in-process adapters around the stable server core.
The core route handlers keep their normal WAV/text request and response
contract. Frontends adapt client-facing behavior without adding format- or
client-specific code to the core server.

Frontend support is off by default:

```bash
cmake -S . -B build/debug -DCMAKE_BUILD_TYPE=Debug
```

Enable frontends and select the modules/capabilities explicitly:

```bash
cmake -S . -B build/debug \
  -DCMAKE_BUILD_TYPE=Debug \
  -DAUDIOCPP_BUILD_SERVER_FRONTENDS=ON \
  '-DAUDIOCPP_SERVER_FRONTEND_MODULES=audio_decode;mp3_encode'
```

`AUDIOCPP_SERVER_FRONTEND_MODULES` is a semicolon-separated ordered list. Only
the selected sources and their private dependencies are compiled and linked.

## Supported Frontends

| Name | Type | Purpose | Docs |
|---|---|---|---|
| `audio_decode` | pre-processing module | Accept MP3/FLAC ASR input and rewrite it to core WAV input | [audio_decode.md](docs/audio_decode.md) |
| `mp3_encode` | pre/post-processing module | Honor TTS `response_format=mp3` by encoding core WAV output to MP3 | [mp3_encode.md](docs/mp3_encode.md) |
| `https` | listener capability | Serve the same in-process server over HTTPS | [https.md](docs/https.md) |

For adding a new module, see [adding_modules.md](docs/adding_modules.md).

## Pipeline

For each HTTP request, the server runs:

```text
client request
  -> selected module pre_process(), in configured order
  -> core server request handler
  -> selected module post_process(), in configured order
  -> client response
```

A pre-processing module mutates `ServerFrontendRequest::request`. It can also
set `ServerFrontendRequest::response` to stop before the core handler and return
an explicit error or replacement response.

A post-processing module mutates `ServerFrontendResponse::response` after the
core handler returns. It also receives both the original client request and the
request that actually reached core.

`https` is not part of this pre/post pipeline. It is a frontend-owned listener
capability that terminates TLS and forwards the request envelope to the same
core handler.

## Contracts

Each active module side declares the HTTP envelope state it accepts and emits:

```cpp
struct FrontendPreContract {
    std::string_view method;
    std::string_view path;
    std::string_view request_in;
    std::string_view request_out;
};

struct FrontendPostContract {
    std::string_view method;
    std::string_view path;
    std::string_view response_in;
    std::string_view response_out;
};
```

The registry validates adjacent declared transforms for the same method/path.
If the previous module's output state does not match the next module's input
state, registration fails at server startup. Use `frontend_contracts::any` only
for modules that deliberately accept or preserve any state.

Current shared states are declared in `app/server/frontend.h`.
