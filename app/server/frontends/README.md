# Server Frontend Modules

Server frontends are optional in-process pre/post processors around the stable
server core. The core route handlers keep their normal WAV/text request and
response contract. Frontend modules adapt client-facing behavior without adding
format- or client-specific code to the core server.

Frontend support is off by default:

```bash
cmake -S . -B build/debug -DCMAKE_BUILD_TYPE=Debug
```

Enable the frontend pipeline and select modules explicitly:

```bash
cmake -S . -B build/debug \
  -DCMAKE_BUILD_TYPE=Debug \
  -DAUDIOCPP_BUILD_SERVER_FRONTENDS=ON \
  '-DAUDIOCPP_SERVER_FRONTEND_MODULES=audio_decode;mp3_encode'
```

`AUDIOCPP_SERVER_FRONTEND_MODULES` is a semicolon-separated ordered list. Only
the selected module sources and their private dependencies are compiled and
linked.

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

## Current Modules

`audio_decode` is a pre-processing module for ASR:

```text
POST /v1/audio/transcriptions
client_encoded_audio_request -> core_wav_audio_request
```

It accepts MP3 and FLAC JSON paths or multipart uploads, decodes them with
miniaudio, writes a temporary WAV, rewrites the request to a core-compatible JSON
request, then lets core run normally.

`mp3_encode` is a TTS output module:

```text
pre:
POST /v1/audio/speech
client_mp3_speech_request -> core_wav_speech_request

post:
POST /v1/audio/speech
core_wav_speech_response -> client_mp3_speech_response
```

It rewrites `response_format=mp3` to `response_format=wav` before core, then
encodes the core WAV response with libmp3lame and returns `audio/mpeg`.

## Adding A Module

1. Add a source file under `app/server/frontends/`, for example
   `my_normalizer.cpp`.

2. Implement `ServerFrontendModule`:

```cpp
class MyNormalizerModule final : public ServerFrontendModule {
public:
    std::string_view name() const override { return "my_normalizer"; }

    std::optional<FrontendPreContract> pre_contract() const override {
        return FrontendPreContract{
            "POST",
            "/v1/audio/speech",
            frontend_contracts::any,
            frontend_contracts::any,
        };
    }

    void pre_process(ServerFrontendContext &, ServerFrontendRequest & request) override {
        // Parse request.request.body, rewrite JSON fields, and assign request.request.
    }
};
```

3. Register the module factory from the source file:

```cpp
namespace {

std::unique_ptr<ServerFrontendModule> make_my_normalizer_module() {
    return std::make_unique<MyNormalizerModule>();
}

} // namespace

void register_my_normalizer_module(ServerFrontendRegistry & registry) {
    registry.add(make_my_normalizer_module);
}
```

4. Add a branch in the `AUDIOCPP_SERVER_FRONTEND_MODULES` loop in
   `app/server/frontends/server_frontends.cmake`:

```cmake
elseif (AUDIOCPP_SERVER_FRONTEND_MODULE STREQUAL "my_normalizer")
    string(APPEND AUDIOCPP_SERVER_FRONTEND_DECLARATIONS
        "void register_my_normalizer_module(ServerFrontendRegistry & registry);\n")
    string(APPEND AUDIOCPP_SERVER_FRONTEND_REGISTRATIONS
        "    register_my_normalizer_module(registry);\n")
    list(APPEND AUDIOCPP_SERVER_FRONTEND_SOURCES
        app/server/frontends/my_normalizer.cpp)
    list(APPEND AUDIOCPP_SERVER_FRONTEND_INCLUDE_DIRS
        /path/to/private/include)
    list(APPEND AUDIOCPP_SERVER_FRONTEND_LIBRARIES
        /path/to/private/library)
```

5. Build with the module name in the ordered module list:

```bash
cmake -S . -B build/debug \
  -DCMAKE_BUILD_TYPE=Debug \
  -DAUDIOCPP_BUILD_SERVER_FRONTENDS=ON \
  '-DAUDIOCPP_SERVER_FRONTEND_MODULES=my_normalizer;audio_decode;mp3_encode'
```

Keep module-specific dependencies in that CMake branch. Do not add them to the
default server target.
