# Adding Frontend Modules

Add a module when client-facing behavior can be adapted before or after the
core server handler without changing the stable core API.

Examples:

- Decode a new input audio format to core WAV input.
- Encode core WAV output to another client format.
- Normalize request text before TTS.
- Rewrite client-specific request fields into core request fields.

## Module Contract

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

The registry validates adjacent transforms for the same method/path at startup.
Use `frontend_contracts::any` only when the module deliberately accepts or
preserves any state.

## Steps

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
        "${AUDIOCPP_SERVER_FRONTENDS_SOURCE_DIR}/my_normalizer.cpp")
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
