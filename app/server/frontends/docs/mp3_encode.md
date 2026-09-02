# mp3_encode

`mp3_encode` is a TTS output module:

```text
pre:
POST /v1/audio/speech
client_mp3_speech_request -> core_wav_speech_request

post:
POST /v1/audio/speech
core_wav_speech_response -> client_mp3_speech_response
```

It rewrites `response_format=mp3` to `response_format=wav` before the core TTS
handler, then encodes the core WAV response with libmp3lame and returns
`audio/mpeg`.

## Build

```bash
cmake -S . -B build/debug \
  -DCMAKE_BUILD_TYPE=Debug \
  -DAUDIOCPP_BUILD_SERVER_FRONTENDS=ON \
  '-DAUDIOCPP_SERVER_FRONTEND_MODULES=mp3_encode'
```

Combine it with input decoding when both non-WAV input and MP3 output are
needed:

```bash
cmake -S . -B build/debug \
  -DCMAKE_BUILD_TYPE=Debug \
  -DAUDIOCPP_BUILD_SERVER_FRONTENDS=ON \
  '-DAUDIOCPP_SERVER_FRONTEND_MODULES=audio_decode;mp3_encode'
```

## Dependency

`mp3_encode` requires libmp3lame headers and library:

```text
lame/lame.h
libmp3lame
```

On Linux, install the development package from your distribution, for example:

```bash
sudo apt install libmp3lame-dev
```

On Windows with vcpkg:

```powershell
vcpkg install lame:x64-windows
cmake -S . -B build/debug `
  -DCMAKE_BUILD_TYPE=Debug `
  -DAUDIOCPP_BUILD_SERVER_FRONTENDS=ON `
  "-DAUDIOCPP_SERVER_FRONTEND_MODULES=mp3_encode" `
  -DAUDIOCPP_LAME_ROOT=C:/path/to/vcpkg/installed/x64-windows
```

On Windows or Linux with conda:

```bash
conda install -c conda-forge lame
cmake -S . -B build/debug \
  -DCMAKE_BUILD_TYPE=Debug \
  -DAUDIOCPP_BUILD_SERVER_FRONTENDS=ON \
  '-DAUDIOCPP_SERVER_FRONTEND_MODULES=mp3_encode'
```

If CMake still cannot find LAME, pass the install prefix explicitly:

```bash
cmake -S . -B build/debug \
  -DCMAKE_BUILD_TYPE=Debug \
  -DAUDIOCPP_BUILD_SERVER_FRONTENDS=ON \
  '-DAUDIOCPP_SERVER_FRONTEND_MODULES=mp3_encode' \
  -DAUDIOCPP_LAME_ROOT=/path/to/lame/prefix
```

## Request

```bash
curl http://127.0.0.1:8080/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"model":"tts","voice":"default","input":"hello","response_format":"mp3"}' \
  -o out.mp3
```
