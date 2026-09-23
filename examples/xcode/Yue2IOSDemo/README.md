# Yue2 iOS Demo

This demo builds a Yue2-only iOS app that links `AudioCpp.xcframework`, loads a bundled `Yue2-3B-GGUF` model folder, generates music from style plus lyrics, and plays the generated WAV.

## Build The XCFramework

```sh
scripts/build_ios_xcframework.sh --clean \
  --build-root build/xcframework-yue2-ios \
  --output build/xcframework-yue2-ios/AudioCpp.xcframework \
  --models yue2
```

The demo project expects the framework at:

```text
build/xcframework-yue2-ios/AudioCpp.xcframework
```

## Bundle The Model

Download the Yue2 iOS files from the `audio-cpp/Yue2-3B-GGUF` Hugging Face repo and place the model folder here:

```text
examples/xcode/Yue2IOSDemo/Models/Yue2-3B-GGUF
```

The folder must contain:

```text
Yue2-3B-GGUF/
  yue2-3b-ios-q4_0.gguf
  yue2-vae-f16.gguf
  sidecars/yue2-model-config.json
  sidecars/yue2-generation-config.json
  sidecars/yue2-qwen.tiktoken
  sidecars/yue2-vae-config.json
```

Do not bundle the desktop `yue2-3b-q8_0.gguf` in this iOS demo.

## Build The App

```sh
xcodebuild \
  -project examples/xcode/Yue2IOSDemo/Yue2IOSDemo.xcodeproj \
  -scheme Yue2IOSDemo \
  -configuration Release \
  -destination 'generic/platform=iOS' \
  -derivedDataPath examples/xcode/Yue2IOSDemo/build \
  CODE_SIGNING_ALLOWED=NO \
  build
```

To run on a device from Xcode, open the project, set a development team, keep the `Yue2-3B-GGUF` folder in the target resources, and select a physical iOS device.
