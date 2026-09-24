# Yue2 iOS Demo

This branch is focused on the Yue2 iOS demo. The app links `AudioCpp.xcframework`, loads a bundled `Yue2-3B-GGUF` model folder, generates a short song from style, lyrics, and optional ABC notation, then plays the generated WAV in the UI with generation time, audio length, and RTF.

> [!WARNING]
> This branch is a demo for showing how far Yue2 VRAM usage can be pushed down on iOS. It is not intended to be a maintained product branch.
>
> The measured peak VRAM for the iOS demo path is about 1.7 GB on an iPhone 16 Pro. The iOS GGUF plus an FP32 VAE can also fit on device, but this demo uses `yue2-vae-f16.gguf` for the packaged path.
>
> You can use the iOS GGUF with the main Yue2 code path, but the generated quality will not match the normal desktop GGUF package.

## Build The XCFramework

Run from the repo root:

```sh
scripts/build_ios_xcframework.sh --clean \
  --build-root build/xcframework-yue2-ios \
  --output build/xcframework-yue2-ios/AudioCpp.xcframework \
  --models yue2
```

The demo project expects the framework here:

```text
build/xcframework-yue2-ios/AudioCpp.xcframework
```

## Download The Model

Download the Yue2 iOS GGUF package from the `audio-cpp/Yue2-3B-GGUF` Hugging Face repo.

The demo expects the model folder here:

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

These model files are intentionally not tracked in this repo. Download them from the GGUF repo and place them in the demo `Models` folder before opening the Xcode project.

Do not use the desktop `yue2-3b-q8_0.gguf` package for this iOS demo.

## Run The Demo

Open the Xcode project:

```sh
open examples/xcode/Yue2IOSDemo/Yue2IOSDemo.xcodeproj
```

In Xcode:

1. Select the `Yue2IOSDemo` scheme.
2. Select a real iOS device.
3. Set your signing team if Xcode asks for one.
4. Build and run.

The app opens with a ready-to-run official Yue2 demo prompt, lyrics, and ABC score. You can adjust CoT mode, inference steps, and semantic token budget before generation.

## Build The App From CLI

For a signing-free build check:

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
