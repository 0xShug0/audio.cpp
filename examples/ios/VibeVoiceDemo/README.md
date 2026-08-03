# VibeVoice iOS Demo

## Build the XCFramework

Run from the repo root:

```sh
bash scripts/build_xcframework.sh \
  --platform ios \
  --archs arm64 \
  --deployment-target <ios-deployment-target> \
  --build-root <xcframework-build-dir> \
  --output <path-to-AudioCpp.xcframework> \
  --model-set custom \
  --models vibevoice \
  --native-cpu OFF \
  --openmp OFF \
  -j 8
```

The demo expects the framework here:

```text
examples/ios/VibeVoiceDemo/Frameworks/AudioCpp.xcframework
```

## Download the Model

Download `VibeVoice-1.5B-GGUF/vibevoice-1.5b-q4-ios.gguf` from the `audio-cpp/audio.cpp-gguf` Hugging Face repo.

The demo expects the model here:

```text
examples/ios/VibeVoiceDemo/Resources/vibevoice-1.5b-q4-ios.gguf
```

## Generate Voice State Files

Build or locate a host `audiocpp_cli` binary, then run:

```sh
<path-to-audiocpp_cli> \
  --task tts \
  --family vibevoice \
  --model <path-to-VibeVoiceDemo-Resources>/vibevoice-1.5b-q4-ios.gguf \
  --backend metal \
  --threads 8 \
  --request-option voice_samples=<speaker1.wav>,<speaker2.wav>,<speaker3.wav>,<speaker4.wav> \
  --request-option voice_prompt_max_seconds=5 \
  --request-option voice_state_out_dir=<path-to-VibeVoiceDemo-Resources>
```

`voice_state_out_dir` makes the CLI export `.vvstate` files during `prepare()` and exit without generating audio.
Each output filename comes from the input WAV stem, so `en-Alice_woman.wav` becomes `en-Alice_woman.vvstate`.

The demo expects these voice-state files:

```text
examples/ios/VibeVoiceDemo/Resources/en-Alice_woman.vvstate
examples/ios/VibeVoiceDemo/Resources/en-Carter_man.vvstate
examples/ios/VibeVoiceDemo/Resources/en-Frank_man.vvstate
examples/ios/VibeVoiceDemo/Resources/en-Maya_woman.vvstate
```

These files are bundled so the iOS app can load fixed speaker state directly. That avoids decoding the reference WAVs and running the VibeVoice acoustic encoder on device at launch or before every generation, which reduces startup work, memory pressure, and first-generation latency.

## Run the Demo

Open the Xcode project:

```sh
open <path-to-VibeVoiceDemo.xcodeproj>
```

In Xcode:

1. Select the `VibeVoiceDemo` scheme.
2. Select a real iOS device.
3. Set your signing team if Xcode asks for one.
4. Build and run.

The app generates a WAV file, plays it in the UI, and shows generation time, audio length, and RTF.
