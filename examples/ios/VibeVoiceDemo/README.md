# VibeVoice iOS Demo

This sample app links the `AudioCpp.xcframework` iOS arm64 slice and runs VibeVoice TTS on Metal.

Build the framework first:

```sh
scripts/build_xcframework.sh --platform ios --archs arm64 --deployment-target 16.3 --build-root build/xcframework-vibevoice-ios --output build/xcframework-vibevoice-ios/AudioCpp.xcframework --model-set custom --models vibevoice --native-cpu OFF --openmp OFF -j 8
```

Open `examples/ios/VibeVoiceDemo/VibeVoiceDemo.xcodeproj` in Xcode and build the `VibeVoiceDemo` scheme for a real iOS device. The project references local demo paths:

- `examples/ios/VibeVoiceDemo/Frameworks/AudioCpp.xcframework`
- `examples/ios/VibeVoiceDemo/Resources/vibevoice-1.5b-q4_k-lookup_q8-convtr_q4.gguf`
- `examples/ios/VibeVoiceDemo/Resources/en-Alice_woman.wav`
- `examples/ios/VibeVoiceDemo/Resources/en-Frank_man.wav`
- `examples/ios/VibeVoiceDemo/Resources/en-Carter_man.wav`
- `examples/ios/VibeVoiceDemo/Resources/en-Maya_woman.wav`

The app writes generated WAV files into the temporary directory, plays them with `AVAudioPlayer`, and reports generation time, audio duration, and RTF.
