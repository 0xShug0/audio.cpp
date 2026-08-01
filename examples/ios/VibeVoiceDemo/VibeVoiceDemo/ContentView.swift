import AVFoundation
import SwiftUI

struct ContentView: View {
    private let presetText = """
    Speaker 1: Welcome everyone. Today we are testing VibeVoice running through audio.cpp on iOS.
    Speaker 2: The app uses four reference voices, generates a WAV file, and measures the runtime.
    Speaker 3: After generation finishes, we can play the result directly from the device.
    Speaker 4: The real time factor is generation time divided by the generated audio length.
    """
    private let randomSentenceBank = [
        "The morning deployment finished before the first support ticket arrived.",
        "Our monitoring dashboard shows stable traffic across every region.",
        "The design review focused on reducing friction in the checkout flow.",
        "A small cache change removed most of the repeated database reads.",
        "The onboarding team found three confusing steps in the setup guide.",
        "Customer feedback suggests the export button should stay visible.",
        "The release train is waiting for one final accessibility pass.",
        "We should compare the new audio output against yesterday's baseline.",
        "The battery test completed with better results than expected.",
        "A slower warmup is acceptable if the second run stays responsive.",
        "The translation model handled short sentences more consistently today.",
        "Our product metrics improved after simplifying the first screen.",
        "The warehouse scanner recovered cleanly after the network drop.",
        "A clearer error message would have saved the operator several minutes.",
        "The finance team needs the report before the afternoon planning call.",
        "The search ranking changed after we removed duplicate metadata.",
        "Every speaker should remain distinct during a longer conversation.",
        "The mobile client now retries failed uploads with exponential backoff.",
        "A single progress indicator is easier to understand than three counters.",
        "The QA run found no regressions in the payment workflow.",
        "We can ship the smaller model first and measure real user latency.",
        "The server stayed healthy while the batch import processed overnight.",
        "The legal review approved the updated data retention language.",
        "A compact transcript view makes long sessions easier to scan.",
        "The installer should verify disk space before copying the model.",
        "The new prompt format gives each speaker a predictable voice.",
        "A shorter reference sample reduces memory pressure on mobile devices.",
        "The field team asked for offline behavior to be tested more often.",
        "The dashboard should highlight failures before showing secondary metrics.",
        "The audio preview needs to start quickly after generation completes.",
        "We found that consistent punctuation improves the synthesized cadence.",
        "The backup service restored the missing configuration file.",
        "A well named preset helps users understand what they are testing.",
        "The app should keep the generated file available for sharing.",
        "The release notes need one paragraph about model bundling.",
        "The benchmark should report generation time and audio duration together.",
        "A deterministic seed makes repeated test runs easier to compare.",
        "The device log confirmed that Metal selected the expected GPU.",
        "The build script now produces the framework in the documented path.",
        "A clean sample project helps catch integration problems earlier.",
        "The customer success team wants clearer examples for multi speaker text.",
        "The latest prototype keeps the controls visible while scrolling.",
        "The database migration finished with no skipped records.",
        "A smaller prompt window gives the app more room for generation.",
        "The recording sounded natural after the noise reduction pass.",
        "The session cache should reuse the model after the first generation.",
        "The network trace showed a temporary delay in the authentication service.",
        "A separate play button makes the result easy to verify on device.",
        "The support article should explain how to unlock the phone during install.",
        "The team agreed to keep the demo focused on the real workflow.",
        "The color palette was adjusted to improve contrast in bright light.",
        "A shared test phrase helps compare speakers across builds.",
        "The import screen now reports invalid files before starting analysis.",
        "The memory entitlement must be present in the signed provisioning profile.",
        "The generated waveform should be saved before the player opens it.",
        "A longer dialogue is useful for checking speaker consistency.",
        "The scheduler completed all queued jobs before the maintenance window.",
        "The simulator build is useful, but the real device result matters most.",
        "The product manager asked for one tap random sample generation.",
        "A stable random sentence bank makes the demo feel less empty.",
        "The tokenizer should receive the same prompt shape for every speaker.",
        "The connector projection expects a consistent latent dimension.",
        "The app should not modify the original bundled voice files.",
        "A clear status line helps separate loading from generation.",
        "The preview file can be shared with another app for quick inspection.",
        "The first run may spend extra time compiling Metal pipelines.",
        "The second run should benefit from cached resources in the session.",
        "The speech turns should be merged when the same speaker continues.",
        "The test text should include short and medium length sentences.",
        "A natural pause between topics helps reveal audio pacing issues.",
        "The team wants the demo to work without any network connection.",
        "The model bundle size is expected because the weights ship with the app.",
        "The user interface should remain simple during long generation runs.",
        "A random sample is useful when validating repeated speaker changes.",
        "The app can show real time factor after the output length is known.",
        "The waveform player should stop before starting a new generation.",
        "The latest build reduced prompt memory without changing the WAV files.",
        "A device with more memory should still follow the same code path.",
        "The generated text should remain meaningful enough to evaluate prosody.",
        "The release candidate includes the same reference voices as the demo.",
        "A single bundled model keeps the example easy to install.",
        "The runtime logs are useful when comparing desktop and iOS behavior.",
        "The team should record the device model next to every benchmark result.",
        "The cache key must include prompt duration to avoid stale voice states.",
        "The app should recover gracefully if generation throws an exception.",
        "A consistent speaker label format helps the text tokenizer parse turns.",
        "The prompt should avoid filler so the audio remains easy to judge.",
        "The result file name should be unique for every generation.",
        "The audio length can vary even when the sentence count is fixed.",
        "A quick randomizer helps stress test the four speaker path.",
        "The model loader should keep using the VibeVoice family hint.",
        "The sharing control is helpful when comparing output on another machine.",
        "The text area should remain editable after random generation.",
        "The app should show failures without hiding the entered prompt.",
        "The benchmark result is more useful when RTF is shown with three digits.",
        "A short voice prompt keeps memory lower while preserving speaker identity.",
        "The team can use the same app to test future quantized models.",
        "The final audio should be playable immediately after the run completes.",
        "The demo should prefer real model behavior over artificial smoke tests.",
        "The next validation pass should run on the target iPhone.",
    ]

    @State private var text: String
    @State private var isGenerating = false
    @State private var status = "Ready"
    @State private var wavURL: URL?
    @State private var generationSeconds: Double = 0
    @State private var audioSeconds: Double = 0
    @State private var rtf: Double = 0
    @State private var inferenceSteps: Double = 8
    @State private var player: AVAudioPlayer?

    init() {
        _text = State(initialValue: presetText)
    }

    var body: some View {
        NavigationStack {
            VStack(spacing: 14) {
                TextEditor(text: $text)
                    .font(.body)
                    .autocorrectionDisabled()
                    .textInputAutocapitalization(.sentences)
                    .padding(10)
                    .frame(minHeight: 220)
                    .overlay(
                        RoundedRectangle(cornerRadius: 8)
                            .stroke(Color.secondary.opacity(0.35), lineWidth: 1)
                    )

                Button(action: generateRandomText) {
                    Label("Random Text", systemImage: "shuffle")
                        .frame(maxWidth: .infinity)
                }
                .buttonStyle(.bordered)
                .disabled(isGenerating)

                VStack(spacing: 6) {
                    HStack {
                        Text("Inference Steps")
                            .foregroundStyle(.secondary)
                        Spacer()
                        Text("\(Int(inferenceSteps))")
                            .monospacedDigit()
                    }
                    .font(.callout)

                    Slider(value: $inferenceSteps, in: 1...32, step: 1)
                        .disabled(isGenerating)
                }

                HStack(spacing: 12) {
                    Button(action: generate) {
                        Label(isGenerating ? "Generating" : "Generate", systemImage: "waveform")
                            .frame(maxWidth: .infinity)
                    }
                    .buttonStyle(.borderedProminent)
                    .disabled(isGenerating || text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)

                    Button(action: play) {
                        Label("Play", systemImage: "play.fill")
                            .frame(maxWidth: .infinity)
                    }
                    .buttonStyle(.bordered)
                    .disabled(wavURL == nil || isGenerating)
                }

                if let wavURL {
                    ShareLink(item: wavURL) {
                        Label(wavURL.lastPathComponent, systemImage: "square.and.arrow.up")
                            .lineLimit(1)
                            .frame(maxWidth: .infinity, alignment: .leading)
                    }
                    .font(.footnote)
                }

                VStack(spacing: 8) {
                    metricRow("Status", status)
                    metricRow("Generation", formatSeconds(generationSeconds))
                    metricRow("Audio Length", formatSeconds(audioSeconds))
                    metricRow("RTF", String(format: "%.3f", rtf))
                }
                .padding(.top, 4)

                Spacer(minLength: 0)
            }
            .padding()
            .navigationTitle("VibeVoice")
            .overlay {
                if isGenerating {
                    ProgressView()
                        .controlSize(.large)
                        .padding(18)
                        .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 8))
                }
            }
        }
    }

    private func metricRow(_ title: String, _ value: String) -> some View {
        HStack {
            Text(title)
                .foregroundStyle(.secondary)
            Spacer()
            Text(value)
                .monospacedDigit()
        }
        .font(.callout)
    }

    private func generate() {
        guard let modelURL = Bundle.main.url(
            forResource: "vibevoice-1.5b-q4_k-lookup_q8-convtr_q4",
            withExtension: "gguf"
        ) else {
            status = "Model resource missing"
            return
        }

        let voiceNames = [
            "en-Alice_woman",
            "en-Frank_man",
            "en-Carter_man",
            "en-Maya_woman",
        ]
        let voiceURLs = voiceNames.compactMap {
            Bundle.main.url(forResource: $0, withExtension: "wav")
        }
        guard voiceURLs.count == voiceNames.count else {
            status = "Voice resources missing"
            return
        }

        isGenerating = true
        status = "Generating"
        wavURL = nil
        player?.stop()
        player = nil

        VibeVoiceEngine.shared().generate(
            withText: text,
            modelURL: modelURL,
            voiceURLs: voiceURLs,
            inferenceSteps: Int(inferenceSteps)
        ) { result, error in
            isGenerating = false
            if let error {
                status = error.localizedDescription
                generationSeconds = 0
                audioSeconds = 0
                rtf = 0
                return
            }
            guard let result else {
                status = "Generation failed"
                return
            }
            wavURL = result.wavURL
            generationSeconds = result.generationSeconds
            audioSeconds = result.audioSeconds
            rtf = result.rtf
            status = "Done"
        }
    }

    private func generateRandomText() {
        let sentenceCount = Int.random(in: 4...50)
        let selectedSentences = randomSentenceBank.shuffled().prefix(sentenceCount)
        var speakers: [Int] = Array(1...4).shuffled()
        while speakers.count < sentenceCount {
            speakers.append(Int.random(in: 1...4))
        }

        var turns: [(speaker: Int, sentences: [String])] = []
        for (index, sentence) in selectedSentences.enumerated() {
            let speaker = speakers[index]
            if turns.last?.speaker == speaker {
                turns[turns.count - 1].sentences.append(sentence)
            } else {
                turns.append((speaker: speaker, sentences: [sentence]))
            }
        }

        text = turns
            .map { "Speaker \($0.speaker): \($0.sentences.joined(separator: " "))" }
            .joined(separator: "\n")
        status = "Ready"
        wavURL = nil
        generationSeconds = 0
        audioSeconds = 0
        rtf = 0
        player?.stop()
        player = nil
    }

    private func play() {
        guard let wavURL else {
            return
        }
        do {
            let audioSession = AVAudioSession.sharedInstance()
            try audioSession.setCategory(.playback, mode: .default)
            try audioSession.setActive(true)
            let nextPlayer = try AVAudioPlayer(contentsOf: wavURL)
            nextPlayer.prepareToPlay()
            nextPlayer.play()
            player = nextPlayer
        } catch {
            status = error.localizedDescription
        }
    }

    private func formatSeconds(_ value: Double) -> String {
        guard value > 0 else {
            return "0.00 s"
        }
        return String(format: "%.2f s", value)
    }
}
