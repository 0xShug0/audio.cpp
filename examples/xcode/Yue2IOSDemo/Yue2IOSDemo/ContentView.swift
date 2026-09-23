import AVFoundation
import SwiftUI

final class Yue2DemoModel: NSObject, ObservableObject, AVAudioPlayerDelegate {
    @Published var style = Yue2DemoModel.loadDefaultText("style", extension: "txt")
    @Published var lyrics = Yue2DemoModel.loadDefaultText("lyrics", extension: "txt")
    @Published var abc = Yue2DemoModel.loadDefaultText("score", extension: "abc")
    @Published var cotMode = "full"
    @Published var inferenceSteps = 8.0
    @Published var semanticMaxTokens = 1792.0
    @Published var status = "Ready"
    @Published var isGenerating = false
    @Published var outputURL: URL?
    @Published var generationSeconds = 0.0
    @Published var audioSeconds = 0.0
    @Published var rtf = 0.0
    @Published var isPlaying = false
    @Published var playProgress = 0.0

    private var bridge: Yue2DemoBridge?
    private var player: AVAudioPlayer?
    private var progressTimer: Timer?

    func generate() {
        guard !isGenerating else { return }
        isGenerating = true
        status = "Generating..."
        stopPlayback()

        let style = style
        let lyrics = lyrics
        let abc = abc
        let cotMode = cotMode
        let steps = Int64(inferenceSteps.rounded())
        let maxTokens = Int64(semanticMaxTokens.rounded())
        let seed = UInt64.random(in: 1..<(UInt64.max / 2))

        DispatchQueue.global(qos: .userInitiated).async {
            do {
                let modelPath = try Self.bundledModelPath()
                let output = Self.outputURL().path
                let runner = try self.runner(modelPath: modelPath)
                let result = try runner.generate(
                    withStyle: style,
                    lyrics: lyrics,
                    abc: abc,
                    cotMode: cotMode,
                    outputPath: output,
                    seed: seed,
                    inferenceSteps: steps,
                    semanticMaxTokens: maxTokens
                )
                DispatchQueue.main.async {
                    self.outputURL = URL(fileURLWithPath: result.outputPath)
                    self.generationSeconds = result.generationSeconds
                    self.audioSeconds = result.audioSeconds
                    self.rtf = result.rtf
                    self.status = "Generated seed \(seed)"
                    self.isGenerating = false
                    self.preparePlayer()
                }
            } catch {
                DispatchQueue.main.async {
                    self.status = error.localizedDescription
                    self.isGenerating = false
                }
            }
        }
    }

    func togglePlayback() {
        guard let outputURL else { return }
        if player == nil || player?.url != outputURL {
            preparePlayer()
        }
        guard let player else { return }
        if player.isPlaying {
            player.pause()
            isPlaying = false
            progressTimer?.invalidate()
        } else {
            do {
                try configureAudioSession()
                if player.play() {
                    isPlaying = true
                    startProgressTimer()
                } else {
                    status = "Failed to start WAV playback"
                }
            } catch {
                status = error.localizedDescription
            }
        }
    }

    func seek(to fraction: Double) {
        guard let player else { return }
        player.currentTime = max(0.0, min(1.0, fraction)) * player.duration
        playProgress = fraction
    }

    func audioPlayerDidFinishPlaying(_ player: AVAudioPlayer, successfully flag: Bool) {
        DispatchQueue.main.async {
            self.isPlaying = false
            self.playProgress = 1.0
            self.progressTimer?.invalidate()
        }
    }

    private func runner(modelPath: String) throws -> Yue2DemoBridge {
        if let bridge {
            return bridge
        }
        let created = try Yue2DemoBridge(
            modelPath: modelPath,
            backend: "metal",
            device: 0,
            threads: 8
        )
        bridge = created
        return created
    }

    private func preparePlayer() {
        guard let outputURL else { return }
        do {
            try configureAudioSession()
            player = try AVAudioPlayer(contentsOf: outputURL)
            player?.delegate = self
            player?.prepareToPlay()
            playProgress = 0.0
        } catch {
            status = error.localizedDescription
        }
    }

    private func configureAudioSession() throws {
        let session = AVAudioSession.sharedInstance()
        try session.setCategory(.playback, mode: .default)
        try session.setActive(true)
    }

    private func stopPlayback() {
        player?.stop()
        player = nil
        isPlaying = false
        playProgress = 0.0
        progressTimer?.invalidate()
    }

    private func startProgressTimer() {
        progressTimer?.invalidate()
        progressTimer = Timer.scheduledTimer(withTimeInterval: 0.1, repeats: true) { [weak self] _ in
            guard let self, let player = self.player, player.duration > 0 else { return }
            self.playProgress = player.currentTime / player.duration
            self.isPlaying = player.isPlaying
        }
    }

    private static func bundledModelPath() throws -> String {
        if let url = Bundle.main.url(forResource: "Yue2-3B-GGUF", withExtension: nil) {
            return url.path
        }
        let fallback = Bundle.main.bundleURL.appendingPathComponent("Yue2-3B-GGUF")
        if FileManager.default.fileExists(atPath: fallback.path) {
            return fallback.path
        }
        throw NSError(
            domain: "Yue2IOSDemo",
            code: 1,
            userInfo: [NSLocalizedDescriptionKey: "Bundle the Yue2-3B-GGUF model folder in the app resources."]
        )
    }

    private static func outputURL() -> URL {
        let directory = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!
        return directory.appendingPathComponent("yue2-output-\(Date().timeIntervalSince1970).wav")
    }

    private static func loadDefaultText(_ name: String, extension ext: String) -> String {
        guard let url = Bundle.main.url(
            forResource: name,
            withExtension: ext,
            subdirectory: "OfficialJingleBellsDemo"
        ) else {
            return ""
        }
        return (try? String(contentsOf: url, encoding: .utf8)) ?? ""
    }
}

struct ContentView: View {
    @StateObject private var model = Yue2DemoModel()

    var body: some View {
        NavigationStack {
            VStack(spacing: 0) {
                ScrollView {
                    VStack(spacing: 14) {
                        editor(title: "Style", text: $model.style, minHeight: 92)
                        editor(title: "Lyrics", text: $model.lyrics, minHeight: 150)
                        editor(title: "ABC", text: $model.abc, minHeight: 180)

                        VStack(spacing: 10) {
                            controlPicker(title: "CoT mode", selection: $model.cotMode) {
                                Text("Off").tag("off")
                                Text("Melody").tag("melody")
                                Text("Full").tag("full")
                            }
                            controlRow(title: "Steps", value: Int(model.inferenceSteps)) {
                                Slider(value: $model.inferenceSteps, in: 1...16, step: 1)
                            }
                            controlRow(title: "Semantic tokens", value: Int(model.semanticMaxTokens)) {
                                Slider(value: $model.semanticMaxTokens, in: 384...4096, step: 128)
                            }
                        }

                        Button {
                            model.generate()
                        } label: {
                            HStack {
                                if model.isGenerating {
                                    ProgressView()
                                        .tint(.black)
                                }
                                Text(model.isGenerating ? "Generating" : "Generate")
                                    .fontWeight(.semibold)
                            }
                            .frame(maxWidth: .infinity)
                        }
                        .buttonStyle(.borderedProminent)
                        .disabled(model.isGenerating)
                    }
                    .padding()
                }
                playerPanel
            }
            .navigationTitle("Yue2 Music")
            .toolbarBackground(.visible, for: .navigationBar)
        }
    }

    private var playerPanel: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack(alignment: .firstTextBaseline) {
                Text(model.status)
                    .lineLimit(2)
                Spacer()
                if let outputURL = model.outputURL {
                    ShareLink(item: outputURL) {
                        Label("WAV", systemImage: "square.and.arrow.up")
                    }
                }
            }
            Text(metricsText)
                .font(.system(.caption, design: .monospaced))
                .foregroundStyle(.secondary)
            HStack {
                Button {
                    model.togglePlayback()
                } label: {
                    Label(model.isPlaying ? "Pause" : "Play", systemImage: model.isPlaying ? "pause.fill" : "play.fill")
                }
                .disabled(model.outputURL == nil)
                Slider(
                    value: Binding(
                        get: { model.playProgress },
                        set: { model.seek(to: $0) }
                    ),
                    in: 0...1
                )
                .disabled(model.outputURL == nil)
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(12)
        .background(Color(.secondarySystemBackground))
    }

    private var metricsText: String {
        guard model.generationSeconds > 0 else {
            return "generation: --  audio: --  rtf: --"
        }
        return String(
            format: "generation: %.2fs  audio: %.2fs  rtf: %.3f",
            model.generationSeconds,
            model.audioSeconds,
            model.rtf
        )
    }

    private func editor(title: String, text: Binding<String>, minHeight: CGFloat) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            Text(title)
                .font(.caption)
                .foregroundStyle(.secondary)
            TextEditor(text: text)
                .font(.system(.body, design: .monospaced))
                .frame(minHeight: minHeight)
                .padding(6)
                .background(Color(.secondarySystemBackground))
                .clipShape(RoundedRectangle(cornerRadius: 8))
        }
    }

    private func controlRow<Content: View>(
        title: String,
        value: Int,
        @ViewBuilder content: () -> Content
    ) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack {
                Text(title)
                Spacer()
                Text("\(value)")
                    .font(.system(.caption, design: .monospaced))
                    .foregroundStyle(.secondary)
            }
            content()
        }
    }

    private func controlPicker<Content: View>(
        title: String,
        selection: Binding<String>,
        @ViewBuilder content: () -> Content
    ) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(title)
            Picker(title, selection: selection) {
                content()
            }
            .pickerStyle(.segmented)
        }
    }
}
