#include "grouper.h"
#include "frontend.h"
#include "engine/framework/debug/profiler.h"
#include <algorithm>
#include <cmath>
#include <future>

namespace engine::models::dettato::detail {
struct WindowGrouper::Impl {
    HarmonicFrontend frontend;
    GrouperNetwork network;
    std::vector<float> prior;
    double window_seconds, hop;
    size_t maximum_notes;

    Impl(const assets::TensorSource & source, const io::json::Value & config, const core::ExecutionContext & execution)
        : frontend(source, "front.h", execution.config().threads), network(source, execution),
          prior(source.require_f32("hd.prior")), window_seconds(config.require("window_s").as_number()),
          hop(window_seconds * config.require("window_hop").as_number()),
          maximum_notes(static_cast<size_t>(config.require("max_notes").as_i64())) {
        if (!(window_seconds > 0) || !(hop > 0) || maximum_notes < 4 || maximum_notes > 768)
            throw std::invalid_argument("Dettato grouper requires positive windows/hops and 4..768 notes per window");
    }
};

WindowGrouper::WindowGrouper(const assets::TensorSource & source, const io::json::Value & config,
    const core::ExecutionContext & execution) : impl_(std::make_unique<Impl>(source, config, execution)) {}
WindowGrouper::~WindowGrouper() = default;

GroupingPrediction WindowGrouper::predict(const std::vector<float> & audio16, const GroupingFeatures & features) {
    if (features.notes.size() % 12 || features.taps.size() != features.notes.size() / 12 * 1104)
        throw std::invalid_argument("Dettato grouping requires aligned note and tap rows");
    const auto start = std::chrono::steady_clock::now();
    const size_t count = features.notes.size() / 12;
    auto & state = *impl_;
    GroupingPrediction out;
    out.window_seconds = state.window_seconds;
    out.embeddings.resize(count * 32); out.probabilities.resize(count * 73); out.hidden.resize(count * 176);
    if (count == 0) return out;
    std::vector<double> onsets(count);
    for (size_t i = 0; i < count; ++i) onsets[i] = features.notes[i * 12];
    if (!std::is_sorted(onsets.begin(), onsets.end())) throw std::invalid_argument("Dettato grouping notes must be sorted by onset");
    // Quantize once at the song boundary, matching _Song's PCM16 storage.
    std::vector<int16_t> pcm(audio16.size());
    for (size_t i = 0; i < pcm.size(); ++i) pcm[i] = static_cast<int16_t>(std::clamp(std::nearbyint(audio16[i] * 32768.f), -32768.f, 32767.f));
    const size_t samples = static_cast<size_t>(std::nearbyint((state.window_seconds + 1.5) * 16000));
    std::vector<float> notes, taps;
    double frontend_ms = 0;
    std::vector<double> mass(count * 73), embeddings(count * 32), weights(count), best(count, -1);
    const auto prepare_frontend = [&](double t0) {
        std::vector<float> audio(samples, 0);
        const int64_t first = static_cast<int64_t>(std::nearbyint((t0 - .5) * 16000));
        const int64_t lo = std::max<int64_t>(first, 0);
        const int64_t hi = std::min(first + static_cast<int64_t>(samples), static_cast<int64_t>(pcm.size()));
        for (int64_t i = lo; i < hi; ++i) audio[static_cast<size_t>(i - first)] = pcm[static_cast<size_t>(i)] / 32768.f;
        const auto frontend_start = std::chrono::steady_clock::now();
        auto front = state.frontend.extract(audio, 1, true);
        return std::make_pair(std::move(front), debug::elapsed_ms(frontend_start));
    };
    // One frontend job at a time; it owns its next-window feature arrays.
    // Current graph consumers read only the already-returned private features.
    std::future<std::pair<HarmonicFeatures, double>> next_features;
    for (double t0 = 0; ; ) {
        const size_t a = static_cast<size_t>(std::lower_bound(onsets.begin(), onsets.end(), t0) - onsets.begin());
        const size_t b = static_cast<size_t>(std::lower_bound(onsets.begin(), onsets.end(), t0 + state.window_seconds) - onsets.begin());
        const size_t end = std::min(b, a + state.maximum_notes);
        const bool done = b >= count && a + state.maximum_notes >= b;
        const double next_t0 = b - a > state.maximum_notes
            ? std::max(t0 + .25 * state.hop, onsets[a + state.maximum_notes / 2])
            : t0 + state.hop;
        if (end > a) {
            const size_t real = end - a, used = std::max<size_t>(real, 4);
            notes.resize(used * 12); taps.resize(used * 1104);
            for (size_t i = 0; i < used; ++i) {
                const size_t source = a + i % real;
                std::copy_n(features.notes.data() + source * 12, 12, notes.data() + i * 12);
                notes[i * 12] -= static_cast<float>(t0);
                notes[i * 12 + 1] = std::min(notes[i * 12 + 1] - static_cast<float>(t0), static_cast<float>(state.window_seconds + 1));
                for (size_t c = 0; c < 1104; ++c) taps[i * 1104 + c] = ggml_fp16_to_fp32(features.taps[source * 1104 + c]);
            }
            auto prepared = next_features.valid() ? next_features.get() : prepare_frontend(t0);
            // Avoid preparing an empty window. The next iteration may advance
            // across silence without ever invoking its neural graphs.
            const auto next_note = std::lower_bound(onsets.begin(), onsets.end(), next_t0);
            if (!done && next_t0 <= onsets.back() &&
                next_note != onsets.end() && *next_note < next_t0 + state.window_seconds)
                next_features = std::async(std::launch::async, prepare_frontend, next_t0);
            frontend_ms += prepared.second;
            const auto &front = prepared.first;
            const auto lanes = state.network.lane_features(front, notes);
            const auto input = make_grouper_inputs(front, notes, taps, state.prior, lanes);
            const auto output = state.network.run(input);
            GroupingWindow window; window.start = t0;
            window.embeddings.resize(real * 32); window.probabilities.resize(real * 73); window.notes.reserve(real);
            for (size_t i = 0; i < real; ++i) {
                const size_t id = a + i;
                window.notes.push_back(static_cast<int>(id));
                const double weight = std::clamp(1. - std::abs((onsets[id] - t0) / state.window_seconds - .5) * 2., .05, 1.);
                weights[id] += weight;
                for (size_t c = 0; c < 73; ++c) {
                    mass[id * 73 + c] += output[4][i * 73 + c] * weight;
                    window.probabilities[i * 73 + c] = ggml_fp16_to_fp32(ggml_fp32_to_fp16(output[4][i * 73 + c]));
                }
                for (size_t c = 0; c < 32; ++c) {
                    embeddings[id * 32 + c] += output[1][i * 32 + c] * weight;
                    window.embeddings[i * 32 + c] = ggml_fp16_to_fp32(ggml_fp32_to_fp16(output[1][i * 32 + c]));
                }
                if (weight > best[id]) {
                    best[id] = weight;
                    for (size_t c = 0; c < 176; ++c) out.hidden[id * 176 + c] = ggml_fp16_to_fp32(ggml_fp32_to_fp16(output[3][i * 176 + c]));
                }
            }
            out.windows.push_back(std::move(window));
        }
        if (done) break;
        t0 = next_t0;
        if (t0 > onsets.back()) break;
    }
    for (size_t i = 0; i < count; ++i) {
        double norm = 0;
        for (size_t c = 0; c < 32; ++c) norm += embeddings[i * 32 + c] * embeddings[i * 32 + c];
        norm = std::max(std::sqrt(norm), 1e-9);
        for (size_t c = 0; c < 32; ++c) out.embeddings[i * 32 + c] = static_cast<float>(embeddings[i * 32 + c] / norm);
        for (size_t c = 0; c < 73; ++c) out.probabilities[i * 73 + c] = static_cast<float>(mass[i * 73 + c] / std::max(weights[i], 1e-9));
    }
    debug::timing_log_scalar("dettato.grouper.frontend_ms", frontend_ms);
    debug::timing_log_scalar("dettato.grouper.windows", out.windows.size());
    debug::timing_log_scalar("dettato.grouper.wall_ms", debug::elapsed_ms(start));
    return out;
}
} // namespace engine::models::dettato::detail
