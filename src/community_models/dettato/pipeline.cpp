#include "engine/community_models/dettato/pipeline.h"
#include "engine/framework/audio/resampling.h"
#include "engine/framework/debug/profiler.h"
#include "engine/community_models/dettato/assets.h"
#include "engine/community_models/dettato/drums.h"
#include "engine/community_models/dettato/frontend.h"
#include "engine/community_models/dettato/grouper.h"
#include "engine/community_models/dettato/note_repair.h"
#include "engine/community_models/dettato/transcriber.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <future>

namespace engine::community_models::dettato {
struct DettatoPipeline::Impl {
    std::shared_ptr<const DettatoAssets> assets;
    const size_t threads;
    detail::TuningEstimator tuning;
    detail::WindowTranscriber transcriber;
    detail::NoteRepairNetwork note_repair;
    detail::WindowGrouper grouper;
    std::unique_ptr<core::ExecutionContext> drum_execution;
    detail::DrumFrontend drum_frontend;
    detail::DrumTranscriber drums;
    detail::NoteThresholds thresholds;

    Impl(std::shared_ptr<const DettatoAssets> a, const core::ExecutionContext &execution, int window_hop_sec)
        : assets(std::move(a)), threads(execution.config().threads), tuning(execution.config().threads),
          transcriber(detail::bind_transcriber_weights(*assets->transcriber, execution), execution, window_hop_sec),
          note_repair(detail::bind_note_repair_weights(*assets->note_repair, execution), execution),
          grouper(detail::bind_grouper_weights(*assets->grouper, execution), assets->config.require("grouper"),
                  execution),
          drum_execution(execution.backend_type() == core::BackendType::Cpu
                             ? nullptr
                             : std::make_unique<core::ExecutionContext>(
                                   core::BackendConfig{execution.backend_type(), execution.config().device,
                                                       execution.config().threads, execution.config().impl})),
          drum_frontend(detail::load_drum_frontend(*assets->drums), execution.config().threads),
          drums(detail::bind_drum_weights(*assets->drums, assets->config.require("drums"),
                                          drum_execution ? *drum_execution : execution),
                assets->config.require("drums"), drum_execution ? *drum_execution : execution) {
        const auto &t = assets->config.require("transcriber").require("thresholds");
        thresholds.onset = t.require("on_th").as_number();
        thresholds.offset = t.require("off_th").as_number();
        thresholds.frame = t.require("fr_th").as_number();
        thresholds.second_onset = t.require("v2_th").as_number();
        thresholds.low_frames = t.require("fr_low_frames").as_i64();
        thresholds.onset_nms = t.require("on_nms").as_i64();
    }
};
DettatoPipeline::DettatoPipeline(std::shared_ptr<const DettatoAssets> assets, const core::ExecutionContext &execution,
                                 int window_hop_sec)
    : impl_(std::make_unique<Impl>(std::move(assets), execution, window_hop_sec)) {}
DettatoPipeline::~DettatoPipeline() = default;

Transcription DettatoPipeline::transcribe(const std::vector<float> &mono, int sample_rate) {
    if (sample_rate <= 0 || mono.empty() ||
        std::any_of(mono.begin(), mono.end(), [](float x) { return !std::isfinite(x); }))
        throw std::invalid_argument("Dettato requires non-empty finite mono audio and a positive sample rate");
    auto &s = *impl_;
    const auto start = std::chrono::steady_clock::now();
    audio::SoxrResampleOptions options;
    options.require_full_input = true;
    options.reject_empty_output = true;
    options.warning_context = "Dettato";
    options.fallback_description = "linear resampling";
    const auto prepare_drums = [&] {
        const auto frontend_start = std::chrono::steady_clock::now();
        auto audio44k = audio::resample_mono_soxr_or_linear(mono, sample_rate, 44100, options);
        auto features = audio44k.size() >= 4410 ? s.drum_frontend.extract(audio44k) : std::vector<float>{};
        debug::timing_log_scalar("dettato.drums.frontend_ms", debug::elapsed_ms(frontend_start));
        return features;
    };
    std::future<std::vector<float>> drum_features;
    const auto run_drums = [&] {
        const auto drum_start = std::chrono::steady_clock::now();
        auto features = drum_features.valid() ? drum_features.get() : prepare_drums();
        auto result = s.drums.transcribe_features(features);
        debug::timing_log_scalar("dettato.drums_ms", debug::elapsed_ms(drum_start));
        return result;
    };
    // Independent original-rate input and private backend/graphs. The future
    // also drains the worker if pitched transcription throws before get().
    std::future<detail::DrumResult> drum_request;
    auto pitched = audio::resample_mono_soxr_or_linear(mono, sample_rate, 16000, options);
    auto tuning = s.tuning.estimate(pitched);
    if (tuning.ratio != 1) {
        // Resampling depends on the rate ratio. Express it using the finite
        // input/output sample counts, avoiding integer-Hz rounding of 16000/r.
        const auto samples = static_cast<int>(pitched.size());
        const auto target = static_cast<int>(std::llround(samples / tuning.ratio));
        pitched = audio::resample_mono_soxr_or_linear(pitched, samples, target, options);
        tuning.ratio = static_cast<double>(samples) / target;
    }
    debug::timing_log_scalar("dettato.tuning_ms", debug::elapsed_ms(start));
    Transcription result;
    result.tuning_cents = tuning.cents;
    // CPU-only drum preprocessing is independent of pitched transcription.
    // The private frontend is used by at most one job; neural drum execution
    // still starts after the transcriber GPU work to avoid device contention.
    if (sample_rate >= 40000 && s.drum_execution)
        drum_features = std::async(std::launch::async, prepare_drums);
    auto phase = std::chrono::steady_clock::now();
    s.transcriber.run_device(pitched, false);
    // Neural device work is complete. Drum inference can now overlap readback,
    // evidence and grouping without contending with the transcriber graph.
    if (sample_rate >= 40000 && s.drum_execution)
        drum_request = std::async(std::launch::async, run_drums);
    auto prediction = s.transcriber.readback();
    auto notes = detail::decode_notes(prediction.maps, prediction.frames, .02, s.thresholds);
    result.transcribed = notes.size();
    debug::timing_log_scalar("dettato.transcriber_ms", debug::elapsed_ms(phase));
    phase = std::chrono::steady_clock::now();
    const auto &cfg = s.assets->config.require("evidence");
    auto repaired =
        detail::repair_notes(s.note_repair, notes, detail::extract_note_evidence(prediction, notes, s.threads),
                             cfg.require("keep_th").as_number(), cfg.require("insert_th").as_number());
    result.dropped = repaired.dropped;
    result.inserted = repaired.inserted;
    debug::timing_log_scalar("dettato.evidence_ms", debug::elapsed_ms(phase));
    phase = std::chrono::steady_clock::now();
    for (auto &note : repaired.notes)
        note.offset = std::max(note.offset, note.onset + .03);
    s.transcriber.readback_grouping(prediction, repaired.notes);
    result.tracks = s.grouper.group(pitched, prediction, repaired.notes, s.thresholds, tuning.ratio);
    debug::timing_log_scalar("dettato.grouping_ms", debug::elapsed_ms(phase));
    if (sample_rate >= 40000) {
        auto drum_result = drum_request.valid() ? drum_request.get() : run_drums();
        result.drum_router = drum_result.router;
        result.drum_model = std::move(drum_result.model);
        if (!drum_result.track.notes.empty())
            result.tracks.push_back(std::move(drum_result.track));
    }
    debug::timing_log_scalar("dettato.transcribe_ms", debug::elapsed_ms(start));
    return result;
}
} // namespace engine::community_models::dettato
