#include "engine/models/gigaam_asr/model.h"

#include "engine/framework/audio/chunking.h"
#include "engine/framework/audio/conversion.h"
#include "engine/framework/io/text.h"
#include "engine/framework/runtime/options.h"
#include "engine/framework/runtime/registry.h"
#include "engine/framework/runtime/session_base.h"
#include "engine/framework/runtime/spec_backed_model.h"

#include <algorithm>
#include <cmath>
#include <future>
#include <stdexcept>

namespace engine::models::gigaam_asr {
namespace {

class GigaAMSession final : public runtime::RuntimeSessionBase, public runtime::IOfflineVoiceTaskSession {
public:
    GigaAMSession(const runtime::TaskSpec & task, const runtime::SessionOptions & options,
                  std::shared_ptr<const GigaAMAssets> assets, std::shared_ptr<const model_spec::ModelContract> contract)
        : RuntimeSessionBase(options), assets_(std::move(assets)), contract_(std::move(contract)) {
        runtime::validate_spec_backed_session_options(options, *contract_, "gigaam_asr", "GigaAM ASR");
        if (task.task != runtime::VoiceTaskKind::Asr || task.mode != runtime::RunMode::Offline) {
            throw std::runtime_error("GigaAM requires an offline ASR session");
        }
        const auto type = assets::parse_tensor_storage_type(
            runtime::find_option(options.options, {"weight_type"}).value_or("native"));
        weights_ = load_gigaam_weights(*assets_, execution_context(), type);
        runtime_ = std::make_unique<GigaAMRuntime>(*assets_, *weights_, execution_context());
    }

    std::string family() const override { return "gigaam_asr"; }
    runtime::VoiceTaskKind task_kind() const override { return runtime::VoiceTaskKind::Asr; }
    runtime::RunMode run_mode() const override { return runtime::RunMode::Offline; }
    void prepare(const runtime::SessionPreparationRequest & request) override {
        runtime::validate_spec_backed_request_options(request.options, *contract_, "GigaAM ASR");
        mark_prepared();
    }
    runtime::TaskResult run(const runtime::TaskRequest & request) override {
        require_prepared("GigaAM run");
        runtime::validate_spec_backed_request_options(request.options, *contract_, "GigaAM ASR");
        if (!request.audio_input) {
            throw std::runtime_error("GigaAM requires audio input");
        }
        const auto start = std::chrono::steady_clock::now();
        const auto & audio_input = *request.audio_input;
        const bool timestamps = runtime::parse_bool_option(
            runtime::find_option(request.options, {"return_timestamps"}).value_or("false"), "return_timestamps");
        const auto mono = audio::convert_interleaved_audio_to_mono_linear_resampled(
            audio_input.samples, audio_input.sample_rate, audio_input.channels, 16000);
        const auto mode = audio::parse_audio_chunk_mode(request.options);
        const float seconds = audio::parse_audio_chunk_seconds_override(request.options).value_or(25.0f);
        if (!std::isfinite(seconds) || seconds < 0.02f || seconds > 25.0f) {
            throw std::runtime_error("GigaAM audio_chunk_duration_sec must be between 0.02 and 25");
        }
        const auto chunk_samples = static_cast<int64_t>(seconds * 16000);
        std::vector<runtime::TimeSpan> chunks;
        if (mode == audio::AudioChunkMode::Auto || mode == audio::AudioChunkMode::QuietEnergy) {
            chunks = audio::plan_quiet_energy_audio_chunks(
                mono, {chunk_samples, std::min<int64_t>(80000, chunk_samples / 2), 1600});
        } else if (mode == audio::AudioChunkMode::Vad) {
            const auto spans = audio::plan_vad_audio_chunks(audio_input, vad_session(),
                                                            {static_cast<int64_t>(seconds * audio_input.sample_rate),
                                                             audio_input.sample_rate / 2, audio_input.sample_rate / 4});
            const double ratio = 16000.0 / audio_input.sample_rate;
            for (const auto & span : spans) {
                chunks.push_back({std::llround(span.start_sample * ratio), std::llround(span.end_sample * ratio)});
            }
        } else if (mode == audio::AudioChunkMode::Fixed || mode == audio::AudioChunkMode::None) {
            const int64_t size =
                mode == audio::AudioChunkMode::None ? static_cast<int64_t>(mono.size()) : chunk_samples;
            for (const auto & c : audio::plan_audio_chunks(static_cast<int64_t>(mono.size()), {size, size})) {
                chunks.push_back({c.copy_start_sample, c.copy_start_sample + c.valid_samples});
            }
        } else {
            throw std::runtime_error("GigaAM supports auto, fixed, quiet_energy, vad, and none audio chunking");
        }
        // Keep a sub-window tail attached to its preceding chunk rather than discarding audio.
        if (chunks.size() > 1 && chunks.back().end_sample - chunks.back().start_sample < 320 &&
            chunks[chunks.size() - 2].end_sample == chunks.back().start_sample) {
            chunks[chunks.size() - 2].end_sample = chunks.back().end_sample;
            chunks.pop_back();
        }
        runtime::TaskResult result;
        result.text_output = runtime::Transcript{"", ""};
        debug::trace_log_scalar("gigaam_asr.audio_chunk_count", static_cast<int64_t>(chunks.size()));
        const auto extract_chunk = [&](size_t index) {
            const auto & span = chunks[index];
            const std::vector<float> samples(mono.begin() + span.start_sample, mono.begin() + span.end_sample);
            const auto frontend_start = std::chrono::steady_clock::now();
            auto features = assets_->frontend->extract(samples, static_cast<size_t>(options().backend.threads));
            return std::make_pair(std::move(features), debug::elapsed_ms(frontend_start));
        };
        std::future<std::pair<audio::AudioTensor, double>> pending;
        const bool prefetch = execution_context().backend_type() != core::BackendType::Cpu;
        double frontend_ms = 0.0;
        double encoder_ms = 0.0;
        double inference_ms = 0.0;
        for (size_t index = 0; index < chunks.size(); ++index) {
            const auto & span = chunks[index];
            auto prepared = pending.valid() ? pending.get() : extract_chunk(index);
            // Keep at most one next-chunk frontend result while the GPU runs.
            if (prefetch && index + 1 < chunks.size()) {
                pending = std::async(std::launch::async, extract_chunk, index + 1);
            }
            frontend_ms += prepared.second;
            const auto & features = prepared.first;
            const auto decoded = runtime_->transcribe(features, timestamps);
            encoder_ms += decoded.encoder_ms;
            inference_ms += decoded.inference_ms;
            const auto text = assets_->decode(decoded.ids);
            if (!result.text_output->text.empty() && !text.empty()) {
                result.text_output->text += ' ';
            }
            result.text_output->text += text;
            const double sample_ratio = static_cast<double>(audio_input.sample_rate) / 16000.0;
            if (chunks.size() > 1 || mode == audio::AudioChunkMode::Vad) {
                result.speech_segments.push_back(
                    {{std::llround(span.start_sample * sample_ratio), std::llround(span.end_sample * sample_ratio)},
                     0.0f,
                     text});
            }
            if (timestamps) {
                // Match upstream frames_to_words: token emission frames, not
                // forced alignment.
                std::string word;
                bool has_word_tokens = false;
                int64_t first_frame = 0, last_frame = 0;
                const double frame_samples =
                    static_cast<double>(span.end_sample - span.start_sample) / decoded.encoder_frames;
                const auto commit_word = [&] {
                    auto trimmed = io::trim_ascii_whitespace(word);
                    if (!trimmed.empty()) {
                        result.word_timestamps.push_back(
                            {{std::llround((span.start_sample + first_frame * frame_samples) * sample_ratio),
                              std::llround((span.start_sample + (last_frame + 1) * frame_samples) * sample_ratio)},
                             std::move(trimmed),
                             0.0f});
                    }
                    word.clear();
                    has_word_tokens = false;
                };
                for (size_t i = 0; i < decoded.ids.size(); ++i) {
                    const auto id = static_cast<size_t>(decoded.ids[i]);
                    auto piece = assets_->pieces.empty() ? assets_->characters.at(id) : assets_->pieces.at(id).text;
                    if (piece.compare(0, 3, "\xE2\x96\x81") == 0) {
                        commit_word();
                        piece.erase(0, 3);
                    } else if (piece == " ") {
                        commit_word();
                        continue;
                    }
                    if (!has_word_tokens)
                        first_frame = decoded.frames[i];
                    has_word_tokens = true;
                    word += piece;
                    last_frame = decoded.frames[i];
                }
                commit_word();
            }
        }
        debug::timing_log_scalar("gigaam_asr.frontend_ms", frontend_ms);
        debug::timing_log_scalar("gigaam_asr.encoder_ms", encoder_ms);
        debug::timing_log_scalar("gigaam_asr.inference_ms", inference_ms);
        debug::timing_log_scalar("session.wall_ms", debug::elapsed_ms(start));
        return result;
    }

private:
    runtime::IOfflineVoiceTaskSession & vad_session() {
        if (!vad_session_) {
            const auto path = runtime::find_option(options().options, {"vad_model_path"});
            if (!path || path->empty()) {
                throw std::runtime_error("GigaAM audio_chunk_mode=vad requires vad_model_path");
            }
            auto model = runtime::make_default_registry().load(std::filesystem::path(*path));
            auto session = model->create_task_session({runtime::VoiceTaskKind::Vad, runtime::RunMode::Offline},
                                                      {options().backend, {}});
            auto * offline = dynamic_cast<runtime::IOfflineVoiceTaskSession *>(session.get());
            if (!offline) {
                throw std::runtime_error("GigaAM VAD model must provide an offline VAD session");
            }
            vad_model_ = std::move(model);
            session.release();
            vad_session_.reset(offline);
        }
        return *vad_session_;
    }

    std::shared_ptr<const GigaAMAssets> assets_;
    std::shared_ptr<const model_spec::ModelContract> contract_;
    std::unique_ptr<GigaAMWeights> weights_;
    std::unique_ptr<GigaAMRuntime> runtime_;
    std::unique_ptr<runtime::ILoadedVoiceModel> vad_model_;
    std::unique_ptr<runtime::IOfflineVoiceTaskSession> vad_session_;
};

}  // namespace

std::shared_ptr<runtime::IVoiceModelLoader> make_gigaam_asr_loader() {
    runtime::SpecBackedVoiceModelConfig<GigaAMAssets> config;
    config.family = "gigaam_asr";
    config.load_assets = load_gigaam_assets;
    config.create_session = [](const runtime::TaskSpec & task, const runtime::SessionOptions & options,
                               std::shared_ptr<const GigaAMAssets> assets,
                               std::shared_ptr<const model_spec::ModelContract> contract) {
        return std::make_unique<GigaAMSession>(task, options, std::move(assets), std::move(contract));
    };
    return runtime::make_spec_backed_voice_loader(std::move(config));
}

}  // namespace engine::models::gigaam_asr
