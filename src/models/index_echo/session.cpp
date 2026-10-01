#include "engine/models/index_echo/session.h"

#include "engine/framework/audio/chunking.h"
#include "engine/framework/audio/conversion.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/runtime/options.h"
#include "engine/framework/runtime/session_base.h"
#include "engine/framework/runtime/spec_backed_model.h"
#include "engine/models/index_echo/assets.h"
#include "engine/models/index_echo/text_translation.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <deque>
#include <iomanip>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <utility>
#include <vector>

namespace engine::models::index_echo {
namespace {

struct TranslationCue {
    int64_t start_centiseconds = 0;
    int64_t end_centiseconds = 0;
    std::string source;
    std::string translation;
};

std::vector<TranslationCue> parse_cues(const std::string & output, int64_t offset_centiseconds) {
    static const std::regex timestamp(R"(^\s*\[(\d+):(\d+(?:\.\d+)?)-(\d+):(\d+(?:\.\d+)?)\]\s*(.*)$)");
    std::istringstream lines(output);
    std::vector<TranslationCue> cues;
    std::string line;
    int pending = -1;
    while (std::getline(lines, line)) {
        if (line.empty()) {
            continue;
        }
        std::smatch match;
        if (std::regex_match(line, match, timestamp)) {
            TranslationCue cue;
            cue.start_centiseconds = offset_centiseconds + std::stoll(match[1].str()) * 6000 +
                                     static_cast<int64_t>(std::llround(std::stod(match[2].str()) * 100.0));
            cue.end_centiseconds = offset_centiseconds + std::stoll(match[3].str()) * 6000 +
                                   static_cast<int64_t>(std::llround(std::stod(match[4].str()) * 100.0));
            cue.source = match[5].str();
            cues.push_back(std::move(cue));
            pending = cues.back().source.empty() ? 0 : 1;
        } else if (!cues.empty() && pending == 0) {
            cues.back().source = std::move(line);
            pending = 1;
        } else if (!cues.empty() && pending == 1) {
            cues.back().translation = std::move(line);
            pending = 2;
        } else {
            throw std::runtime_error("Index-Echo-S2TT returned a malformed three-line subtitle cue");
        }
    }
    if (cues.empty() || std::any_of(cues.begin(), cues.end(), [](const TranslationCue & cue) {
            return cue.source.empty() || cue.translation.empty() || cue.end_centiseconds < cue.start_centiseconds;
        })) {
        throw std::runtime_error("Index-Echo-S2TT returned incomplete subtitle cues");
    }
    return cues;
}

std::string format_centiseconds(int64_t centiseconds) {
    std::ostringstream out;
    out << std::setfill('0') << std::setw(2) << centiseconds / 6000 << ':' << std::setw(2) << (centiseconds / 100) % 60
        << '.' << std::setw(2) << centiseconds % 100;
    return out.str();
}

class IndexEchoS2TTSession final : public runtime::RuntimeSessionBase, public runtime::IOfflineVoiceTaskSession {
public:
    IndexEchoS2TTSession(const runtime::TaskSpec & task, const runtime::SessionOptions & options,
                     std::shared_ptr<const IndexEchoAssets> assets,
                     std::shared_ptr<const model_spec::ModelContract> contract)
        : RuntimeSessionBase(options), task_(task), assets_(std::move(assets)), contract_(std::move(contract)) {
        runtime::validate_spec_backed_session_options(options, *contract_, "index_echo", "Index-Echo");
        if (task_.task != runtime::VoiceTaskKind::Asr || task_.mode != runtime::RunMode::Offline) {
            throw std::runtime_error("Index-Echo-S2TT requires an offline ASR session");
        }
        if (assets_->resources.has_file("mapper_config")) {
            throw std::runtime_error("Index-Echo-S2ST is not supported by this S2TT session");
        }
        const auto storage_type = assets::parse_tensor_storage_type(
            runtime::find_option(options.options, {"index_echo.weight_type"}).value_or("native"));
        text_runtime_ = std::make_unique<IndexEchoQwen3OmniAuTQwen35TranslationRuntime>(
            assets_, execution_context(), storage_type);
    }

    std::string family() const override { return "index_echo"; }
    runtime::VoiceTaskKind task_kind() const override { return task_.task; }
    runtime::RunMode run_mode() const override { return task_.mode; }

    void prepare(const runtime::SessionPreparationRequest & request) override {
        runtime::validate_spec_backed_request_options(request.options, *contract_, "Index-Echo");
        mark_prepared();
    }

    runtime::TaskResult run(const runtime::TaskRequest & request) override {
        const auto started = std::chrono::steady_clock::now();
        require_prepared("Index-Echo run");
        runtime::validate_spec_backed_request_options(request.options, *contract_, "Index-Echo");
        if (!request.audio_input.has_value()) {
            throw std::runtime_error("Index-Echo-S2TT requires audio input");
        }
        const auto language = runtime::find_option(request.options, {"target_language"}).value_or("en");
        const auto glossary = runtime::find_option(request.options, {"glossary"}).value_or("");
        const auto max_tokens = runtime::parse_positive_i64_option(request.options, {"max_text_tokens"}, 2000);
        const float temperature = runtime::parse_finite_float_option(request.options, {"temperature"}).value_or(0.0F);
        const int64_t seed = runtime::parse_i64_option(request.options, {"seed"}).value_or(0);
        if (temperature < 0.0F || temperature > 2.0F || seed < 0 || seed > INT32_MAX) {
            throw std::runtime_error("Index-Echo-S2TT requires temperature in [0, 2] and seed in [0, 2147483647]");
        }
        const auto & input = *request.audio_input;
        const auto mono = audio::convert_interleaved_audio_to_mono_linear_resampled(input.samples, input.sample_rate,
                                                                                    input.channels, 16000);
        const int64_t total_samples = static_cast<int64_t>(mono.size());
        const auto mode = audio::parse_audio_chunk_mode(request.options);
        const float seconds = audio::parse_audio_chunk_seconds_override(request.options).value_or(60.0F);
        if (!std::isfinite(seconds) || seconds < 1.0F || seconds > 60.0F) {
            throw std::runtime_error("Index-Echo-S2TT audio_chunk_duration_sec must be between 1 and 60");
        }
        const int64_t chunk_samples = static_cast<int64_t>(seconds * 16000);
        std::vector<runtime::TimeSpan> windows;
        if (mode == audio::AudioChunkMode::None || total_samples <= chunk_samples) {
            windows.push_back({0, total_samples});
        } else if (mode == audio::AudioChunkMode::Auto || mode == audio::AudioChunkMode::QuietEnergy) {
            windows = audio::plan_quiet_energy_audio_chunks(mono, {chunk_samples, 80000, 1600});
        } else if (mode == audio::AudioChunkMode::Fixed) {
            for (const auto & span : audio::plan_audio_chunks(total_samples, {chunk_samples, chunk_samples})) {
                windows.push_back({span.copy_start_sample, span.copy_start_sample + span.valid_samples});
            }
        } else {
            throw std::runtime_error("Index-Echo-S2TT supports auto, quiet_energy, "
                                     "fixed, and none chunking");
        }
        runtime::AudioBuffer normalized{16000, 1, mono};
        std::deque<std::string> history;
        std::string output;
        for (const auto & window : windows) {
            std::string context;
            for (const auto & entry : history) {
                if (!context.empty()) {
                    context += '\n';
                }
                context += entry;
            }
            const auto chunk = audio::slice_audio_buffer(normalized, window);
            const auto generated = text_runtime_->translate_window(
                chunk, language, context, glossary, max_tokens, temperature, static_cast<uint32_t>(seed));
            const auto cues = parse_cues(
                generated, static_cast<int64_t>(std::llround(static_cast<double>(window.start_sample) / 160.0)));
            std::string next_context;
            for (const auto & cue : cues) {
                if (!output.empty()) {
                    output += '\n';
                }
                output += '[' + format_centiseconds(cue.start_centiseconds) + '-' +
                          format_centiseconds(cue.end_centiseconds) + "]\n" + cue.source + '\n' + cue.translation;
                if (!next_context.empty()) {
                    next_context += '\n';
                }
                next_context += cue.source + '\n' + cue.translation;
            }
            history.push_back(std::move(next_context));
            if (history.size() > 5) {
                history.pop_front();
            }
        }
        runtime::TaskResult result;
        result.text_output = runtime::Transcript{std::move(output), language};
        debug::timing_log_scalar("session.wall_ms", debug::elapsed_ms(started));
        return result;
    }

  private:
    runtime::TaskSpec task_;
    std::shared_ptr<const IndexEchoAssets> assets_;
    std::shared_ptr<const model_spec::ModelContract> contract_;
    std::unique_ptr<IndexEchoQwen3OmniAuTQwen35TranslationRuntime> text_runtime_;
};

} // namespace

std::shared_ptr<runtime::IVoiceModelLoader> make_index_echo_loader() {
    runtime::SpecBackedVoiceModelConfig<IndexEchoAssets> config;
    config.family = "index_echo";
    config.load_assets = load_index_echo_assets;
    config.create_session = [](const runtime::TaskSpec & task, const runtime::SessionOptions & options,
                               std::shared_ptr<const IndexEchoAssets> assets,
                               std::shared_ptr<const model_spec::ModelContract> contract) {
        return std::make_unique<IndexEchoS2TTSession>(task, options, std::move(assets), std::move(contract));
    };
    return runtime::make_spec_backed_voice_loader(std::move(config));
}

} // namespace engine::models::index_echo
