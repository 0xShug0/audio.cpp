#include "engine/community_models/piper_tts/session.h"

#include "engine/framework/debug/profiler.h"
#include "engine/framework/runtime/options.h"
#include "engine/framework/runtime/spec_backed_model.h"
#include "engine/framework/text/chunking.h"
#include "engine/framework/text/utf8.h"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <utility>

namespace engine::models::piper_tts {
namespace {

constexpr const char * kFamily = "piper_tts";

std::shared_ptr<const PiperTtsAssets> require_assets(
    std::shared_ptr<const PiperTtsAssets> assets) {
    if (assets == nullptr) {
        throw std::runtime_error("Piper TTS session requires assets");
    }
    return assets;
}

std::shared_ptr<const engine::model_spec::ModelContract> require_contract(
    std::shared_ptr<const engine::model_spec::ModelContract> contract) {
    if (contract == nullptr) {
        throw std::runtime_error("Piper TTS session requires a model contract");
    }
    return contract;
}

std::filesystem::path session_path(
    const runtime::SessionOptions & options,
    const char * key) {
    const auto found = options.options.find(key);
    return found == options.options.end()
        ? std::filesystem::path{}
        : std::filesystem::path(found->second);
}

void validate_session_options(
    const runtime::SessionOptions & options,
    const engine::model_spec::ModelContract & contract) {
    const std::string family_prefix = std::string(kFamily) + ".";
    for (const auto & [key, _] : options.options) {
        if (key.rfind(family_prefix, 0) == 0 &&
            contract.session_option_keys.find(key) ==
                contract.session_option_keys.end()) {
            throw std::runtime_error(
                "unknown Piper TTS session option: " + key);
        }
    }
}

int64_t chunk_size_from_request(const runtime::TaskRequest & request) {
    const auto value = runtime::parse_i64_option(
        request.options,
        {"text_chunk_size", "chunk_size"});
    const int64_t chunk_size = value.value_or(280);
    if (chunk_size <= 0) {
        throw std::runtime_error("Piper TTS text_chunk_size must be positive");
    }
    return chunk_size;
}

void append_pause(runtime::AudioBuffer & output, double seconds) {
    if (output.sample_rate <= 0 || seconds <= 0.0) {
        return;
    }
    const size_t count = static_cast<size_t>(
        std::llround(seconds * static_cast<double>(output.sample_rate)));
    output.samples.insert(output.samples.end(), count, 0.0F);
}

}  // namespace

PiperTtsSession::PiperTtsSession(
    runtime::TaskSpec task,
    runtime::SessionOptions options,
    std::shared_ptr<const PiperTtsAssets> assets,
    std::shared_ptr<const engine::model_spec::ModelContract> contract)
    : RuntimeSessionBase(options),
      task_(task),
      assets_(require_assets(std::move(assets))),
      contract_(require_contract(std::move(contract))) {
    if (task_.task != runtime::VoiceTaskKind::Tts ||
        task_.mode != runtime::RunMode::Offline) {
        throw std::runtime_error("Piper TTS only supports offline TTS");
    }
    validate_session_options(options, *contract_);
    frontend_ = std::make_unique<PiperTtsFrontend>(
        session_path(options, "piper_tts.espeak_library_path"),
        session_path(options, "piper_tts.espeak_data_path"),
        assets_->config.espeak_voice,
        assets_->config.phoneme_id_map,
        1000);
    runtime_ = std::make_unique<PiperVitsRuntime>(
        assets_,
        options.backend);
}

PiperTtsSession::~PiperTtsSession() = default;

std::string PiperTtsSession::family() const { return "piper_tts"; }
runtime::VoiceTaskKind PiperTtsSession::task_kind() const { return task_.task; }
runtime::RunMode PiperTtsSession::run_mode() const { return task_.mode; }

void PiperTtsSession::prepare(const runtime::SessionPreparationRequest & request) {
    (void)request;
    mark_prepared();
}

PiperTtsGenerationOptions PiperTtsSession::generation_options(
    const runtime::TaskRequest & request) const {
    PiperTtsGenerationOptions out;
    if (const auto value = runtime::parse_positive_finite_float_option(
            request.options,
            {"speed"})) {
        out.speaking_rate = *value;
    }
    if (request.voice.has_value() && request.voice->style.has_value() &&
        request.voice->style->speaking_rate.has_value()) {
        out.speaking_rate = *request.voice->style->speaking_rate;
    }
    if (!std::isfinite(out.speaking_rate) || out.speaking_rate < 0.5F || out.speaking_rate > 2.0F) {
        throw std::runtime_error("Piper TTS speaking_rate must be between 0.5 and 2.0");
    }
    if (const auto value = runtime::parse_finite_float_option(
            request.options,
            {"variation"})) {
        out.variation = *value;
    }
    if (out.variation < 0.0F || out.variation > 1.0F) {
        throw std::runtime_error("Piper TTS variation must be between 0.0 and 1.0");
    }
    if (const auto value = runtime::parse_finite_float_option(
            request.options,
            {"duration_variation"})) {
        out.duration_variation = *value;
    }
    if (out.duration_variation < 0.0F || out.duration_variation > 2.0F) {
        throw std::runtime_error("Piper TTS duration_variation must be between 0.0 and 2.0");
    }
    if (const auto value = runtime::parse_u32_option(request.options, {"seed"})) {
        out.seed = *value;
    }
    return out;
}

runtime::TaskResult PiperTtsSession::run(const runtime::TaskRequest & request) {
    require_prepared("Piper TTS run");
    if (!request.text_input.has_value() || request.text_input->text.empty()) {
        throw std::runtime_error("Piper TTS requires --text input");
    }
    if (request.audio_input.has_value()) {
        throw std::runtime_error("Piper TTS does not accept audio input");
    }
    if (!request.text_input->language.empty() &&
        request.text_input->language != "en" &&
        request.text_input->language != "en-us" &&
        request.text_input->language != "English") {
        throw std::runtime_error("Piper TTS supports English only");
    }
    const int64_t chunk_size = chunk_size_from_request(request);
    const auto chunk_mode = engine::text::parse_text_chunk_mode_override(request.options)
        .value_or(engine::text::TextChunkMode::Default);
    auto chunks = engine::text::split_text_chunks(
        request.text_input->text,
        chunk_size,
        chunk_mode);
    if (chunks.empty()) {
        throw std::runtime_error("Piper TTS text must not be empty");
    }

    const auto base_options = generation_options(request);
    runtime::AudioBuffer merged;
    for (size_t index = 0; index < chunks.size(); ++index) {
        if (index != 0) {
            append_pause(merged, 0.1);
        }
        const auto frontend = frontend_->encode(chunks[index]);
        auto chunk_options = base_options;
        chunk_options.seed += static_cast<uint32_t>(index);
        auto audio = runtime_->synthesize(frontend.token_ids, chunk_options);
        apply_piper_tts_edge_fade(audio.samples, audio.sample_rate);
        runtime::append_audio_buffer(merged, audio);
    }
    for (float & sample : merged.samples) {
        sample = std::clamp(sample, -1.0F, 1.0F);
    }
    engine::debug::trace_log_scalar(
        "piper_tts.text_chunk_size",
        chunk_size);
    engine::debug::trace_log_scalar(
        "piper_tts.text_chunk_mode",
        engine::text::text_chunk_mode_name(chunk_mode));
    engine::debug::trace_log_scalar(
        "piper_tts.text_chunk_count",
        static_cast<int64_t>(chunks.size()));
    runtime::TaskResult result;
    result.audio_output = std::move(merged);
    return result;
}

std::shared_ptr<runtime::IVoiceModelLoader> make_piper_tts_loader() {
    runtime::SpecBackedVoiceModelConfig<PiperTtsAssets> config;
    config.family = kFamily;
    config.load_assets = load_piper_tts_assets;
    config.create_session = [](
                                const runtime::TaskSpec & task,
                                const runtime::SessionOptions & options,
                                std::shared_ptr<const PiperTtsAssets> assets,
                                std::shared_ptr<const engine::model_spec::ModelContract> contract) {
        return std::make_unique<PiperTtsSession>(
            task,
            options,
            std::move(assets),
            std::move(contract));
    };
    return runtime::make_spec_backed_voice_loader(std::move(config));
}

}  // namespace engine::models::piper_tts
