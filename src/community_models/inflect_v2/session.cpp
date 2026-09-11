#include "engine/community_models/inflect_v2/session.h"

#include "engine/framework/debug/profiler.h"
#include "engine/framework/runtime/options.h"
#include "engine/framework/runtime/spec_backed_model.h"
#include "engine/framework/text/utf8.h"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <utility>

namespace engine::models::inflect_v2 {
namespace {

constexpr const char * kFamily = "inflect_v2";

std::shared_ptr<const InflectV2Assets> require_assets(
    std::shared_ptr<const InflectV2Assets> assets) {
    if (assets == nullptr) {
        throw std::runtime_error("Inflect v2 session requires assets");
    }
    return assets;
}

std::shared_ptr<const engine::model_spec::ModelContract> require_contract(
    std::shared_ptr<const engine::model_spec::ModelContract> contract) {
    if (contract == nullptr) {
        throw std::runtime_error("Inflect v2 session requires a model contract");
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
                "unknown Inflect v2 session option: " + key);
        }
    }
}

int64_t chunk_size_from_request(const runtime::TaskRequest & request) {
    const auto value = runtime::parse_i64_option(
        request.options,
        {"text_chunk_size", "chunk_size"});
    const int64_t chunk_size = value.value_or(280);
    if (chunk_size <= 0) {
        throw std::runtime_error("Inflect v2 text_chunk_size must be positive");
    }
    return chunk_size;
}

void validate_chunk_mode(const runtime::TaskRequest & request) {
    if (const auto value = runtime::find_option(request.options, {"text_chunk_mode", "chunk_mode"})) {
        if (*value != "word_budget" && *value != "default") {
            throw std::runtime_error(
                "Inflect v2 text_chunk_mode must be word_budget");
        }
    }
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

InflectV2Session::InflectV2Session(
    runtime::TaskSpec task,
    runtime::SessionOptions options,
    std::shared_ptr<const InflectV2Assets> assets,
    std::shared_ptr<const engine::model_spec::ModelContract> contract)
    : RuntimeSessionBase(options),
      task_(task),
      assets_(require_assets(std::move(assets))),
      contract_(require_contract(std::move(contract))) {
    if (task_.task != runtime::VoiceTaskKind::Tts ||
        task_.mode != runtime::RunMode::Offline) {
        throw std::runtime_error("Inflect v2 only supports offline TTS");
    }
    validate_session_options(options, *contract_);
    frontend_ = std::make_unique<InflectV2Frontend>(
        session_path(options, "inflect_v2.espeak_library_path"),
        session_path(options, "inflect_v2.espeak_data_path"));
    runtime_ = std::make_unique<InflectV2NativeRuntime>(
        assets_,
        options.backend);
}

InflectV2Session::~InflectV2Session() = default;

std::string InflectV2Session::family() const { return "inflect_v2"; }
runtime::VoiceTaskKind InflectV2Session::task_kind() const { return task_.task; }
runtime::RunMode InflectV2Session::run_mode() const { return task_.mode; }

void InflectV2Session::prepare(const runtime::SessionPreparationRequest & request) {
    (void)request;
    mark_prepared();
}

InflectV2GenerationOptions InflectV2Session::generation_options(
    const runtime::TaskRequest & request) const {
    InflectV2GenerationOptions out;
    if (const auto value = runtime::parse_finite_float_option(
            request.options,
            {"speaking_rate"})) {
        out.speaking_rate = *value;
    }
    if (out.speaking_rate < 0.5F || out.speaking_rate > 2.0F) {
        throw std::runtime_error("Inflect v2 speaking_rate must be between 0.5 and 2.0");
    }
    if (const auto value = runtime::parse_finite_float_option(
            request.options,
            {"variation"})) {
        out.variation = *value;
    }
    if (out.variation < 0.0F || out.variation > 1.0F) {
        throw std::runtime_error("Inflect v2 variation must be between 0.0 and 1.0");
    }
    if (const auto value = runtime::parse_u32_option(request.options, {"seed"})) {
        out.seed = *value;
    }
    return out;
}

runtime::TaskResult InflectV2Session::run(const runtime::TaskRequest & request) {
    require_prepared("Inflect v2 run");
    if (!request.text_input.has_value() || request.text_input->text.empty()) {
        throw std::runtime_error("Inflect v2 requires --text input");
    }
    if (request.audio_input.has_value()) {
        throw std::runtime_error("Inflect v2 does not accept audio input");
    }
    if (!request.text_input->language.empty() &&
        request.text_input->language != "en" &&
        request.text_input->language != "en-us" &&
        request.text_input->language != "English") {
        throw std::runtime_error("Inflect v2 supports English only");
    }
    validate_chunk_mode(request);
    const int64_t chunk_size = chunk_size_from_request(request);
    auto chunks = InflectV2Frontend::split_text(
        request.text_input->text,
        chunk_size);
    if (chunks.empty()) {
        throw std::runtime_error("Inflect v2 text must not be empty");
    }

    const auto base_options = generation_options(request);
    runtime::AudioBuffer merged;
    for (size_t index = 0; index < chunks.size(); ++index) {
        if (index != 0) {
            append_pause(
                merged,
                InflectV2Frontend::boundary_pause_seconds(chunks[index - 1]));
        }
        const auto frontend = frontend_->encode(chunks[index]);
        auto chunk_options = base_options;
        chunk_options.seed += static_cast<uint32_t>(index);
        auto audio = runtime_->synthesize(frontend.token_ids, chunk_options);
        apply_inflect_v2_edge_fade(audio.samples, audio.sample_rate);
        runtime::append_audio_buffer(merged, audio);
    }
    for (float & sample : merged.samples) {
        sample = std::clamp(sample, -1.0F, 1.0F);
    }
    engine::debug::trace_log_scalar(
        "inflect_v2.text_chunk_size",
        chunk_size);
    engine::debug::trace_log_scalar(
        "inflect_v2.text_chunk_count",
        static_cast<int64_t>(chunks.size()));
    runtime::TaskResult result;
    result.audio_output = std::move(merged);
    return result;
}

std::shared_ptr<runtime::IVoiceModelLoader> make_inflect_v2_loader() {
    runtime::SpecBackedVoiceModelConfig<InflectV2Assets> config;
    config.family = kFamily;
    config.load_assets = load_inflect_v2_assets;
    config.create_session = [](
                                const runtime::TaskSpec & task,
                                const runtime::SessionOptions & options,
                                std::shared_ptr<const InflectV2Assets> assets,
                                std::shared_ptr<const engine::model_spec::ModelContract> contract) {
        return std::make_unique<InflectV2Session>(
            task,
            options,
            std::move(assets),
            std::move(contract));
    };
    return runtime::make_spec_backed_voice_loader(std::move(config));
}

}  // namespace engine::models::inflect_v2
