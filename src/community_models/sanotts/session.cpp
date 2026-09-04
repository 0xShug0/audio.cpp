#include "engine/community_models/sanotts/session.h"

#include "engine/framework/debug/profiler.h"
#include "engine/framework/runtime/options.h"
#include "engine/framework/runtime/spec_backed_model.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <stdexcept>
#include <string>
#include <utility>

namespace engine::models::sanotts {
namespace {

constexpr const char * kFamily = "sanotts";

std::shared_ptr<const SanoTtsAssets> require_assets(
    std::shared_ptr<const SanoTtsAssets> assets) {
    if (assets == nullptr) {
        throw std::runtime_error("sanoTTS session requires assets");
    }
    return assets;
}

std::shared_ptr<const engine::model_spec::ModelContract> require_contract(
    std::shared_ptr<const engine::model_spec::ModelContract> contract) {
    if (contract == nullptr) {
        throw std::runtime_error("sanoTTS session requires a model contract");
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
            throw std::runtime_error("unknown sanoTTS session option: " + key);
        }
    }
}

int64_t chunk_size_from_request(const runtime::TaskRequest & request) {
    const auto value = runtime::parse_i64_option(
        request.options,
        {"text_chunk_size", "chunk_size"});
    const int64_t chunk_size = value.value_or(280);
    if (chunk_size <= 0) {
        throw std::runtime_error("sanoTTS text_chunk_size must be positive");
    }
    return chunk_size;
}

void validate_chunk_mode(const runtime::TaskRequest & request) {
    if (const auto value = runtime::find_option(
            request.options,
            {"text_chunk_mode", "chunk_mode"})) {
        if (*value != "word_budget" && *value != "default") {
            throw std::runtime_error("sanoTTS text_chunk_mode must be word_budget");
        }
    }
}

struct RequestOptions {
    float speaking_rate = 1.0F;
    uint64_t seed = 0;
    bool seed_from_text = true;
};

RequestOptions parse_request_options(const runtime::TaskRequest & request) {
    RequestOptions out;
    if (const auto value = runtime::parse_finite_float_option(
            request.options,
            {"speaking_rate"})) {
        out.speaking_rate = *value;
    }
    if (out.speaking_rate < 0.5F || out.speaking_rate > 2.0F) {
        throw std::runtime_error("sanoTTS speaking_rate must be between 0.5 and 2.0");
    }
    if (const auto value = runtime::parse_i64_option(request.options, {"seed"})) {
        if (*value < 0) {
            throw std::runtime_error("sanoTTS seed must not be negative");
        }
        out.seed = static_cast<uint64_t>(*value);
        out.seed_from_text = *value == 0;
    }
    return out;
}

void append_pause(runtime::AudioBuffer & output, double seconds) {
    if (output.sample_rate <= 0 || seconds <= 0.0) {
        return;
    }
    const auto count = static_cast<size_t>(
        std::llround(seconds * static_cast<double>(output.sample_rate)));
    output.samples.insert(output.samples.end(), count, 0.0F);
}

}  // namespace

SanoTtsSession::SanoTtsSession(
    runtime::TaskSpec task,
    runtime::SessionOptions options,
    std::shared_ptr<const SanoTtsAssets> assets,
    std::shared_ptr<const engine::model_spec::ModelContract> contract)
    : RuntimeSessionBase(options),
      task_(task),
      assets_(require_assets(std::move(assets))),
      contract_(require_contract(std::move(contract))) {
    if (task_.task != runtime::VoiceTaskKind::Tts ||
        task_.mode != runtime::RunMode::Offline) {
        throw std::runtime_error("sanoTTS only supports offline TTS");
    }
    validate_session_options(options, *contract_);
    frontend_ = std::make_unique<SanoTtsFrontend>(
        session_path(options, "sanotts.espeak_library_path"),
        session_path(options, "sanotts.espeak_data_path"),
        assets_->config.duration_max_tokens);
    runtime_ = std::make_unique<SanoTtsNativeRuntime>(assets_, options.backend);
}

SanoTtsSession::~SanoTtsSession() = default;

std::string SanoTtsSession::family() const { return kFamily; }
runtime::VoiceTaskKind SanoTtsSession::task_kind() const { return task_.task; }
runtime::RunMode SanoTtsSession::run_mode() const { return task_.mode; }

void SanoTtsSession::prepare(const runtime::SessionPreparationRequest & request) {
    (void)request;
    mark_prepared();
}

runtime::TaskResult SanoTtsSession::run(const runtime::TaskRequest & request) {
    require_prepared("sanoTTS run");
    if (!request.text_input.has_value() || request.text_input->text.empty()) {
        throw std::runtime_error("sanoTTS requires --text input");
    }
    if (request.audio_input.has_value()) {
        throw std::runtime_error("sanoTTS does not accept audio input");
    }
    if (!request.text_input->language.empty() &&
        request.text_input->language != "en" &&
        request.text_input->language != "en-us" &&
        request.text_input->language != "English") {
        throw std::runtime_error("sanoTTS supports English only");
    }
    validate_chunk_mode(request);
    const int64_t chunk_size = chunk_size_from_request(request);
    auto chunks = SanoTtsFrontend::split_text(request.text_input->text, chunk_size);
    if (chunks.empty()) {
        throw std::runtime_error("sanoTTS text must not be empty");
    }

    const auto request_options = parse_request_options(request);
    runtime::AudioBuffer merged;
    uint64_t rendered_chunks = 0;
    // Renders one chunk, bisecting at whitespace when it phonemizes past the
    // duration model's token limit -- the codepoint budget cannot see phoneme
    // counts, so a dense 280-codepoint chunk can exceed 207 tokens.
    const std::function<void(const std::string &, int)> render_chunk =
        [&](const std::string & chunk, int depth) {
            SanoTtsEncoded encoded;
            try {
                encoded = frontend_->encode(chunk);
            } catch (const SanoTtsTooLongError &) {
                const size_t middle = chunk.size() / 2;
                size_t split = std::string::npos;
                for (size_t offset = 0; offset < chunk.size(); ++offset) {
                    const size_t after = middle + offset;
                    if (after < chunk.size() && chunk[after] == ' ') {
                        split = after;
                        break;
                    }
                    if (offset <= middle && chunk[middle - offset] == ' ') {
                        split = middle - offset;
                        break;
                    }
                }
                if (depth >= 8 || split == std::string::npos) {
                    throw;
                }
                const std::string left = chunk.substr(0, split);
                const std::string right = chunk.substr(split + 1);
                render_chunk(left, depth + 1);
                append_pause(merged, SanoTtsFrontend::boundary_pause_seconds(left));
                render_chunk(right, depth + 1);
                return;
            }
            SanoTtsGenerationOptions chunk_options;
            chunk_options.speaking_rate = request_options.speaking_rate;
            // The default seed is derived from the chunk's own text -- the
            // reference implementations' sha256(text)[:8] convention -- so a
            // given sentence renders identically wherever it appears. An
            // explicit seed advances per chunk instead, so long-form noise
            // is not reused across chunks.
            chunk_options.seed = request_options.seed_from_text
                ? sanotts_text_seed(chunk)
                : request_options.seed + rendered_chunks;
            ++rendered_chunks;
            auto audio = runtime_->synthesize(encoded.token_ids, chunk_options);
            runtime::append_audio_buffer(merged, audio);
        };
    for (size_t index = 0; index < chunks.size(); ++index) {
        if (index != 0) {
            append_pause(
                merged,
                SanoTtsFrontend::boundary_pause_seconds(chunks[index - 1]));
        }
        render_chunk(chunks[index], 0);
    }
    for (float & sample : merged.samples) {
        sample = std::clamp(sample, -1.0F, 1.0F);
    }
    engine::debug::trace_log_scalar("sanotts.text_chunk_size", chunk_size);
    engine::debug::trace_log_scalar(
        "sanotts.text_chunk_count",
        static_cast<int64_t>(chunks.size()));
    runtime::TaskResult result;
    result.audio_output = std::move(merged);
    return result;
}

std::shared_ptr<runtime::IVoiceModelLoader> make_sanotts_loader() {
    runtime::SpecBackedVoiceModelConfig<SanoTtsAssets> config;
    config.family = kFamily;
    config.load_assets = load_sanotts_assets;
    config.create_session = [](
                                const runtime::TaskSpec & task,
                                const runtime::SessionOptions & options,
                                std::shared_ptr<const SanoTtsAssets> assets,
                                std::shared_ptr<const engine::model_spec::ModelContract> contract) {
        return std::make_unique<SanoTtsSession>(
            task,
            options,
            std::move(assets),
            std::move(contract));
    };
    return runtime::make_spec_backed_voice_loader(std::move(config));
}

}  // namespace engine::models::sanotts
