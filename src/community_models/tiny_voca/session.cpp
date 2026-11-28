#include "engine/community_models/tiny_voca/session.h"

#include "engine/framework/debug/profiler.h"
#include "engine/framework/runtime/errors.h"
#include "engine/framework/runtime/options.h"
#include "engine/framework/runtime/spec_backed_model.h"
#include "engine/framework/text/chunking.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace engine::models::tiny_voca {
namespace {

constexpr const char * kFamily = "tiny_voca";
constexpr const char * kModelName = "tiny-voca";
constexpr int64_t kDefaultTextChunkSize = 400;
constexpr const char * kDefaultVoice = "M2";
// The loudest sample of the audio a request returns. The model sometimes peaks
// above 1.0, which clips when the audio is saved as 16-bit samples.
constexpr float kPeak = 0.99F;
// One semitone, in the units of log F0.
constexpr float kSemitone = 0.057762265F;

// en, en-us and English, in any case, with a hyphen or an underscore.
bool is_english(const std::string & language) {
    std::string name;
    for (const char c : language) {
        name.push_back(c == '_' ? '-' : (c >= 'A' && c <= 'Z' ? static_cast<char>(c - 'A' + 'a') : c));
    }
    return name.empty() || name == "en" || name == "en-us" || name == "english";
}

}  // namespace

void require_english(const runtime::TaskRequest & request) {
    std::string language;
    if (request.text_input.has_value() && !is_english(request.text_input->language)) {
        language = request.text_input->language;
    } else if (const auto option = runtime::find_option(request.options, {"language"}); option.has_value() && !is_english(*option)) {
        language = *option;
    } else {
        return;
    }
    throw runtime::InvalidRequestError(
        "tiny-voca reads English only: language '" + language + "' is not supported. Use en, en-us or English, or leave the language out.");
}

TinyVocaSession::TinyVocaSession(
    runtime::TaskSpec task,
    runtime::SessionOptions options,
    std::shared_ptr<const TinyVocaAssets> assets,
    std::shared_ptr<const engine::model_spec::ModelContract> contract)
    : RuntimeSessionBase(options),
      task_(task),
      assets_(std::move(assets)),
      contract_(std::move(contract)) {
    if (assets_ == nullptr || contract_ == nullptr) {
        throw std::runtime_error("tiny-voca session requires assets and a model contract");
    }
    if (task_.task != runtime::VoiceTaskKind::Tts || task_.mode != runtime::RunMode::Offline) {
        throw std::runtime_error("tiny-voca only supports offline TTS");
    }
    runtime::validate_spec_backed_session_options(RuntimeSessionBase::options(), *contract_, kFamily, kModelName);
    frontend_ = std::make_unique<TinyVocaFrontend>(*assets_);
    runtime_ = std::make_unique<TinyVocaRuntime>(assets_, options.backend);
}

TinyVocaSession::~TinyVocaSession() = default;

std::string TinyVocaSession::family() const { return kFamily; }
runtime::VoiceTaskKind TinyVocaSession::task_kind() const { return task_.task; }
runtime::RunMode TinyVocaSession::run_mode() const { return task_.mode; }

void TinyVocaSession::prepare(const runtime::SessionPreparationRequest & request) {
    (void)request;
    mark_prepared();
}

TinyVocaGenerationOptions TinyVocaSession::generation_options(const runtime::TaskRequest & request) const {
    TinyVocaGenerationOptions out;
    std::string voice = kDefaultVoice;
    if (request.voice.has_value() && request.voice->speaker.has_value() && request.voice->speaker->cached_voice_id.has_value()) {
        voice = *request.voice->speaker->cached_voice_id;
    } else if (const auto value = runtime::find_option(request.options, {"voice_id"})) {
        voice = *value;
    }
    const auto & voices = assets_->config.voices;
    const auto found = std::find(voices.begin(), voices.end(), voice);
    if (found == voices.end()) {
        throw std::runtime_error("tiny-voca has no voice named " + voice + " (voices: M2, F2)");
    }
    out.voice = static_cast<int64_t>(found - voices.begin());
    if (const auto value = runtime::parse_finite_float_option(request.options, {"speed"})) {
        out.speed = *value;
    } else if (request.voice.has_value() && request.voice->style.has_value() && request.voice->style->speaking_rate.has_value()) {
        out.speed = *request.voice->style->speaking_rate;
    }
    if (out.speed < 0.25F || out.speed > 4.0F) {
        throw std::runtime_error("tiny-voca speed must be between 0.25 and 4");
    }
    if (const auto value = runtime::parse_i64_option(request.options, {"semitone_shift"})) {
        if (*value < -24 || *value > 24) {
            throw std::runtime_error("tiny-voca semitone_shift must be between -24 and 24");
        }
        out.pitch_shift = static_cast<float>(*value) * kSemitone;
    }
    // The fine shift is the model's own pitch_shift; the two add up.
    if (const auto value = runtime::parse_finite_float_option(request.options, {"pitch_shift"})) {
        if (*value < -1.4F || *value > 1.4F) {
            throw std::runtime_error("tiny-voca pitch_shift must be between -1.4 and 1.4");
        }
        out.pitch_shift += *value;
    }
    if (const auto value = runtime::parse_finite_float_option(request.options, {"pitch_range"})) {
        out.pitch_range = *value;
    }
    if (out.pitch_range < 0.0F || out.pitch_range > 3.0F) {
        throw std::runtime_error("tiny-voca pitch_range must be between 0 and 3");
    }
    if (const auto value = runtime::parse_u64_option(request.options, {"seed"})) {
        out.seed = static_cast<uint32_t>(*value);
    }
    return out;
}

void TinyVocaSession::speak(
    const std::string & text, const TinyVocaGenerationOptions & options, std::vector<std::vector<float>> & waves, int64_t & spoken) {
    const auto phones = frontend_->phonemize(text);
    if (!phones.has_value()) {
        return;                                 // nothing to pronounce: another script, a face, signs only
    }
    if (static_cast<int64_t>(phones->ids.size()) <= assets_->config.max_tokens - 12) {
        try {
            TinyVocaGenerationOptions own = options;
            own.seed += static_cast<uint32_t>(spoken);
            TinyVocaOutput out = runtime_->synthesize(phones->ids, phones->stress, own);
            for (const float sample : out.samples) {
                if (!std::isfinite(sample)) {
                    throw std::runtime_error("tiny-voca produced non-finite audio");
                }
            }
            waves.push_back(std::move(out.samples));
            ++spoken;
            return;
        } catch (const TinyVocaTooLong &) {
            // cut it smaller, below
        }
    }
    const std::vector<std::string> pieces = TinyVocaFrontend::shorter(text);
    if (pieces.empty()) {
        throw std::runtime_error("tiny-voca: one word is too long to speak: " + text.substr(0, 40));
    }
    for (const std::string & piece : pieces) {
        speak(piece, options, waves, spoken);
    }
}

runtime::TaskResult TinyVocaSession::run(const runtime::TaskRequest & request) {
    require_prepared("tiny-voca run");
    if (!request.text_input.has_value() || request.text_input->text.empty()) {
        throw std::runtime_error("tiny-voca requires text input");
    }
    if (request.audio_input.has_value()) {
        throw std::runtime_error("tiny-voca does not accept audio input");
    }
    require_english(request);
    runtime::validate_spec_backed_request_options(request.options, request.option_arrays, *contract_, kModelName);
    const TinyVocaGenerationOptions options = generation_options(request);
    const int64_t chunk_size = engine::text::parse_text_chunk_size_override(request.options).value_or(kDefaultTextChunkSize);
    const auto chunk_mode =
        engine::text::parse_text_chunk_mode_override(request.options).value_or(engine::text::TextChunkMode::Default);
    const auto chunks = runtime::chunk_text_request(request, chunk_size, chunk_mode);

    const auto start = std::chrono::steady_clock::now();
    // The sentences as they are spoken; they are joined once, at their exact
    // size, so a long request never holds more than its audio twice.
    std::vector<std::vector<float>> waves;
    int64_t spoken = 0;
    for (size_t index = 0; index < chunks.size(); ++index) {
        if (!chunks[index].text_input.has_value()) {
            continue;
        }
        // The model was trained on single sentences and speaks one at a time. A
        // text that ends without a mark gets a full stop: before a final mark the
        // voice lets the sound fall, without one the audio breaks off.
        std::string text = chunks[index].text_input->text;
        if (index + 1 == chunks.size()) {
            text = TinyVocaFrontend::finished(text);
        }
        for (const std::string & sentence : TinyVocaFrontend::sentences(text)) {
            speak(sentence, options, waves, spoken);
        }
    }
    if (spoken == 0) {
        throw std::runtime_error("tiny-voca: nothing in the text could be read. tiny-voca only reads English for now.");
    }
    runtime::AudioBuffer audio;
    audio.sample_rate = static_cast<int>(assets_->config.sample_rate);
    audio.channels = 1;
    size_t total = 0;
    for (const auto & wave : waves) {
        total += wave.size();
    }
    audio.samples.reserve(total);
    for (auto & wave : waves) {
        audio.samples.insert(audio.samples.end(), wave.begin(), wave.end());
        std::vector<float>().swap(wave);
    }
    float loudest = 0.0F;
    for (const float sample : audio.samples) {
        loudest = std::max(loudest, std::fabs(sample));
    }
    if (loudest > kPeak) {
        const float scale = kPeak / loudest;
        for (float & sample : audio.samples) {
            sample *= scale;
        }
    }
    engine::debug::trace_log_scalar("tiny_voca.text_chunk_size", chunk_size);
    engine::debug::trace_log_scalar("tiny_voca.text_chunk_count", static_cast<int64_t>(chunks.size()));
    engine::debug::trace_log_scalar("tiny_voca.utterance_count", spoken);
    engine::debug::timing_log_scalar("session.wall_ms", engine::debug::elapsed_ms(start));
    runtime::TaskResult result;
    result.audio_output = std::move(audio);
    return result;
}

std::shared_ptr<runtime::IVoiceModelLoader> make_tiny_voca_loader() {
    runtime::SpecBackedVoiceModelConfig<TinyVocaAssets> config;
    config.family = kFamily;
    config.load_assets = load_tiny_voca_assets;
    config.create_session = [](
                                const runtime::TaskSpec & task,
                                const runtime::SessionOptions & options,
                                std::shared_ptr<const TinyVocaAssets> assets,
                                std::shared_ptr<const engine::model_spec::ModelContract> contract) {
        return std::make_unique<TinyVocaSession>(task, options, std::move(assets), std::move(contract));
    };
    return runtime::make_spec_backed_voice_loader(std::move(config));
}

}  // namespace engine::models::tiny_voca
