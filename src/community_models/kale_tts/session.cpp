#include "engine/community_models/kale_tts/session.h"

#include "engine/framework/debug/profiler.h"
#include "engine/framework/runtime/errors.h"
#include "engine/framework/runtime/options.h"
#include "engine/framework/runtime/spec_backed_model.h"
#include "engine/framework/text/chunking.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <stdexcept>
#include <utility>

namespace engine::models::kale_tts {
namespace {

constexpr const char * kFamily = "kale_tts";
constexpr const char * kModelName = "kale-tts-1";
constexpr int64_t kDefaultTextChunkSize = 400;
constexpr float kGapSeconds = 0.12F;        // silence between sentences, as the release runtimes
constexpr int32_t kSpaceId = 16;

std::filesystem::path session_path(const runtime::SessionOptions & options, const char * key) {
    const auto found = options.options.find(key);
    return found == options.options.end() ? std::filesystem::path{} : std::filesystem::path(found->second);
}

std::string lower(std::string text) {
    for (char & c : text) c = (c >= 'A' && c <= 'Z') ? static_cast<char>(c - 'A' + 'a') : c;
    return text;
}

}  // namespace

KaleTtsSession::KaleTtsSession(
    runtime::TaskSpec task,
    runtime::SessionOptions options,
    std::shared_ptr<const KaleAssets> assets,
    std::shared_ptr<const engine::model_spec::ModelContract> contract)
    : RuntimeSessionBase(options),
      task_(task),
      assets_(std::move(assets)),
      contract_(std::move(contract)) {
    if (assets_ == nullptr || contract_ == nullptr) {
        throw std::runtime_error("kale-tts session requires assets and a model contract");
    }
    if (task_.task != runtime::VoiceTaskKind::Tts || task_.mode != runtime::RunMode::Offline) {
        throw std::runtime_error("kale-tts only supports offline TTS");
    }
    runtime::validate_spec_backed_session_options(RuntimeSessionBase::options(), *contract_, kFamily, kModelName);
    frontend_ = std::make_unique<KaleFrontend>(*assets_, session_path(options, "kale_tts.espeak_library_path"),
                                               session_path(options, "kale_tts.espeak_data_path"));
    runtime_ = std::make_unique<KaleRuntime>(assets_, options.backend);
    if (const auto path = session_path(options, "kale_tts.parity_noise_file"); !path.empty()) {
        std::ifstream in(path, std::ios::binary);
        if (!in) throw std::runtime_error("kale-tts: cannot read " + path.string());
        in.seekg(0, std::ios::end);
        parity_noise_.resize(static_cast<size_t>(in.tellg()) / sizeof(float));
        in.seekg(0);
        in.read(reinterpret_cast<char *>(parity_noise_.data()), static_cast<std::streamsize>(parity_noise_.size() * sizeof(float)));
    }
}

KaleTtsSession::~KaleTtsSession() = default;

std::string KaleTtsSession::family() const { return kFamily; }
runtime::VoiceTaskKind KaleTtsSession::task_kind() const { return task_.task; }
runtime::RunMode KaleTtsSession::run_mode() const { return task_.mode; }

void KaleTtsSession::prepare(const runtime::SessionPreparationRequest & request) {
    (void)request;
    mark_prepared();
}

runtime::TaskResult KaleTtsSession::run(const runtime::TaskRequest & request) {
    require_prepared("kale-tts run");
    if (!request.text_input.has_value() || request.text_input->text.empty()) {
        throw std::runtime_error("kale-tts requires text input");
    }
    runtime::validate_spec_backed_request_options(request.options, request.option_arrays, *contract_, kModelName);

    std::string voice_name = assets_->voices.front().name;
    if (request.voice.has_value() && request.voice->speaker.has_value() && request.voice->speaker->cached_voice_id.has_value()) {
        voice_name = *request.voice->speaker->cached_voice_id;
    } else if (const auto value = runtime::find_option(request.options, {"voice_id"})) {
        voice_name = *value;
    }
    const auto voice = std::find_if(assets_->voices.begin(), assets_->voices.end(),
                                    [&](const KaleVoice & v) { return v.name == lower(voice_name) || v.id == voice_name; });
    if (voice == assets_->voices.end()) {
        std::string names;
        for (const auto & v : assets_->voices) names += (names.empty() ? "" : ", ") + v.name;
        throw runtime::InvalidRequestError("kale-tts has no voice named " + voice_name + " (voices: " + names + ")");
    }
    float speed = 1.0F;
    if (const auto value = runtime::parse_finite_float_option(request.options, {"speed"})) {
        speed = *value;
    } else if (request.voice.has_value() && request.voice->style.has_value() && request.voice->style->speaking_rate.has_value()) {
        speed = *request.voice->style->speaking_rate;
    }
    if (!(speed >= 0.25F && speed <= 4.0F)) {
        throw runtime::InvalidRequestError("kale-tts speed must be between 0.25 and 4");
    }
    uint32_t seed = 0;
    if (const auto value = runtime::parse_u64_option(request.options, {"seed"})) {
        seed = static_cast<uint32_t>(*value);
    }

    // The sentences to speak: from the caller's phonemes (one entry per sentence) or from the text.
    std::vector<KaleFrontend::Sentence> sentences;
    const auto phonemes = request.option_arrays.find("phonemes");
    if (phonemes != request.option_arrays.end()) {
        if (phonemes->second.empty()) {
            throw runtime::InvalidRequestError("kale-tts 'phonemes' was set but holds no entries");
        }
        // tiny's word reader still reads the words: one text sentence per entry when the counts agree, else the whole text
        const auto texts = frontend_->sentence_texts(request.text_input->text);
        for (size_t k = 0; k < phonemes->second.size(); ++k) {
            KaleFrontend::Sentence s;
            s.text = texts.size() == phonemes->second.size() ? texts[k] : request.text_input->text;
            s.ids = frontend_->phoneme_ids(phonemes->second[k]);
            if (s.ids.empty()) throw runtime::InvalidRequestError("kale-tts phoneme entry " + std::to_string(k) + " is empty");
            sentences.push_back(std::move(s));
        }
    } else {
        const int64_t chunk_size = engine::text::parse_text_chunk_size_override(request.options).value_or(kDefaultTextChunkSize);
        const auto chunk_mode = engine::text::parse_text_chunk_mode_override(request.options).value_or(engine::text::TextChunkMode::Default);
        for (const auto & chunk : runtime::chunk_text_request(request, chunk_size, chunk_mode)) {
            if (!chunk.text_input.has_value()) continue;
            for (auto & s : frontend_->sentences(chunk.text_input->text, voice->accent)) sentences.push_back(std::move(s));
        }
    }
    if (sentences.empty()) {
        throw runtime::InvalidRequestError("kale-tts: nothing in the text could be read");
    }

    const auto start = std::chrono::steady_clock::now();
    const auto rate = static_cast<int>(assets_->config.sample_rate);
    const std::vector<float> gap(static_cast<size_t>(kGapSeconds * static_cast<float>(rate)), 0.0F);
    runtime::AudioBuffer audio;
    audio.sample_rate = rate;
    audio.channels = 1;
    std::string previous;
    for (size_t k = 0; k < sentences.size(); ++k) {
        auto & s = sentences[k];
        KaleRequest r;
        r.voice = &*voice;
        r.speed = speed;
        r.seed = seed + static_cast<uint32_t>(k);
        if (static_cast<int64_t>(s.ids.size()) > assets_->config.max_tokens) {
            s.ids.resize(static_cast<size_t>(assets_->config.max_tokens));
        }
        r.ids = s.ids;
        if (!parity_noise_.empty()) r.noise = &parity_noise_;
        if (assets_->config.ettin) {
            const EttinInputs e = ettin_inputs(*frontend_->ettin(), s.text, previous);
            r.ettin_ids = e.ids;
            r.ettin_wavg = e.wavg;
            r.ettin_words = e.words;
            r.word_index = word_index(r.ids, s.word_ids, e.words, kSpaceId);
        }
        const auto wave = runtime_->synthesize(r);
        for (const float sample : wave) {
            if (!std::isfinite(sample)) throw std::runtime_error("kale-tts produced non-finite audio");
        }
        if (k) audio.samples.insert(audio.samples.end(), gap.begin(), gap.end());
        audio.samples.insert(audio.samples.end(), wave.begin(), wave.end());
        previous = s.text;
    }
    engine::debug::trace_log_scalar("kale_tts.sentence_count", static_cast<int64_t>(sentences.size()));
    engine::debug::timing_log_scalar("session.wall_ms", engine::debug::elapsed_ms(start));
    runtime::TaskResult result;
    result.audio_output = std::move(audio);
    return result;
}

std::shared_ptr<runtime::IVoiceModelLoader> make_kale_tts_loader() {
    runtime::SpecBackedVoiceModelConfig<KaleAssets> config;
    config.family = kFamily;
    config.load_assets = load_kale_assets;
    config.create_session = [](
                                const runtime::TaskSpec & task,
                                const runtime::SessionOptions & options,
                                std::shared_ptr<const KaleAssets> assets,
                                std::shared_ptr<const engine::model_spec::ModelContract> contract) {
        return std::make_unique<KaleTtsSession>(task, options, std::move(assets), std::move(contract));
    };
    return runtime::make_spec_backed_voice_loader(std::move(config));
}

}  // namespace engine::models::kale_tts
