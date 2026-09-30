#include "engine/community_models/lfm2_audio/session.h"

#include "engine/framework/audio/chunking.h"
#include "engine/framework/io/filesystem.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/debug/trace.h"
#include "engine/framework/io/text.h"
#include "engine/framework/runtime/options.h"
#include "engine/framework/runtime/spec_backed_model.h"
#include "engine/models/silero_vad/session.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace engine::community_models::lfm2_audio {
namespace {

constexpr const char * kFamily = "lfm2_audio";
constexpr const char * kModelName = "LFM2-Audio";
constexpr int kSampleRate = 16000;
constexpr float kDefaultChunkSeconds = 30.0f;
// A chunk under a second is at best a cut-off word, and under 10 ms too short
// for the features. The spec's min for audio_chunk_seconds matches; the
// framework does not enforce spec bounds, so plan_chunks does.
constexpr float kMinChunkSeconds = 1.0f;

const engine::model_spec::ModelContract & require_contract(
    const std::shared_ptr<const engine::model_spec::ModelContract> & contract) {
    if (contract == nullptr) {
        throw std::runtime_error("LFM2-Audio session requires a model contract");
    }

    return *contract;
}

// Warnings go to stderr, as other families print theirs.
void warn(const std::string & message) {
    std::cerr << "[warning][" << kFamily << "] " << message << "\n";
}

// A position in the 16 kHz input, in seconds to a tenth.
std::string seconds_at(int64_t sample) {
    char out[32];
    std::snprintf(out, sizeof(out), "%.1f", static_cast<double>(sample) / kSampleRate);
    return out;
}

runtime::SessionOptions validate_session_setup(
    const runtime::TaskSpec & task,
    runtime::SessionOptions options,
    const engine::model_spec::ModelContract & contract) {
    if (task.task != runtime::VoiceTaskKind::Asr) {
        throw std::runtime_error("LFM2-Audio currently supports VoiceTaskKind::Asr only");
    }

    if (task.mode != runtime::RunMode::Offline) {
        throw std::runtime_error("LFM2-Audio currently supports offline sessions only");
    }

    runtime::validate_spec_backed_session_options(options, contract, kFamily, kModelName);
    return options;
}

std::shared_ptr<const Lfm2AudioComponents> select_components(
    const std::shared_ptr<const Lfm2AudioAssets> & assets, const runtime::SessionOptions & options) {
    if (assets == nullptr) {
        throw std::runtime_error("LFM2-Audio session requires assets");
    }

    return load_lfm2_audio_components(
        *assets,
        runtime::find_option(options.options, {"lfm2_audio.model_gguf"}).value_or(""),
        runtime::find_option(options.options, {"lfm2_audio.mmproj_gguf"}).value_or(""));
}

// The published checkpoints are single-language; the ASR system prompt is
// the one each was trained with (liquid-audio README / README_JP). A GGUF
// without general.languages gets the English prompt, the base model's.
std::string model_language(const Lfm2AudioComponents & components) {
    const auto & languages = components.languages;
    const bool en = std::find(languages.begin(), languages.end(), "en") != languages.end();
    const bool ja = std::find(languages.begin(), languages.end(), "ja") != languages.end();
    if (en != ja) {
        return ja ? "ja" : "en";
    }

    if (languages.empty()) {
        return "en";
    }

    std::string listed;
    for (const auto & language : languages) {
        listed += (listed.empty() ? "" : ", ") + language;
    }

    throw std::runtime_error("LFM2-Audio has an ASR prompt for en or ja checkpoints, not general.languages = [" + listed + "]");
}

bool is_ascii_alnum(unsigned char value) {
    return (value >= '0' && value <= '9') || (value >= 'A' && value <= 'Z') || (value >= 'a' && value <= 'z');
}

// Chunk transcripts are joined with a space only after ASCII text and before
// an ASCII word, so Japanese text stays unbroken.
void append_chunk_text(std::string & merged, std::string chunk) {
    chunk = io::trim_ascii_whitespace(std::move(chunk));
    if (chunk.empty()) {
        return;
    }

    if (!merged.empty() && static_cast<unsigned char>(merged.back()) < 0x80 &&
        is_ascii_alnum(static_cast<unsigned char>(chunk.front()))) {
        merged.push_back(' ');
    }

    merged += chunk;
}

// Fixed chunks of audio_chunk_seconds. A short tail joins the previous chunk.
std::vector<runtime::TimeSpan> plan_fixed_chunks(int64_t samples, int64_t chunk_samples) {
    std::vector<runtime::TimeSpan> spans;
    for (const auto & chunk : audio::plan_audio_chunks(samples, {chunk_samples, chunk_samples})) {
        spans.push_back({chunk.output_start_sample, chunk.output_start_sample + chunk.valid_samples});
    }

    const auto min_samples = static_cast<int64_t>(kMinChunkSeconds * kSampleRate);
    if (spans.size() > 1 && spans.back().end_sample - spans.back().start_sample < min_samples) {
        spans.pop_back();
        spans.back().end_sample = samples;
    }

    return spans;
}

std::filesystem::path default_vad_model_path() {
    return std::filesystem::path("assets") / "framework" / "models" / "silero_vad";
}

// Set LFM2_AUDIO_DUMP_DIR to write each stage (features, adapter output,
// prompt ids, first logits, tokens) as raw f32/i32 for stage-by-stage
// comparison against the reference implementation.
void debug_dump(const char * name, const void * data, size_t bytes) {
    const char * dir = std::getenv("LFM2_AUDIO_DUMP_DIR");
    if (dir == nullptr || *dir == '\0') {
        return;
    }

    std::ofstream(std::string(dir) + "/" + name, std::ios::binary).write(static_cast<const char *>(data), static_cast<std::streamsize>(bytes));
}

}  // namespace

Lfm2AudioSession::Lfm2AudioSession(
    runtime::TaskSpec task,
    runtime::SessionOptions options,
    std::shared_ptr<const Lfm2AudioAssets> assets,
    std::shared_ptr<const engine::model_spec::ModelContract> contract)
    : RuntimeSessionBase(validate_session_setup(task, std::move(options), require_contract(contract))),
      task_(std::move(task)),
      assets_(std::move(assets)),
      contract_(std::move(contract)),
      components_(select_components(assets_, RuntimeSessionBase::options())),
      tokenizer_(components_->vocabulary),
      features_(components_->encoder.n_mels, execution_context().config().threads),
      encoder_(components_->mmproj, components_->encoder, execution_context()),
      backbone_(components_->model, components_->backbone, execution_context()),
      language_(model_language(*components_)),
      prompt_(make_lfm2_asr_prompt(tokenizer_, language_)),
      vad_model_path_(runtime::find_option(RuntimeSessionBase::options().options, {"lfm2_audio.vad_model_path"})
                          .value_or(default_vad_model_path().string())) {
    components_->model->release_storage();
    components_->mmproj->release_storage();
}

Lfm2AudioSession::~Lfm2AudioSession() = default;

std::string Lfm2AudioSession::family() const {
    return kFamily;
}

runtime::VoiceTaskKind Lfm2AudioSession::task_kind() const {
    return task_.task;
}

runtime::RunMode Lfm2AudioSession::run_mode() const {
    return task_.mode;
}

void Lfm2AudioSession::prepare(const runtime::SessionPreparationRequest & request) {
    (void)request;
    mark_prepared();
}

Lfm2AudioSession::RequestOptions Lfm2AudioSession::parse_request_options(const runtime::TaskRequest & request) const {
    runtime::validate_spec_backed_request_options(request.options, require_contract(contract_), kModelName);

    RequestOptions out;
    out.max_tokens = runtime::parse_positive_i64_option(request.options, {"max_tokens"}, out.max_tokens);

    if (const auto language = runtime::find_option(request.options, {"language"});
        language.has_value() && *language != "auto" && *language != language_) {
        throw std::runtime_error("this LFM2-Audio checkpoint transcribes " + language_ + ", not " + *language);
    }

    return out;
}

// liquid-audio transcribes the whole input in one pass, which holds up to
// about a minute: on joined LibriSpeech test-clean clips its WER is 2% at
// 60 s, 9% at 90 s (dropped words) and over 80% from 120 s (repetition).
// auto therefore keeps input that fits one chunk whole, like liquid-audio,
// and splits longer audio at pauses found by the bundled Silero VAD, as the
// other ASR families do (vad mode forces that). That stays at 1-2% WER up to
// 180 s; silence between the spans is not transcribed, so it cannot come back
// as words. fixed cuts at the chunk length regardless (2.4-2.9%: words get cut).
std::vector<runtime::TimeSpan> Lfm2AudioSession::plan_chunks(
    const runtime::TaskRequest & request, const std::vector<float> & samples) {
    const auto total = static_cast<int64_t>(samples.size());
    const auto mode = audio::parse_audio_chunk_mode(request.options);
    if (mode == audio::AudioChunkMode::None) {
        return {{0, total}};
    }

    if (mode != audio::AudioChunkMode::Auto && mode != audio::AudioChunkMode::Fixed && mode != audio::AudioChunkMode::Vad) {
        throw std::runtime_error("LFM2-Audio supports audio_chunk_mode=auto, fixed, vad, or none");
    }

    const float seconds = audio::parse_audio_chunk_seconds_override(request.options).value_or(kDefaultChunkSeconds);
    if (!std::isfinite(seconds) || seconds < kMinChunkSeconds) {
        throw std::runtime_error("LFM2-Audio audio_chunk_seconds must be at least 1");
    }

    const auto chunk_samples = static_cast<int64_t>(std::llround(static_cast<double>(seconds) * kSampleRate));
    if (mode == audio::AudioChunkMode::Auto && total <= chunk_samples) {
        return {{0, total}};
    }

    // Without the bundled model, auto falls back to fixed chunks; vad requires it.
    if (mode == audio::AudioChunkMode::Fixed ||
        (mode == audio::AudioChunkMode::Auto && !io::is_existing_directory(vad_model_path_))) {
        return plan_fixed_chunks(total, chunk_samples);
    }

    const audio::VadAudioChunkOptions options{chunk_samples, kSampleRate / 2, kSampleRate / 4};
    return audio::plan_vad_audio_chunks(runtime::AudioBuffer{kSampleRate, 1, samples}, vad_session(), options);
}

runtime::IOfflineVoiceTaskSession & Lfm2AudioSession::vad_session() {
    if (vad_session_ == nullptr) {
        runtime::ModelLoadRequest load_request;
        load_request.model_path = vad_model_path_;
        vad_model_ = engine::models::silero_vad::load_silero_vad_model(load_request);
        auto session = vad_model_->create_task_session(
            {runtime::VoiceTaskKind::Vad, runtime::RunMode::Offline},
            runtime::SessionOptions{RuntimeSessionBase::options().backend, {}});
        auto * offline = dynamic_cast<runtime::IOfflineVoiceTaskSession *>(session.get());
        if (offline == nullptr) {
            throw std::runtime_error("LFM2-Audio VAD session does not support offline execution");
        }

        session.release();
        vad_session_.reset(offline);
    }

    return *vad_session_;
}

// A transcript that reaches max_tokens is cut off there and kept, with a
// warning, as liquid-audio's generate_sequential keeps what it generated when
// max_new_tokens runs out; the other chunks go on.
std::string Lfm2AudioSession::transcribe(
    const std::vector<float> & samples, const runtime::TimeSpan & span, const RequestOptions & options) {
    const std::vector<float> chunk(samples.begin() + span.start_sample, samples.begin() + span.end_sample);
    const auto features = features_.extract(chunk);
    debug_dump("mel.f32", features.values.data(), features.values.size() * sizeof(float));

    const auto audio = encoder_.encode(features);
    debug_dump("adapter.f32", audio.values.data(), audio.values.size() * sizeof(float));

    const auto prompt = prompt_.with_audio(audio.tokens);
    debug_dump("prompt_ids.i32", prompt.input_ids.data(), prompt.input_ids.size() * sizeof(int32_t));

    const auto result = backbone_.generate(prompt, audio, {options.max_tokens, prompt_.stop_token_ids});
    if (!result.stopped) {
        reached_max_tokens_ = true;
        warn("the transcript of " + seconds_at(span.start_sample) + "-" + seconds_at(span.end_sample) + " s reached max_tokens=" +
             std::to_string(options.max_tokens) + " and is cut off there; raise max_tokens for the rest");
    }

    debug_dump("prefill_logits.f32", result.prefill_logits.data(), result.prefill_logits.size() * sizeof(float));
    debug_dump("tokens.i32", result.tokens.data(), result.tokens.size() * sizeof(int32_t));
    debug::trace_log_scalar("lfm2_audio.session.audio_tokens", audio.tokens);
    debug::trace_log_scalar("lfm2_audio.session.generated_tokens", static_cast<int64_t>(result.tokens.size()));

    return tokenizer_.decode(result.tokens);
}

bool Lfm2AudioSession::reached_max_tokens() const {
    return reached_max_tokens_;
}

runtime::TaskResult Lfm2AudioSession::run(const runtime::TaskRequest & request) {
    require_prepared("LFM2-Audio run()");
    reached_max_tokens_ = false;
    if (!request.audio_input.has_value()) {
        throw std::runtime_error("LFM2-Audio run() requires audio_input");
    }

    const auto wall_start = std::chrono::steady_clock::now();
    const auto options = parse_request_options(request);
    const auto samples = lfm2_audio_mono_16k(*request.audio_input);

    std::string text;
    for (const auto & span : plan_chunks(request, samples)) {
        append_chunk_text(text, transcribe(samples, span, options));
    }

    runtime::TaskResult result;
    result.text_output = runtime::Transcript{text, language_};
    debug::timing_log_scalar("session.wall_ms", engine::debug::elapsed_ms(wall_start));
    return result;
}

std::shared_ptr<runtime::IVoiceModelLoader> make_lfm2_audio_loader() {
    class LoadedModel final : public runtime::ILoadedVoiceModel {
    public:
        explicit LoadedModel(std::shared_ptr<const Lfm2AudioAssets> assets)
            : assets_(std::move(assets)), contract_(runtime::require_model_contract(kFamily)) {}

        const runtime::ModelMetadata & metadata() const noexcept override { return contract_->metadata; }
        const runtime::CapabilitySet & capabilities() const noexcept override { return contract_->capabilities; }

        std::unique_ptr<runtime::IVoiceTaskSession> create_task_session(
            const runtime::TaskSpec & task, const runtime::SessionOptions & options) const override {
            return std::make_unique<Lfm2AudioSession>(task, options, assets_, contract_);
        }

    private:
        std::shared_ptr<const Lfm2AudioAssets> assets_;
        std::shared_ptr<const engine::model_spec::ModelContract> contract_;
    };

    // Not spec-backed: the package is a directory of several llama.cpp GGUFs
    // without an embedded model spec, so the loader is hand-written like
    // make_auk_loader (community_models/auk/session.cpp).
    class Loader final : public runtime::IVoiceModelLoader {
    public:
        std::string family() const override { return kFamily; }

        bool can_load(const runtime::ModelLoadRequest & request) const override {
            if (request.family_hint && *request.family_hint != kFamily) {
                return false;
            }

            // An incomplete package is still claimed, so load() reports what
            // is missing instead of the registry finding no loader at all.
            return has_lfm2_audio_component(request.model_path);
        }

        runtime::ModelInspection inspect(const runtime::ModelLoadRequest & request) const override {
            const auto assets = load_lfm2_audio_assets(request.model_path);
            const auto contract = runtime::require_model_contract(kFamily);

            runtime::ModelInspection inspection;
            inspection.model_root = assets->model_root;
            inspection.metadata = contract->metadata;
            inspection.capabilities = contract->capabilities;
            inspection.cli = contract->cli;
            inspection.discovered_configs =
                runtime::discover_named_assets(inspection.model_root, inspection.metadata.config_candidates);
            inspection.discovered_weights =
                runtime::discover_named_assets(inspection.model_root, inspection.metadata.weight_candidates);

            return inspection;
        }

        std::unique_ptr<runtime::ILoadedVoiceModel> load(const runtime::ModelLoadRequest & request) const override {
            return std::make_unique<LoadedModel>(load_lfm2_audio_assets(request.model_path));
        }

        runtime::CapabilitySet advertised_capabilities() const override {
            return runtime::require_model_contract(kFamily)->capabilities;
        }
    };
    return std::make_shared<Loader>();
}

}  // namespace engine::community_models::lfm2_audio
