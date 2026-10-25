#include "engine/community_models/granite5asr/session.h"

#include "engine/community_models/granite5asr/ctc_timestamps.h"
#include "engine/framework/audio/chunking.h"
#include "engine/framework/audio/conversion.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/debug/trace.h"
#include "engine/framework/io/filesystem.h"
#include "engine/framework/io/json.h"
#include "engine/framework/io/text.h"
#include "engine/framework/model_spec/package.h"
#include "engine/framework/runtime/options.h"
#include "engine/framework/runtime/spec_backed_model.h"
#include "engine/framework/runtime/registry.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace engine::community_models::granite5asr {
namespace json = engine::io::json;

std::vector<std::string> languages_for_tokens(const std::vector<std::string> & tokens) {
    const bool english_only_blank = !tokens.empty() && tokens.front() == "<|blank|>";
    bool has_french = false;
    bool has_japanese = false;
    if (!english_only_blank) {
        for (const auto & token : tokens) {
            if (token == "<|fr|>") {
                has_french = true;
            } else if (token == "<|ja|>") {
                has_japanese = true;
            }
        }
    }
    if (english_only_blank || !has_french || !has_japanese) {
        return {"en"};
    }
    return {"en", "fr", "de", "es", "pt", "it", "ja"};
}

namespace {

using Clock = std::chrono::steady_clock;
constexpr float kDefaultChunkSeconds = 30.0f;
constexpr size_t kDefaultGraphArenaBytes = 1024ull * 1024ull * 1024ull;

std::filesystem::path default_spec_path() {
    return engine::model_spec::default_spec_path("granite5asr");
}

std::shared_ptr<const Granite5ASRAssets> require_assets(
    std::shared_ptr<const Granite5ASRAssets> assets) {
    if (assets == nullptr) {
        throw std::runtime_error("Granite 5 ASR session requires assets");
    }
    return assets;
}

assets::TensorStorageType parse_weight_storage(
    const runtime::SessionOptions & options,
    const std::string & family) {
    const std::string key = family + ".weight_type";
    const auto it = options.options.find(key);
    if (it == options.options.end()) {
        return assets::TensorStorageType::Native;
    }
    const auto type = assets::parse_tensor_storage_type(it->second);
    if (type == assets::TensorStorageType::Native ||
        type == assets::TensorStorageType::F32 ||
        type == assets::TensorStorageType::F16 ||
        type == assets::TensorStorageType::BF16 ||
        type == assets::TensorStorageType::Q8_0) {
        return type;
    }
    throw std::runtime_error(key + " supports only native, f32, f16, bf16, and q8_0");
}

size_t parse_graph_arena_bytes(
    const runtime::SessionOptions & options,
    const std::string & family) {
    const std::string key = family + ".encoder_graph_arena_mb";
    const auto it = options.options.find(key);
    if (it == options.options.end()) {
        return kDefaultGraphArenaBytes;
    }
    try {
        const int64_t mb = std::stoll(it->second);
        if (mb > 0) {
            return static_cast<size_t>(mb) * 1024ull * 1024ull;
        }
    } catch (...) {}
    return kDefaultGraphArenaBytes;
}

std::string parse_vad_model_path(
    const runtime::SessionOptions & options,
    const std::string & family) {
    const std::string key = family + ".vad_model_path";
    const auto it = options.options.find(key);
    if (it != options.options.end() && !it->second.empty()) {
        return it->second;
    }
    return "assets/framework/models/silero_vad";
}

const std::string kLanguageOptionHelp =
    "Accepted and ignored. Granite has no language input. A result is labelled en unless decoding emits a trained language token such as <|fr|>.";

void append_chunk_transcript(std::string & merged, std::string chunk_text) {
    chunk_text = engine::io::trim_ascii_whitespace(std::move(chunk_text));
    if (chunk_text.empty()) {
        return;
    }
    if (!merged.empty()) {
        const char last = merged.back();
        const char first = chunk_text.front();
        if (last != ' ' && first != ' ') {
            merged.push_back(' ');
        }
    }
    merged += chunk_text;
}

class Granite5ASRLoader final : public runtime::IVoiceModelLoader {
public:
    std::string family() const override {
        return "granite5asr";
    }

    std::vector<std::string> family_aliases() const override {
        return {"granite_speech5_asr", "granite_speech", "granite_speech5_ctc"};
    }

    runtime::CapabilitySet advertised_capabilities() const override {
        runtime::CapabilitySet out;
        out.supported_tasks = {
            {runtime::VoiceTaskKind::Asr, {runtime::RunMode::Offline, runtime::RunMode::Streaming}},
        };
        out.languages = {"en"};
        out.supports_timestamps = true;
        return out;
    }

    bool can_load(const runtime::ModelLoadRequest & request) const override {
        if (request.family_hint.has_value()) {
            const auto & hint = *request.family_hint;
            if (hint != family()) {
                const auto aliases = family_aliases();
                if (std::find(aliases.begin(), aliases.end(), hint) == aliases.end()) {
                    return false;
                }
            }
        }
        try {
            const auto resources = engine::model_spec::load_resource_bundle(request.model_path, default_spec_path());
            const auto config_root = resources.parse_json("config");
            const auto model_type = json::optional_string(config_root, "model_type", "");
            return model_type == "granite_speech5_ctc" || model_type == "granite5asr" || model_type == "granite_speech";
        } catch (...) {
            return false;
        }
    }

    runtime::ModelInspection inspect(const runtime::ModelLoadRequest & request) const override {
        const auto resources = engine::model_spec::load_resource_bundle(
            request.model_path,
            default_spec_path());
        runtime::ModelInspection inspection;
        inspection.cli.request_options = {
            {"language", "code", kLanguageOptionHelp, false, "en"},
            {"audio_chunk_mode", "auto|fixed|vad|none", "Offline audio chunking mode: auto, fixed, vad, or none. VAD requires a regular model file; otherwise the planner uses fixed chunks. Default: auto.", false, "auto"},
            {"audio_chunk_duration_sec", "float", "Positive maximum audio chunk duration for fixed and VAD chunking. Default: 30.", false, "30", "0.001"},
        };
        inspection.cli.session_options = {
            {"granite5asr.weight_type", "native|f32|f16|bf16|q8_0", "Shared model weight storage type. Default: native.", false, "native"},
            {"granite5asr.encoder_graph_arena_mb", "int", "Encoder graph arena size in MB. Positive MiB allocation; invalid/nonpositive values resolve to 1024 MiB. Default: 1024.", false, "1024", "1"},
            {"granite5asr.vad_model_path", "string", "Path to a VAD weight file of any built-in VAD family (e.g. silero_vad_16k.safetensors), not a directory; VAD runs only when this path is a regular file."},
        };
        inspection.model_root = resources.model_root();
        inspection.metadata.family = family();
        inspection.metadata.variant = "470m-turboctc";
        inspection.metadata.description = "IBM Granite Speech 5.0 470M TurboCTC ASR model.";
        inspection.capabilities.supported_tasks = {
            {runtime::VoiceTaskKind::Asr, {runtime::RunMode::Offline, runtime::RunMode::Streaming}},
        };
        inspection.capabilities.languages = {"en"};
        try {
            const auto tokenizer = engine::tokenizers::load_huggingface_tokenizer_json(
                resources.require_file("tokenizer_json"));
            inspection.capabilities.languages = languages_for_tokens(tokenizer->id_to_token());
        } catch (...) {
        }
        inspection.capabilities.supports_timestamps = true;
        inspection.discovered_configs = runtime::discover_named_assets_from_package_spec(
            request.model_path,
            default_spec_path(),
            engine::model_spec::ResourceKind::Files);
        inspection.discovered_weights = runtime::discover_named_assets_from_package_spec(
            request.model_path,
            default_spec_path(),
            engine::model_spec::ResourceKind::Tensors);
        return inspection;
    }

    std::unique_ptr<runtime::ILoadedVoiceModel> load(const runtime::ModelLoadRequest & request) const override {
        return load_granite5asr_model(request);
    }
};

}  // namespace

Granite5ASRSessionBase::Granite5ASRSessionBase(
    runtime::TaskSpec task,
    runtime::SessionOptions options,
    std::shared_ptr<const Granite5ASRAssets> assets)
    : RuntimeSessionBase(std::move(options)),
      task_(std::move(task)),
      assets_(require_assets(std::move(assets))),
      frontend_(assets_),
      vad_model_path_(parse_vad_model_path(RuntimeSessionBase::options(), family_impl())) {
    if (task_.task != runtime::VoiceTaskKind::Asr) {
        throw std::runtime_error("Granite 5 ASR only supports VoiceTaskKind::Asr");
    }
    const auto storage_type = parse_weight_storage(RuntimeSessionBase::options(), family_impl());
    const auto arena_bytes = parse_graph_arena_bytes(RuntimeSessionBase::options(), family_impl());
    encoder_ = std::unique_ptr<Granite5ConformerEncoderRuntime>(
        new Granite5ConformerEncoderRuntime(
            assets_,
            execution_context(),
            storage_type,
            arena_bytes));
}

Granite5ASRSessionBase::~Granite5ASRSessionBase() = default;

std::string Granite5ASRSessionBase::family_impl() const {
    return "granite5asr";
}

runtime::VoiceTaskKind Granite5ASRSessionBase::task_kind_impl() const {
    return task_.task;
}

runtime::RunMode Granite5ASRSessionBase::run_mode_impl() const {
    return task_.mode;
}

runtime::IOfflineVoiceTaskSession & Granite5ASRSessionBase::vad_session() {
    if (vad_session_ == nullptr) {
        runtime::ModelLoadRequest load_request;
        load_request.model_path = vad_model_path_;
        vad_model_ = runtime::load_vad_model(load_request);
        auto session = vad_model_->create_task_session(
            runtime::TaskSpec{runtime::VoiceTaskKind::Vad, runtime::RunMode::Offline},
            runtime::SessionOptions{options().backend, {}});
        auto * offline = dynamic_cast<runtime::IOfflineVoiceTaskSession *>(session.get());
        if (offline == nullptr) {
            throw std::runtime_error("Granite 5 ASR internal VAD session does not support offline execution");
        }
        session.release();
        vad_session_.reset(offline);
    }
    return *vad_session_;
}

std::vector<runtime::WordTimestamp> words_from_ctc_pieces(
    const engine::tokenizers::HuggingFaceTokenizerJson & tokenizer,
    const std::vector<CtcPiece> & pieces,
    int64_t samples_per_frame,
    int64_t sample_offset,
    int64_t max_end_sample,
    int64_t model_sample_rate,
    int64_t input_sample_rate) {
    std::vector<DecodedCtcPiece> decoded;
    decoded.reserve(pieces.size());
    for (const auto & piece : pieces) {
        decoded.push_back(DecodedCtcPiece{
            tokenizer.decode_ids(std::vector<int32_t>{piece.id}),
            piece.start_frame,
            piece.end_frame,
            piece.log_prob_sum,
            piece.frames,
        });
    }
    return words_from_decoded_ctc_pieces(
        decoded, samples_per_frame, sample_offset, max_end_sample, model_sample_rate, input_sample_rate);
}

TimedTranscript Granite5ASRSessionBase::transcribe_audio(
    const runtime::AudioBuffer & audio,
    const std::unordered_map<std::string, std::string> & options) {
    TimedTranscript empty;
    empty.transcript = {"", "en"};
    if (audio.samples.empty()) {
        return empty;
    }
    const int64_t frame_samples = samples_per_ctc_frame(assets_->config);
    const int64_t audio_samples = static_cast<int64_t>(audio.samples.size() / std::max(1, audio.channels));

    auto transcribe_clip = [this, frame_samples, audio_samples](const runtime::AudioBuffer & clip, int64_t sample_offset) -> TimedTranscript {
        TimedTranscript out;
        if (clip.samples.empty() || assets_->tokenizer == nullptr) {
            return out;
        }
        const auto features = frontend_.extract(clip);
        const auto frames = encoder_->transcribe_features(features);
        const auto pieces = ctc_greedy_pieces(frames, assets_->config.blank_token_id);
        if (pieces.empty()) {
            return out;
        }
        const auto & token_table = assets_->tokenizer->id_to_token();
        std::vector<CtcPiece> spoken;
        spoken.reserve(pieces.size());
        for (const auto & piece : pieces) {
            std::string token;
            if (piece.id >= 0 && static_cast<size_t>(piece.id) < token_table.size()) {
                token = token_table[static_cast<size_t>(piece.id)];
            }
            const std::string language = language_code_from_token(token);
            if (!language.empty()) {
                if (out.transcript.language.empty()) {
                    out.transcript.language = language;
                }
                continue;
            }
            spoken.push_back(piece);
        }
        if (spoken.empty()) {
            return out;
        }
        std::vector<int32_t> ids;
        ids.reserve(spoken.size());
        for (const auto & piece : spoken) {
            ids.push_back(piece.id);
        }
        out.transcript.text = assets_->tokenizer->decode_ids(ids);
        const int64_t clip_samples = static_cast<int64_t>(clip.samples.size() / std::max(1, clip.channels));
        const int64_t max_end_sample = std::min(audio_samples, sample_offset + clip_samples);
        out.words = words_from_ctc_pieces(
            *assets_->tokenizer,
            spoken,
            frame_samples,
            sample_offset,
            max_end_sample,
            assets_->config.frontend.sample_rate,
            clip.sample_rate);
        return out;
    };

    auto take_clip = [&](TimedTranscript && clip) {
        clip.transcript.text = engine::io::trim_ascii_whitespace(std::move(clip.transcript.text));
        return clip;
    };
    auto append_clip = [&](TimedTranscript & merged, TimedTranscript && clip) {
        append_chunk_transcript(merged.transcript.text, std::move(clip.transcript.text));
        merged.words.insert(merged.words.end(), clip.words.begin(), clip.words.end());
        if (merged.transcript.language.empty() && !clip.transcript.language.empty()) {
            merged.transcript.language = std::move(clip.transcript.language);
        }
    };
    auto label_language = [](TimedTranscript & clip) {
        if (clip.transcript.language.empty()) {
            clip.transcript.language = "en";
        }
    };

    const auto mode = engine::audio::parse_audio_chunk_mode(options);
    if (mode == engine::audio::AudioChunkMode::None) {
        auto clip = take_clip(transcribe_clip(audio, 0));
        label_language(clip);
        return clip;
    }

    const float duration_sec = static_cast<float>(audio.samples.size()) / static_cast<float>(std::max(1, audio.sample_rate * audio.channels));
    const float chunk_duration_sec = engine::audio::parse_audio_chunk_seconds_override(options).value_or(kDefaultChunkSeconds);

    if (duration_sec <= chunk_duration_sec) {
        auto clip = take_clip(transcribe_clip(audio, 0));
        label_language(clip);
        return clip;
    }

    TimedTranscript merged;
    if ((mode == engine::audio::AudioChunkMode::Vad || mode == engine::audio::AudioChunkMode::Auto) &&
        engine::io::is_existing_file(vad_model_path_)) {
        try {
            const auto vad_options = engine::audio::VadAudioChunkOptions{
                static_cast<int64_t>(std::llround(static_cast<double>(chunk_duration_sec) * static_cast<double>(audio.sample_rate))),
                static_cast<int64_t>(std::llround(0.5 * static_cast<double>(audio.sample_rate))),
                static_cast<int64_t>(std::llround(0.25 * static_cast<double>(audio.sample_rate))),
            };
            const auto spans = engine::audio::plan_vad_audio_chunks(audio, vad_session(), vad_options);
            for (const auto & span : spans) {
                const auto chunk_buf = engine::audio::slice_audio_buffer(audio, span);
                append_clip(merged, transcribe_clip(chunk_buf, span.start_sample));
            }
            merged.transcript.text = engine::io::trim_ascii_whitespace(std::move(merged.transcript.text));
            label_language(merged);
            return merged;
        } catch (...) {
            merged.words.clear();
            merged.transcript.text.clear();
        }
    }

    const int64_t samples = static_cast<int64_t>(std::llround(static_cast<double>(chunk_duration_sec) * static_cast<double>(audio.sample_rate)));
    const int64_t frames = static_cast<int64_t>(audio.samples.size() / std::max(1, audio.channels));
    const auto chunks = engine::audio::plan_audio_chunks(
        frames,
        {samples, samples, engine::audio::AudioChunkPadMode::Zero, engine::audio::AudioChunkTailAlignment::Start, 0});
    for (const auto & chunk : chunks) {
        runtime::TimeSpan span{chunk.output_start_sample, chunk.output_start_sample + chunk.valid_samples};
        const auto chunk_buf = engine::audio::slice_audio_buffer(audio, span);
        append_clip(merged, transcribe_clip(chunk_buf, span.start_sample));
    }
    merged.transcript.text = engine::io::trim_ascii_whitespace(std::move(merged.transcript.text));
    label_language(merged);
    return merged;
}

// ----------------- Offline Session -----------------

Granite5ASROfflineSession::Granite5ASROfflineSession(
    runtime::TaskSpec task,
    runtime::SessionOptions options,
    std::shared_ptr<const Granite5ASRAssets> assets)
    : Granite5ASRSessionBase(std::move(task), std::move(options), std::move(assets)) {}

std::string Granite5ASROfflineSession::family() const {
    return family_impl();
}

runtime::VoiceTaskKind Granite5ASROfflineSession::task_kind() const {
    return task_kind_impl();
}

runtime::RunMode Granite5ASROfflineSession::run_mode() const {
    return run_mode_impl();
}

void Granite5ASROfflineSession::prepare(const runtime::SessionPreparationRequest & request) {
    mark_prepared();
}

runtime::TaskResult Granite5ASROfflineSession::run(const runtime::TaskRequest & request) {
    require_prepared("Granite 5 ASR run()");
    if (!request.audio_input.has_value()) {
        throw std::runtime_error("Granite 5 ASR run() requires audio_input");
    }
    const auto wall_start = Clock::now();
    const auto transcript = transcribe_audio(*request.audio_input, request.options);
    runtime::TaskResult result;
    result.text_output = transcript.transcript;
    result.word_timestamps = transcript.words;
    engine::debug::timing_log_scalar("session.wall_ms", engine::debug::elapsed_ms(wall_start));
    return result;
}

// ----------------- Streaming Session -----------------

Granite5ASRStreamingSession::Granite5ASRStreamingSession(
    runtime::TaskSpec task,
    runtime::SessionOptions options,
    std::shared_ptr<const Granite5ASRAssets> assets)
    : Granite5ASRSessionBase(std::move(task), std::move(options), std::move(assets)) {}

std::string Granite5ASRStreamingSession::family() const {
    return family_impl();
}

runtime::VoiceTaskKind Granite5ASRStreamingSession::task_kind() const {
    return task_kind_impl();
}

runtime::RunMode Granite5ASRStreamingSession::run_mode() const {
    return run_mode_impl();
}

void Granite5ASRStreamingSession::prepare(const runtime::SessionPreparationRequest & request) {
    mark_prepared();
}

runtime::StreamingPolicy Granite5ASRStreamingSession::streaming_policy() const {
    runtime::StreamingPolicy policy;
    policy.input = runtime::StreamingInputKind::AudioChunks;
    policy.output = runtime::StreamingOutputKind::FinalResult;
    policy.preferred_audio_chunk_samples = 512;
    return policy;
}

void Granite5ASRStreamingSession::start_stream(const runtime::TaskRequest & request) {
    require_prepared("Granite 5 ASR start_stream()");
    streaming_request_ = request;
    streaming_audio_ = runtime::AudioBuffer{};
    streaming_audio_.sample_rate = 16000;
    streaming_audio_.channels = 1;
    stream_input_rate_ = 0;
    stream_input_samples_ = 0;
}

void Granite5ASRStreamingSession::set_stream_event_sink(runtime::StreamEventCallback sink) {
    stream_event_sink_ = std::move(sink);
}

void Granite5ASRStreamingSession::reset() {
    streaming_audio_.samples.clear();
    stream_input_rate_ = 0;
    stream_input_samples_ = 0;
}

runtime::StreamEvent Granite5ASRStreamingSession::process_audio_chunk(const runtime::AudioChunk & chunk) {
    require_prepared("Granite 5 ASR process_audio_chunk()");
    if (!chunk.samples.empty()) {
        const int channels = std::max(1, chunk.channels);
        const int64_t frames = static_cast<int64_t>(chunk.samples.size() / static_cast<size_t>(channels));
        const int32_t rate = chunk.sample_rate > 0 ? chunk.sample_rate : 16000;
        if (stream_input_rate_ == 0) {
            stream_input_rate_ = rate;
        }
        stream_input_samples_ += rate == stream_input_rate_
            ? frames
            : scale_sample_to_rate(frames, rate, stream_input_rate_);
        const auto mono = engine::audio::convert_interleaved_audio_to_mono_linear_resampled(
            chunk.samples,
            chunk.sample_rate,
            chunk.channels,
            16000);
        streaming_audio_.samples.insert(streaming_audio_.samples.end(), mono.begin(), mono.end());
    }
    return {};
}

void Granite5ASRStreamingSession::remap_stream_words(std::vector<runtime::WordTimestamp> & words) const {
    if (stream_input_rate_ <= 0 || stream_input_rate_ == streaming_audio_.sample_rate) {
        return;
    }
    for (auto & word : words) {
        word.span.start_sample = scale_sample_to_rate(
            word.span.start_sample, streaming_audio_.sample_rate, stream_input_rate_);
        word.span.end_sample = scale_sample_to_rate(
            word.span.end_sample, streaming_audio_.sample_rate, stream_input_rate_);
        if (stream_input_samples_ > 0 && word.span.end_sample > stream_input_samples_) {
            word.span.end_sample = stream_input_samples_;
        }
        if (word.span.end_sample <= word.span.start_sample) {
            word.span.end_sample = word.span.start_sample + 1;
        }
    }
}

runtime::TaskResult Granite5ASRStreamingSession::finish_stream() {
    require_prepared("Granite 5 ASR finish_stream()");
    auto transcript = transcribe_audio(streaming_audio_, streaming_request_.options);
    remap_stream_words(transcript.words);
    runtime::TaskResult result;
    result.text_output = std::move(transcript.transcript);
    result.word_timestamps = std::move(transcript.words);
    return result;
}

runtime::TaskResult Granite5ASRStreamingSession::finalize() {
    return finish_stream();
}

// ----------------- Loaded Model & Loader -----------------

Granite5ASRLoadedModel::Granite5ASRLoadedModel(
    runtime::ModelMetadata metadata,
    runtime::CapabilitySet capabilities,
    std::shared_ptr<const Granite5ASRAssets> assets)
    : metadata_(std::move(metadata)),
      capabilities_(std::move(capabilities)),
      assets_(std::move(assets)) {}

const runtime::ModelMetadata & Granite5ASRLoadedModel::metadata() const noexcept {
    return metadata_;
}

const runtime::CapabilitySet & Granite5ASRLoadedModel::capabilities() const noexcept {
    return capabilities_;
}

std::unique_ptr<runtime::IVoiceTaskSession> Granite5ASRLoadedModel::create_task_session(
    const runtime::TaskSpec & task,
    const runtime::SessionOptions & options) const {
    if (task.mode == runtime::RunMode::Streaming) {
        return std::make_unique<Granite5ASRStreamingSession>(task, options, assets_);
    }
    return std::make_unique<Granite5ASROfflineSession>(task, options, assets_);
}

std::unique_ptr<Granite5ASRLoadedModel> load_granite5asr_model(
    const runtime::ModelLoadRequest & request) {
    auto assets = load_granite5asr_assets(request.model_path);

    runtime::ModelMetadata metadata;
    metadata.family = "granite5asr";
    metadata.variant = "470m-turboctc";
    metadata.description = "IBM Granite Speech 5.0 470M TurboCTC ASR model.";

    runtime::CapabilitySet capabilities;
    capabilities.supported_tasks = {
        {runtime::VoiceTaskKind::Asr, {runtime::RunMode::Offline, runtime::RunMode::Streaming}},
    };
    capabilities.languages = assets->tokenizer != nullptr
        ? languages_for_tokens(assets->tokenizer->id_to_token())
        : std::vector<std::string>{"en"};
    capabilities.supports_timestamps = true;

    return std::make_unique<Granite5ASRLoadedModel>(
        std::move(metadata),
        std::move(capabilities),
        std::move(assets));
}

std::shared_ptr<runtime::IVoiceModelLoader> make_granite5asr_loader() {
    return std::make_shared<Granite5ASRLoader>();
}

}  // namespace engine::community_models::granite5asr
