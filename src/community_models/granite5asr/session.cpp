#include "engine/community_models/granite5asr/session.h"

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
#include "engine/models/silero_vad/session.h"

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

std::vector<int32_t> ctc_greedy_collapse(
    const std::vector<int32_t> & raw_ids,
    int64_t blank_id) {
    std::vector<int32_t> collapsed;
    int32_t prev = -1;
    for (int32_t id : raw_ids) {
        if (id != prev) {
            if (id != static_cast<int32_t>(blank_id)) {
                collapsed.push_back(id);
            }
            prev = id;
        }
    }
    return collapsed;
}

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
        out.supports_timestamps = false;
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
        inspection.model_root = resources.model_root();
        inspection.metadata.family = family();
        inspection.metadata.variant = "470m-turboctc";
        inspection.metadata.description = "IBM Granite Speech 5.0 470M TurboCTC ASR model.";
        inspection.capabilities.supported_tasks = {
            {runtime::VoiceTaskKind::Asr, {runtime::RunMode::Offline, runtime::RunMode::Streaming}},
        };
        inspection.capabilities.languages = {"en"};
        inspection.capabilities.supports_timestamps = false;
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
    encoder_ = std::unique_ptr<Granite5EncoderRuntime>(
        new Granite5EncoderRuntime(
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
        vad_model_ = engine::models::silero_vad::load_silero_vad_model(load_request);
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

runtime::Transcript Granite5ASRSessionBase::transcribe_audio(
    const runtime::AudioBuffer & audio,
    const std::unordered_map<std::string, std::string> & options) {
    if (audio.samples.empty()) {
        return {"", "en"};
    }

    auto transcribe_clip = [this](const runtime::AudioBuffer & clip) -> std::string {
        if (clip.samples.empty()) {
            return "";
        }
        const auto features = frontend_.extract(clip);
        const auto raw_tokens = encoder_->transcribe_features(features);
        const auto collapsed = ctc_greedy_collapse(raw_tokens, assets_->config.blank_token_id);
        if (collapsed.empty() || assets_->tokenizer == nullptr) {
            return "";
        }
        return assets_->tokenizer->decode_ids(collapsed);
    };

    const auto mode = engine::audio::parse_audio_chunk_mode(options);
    if (mode == engine::audio::AudioChunkMode::None) {
        return {engine::io::trim_ascii_whitespace(transcribe_clip(audio)), "en"};
    }

    const float duration_sec = static_cast<float>(audio.samples.size()) / static_cast<float>(std::max(1, audio.sample_rate * audio.channels));
    const float chunk_duration_sec = engine::audio::parse_audio_chunk_seconds_override(options).value_or(kDefaultChunkSeconds);

    if (duration_sec <= chunk_duration_sec) {
        return {engine::io::trim_ascii_whitespace(transcribe_clip(audio)), "en"};
    }

    std::string full_text;
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
                append_chunk_transcript(full_text, transcribe_clip(chunk_buf));
            }
            return {engine::io::trim_ascii_whitespace(std::move(full_text)), "en"};
        } catch (...) {
            // Fallback to fixed chunking
        }
    }

    // Fixed chunking
    const int64_t samples = static_cast<int64_t>(std::llround(static_cast<double>(chunk_duration_sec) * static_cast<double>(audio.sample_rate)));
    const int64_t frames = static_cast<int64_t>(audio.samples.size() / std::max(1, audio.channels));
    const auto chunks = engine::audio::plan_audio_chunks(
        frames,
        {samples, samples, engine::audio::AudioChunkPadMode::Zero, engine::audio::AudioChunkTailAlignment::Start, 0});
    for (const auto & chunk : chunks) {
        runtime::TimeSpan span{chunk.output_start_sample, chunk.output_start_sample + chunk.valid_samples};
        const auto chunk_buf = engine::audio::slice_audio_buffer(audio, span);
        append_chunk_transcript(full_text, transcribe_clip(chunk_buf));
    }

    return {engine::io::trim_ascii_whitespace(std::move(full_text)), "en"};
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
    result.text_output = transcript;
    engine::debug::timing_log_scalar("session.wall_ms", engine::debug::elapsed_ms(wall_start));
    return result;
}

// ----------------- Streaming Session -----------------

Granite5ASRStreamingSession::Granite5ASRStreamingSession(
    runtime::TaskSpec task,
    runtime::SessionOptions options,
    std::shared_ptr<const Granite5ASRAssets> assets)
    : Granite5ASRSessionBase(std::move(task), std::move(options), std::move(assets)) {
    // Chunked-streaming geometry (publisher chunked TurboCTC recipe): every
    // center chunk carries its left context through the encoder and is decoded
    // immediately, so partials stream per chunk. Defaults tuned for END-OF-TURN
    // latency: a 1 s center keeps the final flush window small. Raising
    // center_chunk_sec to 2 cuts total encoder compute ~30% at equal accuracy
    // (fewer, larger windows) but adds ~20-25% end-of-turn latency at 1 thread;
    // left contexts below 2 s duplicate words across window boundaries.
    float center_sec = 1.0f;
    float left_sec = 2.0f;
    const auto & opts = RuntimeSessionBase::options().options;
    const std::string family = family_impl();
    auto parse_positive = [](const std::string & value, float fallback) {
        try {
            const float parsed = std::stof(value);
            if (parsed > 0.0f) {
                return parsed;
            }
        } catch (...) {}
        return fallback;
    };
    if (const auto it = opts.find(family + ".center_chunk_sec"); it != opts.end()) {
        center_sec = parse_positive(it->second, center_sec);
    }
    if (const auto it = opts.find(family + ".left_context_sec"); it != opts.end()) {
        left_sec = parse_positive(it->second, left_sec);
    }
    stream_center_samples_ = static_cast<int64_t>(center_sec * 16000.0f);
    stream_left_context_samples_ = static_cast<int64_t>(left_sec * 16000.0f);
}

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
    policy.preferred_audio_chunk_samples = stream_center_samples_ > 0
        ? stream_center_samples_
        : 16000;
    policy.preferred_audio_chunk_seconds = stream_center_samples_ > 0
        ? static_cast<double>(stream_center_samples_) / 16000.0
        : 1.0;
    return policy;
}

void Granite5ASRStreamingSession::start_stream(const runtime::TaskRequest & request) {
    require_prepared("Granite 5 ASR start_stream()");
    streaming_request_ = request;
    streaming_audio_ = runtime::AudioBuffer{};
    streaming_audio_.sample_rate = 16000;
    streaming_audio_.channels = 1;
    stream_next_center_start_ = 0;
    stream_last_raw_token_ = -1;
    stream_collapsed_ids_.clear();
    stream_emitted_text_.clear();
    stream_finalized_ = false;
}

void Granite5ASRStreamingSession::set_stream_event_sink(runtime::StreamEventCallback sink) {
    stream_event_sink_ = std::move(sink);
}

void Granite5ASRStreamingSession::reset() {
    streaming_audio_.samples.clear();
    stream_next_center_start_ = 0;
    stream_last_raw_token_ = -1;
    stream_collapsed_ids_.clear();
    stream_emitted_text_.clear();
    stream_finalized_ = false;
}

bool Granite5ASRStreamingSession::decode_next_center_window(bool flush_tail, std::string & delta_out) {
    if (stream_finalized_) {
        return false;
    }
    const int64_t total = static_cast<int64_t>(streaming_audio_.samples.size());
    const int64_t center_end = std::min<int64_t>(
        stream_next_center_start_ + stream_center_samples_, total);
    if (center_end <= stream_next_center_start_) {
        return false; // nothing new since the last window
    }
    if (!flush_tail && center_end < stream_next_center_start_ + stream_center_samples_) {
        return false; // wait for a full center chunk (low-latency default)
    }

    const int64_t window_start = std::max<int64_t>(
        0, stream_next_center_start_ - stream_left_context_samples_);
    runtime::AudioBuffer window;
    window.sample_rate = 16000;
    window.channels = 1;
    window.samples.assign(
        streaming_audio_.samples.begin() + static_cast<std::ptrdiff_t>(window_start),
        streaming_audio_.samples.begin() + static_cast<std::ptrdiff_t>(center_end));

    const auto features = frontend_.extract_waveform(window.samples);
    const auto raw_tokens = encoder_->transcribe_features(features);
    if (!raw_tokens.empty()) {
        // Map the center region to CTC frames. One CTC frame covers
        // hop * stack * encoder-stride samples: the frontend stacks `stack_factor`
        // mel frames per feature, and each subsample layer halves the encoder
        // sequence (subsample_layers = {0, 1} -> stride 4 total). One frame of
        // slack at the start avoids clipping a word onset; the continuous
        // collapse below absorbs the boundary repeat.
        int64_t encoder_stride = 1;
        for (const int64_t layer_idx : assets_->config.encoder.subsample_layers) {
            (void)layer_idx;
            encoder_stride *= 2;
        }
        const int64_t frame_samples = assets_->config.frontend.hop_length *
            assets_->config.frontend.stack_factor * encoder_stride;
        int64_t start_frame = std::max<int64_t>(
            0, (stream_next_center_start_ - window_start) / frame_samples - 1);
        const int64_t end_frame = std::min<int64_t>(
            static_cast<int64_t>(raw_tokens.size()),
            (center_end - window_start + frame_samples - 1) / frame_samples);
        for (int64_t f = start_frame; f < end_frame; ++f) {
            const int32_t id = raw_tokens[static_cast<size_t>(f)];
            if (id != stream_last_raw_token_) {
                if (id != static_cast<int32_t>(assets_->config.blank_token_id)) {
                    stream_collapsed_ids_.push_back(id);
                }
                stream_last_raw_token_ = id;
            }
        }
    }
    stream_next_center_start_ = center_end;

    if (!stream_collapsed_ids_.empty() && assets_->tokenizer != nullptr) {
        const auto current_text = assets_->tokenizer->decode_ids(stream_collapsed_ids_);
        if (current_text.size() > stream_emitted_text_.size() &&
            current_text.compare(0, stream_emitted_text_.size(), stream_emitted_text_) == 0) {
            delta_out += current_text.substr(stream_emitted_text_.size());
            stream_emitted_text_ = current_text;
        } else if (current_text != stream_emitted_text_) {
            // CTC revised earlier text: re-emit the full transcript (append-style
            // clients surface a repeat; the final result stays correct).
            delta_out += current_text;
            stream_emitted_text_ = current_text;
        }
    }
    return true;
}

runtime::StreamEvent Granite5ASRStreamingSession::process_audio_chunk(const runtime::AudioChunk & chunk) {
    require_prepared("Granite 5 ASR process_audio_chunk()");
    if (!chunk.samples.empty()) {
        const auto mono = engine::audio::convert_interleaved_audio_to_mono_linear_resampled(
            chunk.samples,
            chunk.sample_rate,
            chunk.channels,
            16000);
        streaming_audio_.samples.insert(streaming_audio_.samples.end(), mono.begin(), mono.end());
    }

    runtime::StreamEvent event;
    event.is_final = false;
    std::string delta;
    while (decode_next_center_window(/*flush_tail=*/false, delta)) {
    }
    if (!delta.empty()) {
        event.partial_text = runtime::Transcript{delta, "en"};
        if (stream_event_sink_) {
            stream_event_sink_(event);
            return {};
        }
    }
    return event;
}

runtime::TaskResult Granite5ASRStreamingSession::finish_stream() {
    require_prepared("Granite 5 ASR finish_stream()");
    std::string delta;
    while (decode_next_center_window(/*flush_tail=*/true, delta)) {
    }
    (void)delta;

    runtime::TaskResult result;
    if (stream_collapsed_ids_.empty() || assets_->tokenizer == nullptr) {
        result.text_output = runtime::Transcript{"", "en"};
    } else {
        result.text_output = runtime::Transcript{
            engine::io::trim_ascii_whitespace(assets_->tokenizer->decode_ids(stream_collapsed_ids_)),
            "en"};
    }
    stream_finalized_ = true;
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
    capabilities.languages = {"en"};
    capabilities.supports_timestamps = false;

    return std::make_unique<Granite5ASRLoadedModel>(
        std::move(metadata),
        std::move(capabilities),
        std::move(assets));
}

std::shared_ptr<runtime::IVoiceModelLoader> make_granite5asr_loader() {
    return std::make_shared<Granite5ASRLoader>();
}

}  // namespace engine::community_models::granite5asr
