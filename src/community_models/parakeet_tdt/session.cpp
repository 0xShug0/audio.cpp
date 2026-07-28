#include "engine/community_models/parakeet_tdt/session.h"

#include "engine/framework/debug/profiler.h"
#include "engine/framework/runtime/options.h"
#include "engine/framework/runtime/spec_backed_model.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <stdexcept>
#include <utility>
#include <vector>

namespace engine::community_models::parakeet_tdt {
namespace {

using Clock = std::chrono::steady_clock;

constexpr size_t kDefaultWeightContextBytes = 3072ull * 1024ull * 1024ull;
constexpr size_t kDefaultEncoderGraphArenaBytes = 1024ull * 1024ull * 1024ull;
constexpr size_t kDefaultDecoderGraphArenaBytes = 256ull * 1024ull * 1024ull;
constexpr const char * kFamily = "parakeet_tdt";
constexpr float kDefaultCenterDurationSec = 2.0f;
constexpr float kDefaultLeftContextSec = 10.0f;
constexpr float kDefaultRightContextSec = 2.0f;
constexpr float kDefaultAudioChunkThresholdSec = 30.0f;
constexpr float kMinimumPositiveDurationSec = 0.001f;

std::shared_ptr<const ParakeetTDTAssets> require_assets(std::shared_ptr<const ParakeetTDTAssets> assets) {
    if (assets == nullptr) {
        throw std::runtime_error("Parakeet TDT session requires assets");
    }
    return assets;
}

std::shared_ptr<const engine::model_spec::ModelContract> require_contract(
    std::shared_ptr<const engine::model_spec::ModelContract> contract) {
    if (contract == nullptr) {
        throw std::runtime_error("Parakeet TDT session requires a model contract");
    }
    return contract;
}

void validate_session_option_keys(
    const runtime::SessionOptions & options,
    const engine::model_spec::ModelContract & contract) {
    const std::string family_prefix = std::string(kFamily) + ".";
    for (const auto & [key, _] : options.options) {
        if (key.rfind(family_prefix, 0) == 0 &&
            contract.session_option_keys.find(key) == contract.session_option_keys.end()) {
            throw std::runtime_error("unknown Parakeet TDT session option: " + key);
        }
    }
}

bool use_flash_attention(const runtime::SessionOptions & options) {
    const auto value =
        runtime::find_option(options.options, {"parakeet_tdt.perf_mode"}).value_or("off");
    if (value == "off") {
        return false;
    }
    if (value == "flash_attention") {
        return true;
    }
    throw std::runtime_error(
        "parakeet_tdt.perf_mode must be 'off' or 'flash_attention'");
}

engine::assets::TensorStorageType option_weight_type(
    const runtime::SessionOptions & options,
    const char * key,
    engine::assets::TensorStorageType fallback) {
    const auto it = options.options.find(key);
    if (it == options.options.end()) { return fallback; }
    return engine::assets::parse_tensor_storage_type(it->second);
}

void validate_matmul_weight_storage(engine::assets::TensorStorageType storage_type, const char * option_name) {
    if (storage_type == engine::assets::TensorStorageType::Native ||
        storage_type == engine::assets::TensorStorageType::F32 ||
        storage_type == engine::assets::TensorStorageType::F16 ||
        storage_type == engine::assets::TensorStorageType::BF16 ||
        storage_type == engine::assets::TensorStorageType::Q8_0) { return; }
    throw std::runtime_error(std::string(option_name) + " supports only native, f32, f16, bf16, and q8_0");
}

void validate_conv_weight_storage(engine::assets::TensorStorageType storage_type, const char * option_name) {
    if (storage_type == engine::assets::TensorStorageType::Native ||
        storage_type == engine::assets::TensorStorageType::F32 ||
        storage_type == engine::assets::TensorStorageType::F16) { return; }
    throw std::runtime_error(std::string(option_name) + " supports only native, f32, and f16");
}

int64_t frontend_frames_for_samples(
    int64_t interleaved_samples,
    int channels,
    int source_sample_rate,
    const ParakeetFrontendConfig & config) {
    if (interleaved_samples <= 0 || channels <= 0 || source_sample_rate <= 0) { return 0; }
    const int64_t source_frames = interleaved_samples / channels;
    const double resampled =
        static_cast<double>(source_frames) * static_cast<double>(config.sample_rate) / static_cast<double>(source_sample_rate);
    const int64_t samples = static_cast<int64_t>(std::ceil(resampled));
    return samples / config.hop_length + 1;
}

float duration_option(
    const runtime::SessionOptions& options,
    const char* key,
    float fallback,
    float minimum) {
    const float value = runtime::parse_finite_float_option(options.options, {key}).value_or(fallback);
    if (value < minimum) {
        throw std::runtime_error(
            std::string(key) + " must be at least " + std::to_string(minimum));
    }
    return value;
}

void validate_duration_contract_options(const runtime::SessionOptions& options) {
    (void)duration_option(
        options,
        "parakeet_tdt.audio_chunk_duration_sec",
        kDefaultCenterDurationSec,
        kMinimumPositiveDurationSec);
    (void)duration_option(
        options,
        "parakeet_tdt.left_context_sec",
        kDefaultLeftContextSec,
        0.f);
    (void)duration_option(
        options,
        "parakeet_tdt.right_context_sec",
        kDefaultRightContextSec,
        0.f);
    (void)duration_option(
        options,
        "parakeet_tdt.audio_chunk_threshold_sec",
        kDefaultAudioChunkThresholdSec,
        kMinimumPositiveDurationSec);
}

std::string offline_mode_option(const runtime::SessionOptions& options) {
    const auto value =
        runtime::find_option(options.options, {"parakeet_tdt.offline_mode"})
            .value_or("full_context");
    if (value != "full_context" && value != "long_form" && value != "auto") {
        throw std::runtime_error(
            "parakeet_tdt.offline_mode must be 'full_context', 'long_form', or 'auto'");
    }
    return value;
}

std::string streaming_attention_mode_option(const runtime::SessionOptions& options) {
    const auto value =
        runtime::find_option(options.options, {"parakeet_tdt.streaming_attention_mode"})
            .value_or("full_context");
    if (value != "full_context") {
        throw std::runtime_error(
            "parakeet_tdt.streaming_attention_mode currently supports only 'full_context'");
    }
    return value;
}

void validate_enum_contract_options(const runtime::SessionOptions& options) {
    (void)offline_mode_option(options);
    (void)streaming_attention_mode_option(options);
}

int64_t seconds_to_samples(float seconds, int sample_rate) {
    return static_cast<int64_t>(std::llround(
        static_cast<double>(seconds) * static_cast<double>(sample_rate)));
}

}  // namespace

ParakeetTDTSessionBase::ParakeetTDTSessionBase(
    runtime::TaskSpec task,
    runtime::SessionOptions options,
    std::shared_ptr<const ParakeetTDTAssets> assets,
    std::shared_ptr<const engine::model_spec::ModelContract> contract)
    : RuntimeSessionBase(options),
      task_(task),
      assets_(require_assets(std::move(assets))),
      contract_(require_contract(std::move(contract))),
      weight_context_bytes_(runtime::parse_size_mb_option(options.options, {"parakeet_tdt.weight_context_mb"}, kDefaultWeightContextBytes)),
      encoder_graph_arena_bytes_(runtime::parse_size_mb_option(options.options, {"parakeet_tdt.encoder_graph_arena_mb"}, kDefaultEncoderGraphArenaBytes)),
      decoder_graph_arena_bytes_(runtime::parse_size_mb_option(options.options, {"parakeet_tdt.decoder_graph_arena_mb"}, kDefaultDecoderGraphArenaBytes)),
      matmul_weight_storage_type_(option_weight_type(
          options,
          "parakeet_tdt.matmul_weight_type",
          option_weight_type(options, "parakeet_tdt.weight_type", engine::assets::TensorStorageType::Native))),
      conv_weight_storage_type_(option_weight_type(options, "parakeet_tdt.conv_weight_type", engine::assets::TensorStorageType::Native)),
      encoder_flash_attention_(use_flash_attention(options)),
      frontend_(assets_) {
    if (task_.task != runtime::VoiceTaskKind::Asr) {
        throw std::runtime_error("Parakeet TDT only supports VoiceTaskKind::Asr");
    }
    if (task_.mode != runtime::RunMode::Offline &&
        task_.mode != runtime::RunMode::Streaming) {
        throw std::runtime_error("Parakeet TDT supports offline and buffered-streaming sessions");
    }
    validate_matmul_weight_storage(matmul_weight_storage_type_, "parakeet_tdt.weight_type");
    validate_conv_weight_storage(conv_weight_storage_type_, "parakeet_tdt.conv_weight_type");
    validate_session_option_keys(options, *contract_);
    validate_duration_contract_options(options);
    validate_enum_contract_options(options);
    weights_ = load_parakeet_weights(
        *assets_,
        execution_context().backend(),
        execution_context().backend_type(),
        matmul_weight_storage_type_,
        conv_weight_storage_type_,
        weight_context_bytes_);
    encoder_ = std::make_unique<ParakeetEncoderRuntime>(
        assets_,
        weights_,
        execution_context(),
        encoder_graph_arena_bytes_,
        encoder_flash_attention_);
    decoder_ = std::make_unique<ParakeetDecoderRuntime>(
        assets_,
        weights_,
        execution_context(),
        decoder_graph_arena_bytes_);
}

ParakeetTDTSessionBase::~ParakeetTDTSessionBase() = default;

std::string ParakeetTDTSessionBase::family_impl() const { return "parakeet_tdt"; }
runtime::VoiceTaskKind ParakeetTDTSessionBase::task_kind_impl() const { return task_.task; }
runtime::RunMode ParakeetTDTSessionBase::run_mode_impl() const { return task_.mode; }

ParakeetDecodeOptions ParakeetTDTSessionBase::decode_options_for_request(const runtime::TaskRequest & request) const {
    ParakeetDecodeOptions opts;
    if (const auto value = runtime::parse_i64_option(request.options, {"max_tokens"})) {
        if (*value < 0) { throw std::runtime_error("Parakeet TDT max_tokens must be non-negative"); }
        opts.max_tokens = *value;
    }
    if (const auto value = runtime::find_option(request.options, {"keep_language_tags"})) {
        opts.keep_language_tags = runtime::parse_bool_option(*value, "keep_language_tags");
    }
    return opts;
}

ParakeetTDTOfflineSession::ParakeetTDTOfflineSession(
    runtime::TaskSpec task,
    runtime::SessionOptions options,
    std::shared_ptr<const ParakeetTDTAssets> assets,
    std::shared_ptr<const engine::model_spec::ModelContract> contract)
    : ParakeetTDTSessionBase(
          task,
          std::move(options),
          std::move(assets),
          std::move(contract)) {
    offline_mode_ = offline_mode_option(this->options());
    const int sample_rate = assets_->config.frontend.sample_rate;
    center_samples_ = seconds_to_samples(
        duration_option(
            this->options(),
            "parakeet_tdt.audio_chunk_duration_sec",
            kDefaultCenterDurationSec,
            kMinimumPositiveDurationSec),
        sample_rate);
    left_context_samples_ = seconds_to_samples(
        duration_option(
            this->options(),
            "parakeet_tdt.left_context_sec",
            kDefaultLeftContextSec,
            0.f),
        sample_rate);
    right_context_samples_ = seconds_to_samples(
        duration_option(
            this->options(),
            "parakeet_tdt.right_context_sec",
            kDefaultRightContextSec,
            0.f),
        sample_rate);
    auto_full_context_max_samples_ = seconds_to_samples(
        duration_option(
            this->options(),
            "parakeet_tdt.audio_chunk_threshold_sec",
            kDefaultAudioChunkThresholdSec,
            kMinimumPositiveDurationSec),
        sample_rate);
}

std::string ParakeetTDTOfflineSession::family() const { return family_impl(); }
runtime::VoiceTaskKind ParakeetTDTOfflineSession::task_kind() const { return task_kind_impl(); }
runtime::RunMode ParakeetTDTOfflineSession::run_mode() const { return run_mode_impl(); }

void ParakeetTDTOfflineSession::prepare(const runtime::SessionPreparationRequest & request) {
    const auto prepare_start = Clock::now();
    if (!request.audio.has_value()) {
        throw std::runtime_error("Parakeet TDT prepare() requires an audio contract");
    }
    int64_t capacity_samples = request.audio->max_input_samples;
    int capacity_channels = request.audio->channels;
    int capacity_sample_rate = request.audio->sample_rate;
    if (offline_mode_ == "long_form" ||
        (offline_mode_ == "auto" &&
         request.audio->max_input_samples / std::max(request.audio->channels, 1) >
             auto_full_context_max_samples_)) {
        capacity_samples =
            left_context_samples_ + center_samples_ + right_context_samples_;
        capacity_channels = 1;
        capacity_sample_rate = assets_->config.frontend.sample_rate;
    }
    const int64_t frames = frontend_frames_for_samples(
        capacity_samples,
        capacity_channels,
        capacity_sample_rate,
        assets_->config.frontend);
    if (frames > 0) {
        encoder_->prepare_capacity(frames, assets_->config.frontend.feature_size);
    }
    decoder_->prepare();
    mark_prepared();
    debug::timing_log_scalar("parakeet_tdt.prepare_ms", engine::debug::elapsed_ms(prepare_start, Clock::now()));
}

runtime::TaskResult ParakeetTDTOfflineSession::run(const runtime::TaskRequest & request) {
    require_prepared("Parakeet TDT run()");
    if (!request.audio_input.has_value()) {
        throw std::runtime_error("Parakeet TDT run() requires audio_input");
    }
    if (request.audio_input->sample_rate <= 0 ||
        request.audio_input->channels <= 0 ||
        request.audio_input->samples.size() %
                static_cast<size_t>(request.audio_input->channels) !=
            0) {
        throw std::runtime_error("Parakeet TDT run() received an invalid audio layout");
    }
    const auto wall_start = Clock::now();
    const auto decode_options = decode_options_for_request(request);
    const int64_t source_frames =
        static_cast<int64_t>(request.audio_input->samples.size()) /
        std::max(request.audio_input->channels, 1);
    const int64_t target_samples = static_cast<int64_t>(std::ceil(
        static_cast<double>(source_frames) *
        static_cast<double>(assets_->config.frontend.sample_rate) /
        static_cast<double>(request.audio_input->sample_rate)));
    const bool use_long_form =
        offline_mode_ == "long_form" ||
        (offline_mode_ == "auto" && target_samples > auto_full_context_max_samples_);
    if (use_long_form) {
        auto result = run_long_form(*request.audio_input, decode_options);
        debug::timing_log_scalar(
            "session.wall_ms",
            engine::debug::elapsed_ms(wall_start, Clock::now()));
        return result;
    }

    const auto frontend = frontend_.extract(*request.audio_input, true);
    const auto encoded = encoder_->encode(frontend);
    auto decoded = decoder_->decode(encoded, decode_options);

    runtime::TaskResult result;
    result.text_output = runtime::Transcript{decoded.text, ""};
    result.word_timestamps = std::move(decoded.word_timestamps);
    debug::timing_log_scalar("session.wall_ms", engine::debug::elapsed_ms(wall_start, Clock::now()));
    return result;
}

runtime::TaskResult ParakeetTDTOfflineSession::run_long_form(
    const runtime::AudioBuffer& audio,
    const ParakeetDecodeOptions& options) {
    if (audio.sample_rate != assets_->config.frontend.sample_rate ||
        audio.channels != 1) {
        throw std::runtime_error(
            "Parakeet TDT long-form mode requires mono 16 kHz audio");
    }
    if (audio.samples.empty()) {
        throw std::runtime_error("Parakeet TDT long-form mode requires non-empty audio");
    }

    decoder_->reset_state();
    std::vector<int32_t> token_ids;
    std::vector<int32_t> token_frame_indices;
    std::vector<int32_t> token_durations;
    int64_t center_start = 0;
    int64_t decoded_frame_offset = 0;
    const int64_t total_samples = static_cast<int64_t>(audio.samples.size());
    const int64_t samples_per_frame =
        assets_->config.frontend.hop_length * assets_->config.encoder.subsampling_factor;

    while (center_start < total_samples) {
        const int64_t center_end =
            std::min(center_start + center_samples_, total_samples);
        const int64_t window_start =
            std::max<int64_t>(0, center_start - left_context_samples_);
        const int64_t window_end =
            std::min(total_samples, center_end + right_context_samples_);

        runtime::AudioBuffer window;
        window.sample_rate = audio.sample_rate;
        window.channels = 1;
        window.samples.assign(
            audio.samples.begin() + static_cast<std::ptrdiff_t>(window_start),
            audio.samples.begin() + static_cast<std::ptrdiff_t>(window_end));
        const auto features = frontend_.extract(window, true);
        const auto encoded = encoder_->encode(features);
        const int64_t local_start_frame =
            (center_start - window_start) / samples_per_frame;
        const int64_t local_end_frame = std::min<int64_t>(
            encoded.valid_frames,
            (center_end - window_start + samples_per_frame - 1) / samples_per_frame);
        const int64_t center_frames =
            std::max<int64_t>(0, local_end_frame - local_start_frame);
        if (center_frames > 0) {
            ParakeetEncodedAudio center_encoded;
            center_encoded.frames = center_frames;
            center_encoded.valid_frames = center_frames;
            center_encoded.hidden_size = encoded.hidden_size;
            const auto first = encoded.values.begin() +
                static_cast<std::ptrdiff_t>(local_start_frame * encoded.hidden_size);
            const auto last = first +
                static_cast<std::ptrdiff_t>(center_frames * encoded.hidden_size);
            center_encoded.values.assign(first, last);

            auto decode_options = options;
            if (decode_options.max_tokens > 0) {
                decode_options.max_tokens = std::max<int64_t>(
                    0,
                    decode_options.max_tokens - static_cast<int64_t>(token_ids.size()));
            }
            if (decode_options.max_tokens != 0 || options.max_tokens == 0) {
                auto decoded = decoder_->decode_incremental(
                    center_encoded,
                    decode_options,
                    decoded_frame_offset);
                token_ids.insert(
                    token_ids.end(),
                    decoded.token_ids.begin(),
                    decoded.token_ids.end());
                token_frame_indices.insert(
                    token_frame_indices.end(),
                    decoded.token_frame_indices.begin(),
                    decoded.token_frame_indices.end());
                token_durations.insert(
                    token_durations.end(),
                    decoded.durations.begin(),
                    decoded.durations.end());
            }
            decoded_frame_offset += center_frames;
        }
        center_start = center_end;
    }

    const int64_t audio_end_frame =
        (total_samples + samples_per_frame - 1) / samples_per_frame;
    auto decoded = decoder_->format_tokens(
        std::move(token_ids),
        std::move(token_frame_indices),
        std::move(token_durations),
        options,
        audio_end_frame);
    runtime::TaskResult result;
    result.text_output = runtime::Transcript{decoded.text, ""};
    result.word_timestamps = std::move(decoded.word_timestamps);
    return result;
}

ParakeetTDTStreamingSession::ParakeetTDTStreamingSession(
    runtime::TaskSpec task,
    runtime::SessionOptions options,
    std::shared_ptr<const ParakeetTDTAssets> assets,
    std::shared_ptr<const engine::model_spec::ModelContract> contract)
    : ParakeetTDTSessionBase(
          task,
          std::move(options),
          std::move(assets),
          std::move(contract)) {
    const int sample_rate = assets_->config.frontend.sample_rate;
    center_samples_ = seconds_to_samples(
        duration_option(
            this->options(),
            "parakeet_tdt.audio_chunk_duration_sec",
            kDefaultCenterDurationSec,
            kMinimumPositiveDurationSec),
        sample_rate);
    left_context_samples_ = seconds_to_samples(
        duration_option(
            this->options(),
            "parakeet_tdt.left_context_sec",
            kDefaultLeftContextSec,
            0.f),
        sample_rate);
    right_context_samples_ = seconds_to_samples(
        duration_option(
            this->options(),
            "parakeet_tdt.right_context_sec",
            kDefaultRightContextSec,
            0.f),
        sample_rate);
}

std::string ParakeetTDTStreamingSession::family() const { return family_impl(); }
runtime::VoiceTaskKind ParakeetTDTStreamingSession::task_kind() const { return task_kind_impl(); }
runtime::RunMode ParakeetTDTStreamingSession::run_mode() const { return run_mode_impl(); }

void ParakeetTDTStreamingSession::prepare(
    const runtime::SessionPreparationRequest& request) {
    const auto prepare_start = Clock::now();
    if (!request.audio.has_value()) {
        throw std::runtime_error("Parakeet TDT buffered streaming prepare() requires an audio contract");
    }
    if (request.audio->sample_rate != assets_->config.frontend.sample_rate ||
        request.audio->channels != 1) {
        throw std::runtime_error(
            "Parakeet TDT buffered streaming requires mono 16 kHz audio");
    }
    const int64_t capacity_samples =
        left_context_samples_ + center_samples_ + right_context_samples_;
    const int64_t capacity_frames = frontend_frames_for_samples(
        capacity_samples,
        1,
        assets_->config.frontend.sample_rate,
        assets_->config.frontend);
    encoder_->prepare_capacity(capacity_frames, assets_->config.frontend.feature_size);
    decoder_->prepare();
    mark_prepared();
    reset();
    debug::timing_log_scalar(
        "parakeet_tdt.prepare_ms",
        engine::debug::elapsed_ms(prepare_start, Clock::now()));
}

runtime::StreamingPolicy ParakeetTDTStreamingSession::streaming_policy() const {
    runtime::StreamingPolicy policy;
    policy.input = runtime::StreamingInputKind::AudioChunks;
    policy.output = runtime::StreamingOutputKind::FinalResult;
    policy.preferred_audio_chunk_samples = center_samples_;
    policy.preferred_audio_chunk_seconds =
        static_cast<double>(center_samples_) /
        static_cast<double>(assets_->config.frontend.sample_rate);
    return policy;
}

void ParakeetTDTStreamingSession::start_stream(const runtime::TaskRequest& request) {
    reset();
    streaming_decode_options_ = decode_options_for_request(request);
}

void ParakeetTDTStreamingSession::set_stream_event_sink(
    runtime::StreamEventCallback sink) {
    stream_event_sink_ = std::move(sink);
}

void ParakeetTDTStreamingSession::reset() {
    require_prepared("Parakeet TDT buffered streaming reset()");
    if (task_.mode != runtime::RunMode::Streaming) {
        throw std::runtime_error("Parakeet TDT reset() requires a streaming session");
    }
    streaming_audio_ = runtime::AudioBuffer{
        static_cast<int>(assets_->config.frontend.sample_rate),
        1,
        {},
    };
    streaming_decode_options_ = {};
    next_center_start_sample_ = 0;
    decoded_frame_offset_ = 0;
    buffer_start_sample_ = 0;
    received_samples_ = 0;
    token_ids_.clear();
    token_frame_indices_.clear();
    token_durations_.clear();
    decoder_->reset_state();
    stream_started_ = true;
    finalized_ = false;
}

ParakeetDecodedText ParakeetTDTStreamingSession::merged_decode() const {
    const int64_t samples_per_frame =
        assets_->config.frontend.hop_length * assets_->config.encoder.subsampling_factor;
    const int64_t audio_end_frame =
        (received_samples_ + samples_per_frame - 1) /
        samples_per_frame;
    return decoder_->format_tokens(
        token_ids_,
        token_frame_indices_,
        token_durations_,
        streaming_decode_options_,
        audio_end_frame);
}

bool ParakeetTDTStreamingSession::process_center_window(
    int64_t center_end_sample,
    bool flush_tail) {
    if (next_center_start_sample_ >= received_samples_) {
        return false;
    }
    const int64_t center_end = std::min(center_end_sample, received_samples_);
    const int64_t window_start =
        std::max<int64_t>(0, next_center_start_sample_ - left_context_samples_);
    const int64_t requested_window_end = center_end + right_context_samples_;
    const int64_t window_end = std::min(received_samples_, requested_window_end);
    if (!flush_tail && window_end < requested_window_end) {
        return false;
    }

    runtime::AudioBuffer window;
    window.sample_rate = streaming_audio_.sample_rate;
    window.channels = 1;
    window.samples.assign(
        streaming_audio_.samples.begin() +
            static_cast<std::ptrdiff_t>(window_start - buffer_start_sample_),
        streaming_audio_.samples.begin() +
            static_cast<std::ptrdiff_t>(window_end - buffer_start_sample_));

    const auto features = frontend_.extract(window, true);
    const auto encoded = encoder_->encode(features);
    const int64_t samples_per_frame =
        assets_->config.frontend.hop_length * assets_->config.encoder.subsampling_factor;
    const int64_t local_start_frame =
        (next_center_start_sample_ - window_start) / samples_per_frame;
    const int64_t local_end_frame = std::min<int64_t>(
        encoded.valid_frames,
        (center_end - window_start + samples_per_frame - 1) / samples_per_frame);
    const int64_t center_frames = std::max<int64_t>(0, local_end_frame - local_start_frame);
    if (center_frames > 0) {
        ParakeetEncodedAudio center_encoded;
        center_encoded.frames = center_frames;
        center_encoded.valid_frames = center_frames;
        center_encoded.hidden_size = encoded.hidden_size;
        const auto first = encoded.values.begin() +
            static_cast<std::ptrdiff_t>(local_start_frame * encoded.hidden_size);
        const auto last = first +
            static_cast<std::ptrdiff_t>(center_frames * encoded.hidden_size);
        center_encoded.values.assign(first, last);

        auto decode_options = streaming_decode_options_;
        if (decode_options.max_tokens > 0) {
            decode_options.max_tokens = std::max<int64_t>(
                0,
                decode_options.max_tokens - static_cast<int64_t>(token_ids_.size()));
        }
        if (decode_options.max_tokens != 0 || streaming_decode_options_.max_tokens == 0) {
            auto decoded = decoder_->decode_incremental(
                center_encoded,
                decode_options,
                decoded_frame_offset_);
            token_ids_.insert(
                token_ids_.end(),
                decoded.token_ids.begin(),
                decoded.token_ids.end());
            token_frame_indices_.insert(
                token_frame_indices_.end(),
                decoded.token_frame_indices.begin(),
                decoded.token_frame_indices.end());
            token_durations_.insert(
                token_durations_.end(),
                decoded.durations.begin(),
                decoded.durations.end());
        }
        decoded_frame_offset_ += center_frames;
    }
    next_center_start_sample_ = center_end;
    return true;
}

runtime::StreamEvent ParakeetTDTStreamingSession::process_ready_windows(bool flush_tail) {
    bool changed = false;
    while (next_center_start_sample_ < received_samples_) {
        const int64_t center_end = std::min(
            next_center_start_sample_ + center_samples_,
            received_samples_);
        if (!flush_tail &&
            received_samples_ < center_end + right_context_samples_) {
            break;
        }
        if (!process_center_window(center_end, flush_tail)) {
            break;
        }
        changed = true;
    }

    const int64_t keep_from =
        std::max<int64_t>(0, next_center_start_sample_ - left_context_samples_);
    if (keep_from > buffer_start_sample_) {
        const int64_t discard = keep_from - buffer_start_sample_;
        streaming_audio_.samples.erase(
            streaming_audio_.samples.begin(),
            streaming_audio_.samples.begin() + static_cast<std::ptrdiff_t>(discard));
        buffer_start_sample_ = keep_from;
    }

    runtime::StreamEvent event;
    if (changed && !token_ids_.empty()) {
        auto decoded = merged_decode();
        event.partial_text = runtime::Transcript{decoded.text, ""};
        event.word_timestamps = std::move(decoded.word_timestamps);
        // The last word has no following word boundary yet, so it remains
        // provisional and is withheld from the finalized timestamp list.
        if (!event.word_timestamps.empty()) {
            event.word_timestamps.pop_back();
        }
        if (stream_event_sink_) {
            stream_event_sink_(event);
            return {};
        }
    }
    return event;
}

runtime::StreamEvent ParakeetTDTStreamingSession::process_audio_chunk(
    const runtime::AudioChunk& chunk) {
    require_prepared("Parakeet TDT buffered streaming process_audio_chunk()");
    if (!stream_started_ || finalized_) {
        throw std::runtime_error(
            "Parakeet TDT buffered streaming chunk received outside an active stream");
    }
    if (chunk.sample_rate != assets_->config.frontend.sample_rate ||
        chunk.channels != 1) {
        throw std::runtime_error(
            "Parakeet TDT buffered streaming requires mono 16 kHz audio");
    }
    if (chunk.start_sample != received_samples_) {
        throw std::runtime_error(
            "Parakeet TDT buffered streaming chunks must be contiguous");
    }
    streaming_audio_.samples.insert(
        streaming_audio_.samples.end(),
        chunk.samples.begin(),
        chunk.samples.end());
    received_samples_ += static_cast<int64_t>(chunk.samples.size());
    return process_ready_windows(false);
}

runtime::TaskResult ParakeetTDTStreamingSession::finalize() {
    require_prepared("Parakeet TDT buffered streaming finalize()");
    if (!stream_started_ || finalized_) {
        throw std::runtime_error(
            "Parakeet TDT buffered streaming finalize() requires an active stream");
    }
    if (received_samples_ == 0) {
        throw std::runtime_error(
            "Parakeet TDT buffered streaming finalize() requires streamed audio");
    }
    auto event = process_ready_windows(true);
    (void)event;
    auto decoded = merged_decode();
    runtime::TaskResult result;
    result.text_output = runtime::Transcript{decoded.text, ""};
    result.word_timestamps = std::move(decoded.word_timestamps);
    finalized_ = true;
    stream_started_ = false;
    return result;
}

std::shared_ptr<runtime::IVoiceModelLoader> make_parakeet_tdt_loader() {
    runtime::SpecBackedVoiceModelConfig<ParakeetTDTAssets> config;
    config.family = kFamily;
    config.load_assets = load_parakeet_assets;
    config.create_session = [](
                                const runtime::TaskSpec & task,
                                const runtime::SessionOptions & options,
                                std::shared_ptr<const ParakeetTDTAssets> assets,
                                std::shared_ptr<const engine::model_spec::ModelContract> contract) {
        if (task.mode == runtime::RunMode::Streaming) {
            return std::unique_ptr<runtime::IVoiceTaskSession>(
                std::make_unique<ParakeetTDTStreamingSession>(
                    task,
                    options,
                    std::move(assets),
                    std::move(contract)));
        }
        return std::unique_ptr<runtime::IVoiceTaskSession>(
            std::make_unique<ParakeetTDTOfflineSession>(
                task,
                options,
                std::move(assets),
                std::move(contract)));
    };
    return runtime::make_spec_backed_voice_loader(std::move(config));
}

}  // namespace engine::community_models::parakeet_tdt
