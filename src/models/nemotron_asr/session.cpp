#include "engine/models/nemotron_asr/session.h"

#include "engine/framework/debug/profiler.h"
#include "engine/framework/runtime/options.h"
#include "engine/framework/runtime/spec_backed_model.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <stdexcept>
#include <utility>
#include <vector>

namespace engine::models::nemotron_asr {
namespace {

using Clock = std::chrono::steady_clock;

constexpr size_t kDefaultWeightContextBytes = 3072ull * 1024ull * 1024ull;
constexpr size_t kDefaultEncoderGraphArenaBytes = 1024ull * 1024ull * 1024ull;
constexpr size_t kDefaultDecoderGraphArenaBytes = 256ull * 1024ull * 1024ull;
constexpr double kStreamingFlushSeconds = 0.5;

std::shared_ptr<const NemotronASRAssets> require_assets(std::shared_ptr<const NemotronASRAssets> assets) {
    if (assets == nullptr) {
        throw std::runtime_error("Nemotron ASR session requires assets");
    }
    return assets;
}

engine::assets::TensorStorageType option_weight_type(
    const runtime::SessionOptions & options,
    const char * key,
    engine::assets::TensorStorageType fallback) {
    const auto it = options.options.find(key);
    if (it == options.options.end()) {
        return fallback;
    }
    return engine::assets::parse_tensor_storage_type(it->second);
}

void validate_matmul_weight_storage(engine::assets::TensorStorageType storage_type, const char * option_name) {
    if (storage_type == engine::assets::TensorStorageType::Native ||
        storage_type == engine::assets::TensorStorageType::F32 ||
        storage_type == engine::assets::TensorStorageType::F16 ||
        storage_type == engine::assets::TensorStorageType::BF16 ||
        storage_type == engine::assets::TensorStorageType::Q8_0) {
        return;
    }
    throw std::runtime_error(std::string(option_name) + " supports only native, f32, f16, bf16, and q8_0");
}

void validate_conv_weight_storage(engine::assets::TensorStorageType storage_type, const char * option_name) {
    if (storage_type == engine::assets::TensorStorageType::Native ||
        storage_type == engine::assets::TensorStorageType::F32 ||
        storage_type == engine::assets::TensorStorageType::F16) {
        return;
    }
    throw std::runtime_error(std::string(option_name) + " supports only native, f32, and f16");
}

bool mem_saver_from_options(const runtime::SessionOptions & options) {
    if (const auto value = runtime::find_option(options.options, {"nemotron_asr.mem_saver"})) {
        return runtime::parse_bool_option(*value, "nemotron_asr.mem_saver");
    }
    return false;
}

int64_t frontend_frames_for_samples(
    int64_t interleaved_samples,
    int channels,
    int source_sample_rate,
    const NemotronFrontendConfig & config) {
    if (interleaved_samples <= 0 || channels <= 0 || source_sample_rate <= 0) {
        return 0;
    }
    const int64_t source_frames = interleaved_samples / channels;
    const double resampled =
        static_cast<double>(source_frames) * static_cast<double>(config.sample_rate) / static_cast<double>(source_sample_rate);
    const int64_t samples = static_cast<int64_t>(std::ceil(resampled));
    return samples / config.hop_length + 1;
}

NemotronFrontendFeatures slice_features(const NemotronFrontendFeatures & in, int64_t start_frame, int64_t frames) {
    if (start_frame < 0 || frames <= 0 || start_frame + frames > in.frames) {
        throw std::runtime_error("Nemotron ASR streaming feature slice is out of range");
    }
    NemotronFrontendFeatures out;
    out.frames = frames;
    out.valid_frames = std::min<int64_t>(frames, std::max<int64_t>(0, in.valid_frames - start_frame));
    out.feature_dim = in.feature_dim;
    out.values.resize(static_cast<size_t>(frames * in.feature_dim));
    for (int64_t t = 0; t < frames; ++t) {
        std::copy_n(
            in.values.begin() + static_cast<std::ptrdiff_t>((start_frame + t) * in.feature_dim),
            static_cast<std::ptrdiff_t>(in.feature_dim),
            out.values.begin() + static_cast<std::ptrdiff_t>(t * in.feature_dim));
    }
    return out;
}

}  // namespace

NemotronASRSessionBase::NemotronASRSessionBase(
    runtime::TaskSpec task,
    runtime::SessionOptions options,
    std::shared_ptr<const NemotronASRAssets> assets,
    std::shared_ptr<const model_spec::ModelContract> contract)
    : RuntimeSessionBase(options),
      task_(task),
      contract_(std::move(contract)),
      assets_(require_assets(std::move(assets))),
      weight_context_bytes_(runtime::parse_size_mb_option(options.options, {"nemotron_asr.weight_context_mb"}, kDefaultWeightContextBytes)),
      encoder_graph_arena_bytes_(runtime::parse_size_mb_option(options.options, {"nemotron_asr.encoder_graph_arena_mb"}, kDefaultEncoderGraphArenaBytes)),
      decoder_graph_arena_bytes_(runtime::parse_size_mb_option(options.options, {"nemotron_asr.decoder_graph_arena_mb"}, kDefaultDecoderGraphArenaBytes)),
      mem_saver_(mem_saver_from_options(options)),
      matmul_weight_storage_type_(option_weight_type(
          options,
          "nemotron_asr.matmul_weight_type",
          option_weight_type(options, "nemotron_asr.weight_type", engine::assets::TensorStorageType::Native))),
      conv_weight_storage_type_(option_weight_type(options, "nemotron_asr.conv_weight_type", engine::assets::TensorStorageType::Native)),
      frontend_(assets_) {
    if (task_.task != runtime::VoiceTaskKind::Asr) {
        throw std::runtime_error("Nemotron ASR only supports VoiceTaskKind::Asr");
    }
    if (task_.mode != runtime::RunMode::Offline && task_.mode != runtime::RunMode::Streaming) {
        throw std::runtime_error("Nemotron ASR only supports offline and streaming sessions");
    }
    validate_matmul_weight_storage(matmul_weight_storage_type_, "nemotron_asr.weight_type");
    validate_conv_weight_storage(conv_weight_storage_type_, "nemotron_asr.conv_weight_type");
    if (contract_ != nullptr) {
        runtime::validate_spec_backed_session_options(options, *contract_, "nemotron_asr", "Nemotron ASR");
    } else {
        // Legacy embedded spec without a v1 contract: the pre-migration whitelist.
        for (const auto & [key, value] : options.options) {
            (void)value;
            if (key.rfind("nemotron_asr.", 0) == 0 &&
                key != "nemotron_asr.weight_context_mb" &&
                key != "nemotron_asr.encoder_graph_arena_mb" &&
                key != "nemotron_asr.decoder_graph_arena_mb" &&
                key != "nemotron_asr.weight_type" &&
                key != "nemotron_asr.matmul_weight_type" &&
                key != "nemotron_asr.conv_weight_type" &&
                key != "nemotron_asr.mem_saver") {
                throw std::runtime_error("unknown Nemotron ASR session option: " + key);
            }
        }
    }
    weights_ = load_nemotron_asr_weights(
        *assets_,
        execution_context().backend(),
        execution_context().backend_type(),
        matmul_weight_storage_type_,
        conv_weight_storage_type_,
        weight_context_bytes_);
    encoder_ = std::make_unique<NemotronEncoderRuntime>(
        assets_,
        weights_,
        execution_context(),
        encoder_graph_arena_bytes_);
    decoder_ = std::make_unique<NemotronDecoderRuntime>(
        assets_,
        weights_,
        execution_context(),
        decoder_graph_arena_bytes_);
}

NemotronASRSessionBase::~NemotronASRSessionBase() = default;

std::string NemotronASRSessionBase::family_impl() const {
    return "nemotron_asr";
}

runtime::VoiceTaskKind NemotronASRSessionBase::task_kind_impl() const {
    return task_.task;
}

runtime::RunMode NemotronASRSessionBase::run_mode_impl() const {
    return task_.mode;
}

NemotronASROfflineSession::NemotronASROfflineSession(
    runtime::TaskSpec task,
    runtime::SessionOptions options,
    std::shared_ptr<const NemotronASRAssets> assets,
    std::shared_ptr<const model_spec::ModelContract> contract)
    : NemotronASRSessionBase(task, std::move(options), std::move(assets), std::move(contract)) {}

std::string NemotronASROfflineSession::family() const {
    return family_impl();
}

runtime::VoiceTaskKind NemotronASROfflineSession::task_kind() const {
    return task_kind_impl();
}

runtime::RunMode NemotronASROfflineSession::run_mode() const {
    return run_mode_impl();
}

void NemotronASROfflineSession::prepare(const runtime::SessionPreparationRequest & request) {
    const auto prepare_start = Clock::now();
    if (!request.audio.has_value()) {
        throw std::runtime_error("Nemotron ASR prepare() requires an audio contract");
    }
    validate_request_options(request.options);
    const int64_t lookahead = lookahead_for_options(request.options);
    const int64_t frames = frontend_frames_for_samples(
        request.audio->max_input_samples,
        request.audio->channels,
        request.audio->sample_rate,
        assets_->config.frontend);
    if (frames > 0 && !mem_saver_) {
        encoder_->prepare_capacity(frames, assets_->config.frontend.feature_size, lookahead);
    }
    decoder_->prepare();
    mark_prepared();
    debug::timing_log_scalar("nemotron_asr.prepare_ms", engine::debug::elapsed_ms(prepare_start, Clock::now()));
    debug::trace_log_scalar("nemotron_asr.prepare.max_input_samples", request.audio->max_input_samples);
    debug::trace_log_scalar("nemotron_asr.prepare.lookahead_tokens", lookahead);
    debug::trace_log_scalar("nemotron_asr.prepare.streaming", false);
}

void NemotronASRSessionBase::validate_request_options(
    const std::unordered_map<std::string, std::string> & options) const {
    // Without a v1 contract (legacy embedded spec), request options stay unvalidated as before.
    if (contract_ != nullptr) {
        runtime::validate_spec_backed_request_options(options, *contract_, "Nemotron ASR");
    }
}

int64_t NemotronASRSessionBase::prompt_id_for_request(const runtime::TaskRequest & request) const {
    std::string language;
    if (request.text_input.has_value() && !request.text_input->language.empty()) {
        language = request.text_input->language;
    }
    if (const auto option = runtime::find_option(request.options, {"language"})) {
        language = *option;
    }
    if (language.empty()) {
        return assets_->config.default_prompt_id;
    }
    const auto it = assets_->config.prompt_dictionary.find(language);
    if (it == assets_->config.prompt_dictionary.end()) {
        throw std::runtime_error("Nemotron ASR unsupported language prompt: " + language);
    }
    return it->second;
}

int64_t NemotronASRSessionBase::lookahead_for_options(const std::unordered_map<std::string, std::string> & options) const {
    int64_t lookahead = assets_->config.encoder.default_lookahead_tokens;
    if (const auto value = runtime::parse_i64_option(options, {"lookahead_tokens"})) {
        lookahead = *value;
    }
    if (std::find(
            assets_->config.encoder.supported_lookahead_tokens.begin(),
            assets_->config.encoder.supported_lookahead_tokens.end(),
            lookahead) == assets_->config.encoder.supported_lookahead_tokens.end()) {
        throw std::runtime_error("Nemotron ASR unsupported lookahead_tokens value");
    }
    return lookahead;
}

NemotronDecodeOptions NemotronASRSessionBase::decode_options_for_request(const runtime::TaskRequest & request) const {
    NemotronDecodeOptions options;
    if (const auto value = runtime::parse_i64_option(request.options, {"max_tokens"})) {
        if (*value < 0) {
            throw std::runtime_error("Nemotron ASR max_tokens must be non-negative");
        }
        options.max_tokens = *value;
    }
    if (const auto value = runtime::find_option(request.options, {"keep_language_tags"})) {
        options.keep_language_tags = runtime::parse_bool_option(*value, "keep_language_tags");
    }
    return options;
}

runtime::TaskResult NemotronASROfflineSession::run(const runtime::TaskRequest & request) {
    require_prepared("Nemotron ASR run()");
    if (task_.mode != runtime::RunMode::Offline) {
        throw std::runtime_error("Nemotron ASR offline run called on non-offline session");
    }
    if (!request.audio_input.has_value()) {
        throw std::runtime_error("Nemotron ASR run() requires audio_input");
    }
    validate_request_options(request.options);
    const auto wall_start = Clock::now();
    const auto config_start = Clock::now();
    const int64_t prompt_id = prompt_id_for_request(request);
    const int64_t lookahead = lookahead_for_options(request.options);
    const auto decode_options = decode_options_for_request(request);
    const auto streaming_option = runtime::find_option(request.options, {"streaming"});
    const bool streaming = streaming_option.has_value() && runtime::parse_bool_option(*streaming_option, "streaming");
    if (streaming) {
        throw std::runtime_error("Nemotron ASR streaming request requires a streaming session");
    }
    debug::timing_log_scalar("nemotron_asr.request_config_ms", engine::debug::elapsed_ms(config_start, Clock::now()));
    debug::trace_log_scalar("nemotron_asr.prompt_id", prompt_id);
    debug::trace_log_scalar("nemotron_asr.lookahead_tokens", lookahead);
    debug::trace_log_scalar("nemotron_asr.streaming", streaming);

    NemotronDecodedText decoded;
    const auto frontend = frontend_.extract(*request.audio_input, true);
    const auto encoded = encoder_->encode(frontend, prompt_id, lookahead);
    decoded = decoder_->decode(encoded, decode_options);
    if (mem_saver_) {
        const auto release_start = Clock::now();
        encoder_->release_offline_graph();
        debug::timing_log_scalar(
            "nemotron_asr.encoder_release.offline_graph_ms",
            engine::debug::elapsed_ms(release_start, Clock::now()));
    }

    std::string language;
    if (request.text_input.has_value()) {
        language = request.text_input->language;
    }
    runtime::TaskResult result;
    result.text_output = runtime::Transcript{decoded.text, language};
    result.word_timestamps = std::move(decoded.token_timestamps);
    debug::timing_log_scalar("session.wall_ms", engine::debug::elapsed_ms(wall_start, Clock::now()));
    return result;
}

NemotronASRStreamingSession::NemotronASRStreamingSession(
    runtime::TaskSpec task,
    runtime::SessionOptions options,
    std::shared_ptr<const NemotronASRAssets> assets,
    std::shared_ptr<const model_spec::ModelContract> contract)
    : NemotronASRSessionBase(task, std::move(options), std::move(assets), std::move(contract)) {}

std::string NemotronASRStreamingSession::family() const {
    return family_impl();
}

runtime::VoiceTaskKind NemotronASRStreamingSession::task_kind() const {
    return task_kind_impl();
}

runtime::RunMode NemotronASRStreamingSession::run_mode() const {
    return run_mode_impl();
}

void NemotronASRStreamingSession::prepare(const runtime::SessionPreparationRequest & request) {
    const auto prepare_start = Clock::now();
    if (!request.audio.has_value()) {
        throw std::runtime_error("Nemotron ASR streaming prepare() requires an audio contract");
    }
    validate_request_options(request.options);
    streaming_options_ = request.options;
    streaming_language_ = request.text.has_value() ? request.text->language : "";
    const int64_t lookahead = lookahead_for_options(streaming_options_);
    encoder_->prepare_streaming_capacity(assets_->config.frontend.feature_size, lookahead);
    decoder_->prepare();
    mark_prepared();
    debug::timing_log_scalar("nemotron_asr.prepare_ms", engine::debug::elapsed_ms(prepare_start, Clock::now()));
    debug::trace_log_scalar("nemotron_asr.prepare.max_input_samples", request.audio->max_input_samples);
    debug::trace_log_scalar("nemotron_asr.prepare.lookahead_tokens", lookahead);
    debug::trace_log_scalar("nemotron_asr.prepare.streaming", true);
}

runtime::StreamingPolicy NemotronASRStreamingSession::streaming_policy() const {
    runtime::StreamingPolicy policy;
    policy.input = runtime::StreamingInputKind::AudioChunks;
    policy.output = runtime::StreamingOutputKind::FinalResult;
    const auto & fc = assets_->config.frontend;
    const int64_t mel_frames =
        assets_->config.encoder.subsampling_factor *
        std::max<int64_t>(assets_->config.encoder.default_lookahead_tokens + 1, 4);
    policy.preferred_audio_chunk_samples = mel_frames * fc.hop_length;
    policy.preferred_audio_chunk_seconds =
        static_cast<double>(policy.preferred_audio_chunk_samples) /
        static_cast<double>(fc.sample_rate);
    return policy;
}

void NemotronASRStreamingSession::start_stream(const runtime::TaskRequest & request) {
    require_prepared("Nemotron ASR start_stream()");
    validate_request_options(request.options);
    reset();
    streaming_options_ = request.options;
    streaming_language_ = request.text_input.has_value() ? request.text_input->language : "";
    if (const auto option = runtime::find_option(request.options, {"language"})) {
        streaming_language_ = *option;
    }
    runtime::TaskRequest config_request;
    config_request.text_input = runtime::Transcript{"", streaming_language_};
    config_request.options = streaming_options_;
    prompt_id_ = prompt_id_for_request(config_request);
    lookahead_ = lookahead_for_options(streaming_options_);
    encoder_stream_state_ = encoder_->make_stream_state();
    decoder_stream_state_ = decoder_->make_stream_state(
        decode_options_for_request(config_request));
    stream_wall_start_ = Clock::now();
    stream_started_ = true;
}

void NemotronASRStreamingSession::set_stream_event_sink(runtime::StreamEventCallback sink) {
    stream_event_sink_ = std::move(sink);
}

void NemotronASRStreamingSession::reset() {
    require_prepared("Nemotron ASR reset()");
    if (task_.mode != runtime::RunMode::Streaming) {
        throw std::runtime_error("Nemotron ASR reset called on non-streaming session");
    }
    streaming_waveform_.clear();
    streaming_waveform_base_ = 0;
    received_samples_ = 0;
    next_chunk_start_ = 0;
    prompt_id_ = 0;
    lookahead_ = 0;
    chunks_processed_ = 0;
    first_chunk_processed_ = false;
    stream_started_ = false;
    finalized_ = false;
    stream_wall_start_ = {};
    partials_.reset();
}

runtime::StreamEvent NemotronASRStreamingSession::process_audio_chunk(const runtime::AudioChunk & chunk) {
    require_prepared("Nemotron ASR process_audio_chunk()");
    if (task_.mode != runtime::RunMode::Streaming) {
        throw std::runtime_error("Nemotron ASR process_audio_chunk called on non-streaming session");
    }
    if (!stream_started_ || finalized_) {
        throw std::runtime_error(
            "Nemotron ASR process_audio_chunk requires an active stream");
    }
    if (chunk.sample_rate != assets_->config.frontend.sample_rate ||
        chunk.channels != 1) {
        throw std::runtime_error(
            "Nemotron ASR streaming requires mono audio at the model sample rate");
    }
    if (chunk.start_sample != received_samples_) {
        throw std::runtime_error("Nemotron ASR streaming chunks must be contiguous");
    }
    streaming_waveform_.insert(
        streaming_waveform_.end(), chunk.samples.begin(), chunk.samples.end());
    received_samples_ += static_cast<int64_t>(chunk.samples.size());
    return process_available_chunks(false);
}

void NemotronASRStreamingSession::process_feature_chunk(
    const NemotronFrontendFeatures & features) {
    auto encoded = encoder_->encode_stream_chunk(
        features, prompt_id_, lookahead_, encoder_stream_state_);
    decoder_->decode_stream_chunk(encoded, decoder_stream_state_);
    ++chunks_processed_;
}

runtime::StreamEvent NemotronASRStreamingSession::publish_stream_update() {
    runtime::StreamEvent event;
    auto delta = partials_.publish(decoder_stream_state_.decoded.text);
    if (delta.empty()) {
        return event;
    }
    event.partial_text = runtime::Transcript{std::move(delta), streaming_language_};
    if (stream_event_sink_) {
        stream_event_sink_(event);
        return {};
    }
    return event;
}

runtime::StreamEvent NemotronASRStreamingSession::process_available_chunks(
    bool flush_tail) {
    const auto & fc = assets_->config.frontend;
    const int64_t first_mel_frames = std::max<int64_t>(
        assets_->config.encoder.subsampling_factor,
        1 + assets_->config.encoder.subsampling_factor * lookahead_);
    const int64_t mel_frames_per_chunk =
        assets_->config.encoder.subsampling_factor *
        std::max<int64_t>(lookahead_ + 1, 4);
    const int64_t first_samples =
        (first_mel_frames - 1) * fc.hop_length + fc.win_length / 2;
    const int64_t samples_per_chunk =
        mel_frames_per_chunk * fc.hop_length + fc.win_length;

    runtime::StreamEvent combined;
    auto publish = [&]() {
        auto event = publish_stream_update();
        if (!stream_event_sink_ && event.partial_text.has_value()) {
            if (!combined.partial_text.has_value()) {
                combined.partial_text = runtime::Transcript{"", streaming_language_};
            }
            combined.partial_text->text += event.partial_text->text;
        }
    };
    auto pad_features = [](NemotronFrontendFeatures features, int64_t frames) {
        if (features.frames > frames) {
            return slice_features(features, 0, frames);
        }
        features.values.resize(
            static_cast<size_t>(frames * features.feature_dim), 0.0f);
        features.frames = frames;
        features.valid_frames = frames;
        return features;
    };
    auto waveform_slice = [&](int64_t begin, int64_t end) {
        if (end < begin || end > received_samples_ ||
            std::max<int64_t>(begin, 0) < streaming_waveform_base_) {
            throw std::runtime_error("Nemotron ASR streaming waveform slice is out of range");
        }
        std::vector<float> result(static_cast<size_t>(end - begin), 0.0f);
        const int64_t source_begin = std::max<int64_t>(begin, 0);
        const auto first = streaming_waveform_.begin() +
            static_cast<std::ptrdiff_t>(source_begin - streaming_waveform_base_);
        const auto last = streaming_waveform_.begin() +
            static_cast<std::ptrdiff_t>(end - streaming_waveform_base_);
        std::copy(first, last, result.begin() + static_cast<std::ptrdiff_t>(source_begin - begin));
        return result;
    };

    if (!first_chunk_processed_ &&
        (received_samples_ >= first_samples || flush_tail)) {
        auto first_waveform = waveform_slice(
            0, std::min<int64_t>(received_samples_, first_samples));
        auto features = pad_features(
            frontend_.extract_waveform(first_waveform, true), first_mel_frames);
        process_feature_chunk(features);
        first_chunk_processed_ = true;
        next_chunk_start_ = first_mel_frames * fc.hop_length - fc.n_fft / 2;
        publish();
    }

    while (first_chunk_processed_ &&
           received_samples_ >= next_chunk_start_ + samples_per_chunk) {
        auto waveform = waveform_slice(
            next_chunk_start_, next_chunk_start_ + samples_per_chunk);
        auto features = frontend_.extract_waveform(waveform, false);
        if (features.frames != mel_frames_per_chunk) {
            throw std::runtime_error(
                "Nemotron ASR streaming frontend produced unexpected chunk frame count");
        }
        process_feature_chunk(features);
        next_chunk_start_ += mel_frames_per_chunk * fc.hop_length;
        publish();
    }

    if (flush_tail && first_chunk_processed_) {
        const int64_t tail_samples = received_samples_ - next_chunk_start_;
        if (tail_samples > fc.win_length) {
            auto waveform = waveform_slice(next_chunk_start_, received_samples_);
            auto features = pad_features(
                frontend_.extract_waveform(waveform, false), mel_frames_per_chunk);
            process_feature_chunk(features);
            publish();
        }
        const int64_t flush_chunks = std::max<int64_t>(
            1,
            static_cast<int64_t>(std::ceil(
                kStreamingFlushSeconds * static_cast<double>(fc.sample_rate) /
                static_cast<double>(mel_frames_per_chunk * fc.hop_length))));
        const std::vector<float> silence(static_cast<size_t>(samples_per_chunk), 0.0f);
        const auto silence_features = frontend_.extract_waveform(silence, false);
        for (int64_t chunk = 0; chunk < flush_chunks; ++chunk) {
            process_feature_chunk(silence_features);
            publish();
        }
    }

    if (!flush_tail && first_chunk_processed_) {
        const int64_t keep_from = std::max<int64_t>(0, next_chunk_start_);
        if (keep_from > streaming_waveform_base_) {
            const int64_t discard = keep_from - streaming_waveform_base_;
            streaming_waveform_.erase(
                streaming_waveform_.begin(),
                streaming_waveform_.begin() + static_cast<std::ptrdiff_t>(discard));
            streaming_waveform_base_ = keep_from;
        }
    }
    return combined;
}

runtime::TaskResult NemotronASRStreamingSession::finalize() {
    require_prepared("Nemotron ASR finalize()");
    if (task_.mode != runtime::RunMode::Streaming) {
        throw std::runtime_error("Nemotron ASR finalize called on non-streaming session");
    }
    if (!stream_started_ || finalized_) {
        throw std::runtime_error("Nemotron ASR finalize requires an active stream");
    }
    if (received_samples_ == 0) {
        throw std::runtime_error("Nemotron ASR finalize requires streamed audio");
    }
    (void) process_available_chunks(true);
    auto decoded = decoder_->stream_result(decoder_stream_state_);
    runtime::TaskResult result;
    result.text_output = runtime::Transcript{decoded.text, streaming_language_};
    result.word_timestamps = std::move(decoded.token_timestamps);
    finalized_ = true;
    stream_started_ = false;
    streaming_waveform_.clear();
    debug::trace_log_scalar("nemotron_asr.streaming.chunks", chunks_processed_);
    if (stream_wall_start_ != std::chrono::steady_clock::time_point{}) {
        debug::timing_log_scalar(
            "session.wall_ms", engine::debug::elapsed_ms(stream_wall_start_, Clock::now()));
    }
    return result;
}

runtime::TaskResult NemotronASRStreamingSession::finish_stream() {
    return finalize();
}

}  // namespace engine::models::nemotron_asr
