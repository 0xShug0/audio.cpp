#include "engine/models/nemotron_asr/session.h"

#include "engine/framework/debug/profiler.h"
#include "engine/framework/runtime/options.h"

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

std::shared_ptr<const NemotronAssets> require_assets(std::shared_ptr<const NemotronAssets> assets) {
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
    std::shared_ptr<const NemotronAssets> assets)
    : RuntimeSessionBase(options),
      task_(task),
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
    std::shared_ptr<const NemotronAssets> assets)
    : NemotronASRSessionBase(task, std::move(options), std::move(assets)) {}

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

NemotronDecodedText NemotronASRSessionBase::run_streaming_audio(
    const runtime::AudioBuffer & audio,
    int64_t prompt_id,
    int64_t lookahead,
    const NemotronDecodeOptions & decode_options,
    const NemotronTextDeltaCallback & on_text_delta) {
    const auto & fc = assets_->config.frontend;
    const int64_t first_mel_frames = 1 + assets_->config.encoder.subsampling_factor * lookahead;
    const int64_t mel_frames_per_chunk = assets_->config.encoder.subsampling_factor * (lookahead + 1);
    const int64_t first_samples = (first_mel_frames - 1) * fc.hop_length + fc.win_length / 2;
    const int64_t samples_per_chunk = mel_frames_per_chunk * fc.hop_length + fc.win_length;
    auto waveform = frontend_.prepare_waveform(audio);
    if (static_cast<int64_t>(waveform.size()) < first_samples) {
        throw std::runtime_error("Nemotron ASR streaming request is shorter than the first required chunk");
    }
    NemotronEncoderStreamState stream_state = encoder_->make_stream_state();
    bool first_chunk = true;
    int64_t chunk_count = 0;
    int64_t mel_frame_idx = first_mel_frames;
    int64_t start_idx = mel_frame_idx * fc.hop_length - fc.n_fft / 2;
    auto next_chunk = [&](NemotronEncodedAudio & out) -> bool {
        if (first_chunk) {
            first_chunk = false;
            ++chunk_count;
            std::vector<float> chunk_waveform(
                waveform.begin(),
                waveform.begin() + static_cast<std::ptrdiff_t>(first_samples));
            auto features = frontend_.extract_waveform(chunk_waveform, true);
            if (features.frames > first_mel_frames) {
                features = slice_features(features, 0, first_mel_frames);
            }
            out = encoder_->encode_stream_chunk(features, prompt_id, lookahead, stream_state);
            return true;
        }
        if (start_idx + samples_per_chunk >= static_cast<int64_t>(waveform.size())) {
            return false;
        }
        std::vector<float> chunk_waveform(
            waveform.begin() + static_cast<std::ptrdiff_t>(start_idx),
            waveform.begin() + static_cast<std::ptrdiff_t>(start_idx + samples_per_chunk));
        auto features = frontend_.extract_waveform(chunk_waveform, false);
        if (features.frames != mel_frames_per_chunk) {
            throw std::runtime_error("Nemotron ASR streaming frontend produced unexpected chunk frame count");
        }
        out = encoder_->encode_stream_chunk(features, prompt_id, lookahead, stream_state);
        mel_frame_idx += mel_frames_per_chunk;
        start_idx = mel_frame_idx * fc.hop_length - fc.n_fft / 2;
        ++chunk_count;
        return true;
    };
    auto decoded = decoder_->decode_streaming(decode_options, next_chunk, on_text_delta);
    debug::trace_log_scalar("nemotron_asr.streaming.chunks", chunk_count);
    return decoded;
}

NemotronASRStreamingSession::NemotronASRStreamingSession(
    runtime::TaskSpec task,
    runtime::SessionOptions options,
    std::shared_ptr<const NemotronAssets> assets)
    : NemotronASRSessionBase(task, std::move(options), std::move(assets)) {}

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
    policy.preferred_audio_chunk_samples = assets_->config.frontend.sample_rate;
    return policy;
}

void NemotronASRStreamingSession::start_stream(const runtime::TaskRequest & request) {
    reset();
    streaming_options_ = request.options;
    streaming_language_ = request.text_input.has_value() ? request.text_input->language : "";
    if (const auto option = runtime::find_option(request.options, {"language"})) {
        streaming_language_ = *option;
    }
}

void NemotronASRStreamingSession::set_stream_event_sink(runtime::StreamEventCallback sink) {
    stream_event_sink_ = std::move(sink);
}

void NemotronASRStreamingSession::reset() {
    require_prepared("Nemotron ASR reset()");
    if (task_.mode != runtime::RunMode::Streaming) {
        throw std::runtime_error("Nemotron ASR reset called on non-streaming session");
    }
    streaming_audio_ = runtime::AudioBuffer{};
}

runtime::StreamEvent NemotronASRStreamingSession::process_audio_chunk(const runtime::AudioChunk & chunk) {
    require_prepared("Nemotron ASR process_audio_chunk()");
    if (task_.mode != runtime::RunMode::Streaming) {
        throw std::runtime_error("Nemotron ASR process_audio_chunk called on non-streaming session");
    }
    runtime::AudioBuffer audio;
    audio.sample_rate = chunk.sample_rate;
    audio.channels = chunk.channels;
    audio.samples = chunk.samples;
    runtime::append_audio_buffer(streaming_audio_, audio);
    runtime::StreamEvent event;
    event.is_final = false;
    return event;
}

runtime::TaskResult NemotronASRStreamingSession::finalize() {
    require_prepared("Nemotron ASR finalize()");
    if (task_.mode != runtime::RunMode::Streaming) {
        throw std::runtime_error("Nemotron ASR finalize called on non-streaming session");
    }
    if (streaming_audio_.samples.empty()) {
        throw std::runtime_error("Nemotron ASR finalize requires streamed audio");
    }
    const auto wall_start = Clock::now();
    runtime::TaskRequest config_request;
    config_request.text_input = runtime::Transcript{"", streaming_language_};
    config_request.options = streaming_options_;
    const int64_t prompt_id = prompt_id_for_request(config_request);
    const int64_t lookahead = lookahead_for_options(streaming_options_);
    const auto decode_options = decode_options_for_request(config_request);
    const auto decoded = run_streaming_audio(
        streaming_audio_,
        prompt_id,
        lookahead,
        decode_options,
        [&](const std::string & delta) {
            if (!stream_event_sink_ || delta.empty()) {
                return;
            }
            runtime::StreamEvent event;
            event.partial_text = runtime::Transcript{delta, streaming_language_};
            stream_event_sink_(event);
        });
    runtime::TaskResult result;
    result.text_output = runtime::Transcript{decoded.text, streaming_language_};
    result.word_timestamps = decoded.token_timestamps;
    debug::timing_log_scalar("session.wall_ms", engine::debug::elapsed_ms(wall_start, Clock::now()));
    return result;
}

runtime::TaskResult NemotronASRStreamingSession::finish_stream() {
    return finalize();
}

}  // namespace engine::models::nemotron_asr
