#include "engine/models/nemotron_asr/session.h"

#include "engine/framework/debug/profiler.h"
#include "engine/framework/runtime/options.h"
#include "engine/framework/runtime/spec_backed_model.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
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

void attribute_word(SpeakerSegmentBuilder & builder, const SpeakerProbabilities & probabilities, const TaggedWord & word) {
    builder.append_word(
        attribute_speaker(probabilities, word), word.text,
        static_cast<double>(word.first_frame) * kSpeakerFrameSeconds,
        static_cast<double>(word.last_frame + 1) * kSpeakerFrameSeconds);
}

void attach_speaker_outputs(runtime::TaskResult & result, const std::vector<SpeakerSegment> & segments, bool masked) {
    result.speaker_turns = segments_to_turns(segments);
    result.output_artifacts.push_back(runtime::make_text_artifact(
        runtime::ArtifactKind::TranscriptAlignment, "seglst", seglst_json(segments, "session_0"),
        // SegLST is JSON (RFC 6839 +json suffix); the CLI writes it as seglst.json.
        {{"extension", "json"}, {"mime", "application/seglst+json"}}));
    if (masked && result.text_output.has_value()) {
        result.text_output->text = segments_to_lines(segments);
    }
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
    // Masked speaker tagging runs chunked streams and never uses the offline graph.
    const auto tagging = speaker_tagging_for_options(request.options);
    const bool masked = tagging.has_value() && tagging->masked;
    if (frames > 0 && !mem_saver_ && !masked) {
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

std::optional<SpeakerTaggingOptions> NemotronASRSessionBase::speaker_tagging_for_options(
    const std::unordered_map<std::string, std::string> & options) const {
    const auto path = runtime::find_option(options, {"speaker_probabilities"});
    if (!path.has_value()) {
        for (const char * key : {"speaker_mode", "speaker_mask", "speaker_segment_gap_sec", "speaker_segment_max_sec"}) {
            if (options.count(key) != 0) {
                throw std::runtime_error(std::string("Nemotron ASR ") + key + " requires speaker_probabilities");
            }
        }
        return std::nullopt;
    }
    SpeakerTaggingOptions out;
    out.probabilities = *path;
    const auto mode = runtime::find_option(options, {"speaker_mode"}).value_or("masked");
    if (mode != "attribution" && mode != "masked") {
        throw std::runtime_error("Nemotron ASR speaker_mode must be attribution or masked");
    }
    out.masked = mode == "masked";
    const auto mask = runtime::find_option(options, {"speaker_mask"}).value_or("mel");
    if (mask != "mel" && mask != "audio") {
        throw std::runtime_error("Nemotron ASR speaker_mask must be mel or audio");
    }
    out.audio_mask = mask == "audio";
    if (const auto value = runtime::parse_finite_float_option(options, {"speaker_segment_gap_sec"})) out.gap_sec = *value;
    if (const auto value = runtime::parse_finite_float_option(options, {"speaker_segment_max_sec"})) out.max_event_sec = *value;
    if (out.gap_sec < 0.0 || out.max_event_sec <= 0.0) {
        throw std::runtime_error("Nemotron ASR speaker_segment_gap_sec must be >= 0 and speaker_segment_max_sec > 0");
    }
    return out;
}

int64_t NemotronASRSessionBase::masked_lookahead(
    const std::unordered_map<std::string, std::string> & options,
    const SpeakerProbabilities & probabilities) const {
    std::optional<int64_t> requested;
    if (options.count("lookahead_tokens") != 0) requested = lookahead_for_options(options);
    const auto choice = resolve_masked_lookahead(
        probabilities.metadata, requested, assets_->config.encoder.supported_lookahead_tokens);
    for (const auto & warning : choice.warnings) {
        std::cerr << "[warning][nemotron_asr] " << warning << "\n";
    }
    return choice.lookahead;
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
    const auto tagging = speaker_tagging_for_options(request.options);
    std::optional<SpeakerProbabilities> speaker_probabilities;
    auto waveform = frontend_.prepare_waveform(*request.audio_input);
    if (tagging.has_value()) {
        speaker_probabilities = load_speaker_probabilities(tagging->probabilities);
        require_speaker_probability_rows(*speaker_probabilities, static_cast<int64_t>(waveform.size()));
        if (tagging->masked) {
            MaskedSpeakerStreams streams(
                *encoder_, *decoder_, frontend_, *speaker_probabilities, *tagging, prompt_id,
                masked_lookahead(request.options, *speaker_probabilities), decode_options);
            streams.push_audio(std::move(waveform));
            streams.process(true);
            runtime::TaskResult result;
            result.text_output = runtime::Transcript{"", request.text_input.has_value() ? request.text_input->language : ""};
            attach_speaker_outputs(result, streams.segments().segments(), true);
            debug::timing_log_scalar("session.wall_ms", engine::debug::elapsed_ms(wall_start, Clock::now()));
            return result;
        }
    }

    NemotronDecodedText decoded;
    const auto frontend = frontend_.extract_waveform(waveform, true);
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
    if (tagging.has_value()) {
        SpeakerSegmentBuilder builder(tagging->gap_sec, tagging->max_event_sec);
        for (const auto & word : group_words(result.word_timestamps)) {
            attribute_word(builder, *speaker_probabilities, word);
        }
        attach_speaker_outputs(result, builder.segments(), false);
    }
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
    stream_tagging_ = speaker_tagging_for_options(streaming_options_);
    if (stream_tagging_.has_value()) {
        stream_speaker_probabilities_ = load_speaker_probabilities(stream_tagging_->probabilities);
        if (stream_tagging_->masked) {
            masked_streams_ = std::make_unique<MaskedSpeakerStreams>(
                *encoder_, *decoder_, frontend_, *stream_speaker_probabilities_, *stream_tagging_, prompt_id_,
                masked_lookahead(streaming_options_, *stream_speaker_probabilities_),
                decode_options_for_request(config_request));
        } else {
            stream_segments_ = std::make_unique<SpeakerSegmentBuilder>(
                stream_tagging_->gap_sec, stream_tagging_->max_event_sec);
        }
    }
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
    stream_tagging_.reset();
    stream_speaker_probabilities_.reset();
    stream_segments_.reset();
    masked_streams_.reset();
    masked_text_.clear();
    stream_attributed_words_ = 0;
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
    received_samples_ += static_cast<int64_t>(chunk.samples.size());
    if (masked_streams_ != nullptr) {
        masked_streams_->push_audio(chunk.samples.data(), chunk.samples.size());
        masked_streams_->process(false);
        auto event = take_masked_event(false);
        if (stream_event_sink_ && !event.speaker_turns.empty()) {
            stream_event_sink_(event);
            return {};
        }
        return event;
    }
    streaming_waveform_.insert(
        streaming_waveform_.end(), chunk.samples.begin(), chunk.samples.end());
    return process_available_chunks(false);
}

void NemotronASRStreamingSession::process_feature_chunk(
    const NemotronFrontendFeatures & features) {
    auto encoded = encoder_->encode_stream_chunk(
        features, prompt_id_, lookahead_, encoder_stream_state_);
    decoder_->decode_stream_chunk(encoded, decoder_stream_state_);
    ++chunks_processed_;
}

std::vector<runtime::SpeakerTurn> NemotronASRStreamingSession::attribute_stream_words(bool final) {
    // ponytail: rebuilds every token timestamp per chunk, O(tokens) each; make it incremental if long streams show it.
    const auto words = group_words(decoder_->stream_result(decoder_stream_state_).token_timestamps);
    // The newest word can still grow, so it waits for the next word (or the end).
    const size_t ready = final || words.empty() ? words.size() : words.size() - 1;
    for (; stream_attributed_words_ < ready; ++stream_attributed_words_) {
        attribute_word(*stream_segments_, *stream_speaker_probabilities_, words[stream_attributed_words_]);
    }
    double stream_time = static_cast<double>(decoder_stream_state_.encoded_frames) * kSpeakerFrameSeconds;
    if (ready < words.size()) {
        // A held-back word may still extend the latest segment; do not close it by pause.
        stream_time = std::min(stream_time, static_cast<double>(words[ready].first_frame) * kSpeakerFrameSeconds);
    }
    return segments_to_turns(stream_segments_->take_events(stream_time, final));
}

runtime::StreamEvent NemotronASRStreamingSession::take_masked_event(bool final) {
    const auto pieces = masked_streams_->segments().take_events(masked_streams_->stream_time(), final);
    runtime::StreamEvent event;
    if (pieces.empty()) return event;
    event.speaker_turns = segments_to_turns(pieces);
    const auto lines = segments_to_lines(pieces);
    masked_text_ += lines;
    event.partial_text = runtime::Transcript{lines, streaming_language_};
    return event;
}

runtime::StreamEvent NemotronASRStreamingSession::publish_stream_update() {
    runtime::StreamEvent event;
    if (stream_segments_ != nullptr) {
        event.speaker_turns = attribute_stream_words(false);
    }
    auto delta = partials_.publish(decoder_stream_state_.decoded.text);
    if (delta.empty() && event.speaker_turns.empty()) {
        return event;
    }
    if (!delta.empty()) {
        event.partial_text = runtime::Transcript{std::move(delta), streaming_language_};
    }
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
        if (!stream_event_sink_) {
            combined.speaker_turns.insert(
                combined.speaker_turns.end(), event.speaker_turns.begin(), event.speaker_turns.end());
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
    if (stream_speaker_probabilities_.has_value()) {
        require_speaker_probability_rows(*stream_speaker_probabilities_, received_samples_);
    }
    if (masked_streams_ != nullptr) {
        masked_streams_->process(true);
        const auto event = take_masked_event(true);
        if (stream_event_sink_ && !event.speaker_turns.empty()) stream_event_sink_(event);
        runtime::TaskResult result;
        result.text_output = runtime::Transcript{"", streaming_language_};
        attach_speaker_outputs(result, masked_streams_->segments().segments(), true);
        // The streamed lines, in the order segments finished, so the deltas add up to it.
        result.text_output->text = masked_text_;
        finalized_ = true;
        stream_started_ = false;
        return result;
    }
    (void) process_available_chunks(true);
    auto decoded = decoder_->stream_result(decoder_stream_state_);
    runtime::TaskResult result;
    result.text_output = runtime::Transcript{decoded.text, streaming_language_};
    result.word_timestamps = std::move(decoded.token_timestamps);
    if (stream_segments_ != nullptr) {
        runtime::StreamEvent event;
        event.speaker_turns = attribute_stream_words(true);
        if (stream_event_sink_ && !event.speaker_turns.empty()) stream_event_sink_(event);
        attach_speaker_outputs(result, stream_segments_->segments(), false);
    }
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
