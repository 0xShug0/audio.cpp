#include "engine/models/nemotron_asr/session.h"

#include "engine/framework/debug/profiler.h"
#include "engine/framework/runtime/options.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <stdexcept>
#include <utility>
#include <vector>

namespace engine::models::nemotron_asr {
namespace {

using Clock = std::chrono::steady_clock;

constexpr size_t kDefaultWeightContextBytes = 3072ull * 1024ull * 1024ull;
constexpr size_t kDefaultEncoderGraphArenaBytes = 1024ull * 1024ull * 1024ull;
constexpr size_t kDefaultDecoderGraphArenaBytes = 256ull * 1024ull * 1024ull;

std::shared_ptr<const NemotronASRAssets> require_assets(std::shared_ptr<const NemotronASRAssets> assets) {
    if (assets == nullptr) {
        throw std::runtime_error("Nemotron ASR session requires assets");
    }
    return assets;
}

// Opt-in per-chunk debugging (NEMOTRON_DUMP_CHUNKS=<path prefix>): writes the
// mel features and the encoder output of every streaming chunk as little-endian
// f32 blobs plus a .meta sidecar with the dimensions, so a reference
// implementation (transformers nemotron_asr_streaming) can be diffed
// chunk-for-chunk and frame-for-frame.
void dump_stream_chunk(
    const std::string & prefix,
    int64_t seq,
    const NemotronFrontendFeatures & mel,
    const NemotronEncodedAudio & enc,
    bool center) {
    const char * env = std::getenv("NEMOTRON_DUMP_CHUNKS");
    if (env == nullptr || *env == '\0') {
        return;
    }
    const std::string base = std::string(env) + "_c" + std::to_string(seq);
    auto write_f32 = [&](const std::string & path, const float * data, size_t count) {
        std::ofstream out(path, std::ios::binary);
        out.write(reinterpret_cast<const char *>(data), static_cast<std::streamsize>(count * sizeof(float)));
    };
    write_f32(base + "_mel.f32", mel.values.data(), static_cast<size_t>(mel.frames * mel.feature_dim));
    write_f32(base + "_enc.f32", enc.values.data(), static_cast<size_t>(enc.frames * enc.hidden_size));
    std::ofstream meta(base + ".meta");
    meta << "mel_frames=" << mel.frames << " mel_valid=" << mel.valid_frames
         << " mel_dim=" << mel.feature_dim << " enc_frames=" << enc.frames
         << " enc_valid=" << enc.valid_frames << " enc_hidden=" << enc.hidden_size
         << " center=" << (center ? 1 : 0) << "\n";
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
        storage_type == engine::assets::TensorStorageType::Q8_0 ||
        storage_type == engine::assets::TensorStorageType::Q4_0 ||
        storage_type == engine::assets::TensorStorageType::Q4_1 ||
        storage_type == engine::assets::TensorStorageType::Q5_0 ||
        storage_type == engine::assets::TensorStorageType::Q5_1 ||
        storage_type == engine::assets::TensorStorageType::Q4_K ||
        storage_type == engine::assets::TensorStorageType::Q5_K ||
        storage_type == engine::assets::TensorStorageType::Q6_K) {
        // Sub-q8_0 types are re-quantized from the source weights at load
        // (dequant -> ggml_quantize_chunk): faster CPU GEMMs, some accuracy risk.
        return;
    }
    throw std::runtime_error(std::string(option_name) + " supports only native, f32, f16, bf16, q8_0, q4_0, q4_1, q5_0, q5_1, q4_k, q5_k, and q6_k");
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
        throw std::runtime_error("Nemotron ASR streaming feature slice is out of range (start=" +
                                 std::to_string(start_frame) + ", frames=" + std::to_string(frames) +
                                 ", in.frames=" + std::to_string(in.frames) + ")");
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
    std::shared_ptr<const NemotronASRAssets> assets)
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
    // Streaming encoder graphs are metadata-only arenas but the prefix ladder
    // multiplies them (up to ~15 variants at lookahead 0). The streaming graph
    // caps its node array at 64k entries (~3.4k used), so a 16 MB per-variant
    // arena holds the metadata with a wide margin — about a sixth of the old
    // 96 MB per-variant commit. An explicit
    // nemotron_asr.encoder_graph_arena_mb option is honored as-is for both.
    constexpr size_t kDefaultStreamEncoderGraphArenaBytes = 16ull * 1024ull * 1024ull;
    const size_t stream_arena_bytes = encoder_graph_arena_bytes_ == kDefaultEncoderGraphArenaBytes
        ? kDefaultStreamEncoderGraphArenaBytes
        : encoder_graph_arena_bytes_;
    encoder_ = std::make_unique<NemotronEncoderRuntime>(
        assets_,
        weights_,
        execution_context(),
        encoder_graph_arena_bytes_,
        stream_arena_bytes);
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
    std::shared_ptr<const NemotronASRAssets> assets)
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
    // The GGUF embeds supported {0,3,6,13}, but the model card declares chunk
    // durations 80-1120 ms (lookahead 0..13) as pure runtime knobs. Accept 1
    // (160 ms chunks) on that basis; its geometry (9-frame first window) is
    // well-formed, unlike lookahead 0's degenerate single-frame first window.
    const auto supported = assets_->config.encoder.supported_lookahead_tokens;
    if (lookahead != 1 &&
        std::find(supported.begin(), supported.end(), lookahead) == supported.end()) {
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
    // The first chunk must cover at least one full encoded frame (subsampling
    // factor mel frames — the subsampling conv cannot produce output from less),
    // otherwise encoded frame 0 is computed from zero-padded cache frames.
    const int64_t first_mel_frames = std::max<int64_t>(
        assets_->config.encoder.subsampling_factor,
        1 + assets_->config.encoder.subsampling_factor * lookahead);
    // The sliding chunk carries at least 4 encoded frames: at lookahead 0 the
    // emit-all schedule is exact for any chunk size (no frame needs right
    // context), and larger chunks amortize the per-graph overheads — 1-frame
    // chunks at 80 ms measurably fall behind realtime on one CPU thread.
    const int64_t mel_frames_per_chunk = assets_->config.encoder.subsampling_factor *
        std::max<int64_t>(lookahead + 1, 4);
    const int64_t first_samples = (first_mel_frames - 1) * fc.hop_length + fc.win_length / 2;
    const int64_t samples_per_chunk = mel_frames_per_chunk * fc.hop_length + fc.win_length;
    auto waveform = frontend_.prepare_waveform(audio);
    if (static_cast<int64_t>(waveform.size()) < first_samples) {
        // Ultra-short turn: silence-pad to the first required chunk rather than
        // failing the whole request — the flush below keeps the stream well-formed.
        waveform.resize(static_cast<size_t>(first_samples), 0.0f);
    }
    NemotronEncoderStreamState stream_state = encoder_->make_stream_state();
    bool first_chunk = true;
    bool flushed = false;
    int64_t chunk_count = 0;
    int64_t mel_frame_idx = first_mel_frames;
    int64_t start_idx = mel_frame_idx * fc.hop_length - fc.n_fft / 2;
    // Window [start_idx, start_idx + samples_per_chunk) centered on the chunk's
    // mel frames. start_idx goes negative when the window precedes the signal
    // start (lookahead 0: the second window begins at 1*hop - n_fft/2 = -96);
    // the left context is silence then, exactly like the first chunk's center
    // pad — so build the window zero-padded instead of indexing before begin().
    auto window_at = [&](int64_t from) -> std::vector<float> {
        std::vector<float> window(static_cast<size_t>(samples_per_chunk), 0.0f);
        const int64_t copy_from = std::max<int64_t>(from, 0);
        const int64_t copy_to = std::min<int64_t>(
            from + samples_per_chunk, static_cast<int64_t>(waveform.size()));
        if (copy_to > copy_from) {
            std::copy(
                waveform.begin() + static_cast<std::ptrdiff_t>(copy_from),
                waveform.begin() + static_cast<std::ptrdiff_t>(copy_to),
                window.begin() + static_cast<std::ptrdiff_t>(copy_from - from));
        }
        return window;
    };
    auto next_chunk = [&](NemotronEncodedAudio & out) -> bool {
        if (first_chunk) {
            first_chunk = false;
            ++chunk_count;
            std::vector<float> chunk_waveform(
                waveform.begin(),
                waveform.begin() + static_cast<std::ptrdiff_t>(first_samples));
            auto features = frontend_.extract_waveform(chunk_waveform, true);
            if (features.frames > first_mel_frames) {
                features = slice_features(features, 0, first_mel_frames);  // whole-buffer first chunk
            }
            out = encoder_->encode_stream_chunk(features, prompt_id, lookahead, stream_state);
            return true;
        }
        if (start_idx + samples_per_chunk >= static_cast<int64_t>(waveform.size())) {
            // End of stream: the tail no longer fills a full window. Zero-pad it to
            // the full window (the reference processor right-pads the final chunk)
            // and encode one last chunk. Dropping the tail loses the last word(s)
            // of every turn whose audio does not align with the window stride —
            // and entire short utterances ("Hello"), whose transcript came back
            // empty because nothing beyond the first chunk was ever encoded.
            if (flushed || start_idx + fc.n_fft > static_cast<int64_t>(waveform.size())) {
                // Already flushed, or the leftover is too short to contribute even
                // one full mel frame (it is inside the previous window's right pad).
                return false;
            }
            flushed = true;
            ++chunk_count;
            auto chunk_waveform = window_at(start_idx);
            auto features = frontend_.extract_waveform(chunk_waveform, false);
            if (features.frames != mel_frames_per_chunk) {
                throw std::runtime_error("Nemotron ASR streaming frontend produced unexpected flush chunk frame count");
            }
            out = encoder_->encode_stream_chunk(features, prompt_id, lookahead, stream_state);
            return true;
        }
        auto chunk_waveform = window_at(start_idx);
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
    std::shared_ptr<const NemotronASRAssets> assets)
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

    // Derive the native chunk geometry and the incremental pipeline state. The
    // window math mirrors run_streaming_audio() so a chunked session and a
    // whole-buffer session encode identical windows.
    runtime::TaskRequest config_request;
    config_request.text_input = runtime::Transcript{"", streaming_language_};
    config_request.options = streaming_options_;
    stream_prompt_id_ = prompt_id_for_request(config_request);
    stream_lookahead_ = lookahead_for_options(streaming_options_);
    stream_decode_options_ = decode_options_for_request(config_request);

    const auto & fc = assets_->config.frontend;
    const auto & enc = assets_->config.encoder;
    stream_first_mel_frames_ = std::max<int64_t>(
        enc.subsampling_factor,
        1 + enc.subsampling_factor * stream_lookahead_);
    stream_mel_frames_per_chunk_ = enc.subsampling_factor *
        std::max<int64_t>(stream_lookahead_ + 1, 4);
    stream_first_samples_ = (stream_first_mel_frames_ - 1) * fc.hop_length + fc.win_length / 2;
    stream_samples_per_chunk_ = stream_mel_frames_per_chunk_ * fc.hop_length + fc.win_length;
    stream_next_chunk_start_ = stream_first_mel_frames_ * fc.hop_length - fc.n_fft / 2;
    stream_await_first_chunk_ = true;
    stream_tail_encoded_ = false;
    stream_decode_active_ = false;
    stream_dump_chunk_seq_ = 0;
    spec_valid_ = false;
    spec_frames_ = NemotronEncodedAudio{};
    spec_last_loud_sample_ = 0;
    spec_audio_mark_ = 0;
    encoder_stream_state_ = encoder_->make_stream_state();
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
    stream_await_first_chunk_ = true;
    stream_tail_encoded_ = false;
    stream_decode_active_ = false;
}

bool NemotronASRStreamingSession::encode_and_decode_next_chunk(bool flush_tail, std::string & delta_out) {
    if (stream_tail_encoded_) {
        return false;
    }
    const auto & fc = assets_->config.frontend;
    const int64_t total = static_cast<int64_t>(streaming_audio_.samples.size());

    std::vector<float> window;
    bool center = false;
    bool reuse_spec = false;
    if (stream_await_first_chunk_) {
        if (total < stream_first_samples_) {
            return false;
        }
        window.assign(
            streaming_audio_.samples.begin(),
            streaming_audio_.samples.begin() + static_cast<std::ptrdiff_t>(stream_first_samples_));
        center = true;
    } else if (stream_next_chunk_start_ + stream_samples_per_chunk_ < total) {
        // start_idx goes negative when the window precedes the signal start
        // (lookahead 0: the second window begins at 1*hop - n_fft/2 = -96); the
        // left context is silence then, exactly like the first chunk's center
        // pad — zero-pad instead of indexing before begin().
        const int64_t copy_from = std::max<int64_t>(stream_next_chunk_start_, 0);
        window.assign(
            streaming_audio_.samples.begin() + static_cast<std::ptrdiff_t>(copy_from),
            streaming_audio_.samples.begin() + static_cast<std::ptrdiff_t>(stream_next_chunk_start_ + stream_samples_per_chunk_));
        window.insert(
            window.begin(),
            static_cast<size_t>(std::max<int64_t>(0, -stream_next_chunk_start_)),
            0.0f);
    } else if (flush_tail && stream_next_chunk_start_ < total) {
        // The tail never fills a whole native chunk: zero-pad it so the final
        // audio is encoded too (the padding decodes to blank tokens).
        //
        // The pad MUST carry ~500 ms of post-speech silence: the RNNT fires a
        // word's trailing token only after that much silence in the encoded
        // stream (measured: 210 ms fails, 530 ms works; the padding content is
        // irrelevant — the reference wavs' own tails are digital zeros). The
        // flush window is sized for it (flush_window_mel), and when the
        // speculative flush already encoded this window, its frames are
        // reused instead of re-encoding.
        if (spec_valid_) {
            reuse_spec = true;
        } else {
            build_flush_window(total, window);
        }
        stream_tail_encoded_ = true;
    } else {
        return false;
    }

    NemotronEncodedAudio encoded;
    if (reuse_spec) {
        encoded = std::move(spec_frames_);
        spec_frames_ = NemotronEncodedAudio{};
    } else {
        auto features = frontend_.extract_waveform(window, center);
        if (center && features.frames > stream_first_mel_frames_) {
                features = slice_features(features, 0, stream_first_mel_frames_);
        }
        encoded = encoder_->encode_stream_chunk(
            features,
            stream_prompt_id_,
            stream_lookahead_,
            encoder_stream_state_);
        dump_stream_chunk("stream", stream_dump_chunk_seq_, features, encoded, center);
        ++stream_dump_chunk_seq_;
    }

    if (!stream_decode_active_) {
        decoder_->begin_stream_decode(stream_decode_options_);
        stream_decode_active_ = true;
    }
    decoder_->decode_stream_chunk(encoded, [&](const std::string & delta) {
        delta_out += delta;
    });

    if (stream_await_first_chunk_) {
        stream_await_first_chunk_ = false;
        stream_next_chunk_start_ = stream_first_mel_frames_ * fc.hop_length - fc.n_fft / 2;
    } else if (!reuse_spec) {
        stream_next_chunk_start_ += stream_mel_frames_per_chunk_ * fc.hop_length;
    }
    return true;
}

int64_t NemotronASRStreamingSession::flush_window_mel() const {
    const auto & enc = assets_->config.encoder;
    const char * env = std::getenv("NEMOTRON_FLUSH_WINDOW_MEL");
    const int64_t requested = env != nullptr && *env != 0 ? std::strtoll(env, nullptr, 10) : 8 * 8;
    return std::max<int64_t>(enc.subsampling_factor * std::max<int64_t>(stream_lookahead_ + 1, 4), requested);
}

void NemotronASRStreamingSession::build_flush_window(int64_t total, std::vector<float> & window) const {
    const auto & fc = assets_->config.frontend;
    const int64_t copy_from = std::max<int64_t>(stream_next_chunk_start_, 0);
    window.assign(
        streaming_audio_.samples.begin() + static_cast<std::ptrdiff_t>(copy_from),
        streaming_audio_.samples.begin() + static_cast<std::ptrdiff_t>(total));
    if (stream_next_chunk_start_ < 0) {
        window.insert(window.begin(), static_cast<size_t>(-stream_next_chunk_start_), 0.0f);
    }
    window.resize(static_cast<size_t>(flush_window_mel() * fc.hop_length + fc.win_length), 0.0f);
}

// Speculative flush: once the ingest sees enough trailing silence, encode the
// flush window right here on the ingest thread (the backend cannot run
// concurrent graphs, and during silence there is no realtime encode to
// contend with). Finalize then reuses the frames instead of paying the encode
// after end of turn. The result is discarded — and the encoder state restored
// from the snapshot — whenever speech resumes or a full chunk shifts the
// window origin, because the padded frames are only valid while the audio
// after the snapshot stays silent.
void NemotronASRStreamingSession::maybe_speculative_flush() {
    if (spec_valid_ || stream_await_first_chunk_ || stream_tail_encoded_ || !stream_decode_active_) {
        return;
    }
    const char * env = std::getenv("NEMOTRON_SPEC_FLUSH");
    if (env != nullptr && *env == '0') {
        return;
    }
    const auto & fc = assets_->config.frontend;
    const int64_t total = static_cast<int64_t>(streaming_audio_.samples.size());
    const int64_t flush_samples = flush_window_mel() * fc.hop_length + fc.win_length;
    if (stream_next_chunk_start_ + flush_samples <= total) {
        return;
    }
    const char * silence_env = std::getenv("NEMOTRON_SPEC_SILENCE_MS");
    const double silence_s = silence_env != nullptr && *silence_env != 0 ? std::strtod(silence_env, nullptr) / 1000.0 : 0.15;
    if (total - spec_last_loud_sample_ < static_cast<int64_t>(silence_s * fc.sample_rate)) {
        return;
    }
    spec_state_snapshot_ = encoder_stream_state_;
    std::vector<float> window;
    build_flush_window(total, window);
    auto features = frontend_.extract_waveform(window, /*center=*/false);
    spec_frames_ = encoder_->encode_stream_chunk(
        features,
        stream_prompt_id_,
        stream_lookahead_,
        encoder_stream_state_);
    spec_audio_mark_ = total;
    spec_valid_ = true;
}

void NemotronASRStreamingSession::discard_speculative_flush() {
    if (!spec_valid_) {
        return;
    }
    encoder_stream_state_ = spec_state_snapshot_;
    spec_frames_ = NemotronEncodedAudio{};
    spec_valid_ = false;
}

runtime::StreamEvent NemotronASRStreamingSession::process_audio_chunk(const runtime::AudioChunk & chunk) {
    require_prepared("Nemotron ASR process_audio_chunk()");
    if (task_.mode != runtime::RunMode::Streaming) {
        throw std::runtime_error("Nemotron ASR process_audio_chunk called on non-streaming session");
    }
    const char * energy_env = std::getenv("NEMOTRON_SPEC_ENERGY");
    const double energy = energy_env != nullptr && *energy_env != 0 ? std::strtod(energy_env, nullptr) : 0.01;
    const int64_t scan_from = static_cast<int64_t>(streaming_audio_.samples.size());
    runtime::AudioBuffer audio;
    audio.sample_rate = chunk.sample_rate;
    audio.channels = chunk.channels;
    audio.samples = chunk.samples;
    runtime::append_audio_buffer(streaming_audio_, audio);
    const int64_t total = static_cast<int64_t>(streaming_audio_.samples.size());
    // 10 ms windowed RMS: the speech/silence distinction must ignore decay
    // transients (a per-sample max counts the quiet tail of a word as speech
    // and the speculative flush never fires).
    constexpr int64_t kRmsWindow = 160;
    for (int64_t w = scan_from - scan_from % kRmsWindow; w + kRmsWindow <= total; w += kRmsWindow) {
        double acc = 0.0;
        for (int64_t i = w; i < w + kRmsWindow; ++i) {
            const float v = streaming_audio_.samples[static_cast<size_t>(i)];
            acc += double(v) * double(v);
        }
        if (std::sqrt(acc / double(kRmsWindow)) > energy) {
            spec_last_loud_sample_ = w + kRmsWindow;
        }
    }

    runtime::StreamEvent event;
    event.is_final = false;
    std::string delta;
    bool encoded_full_chunk = false;
    while (encode_and_decode_next_chunk(/*flush_tail=*/false, delta)) {
        encoded_full_chunk = true;
    }
    if (spec_valid_ && (encoded_full_chunk || spec_last_loud_sample_ > spec_audio_mark_)) {
        discard_speculative_flush();
    }
    maybe_speculative_flush();
    if (!delta.empty()) {
        event.partial_text = runtime::Transcript{delta, streaming_language_};
        if (stream_event_sink_) {
            stream_event_sink_(event);
            return {};
        }
    }
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
    std::string delta;
    while (encode_and_decode_next_chunk(/*flush_tail=*/true, delta)) {
    }
    if (!stream_decode_active_) {
        throw std::runtime_error("Nemotron ASR streaming request is shorter than the first required chunk");
    }
    const auto decoded = decoder_->finish_stream_decode();
    // Safety net: a blank incremental final is re-decoded through the offline
    // encoder (full-context single pass). At the default lookahead 3 the
    // chunk-end frames lack right context and short utterances can come back
    // blank; at lookahead 0 the schedule is exact and this rarely fires.
    const auto is_blank = [](const std::string & text) {
        return text.find_first_not_of(" \t\n\r") == std::string::npos;
    };
    if (is_blank(decoded.text)) {
        // The full-context path is the OFFLINE encoder — one pass over the whole
        // turn with full attention and no chunk windows. (run_streaming_audio
        // shares the chunked streaming graph and its chunk-end truncation, so it
        // does not help here.)
        //
        // The re-decode input is padded with ~500 ms of silence: the greedy
        // RNNT has cut-length dead pockets where it emits nothing at all
        // (measured on hi.wav: cuts at 560-570 ms decode empty in the f32
        // reference too, while 550 and 580 decode fine). Extra silence moves
        // the frame grid out of the pocket without touching the speech.
        auto padded_audio = streaming_audio_;
        padded_audio.samples.resize(
            padded_audio.samples.size() + static_cast<size_t>(assets_->config.frontend.sample_rate / 2),
            0.0f);
        const auto frontend = frontend_.extract(padded_audio, true);
        const auto encoded = encoder_->encode(frontend, stream_prompt_id_, stream_lookahead_);
        auto full = decoder_->decode(encoded, stream_decode_options_);
        if (is_blank(full.text)) {
            // Still blank — retry once with a longer pad before giving up.
            padded_audio.samples.resize(
                padded_audio.samples.size() + static_cast<size_t>(assets_->config.frontend.sample_rate),
                0.0f);
            const auto frontend2 = frontend_.extract(padded_audio, true);
            const auto encoded2 = encoder_->encode(frontend2, stream_prompt_id_, stream_lookahead_);
            full = decoder_->decode(encoded2, stream_decode_options_);
        }
        debug::trace_log_scalar(
            "nemotron_asr.streaming.empty_final_rerun",
            full.text.empty() ? 0 : 1);
        runtime::TaskResult result;
        result.text_output = runtime::Transcript{full.text, streaming_language_};
        result.word_timestamps = full.token_timestamps;
        debug::timing_log_scalar("session.wall_ms", engine::debug::elapsed_ms(wall_start, Clock::now()));
        return result;
    }
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
