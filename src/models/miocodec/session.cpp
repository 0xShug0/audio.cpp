#include "engine/models/miocodec/session.h"

#include "engine/models/miocodec/audio_pipeline.h"

#include "engine/framework/audio/dsp.h"
#include "engine/framework/modules/speech_encoders/wavlm_encoder.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/runtime/options.h"
#include "engine/models/miocodec/assets.h"
#include "engine/models/miocodec/components.h"
#include "engine/models/miocodec/weights.h"

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace engine::models::miocodec {
namespace {

using Clock = std::chrono::steady_clock;

constexpr size_t kDefaultCodecWeightContextBytes = 256ull * 1024ull * 1024ull;
constexpr size_t kDefaultContentGraphArenaBytes = 512ull * 1024ull * 1024ull;
constexpr size_t kDefaultGlobalGraphArenaBytes = 256ull * 1024ull * 1024ull;
constexpr size_t kDefaultWaveGraphArenaBytes = 512ull * 1024ull * 1024ull;
constexpr size_t kDefaultConstantContextBytes = 256ull * 1024ull * 1024ull;
constexpr int64_t kMioCodecWaveHeadBins = 394;

engine::assets::TensorStorageType parse_miocodec_weight_type(
    const runtime::SessionOptions & options) {
    const auto it = options.options.find("miocodec.weight_type");
    if (it == options.options.end()) {
        return engine::assets::TensorStorageType::F32;
    }
    const auto storage_type = engine::assets::parse_tensor_storage_type(it->second);
    if (storage_type == engine::assets::TensorStorageType::Native ||
        storage_type == engine::assets::TensorStorageType::F32 ||
        storage_type == engine::assets::TensorStorageType::F16 ||
        storage_type == engine::assets::TensorStorageType::BF16 ||
        storage_type == engine::assets::TensorStorageType::Q8_0) {
        return storage_type;
    }
    throw std::runtime_error("miocodec.weight_type currently supports only native, f32, f16, bf16, and q8_0");
}

}  // namespace

MioCodecSession::MioCodecSession(
    const runtime::TaskSpec & task,
    const runtime::SessionOptions & options,
    std::shared_ptr<const MioCodecAssets> assets)
    : runtime::RuntimeSessionBase(options),
      task_(task),
      assets_(std::move(assets)) {
    if (assets_ == nullptr) {
        throw std::runtime_error("MioCodec session requires loaded assets");
    }
    if (task_.mode != runtime::RunMode::Offline) {
        throw std::runtime_error("MioCodec only supports offline sessions");
    }
    if (task_.task != runtime::VoiceTaskKind::VoiceConversion
        && task_.task != runtime::VoiceTaskKind::SpeechToSpeech) {
        throw std::runtime_error("MioCodec supports VoiceConversion or SpeechToSpeech tasks");
    }

    const size_t weight_context_bytes = runtime::parse_size_mb_option(
        options.options,
        {"miocodec.weight_context_mb", "weight_context_mb"},
        kDefaultCodecWeightContextBytes);
    const auto weight_storage_type = parse_miocodec_weight_type(options);
    codec_weights_ = load_miocodec_weights(
        *assets_->model_weights,
        execution_context(),
        weight_context_bytes,
        assets_->config,
        weight_storage_type);
    engine::modules::WavlmEncoderConfig wavlm_config;
    wavlm_config.output_hidden_layer = 9;
    wavlm_config.weight_storage_type = weight_storage_type;
    auto local_ssl_wavlm = engine::modules::WavlmEncoderComponent::load_from_tensor_source(
        *assets_->wavlm_weights,
        options.backend,
        wavlm_config);
    auto global_ssl_wavlm = engine::modules::WavlmEncoderComponent(local_ssl_wavlm.weights(), options.backend, true);
    const size_t constant_context_bytes = runtime::parse_size_mb_option(
        options.options,
        {"miocodec.constant_context_mb", "constant_context_mb"},
        kDefaultConstantContextBytes);
    const size_t content_graph_arena_bytes = runtime::parse_size_mb_option(
        options.options,
        {"miocodec.content_graph_arena_mb", "graph_arena_mb"},
        kDefaultContentGraphArenaBytes);
    const size_t global_graph_arena_bytes = runtime::parse_size_mb_option(
        options.options,
        {"miocodec.global_graph_arena_mb"},
        kDefaultGlobalGraphArenaBytes);
    const size_t wave_graph_arena_bytes = runtime::parse_size_mb_option(
        options.options,
        {"miocodec.wave_graph_arena_mb"},
        kDefaultWaveGraphArenaBytes);
    content_encoder_ = std::make_unique<MioCodecContentEncoderRuntime>(
        assets_,
        codec_weights_,
        execution_context(),
        content_graph_arena_bytes,
        constant_context_bytes);
    global_encoder_ = std::make_unique<MioCodecGlobalEncoderRuntime>(
        assets_,
        codec_weights_,
        execution_context(),
        global_graph_arena_bytes);
    local_ssl_ = std::make_unique<MioCodecSslFeatureExtractor>(
        assets_,
        std::move(local_ssl_wavlm),
        assets_->config.local_ssl_layers,
        assets_->config.normalize_ssl_features,
        "miocodec.local_ssl");
    global_reference_ = std::make_unique<MioCodecGlobalReferenceEncoder>(
        MioCodecSslFeatureExtractor(
            assets_,
            std::move(global_ssl_wavlm),
            assets_->config.global_ssl_layers,
            false,
            "miocodec.global_ssl"),
        *global_encoder_);
    wave_decoder_ = std::make_unique<MioCodecWaveDecoderRuntime>(
        assets_,
        codec_weights_,
        execution_context(),
        wave_graph_arena_bytes,
        constant_context_bytes);
    waveform_reconstructor_ = std::make_unique<MioCodecWaveformReconstructor>(
        assets_,
        execution_context());
}

MioCodecSession::~MioCodecSession() = default;

std::string MioCodecSession::family() const {
    return "miocodec";
}

runtime::VoiceTaskKind MioCodecSession::task_kind() const {
    return task_.task;
}

runtime::RunMode MioCodecSession::run_mode() const {
    return task_.mode;
}

void MioCodecSession::prepare(const runtime::SessionPreparationRequest &) {
    mark_prepared();
}

runtime::TaskResult MioCodecSession::run(const runtime::TaskRequest & request) {
    const auto wall_start = Clock::now();
    require_prepared("MioCodec run");
    if (!request.audio_input.has_value()) {
        throw std::runtime_error("MioCodec run() requires audio_input");
    }
    if (!request.voice.has_value() || !request.voice->speaker.has_value() ||
        !request.voice->speaker->audio.has_value()) {
        throw std::runtime_error("MioCodec run() requires voice speaker audio for the global reference");
    }
    auto timing_start = Clock::now();
    const auto source_audio = prepare_miocodec_mono_audio(*request.audio_input, assets_->config.sample_rate);
    engine::debug::timing_log_scalar("miocodec.audio.source_prepare_ms", engine::debug::elapsed_ms(timing_start));
    timing_start = Clock::now();
    const auto reference_audio = prepare_miocodec_mono_audio(*request.voice->speaker->audio, assets_->config.sample_rate);
    engine::debug::timing_log_scalar("miocodec.audio.reference_prepare_ms", engine::debug::elapsed_ms(timing_start));
    timing_start = Clock::now();
    const auto source_ssl = local_ssl_->extract(source_audio);
    engine::debug::timing_log_scalar("miocodec.local_ssl_ms", engine::debug::elapsed_ms(timing_start));
    timing_start = Clock::now();
    const auto content = content_encoder_->encode(source_ssl.values, source_ssl.frames);
    engine::debug::timing_log_scalar("miocodec.content_encoder_ms", engine::debug::elapsed_ms(timing_start));
    timing_start = Clock::now();
    const auto global = global_reference_->embedding_for_reference(reference_audio);
    engine::debug::timing_log_scalar("miocodec.global_reference_ms", engine::debug::elapsed_ms(timing_start));
    const int upsample_factor = std::accumulate(
        assets_->config.wave_upsampler_factors.begin(),
        assets_->config.wave_upsampler_factors.end(),
        1,
        [](int lhs, int rhs) {
            return lhs * rhs;
        });
    const auto stft_frames =
        static_cast<int64_t>(source_audio.size()) / assets_->config.hop_length / upsample_factor;
    if (stft_frames <= 0) {
        throw std::runtime_error("MioCodec run() audio input is too short for the configured wave decoder");
    }
    timing_start = Clock::now();
    const auto head = wave_decoder_->decode(content, global, stft_frames);
    engine::debug::timing_log_scalar("miocodec.wave_decoder_ms", engine::debug::elapsed_ms(timing_start));
    if (head.bins != kMioCodecWaveHeadBins) {
        throw std::runtime_error("MioCodec wave head bin count mismatch");
    }
    const engine::audio::STFTConfig stft_config{
        assets_->config.n_fft,
        assets_->config.hop_length,
        assets_->config.n_fft,
        true,
        engine::audio::STFTPadMode::Reflect,
        engine::audio::STFTFamily::Kokoro,
    };
    timing_start = Clock::now();
    runtime::TaskResult result;
    result.audio_output = runtime::AudioBuffer{
        assets_->config.sample_rate,
        1,
        waveform_reconstructor_->reconstruct(head, engine::audio::get_cached_stft_window(stft_config)),
    };
    engine::debug::timing_log_scalar("miocodec.istft_ms", engine::debug::elapsed_ms(timing_start));
    engine::debug::timing_log_scalar("session.wall_ms", engine::debug::elapsed_ms(wall_start));
    return result;
}

const std::shared_ptr<const MioCodecWeights> & MioCodecSession::codec_weights() const noexcept {
    return codec_weights_;
}

const engine::modules::WavlmEncoderComponent & MioCodecSession::wavlm() const noexcept {
    return local_ssl_->wavlm();
}

}  // namespace engine::models::miocodec
