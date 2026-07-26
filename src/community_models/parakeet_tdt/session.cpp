#include "engine/community_models/parakeet_tdt/session.h"

#include "engine/framework/debug/profiler.h"
#include "engine/framework/runtime/options.h"

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

std::shared_ptr<const ParakeetTDTAssets> require_assets(std::shared_ptr<const ParakeetTDTAssets> assets) {
    if (assets == nullptr) {
        throw std::runtime_error("Parakeet TDT session requires assets");
    }
    return assets;
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

}  // namespace

ParakeetTDTSessionBase::ParakeetTDTSessionBase(
    runtime::TaskSpec task,
    runtime::SessionOptions options,
    std::shared_ptr<const ParakeetTDTAssets> assets)
    : RuntimeSessionBase(options),
      task_(task),
      assets_(require_assets(std::move(assets))),
      weight_context_bytes_(runtime::parse_size_mb_option(options.options, {"parakeet_tdt.weight_context_mb"}, kDefaultWeightContextBytes)),
      encoder_graph_arena_bytes_(runtime::parse_size_mb_option(options.options, {"parakeet_tdt.encoder_graph_arena_mb"}, kDefaultEncoderGraphArenaBytes)),
      decoder_graph_arena_bytes_(runtime::parse_size_mb_option(options.options, {"parakeet_tdt.decoder_graph_arena_mb"}, kDefaultDecoderGraphArenaBytes)),
      matmul_weight_storage_type_(option_weight_type(
          options,
          "parakeet_tdt.matmul_weight_type",
          option_weight_type(options, "parakeet_tdt.weight_type", engine::assets::TensorStorageType::Native))),
      conv_weight_storage_type_(option_weight_type(options, "parakeet_tdt.conv_weight_type", engine::assets::TensorStorageType::Native)),
      encoder_flash_attention_(runtime::parse_bool_option(
          runtime::find_option(options.options, {"parakeet_tdt.encoder_flash_attention"}).value_or("false"),
          "parakeet_tdt.encoder_flash_attention")),
      frontend_(assets_) {
    if (task_.task != runtime::VoiceTaskKind::Asr) {
        throw std::runtime_error("Parakeet TDT only supports VoiceTaskKind::Asr");
    }
    if (task_.mode != runtime::RunMode::Offline) {
        throw std::runtime_error("Parakeet TDT currently only supports offline sessions");
    }
    validate_matmul_weight_storage(matmul_weight_storage_type_, "parakeet_tdt.weight_type");
    validate_conv_weight_storage(conv_weight_storage_type_, "parakeet_tdt.conv_weight_type");
    for (const auto & [key, value] : options.options) {
        (void)value;
        if (key.rfind("parakeet_tdt.", 0) == 0 &&
            key != "parakeet_tdt.weight_context_mb" &&
            key != "parakeet_tdt.encoder_graph_arena_mb" &&
            key != "parakeet_tdt.decoder_graph_arena_mb" &&
            key != "parakeet_tdt.weight_type" &&
            key != "parakeet_tdt.matmul_weight_type" &&
            key != "parakeet_tdt.conv_weight_type" &&
            key != "parakeet_tdt.encoder_flash_attention") {
            throw std::runtime_error("unknown Parakeet TDT session option: " + key);
        }
    }
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
    std::shared_ptr<const ParakeetTDTAssets> assets)
    : ParakeetTDTSessionBase(task, std::move(options), std::move(assets)) {}

std::string ParakeetTDTOfflineSession::family() const { return family_impl(); }
runtime::VoiceTaskKind ParakeetTDTOfflineSession::task_kind() const { return task_kind_impl(); }
runtime::RunMode ParakeetTDTOfflineSession::run_mode() const { return run_mode_impl(); }

void ParakeetTDTOfflineSession::prepare(const runtime::SessionPreparationRequest & request) {
    const auto prepare_start = Clock::now();
    if (!request.audio.has_value()) {
        throw std::runtime_error("Parakeet TDT prepare() requires an audio contract");
    }
    const int64_t frames = frontend_frames_for_samples(
        request.audio->max_input_samples,
        request.audio->channels,
        request.audio->sample_rate,
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
    const auto wall_start = Clock::now();
    const auto decode_options = decode_options_for_request(request);

    const auto frontend = frontend_.extract(*request.audio_input, true);
    const auto encoded = encoder_->encode(frontend);
    auto decoded = decoder_->decode(encoded, decode_options);

    runtime::TaskResult result;
    result.text_output = runtime::Transcript{decoded.text, ""};
    result.word_timestamps = std::move(decoded.token_timestamps);
    debug::timing_log_scalar("session.wall_ms", engine::debug::elapsed_ms(wall_start, Clock::now()));
    return result;
}

}  // namespace engine::community_models::parakeet_tdt
