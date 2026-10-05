#include "engine/models/ace_step/cuda_graph_execution.h"
#include "engine/models/ace_step/vae_encoder.h"
#include "vae_common.h"
#include "engine/framework/core/backend.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/io/binary.h"
#include "engine/framework/modules/primitive_modules.h"
#include "engine/framework/sampling/torch_random.h"

#include <ggml-backend.h>
#include <ggml-alloc.h>
#include <ggml.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace engine::models::ace_step::vae_common {
namespace {

core::TensorValue build_encoder_block(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const EncoderBlockWeights & weights,
    int64_t in_channels,
    int64_t out_channels) {
    auto hidden = build_residual_unit(ctx, input, weights.res1, in_channels, 1);
    hidden = build_residual_unit(ctx, hidden, weights.res2, in_channels, 3);
    hidden = build_residual_unit(ctx, hidden, weights.res3, in_channels, 9);
    hidden = build_snake1d_exact_bct(ctx, hidden, weights.snake, in_channels);
    hidden = build_conv1d(ctx, hidden, weights.conv, in_channels, out_channels, true);
    return hidden;
}

}  // namespace

VAEEncoderWeights load_vae_encoder_weights(
    const AceStepAssets & assets,
    ggml_backend_t backend,
    core::BackendType backend_type,
    size_t weight_context_bytes,
    assets::TensorStorageType storage_type) {
    const auto & config = assets.config.vae;
    if (config.encoder_hidden_size <= 0 ||
        config.decoder_channels <= 0 ||
        config.channel_multiples.empty() || config.downsampling_ratios.empty()) {
        throw std::runtime_error("ACE-Step VAE config is incomplete for encoder loading");
    }
    const auto & source = *assets.vae_weights;
    VAEEncoderWeights weights = {};
    weights.store = std::make_shared<core::BackendWeightStore>(
        backend,
        backend_type,
        "ace_step.vae.encoder.weights",
        weight_context_bytes);

    const int64_t encoder_hidden = config.decoder_channels;
    const int64_t encoder_output_channels = config.encoder_hidden_size;

    weights.encoder_conv1 = load_weight_norm_conv1d(
        *weights.store,
        source,
        "encoder.conv1",
        encoder_hidden,
        config.audio_channels,
        7,
        1,
        3,
        1,
        storage_type,
        true);

    weights.encoder_blocks.reserve(config.downsampling_ratios.size());
    for (size_t i = 0; i < config.downsampling_ratios.size(); ++i) {
        const int64_t stride = config.downsampling_ratios[i];
        const int64_t in_channels = encoder_hidden * (i == 0 ? 1 : config.channel_multiples[i - 1]);
        const int64_t out_channels = encoder_hidden * config.channel_multiples[i];
        const std::string prefix = "encoder.block." + std::to_string(i);
        EncoderBlockWeights block = {};
        block.res1 = load_residual_unit(*weights.store, source, prefix + ".res_unit1", in_channels, storage_type);
        block.res2 = load_residual_unit(*weights.store, source, prefix + ".res_unit2", in_channels, storage_type);
        block.res3 = load_residual_unit(*weights.store, source, prefix + ".res_unit3", in_channels, storage_type);
        block.snake = load_snake_exact(*weights.store, source, prefix + ".snake1", in_channels);
        block.conv = load_weight_norm_conv1d(
            *weights.store,
            source,
            prefix + ".conv1",
            out_channels,
            in_channels,
            2 * stride,
            static_cast<int>(stride),
            static_cast<int>((stride + 1) / 2),
            1,
            storage_type,
            true);
        weights.encoder_blocks.push_back(std::move(block));
    }
    const int64_t encoder_deepest_channels = encoder_hidden * config.channel_multiples.back();
    weights.encoder_snake_out = load_snake_exact(
        *weights.store,
        source,
        "encoder.snake1",
        encoder_deepest_channels);
    weights.encoder_conv2 = load_weight_norm_conv1d(
        *weights.store,
        source,
        "encoder.conv2",
        encoder_output_channels,
        encoder_deepest_channels,
        3,
        1,
        1,
        1,
        storage_type,
        true);
    weights.store->upload();
    return weights;
}

AceStepVAEEncodeGraph::AceStepVAEEncodeGraph(
    std::shared_ptr<const AceStepAssets> assets,
    ggml_backend_t backend,
    core::BackendType backend_type,
    int threads,
    std::shared_ptr<const VAEEncoderWeights> weights,
    int64_t audio_frames,
    size_t graph_arena_bytes)
    : assets_(std::move(assets)),
      backend_(backend),
      backend_type_(backend_type),
      threads_(threads),
      weights_(std::move(weights)),
      audio_frames_(audio_frames) {
    build(graph_arena_bytes);
}

AceStepVAEEncodeGraph::~AceStepVAEEncodeGraph() {
    if (backend_ != nullptr && graph_ != nullptr) {
        engine::core::release_backend_graph_resources(backend_, graph_);
    }
    if (gallocr_ != nullptr) {
        ggml_gallocr_free(gallocr_);
    }
}

bool AceStepVAEEncodeGraph::can_run(int64_t audio_frames) const noexcept {
    return audio_frames_ == audio_frames;
}

AceStepLatents AceStepVAEEncodeGraph::encode(
    const runtime::AudioBuffer & audio,
    uint64_t seed,
    uint64_t & noise_offset,
    const std::vector<float> * noise_override) const {
    const auto total_start = Clock::now();
    const auto & config = assets_->config.vae;
    if (audio.sample_rate != config.sample_rate ||
        audio.channels != config.audio_channels ||
        audio.samples.empty()) {
        throw std::runtime_error("ACE-Step VAE encoder audio shape mismatch");
    }
    const int64_t request_audio_frames =
        static_cast<int64_t>(audio.samples.size() / static_cast<size_t>(audio.channels));
    if (request_audio_frames <= 0 || request_audio_frames > audio_frames_) {
        throw std::runtime_error("ACE-Step VAE encoder frame count is invalid");
    }

    const auto input_start = Clock::now();
    bct_input_scratch_.resize(static_cast<size_t>(audio_frames_ * audio.channels));
    if (request_audio_frames < audio_frames_) {
        std::fill(bct_input_scratch_.begin(), bct_input_scratch_.end(), 0.0F);
    }
    for (int channel = 0; channel < audio.channels; ++channel) {
        float * channel_input = bct_input_scratch_.data() + static_cast<size_t>(channel * audio_frames_);
        for (int64_t frame = 0; frame < request_audio_frames; ++frame) {
            channel_input[static_cast<size_t>(frame)] =
                audio.samples[static_cast<size_t>(frame * audio.channels + channel)];
        }
    }
    core::write_tensor_f32(input_value_, bct_input_scratch_);
    core::set_backend_threads(backend_, threads_);
    engine::debug::timing_log_scalar(
        "ace_step.vae.encode.input_upload_ms",
        engine::debug::elapsed_ms(input_start, Clock::now()));
    const auto compute_start = Clock::now();
    const ggml_status status = ace_step_compute_backend_graph(backend_, graph_);
    ggml_backend_synchronize(backend_);
    if (status != GGML_STATUS_SUCCESS) {
        throw std::runtime_error("ACE-Step VAE encoder graph compute failed");
    }
    engine::debug::timing_log_scalar(
        "ace_step.vae.encode.graph.compute_ms",
        engine::debug::elapsed_ms(compute_start, Clock::now()));

    const auto output_start = Clock::now();
    bct_output_scratch_.resize(static_cast<size_t>(latent_frames_ * config.encoder_hidden_size));
    core::read_tensor_f32_into(output_value_.tensor, bct_output_scratch_);
    const int64_t request_latent_frames =
        std::max<int64_t>(1, request_audio_frames / audio_frames_per_latent(config));
    if (request_latent_frames > latent_frames_) {
        throw std::runtime_error("ACE-Step VAE encoder request latent length exceeds graph capacity");
    }
    AceStepLatents out;
    out.frames = request_latent_frames;
    out.channels = latent_channels_;
    out.values.resize(static_cast<size_t>(out.frames * out.channels), 0.0F);
    const size_t latent_value_count = static_cast<size_t>(out.frames * out.channels);
    if (noise_override != nullptr && noise_override->size() < latent_value_count) {
        throw std::runtime_error(
            "ACE-Step VAE encoder noise file is too short for source-audio latent sampling");
    }
    std::vector<float> channel_major_noise;
    if (noise_override == nullptr) {
        channel_major_noise = engine::sampling::generate_torch_cuda_randn(
            latent_value_count,
            seed,
            engine::sampling::TorchRandnPrecision::Float32,
            noise_offset);
        noise_offset += static_cast<uint64_t>(latent_value_count);
    }
    for (int64_t frame = 0; frame < request_latent_frames; ++frame) {
        for (int64_t channel = 0; channel < latent_channels_; ++channel) {
            const size_t latent_index = static_cast<size_t>(frame * latent_channels_ + channel);
            const float mean =
                bct_output_scratch_[static_cast<size_t>(channel * latent_frames_ + frame)];
            const float scale =
                bct_output_scratch_[static_cast<size_t>((channel + latent_channels_) * latent_frames_ + frame)];
            const float stddev = std::log1p(std::exp(scale)) + 1.0e-4F;
            const float noise = noise_override == nullptr
                ? channel_major_noise[static_cast<size_t>(channel * request_latent_frames + frame)]
                : (*noise_override)[latent_index];
            out.values[latent_index] = mean + stddev * noise;
        }
    }
    engine::debug::timing_log_scalar(
        "ace_step.vae.encode.output_read_ms",
        engine::debug::elapsed_ms(output_start, Clock::now()));
    engine::debug::timing_log_scalar("ace_step.vae.encode.total_ms", engine::debug::elapsed_ms(total_start, Clock::now()));
    return out;
}

void AceStepVAEEncodeGraph::build(size_t graph_arena_bytes) {
    const auto & config = assets_->config.vae;
    ggml_init_params params{graph_arena_bytes, nullptr, true};
    ctx_.reset(ggml_init(params));
    if (ctx_ == nullptr) {
        throw std::runtime_error("ACE-Step VAE encoder ggml context initialization failed");
    }
    core::ModuleBuildContext ctx{ctx_.get(), "ace_step.vae_encoder", backend_type_};

    input_value_ = core::make_tensor(
        ctx,
        GGML_TYPE_F32,
        core::TensorShape::from_dims({1, config.audio_channels, audio_frames_}));
    ggml_set_input(input_value_.tensor);

    auto hidden = build_conv1d(
        ctx,
        input_value_,
        weights_->encoder_conv1,
        config.audio_channels,
        config.decoder_channels,
        true);
    for (size_t i = 0; i < weights_->encoder_blocks.size(); ++i) {
        const auto & block = weights_->encoder_blocks[i];
        const int64_t in_channels = hidden.shape.dims[1];
        const int64_t out_channels = block.conv.conv.weight.shape.dims[0];
        hidden = build_encoder_block(ctx, hidden, block, in_channels, out_channels);
    }
    hidden = build_snake1d_exact_bct(
        ctx,
        hidden,
        weights_->encoder_snake_out,
        hidden.shape.dims[1]);
    hidden = build_conv1d(
        ctx,
        hidden,
        weights_->encoder_conv2,
        hidden.shape.dims[1],
        config.encoder_hidden_size,
        true);
    output_value_ = hidden;
    latent_frames_ = output_value_.shape.dims[2];
    if (output_value_.shape.dims[1] % 2 != 0) {
        throw std::runtime_error("ACE-Step VAE encoder output channels must split into mean/scale pairs");
    }
    latent_channels_ = output_value_.shape.dims[1] / 2;
    ggml_set_output(output_value_.tensor);
    graph_ = ggml_new_graph_custom(ctx_.get(), 131072, false);
    ggml_build_forward_expand(graph_, output_value_.tensor);
    gallocr_ = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend_));
    if (gallocr_ == nullptr || !ggml_gallocr_alloc_graph(gallocr_, graph_)) {
        throw std::runtime_error("ACE-Step VAE encoder backend buffer allocation failed");
    }
}

AceStepVAEEncoderRuntimeCore::AceStepVAEEncoderRuntimeCore(
    std::shared_ptr<const AceStepAssets> assets,
    ggml_backend_t backend,
    core::BackendType backend_type,
    int threads,
    std::shared_ptr<const VAEEncoderWeights> weights,
    size_t graph_arena_bytes)
    : assets_(std::move(assets)),
      backend_(backend),
      backend_type_(backend_type),
      threads_(threads),
      weights_(std::move(weights)),
      graph_arena_bytes_(graph_arena_bytes) {}

void AceStepVAEEncoderRuntimeCore::ensure_graph(int64_t audio_frames) {
    if (graph_ && graph_->can_run(audio_frames)) {
        return;
    }
    graph_.reset();
    graph_ = std::make_unique<AceStepVAEEncodeGraph>(
        assets_,
        backend_,
        backend_type_,
        threads_,
        weights_,
        audio_frames,
        graph_arena_bytes_);
}

AceStepLatents AceStepVAEEncoderRuntimeCore::encode(
    const runtime::AudioBuffer & audio,
    uint32_t seed,
    const std::string & noise_file) {
    const auto total_start = Clock::now();
    if (audio.sample_rate != assets_->config.vae.sample_rate ||
        audio.channels != assets_->config.vae.audio_channels ||
        audio.samples.empty()) {
        throw std::runtime_error("ACE-Step VAE encoder requires stereo 48k audio input");
    }
    constexpr int64_t kChunkFrames = 30 * 48000;
    constexpr int64_t kChunkOverlapFrames = 2 * 48000;
    const int64_t total_frames =
        static_cast<int64_t>(audio.samples.size() / static_cast<size_t>(audio.channels));
    std::vector<float> noise_override;
    if (!noise_file.empty()) {
        noise_override = io::read_f32_file(noise_file);
    }
    uint64_t noise_offset = 0;
    if (total_frames <= kChunkFrames) {
        const auto ensure_start = Clock::now();
        ensure_graph(total_frames);
        engine::debug::timing_log_scalar(
            "ace_step.vae.encode.graph.ensure_ms",
            engine::debug::elapsed_ms(ensure_start, Clock::now()));
        AceStepLatents out = graph_->encode(
            audio,
            seed,
            noise_offset,
            noise_override.empty() ? nullptr : &noise_override);
        engine::debug::timing_log_scalar(
            "ace_step.vae.encode.runtime_total_ms",
            engine::debug::elapsed_ms(total_start, Clock::now()));
        return out;
    }

    const int64_t stride = kChunkFrames - 2 * kChunkOverlapFrames;
    if (stride <= 0) {
        throw std::runtime_error("ACE-Step VAE encoder chunked stride must be positive");
    }
    if (!noise_override.empty()) {
        throw std::runtime_error("ACE-Step controlled VAE encode noise is not implemented for chunked source audio");
    }
    AceStepLatents stitched = {};
    const int64_t num_steps = (total_frames + stride - 1) / stride;
    double latent_hop = 0.0;
    double ensure_graph_ms = 0.0;
    double chunk_encode_ms = 0.0;
    for (int64_t i = 0; i < num_steps; ++i) {
        const int64_t core_start = i * stride;
        const int64_t core_end = std::min<int64_t>(core_start + stride, total_frames);
        const int64_t win_start = std::max<int64_t>(0, core_start - kChunkOverlapFrames);
        const int64_t win_end = std::min<int64_t>(total_frames, core_end + kChunkOverlapFrames);
        runtime::AudioBuffer window = {};
        window.sample_rate = audio.sample_rate;
        window.channels = audio.channels;
        const int64_t window_frames = win_end - win_start;
        window.samples.resize(static_cast<size_t>(window_frames * audio.channels), 0.0F);
        std::copy_n(
            audio.samples.data() + static_cast<size_t>(win_start * audio.channels),
            window.samples.size(),
            window.samples.data());
        const auto ensure_start = Clock::now();
        ensure_graph(window_frames);
        ensure_graph_ms += engine::debug::elapsed_ms(ensure_start, Clock::now());
        const auto chunk_start = Clock::now();
        AceStepLatents tile = graph_->encode(window, seed, noise_offset);
        chunk_encode_ms += engine::debug::elapsed_ms(chunk_start, Clock::now());
        if (stitched.channels == 0) {
            stitched.channels = tile.channels;
        }
        if (latent_hop == 0.0) {
            latent_hop = static_cast<double>(window_frames) / static_cast<double>(tile.frames);
        }
        const int64_t trim_left = static_cast<int64_t>(std::llround(
            static_cast<double>(core_start - win_start) / latent_hop));
        const int64_t trim_right = static_cast<int64_t>(std::llround(
            static_cast<double>(win_end - core_end) / latent_hop));
        const int64_t emit_end = tile.frames - trim_right;
        if (trim_left < 0 || emit_end < trim_left || emit_end > tile.frames) {
            throw std::runtime_error("ACE-Step VAE encoder trim range is invalid");
        }
        const int64_t emit_frames = emit_end - trim_left;
        const float * src = tile.values.data() + static_cast<size_t>(trim_left * tile.channels);
        stitched.values.insert(
            stitched.values.end(),
            src,
            src + static_cast<std::ptrdiff_t>(emit_frames * tile.channels));
        stitched.frames += emit_frames;
    }
    engine::debug::timing_log_scalar("ace_step.vae.encode.graph.ensure_ms", ensure_graph_ms);
    engine::debug::timing_log_scalar("ace_step.vae.encode.chunk_encode_ms", chunk_encode_ms);
    engine::debug::timing_log_scalar("ace_step.vae.encode.runtime_total_ms", engine::debug::elapsed_ms(total_start, Clock::now()));
    return stitched;
}

void AceStepVAEEncoderRuntimeCore::release_runtime_graphs() {
    graph_.reset();
}

}  // namespace engine::models::ace_step::vae_common

namespace engine::models::ace_step {

using namespace vae_common;

class AceStepVAEEncoderRuntime::Impl {
public:
    Impl(
        std::shared_ptr<const AceStepAssets> assets,
        core::ExecutionContext & execution,
        assets::TensorStorageType weight_storage_type,
        size_t graph_arena_bytes,
        size_t weight_context_bytes)
        : assets_(require_assets(std::move(assets))),
          backend_(require_backend(execution)),
          backend_type_(execution.backend_type()),
          threads_(std::max(1, execution.config().threads)),
          weights_(assets_->backend_weights_cache.acquire<VAEEncoderWeights>(
              execution, weight_storage_type, [&](core::ExecutionContext & upload) {
                  auto weights = std::make_shared<VAEEncoderWeights>(load_vae_encoder_weights(
                      *assets_, upload.backend(), upload.backend_type(), weight_context_bytes, weight_storage_type));
                  assets_->vae_weights->release_storage();
                  return weights;
              })),
          runtime_(
              std::make_unique<AceStepVAEEncoderRuntimeCore>(
                  assets_,
                  backend_,
                  backend_type_,
                  threads_,
                  weights_,
                  graph_arena_bytes)) {
    }

    AceStepLatents encode(const runtime::AudioBuffer & audio, uint32_t seed, const std::string & noise_file) {
        return runtime_->encode(audio, seed, noise_file);
    }

    void release_runtime_graphs() {
        runtime_->release_runtime_graphs();
    }

private:
    static std::shared_ptr<const AceStepAssets> require_assets(std::shared_ptr<const AceStepAssets> assets) {
        if (assets == nullptr) {
            throw std::runtime_error("ACE-Step VAE encoder runtime requires assets");
        }
        return assets;
    }

    static ggml_backend_t require_backend(core::ExecutionContext & execution) {
        ggml_backend_t backend = execution.backend();
        if (backend == nullptr) {
            throw std::runtime_error("ACE-Step VAE encoder backend is not initialized");
        }
        return backend;
    }

    std::shared_ptr<const AceStepAssets> assets_;
    ggml_backend_t backend_ = nullptr;
    core::BackendType backend_type_ = core::BackendType::Cpu;
    int threads_ = 1;
    std::shared_ptr<const VAEEncoderWeights> weights_;
    std::unique_ptr<AceStepVAEEncoderRuntimeCore> runtime_;
};

AceStepVAEEncoderRuntime::AceStepVAEEncoderRuntime(
    std::shared_ptr<const AceStepAssets> assets,
    core::ExecutionContext & execution,
    assets::TensorStorageType weight_storage_type,
    size_t graph_arena_bytes,
    size_t weight_context_bytes)
    : impl_(std::make_unique<Impl>(
          std::move(assets),
          execution,
          weight_storage_type,
          graph_arena_bytes,
          weight_context_bytes)) {}

AceStepVAEEncoderRuntime::~AceStepVAEEncoderRuntime() = default;

AceStepLatents AceStepVAEEncoderRuntime::encode(
    const runtime::AudioBuffer & audio,
    uint32_t seed,
    const std::string & noise_file) {
    return impl_->encode(audio, seed, noise_file);
}

void AceStepVAEEncoderRuntime::release_runtime_graphs() const {
    impl_->release_runtime_graphs();
}

}  // namespace engine::models::ace_step
