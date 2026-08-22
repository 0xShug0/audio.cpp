#include "engine/models/midashenglm_gen/audio_tokenizer.h"

#include "engine/framework/audio/istft_graph.h"
#include "engine/framework/core/backend.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/modules/activation_modules.h"
#include "engine/framework/modules/primitive_modules.h"
#include "engine/framework/modules/structural_modules.h"
#include "engine/framework/modules/weight_binding.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <stdexcept>
#include <string>
#include <utility>

namespace engine::models::midashenglm_gen {
namespace {

namespace binding = engine::modules::binding;
namespace core = engine::core;
namespace modules = engine::modules;
using Clock = std::chrono::steady_clock;

constexpr int64_t kConvNeXtKernel = 7;
constexpr int64_t kUpsampleKernel = 2;

struct GgmlContextDeleter {
    void operator()(ggml_context * ctx) const noexcept {
        if (ctx != nullptr) {
            ggml_free(ctx);
        }
    }
};

core::TensorValue vocos_backbone(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input_bct,
    const MiDashengLmGenConfig & config,
    const MiDashengLmGenAudioTokenizerWeights & weights) {
    auto x = modules::Conv1dModule(
                 {config.audio_embedding_size, config.audio_embedding_size, kConvNeXtKernel, 1, 3, 1, true})
                 .build(ctx, input_bct, weights.embed);
    x = modules::TransposeModule({{0, 2, 1, 3}, 3}).build(ctx, x);
    x = modules::LayerNormModule({config.audio_embedding_size, 1.0e-6F, true, true}).build(ctx, x, weights.norm);
    x = modules::TransposeModule({{0, 2, 1, 3}, 3}).build(ctx, x);
    for (const auto & block : weights.blocks) {
        const auto residual = x;
        x = modules::DepthwiseConv1dModule(
                {config.audio_embedding_size, kConvNeXtKernel, 1, 3, 1, true})
                .build(ctx, x, block.depthwise);
        x = modules::TransposeModule({{0, 2, 1, 3}, 3}).build(ctx, x);
        x = modules::LayerNormModule({config.audio_embedding_size, 1.0e-6F, true, true}).build(ctx, x, block.norm);
        x = modules::LinearModule({config.audio_embedding_size, config.decoder_intermediate_size, true})
                .build(ctx, x, block.pointwise_in);
        x = modules::GeluModule({modules::GeluApproximation::ExactErf}).build(ctx, x);
        x = modules::LinearModule({config.decoder_intermediate_size, config.audio_embedding_size, true})
                .build(ctx, x, block.pointwise_out);
        const auto gamma = modules::RepeatModule({x.shape}).build(
            ctx,
            core::reshape_tensor(
                ctx,
                block.gamma,
                core::TensorShape::from_dims({1, 1, config.audio_embedding_size})));
        x = modules::MulModule{}.build(ctx, x, gamma);
        x = modules::TransposeModule({{0, 2, 1, 3}, 3}).build(ctx, x);
        x = modules::AddModule{}.build(ctx, residual, x);
    }
    x = modules::TransposeModule({{0, 2, 1, 3}, 3}).build(ctx, x);
    return modules::LayerNormModule({config.audio_embedding_size, 1.0e-6F, true, true})
        .build(ctx, x, weights.final_norm);
}

MiDashengLmGenVocosBlockWeights load_vocos_block(
    core::BackendWeightStore & store,
    const engine::assets::TensorSource & source,
    const std::string & prefix,
    const MiDashengLmGenConfig & config,
    engine::assets::TensorStorageType matmul_storage_type,
    engine::assets::TensorStorageType conv_storage_type) {
    MiDashengLmGenVocosBlockWeights block;
    block.depthwise = binding::depthwise_conv1d_from_source(
        store,
        source,
        prefix + ".dwconv",
        conv_storage_type,
        config.audio_embedding_size,
        kConvNeXtKernel,
        true);
    block.norm = binding::norm_from_source(store, source, prefix + ".norm", config.audio_embedding_size);
    block.pointwise_in = binding::linear_from_source(
        store,
        source,
        prefix + ".pwconv1",
        matmul_storage_type,
        config.decoder_intermediate_size,
        config.audio_embedding_size,
        true);
    block.pointwise_out = binding::linear_from_source(
        store,
        source,
        prefix + ".pwconv2",
        matmul_storage_type,
        config.audio_embedding_size,
        config.decoder_intermediate_size,
        true);
    block.gamma = store.load_f32_tensor(source, prefix + ".gamma", {config.audio_embedding_size});
    return block;
}

std::shared_ptr<const MiDashengLmGenAudioTokenizerWeights> load_weights(
    const MiDashengLmGenAssets & assets,
    ggml_backend_t backend,
    core::BackendType backend_type,
    size_t weight_context_bytes,
    engine::assets::TensorStorageType matmul_storage_type,
    engine::assets::TensorStorageType conv_storage_type) {
    if (assets.weights == nullptr) {
        throw std::runtime_error("MiDashengLM-Gen audio tokenizer requires tensor source");
    }
    auto weights = std::make_shared<MiDashengLmGenAudioTokenizerWeights>();
    weights->store = std::make_shared<core::BackendWeightStore>(
        backend,
        backend_type,
        "midashenglm_gen.audio_tokenizer.weights",
        weight_context_bytes);
    const auto & source = *assets.weights;
    const auto & config = assets.config;
    weights->upsampler = binding::conv_transpose1d_from_source(
        *weights->store,
        source,
        "model.audio_upsampler",
        conv_storage_type,
        config.audio_embedding_size,
        config.audio_embedding_size,
        kUpsampleKernel,
        true);
    weights->embed = binding::conv1d_from_source(
        *weights->store,
        source,
        "model.audio_decoder.backbone.embed",
        conv_storage_type,
        config.audio_embedding_size,
        config.audio_embedding_size,
        kConvNeXtKernel,
        true);
    weights->norm = binding::norm_from_source(
        *weights->store,
        source,
        "model.audio_decoder.backbone.norm",
        config.audio_embedding_size);
    weights->blocks.reserve(static_cast<size_t>(config.decoder_layers));
    for (int64_t layer = 0; layer < config.decoder_layers; ++layer) {
        weights->blocks.push_back(load_vocos_block(
            *weights->store,
            source,
            "model.audio_decoder.backbone.convnext." + std::to_string(layer),
            config,
            matmul_storage_type,
            conv_storage_type));
    }
    weights->final_norm = binding::norm_from_source(
        *weights->store,
        source,
        "model.audio_decoder.backbone.final_layer_norm",
        config.audio_embedding_size);
    weights->head = binding::linear_from_source(
        *weights->store,
        source,
        "model.audio_decoder.head.out",
        matmul_storage_type,
        config.istft_n_fft + 2,
        config.audio_embedding_size,
        true);
    weights->istft_window = source.require_f32("model.audio_decoder.head.istft.window", {config.istft_n_fft});
    weights->store->upload();
    return weights;
}

}  // namespace

class MiDashengLmGenAudioTokenizerRuntime::DecodeGraph {
public:
    DecodeGraph(
        core::ExecutionContext & execution,
        std::shared_ptr<const MiDashengLmGenAudioTokenizerWeights> weights,
        MiDashengLmGenConfig config,
        int64_t batch,
        int64_t frames,
        size_t graph_arena_bytes)
        : execution_(execution),
          weights_(std::move(weights)),
          config_(config),
          batch_(batch),
          frames_(frames) {
        if (weights_ == nullptr) {
            throw std::runtime_error("MiDashengLM-Gen audio tokenizer decode graph requires weights");
        }
        if (batch_ <= 0 || frames_ <= 0) {
            throw std::runtime_error("MiDashengLM-Gen audio tokenizer decode graph requires positive batch and frames");
        }

        const auto build_start = Clock::now();
        ggml_init_params params{graph_arena_bytes, nullptr, true};
        ctx_.reset(ggml_init(params));
        if (ctx_ == nullptr) {
            throw std::runtime_error("failed to initialize MiDashengLM-Gen audio tokenizer decode graph context");
        }
        ggml_init_params input_params{32ull * 1024ull * 1024ull, nullptr, true};
        input_ctx_.reset(ggml_init(input_params));
        if (input_ctx_ == nullptr) {
            throw std::runtime_error("failed to initialize MiDashengLM-Gen audio tokenizer decode input context");
        }
        core::ModuleBuildContext ctx{ctx_.get(), "midashenglm_gen.audio_tokenizer.decode", execution_.backend_type()};
        core::ModuleBuildContext input_ctx{
            input_ctx_.get(),
            "midashenglm_gen.audio_tokenizer.decode.inputs",
            execution_.backend_type()};

        latents_ = core::make_tensor(
            input_ctx,
            GGML_TYPE_F32,
            core::TensorShape::from_dims({batch_, frames_, config_.audio_embedding_size})).tensor;
        ggml_set_input(latents_);
        auto x = core::wrap_tensor(
            latents_,
            core::TensorShape::from_dims({batch_, frames_, config_.audio_embedding_size}),
            GGML_TYPE_F32);
        x = modules::TransposeModule({{0, 2, 1, 3}, 3}).build(ctx, x);
        x = modules::ConvTranspose1dModule(
                {config_.audio_embedding_size, config_.audio_embedding_size, kUpsampleKernel, 2, 0, 1, true})
                .build(ctx, x, weights_->upsampler);
        x = vocos_backbone(ctx, x, config_, *weights_);
        x = modules::LinearModule({config_.audio_embedding_size, config_.istft_n_fft + 2, true})
                .build(ctx, x, weights_->head);
        output_ = core::ensure_backend_addressable_layout(ctx, x).tensor;
        ggml_set_output(output_);

        graph_ = ggml_new_graph_custom(
            ctx_.get(),
            static_cast<size_t>(std::max<int64_t>(65536, batch_ * frames_ * 4096 + 4096)),
            false);
        ggml_build_forward_expand(graph_, output_);
        input_buffer_ = ggml_backend_alloc_ctx_tensors(input_ctx_.get(), execution_.backend());
        if (input_buffer_ == nullptr) {
            throw std::runtime_error("failed to allocate MiDashengLM-Gen audio tokenizer decode input buffer");
        }
        gallocr_ = ggml_gallocr_new(ggml_backend_get_default_buffer_type(execution_.backend()));
        if (gallocr_ == nullptr ||
            !ggml_gallocr_reserve(gallocr_, graph_) ||
            !ggml_gallocr_alloc_graph(gallocr_, graph_)) {
            clear_graph();
            throw std::runtime_error("failed to allocate MiDashengLM-Gen audio tokenizer decode graph");
        }
        engine::debug::timing_log_scalar(
            "midashenglm_gen.audio_tokenizer.decode.graph.build_ms",
            engine::debug::elapsed_ms(build_start, Clock::now()));
    }

    ~DecodeGraph() {
        clear_graph();
    }

    int64_t batch() const noexcept {
        return batch_;
    }

    int64_t frames() const noexcept {
        return frames_;
    }

    MiDashengLmGenAudioTokenizerOutput run(const MiDashengLmGenAudioTokenizerInput & input) {
        if (input.batch != batch_ || input.frames != frames_ || input.dims != config_.audio_embedding_size) {
            throw std::runtime_error("MiDashengLM-Gen audio tokenizer latent input shape does not match decode graph");
        }
        if (static_cast<int64_t>(input.latents.size()) != batch_ * frames_ * config_.audio_embedding_size) {
            throw std::runtime_error("MiDashengLM-Gen audio tokenizer latent value count mismatch");
        }

        auto timing_start = Clock::now();
        ggml_backend_tensor_set(latents_, input.latents.data(), 0, input.latents.size() * sizeof(float));
        engine::debug::timing_log_scalar(
            "midashenglm_gen.audio_tokenizer.decode.input_upload_ms",
            engine::debug::elapsed_ms(timing_start, Clock::now()));

        core::set_backend_threads(execution_.backend(), execution_.config().threads);
        timing_start = Clock::now();
        const ggml_status status = core::compute_backend_graph(execution_.backend(), graph_);
        ggml_backend_synchronize(execution_.backend());
        engine::debug::timing_log_scalar(
            "midashenglm_gen.audio_tokenizer.decode.graph.compute_ms",
            engine::debug::elapsed_ms(timing_start, Clock::now()));
        if (status != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("MiDashengLM-Gen audio tokenizer decode graph compute failed");
        }

        std::vector<float> log_magnitude_phase;
        timing_start = Clock::now();
        core::read_tensor_float_into(output_, log_magnitude_phase);
        engine::debug::timing_log_scalar(
            "midashenglm_gen.audio_tokenizer.decode.output_read_ms",
            engine::debug::elapsed_ms(timing_start, Clock::now()));

        MiDashengLmGenAudioTokenizerOutput result;
        result.audio.reserve(static_cast<size_t>(batch_));
        const int64_t out_frames = 2 * frames_;
        const int64_t out_dim = config_.istft_n_fft + 2;
        if (execution_.backend_type() == core::BackendType::Cuda && cuda_istft_ == nullptr) {
            cuda_istft_ = std::make_unique<engine::audio::CudaLogMagnitudePhaseISTFT>(
                engine::audio::CudaLogMagnitudePhaseISTFTConfig{
                    out_frames,
                    config_.istft_n_fft,
                    config_.istft_hop,
                    out_dim,
                    execution_.config().device,
                });
        } else if (execution_.backend_type() != core::BackendType::Cuda && host_istft_ == nullptr) {
            host_istft_ = std::make_unique<engine::audio::HostLogMagnitudePhaseISTFT>(
                engine::audio::HostLogMagnitudePhaseISTFTConfig{
                    out_frames,
                    config_.istft_n_fft,
                    config_.istft_hop,
                    out_dim,
                    static_cast<size_t>(std::max(1, execution_.config().threads)),
                });
        }
        for (int64_t batch_index = 0; batch_index < batch_; ++batch_index) {
            const size_t item_offset = static_cast<size_t>(batch_index * out_frames * out_dim);
            const size_t item_count = static_cast<size_t>(out_frames * out_dim);
            const std::vector<float> item(
                log_magnitude_phase.begin() + static_cast<std::ptrdiff_t>(item_offset),
                log_magnitude_phase.begin() + static_cast<std::ptrdiff_t>(item_offset + item_count));
            if (execution_.backend_type() == core::BackendType::Cuda) {
                auto decoded = cuda_istft_->compute(item, weights_->istft_window);
                result.audio.push_back(engine::runtime::AudioBuffer{
                    static_cast<int>(config_.sample_rate),
                    1,
                    std::move(decoded.audio)});
            } else {
                auto decoded = host_istft_->compute(item, weights_->istft_window);
                result.audio.push_back(engine::runtime::AudioBuffer{
                    static_cast<int>(config_.sample_rate),
                    1,
                    std::move(decoded.audio)});
            }
        }
        return result;
    }

private:
    void clear_graph() {
        if (graph_ != nullptr) {
            core::release_backend_graph_resources(execution_.backend(), graph_);
            graph_ = nullptr;
        }
        if (gallocr_ != nullptr) {
            ggml_gallocr_free(gallocr_);
            gallocr_ = nullptr;
        }
        if (input_buffer_ != nullptr) {
            ggml_backend_buffer_free(input_buffer_);
            input_buffer_ = nullptr;
        }
    }

    core::ExecutionContext & execution_;
    std::shared_ptr<const MiDashengLmGenAudioTokenizerWeights> weights_;
    MiDashengLmGenConfig config_;
    int64_t batch_ = 0;
    int64_t frames_ = 0;
    std::unique_ptr<ggml_context, GgmlContextDeleter> input_ctx_;
    std::unique_ptr<ggml_context, GgmlContextDeleter> ctx_;
    ggml_tensor * latents_ = nullptr;
    ggml_tensor * output_ = nullptr;
    ggml_cgraph * graph_ = nullptr;
    ggml_gallocr_t gallocr_ = nullptr;
    ggml_backend_buffer_t input_buffer_ = nullptr;
    std::unique_ptr<engine::audio::CudaLogMagnitudePhaseISTFT> cuda_istft_;
    std::unique_ptr<engine::audio::HostLogMagnitudePhaseISTFT> host_istft_;
};

MiDashengLmGenAudioTokenizerRuntime::MiDashengLmGenAudioTokenizerRuntime(
    std::shared_ptr<const MiDashengLmGenAssets> assets,
    core::ExecutionContext & execution,
    size_t graph_arena_bytes,
    size_t weight_context_bytes,
    engine::assets::TensorStorageType matmul_storage_type,
    engine::assets::TensorStorageType conv_storage_type)
    : assets_(std::move(assets)),
      execution_(&execution),
      graph_arena_bytes_(graph_arena_bytes) {
    if (assets_ == nullptr) {
        throw std::runtime_error("MiDashengLM-Gen audio tokenizer runtime requires assets");
    }
    if (graph_arena_bytes_ == 0 || weight_context_bytes == 0) {
        throw std::runtime_error("MiDashengLM-Gen audio tokenizer context sizes must be non-zero");
    }
    weights_ = load_weights(
        *assets_,
        execution.backend(),
        execution.backend_type(),
        weight_context_bytes,
        matmul_storage_type,
        conv_storage_type);
}

MiDashengLmGenAudioTokenizerRuntime::~MiDashengLmGenAudioTokenizerRuntime() = default;

void MiDashengLmGenAudioTokenizerRuntime::prepare_decode(int64_t batch, int64_t frames) {
    if (execution_ == nullptr) {
        throw std::runtime_error("MiDashengLM-Gen audio tokenizer execution context is missing");
    }
    if (decode_graph_ != nullptr && decode_graph_->batch() == batch && decode_graph_->frames() == frames) {
        return;
    }
    decode_graph_.reset();
    decode_graph_ = std::make_unique<DecodeGraph>(
        *execution_,
        weights_,
        assets_->config,
        batch,
        frames,
        graph_arena_bytes_);
}

MiDashengLmGenAudioTokenizerOutput MiDashengLmGenAudioTokenizerRuntime::decode(
    const MiDashengLmGenAudioTokenizerInput & input) {
    if (decode_graph_ == nullptr || decode_graph_->batch() != input.batch || decode_graph_->frames() != input.frames) {
        throw std::runtime_error("MiDashengLM-Gen audio tokenizer decode graph was not prepared for this input shape");
    }
    return decode_graph_->run(input);
}

void MiDashengLmGenAudioTokenizerRuntime::release_graphs() {
    decode_graph_.reset();
}

}  // namespace engine::models::midashenglm_gen
