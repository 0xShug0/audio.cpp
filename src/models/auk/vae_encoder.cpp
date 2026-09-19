#include "engine/models/auk/vae_encoder.h"

#include "engine/framework/core/backend.h"
#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/core/module.h"
#include "engine/framework/modules/conv_modules.h"
#include "engine/framework/modules/weight_binding.h"

#include <ggml.h>

#include <array>
#include <mutex>
#include <numeric>
#include <stdexcept>

namespace engine::models::auk {
namespace {

using engine::modules::Conv1dConfig;
using engine::modules::Conv1dModule;
using engine::modules::Conv1dWeights;

// ⚠ TWO SLOPES, and they are not the same. The encoder body uses LeakyReLU(0.2);
// the ones inside ResStack are a bare nn.LeakyReLU(), i.e. PyTorch's default 0.01.
// Using one throughout is a silent quality bug, not a crash. dots_tts names both
// constants for the same reason.
constexpr float kEncoderSlope = 0.2F;
constexpr float kResStackSlope = 0.01F;

constexpr size_t kGraphContextBytes = 64ull * 1024ull * 1024ull;
constexpr int64_t kGraphNodeCapacity = 262144;

// Conv1d_S: padding = dilation * (kernel - 1) / 2, integer division. For the even
// kernels the downsampling convolutions use (k = 2 * rate) this is NOT symmetric --
// (2f-1)/2 truncates to f-1 -- and that asymmetry is part of the model.
int conv1d_s_padding(int64_t kernel, int64_t dilation) {
    return static_cast<int>(dilation * (kernel - 1) / 2);
}

struct ResStackBlock {
    Conv1dWeights dilated;
    Conv1dWeights plain;
    int64_t dilation = 1;
};

struct EncoderStage {
    Conv1dWeights down;
    int64_t rate = 1;
    int64_t in_channels = 0;
    int64_t out_channels = 0;
    std::vector<ResStackBlock> blocks;
};

struct EncoderWeights {
    Conv1dWeights pre;
    std::vector<EncoderStage> stages;
    Conv1dWeights post;
    int64_t loaded = 0;
};

core::TensorValue leaky_relu(core::ModuleBuildContext & ctx, const core::TensorValue & x, float slope) {
    return core::wrap_tensor(
        ggml_leaky_relu(ctx.ggml, core::ensure_backend_addressable_layout(ctx, x).tensor, slope, false),
        x.shape,
        GGML_TYPE_F32);
}

core::TensorValue conv(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & x,
    const Conv1dWeights & weights,
    int64_t in_channels,
    int64_t out_channels,
    int64_t kernel,
    int64_t stride,
    int64_t dilation) {
    return Conv1dModule({
        in_channels,
        out_channels,
        kernel,
        static_cast<int>(stride),
        conv1d_s_padding(kernel, dilation),
        static_cast<int>(dilation),
        true,
    }).build(ctx, x, weights);
}

core::TensorValue build_encoder(
    core::ModuleBuildContext & ctx,
    const AukVaeEncoderConfig & config,
    const EncoderWeights & weights,
    const core::TensorValue & waveform) {
    auto x = conv(
        ctx, waveform, weights.pre, config.in_channels, config.downsample_channels.front(),
        config.proj_kernel_size, 1, 1);
    x = leaky_relu(ctx, x, kEncoderSlope);

    for (const auto & stage : weights.stages) {
        x = conv(ctx, x, stage.down, stage.in_channels, stage.out_channels, stage.rate * 2, stage.rate, 1);
        for (const auto & block : stage.blocks) {
            // Residual: x + conv(lrelu(conv(lrelu(x)))). The dilation is on the FIRST
            // convolution of each block only; the second is always dilation 1.
            auto inner = leaky_relu(ctx, x, kResStackSlope);
            inner = conv(
                ctx, inner, block.dilated, stage.out_channels, stage.out_channels,
                config.stack_kernel_size, 1, block.dilation);
            inner = leaky_relu(ctx, inner, kResStackSlope);
            inner = conv(
                ctx, inner, block.plain, stage.out_channels, stage.out_channels,
                config.stack_kernel_size, 1, 1);
            x = core::wrap_tensor(
                ggml_add(
                    ctx.ggml,
                    core::ensure_backend_addressable_layout(ctx, x).tensor,
                    core::ensure_backend_addressable_layout(ctx, inner).tensor),
                x.shape,
                GGML_TYPE_F32);
        }
        x = leaky_relu(ctx, x, kEncoderSlope);
    }

    return conv(
        ctx, x, weights.post, config.downsample_channels.back(), config.output_channels(),
        config.proj_kernel_size, 1, 1);
}

EncoderWeights load_weights(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    const AukVaeEncoderConfig & config,
    const std::string & prefix) {
    EncoderWeights out;
    const auto storage = assets::TensorStorageType::Native;
    // nn.Sequential indices: 0 pre-conv, 1 LeakyReLU, then (conv, ResStack, LeakyReLU)
    // per stage, then the output projection. The ReLUs hold no weights but still take
    // an index, which is why the convolutions land on 0, 2, 5, 8, ... and the stacks on
    // 3, 6, 9, ...
    const std::string generator = prefix + ".generator.";
    out.pre = engine::modules::binding::conv1d_from_source_resolving_weight_norm(
        store, source, generator + "0.layer", storage,
        config.downsample_channels.front(), config.in_channels, config.proj_kernel_size, true);
    out.loaded += 2;

    int64_t index = 2;
    for (size_t stage_index = 0; stage_index < config.downsample_rates.size(); ++stage_index) {
        EncoderStage stage;
        stage.rate = config.downsample_rates[stage_index];
        stage.in_channels = config.downsample_channels[stage_index];
        stage.out_channels = config.downsample_channels[stage_index + 1];
        stage.down = engine::modules::binding::conv1d_from_source_resolving_weight_norm(
            store, source, generator + std::to_string(index) + ".layer", storage,
            stage.out_channels, stage.in_channels, stage.rate * 2, true);
        out.loaded += 2;

        const std::string stack = generator + std::to_string(index + 1) + ".layers.";
        int64_t dilation = 1;
        for (int64_t block_index = 0; block_index < config.stacks; ++block_index) {
            ResStackBlock block;
            block.dilation = dilation;
            const std::string block_prefix = stack + std::to_string(block_index) + ".";
            block.dilated = engine::modules::binding::conv1d_from_source_resolving_weight_norm(
                store, source, block_prefix + "1", storage,
                stage.out_channels, stage.out_channels, config.stack_kernel_size, true);
            block.plain = engine::modules::binding::conv1d_from_source_resolving_weight_norm(
                store, source, block_prefix + "3", storage,
                stage.out_channels, stage.out_channels, config.stack_kernel_size, true);
            out.loaded += 4;
            stage.blocks.push_back(std::move(block));
            dilation *= config.stack_dilation_base;
        }
        out.stages.push_back(std::move(stage));
        index += 3;
    }

    out.post = engine::modules::binding::conv1d_from_source_resolving_weight_norm(
        store, source, generator + std::to_string(index) + ".layer", storage,
        config.output_channels(), config.downsample_channels.back(), config.proj_kernel_size, true);
    out.loaded += 2;
    return out;
}

}  // namespace

int64_t AukVaeEncoderConfig::hop_size() const {
    return std::accumulate(
        downsample_rates.begin(), downsample_rates.end(), int64_t{1}, std::multiplies<int64_t>());
}

void AukVaeEncoderConfig::validate() const {
    if (latent_dim <= 0 || in_channels <= 0 || stacks <= 0) {
        throw std::runtime_error("AuK VAE encoder config has non-positive dimensions");
    }
    if (downsample_rates.empty() || downsample_channels.size() != downsample_rates.size() + 1) {
        throw std::runtime_error("AuK VAE encoder needs one more channel count than rates");
    }
}

struct AukVaeEncoder::Impl {
    AukVaeEncoderConfig config;
    core::ExecutionContext * execution = nullptr;
    std::unique_ptr<core::BackendWeightStore> store;
    EncoderWeights weights;

    std::mutex mutex;
    ggml_context * ggml = nullptr;
    ggml_gallocr_t gallocr = nullptr;
    ggml_cgraph * graph = nullptr;
    core::HostGraphPlan plan;
    ggml_tensor * input = nullptr;
    ggml_tensor * output = nullptr;
    int64_t samples = 0;

    void release() {
        if (graph != nullptr) {
            core::release_backend_graph_resources(execution->backend(), graph);
        }
        if (gallocr != nullptr) {
            ggml_gallocr_free(gallocr);
            gallocr = nullptr;
        }
        if (ggml != nullptr) {
            ggml_free(ggml);
            ggml = nullptr;
        }
        plan.reset();
        graph = nullptr;
        input = nullptr;
        output = nullptr;
        samples = 0;
    }

    void ensure_graph(int64_t wanted) {
        if (ggml != nullptr && samples == wanted) {
            return;
        }
        release();
        ggml_init_params params{kGraphContextBytes, nullptr, true};
        ggml = ggml_init(params);
        if (ggml == nullptr) {
            throw std::runtime_error("failed to initialize AuK VAE encoder graph context");
        }
        core::ModuleBuildContext ctx{ggml, "auk.vae.encoder", execution->backend_type()};
        auto waveform = core::make_tensor(ctx, GGML_TYPE_F32, core::TensorShape::from_dims({1, 1, wanted}));
        input = waveform.tensor;
        // ⚠ make_tensor does not flag graph endpoints, and the allocator is free to
        // reuse anything unflagged -- so without this the waveform written before
        // compute is clobbered and the encoder returns all zeros.
        ggml_set_input(input);
        auto latents = build_encoder(ctx, config, weights, waveform);
        output = core::ensure_backend_addressable_layout(ctx, latents).tensor;
        ggml_set_output(output);
        graph = ggml_new_graph_custom(ggml, kGraphNodeCapacity, false);
        ggml_build_forward_expand(graph, output);
        core::validate_backend_graph_supported(execution->backend(), graph, "auk.vae.encoder");
        gallocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(execution->backend()));
        if (gallocr == nullptr || !ggml_gallocr_reserve(gallocr, graph) || !ggml_gallocr_alloc_graph(gallocr, graph)) {
            release();
            throw std::runtime_error("failed to allocate AuK VAE encoder graph memory");
        }
        core::prepare_host_graph_plan(*execution, graph, plan);
        samples = wanted;
    }
};

AukVaeEncoder::AukVaeEncoder(
    AukVaeEncoderConfig config,
    const assets::TensorSource & source,
    core::ExecutionContext & execution,
    std::string prefix)
    : impl_(std::make_unique<Impl>()) {
    config.validate();
    impl_->config = std::move(config);
    impl_->execution = &execution;
    impl_->store = std::make_unique<core::BackendWeightStore>(
        execution.backend(), execution.backend_type(), "auk.vae.encoder.weights",
        512ull * 1024ull * 1024ull);
    impl_->weights = load_weights(*impl_->store, source, impl_->config, prefix);
    // ⚠ The store BATCHES: load_tensor queues an upload and hands back a tensor with no
    // buffer yet. Without this the graph runs on unbacked weights and the encoder
    // returns all zeros -- no error, because nothing was ever wrong with the graph.
    impl_->store->upload();
}

AukVaeEncoder::~AukVaeEncoder() {
    if (impl_ != nullptr) {
        impl_->release();
    }
}

AukLatentDistribution AukVaeEncoder::encode(const std::vector<float> & waveform) {
    if (waveform.empty()) {
        throw std::runtime_error("AuK VAE encoder received an empty waveform");
    }
    std::lock_guard<std::mutex> lock(impl_->mutex);
    impl_->ensure_graph(static_cast<int64_t>(waveform.size()));
    ggml_backend_tensor_set(impl_->input, waveform.data(), 0, waveform.size() * sizeof(float));
    if (core::compute_graph(*impl_->execution, impl_->graph, impl_->plan, "auk.vae.encoder") != GGML_STATUS_SUCCESS) {
        throw std::runtime_error("AuK VAE encoder graph compute failed");
    }
    AukLatentDistribution out;
    out.frames = impl_->output->ne[0];
    out.channels = impl_->output->ne[1];
    out.values = core::read_tensor_f32(impl_->output);
    return out;
}

const AukVaeEncoderConfig & AukVaeEncoder::config() const noexcept { return impl_->config; }

int64_t AukVaeEncoder::loaded_tensor_count() const noexcept { return impl_->weights.loaded; }

}  // namespace engine::models::auk
