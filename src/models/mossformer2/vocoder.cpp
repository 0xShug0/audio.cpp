#include "engine/models/mossformer2/vocoder.h"

#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/modules/activation_modules.h"
#include "engine/framework/modules/conv_modules.h"
#include "engine/framework/modules/primitive_modules.h"
#include "engine/framework/modules/structural_modules.h"

#include <ggml-alloc.h>

#include <stdexcept>
#include <unordered_map>

namespace engine::models::mossformer2 {
namespace {
using core::TensorShape;
using core::TensorValue;
using TensorMap = std::unordered_map<std::string, TensorValue>;

class VocoderGraph {
public:
    VocoderGraph(core::ExecutionContext & execution, const MossFormer2Config & config,
                 const TensorMap & weights, int64_t frames)
        : execution_(execution), weights_(weights) {
        context_.reset(ggml_init({4 * 1024 * 1024, nullptr, true}));
        if (!context_) throw std::runtime_error("MossFormer2 vocoder metadata allocation failed");
        ctx_ = {context_.get(), "mossformer2.vocoder", execution.backend_type()};
        auto x = core::make_tensor(ctx_, GGML_TYPE_F32, TensorShape::from_dims({1, config.output_features, frames}));
        input_ = x.tensor;
        ggml_set_input(input_);
        const auto & c = config.vocoder;
        x = convolution(x, "vocoder.conv_pre", c.initial_channels, 7, 1, 3);
        int64_t channels = c.initial_channels;
        for (size_t i = 0; i < c.upsample_rates.size(); ++i) {
            x = snake(x, "vocoder.snakes." + std::to_string(i));
            const auto base = "vocoder.ups." + std::to_string(i);
            const int64_t padding = (c.upsample_kernels[i] - c.upsample_rates[i]) / 2;
            const bool crop = execution.backend_type() != core::BackendType::Cuda;
            x = modules::ConvTranspose1dModule({channels, channels / 2, c.upsample_kernels[i],
                static_cast<int>(c.upsample_rates[i]), static_cast<int>(crop ? 0 : padding), 1, true})
                .build(ctx_, x, {weights_.at(base + ".weight"), weights_.at(base + ".bias")});
            if (crop && padding > 0)
                x = modules::SliceModule({2, padding, x.shape.dims[2] - 2 * padding}).build(ctx_, x);
            channels /= 2;
            TensorValue sum;
            for (size_t j = 0; j < c.resblock_kernels.size(); ++j) {
                auto residual = x;
                const auto prefix = "vocoder.resblocks." + std::to_string(i * c.resblock_kernels.size() + j);
                for (size_t k = 0; k < c.resblock_dilations[j].size(); ++k) {
                    const auto suffix = std::to_string(k);
                    auto y = snake(residual, prefix + ".convs1_activates." + suffix);
                    const auto kernel = c.resblock_kernels[j], dilation = c.resblock_dilations[j][k];
                    y = convolution(y, prefix + ".convs1." + suffix, channels, kernel, dilation, (kernel - 1) * dilation / 2);
                    y = snake(y, prefix + ".convs2_activates." + suffix);
                    y = convolution(y, prefix + ".convs2." + suffix, channels, kernel, 1, (kernel - 1) / 2);
                    residual = modules::AddModule().build(ctx_, residual, y);
                }
                sum = sum.valid() ? modules::AddModule().build(ctx_, sum, residual) : residual;
            }
            x = modules::ScaleBiasModule({1.0F / static_cast<float>(c.resblock_kernels.size()), 0}).build(ctx_, sum);
        }
        x = snake(x, "vocoder.snake_post");
        x = convolution(x, "vocoder.conv_post", 1, 7, 1, 3);
        output_ = modules::TanhModule().build(ctx_, x).tensor;
        ggml_set_output(output_);
        graph_ = ggml_new_graph_custom(context_.get(), 8192, false);
        ggml_build_forward_expand(graph_, output_);
        allocator_.reset(ggml_gallocr_new(ggml_backend_get_default_buffer_type(execution.backend())));
        if (!ggml_gallocr_alloc_graph(allocator_.get(), graph_)) throw std::runtime_error("MossFormer2 vocoder graph allocation failed");
        core::validate_backend_graph_supported(execution.backend(), graph_, "MossFormer2 vocoder");
        core::prepare_host_graph_plan(execution, graph_, host_plan_);
        debug::timing_log_context_reservation("mossformer2.vocoder", context_.get());
        debug::timing_log_scalar("mossformer2.vocoder.graph.buffer_mb", ggml_gallocr_get_buffer_size(allocator_.get(), 0) / (1024.0 * 1024.0));
    }

    ~VocoderGraph() { core::release_backend_graph_resources(execution_.backend(), graph_, true); }

    std::vector<float> run(const std::vector<float> & features) {
        ggml_backend_tensor_set(input_, features.data(), 0, features.size() * sizeof(float));
        if (core::compute_backend_graph(execution_.backend(), graph_, host_plan_.plan) != GGML_STATUS_SUCCESS)
            throw std::runtime_error("MossFormer2 vocoder execution failed");
        return core::read_tensor_f32(output_);
    }

private:
    TensorValue snake(const TensorValue & x, const std::string & prefix) {
        const auto alpha = core::reshape_tensor(ctx_, weights_.at(prefix + ".alpha"),
            TensorShape::from_dims({x.shape.dims[1]}));
        return modules::SnakeBeta1dModule({x.shape.dims[1], false}).build(ctx_, x, {alpha, alpha});
    }

    TensorValue convolution(const TensorValue & x, const std::string & prefix,
                            int64_t channels, int64_t kernel, int64_t dilation, int64_t padding) {
        return modules::Conv1dModule({x.shape.dims[1], channels, kernel, 1, static_cast<int>(padding), static_cast<int>(dilation), true})
            .build(ctx_, x, {weights_.at(prefix + ".weight"), weights_.at(prefix + ".bias")});
    }

    core::ExecutionContext & execution_;
    const TensorMap & weights_;
    std::unique_ptr<ggml_context, decltype(&ggml_free)> context_{nullptr, ggml_free};
    std::unique_ptr<ggml_gallocr, decltype(&ggml_gallocr_free)> allocator_{nullptr, ggml_gallocr_free};
    core::ModuleBuildContext ctx_;
    core::HostGraphPlan host_plan_;
    ggml_cgraph * graph_ = nullptr;
    ggml_tensor * input_ = nullptr;
    ggml_tensor * output_ = nullptr;
};
}  // namespace

class MossFormer2VocoderRuntime::Impl {
public:
    Impl(std::shared_ptr<const MossFormer2Assets> assets, core::ExecutionContext & execution)
        : assets(std::move(assets)), execution(execution),
          store(execution.backend(), execution.backend_type(), "mossformer2.vocoder.weights", 1024 * 1024) {
        for (const auto & metadata : this->assets->tensors->tensors()) {
            if (metadata.name.rfind("vocoder.", 0) != 0) continue;
            weights.emplace(metadata.name, store.load_tensor(*this->assets->tensors, metadata.name,
                engine::assets::TensorStorageType::Native, metadata.shape));
        }
        store.upload();

    }

    std::shared_ptr<const MossFormer2Assets> assets;
    core::ExecutionContext & execution;
    core::BackendWeightStore store;
    TensorMap weights;
    std::unique_ptr<VocoderGraph> graph;
    int64_t frames = 0;
};

MossFormer2VocoderRuntime::MossFormer2VocoderRuntime(std::shared_ptr<const MossFormer2Assets> assets,
    core::ExecutionContext & execution) : impl_(std::make_unique<Impl>(std::move(assets), execution)) {}
MossFormer2VocoderRuntime::~MossFormer2VocoderRuntime() = default;

std::vector<float> MossFormer2VocoderRuntime::synthesize(const std::vector<float> & features, int64_t frames) {
    auto & p = *impl_;
    if (frames <= 0 || features.size() != static_cast<size_t>(frames) * p.assets->config.output_features)
        throw std::runtime_error("MossFormer2 vocoder feature shape mismatch");
    if (!p.graph || p.frames != frames) {
        p.graph.reset();
        p.graph = std::make_unique<VocoderGraph>(p.execution, p.assets->config, p.weights, frames);
        p.frames = frames;
    }
    const auto start = std::chrono::steady_clock::now();
    auto result = p.graph->run(features);
    debug::timing_log_scalar("mossformer2.vocoder.compute_ms", debug::elapsed_ms(start));
    return result;
}

}  // namespace engine::models::mossformer2
