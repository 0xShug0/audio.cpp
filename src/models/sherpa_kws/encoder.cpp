#include "engine/models/sherpa_kws/encoder.h"

#include "engine/framework/core/backend.h"
#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/modules/conv_modules.h"
#include "engine/framework/modules/linear_module.h"
#include "engine/framework/modules/primitive_modules.h"
#include "engine/framework/modules/structural_modules.h"

#include <ggml-backend.h>
#include <ggml.h>

#include <array>
#include <chrono>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>

namespace engine::models::sherpa_kws {
namespace {

constexpr size_t kWeightContextBytes = 8ull * 1024ull * 1024ull;
constexpr size_t kGraphContextBytes = 16ull * 1024ull * 1024ull;
constexpr size_t kGraphNodes = 8192;
constexpr int64_t kFeatureDim = 80;
constexpr int64_t kEmbedCacheFrames = 3;
constexpr int64_t kEmbedChannels = 128;
constexpr int64_t kEmbedFrequency = 19;

struct GgmlContextDeleter {
    void operator()(ggml_context * context) const noexcept {
        if (context != nullptr) {
            ggml_free(context);
        }
    }
};

struct Param {
    core::TensorValue tensor;
};

const Param & require_param(
    const std::unordered_map<std::string, Param> & params,
    const std::string & name) {
    const auto it = params.find(name);
    if (it == params.end()) {
        throw std::runtime_error(
            "missing sherpa KWS encoder tensor: " + name);
    }
    return it->second;
}

core::TensorValue transpose(
    core::ModuleBuildContext & context,
    const core::TensorValue & input,
    const std::array<int, 4> & axes) {
    const size_t rank = input.shape.rank;
    core::TensorShape output_shape = {};
    output_shape.rank = rank;
    std::array<bool, 4> seen = {false, false, false, false};
    std::array<int, 4> ggml_axes = {0, 1, 2, 3};
    for (size_t output_axis = 0; output_axis < rank; ++output_axis) {
        const int input_axis = axes[output_axis];
        if (input_axis < 0 ||
            input_axis >= static_cast<int>(rank) ||
            seen[static_cast<size_t>(input_axis)]) {
            throw std::runtime_error(
                "sherpa KWS encoder transpose axes are invalid");
        }
        seen[static_cast<size_t>(input_axis)] = true;
        output_shape.dims[output_axis] =
            input.shape.dims[static_cast<size_t>(input_axis)];
        const int output_ggml_axis =
            core::logical_axis_to_ggml_axis(
                rank, static_cast<int>(output_axis));
        const int input_ggml_axis =
            core::logical_axis_to_ggml_axis(rank, input_axis);
        ggml_axes[static_cast<size_t>(input_ggml_axis)] =
            output_ggml_axis;
    }
    const auto contiguous =
        core::ensure_backend_addressable_layout(context, input);
    return core::wrap_tensor(
        ggml_permute(
            context.ggml,
            contiguous.tensor,
            ggml_axes[0],
            ggml_axes[1],
            ggml_axes[2],
            ggml_axes[3]),
        output_shape,
        input.type);
}

core::TensorValue scale_bias(
    core::ModuleBuildContext & context,
    const core::TensorValue & input,
    float scale,
    float bias) {
    return core::wrap_tensor(
        ggml_scale_bias(context.ggml, input.tensor, scale, bias),
        input.shape,
        GGML_TYPE_F32);
}

core::TensorValue add(
    core::ModuleBuildContext & context,
    const core::TensorValue & lhs,
    const core::TensorValue & rhs) {
    return core::wrap_tensor(
        ggml_add(context.ggml, lhs.tensor, rhs.tensor),
        lhs.shape,
        GGML_TYPE_F32);
}

core::TensorValue swoosh(
    core::ModuleBuildContext & context,
    const core::TensorValue & input,
    float offset,
    float constant) {
    auto shifted = scale_bias(context, input, 1.0F, -offset);
    auto output = core::wrap_tensor(
        ggml_softplus(context.ggml, shifted.tensor),
        input.shape,
        GGML_TYPE_F32);
    output = add(
        context,
        output,
        core::wrap_tensor(
            ggml_scale(context.ggml, input.tensor, -0.08F),
            input.shape,
            GGML_TYPE_F32));
    return scale_bias(context, output, 1.0F, constant);
}

core::TensorValue swoosh_r(
    core::ModuleBuildContext & context,
    const core::TensorValue & input) {
    return swoosh(
        context, input, 1.0F, -0.313261687F);
}

modules::Conv2dWeights conv_weights(
    const std::unordered_map<std::string, Param> & params,
    const std::string & prefix) {
    return {
        require_param(params, prefix + ".weight").tensor,
        require_param(params, prefix + ".bias").tensor,
    };
}

core::TensorValue bias_norm(
    core::ModuleBuildContext & context,
    const core::TensorValue & input,
    const std::unordered_map<std::string, Param> & params,
    float scale) {
    const auto & bias =
        require_param(params, "encoder.encoder_embed.out_norm.bias").tensor;
    auto bias_view = core::reshape_tensor(
        context,
        bias,
        core::TensorShape::from_dims({1, 1, input.shape.dims[2]}));
    auto bias_repeated = core::wrap_tensor(
        ggml_repeat(context.ggml, bias_view.tensor, input.tensor),
        input.shape,
        GGML_TYPE_F32);
    auto centered = core::wrap_tensor(
        ggml_sub(context.ggml, input.tensor, bias_repeated.tensor),
        input.shape,
        GGML_TYPE_F32);
    auto squared = core::wrap_tensor(
        ggml_sqr(context.ggml, centered.tensor),
        input.shape,
        GGML_TYPE_F32);
    auto mean = modules::ReduceMeanModule({2}).build(
        context, squared);
    auto denominator = core::wrap_tensor(
        ggml_sqrt(context.ggml, mean.tensor),
        mean.shape,
        GGML_TYPE_F32);
    auto denominator_repeated = core::wrap_tensor(
        ggml_repeat(
            context.ggml, denominator.tensor, input.tensor),
        input.shape,
        GGML_TYPE_F32);
    auto normalized = core::wrap_tensor(
        ggml_div(
            context.ggml,
            input.tensor,
            denominator_repeated.tensor),
        input.shape,
        GGML_TYPE_F32);
    return core::wrap_tensor(
        ggml_scale(context.ggml, normalized.tensor, scale),
        normalized.shape,
        GGML_TYPE_F32);
}

class SubsamplingGraph {
public:
    SubsamplingGraph(
        ggml_backend_t backend,
        core::BackendType backend_type,
        const std::unordered_map<std::string, Param> & params,
        float norm_scale,
        int64_t input_frames,
        int64_t subsampled_frames,
        int64_t subsampled_dim)
        : backend_(backend),
          input_frames_(input_frames),
          subsampled_frames_(subsampled_frames),
          subsampled_dim_(subsampled_dim) {
        if (input_frames_ <= 0 || subsampled_frames_ <= kEmbedCacheFrames) {
            throw std::runtime_error(
                "Sherpa subsampling graph dimensions are invalid");
        }
        ggml_init_params init{
            kGraphContextBytes, nullptr, true};
        context_.reset(ggml_init(init));
        if (context_ == nullptr) {
            throw std::runtime_error(
                "failed to initialize Sherpa subsampling graph context");
        }
        core::ModuleBuildContext build{
            context_.get(), "sherpa_kws.subsampling", backend_type};
        auto input = core::make_tensor(
            build,
            GGML_TYPE_F32,
            core::TensorShape::from_dims(
                {1, 1, input_frames_, kFeatureDim}));
        input_ = input.tensor;
        ggml_set_input(input_);

        auto output = modules::Conv2dModule(
            {1, 8, 3, 3, 1, 1, 0, 1, 1, 1, true})
                          .build(
                              build,
                              input,
                              conv_weights(
                                  params,
                                  "encoder.encoder_embed.conv.0"));
        output = swoosh_r(build, output);
        output = modules::Conv2dModule(
            {8, 32, 3, 3, 2, 2, 0, 0, 1, 1, true})
                     .build(
                         build,
                         output,
                         conv_weights(
                             params,
                             "encoder.encoder_embed.conv.4"));
        output = swoosh_r(build, output);
        output = modules::Conv2dModule(
            {32, 128, 3, 3, 1, 2, 0, 0, 1, 1, true})
                     .build(
                         build,
                         output,
                         conv_weights(
                             params,
                             "encoder.encoder_embed.conv.7"));
        output = swoosh_r(build, output);

        const auto bypass =
            modules::SliceModule({2, 0, subsampled_frames_})
                .build(build, output);
        auto cache = core::make_tensor(
            build,
            GGML_TYPE_F32,
            core::TensorShape::from_dims(
                {1,
                 kEmbedChannels,
                 kEmbedCacheFrames,
                 kEmbedFrequency}));
        cache_ = cache.tensor;
        ggml_set_input(cache_);
        ggml_set_output(cache_);
        auto new_cache =
            core::ensure_backend_addressable_layout(
                build,
                modules::SliceModule(
                    {2,
                     subsampled_frames_ - kEmbedCacheFrames,
                     kEmbedCacheFrames})
                    .build(build, output));
        new_cache_ = new_cache.tensor;
        auto padded = modules::ConcatModule({2}).build(
            build, cache, output);
        output = modules::DepthwiseConv2dModule(
            {128, 7, 7, 1, 1, 0, 3, 1, 1, true})
                     .build(
                         build,
                         padded,
                         conv_weights(
                             params,
                             "encoder.encoder_embed.convnext.depthwise_conv"));
        // Pointwise convolutions are channel projections; retain channel-last
        // layout across both projections instead of materializing im2col twice.
        output = modules::TransposeModule({{0, 2, 3, 1}, 4}).build(build, output);
        const auto pointwise1 = conv_weights(params, "encoder.encoder_embed.convnext.pointwise_conv1");
        output = modules::LinearModule({128, 384, true}).build(build, output, {
            core::reshape_tensor(build, pointwise1.weight, core::TensorShape::from_dims({384, 128})),
            pointwise1.bias});
        output = swoosh(
            build, output, 4.0F, -0.035F);
        const auto pointwise2 = conv_weights(params, "encoder.encoder_embed.convnext.pointwise_conv2");
        output = modules::LinearModule({384, 128, true}).build(build, output, {
            core::reshape_tensor(build, pointwise2.weight, core::TensorShape::from_dims({128, 384})),
            pointwise2.bias});
        output = modules::TransposeModule({{0, 3, 1, 2}, 4}).build(build, output);
        output = add(build, bypass, output);

        output = transpose(build, output, {0, 2, 1, 3});
        output = core::reshape_tensor(
            build,
            core::ensure_backend_addressable_layout(build, output),
            core::TensorShape::from_dims(
                {1, subsampled_frames_, 128 * 19}));
        output = modules::LinearModule(
            {128 * 19, subsampled_dim_, true})
                     .build(
                         build,
                         output,
                         {
                             require_param(
                                 params,
                                 "encoder.encoder_embed.out.weight")
                                 .tensor,
                             require_param(
                                 params,
                                 "encoder.encoder_embed.out.bias")
                                 .tensor,
                         });
        output = bias_norm(
            build, output, params, norm_scale);
        output_ = output.tensor;
        ggml_set_output(output_);
        ggml_set_output(new_cache_);

        graph_ = ggml_new_graph_custom(
            context_.get(), kGraphNodes, false);
        ggml_build_forward_expand(graph_, output_);
        ggml_build_forward_expand(graph_, new_cache_);
        ggml_build_forward_expand(graph_, ggml_cpy(build.ggml, new_cache_, cache_));
        allocator_ = ggml_gallocr_new(
            ggml_backend_get_default_buffer_type(backend_));
        if (allocator_ == nullptr ||
            !ggml_gallocr_reserve(allocator_, graph_) ||
            !ggml_gallocr_alloc_graph(allocator_, graph_)) {
            throw std::runtime_error(
                "failed to allocate Sherpa subsampling graph");
        }
        plan_ =
            core::create_backend_graph_plan_if_host(backend_, graph_);
        reset();
    }

    ~SubsamplingGraph() {
        core::release_backend_graph_resources(backend_, graph_);
        if (plan_ != nullptr) {
            core::free_backend_graph_plan(backend_, plan_);
        }
        if (allocator_ != nullptr) {
            ggml_gallocr_free(allocator_);
        }
    }

    SherpaSubsampledChunk run(
        const std::vector<float> & features) const {
        if (features.size() !=
            static_cast<size_t>(input_frames_ * kFeatureDim)) {
            throw std::runtime_error(
                "Sherpa subsampling feature shape does not match the package chunk size");
        }
        ggml_backend_tensor_set_async(
            backend_,
            input_,
            features.data(),
            0,
            features.size() * sizeof(float));
        ggml_backend_synchronize(backend_);
        const auto status = core::compute_backend_graph(
            backend_,
            graph_,
            plan_,
            "Sherpa KWS subsampling");
        ggml_backend_synchronize(backend_);
        if (status != GGML_STATUS_SUCCESS) {
            throw std::runtime_error(
                "Sherpa subsampling graph compute failed");
        }
        SherpaSubsampledChunk result;
        result.frames = subsampled_frames_;
        result.channels = subsampled_dim_;
        result.values.resize(
            static_cast<size_t>(
                subsampled_frames_ * subsampled_dim_));
        ggml_backend_tensor_get_async(
            backend_,
            output_,
            result.values.data(),
            0,
            result.values.size() * sizeof(float));
        ggml_backend_synchronize(backend_);
        return result;
    }

    void reset() {
        ggml_backend_tensor_memset(cache_, 0, 0, ggml_nbytes(cache_));
    }

private:
    ggml_backend_t backend_ = nullptr;
    std::unique_ptr<ggml_context, GgmlContextDeleter> context_;
    ggml_tensor * input_ = nullptr;
    ggml_tensor * cache_ = nullptr;
    ggml_tensor * new_cache_ = nullptr;
    ggml_tensor * output_ = nullptr;
    ggml_cgraph * graph_ = nullptr;
    ggml_gallocr_t allocator_ = nullptr;
    ggml_backend_graph_plan_t plan_ = nullptr;
    int64_t input_frames_ = 0;
    int64_t subsampled_frames_ = 0;
    int64_t subsampled_dim_ = 0;
};

}  // namespace

struct SherpaEncoderRuntime::Impl {
    std::shared_ptr<const SherpaKwsAssets> assets;
    std::shared_ptr<core::BackendWeightStore> weights;
    std::unordered_map<std::string, Param> params;
    std::unique_ptr<SubsamplingGraph> graph;
};

SherpaEncoderRuntime::SherpaEncoderRuntime(
    std::shared_ptr<const SherpaKwsAssets> assets,
    core::ExecutionContext & execution_context)
    : impl_(std::make_unique<Impl>()) {
    if (assets == nullptr || assets->weights == nullptr) {
        throw std::runtime_error(
            "sherpa KWS encoder requires tensor assets");
    }
    impl_->assets = std::move(assets);
    impl_->weights =
        std::make_shared<core::BackendWeightStore>(
            execution_context.backend(),
            execution_context.backend_type(),
            "Sherpa KWS subsampling weights",
            kWeightContextBytes);
    const auto & source = *impl_->assets->weights;
    const int64_t subsampled_dim = impl_->assets->config.encoder_dims.front();
    const auto load = [&](const std::string & name,
                          const std::vector<int64_t> & shape) {
        impl_->params.emplace(
            name,
            Param{impl_->weights->load_f32_tensor(
                source, name, shape)});
    };
    load("encoder.encoder_embed.conv.0.weight", {8, 1, 3, 3});
    load("encoder.encoder_embed.conv.0.bias", {8});
    load("encoder.encoder_embed.conv.4.weight", {32, 8, 3, 3});
    load("encoder.encoder_embed.conv.4.bias", {32});
    load("encoder.encoder_embed.conv.7.weight", {128, 32, 3, 3});
    load("encoder.encoder_embed.conv.7.bias", {128});
    load(
        "encoder.encoder_embed.convnext.depthwise_conv.weight",
        {128, 1, 7, 7});
    load(
        "encoder.encoder_embed.convnext.depthwise_conv.bias",
        {128});
    load(
        "encoder.encoder_embed.convnext.pointwise_conv1.weight",
        {384, 128, 1, 1});
    load(
        "encoder.encoder_embed.convnext.pointwise_conv1.bias",
        {384});
    load(
        "encoder.encoder_embed.convnext.pointwise_conv2.weight",
        {128, 384, 1, 1});
    load(
        "encoder.encoder_embed.convnext.pointwise_conv2.bias",
        {128});
    load(
        "encoder.encoder_embed.out.weight",
        {subsampled_dim, 128 * 19});
    load(
        "encoder.encoder_embed.out.bias",
        {subsampled_dim});
    load(
        "encoder.encoder_embed.out_norm.bias",
        {subsampled_dim});
    impl_->weights->upload();
    const auto log_scale = source.require_f32(
        "encoder.encoder_embed.out_norm.log_scale", {1});
    impl_->graph = std::make_unique<SubsamplingGraph>(
        execution_context.backend(),
        execution_context.backend_type(),
        impl_->params,
        std::exp(log_scale.front()),
        impl_->assets->config.chunk_size,
        impl_->assets->config.chunk_shift / 2,
        subsampled_dim);
}

SherpaEncoderRuntime::~SherpaEncoderRuntime() = default;

SherpaSubsampledChunk
SherpaEncoderRuntime::encode_subsampled_chunk(
    const std::vector<float> & features) const {
    if (impl_ == nullptr || impl_->graph == nullptr) {
        throw std::runtime_error(
            "sherpa KWS encoder is not initialized");
    }
    const auto start = std::chrono::steady_clock::now();
    auto result = impl_->graph->run(features);
    engine::debug::timing_log_scalar(
        impl_->assets->config.family + ".subsampling_ms",
        engine::debug::elapsed_ms(start));
    return result;
}

void SherpaEncoderRuntime::reset() {
    if (impl_ == nullptr || impl_->graph == nullptr) {
        throw std::runtime_error(
            "sherpa KWS encoder is not initialized");
    }
    impl_->graph->reset();
}

}  // namespace engine::models::sherpa_kws
