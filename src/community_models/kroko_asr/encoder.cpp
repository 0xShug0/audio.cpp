#include "engine/community_models/kroko_asr/encoder.h"

#include "engine/framework/core/backend.h"
#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/core/prepared_graph.h"
#include "engine/framework/modules/activation_modules.h"
#include "engine/framework/modules/norm_modules.h"
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

namespace engine::models::kroko_asr {
namespace {

constexpr size_t kWeightContextBytes = 8ull * 1024ull * 1024ull;
constexpr size_t kGraphContextBytes = 16ull * 1024ull * 1024ull;
constexpr size_t kGraphNodes = 8192;
constexpr int64_t kFeatureDim = 80;
constexpr int64_t kSubsampledDim = 192;
constexpr int64_t kEmbedCacheFrames = 3;
constexpr int64_t kEmbedChannels = 128;
constexpr int64_t kEmbedFrequency = 19;

struct Param {
    core::TensorValue tensor;
};

const Param & require_param(
    const std::unordered_map<std::string, Param> & params,
    const std::string & name) {
    const auto it = params.find(name);
    if (it == params.end()) {
        throw std::runtime_error(
            "missing Kroko encoder tensor: " + name);
    }
    return it->second;
}

modules::Conv2dWeights conv_weights(
    const std::unordered_map<std::string, Param> & params,
    const std::string & prefix) {
    return {
        require_param(params, prefix + ".weight").tensor,
        require_param(params, prefix + ".bias").tensor,
    };
}

class SubsamplingGraph {
public:
    SubsamplingGraph(
        core::ExecutionContext & execution,
        const std::unordered_map<std::string, Param> & params,
        float norm_scale,
        int64_t input_frames,
        int64_t subsampled_frames, bool streaming = true)
        : backend_(execution.backend()),
          streaming_(streaming),
          input_frames_(input_frames),
          subsampled_frames_(subsampled_frames),
          prepared_(execution, {"kroko_asr.subsampling", kGraphContextBytes, kGraphNodes}) {
        if (input_frames_ <= 0 || subsampled_frames_ <= 0 ||
            (streaming && subsampled_frames_ <= kEmbedCacheFrames)) {
            throw std::runtime_error(
                "Kroko subsampling graph dimensions are invalid");
        }
        auto & build = prepared_.build_context();
        auto input = core::make_tensor(
            build,
            GGML_TYPE_F32,
            core::TensorShape::from_dims(
                {1, input_frames_, kFeatureDim, 1}));
        input_ = input.tensor;
        ggml_set_input(input_);

        modules::Conv2dConfig conv0{1, 8, 3, 3, 1, 1, 0, 1, 1, 1, true};
        conv0.impl = modules::Conv2dImpl::ChannelFast;
        auto output = modules::Conv2dModule(conv0)
                          .build(
                              build,
                              input,
                              conv_weights(
                                  params,
                                  "encoder.encoder_embed.conv.0"), modules::Conv2dLayout::ChannelsLast);
        output = modules::SwooshRModule().build(build, output);
        modules::Conv2dConfig conv4{8, 32, 3, 3, 2, 2, 0, 0, 1, 1, true};
        conv4.impl = modules::Conv2dImpl::ChannelFast;
        output = modules::Conv2dModule(conv4)
                     .build(
                         build,
                         output,
                         conv_weights(
                             params,
                             "encoder.encoder_embed.conv.4"), modules::Conv2dLayout::ChannelsLast);
        output = modules::SwooshRModule().build(build, output);
        modules::Conv2dConfig conv7{32, 128, 3, 3, 1, 2, 0, 0, 1, 1, true};
        conv7.impl = modules::Conv2dImpl::ChannelFast;
        output = modules::Conv2dModule(conv7)
                     .build(
                         build,
                         output,
                         conv_weights(
                             params,
                             "encoder.encoder_embed.conv.7"), modules::Conv2dLayout::ChannelsLast);
        output = modules::SwooshRModule().build(build, output);

        output = modules::TransposeModule({{0, 3, 1, 2}, 4}).build(build, output);
        output = core::ensure_backend_addressable_layout(build, output);

        const auto bypass =
            modules::SliceModule({2, 0, subsampled_frames_})
                .build(build, output);
        auto padded = output;
        if (streaming) {
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
            auto new_cache =
                core::ensure_backend_addressable_layout(
                    build,
                    modules::SliceModule(
                        {2,
                         subsampled_frames_ - kEmbedCacheFrames,
                         kEmbedCacheFrames})
                        .build(build, output));
            new_cache_ = new_cache.tensor;
            padded = modules::ConcatModule({2}).build(build, cache, output);
        }
        modules::DepthwiseConv2dConfig depthwise_config{
            128, 7, 7, 1, 1, streaming ? 0 : 3, 3, 1, 1, true};
        depthwise_config.impl = modules::Conv2dImpl::ChannelFast;
        const modules::DepthwiseConv2dModule depthwise(depthwise_config);
        const auto depthwise_weights = conv_weights(params, "encoder.encoder_embed.convnext.depthwise_conv");
        if (build.backend_type == core::BackendType::Cpu) {
            padded = modules::TransposeModule({{0, 2, 3, 1}, 4}).build(build, padded);
            output = depthwise.build(build, padded, depthwise_weights, modules::Conv2dLayout::ChannelsLast);
        } else {
            output = depthwise.build(build, padded, depthwise_weights);
            output = modules::TransposeModule({{0, 2, 3, 1}, 4}).build(build, output);
        }
        // Pointwise convolutions are channel projections. Keep channel-fast
        // activations through both projections instead of building im2col matrices.
        auto pointwise1 = conv_weights(params, "encoder.encoder_embed.convnext.pointwise_conv1");
        output = modules::LinearModule({128, 384, true}).build(build, output,
            {core::reshape_tensor(build, pointwise1.weight, core::TensorShape::from_dims({384, 128})),
             pointwise1.bias});
        output = modules::SwooshLModule().build(build, output);
        auto pointwise2 = conv_weights(params, "encoder.encoder_embed.convnext.pointwise_conv2");
        output = modules::LinearModule({384, 128, true}).build(build, output,
            {core::reshape_tensor(build, pointwise2.weight, core::TensorShape::from_dims({128, 384})),
             pointwise2.bias});
        output = modules::TransposeModule({{0, 1, 3, 2}, 4}).build(build, output);
        const auto time_major_bypass = modules::TransposeModule({{0, 2, 1, 3}, 4}).build(build, bypass);
        output = modules::AddModule().build(build, time_major_bypass, output);
        output = core::reshape_tensor(
            build,
            core::ensure_backend_addressable_layout(build, output),
            core::TensorShape::from_dims(
                {1, subsampled_frames_, 128 * 19}));
        output = modules::LinearModule(
            {128 * 19, kSubsampledDim, true})
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
        output = modules::BiasNormModule({kSubsampledDim}).build(
            build, output,
            {require_param(params, "encoder.encoder_embed.out_norm.bias").tensor, norm_scale});
        output_ = output.tensor;
        ggml_set_output(output_);
        if (streaming) ggml_set_output(new_cache_);

        auto * graph = prepared_.graph();
        ggml_build_forward_expand(graph, output_);
        if (streaming) ggml_build_forward_expand(graph, new_cache_);
        prepared_.allocate();
    }

    KrokoSubsampledChunk run(
        const std::vector<float> & features) {
        if (features.size() !=
            static_cast<size_t>(input_frames_ * kFeatureDim)) {
            throw std::runtime_error(
                "Kroko subsampling feature shape does not match the package chunk size");
        }
        ggml_backend_tensor_set_async(
            backend_,
            input_,
            features.data(),
            0,
            features.size() * sizeof(float));
        if (streaming_) ggml_backend_tensor_set_async(
            backend_,
            cache_,
            cache_values_.data(),
            0,
            cache_values_.size() * sizeof(float));
        prepared_.compute();
        KrokoSubsampledChunk result;
        result.frames = subsampled_frames_;
        result.channels = kSubsampledDim;
        result.values.resize(
            static_cast<size_t>(
                subsampled_frames_ * kSubsampledDim));
        ggml_backend_tensor_get_async(
            backend_,
            output_,
            result.values.data(),
            0,
            result.values.size() * sizeof(float));
        if (streaming_) ggml_backend_tensor_get_async(
            backend_,
            new_cache_,
            cache_values_.data(),
            0,
            cache_values_.size() * sizeof(float));
        ggml_backend_synchronize(backend_);
        return result;
    }

    void reset() {
        std::fill(
            cache_values_.begin(), cache_values_.end(), 0.0F);
    }

private:
    ggml_backend_t backend_ = nullptr;
    ggml_tensor * input_ = nullptr;
    ggml_tensor * cache_ = nullptr;
    ggml_tensor * new_cache_ = nullptr;
    ggml_tensor * output_ = nullptr;
    mutable std::vector<float> cache_values_ = std::vector<float>(
        static_cast<size_t>(
            kEmbedChannels * kEmbedCacheFrames * kEmbedFrequency),
        0.0F);
    bool streaming_ = true;
    int64_t input_frames_ = 0;
    int64_t subsampled_frames_ = 0;
    core::PreparedGraph prepared_;
};

}  // namespace

struct KrokoEncoderRuntime::Impl {
    std::shared_ptr<const KrokoASRAssets> assets;
    std::shared_ptr<core::BackendWeightStore> weights;
    std::unordered_map<std::string, Param> params;
    std::unique_ptr<SubsamplingGraph> graph;
    core::ExecutionContext * execution = nullptr;
    float norm_scale = 0;
    int64_t input_frames = 0;
};

KrokoEncoderRuntime::KrokoEncoderRuntime(
    std::shared_ptr<const KrokoASRAssets> assets,
    core::ExecutionContext & execution_context)
    : impl_(std::make_unique<Impl>()) {
    if (assets == nullptr || assets->weights == nullptr) {
        throw std::runtime_error(
            "Kroko encoder requires tensor assets");
    }
    impl_->assets = std::move(assets);
    impl_->execution = &execution_context;

    impl_->weights =
        std::make_shared<core::BackendWeightStore>(
            execution_context.backend(),
            execution_context.backend_type(),
            "Kroko ASR subsampling weights",
            kWeightContextBytes);
    const auto & source = *impl_->assets->weights;
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
        {kSubsampledDim, 128 * 19});
    load(
        "encoder.encoder_embed.out.bias",
        {kSubsampledDim});
    load(
        "encoder.encoder_embed.out_norm.bias",
        {kSubsampledDim});
    impl_->weights->upload();
    const auto log_scale = source.require_f32(
        "encoder.encoder_embed.out_norm.log_scale", {1});
    impl_->norm_scale = log_scale.front();
    if (impl_->assets->config.streaming) impl_->graph = std::make_unique<SubsamplingGraph>(
        execution_context,
        impl_->params,
        impl_->norm_scale,
        impl_->assets->config.chunk_size,
        impl_->assets->config.chunk_shift / 2);
}

KrokoEncoderRuntime::~KrokoEncoderRuntime() = default;

KrokoSubsampledChunk
KrokoEncoderRuntime::encode_subsampled_chunk(
    const std::vector<float> & features) const {
    if (!impl_->assets->config.streaming) {
        const auto frames = static_cast<int64_t>(features.size() / kFeatureDim);
        if (!impl_->graph || impl_->input_frames != frames) {
            impl_->graph.reset();
            impl_->graph = std::make_unique<SubsamplingGraph>(*impl_->execution, impl_->params, impl_->norm_scale,
                frames, (frames - 7) / 2, false);
            impl_->input_frames = frames;
        }
    }
    const auto start = std::chrono::steady_clock::now();
    auto result = impl_->graph->run(features);
    engine::debug::timing_log_scalar(
        "kroko_asr.subsampling_ms",
        engine::debug::elapsed_ms(start));
    return result;
}

void KrokoEncoderRuntime::reset() {
    if (impl_->graph) impl_->graph->reset();
}

}  // namespace engine::models::kroko_asr
