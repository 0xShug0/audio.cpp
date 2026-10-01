#include "engine/models/kugelaudio/codec.h"

#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/debug/trace.h"
#include "engine/framework/modules/activation_modules.h"
#include "engine/framework/modules/primitive_modules.h"
#include "engine/framework/modules/structural_modules.h"
#include "engine/framework/modules/weight_binding.h"

#include <ggml-alloc.h>
#include <stdexcept>

namespace engine::models::kugelaudio {

struct AcousticDecoder::Impl {
    core::ExecutionContext & execution;
    CodecConfig config;
    core::BackendWeightStore weights;
    std::unique_ptr<ggml_context, decltype(&ggml_free)> context{nullptr, ggml_free};
    std::unique_ptr<ggml_gallocr, decltype(&ggml_gallocr_free)> allocator{nullptr, ggml_gallocr_free};
    core::HostGraphPlan plan;
    ggml_cgraph * graph = nullptr;
    core::TensorValue input, output;
    std::vector<core::TensorValue> state_inputs, state_outputs;

    Impl(const assets::TensorSource & source, core::ExecutionContext & execution_,
         assets::TensorStorageType storage, CodecConfig config_)
        : execution(execution_), config(std::move(config_)),
          weights(execution.backend(), execution.backend_type(), "kugelaudio.codec.weights", 4 * 1024 * 1024) {
        if (config.depths.size() != config.ratios.size() + 1) {
            throw std::runtime_error("KugelAudio codec depths/ratios mismatch");
        }
        namespace binding = modules::binding;
        context.reset(ggml_init({4 * 1024 * 1024, nullptr, true}));
        core::ModuleBuildContext ctx{context.get(), "kugelaudio.codec", execution.backend_type()};
        input = core::make_tensor(ctx, GGML_TYPE_F32, core::TensorShape::from_dims({1, config.latent_size, 1}));
        ggml_set_input(input.tensor);

        // State is device-resident and fixed-size. The graph always consumes one
        // latent frame; neither longform duration nor request count grows it.
        auto with_history = [&](const core::TensorValue & value, int64_t frames) {
            auto cache = core::make_tensor(ctx, GGML_TYPE_F32,
                core::TensorShape::from_dims({1, value.shape.dims[1], frames}));
            ggml_set_input(cache.tensor);
            // Inputs alone may be recycled after their final graph consumer.
            // History must remain allocated until the post-compute state copies.
            ggml_set_output(cache.tensor);
            auto joined = modules::ConcatModule({2}).build(ctx, cache, value);
            auto tail = modules::SliceModule({2, joined.shape.dims[2] - frames, frames}).build(ctx, joined);
            tail = core::ensure_backend_addressable_layout(ctx, tail);
            ggml_set_output(tail.tensor);
            state_inputs.push_back(cache);
            state_outputs.push_back(tail);
            return joined;
        };

        const std::string prefix = "model.acoustic_tokenizer.decoder.";
        auto x = input;
        for (size_t stage = 0; stage < config.depths.size(); ++stage) {
            const int64_t channels = config.filters * (int64_t{1} << (config.depths.size() - 1 - stage));
            const auto up_prefix = prefix + "upsample_layers." + std::to_string(stage) + ".0.";
            if (stage == 0) {
                auto conv = binding::conv1d_from_source(weights, source, up_prefix + "conv.conv",
                    storage, channels, config.latent_size, 7, true);
                x = with_history(x, 6);
                x = modules::Conv1dModule({config.latent_size, channels, 7, 1, 0, 1, true}).build(ctx, x, conv);
            } else {
                const int stride = config.ratios[stage - 1];
                auto conv = binding::conv_transpose1d_from_source(weights, source, up_prefix + "convtr.convtr",
                    storage, channels * 2, channels, 2 * stride, true);
                const int64_t new_frames = x.shape.dims[2] * stride;
                // Kernel = 2 * stride, so only one preceding input frame can
                // contribute to the new output. Older history has zero overlap.
                x = with_history(x, 1);
                x = modules::ConvTranspose1dModule({channels * 2, channels, 2 * stride, stride, 0, 1, true})
                        .build(ctx, x, conv);
                x = modules::SliceModule({2, x.shape.dims[2] - stride - new_frames, new_frames}).build(ctx, x);
            }

            for (int layer = 0; layer < config.depths[stage]; ++layer) {
                const auto name = prefix + "stages." + std::to_string(stage) + "." + std::to_string(layer) + ".";
                auto norm = binding::norm_weight_from_source(weights, source, name + "norm", channels);
                auto dw = binding::depthwise_conv1d_from_source(weights, source, name + "mixer.conv.conv.conv",
                    storage, channels, 7, true);
                auto gamma = weights.load_f32_tensor(source, name + "gamma", {channels});
                auto ffn_norm = binding::norm_weight_from_source(weights, source, name + "ffn_norm", channels);
                auto expansion = binding::linear_from_source(weights, source, name + "ffn.linear1", storage, 4 * channels, channels, true);
                auto projection = binding::linear_from_source(weights, source, name + "ffn.linear2", storage, channels, 4 * channels, true);
                auto ffn_gamma = weights.load_f32_tensor(source, name + "ffn_gamma", {channels});

                auto residual = x;
                x = modules::TransposeModule({{0, 2, 1, 3}, 3}).build(ctx, x);
                x = modules::RMSNormModule({channels, config.norm_eps, true, false}).build(ctx, x, norm);
                x = modules::TransposeModule({{0, 2, 1, 3}, 3}).build(ctx, x);
                x = with_history(x, 6);
                x = modules::DepthwiseConv1dModule({channels, 7, 1, 0, 1, true}).build(ctx, x, dw);
                gamma = core::reshape_tensor(ctx, gamma, core::TensorShape::from_dims({1, channels, 1}));
                x = modules::MulModule{}.build(ctx, x, modules::RepeatModule({x.shape}).build(ctx, gamma));
                x = modules::AddModule{}.build(ctx, residual, x);

                residual = x;
                x = modules::TransposeModule({{0, 2, 1, 3}, 3}).build(ctx, x);
                x = modules::RMSNormModule({channels, config.norm_eps, true, false}).build(ctx, x, ffn_norm);
                x = modules::LinearModule({channels, 4 * channels, true}).build(ctx, x, expansion);
                x = modules::GeluModule({modules::GeluApproximation::ExactErf}).build(ctx, x);
                x = modules::LinearModule({4 * channels, channels, true}).build(ctx, x, projection);
                ffn_gamma = core::reshape_tensor(ctx, ffn_gamma, core::TensorShape::from_dims({1, 1, channels}));
                x = modules::MulModule{}.build(ctx, x, modules::RepeatModule({x.shape}).build(ctx, ffn_gamma));
                x = modules::TransposeModule({{0, 2, 1, 3}, 3}).build(ctx, x);
                x = modules::AddModule{}.build(ctx, residual, x);
            }
        }
        auto head = binding::conv1d_from_source(weights, source, prefix + "head.conv.conv", storage, 1, config.filters, 7, true);
        x = with_history(x, 6);
        x = modules::Conv1dModule({config.filters, 1, 7, 1, 0, 1, true}).build(ctx, x, head);
        output = core::ensure_backend_addressable_layout(ctx, x);
        ggml_set_output(output.tensor);
        weights.upload();

        graph = ggml_new_graph_custom(context.get(), 8192, false);
        ggml_build_forward_expand(graph, output.tensor);
        for (const auto & state : state_outputs) {
            ggml_build_forward_expand(graph, state.tensor);
        }
        core::validate_backend_graph_supported(execution.backend(), graph, "KugelAudio codec");
        allocator.reset(ggml_gallocr_new(ggml_backend_get_default_buffer_type(execution.backend())));
        if (!ggml_gallocr_alloc_graph(allocator.get(), graph)) {
            throw std::runtime_error("KugelAudio codec graph allocation failed");
        }
        core::prepare_host_graph_plan(execution, graph, plan);
        debug::timing_log_context_reservation("kugelaudio.codec.graph", context.get());
        debug::timing_log_scalar("kugelaudio.codec.graph.buffer_mb",
            static_cast<double>(ggml_gallocr_get_buffer_size(allocator.get(), 0)) / (1024 * 1024));
        reset();
    }

    ~Impl() {
        if (graph != nullptr) {
            core::release_backend_graph_resources(execution.backend(), graph, true);
        }
    }

    void reset() {
        for (const auto & state : state_inputs) {
            ggml_backend_tensor_memset(state.tensor, 0, 0, ggml_nbytes(state.tensor));
        }
    }

    std::vector<float> decode_frame(const std::vector<float> & latent) {
        core::write_tensor_f32(input, latent);
        if (core::compute_graph(execution, graph, plan, "KugelAudio codec") != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("KugelAudio codec compute failed");
        }
        for (size_t i = 0; i < state_inputs.size(); ++i) {
            ggml_backend_tensor_copy(state_outputs[i].tensor, state_inputs[i].tensor);
        }
        return core::read_tensor_f32(output.tensor);
    }
};

AcousticDecoder::AcousticDecoder(const assets::TensorSource & source, core::ExecutionContext & execution,
    assets::TensorStorageType storage, CodecConfig config)
    : impl_(std::make_unique<Impl>(source, execution, storage, std::move(config))) {}
AcousticDecoder::~AcousticDecoder() = default;
void AcousticDecoder::reset() { impl_->reset(); }
std::vector<float> AcousticDecoder::decode_frame(const std::vector<float> & latent) { return impl_->decode_frame(latent); }

}  // namespace engine::models::kugelaudio
