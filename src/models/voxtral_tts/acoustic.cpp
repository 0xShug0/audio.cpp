#include "engine/models/voxtral_tts/acoustic.h"

#include "engine/models/voxtral_tts/ar.h"
#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/core/constant_tensor_cache.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/modules/primitive_modules.h"
#include "engine/framework/modules/structural_modules.h"
#include "engine/framework/modules/weight_binding.h"

#include <ggml-alloc.h>
#include <algorithm>
#include <cmath>

namespace engine::models::voxtral_tts {

struct VoxtralAcousticFlowRuntime::Impl {
    core::ExecutionContext & execution;
    const VoxtralTtsConfig & config;
    core::BackendWeightStore store;
    modules::LinearWeights input_projection, time_projection, hidden_projection, output_projection;
    modules::NormWeights norm;
    std::vector<modules::DecoderLayerWeights> layers;
    std::unique_ptr<ggml_context, decltype(&ggml_free)> context{nullptr, ggml_free};
    std::unique_ptr<ggml_gallocr, decltype(&ggml_gallocr_free)> allocator{nullptr, ggml_gallocr_free};
    std::unique_ptr<core::ConstantTensorCache> constants;
    ggml_cgraph * graph = nullptr;
    core::HostGraphPlan plan;
    core::TensorValue hidden, noise, time, guidance, unconditioned_guidance, delta, mask, output;
    int cached_steps = 0;

    Impl(const VoxtralTtsAssets & assets, core::ExecutionContext & execution_, assets::TensorStorageType storage)
        : execution(execution_), config(assets.config),
          store(execution.backend(), execution.backend_type(), "voxtral_tts.acoustic.weights", kContextBytes) {
        namespace binding = modules::binding;
        const auto & source = *assets.weights;
        const auto width = config.acoustic.hidden_size;
        const std::string prefix = "acoustic_transformer.";
        input_projection = binding::linear_from_source(store, source, prefix + "input_projection", storage,
                                                       width, config.acoustic_dim, false);
        time_projection = binding::linear_from_source(store, source, prefix + "time_projection", storage,
                                                      width, width, false);
        hidden_projection = binding::linear_from_source(store, source, prefix + "llm_projection", storage,
                                                        width, config.ar.hidden_size, false);
        output_projection = binding::linear_from_source(store, source, prefix + "acoustic_codebook_output", storage,
                                                        config.acoustic_dim, width, false);
        norm = binding::norm_weight_from_source(store, source, prefix + "norm", width);
        for (int64_t i = 0; i < config.acoustic.layers; ++i) {
            layers.push_back(load_transformer_layer(store, source, prefix + "layers." + std::to_string(i),
                                                    config.acoustic, storage));
        }
        store.upload();
        prepare(7);
    }

    ~Impl() { release_graph(); }

    void release_graph() {
        if (graph != nullptr) {
            core::release_backend_graph_resources(execution.backend(), graph, true);
        }
        graph = nullptr;
        plan.reset();
        allocator.reset();
        context.reset();
        constants.reset();
        cached_steps = 0;
    }

    void prepare(int steps) {
        if (steps <= 0) {
            throw std::runtime_error("Voxtral TTS acoustic steps must be positive");
        }
        if (steps == cached_steps) {
            return;
        }
        release_graph();
        context.reset(ggml_init({kContextBytes, nullptr, true}));
        core::ModuleBuildContext ctx{context.get(), "voxtral_tts.acoustic", execution.backend_type()};
        const auto width = config.acoustic.hidden_size;
        const auto latent_shape = core::TensorShape::from_dims({1, 1, config.acoustic_dim});
        constants = std::make_unique<core::ConstantTensorCache>(execution.backend(), execution.config().threads,
                                                               "voxtral_tts.acoustic.constants", kContextBytes);
        constants->begin_graph();
        std::vector<float> time_features(static_cast<size_t>(steps * width));
        for (int step = 0; step < steps; ++step) {
            for (int64_t i = 0; i < width / 2; ++i) {
                const auto frequency = std::exp(-std::log(10000.0F) * static_cast<float>(i) / (width / 2));
                const auto phase = static_cast<float>(step) / steps * frequency;
                time_features[step * width + i] = std::cos(phase);
                time_features[step * width + width / 2 + i] = std::sin(phase);
            }
        }
        hidden = core::make_tensor(ctx, GGML_TYPE_F32, core::TensorShape::from_dims({2, 1, config.ar.hidden_size}));
        noise = core::make_tensor(ctx, GGML_TYPE_F32, latent_shape);
        time = constants->make_f32(core::TensorShape::from_dims({steps, width}), time_features);
        guidance = core::make_tensor(ctx, GGML_TYPE_F32, latent_shape);
        unconditioned_guidance = core::make_tensor(ctx, GGML_TYPE_F32, latent_shape);
        delta = constants->make_f32(latent_shape, std::vector<float>(config.acoustic_dim, 1.0F / steps));
        // The acoustic flow transformer is bidirectional, unlike the AR backbone.
        std::vector<ggml_fp16_t> mask_values(18, ggml_fp32_to_fp16(0.0F));
        mask = constants->make_tensor(core::TensorShape::from_dims({2, 1, 3, 3}), GGML_TYPE_F16,
                                      mask_values.data(), mask_values.size() * sizeof(ggml_fp16_t));
        constants->finish_graph();
        constants->ensure_uploaded();
        debug::timing_log_scalar("voxtral_tts.acoustic.constants.buffer_mb",
            static_cast<double>(ggml_backend_buffer_get_size(time.tensor->buffer)) / (1024 * 1024));
        for (const auto & input : {hidden, noise, guidance, unconditioned_guidance}) {
            ggml_set_input(input.tensor);
        }
        auto projected_hidden = modules::LinearModule({config.ar.hidden_size, width, false})
                                    .build(ctx, hidden, hidden_projection);
        auto projected_time = modules::LinearModule({width, width, false}).build(ctx, time, time_projection);
        auto x = noise;
        modules::DecoderLayerConfig layer_config;
        layer_config.hidden_size = width;
        layer_config.intermediate_size = config.acoustic.intermediate_size;
        layer_config.num_attention_heads = config.acoustic.attention_heads;
        layer_config.num_key_value_heads = config.acoustic.kv_heads;
        layer_config.head_dim = config.acoustic.head_dim;
        layer_config.rms_norm_eps = config.acoustic.norm_eps;
        layer_config.use_qk_norm = false;
        layer_config.position_encoding = modules::DecoderPositionEncoding::None;
        layer_config.runtime.attention.prefill_mode = modules::DecoderAttentionMode::FlashGroupedViewKV;
        for (int i = 0; i < steps; ++i) {
            auto repeated = modules::RepeatModule({core::TensorShape::from_dims({2, 1, config.acoustic_dim})})
                                .build(ctx, x);
            auto acoustic = modules::LinearModule({config.acoustic_dim, width, false})
                                .build(ctx, repeated, input_projection);
            auto step_time = modules::SliceModule({0, i, 1}).build(ctx, projected_time);
            step_time = core::reshape_tensor(ctx, step_time, core::TensorShape::from_dims({1, 1, width}));
            step_time = modules::RepeatModule({core::TensorShape::from_dims({2, 1, width})}).build(ctx, step_time);
            auto tokens = modules::ConcatModule({1}).build(ctx, acoustic, step_time);
            tokens = modules::ConcatModule({1}).build(ctx, tokens, projected_hidden);
            for (const auto & layer : layers) {
                tokens = modules::DecoderLayerModule(layer_config)
                             .build(ctx, tokens, {}, layer, std::nullopt, std::nullopt, mask).output;
            }
            tokens = modules::RMSNormModule({width, config.acoustic.norm_eps, true, false}).build(ctx, tokens, norm);
            tokens = modules::SliceModule({1, 0, 1}).build(ctx, tokens);
            auto velocity = modules::LinearModule({width, config.acoustic_dim, false})
                                .build(ctx, tokens, output_projection);
            auto conditional = modules::SliceModule({0, 0, 1}).build(ctx, velocity);
            auto unconditional = modules::SliceModule({0, 1, 1}).build(ctx, velocity);
            conditional = modules::MulModule{}.build(ctx, conditional, guidance);
            unconditional = modules::MulModule{}.build(ctx, unconditional, unconditioned_guidance);
            velocity = modules::AddModule{}.build(ctx, conditional, unconditional);
            velocity = modules::MulModule{}.build(ctx, velocity, delta);
            x = modules::AddModule{}.build(ctx, x, velocity);
        }
        output = x;
        ggml_set_output(output.tensor);
        graph = ggml_new_graph_custom(context.get(), static_cast<size_t>(steps) * 512 + 256, false);
        ggml_build_forward_expand(graph, output.tensor);
        core::validate_backend_graph_supported(execution.backend(), graph, "Voxtral TTS acoustic flow");
        allocator.reset(ggml_gallocr_new(ggml_backend_get_default_buffer_type(execution.backend())));
        if (!ggml_gallocr_alloc_graph(allocator.get(), graph)) {
            throw std::runtime_error("Voxtral TTS acoustic graph allocation failed");
        }
        core::prepare_host_graph_plan(execution, graph, plan);
        cached_steps = steps;
        debug::timing_log_context_reservation("voxtral_tts.acoustic.graph", context.get());
        debug::timing_log_scalar("voxtral_tts.acoustic.graph.buffer_mb",
            static_cast<double>(ggml_gallocr_get_buffer_size(allocator.get(), 0)) / (1024 * 1024));
    }

    std::vector<int32_t> sample(const std::vector<float> & condition, const std::vector<float> & initial_noise,
                               int steps, float scale) {
        prepare(steps);
        if (condition.size() != static_cast<size_t>(config.ar.hidden_size) ||
            initial_noise.size() != static_cast<size_t>(config.acoustic_dim)) {
            throw std::runtime_error("Voxtral TTS acoustic conditioning shape mismatch");
        }
        std::vector<float> batched(2 * condition.size(), 0.0F);
        std::copy(condition.begin(), condition.end(), batched.begin());
        core::write_tensor_f32(hidden, batched);
        core::write_tensor_f32(noise, initial_noise);
        core::write_tensor_f32(guidance, std::vector<float>(config.acoustic_dim, scale));
        core::write_tensor_f32(unconditioned_guidance, std::vector<float>(config.acoustic_dim, 1.0F - scale));
        if (core::compute_graph(execution, graph, plan, "Voxtral TTS acoustic flow") != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("Voxtral TTS acoustic flow compute failed");
        }
        auto values = core::read_tensor_f32(output.tensor);
        std::vector<int32_t> result(values.size());
        for (size_t i = 0; i < values.size(); ++i) {
            const auto value = (std::clamp(values[i], -1.0F, 1.0F) + 1.0F) * 0.5F * (config.acoustic_levels - 1);
            result[i] = static_cast<int32_t>(std::nearbyint(value)) + kAudioTokenOffset;
        }
        return result;
    }
};

VoxtralAcousticFlowRuntime::VoxtralAcousticFlowRuntime(
    const VoxtralTtsAssets & assets, core::ExecutionContext & execution, assets::TensorStorageType storage)
    : impl_(std::make_unique<Impl>(assets, execution, storage)) {}
VoxtralAcousticFlowRuntime::~VoxtralAcousticFlowRuntime() = default;
std::vector<int32_t> VoxtralAcousticFlowRuntime::sample(
    const std::vector<float> & hidden, const std::vector<float> & noise, int steps, float guidance) {
    return impl_->sample(hidden, noise, steps, guidance);
}

}  // namespace engine::models::voxtral_tts
