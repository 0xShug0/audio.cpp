#include "engine/models/voxtral_tts/codec.h"

#include "engine/models/voxtral_tts/ar.h"
#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/core/constant_tensor_cache.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/modules/activation_modules.h"
#include "engine/framework/modules/attention/scaled_dot_product_attention.h"
#include "engine/framework/modules/conv_modules.h"
#include "engine/framework/modules/lookup_modules.h"
#include "engine/framework/modules/primitive_modules.h"
#include "engine/framework/modules/structural_modules.h"
#include "engine/framework/modules/weight_binding.h"

#include <ggml-alloc.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <numeric>

namespace engine::models::voxtral_tts {

struct VoxtralCausalCodecDecoderRuntime::Impl {
    struct LayerWeights {
        modules::DecoderLayerWeights transformer;
        modules::NormWeights q_norm, k_norm;
        core::TensorValue attention_scale, ffn_scale;
    };
    struct DecoderGraph {
        std::unique_ptr<ggml_context, decltype(&ggml_free)> context{nullptr, ggml_free};
        std::unique_ptr<ggml_gallocr, decltype(&ggml_gallocr_free)> allocator{nullptr, ggml_gallocr_free};
        std::unique_ptr<core::ConstantTensorCache> constants;
        ggml_cgraph * graph = nullptr;
        core::HostGraphPlan plan;
        core::TensorValue semantic, acoustic, output;
        std::vector<core::TensorValue> masks;
        int64_t frames = 0;
    };
    core::ExecutionContext & execution;
    const VoxtralTtsConfig & config;
    core::BackendWeightStore store;
    core::TensorValue semantic_embedding;
    std::vector<modules::Conv1dWeights> convolutions;
    modules::Conv1dWeights output_projection;
    std::vector<std::vector<LayerWeights>> layers;
    std::array<DecoderGraph, 2> graphs;

    Impl(const VoxtralTtsAssets & assets, core::ExecutionContext & execution_, assets::TensorStorageType storage)
        : execution(execution_), config(assets.config),
          store(execution.backend(), execution.backend_type(), "voxtral_tts.codec.weights", kContextBytes) {
        const auto & source = *assets.weights;
        const auto width = config.codec.hidden_size;
        semantic_embedding = store.load_tensor(source,
            "audio_tokenizer.quantizer.semantic_codebook.embedding", assets::TensorStorageType::Native,
            {config.semantic_vocab_size, config.semantic_dim});
        for (size_t stage = 0; stage < config.codec_layers.size(); ++stage) {
            const auto prefix = "audio_tokenizer.decoder_blocks." + std::to_string(stage * 2);
            const auto input = stage == 0 ? config.semantic_dim + config.acoustic_dim : width;
            // Transposed convolution consumes F32 on GPU. One shared upload avoids
            // a separate BF16-to-F32 weight copy in each cached decode graph.
            convolutions.push_back({store.load_f32_tensor(source, prefix + ".conv.weight",
                                                           {width, input, config.codec_kernels[stage]}), std::nullopt});
            std::vector<LayerWeights> block;
            for (int layer = 0; layer < config.codec_layers[stage]; ++layer) {
                const auto name = "audio_tokenizer.decoder_blocks." + std::to_string(stage * 2 + 1) +
                                  ".layers." + std::to_string(layer);
                LayerWeights weights;
                weights.transformer = load_transformer_layer(store, source, name, config.codec, storage);
                weights.q_norm = modules::binding::norm_weight_from_source(store, source,
                    name + ".attention.q_norm", config.codec.attention_heads * config.codec.head_dim);
                weights.k_norm = modules::binding::norm_weight_from_source(store, source,
                    name + ".attention.k_norm", config.codec.kv_heads * config.codec.head_dim);
                weights.attention_scale = store.load_f32_tensor(source, name + ".attention_scale", {width});
                weights.ffn_scale = store.load_f32_tensor(source, name + ".ffn_scale", {width});
                block.push_back(std::move(weights));
            }
            layers.push_back(std::move(block));
        }
        output_projection = {store.load_f32_tensor(source, "audio_tokenizer.output_proj.conv.weight",
                                                   {config.patch_size, width, config.patch_kernel_size}), std::nullopt};
        store.upload();
        build_graph(graphs[0], kCodecInitialFrames);
        build_graph(graphs[1], kCodecContextFrames + kCodecChunkFrames);
    }

    ~Impl() {
        for (auto & graph : graphs) {
            core::release_backend_graph_resources(execution.backend(), graph.graph, true);
            graph.plan.reset();
            graph.allocator.reset();
            graph.context.reset();
        }
    }

    void build_graph(DecoderGraph & graph, int64_t input_frames) {
        graph.context.reset(ggml_init({kContextBytes, nullptr, true}));
        graph.constants = std::make_unique<core::ConstantTensorCache>(execution.backend(), execution.config().threads,
                                                                     "voxtral_tts.codec.constants", kContextBytes);
        graph.constants->begin_graph();
        core::ModuleBuildContext ctx{graph.context.get(), "voxtral_tts.codec", execution.backend_type()};
        const auto width = config.codec.hidden_size;
        graph.frames = input_frames;
        graph.semantic = core::make_tensor(ctx, GGML_TYPE_I32, core::TensorShape::from_dims({input_frames}));
        graph.acoustic = core::make_tensor(ctx, GGML_TYPE_F32,
            core::TensorShape::from_dims({1, input_frames, config.acoustic_dim}));
        ggml_set_input(graph.semantic.tensor);
        ggml_set_input(graph.acoustic.tensor);
        auto semantic = modules::EmbeddingModule({config.semantic_vocab_size, config.semantic_dim})
                            .build(ctx, graph.semantic, semantic_embedding);
        semantic = core::reshape_tensor(ctx, semantic, core::TensorShape::from_dims({1, input_frames, config.semantic_dim}));
        auto x = modules::ConcatModule({2}).build(ctx, semantic, graph.acoustic);
        int64_t frames = input_frames;
        for (size_t stage = 0; stage < layers.size(); ++stage) {
            x = modules::TransposeModule({{0, 2, 1, 3}, 3}).build(ctx, x);
            if (stage == 0) {
                const auto pad = config.codec_kernels[stage] - config.codec_strides[stage];
                auto left = modules::SliceModule({2, 0, 1}).build(ctx, x);
                left = modules::RepeatModule({core::TensorShape::from_dims({1, config.semantic_dim + config.acoustic_dim, pad})})
                           .build(ctx, left);
                x = modules::ConcatModule({2}).build(ctx, left, x);
                x = modules::Conv1dModule({config.semantic_dim + config.acoustic_dim, width,
                                          config.codec_kernels[stage], 1, 0, 1, false})
                        .build(ctx, x, convolutions[stage]);
            } else {
                x = modules::ConvTranspose1dModule({width, width, config.codec_kernels[stage],
                                                    config.codec_strides[stage], 0, 1, false})
                        .build(ctx, x, {convolutions[stage].weight, std::nullopt});
                frames *= config.codec_strides[stage];
                x = modules::SliceModule({2, 0, frames}).build(ctx, x);
            }
            x = modules::TransposeModule({{0, 2, 1, 3}, 3}).build(ctx, x);
            const auto mask_shape = core::TensorShape::from_dims({1, config.codec.attention_heads, frames, frames});
            const auto minus_inf = ggml_fp32_to_fp16(-std::numeric_limits<float>::infinity());
            std::vector<ggml_fp16_t> values(static_cast<size_t>(mask_shape.num_elements()), minus_inf);
            const auto window = config.codec_windows[stage];
            for (int64_t head = 0; head < config.codec.attention_heads; ++head) {
                const float slope = std::pow(2.0F, -8.0F * head / config.codec.attention_heads);
                for (int64_t query = 0; query < frames; ++query) {
                    for (int64_t key = std::max<int64_t>(0, query - window); key <= query; ++key) {
                        values[(head * frames + query) * frames + key] = ggml_fp32_to_fp16(slope * (key - query));
                    }
                }
            }
            auto mask = graph.constants->make_tensor(mask_shape, GGML_TYPE_F16, values.data(),
                                                     values.size() * sizeof(ggml_fp16_t));
            graph.masks.push_back(mask);
            for (const auto & layer : layers[stage]) {
                auto normalized = modules::RMSNormModule({width, config.codec.norm_eps, true, false})
                                      .build(ctx, x, layer.transformer.input_norm);
                const auto & attention = layer.transformer.self_attention;
                const auto query_width = config.codec.attention_heads * config.codec.head_dim;
                const auto kv_width = config.codec.kv_heads * config.codec.head_dim;
                auto q = modules::LinearModule({width, query_width, false})
                             .build(ctx, normalized, {attention.q_weight, std::nullopt});
                auto k = modules::LinearModule({width, kv_width, false})
                             .build(ctx, normalized, {attention.k_weight, std::nullopt});
                auto v = modules::LinearModule({width, kv_width, false})
                             .build(ctx, normalized, {attention.v_weight, std::nullopt});
                q = modules::RMSNormModule({query_width, config.codec_qk_norm_eps, true, false}).build(ctx, q, layer.q_norm);
                k = modules::RMSNormModule({kv_width, config.codec_qk_norm_eps, true, false}).build(ctx, k, layer.k_norm);
                q = core::reshape_tensor(ctx, q, core::TensorShape::from_dims({1, frames, config.codec.attention_heads, config.codec.head_dim}));
                k = core::reshape_tensor(ctx, k, core::TensorShape::from_dims({1, frames, config.codec.kv_heads, config.codec.head_dim}));
                v = core::reshape_tensor(ctx, v, core::TensorShape::from_dims({1, frames, config.codec.kv_heads, config.codec.head_dim}));
                q = modules::TransposeModule({{0, 2, 1, 3}, 4}).build(ctx, q);
                k = modules::TransposeModule({{0, 2, 1, 3}, 4}).build(ctx, k);
                v = modules::TransposeModule({{0, 2, 1, 3}, 4}).build(ctx, v);
                modules::ScaledDotProductAttentionConfig attention_config;
                attention_config.head_dim = config.codec.head_dim;
                attention_config.lowering = modules::ScaledDotProductAttentionLowering::Flash;
                auto attended = modules::ScaledDotProductAttentionModule(attention_config).build(ctx, q, k, v, mask);
                attended = core::reshape_tensor(ctx, core::ensure_backend_addressable_layout(ctx, attended),
                                                core::TensorShape::from_dims({1, frames, query_width}));
                attended = modules::LinearModule({query_width, width, false})
                               .build(ctx, attended, {attention.out_weight, std::nullopt});
                auto scale = core::reshape_tensor(ctx, layer.attention_scale, core::TensorShape::from_dims({1, 1, width}));
                scale = modules::RepeatModule({attended.shape}).build(ctx, scale);
                attended = modules::MulModule{}.build(ctx, attended, scale);
                x = modules::AddModule{}.build(ctx, x, attended);
                normalized = modules::RMSNormModule({width, config.codec.norm_eps, true, false})
                                 .build(ctx, x, layer.transformer.post_norm);
                const auto intermediate = config.codec.intermediate_size;
                auto gate = modules::LinearModule({width, intermediate, false}).build(ctx, normalized, layer.transformer.mlp.gate_proj);
                gate = modules::SiluModule{}.build(ctx, gate);
                auto up = modules::LinearModule({width, intermediate, false}).build(ctx, normalized, layer.transformer.mlp.up_proj);
                auto ff = modules::MulModule{}.build(ctx, gate, up);
                ff = modules::LinearModule({intermediate, width, false}).build(ctx, ff, layer.transformer.mlp.down_proj);
                scale = core::reshape_tensor(ctx, layer.ffn_scale, core::TensorShape::from_dims({1, 1, width}));
                scale = modules::RepeatModule({ff.shape}).build(ctx, scale);
                ff = modules::MulModule{}.build(ctx, ff, scale);
                x = modules::AddModule{}.build(ctx, x, ff);
            }
        }
        x = modules::TransposeModule({{0, 2, 1, 3}, 3}).build(ctx, x);
        x = core::ensure_backend_addressable_layout(ctx, x);
        x = modules::ReflectPad1dModule({config.patch_kernel_size - 1, 0}).build(ctx, x);
        x = modules::Conv1dModule({width, config.patch_size, config.patch_kernel_size, 1, 0, 1, false})
                .build(ctx, x, output_projection);
        x = modules::TransposeModule({{0, 2, 1, 3}, 3}).build(ctx, x);
        graph.output = core::reshape_tensor(ctx, core::ensure_backend_addressable_layout(ctx, x),
                                            core::TensorShape::from_dims({frames * config.patch_size}));
        ggml_set_output(graph.output.tensor);
        graph.graph = ggml_new_graph_custom(graph.context.get(), 2048, false);
        ggml_build_forward_expand(graph.graph, graph.output.tensor);
        graph.constants->finish_graph();
        graph.constants->ensure_uploaded();
        core::validate_backend_graph_supported(execution.backend(), graph.graph, "Voxtral TTS codec decoder");
        graph.allocator.reset(ggml_gallocr_new(ggml_backend_get_default_buffer_type(execution.backend())));
        if (!ggml_gallocr_alloc_graph(graph.allocator.get(), graph.graph)) {
            throw std::runtime_error("Voxtral TTS codec graph allocation failed");
        }
        core::prepare_host_graph_plan(execution, graph.graph, graph.plan);
        const auto name = "voxtral_tts.codec.graph." + std::to_string(input_frames);
        debug::timing_log_context_reservation(name, graph.context.get());
        debug::timing_log_scalar(name + ".buffer_mb",
            static_cast<double>(ggml_gallocr_get_buffer_size(graph.allocator.get(), 0)) / (1024 * 1024));
        debug::timing_log_scalar(name + ".constants.buffer_mb",
            static_cast<double>(ggml_backend_buffer_get_size(graph.masks.front().tensor->buffer)) / (1024 * 1024));
    }

    std::vector<float> decode(const std::vector<int32_t> & codes, bool initial) {
        auto & graph = graphs[initial ? 0 : 1];
        const auto books = config.acoustic_dim + 1;
        const auto frames = static_cast<int64_t>(codes.size()) / books;
        if (frames <= 0 || frames > graph.frames || frames * books != static_cast<int64_t>(codes.size())) {
            throw std::runtime_error("Voxtral TTS codec input exceeds its bounded chunk graph");
        }
        std::vector<int32_t> semantic(static_cast<size_t>(graph.frames));
        std::vector<float> acoustic(static_cast<size_t>(graph.frames * config.acoustic_dim));
        for (int64_t frame = 0; frame < graph.frames; ++frame) {
            const auto offset = std::min(frame, frames - 1) * books;
            semantic[frame] = codes[offset] - kAudioTokenOffset;
            for (int64_t book = 0; book < config.acoustic_dim; ++book) {
                acoustic[frame * config.acoustic_dim + book] =
                    2.0F * (codes[offset + book + 1] - kAudioTokenOffset) / (config.acoustic_levels - 1) - 1.0F;
            }
        }
        ggml_backend_tensor_set(graph.semantic.tensor, semantic.data(), 0, semantic.size() * sizeof(int32_t));
        core::write_tensor_f32(graph.acoustic, acoustic);
        if (core::compute_graph(execution, graph.graph, graph.plan, "Voxtral TTS codec") != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("Voxtral TTS codec compute failed");
        }
        auto result = core::read_tensor_f32(graph.output.tensor);
        const auto stride = config.patch_size * std::accumulate(config.codec_strides.begin(), config.codec_strides.end(), int64_t{1}, std::multiplies<int64_t>{});
        result.resize(static_cast<size_t>(frames * stride));
        return result;
    }
};

VoxtralCausalCodecDecoderRuntime::VoxtralCausalCodecDecoderRuntime(
    const VoxtralTtsAssets & assets, core::ExecutionContext & execution, assets::TensorStorageType storage)
    : impl_(std::make_unique<Impl>(assets, execution, storage)) {}
VoxtralCausalCodecDecoderRuntime::~VoxtralCausalCodecDecoderRuntime() = default;
std::vector<float> VoxtralCausalCodecDecoderRuntime::decode(const std::vector<int32_t> & codes, bool initial) {
    return impl_->decode(codes, initial);
}

}  // namespace engine::models::voxtral_tts
