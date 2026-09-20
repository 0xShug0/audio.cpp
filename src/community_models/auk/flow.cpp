#include "engine/community_models/auk/flow.h"
#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/debug/trace.h"
#include "engine/framework/modules/activation_modules.h"
#include "engine/framework/modules/structural_modules.h"
#include "engine/framework/modules/positional_modules.h"
#include "engine/framework/modules/attention/feed_forward.h"
#include "engine/framework/modules/attention/scaled_dot_product_attention.h"
#include "engine/framework/modules/weight_binding.h"
#include "engine/framework/modules/flow_sampler_runtime.h"
#include "engine/framework/runtime/graph_optimizer.h"

#include <ggml-alloc.h>
#include <array>
#include <cmath>
#include <limits>
#include <unordered_set>
#include <numeric>

namespace engine::models::auk {
namespace {

using core::TensorValue;
using core::TensorShape;

struct EmbeddingWeights {
    TensorValue frequencies;
    modules::LinearWeights time_first;
    modules::LinearWeights time_second;
    modules::LinearWeights text;
    modules::NormWeights text_norm;
    modules::LinearWeights audio;
    std::array<modules::Conv1dWeights, 2> position;
};

struct StreamWeights {
    modules::LinearWeights modulation;
    modules::LinearWeights qkv;
    modules::NormWeights q_norm;
    modules::NormWeights k_norm;
    modules::LinearWeights output;
    modules::LinearWeights feed_forward_in;
    modules::LinearWeights feed_forward_out;
};

template <size_t Streams>
std::array<StreamWeights, Streams> load_transformer_block(core::BackendWeightStore & store,
    const assets::TensorSource & source, const std::string & prefix) {
    using namespace modules::binding;
    constexpr auto storage = assets::TensorStorageType::Native;
    std::array<StreamWeights, Streams> weights;
    for (size_t stream = 0; stream < Streams; ++stream) {
        const std::string branch = Streams == 1 ? "" : stream == 0 ? "_x" : "_c";
        const std::string suffix = stream == 0 ? "" : "_c";
        auto & w = weights[stream];
        w.modulation = linear_from_source(store, source, prefix + ".attn_norm" + branch + ".linear", storage, 9216, 1536, true);
        w.qkv = linear_from_source(store, source, prefix + ".attn.to_qkv" + suffix, storage, 4608, 1536, true);
        w.q_norm = norm_weight_from_source(store, source, prefix + ".attn." + (stream == 0 ? "q_norm" : "c_q_norm"), 64);
        w.k_norm = norm_weight_from_source(store, source, prefix + ".attn." + (stream == 0 ? "k_norm" : "c_k_norm"), 64);
        w.output = linear_from_source(store, source, prefix + ".attn.to_out" + (stream == 0 ? ".0" : "_c"), storage, 1536, 1536, true);
        w.feed_forward_in = linear_from_source(store, source, prefix + ".ff" + branch + ".linear_in", storage, 6144, 1536, false);
        w.feed_forward_out = linear_from_source(store, source, prefix + ".ff" + branch + ".linear_out", storage, 1536, 3072, false);
    }
    return weights;
}

template <size_t Streams>
std::array<TensorValue, Streams> build_transformer_block(core::ModuleBuildContext & ctx,
    const std::array<StreamWeights, Streams> & weights, const std::array<TensorValue, Streams> & inputs,
    const std::array<TensorValue, Streams> & parameters, const std::array<TensorValue, Streams> & positions, const TensorValue & rope_factors,
    bool flash_attention,
    std::map<std::string, TensorValue> * boundaries = nullptr,
    const std::optional<TensorValue> & attention_mask = std::nullopt,
    const std::array<std::optional<TensorValue>, Streams> & output_masks = {}) {
    std::array<std::array<TensorValue, 6>, Streams> modulation;
    std::array<std::array<TensorValue, 3>, Streams> projections;
    const auto batch = inputs[0].shape.dims[0];
    const modules::LayerNormModule norm({1536, 1e-6F, false, false});
    int64_t total_tokens = 0;
    for (size_t stream = 0; stream < Streams; ++stream) {
        total_tokens += inputs[stream].shape.dims[1];
        for (int part = 0; part < 6; ++part) {
            const auto parameter = modules::SliceModule({1, part * 1536, 1536}).build(ctx, parameters[stream]);
            modulation[stream][part] = core::reshape_tensor(ctx,
                core::ensure_backend_addressable_layout(ctx, parameter), TensorShape::from_dims({batch, 1, 1536}));
        }
        const auto normalized = norm.build(ctx, inputs[stream], {});
        const auto modulated = core::wrap_tensor(ggml_add(ctx.ggml,
            ggml_mul(ctx.ggml, normalized.tensor, ggml_scale_bias(ctx.ggml, modulation[stream][1].tensor, 1.0F, 1.0F)),
            modulation[stream][0].tensor), normalized.shape);
        const auto packed = modules::LinearModule({1536, 4608, true, GGML_PREC_F32}).build(ctx, modulated, weights[stream].qkv);
        if (boundaries) {
            const std::string branch = stream == 0 ? "x." : "c.";
            (*boundaries)[branch + "mod"] = modulated;
            (*boundaries)[branch + "qkv"] = packed;
        }
        for (int part = 0; part < 3; ++part) {
            const auto projection = modules::SliceModule({2, part * 1536, 1536}).build(ctx, packed);
            auto heads = core::reshape_tensor(ctx, core::ensure_backend_addressable_layout(ctx, projection),
                TensorShape::from_dims({batch, inputs[stream].shape.dims[1], 24, 64}));
            if (part < 2) {
                heads = modules::RMSNormModule({64, std::numeric_limits<float>::epsilon(), true, false})
                    .build(ctx, heads, part == 0 ? weights[stream].q_norm : weights[stream].k_norm);
                if (boundaries) {
                    (*boundaries)[std::string(stream == 0 ? "x." : "c.") + (part == 0 ? "q_norm" : "k_norm")] = heads;
                }
                heads = modules::RoPEModule({64, GGML_ROPE_TYPE_NORMAL, 1.0F}).build(ctx, heads, positions[stream], &rope_factors);
            }
            projections[stream][part] = modules::TransposeModule({{0, 2, 1, 3}, 4}).build(ctx, heads);
        }
    }
    std::array<TensorValue, 3> joint;
    for (int part = 0; part < 3; ++part) {
        joint[part] = projections[0][part];
        for (size_t stream = 1; stream < Streams; ++stream) {
            joint[part] = modules::ConcatModule({2}).build(ctx, joint[part], projections[stream][part]);
        }
    }
    auto attention = modules::ScaledDotProductAttentionModule({64, flash_attention
        ? modules::ScaledDotProductAttentionLowering::Flash : modules::ScaledDotProductAttentionLowering::Explicit, GGML_PREC_F32})
        .build(ctx, joint[0], joint[1], joint[2], attention_mask);
    attention = core::reshape_tensor(ctx, core::ensure_backend_addressable_layout(ctx, attention),
        TensorShape::from_dims({batch, total_tokens, 1536}));
    std::array<TensorValue, Streams> result;
    int64_t offset = 0;
    for (size_t stream = 0; stream < Streams; ++stream) {
        const auto attended = modules::SliceModule({1, offset, inputs[stream].shape.dims[1]}).build(ctx, attention);
        if (boundaries) (*boundaries)[std::string(stream == 0 ? "x." : "c.") + "attention"] = attended;
        offset += inputs[stream].shape.dims[1];
        auto projected = modules::LinearModule({1536, 1536, true, GGML_PREC_F32}).build(ctx, attended, weights[stream].output);
        if (boundaries) (*boundaries)[std::string(stream == 0 ? "x." : "c.") + "attn_output"] = projected;
        if (output_masks[stream]) {
            projected = core::wrap_tensor(ggml_mul(ctx.ggml, projected.tensor, output_masks[stream]->tensor), projected.shape);
        }
        auto residual = core::wrap_tensor(ggml_add(ctx.ggml, inputs[stream].tensor,
            ggml_mul(ctx.ggml, projected.tensor, modulation[stream][2].tensor)), inputs[stream].shape);
        const auto normalized = norm.build(ctx, residual, {});
        const auto modulated = core::wrap_tensor(ggml_add(ctx.ggml,
            ggml_mul(ctx.ggml, normalized.tensor, ggml_scale_bias(ctx.ggml, modulation[stream][4].tensor, 1.0F, 1.0F)),
            modulation[stream][3].tensor), normalized.shape);
        const auto packed = modules::LinearModule({1536, 6144, false, GGML_PREC_F32})
            .build(ctx, modulated, weights[stream].feed_forward_in);
        // AuK stores gate and up in one projection. The public feed-forward
        // module takes separate projections; retain the checkpoint's packing.
        const auto gated = core::wrap_tensor(ggml_swiglu(ctx.ggml, packed.tensor),
            packed.shape.with_last_dim(3072));
        const auto output = modules::LinearModule({3072, 1536, false, GGML_PREC_F32})
            .build(ctx, gated, weights[stream].feed_forward_out);
        if (boundaries) {
            const std::string branch = stream == 0 ? "x." : "c.";
            (*boundaries)[branch + "ff_input"] = modulated;
            (*boundaries)[branch + "ff_output"] = output;
        }
        result[stream] = core::wrap_tensor(ggml_add(ctx.ggml, residual.tensor,
            ggml_mul(ctx.ggml, output.tensor, modulation[stream][5].tensor)), residual.shape);
    }
    return result;
}

EmbeddingWeights load_embeddings(core::BackendWeightStore & store, const assets::TensorSource & source) {
    using namespace modules::binding;
    constexpr auto storage = assets::TensorStorageType::Native;
    EmbeddingWeights weights;
    std::vector<float> frequencies(128);
    for (size_t i = 0; i < frequencies.size(); ++i) {
        frequencies[i] = std::exp(static_cast<float>(i) * static_cast<float>(-std::log(10000.0) / 127.0));
    }
    weights.frequencies = store.make_f32(TensorShape::from_dims({128}), frequencies);
    weights.time_first = linear_from_source(store, source, "transformer.time_embed.time_mlp.0", storage, 1536, 256, true);
    weights.time_second = linear_from_source(store, source, "transformer.time_embed.time_mlp.2", storage, 1536, 1536, true);
    weights.text = linear_from_source(store, source, "transformer.txt_proj", storage, 1536, 2048, true);
    weights.text_norm = norm_weight_from_source(store, source, "transformer.txt_norm", 1536);
    weights.audio = linear_from_source(store, source, "transformer.audio_embed.linear", storage, 1536, 64, true);
    for (int layer = 0; layer < 2; ++layer) {
        const auto prefix = "transformer.audio_embed.conv_pos_embed.conv1d." + std::to_string(layer * 2);
        weights.position[layer].weight = store.load_tensor(source, prefix + ".weight", storage, {1536, 96, 31});
        weights.position[layer].bias = store.load_f32_tensor(source, prefix + ".bias", {1536});
    }
    return weights;
}

TensorValue build_audio_embedding(core::ModuleBuildContext & ctx, const EmbeddingWeights & weights,
    const TensorValue & audio, const std::optional<TensorValue> & mask = std::nullopt) {
    auto audio_embedding = modules::LinearModule({64, 1536, true, GGML_PREC_F32}).build(ctx, audio, weights.audio);
    auto position = modules::TransposeModule({{0, 2, 1}, 3}).build(ctx, audio_embedding);
    if (mask) position = core::wrap_tensor(ggml_mul(ctx.ggml, position.tensor, mask->tensor), position.shape);
    for (const auto & conv : weights.position) {
        const auto input = core::ensure_backend_addressable_layout(ctx, position);
        TensorValue grouped;
        for (int group = 0; group < 16; ++group) {
            const auto group_input = modules::SliceModule({1, group * 96, 96}).build(ctx, input);
            const auto weight = modules::SliceModule({0, group * 96, 96}).build(ctx, conv.weight);
            const auto bias = modules::SliceModule({0, group * 96, 96}).build(ctx, *conv.bias);
            const auto output = modules::Conv1dModule({96, 96, 31, 1, 15, 1, true}).build(ctx, group_input, {weight, bias});
            grouped = group == 0 ? output : modules::ConcatModule({1}).build(ctx, grouped, output);
        }
        if (mask) grouped = core::wrap_tensor(ggml_mul(ctx.ggml, grouped.tensor, mask->tensor), grouped.shape);
        // Mish has no framework module; use its primitive definition locally.
        position = core::wrap_tensor(ggml_mul(ctx.ggml, grouped.tensor,
            ggml_tanh(ctx.ggml, ggml_softplus(ctx.ggml, grouped.tensor))), grouped.shape);
    }
    position = modules::TransposeModule({{0, 2, 1}, 3}).build(ctx, position);
    audio_embedding = core::wrap_tensor(ggml_add(ctx.ggml, position.tensor, audio_embedding.tensor), audio_embedding.shape);
    return audio_embedding;
}

std::array<TensorValue, 3> build_embeddings(core::ModuleBuildContext & ctx, const EmbeddingWeights & weights,
    const TensorValue & audio, const TensorValue & text, const TensorValue & time) {
    const auto phase_shape = TensorShape::from_dims({time.shape.dims[0], 128});
    auto * frequencies = ggml_repeat(ctx.ggml, weights.frequencies.tensor,
        core::make_tensor(ctx, GGML_TYPE_F32, phase_shape).tensor);
    auto * phase = ggml_mul(ctx.ggml, frequencies, ggml_scale(ctx.ggml, time.tensor, 1000.0F));
    const auto sin = core::wrap_tensor(ggml_sin(ctx.ggml, phase), phase_shape);
    const auto cos = core::wrap_tensor(ggml_cos(ctx.ggml, phase), phase_shape);
    auto time_embedding = modules::ConcatModule({1}).build(ctx, sin, cos);
    time_embedding = modules::LinearModule({256, 1536, true, GGML_PREC_F32}).build(ctx, time_embedding, weights.time_first);
    time_embedding = modules::SiluModule().build(ctx, time_embedding);
    time_embedding = modules::LinearModule({1536, 1536, true, GGML_PREC_F32}).build(ctx, time_embedding, weights.time_second);
    auto text_embedding = modules::LinearModule({2048, 1536, true, GGML_PREC_F32}).build(ctx, text, weights.text);
    // PyTorch nn.RMSNorm's unspecified epsilon is finfo(input.dtype).eps.
    text_embedding = modules::RMSNormModule({1536, std::numeric_limits<float>::epsilon(), true, false})
        .build(ctx, text_embedding, weights.text_norm);
    return {time_embedding, text_embedding, build_audio_embedding(ctx, weights, audio)};
}

}  // namespace

struct FlowWeights {
    core::BackendWeightStore store;
    EmbeddingWeights weights;
    TensorValue rope_factors;
    std::vector<std::array<StreamWeights, 2>> double_blocks;
    std::vector<std::array<StreamWeights, 1>> single_blocks;
    modules::LinearWeights final_modulation;
    modules::LinearWeights final_projection;

    FlowWeights(core::ExecutionContext & execution, const assets::TensorSource & source)
        : store(execution.backend(), execution.backend_type(), "auk.flow", 1024 * 1024),
          weights(load_embeddings(store, source)) {
        // Preserve the checkpoint's rounded frequencies, not a regenerated formula.
        auto frequencies = source.require_f32("transformer.rotary_embed.inv_freq");
        if (frequencies.size() != 32) throw std::runtime_error("AuK rotary frequency size mismatch");
        for (auto & frequency : frequencies) frequency = 1.0F / frequency;
        rope_factors = store.make_f32(TensorShape::from_dims({32}), frequencies);
        double_blocks.reserve(10);
        for (int layer = 0; layer < 10; ++layer) {
            double_blocks.push_back(load_transformer_block<2>(store, source, "transformer.transformer_blocks." + std::to_string(layer)));
        }
        single_blocks.reserve(20);
        for (int layer = 0; layer < 20; ++layer) {
            single_blocks.push_back(load_transformer_block<1>(store, source, "transformer.single_transformer_blocks." + std::to_string(layer)));
        }
        final_modulation = modules::binding::linear_from_source(store, source, "transformer.norm_out.linear",
            assets::TensorStorageType::Native, 3072, 1536, true);
        final_projection = modules::binding::linear_from_source(store, source, "transformer.proj_out",
            assets::TensorStorageType::Native, 64, 1536, true);
        store.upload();
    }
};

struct FlowGraph {
    core::ExecutionContext & execution;
    int64_t frames;
    int64_t tokens;
    int64_t reference_frames;
    int64_t valid_reference_frames;
    bool cfg;
    bool capture_intermediates;
    std::unique_ptr<ggml_context, decltype(&ggml_free)> context{nullptr, ggml_free};
    std::unique_ptr<ggml_gallocr, decltype(&ggml_gallocr_free)> allocator{nullptr, ggml_gallocr_free};
    ggml_cgraph * graph = nullptr;
    ggml_cgraph * conditioning_graph = nullptr;
    ggml_cgraph * step_graph = nullptr;
    TensorValue audio;
    TensorValue reference;
    TensorValue text;
    TensorValue time;
    TensorValue step_index;
    int schedule_steps;
    TensorValue guidance_strength;
    std::array<TensorValue, 3> outputs;
    std::array<TensorValue, 2> block_outputs;
    std::array<TensorValue, 2> double_outputs;
    TensorValue single_output;
    TensorValue velocity;
    TensorValue guided_velocity;
    std::map<std::string, TensorValue> first_block_boundaries;

    FlowGraph(core::ExecutionContext & execution_, const FlowWeights & weight_set, int64_t frames_, int64_t tokens_, bool cfg_, bool flash_attention,
          bool capture_intermediates_, int64_t reference_frames_, int64_t valid_reference_frames_, int schedule_steps_)
        : execution(execution_), frames(frames_), tokens(tokens_), reference_frames(reference_frames_),
          valid_reference_frames(valid_reference_frames_), cfg(cfg_), capture_intermediates(capture_intermediates_),
          schedule_steps(schedule_steps_) {
        if (frames <= 0 || tokens <= 0) throw std::runtime_error("AuK flow lengths must be positive");
        const auto & weights = weight_set.weights;
        const auto & rope_factors = weight_set.rope_factors;
        const auto & double_blocks = weight_set.double_blocks;
        const auto & single_blocks = weight_set.single_blocks;
        const auto & final_modulation = weight_set.final_modulation;
        const auto & final_projection = weight_set.final_projection;
        constexpr size_t nodes = 32768;
        context.reset(ggml_init({nodes * ggml_tensor_overhead() + 3 * ggml_graph_overhead_custom(nodes, false), nullptr, true}));
        if (!context) throw std::runtime_error("AuK flow graph context allocation failed");
        core::ModuleBuildContext ctx{context.get(), "auk.flow", execution.backend_type()};
        audio = core::make_tensor(ctx, GGML_TYPE_F32, TensorShape::from_dims({1, frames, 64}));
        text = core::make_tensor(ctx, GGML_TYPE_F32, TensorShape::from_dims({1, tokens, 2048}));
        time = core::make_tensor(ctx, GGML_TYPE_F32, TensorShape::from_dims({schedule_steps, 1}));
        step_index = core::make_tensor(ctx, GGML_TYPE_I32, TensorShape::from_dims({1}));
        ggml_set_input(step_index.tensor);
        ggml_set_output(step_index.tensor);
        ggml_set_input(audio.tensor);
        ggml_set_input(text.tensor);
        // Conditioning persists across all steps; INPUT alone permits gallocr
        // to recycle its storage after the last consumer in one execution.
        ggml_set_output(text.tensor);
        ggml_set_input(time.tensor);
        outputs = build_embeddings(ctx, weights, audio, text, time);
        const auto scheduled_time = outputs[0];
        outputs[0] = core::wrap_tensor(ggml_get_rows(ctx.ggml, scheduled_time.tensor, step_index.tensor),
            TensorShape::from_dims({1, 1536}));
        const auto time_silu = modules::SiluModule().build(ctx, scheduled_time);
        std::vector<TensorValue> scheduled_parameters;
        const auto modulation = [&](const modules::LinearWeights & projection, int width) {
            const auto table = modules::LinearModule({1536, width, true, GGML_PREC_F32}).build(ctx, time_silu, projection);
            scheduled_parameters.push_back(table);
            auto row = core::wrap_tensor(ggml_get_rows(ctx.ggml, table.tensor, step_index.tensor),
                TensorShape::from_dims({1, width}));
            return cfg ? modules::ConcatModule({0}).build(ctx, row, row) : row;
        };
        auto transformer_text = outputs[1];
        auto transformer_audio = outputs[2];
        TensorValue cached_reference;
        std::optional<TensorValue> double_mask, single_mask, audio_output_mask, single_output_mask;
        std::vector<std::pair<TensorValue, std::vector<float>>> persistent_masks;
        if (reference_frames > 0) {
            reference = core::make_tensor(ctx, GGML_TYPE_F32, TensorShape::from_dims({1, reference_frames, 64}));
            ggml_set_input(reference.tensor);
            ggml_set_output(reference.tensor);
            auto reference_mask = core::make_tensor(ctx, GGML_TYPE_F32, TensorShape::from_dims({reference_frames}));
            std::vector<float> values(reference_frames, 0.0F);
            std::fill_n(values.begin(), valid_reference_frames, 1.0F);
            persistent_masks.emplace_back(reference_mask, values);
            auto reference_input = reference;
            if (cfg) {
                const auto zero_reference = core::wrap_tensor(ggml_scale(ctx.ggml, reference.tensor, 0.0F), reference.shape);
                reference_input = modules::ConcatModule({0}).build(ctx, reference, zero_reference);
            }
            const auto reference_embedding = build_audio_embedding(ctx, weights, reference_input, reference_mask);
            cached_reference = reference_embedding;
            if (cfg) transformer_audio = modules::ConcatModule({0}).build(ctx, transformer_audio, transformer_audio);
            transformer_audio = modules::ConcatModule({1}).build(ctx, reference_embedding, transformer_audio);
            const int64_t total_audio = reference_frames + frames;
            const int64_t total = total_audio + tokens;
            audio_output_mask = core::make_tensor(ctx, GGML_TYPE_F32, TensorShape::from_dims({1, total_audio, 1}));
            std::vector<float> audio_values(total_audio, 1.0F);
            std::fill(audio_values.begin() + valid_reference_frames, audio_values.begin() + reference_frames, 0.0F);
            persistent_masks.emplace_back(*audio_output_mask, audio_values);
            single_output_mask = core::make_tensor(ctx, GGML_TYPE_F32, TensorShape::from_dims({1, total, 1}));
            std::vector<float> single_values(tokens, 1.0F);
            single_values.insert(single_values.end(), audio_values.begin(), audio_values.end());
            persistent_masks.emplace_back(*single_output_mask, single_values);
            const auto attention_mask_type = flash_attention ? GGML_TYPE_F16 : GGML_TYPE_F32;
            double_mask = core::make_tensor(ctx, attention_mask_type, TensorShape::from_dims({total, total}));
            single_mask = core::make_tensor(ctx, attention_mask_type, TensorShape::from_dims({total, total}));
            std::vector<float> double_values(total * total, 0.0F), joint_values(total * total, 0.0F);
            for (int64_t query = 0; query < total; ++query) {
                for (int64_t key = valid_reference_frames; key < reference_frames; ++key) {
                    double_values[query * total + key] = -std::numeric_limits<float>::infinity();
                    joint_values[query * total + tokens + key] = -std::numeric_limits<float>::infinity();
                }
            }
            persistent_masks.emplace_back(*double_mask, std::move(double_values));
            persistent_masks.emplace_back(*single_mask, std::move(joint_values));
            for (const auto & mask : persistent_masks) {
                ggml_set_input(mask.first.tensor);
                ggml_set_output(mask.first.tensor);
            }
        }
        if (cfg) {
            if (reference_frames == 0) transformer_audio = modules::ConcatModule({0}).build(ctx, transformer_audio, transformer_audio);
            const auto unconditioned_text = core::wrap_tensor(ggml_scale(ctx.ggml, transformer_text.tensor, 0.0F), transformer_text.shape);
            transformer_text = modules::ConcatModule({0}).build(ctx, transformer_text, unconditioned_text);
        }
        const int64_t batch = cfg ? 2 : 1;
        const auto audio_positions = core::make_tensor(ctx, GGML_TYPE_I32, TensorShape::from_dims({frames + reference_frames}));
        const auto text_positions = core::make_tensor(ctx, GGML_TYPE_I32, TensorShape::from_dims({tokens}));
        const auto joint_positions = core::make_tensor(ctx, GGML_TYPE_I32, TensorShape::from_dims({tokens + frames + reference_frames}));
        ggml_set_input(audio_positions.tensor);
        ggml_set_input(text_positions.tensor);
        ggml_set_input(joint_positions.tensor);
        ggml_set_output(audio_positions.tensor);
        ggml_set_output(text_positions.tensor);
        ggml_set_output(joint_positions.tensor);
        double_outputs = {transformer_audio, transformer_text};
        for (size_t layer = 0; layer < double_blocks.size(); ++layer) {
            const std::array<TensorValue, 2> parameters = {
                modulation(double_blocks[layer][0].modulation, 9216),
                modulation(double_blocks[layer][1].modulation, 9216)};
            double_outputs = build_transformer_block<2>(ctx, double_blocks[layer], double_outputs,
                parameters, {audio_positions, text_positions}, rope_factors, flash_attention,
                capture_intermediates && layer == 0 ? &first_block_boundaries : nullptr,
                double_mask, {audio_output_mask, std::nullopt});
            if (layer == 0) block_outputs = double_outputs;
        }
        // The single-stream stage puts text first and restarts positions over
        // the combined sequence, unlike the double-stream joint attention.
        single_output = modules::ConcatModule({1}).build(ctx, double_outputs[1], double_outputs[0]);
        for (const auto & block : single_blocks) {
            single_output = build_transformer_block<1>(ctx, block, {single_output}, {modulation(block[0].modulation, 9216)}, {joint_positions},
                rope_factors, flash_attention, nullptr, single_mask, {single_output_mask})[0];
        }
        const auto target = modules::SliceModule({1, tokens + reference_frames, frames}).build(ctx, single_output);
        const auto parameters = modulation(final_modulation, 3072);
        const auto scale = core::reshape_tensor(ctx, core::ensure_backend_addressable_layout(ctx,
            modules::SliceModule({1, 0, 1536}).build(ctx, parameters)), TensorShape::from_dims({batch, 1, 1536}));
        const auto shift = core::reshape_tensor(ctx, core::ensure_backend_addressable_layout(ctx,
            modules::SliceModule({1, 1536, 1536}).build(ctx, parameters)), TensorShape::from_dims({batch, 1, 1536}));
        const auto normalized = modules::LayerNormModule({1536, 1e-6F, false, false}).build(ctx, target, {});
        const auto modulated = core::wrap_tensor(ggml_add(ctx.ggml,
            ggml_mul(ctx.ggml, normalized.tensor, ggml_scale_bias(ctx.ggml, scale.tensor, 1.0F, 1.0F)), shift.tensor), normalized.shape);
        velocity = modules::LinearModule({1536, 64, true, GGML_PREC_F32}).build(ctx, modulated, final_projection);
        guided_velocity = velocity;
        if (cfg) {
            guidance_strength = core::make_tensor(ctx, GGML_TYPE_F32, TensorShape::from_dims({1}));
            ggml_set_input(guidance_strength.tensor);
            ggml_set_output(guidance_strength.tensor);
            const auto conditional = modules::SliceModule({0, 0, 1}).build(ctx, velocity);
            const auto unconditional = modules::SliceModule({0, 1, 1}).build(ctx, velocity);
            // AuK uses cond + strength*(cond-uncond), not uncond + strength*(cond-uncond).
            guided_velocity = core::wrap_tensor(ggml_add(ctx.ggml, conditional.tensor,
                ggml_mul(ctx.ggml, ggml_sub(ctx.ggml, conditional.tensor, unconditional.tensor), guidance_strength.tensor)), conditional.shape);
        }
        graph = ggml_new_graph_custom(context.get(), nodes, false);
        ggml_set_output(outputs[1].tensor);
        if (reference_frames) ggml_set_output(cached_reference.tensor);
        ggml_build_forward_expand(graph, outputs[1].tensor);
        if (reference_frames) ggml_build_forward_expand(graph, cached_reference.tensor);
        ggml_set_output(scheduled_time.tensor);
        ggml_build_forward_expand(graph, scheduled_time.tensor);
        for (const auto & table : scheduled_parameters) {
            ggml_set_output(table.tensor);
            ggml_build_forward_expand(graph, table.tensor);
        }
        const auto * conditioning_end = ggml_graph_node(graph, -1);
        if (capture_intermediates) {
            std::vector<TensorValue> captures(outputs.begin(), outputs.end());
            captures.insert(captures.end(), block_outputs.begin(), block_outputs.end());
            captures.insert(captures.end(), double_outputs.begin(), double_outputs.end());
            captures.push_back(single_output);
            captures.push_back(velocity);
            for (const auto & entry : first_block_boundaries) captures.push_back(entry.second);
            for (const auto & output : captures) {
                ggml_set_output(output.tensor);
                if (output.tensor->view_src) ggml_set_output(output.tensor->view_src);
                ggml_build_forward_expand(graph, output.tensor);
            }
        }
        ggml_set_output(guided_velocity.tensor);
        if (guided_velocity.tensor->view_src) ggml_set_output(guided_velocity.tensor->view_src);
        ggml_build_forward_expand(graph, guided_velocity.tensor);
        auto optimization = runtime::graph_optimization_options_for_backend(execution.backend_type() == core::BackendType::Cpu
            ? runtime::GraphOptimizationBackend::Cpu : runtime::GraphOptimizationBackend::Gpu);
        // Identity folding updates src edges but not view_src. Keep copies
        // backing live views so gallocr can allocate the sliced conv weights.
        std::unordered_set<const ggml_tensor *> view_backings;
        for (int index = 0; index < ggml_graph_n_nodes(graph); ++index) {
            if (const auto * backing = ggml_graph_node(graph, index)->view_src) view_backings.insert(backing);
        }
        optimization.exclude_rewrite_user_data = &view_backings;
        optimization.exclude_rewrite = [](const runtime::GraphOptimizationRewriteContext & rewrite, const void * data) {
            if (rewrite.kind != runtime::GraphOptimizationRewriteKind::IdentityMaterialization) return false;
            return static_cast<const std::unordered_set<const ggml_tensor *> *>(data)->count(rewrite.node) != 0;
        };
        runtime::optimize_graph(*graph, optimization);
        int prefix = 0;
        while (prefix < ggml_graph_n_nodes(graph) && ggml_graph_node(graph, prefix) != conditioning_end) ++prefix;
        if (prefix == ggml_graph_n_nodes(graph)) throw std::runtime_error("AuK conditioning graph boundary was removed");
        ++prefix;
        conditioning_graph = ggml_new_graph_custom(context.get(), nodes, false);
        step_graph = ggml_new_graph_custom(context.get(), nodes, false);
        std::unordered_set<const ggml_tensor *> fixed_nodes;
        for (int i = 0; i < prefix; ++i) {
            auto * node = ggml_graph_node(graph, i);
            fixed_nodes.insert(node);
            ggml_graph_add_node(conditioning_graph, node);
        }
        // Retain values crossing the graph boundary, including storage backing views.
        for (int i = prefix; i < ggml_graph_n_nodes(graph); ++i) {
            auto * node = ggml_graph_node(graph, i);
            ggml_graph_add_node(step_graph, node);
            for (auto * source : node->src) {
                if (fixed_nodes.count(source)) {
                    for (auto * storage = source; storage; storage = storage->view_src) ggml_set_output(storage);
                }
            }
        }
        allocator.reset(ggml_gallocr_new(ggml_backend_get_default_buffer_type(execution.backend())));
        if (!allocator || !ggml_gallocr_alloc_graph(allocator.get(), graph)) {
            throw std::runtime_error("AuK flow graph buffer allocation failed");
        }
        std::vector<int32_t> audio_ids(frames + reference_frames), text_ids(tokens), joint_ids(tokens + frames + reference_frames);
        std::iota(audio_ids.begin(), audio_ids.end(), 0);
        std::iota(text_ids.begin(), text_ids.end(), 0);
        std::iota(joint_ids.begin(), joint_ids.end(), 0);
        core::write_tensor_i32(audio_positions, audio_ids);
        core::write_tensor_i32(text_positions, text_ids);
        core::write_tensor_i32(joint_positions, joint_ids);
        for (const auto & mask : persistent_masks) core::write_tensor_float(mask.first, mask.second);
    }

    void compute(const std::vector<float> & latent, int step) {
        if (latent.size() != static_cast<size_t>(frames * 64)) {
            throw std::runtime_error("AuK flow latent length differs from prepared graph");
        }
        core::write_tensor_f32(audio, latent);
        core::write_tensor_i32(step_index, {step});
        if (core::compute_backend_graph(execution.backend(), step_graph, nullptr, "auk.flow") != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("AuK flow graph computation failed");
        }
    }

    void prepare_conditioning() {
        if (core::compute_backend_graph(execution.backend(), conditioning_graph, nullptr, "auk.flow.conditioning") != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("AuK flow conditioning graph computation failed");
        }
    }

    ~FlowGraph() {
        core::release_backend_graph_resources(execution.backend(), step_graph, true);
        core::release_backend_graph_resources(execution.backend(), conditioning_graph, true);
        core::release_backend_graph_resources(execution.backend(), graph, true);
    }
};

struct FlowRuntime::State {
    core::ExecutionContext & execution;
    bool cfg;
    bool flash_attention;
    bool capture_intermediates;
    FlowWeights weights;
    std::unique_ptr<FlowGraph> prepared;

    State(core::ExecutionContext & execution_, const assets::TensorSource & source,
          bool cfg_, bool flash_attention_, bool capture_intermediates_)
        : execution(execution_), cfg(cfg_), flash_attention(flash_attention_),
          capture_intermediates(capture_intermediates_), weights(execution, source) {}
};

FlowRuntime::FlowRuntime(core::ExecutionContext & execution, const assets::TensorSource & source,
                         int64_t audio_frames, int64_t text_tokens, bool cfg, bool flash_attention, bool capture_intermediates,
                         int64_t reference_frames, int64_t valid_reference_frames, int schedule_steps)
    : state_(std::make_unique<State>(execution, source, cfg, flash_attention, capture_intermediates)) {
    prepare(audio_frames, text_tokens, cfg, reference_frames, valid_reference_frames, schedule_steps);
}
FlowRuntime::~FlowRuntime() = default;

void FlowRuntime::prepare(int64_t audio_frames, int64_t text_tokens) {
    prepare(audio_frames, text_tokens, state_->cfg);
}

void FlowRuntime::prepare(int64_t audio_frames, int64_t text_tokens, bool cfg,
                          int64_t reference_frames, int64_t valid_reference_frames, int schedule_steps) {
    if (schedule_steps <= 0) throw std::runtime_error("AuK schedule must contain at least one step");
    if (audio_frames <= 0 || text_tokens <= 0) throw std::runtime_error("AuK flow lengths must be positive");
    auto & state = *state_;
    if (reference_frames < 0 || valid_reference_frames < 0 || valid_reference_frames > reference_frames) {
        throw std::runtime_error("AuK reference length is invalid");
    }
    if (state.prepared && state.prepared->frames == audio_frames && state.prepared->tokens == text_tokens && state.prepared->cfg == cfg &&
        state.prepared->reference_frames == reference_frames && state.prepared->valid_reference_frames == valid_reference_frames &&
        state.prepared->schedule_steps == schedule_steps) return;
    state.prepared.reset();
    state.prepared = std::make_unique<FlowGraph>(state.execution, state.weights, audio_frames, text_tokens,
        cfg, state.flash_attention, state.capture_intermediates, reference_frames, valid_reference_frames, schedule_steps);
    state.cfg = cfg;
}

FlowEmbeddings FlowRuntime::embed(const std::vector<float> & audio, const std::vector<float> & text, float time,
                                  float cfg_strength, const std::vector<float> & reference) {
    if (!state_->prepared) throw std::runtime_error("AuK flow graph is not prepared");
    auto & state = *state_->prepared;
    if (!state.capture_intermediates) throw std::runtime_error("AuK intermediate capture was not enabled");
    if (audio.size() != static_cast<size_t>(state.frames * 64) || text.size() != static_cast<size_t>(state.tokens * 2048)) {
        throw std::runtime_error("AuK flow input lengths differ from prepared graph");
    }
    core::write_tensor_f32(state.text, text);
    if (reference.size() != static_cast<size_t>(state.reference_frames * 64)) throw std::runtime_error("AuK flow reference size differs");
    if (state.reference_frames) core::write_tensor_f32(state.reference, reference);
    if (state.cfg) core::write_tensor_f32(state.guidance_strength, {cfg_strength});
    core::write_tensor_f32(state.time, std::vector<float>(state.schedule_steps, time));
    state.prepare_conditioning();
    state.compute(audio, 0);
    FlowEmbeddings result{core::read_tensor_f32(state.outputs[0].tensor), core::read_tensor_f32(state.outputs[1].tensor),
        core::read_tensor_f32(state.outputs[2].tensor), core::read_tensor_f32(state.block_outputs[0].tensor),
        core::read_tensor_f32(state.block_outputs[1].tensor), core::read_tensor_f32(state.double_outputs[0].tensor),
        core::read_tensor_f32(state.double_outputs[1].tensor), core::read_tensor_f32(state.single_output.tensor),
        core::read_tensor_f32(state.velocity.tensor), core::read_tensor_f32(state.guided_velocity.tensor), {}};
    for (const auto & entry : state.first_block_boundaries) {
        result.first_block_boundaries.emplace(entry.first, core::read_tensor_f32(entry.second.tensor));
    }
    debug::trace_log_f32("auk.flow.time", {1, 1536}, result.time);
    debug::trace_log_f32("auk.flow.text", {1, state.tokens, 1536}, result.text);
    debug::trace_log_f32("auk.flow.audio", {1, state.frames, 1536}, result.audio);
    return result;
}

std::vector<float> FlowRuntime::sample(const std::vector<float> & initial_noise,
    const std::vector<float> & text, int steps, float sway, float cfg_strength,
    const std::vector<float> & reference, bool distilled_flash) {
    if (!state_->prepared) throw std::runtime_error("AuK flow graph is not prepared");
    if (steps <= 0 || !std::isfinite(sway) || !std::isfinite(cfg_strength)) {
        throw std::runtime_error("invalid AuK sampling configuration");
    }
    const auto & previous = *state_->prepared;
    prepare(previous.frames, previous.tokens, previous.cfg, previous.reference_frames, previous.valid_reference_frames, steps);
    auto & state = *state_->prepared;
    if (state.cfg != (cfg_strength >= 1e-5F)) {
        throw std::runtime_error("AuK guidance mode differs from prepared graph");
    }
    if (text.size() != static_cast<size_t>(state.tokens * 2048)) {
        throw std::runtime_error("AuK text length differs from prepared graph");
    }
    core::write_tensor_f32(state.text, text);
    if (reference.size() != static_cast<size_t>(state.reference_frames * 64)) throw std::runtime_error("AuK flow reference size differs");
    if (state.reference_frames) core::write_tensor_f32(state.reference, reference);
    if (state.cfg) core::write_tensor_f32(state.guidance_strength, {cfg_strength});
    std::vector<float> times;
    if (distilled_flash) {
        times = {0.0F, 0.07612049579620361F, 0.2928932309150696F, 0.6173166036605835F, 1.0F};
    } else {
        times.resize(steps + 1);
        for (int i = 0; i <= steps; ++i) {
            const float t = static_cast<float>(i) / steps;
            times[i] = t + sway * (std::cos(1.5707963267948966F * t) - 1.0F + t);
        }
    }
    core::write_tensor_f32(state.time, std::vector<float>(times.begin(), times.end() - 1));
    state.prepare_conditioning();
    modules::FlowSamplerRuntimeConfig config;
    config.label = "auk.flow";
    config.latent_shape = {1, state.frames, 64};
    config.initial_latent = initial_noise;
    config.branches = {{"velocity", 1.0F}};
    for (int i = 0; i < steps; ++i) {
        config.schedule.push_back({i, times[i], times[i + 1], times[i], times[i + 1]});
    }
    modules::FlowSamplerSingleBranchDenoiserConfig denoiser;
    denoiser.label = "auk.flow";
    denoiser.sampler_mode = [](const modules::FlowSamplerStepState &) { return "euler"; };
    denoiser.predict = [&state](const modules::FlowSamplerDenoiserInput & input) {
        state.compute(input.latent, static_cast<int>(input.state.schedule.index));
        return core::read_tensor_f32(state.guided_velocity.tensor);
    };
    modules::FlowSamplerRuntime sampler(std::move(config), modules::make_flow_sampler_single_branch_denoiser(std::move(denoiser)));
    sampler.run_sequence();
    return sampler.latent();
}

}  // namespace engine::models::auk
