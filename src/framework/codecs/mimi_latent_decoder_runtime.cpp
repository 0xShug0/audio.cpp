#include "engine/framework/codecs/mimi_codec_runtime.h"

#include "engine/framework/core/prepared_graph.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/modules/activation_modules.h"
#include "engine/framework/modules/primitive_modules.h"
#include "engine/framework/modules/structural_modules.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <numeric>
#include <functional>
#include <stdexcept>
#include <utility>

namespace engine::codecs {
namespace {
using core::TensorValue;
using core::TensorShape;

// Representation binding is local to the owning graph metadata. Keep three
// independent projections, even when the package stores one packed QKV matrix.
modules::TransformerEncoderBlockWeights bind_layer(core::ModuleBuildContext & ctx,
    modules::TransformerEncoderBlockWeights w, int64_t hidden) {
    auto & a = w.self_attention;
    if (!a.q_weight.valid() || !a.k_weight.valid() || !a.v_weight.valid()) {
        if (!a.qkv_weight) throw std::invalid_argument("Mimi attention requires Q/K/V or packed QKV");
        core::validate_shape(*a.qkv_weight, TensorShape::from_dims({3 * hidden, hidden}), "Mimi QKV");
        auto * packed = a.qkv_weight->tensor;
        const size_t row = packed->nb[1];
        const auto shape = TensorShape::from_dims({hidden, hidden});
        if (!a.q_weight.valid()) a.q_weight = core::wrap_tensor(
            ggml_view_2d(ctx.ggml, packed, hidden, hidden, row, 0), shape, packed->type);
        if (!a.k_weight.valid()) a.k_weight = core::wrap_tensor(
            ggml_view_2d(ctx.ggml, packed, hidden, hidden, row, hidden * row), shape, packed->type);
        if (!a.v_weight.valid()) a.v_weight = core::wrap_tensor(
            ggml_view_2d(ctx.ggml, packed, hidden, hidden, row, 2 * hidden * row), shape, packed->type);
    }
    return w;
}

// Full sequence keeps uncropped transposed-convolution tails between stages,
// exactly as the established single graph; only the final PCM is cropped.
TensorValue build_full_tail(core::ModuleBuildContext & ctx, TensorValue x,
    const MimiLatentDecoderConfig & config, const SEANetDecoderWeights & w, int64_t samples) {
    const auto conv = [&](TensorValue input, const modules::Conv1dWeights & weights) {
        const auto & d = weights.weight.shape.dims;
        return modules::StreamingConv1dModule({d[1], d[0], d[2], 1, 1,
            weights.bias.has_value(), modules::StreamingPadMode::Constant}).build(ctx, input, weights);
    };
    x = conv(x, w.input_projection);
    for (size_t i = 0; i < w.stages.size(); ++i) {
        const auto & stage = w.stages[i];
        const auto & d = stage.upsample.weight.shape.dims;
        x = modules::EluModule().build(ctx, x);
        x = modules::ConvTranspose1dModule({d[0], d[1], d[2],
            static_cast<int>(config.stage_strides[i]), 0, 1, stage.upsample.bias.has_value()})
            .build(ctx, x, stage.upsample);
        for (const auto & block : stage.residual_blocks) {
            auto residual = conv(modules::EluModule().build(ctx, x), block.conv1);
            residual = conv(modules::EluModule().build(ctx, residual), block.conv2);
            x = modules::ResidualAddModule().build(ctx, residual, x);
        }
    }
    x = conv(modules::EluModule().build(ctx, x), w.output_projection);
    return modules::SliceModule({2, 0, samples}).build(ctx, x);
}

// Default blocks execute the complete decoder in one graph. Host full-sequence
// decoding uses a front graph and two bounded tail stages. All streaming history
// and KV live in persistent backend storage; only final PCM is read back.
enum class DecoderGraphMode { Full, Block, Front, TailPrefix, TailFinal };

struct DecoderGraph {
    core::PreparedGraph graph;
    TensorValue input, output, positions, mask;
    std::vector<TensorValue> keys, values;
    int64_t frames, prefix, context, hidden, end = 0, kept = 0, mask_kept = -1, host_mask_kept = -1;
    std::vector<int32_t> position_values;
    std::vector<float> mask_values;
    bool full_sequence, transformer_stage;
    std::vector<std::pair<TensorValue, TensorValue>> state_updates;
    DecoderGraph(const core::ExecutionContext & execution, const MimiLatentDecoderConfig & c,
        const MimiLatentDecoderWeights & w, size_t arena, const std::string & label, int64_t n,
        DecoderGraphMode mode)
        : graph(execution, {label, arena, 32768,
              mode == DecoderGraphMode::Full ? 0 :
                  (8 + static_cast<size_t>(2 * c.transformer_layers) + w.seanet.stages.size() +
                   std::accumulate(w.seanet.stages.begin(), w.seanet.stages.end(), size_t{0},
                       [](size_t total, const auto & stage) { return total + 2 * stage.residual_blocks.size(); })) *
                      ggml_tensor_overhead()}), frames(n),
          prefix(std::max<int64_t>(0, c.context - (mode == DecoderGraphMode::Full ? n : 1))),
          context(c.context), hidden(c.hidden_size), full_sequence(mode == DecoderGraphMode::Full),
          transformer_stage(mode != DecoderGraphMode::TailPrefix && mode != DecoderGraphMode::TailFinal) {
        if (transformer_stage) {
            position_values.resize(static_cast<size_t>(n));
            mask_values.resize(static_cast<size_t>(n * (prefix + n)));
        }
        auto & ctx = graph.build_context();
        const bool latent_input = transformer_stage;
        const auto input_channels = mode == DecoderGraphMode::TailFinal
            ? w.seanet.stages.back().upsample.weight.shape.dims[0] : hidden;
        input = core::make_tensor(ctx, GGML_TYPE_F32, latent_input
            ? TensorShape::from_dims({1, c.latent_size, n / c.encoder_upsample_stride})
            : TensorShape::from_dims({1, input_channels, n}));
        ggml_set_input(input.tensor);
        auto x = input;
        if (latent_input) {
            x = modules::Conv1dModule({c.latent_size, hidden, 1, 1, 0, 1, false})
                .build(ctx, x, {w.quantizer_output_proj_weight, std::nullopt});
            if (full_sequence) {
                x = modules::ConvTranspose1dModule({hidden, hidden, c.encoder_upsample_stride * 2,
                    static_cast<int>(c.encoder_upsample_stride), 0, 1, false})
                    .build(ctx, x, {w.encoder_upsample_weight, std::nullopt});
                x = modules::SliceModule({2, 0, n}).build(ctx, x);
            } else {
                x = upsample(ctx, x, {w.encoder_upsample_weight, std::nullopt},
                    static_cast<int>(c.encoder_upsample_stride), true, execution.uses_host_graph_plan());
            }
        }
        if (transformer_stage) output = build(ctx, x, c, w, !full_sequence);
        else output = x;
        if (mode == DecoderGraphMode::Block)
            output = build_tail(ctx, output, c, w.seanet, 0, w.seanet.stages.size(), true, true);
        else if (mode == DecoderGraphMode::TailPrefix)
            output = build_tail(ctx, output, c, w.seanet, 0, w.seanet.stages.size() - 1, true, false);
        else if (mode == DecoderGraphMode::TailFinal)
            output = build_tail(ctx, output, c, w.seanet, w.seanet.stages.size() - 1, w.seanet.stages.size(), false, true);
        if (full_sequence) {
            int64_t samples = n;
            for (auto stride : c.stage_strides) samples *= stride;
            output = build_full_tail(ctx, output, c, w.seanet, samples);
        }
        ggml_set_output(output.tensor);
        for (auto * root = output.tensor->view_src; root; root = root->view_src) ggml_set_output(root);
        ggml_build_forward_expand(graph.graph(), output.tensor);
        // Copies are appended after the entire transformer so no earlier layer
        // can overwrite a prefix still consumed by the current graph.
        for (size_t i = 0; i < next_keys.size(); ++i) {
            ggml_build_forward_expand(graph.graph(), ggml_cpy(ctx.ggml, next_keys[i].tensor, keys[i].tensor));
            ggml_build_forward_expand(graph.graph(), ggml_cpy(ctx.ggml, next_values[i].tensor, values[i].tensor));
        }
        for (const auto & update : state_updates)
            ggml_build_forward_expand(graph.graph(), ggml_cpy(ctx.ggml, update.second.tensor, update.first.tensor));
        allocate();
    }
    TensorValue history(int64_t channels, int64_t length) {
        auto tensor = core::make_tensor(graph.persistent_build_context(), GGML_TYPE_F32,
            TensorShape::from_dims({1, channels, length}));
        ggml_set_input(tensor.tensor);
        return tensor;
    }
    TensorValue convolve(core::ModuleBuildContext & ctx, TensorValue x,
        const modules::Conv1dWeights & w, bool elu) {
        const auto & d = w.weight.shape.dims;
        const int64_t frames = x.shape.dims[2];
        if (d[2] > 1) {
            auto previous = history(d[1], d[2] - 1);
            x = modules::ConcatModule({2}).build(ctx, previous, x);
            state_updates.emplace_back(previous,
                modules::SliceModule({2, frames, d[2] - 1}).build(ctx, x));
        }
        if (elu) x = modules::EluModule().build(ctx, x);
        return modules::Conv1dModule({d[1], d[0], d[2], 1, 0, 1, w.bias.has_value()}).build(ctx, x, w);
    }
    TensorValue upsample(core::ModuleBuildContext & ctx, const TensorValue & x,
        const modules::ConvTranspose1dWeights & w, int stride, bool depthwise = false, bool host = false) {
        const auto & shape = w.weight.shape;
        const int64_t ci = shape.dims[0], co = depthwise ? ci : shape.dims[1];
        const int64_t kernel = shape.dims[shape.rank - 1];
        const int64_t count = x.shape.dims[2] * stride;
        TensorValue raw;
        if (depthwise && host) {
            raw = modules::DepthwiseConvTranspose1dModule({ci, kernel,
                x.shape.dims[2] == 1 ? 1 : stride, false}).build(ctx, x, {w.weight, std::nullopt});
        } else {
            raw = modules::ConvTranspose1dModule({ci, co, kernel, stride, 0, 1, w.bias.has_value()}).build(ctx, x, w);
        }
        const int64_t overlap = kernel - stride;
        if (overlap > 0) {
            auto previous = history(co, overlap);
            auto prefix = modules::AddModule().build(ctx,
                modules::SliceModule({2, 0, overlap}).build(ctx, raw), previous);
            raw = modules::ConcatModule({2}).build(ctx, prefix,
                modules::SliceModule({2, overlap, raw.shape.dims[2] - overlap}).build(ctx, raw));
            auto next = modules::SliceModule({2, count, overlap}).build(ctx, raw);
            if (w.bias) {
                auto bias = *w.bias;
                if (bias.type != GGML_TYPE_F32)
                    bias = core::wrap_tensor(ggml_cast(ctx.ggml, bias.tensor, GGML_TYPE_F32), bias.shape, GGML_TYPE_F32);
                bias = core::reshape_tensor(ctx, bias, TensorShape::from_dims({1, co, 1}));
                next = core::wrap_tensor(ggml_sub(ctx.ggml, next.tensor, bias.tensor), next.shape, next.type);
            }
            state_updates.emplace_back(previous, next);
        }
        return modules::SliceModule({2, 0, count}).build(ctx, raw);
    }
    TensorValue build_tail(core::ModuleBuildContext & ctx, TensorValue x,
        const MimiLatentDecoderConfig & c, const SEANetDecoderWeights & w,
        size_t begin, size_t end, bool project_input, bool project_output) {
        if (project_input) x = convolve(ctx, x, w.input_projection, false);
        for (size_t i = begin; i < end; ++i) {
            x = upsample(ctx, modules::EluModule().build(ctx, x), w.stages[i].upsample,
                static_cast<int>(c.stage_strides[i]));
            for (const auto & block : w.stages[i].residual_blocks) {
                auto residual = convolve(ctx, x, block.conv1, true);
                residual = convolve(ctx, residual, block.conv2, true);
                x = modules::AddModule().build(ctx, x, residual);
            }
        }
        return project_output ? convolve(ctx, x, w.output_projection, true) : x;
    }
    std::vector<TensorValue> next_keys, next_values;
    TensorValue build(core::ModuleBuildContext & ctx, const TensorValue & x,
        const MimiLatentDecoderConfig & c, const MimiLatentDecoderWeights & w, bool update) {
        auto & inputs = full_sequence ? ctx : graph.persistent_build_context();
        positions = core::make_tensor(inputs, GGML_TYPE_I32, TensorShape::from_dims({frames}));
        mask = core::make_tensor(inputs, GGML_TYPE_F32, TensorShape::from_dims({1, 1, frames, prefix + frames}));
        ggml_set_input(positions.tensor);
        ggml_set_input(mask.tensor);
        if (full_sequence) {
            ggml_set_output(positions.tensor);
            ggml_set_output(mask.tensor);
        }
        auto y = modules::TransposeModule({{0, 2, 1, 3}, 3}).build(ctx, x);
        for (const auto & layer : w.transformer_layers) {
            std::optional<TensorValue> k, v;
            if (prefix) {
                auto & persistent = full_sequence ? ctx : graph.persistent_build_context();
                const auto shape = TensorShape::from_dims({1, prefix, c.num_heads, hidden / c.num_heads});
                k = core::make_tensor(persistent, GGML_TYPE_F32, shape);
                v = core::make_tensor(persistent, GGML_TYPE_F32, shape);
                keys.push_back(*k); values.push_back(*v);
                if (full_sequence) {
                    ggml_set_output(k->tensor);
                    ggml_set_output(v->tensor);
                }
            }
            modules::TransformerEncoderBlockConfig tc{hidden, c.num_heads, c.intermediate_size, 1.e-5F, false};
            tc.gelu_approximation = c.gelu_approximation;
            auto result = modules::StreamingTransformerEncoderBlockModule(tc).build(
                ctx, y, positions, bind_layer(ctx, layer, hidden), k, v, mask);
            y = result.output;
            if (prefix && update) {
                next_keys.push_back(modules::SliceModule({1, frames, prefix}).build(ctx,
                    modules::ConcatModule({1}).build(ctx, *k, result.key)));
                next_values.push_back(modules::SliceModule({1, frames, prefix}).build(ctx,
                    modules::ConcatModule({1}).build(ctx, *v, result.value)));
            }
        }
        return modules::TransposeModule({{0, 2, 1, 3}, 3}).build(ctx, y);
    }
    void allocate() {
        if (!full_sequence) graph.allocate_persistent();
        graph.allocate();
        mask_kept = -1;
        if (full_sequence) {
            for (const auto * tensors : {&keys, &values}) {
                for (const auto & tensor : *tensors) ggml_backend_tensor_memset(tensor.tensor, 0, 0, ggml_nbytes(tensor.tensor));
            }
            prepare_inputs();
        } else graph.clear_persistent();
    }
    void reset() {
        end = kept = 0;
        if (!full_sequence) {
            mask_kept = -1;
            if (graph.allocated()) graph.clear_persistent();
        }
    }
    void prepare_inputs() {
        if (end > std::numeric_limits<int32_t>::max() - frames)
            throw std::overflow_error("Mimi RoPE position exceeds I32 range");
        for (int64_t i = 0; i < frames; ++i) position_values[static_cast<size_t>(i)] = static_cast<int32_t>(end + i);
        core::write_tensor_i32(positions, position_values);
        if (mask_kept == kept) return;
        if (host_mask_kept != kept) {
            std::fill(mask_values.begin(), mask_values.end(), -INFINITY);
            for (int64_t q = 0; q < frames; ++q) {
                const auto begin = std::max(prefix - kept, std::max<int64_t>(0, prefix + q + 1 - context));
                for (int64_t k = begin; k <= prefix + q; ++k)
                    mask_values[static_cast<size_t>(q * (prefix + frames) + k)] = 0.0F;
            }
            host_mask_kept = kept;
        }
        core::write_tensor_f32(mask, mask_values);
        mask_kept = kept;
    }
    std::vector<float> run(const std::vector<float> & x) {
        core::write_tensor_f32(input, x);
        if (!full_sequence && transformer_stage) prepare_inputs();
        graph.compute();
        if (!full_sequence && transformer_stage) { end += frames; kept = std::min(prefix, kept + frames); }
        return core::read_tensor_f32(output.tensor);
    }
};

} // namespace

struct MimiLatentDecoderRuntime::Impl {
    const core::ExecutionContext & execution;
    MimiLatentDecoderConfig config;
    MimiLatentDecoderWeights weights;
    MimiLatentDecoderExecutionConfig settings;
    std::unique_ptr<DecoderGraph> full_graph;
    std::unique_ptr<DecoderGraph> offline_block_graph, streaming_block_graph;
    int64_t full_frames = 0;
    std::unique_ptr<DecoderGraph> full_front, full_tail_prefix, full_tail_final;
    int64_t full_front_frames = 0;
    int64_t stream_block = 0;

    Impl(const core::ExecutionContext & e, MimiLatentDecoderConfig c,
        MimiLatentDecoderWeights w, MimiLatentDecoderExecutionConfig s)
        : execution(e), config(std::move(c)), weights(std::move(w)), settings(std::move(s)) {
        if (config.latent_size <= 0 || config.hidden_size <= 0 || config.num_heads <= 0 ||
            config.hidden_size % config.num_heads || config.intermediate_size <= 0 ||
            config.transformer_layers <= 0 || config.encoder_upsample_stride <= 0 || config.context <= 0)
            throw std::invalid_argument("Invalid Mimi latent decoder architecture");
        if (weights.transformer_layers.size() != static_cast<size_t>(config.transformer_layers) ||
            weights.seanet.stages.size() != config.stage_strides.size() || config.stage_strides.empty())
            throw std::invalid_argument("Mimi decoder weights do not match architecture");
        if (!weights.store || settings.log_prefix.empty() || !settings.conv_graph_context_bytes ||
            !settings.transformer_graph_context_bytes || !settings.tail_graph_context_bytes ||
            settings.offline_block_frames <= 0 || settings.tail_chunk_frames <= 0 || settings.final_stage_chunk_frames <= 0)
            throw std::invalid_argument("Invalid Mimi latent decoder execution configuration");
        core::validate_shape(weights.quantizer_output_proj_weight,
            TensorShape::from_dims({config.hidden_size, config.latent_size, 1}), "Mimi continuous projection");
        core::validate_shape(weights.encoder_upsample_weight,
            execution.uses_host_graph_plan()
                ? TensorShape::from_dims({config.hidden_size, 1, 1, config.encoder_upsample_stride * 2})
                : TensorShape::from_dims({config.hidden_size, config.hidden_size, config.encoder_upsample_stride * 2}),
            "Mimi encoder upsample");
        const auto validate_conv = [](const modules::Conv1dWeights & cw, int64_t channels) {
            if (!cw.weight.valid() || cw.weight.shape.rank != 3 || cw.weight.shape.dims[1] != channels ||
                cw.weight.shape.dims[0] <= 0 || cw.weight.shape.dims[2] <= 0)
                throw std::invalid_argument("Invalid Mimi causal convolution geometry");
            return cw.weight.shape.dims[0];
        };
        int64_t channels = validate_conv(weights.seanet.input_projection, config.hidden_size);
        for (size_t i = 0; i < weights.seanet.stages.size(); ++i) {
            const auto & stage = weights.seanet.stages[i];
            const auto & up = stage.upsample.weight;
            if (!up.valid() || up.shape.rank != 3 || up.shape.dims[0] != channels ||
                up.shape.dims[1] <= 0 || config.stage_strides[i] <= 0 ||
                up.shape.dims[2] < config.stage_strides[i] ||
                (stage.upsample.bias && stage.upsample.bias->shape.num_elements() != up.shape.dims[1]))
                throw std::invalid_argument("Invalid Mimi stage upsample geometry or bias");
            channels = up.shape.dims[1];
            for (const auto & block : stage.residual_blocks) {
                auto inner = validate_conv(block.conv1, channels);
                if (validate_conv(block.conv2, inner) != channels)
                    throw std::invalid_argument("Invalid Mimi residual channel geometry");
            }
        }
        if (validate_conv(weights.seanet.output_projection, channels) != 1)
            throw std::invalid_argument("Mimi latent decoder requires mono output");
    }

    std::vector<float> pack(const std::vector<float> & x, int64_t frames) const {
        if (frames <= 0 || static_cast<uint64_t>(frames) > std::numeric_limits<size_t>::max() /
                static_cast<uint64_t>(config.latent_size) ||
            x.size() != static_cast<size_t>(frames) * static_cast<size_t>(config.latent_size))
            throw std::invalid_argument("Mimi decoder expects complete frame-major latents");
        std::vector<float> result(x.size());
        for (int64_t t = 0; t < frames; ++t)
            for (int64_t c = 0; c < config.latent_size; ++c)
                result[static_cast<size_t>(c * frames + t)] = x[static_cast<size_t>(t * config.latent_size + c)];
        return result;
    }

    std::vector<float> decode(const std::vector<float> & latents, int64_t frames, bool full) {
        auto packed = pack(latents, frames);
        if (full && !execution.uses_host_graph_plan()) {
            // Offline full shape has one retained entry, independent of streams.
            if (!full_graph || full_frames != frames) {
                full_graph.reset();
                full_graph = std::make_unique<DecoderGraph>(execution, config, weights,
                    std::max({settings.conv_graph_context_bytes, settings.transformer_graph_context_bytes, settings.tail_graph_context_bytes}),
                    settings.log_prefix + ".full", frames * config.encoder_upsample_stride, DecoderGraphMode::Full);
                full_frames = frames;
            }
            full_graph->reset();
            return full_graph->run(packed);
        }
        if (full) {
            const int64_t encoder_frames = frames * config.encoder_upsample_stride;
            const auto arena = std::max({settings.conv_graph_context_bytes,
                settings.transformer_graph_context_bytes, settings.tail_graph_context_bytes});
            if (!full_front || full_front_frames != encoder_frames) {
                full_front.reset();
                full_front = std::make_unique<DecoderGraph>(execution, config, weights, arena,
                    settings.log_prefix + ".full_front", encoder_frames, DecoderGraphMode::Front);
                full_front_frames = encoder_frames;
            }
            if (!full_tail_prefix) full_tail_prefix = std::make_unique<DecoderGraph>(execution, config, weights, arena,
                settings.log_prefix + ".tail_prefix", settings.tail_chunk_frames, DecoderGraphMode::TailPrefix);
            if (!full_tail_final) full_tail_final = std::make_unique<DecoderGraph>(execution, config, weights, arena,
                settings.log_prefix + ".tail_final", settings.final_stage_chunk_frames, DecoderGraphMode::TailFinal);
            full_front->reset(); full_tail_prefix->reset(); full_tail_final->reset();
            const auto transformed = full_front->run(packed);
            std::vector<float> pcm;
            const int64_t channels = weights.seanet.stages.back().upsample.weight.shape.dims[0];
            const int64_t prefix_rate = std::accumulate(config.stage_strides.begin(), config.stage_strides.end() - 1,
                int64_t{1}, std::multiplies<int64_t>{});
            std::vector<float> tail(static_cast<size_t>(channels * settings.final_stage_chunk_frames));
            int64_t pending = 0;
            for (int64_t start = 0; start < encoder_frames; start += settings.tail_chunk_frames) {
                const int64_t count = std::min(settings.tail_chunk_frames, encoder_frames - start);
                std::vector<float> input(static_cast<size_t>(config.hidden_size * settings.tail_chunk_frames), 0.0F);
                for (int64_t c = 0; c < config.hidden_size; ++c)
                    std::copy_n(transformed.data() + c * encoder_frames + start, count,
                        input.data() + c * settings.tail_chunk_frames);
                const auto prefix = full_tail_prefix->run(input);
                for (int64_t offset = 0; offset < count * prefix_rate;) {
                    const int64_t valid = std::min(settings.final_stage_chunk_frames - pending, count * prefix_rate - offset);
                    for (int64_t c = 0; c < channels; ++c)
                        std::copy_n(prefix.data() + c * settings.tail_chunk_frames * prefix_rate + offset, valid,
                            tail.data() + c * settings.final_stage_chunk_frames + pending);
                    pending += valid;
                    offset += valid;
                    if (pending == settings.final_stage_chunk_frames) {
                        const auto samples = full_tail_final->run(tail);
                        pcm.insert(pcm.end(), samples.begin(), samples.end());
                        pending = 0;
                    }
                }
            }
            if (pending) {
                for (int64_t c = 0; c < channels; ++c)
                    std::fill_n(tail.data() + c * settings.final_stage_chunk_frames + pending,
                        settings.final_stage_chunk_frames - pending, 0.0F);
                const auto samples = full_tail_final->run(tail);
                pcm.insert(pcm.end(), samples.begin(), samples.begin() + pending * config.stage_strides.back());
            }
            return pcm;
        }
        // Fixed latent blocks preserve both the attention capacity and KV
        // history on the padded final block; only valid PCM escapes the call.
        std::vector<float> pcm;
        const int64_t block = settings.offline_block_frames;
        if (!offline_block_graph) offline_block_graph = std::make_unique<DecoderGraph>(execution, config, weights,
            std::max({settings.conv_graph_context_bytes, settings.transformer_graph_context_bytes, settings.tail_graph_context_bytes}),
            settings.log_prefix + ".block", block * config.encoder_upsample_stride, DecoderGraphMode::Block);
        auto & decoder = *offline_block_graph;
        decoder.reset();
        for (int64_t start = 0; start < frames; start += block) {
            const int64_t count = std::min(block, frames - start);
            std::vector<float> input(static_cast<size_t>(config.latent_size * block), 0.0F);
            for (int64_t c = 0; c < config.latent_size; ++c)
                std::copy_n(packed.data() + c * frames + start, count, input.data() + c * block);
            auto x = decoder.run(input);
            const auto valid_samples = x.size() / static_cast<size_t>(block) * static_cast<size_t>(count);
            pcm.insert(pcm.end(), x.begin(), x.begin() + static_cast<ptrdiff_t>(valid_samples));
        }
        return pcm;
    }
};

MimiLatentDecoderRuntime::MimiLatentDecoderRuntime(const core::ExecutionContext & execution,
    MimiLatentDecoderConfig config, MimiLatentDecoderWeights weights, MimiLatentDecoderExecutionConfig settings)
    : impl_(std::make_unique<Impl>(execution, std::move(config), std::move(weights), std::move(settings))) {}
MimiLatentDecoderRuntime::~MimiLatentDecoderRuntime() = default;
const MimiLatentDecoderConfig & MimiLatentDecoderRuntime::config() const noexcept { return impl_->config; }
std::vector<float> MimiLatentDecoderRuntime::decode(const std::vector<float> & latents, int64_t frames, bool full) {
    const auto start = std::chrono::steady_clock::now();
    auto result = impl_->decode(latents, frames, full);
    debug::timing_log_scalar(impl_->settings.log_prefix + ".decoder_ms", debug::elapsed_ms(start));
    return result;
}
std::vector<float> MimiLatentDecoderRuntime::decode_streaming(const std::vector<float> & latents) {
    const auto start = std::chrono::steady_clock::now();
    const int64_t frames = static_cast<int64_t>(latents.size() / impl_->config.latent_size);
    auto packed = impl_->pack(latents, frames);
    if (impl_->stream_block && impl_->stream_block != frames)
        throw std::invalid_argument("Reset Mimi streaming state before changing block size");
    if (!impl_->streaming_block_graph || (!impl_->stream_block && impl_->streaming_block_graph->frames != frames * impl_->config.encoder_upsample_stride)) {
        impl_->streaming_block_graph.reset();
        impl_->streaming_block_graph = std::make_unique<DecoderGraph>(impl_->execution, impl_->config, impl_->weights,
            std::max({impl_->settings.conv_graph_context_bytes, impl_->settings.transformer_graph_context_bytes, impl_->settings.tail_graph_context_bytes}),
            impl_->settings.log_prefix + ".stream", frames * impl_->config.encoder_upsample_stride, DecoderGraphMode::Block);
    }
    if (!impl_->stream_block) impl_->streaming_block_graph->reset();
    impl_->stream_block = frames;
    auto result = impl_->streaming_block_graph->run(packed);
    debug::timing_log_scalar(impl_->settings.log_prefix + ".streaming_decoder_step_ms", debug::elapsed_ms(start));
    return result;
}
void MimiLatentDecoderRuntime::reset_streaming() {
    impl_->stream_block = 0;
    // The next first block resets KV once, after selecting its graph shape.
}
void MimiLatentDecoderRuntime::clear_runtime_cache() {
    impl_->full_graph.reset(); impl_->full_frames = 0;
    impl_->offline_block_graph.reset(); impl_->streaming_block_graph.reset();
    impl_->full_front.reset(); impl_->full_front_frames = 0;
    impl_->full_tail_prefix.reset(); impl_->full_tail_final.reset();
    impl_->stream_block = 0;
}
} // namespace engine::codecs
