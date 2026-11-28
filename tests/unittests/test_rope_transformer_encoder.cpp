#include "engine/framework/modules/speech_encoders/rope_transformer_encoder_runtime.h"
#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/modules/attention/grouped_query_attention.h"
#include "engine/framework/modules/positional_modules.h"
#include "engine/framework/modules/structural_modules.h"

#include <ggml-alloc.h>

#include <cmath>
#include <iostream>
#include <numeric>
#include <stdexcept>

using namespace engine;

namespace {
void require(bool ok, const char * message) { if (!ok) throw std::runtime_error(message); }

struct Fixture {
    core::BackendWeightStore store;
    modules::RopeTransformerEncoderWeights weights;
    modules::LinearWeights projection;
    Fixture(core::ExecutionContext & execution, const modules::RopeTransformerEncoderConfig & c)
        : store(execution.backend(), execution.backend_type(), "encoder unit weights", 1024 * 1024) {
        weights.input_norm = norm(c.hidden_size);
        weights.final_norm = norm(c.hidden_size);
        projection = linear(32, c.hidden_size);
        for (int i = 0; i < 2; ++i) {
            modules::RopeTransformerEncoderLayerWeights layer;
            layer.norm1 = norm(c.hidden_size);
            layer.norm2 = norm(c.hidden_size);
            if (c.qk_norm) { layer.q_norm = norm(c.hidden_size / c.heads); layer.k_norm = norm(c.hidden_size / c.heads); }
            layer.qkv = linear(c.hidden_size, 3 * c.hidden_size);
            layer.qkv.bias.reset();
            layer.attention_output = linear(c.hidden_size, c.hidden_size);
            layer.ffn_in = linear(c.hidden_size, c.intermediate_size);
            layer.ffn_out = linear(c.intermediate_size, c.hidden_size);
            weights.layers.push_back(layer);
        }
        store.upload();
    }
    modules::NormWeights norm(int64_t width) {
        std::vector<float> scale(static_cast<size_t>(width)), bias(scale.size());
        for (size_t i = 0; i < scale.size(); ++i) { scale[i] = 0.9f + float(i % 7) / 20; bias[i] = float(i % 5) / 100; }
        return {store.make_f32(core::TensorShape::from_dims({width}), scale),
            store.make_f32(core::TensorShape::from_dims({width}), bias)};
    }
    modules::LinearWeights linear(int64_t in, int64_t out) {
        std::vector<float> values(static_cast<size_t>(in * out)), bias(static_cast<size_t>(out));
        for (size_t i = 0; i < values.size(); ++i) values[i] = std::sin(float(i + 11) * 0.713f) / std::sqrt(float(in));
        for (size_t i = 0; i < bias.size(); ++i) bias[i] = float(int(i % 11) - 5) / 100;
        return {store.make_f32(core::TensorShape::from_dims({out, in}), values),
            store.make_f32(core::TensorShape::from_dims({out}), bias)};
    }
};

// The pre-refactor graph, kept as an independent regression reference. In
// particular, split-table attention retains its original Slice/Transpose/GQA
// lowering while indexed attention retains strided QKV views and F16 K/V.
core::TensorValue reference(core::ModuleBuildContext & ctx, core::TensorValue input,
    const modules::RopeTransformerEncoderInputs & rotary,
    const modules::RopeTransformerEncoderConfig & c, const modules::RopeTransformerEncoderWeights & weights) {
    auto * ggml = ctx.ggml;
    const auto B = input.shape.at(0), T = input.shape.at(1), D = c.hidden_size, K = D / c.heads;
    const auto norm = [&](ggml_tensor * x, const modules::NormWeights & w) {
        if (!ggml_is_contiguous(x)) x = ggml_cont(ggml, x);
        return ggml_add(ggml, ggml_mul(ggml, ggml_norm(ggml, x, c.norm_epsilon), w.weight->tensor), w.bias->tensor);
    };
    const auto linear = [&](ggml_tensor * x, const modules::LinearWeights & w) {
        // LinearModule projects a flattened [batch*time, channels] matrix.
        // The original indexed encoder had batch=1, so these layouts coincided.
        const auto rows = ggml_nelements(x) / x->ne[0];
        auto * matrix = ggml_reshape_2d(ggml, x, x->ne[0], rows);
        auto * y = ggml_mul_mat(ggml, w.weight.tensor, matrix);
        if (c.projection_precision != GGML_PREC_DEFAULT) ggml_mul_mat_set_prec(y, c.projection_precision);
        if (w.bias) y = ggml_add(ggml, y, w.bias->tensor);
        return ggml_reshape_3d(ggml, y, w.weight.tensor->ne[1], T, B);
    };
    auto * x = input.tensor;
    if (c.input_norm) x = norm(x, weights.input_norm);
    for (const auto & layer : weights.layers) {
        auto * qkv = linear(norm(x, layer.norm1), layer.qkv);
        ggml_tensor * attended;
        if (c.rotary == modules::EncoderRoPE::Indexed) {
            const auto split = [&](int part) {
                return ggml_view_4d(ggml, qkv, K, c.heads, T, B, K * sizeof(float), qkv->nb[1],
                    qkv->nb[2], part * D * sizeof(float));
            };
            auto * q = split(0), * k = split(1), * v = split(2);
            if (c.qk_norm) { q = norm(q, layer.q_norm); k = norm(k, layer.k_norm); }
            q = ggml_rope_ext(ggml, q, rotary.positions.tensor, nullptr, c.rotary_dimensions,
                GGML_ROPE_TYPE_NEOX, 0, c.rope_theta, 1, 0, 1, 0, 0);
            k = ggml_rope_ext(ggml, k, rotary.positions.tensor, nullptr, c.rotary_dimensions,
                GGML_ROPE_TYPE_NEOX, 0, c.rope_theta, 1, 0, 1, 0, 0);
            auto * qp = ggml_permute(ggml, q, 0, 2, 1, 3);
            auto * kp = ggml_cast(ggml, ggml_permute(ggml, k, 0, 2, 1, 3), GGML_TYPE_F16);
            auto * vp = ggml_cast(ggml, ggml_permute(ggml, v, 0, 2, 1, 3), GGML_TYPE_F16);
            if (c.flash_attention) {
                attended = ggml_flash_attn_ext(ggml, qp, kp, vp, (rotary.attention_mask ? rotary.attention_mask->tensor : nullptr),
                    1 / std::sqrt(float(K)), 0, 0);
                ggml_flash_attn_ext_set_prec(attended, GGML_PREC_F32);
            } else {
                auto * scores = ggml_mul_mat(ggml, kp, qp);
                ggml_mul_mat_set_prec(scores, GGML_PREC_F32);
                auto * probs = ggml_soft_max_ext(ggml, scores, (rotary.attention_mask ? rotary.attention_mask->tensor : nullptr), 1 / std::sqrt(float(K)), 0);
                auto * values = ggml_cont(ggml, ggml_permute(ggml, vp, 1, 0, 2, 3));
                attended = ggml_mul_mat(ggml, values, probs);
                ggml_mul_mat_set_prec(attended, GGML_PREC_F32);
                attended = ggml_cont(ggml, ggml_permute(ggml, attended, 0, 2, 1, 3));
            }
        } else {
            auto packed = core::wrap_tensor(qkv, input.shape.with_last_dim(3 * D));
            const auto split = [&](int64_t offset) {
                auto value = modules::SliceModule({2, offset, D}).build(ctx, packed);
                value = core::ensure_backend_addressable_layout(ctx, value);
                value = core::reshape_tensor(ctx, value, core::TensorShape::from_dims({B, T, c.heads, K}));
                return modules::TransposeModule({{0, 2, 1, 3}, 4}).build(ctx, value);
            };
            auto q = modules::SplitRoPEModule({K}).build(ctx, split(0), rotary.cosine, rotary.sine);
            auto k = modules::SplitRoPEModule({K}).build(ctx, split(D), rotary.cosine, rotary.sine);
            auto context = modules::GroupedQueryAttentionModule({K, c.flash_attention
                ? modules::GroupedQueryAttentionLowering::FlashGroupedViewKV
                : modules::GroupedQueryAttentionLowering::ManualRepeat, GGML_PREC_F32,
                modules::AttentionCausality::NonCausal}).build(ctx, q, k, split(2 * D), rotary.attention_mask);
            context = core::ensure_backend_addressable_layout(ctx, context);
            attended = context.tensor;
        }
        auto * merged = ggml_reshape_3d(ggml, attended, D, T, B);
        x = ggml_add(ggml, x, linear(merged, layer.attention_output));
        x = ggml_add(ggml, x, linear(ggml_gelu_erf(ggml, linear(norm(x, layer.norm2), layer.ffn_in)), layer.ffn_out));
    }
    return core::wrap_tensor(norm(x, weights.final_norm), input.shape);
}

struct ReferenceGraph {
    ggml_backend_t backend;
    ggml_context * context;
    ggml_cgraph * graph = nullptr;
    ggml_gallocr_t allocator = nullptr;
    explicit ReferenceGraph(ggml_backend_t b) : backend(b), context(ggml_init({32 * 1024 * 1024, nullptr, true})) {}
    ~ReferenceGraph() {
        if (graph) core::release_backend_graph_resources(backend, graph, true);
        if (allocator) ggml_gallocr_free(allocator);
        ggml_free(context);
    }
    void allocate(core::TensorValue output) {
        graph = ggml_new_graph_custom(context, 16384, false);
        ggml_set_output(output.tensor);
        ggml_build_forward_expand(graph, output.tensor);
        allocator = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
        require(ggml_gallocr_alloc_graph(allocator, graph), "reference graph allocation");
    }
};

void check(core::ExecutionContext & execution, bool tables, bool flash, int64_t batch, int64_t frames, bool masked) {
    std::cout << "case tables=" << tables << " flash=" << flash << " batch=" << batch
        << " frames=" << frames << " masked=" << masked << std::endl;
    modules::RopeTransformerEncoderConfig c;
    c.hidden_size = 128; c.heads = 2; c.intermediate_size = 192;
    c.rotary_dimensions = tables ? 64 : 32;
    c.rotary = tables ? modules::EncoderRoPE::SplitTables : modules::EncoderRoPE::Indexed;
    c.qk_norm = !tables; c.f16_kv = !tables;
    c.projection_precision = tables ? GGML_PREC_DEFAULT : GGML_PREC_F32;
    c.flash_attention = flash;
    Fixture fixture(execution, c);
    ReferenceGraph graph(execution.backend());
    core::ModuleBuildContext ctx{graph.context, "encoder reference", execution.backend_type()};
    auto input = core::make_tensor(ctx, GGML_TYPE_F32, core::TensorShape::from_dims({batch, frames, 32}));
    ggml_set_input(input.tensor);
    modules::RopeTransformerEncoderInputs rotary;
    if (masked) {
        rotary.attention_mask = core::make_tensor(ctx, GGML_TYPE_F16, core::TensorShape::from_dims({batch, 1, frames, frames}));
        ggml_set_input(rotary.attention_mask->tensor);
    }
    if (tables) {
        const auto shape = core::TensorShape::from_dims({1, c.heads, frames, 32});
        rotary.cosine = core::make_tensor(ctx, GGML_TYPE_F32, shape);
        rotary.sine = core::make_tensor(ctx, GGML_TYPE_F32, shape);
        for (auto value : {rotary.cosine, rotary.sine}) { ggml_set_input(value.tensor); ggml_set_output(value.tensor); }
    } else {
        rotary.positions = core::make_tensor(ctx, GGML_TYPE_I32, core::TensorShape::from_dims({frames}));
        ggml_set_input(rotary.positions.tensor); ggml_set_output(rotary.positions.tensor);
    }
    auto projected = modules::LinearModule({32, 128, true, c.projection_precision}).build(ctx, input, fixture.projection);
    auto expected = reference(ctx, projected, rotary, c, fixture.weights);
    graph.allocate(expected);
    std::vector<float> values(static_cast<size_t>(batch * frames * 32)), mask(static_cast<size_t>(batch * frames * frames));
    for (size_t i = 0; i < values.size(); ++i) values[i] = std::cos(float(i) * 0.31f);
    for (int64_t b = 0; b < batch; ++b) for (int64_t t = 0; t < frames; ++t)
        for (int64_t k = frames - 2; k < frames; ++k) mask[static_cast<size_t>((b * frames + t) * frames + k)] = -10000;
    core::write_tensor_f32(input, values);
    if (masked) core::write_tensor_f16(*rotary.attention_mask, mask);
    if (tables) {
        std::vector<float> cosine(static_cast<size_t>(c.heads * frames * 32)), sine(cosine.size());
        for (int64_t h = 0; h < c.heads; ++h) for (int64_t t = 0; t < frames; ++t) for (int64_t i = 0; i < 32; ++i) {
            const double phase = t * std::pow(double(c.rope_theta), -2.0 * i / 64);
            const auto index = static_cast<size_t>((h * frames + t) * 32 + i);
            cosine[index] = float(std::cos(phase)); sine[index] = float(std::sin(phase));
        }
        core::write_tensor_f32(rotary.cosine, cosine); core::write_tensor_f32(rotary.sine, sine);
    } else {
        std::vector<int32_t> positions(static_cast<size_t>(frames)); std::iota(positions.begin(), positions.end(), 0);
        core::write_tensor_i32(rotary.positions, positions.data(), positions.size());
    }
    require(core::compute_backend_graph(execution.backend(), graph.graph, nullptr) == GGML_STATUS_SUCCESS, "reference execution");
    const auto reference_values = core::read_tensor_f32(expected.tensor);
    int head_builds = 0;
    modules::RopeTransformerEncoderGraphConfig graph_config;
    graph_config.input_projection = fixture.projection;
    graph_config.masked_attention = masked;
    graph_config.head = [&](core::ModuleBuildContext &, const core::TensorValue & encoded) {
        ++head_builds;
        return std::vector<core::TensorValue>{encoded};
    };
    modules::RopeTransformerEncoderRuntime runtime(execution, c, fixture.weights, std::move(graph_config));
    for (int pass = 0; pass < 3; ++pass) {
        runtime.prepare(batch, frames);
        core::write_tensor_f32(runtime.input(), values);
        if (masked) core::write_tensor_f16(runtime.attention_mask(), mask);
        runtime.run();
        const auto actual = core::read_tensor_f32(runtime.output().tensor);
        require(actual.size() == reference_values.size(), "encoder output size");
        for (size_t i = 0; i < actual.size(); ++i) {
            if (!std::isfinite(actual[i]) || std::abs(actual[i] - reference_values[i]) >= 2e-5f) {
                std::cerr << "index=" << i << " expected=" << reference_values[i] << " actual=" << actual[i]
                    << " difference=" << actual[i] - reference_values[i] << std::endl;
                throw std::runtime_error("encoder changed original graph math");
            }
        }
    }
    require(runtime.graphs_built() == 1 && head_builds == 1, "same-shape encoder graph rebuilt");
    runtime.prepare(batch, frames + 1);
    require(runtime.graphs_built() == 2, "shape change did not rebuild graph");
    runtime.release_workspace(); runtime.prepare(batch, frames);
    core::write_tensor_f32(runtime.input(), values);
    if (masked) core::write_tensor_f16(runtime.attention_mask(), mask);
    runtime.run();
    require(runtime.graphs_built() == 3 && head_builds == 3, "release/rebuild lifecycle");
    const auto restored = core::read_tensor_f32(runtime.output().tensor);
    for (size_t i = 0; i < restored.size(); ++i) require(std::abs(restored[i] - reference_values[i]) < 2e-5f, "release changed output");
}
} // namespace

int main(int argc, char ** argv) {
    const auto started = std::chrono::steady_clock::now();
    try {
        core::BackendConfig backend; backend.type = core::BackendType::Cpu; backend.threads = 8;
        debug::LoggingConfig logging;
        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            if (arg == "--backend" && i + 1 < argc) {
                const std::string name = argv[++i];
                if (name == "cpu") backend.type = core::BackendType::Cpu;
                else if (name == "cuda") backend.type = core::BackendType::Cuda;
                else if (name == "vulkan") backend.type = core::BackendType::Vulkan;
                else throw std::invalid_argument("unsupported test backend");
            } else if (arg == "--log" && i + 1 < argc) { logging.enabled = true; logging.file_path = argv[++i]; }
            else throw std::invalid_argument("unknown test option");
        }
        debug::configure_logging(logging);
        core::ExecutionContext execution(backend);
        for (bool tables : {false, true}) for (bool flash : {false, true})
            for (int64_t batch : {1, 2}) for (int64_t frames : {7, 32})
                for (bool masked : {false, true}) check(execution, tables, flash, batch, frames, masked);
        debug::timing_log_scalar("session.wall_ms", debug::elapsed_ms(started));
        std::cout << "rope_transformer_encoder_test: PASS\n";
        return 0;
    } catch (const std::exception & e) {
        debug::timing_log_scalar("session.wall_ms", debug::elapsed_ms(started));
        std::cerr << e.what() << '\n'; return 1;
    }
}
