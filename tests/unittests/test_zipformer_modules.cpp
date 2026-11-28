#include "engine/framework/core/prepared_graph.h"
#include "engine/framework/modules/zipformer_modules.h"
#include "test_assert.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <utility>
#include <vector>

namespace {

void check_attention(int64_t frames, int64_t left, int64_t heads,
    engine::modules::ZipformerAttentionImpl impl) {
    using namespace engine;
    constexpr int64_t query_dim = 3;
    constexpr int64_t position_dim = 6;
    constexpr int64_t position_head_dim = 4;
    const int64_t channels = heads * (2 * query_dim + position_head_dim);
    const int64_t sources = frames + left;
    core::ExecutionContext execution({core::BackendType::Cpu, 0, 2});
    core::PreparedGraph graph(execution, {"zipformer attention test", 4 * 1024 * 1024, 1024});
    auto & ctx = graph.build_context();
    std::vector<std::pair<core::TensorValue, std::vector<float>>> inputs;
    const auto tensor = [&](const core::TensorShape & shape, std::vector<float> values) {
        auto value = core::make_tensor(ctx, GGML_TYPE_F32, shape);
        ggml_set_input(value.tensor);
        inputs.emplace_back(value, std::move(values));
        return value;
    };
    std::vector<float> projected(static_cast<size_t>(frames * channels));
    for (size_t i = 0; i < projected.size(); ++i) projected[i] = 0.3F * std::sin(static_cast<float>(i));
    std::vector<float> identity(static_cast<size_t>(channels * channels));
    for (int64_t i = 0; i < channels; ++i) identity[static_cast<size_t>(i * channels + i)] = 1.0F;
    auto input = tensor(core::TensorShape::from_dims({frames, 1, channels}), projected);
    auto weight = tensor(core::TensorShape::from_dims({channels, channels}), identity);
    auto bias = tensor(core::TensorShape::from_dims({channels}), std::vector<float>(static_cast<size_t>(channels)));
    std::vector<float> cached(static_cast<size_t>(left * heads * query_dim));
    for (size_t i = 0; i < cached.size(); ++i) cached[i] = 0.2F * std::cos(static_cast<float>(i));
    core::TensorValue cache;
    if (left != 0) cache = tensor(core::TensorShape::from_dims({left, 1, heads * query_dim}), cached);
    std::vector<float> mask(static_cast<size_t>(frames * sources));
    if (left != 0) for (int64_t q = 0; q < frames; ++q) mask[static_cast<size_t>(q * sources)] = -1000.0F;
    auto mask_tensor = left == 0 ? core::TensorValue{} : tensor(core::TensorShape::from_dims({1, frames, sources}), mask);
    std::vector<float> pos_weight(static_cast<size_t>(heads * position_head_dim * position_dim));
    for (size_t i = 0; i < pos_weight.size(); ++i) pos_weight[i] = 0.1F * std::cos(static_cast<float>(i) * 0.7F);
    modules::ZipformerConstantFactory constants = tensor;
    modules::ZipformerRelativeAttentionConfig config{heads, query_dim, position_dim, position_head_dim};
    if (impl == modules::ZipformerAttentionImpl::Batched) config.impl = impl;
    auto result = modules::ZipformerRelativeAttentionModule(config)
        .build(ctx, constants, input, cache, mask_tensor, {{weight, bias}, pos_weight});
    test::require(result.batched_weights.valid() == (impl == modules::ZipformerAttentionImpl::Batched),
        "Zipformer attention implementation selection was ignored");
    std::vector<core::TensorValue> outputs;
    for (auto value : result.weights) {
        value = core::ensure_backend_addressable_layout(ctx, value);
        ggml_set_output(value.tensor);
        ggml_build_forward_expand(graph.graph(), value.tensor);
        outputs.push_back(value);
    }
    graph.allocate();
    for (const auto & value : inputs) core::write_tensor_f32(value.first, value.second);
    graph.compute();
    for (int64_t head = 0; head < heads; ++head) {
        const auto actual = core::read_tensor_f32(outputs[static_cast<size_t>(head)].tensor);
        for (int64_t q = 0; q < frames; ++q) {
            std::vector<float> expected(static_cast<size_t>(sources));
            for (int64_t k = 0; k < sources; ++k) {
                float score = mask[static_cast<size_t>(q * sources + k)];
                for (int64_t d = 0; d < query_dim; ++d) {
                    const float key = k < left ? cached[static_cast<size_t>(k * heads * query_dim + head * query_dim + d)] :
                        projected[static_cast<size_t>((k - left) * channels + heads * query_dim + head * query_dim + d)];
                    score += projected[static_cast<size_t>(q * channels + head * query_dim + d)] * key;
                }
                const float x = static_cast<float>(k - q - left);
                const float compression = std::sqrt(static_cast<float>(position_dim));
                const float sign = x < 0 ? -1.0F : (x > 0 ? 1.0F : 0.0F);
                const float angle = std::atan(compression * sign * (std::log(std::fabs(x) + compression) - std::log(compression)) /
                    (static_cast<float>(position_dim) / (2.0F * 3.14159265358979323846F)));
                for (int64_t d = 0; d < position_head_dim; ++d) {
                    double coefficient = 0;
                    for (int64_t p = 0; p < position_dim; ++p) {
                        const float position = p == position_dim - 1 ? 1.0F :
                            (p % 2 == 0 ? std::cos(angle * static_cast<float>(p / 2 + 1)) : std::sin(angle * static_cast<float>(p / 2 + 1)));
                        coefficient += static_cast<double>(position) * pos_weight[static_cast<size_t>((head * position_head_dim + d) * position_dim + p)];
                    }
                    score += projected[static_cast<size_t>(q * channels + 2 * heads * query_dim + head * position_head_dim + d)] * static_cast<float>(coefficient);
                }
                expected[static_cast<size_t>(k)] = score;
            }
            const float maximum = *std::max_element(expected.begin(), expected.end());
            double sum = 0;
            for (auto & value : expected) { value = std::exp(value - maximum); sum += value; }
            for (int64_t k = 0; k < sources; ++k) {
                const auto index = static_cast<size_t>(q * sources + k);
                test::require(std::isfinite(actual[index]), "Zipformer attention produced a non-finite value");
                test::require_close(actual[index], expected[static_cast<size_t>(k)] / static_cast<float>(sum), 2e-6F,
                    "Zipformer compact attention differs from scalar relative-position attention");
            }
        }
    }
}

}  // namespace

int main() try {
    for (auto impl : {engine::modules::ZipformerAttentionImpl::Default, engine::modules::ZipformerAttentionImpl::Batched})
        for (int64_t frames : {1, 5, 13}) for (int64_t left : {0, 3}) for (int64_t heads : {2, 4}) check_attention(frames, left, heads, impl);
    std::cout << "Zipformer module tests passed\n";
    return 0;
} catch (const std::exception & e) {
    std::cerr << e.what() << '\n';
    return 1;
}
