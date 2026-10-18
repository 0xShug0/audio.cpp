#include "engine/models/moonshine_asr/neon_decoder.h"
#include "test_assert.h"

#include <cmath>
#include <iostream>

namespace model = engine::models::moonshine_asr;
namespace {
using Vector = std::vector<double>;
std::vector<float> values(size_t count) {
    std::vector<float> result(count);
    for (size_t i = 0; i < count; ++i) {
        result[i] = 0.1F * std::sin(float(i) * 0.19F);
    }
    return result;
}
model::MoonshineNeonLinear matrix(int input, int output, bool bias = false) {
    return {input, output, values(input * output), bias ? values(output) : std::vector<float>{}};
}
Vector linear(const Vector &input, const model::MoonshineNeonLinear &weight) {
    const auto &w = std::get<std::vector<float>>(weight.weight);
    Vector output(weight.output_features);
    for (int row = 0; row < weight.output_features; ++row) {
        double sum = weight.bias.empty() ? 0 : weight.bias[row];
        for (int column = 0; column < weight.input_features; ++column) {
            sum += input[column] * w[row * weight.input_features + column];
        }
        output[row] = sum;
    }
    return output;
}
Vector norm(Vector input, const std::vector<float> &weight) {
    double mean = 0, variance = 0;
    for (double value : input) {
        mean += value;
    }
    mean /= input.size();
    for (double value : input) {
        variance += (value - mean) * (value - mean);
    }
    const double inverse = 1 / std::sqrt(variance / input.size() + 1e-5);
    for (size_t i = 0; i < input.size(); ++i) {
        input[i] = (input[i] - mean) * inverse * weight[i];
    }
    return input;
}
Vector rope(Vector input, int position) {
    for (size_t head = 0; head < input.size() / 5; ++head) {
        for (int pair = 0; pair < 2; ++pair) {
            const double angle = position * std::pow(10000.0, -pair * 0.5);
            const size_t a = head * 5 + pair * 2, b = a + 1;
            const double x = input[a], y = input[b];
            input[a] = x * std::cos(angle) - y * std::sin(angle);
            input[b] = x * std::sin(angle) + y * std::cos(angle);
        }
    }
    return input;
}
Vector attention(const Vector &query, const std::vector<Vector> &keys, const std::vector<Vector> &values) {
    Vector output(10);
    for (int head = 0; head < 2; ++head) {
        Vector probability(keys.size());
        double denominator = 0;
        for (size_t frame = 0; frame < keys.size(); ++frame) {
            double score = 0;
            for (int d = 0; d < 5; ++d) {
                score += query[head * 5 + d] * keys[frame][d];
            }
            probability[frame] = std::exp(score / std::sqrt(5.0));
            denominator += probability[frame];
        }
        for (int d = 0; d < 5; ++d) {
            for (size_t frame = 0; frame < keys.size(); ++frame) {
                output[head * 5 + d] += probability[frame] / denominator * values[frame][d];
            }
        }
    }
    return output;
}
void add(Vector &input, const Vector &residual) {
    for (size_t i = 0; i < input.size(); ++i) {
        input[i] += residual[i];
    }
}
} // namespace

int main() {
    try {
        model::MoonshineDecoderConfig config;
        config.hidden_size = 10;
        config.intermediate_size = 7;
        config.heads = 2;
        config.kv_heads = 1;
        config.head_dim = 5;
        config.rotary_dim = 4;
        config.layers = 2;
        config.vocab_size = 17;
        config.max_position_embeddings = 16;
        model::MoonshineNeonDecoderWeights weights;
        weights.token_embedding = matrix(10, 17);
        weights.norm.assign(10, 1);
        weights.head = matrix(10, 17);
        for (int i = 0; i < 2; ++i) {
            model::MoonshineNeonDecoderLayer layer;
            layer.input_norm.assign(10, 1);
            layer.cross_norm.assign(10, 1);
            layer.mlp_norm.assign(10, 1);
            layer.self_query = matrix(10, 10);
            layer.self_key = matrix(10, 5);
            layer.self_value = matrix(10, 5);
            layer.self_output = matrix(10, 10);
            layer.cross_query = matrix(10, 10);
            layer.cross_key = matrix(10, 5);
            layer.cross_value = matrix(10, 5);
            layer.cross_output = matrix(10, 10);
            layer.mlp_input = matrix(10, 14, true);
            layer.mlp_output = matrix(7, 10, true);
            weights.layers.push_back(std::move(layer));
        }
        model::MoonshineNeonDecoder decoder(config, weights);
        for (int threads : {1, 4, 7, 1}) {
            for (int memory_frames : {3, 1, 5}) {
                const auto memory = values(memory_frames * 10);
                decoder.prepare_memory(memory, 6, threads);
                std::vector<std::vector<Vector>> self_keys(2), self_values(2), cross_keys(2), cross_values(2);
                for (size_t layer = 0; layer < 2; ++layer) {
                    for (int frame = 0; frame < memory_frames; ++frame) {
                        const Vector row(memory.begin() + frame * 10, memory.begin() + (frame + 1) * 10);
                        cross_keys[layer].push_back(linear(row, weights.layers[layer].cross_key));
                        cross_values[layer].push_back(linear(row, weights.layers[layer].cross_value));
                    }
                }
                int32_t position = 0;
                for (int32_t token : {1, 4, 0, 7, 16, 2}) {
                    const auto &table = std::get<std::vector<float>>(weights.token_embedding.weight);
                    Vector state(table.begin() + token * 10, table.begin() + (token + 1) * 10);
                    for (size_t i = 0; i < 2; ++i) {
                        const auto &layer = weights.layers[i];
                        auto hidden = norm(state, layer.input_norm);
                        const auto query = rope(linear(hidden, layer.self_query), position);
                        self_keys[i].push_back(rope(linear(hidden, layer.self_key), position));
                        self_values[i].push_back(linear(hidden, layer.self_value));
                        add(state, linear(attention(query, self_keys[i], self_values[i]), layer.self_output));
                        hidden = norm(state, layer.cross_norm);
                        add(state, linear(attention(linear(hidden, layer.cross_query), cross_keys[i], cross_values[i]),
                                          layer.cross_output));
                        hidden = linear(norm(state, layer.mlp_norm), layer.mlp_input);
                        Vector gated(7);
                        for (int j = 0; j < 7; ++j) {
                            gated[j] = hidden[j] * hidden[j + 7] / (1 + std::exp(-hidden[j + 7]));
                        }
                        add(state, linear(gated, layer.mlp_output));
                    }
                    const auto expected = linear(norm(state, weights.norm), weights.head);
                    const auto &actual = decoder.run_step(token, position++, threads);
                    engine::test::require(actual.size() == expected.size(), "decoder vocabulary size");
                    for (size_t i = 0; i < actual.size(); ++i) {
                        engine::test::require(std::isfinite(actual[i]) && std::abs(actual[i] - expected[i]) <
                                                                              3e-5 * (1 + std::abs(expected[i])),
                                              "whole decoder matches independent double reference");
                    }
                }
            }
        }
        std::cout << "Moonshine complete native decoder passed\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
