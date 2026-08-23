#include "test_assert.h"

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/modules/speech_encoders/hubert_encoder.h"

#include <ggml-backend.h>
#include <ggml.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

using engine::test::require;
using engine::test::require_close;

constexpr int64_t kHidden = 8;
constexpr int64_t kIntermediate = 16;
constexpr int64_t kHeads = 2;
constexpr int64_t kPosConvGroups = 2;
constexpr int64_t kPosConvKernel = 5;

// Deterministic per-index values, distinct from zeros.
float fill_value(size_t index) {
    return static_cast<float>((index * 7 + 3) % 11) / 5.0F - 0.4F;
}

class FakeTensorSource final : public engine::assets::TensorSource {
public:
    struct Tensor {
        std::vector<int64_t> shape;
        std::vector<float> values;
    };

    void add(std::string name, std::vector<int64_t> shape, std::vector<float> values) {
        tensors_.emplace(std::move(name), Tensor{std::move(shape), std::move(values)});
    }

    const std::filesystem::path & source_path() const noexcept override {
        static const std::filesystem::path kPath = std::filesystem::path("fake");
        return kPath;
    }
    bool has_tensor(std::string_view name) const noexcept override {
        return tensors_.find(std::string(name)) != tensors_.end();
    }
    engine::assets::TensorMetadata require_metadata(std::string_view name) const override {
        return {std::string(name), "f32", require(name)->shape};
    }
    std::vector<engine::assets::TensorMetadata> tensors() const override {
        std::vector<engine::assets::TensorMetadata> out;
        for (const auto & [name, tensor] : tensors_) {
            out.push_back({name, "f32", tensor.shape});
        }
        return out;
    }
    void release_storage() const override {}
    engine::assets::RawTensorData require_tensor_data(std::string_view name) const override {
        const auto & tensor = *require(name);
        engine::assets::RawTensorData raw;
        raw.metadata = {std::string(name), "f32", tensor.shape};
        raw.bytes.resize(tensor.values.size() * sizeof(float));
        std::memcpy(raw.bytes.data(), tensor.values.data(), raw.bytes.size());
        return raw;
    }
    std::vector<float> require_f32(
        std::string_view name,
        const std::optional<std::vector<int64_t>> & expected_shape) const override {
        const auto & tensor = *require(name);
        if (expected_shape.has_value() && *expected_shape != tensor.shape) {
            throw std::runtime_error("shape mismatch for " + std::string(name));
        }
        return tensor.values;
    }
    std::optional<std::vector<float>> optional_f32(
        std::string_view name,
        const std::optional<std::vector<int64_t>> & expected_shape) const override {
        if (!has_tensor(name)) {
            return std::nullopt;
        }
        return require_f32(name, expected_shape);
    }
    int64_t require_i64_scalar(std::string_view name) const override {
        throw std::runtime_error("unexpected i64 scalar request: " + std::string(name));
    }

private:
    const Tensor * require(std::string_view name) const {
        const auto it = tensors_.find(std::string(name));
        if (it == tensors_.end()) {
            throw std::runtime_error("missing tensor: " + std::string(name));
        }
        return &it->second;
    }

    std::unordered_map<std::string, Tensor> tensors_;
};

engine::modules::HubertEncoderConfig tiny_config() {
    engine::modules::HubertEncoderConfig config;
    config.hidden_size = kHidden;
    config.intermediate_size = kIntermediate;
    config.num_hidden_layers = 1;
    config.output_hidden_layer = 1;
    config.num_attention_heads = kHeads;
    config.conv_dim = {kHidden};
    config.conv_kernel = {3};
    config.conv_stride = {2};
    config.layer_norm_eps = 1.0e-5F;
    config.num_conv_pos_embeddings = kPosConvKernel;
    config.num_conv_pos_embedding_groups = kPosConvGroups;
    config.feature_extractor_norm = engine::modules::HubertFeatureExtractorNorm::LayerNormEveryLayer;
    config.encoder_layer_norm_order = engine::modules::HubertEncoderLayerNormOrder::PreNorm;
    return config;
}

// Populates every non-positional tensor the tiny config needs. Returns the
// effective (out, in, kernel) positional-conv kernel computed in the same way
// the component folds weight-norm layouts.
std::vector<float> populate_common_tensors(FakeTensorSource & source) {
    const auto fill = [](const std::vector<int64_t> & shape) {
        size_t count = 1;
        for (const int64_t dim : shape) {
            count *= static_cast<size_t>(dim);
        }
        std::vector<float> values(count, 0.0F);
        for (size_t i = 0; i < values.size(); ++i) {
            values[i] = fill_value(i + 1);
        }
        return values;
    };
    // feature_extractor.conv_layers.0
    source.add("feature_extractor.conv_layers.0.conv.weight", {kHidden, 1, 3}, fill({kHidden, 1, 3}));
    source.add("feature_extractor.conv_layers.0.conv.bias", {kHidden}, std::vector<float>(static_cast<size_t>(kHidden), 0.1F));
    source.add("feature_extractor.conv_layers.0.layer_norm.weight", {kHidden}, std::vector<float>(static_cast<size_t>(kHidden), 1.0F));
    source.add("feature_extractor.conv_layers.0.layer_norm.bias", {kHidden}, std::vector<float>(static_cast<size_t>(kHidden), 0.0F));
    // feature_projection
    source.add("feature_projection.layer_norm.weight", {kHidden}, std::vector<float>(static_cast<size_t>(kHidden), 1.0F));
    source.add("feature_projection.layer_norm.bias", {kHidden}, std::vector<float>(static_cast<size_t>(kHidden), 0.0F));
    source.add("feature_projection.projection.weight", {kHidden, kHidden}, fill({kHidden, kHidden}));
    source.add("feature_projection.projection.bias", {kHidden}, std::vector<float>(static_cast<size_t>(kHidden), 0.0F));
    // encoder.layer_norm (apply_final_layer_norm)
    source.add("encoder.layer_norm.weight", {kHidden}, std::vector<float>(static_cast<size_t>(kHidden), 1.0F));
    source.add("encoder.layer_norm.bias", {kHidden}, std::vector<float>(static_cast<size_t>(kHidden), 0.0F));
    // encoder.layers.0 attention
    for (const char * proj : {"q_proj", "k_proj", "v_proj", "out_proj"}) {
        const std::string prefix = std::string("encoder.layers.0.attention.") + proj;
        source.add(prefix + ".weight", {kHidden, kHidden}, fill({kHidden, kHidden}));
        source.add(prefix + ".bias", {kHidden}, std::vector<float>(static_cast<size_t>(kHidden), 0.0F));
    }
    // encoder.layers.0 PreNorm layer norm
    source.add("encoder.layers.0.layer_norm.weight", {kHidden}, std::vector<float>(static_cast<size_t>(kHidden), 1.0F));
    source.add("encoder.layers.0.layer_norm.bias", {kHidden}, std::vector<float>(static_cast<size_t>(kHidden), 0.0F));
    // encoder.layers.0 feed forward
    source.add("encoder.layers.0.feed_forward.intermediate_dense.weight", {kIntermediate, kHidden}, fill({kIntermediate, kHidden}));
    source.add("encoder.layers.0.feed_forward.intermediate_dense.bias", {kIntermediate}, std::vector<float>(static_cast<size_t>(kIntermediate), 0.0F));
    source.add("encoder.layers.0.feed_forward.output_dense.weight", {kHidden, kIntermediate}, fill({kHidden, kIntermediate}));
    source.add("encoder.layers.0.feed_forward.output_dense.bias", {kHidden}, std::vector<float>(static_cast<size_t>(kHidden), 0.0F));
    // encoder.layers.0 final layer norm
    source.add("encoder.layers.0.final_layer_norm.weight", {kHidden}, std::vector<float>(static_cast<size_t>(kHidden), 1.0F));
    source.add("encoder.layers.0.final_layer_norm.bias", {kHidden}, std::vector<float>(static_cast<size_t>(kHidden), 0.0F));
    // positional-conv bias
    source.add("encoder.pos_conv_embed.conv.bias", {kHidden}, std::vector<float>(static_cast<size_t>(kHidden), 0.0F));

    // Effective kernel from fairseq-style g/v folding, mirroring the component.
    std::vector<float> v(static_cast<size_t>(kHidden * (kHidden / kPosConvGroups) * kPosConvKernel));
    for (size_t i = 0; i < v.size(); ++i) {
        v[i] = fill_value(i + 1000);
    }
    std::vector<float> g(static_cast<size_t>(kPosConvKernel));
    for (int64_t k = 0; k < kPosConvKernel; ++k) {
        g[static_cast<size_t>(k)] = 0.5F + 0.1F * static_cast<float>(k);
    }
    std::vector<float> weight(v.size());
    const int64_t out_channels = kHidden;
    const int64_t in_channels = kHidden / kPosConvGroups;
    for (int64_t k = 0; k < kPosConvKernel; ++k) {
        double sum = 0.0;
        for (int64_t out = 0; out < out_channels; ++out) {
            for (int64_t in = 0; in < in_channels; ++in) {
                const size_t index = static_cast<size_t>((out * in_channels + in) * kPosConvKernel + k);
                sum += static_cast<double>(v[index]) * static_cast<double>(v[index]);
            }
        }
        const double norm = std::sqrt(sum);
        const float scale = static_cast<float>(static_cast<double>(g[static_cast<size_t>(k)]) / norm);
        for (int64_t out = 0; out < out_channels; ++out) {
            for (int64_t in = 0; in < in_channels; ++in) {
                const size_t index = static_cast<size_t>((out * in_channels + in) * kPosConvKernel + k);
                weight[index] = v[index] * scale;
            }
        }
    }
    return weight;
}

std::vector<float> read_backend_tensor(const engine::core::TensorValue & tensor) {
    require(tensor.type == GGML_TYPE_F32, "expected f32 backend tensor");
    const size_t count = static_cast<size_t>(ggml_nelements(tensor.tensor));
    std::vector<float> out(count);
    ggml_backend_tensor_get(tensor.tensor, out.data(), 0, count * sizeof(float));
    return out;
}

std::vector<float> load_and_read_pos_conv(std::shared_ptr<const engine::assets::TensorSource> source) {
    const engine::core::BackendConfig backend{engine::core::BackendType::Cpu, 0, 1};
    (void) engine::core::init_backend(backend);
    const auto component = engine::modules::HubertEncoderComponent::load_from_tensor_source(
        std::move(source),
        backend,
        tiny_config());
    const auto & tensors = component.weights()->tensors;
    const auto it = tensors.find("encoder.pos_conv_embed.conv.weight");
    require(it != tensors.end(), "positional-conv logical tensor missing");
    return read_backend_tensor(it->second);
}

void test_prefolded_layout() {
    auto source = std::make_shared<FakeTensorSource>();
    const auto expected = populate_common_tensors(*source);
    source->add("encoder.pos_conv_embed.conv.weight", {kHidden, kHidden / kPosConvGroups, kPosConvKernel}, expected);
    const auto actual = load_and_read_pos_conv(std::move(source));
    require(actual.size() == expected.size(), "prefolded kernel size");
    for (size_t i = 0; i < expected.size(); ++i) {
        require_close(actual[i], expected[i], 1.0e-6F, "prefolded kernel element");
    }
}

void test_legacy_weight_norm_layout() {
    auto source = std::make_shared<FakeTensorSource>();
    const auto expected = populate_common_tensors(*source);
    const auto v = [&] {
        std::vector<float> out(static_cast<size_t>(kHidden * (kHidden / kPosConvGroups) * kPosConvKernel));
        for (size_t i = 0; i < out.size(); ++i) {
            out[i] = fill_value(i + 1000);
        }
        return out;
    }();
    std::vector<float> g(static_cast<size_t>(kPosConvKernel));
    for (int64_t k = 0; k < kPosConvKernel; ++k) {
        g[static_cast<size_t>(k)] = 0.5F + 0.1F * static_cast<float>(k);
    }
    source->add("encoder.pos_conv_embed.conv.weight_g", {1, 1, kPosConvKernel}, g);
    source->add("encoder.pos_conv_embed.conv.weight_v", {kHidden, kHidden / kPosConvGroups, kPosConvKernel}, v);
    const auto actual = load_and_read_pos_conv(std::move(source));
    require(actual.size() == expected.size(), "legacy kernel size");
    for (size_t i = 0; i < expected.size(); ++i) {
        require_close(actual[i], expected[i], 1.0e-6F, "legacy kernel element");
    }
}

void test_parametrized_layout() {
    auto source = std::make_shared<FakeTensorSource>();
    const auto expected = populate_common_tensors(*source);
    const auto v = [&] {
        std::vector<float> out(static_cast<size_t>(kHidden * (kHidden / kPosConvGroups) * kPosConvKernel));
        for (size_t i = 0; i < out.size(); ++i) {
            out[i] = fill_value(i + 1000);
        }
        return out;
    }();
    std::vector<float> g(static_cast<size_t>(kPosConvKernel));
    for (int64_t k = 0; k < kPosConvKernel; ++k) {
        g[static_cast<size_t>(k)] = 0.5F + 0.1F * static_cast<float>(k);
    }
    source->add(
        "encoder.pos_conv_embed.conv.parametrizations.weight.original0",
        {1, 1, kPosConvKernel},
        g);
    source->add(
        "encoder.pos_conv_embed.conv.parametrizations.weight.original1",
        {kHidden, kHidden / kPosConvGroups, kPosConvKernel},
        v);
    const auto actual = load_and_read_pos_conv(std::move(source));
    require(actual.size() == expected.size(), "parametrized kernel size");
    for (size_t i = 0; i < expected.size(); ++i) {
        require_close(actual[i], expected[i], 1.0e-6F, "parametrized kernel element");
    }
}

bool load_rejected(std::shared_ptr<const engine::assets::TensorSource> source) {
    const engine::core::BackendConfig backend{engine::core::BackendType::Cpu, 0, 1};
    (void) engine::core::init_backend(backend);
    try {
        (void) engine::modules::HubertEncoderComponent::load_from_tensor_source(
            std::move(source),
            backend,
            tiny_config());
    } catch (const std::exception &) {
        return true;
    }
    return false;
}

void test_missing_scale_pair_rejected() {
    auto source = std::make_shared<FakeTensorSource>();
    (void) populate_common_tensors(*source);
    std::vector<float> g(static_cast<size_t>(kPosConvKernel), 1.0F);
    source->add("encoder.pos_conv_embed.conv.weight_g", {1, 1, kPosConvKernel}, g);
    require(load_rejected(std::move(source)), "half-pair must be rejected");
}

void test_wrong_scale_shape_rejected() {
    auto source = std::make_shared<FakeTensorSource>();
    (void) populate_common_tensors(*source);
    std::vector<float> g(4, 1.0F);
    source->add("encoder.pos_conv_embed.conv.weight_g", {1, 1, 4}, g);
    std::vector<float> v(static_cast<size_t>(kHidden * (kHidden / kPosConvGroups) * kPosConvKernel), 1.0F);
    source->add("encoder.pos_conv_embed.conv.weight_v", {kHidden, kHidden / kPosConvGroups, kPosConvKernel}, v);
    require(load_rejected(std::move(source)), "wrong scale shape must be rejected");
}

void test_zero_norm_rejected() {
    auto source = std::make_shared<FakeTensorSource>();
    (void) populate_common_tensors(*source);
    std::vector<float> g(static_cast<size_t>(kPosConvKernel), 1.0F);
    source->add("encoder.pos_conv_embed.conv.weight_g", {1, 1, kPosConvKernel}, g);
    std::vector<float> v(static_cast<size_t>(kHidden * (kHidden / kPosConvGroups) * kPosConvKernel), 0.0F);
    source->add("encoder.pos_conv_embed.conv.weight_v", {kHidden, kHidden / kPosConvGroups, kPosConvKernel}, v);
    require(load_rejected(std::move(source)), "zero norm must be rejected");
}

void test_prefolded_shape_mismatch_rejected() {
    auto source = std::make_shared<FakeTensorSource>();
    const auto expected = populate_common_tensors(*source);
    source->add(
        "encoder.pos_conv_embed.conv.weight",
        {kHidden, kHidden / kPosConvGroups, kPosConvKernel - 1},
        std::vector<float>(expected.size() - static_cast<size_t>(kHidden * (kHidden / kPosConvGroups)), 0.5F));
    require(load_rejected(std::move(source)), "prefolded shape mismatch must be rejected");
}

}  // namespace

int main() {
    try {
        test_prefolded_layout();
        test_legacy_weight_norm_layout();
        test_parametrized_layout();
        test_missing_scale_pair_rejected();
        test_wrong_scale_shape_rejected();
        test_zero_norm_rejected();
        test_prefolded_shape_mismatch_rejected();
        std::cout << "mms_hubert_positional_conv_test passed\n";
        return 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "mms_hubert_positional_conv_test: %s\n", error.what());
        return 1;
    }
}
