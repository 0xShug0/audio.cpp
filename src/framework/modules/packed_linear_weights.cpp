#include "engine/framework/modules/packed_linear_weights.h"

#include <stdexcept>
#include <string>
#include <utility>

namespace engine::modules {
namespace {

size_t row_bytes(ggml_type type, int64_t columns, const std::string & name) {
    const int64_t block_size = ggml_blck_size(type);
    if (block_size <= 0 || columns % block_size != 0) {
        throw std::runtime_error("Packed linear tensor is not block-aligned: " + name);
    }
    return ggml_row_size(type, columns);
}

}  // namespace

PackedLinearWeightsBuilder::PackedLinearWeightsBuilder(PackedLinearWeightsConfig config) : config_(std::move(config)) {
    if (config_.sources.empty()) {
        throw std::runtime_error("PackedLinearWeightsBuilder requires at least one source");
    }
    if (config_.in_features <= 0) {
        throw std::runtime_error("PackedLinearWeightsConfig.in_features must be positive");
    }
    for (const auto & source : config_.sources) {
        if (source.out_features <= 0) {
            throw std::runtime_error("PackedLinearSource.out_features must be positive");
        }
        if (config_.use_bias && !source.bias.has_value()) {
            throw std::runtime_error("Packed linear source is missing bias");
        }
    }
}

const PackedLinearWeightsConfig & PackedLinearWeightsBuilder::config() const noexcept {
    return config_;
}

LinearWeights PackedLinearWeightsBuilder::build(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    assets::TensorStorageType storage_type) const {
    const int64_t in_features = config_.in_features;
    int64_t out_features = 0;
    ggml_type packed_type = GGML_TYPE_COUNT;
    std::vector<std::byte> packed_weight;
    std::vector<std::byte> packed_bias;

    if (config_.defer_upload) {
        for (const auto & item : config_.sources) {
            const auto type = assets::ggml_type_for_tensor_storage(
                assets::resolve_tensor_storage_type(source, item.weight, storage_type));
            if (packed_type != GGML_TYPE_COUNT && packed_type != type) {
                throw std::runtime_error("Packed linear sources resolved to different tensor types");
            }
            packed_type = type;
            row_bytes(type, in_features, item.weight);
            out_features += item.out_features;
        }
        const auto config = config_;
        auto initialize = [&source, storage_type, config](ggml_tensor *dst, bool bias) {
            size_t offset = 0;
            for (const auto & item : config.sources) {
                const auto tensor = bias
                    ? source.require_tensor(*item.bias, assets::TensorStorageType::F32, {item.out_features})
                    : source.require_tensor(item.weight, storage_type, {item.out_features, config.in_features});
                const size_t expected_bytes = bias
                    ? static_cast<size_t>(item.out_features) * sizeof(float)
                    : row_bytes(tensor.type, config.in_features, item.weight) * item.out_features;
                if (tensor.type != dst->type || tensor.bytes.size() != expected_bytes) {
                    throw std::runtime_error("Packed linear tensor type or byte size mismatch: " + item.weight);
                }
                ggml_backend_tensor_set(dst, tensor.bytes.data(), offset, tensor.bytes.size());
                offset += tensor.bytes.size();
            }
            if (offset != ggml_nbytes(dst)) {
                throw std::runtime_error("Packed linear upload did not cover destination tensor");
            }
        };
        LinearWeights weights;
        weights.weight = store.make_tensor_deferred(
            core::TensorShape::from_dims({out_features, in_features}), packed_type,
            [initialize](ggml_tensor *dst) { initialize(dst, false); });
        if (config_.use_bias) {
            weights.bias = store.make_tensor_deferred(
                core::TensorShape::from_dims({out_features}), GGML_TYPE_F32,
                [initialize](ggml_tensor *dst) { initialize(dst, true); });
        }
        return weights;
    }

    for (const auto & item : config_.sources) {
        const auto tensor = source.require_tensor(
            item.weight,
            storage_type,
            {item.out_features, in_features});
        if (packed_type == GGML_TYPE_COUNT) {
            packed_type = tensor.type;
        } else if (packed_type != tensor.type) {
            throw std::runtime_error("Packed linear sources resolved to different tensor types");
        }
        const size_t expected_bytes = row_bytes(tensor.type, in_features, item.weight) * item.out_features;
        if (tensor.bytes.size() != expected_bytes) {
            throw std::runtime_error("Packed linear tensor byte size mismatch: " + item.weight);
        }
        packed_weight.insert(packed_weight.end(), tensor.bytes.begin(), tensor.bytes.end());

        if (config_.use_bias) {
            const auto bias = source.require_tensor(*item.bias, assets::TensorStorageType::F32, {item.out_features});
            if (bias.type != GGML_TYPE_F32 || bias.bytes.size() != static_cast<size_t>(item.out_features) * sizeof(float)) {
                throw std::runtime_error("Packed linear bias must be F32: " + *item.bias);
            }
            packed_bias.insert(packed_bias.end(), bias.bytes.begin(), bias.bytes.end());
        }
        out_features += item.out_features;
    }

    LinearWeights weights;
    weights.weight = store.make_tensor(
        core::TensorShape::from_dims({out_features, in_features}),
        packed_type,
        packed_weight.data(),
        packed_weight.size());
    if (config_.use_bias) {
        weights.bias = store.make_tensor(
            core::TensorShape::from_dims({out_features}),
            GGML_TYPE_F32,
            packed_bias.data(),
            packed_bias.size());
    }
    return weights;
}

}  // namespace engine::modules
