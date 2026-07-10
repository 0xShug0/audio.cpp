#pragma once

#include "engine/framework/core/backend.h"
#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/core/execution_context.h"
#include "engine/framework/core/module.h"

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace engine::modules {

struct HubertEncoderConfig {
    int64_t hidden_size = 1024;
    int64_t intermediate_size = 4096;
    int64_t num_hidden_layers = 24;
    int64_t output_hidden_layer = 24;
    int64_t num_attention_heads = 16;
    int64_t conv_in_channels = 1;
    int64_t num_conv_pos_embeddings = 128;
    int64_t num_conv_pos_embedding_groups = 16;
    std::vector<int64_t> conv_dim{512, 512, 512, 512, 512, 512, 512};
    std::vector<int64_t> conv_kernel{10, 3, 3, 3, 3, 2, 2};
    std::vector<int64_t> conv_stride{5, 2, 2, 2, 2, 2, 2};
    float layer_norm_eps = 1.0e-5F;
    bool apply_positional_embedding = true;
    bool apply_final_layer_norm = true;
};

struct HubertEncoderOutput {
    std::vector<float> hidden_states;
    int64_t batch = 0;
    int64_t tokens = 0;
    int64_t hidden_size = 0;
};

struct HubertEncoderWeights {
    HubertEncoderConfig config;
    std::filesystem::path source_path;
    std::shared_ptr<core::ExecutionContext> execution_context;
    std::shared_ptr<core::BackendWeightStore> store;
    std::unordered_map<std::string, core::TensorValue> tensors;
    int64_t loaded_tensor_count = 0;
    int64_t parameter_count = 0;
};

class HubertEncoderComponent {
public:
    static HubertEncoderComponent load_from_safetensors(
        const std::filesystem::path & checkpoint_path,
        core::BackendConfig backend,
        HubertEncoderConfig config = {});

    HubertEncoderComponent() = default;
    HubertEncoderComponent(
        std::shared_ptr<const HubertEncoderWeights> weights,
        core::BackendConfig backend);

    const core::BackendConfig & backend() const noexcept;
    const std::shared_ptr<const HubertEncoderWeights> & weights() const noexcept;
    int64_t hidden_size() const noexcept;
    int64_t loaded_tensor_count() const noexcept;
    int64_t parameter_count() const noexcept;
    HubertEncoderOutput encode(
        const std::vector<float> & input_values,
        int64_t batch,
        int64_t samples) const;
    void release_runtime_graph();

private:
    struct State;

    std::shared_ptr<const HubertEncoderWeights> weights_;
    core::BackendConfig backend_;
    std::shared_ptr<State> state_;
};

}  // namespace engine::modules
