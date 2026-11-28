#pragma once

#include "engine/framework/core/execution_context.h"
#include "engine/framework/modules/text_encoders/t5_base_encoder.h"
#include "engine/framework/modules/text_encoders/t5_gemma_encoder.h"

#include <cstddef>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <variant>
#include <vector>

namespace engine::modules {

using T5EncoderArchitecture = std::variant<T5BaseEncoderConfig, T5GemmaEncoderConfig>;

// Owns the storage referenced by the tensor bindings, including an optional
// projection. The loader is also retained so released weights can be reloaded.
struct T5EncoderRuntimeWeights {
    std::shared_ptr<const void> storage;
    std::variant<T5BaseEncoderWeights, T5GemmaEncoderWeights> encoder;
    std::optional<LinearWeights> projection;
};

struct T5EncoderRuntimeConfig {
    std::string label = "t5_encoder";
    size_t graph_metadata_bytes = 0; // Zero derives metadata size from graph_nodes.
    size_t graph_nodes = 65536;
    int64_t sequence_padding_multiple = 1;
    int32_t pad_token_id = 0;
    bool retain_weights = true;
    bool retain_graphs = true;
    bool optimize_graph = false;
    bool prepare_host_graph_plan = false;
    std::optional<LinearConfig> projection;
};

// Session-owned, serial encoder execution. Retains at most one shape. The
// execution context must outlive this runtime. Inputs are unpadded token IDs;
// results contain only their rows. Padding is masked from attention.
class T5EncoderRuntime {
public:
    using WeightLoader = std::function<T5EncoderRuntimeWeights()>;
    T5EncoderRuntime(core::ExecutionContext & execution, T5EncoderArchitecture architecture,
                     WeightLoader loader, T5EncoderRuntimeConfig config = {});
    ~T5EncoderRuntime();
    T5EncoderRuntime(const T5EncoderRuntime &) = delete;
    T5EncoderRuntime & operator=(const T5EncoderRuntime &) = delete;

    std::vector<float> encode(const std::vector<int32_t> & token_ids);
    // Optional valid lengths allow already-padded inputs without copying each
    // row first. Outputs contain only valid rows, one vector per actual input.
    // A zero valid length returns an empty row. Batch size always matches input.
    std::vector<std::vector<float>> encode_batch(
        const std::vector<std::vector<int32_t>> & token_ids,
        const std::vector<int64_t> & valid_lengths = {});
    // Explicit phase boundaries remain available to pipelines that encode more
    // than one prompt before releasing workspace. Neither operation changes config.
    void release_graphs();
    void release_weights(); // Releases graphs before their referenced weights.

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace engine::modules
