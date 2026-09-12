#include "engine/models/xtts_v2/gpt.h"

#include "engine/framework/modules/weight_binding.h"

#include <stdexcept>
#include <string>

namespace engine::models::xtts_v2 {

std::shared_ptr<const XttsV2GptWeights> load_xtts_v2_gpt_weights(
    const XttsV2Assets & assets,
    core::ExecutionContext & execution,
    size_t weight_context_bytes,
    assets::TensorStorageType storage_type) {
    namespace binding = engine::modules::binding;
    constexpr int64_t dim = 1024;
    constexpr int64_t mlp = 4096;
    auto weights = std::make_shared<XttsV2GptWeights>();
    weights->store = std::make_shared<core::BackendWeightStore>(
        execution.backend(), execution.backend_type(), "xtts_v2.gpt.weights", weight_context_bytes);
    const auto & source = *assets.gpt;
    weights->text_embedding = weights->store->load_tensor(source, "text_embedding.weight", storage_type, {6681, dim});
    weights->audio_embedding = weights->store->load_tensor(source, "mel_embedding.weight", storage_type, {1026, dim});
    weights->text_positions = weights->store->load_f32_tensor(source, "text_pos_embedding.emb.weight", {404, dim});
    weights->audio_positions = weights->store->load_f32_tensor(source, "mel_pos_embedding.emb.weight", {608, dim});
    weights->layers.reserve(30);
    for (int64_t index = 0; index < 30; ++index) {
        const std::string prefix = "gpt.h." + std::to_string(index);
        XttsV2GptLayerWeights layer;
        layer.attn_norm = binding::norm_from_source(*weights->store, source, prefix + ".ln_1", dim);
        layer.qkv = binding::hf_conv1d_linear_from_source(
            *weights->store, source, prefix + ".attn.c_attn", storage_type, dim, 3 * dim, true);
        layer.attn_out = binding::hf_conv1d_linear_from_source(
            *weights->store, source, prefix + ".attn.c_proj", storage_type, dim, dim, true);
        layer.mlp_norm = binding::norm_from_source(*weights->store, source, prefix + ".ln_2", dim);
        layer.mlp_in = binding::hf_conv1d_linear_from_source(
            *weights->store, source, prefix + ".mlp.c_fc", storage_type, dim, mlp, true);
        layer.mlp_out = binding::hf_conv1d_linear_from_source(
            *weights->store, source, prefix + ".mlp.c_proj", storage_type, mlp, dim, true);
        weights->layers.push_back(std::move(layer));
    }
    weights->transformer_norm = binding::norm_from_source(*weights->store, source, "gpt.ln_f", dim);
    weights->output_norm = binding::norm_from_source(*weights->store, source, "final_norm", dim);
    weights->audio_head = binding::linear_from_source(
        *weights->store, source, "mel_head", storage_type, 1026, dim, true);
    weights->store->upload();
    return weights;
}

}  // namespace engine::models::xtts_v2
