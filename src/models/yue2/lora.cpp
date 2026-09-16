#include "engine/models/yue2/assets.h"

#include "engine/framework/assets/lora_tensor_source.h"
#include "engine/framework/debug/trace.h"

#include <cmath>
#include <unordered_set>

namespace engine::models::yue2 {

std::shared_ptr<const assets::TensorSource> make_yue2_lora_source(
    std::shared_ptr<const assets::TensorSource> base,
    const std::filesystem::path & adapter_path, float scale, int64_t layer_count) {
    if (!std::isfinite(scale)) throw std::runtime_error("yue2.lora_scale must be finite");
    if (adapter_path.extension() != ".safetensors") {
        throw std::runtime_error("yue2.lora must be an AR adapter safetensors file (unfused A/B layout)");
    }
    const auto adapter = assets::open_tensor_source(adapter_path);
    std::unordered_set<std::string> consumed;
    std::unordered_map<std::string, assets::LoraTensorDelta> deltas;
    for (int64_t layer = 0; layer < layer_count; ++layer) {
        for (const auto * projection : {"self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
                                      "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"}) {
            const auto prefix = "layers." + std::to_string(layer) + "." + projection;
            const auto a = prefix + ".lora_A";
            const auto b = prefix + ".lora_B";
            if (!adapter->has_tensor(a) && !adapter->has_tensor(b)) continue;
            if (!adapter->has_tensor(a) || !adapter->has_tensor(b)) {
                throw std::runtime_error("YuE2 LoRA is missing an A/B pair: " + prefix);
            }
            const auto name = "model." + prefix + ".weight";
            auto delta = assets::load_lora_tensor_delta(*base, *adapter, name, a, b, scale);
            delta.merge_mode = assets::LoraMergeMode::RoundedBF16Delta;
            deltas.emplace(name, std::move(delta));
            consumed.insert(a);
            consumed.insert(b);
        }
    }
    for (const auto & tensor : adapter->tensors()) {
        if (!consumed.count(tensor.name)) {
            throw std::runtime_error("Unsupported YuE2 AR LoRA tensor: " + tensor.name +
                                     " (use the unfused AR adapter, not ComfyUI or NAR weights)");
        }
    }
    if (deltas.empty()) throw std::runtime_error("YuE2 LoRA contains no AR A/B pairs");
    engine::debug::log_message(engine::debug::LogLevel::Info, "yue2",
        "AR LoRA: " + adapter_path.string() + ", projections=" + std::to_string(deltas.size()) +
        ", scale=" + std::to_string(scale));
    if (scale == 0.0F) return base;
    return assets::make_lora_tensor_source(std::move(base), std::move(deltas), {}, "yue2.lora", true);
}

}  // namespace engine::models::yue2
