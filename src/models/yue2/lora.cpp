#include "engine/models/yue2/assets.h"

#include "engine/framework/assets/lora_tensor_source.h"
#include "engine/framework/debug/trace.h"

#include <cmath>
#include <unordered_set>

namespace engine::models::yue2 {

std::shared_ptr<const assets::TensorSource> make_yue2_lora_source(
    std::shared_ptr<const assets::TensorSource> base,
    const std::filesystem::path & adapter_path, float scale, int64_t layer_count,
    const std::filesystem::path & nar_adapter_path, float nar_scale) {
    std::unordered_map<std::string, assets::LoraTensorDelta> deltas;
    std::unordered_map<std::string, assets::TensorOverride> overrides;
    for (const bool nar : {false, true}) {
        const auto & path = nar ? nar_adapter_path : adapter_path;
        if (path.empty()) continue;
        const float strength = nar ? nar_scale : scale;
        const std::string option = nar ? "yue2.nar_lora" : "yue2.ar_lora";
        if (!std::isfinite(strength)) throw std::runtime_error(option + "_scale must be finite");
        if (path.extension() != ".safetensors") {
            throw std::runtime_error(option + " must be an unfused adapter safetensors file");
        }
        const auto adapter = assets::open_tensor_source(path);
        std::unordered_set<std::string> consumed;
        size_t projections = 0;
        for (int64_t layer = 0; layer < layer_count; ++layer) {
            for (const auto * projection : {"self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
                                            "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"}) {
                const auto prefix = "layers." + std::to_string(layer) + "." + (nar ? "nar_" : "") + projection;
                const auto a = prefix + ".lora_A";
                const auto b = prefix + ".lora_B";
                if (!adapter->has_tensor(a) && !adapter->has_tensor(b)) continue;
                if (!adapter->has_tensor(a) || !adapter->has_tensor(b)) {
                    throw std::runtime_error("YuE2 LoRA is missing an A/B pair: " + prefix);
                }
                const auto name = "model." + prefix + ".weight";
                auto delta = assets::load_lora_tensor_delta(*base, *adapter, name, a, b, strength);
                delta.merge_mode = assets::LoraMergeMode::RoundedBF16Delta;
                if (strength != 0.0F) deltas.emplace(name, std::move(delta));
                ++projections;
                consumed.insert(a);
                consumed.insert(b);
            }
        }
        if (nar) {
            for (const auto * name : {"vae2llm.weight", "vae2llm.bias", "llm2vae.weight", "llm2vae.bias"}) {
                if (!adapter->has_tensor(name)) continue;
                const auto shape = base->require_metadata(name).shape;
                auto values = adapter->require_f32(name, shape);
                // The reference loads these full replacements into BF16 Linear layers.
                for (auto & value : values)
                    value = ggml_bf16_to_fp32(ggml_fp32_to_bf16(value));
                if (strength != 0.0F) overrides.emplace(name, assets::TensorOverride{shape, std::move(values)});
                consumed.insert(name);
            }
        }
        for (const auto & tensor : adapter->tensors()) {
            if (!consumed.count(tensor.name)) {
                throw std::runtime_error("Unsupported " + option + " tensor: " + tensor.name +
                                         " (use the matching unfused adapter, not ComfyUI weights)");
            }
        }
        if (projections == 0) throw std::runtime_error(option + " contains no matching A/B pairs");
        engine::debug::log_message(engine::debug::LogLevel::Info, "yue2",
            option + ": " + path.string() + ", projections=" + std::to_string(projections) +
            ", scale=" + std::to_string(strength));
    }
    if (deltas.empty() && overrides.empty()) return base;
    return assets::make_lora_tensor_source(std::move(base), std::move(deltas), std::move(overrides), "yue2.lora", true);
}

}  // namespace engine::models::yue2
