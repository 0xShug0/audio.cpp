#include "engine/framework/assets/weight_adapter.h"
#include "engine/framework/debug/trace.h"
#include "engine/framework/io/safetensors.h"
#include "test_assert.h"

#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>

namespace {
using namespace engine;
using test::require;

io::SafeTensorWriteEntry tensor(const std::string & name, std::vector<int64_t> shape,
                                 const std::vector<float> & values) {
    std::vector<unsigned char> data(values.size() * sizeof(float));
    std::memcpy(data.data(), values.data(), data.size());
    return {name, "F32", std::move(shape), std::move(data)};
}

template<class F>
void rejects(F && fn, const std::string & message) {
    try { fn(); } catch (const std::runtime_error &) { return; }
    throw std::runtime_error(message);
}

void run(const std::filesystem::path & root) {
    const auto base_path = root / "base.safetensors";
    const std::vector<float> weights(12, 1.0F);
    io::write_safetensors_file(base_path, {
        tensor("model.layers.0.proj.weight", {3, 4}, weights),
        tensor("model.q.weight", {1, 4}, {1, 1, 1, 1}),
        tensor("model.k.weight", {2, 4}, std::vector<float>(8, 1)),
    });
    const auto base = assets::open_tensor_source(base_path);
    assets::WeightAdapterBindings bindings;
    bindings.add("layers.0.proj", "model.layers.0.proj.weight");
    bindings.add("qkv", std::vector<assets::WeightAdapterTarget>{
        {"model.q.weight", 0, 1}, {"model.k.weight", 1, 2}});
    const std::vector<float> a{1, 2, 3, 4, -2, -1, 0, 1};
    const std::vector<float> b{1, 0, 0, 1, 1, 1};
    const auto native_path = root / "native.safetensors";
    const auto comfy_path = root / "comfy.safetensors";
    io::write_safetensors_file(native_path, {
        tensor("layers.0.proj.lora_A", {2, 4}, a),
        tensor("layers.0.proj.lora_B", {3, 2}, b),
    });
    io::write_safetensors_file(comfy_path, {
        tensor("lora_unet_layers_0_proj.lora_down.weight", {2, 4}, a),
        tensor("lora_unet_layers_0_proj.lora_up.weight", {3, 2}, b),
        tensor("lora_unet_layers_0_proj.alpha", {}, {2}),
    });
    const auto materialize = [&](const std::filesystem::path & path,
                                 assets::WeightAdapterReadOptions options = {}) {
        return assets::make_weight_adapter_source(base, assets::bind_weight_adapter(
            *base, assets::read_weight_adapter(path, options), bindings, assets::WeightMergeMode::DeltaF32));
    };
    const auto native = materialize(native_path)->require_f32("model.layers.0.proj.weight");
    require(native == materialize(comfy_path)->require_f32("model.layers.0.proj.weight"),
            "ComfyUI alpha/r mapping differs from native A/B");
    assets::WeightAdapterReadOptions embedded_scale;
    embedded_scale.require_scaling_metadata = true;
    require(native == materialize(comfy_path, embedded_scale)->require_f32("model.layers.0.proj.weight"),
            "per-tensor alpha must not require a PEFT config file");
    require(native == std::vector<float>({2, 3, 4, 5, -1, 0, 1, 2, 0, 2, 4, 6}), "LoRA merge math differs");

    const auto peft_dir = root / "peft";
    std::filesystem::create_directories(peft_dir);
    io::write_safetensors_file(peft_dir / "adapter_model.safetensors", {
        tensor("base_model.model.layers.0.proj.lora_A.weight", {2, 4}, a),
        tensor("base_model.model.layers.0.proj.lora_B.weight", {3, 2}, b),
    });
    std::ofstream(peft_dir / "adapter_config.json") << R"({"r":2,"lora_alpha":2,"use_rslora":false})";
    require(native == materialize(peft_dir)->require_f32("model.layers.0.proj.weight"), "PEFT import differs");
    std::ofstream(peft_dir / "adapter_config.json") << R"({"r":2,"lora_alpha":2,"use_rslora":true})";
    const auto rs = assets::read_weight_adapter(peft_dir);
    require(std::abs(rs.updates.begin()->second.scale - std::sqrt(2.0F)) < 1e-6F, "rsLoRA scale differs");
    assets::WeightAdapterReadOptions scale;
    scale.strength = 0.5F;
    scale.scale_override = 3.0F;
    require(assets::read_weight_adapter(peft_dir, scale).updates.begin()->second.scale == 1.5F,
            "explicit scale was multiplied by metadata twice");

    const auto fused = root / "fused.safetensors";
    io::write_safetensors_file(fused, {
        tensor("qkv.lora_down.weight", {2, 4}, a),
        tensor("qkv.lora_up.weight", {3, 2}, b),
    });
    const auto split = materialize(fused);
    require(split->require_f32("model.q.weight") == std::vector<float>({2, 3, 4, 5}), "fused Q slice differs");
    require(split->require_f32("model.k.weight") == std::vector<float>({-1, 0, 1, 2, 0, 2, 4, 6}), "fused K slice differs");

    const auto kron_path = root / "kron.safetensors";
    io::write_safetensors_file(kron_path, {
        tensor("lycoris_layers_0_proj.lokr_w1", {3, 2}, {1, 2, 3, 4, 5, 6}),
        tensor("lycoris_layers_0_proj.lokr_w2", {1, 2}, {-1, 0.5F}),
        tensor("lycoris_layers_0_proj.alpha", {}, {100}),
    });
    const auto kron = materialize(kron_path)->require_f32("model.layers.0.proj.weight");
    require(kron == std::vector<float>({0, 1.5F, -1, 2, -2, 2.5F, -3, 3, -4, 3.5F, -5, 4}),
            "full-matrix LoKr shape, scaling, or Kronecker order differs");
    io::write_safetensors_file(kron_path, {
        tensor("layers.0.proj.lokr_w1_a", {3, 1}, {1, 3, 5}),
        tensor("layers.0.proj.lokr_w1_b", {1, 2}, {1, 2}),
        tensor("layers.0.proj.lokr_w2", {1, 2}, {-1, 0.5F}),
        tensor("layers.0.proj.alpha", {}, {0.5F}),
    });
    const auto decomposed = materialize(kron_path)->require_f32("model.layers.0.proj.weight");
    require(decomposed == std::vector<float>({0.5F, 1.25F, 0, 1.5F, -0.5F, 1.75F, -2, 2.5F, -1.5F, 2.25F, -4, 3.5F}),
            "decomposed LoKr reconstruction differs");

    scale.strength = 0;
    const auto disabled = materialize(native_path, scale);
    require(disabled->require_f32("model.layers.0.proj.weight") == weights, "zero adapter changed weights");
    require(base->require_f32("model.layers.0.proj.weight") == weights, "base tensor was modified");
    require(assets::make_weight_adapter_source(base, {}).get() == base.get(), "empty plan is not identity");
    rejects([&] { bindings.add("layers_0_proj", "model.q.weight"); }, "ambiguous bindings accepted");
    assets::WeightAdapterBindings missing;
    rejects([&] { assets::bind_weight_adapter(*base, assets::read_weight_adapter(native_path), missing,
        assets::WeightMergeMode::DeltaF32); }, "unbound target accepted");
    const auto bad_path = root / "bad.safetensors";
    io::write_safetensors_file(bad_path, {tensor("layers.0.proj.lora_A", {2, 4}, a)});
    rejects([&] { assets::read_weight_adapter(bad_path); }, "unpaired adapter accepted");
    io::write_safetensors_file(bad_path, {tensor("layers.0.proj.dora_scale", {1}, {1})});
    rejects([&] { assets::read_weight_adapter(bad_path); }, "unsupported DoRA silently ignored");
    assets::WeightAdapterReadOptions needs_scale;
    needs_scale.require_scaling_metadata = true;
    rejects([&] { assets::read_weight_adapter(native_path, needs_scale); }, "required scale metadata ignored");
    std::ofstream(peft_dir / "adapter_config.json") << R"({"r":3,"lora_alpha":2})";
    rejects([&] { assets::read_weight_adapter(peft_dir); }, "incorrect configured rank accepted");

    io::write_safetensors_file(peft_dir / "native.safetensors", {
        tensor("layers.0.proj.lora_A", {7, 4}, std::vector<float>(28, 1)),
        tensor("layers.0.proj.lora_B", {3, 7}, std::vector<float>(21, 1)),
    });
    assets::WeightAdapterReadOptions native_strength;
    native_strength.strength = 0.1F;
    require(assets::read_weight_adapter(peft_dir / "native.safetensors", native_strength)
        .updates.begin()->second.scale == native_strength.strength,
        "native direct scale was rounded through alpha/r or used adjacent PEFT config");
    io::write_safetensors_file(peft_dir / "comfy.safetensors", {
        tensor("layers.0.proj.lora_down.weight", {2, 4}, a),
        tensor("layers.0.proj.lora_up.weight", {3, 2}, b),
        tensor("layers.0.proj.alpha", {}, {2}),
    });
    require(materialize(peft_dir / "comfy.safetensors")->require_f32("model.layers.0.proj.weight") == native,
            "ComfyUI factors used adjacent PEFT rank/alpha metadata");
}
}  // namespace

int main(int argc, char ** argv) {
    const auto root = std::filesystem::temp_directory_path() / "audiocpp_weight_adapter_test";
    try {
        debug::configure_logging({argc > 1 && std::string(argv[1]) == "--log", std::nullopt});
        std::filesystem::create_directories(root);
        run(root);
        std::filesystem::remove_all(root);
        std::cout << "weight_adapter_test passed\n";
        return 0;
    } catch (const std::exception & e) {
        std::filesystem::remove_all(root);
        std::cerr << e.what() << '\n';
        return 1;
    }
}
