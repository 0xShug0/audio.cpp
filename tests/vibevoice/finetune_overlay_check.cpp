// Verifies the VibeVoice fine-tune overlay: the diffusion head + connectors fully replace their
// base tensors and the LM LoRA changes the decoder linears. Run with the base 7B model and the
// mp1 adapter (defaults below), or pass <model_dir> <adapter_dir>.
#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/assets/torch_bin.h"
#include "engine/framework/debug/trace.h"
#include "engine/models/vibevoice/assets.h"
#include "engine/models/vibevoice/lora.h"

#include <cmath>
#include <cstdio>
#include <filesystem>
#include <string>
#include <vector>

namespace {

void print8(const char * label, const std::vector<float> & values) {
    std::printf("%-26s", label);
    for (int i = 0; i < 8 && i < static_cast<int>(values.size()); ++i) {
        std::printf(" % .6f", values[i]);
    }
    std::printf("\n");
}

bool equal8(const std::vector<float> & a, const std::vector<float> & b) {
    for (int i = 0; i < 8; ++i) {
        if (std::fabs(a[i] - b[i]) > 1.0e-6F * (1.0F + std::fabs(b[i]))) {
            return false;
        }
    }
    return true;
}

bool differ8(const std::vector<float> & a, const std::vector<float> & b) {
    for (int i = 0; i < 8; ++i) {
        if (std::fabs(a[i] - b[i]) > 1.0e-7F) {
            return true;
        }
    }
    return false;
}

}  // namespace

int main(int argc, char ** argv) {
    engine::debug::configure_logging(engine::debug::LoggingConfig{true, std::nullopt});

    const std::filesystem::path model = argc > 1 ? argv[1] : "models/VibeVoice-7B";
    const std::filesystem::path adapter =
        argc > 2 ? argv[2] : "E:/REPOS/VIBEVOICE-MAIN/VibeVoice/loras/mp1";

    const auto assets = engine::models::vibevoice::load_vibevoice_assets(model);
    const auto base = assets->model_weights;
    const auto overlay = engine::models::vibevoice::make_vibevoice_finetune_source(base, adapter, -1.0F);

    bool ok = true;

    const auto acoustic = engine::assets::open_torch_bin_tensor_source(
        adapter / "acoustic_connector" / "pytorch_model.bin");
    const auto acoustic_ref = acoustic->require_f32("fc1.weight");
    const auto acoustic_overlay = overlay->require_f32("model.acoustic_connector.fc1.weight");
    print8("acoustic.fc1 adapter", acoustic_ref);
    print8("acoustic.fc1 overlay", acoustic_overlay);
    const bool a_ok = equal8(acoustic_ref, acoustic_overlay);
    std::printf("(a) acoustic connector override: %s\n\n", a_ok ? "PASS" : "FAIL");
    ok = ok && a_ok;

    const auto head = engine::assets::open_tensor_source(adapter / "diffusion_head" / "model.safetensors");
    const auto head_ref = head->require_f32("cond_proj.weight");
    const auto head_overlay = overlay->require_f32("model.prediction_head.cond_proj.weight");
    print8("dhead.cond_proj adapter", head_ref);
    print8("dhead.cond_proj overlay", head_overlay);
    const bool b_ok = equal8(head_ref, head_overlay);
    std::printf("(b) diffusion head override: %s\n\n", b_ok ? "PASS" : "FAIL");
    ok = ok && b_ok;

    const std::string lora_weight = "model.language_model.layers.0.mlp.gate_proj.weight";
    const auto lora_base = base->require_f32(lora_weight);
    const auto lora_overlay = overlay->require_f32(lora_weight);
    print8("gate_proj base", lora_base);
    print8("gate_proj overlay", lora_overlay);
    const bool c_ok = differ8(lora_base, lora_overlay);
    std::printf("(c) LM LoRA merge changed weights: %s\n\n", c_ok ? "PASS" : "FAIL");
    ok = ok && c_ok;

    std::printf("=== OVERLAY CHECK %s ===\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
