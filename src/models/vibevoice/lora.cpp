#include "engine/models/vibevoice/lora.h"

#include "engine/framework/assets/torch_bin.h"
#include "engine/framework/assets/lora_tensor_source.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/debug/trace.h"
#include "engine/framework/io/filesystem.h"
#include "engine/framework/io/json.h"
#include "engine/framework/runtime/options.h"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace engine::models::vibevoice {
namespace json = engine::io::json;
namespace {

constexpr std::string_view kLoraWeightsFile = "adapter_model.safetensors";
constexpr std::string_view kLoraConfigFile = "adapter_config.json";
constexpr std::string_view kPeftPrefix = "base_model.model.";
constexpr std::string_view kLanguageModelPrefix = "model.language_model.";
constexpr std::string_view kLoraASuffix = ".lora_A.weight";
constexpr std::string_view kLoraBSuffix = ".lora_B.weight";
constexpr std::string_view kPredictionHeadPrefix = "model.prediction_head.";
constexpr std::string_view kAcousticConnectorPrefix = "model.acoustic_connector.";
constexpr std::string_view kSemanticConnectorPrefix = "model.semantic_connector.";

void log_line(const std::string & message, engine::debug::LogLevel level = engine::debug::LogLevel::Info) {
    engine::debug::log_message(level, "vibevoice", message);
}

struct LoraAdapterPaths {
    std::filesystem::path weights;
    std::optional<std::filesystem::path> config;
};

using assets::LoraTensorDelta;
using OverrideTensor = assets::TensorOverride;

struct LoraTensorNames {
    std::optional<std::string> a_name;
    std::optional<std::string> b_name;
};

bool has_suffix(const std::string & name, std::string_view suffix) {
    return name.size() >= suffix.size() &&
        name.compare(name.size() - suffix.size(), suffix.size(), suffix) == 0;
}

std::string strip_prefix(const std::string & name, std::string_view prefix) {
    if (name.rfind(prefix, 0) == 0) {
        return name.substr(prefix.size());
    }
    return name;
}

LoraAdapterPaths resolve_adapter_paths(const std::filesystem::path & adapter_path) {
    LoraAdapterPaths paths;
    if (engine::io::is_existing_directory(adapter_path)) {
        paths.weights = adapter_path / kLoraWeightsFile;
        const auto config = adapter_path / kLoraConfigFile;
        if (engine::io::is_existing_file(config)) {
            paths.config = config;
        }
    } else if (engine::io::is_existing_file(adapter_path)) {
        paths.weights = adapter_path;
        const auto config = adapter_path.parent_path() / kLoraConfigFile;
        if (engine::io::is_existing_file(config)) {
            paths.config = config;
        }
    } else {
        throw std::runtime_error("VibeVoice LoRA path does not exist: " + adapter_path.string());
    }
    if (!engine::io::is_existing_file(paths.weights)) {
        throw std::runtime_error("VibeVoice LoRA adapter weights not found: " + paths.weights.string());
    }
    return paths;
}

std::filesystem::path resolve_adapter_dir(const std::filesystem::path & adapter_path) {
    if (engine::io::is_existing_directory(adapter_path)) {
        return adapter_path;
    }
    return adapter_path.parent_path();
}

float resolve_adapter_scale(const std::optional<std::filesystem::path> & config_path, float scale_override) {
    if (scale_override > 0.0F) {
        return scale_override;
    }
    if (!config_path.has_value()) {
        throw std::runtime_error(
            "VibeVoice LoRA has no adapter_config.json; pass vibevoice.lora_scale to set the merge scale");
    }
    const auto root = json::parse_file(*config_path);
    const auto rank = json::require_i64(root, "r");
    if (rank <= 0) {
        throw std::runtime_error("VibeVoice LoRA adapter_config.json has non-positive r");
    }
    const float alpha = json::optional_f32(root, "lora_alpha", static_cast<float>(rank));
    const bool use_rslora = json::optional_bool(root, "use_rslora", false);
    const float denom = use_rslora ? std::sqrt(static_cast<float>(rank)) : static_cast<float>(rank);
    return alpha / denom;
}

std::string resolve_base_weight_name(const assets::TensorSource & base, const std::string & module_path) {
    const std::string prefixed = std::string(kLanguageModelPrefix) + module_path + ".weight";
    if (base.has_tensor(prefixed)) {
        return prefixed;
    }
    const std::string bare = module_path + ".weight";
    if (base.has_tensor(bare)) {
        return bare;
    }
    throw std::runtime_error("VibeVoice LoRA targets a module with no matching base weight: " + module_path);
}

std::unordered_map<std::string, LoraTensorNames> collect_lora_tensor_names(const assets::TensorSource & adapter) {
    std::unordered_map<std::string, LoraTensorNames> names;
    for (const auto & metadata : adapter.tensors()) {
        const std::string stripped = strip_prefix(metadata.name, kPeftPrefix);
        if (has_suffix(stripped, kLoraASuffix)) {
            names[stripped.substr(0, stripped.size() - kLoraASuffix.size())].a_name = metadata.name;
        } else if (has_suffix(stripped, kLoraBSuffix)) {
            names[stripped.substr(0, stripped.size() - kLoraBSuffix.size())].b_name = metadata.name;
        }
    }
    return names;
}

// Opens the diffusion-head fine-tune weights, preferring safetensors over the pickled variants.
std::shared_ptr<const assets::TensorSource> open_diffusion_head_source(const std::filesystem::path & adapter_dir) {
    const auto safetensors = adapter_dir / "diffusion_head" / "model.safetensors";
    if (engine::io::is_existing_file(safetensors)) {
        return assets::open_tensor_source(safetensors);
    }
    const auto bin = adapter_dir / "diffusion_head_full.bin";
    if (engine::io::is_existing_file(bin)) {
        return assets::open_torch_bin_tensor_source(bin);
    }
    const auto nested_bin = adapter_dir / "diffusion_head" / "diffusion_head_full.bin";
    if (engine::io::is_existing_file(nested_bin)) {
        return assets::open_torch_bin_tensor_source(nested_bin);
    }
    return nullptr;
}

// Registers full-weight overrides for every adapter tensor that has a matching base tensor. A shape
// mismatch is a hard error (wrong model size); a missing base tensor is skipped like strict=False.
int add_full_overrides(
    const assets::TensorSource & base,
    const assets::TensorSource & source,
    std::string_view base_prefix,
    std::unordered_map<std::string, OverrideTensor> & overrides) {
    int count = 0;
    for (const auto & metadata : source.tensors()) {
        const std::string base_name = std::string(base_prefix) + metadata.name;
        if (!base.has_tensor(base_name)) {
            log_line("skip " + base_name + " (no matching base tensor)", engine::debug::LogLevel::Warning);
            continue;
        }
        if (base.require_metadata(base_name).shape != metadata.shape) {
            throw std::runtime_error(
                "VibeVoice fine-tune shape mismatch for " + base_name +
                " (adapter is trained for a different model size)");
        }
        OverrideTensor override_tensor;
        override_tensor.shape = metadata.shape;
        override_tensor.values = source.require_f32(metadata.name);
        overrides[base_name] = std::move(override_tensor);
        ++count;
    }
    return count;
}

void add_connector_overrides(
    const assets::TensorSource & base,
    const std::filesystem::path & adapter_dir,
    const char * subdir,
    std::string_view base_prefix,
    const char * label,
    std::unordered_map<std::string, OverrideTensor> & overrides) {
    const auto path = adapter_dir / subdir / "pytorch_model.bin";
    if (!engine::io::is_existing_file(path)) {
        log_line(std::string(label) + ": not present, skipped");
        return;
    }
    const auto source = assets::open_torch_bin_tensor_source(path);
    const int count = add_full_overrides(base, *source, base_prefix, overrides);
    log_line(std::string(label) + ": overrode " + std::to_string(count) + " tensors");
}

}  // namespace

std::shared_ptr<const assets::TensorSource> make_vibevoice_finetune_source(
    std::shared_ptr<const assets::TensorSource> base,
    const std::filesystem::path & adapter_path,
    float scale_override) {
    if (base == nullptr) {
        throw std::runtime_error("VibeVoice LoRA merge requires a base tensor source");
    }
    const auto prepare_started = std::chrono::steady_clock::now();
    const auto paths = resolve_adapter_paths(adapter_path);
    const float scale = resolve_adapter_scale(paths.config, scale_override);
    const auto adapter_open_started = std::chrono::steady_clock::now();
    const auto adapter = assets::open_tensor_source(paths.weights);
    engine::debug::timing_log_scalar("vibevoice.lora.adapter_open_ms", engine::debug::elapsed_ms(adapter_open_started));

    const auto tensor_names = collect_lora_tensor_names(*adapter);
    if (tensor_names.empty()) {
        throw std::runtime_error("VibeVoice LoRA adapter contains no lora_A/lora_B tensors: " + paths.weights.string());
    }

    std::unordered_map<std::string, LoraTensorDelta> deltas;
    deltas.reserve(tensor_names.size());
    const auto delta_load_started = std::chrono::steady_clock::now();
    for (const auto & [module_path, names] : tensor_names) {
        const std::string base_name = resolve_base_weight_name(*base, module_path);
        if (!names.a_name.has_value() || !names.b_name.has_value()) {
            throw std::runtime_error("VibeVoice LoRA module is missing an A/B pair: " + module_path);
        }
        deltas.emplace(base_name, assets::load_lora_tensor_delta(
            *base, *adapter, base_name, *names.a_name, *names.b_name, scale));
    }
    engine::debug::timing_log_scalar("vibevoice.lora.delta_load_ms", engine::debug::elapsed_ms(delta_load_started));

    log_line("applying fine-tune adapter: " + adapter_path.string());
    log_line("LM LoRA: merged " + std::to_string(deltas.size()) + " decoder modules (scale " +
        std::to_string(scale) + ")");

    const auto adapter_dir = resolve_adapter_dir(adapter_path);
    std::unordered_map<std::string, OverrideTensor> overrides;
    if (const auto head = open_diffusion_head_source(adapter_dir)) {
        const auto override_started = std::chrono::steady_clock::now();
        const int count = add_full_overrides(*base, *head, kPredictionHeadPrefix, overrides);
        engine::debug::timing_log_scalar(
            "vibevoice.lora.diffusion_head_override_load_ms",
            engine::debug::elapsed_ms(override_started));
        log_line("diffusion head: overrode " + std::to_string(count) + " tensors");
    } else {
        log_line("diffusion head: not present, skipped");
    }
    const auto acoustic_started = std::chrono::steady_clock::now();
    add_connector_overrides(*base, adapter_dir, "acoustic_connector", kAcousticConnectorPrefix,
        "acoustic connector", overrides);
    engine::debug::timing_log_scalar(
        "vibevoice.lora.acoustic_connector_override_load_ms",
        engine::debug::elapsed_ms(acoustic_started));
    const auto semantic_started = std::chrono::steady_clock::now();
    add_connector_overrides(*base, adapter_dir, "semantic_connector", kSemanticConnectorPrefix,
        "semantic connector", overrides);
    engine::debug::timing_log_scalar(
        "vibevoice.lora.semantic_connector_override_load_ms",
        engine::debug::elapsed_ms(semantic_started));
    engine::debug::timing_log_scalar("vibevoice.lora.prepare_ms", engine::debug::elapsed_ms(prepare_started));
    engine::debug::timing_log_scalar("vibevoice.lora.decoder_delta_tensors", deltas.size());
    engine::debug::timing_log_scalar("vibevoice.lora.full_override_tensors", overrides.size());

    return assets::make_lora_tensor_source(
        std::move(base), std::move(deltas), std::move(overrides), "vibevoice.lora");
}

std::shared_ptr<const VibeVoiceAssets> apply_vibevoice_finetune_options(
    std::shared_ptr<const VibeVoiceAssets> assets,
    const std::unordered_map<std::string, std::string> & options) {
    const auto lora_path = runtime::find_option(options, {"vibevoice.lora"});
    if (!lora_path.has_value() || lora_path->empty()) {
        return assets;
    }
    if (assets == nullptr) {
        throw std::runtime_error("VibeVoice fine-tune requires assets");
    }
    if (assets->fine_tune_applied) {
        throw std::runtime_error(
            "VibeVoice LoRA already applied via --load-option; do not also pass it via --session-option");
    }
    float scale_override = -1.0F;
    if (const auto value = runtime::parse_finite_float_option(options, {"vibevoice.lora_scale"})) {
        if (*value <= 0.0F) {
            throw std::runtime_error("VibeVoice vibevoice.lora_scale must be positive");
        }
        scale_override = *value;
    }
    auto updated = std::make_shared<VibeVoiceAssets>(*assets);
    updated->model_weights = make_vibevoice_finetune_source(assets->model_weights, *lora_path, scale_override);
    updated->fine_tune_applied = true;
    return updated;
}

}  // namespace engine::models::vibevoice
