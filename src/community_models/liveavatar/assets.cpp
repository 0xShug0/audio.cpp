#include "engine/community_models/liveavatar/assets.h"

#include "engine/framework/runtime/options.h"

#include <filesystem>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>

namespace engine::community_models::liveavatar {
namespace {

constexpr const char * kFamily = "liveavatar";
constexpr const char * kDefaultDenoiserGguf = "Wan2.2-S2V-14B-NVFP4-LORA.gguf";
constexpr const char * kDefaultSupportGguf = "Wan2.2-S2V-Support-Q4_K_S-F16.gguf";
constexpr const char * kDefaultVaeGguf = "Wan2.2-S2V-VAE-F16.gguf";
constexpr const char * kTokenizerSidecar = "tokenizer/spiece.model";

std::filesystem::path require_file(const std::filesystem::path & root, const std::filesystem::path & relative) {
    const auto path = root / relative;
    if (!std::filesystem::is_regular_file(path)) {
        throw std::runtime_error(std::string(kFamily) + " required file does not exist: " + path.string());
    }
    return path;
}

void require_gguf_path(std::string_view option_name, const std::filesystem::path & path) {
    if (path.extension() != ".gguf") {
        throw std::runtime_error(std::string(option_name) + " must point to a GGUF file");
    }
}

std::filesystem::path require_file_path(const std::filesystem::path & path) {
    if (!std::filesystem::is_regular_file(path)) {
        throw std::runtime_error(std::string(kFamily) + " required file does not exist: " + path.string());
    }
    return path;
}

std::filesystem::path resolve_component_path(
    const std::filesystem::path & root,
    const std::unordered_map<std::string, std::string> & options,
    std::initializer_list<std::string_view> option_names,
    std::string_view canonical_option_name,
    const char * default_file) {
    const auto option = engine::runtime::find_option(options, option_names);
    if (!option.has_value()) {
        const auto path = require_file(root, default_file);
        require_gguf_path(canonical_option_name, path);
        return path;
    }

    const std::filesystem::path requested(*option);
    const auto path = requested.is_absolute()
        ? require_file_path(requested)
        : require_file(root, requested);
    require_gguf_path(canonical_option_name, path);
    return path;
}

std::filesystem::path resolve_denoiser_path(
    const std::filesystem::path & root,
    const std::unordered_map<std::string, std::string> & options,
    const std::optional<std::filesystem::path> & standalone_gguf) {
    if (!engine::runtime::find_option(options, {"denoiser_gguf", "liveavatar.denoiser_gguf"}).has_value() &&
        standalone_gguf.has_value()) {
        require_gguf_path("liveavatar.denoiser_gguf", *standalone_gguf);
        return require_file_path(*standalone_gguf);
    }
    return resolve_component_path(
        root,
        options,
        {"denoiser_gguf", "liveavatar.denoiser_gguf"},
        "liveavatar.denoiser_gguf",
        kDefaultDenoiserGguf);
}

std::filesystem::path resolve_support_path(
    const std::filesystem::path & root,
    const std::unordered_map<std::string, std::string> & options) {
    return resolve_component_path(
        root,
        options,
        {"support_gguf", "liveavatar.support_gguf"},
        "liveavatar.support_gguf",
        kDefaultSupportGguf);
}

std::filesystem::path resolve_vae_path(
    const std::filesystem::path & root,
    const std::unordered_map<std::string, std::string> & options) {
    return resolve_component_path(
        root,
        options,
        {"vae_gguf", "liveavatar.vae_gguf"},
        "liveavatar.vae_gguf",
        kDefaultVaeGguf);
}

void validate_assets(const LiveAvatarAssets & assets) {
    if (assets.tokenizer_pieces.empty()) {
        throw std::runtime_error(std::string(kFamily) + " tokenizer is empty");
    }
    assets.text_encoder_weights->require_metadata("token_embd.weight");
    assets.denoiser_weights->require_metadata("patch_embedding.weight");
    assets.denoiser_weights->require_metadata("cond_encoder.weight");
    assets.denoiser_weights->require_metadata("blocks.0.self_attn.q.weight");
    assets.denoiser_weights->require_metadata("blocks.0.cross_attn.k.weight");
    assets.denoiser_weights->require_metadata("casual_audio_encoder.weights");
    assets.denoiser_weights->require_metadata("audio_injector.injector.0.q.weight");
    assets.denoiser_weights->require_metadata("head.head.weight");
    assets.audio_encoder_weights->require_metadata("wav2vec2.feature_extractor.conv_layers.0.conv.weight");
    assets.audio_encoder_weights->require_metadata("wav2vec2.encoder.layers.0.attention.q_proj.weight");
    assets.vae_weights->require_metadata("encoder.conv1.weight");
    assets.vae_weights->require_metadata("decoder.head.2.weight");
}

}  // namespace

std::shared_ptr<const LiveAvatarAssets>
load_liveavatar_assets(
    const std::filesystem::path & model_path,
    const std::unordered_map<std::string, std::string> & options) {
    auto assets = std::make_shared<LiveAvatarAssets>();
    const auto prepared = engine::assets::prepare_model_directory(model_path, kDefaultDenoiserGguf);
    assets->model_root = prepared.model_root;
    const auto support_path = resolve_support_path(assets->model_root, options);
    const auto vae_path = resolve_vae_path(assets->model_root, options);
    const auto sidecar_root = engine::assets::materialize_gguf_sidecars(support_path);
    assets->tokenizer_model = require_file(sidecar_root, kTokenizerSidecar);
    assets->tokenizer_pieces = engine::tokenizers::load_sentencepiece_model(assets->tokenizer_model);
    assets->support_weights = engine::assets::open_tensor_source(support_path);
    assets->text_encoder_weights = engine::assets::make_prefixed_tensor_source(assets->support_weights, "text_encoder");
    assets->audio_encoder_weights = engine::assets::make_prefixed_tensor_source(assets->support_weights, "audio_encoder");
    assets->vae_weights = engine::assets::make_prefixed_tensor_source(
        engine::assets::open_tensor_source(vae_path), "vae");
    assets->denoiser_weights = engine::assets::open_tensor_source(
        resolve_denoiser_path(assets->model_root, options, prepared.standalone_gguf));
    validate_assets(*assets);
    return assets;
}

}  // namespace engine::community_models::liveavatar
