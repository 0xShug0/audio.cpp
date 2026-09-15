#include "engine/models/yue2/session.h"

#include "engine/framework/debug/profiler.h"
#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/io/filesystem.h"
#include "engine/framework/runtime/options.h"

#include <chrono>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>

namespace engine::models::yue2 {
namespace {

using Clock = std::chrono::steady_clock;
constexpr const char * kFamily = "yue2";

std::shared_ptr<const Yue2Assets> require_assets(std::shared_ptr<const Yue2Assets> assets) {
    if (assets == nullptr) {
        throw std::runtime_error("Yue2 session requires assets");
    }
    return assets;
}

engine::assets::TensorStorageType parse_weight_type(
    const runtime::SessionOptions & options,
    const char * key,
    assets::TensorStorageType fallback) {
    return runtime::parse_tensor_storage_option(
        options.options,
        key,
        "yue2.weight_type",
        fallback,
        {
            assets::TensorStorageType::Native,
            assets::TensorStorageType::F32,
            assets::TensorStorageType::F16,
            assets::TensorStorageType::BF16,
            assets::TensorStorageType::Q8_0,
            assets::TensorStorageType::Q4_0,
            assets::TensorStorageType::Q4_K,
        });
}

std::filesystem::path resolve_component_gguf_path(
    const Yue2Assets & assets,
    std::string_view option_name,
    const std::string & value) {
    if (value.empty()) {
        throw std::runtime_error(std::string(option_name) + " must not be empty");
    }
    const std::filesystem::path relative(value);
    if (relative.is_absolute()) {
        throw std::runtime_error(std::string(option_name) + " must be relative to the Yue2 model root");
    }
    const auto path = assets.model_root / relative;
    if (!engine::io::is_existing_file(path)) {
        throw std::runtime_error(std::string(option_name) + " file does not exist: " + path.string());
    }
    if (path.extension() != ".gguf") {
        throw std::runtime_error(std::string(option_name) + " must point to a GGUF file");
    }
    return path;
}

void validate_component_anchors(const Yue2Assets & assets) {
    const auto & source = *assets.model_weights;
    assets::require_tensor_shape(
        source,
        "model.embed_tokens.weight",
        {assets.config.model.vocab_size, assets.config.model.hidden_size});
    assets::require_tensor_shape(
        source,
        "model.layers.0.self_attn.q_proj.weight",
        {assets.config.model.attention_heads * assets.config.model.head_dim, assets.config.model.hidden_size});
    assets::require_tensor_shape(
        source,
        "model.layers.0.nar_self_attn.q_proj.weight",
        {assets.config.model.attention_heads * assets.config.model.head_dim, assets.config.model.hidden_size});
    assets::require_tensor_shape(source, "vae2llm.weight", {assets.config.model.hidden_size, assets.config.model.latent_dim});
    assets::require_tensor_shape(source, "llm2vae.weight", {assets.config.model.latent_dim, assets.config.model.hidden_size});
    if (assets.vae_weights->has_tensor("decoder.layers.0.weight")) {
        assets::require_tensor_shape(*assets.vae_weights, "decoder.layers.0.weight", {2048, 64, 7});
    } else {
        assets::require_tensor_shape(*assets.vae_weights, "decoder.layers.0.weight_g", {2048, 1, 1});
    }
}

std::shared_ptr<const Yue2Assets> select_component_assets(
    std::shared_ptr<const Yue2Assets> base,
    const std::unordered_map<std::string, std::string> & options) {
    auto selected = std::make_shared<Yue2Assets>(*base);
    const std::string model_gguf =
        runtime::find_option(options, {"yue2.model_gguf"}).value_or("yue2-3b-q8_0.gguf");
    selected->model_weights = assets::open_tensor_source(
        resolve_component_gguf_path(*base, "yue2.model_gguf", model_gguf),
        "model_weights");
    const std::string vae_gguf =
        runtime::find_option(options, {"yue2.vae_gguf"}).value_or("yue2-vae-f16.gguf");
    selected->vae_weights = assets::open_tensor_source(
        resolve_component_gguf_path(*base, "yue2.vae_gguf", vae_gguf),
        "vae_weights");
    validate_component_anchors(*selected);
    return selected;
}

std::unique_ptr<runtime::IVoiceTaskSession> create_yue2_session(
    const runtime::TaskSpec & task,
    const runtime::SessionOptions & options,
    std::shared_ptr<const Yue2Assets> assets) {
    return std::make_unique<Yue2Session>(
        task,
        options,
        std::move(assets));
}

const runtime::ModelMetadata & yue2_metadata() noexcept {
    static const runtime::ModelMetadata metadata{
        kFamily,
        "Yue2",
        "Yue2 music generation with lyrics, style, and optional ABC conditioning.",
        {
            "sidecars/yue2-model-config.json",
            "sidecars/yue2-generation-config.json",
            "sidecars/yue2-qwen.tiktoken",
            "sidecars/yue2-vae-config.json",
        },
        {
            "yue2-3b-q8_0.gguf",
            "yue2-3b-q4_0.gguf",
            "yue2-3b-bf16.gguf",
            "yue2-vae-f16.gguf",
            "yue2-vae-f32.gguf",
        }};
    return metadata;
}

const runtime::CapabilitySet & yue2_capabilities() noexcept {
    static const runtime::CapabilitySet capabilities{
        {runtime::TaskCapability{
            runtime::VoiceTaskKind::AudioGeneration,
            {runtime::RunMode::Offline},
        }},
        {"auto"},
        false,
        true,
        false,
    };
    return capabilities;
}

runtime::ModelCliInterface yue2_cli_interface() {
    runtime::ModelCliInterface out;
    out.request_options = {
        {"style", "string", "Music style prompt.", true},
        {"lyrics", "string", "Lyrics to generate.", false},
        {"abc", "string", "ABC score conditioning text.", false},
        {"abc_file", "path", "Path to ABC score conditioning text.", false},
        {"cot", "off|melody|full", "Planning mode.", false, "off"},
        {"seed", "int", "Generation seed.", false, "0"},
        {"cfg_scale", "float", "Classifier-free guidance scale.", false, "1.0", "0.0", "20.0"},
        {"num_inference_steps", "int", "NAR ODE steps.", false, "10", "1"},
    };
    out.session_options = {
        {"yue2.model_gguf", "string", "Yue2 main AR/NAR component GGUF file relative to the model root.", false, "yue2-3b-q8_0.gguf"},
        {"yue2.vae_gguf", "string", "Yue2 VAE component GGUF file relative to the model root.", false, "yue2-vae-f16.gguf"},
        {"yue2.model_weight_type", "native|f32|f16|bf16|q8_0|q4_0|q4_k", "Yue2 main model weight storage type.", false, "native"},
        {"yue2.vae_weight_type", "native|f32|f16|bf16|q8_0|q4_0|q4_k", "Yue2 VAE weight storage type.", false, "native"},
        {"yue2.model_weight_context_mb", "int", "Yue2 main model weight context size in MiB.", false, "6144", "1"},
        {"yue2.vae_weight_context_mb", "int", "Yue2 VAE weight context size in MiB.", false, "1536", "1"},
        {"yue2.ar_prefill_graph_arena_mb", "int", "AR prefill graph arena size in MiB.", false, "4096", "1"},
        {"yue2.ar_decode_graph_arena_mb", "int", "AR one-token decode graph arena size in MiB.", false, "1536", "1"},
        {"yue2.nar_graph_arena_mb", "int", "NAR acoustic flow graph arena size in MiB.", false, "6144", "1"},
        {"yue2.vae_graph_arena_mb", "int", "VAE decode graph arena size in MiB.", false, "1536", "1"},
    };
    return out;
}

}  // namespace

Yue2Session::Yue2Session(
    runtime::TaskSpec task,
    runtime::SessionOptions options,
    std::shared_ptr<const Yue2Assets> assets)
    : runtime::RuntimeSessionBase(options),
      task_(task),
      assets_(select_component_assets(require_assets(std::move(assets)), options.options)) {
    if (task_.task != runtime::VoiceTaskKind::AudioGeneration || task_.mode != runtime::RunMode::Offline) {
        throw std::runtime_error("Yue2 supports only offline gen/music");
    }
    pipeline_ = std::make_unique<Yue2PipelineRuntime>(
        execution_context(),
        assets_,
        parse_weight_type(options, "yue2.model_weight_type", assets::TensorStorageType::Native),
        parse_weight_type(options, "yue2.vae_weight_type", assets::TensorStorageType::Native),
        runtime::parse_size_mb_option(options.options, {"yue2.model_weight_context_mb"}, 6144ull * 1024ull * 1024ull),
        runtime::parse_size_mb_option(options.options, {"yue2.vae_weight_context_mb"}, 1536ull * 1024ull * 1024ull),
        runtime::parse_size_mb_option(options.options, {"yue2.ar_prefill_graph_arena_mb"}, 4096ull * 1024ull * 1024ull),
        runtime::parse_size_mb_option(options.options, {"yue2.ar_decode_graph_arena_mb"}, 1536ull * 1024ull * 1024ull),
        runtime::parse_size_mb_option(options.options, {"yue2.nar_graph_arena_mb"}, 6144ull * 1024ull * 1024ull),
        runtime::parse_size_mb_option(options.options, {"yue2.vae_graph_arena_mb"}, 1536ull * 1024ull * 1024ull));
}

Yue2Session::~Yue2Session() = default;

std::string Yue2Session::family() const {
    return kFamily;
}

runtime::VoiceTaskKind Yue2Session::task_kind() const {
    return task_.task;
}

runtime::RunMode Yue2Session::run_mode() const {
    return task_.mode;
}

void Yue2Session::prepare(const runtime::SessionPreparationRequest & request) {
    if (request.text.has_value() || !request.options.empty()) {
        (void) parse_yue2_preparation_request(request, assets_->config.generation);
    }
    mark_prepared();
}

runtime::TaskResult Yue2Session::run(const runtime::TaskRequest & request) {
    require_prepared("Yue2 run");
    const auto wall_start = Clock::now();
    const auto parsed = parse_yue2_request(request, assets_->config.generation);
    runtime::TaskResult result;
    result.audio_output = pipeline_->run(parsed);
    engine::debug::timing_log_scalar("session.wall_ms", engine::debug::elapsed_ms(wall_start, Clock::now()));
    return result;
}

std::shared_ptr<runtime::IVoiceModelLoader> make_yue2_loader() {
    class LoadedModel final : public runtime::ILoadedVoiceModel {
    public:
        explicit LoadedModel(std::shared_ptr<const Yue2Assets> assets)
            : assets_(require_assets(std::move(assets))) {}

        const runtime::ModelMetadata & metadata() const noexcept override {
            return yue2_metadata();
        }

        const runtime::CapabilitySet & capabilities() const noexcept override {
            return yue2_capabilities();
        }

        std::unique_ptr<runtime::IVoiceTaskSession> create_task_session(
            const runtime::TaskSpec & task,
            const runtime::SessionOptions & options) const override {
            return create_yue2_session(task, options, assets_);
        }

    private:
        std::shared_ptr<const Yue2Assets> assets_;
    };

    class Loader final : public runtime::IVoiceModelLoader {
    public:
        std::string family() const override {
            return kFamily;
        }

        bool can_load(const runtime::ModelLoadRequest & request) const override {
            if (request.family_hint.has_value() && *request.family_hint != kFamily) {
                return false;
            }
            try {
                (void) load_yue2_assets(request.model_path);
                return true;
            } catch (const std::exception &) {
                return false;
            }
        }

        runtime::ModelInspection inspect(const runtime::ModelLoadRequest & request) const override {
            auto assets = load_yue2_assets(request.model_path);
            runtime::ModelInspection inspection;
            inspection.model_root = assets->model_root;
            inspection.metadata = yue2_metadata();
            inspection.capabilities = yue2_capabilities();
            inspection.cli = yue2_cli_interface();
            inspection.discovered_configs = runtime::discover_named_assets(
                inspection.model_root,
                inspection.metadata.config_candidates);
            inspection.discovered_weights = runtime::discover_named_assets(
                inspection.model_root,
                inspection.metadata.weight_candidates);
            return inspection;
        }

        std::unique_ptr<runtime::ILoadedVoiceModel> load(const runtime::ModelLoadRequest & request) const override {
            return std::make_unique<LoadedModel>(load_yue2_assets(request.model_path));
        }

        runtime::CapabilitySet advertised_capabilities() const override {
            return yue2_capabilities();
        }
    };

    return std::make_shared<Loader>();
}

}  // namespace engine::models::yue2
