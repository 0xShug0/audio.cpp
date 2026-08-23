#include "engine/community_models/minimax_music3/session.h"

#include "engine/framework/debug/profiler.h"
#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/io/filesystem.h"
#include "engine/framework/runtime/model.h"
#include "engine/framework/runtime/options.h"

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>

namespace engine::models::minimax_music3 {
namespace {

using Clock = std::chrono::steady_clock;

constexpr const char * kFamily = "minimax_music3";
constexpr size_t kDefaultGraphArenaBytes = 32ull * 1024ull * 1024ull;
constexpr size_t kDefaultWeightContextBytes = 32ull * 1024ull * 1024ull;

std::shared_ptr<const MiniMaxMusic3Assets> require_assets(std::shared_ptr<const MiniMaxMusic3Assets> assets) {
    if (assets == nullptr) {
        throw std::runtime_error("MiniMax Music 3 session requires assets");
    }
    return assets;
}

std::filesystem::path resolve_component_gguf_path(
    const MiniMaxMusic3Assets & assets,
    std::string_view option_name,
    const std::string & value) {
    if (value.empty()) {
        throw std::runtime_error(std::string(option_name) + " must not be empty");
    }
    const std::filesystem::path relative(value);
    if (relative.is_absolute()) {
        throw std::runtime_error(std::string(option_name) + " must be relative to the MiniMax Music 3 model root");
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

std::shared_ptr<const MiniMaxMusic3Assets> select_component_assets(
    std::shared_ptr<const MiniMaxMusic3Assets> base,
    const std::unordered_map<std::string, std::string> & options) {
    auto selected = std::make_shared<MiniMaxMusic3Assets>(*base);
    bool changed = false;
    if (const auto value = runtime::find_option(options, {"minimax_music3.language_model_gguf"})) {
        selected->language_model_weights = assets::open_tensor_source(resolve_component_gguf_path(
            *base,
            "minimax_music3.language_model_gguf",
            *value));
        changed = true;
    }
    if (const auto value = runtime::find_option(options, {"minimax_music3.rvq_depth_decoder_gguf"})) {
        selected->depth_decoder_weights = assets::open_tensor_source(resolve_component_gguf_path(
            *base,
            "minimax_music3.rvq_depth_decoder_gguf",
            *value));
        changed = true;
    }
    if (const auto value = runtime::find_option(options, {"minimax_music3.flow_transformer_gguf"})) {
        selected->transformer_weights = assets::open_tensor_source(resolve_component_gguf_path(
            *base,
            "minimax_music3.flow_transformer_gguf",
            *value));
        changed = true;
    }
    if (!changed) {
        return base;
    }
    validate_minimax_music3_anchors(*selected);
    return selected;
}

const runtime::ModelMetadata & minimax_music3_metadata() noexcept {
    static const runtime::ModelMetadata metadata{
        kFamily,
        "MiniMax Music 3",
        "MiniMax Music 3 text-to-music generation with lyrics conditioning.",
        {
            "config.json",
            "config/language_model.json",
            "config/rvq_depth_decoder.json",
            "config/condition_encoder.json",
            "config/transformer.json",
            "config/vocoder.json",
            "tokenizer/tokenizer.json",
            "tokenizer/tokenizer_config.json",
        },
        {
            "language_model_q4_0.gguf",
            "language_model_q4_k.gguf",
            "language_model_bf16.gguf",
            "rvq_depth_decoder_bf16.gguf",
            "rvq_depth_decoder_q8_0.gguf",
            "rvq_depth_decoder_q4_k.gguf",
            "condition_encoder.gguf",
            "transformer_q4_0.gguf",
            "transformer_q4_k.gguf",
            "transformer_bf16.gguf",
            "vocoder.gguf",
        }};
    return metadata;
}

const runtime::CapabilitySet & minimax_music3_capabilities() noexcept {
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

runtime::ModelCliInterface minimax_music3_cli_interface() {
    runtime::ModelCliInterface out;
    out.request_options = {
        {"lyrics", "string", "Lyrics to sing.", true},
        {"duration_sec", "float", "Maximum generated audio duration in seconds.", false, "20.0", "0.1"},
        {"num_inference_steps", "int", "Flow matching Euler steps per chunk.", false, "30", "1"},
        {"guidance_scale", "float", "Flow transformer classifier-free guidance scale.", false, "1.7", "0.0"},
        {"ar_guidance_scale", "float", "Autoregressive semantic and depth CFG scale.", false, "1.5", "0.0"},
        {"top_k", "int", "Top-k sampling for semantic and residual code sampling.", false, "50", "1"},
        {"seed", "int", "Generation seed.", false, "0", "0"},
    };
    out.session_options = {
        {"minimax_music3.weight_type", "native|bf16|f16|q8_0|q4_0|q4_k", "Shared weight storage type.", false, "native"},
        {"minimax_music3.language_model_gguf", "string", "Language model component GGUF file relative to the model root.", false, "language_model_q4_0.gguf"},
        {"minimax_music3.rvq_depth_decoder_gguf", "string", "RVQ depth decoder component GGUF file relative to the model root.", false, "rvq_depth_decoder_q8_0.gguf"},
        {"minimax_music3.flow_transformer_gguf", "string", "Flow transformer component GGUF file relative to the model root.", false, "transformer_q4_0.gguf"},
        {"minimax_music3.graph_context_mb", "int", "Runtime graph arena size in MiB.", false, "32", "1"},
        {"minimax_music3.weight_context_mb", "int", "Weight context size in MiB.", false, "32", "1"},
        {"minimax_music3.mem_saver", "bool", "Load large generation stages only while they are needed to reduce peak VRAM.", false, "true"},
    };
    return out;
}

std::unique_ptr<runtime::IVoiceTaskSession> create_minimax_music3_session(
    const runtime::TaskSpec & task,
    const runtime::SessionOptions & options,
    std::shared_ptr<const MiniMaxMusic3Assets> assets) {
    return std::make_unique<MiniMaxMusic3Session>(
        task,
        options,
        std::move(assets));
}

}  // namespace

MiniMaxMusic3Session::MiniMaxMusic3Session(
    runtime::TaskSpec task,
    runtime::SessionOptions options,
    std::shared_ptr<const MiniMaxMusic3Assets> assets)
    : RuntimeSessionBase(options),
      task_(task),
      assets_(select_component_assets(require_assets(std::move(assets)), options.options)) {
    if (task_.mode != runtime::RunMode::Offline) {
        throw std::runtime_error("MiniMax Music 3 currently supports offline sessions");
    }
    if (task_.task != runtime::VoiceTaskKind::AudioGeneration) {
        throw std::runtime_error("MiniMax Music 3 supports the gen/music task");
    }
    const size_t graph_arena_bytes = runtime::parse_size_mb_option(
        this->options().options,
        {"minimax_music3.graph_context_mb"},
        kDefaultGraphArenaBytes);
    const size_t weight_context_bytes = runtime::parse_size_mb_option(
        this->options().options,
        {"minimax_music3.weight_context_mb"},
        kDefaultWeightContextBytes);
    const auto weight_type = runtime::parse_tensor_storage_option(
        this->options().options,
        "minimax_music3.weight_type",
        assets::TensorStorageType::Native,
        {assets::TensorStorageType::Native,
         assets::TensorStorageType::BF16,
         assets::TensorStorageType::F16,
         assets::TensorStorageType::Q8_0,
         assets::TensorStorageType::Q4_0,
         assets::TensorStorageType::Q4_K});
    bool memory_saver = true;
    if (const auto value = runtime::find_option(this->options().options, {"minimax_music3.mem_saver"})) {
        memory_saver = runtime::parse_bool_option(*value, "minimax_music3.mem_saver");
    }
    pipeline_ = std::make_unique<MiniMaxMusic3PipelineRuntime>(
        execution_context(),
        assets_,
        graph_arena_bytes,
        weight_context_bytes,
        weight_type,
        memory_saver);
}

MiniMaxMusic3Session::~MiniMaxMusic3Session() = default;

std::string MiniMaxMusic3Session::family() const {
    return kFamily;
}

runtime::VoiceTaskKind MiniMaxMusic3Session::task_kind() const {
    return task_.task;
}

runtime::RunMode MiniMaxMusic3Session::run_mode() const {
    return task_.mode;
}

void MiniMaxMusic3Session::prepare(const runtime::SessionPreparationRequest & request) {
    (void) request;
    mark_prepared();
}

MiniMaxMusic3Request MiniMaxMusic3Session::parse_request(const runtime::TaskRequest & request) const {
    if (!request.text_input.has_value() || request.text_input->text.empty()) {
        throw std::runtime_error("MiniMax Music 3 requires non-empty text input");
    }
    MiniMaxMusic3Request out;
    out.prompt = request.text_input->text;
    if (const auto lyrics = runtime::find_option(request.options, {"lyrics"})) {
        out.lyrics = *lyrics;
    }
    if (out.lyrics.empty()) {
        throw std::runtime_error("MiniMax Music 3 requires lyrics");
    }
    if (const auto value = runtime::parse_finite_float_option(request.options, {"duration_sec", "duration_seconds"})) {
        out.duration_sec = *value;
    }
    if (const auto value = runtime::parse_i64_option(request.options, {"num_inference_steps"})) {
        out.num_inference_steps = *value;
    }
    if (const auto value = runtime::parse_finite_float_option(request.options, {"guidance_scale"})) {
        out.guidance_scale = *value;
    }
    if (const auto value = runtime::parse_finite_float_option(request.options, {"ar_guidance_scale"})) {
        out.ar_guidance_scale = *value;
    }
    if (const auto value = runtime::parse_i64_option(request.options, {"top_k"})) {
        out.top_k = *value;
    }
    if (const auto value = runtime::parse_u64_option(request.options, {"seed"})) {
        out.seed = *value;
    }
    return out;
}

runtime::TaskResult MiniMaxMusic3Session::run(const runtime::TaskRequest & request) {
    require_prepared("MiniMax Music 3 run");
    const auto parsed = parse_request(request);
    const auto start = Clock::now();
    runtime::TaskResult result;
    result.audio_output = pipeline_->generate(parsed);
    engine::debug::timing_log_scalar(
        "session.wall_ms",
        engine::debug::elapsed_ms(start, Clock::now()));
    return result;
}

std::shared_ptr<runtime::IVoiceModelLoader> make_minimax_music3_loader() {
    class LoadedModel final : public runtime::ILoadedVoiceModel {
    public:
        explicit LoadedModel(std::shared_ptr<const MiniMaxMusic3Assets> assets)
            : assets_(require_assets(std::move(assets))) {}

        const runtime::ModelMetadata & metadata() const noexcept override {
            return minimax_music3_metadata();
        }

        const runtime::CapabilitySet & capabilities() const noexcept override {
            return minimax_music3_capabilities();
        }

        std::unique_ptr<runtime::IVoiceTaskSession> create_task_session(
            const runtime::TaskSpec & task,
            const runtime::SessionOptions & options) const override {
            return create_minimax_music3_session(task, options, assets_);
        }

    private:
        std::shared_ptr<const MiniMaxMusic3Assets> assets_;
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
                (void) load_minimax_music3_assets(request.model_path);
                return true;
            } catch (const std::exception &) {
                return false;
            }
        }

        runtime::ModelInspection inspect(const runtime::ModelLoadRequest & request) const override {
            auto assets = load_minimax_music3_assets(request.model_path);
            runtime::ModelInspection inspection;
            inspection.model_root = assets->model_root;
            inspection.metadata = minimax_music3_metadata();
            inspection.capabilities = minimax_music3_capabilities();
            inspection.cli = minimax_music3_cli_interface();
            inspection.discovered_configs = runtime::discover_named_assets(
                inspection.model_root,
                inspection.metadata.config_candidates);
            inspection.discovered_weights = runtime::discover_named_assets(
                inspection.model_root,
                inspection.metadata.weight_candidates);
            return inspection;
        }

        std::unique_ptr<runtime::ILoadedVoiceModel> load(const runtime::ModelLoadRequest & request) const override {
            return std::make_unique<LoadedModel>(load_minimax_music3_assets(request.model_path));
        }

        runtime::CapabilitySet advertised_capabilities() const override {
            return minimax_music3_capabilities();
        }
    };

    return std::make_shared<Loader>();
}

}  // namespace engine::models::minimax_music3
