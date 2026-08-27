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
        {"flow_uncond_interval", "int", "Evaluate the flow unconditional CFG branch only every N-th step and reuse the cached guidance delta in between. Default 2 is the accepted recipe (~-15..-21% wall, mel-L1 ~0.4 dB); 1 restores the exact reference trajectory.", false, "2", "1"},
        {"flow_uncond_warmup", "int", "Number of initial flow steps that always evaluate both CFG branches when delta reuse is enabled.", false, "2", "0"},
        {"ensemble_takes", "int", "Decode N independent takes of the same prompt in one batched AR pass (seeds seed..seed+N-1); flow and vocoder run per take. Outputs are returned as named audio (take_01..take_NN) for --out-dir.", false, "1", "1"},
        {"ensemble_prefix_frames", "int", "Intro-lock for ensembles: decode the first N frames once as a shared master trajectory (~25 frames per second), then fork the takes (take_01 continues the master exactly). 0 disables.", false, "0", "0"},
    };
    out.session_options = {
        {"minimax_music3.weight_type", "native|bf16|f16|q8_0|q4_0|q4_k", "Shared weight storage type.", false, "native"},
        {"minimax_music3.language_model_gguf", "string", "Language model component GGUF file relative to the model root.", false, "language_model_q4_0.gguf"},
        {"minimax_music3.rvq_depth_decoder_gguf", "string", "RVQ depth decoder component GGUF file relative to the model root. The q4_k default is the wave-2 recipe (depth is weight-bandwidth-bound: -24% depth, ~-6% wall, instrument-panel clean); q8_0 restores the previous default.", false, "rvq_depth_decoder_q4_k.gguf"},
        {"minimax_music3.flow_transformer_gguf", "string", "Flow transformer component GGUF file relative to the model root.", false, "transformer_q4_0.gguf"},
        {"minimax_music3.graph_context_mb", "int", "Runtime graph arena size in MiB.", false, "32", "1"},
        {"minimax_music3.weight_context_mb", "int", "Weight context size in MiB.", false, "32", "1"},
        {"minimax_music3.mem_saver", "bool", "Load large generation stages only while they are needed to reduce peak VRAM.", false, "true"},
        {"minimax_music3.pipeline_overlap", "bool", "Overlap AR decoding with per-chunk condition/flow/vocoder work on a second backend stream. Requires mem_saver=false; output is validated to stay identical to the sequential pipeline (automatic sequential fallback otherwise).", false, "false"},
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
    bool pipeline_overlap = false;
    if (const auto value = runtime::find_option(this->options().options, {"minimax_music3.pipeline_overlap"})) {
        pipeline_overlap = runtime::parse_bool_option(*value, "minimax_music3.pipeline_overlap");
    }
    pipeline_ = std::make_unique<MiniMaxMusic3PipelineRuntime>(
        execution_context(),
        assets_,
        graph_arena_bytes,
        weight_context_bytes,
        weight_type,
        memory_saver,
        pipeline_overlap);
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
    if (const auto value = runtime::parse_i64_option(request.options, {"flow_uncond_interval"})) {
        out.flow_uncond_interval = *value;
    }
    if (const auto value = runtime::parse_i64_option(request.options, {"flow_uncond_warmup"})) {
        out.flow_uncond_warmup = *value;
    }
    if (const auto value = runtime::parse_i64_option(request.options, {"ensemble_takes"})) {
        out.ensemble_takes = *value;
    }
    if (out.ensemble_takes < 1 || out.ensemble_takes > 16) {
        throw std::runtime_error("MiniMax Music 3 ensemble_takes must be in [1, 16]");
    }
    if (const auto value = runtime::parse_i64_option(request.options, {"ensemble_prefix_frames"})) {
        out.ensemble_prefix_frames = *value;
    }
    if (out.ensemble_prefix_frames < 0) {
        throw std::runtime_error("MiniMax Music 3 ensemble_prefix_frames must be non-negative");
    }
    return out;
}

runtime::TaskResult MiniMaxMusic3Session::run(const runtime::TaskRequest & request) {
    require_prepared("MiniMax Music 3 run");
    const auto parsed = parse_request(request);
    const auto start = Clock::now();
    runtime::TaskResult result;
    // Debug aid: route K=1 through the ensemble machinery to verify the
    // batched path reproduces the plain path (it must at batch 2).
    const char * force_env = std::getenv("MM3_ENSEMBLE_FORCE");
    const bool force_ensemble = force_env != nullptr && force_env[0] == '1';
    if (parsed.ensemble_takes > 1 || force_ensemble) {
        std::vector<uint64_t> take_seeds(static_cast<size_t>(parsed.ensemble_takes));
        for (size_t take = 0; take < take_seeds.size(); ++take) {
            take_seeds[take] = parsed.seed + static_cast<uint64_t>(take);
        }
        auto takes = pipeline_->generate_ensemble(parsed, take_seeds);
        result.audio_output = takes.front();
        for (size_t take = 0; take < takes.size(); ++take) {
            runtime::NamedAudioBuffer named;
            char id[16];
            std::snprintf(id, sizeof(id), "take_%02zu", take + 1);
            named.id = id;
            named.audio = std::move(takes[take]);
            named.meta.emplace("seed", std::to_string(take_seeds[take]));
            result.named_audio_outputs.push_back(std::move(named));
        }
    } else {
        result.audio_output = pipeline_->generate(parsed);
    }
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
