#include "engine/community_models/liveavatar/session.h"

#include "engine/framework/runtime/options.h"

#include <filesystem>
#include <stdexcept>
#include <string_view>
#include <unordered_map>
#include <utility>

namespace engine::community_models::liveavatar {
namespace {

constexpr const char * kFamily = "liveavatar";

std::shared_ptr<const LiveAvatarAssets> require_assets(std::shared_ptr<const LiveAvatarAssets> assets) {
    if (assets == nullptr) {
        throw std::runtime_error("LiveAvatar session requires assets");
    }
    return assets;
}

engine::runtime::CapabilitySet make_capabilities() {
    engine::runtime::CapabilitySet out;
    out.supported_tasks.push_back({
        engine::runtime::VoiceTaskKind::AudioGeneration,
        {engine::runtime::RunMode::Offline, engine::runtime::RunMode::Streaming},
    });
    out.languages.push_back("auto");
    return out;
}

engine::runtime::ModelMetadata make_metadata(const LiveAvatarAssets &) {
    engine::runtime::ModelMetadata out;
    out.family = kFamily;
    out.variant = "LiveAvatar";
    out.description = "LiveAvatar speech-to-video generation loaded from local assets.";
    return out;
}

engine::runtime::VoiceArtifact make_video_artifact(
    LiveAvatarVideoResult video,
    std::string id,
    std::unordered_map<std::string, std::string> meta = {}) {
    meta.emplace("format", "rgb24");
    meta.emplace("width", std::to_string(video.width));
    meta.emplace("height", std::to_string(video.height));
    meta.emplace("frames", std::to_string(video.frames));
    meta.emplace("fps", std::to_string(video.fps));
    return engine::runtime::make_voice_artifact(
        engine::runtime::ArtifactKind::Custom,
        std::move(id),
        std::move(video.rgb24),
        std::move(meta));
}

engine::runtime::TaskResult make_video_result(LiveAvatarVideoResult video) {
    engine::runtime::TaskResult result;
    result.output_artifacts.push_back(make_video_artifact(std::move(video), "liveavatar_video_rgb24"));
    return result;
}

engine::runtime::ModelCliInterface make_cli() {
    engine::runtime::ModelCliInterface out;
    out.load_options = {
        {"support_gguf", "path", "Support GGUF path containing the text encoder, audio encoder, and embedded tokenizer sidecar."},
        {"vae_gguf", "path", "VAE GGUF path, resolved from the model root."},
        {"denoiser_gguf", "path", "Denoiser GGUF path, resolved from the model root."},
    };
    out.session_options = {
        {"liveavatar.denoiser_weight_streaming", "true|false", "Keep denoiser transformer blocks in host memory and stage layer groups on the accelerator."},
    };
    out.request_options = {
        {"reference_image_path", "path", "Reference image path used as the identity/appearance condition.", true},
        {"height", "n", "Generated frame height."},
        {"width", "n", "Generated frame width."},
        {"num_frames", "n", "Frames generated per LiveAvatar segment."},
        {"num_clips", "n", "Maximum LiveAvatar clips to generate from the input audio."},
        {"num_inference_steps", "n", "Flow denoising steps."},
        {"generation_mode", "offline|liveavatar", "Use whole-clip generation or LiveAvatar blockwise generation."},
        {"guidance_scale", "float", "Classifier-free guidance scale."},
        {"fused_cfg", "true|false", "Run classifier-free guidance in one denoiser graph."},
        {"sage_attention", "true|false", "Use H3-style CUDA SAGE attention in the denoiser graph."},
        {"memory_saver", "true|false", "Reuse immutable condition data and temporary denoiser graph storage across sequential blocks."},
        {"denoiser_layerwise", "true|false", "Run the denoiser in explicit layerwise mode."},
        {"denoiser_layerwise_batch", "n", "Denoiser layer group size when denoiser_layerwise=true."},
        {"vae_encoder_chunk_size", "n", "Cached VAE encoder chunk size."},
        {"vae_decoder_tile_size", "n", "Opt-in VAE decoder spatial tile size for large outputs."},
        {"target_cache_blocks", "n", "Target KV cache window in block units; 0 keeps the full clip cache."},
        {"vae_cache_f16", "true|false", "Store cached VAE encoder state in F16."},
        {"shift", "float", "Flow scheduler shift."},
        {"seed", "n", "Generation seed."},
        {"negative_prompt", "text", "Negative prompt for classifier-free guidance."},
    };
    return out;
}

bool has_desktop_layout(const std::filesystem::path & root) {
    const auto package_root = std::filesystem::is_regular_file(root) ? root.parent_path() : root;
    return std::filesystem::is_regular_file(package_root / "Wan2.2-S2V-Support-Q4_K_S-F16.gguf") &&
           std::filesystem::is_regular_file(package_root / "Wan2.2-S2V-VAE-F16.gguf") &&
           (std::filesystem::is_regular_file(package_root / "Wan2.2-S2V-14B-NVFP4-LORA.gguf") ||
            (std::filesystem::is_regular_file(root) && root.extension() == ".gguf"));
}

}  // namespace

LiveAvatarSession::LiveAvatarSession(
    engine::runtime::TaskSpec task,
    engine::runtime::SessionOptions options,
    std::shared_ptr<const LiveAvatarAssets> assets)
    : engine::runtime::RuntimeSessionBase(std::move(options)),
      task_(task),
      assets_(require_assets(std::move(assets))) {
    if (task_.task != engine::runtime::VoiceTaskKind::AudioGeneration ||
        (task_.mode != engine::runtime::RunMode::Offline &&
         task_.mode != engine::runtime::RunMode::Streaming)) {
        throw std::runtime_error("LiveAvatar supports only offline or streaming audio generation sessions");
    }
    bool denoiser_weight_streaming = false;
    if (const auto value = engine::runtime::find_option(
            RuntimeSessionBase::options().options,
            {"liveavatar.denoiser_weight_streaming"})) {
        denoiser_weight_streaming = engine::runtime::parse_bool_option(
            *value,
            "liveavatar.denoiser_weight_streaming");
    }
    pipeline_ = std::make_unique<LiveAvatarPipelineRuntime>(
        assets_,
        execution_context(),
        denoiser_weight_streaming);
    mark_prepared();
}

LiveAvatarSession::~LiveAvatarSession() = default;

std::string LiveAvatarSession::family() const {
    return kFamily;
}

engine::runtime::VoiceTaskKind LiveAvatarSession::task_kind() const {
    return task_.task;
}

engine::runtime::RunMode LiveAvatarSession::run_mode() const {
    return task_.mode;
}

void LiveAvatarSession::prepare(const engine::runtime::SessionPreparationRequest & request) {
    (void)request;
    mark_prepared();
}

engine::runtime::TaskResult LiveAvatarSession::run(const engine::runtime::TaskRequest & request) {
    require_prepared("LiveAvatar run");
    if (task_.mode != engine::runtime::RunMode::Offline) {
        throw std::runtime_error("LiveAvatar run requires an offline session");
    }
    auto generated = pipeline_->generate(make_request(request));
    return make_video_result(std::move(generated));
}

engine::runtime::StreamingPolicy LiveAvatarSession::streaming_policy() const {
    engine::runtime::StreamingPolicy policy;
    policy.input = engine::runtime::StreamingInputKind::None;
    policy.output = engine::runtime::StreamingOutputKind::FinalResult;
    return policy;
}

void LiveAvatarSession::start_stream(const engine::runtime::TaskRequest & request) {
    require_prepared("LiveAvatar streaming");
    if (task_.mode != engine::runtime::RunMode::Streaming) {
        throw std::runtime_error("LiveAvatar start_stream requires a streaming session");
    }
    reset();
    auto generation = make_request(request);
    if (!generation.blockwise_generation) {
        throw std::runtime_error("LiveAvatar streaming requires generation_mode=liveavatar");
    }
    int64_t chunk_index = 0;
    int64_t start_frame = 0;
    auto generated = pipeline_->generate(
        generation,
        [&](LiveAvatarVideoResult chunk) {
            const int64_t frames = chunk.frames;
            std::unordered_map<std::string, std::string> meta;
            meta.emplace("chunk_index", std::to_string(chunk_index));
            meta.emplace("start_frame", std::to_string(start_frame));
            auto artifact = make_video_artifact(
                std::move(chunk),
                "liveavatar_video_rgb24_chunk_" + std::to_string(chunk_index),
                std::move(meta));
            start_frame += frames;
            ++chunk_index;
            if (stream_event_sink_) {
                engine::runtime::StreamEvent event;
                event.output_artifacts.push_back(std::move(artifact));
                stream_event_sink_(event);
            }
        });
    stream_result_ = make_video_result(std::move(generated));
    stream_started_ = true;
}

void LiveAvatarSession::set_stream_event_sink(engine::runtime::StreamEventCallback sink) {
    stream_event_sink_ = std::move(sink);
}

engine::runtime::TaskResult LiveAvatarSession::finish_stream() {
    if (!stream_started_) {
        throw std::runtime_error("LiveAvatar streaming has not been started");
    }
    stream_started_ = false;
    auto result = std::move(stream_result_);
    stream_result_ = engine::runtime::TaskResult{};
    return result;
}

void LiveAvatarSession::reset() {
    stream_result_ = engine::runtime::TaskResult{};
    stream_started_ = false;
}

engine::runtime::StreamEvent LiveAvatarSession::process_audio_chunk(const engine::runtime::AudioChunk & chunk) {
    (void)chunk;
    throw std::runtime_error("LiveAvatar streaming does not consume audio chunks");
}

engine::runtime::TaskResult LiveAvatarSession::finalize() {
    return finish_stream();
}

LiveAvatarGenerateRequest LiveAvatarSession::make_request(const engine::runtime::TaskRequest & request) const {
    LiveAvatarGenerateRequest out;
    out.blockwise_generation = task_.mode == engine::runtime::RunMode::Streaming;
    if (request.text_input.has_value()) {
        out.prompt = request.text_input->text;
    }
    if (request.audio_input.has_value()) {
        out.audio = *request.audio_input;
    }
    if (const auto value = engine::runtime::find_option(request.options, {"reference_image_path", "ref_image"})) {
        out.reference_image_path = *value;
    }
    if (const auto value = engine::runtime::find_option(request.options, {"generation_mode", "liveavatar.generation_mode"})) {
        if (*value == "liveavatar") {
            out.blockwise_generation = true;
        } else if (*value != "offline") {
            throw std::runtime_error("LiveAvatar generation_mode must be offline or liveavatar");
        }
    }
    const int64_t default_frames_per_clip = out.blockwise_generation ? 48 : assets_->config.default_frames;
    const int64_t default_steps = out.blockwise_generation ? 4 : assets_->config.sample_steps;
    const float default_guidance_scale = out.blockwise_generation ? 0.0F : assets_->config.guidance_scale;
    if (out.blockwise_generation) {
        out.sage_attention = true;
    }
    out.height = engine::runtime::parse_int_option(request.options, {"height"}).value_or(assets_->config.height);
    out.width = engine::runtime::parse_int_option(request.options, {"width"}).value_or(assets_->config.width);
    out.frames_per_clip = engine::runtime::parse_int_option(request.options, {"num_frames"})
                              .value_or(default_frames_per_clip);
    out.max_clips = engine::runtime::parse_int_option(request.options, {"num_clips"})
                        .value_or(out.max_clips);
    out.steps = engine::runtime::parse_int_option(request.options, {"num_inference_steps", "steps"})
                    .value_or(default_steps);
    out.seed = engine::runtime::parse_u64_option(request.options, {"seed"}).value_or(out.seed);
    out.guidance_scale = engine::runtime::parse_float_option(request.options, {"guidance_scale"})
                             .value_or(default_guidance_scale);
    if (const auto value = engine::runtime::find_option(request.options, {"fused_cfg", "liveavatar.fused_cfg"})) {
        out.fused_cfg = engine::runtime::parse_bool_option(*value, "fused_cfg");
    }
    if (const auto value = engine::runtime::find_option(request.options, {"sage_attention", "liveavatar.sage_attention"})) {
        out.sage_attention = engine::runtime::parse_bool_option(*value, "sage_attention");
    }
    if (const auto value = engine::runtime::find_option(request.options, {"memory_saver", "liveavatar.memory_saver"})) {
        out.memory_saver = engine::runtime::parse_bool_option(*value, "memory_saver");
    }
    if (const auto value = engine::runtime::find_option(request.options, {"denoiser_layerwise", "liveavatar.denoiser_layerwise"})) {
        out.denoiser_layerwise = engine::runtime::parse_bool_option(*value, "denoiser_layerwise");
    }
    out.denoiser_layerwise_batch =
        engine::runtime::parse_int_option(request.options, {"denoiser_layerwise_batch", "liveavatar.denoiser_layerwise_batch"})
            .value_or(static_cast<int>(out.denoiser_layerwise_batch));
    if (out.denoiser_layerwise_batch <= 0) {
        throw std::runtime_error("LiveAvatar denoiser_layerwise_batch must be positive");
    }
    out.vae_encoder_chunk_size =
        engine::runtime::parse_int_option(request.options, {"vae_encoder_chunk_size", "liveavatar.vae_encoder_chunk_size"})
            .value_or(static_cast<int>(out.vae_encoder_chunk_size));
    if (out.vae_encoder_chunk_size <= 0) {
        throw std::runtime_error("LiveAvatar vae_encoder_chunk_size must be positive");
    }
    out.vae_decoder_tile_size =
        engine::runtime::parse_int_option(request.options, {"vae_decoder_tile_size", "liveavatar.vae_decoder_tile_size"})
            .value_or(static_cast<int>(out.vae_decoder_tile_size));
    if (out.vae_decoder_tile_size < 0) {
        throw std::runtime_error("LiveAvatar vae_decoder_tile_size must be non-negative");
    }
    out.target_cache_blocks =
        engine::runtime::parse_int_option(request.options, {"target_cache_blocks", "liveavatar.target_cache_blocks"})
            .value_or(static_cast<int>(out.target_cache_blocks));
    if (out.target_cache_blocks < 0) {
        throw std::runtime_error("LiveAvatar target_cache_blocks must be non-negative");
    }
    if (const auto value = engine::runtime::find_option(request.options, {"vae_cache_f16", "liveavatar.vae_cache_f16"})) {
        out.vae_cache_f16 = engine::runtime::parse_bool_option(*value, "vae_cache_f16");
    }
    out.shift = engine::runtime::parse_float_option(request.options, {"shift", "sample_shift"})
                    .value_or(assets_->config.sample_shift);
    if (const auto value = engine::runtime::find_option(request.options, {"negative_prompt", "negative"})) {
        out.negative_prompt = *value;
    }
    return out;
}

LiveAvatarLoadedModel::LiveAvatarLoadedModel(std::shared_ptr<const LiveAvatarAssets> assets)
    : metadata_(make_metadata(*require_assets(assets))),
      capabilities_(make_capabilities()),
      assets_(require_assets(std::move(assets))) {}

const engine::runtime::ModelMetadata & LiveAvatarLoadedModel::metadata() const noexcept {
    return metadata_;
}

const engine::runtime::CapabilitySet & LiveAvatarLoadedModel::capabilities() const noexcept {
    return capabilities_;
}

std::unique_ptr<engine::runtime::IVoiceTaskSession> LiveAvatarLoadedModel::create_task_session(
    const engine::runtime::TaskSpec & task,
    const engine::runtime::SessionOptions & options) const {
    return std::make_unique<LiveAvatarSession>(task, options, assets_);
}

std::shared_ptr<engine::runtime::IVoiceModelLoader> make_liveavatar_loader() {
    class Loader final : public engine::runtime::IVoiceModelLoader {
    public:
        std::string family() const override {
            return kFamily;
        }

        engine::runtime::CapabilitySet advertised_capabilities() const override {
            return make_capabilities();
        }

        bool can_load(const engine::runtime::ModelLoadRequest & request) const override {
            if (request.family_hint.has_value() && *request.family_hint != kFamily) {
                return false;
            }
            return has_desktop_layout(request.model_path);
        }

        engine::runtime::ModelInspection inspect(const engine::runtime::ModelLoadRequest & request) const override {
            auto assets = load_liveavatar_assets(request.model_path, request.options);
            engine::runtime::ModelInspection inspection;
            inspection.model_root = assets->model_root;
            inspection.metadata = make_metadata(*assets);
            inspection.capabilities = make_capabilities();
            inspection.cli = make_cli();
            inspection.discovered_weights = {
                {"support_weights", assets->support_weights->source_path()},
                {"text_encoder_weights", assets->text_encoder_weights->source_path()},
                {"denoiser_weights", assets->denoiser_weights->source_path()},
                {"audio_encoder_weights", assets->audio_encoder_weights->source_path()},
                {"vae_weights", assets->vae_weights->source_path()},
            };
            return inspection;
        }

        std::unique_ptr<engine::runtime::ILoadedVoiceModel> load(const engine::runtime::ModelLoadRequest & request) const override {
            return std::make_unique<LiveAvatarLoadedModel>(load_liveavatar_assets(request.model_path, request.options));
        }
    };
    return std::make_shared<Loader>();
}

}  // namespace engine::community_models::liveavatar
