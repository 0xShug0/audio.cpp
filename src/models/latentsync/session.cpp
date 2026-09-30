#include "engine/models/latentsync/session.h"
#include "engine/models/latentsync/pipeline.h"
#include "engine/models/latentsync/video.h"
#include "engine/framework/audio/conversion.h"
#include "engine/framework/assets/resource_bundle.h"
#include "engine/framework/model_spec/package.h"
#include "engine/framework/runtime/session_base.h"
#include "engine/framework/runtime/spec_backed_model.h"
#include "engine/framework/runtime/options.h"
#include "engine/framework/io/json.h"
#include "engine/framework/debug/profiler.h"

#include <chrono>
#include <stdexcept>

namespace engine::models::latentsync {
namespace {
struct LatentSyncAssets {
    assets::ResourceBundle resources;
    LatentSyncPipelineSources sources;
    std::vector<float> mask;
};

std::shared_ptr<const LatentSyncAssets> load_assets(const std::filesystem::path & path) {
    auto out = std::make_shared<LatentSyncAssets>();
    out->resources = model_spec::load_resource_bundle_for_family(path, "latentsync");
    const auto weights = out->resources.open_tensor_source("weights");
    out->sources.unet = out->sources.vae = out->sources.whisper = out->sources.detector = out->sources.landmarks = weights;
    out->sources.detector_config = out->resources.require_file("detector_config");
    out->sources.landmark_config = out->resources.require_file("landmark_config");
    out->mask = weights->require_f32("face.mask");
    return out;
}

class LatentSyncSession final : public runtime::RuntimeSessionBase, public runtime::IOfflineVoiceTaskSession {
public:
    LatentSyncSession(runtime::TaskSpec task, runtime::SessionOptions options,
        std::shared_ptr<const LatentSyncAssets> assets, std::shared_ptr<const model_spec::ModelContract> contract)
        : RuntimeSessionBase(std::move(options)), task_(task), assets_(std::move(assets)), contract_(std::move(contract)) {
        if (task_.task != runtime::VoiceTaskKind::AudioGeneration || task_.mode != runtime::RunMode::Offline) {
            throw std::runtime_error("LatentSync supports offline audio-conditioned video generation");
        }
        runtime::validate_spec_backed_session_options(RuntimeSessionBase::options(), *contract_, "latentsync", "LatentSync");
        auto type = assets::TensorStorageType::Native;
        if (const auto value = runtime::find_option(RuntimeSessionBase::options().options, {"latentsync.weight_type"})) type = assets::parse_tensor_storage_type(*value);
        bool vae_cudnn_f16 = false;
        if (const auto value = runtime::find_option(RuntimeSessionBase::options().options, {"latentsync.vae_cudnn_f16"})) {
            vae_cudnn_f16 = runtime::parse_bool_option(*value, "latentsync.vae_cudnn_f16");
        }
        pipeline_ = std::make_unique<LatentSyncPipeline>(execution_context(), assets_->sources, type, vae_cudnn_f16);
    }
    std::string family() const override { return "latentsync"; }
    runtime::VoiceTaskKind task_kind() const override { return task_.task; }
    runtime::RunMode run_mode() const override { return task_.mode; }
    void prepare(const runtime::SessionPreparationRequest &) override { mark_prepared(); }
    runtime::TaskResult run(const runtime::TaskRequest & request) override {
        const auto start = std::chrono::steady_clock::now();
        require_prepared("LatentSync run");
        runtime::validate_spec_backed_request_options(request.options, request.option_arrays, *contract_, "LatentSync");
        const auto video_path = runtime::find_option(request.options, {"reference_video_path"});
        if (!request.audio_input || !video_path || video_path->empty()) throw std::runtime_error("LatentSync requires audio and reference_video_path");
        LatentSyncGenerationOptions options;
        options.steps = runtime::parse_positive_i64_option(request.options, {"num_inference_steps"}, 20);
        options.cache_interval = runtime::parse_positive_i64_option(request.options, {"cache_interval"}, 1);
        options.guidance_scale = runtime::parse_finite_float_option(request.options, {"guidance_scale"}).value_or(1.5F);
        options.frames_per_chunk = runtime::parse_positive_i64_option(request.options, {"frames_per_chunk"}, 16);
        options.fps = runtime::parse_finite_float_option(request.options, {"fps"}).value_or(25);
        options.seed = runtime::parse_u64_option(request.options, {"seed"}).value_or(1247);
        auto video = load_latentsync_video(*video_path, options.fps);
        const auto & input = *request.audio_input;
        auto mono = audio::convert_interleaved_audio_to_mono_torchaudio_sinc_hann_resampled(input.samples, input.sample_rate, input.channels, 16000);
        std::vector<std::byte> output;
        size_t frame_count = 0;
        pipeline_->run(video.frames, video.height, video.width, mono, assets_->mask, options,
            [&](size_t, const std::vector<uint8_t> & frame) {
                const auto * bytes = reinterpret_cast<const std::byte *>(frame.data());
                output.insert(output.end(), bytes, bytes + frame.size());
                ++frame_count;
            });
        runtime::TaskResult result;
        using Json = io::json::Value;
        const auto metadata = Json::make_object({
            {"format", Json::make_string("rgb24")},
            {"width", Json::make_number(video.width)},
            {"height", Json::make_number(video.height)},
            {"frames", Json::make_number(frame_count)},
            {"fps", Json::make_number(options.fps)},
            {"file", Json::make_string("latentsync_video_rgb24.rgb")}});
        result.artifact_output = runtime::make_voice_artifact(runtime::ArtifactKind::Custom,
            "latentsync_video", runtime::bytes_from_string(io::json::stringify(metadata)),
            {{"extension", "json"}, {"mime", "application/json"}});
        result.output_artifacts.push_back(runtime::make_voice_artifact(runtime::ArtifactKind::Custom,
            "latentsync_video_rgb24", std::move(output), {{"extension", "rgb"}, {"mime", "application/octet-stream"},
                {"format", "rgb24"}, {"width", std::to_string(video.width)},
                {"height", std::to_string(video.height)}, {"frames", std::to_string(frame_count)}, {"fps", std::to_string(options.fps)}}));
        debug::timing_log_scalar("session.wall_ms",
            debug::elapsed_ms(start, std::chrono::steady_clock::now()));
        return result;
    }
private:
    runtime::TaskSpec task_;
    std::shared_ptr<const LatentSyncAssets> assets_;
    std::shared_ptr<const model_spec::ModelContract> contract_;
    std::unique_ptr<LatentSyncPipeline> pipeline_;
};
}

std::shared_ptr<runtime::IVoiceModelLoader> make_latentsync_loader() {
    runtime::SpecBackedVoiceModelConfig<LatentSyncAssets> config;
    config.family = "latentsync";
    config.load_assets = load_assets;
    config.create_session = [](const runtime::TaskSpec & task, const runtime::SessionOptions & options,
        std::shared_ptr<const LatentSyncAssets> assets, std::shared_ptr<const model_spec::ModelContract> contract) {
        return std::make_unique<LatentSyncSession>(task, options, std::move(assets), std::move(contract));
    };
    return runtime::make_spec_backed_voice_loader(std::move(config));
}

}  // namespace engine::models::latentsync
