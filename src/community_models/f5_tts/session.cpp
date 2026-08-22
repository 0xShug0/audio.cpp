#include "engine/community_models/f5_tts/session.h"

#include "engine/community_models/f5_tts/synthesize.h"

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/runtime/options.h"
#include "engine/framework/runtime/spec_backed_model.h"

#include <algorithm>
#include <filesystem>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

namespace engine::models::f5_tts {
namespace {

constexpr const char * kFamily = "f5_tts";

const runtime::AudioBuffer * reference_audio(const runtime::TaskRequest & request) {
    if (request.voice.has_value() &&
        request.voice->speaker.has_value() &&
        request.voice->speaker->audio.has_value()) {
        return &*request.voice->speaker->audio;
    }
    return request.audio_input.has_value()
        ? &*request.audio_input
        : nullptr;
}

// Locate the DiT checkpoint inside the model directory: exactly one
// *.safetensors / *.gguf is expected (Habibi Unified/Specialized layout).
// GGUF is preferred when both are present (it is the packaged format).
std::filesystem::path find_checkpoint(const std::filesystem::path & model_path) {
    namespace fs = std::filesystem;
    if (fs::is_regular_file(model_path)) {
        return model_path;  // direct path to the checkpoint file
    }
    std::vector<fs::path> ggufs, safetensors;
    for (const auto & entry : fs::directory_iterator(model_path)) {
        if (entry.path().extension() == ".gguf") ggufs.push_back(entry.path());
        else if (entry.path().extension() == ".safetensors") safetensors.push_back(entry.path());
    }
    std::sort(ggufs.begin(), ggufs.end());
    std::sort(safetensors.begin(), safetensors.end());
    if (!ggufs.empty()) {
        return ggufs.back();  // prefer gguf; highest sort key if several
    }
    if (!safetensors.empty()) {
        return safetensors.back();  // highest-numbered model_*.safetensors
    }
    throw std::runtime_error(
        "F5-TTS: no .gguf/.safetensors checkpoint found in " + model_path.string());
}

// First tensor-source file (safetensors preferred, then gguf) in a directory.
std::optional<std::filesystem::path> find_tensor_file(const std::filesystem::path & dir) {
    namespace fs = std::filesystem;
    if (!fs::is_directory(dir)) return std::nullopt;
    for (const char * ext : {".safetensors", ".gguf"}) {
        for (const auto & entry : fs::directory_iterator(dir)) {
            if (entry.path().extension() == ext) return entry.path();
        }
    }
    return std::nullopt;
}

}  // namespace

std::shared_ptr<const F5TTSAssets> load_f5_tts_assets(
    const std::filesystem::path & model_path) {
    auto assets = std::make_shared<F5TTSAssets>();
    assets->resources = assets::ResourceBundle(model_path);
    assets->checkpoint = find_checkpoint(model_path);
    return assets;
}

F5TTSSession::F5TTSSession(
    const runtime::TaskSpec & task,
    const runtime::SessionOptions & options,
    std::shared_ptr<const F5TTSAssets> assets,
    std::shared_ptr<const engine::model_spec::ModelContract> contract)
    : task_kind_(task.task),
      run_mode_(task.mode),
      assets_(std::move(assets)),
      contract_(std::move(contract)) {
    if (assets_ == nullptr) {
        throw std::runtime_error("F5-TTS session requires assets");
    }
    if (contract_ == nullptr) {
        throw std::runtime_error("F5-TTS session requires a model contract");
    }
    // Vocos vocoder resolution order:
    //   1. f5_tts.vocos_path session option
    //   2. bundled "vocos" namespace inside a GGUF checkpoint
    //   3. vocos.safetensors next to the checkpoint
    //   4. the vocos-mel-24khz package installed next to the model directory
    const auto vocos_opt = runtime::find_option(
        options.options, {"f5_tts.vocos_path", "vocos_path"});
    namespace fs = std::filesystem;
    if (vocos_opt.has_value()) {
        vocos_path_ = *vocos_opt;
    } else {
        const fs::path ckpt_dir = assets_->checkpoint.parent_path();
        const fs::path models_root = ckpt_dir.parent_path().parent_path();
        if (assets_->checkpoint.extension() == ".gguf") {
            const auto probe = assets::open_tensor_source(assets_->checkpoint);
            // packed GGUF namespaces are slash-separated (vocos/backbone...)
            if (probe->has_tensor("vocos/backbone.embed.weight") ||
                probe->has_tensor("vocos.backbone.embed.weight")) {
                vocos_path_ = assets_->checkpoint.string();
            }
        }
        if (vocos_path_.empty()) {
            const fs::path sibling = ckpt_dir / "vocos.safetensors";
            if (fs::exists(sibling)) {
                vocos_path_ = sibling.string();
            } else if (const auto pkg = find_tensor_file(models_root / "vocos-mel-24khz")) {
                vocos_path_ = pkg->string();
            }
        }
        if (vocos_path_.empty()) {
            throw std::runtime_error(
                "F5-TTS: no vocos vocoder found; install the vocos_mel_24khz "
                "package or set session option f5_tts.vocos_path");
        }
    }
    if (const auto d = runtime::find_option(options.options, {"f5_tts.dialect", "dialect"})) {
        dialect_ = *d;
    }
    if (const auto fb = runtime::find_option(options.options, {"f5_tts.frame_budget", "frame_budget"})) {
        frame_budget_ = std::stoi(*fb);
        if (frame_budget_ < 256 || frame_budget_ > 8192) {
            throw std::runtime_error(
                "f5_tts.frame_budget must be within [256, 8192] mel frames");
        }
    }
    use_cuda_ = options.backend.type == core::BackendType::Cuda;
    cuda_device_ = options.backend.device;
    threads_ = options.backend.threads;
}

std::string F5TTSSession::family() const noexcept {
    return kFamily;
}

runtime::VoiceTaskKind F5TTSSession::task_kind() const noexcept {
    return task_kind_;
}

runtime::RunMode F5TTSSession::run_mode() const noexcept {
    return run_mode_;
}

void F5TTSSession::prepare(const runtime::SessionPreparationRequest & request) {
    (void) request;
    // Graphs are built lazily on first synthesis (bucketed by duration).
}

runtime::TaskResult F5TTSSession::run(const runtime::TaskRequest & request) {
    if (!request.text_input.has_value() || request.text_input->text.empty()) {
        throw std::runtime_error("F5-TTS requires input text");
    }
    const runtime::AudioBuffer * ref = reference_audio(request);
    if (ref == nullptr || ref->samples.empty()) {
        throw std::runtime_error(
            "F5-TTS requires reference voice audio (voice preset or voice_ref)");
    }
    const auto ref_text_it = request.options.find("reference_text");
    if (ref_text_it == request.options.end() || ref_text_it->second.empty()) {
        throw std::runtime_error(
            "F5-TTS requires reference_text (transcript of the reference audio)");
    }

    F5SynthesisRequest req;
    req.text = request.text_input->text;
    req.ref_audio = ref->samples;
    req.ref_sample_rate = ref->sample_rate;
    req.ref_text = ref_text_it->second;
    if (const auto v = runtime::find_option(request.options, {"dialect"})) {
        req.dialect = *v;
    } else {
        req.dialect = dialect_;
    }
    if (const auto v = runtime::find_option(request.options, {"speed"})) {
        req.speed = std::stof(*v);
    }
    if (const auto v = runtime::find_option(request.options, {"num_inference_steps"})) {
        req.steps = std::stoi(*v);
    }
    if (const auto v = runtime::find_option(request.options, {"cfg_strength", "guidance_scale"})) {
        req.cfg_strength = std::stof(*v);
    }
    if (const auto v = runtime::find_option(request.options, {"sway_sampling_coef"})) {
        req.sway_sampling_coef = std::stof(*v);
    }
    if (const auto v = runtime::find_option(request.options, {"seed"})) {
        req.seed = static_cast<uint32_t>(std::stoul(*v));
        req.fixed_seed = true;
    }
    req.use_cuda = use_cuda_;
    req.frame_budget = frame_budget_;
    req.cuda_device = cuda_device_;
    req.threads = threads_;

    const auto out = f5_synthesize(
        assets_->checkpoint.string(), vocos_path_, req);

    runtime::TaskResult result;
    runtime::AudioBuffer audio;
    audio.sample_rate = static_cast<int>(out.sample_rate);
    audio.channels = 1;
    audio.samples = std::move(out.audio);
    result.audio_output = std::move(audio);
    return result;
}

std::shared_ptr<runtime::IVoiceModelLoader> make_f5_tts_loader() {
    runtime::SpecBackedVoiceModelConfig<F5TTSAssets> config;
    config.family = std::string(kFamily);
    config.aliases = {"habibi", "habibi_tts"};
    config.load_assets = load_f5_tts_assets;
    config.create_session = [](
                                const runtime::TaskSpec & task,
                                const runtime::SessionOptions & options,
                                std::shared_ptr<const F5TTSAssets> assets,
                                std::shared_ptr<const engine::model_spec::ModelContract> contract) {
        return std::make_unique<F5TTSSession>(
            task,
            options,
            std::move(assets),
            std::move(contract));
    };
    return runtime::make_spec_backed_voice_loader(std::move(config));
}

}  // namespace engine::models::f5_tts
