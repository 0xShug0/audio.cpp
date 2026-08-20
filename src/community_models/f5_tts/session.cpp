#include "engine/community_models/f5_tts/session.h"

#include "engine/community_models/f5_tts/synthesize.h"

#include "engine/framework/runtime/options.h"
#include "engine/framework/runtime/spec_backed_model.h"

#include <algorithm>
#include <filesystem>
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
// *.safetensors is expected (Habibi Unified/Specialized layout).
std::filesystem::path find_checkpoint(const std::filesystem::path & model_path) {
    namespace fs = std::filesystem;
    if (fs::is_regular_file(model_path)) {
        return model_path;  // direct path to the .safetensors
    }
    std::vector<fs::path> found;
    for (const auto & entry : fs::directory_iterator(model_path)) {
        if (entry.path().extension() == ".safetensors") {
            found.push_back(entry.path());
        }
    }
    if (found.empty()) {
        throw std::runtime_error(
            "F5-TTS: no .safetensors checkpoint found in " + model_path.string());
    }
    if (found.size() > 1) {
        // prefer the highest-numbered model_*.safetensors (latest step)
        std::sort(found.begin(), found.end());
    }
    return found.back();
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
    // Vocos vocoder checkpoint: session option, else auto-discover (next to
    // the DiT checkpoint, or the vocos-mel-24khz package installed alongside
    // the model directory, e.g. <models>/vocos-mel-24khz/model.safetensors).
    const auto vocos_opt = runtime::find_option(
        options.options, {"f5_tts.vocos_path", "vocos_path"});
    namespace fs = std::filesystem;
    if (vocos_opt.has_value()) {
        vocos_path_ = *vocos_opt;
    } else {
        const fs::path ckpt_dir = assets_->checkpoint.parent_path();
        const fs::path models_root = ckpt_dir.parent_path().parent_path();
        const fs::path candidates[] = {
            ckpt_dir / "vocos.safetensors",
            models_root / "vocos-mel-24khz" / "vocos.safetensors",
            models_root / "vocos-mel-24khz" / "model.safetensors",
        };
        for (const auto & c : candidates) {
            if (fs::exists(c)) {
                vocos_path_ = c.string();
                break;
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
    if (const auto v = runtime::find_option(request.options, {"guidance_scale"})) {
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
