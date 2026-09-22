#include "engine/community_models/zipvoice/session.h"

#include "engine/community_models/zipvoice/synthesize.h"

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/model_spec/package.h"
#include "engine/framework/runtime/options.h"
#include "engine/framework/text/chunking.h"
#include "engine/framework/runtime/spec_backed_model.h"

#include <algorithm>
#include <filesystem>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

namespace engine::models::zipvoice {
namespace {

constexpr const char * kFamily = "zipvoice";

struct ZipVoiceAssets {
    assets::ResourceBundle resources;
    std::filesystem::path checkpoint;   // .gguf or safetensors dev dir
    std::filesystem::path model_dir;
    // session defaults (mutable only during session construction)
    int num_steps = 8;
    float guidance_scale = 3.0F;
    float t_shift = 0.5F;
};

const runtime::AudioBuffer * reference_audio(const runtime::TaskRequest & request) {
    if (request.voice.has_value() &&
        request.voice->speaker.has_value() &&
        request.voice->speaker->audio.has_value()) {
        return &*request.voice->speaker->audio;
    }
    return request.audio_input.has_value() ? &*request.audio_input : nullptr;
}

std::filesystem::path find_checkpoint(const std::filesystem::path & model_path) {
    namespace fs = std::filesystem;
    if (fs::is_regular_file(model_path)) {
        return model_path;
    }
    std::vector<fs::path> ggufs, safetensors;
    for (const auto & entry : fs::directory_iterator(model_path)) {
        const auto ext = entry.path().extension();
        // skip the vocoder package when it is staged in the same directory
        if (ext == ".gguf") ggufs.push_back(entry.path());
        else if (ext == ".safetensors") safetensors.push_back(entry.path());
    }
    std::sort(ggufs.begin(), ggufs.end());
    std::sort(safetensors.begin(), safetensors.end());
    if (!ggufs.empty()) return ggufs.back();
    if (!safetensors.empty()) return safetensors.back();
    throw std::runtime_error(
        "zipvoice: no .gguf/.safetensors checkpoint found in " + model_path.string());
}

bool tensor_file_has_namespace(const std::filesystem::path & path, const char * namespace_name) {
    try {
        const auto source = assets::open_tensor_source(path);
        const std::string prefix = std::string(namespace_name) + "/backbone.embed.weight";
        const std::string prefix_dots = std::string(namespace_name) + ".backbone.embed.weight";
        return source->has_tensor(prefix) || source->has_tensor(prefix_dots);
    } catch (...) {
        return false;
    }
}

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

std::shared_ptr<const ZipVoiceAssets> load_assets(const std::filesystem::path & model_path) {
    auto holder = std::make_shared<ZipVoiceAssets>();
    // Spec-resolved bundle: registered sidecars (tokens, model_config and the
    // optional zh_* frontend tables) resolve to materialized GGUF-embedded
    // copies or development-directory files; audio.cpp community models
    // (audio8_asr, vibeasr, glm_tts, ...) follow the same pattern.
    holder->resources = engine::model_spec::load_resource_bundle_for_family(model_path, kFamily);
    holder->checkpoint = find_checkpoint(model_path);
    holder->model_dir =
        std::filesystem::is_regular_file(model_path) ? model_path.parent_path() : model_path;
    return holder;
}

class ZipVoiceSession final : public runtime::IOfflineVoiceTaskSession {
public:
    ZipVoiceSession(
        const runtime::TaskSpec & task,
        const runtime::SessionOptions & options,
        std::shared_ptr<const ZipVoiceAssets> assets,
        std::shared_ptr<const engine::model_spec::ModelContract> contract)
        : task_kind_(task.task),
          run_mode_(task.mode),
          assets_(std::move(assets)),
          contract_(std::move(contract)) {
        if (assets_ == nullptr) {
            throw std::runtime_error("zipvoice session requires assets");
        }
        if (contract_ == nullptr) {
            throw std::runtime_error("zipvoice session requires a model contract");
        }
        runtime::validate_spec_backed_session_options(options, *contract_, kFamily, "ZipVoice");
        // Vocos resolution order (f5_tts pattern):
        //   1. zipvoice.vocos_path session option
        //   2. bundled "vocos" namespace in the GGUF checkpoint
        //   3. vocos.safetensors next to the checkpoint
        //   4. vocos-mel-24khz package next to the model directory
        namespace fs = std::filesystem;
        if (const auto v = runtime::find_option(options.options, {"zipvoice.vocos_path", "vocos_path"})) {
            vocos_path_ = *v;
        } else if (assets_->checkpoint.extension() == ".gguf" &&
                   tensor_file_has_namespace(assets_->checkpoint, "vocos")) {
            vocos_path_ = assets_->checkpoint.string();
        } else {
            const fs::path sibling = assets_->checkpoint.parent_path() / "vocos.safetensors";
            if (fs::exists(sibling)) {
                vocos_path_ = sibling.string();
            } else if (const auto pkg = find_tensor_file(
                           assets_->model_dir.parent_path() / "vocos-mel-24khz")) {
                vocos_path_ = pkg->string();
            }
        }
        if (vocos_path_.empty()) {
            throw std::runtime_error(
                "zipvoice: no vocos vocoder found; install the vocos-mel-24khz "
                "package or set session option zipvoice.vocos_path");
        }
        if (const auto v = runtime::find_option(options.options, {"zipvoice.num_inference_steps", "num_inference_steps"})) {
            num_steps_ = std::stoi(*v);
        }
        if (const auto v = runtime::find_option(options.options, {"zipvoice.guidance_scale", "guidance_scale"})) {
            guidance_scale_ = std::stof(*v);
        }
        if (const auto v = runtime::find_option(options.options, {"zipvoice.t_shift", "t_shift"})) {
            t_shift_ = std::stof(*v);
        }
        if (const auto v = runtime::find_option(options.options, {"zipvoice.espeak_data_path", "espeak_data_path"})) {
            espeak_data_path_ = *v;
        }
        if (const auto v = runtime::find_option(options.options, {"zipvoice.espeak_library_path", "espeak_library_path"})) {
            espeak_library_path_ = *v;
        }
        device_.backend_type = options.backend.type;
        device_.device_index = options.backend.device;
        device_.threads = options.backend.threads;
    }

    std::string family() const noexcept override { return kFamily; }
    runtime::VoiceTaskKind task_kind() const noexcept override { return task_kind_; }
    runtime::RunMode run_mode() const noexcept override { return run_mode_; }

    void prepare(const runtime::SessionPreparationRequest & request) override {
        (void) request;
    }

    runtime::TaskResult run(const runtime::TaskRequest & request) override {
        const auto budget = text::parse_text_chunk_size_override(request.options).value_or(128);
        if (budget <= 0) throw std::invalid_argument("zipvoice: text_chunk_size must be positive");
        const auto mode = text::parse_text_chunk_mode_override(request.options).value_or(text::TextChunkMode::Default);
        const auto chunks = runtime::chunk_text_request(request, budget, mode);
        runtime::AudioBuffer merged;
        for (const auto & chunk : chunks) {
            auto result = run_chunk(chunk);
            runtime::append_audio_buffer(merged, *result.audio_output);
        }
        if (merged.samples.empty()) throw std::invalid_argument("zipvoice: empty text chunks");
        runtime::TaskResult result;
        result.audio_output = std::move(merged);
        return result;
    }

    runtime::TaskResult run_chunk(const runtime::TaskRequest & request) {
        if (!request.text_input.has_value() || request.text_input->text.empty()) {
            throw std::runtime_error("zipvoice requires input text");
        }
        const runtime::AudioBuffer * ref = reference_audio(request);
        if (ref == nullptr || ref->samples.empty()) {
            throw std::runtime_error(
                "zipvoice requires reference voice audio (voice preset or voice_ref)");
        }
        const auto ref_text_it = request.options.find("reference_text");
        if (ref_text_it == request.options.end() || ref_text_it->second.empty()) {
            throw std::runtime_error(
                "zipvoice requires reference_text (transcript of the reference audio)");
        }

        ZipVoiceSynthesisRequest req;
        req.text = request.text_input->text;
        req.ref_text = ref_text_it->second;
        req.ref_audio = ref->samples;
        req.ref_sample_rate = ref->sample_rate;
        req.ref_channels = ref->channels;
        req.tokenizer = "emilia";  // fixed frontend: zh/en/mixed via the Emilia pipeline
        req.espeak_library_path = espeak_library_path_;
        req.espeak_data_path = espeak_data_path_;
        if (const auto v = runtime::find_option(request.options, {"lang"})) req.lang = *v;
        req.num_steps = num_steps_;
        req.guidance_scale = guidance_scale_;
        req.t_shift = t_shift_;
        if (const auto v = runtime::find_option(request.options, {"num_inference_steps"})) {
            req.num_steps = std::stoi(*v);
        }
        if (const auto v = runtime::find_option(request.options, {"guidance_scale", "cfg_strength"})) {
            req.guidance_scale = std::stof(*v);
        }
        if (const auto v = runtime::find_option(request.options, {"t_shift"})) {
            req.t_shift = std::stof(*v);
        }
        if (const auto v = runtime::find_option(request.options, {"speed"})) {
            req.speed = std::stof(*v);
        }
        if (const auto v = runtime::find_option(request.options, {"feat_scale"})) {
            req.feat_scale = std::stof(*v);
        }
        if (const auto v = runtime::find_option(request.options, {"target_rms"})) {
            req.target_rms = std::stof(*v);
        }
        if (const auto v = runtime::find_option(request.options, {"seed"})) {
            req.seed = static_cast<uint32_t>(std::stoul(*v));
            req.fixed_seed = true;
        }

        auto out = zipvoice_synthesize(
            assets_->checkpoint.string(), vocos_path_, req, device_, &assets_->resources);

        runtime::TaskResult result;
        runtime::AudioBuffer audio;
        audio.sample_rate = out.sample_rate;
        audio.channels = 1;
        audio.samples = std::move(out.audio);
        result.audio_output = std::move(audio);
        return result;
    }

private:
    runtime::VoiceTaskKind task_kind_;
    runtime::RunMode run_mode_;
    std::shared_ptr<const ZipVoiceAssets> assets_;
    std::shared_ptr<const engine::model_spec::ModelContract> contract_;
    std::string vocos_path_;
    int num_steps_ = 8;
    float guidance_scale_ = 3.0F;
    float t_shift_ = 0.5F;
    std::string espeak_library_path_;
    std::string espeak_data_path_;
    ZipVoiceComputeDevice device_;
};

}  // namespace

std::shared_ptr<runtime::IVoiceModelLoader> make_zipvoice_loader() {
    runtime::SpecBackedVoiceModelConfig<ZipVoiceAssets> config;
    config.family = std::string(kFamily);
    config.load_assets = load_assets;
    config.create_session = [](
                                const runtime::TaskSpec & task,
                                const runtime::SessionOptions & options,
                                std::shared_ptr<const ZipVoiceAssets> assets,
                                std::shared_ptr<const engine::model_spec::ModelContract> contract) {
        return std::make_unique<ZipVoiceSession>(
            task, options, std::move(assets), std::move(contract));
    };
    return runtime::make_spec_backed_voice_loader(std::move(config));
}

}  // namespace engine::models::zipvoice
