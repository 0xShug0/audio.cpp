#include "engine/models/index_tts2/loader.h"

#include "engine/framework/assets/model_package.h"
#include "engine/models/index_tts2/session.h"

#include <stdexcept>
#include <utility>

namespace engine::models::index_tts2 {
namespace {

class IndexTTS2Loader final : public runtime::IVoiceModelLoader {
public:
    std::string family() const override {
        return "index_tts2";
    }

    bool can_load(const runtime::ModelLoadRequest & request) const override {
        try {
            (void) engine::assets::load_resource_bundle_from_package_spec(
                request.model_path,
                engine::assets::default_model_package_spec_path(family()));
            return !request.family_hint.has_value() || *request.family_hint == family();
        } catch (...) {
            return false;
        }
    }

    runtime::ModelInspection inspect(const runtime::ModelLoadRequest & request) const override {
        const auto assets = load_index_tts2_assets(request.model_path);
        runtime::ModelInspection inspection;
        inspection.model_root = assets->resources.model_root();
        inspection.metadata.family = family();
        inspection.metadata.variant = assets->config.version;
        inspection.metadata.description = "IndexTTS2 loaded from local extracted assets.";
        inspection.capabilities.supported_tasks = {
            {runtime::VoiceTaskKind::Tts, {runtime::RunMode::Offline}},
            {runtime::VoiceTaskKind::VoiceCloning, {runtime::RunMode::Offline}},
        };
        inspection.capabilities.supports_speaker_reference = true;
        inspection.capabilities.supports_style_condition = true;
        inspection.capabilities.languages = {"English", "Chinese"};
        inspection.cli.request_options = {
            {"emotion_alpha", "float", "Blend strength for explicit emotion conditioning."},
            {"emotion_vector", "float[,float...]", "Eight-value explicit emotion vector."},
            {"use_emotion_text", "bool", "Infer emotion from text instead of reference audio."},
            {"emotion_text", "text", "Text used when emotion-text conditioning is enabled."},
            {"use_random_emotion", "bool", "Use random emotion weights in the emotion mixer."},
            {"interval_silence_ms", "n", "Silence inserted between generated text chunks."},
            {"text_chunk_mode", "default|tag_aware|japanese|endline", "Framework text chunking mode used when text_chunk_size is set."},
            {"length_penalty", "float", "GPT beam-search length penalty."},
            {"num_beams", "n", "GPT beam count."},
        };
        inspection.cli.session_options = {
            {"index_tts2.weight_type", "native|f32|f16|bf16|q8_0", "Matmul weight storage type."},
            {"index_tts2.conv_weight_type", "native|f32|f16", "Convolution weight storage type."},
            {"index_tts2.gpt_graph_arena_mb", "n", "GPT graph arena size."},
            {"index_tts2.s2mel_graph_arena_mb", "n", "S2Mel graph arena size."},
            {"index_tts2.reference_graph_arena_mb", "n", "Reference encoder and codec graph arena size."},
            {"index_tts2.emotion_text_prefill_graph_arena_mb", "n", "Emotion-text prefill graph arena size."},
            {"index_tts2.emotion_text_decode_graph_arena_mb", "n", "Emotion-text cached-step graph arena size."},
            {"index_tts2.emotion_text_max_new_tokens", "n", "Maximum generated tokens for emotion-text classification; default 256."},
            {"index_tts2.weight_context_mb", "n", "Shared weight context size."},
            {"index_tts2.mem_saver", "true|false", "Release staged reference and conditioning graphs after request phases; default false."},
            {"index_tts2.speaker_cache_slots", "n", "Prepared speaker-reference cache slots; default 1."},
            {"index_tts2.emotion_cache_slots", "n", "Prepared emotion-reference cache slots; default 1."},
            {"index_tts2.emotion_text_cache_slots", "n", "Emotion-text weight cache slots; default 1."},
        };
        const auto spec_path = engine::assets::default_model_package_spec_path(family());
        inspection.discovered_configs = runtime::discover_named_assets_from_package_spec(
            request.model_path,
            spec_path,
            engine::assets::ModelPackageResourceKind::Files);
        inspection.discovered_weights = runtime::discover_named_assets_from_package_spec(
            request.model_path,
            spec_path,
            engine::assets::ModelPackageResourceKind::Tensors);
        return inspection;
    }

    std::unique_ptr<runtime::ILoadedVoiceModel> load(const runtime::ModelLoadRequest & request) const override {
        return load_index_tts2_model(request.model_path);
    }
};

}  // namespace

IndexTTS2LoadedModel::IndexTTS2LoadedModel(
    runtime::ModelMetadata metadata,
    runtime::CapabilitySet capabilities,
    std::shared_ptr<const IndexTTS2Assets> assets)
    : metadata_(std::move(metadata)),
      capabilities_(std::move(capabilities)),
      assets_(std::move(assets)) {}

const runtime::ModelMetadata & IndexTTS2LoadedModel::metadata() const noexcept {
    return metadata_;
}

const runtime::CapabilitySet & IndexTTS2LoadedModel::capabilities() const noexcept {
    return capabilities_;
}

std::unique_ptr<runtime::IVoiceTaskSession> IndexTTS2LoadedModel::create_task_session(
    const runtime::TaskSpec & task,
    const runtime::SessionOptions & options) const {
    if (task.mode != runtime::RunMode::Offline ||
        (task.task != runtime::VoiceTaskKind::Tts && task.task != runtime::VoiceTaskKind::VoiceCloning)) {
        throw std::runtime_error("IndexTTS2 only supports offline TTS and voice-cloning sessions");
    }
    return std::make_unique<IndexTTS2Session>(task, options, assets_);
}

std::unique_ptr<IndexTTS2LoadedModel> load_index_tts2_model(const std::filesystem::path & model_path) {
    auto assets = load_index_tts2_assets(model_path);

    runtime::ModelMetadata metadata;
    metadata.family = "index_tts2";
    metadata.variant = assets->config.version;
    metadata.description = "IndexTTS2 loaded from local extracted assets.";

    runtime::CapabilitySet capabilities;
    capabilities.supported_tasks = {
        {runtime::VoiceTaskKind::Tts, {runtime::RunMode::Offline}},
        {runtime::VoiceTaskKind::VoiceCloning, {runtime::RunMode::Offline}},
    };
    capabilities.supports_speaker_reference = true;
    capabilities.supports_style_condition = true;
    capabilities.languages = {"English", "Chinese"};

    return std::make_unique<IndexTTS2LoadedModel>(
        std::move(metadata),
        std::move(capabilities),
        std::move(assets));
}

std::shared_ptr<runtime::IVoiceModelLoader> make_index_tts2_loader() {
    return std::make_shared<IndexTTS2Loader>();
}

}  // namespace engine::models::index_tts2
