#include "engine/models/index_tts2_5/loader.h"

#include "engine/framework/model_spec/package.h"
#include "engine/models/index_tts2_5/session.h"

#include <stdexcept>
#include <utility>

namespace engine::models::index_tts2_5 {
namespace {

runtime::ModelMetadata metadata(const IndexTTS25Assets & assets) {
    runtime::ModelMetadata out;
    out.family = "index_tts2_5";
    out.variant = assets.config.version;
    out.description = "IndexTTS2.5 loaded from local extracted assets.";
    return out;
}

runtime::CapabilitySet capabilities(const IndexTTS25Assets &) {
    runtime::CapabilitySet out;
    out.supported_tasks = {
        {runtime::VoiceTaskKind::Tts, {runtime::RunMode::Offline}},
        {runtime::VoiceTaskKind::VoiceCloning, {runtime::RunMode::Offline}},
    };
    out.supports_speaker_reference = true;
    out.supports_style_condition = true;
    out.languages = {"Chinese", "English", "Japanese", "Spanish", "Arabic"};
    return out;
}

runtime::ModelCliInterface cli(const IndexTTS25Assets &) {
    runtime::ModelCliInterface out;
    out.request_options = {
        {"lang", "auto|zh|en|ja|es|ar|...", "Text language hint; auto infers zh when the text contains Han characters, otherwise en."},
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
    out.session_options = {
        {"index_tts2_5.weight_type", "native|f32|f16|bf16|q8_0", "Matmul weight storage type."},
        {"index_tts2_5.conv_weight_type", "native|f32|f16", "Convolution weight storage type."},
        {"index_tts2_5.gpt_graph_arena_mb", "n", "GPT graph arena size."},
        {"index_tts2_5.s2mel_graph_arena_mb", "n", "S2Mel graph arena size."},
        {"index_tts2_5.reference_graph_arena_mb", "n", "Reference encoder and codec graph arena size."},
        {"index_tts2_5.emotion_text_prefill_graph_arena_mb", "n", "Emotion-text prefill graph arena size."},
        {"index_tts2_5.emotion_text_decode_graph_arena_mb", "n", "Emotion-text cached-step graph arena size."},
        {"index_tts2_5.emotion_text_max_new_tokens", "n", "Maximum generated tokens for emotion-text classification; default 256."},
        {"index_tts2_5.weight_context_mb", "n", "Shared weight context size."},
        {"index_tts2_5.mem_saver", "true|false", "Release staged reference and conditioning graphs after request phases; default false."},
        {"index_tts2_5.speaker_cache_slots", "n", "Prepared speaker-reference cache slots; default 1."},
        {"index_tts2_5.emotion_cache_slots", "n", "Prepared emotion-reference cache slots; default 1."},
        {"index_tts2_5.emotion_text_cache_slots", "n", "Emotion-text weight cache slots; default 1."},
    };
    return out;
}

class IndexTTS25Loader final : public runtime::IVoiceModelLoader {
public:
    std::string family() const override {
        return "index_tts2_5";
    }

    runtime::CapabilitySet advertised_capabilities() const override {
        runtime::CapabilitySet out;
        out.supported_tasks = {
            {runtime::VoiceTaskKind::Tts, {runtime::RunMode::Offline}},
            {runtime::VoiceTaskKind::VoiceCloning, {runtime::RunMode::Offline}},
        };
        out.supports_speaker_reference = true;
        out.supports_style_condition = true;
        return out;
    }

    bool can_load(const runtime::ModelLoadRequest & request) const override {
        try {
            const auto package_spec = engine::model_spec::default_spec_path(family());
            (void) engine::model_spec::load_resource_bundle(
                request.model_path,
                package_spec);
            return !request.family_hint.has_value() || *request.family_hint == family();
        } catch (...) {
            return false;
        }
    }

    runtime::ModelInspection inspect(const runtime::ModelLoadRequest & request) const override {
        const auto assets = load_index_tts2_5_assets(request.model_path);
        runtime::ModelInspection inspection;
        inspection.model_root = assets->resources.model_root();
        inspection.metadata = metadata(*assets);
        inspection.capabilities = capabilities(*assets);
        inspection.cli = cli(*assets);
        const auto package_spec = engine::model_spec::default_spec_path(family());
        inspection.discovered_configs = runtime::discover_named_assets_from_package_spec(
            request.model_path,
            package_spec,
            engine::model_spec::ResourceKind::Files);
        inspection.discovered_weights = runtime::discover_named_assets_from_package_spec(
            request.model_path,
            package_spec,
            engine::model_spec::ResourceKind::Tensors);
        return inspection;
    }

    std::unique_ptr<runtime::ILoadedVoiceModel> load(const runtime::ModelLoadRequest & request) const override {
        return load_index_tts2_5_model(request.model_path);
    }
};

}  // namespace

IndexTTS25LoadedModel::IndexTTS25LoadedModel(
    runtime::ModelMetadata metadata,
    runtime::CapabilitySet capabilities,
    std::shared_ptr<const IndexTTS25Assets> assets)
    : metadata_(std::move(metadata)),
      capabilities_(std::move(capabilities)),
      assets_(std::move(assets)) {}

const runtime::ModelMetadata & IndexTTS25LoadedModel::metadata() const noexcept {
    return metadata_;
}

const runtime::CapabilitySet & IndexTTS25LoadedModel::capabilities() const noexcept {
    return capabilities_;
}

std::unique_ptr<runtime::IVoiceTaskSession> IndexTTS25LoadedModel::create_task_session(
    const runtime::TaskSpec & task,
    const runtime::SessionOptions & options) const {
    if (task.mode != runtime::RunMode::Offline ||
        (task.task != runtime::VoiceTaskKind::Tts && task.task != runtime::VoiceTaskKind::VoiceCloning)) {
        throw std::runtime_error("IndexTTS2.5 only supports offline TTS and voice-cloning sessions");
    }
    return std::make_unique<IndexTTS25Session>(task, options, assets_);
}

std::unique_ptr<IndexTTS25LoadedModel> load_index_tts2_5_model(const std::filesystem::path & model_path) {
    auto assets = load_index_tts2_5_assets(model_path);
    return std::make_unique<IndexTTS25LoadedModel>(
        metadata(*assets),
        capabilities(*assets),
        std::move(assets));
}

std::shared_ptr<runtime::IVoiceModelLoader> make_index_tts2_5_loader() {
    return std::make_shared<IndexTTS25Loader>();
}

}  // namespace engine::models::index_tts2_5
