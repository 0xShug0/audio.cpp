#include "engine/models/index_tts2/loader.h"

#include "engine/framework/model_spec/package.h"
#include "engine/models/index_tts2/session.h"

#include <stdexcept>
#include <utility>

namespace engine::models::index_tts2 {
namespace {

runtime::ModelMetadata metadata(const IndexTTS2Assets & assets) {
    runtime::ModelMetadata out;
    out.family = "index_tts2";
    out.variant = assets.config.version;
    out.description = index_tts2_variant_from_version(assets.config.version) == IndexTTS2Variant::kV2_5
        ? "IndexTTS2.5 (index_tts2 family variant) loaded from local extracted assets."
        : "IndexTTS2 loaded from local extracted assets.";
    return out;
}

runtime::CapabilitySet capabilities(const IndexTTS2Assets & assets) {
    runtime::CapabilitySet out;
    out.supported_tasks = {
        {runtime::VoiceTaskKind::Tts, {runtime::RunMode::Offline}},
        {runtime::VoiceTaskKind::VoiceCloning, {runtime::RunMode::Offline}},
    };
    out.supports_speaker_reference = true;
    out.supports_style_condition = true;
    if (index_tts2_variant_from_version(assets.config.version) == IndexTTS2Variant::kV2_5) {
        out.languages = {"Chinese", "English", "Japanese", "Spanish", "Arabic"};
    } else {
        out.languages = {"English", "Chinese"};
    }
    return out;
}

runtime::ModelCliInterface cli(const IndexTTS2Assets & assets) {
    runtime::ModelCliInterface out;
    out.request_options = {
        {"text", "text", "Text fallback when structured text input is absent; prompt is also accepted."},
        {"do_sample", "true|false", "Enable GPT sampling.", false, "true"},
        {"top_p", "float", "GPT nucleus sampling probability in (0, 1].", false, "0.8", std::nullopt, "1"},
        {"top_k", "n", "GPT top-k sampling limit.", false, "30", "1"},
        {"temperature", "float", "Positive GPT sampling temperature.", false, "0.8"},
        {"repetition_penalty", "float", "GPT repetition penalty.", false, "10"},
        {"max_tokens", "n", "Maximum generated GPT speech tokens.", false, "1500", "1"},
        {"seed", "int", "Generation seed; omitted chooses a random seed.", false, std::nullopt, "0", "4294967295"},
        {"emotion_alpha", "float", "Blend strength for explicit emotion conditioning.", false, "1", "0", "1"},
        {"emotion_vector", "float[,float...]", "Eight-value explicit emotion vector."},
        {"use_emotion_text", "bool", "Infer emotion from text instead of reference audio.", false, "false"},
        {"emotion_text", "text", "Text used when emotion-text conditioning is enabled."},
        {"use_random_emotion", "bool", "Use random emotion weights in the emotion mixer.", false, "false"},
        {"interval_silence_ms", "n", "Silence inserted between generated text chunks.", false, "200", "0"},
        {"text_chunk_mode", "default|tag_aware|japanese|endline", "Framework text chunking mode used when text_chunk_size is set.", false, "default"},
        {"length_penalty", "float", "GPT beam-search length penalty.", false, "0"},
        {"num_beams", "n", "GPT beam count.", false, "3", "1"},
        {"duration_scale", "float", "Positive waveform duration rescaling, supported by both IndexTTS2 and IndexTTS2.5. Default: 1.", false, "1"},
        {"text_chunk_size", "int", "Optional framework long-form text chunk size; omitted keeps the full input together. Must be positive."},
    };
    if (index_tts2_variant_from_version(assets.config.version) == IndexTTS2Variant::kV2_5) {
        out.request_options.push_back(
            {"language", "auto|zh|en|ja|es|ar|...", "Text language hint; auto infers zh when the text contains Han characters, otherwise en."});
    }
    out.session_options = {
        {"index_tts2.weight_type", "native|f32|f16|bf16|q8_0", "Matmul weight storage type."},
        {"index_tts2.conv_weight_type", "native|f32|f16", "Convolution weight storage type.", false, "native"},
        {"index_tts2.gpt_graph_arena_mb", "n", "GPT graph arena size.", false, "32", "1"},
        {"index_tts2.s2mel_graph_arena_mb", "n", "S2Mel graph arena size.", false, "32", "1"},
        {"index_tts2.reference_graph_arena_mb", "n", "Reference encoder and codec graph arena size.", false, "512", "1"},
        {"index_tts2.emotion_text_prefill_graph_arena_mb", "n", "Emotion-text prefill graph arena size.", false, "2048", "1"},
        {"index_tts2.emotion_text_decode_graph_arena_mb", "n", "Emotion-text cached-step graph arena size.", false, "512", "1"},
        {"index_tts2.emotion_text_max_tokens", "n", "Maximum generated tokens for emotion-text classification; default 256.", false, "256", "1"},
        {"index_tts2.weight_context_mb", "n", "Shared weight context size.", false, "32", "1"},
        {"index_tts2.mem_saver", "true|false", "Release staged reference and conditioning graphs after request phases; default false.", false, "false"},
        {"index_tts2.speaker_cache_slots", "n", "Prepared speaker-reference cache slots; default 1.", false, "1", "0"},
        {"index_tts2.emotion_cache_slots", "n", "Prepared emotion-reference cache slots; default 1.", false, "1", "0"},
        {"index_tts2.emotion_text_cache_slots", "n", "Emotion-text weight cache slots; default 1.", false, "1", "0"},
    };
    return out;
}

class IndexTTS2Loader final : public runtime::IVoiceModelLoader {
public:
    std::string family() const override {
        return "index_tts2";
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
        const auto assets = load_index_tts2_assets(request.model_path);
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
    return std::make_unique<IndexTTS2LoadedModel>(
        metadata(*assets),
        capabilities(*assets),
        std::move(assets));
}

std::shared_ptr<runtime::IVoiceModelLoader> make_index_tts2_loader() {
    return std::make_shared<IndexTTS2Loader>();
}

}  // namespace engine::models::index_tts2
