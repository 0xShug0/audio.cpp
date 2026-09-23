#include "engine/community_models/vieneu_v3_turbo/loader.h"

#include "engine/framework/debug/trace.h"
#include "engine/framework/model_spec/package.h"
#include "engine/community_models/vieneu_v3_turbo/session.h"

#include <algorithm>
#include <string>
#include <string_view>
#include <stdexcept>
#include <utility>

namespace engine::models::vieneu_v3_turbo {
namespace {

std::vector<std::string> supported_languages(const VieNeuTTSConfig & config) {
    std::vector<std::string> languages;
    languages.reserve(config.talker.codec_language_id.size() + 1);
    languages.push_back("Auto");
    for (const auto & [language, id] : config.talker.codec_language_id) {
        (void) id;
        languages.push_back(language);
    }
    std::sort(languages.begin() + 1, languages.end());
    return languages;
}

runtime::ModelMetadata metadata(const VieNeuTTSAssets & assets) {
    runtime::ModelMetadata out;
    out.family = kFamily;
    out.variant = assets.config.tts_model_size + "-" + assets.config.tts_model_type;
    out.description = "VieNeu-TTS loaded from local extracted assets.";
    return out;
}

runtime::CapabilitySet capabilities(const VieNeuTTSAssets & assets) {
    runtime::CapabilitySet out;
    if (assets.config.variant == VieNeuTTSVariant::Base) {
        out.supported_tasks = {
            {runtime::VoiceTaskKind::Tts, {runtime::RunMode::Offline}},
        };
        out.supports_speaker_reference = true;
    }
    out.languages = supported_languages(assets.config);
    return out;
}

runtime::ModelCliInterface cli(const VieNeuTTSAssets &) {
    runtime::ModelCliInterface out;
    out.request_options = {
        {"speaker_embedding_file", "<path>", "Path to a 192-d speaker embedding file (comma-separated floats)."},
        {"speaker_embedding", "<csv>", "Comma-separated list of 192 speaker embedding float values."},
        {"reference_codes", "<ints>", "Pre-encoded reference codes inline (row-major, 16 per frame, comma- or space-separated) for callers holding a voice in memory."},
        {"reference_codes_file", "<path>", "Pre-encoded reference codes (one frame per line, 16 ints); replaces the codec encoder pass and makes --voice-ref optional."},
        {"reference_text", "<text>", "Transcript of the reference WAV (accepted for compatibility; v3 Turbo does not condition on it)."},
        {"x_vector_only_mode", "true|false", "Clone from the speaker embedding only, without reference codes (default false)."},
        {"encode_reference_only", "true|false", "Encode --voice-ref into reference codes and return them as an acoustic-tokens artifact - the text reference_codes_file reads - without generating audio (default false)."},
        {"max_tokens", "<int>", "Maximum generated frames per chunk, 80 ms each (default 300)."},
        {"do_sample", "true|false", "Sample the acoustic decoder (default true); false = greedy."},
        {"temperature", "<float>", "Sampling temperature for all 16 codebooks (default 0.8)."},
        {"top_k", "<int>", "Top-k for all codebooks (default 25)."},
        {"top_p", "<float>", "Nucleus limit within the top-k candidates (default 0.95)."},
        {"repetition_penalty", "<float>", "Penalty on codes seen in the recent window of each codebook (default 1.2)."},
        {"repetition_window", "<int>", "Frames each codebook remembers for the penalty (default 64; 0 = unbounded)."},
        {"babble_retries", "<int>", "Re-generations for a short chunk that kept talking past its text (default 2; 0 disables the guard)."},
        {"text_chunk_min", "<int>", "Chunks shorter than this many characters join a neighbour (default 20)."},
        {"frame_cap", "true|false", "Cap max_tokens by the phoneme count of the chunk (default true)."},
        {"subtalker_temperature", "<float>", "Acoustic decoder temperature override (defaults to temperature)."},
        {"subtalker_top_k", "<int>", "Acoustic decoder top-k override (defaults to top_k)."},
        {"subtalker_top_p", "<float>", "Acoustic decoder top-p override (defaults to top_p)."},
        {"subtalker_do_sample", "true|false", "Alias of do_sample for the acoustic decoder."},
        {"seed", "<int>", "Sampling seed (random when omitted)."},
        {"text_chunk_size", "<int>", "Maximum character budget per chunk of the phoneme string (default 200)."},
        {"codes_dump_file", "<path>", "Append prompt ids, reference codes and generated codes as text (parity debugging)."},
    };
    out.session_options = {
        {"vieneu_v3_turbo.g2p_dict", "<path>", "sea_g2p.bin; with it, --text is Vietnamese/English text instead of phonemes."},
        {"vieneu_v3_turbo.g2p_library", "<path>", "The sea-g2p shared library built with --features capi; found by name when unset."},
        {"vieneu_v3_turbo.mem_saver", "true|false", "Release the talker cached-step graph after each request; default false."},
        {"vieneu_v3_turbo.voice_prompt_cache_slots", "n", "Voice prompt cache slots; default 1."},
    };
    return out;
}

class VieNeuTTSLoader final : public runtime::IVoiceModelLoader {
public:
    std::string family() const override {
        return kFamily;
    }

    std::vector<std::string> family_aliases() const override {
        return {kLegacyFamily};
    }

    runtime::CapabilitySet advertised_capabilities() const override {
        runtime::CapabilitySet out;
        out.supported_tasks = {
            {runtime::VoiceTaskKind::Tts, {runtime::RunMode::Offline}},
            {runtime::VoiceTaskKind::VoiceDesign, {runtime::RunMode::Offline}},
        };
        out.supports_speaker_reference = true;
        out.supports_style_condition = true;
        return out;
    }

    bool can_load(const runtime::ModelLoadRequest & request) const override {
        try {
            const auto package_spec = resolve_package_spec_path();
            (void) engine::model_spec::load_resource_bundle(request.model_path, package_spec);
            if (!request.family_hint.has_value()) {
                return true;
            }
            const auto aliases = family_aliases();
            return *request.family_hint == family()
                || std::find(aliases.begin(), aliases.end(), *request.family_hint) != aliases.end();
        } catch (...) {
            return false;
        }
    }

    runtime::ModelInspection inspect(const runtime::ModelLoadRequest & request) const override {
        const auto assets = load_vieneu_v3_turbo_assets(request.model_path);
        runtime::ModelInspection inspection;
        inspection.model_root = assets->resources.model_root();
        inspection.metadata = metadata(*assets);
        inspection.capabilities = capabilities(*assets);
        inspection.cli = cli(*assets);
        const auto package_spec = resolve_package_spec_path();
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
        return load_vieneu_v3_turbo_model(request.model_path);
    }
};

}  // namespace

VieNeuTTSLoadedModel::VieNeuTTSLoadedModel(
    runtime::ModelMetadata metadata,
    runtime::CapabilitySet capabilities,
    std::shared_ptr<const VieNeuTTSAssets> assets)
    : metadata_(std::move(metadata)),
      capabilities_(std::move(capabilities)),
      assets_(std::move(assets)) {}

const runtime::ModelMetadata & VieNeuTTSLoadedModel::metadata() const noexcept {
    return metadata_;
}

const runtime::CapabilitySet & VieNeuTTSLoadedModel::capabilities() const noexcept {
    return capabilities_;
}

namespace {

// Session options were named `vietneu_tts.*` before this family was renamed. Map them onto
// the current prefix (and say so once) instead of ignoring them silently; a key that is
// unknown after mapping is then rejected by the session like any other unknown option.
runtime::SessionOptions map_legacy_session_options(const runtime::SessionOptions & options) {
    constexpr std::string_view kLegacyPrefix = "vietneu_tts.";
    runtime::SessionOptions mapped = options;
    bool mapped_any = false;
    for (auto it = mapped.options.begin(); it != mapped.options.end();) {
        if (it->first.rfind(kLegacyPrefix, 0) != 0) {
            ++it;
            continue;
        }
        const std::string current = std::string(kFamily) + "." + it->first.substr(kLegacyPrefix.size());
        mapped.options.emplace(current, it->second);
        it = mapped.options.erase(it);
        mapped_any = true;
    }
    if (mapped_any) {
        debug::log_message(
            debug::LogLevel::Info,
            kFamily,
            std::string("session options '") + std::string(kLegacyPrefix) + "*' are deprecated; use '" +
                std::string(kFamily) + ".*'");
    }
    return mapped;
}

}  // namespace

std::unique_ptr<runtime::IVoiceTaskSession> VieNeuTTSLoadedModel::create_task_session(
    const runtime::TaskSpec & task,
    const runtime::SessionOptions & options) const {
    if (task.mode != runtime::RunMode::Offline) {
        throw std::runtime_error("VieNeu-TTS TTS only supports offline sessions");
    }
    if (assets_->config.variant == VieNeuTTSVariant::Base && task.task != runtime::VoiceTaskKind::Tts) {
        throw std::runtime_error("VieNeu-TTS base TTS model only supports the Tts task");
    }
    return std::make_unique<VieNeuTTSSession>(task, map_legacy_session_options(options), assets_);
}

std::unique_ptr<VieNeuTTSLoadedModel> load_vieneu_v3_turbo_model(const std::filesystem::path & model_path) {
    auto assets = load_vieneu_v3_turbo_assets(model_path);
    return std::make_unique<VieNeuTTSLoadedModel>(metadata(*assets), capabilities(*assets), std::move(assets));
}

std::shared_ptr<runtime::IVoiceModelLoader> make_vieneu_v3_turbo_loader() {
    return std::make_shared<VieNeuTTSLoader>();
}

}  // namespace engine::models::vieneu_v3_turbo
