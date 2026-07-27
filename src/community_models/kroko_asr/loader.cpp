#include "engine/community_models/kroko_asr/loader.h"

#include "engine/community_models/kroko_asr/session.h"
#include "engine/framework/model_spec/package.h"

#include <algorithm>
#include <cctype>
#include <stdexcept>
#include <utility>

namespace engine::models::kroko_asr {
namespace {

std::string language(const KrokoASRAssets & assets) {
    std::string value = assets.config.language;
    std::transform(
        value.begin(),
        value.end(),
        value.begin(),
        [](unsigned char character) {
            return static_cast<char>(std::tolower(character));
        });
    if (value == "iw") {
        return "he";
    }
    return value.empty() || value == "auto" ? "en" : value;
}

runtime::ModelMetadata metadata(
    const KrokoASRAssets & assets) {
    runtime::ModelMetadata result;
    result.family = "kroko_asr";
    result.variant = assets.config.variant;
    result.description =
        "Community Kroko Zipformer2 RNN-T ASR port with a native "
        "Kaldi-compatible frontend and ggml encoder execution.";
    return result;
}

runtime::CapabilitySet capabilities(
    const KrokoASRAssets & assets) {
    runtime::CapabilitySet result;
    result.supported_tasks = {
        {runtime::VoiceTaskKind::Asr,
         {runtime::RunMode::Offline,
          runtime::RunMode::Streaming}},
    };
    result.languages = {language(assets)};
    result.supports_timestamps = true;
    return result;
}

runtime::ModelCliInterface cli() {
    runtime::ModelCliInterface result;
    result.request_options = {
        {"language",
         "code",
         "Language code matching the converted Kroko package."},
        {"decoding_method",
         "greedy_search|modified_beam_search",
         "RNN-T decoder; greedy_search preserves the parity-tested default.",
         false,
         std::string("greedy_search")},
        {"max_active_paths",
         "n",
         "Maximum hypotheses retained by modified beam search.",
         false,
         std::string("4"),
         std::string("1"),
         std::string("64")},
        {"blank_penalty",
         "score",
         "Non-negative score subtracted from the blank logit.",
         false,
         std::string("0")},
        {"hotwords",
         "phrase[/phrase...]",
         "Slash- or newline-separated text phrases; requires modified beam search."},
        {"hotwords_score",
         "score",
         "Non-negative per-token hotword context boost.",
         false,
         std::string("1.5")},
        {"enable_endpoint",
         "true|false",
         "Emit automatic endpoint speech segments.",
         false,
         std::string("false")},
        {"rule1_min_trailing_silence",
         "seconds",
         "Endpoint after this silence duration even without decoded speech.",
         false,
         std::string("2.4")},
        {"rule2_min_trailing_silence",
         "seconds",
         "Endpoint after this silence duration once speech was decoded.",
         false,
         std::string("1.2")},
        {"rule3_min_utterance_length",
         "seconds",
         "Endpoint after this utterance duration regardless of silence.",
         false,
         std::string("20")},
    };
    return result;
}

class KrokoASRLoader final
    : public runtime::IVoiceModelLoader {
public:
    std::string family() const override {
        return "kroko_asr";
    }

    runtime::CapabilitySet advertised_capabilities()
        const override {
        runtime::CapabilitySet result;
        result.supported_tasks = {
            {runtime::VoiceTaskKind::Asr,
             {runtime::RunMode::Offline,
              runtime::RunMode::Streaming}},
        };
        result.languages = {
            "de", "en", "es", "fr", "it",
            "he", "nl", "pt", "sv", "tr",
        };
        result.supports_timestamps = true;
        return result;
    }

    bool can_load(
        const runtime::ModelLoadRequest & request)
        const override {
        if (request.family_hint.has_value() &&
            *request.family_hint != family()) {
            return false;
        }
        try {
            (void)model_spec::load_resource_bundle(
                request.model_path,
                model_spec::default_spec_path(family()));
            return true;
        } catch (...) {
            return false;
        }
    }

    runtime::ModelInspection inspect(
        const runtime::ModelLoadRequest & request)
        const override {
        const auto assets =
            load_kroko_asr_assets(request.model_path);
        runtime::ModelInspection result;
        result.model_root = assets->resources.model_root();
        result.metadata = metadata(*assets);
        result.capabilities = capabilities(*assets);
        result.cli = cli();
        const auto spec =
            model_spec::default_spec_path(family());
        result.discovered_configs =
            runtime::discover_named_assets_from_package_spec(
                request.model_path,
                spec,
                model_spec::ResourceKind::Files);
        result.discovered_weights =
            runtime::discover_named_assets_from_package_spec(
                request.model_path,
                spec,
                model_spec::ResourceKind::Tensors);
        return result;
    }

    std::unique_ptr<runtime::ILoadedVoiceModel> load(
        const runtime::ModelLoadRequest & request)
        const override {
        return load_kroko_asr_model(request.model_path);
    }
};

}  // namespace

KrokoASRLoadedModel::KrokoASRLoadedModel(
    runtime::ModelMetadata metadata,
    runtime::CapabilitySet capabilities,
    std::shared_ptr<const KrokoASRAssets> assets)
    : metadata_(std::move(metadata)),
      capabilities_(std::move(capabilities)),
      assets_(std::move(assets)) {}

const runtime::ModelMetadata &
KrokoASRLoadedModel::metadata() const noexcept {
    return metadata_;
}

const runtime::CapabilitySet &
KrokoASRLoadedModel::capabilities() const noexcept {
    return capabilities_;
}

std::unique_ptr<runtime::IVoiceTaskSession>
KrokoASRLoadedModel::create_task_session(
    const runtime::TaskSpec & task,
    const runtime::SessionOptions & options) const {
    return std::make_unique<KrokoASRSession>(
        task, options, assets_);
}

std::unique_ptr<KrokoASRLoadedModel>
load_kroko_asr_model(
    const std::filesystem::path & model_path) {
    auto assets = load_kroko_asr_assets(model_path);
    return std::make_unique<KrokoASRLoadedModel>(
        metadata(*assets),
        capabilities(*assets),
        std::move(assets));
}

std::shared_ptr<runtime::IVoiceModelLoader>
make_kroko_asr_loader() {
    return std::make_shared<KrokoASRLoader>();
}

}  // namespace engine::models::kroko_asr
