#include "engine/community_models/parakeet_tdt/implementation.h"
#include "engine/community_models/parakeet_tdt/bark3.h"
#include "engine/community_models/parakeet_tdt/kda_predictor.h"
#include "engine/community_models/parakeet_tdt/phonon_streaming.h"
#include "engine/community_models/parakeet_tdt/session.h"
#include "engine/framework/runtime/spec_backed_model.h"

#include <algorithm>
#include <array>
#include <stdexcept>
#include <utility>

namespace engine::community_models::parakeet_tdt {
namespace {

template <class Session>
std::unique_ptr<runtime::IVoiceTaskSession> create_session(
    const runtime::TaskSpec & task, const runtime::SessionOptions & options,
    std::shared_ptr<const ParakeetTDTAssets> assets, std::shared_ptr<const model_spec::ModelContract> contract) {
    return std::make_unique<Session>(task, options, std::move(assets), std::move(contract));
}

std::unique_ptr<runtime::IVoiceTaskSession> create_bark_offline(
    const runtime::TaskSpec & task, const runtime::SessionOptions & options,
    std::shared_ptr<const ParakeetTDTAssets> assets, std::shared_ptr<const model_spec::ModelContract> contract) {
    return wrap_bark3_offline(std::make_unique<ParakeetTDTOfflineSession>(
        task, options, std::move(assets), std::move(contract)));
}

std::unique_ptr<runtime::IVoiceTaskSession> create_bark_streaming(
    const runtime::TaskSpec & task, const runtime::SessionOptions & options,
    std::shared_ptr<const ParakeetTDTAssets> assets, std::shared_ptr<const model_spec::ModelContract> contract) {
    return wrap_bark3_streaming(std::make_unique<ParakeetTDTStreamingSession>(
        task, options, std::move(assets), std::move(contract)));
}

std::unique_ptr<WindowFrontend> create_bark_frontend(std::shared_ptr<const ParakeetTDTAssets> assets) {
    return std::make_unique<BarkStreamingFrontend>(std::move(assets));
}

const auto default_encoder_options = +[](runtime::GraphOptimizationBackend backend) {
    return runtime::graph_optimization_options_for_backend(backend);
};

}  // namespace

const ParakeetImplementation kParakeetImplementation{
    create_session<ParakeetTDTOfflineSession>, create_session<ParakeetTDTStreamingSession>,
    load_lstm_predictor, default_encoder_options};

const ParakeetImplementation kPhononImplementation = [] {
    auto implementation = kParakeetImplementation;
    implementation.streaming = create_session<PhononStreamingSession>;
    implementation.hotwords = true;
    return implementation;
}();

const ParakeetImplementation kBarkImplementation = [] {
    auto implementation = kParakeetImplementation;
    implementation.offline = create_bark_offline;
    implementation.streaming = create_bark_streaming;
    implementation.encoder_graph_options = bark3_encoder_graph_options;
    implementation.window_frontend = create_bark_frontend;
    implementation.input_prefix_samples = kBark3InputPrefixSamples;
    implementation.hotwords = true;
    implementation.hotword_name = "Bark";
    implementation.invalid_request_errors = true;
    implementation.window_timestamps = remove_bark3_prefix_timestamps;
    return implementation;
}();

const ParakeetImplementation kHaasImplementation = [] {
    auto implementation = kParakeetImplementation;
    implementation.load_predictor = load_kda_predictor;
    return implementation;
}();

const ParakeetImplementation & implementation_for_family(std::string_view family) {
    struct Registration {
        std::string_view family;
        const ParakeetImplementation * implementation;
    };
    static constexpr std::array<Registration, 4> registrations{{
        {"parakeet_tdt", &kParakeetImplementation},
        {"phonon2_asr", &kPhononImplementation},
        {"underdog_bark3", &kBarkImplementation},
        {"haas_asr", &kHaasImplementation},
    }};
    const auto found = std::find_if(registrations.begin(), registrations.end(),
        [family](const auto & entry) { return entry.family == family; });
    if (found == registrations.end()) {
        throw std::runtime_error("unregistered Parakeet implementation family: " + std::string(family));
    }
    return *found->implementation;
}

namespace {
std::shared_ptr<runtime::IVoiceModelLoader> make_family_loader(std::string family) {
    (void)implementation_for_family(family);
    runtime::SpecBackedVoiceModelConfig<ParakeetTDTAssets> config;
    config.family = family;
    config.load_assets = [family](const auto & path) {
        return load_parakeet_assets(path, family);
    };
    config.create_session = [](const runtime::TaskSpec & task, const runtime::SessionOptions & options,
        std::shared_ptr<const ParakeetTDTAssets> assets, std::shared_ptr<const model_spec::ModelContract> contract) {
        const auto & implementation = *assets->implementation;
        const auto factory = task.mode == runtime::RunMode::Streaming
            ? implementation.streaming : implementation.offline;
        return factory(task, options, std::move(assets), std::move(contract));
    };
    return runtime::make_spec_backed_voice_loader(std::move(config));
}
}  // namespace

std::shared_ptr<runtime::IVoiceModelLoader> make_parakeet_tdt_loader() {
    return make_family_loader("parakeet_tdt");
}

std::shared_ptr<runtime::IVoiceModelLoader> make_phonon2_asr_loader() {
    return make_family_loader("phonon2_asr");
}

std::shared_ptr<runtime::IVoiceModelLoader> make_underdog_bark3_loader() {
    return make_family_loader("underdog_bark3");
}

std::shared_ptr<runtime::IVoiceModelLoader> make_haas_asr_loader() {
    return make_family_loader("haas_asr");
}

}  // namespace engine::community_models::parakeet_tdt
