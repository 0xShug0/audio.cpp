#pragma once

#include "engine/community_models/echo_tts/config.h"
#include "engine/framework/assets/resource_bundle.h"
#include "engine/framework/model_spec/metadata.h"
#include "engine/framework/runtime/cache_slots.h"
#include "engine/framework/runtime/session_base.h"
#include "engine/models/fish_audio/assets.h"
#include "engine/models/fish_audio/codec.h"

#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace engine::models::echo_tts {

class EchoDitRuntime;

struct EchoTtsAssets {
    assets::ResourceBundle resources;
    EchoTtsConfig config;
    EchoPcaState pca;
    std::shared_ptr<const assets::TensorSource> dit_weights;
    // The Fish S1-DAC autoencoder, packaged inside Echo's own GGUF. audio.cpp
    // implements this codec for the fish_audio family; Echo reuses the
    // implementation but supplies the S1 weights its PCA basis was fitted to.
    // See docs/community_models/echo_tts_autoencoder_reuse.md.
    std::shared_ptr<const fish_audio::FishAudioAssets> codec_assets;
};

// Encoding a speaker reference is linear in its length -- a 4.5-minute clip is
// ten Fish encode passes -- and the result depends only on the audio and the
// trim length, so it is cached across requests. Servers reusing a handful of
// voices then pay it once per voice rather than once per request.
struct EchoReferenceIdentity {
    std::string id;
    int64_t max_samples = 0;
};

struct EchoReferenceIdentityEqual {
    bool operator()(const EchoReferenceIdentity & a, const EchoReferenceIdentity & b) const {
        return a.max_samples == b.max_samples && a.id == b.id;
    }
};

struct EchoPreparedSpeaker {
    std::vector<float> latent;
    int64_t frames = 0;
};

std::shared_ptr<runtime::IVoiceModelLoader> make_echo_tts_loader();

class EchoTtsSession final
    : public runtime::RuntimeSessionBase,
      public runtime::IOfflineVoiceTaskSession {
public:
    EchoTtsSession(
        runtime::TaskSpec task,
        runtime::SessionOptions options,
        std::shared_ptr<const EchoTtsAssets> assets,
        std::shared_ptr<const engine::model_spec::ModelContract> contract);
    ~EchoTtsSession() override;

    std::string family() const override;
    runtime::VoiceTaskKind task_kind() const override;
    runtime::RunMode run_mode() const override;
    void prepare(const runtime::SessionPreparationRequest & request) override;
    runtime::TaskResult run(const runtime::TaskRequest & request) override;
    void reset();

private:
    // Reference trim limit: request option, else session default, else the
    // trained maximum. Returned in samples at the codec rate.
    int64_t resolve_reference_max_samples(
        const std::unordered_map<std::string, std::string> & request_options) const;
    EchoSamplerOptions parse_sampler_options(
        const std::unordered_map<std::string, std::string> & options) const;
    // Encodes reference audio to 80-D PCA latents, mirroring
    // inference.py::get_speaker_latent_and_mask.
    void encode_speaker(const runtime::AudioBuffer & audio);
    runtime::AudioBuffer synthesize_chunk(
        const std::string & text,
        const EchoSamplerOptions & sampler);

    runtime::TaskSpec task_;
    std::shared_ptr<const EchoTtsAssets> assets_;
    std::shared_ptr<const engine::model_spec::ModelContract> contract_;
    std::unique_ptr<EchoDitRuntime> dit_;
    std::unique_ptr<fish_audio::FishAudioCodecRuntime> codec_;
    int64_t reference_max_samples_ = 0;
    std::vector<float> speaker_latent_;
    int64_t speaker_frames_ = 0;
    runtime::CacheSlots<EchoReferenceIdentity, EchoPreparedSpeaker, EchoReferenceIdentityEqual>
        reference_cache_;
};

}  // namespace engine::models::echo_tts
