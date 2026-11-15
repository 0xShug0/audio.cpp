#pragma once
#include "engine/framework/model_spec/package.h"
#include "engine/framework/core/execution_context.h"
#include "engine/framework/core/attention_fallback.h"
#include "engine/framework/modules/transformers/dit_runtime.h"
#include "engine/framework/runtime/session.h"
#include "engine/models/soulx_singer/score.h"
#include <memory>

namespace engine::models::soulx_singer {
struct SingerImplementation;
struct ModelAssets {
    assets::ResourceBundle resources;
    const SingerImplementation *impl = nullptr;
    std::map<std::string, int32_t> phones;
    std::shared_ptr<const assets::TensorSource> model, whisper, vocos, rmvpe;
    int64_t layers = 22, hidden = 1024, intermediate = 4096, heads = 16;
};
std::shared_ptr<const ModelAssets> load_assets(const std::filesystem::path &path);
struct EncodedAudio {
    std::vector<float> waveform, mel, content, f0;
    int64_t frames = 0;
};
struct GenerationOptions {
    int64_t steps = 32;
    float guidance = 3;
    int pitch_shift = 0;
    bool auto_pitch_shift = false;
    uint64_t seed = 2026;
};
class SoulXSingerRuntime {
  public:
    SoulXSingerRuntime(std::shared_ptr<const ModelAssets> assets, core::ExecutionContext &execution,
                       assets::TensorStorageType storage, core::AttentionPreference attention);
    ~SoulXSingerRuntime();
    EncodedAudio encode(const runtime::AudioBuffer &audio, bool reference);
    std::pair<EncodedAudio, EncodedAudio> encode_scores(const SingingScore &reference,
                                                        const SingingScore &target, int shift, bool melody,
                                                        const runtime::AudioBuffer &reference_audio);
    void prepare_conditioning(const EncodedAudio &reference, const EncodedAudio &source, int pitch_shift);
    std::vector<float> denoise(const std::vector<float> &noise, const GenerationOptions &options);
    std::vector<float> decode(const std::vector<float> &mel);
    runtime::AudioBuffer convert(const runtime::AudioBuffer &source, const runtime::AudioBuffer &reference,
                                 const GenerationOptions &options);

  private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace engine::models::soulx_singer
