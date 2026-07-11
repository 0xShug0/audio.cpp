#pragma once

#include "engine/framework/runtime/session_base.h"
#include "engine/models/moss_tts_local/assets.h"
#include "engine/models/moss_tts_local/backbone.h"
#include "engine/models/moss_tts_local/codec_decoder.h"
#include "engine/models/moss_tts_local/codec_encoder.h"
#include "engine/models/moss_tts_local/depth_transformer.h"
#include "engine/models/moss_tts_local/generator.h"
#include "engine/models/moss_tts_local/tokenizer_text.h"

#include <memory>

namespace engine::models::moss_tts_local {

// Offline TTS session: renders text into a 48 kHz stereo waveform by chaining the verified
// pieces -- the text processor builds the generation prefix, the generator (Qwen3 backbone +
// depth transformer) emits RVQ codes frame by frame, and the codec decoder turns those codes
// into audio. When a speaker reference is supplied, the codec encoder turns it into RLFQ
// codes that seed a voice-clone prompt.
class MossTTSLocalSession final
    : public runtime::RuntimeSessionBase
    , public runtime::IOfflineVoiceTaskSession {
public:
    MossTTSLocalSession(
        runtime::TaskSpec task,
        runtime::SessionOptions options,
        std::shared_ptr<const MossTTSLocalAssets> assets);

    std::string family() const override;
    runtime::VoiceTaskKind task_kind() const override;
    runtime::RunMode run_mode() const override;
    void prepare(const runtime::SessionPreparationRequest & request) override;
    runtime::TaskResult run(const runtime::TaskRequest & request) override;

private:
    MossCodecEncoder & encoder();

    runtime::TaskSpec task_;
    std::shared_ptr<const MossTTSLocalAssets> assets_;
    // Declared before the generator so the generator (which holds references to them) is
    // destroyed first.
    std::unique_ptr<MossBackboneRuntime> backbone_;
    std::unique_ptr<MossDepthTransformer> depth_;
    std::unique_ptr<MossTextProcessor> processor_;
    std::unique_ptr<MossCodecDecoder> codec_;
    std::unique_ptr<MossGenerator> generator_;
    // Lazily built the first time a speaker reference is provided (voice cloning).
    std::unique_ptr<MossCodecEncoder> encoder_;
};

}  // namespace engine::models::moss_tts_local
