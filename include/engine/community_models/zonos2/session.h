#pragma once

#include "engine/community_models/zonos2/assets.h"
#include "engine/community_models/zonos2/backbone.h"
#include "engine/framework/model_spec/metadata.h"
#include "engine/framework/runtime/cache_slots.h"
#include "engine/framework/runtime/session_base.h"
#include "engine/framework/runtime/streaming_tts_session.h"

#include <cstdint>
#include <memory>
#include <vector>

namespace engine::models::zonos2 {

class Zonos2CodecDecoder;
class Zonos2SpeakerEncoder;

class Zonos2Session final : public runtime::RuntimeSessionBase,
                            public runtime::IOfflineVoiceTaskSession,
                            public runtime::StreamingTtsSessionBase {
public:
    Zonos2Session(runtime::TaskSpec task, const runtime::SessionOptions & options,
                  std::shared_ptr<const Zonos2Assets> assets,
                  std::shared_ptr<const engine::model_spec::ModelContract> contract);
    ~Zonos2Session() override;
    std::string family() const override;
    runtime::VoiceTaskKind task_kind() const override;
    runtime::RunMode run_mode() const override;
    void prepare(const runtime::SessionPreparationRequest & request) override;
    runtime::TaskResult run(const runtime::TaskRequest & request) override;

private:
    runtime::TaskResult generate_stream(const runtime::TaskRequest & request) override;
    void reset_stream_state() override {}
    runtime::TaskResult synthesize(const runtime::TaskRequest & request, bool streaming);
    std::vector<float> encode_voice(const runtime::AudioBuffer & audio);

    runtime::TaskSpec task_;
    std::shared_ptr<const Zonos2Assets> assets_;
    std::shared_ptr<const engine::model_spec::ModelContract> contract_;
    std::unique_ptr<Zonos2Backbone> backbone_;
    std::unique_ptr<Zonos2CodecDecoder> decoder_;
    std::unique_ptr<core::ExecutionContext> encoder_execution_;
    std::unique_ptr<Zonos2SpeakerEncoder> speaker_encoder_;
    int stream_chunk_frames_ = 40;
    int stream_context_frames_ = 24;
    runtime::CacheSlots<uint64_t, std::vector<float>> voices_;
    runtime::TaskRequest defaults_;
};

}  // namespace engine::models::zonos2
