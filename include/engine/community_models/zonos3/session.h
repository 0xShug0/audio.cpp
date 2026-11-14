#pragma once

#include "engine/community_models/zonos3/assets.h"
#include "engine/community_models/zonos3/backbone.h"
#include "engine/community_models/zonos3/tokenizer.h"
#include "engine/framework/model_spec/metadata.h"
#include "engine/framework/runtime/cache_slots.h"
#include "engine/framework/runtime/session_base.h"
#include "engine/framework/runtime/streaming_tts_session.h"

#include <cstdint>
#include <memory>
#include <vector>

namespace engine::models::zonos3 {

class Zonos3CodecDecoder;
class Zonos3SpeakerEncoder;
class Zonos3ReferenceEncoder;

// A voice computed from reference audio: a speaker embedding or reference codes [frames][10].
struct Zonos3Voice {
    std::vector<float> embedding;
    std::vector<int32_t> codes;
};

class Zonos3Session final : public runtime::RuntimeSessionBase,
                            public runtime::IOfflineVoiceTaskSession,
                            public runtime::StreamingTtsSessionBase {
public:
    Zonos3Session(runtime::TaskSpec task, const runtime::SessionOptions & options,
                  std::shared_ptr<const Zonos3Assets> assets,
                  std::shared_ptr<const engine::model_spec::ModelContract> contract);
    ~Zonos3Session() override;
    std::string family() const override;
    runtime::VoiceTaskKind task_kind() const override;
    runtime::RunMode run_mode() const override;
    void prepare(const runtime::SessionPreparationRequest & request) override;
    runtime::TaskResult run(const runtime::TaskRequest & request) override;

private:
    struct Request;
    runtime::TaskResult generate_stream(const runtime::TaskRequest & request) override;
    void reset_stream_state() override {}
    runtime::TaskResult synthesize(const runtime::TaskRequest & request, bool streaming);
    Zonos3Voice encode_voice(const runtime::AudioBuffer & audio, bool reference_mode);

    runtime::TaskSpec task_;
    std::shared_ptr<const Zonos3Assets> assets_;
    std::shared_ptr<const engine::model_spec::ModelContract> contract_;
    Zonos3Tokenizer tokenizer_;
    std::unique_ptr<Zonos3Backbone> backbone_;
    std::unique_ptr<Zonos3CodecDecoder> decoder_;
    std::unique_ptr<core::ExecutionContext> encoder_execution_;
    std::unique_ptr<Zonos3SpeakerEncoder> speaker_encoder_;
    std::unique_ptr<Zonos3ReferenceEncoder> reference_encoder_;
    bool keep_reference_encoder_ = false;
    int stream_chunk_frames_ = 16;
    runtime::CacheSlots<uint64_t, Zonos3Voice> voices_;
    runtime::TaskRequest defaults_;
};

}  // namespace engine::models::zonos3
