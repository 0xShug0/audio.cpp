#pragma once

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/runtime/session_base.h"
#include "engine/framework/runtime/streaming_tts_session.h"
#include "engine/models/supertonic/assets.h"
#include "engine/models/supertonic/tokenizer_text.h"

#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace engine::models::supertonic {

class SupertonicRuntime;

struct SupertonicGenerationOptions {
    int num_inference_steps = 8;
    float speaking_rate = 1.05F;
    uint32_t seed = 1234U;
    std::string voice = "M1";
    std::string language = "en";
};

class SupertonicSession final
    : public runtime::RuntimeSessionBase
    , public runtime::IOfflineVoiceTaskSession
    , public runtime::StreamingTtsSessionBase {
public:
    SupertonicSession(
        runtime::TaskSpec task,
        runtime::SessionOptions options,
        std::shared_ptr<const SupertonicAssets> assets);
    ~SupertonicSession() override;

    std::string family() const override;
    runtime::VoiceTaskKind task_kind() const override;
    runtime::RunMode run_mode() const override;
    void prepare(const runtime::SessionPreparationRequest & request) override;
    runtime::TaskResult run(const runtime::TaskRequest & request) override;

private:
    runtime::TaskResult generate_stream(const runtime::TaskRequest & request) override;
    void reset_stream_state() override {}
    SupertonicGenerationOptions generation_options_from_request(const runtime::TaskRequest & request) const;
    void validate_request(const runtime::TaskRequest & request) const;
    std::vector<runtime::TaskRequest> build_chunk_requests(const runtime::TaskRequest & request) const;
    runtime::AudioBuffer synthesize_chunk(const runtime::TaskRequest & request);

    runtime::TaskSpec task_;
    std::shared_ptr<const SupertonicAssets> assets_;
    SupertonicTextTokenizer tokenizer_;
    assets::TensorStorageType weight_storage_type_ = assets::TensorStorageType::Native;
    std::size_t style_cache_slots_ = 4;
    std::unique_ptr<SupertonicRuntime> runtime_;
};

}  // namespace engine::models::supertonic
