#pragma once

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/runtime/session_base.h"
#include "engine/models/hviske_asr/assets.h"
#include "engine/models/hviske_asr/decoder.h"
#include "engine/models/hviske_asr/encoder.h"
#include "engine/models/hviske_asr/frontend.h"
#include "engine/models/hviske_asr/weights.h"

#include <cstddef>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace engine::models::hviske_asr {

class HviskeASRSession final
    : public runtime::RuntimeSessionBase
    , public runtime::IOfflineVoiceTaskSession {
public:
    HviskeASRSession(
        runtime::TaskSpec task,
        runtime::SessionOptions options,
        std::shared_ptr<const HviskeAssets> assets);
    ~HviskeASRSession() override;

    std::string family() const override;
    runtime::VoiceTaskKind task_kind() const override;
    runtime::RunMode run_mode() const override;
    void prepare(const runtime::SessionPreparationRequest & request) override;
    runtime::TaskResult run(const runtime::TaskRequest & request) override;

private:
    struct Segment {
        runtime::AudioBuffer audio;
    };

    std::vector<Segment> prepare_segments(
        const runtime::AudioBuffer & audio,
        const std::unordered_map<std::string, std::string> & options) const;
    std::string language_for_request(const runtime::TaskRequest & request) const;
    bool punctuation_for_request(const runtime::TaskRequest & request) const;
    HviskeDecodingOptions decoding_options_for_request(const runtime::TaskRequest & request) const;

    runtime::TaskSpec task_;
    std::shared_ptr<const HviskeAssets> assets_;
    std::shared_ptr<const HviskeWeights> weights_;
    size_t weight_context_bytes_ = 512ull * 1024ull * 1024ull;
    size_t encoder_graph_arena_bytes_ = 512ull * 1024ull * 1024ull;
    size_t decoder_prefill_graph_arena_bytes_ = 512ull * 1024ull * 1024ull;
    size_t decoder_decode_graph_arena_bytes_ = 512ull * 1024ull * 1024ull;
    engine::assets::TensorStorageType matmul_weight_storage_type_ = engine::assets::TensorStorageType::Native;
    engine::assets::TensorStorageType conv_weight_storage_type_ = engine::assets::TensorStorageType::Native;
    HviskeFrontend frontend_;
    std::unique_ptr<HviskeEncoderRuntime> encoder_;
    std::unique_ptr<HviskeDecoderRuntime> decoder_;
};

}  // namespace engine::models::hviske_asr
