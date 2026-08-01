#pragma once

#include "engine/framework/runtime/session_base.h"
#include "engine/models/heartmula/assets.h"
#include "engine/models/heartmula/codec.h"
#include "engine/models/heartmula/generator.h"
#include "engine/models/heartmula/mula.h"
#include "engine/models/heartmula/tokenizer_text.h"

#include <cstddef>
#include <memory>
#include <string>

namespace engine::models::heartmula {

class HeartMuLaSession final
    : public runtime::RuntimeSessionBase
    , public runtime::IOfflineVoiceTaskSession {
public:
    HeartMuLaSession(
        runtime::TaskSpec task,
        runtime::SessionOptions options,
        std::shared_ptr<const HeartMuLaAssets> assets);

    std::string family() const override;
    runtime::VoiceTaskKind task_kind() const override;
    runtime::RunMode run_mode() const override;
    void prepare(const runtime::SessionPreparationRequest & request) override;
    runtime::TaskResult run(const runtime::TaskRequest & request) override;

private:
    HeartMuLaPromptRequest make_request(const runtime::TaskRequest & request) const;

    runtime::TaskSpec task_;
    std::shared_ptr<const HeartMuLaAssets> assets_;
    size_t mula_weight_context_bytes_ = 512ull * 1024ull * 1024ull;
    size_t mula_constant_context_bytes_ = 256ull * 1024ull * 1024ull;
    size_t mula_backbone_prefill_graph_arena_bytes_ = 1536ull * 1024ull * 1024ull;
    size_t mula_backbone_step_graph_arena_bytes_ = 1536ull * 1024ull * 1024ull;
    size_t mula_decoder_prefill_graph_arena_bytes_ = 512ull * 1024ull * 1024ull;
    size_t mula_decoder_step_graph_arena_bytes_ = 512ull * 1024ull * 1024ull;
    size_t mula_frame_embedding_graph_arena_bytes_ = 512ull * 1024ull * 1024ull;
    size_t codec_weight_context_bytes_ = 512ull * 1024ull * 1024ull;
    size_t codec_flow_estimator_graph_arena_bytes_ = 2048ull * 1024ull * 1024ull;
    size_t codec_conditioning_graph_arena_bytes_ = 512ull * 1024ull * 1024ull;
    size_t codec_scalar_decoder_graph_arena_bytes_ = 1536ull * 1024ull * 1024ull;
    assets::TensorStorageType mula_weight_storage_type_ = assets::TensorStorageType::Native;
    assets::TensorStorageType codec_weight_storage_type_ = assets::TensorStorageType::Native;
    bool mem_saver_ = false;
    HeartMuLaTextTokenizer text_tokenizer_;
    HeartMuLaWeightsRuntime mula_;
    HeartCodecWeightsRuntime codec_;
};

}  // namespace engine::models::heartmula
