#pragma once

#include "engine/framework/runtime/session_base.h"
#include "engine/models/kokoro_tts/assets.h"

#include <memory>
#include <cstddef>

namespace kokoro_ggml {
class KokoroDecoderRuntime;
}

namespace engine::models::kokoro_tts {

struct KokoroSynthesisInput;
struct KokoroFrontendSessionState;

class KokoroTTSSession final
    : public runtime::RuntimeSessionBase
    , public runtime::IOfflineVoiceTaskSession {
public:
    KokoroTTSSession(
        runtime::TaskSpec task,
        runtime::SessionOptions options,
        std::shared_ptr<const KokoroAssets> assets);
    ~KokoroTTSSession() override;

    std::string family() const override;
    runtime::VoiceTaskKind task_kind() const override;
    runtime::RunMode run_mode() const override;
    void prepare(const runtime::SessionPreparationRequest & request) override;
    runtime::TaskResult run(const runtime::TaskRequest & request) override;

private:
    struct DecoderCapacityContract {
        int64_t decoder_frame_capacity = 0;
        int64_t conditioning_sample_capacity = 0;
        int64_t conditioning_frame_capacity = 0;
    };

    struct PreparedRuntime;

    runtime::MappedGraphCapacityAdapter make_graph_capacity_adapter();
    int64_t base_graph_capacity_tokens() const;
    std::vector<int64_t> prepared_graph_capacities() const;
    DecoderCapacityContract make_decoder_capacity_contract(int64_t decoder_frame_capacity) const;
    void prepare_graph_capacity(int64_t capacity);
    void prepare_decoder_graph_capacity(int64_t capacity);

    runtime::TaskSpec task_;
    std::shared_ptr<const KokoroAssets> assets_;
    std::shared_ptr<const kokoro_ggml::KokoroWeights> weights_;
    runtime::GraphCapacityController graph_capacity_controller_;
    int64_t fixed_token_capacity_ = 0;
    int64_t pre_tail_token_capacity_ = 0;
    uint64_t rng_seed_ = 0;
    engine::assets::TensorStorageType matmul_weight_storage_type_ = engine::assets::TensorStorageType::Native;
    engine::assets::TensorStorageType conv_weight_storage_type_ = engine::assets::TensorStorageType::Native;
    size_t weight_context_bytes_ = 512ull * 1024ull * 1024ull;
    size_t predictor_duration_graph_bytes_ = 384ull * 1024ull * 1024ull;
    size_t predictor_text_graph_bytes_ = 256ull * 1024ull * 1024ull;
    size_t predictor_tail_graph_bytes_ = 640ull * 1024ull * 1024ull;
    std::string cached_request_key_;
    std::unique_ptr<KokoroSynthesisInput> cached_input_;
    std::unique_ptr<KokoroFrontendSessionState> frontend_session_state_;
    int64_t prepared_decoder_capacity_ = 0;
    std::unique_ptr<kokoro_ggml::KokoroDecoderRuntime> prepared_decoder_;
    DecoderCapacityContract prepared_decoder_context_ = {};
    int64_t prepared_session_capacity_ = 0;
    std::unique_ptr<PreparedRuntime> prepared_session_;
};

}  // namespace engine::models::kokoro_tts
