#pragma once

#include "engine/community_models/kitten_tts/assets.h"
#include "engine/framework/audio/espeak_phonemizer.h"
#include "engine/framework/model_spec/metadata.h"
#include "engine/framework/runtime/session_base.h"

#include <cstddef>
#include <memory>

namespace engine::models::kitten_tts {
class KittenDecoderRuntime;
class KittenPredictorRuntime;
} // namespace engine::models::kitten_tts

namespace engine::models::kitten_tts {

std::shared_ptr<runtime::IVoiceModelLoader> make_kitten_tts_loader();

struct KittenSynthesisInput;
class KittenTTSSession final : public runtime::RuntimeSessionBase, public runtime::IOfflineVoiceTaskSession {
  public:
    KittenTTSSession(runtime::TaskSpec task, runtime::SessionOptions options,
                     std::shared_ptr<const KittenAssets> assets,
                     std::shared_ptr<const engine::model_spec::ModelContract> contract);
    ~KittenTTSSession() override;

    std::string family() const override;
    runtime::VoiceTaskKind task_kind() const override;
    runtime::RunMode run_mode() const override;
    void prepare(const runtime::SessionPreparationRequest &request) override;
    runtime::TaskResult run(const runtime::TaskRequest &request) override;

  private:
    struct DecoderCapacityContract {
        int64_t decoder_frame_capacity = 0;
        int64_t conditioning_sample_capacity = 0;
        int64_t conditioning_frame_capacity = 0;
    };

    runtime::MappedGraphCapacityAdapter make_graph_capacity_adapter();
    std::vector<int64_t> prepared_graph_capacities() const;
    DecoderCapacityContract make_decoder_capacity_contract(int64_t decoder_frame_capacity) const;
    void prepare_graph_capacity(int64_t capacity);
    void prepare_decoder_graph_capacity(int64_t capacity);

    runtime::TaskSpec task_;
    std::shared_ptr<const KittenAssets> assets_;
    std::shared_ptr<const engine::model_spec::ModelContract> contract_;
    std::unique_ptr<engine::audio::EspeakPhonemizer> phonemizer_;
    std::shared_ptr<const KittenWeights> weights_;
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
    std::unique_ptr<KittenSynthesisInput> cached_input_;
    int64_t prepared_decoder_capacity_ = 0;
    std::unique_ptr<KittenDecoderRuntime> prepared_decoder_;
    DecoderCapacityContract prepared_decoder_context_ = {};
    int64_t prepared_session_capacity_ = 0;
    std::unique_ptr<KittenPredictorRuntime> prepared_predictor_;
};

} // namespace engine::models::kitten_tts
