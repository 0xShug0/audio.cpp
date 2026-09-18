#pragma once

#include "engine/community_models/sortformer_diar/aosc_state.h"
#include "engine/community_models/sortformer_diar/encoder.h"
#include "engine/community_models/sortformer_diar/frontend.h"
#include "engine/community_models/sortformer_diar/stream_schedule.h"
#include "engine/community_models/sortformer_diar/weights.h"
#include "engine/framework/model_spec/metadata.h"
#include "engine/framework/runtime/session_base.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace engine::community_models::sortformer_diar {

std::shared_ptr<runtime::IVoiceModelLoader> make_sortformer_diar_v2_loader();

class SortformerV2Session final
    : public runtime::RuntimeSessionBase
    , public runtime::IOfflineVoiceTaskSession
    , public runtime::IStreamingVoiceTaskSession {
public:
    SortformerV2Session(
        runtime::TaskSpec task,
        runtime::SessionOptions options,
        std::shared_ptr<const SortformerV2Assets> assets,
        std::shared_ptr<const engine::model_spec::ModelContract> contract);
    ~SortformerV2Session() override;

    std::string family() const override;
    runtime::VoiceTaskKind task_kind() const override;
    runtime::RunMode run_mode() const override;
    void prepare(const runtime::SessionPreparationRequest & request) override;

    runtime::TaskResult run(const runtime::TaskRequest & request) override;

    runtime::StreamingPolicy streaming_policy() const override;
    void start_stream(const runtime::TaskRequest & request) override;
    void set_stream_event_sink(runtime::StreamEventCallback sink) override;
    void reset() override;
    runtime::StreamEvent process_audio_chunk(const runtime::AudioChunk & chunk) override;
    runtime::TaskResult finalize() override;

private:
    struct DecodeConfig {
        float threshold = 0.5f;
        int64_t min_frames = 0;
        int64_t pad_frames = 0;
    };

    DecodeConfig decode_config(const std::unordered_map<std::string, std::string> & options) const;
    std::vector<runtime::SpeakerTurn> decode_turns(
        const std::vector<float> & probabilities,
        int64_t frames,
        const DecodeConfig & config,
        bool include_open_turns) const;
    void prepare_graph(
        const SortformerV2FeatureBatch & features,
        int64_t state_frames,
        int64_t valid_frames);
    std::vector<float> run_graph(
        const SortformerV2FeatureBatch & features,
        int64_t state_frames,
        int64_t valid_frames,
        const std::vector<float> * state,
        std::vector<float> * chunk_embeddings);
    runtime::StreamEvent process_windows(
        const std::vector<SortformerV2StreamWindow> & windows);
    runtime::StreamEvent process_window(const SortformerV2StreamWindow & window);
    runtime::TaskResult finalize_stream_result();

    runtime::TaskSpec task_;
    std::shared_ptr<const SortformerV2Assets> assets_;
    std::shared_ptr<const engine::model_spec::ModelContract> contract_;
    std::shared_ptr<const SortformerV2Weights> weights_;
    SortformerV2DiarGeometry geometry_;
    SortformerV2DiarScoringConfig scoring_;
    std::unique_ptr<SortformerV2StreamScheduler> scheduler_;
    std::unique_ptr<SortformerV2AoscState> aosc_;
    std::unique_ptr<SortformerV2ChannelBirthGate> birth_gate_;
    std::unique_ptr<SortformerV2InferenceGraph> graph_;
    DecodeConfig streaming_decode_config_;
    DecodeConfig default_decode_config_;
    runtime::StreamEventCallback stream_event_sink_;
    std::vector<float> probabilities_;
    int64_t probability_frames_ = 0;
    size_t emitted_stream_turns_ = 0;
    bool stream_started_ = false;
    bool finalized_ = false;
    size_t graph_arena_bytes_ = 1024ull * 1024ull * 1024ull;
    size_t weight_context_bytes_ = 1024ull * 1024ull * 1024ull;
    assets::TensorStorageType matmul_weight_storage_type_ = assets::TensorStorageType::F32;
    assets::TensorStorageType conv_weight_storage_type_ = assets::TensorStorageType::F32;
};

}  // namespace engine::community_models::sortformer_diar
