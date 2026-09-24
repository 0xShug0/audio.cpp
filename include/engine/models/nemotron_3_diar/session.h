#pragma once

#include "engine/models/nemotron_3_diar/encoder.h"
#include "engine/models/nemotron_3_diar/frontend.h"
#include "engine/models/nemotron_3_diar/streaming.h"
#include "engine/framework/model_spec/metadata.h"
#include "engine/framework/runtime/session_base.h"

#include <memory>
#include <set>
#include <tuple>
#include <unordered_map>

namespace engine::models::nemotron_3_diar {

std::shared_ptr<runtime::IVoiceModelLoader> make_nemotron_3_diar_loader();

class Session final
    : public runtime::RuntimeSessionBase
    , public runtime::IOfflineVoiceTaskSession
    , public runtime::IBatchedOfflineVoiceTaskSession
    , public runtime::IStreamingVoiceTaskSession {
public:
    Session(
        runtime::TaskSpec task,
        runtime::SessionOptions options,
        std::shared_ptr<const Assets> assets,
        std::shared_ptr<const model_spec::ModelContract> contract);
    ~Session() override;

    std::string family() const override;
    runtime::VoiceTaskKind task_kind() const override;
    runtime::RunMode run_mode() const override;
    void prepare(const runtime::SessionPreparationRequest & request) override;
    runtime::TaskResult run(const runtime::TaskRequest & request) override;
    std::vector<runtime::TaskResult> run_batch(const std::vector<runtime::TaskRequest> & requests) override;
    void run_batch(
        const std::vector<runtime::TaskRequest> & requests,
        const runtime::IBatchedOfflineVoiceTaskSession::ResultCallback & on_result) override;

    runtime::StreamingPolicy streaming_policy() const override;
    void start_stream(const runtime::TaskRequest & request) override;
    void set_stream_event_sink(runtime::StreamEventCallback sink) override;
    void reset() override;
    runtime::StreamEvent process_audio_chunk(const runtime::AudioChunk & chunk) override;
    runtime::TaskResult finalize() override;

private:
    struct DecodeConfig {
        float threshold = 0.5F;
        int64_t min_frames = 0;
        int64_t pad_frames = 0;
    };

    DecodeConfig decode_config(const std::unordered_map<std::string, std::string> & options) const;
    std::vector<runtime::SpeakerTurn> decode_turns(
        const std::vector<float> & probabilities,
        int64_t frames,
        const DecodeConfig & config,
        bool include_open_turns) const;
    std::vector<float> pre_encode(const FeatureBatch & features);
    std::vector<float> encode(
        const std::vector<float> & embeddings,
        int64_t batch,
        int64_t frames,
        const std::vector<int64_t> & valid_frames);
    void process_window_batch(
        const std::vector<StreamWindow> & windows,
        const std::vector<AoscState *> & states,
        const std::vector<std::vector<float> *> & timelines);
    void process_embedding_batch(
        const std::vector<float> & chunk_embeddings,
        int64_t batch,
        int64_t chunk_capacity,
        const std::vector<int64_t> & chunk_frames,
        const std::vector<int64_t> & central_frames,
        const std::vector<int64_t> & left_context_frames,
        const std::vector<int64_t> & right_context_frames,
        const std::vector<AoscState *> & states,
        const std::vector<std::vector<float> *> & timelines);
    runtime::StreamEvent process_stream_windows(const std::vector<StreamWindow> & windows);

    runtime::TaskSpec task_;
    std::shared_ptr<const Assets> assets_;
    std::shared_ptr<const model_spec::ModelContract> contract_;
    std::shared_ptr<const ModelWeights> weights_;
    std::unique_ptr<PreEncodeGraph> pre_encode_graph_;
    std::unique_ptr<EncoderGraph> encoder_graph_;
    StreamingConfig streaming_config_;
    std::unique_ptr<StreamScheduler> stream_scheduler_;
    std::unique_ptr<AoscState> stream_state_;
    runtime::TaskRequest stream_request_;
    std::vector<float> stream_probabilities_;
    int64_t stream_samples_ = 0;
    std::set<std::tuple<int64_t, int64_t, std::string>> emitted_stream_turns_;
    runtime::StreamEventCallback stream_event_sink_;
    bool stream_started_ = false;
    size_t graph_arena_bytes_ = 1024ull * 1024ull * 1024ull;
    size_t weight_context_bytes_ = 1024ull * 1024ull * 1024ull;
    bool use_flash_attention_ = true;
};

}  // namespace engine::models::nemotron_3_diar
