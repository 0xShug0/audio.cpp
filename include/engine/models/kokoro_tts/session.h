#pragma once

#include "engine/framework/model_spec/metadata.h"
#include "engine/framework/runtime/session_base.h"
#include "engine/models/kokoro_tts/assets.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace kokoro_ggml {
class KokoroDecoderRuntime;
class KokoroPredictorRuntime;
}

namespace engine::models::kokoro_tts {

std::shared_ptr<runtime::IVoiceModelLoader> make_kokoro_tts_loader();

/// One timing per PHONEME GROUP -- a run of tokens between the space tokens Kokoro's vocabulary
/// carries -- from the durations its duration predictor produced, appended to `out`.
///
/// A group is NOT a written word: on the text path eSpeak-ng merges function words, so
/// `on the` arrives as one group. See the note on the definition in session.cpp.
///
/// Declared here rather than kept in session.cpp's anonymous namespace so it can be tested
/// directly: it is a pure function of its arguments, and the pad/space boundaries, the
/// punctuation-only groups and the frames->samples scale are all worth pinning down without
/// standing up a session. It takes the vocabulary rather than the whole KokoroAssets for the same
/// reason -- that is all it needs.
///
/// `chunk_start_sample` offsets the spans into a buffer several chunks are being merged into.
/// Reports nothing at all rather than throwing if the token and duration counts disagree: the
/// audio is the product and an empty word list is a state every caller already handles.
void append_kokoro_word_timings(
    std::vector<runtime::WordTimestamp> & out,
    const std::vector<int32_t> & input_ids,
    const std::vector<int32_t> & durations,
    const std::unordered_map<std::string, int32_t> & vocab,
    size_t chunk_samples,
    int64_t chunk_start_sample);

struct KokoroSynthesisInput;
class KokoroTTSSession final
    : public runtime::RuntimeSessionBase
    , public runtime::IOfflineVoiceTaskSession {
public:
    KokoroTTSSession(
        runtime::TaskSpec task,
        runtime::SessionOptions options,
        std::shared_ptr<const KokoroAssets> assets,
        std::shared_ptr<const engine::model_spec::ModelContract> contract);
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

    runtime::MappedGraphCapacityAdapter make_graph_capacity_adapter();
    std::vector<int64_t> prepared_graph_capacities() const;
    DecoderCapacityContract make_decoder_capacity_contract(int64_t decoder_frame_capacity) const;
    void prepare_graph_capacity(int64_t capacity);
    void prepare_decoder_graph_capacity(int64_t capacity);

    runtime::TaskSpec task_;
    std::shared_ptr<const KokoroAssets> assets_;
    std::shared_ptr<const engine::model_spec::ModelContract> contract_;
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
    int64_t prepared_decoder_capacity_ = 0;
    std::unique_ptr<kokoro_ggml::KokoroDecoderRuntime> prepared_decoder_;
    DecoderCapacityContract prepared_decoder_context_ = {};
    int64_t prepared_session_capacity_ = 0;
    std::unique_ptr<kokoro_ggml::KokoroPredictorRuntime> prepared_predictor_;
};

}  // namespace engine::models::kokoro_tts
