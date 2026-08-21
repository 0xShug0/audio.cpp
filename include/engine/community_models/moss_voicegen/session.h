#pragma once

#include "engine/community_models/moss_voicegen/assets.h"
#include "engine/community_models/moss_voicegen/backbone.h"
#include "engine/community_models/moss_voicegen/delay_decoder.h"
#include "engine/community_models/moss_voicegen/heads.h"
#include "engine/community_models/moss_voicegen/tokenizer_text.h"
#include "engine/framework/core/execution_context.h"
#include "engine/framework/runtime/session_base.h"
#include "engine/models/moss/shared/audio_tokenizer_decoder.h"
#include "engine/models/moss/shared/token_rows.h"

#include <cstddef>
#include <memory>
#include <optional>
#include <string>

namespace engine::models::moss_voicegen {

class MossVoiceGenSession final
    : public runtime::RuntimeSessionBase
    , public runtime::IOfflineVoiceTaskSession {
public:
    MossVoiceGenSession(
        runtime::TaskSpec task,
        runtime::SessionOptions options,
        std::shared_ptr<const MossVoiceGenAssets> assets);
    ~MossVoiceGenSession() override;

    std::string family() const override;
    runtime::VoiceTaskKind task_kind() const override;
    runtime::RunMode run_mode() const override;
    void prepare(const runtime::SessionPreparationRequest & request) override;
    runtime::TaskResult run(const runtime::TaskRequest & request) override;

private:
    struct GeneratedChunk {
        std::vector<int32_t> codes;  // [n_vq, frames] row-major
        int64_t codebooks = 0;
        int64_t frames = 0;
        bool started_audio = false;
        bool hit_frame_ceiling = false;
    };

    GeneratedChunk generate_chunk(
        const std::string & text,
        const std::string & instruction,
        const std::optional<std::string> & language,
        const MossVoiceGenSamplingOptions & sampling,
        uint32_t seed,
        MossVoiceGenLengthBounds bounds_override);
    std::vector<float> decode_codes(const GeneratedChunk & chunk);

    runtime::TaskSpec task_;
    std::shared_ptr<const MossVoiceGenAssets> assets_;
    engine::assets::TensorStorageType weight_storage_type_ = engine::assets::TensorStorageType::BF16;
    size_t backbone_graph_arena_bytes_ = 512ull * 1024ull * 1024ull;
    size_t backbone_weight_context_bytes_ = 8192ull * 1024ull * 1024ull;
    size_t heads_graph_arena_bytes_ = 256ull * 1024ull * 1024ull;
    size_t heads_weight_context_bytes_ = 4096ull * 1024ull * 1024ull;
    size_t codec_graph_arena_bytes_ = 2048ull * 1024ull * 1024ull;
    size_t codec_weight_context_bytes_ = 4096ull * 1024ull * 1024ull;

    // The execution context comes from RuntimeSessionBase; the runtimes below borrow it.
    std::unique_ptr<MossVoiceGenTextProcessor> text_processor_;
    std::unique_ptr<moss::AudioCodebookEmbeddings> codebooks_;
    std::unique_ptr<MossVoiceGenBackboneRuntime> backbone_;
    std::unique_ptr<MossVoiceGenHeadsRuntime> heads_;
    std::unique_ptr<moss::MossAudioTokenizerDecoder> codec_;
};

}  // namespace engine::models::moss_voicegen
