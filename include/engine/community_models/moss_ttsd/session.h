#pragma once

#include "engine/community_models/moss_ttsd/assets.h"
#include "engine/community_models/moss_ttsd/prompt.h"
#include "engine/framework/decoders/moss_tts_delay/backbone.h"
#include "engine/framework/decoders/moss_tts_delay/delay_decoder.h"
#include "engine/framework/decoders/moss_tts_delay/heads.h"
#include "engine/framework/core/execution_context.h"
#include "engine/framework/codecs/moss_audio_tokenizer_codec_runtime.h"
#include "engine/framework/modules/multi_codebook_embedding.h"
#include "engine/framework/runtime/session_base.h"

#include <cstddef>
#include <memory>
#include <optional>
#include <string>

namespace engine::models::moss_ttsd {

// Splits the `voice_samples` option into one entry per speaker, positional:
// entry i is [S(i+1)], and an entry with no path is a speaker that is NAMED but
// not cloned, which renders "[S<n>]: None".
//
// ⚠ THE ENTRY COUNT IS THE SPEAKER COUNT, trailing separator included. So
// "a.wav," is two speakers with the second invented, which is the documented way
// to clone one voice and not the other. An earlier version discarded a trailing
// blank as a typo, which silently turned exactly that spelling into a
// single-speaker prompt -- the one case the option exists to express.
//
// Declared here rather than kept in the session's anonymous namespace so the
// rule can be tested without a model.
std::vector<std::optional<std::string>> parse_speaker_paths(const std::string & value);

class MossTtsdSession final
    : public runtime::RuntimeSessionBase
    , public runtime::IOfflineVoiceTaskSession {
public:
    MossTtsdSession(
        runtime::TaskSpec task,
        runtime::SessionOptions options,
        std::shared_ptr<const Assets> assets);
    ~MossTtsdSession() override;

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

    // `assistant_audio` present means CONTINUATION: the model is handed the
    // reference recording and carries on from it, which is how this checkpoint
    // clones. Absent means a fresh take with no cloning.
    //
    // There is no duration budget to pass. This checkpoint's "- Tokens:" field is
    // baked to None in its own template, so the ceiling is derived from the text
    // rather than requested.
    GeneratedChunk generate_chunk(
        const PromptFields & fields,
        const std::optional<ReferenceAudio> & assistant_audio,
        const decoders::MossTtsDelaySamplingOptions & sampling,
        uint32_t seed,
        decoders::MossTtsDelayLengthBounds bounds_override,
        int64_t spoken_characters);
    std::vector<float> decode_codes(const GeneratedChunk & chunk);
    // Resamples the caller's recording to the codec rate if needed, then encodes it.
    ReferenceAudio encode_reference(const runtime::AudioBuffer & audio);
    // The speakers' references end to end, in speaker order -- what the assistant
    // turn carries in continuation mode. Concatenated as CODES rather than as
    // waveforms so each speaker is encoded once and the join lands on a frame
    // boundary.
    ReferenceAudio concatenate_references(const std::vector<std::optional<ReferenceAudio>> & speakers) const;

    runtime::TaskSpec task_;
    std::shared_ptr<const Assets> assets_;
    // Native, not BF16: a GGUF package carries its own type, and forcing BF16
    // dequantises it on load, so q4_k and q8_0 cost exactly as much VRAM as bf16
    // and the quantisation buys nothing. Measured on a 3090: q4_k TTS peaks at
    // 21.2 GiB forced to BF16 and 10.6 GiB left native. Safetensors are bf16
    // upstream, so native is the same thing there.
    engine::assets::TensorStorageType weight_storage_type_ = engine::assets::TensorStorageType::Native;
    size_t backbone_graph_arena_bytes_ = 512ull * 1024ull * 1024ull;
#if defined(INTPTR_MAX) && (INTPTR_MAX == INT32_MAX)
    size_t backbone_weight_context_bytes_ = 1024ull * 1024ull * 1024ull;
    size_t heads_graph_arena_bytes_ = 256ull * 1024ull * 1024ull;
    size_t heads_weight_context_bytes_ = 1024ull * 1024ull * 1024ull;
    size_t codec_graph_arena_bytes_ = 512ull * 1024ull * 1024ull;
    size_t codec_weight_context_bytes_ = 1024ull * 1024ull * 1024ull;
#else
    size_t backbone_weight_context_bytes_ = 8192ull * 1024ull * 1024ull;
    size_t heads_graph_arena_bytes_ = 256ull * 1024ull * 1024ull;
    size_t heads_weight_context_bytes_ = 4096ull * 1024ull * 1024ull;
    // Charged twice -- the codec runtime takes one arena size for the encoder and
    // one for the decoder -- and this model, unlike moss_voicegen, holds both
    // halves resident because cloning encodes a reference. The graphs are small
    // (a few hundred codec frames at 12.5 a second), so moss_voicegen's 2 GiB was
    // 4 GiB of VRAM here for no benefit, and on a 24 GiB card it was the
    // difference between cloning running and failing to allocate the encoder.
    size_t codec_graph_arena_bytes_ = 512ull * 1024ull * 1024ull;
    size_t codec_weight_context_bytes_ = 4096ull * 1024ull * 1024ull;
#endif

    // The execution context comes from RuntimeSessionBase; the runtimes below borrow it.
    std::shared_ptr<tokenizers::LlamaBpeTokenizer> tokenizer_;
    std::unique_ptr<engine::modules::MultiCodebookEmbedding> codebooks_;
    std::unique_ptr<decoders::MossTtsDelayBackboneRuntime> backbone_;
    std::unique_ptr<decoders::MossTtsDelayHeadsRuntime> heads_;
    std::unique_ptr<engine::codecs::MossAudioTokenizerCodecRuntime> codec_;
};

}  // namespace engine::models::moss_ttsd
