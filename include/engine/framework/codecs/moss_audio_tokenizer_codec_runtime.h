#pragma once

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/core/execution_context.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace engine::codecs {

struct MossAudioTokenizerTransformerStage {
    int64_t input_dimension = 0;
    int64_t output_dimension = 0;
    int64_t model_dimension = 0;
    int64_t num_heads = 0;
    int64_t num_layers = 0;
    int64_t feedforward_dimension = 0;
    int64_t context_window = 0;
    int64_t patch_size = 0;
};

struct MossAudioTokenizerQuantizerConfig {
    int64_t codebook_size = 1024;
    int64_t codebook_dim = 8;
    int64_t rvq_dim = 512;
    int64_t code_dim = 768;
    int64_t num_quantizers = 12;
};

struct MossAudioTokenizerConfig {
    int64_t sampling_rate = 48000;
    int64_t samples_per_frame = 3840;
    int64_t channels = 2;
    MossAudioTokenizerQuantizerConfig quantizer;
    std::vector<MossAudioTokenizerTransformerStage> encoder_stages;
    std::vector<MossAudioTokenizerTransformerStage> decoder_stages;
    int64_t encoder_final_patch = 1;
    int64_t decoder_initial_patch = 1;
    int64_t encoder_module_start = 1;
    int64_t encoder_module_stride = 2;
    int64_t decoder_module_start = 0;
    int64_t decoder_module_stride = 2;
};

MossAudioTokenizerConfig moss_audio_tokenizer_v1_config();
MossAudioTokenizerConfig moss_audio_tokenizer_v2_config();
MossAudioTokenizerConfig moss_audio_tokenizer_nano_config();

struct MossAudioTokenizerAudio {
    int64_t sampling_rate = 0;
    std::vector<std::vector<float>> channels;
};

struct MossAudioTokenizerCodes {
    int64_t frames = 0;
    std::vector<std::vector<int32_t>> codebooks;
};

struct MossTokenRows {
    std::vector<int32_t> text_tokens;
    std::vector<int32_t> audio_codes;
};

class MossTokenRowBuilder {
public:
    MossTokenRowBuilder(int64_t num_codebooks, int32_t audio_pad_token_id);

    void push_text_token(int32_t token_id);
    void push_text_tokens(const std::vector<int32_t> & token_ids);
    void push_audio_row(int32_t text_slot_token_id, const int32_t * codes, int64_t num_codebooks);
    void push_audio_row(int32_t text_slot_token_id, const std::vector<std::vector<int32_t>> & codes, int64_t frame);
    MossTokenRows finish();

private:
    int64_t num_codebooks_ = 0;
    int32_t audio_pad_token_id_ = 0;
    MossTokenRows rows_;
};

struct MossAudioTokenizerCodecRuntimeOptions {
    size_t weight_context_bytes = 256ull * 1024ull * 1024ull;
    size_t encoder_graph_arena_bytes = 2048ull * 1024ull * 1024ull;
    size_t decoder_graph_arena_bytes = 1536ull * 1024ull * 1024ull;
    bool separate_encoder_context = false;
};

class MossAudioTokenizerCodecRuntime {
public:
    MossAudioTokenizerCodecRuntime(
        std::shared_ptr<const assets::TensorSource> source,
        core::ExecutionContext & execution_context,
        int64_t num_quantizers,
        MossAudioTokenizerCodecRuntimeOptions options,
        MossAudioTokenizerConfig config = moss_audio_tokenizer_v2_config());
    ~MossAudioTokenizerCodecRuntime();

    MossAudioTokenizerCodecRuntime(const MossAudioTokenizerCodecRuntime &) = delete;
    MossAudioTokenizerCodecRuntime & operator=(const MossAudioTokenizerCodecRuntime &) = delete;

    int64_t sampling_rate() const noexcept;
    void prepare_encoder();
    void prepare_decoder();
    MossAudioTokenizerCodes encode(const MossAudioTokenizerAudio & audio);
    MossAudioTokenizerAudio decode(const MossAudioTokenizerCodes & codes);
    void release_runtime_graphs();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace engine::codecs
