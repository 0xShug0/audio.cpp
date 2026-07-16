#pragma once

#include <cstdint>
#include <vector>

namespace engine::models::moss {

struct AudioTokenizerTransformerStage {
    int64_t input_dimension = 0;
    int64_t output_dimension = 0;
    int64_t model_dimension = 0;
    int64_t num_heads = 0;
    int64_t num_layers = 0;
    int64_t feedforward_dimension = 0;
    int64_t context_window = 0;
    int64_t patch_size = 0;
};

struct AudioTokenizerQuantizerConfig {
    int64_t codebook_size = 1024;
    int64_t codebook_dim = 8;
    int64_t rvq_dim = 512;
    int64_t code_dim = 768;
    int64_t num_quantizers = 12;
};

struct AudioTokenizerConfig {
    int64_t sampling_rate = 48000;
    int64_t samples_per_frame = 3840;
    AudioTokenizerQuantizerConfig quantizer;
    std::vector<AudioTokenizerTransformerStage> encoder_stages;
    std::vector<AudioTokenizerTransformerStage> decoder_stages;
    int64_t encoder_final_patch = 1;
    int64_t decoder_initial_patch = 1;
    int64_t encoder_module_start = 1;
    int64_t encoder_module_stride = 2;
    int64_t decoder_module_start = 0;
    int64_t decoder_module_stride = 2;
};

AudioTokenizerConfig moss_audio_tokenizer_v2_config();
AudioTokenizerConfig moss_audio_tokenizer_nano_config();

}  // namespace engine::models::moss
