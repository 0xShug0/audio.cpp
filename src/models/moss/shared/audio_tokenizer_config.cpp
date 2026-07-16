#include "engine/models/moss/shared/audio_tokenizer_config.h"

namespace engine::models::moss {

AudioTokenizerConfig moss_audio_tokenizer_v2_config() {
    AudioTokenizerConfig config;
    config.sampling_rate = 48000;
    config.samples_per_frame = 3840;
    config.quantizer = AudioTokenizerQuantizerConfig{
        1024,
        8,
        512,
        768,
        12,
    };
    config.encoder_stages = {
        {240, 384, 768, 12, 12, 3072, 400, 240},
        {768, 384, 768, 12, 12, 3072, 400, 2},
        {768, 384, 768, 12, 12, 3072, 400, 2},
        {768, 384, 768, 12, 12, 3072, 400, 2},
        {768, 640, 768, 12, 12, 3072, 250, 2},
        {1280, 768, 1280, 20, 32, 5120, 125, 2},
    };
    config.decoder_stages = {
        {768, 1280, 1280, 20, 32, 5120, 125, 2},
        {640, 768, 768, 12, 12, 3072, 250, 2},
        {384, 768, 768, 12, 12, 3072, 400, 2},
        {384, 768, 768, 12, 12, 3072, 400, 2},
        {384, 768, 768, 12, 12, 3072, 400, 2},
        {384, 240, 768, 12, 12, 3072, 400, 240},
    };
    config.encoder_module_start = 1;
    config.encoder_module_stride = 2;
    config.decoder_module_start = 0;
    config.decoder_module_stride = 2;
    return config;
}

AudioTokenizerConfig moss_audio_tokenizer_nano_config() {
    AudioTokenizerConfig config;
    config.sampling_rate = 48000;
    config.samples_per_frame = 3840;
    config.quantizer = AudioTokenizerQuantizerConfig{
        1024,
        8,
        512,
        768,
        16,
    };
    config.encoder_stages = {
        {240, 384, 256, 4, 4, 1024, 1600, 240},
        {768, 384, 256, 4, 2, 1024, 1200, 2},
        {768, 384, 256, 4, 2, 1024, 800, 2},
        {768, 192, 256, 4, 4, 1024, 500, 2},
    };
    config.decoder_stages = {
        {192, 768, 256, 4, 4, 1024, 1000, 2},
        {384, 768, 256, 4, 2, 1024, 1600, 2},
        {384, 768, 256, 4, 2, 1024, 2400, 2},
        {384, 240, 256, 4, 4, 1024, 3200, 240},
    };
    config.encoder_final_patch = 4;
    config.decoder_initial_patch = 4;
    config.encoder_module_start = 1;
    config.encoder_module_stride = 2;
    config.decoder_module_start = 1;
    config.decoder_module_stride = 2;
    return config;
}

}  // namespace engine::models::moss
