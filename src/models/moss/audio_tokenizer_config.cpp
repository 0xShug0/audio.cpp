#include "engine/models/moss/audio_tokenizer_config.h"

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
    return config;
}

}  // namespace engine::models::moss
