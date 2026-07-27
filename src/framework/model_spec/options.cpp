#include "engine/framework/model_spec/options.h"

#include <stdexcept>

namespace engine::model_spec {

const std::unordered_map<std::string, std::vector<std::string>> & option_presets() {
    static const std::unordered_map<std::string, std::vector<std::string>> values = {
        {"best_of_n_language", {"auto", "en", "ja"}},
        {"perf_mode_flash_attention", {"off", "flash_attention"}},
        {"text_chunk_mode_full", {"word_budget", "tag_aware", "japanese", "endline"}},
        {"weight_type_codec_q8", {"native", "f32", "f16", "q8_0"}},
        {"weight_type_conv", {"native", "f32", "f16"}},
        {"weight_type_full", {"native", "f32", "f16", "bf16", "q8_0"}},
    };
    return values;
}

const std::unordered_map<std::string, std::unordered_set<std::string>> & shared_option_contracts() {
    static const std::unordered_map<std::string, std::unordered_set<std::string>> values = {
        {"audio_chunk_mode", {"enum"}},
        {"audio_chunk_seconds", {"float"}},
        {"audio_encoder_weight_type", {"enum"}},
        {"ar_weight_type", {"enum"}},
        {"batch_size", {"int"}},
        {"codec_weight_type", {"enum"}},
        {"connector_weight_type", {"enum"}},
        {"conv_weight_type", {"enum"}},
        {"decoder_weight_type", {"enum"}},
        {"do_sample", {"bool"}},
        {"duration_seconds", {"float", "float_list"}},
        {"guidance_scale", {"float"}},
        {"language", {"string"}},
        {"length_penalty", {"float"}},
        {"lyrics", {"string"}},
        {"max_new_tokens", {"int"}},
        {"max_tokens", {"int"}},
        {"mem_saver", {"bool"}},
        {"min_p", {"float"}},
        {"matmul_weight_type", {"enum"}},
        {"negative_prompt", {"string"}},
        {"num_beams", {"int"}},
        {"num_inference_steps", {"int"}},
        {"reference_language", {"string"}},
        {"reference_text", {"string"}},
        {"repetition_penalty", {"float"}},
        {"repetition_window", {"int"}},
        {"return_timestamps", {"bool"}},
        {"route", {"enum"}},
        {"sampler", {"enum"}},
        {"sampler_mode", {"enum"}},
        {"seed", {"int"}},
        {"source_audio", {"audio_path"}},
        {"target_text", {"string"}},
        {"target_voice", {"audio_path"}},
        {"temperature", {"float"}},
        {"text_decoder_weight_type", {"enum"}},
        {"text_chunk_mode", {"enum"}},
        {"text_chunk_size", {"int"}},
        {"text_temperature", {"float"}},
        {"text_top_k", {"int"}},
        {"text_top_p", {"float"}},
        {"tokenizer_weight_type", {"enum"}},
        {"top_k", {"int"}},
        {"top_p", {"float"}},
        {"voice_ref", {"audio_path"}},
        {"weight_type", {"enum"}},
    };
    return values;
}

const std::vector<std::string> & require_option_preset(std::string_view preset) {
    const auto & presets = option_presets();
    const auto it = presets.find(std::string(preset));
    if (it == presets.end()) {
        throw std::runtime_error("unknown model spec option preset: " + std::string(preset));
    }
    return it->second;
}

}  // namespace engine::model_spec
