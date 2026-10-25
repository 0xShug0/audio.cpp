#include "engine/community_models/fish_drama3/loader.h"
#include "engine/community_models/fish_drama3/session.h"
#include "engine/framework/runtime/spec_backed_model.h"

namespace engine::models::fish_drama3 {
std::shared_ptr<runtime::IVoiceModelLoader> make_fish_drama3_loader() {
    runtime::SpecBackedVoiceModelConfig<Drama3Assets> config;
    config.family = "fish_drama3";
    config.load_assets = load_drama3_assets;
    config.create_session = [](const runtime::TaskSpec & task, const runtime::SessionOptions & options,
        std::shared_ptr<const Drama3Assets> assets, std::shared_ptr<const engine::model_spec::ModelContract> contract) {
        auto validation_options = options;
        // Published standalone contracts predate these opt-in controls.
        // Their values are checked by the generator and backbone constructors.
        for (const auto * key : {"fish_drama3.disk_experts", "fish_drama3.expert_cache_size", "fish_drama3.disk_expert_read_threads", "fish_drama3.pin_experts",
                                 "fish_drama3.codec_encoder_f32_accumulation", "fish_drama3.codec_encoder_bf16_compute",
                                 "fish_drama3.codec_encoder_chunk_samples", "fish_drama3.device_sampling",
                                 "fish_drama3.codec_batch_frames", "fish_drama3.codec_async_decode",
                                 "fish_drama3.fused_moe", "fish_drama3.kv_cache_type", "fish_drama3.kv_cache_alignment",
                                 "fish_drama3.perf_mode"}) {
            if (!contract->session_option_keys.count(key)) validation_options.options.erase(key);
        }
        runtime::validate_spec_backed_session_options(validation_options, *contract, "fish_drama3", "Drama 3");
        return std::make_unique<Drama3Session>(task, options, std::move(assets));
    };
    return runtime::make_spec_backed_voice_loader(std::move(config));
}
}  // namespace engine::models::fish_drama3
