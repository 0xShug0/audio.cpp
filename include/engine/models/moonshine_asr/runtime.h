#pragma once

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/core/module.h"
#include "engine/framework/modules/activation_modules.h"
#include "engine/framework/runtime/session.h"

#include <cstddef>
#include <string>
#include <unordered_map>

namespace engine::core {
class ExecutionContext;
}

namespace engine::models::moonshine_asr {

struct MoonshineAssets;
struct MoonshineConfig;
struct MoonshineWeights;

struct MoonshineRuntimeConfig {
    size_t weight_context_bytes = 256ull * 1024ull * 1024ull;
    size_t graph_arena_bytes = 512ull * 1024ull * 1024ull;
    engine::modules::GeluApproximation encoder_gelu = engine::modules::GeluApproximation::Quick;
    engine::assets::TensorStorageType encoder_weight_storage_type = engine::assets::TensorStorageType::Native;
    engine::assets::TensorStorageType decoder_weight_storage_type = engine::assets::TensorStorageType::Native;
    engine::assets::TensorStorageType conv_weight_storage_type = engine::assets::TensorStorageType::Native;
    bool cpu_blas_scheduler = true;
};

MoonshineRuntimeConfig make_moonshine_runtime_config(
    const MoonshineConfig & config,
    engine::core::BackendType backend_type,
    const std::unordered_map<std::string, std::string> & options);

runtime::TaskResult transcribe_moonshine_asr(
    const MoonshineAssets & assets,
    const MoonshineWeights & weights,
    const engine::core::ExecutionContext & execution_context,
    const runtime::AudioBuffer & audio,
    const std::unordered_map<std::string, std::string> & options,
    const MoonshineRuntimeConfig & config);

}  // namespace engine::models::moonshine_asr
