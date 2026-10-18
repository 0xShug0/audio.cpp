#pragma once

#include "engine/framework/runtime/session.h"
#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/core/execution_context.h"
#include "engine/models/fireredtts3/assets.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace engine::models::fireredtts3 {

struct FireRedTTS3BaseRequest {
    std::vector<int32_t> token_ids;
    engine::runtime::AudioBuffer prompt_audio;
    std::string language;
    std::string reference_text;
    uint32_t seed = 1234;
    int64_t num_inference_steps = 10;
    float guidance_scale = 2.0F;
    float stop_threshold = 0.5F;
};

enum class FireRedTTS3InstructTask {
    Clone,
    VoiceDesign,
    SemanticEdit,
    AcousticEdit,
};

struct FireRedTTS3InstructRequest {
    FireRedTTS3InstructTask task = FireRedTTS3InstructTask::Clone;
    std::vector<int32_t> token_ids;
    std::vector<uint8_t> latent_in_mask;
    std::vector<uint8_t> latent_out_mask;
    std::optional<engine::runtime::AudioBuffer> input_audio;
    std::optional<engine::runtime::AudioBuffer> prompt_audio;
    std::string instruction;
    std::string text;
    bool infer_text = false;
    bool text_do_sample = false;
    uint32_t seed = 1234;
    int64_t num_inference_steps = 10;
    float guidance_scale = 2.0F;
    float stop_threshold = 0.5F;
};

struct FireRedTTS3InstructResult {
    engine::runtime::AudioBuffer audio;
    std::string generated_text;
};

class FireRedTTS3BaseRuntime {
public:
    FireRedTTS3BaseRuntime(
        std::shared_ptr<const FireRedTTS3Assets> assets,
        engine::core::ExecutionContext & execution,
        size_t graph_arena_bytes,
        size_t helper_graph_arena_bytes,
        size_t weight_context_bytes,
        engine::assets::TensorStorageType storage_type,
        size_t reference_cache_slots,
        bool mem_saver);
    ~FireRedTTS3BaseRuntime();

    FireRedTTS3BaseRuntime(const FireRedTTS3BaseRuntime &) = delete;
    FireRedTTS3BaseRuntime & operator=(const FireRedTTS3BaseRuntime &) = delete;

    engine::runtime::AudioBuffer generate(const FireRedTTS3BaseRequest & request);
    void release_graphs();

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

class FireRedTTS3InstructRuntime {
public:
    FireRedTTS3InstructRuntime(
        std::shared_ptr<const FireRedTTS3Assets> assets,
        engine::core::ExecutionContext & execution,
        size_t graph_arena_bytes,
        size_t helper_graph_arena_bytes,
        size_t weight_context_bytes,
        engine::assets::TensorStorageType storage_type,
        bool mem_saver);
    ~FireRedTTS3InstructRuntime();

    FireRedTTS3InstructRuntime(const FireRedTTS3InstructRuntime &) = delete;
    FireRedTTS3InstructRuntime & operator=(const FireRedTTS3InstructRuntime &) = delete;

    FireRedTTS3InstructResult generate(const FireRedTTS3InstructRequest & request);
    void release_graphs();

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace engine::models::fireredtts3
