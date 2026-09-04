#pragma once

#include "engine/framework/runtime/session.h"
#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/core/execution_context.h"
#include "engine/models/fireredtts3/assets.h"

#include <cstdint>
#include <functional>
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

// 流式块回调：每生成一块音频调用一次（chunk_index 从 0 起）。
using FireRedStreamChunkCallback = std::function<void(int64_t chunk_index, engine::runtime::AudioBuffer && audio)>;

// 增量流式会话：持有 AR 循环状态，逐块产出音频。
// 由 FireRedTTS3BaseRuntime 创建；调用方每取一块调 next_chunk()。
class FireRedTTS3StreamSession {
public:
    virtual ~FireRedTTS3StreamSession() = default;
    // 返回下一块音频；流结束返回 empty（samples 为空）。
    virtual engine::runtime::AudioBuffer next_chunk() = 0;
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

    // 增量流式：启动一个流式会话（AR 逐 patch 生成，按 chunk 边界增量解码）。
    // chunk_patches[i] 为第 i 块的 patch 数（如 {3,12,12,...} 首块 0.5s 后续 2s）。
    // 返回的会话每次 next_chunk() 产出一块音频；结束后 next_chunk() 返回空 audio。
    std::unique_ptr<FireRedTTS3StreamSession> begin_streaming(
        const FireRedTTS3BaseRequest & request,
        const std::vector<int64_t> & chunk_patches);

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
