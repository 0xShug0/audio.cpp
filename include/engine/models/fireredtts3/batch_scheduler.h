#pragma once

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/core/execution_context.h"
#include "engine/framework/runtime/kv_cache.h"
#include "engine/framework/runtime/session.h"
#include "engine/models/fireredtts3/assets.h"
#include "engine/models/fireredtts3/pipeline.h"
#include "engine/models/fireredtts3/redae.h"

#include <cstddef>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <optional>
#include <vector>

namespace engine::models::fireredtts3 {



// 真·batch 并发调度器（llama.cpp update_slots 的模拟）。
// 一个共享 FireRedArRuntime + FireRedFlowRuntime + FireRedRedAeRuntime，
// 多个 slot（请求）每轮把活跃 slot 的 AR decode 合并成一个 batch
// （decode_embeddings_batched，per-member position）。固定 max_batch graph，
// 运行期不重建；slot 中途 stop 掉队（dead 行零填充），新 slot 通过
// export -> splice -> import 加入。
class FireRedTTS3BatchScheduler {
public:
    // 一个并发请求 = 一个 slot（持有其 AR + RedAE 增量解码状态 + chunk 队列）
    struct Slot {
        enum class State { Idle, Active, Dead, Done, Failed };
        State state = State::Idle;
        FireRedTTS3BaseRequest request;
        std::vector<int64_t> chunks;          // 块 patch 数（流式）
        // AR 状态
        std::vector<float> latents_gen, backbone_cond, schedule, next_input, prefill_hidden;
        std::vector<float> spk_dit;
        int64_t prefill_steps = 0, prompt_latent_frames = 0, step = 0, generated_patches = 0;
        int64_t chunk_index = 0, chunk_target = 0;
        std::vector<float> chunk_latents;
        // 参考音色 prep 产物（wave 形成时拷进 slot，不持有 cache 引用跨轮）
        std::vector<float> prompt_latents;
        // prefill 单路径 KV state（CPU，用于拼进 batched KV）
        engine::runtime::TransformerKVState prefill_state;
        bool prefill_done = false;
        // RedAE 增量解码状态（per-slot）
        FireRedRedAeRuntime::DecodeState redae_state;
        // 流式输出
        std::deque<engine::runtime::AudioBuffer> chunk_queue;
        bool finished = false;
        std::exception_ptr error;
    };

    FireRedTTS3BatchScheduler(
        std::shared_ptr<const FireRedTTS3Assets> assets,
        engine::core::ExecutionContext & execution,
        size_t graph_arena_bytes,
        size_t helper_graph_arena_bytes,
        size_t weight_context_bytes,
        engine::assets::TensorStorageType storage_type,
        size_t reference_cache_slots,
        bool mem_saver,
        int64_t max_batch);
    ~FireRedTTS3BatchScheduler();

    FireRedTTS3BatchScheduler(const FireRedTTS3BatchScheduler &) = delete;
    FireRedTTS3BatchScheduler & operator=(const FireRedTTS3BatchScheduler &) = delete;

    // 启动一个请求（分配 slot），返回 slot id。返回 -1 表示槽池耗尽。
    int64_t launch(const FireRedTTS3BaseRequest & request, const std::vector<int64_t> & chunk_patches);
    // 取下一个音频块（驱动调度轮次直到该 slot 产出块或结束）。
    engine::runtime::AudioBuffer next_chunk(int64_t slot_id);
    // 离线整句：launch + drain + 拼接（trim prompt）。
    engine::runtime::AudioBuffer generate(const FireRedTTS3BaseRequest & request);
    void release_graphs();

    // 槽池容量（max_batch）。
    int64_t max_batch() const noexcept;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace engine::models::fireredtts3
