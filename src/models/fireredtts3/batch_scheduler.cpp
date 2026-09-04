#include "engine/models/fireredtts3/batch_scheduler.h"

#include "engine/framework/audio/conversion.h"
#include "engine/framework/audio/kaldi_fbank.h"
#include "engine/framework/audio/resampling.h"
#include "engine/framework/modules/speech_encoders/campplus_encoder.h"
#include "engine/framework/runtime/cache_slots.h"
#include "engine/framework/sampling/torch_random.h"
#include "engine/models/fireredtts3/ar.h"
#include "engine/models/fireredtts3/flow.h"
#include "engine/models/fireredtts3/redae.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <thread>

namespace engine::models::fireredtts3 {
namespace {

namespace core = engine::core;
namespace modules = engine::modules;
using Clock = std::chrono::steady_clock;

constexpr int64_t kMaxArSteps = 400;
// prefill graph 固定 steps（grow-only 复用）：所有请求的 prefill 都 padding 到该长度，
// 避免因 steps 变化重建 prefill graph 破坏 CUDA pool 逆序 free 约束。
constexpr int64_t kMaxPrefillSteps = 1024;
// batched decode KV cache 固定上限（首次 build 后永不重建）：
// 最大可能 prefill（参考 + 长文本 ~300） + kMaxArSteps(400) + 尾部余量。
constexpr int64_t kMaxDecodeCacheSteps = 700;

// 与 pipeline.cpp 一致的参考音色 key/entry（避免引用失效跨轮，slot 拷贝产物）。
struct ReferenceVoiceCacheKey {
    int sample_rate = 0;
    int channels = 0;
    uint64_t sample_count = 0;
    uint64_t sample_hash = 0;
};

struct ReferenceVoiceCacheKeyEqual {
    bool operator()(const ReferenceVoiceCacheKey & a, const ReferenceVoiceCacheKey & b) const noexcept {
        return a.sample_rate == b.sample_rate && a.channels == b.channels &&
            a.sample_count == b.sample_count && a.sample_hash == b.sample_hash;
    }
};

struct ReferenceVoiceCacheEntry {
    std::vector<float> prompt_audio_24k;
    std::vector<float> prompt_latents;
    std::vector<float> speaker_embedding;
};

uint64_t hash_audio_samples(const runtime::AudioBuffer & audio) {
    uint64_t key = 1469598103934665603ull;
    for (const float sample : audio.samples) {
        uint32_t bits = 0;
        std::memcpy(&bits, &sample, sizeof(bits));
        key ^= bits;
        key *= 1099511628211ull;
    }
    return key;
}

std::vector<float> last_rows(
    const std::vector<float> & values,
    int64_t rows,
    int64_t width,
    int64_t keep) {
    if (keep <= 0 || rows < keep || static_cast<int64_t>(values.size()) != rows * width) {
        throw std::runtime_error("FireRedTTS3 hidden row slice is out of range");
    }
    return std::vector<float>(
        values.begin() + static_cast<std::ptrdiff_t>((rows - keep) * width),
        values.end());
}

std::vector<float> campplus_fbank(const std::vector<float> & prompt_24k) {
    auto audio_16k = audio::resample_mono_torchaudio_sinc_hann(prompt_24k, 24000, 16000);
    audio::KaldiFbankOptions options;
    options.sample_rate = 16000;
    options.num_mels = 80;
    options.window_type = audio::KaldiFbankWindowType::Povey;
    options.lfr_m = 1;
    options.lfr_n = 1;
    options.apply_cmvn = false;
    options.upscale_samples = false;
    auto features = audio::extract_kaldi_fbank(audio_16k, options);
    if (features.frames <= 0 || features.feature_dim != 80) {
        throw std::runtime_error("FireRedTTS3 CAM++ fbank extraction failed");
    }
    for (int64_t m = 0; m < features.feature_dim; ++m) {
        double sum = 0.0;
        for (int64_t t = 0; t < features.frames; ++t) {
            sum += features.values[static_cast<size_t>(t * features.feature_dim + m)];
        }
        const float mean = static_cast<float>(sum / static_cast<double>(features.frames));
        for (int64_t t = 0; t < features.frames; ++t) {
            features.values[static_cast<size_t>(t * features.feature_dim + m)] -= mean;
        }
    }
    return std::move(features.values);
}

}  // namespace

class FireRedTTS3BatchScheduler::Impl {
public:
    Impl(
        std::shared_ptr<const FireRedTTS3Assets> assets,
        engine::core::ExecutionContext & execution,
        size_t graph_arena_bytes,
        size_t helper_graph_arena_bytes,
        size_t weight_context_bytes,
        engine::assets::TensorStorageType storage_type,
        size_t reference_cache_slots,
        bool mem_saver,
        int64_t max_batch)
        : assets_(std::move(assets)),
          execution_(execution),
          mem_saver_(mem_saver),
          max_batch_(std::max<int64_t>(1, max_batch)),
          sampling_policy_(sampling::resolve_torch_cuda_sampling_policy(
              execution.backend_type(),
              execution.config().device,
              "fireredtts3",
              "FireRedTTS3Batch",
              sampling::TorchCudaSamplingPolicyFailureMode::FallbackToDefault)),
          ar_(std::make_unique<FireRedArRuntime>(
              assets_, execution_, graph_arena_bytes, helper_graph_arena_bytes,
              weight_context_bytes, storage_type, false)),
          redae_(std::make_unique<FireRedRedAeRuntime>(
              assets_, execution_, graph_arena_bytes, weight_context_bytes, storage_type)),
          // flow 图按需重建（batch 变则 rebuild）。曾试过 fixed-capacity + 零填充复用，
          // 但 padding 破坏 DiT 输出（单路 MD5 就变），故保持每轮实际 batch 建图。
          flow_(std::make_unique<FireRedFlowRuntime>(
              assets_, execution_, graph_arena_bytes, weight_context_bytes, storage_type, false)),
          reference_voice_cache_(
              runtime::CacheSlots<ReferenceVoiceCacheKey, ReferenceVoiceCacheEntry, ReferenceVoiceCacheKeyEqual>(
                  reference_cache_slots)) {
        if (assets_ == nullptr) {
            throw std::runtime_error("FireRedTTS3BatchScheduler requires assets");
        }
        slots_.resize(static_cast<size_t>(max_batch_));
        for (size_t i = 0; i < slots_.size(); ++i) {
            free_slots_.push_back(static_cast<int64_t>(i));
        }
        // 专用调度线程：唯一执行 GPU decode graph 的线程。
        // ggml CUDA memory pool 要求分配/释放严格逆序（GGML_ASSERT(ptr == pool_addr + pool_used)），
        // 多请求线程并发 tick 会破坏该不变量导致崩溃。单调度线程 = llama.cpp update_slots 模式。
        scheduler_thread_ = std::thread(&Impl::scheduler_loop, this);
    }

    ~Impl() {
        {
            std::lock_guard<std::mutex> lock(mu_);
            stop_ = true;
        }
        cv_wake_.notify_all();
        if (scheduler_thread_.joinable()) {
            scheduler_thread_.join();
        }
    }

    SlotHandle launch(const FireRedTTS3BaseRequest & request, const std::vector<int64_t> & chunk_patches) {
        std::lock_guard<std::mutex> lock(mu_);
        if (free_slots_.empty()) {
            return SlotHandle{};
        }
        const int64_t slot_id = free_slots_.front();
        free_slots_.pop_front();
        auto & slot = slots_[static_cast<size_t>(slot_id)];
        // epoch 先取后回写：slot = Slot{} 会把 epoch 清零，故先 +1 再重置、回写单调代次。
        const uint64_t epoch = slot.epoch + 1;
        slot = Slot{};
        slot.epoch = epoch;
        slot.state = Slot::State::Active;
        slot.request = request;
        slot.chunks = chunk_patches.empty() ? std::vector<int64_t>{400} : chunk_patches;
        slot.chunk_target = slot.chunks[0];
        pending_.push_back(slot_id);
        fprintf(stderr, "[LAUNCH] slot=%ld epoch=%llu tokens=%zu\n",
                slot_id, (unsigned long long)epoch, request.token_ids.size());
        fflush(stderr);
        cv_wake_.notify_all();
        return SlotHandle{slot_id, epoch};
    }

    engine::runtime::AudioBuffer next_chunk(const SlotHandle & handle) {
        std::unique_lock<std::mutex> lock(mu_);
        // 首行所有权校验（在读 slot.error/queue/finished 之前）：
        // 句柄无效 / 越界 / 代次不匹配（stale，slot 已被 release 或复用）→ 立即返回空 = 流结束，
        // 绝不读新请求的 slot 状态（防串音），也避免旧 owner 在已复用 slot 上继续 move/析构。
        if (!handle.valid() || handle.id >= max_batch_) {
            return {};
        }
        auto & slot = slots_[static_cast<size_t>(handle.id)];
        if (slot.epoch != handle.epoch) {
            return {};
        }
        if (slot.error) {
            std::rethrow_exception(slot.error);
        }
        if (!slot.chunk_queue.empty()) {
            auto audio = std::move(slot.chunk_queue.front());
            slot.chunk_queue.pop_front();
            return audio;
        }
        if (slot.finished) {
            return {};
        }
        // 纯等待模式：调度线程负责 tick（batch decode），本线程只等自己的 chunk。
        cv_.wait(lock, [&] {
            return slot.finished || !slot.chunk_queue.empty() || slot.state == Slot::State::Failed || slot.error != nullptr;
        });
        if (slot.error) {
            std::rethrow_exception(slot.error);
        }
        if (!slot.chunk_queue.empty()) {
            auto audio = std::move(slot.chunk_queue.front());
            slot.chunk_queue.pop_front();
            return audio;
        }
        return {};
    }

    engine::runtime::AudioBuffer generate(const FireRedTTS3BaseRequest & request) {
        const SlotHandle handle = launch(request, {});
        if (!handle.valid()) {
            throw std::runtime_error("FireRedTTS3BatchScheduler slot pool exhausted");
        }
        engine::runtime::AudioBuffer merged;
        merged.sample_rate = static_cast<int>(assets_->redae.sample_rate);
        merged.channels = 1;
        try {
            while (true) {
                auto chunk = next_chunk(handle);
                if (chunk.samples.empty()) {
                    break;
                }
                runtime::append_audio_buffer(merged, chunk);
            }
        } catch (...) {
            // 异常：快速终结仍 Active 的 slot，排空到 finished，归还，再抛。
            abort(handle);
            try {
                while (!next_chunk(handle).samples.empty()) {
                }
            } catch (...) {
            }
            release_slot(handle);
            throw;
        }
        release_slot(handle);
        return merged;
    }

    int64_t max_batch() const noexcept {
        return max_batch_;
    }

    void release_graphs() {
        std::lock_guard<std::mutex> lock(mu_);
        if (redae_) {
            redae_->release_graphs();
        }
        if (ar_) {
            ar_->release_graphs();
            ar_->release_backbone_graphs();
        }
        if (flow_) {
            flow_->release_graph();
        }
        campplus_.release_runtime_graph();
    }

    // owner 在排空结束（next_chunk 见空 / reset / generate drain 完）时显式归还 slot。
    // 只回收 terminal（Idle 且 finished）的 slot；过早调用 no-op。幂等。
    // 归还即"slot 回到干净空闲态"，下一次 launch 复用它时行是零（每轮 import 全量清零）。
    void release_slot(const SlotHandle & handle) {
        std::lock_guard<std::mutex> lock(mu_);
        if (!handle.valid() || handle.id >= max_batch_) {
            return;
        }
        auto & slot = slots_[static_cast<size_t>(handle.id)];
        if (slot.epoch != handle.epoch) {
            return;  // 已 release 或复用 → 幂等
        }
        if (slot.state != Slot::State::Idle || !slot.finished) {
            return;  // 过早：调用方应先排空（Dead 在 finish_slot 转 Idle），不回收活 slot
        }
        slot.epoch++;  // 残留句柄立即失效
        if (std::find(free_slots_.begin(), free_slots_.end(), handle.id) == free_slots_.end()) {
            free_slots_.push_back(handle.id);
        }
        fprintf(stderr, "[RELEASED] slot=%ld epoch=%llu\n",
                handle.id, (unsigned long long)slot.epoch);
        fflush(stderr);
        cv_wake_.notify_all();  // 空闲槽可能让新一轮开跑
        cv_.notify_all();
    }

    // 快速终止仍 Active 的 slot（reset/异常清理用）：Active → Dead，调度线程收尾。
    void abort(const SlotHandle & handle) {
        std::lock_guard<std::mutex> lock(mu_);
        if (!handle.valid() || handle.id >= max_batch_) {
            return;
        }
        auto & slot = slots_[static_cast<size_t>(handle.id)];
        if (slot.epoch != handle.epoch) {
            return;
        }
        if (slot.state != Slot::State::Active) {
            return;  // 已 Dead/Idle/Failed，无需干预
        }
        if (!slot.prefill_done) {
            // 尚未被领进 round：直接从 pending 摘除，避免 Dead slot 被 prefill。
            pending_.erase(
                std::remove(pending_.begin(), pending_.end(), handle.id),
                pending_.end());
        }
        slot.state = Slot::State::Dead;
        cv_wake_.notify_all();
    }

private:
    // ---- 参考音色（锁内串行，产物拷进 slot）----
    const ReferenceVoiceCacheEntry & prepare_reference_voice_locked(const runtime::AudioBuffer & audio) {
        ReferenceVoiceCacheKey key;
        key.sample_rate = audio.sample_rate;
        key.channels = audio.channels;
        key.sample_count = static_cast<uint64_t>(audio.samples.size());
        key.sample_hash = hash_audio_samples(audio);
        const ReferenceVoiceCacheEntry * cached = reference_voice_cache_.find(key);
        if (cached != nullptr) {
            return *cached;
        }
        ReferenceVoiceCacheEntry entry;
        entry.prompt_audio_24k = prepare_firered_prompt_audio_24k(audio, assets_->redae, assets_->base.patch_size);
        entry.prompt_latents = redae_->encode(entry.prompt_audio_24k);
        ensure_campplus_locked();
        auto speaker_features = campplus_fbank(entry.prompt_audio_24k);
        auto speaker = campplus_.embed_from_features(
            speaker_features, static_cast<int64_t>(speaker_features.size()) / 80, 80);
        entry.speaker_embedding = std::move(speaker.embedding);
        reference_voice_cache_.put(key, std::move(entry));
        const ReferenceVoiceCacheEntry * stored = reference_voice_cache_.find(key);
        if (stored == nullptr) {
            throw std::runtime_error("FireRedTTS3 reference voice cache insert failed");
        }
        return *stored;
    }

    void ensure_campplus_locked() {
        if (campplus_.weights() != nullptr) {
            return;
        }
        modules::CampplusEncoderConfig cfg;
        cfg.feat_dim = 80;
        cfg.embedding_size = assets_->base.speaker_dim;
        cfg.tensor_prefix = "campplus";
        cfg.weight_storage_type = storage_type_;
        campplus_ = modules::CampplusEncoderComponent::load_from_tensor_source(
            assets_->campplus_weights, execution_.config(), std::move(cfg));
    }

    // 单个 slot 的 prefill（锁内串行）：参考 prep + 单路径 padded prefill，KV 存 CPU。
    void prefill_slot_locked(int64_t slot_id) {
        auto & slot = slots_[static_cast<size_t>(slot_id)];
        const auto & ref = prepare_reference_voice_locked(slot.request.prompt_audio);
        slot.prompt_latents = ref.prompt_latents;
        slot.prompt_latent_frames = static_cast<int64_t>(ref.prompt_latents.size()) / assets_->base.redae_dim;
        auto spk_llm = ar_->speaker_llm(ref.speaker_embedding);
        slot.spk_dit = ar_->speaker_dit(ref.speaker_embedding);
        auto text_embeds = ar_->token_embedding(slot.request.token_ids);
        auto patch_prompt = ar_->patch_encode(ref.prompt_latents);
        std::vector<float> input_embeddings;
        input_embeddings.reserve(spk_llm.size() + text_embeds.size() + patch_prompt.size());
        input_embeddings.insert(input_embeddings.end(), spk_llm.begin(), spk_llm.end());
        input_embeddings.insert(input_embeddings.end(), text_embeds.begin(), text_embeds.end());
        input_embeddings.insert(input_embeddings.end(), patch_prompt.begin(), patch_prompt.end());
        slot.prefill_steps = 1 + static_cast<int64_t>(slot.request.token_ids.size())
            + slot.prompt_latent_frames / assets_->base.patch_size;
        if (slot.prefill_steps > kMaxPrefillSteps) {
            throw std::runtime_error("FireRedTTS3 prefill steps exceed scheduler max (increase kMaxPrefillSteps)");
        }
        // padded prefill：input padding 到 kMaxPrefillSteps（零 embedding），
        // 固定 prefill graph（不重建），返回 state/hidden 截断到 prefill_steps。
        std::vector<float> padded(
            static_cast<size_t>(kMaxPrefillSteps) * static_cast<size_t>(assets_->base.hidden_size), 0.0F);
        std::copy(input_embeddings.begin(), input_embeddings.end(), padded.begin());
        auto prefill = ar_->prefill_embeddings_padded(
            padded, kMaxPrefillSteps, slot.prefill_steps);
        slot.prefill_hidden = std::move(prefill.hidden);
        slot.prefill_state = std::move(prefill.state);
        slot.prefill_done = true;

        slot.latents_gen.assign(
            static_cast<size_t>(assets_->base.history_patches * assets_->base.patch_size * assets_->base.redae_dim), 0.0F);
        slot.latents_gen.insert(slot.latents_gen.end(), slot.prompt_latents.begin(), slot.prompt_latents.end());
        slot.backbone_cond.assign(static_cast<size_t>(assets_->base.history_patches * assets_->base.hidden_size), 0.0F);
        slot.schedule = firered_cosine_time_schedule(slot.request.num_inference_steps);
        redae_->decode_reset(slot.redae_state);
    }

    // 是否存在某 slot 已 prefill（即已占用 decode graph 的一行）。非空 → 当前有 round 在跑。
    bool any_round_participant_locked() const {
        for (int64_t i = 0; i < max_batch_; ++i) {
            const auto & slot = slots_[static_cast<size_t>(i)];
            if (slot.state == Slot::State::Active && slot.prefill_done) {
                return true;
            }
        }
        return false;
    }

    // 当前 round 的活跃 decode 行（step>=1，真正在 decode）。
    std::vector<int64_t> active_decode_rows_locked() const {
        std::vector<int64_t> out;
        for (int64_t i = 0; i < max_batch_; ++i) {
            const auto & slot = slots_[static_cast<size_t>(i)];
            if (slot.state == Slot::State::Active && slot.prefill_done && slot.step >= 1) {
                out.push_back(i);
            }
        }
        return out;
    }

    // ---- 调度主循环（锁内）----
    // round 制：
    //   begin_round: prefill 所有 pending 请求 → 一次全量干净 import（只写本轮行，
    //                其余行零）→ 对本轮每个 slot 走 step-0 AR（AR0，不 consume decode）。
    //   之后每 tick: 一次 batched decode 同时推进本轮所有 step>=1 活跃行。
    //   slot 结束（stop/上限）→ Dead → finish_slot 收尾（flush RedAE）→ Idle+finished，
    //   等 owner 排空后 release_slot 归还空闲。
    //   round 全部 drain 后，新 pending 才开新一轮（全量 import 天然清零旧内容）。
    void tick_locked() {
        tick_count_++;
        // 1. 开新一轮：有待 prefill 的请求，且当前无任何已 prefill 的 round 参与者在跑。
        if (!pending_.empty() && !any_round_participant_locked()) {
            begin_round_locked();
        }
        // 2. 一次 batched decode 推进当前 round 所有活跃行。
        const auto active = active_decode_rows_locked();
        if (!active.empty()) {
            decode_round_step_locked(active);
        }
        // 3. 收尾 Dead/Failed slot（flush RedAE 尾部、标记 finished、唤醒 owner）。
        for (int64_t i = 0; i < max_batch_; ++i) {
            auto & slot = slots_[static_cast<size_t>(i)];
            if (slot.state == Slot::State::Dead || slot.state == Slot::State::Failed) {
                finish_slot_locked(i);
            }
        }
    }

    // 开一轮：prefill 全部 pending，全量干净 import，AR0。
    void begin_round_locked() {
        std::vector<int64_t> round_slots;
        while (!pending_.empty()) {
            const int64_t slot_id = pending_.front();
            pending_.pop_front();
            round_slots.push_back(slot_id);
        }
        for (int64_t slot_id : round_slots) {
            try {
                prefill_slot_locked(slot_id);
            } catch (...) {
                auto & slot = slots_[static_cast<size_t>(slot_id)];
                slot.error = std::current_exception();
                slot.state = Slot::State::Failed;
                slot.finished = true;
                // 失败 slot 留待 owner next_chunk 抛错后 release_slot 回收。
            }
        }
        // 成功的 round 成员（Active && prefill_done）
        std::vector<int64_t> members;
        for (int64_t i = 0; i < max_batch_; ++i) {
            const auto & slot = slots_[static_cast<size_t>(i)];
            if (slot.state == Slot::State::Active && slot.prefill_done) {
                members.push_back(i);
            }
        }
        if (members.empty()) {
            return;  // 全失败；本 tick 末尾 finish_slot 收尾
        }
        if (diag_enabled()) {
            std::string m;
            for (size_t k = 0; k < members.size(); ++k) {
                m += std::to_string(members[k]);
                if (k + 1 < members.size()) {
                    m += ",";
                }
            }
            fprintf(stderr, "[ROUND] t=%lld members=[%s] prefill_ends=[",
                    (long long)tick_count_, m.c_str());
            for (int64_t i : members) {
                fprintf(stderr, "%lld,", (long long)slots_[static_cast<size_t>(i)].prefill_steps);
            }
            fprintf(stderr, "]\n");
            fflush(stderr);
        }
        // 全量干净 import：只写本轮成员行的 prefill KV，其余行全零。
        import_round_state_locked(members);
        // AR0：对本轮每个成员走 step-0 AR（用 CPU prefill_hidden，不 consume decode）。
        for (int64_t slot_id : members) {
            advance_slot_from_prefill_locked(slot_id);
        }
    }

    // 组装全量 batched state 并 import（写满所有行：成员行=prefill，非成员行=零）。
    void import_round_state_locked(const std::vector<int64_t> & members) {
        int64_t max_steps = 0;
        for (int64_t i : members) {
            max_steps = std::max(max_steps, slots_[static_cast<size_t>(i)].prefill_steps);
        }
        const size_t layer_count = slot_prefill_layer_count();
        if (layer_count == 0) {
            throw std::runtime_error("FireRedTTS3 round import requires prefill layer state");
        }
        const size_t row_elems = static_cast<size_t>(assets_->base.kv_heads * assets_->base.head_dim);
        runtime::TransformerBatchedKVState state;
        state.batch_size = max_batch_;
        state.current_end = max_steps;
        state.current_ends.assign(static_cast<size_t>(max_batch_), 0);
        state.layers.resize(layer_count);
        for (size_t layer = 0; layer < layer_count; ++layer) {
            auto & out_layer = state.layers[layer];
            out_layer.valid_steps = max_steps;
            out_layer.key.assign(
                static_cast<size_t>(max_batch_) * static_cast<size_t>(max_steps) * row_elems, 0.0F);
            out_layer.value.assign(
                static_cast<size_t>(max_batch_) * static_cast<size_t>(max_steps) * row_elems, 0.0F);
        }
        for (int64_t b : members) {
            auto & slot = slots_[static_cast<size_t>(b)];
            state.current_ends[static_cast<size_t>(b)] = slot.prefill_steps;
            for (size_t layer = 0; layer < layer_count; ++layer) {
                const size_t copy_elems = static_cast<size_t>(slot.prefill_steps) * row_elems;
                float * dst_key = state.layers[layer].key.data()
                    + static_cast<size_t>(b) * static_cast<size_t>(max_steps) * row_elems;
                float * dst_value = state.layers[layer].value.data()
                    + static_cast<size_t>(b) * static_cast<size_t>(max_steps) * row_elems;
                const auto & src = slot.prefill_state.layers[layer];
                if (src.key.size() != copy_elems || src.value.size() != copy_elems) {
                    throw std::runtime_error("FireRedTTS3 prefill state size mismatch during round import");
                }
                std::copy(src.key.begin(), src.key.end(), dst_key);
                std::copy(src.value.begin(), src.value.end(), dst_value);
            }
        }
        // start_decode_embeddings_batched：graph 未建则建（固定 cache），已建则仅 import
        // （重写整张 cache tensor —— 本轮成员行写入，非成员行清零）。
        ar_->start_decode_embeddings_batched(state, kMaxDecodeCacheSteps);
    }

    // 一次 batched decode 推进本轮所有活跃行，并把 flow（DiT denoise）也跨 slot batch。
    // 两阶段：
    //   A. per-slot AR 后处理（stop 判定 / backbone_cond / dit_cond3）——host 计算，逐 slot。
    //   B. 对存活 slot 同步做 flow denoise：所有 slot 的同一 denoise 步拼一次大 batch
    //      flow.run（行 = 各 slot 的 cfg_batch 行首尾相接），输出按 slot 拆回各自 CFG 合并。
    //      DiT 每行独立 attend，故与各 slot 单独跑逐位一致。异构 steps 则逐个单跑（回退）。
    //   C. per-slot：patch_encode -> next_input，累加 latents/chunk，chunk 边界 redae。
    struct FlowSurvivor {
        int64_t id;
        int64_t cfg_batch;              // guidance_scale>0 ? 2 : 1
        std::vector<float> dit_cond3;   // [history_patches+1, dit_hidden]
    };
    void decode_round_step_locked(const std::vector<int64_t> & active) {
        const int64_t hidden = assets_->base.hidden_size;
        std::vector<float> embeddings(static_cast<size_t>(max_batch_ * hidden), 0.0F);
        std::vector<uint8_t> active_mask(static_cast<size_t>(max_batch_), 0);
        for (int64_t i : active) {
            auto & slot = slots_[static_cast<size_t>(i)];
            std::copy(slot.next_input.begin(), slot.next_input.end(),
                      embeddings.begin() + static_cast<std::ptrdiff_t>(i * hidden));
            active_mask[static_cast<size_t>(i)] = 1;
        }
        auto t_dec0 = std::chrono::steady_clock::now();
        auto out = ar_->decode_embeddings_batched(embeddings, max_batch_, active_mask);
        prof_.ar_decode_ms += std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - t_dec0).count();
        if (diag_enabled()) {
            for (int64_t i : active) {
                const size_t base = static_cast<size_t>(i) * static_cast<size_t>(hidden);
                bool nan = false;
                for (int64_t e = 0; e < hidden; ++e) {
                    if (std::isnan(out.hidden[base + static_cast<size_t>(e)])) {
                        nan = true;
                        break;
                    }
                }
                if (nan) {
                    fprintf(stderr, "[NAN] t=%lld row=%lld step=%lld\n",
                            (long long)tick_count_, (long long)i,
                            (long long)slots_[static_cast<size_t>(i)].step);
                    fflush(stderr);
                }
            }
        }
        // ---- 阶段 A：per-slot AR 后处理 + stop 判定，收集存活 slot ----
        std::vector<FlowSurvivor> survivors;
        survivors.reserve(active.size());
        for (int64_t i : active) {
            auto & slot = slots_[static_cast<size_t>(i)];
            if (slot.state != Slot::State::Active || !slot.prefill_done || slot.step < 1) {
                continue;
            }
            std::vector<float> row(
                out.hidden.begin() + static_cast<std::ptrdiff_t>(i * hidden),
                out.hidden.begin() + static_cast<std::ptrdiff_t>((i + 1) * hidden));
            if (slot.step >= kMaxArSteps) {
                slot.state = Slot::State::Dead;
                continue;
            }
            const float stop = ar_->stop(row);
            if (stop >= slot.request.stop_threshold && slot.step >= 6) {
                slot.state = Slot::State::Dead;
                continue;
            }
            slot.backbone_cond.insert(slot.backbone_cond.end(), row.begin(), row.end());
            const int64_t cond_rows = static_cast<int64_t>(slot.backbone_cond.size()) / hidden;
            auto cond3 = last_rows(slot.backbone_cond, cond_rows, hidden, assets_->base.history_patches + 1);
            FlowSurvivor m;
            m.id = i;
            m.cfg_batch = slot.request.guidance_scale > 0.0F ? 2 : 1;
            m.dit_cond3 = ar_->dit_head(cond3, assets_->base.history_patches + 1);
            survivors.push_back(std::move(m));
        }
        if (survivors.empty()) {
            return;
        }
        // ---- 阶段 B/C：flow denoise（batch 或逐 slot）+ per-slot 收尾 ----
        const size_t s0_steps = slots_[static_cast<size_t>(survivors[0].id)].schedule.size();
        const bool same_steps = std::all_of(survivors.begin(), survivors.end(), [&](const FlowSurvivor & m) {
            return slots_[static_cast<size_t>(m.id)].schedule.size() == s0_steps;
        });
        if (same_steps && survivors.size() > 1) {
            auto latents = batched_flow_latents_locked(survivors);
            for (size_t s = 0; s < survivors.size(); ++s) {
                finish_slot_patch_locked(survivors[s].id, std::move(latents[s]));
            }
        } else {
            for (const FlowSurvivor & m : survivors) {
                auto & slot = slots_[static_cast<size_t>(m.id)];
                auto latent = flow_one_patch_locked(slot, m.dit_cond3, static_cast<uint64_t>(slot.step));
                finish_slot_patch_locked(m.id, std::move(latent));
            }
        }
    }

    // 多 slot 同步 flow denoise，返回每 slot 的 next_latent（[patch, redae_dim]）。
    // pred 布局：flow 输出每行 = [patch, redae_dim]（graph 已 slice 掉 history token），
    // 行 stride = patch*redae_dim；行按 slot 的 cfg_batch 首尾相接。
    std::vector<std::vector<float>> batched_flow_latents_locked(const std::vector<FlowSurvivor> & survivors) {
        const int64_t history_tokens = assets_->base.history_patches * assets_->base.patch_size;
        const int64_t tokens = history_tokens + assets_->base.patch_size;
        const int64_t in_channels = assets_->base.redae_dim + assets_->base.dit_hidden_size + assets_->base.speaker_dim;
        const int64_t patch = assets_->base.patch_size;
        const int64_t redae_dim = assets_->base.redae_dim;
        const int64_t n = static_cast<int64_t>(survivors.size());
        std::vector<std::vector<float>> history(n);
        std::vector<std::vector<float>> current(n);
        std::vector<int64_t> base_row(n);
        int64_t total_batch = 0;
        for (int64_t s = 0; s < n; ++s) {
            auto & slot = slots_[static_cast<size_t>(survivors[s].id)];
            history[s] = last_rows(
                slot.latents_gen,
                static_cast<int64_t>(slot.latents_gen.size()) / redae_dim,
                redae_dim,
                history_tokens);
            const uint64_t noise_offset =
                sampling::torch_cuda_tensor_iterator_offset_blocks(
                    static_cast<uint64_t>(patch * redae_dim),
                    sampling_policy_) *
                static_cast<uint64_t>(slot.step);
            current[s] = sampling::generate_torch_cuda_tensor_iterator_randn(
                static_cast<size_t>(patch * redae_dim),
                slot.request.seed,
                noise_offset,
                sampling_policy_,
                sampling::TorchRandnPrecision::Float32);
            base_row[s] = total_batch;
            total_batch += survivors[s].cfg_batch;
        }
        const auto & schedule = slots_[static_cast<size_t>(survivors[0].id)].schedule;
        for (size_t i = 0; i + 1 < schedule.size(); ++i) {
            std::vector<float> x_in(static_cast<size_t>(total_batch * tokens * in_channels), 0.0F);
            std::vector<float> time_all(static_cast<size_t>(total_batch * 256), 0.0F);
            const auto te = firered_timestep_embedding(schedule[i]);
            for (int64_t s = 0; s < n; ++s) {
                auto & slot = slots_[static_cast<size_t>(survivors[s].id)];
                const int64_t b0 = base_row[s];
                const int64_t cfg = survivors[s].cfg_batch;
                for (int64_t b = 0; b < cfg; ++b) {
                    const int64_t g = b0 + b;
                    std::copy(te.begin(), te.end(),
                              time_all.begin() + static_cast<std::ptrdiff_t>(g * 256));
                    for (int64_t t = 0; t < tokens; ++t) {
                        float * row = x_in.data() + static_cast<size_t>((g * tokens + t) * in_channels);
                        if (t < history_tokens) {
                            std::copy(
                                history[s].begin() + static_cast<std::ptrdiff_t>(t * redae_dim),
                                history[s].begin() + static_cast<std::ptrdiff_t>((t + 1) * redae_dim),
                                row);
                        } else {
                            const int64_t local = t - history_tokens;
                            std::copy(
                                current[s].begin() + static_cast<std::ptrdiff_t>(local * redae_dim),
                                current[s].begin() + static_cast<std::ptrdiff_t>((local + 1) * redae_dim),
                                row);
                        }
                        if (b == 0) {
                            const int64_t cond_row = t / patch;
                            std::copy(
                                survivors[s].dit_cond3.begin() + static_cast<std::ptrdiff_t>(cond_row * assets_->base.dit_hidden_size),
                                survivors[s].dit_cond3.begin() + static_cast<std::ptrdiff_t>((cond_row + 1) * assets_->base.dit_hidden_size),
                                row + redae_dim);
                            std::copy(slot.spk_dit.begin(), slot.spk_dit.end(),
                                      row + redae_dim + assets_->base.dit_hidden_size);
                        }
                    }
                }
            }
            auto t_fl0 = std::chrono::steady_clock::now();
            auto pred = flow_->run(x_in, time_all, total_batch);
            prof_.flow_ms += std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - t_fl0).count();
            const float dt = schedule[i + 1] - schedule[i];
            const int64_t row_elems = patch * redae_dim;
            for (int64_t s = 0; s < n; ++s) {
                auto & slot = slots_[static_cast<size_t>(survivors[s].id)];
                const int64_t b0 = base_row[s];
                const int64_t cfg = survivors[s].cfg_batch;
                for (int64_t t = 0; t < patch; ++t) {
                    for (int64_t c = 0; c < redae_dim; ++c) {
                        const size_t idx = static_cast<size_t>(t * redae_dim + c);
                        const size_t cond_off = static_cast<size_t>(b0 * row_elems + idx);
                        float vt = pred[cond_off];
                        if (cfg == 2) {
                            const float uncond = pred[static_cast<size_t>((b0 + 1) * row_elems + idx)];
                            vt = (1.0F + slot.request.guidance_scale) * vt - slot.request.guidance_scale * uncond;
                        }
                        current[s][idx] += dt * vt;
                    }
                }
            }
        }
        return current;
    }

    // 阶段 C：一个 slot 的 AR patch 收尾（flow 产出 next_latent 后）。
    void finish_slot_patch_locked(int64_t slot_id, std::vector<float> next_latent) {
        auto & slot = slots_[static_cast<size_t>(slot_id)];
        slot.latents_gen.insert(slot.latents_gen.end(), next_latent.begin(), next_latent.end());
        slot.next_input = ar_->patch_encode(next_latent);
        slot.chunk_latents.insert(slot.chunk_latents.end(), next_latent.begin(), next_latent.end());
        slot.generated_patches++;
        slot.step++;
        prof_.n_steps++;
        prof_.n_patches++;
        if (slot.generated_patches >= slot.chunk_target && slot.chunk_index < slot.chunks.size()) {
            auto t_red0 = std::chrono::steady_clock::now();
            auto audio = redae_->decode_incremental(slot.redae_state, slot.chunk_latents);
            prof_.redae_ms += std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - t_red0).count();
            prof_.n_redae++;
            slot.chunk_latents.clear();
            slot.chunk_index++;
            if (slot.chunk_index < slot.chunks.size()) {
                slot.chunk_target += slot.chunks[slot.chunk_index];
            }
            if (!audio.samples.empty()) {
                slot.chunk_queue.push_back(std::move(audio));
                cv_.notify_all();
            }
        }
    }

    // step-0 AR：用 prefill_hidden 的 stop/one_backbone 走 AR，不 consume decode。
    // 生成的第一个 latent 作为 next_input，随后进入 batched decode（step>=1）。
    void advance_slot_from_prefill_locked(int64_t slot_id) {
        auto & slot = slots_[static_cast<size_t>(slot_id)];
        const int64_t hidden = assets_->base.hidden_size;
        const int64_t prompt_patches = slot.prompt_latent_frames / assets_->base.patch_size;

        const std::vector<float> last(
            slot.prefill_hidden.end() - hidden, slot.prefill_hidden.end());
        const float stop = ar_->stop(last);
        if (stop >= slot.request.stop_threshold && slot.step >= 6) {
            slot.state = Slot::State::Dead;
            return;
        }
        std::vector<float> one_backbone(
            slot.prefill_hidden.end() - static_cast<std::ptrdiff_t>(prompt_patches * hidden),
            slot.prefill_hidden.end());
        slot.backbone_cond.insert(slot.backbone_cond.end(), one_backbone.begin(), one_backbone.end());
        const int64_t cond_rows = static_cast<int64_t>(slot.backbone_cond.size()) / hidden;
        auto cond3 = last_rows(slot.backbone_cond, cond_rows, hidden, assets_->base.history_patches + 1);
        auto dit_cond3 = ar_->dit_head(cond3, assets_->base.history_patches + 1);
        const auto next_latent = flow_one_patch_locked(slot, dit_cond3, 0);
        slot.latents_gen.insert(slot.latents_gen.end(), next_latent.begin(), next_latent.end());
        slot.next_input = ar_->patch_encode(next_latent);
        slot.chunk_latents.insert(slot.chunk_latents.end(), next_latent.begin(), next_latent.end());
        slot.generated_patches++;
        slot.step = 1;
    }

    // flow + CFG 单 patch（与 pipeline.cpp 的 flow_one_patch 一致，但使用 slot 状态）。
    std::vector<float> flow_one_patch_locked(Slot & slot, const std::vector<float> & dit_cond3, uint64_t step_index) {
        const int64_t history_tokens = assets_->base.history_patches * assets_->base.patch_size;
        const int64_t tokens = history_tokens + assets_->base.patch_size;
        const int64_t input_channels = assets_->base.redae_dim + assets_->base.dit_hidden_size + assets_->base.speaker_dim;
        auto history = last_rows(
            slot.latents_gen,
            static_cast<int64_t>(slot.latents_gen.size()) / assets_->base.redae_dim,
            assets_->base.redae_dim,
            history_tokens);
        const uint64_t noise_offset =
            sampling::torch_cuda_tensor_iterator_offset_blocks(
                static_cast<uint64_t>(assets_->base.patch_size * assets_->base.redae_dim),
                sampling_policy_) *
            step_index;
        auto current = sampling::generate_torch_cuda_tensor_iterator_randn(
            static_cast<size_t>(assets_->base.patch_size * assets_->base.redae_dim),
            slot.request.seed,
            noise_offset,
            sampling_policy_,
            sampling::TorchRandnPrecision::Float32);

        const int64_t batch = slot.request.guidance_scale > 0.0F ? 2 : 1;
        for (size_t i = 0; i + 1 < slot.schedule.size(); ++i) {
            std::vector<float> x_in(static_cast<size_t>(batch * tokens * input_channels), 0.0F);
            for (int64_t b = 0; b < batch; ++b) {
                for (int64_t t = 0; t < tokens; ++t) {
                    float * row = x_in.data() + static_cast<size_t>((b * tokens + t) * input_channels);
                    if (t < history_tokens) {
                        std::copy(
                            history.begin() + static_cast<std::ptrdiff_t>(t * assets_->base.redae_dim),
                            history.begin() + static_cast<std::ptrdiff_t>((t + 1) * assets_->base.redae_dim),
                            row);
                    } else {
                        const int64_t local = t - history_tokens;
                        std::copy(
                            current.begin() + static_cast<std::ptrdiff_t>(local * assets_->base.redae_dim),
                            current.begin() + static_cast<std::ptrdiff_t>((local + 1) * assets_->base.redae_dim),
                            row);
                    }
                    if (b == 0) {
                        const int64_t cond_row = t / assets_->base.patch_size;
                        std::copy(
                            dit_cond3.begin() + static_cast<std::ptrdiff_t>(cond_row * assets_->base.dit_hidden_size),
                            dit_cond3.begin() + static_cast<std::ptrdiff_t>((cond_row + 1) * assets_->base.dit_hidden_size),
                            row + assets_->base.redae_dim);
                        std::copy(slot.spk_dit.begin(), slot.spk_dit.end(),
                                  row + assets_->base.redae_dim + assets_->base.dit_hidden_size);
                    }
                }
            }
            auto te = firered_timestep_embedding(slot.schedule[i]);
            std::vector<float> time(static_cast<size_t>(batch * 256));
            for (int64_t b = 0; b < batch; ++b) {
                std::copy(te.begin(), te.end(), time.begin() + static_cast<std::ptrdiff_t>(b * 256));
            }
            auto pred = flow_->run(x_in, time, batch);
            const float dt = slot.schedule[i + 1] - slot.schedule[i];
            for (int64_t t = 0; t < assets_->base.patch_size; ++t) {
                for (int64_t c = 0; c < assets_->base.redae_dim; ++c) {
                    const size_t idx = static_cast<size_t>(t * assets_->base.redae_dim + c);
                    float vt = pred[idx];
                    if (batch == 2) {
                        const float uncond = pred[static_cast<size_t>((assets_->base.patch_size + t) * assets_->base.redae_dim + c)];
                        vt = (1.0F + slot.request.guidance_scale) * vt - slot.request.guidance_scale * uncond;
                    }
                    current[idx] += dt * vt;
                }
            }
        }
        return current;
    }

    // 完成一个 slot：flush RedAE 尾部、标记 finished（Idle），等 owner 排空 + release。
    // 只处理 Dead / Failed；finished 后槽位暂归 owner，绝不在此 push free_slots_。
    void finish_slot_locked(int64_t slot_id) {
        auto & slot = slots_[static_cast<size_t>(slot_id)];
        if (slot.state != Slot::State::Dead && slot.state != Slot::State::Failed) {
            return;
        }
        fprintf(stderr, "[FINISH] slot=%ld tokens=%zu gen_patches=%ld\n",
                slot_id, slot.request.token_ids.size(), slot.generated_patches);
        fflush(stderr);
        if (slot.state == Slot::State::Dead) {
            // 有剩余 chunk_latents → decode_incremental 输出 tail（含 istft 尾部）；
            // 无剩余 → flush_incremental 输出尾部。二选一，避免 decode + flush 双输出。
            if (!slot.chunk_latents.empty() && slot.generated_patches > 0) {
                auto t_red0 = std::chrono::steady_clock::now();
                auto audio = redae_->decode_incremental(slot.redae_state, slot.chunk_latents);
                prof_.redae_ms += std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - t_red0).count();
                prof_.n_redae++;
                slot.chunk_latents.clear();
                if (!audio.samples.empty()) {
                    slot.chunk_queue.push_back(std::move(audio));
                }
            } else {
                auto t_red0 = std::chrono::steady_clock::now();
                auto flush = redae_->flush_incremental(slot.redae_state);
                prof_.redae_ms += std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - t_red0).count();
                prof_.n_redae++;
                if (!flush.samples.empty()) {
                    slot.chunk_queue.push_back(std::move(flush));
                }
            }
        }
        // Failed slot 的 error 由 owner next_chunk rethrow；此处仅置 finished。
        slot.state = Slot::State::Idle;
        slot.finished = true;
        // 该 slot 的 GPU 行内容由下一轮全量 import 清零；期间它不在 active（active_mask
        // 置 0 → 整行 -inf 且不 advance），绝不参与 decode、绝不污染其他行。
        // [PROF] 打印自上次 finish 以来的分项累计（单路时即该请求自身）
        const double d_ar = prof_.ar_decode_ms - prof_last_.ar_decode_ms;
        const double d_flow = prof_.flow_ms - prof_last_.flow_ms;
        const double d_red = prof_.redae_ms - prof_last_.redae_ms;
        fprintf(stderr,
                "[PROF] slot=%ld steps=%lld patches=%lld redae=%lld | AR_decode=%.1fms flow=%.1fms "
                "redae=%.1fms | ar%%=%.0f flow%%=%.0f redae%%=%.0f | per_patch_ar=%.2fms per_patch_flow=%.2fms\n",
                slot_id,
                (long long)(prof_.n_steps - prof_last_.n_steps),
                (long long)(prof_.n_patches - prof_last_.n_patches),
                (long long)(prof_.n_redae - prof_last_.n_redae),
                d_ar, d_flow, d_red,
                d_ar + d_flow + d_red > 0 ? 100.0 * d_ar / (d_ar + d_flow + d_red) : 0,
                d_ar + d_flow + d_red > 0 ? 100.0 * d_flow / (d_ar + d_flow + d_red) : 0,
                d_ar + d_flow + d_red > 0 ? 100.0 * d_red / (d_ar + d_flow + d_red) : 0,
                prof_.n_patches - prof_last_.n_patches > 0 ? d_ar / (prof_.n_patches - prof_last_.n_patches) : 0,
                prof_.n_patches - prof_last_.n_patches > 0 ? d_flow / (prof_.n_patches - prof_last_.n_patches) : 0);
        fflush(stderr);
        prof_last_ = prof_;
        cv_.notify_all();
    }

    // 是否有待启动请求或待收尾 slot（决定 scheduler_loop 是否休眠）。
    bool has_work_locked() const {
        if (!pending_.empty()) {
            return true;
        }
        for (int64_t i = 0; i < max_batch_; ++i) {
            const auto & slot = slots_[static_cast<size_t>(i)];
            if (slot.state == Slot::State::Active || slot.state == Slot::State::Dead ||
                slot.state == Slot::State::Failed) {
                return true;
            }
        }
        return false;
    }

    // 开轮收集窗口：pending 有待启动请求、且当前无任何 round 参与者在跑时，说明
    // "下一轮即将开始"。此时若立刻 tick，第一个请求会单独 prefill 成一轮（round
    // 拆轮 —— 同一瞬间到达的其余请求被 gate 挡到整轮跑完才并下一轮，flow batch
    // 因而失效）。解法：短暂等待，让同刻到达的请求在 pending 里聚集，再一次性开轮。
    // 已攒满（pending 能占满空闲 slot）则不等直接开。
    bool should_collect_for_round_locked() const {
        if (pending_.empty()) {
            return false;
        }
        if (any_round_participant_locked()) {
            return false;  // 已有轮在跑，无需收集（新 pending 等本轮 drain）
        }
        // pending 已占满 max_batch → 一轮已能打满，不必再等。
        return static_cast<int64_t>(pending_.size()) < max_batch_;
    }

    // 专用调度线程主循环（唯一 GPU decode 线程）。
    // 持锁跑 tick（一次 batched decode 推进所有活跃行），无工作/无 pending 时等待。
    void scheduler_loop() {
        std::unique_lock<std::mutex> lock(mu_);
        constexpr auto kRoundCollectWindow = std::chrono::microseconds(8000);
        while (!stop_) {
            // 开轮前收集：若下一轮即将开始且还没攒满，等一小段让同刻请求加入同一轮，
            // 避免 round 拆轮。collect_started_ 防连续多个 1.5ms 空等（攒满/超时即开）。
            if (should_collect_for_round_locked() && !collect_started_) {
                collect_started_ = true;
                fprintf(stderr, "[COLLECT] t=%lld pend=%zu free=%zu waiting %dus\n",
                        (long long)tick_count_, pending_.size(), free_slots_.size(),
                        (int)std::chrono::duration_cast<std::chrono::microseconds>(kRoundCollectWindow).count());
                fflush(stderr);
                cv_wake_.wait_for(lock, kRoundCollectWindow);
                collect_started_ = false;
                // wait_for 醒来即继续到 tick —— pending 已尽量聚集，本轮一并 prefill
            }
            // 记录 tick 前各 slot 的已产出 chunk 数
            size_t chunks_before = 0;
            for (int64_t i = 0; i < max_batch_; ++i) {
                chunks_before += slots_[static_cast<size_t>(i)].chunk_queue.size();
            }
            tick_locked();
            // 若本 tick 产出了新 chunk，短暂让出锁：让 next_chunk（cv_.wait）能抢到锁
            // 取走 chunk，实现真正的增量流式（否则调度线程持锁持续 tick，一次性
            // 生成完整个 round，流式退化成离线）。
            size_t chunks_after = 0;
            for (int64_t i = 0; i < max_batch_; ++i) {
                chunks_after += slots_[static_cast<size_t>(i)].chunk_queue.size();
            }
            if (chunks_after > chunks_before) {
                cv_wake_.wait_for(lock, std::chrono::milliseconds(1));
            }
            if (!has_work_locked()) {
                cv_wake_.wait(lock);
            }
        }
    }

    // [DIAG] 轮/步日志是否开启（仅看是否有待处理请求，避免纯空闲刷屏）。
    bool diag_enabled() const {
        return tick_count_ < 1000000;
    }

    size_t slot_prefill_layer_count() const {
        for (int64_t i = 0; i < max_batch_; ++i) {
            auto & slot = slots_[static_cast<size_t>(i)];
            if (slot.prefill_done && !slot.prefill_state.layers.empty()) {
                return slot.prefill_state.layers.size();
            }
        }
        return 0;
    }

    std::shared_ptr<const FireRedTTS3Assets> assets_;
    engine::core::ExecutionContext & execution_;
    bool mem_saver_ = false;
    int64_t max_batch_ = 1;
    engine::assets::TensorStorageType storage_type_ = engine::assets::TensorStorageType::Native;
    sampling::TorchCudaSamplingPolicy sampling_policy_;
    std::unique_ptr<FireRedArRuntime> ar_;
    std::unique_ptr<FireRedRedAeRuntime> redae_;
    std::unique_ptr<FireRedFlowRuntime> flow_;
    modules::CampplusEncoderComponent campplus_;
    runtime::CacheSlots<ReferenceVoiceCacheKey, ReferenceVoiceCacheEntry, ReferenceVoiceCacheKeyEqual> reference_voice_cache_;

    std::mutex mu_;
    std::condition_variable cv_;       // chunk 产出 / slot 结束通知（next_chunk 等待）
    std::condition_variable cv_wake_;  // 调度线程唤醒（launch / release / stop）
    std::vector<Slot> slots_;
    std::deque<int64_t> free_slots_;
    std::deque<int64_t> pending_;
    bool stop_ = false;
    std::thread scheduler_thread_;
    int64_t tick_count_ = 0;  // [DIAG]
    bool collect_started_ = false;  // 开轮收集窗口进行中（防连续空等）

    // [PROF] 各组件累计耗时（单调度线程，串行累计；finish_slot 时打印分项）
    struct ProfAccum {
        double ar_decode_ms = 0;    // batched AR decode（图前 + 图后 readback）
        double flow_ms = 0;         // flow_one_patch_locked（整函数，含 dit_head? 不含，dit_head 单独在 advance）
        double redae_ms = 0;        // RedAE decode_incremental / flush
        double other_ms = 0;        // 其余（advance 的 stop/dit_head/patch_encode/backbone 等）
        int64_t n_steps = 0;        // AR step 数
        int64_t n_patches = 0;      // generated patches
        int64_t n_redae = 0;        // redae 调用次数
    } prof_;
    ProfAccum prof_last_;  // [PROF] 上次打印时的累计快照
    // 一段代码的计时 RAII：析构时把 elapsed_ms 累加到 acc
    struct ProfScoped {
        ProfAccum & acc;
        double & slot;
        std::chrono::steady_clock::time_point t0;
        ProfScoped(ProfAccum & a, double & target) : acc(a), slot(target) {
            (void)acc;
            t0 = std::chrono::steady_clock::now();
        }
        ~ProfScoped() {
            slot += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        }
    };
};

FireRedTTS3BatchScheduler::FireRedTTS3BatchScheduler(
    std::shared_ptr<const FireRedTTS3Assets> assets,
    engine::core::ExecutionContext & execution,
    size_t graph_arena_bytes,
    size_t helper_graph_arena_bytes,
    size_t weight_context_bytes,
    engine::assets::TensorStorageType storage_type,
    size_t reference_cache_slots,
    bool mem_saver,
    int64_t max_batch)
    : impl_(std::make_unique<Impl>(
          std::move(assets), execution, graph_arena_bytes, helper_graph_arena_bytes,
          weight_context_bytes, storage_type, reference_cache_slots, mem_saver, max_batch)) {}

FireRedTTS3BatchScheduler::~FireRedTTS3BatchScheduler() = default;

FireRedTTS3BatchScheduler::SlotHandle FireRedTTS3BatchScheduler::launch(
    const FireRedTTS3BaseRequest & request,
    const std::vector<int64_t> & chunk_patches) {
    return impl_->launch(request, chunk_patches);
}

engine::runtime::AudioBuffer FireRedTTS3BatchScheduler::next_chunk(const SlotHandle & handle) {
    return impl_->next_chunk(handle);
}

void FireRedTTS3BatchScheduler::release_slot(const SlotHandle & handle) {
    impl_->release_slot(handle);
}

void FireRedTTS3BatchScheduler::abort(const SlotHandle & handle) {
    impl_->abort(handle);
}

engine::runtime::AudioBuffer FireRedTTS3BatchScheduler::generate(const FireRedTTS3BaseRequest & request) {
    return impl_->generate(request);
}

int64_t FireRedTTS3BatchScheduler::max_batch() const noexcept {
    return impl_->max_batch();
}

void FireRedTTS3BatchScheduler::release_graphs() {
    impl_->release_graphs();
}

}  // namespace engine::models::fireredtts3
