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
#include <set>
#include <stdexcept>
#include <thread>

namespace engine::models::fireredtts3 {
namespace {

namespace core = engine::core;
namespace modules = engine::modules;
using Clock = std::chrono::steady_clock;

constexpr int64_t kMaxArSteps = 400;
// prefill graph 固定 steps（grow-only 复用）：所有 wave 的 prefill 都 padding 到该长度，
// 避免因 steps 变化重建 prefill graph 破坏 CUDA pool 逆序 free 约束。
constexpr int64_t kMaxPrefillSteps = 1024;
// batched decode KV cache 固定上限（首次 build 后永不重建）：
// 最大可能 prefill（参考 + 长文本 ~300） + kMaxArSteps(400) + 尾部余量。
// 若实际 prefill_steps 超此预算，decode 会 cache 耗尽（罕见，可调大）。
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

    int64_t launch(const FireRedTTS3BaseRequest & request, const std::vector<int64_t> & chunk_patches) {
        std::lock_guard<std::mutex> lock(mu_);
        if (free_slots_.empty()) {
            return -1;
        }
        const int64_t slot_id = free_slots_.front();
        free_slots_.pop_front();
        auto & slot = slots_[static_cast<size_t>(slot_id)];
        slot = Slot{};
        slot.state = Slot::State::Active;
        slot.request = request;
        slot.chunks = chunk_patches.empty() ? std::vector<int64_t>{400} : chunk_patches;
        slot.chunk_target = slot.chunks[0];
        pending_.push_back(slot_id);
        cv_wake_.notify_all();
        return slot_id;
    }

    engine::runtime::AudioBuffer next_chunk(int64_t slot_id) {
        std::unique_lock<std::mutex> lock(mu_);
        auto & slot = slots_[static_cast<size_t>(slot_id)];
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
        // ggml CUDA pool 逆序约束 -> 唯一 GPU decode 线程必须是调度线程。
        // 用带谓词的 wait（wait 内部重查条件），避免"检查 queue 后、wait 前"的
        // notify 竞态窗口导致死等。
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
        const int64_t slot_id = launch(request, {});
        if (slot_id < 0) {
            throw std::runtime_error("FireRedTTS3BatchScheduler slot pool exhausted");
        }
        engine::runtime::AudioBuffer merged;
        merged.sample_rate = static_cast<int>(assets_->redae.sample_rate);
        merged.channels = 1;
        while (true) {
            auto chunk = next_chunk(slot_id);
            if (chunk.samples.empty()) {
                break;
            }
            runtime::append_audio_buffer(merged, chunk);
        }
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

    // 单个 slot 的 prefill（锁内串行）：参考 prep + 单路径 prefill，KV 存 CPU。
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

    // 是否存在活跃 decode slot（step>=1，已在 batched KV 中）。
    bool has_active_decode_locked() const {
        for (int64_t i = 0; i < max_batch_; ++i) {
            const auto & slot = slots_[static_cast<size_t>(i)];
            if (slot.state == Slot::State::Active && slot.prefill_done && slot.step >= 1) {
                return true;
            }
        }
        return false;
    }

    // ---- 调度主循环（锁内）----
    // 与 pipeline.cpp 的单路径 AR 循环逐位对应：
    //   step 0: 用 prefill_hidden 的 stop/one_backbone 走 AR（不 consume decode）
    //   step>=1: decode_embeddings_batched 产出 hidden 后走 AR
    void tick_locked() {
        // 1. 处理 pending 启动（prefill 新 slot）。仅当无活跃 decode slot 时才 prefill
        //    （wave batching）：prefill graph 与 batched decode graph 共用同一 CUDA
        //    device pool，若在 decode 进行中 prefill（重建 prefill graph），会破坏
        //    pool 的逆序 free 约束（GGML_ASSERT(ptr == pool_addr + pool_used)）。
        if (!pending_.empty() && !has_active_decode_locked()) {
            form_wave_locked();
        }
        // 2. 对 step==0 且已 prefill 的 slot 走 step-0 AR（不进 batch decode，
        //    否则零行会被 advance_member 污染 KV 位置）。
        for (int64_t i = 0; i < max_batch_; ++i) {
            auto & slot = slots_[static_cast<size_t>(i)];
            if (slot.state == Slot::State::Active && slot.prefill_done && slot.step == 0) {
                advance_slot_from_prefill_locked(i);
            }
        }
        // 3. 收集 step>=1 的活跃 slot（参与 batched decode）
        std::vector<int64_t> active;
        for (int64_t i = 0; i < max_batch_; ++i) {
            auto & slot = slots_[static_cast<size_t>(i)];
            if (slot.state == Slot::State::Active && slot.prefill_done && slot.step >= 1) {
                active.push_back(i);
            }
        }
        if (active.empty()) {
            // 没有需要 decode 的 slot；若 step-0 AR 让某些 slot 进入 Dead，
            // 本轮结束前清理。
            for (int64_t i = 0; i < max_batch_; ++i) {
                auto & slot = slots_[static_cast<size_t>(i)];
                if (slot.state == Slot::State::Dead || slot.state == Slot::State::Done) {
                    finish_slot_locked(i);
                }
            }
            return;
        }
        // 4. 若 batched decode 未启动，或需要 splice 新 slot（step-0 AR 后），重建。
        if (!batched_started_ || !imported_slots_.empty()) {
            rebuild_batched_state_locked();
        }
        // 5. 收集 embeddings；非活跃行零填充
        const int64_t hidden = assets_->base.hidden_size;
        std::vector<float> embeddings(static_cast<size_t>(max_batch_ * hidden), 0.0F);
        for (int64_t i = 0; i < max_batch_; ++i) {
            auto & slot = slots_[static_cast<size_t>(i)];
            if (slot.state == Slot::State::Active && slot.prefill_done && slot.step >= 1) {
                std::copy(slot.next_input.begin(), slot.next_input.end(),
                          embeddings.begin() + static_cast<std::ptrdiff_t>(i * hidden));
            }
        }
        // 6. 一次 batched decode（每成员独立 pos/cache-slot/mask）
        auto out = ar_->decode_embeddings_batched(embeddings, max_batch_);
        // 7. 分发 hidden 到各 step>=1 活跃 slot
        for (int64_t i = 0; i < max_batch_; ++i) {
            auto & slot = slots_[static_cast<size_t>(i)];
            if (slot.state != Slot::State::Active || !slot.prefill_done || slot.step < 1) {
                continue;
            }
            std::vector<float> row(
                out.hidden.begin() + static_cast<std::ptrdiff_t>(i * hidden),
                out.hidden.begin() + static_cast<std::ptrdiff_t>((i + 1) * hidden));
            advance_slot_locked(i, row);
        }
        // 8. 处理完成/死亡的 slot（flush RedAE 尾部，标记 finished，归还槽位）
        for (int64_t i = 0; i < max_batch_; ++i) {
            auto & slot = slots_[static_cast<size_t>(i)];
            if (slot.state == Slot::State::Dead || slot.state == Slot::State::Done) {
                finish_slot_locked(i);
            }
        }
    }

    // step-0 AR：用 prefill_hidden 的 stop/one_backbone 走 AR，不 consume decode。
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

    // 单 slot 一步 AR（step>=1：hidden 来自一次 batched decode）。
    void advance_slot_locked(int64_t slot_id, const std::vector<float> & hidden_row) {
        auto & slot = slots_[static_cast<size_t>(slot_id)];
        const int64_t hidden = assets_->base.hidden_size;

        // AR 步数上限（与 pipeline.cpp 的 max_steps=400 一致）：stop 未触发时强制结束。
        if (slot.step >= kMaxArSteps) {
            slot.state = Slot::State::Dead;
            return;
        }
        const float stop = ar_->stop(hidden_row);
        if (stop >= slot.request.stop_threshold && slot.step >= 6) {
            slot.state = Slot::State::Dead;
            return;
        }

        slot.backbone_cond.insert(slot.backbone_cond.end(), hidden_row.begin(), hidden_row.end());
        const int64_t cond_rows = static_cast<int64_t>(slot.backbone_cond.size()) / hidden;
        auto cond3 = last_rows(slot.backbone_cond, cond_rows, hidden, assets_->base.history_patches + 1);
        auto dit_cond3 = ar_->dit_head(cond3, assets_->base.history_patches + 1);
        const auto next_latent = flow_one_patch_locked(slot, dit_cond3, static_cast<uint64_t>(slot.step));
        slot.latents_gen.insert(slot.latents_gen.end(), next_latent.begin(), next_latent.end());
        slot.next_input = ar_->patch_encode(next_latent);
        slot.chunk_latents.insert(slot.chunk_latents.end(), next_latent.begin(), next_latent.end());
        slot.generated_patches++;
        slot.step++;

        // chunk 边界（与 pipeline.cpp 相同）
        if (slot.generated_patches >= slot.chunk_target && slot.chunk_index < slot.chunks.size()) {
            auto audio = redae_->decode_incremental(slot.redae_state, slot.chunk_latents);
            slot.chunk_latents.clear();
            slot.chunk_index++;
            if (slot.chunk_index < slot.chunks.size()) {
                slot.chunk_target += slot.chunks[slot.chunk_index];
            }
            if (!audio.samples.empty()) {
                slot.chunk_queue.push_back(std::move(audio));
                cv_.notify_all();  // 唤醒等待该 slot 的 next_chunk
            }
        }
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

    // 完成一个 slot：flush RedAE 尾部、标记 finished、归还槽位。
    // 只处理 Dead（Done 已处理过并归还）；Done 状态由 launch 复用前的 reset 清除。
    void finish_slot_locked(int64_t slot_id) {
        auto & slot = slots_[static_cast<size_t>(slot_id)];
        if (slot.state != Slot::State::Dead) {
            return;
        }
        // 还有未解完的 chunk_latents → flush
        if (!slot.chunk_latents.empty() && slot.generated_patches > 0) {
            auto audio = redae_->decode_incremental(slot.redae_state, slot.chunk_latents);
            slot.chunk_latents.clear();
            if (!audio.samples.empty()) {
                slot.chunk_queue.push_back(std::move(audio));
            }
        }
        auto flush = redae_->flush_incremental(slot.redae_state);
        if (!flush.samples.empty()) {
            slot.chunk_queue.push_back(std::move(flush));
        }
        slot.state = Slot::State::Idle;
        slot.finished = true;
        // 防重复归还：仅当 slot 不在 free_slots_ 时才 push（避免并发下重复导致
        // 同一 slot 被多个 session launch）。
        if (std::find(free_slots_.begin(), free_slots_.end(), slot_id) == free_slots_.end()) {
            free_slots_.push_back(slot_id);
        }
        // 该行不再参与 decode：从 running 集合移除（下次 join 时重建）。
        running_active_.erase(
            std::remove(running_active_.begin(), running_active_.end(), slot_id),
            running_active_.end());
        imported_slots_.erase(slot_id);
        // 该行 KV 在 GPU 上仍残留 decode 产物；标记脏，下次 rebuild 时清零。
        dirty_rows_.insert(slot_id);
        cv_.notify_all();
    }

    // 专用调度线程主循环（唯一 GPU decode 线程）。
    // 持锁跑 tick（batch decode 推进所有活跃 slot），无活跃 slot 时等待 launch/stop。
    void scheduler_loop() {
        std::unique_lock<std::mutex> lock(mu_);
        while (!stop_) {
            tick_locked();
            // 有工作（Active 未完成 / Dead 待 finish）则继续 tick；否则等新工作。
            bool has_work = false;
            for (int64_t i = 0; i < max_batch_; ++i) {
                auto & slot = slots_[static_cast<size_t>(i)];
                if (slot.state == Slot::State::Active ||
                    slot.state == Slot::State::Dead) {
                    has_work = true;
                    break;
                }
            }
            if (!has_work) {
                cv_wake_.wait(lock);
            }
        }
    }

    // 形成 wave：prefill pending slot（锁内串行）。prefill 后标记需 splice。
    void form_wave_locked() {
        // 一次最多 prefill 直到槽池满（但同一轮只 prefill，不挤占已活跃 slot 的 GPU）
        std::vector<int64_t> newly;
        while (!pending_.empty() && newly.size() < static_cast<size_t>(max_batch_)) {
            const int64_t slot_id = pending_.front();
            pending_.pop_front();
            newly.push_back(slot_id);
        }
        for (int64_t slot_id : newly) {
            try {
                prefill_slot_locked(slot_id);
                imported_slots_.insert(slot_id);  // 标记需要 splice 进 batched KV
            } catch (...) {
                auto & slot = slots_[static_cast<size_t>(slot_id)];
                slot.error = std::current_exception();
                slot.state = Slot::State::Failed;
                slot.finished = true;
                if (std::find(free_slots_.begin(), free_slots_.end(), slot_id) == free_slots_.end()) {
                    free_slots_.push_back(slot_id);
                }
            }
        }
    }

    // 重建 batched KV state：export 当前 GPU state + splice 新 prefill slot 的 CPU KV + 重新 start。
    // 固定 max_batch graph，不重建（batch_size 不变、cache_steps 不减）。
    void rebuild_batched_state_locked() {
        // 需要 splice 的新 slot（尚未在 batched 中）
        std::vector<int64_t> joining;
        for (int64_t i = 0; i < max_batch_; ++i) {
            auto & slot = slots_[static_cast<size_t>(i)];
            if (slot.prefill_done && imported_slots_.count(i) > 0) {
                joining.push_back(i);
            }
        }
        if (batched_started_ && !joining.empty()) {
            // 从 GPU 导出当前 batched state（active slot 的中间 KV 保留）
            runtime::TransformerBatchedKVState gpu = ar_->export_batched_decode_state();
            // 层数/row_elems 校验（与 import_state 的期望布局一致）
            if (gpu.layers.size() != slot_prefill_layer_count()) {
                throw std::runtime_error("FireRedTTS3BatchScheduler batched state layer mismatch");
            }
            // 组装全量 batched state：max_steps = 所有"有数据行"（imported/running）的最大 end。
            // 空行（无 prefill、非 running）end=0，不参与 max_steps（避免 gpu 残留的空行
            // 高 end 撑大 state，导致与 current_ends 的 import max 不一致）。
            runtime::TransformerBatchedKVState state;
            state.batch_size = max_batch_;
            state.current_ends.assign(static_cast<size_t>(max_batch_), 0);
            for (int64_t i = 0; i < max_batch_; ++i) {
                if (imported_slots_.count(i) > 0) {
                    state.current_ends[static_cast<size_t>(i)] = slots_[static_cast<size_t>(i)].prefill_steps;
                } else if (std::find(running_active_.begin(), running_active_.end(), i) != running_active_.end()) {
                    state.current_ends[static_cast<size_t>(i)] = gpu_end_for(gpu, i);
                }
            }
            int64_t max_steps = 0;
            for (int64_t i = 0; i < max_batch_; ++i) {
                max_steps = std::max(max_steps, state.current_ends[static_cast<size_t>(i)]);
            }
            state.current_end = max_steps;
            state.layers.resize(gpu.layers.size());
            for (size_t layer = 0; layer < state.layers.size(); ++layer) {
                auto & out_layer = state.layers[layer];
                out_layer.valid_steps = max_steps;
                const size_t row_elems = row_elems_for_layer(layer);
                out_layer.key.assign(static_cast<size_t>(max_batch_) * static_cast<size_t>(max_steps) * row_elems, 0.0F);
                out_layer.value.assign(static_cast<size_t>(max_batch_) * static_cast<size_t>(max_steps) * row_elems, 0.0F);
            }
            // 填充每行：已运行 slot（running）从 gpu 拷贝其 ends；新 slot 从 prefill_state 拷贝；
            // 其余（dead/脏/空行）保持全零。
            for (int64_t b = 0; b < max_batch_; ++b) {
                if (imported_slots_.count(b) == 0 && std::find(running_active_.begin(), running_active_.end(), b) == running_active_.end()) {
                    continue;  // 非活跃行：全零（不拷 gpu 残留）
                }
                const int64_t end = (imported_slots_.count(b) > 0)
                    ? slots_[static_cast<size_t>(b)].prefill_steps
                    : gpu_end_for(gpu, b);
                if (end <= 0) {
                    continue;
                }
                const int64_t current_end = end;
                for (size_t layer = 0; layer < state.layers.size(); ++layer) {
                    const size_t row_elems = row_elems_for_layer(layer);
                    const size_t copy_elems = static_cast<size_t>(current_end) * row_elems;
                    float * dst_key = state.layers[layer].key.data()
                        + static_cast<size_t>(b) * static_cast<size_t>(max_steps) * row_elems;
                    float * dst_value = state.layers[layer].value.data()
                        + static_cast<size_t>(b) * static_cast<size_t>(max_steps) * row_elems;
                    if (imported_slots_.count(b) > 0) {
                        const auto & src = slots_[static_cast<size_t>(b)].prefill_state.layers[layer];
                        std::copy(src.key.begin(), src.key.end(), dst_key);
                        std::copy(src.value.begin(), src.value.end(), dst_value);
                    } else {
                        const auto & src = gpu.layers[layer];
                        const size_t src_row_elems = row_elems;
                        const int64_t gpu_valid_steps = gpu.layers.empty() ? 0 : gpu.layers.front().valid_steps;
                        const size_t src_offset = static_cast<size_t>(b) * static_cast<size_t>(gpu_valid_steps) * src_row_elems;
                        std::copy(
                            src.key.begin() + static_cast<std::ptrdiff_t>(src_offset),
                            src.key.begin() + static_cast<std::ptrdiff_t>(src_offset + copy_elems),
                            dst_key);
                        std::copy(
                            src.value.begin() + static_cast<std::ptrdiff_t>(src_offset),
                            src.value.begin() + static_cast<std::ptrdiff_t>(src_offset + copy_elems),
                            dst_value);
                    }
                }
            }
            const int64_t required_steps = compute_required_steps(max_steps);
            ar_->start_decode_embeddings_batched(state, required_steps);
            for (int64_t i : joining) {
                imported_slots_.erase(i);
            }
        } else if (!joining.empty()) {
            // 首次启动（无 GPU 导出，全部新 slot）
            int64_t max_steps = 0;
            for (int64_t i = 0; i < max_batch_; ++i) {
                auto & slot = slots_[static_cast<size_t>(i)];
                if (slot.prefill_done) {
                    max_steps = std::max(max_steps, slot.prefill_steps);
                }
            }
            runtime::TransformerBatchedKVState state;
            state.batch_size = max_batch_;
            state.current_end = max_steps;
            state.current_ends.assign(static_cast<size_t>(max_batch_), 0);
            state.layers.resize(slot_prefill_layer_count());
            for (size_t layer = 0; layer < state.layers.size(); ++layer) {
                auto & out_layer = state.layers[layer];
                out_layer.valid_steps = max_steps;
                const size_t row_elems = row_elems_for_layer(layer);
                out_layer.key.assign(static_cast<size_t>(max_batch_) * static_cast<size_t>(max_steps) * row_elems, 0.0F);
                out_layer.value.assign(static_cast<size_t>(max_batch_) * static_cast<size_t>(max_steps) * row_elems, 0.0F);
            }
            for (int64_t b = 0; b < max_batch_; ++b) {
                auto & slot = slots_[static_cast<size_t>(b)];
                if (!slot.prefill_done) {
                    continue;
                }
                state.current_ends[static_cast<size_t>(b)] = slot.prefill_steps;
                for (size_t layer = 0; layer < state.layers.size(); ++layer) {
                    const size_t row_elems = row_elems_for_layer(layer);
                    const size_t copy_elems = static_cast<size_t>(slot.prefill_steps) * row_elems;
                    float * dst_key = state.layers[layer].key.data()
                        + static_cast<size_t>(b) * static_cast<size_t>(max_steps) * row_elems;
                    float * dst_value = state.layers[layer].value.data()
                        + static_cast<size_t>(b) * static_cast<size_t>(max_steps) * row_elems;
                    const auto & src = slot.prefill_state.layers[layer];
                    std::copy(src.key.begin(), src.key.end(), dst_key);
                    std::copy(src.value.begin(), src.value.end(), dst_value);
                }
            }
            const int64_t required_steps = compute_required_steps(max_steps);
            ar_->start_decode_embeddings_batched(state, required_steps);
            // 全量分支：所有 prefill slot 已 splice 进 batched KV，清空 imported 标记，
            // 否则每次 tick 都会重复 rebuild（用 prefill_steps 覆盖 GPU 增量 KV）。
            imported_slots_.clear();
        }
        batched_started_ = true;
        dirty_rows_.clear();
        running_active_.clear();
        for (int64_t i = 0; i < max_batch_; ++i) {
            auto & slot = slots_[static_cast<size_t>(i)];
            if (slot.state == Slot::State::Active && slot.prefill_done) {
                running_active_.push_back(i);
            }
        }
    }

    // 计算 batched decode graph 所需 cache steps。为避免 graph 重建（昂贵），
    // 首次 build 用保守上限：max_steps + 完整 AR 预算。后续 join 若 max_steps
    // 增长超预算才重建（罕见），通常复用首次 graph。
    int64_t compute_required_steps(int64_t /*max_prefill_steps*/) const {
        // 固定 cache 上限：首次 build 后永不重建（避免 export 空 state / pool 顺序破坏）。
        // 预算 = 最大可能 prefill（参考+文本） + kMaxArSteps + 尾部。
        // KV 显存 = max_batch × cache_steps × layers × row_elems ≈ 可控。
        return kMaxDecodeCacheSteps;
    }

    // GPU batched state 中某行的当前 end（运行中 slot 用它作为拷贝长度）
    int64_t gpu_end_for(const runtime::TransformerBatchedKVState & gpu, int64_t batch) const {
        if (!gpu.current_ends.empty()) {
            if (static_cast<int64_t>(gpu.current_ends.size()) == gpu.batch_size) {
                return gpu.current_ends[static_cast<size_t>(batch)];
            }
        }
        return gpu.current_end;
    }

    size_t slot_prefill_layer_count() const {
        if (max_batch_ == 0) {
            return 0;
        }
        for (int64_t i = 0; i < max_batch_; ++i) {
            auto & slot = slots_[static_cast<size_t>(i)];
            if (slot.prefill_done && !slot.prefill_state.layers.empty()) {
                return slot.prefill_state.layers.size();
            }
        }
        return 0;
    }

    size_t row_elems_for_layer(size_t /*layer*/) const {
        // FireRed TTS3 AR: kv_heads × head_dim（同 pipeline prefill state 的层元素数）。
        return static_cast<size_t>(assets_->base.kv_heads * assets_->base.head_dim);
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
    std::condition_variable cv_wake_;  // 调度线程唤醒（launch / stop）
    std::vector<Slot> slots_;
    std::deque<int64_t> free_slots_;
    std::deque<int64_t> pending_;
    bool batched_started_ = false;
    bool stop_ = false;
    std::thread scheduler_thread_;
    // 已 spliced 进 batched KV 的 slot（运行中）；新 prefill 的 slot 在此集合外，需 join。
    std::vector<int64_t> running_active_;
    // 已 prefill 但尚未 splice 进 batched KV 的 slot id
    std::set<int64_t> imported_slots_;
    // 刚结束、GPU 行残留 KV 的 slot id（下次 rebuild 时该行清零）
    std::set<int64_t> dirty_rows_;
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

int64_t FireRedTTS3BatchScheduler::launch(const FireRedTTS3BaseRequest & request, const std::vector<int64_t> & chunk_patches) {
    return impl_->launch(request, chunk_patches);
}

engine::runtime::AudioBuffer FireRedTTS3BatchScheduler::next_chunk(int64_t slot_id) {
    return impl_->next_chunk(slot_id);
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
