#include "engine/models/firered_audio/pipeline.h"

#include "engine/framework/debug/profiler.h"
#include "engine/framework/runtime/cache_slots.h"
#include "engine/framework/sampling/hf_sampler.h"
#include "engine/framework/sampling/torch_random.h"
#include "engine/models/firered_audio/flow.h"
#include "engine/models/firered_audio/patch_encoder.h"
#include "engine/models/firered_audio/qwen35_runtime.h"
#include "engine/models/firered_audio/redae.h"
#include "engine/models/firered_audio/tokenizer.h"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstring>
#include <memory>
#include <random>
#include <stdexcept>
#include <limits>
#include <utility>
#include <vector>

namespace engine::models::firered_audio {
namespace {

namespace core = engine::core;
using Clock = std::chrono::steady_clock;

constexpr int64_t kDecodeCacheGrowthWindow = 1024;

uint64_t mix_cache_key(uint64_t key, uint64_t value) {
    key ^= value;
    key *= 1099511628211ull;
    return key;
}

uint64_t hash_audio_samples(const runtime::AudioBuffer & audio) {
    uint64_t key = 1469598103934665603ull;
    for (const float sample : audio.samples) {
        uint32_t bits = 0;
        std::memcpy(&bits, &sample, sizeof(bits));
        key = mix_cache_key(key, bits);
    }
    return key;
}

struct ReferenceAudioCacheKey {
    int sample_rate = 0;
    int channels = 0;
    uint64_t sample_count = 0;
    uint64_t sample_hash = 0;
};

struct ReferenceAudioCacheKeyEqual {
    bool operator()(const ReferenceAudioCacheKey & lhs, const ReferenceAudioCacheKey & rhs) const noexcept {
        return lhs.sample_rate == rhs.sample_rate &&
            lhs.channels == rhs.channels &&
            lhs.sample_count == rhs.sample_count &&
            lhs.sample_hash == rhs.sample_hash;
    }
};

struct ReferenceAudioCacheEntry {
    std::vector<float> audio_24k;
    std::vector<float> latents;
    std::vector<float> patch_embeddings;
};

std::vector<float> row_slice(
    const std::vector<float> & values,
    int64_t rows,
    int64_t width,
    int64_t begin,
    int64_t end) {
    if (begin < 0 || end < begin || end > rows || static_cast<int64_t>(values.size()) != rows * width) {
        throw std::runtime_error("FireRedAudio row slice bounds are invalid");
    }
    return std::vector<float>(
        values.begin() + static_cast<std::ptrdiff_t>(begin * width),
        values.begin() + static_cast<std::ptrdiff_t>(end * width));
}

int64_t last_token_position(const std::vector<int32_t> & ids, int32_t token) {
    for (int64_t i = static_cast<int64_t>(ids.size()) - 1; i >= 0; --i) {
        if (ids[static_cast<size_t>(i)] == token) {
            return i;
        }
    }
    return -1;
}

void replace_masked_embeddings(
    std::vector<float> & embeddings,
    const std::vector<uint8_t> & mask,
    const std::vector<float> & replacement,
    int64_t hidden) {
    const int64_t steps = static_cast<int64_t>(mask.size());
    const int64_t rows = std::count(mask.begin(), mask.end(), static_cast<uint8_t>(1));
    if (static_cast<int64_t>(embeddings.size()) != steps * hidden ||
        static_cast<int64_t>(replacement.size()) != rows * hidden) {
        throw std::runtime_error("FireRedAudio prompt embedding replacement size mismatch");
    }
    int64_t source_row = 0;
    for (int64_t row = 0; row < steps; ++row) {
        if (mask[static_cast<size_t>(row)] == 0) {
            continue;
        }
        std::copy(
            replacement.begin() + static_cast<std::ptrdiff_t>(source_row * hidden),
            replacement.begin() + static_cast<std::ptrdiff_t>((source_row + 1) * hidden),
            embeddings.begin() + static_cast<std::ptrdiff_t>(row * hidden));
        ++source_row;
    }
}

int32_t argmax_token(const std::vector<float> & logits, int32_t masked_token = -1) {
    if (logits.empty()) {
        throw std::runtime_error("FireRedAudio empty logits");
    }
    int32_t best = 0;
    float best_value = -std::numeric_limits<float>::infinity();
    for (int32_t i = 0; i < static_cast<int32_t>(logits.size()); ++i) {
        if (i == masked_token) {
            continue;
        }
        const float value = logits[static_cast<size_t>(i)];
        if (value > best_value) {
            best_value = value;
            best = i;
        }
    }
    return best;
}

}  // namespace

class FireRedAudioGenerationRuntime::Impl {
public:
    Impl(
        std::shared_ptr<const FireRedAudioAssets> assets,
        core::ExecutionContext & execution,
        size_t graph_arena_bytes,
        size_t weight_context_bytes,
        assets::TensorStorageType storage_type,
        size_t reference_cache_slots,
        bool mem_saver)
        : assets_(std::move(assets)),
          execution_(execution),
          mem_saver_(mem_saver),
          sampling_policy_(sampling::resolve_torch_cuda_sampling_policy(
              execution.backend_type(),
              execution.config().device,
              "firered_audio",
              "FireRedAudio",
              sampling::TorchCudaSamplingPolicyFailureMode::FallbackToDefault)),
          reference_cache_(reference_cache_slots) {
        if (assets_ == nullptr) {
            throw std::runtime_error("FireRedAudio generation runtime requires assets");
        }
        qwen_ = std::make_unique<FireRedAudioQwen35Runtime>(assets_, execution_, graph_arena_bytes, weight_context_bytes, storage_type);
        patch_encoder_ = std::make_unique<FireRedAudioPatchEncoderRuntime>(assets_, execution_, graph_arena_bytes, weight_context_bytes, storage_type);
        redae_ = std::make_unique<FireRedAudioRedAeRuntime>(assets_, execution_, graph_arena_bytes, weight_context_bytes, storage_type);
        flow_ = std::make_unique<FireRedAudioFlowRuntime>(assets_, execution_, graph_arena_bytes, weight_context_bytes, storage_type);
        tokenizer_ = std::make_unique<FireRedAudioTokenizer>(assets_);
    }

    FireRedAudioGenerationResult generate(const FireRedAudioGenerationRequest & request) {
        const auto total_start = Clock::now();
        auto input_embeddings = qwen_->token_embedding(request.token_ids);
        std::vector<float> history_vae_latents;

        if (request.prompt_audio.has_value()) {
            const auto & ref = prepare_reference_audio(*request.prompt_audio);
            replace_masked_embeddings(
                input_embeddings,
                request.audio_no_latent_mask,
                ref.patch_embeddings,
                assets_->backbone.hidden_size);
            if (request.prompt_audio_is_assistant) {
                history_vae_latents = ref.latents;
            }
        }
        if (mem_saver_) {
            redae_->release_graphs();
            patch_encoder_->release_graph();
        }

        auto generation = generate_latents(request, input_embeddings, history_vae_latents);
        if (generation.latents.empty()) {
            throw std::runtime_error("FireRedAudio generated no audio latents");
        }
        auto decoded = redae_->decode(generation.latents);
        if (mem_saver_) {
            release_graphs();
        }
        debug::timing_log_scalar(
            "firered_audio.total_ms",
            std::chrono::duration<double, std::milli>(Clock::now() - total_start).count());
        FireRedAudioGenerationResult result;
        result.audio = std::move(decoded);
        result.generated_text = tokenizer_->decode(generation.text_ids, true);
        return result;
    }

    void release_graphs() {
        if (qwen_) {
            qwen_->release_graphs();
        }
        if (patch_encoder_) {
            patch_encoder_->release_graph();
        }
        if (redae_) {
            redae_->release_graphs();
        }
        if (flow_) {
            flow_->release_graph();
        }
    }

private:
    struct Generation {
        std::vector<float> latents;
        std::vector<int32_t> text_ids;
    };

    const ReferenceAudioCacheEntry & prepare_reference_audio(const runtime::AudioBuffer & audio) {
        ReferenceAudioCacheKey key;
        key.sample_rate = audio.sample_rate;
        key.channels = audio.channels;
        key.sample_count = static_cast<uint64_t>(audio.samples.size());
        key.sample_hash = hash_audio_samples(audio);
        if (const auto * cached = reference_cache_.find(key)) {
            debug::trace_log_scalar("firered_audio.reference_cache.hit", 1);
            return *cached;
        }
        ReferenceAudioCacheEntry entry;
        entry.audio_24k = prepare_firered_prompt_audio_24k(audio, assets_->redae, assets_->patch_encoder.patch_size);
        entry.latents = redae_->encode(entry.audio_24k);
        entry.patch_embeddings = patch_encoder_->encode(entry.latents);
        reference_cache_.put(key, std::move(entry));
        const auto * cached = reference_cache_.find(key);
        if (cached == nullptr) {
            throw std::runtime_error("FireRedAudio reference cache insert failed");
        }
        debug::trace_log_scalar("firered_audio.reference_cache.hit", 0);
        return *cached;
    }

    Generation generate_latents(
        const FireRedAudioGenerationRequest & request,
        const std::vector<float> & input_embeddings,
        const std::vector<float> & reference_latents) {
        const int64_t steps = static_cast<int64_t>(request.token_ids.size());
        const int64_t decode_budget = std::min(
            request.max_new_text_tokens + request.max_new_audio_steps,
            kDecodeCacheGrowthWindow);
        auto decode = qwen_->create_decode_session(steps + decode_budget + 4);
        auto prefill = decode->prefill_embeddings(input_embeddings, steps);
        std::vector<float> current_h(
            prefill.hidden.end() - static_cast<std::ptrdiff_t>(assets_->backbone.hidden_size),
            prefill.hidden.end());

        const int64_t last_sosp = last_token_position(request.token_ids, assets_->special_tokens.sosp);
        const int64_t last_eosp = last_token_position(request.token_ids, assets_->special_tokens.eosp);
        bool audio_mode = last_sosp > last_eosp;
        std::vector<float> backbone_audio_hiddens;
        if (audio_mode) {
            backbone_audio_hiddens = row_slice(
                prefill.hidden,
                steps,
                assets_->backbone.hidden_size,
                last_sosp,
                steps - 1);
        }

        std::vector<float> history_vae_latents = reference_latents;
        std::vector<float> generated_latents;
        std::vector<int32_t> generated_text_ids;
        sampling::HfSampler sampler;
        sampling::HfSamplerScratch scratch;
        sampling::HfSamplingOptions text_options;
        text_options.do_sample = true;
        text_options.temperature = 0.7F;
        text_options.top_p = 0.8F;
        text_options.top_k = 20;
        text_options.repetition_penalty = 1.0F;
        std::mt19937 fallback(request.seed);
        uint64_t sample_call_index = 0;
        uint64_t rng_offset_blocks = 0;
        int64_t audio_steps = 0;

        while (true) {
            if (!audio_mode) {
                auto logits = qwen_->lm_head(current_h);
                rng_offset_blocks += sampling::torch_cuda_tensor_iterator_offset_blocks(
                    static_cast<uint64_t>(logits.size()),
                    sampling_policy_);
                const sampling::HfTorchSamplingState torch_state{
                    &sampling_policy_,
                    request.seed,
                    sample_call_index++,
                };
                const int32_t token = sampler.sample(
                    logits,
                    generated_text_ids,
                    text_options,
                    scratch,
                    fallback,
                    sampling_policy_.cuda_fast_path ? &torch_state : nullptr,
                    "FireRedAudio text sampler");
                generated_text_ids.push_back(token);
                if (token == assets_->special_tokens.im_end || token == assets_->special_tokens.endoftext) {
                    break;
                }
                auto next_embedding = qwen_->token_embedding({token});
                current_h = decode->run_embedding_step(next_embedding);
                audio_mode = token == assets_->special_tokens.sosp;
                if (!audio_mode && static_cast<int64_t>(generated_text_ids.size()) >= request.max_new_text_tokens) {
                    break;
                }
                continue;
            }

            append_embedding(backbone_audio_hiddens, current_h);
            auto next_latent = flow_one_patch(
                history_vae_latents,
                backbone_audio_hiddens,
                request.guidance_scale,
                firered_cosine_time_schedule(request.num_inference_steps),
                request.seed,
                rng_offset_blocks,
                static_cast<uint64_t>(audio_steps));
            append_embedding(history_vae_latents, next_latent);
            append_embedding(generated_latents, next_latent);
            auto next_embedding = patch_encoder_->encode(next_latent);
            current_h = decode->run_embedding_step(next_embedding);
            ++audio_steps;

            auto exit_logits = qwen_->lm_head(current_h);
            const int32_t next = audio_steps < request.min_new_audio_steps
                ? argmax_token(exit_logits, assets_->special_tokens.eosp)
                : argmax_token(exit_logits);
            if (next == assets_->special_tokens.eosp) {
                generated_text_ids.push_back(next);
                auto eosp_embedding = qwen_->token_embedding({next});
                current_h = decode->run_embedding_step(eosp_embedding);
                backbone_audio_hiddens.clear();
                audio_mode = false;
            } else if (audio_steps >= request.max_new_audio_steps) {
                break;
            }
        }

        Generation out;
        out.latents = std::move(generated_latents);
        out.text_ids = std::move(generated_text_ids);
        return out;
    }

    void append_embedding(std::vector<float> & target, const std::vector<float> & value) {
        target.insert(target.end(), value.begin(), value.end());
    }

    std::vector<float> flow_one_patch(
        const std::vector<float> & history_vae_latents,
        const std::vector<float> & backbone_audio_hiddens,
        float cfg,
        const std::vector<float> & schedule,
        uint32_t seed,
        uint64_t rng_offset_blocks,
        uint64_t step_index) {
        const int64_t history_tokens = assets_->flow.history_patches * assets_->flow.patch_size;
        const int64_t tokens = history_tokens + assets_->flow.patch_size;
        const int64_t input_channels = flow_->input_channels();
        history_scratch_.assign(static_cast<size_t>(history_tokens * assets_->flow.vae_channels), 0.0F);
        const int64_t source_history_rows = static_cast<int64_t>(history_vae_latents.size()) / assets_->flow.vae_channels;
        const int64_t keep_history = std::min(history_tokens, source_history_rows);
        if (keep_history > 0) {
            std::copy(
                history_vae_latents.end() - static_cast<std::ptrdiff_t>(keep_history * assets_->flow.vae_channels),
                history_vae_latents.end(),
                history_scratch_.end() - static_cast<std::ptrdiff_t>(keep_history * assets_->flow.vae_channels));
        }

        cond_scratch_.assign(static_cast<size_t>((assets_->flow.history_patches + 1) * assets_->flow.backbone_hidden_size), 0.0F);
        const int64_t cond_rows = static_cast<int64_t>(backbone_audio_hiddens.size()) / assets_->flow.backbone_hidden_size;
        const int64_t keep_cond = std::min<int64_t>(assets_->flow.history_patches + 1, cond_rows);
        if (keep_cond > 0) {
            std::copy(
                backbone_audio_hiddens.end() - static_cast<std::ptrdiff_t>(keep_cond * assets_->flow.backbone_hidden_size),
                backbone_audio_hiddens.end(),
                cond_scratch_.end() - static_cast<std::ptrdiff_t>(keep_cond * assets_->flow.backbone_hidden_size));
        }

        const uint64_t noise_offset = rng_offset_blocks +
            sampling::torch_cuda_tensor_iterator_offset_blocks(
                static_cast<uint64_t>(assets_->flow.patch_size * assets_->flow.vae_channels),
                sampling_policy_) *
            step_index;
        current_scratch_.resize(static_cast<size_t>(assets_->flow.patch_size * assets_->flow.vae_channels));
        sampling::fill_torch_cuda_tensor_iterator_randn(
            current_scratch_.data(),
            current_scratch_.size(),
            seed,
            noise_offset,
            sampling_policy_,
            sampling::TorchRandnPrecision::Float32);

        for (size_t i = 0; i + 1 < schedule.size(); ++i) {
            const int64_t batch = cfg > 0.0F ? 2 : 1;
            x_in_scratch_.assign(static_cast<size_t>(batch * tokens * input_channels), 0.0F);
            for (int64_t b = 0; b < batch; ++b) {
                for (int64_t t = 0; t < tokens; ++t) {
                    float * row = x_in_scratch_.data() + static_cast<size_t>((b * tokens + t) * input_channels);
                    if (t < history_tokens) {
                        std::copy(
                            history_scratch_.begin() + static_cast<std::ptrdiff_t>(t * assets_->flow.vae_channels),
                            history_scratch_.begin() + static_cast<std::ptrdiff_t>((t + 1) * assets_->flow.vae_channels),
                            row);
                    } else {
                        const int64_t local = t - history_tokens;
                        std::copy(
                            current_scratch_.begin() + static_cast<std::ptrdiff_t>(local * assets_->flow.vae_channels),
                            current_scratch_.begin() + static_cast<std::ptrdiff_t>((local + 1) * assets_->flow.vae_channels),
                            row);
                    }
                    if (b == 0) {
                        const int64_t cond_row = t / assets_->flow.patch_size;
                        std::copy(
                            cond_scratch_.begin() + static_cast<std::ptrdiff_t>(cond_row * assets_->flow.backbone_hidden_size),
                            cond_scratch_.begin() + static_cast<std::ptrdiff_t>((cond_row + 1) * assets_->flow.backbone_hidden_size),
                            row + assets_->flow.vae_channels);
                    }
                }
            }
            auto te = firered_timestep_embedding(schedule[i]);
            time_scratch_.resize(static_cast<size_t>(batch * 256));
            for (int64_t b = 0; b < batch; ++b) {
                std::copy(te.begin(), te.end(), time_scratch_.begin() + static_cast<std::ptrdiff_t>(b * 256));
            }
            auto pred = flow_->run(x_in_scratch_, time_scratch_, batch);
            const float dt = schedule[i + 1] - schedule[i];
            for (int64_t t = 0; t < assets_->flow.patch_size; ++t) {
                for (int64_t c = 0; c < assets_->flow.vae_channels; ++c) {
                    const size_t idx = static_cast<size_t>(t * assets_->flow.vae_channels + c);
                    float vt = pred[idx];
                    if (batch == 2) {
                        const float uncond =
                            pred[static_cast<size_t>((assets_->flow.patch_size + t) * assets_->flow.vae_channels + c)];
                        vt = (1.0F + cfg) * vt - cfg * uncond;
                    }
                    current_scratch_[idx] += dt * vt;
                }
            }
        }
        return current_scratch_;
    }

    std::shared_ptr<const FireRedAudioAssets> assets_;
    core::ExecutionContext & execution_;
    bool mem_saver_ = false;
    sampling::TorchCudaSamplingPolicy sampling_policy_;
    runtime::CacheSlots<ReferenceAudioCacheKey, ReferenceAudioCacheEntry, ReferenceAudioCacheKeyEqual> reference_cache_;
    std::unique_ptr<FireRedAudioQwen35Runtime> qwen_;
    std::unique_ptr<FireRedAudioPatchEncoderRuntime> patch_encoder_;
    std::unique_ptr<FireRedAudioRedAeRuntime> redae_;
    std::unique_ptr<FireRedAudioFlowRuntime> flow_;
    std::unique_ptr<FireRedAudioTokenizer> tokenizer_;
    std::vector<float> history_scratch_;
    std::vector<float> cond_scratch_;
    std::vector<float> current_scratch_;
    std::vector<float> x_in_scratch_;
    std::vector<float> time_scratch_;
};

FireRedAudioGenerationRuntime::FireRedAudioGenerationRuntime(
    std::shared_ptr<const FireRedAudioAssets> assets,
    engine::core::ExecutionContext & execution,
    size_t graph_arena_bytes,
    size_t weight_context_bytes,
    engine::assets::TensorStorageType storage_type,
    size_t reference_cache_slots,
    bool mem_saver)
    : impl_(std::make_unique<Impl>(
          std::move(assets),
          execution,
          graph_arena_bytes,
          weight_context_bytes,
          storage_type,
          reference_cache_slots,
          mem_saver)) {}

FireRedAudioGenerationRuntime::~FireRedAudioGenerationRuntime() = default;

FireRedAudioGenerationResult FireRedAudioGenerationRuntime::generate(const FireRedAudioGenerationRequest & request) {
    return impl_->generate(request);
}

void FireRedAudioGenerationRuntime::release_graphs() {
    impl_->release_graphs();
}

}  // namespace engine::models::firered_audio
