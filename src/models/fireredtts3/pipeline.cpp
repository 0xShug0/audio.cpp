#include "engine/models/fireredtts3/pipeline.h"
#include "engine/models/fireredtts3/ar.h"
#include "engine/models/fireredtts3/flow.h"
#include "engine/models/fireredtts3/redae.h"

#include "engine/framework/audio/conversion.h"
#include "engine/framework/audio/kaldi_fbank.h"
#include "engine/framework/audio/resampling.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/modules/speech_encoders/campplus_encoder.h"
#include "engine/framework/runtime/cache_slots.h"
#include "engine/framework/sampling/hf_sampler.h"
#include "engine/framework/sampling/torch_random.h"
#include "engine/models/fireredtts3/tokenizer_text.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace engine::models::fireredtts3 {
namespace {

namespace core = engine::core;
namespace modules = engine::modules;
using Clock = std::chrono::steady_clock;

constexpr float kRedAeScale = 0.4F;
constexpr int32_t kFireRedTextEotId = 151677;
constexpr int32_t kFireRedAudioSosId = 151669;
uint64_t mix_reference_voice_key(uint64_t key, uint64_t value) {
    key ^= value;
    key *= 1099511628211ull;
    return key;
}

uint64_t hash_audio_samples(const runtime::AudioBuffer & audio) {
    uint64_t key = 1469598103934665603ull;
    for (const float sample : audio.samples) {
        uint32_t bits = 0;
        std::memcpy(&bits, &sample, sizeof(bits));
        key = mix_reference_voice_key(key, bits);
    }
    return key;
}

struct ReferenceVoiceCacheKey {
    int sample_rate = 0;
    int channels = 0;
    uint64_t sample_count = 0;
    uint64_t sample_hash = 0;
};

struct ReferenceVoiceCacheKeyEqual {
    bool operator()(const ReferenceVoiceCacheKey & lhs, const ReferenceVoiceCacheKey & rhs) const noexcept {
        return lhs.sample_rate == rhs.sample_rate &&
            lhs.channels == rhs.channels &&
            lhs.sample_count == rhs.sample_count &&
            lhs.sample_hash == rhs.sample_hash;
    }
};

struct ReferenceVoiceCacheEntry {
    std::vector<float> prompt_audio_24k;
    std::vector<float> prompt_latents;
    std::vector<float> speaker_embedding;
};

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

class FireRedTTS3BaseRuntime::Impl {
public:
    Impl(
        std::shared_ptr<const FireRedTTS3Assets> assets,
        core::ExecutionContext & execution,
        size_t graph_arena_bytes,
        size_t helper_graph_arena_bytes,
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
              "fireredtts3",
              "FireRedTTS3",
              sampling::TorchCudaSamplingPolicyFailureMode::FallbackToDefault)) {
        if (assets_ == nullptr) {
            throw std::runtime_error("FireRedTTS3 base runtime requires assets");
        }
        reference_voice_cache_ =
            runtime::CacheSlots<ReferenceVoiceCacheKey, ReferenceVoiceCacheEntry, ReferenceVoiceCacheKeyEqual>(
                reference_cache_slots);
        storage_type_ = storage_type;
        ar_ = std::make_unique<FireRedArRuntime>(
            assets_,
            execution_,
            graph_arena_bytes,
            helper_graph_arena_bytes,
            weight_context_bytes,
            storage_type,
            false);
        redae_ = std::make_unique<FireRedRedAeRuntime>(assets_, execution_, graph_arena_bytes, weight_context_bytes, storage_type);
        flow_ = std::make_unique<FireRedFlowRuntime>(assets_, execution_, graph_arena_bytes, weight_context_bytes, storage_type, false);
    }

    runtime::AudioBuffer generate(const FireRedTTS3BaseRequest & request) {
        const auto total_start = Clock::now();
        const auto & reference = prepare_reference_voice(request.prompt_audio);
        const auto & prompt_audio = reference.prompt_audio_24k;
        const auto & prompt_latents = reference.prompt_latents;
        const int64_t prompt_latent_frames = static_cast<int64_t>(prompt_latents.size()) / assets_->base.redae_dim;
        auto spk_llm = ar_->speaker_llm(reference.speaker_embedding);
        auto spk_dit = ar_->speaker_dit(reference.speaker_embedding);
        auto text_embeds = ar_->token_embedding(request.token_ids);
        if (mem_saver_) {
            ar_->release_graphs();
        }
        auto patch_prompt = ar_->patch_encode(prompt_latents);

        std::vector<float> input_embeddings;
        input_embeddings.reserve(spk_llm.size() + text_embeds.size() + patch_prompt.size());
        input_embeddings.insert(input_embeddings.end(), spk_llm.begin(), spk_llm.end());
        input_embeddings.insert(input_embeddings.end(), text_embeds.begin(), text_embeds.end());
        input_embeddings.insert(input_embeddings.end(), patch_prompt.begin(), patch_prompt.end());
        const int64_t prefill_steps = 1 + static_cast<int64_t>(request.token_ids.size()) + prompt_latent_frames / assets_->base.patch_size;
        auto prefill = ar_->prefill_embeddings(input_embeddings, prefill_steps);
        ar_->start_decode_embeddings(prefill.state, prefill_steps + 400 + 1);

        std::vector<float> latents_gen(static_cast<size_t>(assets_->base.history_patches * assets_->base.patch_size * assets_->base.redae_dim), 0.0F);
        latents_gen.insert(latents_gen.end(), prompt_latents.begin(), prompt_latents.end());
        std::vector<float> backbone_cond(static_cast<size_t>(assets_->base.history_patches * assets_->base.hidden_size), 0.0F);
        auto schedule = firered_cosine_time_schedule(request.num_inference_steps);
        auto next_input = std::vector<float>();
        const int64_t max_steps = 400;
        int64_t generated_patches = 0;

        for (int64_t step = 0; step < max_steps; ++step) {
            std::vector<float> hidden;
            int64_t hidden_rows = 0;
            if (step == 0) {
                hidden = std::move(prefill.hidden);
                hidden_rows = prefill_steps;
            } else {
                auto decoded = ar_->decode_embedding(next_input);
                hidden = decoded.hidden;
                hidden_rows = 1;
            }
            const auto last = last_rows(hidden, hidden_rows, assets_->base.hidden_size, 1);
            const float stop = ar_->stop(last);
            if (stop >= request.stop_threshold && step >= 6) {
                break;
            }

            std::vector<float> one_backbone;
            if (step == 0) {
                const int64_t prompt_patches = prompt_latent_frames / assets_->base.patch_size;
                one_backbone = last_rows(hidden, hidden_rows, assets_->base.hidden_size, prompt_patches);
            } else {
                one_backbone = last;
            }
            backbone_cond.insert(backbone_cond.end(), one_backbone.begin(), one_backbone.end());
            const int64_t cond_rows = static_cast<int64_t>(backbone_cond.size()) / assets_->base.hidden_size;
            auto cond3 = last_rows(
                backbone_cond,
                cond_rows,
                assets_->base.hidden_size,
                assets_->base.history_patches + 1);
            auto dit_cond3 = ar_->dit_head(cond3, assets_->base.history_patches + 1);
            const auto next_latent = flow_one_patch(
                latents_gen,
                dit_cond3,
                spk_dit,
                request.guidance_scale,
                schedule,
                request.seed,
                static_cast<uint64_t>(step));
            latents_gen.insert(latents_gen.end(), next_latent.begin(), next_latent.end());
            next_input = ar_->patch_encode(next_latent);
            ++generated_patches;
        }

        if (generated_patches <= 0) {
            throw std::runtime_error("FireRedTTS3 generated no latent patches");
        }
        latents_gen.erase(
            latents_gen.begin(),
            latents_gen.begin() + static_cast<std::ptrdiff_t>(assets_->base.history_patches * assets_->base.patch_size * assets_->base.redae_dim));
        if (mem_saver_) {
            ar_->release_graphs();
            flow_->release_graph();
            ar_->release_backbone_graphs();
        }
        auto decoded = redae_->decode(latents_gen);
        if (mem_saver_) {
            redae_->release_graphs();
        }
        if (static_cast<int64_t>(decoded.samples.size()) > static_cast<int64_t>(prompt_audio.size())) {
            decoded.samples.erase(
                decoded.samples.begin(),
                decoded.samples.begin() + static_cast<std::ptrdiff_t>(prompt_audio.size()));
        } else {
            decoded.samples.clear();
        }
        debug::timing_log_scalar(
            "fireredtts3.total_ms",
            std::chrono::duration<double, std::milli>(Clock::now() - total_start).count());
        return decoded;
    }

    void release_graphs() {
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
    const ReferenceVoiceCacheEntry & prepare_reference_voice(const runtime::AudioBuffer & audio) {
        ReferenceVoiceCacheKey key;
        key.sample_rate = audio.sample_rate;
        key.channels = audio.channels;
        key.sample_count = static_cast<uint64_t>(audio.samples.size());
        key.sample_hash = hash_audio_samples(audio);

        const ReferenceVoiceCacheEntry * cached = reference_voice_cache_.find(key);
        if (cached != nullptr) {
            debug::trace_log_scalar("fireredtts3.reference_voice_cache.hit", 1);
            return *cached;
        }

        ReferenceVoiceCacheEntry entry;
        entry.prompt_audio_24k = prepare_firered_prompt_audio_24k(audio, assets_->redae, assets_->base.patch_size);
        entry.prompt_latents = redae_->encode(entry.prompt_audio_24k);
        if (mem_saver_) {
            redae_->release_graphs();
        }
        ensure_campplus_loaded();
        auto speaker_features = campplus_fbank(entry.prompt_audio_24k);
        auto speaker = campplus_.embed_from_features(
            speaker_features,
            static_cast<int64_t>(speaker_features.size()) / 80,
            80);
        entry.speaker_embedding = std::move(speaker.embedding);
        if (mem_saver_) {
            campplus_ = modules::CampplusEncoderComponent();
        }

        reference_voice_cache_.put(key, std::move(entry));
        cached = reference_voice_cache_.find(key);
        if (cached == nullptr) {
            throw std::runtime_error("FireRedTTS3 reference voice cache insert failed");
        }
        debug::trace_log_scalar("fireredtts3.reference_voice_cache.hit", 0);
        return *cached;
    }

    void ensure_campplus_loaded() {
        if (campplus_.weights() != nullptr) {
            return;
        }
        modules::CampplusEncoderConfig campplus_config;
        campplus_config.feat_dim = 80;
        campplus_config.embedding_size = assets_->base.speaker_dim;
        campplus_config.tensor_prefix = "campplus";
        campplus_config.weight_storage_type = storage_type_;
        campplus_ = modules::CampplusEncoderComponent::load_from_tensor_source(
            assets_->campplus_weights,
            execution_.config(),
            std::move(campplus_config));
    }

    std::vector<float> flow_one_patch(
        const std::vector<float> & latents_gen,
        const std::vector<float> & dit_cond3,
        const std::vector<float> & speaker_dit,
        float cfg,
        const std::vector<float> & schedule,
        uint32_t seed,
        uint64_t step_index) {
        const int64_t history_tokens = assets_->base.history_patches * assets_->base.patch_size;
        const int64_t tokens = history_tokens + assets_->base.patch_size;
        const int64_t input_channels = assets_->base.redae_dim + assets_->base.dit_hidden_size + assets_->base.speaker_dim;
        auto history = last_rows(
            latents_gen,
            static_cast<int64_t>(latents_gen.size()) / assets_->base.redae_dim,
            assets_->base.redae_dim,
            history_tokens);
        const uint64_t noise_offset =
            sampling::torch_cuda_tensor_iterator_offset_blocks(
                static_cast<uint64_t>(assets_->base.patch_size * assets_->base.redae_dim),
                sampling_policy_) *
            step_index;
        auto current = sampling::generate_torch_cuda_tensor_iterator_randn(
            static_cast<size_t>(assets_->base.patch_size * assets_->base.redae_dim),
            seed,
            noise_offset,
            sampling_policy_,
            sampling::TorchRandnPrecision::Float32);

        for (size_t i = 0; i + 1 < schedule.size(); ++i) {
            const int64_t batch = cfg > 0.0F ? 2 : 1;
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
                        std::copy(
                            speaker_dit.begin(),
                            speaker_dit.end(),
                            row + assets_->base.redae_dim + assets_->base.dit_hidden_size);
                    }
                }
            }
            auto te = firered_timestep_embedding(schedule[i]);
            std::vector<float> time(static_cast<size_t>(batch * 256));
            for (int64_t b = 0; b < batch; ++b) {
                std::copy(te.begin(), te.end(), time.begin() + static_cast<std::ptrdiff_t>(b * 256));
            }
            auto pred = flow_->run(x_in, time, batch);
            const float dt = schedule[i + 1] - schedule[i];
            for (int64_t t = 0; t < assets_->base.patch_size; ++t) {
                for (int64_t c = 0; c < assets_->base.redae_dim; ++c) {
                    const size_t idx = static_cast<size_t>(t * assets_->base.redae_dim + c);
                    float vt = pred[idx];
                    if (batch == 2) {
                        const float uncond = pred[static_cast<size_t>((assets_->base.patch_size + t) * assets_->base.redae_dim + c)];
                        vt = (1.0F + cfg) * vt - cfg * uncond;
                    }
                    current[idx] += dt * vt;
                }
            }
        }
        return current;
    }

    std::shared_ptr<const FireRedTTS3Assets> assets_;
    core::ExecutionContext & execution_;
    bool mem_saver_ = false;
    sampling::TorchCudaSamplingPolicy sampling_policy_;
    assets::TensorStorageType storage_type_ = assets::TensorStorageType::Native;
    modules::CampplusEncoderComponent campplus_;
    runtime::CacheSlots<ReferenceVoiceCacheKey, ReferenceVoiceCacheEntry, ReferenceVoiceCacheKeyEqual> reference_voice_cache_;
    std::unique_ptr<FireRedArRuntime> ar_;
    std::unique_ptr<FireRedRedAeRuntime> redae_;
    std::unique_ptr<FireRedFlowRuntime> flow_;
};

void apply_masked_rows(
    std::vector<float> & embeddings,
    int64_t steps,
    int64_t hidden,
    const std::vector<uint8_t> & mask,
    const std::vector<float> & replacement) {
    const int64_t masked_rows = std::count(mask.begin(), mask.end(), static_cast<uint8_t>(1));
    if (static_cast<int64_t>(mask.size()) != steps || static_cast<int64_t>(embeddings.size()) != steps * hidden ||
        static_cast<int64_t>(replacement.size()) != masked_rows * hidden) {
        throw std::runtime_error("FireRedTTS3 Instruct masked embedding replacement size mismatch");
    }
    int64_t row = 0;
    for (int64_t step = 0; step < steps; ++step) {
        if (mask[static_cast<size_t>(step)] == 0) {
            continue;
        }
        std::copy(
            replacement.begin() + static_cast<std::ptrdiff_t>(row * hidden),
            replacement.begin() + static_cast<std::ptrdiff_t>((row + 1) * hidden),
            embeddings.begin() + static_cast<std::ptrdiff_t>(step * hidden));
        ++row;
    }
}

class FireRedTTS3InstructRuntime::Impl {
public:
    Impl(
        std::shared_ptr<const FireRedTTS3Assets> assets,
        core::ExecutionContext & execution,
        size_t graph_arena_bytes,
        size_t helper_graph_arena_bytes,
        size_t weight_context_bytes,
        assets::TensorStorageType storage_type,
        bool mem_saver)
        : assets_(std::move(assets)),
          execution_(execution),
          tokenizer_(assets_),
          mem_saver_(mem_saver),
          sampling_policy_(sampling::resolve_torch_cuda_sampling_policy(
              execution.backend_type(),
              execution.config().device,
              "fireredtts3",
              "FireRedTTS3",
              sampling::TorchCudaSamplingPolicyFailureMode::FallbackToDefault)) {
        if (assets_ == nullptr) {
            throw std::runtime_error("FireRedTTS3 instruct runtime requires assets");
        }
        ar_ = std::make_unique<FireRedArRuntime>(
            assets_,
            execution_,
            graph_arena_bytes,
            helper_graph_arena_bytes,
            weight_context_bytes,
            storage_type,
            true);
        redae_ = std::make_unique<FireRedRedAeRuntime>(assets_, execution_, graph_arena_bytes, weight_context_bytes, storage_type);
        flow_ = std::make_unique<FireRedFlowRuntime>(assets_, execution_, graph_arena_bytes, weight_context_bytes, storage_type, true);
    }

    FireRedTTS3InstructResult generate(const FireRedTTS3InstructRequest & request) {
        const auto total_start = Clock::now();
        std::vector<float> latents_in;
        std::vector<float> latents_out;
        runtime::AudioBuffer prompt_audio_24k;
        prompt_audio_24k.sample_rate = assets_->redae.sample_rate;
        prompt_audio_24k.channels = 1;

        FireRedTTS3InstructTokens tokens;
        if (request.task == FireRedTTS3InstructTask::Clone) {
            if (!request.prompt_audio.has_value()) {
                throw std::runtime_error("FireRedTTS3 Instruct clone requires prompt audio");
            }
            prompt_audio_24k.samples = prepare_firered_prompt_audio_24k(*request.prompt_audio, assets_->redae, assets_->base.patch_size);
            latents_out = redae_->encode(prompt_audio_24k.samples);
            for (float & value : latents_out) {
                value *= kRedAeScale;
            }
            const int64_t patches = static_cast<int64_t>(latents_out.size()) /
                (assets_->base.patch_size * assets_->base.redae_dim);
            tokens = tokenizer_.encode_instruct_clone_prompt(patches, request.instruction, request.text);
        } else if (request.task == FireRedTTS3InstructTask::VoiceDesign) {
            tokens = tokenizer_.encode_voice_design_prompt(request.instruction, request.text);
        } else {
            if (!request.input_audio.has_value()) {
                throw std::runtime_error("FireRedTTS3 Instruct edit requires input audio");
            }
            prompt_audio_24k.samples = prepare_firered_prompt_audio_24k(*request.input_audio, assets_->redae, assets_->base.patch_size);
            latents_in = redae_->encode(prompt_audio_24k.samples);
            for (float & value : latents_in) {
                value *= kRedAeScale;
            }
            const int64_t patches = static_cast<int64_t>(latents_in.size()) /
                (assets_->base.patch_size * assets_->base.redae_dim);
            if (request.task == FireRedTTS3InstructTask::SemanticEdit) {
                tokens = tokenizer_.encode_semantic_edit_prompt(patches, request.instruction);
            } else {
                tokens = tokenizer_.encode_acoustic_edit_prompt(patches, request.instruction);
            }
        }
        if (mem_saver_ && (!latents_in.empty() || !latents_out.empty())) {
            redae_->release_graphs();
        }

        auto generation = generate_latents(request, tokens, latents_in, latents_out);
        for (float & value : generation.latents) {
            value *= 1.0F / kRedAeScale;
        }
        if (mem_saver_) {
            ar_->release_graphs();
            flow_->release_graph();
            ar_->release_backbone_graphs();
        }
        auto decoded = redae_->decode(generation.latents);
        if (mem_saver_) {
            redae_->release_graphs();
        }
        if (request.task == FireRedTTS3InstructTask::Clone) {
            const int64_t remove = static_cast<int64_t>(prompt_audio_24k.samples.size());
            if (static_cast<int64_t>(decoded.samples.size()) > remove) {
                decoded.samples.erase(decoded.samples.begin(), decoded.samples.begin() + static_cast<std::ptrdiff_t>(remove));
            } else {
                decoded.samples.clear();
            }
        }
        debug::timing_log_scalar(
            "fireredtts3.instruct.total_ms",
            std::chrono::duration<double, std::milli>(Clock::now() - total_start).count());
        FireRedTTS3InstructResult result;
        result.audio = std::move(decoded);
        if (!generation.text_ids.empty()) {
            result.generated_text = tokenizer_.decode(generation.text_ids);
        }
        return result;
    }

    void release_graphs() {
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
    }

private:
    struct InstructGeneration {
        std::vector<float> latents;
        std::vector<int32_t> text_ids;
    };

    InstructGeneration generate_latents(
        const FireRedTTS3InstructRequest & request,
        const FireRedTTS3InstructTokens & tokens,
        const std::vector<float> & latents_in,
        const std::vector<float> & latents_out) {
        auto input_embeddings = ar_->token_embedding(tokens.token_ids);
        const int64_t steps = static_cast<int64_t>(tokens.token_ids.size());
        if (!latents_in.empty()) {
            apply_masked_rows(
                input_embeddings,
                steps,
                assets_->base.hidden_size,
                tokens.latent_in_mask,
                ar_->patch_encode(latents_in));
        }
        bool has_latents_out = !latents_out.empty();
        int64_t latents_out_patches = 0;
        if (has_latents_out) {
            const auto latents_out_patch = ar_->patch_encode(latents_out);
            latents_out_patches = static_cast<int64_t>(latents_out_patch.size()) / assets_->base.hidden_size;
            apply_masked_rows(
                input_embeddings,
                steps,
                assets_->base.hidden_size,
                tokens.latent_out_mask,
                latents_out_patch);
        }

        int64_t cache_steps = 0;
        std::vector<float> next_input;
        std::vector<int32_t> generated_text_ids;
        int32_t last_text_token = kFireRedTextEotId;
        std::vector<float> prefill_hidden;
        int64_t prefill_steps = 0;
        uint64_t rng_offset_blocks = 0;
        if (request.infer_text) {
            auto prefill = ar_->prefill_embeddings(input_embeddings, steps);
            cache_steps = steps;
            ar_->start_decode_embeddings(prefill.state, steps + 200 + 400 + 4);
            auto last_hidden = last_rows(prefill.hidden, steps, assets_->base.hidden_size, 1);
            generated_text_ids = infer_text_tokens(request, last_hidden, cache_steps, rng_offset_blocks, last_text_token);
            auto eot_embed = ar_->token_embedding({last_text_token});
            auto eot_out = ar_->decode_embedding(eot_embed);
            ++cache_steps;
            (void)eot_out;
            auto sos_embed = ar_->token_embedding({kFireRedAudioSosId});
            auto sos_out = ar_->decode_embedding(sos_embed);
            ++cache_steps;
            (void)sos_out;
            next_input = std::move(sos_embed);
        } else {
            auto prefill = ar_->prefill_embeddings(input_embeddings, steps);
            ar_->start_decode_embeddings(prefill.state, steps + 400 + 1);
            cache_steps = steps;
            prefill_hidden = std::move(prefill.hidden);
            prefill_steps = steps;
        }

        std::vector<float> latents_gen(
            static_cast<size_t>(assets_->base.history_patches * assets_->base.patch_size * assets_->base.redae_dim),
            0.0F);
        latents_gen.insert(latents_gen.end(), latents_out.begin(), latents_out.end());
        std::vector<float> backbone_cond(static_cast<size_t>(assets_->base.history_patches * assets_->base.hidden_size), 0.0F);
        auto schedule = firered_cosine_time_schedule(request.num_inference_steps);
        int64_t generated_patches = 0;

        for (int64_t step = 0; step < 400; ++step) {
            std::vector<float> hidden;
            int64_t hidden_rows = 1;
            if (request.infer_text || step > 0) {
                auto decoded = ar_->decode_embedding(next_input);
                hidden = std::move(decoded.hidden);
                ++cache_steps;
            } else {
                hidden = std::move(prefill_hidden);
                hidden_rows = prefill_steps;
            }
            const auto last = last_rows(hidden, hidden_rows, assets_->base.hidden_size, 1);
            const float stop = ar_->stop(last);
            if (stop >= request.stop_threshold && step >= 6) {
                break;
            }

            std::vector<float> one_backbone;
            if (!request.infer_text && step == 0 && has_latents_out) {
                one_backbone = last_rows(hidden, hidden_rows, assets_->base.hidden_size, latents_out_patches);
            } else {
                one_backbone = last;
            }
            backbone_cond.insert(backbone_cond.end(), one_backbone.begin(), one_backbone.end());
            const int64_t cond_rows = static_cast<int64_t>(backbone_cond.size()) / assets_->base.hidden_size;
            auto cond3 = last_rows(
                backbone_cond,
                cond_rows,
                assets_->base.hidden_size,
                assets_->base.history_patches + 1);
            auto dit_cond3 = ar_->dit_head(cond3, assets_->base.history_patches + 1);
            const auto next_latent = flow_one_patch(
                latents_gen,
                dit_cond3,
                request.guidance_scale,
                schedule,
                request.seed,
                rng_offset_blocks,
                static_cast<uint64_t>(step));
            latents_gen.insert(latents_gen.end(), next_latent.begin(), next_latent.end());
            next_input = ar_->patch_encode(next_latent);
            ++generated_patches;
        }

        if (generated_patches <= 0) {
            throw std::runtime_error("FireRedTTS3 Instruct generated no latent patches");
        }
        latents_gen.erase(
            latents_gen.begin(),
            latents_gen.begin() + static_cast<std::ptrdiff_t>(assets_->base.history_patches * assets_->base.patch_size * assets_->base.redae_dim));
        InstructGeneration out;
        out.latents = std::move(latents_gen);
        out.text_ids = std::move(generated_text_ids);
        return out;
    }

    std::vector<int32_t> infer_text_tokens(
        const FireRedTTS3InstructRequest & request,
        std::vector<float> last_hidden,
        int64_t & cache_steps,
        uint64_t & rng_offset_blocks,
        int32_t & last_text_token) {
        sampling::HfSampler sampler;
        sampling::HfSamplerScratch scratch;
        sampling::HfSamplingOptions options;
        options.do_sample = request.text_do_sample;
        options.temperature = request.text_do_sample ? 0.7F : 1.0F;
        options.top_p = request.text_do_sample ? 0.8F : 1.0F;
        options.top_k = request.text_do_sample ? 20 : 0;
        options.repetition_penalty = 1.0F;
        std::mt19937 fallback(request.seed);
        sampling::HfTorchSamplingState torch_state;
        torch_state.policy = &sampling_policy_;
        torch_state.seed = request.seed;
        std::vector<int32_t> text_ids;
        int32_t next = kFireRedTextEotId;
        for (int64_t i = 0; i < 200; ++i) {
            auto logits = ar_->text_logits(last_hidden);
            if (request.text_do_sample) {
                rng_offset_blocks += sampling::torch_cuda_tensor_iterator_offset_blocks(
                    static_cast<uint64_t>(logits.size()),
                    sampling_policy_);
            }
            next = sampler.sample(
                logits,
                text_ids,
                options,
                scratch,
                fallback,
                request.text_do_sample ? &torch_state : nullptr,
                "FireRedTTS3 text sampler");
            ++torch_state.call_index;
            last_text_token = next;
            auto next_embed = ar_->token_embedding({next});
            if (next == kFireRedTextEotId) {
                break;
            }
            text_ids.push_back(next);
            auto decoded = ar_->decode_embedding(next_embed);
            ++cache_steps;
            last_hidden = std::move(decoded.hidden);
        }
        return text_ids;
    }

    std::vector<float> flow_one_patch(
        const std::vector<float> & latents_gen,
        const std::vector<float> & dit_cond3,
        float cfg,
        const std::vector<float> & schedule,
        uint32_t seed,
        uint64_t rng_offset_blocks,
        uint64_t step_index) {
        const int64_t history_tokens = assets_->base.history_patches * assets_->base.patch_size;
        const int64_t tokens = history_tokens + assets_->base.patch_size;
        const int64_t input_channels = assets_->base.redae_dim + assets_->base.dit_hidden_size;
        auto history = last_rows(
            latents_gen,
            static_cast<int64_t>(latents_gen.size()) / assets_->base.redae_dim,
            assets_->base.redae_dim,
            history_tokens);
        const uint64_t noise_offset = rng_offset_blocks +
            sampling::torch_cuda_tensor_iterator_offset_blocks(
                static_cast<uint64_t>(assets_->base.patch_size * assets_->base.redae_dim),
                sampling_policy_) *
            step_index;
        auto current = sampling::generate_torch_cuda_tensor_iterator_randn(
            static_cast<size_t>(assets_->base.patch_size * assets_->base.redae_dim),
            seed,
            noise_offset,
            sampling_policy_,
            sampling::TorchRandnPrecision::Float32);

        for (size_t i = 0; i + 1 < schedule.size(); ++i) {
            const int64_t batch = cfg > 0.0F ? 2 : 1;
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
                    }
                }
            }
            auto te = firered_timestep_embedding(schedule[i]);
            std::vector<float> time(static_cast<size_t>(batch * 256));
            for (int64_t b = 0; b < batch; ++b) {
                std::copy(te.begin(), te.end(), time.begin() + static_cast<std::ptrdiff_t>(b * 256));
            }
            auto pred = flow_->run(x_in, time, batch);
            const float dt = schedule[i + 1] - schedule[i];
            for (int64_t t = 0; t < assets_->base.patch_size; ++t) {
                for (int64_t c = 0; c < assets_->base.redae_dim; ++c) {
                    const size_t idx = static_cast<size_t>(t * assets_->base.redae_dim + c);
                    float vt = pred[idx];
                    if (batch == 2) {
                        const float uncond = pred[static_cast<size_t>((assets_->base.patch_size + t) * assets_->base.redae_dim + c)];
                        vt = (1.0F + cfg) * vt - cfg * uncond;
                    }
                    current[idx] += dt * vt;
                }
            }
        }
        return current;
    }

    std::shared_ptr<const FireRedTTS3Assets> assets_;
    core::ExecutionContext & execution_;
    FireRedTTS3TextTokenizer tokenizer_;
    bool mem_saver_ = false;
    sampling::TorchCudaSamplingPolicy sampling_policy_;
    std::unique_ptr<FireRedArRuntime> ar_;
    std::unique_ptr<FireRedRedAeRuntime> redae_;
    std::unique_ptr<FireRedFlowRuntime> flow_;
};

FireRedTTS3BaseRuntime::FireRedTTS3BaseRuntime(
    std::shared_ptr<const FireRedTTS3Assets> assets,
    engine::core::ExecutionContext & execution,
    size_t graph_arena_bytes,
    size_t helper_graph_arena_bytes,
    size_t weight_context_bytes,
    engine::assets::TensorStorageType storage_type,
    size_t reference_cache_slots,
    bool mem_saver)
    : impl_(std::make_unique<Impl>(
          std::move(assets),
          execution,
          graph_arena_bytes,
          helper_graph_arena_bytes,
          weight_context_bytes,
          storage_type,
          reference_cache_slots,
          mem_saver)) {}

FireRedTTS3BaseRuntime::~FireRedTTS3BaseRuntime() = default;

engine::runtime::AudioBuffer FireRedTTS3BaseRuntime::generate(const FireRedTTS3BaseRequest & request) {
    return impl_->generate(request);
}

void FireRedTTS3BaseRuntime::release_graphs() {
    impl_->release_graphs();
}

FireRedTTS3InstructRuntime::FireRedTTS3InstructRuntime(
    std::shared_ptr<const FireRedTTS3Assets> assets,
    engine::core::ExecutionContext & execution,
    size_t graph_arena_bytes,
    size_t helper_graph_arena_bytes,
    size_t weight_context_bytes,
    engine::assets::TensorStorageType storage_type,
    bool mem_saver)
    : impl_(std::make_unique<Impl>(
          std::move(assets),
          execution,
          graph_arena_bytes,
          helper_graph_arena_bytes,
          weight_context_bytes,
          storage_type,
          mem_saver)) {}

FireRedTTS3InstructRuntime::~FireRedTTS3InstructRuntime() = default;

FireRedTTS3InstructResult FireRedTTS3InstructRuntime::generate(const FireRedTTS3InstructRequest & request) {
    return impl_->generate(request);
}

void FireRedTTS3InstructRuntime::release_graphs() {
    impl_->release_graphs();
}

}  // namespace engine::models::fireredtts3
