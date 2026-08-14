#include "engine/community_models/minimax_music3/pipeline.h"

#include "engine/community_models/minimax_music3/condition_encoder.h"
#include "engine/community_models/minimax_music3/depth_decoder.h"
#include "engine/community_models/minimax_music3/dit.h"
#include "engine/community_models/minimax_music3/lm.h"
#include "engine/community_models/minimax_music3/tokenizer_text.h"
#include "engine/community_models/minimax_music3/types.h"
#include "engine/community_models/minimax_music3/vocoder.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/sampling/noise.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <functional>
#include <cstdint>
#include <limits>
#include <random>
#include <stdexcept>
#include <utility>
#include <vector>

namespace engine::models::minimax_music3 {
namespace {

using Clock = std::chrono::steady_clock;
using Contract = MiniMaxMusic3Contract;

// Reference top-k sampling: keep the top_k largest logits, softmax, and draw one sample.
int32_t sample_top_k(const std::vector<float> & logits, int top_k, std::mt19937 & rng) {
    const size_t size = logits.size();
    const size_t keep = std::min<size_t>(static_cast<size_t>(top_k), size);
    std::vector<int32_t> order(size);
    for (size_t index = 0; index < size; ++index) {
        order[index] = static_cast<int32_t>(index);
    }
    std::partial_sort(
        order.begin(),
        order.begin() + static_cast<int64_t>(keep),
        order.end(),
        [&](int32_t a, int32_t b) { return logits[static_cast<size_t>(a)] > logits[static_cast<size_t>(b)]; });
    double max_logit = -std::numeric_limits<double>::infinity();
    for (size_t rank = 0; rank < keep; ++rank) {
        max_logit = std::max(max_logit, static_cast<double>(logits[static_cast<size_t>(order[rank])]));
    }
    if (!std::isfinite(max_logit)) {
        throw std::runtime_error("MiniMax-Music3 sampling has no finite logits");
    }
    std::vector<double> probabilities(keep);
    double total = 0.0;
    for (size_t rank = 0; rank < keep; ++rank) {
        const double value = static_cast<double>(logits[static_cast<size_t>(order[rank])]);
        probabilities[rank] = std::isfinite(value) ? std::exp(value - max_logit) : 0.0;
        total += probabilities[rank];
    }
    std::uniform_real_distribution<double> uniform(0.0, total);
    double draw = uniform(rng);
    for (size_t rank = 0; rank < keep; ++rank) {
        draw -= probabilities[rank];
        if (draw <= 0.0) {
            return order[rank];
        }
    }
    return order[keep - 1];
}

}  // namespace

struct MiniMaxMusic3PipelineRuntime::Impl {
    size_t weight_context_bytes = 0;
    bool mem_saver = true;
};

MiniMaxMusic3PipelineRuntime::MiniMaxMusic3PipelineRuntime(
    engine::core::ExecutionContext & execution,
    std::shared_ptr<const MiniMaxMusic3Assets> assets,
    size_t weight_context_bytes,
    bool mem_saver)
    : execution_(execution),
      assets_(std::move(assets)),
      impl_(std::make_unique<Impl>()) {
    if (assets_ == nullptr) {
        throw std::runtime_error("MiniMax-Music3 pipeline requires assets");
    }
    impl_->weight_context_bytes = weight_context_bytes;
    impl_->mem_saver = mem_saver;
}

MiniMaxMusic3PipelineRuntime::~MiniMaxMusic3PipelineRuntime() = default;

MiniMaxMusic3GenerateResult MiniMaxMusic3PipelineRuntime::generate(const MiniMaxMusic3GenerateRequest & request) {
    const auto & config = assets_->config;
    const int64_t hidden = config.lm_hidden;
    const int64_t frame_width = config.cond_layers * hidden;
    auto wall_start = Clock::now();

    // Stage 1: prompt.
    MiniMaxMusic3TextTokenizer tokenizer(assets_->resources);
    const auto prompt = tokenizer.encode_prompt(request.caption, request.lyrics);

    const int64_t max_frames = std::min<int64_t>(
        static_cast<int64_t>(request.audio_duration * Contract::kFrameRate),
        Contract::kMaxAudioFrames);
    if (max_frames <= 0) {
        throw std::runtime_error("MiniMax-Music3 audio_duration is shorter than one audio frame");
    }
    std::mt19937 rng(request.seed);

    // Stage 2: autoregressive semantic and residual codes.
    std::vector<float> frame_hiddens;
    frame_hiddens.reserve(static_cast<size_t>(std::min<int64_t>(max_frames, 2048) * frame_width));
    int64_t frames = 0;
    {
        MiniMaxMusic3LmRuntime lm(
            execution_, assets_->lm_weights, config, impl_->weight_context_bytes);
        MiniMaxMusic3DepthDecoderRuntime depth(
            execution_,
            assets_->depth_decoder_weights,
            lm.token_embedding(),
            config,
            impl_->weight_context_bytes);
        const int64_t required_cache_steps =
            static_cast<int64_t>(prompt.cond_ids.size()) + max_frames + 2;
        auto step = lm.prefill(prompt.cond_ids, prompt.uncond_ids, required_cache_steps);

        // Gumbel(0, 1) noise drives the depth decoder's on-device top-k sampling.
        const size_t gumbel_size =
            static_cast<size_t>((config.depth_codebooks - 1) * config.depth_audio_vocab);
        std::vector<float> gumbel(gumbel_size);
        std::uniform_real_distribution<float> uniform(1.0e-20F, 1.0F);
        const auto refill_gumbel = [&]() {
            for (float & value : gumbel) {
                value = -std::log(-std::log(uniform(rng)));
            }
        };

        const auto ar_start = Clock::now();
        double lm_ms = 0.0;
        double depth_ms = 0.0;
        // The first decode step only advances the state past `<|audio_start|>` and is not
        // an emitted frame.
        for (int64_t frame_index = 0; frame_index <= max_frames; ++frame_index) {
            // CFG over the sliced logits, restricted to the conditional branch's top-k.
            const auto & cond = step.cond_logits;
            const auto & uncond = step.uncond_logits;
            std::vector<float> guided(cond.size());
            for (size_t index = 0; index < cond.size(); ++index) {
                guided[index] = uncond[index] + Contract::kArCfgScale * (cond[index] - uncond[index]);
            }
            std::vector<float> cond_sorted(cond);
            const size_t keep = std::min<size_t>(static_cast<size_t>(Contract::kArCfgTopK), cond.size());
            std::nth_element(
                cond_sorted.begin(),
                cond_sorted.begin() + static_cast<int64_t>(keep) - 1,
                cond_sorted.end(),
                std::greater<float>());
            const float threshold = cond_sorted[keep - 1];
            for (size_t index = 0; index < guided.size(); ++index) {
                if (cond[index] < threshold) {
                    guided[index] = -std::numeric_limits<float>::infinity();
                }
            }
            const int32_t sampled = sample_top_k(guided, Contract::kArSamplingTopK, rng);
            if (sampled == 0) {
                break;  // sliced row 0 is the audio end token
            }
            const int32_t semantic_code = sampled - 1;

            const auto depth_start = Clock::now();
            refill_gumbel();
            auto frame = depth.decode_frame(step.last_hidden, semantic_code, gumbel);
            depth_ms += engine::debug::elapsed_ms(depth_start, Clock::now());
            if (frame_index > 0) {
                frame_hiddens.insert(
                    frame_hiddens.end(),
                    step.last_hidden.begin(),
                    step.last_hidden.begin() + hidden);
                frame_hiddens.insert(
                    frame_hiddens.end(),
                    frame.depth_hidden.begin(),
                    frame.depth_hidden.end());
                ++frames;
                if (frames >= max_frames) {
                    break;
                }
            }
            const auto lm_start = Clock::now();
            step = lm.decode_embedding(frame.feedback_embedding);
            lm_ms += engine::debug::elapsed_ms(lm_start, Clock::now());
        }
        engine::debug::timing_log_scalar("minimax_music3.ar_lm_decode_ms", lm_ms);
        engine::debug::timing_log_scalar("minimax_music3.ar_depth_ms", depth_ms);
        engine::debug::timing_log_scalar(
            "minimax_music3.ar_ms", engine::debug::elapsed_ms(ar_start, Clock::now()));
    }
    if (frames == 0) {
        throw std::runtime_error("MiniMax-Music3 generated zero audio frames; the prompt ended generation immediately");
    }

    // Stage 3: chunked flow matching.
    std::vector<int64_t> chunk_starts;
    if (frames <= Contract::kChunkFrames) {
        chunk_starts.push_back(0);
    } else {
        for (int64_t start = 0; start < frames - Contract::kChunkHop; start += Contract::kChunkHop) {
            chunk_starts.push_back(start);
        }
    }

    std::vector<std::vector<float>> latent_chunks;
    std::vector<int64_t> latent_lengths;
    const int64_t latent_channels = config.dit_in_channels;
    {
        MiniMaxMusic3ConditionEncoderRuntime condition_encoder(
            execution_, assets_->condition_encoder_weights, config, impl_->weight_context_bytes);
        MiniMaxMusic3DitRuntime dit(
            execution_, assets_->dit_weights, config, impl_->weight_context_bytes);

        const auto flow_start = Clock::now();
        std::vector<float> previous_latent;    // [latent_channels, overlap]
        std::vector<float> previous_condition; // [cond_out_dim, overlap]
        int64_t previous_overlap = 0;
        const int64_t steps = request.num_inference_steps;
        for (const int64_t chunk_start : chunk_starts) {
            const int64_t chunk_end = std::min(chunk_start + Contract::kChunkFrames, frames);
            const int64_t chunk_frames = chunk_end - chunk_start;
            auto condition_rows = condition_encoder.encode(
                std::vector<float>(
                    frame_hiddens.begin() + chunk_start * frame_width,
                    frame_hiddens.begin() + chunk_end * frame_width),
                chunk_frames);
            const int64_t length = condition_encoder.latent_length(chunk_frames);
            // Row-major [length, cond_dim] -> channel-major [cond_dim, length].
            std::vector<float> condition(static_cast<size_t>(config.cond_out_dim * length));
            for (int64_t index = 0; index < length; ++index) {
                for (int64_t channel = 0; channel < config.cond_out_dim; ++channel) {
                    condition[static_cast<size_t>(channel * length + index)] =
                        condition_rows[static_cast<size_t>(index * config.cond_out_dim + channel)];
                }
            }
            const int64_t overlap = previous_overlap > 0 ? std::min(previous_overlap, length) : 0;
            for (int64_t channel = 0; channel < config.cond_out_dim && overlap > 0; ++channel) {
                std::copy(
                    previous_condition.begin() + channel * previous_overlap,
                    previous_condition.begin() + channel * previous_overlap + overlap,
                    condition.begin() + channel * length);
            }
            dit.begin_chunk(condition, length);

            auto latent = engine::sampling::generate_normal_noise(
                static_cast<size_t>(latent_channels * length),
                rng(),
                1.0F);
            std::vector<float> noise_prompt;
            if (overlap > 0) {
                noise_prompt.resize(static_cast<size_t>(latent_channels * overlap));
                for (int64_t channel = 0; channel < latent_channels; ++channel) {
                    std::copy(
                        latent.begin() + channel * length,
                        latent.begin() + channel * length + overlap,
                        noise_prompt.begin() + channel * overlap);
                }
            }

            for (int64_t step_index = 0; step_index < steps; ++step_index) {
                const float t = static_cast<float>(step_index) / static_cast<float>(steps);
                if (overlap > 0) {
                    const float noise_weight = 1.0F - (1.0F - 1.0e-6F) * t;
                    for (int64_t channel = 0; channel < latent_channels; ++channel) {
                        for (int64_t index = 0; index < overlap; ++index) {
                            latent[static_cast<size_t>(channel * length + index)] =
                                noise_weight * noise_prompt[static_cast<size_t>(channel * overlap + index)] +
                                t * previous_latent[static_cast<size_t>(channel * previous_overlap + index)];
                        }
                    }
                }
                const auto velocity = dit.guided_velocity(latent, t, request.guidance_scale);
                const float dt = 1.0F / static_cast<float>(steps);
                for (size_t index = 0; index < latent.size(); ++index) {
                    latent[index] += dt * velocity[index];
                }
            }
            if (overlap > 0) {
                for (int64_t channel = 0; channel < latent_channels; ++channel) {
                    std::copy(
                        previous_latent.begin() + channel * previous_overlap,
                        previous_latent.begin() + channel * previous_overlap + overlap,
                        latent.begin() + channel * length);
                }
            }

            // Carry latent frames [length - 344, length - 172) and their conditioning.
            const int64_t overlap_start = std::max<int64_t>(0, length - 2 * Contract::kOverlapLatentLength);
            const int64_t overlap_end = std::max(overlap_start, length - Contract::kOverlapLatentLength);
            const int64_t carry = overlap_end - overlap_start;
            previous_latent.assign(static_cast<size_t>(latent_channels * carry), 0.0F);
            previous_condition.assign(static_cast<size_t>(config.cond_out_dim * carry), 0.0F);
            for (int64_t channel = 0; channel < latent_channels; ++channel) {
                std::copy(
                    latent.begin() + channel * length + overlap_start,
                    latent.begin() + channel * length + overlap_end,
                    previous_latent.begin() + channel * carry);
            }
            for (int64_t channel = 0; channel < config.cond_out_dim; ++channel) {
                std::copy(
                    condition.begin() + channel * length + overlap_start,
                    condition.begin() + channel * length + overlap_end,
                    previous_condition.begin() + channel * carry);
            }
            previous_overlap = carry;

            latent_chunks.push_back(std::move(latent));
            latent_lengths.push_back(length);
        }
        engine::debug::timing_log_scalar(
            "minimax_music3.flow_ms", engine::debug::elapsed_ms(flow_start, Clock::now()));
    }

    // Stage 4: vocode and stitch.
    MiniMaxMusic3GenerateResult result;
    result.sample_rate = config.sample_rate;
    result.channels = 2;
    {
        MiniMaxMusic3VocoderRuntime vocoder(
            execution_, assets_->vocoder_weights, config, impl_->weight_context_bytes);
        const auto vocode_start = Clock::now();
        const int64_t hop = config.cond_output_hop;
        for (size_t chunk_index = 0; chunk_index < latent_chunks.size(); ++chunk_index) {
            auto waveform = vocoder.decode(latent_chunks[chunk_index], latent_lengths[chunk_index]);
            const int64_t samples = static_cast<int64_t>(waveform.size()) / 2;
            const int64_t left =
                chunk_index == 0 ? 0 : Contract::kCropLeftLatent * hop;
            const int64_t right =
                chunk_index + 1 == latent_chunks.size() ? 0 : Contract::kCropRightLatent * hop;
            if (left + right >= samples) {
                throw std::runtime_error("MiniMax-Music3 stitching would drop an entire window");
            }
            result.samples.insert(
                result.samples.end(),
                waveform.begin() + 2 * left,
                waveform.end() - 2 * right);
        }
        engine::debug::timing_log_scalar(
            "minimax_music3.vocode_ms", engine::debug::elapsed_ms(vocode_start, Clock::now()));
    }
    engine::debug::timing_log_scalar(
        "minimax_music3.total_ms", engine::debug::elapsed_ms(wall_start, Clock::now()));
    return result;
}

}  // namespace engine::models::minimax_music3
