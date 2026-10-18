#include "engine/community_models/echo_tts/sampler.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace engine::models::echo_tts {
namespace {

// inference.py::sample_euler_cfg_independent_guidances, INIT_SCALE.
constexpr float kInitScale = 0.999F;

}  // namespace

std::vector<float> euler_timestep_schedule(int num_steps) {
    if (num_steps <= 0) {
        throw std::runtime_error("Echo-TTS sampler requires at least one step");
    }
    std::vector<float> schedule(static_cast<size_t>(num_steps) + 1);
    for (int i = 0; i <= num_steps; ++i) {
        // torch.linspace(1, 0, n + 1) puts exact endpoints at both ends.
        const float ramp =
            1.0F - static_cast<float>(i) / static_cast<float>(num_steps);
        schedule[static_cast<size_t>(i)] = ramp * kInitScale;
    }
    return schedule;
}

bool cfg_active(float t, float cfg_min_t, float cfg_max_t) {
    return t >= cfg_min_t && t <= cfg_max_t;
}

std::vector<float> combine_cfg_lanes(
    const std::vector<float> & lanes,
    int64_t lane_elements,
    float cfg_scale_text,
    float cfg_scale_speaker) {
    if (lane_elements <= 0 ||
        static_cast<int64_t>(lanes.size()) != lane_elements * 3) {
        throw std::runtime_error("Echo-TTS CFG combine expects exactly three lanes");
    }
    const float * cond = lanes.data();
    const float * uncond_text = lanes.data() + lane_elements;
    const float * uncond_speaker = lanes.data() + 2 * lane_elements;

    std::vector<float> out(static_cast<size_t>(lane_elements));
    for (int64_t i = 0; i < lane_elements; ++i) {
        const float c = cond[i];
        out[static_cast<size_t>(i)] =
            c + cfg_scale_text * (c - uncond_text[i]) +
            cfg_scale_speaker * (c - uncond_speaker[i]);
    }
    return out;
}

std::vector<float> run_euler_sampler(
    const EchoSamplerOptions & options,
    int64_t sequence_length,
    int64_t latent_size,
    std::vector<float> initial_noise,
    const EchoDenoiseFn & denoise,
    const std::function<void()> & on_kv_rescale) {
    const int64_t elements = sequence_length * latent_size;
    if (static_cast<int64_t>(initial_noise.size()) != elements) {
        throw std::runtime_error("Echo-TTS sampler received a mis-shaped noise buffer");
    }
    if (!denoise) {
        throw std::runtime_error("Echo-TTS sampler requires a denoise callback");
    }

    std::vector<float> x_t = std::move(initial_noise);
    if (options.truncation_factor.has_value()) {
        const float factor = *options.truncation_factor;
        for (auto & value : x_t) {
            value *= factor;
        }
    }

    const auto schedule = euler_timestep_schedule(options.num_steps);
    bool kv_scaled = options.speaker_kv_scale.has_value();

    const int cfg_interval = std::max(1, options.cfg_interval);
    // Guidance correction carried between refreshes when cfg_interval > 1. This
    // is the whole additive term, w_text * (v_cond - v_text) + w_speaker *
    // (v_cond - v_speaker), held in absolute units rather than as a ratio so a
    // reused correction cannot amplify a small v_cond.
    std::vector<float> cfg_delta;
    int steps_since_refresh = 0;

    for (int step = 0; step < options.num_steps; ++step) {
        const float t = schedule[static_cast<size_t>(step)];
        const float t_next = schedule[static_cast<size_t>(step) + 1];
        const bool use_cfg = cfg_active(t, options.cfg_min_t, options.cfg_max_t);

        std::vector<float> v_pred;
        if (use_cfg) {
            // The first CFG step always refreshes, so a stale delta is never
            // applied before one has been measured.
            const bool refresh = cfg_delta.empty() || steps_since_refresh >= cfg_interval - 1;
            if (refresh) {
                auto lanes = denoise(x_t, t, 3);
                if (static_cast<int64_t>(lanes.size()) != elements * 3) {
                    throw std::runtime_error("Echo-TTS denoiser returned mis-shaped CFG lanes");
                }
                v_pred = combine_cfg_lanes(
                    lanes, elements, options.cfg_scale_text, options.cfg_scale_speaker);
                if (cfg_interval > 1) {
                    cfg_delta.resize(static_cast<size_t>(elements));
                    for (int64_t i = 0; i < elements; ++i) {
                        cfg_delta[static_cast<size_t>(i)] =
                            v_pred[static_cast<size_t>(i)] - lanes[static_cast<size_t>(i)];
                    }
                }
                steps_since_refresh = 0;
            } else {
                v_pred = denoise(x_t, t, 1);
                if (static_cast<int64_t>(v_pred.size()) != elements) {
                    throw std::runtime_error("Echo-TTS denoiser returned a mis-shaped velocity");
                }
                for (int64_t i = 0; i < elements; ++i) {
                    v_pred[static_cast<size_t>(i)] += cfg_delta[static_cast<size_t>(i)];
                }
                ++steps_since_refresh;
            }
        } else {
            v_pred = denoise(x_t, t, 1);
            if (static_cast<int64_t>(v_pred.size()) != elements) {
                throw std::runtime_error("Echo-TTS denoiser returned a mis-shaped velocity");
            }
        }

        // Speaker KV scaling is undone once the schedule crosses below
        // speaker_kv_min_t, matching upstream's boundary test on (t, t_next).
        if (kv_scaled && options.speaker_kv_min_t.has_value()) {
            const float threshold = *options.speaker_kv_min_t;
            if (t_next < threshold && t >= threshold) {
                if (on_kv_rescale) {
                    on_kv_rescale();
                }
                kv_scaled = false;
            }
        }

        const float dt = t_next - t;
        for (int64_t i = 0; i < elements; ++i) {
            x_t[static_cast<size_t>(i)] += v_pred[static_cast<size_t>(i)] * dt;
        }
    }
    return x_t;
}

}  // namespace engine::models::echo_tts
