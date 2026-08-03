#pragma once

#include "engine/models/dramabox/types.h"

#include <cstdint>
#include <string>
#include <vector>

namespace engine::models::dramabox {

struct DramaBoxEncodedReferenceLatents;

struct DramaBoxLatentShape {
    int64_t batch = 1;
    int64_t channels = 8;
    int64_t frames = 0;
    int64_t mel_bins = 16;

    int64_t token_count() const noexcept;
    int64_t latent_value_count() const noexcept;
};

struct DramaBoxLatentState {
    DramaBoxLatentShape target_shape;
    int64_t tokens = 0;
    std::vector<float> latent;
    std::vector<float> clean_latent;
    std::vector<float> denoise_mask;
    std::vector<float> positions;
};

struct DramaBoxPromptChunk {
    std::string text;
    double estimated_duration_seconds = 0.0;
};

double estimate_dramabox_speech_duration_seconds(const std::string & prompt);
double estimate_dramabox_duration_seconds(const std::string & prompt, float duration_scale);
std::vector<DramaBoxPromptChunk> chunk_prompt_for_duration(
    const std::string & prompt,
    float max_duration_seconds,
    float target_duration_seconds,
    float duration_scale);
int64_t dramabox_aligned_pixel_frames(double duration_seconds, float fps);
DramaBoxLatentShape dramabox_target_latent_shape(double duration_seconds, const DramaBoxConfig & config);
DramaBoxLatentState create_dramabox_initial_state(const DramaBoxLatentShape & shape, const DramaBoxConfig & config);
std::vector<float> make_dramabox_ltx2_sigmas(int64_t steps, int64_t latent_tokens);
float auto_rescale_for_cfg(float cfg);
void append_reference_latents(
    DramaBoxLatentState & state,
    const DramaBoxEncodedReferenceLatents & ref,
    const DramaBoxConfig & config);
void guided_prediction_from_velocity(
    const std::vector<float> & velocity,
    const DramaBoxLatentState & state,
    int64_t branch_count,
    float sigma,
    float cfg_scale,
    float spatio_temporal_guidance_scale,
    float guidance_rescale,
    bool cfg_enabled,
    bool stg_enabled,
    std::vector<float> & cond,
    std::vector<float> & pred);
void post_process_and_euler_step(
    DramaBoxLatentState & state,
    std::vector<float> & denoised,
    float sigma,
    float sigma_next);

}  // namespace engine::models::dramabox
