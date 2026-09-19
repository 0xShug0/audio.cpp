#pragma once

#include "engine/community_models/liveavatar/pipeline.h"

#include "engine/framework/codecs/wan_video_vae_runtime.h"
#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/core/execution_context.h"
#include "engine/framework/core/module.h"
#include "engine/framework/modules/conv_modules.h"
#include "engine/framework/modules/norm_modules.h"
#include "engine/framework/modules/speech_encoders/hubert_encoder.h"
#include "wan_s2v_audio_conditioner.h"
#include "engine/framework/modules/text_encoders/t5_base_encoder.h"

#include <ggml.h>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <random>
#include <string>
#include <vector>

namespace engine::community_models::liveavatar {

using Clock = std::chrono::steady_clock;

struct GgmlContextDeleter {
    void operator()(ggml_context * ctx) const {
        if (ctx != nullptr) {
            ggml_free(ctx);
        }
    }
};

class LiveAvatarDenoiserStaticCache;


struct LiveAvatarPreparedAudio {
    WanS2VAudioBuckets buckets;
    std::vector<float> hubert_layer_stack;
};

struct LiveAvatarDenoiserAttentionWeights {
    engine::modules::LinearWeights q;
    engine::modules::LinearWeights k;
    engine::modules::LinearWeights v;
    engine::modules::LinearWeights o;
    engine::core::TensorValue norm_q;
    engine::core::TensorValue norm_k;
};

struct LiveAvatarDenoiserBlockWeights {
    engine::core::TensorValue modulation;
    LiveAvatarDenoiserAttentionWeights self_attention;
    LiveAvatarDenoiserAttentionWeights cross_attention;
    engine::modules::NormWeights norm3;
    engine::modules::LinearWeights ffn_in;
    engine::modules::LinearWeights ffn_out;
};

struct LiveAvatarDenoiserAudioEncoderWeights {
    engine::core::TensorValue layer_weights;
    engine::modules::Conv1dWeights conv1_local;
    engine::modules::Conv1dWeights conv1_global;
    engine::modules::Conv1dWeights conv2;
    engine::modules::Conv1dWeights conv3;
    engine::modules::LinearWeights final_linear;
    engine::core::TensorValue padding_tokens;
};

struct LiveAvatarDenoiserAudioInjectorWeights {
    LiveAvatarDenoiserAttentionWeights attention;
    engine::modules::LinearWeights adain_linear;
};

struct LiveAvatarDenoiserFramePackerWeights {
    engine::modules::Conv3dWeights proj;
    engine::modules::Conv3dWeights proj_2x;
    engine::modules::Conv3dWeights proj_4x;
};

struct LiveAvatarDenoiserHeadWeights {
    engine::core::TensorValue modulation;
    engine::modules::LinearWeights projection;
};

struct LiveAvatarDenoiserWeights {
    std::shared_ptr<engine::core::BackendWeightStore> store;
    std::shared_ptr<engine::core::BackendWeightStore> block_store;
    bool block_weights_host_resident = false;
    engine::modules::Conv3dWeights patch_embedding;
    engine::modules::Conv3dWeights cond_encoder;
    engine::modules::LinearWeights text_embedding_0;
    engine::modules::LinearWeights text_embedding_2;
    engine::modules::LinearWeights time_embedding_0;
    engine::modules::LinearWeights time_embedding_2;
    engine::modules::LinearWeights time_projection;
    std::vector<LiveAvatarDenoiserBlockWeights> blocks;
    LiveAvatarDenoiserAudioEncoderWeights audio_encoder;
    std::vector<LiveAvatarDenoiserAudioInjectorWeights> audio_injectors;
    LiveAvatarDenoiserFramePackerWeights frame_packer;
    engine::core::TensorValue trainable_cond_mask;
    LiveAvatarDenoiserHeadWeights head;
};

struct LiveAvatarRopeRange {
    int64_t start_t = 0;
    int64_t start_h = 0;
    int64_t start_w = 0;
    int64_t end_t = 0;
    int64_t end_h = 0;
    int64_t end_w = 0;
    int64_t total_t = 0;
    int64_t total_h = 0;
    int64_t total_w = 0;
};

struct LiveAvatarAttentionKV {
    engine::core::TensorValue key;
    engine::core::TensorValue value;
};

struct LiveAvatarDenoiserRunInput {
    const std::vector<float> * latent = nullptr;
    const std::vector<float> * ref_latents = nullptr;
    const std::vector<float> * cond_latents = nullptr;
    const std::vector<float> * motion_latents = nullptr;
    const std::vector<float> * projected_text_context = nullptr;
    const std::vector<float> * encoded_audio_local = nullptr;
    const std::vector<float> * encoded_audio_global = nullptr;
    float timestep = 0.0F;
    float guidance_scale = 1.0F;
};

struct LiveAvatarDenoiserConditionRunInput {
    const std::vector<float> * text_context = nullptr;
    const std::vector<float> * audio_input = nullptr;
};

struct LiveAvatarDenoiserPreparedCondition {
    std::vector<float> projected_text_context;
    std::vector<float> encoded_audio_local;
    std::vector<float> encoded_audio_global;
};

struct LiveAvatarEncodedAudio {
    engine::core::TensorValue local;
    engine::core::TensorValue global;
};

struct LiveAvatarGenerateShared {
    const LiveAvatarGenerateRequest & request;
    Clock::time_point total_start;
    std::vector<float> text_context;
    std::vector<float> negative_context;
    std::vector<float> reference_image_input;
    std::vector<float> ref_latents;
    WanS2VAudioConditionerConfig audio_config;
    WanS2VAudioBuckets audio_buckets;
    int64_t latent_target_frames = 0;
};

class LiveAvatarTextEncoderRuntime {
public:
    LiveAvatarTextEncoderRuntime(
        engine::core::ExecutionContext & execution,
        std::shared_ptr<const LiveAvatarAssets> assets,
        engine::modules::T5BaseEncoderConfig config);
    ~LiveAvatarTextEncoderRuntime();

    std::vector<std::vector<float>> encode_batch(
        const std::vector<std::vector<int32_t>> & input_ids,
        const std::vector<int64_t> & token_counts);

private:
    struct Data;
    std::unique_ptr<Data> data_;
};

class LiveAvatarVAERuntime {
public:
    LiveAvatarVAERuntime(std::shared_ptr<const LiveAvatarAssets> assets);
    ~LiveAvatarVAERuntime();

    void release(engine::core::ExecutionContext & execution);
    std::vector<float> encode_raw(
        engine::core::ExecutionContext & execution,
        const std::vector<float> & values,
        int64_t channels,
        int64_t frames,
        int64_t height,
        int64_t width,
        engine::core::TensorShape * output_shape);
    std::vector<float> encode_cached(
        engine::core::ExecutionContext & execution,
        const std::vector<float> & values,
        int64_t channels,
        int64_t frames,
        int64_t height,
        int64_t width,
        int64_t chunk_size,
        bool cache_f16,
        engine::core::TensorShape * output_shape);
    std::vector<float> decode(
        engine::core::ExecutionContext & execution,
        const std::vector<float> & values,
        int64_t channels,
        int64_t frames,
        int64_t height,
        int64_t width,
        int64_t tile_size,
        engine::core::TensorShape * output_shape);

private:
    struct Data;
    std::unique_ptr<Data> data_;
};

class LiveAvatarDenoiserRuntime {
public:
    LiveAvatarDenoiserRuntime(
        std::shared_ptr<const LiveAvatarAssets> assets,
        bool denoiser_weight_streaming);
    ~LiveAvatarDenoiserRuntime();

    LiveAvatarDenoiserWeights & ensure_weights(engine::core::ExecutionContext & execution);
    void release(engine::core::ExecutionContext & execution);
    void release_condition_graph();
    LiveAvatarDenoiserPreparedCondition prepare_condition(
        engine::core::ExecutionContext & execution,
        const LiveAvatarDenoiserConditionRunInput & input,
        int64_t target_frames,
        int64_t audio_frames,
        int64_t latent_audio_start_frame);
    LiveAvatarDenoiserStaticCache & prepare_static_cache(
        engine::core::ExecutionContext & execution,
        const LiveAvatarDenoiserPreparedCondition & condition,
        const engine::core::TensorShape & latent_shape,
        int64_t lanes,
        bool use_sage_attention);
    std::vector<float> denoise(
        engine::core::ExecutionContext & execution,
        const LiveAvatarDenoiserRunInput & input,
        const engine::core::TensorShape & latent_shape,
        bool cfg_enabled = false,
        bool use_sage_attention = false,
        const LiveAvatarDenoiserStaticCache * static_cache = nullptr);
    std::vector<float> denoise_layerwise(
        engine::core::ExecutionContext & execution,
        const LiveAvatarDenoiserRunInput & input,
        const engine::core::TensorShape & latent_shape,
        bool use_sage_attention,
        const LiveAvatarDenoiserStaticCache & static_cache,
        int64_t layer_batch);

private:
    struct Data;
    std::unique_ptr<Data> data_;
};

class LiveAvatarPipelineState {
public:
    LiveAvatarPipelineState(
        std::shared_ptr<const LiveAvatarAssets> assets,
        engine::core::ExecutionContext & execution,
        bool denoiser_weight_streaming);
    ~LiveAvatarPipelineState();

    std::vector<std::vector<float>> encode_text_batch(
        engine::core::ExecutionContext & execution,
        const std::vector<std::vector<int32_t>> & input_ids,
        const std::vector<int64_t> & token_counts);
    LiveAvatarPreparedAudio prepare_audio_buckets(
        engine::core::ExecutionContext & execution,
        const std::vector<float> & audio_hubert_input,
        int64_t batch_frames,
        int64_t audio_layers);

    LiveAvatarDenoiserWeights & ensure_denoiser_weights(engine::core::ExecutionContext & execution);
    void release_vae_weights(engine::core::ExecutionContext & execution);
    void release_denoiser(engine::core::ExecutionContext & execution);
    void release_denoiser_condition_graph();
    LiveAvatarDenoiserPreparedCondition prepare_denoiser_condition(
        engine::core::ExecutionContext & execution,
        const LiveAvatarDenoiserConditionRunInput & input,
        int64_t target_frames,
        int64_t audio_frames,
        int64_t latent_audio_start_frame);
    LiveAvatarDenoiserStaticCache & prepare_denoiser_static_cache(
        engine::core::ExecutionContext & execution,
        const LiveAvatarDenoiserPreparedCondition & condition,
        const engine::core::TensorShape & latent_shape,
        int64_t lanes,
        bool use_sage_attention);
    std::vector<float> denoise(
        engine::core::ExecutionContext & execution,
        const LiveAvatarDenoiserRunInput & input,
        const engine::core::TensorShape & latent_shape,
        bool cfg_enabled = false,
        bool use_sage_attention = false,
        const LiveAvatarDenoiserStaticCache * static_cache = nullptr);
    std::vector<float> denoise_layerwise(
        engine::core::ExecutionContext & execution,
        const LiveAvatarDenoiserRunInput & input,
        const engine::core::TensorShape & latent_shape,
        bool use_sage_attention,
        const LiveAvatarDenoiserStaticCache & static_cache,
        int64_t layer_batch);
    std::vector<float> vae_encode_raw(
        engine::core::ExecutionContext & execution,
        const std::vector<float> & values,
        int64_t channels,
        int64_t frames,
        int64_t height,
        int64_t width,
        engine::core::TensorShape * output_shape);
    std::vector<float> vae_encode_cached(
        engine::core::ExecutionContext & execution,
        const std::vector<float> & values,
        int64_t channels,
        int64_t frames,
        int64_t height,
        int64_t width,
        int64_t chunk_size,
        bool cache_f16,
        engine::core::TensorShape * output_shape);
    std::vector<float> vae_decode(
        engine::core::ExecutionContext & execution,
        const std::vector<float> & values,
        int64_t channels,
        int64_t frames,
        int64_t height,
        int64_t width,
        int64_t tile_size,
        engine::core::TensorShape * output_shape);

private:
    struct Data;
    std::unique_ptr<Data> data_;
};

std::shared_ptr<const LiveAvatarAssets> require_assets(std::shared_ptr<const LiveAvatarAssets> assets);
LiveAvatarGenerateShared prepare_liveavatar_generate_shared(
    LiveAvatarPipelineState & state,
    engine::core::ExecutionContext & execution,
    const std::shared_ptr<const LiveAvatarAssets> & assets,
    const LiveAvatarGenerateRequest & request);

std::vector<float> slice_video_frames(
    const std::vector<float> & values,
    int64_t channels,
    int64_t frames,
    int64_t height,
    int64_t width,
    int64_t start_frame,
    int64_t take_frames);
std::vector<float> repeat_video_frame(
    const std::vector<float> & frame,
    int64_t channels,
    int64_t height,
    int64_t width,
    int64_t frames);
std::vector<float> slice_video_time(
    const std::vector<float> & values,
    int64_t channels,
    int64_t frames,
    int64_t height,
    int64_t width,
    int64_t start_frame,
    int64_t take_frames);
std::vector<float> slice_video_spatial(
    const std::vector<float> & values,
    int64_t channels,
    int64_t frames,
    int64_t height,
    int64_t width,
    int64_t start_y,
    int64_t take_h,
    int64_t start_x,
    int64_t take_w);
void append_video_time(
    std::vector<float> & out,
    int64_t channels,
    int64_t lhs_frames,
    int64_t rhs_frames,
    int64_t height,
    int64_t width,
    const std::vector<float> & rhs);
void copy_video_time(
    std::vector<float> & dst,
    int64_t channels,
    int64_t dst_frames,
    int64_t height,
    int64_t width,
    int64_t dst_frame_start,
    const std::vector<float> & values,
    int64_t value_frames);
std::vector<float> concat_video_time(
    const std::vector<float> & lhs,
    int64_t channels,
    int64_t lhs_frames,
    int64_t height,
    int64_t width,
    const std::vector<float> & rhs,
    int64_t rhs_frames);
std::vector<float> slice_audio_bucket_frames(
    const std::vector<float> & values,
    int64_t layers,
    int64_t dims,
    int64_t frames,
    int64_t start_frame,
    int64_t take_frames);
std::vector<std::byte> video_to_rgb24(
    const std::vector<float> & video,
    int64_t channels,
    int64_t frames,
    int64_t height,
    int64_t width,
    int64_t start_frame,
    int64_t take_frames);
std::vector<float> make_flow_sigmas(int64_t steps, float shift);
std::vector<float> make_initial_noise(const engine::core::TensorShape & shape, uint64_t seed);
std::vector<float> make_timestep_features(float timestep);
void scale_values(std::vector<float> & values, float scale);
void apply_wan21_latent_process_in(std::vector<float> & values, const engine::core::TensorShape & shape);
void apply_wan21_latent_process_out(std::vector<float> & values, const engine::core::TensorShape & shape);
float unipc_alpha(float sigma);
std::vector<float> unipc_bh2_order1_step(
    const std::vector<float> & sample,
    const std::vector<float> & model_prev,
    const std::vector<float> * model_current,
    float sigma_from,
    float sigma_to);
std::vector<float> combine_cfg(
    const std::vector<float> & cond,
    const std::vector<float> & uncond,
    float guidance_scale);
void flow_euler_step_in_place(
    std::vector<float> & sample,
    const std::vector<float> & model_output,
    float sigma,
    float sigma_next);
std::vector<float> make_liveavatar_rope_table(
    int64_t heads,
    int64_t head_dim,
    const std::vector<LiveAvatarRopeRange> & ranges,
    bool sine);
engine::core::TensorValue apply_shift_scale(
    engine::core::ModuleBuildContext & ctx,
    const engine::core::TensorValue & x,
    const engine::core::TensorValue & shift,
    const engine::core::TensorValue & scale);
engine::core::TensorValue linear_native(
    engine::core::ModuleBuildContext & ctx,
    const engine::core::TensorValue & input,
    const engine::modules::LinearWeights & weights,
    int64_t input_dim,
    int64_t output_dim);
engine::core::TensorValue conv3d_tokens(
    engine::core::ModuleBuildContext & ctx,
    const engine::core::TensorValue & input,
    const engine::modules::Conv3dWeights & weights,
    int64_t input_channels,
    int64_t output_channels,
    int64_t kernel_t,
    int64_t kernel_h,
    int64_t kernel_w,
    int64_t stride_t,
    int64_t stride_h,
    int64_t stride_w);
engine::core::TensorValue build_condition_mask(
    engine::core::ModuleBuildContext & ctx,
    const engine::core::TensorValue & embedding,
    const engine::core::TensorValue & target_like,
    const engine::core::TensorValue & ref_like);
engine::core::TensorValue build_denoiser_block_with_cached_cross_attention(
    engine::core::ModuleBuildContext & ctx,
    ggml_backend_t backend,
    bool use_sage_attention,
    const engine::core::TensorValue & input,
    const LiveAvatarAttentionKV & kv,
    const engine::core::TensorValue & actual_e0,
    const engine::core::TensorValue & zero_e0,
    const engine::core::TensorValue & rope_cos,
    const engine::core::TensorValue & rope_sin,
    const LiveAvatarDenoiserBlockWeights & weights,
    const LiveAvatarConfig & config,
    int64_t original_tokens);
LiveAvatarAttentionKV build_cross_attention_kv(
    engine::core::ModuleBuildContext & ctx,
    const engine::core::TensorValue & context,
    const LiveAvatarDenoiserAttentionWeights & weights,
    const LiveAvatarConfig & config);
engine::core::TensorValue build_cross_attention_with_cached_kv(
    engine::core::ModuleBuildContext & ctx,
    ggml_backend_t backend,
    bool use_sage_attention,
    const engine::core::TensorValue & input,
    const LiveAvatarAttentionKV & kv,
    const LiveAvatarDenoiserAttentionWeights & weights,
    const LiveAvatarConfig & config);
engine::core::TensorValue build_audio_injection(
    engine::core::ModuleBuildContext & ctx,
    ggml_backend_t backend,
    bool use_sage_attention,
    const engine::core::TensorValue & input,
    const LiveAvatarEncodedAudio & audio,
    const LiveAvatarDenoiserAudioInjectorWeights & weights,
    const LiveAvatarConfig & config,
    int64_t original_tokens,
    int64_t target_latent_frames);
engine::core::TensorValue build_audio_injection_with_cached_condition(
    engine::core::ModuleBuildContext & ctx,
    ggml_backend_t backend,
    bool use_sage_attention,
    const engine::core::TensorValue & input,
    const LiveAvatarAttentionKV & audio_kv,
    const engine::core::TensorValue & shift,
    const engine::core::TensorValue & scale,
    const LiveAvatarDenoiserAudioInjectorWeights & weights,
    const LiveAvatarConfig & config,
    int64_t original_tokens,
    int64_t target_latent_frames);
engine::core::TensorValue apply_wan_rope(
    engine::core::ModuleBuildContext & ctx,
    const engine::core::TensorValue & input,
    const engine::core::TensorValue & cos,
    const engine::core::TensorValue & sin,
    int64_t head_dim);
engine::core::TensorValue scaled_attention(
    engine::core::ModuleBuildContext & ctx,
    ggml_backend_t backend,
    bool use_sage_attention,
    const engine::core::TensorValue & q_heads,
    const engine::core::TensorValue & k_heads,
    const engine::core::TensorValue & v_heads,
    int64_t head_dim,
    const std::optional<engine::core::TensorValue> & attention_mask = std::nullopt);
engine::core::TensorValue build_cross_attention(
    engine::core::ModuleBuildContext & ctx,
    ggml_backend_t backend,
    bool use_sage_attention,
    const engine::core::TensorValue & input,
    const engine::core::TensorValue & context,
    const LiveAvatarDenoiserAttentionWeights & weights,
    const LiveAvatarConfig & config);
engine::core::TensorValue segmented_adaln_input(
    engine::core::ModuleBuildContext & ctx,
    const engine::core::TensorValue & normalized,
    const engine::core::TensorValue & actual,
    const engine::core::TensorValue & zero,
    int64_t shift_index,
    int64_t scale_index,
    int64_t segment_tokens);
engine::core::TensorValue split_modulated_tokens(
    engine::core::ModuleBuildContext & ctx,
    const engine::core::TensorValue & values,
    const engine::core::TensorValue & actual,
    const engine::core::TensorValue & zero,
    int64_t coefficient,
    int64_t segment_tokens,
    bool scale_add_one);
bool same_shape(const engine::core::TensorShape & lhs, const engine::core::TensorShape & rhs) noexcept;
std::vector<float> unpatchify_head_tokens(
    const std::vector<float> & tokens,
    int64_t frames,
    int64_t latent_height,
    int64_t latent_width);

LiveAvatarVideoResult generate_liveavatar_offline(
    LiveAvatarPipelineState & state,
    engine::core::ExecutionContext & execution,
    const std::shared_ptr<const LiveAvatarAssets> & assets,
    LiveAvatarGenerateShared & shared);
LiveAvatarVideoResult generate_liveavatar_blockwise(
    LiveAvatarPipelineState & state,
    engine::core::ExecutionContext & execution,
    const std::shared_ptr<const LiveAvatarAssets> & assets,
    LiveAvatarGenerateShared & shared,
    const LiveAvatarVideoChunkCallback & chunk_callback);

}  // namespace engine::community_models::liveavatar
