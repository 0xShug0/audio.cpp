#include "engine/models/chatterbox/s3gen_inference.h"
#include "components/component_weights.h"
#include "components/s3gen_weights.h"
#include "engine/framework/core/backend.h"
#include "engine/framework/core/execution_context.h"
#include "engine/framework/core/module.h"
#include "engine/framework/sampling/torch_random.h"

#include "ggml.h"

#include <chrono>
#include <limits>
#include <memory>

namespace engine::models::chatterbox {
namespace {

int64_t token_capacity_quantum(const engine::core::BackendConfig & backend) {
    (void) backend;
    return 1;
}

int64_t round_up_capacity(int64_t value, int64_t quantum) {
    if (value <= 0 || quantum <= 0) {
        throw std::runtime_error("S3 capacity rounding requires positive inputs");
    }
    return ((value + quantum - 1) / quantum) * quantum;
}

std::vector<float> make_prefix_mask_bct(int64_t batch, int64_t frames, int64_t valid_frames) {
    std::vector<float> mask(static_cast<size_t>(batch * frames), 0.0f);
    for (int64_t batch_index = 0; batch_index < batch; ++batch_index) {
        for (int64_t frame = 0; frame < frames; ++frame) {
            mask[static_cast<size_t>(batch_index * frames + frame)] = frame < valid_frames ? 1.0f : 0.0f;
        }
    }
    return mask;
}

std::vector<int32_t> zero_pad_tokens(
    const std::vector<int32_t> & tokens,
    int64_t capacity) {
    std::vector<int32_t> padded(static_cast<size_t>(capacity), 0);
    std::copy(tokens.begin(), tokens.end(), padded.begin());
    return padded;
}

std::vector<float> zero_pad_btc(
    const std::vector<float> & values,
    int64_t valid_frames,
    int64_t capacity_frames,
    int64_t channels) {
    std::vector<float> padded(static_cast<size_t>(capacity_frames * channels), 0.0f);
    for (int64_t frame = 0; frame < valid_frames; ++frame) {
        for (int64_t channel = 0; channel < channels; ++channel) {
            padded[static_cast<size_t>(frame * channels + channel)] =
                values[static_cast<size_t>(frame * channels + channel)];
        }
    }
    return padded;
}

std::vector<float> slice_bct(
    const std::vector<float> & values,
    int64_t source_frames,
    int64_t frames,
    int64_t channels) {
    std::vector<float> sliced(static_cast<size_t>(frames * channels), 0.0f);
    for (int64_t channel = 0; channel < channels; ++channel) {
        for (int64_t frame = 0; frame < frames; ++frame) {
            sliced[static_cast<size_t>(channel * frames + frame)] =
                values[static_cast<size_t>(channel * source_frames + frame)];
        }
    }
    return sliced;
}

void zero_tail_btc(
    std::vector<float> & values,
    int64_t valid_frames,
    int64_t capacity_frames,
    int64_t channels) {
    if (valid_frames >= capacity_frames) {
        return;
    }
    for (int64_t frame = valid_frames; frame < capacity_frames; ++frame) {
        std::fill_n(
            values.begin() + static_cast<ptrdiff_t>(frame * channels),
            static_cast<size_t>(channels),
            0.0f);
    }
}

std::vector<float> make_zero_padded_conditions_bct(
    const EmbedReferenceOutputs & ref_dict,
    int64_t total_frames) {
    std::vector<float> cond(static_cast<size_t>(80 * total_frames), 0.0f);
    for (int64_t frame = 0; frame < ref_dict.prompt_feat_frames; ++frame) {
        for (int64_t dim = 0; dim < ref_dict.prompt_feat_dims; ++dim) {
            cond[static_cast<size_t>(dim * total_frames + frame)] =
                ref_dict.prompt_feat[static_cast<size_t>(frame * ref_dict.prompt_feat_dims + dim)];
        }
    }
    return cond;
}

engine::core::TensorValue contiguous(
    engine::core::ModuleBuildContext & ctx,
    const engine::core::TensorValue & input) {
    return engine::core::ensure_backend_addressable_layout(ctx, input);
}

engine::core::TensorValue add_tensor_values(
    engine::core::ModuleBuildContext & ctx,
    const engine::core::TensorValue & lhs,
    const engine::core::TensorValue & rhs) {
    auto lhs_contiguous = contiguous(ctx, lhs);
    auto rhs_contiguous = contiguous(ctx, rhs);
    return engine::core::wrap_tensor(
        ggml_add(ctx.ggml, lhs_contiguous.tensor, rhs_contiguous.tensor),
        lhs.shape,
        GGML_TYPE_F32);
}

engine::core::TensorValue linear_rows_graph(
    engine::core::ModuleBuildContext & ctx,
    const engine::core::TensorValue & input,
    const S3FlowEncoderWeights::LinearWeights & weights) {
    auto projected = engine::core::wrap_tensor(
        ggml_mul_mat(ctx.ggml, weights.weight_tensor.tensor, contiguous(ctx, input).tensor),
        engine::core::TensorShape::from_dims({input.shape.dims[0], weights.out_features}),
        GGML_TYPE_F32);
    if (weights.use_bias) {
        auto bias_view = engine::core::reshape_tensor(
            ctx,
            weights.bias_tensor,
            engine::core::TensorShape::from_dims({1, weights.out_features}));
        auto repeated = engine::core::wrap_tensor(ggml_repeat(ctx.ggml, bias_view.tensor, projected.tensor), projected.shape, GGML_TYPE_F32);
        projected = add_tensor_values(ctx, projected, repeated);
    }
    return projected;
}

class S3TokenEmbeddingGraph {
public:
    S3TokenEmbeddingGraph(
        const S3FlowEncoderWeights & weights,
        int64_t token_count,
        const engine::core::ExecutionContext & execution_context)
        : token_count_(token_count),
          execution_context_(&execution_context) {
        ggml_init_params params = {};
        params.mem_size = 32ull * 1024ull * 1024ull;
        params.mem_buffer = nullptr;
        params.no_alloc = true;
        ggml_ = ggml_init(params);
        if (ggml_ == nullptr) {
            throw std::runtime_error("failed to initialize S3 token embedding graph");
        }

        engine::core::ModuleBuildContext ctx = {};
        ctx.ggml = ggml_;
        ctx.module_instance_name = "s3_token_embedding";
        ctx.backend_type = execution_context.backend_type();
        token_ids_ = engine::core::make_tensor(
            ctx,
            GGML_TYPE_I32,
            engine::core::TensorShape::from_dims({token_count_}));
        embedding_out_ = engine::core::wrap_tensor(
            ggml_get_rows(ctx.ggml, weights.input_embedding_tensor.tensor, token_ids_.tensor),
            engine::core::TensorShape::from_dims({token_count_, 512}),
            GGML_TYPE_F32);

        graph_ = ggml_new_graph_custom(ggml_, 1024, false);
        ggml_build_forward_expand(graph_, embedding_out_.tensor);
        buffer_ = ggml_backend_alloc_ctx_tensors(ggml_, execution_context_->backend());
        if (buffer_ == nullptr) {
            throw std::runtime_error("failed to allocate S3 token embedding graph tensors");
        }
    }

    ~S3TokenEmbeddingGraph() {
        if (execution_context_ != nullptr && graph_ != nullptr) {
            engine::core::release_backend_graph_resources(execution_context_->backend(), graph_);
        }
        if (buffer_ != nullptr) {
            ggml_backend_buffer_free(buffer_);
        }
        if (ggml_ != nullptr) {
            ggml_free(ggml_);
        }
    }

    std::vector<float> run(const std::vector<int32_t> & ids) {
        if (static_cast<int64_t>(ids.size()) != token_count_) {
            throw std::runtime_error("S3 token embedding graph token count mismatch");
        }
        engine::core::write_tensor_i32(token_ids_, ids);
        if (engine::core::compute_backend_graph(execution_context_->backend(), graph_) != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("S3 token embedding graph compute failed");
        }
        return engine::core::read_tensor_f32(embedding_out_.tensor);
    }

private:
    int64_t token_count_ = 0;
    const engine::core::ExecutionContext * execution_context_ = nullptr;
    ggml_context * ggml_ = nullptr;
    ggml_backend_buffer_t buffer_ = nullptr;
    ggml_cgraph * graph_ = nullptr;
    engine::core::TensorValue token_ids_;
    engine::core::TensorValue embedding_out_;
};

class S3Token2MelPrepareGraph {
public:
    struct Outputs {
        std::vector<float> mu_bct;
        std::vector<float> speaker;
    };

    S3Token2MelPrepareGraph(
        const S3FlowEncoderWeights & weights,
        int64_t frames,
        const engine::core::ExecutionContext & execution_context)
        : frames_(frames),
          execution_context_(&execution_context) {
        ggml_init_params params = {};
        params.mem_size = 64ull * 1024ull * 1024ull;
        params.mem_buffer = nullptr;
        params.no_alloc = true;
        ggml_ = ggml_init(params);
        if (ggml_ == nullptr) {
            throw std::runtime_error("failed to initialize S3 token2mel prepare graph");
        }

        engine::core::ModuleBuildContext ctx = {};
        ctx.ggml = ggml_;
        ctx.module_instance_name = "s3_token2mel_prepare";
        ctx.backend_type = execution_context.backend_type();
        encoded_in_ = engine::core::make_tensor(
            ctx,
            GGML_TYPE_F32,
            engine::core::TensorShape::from_dims({frames_, 512}));
        speaker_in_ = engine::core::make_tensor(
            ctx,
            GGML_TYPE_F32,
            engine::core::TensorShape::from_dims({1, 192}));

        auto mu_rows = linear_rows_graph(ctx, encoded_in_, weights.encoder_proj);
        mu_bct_ = engine::core::wrap_tensor(
            ggml_cont(ctx.ggml, ggml_permute(ctx.ggml, mu_rows.tensor, 1, 0, 2, 3)),
            engine::core::TensorShape::from_dims({80, frames_}),
            GGML_TYPE_F32);

        auto speaker_sq = engine::core::wrap_tensor(ggml_sqr(ctx.ggml, speaker_in_.tensor), speaker_in_.shape, GGML_TYPE_F32);
        auto norm_sq = engine::core::wrap_tensor(
            ggml_sum_rows(ctx.ggml, speaker_sq.tensor),
            engine::core::TensorShape::from_dims({1, 1}),
            GGML_TYPE_F32);
        auto norm = engine::core::wrap_tensor(ggml_sqrt(ctx.ggml, norm_sq.tensor), norm_sq.shape, GGML_TYPE_F32);
        auto norm_rep = engine::core::wrap_tensor(ggml_repeat(ctx.ggml, norm.tensor, speaker_in_.tensor), speaker_in_.shape, GGML_TYPE_F32);
        auto speaker_norm = engine::core::wrap_tensor(
            ggml_div(ctx.ggml, speaker_in_.tensor, norm_rep.tensor),
            speaker_in_.shape,
            GGML_TYPE_F32);
        speaker_out_ = linear_rows_graph(ctx, speaker_norm, weights.speaker_affine);

        graph_ = ggml_new_graph_custom(ggml_, 4096, false);
        ggml_build_forward_expand(graph_, mu_bct_.tensor);
        ggml_build_forward_expand(graph_, speaker_out_.tensor);
        buffer_ = ggml_backend_alloc_ctx_tensors(ggml_, execution_context_->backend());
        if (buffer_ == nullptr) {
            throw std::runtime_error("failed to allocate S3 token2mel prepare graph tensors");
        }

    }

    ~S3Token2MelPrepareGraph() {
        if (execution_context_ != nullptr && graph_ != nullptr) {
            engine::core::release_backend_graph_resources(execution_context_->backend(), graph_);
        }
        if (buffer_ != nullptr) {
            ggml_backend_buffer_free(buffer_);
        }
        if (ggml_ != nullptr) {
            ggml_free(ggml_);
        }
    }

    Outputs run(
        const std::vector<float> & encoded,
        const std::vector<float> & speaker_embedding) {
        engine::core::write_tensor_f32(encoded_in_, encoded);
        engine::core::write_tensor_f32(speaker_in_, speaker_embedding);
        if (engine::core::compute_backend_graph(execution_context_->backend(), graph_) != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("S3 token2mel prepare graph compute failed");
        }
        Outputs outputs;
        outputs.mu_bct = engine::core::read_tensor_f32(mu_bct_.tensor);
        outputs.speaker = engine::core::read_tensor_f32(speaker_out_.tensor);
        return outputs;
    }

private:
    int64_t frames_ = 0;
    const engine::core::ExecutionContext * execution_context_ = nullptr;
    ggml_context * ggml_ = nullptr;
    ggml_backend_buffer_t buffer_ = nullptr;
    ggml_cgraph * graph_ = nullptr;
    engine::core::TensorValue encoded_in_;
    engine::core::TensorValue speaker_in_;
    engine::core::TensorValue mu_bct_;
    engine::core::TensorValue speaker_out_;
};

std::vector<float> make_gaussian_full_noise(int64_t channels, int64_t frames, uint64_t seed) {
    return engine::sampling::generate_torch_cuda_randn(
        static_cast<size_t>(channels * frames),
        components::choose_seed(seed),
        engine::sampling::TorchRandnPrecision::Float32);
}

void apply_s3_trim_fade_inplace(std::vector<float> & waveform, int sample_rate) {
    const int64_t n_trim = sample_rate / 50;
    const int64_t fade_size = 2 * n_trim;
    if (waveform.empty() || n_trim <= 0) {
        return;
    }
    const int64_t keep = std::min<int64_t>(static_cast<int64_t>(waveform.size()), fade_size);
    for (int64_t i = 0; i < std::min<int64_t>(keep, n_trim); ++i) {
        waveform[static_cast<size_t>(i)] = 0.0f;
    }
    constexpr double kPi = 3.141592653589793238462643383279502884;
    for (int64_t i = n_trim; i < keep; ++i) {
        const double alpha = static_cast<double>(i - n_trim) / static_cast<double>(std::max<int64_t>(n_trim - 1, 1));
        const float fade = static_cast<float>((std::cos(kPi * (1.0 - alpha)) + 1.0) / 2.0);
        waveform[static_cast<size_t>(i)] *= fade;
    }
}

}  // namespace

struct S3GenSessionCache::State {
    explicit State(engine::core::BackendConfig backend_value)
        : backend(std::move(backend_value)),
          token_capacity_controller(runtime::GraphCapacityMode::Grow),
          flow_cache(backend) {}

    struct PreparedCapacity {
        int64_t token_capacity = 0;
        int64_t frame_capacity = 0;
        std::unique_ptr<S3TokenEmbeddingGraph> token_embedding;
        std::unique_ptr<S3Token2MelPrepareGraph> token2mel_prepare;
    };

    runtime::MappedGraphCapacityAdapter make_token_capacity_adapter(
        const S3FlowEncoderWeights & weights,
        const engine::core::BackendConfig & backend_config) {
        const int64_t quantum = token_capacity_quantum(backend_config);
        return runtime::MappedGraphCapacityAdapter(
            quantum,
            quantum,
            [quantum](int64_t request_size) {
                return round_up_capacity(request_size, quantum);
            },
            [this]() {
                std::vector<int64_t> capacities;
                if (prepared_capacity.token_embedding && prepared_capacity.token2mel_prepare) {
                    capacities.push_back(prepared_capacity.token_capacity);
                }
                return capacities;
            },
            [this, &weights](int64_t capacity) {
                prepare_token_capacity(weights, capacity);
            });
    }

    void prepare_token_capacity(
        const S3FlowEncoderWeights & weights,
        int64_t token_capacity) {
        if (token_capacity <= 0) {
            throw std::runtime_error("S3 token capacity must be positive");
        }
        if (prepared_capacity.token_embedding &&
            prepared_capacity.token2mel_prepare &&
            prepared_capacity.token_capacity == token_capacity) {
            return;
        }
        if (weights.execution_context == nullptr) {
            throw std::runtime_error("S3 token2mel prepare requires loaded backend weights");
        }
        prepared_capacity.token_capacity = token_capacity;
        prepared_capacity.frame_capacity = token_capacity * 2;
        prepared_capacity.token_embedding =
            std::make_unique<S3TokenEmbeddingGraph>(weights, token_capacity, *weights.execution_context);
        prepared_capacity.token2mel_prepare =
            std::make_unique<S3Token2MelPrepareGraph>(weights, prepared_capacity.frame_capacity, *weights.execution_context);
    }

    PreparedCapacity & prepared_bundle(
        const S3FlowEncoderWeights & weights,
        int64_t token_count,
        const engine::core::BackendConfig & backend_config) {
        auto adapter = make_token_capacity_adapter(weights, backend_config);
        token_capacity_controller.ensure_prepared(adapter, token_count);
        const int64_t selected_capacity = token_capacity_controller.select_capacity_for_run(adapter, token_count);
        if (prepared_capacity.token_capacity != selected_capacity) {
            throw std::runtime_error("S3 token capacity controller selected an unprepared capacity");
        }
        return prepared_capacity;
    }

    void release_pre_cfm_graphs() {
        prepared_capacity.token_embedding.reset();
        prepared_capacity.token2mel_prepare.reset();
        flow_cache.release_encoder_graphs();
    }

    void release_cfm_decoder_graphs() {
        flow_cache.release_decoder_graphs();
    }

    engine::core::BackendConfig backend;
    runtime::GraphCapacityController token_capacity_controller;
    S3FlowSessionCache flow_cache;
    PreparedCapacity prepared_capacity;
};

S3GenSessionCache::S3GenSessionCache(engine::core::BackendConfig backend)
    : state_(std::make_unique<State>(backend)) {}

S3GenSessionCache::~S3GenSessionCache() = default;
S3GenSessionCache::S3GenSessionCache(S3GenSessionCache &&) noexcept = default;
S3GenSessionCache & S3GenSessionCache::operator=(S3GenSessionCache &&) noexcept = default;

void S3GenSessionCache::release_runtime_graphs() {
    state_->release_pre_cfm_graphs();
    state_->release_cfm_decoder_graphs();
}

S3Token2MelOutputs compute_s3_token2mel_inference(
    S3GenSessionCache & cache,
    const S3FlowEncoderWeights & encoder_weights,
    const S3FlowDecoderWeights & decoder_weights,
    const EmbedReferenceOutputs & ref_dict,
    const std::vector<int32_t> & speech_tokens,
    int64_t speech_token_count,
    int64_t num_steps,
    float cfg_rate,
    bool cosine_schedule,
    const std::vector<float> & full_noise,
    uint64_t flow_seed,
    engine::core::BackendConfig backend,
    S3GenTimingBreakdown * timing) {
    constexpr int64_t batch = 1;
    constexpr int64_t token_dim = 512;
    constexpr int64_t mel_dim = 80;
    const int64_t prompt_tokens = ref_dict.prompt_token_count;
    const int64_t total_tokens = prompt_tokens + speech_token_count;
    std::vector<int32_t> concat_tokens;
    concat_tokens.reserve(static_cast<size_t>(total_tokens));
    concat_tokens.insert(concat_tokens.end(), ref_dict.prompt_tokens.begin(), ref_dict.prompt_tokens.end());
    concat_tokens.insert(concat_tokens.end(), speech_tokens.begin(), speech_tokens.begin() + static_cast<ptrdiff_t>(speech_token_count));

    auto & capacity_bundle = cache.state_->prepared_bundle(encoder_weights, total_tokens, backend);
    const int64_t token_capacity = capacity_bundle.token_capacity;
    const int64_t frame_capacity = capacity_bundle.frame_capacity;
    const auto padded_tokens = zero_pad_tokens(concat_tokens, token_capacity);

    const auto embed_started = std::chrono::steady_clock::now();
    auto token_emb = capacity_bundle.token_embedding->run(padded_tokens);
    zero_tail_btc(token_emb, total_tokens, token_capacity, token_dim);
    if (timing != nullptr) {
        timing->token2mel_embed_ms =
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - embed_started).count();
    }

    const auto encoder_started = std::chrono::steady_clock::now();
    const auto encoded =
        compute_s3_flow_encoder_forward(
            cache.state_->flow_cache,
            encoder_weights,
            token_emb,
            total_tokens,
            token_capacity,
            token_dim,
            backend);
    std::vector<float> encoded_hidden = encoded.hidden;
    if (static_cast<int64_t>(encoded_hidden.size()) != encoded.storage_frames * token_dim) {
        encoded_hidden = zero_pad_btc(encoded_hidden, encoded.frames, encoded.storage_frames, token_dim);
    }
    if (timing != nullptr) {
        timing->token2mel_encoder_ms =
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - encoder_started).count();
    }

    const auto mu_started = std::chrono::steady_clock::now();
    const int64_t total_frames = encoded.frames;
    const auto prepared = capacity_bundle.token2mel_prepare->run(
        encoded_hidden,
        ref_dict.embedding);
    auto cond = make_zero_padded_conditions_bct(ref_dict, total_frames);
    auto mask = make_prefix_mask_bct(batch, total_frames, total_frames);
    auto mu = slice_bct(prepared.mu_bct, frame_capacity, total_frames, mel_dim);
    if (timing != nullptr) {
        timing->token2mel_mu_ms =
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - mu_started).count();
    }

    std::vector<float> noise = full_noise.empty()
        ? make_gaussian_full_noise(mel_dim, total_frames, flow_seed)
        : full_noise;
    if (static_cast<int64_t>(noise.size()) != mel_dim * total_frames) {
        throw std::runtime_error(
            "S3Token2Mel full_noise size mismatch: got=" +
            std::to_string(noise.size()) +
            " expected=" +
            std::to_string(mel_dim * total_frames) +
            " total_frames=" +
            std::to_string(total_frames) +
            " prompt_tokens=" +
            std::to_string(prompt_tokens) +
            " speech_tokens=" +
            std::to_string(speech_token_count));
    }
    cache.state_->release_pre_cfm_graphs();
    const auto cfm_started = std::chrono::steady_clock::now();
    const auto mel_full = compute_s3_flow_cfm_euler(
        cache.state_->flow_cache,
        decoder_weights,
        noise,
        mask,
        mu,
        prepared.speaker,
        cond,
        batch,
        total_frames,
        frame_capacity,
        num_steps,
        cfg_rate,
        cosine_schedule,
        backend,
        timing == nullptr ? nullptr : &timing->token2mel_cfm);
    if (timing != nullptr) {
        timing->token2mel_cfm_ms =
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - cfm_started).count();
    }

    const int64_t prompt_frames = ref_dict.prompt_feat_frames;
    const int64_t output_frames = total_frames - prompt_frames;
    S3Token2MelOutputs outputs;
    outputs.channels = mel_dim;
    outputs.frames = output_frames;
    outputs.mel.assign(static_cast<size_t>(mel_dim * output_frames), 0.0f);
    const int64_t mel_stride = mel_full.storage_frames;
    for (int64_t dim = 0; dim < mel_dim; ++dim) {
        for (int64_t frame = 0; frame < output_frames; ++frame) {
            outputs.mel[static_cast<size_t>(dim * output_frames + frame)] =
                mel_full.mel[static_cast<size_t>(dim * mel_stride + prompt_frames + frame)];
        }
    }
    cache.state_->release_cfm_decoder_graphs();
    return outputs;
}

S3GenInferenceOutputs compute_s3gen_inference(
    S3GenSessionCache & cache,
    const S3FlowEncoderWeights & encoder_weights,
    const S3FlowDecoderWeights & decoder_weights,
    const engine::models::chatterbox::HiFTVocoderComponent & vocoder,
    const EmbedReferenceOutputs & ref_dict,
    const std::vector<int32_t> & speech_tokens,
    int64_t speech_token_count,
    int64_t num_steps,
    float cfg_rate,
    bool cosine_schedule,
    const std::vector<float> & full_noise,
    uint64_t flow_seed,
    uint64_t vocoder_seed,
    engine::core::BackendConfig backend,
    S3GenTimingBreakdown * timing) {
    const auto token2mel_started = std::chrono::steady_clock::now();
    const auto mel = compute_s3_token2mel_inference(
        cache,
        encoder_weights,
        decoder_weights,
        ref_dict,
        speech_tokens,
        speech_token_count,
        num_steps,
        cfg_rate,
        cosine_schedule,
        full_noise,
        flow_seed,
        backend,
        timing);
    if (timing != nullptr) {
        timing->token2mel_ms =
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - token2mel_started).count();
    }
    const uint64_t prior_noise_values = full_noise.empty()
        ? static_cast<uint64_t>(mel.channels * (ref_dict.prompt_feat_frames + mel.frames))
        : 0;
    const auto vocoder_started = std::chrono::steady_clock::now();
    const auto vocoder_outputs = vocoder.infer(
        mel.mel,
        1,
        mel.frames,
        vocoder_seed,
        prior_noise_values,
        {});
    if (timing != nullptr) {
        timing->vocoder_ms =
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - vocoder_started).count();
    }
    S3GenInferenceOutputs outputs;
    outputs.waveform = vocoder_outputs.waveform;
    outputs.samples = vocoder_outputs.samples;
    outputs.source = vocoder_outputs.source;
    outputs.source_channels = vocoder_outputs.source_channels;
    outputs.source_frames = vocoder_outputs.source_frames;
    outputs.mel = mel.mel;
    outputs.mel_channels = mel.channels;
    outputs.mel_frames = mel.frames;
    apply_s3_trim_fade_inplace(outputs.waveform, 24000);
    return outputs;
}

S3GenInferenceOutputs compute_s3gen_inference(
    S3GenSessionCache & cache,
    const S3FlowEncoderWeights & encoder_weights,
    const S3FlowDecoderWeights & decoder_weights,
    const engine::models::chatterbox::HiFTVocoderComponent & vocoder,
    const EmbedReferenceOutputs & ref_dict,
    const std::vector<int32_t> & speech_tokens,
    int64_t speech_token_count,
    engine::core::BackendConfig backend,
    S3GenTimingBreakdown * timing) {
    return compute_s3gen_inference(
        cache,
        encoder_weights,
        decoder_weights,
        vocoder,
        ref_dict,
        speech_tokens,
        speech_token_count,
        10,
        0.7f,
        true,
        {},
        0,
        0,
        backend,
        timing);
}

}  // namespace engine::models::chatterbox
