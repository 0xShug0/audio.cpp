#include "dit_blocks.inc"

namespace engine::models::echo_tts {
namespace {

// Joint attention over [self | text | speaker]. Two details differ from an
// ordinary cross-attention block and neither is caught by a shape check:
//
//  1. RoPE covers only the first half of the heads. Upstream's
//     _apply_rotary_half chunks along dim=-2, which is the head axis, so heads
//     0..7 rotate and 8..15 do not.
//  2. The text and speaker keys arrive already k_norm'd from the cache. They
//     must not be normalised again here, and they never receive RoPE.
core::TensorValue joint_attention(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const core::TensorValue & positions,
    const core::TensorValue & k_text,
    const core::TensorValue & v_text,
    const core::TensorValue & k_speaker,
    const core::TensorValue & v_speaker,
    const core::TensorValue & mask,
    const EchoJointAttentionWeights & weights,
    const EchoTtsConfig & config) {
    const int64_t D = config.model_size;
    const int64_t heads = config.num_heads;
    const int64_t head_dim = config.head_dim();
    const int64_t rope_heads = config.rope_heads();
    const int64_t batch = input.shape.dims[0];
    const int64_t seq = input.shape.dims[1];

    auto q = modules::LinearModule({D, D, false, GGML_PREC_F32}).build(ctx, input, weights.wq);
    auto k = modules::LinearModule({D, D, false, GGML_PREC_F32}).build(ctx, input, weights.wk);
    auto v = modules::LinearModule({D, D, false, GGML_PREC_F32}).build(ctx, input, weights.wv);

    q = head_rms_norm(ctx, reshape_heads(ctx, q, heads, head_dim), weights.q_norm.weight, config.norm_eps);
    k = head_rms_norm(ctx, reshape_heads(ctx, k, heads, head_dim), weights.k_norm.weight, config.norm_eps);
    v = reshape_heads(ctx, v, heads, head_dim);

    const modules::RoPEModule rope({head_dim, GGML_ROPE_TYPE_NORMAL, kRopeTheta});
    auto rotate_half = [&](const core::TensorValue & value) {
        auto front = modules::SliceModule({2, 0, rope_heads}).build(ctx, value);
        auto back = modules::SliceModule({2, rope_heads, heads - rope_heads}).build(ctx, value);
        front = rope.build(ctx, contiguous(ctx, front), positions);
        return modules::ConcatModule({2}).build(ctx, contiguous(ctx, front), contiguous(ctx, back));
    };
    q = rotate_half(q);
    k = rotate_half(k);

    // Broadcast the batch-1 cache across the CFG lanes, matching upstream's
    // _concat_kv_caches(cond, cond, cond).
    auto expand = [&](const core::TensorValue & value) {
        if (value.shape.dims[0] == batch) {
            return contiguous(ctx, value);
        }
        return contiguous(
            ctx,
            modules::RepeatModule({core::TensorShape::from_dims(
                                       {batch, value.shape.dims[1], heads, head_dim})})
                .build(ctx, contiguous(ctx, value)));
    };

    // Sequence-axis order is self, text, speaker; the mask uses the same order.
    auto k_all = modules::ConcatModule({1}).build(ctx, contiguous(ctx, k), expand(k_text));
    k_all = modules::ConcatModule({1}).build(ctx, contiguous(ctx, k_all), expand(k_speaker));
    auto v_all = modules::ConcatModule({1}).build(ctx, contiguous(ctx, v), expand(v_text));
    v_all = modules::ConcatModule({1}).build(ctx, contiguous(ctx, v_all), expand(v_speaker));

    // Flash attention avoids materialising the (lanes, heads, seq, keys) scores
    // tensor, which at 640 queries is 90-370 MB per attention and is the largest
    // per-request allocation in the model. It is safe here because the self
    // block of the mask is never masked, so no query row can be fully masked and
    // the softmax is always well defined. Set AUDIOCPP_ECHO_TTS_NO_FLASH=1 to
    // fall back to the explicit lowering for comparison.
    modules::ScaledDotProductAttentionConfig attn_config;
    attn_config.head_dim = head_dim;
    attn_config.lowering = echo_flash_disabled()
                               ? modules::ScaledDotProductAttentionLowering::Explicit
                               : modules::ScaledDotProductAttentionLowering::Flash;
    attn_config.precision = GGML_PREC_F32;
    attn_config.causality = modules::AttentionCausality::NonCausal;
    auto context = modules::ScaledDotProductAttentionModule(attn_config)
                       .build(ctx, to_bhsd(ctx, q), to_bhsd(ctx, k_all), to_bhsd(ctx, v_all), mask);

    context = core::reshape_tensor(
        ctx, contiguous(ctx, context), core::TensorShape::from_dims({batch, seq, D}));
    context = apply_attention_gate(ctx, context, input, weights.gate, D);
    return modules::LinearModule({D, D, false, GGML_PREC_F32}).build(ctx, context, weights.wo);
}

core::TensorValue dit_block(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & x,
    const core::TensorValue & cond_embed,
    const core::TensorValue & positions,
    const core::TensorValue & k_text,
    const core::TensorValue & v_text,
    const core::TensorValue & k_speaker,
    const core::TensorValue & v_speaker,
    const core::TensorValue & mask,
    const EchoDitBlockWeights & weights,
    const EchoTtsConfig & config) {
    auto attn_ada = adaln(
        ctx, x, cond_embed, weights.attention_adaln,
        config.model_size, config.adaln_rank, config.norm_eps);
    auto attn = joint_attention(
        ctx, attn_ada.normed, positions, k_text, v_text, k_speaker, v_speaker,
        mask, weights.attention, config);
    // gate is (lanes, 1, dim); it scales every sequence position.
    attn = broadcast_mul(ctx, attn, attn_ada.gate);
    auto hidden = modules::AddModule{}.build(ctx, x, attn);

    auto mlp_ada = adaln(
        ctx, hidden, cond_embed, weights.mlp_adaln,
        config.model_size, config.adaln_rank, config.norm_eps);
    auto mlp_out = mlp(ctx, mlp_ada.normed, weights.mlp, config.model_size, config.intermediate_size);
    mlp_out = broadcast_mul(ctx, mlp_out, mlp_ada.gate);
    return modules::AddModule{}.build(ctx, hidden, mlp_out);
}

// model.py::get_timestep_embedding, evaluated on the host because it depends
// only on t, which changes once per sampler step.
std::vector<float> timestep_embedding(float t, int64_t embed_size, int64_t lanes) {
    const int64_t half = embed_size / 2;
    std::vector<float> out(static_cast<size_t>(lanes * embed_size));
    for (int64_t i = 0; i < half; ++i) {
        const double freq =
            1000.0 * std::exp(-std::log(10000.0) * static_cast<double>(i) /
                              static_cast<double>(half));
        const double arg = static_cast<double>(t) * freq;
        const auto cos_v = static_cast<float>(std::cos(arg));
        const auto sin_v = static_cast<float>(std::sin(arg));
        for (int64_t lane = 0; lane < lanes; ++lane) {
            float * row = out.data() + lane * embed_size;
            row[i] = cos_v;
            row[half + i] = sin_v;
        }
    }
    return out;
}

// Parity debugging. Set AUDIOCPP_ECHO_TTS_DEBUG=1 to tap every DiT block output
// and the two encoder outputs, printing the same mean/std/min/max summary that
// tools/community_models/echo_tts_reference.py emits, so a C++ run can be
// compared block by block against the reference dump.
bool echo_debug_enabled() {
    static const bool enabled = [] {
        const char * value = std::getenv("AUDIOCPP_ECHO_TTS_DEBUG");
        return value != nullptr && value[0] != '\0' && value[0] != '0';
    }();
    return enabled;
}

void print_tensor_stats(const std::string & label, const std::vector<float> & values) {
    if (values.empty()) {
        std::fprintf(stderr, "  %-28s <empty>\n", label.c_str());
        return;
    }
    double sum = 0.0;
    double sum_sq = 0.0;
    float low = values[0];
    float high = values[0];
    for (const float value : values) {
        sum += value;
        sum_sq += static_cast<double>(value) * value;
        low = std::min(low, value);
        high = std::max(high, value);
    }
    const double mean = sum / static_cast<double>(values.size());
    const double variance = sum_sq / static_cast<double>(values.size()) - mean * mean;
    std::fprintf(
        stderr,
        "  %-28s mean=%+.6f std=%.6f min=%+.4f max=%+.4f  n=%zu\n",
        label.c_str(), mean, variance > 0.0 ? std::sqrt(variance) : 0.0,
        static_cast<double>(low), static_cast<double>(high), values.size());
}

std::vector<int32_t> iota_positions(int64_t count) {
    std::vector<int32_t> positions(static_cast<size_t>(count));
    for (int64_t i = 0; i < count; ++i) {
        positions[static_cast<size_t>(i)] = static_cast<int32_t>(i);
    }
    return positions;
}

}  // namespace

// --- runtime ------------------------------------------------------------

class EchoDitRuntime::Impl {
public:
    Impl(
        const EchoTtsConfig & config,
        const assets::TensorSource & source,
        const std::string & tensor_prefix,
        core::ExecutionContext & execution,
        assets::TensorStorageType matmul_storage_type)
        : config_(config),
          execution_(execution),
          backend_(execution.backend()),
          backend_type_(execution.backend_type()),
          threads_(std::max(1, execution.config().threads)),
          store_(backend_, backend_type_, "Echo-TTS DiT weights", kWeightContextBytes) {
        config_.validate();
        if (backend_ == nullptr) {
            throw std::runtime_error("Echo-TTS DiT backend initialization failed");
        }
        weights_ = load_dit_weights(config_, store_, source, tensor_prefix, matmul_storage_type);
        store_.upload();
    }

    ~Impl() { release_all(); }

    const EchoTtsConfig & config() const noexcept { return config_; }
    bool conditioning_ready() const noexcept { return conditioning_ready_; }

    void prepare_conditioning(const EchoConditioning & conditioning) {
        validate_conditioning(conditioning);

        // Any change in conditioning length invalidates every cached graph,
        // because all of them are built for fixed key counts.
        release_denoiser_graphs();
        release_conditioning_graph();
        release_kv_cache();

        text_length_ = conditioning.text_length;
        speaker_frames_ = conditioning.speaker_frames;
        speaker_tokens_ = speaker_frames_ / config_.speaker_patch_size;

        text_mask_ = conditioning.text_mask;
        speaker_mask_.assign(static_cast<size_t>(speaker_tokens_), 0.0F);
        for (int64_t i = 0; i < speaker_tokens_; ++i) {
            // model.py subsamples the speaker mask by the patch size before use.
            speaker_mask_[static_cast<size_t>(i)] =
                conditioning.speaker_mask[static_cast<size_t>(i * config_.speaker_patch_size)];
        }

        allocate_kv_cache();
        build_conditioning_graph();

        core::write_tensor_i32(text_ids_, conditioning.text_input_ids);
        core::write_tensor_f32(text_attn_mask_, make_text_self_mask());
        core::write_tensor_i32(text_positions_, iota_positions(text_length_));
        core::write_tensor_f32(speaker_latent_, conditioning.speaker_latent);
        core::write_tensor_i32(speaker_positions_, iota_positions(speaker_tokens_));

        core::set_backend_threads(backend_, threads_);
        const auto status = core::compute_graph(
            execution_, conditioning_graph_, conditioning_plan_, "echo_tts.conditioning");
        // compute_graph does not synchronise. On CUDA the copies into the
        // persistent KV cache are still in flight when it returns, so the
        // denoiser would read whatever the buffer happened to hold.
        ggml_backend_synchronize(backend_);
        if (status != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("Echo-TTS conditioning graph execution failed");
        }

        if (echo_debug_enabled()) {
            std::fprintf(stderr, "\n[echo_tts] conditioning: text_len=%lld speaker_frames=%lld "
                         "speaker_tokens=%lld\n",
                         static_cast<long long>(text_length_),
                         static_cast<long long>(speaker_frames_),
                         static_cast<long long>(speaker_tokens_));
            print_tensor_stats("kv_text.0.k", core::read_tensor_f32(kv_.k_text[0].tensor));
            print_tensor_stats("kv_text.23.k",
                               core::read_tensor_f32(kv_.k_text[kv_.k_text.size() - 1].tensor));
            print_tensor_stats("kv_speaker.0.k", core::read_tensor_f32(kv_.k_speaker[0].tensor));
            print_tensor_stats("kv_speaker.23.k",
                               core::read_tensor_f32(kv_.k_speaker[kv_.k_speaker.size() - 1].tensor));
            std::fflush(stderr);
        }

        // The encoders are not needed again for this request; only the cached
        // projections they wrote into the persistent buffer are.
        release_conditioning_graph();
        conditioning_ready_ = true;
        debug_printed_ = false;
    }

    std::vector<float> denoise(const std::vector<float> & x, float t, int lanes) {
        if (!conditioning_ready_) {
            throw std::runtime_error("Echo-TTS denoise() called before prepare_conditioning()");
        }
        if (lanes != 1 && lanes != 3) {
            throw std::runtime_error("Echo-TTS denoiser supports 1 or 3 CFG lanes");
        }
        const int64_t elements = sequence_length_ * config_.latent_size;
        if (static_cast<int64_t>(x.size()) != elements) {
            throw std::runtime_error("Echo-TTS denoiser received a mis-shaped latent");
        }

        auto & graph = lanes == 1 ? single_ : triple_;
        if (graph.graph == nullptr) {
            build_denoiser_graph(graph, lanes);
        }

        // x and the timestep embedding change every step and live in gallocr
        // space, so they are rewritten here. Positions and the mask are constant
        // and live in the graph's own persistent buffer; see DenoiserGraph.
        for (int lane = 0; lane < lanes; ++lane) {
            core::write_tensor_f32_slice(
                graph.x, static_cast<size_t>(lane * elements), x.data(), x.size());
        }
        core::write_tensor_f32(
            graph.timestep, timestep_embedding(t, config_.timestep_embed_size, lanes));

        core::set_backend_threads(backend_, threads_);
        const auto status =
            core::compute_graph(execution_, graph.graph, graph.plan, "echo_tts.denoise");
        // Must complete before the output is read back to the host; without this
        // the sampler integrates stale device memory.
        ggml_backend_synchronize(backend_);
        if (status != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("Echo-TTS denoiser graph execution failed");
        }
        if (echo_debug_enabled() && !debug_printed_) {
            debug_printed_ = true;
            std::fprintf(stderr, "[echo_tts] denoiser t=%.4f lanes=%d keys=%lld\n",
                         t, lanes,
                         static_cast<long long>(sequence_length_ + text_length_ + speaker_tokens_));
            print_tensor_stats("dit.v_pred", core::read_tensor_f32(graph.output));
            std::fflush(stderr);
        }
        return core::read_tensor_f32(graph.output);
    }

    void set_sequence_length(int64_t sequence_length) {
        if (sequence_length <= 0 || sequence_length > config_.max_sequence_length) {
            throw std::runtime_error("Echo-TTS sequence_length is out of range");
        }
        if (sequence_length != sequence_length_) {
            release_denoiser_graphs();
            sequence_length_ = sequence_length;
        }
    }

    std::vector<float> initial_noise(const EchoSamplerOptions & options) const {
        const size_t count =
            static_cast<size_t>(options.sequence_length * config_.latent_size);
        // Reproduces torch's CUDA Philox stream so a fixed seed is comparable
        // against the reference implementation.
        auto noise = sampling::generate_torch_cuda_randn(count, options.seed);
        if (echo_debug_enabled() && noise.size() >= 3) {
            std::fprintf(
                stderr,
                "[echo_tts] seed=%llu steps=%d cfg_text=%.2f cfg_speaker=%.2f "
                "truncation=%.2f\n  initial_noise first3=[% .6f % .6f % .6f]\n",
                static_cast<unsigned long long>(options.seed),
                options.num_steps,
                static_cast<double>(options.cfg_scale_text),
                static_cast<double>(options.cfg_scale_speaker),
                options.truncation_factor.has_value()
                    ? static_cast<double>(*options.truncation_factor) : 1.0,
                static_cast<double>(noise[0]), static_cast<double>(noise[1]),
                static_cast<double>(noise[2]));
            std::fflush(stderr);
        }
        return noise;
    }

    // Scales the cached speaker keys and values in place, matching
    // inference.py::_multiply_kv_cache. Done on a host round trip because the
    // cache is a plain backend buffer with no graph attached.
    void scale_speaker_kv(float scale, std::optional<int> max_layers) {
        const int64_t limit = max_layers.has_value()
                                  ? std::min<int64_t>(*max_layers, config_.num_layers)
                                  : config_.num_layers;
        for (int64_t layer = 0; layer < limit; ++layer) {
            scale_tensor_in_place(kv_.k_speaker[static_cast<size_t>(layer)], scale);
            scale_tensor_in_place(kv_.v_speaker[static_cast<size_t>(layer)], scale);
        }
    }

    void release_conditioning_graph() {
        conditioning_plan_.reset();
        if (conditioning_graph_ != nullptr) {
            core::release_backend_graph_resources(backend_type_, backend_, conditioning_graph_);
            conditioning_graph_ = nullptr;
        }
        if (conditioning_alloc_ != nullptr) {
            ggml_gallocr_free(conditioning_alloc_);
            conditioning_alloc_ = nullptr;
        }
        conditioning_ctx_.reset();
    }

private:
    struct DenoiserGraph {
        // Positions and the attention mask are constant once the graph is built,
        // but ggml_gallocr reclaims an input's memory after its last consumer and
        // reuses it for intermediates, so a value written into gallocr space does
        // not survive the next compute. They live in their own context and
        // backend buffer: pre-allocated tensors are skipped by gallocr, so they
        // are written once and read by every sampler step. This also avoids
        // rebuilding and re-uploading a multi-megabyte mask 40 times per chunk.
        GgmlContextPtr const_ctx;
        ggml_backend_buffer_t const_buffer = nullptr;
        GgmlContextPtr ctx;
        ggml_cgraph * graph = nullptr;
        ggml_gallocr_t alloc = nullptr;
        core::HostGraphPlan plan;
        core::TensorValue x;
        core::TensorValue timestep;
        core::TensorValue positions;
        core::TensorValue mask;
        ggml_tensor * output = nullptr;
        int lanes = 0;
    };

    struct KvCache {
        std::vector<core::TensorValue> k_text;
        std::vector<core::TensorValue> v_text;
        std::vector<core::TensorValue> k_speaker;
        std::vector<core::TensorValue> v_speaker;
    };

    void validate_conditioning(const EchoConditioning & c) const {
        if (c.text_length <= 0 || c.text_length > config_.max_text_length) {
            throw std::runtime_error("Echo-TTS text length out of range");
        }
        if (static_cast<int64_t>(c.text_input_ids.size()) != c.text_length ||
            static_cast<int64_t>(c.text_mask.size()) != c.text_length) {
            throw std::runtime_error("Echo-TTS text buffers disagree with text_length");
        }
        if (c.speaker_frames < config_.speaker_patch_size ||
            c.speaker_frames % config_.speaker_patch_size != 0) {
            throw std::runtime_error(
                "Echo-TTS speaker latent length must be a positive multiple of the patch size");
        }
        if (c.speaker_frames > config_.max_speaker_latent_length) {
            throw std::runtime_error("Echo-TTS speaker latent exceeds the trained maximum");
        }
        if (static_cast<int64_t>(c.speaker_latent.size()) !=
            c.speaker_frames * config_.latent_size) {
            throw std::runtime_error("Echo-TTS speaker latent has an unexpected element count");
        }
        if (static_cast<int64_t>(c.speaker_mask.size()) != c.speaker_frames) {
            throw std::runtime_error("Echo-TTS speaker mask disagrees with speaker_frames");
        }
    }

    // Bidirectional text encoder mask: padded key positions are suppressed for
    // every query row.
    std::vector<float> make_text_self_mask() const {
        std::vector<float> mask(static_cast<size_t>(text_length_ * text_length_), 0.0F);
        for (int64_t q = 0; q < text_length_; ++q) {
            float * row = mask.data() + q * text_length_;
            for (int64_t k = 0; k < text_length_; ++k) {
                row[k] = text_mask_[static_cast<size_t>(k)] > 0.5F ? 0.0F : kMaskedBias;
            }
        }
        return mask;
    }

    // Denoiser mask, laid out per lane as [self | text | speaker]. Lane 0 is
    // fully conditional, lane 1 drops text, lane 2 drops speaker, reproducing
    // upstream's concatenated cond/uncond masks.
    std::vector<float> make_denoiser_mask(int lanes) const {
        const int64_t keys = sequence_length_ + text_length_ + speaker_tokens_;
        std::vector<float> mask(
            static_cast<size_t>(static_cast<int64_t>(lanes) * sequence_length_ * keys), 0.0F);
        for (int lane = 0; lane < lanes; ++lane) {
            const bool text_on = lane != 1;
            const bool speaker_on = lane != 2;
            for (int64_t q = 0; q < sequence_length_; ++q) {
                float * row = mask.data() +
                              (static_cast<int64_t>(lane) * sequence_length_ + q) * keys;
                const float masked = echo_flash_disabled() ? kMaskedBias : kMaskedBiasF16;
                for (int64_t i = 0; i < text_length_; ++i) {
                    const bool keep = text_on && text_mask_[static_cast<size_t>(i)] > 0.5F;
                    row[sequence_length_ + i] = keep ? 0.0F : masked;
                }
                for (int64_t i = 0; i < speaker_tokens_; ++i) {
                    const bool keep = speaker_on && speaker_mask_[static_cast<size_t>(i)] > 0.5F;
                    row[sequence_length_ + text_length_ + i] = keep ? 0.0F : masked;
                }
            }
        }
        return mask;
    }

    void scale_tensor_in_place(const core::TensorValue & tensor, float scale) {
        auto values = core::read_tensor_f32(tensor.tensor);
        for (auto & value : values) {
            value *= scale;
        }
        core::write_tensor_f32(tensor, values);
    }

    // The cache lives in its own context and backend buffer so that both the
    // conditioning graph (which writes it) and the denoiser graphs (which read
    // it) can reference the same tensors as leaves.
    void allocate_kv_cache() {
        const int64_t heads = config_.num_heads;
        const int64_t head_dim = config_.head_dim();
        const size_t tensor_count = static_cast<size_t>(config_.num_layers) * 4;
        ggml_init_params params{
            ggml_tensor_overhead() * (tensor_count + 16), nullptr, true};
        kv_ctx_.reset(ggml_init(params));
        if (kv_ctx_ == nullptr) {
            throw std::runtime_error("Echo-TTS KV cache context initialization failed");
        }
        core::ModuleBuildContext ctx{kv_ctx_.get(), "echo_tts.kv_cache", backend_type_};
        // The cache is allocated at the widest lane count any graph will ask
        // for, so joint_attention's expand() finds dims[0] already equal to the
        // batch and short-circuits to a passthrough. The single-lane graph
        // reads lane 0, which is a contiguous prefix because the lane axis is
        // outermost in ggml's layout.
        kv_lanes_ = echo_kv_expand_disabled() ? 1 : kMaxCfgLanes;
        auto make = [&](int64_t tokens) {
            return core::make_tensor(
                ctx, GGML_TYPE_F32,
                core::TensorShape::from_dims({kv_lanes_, tokens, heads, head_dim}));
        };
        for (int64_t layer = 0; layer < config_.num_layers; ++layer) {
            kv_.k_text.push_back(make(text_length_));
            kv_.v_text.push_back(make(text_length_));
            kv_.k_speaker.push_back(make(speaker_tokens_));
            kv_.v_speaker.push_back(make(speaker_tokens_));
        }
        kv_buffer_ = ggml_backend_alloc_ctx_tensors(kv_ctx_.get(), backend_);
        if (kv_buffer_ == nullptr) {
            throw std::runtime_error("Echo-TTS KV cache buffer allocation failed");
        }
    }

    void build_conditioning_graph() {
        ggml_init_params params{kGraphArenaBytes, nullptr, true};
        conditioning_ctx_.reset(ggml_init(params));
        if (conditioning_ctx_ == nullptr) {
            throw std::runtime_error("Echo-TTS conditioning graph context initialization failed");
        }
        core::ModuleBuildContext ctx{conditioning_ctx_.get(), "echo_tts.conditioning", backend_type_};

        const int64_t TD = config_.text_model_size;
        const int64_t SD = config_.speaker_model_size;
        const int64_t D = config_.model_size;
        const int64_t heads = config_.num_heads;
        const int64_t head_dim = config_.head_dim();

        text_ids_ = core::make_tensor(
            ctx, GGML_TYPE_I32, core::TensorShape::from_dims({1, text_length_}));
        text_attn_mask_ = core::make_tensor(
            ctx, GGML_TYPE_F32, core::TensorShape::from_dims({1, 1, text_length_, text_length_}));
        text_positions_ = core::make_tensor(
            ctx, GGML_TYPE_I32, core::TensorShape::from_dims({text_length_}));
        speaker_latent_ = core::make_tensor(
            ctx, GGML_TYPE_F32,
            core::TensorShape::from_dims({1, speaker_frames_, config_.latent_size}));
        speaker_positions_ = core::make_tensor(
            ctx, GGML_TYPE_I32, core::TensorShape::from_dims({speaker_tokens_}));
        for (auto * input : {text_ids_.tensor, text_attn_mask_.tensor, text_positions_.tensor,
                             speaker_latent_.tensor, speaker_positions_.tensor}) {
            ggml_set_input(input);
        }

        // Text encoder: byte embedding, then bidirectional blocks.
        auto text_state = modules::EmbeddingModule({config_.text_vocab_size, TD})
                              .build(ctx, text_ids_, weights_.text_encoder.text_embedding);
        const std::optional<core::TensorValue> text_mask_opt{text_attn_mask_};
        for (const auto & block : weights_.text_encoder.blocks) {
            text_state = encoder_block(
                ctx, text_state, text_positions_, text_mask_opt, block,
                TD, config_.text_intermediate_size, config_.text_num_heads,
                config_.norm_eps, false);
        }
        text_state = rms_norm(ctx, text_state, weights_.text_norm.weight, config_.norm_eps);

        // Speaker encoder: patchify by folding groups of `patch_size` frames into
        // the feature axis, project, then causal blocks. The /6 scale is
        // upstream's activation-dynamics fix, not a normalisation.
        auto speaker_state = core::reshape_tensor(
            ctx,
            contiguous(ctx, speaker_latent_),
            core::TensorShape::from_dims(
                {1, speaker_tokens_, config_.latent_size * config_.speaker_patch_size}));
        speaker_state = modules::LinearModule(
                            {config_.latent_size * config_.speaker_patch_size, SD, true, GGML_PREC_F32})
                            .build(ctx, speaker_state, weights_.speaker_encoder.in_proj);
        speaker_state = core::wrap_tensor(
            ggml_scale(ctx.ggml, contiguous(ctx, speaker_state).tensor, 1.0F / 6.0F),
            speaker_state.shape,
            GGML_TYPE_F32);
        const std::optional<core::TensorValue> no_mask;
        for (const auto & block : weights_.speaker_encoder.blocks) {
            speaker_state = encoder_block(
                ctx, speaker_state, speaker_positions_, no_mask, block,
                SD, config_.speaker_intermediate_size, config_.speaker_num_heads,
                config_.norm_eps, true);
        }
        speaker_state = rms_norm(ctx, speaker_state, weights_.speaker_norm.weight, config_.norm_eps);

        conditioning_graph_ = ggml_new_graph_custom(conditioning_ctx_.get(), 1048576, false);

        // Project the encoder outputs into each block's key/value space and copy
        // the result into the persistent cache. Keys are k_norm'd here, exactly
        // once, so the denoiser must not normalise them again.
        for (int64_t layer = 0; layer < config_.num_layers; ++layer) {
            const size_t index = static_cast<size_t>(layer);
            const auto & attn = weights_.blocks[index].attention;
            auto project = [&](const core::TensorValue & state,
                               const modules::LinearWeights & weight,
                               int64_t in_dim,
                               int64_t tokens,
                               bool normalise) {
                auto value = modules::LinearModule({in_dim, D, false, GGML_PREC_F32})
                                 .build(ctx, state, weight);
                value = core::reshape_tensor(
                    ctx, contiguous(ctx, value),
                    core::TensorShape::from_dims({1, tokens, heads, head_dim}));
                if (normalise) {
                    value = head_rms_norm(ctx, value, attn.k_norm.weight, config_.norm_eps);
                }
                if (kv_lanes_ > 1) {
                    // Broadcast to the cache's lane count once, here, instead of
                    // once per layer per sampler step inside joint_attention.
                    // Upstream's _concat_kv_caches(cond, cond, cond) is the same
                    // operation; only its position in the schedule changes.
                    value = modules::RepeatModule(
                                {core::TensorShape::from_dims(
                                    {kv_lanes_, tokens, heads, head_dim})})
                                .build(ctx, contiguous(ctx, value));
                }
                return value;
            };
            struct Slot {
                core::TensorValue source;
                core::TensorValue destination;
            };
            const Slot slots[] = {
                {project(text_state, attn.wk_text, TD, text_length_, true), kv_.k_text[index]},
                {project(text_state, attn.wv_text, TD, text_length_, false), kv_.v_text[index]},
                {project(speaker_state, attn.wk_speaker, SD, speaker_tokens_, true), kv_.k_speaker[index]},
                {project(speaker_state, attn.wv_speaker, SD, speaker_tokens_, false), kv_.v_speaker[index]},
            };
            for (const auto & slot : slots) {
                auto * copy = ggml_cpy(
                    ctx.ggml, contiguous(ctx, slot.source).tensor, slot.destination.tensor);
                ggml_build_forward_expand(conditioning_graph_, copy);
            }
        }

        conditioning_alloc_ = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend_));
        if (conditioning_alloc_ == nullptr ||
            !ggml_gallocr_reserve(conditioning_alloc_, conditioning_graph_) ||
            !ggml_gallocr_alloc_graph(conditioning_alloc_, conditioning_graph_)) {
            throw std::runtime_error("Echo-TTS conditioning graph allocation failed");
        }
        core::prepare_host_graph_plan(execution_, conditioning_graph_, conditioning_plan_);
    }

    // Reconciles the cache's lane count with the graph's.
    //
    // With pre-expansion on, the cache is allocated at kMaxCfgLanes and the
    // single-lane graph reads a prefix of it; the lane axis is outermost, so
    // that is a view with no copy.
    //
    // With pre-expansion off (AUDIOCPP_ECHO_TTS_NO_KV_EXPAND=1) the cache is
    // batch-1 and the three-lane graph is *wider* than it. Narrowing is not
    // possible and not wanted: returning the cache unchanged leaves
    // joint_attention's expand() to broadcast it per step, which is exactly the
    // pre-patch behaviour the flag exists to restore.
    core::TensorValue kv_for_lanes(
        core::ModuleBuildContext & ctx, const core::TensorValue & cached, int lanes) const {
        if (cached.shape.dims[0] <= static_cast<int64_t>(lanes)) {
            return cached;
        }
        return modules::SliceModule({0, 0, static_cast<int64_t>(lanes)}).build(ctx, cached);
    }

    void build_denoiser_graph(DenoiserGraph & target, int lanes) {
        ggml_init_params params{kGraphArenaBytes, nullptr, true};
        target.ctx.reset(ggml_init(params));
        if (target.ctx == nullptr) {
            throw std::runtime_error("Echo-TTS denoiser graph context initialization failed");
        }
        core::ModuleBuildContext ctx{target.ctx.get(), "echo_tts.denoise", backend_type_};

        const int64_t D = config_.model_size;
        const int64_t lane_count = lanes;
        const int64_t keys = sequence_length_ + text_length_ + speaker_tokens_;

        target.lanes = lanes;
        target.x = core::make_tensor(
            ctx, GGML_TYPE_F32,
            core::TensorShape::from_dims({lane_count, sequence_length_, config_.latent_size}));
        target.timestep = core::make_tensor(
            ctx, GGML_TYPE_F32,
            core::TensorShape::from_dims({lane_count, config_.timestep_embed_size}));
        ggml_init_params const_params{ggml_tensor_overhead() * 8, nullptr, true};
        target.const_ctx.reset(ggml_init(const_params));
        if (target.const_ctx == nullptr) {
            throw std::runtime_error("Echo-TTS denoiser constant context initialization failed");
        }
        core::ModuleBuildContext const_ctx{
            target.const_ctx.get(), "echo_tts.denoise.const", backend_type_};
        target.positions = core::make_tensor(
            const_ctx, GGML_TYPE_I32, core::TensorShape::from_dims({sequence_length_}));
        // ggml_flash_attn_ext requires an F16 mask; the explicit path takes F32.
        target.mask = core::make_tensor(
            const_ctx, echo_flash_disabled() ? GGML_TYPE_F32 : GGML_TYPE_F16,
            core::TensorShape::from_dims({lane_count, 1, sequence_length_, keys}));
        target.const_buffer =
            ggml_backend_alloc_ctx_tensors(target.const_ctx.get(), backend_);
        if (target.const_buffer == nullptr) {
            throw std::runtime_error("Echo-TTS denoiser constant buffer allocation failed");
        }
        core::write_tensor_i32(target.positions, iota_positions(sequence_length_));
        if (echo_flash_disabled()) {
            core::write_tensor_f32(target.mask, make_denoiser_mask(lanes));
        } else {
            core::write_tensor_f16(target.mask, make_denoiser_mask(lanes));
        }

        for (auto * input : {target.x.tensor, target.timestep.tensor}) {
            ggml_set_input(input);
        }

        // cond_module: Linear, SiLU, Linear, SiLU, Linear -> 3 * model_size.
        auto cond = modules::LinearModule({config_.timestep_embed_size, D, false, GGML_PREC_F32})
                        .build(ctx, target.timestep, weights_.cond_0);
        cond = modules::SiluModule{}.build(ctx, cond);
        cond = modules::LinearModule({D, D, false, GGML_PREC_F32}).build(ctx, cond, weights_.cond_2);
        cond = modules::SiluModule{}.build(ctx, cond);
        cond = modules::LinearModule({D, D * 3, false, GGML_PREC_F32}).build(ctx, cond, weights_.cond_4);
        // Insert the sequence axis so the conditioning broadcasts over steps.
        cond = core::reshape_tensor(
            ctx, contiguous(ctx, cond), core::TensorShape::from_dims({lane_count, 1, D * 3}));

        auto hidden = modules::LinearModule({config_.latent_size, D, true, GGML_PREC_F32})
                          .build(ctx, target.x, weights_.in_proj);
        for (int64_t layer = 0; layer < config_.num_layers; ++layer) {
            const size_t index = static_cast<size_t>(layer);
            hidden = dit_block(
                ctx, hidden, cond, target.positions,
                kv_for_lanes(ctx, kv_.k_text[index], lanes),
                kv_for_lanes(ctx, kv_.v_text[index], lanes),
                kv_for_lanes(ctx, kv_.k_speaker[index], lanes),
                kv_for_lanes(ctx, kv_.v_speaker[index], lanes),
                target.mask, weights_.blocks[index], config_);
        }
        hidden = rms_norm(ctx, hidden, weights_.out_norm.weight, config_.norm_eps);
        hidden = modules::LinearModule({D, config_.latent_size, true, GGML_PREC_F32})
                     .build(ctx, hidden, weights_.out_proj);

        target.output = contiguous(ctx, hidden).tensor;
        ggml_set_output(target.output);
        target.graph = ggml_new_graph_custom(target.ctx.get(), 1048576, false);
        ggml_build_forward_expand(target.graph, target.output);

        target.alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend_));
        if (target.alloc == nullptr ||
            !ggml_gallocr_reserve(target.alloc, target.graph) ||
            !ggml_gallocr_alloc_graph(target.alloc, target.graph)) {
            throw std::runtime_error("Echo-TTS denoiser graph allocation failed");
        }
        core::prepare_host_graph_plan(execution_, target.graph, target.plan);
    }

    void release_denoiser_graph(DenoiserGraph & target) {
        target.plan.reset();
        if (target.graph != nullptr) {
            core::release_backend_graph_resources(backend_type_, backend_, target.graph);
            target.graph = nullptr;
        }
        if (target.alloc != nullptr) {
            ggml_gallocr_free(target.alloc);
            target.alloc = nullptr;
        }
        target.ctx.reset();
        if (target.const_buffer != nullptr) {
            ggml_backend_buffer_free(target.const_buffer);
            target.const_buffer = nullptr;
        }
        target.const_ctx.reset();
        target.output = nullptr;
        target.lanes = 0;
    }

    void release_denoiser_graphs() {
        release_denoiser_graph(single_);
        release_denoiser_graph(triple_);
    }

    void release_kv_cache() {
        kv_ = KvCache{};
        if (kv_buffer_ != nullptr) {
            ggml_backend_buffer_free(kv_buffer_);
            kv_buffer_ = nullptr;
        }
        kv_ctx_.reset();
        conditioning_ready_ = false;
    }

    void release_all() {
        release_denoiser_graphs();
        release_conditioning_graph();
        release_kv_cache();
    }

    EchoTtsConfig config_;
    core::ExecutionContext & execution_;
    ggml_backend_t backend_ = nullptr;
    core::BackendType backend_type_ = core::BackendType::Cpu;
    int threads_ = 1;
    core::BackendWeightStore store_;
    EchoDitWeights weights_;

    bool conditioning_ready_ = false;
    bool debug_printed_ = false;
    int64_t text_length_ = 0;
    int64_t speaker_frames_ = 0;
    int64_t speaker_tokens_ = 0;
    int64_t sequence_length_ = 640;
    int64_t kv_lanes_ = 1;
    std::vector<float> text_mask_;
    std::vector<float> speaker_mask_;

    GgmlContextPtr kv_ctx_;
    ggml_backend_buffer_t kv_buffer_ = nullptr;
    KvCache kv_;

    GgmlContextPtr conditioning_ctx_;
    ggml_cgraph * conditioning_graph_ = nullptr;
    ggml_gallocr_t conditioning_alloc_ = nullptr;
    core::HostGraphPlan conditioning_plan_;
    core::TensorValue text_ids_;
    core::TensorValue text_attn_mask_;
    core::TensorValue text_positions_;
    core::TensorValue speaker_latent_;
    core::TensorValue speaker_positions_;

    DenoiserGraph single_;
    DenoiserGraph triple_;
};

EchoDitRuntime::EchoDitRuntime(
    const EchoTtsConfig & config,
    const assets::TensorSource & source,
    const std::string & tensor_prefix,
    core::ExecutionContext & execution,
    assets::TensorStorageType matmul_storage_type)
    : impl_(std::make_unique<Impl>(config, source, tensor_prefix, execution, matmul_storage_type)) {}

EchoDitRuntime::~EchoDitRuntime() = default;

const EchoTtsConfig & EchoDitRuntime::config() const noexcept { return impl_->config(); }

void EchoDitRuntime::prepare_conditioning(const EchoConditioning & conditioning) {
    impl_->prepare_conditioning(conditioning);
}

std::vector<float> EchoDitRuntime::denoise_once(const std::vector<float> & x, float t, int lanes) {
    if (!impl_->conditioning_ready()) {
        throw std::runtime_error("Echo-TTS denoise_once() called before prepare_conditioning()");
    }
    if (lanes < 1) {
        throw std::runtime_error("Echo-TTS denoise_once() requires at least one lane");
    }
    // `x` is always ONE lane. `lanes` selects how many velocity fields the
    // denoiser returns -- see sampler.cpp, which calls denoise(x_t, t, 3) with
    // an x_t of exactly `elements`, then expects elements * 3 back. Dividing
    // the input by `lanes` here would set a sequence length 3x too small.
    const int64_t latent_size = impl_->config().latent_size;
    if (latent_size <= 0 || static_cast<int64_t>(x.size()) % latent_size != 0) {
        throw std::runtime_error("Echo-TTS denoise_once() received a mis-shaped latent buffer");
    }
    impl_->set_sequence_length(static_cast<int64_t>(x.size()) / latent_size);
    return impl_->denoise(x, t, lanes);
}

std::vector<float> EchoDitRuntime::sample(const EchoSamplerOptions & options) {
    if (!impl_->conditioning_ready()) {
        throw std::runtime_error("Echo-TTS sample() called before prepare_conditioning()");
    }
    impl_->set_sequence_length(options.sequence_length);

    Impl * impl = impl_.get();
    if (options.speaker_kv_scale.has_value()) {
        impl->scale_speaker_kv(*options.speaker_kv_scale, options.speaker_kv_max_layers);
    }
    std::function<void()> on_kv_rescale;
    if (options.speaker_kv_scale.has_value()) {
        const float inverse = 1.0F / *options.speaker_kv_scale;
        const auto max_layers = options.speaker_kv_max_layers;
        on_kv_rescale = [impl, inverse, max_layers]() {
            impl->scale_speaker_kv(inverse, max_layers);
        };
    }

    auto denoise = [impl](const std::vector<float> & x, float t, int lanes) {
        return impl->denoise(x, t, lanes);
    };
    return run_euler_sampler(
        options,
        options.sequence_length,
        impl->config().latent_size,
        impl->initial_noise(options),
        denoise,
        on_kv_rescale);
}

void EchoTtsConfig::validate() const {
    if (model_size % num_heads != 0 || text_model_size % text_num_heads != 0 ||
        speaker_model_size % speaker_num_heads != 0) {
        throw std::runtime_error("Echo-TTS head counts must divide their model sizes");
    }
    if (num_heads % 2 != 0) {
        throw std::runtime_error("Echo-TTS requires an even head count for half-rotary attention");
    }
    if (timestep_embed_size % 2 != 0) {
        throw std::runtime_error("Echo-TTS timestep embedding size must be even");
    }
    if (latent_size <= 0 || speaker_patch_size <= 0) {
        throw std::runtime_error("Echo-TTS latent and patch sizes must be positive");
    }
}

}  // namespace engine::models::echo_tts
