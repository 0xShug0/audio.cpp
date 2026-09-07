#include "engine/community_models/sopro_tts/speaker_encoder.h"

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/audio/dsp.h"
#include "engine/framework/core/backend.h"
#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/core/execution_context.h"
#include "engine/framework/modules/activation_modules.h"
#include "engine/framework/modules/conv_modules.h"
#include "engine/framework/modules/norm_modules.h"
#include "engine/framework/modules/primitive_modules.h"
#include "engine/framework/modules/streaming_conv_modules.h"
#include "engine/framework/modules/structural_modules.h"
#include "engine/framework/modules/weight_binding.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace engine::community_models::sopro_tts {

// A two-layer MLP with SiLU between, i.e. nn.Sequential(Linear, SiLU,
// Identity, Linear) -> keys "<prefix>.0" and "<prefix>.3".
struct SoproSpeakerHeadWeights {
    std::vector<float> fc1_weight;  // [hidden, in]
    std::vector<float> fc1_bias;
    std::vector<float> fc2_weight;  // [out, hidden]
    std::vector<float> fc2_bias;
    int64_t in_features = 0;
    int64_t hidden = 0;
    int64_t out_features = 0;
};

struct SoproSpeakerResBlockWeights {
    engine::modules::NormWeights norm1;
    engine::modules::Conv1dWeights pw_in;
    engine::modules::DepthwiseConv1dWeights dw;
    engine::modules::NormWeights norm2;
    engine::modules::Conv1dWeights se_reduce;
    engine::modules::Conv1dWeights se_expand;
    engine::modules::Conv1dWeights pw_out;
    int64_t channels = 0;
    int64_t se_hidden = 0;
    int dilation = 1;
};

struct SoproSpeakerStageWeights {
    engine::modules::Conv1dWeights transition_conv;
    engine::modules::NormWeights transition_norm;
    int transition_stride = 1;
    int64_t out_channels = 0;
    std::vector<SoproSpeakerResBlockWeights> blocks;
};

struct SoproSpeakerWeights {
    std::shared_ptr<engine::core::BackendWeightStore> store;
    engine::modules::Conv1dWeights stem_conv;
    engine::modules::NormWeights stem_norm;
    std::vector<SoproSpeakerStageWeights> stages;
    engine::modules::Conv1dWeights fuse_conv;
    engine::modules::NormWeights fuse_norm;
    // Host-side pooling heads.
    std::vector<float> attn_conv1_weight;  // [attn_hidden, channels]
    std::vector<float> attn_conv1_bias;
    std::vector<float> attn_conv2_weight;  // [1, attn_hidden]
    std::vector<float> attn_conv2_bias;
    SoproSpeakerHeadWeights id_head;
    SoproSpeakerHeadWeights style_head;
    SoproSpeakerHeadWeights style_ctrl_head;
    // torchaudio MelSpectrogram buffers.
    std::vector<float> analysis_window;   // win_length taps
    std::vector<float> mel_filterbank;    // [freq_bins, n_mels]
};

namespace {

namespace binding = engine::modules::binding;

constexpr float kGroupNormEps = 1.0e-5F;  // torch.nn.GroupNorm default
constexpr float kLayerNormEps = 1.0e-5F;  // torch.nn.functional.layer_norm default

// SOPRO_DUMP_DIR: raw f32 dumps for stage comparison against the reference.
void dump(const std::string & name, const std::vector<float> & values) {
    const char * dir = std::getenv("SOPRO_DUMP_DIR");
    if (dir == nullptr) {
        return;
    }
    std::FILE * fh = std::fopen((std::string(dir) + "/" + name).c_str(), "wb");
    if (fh != nullptr) {
        std::fwrite(values.data(), sizeof(float), values.size(), fh);
        std::fclose(fh);
    }
}

struct GgmlContextDeleter {
    void operator()(ggml_context * ctx) const noexcept {
        if (ctx != nullptr) {
            ggml_free(ctx);
        }
    }
};

engine::modules::NormWeights group_norm(
    engine::core::BackendWeightStore & store,
    const engine::assets::TensorSource & source,
    const std::string & prefix,
    int64_t channels) {
    return binding::norm_from_source(store, source, prefix, channels);
}

SoproSpeakerHeadWeights load_head(
    const engine::assets::TensorSource & source,
    const std::string & prefix,
    int64_t in_features,
    int64_t hidden,
    int64_t out_features) {
    SoproSpeakerHeadWeights out;
    out.in_features = in_features;
    out.hidden = hidden;
    out.out_features = out_features;
    out.fc1_weight = source.require_f32(prefix + ".0.weight", {hidden, in_features});
    out.fc1_bias = source.require_f32(prefix + ".0.bias", {hidden});
    out.fc2_weight = source.require_f32(prefix + ".3.weight", {out_features, hidden});
    out.fc2_bias = source.require_f32(prefix + ".3.bias", {out_features});
    return out;
}

std::vector<float> apply_head(const SoproSpeakerHeadWeights & head, const std::vector<float> & input) {
    if (static_cast<int64_t>(input.size()) != head.in_features) {
        throw std::runtime_error("Sopro speaker head input size mismatch");
    }
    std::vector<float> hidden(static_cast<size_t>(head.hidden), 0.0F);
    for (int64_t h = 0; h < head.hidden; ++h) {
        double sum = head.fc1_bias[static_cast<size_t>(h)];
        const float * row = head.fc1_weight.data() + static_cast<size_t>(h * head.in_features);
        for (int64_t i = 0; i < head.in_features; ++i) {
            sum += static_cast<double>(row[i]) * static_cast<double>(input[static_cast<size_t>(i)]);
        }
        const auto value = static_cast<float>(sum);
        hidden[static_cast<size_t>(h)] = value / (1.0F + std::exp(-value));  // SiLU
    }
    std::vector<float> out(static_cast<size_t>(head.out_features), 0.0F);
    for (int64_t o = 0; o < head.out_features; ++o) {
        double sum = head.fc2_bias[static_cast<size_t>(o)];
        const float * row = head.fc2_weight.data() + static_cast<size_t>(o * head.hidden);
        for (int64_t h = 0; h < head.hidden; ++h) {
            sum += static_cast<double>(row[h]) * static_cast<double>(hidden[static_cast<size_t>(h)]);
        }
        out[static_cast<size_t>(o)] = static_cast<float>(sum);
    }
    return out;
}

std::shared_ptr<const SoproSpeakerWeights> load_speaker_weights(
    ggml_backend_t backend,
    engine::core::BackendType backend_type,
    const engine::assets::TensorSource & source,
    const SoproSpeakerEncoderConfig & config,
    size_t weight_context_bytes,
    engine::assets::TensorStorageType matmul_storage_type,
    engine::assets::TensorStorageType conv_storage_type) {
    (void) matmul_storage_type;
    auto weights = std::make_shared<SoproSpeakerWeights>();
    require_frontend_buffers(
        source, "speaker encoder",
        {"frontend.mel.spectrogram.window",
         "frontend.mel.mel_scale.fb"});
    weights->store = std::make_shared<engine::core::BackendWeightStore>(
        backend, backend_type, "sopro_tts.speaker_encoder.weights", weight_context_bytes);
    auto & store = *weights->store;

    weights->stem_conv = binding::conv1d_from_source(
        store, source, "stem.0.conv", conv_storage_type,
        config.stem_channels, config.n_mels, 5, true);
    weights->stem_norm = group_norm(store, source, "stem.1", config.stem_channels);

    int64_t in_channels = config.stem_channels;
    int64_t fused_in = 0;
    weights->stages.reserve(config.stage_channels.size());
    for (size_t stage = 0; stage < config.stage_channels.size(); ++stage) {
        SoproSpeakerStageWeights out;
        out.out_channels = config.stage_channels[stage];
        out.transition_stride = stage == 0 ? 2 : 1;
        const std::string transition = "transitions." + std::to_string(stage);
        out.transition_conv = binding::conv1d_from_source(
            store, source, transition + ".conv", conv_storage_type,
            out.out_channels, in_channels, 3, true);
        out.transition_norm = group_norm(store, source, transition + ".norm", out.out_channels);
        const int64_t blocks = config.blocks_per_stage[stage];
        out.blocks.reserve(static_cast<size_t>(blocks));
        for (int64_t block = 0; block < blocks; ++block) {
            const std::string prefix =
                "stages." + std::to_string(stage) + "." + std::to_string(block);
            SoproSpeakerResBlockWeights weights_block;
            weights_block.channels = out.out_channels;
            weights_block.dilation = static_cast<int>(
                config.dilation_cycle[static_cast<size_t>(block) % config.dilation_cycle.size()]);
            weights_block.se_hidden = std::max<int64_t>(8, out.out_channels / config.se_reduction);
            weights_block.norm1 = group_norm(store, source, prefix + ".norm1", out.out_channels);
            weights_block.pw_in = binding::conv1d_from_source(
                store, source, prefix + ".pw_in", conv_storage_type,
                out.out_channels * 2, out.out_channels, 1, true);
            weights_block.dw = binding::depthwise_conv1d_from_source(
                store, source, prefix + ".dw.conv", conv_storage_type,
                out.out_channels, config.depthwise_kernel_size, true);
            weights_block.norm2 = group_norm(store, source, prefix + ".norm2", out.out_channels);
            weights_block.se_reduce = binding::conv1d_from_source(
                store, source, prefix + ".se.net.1", conv_storage_type,
                weights_block.se_hidden, out.out_channels, 1, true);
            weights_block.se_expand = binding::conv1d_from_source(
                store, source, prefix + ".se.net.3", conv_storage_type,
                out.out_channels, weights_block.se_hidden, 1, true);
            weights_block.pw_out = binding::conv1d_from_source(
                store, source, prefix + ".pw_out", conv_storage_type,
                out.out_channels, out.out_channels, 1, true);
            out.blocks.push_back(std::move(weights_block));
        }
        fused_in += out.out_channels;
        in_channels = out.out_channels;
        weights->stages.push_back(std::move(out));
    }

    const int64_t head_channels = config.stage_channels.back();
    weights->fuse_conv = binding::conv1d_from_source(
        store, source, "fuse.0", conv_storage_type, head_channels, fused_in, 1, true);
    weights->fuse_norm = group_norm(store, source, "fuse.1", head_channels);

    weights->attn_conv1_weight = source.require_f32(
        "id_pool.attn.0.weight", {config.attn_hidden, head_channels, 1});
    weights->attn_conv1_bias = source.require_f32("id_pool.attn.0.bias", {config.attn_hidden});
    weights->attn_conv2_weight = source.require_f32(
        "id_pool.attn.2.weight", {1, config.attn_hidden, 1});
    weights->attn_conv2_bias = source.require_f32("id_pool.attn.2.bias", {1});
    weights->id_head = load_head(
        source, "id_head", head_channels * 2, config.id_head_hidden, config.id_emb_dim);
    weights->style_head = load_head(
        source, "style_head", head_channels * 4, config.style_head_hidden, config.style_emb_dim);
    weights->style_ctrl_head = load_head(
        source, "style_ctrl_head", head_channels * 4, config.style_head_hidden, config.style_ctrl_dim);

    weights->analysis_window = source.require_f32(
        "frontend.mel.spectrogram.window", {config.win_length});
    weights->mel_filterbank = source.require_f32(
        "frontend.mel.mel_scale.fb", {config.n_fft / 2 + 1, config.n_mels});

    store.upload();
    return weights;
}

engine::core::TensorValue build_res_block(
    engine::core::ModuleBuildContext & ctx,
    const engine::core::TensorValue & input,
    const SoproSpeakerResBlockWeights & weights,
    int64_t kernel_size,
    int64_t frames) {
    const int64_t channels = weights.channels;
    auto hidden = engine::modules::GroupNormModule({channels, 1, kGroupNormEps, true, true})
                      .build(ctx, input, weights.norm1);
    hidden = engine::modules::Conv1dModule({channels, channels * 2, 1, 1, 0, 1, true})
                 .build(ctx, hidden, weights.pw_in);
    // chunk(2, dim=1): the gate multiplies the first half by sigmoid(second).
    auto gate_a = engine::modules::SliceModule({1, 0, channels}).build(ctx, hidden);
    auto gate_b = engine::modules::SliceModule({1, channels, channels}).build(ctx, hidden);
    gate_b = engine::modules::SigmoidModule{}.build(ctx, gate_b);
    hidden = engine::modules::MulModule{}.build(ctx, gate_a, gate_b);
    hidden = engine::modules::DepthwiseConv1dModule({
        channels, kernel_size, 1,
        static_cast<int>(weights.dilation * (kernel_size - 1) / 2), weights.dilation, true,
    }).build(ctx, hidden, weights.dw);
    hidden = engine::modules::GroupNormModule({channels, 1, kGroupNormEps, true, true})
                 .build(ctx, hidden, weights.norm2);
    hidden = engine::modules::SiluModule{}.build(ctx, hidden);
    // SqueezeExcite1d: global average pool -> 1x1 bottleneck -> sigmoid gate.
    auto pooled = engine::modules::ReduceMeanModule({2}).build(ctx, hidden);
    pooled = engine::modules::Conv1dModule({channels, weights.se_hidden, 1, 1, 0, 1, true})
                 .build(ctx, pooled, weights.se_reduce);
    pooled = engine::modules::SiluModule{}.build(ctx, pooled);
    pooled = engine::modules::Conv1dModule({weights.se_hidden, channels, 1, 1, 0, 1, true})
                 .build(ctx, pooled, weights.se_expand);
    pooled = engine::modules::SigmoidModule{}.build(ctx, pooled);
    auto scale = engine::modules::RepeatModule({
        engine::core::TensorShape::from_dims({1, channels, frames})}).build(ctx, pooled);
    hidden = engine::modules::MulModule{}.build(ctx, hidden, scale);
    hidden = engine::modules::Conv1dModule({channels, channels, 1, 1, 0, 1, true})
                 .build(ctx, hidden, weights.pw_out);
    return engine::modules::AddModule{}.build(ctx, input, hidden);
}

}  // namespace

struct SoproSpeakerGraph {
    SoproSpeakerGraph(
        ggml_backend_t backend_in,
        engine::core::BackendType backend_type,
        size_t graph_context_bytes,
        const SoproSpeakerEncoderConfig & config,
        std::shared_ptr<const SoproSpeakerWeights> weights_in,
        int64_t frames_in)
        : backend(backend_in),
          weights(std::move(weights_in)),
          frames(frames_in),
          mel_bins(config.n_mels) {
        if (backend == nullptr || weights == nullptr) {
            throw std::runtime_error("Sopro speaker encoder graph requires a backend and weights");
        }
        if (frames <= 0) {
            throw std::runtime_error("Sopro speaker encoder graph requires a positive frame count");
        }
        ggml_init_params params{graph_context_bytes, nullptr, true};
        ctx.reset(ggml_init(params));
        if (ctx == nullptr) {
            throw std::runtime_error("failed to initialize the Sopro speaker encoder graph context");
        }
        engine::core::ModuleBuildContext build_ctx{ctx.get(), "sopro_tts.speaker_encoder", backend_type};
        const auto shape = engine::core::TensorShape::from_dims({1, mel_bins, frames});
        input = engine::core::make_tensor(build_ctx, GGML_TYPE_F32, shape).tensor;
        ggml_set_input(input);

        auto hidden = engine::modules::Conv1dModule({
            mel_bins, config.stem_channels, 5, 1, 2, 1, true,
        }).build(build_ctx, engine::core::wrap_tensor(input, shape, GGML_TYPE_F32), weights->stem_conv);
        hidden = engine::modules::GroupNormModule({config.stem_channels, 1, kGroupNormEps, true, true})
                     .build(build_ctx, hidden, weights->stem_norm);
        hidden = engine::modules::SiluModule{}.build(build_ctx, hidden);

        int64_t in_channels = config.stem_channels;
        int64_t stage_frames = frames;
        std::vector<engine::core::TensorValue> stage_outputs;
        for (const auto & stage : weights->stages) {
            if (stage.transition_stride == 2) {
                // F.pad(x, (1, 1)) then Conv1d(kernel 3, stride 2).
                stage_frames = (stage_frames + 2 - 3) / 2 + 1;
            }
            hidden = engine::modules::Conv1dModule({
                in_channels, stage.out_channels, 3, stage.transition_stride, 1, 1, true,
            }).build(build_ctx, hidden, stage.transition_conv);
            hidden = engine::modules::GroupNormModule({stage.out_channels, 1, kGroupNormEps, true, true})
                         .build(build_ctx, hidden, stage.transition_norm);
            hidden = engine::modules::SiluModule{}.build(build_ctx, hidden);
            for (const auto & block : stage.blocks) {
                hidden = build_res_block(
                    build_ctx, hidden, block, config.depthwise_kernel_size, stage_frames);
            }
            stage_outputs.push_back(hidden);
            in_channels = stage.out_channels;
        }
        trunk_frames = stage_frames;

        auto fused = stage_outputs.front();
        for (size_t i = 1; i < stage_outputs.size(); ++i) {
            fused = engine::modules::ConcatModule({1}).build(build_ctx, fused, stage_outputs[i]);
        }
        int64_t fused_in = 0;
        for (const auto & stage : weights->stages) {
            fused_in += stage.out_channels;
        }
        channels = config.stage_channels.back();
        fused = engine::modules::Conv1dModule({fused_in, channels, 1, 1, 0, 1, true})
                    .build(build_ctx, fused, weights->fuse_conv);
        fused = engine::modules::GroupNormModule({channels, 1, kGroupNormEps, true, true})
                    .build(build_ctx, fused, weights->fuse_norm);
        fused = engine::modules::SiluModule{}.build(build_ctx, fused);
        fused = engine::core::ensure_backend_addressable_layout(build_ctx, fused);
        output = fused.tensor;
        ggml_set_output(output);
        graph = ggml_new_graph_custom(ctx.get(), 65536, false);
        ggml_build_forward_expand(graph, output);
        gallocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
        if (gallocr == nullptr || !ggml_gallocr_reserve(gallocr, graph) ||
            !ggml_gallocr_alloc_graph(gallocr, graph)) {
            throw std::runtime_error("failed to allocate the Sopro speaker encoder graph");
        }
    }

    ~SoproSpeakerGraph() {
        if (gallocr != nullptr) {
            ggml_gallocr_free(gallocr);
            gallocr = nullptr;
        }
    }

    bool matches(const SoproSpeakerWeights & other, int64_t other_frames) const noexcept {
        return weights.get() == &other && frames == other_frames;
    }

    std::vector<float> run(const std::vector<float> & log_mel) {
        ggml_backend_tensor_set(input, log_mel.data(), 0, log_mel.size() * sizeof(float));
        const ggml_status status = engine::core::compute_backend_graph(backend, graph);
        ggml_backend_synchronize(backend);
        if (status != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("Sopro speaker encoder graph compute failed");
        }
        std::vector<float> out(static_cast<size_t>(channels * trunk_frames), 0.0F);
        ggml_backend_tensor_get(output, out.data(), 0, out.size() * sizeof(float));
        return out;
    }

    ggml_backend_t backend = nullptr;
    std::shared_ptr<const SoproSpeakerWeights> weights;
    int64_t frames = 0;
    int64_t mel_bins = 0;
    int64_t trunk_frames = 0;
    int64_t channels = 0;
    std::unique_ptr<ggml_context, GgmlContextDeleter> ctx;
    ggml_tensor * input = nullptr;
    ggml_tensor * output = nullptr;
    ggml_cgraph * graph = nullptr;
    ggml_gallocr_t gallocr = nullptr;
};

SoproSpeakerEncoderRuntime::SoproSpeakerEncoderRuntime(
    const SoproTTSAssets & assets,
    engine::core::ExecutionContext & execution_context,
    size_t weight_context_bytes,
    size_t graph_context_bytes,
    engine::assets::TensorStorageType matmul_storage_type,
    engine::assets::TensorStorageType conv_storage_type)
    : config_(assets.config.speaker_encoder),
      execution_context_(execution_context),
      graph_context_bytes_(graph_context_bytes),
      weights_(load_speaker_weights(
          execution_context.backend(),
          execution_context.backend_type(),
          *assets.speaker_encoder_weights,
          assets.config.speaker_encoder,
          weight_context_bytes,
          matmul_storage_type,
          conv_storage_type)) {}

SoproSpeakerEncoderRuntime::~SoproSpeakerEncoderRuntime() = default;

int64_t SoproSpeakerEncoderRuntime::sample_rate() const noexcept {
    return config_.sample_rate;
}

std::vector<float> SoproSpeakerEncoderRuntime::trunk(
    const std::vector<float> & log_mel,
    int64_t frames,
    int64_t & out_frames) const {
    if (graph_ == nullptr || !graph_->matches(*weights_, frames)) {
        // Free the previous arena first; otherwise both are resident while the
        // replacement is allocated, and every segment rebuilds this graph.
        graph_.reset();
        graph_ = std::make_unique<SoproSpeakerGraph>(
            execution_context_.backend(),
            execution_context_.backend_type(),
            graph_context_bytes_,
            config_,
            weights_,
            frames);
    }
    out_frames = graph_->trunk_frames;
    return graph_->run(log_mel);
}

SoproSpeakerEmbeddings SoproSpeakerEncoderRuntime::encode(const std::vector<float> & audio16) const {
    if (audio16.empty()) {
        throw std::runtime_error("Sopro speaker encoder requires a non-empty reference waveform");
    }
    // LogMelFrontend: power-2 mel, log with a floor, then a per-frame LayerNorm
    // across the mel axis (no affine parameters).
    const int64_t freq_bins = config_.n_fft / 2 + 1;
    engine::audio::STFTConfig stft;
    stft.n_fft = config_.n_fft;
    stft.hop_length = config_.hop_length;
    stft.win_length = config_.win_length;
    stft.center = true;
    stft.pad_mode = engine::audio::STFTPadMode::Reflect;
    const auto magnitude = engine::audio::STFT{}.compute_magnitude(
        audio16, weights_->analysis_window, 1, static_cast<int64_t>(audio16.size()), stft,
        static_cast<size_t>(execution_context_.config().threads));
    if (magnitude.shape.size() != 3 || magnitude.shape[1] != freq_bins) {
        throw std::runtime_error("Sopro speaker encoder STFT produced an unexpected layout");
    }
    const int64_t frames = magnitude.shape[2];
    if (frames < 3) {
        throw std::runtime_error("Sopro speaker encoder reference audio is too short");
    }
    std::vector<float> mel(static_cast<size_t>(config_.n_mels * frames), 0.0F);
    for (int64_t f = 0; f < freq_bins; ++f) {
        const float * fb_row = weights_->mel_filterbank.data() + static_cast<size_t>(f * config_.n_mels);
        const float * spec_row = magnitude.values.data() + static_cast<size_t>(f * frames);
        for (int64_t m = 0; m < config_.n_mels; ++m) {
            const float weight = fb_row[m];
            if (weight == 0.0F) {
                continue;
            }
            float * out_row = mel.data() + static_cast<size_t>(m * frames);
            for (int64_t t = 0; t < frames; ++t) {
                // power=2.0: the mel filters see squared magnitudes.
                out_row[t] += weight * spec_row[t] * spec_row[t];
            }
        }
    }
    for (auto & value : mel) {
        value = std::log(std::max(value, config_.mel_log_floor));
    }
    for (int64_t t = 0; t < frames; ++t) {
        double sum = 0.0;
        for (int64_t m = 0; m < config_.n_mels; ++m) {
            sum += mel[static_cast<size_t>(m * frames + t)];
        }
        const double mean = sum / static_cast<double>(config_.n_mels);
        double variance = 0.0;
        for (int64_t m = 0; m < config_.n_mels; ++m) {
            const double centred = mel[static_cast<size_t>(m * frames + t)] - mean;
            variance += centred * centred;
        }
        variance /= static_cast<double>(config_.n_mels);
        const double inv_std = 1.0 / std::sqrt(variance + kLayerNormEps);
        for (int64_t m = 0; m < config_.n_mels; ++m) {
            auto & value = mel[static_cast<size_t>(m * frames + t)];
            value = static_cast<float>((value - mean) * inv_std);
        }
    }

    dump("spk_wav16.f32", audio16);
    dump("spk_mel.f32", mel);
    int64_t trunk_frames = 0;
    const auto features = trunk(mel, frames, trunk_frames);
    dump("spk_trunk.f32", features);
    const int64_t channels = config_.stage_channels.back();
    if (trunk_frames <= 0) {
        throw std::runtime_error("Sopro speaker encoder produced no trunk frames");
    }

    // AttentiveStatsPool: softmax attention over time, then weighted mean/std.
    std::vector<float> scores(static_cast<size_t>(trunk_frames), 0.0F);
    std::vector<float> attn_hidden(static_cast<size_t>(config_.attn_hidden), 0.0F);
    for (int64_t t = 0; t < trunk_frames; ++t) {
        for (int64_t h = 0; h < config_.attn_hidden; ++h) {
            double sum = weights_->attn_conv1_bias[static_cast<size_t>(h)];
            const float * row = weights_->attn_conv1_weight.data() + static_cast<size_t>(h * channels);
            for (int64_t c = 0; c < channels; ++c) {
                sum += static_cast<double>(row[c]) *
                       static_cast<double>(features[static_cast<size_t>(c * trunk_frames + t)]);
            }
            attn_hidden[static_cast<size_t>(h)] = std::tanh(static_cast<float>(sum));
        }
        double sum = weights_->attn_conv2_bias[0];
        for (int64_t h = 0; h < config_.attn_hidden; ++h) {
            sum += static_cast<double>(weights_->attn_conv2_weight[static_cast<size_t>(h)]) *
                   static_cast<double>(attn_hidden[static_cast<size_t>(h)]);
        }
        scores[static_cast<size_t>(t)] = static_cast<float>(sum);
    }
    const float max_score = *std::max_element(scores.begin(), scores.end());
    double score_sum = 0.0;
    for (auto & score : scores) {
        score = std::exp(score - max_score);
        score_sum += score;
    }
    for (auto & score : scores) {
        score = static_cast<float>(score / score_sum);
    }

    std::vector<float> id_input(static_cast<size_t>(channels * 2), 0.0F);
    for (int64_t c = 0; c < channels; ++c) {
        const float * row = features.data() + static_cast<size_t>(c * trunk_frames);
        double mean = 0.0;
        for (int64_t t = 0; t < trunk_frames; ++t) {
            mean += static_cast<double>(scores[static_cast<size_t>(t)]) * static_cast<double>(row[t]);
        }
        double variance = 0.0;
        for (int64_t t = 0; t < trunk_frames; ++t) {
            const double centred = static_cast<double>(row[t]) - mean;
            variance += static_cast<double>(scores[static_cast<size_t>(t)]) * centred * centred;
        }
        id_input[static_cast<size_t>(c)] = static_cast<float>(mean);
        id_input[static_cast<size_t>(channels + c)] =
            static_cast<float>(std::sqrt(std::max(variance, 1.0e-6)));
    }

    // MultiScaleStylePool: mean/std of the trunk features and of a length-5
    // moving average of them (AvgPool1d(kernel 5, stride 1, padding 2), which
    // divides by the kernel size including the zero padding).
    std::vector<float> style_input(static_cast<size_t>(channels * 4), 0.0F);
    std::vector<float> smoothed(static_cast<size_t>(trunk_frames), 0.0F);
    const auto denominator = static_cast<double>(trunk_frames);
    for (int64_t c = 0; c < channels; ++c) {
        const float * row = features.data() + static_cast<size_t>(c * trunk_frames);
        for (int64_t t = 0; t < trunk_frames; ++t) {
            double sum = 0.0;
            for (int64_t k = -2; k <= 2; ++k) {
                const int64_t index = t + k;
                if (index >= 0 && index < trunk_frames) {
                    sum += static_cast<double>(row[index]);
                }
            }
            smoothed[static_cast<size_t>(t)] = static_cast<float>(sum / 5.0);
        }
        const float * scales[2] = {row, smoothed.data()};
        for (int scale = 0; scale < 2; ++scale) {
            const float * values = scales[scale];
            double sum = 0.0;
            for (int64_t t = 0; t < trunk_frames; ++t) {
                sum += static_cast<double>(values[t]);
            }
            const double mean = sum / denominator;
            double variance = 0.0;
            for (int64_t t = 0; t < trunk_frames; ++t) {
                const double centred = static_cast<double>(values[t]) - mean;
                variance += centred * centred;
            }
            variance /= denominator;
            style_input[static_cast<size_t>(scale * 2 * channels + c)] = static_cast<float>(mean);
            style_input[static_cast<size_t>((scale * 2 + 1) * channels + c)] =
                static_cast<float>(std::sqrt(std::max(variance, 1.0e-6)));
        }
    }

    SoproSpeakerEmbeddings out;
    out.id_emb = apply_head(weights_->id_head, id_input);
    double norm = 0.0;
    for (const float value : out.id_emb) {
        norm += static_cast<double>(value) * static_cast<double>(value);
    }
    const auto inv_norm = static_cast<float>(1.0 / std::max(std::sqrt(norm), 1.0e-12));
    for (auto & value : out.id_emb) {
        value *= inv_norm;
    }
    out.style_emb = apply_head(weights_->style_head, style_input);
    out.style_ctrl = apply_head(weights_->style_ctrl_head, style_input);
    dump("spk_id_emb.f32", out.id_emb);
    dump("spk_style_emb.f32", out.style_emb);
    dump("spk_style_ctrl.f32", out.style_ctrl);
    return out;
}

}  // namespace engine::community_models::sopro_tts
