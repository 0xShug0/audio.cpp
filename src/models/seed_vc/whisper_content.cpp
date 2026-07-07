#include "engine/models/seed_vc/whisper_content.h"

#include "engine/framework/audio/dsp.h"
#include "engine/framework/audio/waveform_ops.h"
#include "engine/framework/core/backend.h"
#include "engine/framework/modules/whisper_embedding.h"

#include <ggml-alloc.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace engine::models::seed_vc {
namespace {

using engine::core::TensorShape;

const engine::core::TensorValue & require_tensor(const SeedVcWeightBundle & weights, const std::string & name) {
    const auto it = weights.tensors.find(name);
    if (it == weights.tensors.end()) {
        throw std::runtime_error("Seed-VC Whisper missing tensor: " + name);
    }
    return it->second;
}

engine::modules::LinearWeights linear_weights(
    const SeedVcWeightBundle & weights,
    const std::string & prefix,
    bool use_bias) {
    return engine::modules::LinearWeights{
        require_tensor(weights, prefix + ".weight"),
        use_bias ? std::optional<engine::core::TensorValue>(require_tensor(weights, prefix + ".bias")) : std::nullopt};
}

engine::modules::NormWeights norm_weights(const SeedVcWeightBundle & weights, const std::string & prefix) {
    return engine::modules::NormWeights{
        require_tensor(weights, prefix + ".weight"),
        require_tensor(weights, prefix + ".bias")};
}

engine::modules::WhisperEmbeddingConfig make_whisper_config(const SeedVcWeightBundle & weights) {
    const auto positions = require_tensor(weights, "model.encoder.embed_positions.weight");
    const auto conv1 = require_tensor(weights, "model.encoder.conv1.weight");
    const auto q_proj = require_tensor(weights, "model.encoder.layers.0.self_attn.q_proj.weight");
    engine::modules::WhisperEmbeddingConfig config;
    config.n_mels = conv1.shape.dims[1];
    config.n_audio_state = conv1.shape.dims[0];
    config.n_audio_ctx = positions.shape.dims[0];
    config.n_audio_head = 12;
    config.n_audio_layer = 0;
    while (weights.tensors.find("model.encoder.layers." + std::to_string(config.n_audio_layer) + ".fc1.weight") !=
           weights.tensors.end()) {
        ++config.n_audio_layer;
    }
    if (q_proj.shape.dims[0] != config.n_audio_state || config.n_audio_layer <= 0) {
        throw std::runtime_error("Seed-VC Whisper encoder config is invalid");
    }
    return config;
}

engine::modules::WhisperEmbeddingWeights make_whisper_weights(
    const SeedVcWeightBundle & weights,
    const engine::modules::WhisperEmbeddingConfig & config) {
    engine::modules::WhisperEmbeddingWeights out;
    out.conv1 = {
        require_tensor(weights, "model.encoder.conv1.weight"),
        require_tensor(weights, "model.encoder.conv1.bias"),
    };
    out.conv2 = {
        require_tensor(weights, "model.encoder.conv2.weight"),
        require_tensor(weights, "model.encoder.conv2.bias"),
    };
    out.positional_embedding = require_tensor(weights, "model.encoder.embed_positions.weight");
    out.layers.reserve(static_cast<size_t>(config.n_audio_layer));
    for (int64_t layer = 0; layer < config.n_audio_layer; ++layer) {
        const std::string prefix = "model.encoder.layers." + std::to_string(layer);
        engine::modules::WhisperEncoderLayerWeights layer_weights;
        layer_weights.attention_norm = norm_weights(weights, prefix + ".self_attn_layer_norm");
        layer_weights.attention.query = linear_weights(weights, prefix + ".self_attn.q_proj", true);
        layer_weights.attention.key = linear_weights(weights, prefix + ".self_attn.k_proj", false);
        layer_weights.attention.value = linear_weights(weights, prefix + ".self_attn.v_proj", true);
        layer_weights.attention.out = linear_weights(weights, prefix + ".self_attn.out_proj", true);
        layer_weights.mlp_norm = norm_weights(weights, prefix + ".final_layer_norm");
        layer_weights.mlp.fc1_weight = require_tensor(weights, prefix + ".fc1.weight");
        layer_weights.mlp.fc1_bias = require_tensor(weights, prefix + ".fc1.bias");
        layer_weights.mlp.fc2_weight = require_tensor(weights, prefix + ".fc2.weight");
        layer_weights.mlp.fc2_bias = require_tensor(weights, prefix + ".fc2.bias");
        out.layers.push_back(std::move(layer_weights));
    }
    out.final_norm = norm_weights(weights, "model.encoder.layer_norm");
    return out;
}

std::vector<float> compute_whisper_log_mel(const std::vector<float> & waveform_16k, size_t threads) {
    constexpr int64_t kSampleRate = 16000;
    constexpr int64_t kNfft = 400;
    constexpr int64_t kHop = 160;
    constexpr int64_t kMels = 80;
    constexpr int64_t kOutputFrames = 3000;
    const engine::audio::STFTConfig stft_config{
        kNfft,
        kHop,
        kNfft,
        true,
        engine::audio::STFTPadMode::Reflect,
        engine::audio::STFTFamily::Kokoro,
    };
    const auto & window = engine::audio::get_cached_stft_window(stft_config);
    const auto magnitude = engine::audio::STFT().compute_magnitude(
        waveform_16k,
        window,
        1,
        static_cast<int64_t>(waveform_16k.size()),
        stft_config,
        threads);
    const int64_t freq_bins = magnitude.shape[1];
    const int64_t stft_frames = magnitude.shape[2];
    if (freq_bins != (kNfft / 2 + 1) || stft_frames <= kOutputFrames) {
        throw std::runtime_error("Seed-VC Whisper frontend STFT shape mismatch");
    }
    static const auto mel_filter = engine::audio::MelFilterbank().build(
        engine::audio::MelFilterbankConfig{kSampleRate, kNfft, kMels, 0.0F, 0.0F, true});
    std::vector<float> log_mel(static_cast<size_t>(kMels * kOutputFrames), 0.0F);
    float max_log = -std::numeric_limits<float>::infinity();
    for (int64_t mel = 0; mel < kMels; ++mel) {
        for (int64_t frame = 0; frame < kOutputFrames; ++frame) {
            float sum = 0.0F;
            for (int64_t freq = 0; freq < freq_bins; ++freq) {
                const float mag = magnitude.values[static_cast<size_t>(freq * stft_frames + frame)];
                sum += mel_filter.values[static_cast<size_t>(mel * freq_bins + freq)] * mag * mag;
            }
            const float value = std::log10(std::max(sum, 1.0e-10F));
            log_mel[static_cast<size_t>(mel * kOutputFrames + frame)] = value;
            max_log = std::max(max_log, value);
        }
    }
    const float floor = max_log - 8.0F;
    for (float & value : log_mel) {
        value = (std::max(value, floor) + 4.0F) / 4.0F;
    }
    return log_mel;
}

}  // namespace

struct SeedVcWhisperContentEncoder::State {
    explicit State(const SeedVcWeightBundle & bundle)
        : weights(make_whisper_weights(bundle, make_whisper_config(bundle))) {}

    ~State() {
        if (gallocr != nullptr) {
            ggml_gallocr_free(gallocr);
            gallocr = nullptr;
        }
        if (ctx != nullptr) {
            ggml_free(ctx);
            ctx = nullptr;
        }
    }

    void ensure_graph(const SeedVcWeightBundle & bundle, const engine::modules::WhisperEmbeddingConfig & config) {
        if (ctx != nullptr) {
            return;
        }
        if (bundle.execution_context == nullptr) {
            throw std::runtime_error("Seed-VC Whisper encoder requires execution context");
        }
        ggml_init_params params{512ull * 1024ull * 1024ull, nullptr, true};
        ctx = ggml_init(params);
        if (ctx == nullptr) {
            throw std::runtime_error("failed to initialize Seed-VC Whisper encoder graph context");
        }
        engine::core::ModuleBuildContext build_ctx{ctx, "seed_vc.whisper.encoder", bundle.execution_context->backend_type()};
        const auto input = engine::core::make_tensor(
            build_ctx,
            GGML_TYPE_F32,
            TensorShape::from_dims({1, config.n_mels, config.n_audio_ctx * 2}));
        input_tensor = input.tensor;
        const auto output = engine::modules::WhisperEmbeddingModule(config).build(build_ctx, input, weights);
        output_tensor = output.tensor;
        ggml_set_output(output_tensor);
        graph = ggml_new_graph_custom(ctx, 65536, false);
        ggml_build_forward_expand(graph, output_tensor);
        gallocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(bundle.execution_context->backend()));
        if (gallocr == nullptr ||
            !ggml_gallocr_reserve(gallocr, graph) ||
            !ggml_gallocr_alloc_graph(gallocr, graph)) {
            if (gallocr != nullptr) {
                ggml_gallocr_free(gallocr);
                gallocr = nullptr;
            }
            ggml_free(ctx);
            ctx = nullptr;
            throw std::runtime_error("failed to allocate Seed-VC Whisper encoder graph tensors");
        }
    }

    engine::modules::WhisperEmbeddingWeights weights;
    std::mutex mutex;
    ggml_context * ctx = nullptr;
    ggml_gallocr_t gallocr = nullptr;
    ggml_cgraph * graph = nullptr;
    ggml_tensor * input_tensor = nullptr;
    ggml_tensor * output_tensor = nullptr;
};

SeedVcWhisperContentEncoder::SeedVcWhisperContentEncoder(std::shared_ptr<const SeedVcWeightBundle> weights)
    : weights_(std::move(weights)) {
    if (weights_ == nullptr) {
        throw std::runtime_error("Seed-VC Whisper encoder requires weights");
    }
    config_ = make_whisper_config(*weights_);
    state_ = std::make_shared<State>(*weights_);
}

SeedVcWhisperContentEncoder::~SeedVcWhisperContentEncoder() = default;
SeedVcWhisperContentEncoder::SeedVcWhisperContentEncoder(SeedVcWhisperContentEncoder &&) noexcept = default;
SeedVcWhisperContentEncoder & SeedVcWhisperContentEncoder::operator=(SeedVcWhisperContentEncoder &&) noexcept = default;

int64_t SeedVcWhisperContentEncoder::channels() const noexcept {
    return config_.n_audio_state;
}

std::vector<float> SeedVcWhisperContentEncoder::extract_16k_mono(
    const std::vector<float> & waveform_16k,
    size_t threads) const {
    if (weights_ == nullptr || state_ == nullptr) {
        throw std::runtime_error("Seed-VC Whisper encoder is not initialized");
    }
    const int64_t wanted_frames = static_cast<int64_t>(waveform_16k.size()) / 320 + 1;
    constexpr size_t kWhisperSamples = 480000;
    const auto log_mel = compute_whisper_log_mel(
        engine::audio::copy_or_zero_pad_samples_to_count(waveform_16k, kWhisperSamples),
        threads);
    std::lock_guard<std::mutex> lock(state_->mutex);
    state_->ensure_graph(*weights_, config_);
    ggml_backend_tensor_set(state_->input_tensor, log_mel.data(), 0, log_mel.size() * sizeof(float));
    if (engine::core::compute_backend_graph(weights_->execution_context->backend(), state_->graph) != GGML_STATUS_SUCCESS) {
        throw std::runtime_error("ggml_backend_graph_compute failed for Seed-VC Whisper encoder");
    }
    std::vector<float> full(static_cast<size_t>(config_.n_audio_ctx * config_.n_audio_state), 0.0F);
    ggml_backend_tensor_get(state_->output_tensor, full.data(), 0, full.size() * sizeof(float));
    const int64_t frames = std::min<int64_t>(wanted_frames, config_.n_audio_ctx);
    std::vector<float> out(static_cast<size_t>(frames * config_.n_audio_state), 0.0F);
    std::copy_n(full.begin(), out.size(), out.begin());
    return out;
}

}  // namespace engine::models::seed_vc
