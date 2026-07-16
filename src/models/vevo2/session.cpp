#include "engine/models/vevo2/session.h"

#include "audio_cache.h"

#include "engine/framework/audio/conversion.h"
#include "engine/framework/audio/dsp.h"
#include "engine/framework/audio/resampling.h"
#include "engine/framework/audio/waveform_ops.h"
#include "engine/framework/audio/wav_reader.h"
#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/core/backend.h"
#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/core/execution_context.h"
#include "engine/framework/core/module.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/modules/weight_binding.h"
#include "engine/framework/modules/whisper_embedding.h"
#include "engine/framework/runtime/options.h"
#include "engine/framework/text/chunking.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <utility>

namespace engine::models::vevo2 {

using Clock = std::chrono::steady_clock;

namespace {

struct GgmlContextDeleter {
    void operator()(ggml_context * ctx) const noexcept {
        if (ctx != nullptr) {
            ggml_free(ctx);
        }
    }
};

constexpr size_t kMaxAudioCacheEntries = 8;
constexpr int64_t kDefaultTextChunkSize = 128;

std::shared_ptr<const Vevo2Assets> require_assets(std::shared_ptr<const Vevo2Assets> assets) {
    if (assets == nullptr) {
        throw std::runtime_error("Vevo2 session requires assets");
    }
    return assets;
}

struct ParsedVevo2Route {
    Vevo2InferencePath path = Vevo2InferencePath::TextProsodyToTargetVoice;
    Vevo2RouteKind route = Vevo2RouteKind::ZeroShotTts;
};

ParsedVevo2Route parse_route(const std::string & value) {
    if (value == "zero_shot_tts") {
        return {Vevo2InferencePath::TextProsodyToTargetVoice, Vevo2RouteKind::ZeroShotTts};
    }
    if (value == "text_to_singing") {
        return {Vevo2InferencePath::TextProsodyToTargetVoice, Vevo2RouteKind::TextToSinging};
    }
    if (value == "svs") {
        return {Vevo2InferencePath::TextProsodyToTargetVoice, Vevo2RouteKind::Svs};
    }
    if (value == "style_preserved_vc") {
        return {Vevo2InferencePath::SourceAudioToTargetVoice, Vevo2RouteKind::StylePreservedVoiceConversion};
    }
    if (value == "style_preserved_svc") {
        return {Vevo2InferencePath::SourceAudioToTargetVoice, Vevo2RouteKind::StylePreservedSingingConversion};
    }
    if (value == "style_converted_vc") {
        return {Vevo2InferencePath::TextProsodyToTargetVoice, Vevo2RouteKind::StyleConvertedVoiceConversion};
    }
    if (value == "style_converted_svc") {
        return {Vevo2InferencePath::TextProsodyToTargetVoice, Vevo2RouteKind::StyleConvertedSingingConversion};
    }
    if (value == "editing") {
        return {Vevo2InferencePath::TextProsodyToTargetVoice, Vevo2RouteKind::Editing};
    }
    if (value == "singing_style_conversion") {
        return {Vevo2InferencePath::TextProsodyToTargetVoice, Vevo2RouteKind::SingingStyleConversion};
    }
    if (value == "humming_to_singing" || value == "instrument_to_singing") {
        return {Vevo2InferencePath::TextProsodyToTargetVoice, Vevo2RouteKind::MelodyControl};
    }
    throw std::runtime_error(
        "invalid Vevo2 route: " + value +
        " (expected zero_shot_tts, text_to_singing, svs, style_preserved_vc, "
        "style_preserved_svc, style_converted_vc, style_converted_svc, editing, "
        "singing_style_conversion, humming_to_singing, or instrument_to_singing)");
}

const char * default_route_for_task(runtime::VoiceTaskKind task) {
    switch (task) {
    case runtime::VoiceTaskKind::Tts:
        return "zero_shot_tts";
    case runtime::VoiceTaskKind::VoiceConversion:
        return "style_preserved_vc";
    case runtime::VoiceTaskKind::SpeechToSpeech:
        return "editing";
    case runtime::VoiceTaskKind::Svc:
        return "style_preserved_svc";
    default:
        throw std::runtime_error("Vevo2 supports tts, vc, s2s, and svc tasks");
    }
}

bool route_matches_task(const Vevo2RouteKind route, runtime::VoiceTaskKind task) {
    switch (task) {
    case runtime::VoiceTaskKind::Tts:
        return route == Vevo2RouteKind::ZeroShotTts ||
            route == Vevo2RouteKind::TextToSinging ||
            route == Vevo2RouteKind::Svs;
    case runtime::VoiceTaskKind::VoiceConversion:
        return route == Vevo2RouteKind::StylePreservedVoiceConversion ||
            route == Vevo2RouteKind::StyleConvertedVoiceConversion;
    case runtime::VoiceTaskKind::SpeechToSpeech:
        return route == Vevo2RouteKind::Editing;
    case runtime::VoiceTaskKind::Svc:
        return route == Vevo2RouteKind::StylePreservedSingingConversion ||
            route == Vevo2RouteKind::StyleConvertedSingingConversion ||
            route == Vevo2RouteKind::SingingStyleConversion ||
            route == Vevo2RouteKind::MelodyControl;
    default:
        return false;
    }
}

bool route_defaults_to_prosody(const Vevo2RouteKind route) {
    return route == Vevo2RouteKind::StyleConvertedVoiceConversion ||
        route == Vevo2RouteKind::StyleConvertedSingingConversion ||
        route == Vevo2RouteKind::Editing ||
        route == Vevo2RouteKind::SingingStyleConversion ||
        route == Vevo2RouteKind::MelodyControl;
}

bool route_defaults_to_pitch_shift(const Vevo2RouteKind route) {
    return route == Vevo2RouteKind::StylePreservedVoiceConversion ||
        route == Vevo2RouteKind::StylePreservedSingingConversion ||
        route == Vevo2RouteKind::StyleConvertedSingingConversion ||
        route == Vevo2RouteKind::SingingStyleConversion ||
        route == Vevo2RouteKind::MelodyControl;
}

bool route_uses_source_audio_as_reference(const Vevo2RouteKind route) {
    return route == Vevo2RouteKind::Editing ||
        route == Vevo2RouteKind::StyleConvertedVoiceConversion ||
        route == Vevo2RouteKind::StyleConvertedSingingConversion ||
        route == Vevo2RouteKind::SingingStyleConversion;
}

void reject_unknown_options(const runtime::SessionOptions & options) {
    for (const auto & [key, value] : options.options) {
        (void) value;
        if (key.rfind("vevo2.", 0) == 0 &&
            key != "vevo2.ar_weight_context_mb" &&
            key != "vevo2.ar_weight_type" &&
            key != "vevo2.ar_prefill_graph_context_mb" &&
            key != "vevo2.ar_decode_graph_context_mb" &&
            key != "vevo2.conv_weight_type" &&
            key != "vevo2.tokenizer_weight_context_mb" &&
            key != "vevo2.tokenizer_graph_context_mb" &&
            key != "vevo2.tokenizer_weight_type" &&
            key != "vevo2.tokenizer_conv_weight_type" &&
            key != "vevo2.whisper_weight_context_mb" &&
            key != "vevo2.whisper_graph_context_mb" &&
            key != "vevo2.whisper_conv_weight_type" &&
            key != "vevo2.fm_weight_context_mb" &&
            key != "vevo2.fm_graph_context_mb" &&
            key != "vevo2.fm_weight_type" &&
            key != "vevo2.fm_conv_weight_type" &&
            key != "vevo2.vocoder_weight_context_mb" &&
            key != "vevo2.vocoder_graph_context_mb" &&
            key != "vevo2.vocoder_weight_type" &&
            key != "vevo2.vocoder_conv_weight_type" &&
            key != "vevo2.weight_type" &&
            key != "vevo2.whisper_weight_type") {
            throw std::runtime_error("unknown Vevo2 session option: " + key);
        }
    }
}

runtime::AudioBuffer make_audio_buffer(int sample_rate, std::vector<float> samples) {
    runtime::AudioBuffer out;
    out.sample_rate = sample_rate;
    out.channels = 1;
    out.samples = std::move(samples);
    return out;
}

runtime::AudioBuffer normalize_audio_to_24k_mono(const runtime::AudioBuffer & audio) {
    return make_audio_buffer(
        24000,
        engine::audio::convert_interleaved_audio_to_mono_linear_resampled(
            audio.samples,
            audio.sample_rate,
            audio.channels,
            24000));
}

runtime::AudioBuffer load_audio_24k_mono(const std::filesystem::path & path) {
    return make_audio_buffer(24000, engine::audio::read_wav_f32_as_mono_linear_resampled(path, 24000));
}

void trim_audio_seconds(runtime::AudioBuffer & audio, float seconds) {
    if (seconds < 0.0F) {
        throw std::runtime_error("Vevo2 reference_duration_seconds must be non-negative");
    }
    const size_t max_samples = static_cast<size_t>(seconds * static_cast<float>(audio.sample_rate)) *
        static_cast<size_t>(audio.channels);
    engine::audio::truncate_samples_to_count(audio.samples, max_samples);
}

int64_t frame_count_24k(const runtime::AudioBuffer & audio) {
    return static_cast<int64_t>(engine::audio::convert_interleaved_audio_to_mono_linear_resampled(
        audio.samples,
        audio.sample_rate,
        audio.channels,
        24000).size()) / 480;
}

double median_voiced_frequency_hz_16k(const runtime::AudioBuffer & audio) {
    const auto waveform = engine::audio::convert_interleaved_audio_to_mono_linear_resampled(
        audio.samples,
        audio.sample_rate,
        audio.channels,
        16000);
    constexpr int64_t kSampleRate = 16000;
    constexpr int64_t kHop = 256;
    constexpr int64_t kFrame = 2048;
    constexpr double kMinHz = 50.0;
    constexpr double kMaxHz = 1100.0;
    constexpr double kYinThreshold = 0.25;
    constexpr double kPi = 3.14159265358979323846264338327950288;
    if (static_cast<int64_t>(waveform.size()) < kFrame) {
        throw std::runtime_error("Vevo2 pitch shift estimation requires at least one analysis frame");
    }
    const int64_t min_lag = std::max<int64_t>(1, static_cast<int64_t>(std::floor(kSampleRate / kMaxHz)));
    const int64_t max_lag = std::min<int64_t>(kFrame - 2, static_cast<int64_t>(std::ceil(kSampleRate / kMinHz)));
    std::vector<double> window(static_cast<size_t>(kFrame), 0.0);
    for (int64_t i = 0; i < kFrame; ++i) {
        window[static_cast<size_t>(i)] =
            0.5 - 0.5 * std::cos((2.0 * kPi * static_cast<double>(i)) / static_cast<double>(kFrame - 1));
    }
    std::vector<double> frequencies;
    frequencies.reserve(static_cast<size_t>((static_cast<int64_t>(waveform.size()) - kFrame) / kHop + 1));
    std::vector<double> frame(static_cast<size_t>(kFrame), 0.0);
    std::vector<double> difference(static_cast<size_t>(max_lag + 1), 0.0);
    std::vector<double> cumulative_mean_normalized(static_cast<size_t>(max_lag + 1), 1.0);
    for (int64_t start = 0; start + kFrame <= static_cast<int64_t>(waveform.size()); start += kHop) {
        double mean = 0.0;
        for (int64_t i = 0; i < kFrame; ++i) {
            mean += static_cast<double>(waveform[static_cast<size_t>(start + i)]);
        }
        mean /= static_cast<double>(kFrame);
        double energy = 0.0;
        for (int64_t i = 0; i < kFrame; ++i) {
            const double sample = (static_cast<double>(waveform[static_cast<size_t>(start + i)]) - mean) *
                window[static_cast<size_t>(i)];
            frame[static_cast<size_t>(i)] = sample;
            energy += sample * sample;
        }
        if (energy < 1.0e-8) {
            continue;
        }
        for (int64_t lag = 1; lag <= max_lag; ++lag) {
            double sum = 0.0;
            for (int64_t i = 0; i + lag < kFrame; ++i) {
                const double delta = frame[static_cast<size_t>(i)] - frame[static_cast<size_t>(i + lag)];
                sum += delta * delta;
            }
            difference[static_cast<size_t>(lag)] = sum;
        }
        double running_sum = 0.0;
        cumulative_mean_normalized[0] = 1.0;
        for (int64_t lag = 1; lag <= max_lag; ++lag) {
            running_sum += difference[static_cast<size_t>(lag)];
            cumulative_mean_normalized[static_cast<size_t>(lag)] =
                running_sum > 1.0e-12
                    ? difference[static_cast<size_t>(lag)] * static_cast<double>(lag) / running_sum
                    : 1.0;
        }
        int64_t best_lag = 0;
        for (int64_t lag = min_lag; lag <= max_lag; ++lag) {
            if (cumulative_mean_normalized[static_cast<size_t>(lag)] < kYinThreshold) {
                best_lag = lag;
                while (best_lag + 1 <= max_lag &&
                       cumulative_mean_normalized[static_cast<size_t>(best_lag + 1)] <
                           cumulative_mean_normalized[static_cast<size_t>(best_lag)]) {
                    ++best_lag;
                }
                break;
            }
        }
        if (best_lag == 0) {
            best_lag = min_lag;
            for (int64_t lag = min_lag + 1; lag <= max_lag; ++lag) {
                if (cumulative_mean_normalized[static_cast<size_t>(lag)] <
                    cumulative_mean_normalized[static_cast<size_t>(best_lag)]) {
                    best_lag = lag;
                }
            }
            if (cumulative_mean_normalized[static_cast<size_t>(best_lag)] > 0.3) {
                continue;
            }
        }
        double lag = static_cast<double>(best_lag);
        if (best_lag > 1 && best_lag < max_lag) {
            const double left = cumulative_mean_normalized[static_cast<size_t>(best_lag - 1)];
            const double center = cumulative_mean_normalized[static_cast<size_t>(best_lag)];
            const double right = cumulative_mean_normalized[static_cast<size_t>(best_lag + 1)];
            const double denom = left - 2.0 * center + right;
            if (std::abs(denom) > 1.0e-12) {
                lag += 0.5 * (left - right) / denom;
            }
        }
        if (lag > 0.0) {
            frequencies.push_back(static_cast<double>(kSampleRate) / lag);
        }
    }
    if (frequencies.empty()) {
        throw std::runtime_error("Vevo2 pitch shift estimation found no voiced frames");
    }
    auto mid = frequencies.begin() + static_cast<std::ptrdiff_t>(frequencies.size() / 2);
    std::nth_element(frequencies.begin(), mid, frequencies.end());
    double median = *mid;
    if (frequencies.size() % 2 == 0) {
        const auto prev = std::max_element(frequencies.begin(), mid);
        median = (*prev + *mid) / 2.0;
    }
    return median;
}

int estimate_pitch_shift_steps(const runtime::AudioBuffer & source, const runtime::AudioBuffer & target) {
    const double source_median = median_voiced_frequency_hz_16k(source);
    const double target_median = median_voiced_frequency_hz_16k(target);
    if (source_median <= 0.0 || target_median <= 0.0) {
        throw std::runtime_error("Vevo2 pitch shift estimation produced non-positive median F0");
    }
    int steps = static_cast<int>(std::llround(12.0 * std::log2(target_median / source_median)));
    if (steps > 12) {
        steps %= 12;
    } else if (steps < -12) {
        steps %= -12;
    }
    return steps;
}

engine::assets::TensorStorageType option_weight_type(
    const runtime::SessionOptions & options,
    const char * key,
    engine::assets::TensorStorageType default_value) {
    const auto it = options.options.find(key);
    if (it == options.options.end()) {
        return default_value;
    }
    return engine::assets::parse_tensor_storage_type(it->second);
}

void validate_matmul_weight_storage(engine::assets::TensorStorageType storage_type, const char * option_name) {
    if (storage_type == engine::assets::TensorStorageType::Native ||
        storage_type == engine::assets::TensorStorageType::F32 ||
        storage_type == engine::assets::TensorStorageType::F16 ||
        storage_type == engine::assets::TensorStorageType::BF16 ||
        storage_type == engine::assets::TensorStorageType::Q8_0) {
        return;
    }
    throw std::runtime_error(std::string(option_name) + " currently supports only native, f32, f16, bf16, and q8_0");
}

void validate_conv_weight_storage(engine::assets::TensorStorageType storage_type, const char * option_name) {
    if (storage_type == engine::assets::TensorStorageType::Native ||
        storage_type == engine::assets::TensorStorageType::F32 ||
        storage_type == engine::assets::TensorStorageType::F16) {
        return;
    }
    throw std::runtime_error(std::string(option_name) + " currently supports only native, f32, and f16");
}

engine::assets::TensorStorageType resolve_matmul_weight_type(
    const runtime::SessionOptions & options,
    const char * option_name,
    engine::assets::TensorStorageType default_value) {
    const auto storage_type = option_weight_type(
        options,
        option_name,
        option_weight_type(options, "vevo2.weight_type", default_value));
    validate_matmul_weight_storage(storage_type, option_name);
    return storage_type;
}

engine::assets::TensorStorageType resolve_conv_weight_type(
    const runtime::SessionOptions & options,
    const char * option_name,
    engine::assets::TensorStorageType default_value) {
    const auto storage_type = option_weight_type(
        options,
        option_name,
        option_weight_type(options, "vevo2.conv_weight_type", default_value));
    validate_conv_weight_storage(storage_type, option_name);
    return storage_type;
}

engine::modules::WhisperEmbeddingWeights load_whisper_weights(
    const engine::modules::WhisperEmbeddingConfig & config,
    engine::core::BackendWeightStore & store,
    const engine::assets::TensorSource & source,
    engine::assets::TensorStorageType matmul_storage_type,
    engine::assets::TensorStorageType conv_storage_type) {
    engine::modules::WhisperEmbeddingWeights weights;
    weights.conv1 = {
        store.load_tensor(
            source,
            "encoder.conv1.weight",
            conv_storage_type,
            {config.n_audio_state, config.n_mels, 3}),
        store.load_f32_tensor(source, "encoder.conv1.bias", {config.n_audio_state}),
    };
    weights.conv2 = {
        store.load_tensor(
            source,
            "encoder.conv2.weight",
            conv_storage_type,
            {config.n_audio_state, config.n_audio_state, 3}),
        store.load_f32_tensor(source, "encoder.conv2.bias", {config.n_audio_state}),
    };
    weights.positional_embedding = store.load_f32_tensor(
        source,
        "encoder.positional_embedding",
        {config.n_audio_ctx, config.n_audio_state});
    weights.layers.reserve(static_cast<size_t>(config.n_audio_layer));
    for (int64_t layer = 0; layer < config.n_audio_layer; ++layer) {
        const std::string prefix = "encoder.blocks." + std::to_string(layer);
        engine::modules::WhisperEncoderLayerWeights layer_weights;
        layer_weights.attention_norm = engine::modules::binding::norm_from_source(store, source, prefix + ".attn_ln", config.n_audio_state);
        layer_weights.attention.query = engine::modules::binding::linear_from_source(
            store,
            source,
            prefix + ".attn.query",
            matmul_storage_type,
            config.n_audio_state,
            config.n_audio_state,
            true);
        layer_weights.attention.key = engine::modules::binding::linear_from_source(
            store,
            source,
            prefix + ".attn.key",
            matmul_storage_type,
            config.n_audio_state,
            config.n_audio_state,
            false);
        layer_weights.attention.value = engine::modules::binding::linear_from_source(
            store,
            source,
            prefix + ".attn.value",
            matmul_storage_type,
            config.n_audio_state,
            config.n_audio_state,
            true);
        layer_weights.attention.out = engine::modules::binding::linear_from_source(
            store,
            source,
            prefix + ".attn.out",
            matmul_storage_type,
            config.n_audio_state,
            config.n_audio_state,
            true);
        layer_weights.mlp_norm = engine::modules::binding::norm_from_source(store, source, prefix + ".mlp_ln", config.n_audio_state);
        layer_weights.mlp.fc1_weight = store.load_tensor(
            source,
            prefix + ".mlp.0.weight",
            matmul_storage_type,
            {config.n_audio_state * 4, config.n_audio_state});
        layer_weights.mlp.fc1_bias = store.load_f32_tensor(
            source,
            prefix + ".mlp.0.bias",
            {config.n_audio_state * 4});
        layer_weights.mlp.fc2_weight = store.load_tensor(
            source,
            prefix + ".mlp.2.weight",
            matmul_storage_type,
            {config.n_audio_state, config.n_audio_state * 4});
        layer_weights.mlp.fc2_bias = store.load_f32_tensor(source, prefix + ".mlp.2.bias", {config.n_audio_state});
        weights.layers.push_back(std::move(layer_weights));
    }
    weights.final_norm = engine::modules::binding::norm_from_source(store, source, "encoder.ln_post", config.n_audio_state);
    return weights;
}

std::vector<float> compute_openai_whisper_log_mel(const std::vector<float> & waveform16k, size_t threads) {
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
        waveform16k,
        window,
        1,
        static_cast<int64_t>(waveform16k.size()),
        stft_config,
        threads);
    const int64_t freq_bins = magnitude.shape[1];
    const int64_t stft_frames = magnitude.shape[2];
    if (freq_bins != (kNfft / 2 + 1) || stft_frames <= kOutputFrames) {
        throw std::runtime_error("Vevo2 Whisper frontend STFT shape mismatch");
    }
    static const auto mel_filter = engine::audio::MelFilterbank().build(
        engine::audio::MelFilterbankConfig{kSampleRate, kNfft, kMels, 0.0F, 0.0F, true});
    if (mel_filter.shape[0] != kMels || mel_filter.shape[1] != freq_bins) {
        throw std::runtime_error("Vevo2 Whisper frontend mel filter shape mismatch");
    }

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

struct Vevo2WhisperEmbeddingRuntime::Graph {
    Graph(
        ggml_backend_t backend,
        engine::core::BackendType backend_type,
        size_t graph_context_bytes,
        const engine::modules::WhisperEmbeddingConfig & config,
        const engine::modules::WhisperEmbeddingWeights & weights)
        : backend(backend),
          config(config) {
        ggml_init_params params{graph_context_bytes, nullptr, true};
        ctx.reset(ggml_init(params));
        if (ctx == nullptr) {
            throw std::runtime_error("failed to initialize Vevo2 Whisper embedding graph context");
        }
        engine::core::ModuleBuildContext build_ctx{ctx.get(), "vevo2.whisper_embedding", backend_type};
        const auto input = engine::core::make_tensor(
            build_ctx,
            GGML_TYPE_F32,
            engine::core::TensorShape::from_dims({1, config.n_mels, config.n_audio_ctx * 2}));
        input_tensor = input.tensor;
        const auto output = engine::modules::WhisperEmbeddingModule(config).build(build_ctx, input, weights);
        output_tensor = output.tensor;
        ggml_set_output(output_tensor);
        graph = ggml_new_graph_custom(ctx.get(), 65536, false);
        ggml_build_forward_expand(graph, output_tensor);
        gallocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
        if (gallocr == nullptr || !ggml_gallocr_reserve(gallocr, graph) || !ggml_gallocr_alloc_graph(gallocr, graph)) {
            throw std::runtime_error("failed to allocate Vevo2 Whisper embedding graph");
        }
    }

    ~Graph() {
        if (gallocr != nullptr) {
            ggml_gallocr_free(gallocr);
            gallocr = nullptr;
        }
    }

    ggml_backend_t backend = nullptr;
    engine::modules::WhisperEmbeddingConfig config;
    std::unique_ptr<ggml_context, GgmlContextDeleter> ctx;
    ggml_tensor * input_tensor = nullptr;
    ggml_tensor * output_tensor = nullptr;
    ggml_cgraph * graph = nullptr;
    ggml_gallocr_t gallocr = nullptr;
};

Vevo2WhisperEmbeddingRuntime::Vevo2WhisperEmbeddingRuntime(
    const Vevo2Assets & assets,
    engine::core::ExecutionContext & execution_context,
    size_t weight_context_bytes,
    size_t graph_context_bytes,
    engine::assets::TensorStorageType matmul_weight_storage_type,
    engine::assets::TensorStorageType conv_weight_storage_type)
    : execution_context_(execution_context),
      backend_type_(execution_context.backend_type()),
      graph_context_bytes_(graph_context_bytes),
      config_(assets.config.whisper),
      weight_source_(assets.whisper_weights),
      weight_store_(std::make_shared<engine::core::BackendWeightStore>(
          execution_context.backend(),
          execution_context.backend_type(),
          "vevo2.whisper_embedding.weights",
          weight_context_bytes)),
      weights_(load_whisper_weights(
          config_,
          *weight_store_,
          *weight_source_,
          matmul_weight_storage_type,
          conv_weight_storage_type)) {
    weight_store_->upload();
    weight_source_->release_storage();
}

Vevo2WhisperEmbeddingRuntime::~Vevo2WhisperEmbeddingRuntime() = default;

void Vevo2WhisperEmbeddingRuntime::ensure_graph() {
    if (graph_ != nullptr) {
        return;
    }
    const auto build_start = Clock::now();
    graph_ = std::make_unique<Graph>(
        execution_context_.backend(),
        backend_type_,
        graph_context_bytes_,
        config_,
        weights_);
    engine::debug::timing_log_scalar("vevo2.whisper.graph.build_ms", engine::debug::elapsed_ms(build_start));
}

std::vector<float> Vevo2WhisperEmbeddingRuntime::encode_log_mel(const std::vector<float> & log_mel) {
    const size_t input_size = static_cast<size_t>(config_.n_mels * config_.n_audio_ctx * 2);
    if (log_mel.size() != input_size) {
        throw std::runtime_error("Vevo2 Whisper embedding input shape mismatch");
    }
    ensure_graph();
    const auto upload_start = Clock::now();
    ggml_backend_tensor_set(graph_->input_tensor, log_mel.data(), 0, log_mel.size() * sizeof(float));
    const double upload_ms = engine::debug::elapsed_ms(upload_start);
    const auto compute_start = Clock::now();
    const ggml_status status = engine::core::compute_backend_graph(execution_context_.backend(), graph_->graph);
    ggml_backend_synchronize(execution_context_.backend());
    const double compute_ms = engine::debug::elapsed_ms(compute_start);
    if (status != GGML_STATUS_SUCCESS) {
        throw std::runtime_error("Vevo2 Whisper embedding graph compute failed");
    }
    std::vector<float> output(static_cast<size_t>(config_.n_audio_ctx * config_.n_audio_state));
    const auto readback_start = Clock::now();
    ggml_backend_tensor_get(graph_->output_tensor, output.data(), 0, output.size() * sizeof(float));
    const double readback_ms = engine::debug::elapsed_ms(readback_start);
    engine::debug::timing_log_scalar("vevo2.whisper.input_upload_ms", upload_ms);
    engine::debug::timing_log_scalar("vevo2.whisper.graph.compute_ms", compute_ms);
    engine::debug::timing_log_scalar("vevo2.whisper.output_readback_ms", readback_ms);
    return output;
}

std::vector<float> Vevo2WhisperEmbeddingRuntime::extract_features(
    const runtime::AudioBuffer & audio,
    int64_t target_frames,
    size_t threads) {
    if (target_frames <= 0) {
        throw std::runtime_error("Vevo2 Whisper extraction requires positive target frames");
    }
    engine::audio::WavData wav;
    wav.sample_rate = audio.sample_rate;
    wav.channels = audio.channels;
    wav.samples = audio.samples;
    if (wav.sample_rate != 24000 || wav.channels != 1) {
        throw std::runtime_error("Vevo2 Whisper extraction expects 24 kHz mono audio");
    }
    const auto resample_start = Clock::now();
    engine::audio::TorchaudioSincHannResampleOptions resample_options;
    resample_options.kernel_mode = engine::audio::TorchaudioSincHannKernelMode::Float32ComputationStoredAsFloat32;
    resample_options.accumulation = engine::audio::TorchaudioSincHannAccumulation::Float32;
    constexpr size_t kWhisperSamples = 480000;
    const auto waveform16k = engine::audio::copy_or_zero_pad_samples_to_count(
        engine::audio::resample_mono_torchaudio_sinc_hann(wav.samples, 24000, 16000, resample_options),
        kWhisperSamples);
    const double resample_pad_ms = engine::debug::elapsed_ms(resample_start);
    const auto log_mel_start = Clock::now();
    const auto log_mel = compute_openai_whisper_log_mel(waveform16k, threads);
    const double log_mel_ms = engine::debug::elapsed_ms(log_mel_start);
    const auto encode_start = Clock::now();
    const auto embedded = encode_log_mel(log_mel);
    const double encode_ms = engine::debug::elapsed_ms(encode_start);
    const auto postprocess_start = Clock::now();
    const int64_t source_frames = config_.n_audio_ctx;
    if (static_cast<int64_t>(embedded.size()) != source_frames * config_.n_audio_state) {
        throw std::runtime_error("Vevo2 Whisper embedding output shape mismatch");
    }
    std::vector<float> out(static_cast<size_t>(target_frames * config_.n_audio_state), 0.0F);
    const int64_t copied_frames = std::min<int64_t>(target_frames, source_frames);
    for (int64_t frame = 0; frame < copied_frames; ++frame) {
        std::copy_n(
            embedded.data() + static_cast<size_t>(frame * config_.n_audio_state),
            static_cast<size_t>(config_.n_audio_state),
            out.data() + static_cast<size_t>(frame * config_.n_audio_state));
    }
    if (target_frames > source_frames) {
        const float * last_frame = embedded.data() + static_cast<size_t>((source_frames - 1) * config_.n_audio_state);
        for (int64_t frame = source_frames; frame < target_frames; ++frame) {
            std::copy_n(
                last_frame,
                static_cast<size_t>(config_.n_audio_state),
                out.data() + static_cast<size_t>(frame * config_.n_audio_state));
        }
    }
    engine::debug::timing_log_scalar("vevo2.whisper.resample_pad_ms", resample_pad_ms);
    engine::debug::timing_log_scalar("vevo2.whisper.log_mel_ms", log_mel_ms);
    engine::debug::timing_log_scalar("vevo2.whisper.encode_ms", encode_ms);
    engine::debug::timing_log_scalar("vevo2.whisper.postprocess_ms", engine::debug::elapsed_ms(postprocess_start));
    engine::debug::trace_log_scalar("vevo2.whisper.target_frames", target_frames);
    return out;
}

Vevo2Session::Vevo2Session(
    runtime::TaskSpec task,
    runtime::SessionOptions options,
    std::shared_ptr<const Vevo2Assets> assets)
    : RuntimeSessionBase(options),
      task_(task),
      assets_(require_assets(std::move(assets))),
      ar_weight_context_bytes_(runtime::parse_size_mb_option(options.options, {"vevo2.ar_weight_context_mb"}, 2048ull * 1024ull * 1024ull)),
      ar_prefill_graph_context_bytes_(runtime::parse_size_mb_option(options.options, {"vevo2.ar_prefill_graph_context_mb"}, 512ull * 1024ull * 1024ull)),
      ar_decode_graph_context_bytes_(runtime::parse_size_mb_option(options.options, {"vevo2.ar_decode_graph_context_mb"}, 512ull * 1024ull * 1024ull)),
      ar_weight_storage_type_(
          resolve_matmul_weight_type(options, "vevo2.ar_weight_type", engine::assets::TensorStorageType::F32)),
      reference_execution_context_(options.backend),
      whisper_weight_context_bytes_(runtime::parse_size_mb_option(options.options, {"vevo2.whisper_weight_context_mb"}, 256ull * 1024ull * 1024ull)),
      whisper_graph_context_bytes_(runtime::parse_size_mb_option(options.options, {"vevo2.whisper_graph_context_mb"}, 512ull * 1024ull * 1024ull)),
      whisper_matmul_weight_storage_type_(
          resolve_matmul_weight_type(options, "vevo2.whisper_weight_type", engine::assets::TensorStorageType::Native)),
      whisper_conv_weight_storage_type_(
          resolve_conv_weight_type(options, "vevo2.whisper_conv_weight_type", engine::assets::TensorStorageType::Native)),
      tokenizer_weight_context_bytes_(runtime::parse_size_mb_option(options.options, {"vevo2.tokenizer_weight_context_mb"}, 768ull * 1024ull * 1024ull)),
      tokenizer_graph_context_bytes_(runtime::parse_size_mb_option(options.options, {"vevo2.tokenizer_graph_context_mb"}, 768ull * 1024ull * 1024ull)),
      tokenizer_matmul_weight_storage_type_(
          resolve_matmul_weight_type(options, "vevo2.tokenizer_weight_type", engine::assets::TensorStorageType::Native)),
      tokenizer_conv_weight_storage_type_(
          resolve_conv_weight_type(options, "vevo2.tokenizer_conv_weight_type", engine::assets::TensorStorageType::Native)),
      fm_weight_context_bytes_(runtime::parse_size_mb_option(options.options, {"vevo2.fm_weight_context_mb"}, 2048ull * 1024ull * 1024ull)),
      fm_graph_context_bytes_(runtime::parse_size_mb_option(options.options, {"vevo2.fm_graph_context_mb"}, 1024ull * 1024ull * 1024ull)),
      fm_matmul_weight_storage_type_(
          resolve_matmul_weight_type(options, "vevo2.fm_weight_type", engine::assets::TensorStorageType::Native)),
      fm_conv_weight_storage_type_(
          resolve_conv_weight_type(options, "vevo2.fm_conv_weight_type", engine::assets::TensorStorageType::Native)),
      vocoder_weight_context_bytes_(runtime::parse_size_mb_option(options.options, {"vevo2.vocoder_weight_context_mb"}, 1024ull * 1024ull * 1024ull)),
      vocoder_graph_context_bytes_(runtime::parse_size_mb_option(options.options, {"vevo2.vocoder_graph_context_mb"}, 768ull * 1024ull * 1024ull)),
      vocoder_matmul_weight_storage_type_(
          resolve_matmul_weight_type(options, "vevo2.vocoder_weight_type", engine::assets::TensorStorageType::Native)),
      vocoder_conv_weight_storage_type_(
          resolve_conv_weight_type(options, "vevo2.vocoder_conv_weight_type", engine::assets::TensorStorageType::Native)),
      whisper_embedding_(
          *assets_,
          reference_execution_context_,
          whisper_weight_context_bytes_,
          whisper_graph_context_bytes_,
          whisper_matmul_weight_storage_type_,
          whisper_conv_weight_storage_type_),
      prosody_tokenizer_(
          *assets_,
          reference_execution_context_,
          tokenizer_weight_context_bytes_,
          tokenizer_graph_context_bytes_,
          tokenizer_matmul_weight_storage_type_,
          tokenizer_conv_weight_storage_type_),
      content_style_tokenizer_(
          *assets_,
          reference_execution_context_,
          tokenizer_weight_context_bytes_,
          tokenizer_graph_context_bytes_,
          tokenizer_matmul_weight_storage_type_,
          tokenizer_conv_weight_storage_type_),
      autoregressive_model_(
          assets_,
          reference_execution_context_,
          ar_weight_context_bytes_,
          ar_prefill_graph_context_bytes_,
          ar_decode_graph_context_bytes_,
          ar_weight_storage_type_),
      flow_matching_model_(
          *assets_,
          execution_context(),
          fm_weight_context_bytes_,
          fm_graph_context_bytes_,
          fm_matmul_weight_storage_type_,
          fm_conv_weight_storage_type_),
      vocoder_(
          *assets_,
          execution_context(),
          vocoder_weight_context_bytes_,
          vocoder_graph_context_bytes_,
          vocoder_matmul_weight_storage_type_,
          vocoder_conv_weight_storage_type_) {
    if (task_.task != runtime::VoiceTaskKind::Tts &&
        task_.task != runtime::VoiceTaskKind::VoiceConversion &&
        task_.task != runtime::VoiceTaskKind::SpeechToSpeech &&
        task_.task != runtime::VoiceTaskKind::Svc) {
        throw std::runtime_error("Vevo2 supports tts, vc, s2s, and svc tasks");
    }
    if (task_.mode != runtime::RunMode::Offline) {
        throw std::runtime_error("Vevo2 currently supports offline sessions");
    }
    reject_unknown_options(RuntimeSessionBase::options());
}

std::string Vevo2Session::family() const {
    return "vevo2";
}

runtime::VoiceTaskKind Vevo2Session::task_kind() const {
    return task_.task;
}

runtime::RunMode Vevo2Session::run_mode() const {
    return task_.mode;
}

void Vevo2Session::prepare(const runtime::SessionPreparationRequest & request) {
    (void) request;
    mark_prepared();
}

std::vector<float> Vevo2Session::cached_whisper_features(
    const runtime::AudioBuffer & audio,
    int64_t target_frames,
    size_t threads) {
    const uint64_t key = audio_buffer_key(audio);
    for (const auto & entry : whisper_feature_cache_) {
        if (matches_audio_cache_key(
                entry.key,
                key,
                entry.sample_rate,
                entry.channels,
            entry.samples,
            audio) &&
            entry.target_frames == target_frames) {
            return entry.features;
        }
    }

    auto features = whisper_embedding_.extract_features(audio, target_frames, threads);
    if (whisper_feature_cache_.size() >= kMaxAudioCacheEntries) {
        whisper_feature_cache_.erase(whisper_feature_cache_.begin());
    }
    whisper_feature_cache_.push_back({
        key,
        audio.sample_rate,
        audio.channels,
        target_frames,
        audio.samples.size(),
        features,
    });
    return features;
}

Vevo2TokenSequence Vevo2Session::cached_content_style_tokens(
    const runtime::AudioBuffer & audio,
    const std::vector<float> & whisper_features,
    int64_t feature_frames) {
    const uint64_t key = audio_buffer_key(audio);
    for (const auto & entry : content_style_token_cache_) {
        if (matches_audio_cache_key(
                entry.key,
                key,
                entry.sample_rate,
                entry.channels,
            entry.samples,
            audio) &&
            entry.feature_frames == feature_frames) {
            return entry.tokens;
        }
    }

    auto tokens = content_style_tokenizer_.encode_timbre_reference(audio, whisper_features, feature_frames);
    if (content_style_token_cache_.size() >= kMaxAudioCacheEntries) {
        content_style_token_cache_.erase(content_style_token_cache_.begin());
    }
    content_style_token_cache_.push_back({
        key,
        audio.sample_rate,
        audio.channels,
        feature_frames,
        audio.samples.size(),
        tokens,
    });
    return tokens;
}

runtime::TaskResult Vevo2Session::run(const runtime::TaskRequest & request) {
    require_prepared("Vevo2 run()");
    const auto total_start = Clock::now();
    double prosody_tokenizer_ms = 0.0;
    double style_whisper_ms = 0.0;
    double style_tokenizer_ms = 0.0;
    double prompt_build_ms = 0.0;
    double ar_generate_ms = 0.0;
    double source_whisper_ms = 0.0;
    double source_tokenizer_ms = 0.0;
    double timbre_whisper_ms = 0.0;
    double timbre_tokenizer_ms = 0.0;
    double fm_ms = 0.0;
    double vocoder_ms = 0.0;
    const auto text_chunk_size_override = engine::text::parse_text_chunk_size_override(request.options);
    auto vevo2_request = make_request(request);
    std::vector<Vevo2Request> chunk_requests;
    if (vevo2_request.path == Vevo2InferencePath::TextProsodyToTargetVoice) {
        const int64_t text_chunk_size = text_chunk_size_override.value_or(kDefaultTextChunkSize);
        const auto text_chunks = engine::text::split_text_chunks(
            vevo2_request.refs.target_text,
            text_chunk_size);
        engine::debug::trace_log_scalar("vevo2.text_chunk_size", text_chunk_size);
        engine::debug::trace_log_scalar("vevo2.text_chunk_count", static_cast<int64_t>(text_chunks.empty() ? 1 : text_chunks.size()));
        chunk_requests.reserve(text_chunks.empty() ? 1 : text_chunks.size());
        if (text_chunks.empty()) {
            chunk_requests.push_back(std::move(vevo2_request));
        } else {
            const bool split_across_chunks = text_chunks.size() > 1;
            for (const auto & text_chunk : text_chunks) {
                Vevo2Request chunk_request = vevo2_request;
                chunk_request.refs.target_text = text_chunk;
                if (split_across_chunks && chunk_request.route == Vevo2RouteKind::ZeroShotTts) {
                    chunk_request.refs.style_ref_text.clear();
                }
                chunk_requests.push_back(std::move(chunk_request));
            }
        }
    } else {
        chunk_requests.push_back(std::move(vevo2_request));
    }

    runtime::TaskResult result;
    runtime::AudioBuffer merged_audio;
    bool have_audio_output = false;
    Vevo2TokenSequence cached_prosody_tokens;
    bool have_cached_prosody_tokens = false;

    for (const auto & chunk_request : chunk_requests) {
        Vevo2TokenSequence prosody_tokens;
        Vevo2TokenSequence style_content_tokens;
        Vevo2TokenSequence generated_tokens;
        Vevo2PromptParts prompt;

        if (chunk_request.path == Vevo2InferencePath::TextProsodyToTargetVoice) {
            if (chunk_request.generation.use_prosody_code) {
                if (!have_cached_prosody_tokens) {
                    const auto start = Clock::now();
                    cached_prosody_tokens = prosody_tokenizer_.encode(
                        *chunk_request.refs.prosody_audio,
                        chunk_request.refs.style_ref_audio,
                        chunk_request.generation);
                    prosody_tokenizer_ms += engine::debug::elapsed_ms(start);
                    have_cached_prosody_tokens = true;
                }
                prosody_tokens = cached_prosody_tokens;
            }
            std::optional<std::vector<float>> style_whisper_features;
            int64_t style_frames = 0;
            if (chunk_request.refs.style_ref_audio.has_value()) {
                style_frames = frame_count_24k(*chunk_request.refs.style_ref_audio);
                const auto start = Clock::now();
                style_whisper_features = cached_whisper_features(*chunk_request.refs.style_ref_audio, style_frames, 0);
                style_whisper_ms += engine::debug::elapsed_ms(start);
            }
            const auto style_tokenizer_start = Clock::now();
            if (chunk_request.refs.style_ref_audio.has_value()) {
                if (chunk_request.generation.use_pitch_shift) {
                    style_content_tokens = content_style_tokenizer_.encode_style_reference(
                        chunk_request.refs.style_ref_audio,
                        style_whisper_features,
                        style_frames,
                        chunk_request.generation);
                } else {
                    style_content_tokens = cached_content_style_tokens(
                        *chunk_request.refs.style_ref_audio,
                        *style_whisper_features,
                        style_frames);
                }
            } else {
                style_content_tokens = content_style_tokenizer_.encode_style_reference(
                    chunk_request.refs.style_ref_audio,
                    style_whisper_features,
                    style_frames,
                    chunk_request.generation);
            }
            style_tokenizer_ms += engine::debug::elapsed_ms(style_tokenizer_start);
            const auto prompt_start = Clock::now();
            prompt = prompt_builder_.build(chunk_request, prosody_tokens, style_content_tokens);
            prompt_build_ms += engine::debug::elapsed_ms(prompt_start);
            const auto ar_start = Clock::now();
            generated_tokens = autoregressive_model_.generate_content_style(prompt, chunk_request.generation);
            ar_generate_ms += engine::debug::elapsed_ms(ar_start);
        } else {
            const int64_t source_frames = frame_count_24k(*chunk_request.refs.source_audio);
            const auto source_whisper_start = Clock::now();
            const auto source_whisper_features = cached_whisper_features(
                *chunk_request.refs.source_audio,
                source_frames,
                static_cast<size_t>(reference_execution_context_.config().threads));
            source_whisper_ms += engine::debug::elapsed_ms(source_whisper_start);
            const auto source_tokenizer_start = Clock::now();
            if (chunk_request.generation.use_pitch_shift) {
                generated_tokens = content_style_tokenizer_.encode_shifted_reference(
                    *chunk_request.refs.source_audio,
                    source_whisper_features,
                    source_frames,
                    chunk_request.generation.source_shift_steps);
            } else {
                generated_tokens = cached_content_style_tokens(
                    *chunk_request.refs.source_audio,
                    source_whisper_features,
                    source_frames);
            }
            source_tokenizer_ms += engine::debug::elapsed_ms(source_tokenizer_start);
        }

        const int64_t timbre_frames = frame_count_24k(chunk_request.refs.timbre_ref_audio);
        const auto timbre_whisper_start = Clock::now();
        const auto timbre_whisper_features = cached_whisper_features(
            chunk_request.refs.timbre_ref_audio,
            timbre_frames,
            0);
        timbre_whisper_ms += engine::debug::elapsed_ms(timbre_whisper_start);
        const auto timbre_tokenizer_start = Clock::now();
        const auto timbre_tokens = cached_content_style_tokens(
            chunk_request.refs.timbre_ref_audio,
            timbre_whisper_features,
            timbre_frames);
        timbre_tokenizer_ms += engine::debug::elapsed_ms(timbre_tokenizer_start);
        const auto fm_start = Clock::now();
        const auto mel = flow_matching_model_.generate_mel(
            chunk_request.refs.timbre_ref_audio,
            timbre_tokens,
            generated_tokens,
            chunk_request.generation);
        fm_ms += engine::debug::elapsed_ms(fm_start);

        const auto vocoder_start = Clock::now();
        const auto chunk_audio = vocoder_.decode(mel);
        vocoder_ms += engine::debug::elapsed_ms(vocoder_start);
        if (!have_audio_output) {
            merged_audio = chunk_audio;
            have_audio_output = true;
        } else {
            runtime::append_audio_buffer(merged_audio, chunk_audio);
        }
    }

    if (have_audio_output) {
        result.audio_output = std::move(merged_audio);
    }
    engine::debug::timing_log_scalar("vevo2.session.prosody_tokenizer_ms", prosody_tokenizer_ms);
    engine::debug::timing_log_scalar("vevo2.session.style_whisper_ms", style_whisper_ms);
    engine::debug::timing_log_scalar("vevo2.session.style_tokenizer_ms", style_tokenizer_ms);
    engine::debug::timing_log_scalar("vevo2.session.prompt_build_ms", prompt_build_ms);
    engine::debug::timing_log_scalar("vevo2.session.ar_generate_ms", ar_generate_ms);
    engine::debug::timing_log_scalar("vevo2.session.source_whisper_ms", source_whisper_ms);
    engine::debug::timing_log_scalar("vevo2.session.source_tokenizer_ms", source_tokenizer_ms);
    engine::debug::timing_log_scalar("vevo2.session.timbre_whisper_ms", timbre_whisper_ms);
    engine::debug::timing_log_scalar("vevo2.session.timbre_tokenizer_ms", timbre_tokenizer_ms);
    engine::debug::timing_log_scalar("vevo2.session.fm_ms", fm_ms);
    engine::debug::timing_log_scalar("vevo2.session.vocoder_ms", vocoder_ms);
    engine::debug::timing_log_scalar("session.wall_ms", engine::debug::elapsed_ms(total_start));
    return result;
}

Vevo2Request Vevo2Session::make_request(const runtime::TaskRequest & request) const {
    Vevo2Request out;
    const std::string route_name = runtime::find_option(request.options, {"route"}).value_or(default_route_for_task(task_.task));
    const auto parsed = parse_route(route_name);
    if (!route_matches_task(parsed.route, task_.task)) {
        throw std::runtime_error("Vevo2 route " + route_name + " is not valid for task " +
            std::string(runtime::to_string(task_.task)));
    }
    out.path = parsed.path;
    out.route = parsed.route;
    out.refs.target_text = request.text_input.has_value() ? request.text_input->text : std::string{};
    out.generation.top_k = static_cast<int>(assets_->config.ar.generation_top_k);
    out.generation.top_p = assets_->config.ar.generation_top_p;
    out.generation.temperature = assets_->config.ar.generation_temperature;
    out.generation.repetition_penalty = assets_->config.ar.generation_repetition_penalty;
    out.generation.use_prosody_code = route_defaults_to_prosody(out.route);
    out.generation.use_pitch_shift = route_defaults_to_pitch_shift(out.route);
    if (const auto target_text = runtime::find_option(request.options, {"target_text"})) {
        out.refs.target_text = *target_text;
    }
    if (out.path == Vevo2InferencePath::TextProsodyToTargetVoice && out.refs.target_text.empty()) {
        throw std::runtime_error("Vevo2 text/prosody route requires text_input or target_text");
    }

    if (const auto style_ref_text = runtime::find_option(request.options, {"style_ref_text"})) {
        out.refs.style_ref_text = *style_ref_text;
    } else if (const auto reference_text = runtime::find_option(request.options, {"reference_text"})) {
        out.refs.style_ref_text = *reference_text;
    }
    if (const auto value = runtime::find_option(request.options, {"use_prosody_code"})) {
        out.generation.use_prosody_code = runtime::parse_bool_option(*value, "use_prosody_code");
    }
    if (const auto value = runtime::find_option(request.options, {"predict_target_prosody"})) {
        out.generation.predict_target_prosody = runtime::parse_bool_option(*value, "predict_target_prosody");
    }
    if (const auto value = runtime::parse_int_option(request.options, {"top_k"})) {
        out.generation.top_k = *value;
    }
    if (const auto value = runtime::parse_float_option(request.options, {"top_p"})) {
        out.generation.top_p = *value;
    }
    if (const auto value = runtime::parse_float_option(request.options, {"temperature"})) {
        out.generation.temperature = *value;
    }
    if (const auto value = runtime::parse_float_option(request.options, {"repetition_penalty"})) {
        out.generation.repetition_penalty = *value;
    }
    if (const auto value = runtime::parse_int_option(request.options, {"max_tokens"})) {
        out.generation.max_new_tokens = *value;
    }
    out.generation.seed = runtime::parse_u32_option(request.options, {"seed"})
        .value_or(runtime::random_u32_seed());
    if (const auto value = runtime::find_option(request.options, {"use_pitch_shift"})) {
        out.generation.use_pitch_shift = runtime::parse_bool_option(*value, "use_pitch_shift");
    }
    if (const auto value = runtime::parse_int_option(request.options, {"source_shift_steps"})) {
        out.generation.source_shift_steps = *value;
    }
    if (const auto value = runtime::parse_int_option(request.options, {"prosody_shift_steps"})) {
        out.generation.prosody_shift_steps = *value;
    }
    if (const auto value = runtime::parse_int_option(request.options, {"style_shift_steps"})) {
        out.generation.style_shift_steps = *value;
    }
    if (const auto value = runtime::parse_float_option(request.options, {"target_duration_seconds"})) {
        out.generation.target_duration_seconds = *value;
    }
    if (const auto value = runtime::parse_float_option(request.options, {"reference_duration_seconds"})) {
        out.generation.reference_duration_seconds = *value;
    }
    if (const auto value = runtime::parse_int_option(request.options, {"num_inference_steps"})) {
        out.generation.num_inference_steps = *value;
    }
    if (out.generation.num_inference_steps <= 0) {
        throw std::runtime_error("Vevo2 num_inference_steps must be positive");
    }
    if (const auto fm_noise_file = runtime::find_option(request.options, {"fm_noise_file"})) {
        out.generation.fm_noise_file = *fm_noise_file;
    }
    if (out.generation.max_new_tokens <= 0) {
        throw std::runtime_error("Vevo2 max_tokens must be positive");
    }
    if (!(out.generation.temperature > 0.0F)) {
        throw std::runtime_error("Vevo2 temperature must be positive");
    }
    if (!(out.generation.repetition_penalty > 0.0F)) {
        throw std::runtime_error("Vevo2 repetition_penalty must be positive");
    }

    if (const auto source_path = runtime::find_option(request.options, {"source_audio"})) {
        out.refs.source_audio = load_audio_24k_mono(*source_path);
    } else if (out.path == Vevo2InferencePath::SourceAudioToTargetVoice && request.audio_input.has_value()) {
        out.refs.source_audio = normalize_audio_to_24k_mono(*request.audio_input);
    }
    if (out.path == Vevo2InferencePath::SourceAudioToTargetVoice && !out.refs.source_audio.has_value()) {
        throw std::runtime_error("Vevo2 source-audio route requires source_audio or audio_input");
    }

    if (const auto prosody_path = runtime::find_option(request.options, {"prosody_ref"})) {
        out.refs.prosody_audio = load_audio_24k_mono(*prosody_path);
    } else if (out.path == Vevo2InferencePath::TextProsodyToTargetVoice && request.audio_input.has_value()) {
        out.refs.prosody_audio = normalize_audio_to_24k_mono(*request.audio_input);
    } else if (out.path == Vevo2InferencePath::TextProsodyToTargetVoice &&
        route_uses_source_audio_as_reference(out.route)) {
        if (const auto source_path = runtime::find_option(request.options, {"source_audio"})) {
            out.refs.prosody_audio = load_audio_24k_mono(*source_path);
        }
    }
    if (out.path == Vevo2InferencePath::TextProsodyToTargetVoice && out.generation.use_prosody_code &&
        !out.refs.prosody_audio.has_value()) {
        throw std::runtime_error("Vevo2 use_prosody_code requires prosody_ref or audio_input");
    }

    if (const auto style_path = runtime::find_option(request.options, {"style_ref"})) {
        out.refs.style_ref_audio = load_audio_24k_mono(*style_path);
    } else if (out.path == Vevo2InferencePath::TextProsodyToTargetVoice &&
        out.route == Vevo2RouteKind::Editing) {
        if (out.refs.prosody_audio.has_value()) {
            out.refs.style_ref_audio = out.refs.prosody_audio;
        }
    }

    if (const auto timbre_path = runtime::find_option(request.options, {"target_voice"})) {
        out.refs.timbre_ref_audio = load_audio_24k_mono(*timbre_path);
    } else if (out.path == Vevo2InferencePath::TextProsodyToTargetVoice &&
        route_uses_source_audio_as_reference(out.route) &&
        out.refs.prosody_audio.has_value()) {
        out.refs.timbre_ref_audio = *out.refs.prosody_audio;
    } else if (request.voice.has_value() && request.voice->speaker.has_value() &&
        request.voice->speaker->audio.has_value()) {
        out.refs.timbre_ref_audio = normalize_audio_to_24k_mono(*request.voice->speaker->audio);
    } else {
        throw std::runtime_error("Vevo2 requires target_voice or voice speaker audio");
    }

    if (out.generation.reference_duration_seconds.has_value()) {
        trim_audio_seconds(out.refs.timbre_ref_audio, *out.generation.reference_duration_seconds);
    }
    if (out.generation.use_pitch_shift) {
        if (out.path == Vevo2InferencePath::SourceAudioToTargetVoice && out.generation.source_shift_steps == 0) {
            out.generation.source_shift_steps =
                estimate_pitch_shift_steps(*out.refs.source_audio, out.refs.timbre_ref_audio);
        }
        if (out.path == Vevo2InferencePath::TextProsodyToTargetVoice && out.refs.prosody_audio.has_value() &&
            out.generation.prosody_shift_steps == 0) {
            out.generation.prosody_shift_steps =
                estimate_pitch_shift_steps(*out.refs.prosody_audio, out.refs.timbre_ref_audio);
        }
        if (out.path == Vevo2InferencePath::TextProsodyToTargetVoice && out.refs.style_ref_audio.has_value() &&
            out.generation.style_shift_steps == 0) {
            out.generation.style_shift_steps =
                estimate_pitch_shift_steps(*out.refs.style_ref_audio, out.refs.timbre_ref_audio);
        }
    }
    return out;
}

}  // namespace engine::models::vevo2
