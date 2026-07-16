#include "engine/models/seed_vc/session.h"

#include "engine/models/seed_vc/assets.h"

#include "engine/framework/modules/bigvgan_vocoder.h"
#include "engine/framework/modules/campplus_encoder.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/debug/trace.h"
#include "engine/framework/modules/hift_vocoder.h"
#include "engine/framework/modules/hubert_encoder.h"
#include "engine/framework/io/binary.h"
#include "engine/framework/runtime/options.h"
#include "engine/framework/sampling/torch_random.h"
#include "engine/models/seed_vc/astral_quantizer.h"
#include "engine/models/seed_vc/audio_features.h"
#include "engine/models/seed_vc/content_features.h"
#include "engine/models/seed_vc/length_regulator.h"
#include "engine/models/seed_vc/rmvpe.h"
#include "engine/models/seed_vc/v1_cfm.h"
#include "engine/models/seed_vc/v2_cfm.h"
#include "engine/models/seed_vc/whisper_content.h"

#include <algorithm>
#include <chrono>
#include <cctype>
#include <cmath>
#include <functional>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace engine::models::seed_vc {
namespace {

constexpr int64_t kSeedVcBigVganActiveFrames = 1408;
constexpr int64_t kSeedVcBigVganOverlapFrames = 32;

std::shared_ptr<const SeedVcAssets> require_assets(std::shared_ptr<const SeedVcAssets> assets) {
    if (assets == nullptr) {
        throw std::runtime_error("Seed-VC session requires assets");
    }
    return assets;
}

}  // namespace

struct SeedVcRouteRuntime {
    enum class Route {
        V2VoiceConversion,
        V1SingingVoiceConversion,
        V1WhisperBigVganVoiceConversion,
        V1XlsrHiftVoiceConversion,
    };

    Route route = Route::V2VoiceConversion;
    SeedVcDiscreteLengthRegulator v2_ar_length_regulator;
    SeedVcCfmLengthRegulator v2_cfm_length_regulator;
    SeedVcV2CfmEstimator v2_cfm_estimator;
    SeedVcAstralQuantizer astral_bsq32_quantizer;
    SeedVcAstralQuantizer astral_bsq2048_quantizer;
    SeedVcV1LengthRegulator v1_length_regulator;
    engine::modules::CampplusEncoderComponent campplus;
    engine::modules::BigVganVocoderComponent bigvgan;
    engine::modules::HiftVocoderComponent hift;
    engine::modules::HubertEncoderComponent hubert_large;
    engine::modules::HubertEncoderComponent wav2vec2_xlsr;
    SeedVcV1CfmEstimator v1_cfm_estimator;
    SeedVcRmvpeF0Extractor rmvpe_extractor;
    SeedVcWhisperContentEncoder whisper_content;
};

struct SeedVcV2RequestConfig {
    int num_inference_steps = 30;
    float length_adjust = 1.0F;
    float intelligibility_cfg_rate = 0.7F;
    float similarity_cfg_rate = 0.7F;
    float top_p = 0.9F;
    float temperature = 1.0F;
    float repetition_penalty = 1.0F;
    bool convert_style = false;
    bool anonymization_only = false;
    uint64_t seed = 1234;
    std::string noise_file;
};

struct SeedVcV1RequestConfig {
    int num_inference_steps = 30;
    float length_adjust = 1.0F;
    float inference_cfg_rate = 0.7F;
    bool f0_condition = false;
    bool auto_f0_adjust = false;
    int semi_tone_shift = 0;
    bool fp16 = true;
};

struct SeedVcExecutionPlan {
    std::string path;
    int source_sample_rate = 0;
    int source_channels = 0;
    int target_sample_rate = 0;
    int target_channels = 0;
    int64_t source_frames = 0;
    int64_t target_frames = 0;
    std::optional<SeedVcV2RequestConfig> v2;
    std::optional<SeedVcV1RequestConfig> v1;
};

std::string request_route_or_default(const runtime::TaskRequest & request, runtime::VoiceTaskKind task);
std::string route_path_or_default_from_options(
    const std::unordered_map<std::string, std::string> & options,
    runtime::VoiceTaskKind task);

engine::modules::BigVganVocoderConfig make_bigvgan_config(
    const SeedVcBigVganConfig & config,
    engine::assets::TensorStorageType weight_storage_type) {
    engine::modules::BigVganVocoderConfig out;
    out.sampling_rate = config.sampling_rate;
    out.num_mels = config.num_mels;
    out.n_fft = config.n_fft;
    out.hop_size = config.hop_size;
    out.win_size = config.win_size;
    out.upsample_initial_channel = config.upsample_initial_channel;
    out.snake_logscale = config.snake_logscale;
    out.upsample_rates = config.upsample_rates;
    out.upsample_kernel_sizes = config.upsample_kernel_sizes;
    out.resblock_kernel_sizes = config.resblock_kernel_sizes;
    out.weight_storage_type = weight_storage_type;
    return out;
}

engine::modules::HiftVocoderConfig make_hift_config(
    const SeedVcHiftConfig & config,
    engine::assets::TensorStorageType weight_storage_type) {
    engine::modules::HiftVocoderConfig out;
    out.in_channels = config.in_channels;
    out.base_channels = config.base_channels;
    out.nb_harmonics = config.nb_harmonics;
    out.sampling_rate = config.sampling_rate;
    out.nsf_alpha = config.nsf_alpha;
    out.nsf_sigma = config.nsf_sigma;
    out.nsf_voiced_threshold = config.nsf_voiced_threshold;
    out.upsample_rates = config.upsample_rates;
    out.upsample_kernel_sizes = config.upsample_kernel_sizes;
    out.istft_n_fft = config.istft_n_fft;
    out.istft_hop = config.istft_hop;
    out.resblock_kernel_sizes = config.resblock_kernel_sizes;
    out.resblock_dilation_sizes = config.resblock_dilation_sizes;
    out.source_resblock_kernel_sizes = config.source_resblock_kernel_sizes;
    out.source_resblock_dilation_sizes = config.source_resblock_dilation_sizes;
    out.lrelu_slope = config.lrelu_slope;
    out.audio_limit = config.audio_limit;
    out.f0_num_class = config.f0_num_class;
    out.f0_in_channels = config.f0_in_channels;
    out.f0_cond_channels = config.f0_cond_channels;
    out.weight_storage_type = weight_storage_type;
    return out;
}

bool is_v1_voice_conversion_path(const std::string & path) {
    return path == "v1_whisper_bigvgan_vc" || path == "v1_xlsr_hift_vc";
}

bool is_v1_path(const std::string & path) {
    return path == "v1_svc" || is_v1_voice_conversion_path(path);
}

bool route_is_v1(const SeedVcRouteRuntime::Route route) {
    return route == SeedVcRouteRuntime::Route::V1SingingVoiceConversion ||
        route == SeedVcRouteRuntime::Route::V1WhisperBigVganVoiceConversion ||
        route == SeedVcRouteRuntime::Route::V1XlsrHiftVoiceConversion;
}

std::optional<engine::assets::TensorStorageType> parse_seed_vc_weight_type(
    const runtime::SessionOptions & options) {
    const auto it = options.options.find("seed_vc.weight_type");
    if (it == options.options.end()) {
        return std::nullopt;
    }
    const auto storage_type = engine::assets::parse_tensor_storage_type(it->second);
    if (storage_type == engine::assets::TensorStorageType::Native ||
        storage_type == engine::assets::TensorStorageType::F32 ||
        storage_type == engine::assets::TensorStorageType::F16 ||
        storage_type == engine::assets::TensorStorageType::BF16 ||
        storage_type == engine::assets::TensorStorageType::Q8_0) {
        return storage_type;
    }
    throw std::runtime_error("seed_vc.weight_type currently supports only native, f32, f16, bf16, and q8_0");
}

const SeedVcMelConfig & v1_mel_config_for_path(const SeedVcAssets & assets, const std::string & path) {
    if (path == "v1_svc") {
        return assets.config.v1_mel;
    }
    if (path == "v1_whisper_bigvgan_vc") {
        return assets.config.v1_whisper_bigvgan_mel;
    }
    if (path == "v1_xlsr_hift_vc") {
        return assets.config.v1_xlsr_hift_mel;
    }
    throw std::runtime_error("Seed-VC unsupported V1 path: " + path);
}

const SeedVcV1DitConfig & v1_dit_config_for_path(const SeedVcAssets & assets, const std::string & path) {
    if (path == "v1_svc") {
        return assets.config.v1_dit;
    }
    if (path == "v1_whisper_bigvgan_vc") {
        return assets.config.v1_whisper_bigvgan_dit;
    }
    if (path == "v1_xlsr_hift_vc") {
        return assets.config.v1_xlsr_hift_dit;
    }
    throw std::runtime_error("Seed-VC unsupported V1 path: " + path);
}

int64_t v1_style_dim_for_path(const SeedVcAssets & assets, const std::string & path) {
    if (path == "v1_svc") {
        return assets.config.v1_style_dim;
    }
    if (path == "v1_whisper_bigvgan_vc") {
        return assets.config.v1_whisper_bigvgan_style_dim;
    }
    if (path == "v1_xlsr_hift_vc") {
        return assets.config.v1_xlsr_hift_style_dim;
    }
    throw std::runtime_error("Seed-VC unsupported V1 path: " + path);
}

void validate_positive_audio(const runtime::AudioBuffer & audio, const char * role) {
    if (audio.sample_rate <= 0) {
        throw std::runtime_error(std::string("Seed-VC ") + role + " audio sample rate must be positive");
    }
    if (audio.channels <= 0) {
        throw std::runtime_error(std::string("Seed-VC ") + role + " audio channel count must be positive");
    }
    if (audio.samples.empty()) {
        throw std::runtime_error(std::string("Seed-VC ") + role + " audio must not be empty");
    }
    if (audio.samples.size() % static_cast<size_t>(audio.channels) != 0) {
        throw std::runtime_error(std::string("Seed-VC ") + role + " audio samples must be divisible by channels");
    }
}

const runtime::AudioBuffer & require_source_audio(const runtime::TaskRequest & request) {
    if (!request.audio_input.has_value()) {
        throw std::runtime_error("Seed-VC request requires source audio_input");
    }
    validate_positive_audio(*request.audio_input, "source");
    return *request.audio_input;
}

const runtime::AudioBuffer & require_target_audio(const runtime::TaskRequest & request) {
    if (!request.voice.has_value() || !request.voice->speaker.has_value() ||
        !request.voice->speaker->audio.has_value()) {
        throw std::runtime_error("Seed-VC request requires target speaker reference audio");
    }
    validate_positive_audio(*request.voice->speaker->audio, "target");
    return *request.voice->speaker->audio;
}

void validate_common_generation_options(int num_inference_steps, float length_adjust) {
    if (num_inference_steps <= 0) {
        throw std::runtime_error("Seed-VC num_inference_steps must be positive");
    }
    if (!(length_adjust > 0.0F)) {
        throw std::runtime_error("Seed-VC length_adjust must be positive");
    }
}

float median_like_torch_1d(std::vector<float> values) {
    if (values.empty()) {
        throw std::runtime_error("Seed-VC V1 f0 adjustment requires voiced F0 values");
    }
    const size_t median_index = (values.size() - 1) / 2;
    std::nth_element(values.begin(), values.begin() + static_cast<std::ptrdiff_t>(median_index), values.end());
    return values[median_index];
}

std::vector<float> adjust_source_f0_like_python(
    const std::vector<float> & source_f0,
    const std::vector<float> & target_f0,
    bool auto_f0_adjust,
    int semi_tone_shift) {
    std::vector<float> source_voiced_log;
    std::vector<float> target_voiced_log;
    source_voiced_log.reserve(source_f0.size());
    target_voiced_log.reserve(target_f0.size());
    for (const float value : source_f0) {
        if (value > 1.0F) {
            source_voiced_log.push_back(std::log(value + 1.0e-5F));
        }
    }
    for (const float value : target_f0) {
        if (value > 1.0F) {
            target_voiced_log.push_back(std::log(value + 1.0e-5F));
        }
    }
    const float source_median = median_like_torch_1d(std::move(source_voiced_log));
    const float target_median = median_like_torch_1d(std::move(target_voiced_log));
    const float pitch_scale = std::pow(2.0F, static_cast<float>(semi_tone_shift) / 12.0F);
    std::vector<float> shifted(source_f0.size(), 0.0F);
    for (size_t index = 0; index < source_f0.size(); ++index) {
        const float value = source_f0[index];
        float log_value = std::log(value + 1.0e-5F);
        if (value > 1.0F && auto_f0_adjust) {
            log_value = log_value - source_median + target_median;
        }
        float out = std::exp(log_value);
        if (value > 1.0F && semi_tone_shift != 0) {
            out *= pitch_scale;
        }
        shifted[index] = out;
    }
    return shifted;
}

SeedVcV2RequestConfig parse_v2_config(const runtime::TaskRequest & request) {
    SeedVcV2RequestConfig config;
    config.num_inference_steps = runtime::parse_int_option(
        request.options,
        {"num_inference_steps"})
        .value_or(config.num_inference_steps);
    config.length_adjust = runtime::parse_finite_float_option(
        request.options,
        {"length_adjust"})
        .value_or(config.length_adjust);
    config.intelligibility_cfg_rate = runtime::parse_finite_float_option(
        request.options,
        {"intelligibility_cfg_rate"})
        .value_or(config.intelligibility_cfg_rate);
    config.similarity_cfg_rate = runtime::parse_finite_float_option(
        request.options,
        {"similarity_cfg_rate"})
        .value_or(config.similarity_cfg_rate);
    config.top_p = runtime::parse_finite_float_option(request.options, {"top_p"})
        .value_or(config.top_p);
    config.temperature = runtime::parse_finite_float_option(request.options, {"temperature"})
        .value_or(config.temperature);
    config.repetition_penalty = runtime::parse_finite_float_option(
        request.options,
        {"repetition_penalty"})
        .value_or(config.repetition_penalty);
    if (const auto value = runtime::find_option(request.options, {"convert_style"})) {
        config.convert_style = runtime::parse_bool_option(*value, "convert_style");
    }
    if (const auto value = runtime::find_option(request.options, {"anonymization_only"})) {
        config.anonymization_only = runtime::parse_bool_option(*value, "anonymization_only");
    }
    config.seed = runtime::parse_u64_option(request.options, {"seed"})
        .value_or(runtime::random_u64_seed());
    config.noise_file = runtime::find_option(request.options, {"noise_file"}).value_or("");
    validate_common_generation_options(config.num_inference_steps, config.length_adjust);
    if (!(config.top_p > 0.0F && config.top_p <= 1.0F)) {
        throw std::runtime_error("Seed-VC top_p must be in (0, 1]");
    }
    if (!(config.temperature > 0.0F)) {
        throw std::runtime_error("Seed-VC temperature must be positive");
    }
    if (!(config.repetition_penalty > 0.0F)) {
        throw std::runtime_error("Seed-VC repetition_penalty must be positive");
    }
    return config;
}

std::vector<float> make_full_prompt_mel(
    const std::vector<float> & target_mel,
    int64_t channels,
    int64_t target_frames,
    int64_t total_frames) {
    if (target_frames < 0 || target_frames > total_frames ||
        static_cast<int64_t>(target_mel.size()) != channels * target_frames) {
        throw std::runtime_error("Seed-VC V2 prompt mel shape mismatch");
    }
    std::vector<float> out(static_cast<size_t>(channels * total_frames), 0.0F);
    for (int64_t channel = 0; channel < channels; ++channel) {
        std::copy(
            target_mel.begin() + static_cast<std::ptrdiff_t>(channel * target_frames),
            target_mel.begin() + static_cast<std::ptrdiff_t>((channel + 1) * target_frames),
            out.begin() + static_cast<std::ptrdiff_t>(channel * total_frames));
    }
    return out;
}

std::vector<float> concat_conditions(
    const std::vector<float> & prompt,
    int64_t prompt_frames,
    const std::vector<float> & source,
    int64_t source_start,
    int64_t source_frames,
    int64_t channels) {
    if (prompt_frames < 0 || source_start < 0 || source_frames <= 0 ||
        static_cast<int64_t>(prompt.size()) != prompt_frames * channels) {
        throw std::runtime_error("Seed-VC V2 condition shape mismatch");
    }
    const int64_t source_total_frames = static_cast<int64_t>(source.size()) / channels;
    if (static_cast<int64_t>(source.size()) != source_total_frames * channels ||
        source_start + source_frames > source_total_frames) {
        throw std::runtime_error("Seed-VC V2 source condition slice is out of range");
    }
    std::vector<float> out(static_cast<size_t>((prompt_frames + source_frames) * channels), 0.0F);
    std::copy(prompt.begin(), prompt.end(), out.begin());
    const auto src_begin = source.begin() + static_cast<std::ptrdiff_t>(source_start * channels);
    const auto src_end = src_begin + static_cast<std::ptrdiff_t>(source_frames * channels);
    std::copy(src_begin, src_end, out.begin() + static_cast<std::ptrdiff_t>(prompt.size()));
    return out;
}

std::vector<float> slice_mel_source_region(
    const std::vector<float> & mel,
    int64_t channels,
    int64_t total_frames,
    int64_t start_frame,
    int64_t end_frame) {
    if (start_frame < 0 || end_frame < start_frame || end_frame > total_frames ||
        static_cast<int64_t>(mel.size()) != channels * total_frames) {
        throw std::runtime_error("Seed-VC V2 mel slice is out of range");
    }
    const int64_t frames = end_frame - start_frame;
    std::vector<float> out(static_cast<size_t>(channels * frames), 0.0F);
    for (int64_t channel = 0; channel < channels; ++channel) {
        const auto src = mel.begin() + static_cast<std::ptrdiff_t>(channel * total_frames + start_frame);
        std::copy(
            src,
            src + static_cast<std::ptrdiff_t>(frames),
            out.begin() + static_cast<std::ptrdiff_t>(channel * frames));
    }
    return out;
}

std::vector<float> crossfade_wave(
    const std::vector<float> & previous_tail,
    const std::vector<float> & current,
    int64_t overlap) {
    if (overlap < 0 || static_cast<int64_t>(previous_tail.size()) < overlap) {
        throw std::runtime_error("Seed-VC V2 crossfade overlap is invalid");
    }
    std::vector<float> out = current;
    const int64_t effective = std::min<int64_t>(overlap, static_cast<int64_t>(out.size()));
    constexpr double kPi = 3.141592653589793238462643383279502884;
    for (int64_t index = 0; index < effective; ++index) {
        const double fade_out = std::pow(std::cos((static_cast<double>(index) / static_cast<double>(overlap - 1)) * kPi * 0.5), 2.0);
        const double fade_in = std::pow(std::cos((1.0 - static_cast<double>(index) / static_cast<double>(overlap - 1)) * kPi * 0.5), 2.0);
        out[static_cast<size_t>(index)] =
            static_cast<float>(static_cast<double>(out[static_cast<size_t>(index)]) * fade_in +
                static_cast<double>(previous_tail[static_cast<size_t>(index)]) * fade_out);
    }
    return out;
}

std::vector<float> without_tail(const std::vector<float> & wave, int64_t tail) {
    if (tail <= 0 || static_cast<int64_t>(wave.size()) <= tail) {
        return {};
    }
    return std::vector<float>(wave.begin(), wave.end() - static_cast<std::ptrdiff_t>(tail));
}

std::vector<float> tail_of(const std::vector<float> & wave, int64_t tail) {
    if (tail <= 0 || wave.empty()) {
        return {};
    }
    const int64_t count = std::min<int64_t>(tail, static_cast<int64_t>(wave.size()));
    return std::vector<float>(wave.end() - static_cast<std::ptrdiff_t>(count), wave.end());
}

void append_wave(std::vector<float> & output, const std::vector<float> & chunk) {
    output.insert(output.end(), chunk.begin(), chunk.end());
}

std::vector<float> synthesize_bigvgan_fixed_chunks(
    const engine::modules::BigVganVocoderComponent & bigvgan,
    const std::vector<float> & mel,
    int64_t frames,
    int64_t hop_size,
    int64_t active_frames,
    int64_t overlap_frames) {
    const int64_t channels = bigvgan.num_mels();
    if (frames <= 0 || channels <= 0 || hop_size <= 0 || active_frames <= 0 || overlap_frames < 0 ||
        static_cast<int64_t>(mel.size()) != channels * frames) {
        throw std::runtime_error("Seed-VC BigVGAN fixed chunk synthesis received invalid dimensions");
    }
    const int64_t graph_frames = active_frames + 2 * overlap_frames;
    if (graph_frames < active_frames) {
        throw std::runtime_error("Seed-VC BigVGAN fixed chunk graph size is invalid");
    }

    std::vector<float> waveform;
    waveform.reserve(static_cast<size_t>(frames * hop_size));
    std::vector<float> chunk_mel(static_cast<size_t>(channels * graph_frames), 0.0F);
    for (int64_t position = 0; position < frames; position += active_frames) {
        const int64_t active = std::min(active_frames, frames - position);
        const int64_t source_start = std::max<int64_t>(0, position - overlap_frames);
        const int64_t source_end = std::min<int64_t>(frames, position + active + overlap_frames);
        const int64_t left_context = position - source_start;
        const int64_t target_start = overlap_frames - left_context;
        const int64_t copy_frames = source_end - source_start;
        if (target_start < 0 || target_start + copy_frames > graph_frames) {
            throw std::runtime_error("Seed-VC BigVGAN fixed chunk copy range is invalid");
        }

        std::fill(chunk_mel.begin(), chunk_mel.end(), 0.0F);
        for (int64_t channel = 0; channel < channels; ++channel) {
            const auto source = mel.begin() + static_cast<std::ptrdiff_t>(channel * frames + source_start);
            std::copy_n(
                source,
                copy_frames,
                chunk_mel.begin() + static_cast<std::ptrdiff_t>(channel * graph_frames + target_start));
        }

        const auto chunk_wave = bigvgan.synthesize(chunk_mel, graph_frames).waveform;
        const int64_t sample_start = overlap_frames * hop_size;
        const int64_t sample_end = sample_start + active * hop_size;
        if (sample_start < 0 || sample_end > static_cast<int64_t>(chunk_wave.size())) {
            throw std::runtime_error("Seed-VC BigVGAN fixed chunk crop range is invalid");
        }
        waveform.insert(
            waveform.end(),
            chunk_wave.begin() + static_cast<std::ptrdiff_t>(sample_start),
            chunk_wave.begin() + static_cast<std::ptrdiff_t>(sample_end));
    }
    return waveform;
}

std::vector<float> load_seed_vc_noise_or_sample(
    const std::string & noise_file,
    size_t count,
    uint64_t seed,
    uint64_t offset) {
    if (noise_file.empty()) {
        return engine::sampling::generate_torch_cuda_randn(
            count,
            seed,
            engine::sampling::TorchRandnPrecision::Float32,
            offset);
    }
    auto values = engine::io::read_f32_file(noise_file);
    if (values.size() < offset + count) {
        throw std::runtime_error(
            "Seed-VC noise file is too short: expected at least " +
            std::to_string(offset + count) + " floats, got " + std::to_string(values.size()));
    }
    std::vector<float> out(count);
    std::copy(
        values.begin() + static_cast<std::ptrdiff_t>(offset),
        values.begin() + static_cast<std::ptrdiff_t>(offset + count),
        out.begin());
    return out;
}

runtime::TaskResult run_v2_voice_conversion(
    const runtime::TaskRequest & request,
    const SeedVcExecutionPlan & plan,
    const SeedVcRouteRuntime & sources,
    const SeedVcAssets & assets,
    size_t threads) {
    if (!plan.v2.has_value()) {
        throw std::runtime_error("Seed-VC V2 route requires V2 config");
    }
    const auto & config = *plan.v2;
    if (config.convert_style) {
        throw std::runtime_error("Seed-VC V2 convert_style path requires AR generation and is not implemented yet");
    }
    if (sources.route != SeedVcRouteRuntime::Route::V2VoiceConversion) {
        throw std::runtime_error("Seed-VC V2 route requires V2 component sources");
    }
    constexpr int64_t kDitMaxContextSeconds = 30;
    constexpr int64_t kOverlapFrameLen = 16;
    const auto & source_audio = require_source_audio(request);
    const auto & target_audio = require_target_audio(request);
    const int64_t target_limit_samples =
        static_cast<int64_t>(assets.config.v2_mel.sample_rate) * (kDitMaxContextSeconds - 5);
    auto timing_start = std::chrono::steady_clock::now();
    const auto source_prepared = seed_vc_prepare_audio(
        source_audio.samples,
        source_audio.sample_rate,
        source_audio.channels);
    const auto target_prepared = seed_vc_prepare_audio(
        target_audio.samples,
        target_audio.sample_rate,
        target_audio.channels,
        target_limit_samples);
    auto timing_end = std::chrono::steady_clock::now();
    engine::debug::timing_log_scalar(
        "seed_vc.v2.prepare_audio_ms",
        engine::debug::elapsed_ms(timing_start, timing_end));

    timing_start = std::chrono::steady_clock::now();
    const auto source_mel = compute_seed_vc_mel_spectrogram(
        source_prepared.waveform_22k,
        assets.config.v2_mel,
        threads);
    timing_end = std::chrono::steady_clock::now();
    engine::debug::timing_log_scalar(
        "seed_vc.v2.source_mel_ms",
        engine::debug::elapsed_ms(timing_start, timing_end));
    timing_start = std::chrono::steady_clock::now();
    const auto target_mel = compute_seed_vc_mel_spectrogram(
        target_prepared.waveform_22k,
        assets.config.v2_mel,
        threads);
    timing_end = std::chrono::steady_clock::now();
    engine::debug::timing_log_scalar(
        "seed_vc.v2.target_mel_ms",
        engine::debug::elapsed_ms(timing_start, timing_end));
    if (source_mel.channels != assets.config.v2_cfm.in_channels ||
        target_mel.channels != assets.config.v2_cfm.in_channels) {
        throw std::runtime_error("Seed-VC V2 mel channel count does not match CFM input");
    }

    SeedVcContentFeatureExtractor content_extractor(
        &sources.hubert_large,
        &sources.astral_bsq2048_quantizer,
        &sources.astral_bsq32_quantizer);
    timing_start = std::chrono::steady_clock::now();
    const auto source_content = content_extractor.extract_16k_mono(
        source_prepared.waveform_16k,
        SeedVcContentFeatureKind::Wide);
    timing_end = std::chrono::steady_clock::now();
    engine::debug::timing_log_scalar(
        "seed_vc.v2.source_content_ms",
        engine::debug::elapsed_ms(timing_start, timing_end));
    timing_start = std::chrono::steady_clock::now();
    const auto target_content = content_extractor.extract_16k_mono(
        target_prepared.waveform_16k,
        SeedVcContentFeatureKind::Wide);
    timing_end = std::chrono::steady_clock::now();
    engine::debug::timing_log_scalar(
        "seed_vc.v2.target_content_ms",
        engine::debug::elapsed_ms(timing_start, timing_end));

    timing_start = std::chrono::steady_clock::now();
    const auto fbank = compute_seed_vc_campplus_fbank_16k(target_prepared.waveform_16k);
    if (fbank.frames <= 0 || fbank.dims <= 0) {
        throw std::runtime_error("Seed-VC V2 target style fbank is empty");
    }
    const auto target_style = sources.campplus.embed_from_features(
        fbank.features,
        fbank.frames,
        fbank.dims);
    timing_end = std::chrono::steady_clock::now();
    engine::debug::timing_log_scalar(
        "seed_vc.v2.style_ms",
        engine::debug::elapsed_ms(timing_start, timing_end));

    timing_start = std::chrono::steady_clock::now();
    const auto prompt_condition = sources.v2_cfm_length_regulator.run(
        target_content.indices,
        {target_mel.frames},
        target_content.batch,
        target_content.tokens);
    timing_end = std::chrono::steady_clock::now();
    engine::debug::timing_log_scalar(
        "seed_vc.v2.prompt_length_regulator_ms",
        engine::debug::elapsed_ms(timing_start, timing_end));
    timing_start = std::chrono::steady_clock::now();
    const auto source_condition = sources.v2_cfm_length_regulator.run(
        source_content.indices,
        {source_mel.frames},
        source_content.batch,
        source_content.tokens);
    timing_end = std::chrono::steady_clock::now();
    engine::debug::timing_log_scalar(
        "seed_vc.v2.source_length_regulator_ms",
        engine::debug::elapsed_ms(timing_start, timing_end));

    const int64_t max_context_window =
        (assets.config.v2_mel.sample_rate / assets.config.v2_mel.hop_size) * kDitMaxContextSeconds;
    const int64_t max_source_window = max_context_window - target_mel.frames;
    if (max_source_window <= 0) {
        throw std::runtime_error("Seed-VC V2 target prompt is too long for the DiT context window");
    }
    const int64_t overlap_wave_len = kOverlapFrameLen * assets.config.v2_mel.hop_size;
    int64_t processed_frames = 0;
    uint64_t random_offset = 0;
    std::vector<float> generated_wave;
    std::vector<float> previous_chunk;

    while (processed_frames < source_condition.tokens) {
        const int64_t chunk_frames = std::min(max_source_window, source_condition.tokens - processed_frames);
        const bool is_last_chunk = processed_frames + max_source_window >= source_condition.tokens;
        const auto cat_condition = concat_conditions(
            prompt_condition.values,
            target_mel.frames,
            source_condition.values,
            processed_frames,
            chunk_frames,
            source_condition.channels);
        const int64_t original_len = target_mel.frames + chunk_frames;
        auto prompt = make_full_prompt_mel(
            target_mel.mel,
            target_mel.channels,
            target_mel.frames,
            original_len);
        const size_t noise_count =
            static_cast<size_t>(assets.config.v2_cfm.in_channels * original_len);
        auto initial_noise = load_seed_vc_noise_or_sample(
            config.noise_file,
            noise_count,
            config.seed,
            random_offset);
        random_offset += static_cast<uint64_t>(noise_count);

        SeedVcV2CfmInferenceInput cfm_input;
        cfm_input.mu = cat_condition;
        cfm_input.prompt = std::move(prompt);
        cfm_input.style = target_style.embedding;
        cfm_input.initial_noise = std::move(initial_noise);
        cfm_input.batch = 1;
        cfm_input.frames = original_len;
        cfm_input.prompt_frames = target_mel.frames;
        cfm_input.num_inference_steps = config.num_inference_steps;
        cfm_input.temperature = 1.0F;
        cfm_input.intelligibility_cfg_rate = config.intelligibility_cfg_rate;
        cfm_input.similarity_cfg_rate = config.similarity_cfg_rate;
        cfm_input.random_voice = config.anonymization_only;

        timing_start = std::chrono::steady_clock::now();
        const auto cfm_output = sources.v2_cfm_estimator.infer(cfm_input);
        timing_end = std::chrono::steady_clock::now();
        engine::debug::timing_log_scalar(
            "seed_vc.v2.cfm_ms",
            engine::debug::elapsed_ms(timing_start, timing_end));
        const auto vc_mel = slice_mel_source_region(
            cfm_output.velocity,
            cfm_output.channels,
            cfm_output.frames,
            target_mel.frames,
            original_len);
        timing_start = std::chrono::steady_clock::now();
        const auto vc_wave = synthesize_bigvgan_fixed_chunks(
            sources.bigvgan,
            vc_mel,
            chunk_frames,
            assets.config.bigvgan_22k.hop_size,
            kSeedVcBigVganActiveFrames,
            kSeedVcBigVganOverlapFrames);
        timing_end = std::chrono::steady_clock::now();
        engine::debug::timing_log_scalar(
            "seed_vc.v2.bigvgan_ms",
            engine::debug::elapsed_ms(timing_start, timing_end));

        if (processed_frames == 0) {
            if (is_last_chunk) {
                append_wave(generated_wave, vc_wave);
                break;
            }
            append_wave(generated_wave, without_tail(vc_wave, overlap_wave_len));
            previous_chunk = tail_of(vc_wave, overlap_wave_len);
        } else {
            if (is_last_chunk) {
                append_wave(generated_wave, crossfade_wave(previous_chunk, vc_wave, overlap_wave_len));
                break;
            }
            const auto faded = crossfade_wave(previous_chunk, vc_wave, overlap_wave_len);
            append_wave(generated_wave, without_tail(faded, overlap_wave_len));
            previous_chunk = tail_of(vc_wave, overlap_wave_len);
        }
        processed_frames += chunk_frames - kOverlapFrameLen;
        if (processed_frames <= 0) {
            throw std::runtime_error("Seed-VC V2 streaming chunking did not make progress");
        }
    }

    runtime::TaskResult result;
    result.audio_output = runtime::AudioBuffer{
        static_cast<int>(assets.config.bigvgan_22k.sampling_rate),
        1,
        std::move(generated_wave)};
    return result;
}

runtime::TaskResult run_v1_singing_voice_conversion(
    const runtime::TaskRequest & request,
    const SeedVcExecutionPlan & plan,
    const SeedVcRouteRuntime & sources,
    const SeedVcAssets & assets,
    size_t threads) {
    if (!plan.v1.has_value()) {
        throw std::runtime_error("Seed-VC V1 route requires V1 config");
    }
    const auto & config = *plan.v1;
    if (!route_is_v1(sources.route)) {
        throw std::runtime_error("Seed-VC V1 route requires V1 component sources");
    }
    const auto & mel_config = v1_mel_config_for_path(assets, plan.path);
    const auto & dit_config = v1_dit_config_for_path(assets, plan.path);
    constexpr int64_t kDitMaxContextSeconds = 30;
    constexpr int64_t kOverlapFrameLen = 16;
    auto timing_start = std::chrono::steady_clock::now();
    const auto & source_audio = require_source_audio(request);
    const auto & target_audio = require_target_audio(request);
    const int64_t target_limit_samples =
        static_cast<int64_t>(mel_config.sample_rate) * (kDitMaxContextSeconds - 5);
    const auto source_prepared = seed_vc_prepare_audio_for_sample_rate(
        source_audio.samples,
        source_audio.sample_rate,
        source_audio.channels,
        static_cast<int>(mel_config.sample_rate));
    const auto target_prepared = seed_vc_prepare_audio_for_sample_rate(
        target_audio.samples,
        target_audio.sample_rate,
        target_audio.channels,
        static_cast<int>(mel_config.sample_rate),
        target_limit_samples);
    auto timing_end = std::chrono::steady_clock::now();
    engine::debug::timing_log_scalar(
        "seed_vc.v1.audio_prepare_ms",
        engine::debug::elapsed_ms(timing_start, timing_end));
    timing_start = timing_end;

    const auto source_mel = compute_seed_vc_mel_spectrogram(
        source_prepared.waveform,
        mel_config,
        threads);
    const auto target_mel = compute_seed_vc_mel_spectrogram(
        target_prepared.waveform,
        mel_config,
        threads);
    timing_end = std::chrono::steady_clock::now();
    engine::debug::timing_log_scalar(
        "seed_vc.v1.mel_ms",
        engine::debug::elapsed_ms(timing_start, timing_end));
    timing_start = timing_end;
    std::vector<float> source_content;
    std::vector<float> target_content;
    int64_t content_channels = 0;
    if (sources.route == SeedVcRouteRuntime::Route::V1XlsrHiftVoiceConversion) {
        const auto source_normalized = seed_vc_wav2vec2_normalize_16k(source_prepared.waveform_16k);
        const auto target_normalized = seed_vc_wav2vec2_normalize_16k(target_prepared.waveform_16k);
        const auto source_encoded = sources.wav2vec2_xlsr.encode(source_normalized, 1, static_cast<int64_t>(source_normalized.size()));
        const auto target_encoded = sources.wav2vec2_xlsr.encode(target_normalized, 1, static_cast<int64_t>(target_normalized.size()));
        source_content = source_encoded.hidden_states;
        target_content = target_encoded.hidden_states;
        content_channels = source_encoded.hidden_size;
    } else {
        source_content = sources.whisper_content.extract_16k_mono(
            source_prepared.waveform_16k,
            threads);
        target_content = sources.whisper_content.extract_16k_mono(
            target_prepared.waveform_16k,
            threads);
        content_channels = sources.whisper_content.channels();
    }
    timing_end = std::chrono::steady_clock::now();
    engine::debug::timing_log_scalar(
        "seed_vc.v1.content_ms",
        engine::debug::elapsed_ms(timing_start, timing_end));
    timing_start = timing_end;
    const int64_t source_content_tokens = static_cast<int64_t>(source_content.size()) / content_channels;
    const int64_t target_content_tokens = static_cast<int64_t>(target_content.size()) / content_channels;
    if (source_content_tokens <= 0 || target_content_tokens <= 0) {
        throw std::runtime_error("Seed-VC V1 content is empty");
    }

    const auto fbank = compute_seed_vc_campplus_fbank_16k(target_prepared.waveform_16k);
    if (fbank.frames <= 0 || fbank.dims <= 0) {
        throw std::runtime_error("Seed-VC V1 target style fbank is empty");
    }
    const auto target_style = sources.campplus.embed_from_features(
        fbank.features,
        fbank.frames,
        fbank.dims);
    timing_end = std::chrono::steady_clock::now();
    engine::debug::timing_log_scalar(
        "seed_vc.v1.style_ms",
        engine::debug::elapsed_ms(timing_start, timing_end));
    timing_start = timing_end;

    std::vector<float> source_f0;
    std::vector<float> target_f0;
    if (config.f0_condition) {
        target_f0 = sources.rmvpe_extractor.infer_16k_mono(
            target_prepared.waveform_16k,
            0.03F,
            threads);
        const auto raw_source_f0 = sources.rmvpe_extractor.infer_16k_mono(
            source_prepared.waveform_16k,
            0.03F,
            threads);
        source_f0 = adjust_source_f0_like_python(
            raw_source_f0,
            target_f0,
            config.auto_f0_adjust,
            config.semi_tone_shift);
    }
    timing_end = std::chrono::steady_clock::now();
    engine::debug::timing_log_scalar(
        "seed_vc.v1.f0_ms",
        engine::debug::elapsed_ms(timing_start, timing_end));
    timing_start = timing_end;

    SeedVcV1LengthRegulatorInput source_lr;
    source_lr.content = source_content;
    source_lr.f0 = source_f0;
    source_lr.output_lengths = {static_cast<int64_t>(static_cast<double>(source_mel.frames) * config.length_adjust)};
    source_lr.batch = 1;
    source_lr.tokens = source_content_tokens;
    source_lr.f0_tokens = static_cast<int64_t>(source_f0.size());
    source_lr.has_f0 = config.f0_condition;
    const auto source_condition = sources.v1_length_regulator.run(source_lr);

    SeedVcV1LengthRegulatorInput target_lr;
    target_lr.content = target_content;
    target_lr.f0 = target_f0;
    target_lr.output_lengths = {target_mel.frames};
    target_lr.batch = 1;
    target_lr.tokens = target_content_tokens;
    target_lr.f0_tokens = static_cast<int64_t>(target_f0.size());
    target_lr.has_f0 = config.f0_condition;
    const auto prompt_condition = sources.v1_length_regulator.run(target_lr);
    timing_end = std::chrono::steady_clock::now();
    engine::debug::timing_log_scalar(
        "seed_vc.v1.length_regulator_ms",
        engine::debug::elapsed_ms(timing_start, timing_end));
    timing_start = timing_end;

    const int64_t max_context_window =
        (mel_config.sample_rate / mel_config.hop_size) * kDitMaxContextSeconds;
    const int64_t max_source_window = max_context_window - target_mel.frames;
    if (max_source_window <= 0) {
        throw std::runtime_error("Seed-VC V1 target prompt is too long for the DiT context window");
    }

    const int64_t overlap_wave_len = kOverlapFrameLen * mel_config.hop_size;
    int64_t processed_frames = 0;
    uint64_t random_offset = 0;
    std::vector<float> generated_wave;
    std::vector<float> previous_chunk;
    const uint64_t seed = runtime::parse_u64_option(request.options, {"seed"})
        .value_or(runtime::random_u64_seed());
    const std::string noise_file =
        runtime::find_option(request.options, {"noise_file"}).value_or("");

    while (processed_frames < source_condition.tokens) {
        const int64_t chunk_frames = std::min(max_source_window, source_condition.tokens - processed_frames);
        const bool is_last_chunk = processed_frames + max_source_window >= source_condition.tokens;
        const auto cat_condition = concat_conditions(
            prompt_condition.values,
            target_mel.frames,
            source_condition.values,
            processed_frames,
            chunk_frames,
            source_condition.channels);
        const int64_t original_len = target_mel.frames + chunk_frames;
        auto prompt = make_full_prompt_mel(
            target_mel.mel,
            target_mel.channels,
            target_mel.frames,
            original_len);
        const size_t noise_count =
            static_cast<size_t>(dit_config.in_channels * original_len);
        auto initial_noise = load_seed_vc_noise_or_sample(
            noise_file,
            noise_count,
            seed,
            random_offset);
        random_offset += static_cast<uint64_t>(noise_count);

        SeedVcV1CfmInferenceInput cfm_input;
        cfm_input.mu = cat_condition;
        cfm_input.prompt = std::move(prompt);
        cfm_input.style = target_style.embedding;
        cfm_input.initial_noise = std::move(initial_noise);
        cfm_input.batch = 1;
        cfm_input.frames = original_len;
        cfm_input.prompt_frames = target_mel.frames;
        cfm_input.num_inference_steps = config.num_inference_steps;
        cfm_input.temperature = 1.0F;
        cfm_input.inference_cfg_rate = config.inference_cfg_rate;

        const auto cfm_output = sources.v1_cfm_estimator.infer(cfm_input);
        timing_end = std::chrono::steady_clock::now();
        engine::debug::timing_log_scalar(
            "seed_vc.v1.cfm_ms",
            engine::debug::elapsed_ms(timing_start, timing_end));
        timing_start = timing_end;
        const auto vc_mel = slice_mel_source_region(
            cfm_output.velocity,
            cfm_output.channels,
            cfm_output.frames,
            target_mel.frames,
            original_len);
        const bool use_hift = sources.route == SeedVcRouteRuntime::Route::V1XlsrHiftVoiceConversion;
        std::vector<float> hift_source_random;
        const std::vector<float> * hift_source_random_ptr = nullptr;
        uint64_t hift_source_advance_count = 0;
        if (use_hift) {
            const int64_t hift_scale = assets.config.hift.istft_hop *
                std::accumulate(
                    assets.config.hift.upsample_rates.begin(),
                    assets.config.hift.upsample_rates.end(),
                    int64_t{1},
                    std::multiplies<int64_t>{});
            const int64_t harmonics = assets.config.hift.nb_harmonics + 1;
            const size_t source_samples = static_cast<size_t>(chunk_frames * hift_scale);
            const size_t used_random_count =
                static_cast<size_t>(harmonics) + static_cast<size_t>(harmonics) * source_samples;
            const size_t stream_random_count = used_random_count + source_samples;
            hift_source_advance_count = static_cast<uint64_t>(stream_random_count);
            if (!noise_file.empty()) {
                auto hift_random_stream = load_seed_vc_noise_or_sample(
                    noise_file,
                    stream_random_count,
                    seed,
                    random_offset);
                hift_source_random.assign(
                    hift_random_stream.begin(),
                    hift_random_stream.begin() + static_cast<std::ptrdiff_t>(used_random_count));
                random_offset += hift_source_advance_count;
                hift_source_random_ptr = &hift_source_random;
            }
        }
        std::vector<float> vc_wave;
        if (use_hift) {
            vc_wave = sources.hift.synthesize(vc_mel, chunk_frames, seed, random_offset, hift_source_random_ptr).waveform;
        } else {
            const int64_t bigvgan_hop_size = sources.route == SeedVcRouteRuntime::Route::V1SingingVoiceConversion
                ? assets.config.bigvgan_44k.hop_size
                : assets.config.bigvgan_22k.hop_size;
            vc_wave = synthesize_bigvgan_fixed_chunks(
                sources.bigvgan,
                vc_mel,
                chunk_frames,
                bigvgan_hop_size,
                kSeedVcBigVganActiveFrames,
                kSeedVcBigVganOverlapFrames);
        }
        timing_end = std::chrono::steady_clock::now();
        engine::debug::timing_log_scalar(
            use_hift ? "seed_vc.v1.hift_ms" : "seed_vc.v1.bigvgan_ms",
            engine::debug::elapsed_ms(timing_start, timing_end));
        timing_start = timing_end;
        if (use_hift && noise_file.empty()) {
            random_offset += hift_source_advance_count;
        }

        if (processed_frames == 0) {
            if (is_last_chunk) {
                append_wave(generated_wave, vc_wave);
                break;
            }
            append_wave(generated_wave, without_tail(vc_wave, overlap_wave_len));
            previous_chunk = tail_of(vc_wave, overlap_wave_len);
        } else {
            if (is_last_chunk) {
                append_wave(generated_wave, crossfade_wave(previous_chunk, vc_wave, overlap_wave_len));
                break;
            }
            const auto faded = crossfade_wave(previous_chunk, vc_wave, overlap_wave_len);
            append_wave(generated_wave, without_tail(faded, overlap_wave_len));
            previous_chunk = tail_of(vc_wave, overlap_wave_len);
        }
        processed_frames += chunk_frames - kOverlapFrameLen;
        if (processed_frames <= 0) {
            throw std::runtime_error("Seed-VC V1 streaming chunking did not make progress");
        }
    }
    timing_end = std::chrono::steady_clock::now();
    engine::debug::timing_log_scalar(
        "seed_vc.v1.wave_assemble_ms",
        engine::debug::elapsed_ms(timing_start, timing_end));

    runtime::TaskResult result;
    const int output_sample_rate = sources.route == SeedVcRouteRuntime::Route::V1SingingVoiceConversion
        ? static_cast<int>(assets.config.bigvgan_44k.sampling_rate)
        : static_cast<int>(mel_config.sample_rate);
    result.audio_output = runtime::AudioBuffer{output_sample_rate, 1, std::move(generated_wave)};
    return result;
}

SeedVcV1RequestConfig parse_v1_config(const runtime::TaskRequest & request) {
    SeedVcV1RequestConfig config;
    config.num_inference_steps = runtime::parse_int_option(
        request.options,
        {"num_inference_steps"})
        .value_or(config.num_inference_steps);
    config.length_adjust = runtime::parse_finite_float_option(
        request.options,
        {"length_adjust"})
        .value_or(config.length_adjust);
    config.inference_cfg_rate = runtime::parse_finite_float_option(
        request.options,
        {"inference_cfg_rate"})
        .value_or(config.inference_cfg_rate);
    if (const auto value = runtime::find_option(request.options, {"f0_condition"})) {
        config.f0_condition = runtime::parse_bool_option(*value, "f0_condition");
    }
    if (const auto value = runtime::find_option(request.options, {"auto_f0_adjust"})) {
        config.auto_f0_adjust = runtime::parse_bool_option(*value, "auto_f0_adjust");
    }
    config.semi_tone_shift = runtime::parse_int_option(
        request.options,
        {"semi_tone_shift"})
        .value_or(config.semi_tone_shift);
    if (const auto value = runtime::find_option(request.options, {"fp16"})) {
        config.fp16 = runtime::parse_bool_option(*value, "fp16");
    }
    validate_common_generation_options(config.num_inference_steps, config.length_adjust);
    return config;
}

SeedVcExecutionPlan make_execution_plan(const runtime::TaskRequest & request, runtime::VoiceTaskKind task) {
    SeedVcExecutionPlan plan;
    plan.path = request_route_or_default(request, task);
    const auto & source = require_source_audio(request);
    const auto & target = require_target_audio(request);
    plan.source_sample_rate = source.sample_rate;
    plan.source_channels = source.channels;
    plan.target_sample_rate = target.sample_rate;
    plan.target_channels = target.channels;
    plan.source_frames = static_cast<int64_t>(source.samples.size() / static_cast<size_t>(source.channels));
    plan.target_frames = static_cast<int64_t>(target.samples.size() / static_cast<size_t>(target.channels));
    if (plan.path == "v2_vc") {
        plan.v2 = parse_v2_config(request);
    } else if (is_v1_path(plan.path)) {
        plan.v1 = parse_v1_config(request);
    }
    return plan;
}

std::shared_ptr<SeedVcRouteRuntime> open_route_runtime(
    runtime::VoiceTaskKind task,
    const engine::core::BackendConfig & backend,
    const SeedVcAssets & assets,
    const std::string & route_path,
    std::optional<engine::assets::TensorStorageType> weight_storage_type) {
    const auto default_weight_storage_type = weight_storage_type.value_or(engine::assets::TensorStorageType::Native);
    const auto rmvpe_weight_storage_type = weight_storage_type.value_or(engine::assets::TensorStorageType::F32);
    auto sources = std::make_shared<SeedVcRouteRuntime>();
    sources->campplus = engine::modules::CampplusEncoderComponent::load_from_tensor_source(
        assets.campplus_weights,
        backend,
        engine::modules::CampplusEncoderConfig{80, 192, default_weight_storage_type});
    if (route_path == "v2_vc") {
        if (task != runtime::VoiceTaskKind::VoiceConversion) {
            throw std::runtime_error("Seed-VC v2_vc sources require a VoiceConversion session");
        }
        sources->route = SeedVcRouteRuntime::Route::V2VoiceConversion;
        sources->v2_ar_length_regulator = SeedVcDiscreteLengthRegulator(
            assets.v2_ar_weights,
            backend,
            default_weight_storage_type,
            "length_regulator");
        sources->v2_cfm_length_regulator = SeedVcCfmLengthRegulator(
            assets.v2_cfm_weights,
            backend,
            default_weight_storage_type,
            "length_regulator");
        sources->v2_cfm_estimator = SeedVcV2CfmEstimator(
            assets.v2_cfm_weights,
            backend,
            default_weight_storage_type,
            assets.config.v2_cfm);
        sources->astral_bsq32_quantizer = SeedVcAstralQuantizer(
            assets.astral_bsq32_weights,
            backend,
            default_weight_storage_type,
            "",
            assets.config.v2_astral_narrow.encoder_input_dim,
            assets.config.v2_astral_narrow.encoder_dim,
            assets.config.v2_astral_narrow.encoder_intermediate_dim,
            assets.config.v2_astral_narrow.encoder_blocks,
            assets.config.v2_astral_narrow.quantizer_codebook_size);
        sources->astral_bsq2048_quantizer = SeedVcAstralQuantizer(
            assets.astral_bsq2048_weights,
            backend,
            default_weight_storage_type,
            "",
            assets.config.v2_astral_wide.encoder_input_dim,
            assets.config.v2_astral_wide.encoder_dim,
            assets.config.v2_astral_wide.encoder_intermediate_dim,
            assets.config.v2_astral_wide.encoder_blocks,
            assets.config.v2_astral_wide.quantizer_codebook_size);
        sources->bigvgan = engine::modules::BigVganVocoderComponent::load_from_tensor_source(
            assets.bigvgan_22k_weights,
            backend,
            make_bigvgan_config(assets.config.bigvgan_22k, default_weight_storage_type));
        engine::modules::HubertEncoderConfig hubert_config;
        hubert_config.output_hidden_layer = assets.config.v2_astral_wide.ssl_output_layer;
        hubert_config.apply_final_layer_norm = false;
        sources->hubert_large = engine::modules::HubertEncoderComponent::load_from_tensor_source(
            assets.hubert_large_weights,
            backend,
            hubert_config);
    } else if (route_path == "v1_svc") {
        if (task != runtime::VoiceTaskKind::Svc) {
            throw std::runtime_error("Seed-VC v1_svc sources require an Svc session");
        }
        sources->route = SeedVcRouteRuntime::Route::V1SingingVoiceConversion;
        sources->v1_length_regulator = SeedVcV1LengthRegulator(
            assets.v1_svc_weights,
            backend,
            default_weight_storage_type,
            "length_regulator");
        sources->v1_cfm_estimator = SeedVcV1CfmEstimator(
            assets.v1_svc_weights,
            backend,
            default_weight_storage_type,
            assets.config.v1_dit,
            assets.config.v1_wavenet,
            assets.config.v1_style_dim);
        sources->rmvpe_extractor = SeedVcRmvpeF0Extractor(
            assets.rmvpe_weights,
            backend,
            rmvpe_weight_storage_type);
        sources->whisper_content = SeedVcWhisperContentEncoder(
            assets.whisper_small_weights,
            backend,
            default_weight_storage_type);
        sources->bigvgan = engine::modules::BigVganVocoderComponent::load_from_tensor_source(
            assets.bigvgan_44k_weights,
            backend,
            make_bigvgan_config(assets.config.bigvgan_44k, default_weight_storage_type));
    } else if (route_path == "v1_whisper_bigvgan_vc") {
        if (task != runtime::VoiceTaskKind::VoiceConversion) {
            throw std::runtime_error("Seed-VC v1_whisper_bigvgan_vc sources require a VoiceConversion session");
        }
        sources->route = SeedVcRouteRuntime::Route::V1WhisperBigVganVoiceConversion;
        sources->v1_length_regulator = SeedVcV1LengthRegulator(
            assets.v1_whisper_bigvgan_weights,
            backend,
            default_weight_storage_type,
            "length_regulator");
        sources->v1_cfm_estimator = SeedVcV1CfmEstimator(
            assets.v1_whisper_bigvgan_weights,
            backend,
            default_weight_storage_type,
            assets.config.v1_whisper_bigvgan_dit,
            assets.config.v1_whisper_bigvgan_wavenet,
            assets.config.v1_whisper_bigvgan_style_dim);
        sources->whisper_content = SeedVcWhisperContentEncoder(
            assets.whisper_small_weights,
            backend,
            default_weight_storage_type);
        sources->bigvgan = engine::modules::BigVganVocoderComponent::load_from_tensor_source(
            assets.bigvgan_22k_weights,
            backend,
            make_bigvgan_config(assets.config.bigvgan_22k, default_weight_storage_type));
    } else if (route_path == "v1_xlsr_hift_vc") {
        if (task != runtime::VoiceTaskKind::VoiceConversion) {
            throw std::runtime_error("Seed-VC v1_xlsr_hift_vc sources require a VoiceConversion session");
        }
        sources->route = SeedVcRouteRuntime::Route::V1XlsrHiftVoiceConversion;
        sources->v1_length_regulator = SeedVcV1LengthRegulator(
            assets.v1_xlsr_hift_weights,
            backend,
            default_weight_storage_type,
            "length_regulator");
        sources->v1_cfm_estimator = SeedVcV1CfmEstimator(
            assets.v1_xlsr_hift_weights,
            backend,
            default_weight_storage_type,
            assets.config.v1_xlsr_hift_dit,
            assets.config.v1_xlsr_hift_wavenet,
            assets.config.v1_xlsr_hift_style_dim);
        engine::modules::HubertEncoderConfig xlsr_config;
        xlsr_config.output_hidden_layer = 12;
        xlsr_config.apply_final_layer_norm = true;
        sources->wav2vec2_xlsr = engine::modules::HubertEncoderComponent::load_from_tensor_source(
            assets.wav2vec2_xlsr_weights,
            backend,
            xlsr_config);
        sources->hift = engine::modules::HiftVocoderComponent::load_from_tensor_source(
            assets.hift_weights,
            backend,
            make_hift_config(assets.config.hift, default_weight_storage_type));
    } else {
        throw std::runtime_error("unsupported Seed-VC source route: " + route_path);
    }
    return sources;
}

std::string request_route_or_default(const runtime::TaskRequest & request, runtime::VoiceTaskKind task) {
    return route_path_or_default_from_options(request.options, task);
}

std::string route_path_or_default_from_options(
    const std::unordered_map<std::string, std::string> & options,
    runtime::VoiceTaskKind task) {
    const auto it = options.find("route");
    if (it != options.end() && !it->second.empty()) {
        return it->second;
    }
    return task == runtime::VoiceTaskKind::Svc ? "v1_svc" : "v2_vc";
}

void validate_request_route(const runtime::TaskRequest & request, runtime::VoiceTaskKind task) {
    const auto route = request_route_or_default(request, task);
    if (route == "v2_vc") {
        if (task != runtime::VoiceTaskKind::VoiceConversion) {
            throw std::runtime_error("Seed-VC v2_vc request requires a VoiceConversion session");
        }
        return;
    }
    if (route == "v1_svc") {
        if (task != runtime::VoiceTaskKind::Svc) {
            throw std::runtime_error("Seed-VC v1_svc request requires an Svc session");
        }
        return;
    }
    if (is_v1_voice_conversion_path(route)) {
        if (task != runtime::VoiceTaskKind::VoiceConversion) {
            throw std::runtime_error("Seed-VC " + route + " request requires a VoiceConversion session");
        }
        return;
    }
    throw std::runtime_error("unsupported Seed-VC route: " + route);
}

std::string route_path_for_runtime(SeedVcRouteRuntime::Route route) {
    switch (route) {
        case SeedVcRouteRuntime::Route::V2VoiceConversion:
            return "v2_vc";
        case SeedVcRouteRuntime::Route::V1SingingVoiceConversion:
            return "v1_svc";
        case SeedVcRouteRuntime::Route::V1WhisperBigVganVoiceConversion:
            return "v1_whisper_bigvgan_vc";
        case SeedVcRouteRuntime::Route::V1XlsrHiftVoiceConversion:
            return "v1_xlsr_hift_vc";
    }
    throw std::runtime_error("Seed-VC prepared route is unknown");
}

SeedVcSession::SeedVcSession(
    runtime::TaskSpec task,
    runtime::SessionOptions options,
    std::shared_ptr<const SeedVcAssets> assets)
    : RuntimeSessionBase(options),
      task_(task),
      assets_(require_assets(std::move(assets))),
      weight_storage_type_(parse_seed_vc_weight_type(this->options())) {}

std::string SeedVcSession::family() const {
    return "seed_vc";
}

runtime::VoiceTaskKind SeedVcSession::task_kind() const {
    return task_.task;
}

runtime::RunMode SeedVcSession::run_mode() const {
    return task_.mode;
}

void SeedVcSession::prepare(const runtime::SessionPreparationRequest & request) {
    const std::string route_path = route_path_or_default_from_options(request.options, task_.task);
    if (route_runtime_ != nullptr) {
        const std::string prepared_route = route_path_for_runtime(route_runtime_->route);
        if (prepared_route != route_path) {
            throw std::runtime_error(
                "Seed-VC session is already prepared for route " + prepared_route +
                " and cannot prepare route " + route_path);
        }
        mark_prepared();
        return;
    }
    route_runtime_ = open_route_runtime(
        task_.task,
        options().backend,
        *assets_,
        route_path,
        weight_storage_type_);
    int64_t bigvgan_hop_size = 0;
    switch (route_runtime_->route) {
        case SeedVcRouteRuntime::Route::V2VoiceConversion:
        case SeedVcRouteRuntime::Route::V1WhisperBigVganVoiceConversion:
            bigvgan_hop_size = assets_->config.bigvgan_22k.hop_size;
            break;
        case SeedVcRouteRuntime::Route::V1SingingVoiceConversion:
            bigvgan_hop_size = assets_->config.bigvgan_44k.hop_size;
            break;
        case SeedVcRouteRuntime::Route::V1XlsrHiftVoiceConversion:
            break;
    }
    if (bigvgan_hop_size > 0) {
        const int64_t channels = route_runtime_->bigvgan.num_mels();
        std::vector<float> mel(static_cast<size_t>(channels * kSeedVcBigVganActiveFrames), 0.0F);
        (void) synthesize_bigvgan_fixed_chunks(
            route_runtime_->bigvgan,
            mel,
            kSeedVcBigVganActiveFrames,
            bigvgan_hop_size,
            kSeedVcBigVganActiveFrames,
            kSeedVcBigVganOverlapFrames);
    }
    mark_prepared();
}

runtime::TaskResult SeedVcSession::run(const runtime::TaskRequest & request) {
    require_prepared("Seed-VC run");
    const auto wall_start = std::chrono::steady_clock::now();
    validate_request_route(request, task_.task);
    const auto plan = make_execution_plan(request, task_.task);
    if (route_runtime_ == nullptr) {
        throw std::runtime_error("Seed-VC session has no prepared route");
    }
    const std::string prepared_route = route_path_for_runtime(route_runtime_->route);
    if (prepared_route != plan.path) {
        throw std::runtime_error(
            "Seed-VC session was prepared for route " + prepared_route +
            " but request route is " + plan.path);
    }
    if (plan.path == "v2_vc") {
        auto result = run_v2_voice_conversion(request, plan, *route_runtime_, *assets_, static_cast<size_t>(options().backend.threads));
        engine::debug::timing_log_scalar("session.wall_ms", engine::debug::elapsed_ms(wall_start));
        return result;
    }
    if (is_v1_path(plan.path)) {
        auto result = run_v1_singing_voice_conversion(request, plan, *route_runtime_, *assets_, static_cast<size_t>(options().backend.threads));
        engine::debug::timing_log_scalar("session.wall_ms", engine::debug::elapsed_ms(wall_start));
        return result;
    }
    throw std::runtime_error("Seed-VC " + plan.path + " graph execution is not implemented yet");
}

}  // namespace engine::models::seed_vc
