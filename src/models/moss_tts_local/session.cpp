#include "engine/models/moss_tts_local/session.h"

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/audio/resampling.h"
#include "engine/framework/core/backend.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/runtime/options.h"
#include "engine/models/moss_tts_local/assets.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace engine::models::moss_tts_local {
namespace {

constexpr size_t kBackboneWeightContextBytes = 64ull * 1024 * 1024;
constexpr size_t kBackboneGraphArenaBytes = 1024ull * 1024 * 1024;
constexpr size_t kDepthWeightContextBytes = 16ull * 1024 * 1024;
constexpr size_t kDepthGraphArenaBytes = 64ull * 1024 * 1024;
constexpr size_t kGeneratorProjectionWeightContextBytes = 16ull * 1024 * 1024;
constexpr size_t kGeneratorProjectionGraphArenaBytes = 16ull * 1024 * 1024;
constexpr size_t kCodecWeightContextBytes = 256ull * 1024 * 1024;
constexpr size_t kCodecGraphArenaBytes = 1536ull * 1024 * 1024;
constexpr size_t kEncoderWeightContextBytes = 256ull * 1024 * 1024;
constexpr size_t kEncoderGraphArenaBytes = 2048ull * 1024 * 1024;
constexpr int kCodecSampleRate = 48000;

// Prepares a speaker reference for the codec encoder, mirroring the processor's
// encode_audios_from_wav front end: de-interleave, force stereo (mono duplicated),
// resample to 48 kHz with the torchaudio sinc-Hann kernel, then loudness-normalize the
// whole clip to -20 dBFS (power) with a +/-3 dB gain clamp.
std::vector<std::vector<float>> reference_to_codec_stereo(const runtime::AudioBuffer & audio) {
    const int channels = std::max(1, audio.channels);
    const int64_t frames = channels > 0 ? static_cast<int64_t>(audio.samples.size()) / channels : 0;
    if (frames <= 0) {
        throw std::runtime_error("MOSS-TTS-Local voice reference audio is empty");
    }

    // De-interleave into per-channel streams, then force exactly two channels
    // (mono is duplicated; extra channels are dropped) before resampling.
    std::vector<std::vector<float>> stereo(2, std::vector<float>(static_cast<size_t>(frames)));
    for (int64_t t = 0; t < frames; ++t) {
        const float left = audio.samples[static_cast<size_t>(t * channels)];
        const float right = channels > 1 ? audio.samples[static_cast<size_t>(t * channels + 1)] : left;
        stereo[0][static_cast<size_t>(t)] = left;
        stereo[1][static_cast<size_t>(t)] = right;
    }

    if (audio.sample_rate != kCodecSampleRate) {
        if (audio.sample_rate <= 0) {
            throw std::runtime_error("MOSS-TTS-Local voice reference has an invalid sample rate");
        }
        for (auto & channel : stereo) {
            channel = engine::audio::resample_mono_torchaudio_sinc_hann(
                channel, audio.sample_rate, kCodecSampleRate);
        }
    }

    double sum_sq = 0.0;
    size_t count = 0;
    for (const auto & channel : stereo) {
        for (const float value : channel) {
            sum_sq += static_cast<double>(value) * value;
        }
        count += channel.size();
    }
    if (count > 0) {
        const double mean_sq = sum_sq / static_cast<double>(count);
        const double current_dbfs = 10.0 * std::log10(mean_sq + 1.0e-9);
        const double gain = std::max(-3.0, std::min(-20.0 - current_dbfs, 3.0));
        const float factor = static_cast<float>(std::pow(10.0, gain / 20.0));
        for (auto & channel : stereo) {
            for (float & value : channel) {
                value *= factor;
            }
        }
    }
    return stereo;
}

// Hardware-adaptive backbone dtype for the "auto" mode: GPUs run the model's
// native bf16, while CPU uses f32 (bf16/f16 matmul is poorly accelerated on CPU,
// so f32 is both faster and more accurate there). Non-CUDA GPU backends keep the
// stored (native) dtype to stay on the known-good path.
engine::assets::TensorStorageType resolve_auto_weight_type(engine::core::BackendType backend) {
    switch (backend) {
        case engine::core::BackendType::Cpu:
            return engine::assets::TensorStorageType::F32;
        case engine::core::BackendType::Cuda:
            return engine::assets::TensorStorageType::BF16;
        default:  // Metal, Vulkan, BestAvailable
            return engine::assets::TensorStorageType::Native;
    }
}

// Resolves moss_tts_local.weight_type. Defaults to "auto" (hardware-adaptive per
// resolve_auto_weight_type); an explicit value (native/f32/f16/bf16/q8_0/...)
// overrides. Handled here because the generic parser maps "auto" -> Native.
engine::assets::TensorStorageType option_weight_type(
    const runtime::SessionOptions & options,
    const char * key) {
    std::string value = "auto";
    const auto it = options.options.find(key);
    if (it != options.options.end() && !it->second.empty()) {
        value = it->second;
    }
    std::string lowered = value;
    std::transform(lowered.begin(), lowered.end(), lowered.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    if (lowered == "auto") {
        return resolve_auto_weight_type(options.backend.type);
    }
    return engine::assets::parse_tensor_storage_type(value);
}

std::shared_ptr<const MossTTSLocalAssets> require_assets(std::shared_ptr<const MossTTSLocalAssets> assets) {
    if (assets == nullptr) {
        throw std::runtime_error("MOSS-TTS-Local session requires assets");
    }
    return assets;
}

MossGenerationOptions generation_options_from_request(const runtime::TaskRequest & request) {
    MossGenerationOptions options;
    if (const auto value = runtime::parse_i64_option(request.options, {"max_frames", "max_tokens"})) {
        if (*value <= 0) {
            throw std::runtime_error("MOSS-TTS-Local max_tokens must be positive");
        }
        options.max_new_frames = *value;
    }
    if (const auto value = runtime::find_option(request.options, {"do_sample"})) {
        options.do_sample = runtime::parse_bool_option(*value, "do_sample");
    }
    if (const auto value = runtime::parse_float_option(request.options, {"audio_temperature", "temperature"})) {
        options.audio_temperature = *value;
    }
    if (const auto value = runtime::parse_int_option(request.options, {"audio_top_k", "top_k"})) {
        options.audio_top_k = *value;
    }
    if (const auto value = runtime::parse_float_option(request.options, {"audio_top_p", "top_p"})) {
        options.audio_top_p = *value;
    }
    if (const auto value =
            runtime::parse_float_option(request.options, {"audio_repetition_penalty", "repetition_penalty"})) {
        options.audio_repetition_penalty = *value;
    }
    options.seed = runtime::parse_u32_option(request.options, {"seed"}).value_or(runtime::random_u32_seed());
    return options;
}

}  // namespace

MossTTSLocalSession::MossTTSLocalSession(
    runtime::TaskSpec task,
    runtime::SessionOptions options,
    std::shared_ptr<const MossTTSLocalAssets> assets)
    : RuntimeSessionBase(options),
      task_(task),
      assets_(require_assets(std::move(assets))) {
    backbone_ = std::make_unique<MossBackboneRuntime>(
        assets_,
        execution_context(),
        kBackboneGraphArenaBytes,
        kBackboneWeightContextBytes,
        option_weight_type(options, "moss_tts_local.weight_type"));
    depth_ = std::make_unique<MossDepthTransformer>(
        assets_, execution_context(), kDepthGraphArenaBytes, kDepthWeightContextBytes);
    processor_ = std::make_unique<MossTextProcessor>(assets_);
    codec_ = std::make_unique<MossCodecDecoder>(
        resolve_moss_codec_dir(*assets_),
        execution_context(),
        assets_->config.num_codebooks,
        kCodecWeightContextBytes,
        kCodecGraphArenaBytes);
    generator_ = std::make_unique<MossGenerator>(
        assets_,
        execution_context(),
        kGeneratorProjectionGraphArenaBytes,
        kGeneratorProjectionWeightContextBytes,
        *backbone_,
        *depth_);
}

std::string MossTTSLocalSession::family() const {
    return "moss_tts_local";
}

runtime::VoiceTaskKind MossTTSLocalSession::task_kind() const {
    return task_.task;
}

runtime::RunMode MossTTSLocalSession::run_mode() const {
    return task_.mode;
}

void MossTTSLocalSession::prepare(const runtime::SessionPreparationRequest & request) {
    const bool has_reference = request.voice.has_value() && request.voice->speaker.has_value() &&
        request.voice->speaker->audio.has_value();
    if (has_reference) {
        (void) encoder();
    }
    mark_prepared();
}

MossCodecEncoder & MossTTSLocalSession::encoder() {
    if (encoder_ == nullptr) {
        encoder_ = std::make_unique<MossCodecEncoder>(
            resolve_moss_codec_dir(*assets_),
            execution_context(),
            assets_->config.num_codebooks,
            kEncoderWeightContextBytes,
            kEncoderGraphArenaBytes);
    }
    return *encoder_;
}

runtime::TaskResult MossTTSLocalSession::run(const runtime::TaskRequest & request) {
    const auto wall_start = std::chrono::steady_clock::now();
    if (!request.text_input.has_value() || request.text_input->text.empty()) {
        throw std::runtime_error("MOSS-TTS-Local requires text input");
    }
    const bool has_reference = request.voice.has_value() && request.voice->speaker.has_value() &&
        request.voice->speaker->audio.has_value();

    std::optional<std::string> language;
    const std::string & language_tag = request.text_input->language;
    if (!language_tag.empty() && language_tag != "Auto") {
        language = language_tag;
    }

    const auto options = generation_options_from_request(request);
    const bool collect_timing = engine::debug::timing_log_enabled();
    const auto time_once = [&](double & target, auto && fn) {
        if (collect_timing) {
            target = engine::debug::measure_ms(fn);
        } else {
            fn();
        }
    };
    MossGenerationPrefix prefix;
    double reference_ms = 0.0;
    double reference_encode_ms = 0.0;
    double prefix_ms = 0.0;
    double generate_ms = 0.0;
    double code_pack_ms = 0.0;
    double codec_decode_ms = 0.0;
    double interleave_ms = 0.0;
    if (has_reference) {
        std::vector<std::vector<float>> stereo;
        time_once(reference_ms, [&]() {
            stereo = reference_to_codec_stereo(*request.voice->speaker->audio);
        });
        std::vector<std::vector<int32_t>> reference_codes;
        time_once(reference_encode_ms, [&]() {
            reference_codes = encoder().encode(stereo);
        });
        time_once(prefix_ms, [&]() {
            prefix = processor_->build_clone_prefix(request.text_input->text, reference_codes, language);
        });
    } else {
        time_once(prefix_ms, [&]() {
            prefix = processor_->build_generation_prefix(request.text_input->text, language);
        });
    }
    std::vector<std::vector<int32_t>> frames;
    time_once(generate_ms, [&]() {
        frames = generator_->generate(prefix.text_tokens, prefix.audio_codes, options);
    });
    if (frames.empty()) {
        throw std::runtime_error("MOSS-TTS-Local generated no audio frames");
    }

    const int64_t num_codebooks = assets_->config.num_codebooks;
    std::vector<std::vector<int32_t>> codes;
    time_once(code_pack_ms, [&]() {
        codes.assign(
            static_cast<size_t>(num_codebooks),
            std::vector<int32_t>(frames.size()));
        for (size_t frame = 0; frame < frames.size(); ++frame) {
            for (int64_t codebook = 0; codebook < num_codebooks; ++codebook) {
                codes[static_cast<size_t>(codebook)][frame] =
                    frames[frame][static_cast<size_t>(codebook)];
            }
        }
    });

    std::vector<std::vector<float>> channels;
    time_once(codec_decode_ms, [&]() {
        channels = codec_->decode(codes);
    });
    const int channel_count = static_cast<int>(channels.size());
    const size_t samples_per_channel = channels.empty() ? 0 : channels.front().size();

    runtime::AudioBuffer audio;
    audio.sample_rate = static_cast<int>(codec_->sampling_rate());
    audio.channels = channel_count;
    time_once(interleave_ms, [&]() {
        audio.samples.resize(samples_per_channel * static_cast<size_t>(channel_count));
        for (size_t sample = 0; sample < samples_per_channel; ++sample) {
            for (int channel = 0; channel < channel_count; ++channel) {
                audio.samples[sample * static_cast<size_t>(channel_count) + static_cast<size_t>(channel)] =
                    channels[static_cast<size_t>(channel)][sample];
            }
        }
    });

    engine::debug::timing_log_scalar("moss_tts_local.reference_ms", reference_ms);
    engine::debug::timing_log_scalar("moss_tts_local.reference_encode_ms", reference_encode_ms);
    engine::debug::timing_log_scalar("moss_tts_local.prefix_ms", prefix_ms);
    engine::debug::timing_log_scalar("moss_tts_local.generate_ms", generate_ms);
    engine::debug::timing_log_scalar("moss_tts_local.code_pack_ms", code_pack_ms);
    engine::debug::timing_log_scalar("moss_tts_local.codec_decode_ms", codec_decode_ms);
    engine::debug::timing_log_scalar("moss_tts_local.interleave_ms", interleave_ms);
    engine::debug::trace_log_scalar("moss_tts_local.generated_frames", static_cast<int64_t>(frames.size()));
    engine::debug::timing_log_scalar("session.wall_ms", engine::debug::elapsed_ms(wall_start));

    runtime::TaskResult result;
    result.audio_output = std::move(audio);
    return result;
}

}  // namespace engine::models::moss_tts_local
