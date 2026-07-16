#include "engine/models/moss/moss_tts_local/session.h"

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/audio/resampling.h"
#include "engine/framework/core/backend.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/runtime/options.h"
#include "engine/framework/text/chunking.h"
#include "engine/models/moss/moss_tts_local/assets.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstring>
#include <limits>
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
constexpr int64_t kDefaultTextChunkSize = 2048;

uint64_t mix_reference_audio_key(uint64_t key, uint64_t value) {
    key ^= value;
    key *= 1099511628211ull;
    return key;
}

uint64_t prefix_hash(const moss::TokenRows & prefix, int64_t num_codebooks) {
    uint64_t key = 1469598103934665603ull;
    for (size_t row = 0; row < prefix.text_tokens.size(); ++row) {
        key = mix_reference_audio_key(key, static_cast<uint32_t>(prefix.text_tokens[row]));
        const size_t audio_offset = row * static_cast<size_t>(num_codebooks);
        for (int64_t codebook = 0; codebook < num_codebooks; ++codebook) {
            key = mix_reference_audio_key(
                key,
                static_cast<uint32_t>(prefix.audio_codes[audio_offset + static_cast<size_t>(codebook)]));
        }
    }
    return key;
}

int64_t prefix_audio_nonpad_count(const moss::TokenRows & prefix, int32_t audio_pad_token_id) {
    int64_t count = 0;
    for (const int32_t code : prefix.audio_codes) {
        if (code != audio_pad_token_id) {
            ++count;
        }
    }
    return count;
}

uint64_t reference_audio_cache_hash(const runtime::AudioBuffer & audio) {
    uint64_t key = 1469598103934665603ull;
    key = mix_reference_audio_key(key, static_cast<uint64_t>(audio.sample_rate));
    key = mix_reference_audio_key(key, static_cast<uint64_t>(audio.channels));
    key = mix_reference_audio_key(key, static_cast<uint64_t>(audio.samples.size()));
    for (const float sample : audio.samples) {
        uint32_t bits = 0;
        std::memcpy(&bits, &sample, sizeof(bits));
        key = mix_reference_audio_key(key, static_cast<uint64_t>(bits));
    }
    return key;
}

std::size_t resolve_reference_cache_slots(const runtime::SessionOptions & options) {
    constexpr int64_t kDefaultReferenceCacheSlots = 1;
    const int64_t slots = runtime::parse_i64_option(
        options.options,
        {"moss_tts_local.reference_cache_slots", "reference_cache_slots"})
        .value_or(kDefaultReferenceCacheSlots);
    if (slots < 0) {
        throw std::runtime_error("moss_tts_local.reference_cache_slots must be non-negative");
    }
    if (static_cast<std::uint64_t>(slots) > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
        throw std::runtime_error("moss_tts_local.reference_cache_slots is too large");
    }
    return static_cast<std::size_t>(slots);
}

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
#ifdef _OPENMP
#pragma omp parallel for if(frames >= 4096)
#endif
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
    if (const auto value = runtime::parse_i64_option(request.options, {"max_tokens"})) {
        if (*value <= 0) {
            throw std::runtime_error("MOSS-TTS-Local max_tokens must be positive");
        }
        options.max_new_frames = *value;
    }
    if (const auto value = runtime::find_option(request.options, {"do_sample"})) {
        options.do_sample = runtime::parse_bool_option(*value, "do_sample");
    }
    if (const auto value = runtime::parse_float_option(request.options, {"temperature"})) {
        options.audio_temperature = *value;
    }
    if (const auto value = runtime::parse_float_option(request.options, {"text_temperature"})) {
        options.text_temperature = *value;
    }
    if (const auto value = runtime::parse_int_option(request.options, {"text_top_k"})) {
        options.text_top_k = *value;
    }
    if (const auto value = runtime::parse_float_option(request.options, {"text_top_p"})) {
        options.text_top_p = *value;
    }
    if (const auto value = runtime::parse_int_option(request.options, {"top_k"})) {
        options.audio_top_k = *value;
    }
    if (const auto value = runtime::parse_float_option(request.options, {"top_p"})) {
        options.audio_top_p = *value;
    }
    if (const auto value = runtime::parse_float_option(request.options, {"repetition_penalty"})) {
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
      assets_(require_assets(std::move(assets))),
      reference_voice_cache_(resolve_reference_cache_slots(this->options())) {
    backbone_ = std::make_unique<MossBackboneRuntime>(
        assets_,
        execution_context(),
        kBackboneGraphArenaBytes,
        kBackboneWeightContextBytes,
        option_weight_type(options, "moss_tts_local.weight_type"));
    depth_ = std::make_unique<MossDepthTransformer>(
        assets_, execution_context(), kDepthGraphArenaBytes, kDepthWeightContextBytes);
    processor_ = std::make_unique<MossTextProcessor>(assets_);
    codec_ = std::make_unique<moss::MossAudioTokenizerDecoder>(
        *assets_->audio_tokenizer_weights,
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
    assets_->model_weights->release_storage();
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

moss::MossAudioTokenizerEncoder & MossTTSLocalSession::encoder() {
    if (encoder_ == nullptr) {
        reference_encoder_execution_context_ = std::make_unique<core::ExecutionContext>(options().backend);
        encoder_ = std::make_unique<moss::MossAudioTokenizerEncoder>(
            *assets_->audio_tokenizer_weights,
            *reference_encoder_execution_context_,
            assets_->config.num_codebooks,
            kEncoderWeightContextBytes,
            kEncoderGraphArenaBytes);
    }
    return *encoder_;
}

bool MossTTSLocalSession::ReferenceAudioCacheKeyEqual::operator()(
    const ReferenceAudioCacheKey & lhs,
    const ReferenceAudioCacheKey & rhs) const {
    return lhs.hash == rhs.hash &&
        lhs.sample_rate == rhs.sample_rate &&
        lhs.channels == rhs.channels &&
        lhs.sample_count == rhs.sample_count;
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
            target += engine::debug::measure_ms(fn);
        } else {
            fn();
        }
    };
    double reference_ms = 0.0;
    double reference_encode_ms = 0.0;
    double prefix_ms = 0.0;
    double generate_ms = 0.0;
    double backbone_step_release_ms = 0.0;
    int64_t backbone_step_released_steps = 0;
    double code_pack_ms = 0.0;
    double codec_decode_ms = 0.0;
    double interleave_ms = 0.0;
    std::vector<std::vector<int32_t>> reference_codes;
    if (has_reference) {
        const auto & reference_audio = *request.voice->speaker->audio;
        const ReferenceAudioCacheKey reference_key{
            reference_audio_cache_hash(reference_audio),
            reference_audio.sample_rate,
            reference_audio.channels,
            reference_audio.samples.size(),
        };
        if (const auto * cached = reference_voice_cache_.find(reference_key)) {
            reference_codes = cached->codes;
            engine::debug::trace_log_scalar("moss_tts_local.reference_cache.hit", 1);
            engine::debug::trace_log_scalar(
                "moss_tts_local.reference_cache.slots",
                static_cast<int64_t>(reference_voice_cache_.capacity()));
            engine::debug::trace_log_scalar(
                "moss_tts_local.reference_cache.entries",
                static_cast<int64_t>(reference_voice_cache_.size()));
            engine::debug::trace_log_scalar("moss_tts_local.reference_cache.evicted", 0);
        } else {
            const bool will_evict = reference_voice_cache_.capacity() > 0 &&
                reference_voice_cache_.size() >= reference_voice_cache_.capacity();
            std::vector<std::vector<float>> stereo;
            time_once(reference_ms, [&]() {
                stereo = reference_to_codec_stereo(reference_audio);
            });
            time_once(reference_encode_ms, [&]() {
                reference_codes = encoder().encode(stereo);
            });
            ReferenceVoiceCacheEntry entry;
            entry.codes = reference_codes;
            reference_voice_cache_.put(reference_key, std::move(entry));
            engine::debug::trace_log_scalar("moss_tts_local.reference_cache.hit", 0);
            engine::debug::trace_log_scalar(
                "moss_tts_local.reference_cache.slots",
                static_cast<int64_t>(reference_voice_cache_.capacity()));
            engine::debug::trace_log_scalar(
                "moss_tts_local.reference_cache.entries",
                static_cast<int64_t>(reference_voice_cache_.size()));
            engine::debug::trace_log_scalar("moss_tts_local.reference_cache.evicted", will_evict ? 1 : 0);
        }
    }

    const int64_t text_chunk_size =
        engine::text::parse_text_chunk_size_override(request.options).value_or(kDefaultTextChunkSize);
    const auto text_chunk_mode =
        engine::text::parse_text_chunk_mode_override(request.options).value_or(engine::text::TextChunkMode::Default);
    const auto text_chunks = engine::text::split_text_chunks(request.text_input->text, text_chunk_size, text_chunk_mode);
    if (text_chunks.empty()) {
        throw std::runtime_error("MOSS-TTS-Local text chunking produced no chunks");
    }
    engine::debug::trace_log_scalar("moss_tts_local.text_chunk_size", text_chunk_size);
    engine::debug::trace_log_scalar("moss_tts_local.text_chunk_mode", engine::text::text_chunk_mode_name(text_chunk_mode));
    engine::debug::trace_log_scalar("moss_tts_local.text_chunk_count", static_cast<int64_t>(text_chunks.size()));

    runtime::AudioBuffer merged_audio;
    int64_t last_prefix_len = 0;
    int64_t last_prefix_hash = 0;
    int64_t last_prefix_audio_nonpad = 0;
    int64_t total_generated_frames = 0;
    const int64_t num_codebooks = assets_->config.num_codebooks;
    for (const auto & text_chunk : text_chunks) {
        MossGenerationPrefix prefix;
        time_once(prefix_ms, [&]() {
            if (has_reference) {
                prefix = processor_->build_clone_prefix(text_chunk, reference_codes, language);
            } else {
                prefix = processor_->build_generation_prefix(text_chunk, language);
            }
        });
        last_prefix_len = static_cast<int64_t>(prefix.text_tokens.size());
        last_prefix_hash = static_cast<int64_t>(
            prefix_hash(prefix, assets_->config.num_codebooks) & 0x7fffffffffffffffull);
        last_prefix_audio_nonpad =
            prefix_audio_nonpad_count(prefix, static_cast<int32_t>(assets_->config.audio_pad_token_id));

        std::vector<std::vector<int32_t>> frames;
        time_once(generate_ms, [&]() {
            frames = generator_->generate(prefix.text_tokens, prefix.audio_codes, options);
        });
        if (frames.empty()) {
            throw std::runtime_error("MOSS-TTS-Local generated no audio frames");
        }
        time_once(backbone_step_release_ms, [&]() {
            backbone_step_released_steps += backbone_->release_cached_step_graph();
        });

        std::vector<std::vector<int32_t>> codes;
        time_once(code_pack_ms, [&]() {
            codes.assign(
                static_cast<size_t>(num_codebooks),
                std::vector<int32_t>(frames.size()));
            const int64_t generated_frames = static_cast<int64_t>(frames.size());
#ifdef _OPENMP
#pragma omp parallel for collapse(2) if(generated_frames * num_codebooks >= 4096)
#endif
            for (int64_t frame = 0; frame < generated_frames; ++frame) {
                for (int64_t codebook = 0; codebook < num_codebooks; ++codebook) {
                    codes[static_cast<size_t>(codebook)][static_cast<size_t>(frame)] =
                        frames[static_cast<size_t>(frame)][static_cast<size_t>(codebook)];
                }
            }
        });

        std::vector<std::vector<float>> channels;
        time_once(codec_decode_ms, [&]() {
            channels = codec_->decode(codes);
        });
        const int channel_count = static_cast<int>(channels.size());
        const size_t samples_per_channel = channels.empty() ? 0 : channels.front().size();

        runtime::AudioBuffer chunk_audio;
        chunk_audio.sample_rate = static_cast<int>(codec_->sampling_rate());
        chunk_audio.channels = channel_count;
        time_once(interleave_ms, [&]() {
            chunk_audio.samples.resize(samples_per_channel * static_cast<size_t>(channel_count));
            const int64_t sample_count = static_cast<int64_t>(samples_per_channel);
#ifdef _OPENMP
#pragma omp parallel for if(sample_count * channel_count >= 4096)
#endif
            for (int64_t sample = 0; sample < sample_count; ++sample) {
                for (int channel = 0; channel < channel_count; ++channel) {
                    chunk_audio.samples[
                        static_cast<size_t>(sample) * static_cast<size_t>(channel_count) +
                        static_cast<size_t>(channel)] =
                        channels[static_cast<size_t>(channel)][static_cast<size_t>(sample)];
                }
            }
        });
        total_generated_frames += static_cast<int64_t>(frames.size());
        runtime::append_audio_buffer(merged_audio, chunk_audio);
    }

    engine::debug::trace_log_scalar("moss_tts_local.prefix_len", last_prefix_len);
    engine::debug::trace_log_scalar("moss_tts_local.prefix_hash", last_prefix_hash);
    engine::debug::trace_log_scalar("moss_tts_local.prefix_audio_nonpad", last_prefix_audio_nonpad);
    if (merged_audio.samples.empty()) {
        throw std::runtime_error("MOSS-TTS-Local generated no audio samples");
    }
    engine::debug::timing_log_scalar("moss_tts_local.reference_ms", reference_ms);
    engine::debug::timing_log_scalar("moss_tts_local.reference_encode_ms", reference_encode_ms);
    engine::debug::timing_log_scalar("moss_tts_local.prefix_ms", prefix_ms);
    engine::debug::timing_log_scalar("moss_tts_local.generate_ms", generate_ms);
    engine::debug::timing_log_scalar("moss_tts_local.backbone.step.release_ms", backbone_step_release_ms);
    engine::debug::timing_log_scalar("moss_tts_local.code_pack_ms", code_pack_ms);
    engine::debug::timing_log_scalar("moss_tts_local.codec_decode_ms", codec_decode_ms);
    engine::debug::timing_log_scalar("moss_tts_local.interleave_ms", interleave_ms);
    engine::debug::trace_log_scalar("moss_tts_local.backbone.step.released_cache_steps", backbone_step_released_steps);
    engine::debug::trace_log_scalar("moss_tts_local.generated_frames", total_generated_frames);
    engine::debug::timing_log_scalar("session.wall_ms", engine::debug::elapsed_ms(wall_start));

    runtime::TaskResult result;
    result.audio_output = std::move(merged_audio);
    return result;
}

}  // namespace engine::models::moss_tts_local
