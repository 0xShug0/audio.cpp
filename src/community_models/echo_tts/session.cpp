#include "engine/community_models/echo_tts/session.h"

#include "engine/community_models/echo_tts/dit.h"
#include "engine/community_models/echo_tts/latent_post.h"
#include "engine/community_models/echo_tts/tokenizer.h"
#include "engine/framework/model_spec/package.h"
#include "engine/framework/audio/conversion.h"
#include "engine/framework/audio/resampling.h"
#include "engine/framework/audio/waveform_ops.h"
#include "engine/framework/runtime/options.h"
#include "engine/framework/runtime/spec_backed_model.h"
#include "engine/framework/text/chunking.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <stdexcept>
#include <utility>

namespace engine::models::echo_tts {
namespace {

// Mirrors the tap in dit.cpp so the whole pipeline can be traced with one flag.
bool echo_session_debug_enabled() {
    static const bool enabled = [] {
        const char * value = std::getenv("AUDIOCPP_ECHO_TTS_DEBUG");
        return value != nullptr && value[0] != '\0' && value[0] != '0';
    }();
    return enabled;
}

void report_stats(const char * label, const std::vector<float> & values) {
    if (!echo_session_debug_enabled()) {
        return;
    }
    if (values.empty()) {
        std::fprintf(stderr, "  %-26s <empty>\n", label);
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
    std::fprintf(stderr, "  %-26s mean=%+.6f std=%.6f min=%+.4f max=%+.4f  n=%zu\n",
                 label, mean, variance > 0.0 ? std::sqrt(variance) : 0.0,
                 static_cast<double>(low), static_cast<double>(high), values.size());
}

constexpr const char * kFamily = "echo_tts";
constexpr int kSampleRate = 44100;
// ~20 s of English, leaving headroom before the model starts compressing speech
// to fit the fixed 29.72 s window. Overridable per request.
// ~20 s of typical English, against a fixed 29.72 s generation window. Dense or
// fast-reading text can still overrun it, which is the main prompt-dependent
// failure mode; text_chunk_size overrides this per request.
constexpr int64_t kDefaultTextChunkSize = 300;
// Enough for a server rotating a few voices; each slot holds only the projected
// latent, at most 6400 frames x 80 floats = 2 MB.
constexpr std::size_t kDefaultReferenceCacheSlots = 4;

// Default reference trim. Every speaker token stays resident in `keys` for
// every attention in every block at every sampler step, and the speaker encoder
// itself is linear in reference length, so an untrimmed 4.5-minute clip charges
// 1600 tokens to all 24 blocks x 40 steps. 15 s is ~81 tokens, and the model
// card's own guidance is that ~10 s clones at least as well. Override with
// reference_max_seconds per request or echo_tts.reference_max_seconds per
// session; the trained maximum is still reachable that way.
constexpr int64_t kDefaultReferenceMaxSamples = 15 * kSampleRate;

// Adaptive generation window.
//
// Denoiser cost is linear in sequence_length for the projections and the MLP,
// and worse than linear for the self block of the attention, so generating the
// full 640-latent window for a six-second sentence pays roughly five times over
// for latents that find_flattening_point then discards.
//
// The byte-to-frame rate is fixed by the model: 640 frames span 29.7215 s, so
// one second is 21.53 frames. kDefaultTextChunkSize is documented in this file
// as ~20 s of typical English at 300 codepoints, i.e. ~15 bytes/s, which puts
// the ratio at 21.53 / 15 = 1.435 frames per UTF-8 byte. The margin covers
// slower delivery, and a short utterance still needs room for the leading
// silence and the flat tail the crop looks for.
constexpr float kFramesPerTextByte = 1.435F;
constexpr float kWindowMargin = 1.30F;
constexpr int64_t kMinWindowFrames = 128;
// Denoiser graphs are keyed on sequence_length and rebuilt whenever it changes,
// so estimates are snapped to a coarse grid: consecutive chunks of similar
// length then reuse the same graph and the same gallocr reservation.
constexpr int64_t kWindowQuantum = 64;

bool echo_adaptive_window_disabled() {
    static const bool disabled = [] {
        const char * value = std::getenv("AUDIOCPP_ECHO_TTS_NO_ADAPTIVE_WINDOW");
        return value != nullptr && value[0] != '\0' && value[0] != '0';
    }();
    return disabled;
}

// Rounds `frames` up to the graph-reuse grid and clamps into range.
int64_t quantize_window(int64_t frames, int64_t max_frames) {
    frames = std::max<int64_t>(frames, kMinWindowFrames);
    frames = ((frames + kWindowQuantum - 1) / kWindowQuantum) * kWindowQuantum;
    return std::min<int64_t>(frames, max_frames);
}

// Predicts how many latents this chunk needs. Deliberately generous: a window
// that is too short costs a full-length retry, while one that is slightly too
// long only wastes the difference.
int64_t estimate_window_frames(int64_t text_bytes, int64_t max_frames) {
    const auto predicted = static_cast<int64_t>(
        std::ceil(static_cast<double>(text_bytes) * kFramesPerTextByte * kWindowMargin));
    return quantize_window(predicted, max_frames);
}

bool echo_debug_enabled() {
    static const bool enabled = [] {
        const char * value = std::getenv("AUDIOCPP_ECHO_TTS_DEBUG");
        return value != nullptr && value[0] != '\0' && value[0] != '0';
    }();
    return enabled;
}
constexpr size_t kDefaultDitWeightContextBytes = 6144ull * 1024ull * 1024ull;
constexpr size_t kDefaultCodecGraphArenaBytes = 1024ull * 1024ull * 1024ull;
constexpr size_t kDefaultCodecWeightContextBytes = 2048ull * 1024ull * 1024ull;

EchoPcaState load_pca_state(
    const assets::TensorSource & source,
    const EchoTtsConfig & config) {
    EchoPcaState pca;
    // `source` is already a view scoped to the "pca" namespace, so lookups here
    // are bare names; prefixing again would ask for "pca/pca.components".
    pca.components = source.require_f32(
        "components", {config.latent_size, config.ae_latent_dim});
    pca.mean = source.require_f32("mean", {config.ae_latent_dim});
    return pca;
}

std::shared_ptr<const EchoTtsAssets> load_echo_tts_assets(
    const std::filesystem::path & model_path) {
    auto assets = std::make_shared<EchoTtsAssets>();
    assets->resources = engine::model_spec::load_resource_bundle_for_family(model_path, kFamily);
    assets->dit_weights = assets->resources.open_tensor_source("dit_weights");
    auto pca_source = assets->resources.open_tensor_source("pca");
    assets->pca = load_pca_state(*pca_source, assets->config);
    // Published as float32(1/18); reconstructed exactly rather than stored.
    assets->pca.latent_scale = 1.0F / 18.0F;
    assets->config.validate();

    // The Fish S1-DAC travels inside Echo's own GGUF. audio.cpp implements this
    // codec for the fish_audio family and Echo reuses that implementation, but
    // not its weights: fish_audio ships S2 Pro, while Echo's PCA basis is fitted
    // to the S1 DAC's latent space. Only four config fields reach the codec
    // graphs, and their defaults already describe S1-DAC.
    auto codec_assets = std::make_shared<fish_audio::FishAudioAssets>();
    codec_assets->codec_weights = assets->resources.open_tensor_source("codec_weights");
    codec_assets->config.codec.sample_rate = kSampleRate;
    codec_assets->config.codec.frame_length = assets->config.ae_downsample_factor;
    codec_assets->config.codec.total_codebooks = 10;
    codec_assets->config.codec.quantizer_codebooks = 9;
    assets->codec_assets = std::move(codec_assets);
    return assets;
}

}  // namespace

EchoTtsSession::EchoTtsSession(
    runtime::TaskSpec task,
    runtime::SessionOptions options,
    std::shared_ptr<const EchoTtsAssets> assets,
    std::shared_ptr<const engine::model_spec::ModelContract> contract)
    : RuntimeSessionBase(std::move(options)),
      task_(task),
      assets_(std::move(assets)),
      contract_(std::move(contract)) {
    if (contract_ == nullptr) {
        throw std::runtime_error("Echo-TTS session requires a model contract");
    }
    // Without this a typo in server config is silently ignored.
    runtime::validate_spec_backed_session_options(
        RuntimeSessionBase::options(), *contract_, kFamily, "Echo-TTS");
    const auto slots = runtime::parse_int_option(
        RuntimeSessionBase::options().options, {"echo_tts.reference_cache_slots"});
    if (slots.has_value()) {
        if (*slots < 0) {
            throw std::runtime_error("echo_tts.reference_cache_slots must be non-negative");
        }
        reference_cache_.set_capacity(static_cast<std::size_t>(*slots));
    } else {
        reference_cache_.set_capacity(kDefaultReferenceCacheSlots);
    }
    if (assets_ == nullptr) {
        throw std::runtime_error("Echo-TTS session requires loaded assets");
    }
    if (task_.task != runtime::VoiceTaskKind::VoiceCloning ||
        task_.mode != runtime::RunMode::Offline) {
        throw std::runtime_error("Echo-TTS only supports offline voice cloning");
    }
}

EchoTtsSession::~EchoTtsSession() = default;

std::string EchoTtsSession::family() const { return kFamily; }
runtime::VoiceTaskKind EchoTtsSession::task_kind() const { return task_.task; }
runtime::RunMode EchoTtsSession::run_mode() const { return task_.mode; }

EchoSamplerOptions EchoTtsSession::parse_sampler_options(
    const std::unordered_map<std::string, std::string> & options) const {
    EchoSamplerOptions sampler;
    sampler.num_steps =
        runtime::parse_int_option(options, {"num_steps"}).value_or(sampler.num_steps);
    sampler.cfg_scale_text =
        runtime::parse_float_option(options, {"cfg_scale_text"}).value_or(sampler.cfg_scale_text);
    sampler.cfg_scale_speaker =
        runtime::parse_float_option(options, {"cfg_scale_speaker"}).value_or(sampler.cfg_scale_speaker);
    if (const auto truncation = runtime::parse_float_option(options, {"truncation_factor"})) {
        sampler.truncation_factor = *truncation;
    }
    if (const auto kv_scale = runtime::parse_float_option(options, {"speaker_kv_scale"})) {
        // 1.0 is the documented "disabled" value, not a scale to apply.
        if (*kv_scale != 1.0F) {
            sampler.speaker_kv_scale = *kv_scale;
            sampler.speaker_kv_min_t = 0.5F;
        }
    }
    if (const auto seed = runtime::parse_int_option(options, {"seed"})) {
        sampler.seed = static_cast<uint64_t>(std::max(0, *seed));
    }
    // The window defaults to the trained maximum. synthesize_chunk narrows it
    // per chunk unless the caller pins it here, in which case the estimate is
    // skipped entirely and the requested value is used verbatim.
    sampler.sequence_length = assets_->config.max_sequence_length;
    if (const auto window = runtime::parse_int_option(options, {"sequence_length"})) {
        if (*window <= 0 || *window > assets_->config.max_sequence_length) {
            throw std::runtime_error(
                "Echo-TTS sequence_length must be in 1..max_sequence_length");
        }
        sampler.sequence_length = *window;
        sampler.window_pinned = true;
    }
    if (const auto interval = runtime::parse_int_option(options, {"cfg_interval"})) {
        if (*interval < 1) {
            throw std::runtime_error("Echo-TTS cfg_interval must be at least 1");
        }
        sampler.cfg_interval = *interval;
    }
    if (sampler.num_steps <= 0) {
        throw std::runtime_error("Echo-TTS num_steps must be positive");
    }
    return sampler;
}

void EchoTtsSession::prepare(const runtime::SessionPreparationRequest & request) {
    (void)request;
    if (dit_ == nullptr) {
        dit_ = std::make_unique<EchoDitRuntime>(
            assets_->config,
            *assets_->dit_weights,
            // Namespace-scoped source: tensor names are already stripped of the
            // "dit_weights/" prefix by the resource bundle.
            "",
            execution_context(),
            assets::TensorStorageType::Native);
    }
    if (codec_ == nullptr) {
        if (assets_->codec_assets == nullptr ||
            assets_->codec_assets->codec_weights == nullptr) {
            throw std::runtime_error(
                "Echo-TTS GGUF has no codec_weights; re-run convert_echo_tts.py with "
                "--fish-dir pointing at the Fish S1-DAC checkpoint");
        }
        const int threads = options().backend.threads > 0 ? options().backend.threads : 1;
        codec_ = std::make_unique<fish_audio::FishAudioCodecRuntime>(
            assets_->codec_assets,
            options().backend,
            threads,
            kDefaultCodecGraphArenaBytes,
            kDefaultCodecWeightContextBytes,
            assets::TensorStorageType::Native,
            assets::TensorStorageType::Native);
    }
    mark_prepared();
}

int64_t EchoTtsSession::resolve_reference_max_samples(
    const std::unordered_map<std::string, std::string> & request_options) const {
    const auto & config = assets_->config;
    const int64_t trained_max = config.max_speaker_latent_length * config.ae_downsample_factor;

    // A per-request value wins; otherwise the session default from CLI or server
    // config; otherwise kDefaultReferenceMaxSamples (15 s), NOT the trained
    // maximum -- see that constant for why, and model_specs/echo_tts.json which
    // publishes 15.0 as the default in both scopes. Request options are bare names,
    // session options carry the family prefix -- parse_cli_options adds it for
    // the session and load scopes only.
    auto seconds = runtime::parse_float_option(request_options, {"reference_max_seconds"});
    if (!seconds.has_value()) {
        seconds = runtime::parse_float_option(
            options().options, {"echo_tts.reference_max_seconds"});
    }
    if (!seconds.has_value()) {
        return kDefaultReferenceMaxSamples;
    }
    if (!(*seconds > 0.0F)) {
        throw std::runtime_error("Echo-TTS reference_max_seconds must be positive");
    }
    const auto requested = static_cast<int64_t>(
        static_cast<double>(*seconds) * static_cast<double>(kSampleRate));
    // Clamped rather than rejected: asking for more than the model was trained
    // on is a reasonable thing to type, and silently exceeding it is not.
    return std::min<int64_t>(requested, trained_max);
}

namespace {

// Cheap content hash over the reference samples. A collision would swap one
// speaker for another, so it mixes length, rate, channels and every sample
// rather than sampling, and the cache key adds the trim length.
std::string reference_identity(const runtime::AudioBuffer & audio) {
    std::uint64_t hash = 1469598103934665603ULL;
    auto mix = [&hash](std::uint64_t value) {
        hash ^= value;
        hash *= 1099511628211ULL;
    };
    mix(static_cast<std::uint64_t>(audio.samples.size()));
    mix(static_cast<std::uint64_t>(audio.sample_rate));
    mix(static_cast<std::uint64_t>(audio.channels));
    for (const float sample : audio.samples) {
        std::uint32_t bits = 0;
        std::memcpy(&bits, &sample, sizeof(bits));
        mix(bits);
    }
    return std::to_string(hash);
}

}  // namespace

void EchoTtsSession::encode_speaker(const runtime::AudioBuffer & audio) {
    const auto & config = assets_->config;

    const EchoReferenceIdentity identity{reference_identity(audio), reference_max_samples_};
    if (const auto * cached = reference_cache_.find(identity)) {
        speaker_latent_ = cached->latent;
        speaker_frames_ = cached->frames;
        if (echo_debug_enabled()) {
            std::fprintf(stderr, "[echo_tts] speaker reference cache hit (%lld frames)\n",
                         static_cast<long long>(speaker_frames_));
        }
        return;
    }

    // Mixed down and resampled once, so chunk boundaries land on exact codec
    // frames rather than on pre-resample sample indices.
    auto mono = engine::audio::mixdown_interleaved_to_mono_average(audio.samples, audio.channels);
    if (audio.sample_rate != kSampleRate) {
        mono = engine::audio::resample_mono_torchaudio_sinc_hann(
            mono, audio.sample_rate, kSampleRate);
    }

    const int64_t chunk_samples = config.speaker_chunk_latents * config.ae_downsample_factor;
    if (static_cast<int64_t>(mono.size()) > reference_max_samples_) {
        if (echo_debug_enabled()) {
            std::fprintf(stderr, "[echo_tts] reference trimmed %.2f s -> %.2f s\n",
                         static_cast<double>(mono.size()) / kSampleRate,
                         static_cast<double>(reference_max_samples_) / kSampleRate);
        }
        mono.resize(static_cast<size_t>(reference_max_samples_));
    }
    const int64_t actual_frames = static_cast<int64_t>(mono.size()) / config.ae_downsample_factor;

    // Encoded in ~30 s chunks, zero-padded to a fixed length, exactly as
    // inference.py::get_speaker_latent_and_mask does. That is not only a memory
    // measure: the chunk size is the longest span seen in training, and a single
    // pass over several minutes of audio is a different computation. It also
    // keeps every encode graph the same shape, so one graph is reused.
    std::vector<float> latents;
    latents.reserve(static_cast<size_t>(actual_frames + config.speaker_chunk_latents) *
                    static_cast<size_t>(config.latent_size));
    for (int64_t offset = 0; offset < static_cast<int64_t>(mono.size()); offset += chunk_samples) {
        const int64_t available =
            std::min<int64_t>(chunk_samples, static_cast<int64_t>(mono.size()) - offset);
        runtime::AudioBuffer chunk{kSampleRate, 1, std::vector<float>(
            static_cast<size_t>(chunk_samples), 0.0F)};
        std::copy_n(mono.begin() + offset, available, chunk.samples.begin());

        auto chunk_latents = codec_->encode_zq(chunk);
        auto projected = pca_project(
            assets_->pca, config, chunk_latents.values, chunk_latents.frames);
        latents.insert(latents.end(), projected.begin(), projected.end());
    }

    // Trim the padding introduced by the final chunk, then crop to a multiple of
    // the patch size the speaker encoder folds over.
    int64_t frames = std::min<int64_t>(
        actual_frames, static_cast<int64_t>(latents.size()) / config.latent_size);
    frames = frames / config.speaker_patch_size * config.speaker_patch_size;
    if (frames <= 0) {
        throw std::runtime_error(
            "Echo-TTS speaker reference is too short; at least "
            "4 latent frames (~0.19 s) are required");
    }
    latents.resize(static_cast<size_t>(frames * config.latent_size));
    speaker_latent_ = std::move(latents);
    speaker_frames_ = frames;
    reference_cache_.put(identity, EchoPreparedSpeaker{speaker_latent_, speaker_frames_});
}

runtime::AudioBuffer EchoTtsSession::synthesize_chunk(
    const std::string & text,
    const EchoSamplerOptions & sampler) {
    const auto & config = assets_->config;
    auto tokens = tokenize_echo_text(text, config.max_text_length, true, false);
    if (tokens.truncated) {
        std::fprintf(
            stderr,
            "[echo_tts] warning: text truncated at %lld bytes; the tail will not be spoken\n",
            static_cast<long long>(config.max_text_length));
    }

    EchoConditioning conditioning;
    conditioning.text_input_ids = tokens.input_ids;
    conditioning.text_mask = tokens.mask;
    conditioning.text_length = static_cast<int64_t>(tokens.input_ids.size());
    conditioning.speaker_latent = speaker_latent_;
    conditioning.speaker_mask.assign(static_cast<size_t>(speaker_frames_), 1.0F);
    conditioning.speaker_frames = speaker_frames_;

    dit_->prepare_conditioning(conditioning);

    // Adaptive window. The estimate is attempted first; a missing flattening
    // point means the model was still speaking when the window closed, so the
    // chunk is regenerated once at the full trained length. The seed is
    // unchanged between attempts, so the retry is the run that would have
    // happened without this optimisation -- an under-estimate costs time, never
    // fidelity.
    EchoSamplerOptions attempt = sampler;
    const bool adaptive = !sampler.window_pinned && !echo_adaptive_window_disabled();
    if (adaptive) {
        attempt.sequence_length = estimate_window_frames(
            static_cast<int64_t>(tokens.input_ids.size()), config.max_sequence_length);
    }

    std::vector<float> latent;
    int64_t frames = 0;
    for (int pass = 0; pass < 2; ++pass) {
        latent = dit_->sample(attempt);
        frames = find_flattening_point(latent, attempt.sequence_length, config.latent_size);
        const bool ran_out = frames >= attempt.sequence_length;
        const bool can_retry =
            adaptive && ran_out && attempt.sequence_length < config.max_sequence_length;
        if (!can_retry) {
            break;
        }
        if (echo_debug_enabled()) {
            std::fprintf(
                stderr,
                "[echo_tts] window estimate of %lld frames was short for %lld bytes; "
                "retrying at %lld\n",
                static_cast<long long>(attempt.sequence_length),
                static_cast<long long>(tokens.input_ids.size()),
                static_cast<long long>(config.max_sequence_length));
        }
        attempt.sequence_length = config.max_sequence_length;
    }
    report_stats("sampler.latent", latent);

    // The generated tail goes flat once the model finishes speaking; cropping
    // there is what sets the output duration.
    const double window_seconds =
        static_cast<double>(attempt.sequence_length * config.ae_downsample_factor) /
        static_cast<double>(config.sample_rate);
    if (frames >= attempt.sequence_length) {
        // No flat tail means the model was still speaking when the window ended,
        // so the audio is cut mid-utterance. Almost always too much text for one
        // chunk rather than a sampling problem.
        std::fprintf(
            stderr,
            "[echo_tts] warning: no silence found within the %.2f s window for a "
            "%lld-byte chunk; output is truncated mid-utterance. Try a smaller "
            "text_chunk_size.\n",
            window_seconds, static_cast<long long>(tokens.input_ids.size()));
    } else if (echo_debug_enabled()) {
        std::fprintf(
            stderr,
            "[echo_tts] chunk: %lld tokens -> %lld/%lld frames (%.2f s of %.2f s window)\n",
            static_cast<long long>(tokens.input_ids.size()),
            static_cast<long long>(frames),
            static_cast<long long>(attempt.sequence_length),
            static_cast<double>(frames * config.ae_downsample_factor) /
                static_cast<double>(config.sample_rate),
            window_seconds);
    }
    if (frames <= 0) {
        return runtime::AudioBuffer{kSampleRate, 1, {}};
    }
    if (echo_session_debug_enabled()) {
        std::fprintf(stderr, "  %-26s %lld of %lld frames (%.3f s)\n",
                     "flattening_point", static_cast<long long>(frames),
                     static_cast<long long>(attempt.sequence_length),
                     static_cast<double>(frames * config.ae_downsample_factor) /
                         static_cast<double>(config.sample_rate));
    }
    latent.resize(static_cast<size_t>(frames * config.latent_size));
    report_stats("latent.cropped", latent);

    auto z_q = pca_unproject(assets_->pca, config, latent, frames);
    report_stats("decode.z_q", z_q);
    auto audio = codec_->decode_zq(z_q, frames);
    report_stats("decode.audio", audio.samples);
    return audio;
}

runtime::TaskResult EchoTtsSession::run(const runtime::TaskRequest & request) {
    require_prepared("Echo-TTS run");
    runtime::validate_spec_backed_request_options(request.options, *contract_, "Echo-TTS");

    if (!request.text_input.has_value() || request.text_input->text.empty()) {
        throw std::runtime_error("Echo-TTS requires text input");
    }
    if (!request.voice.has_value() || !request.voice->speaker.has_value() ||
        !request.voice->speaker->audio.has_value()) {
        throw std::runtime_error(
            "Echo-TTS requires speaker reference audio; pass --voice-ref <wav> "
            "(--target-voice is for path-based voice conversion, not cloning)");
    }

    const auto sampler = parse_sampler_options(request.options);
    reference_max_samples_ = resolve_reference_max_samples(request.options);
    // Encoded once per request; the timbre is then identical across chunk seams
    // by construction.
    encode_speaker(*request.voice->speaker->audio);
    report_stats("speaker.latent", speaker_latent_);

    const int64_t chunk_size =
        engine::text::parse_text_chunk_size_override(request.options)
            .value_or(kDefaultTextChunkSize);
    const auto chunks = runtime::chunk_text_request(request, chunk_size);
    if (echo_debug_enabled()) {
        std::fprintf(
            stderr, "[echo_tts] %zu chunk(s) at a %lld-codepoint budget\n",
            chunks.size(), static_cast<long long>(chunk_size));
    }

    runtime::TaskResult result;
    runtime::AudioBuffer output{kSampleRate, 1, {}};
    for (const auto & chunk : chunks) {
        if (!chunk.text_input.has_value() || chunk.text_input->text.empty()) {
            continue;
        }
        auto audio = synthesize_chunk(chunk.text_input->text, sampler);
        runtime::append_audio_buffer(output, audio);
    }
    // Echo's output level is prosody-dependent and can exceed full scale on
    // emphatic prompts; upstream's own loader carries a "should we target a
    // specific energy level?" note. Divide by the peak only when it exceeds
    // 1.0, so quiet output is left untouched and loud output is limited rather
    // than clipped at the WAV writer.
    engine::audio::normalize_peak_to_unit_range_and_clamp_in_place(output.samples);
    result.audio_output = std::move(output);
    return result;
}

void EchoTtsSession::reset() {
    speaker_latent_.clear();
    speaker_frames_ = 0;
}

std::shared_ptr<runtime::IVoiceModelLoader> make_echo_tts_loader() {
    runtime::SpecBackedVoiceModelConfig<EchoTtsAssets> config;
    config.family = kFamily;
    config.load_assets = load_echo_tts_assets;
    config.create_session = [](
                                const runtime::TaskSpec & task,
                                const runtime::SessionOptions & options,
                                std::shared_ptr<const EchoTtsAssets> assets,
                                std::shared_ptr<const engine::model_spec::ModelContract> contract) {
        return std::make_unique<EchoTtsSession>(
            task,
            options,
            std::move(assets),
            std::move(contract));
    };
    return runtime::make_spec_backed_voice_loader(std::move(config));
}

}  // namespace engine::models::echo_tts
