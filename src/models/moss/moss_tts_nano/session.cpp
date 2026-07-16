#include "engine/models/moss/moss_tts_nano/session.h"

#include "engine/framework/audio/resampling.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/runtime/options.h"
#include "engine/framework/text/chunking.h"
#include "engine/models/moss/shared/audio_tokenizer_config.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string_view>
#include <utility>
#include <vector>

namespace engine::models::moss_tts_nano {
namespace {

using Clock = std::chrono::steady_clock;

constexpr int64_t kDefaultTextChunkSize = 256;

constexpr size_t kDefaultGlobalPrefillGraphArenaBytes = 256ull * 1024ull * 1024ull;
constexpr size_t kDefaultGlobalDecodeGraphArenaBytes = 128ull * 1024ull * 1024ull;
constexpr size_t kDefaultGlobalWeightContextBytes = 512ull * 1024ull * 1024ull;
constexpr size_t kDefaultLocalFrameGraphArenaBytes = 64ull * 1024ull * 1024ull;
constexpr size_t kDefaultLocalFrameWeightContextBytes = 128ull * 1024ull * 1024ull;
constexpr size_t kDefaultAudioTokenizerEncoderGraphArenaBytes = 64ull * 1024ull * 1024ull;
constexpr size_t kDefaultAudioTokenizerDecoderGraphArenaBytes = 64ull * 1024ull * 1024ull;
constexpr size_t kDefaultAudioTokenizerWeightContextBytes = 128ull * 1024ull * 1024ull;
constexpr int kAudioTokenizerSampleRate = 48000;

std::shared_ptr<const MossTTSNanoAssets> require_assets(std::shared_ptr<const MossTTSNanoAssets> assets) {
    if (assets == nullptr) {
        throw std::runtime_error("MOSS-TTS-Nano session requires assets");
    }
    return assets;
}

engine::assets::TensorStorageType require_matmul_weight_storage(
    engine::assets::TensorStorageType storage_type,
    const char * option_name) {
    if (storage_type == engine::assets::TensorStorageType::Native ||
        storage_type == engine::assets::TensorStorageType::F32 ||
        storage_type == engine::assets::TensorStorageType::F16 ||
        storage_type == engine::assets::TensorStorageType::BF16 ||
        storage_type == engine::assets::TensorStorageType::Q8_0) {
        return storage_type;
    }
    throw std::runtime_error(std::string(option_name) + " currently supports only native, f32, f16, bf16, and q8_0");
}

engine::assets::TensorStorageType option_weight_type(
    const runtime::SessionOptions & options,
    const char * key,
    engine::assets::TensorStorageType default_value) {
    const auto it = options.options.find(key);
    if (it == options.options.end()) {
        return default_value;
    }
    return require_matmul_weight_storage(engine::assets::parse_tensor_storage_type(it->second), key);
}

MossTTSNanoGenerationOptions generation_options_from_request(
    const runtime::TaskRequest & request,
    const MossTTSNanoConfig & config) {
    MossTTSNanoGenerationOptions options;
    options.active_codebooks = config.n_vq;
    if (const auto value = runtime::parse_int_option(request.options, {"max_tokens"})) {
        options.max_new_frames = *value;
        if (options.max_new_frames <= 0) {
            throw std::runtime_error("MOSS-TTS-Nano max_tokens must be positive");
        }
    }
    if (const auto value = runtime::parse_int_option(request.options, {"active_codebooks"})) {
        options.active_codebooks = *value;
        if (options.active_codebooks <= 0 || options.active_codebooks > config.n_vq) {
            throw std::runtime_error("MOSS-TTS-Nano active_codebooks is out of range");
        }
    }
    options.seed = runtime::parse_u32_option(request.options, {"seed"})
        .value_or(runtime::random_u32_seed());
    if (const auto value = runtime::find_option(request.options, {"do_sample"})) {
        options.sampling.do_sample = runtime::parse_bool_option(*value, "do_sample");
    }
    if (const auto value = runtime::parse_float_option(request.options, {"text_temperature"})) {
        options.sampling.text_temperature = *value;
    }
    if (const auto value = runtime::parse_float_option(request.options, {"text_top_p"})) {
        options.sampling.text_top_p = *value;
    }
    if (const auto value = runtime::parse_int_option(request.options, {"text_top_k"})) {
        options.sampling.text_top_k = *value;
    }
    if (const auto value = runtime::parse_float_option(request.options, {"temperature"})) {
        options.sampling.audio_temperature = *value;
    }
    if (const auto value = runtime::parse_float_option(request.options, {"top_p"})) {
        options.sampling.audio_top_p = *value;
    }
    if (const auto value = runtime::parse_int_option(request.options, {"top_k"})) {
        options.sampling.audio_top_k = *value;
    }
    if (const auto value = runtime::parse_float_option(request.options, {"repetition_penalty"})) {
        options.sampling.audio_repetition_penalty = *value;
    }
    if (options.sampling.text_temperature <= 0.0F) {
        throw std::runtime_error("MOSS-TTS-Nano text_temperature must be positive");
    }
    if (options.sampling.audio_temperature <= 0.0F) {
        throw std::runtime_error("MOSS-TTS-Nano temperature must be positive");
    }
    if (options.sampling.text_top_p < 0.0F || options.sampling.text_top_p > 1.0F) {
        throw std::runtime_error("MOSS-TTS-Nano text_top_p must be within [0, 1]");
    }
    if (options.sampling.audio_top_p < 0.0F || options.sampling.audio_top_p > 1.0F) {
        throw std::runtime_error("MOSS-TTS-Nano top_p must be within [0, 1]");
    }
    if (options.sampling.text_top_k < 0) {
        throw std::runtime_error("MOSS-TTS-Nano text_top_k must be non-negative");
    }
    if (options.sampling.audio_top_k < 0) {
        throw std::runtime_error("MOSS-TTS-Nano top_k must be non-negative");
    }
    if (options.sampling.audio_repetition_penalty <= 0.0F) {
        throw std::runtime_error("MOSS-TTS-Nano repetition_penalty must be positive");
    }
    return options;
}

bool audio_buffers_equal(const runtime::AudioBuffer & lhs, const runtime::AudioBuffer & rhs) {
    return lhs.sample_rate == rhs.sample_rate &&
           lhs.channels == rhs.channels &&
           lhs.samples.size() == rhs.samples.size() &&
           std::memcmp(lhs.samples.data(), rhs.samples.data(), lhs.samples.size() * sizeof(float)) == 0;
}

std::vector<std::vector<float>> reference_to_audio_tokenizer_stereo(const runtime::AudioBuffer & audio) {
    const int channels = std::max(1, audio.channels);
    const int64_t frames = channels > 0 ? static_cast<int64_t>(audio.samples.size()) / channels : 0;
    if (frames <= 0) {
        throw std::runtime_error("MOSS-TTS-Nano voice reference audio is empty");
    }
    std::vector<std::vector<float>> stereo(2, std::vector<float>(static_cast<size_t>(frames)));
#ifdef _OPENMP
#pragma omp parallel for if(frames >= 4096)
#endif
    for (int64_t frame = 0; frame < frames; ++frame) {
        const float left = audio.samples[static_cast<size_t>(frame * channels)];
        const float right = channels > 1 ? audio.samples[static_cast<size_t>(frame * channels + 1)] : left;
        stereo[0][static_cast<size_t>(frame)] = left;
        stereo[1][static_cast<size_t>(frame)] = right;
    }
    if (audio.sample_rate != kAudioTokenizerSampleRate) {
        if (audio.sample_rate <= 0) {
            throw std::runtime_error("MOSS-TTS-Nano voice reference has an invalid sample rate");
        }
        for (auto & channel : stereo) {
            channel = engine::audio::resample_mono_torchaudio_sinc_hann(channel, audio.sample_rate, kAudioTokenizerSampleRate);
        }
    }
    return stereo;
}

}  // namespace

MossTTSNanoSession::MossTTSNanoSession(
    runtime::TaskSpec task,
    runtime::SessionOptions options,
    std::shared_ptr<const MossTTSNanoAssets> assets)
    : RuntimeSessionBase(options),
      task_(task),
      assets_(require_assets(std::move(assets))),
      global_prefill_graph_arena_bytes_(runtime::parse_size_mb_option(options.options, {"moss_tts_nano.global_prefill_graph_arena_mb"}, kDefaultGlobalPrefillGraphArenaBytes)),
      global_decode_graph_arena_bytes_(runtime::parse_size_mb_option(options.options, {"moss_tts_nano.global_decode_graph_arena_mb"}, kDefaultGlobalDecodeGraphArenaBytes)),
      global_weight_context_bytes_(runtime::parse_size_mb_option(options.options, {"moss_tts_nano.global_weight_context_mb"}, kDefaultGlobalWeightContextBytes)),
      local_frame_graph_arena_bytes_(runtime::parse_size_mb_option(options.options, {"moss_tts_nano.local_frame_graph_arena_mb"}, kDefaultLocalFrameGraphArenaBytes)),
      local_frame_weight_context_bytes_(runtime::parse_size_mb_option(options.options, {"moss_tts_nano.local_frame_weight_context_mb"}, kDefaultLocalFrameWeightContextBytes)),
      audio_tokenizer_encoder_graph_arena_bytes_(runtime::parse_size_mb_option(options.options, {"moss_tts_nano.audio_tokenizer_encoder_graph_arena_mb"}, kDefaultAudioTokenizerEncoderGraphArenaBytes)),
      audio_tokenizer_decoder_graph_arena_bytes_(runtime::parse_size_mb_option(options.options, {"moss_tts_nano.audio_tokenizer_decoder_graph_arena_mb"}, kDefaultAudioTokenizerDecoderGraphArenaBytes)),
      audio_tokenizer_weight_context_bytes_(runtime::parse_size_mb_option(options.options, {"moss_tts_nano.audio_tokenizer_weight_context_mb"}, kDefaultAudioTokenizerWeightContextBytes)),
      global_weight_storage_type_(option_weight_type(
          options,
          "moss_tts_nano.global_weight_type",
          option_weight_type(options, "moss_tts_nano.weight_type", engine::assets::TensorStorageType::Native))),
      local_frame_weight_storage_type_(option_weight_type(
          options,
          "moss_tts_nano.local_frame_weight_type",
          option_weight_type(options, "moss_tts_nano.weight_type", engine::assets::TensorStorageType::Native))),
      text_tokenizer_(assets_),
      prompt_builder_(assets_, text_tokenizer_),
      global_transformer_(
          assets_,
          execution_context(),
          global_prefill_graph_arena_bytes_,
          global_decode_graph_arena_bytes_,
          global_weight_context_bytes_,
          global_weight_storage_type_),
      local_frame_decoder_(
          assets_,
          execution_context(),
          local_frame_graph_arena_bytes_,
          local_frame_weight_context_bytes_,
          local_frame_weight_storage_type_),
      generator_(global_transformer_, local_frame_decoder_),
      decoder_(
          *assets_->audio_tokenizer_weights,
          execution_context(),
          assets_->config.n_vq,
          audio_tokenizer_weight_context_bytes_,
          audio_tokenizer_decoder_graph_arena_bytes_,
          moss::moss_audio_tokenizer_nano_config()) {
    if (task_.task != runtime::VoiceTaskKind::Tts && task_.task != runtime::VoiceTaskKind::VoiceCloning) {
        throw std::runtime_error("MOSS-TTS-Nano only supports the Tts and VoiceCloning tasks");
    }
    if (task_.mode != runtime::RunMode::Offline) {
        throw std::runtime_error("MOSS-TTS-Nano currently supports offline sessions");
    }
    for (const auto & [key, value] : options.options) {
        (void) value;
        if (key.rfind("moss_tts_nano.", 0) == 0 &&
            key != "moss_tts_nano.global_prefill_graph_arena_mb" &&
            key != "moss_tts_nano.global_decode_graph_arena_mb" &&
            key != "moss_tts_nano.global_weight_context_mb" &&
            key != "moss_tts_nano.local_frame_graph_arena_mb" &&
            key != "moss_tts_nano.local_frame_weight_context_mb" &&
            key != "moss_tts_nano.audio_tokenizer_encoder_graph_arena_mb" &&
            key != "moss_tts_nano.audio_tokenizer_decoder_graph_arena_mb" &&
            key != "moss_tts_nano.audio_tokenizer_weight_context_mb" &&
            key != "moss_tts_nano.global_weight_type" &&
            key != "moss_tts_nano.local_frame_weight_type" &&
            key != "moss_tts_nano.weight_type") {
            throw std::runtime_error("unknown MOSS-TTS-Nano session option: " + key);
        }
    }
    assets_->model_weights->release_storage();
}

std::string MossTTSNanoSession::family() const {
    return "moss_tts_nano";
}

runtime::VoiceTaskKind MossTTSNanoSession::task_kind() const {
    return task_.task;
}

runtime::RunMode MossTTSNanoSession::run_mode() const {
    return task_.mode;
}

void MossTTSNanoSession::prepare(const runtime::SessionPreparationRequest & request) {
    if (!request.text.has_value()) {
        throw std::runtime_error("MOSS-TTS-Nano prepare() requires text");
    }
    if (request.voice.has_value() && request.voice->speaker.has_value() &&
        request.voice->speaker->audio.has_value()) {
        prepared_prompt_audio_ = *request.voice->speaker->audio;
        prepared_reference_codes_ = encode_reference_audio(
            *request.voice->speaker->audio,
            assets_->config.n_vq);
    } else {
        prepared_prompt_audio_.reset();
        prepared_reference_codes_.reset();
    }
    mark_prepared();
}

moss::MossAudioTokenizerEncoder & MossTTSNanoSession::encoder() {
    if (encoder_ == nullptr) {
        reference_encoder_execution_context_ = std::make_unique<core::ExecutionContext>(options().backend);
        encoder_ = std::make_unique<moss::MossAudioTokenizerEncoder>(
            *assets_->audio_tokenizer_weights,
            *reference_encoder_execution_context_,
            assets_->config.n_vq,
            audio_tokenizer_weight_context_bytes_,
            audio_tokenizer_encoder_graph_arena_bytes_,
            moss::moss_audio_tokenizer_nano_config());
    }
    return *encoder_;
}

MossTTSNanoAudioCodes MossTTSNanoSession::encode_reference_audio(
    const runtime::AudioBuffer & audio,
    int64_t active_codebooks) {
    const auto stereo = reference_to_audio_tokenizer_stereo(audio);
    const auto codes = encoder().encode(stereo);
    if (static_cast<int64_t>(codes.size()) < active_codebooks) {
        throw std::runtime_error("MOSS-TTS-Nano reference encoder returned too few codebooks");
    }
    const int64_t frames = codes.empty() ? 0 : static_cast<int64_t>(codes.front().size());
    if (frames <= 0) {
        throw std::runtime_error("MOSS-TTS-Nano reference encoder returned no frames");
    }
    MossTTSNanoAudioCodes out;
    out.frames = frames;
    out.codebooks = active_codebooks;
    out.token_ids.resize(static_cast<size_t>(frames * active_codebooks));
#ifdef _OPENMP
#pragma omp parallel for collapse(2) if(frames * active_codebooks >= 4096)
#endif
    for (int64_t frame = 0; frame < frames; ++frame) {
        for (int64_t codebook = 0; codebook < active_codebooks; ++codebook) {
            out.token_ids[static_cast<size_t>(frame * active_codebooks + codebook)] =
                codes[static_cast<size_t>(codebook)][static_cast<size_t>(frame)];
        }
    }
    return out;
}

runtime::AudioBuffer MossTTSNanoSession::decode_generated_audio(
    const MossTTSNanoAudioCodes & codes,
    int64_t active_codebooks) {
    if (codes.frames <= 0 || active_codebooks <= 0 || active_codebooks > codes.codebooks) {
        throw std::runtime_error("MOSS-TTS-Nano decode requires non-empty generated codes");
    }
    std::vector<std::vector<int32_t>> tokenizer_codes(
        static_cast<size_t>(active_codebooks),
        std::vector<int32_t>(static_cast<size_t>(codes.frames)));
#ifdef _OPENMP
#pragma omp parallel for collapse(2) if(codes.frames * active_codebooks >= 4096)
#endif
    for (int64_t frame = 0; frame < codes.frames; ++frame) {
        for (int64_t codebook = 0; codebook < active_codebooks; ++codebook) {
            tokenizer_codes[static_cast<size_t>(codebook)][static_cast<size_t>(frame)] =
                codes.token_ids[static_cast<size_t>(frame * codes.codebooks + codebook)];
        }
    }
    const auto channels = decoder_.decode(tokenizer_codes);
    const int channel_count = static_cast<int>(channels.size());
    const size_t samples_per_channel = channels.empty() ? 0 : channels.front().size();
    if (channel_count <= 0 || samples_per_channel == 0) {
        throw std::runtime_error("MOSS-TTS-Nano audio tokenizer decoder produced no audio");
    }
    runtime::AudioBuffer audio;
    audio.sample_rate = static_cast<int>(decoder_.sampling_rate());
    audio.channels = channel_count;
    audio.samples.resize(samples_per_channel * static_cast<size_t>(channel_count));
    const int64_t sample_count = static_cast<int64_t>(samples_per_channel);
#ifdef _OPENMP
#pragma omp parallel for if(sample_count * channel_count >= 4096)
#endif
    for (int64_t sample = 0; sample < sample_count; ++sample) {
        for (int channel = 0; channel < channel_count; ++channel) {
            audio.samples[static_cast<size_t>(sample) * static_cast<size_t>(channel_count) + static_cast<size_t>(channel)] =
                channels[static_cast<size_t>(channel)][static_cast<size_t>(sample)];
        }
    }
    return audio;
}

runtime::TaskResult MossTTSNanoSession::run(const runtime::TaskRequest & request) {
    require_prepared("MOSS-TTS-Nano run()");
    const auto wall_start = Clock::now();
    const int64_t text_chunk_size =
        engine::text::parse_text_chunk_size_override(request.options).value_or(kDefaultTextChunkSize);
    const auto text_chunk_mode =
        engine::text::parse_text_chunk_mode_override(request.options).value_or(engine::text::TextChunkMode::Default);
    const auto chunk_requests = runtime::chunk_text_request(request, text_chunk_size, text_chunk_mode);
    debug::trace_log_scalar("moss_tts_nano.text_chunk_size", text_chunk_size);
    debug::trace_log_scalar("moss_tts_nano.text_chunk_mode", engine::text::text_chunk_mode_name(text_chunk_mode));
    debug::trace_log_scalar("moss_tts_nano.text_chunk_count", static_cast<int64_t>(chunk_requests.size()));
    double audio_tokenizer_encode_ms = 0.0;
    double prompt_ms = 0.0;
    double generate_ms = 0.0;
    double audio_tokenizer_decode_ms = 0.0;
    int64_t global_graph_rebuilds = 0;
    int64_t chunk_index = 0;
    int64_t max_new_frames_hit_count = 0;
    runtime::AudioBuffer merged_audio;
    for (const auto & chunk_request : chunk_requests) {
        const auto tts_request = make_request(chunk_request);
        const MossTTSNanoAudioCodes * reference_codes = nullptr;
        const auto audio_tokenizer_encode_start = Clock::now();
        if (tts_request.has_prompt_audio) {
            reference_codes = &reference_codes_for_request(tts_request);
        }
        audio_tokenizer_encode_ms += engine::debug::elapsed_ms(audio_tokenizer_encode_start, Clock::now());
        const auto prompt_start = Clock::now();
        const auto prompt = prompt_builder_.build(tts_request, reference_codes);
        prompt_ms += engine::debug::elapsed_ms(prompt_start, Clock::now());
        const auto generate_start = Clock::now();
        const int64_t global_graph_builds_before = global_transformer_.graph_builds();
        const auto generated_codes = generator_.generate(prompt, tts_request.generation);
        const int64_t global_graph_builds_after = global_transformer_.graph_builds();
        generate_ms += engine::debug::elapsed_ms(generate_start, Clock::now());
        global_graph_rebuilds += global_graph_builds_after - global_graph_builds_before;
        if (generated_codes.hit_max_new_frames) {
            ++max_new_frames_hit_count;
        }
        debug::trace_log_scalar("moss_tts_nano.chunk.index", chunk_index);
        debug::trace_log_scalar(
            "moss_tts_nano.chunk.text_chars",
            static_cast<int64_t>(chunk_request.text_input->text.size()));
        debug::trace_log_scalar("moss_tts_nano.chunk.generated_frames", generated_codes.frames);
        debug::trace_log_scalar("moss_tts_nano.chunk.max_new_frames", tts_request.generation.max_new_frames);
        debug::trace_log_scalar("moss_tts_nano.chunk.hit_max_new_frames", generated_codes.hit_max_new_frames);
        debug::trace_log_scalar(
            "moss_tts_nano.chunk.stop_reason",
            std::string_view(generated_codes.hit_max_new_frames ? "max_new_frames" : "eoc"));
        const auto audio_tokenizer_decode_start = Clock::now();
        runtime::append_audio_buffer(
            merged_audio,
            decode_generated_audio(
                generated_codes,
                tts_request.generation.active_codebooks));
        audio_tokenizer_decode_ms += engine::debug::elapsed_ms(audio_tokenizer_decode_start, Clock::now());
        ++chunk_index;
    }

    runtime::TaskResult result;
    result.audio_output = std::move(merged_audio);
    debug::timing_log_scalar("moss_tts_nano.session.audio_tokenizer_encode_ms", audio_tokenizer_encode_ms);
    debug::timing_log_scalar("moss_tts_nano.session.prompt_ms", prompt_ms);
    debug::timing_log_scalar("moss_tts_nano.session.generate_ms", generate_ms);
    debug::timing_log_scalar("moss_tts_nano.session.audio_tokenizer_decode_ms", audio_tokenizer_decode_ms);
    debug::timing_log_scalar("moss_tts_nano.session.max_new_frames_hit_count", max_new_frames_hit_count);
    debug::timing_log_scalar("moss_tts_nano.global.graph.rebuilds", global_graph_rebuilds);
    debug::timing_log_scalar("session.wall_ms", engine::debug::elapsed_ms(wall_start, Clock::now()));
    return result;
}

MossTTSNanoRequest MossTTSNanoSession::make_request(const runtime::TaskRequest & request) const {
    if (!request.text_input.has_value() || request.text_input->text.empty()) {
        throw std::runtime_error("MOSS-TTS-Nano run() requires --text");
    }
    MossTTSNanoRequest out;
    out.text = request.text_input->text;
    if (const auto it = request.options.find("reference_text"); it != request.options.end()) {
        out.prompt_text = it->second;
    }
    if (request.voice.has_value() && request.voice->speaker.has_value() &&
        request.voice->speaker->audio.has_value()) {
        out.prompt_audio = *request.voice->speaker->audio;
        out.has_prompt_audio = true;
    } else if (!out.prompt_text.empty()) {
        throw std::runtime_error("MOSS-TTS-Nano reference_text requires --voice-ref");
    }
    out.generation = generation_options_from_request(request, assets_->config);
    return out;
}

const MossTTSNanoAudioCodes & MossTTSNanoSession::reference_codes_for_request(const MossTTSNanoRequest & request) {
    if (prepared_prompt_audio_.has_value() &&
        prepared_reference_codes_.has_value() &&
        audio_buffers_equal(*prepared_prompt_audio_, request.prompt_audio) &&
        prepared_reference_codes_->codebooks == request.generation.active_codebooks) {
        return *prepared_reference_codes_;
    }
    prepared_prompt_audio_ = request.prompt_audio;
    prepared_reference_codes_ = encode_reference_audio(
        request.prompt_audio,
        request.generation.active_codebooks);
    return *prepared_reference_codes_;
}

}  // namespace engine::models::moss_tts_nano
