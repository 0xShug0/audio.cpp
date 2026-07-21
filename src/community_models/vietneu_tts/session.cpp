#include "engine/community_models/vietneu_tts/session.h"

#include "engine/framework/debug/profiler.h"
#include "engine/framework/runtime/options.h"
#include "engine/framework/text/chunking.h"


#include <algorithm>
#include <chrono>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <utility>
#include <sstream>
#include <fstream>

namespace engine::models::vietneu_tts {
namespace {

using Clock = std::chrono::steady_clock;
constexpr int64_t kDefaultTextChunkSize = 200;

std::vector<float> parse_speaker_embedding_file(const std::string & filepath) {
    std::ifstream ifs(filepath);
    if (!ifs.is_open()) {
        throw std::runtime_error("Failed to open speaker embedding file: " + filepath);
    }
    std::string content;
    if (!std::getline(ifs, content)) {
        throw std::runtime_error("Speaker embedding file is empty: " + filepath);
    }
    std::vector<float> vals;
    std::stringstream ss(content);
    std::string token;
    while (std::getline(ss, token, ',')) {
        try {
            token.erase(0, token.find_first_not_of(" \t\r\n"));
            token.erase(token.find_last_not_of(" \t\r\n") + 1);
            if (!token.empty()) {
                vals.push_back(std::stof(token));
            }
        } catch (const std::exception & e) {
            throw std::runtime_error("Failed to parse float value '" + token + "' in speaker embedding file: " + filepath + " (" + e.what() + ")");
        }
    }
    if (vals.size() != 192) {
        throw std::runtime_error("Speaker embedding file " + filepath + " must contain exactly 192 float values, but got " + std::to_string(vals.size()));
    }
    return vals;
}

runtime::AudioBuffer decode_moss_audio(
    const Qwen3SpeechCodes & codes,
    const engine::models::moss::MossAudioTokenizerDecoder & decoder) {
    const int64_t frames = codes.frames;
    const int64_t code_groups = codes.code_groups;
    std::vector<std::vector<int32_t>> transposed(static_cast<size_t>(code_groups), std::vector<int32_t>(static_cast<size_t>(frames)));
    for (int64_t f = 0; f < frames; ++f) {
        for (int64_t g = 0; g < code_groups; ++g) {
            transposed[static_cast<size_t>(g)][static_cast<size_t>(f)] = codes.codes[static_cast<size_t>(f * code_groups + g)];
        }
    }
    auto stereo = decoder.decode(transposed);
    runtime::AudioBuffer out;
    out.sample_rate = 48000;
    out.channels = 2;
    if (stereo.size() >= 2) {
        const auto & left = stereo[0];
        const auto & right = stereo[1];
        out.samples.resize(left.size() * 2);
        for (size_t i = 0; i < left.size(); ++i) {
            out.samples[i * 2] = left[i];
            out.samples[i * 2 + 1] = right[i];
        }
    }
    return out;
}

std::shared_ptr<const VietneuTTSAssets> require_assets(std::shared_ptr<const VietneuTTSAssets> assets) {
    if (assets == nullptr) {
        throw std::runtime_error("VieNeu-TTS TTS session requires assets");
    }
    return assets;
}

VietneuTTSGenerationOptions generation_options_from_request(
    const runtime::TaskRequest & request,
    const VietneuTTSConfig & config) {
    VietneuTTSGenerationOptions options;
    options.max_new_tokens = config.max_new_tokens;
    if (const auto value = runtime::parse_int_option(request.options, {"max_tokens"})) {
        if (*value <= 0) {
            throw std::runtime_error("VieNeu-TTS TTS max_tokens must be positive");
        }
        options.max_new_tokens = *value;
    }
    if (const auto value = runtime::find_option(request.options, {"do_sample"})) {
        options.do_sample = runtime::parse_bool_option(*value, "do_sample");
    }
    if (const auto value = runtime::find_option(
            request.options,
            {"subtalker_do_sample"})) {
        options.subtalker_do_sample = runtime::parse_bool_option(*value, "subtalker_do_sample");
    }
    if (const auto value = runtime::parse_float_option(request.options, {"temperature"})) {
        options.temperature = *value;
    }
    if (const auto value = runtime::parse_int_option(request.options, {"top_k"})) {
        options.top_k = *value;
    }
    if (const auto value = runtime::parse_float_option(request.options, {"top_p"})) {
        options.top_p = *value;
    }
    if (const auto value = runtime::parse_float_option(
            request.options,
            {"repetition_penalty"})) {
        options.repetition_penalty = *value;
    }
    if (const auto value = runtime::parse_float_option(
            request.options,
            {"subtalker_temperature"})) {
        options.subtalker_temperature = *value;
    }
    if (const auto value = runtime::parse_int_option(
            request.options,
            {"subtalker_top_k"})) {
        options.subtalker_top_k = *value;
    }
    if (const auto value = runtime::parse_float_option(
            request.options,
            {"subtalker_top_p"})) {
        options.subtalker_top_p = *value;
    }
    options.seed = runtime::parse_u32_option(request.options, {"seed"})
        .value_or(runtime::random_u32_seed());
    return options;
}

uint64_t fnv1a_mix(uint64_t hash, const void * data, size_t size) {
    const auto * bytes = static_cast<const uint8_t *>(data);
    for (size_t i = 0; i < size; ++i) {
        hash ^= bytes[i];
        hash *= 1099511628211ull;
    }
    return hash;
}

uint64_t hash_audio_samples(const runtime::AudioBuffer & audio) {
    uint64_t hash = 1469598103934665603ull;
    for (const float sample : audio.samples) {
        uint32_t bits = 0;
        std::memcpy(&bits, &sample, sizeof(bits));
        hash = fnv1a_mix(hash, &bits, sizeof(bits));
    }
    return hash;
}

core::BackendConfig voice_prompt_backend_config(const runtime::SessionOptions & options) {
    core::BackendConfig config = options.backend;
    // Voice-clone prompt codes are discrete argmax outputs; keep this stage on CPU so CUDA TF32 math
    // cannot change the reference prompt that the main talker conditions on.
    config.type = core::BackendType::Cpu;
    return config;
}

bool mem_saver_from_options(const runtime::SessionOptions & options) {
    if (const auto value = runtime::find_option(options.options, {"vietneu_tts.mem_saver", "mem_saver"})) {
        return runtime::parse_bool_option(*value, "vietneu_tts.mem_saver");
    }
    return false;
}

std::size_t voice_prompt_cache_slots_from_options(const runtime::SessionOptions & options) {
    constexpr int64_t kDefaultCacheSlots = 1;
    const int64_t slots = runtime::parse_i64_option(options.options, {"vietneu_tts.voice_prompt_cache_slots"})
        .value_or(kDefaultCacheSlots);
    if (slots < 0) {
        throw std::runtime_error("vietneu_tts.voice_prompt_cache_slots must be non-negative");
    }
    if (static_cast<std::uint64_t>(slots) > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
        throw std::runtime_error("vietneu_tts.voice_prompt_cache_slots is too large");
    }
    return static_cast<std::size_t>(slots);
}

void validate_talker_weight_storage(engine::assets::TensorStorageType storage_type) {
    if (storage_type == engine::assets::TensorStorageType::Native ||
        storage_type == engine::assets::TensorStorageType::F32 ||
        storage_type == engine::assets::TensorStorageType::F16 ||
        storage_type == engine::assets::TensorStorageType::BF16 ||
        storage_type == engine::assets::TensorStorageType::Q8_0) {
        return;
    }
    throw std::runtime_error("VieNeu-TTS TTS talker_weight_type currently supports only native, f32, f16, bf16, and q8_0");
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

}  // namespace

bool VietneuTTSSession::VoicePromptCacheKeyEqual::operator()(
    const VoicePromptCacheKey & lhs,
    const VoicePromptCacheKey & rhs) const noexcept {
    return lhs.reference_text == rhs.reference_text &&
        lhs.mode == rhs.mode &&
        lhs.sample_rate == rhs.sample_rate &&
        lhs.channels == rhs.channels &&
        lhs.sample_count == rhs.sample_count &&
        lhs.sample_hash == rhs.sample_hash;
}

VietneuTTSSession::VietneuTTSSession(
    runtime::TaskSpec task,
    runtime::SessionOptions options,
    std::shared_ptr<const VietneuTTSAssets> assets)
    : RuntimeSessionBase(options),
      task_(task),
      assets_(require_assets(std::move(assets))),
      mem_saver_(mem_saver_from_options(options)),
      text_tokenizer_(assets_),
      talker_(assets_->config.talker),
      voice_prompt_context_(voice_prompt_backend_config(options)),
      voice_prompt_cache_(voice_prompt_cache_slots_from_options(options)) {
    talker_graph_arena_bytes_ = runtime::parse_size_mb_option(
        options.options, {"vietneu_tts.talker_graph_arena_mb"}, talker_graph_arena_bytes_);
    speech_encoder_graph_arena_bytes_ = runtime::parse_size_mb_option(
        options.options, {"vietneu_tts.speech_encoder_graph_arena_mb"}, speech_encoder_graph_arena_bytes_);
    speech_decoder_graph_arena_bytes_ = runtime::parse_size_mb_option(
        options.options, {"vietneu_tts.speech_decoder_graph_arena_mb"}, speech_decoder_graph_arena_bytes_);
    speaker_encoder_graph_arena_bytes_ = runtime::parse_size_mb_option(
        options.options, {"vietneu_tts.speaker_encoder_graph_arena_mb"}, speaker_encoder_graph_arena_bytes_);
    talker_constant_context_bytes_ = runtime::parse_size_mb_option(
        options.options, {"vietneu_tts.talker_constant_context_mb"}, talker_constant_context_bytes_);
    code_predictor_constant_context_bytes_ = runtime::parse_size_mb_option(
        options.options, {"vietneu_tts.code_predictor_constant_context_mb"}, code_predictor_constant_context_bytes_);
    speech_decoder_constant_context_bytes_ = runtime::parse_size_mb_option(
        options.options, {"vietneu_tts.speech_decoder_constant_context_mb"}, speech_decoder_constant_context_bytes_);
    if (const auto it = options.options.find("vietneu_tts.weight_type"); it != options.options.end()) {
        const auto storage_type = engine::assets::parse_tensor_storage_type(it->second);
        validate_matmul_weight_storage(storage_type, "vietneu_tts.weight_type");
        validate_talker_weight_storage(storage_type);
        talker_weight_storage_type_ = storage_type;
    }
    if (const auto it = options.options.find("vietneu_tts.conv_weight_type"); it != options.options.end()) {
        conv_weight_storage_type_ = engine::assets::parse_tensor_storage_type(it->second);
        validate_conv_weight_storage(conv_weight_storage_type_, "vietneu_tts.conv_weight_type");
    }
    if (const auto it = options.options.find("vietneu_tts.talker_weight_type"); it != options.options.end()) {
        talker_weight_storage_type_ = engine::assets::parse_tensor_storage_type(it->second);
        validate_talker_weight_storage(talker_weight_storage_type_);
    }
    if (const auto it = options.options.find("vietneu_tts.speech_encoder_weight_type"); it != options.options.end()) {
        speech_encoder_weight_storage_type_ = engine::assets::parse_tensor_storage_type(it->second);
        validate_matmul_weight_storage(speech_encoder_weight_storage_type_, "vietneu_tts.speech_encoder_weight_type");
    }
    if (const auto it = options.options.find("vietneu_tts.speech_decoder_weight_type"); it != options.options.end()) {
        speech_decoder_weight_storage_type_ = engine::assets::parse_tensor_storage_type(it->second);
        validate_matmul_weight_storage(speech_decoder_weight_storage_type_, "vietneu_tts.speech_decoder_weight_type");
    }
    for (const auto & [key, _] : options.options) {
        if (key.rfind("vietneu_tts.", 0) == 0 &&
            key != "vietneu_tts.talker_graph_arena_mb" &&
            key != "vietneu_tts.speech_encoder_graph_arena_mb" &&
            key != "vietneu_tts.speech_decoder_graph_arena_mb" &&
            key != "vietneu_tts.speaker_encoder_graph_arena_mb" &&
            key != "vietneu_tts.talker_constant_context_mb" &&
            key != "vietneu_tts.code_predictor_constant_context_mb" &&
            key != "vietneu_tts.speech_decoder_constant_context_mb" &&
            key != "vietneu_tts.weight_type" &&
            key != "vietneu_tts.conv_weight_type" &&
            key != "vietneu_tts.talker_weight_type" &&
            key != "vietneu_tts.speech_encoder_weight_type" &&
            key != "vietneu_tts.speech_decoder_weight_type" &&
            key != "vietneu_tts.voice_prompt_cache_slots" &&
            key != "vietneu_tts.mem_saver") {
            throw std::runtime_error("unknown VieNeu-TTS TTS session option: " + key);
        }
    }
    talker_weights_ = talker_.create_weights_runtime(
        assets_,
        options.backend.type,
        options.backend.device,
        std::max(1, options.backend.threads),
        talker_graph_arena_bytes_,
        talker_constant_context_bytes_,
        code_predictor_constant_context_bytes_,
        talker_weight_storage_type_);
    talker_step_ = talker_.create_step_runtime(
        talker_weights_,
        assets_->config.talker.max_position_embeddings,
        assets_->config.max_new_tokens);
    moss_speech_decoder_ = std::make_unique<engine::models::moss::MossAudioTokenizerDecoder>(
        *assets_->speech_tokenizer_weights,
        execution_context(),
        assets_->config.speech_tokenizer.num_quantizers,
        speech_decoder_constant_context_bytes_,
        speech_decoder_graph_arena_bytes_,
        engine::models::moss::moss_audio_tokenizer_nano_config());
    if (task_.mode != runtime::RunMode::Offline) {
        throw std::runtime_error("Vietneu TTS currently supports offline sessions");
    }
    if (assets_->config.variant == VietneuTTSVariant::Base && task_.task != runtime::VoiceTaskKind::Tts) {
        throw std::runtime_error("VieNeu-TTS base TTS model only supports the Tts task");
    }
    if (assets_->config.variant == VietneuTTSVariant::Base) {
        if (assets_->speech_tokenizer_weights->has_tensor("encoder.model.0.weight")) {
            speech_encoder_ = std::make_unique<Qwen3SpeechTokenizerEncoderRuntime>(
                assets_,
                voice_prompt_context_,
                speech_encoder_graph_arena_bytes_,
                speech_encoder_weight_storage_type_,
                conv_weight_storage_type_);
        }
        if (assets_->model_weights->has_tensor("speaker_encoder.layer1.0.weight")) {
            speaker_encoder_ = std::make_unique<VietneuSpeakerEncoderRuntime>(
                assets_,
                voice_prompt_context_,
                speaker_encoder_graph_arena_bytes_,
                conv_weight_storage_type_);
        }
    }
}

std::string VietneuTTSSession::family() const {
    return "vietneu_tts";
}

runtime::VoiceTaskKind VietneuTTSSession::task_kind() const {
    return task_.task;
}

runtime::RunMode VietneuTTSSession::run_mode() const {
    return task_.mode;
}

void VietneuTTSSession::prepare(const runtime::SessionPreparationRequest & request) {
    (void) request;
    mark_prepared();
}

runtime::TaskResult VietneuTTSSession::run(const runtime::TaskRequest & request) {
    require_prepared("VieNeu-TTS TTS run");
    const auto wall_start = Clock::now();
    auto release_talker_cached_step_graph = [&]() {
        if (mem_saver_) {
            const auto release_start = Clock::now();
            const int64_t released_steps = talker_step_->release_cached_step_graph();
            debug::timing_log_scalar(
                "vietneu_tts.talker.cached_step_release_ms",
                engine::debug::elapsed_ms(release_start, Clock::now()));
            debug::timing_log_scalar("vietneu_tts.talker.cached_step_released_steps", released_steps);
        }
    };
    const int64_t text_chunk_size =
        engine::text::parse_text_chunk_size_override(request.options).value_or(kDefaultTextChunkSize);
    const auto text_chunk_mode =
        engine::text::parse_text_chunk_mode_override(request.options).value_or(engine::text::TextChunkMode::Default);
    const auto chunk_requests = runtime::chunk_text_request(request, text_chunk_size, text_chunk_mode);

    const VietneuTTSRequest first_request = make_request(chunk_requests.front());
    if (!first_request.voice_clone.has_value()) {
        throw std::runtime_error("VieNeu-TTS base TTS requires voice clone reference audio");
    }
    VietneuTTSVoiceClonePromptBuilder prompt_builder(
        text_tokenizer_,
        speech_encoder_.get(),
        speaker_encoder_.get(),
        assets_->config.talker.max_position_embeddings);
    double prompt_ms = 0.0;
    double prefill_ms = 0.0;
    double talker_ms = 0.0;
    double decoder_ms = 0.0;
    runtime::AudioBuffer merged_audio;
    for (const auto & chunk_request : chunk_requests) {
        const VietneuTTSRequest qwen_request = make_request(chunk_request);
        const auto prompt_start = Clock::now();
        const auto & voice_prompt = resolve_voice_prompt(*qwen_request.voice_clone, prompt_builder);
        prompt_ms += engine::debug::elapsed_ms(prompt_start, Clock::now());
        const auto prefill_start = Clock::now();
        const auto prefill = prompt_builder.build_prefill(qwen_request, voice_prompt);
        prefill_ms += engine::debug::elapsed_ms(prefill_start, Clock::now());
        const auto talker_start = Clock::now();
        const auto codes = talker_step_->generate(
            prefill,
            qwen_request.generation,
            qwen_request.generation.repetition_penalty);
        talker_ms += engine::debug::elapsed_ms(talker_start, Clock::now());
        const auto decoder_start = Clock::now();
        runtime::append_audio_buffer(
            merged_audio,
            decode_moss_audio(codes.generated_codes, *moss_speech_decoder_));
        decoder_ms += engine::debug::elapsed_ms(decoder_start, Clock::now());
    }
    release_talker_cached_step_graph();
    runtime::TaskResult result;
    result.audio_output = std::move(merged_audio);
    debug::timing_log_scalar("vietneu_tts.voice_prompt_ms", prompt_ms);
    debug::timing_log_scalar("vietneu_tts.prefill_build_ms", prefill_ms);
    debug::timing_log_scalar("vietneu_tts.talker_ms", talker_ms);
    debug::timing_log_scalar("vietneu_tts.speech_decoder_ms", decoder_ms);
    debug::timing_log_scalar("session.wall_ms", engine::debug::elapsed_ms(wall_start, Clock::now()));
    return result;
}

const Qwen3VoiceClonePrompt & VietneuTTSSession::resolve_voice_prompt(
    const Qwen3VoiceCloneInput & input,
    const VietneuTTSVoiceClonePromptBuilder & prompt_builder) {
    const uint64_t sample_count = static_cast<uint64_t>(input.reference_audio.samples.size());
    const uint64_t sample_hash = hash_audio_samples(input.reference_audio);
    VoicePromptCacheKey key;
    key.reference_text = input.reference_text;
    key.mode = input.mode;
    key.sample_rate = input.reference_audio.sample_rate;
    key.channels = input.reference_audio.channels;
    key.sample_count = sample_count;
    key.sample_hash = sample_hash;
    if (auto * cached = voice_prompt_cache_.find(key)) {
        debug::trace_log_scalar("vietneu_tts.voice_prompt_cache.hit", 1);
        debug::trace_log_scalar("vietneu_tts.voice_prompt_cache.slots", static_cast<int64_t>(voice_prompt_cache_.capacity()));
        debug::trace_log_scalar("vietneu_tts.voice_prompt_cache.entries", static_cast<int64_t>(voice_prompt_cache_.size()));
        debug::trace_log_scalar("vietneu_tts.voice_prompt_cache.evicted", 0);
        return cached->prompt;
    }

    VoicePromptCacheEntry entry;
    entry.prompt = prompt_builder.build_voice_prompt(input);
    if (voice_prompt_cache_.capacity() == 0) {
        uncached_voice_prompt_ = std::move(entry);
        debug::trace_log_scalar("vietneu_tts.voice_prompt_cache.hit", 0);
        debug::trace_log_scalar("vietneu_tts.voice_prompt_cache.slots", 0);
        debug::trace_log_scalar("vietneu_tts.voice_prompt_cache.entries", 0);
        debug::trace_log_scalar("vietneu_tts.voice_prompt_cache.evicted", 0);
        return uncached_voice_prompt_->prompt;
    }
    const bool will_evict = voice_prompt_cache_.size() >= voice_prompt_cache_.capacity();
    voice_prompt_cache_.put(std::move(key), std::move(entry));
    auto * cached = voice_prompt_cache_.find(VoicePromptCacheKey{
        input.reference_text,
        input.mode,
        input.reference_audio.sample_rate,
        input.reference_audio.channels,
        sample_count,
        sample_hash,
    });
    if (cached == nullptr) {
        throw std::runtime_error("VieNeu-TTS TTS voice prompt cache insert failed");
    }
    debug::trace_log_scalar("vietneu_tts.voice_prompt_cache.hit", 0);
    debug::trace_log_scalar("vietneu_tts.voice_prompt_cache.slots", static_cast<int64_t>(voice_prompt_cache_.capacity()));
    debug::trace_log_scalar("vietneu_tts.voice_prompt_cache.entries", static_cast<int64_t>(voice_prompt_cache_.size()));
    debug::trace_log_scalar("vietneu_tts.voice_prompt_cache.evicted", will_evict ? 1 : 0);
    return cached->prompt;
}

VietneuTTSRequest VietneuTTSSession::make_request(const runtime::TaskRequest & request) const {
    if (!request.text_input.has_value()) {
        throw std::runtime_error("VieNeu-TTS TTS requires text input");
    }
    VietneuTTSRequest out;
    out.text = request.text_input->text;
    out.language = !request.text_input->language.empty() ? request.text_input->language : "Auto";
    out.generation = generation_options_from_request(request, assets_->config);
    if (assets_->config.variant == VietneuTTSVariant::Base) {
        const runtime::AudioBuffer * reference_audio = nullptr;
        if (request.voice.has_value()
            && request.voice->speaker.has_value()
            && request.voice->speaker->audio.has_value()) {
            reference_audio = &*request.voice->speaker->audio;
        } else if (request.audio_input.has_value()) {
            reference_audio = &*request.audio_input;
        }
        if (reference_audio != nullptr) {
            Qwen3VoiceCloneInput voice_clone;
            voice_clone.reference_audio = *reference_audio;
            if (const auto reference_text = runtime::find_option(
                    request.options,
                    {"reference_text"})) {
                voice_clone.reference_text = *reference_text;
            }
            if (const auto spk_emb_file = runtime::find_option(request.options, {"speaker_embedding_file"})) {
                voice_clone.speaker_embedding = parse_speaker_embedding_file(*spk_emb_file);
            } else if (const auto spk_emb_str = runtime::find_option(request.options, {"speaker_embedding"})) {
                std::vector<float> vals;
                std::stringstream ss(*spk_emb_str);
                std::string token;
                while (std::getline(ss, token, ',')) {
                    try {
                        token.erase(0, token.find_first_not_of(" \t\r\n"));
                        token.erase(token.find_last_not_of(" \t\r\n") + 1);
                        if (!token.empty()) {
                            vals.push_back(std::stof(token));
                        }
                    } catch (const std::exception & e) {
                        throw std::runtime_error("Failed to parse float value '" + token + "' in speaker_embedding option: " + e.what());
                    }
                }
                if (vals.size() != 192) {
                    throw std::runtime_error("speaker_embedding option must contain exactly 192 comma-separated float values, but got " + std::to_string(vals.size()));
                }
                voice_clone.speaker_embedding = vals;
            }
            bool x_vector_only = false;
            if (const auto value = runtime::find_option(
                    request.options,
                    {"x_vector_only_mode"})) {
                x_vector_only = runtime::parse_bool_option(*value, "x_vector_only_mode");
            }
            voice_clone.mode = x_vector_only
                ? Qwen3VoiceCloneMode::SpeakerEmbeddingOnly
                : Qwen3VoiceCloneMode::Icl;
            out.voice_clone = std::move(voice_clone);
        }
    }
    return out;
}

}  // namespace engine::models::vietneu_tts
