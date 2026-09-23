#include "engine/community_models/vieneu_v3_turbo/session.h"

#include "engine/community_models/vieneu_v3_turbo/frame_cap.h"
#include "engine/community_models/vieneu_v3_turbo/orchestration.h"

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

namespace engine::models::vieneu_v3_turbo {
namespace {

using Clock = std::chrono::steady_clock;
constexpr int64_t kDefaultTextChunkSize = 200;
/// A chunk shorter than this joins a neighbour: on its own it reads as a stutter,
/// and short chunks are also the ones that run away (see the babble guard).
constexpr int64_t kDefaultTextChunkMin = 20;
/// Re-generations allowed for a chunk that ran to its ceiling rather than
/// emitting EOS. Two bounds the worst case at three generations.
constexpr int64_t kMaxRerolls = 2;

/// Where the text front end comes from, when the caller wants one. Both are
/// paths: the sea-g2p shared library (empty = look for it by name) and its
/// `sea_g2p.bin` dictionary.
std::filesystem::path option_path(
    const std::unordered_map<std::string, std::string> & options,
    std::initializer_list<std::string_view> keys) {
    if (const auto value = runtime::find_option(options, keys)) {
        return std::filesystem::path(*value);
    }
    return {};
}

// Hand back what the codec encoder just produced. Enrolling a voice costs one
// encoder pass over the clip; a caller that keeps these codes passes them back
// through `reference_codes_file` and never pays for it again - nor has to carry
// a second copy of the encoder to derive them somewhere else.
//
// The payload is exactly what that option reads: one frame per line,
// code_groups integers each. A binary payload would be smaller and would land
// as `<id>.json` full of hex, which is not something this family can be handed
// back - the round trip has to close for the artifact to be worth anything.
void append_reference_codes_artifact(
    runtime::TaskResult & result,
    const Qwen3SpeechCodes & codes,
    int64_t sample_rate) {
    std::ostringstream text;
    for (int64_t frame = 0; frame < codes.frames; ++frame) {
        for (int64_t group = 0; group < codes.code_groups; ++group) {
            if (group > 0) {
                text << ' ';
            }
            text << codes.codes[static_cast<size_t>(frame * codes.code_groups + group)];
        }
        text << '\n';
    }
    result.output_artifacts.push_back(runtime::make_text_artifact(
        runtime::ArtifactKind::AcousticTokens,
        "vieneu_v3_turbo.reference_codes",
        text.str(),
        {
            // So the CLI writes `<id>.txt` verbatim rather than wrapping it.
            {"mime", "text/plain"},
            {"extension", "txt"},
            {"frames", std::to_string(codes.frames)},
            {"code_groups", std::to_string(codes.code_groups)},
            {"sample_rate", std::to_string(sample_rate)},
        }));
}

// `reference_codes`: the same codes inline, whitespace- or comma-separated and
// row-major, for embedders that hold a voice in memory rather than on disk (the
// C API has no way to pass a file that does not exist). `code_groups` values per
// frame; the count must divide evenly.
Qwen3SpeechCodes parse_reference_codes_inline(const std::string & text, int64_t code_groups) {
    Qwen3SpeechCodes out;
    std::string normalized = text;
    std::replace(normalized.begin(), normalized.end(), ',', ' ');
    std::istringstream stream(normalized);
    int32_t value = 0;
    while (stream >> value) {
        out.codes.push_back(value);
    }
    if (code_groups <= 0 || out.codes.empty() || out.codes.size() % static_cast<size_t>(code_groups) != 0) {
        throw std::runtime_error(
            "VieNeu-TTS reference_codes must hold a whole number of frames of " +
            std::to_string(code_groups) + " codes");
    }
    out.code_groups = code_groups;
    out.frames = static_cast<int64_t>(out.codes.size()) / code_groups;
    return out;
}

// `reference_codes_file`: one frame per line, code_groups integers per line (the layout
// `numpy.savetxt(codes, fmt="%d")` writes for the Python engine's ref_codes).
Qwen3SpeechCodes parse_reference_codes_file(const std::string & path) {
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("VieNeu-TTS could not open reference_codes_file: " + path);
    }
    Qwen3SpeechCodes out;
    std::string line;
    int64_t groups = -1;
    while (std::getline(in, line)) {
        std::istringstream row(line);
        std::vector<int32_t> values;
        int32_t v = 0;
        while (row >> v) {
            values.push_back(v);
        }
        if (values.empty()) {
            continue;
        }
        if (groups < 0) {
            groups = static_cast<int64_t>(values.size());
        } else if (static_cast<int64_t>(values.size()) != groups) {
            throw std::runtime_error("VieNeu-TTS reference_codes_file has ragged rows: " + path);
        }
        out.codes.insert(out.codes.end(), values.begin(), values.end());
        ++out.frames;
    }
    if (out.frames == 0) {
        throw std::runtime_error("VieNeu-TTS reference_codes_file is empty: " + path);
    }
    out.code_groups = groups;
    return out;
}

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

// Interleaved stereo to mono, for the measurements that judge a chunk.
std::vector<float> mono_samples(const runtime::AudioBuffer & audio) {
    const int channels = std::max(1, audio.channels);
    std::vector<float> out(audio.samples.size() / static_cast<size_t>(channels), 0.0F);
    for (size_t i = 0; i < out.size(); ++i) {
        float sum = 0.0F;
        for (int c = 0; c < channels; ++c) {
            sum += audio.samples[i * static_cast<size_t>(channels) + static_cast<size_t>(c)];
        }
        out[i] = sum / static_cast<float>(channels);
    }
    return out;
}

runtime::AudioBuffer decode_moss_audio(
    const Qwen3SpeechCodes & codes,
    engine::codecs::MossAudioTokenizerCodecRuntime & decoder) {
    const int64_t frames = codes.frames;
    const int64_t code_groups = codes.code_groups;
    std::vector<std::vector<int32_t>> transposed(static_cast<size_t>(code_groups), std::vector<int32_t>(static_cast<size_t>(frames)));
    for (int64_t f = 0; f < frames; ++f) {
        for (int64_t g = 0; g < code_groups; ++g) {
            transposed[static_cast<size_t>(g)][static_cast<size_t>(f)] = codes.codes[static_cast<size_t>(f * code_groups + g)];
        }
    }
    auto decoded = decoder.decode(engine::codecs::MossAudioTokenizerCodes{frames, std::move(transposed)});
    runtime::AudioBuffer out;
    out.sample_rate = static_cast<int>(decoded.sampling_rate);
    out.channels = static_cast<int>(decoded.channels.size());
    if (decoded.channels.size() >= 2) {
        const auto & left = decoded.channels[0];
        const auto & right = decoded.channels[1];
        out.samples.resize(left.size() * 2);
        for (size_t i = 0; i < left.size(); ++i) {
            out.samples[i * 2] = left[i];
            out.samples[i * 2 + 1] = right[i];
        }
    }
    return out;
}

std::shared_ptr<const VieNeuTTSAssets> require_assets(std::shared_ptr<const VieNeuTTSAssets> assets) {
    if (assets == nullptr) {
        throw std::runtime_error("VieNeu-TTS TTS session requires assets");
    }
    return assets;
}

VieNeuTTSGenerationOptions generation_options_from_request(
    const runtime::TaskRequest & request,
    const VieNeuTTSConfig & config) {
    VieNeuTTSGenerationOptions options;
    options.max_new_tokens = config.max_new_tokens;
    if (const auto value = runtime::parse_int_option(request.options, {"max_tokens"})) {
        if (*value <= 0) {
            throw std::runtime_error("VieNeu-TTS TTS max_tokens must be positive");
        }
        options.max_new_tokens = *value;
    }
    if (const auto value = runtime::find_option(request.options, {"do_sample"})) {
        options.do_sample = runtime::parse_bool_option(*value, "do_sample");
        options.subtalker_do_sample = options.do_sample;   // one switch for VieNeu
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
    if (const auto value = runtime::parse_int_option(request.options, {"repetition_window"})) {
        options.repetition_window = *value;
    }
    if (const auto value = runtime::find_option(request.options, {"frame_cap"})) {
        options.frame_cap = runtime::parse_bool_option(*value, "frame_cap");
    }
    if (const auto value = runtime::parse_int_option(request.options, {"babble_retries"})) {
        options.babble_retries = std::max<int64_t>(0, *value);
    }
    // The acoustic decoder follows the main sampler unless overridden explicitly.
    if (options.subtalker_temperature < 0.0F) options.subtalker_temperature = options.temperature;
    if (options.subtalker_top_k < 0) options.subtalker_top_k = options.top_k;
    if (options.subtalker_top_p < 0.0F) options.subtalker_top_p = options.top_p;

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

uint64_t hash_reference_codes(const std::optional<Qwen3SpeechCodes> & codes) {
    uint64_t hash = 1469598103934665603ull;
    if (!codes.has_value()) {
        return hash;
    }
    hash = fnv1a_mix(hash, &codes->frames, sizeof(codes->frames));
    hash = fnv1a_mix(hash, &codes->code_groups, sizeof(codes->code_groups));
    if (!codes->codes.empty()) {
        hash = fnv1a_mix(hash, codes->codes.data(), codes->codes.size() * sizeof(int32_t));
    }
    return hash;
}

uint64_t hash_speaker_embedding(const std::optional<std::vector<float>> & values) {
    uint64_t hash = 1469598103934665603ull;
    if (!values.has_value()) {
        return hash;
    }
    for (const float value : *values) {
        uint32_t bits = 0;
        std::memcpy(&bits, &value, sizeof(bits));
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
    if (const auto value = runtime::find_option(options.options, {"vieneu_v3_turbo.mem_saver", "mem_saver"})) {
        return runtime::parse_bool_option(*value, "vieneu_v3_turbo.mem_saver");
    }
    return false;
}

std::size_t voice_prompt_cache_slots_from_options(const runtime::SessionOptions & options) {
    constexpr int64_t kDefaultCacheSlots = 1;
    const int64_t slots = runtime::parse_i64_option(options.options, {"vieneu_v3_turbo.voice_prompt_cache_slots"})
        .value_or(kDefaultCacheSlots);
    if (slots < 0) {
        throw std::runtime_error("vieneu_v3_turbo.voice_prompt_cache_slots must be non-negative");
    }
    if (static_cast<std::uint64_t>(slots) > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
        throw std::runtime_error("vieneu_v3_turbo.voice_prompt_cache_slots is too large");
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

bool VieNeuTTSSession::VoicePromptCacheKeyEqual::operator()(
    const VoicePromptCacheKey & lhs,
    const VoicePromptCacheKey & rhs) const noexcept {
    return lhs.reference_text == rhs.reference_text &&
        lhs.mode == rhs.mode &&
        lhs.sample_rate == rhs.sample_rate &&
        lhs.channels == rhs.channels &&
        lhs.sample_count == rhs.sample_count &&
        lhs.sample_hash == rhs.sample_hash &&
        lhs.reference_codes_count == rhs.reference_codes_count &&
        lhs.reference_codes_hash == rhs.reference_codes_hash &&
        lhs.speaker_embedding_count == rhs.speaker_embedding_count &&
        lhs.speaker_embedding_hash == rhs.speaker_embedding_hash;
}

VieNeuTTSSession::VieNeuTTSSession(
    runtime::TaskSpec task,
    runtime::SessionOptions options,
    std::shared_ptr<const VieNeuTTSAssets> assets)
    : RuntimeSessionBase(options),
      task_(task),
      assets_(require_assets(std::move(assets))),
      mem_saver_(mem_saver_from_options(options)),
      text_tokenizer_(assets_),
      talker_(assets_->config.talker),
      voice_prompt_context_(voice_prompt_backend_config(options)),
      voice_prompt_cache_(voice_prompt_cache_slots_from_options(options)) {
    talker_graph_arena_bytes_ = runtime::parse_size_mb_option(
        options.options, {"vieneu_v3_turbo.talker_graph_arena_mb"}, talker_graph_arena_bytes_);
    speech_encoder_graph_arena_bytes_ = runtime::parse_size_mb_option(
        options.options, {"vieneu_v3_turbo.speech_encoder_graph_arena_mb"}, speech_encoder_graph_arena_bytes_);
    speech_decoder_graph_arena_bytes_ = runtime::parse_size_mb_option(
        options.options, {"vieneu_v3_turbo.speech_decoder_graph_arena_mb"}, speech_decoder_graph_arena_bytes_);
    speaker_encoder_graph_arena_bytes_ = runtime::parse_size_mb_option(
        options.options, {"vieneu_v3_turbo.speaker_encoder_graph_arena_mb"}, speaker_encoder_graph_arena_bytes_);
    talker_constant_context_bytes_ = runtime::parse_size_mb_option(
        options.options, {"vieneu_v3_turbo.talker_constant_context_mb"}, talker_constant_context_bytes_);
    code_predictor_constant_context_bytes_ = runtime::parse_size_mb_option(
        options.options, {"vieneu_v3_turbo.code_predictor_constant_context_mb"}, code_predictor_constant_context_bytes_);
    speech_decoder_constant_context_bytes_ = runtime::parse_size_mb_option(
        options.options, {"vieneu_v3_turbo.speech_decoder_constant_context_mb"}, speech_decoder_constant_context_bytes_);
    if (const auto it = options.options.find("vieneu_v3_turbo.weight_type"); it != options.options.end()) {
        const auto storage_type = engine::assets::parse_tensor_storage_type(it->second);
        validate_matmul_weight_storage(storage_type, "vieneu_v3_turbo.weight_type");
        validate_talker_weight_storage(storage_type);
        talker_weight_storage_type_ = storage_type;
    }
    if (const auto it = options.options.find("vieneu_v3_turbo.conv_weight_type"); it != options.options.end()) {
        conv_weight_storage_type_ = engine::assets::parse_tensor_storage_type(it->second);
        validate_conv_weight_storage(conv_weight_storage_type_, "vieneu_v3_turbo.conv_weight_type");
    }
    if (const auto it = options.options.find("vieneu_v3_turbo.talker_weight_type"); it != options.options.end()) {
        talker_weight_storage_type_ = engine::assets::parse_tensor_storage_type(it->second);
        validate_talker_weight_storage(talker_weight_storage_type_);
    }
    if (const auto it = options.options.find("vieneu_v3_turbo.speech_encoder_weight_type"); it != options.options.end()) {
        speech_encoder_weight_storage_type_ = engine::assets::parse_tensor_storage_type(it->second);
        validate_matmul_weight_storage(speech_encoder_weight_storage_type_, "vieneu_v3_turbo.speech_encoder_weight_type");
    }
    if (const auto it = options.options.find("vieneu_v3_turbo.speech_decoder_weight_type"); it != options.options.end()) {
        speech_decoder_weight_storage_type_ = engine::assets::parse_tensor_storage_type(it->second);
        validate_matmul_weight_storage(speech_decoder_weight_storage_type_, "vieneu_v3_turbo.speech_decoder_weight_type");
    }
    for (const auto & [key, _] : options.options) {
        if (key.rfind("vieneu_v3_turbo.", 0) == 0 &&
            key != "vieneu_v3_turbo.g2p_dict" &&
            key != "vieneu_v3_turbo.g2p_library" &&
            key != "vieneu_v3_turbo.talker_graph_arena_mb" &&
            key != "vieneu_v3_turbo.speech_encoder_graph_arena_mb" &&
            key != "vieneu_v3_turbo.speech_decoder_graph_arena_mb" &&
            key != "vieneu_v3_turbo.speaker_encoder_graph_arena_mb" &&
            key != "vieneu_v3_turbo.talker_constant_context_mb" &&
            key != "vieneu_v3_turbo.code_predictor_constant_context_mb" &&
            key != "vieneu_v3_turbo.speech_decoder_constant_context_mb" &&
            key != "vieneu_v3_turbo.weight_type" &&
            key != "vieneu_v3_turbo.conv_weight_type" &&
            key != "vieneu_v3_turbo.talker_weight_type" &&
            key != "vieneu_v3_turbo.speech_encoder_weight_type" &&
            key != "vieneu_v3_turbo.speech_decoder_weight_type" &&
            key != "vieneu_v3_turbo.voice_prompt_cache_slots" &&
            key != "vieneu_v3_turbo.mem_saver") {
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
    // Two different ceilings: `max_new_tokens` is what a request gets by default
    // (300 frames, the Python engine's `max_new_frames`), while the session can
    // serve anything the backbone has positions for — the KV cache grows on
    // demand, so sizing the guard by the position budget costs nothing and lets a
    // caller with its own frame budget ask for more.
    talker_step_ = talker_.create_step_runtime(
        talker_weights_,
        assets_->config.talker.max_position_embeddings,
        std::max(assets_->config.max_new_tokens, assets_->config.talker.max_position_embeddings));
    moss_speech_decoder_ = std::make_unique<engine::codecs::MossAudioTokenizerCodecRuntime>(
        assets_->speech_tokenizer_weights,
        execution_context(),
        assets_->config.speech_tokenizer.num_quantizers,
        engine::codecs::MossAudioTokenizerCodecRuntimeOptions{
            speech_decoder_constant_context_bytes_,
            speech_decoder_graph_arena_bytes_,
            speech_decoder_graph_arena_bytes_,
            false,
        },
        engine::codecs::moss_audio_tokenizer_nano_config());
    moss_speech_decoder_->prepare_decoder();
    // The same MOSS runtime encodes the voice reference into prompt codes.
    moss_speech_decoder_->prepare_encoder();
    if (task_.mode != runtime::RunMode::Offline) {
        throw std::runtime_error("VieNeu TTS currently supports offline sessions");
    }
    if (assets_->config.variant == VieNeuTTSVariant::Base && task_.task != runtime::VoiceTaskKind::Tts) {
        throw std::runtime_error("VieNeu-TTS base TTS model only supports the Tts task");
    }
    // A text front end only when asked for: without it the family behaves exactly
    // as before and `--text` carries phonemes.
    const auto g2p_dictionary = option_path(options.options, {"vieneu_v3_turbo.g2p_dict", "g2p_dict"});
    if (!g2p_dictionary.empty()) {
        text_frontend_ = std::make_unique<TextFrontend>(
            option_path(options.options, {"vieneu_v3_turbo.g2p_library", "g2p_library"}),
            g2p_dictionary);
        debug::log_message(
            debug::LogLevel::Info,
            kFamily,
            "text front end ready: --text may be Vietnamese/English text");
    }
    if (assets_->config.variant == VieNeuTTSVariant::Base) {
        if (assets_->model_weights->has_tensor("speaker_encoder.layer1.0.weight")) {
            speaker_encoder_ = std::make_unique<VieNeuSpeakerEncoderRuntime>(
                assets_,
                voice_prompt_context_,
                speaker_encoder_graph_arena_bytes_,
                conv_weight_storage_type_);
        }
    }
}

std::string VieNeuTTSSession::family() const {
    return "vieneu_v3_turbo";
}

runtime::VoiceTaskKind VieNeuTTSSession::task_kind() const {
    return task_.task;
}

runtime::RunMode VieNeuTTSSession::run_mode() const {
    return task_.mode;
}

void VieNeuTTSSession::prepare(const runtime::SessionPreparationRequest & request) {
    (void) request;
    mark_prepared();
}

runtime::TaskResult VieNeuTTSSession::run(const runtime::TaskRequest & request) {
    require_prepared("VieNeu-TTS TTS run");
    const auto wall_start = Clock::now();
    auto release_talker_cached_step_graph = [&]() {
        if (mem_saver_) {
            const auto release_start = Clock::now();
            const int64_t released_steps = talker_step_->release_cached_step_graph();
            debug::timing_log_scalar(
                "vieneu_v3_turbo.talker.cached_step_release_ms",
                engine::debug::elapsed_ms(release_start, Clock::now()));
            debug::timing_log_scalar("vieneu_v3_turbo.talker.cached_step_released_steps", released_steps);
        }
    };
    // Enrolment without synthesis: encode the clip, hand back the codes, read
    // nothing. Adding a voice needs the codes and not a spoken sentence, and a
    // caller that stops here keeps the encoder pass it already paid for.
    if (const auto encode_only = runtime::find_option(request.options, {"encode_reference_only"});
        encode_only.has_value() && runtime::parse_bool_option(*encode_only, "encode_reference_only")) {
        const auto voice_clone = make_voice_clone_input(request);
        if (!voice_clone.has_value() || voice_clone->reference_audio.samples.empty()) {
            throw std::runtime_error(
                "VieNeu-TTS encode_reference_only needs reference audio (--voice-ref)");
        }
        VieNeuTTSVoiceClonePromptBuilder prompt_builder(
            text_tokenizer_,
            moss_speech_decoder_.get(),
            speaker_encoder_.get(),
            assets_->config.talker.max_position_embeddings);
        const auto & prompt = resolve_voice_prompt(*voice_clone, prompt_builder);
        if (!prompt.reference_codes.has_value()) {
            throw std::runtime_error("VieNeu-TTS encode_reference_only produced no reference codes");
        }
        runtime::TaskResult result;
        append_reference_codes_artifact(
            result, *prompt.reference_codes, assets_->config.speech_tokenizer.output_sample_rate);
        debug::timing_log_scalar("session.wall_ms", engine::debug::elapsed_ms(wall_start, Clock::now()));
        return result;
    }
    const int64_t text_chunk_size =
        engine::text::parse_text_chunk_size_override(request.options).value_or(kDefaultTextChunkSize);
    // Sentence-aware cut of the phoneme string, and the seam type between every
    // pair, so the pauses below can be the ones the model was measured to leave
    // on its own (see orchestration.h).
    const int64_t min_chunk_chars =
        runtime::parse_int_option(request.options, {"text_chunk_min"}).value_or(kDefaultTextChunkMin);
    // With a front end the request carries text; without one it already carries
    // phonemes, which is what the model reads either way.
    std::string phonemes = request.text_input.has_value() ? request.text_input->text : std::string();
    if (text_frontend_ != nullptr && !phonemes.empty()) {
        const auto g2p_start = Clock::now();
        phonemes = text_frontend_->phonemize(phonemes);
        debug::timing_log_scalar("vieneu_v3_turbo.g2p_ms", engine::debug::elapsed_ms(g2p_start, Clock::now()));
    }
    const auto phoneme_chunks = split_phoneme_chunks(phonemes, text_chunk_size, min_chunk_chars);
    if (phoneme_chunks.empty()) {
        throw std::runtime_error("VieNeu-TTS TTS requires non-empty text");
    }
    const VieNeuTTSRequest first_request = make_request(request);
    if (!first_request.voice_clone.has_value()) {
        throw std::runtime_error(
            "VieNeu-TTS TTS requires a voice: --voice-ref, reference_codes_file, or "
            "speaker_embedding_file with x_vector_only_mode=true");
    }
    VieNeuTTSVoiceClonePromptBuilder prompt_builder(
        text_tokenizer_,
        moss_speech_decoder_.get(),
        speaker_encoder_.get(),
        assets_->config.talker.max_position_embeddings);
    double prompt_ms = 0.0;
    double prefill_ms = 0.0;
    double talker_ms = 0.0;
    double decoder_ms = 0.0;
    runtime::AudioBuffer merged_audio;
    const auto prompt_start = Clock::now();
    // One voice per request, and the codec that resolves it must not be busy
    // decoding when a later chunk asks for it.
    const auto & voice_prompt = resolve_voice_prompt(*first_request.voice_clone, prompt_builder);
    prompt_ms += engine::debug::elapsed_ms(prompt_start, Clock::now());
    // Codes the caller passed in are already theirs; only the ones this run
    // derived from a clip are worth handing back.
    const bool derived_reference_codes =
        !first_request.voice_clone->reference_codes.has_value() &&
        !first_request.voice_clone->reference_audio.samples.empty();
    // Mono tail of the previous chunk, for measuring the pause already there.
    std::vector<float> previous_tail;
    const int64_t sample_rate = assets_->config.speech_tokenizer.output_sample_rate;
    for (size_t chunk_index = 0; chunk_index < phoneme_chunks.size(); ++chunk_index) {
        VieNeuTTSRequest qwen_request = first_request;
        qwen_request.text = phoneme_chunks[chunk_index].phonemes;
        const int64_t frame_limit = qwen_request.generation.frame_cap
            ? std::min(qwen_request.generation.max_new_tokens,
                       std::max<int64_t>(1, max_expected_frames(qwen_request.text)))
            : qwen_request.generation.max_new_tokens;
        qwen_request.generation.max_new_tokens = frame_limit;
        const auto prefill_start = Clock::now();
        const auto prefill = prompt_builder.build_prefill(qwen_request, voice_prompt);
        prefill_ms += engine::debug::elapsed_ms(prefill_start, Clock::now());
        const auto talker_start = Clock::now();
        auto codes = talker_step_->generate(
            prefill,
            qwen_request.generation,
            qwen_request.generation.repetition_penalty);
        // Ran to the ceiling: the chunk never emitted EOS, so its tail is gone
        // and nothing downstream can tell. Generating it again is worth the wait
        // — the sampler has moved on, so the retry differs. Greedy decoding is
        // deterministic, so there is nothing to retry there.
        if (qwen_request.generation.do_sample) {
            for (int64_t attempt = 0; attempt < kMaxRerolls && codes.generated_codes.frames >= frame_limit; ++attempt) {
                auto retry_options = qwen_request.generation;
                retry_options.seed = qwen_request.generation.seed + static_cast<uint32_t>(attempt) + 1U;
                auto retry = talker_step_->generate(prefill, retry_options, retry_options.repetition_penalty);
                if (retry.generated_codes.frames > 0 && retry.generated_codes.frames < codes.generated_codes.frames) {
                    codes = std::move(retry);
                }
            }
        }
        talker_ms += engine::debug::elapsed_ms(talker_start, Clock::now());
        // Parity hook: `codes_dump_file=<path>` appends the prompt ids, the reference
        // codes and the generated codes as text so a Python reference run can be
        // compared token-for-token (see tools/community_models/vieneu_v3_turbo).
        if (const auto dump_path = runtime::find_option(request.options, {"codes_dump_file"})) {
            std::ofstream dump(*dump_path, std::ios::app);
            dump << "input_ids";
            for (const auto id : prefill.input_ids) dump << ' ' << id;
            dump << '\n';
            if (prefill.reference_codes.has_value()) {
                const auto & ref = *prefill.reference_codes;
                dump << "reference_codes " << ref.frames << ' ' << ref.code_groups;
                for (const auto code : ref.codes) dump << ' ' << code;
                dump << '\n';
            }
            const auto & gen = codes.generated_codes;
            dump << "generated_codes " << gen.frames << ' ' << gen.code_groups;
            for (const auto code : gen.codes) dump << ' ' << code;
            dump << '\n';
        }
        const auto decoder_start = Clock::now();
        auto chunk_audio = decode_moss_audio(codes.generated_codes, *moss_speech_decoder_);
        decoder_ms += engine::debug::elapsed_ms(decoder_start, Clock::now());

        // The babble guard: a short chunk that kept talking past its text. It
        // cannot be seen before decoding — neither the EOS probability nor the
        // codes warn — so a suspect chunk is generated again and the best
        // attempt kept.
        if (qwen_request.generation.do_sample && qwen_request.generation.babble_retries > 0) {
            auto mono = mono_samples(chunk_audio);
            auto best = babble_suspect(
                mono, sample_rate, qwen_request.text, frame_limit, codes.generated_codes.frames);
            const int64_t retries = best.syllables == 0
                ? std::min(qwen_request.generation.babble_retries, kBabbleMaxRetriesCue)
                : std::min(qwen_request.generation.babble_retries, kBabbleMaxRetries);
            for (int64_t attempt = 0; best.suspect && attempt < retries; ++attempt) {
                auto retry_options = qwen_request.generation;
                retry_options.seed = qwen_request.generation.seed + 1000U + static_cast<uint32_t>(attempt);
                const auto retry_start = Clock::now();
                auto retry = talker_step_->generate(prefill, retry_options, retry_options.repetition_penalty);
                talker_ms += engine::debug::elapsed_ms(retry_start, Clock::now());
                if (retry.generated_codes.frames <= 0) {
                    continue;
                }
                const auto retry_decode_start = Clock::now();
                auto retry_audio = decode_moss_audio(retry.generated_codes, *moss_speech_decoder_);
                decoder_ms += engine::debug::elapsed_ms(retry_decode_start, Clock::now());
                auto retry_mono = mono_samples(retry_audio);
                const auto candidate = babble_suspect(
                    retry_mono, sample_rate, qwen_request.text, frame_limit, retry.generated_codes.frames);
                if (babble_prefer(candidate, best)) {
                    best = candidate;
                    chunk_audio = std::move(retry_audio);
                    mono = std::move(retry_mono);
                }
            }
        }

        // The seam: keep the model's own audio and only pad zeros when the
        // silence already there falls short of the minimum for this boundary.
        auto mono = mono_samples(chunk_audio);
        if (chunk_index > 0) {
            const double pause = gap_pause_sec(phoneme_chunks[chunk_index].gap_before);
            const int64_t pad = pause_pad_samples(previous_tail, mono, sample_rate, pause);
            if (pad > 0) {
                runtime::AudioBuffer silence;
                silence.sample_rate = chunk_audio.sample_rate;
                silence.channels = chunk_audio.channels;
                silence.samples.assign(static_cast<size_t>(pad * std::max(1, chunk_audio.channels)), 0.0F);
                runtime::append_audio_buffer(merged_audio, silence);
            }
        }
        // Up to two seconds is all `pause_pad_samples` can use.
        const size_t keep = std::min<size_t>(mono.size(), static_cast<size_t>(2 * sample_rate));
        previous_tail.assign(mono.end() - static_cast<std::ptrdiff_t>(keep), mono.end());
        runtime::append_audio_buffer(merged_audio, chunk_audio);
    }
    release_talker_cached_step_graph();
    runtime::TaskResult result;
    result.audio_output = std::move(merged_audio);
    if (derived_reference_codes && voice_prompt.reference_codes.has_value()) {
        append_reference_codes_artifact(result, *voice_prompt.reference_codes, sample_rate);
    }
    debug::timing_log_scalar("vieneu_v3_turbo.voice_prompt_ms", prompt_ms);
    debug::timing_log_scalar("vieneu_v3_turbo.prefill_build_ms", prefill_ms);
    debug::timing_log_scalar("vieneu_v3_turbo.talker_ms", talker_ms);
    debug::timing_log_scalar("vieneu_v3_turbo.speech_decoder_ms", decoder_ms);
    debug::timing_log_scalar("session.wall_ms", engine::debug::elapsed_ms(wall_start, Clock::now()));
    return result;
}

VieNeuTTSSession::VoicePromptCacheKey VieNeuTTSSession::voice_prompt_cache_key(
    const Qwen3VoiceCloneInput & input) {
    VoicePromptCacheKey key;
    key.reference_text = input.reference_text;
    key.mode = input.mode;
    key.sample_rate = input.reference_audio.sample_rate;
    key.channels = input.reference_audio.channels;
    key.sample_count = static_cast<uint64_t>(input.reference_audio.samples.size());
    key.sample_hash = hash_audio_samples(input.reference_audio);
    key.reference_codes_count = input.reference_codes.has_value()
        ? static_cast<uint64_t>(input.reference_codes->codes.size())
        : 0;
    key.reference_codes_hash = hash_reference_codes(input.reference_codes);
    key.speaker_embedding_count = input.speaker_embedding.has_value()
        ? static_cast<uint64_t>(input.speaker_embedding->size())
        : 0;
    key.speaker_embedding_hash = hash_speaker_embedding(input.speaker_embedding);
    return key;
}

const Qwen3VoiceClonePrompt & VieNeuTTSSession::resolve_voice_prompt(
    const Qwen3VoiceCloneInput & input,
    const VieNeuTTSVoiceClonePromptBuilder & prompt_builder) {
    VoicePromptCacheKey key = voice_prompt_cache_key(input);
    if (auto * cached = voice_prompt_cache_.find(key)) {
        debug::trace_log_scalar("vieneu_v3_turbo.voice_prompt_cache.hit", 1);
        debug::trace_log_scalar("vieneu_v3_turbo.voice_prompt_cache.slots", static_cast<int64_t>(voice_prompt_cache_.capacity()));
        debug::trace_log_scalar("vieneu_v3_turbo.voice_prompt_cache.entries", static_cast<int64_t>(voice_prompt_cache_.size()));
        debug::trace_log_scalar("vieneu_v3_turbo.voice_prompt_cache.evicted", 0);
        return cached->prompt;
    }

    VoicePromptCacheEntry entry;
    entry.prompt = prompt_builder.build_voice_prompt(input);
    if (voice_prompt_cache_.capacity() == 0) {
        uncached_voice_prompt_ = std::move(entry);
        debug::trace_log_scalar("vieneu_v3_turbo.voice_prompt_cache.hit", 0);
        debug::trace_log_scalar("vieneu_v3_turbo.voice_prompt_cache.slots", 0);
        debug::trace_log_scalar("vieneu_v3_turbo.voice_prompt_cache.entries", 0);
        debug::trace_log_scalar("vieneu_v3_turbo.voice_prompt_cache.evicted", 0);
        return uncached_voice_prompt_->prompt;
    }
    const bool will_evict = voice_prompt_cache_.size() >= voice_prompt_cache_.capacity();
    voice_prompt_cache_.put(std::move(key), std::move(entry));
    auto * cached = voice_prompt_cache_.find(voice_prompt_cache_key(input));
    if (cached == nullptr) {
        throw std::runtime_error("VieNeu-TTS TTS voice prompt cache insert failed");
    }
    debug::trace_log_scalar("vieneu_v3_turbo.voice_prompt_cache.hit", 0);
    debug::trace_log_scalar("vieneu_v3_turbo.voice_prompt_cache.slots", static_cast<int64_t>(voice_prompt_cache_.capacity()));
    debug::trace_log_scalar("vieneu_v3_turbo.voice_prompt_cache.entries", static_cast<int64_t>(voice_prompt_cache_.size()));
    debug::trace_log_scalar("vieneu_v3_turbo.voice_prompt_cache.evicted", will_evict ? 1 : 0);
    return cached->prompt;
}

VieNeuTTSRequest VieNeuTTSSession::make_request(const runtime::TaskRequest & request) const {
    if (!request.text_input.has_value()) {
        throw std::runtime_error("VieNeu-TTS TTS requires text input");
    }
    VieNeuTTSRequest out;
    out.text = request.text_input->text;
    out.language = !request.text_input->language.empty() ? request.text_input->language : "Auto";
    out.generation = generation_options_from_request(request, assets_->config);
    out.voice_clone = make_voice_clone_input(request);
    return out;
}

// The voice a request carries, however it was expressed: a reference clip,
// pre-encoded codes, or the speaker embedding on its own. It sits apart from
// make_request because enrolment needs a voice without a sentence to read.
std::optional<Qwen3VoiceCloneInput> VieNeuTTSSession::make_voice_clone_input(
    const runtime::TaskRequest & request) const {
    if (assets_->config.variant != VieNeuTTSVariant::Base) {
        return std::nullopt;
    }
    const runtime::AudioBuffer * reference_audio = nullptr;
    if (request.voice.has_value()
        && request.voice->speaker.has_value()
        && request.voice->speaker->audio.has_value()) {
        reference_audio = &*request.voice->speaker->audio;
    } else if (request.audio_input.has_value()) {
        reference_audio = &*request.audio_input;
    }
    const auto reference_codes_file = runtime::find_option(request.options, {"reference_codes_file"});
    const auto reference_codes_inline = runtime::find_option(request.options, {"reference_codes"});
    const auto speaker_embedding_file = runtime::find_option(request.options, {"speaker_embedding_file"});
    const auto speaker_embedding_csv = runtime::find_option(request.options, {"speaker_embedding"});
    // A voice is either reference audio, pre-encoded reference codes, or - in
    // x-vector-only mode - the speaker embedding on its own.
    if (reference_audio != nullptr || reference_codes_file.has_value() ||
        reference_codes_inline.has_value() || speaker_embedding_file.has_value() ||
        speaker_embedding_csv.has_value()) {
        Qwen3VoiceCloneInput voice_clone;
        if (reference_audio != nullptr) {
            voice_clone.reference_audio = *reference_audio;
        }
        if (reference_codes_file.has_value()) {
            voice_clone.reference_codes = parse_reference_codes_file(*reference_codes_file);
        } else if (reference_codes_inline.has_value()) {
            voice_clone.reference_codes =
                parse_reference_codes_inline(*reference_codes_inline, assets_->config.talker.num_code_groups);
        }
        if (const auto reference_text = runtime::find_option(
                request.options,
                {"reference_text"})) {
            voice_clone.reference_text = *reference_text;
        }
        if (const auto spk_emb_file = speaker_embedding_file) {
            voice_clone.speaker_embedding = parse_speaker_embedding_file(*spk_emb_file);
        } else if (const auto spk_emb_str = speaker_embedding_csv) {
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
        bool x_vector_only = reference_audio == nullptr && !reference_codes_file.has_value() &&
            !reference_codes_inline.has_value();
        if (const auto value = runtime::find_option(
                request.options,
                {"x_vector_only_mode"})) {
            x_vector_only = runtime::parse_bool_option(*value, "x_vector_only_mode");
        }
        if (!x_vector_only && reference_audio == nullptr && !reference_codes_file.has_value() &&
            !reference_codes_inline.has_value()) {
            throw std::runtime_error(
                "VieNeu-TTS voice cloning needs reference audio (--voice-ref) or "
                "reference_codes_file; pass x_vector_only_mode=true to clone from the "
                "speaker embedding alone");
        }
        if (x_vector_only && !voice_clone.speaker_embedding.has_value()) {
            throw std::runtime_error(
                "VieNeu-TTS x_vector_only_mode needs speaker_embedding_file or speaker_embedding");
        }
        voice_clone.mode = x_vector_only
            ? Qwen3VoiceCloneMode::SpeakerEmbeddingOnly
            : Qwen3VoiceCloneMode::Icl;
        return voice_clone;
    }
    return std::nullopt;
}

}  // namespace engine::models::vieneu_v3_turbo
