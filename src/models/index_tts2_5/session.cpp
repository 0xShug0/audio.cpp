#include "engine/models/index_tts2_5/session.h"

#include "engine/framework/debug/profiler.h"
#include "engine/framework/io/text.h"
#include "engine/framework/runtime/options.h"
#include "engine/framework/text/chunking.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace engine::models::index_tts2_5 {
namespace {

using Clock = std::chrono::steady_clock;
constexpr int64_t kConditionDim = 512;
constexpr int64_t kGptDim = 1280;
constexpr int64_t kStyleDim = 192;
constexpr int64_t kEmotionCount = 8;
constexpr int64_t kDiffusionSteps = 25;
constexpr float kInferenceCfgRate = 0.7F;

std::shared_ptr<const IndexTTS25Assets> require_assets(std::shared_ptr<const IndexTTS25Assets> assets) {
    if (assets == nullptr) {
        throw std::runtime_error("IndexTTS2.5 session requires assets");
    }
    return assets;
}

void validate_matmul_weight_storage(engine::assets::TensorStorageType storage_type, const char * option_name) {
    if (storage_type == engine::assets::TensorStorageType::Native ||
        storage_type == engine::assets::TensorStorageType::F32 ||
        storage_type == engine::assets::TensorStorageType::F16 ||
        storage_type == engine::assets::TensorStorageType::BF16 ||
        storage_type == engine::assets::TensorStorageType::Q8_0) {
        return;
    }
    throw std::runtime_error(std::string(option_name) + " supports only native, f32, f16, bf16, and q8_0");
}

void validate_conv_weight_storage(engine::assets::TensorStorageType storage_type, const char * option_name) {
    if (storage_type == engine::assets::TensorStorageType::Native ||
        storage_type == engine::assets::TensorStorageType::F32 ||
        storage_type == engine::assets::TensorStorageType::F16) {
        return;
    }
    throw std::runtime_error(std::string(option_name) + " supports only native, f32, and f16");
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

IndexTTS25AudioIdentity audio_identity(const runtime::AudioBuffer & audio) {
    return {
        audio.sample_rate,
        audio.channels,
        static_cast<uint64_t>(audio.samples.size()),
        hash_audio_samples(audio),
    };
}

bool same_identity(const IndexTTS25AudioIdentity & lhs, const IndexTTS25AudioIdentity & rhs) {
    return lhs.sample_rate == rhs.sample_rate &&
        lhs.channels == rhs.channels &&
        lhs.sample_count == rhs.sample_count &&
        lhs.sample_hash == rhs.sample_hash;
}

std::size_t resolve_cache_slots(
    const runtime::SessionOptions & options,
    std::initializer_list<std::string_view> keys,
    const char * option_name) {
    constexpr int64_t kDefaultCacheSlots = 1;
    const int64_t slots = runtime::parse_i64_option(options.options, keys)
        .value_or(kDefaultCacheSlots);
    if (slots < 0) {
        throw std::runtime_error(std::string(option_name) + " must be non-negative");
    }
    if (static_cast<std::uint64_t>(slots) > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
        throw std::runtime_error(std::string(option_name) + " is too large");
    }
    return static_cast<std::size_t>(slots);
}

std::vector<float> channel_first_to_time_major(
    const std::vector<float> & values,
    int64_t channels,
    int64_t frames) {
    if (static_cast<int64_t>(values.size()) != channels * frames) {
        throw std::runtime_error("IndexTTS2.5 channel-first tensor size mismatch");
    }
    std::vector<float> out(static_cast<size_t>(frames * channels));
    for (int64_t frame = 0; frame < frames; ++frame) {
        for (int64_t channel = 0; channel < channels; ++channel) {
            out[static_cast<size_t>(frame * channels + channel)] =
                values[static_cast<size_t>(channel * frames + frame)];
        }
    }
    return out;
}

std::vector<float> concat_conditions(
    const IndexTTS25S2MelSequence & prompt,
    const IndexTTS25S2MelSequence & generated) {
    if (prompt.dims != kConditionDim || generated.dims != kConditionDim) {
        throw std::runtime_error("IndexTTS2.5 condition dimension mismatch");
    }
    std::vector<float> out;
    out.reserve(prompt.values.size() + generated.values.size());
    out.insert(out.end(), prompt.values.begin(), prompt.values.end());
    out.insert(out.end(), generated.values.begin(), generated.values.end());
    return out;
}

void append_silence(runtime::AudioBuffer & audio, int ms) {
    if (ms <= 0 || audio.sample_rate <= 0 || audio.channels <= 0) {
        return;
    }
    const int64_t samples = static_cast<int64_t>(audio.sample_rate) * ms / 1000;
    audio.samples.insert(
        audio.samples.end(),
        static_cast<size_t>(samples * audio.channels),
        0.0F);
}

std::vector<float> scaled_emotion_weights(const std::vector<float> & values, float alpha) {
    if (static_cast<int64_t>(values.size()) != kEmotionCount) {
        throw std::runtime_error("IndexTTS2.5 emotion vector must contain exactly 8 values");
    }
    std::vector<float> out = values;
    const float scale = std::clamp(alpha, 0.0F, 1.0F);
    if (scale != 1.0F) {
        for (float & value : out) {
            value = static_cast<float>(static_cast<int>(value * scale * 10000.0F)) / 10000.0F;
        }
    }
    return out;
}

bool mem_saver_from_options(const runtime::SessionOptions & options) {
    if (const auto value = runtime::find_option(options.options, {"index_tts2_5.mem_saver", "mem_saver"})) {
        return runtime::parse_bool_option(*value, "index_tts2_5.mem_saver");
    }
    return false;
}

// Debug intermediate dumps, enabled by setting INDEXTTS25_DUMP_DIR. Files are
// written as NPY v1.0 so they can be compared against the official Python
// golden tensors with numpy directly.
std::string dump_dir_from_env() {
    const char * dir = std::getenv("INDEXTTS25_DUMP_DIR");
    if (dir == nullptr || *dir == '\0') {
        return {};
    }
    return std::string(dir);
}

void write_npy(
    const std::filesystem::path & path,
    const char * descr,
    const std::vector<int64_t> & shape,
    const void * data,
    size_t byte_count) {
    std::string header = "{'descr': '";
    header += descr;
    header += "', 'fortran_order': False, 'shape': (";
    for (const int64_t dim : shape) {
        header += std::to_string(dim);
        header += ", ";
    }
    header += "), }";
    const size_t prefix = 10;  // magic(6) + version(2) + header_len(2)
    const size_t total = prefix + header.size() + 1;
    const size_t padded = (total + 63) / 64 * 64;
    header.append(padded - prefix - header.size() - 1, ' ');
    header.push_back('\n');
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) {
        throw std::runtime_error("IndexTTS2.5 failed to open dump file: " + path.string());
    }
    out.write("\x93NUMPY\x01\x00", 8);
    const uint16_t header_len = static_cast<uint16_t>(header.size());
    out.write(reinterpret_cast<const char *>(&header_len), sizeof(header_len));
    out.write(header.data(), static_cast<std::streamsize>(header.size()));
    out.write(static_cast<const char *>(data), static_cast<std::streamsize>(byte_count));
}

void dump_f32(
    const std::string & dir,
    const std::string & name,
    std::vector<int64_t> shape,
    const std::vector<float> & values) {
    if (dir.empty()) {
        return;
    }
    int64_t elements = 1;
    for (const int64_t dim : shape) {
        elements *= dim;
    }
    if (elements != static_cast<int64_t>(values.size())) {
        throw std::runtime_error("IndexTTS2.5 dump shape does not match value count: " + name);
    }
    write_npy(std::filesystem::path(dir) / (name + ".npy"), "<f4", shape, values.data(), values.size() * sizeof(float));
}

void dump_i32(
    const std::string & dir,
    const std::string & name,
    std::vector<int64_t> shape,
    const std::vector<int32_t> & values) {
    if (dir.empty()) {
        return;
    }
    int64_t elements = 1;
    for (const int64_t dim : shape) {
        elements *= dim;
    }
    if (elements != static_cast<int64_t>(values.size())) {
        throw std::runtime_error("IndexTTS2.5 dump shape does not match value count: " + name);
    }
    write_npy(std::filesystem::path(dir) / (name + ".npy"), "<i4", shape, values.data(), values.size() * sizeof(int32_t));
}

}  // namespace

IndexTTS25Session::IndexTTS25Session(
    runtime::TaskSpec task,
    runtime::SessionOptions options,
    std::shared_ptr<const IndexTTS25Assets> assets)
    : RuntimeSessionBase(options),
      task_(task),
      assets_(require_assets(std::move(assets))),
      tokenizer_(assets_),
      speaker_cache_(resolve_cache_slots(this->options(), {"index_tts2_5.speaker_cache_slots"}, "index_tts2_5.speaker_cache_slots")),
      emotion_cache_(resolve_cache_slots(this->options(), {"index_tts2_5.emotion_cache_slots"}, "index_tts2_5.emotion_cache_slots")),
      emotion_text_weights_cache_(resolve_cache_slots(this->options(), {"index_tts2_5.emotion_text_cache_slots"}, "index_tts2_5.emotion_text_cache_slots")) {
    gpt_graph_arena_bytes_ = runtime::parse_size_mb_option(
        options.options, {"index_tts2_5.gpt_graph_arena_mb"}, gpt_graph_arena_bytes_);
    s2mel_graph_arena_bytes_ = runtime::parse_size_mb_option(
        options.options, {"index_tts2_5.s2mel_graph_arena_mb"}, s2mel_graph_arena_bytes_);
    reference_graph_arena_bytes_ = runtime::parse_size_mb_option(
        options.options, {"index_tts2_5.reference_graph_arena_mb"}, reference_graph_arena_bytes_);
    emotion_text_prefill_graph_arena_bytes_ = runtime::parse_size_mb_option(
        options.options, {"index_tts2_5.emotion_text_prefill_graph_arena_mb"}, emotion_text_prefill_graph_arena_bytes_);
    emotion_text_decode_graph_arena_bytes_ = runtime::parse_size_mb_option(
        options.options, {"index_tts2_5.emotion_text_decode_graph_arena_mb"}, emotion_text_decode_graph_arena_bytes_);
    weight_context_bytes_ = runtime::parse_size_mb_option(
        options.options, {"index_tts2_5.weight_context_mb"}, weight_context_bytes_);
    if (const auto value = runtime::parse_int_option(options.options, {"index_tts2_5.emotion_text_max_new_tokens"})) {
        if (*value <= 0) {
            throw std::runtime_error("index_tts2_5.emotion_text_max_new_tokens must be positive");
        }
        emotion_text_max_new_tokens_ = *value;
    }
    if (const auto it = options.options.find("index_tts2_5.weight_type"); it != options.options.end()) {
        matmul_weight_storage_type_ = engine::assets::parse_tensor_storage_type(it->second);
        validate_matmul_weight_storage(matmul_weight_storage_type_, "index_tts2_5.weight_type");
    }
    if (const auto it = options.options.find("index_tts2_5.conv_weight_type"); it != options.options.end()) {
        conv_weight_storage_type_ = engine::assets::parse_tensor_storage_type(it->second);
        validate_conv_weight_storage(conv_weight_storage_type_, "index_tts2_5.conv_weight_type");
    }
    mem_saver_ = mem_saver_from_options(options);
    for (const auto & [key, _] : options.options) {
        if (key.rfind("index_tts2_5.", 0) == 0 &&
            key != "index_tts2_5.gpt_graph_arena_mb" &&
            key != "index_tts2_5.s2mel_graph_arena_mb" &&
            key != "index_tts2_5.reference_graph_arena_mb" &&
            key != "index_tts2_5.emotion_text_prefill_graph_arena_mb" &&
            key != "index_tts2_5.emotion_text_decode_graph_arena_mb" &&
            key != "index_tts2_5.emotion_text_max_new_tokens" &&
            key != "index_tts2_5.weight_context_mb" &&
            key != "index_tts2_5.weight_type" &&
            key != "index_tts2_5.conv_weight_type" &&
            key != "index_tts2_5.mem_saver" &&
            key != "index_tts2_5.speaker_cache_slots" &&
            key != "index_tts2_5.emotion_cache_slots" &&
            key != "index_tts2_5.emotion_text_cache_slots" &&
            key != "index_tts2_5.gpt.cuda_sampling_policy" &&
            key != "index_tts2_5.s2mel.cuda_sampling_policy") {
            throw std::runtime_error("unknown IndexTTS2.5 session option: " + key);
        }
    }
    if (task_.mode != runtime::RunMode::Offline ||
        (task_.task != runtime::VoiceTaskKind::Tts && task_.task != runtime::VoiceTaskKind::VoiceCloning)) {
        throw std::runtime_error("IndexTTS2.5 currently supports offline TTS and voice-cloning sessions");
    }

    semantic_encoder_ = std::make_unique<IndexTTS25Wav2Vec2BertRuntime>(
        assets_,
        execution_context(),
        reference_graph_arena_bytes_,
        weight_context_bytes_,
        matmul_weight_storage_type_,
        conv_weight_storage_type_);
    semantic_codec_ = std::make_unique<IndexTTS25SemanticCodecRuntime>(
        assets_,
        execution_context(),
        reference_graph_arena_bytes_,
        weight_context_bytes_,
        matmul_weight_storage_type_,
        conv_weight_storage_type_);
    style_encoder_ = std::make_unique<IndexTTS25StyleEncoder>(
        assets_,
        options.backend,
        conv_weight_storage_type_);
    gpt_ = std::make_unique<IndexTTS25GptRuntime>(
        assets_,
        execution_context(),
        gpt_graph_arena_bytes_,
        weight_context_bytes_,
        matmul_weight_storage_type_,
        conv_weight_storage_type_);
    s2mel_ = std::make_unique<IndexTTS25S2MelRuntime>(
        assets_,
        execution_context(),
        s2mel_graph_arena_bytes_,
        weight_context_bytes_,
        matmul_weight_storage_type_,
        conv_weight_storage_type_);
    vocoder_ = std::make_unique<IndexTTS25BigVganVocoder>(
        assets_,
        options.backend,
        conv_weight_storage_type_);
    qwen_emotion_ = std::make_unique<IndexTTS25QwenEmotionRuntime>(
        assets_,
        execution_context(),
        emotion_text_prefill_graph_arena_bytes_,
        emotion_text_decode_graph_arena_bytes_,
        weight_context_bytes_,
        matmul_weight_storage_type_);
    int64_t matrix_rows = 0;
    for (const int64_t count : assets_->config.emo_num) {
        matrix_rows += count;
    }
    speaker_matrix_ = assets_->speaker_matrix->require_f32("tensor", {matrix_rows, kStyleDim});
    emotion_matrix_ = assets_->emotion_matrix->require_f32("tensor", {matrix_rows, kGptDim});
    assets_->speaker_matrix->release_storage();
    assets_->emotion_matrix->release_storage();
}

bool IndexTTS25Session::AudioIdentityEqual::operator()(
    const IndexTTS25AudioIdentity & lhs,
    const IndexTTS25AudioIdentity & rhs) const {
    return same_identity(lhs, rhs);
}

std::string IndexTTS25Session::family() const {
    return "index_tts2_5";
}

runtime::VoiceTaskKind IndexTTS25Session::task_kind() const {
    return task_.task;
}

runtime::RunMode IndexTTS25Session::run_mode() const {
    return task_.mode;
}

void IndexTTS25Session::prepare(const runtime::SessionPreparationRequest & request) {
    if (request.text.has_value() || runtime::find_option(request.options, {"text", "prompt"}).has_value()) {
        std::string text;
        if (request.text.has_value()) {
            text = request.text->text;
        } else if (const auto value = runtime::find_option(request.options, {"text", "prompt"})) {
            text = *value;
        }
        text = engine::io::trim_ascii_whitespace(text);
        if (text.empty()) {
            throw std::runtime_error("IndexTTS2.5 request requires text_input or text option");
        }
        IndexTTS25GenerationOptions generation;
        if (const auto value = runtime::parse_int_option(request.options, {"max_tokens"})) {
            if (*value <= 0) {
                throw std::runtime_error("IndexTTS2.5 max_tokens must be positive");
            }
            generation.max_mel_tokens = *value;
        }
        if (const auto value = runtime::parse_int_option(request.options, {"num_beams"})) {
            if (*value <= 0) {
                throw std::runtime_error("IndexTTS2.5 num_beams must be positive");
            }
            generation.num_beams = *value;
        }
        std::string lang;
        if (const auto value = runtime::find_option(request.options, {"lang"})) {
            lang = normalize_index_tts2_5_lang(*value);
        }
        const auto text_encoding = tokenizer_.encode_for_inference(
            text,
            IndexTTS25Request{}.max_text_tokens_per_segment,
            lang);
        for (const auto & segment : text_encoding.segment_token_ids) {
            gpt_->prepare_generation(
                static_cast<int64_t>(align_index_tts2_5_gpt_text_tokens(segment).size()),
                generation.max_mel_tokens,
                generation.num_beams);
        }
    }
    mark_prepared();
}

const IndexTTS25Session::SpeakerState & IndexTTS25Session::resolve_speaker_state(const runtime::AudioBuffer & audio) {
    const auto identity = audio_identity(audio);
    if (const auto * cached = speaker_cache_.find(identity)) {
        debug::trace_log_scalar("index_tts2_5.speaker_cache.hit", 1);
        debug::trace_log_scalar("index_tts2_5.speaker_cache.slots", static_cast<int64_t>(speaker_cache_.capacity()));
        debug::trace_log_scalar("index_tts2_5.speaker_cache.entries", static_cast<int64_t>(speaker_cache_.size()));
        debug::trace_log_scalar("index_tts2_5.speaker_cache.evicted", 0);
        return *cached;
    }
    const bool will_evict = speaker_cache_.capacity() > 0 && speaker_cache_.size() >= speaker_cache_.capacity();
    const auto start = Clock::now();
    const auto prepared = prepare_index_tts2_5_reference_audio(
        audio.samples,
        audio.sample_rate,
        audio.channels,
        assets_->config.s2mel,
        static_cast<size_t>(std::max(1, options().backend.threads)),
        true);
    semantic_encoder_->prepare(prepared.semantic_features.frames);
    auto semantic = semantic_encoder_->encode(prepared.semantic_features);
    // Official 2.5 regulates the raw (normalized) w2v-bert semantic directly;
    // the semantic codec is only used to decode generated codes.
    debug::trace_log_scalar("index_tts2_5.s2mel.reference_mel_frames", static_cast<double>(prepared.mel.frames));
    s2mel_->prepare_length_regulator(semantic.frames, prepared.mel.frames);
    auto prompt_condition = s2mel_->regulate_length(
        semantic.values,
        semantic.frames,
        prepared.mel.frames);

    SpeakerState state;
    state.identity = identity;
    state.semantic = std::move(semantic);
    state.reference_mel = prepared.mel;
    state.style = style_encoder_->embed_fbank(
        prepared.campplus_fbank.values,
        prepared.campplus_fbank.frames,
        prepared.campplus_fbank.dims);
    state.prompt_condition = std::move(prompt_condition);
    if (speaker_cache_.capacity() == 0) {
        uncached_speaker_state_ = std::move(state);
    } else {
        speaker_cache_.put(identity, std::move(state));
    }
    if (mem_saver_) {
        semantic_encoder_->release_graph();
        semantic_codec_->release_graphs();
        s2mel_->release_pre_cfm_graphs();
        style_encoder_->release_graph();
    }
    debug::trace_log_scalar("index_tts2_5.speaker_cache.hit", 0);
    debug::trace_log_scalar("index_tts2_5.speaker_cache.slots", static_cast<int64_t>(speaker_cache_.capacity()));
    debug::trace_log_scalar("index_tts2_5.speaker_cache.entries", static_cast<int64_t>(speaker_cache_.size()));
    debug::trace_log_scalar("index_tts2_5.speaker_cache.evicted", will_evict ? 1 : 0);
    debug::timing_log_scalar("index_tts2_5.speaker_state_ms", engine::debug::elapsed_ms(start));
    if (speaker_cache_.capacity() == 0) {
        return *uncached_speaker_state_;
    }
    const auto * cached = speaker_cache_.find(identity);
    if (cached == nullptr) {
        throw std::runtime_error("IndexTTS2.5 speaker cache insert failed");
    }
    return *cached;
}

const IndexTTS25Session::EmotionState & IndexTTS25Session::resolve_emotion_state(const runtime::AudioBuffer & audio) {
    const auto identity = audio_identity(audio);
    if (const auto * cached = emotion_cache_.find(identity)) {
        debug::trace_log_scalar("index_tts2_5.emotion_cache.hit", 1);
        debug::trace_log_scalar("index_tts2_5.emotion_cache.slots", static_cast<int64_t>(emotion_cache_.capacity()));
        debug::trace_log_scalar("index_tts2_5.emotion_cache.entries", static_cast<int64_t>(emotion_cache_.size()));
        debug::trace_log_scalar("index_tts2_5.emotion_cache.evicted", 0);
        return *cached;
    }
    const bool will_evict = emotion_cache_.capacity() > 0 && emotion_cache_.size() >= emotion_cache_.capacity();
    const auto start = Clock::now();
    const auto prepared = prepare_index_tts2_5_reference_audio(
        audio.samples,
        audio.sample_rate,
        audio.channels,
        assets_->config.s2mel,
        static_cast<size_t>(std::max(1, options().backend.threads)),
        false);
    semantic_encoder_->prepare(prepared.semantic_features.frames);
    EmotionState state;
    state.identity = identity;
    state.semantic = semantic_encoder_->encode(prepared.semantic_features);
    if (emotion_cache_.capacity() == 0) {
        uncached_emotion_state_ = std::move(state);
    } else {
        emotion_cache_.put(identity, std::move(state));
    }
    if (mem_saver_) {
        semantic_encoder_->release_graph();
    }
    debug::trace_log_scalar("index_tts2_5.emotion_cache.hit", 0);
    debug::trace_log_scalar("index_tts2_5.emotion_cache.slots", static_cast<int64_t>(emotion_cache_.capacity()));
    debug::trace_log_scalar("index_tts2_5.emotion_cache.entries", static_cast<int64_t>(emotion_cache_.size()));
    debug::trace_log_scalar("index_tts2_5.emotion_cache.evicted", will_evict ? 1 : 0);
    debug::timing_log_scalar("index_tts2_5.emotion_state_ms", engine::debug::elapsed_ms(start));
    if (emotion_cache_.capacity() == 0) {
        return *uncached_emotion_state_;
    }
    const auto * cached = emotion_cache_.find(identity);
    if (cached == nullptr) {
        throw std::runtime_error("IndexTTS2.5 emotion cache insert failed");
    }
    return *cached;
}

std::vector<float> IndexTTS25Session::explicit_emotion_matrix_vector(
    const std::vector<float> & emotion_weights,
    const IndexTTS25StyleEmbedding & style,
    bool use_random,
    uint32_t seed) const {
    if (static_cast<int64_t>(emotion_weights.size()) != kEmotionCount ||
        style.dims != kStyleDim ||
        static_cast<int64_t>(style.values.size()) != kStyleDim) {
        throw std::runtime_error("IndexTTS2.5 explicit emotion matrix input shape mismatch");
    }
    std::vector<float> out(static_cast<size_t>(kGptDim), 0.0F);
    int64_t row_offset = 0;
    std::mt19937 rng(seed);
    for (int64_t emotion = 0; emotion < kEmotionCount; ++emotion) {
        const int64_t rows = assets_->config.emo_num[static_cast<size_t>(emotion)];
        int64_t best_row = 0;
        if (rows <= 0) {
            throw std::runtime_error("IndexTTS2.5 emo_num contains non-positive group size");
        }
        if (use_random) {
            std::uniform_int_distribution<int64_t> distribution(0, rows - 1);
            best_row = distribution(rng);
        } else {
            float best_score = -std::numeric_limits<float>::infinity();
            for (int64_t row = 0; row < rows; ++row) {
                const float * matrix_row = speaker_matrix_.data() + static_cast<std::ptrdiff_t>((row_offset + row) * kStyleDim);
                float dot = 0.0F;
                float lhs_norm = 0.0F;
                float rhs_norm = 0.0F;
                for (int64_t dim = 0; dim < kStyleDim; ++dim) {
                    const float lhs = style.values[static_cast<size_t>(dim)];
                    const float rhs = matrix_row[dim];
                    dot += lhs * rhs;
                    lhs_norm += lhs * lhs;
                    rhs_norm += rhs * rhs;
                }
                const float score = dot / (std::sqrt(lhs_norm) * std::sqrt(rhs_norm) + 1.0e-12F);
                if (score > best_score) {
                    best_score = score;
                    best_row = row;
                }
            }
        }
        const float * emotion_row = emotion_matrix_.data() + static_cast<std::ptrdiff_t>((row_offset + best_row) * kGptDim);
        for (int64_t dim = 0; dim < kGptDim; ++dim) {
            out[static_cast<size_t>(dim)] += emotion_weights[static_cast<size_t>(emotion)] * emotion_row[dim];
        }
        row_offset += rows;
    }
    return out;
}

std::vector<float> IndexTTS25Session::resolve_emotion_vector(
    const IndexTTS25Request & request,
    const SpeakerState & speaker,
    const EmotionState & emotion) {
    std::vector<float> explicit_weights;
    if (request.use_emotion_text) {
        const auto emotion_text = request.emotion_text.value_or(request.text);
        if (const auto * cached = emotion_text_weights_cache_.find(emotion_text)) {
            debug::trace_log_scalar("index_tts2_5.qwen_emotion.text_cache.hit", 1);
            debug::trace_log_scalar("index_tts2_5.qwen_emotion.text_cache.slots", static_cast<int64_t>(emotion_text_weights_cache_.capacity()));
            debug::trace_log_scalar("index_tts2_5.qwen_emotion.text_cache.entries", static_cast<int64_t>(emotion_text_weights_cache_.size()));
            debug::trace_log_scalar("index_tts2_5.qwen_emotion.text_cache.evicted", 0);
            explicit_weights = *cached;
        } else {
            const bool will_evict =
                emotion_text_weights_cache_.capacity() > 0 &&
                emotion_text_weights_cache_.size() >= emotion_text_weights_cache_.capacity();
            explicit_weights = qwen_emotion_->infer(emotion_text, emotion_text_max_new_tokens_).values;
            if (mem_saver_) {
                qwen_emotion_->release_graphs();
            }
            emotion_text_weights_cache_.put(emotion_text, explicit_weights);
            debug::trace_log_scalar("index_tts2_5.qwen_emotion.text_cache.hit", 0);
            debug::trace_log_scalar("index_tts2_5.qwen_emotion.text_cache.slots", static_cast<int64_t>(emotion_text_weights_cache_.capacity()));
            debug::trace_log_scalar("index_tts2_5.qwen_emotion.text_cache.entries", static_cast<int64_t>(emotion_text_weights_cache_.size()));
            debug::trace_log_scalar("index_tts2_5.qwen_emotion.text_cache.evicted", will_evict ? 1 : 0);
        }
    } else if (request.emotion_vector.has_value()) {
        explicit_weights = *request.emotion_vector;
    }

    if (explicit_weights.empty()) {
        return gpt_->merge_emotion_vector(
            speaker.semantic.values,
            speaker.semantic.frames,
            emotion.semantic.values,
            emotion.semantic.frames,
            request.emotion_alpha);
    }

    auto scaled_weights = scaled_emotion_weights(explicit_weights, request.emotion_alpha);
    auto matrix = explicit_emotion_matrix_vector(
        scaled_weights,
        speaker.style,
        request.use_random_emotion,
        request.generation.seed);
    const float weight_sum = std::accumulate(scaled_weights.begin(), scaled_weights.end(), 0.0F);
    if (weight_sum != 1.0F) {
        auto base = gpt_->merge_emotion_vector(
            speaker.semantic.values,
            speaker.semantic.frames,
            emotion.semantic.values,
            emotion.semantic.frames,
            1.0F);
        for (size_t i = 0; i < matrix.size(); ++i) {
            matrix[i] += (1.0F - weight_sum) * base[i];
        }
    }
    return matrix;
}

runtime::AudioBuffer IndexTTS25Session::synthesize_segment(
    const std::vector<int32_t> & text_tokens,
    int32_t lang_id,
    size_t segment_index,
    const std::string & dump_dir,
    const SpeakerState & speaker,
    const EmotionState & emotion,
    const std::vector<float> & emotion_vector,
    const IndexTTS25GenerationOptions & options,
    uint32_t segment_seed) {
    const std::string seg = "_seg" + std::to_string(segment_index);
    dump_i32(dump_dir, "text_tokens" + seg, {1, static_cast<int64_t>(text_tokens.size())}, text_tokens);
    IndexTTS25GptGenerationRequest generation;
    generation.text_tokens = text_tokens;
    generation.speaker_style = speaker.style.values;
    generation.lang_id = lang_id;
    generation.emotion_semantic = emotion.semantic.values;
    generation.emotion_frames = emotion.semantic.frames;
    generation.emotion_vector = emotion_vector;
    generation.top_p = options.top_p;
    generation.top_k = options.top_k;
    generation.temperature = options.temperature;
    generation.repetition_penalty = options.repetition_penalty;
    generation.do_sample = options.do_sample;
    generation.length_penalty = options.length_penalty;
    generation.num_beams = options.num_beams;
    generation.max_mel_tokens = options.max_mel_tokens;
    generation.seed = segment_seed;

    const auto gen_start = Clock::now();
    auto generated = gpt_->generate_speech(generation);
    if (mem_saver_) {
        gpt_->release_conditioning_graphs();
    }
    debug::timing_log_scalar("index_tts2_5.gpt.generate_ms", engine::debug::elapsed_ms(gen_start));
    dump_i32(dump_dir, "gpt_codes" + seg, {1, static_cast<int64_t>(generated.codes.size())}, generated.codes);
    if (generated.codes.empty()) {
        throw std::runtime_error("IndexTTS2.5 GPT generated no acoustic codes");
    }
    const int64_t code_frames = static_cast<int64_t>(generated.codes.size());

    const auto s2mel_start = Clock::now();
    semantic_codec_->prepare_codes(code_frames);
    auto semantic = semantic_codec_->codes_to_embedding(generated.codes, code_frames);
    if (mem_saver_) {
        semantic_codec_->release_graphs();
    }
    auto content = channel_first_to_time_major(
        semantic.embedding_channel_first,
        semantic.dims,
        semantic.frames);
    dump_f32(dump_dir, "s_infer" + seg, {1, semantic.frames, semantic.dims}, content);
    const int64_t target_frames = static_cast<int64_t>(static_cast<float>(semantic.frames) * 1.72F);
    const int64_t total_frames = speaker.prompt_condition.frames + target_frames;
    s2mel_->prepare_length_regulator(semantic.frames, target_frames);
    auto generated_condition = s2mel_->regulate_length(content, semantic.frames, target_frames);
    dump_f32(
        dump_dir,
        "lr_gen" + seg,
        {1, generated_condition.frames, generated_condition.dims},
        generated_condition.values);
    auto condition = concat_conditions(speaker.prompt_condition, generated_condition);
    dump_f32(dump_dir, "cat_condition" + seg, {1, total_frames, kConditionDim}, condition);
    if (mem_saver_) {
        gpt_->release_generation_graphs();
        s2mel_->release_pre_cfm_graphs();
    }
    s2mel_->prepare_cfm(total_frames, kInferenceCfgRate > 0.0F);
    auto mel = s2mel_->infer_mel(
        condition,
        total_frames,
        speaker.reference_mel.values,
        speaker.reference_mel.frames,
        speaker.style.values,
        kDiffusionSteps,
        kInferenceCfgRate,
        segment_seed,
        generated.rng_offset_blocks);
    dump_f32(dump_dir, "mel_out" + seg, {1, mel.channels, mel.frames}, mel.values);
    debug::timing_log_scalar("index_tts2_5.s2mel.total_ms", engine::debug::elapsed_ms(s2mel_start));
    if (mem_saver_) {
        s2mel_->release_cfm_graph();
    }

    const auto vocoder_start = Clock::now();
    auto audio = vocoder_->synthesize(mel.values, mel.frames);
    debug::timing_log_scalar("index_tts2_5.vocoder_ms", engine::debug::elapsed_ms(vocoder_start));
    if (mem_saver_) {
        vocoder_->release_runtime_graph();
    }
    runtime::AudioBuffer out;
    out.sample_rate = audio.sample_rate;
    out.channels = 1;
    out.samples = std::move(audio.waveform);
    return out;
}

runtime::TaskResult IndexTTS25Session::run(const runtime::TaskRequest & request) {
    require_prepared("IndexTTS2.5 run");
    const auto wall_start = Clock::now();
    auto parsed = parse_index_tts2_5_request(request);
    if (!parsed.speaker_audio.has_value()) {
        throw std::runtime_error("IndexTTS2.5 request requires speaker audio");
    }
    const runtime::AudioBuffer * emotion_audio = parsed.emotion_audio.has_value()
        ? &*parsed.emotion_audio
        : &*parsed.speaker_audio;
    if (parsed.emotion_vector.has_value() || parsed.use_emotion_text) {
        emotion_audio = &*parsed.speaker_audio;
    }

    const auto & speaker = resolve_speaker_state(*parsed.speaker_audio);
    const bool emotion_same_as_speaker = same_identity(audio_identity(*emotion_audio), speaker.identity);
    debug::trace_log_scalar("index_tts2_5.emotion_audio.same_as_speaker", emotion_same_as_speaker);
    const auto & emotion = resolve_emotion_state(*emotion_audio);
    const auto emotion_vector = resolve_emotion_vector(parsed, speaker, emotion);
    if (mem_saver_) {
        gpt_->release_conditioning_graphs();
    }
    const std::string dump_dir = dump_dir_from_env();
    dump_f32(dump_dir, "campplus_embedding", {1, speaker.style.dims}, speaker.style.values);
    dump_f32(dump_dir, "speech_condition", {1, speaker.semantic.frames, speaker.semantic.dims}, speaker.semantic.values);
    dump_f32(
        dump_dir,
        "prompt_condition",
        {1, speaker.prompt_condition.frames, speaker.prompt_condition.dims},
        speaker.prompt_condition.values);
    dump_f32(dump_dir, "ref_mel", {1, speaker.reference_mel.channels, speaker.reference_mel.frames}, speaker.reference_mel.values);
    dump_f32(dump_dir, "emo_vec", {1, static_cast<int64_t>(emotion_vector.size())}, emotion_vector);
    const auto text_chunk_size = engine::text::parse_text_chunk_size_override(request.options);
    const auto text_chunk_mode = engine::text::parse_text_chunk_mode_override(request.options)
        .value_or(engine::text::TextChunkMode::Default);
    const std::vector<std::string> text_chunks = text_chunk_size.has_value()
        ? engine::text::split_text_chunks(parsed.text, *text_chunk_size, text_chunk_mode)
        : std::vector<std::string>{parsed.text};
    if (text_chunks.empty()) {
        throw std::runtime_error("IndexTTS2.5 text chunking produced no chunks");
    }

    std::vector<std::vector<int32_t>> segment_token_ids;
    std::vector<int32_t> segment_lang_ids;
    for (const auto & text_chunk : text_chunks) {
        const auto text_encoding = tokenizer_.encode_for_inference(
            text_chunk,
            parsed.max_text_tokens_per_segment,
            parsed.lang);
        const int32_t lang_id = IndexTTS25TextTokenizer::lang_to_id(text_encoding.lang);
        for (const auto & ids : text_encoding.segment_token_ids) {
            segment_token_ids.push_back(ids);
            segment_lang_ids.push_back(lang_id);
        }
    }

    runtime::AudioBuffer merged;
    for (size_t i = 0; i < segment_token_ids.size(); ++i) {
        if (i > 0) {
            append_silence(merged, parsed.interval_silence_ms);
        }
        auto segment_audio = synthesize_segment(
            segment_token_ids[i],
            segment_lang_ids[i],
            i,
            dump_dir,
            speaker,
            emotion,
            emotion_vector,
            parsed.generation,
            parsed.generation.seed + static_cast<uint32_t>(i));
        runtime::append_audio_buffer(merged, segment_audio);
    }

    runtime::TaskResult result;
    result.audio_output = std::move(merged);
    debug::trace_log_scalar("index_tts2_5.path.use_emotion_text", parsed.use_emotion_text);
    debug::trace_log_scalar("index_tts2_5.path.has_emotion_vector", parsed.emotion_vector.has_value());
    debug::trace_log_scalar("index_tts2_5.path.has_emotion_audio", parsed.emotion_audio.has_value());
    if (text_chunk_size.has_value()) {
        debug::trace_log_scalar("index_tts2_5.text_chunk_size", *text_chunk_size);
        debug::trace_log_scalar("index_tts2_5.text_chunk_mode", engine::text::text_chunk_mode_name(text_chunk_mode));
    }
    debug::trace_log_scalar("index_tts2_5.text_chunk_count", static_cast<int64_t>(text_chunks.size()));
    debug::trace_log_scalar("index_tts2_5.text.segment_count", static_cast<int64_t>(segment_token_ids.size()));
    debug::timing_log_scalar("session.wall_ms", engine::debug::elapsed_ms(wall_start));
    return result;
}

}  // namespace engine::models::index_tts2_5
