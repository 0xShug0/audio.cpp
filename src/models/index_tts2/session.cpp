#include "engine/models/index_tts2/session.h"

#include "engine/framework/debug/profiler.h"
#include "engine/framework/io/text.h"
#include "engine/framework/runtime/options.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>

namespace engine::models::index_tts2 {
namespace {

using Clock = std::chrono::steady_clock;
constexpr int64_t kSemanticDim = 1024;
constexpr int64_t kConditionDim = 512;
constexpr int64_t kGptDim = 1280;
constexpr int64_t kStyleDim = 192;
constexpr int64_t kEmotionCount = 8;
constexpr int64_t kDiffusionSteps = 25;
constexpr float kInferenceCfgRate = 0.7F;

std::shared_ptr<const IndexTTS2Assets> require_assets(std::shared_ptr<const IndexTTS2Assets> assets) {
    if (assets == nullptr) {
        throw std::runtime_error("IndexTTS2 session requires assets");
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

IndexTTS2AudioIdentity audio_identity(const runtime::AudioBuffer & audio) {
    return {
        audio.sample_rate,
        audio.channels,
        static_cast<uint64_t>(audio.samples.size()),
        hash_audio_samples(audio),
    };
}

bool same_identity(const IndexTTS2AudioIdentity & lhs, const IndexTTS2AudioIdentity & rhs) {
    return lhs.sample_rate == rhs.sample_rate &&
        lhs.channels == rhs.channels &&
        lhs.sample_count == rhs.sample_count &&
        lhs.sample_hash == rhs.sample_hash;
}

std::vector<float> channel_first_to_time_major(
    const std::vector<float> & values,
    int64_t channels,
    int64_t frames) {
    if (static_cast<int64_t>(values.size()) != channels * frames) {
        throw std::runtime_error("IndexTTS2 channel-first tensor size mismatch");
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

std::vector<float> add_latent_to_semantic(
    const IndexTTS2SemanticCodecOutput & semantic,
    const IndexTTS2S2MelSequence & latent) {
    if (semantic.frames != latent.frames || semantic.dims != latent.dims || semantic.dims != kSemanticDim) {
        throw std::runtime_error("IndexTTS2 semantic embedding and GPT latent shape mismatch");
    }
    auto content = channel_first_to_time_major(semantic.embedding_channel_first, semantic.dims, semantic.frames);
    for (size_t i = 0; i < content.size(); ++i) {
        content[i] += latent.values[i];
    }
    return content;
}

std::vector<float> concat_conditions(
    const IndexTTS2S2MelSequence & prompt,
    const IndexTTS2S2MelSequence & generated) {
    if (prompt.dims != kConditionDim || generated.dims != kConditionDim) {
        throw std::runtime_error("IndexTTS2 condition dimension mismatch");
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
        throw std::runtime_error("IndexTTS2 emotion vector must contain exactly 8 values");
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
    if (const auto value = runtime::find_option(options.options, {"index_tts2.mem_saver", "mem_saver"})) {
        return runtime::parse_bool_option(*value, "index_tts2.mem_saver");
    }
    return false;
}

}  // namespace

IndexTTS2Session::IndexTTS2Session(
    runtime::TaskSpec task,
    runtime::SessionOptions options,
    std::shared_ptr<const IndexTTS2Assets> assets)
    : RuntimeSessionBase(options),
      task_(task),
      assets_(require_assets(std::move(assets))),
      tokenizer_(assets_) {
    gpt_graph_arena_bytes_ = runtime::parse_size_mb_option(
        options.options, {"index_tts2.gpt_graph_arena_mb"}, gpt_graph_arena_bytes_);
    s2mel_graph_arena_bytes_ = runtime::parse_size_mb_option(
        options.options, {"index_tts2.s2mel_graph_arena_mb"}, s2mel_graph_arena_bytes_);
    reference_graph_arena_bytes_ = runtime::parse_size_mb_option(
        options.options, {"index_tts2.reference_graph_arena_mb"}, reference_graph_arena_bytes_);
    emotion_text_prefill_graph_arena_bytes_ = runtime::parse_size_mb_option(
        options.options, {"index_tts2.emotion_text_prefill_graph_arena_mb"}, emotion_text_prefill_graph_arena_bytes_);
    emotion_text_decode_graph_arena_bytes_ = runtime::parse_size_mb_option(
        options.options, {"index_tts2.emotion_text_decode_graph_arena_mb"}, emotion_text_decode_graph_arena_bytes_);
    weight_context_bytes_ = runtime::parse_size_mb_option(
        options.options, {"index_tts2.weight_context_mb"}, weight_context_bytes_);
    if (const auto value = runtime::parse_int_option(options.options, {"index_tts2.emotion_text_max_new_tokens"})) {
        if (*value <= 0) {
            throw std::runtime_error("index_tts2.emotion_text_max_new_tokens must be positive");
        }
        emotion_text_max_new_tokens_ = *value;
    }
    if (const auto it = options.options.find("index_tts2.weight_type"); it != options.options.end()) {
        matmul_weight_storage_type_ = engine::assets::parse_tensor_storage_type(it->second);
        validate_matmul_weight_storage(matmul_weight_storage_type_, "index_tts2.weight_type");
    }
    if (const auto it = options.options.find("index_tts2.conv_weight_type"); it != options.options.end()) {
        conv_weight_storage_type_ = engine::assets::parse_tensor_storage_type(it->second);
        validate_conv_weight_storage(conv_weight_storage_type_, "index_tts2.conv_weight_type");
    }
    mem_saver_ = mem_saver_from_options(options);
    for (const auto & [key, _] : options.options) {
        if (key.rfind("index_tts2.", 0) == 0 &&
            key != "index_tts2.gpt_graph_arena_mb" &&
            key != "index_tts2.s2mel_graph_arena_mb" &&
            key != "index_tts2.reference_graph_arena_mb" &&
            key != "index_tts2.emotion_text_prefill_graph_arena_mb" &&
            key != "index_tts2.emotion_text_decode_graph_arena_mb" &&
            key != "index_tts2.emotion_text_max_new_tokens" &&
            key != "index_tts2.weight_context_mb" &&
            key != "index_tts2.weight_type" &&
            key != "index_tts2.conv_weight_type" &&
            key != "index_tts2.mem_saver" &&
            key != "index_tts2.gpt.cuda_sampling_policy" &&
            key != "index_tts2.s2mel.cuda_sampling_policy") {
            throw std::runtime_error("unknown IndexTTS2 session option: " + key);
        }
    }
    if (task_.mode != runtime::RunMode::Offline || task_.task != runtime::VoiceTaskKind::Tts) {
        throw std::runtime_error("IndexTTS2 currently supports offline TTS sessions");
    }

    semantic_encoder_ = std::make_unique<IndexTTS2Wav2Vec2BertRuntime>(
        assets_,
        execution_context(),
        reference_graph_arena_bytes_,
        weight_context_bytes_,
        matmul_weight_storage_type_,
        conv_weight_storage_type_);
    semantic_codec_ = std::make_unique<IndexTTS2SemanticCodecRuntime>(
        assets_,
        execution_context(),
        reference_graph_arena_bytes_,
        weight_context_bytes_,
        matmul_weight_storage_type_,
        conv_weight_storage_type_);
    style_encoder_ = std::make_unique<IndexTTS2StyleEncoder>(
        assets_,
        options.backend,
        conv_weight_storage_type_);
    gpt_ = std::make_unique<IndexTTS2GptRuntime>(
        assets_,
        execution_context(),
        gpt_graph_arena_bytes_,
        weight_context_bytes_,
        matmul_weight_storage_type_,
        conv_weight_storage_type_);
    s2mel_ = std::make_unique<IndexTTS2S2MelRuntime>(
        assets_,
        execution_context(),
        s2mel_graph_arena_bytes_,
        weight_context_bytes_,
        matmul_weight_storage_type_,
        conv_weight_storage_type_);
    vocoder_ = std::make_unique<IndexTTS2BigVganVocoder>(
        assets_,
        options.backend,
        conv_weight_storage_type_);
    qwen_emotion_ = std::make_unique<IndexTTS2QwenEmotionRuntime>(
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

std::string IndexTTS2Session::family() const {
    return "index_tts2";
}

runtime::VoiceTaskKind IndexTTS2Session::task_kind() const {
    return task_.task;
}

runtime::RunMode IndexTTS2Session::run_mode() const {
    return task_.mode;
}

void IndexTTS2Session::prepare(const runtime::SessionPreparationRequest & request) {
    if (request.text.has_value() || runtime::find_option(request.options, {"text", "prompt"}).has_value()) {
        std::string text;
        if (request.text.has_value()) {
            text = request.text->text;
        } else if (const auto value = runtime::find_option(request.options, {"text", "prompt"})) {
            text = *value;
        }
        text = engine::io::trim_ascii_whitespace(text);
        if (text.empty()) {
            throw std::runtime_error("IndexTTS2 request requires text_input or text option");
        }
        IndexTTS2GenerationOptions generation;
        if (const auto value = runtime::parse_int_option(request.options, {"max_tokens"})) {
            if (*value <= 0) {
                throw std::runtime_error("IndexTTS2 max_tokens must be positive");
            }
            generation.max_mel_tokens = *value;
        }
        if (const auto value = runtime::parse_int_option(request.options, {"num_beams"})) {
            if (*value <= 0) {
                throw std::runtime_error("IndexTTS2 num_beams must be positive");
            }
            generation.num_beams = *value;
        }
        const auto text_encoding = tokenizer_.encode_for_inference(
            text,
            IndexTTS2Request{}.max_text_tokens_per_segment);
        for (const auto & segment : text_encoding.segment_token_ids) {
            gpt_->prepare_generation(
                static_cast<int64_t>(segment.size()),
                generation.max_mel_tokens,
                generation.num_beams);
        }
    }
    mark_prepared();
}

const IndexTTS2Session::SpeakerState & IndexTTS2Session::resolve_speaker_state(const runtime::AudioBuffer & audio) {
    const auto identity = audio_identity(audio);
    const bool cache_hit = speaker_cache_.has_value() && same_identity(speaker_cache_->identity, identity);
    debug::trace_log_scalar("index_tts2.speaker_cache.hit", cache_hit);
    if (cache_hit) {
        return *speaker_cache_;
    }
    const auto start = Clock::now();
    const auto prepared = prepare_index_tts2_reference_audio(
        audio.samples,
        audio.sample_rate,
        audio.channels,
        assets_->config.s2mel,
        static_cast<size_t>(std::max(1, options().backend.threads)),
        true);
    semantic_encoder_->prepare(prepared.semantic_features.frames);
    auto semantic = semantic_encoder_->encode(prepared.semantic_features);
    semantic_codec_->prepare_quantize(semantic.frames);
    auto reference_codes = semantic_codec_->quantize(semantic);
    const auto reference_content = channel_first_to_time_major(
        reference_codes.embedding_channel_first,
        reference_codes.dims,
        reference_codes.frames);
    debug::trace_log_scalar("index_tts2.s2mel.reference_mel_frames", static_cast<double>(prepared.mel.frames));
    s2mel_->prepare_length_regulator(reference_codes.frames, prepared.mel.frames);
    auto prompt_condition = s2mel_->regulate_length(
        reference_content,
        reference_codes.frames,
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
    speaker_cache_ = std::move(state);
    if (mem_saver_) {
        semantic_encoder_->release_graph();
        semantic_codec_->release_graphs();
        s2mel_->release_pre_cfm_graphs();
    }
    debug::timing_log_scalar("index_tts2.speaker_state_ms", engine::debug::elapsed_ms(start));
    return *speaker_cache_;
}

const IndexTTS2Session::EmotionState & IndexTTS2Session::resolve_emotion_state(const runtime::AudioBuffer & audio) {
    const auto identity = audio_identity(audio);
    const bool cache_hit = emotion_cache_.has_value() && same_identity(emotion_cache_->identity, identity);
    debug::trace_log_scalar("index_tts2.emotion_cache.hit", cache_hit);
    if (cache_hit) {
        return *emotion_cache_;
    }
    const auto start = Clock::now();
    const auto prepared = prepare_index_tts2_reference_audio(
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
    emotion_cache_ = std::move(state);
    if (mem_saver_) {
        semantic_encoder_->release_graph();
    }
    debug::timing_log_scalar("index_tts2.emotion_state_ms", engine::debug::elapsed_ms(start));
    return *emotion_cache_;
}

std::vector<float> IndexTTS2Session::explicit_emotion_matrix_vector(
    const std::vector<float> & emotion_weights,
    const IndexTTS2StyleEmbedding & style,
    bool use_random,
    uint32_t seed) const {
    if (static_cast<int64_t>(emotion_weights.size()) != kEmotionCount ||
        style.dims != kStyleDim ||
        static_cast<int64_t>(style.values.size()) != kStyleDim) {
        throw std::runtime_error("IndexTTS2 explicit emotion matrix input shape mismatch");
    }
    std::vector<float> out(static_cast<size_t>(kGptDim), 0.0F);
    int64_t row_offset = 0;
    std::mt19937 rng(seed);
    for (int64_t emotion = 0; emotion < kEmotionCount; ++emotion) {
        const int64_t rows = assets_->config.emo_num[static_cast<size_t>(emotion)];
        int64_t best_row = 0;
        if (rows <= 0) {
            throw std::runtime_error("IndexTTS2 emo_num contains non-positive group size");
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

std::vector<float> IndexTTS2Session::resolve_emotion_vector(
    const IndexTTS2Request & request,
    const SpeakerState & speaker,
    const EmotionState & emotion) {
    std::vector<float> explicit_weights;
    if (request.use_emotion_text) {
        const auto emotion_text = request.emotion_text.value_or(request.text);
        const bool cache_hit = emotion_text_weights_cache_.has_value() && emotion_text_weights_cache_->text == emotion_text;
        debug::trace_log_scalar("index_tts2.qwen_emotion.text_cache.hit", cache_hit);
        if (cache_hit) {
            explicit_weights = emotion_text_weights_cache_->weights;
        } else {
            explicit_weights = qwen_emotion_->infer(emotion_text, emotion_text_max_new_tokens_).values;
            if (mem_saver_) {
                qwen_emotion_->release_graphs();
            }
            emotion_text_weights_cache_ = EmotionTextWeightsCache{emotion_text, explicit_weights};
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

runtime::AudioBuffer IndexTTS2Session::synthesize_segment(
    const std::vector<int32_t> & text_tokens,
    const SpeakerState & speaker,
    const EmotionState & emotion,
    const std::vector<float> & emotion_vector,
    const IndexTTS2GenerationOptions & options,
    uint32_t segment_seed) {
    IndexTTS2GptGenerationRequest generation;
    generation.text_tokens = text_tokens;
    generation.speaker_semantic = speaker.semantic.values;
    generation.speaker_frames = speaker.semantic.frames;
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
    debug::timing_log_scalar("index_tts2.gpt.generate_ms", engine::debug::elapsed_ms(gen_start));
    if (generated.codes.empty()) {
        throw std::runtime_error("IndexTTS2 GPT generated no acoustic codes");
    }
    const int64_t code_frames = static_cast<int64_t>(generated.codes.size());
    const int64_t target_frames = static_cast<int64_t>(static_cast<float>(code_frames) * 1.72F);
    const int64_t total_frames = speaker.prompt_condition.frames + target_frames;
    const auto forward_start = Clock::now();
    auto latent = gpt_->forward_latent(
        generated.speech_conditioning_latent,
        text_tokens,
        generated.codes,
        emotion.semantic.values,
        emotion.semantic.frames,
        emotion_vector);
    debug::timing_log_scalar("index_tts2.gpt.forward_ms", engine::debug::elapsed_ms(forward_start));

    const auto s2mel_start = Clock::now();
    s2mel_->prepare_gpt_layer(latent.frames);
    auto projected = s2mel_->project_gpt_latent(latent.values, latent.frames);
    semantic_codec_->prepare_codes(code_frames);
    auto semantic = semantic_codec_->codes_to_embedding(generated.codes, code_frames);
    if (mem_saver_) {
        semantic_codec_->release_graphs();
    }
    auto content = add_latent_to_semantic(semantic, projected);
    s2mel_->prepare_length_regulator(code_frames, target_frames);
    auto generated_condition = s2mel_->regulate_length(content, code_frames, target_frames);
    auto condition = concat_conditions(speaker.prompt_condition, generated_condition);
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
    debug::timing_log_scalar("index_tts2.s2mel.total_ms", engine::debug::elapsed_ms(s2mel_start));
    if (mem_saver_) {
        s2mel_->release_cfm_graph();
    }

    const auto vocoder_start = Clock::now();
    auto audio = vocoder_->synthesize(mel.values, mel.frames);
    debug::timing_log_scalar("index_tts2.vocoder_ms", engine::debug::elapsed_ms(vocoder_start));
    runtime::AudioBuffer out;
    out.sample_rate = audio.sample_rate;
    out.channels = 1;
    out.samples = std::move(audio.waveform);
    return out;
}

runtime::TaskResult IndexTTS2Session::run(const runtime::TaskRequest & request) {
    require_prepared("IndexTTS2 run");
    const auto wall_start = Clock::now();
    auto parsed = parse_index_tts2_request(request);
    if (!parsed.speaker_audio.has_value()) {
        throw std::runtime_error("IndexTTS2 request requires speaker audio");
    }
    const runtime::AudioBuffer * emotion_audio = parsed.emotion_audio.has_value()
        ? &*parsed.emotion_audio
        : &*parsed.speaker_audio;
    if (parsed.emotion_vector.has_value() || parsed.use_emotion_text) {
        emotion_audio = &*parsed.speaker_audio;
    }

    const auto & speaker = resolve_speaker_state(*parsed.speaker_audio);
    const bool emotion_same_as_speaker = same_identity(audio_identity(*emotion_audio), speaker.identity);
    debug::trace_log_scalar("index_tts2.emotion_audio.same_as_speaker", emotion_same_as_speaker);
    const auto & emotion = resolve_emotion_state(*emotion_audio);
    const auto emotion_vector = resolve_emotion_vector(parsed, speaker, emotion);
    if (mem_saver_) {
        gpt_->release_conditioning_graphs();
    }
    const auto text_encoding = tokenizer_.encode_for_inference(
        parsed.text,
        parsed.max_text_tokens_per_segment);

    runtime::AudioBuffer merged;
    for (size_t i = 0; i < text_encoding.segment_token_ids.size(); ++i) {
        if (i > 0) {
            append_silence(merged, parsed.interval_silence_ms);
        }
        auto segment_audio = synthesize_segment(
            text_encoding.segment_token_ids[i],
            speaker,
            emotion,
            emotion_vector,
            parsed.generation,
            parsed.generation.seed + static_cast<uint32_t>(i));
        runtime::append_audio_buffer(merged, segment_audio);
    }

    runtime::TaskResult result;
    result.audio_output = std::move(merged);
    debug::trace_log_scalar("index_tts2.path.use_emotion_text", parsed.use_emotion_text);
    debug::trace_log_scalar("index_tts2.path.has_emotion_vector", parsed.emotion_vector.has_value());
    debug::trace_log_scalar("index_tts2.path.has_emotion_audio", parsed.emotion_audio.has_value());
    debug::trace_log_scalar("index_tts2.text.segment_count", text_encoding.segment_token_ids.size());
    debug::timing_log_scalar("session.wall_ms", engine::debug::elapsed_ms(wall_start));
    return result;
}

}  // namespace engine::models::index_tts2
