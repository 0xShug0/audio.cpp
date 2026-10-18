#include "engine/models/personaplex/session.h"

#include "engine/framework/audio/conversion.h"
#include "engine/framework/runtime/options.h"
#include "engine/framework/runtime/spec_backed_model.h"
#include "engine/framework/sampling/hf_sampler.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/tokenizers/sentencepiece.h"
#include "engine/models/personaplex/request.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cctype>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace engine::models::personaplex {
namespace {

using Clock = std::chrono::steady_clock;

constexpr const char * kFamily = "personaplex";
constexpr const char * kModelName = "PersonaPlex";
constexpr int kSampleRate = 24000;
constexpr int64_t kFrameSamples = 1920;
constexpr int64_t kMimiFrameCodebooks = 8;
constexpr int64_t kDelayCacheSteps = 4;
constexpr int64_t kTextInitialToken = 32000;
constexpr int64_t kAudioInitialToken = 2048;
constexpr int64_t kZeroTextToken = 3;
constexpr int64_t kUngeneratedToken = -2;
constexpr std::array<int64_t, 17> kDelays = {
    0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1,
};
constexpr std::array<int32_t, 8> kInitialAudioTokens = {
    2048, 2048, 2048, 2048, 2048, 2048, 2048, 2048,
};
constexpr std::array<int32_t, 8> kSilenceTokens = {948, 243, 1178, 546, 1736, 1030, 1978, 2008};
constexpr std::array<int32_t, 8> kSineTokens = {430, 1268, 381, 1611, 1095, 1495, 56, 472};

std::shared_ptr<const PersonaPlexAssets> require_assets(std::shared_ptr<const PersonaPlexAssets> assets) {
    if (assets == nullptr) {
        throw std::runtime_error("PersonaPlex session requires loaded assets");
    }
    return assets;
}

engine::codecs::MimiCodecConfig mimi_codec_config(const PersonaPlexMimiConfig & config) {
    engine::codecs::MimiCodecConfig out;
    out.sample_rate = config.sample_rate;
    out.frame_rate = config.frame_rate;
    out.channels = config.channels;
    out.hidden_size = config.hidden_size;
    out.num_heads = config.num_heads;
    out.intermediate_size = config.intermediate_size;
    out.transformer_layers = config.transformer_layers;
    out.context = config.context;
    out.latent_size = config.latent_size;
    out.codebooks = config.codebooks;
    out.total_codebooks = config.total_codebooks;
    out.codebook_size = config.codebook_size;
    out.encoder_upsample_stride = config.encoder_upsample_stride;
    return out;
}

void validate_task(const runtime::TaskSpec & task) {
    if (task.task != runtime::VoiceTaskKind::SpeechToSpeech) {
        throw std::runtime_error("PersonaPlex supports only speech-to-speech sessions");
    }
    if (task.mode != runtime::RunMode::Offline && task.mode != runtime::RunMode::Streaming) {
        throw std::runtime_error("PersonaPlex supports offline and streaming sessions");
    }
}

std::string wrap_system_prompt(std::string text) {
    text.erase(text.begin(), std::find_if(text.begin(), text.end(), [](unsigned char ch) {
        return !std::isspace(ch);
    }));
    text.erase(std::find_if(text.rbegin(), text.rend(), [](unsigned char ch) {
        return !std::isspace(ch);
    }).base(), text.end());
    if (text.empty()) {
        return {};
    }
    if (text.rfind("<system>", 0) == 0 && text.size() >= 8 &&
        text.compare(text.size() - 8, 8, "<system>") == 0) {
        return text;
    }
    return "<system> " + text + " <system>";
}

int64_t resampled_mono_sample_count(
    int64_t source_frames,
    int source_sample_rate,
    int target_sample_rate) {
    if (source_frames < 0 || source_sample_rate <= 0 || target_sample_rate <= 0) {
        throw std::runtime_error("PersonaPlex audio sample count requires valid audio format");
    }
    if (source_sample_rate == target_sample_rate) {
        return source_frames;
    }
    return static_cast<int64_t>(std::llround(
        static_cast<double>(source_frames) *
        static_cast<double>(target_sample_rate) /
        static_cast<double>(source_sample_rate)));
}

int64_t resampled_mono_sample_count(const runtime::AudioBuffer & audio, int target_sample_rate) {
    if (audio.channels <= 0 ||
        audio.samples.size() % static_cast<size_t>(audio.channels) != 0) {
        throw std::runtime_error("PersonaPlex audio sample count requires valid channel layout");
    }
    return resampled_mono_sample_count(
        static_cast<int64_t>(audio.samples.size() / static_cast<size_t>(audio.channels)),
        audio.sample_rate,
        target_sample_rate);
}

class PersonaPlexDelayState {
public:
    PersonaPlexDelayState()
        : cache_(static_cast<size_t>(kDelays.size() * kDelayCacheSteps), static_cast<int32_t>(kUngeneratedToken)),
          provided_(cache_.size(), 0) {}

    void import_cache(const PersonaPlexVoicePromptState & state) {
        if (state.cache.size() != cache_.size()) {
            throw std::runtime_error("PersonaPlex voice prompt delay cache size mismatch");
        }
        for (size_t i = 0; i < cache_.size(); ++i) {
            cache_[i] = static_cast<int32_t>(state.cache[i]);
        }
    }

    std::optional<PersonaPlexDelayedStep> prepare(
        const int32_t * user_tokens,
        const int32_t * moshi_tokens,
        std::optional<int32_t> text_token) {
        if (user_tokens != nullptr) {
            for (int64_t q = 0; q < kMimiFrameCodebooks; ++q) {
                write_stream(
                    1 + kMimiFrameCodebooks + q,
                    offset_ + kDelays[static_cast<size_t>(1 + kMimiFrameCodebooks + q)],
                    user_tokens[q]);
            }
        }
        if (moshi_tokens != nullptr) {
            for (int64_t q = 0; q < kMimiFrameCodebooks; ++q) {
                write_stream(1 + q, offset_ + kDelays[static_cast<size_t>(1 + q)], moshi_tokens[q]);
            }
        }
        if (text_token.has_value()) {
            write_stream(0, offset_ + kDelays[0], *text_token);
        }

        for (size_t stream = 0; stream < kDelays.size(); ++stream) {
            if (offset_ <= kDelays[stream]) {
                write_stream(static_cast<int64_t>(stream), offset_, initial_token(stream));
            }
        }

        if (offset_ == 0) {
            for (size_t stream = 0; stream < kDelays.size(); ++stream) {
                write_stream(static_cast<int64_t>(stream), 0, initial_token(stream));
            }
            ++offset_;
            return std::nullopt;
        }

        PersonaPlexDelayedStep step;
        step.model_input_position = (offset_ - 1) % kDelayCacheSteps;
        step.target_position = offset_ % kDelayCacheSteps;
        for (size_t stream = 0; stream < kDelays.size(); ++stream) {
            step.tokens[stream] = cache_[flat_index(static_cast<int64_t>(stream), step.model_input_position)];
            step.target[stream] = cache_[flat_index(static_cast<int64_t>(stream), step.target_position)];
            step.provided[stream] = provided_[flat_index(static_cast<int64_t>(stream), step.target_position)];
        }
        return step;
    }

    std::optional<std::array<int32_t, kMimiFrameCodebooks>> finish_with_sampling(
        int32_t sampled_text_token,
        const std::array<int32_t, kPersonaPlexDepformerAudioStreams> & sampled_audio_tokens) {
        const int64_t target_position = offset_ % kDelayCacheSteps;
        if (provided_[flat_index(0, target_position)] == 0) {
            cache_[flat_index(0, target_position)] = sampled_text_token;
        }
        for (size_t q = 0; q < kPersonaPlexDepformerAudioStreams; ++q) {
            const int64_t stream = 1 + static_cast<int64_t>(q);
            if (provided_[flat_index(stream, target_position)] == 0) {
                cache_[flat_index(stream, target_position)] = sampled_audio_tokens[q];
            }
        }
        const int64_t model_input_position = (offset_ - 1) % kDelayCacheSteps;
        for (size_t stream = 0; stream < kDelays.size(); ++stream) {
            provided_[flat_index(static_cast<int64_t>(stream), model_input_position)] = 0;
        }
        std::optional<std::array<int32_t, kMimiFrameCodebooks>> output = std::nullopt;
        constexpr int64_t max_delay = 1;
        if (offset_ > max_delay) {
            output.emplace();
            for (int64_t q = 0; q < kMimiFrameCodebooks; ++q) {
                const int64_t stream = 1 + q;
                const int64_t source_position =
                    (offset_ - max_delay + kDelays[static_cast<size_t>(stream)]) % kDelayCacheSteps;
                (*output)[static_cast<size_t>(q)] = cache_[flat_index(stream, source_position)];
            }
        }
        ++offset_;
        return output;
    }

    int64_t offset() const noexcept {
        return offset_;
    }

private:
    static size_t flat_index(int64_t stream, int64_t step) {
        return static_cast<size_t>(stream * kDelayCacheSteps + (step % kDelayCacheSteps));
    }

    static int32_t initial_token(size_t stream) {
        return stream == 0 ? static_cast<int32_t>(kTextInitialToken) : static_cast<int32_t>(kAudioInitialToken);
    }

    void write_stream(int64_t stream, int64_t logical_step, int32_t token) {
        const size_t index = flat_index(stream, logical_step);
        cache_[index] = token;
        provided_[index] = 1;
    }

    std::vector<int32_t> cache_;
    std::vector<uint8_t> provided_;
    int64_t offset_ = 0;
};

std::optional<std::array<int32_t, kMimiFrameCodebooks>> run_prepared_embedding_step(
    PersonaPlexDelayState & state,
    PersonaPlexMainStepGraph & graph,
    PersonaPlexDepformerRuntime & depformer,
    const PersonaPlexGenerationOptions & options,
    engine::sampling::HfSamplerScratch & sampler_scratch,
    std::mt19937 & rng,
    const std::vector<float> & embedding) {
    std::optional<PersonaPlexDelayedStep> step;
    while (!step.has_value()) {
        step = state.prepare(kInitialAudioTokens.data(), kInitialAudioTokens.data(), static_cast<int32_t>(kZeroTextToken));
    }
    const auto main = graph.run_embedding_step(embedding);
    engine::sampling::HfSampler sampler;
    engine::sampling::HfSamplingOptions text_sampling;
    text_sampling.do_sample = options.do_sample;
    text_sampling.temperature = options.text_temperature;
    text_sampling.top_k = options.text_top_k;
    const int32_t sampled_text = sampler.sample(
        main.text_logits,
        {},
        text_sampling,
        sampler_scratch,
        rng,
        nullptr,
        "PersonaPlex text");
    const int32_t next_text = step->provided[0] != 0 ? step->target[0] : sampled_text;
    std::array<int32_t, kPersonaPlexDepformerAudioStreams> audio_target{};
    std::array<uint8_t, kPersonaPlexDepformerAudioStreams> audio_provided{};
    std::copy_n(step->target.begin() + 1, audio_target.size(), audio_target.begin());
    std::copy_n(step->provided.begin() + 1, audio_provided.size(), audio_provided.begin());
    const auto dep = depformer.run(
        next_text,
        main.hidden,
        audio_target,
        audio_provided,
        PersonaPlexDepformerSamplingOptions{options.do_sample, options.temperature, options.top_k},
        sampler_scratch,
        rng,
        nullptr);
    return state.finish_with_sampling(sampled_text, dep.sampled_audio_tokens);
}

std::optional<std::array<int32_t, kMimiFrameCodebooks>> run_prepared_token_step(
    PersonaPlexDelayState & state,
    PersonaPlexMainStepGraph & graph,
    PersonaPlexDepformerRuntime & depformer,
    const PersonaPlexGenerationOptions & options,
    engine::sampling::HfSamplerScratch & sampler_scratch,
    std::mt19937 & rng,
    const PersonaPlexDelayedStep & step) {
    const auto main = graph.run_token_step(step.tokens);
    engine::sampling::HfSampler sampler;
    engine::sampling::HfSamplingOptions text_sampling;
    text_sampling.do_sample = options.do_sample;
    text_sampling.temperature = options.text_temperature;
    text_sampling.top_k = options.text_top_k;
    const int32_t sampled_text = sampler.sample(
        main.text_logits,
        {},
        text_sampling,
        sampler_scratch,
        rng,
        nullptr,
        "PersonaPlex text");
    const int32_t next_text = step.provided[0] != 0 ? step.target[0] : sampled_text;
    std::array<int32_t, kPersonaPlexDepformerAudioStreams> audio_target{};
    std::array<uint8_t, kPersonaPlexDepformerAudioStreams> audio_provided{};
    std::copy_n(step.target.begin() + 1, audio_target.size(), audio_target.begin());
    std::copy_n(step.provided.begin() + 1, audio_provided.size(), audio_provided.begin());
    const auto dep = depformer.run(
        next_text,
        main.hidden,
        audio_target,
        audio_provided,
        PersonaPlexDepformerSamplingOptions{options.do_sample, options.temperature, options.top_k},
        sampler_scratch,
        rng,
        nullptr);
    return state.finish_with_sampling(sampled_text, dep.sampled_audio_tokens);
}

}  // namespace

struct PersonaPlexSession::ConversationState {
    PersonaPlexGenerationOptions generation;
    PersonaPlexDelayState delay_state;
    runtime::AudioBuffer raw_audio;
    int input_sample_rate = 0;
    int input_channels = 0;
    int64_t input_source_frames = 0;
    Clock::time_point wall_start = Clock::now();
    int64_t encoded_frames = 0;
    std::vector<int32_t> preencoded_user_codes;
    int64_t preencoded_user_frames = 0;
    bool incremental_user_encoder = false;
    std::vector<int32_t> output_codes;
    int64_t emitted_samples = 0;
    runtime::AudioBuffer final_audio{kSampleRate, 1, {}};
    double ar_generate_ms = 0.0;
    double mimi_decode_ms = 0.0;
};

PersonaPlexSession::PersonaPlexSession(
    runtime::TaskSpec task,
    runtime::SessionOptions options,
    std::shared_ptr<const PersonaPlexAssets> assets,
    std::shared_ptr<const engine::model_spec::ModelContract> contract)
    : runtime::RuntimeSessionBase(options),
      task_(task),
      assets_(require_assets(std::move(assets))),
      contract_(std::move(contract)) {
    validate_task(task_);
    fallback_rng_.seed(42424242);
    if (contract_ == nullptr) {
        throw std::runtime_error("PersonaPlex session requires a model contract");
    }
    runtime::validate_spec_backed_session_options(options, *contract_, kFamily, kModelName);
    graph_arena_bytes_ =
        runtime::parse_size_mb_option(options.options, {"personaplex.graph_arena_mb"}, graph_arena_bytes_);
    lm_weight_context_bytes_ =
        runtime::parse_size_mb_option(options.options, {"personaplex.lm_weight_context_mb"}, lm_weight_context_bytes_);
    depformer_weight_context_bytes_ =
        runtime::parse_size_mb_option(options.options, {"personaplex.depformer_weight_context_mb"}, depformer_weight_context_bytes_);
    mimi_weight_context_bytes_ =
        runtime::parse_size_mb_option(options.options, {"personaplex.mimi_weight_context_mb"}, mimi_weight_context_bytes_);
    weight_storage_type_ = runtime::parse_tensor_storage_option(
        options.options,
        "personaplex.weight_type",
        assets::TensorStorageType::Native,
        {assets::TensorStorageType::Native,
         assets::TensorStorageType::F32,
         assets::TensorStorageType::F16,
         assets::TensorStorageType::BF16,
         assets::TensorStorageType::Q8_0});
}

PersonaPlexSession::~PersonaPlexSession() = default;

std::string PersonaPlexSession::family() const {
    return kFamily;
}

runtime::VoiceTaskKind PersonaPlexSession::task_kind() const {
    return task_.task;
}

runtime::RunMode PersonaPlexSession::run_mode() const {
    return task_.mode;
}

void PersonaPlexSession::prepare(const runtime::SessionPreparationRequest &) {
    if (lm_weights_ == nullptr) {
        lm_weights_ = load_personaplex_lm_weights(
            *assets_,
            execution_context().backend(),
            execution_context().backend_type(),
            lm_weight_context_bytes_,
            weight_storage_type_);
    }
    if (depformer_weights_ == nullptr) {
        depformer_weights_ = load_personaplex_depformer_weights(
            *assets_,
            execution_context().backend(),
            execution_context().backend_type(),
            depformer_weight_context_bytes_,
            weight_storage_type_);
    }
    if (mimi_codec_ == nullptr) {
        mimi_codec_ = std::make_unique<engine::codecs::MimiCodecComponent>(
            engine::codecs::MimiCodecComponent::load_from_tensor_source(
                assets_->mimi_weights,
                mimi_codec_config(assets_->config.mimi),
                {},
                execution_context().backend(),
                execution_context().backend_type(),
                mimi_weight_context_bytes_,
                weight_storage_type_));
    }
    mark_prepared();
}

void PersonaPlexSession::ensure_runtime_graphs() {
    if (main_step_graph_ == nullptr) {
        main_step_graph_ = std::make_unique<PersonaPlexMainStepGraph>(
            lm_weights_,
            assets_->config,
            execution_context().backend(),
            execution_context().backend_type(),
            execution_context().config().threads,
            graph_arena_bytes_);
    }
    if (depformer_runtime_ == nullptr) {
        depformer_runtime_ = std::make_unique<PersonaPlexDepformerRuntime>(
            depformer_weights_,
            assets_->config,
            execution_context().backend(),
            execution_context().backend_type(),
            execution_context().config().threads,
            graph_arena_bytes_);
    }
    if (mimi_encoder_ == nullptr) {
        mimi_encoder_ = std::make_unique<engine::codecs::MimiEncoderRuntime>(
            mimi_codec_->weights(),
            mimi_codec_->config(),
            execution_context().backend(),
            execution_context().backend_type(),
            execution_context().config().threads,
            graph_arena_bytes_);
    }
    if (mimi_decoder_ == nullptr) {
        mimi_decoder_ = std::make_unique<engine::codecs::MimiDecoderRuntime>(
            mimi_codec_->weights(),
            mimi_codec_->config(),
            execution_context().backend(),
            execution_context().backend_type(),
            execution_context().config().threads,
            graph_arena_bytes_);
    }
}

std::unique_ptr<PersonaPlexSession::ConversationState> PersonaPlexSession::start_conversation(
    const PersonaPlexGenerationOptions & generation) {
    ensure_runtime_graphs();
    auto state = std::make_unique<ConversationState>();
    state->generation = generation;

    if (generation.seed.has_value()) {
        fallback_rng_.seed(*generation.seed);
    }
    main_step_graph_->reset();
    if (generation.voice_prompt_audio.has_value()) {
        const auto voice_codes = mimi_encoder_->encode(*generation.voice_prompt_audio);
        const int64_t voice_frames = static_cast<int64_t>(voice_codes.size()) / kMimiFrameCodebooks;
        for (int64_t frame = 0; frame < voice_frames; ++frame) {
            const auto step = state->delay_state.prepare(
                kSineTokens.data(),
                voice_codes.data() + static_cast<std::ptrdiff_t>(frame * kMimiFrameCodebooks),
                static_cast<int32_t>(kZeroTextToken));
            if (step.has_value()) {
                run_prepared_token_step(
                    state->delay_state,
                    *main_step_graph_,
                    *depformer_runtime_,
                    generation,
                    sampler_scratch_,
                    fallback_rng_,
                    *step);
            }
        }
    } else {
        const auto voice_prompt = load_personaplex_voice_prompt(*assets_, generation.voice_id);
        const size_t per_embedding = static_cast<size_t>(assets_->config.lm.hidden_size);
        for (int64_t frame = 0; frame < voice_prompt.frames; ++frame) {
            std::vector<float> embedding(per_embedding);
            std::copy_n(
                voice_prompt.embeddings.begin() + static_cast<std::ptrdiff_t>(frame * assets_->config.lm.hidden_size),
                per_embedding,
                embedding.begin());
            run_prepared_embedding_step(
                state->delay_state,
                *main_step_graph_,
                *depformer_runtime_,
                generation,
                sampler_scratch_,
                fallback_rng_,
                embedding);
        }
        state->delay_state.import_cache(voice_prompt);
    }
    engine::debug::trace_log_scalar("personaplex.main_prompt_replay_steps", main_step_graph_->valid_steps());
    engine::debug::trace_log_scalar("personaplex.delay_state.offset_after_prompt", state->delay_state.offset());

    const int silence_frames = static_cast<int>(0.5F * assets_->config.mimi.frame_rate);
    for (int frame = 0; frame < silence_frames; ++frame) {
        const auto step = state->delay_state.prepare(kSineTokens.data(), kSilenceTokens.data(), static_cast<int32_t>(kZeroTextToken));
        if (step.has_value()) {
            run_prepared_token_step(
                state->delay_state,
                *main_step_graph_,
                *depformer_runtime_,
                generation,
                sampler_scratch_,
                fallback_rng_,
                *step);
        }
    }
    const auto prompt = wrap_system_prompt(generation.system_prompt);
    if (!prompt.empty()) {
        const auto prompt_tokens = engine::tokenizers::tokenize_sentencepiece(assets_->tokenizer_pieces, prompt);
        engine::debug::trace_log_scalar("personaplex.system_prompt_tokens", static_cast<int64_t>(prompt_tokens.size()));
        for (const int32_t token : prompt_tokens) {
            const auto step = state->delay_state.prepare(kSineTokens.data(), kSilenceTokens.data(), token);
            if (step.has_value()) {
                run_prepared_token_step(
                    state->delay_state,
                    *main_step_graph_,
                    *depformer_runtime_,
                    generation,
                    sampler_scratch_,
                    fallback_rng_,
                    *step);
            }
        }
    }
    for (int frame = 0; frame < silence_frames; ++frame) {
        const auto step = state->delay_state.prepare(kSineTokens.data(), kSilenceTokens.data(), static_cast<int32_t>(kZeroTextToken));
        if (step.has_value()) {
            run_prepared_token_step(
                state->delay_state,
                *main_step_graph_,
                *depformer_runtime_,
                generation,
                sampler_scratch_,
                fallback_rng_,
                *step);
        }
    }
    engine::debug::trace_log_scalar("personaplex.delay_state.offset_after_system_prompts", state->delay_state.offset());
    return state;
}

std::optional<std::array<int32_t, 8>> PersonaPlexSession::run_user_frame(
    ConversationState & state,
    const int32_t * user_codes) {
    const auto step = state.delay_state.prepare(user_codes, nullptr, std::nullopt);
    if (!step.has_value()) {
        return std::nullopt;
    }
    return run_prepared_token_step(
        state.delay_state,
        *main_step_graph_,
        *depformer_runtime_,
        state.generation,
        sampler_scratch_,
        fallback_rng_,
        *step);
}

runtime::TaskResult PersonaPlexSession::run(const runtime::TaskRequest & request) {
    const auto wall_start = Clock::now();
    require_prepared("PersonaPlex run");
    const auto parse_start = Clock::now();
    runtime::validate_spec_backed_request_options(request.options, *contract_, kModelName);
    const auto parsed = make_personaplex_request(request, *assets_);
    engine::debug::timing_log_scalar("personaplex.request.parse_ms", engine::debug::elapsed_ms(parse_start));

    const auto graph_start = Clock::now();
    ensure_runtime_graphs();
    engine::debug::timing_log_scalar("personaplex.graph_prepare_ms", engine::debug::elapsed_ms(graph_start));

    const auto prompt_start = Clock::now();
    auto state = start_conversation(parsed.generation);
    engine::debug::timing_log_scalar("personaplex.prompt_ms", engine::debug::elapsed_ms(prompt_start));
    const auto user_encode_start = Clock::now();
    const auto user_codes = mimi_encoder_->encode(parsed.audio);
    engine::debug::timing_log_scalar("personaplex.user_encode_ms", engine::debug::elapsed_ms(user_encode_start));

    const auto generate_start = Clock::now();
    const int64_t user_frames = static_cast<int64_t>(user_codes.size()) / kMimiFrameCodebooks;
    std::vector<int32_t> output_codes;
    output_codes.reserve(static_cast<size_t>(user_frames * kMimiFrameCodebooks));
    for (int64_t frame = 0; frame < user_frames; ++frame) {
        const auto out = run_user_frame(
            *state,
            user_codes.data() + static_cast<std::ptrdiff_t>(frame * kMimiFrameCodebooks));
        if (out.has_value()) {
            output_codes.insert(output_codes.end(), out->begin(), out->end());
        }
    }
    engine::debug::timing_log_scalar("personaplex.ar.generate_ms", engine::debug::elapsed_ms(generate_start));
    engine::debug::trace_log_scalar("personaplex.ar.output_frames", static_cast<int64_t>(output_codes.size()) / kMimiFrameCodebooks);

    runtime::TaskResult result;
    const auto decode_start = Clock::now();
    if (!output_codes.empty()) {
        auto audio = mimi_decoder_->decode(
            output_codes,
            static_cast<int64_t>(output_codes.size()) / kMimiFrameCodebooks);
        const int64_t target_samples = resampled_mono_sample_count(parsed.audio, kSampleRate);
        if (target_samples > 0) {
            audio.samples.resize(static_cast<size_t>(target_samples), 0.0F);
        }
        result.audio_output = std::move(audio);
    }
    engine::debug::timing_log_scalar("personaplex.mimi.decode_ms", engine::debug::elapsed_ms(decode_start));
    engine::debug::timing_log_scalar("session.wall_ms", engine::debug::elapsed_ms(wall_start));
    return result;
}

runtime::StreamEvent PersonaPlexSession::process_stream_audio(bool flush) {
    if (stream_state_ == nullptr) {
        throw std::runtime_error("PersonaPlex streaming has not been started");
    }
    runtime::StreamEvent event;
    auto & state = *stream_state_;
    if (state.input_source_frames == 0) {
        return event;
    }

    int64_t target_samples = resampled_mono_sample_count(
        state.input_source_frames,
        state.input_sample_rate,
        kSampleRate);
    int64_t encode_samples = target_samples;
    if (flush) {
        encode_samples = ((encode_samples + kFrameSamples - 1) / kFrameSamples) * kFrameSamples;
    } else {
        encode_samples = (encode_samples / kFrameSamples) * kFrameSamples;
    }
    const int64_t encoded_samples = state.encoded_frames * kFrameSamples;
    std::vector<int32_t> new_output_codes;
    if (encode_samples > encoded_samples) {
        std::vector<int32_t> encoded_prefix;
        const int32_t * all_codes = nullptr;
        int64_t available_frames = 0;
        if (!state.preencoded_user_codes.empty()) {
            available_frames = std::min<int64_t>(
                state.preencoded_user_frames,
                encode_samples / kFrameSamples);
            all_codes = state.preencoded_user_codes.data();
        } else {
            auto mono = engine::audio::convert_interleaved_audio_to_mono_linear_resampled(
                state.raw_audio.samples,
                state.raw_audio.sample_rate,
                state.raw_audio.channels,
                kSampleRate);
            if (mono.empty()) {
                return event;
            }
            mono.resize(static_cast<size_t>(encode_samples), 0.0F);
            runtime::AudioBuffer prefix_audio{kSampleRate, 1, std::move(mono)};
            encoded_prefix = mimi_encoder_->encode(prefix_audio);
            available_frames = static_cast<int64_t>(encoded_prefix.size()) / kMimiFrameCodebooks;
            if (available_frames * kFrameSamples != encode_samples) {
                throw std::runtime_error("PersonaPlex streaming Mimi encoder frame count mismatch");
            }
            all_codes = encoded_prefix.data();
        }
        const auto ar_start = Clock::now();
        for (int64_t frame = state.encoded_frames; frame < available_frames; ++frame) {
            const auto out = run_user_frame(
                state,
                all_codes + static_cast<std::ptrdiff_t>(frame * kMimiFrameCodebooks));
            if (out.has_value()) {
                state.output_codes.insert(state.output_codes.end(), out->begin(), out->end());
                new_output_codes.insert(new_output_codes.end(), out->begin(), out->end());
            }
        }
        state.ar_generate_ms += engine::debug::elapsed_ms(ar_start);
        state.encoded_frames = available_frames;
    }

    if (new_output_codes.empty()) {
        if (flush && !state.final_audio.samples.empty()) {
            if (static_cast<int64_t>(state.final_audio.samples.size()) > target_samples) {
                state.final_audio.samples.resize(static_cast<size_t>(target_samples));
            }
        }
        return event;
    }
    const auto decode_start = Clock::now();
    auto decoded = mimi_decoder_->decode_streaming(
        new_output_codes,
        static_cast<int64_t>(new_output_codes.size()) / kMimiFrameCodebooks);
    state.mimi_decode_ms += engine::debug::elapsed_ms(decode_start);
    if (state.final_audio.samples.empty()) {
        state.final_audio.sample_rate = decoded.sample_rate;
        state.final_audio.channels = decoded.channels;
    }
    state.final_audio.samples.insert(
        state.final_audio.samples.end(),
        decoded.samples.begin(),
        decoded.samples.end());
    if (flush) {
        if (static_cast<int64_t>(state.final_audio.samples.size()) > target_samples) {
            state.final_audio.samples.resize(static_cast<size_t>(target_samples));
        }
    }
    const int64_t visible_samples = static_cast<int64_t>(state.final_audio.samples.size());
    if (visible_samples <= state.emitted_samples) {
        return event;
    }

    runtime::AudioBuffer chunk;
    chunk.sample_rate = state.final_audio.sample_rate;
    chunk.channels = state.final_audio.channels;
    chunk.samples.assign(
        state.final_audio.samples.begin() + static_cast<std::ptrdiff_t>(state.emitted_samples),
        state.final_audio.samples.end());
    state.emitted_samples = visible_samples;
    event.audio_output = std::move(chunk);
    return event;
}

runtime::StreamingPolicy PersonaPlexSession::streaming_policy() const {
    runtime::StreamingPolicy policy;
    policy.input = runtime::StreamingInputKind::AudioChunks;
    policy.output = runtime::StreamingOutputKind::FinalResult;
    policy.preferred_audio_chunk_samples = kFrameSamples;
    policy.preferred_audio_chunk_seconds = static_cast<double>(kFrameSamples) / static_cast<double>(kSampleRate);
    return policy;
}

void PersonaPlexSession::start_stream(const runtime::TaskRequest & request) {
    const auto wall_start = Clock::now();
    require_prepared("PersonaPlex start_stream");
    runtime::validate_spec_backed_request_options(request.options, *contract_, kModelName);
    const auto parse_start = Clock::now();
    const auto generation = make_personaplex_generation_options(request, *assets_);
    engine::debug::timing_log_scalar("personaplex.request.parse_ms", engine::debug::elapsed_ms(parse_start));
    const auto graph_start = Clock::now();
    ensure_runtime_graphs();
    engine::debug::timing_log_scalar("personaplex.graph_prepare_ms", engine::debug::elapsed_ms(graph_start));
    const auto prompt_start = Clock::now();
    stream_state_ = start_conversation(generation);
    stream_state_->wall_start = wall_start;
    mimi_encoder_->reset_streaming();
    mimi_decoder_->reset_streaming();
    engine::debug::timing_log_scalar("personaplex.prompt_ms", engine::debug::elapsed_ms(prompt_start));
    if (request.audio_input.has_value() && !request.audio_input->samples.empty()) {
        const auto user_encode_start = Clock::now();
        stream_state_->preencoded_user_codes = mimi_encoder_->encode(*request.audio_input);
        stream_state_->preencoded_user_frames =
            static_cast<int64_t>(stream_state_->preencoded_user_codes.size()) / kMimiFrameCodebooks;
        engine::debug::timing_log_scalar("personaplex.user_encode_ms", engine::debug::elapsed_ms(user_encode_start));
    }
    stream_request_ = request;
    stream_request_.audio_input = std::nullopt;
    stream_started_ = true;
}

std::optional<runtime::StreamEvent> PersonaPlexSession::next_stream_event() {
    if (!stream_started_) {
        throw std::runtime_error("PersonaPlex streaming has not been started");
    }
    return std::nullopt;
}

void PersonaPlexSession::set_stream_event_sink(runtime::StreamEventCallback sink) {
    stream_sink_ = std::move(sink);
}

runtime::TaskResult PersonaPlexSession::finish_stream() {
    if (!stream_started_) {
        throw std::runtime_error("PersonaPlex streaming has not been started");
    }
    stream_started_ = false;
    if (stream_state_ == nullptr || stream_state_->input_source_frames == 0) {
        throw std::runtime_error("PersonaPlex streaming requires audio chunks before finish_stream");
    }
    if (stream_state_->incremental_user_encoder) {
        runtime::AudioBuffer audio;
        audio.sample_rate = kSampleRate;
        audio.channels = 1;
        const auto codes = mimi_encoder_->encode_streaming(audio, true);
        stream_state_->preencoded_user_codes.insert(
            stream_state_->preencoded_user_codes.end(),
            codes.begin(),
            codes.end());
        stream_state_->preencoded_user_frames =
            static_cast<int64_t>(stream_state_->preencoded_user_codes.size()) / kMimiFrameCodebooks;
    }
    (void)process_stream_audio(true);
    engine::debug::timing_log_scalar("personaplex.ar.generate_ms", stream_state_->ar_generate_ms);
    engine::debug::timing_log_scalar("personaplex.mimi.decode_ms", stream_state_->mimi_decode_ms);
    engine::debug::timing_log_scalar("session.wall_ms", engine::debug::elapsed_ms(stream_state_->wall_start));
    runtime::TaskResult result;
    if (!stream_state_->final_audio.samples.empty()) {
        result.audio_output = std::move(stream_state_->final_audio);
    }
    stream_request_ = runtime::TaskRequest{};
    stream_state_.reset();
    return result;
}

void PersonaPlexSession::reset() {
    stream_request_ = runtime::TaskRequest{};
    stream_state_.reset();
    stream_started_ = false;
}

runtime::StreamEvent PersonaPlexSession::process_audio_chunk(const runtime::AudioChunk & chunk) {
    if (!stream_started_) {
        throw std::runtime_error("PersonaPlex streaming has not been started");
    }
    if (chunk.sample_rate <= 0 || chunk.channels <= 0) {
        throw std::runtime_error("PersonaPlex streaming audio chunks require positive sample rate and channels");
    }
    if (chunk.samples.size() % static_cast<size_t>(chunk.channels) != 0) {
        throw std::runtime_error("PersonaPlex streaming audio chunks must contain whole frames");
    }
    auto & state = *stream_state_;
    const int64_t chunk_frames = static_cast<int64_t>(chunk.samples.size() / static_cast<size_t>(chunk.channels));
    if (state.input_sample_rate == 0) {
        state.input_sample_rate = chunk.sample_rate;
        state.input_channels = chunk.channels;
    } else if (state.input_sample_rate != chunk.sample_rate ||
               state.input_channels != chunk.channels) {
        throw std::runtime_error("PersonaPlex streaming audio chunk format changed during the stream");
    }
    state.input_source_frames += chunk_frames;
    if (state.input_source_frames == chunk_frames &&
        chunk.sample_rate == kSampleRate &&
        chunk.channels == 1 &&
        state.preencoded_user_codes.empty()) {
        state.incremental_user_encoder = true;
    }
    if (state.incremental_user_encoder) {
        runtime::AudioBuffer audio;
        audio.sample_rate = chunk.sample_rate;
        audio.channels = chunk.channels;
        audio.samples = chunk.samples;
        const auto codes = mimi_encoder_->encode_streaming(audio, false);
        state.preencoded_user_codes.insert(state.preencoded_user_codes.end(), codes.begin(), codes.end());
        state.preencoded_user_frames =
            static_cast<int64_t>(state.preencoded_user_codes.size()) / kMimiFrameCodebooks;
    } else if (state.preencoded_user_codes.empty()) {
        auto & audio = state.raw_audio;
        if (audio.sample_rate == 0) {
            audio.sample_rate = chunk.sample_rate;
            audio.channels = chunk.channels;
        }
        audio.samples.insert(audio.samples.end(), chunk.samples.begin(), chunk.samples.end());
    }
    return process_stream_audio(false);
}

runtime::TaskResult PersonaPlexSession::finalize() {
    return finish_stream();
}

std::shared_ptr<runtime::IVoiceModelLoader> make_personaplex_loader() {
    runtime::SpecBackedVoiceModelConfig<PersonaPlexAssets> config;
    config.family = kFamily;
    config.load_assets = load_personaplex_assets;
    config.create_session = [](const runtime::TaskSpec & task,
                                const runtime::SessionOptions & options,
                                std::shared_ptr<const PersonaPlexAssets> assets,
                                std::shared_ptr<const engine::model_spec::ModelContract> contract) {
        return std::make_unique<PersonaPlexSession>(
            task,
            options,
            std::move(assets),
            std::move(contract));
    };
    return runtime::make_spec_backed_voice_loader(std::move(config));
}

}  // namespace engine::models::personaplex
