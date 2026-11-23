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

struct PersonaPlexSession::InferenceState {
    PersonaPlexGenerationOptions generation;
    PersonaPlexDelayState delay_state;
    double user_encode_ms = 0.0;
    double ar_generate_ms = 0.0;
    double mimi_decode_ms = 0.0;

    ~InferenceState() {
        debug::timing_log_scalar("personaplex.user_encode_ms", user_encode_ms);
        debug::timing_log_scalar("personaplex.ar.generate_ms", ar_generate_ms);
        debug::timing_log_scalar("personaplex.mimi.decode_ms", mimi_decode_ms);
    }

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
    runtime::StreamingPolicy policy{runtime::StreamingInputKind::AudioChunks, runtime::StreamingOutputKind::FinalResult,
        kFrameSamples, static_cast<double>(kFrameSamples) / kSampleRate};
    bind_chat(kModelName, [this](const runtime::TaskRequest & request, chat::HistoryMode) {
        require_prepared("PersonaPlex chat");
        const auto parse_start = Clock::now();
        runtime::validate_spec_backed_request_options(request.options, *contract_, kModelName);
        const auto generation = make_personaplex_generation_options(request, *assets_);
        debug::timing_log_scalar("personaplex.request.parse_ms", debug::elapsed_ms(parse_start));
        const auto graph_start = Clock::now();
        ensure_runtime_graphs();
        debug::timing_log_scalar("personaplex.graph_prepare_ms", debug::elapsed_ms(graph_start));
        const auto prompt_start = Clock::now();
        std::shared_ptr<InferenceState> state = start_conversation(generation);
        debug::timing_log_scalar("personaplex.prompt_ms", debug::elapsed_ms(prompt_start));
        chat::UserTurn turn;
        turn.output_sample_rate = kSampleRate;
        turn.duplex = [this, state] {
            chat::DuplexSource source;
            source.sample_rate = kSampleRate;
            source.frame_samples = kFrameSamples;
            source.codebooks = kMimiFrameCodebooks;
            source.encode = [this, state](const runtime::AudioBuffer & audio) {
                const auto start = Clock::now();
                auto codes = mimi_encoder_->encode(audio);
                state->user_encode_ms += debug::elapsed_ms(start);
                return codes;
            };
            source.encode_stream = [this, state](const runtime::AudioBuffer & audio, bool flush) {
                const auto start = Clock::now();
                auto codes = mimi_encoder_->encode_streaming(audio, flush);
                state->user_encode_ms += debug::elapsed_ms(start);
                return codes;
            };
            source.reset_codec = [this] { mimi_encoder_->reset_streaming(); mimi_decoder_->reset_streaming(); };
            source.predict = [this, state](const int32_t * input, int32_t * codes) {
                const auto start = Clock::now();
                const auto output = run_user_frame(*state, input);
                state->ar_generate_ms += debug::elapsed_ms(start);
                if (output) std::copy(output->begin(), output->end(), codes);
                return output.has_value();
            };
            source.decode = [this, state](const std::vector<int32_t> & codes, int64_t frames) {
                const auto start = Clock::now();
                auto audio = mimi_decoder_->decode(codes, frames);
                state->mimi_decode_ms += debug::elapsed_ms(start);
                return audio;
            };
            source.decode_stream = [this, state](const std::vector<int32_t> & codes, int64_t frames) {
                const auto start = Clock::now();
                auto audio = mimi_decoder_->decode_streaming(codes, frames);
                state->mimi_decode_ms += debug::elapsed_ms(start);
                return audio;
            };
            return std::make_unique<chat::Duplex>(std::move(source));
        };
        return turn;
    }, policy);
}

PersonaPlexSession::~PersonaPlexSession() { chat::TaskSession::reset(); }

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

std::unique_ptr<PersonaPlexSession::InferenceState> PersonaPlexSession::start_conversation(
    const PersonaPlexGenerationOptions & generation) {
    ensure_runtime_graphs();
    auto state = std::make_unique<InferenceState>();
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
    InferenceState & state,
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
