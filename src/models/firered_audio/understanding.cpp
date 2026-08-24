#include "engine/models/firered_audio/understanding.h"

#include "engine/framework/debug/profiler.h"
#include "engine/framework/sampling/hf_sampler.h"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>

namespace engine::models::firered_audio {
namespace {

namespace assets = engine::assets;
namespace core = engine::core;

using Clock = std::chrono::steady_clock;
constexpr int64_t kDecodeCacheGrowthWindow = 1024;

void replace_audio_embeddings(
    std::vector<float> & embeddings,
    const std::vector<uint8_t> & audio_mask,
    const std::vector<float> & audio_embeddings,
    int64_t hidden_size) {
    if (hidden_size <= 0 || embeddings.size() % static_cast<size_t>(hidden_size) != 0) {
        throw std::runtime_error("FireRedAudio understanding embedding shape is invalid");
    }
    const int64_t steps = static_cast<int64_t>(embeddings.size()) / hidden_size;
    if (static_cast<int64_t>(audio_mask.size()) != steps) {
        throw std::runtime_error("FireRedAudio understanding audio mask size mismatch");
    }
    int64_t audio_step = 0;
    for (int64_t step = 0; step < steps; ++step) {
        if (audio_mask[static_cast<size_t>(step)] == 0) {
            continue;
        }
        const auto src_offset = static_cast<size_t>(audio_step * hidden_size);
        const auto dst_offset = static_cast<size_t>(step * hidden_size);
        if (src_offset + static_cast<size_t>(hidden_size) > audio_embeddings.size()) {
            throw std::runtime_error("FireRedAudio understanding audio embedding replacement underrun");
        }
        std::copy(
            audio_embeddings.data() + src_offset,
            audio_embeddings.data() + src_offset + hidden_size,
            embeddings.data() + dst_offset);
        ++audio_step;
    }
    if (audio_step * hidden_size != static_cast<int64_t>(audio_embeddings.size())) {
        throw std::runtime_error("FireRedAudio understanding audio embedding replacement size mismatch");
    }
}

std::string strip_thinking(std::string text) {
    constexpr std::string_view marker = "</think>";
    const auto pos = text.find(marker);
    if (pos == std::string::npos) {
        return text;
    }
    text.erase(0, pos + marker.size());
    while (!text.empty() && (text.front() == '\n' || text.front() == '\r' || text.front() == ' ' || text.front() == '\t')) {
        text.erase(text.begin());
    }
    return text;
}

}  // namespace

class FireRedAudioUnderstandingRuntime::Impl {
public:
    Impl(
        std::shared_ptr<const FireRedAudioAssets> assets,
        core::ExecutionContext & execution,
        size_t graph_arena_bytes,
        size_t weight_context_bytes,
        assets::TensorStorageType storage_type,
        bool mem_saver)
        : assets_(std::move(assets)),
          tokenizer_(std::make_unique<FireRedAudioTokenizer>(assets_)),
          audio_encoder_(std::make_unique<FireRedAudioAudioEncoderRuntime>(
              assets_,
              execution,
              graph_arena_bytes,
              weight_context_bytes,
              storage_type)),
          qwen_(std::make_unique<FireRedAudioQwen35Runtime>(
              assets_,
              execution,
              graph_arena_bytes,
              weight_context_bytes,
              storage_type)),
          mem_saver_(mem_saver) {
        if (assets_ == nullptr) {
            throw std::runtime_error("FireRedAudio understanding runtime requires assets");
        }
    }

    FireRedAudioUnderstandingResult generate(const FireRedAudioUnderstandingRequest & request) {
        const auto audio_start = Clock::now();
        auto audio = audio_encoder_->encode(request.audio);
        const auto audio_end = Clock::now();
        if (audio.tokens <= 0 || audio.tokens != static_cast<int64_t>(audio.embeddings.size()) / assets_->backbone.hidden_size) {
            throw std::runtime_error("FireRedAudio understanding audio encoder output shape mismatch");
        }
        const auto prompt_start = Clock::now();
        auto encoded = tokenizer_->encode_understanding_prompt(request.prompt, audio.tokens, request.enable_thinking);
        const auto prompt_end = Clock::now();
        int64_t masked_tokens = 0;
        for (uint8_t value : encoded.audio_mask) {
            masked_tokens += value != 0 ? 1 : 0;
        }
        if (masked_tokens != audio.tokens) {
            throw std::runtime_error("FireRedAudio understanding prompt/audio token count mismatch");
        }
        const auto token_embedding_start = Clock::now();
        auto embeddings = qwen_->token_embedding(encoded.token_ids);
        const auto token_embedding_end = Clock::now();
        const auto replace_start = Clock::now();
        replace_audio_embeddings(embeddings, encoded.audio_mask, audio.embeddings, assets_->backbone.hidden_size);
        const auto replace_end = Clock::now();
        const int64_t prompt_steps = static_cast<int64_t>(embeddings.size()) / assets_->backbone.hidden_size;
        const auto decode_session_start = Clock::now();
        auto decode = qwen_->create_decode_session(prompt_steps + std::min(request.max_new_tokens, kDecodeCacheGrowthWindow) + 1);
        const auto decode_session_end = Clock::now();
        const auto prefill_start = Clock::now();
        auto prefill = decode->prefill_embeddings(embeddings, prompt_steps);
        std::vector<float> hidden(
            prefill.hidden.end() - static_cast<std::ptrdiff_t>(assets_->backbone.hidden_size),
            prefill.hidden.end());
        const auto prefill_end = Clock::now();
        std::vector<int32_t> generated;
        std::vector<float> logits;
        std::mt19937 rng(request.seed);
        engine::sampling::HfSampler sampler;
        engine::sampling::HfSamplerScratch scratch;
        engine::sampling::HfSamplingOptions sampling_options;
        sampling_options.do_sample = request.do_sample;
        sampling_options.temperature = request.temperature;
        sampling_options.top_k = request.top_k;
        sampling_options.top_p = request.top_p;
        sampling_options.repetition_penalty = request.repetition_penalty;
        double lm_head_ms = 0.0;
        double sample_ms = 0.0;
        double token_embedding_ms = 0.0;
        double qwen_step_ms = 0.0;
        const auto decode_start = Clock::now();
        for (int64_t step = 0; step < request.max_new_tokens; ++step) {
            const auto lm_head_start = Clock::now();
            logits = qwen_->lm_head(hidden);
            lm_head_ms += engine::debug::elapsed_ms(lm_head_start, Clock::now());
            const auto sample_start = Clock::now();
            const int32_t token = sampler.sample(
                logits,
                generated,
                sampling_options,
                scratch,
                rng,
                nullptr,
                "FireRedAudio understanding sampler");
            sample_ms += engine::debug::elapsed_ms(sample_start, Clock::now());
            if (token == assets_->special_tokens.im_end || token == assets_->special_tokens.endoftext) {
                break;
            }
            generated.push_back(token);
            const auto embedding_start = Clock::now();
            auto next_embedding = qwen_->token_embedding({token});
            token_embedding_ms += engine::debug::elapsed_ms(embedding_start, Clock::now());
            const auto step_start = Clock::now();
            hidden = decode->run_embedding_step(next_embedding);
            qwen_step_ms += engine::debug::elapsed_ms(step_start, Clock::now());
        }
        const auto decode_end = Clock::now();
        FireRedAudioUnderstandingResult result;
        result.text = strip_thinking(tokenizer_->decode(generated, false));
        result.language = request.language;
        engine::debug::timing_log_scalar("firered_audio.understanding.audio_encoder_ms", engine::debug::elapsed_ms(audio_start, audio_end));
        engine::debug::timing_log_scalar("firered_audio.understanding.prompt_tokenize_ms", engine::debug::elapsed_ms(prompt_start, prompt_end));
        engine::debug::timing_log_scalar("firered_audio.understanding.prompt_token_embedding_ms", engine::debug::elapsed_ms(token_embedding_start, token_embedding_end));
        engine::debug::timing_log_scalar("firered_audio.understanding.audio_embedding_replace_ms", engine::debug::elapsed_ms(replace_start, replace_end));
        engine::debug::timing_log_scalar("firered_audio.understanding.decode_session_ms", engine::debug::elapsed_ms(decode_session_start, decode_session_end));
        engine::debug::timing_log_scalar("firered_audio.understanding.prefill_ms", engine::debug::elapsed_ms(prefill_start, prefill_end));
        engine::debug::timing_log_scalar("firered_audio.understanding.decode_ms", engine::debug::elapsed_ms(decode_start, decode_end));
        engine::debug::timing_log_scalar("firered_audio.understanding.generated_tokens", static_cast<double>(generated.size()));
        engine::debug::timing_log_scalar("firered_audio.understanding.qwen_step_ms", qwen_step_ms);
        engine::debug::timing_log_scalar("firered_audio.understanding.lm_head_ms", lm_head_ms);
        engine::debug::timing_log_scalar("firered_audio.understanding.sample_ms", sample_ms);
        engine::debug::timing_log_scalar("firered_audio.understanding.token_embedding_ms", token_embedding_ms);
        if (mem_saver_) {
            release_graphs();
        }
        return result;
    }

    void release_graphs() {
        audio_encoder_->release_graphs();
        qwen_->release_graphs();
    }

private:
    std::shared_ptr<const FireRedAudioAssets> assets_;
    std::unique_ptr<FireRedAudioTokenizer> tokenizer_;
    std::unique_ptr<FireRedAudioAudioEncoderRuntime> audio_encoder_;
    std::unique_ptr<FireRedAudioQwen35Runtime> qwen_;
    bool mem_saver_ = false;
};

FireRedAudioUnderstandingRuntime::FireRedAudioUnderstandingRuntime(
    std::shared_ptr<const FireRedAudioAssets> assets,
    engine::core::ExecutionContext & execution,
    size_t graph_arena_bytes,
    size_t weight_context_bytes,
    engine::assets::TensorStorageType storage_type,
    bool mem_saver)
    : impl_(std::make_unique<Impl>(
          std::move(assets),
          execution,
          graph_arena_bytes,
          weight_context_bytes,
          storage_type,
          mem_saver)) {}

FireRedAudioUnderstandingRuntime::~FireRedAudioUnderstandingRuntime() = default;

FireRedAudioUnderstandingResult FireRedAudioUnderstandingRuntime::generate(const FireRedAudioUnderstandingRequest & request) {
    return impl_->generate(request);
}

void FireRedAudioUnderstandingRuntime::release_graphs() {
    impl_->release_graphs();
}

}  // namespace engine::models::firered_audio
