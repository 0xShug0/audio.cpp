#include "engine/community_models/audio8_tts/generator.h"

#include "engine/framework/debug/profiler.h"

#include <chrono>
#include <stdexcept>
#include <utility>

namespace engine::models::audio8_tts {
namespace {

using Clock = std::chrono::steady_clock;

}  // namespace

Audio8TtsGenerator::Audio8TtsGenerator(
    std::shared_ptr<const Audio8TtsAssets> assets,
    std::unique_ptr<Audio8TtsARRuntime> ar,
    std::unique_ptr<Audio8TtsCodecRuntime> codec)
    : assets_(std::move(assets)),
      tokenizer_(assets_),
      prompt_builder_(assets_, tokenizer_),
      ar_(std::move(ar)),
      codec_(std::move(codec)) {
    if (assets_ == nullptr || ar_ == nullptr || codec_ == nullptr) {
        throw std::runtime_error("Audio8 TTS generator requires assets, AR runtime, and codec runtime");
    }
}

Audio8TtsGenerator::~Audio8TtsGenerator() = default;

Audio8TtsCodes Audio8TtsGenerator::encode_reference(const runtime::AudioBuffer & audio) {
    auto codes = codec_->encode_reference(audio);
    codec_->release_encode_graph();
    return codes;
}

Audio8TtsGenerationResult Audio8TtsGenerator::generate(
    const Audio8TtsRequest & request,
    const std::vector<Audio8TtsCodes> & reference_codes,
    const std::optional<Audio8TtsConversationTurn> & previous_turn,
    bool mem_saver) {
    engine::debug::trace_log_scalar("audio8_tts.request.has_reference", !request.references.empty());
    engine::debug::trace_log_scalar("audio8_tts.request.reference_count", static_cast<int64_t>(request.references.size()));
    engine::debug::trace_log_scalar("audio8_tts.request.text_chars", static_cast<int64_t>(request.text.size()));
    engine::debug::trace_log_scalar("audio8_tts.request.has_previous_turn", previous_turn.has_value());
    engine::debug::trace_log_scalar("audio8_tts.sampler.seed", request.generation.seed);
    const auto prompt_start = Clock::now();
    const auto prompt = prompt_builder_.build(request, reference_codes, previous_turn);
    engine::debug::timing_log_scalar(
        "audio8_tts.prompt_build_ms",
        engine::debug::elapsed_ms(prompt_start, Clock::now()));

    const auto ar_start = Clock::now();
    Audio8TtsGenerationResult result;
    result.codes = ar_->generate(prompt, request.generation);
    engine::debug::trace_log_scalar("audio8_tts.generated.frames", result.codes.frames);
    engine::debug::trace_log_scalar("audio8_tts.generated.codebooks", result.codes.codebooks);
    engine::debug::timing_log_scalar(
        "audio8_tts.ar_generate_ms",
        engine::debug::elapsed_ms(ar_start, Clock::now()));

    const auto decode_start = Clock::now();
    result.audio = codec_->decode(result.codes);
    engine::debug::timing_log_scalar(
        "audio8_tts.codec_decode_ms",
        engine::debug::elapsed_ms(decode_start, Clock::now()));
    codec_->release_runtime_graphs();
    if (mem_saver) {
        ar_->release_runtime_graphs();
    }
    return result;
}

}  // namespace engine::models::audio8_tts
