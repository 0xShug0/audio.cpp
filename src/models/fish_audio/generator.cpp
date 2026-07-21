#include "engine/models/fish_audio/generator.h"

#include "engine/framework/debug/profiler.h"

#include <chrono>
#include <stdexcept>
#include <utility>

namespace engine::models::fish_audio {
namespace {

using Clock = std::chrono::steady_clock;

}  // namespace

FishAudioGenerator::FishAudioGenerator(
    std::shared_ptr<const FishAudioAssets> assets,
    std::unique_ptr<FishAudioARRuntime> ar,
    std::unique_ptr<FishAudioCodecRuntime> codec)
    : assets_(std::move(assets)),
      tokenizer_(assets_),
      prompt_builder_(assets_, tokenizer_),
      ar_(std::move(ar)),
      codec_(std::move(codec)) {
    if (assets_ == nullptr || ar_ == nullptr || codec_ == nullptr) {
        throw std::runtime_error("Fish Audio generator requires assets, AR runtime, and codec runtime");
    }
}

FishAudioGenerator::~FishAudioGenerator() = default;

FishAudioCodes FishAudioGenerator::encode_reference(const runtime::AudioBuffer & audio) {
    return codec_->encode_reference(audio);
}

FishAudioGenerationResult FishAudioGenerator::generate(
    const FishAudioRequest & request,
    const std::optional<FishAudioCodes> & reference_codes,
    const std::optional<FishAudioConversationTurn> & previous_turn,
    bool mem_saver) {
    engine::debug::trace_log_scalar("fish_audio.request.has_reference", request.reference.has_value());
    engine::debug::trace_log_scalar("fish_audio.request.text_chars", static_cast<int64_t>(request.text.size()));
    engine::debug::trace_log_scalar("fish_audio.request.has_previous_turn", previous_turn.has_value());
    const auto prompt_start = Clock::now();
    const auto prompt = prompt_builder_.build(request, reference_codes, previous_turn);
    engine::debug::timing_log_scalar(
        "fish_audio.prompt_build_ms",
        engine::debug::elapsed_ms(prompt_start, Clock::now()));

    const auto ar_start = Clock::now();
    FishAudioGenerationResult result;
    result.codes = ar_->generate(prompt, request.generation);
    engine::debug::trace_log_scalar("fish_audio.generated.frames", result.codes.frames);
    engine::debug::trace_log_scalar("fish_audio.generated.codebooks", result.codes.codebooks);
    engine::debug::timing_log_scalar(
        "fish_audio.ar_generate_ms",
        engine::debug::elapsed_ms(ar_start, Clock::now()));

    const auto decode_start = Clock::now();
    result.audio = codec_->decode(result.codes);
    engine::debug::timing_log_scalar(
        "fish_audio.codec_decode_ms",
        engine::debug::elapsed_ms(decode_start, Clock::now()));
    if (mem_saver) {
        ar_->release_runtime_graphs();
        codec_->release_runtime_graphs();
    }
    return result;
}

}  // namespace engine::models::fish_audio
