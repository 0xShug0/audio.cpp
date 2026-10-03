// The ASR prompt and the audio preparation, two places where a mistake still
// produces fluent but wrong transcripts.
#include "engine/community_models/lfm2_audio/asr_inputs.h"
#include "lfm2_audio_test_package.h"
#include "test_assert.h"

#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

namespace {

namespace lfm2 = engine::community_models::lfm2_audio;
namespace runtime = engine::runtime;
using engine::test::require;
using engine::test::require_close;
using engine::test::require_eq;
using lfm2_audio_test::require_throws_with;

constexpr double kPi = 3.14159265358979323846;

lfm2::Lfm2TextVocabulary text_vocabulary() {
    const auto vocab = lfm2_audio_test::byte_level_vocabulary();
    lfm2::Lfm2TextVocabulary out;
    out.tokens = vocab.tokens;
    out.token_types = vocab.types;
    out.merges = vocab.merges;
    out.pre_tokenizer = "lfm2";
    return out;
}

// The token strings of `ids`, space-separated.
std::string spell(const lfm2::Lfm2TextVocabulary & vocab, const std::vector<int32_t> & ids) {
    std::string out;
    for (const int32_t id : ids) {
        out += (out.empty() ? "" : " ") + vocab.tokens.at(static_cast<size_t>(id));
    }

    return out;
}

// ChatState encodes "<|im_start|>system\n", the system prompt and each other
// piece on its own, so no BPE merge crosses a piece boundary, and
// "assistant" is the vocabulary's user-defined token.
void test_prompt_pieces() {
    const auto vocab = text_vocabulary();
    const lfm2::Lfm2TextTokenizer tokenizer(vocab);

    const auto en = lfm2::make_lfm2_asr_prompt(tokenizer, "en");
    require_eq(spell(vocab, en.prefix),
        std::string("<|startoftext|> <|im_start|> s y s t e m Ċ P e r f o r m ĠA S R . <|im_end|> Ċ <|im_start|> u s e r Ċ"),
        "English prefix");
    require_eq(spell(vocab, en.suffix), std::string("<|im_end|> Ċ <|im_start|> assistant Ċ"), "suffix");
    require_eq(spell(vocab, en.stop_token_ids), std::string("<|im_end|> <|audio_start|>"), "stop tokens");

    const auto ja = lfm2::make_lfm2_asr_prompt(tokenizer, "ja");
    require_eq(spell(vocab, ja.prefix),
        std::string("<|startoftext|> <|im_start|> s y s t e m Ċ P e r f o r m ĠA S R Ġ i n Ġ j a p a n e s e . "
                    "<|im_end|> Ċ <|im_start|> u s e r Ċ"),
        "Japanese prefix");
    require_eq(spell(vocab, ja.suffix), spell(vocab, en.suffix), "Japanese suffix");

    require_throws_with([&] { (void)lfm2::make_lfm2_asr_prompt(tokenizer, "de"); }, "no ASR prompt", "language de");
}

void test_audio_positions() {
    const auto vocab = text_vocabulary();
    const lfm2::Lfm2TextTokenizer tokenizer(vocab);
    const auto prompt = lfm2::make_lfm2_asr_prompt(tokenizer, "en");

    const auto filled = prompt.with_audio(5);
    require_eq(filled.input_ids.size(), prompt.prefix.size() + 5 + prompt.suffix.size(), "prompt length");
    require_eq(filled.audio_positions.size(), size_t{5}, "audio positions");
    for (size_t i = 0; i < 5; ++i) {
        require_eq(filled.audio_positions[i], static_cast<int32_t>(prompt.prefix.size() + i), "audio position " + std::to_string(i));
    }

    const auto prefix_end = filled.input_ids.begin() + static_cast<std::ptrdiff_t>(prompt.prefix.size());
    const auto suffix_begin = filled.input_ids.end() - static_cast<std::ptrdiff_t>(prompt.suffix.size());
    require(std::vector<int32_t>(filled.input_ids.begin(), prefix_end) == prompt.prefix &&
            std::vector<int32_t>(suffix_begin, filled.input_ids.end()) == prompt.suffix,
        "the audio goes between the prefix and the suffix");

    const auto empty = prompt.with_audio(0);
    require(empty.audio_positions.empty() && empty.input_ids.size() == prompt.prefix.size() + prompt.suffix.size(),
        "a prompt without audio");
}

// 0.1 s at 44.1 kHz: a 440 Hz tone and two tones above the 8 kHz Nyquist
// frequency of the output, which the resampler's low-pass has to remove.
std::vector<float> tones_44k() {
    std::vector<float> out(4410);
    for (size_t i = 0; i < out.size(); ++i) {
        const double t = static_cast<double>(i) / 44100.0;
        out[i] = static_cast<float>(0.4 * std::sin(2.0 * kPi * 440.0 * t) + 0.2 * std::sin(2.0 * kPi * 6000.0 * t) +
                                    0.1 * std::sin(2.0 * kPi * 12000.0 * t));
    }

    return out;
}

// torchaudio.functional.resample(x, 44100, 16000) on the float32 samples, as
// liquid-audio's ChatState.add_audio calls it.
struct ResampledPoint {
    size_t index;
    float value;
};
const ResampledPoint kTorchaudio[] = {
    {0, 0.09094977f}, {1, 0.18527219f}, {7, 0.23517486f}, {100, -0.40009186f}, {800, -2.97e-09f}, {1599, -0.18527202f},
};

void test_resampling() {
    const auto samples = tones_44k();
    const auto mono = lfm2::lfm2_audio_mono_16k({44100, 1, samples});
    require_eq(mono.size(), size_t{1600}, "resampled length");
    for (const auto & point : kTorchaudio) {
        require_close(mono[point.index], point.value, 1e-5f, "resampled sample " + std::to_string(point.index));
    }

    // Channels are averaged before resampling.
    std::vector<float> stereo;
    for (const float sample : samples) {
        stereo.push_back(sample);
        stereo.push_back(0.5f * sample);
    }

    const auto mixed = lfm2::lfm2_audio_mono_16k({44100, 2, stereo});
    for (const auto & point : kTorchaudio) {
        require_close(mixed[point.index], 0.75f * point.value, 1e-5f, "mixed sample " + std::to_string(point.index));
    }
}

void test_16k_passes_through() {
    const std::vector<float> samples = {0.25f, -0.5f, 0.125f, 1.0f};
    require(lfm2::lfm2_audio_mono_16k({16000, 1, samples}) == samples, "16 kHz mono is used as is");

    const auto mixed = lfm2::lfm2_audio_mono_16k({16000, 2, {1.0f, 0.0f, -0.5f, 0.5f}});
    require(mixed == std::vector<float>({0.5f, 0.0f}), "16 kHz stereo is only averaged");
}

void test_rejects_bad_audio() {
    require_throws_with([] { (void)lfm2::lfm2_audio_mono_16k({16000, 1, {}}); }, "non-empty audio", "empty audio");
    require_throws_with([] { (void)lfm2::lfm2_audio_mono_16k({16000, 0, {0.1f}}); }, "non-empty audio", "zero channels");
    require_throws_with(
        [] { (void)lfm2::lfm2_audio_mono_16k({16000, 1, {0.1f, std::numeric_limits<float>::quiet_NaN()}}); },
        "non-finite", "a NaN sample");
    require_throws_with(
        [] { (void)lfm2::lfm2_audio_mono_16k({44100, 1, {0.1f, -std::numeric_limits<float>::infinity()}}); },
        "non-finite", "an Inf sample");
}

}  // namespace

int main() {
    try {
        test_prompt_pieces();
        test_audio_positions();
        test_resampling();
        test_16k_passes_through();
        test_rejects_bad_audio();
        std::cout << "lfm2_audio_asr_inputs_test: PASS\n";
        return 0;
    } catch (const std::exception & error) {
        std::cerr << "lfm2_audio_asr_inputs_test: " << error.what() << '\n';
        return 1;
    }
}
