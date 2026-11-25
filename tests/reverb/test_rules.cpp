// Self-contained checks of the Reverb frontend and sampler schedule against values produced by
// reverb-tts 1.0.1 / PyTorch (see README.md for how they were generated).
#include "engine/community_models/reverb/frontend.h"
#include "engine/community_models/reverb/generator.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

using namespace engine::community_models::reverb;

namespace {

int failures = 0;

void check(bool ok, const std::string & what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what.c_str());
        ++failures;
    }
}

}  // namespace

int main() {
    // torch.linspace(0, 1, steps + 1) warped by t_shift = 0.1, as float32 then .tolist().
    const std::vector<double> t8 = {0x0.0p+0, 0x1.cd856ap-7, 0x1.084212p-5, 0x1.cfb2b8p-5, 0x1.745d18p-4,
        0x1.24924ap-3, 0x1.d89d86p-3, 0x1.a5a5a0p-2, 0x1.fffff8p-1};
    const std::vector<double> t12 = {0x0.0p+0, 0x1.27350cp-7, 0x1.414142p-6, 0x1.084212p-5, 0x1.861864p-5,
        0x1.111112p-4, 0x1.745d18p-4, 0x1.f7047cp-4, 0x1.555552p-3, 0x1.d89d86p-3, 0x1.555550p-2, 0x1.0c30c2p-1,
        0x1.fffff8p-1};
    check(reverb_time_steps(8, 0.1F) == t8, "time steps, 8 passes");
    check(reverb_time_steps(12, 0.1F) == t12, "time steps, 12 passes");

    // reverb_tts.text.estimate_frames
    check(estimate_frames("Hello there.", "Some call me nature.", 100) == 62, "estimate_frames short text");
    std::string many;
    for (int i = 0; i < 20; ++i) {
        many += "Zażółć gęślą jaźń 12, 3!";
    }
    check(estimate_frames(many, "Ref text.", 30) == 2127, "estimate_frames with digits, punctuation, diacritics");

    // Python round(): halves to even.
    check(round_half_even(0.5) == 0 && round_half_even(1.5) == 2 && round_half_even(2.5) == 2 &&
              round_half_even(-0.5) == 0 && round_half_even(2.6) == 3,
        "round_half_even");

    // prepare_text: brackets dropped, Unicode whitespace collapsed.
    check(prepare_text("  Hello  (big)\t[world] {x}\n") == "Hello big world x", "prepare_text");
    check(is_supported_language("pl") && !is_supported_language("nl"), "languages");

    // split_text keeps a short text whole and cuts a long one at sentence ends.
    check(split_text("One. Two.", "Some reference words.", 50).size() == 1, "split_text short");
    std::string longer;
    for (int i = 0; i < 40; ++i) {
        longer += "This sentence is here to make the text long enough to split. ";
    }
    const auto pieces = split_text(prepare_text(longer), "Some reference words here.", 40);
    check(pieces.size() > 1, "split_text long");
    for (const auto & piece : pieces) {
        check(!piece.empty() && piece.back() == '.', "split_text cuts at sentence ends");
    }

    // trim_silence: a 120 ms burst alone at the start is dropped, speech kept with 100 ms margins.
    const int sr = 24000;
    std::vector<float> audio(static_cast<size_t>(sr * 2), 0.0F);
    for (int i = 0; i < sr / 10; ++i) {
        audio[static_cast<size_t>(i)] = 0.3F * std::sin(0.05F * static_cast<float>(i));  // burst
    }
    for (int i = sr / 2; i < sr; ++i) {
        audio[static_cast<size_t>(i)] = 0.3F * std::sin(0.03F * static_cast<float>(i));  // speech
    }
    const auto trimmed = trim_silence(audio, sr);
    check(trimmed.size() == static_cast<size_t>(sr / 2 + 2 * sr / 10), "trim_silence drops the edge burst");

    // raise_quiet leaves normal levels alone and lifts a very quiet clip to about -20 dBFS.
    std::vector<float> quiet(static_cast<size_t>(sr), 0.0F);
    for (int i = 0; i < sr; ++i) {
        quiet[static_cast<size_t>(i)] = 0.001F * std::sin(0.02F * static_cast<float>(i));
    }
    const auto raised = raise_quiet(quiet);
    double sum = 0.0;
    for (float v : raised) {
        sum += static_cast<double>(v) * v;
    }
    check(std::fabs(20.0 * std::log10(std::sqrt(sum / sr)) + 20.0) < 0.01, "raise_quiet level");
    check(raise_quiet(audio) == audio, "raise_quiet leaves loud audio unchanged");

    if (failures == 0) {
        std::puts("reverb_rules_test: all checks passed");
    }
    return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
