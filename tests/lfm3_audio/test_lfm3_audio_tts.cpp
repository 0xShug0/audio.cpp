// LFM3-Audio TTS end to end on real weights (see lfm3_audio_model_test.h for the arguments and tolerances):
// - greedy speech of the fixture's fox sentence with its MaskGIT orders injected: prompt ids, the frames up to
//   the fixture's EOS frame, the orders and per-pass depth logits, and with identical frames the codec's codes;
// - the default sampling is deterministic for a fixed seed, and streaming (1 and 4 frames per event) gives
//   the offline audio;
// - five English sentences spoken and transcribed back by the ASR session: WER <= 10% overall;
// - every built-in voice speaks a short sentence (--skip-voices leaves this out).
#include "lfm3_audio_model_test.h"

#include "engine/community_models/lfm3_audio/prompt.h"

#include <exception>

namespace {

using namespace lfm3_model_test;

constexpr double kRoundTripWer = 0.10;

const char * const kRoundTrip[] = {
    "The weather is lovely today, so we will walk to the park.",
    "Please send me the report before noon on Friday.",
    "She bought three apples and a loaf of fresh bread.",
    "Our train leaves at half past seven from platform two.",
    "Learning a new language takes patience and practice.",
};

runtime::TaskRequest tts_request(const std::string & text, std::unordered_map<std::string, std::string> options, const std::string & voice = "") {
    auto request = text_request(text, std::move(options));
    if (!voice.empty()) {
        request.voice = runtime::VoiceCondition{runtime::VoiceReference{std::nullopt, voice}, std::nullopt};
    }
    return request;
}

std::vector<float> stream_audio(lfm3::Lfm3TtsSession & session, const runtime::TaskRequest & request) {
    session.start_stream(request);
    std::vector<float> out;
    while (auto event = session.next_stream_event()) {
        if (event->audio_output) out.insert(out.end(), event->audio_output->samples.begin(), event->audio_output->samples.end());
    }
    (void)session.finish_stream();
    return out;
}

void check_greedy(Checks & checks, const Context & ctx, lfm3::Lfm3TtsSession & tts) {
    ScopedDumps dumps("tts_greedy", ctx.fixture("gen/tts_fox_greedy_frame_order.npy"));
    tts.run(tts_request("The quick brown fox jumps over the lazy dog",
                        {{"temperature", "0"}, {"repetition_penalty", "1"}, {"max_tokens", "400"}, {"seed", "42"}}, "american_english_female"));
    dumps.stop();
    checks.expect(dumps.i32("prompt_ids") == ctx.i32("prompt/tts_fox_ids"), "tts_fox prompt ids");
    if (!check_greedy_run(checks, ctx, dumps, "tts_fox_greedy", true)) {
        return;
    }

    // The same frames, so the codec saw the fixture's codes.
    std::vector<int64_t> shape;
    const auto codes = dumps.i32("codes", &shape);
    const auto want = ctx.i32("codec/tts_codes", &shape);  // [12, T]
    const auto count = static_cast<size_t>(shape[1]);
    bool same = codes.size() == want.size();
    for (size_t f = 0; same && f < count; ++f) {
        for (size_t k = 0; k < kCodebooks; ++k) same = same && codes[f * kCodebooks + k] == want[k * count + f];
    }
    checks.expect(same, "tts_fox codes dump equals codec/tts_codes");
}

void check_sampling_and_stream(Checks & checks, Context & ctx, lfm3::Lfm3TtsSession & tts) {
    const auto request = tts_request("Good morning, and welcome to the show.", {{"seed", "42"}});
    const auto first = tts.run(request).audio_output->samples;
    const auto second = tts.run(request).audio_output->samples;
    checks.expect(!first.empty() && first == second, "sampled speech is deterministic for seed 42", std::to_string(first.size()) + " samples");

    // The codec's stream ends each event on final samples, so the events add up to the offline audio (up to the
    // numerics of other batch shapes outside the CPU F32 mode).
    const double tolerance = ctx.tolerance("codec.tts_wave", true);
    auto streaming = ctx.open<lfm3::Lfm3TtsSession>(runtime::VoiceTaskKind::Tts, runtime::RunMode::Streaming);
    for (const char * frames : {"1", "4"}) {
        auto stream_request = request;
        stream_request.options["stream_frames_per_event"] = frames;
        const auto streamed = stream_audio(*streaming, stream_request);
        checks.expect(streamed.size() == first.size(), std::string("streaming length, ") + frames + " frames per event",
                      std::to_string(streamed.size()) + " vs " + std::to_string(first.size()));
        if (streamed.size() == first.size()) {
            checks.expect_rel(diff(streamed, first), tolerance, std::string("streaming == offline, ") + frames + " frames per event");
        }
    }
}

void check_round_trip(Checks & checks, Context & ctx, lfm3::Lfm3TtsSession & tts) {
    auto asr = ctx.open<lfm3::Lfm3AsrSession>(runtime::VoiceTaskKind::Asr, runtime::RunMode::Offline);
    size_t errors = 0, words = 0;
    int seed = 42;
    for (const char * sentence : kRoundTrip) {
        auto speech = *tts.run(tts_request(sentence, {{"seed", std::to_string(seed++)}}, "american_english_female")).audio_output;
        const auto heard = text_of(asr->run(audio_request(std::move(speech), {{"language", "en"}})));
        const auto count = wer_words(sentence).size();
        const double score = wer(sentence, heard);
        errors += static_cast<size_t>(std::llround(score * static_cast<double>(count)));
        words += count;
        checks.info("round trip", "WER " + sci(score) + " \"" + heard + "\"");
    }
    const double total = static_cast<double>(errors) / static_cast<double>(words);
    checks.expect_or_info(ctx.strict_text(), total <= kRoundTripWer, "TTS -> ASR round trip on 5 sentences",
                          "WER " + sci(total) + " (" + std::to_string(errors) + "/" + std::to_string(words) + " words, limit " + sci(kRoundTripWer) + ")");
}

// Sampled speech can end early (portuguese_male with seed 1 says one frame here, while HF fp32 gave 16-48 frames
// over 64 seeds), so a short reply is reported; a voice fails only without speech or when cut at max_tokens.
void check_voices(Checks & checks, lfm3::Lfm3TtsSession & tts) {
    size_t spoken = 0;
    std::string failed, short_ones;
    for (const auto & voice : lfm3::lfm3_tts_voices()) {
        const auto audio = tts.run(tts_request("Hello there, nice to meet you.", {{"seed", "1"}}, voice.id)).audio_output;
        const double seconds = audio ? static_cast<double>(audio->samples.size()) / audio->sample_rate : 0.0;
        if (seconds <= 0.0 || tts.reached_max_tokens()) {
            failed += " " + voice.id + "(" + sci(seconds) + " s)";
            continue;
        }
        ++spoken;
        if (seconds < 0.5) short_ones += " " + voice.id + "(" + sci(seconds) + " s)";
    }
    checks.expect(failed.empty(), "every built-in voice speaks and ends", std::to_string(spoken) + "/" + std::to_string(lfm3::lfm3_tts_voices().size()) + failed);
    if (!short_ones.empty()) checks.info("voices with under 0.5 s of speech for seed 1", short_ones);
}

}  // namespace

int main(int argc, char ** argv) {
    int exit_code = kExitPass;
    auto parsed = parse_context(argc, argv, "test_lfm3_audio_tts", exit_code);
    if (!parsed) {
        return exit_code;
    }
    auto & ctx = *parsed;

    Checks checks;
    try {
        auto tts = ctx.open<lfm3::Lfm3TtsSession>(runtime::VoiceTaskKind::Tts, runtime::RunMode::Offline);
        check_greedy(checks, ctx, *tts);
        check_sampling_and_stream(checks, ctx, *tts);
        check_round_trip(checks, ctx, *tts);
        if (has_flag(argc, argv, "--skip-voices")) {
            checks.info("voices skipped (--skip-voices)");
        } else {
            check_voices(checks, *tts);
        }
    } catch (const std::exception & error) {
        std::cerr << "FAIL: " << error.what() << "\n";
        return kExitFail;
    }

    return checks.finish("test_lfm3_audio_tts");
}
