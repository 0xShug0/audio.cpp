// LFM3-Audio ASR end to end on real weights (see lfm3_audio_model_test.h for the arguments and tolerances):
// - c_16k with language=auto: the fp32 reference's greedy transcript (fixture manifest);
// - chat=true: the reference's chat_nothink_c prompt ids, and an answer without <think> markup.
// q8_0 and q4_k transcripts are reported, not checked.
#include "lfm3_audio_model_test.h"

#include <exception>

namespace {

using namespace lfm3_model_test;

void check_transcript(Checks & checks, const Context & ctx, const std::string & actual, const std::string & expected, const std::string & label) {
    checks.expect_or_info(ctx.strict_text(), actual == expected, label,
                          "\"" + actual + "\"" + (actual == expected ? "" : " vs \"" + expected + "\", WER " + sci(wer(expected, actual))));
}

}  // namespace

int main(int argc, char ** argv) {
    int exit_code = kExitPass;
    auto parsed = parse_context(argc, argv, "test_lfm3_audio_asr", exit_code);
    if (!parsed) {
        return exit_code;
    }
    auto & ctx = *parsed;

    Checks checks;
    try {
        auto asr = ctx.open<lfm3::Lfm3AsrSession>(runtime::VoiceTaskKind::Asr, runtime::RunMode::Offline);
        const auto c16k = read_audio(ctx.fixture("audio/c_16k.wav"));
        check_transcript(checks, ctx, text_of(asr->run(audio_request(c16k))), ctx.result_text("asr_plain_c_greedy_fp32"),
                         "c_16k transcript, language=auto");

        ScopedDumps dumps("asr_chat");
        const auto answer = text_of(asr->run(audio_request(c16k, {{"chat", "true"}})));
        dumps.stop();
        checks.expect(dumps.i32("prompt_ids") == ctx.i32("prompt/chat_nothink_c_ids"), "chat_nothink_c prompt ids");
        checks.expect(!answer.empty() && answer.find("<think>") == std::string::npos && answer.find("</think>") == std::string::npos,
                      "chat answer has text and no <think> markup", std::to_string(answer.size()) + " bytes: \"" + answer.substr(0, 80) + "...\"");
    } catch (const std::exception & error) {
        std::cerr << "FAIL: " << error.what() << "\n";
        return kExitFail;
    }

    return checks.finish("test_lfm3_audio_asr");
}
