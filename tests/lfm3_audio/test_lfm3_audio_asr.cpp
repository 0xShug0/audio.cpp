// LFM3-Audio ASR end to end on real weights (see lfm3_audio_model_test.h for the arguments and tolerances):
// - c_16k with language=auto: the fp32 reference's greedy transcript (fixture manifest);
// - chat=true: the reference's chat_nothink_c prompt ids, and an answer without <think> markup.
// q8_0 and q4_k transcripts are reported, not checked.
#include "lfm3_audio_model_test.h"
#include "engine/community_models/lfm3_audio/prompt.h"

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

        const auto answer = text_of(asr->run(audio_request(c16k, {{"chat", "true"}})));
        const auto assets = lfm3::load_lfm3_audio_assets(ctx.model);
        const lfm3::Lfm3Tokenizer tokenizer(*assets);
        const auto prompt = lfm3::make_lfm3_chat_prompt(tokenizer, lfm3::lfm3_question_tokens(c16k), false);
        checks.expect(prompt.ids == ctx.i32("prompt/chat_nothink_c_ids"), "chat_nothink_c prompt ids");
        checks.expect(!answer.empty() && answer.find("<think>") == std::string::npos && answer.find("</think>") == std::string::npos,
                      "chat answer has text and no <think> markup", std::to_string(answer.size()) + " bytes: \"" + answer.substr(0, 80) + "...\"");
    } catch (const std::exception & error) {
        std::cerr << "FAIL: " << error.what() << "\n";
        return kExitFail;
    }

    return checks.finish("test_lfm3_audio_asr");
}
