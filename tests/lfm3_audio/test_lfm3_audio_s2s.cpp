// LFM3-Audio speech-to-speech on real weights (see lfm3_audio_model_test.h for the arguments and tolerances):
// - greedy replies to c_16k with the fixture's MaskGIT orders injected: the slot sequence that ends on its own
//   (the fixture's slot pattern) and the one cut at max_tokens=96, whose reply artifact says ended=false;
// - streaming gives the offline text, reply artifact and audio for a sampled reply;
// - turn 2 after a replayed turn 1 (the full reply, and the cut one closed with an EOS frame and <|im_end|>):
//   the prompt ids and the prefill logits of the fixture's re-prefill (gen/multiturn_*).
#include "lfm3_audio_model_test.h"

#include "engine/community_models/lfm3_audio/chat.h"
#include "engine/framework/io/binary.h"

#include <exception>
#include <numeric>

namespace {

using namespace lfm3_model_test;

const lfm3::Lfm3ReplyCheckpoint kCheckpoint{};

void check_greedy(Checks & checks, const Context & ctx, lfm3::Lfm3ChatSession & s2s, const std::string & tag, const std::string & max_tokens) {
    ScopedDumps dumps(tag, ctx.fixture("gen/" + tag + "_frame_order.npy"));
    const auto result = s2s.run(audio_request(read_audio(ctx.fixture("audio/c_16k.wav")),
                                              {{"temperature", "0"}, {"repetition_penalty", "1"}, {"max_tokens", max_tokens}, {"return_codes", "true"},
                                               {"seed", "42"}}));  // MaskGIT orders past the fixture's frames are random
    dumps.stop();
    checks.expect(dumps.i32("prompt_ids") == ctx.i32("prompt/s2s_c_ids"), tag + " prompt ids");
    const bool same = check_greedy_run(checks, ctx, dumps, tag, false);

    const auto & artifact = result.output_artifacts.at(0);
    const auto steps = lfm3::read_lfm3_reply_artifact(artifact, kCheckpoint);
    const bool cut = tag.find("trunc") != std::string::npos;
    // A run that parted from the reference may end elsewhere; the artifact must still say how it ended.
    checks.expect(artifact.meta.at("ended") == (s2s.reached_max_tokens() ? "false" : "true"), tag + " reply artifact ended matches the run",
                  "ended=" + artifact.meta.at("ended") + ", " + std::to_string(steps.size()) + " steps");
    checks.expect_or_info(same, s2s.reached_max_tokens() == cut, tag + std::string(cut ? " is cut at max_tokens" : " ends on its own"));
    if (!cut) {
        checks.expect_or_info(same, text_of(result) == ctx.result_text(tag), tag + " reply text", "\"" + text_of(result) + "\"");
    }
}

void check_stream(Checks & checks, Context & ctx, lfm3::Lfm3ChatSession & s2s) {
    const auto request = audio_request(read_audio(ctx.fixture("audio/c_16k.wav")), {{"seed", "42"}, {"max_tokens", "200"}, {"return_codes", "true"}});
    const auto offline = s2s.run(request);
    auto streaming = ctx.open<lfm3::Lfm3ChatSession>(runtime::VoiceTaskKind::SpeechToSpeech, runtime::RunMode::Streaming);
    streaming->start_stream(request);
    const auto & audio = *request.audio_input;
    for (size_t at = 0; at < audio.samples.size(); at += 1600) {
        const auto end = std::min(audio.samples.size(), at + 1600);
        streaming->process_audio_chunk({audio.sample_rate, 1, static_cast<int64_t>(at), {audio.samples.begin() + at, audio.samples.begin() + end}});
    }
    std::string text;
    std::vector<float> samples;
    while (auto event = streaming->next_stream_event()) {
        if (event->partial_text) text += event->partial_text->text;
        if (event->audio_output) samples.insert(samples.end(), event->audio_output->samples.begin(), event->audio_output->samples.end());
    }
    const auto finished = streaming->finish_stream();
    checks.expect(text == text_of(offline) && text_of(finished) == text, "s2s streaming text == offline", "\"" + text + "\"");
    checks.expect(finished.output_artifacts.at(0).payload == offline.output_artifacts.at(0).payload, "s2s streaming reply artifact == offline");
    const auto & want = offline.audio_output->samples;
    checks.expect(samples.size() == want.size(), "s2s streaming audio length", std::to_string(samples.size()) + " vs " + std::to_string(want.size()));
    if (samples.size() == want.size()) {
        checks.expect_rel(diff(samples, want), ctx.tolerance("codec.tts_wave", true), "s2s streaming audio == offline");
    }
}

void check_turn2(Checks & checks, const Context & ctx, lfm3::Lfm3ChatSession & s2s, const std::string & turn1, const std::string & tag) {
    std::vector<lfm3::Lfm3Step> reply;
    for (const auto & slot : fixture_slots(ctx, turn1)) {
        reply.push_back({slot.token, slot.codes});
    }
    const bool ended = !reply.empty() && reply.back().codes.empty();  // ends on <|im_end|>; a cut reply ends inside its audio
    auto request = audio_request(read_audio(ctx.fixture("audio/a_16k.wav")), {{"temperature", "0"}, {"max_tokens", "7"}});
    request.input_artifacts = {
        runtime::make_voice_artifact(runtime::ArtifactKind::Custom, lfm3::kLfm3QuestionArtifactId,
                                     engine::io::read_binary_file(ctx.fixture("audio/c_16k.wav"))),
        lfm3::make_lfm3_reply_artifact(reply, ended, kCheckpoint)};
    ScopedDumps dumps(tag);
    s2s.run(request);
    dumps.stop();

    const auto ids = dumps.i32("prompt_ids");
    const auto want_ids = ctx.i32("gen/" + tag + "_prompt_ids");
    checks.expect(ids == want_ids, tag + " turn-2 prompt ids", std::to_string(ids.size()) + " vs " + std::to_string(want_ids.size()));
    const auto logits = dumps.f32("prefill_logits");
    const auto want = ctx.f32("gen/" + tag + "_last_logits");
    if (logits.size() != want.size()) {
        checks.expect(false, tag + " turn-2 prefill logits", "size " + std::to_string(logits.size()));
        return;
    }
    std::vector<int32_t> order(logits.size());
    std::iota(order.begin(), order.end(), 0);
    std::partial_sort(order.begin(), order.begin() + 20, order.end(), [&](int32_t a, int32_t b) { return logits[a] > logits[b]; });
    const auto top20 = ctx.i32("gen/" + tag + "_top20_ids");
    const auto overlap = std::count_if(order.begin(), order.begin() + 20, [&](int32_t id) { return std::find(top20.begin(), top20.end(), id) != top20.end(); });
    checks.expect_rel(diff(logits, want), ctx.tolerance("backbone.last_logits"), tag + " turn-2 prefill logits",
                      ", argmax " + std::to_string(order[0]) + " vs " + std::to_string(top20[0]) + ", top-20 overlap " + std::to_string(overlap) + "/20");
    checks.expect_or_info(ctx.profile != Profile::Q4_K, order[0] == top20[0], tag + " turn-2 prefill argmax");
}

}  // namespace

int main(int argc, char ** argv) {
    int exit_code = kExitPass;
    auto parsed = parse_context(argc, argv, "test_lfm3_audio_s2s", exit_code);
    if (!parsed) {
        return exit_code;
    }
    auto & ctx = *parsed;

    Checks checks;
    try {
        auto s2s = ctx.open<lfm3::Lfm3ChatSession>(runtime::VoiceTaskKind::SpeechToSpeech, runtime::RunMode::Offline);
        check_greedy(checks, ctx, *s2s, "s2s_c_greedy", "1024");
        check_greedy(checks, ctx, *s2s, "s2s_c_greedy_trunc", "96");
        check_stream(checks, ctx, *s2s);
        check_turn2(checks, ctx, *s2s, "s2s_c_greedy", "multiturn_s2s_turn2");
        check_turn2(checks, ctx, *s2s, "s2s_c_greedy_trunc", "multiturn_s2s_trunc_turn2");
    } catch (const std::exception & error) {
        std::cerr << "FAIL: " << error.what() << "\n";
        return kExitFail;
    }

    return checks.finish("test_lfm3_audio_s2s");
}
