// The generator. The mode machine: scripted slot sequences for every
// transition of the upstream generate() loop. The slot loop, on the synthetic
// package (CPU, F32): it equals a hand-written loop over the backbone,
// MaskGIT and the samplers with injected orders, stops at max_steps, sets
// ended, skips the last backbone step, and caps the audio tail with an
// unsampled EOS frame.
#include "engine/community_models/lfm3_audio/assets.h"
#include "engine/community_models/lfm3_audio/debug.h"
#include "engine/community_models/lfm3_audio/generator.h"
#include "engine/community_models/lfm3_audio/prompt.h"
#include "engine/framework/core/execution_context.h"
#include "lfm3_audio_test_package.h"
#include "test_assert.h"

#include <algorithm>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

namespace lfm3 = engine::community_models::lfm3_audio;
using engine::test::require;
using engine::test::require_eq;
using lfm3_audio_test::require_throws_with;
using lfm3::Lfm3Mode;
using lfm3::Lfm3ModeMachine;
using lfm3::Lfm3SlotKind;
using lfm3::Lfm3Step;

constexpr int32_t kEndOfAudio = 2016;
constexpr int64_t kCodebooks = 12;

// The checkpoint's special ids (tokenizer.json added tokens).
lfm3::Lfm3TokenIds token_ids() {
    lfm3::Lfm3TokenIds ids{};
    ids.bos = 124894;
    ids.im_start = 124899;
    ids.im_end = 124900;
    ids.nl = 207;
    ids.text_end = 125014;
    ids.audio_start = 125012;
    ids.mixed_start = 125015;
    ids.audio = 125031;
    ids.audio_out = 125032;
    ids.think = 124901;
    ids.think_end = 124902;
    return ids;
}

// generation_config.json eos_token_id: <|im_end|> and <|text_end|>.
std::vector<int32_t> eos_ids() {
    return {124900, 125014};
}

Lfm3ModeMachine make_machine(Lfm3Mode initial) {
    return Lfm3ModeMachine(initial, token_ids(), eos_ids(), 6, 12, kEndOfAudio);
}

Lfm3Step text(int32_t token) {
    return Lfm3Step{token, {}};
}

Lfm3Step frame(int32_t code = 5) {
    return Lfm3Step{-1, std::vector<int32_t>(kCodebooks, code)};
}

Lfm3Step eos_frame() {
    return frame(kEndOfAudio);
}

void append(std::vector<Lfm3Step> & steps, int64_t count, const Lfm3Step & step) {
    steps.insert(steps.end(), static_cast<size_t>(count), step);
}

void replay(Lfm3ModeMachine & machine, const std::vector<Lfm3Step> & steps, bool ends, const std::string & label) {
    for (size_t i = 0; i < steps.size(); ++i) {
        const auto where = label + " slot " + std::to_string(i);
        require(!machine.finished(), where + ": finished early");
        const auto expected = steps[i].codes.empty() ? Lfm3SlotKind::Text : Lfm3SlotKind::Audio;
        require(machine.next_slot() == expected, where + ": wrong slot kind");
        machine.advance(steps[i]);
    }
    require_eq(machine.finished(), ends, label + " finished");
}

void test_s2s_pattern() {
    std::vector<Lfm3Step> steps;
    append(steps, 6, text(100));
    append(steps, 12, frame());
    append(steps, 6, text(101));
    append(steps, 12, frame());
    append(steps, 5, text(102));
    steps.push_back(text(token_ids().text_end));
    append(steps, 168, frame());
    steps.push_back(eos_frame());
    steps.push_back(text(token_ids().im_end));

    auto machine = make_machine(Lfm3Mode::Interleaved);
    replay(machine, {steps.begin(), steps.end() - 2}, false, "s2s");
    require(machine.mode() == Lfm3Mode::Interleaved, "s2s tail stays interleaved");
    replay(machine, {steps.end() - 2, steps.end() - 1}, false, "s2s eos frame");
    require(machine.mode() == Lfm3Mode::Text, "s2s EOS frame returns to text");
    replay(machine, {steps.end() - 1, steps.end()}, true, "s2s im_end");
}

// TTS: the prompt ends with <|audio_start|>.
void test_tts_pattern() {
    std::vector<Lfm3Step> steps;
    append(steps, 14, frame());
    steps.push_back(eos_frame());
    auto machine = make_machine(Lfm3Mode::Audio);
    replay(machine, steps, false, "tts");
    require(machine.mode() == Lfm3Mode::Text, "tts EOS frame returns to text");
    replay(machine, {text(token_ids().im_end)}, true, "tts im_end");
}

// An EOS frame inside an audio block before <|text_end|>: back to Text, where
// tokens no longer count against blocks and <|text_end|> ends the turn.
void test_eos_frame_before_text_end() {
    std::vector<Lfm3Step> steps;
    append(steps, 6, text(100));
    append(steps, 5, frame());
    steps.push_back(eos_frame());
    append(steps, 20, text(101));
    auto machine = make_machine(Lfm3Mode::Interleaved);
    replay(machine, steps, false, "early eos");
    require(machine.mode() == Lfm3Mode::Text, "early eos mode");
    replay(machine, {text(token_ids().text_end)}, true, "text_end in text mode");

    // Only one code at end_of_audio already makes the EOS frame.
    auto partial = make_machine(Lfm3Mode::Audio);
    auto codes = frame();
    codes.codes[7] = kEndOfAudio;
    replay(partial, {frame(), codes}, false, "partial eos");
    require(partial.mode() == Lfm3Mode::Text, "one EOS code ends the audio");
}

// <|im_end|> in an interleaved text slot ends the turn; <|text_end|>, though an
// eos id, only ends the text there.
void test_im_end_in_interleaved() {
    auto machine = make_machine(Lfm3Mode::Interleaved);
    replay(machine, {text(1), text(2), text(3), text(token_ids().im_end)}, true, "interleaved im_end");

    std::vector<Lfm3Step> steps;
    append(steps, 6, text(1));
    append(steps, 12, frame());
    append(steps, 2, text(2));
    steps.push_back(text(token_ids().im_end));
    auto second = make_machine(Lfm3Mode::Interleaved);
    replay(second, steps, true, "second block im_end");
}

// <|audio_start|> and <|mixed_start|> switch modes in Text only; inside a block
// they are plain text and count as text slots.
void test_mode_markers() {
    const auto ids = token_ids();

    auto audio = make_machine(Lfm3Mode::Text);
    replay(audio, {text(1), text(2), text(ids.audio_start)}, false, "audio_start");
    require(audio.mode() == Lfm3Mode::Audio, "audio_start opens audio");
    replay(audio, {frame(), frame(), eos_frame(), text(3), text(ids.im_end)}, true, "audio_start turn");

    std::vector<Lfm3Step> steps = {text(1), text(ids.mixed_start)};
    append(steps, 6, text(2));
    append(steps, 12, frame());
    append(steps, 6, text(3));
    append(steps, 3, frame());
    steps.push_back(eos_frame());
    // Back in Text, a second <|mixed_start|> opens a fresh block.
    steps.push_back(text(ids.mixed_start));
    append(steps, 6, text(4));
    append(steps, 12, frame());
    steps.push_back(text(ids.im_end));
    auto mixed = make_machine(Lfm3Mode::Text);
    replay(mixed, steps, true, "mixed_start");

    std::vector<Lfm3Step> inside = {text(ids.audio_start), text(ids.mixed_start), text(1), text(2), text(3), text(4)};
    append(inside, 12, frame());
    append(inside, 1, text(5));
    auto block = make_machine(Lfm3Mode::Interleaved);
    replay(block, inside, false, "markers inside a block");
    require(block.mode() == Lfm3Mode::Interleaved, "markers inside a block keep the mode");
}

// max_audio_tail_frames: after 3 tail frames the next slot must be the EOS
// frame, which returns to Text.
void test_audio_tail_cap() {
    std::vector<Lfm3Step> steps;
    append(steps, 6, text(1));
    append(steps, 12, frame());
    append(steps, 2, text(2));
    steps.push_back(text(token_ids().text_end));
    auto machine = make_machine(Lfm3Mode::Interleaved);
    replay(machine, steps, false, "tail head");
    for (int i = 0; i < 3; ++i) {
        require(!machine.force_audio_eos(3), "tail frame " + std::to_string(i) + " is sampled");
        require(machine.next_slot() == Lfm3SlotKind::Audio, "tail slot is audio");
        machine.advance(frame());
    }
    require(machine.force_audio_eos(3), "cap of 3 forces the EOS frame");
    require(!machine.force_audio_eos(0), "cap 0 never forces");
    require(!machine.force_audio_eos(4), "cap 4 not reached yet");
    machine.advance(eos_frame());
    require(machine.next_slot() == Lfm3SlotKind::Text, "after the forced EOS the next slot is text");
    require(!machine.force_audio_eos(3), "no force in text mode");

    // Frames before <|text_end|> are not tail frames.
    auto early = make_machine(Lfm3Mode::Interleaved);
    std::vector<Lfm3Step> head;
    append(head, 6, text(1));
    append(head, 12, frame());
    replay(early, head, false, "no tail yet");
    require(!early.force_audio_eos(1), "regular block frames are not a tail");
}

void test_errors() {
    const auto ids = token_ids();
    require_throws_with([&] { Lfm3ModeMachine(Lfm3Mode::Text, ids, eos_ids(), 0, 12); }, "positive", "zero text block");
    require_throws_with([&] { Lfm3ModeMachine(Lfm3Mode::Text, ids, eos_ids(), 6, 0); }, "positive", "zero audio block");

    auto machine = make_machine(Lfm3Mode::Text);
    require_throws_with([&] { machine.advance(frame()); }, "frame for a text slot", "frame in text");
    auto audio = make_machine(Lfm3Mode::Audio);
    require_throws_with([&] { audio.advance(text(1)); }, "text token for an audio slot", "text in audio");
    machine.advance(text(ids.im_end));
    require_throws_with([&] { machine.advance(text(1)); }, "past the end", "advance after the end");
}

void test_suppress() {
    const auto ids = token_ids();
    require(lfm3::lfm3_text_suppress(ids) == std::vector<int32_t>{ids.audio, ids.audio_out}, "the placeholders are suppressed");
}

struct Synthetic {
    lfm3_audio_test::Package package;
    std::shared_ptr<const lfm3::Lfm3AudioAssets> assets;
    std::unique_ptr<lfm3::Lfm3Tokenizer> tokenizer;
    std::unique_ptr<engine::core::ExecutionContext> execution;
    std::unique_ptr<lfm3::Lfm3BackboneRuntime> backbone;
    std::unique_ptr<lfm3::Lfm3DepthDecoderRuntime> depth;
};

Synthetic make_synthetic() {
    Synthetic s;
    s.package = lfm3_audio_test::write_package("lfm3_audio_generator_test");
    s.assets = lfm3::load_lfm3_audio_assets(s.package.root);
    s.tokenizer = std::make_unique<lfm3::Lfm3Tokenizer>(s.assets->tokenizer_json);
    s.execution = std::make_unique<engine::core::ExecutionContext>(engine::core::BackendConfig{engine::core::BackendType::Cpu, 0, 2});
    const lfm3::Lfm3WeightOptions f32{lfm3::Lfm3WeightType::F32, true};
    s.backbone = std::make_unique<lfm3::Lfm3BackboneRuntime>(s.assets->model, s.assets->backbone, s.assets->depth.codebooks,
                                                              s.assets->depth.audio_vocab_size, *s.execution, f32);
    s.depth = std::make_unique<lfm3::Lfm3DepthDecoderRuntime>(s.assets->model, s.assets->depth, *s.execution, f32);
    return s;
}

std::unique_ptr<lfm3::Lfm3Generator> make_generator(Synthetic & s, const lfm3::Lfm3Prompt & prompt, lfm3::Lfm3GenerationOptions options,
                                                    bool with_depth = true) {
    return std::make_unique<lfm3::Lfm3Generator>(*s.backbone, with_depth ? s.depth.get() : nullptr, *s.tokenizer, s.assets->generation,
                                                 s.assets->depth, prompt, lfm3::Lfm3AudioEmbeddings{}, std::move(options));
}

bool same_step(const Lfm3Step & a, const Lfm3Step & b) {
    return a.token == b.token && a.codes == b.codes;
}

std::vector<Lfm3Step> run_all(lfm3::Lfm3Generator & generator) {
    std::vector<Lfm3Step> out;
    while (auto step = generator.next()) out.push_back(*step);
    const auto & kept = generator.steps();
    require(kept.size() == out.size() && std::equal(out.begin(), out.end(), kept.begin(), same_step), "steps() holds every emitted step");
    return out;
}

// TTS-shaped turn (audio from the first slot): the generator equals the loop written out by hand, with the
// first two orders injected and the rest drawn from the sampler, greedy codes and a repetition penalty (so
// committed history matters).
void test_matches_manual_loop(Synthetic & s) {
    const auto prompt = lfm3::make_lfm3_tts_prompt(*s.tokenizer, "Perform TTS in English.", "the user");
    const int32_t eoa = s.assets->depth.end_of_audio();
    lfm3::Lfm3GenerationOptions options;
    options.max_steps = 6;
    options.audio.temperature = 0.0f;
    options.audio.repetition_penalty = 1.5f;
    options.seed = 7;
    options.stop_after_audio_eos = true;
    options.maskgit_orders = {{11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0}, {0, 2, 4, 6, 8, 10, 1, 3, 5, 7, 9, 11}};

    auto generator = make_generator(s, prompt, options);
    const auto steps = run_all(*generator);
    const int64_t after = s.backbone->position();

    lfm3::Lfm3AudioSampler sampler(options.audio, options.seed);
    auto hidden = s.backbone->prefill(prompt, s.tokenizer->ids().audio, s.tokenizer->ids().audio_out, {}, options.max_steps,
                                      lfm3::Lfm3Output::Hidden);
    std::vector<Lfm3Step> expected;
    for (size_t i = 0; i < static_cast<size_t>(options.max_steps); ++i) {
        const auto order = i < options.maskgit_orders.size() ? options.maskgit_orders[i] : sampler.next_order();
        const auto codes = lfm3::lfm3_maskgit_frame(*s.depth, hidden, order, s.assets->depth.maskgit_schedule, sampler, eoa);
        expected.push_back(Lfm3Step{-1, {codes.begin(), codes.end()}});
        if (codes[0] == eoa) break;
        sampler.commit(codes);
        if (i + 1 < static_cast<size_t>(options.max_steps)) hidden = s.backbone->step_frame(expected.back().codes, lfm3::Lfm3Output::Hidden);
    }

    require_eq(steps.size(), expected.size(), "manual loop steps");
    for (size_t i = 0; i < steps.size(); ++i) {
        require(same_step(steps[i], expected[i]), "manual loop step " + std::to_string(i) + " differs");
    }
    require_eq(generator->ended(), expected.back().codes[0] == eoa, "manual loop ended");
    // The last step does not go through the backbone.
    require_eq(after, static_cast<int64_t>(prompt.ids.size() + steps.size()) - 1, "manual loop backbone position");
    require(!generator->next().has_value(), "manual loop: nothing after the end");
    std::cout << "  manual loop: equal, injected orders used, last step skipped\n";
}

// Interleaved turn where every text slot can only pick <|text_end|>: it ends the text at once, the tail is
// capped at 2 sampled frames, the 3rd audio slot is the EOS frame without sampling, and <|text_end|> in Text
// ends the turn. (With these random weights the 3rd sampled tail frame would itself be an EOS frame.)
void test_tail_cap_and_ended(Synthetic & s) {
    const auto & ids = s.tokenizer->ids();
    const int32_t eoa = s.assets->depth.end_of_audio();
    const lfm3::Lfm3Prompt prompt{s.tokenizer->encode("<|startoftext|><|im_start|>assistant\n<|mixed_start|>"), {}};
    lfm3::Lfm3GenerationOptions options;
    options.max_steps = 32;
    options.audio.temperature = 0.0f;
    options.max_audio_tail_frames = 2;
    for (int32_t id = 0; id < s.assets->backbone.vocab_size; ++id) {
        if (id != ids.text_end) options.suppress.push_back(id);
    }

    // The maskgit_orders dump has a row per sampled frame.
    const auto dumps = lfm3_audio_test::fresh_directory("lfm3_audio_generator_test_dumps");
    lfm3_audio_test::set_env("LFM3_AUDIO_DUMP_DIR", dumps.string());
    auto generator = make_generator(s, prompt, options);
    const auto steps = run_all(*generator);
    lfm3_audio_test::set_env("LFM3_AUDIO_DUMP_DIR", "");
    require_eq(steps.size(), size_t{5}, "tail cap steps");
    require_eq(steps[0].token, ids.text_end, "tail cap: text ends first");
    for (size_t i = 1; i <= 2; ++i) {
        require(!steps[i].codes.empty() && steps[i].codes[0] != eoa, "tail frame " + std::to_string(i) + " is sampled speech");
    }
    require(steps[3].codes == std::vector<int32_t>(12, eoa), "the capped tail ends with the EOS frame");
    std::vector<int64_t> shape;
    (void)lfm3::lfm3_read_npy_i32(dumps / "maskgit_orders.npy", &shape);
    require(shape == std::vector<int64_t>{2, 12}, "the EOS frame is not sampled");
    require_eq(steps[4].token, ids.text_end, "tail cap: <|text_end|> in text ends the turn");
    require(generator->ended(), "tail cap: ended");
    require_eq(s.backbone->position(), static_cast<int64_t>(prompt.ids.size()) + 4, "tail cap: last step skipped");

    options.stop_after_audio_eos = true;
    generator = make_generator(s, prompt, options);
    require_eq(run_all(*generator).size(), size_t{4}, "stop_after_audio_eos steps");
    require(generator->ended(), "stop_after_audio_eos: ended");

    options.stop_after_audio_eos = false;
    options.max_steps = 3;
    generator = make_generator(s, prompt, options);
    require_eq(run_all(*generator).size(), size_t{3}, "max_steps steps");
    require(!generator->ended(), "max_steps: not ended");
    require_eq(s.backbone->position(), static_cast<int64_t>(prompt.ids.size()) + 2, "max_steps: last step skipped");
    std::cout << "  tail cap, stop rules and ended: ok\n";
}

void test_generator_errors(Synthetic & s) {
    const auto prompt = lfm3::make_lfm3_tts_prompt(*s.tokenizer, "", "x");
    lfm3::Lfm3GenerationOptions options;
    require_throws_with([&] { make_generator(s, prompt, options); }, "max_steps", "max_steps 0");
    options.max_steps = 4;
    auto text_only = make_generator(s, prompt, options, false);
    require_throws_with([&] { text_only->next(); }, "without a depth decoder", "audio slot without depth");
}

}  // namespace

int main() {
    try {
        test_s2s_pattern();
        test_tts_pattern();
        test_eos_frame_before_text_end();
        test_im_end_in_interleaved();
        test_mode_markers();
        test_audio_tail_cap();
        test_errors();
        test_suppress();
        std::cout << "lfm3_audio mode machine: scripted patterns ok\n";
        {
            auto synthetic = make_synthetic();
            test_matches_manual_loop(synthetic);
            test_tail_cap_and_ended(synthetic);
            test_generator_errors(synthetic);
        }
        std::cout << "lfm3_audio generator: synthetic slot loop ok\n";
    } catch (const std::exception & error) {
        std::cerr << "lfm3_audio_generator_test failed: " << error.what() << "\n";
        return 1;
    }
    return 0;
}
