// ZONOS2 prompt rows, conditioning ids and quality_buckets parsing; with an oracle directory, equality with
// zonos2-cli --build-prompt (make_oracle.sh) and with ZONOS2's long-form continuation prompts
// (make_longform_oracle.py).
#include "engine/community_models/zonos2/npy.h"
#include "engine/community_models/zonos2/prompt.h"
#include "engine/framework/runtime/errors.h"

#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace zonos2 = engine::models::zonos2;

namespace {

void expect(bool condition, const std::string & message) {
    if (!condition) throw std::runtime_error(message);
}

void expect_rejected(const std::function<void()> & action, const std::string & needle, const std::string & what) {
    try {
        action();
    } catch (const engine::runtime::InvalidRequestError & error) {
        expect(std::string(error.what()).find(needle) != std::string::npos, what + ": unexpected message: " + error.what());
        return;
    }
    throw std::runtime_error(what + " was accepted");
}

// The Zyphra/ZONOS2 release values.
zonos2::Zonos2Hparams release_hparams() {
    zonos2::Zonos2Hparams hp;
    hp.n_layer = 28;
    hp.n_embd = 2048;
    hp.head_dim = 128;
    hp.n_head = 16;
    hp.n_head_kv = 4;
    hp.n_ff = 3072;
    hp.n_expert = 16;
    hp.n_ctx_train = 6144;
    hp.n_codebooks = 9;
    hp.codebook_size = 1024;
    hp.audio_vocab = 1026;
    hp.eoa_id = 1024;
    hp.audio_pad_id = 1025;
    hp.text_vocab = 519;
    hp.spk_dim = 2048;
    hp.spk_lda_dim = 1024;
    return hp;
}

int text_at(const zonos2::Zonos2Prompt & p, int row) { return p.ids[static_cast<size_t>(row) * 10 + 9]; }
int audio_at(const zonos2::Zonos2Prompt & p, int row, int cb) { return p.ids[static_cast<size_t>(row) * 10 + static_cast<size_t>(cb)]; }

void expect_oracle(const zonos2::Zonos2Prompt & prompt, const std::filesystem::path & path) {
    const auto oracle = zonos2::read_f32_npy(path);
    expect(oracle.shape == std::vector<int64_t>({prompt.n_rows, 10}),
           path.string() + ": shape differs from the built prompt (" + std::to_string(prompt.n_rows) + " rows)");
    for (size_t i = 0; i < prompt.ids.size(); ++i) {
        expect(static_cast<float>(prompt.ids[i]) == oracle.data[i],
               path.string() + ": id " + std::to_string(i) + " differs (" + std::to_string(prompt.ids[i]) + " vs " +
                   std::to_string(oracle.data[i]) + ")");
    }
}

}  // namespace

int main(int argc, char ** argv) {
    try {
        const auto hp = release_hparams();
        const std::string hello = "Hello.";

        const auto plain = zonos2::build_zonos2_prompt(hp, hello, {});
        expect(plain.n_rows == 26 && plain.speaker_row == -1, "plain prompt has 26 rows and no speaker");
        expect(text_at(plain, 0) == 511, "default quality row is trailing_silence_s:3 = 511");
        expect(text_at(plain, 1) == 2, "BOS");
        for (int i = 0; i < 6; ++i) {
            expect(text_at(plain, 2 + i) == 192 + static_cast<unsigned char>(hello[static_cast<size_t>(i)]), "byte ids");
        }
        expect(text_at(plain, 8) == 3, "EOS");
        for (int r = 9; r < 26; ++r) expect(text_at(plain, r) == 519, "silence rows carry the text pad");
        expect(audio_at(plain, 9, 0) == 568 && audio_at(plain, 9, 1) == 1025, "first sheared silence row");
        expect(audio_at(plain, 25, 8) == 731 && audio_at(plain, 17, 8) == 90, "sheared silence tail");
        for (int cb = 0; cb < 9; ++cb) expect(audio_at(plain, 0, cb) == 1025, "conditioning rows are audio pad");

        zonos2::Zonos2PromptOptions clone;
        clone.speaker = true;
        const auto spk = zonos2::build_zonos2_prompt(hp, hello, clone);
        expect(spk.speaker_row == 0 && spk.n_rows == 29, "speaker prompt adds three rows");
        expect(text_at(spk, 0) == 519 && text_at(spk, 1) == 517 && text_at(spk, 2) == 518,
               "speaker slot, noisy background, accurate marker");
        clone.clean_speaker_background = true;
        clone.accurate_mode = false;
        const auto expressive = zonos2::build_zonos2_prompt(hp, hello, clone);
        expect(expressive.n_rows == 28 && text_at(expressive, 1) == 516 && text_at(expressive, 2) == 511,
               "clean background, no accurate row");

        zonos2::Zonos2PromptOptions rate;
        rate.speaking_rate_bucket = 3;
        const auto rated = zonos2::build_zonos2_prompt(hp, hello, rate);
        expect(text_at(rated, 0) == 451 && text_at(rated, 1) == 511, "speaking-rate row precedes the quality row");

        zonos2::Zonos2PromptOptions none;
        none.quality_buckets = zonos2::parse_zonos2_quality_buckets(hp, "none");
        const auto bare = zonos2::build_zonos2_prompt(hp, hello, none);
        expect(bare.n_rows == 25 && text_at(bare, 0) == 2, "quality_buckets=none drops the quality row");

        expect(zonos2::parse_zonos2_quality_buckets(hp, "trailing_silence_s:3, lufs:5") ==
                   std::vector<int>({5, -1, -1, -1, -1, 3}),
               "quality bucket parse");
        zonos2::Zonos2PromptOptions two;
        two.quality_buckets = zonos2::parse_zonos2_quality_buckets(hp, "lufs:11,estimated_bandlimit_hz:7");
        const auto q = zonos2::build_zonos2_prompt(hp, hello, two);
        expect(text_at(q, 0) == 456 + 11 && text_at(q, 1) == 456 + 36 + 7, "quality rows in feature order");
        expect_rejected([&] { (void) zonos2::parse_zonos2_quality_buckets(hp, "bogus:1"); },
                        "zonos2 quality_buckets: unknown feature 'bogus'", "unknown feature");
        expect_rejected([&] { (void) zonos2::parse_zonos2_quality_buckets(hp, "leading_silence_s:8"); },
                        "zonos2 quality_buckets: leading_silence_s bucket must be 0..7", "bucket range");
        expect_rejected([&] { (void) zonos2::parse_zonos2_quality_buckets(hp, "lufs"); },
                        "zonos2 quality_buckets: expected feature:bucket", "missing bucket");
        zonos2::Zonos2PromptOptions bad_rate;
        bad_rate.speaking_rate_bucket = 8;
        expect_rejected([&] { (void) zonos2::build_zonos2_prompt(hp, hello, bad_rate); },
                        "zonos2 speaking_rate_bucket must be 0..7", "speaking rate range");

        // Long-form continuation: conditioning rows, BOS, "context texts + chunk" joined by spaces, EOS, then the
        // context rows as the audio prefix in place of the silence rows.
        const std::vector<int32_t> first_rows = {1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 18, 19};
        const std::vector<int32_t> previous_rows = {21, 22, 23, 24, 25, 26, 27, 28, 1024};
        const std::vector<zonos2::Zonos2ContextChunk> context = {{"Hi.", first_rows.data(), 2}, {"Yo.", previous_rows.data(), 1}};
        zonos2::Zonos2PromptOptions continued;
        continued.trailing_silence = false;
        const auto cont = zonos2::build_zonos2_continuation_prompt(hp, context, hello, continued);
        const std::string joined = "Hi. Yo. Hello.";
        expect(cont.n_rows == 19 && cont.speaker_row == -1, "continuation prompt: BOS, 14 bytes, EOS, 3 audio rows");
        expect(text_at(cont, 0) == 2 && text_at(cont, 15) == 3, "a continued chunk has no trailing-silence row");
        for (int i = 0; i < 14; ++i) {
            expect(text_at(cont, 1 + i) == 192 + static_cast<unsigned char>(joined[static_cast<size_t>(i)]),
                   "context and chunk text joined by single spaces");
        }
        for (int r = 16; r < 19; ++r) expect(text_at(cont, r) == 519, "audio prefix rows carry the text pad");
        expect(audio_at(cont, 16, 0) == 1 && audio_at(cont, 17, 8) == 19 && audio_at(cont, 18, 0) == 21 &&
                   audio_at(cont, 18, 8) == 1024,
               "audio prefix is the context rows in order, unchanged");
        const auto last = zonos2::build_zonos2_continuation_prompt(hp, context, hello, {});
        expect(last.n_rows == 20 && text_at(last, 0) == 511 && text_at(last, 1) == 2,
               "the last chunk keeps the trailing-silence row");
        zonos2::Zonos2PromptOptions continued_clone = continued;
        continued_clone.speaker = true;
        continued_clone.quality_buckets = zonos2::parse_zonos2_quality_buckets(hp, "lufs:5,trailing_silence_s:3");
        const auto cont_spk = zonos2::build_zonos2_continuation_prompt(hp, context, hello, continued_clone);
        expect(cont_spk.speaker_row == 0 && cont_spk.n_rows == 23 && text_at(cont_spk, 0) == 519 &&
                   text_at(cont_spk, 1) == 517 && text_at(cont_spk, 2) == 518 && text_at(cont_spk, 3) == 456 + 5 &&
                   text_at(cont_spk, 4) == 2,
               "continuation keeps the speaker rows and the other quality rows");
        const auto open = zonos2::build_zonos2_prompt(hp, hello, continued);
        expect(open.n_rows == 25 && text_at(open, 0) == 2 && audio_at(open, 8, 0) == 568,
               "a continued first chunk drops the trailing-silence row and keeps the silence rows");

        // Long-form word packing (ZONOS2 split_text_words): greedy, single spaces, never inside a word.
        using Chunks = std::vector<std::string>;
        expect(zonos2::split_zonos2_long_form_text("  one two\tthree \n four  ", 9) == Chunks({"one two", "three", "four"}),
               "words pack up to the budget and whitespace collapses");
        expect(zonos2::split_zonos2_long_form_text("one two three", 13) == Chunks({"one two three"}),
               "text that fits is one chunk");
        expect(zonos2::split_zonos2_long_form_text("a extraordinarily b", 5) == Chunks({"a", "extraordinarily", "b"}),
               "a word longer than the budget is its own chunk");
        expect(zonos2::split_zonos2_long_form_text("\xC3\xA9t\xC3\xA9 \xC3\xA9t\xC3\xA9 \xC3\xA9t\xC3\xA9", 7) ==
                   Chunks({"\xC3\xA9t\xC3\xA9 \xC3\xA9t\xC3\xA9", "\xC3\xA9t\xC3\xA9"}),
               "the budget counts code points, not bytes");
        expect(zonos2::split_zonos2_long_form_text(" \n ", 10).empty(), "blank text has no chunks");

        if (argc > 1) {
            const std::filesystem::path oracle = argv[1];
            expect_oracle(plain, oracle / "prompt_hello.npy");
            zonos2::Zonos2PromptOptions noisy;
            noisy.speaker = true;
            expect_oracle(zonos2::build_zonos2_prompt(hp, hello, noisy), oracle / "prompt_hello_spk.npy");
            std::cout << "zonos2 prompt oracle: PASS\n";
            if (std::filesystem::exists(oracle / "longform_texts.txt")) {
                std::ifstream in(oracle / "longform_texts.txt");
                std::vector<std::string> texts;
                for (std::string line; std::getline(in, line);) texts.push_back(line);
                expect(texts.size() == 3, "longform_texts.txt holds three chunks");
                auto rows = [&](const char * name) {
                    const auto npy = zonos2::read_f32_npy(oracle / name);
                    expect(npy.shape.size() == 2 && npy.shape[1] == 9, std::string(name) + " must be [rows, 9]");
                    return std::vector<int32_t>(npy.data.begin(), npy.data.end());
                };
                const auto anchor_rows = rows("longform_anchor_rows.npy");
                const auto recent_rows = rows("longform_recent_rows.npy");
                const zonos2::Zonos2ContextChunk anchor{texts[0], anchor_rows.data(), static_cast<int>(anchor_rows.size() / 9)};
                const zonos2::Zonos2ContextChunk recent{texts[1], recent_rows.data(), static_cast<int>(recent_rows.size() / 9)};
                expect_oracle(zonos2::build_zonos2_continuation_prompt(hp, {anchor, recent}, texts[2], {}),
                              oracle / "longform_cont_anchor.npy");
                zonos2::Zonos2PromptOptions open_chunk;
                open_chunk.trailing_silence = false;
                expect_oracle(zonos2::build_zonos2_continuation_prompt(hp, {recent}, texts[2], open_chunk),
                              oracle / "longform_cont_recent.npy");
                zonos2::Zonos2PromptOptions cloned;
                cloned.speaker = true;
                cloned.speaking_rate_bucket = 3;
                expect_oracle(zonos2::build_zonos2_continuation_prompt(hp, {anchor, recent}, texts[2], cloned),
                              oracle / "longform_cont_spk.npy");
                std::cout << "zonos2 long-form prompt oracle: PASS\n";
            }
            if (std::filesystem::exists(oracle / "longform_split_input.txt")) {
                auto slurp = [](const std::filesystem::path & path) {
                    std::ifstream in(path, std::ios::binary);
                    return std::string(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
                };
                std::ifstream in(oracle / "longform_split_chunks.txt");
                std::vector<std::string> expected;
                for (std::string line; std::getline(in, line);) expected.push_back(line);
                expect(zonos2::split_zonos2_long_form_text(slurp(oracle / "longform_split_input.txt"), 150) == expected,
                       "word packing differs from ZONOS2's split_text_words");
                std::cout << "zonos2 long-form split oracle: PASS (" << expected.size() << " chunks)\n";
            }
        }
        std::cout << "zonos2 prompt tests passed\n";
        return 0;
    } catch (const std::exception & error) {
        std::cerr << "zonos2_prompt_test: " << error.what() << "\n";
        return 1;
    }
}
