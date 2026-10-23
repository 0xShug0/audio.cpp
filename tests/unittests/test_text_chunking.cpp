// Default-mode text chunking regression: CJK text (no ASCII spaces) must still
// split at the codepoint budget, with full-width punctuation as word boundaries.
#include "engine/framework/text/chunking.h"

#include <cstdlib>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void require(bool ok, const std::string & what) {
    if (!ok) {
        std::cerr << "FAIL: " << what << "\n";
        std::exit(1);
    }
}

void check_chunks(const std::string & text, int64_t budget, size_t min_chunks, const std::string & label) {
    const auto chunks = engine::text::split_text_chunks(text, budget);
    if (chunks.size() < min_chunks) {
        std::cerr << "FAIL: " << label << " expected >= " << min_chunks << " chunks, got " << chunks.size()
                  << "\n";
        std::exit(1);
    }
    std::string joined;
    for (const auto & c : chunks) {
        joined += c;
    }
    require(joined == text, label + ": chunks must concatenate back to the trimmed input verbatim");
    for (const auto & c : chunks) {
        require(!c.empty(), label + ": empty chunk");
    }
}

}  // namespace

int main() {
    using engine::text::parse_text_chunk_size_override;
    require(!parse_text_chunk_size_override({}).has_value(), "omitted budget stays unset");
    for (const std::string key : {"text_chunk_size", "chunk_size"}) {
        require(!parse_text_chunk_size_override({{key, ""}}).has_value(), key + ": empty budget stays unset");
        for (const int64_t value : {int64_t{1}, int64_t{12}, int64_t{300}, std::numeric_limits<int64_t>::max()}) {
            require(parse_text_chunk_size_override({{key, std::to_string(value)}}) == value,
                    key + ": valid integer preserved");
        }
        for (const std::string input : {"12.5", "12junk", "12 ", "0", "-1", "junk", "9223372036854775808"}) {
            bool rejected = false;
            try {
                (void) parse_text_chunk_size_override({{key, input}});
            } catch (const std::exception & error) {
                rejected = true;
                if (input == "12.5" || input == "12junk" || input == "12 ") {
                    require(error.what() == key + " must be an integer", key + ": useful integer error");
                }
            }
            require(rejected, key + ": reject invalid budget " + input);
        }
    }

    // Regression: a 236-codepoint CJK paragraph with the 200-codepoint default
    // budget used to come back as a single chunk, because split_word_ranges only
    // cut at ASCII spaces and the whole paragraph parsed as one word. Must now
    // split into at least two sentence-aligned chunks.
    const std::string cjk_long =
        "大家好，我是零一B语音合成模型，现在进行流式输出测试。这段文字比较长，目的是看模型能否把整段话分成多个音频块，边合成边推送。"
        "今天的天气很好，适合外出散步，湖边的柳树已经抽出了新芽，水面倒映着蓝天白云。如果流式工作正常，客户端应该能很快收到第一个音频块。"
        "第二段继续测试分块是否稳定，这里再补充一些内容，让总字数超过上限，这样模型必须把文字切开分多次合成。流式的意义在于长文本不必等全部算完。"
        "最后再来一句收尾的话，确认整个流程完整结束，谢谢大家。";
    check_chunks(cjk_long, 200, 2, "CJK long text over budget splits");

    // Under budget: still one chunk, verbatim.
    check_chunks("短文本，不需要分块。", 200, 1, "short CJK text stays whole");

    // Clause punctuation is a valid rollback boundary when the budget cuts a
    // run of clauses: budget 12 lands inside the second clause run, so the
    // first chunk must end at the 、 (clause) boundary, not mid-run.
    {
        const std::string text = "一二三四五六七八九十，一二三四五六七八九十。";
        const auto chunks = engine::text::split_text_chunks(text, 12);
        require(chunks.size() == 2, "clause rollback produces two chunks");
        require(chunks[0] == "一二三四五六七八九十，", "first chunk ends at clause punctuation");
        require(chunks[1] == "一二三四五六七八九十。", "second chunk keeps the sentence");
    }

    // Latin text is unaffected: ASCII punctuation stays attached to its word
    // and boundaries still fall on ASCII-space words / sentence breaks.
    {
        const std::string text = "Hello world. How are you today?";
        const auto chunks = engine::text::split_text_chunks(text, 20);
        require(chunks.size() == 2, "latin splits at the sentence break");
        require(chunks[0] == "Hello world.", "first latin chunk ends at the period");
        require(chunks[1] == "How are you today?", "second latin chunk verbatim");
    }

    // A CJK run without punctuation has no word boundary. It is cut into equal
    // pieces that fit the budget instead of coming back as one over-budget
    // chunk; a Latin word longer than the budget still stays whole.
    {
        const std::string run = "一二三四五六七八九十一二三四五六七八九十一二三四五";
        const auto chunks = engine::text::split_text_chunks(run, 10);
        require(chunks.size() == 3, "unpunctuated CJK run splits into three pieces");
        require(chunks[0] == "一二三四五六七八", "first piece");
        require(chunks[1] == "九十一二三四五六", "second piece");
        require(chunks[2] == "七八九十一二三四五", "third piece");
        check_chunks(run, 10, 3, "unpunctuated CJK run");
        const auto latin = engine::text::split_text_chunks("supercalifragilistic is long", 10);
        require(latin.size() == 2 && latin[0] == "supercalifragilistic" && latin[1] == "is long",
                "latin word over budget stays whole");

        // Cuts fall only between two CJK letters: a tag or a Latin word inside
        // the run stays in one piece.
        const std::string tag = "<|emotion:contemplation|>";
        const std::string tagged = "一二三四五六七八九十" + tag + "一二三四五六七八九十";
        const auto tag_chunks = engine::text::split_text_chunks(tagged, 12);
        check_chunks(tagged, 12, 2, "CJK run with a tag");
        size_t tag_count = 0;
        for (const auto & c : tag_chunks) {
            tag_count += c.find(tag) != std::string::npos;
            require(c.find(tag) != std::string::npos || c.find('|') == std::string::npos, "tag is not cut");
        }
        require(tag_count == 1, "tag stays in exactly one chunk");
        const std::string mixed = "一二三四五六七八九十GPT一二三四五六七八九十";
        const auto mixed_chunks = engine::text::split_text_chunks(mixed, 8);
        check_chunks(mixed, 8, 2, "CJK run with a Latin word");
        size_t gpt = 0;
        for (const auto & c : mixed_chunks) {
            gpt += c.find("GPT") != std::string::npos;
        }
        require(gpt == 1, "Latin word inside a CJK run stays whole");
    }

    // Tag-aware mode: a closing quote or bracket after a sentence stop belongs to
    // that sentence. Units used to end at the stop, so the closer opened the next
    // chunk (`."` split between `.` and `"`) and dialogue attribution shifted.
    // Each budget puts the chunk boundary right after the closer.
    const auto tag_aware = [](const std::string & text, int64_t budget) {
        return engine::text::split_text_chunks(text, budget, engine::text::TextChunkMode::TagAware);
    };
    const auto expect = [&](const std::string & text, int64_t budget, const std::vector<std::string> & want) {
        const auto got = tag_aware(text, budget);
        std::string shown;
        for (const auto & c : got) shown += "[" + c + "]";
        require(got == want, "tag_aware " + text + " -> " + shown);
    };
    expect("He said \"stop.\" Then left.", 16, {"He said \"stop.\"", "Then left."});
    expect("He said 'stop.' Then left.", 16, {"He said 'stop.'", "Then left."});
    expect("He said \u201cstop.\u201d Then left.", 16, {"He said \u201cstop.\u201d", "Then left."});
    expect("He said \u2018stop.\u2019 Then left.", 16, {"He said \u2018stop.\u2019", "Then left."});
    expect("Il dit \u00abstop!\u00bb Puis parti.", 14, {"Il dit \u00abstop!\u00bb", "Puis parti."});
    expect("See (this.) Next one.", 11, {"See (this.)", "Next one."});
    expect("Ask \"why?\") Then go.", 11, {"Ask \"why?\")", "Then go."});
    expect("\u300c\u4e00\u4e8c\u4e09\u3002\u300d\u56db\u4e94\u3002", 6,
           {"\u300c\u4e00\u4e8c\u4e09\u3002\u300d", "\u56db\u4e94\u3002"});
    expect("\u300e\u4e00\u4e8c\u4e09\u3002\u300f\u56db\u4e94\u3002", 6,
           {"\u300e\u4e00\u4e8c\u4e09\u3002\u300f", "\u56db\u4e94\u3002"});
    expect("\uff08\u4e00\u4e8c\u4e09\u3002\uff09\u56db\u4e94\u3002", 6,
           {"\uff08\u4e00\u4e8c\u4e09\u3002\uff09", "\u56db\u4e94\u3002"});
    // `"`, `'` and `\u2019` that open the next word stay with it.
    expect("Wait.'Tis true.", 6, {"Wait.", "'Tis", "true."});
    expect("Wait.\u2019Tis true.", 6, {"Wait.", "\u2019Tis", "true."});
    expect("He left.\"Hi,\" she said.", 15, {"He left.", "\"Hi,\" she said."});
    expect("He left.\"'Tis.\"", 9, {"He left.", "\"'Tis.\""});
    expect("He left.\"\u2019Tis.\"", 9, {"He left.", "\"\u2019Tis.\""});
    // A long run of quotes after a stop is scanned once.
    {
        const auto chunks = tag_aware("Stop." + std::string(100000, '"') + " Next.", 200);
        require(!chunks.empty() && chunks.back() == "Next.", "tag_aware long quote run");
    }
    // Long dialogue: no chunk opens with a stray closing quote.
    {
        const std::string text =
            "\"You're late,\" Marcus said. \"I know. I'm sorry.\" She sat down. "
            "\"The bridge was closed!\" (It always was.) He shrugged.";
        for (const auto & c : tag_aware(text, 30)) {
            require(c.rfind("\" ", 0) != 0 && c.rfind(") ", 0) != 0 && c != "\"" && c != ")",
                    "tag_aware dialogue chunk opens with a closer: " + c);
        }
    }

    // A single oversized word without any boundary still passes through whole
    // (pre-existing behavior: no hard character-level cut).
    {
        const std::string word(300, 'a');
        const auto chunks = engine::text::split_text_chunks(word, 200);
        require(chunks.size() == 1, "unbreakable oversized word stays one chunk");
        require(chunks[0] == word, "oversized word verbatim");
    }

    std::cout << "text_chunking_test: all passed\n";
    return 0;
}
