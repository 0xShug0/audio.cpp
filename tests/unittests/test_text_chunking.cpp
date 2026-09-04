// Default-mode text chunking regression: CJK text (no ASCII spaces) must still
// split at the codepoint budget, with full-width punctuation as word boundaries.
#include "engine/framework/text/chunking.h"

#include <cstdlib>
#include <iostream>
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
