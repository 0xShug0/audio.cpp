#include "engine/models/crisperwhisper/model.h"

#include "test_assert.h"

#include <vector>

using engine::models::crisperwhisper::CrisperWhisperWord;
using engine::models::crisperwhisper::align_transcript;
using engine::test::require_eq;

int main() {
    const std::vector<CrisperWhisperWord> hypothesis = {
        {"первое", 0.1, 0.4},
        {"слово", 0.5, 0.8},
    };

    const auto aligned = align_transcript("ПЕРВОЕ вставленное СЛОВО", hypothesis, 1.0);
    require_eq(aligned.size(), 3U, "word count");
    require_eq(aligned[0].start, 0.1, "first start");
    require_eq(aligned[0].end, 0.4, "first end");
    require_eq(aligned[1].start, 0.4, "insertion start");
    require_eq(aligned[1].end, 0.5, "insertion end");
    require_eq(aligned[2].start, 0.5, "last start");
    require_eq(aligned[2].end, 0.8, "last end");
    return 0;
}
