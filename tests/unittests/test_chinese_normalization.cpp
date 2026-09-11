#include "engine/framework/text/chinese_normalization.h"

#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

namespace {

struct Case {
    const char * input;
    const char * expected;
};

void expect_equal(const std::string & actual, const std::string & expected, const char * input) {
    if (actual != expected) {
        std::cerr << "Chinese normalization mismatch for: " << input << "\nexpected: " << expected
                  << "\nactual:   " << actual << "\n";
        std::exit(1);
    }
}

}  // namespace

int main() {
    const std::vector<Case> cases = {
        {"3.14", "三点一四"},
        {"1.618", "一点六一八"},
        {"50%", "百分之五十"},
        {"12.5%", "百分之十二点五"},
        {"2024年10月1日", "二零二四年十月一日"},
        {"3月15日", "三月十五日"},
        {"第3名", "第三名"},
        {"第100次", "第一百次"},
        {"1/2", "二分之一"},
        {"2/3", "三分之二"},
        {"零下5", "零下五"},
        {"3.2个百分点", "三点二个百分点"},
        {"公元前221年到公元前121年", "公元前两百二十一年到公元前一百二十一年"},
        // Pinyin-tone and name placeholders must survive the number normalizer
        // (official TextNormalizer uses letter-suffixed placeholders for this).
        {"我试了试wan22的还能正常跑", "我试了试wan2二的还能正常跑"},
        {"chong2", "chong2"},
        {"ju2", "JV2"},
        {"克里斯托弗·诺兰", "克里斯托弗-诺兰"},
        {"克里斯托弗·诺兰2024年", "克里斯托弗-诺兰二零二四年"},
    };

    for (const auto & item : cases) {
        expect_equal(
            engine::text::normalize_chinese_text(
                item.input,
                engine::text::ChineseTextNormalizationTarget::IndexTTS),
            item.expected,
            item.input);
    }

    std::cout << "chinese_normalization_test passed\n";
    return 0;
}
