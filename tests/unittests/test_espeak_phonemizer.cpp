#include "engine/framework/audio/espeak_phonemizer.h"
#include "test_assert.h"
#include <future>
#include <iostream>

int main(int argc, char ** argv) try {
    using engine::audio::EspeakPhonemizer;
    using engine::test::require;
    require(argc == 3, "expected complete and incomplete test libraries");
    const auto fails = [](auto action) {
        try { action(); } catch (const std::exception &) { return true; }
        return false;
    };
    require(fails([] { EspeakPhonemizer p("/nonexistent/espeak-library", {}, {"en-us"}); }), "missing library");
    require(fails([&] { EspeakPhonemizer p(argv[1], "/nonexistent/espeak-ng-data", {"en-us"}); }), "missing data");
    require(fails([&] { EspeakPhonemizer p(argv[2], {}, {"en-us"}); }), "missing symbol");
    require(fails([&] { EspeakPhonemizer p(argv[1], {}, {}); }), "empty candidates");
    require(fails([&] { EspeakPhonemizer p(argv[1], {}, {""}); }), "empty voice");
    EspeakPhonemizer english(argv[1], {}, {"missing", "en-us"});
    EspeakPhonemizer french(argv[1], {}, {"fr"});
    require(english.phonemize("", 2).empty(), "empty text");
    require(english.phonemize("hello|world", 7, "/") == "en-us:7:hello/en-us:7:world", "clauses and mode");
    require(fails([&] { EspeakPhonemizer p(argv[1], {}, {"missing"}); }), "missing voice");
    require(fails([&] { english.phonemize("stuck", 2); }), "cursor guard");
    require(fails([&] { EspeakPhonemizer p(argv[2], {}, {"en-us"}); }), "failed library switch");
    require(english.phonemize("hello", 2) == "en-us:2:hello", "recovery after failed switch");
    {
        EspeakPhonemizer temporary(argv[1], {}, {"de"});
        require(temporary.phonemize("hello", 2) == "de:2:hello", "temporary client");
    }
    auto run = [](const EspeakPhonemizer & p, const std::string & expected) {
        for (int i = 0; i < 500; ++i) require(p.phonemize("hello", 2) == expected, "voice isolation");
    };
    auto first = std::async(std::launch::async, [&] { run(english, "en-us:2:hello"); });
    auto second = std::async(std::launch::async, [&] { run(french, "fr:2:hello"); });
    first.get(); second.get();
    std::cout << "Shared eSpeak tests passed (1000 concurrent calls)\n";
    return 0;
} catch (const std::exception & e) {
    std::cerr << e.what() << '\n';
    return 1;
}
