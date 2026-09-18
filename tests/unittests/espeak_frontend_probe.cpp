// Optional real-library parity probe. No neural-model weights required.
// Usage: probe <library> <espeak-ng-data> <sanotts|inflect|piper|concurrent>
#include "engine/community_models/sanotts/frontend.h"
#include "engine/community_models/inflect_v2/frontend.h"
#include <future>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include <unordered_map>

static const std::vector<std::string> prompts = {
    "Hello, this is a native speech test.",
    "Good morning! How are you today?",
    "The price is $12.50, and the date is September 10, 2026.",
    "A short sentence; another clause: and a question?",
    "We should preserve pronunciation, punctuation, and token IDs."
};
template <class Frontend> void print(const Frontend & frontend) {
    for (const auto & text : prompts) {
        for (auto id : frontend.encode(text).token_ids) std::cout << id << ',';
        std::cout << '\n';
    }
}
int main(int argc, char ** argv) try {
    if (argc != 4) throw std::runtime_error("expected library, data directory and mode");
    using engine::models::sanotts::SanoTtsFrontend;
    using engine::models::inflect_v2::InflectV2Frontend;
    const std::string mode = argv[3];
    // '-' selects the compiled-in engine / executable-local data in static builds.
    if (std::string(argv[1]) == "-") argv[1][0] = '\0';
    if (std::string(argv[2]) == "-") argv[2][0] = '\0';
    if (mode == "sanotts") { SanoTtsFrontend f(argv[1], argv[2], 1000); print(f); }
    else if (mode == "inflect") { InflectV2Frontend f(argv[1], argv[2]); print(f); }
    else if (mode == "piper") {
        // Synthetic exhaustive IPA-range map tests the frontend independently
        // of model weights and per-voice token inventories.
        std::unordered_map<std::string, int32_t> ids;
        for (int cp = 0; cp < 1024; ++cp) {
            std::string utf8;
            if (cp < 128) utf8 += static_cast<char>(cp);
            else {
                utf8 += static_cast<char>(0xc0 | (cp >> 6));
                utf8 += static_cast<char>(0x80 | (cp & 63));
            }
            ids[utf8] = cp;
        }
        const std::vector<std::pair<std::string, std::string>> cases = {
            {"en-us", "Hello, how are you?"}, {"es", "Hola, buenos días."},
            {"fr", "Bonjour, comment allez-vous?"}, {"de", "Guten Morgen, wie geht es Ihnen?"},
            {"it", "Buongiorno, come stai?"}, {"pt-br", "Bom dia, como vai?"},
            {"pl", "Dzień dobry, jak się masz?"}, {"ru", "Доброе утро, как дела?"},
            {"hi", "नमस्ते आप कैसे हैं?"}, {"vi", "Xin chào, bạn khỏe không?"},
            {"id", "Selamat pagi, apa kabar?"}
        };
        for (const auto & item : cases) {
            engine::models::sanotts::SanoTtsPiperFrontend f(argv[1], argv[2], item.first, ids, 1000);
            std::cout << item.first << ':';
            for (auto id : f.encode(item.second).token_ids) std::cout << id << ',';
            std::cout << '\n';
        }
    }
    else if (mode == "concurrent") {
        SanoTtsFrontend sano(argv[1], argv[2], 1000);
        InflectV2Frontend inflect(argv[1], argv[2]);
        const auto run = [](const auto & frontend) {
            std::vector<std::vector<int32_t>> expected;
            for (const auto & text : prompts) expected.push_back(frontend.encode(text).token_ids);
            for (int repeat = 0; repeat < 20; ++repeat)
                for (size_t i = 0; i < prompts.size(); ++i)
                    if (frontend.encode(prompts[i]).token_ids != expected[i])
                        throw std::runtime_error("cross-model token mismatch");
        };
        auto a = std::async(std::launch::async, [&] { run(sano); });
        auto b = std::async(std::launch::async, [&] { run(inflect); });
        a.get(); b.get();
        std::cout << "200 cross-model real-library requests passed\n";
    } else throw std::runtime_error("unknown probe mode");
    return 0;
} catch (const std::exception & e) { std::cerr << e.what() << '\n'; return 1; }
