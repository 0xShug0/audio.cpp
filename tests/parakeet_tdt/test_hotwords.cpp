#include "engine/community_models/parakeet_tdt/hotwords.h"
#include <cmath>
#include <iostream>
#include <stdexcept>

using namespace engine::community_models::parakeet_tdt;
void check(bool value, const char * message) { if (!value) throw std::runtime_error(message); }
int main() {
    try {
        const std::vector<std::string> vocab = {"\xE2\x96\x81" "Ada", "\xE2\x96\x81" "Qu", "illon", "x", "<blank>"};
        auto words = parse_phonon_hotwords("Ada, Quillon, ada");
        check(words.size() == 2, "deduplication");
        PhononHotwordAutomaton a(words, vocab, 4, 2);
        check(a.bonus(0, 0) == 2 && a.bonus(0, 1) == 2, "initial pieces");
        check(a.bonus(0, 2) == 0 && a.bonus(0, 4) == 0, "suffix and blank not boosted");
        auto state = a.step(0, 1);
        check(a.bonus(state, 2) == 2, "continuation");
        check(a.step(state, 4) == state, "blank preserves state");
        check(a.step(state, 3) == 0, "nonmatching token resets state");
        check(a.select({0, 1, 0, 2.5f, 1}, 0) == 1, "bias changes token selection");
        check(a.select({0, 1, 0, 2.5f, 4}, 0) == 4, "blank can still win");
        PhononHotwordAutomaton disabled(words, vocab, 4, 0);
        check(disabled.empty(), "zero strength disables policy");
        check(parse_phonon_hotwords("[\"Ada Lovelace\",{\"word\":\"Quillon\",\"spoken\":[\"kwil on\"]}]")[0].word == "Ada Lovelace", "JSON phrases");
        check(parse_phonon_hotwords("Ada Lovelace\nQuillon")[0].word == "Ada Lovelace", "newline phrases");
        bool rejected = false;
        try { PhononHotwordAutomaton bad(words, vocab, 4, NAN); } catch (const std::exception &) { rejected = true; }
        check(rejected, "nonfinite strength rejected");
        std::cout << "PASS: hotword continuation, blank, selection, parsing and disabled policy\n";
    } catch (const std::exception & e) { std::cerr << e.what() << '\n'; return 1; }
}
