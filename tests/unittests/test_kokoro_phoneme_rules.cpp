#include "engine/models/kokoro_tts/frontend.h"
#include "test_assert.h"

#include <iostream>

using namespace engine::models::kokoro_tts;
using engine::test::require;

int main() {
    try {
        KokoroAssets assets;
        KokoroVoicePack pack;
        pack.id = "af_heart";
        pack.rows = 1;
        pack.cols = 256;
        pack.values.resize(256);
        assets.voices.emplace(pack.id, std::move(pack));
        assets.vocab = {{"I", 1}};
        const engine::runtime::Transcript text{"I foo.", "en-us"};
        const auto plain = resolve_kokoro_frontend_session_state(text, std::nullopt, assets);
        require(plain.phoneme_rules.empty(), "default must not install pronunciation rules");
        const auto empty = resolve_kokoro_frontend_session_state(
            text, std::nullopt, assets, {{"phoneme_rules", "[]"}});
        require(empty.phoneme_rules.empty(), "explicit empty rules must remain inactive");
        const auto configured = resolve_kokoro_frontend_session_state(text, std::nullopt, assets,
            {{"phoneme_rules", R"([{"pattern":"I","replacement":"ˌI"}])"}});
        require(configured.phoneme_rules.size() == 1, "requested rule must be compiled");
        const auto supplied = build_kokoro_synthesis_input(text, configured, assets, "I");
        require(supplied.phonemes == "I", "supplied phonemes must bypass rewrite rules");

        for (const auto * invalid : {"{", "{}", "[1]", "[{}]",
                 R"([{"pattern":"I"}])", R"([{"pattern":1,"replacement":"x"}])",
                 R"([{"pattern":"[","replacement":"x"}])"}) {
            bool rejected = false;
            try {
                (void)resolve_kokoro_frontend_session_state(
                    text, std::nullopt, assets, {{"phoneme_rules", invalid}});
            } catch (const std::exception & error) {
                rejected = std::string(error.what()).find("Invalid Kokoro phoneme_rules:") == 0;
            }
            require(rejected, std::string("invalid rule accepted: ") + invalid);
        }
        std::cout << "PASS Kokoro phoneme rule defaults, parsing and supplied-phoneme bypass\n";
        return 0;
    } catch (const std::exception & error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
