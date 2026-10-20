#include "engine/models/kokoro_tts/frontend.h"
#include "test_assert.h"

#include <iostream>

using namespace engine::models::kokoro_tts;
using engine::test::require;

int main() {
    try {
        KokoroAssets assets;
        assets.vocab = {{"a", 1}, {"b", 2}};
        for (const auto & id : {"af_heart", "af_bella", "jf_alpha"}) {
            KokoroVoicePack pack;
            pack.id = id;
            pack.rows = 3;
            pack.cols = 256;
            const float offset = pack.id == "af_heart" ? 0.0f : pack.id == "af_bella" ? 2.0f : -1.0f;
            for (int row = 0; row < 3; ++row) {
                for (int col = 0; col < 256; ++col) pack.values.push_back(offset + row + col / 1024.0f);
            }
            assets.voices.emplace(pack.id, std::move(pack));
        }
        engine::runtime::Transcript text;
        text.text = "Test.";
        auto state_for = [&](const std::string & expression) {
            engine::runtime::VoiceCondition voice;
            voice.speaker.emplace();
            voice.speaker->cached_voice_id = expression;
            return resolve_kokoro_frontend_session_state(text, voice, assets);
        };
        auto synthesize = [&](const std::string & expression, std::string_view phonemes = "ab") {
            return build_kokoro_synthesis_input(text, state_for(expression), assets, phonemes);
        };
        const auto single = synthesize("af_heart");
        require(single.style == synthesize("af_heart(8)").style, "single weighted voice must be unchanged");
        const auto default_state = resolve_kokoro_frontend_session_state(text, std::nullopt, assets);
        require(build_kokoro_synthesis_input(text, default_state, assets, "ab").style == single.style,
                "default voice must be unchanged");
        require(synthesize("af_heart(1)+af_heart(2)").style == single.style, "duplicate voices preserve style");

        for (const auto & expression : {"af_heart(1)+jf_alpha(0.5)", "af_heart(2)+jf_alpha(1)",
                                       " af_heart ( 1e+2 ) + jf_alpha(50) "}) {
            const auto mixed = synthesize(expression);
            require(mixed.language_code == "a", "first voice selects English for a cross-language blend");
            require(mixed.input_ids == single.input_ids, "blending must not change text tokens");
            for (size_t i = 0; i < 256; ++i) {
                const auto a = assets.voices.at("af_heart").values[256 + i];
                const auto b = assets.voices.at("jf_alpha").values[256 + i];
                engine::test::require_close(mixed.style[i], static_cast<float>((a + 0.5 * b) / 1.5),
                                            1e-7f, "weighted phoneme-count row");
            }
        }
        const auto equal = synthesize("af_heart+af_bella");
        for (size_t i = 0; i < 256; ++i) {
            engine::test::require_close(equal.style[i], single.style[i] + 1.0f, 0.0f, "equal weights");
        }
        require(synthesize("af_heart(0)+jf_alpha(1)").style == synthesize("jf_alpha").style,
                "zero weight must not contribute");
        require(state_for("jf_alpha+af_heart").language_code == "j", "first Japanese voice selects Japanese");
        require(synthesize("af_heart(1)+jf_alpha(0.5)", "a").style !=
                    synthesize("af_heart(1)+jf_alpha(0.5)", "ab").style,
                "each phoneme count must use its own style row");

        for (const auto & expression : {"", "missing", "af_heart+missing", "af_heart+", "+af_heart",
                                       "af_heart++jf_alpha", "af_heart jf_alpha", "af_heart()",
                                       "af_heart(1", "af_heart(1))", "af_heart(1)(2)",
                                       "af_heart(-1)+jf_alpha", "af_heart(nan)", "af_heart(inf)",
                                       "af_heart(1junk)", "af_heart(0)", "af_heart(0)+jf_alpha(0)",
                                       "af_heart(1e309)", "af_heart(1e308)+jf_alpha(1e308)"}) {
            bool rejected = false;
            try { (void)state_for(expression); } catch (const std::exception &) { rejected = true; }
            require(rejected, std::string("invalid expression accepted: ") + expression);
        }
        text.language = "ja";
        bool mismatch = false;
        try { (void)state_for("af_heart+jf_alpha"); } catch (const std::exception &) { mismatch = true; }
        require(mismatch, "explicit language must match first voice");
        require(state_for("jf_alpha+af_heart").language_code == "j", "matching explicit language is accepted");
        text.language.clear();
        require(synthesize("af_heart").style == single.style, "blending must not mutate source voices");
        assets.voices.at("jf_alpha").rows = 1;
        bool short_pack = false;
        try { (void)synthesize("af_heart+jf_alpha"); } catch (const std::exception &) { short_pack = true; }
        require(short_pack, "every contributing voice must contain the selected style row");
        std::cout << "kokoro_voice_blending_test: passed\n";
        return 0;
    } catch (const std::exception & error) {
        std::cerr << "kokoro_voice_blending_test: " << error.what() << '\n';
        return 1;
    }
}
