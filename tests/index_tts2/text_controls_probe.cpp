#include "engine/models/index_tts2/assets.h"
#include "engine/models/index_tts2/tokenizer_text.h"
#include "../unittests/test_assert.h"

#include <iostream>
#include <stdexcept>

int main(int argc, char ** argv) {
    if (argc != 2) {
        std::cerr << "usage: index_tts2_text_controls_probe /path/to/local/model\n";
        return 2;
    }
    try {
        auto assets = engine::models::index_tts2::load_index_tts2_assets(argv[1]);
        engine::models::index_tts2::IndexTTS2TextTokenizer tokenizer(assets);
        const std::string text = "There are 12 apples and 24 oranges. We can share them today. Everyone gets 3 pieces.";
        const auto defaults = tokenizer.encode_for_inference(text, 120, "en");
        const auto normalized = tokenizer.encode_for_inference(text, 120, "en", true);
        engine::test::require(defaults.segment_token_ids == normalized.segment_token_ids, "explicit default must preserve tokenization");
        const auto short_segments = tokenizer.encode_for_inference(text, 12, "en");
        engine::test::require(short_segments.segment_token_ids.size() > defaults.segment_token_ids.size(), "smaller text budget must split the actual tokenizer output");
        if (tokenizer.variant() == engine::models::index_tts2::IndexTTS2Variant::kV2_5) {
            const auto raw = tokenizer.encode_for_inference(text, 120, "en", false);
            engine::test::require(raw.normalized_text.find("12") != std::string::npos, "normalization bypass must keep digits");
            engine::test::require(raw.normalized_text != normalized.normalized_text, "normalization switch must change processed text");
            std::cout << "normalized=" << normalized.normalized_text << "\nraw=" << raw.normalized_text << '\n';
        } else {
            bool rejected = false;
            try { (void)tokenizer.encode_for_inference(text, 120, "en", false); }
            catch (const std::exception &) { rejected = true; }
            engine::test::require(rejected, "2.0 must reject unsupported normalization bypass");
        }
        std::cout << "default_segments=" << defaults.segment_token_ids.size()
                  << " short_segments=" << short_segments.segment_token_ids.size() << '\n';
        return 0;
    } catch (const std::exception & error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
