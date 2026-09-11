#include "engine/models/kokoro_tts/g2p_multilingual.h"
#include "engine/framework/io/json.h"
#include <iostream>
#include <stdexcept>
int main(int argc, char ** argv) {
    try {
        if (argc != 3) throw std::runtime_error("usage: kokoro_g2p_probe <resources> <cases.json>");
        engine::models::kokoro_tts::MultilingualG2P g2p(argv[1]);
        auto cases = engine::io::json::parse_file(argv[2]);
        for (const auto & item : cases.as_array()) {
            try {
                std::cout << g2p.phonemize(item.find("text")->as_string(), item.find("language")->as_string()) << '\n';
            } catch (const std::exception & e) { std::cout << "ERROR: " << e.what() << '\n'; }
        }
        return 0;
    } catch (const std::exception & e) { std::cerr << e.what() << '\n'; return 1; }
}
