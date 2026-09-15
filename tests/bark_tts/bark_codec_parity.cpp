#include "engine/framework/core/backend.h"
#include "engine/framework/core/execution_context.h"
#include "engine/framework/model_spec/package.h"
#include "engine/models/bark_tts/assets.h"
#include "engine/models/bark_tts/codec.h"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <stdexcept>

int main(int argc, char ** argv) try {
    if (argc != 5) {
        std::cerr << "usage: bark_codec_parity <gguf> <model-spec> <frames> <out.f32>\n";
        return 2;
    }
    const int frames = std::stoi(argv[3]);
    engine::model_spec::ScopedSpecOverride spec{std::filesystem::path(argv[2]), std::filesystem::path(argv[1])};
    auto assets = engine::models::bark_tts::load_bark_assets(argv[1]);
    const auto found = assets->presets.find("v2/en_speaker_6");
    if (found == assets->presets.end()) throw std::runtime_error("missing parity preset");
    std::vector<std::vector<int32_t>> codes(8);
    for (int book = 0; book < 8; ++book) {
        codes[book].assign(found->second.fine[book].begin(),
                           found->second.fine[book].begin() + std::min<int>(frames, found->second.fine[book].size()));
    }
    engine::core::BackendConfig config;
    config.type = engine::core::BackendType::Cpu;
    config.threads = 8;
    engine::core::ExecutionContext execution(config);
    engine::models::bark_tts::BarkCodecDecoder decoder(assets, execution, engine::assets::TensorStorageType::Native);
    const auto waveform = decoder.decode(codes);
    std::ofstream output(argv[4], std::ios::binary);
    output.write(reinterpret_cast<const char *>(waveform.data()), static_cast<std::streamsize>(waveform.size() * sizeof(float)));
    std::cout << "samples=" << waveform.size() << "\n";
    return 0;
} catch (const std::exception & error) {
    std::cerr << error.what() << "\n";
    return 1;
}
