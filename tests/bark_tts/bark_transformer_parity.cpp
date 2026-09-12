#include "engine/framework/core/backend.h"
#include "engine/framework/core/execution_context.h"
#include "engine/framework/model_spec/package.h"
#include "engine/models/bark_tts/assets.h"
#include "engine/models/bark_tts/tokenizer.h"
#include "engine/models/bark_tts/transformer.h"

#include <fstream>
#include <iostream>
#include <stdexcept>

int main(int argc, char ** argv) try {
    if (argc != 4) {
        std::cerr << "usage: bark_transformer_parity <gguf> <model-spec> <out.f32>\n";
        return 2;
    }
    engine::model_spec::ScopedSpecOverride spec{std::filesystem::path(argv[2]), std::filesystem::path(argv[1])};
    auto assets = engine::models::bark_tts::load_bark_assets(argv[1]);
    const auto & preset = assets->presets.at("v2/en_speaker_6");
    engine::models::bark_tts::BarkTokenizer tokenizer(assets->resources.read_text("tokenizer_json"));
    auto text = tokenizer.encode("Hello, Bark!");
    for (auto & id : text) id += 10048;
    text.resize(256, 129595);
    std::vector<int32_t> history(preset.semantic.end() - 256, preset.semantic.end());
    text.push_back(129599);
    history.push_back(10000);
    engine::core::BackendConfig config;
    config.type = engine::core::BackendType::Cpu;
    config.threads = 8;
    engine::core::ExecutionContext execution(config);
    engine::models::bark_tts::BarkTransformer transformer(assets, execution, "semantic", assets->config.semantic,
        true, engine::assets::TensorStorageType::Native);
    const auto logits = transformer.causal_logits({text, history});
    std::ofstream output(argv[3], std::ios::binary);
    output.write(reinterpret_cast<const char *>(logits.data()), static_cast<std::streamsize>(logits.size() * sizeof(float)));
    output.close();
    std::vector<std::vector<int32_t>> fine_codes(8);
    for (int book = 0; book < 8; ++book)
        fine_codes[book].assign(preset.fine[book].begin(), preset.fine[book].begin() + 64);
    engine::models::bark_tts::BarkTransformer fine(assets, execution, "fine_acoustics", assets->config.fine,
        false, engine::assets::TensorStorageType::Native);
    const auto fine_logits = fine.fine_logits(fine_codes, 2);
    std::ofstream fine_output(std::string(argv[3]) + ".fine", std::ios::binary);
    fine_output.write(reinterpret_cast<const char *>(fine_logits.data()),
                      static_cast<std::streamsize>(fine_logits.size() * sizeof(float)));
    std::cout << "semantic_logits=" << logits.size() << " fine_logits=" << fine_logits.size() << "\n";
    return 0;
} catch (const std::exception & error) {
    std::cerr << error.what() << "\n";
    return 1;
}
