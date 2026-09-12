#include "engine/framework/core/backend.h"
#include "engine/framework/core/execution_context.h"
#include "engine/framework/model_spec/package.h"
#include "engine/models/bark_tts/assets.h"
#include "engine/models/bark_tts/tokenizer.h"
#include "engine/models/bark_tts/transformer.h"

#include <fstream>
#include <iostream>
#include <stdexcept>
#include <cmath>

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
    history.push_back(-1);
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
    text.push_back(147);
    history.push_back(-1);
    const auto logits2 = transformer.causal_logits({text, history});
    std::ofstream output2(std::string(argv[3]) + ".semantic2", std::ios::binary);
    output2.write(reinterpret_cast<const char *>(logits2.data()),
        static_cast<std::streamsize>(logits2.size() * sizeof(float)));
    constexpr double ratio = 75.0 / 49.9 * 2.0;
    std::vector<int32_t> semantic_seed{147, 302, 2089, 330, 602, 206, 218, 2009};
    std::vector<int32_t> sem_history = preset.semantic;
    std::vector<int32_t> coarse_history;
    for (size_t frame = 0; frame < preset.coarse[0].size(); ++frame)
        for (int book = 0; book < 2; ++book)
            coarse_history.push_back(preset.coarse[book][frame] + book * 1024 + 10000);
    const int max_sem_history = static_cast<int>(std::floor(630.0 / ratio));
    const int sem_count = std::min<int>({max_sem_history, static_cast<int>(sem_history.size() / 2 * 2),
        static_cast<int>(std::floor(coarse_history.size() / ratio))});
    const int coarse_count = static_cast<int>(std::round(sem_count * ratio));
    sem_history.erase(sem_history.begin(), sem_history.end() - sem_count);
    coarse_history.erase(coarse_history.begin(), coarse_history.end() - coarse_count);
    coarse_history.resize(coarse_history.size() - 2);
    sem_history.insert(sem_history.end(), semantic_seed.begin(), semantic_seed.end());
    if (sem_history.size() > 256) sem_history.resize(256);
    sem_history.resize(256, 12048);
    sem_history.push_back(12050);
    const size_t take = std::min<size_t>(630, coarse_history.size());
    sem_history.insert(sem_history.end(), coarse_history.end() - static_cast<std::ptrdiff_t>(take), coarse_history.end());
    engine::models::bark_tts::BarkTransformer coarse(assets, execution, "coarse_acoustics", assets->config.coarse,
        true, engine::assets::TensorStorageType::Native);
    const auto coarse_logits = coarse.causal_logits({sem_history});
    std::ofstream coarse_output(std::string(argv[3]) + ".coarse", std::ios::binary);
    coarse_output.write(reinterpret_cast<const char *>(coarse_logits.data()),
        static_cast<std::streamsize>(coarse_logits.size() * sizeof(float)));
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
