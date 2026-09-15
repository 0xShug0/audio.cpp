#include "engine/models/bark_tts/generator.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>

namespace engine::models::bark_tts {
namespace {

int32_t sample(std::vector<float> logits, int begin, int end, int top_k, float temperature, std::mt19937_64 & rng) {
    if (begin < 0 || end > static_cast<int>(logits.size()) || begin >= end) throw std::runtime_error("invalid Bark sampling range");
    std::vector<int32_t> indices(static_cast<size_t>(end - begin));
    std::iota(indices.begin(), indices.end(), begin);
    const int keep = std::min<int>(top_k, static_cast<int>(indices.size()));
    std::partial_sort(indices.begin(), indices.begin() + keep, indices.end(), [&](int32_t a, int32_t b) { return logits[a] > logits[b]; });
    indices.resize(static_cast<size_t>(keep));
    const float maximum = logits[indices.front()] / temperature;
    std::vector<double> weights;
    weights.reserve(indices.size());
    for (int32_t id : indices) weights.push_back(std::exp(static_cast<double>(logits[id] / temperature - maximum)));
    return indices[std::discrete_distribution<size_t>(weights.begin(), weights.end())(rng)];
}

std::vector<int32_t> semantic_tokens(BarkTransformer & model, const BarkTokenizer & tokenizer,
                                     const BarkSpeakerPreset & preset, const std::string & text,
                                     const BarkGenerationOptions & options, std::mt19937_64 & rng) {
    auto text_ids = tokenizer.encode(text);
    if (text_ids.size() > 256) text_ids.resize(256);
    for (auto & id : text_ids) id += 10048;
    text_ids.resize(256, 129595);
    std::vector<int32_t> history;
    const size_t history_start = preset.semantic.size() > 256 ? preset.semantic.size() - 256 : 0;
    history.assign(preset.semantic.begin() + static_cast<std::ptrdiff_t>(history_start), preset.semantic.end());
    history.resize(256, 10000);
    // Bark's merge_context path sums the first 256 text embeddings with the
    // 256 semantic-history embeddings, then appends SEMANTIC_INFER without a
    // history embedding at that position.  Two channels reproduce that
    // merged 257-position sequence; negative ids are masked by the runtime.
    text_ids.push_back(129599);
    history.push_back(-1);
    std::vector<int32_t> generated;
    for (int64_t step = 0; step < options.max_tokens && text_ids.size() < 1024; ++step) {
        auto logits = model.causal_logits({text_ids, history});
        const int32_t id = sample(std::move(logits), 0, 10001, options.top_k, options.temperature, rng);
        if (id == 10000) break;
        generated.push_back(id);
        text_ids.push_back(id);
        history.push_back(-1);
    }
    if (generated.empty()) throw std::runtime_error("Bark semantic model generated no speech tokens");
    return generated;
}

std::vector<int32_t> coarse_tokens(BarkTransformer & model, const BarkSpeakerPreset & preset,
                                   const std::vector<int32_t> & semantic, const BarkGenerationOptions & options,
                                   std::mt19937_64 & rng) {
    constexpr double ratio = 75.0 / 49.9 * 2.0;
    std::vector<int32_t> sem_history = preset.semantic;
    std::vector<int32_t> coarse_history;
    if (preset.coarse.size() != 2) throw std::runtime_error("Bark preset coarse history must have two codebooks");
    for (size_t frame = 0; frame < preset.coarse[0].size(); ++frame)
        for (int book = 0; book < 2; ++book)
            coarse_history.push_back(preset.coarse[book][frame] + book * 1024 + 10000);
    const int max_sem_history = static_cast<int>(std::floor(630.0 / ratio));
    int sem_count = std::min<int>({max_sem_history, static_cast<int>(sem_history.size() / 2 * 2),
                                   static_cast<int>(std::floor(coarse_history.size() / ratio))});
    int coarse_count = static_cast<int>(std::round(sem_count * ratio));
    if (sem_count > 0) sem_history.erase(sem_history.begin(), sem_history.end() - sem_count);
    if (coarse_count > 0) coarse_history.erase(coarse_history.begin(), coarse_history.end() - coarse_count);
    if (coarse_history.size() >= 2) coarse_history.resize(coarse_history.size() - 2);
    const size_t original_history = coarse_history.size();
    std::vector<int32_t> all_semantic = sem_history;
    all_semantic.insert(all_semantic.end(), semantic.begin(), semantic.end());
    const int generated_length = static_cast<int>(std::round(std::floor(semantic.size() * ratio / 2.0) * 2.0));
    for (int total = 0; total < generated_length;) {
        const int semantic_index = sem_count + static_cast<int>(std::round(total / ratio));
        const int begin = std::max(0, semantic_index - max_sem_history);
        // This deliberately takes the suffix beginning at semantic_index's
        // history boundary and then its first 256 values.  It matches Bark's
        // reference x_semantic[:, max(0, semantic_idx-max_history):][:, :256].
        std::vector<int32_t> input(all_semantic.begin() + begin, all_semantic.end());
        if (input.size() > 256) input.resize(256);
        input.resize(256, 12048);
        input.push_back(12050);
        const size_t take = std::min<size_t>(630, coarse_history.size());
        input.insert(input.end(), coarse_history.end() - static_cast<std::ptrdiff_t>(take), coarse_history.end());
        const int window = std::min(60, generated_length - total);
        for (int i = 0; i < window; ++i) {
            auto logits = model.causal_logits({input});
            const int book = total % 2;
            const int32_t id = sample(std::move(logits), 10000 + book * 1024,
                                      10000 + (book + 1) * 1024, options.top_k, options.temperature, rng);
            input.push_back(id);
            coarse_history.push_back(id);
            ++total;
        }
    }
    return {coarse_history.begin() + static_cast<std::ptrdiff_t>(original_history), coarse_history.end()};
}

std::vector<std::vector<int32_t>> fine_tokens(BarkTransformer & model, const BarkSpeakerPreset & preset,
                                              const std::vector<int32_t> & coarse) {
    const int64_t frames = static_cast<int64_t>(coarse.size() / 2);
    std::vector<std::vector<int32_t>> values(8);
    for (int64_t frame = 0; frame < frames; ++frame) for (int book = 0; book < 2; ++book)
        values[book].push_back((coarse[static_cast<size_t>(frame * 2 + book)] - 10000) % 1024);
    const int64_t history = std::min<int64_t>(512, preset.fine.empty() ? 0 : preset.fine.front().size());
    std::vector<std::vector<int32_t>> input(8);
    for (int book = 0; book < 8; ++book) {
        if (history) input[book].insert(input[book].end(), preset.fine[book].end() - history, preset.fine[book].end());
        if (book < 2) input[book].insert(input[book].end(), values[book].begin(), values[book].end());
        else input[book].resize(static_cast<size_t>(history + frames), 1024);
    }
    const int64_t padded = std::max<int64_t>(1024, history + frames);
    for (auto & row : input) row.resize(static_cast<size_t>(padded), 1024);
    const int loops = std::max(0, static_cast<int>(std::ceil((frames - (1024 - history)) / 512.0))) + 1;
    for (int outer = 0; outer < loops; ++outer) {
        const int64_t start = std::min<int64_t>(outer * 512, padded - 1024);
        const int64_t fill = std::min<int64_t>(history + outer * 512, padded - 512);
        const int64_t relative = fill - start;
        std::vector<std::vector<int32_t>> window(8);
        for (int book = 0; book < 8; ++book) window[book].assign(input[book].begin() + start, input[book].begin() + start + 1024);
        for (int book = 2; book < 8; ++book) {
            const auto logits = model.fine_logits(window, book);
            for (int64_t position = relative; position < 1024; ++position) {
                const auto begin = logits.begin() + position * 1056;
                window[book][static_cast<size_t>(position)] = static_cast<int32_t>(std::distance(begin,
                    std::max_element(begin, begin + 1024)));
            }
        }
        for (int book = 2; book < 8; ++book)
            std::copy(window[book].begin() + relative, window[book].end(), input[book].begin() + fill);
    }
    for (int book = 0; book < 8; ++book)
        values[book].assign(input[book].begin() + history, input[book].begin() + history + frames);
    return values;
}

}  // namespace

BarkGenerator::BarkGenerator(std::shared_ptr<const BarkAssets> assets, engine::core::ExecutionContext & execution,
                             engine::assets::TensorStorageType transformer_storage,
                             engine::assets::TensorStorageType codec_storage)
    : assets_(std::move(assets)), tokenizer_(assets_->resources.read_text("tokenizer_json")),
      semantic_(assets_, execution, "semantic", assets_->config.semantic, true, transformer_storage),
      coarse_(assets_, execution, "coarse_acoustics", assets_->config.coarse, true, transformer_storage),
      fine_(assets_, execution, "fine_acoustics", assets_->config.fine, false, transformer_storage),
      codec_(assets_, execution, codec_storage) {}

std::vector<float> BarkGenerator::synthesize(const std::string & text, const BarkGenerationOptions & options) const {
    const auto found = assets_->presets.find(options.voice_id);
    if (found == assets_->presets.end()) throw std::runtime_error("unknown Bark voice_id: " + options.voice_id);
    std::mt19937_64 rng(options.seed);
    auto semantic = semantic_tokens(const_cast<BarkTransformer &>(semantic_), tokenizer_, found->second, text, options, rng);
    auto coarse = coarse_tokens(const_cast<BarkTransformer &>(coarse_), found->second, semantic, options, rng);
    return codec_.decode(fine_tokens(const_cast<BarkTransformer &>(fine_), found->second, coarse));
}

}  // namespace engine::models::bark_tts
