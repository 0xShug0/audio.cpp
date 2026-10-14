#include "engine/community_models/fish_drama3/generator.h"

#include "engine/framework/debug/profiler.h"
#include "engine/framework/runtime/options.h"

#include <algorithm>
#include <cmath>
#include <deque>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>

namespace engine::models::fish_drama3 {
namespace {

int32_t sample(const std::vector<float> & logits, float temperature, int top_k,
               float top_p, std::mt19937 & random) {
    if (logits.empty()) throw std::runtime_error("Drama 3 sampler has no logits");
    std::vector<int32_t> order(logits.size());
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](int32_t left, int32_t right) { return logits[left] > logits[right]; });
    if (temperature == 0 || top_k == 1) return order.front();
    const double maximum = logits[order.front()] / temperature;
    std::vector<double> probabilities(order.size());
    double total = 0;
    for (size_t rank = 0; rank < order.size(); ++rank) {
        probabilities[rank] = std::exp(logits[order[rank]] / temperature - maximum);
        total += probabilities[rank];
    }
    if (!(total > 0) || !std::isfinite(total)) throw std::runtime_error("Drama 3 sampler encountered nonfinite logits");
    double cumulative = 0;
    for (size_t rank = 0; rank < order.size(); ++rank) {
        cumulative += probabilities[rank] / total;
        if (rank > 0 && (cumulative > top_p || rank >= static_cast<size_t>(top_k))) probabilities[rank] = 0;
    }
    std::discrete_distribution<size_t> choice(probabilities.begin(), probabilities.end());
    return order[choice(random)];
}

std::vector<float> last_hidden(const std::vector<float> & values, int64_t width) {
    if (values.size() < static_cast<size_t>(width)) throw std::runtime_error("Drama 3 backbone returned too few hidden values");
    return {values.end() - width, values.end()};
}

}  // namespace

Drama3Generator::Drama3Generator(std::shared_ptr<const Drama3Assets> assets,
    core::ExecutionContext & execution, const runtime::SessionOptions & options)
    : assets_(std::move(assets)), tokenizer_(assets_->tokenizer_assets),
      cache_steps_(runtime::parse_positive_i64_option(options.options, {"fish_drama3.context_length"}, 2048)) {
    if (cache_steps_ > 262144) throw std::runtime_error("Drama 3 context_length exceeds the checkpoint's maximum position count");
    auto backbone_config = assets_->backbone;
    backbone_config.decode_cache_steps = cache_steps_;
    backbone_config.cpu_expert_layers = runtime::parse_i64_option(options.options, {"fish_drama3.cpu_expert_layers"}).value_or(4);
    backbone_ = std::make_unique<Drama3BackboneDecoderRuntime>(assets_->model_weights, backbone_config, execution,
        64 * 1024 * 1024, 8 * 1024 * 1024, assets::TensorStorageType::Native);
    tower_ = std::make_unique<Drama3Tower>(assets_->model_weights, assets_->tower, execution, cache_steps_);
    auto codec_backend = options.backend;
    if (const auto value = runtime::find_option(options.options, {"fish_drama3.codec_backend"})) {
        if (*value == "cpu") codec_backend.type = core::BackendType::Cpu;
        else if (*value != "same") throw std::runtime_error("fish_drama3.codec_backend requires cpu or same");
    }
    codec_execution_ = std::make_unique<core::ExecutionContext>(codec_backend);
    codecs::FishDacCodecRuntimeOptions codec_options;
    codec_options.weight_context_bytes = 16 * 1024 * 1024;
    const auto component = codecs::FishDacCodecComponent::load_from_tensor_source(assets_->codec_weights,
        assets_->codec, {}, codec_execution_->backend(), codec_backend.type, codec_options);
    codec_ = std::make_unique<codecs::FishDacCodecRuntime>(component, codec_backend,
        codec_backend.threads, codec_options.graph_arena_bytes);
    assets_->model_weights->release_storage();
    assets_->codec_weights->release_storage();
}

Drama3Generator::~Drama3Generator() = default;

codecs::FishDacCodes Drama3Generator::encode_reference(const runtime::AudioBuffer & audio) {
    auto result = codec_->encode_codes(audio);
    codec_->release_encode_graph();
    validate_drama3_codes(result, assets_->tower.codebook_size);
    return result;
}

Drama3Generation Drama3Generator::generate(const std::string & text, const std::vector<Drama3Take> & references,
    const std::vector<Drama3Take> & history, const Drama3Sampling & sampling, bool streaming,
    const std::function<void(const runtime::AudioBuffer &)> & on_audio) {
    if (!std::isfinite(sampling.temperature) || sampling.temperature < 0 ||
        !std::isfinite(sampling.top_p) || sampling.top_p <= 0 || sampling.top_p > 1 ||
        sampling.top_k <= 0 || sampling.max_frames <= 0) throw std::runtime_error("invalid Drama 3 sampling parameters");
    const auto prompt = build_drama3_prompt(*assets_, tokenizer_, text, references, history);
    if (sampling.max_frames > cache_steps_ ||
        prompt.size() > static_cast<size_t>(cache_steps_ - sampling.max_frames)) {
        throw std::runtime_error("Drama 3 prompt plus max_tokens exceeds fish_drama3.context_length; increase it or shorten history");
    }
    auto decoder = backbone_->create_decode_session(cache_steps_);
    decoder->reset();
    tower_->reset();
    std::vector<int32_t> ids;
    for (const auto & frame : prompt) ids.push_back(frame[0]);
    auto embeddings = backbone_->token_embedding(ids);
    tower_->mix_backbone_embeddings(embeddings, prompt);
    const auto prefill = decoder->prefill_embeddings(embeddings, static_cast<int64_t>(prompt.size()));
    tower_->prefill(prefill.hidden, prompt);
    auto hidden = last_hidden(prefill.hidden, assets_->backbone.hidden_size);
    Drama3Frame current = prompt.back();
    std::mt19937 random(sampling.seed);
    std::deque<int32_t> recent;
    std::vector<Drama3Frame> output_frames;
    Drama3Generation result;
    if (streaming) codec_->reset_decode_stream();
    for (int64_t step = 0; step < sampling.max_frames; ++step) {
        const auto logits = backbone_->lm_head(hidden);
        std::vector<float> candidates(static_cast<size_t>(assets_->tower.codebook_size + 1));
        candidates[0] = logits.at(assets_->eos);
        std::copy_n(logits.begin() + assets_->tower.semantic_begin, assets_->tower.codebook_size, candidates.begin() + 1);
        const int32_t first = sample(candidates, sampling.temperature, sampling.top_k, sampling.top_p, random);
        const int32_t second = sample(candidates, 1.0F, 30, 0.95F, random);
        const int32_t first_token = first == 0 ? assets_->eos : assets_->tower.semantic_begin + first - 1;
        const bool repeats = first != 0 && std::find(recent.begin(), recent.end(), first_token) != recent.end();
        const int32_t selected = repeats ? second : first;
        if (selected == 0) break;
        Drama3Frame next{};
        next[0] = assets_->tower.semantic_begin + selected - 1;
        next[1] = selected - 1;
        const auto shared_input = tower_->shared_input({current});
        auto tower_hidden = hidden;
        const int64_t position = decoder->valid_steps() - 1;
        const auto & config = assets_->tower;
        for (int64_t layer = 0; layer < config.shared_layers + config.residual_layers; ++layer) {
            const int64_t book = layer < config.shared_layers ? 0 : layer - config.shared_layers;
            const auto tower_output = tower_->run_layer(layer, tower_hidden, {next[1 + book]}, {1.0F},
                                                        layer == 0 ? shared_input : std::vector<float>{}, position);
            tower_hidden = tower_output.hidden;
            if (layer >= config.shared_layers) {
                next[2 + book] = sample(tower_output.logits, sampling.temperature, sampling.top_k, sampling.top_p, random);
            }
        }
        output_frames.push_back(next);
        recent.push_back(next[0]);
        if (recent.size() > 16) recent.pop_front();
        if (streaming) {
            codecs::FishDacCodes codes;
            codes.codebooks = 10;
            codes.frames = 1;
            codes.codes.assign(next.begin() + 1, next.end());
            auto chunk = codec_->decode_stream(codes);
            runtime::append_audio_buffer(result.audio, chunk);
            if (on_audio) on_audio(chunk);
        }
        if (step + 1 == sampling.max_frames) { result.reached_limit = true; break; }
        current = next;
        auto embedding = backbone_->token_embedding({current[0]});
        tower_->mix_backbone_embeddings(embedding, {current});
        hidden = decoder->run_embedding_step(embedding);
    }
    if (output_frames.empty()) throw std::runtime_error("Drama 3 ended before producing an audio frame");
    result.codes.codebooks = 10;
    result.codes.frames = static_cast<int64_t>(output_frames.size());
    result.codes.codes.resize(output_frames.size() * 10);
    for (int64_t book = 0; book < 10; ++book) {
        for (size_t row = 0; row < output_frames.size(); ++row) result.codes.codes[book * output_frames.size() + row] = output_frames[row][book + 1];
    }
    if (!streaming) {
        result.audio = codec_->decode_codes(result.codes);
        codec_->release_runtime_graphs();
    }
    if (result.reached_limit) std::cerr << "[fish_drama3] max_tokens reached; audio may be incomplete\n";
    return result;
}

}  // namespace engine::models::fish_drama3
