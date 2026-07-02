#include "engine/models/moss_tts_local/generator.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

namespace engine::models::moss_tts_local {
namespace {

// Projects a hidden-state vector through a [rows, hidden] row-major weight matrix. The
// audio codebook heads are tied to the audio embeddings and the gate head is tiny, so
// these products are cheap to run on the host.
std::vector<float> project(
    const std::vector<float> & weight,
    const std::vector<float> & hidden,
    int64_t rows,
    int64_t hidden_size) {
    std::vector<float> logits(static_cast<size_t>(rows));
    for (int64_t row = 0; row < rows; ++row) {
        const float * weight_row = weight.data() + static_cast<size_t>(row) * hidden_size;
        double accumulator = 0.0;
        for (int64_t index = 0; index < hidden_size; ++index) {
            accumulator += static_cast<double>(weight_row[index]) * static_cast<double>(hidden[static_cast<size_t>(index)]);
        }
        logits[static_cast<size_t>(row)] = static_cast<float>(accumulator);
    }
    return logits;
}

int32_t argmax_index(const std::vector<float> & logits) {
    if (logits.empty()) {
        throw std::runtime_error("MOSS-TTS-Local sampler received empty logits");
    }
    size_t best = 0;
    for (size_t index = 1; index < logits.size(); ++index) {
        if (logits[index] > logits[best]) {
            best = index;
        }
    }
    return static_cast<int32_t>(best);
}

void apply_repetition_penalty(
    std::vector<float> & logits,
    const std::vector<int32_t> & previous_codes,
    float penalty) {
    if (penalty == 1.0F) {
        return;
    }
    if (penalty <= 0.0F) {
        throw std::runtime_error("MOSS-TTS-Local repetition penalty must be positive");
    }
    std::unordered_set<int32_t> seen_codes;
    for (const int32_t code : previous_codes) {
        if (code < 0 || code >= static_cast<int32_t>(logits.size())) {
            continue;
        }
        if (!seen_codes.insert(code).second) {
            continue;
        }
        float & value = logits[static_cast<size_t>(code)];
        value = value < 0.0F ? value * penalty : value / penalty;
    }
}

int32_t sample_index(
    const std::vector<float> & logits,
    int top_k,
    float top_p,
    float temperature,
    std::mt19937 & rng) {
    if (temperature <= 0.0F) {
        throw std::runtime_error("MOSS-TTS-Local sampler temperature must be positive");
    }
    std::vector<int32_t> indices;
    indices.reserve(logits.size());
    for (size_t index = 0; index < logits.size(); ++index) {
        if (std::isfinite(logits[index])) {
            indices.push_back(static_cast<int32_t>(index));
        }
    }
    if (indices.empty()) {
        throw std::runtime_error("MOSS-TTS-Local sampler has no finite logits");
    }
    std::sort(indices.begin(), indices.end(), [&](int32_t lhs, int32_t rhs) {
        return logits[static_cast<size_t>(lhs)] > logits[static_cast<size_t>(rhs)];
    });
    if (top_k > 0 && static_cast<int>(indices.size()) > top_k) {
        indices.resize(static_cast<size_t>(top_k));
    }

    const float max_logit = logits[static_cast<size_t>(indices.front())] / temperature;
    std::vector<double> weights;
    weights.reserve(indices.size());
    double total = 0.0;
    for (const int32_t index : indices) {
        const double weight = std::exp(static_cast<double>(logits[static_cast<size_t>(index)] / temperature - max_logit));
        weights.push_back(weight);
        total += weight;
    }
    if (top_p > 0.0F && top_p < 1.0F) {
        double cumulative = 0.0;
        size_t keep = weights.size();
        for (size_t index = 0; index < weights.size(); ++index) {
            cumulative += weights[index] / total;
            if (cumulative >= top_p) {
                keep = index + 1;
                break;
            }
        }
        indices.resize(keep);
        weights.resize(keep);
    }
    std::discrete_distribution<size_t> distribution(weights.begin(), weights.end());
    return indices[distribution(rng)];
}

}  // namespace

MossGenerator::MossGenerator(
    std::shared_ptr<const MossTTSLocalAssets> assets,
    const MossBackboneRuntime & backbone,
    const MossDepthTransformer & depth)
    : assets_(std::move(assets)),
      backbone_(backbone),
      depth_(depth) {
    if (assets_ == nullptr) {
        throw std::runtime_error("MOSS-TTS-Local generator requires assets");
    }
    if (assets_->model_weights == nullptr) {
        throw std::runtime_error("MOSS-TTS-Local generator requires model weights");
    }
    const auto & config = assets_->config;
    hidden_size_ = config.backbone.hidden_size;
    num_codebooks_ = config.num_codebooks;
    if (config.local_text_head_mode != "binary") {
        throw std::runtime_error("MOSS-TTS-Local generator only supports the binary local text head");
    }
    const auto & source = *assets_->model_weights;
    audio_embeddings_.reserve(static_cast<size_t>(num_codebooks_));
    for (int64_t codebook = 0; codebook < num_codebooks_; ++codebook) {
        const int64_t codebook_size = config.audio_codebook_sizes.empty()
            ? config.audio_vocab_size
            : config.audio_codebook_sizes[static_cast<size_t>(codebook)];
        if (codebook_size <= 0) {
            throw std::runtime_error("MOSS-TTS-Local generator has an invalid audio codebook size");
        }
        audio_embeddings_.push_back(source.require_f32(
            "audio_embeddings." + std::to_string(codebook) + ".weight", {codebook_size, hidden_size_}));
    }
    local_text_head_ = source.require_f32("local_text_lm_head.weight", {2, hidden_size_});
}

std::vector<std::vector<int32_t>> MossGenerator::generate(
    const std::vector<int32_t> & text_tokens,
    const std::vector<int32_t> & audio_codes,
    const MossGenerationOptions & options) const {
    const auto & config = assets_->config;
    const int64_t hidden = hidden_size_;
    const int64_t n_vq = num_codebooks_;
    const int32_t pad_code = static_cast<int32_t>(config.audio_pad_token_id);
    const int32_t assistant_slot = static_cast<int32_t>(config.audio_assistant_slot_token_id);

    if (text_tokens.empty()) {
        throw std::runtime_error("MOSS-TTS-Local generation requires a non-empty prompt");
    }
    if (static_cast<int64_t>(audio_codes.size()) != static_cast<int64_t>(text_tokens.size()) * n_vq) {
        throw std::runtime_error("MOSS-TTS-Local generation audio codes must be [seq, n_vq]");
    }

    std::vector<std::vector<int32_t>> generated_frames;
    std::vector<std::vector<int32_t>> code_history(static_cast<size_t>(n_vq));
    std::mt19937 rng(options.seed);

    // Sums the audio-codebook embeddings for one decoder row into the additive bias the
    // backbone expects (padding codes contribute nothing).
    const auto bias_for = [&](const int32_t * codes) {
        std::vector<float> bias(static_cast<size_t>(hidden), 0.0F);
        for (int64_t codebook = 0; codebook < n_vq; ++codebook) {
            const int32_t code = codes[codebook];
            if (code == pad_code) {
                continue;
            }
            const float * embedding =
                audio_embeddings_[static_cast<size_t>(codebook)].data() + static_cast<size_t>(code) * hidden;
            for (int64_t index = 0; index < hidden; ++index) {
                bias[static_cast<size_t>(index)] += embedding[index];
            }
        }
        return bias;
    };

    // Prefill the whole prompt in a single batched forward (fills the cache in one pass
    // instead of one graph launch per token), then generate one row at a time from the cache.
    // The last prompt position's hidden state seeds the first generated frame.
    const int64_t prefix_len = static_cast<int64_t>(text_tokens.size());
    backbone_.begin_generation(prefix_len + options.max_new_frames);
    std::vector<float> prefix_bias(static_cast<size_t>(prefix_len * hidden), 0.0F);
    for (int64_t position = 0; position < prefix_len; ++position) {
        const std::vector<float> row = bias_for(audio_codes.data() + position * n_vq);
        std::copy(row.begin(), row.end(), prefix_bias.begin() + position * hidden);
    }
    std::vector<float> last_hidden = backbone_.prefill(text_tokens, prefix_bias);

    for (int64_t frame = 0; frame < options.max_new_frames; ++frame) {
        // Seed the depth transformer with the backbone hidden state (local position 0).
        std::vector<float> local_embeds = last_hidden;
        std::vector<float> local_hidden = depth_.forward(local_embeds, 1);

        // Binary gate: continue with the assistant slot or stop on the audio-end token.
        std::vector<float> gate_logits = project(local_text_head_, local_hidden, 2, hidden);
        const int32_t gate_index = options.do_sample
            ? sample_index(gate_logits, options.text_top_k, options.text_top_p, options.text_temperature, rng)
            : argmax_index(gate_logits);
        if (gate_index != 0) {
            break;
        }

        std::vector<int32_t> frame_codes(static_cast<size_t>(n_vq));
        for (int64_t codebook = 0; codebook < n_vq; ++codebook) {
            const int64_t codebook_size =
                static_cast<int64_t>(audio_embeddings_[static_cast<size_t>(codebook)].size()) / hidden;
            std::vector<float> logits =
                project(audio_embeddings_[static_cast<size_t>(codebook)], local_hidden, codebook_size, hidden);
            apply_repetition_penalty(
                logits, code_history[static_cast<size_t>(codebook)], options.audio_repetition_penalty);
            const int32_t code = options.do_sample
                ? sample_index(logits, options.audio_top_k, options.audio_top_p, options.audio_temperature, rng)
                : argmax_index(logits);
            frame_codes[static_cast<size_t>(codebook)] = code;
            code_history[static_cast<size_t>(codebook)].push_back(code);

            if (codebook + 1 < n_vq) {
                const float * embedding =
                    audio_embeddings_[static_cast<size_t>(codebook)].data() + static_cast<size_t>(code) * hidden;
                local_embeds.insert(local_embeds.end(), embedding, embedding + hidden);
                local_hidden = depth_.forward(local_embeds, codebook + 2);
            }
        }
        generated_frames.push_back(frame_codes);

        // Append the emitted frame as the next decoder row (assistant slot + codes) and
        // advance the backbone cache; the returned hidden seeds the next frame.
        last_hidden = backbone_.step(assistant_slot, bias_for(frame_codes.data()));
    }

    return generated_frames;
}

}  // namespace engine::models::moss_tts_local
