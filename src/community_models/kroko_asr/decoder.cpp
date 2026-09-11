#include "engine/community_models/kroko_asr/decoder.h"

#include "engine/framework/debug/profiler.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <queue>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

namespace engine::models::kroko_asr {
namespace {

constexpr int64_t kHidden = 512;
constexpr int64_t kGroups = 128;
constexpr int64_t kInputsPerGroup = 4;
constexpr size_t kPredictorCacheEntries = 4096;

double log_add(double lhs, double rhs) {
    if (lhs < rhs) {
        std::swap(lhs, rhs);
    }
    if (!std::isfinite(rhs)) {
        return lhs;
    }
    return lhs + std::log1p(std::exp(rhs - lhs));
}

class HotwordGraph {
public:
    explicit HotwordGraph(
        const std::vector<std::vector<int32_t>> & hotwords,
        float score) {
        nodes_.push_back(Node{});
        nodes_.front().fail = 0;
        for (const auto & tokens : hotwords) {
            if (tokens.empty()) {
                continue;
            }
            int32_t node = 0;
            for (size_t index = 0; index < tokens.size(); ++index) {
                const int32_t token = tokens[index];
                auto [it, inserted] =
                    nodes_[static_cast<size_t>(node)].next.emplace(
                        token, static_cast<int32_t>(nodes_.size()));
                if (inserted) {
                    Node child;
                    child.token = token;
                    child.token_score = score;
                    child.node_score =
                        nodes_[static_cast<size_t>(node)].node_score + score;
                    child.is_end = index + 1 == tokens.size();
                    child.output_score =
                        child.is_end ? child.node_score : 0.0F;
                    nodes_.push_back(std::move(child));
                } else {
                    Node & child =
                        nodes_[static_cast<size_t>(it->second)];
                    child.token_score =
                        std::max(child.token_score, score);
                    child.node_score =
                        nodes_[static_cast<size_t>(node)].node_score +
                        child.token_score;
                    child.is_end =
                        child.is_end || index + 1 == tokens.size();
                    child.output_score =
                        child.is_end ? child.node_score : 0.0F;
                }
                node = it->second;
            }
        }
        fill_failure_links();
    }

    std::pair<float, int32_t> forward(
        int32_t state,
        int32_t token) const {
        if (state < 0 ||
            state >= static_cast<int32_t>(nodes_.size())) {
            state = 0;
        }
        int32_t node = state;
        float score = 0.0F;
        const auto direct =
            nodes_[static_cast<size_t>(node)].next.find(token);
        if (direct !=
            nodes_[static_cast<size_t>(node)].next.end()) {
            node = direct->second;
            score =
                nodes_[static_cast<size_t>(node)].token_score;
        } else {
            node = nodes_[static_cast<size_t>(node)].fail;
            while (node != 0 &&
                   nodes_[static_cast<size_t>(node)].next.count(token) == 0) {
                node = nodes_[static_cast<size_t>(node)].fail;
            }
            if (const auto it =
                    nodes_[static_cast<size_t>(node)].next.find(token);
                it != nodes_[static_cast<size_t>(node)].next.end()) {
                node = it->second;
            }
            score =
                nodes_[static_cast<size_t>(node)].node_score -
                nodes_[static_cast<size_t>(state)].node_score;
        }

        const Node & current =
            nodes_[static_cast<size_t>(node)];
        if (current.output_score != 0.0F) {
            const int32_t matched =
                current.is_end ? node : current.output;
            const float matched_score =
                matched >= 0
                ? nodes_[static_cast<size_t>(matched)].node_score
                : current.node_score;
            return {
                score + matched_score - current.node_score,
                0,
            };
        }
        return {
            score + current.output_score,
            node,
        };
    }

private:
    struct Node {
        std::unordered_map<int32_t, int32_t> next;
        int32_t token = -1;
        int32_t fail = 0;
        int32_t output = -1;
        float token_score = 0.0F;
        float node_score = 0.0F;
        float output_score = 0.0F;
        bool is_end = false;
    };

    void fill_failure_links() {
        std::queue<int32_t> pending;
        for (const auto & [token, child] : nodes_.front().next) {
            (void)token;
            nodes_[static_cast<size_t>(child)].fail = 0;
            pending.push(child);
        }
        while (!pending.empty()) {
            const int32_t parent = pending.front();
            pending.pop();
            for (const auto & [token, child] :
                 nodes_[static_cast<size_t>(parent)].next) {
                int32_t failure =
                    nodes_[static_cast<size_t>(parent)].fail;
                while (failure != 0 &&
                       nodes_[static_cast<size_t>(failure)].next.count(token) == 0) {
                    failure =
                        nodes_[static_cast<size_t>(failure)].fail;
                }
                if (const auto it =
                        nodes_[static_cast<size_t>(failure)].next.find(token);
                    it != nodes_[static_cast<size_t>(failure)].next.end() &&
                    it->second != child) {
                    failure = it->second;
                }
                Node & node = nodes_[static_cast<size_t>(child)];
                node.fail = failure;
                int32_t output = failure;
                while (output != 0 &&
                       !nodes_[static_cast<size_t>(output)].is_end) {
                    output =
                        nodes_[static_cast<size_t>(output)].fail;
                }
                node.output =
                    nodes_[static_cast<size_t>(output)].is_end
                    ? output
                    : -1;
                if (node.output >= 0) {
                    node.output_score +=
                        nodes_[static_cast<size_t>(node.output)]
                            .output_score;
                }
                pending.push(child);
            }
        }
    }

    std::vector<Node> nodes_;
};

struct BeamHypothesis {
    std::vector<int32_t> ids;
    std::vector<int64_t> frame_indices;
    std::array<int32_t, 2> context{};
    double log_prob = 0.0;
    int32_t context_state = 0;
    int64_t trailing_blank_frames = 0;
};

bool same_tokens(
    const BeamHypothesis & lhs,
    const BeamHypothesis & rhs) {
    return lhs.context == rhs.context &&
        lhs.ids == rhs.ids;
}

}  // namespace

struct KrokoTransducerDecoder::Impl {
    std::vector<BeamHypothesis> hypotheses;
    std::unique_ptr<HotwordGraph> hotword_graph;
    std::unordered_map<uint64_t, std::array<float, 512>>
        predictor_cache;
};

KrokoTransducerDecoder::KrokoTransducerDecoder(
    std::shared_ptr<const KrokoASRAssets> assets)
    : assets_(std::move(assets)),
      impl_(std::make_unique<Impl>()) {
    if (assets_ == nullptr) {
        throw std::runtime_error(
            "Kroko transducer decoder requires assets");
    }
    const auto & source = *assets_->weights;
    const int64_t vocab = assets_->config.vocab_size;
    embedding_ = source.require_f32(
        "decoder.decoder.embedding.weight",
        {vocab, kHidden});
    conv_ = source.require_f32(
        "decoder.decoder.conv.weight",
        {kHidden, kInputsPerGroup, 2});
    decoder_projection_ = source.require_f32(
        "decoder.decoder_proj.weight",
        {kHidden, kHidden});
    decoder_bias_ = source.require_f32(
        "decoder.decoder_proj.bias",
        {kHidden});
    joiner_projection_ = source.require_f32(
        "joiner.output_linear.weight",
        {vocab, kHidden});
    joiner_bias_ = source.require_f32(
        "joiner.output_linear.bias",
        {vocab});
    joiner_scores_.resize(static_cast<size_t>(vocab));
    reset();
}

KrokoTransducerDecoder::~KrokoTransducerDecoder() =
    default;

void KrokoTransducerDecoder::configure(
    KrokoDecoderOptions options) {
    if (options.max_active_paths <= 0 ||
        options.max_active_paths > 64) {
        throw std::runtime_error(
            "Kroko max_active_paths must be between 1 and 64");
    }
    if (!std::isfinite(options.blank_penalty) ||
        options.blank_penalty < 0.0F) {
        throw std::runtime_error(
            "Kroko blank_penalty must be a finite non-negative number");
    }
    if (!std::isfinite(options.hotwords_score) ||
        options.hotwords_score < 0.0F) {
        throw std::runtime_error(
            "Kroko hotwords_score must be a finite non-negative number");
    }
    if (!options.hotwords.empty() &&
        options.method !=
            KrokoDecodingMethod::ModifiedBeamSearch) {
        throw std::runtime_error(
            "Kroko hotwords require decoding_method=modified_beam_search");
    }
    options_ = std::move(options);
    reset();
}

std::array<float, 512>
KrokoTransducerDecoder::predictor(
    const std::array<int32_t, 2> & context) const {
    std::array<float, kHidden * 2> embedded{};
    const int64_t vocab = assets_->config.vocab_size;
    for (int64_t position = 0; position < 2; ++position) {
        const int32_t token =
            context[static_cast<size_t>(position)];
        if (token < 0) {
            continue;
        }
        if (token >= vocab) {
            throw std::runtime_error(
                "Kroko predictor token is outside the vocabulary");
        }
        std::copy_n(
            embedding_.data() +
                static_cast<int64_t>(token) * kHidden,
            kHidden,
            embedded.data() + position * kHidden);
    }

    std::array<float, kHidden> convolved{};
    for (int64_t output = 0; output < kHidden; ++output) {
        const int64_t group =
            output / (kHidden / kGroups);
        const int64_t input_start =
            group * kInputsPerGroup;
        float value = 0.0F;
        for (int64_t input = 0;
             input < kInputsPerGroup;
             ++input) {
            for (int64_t position = 0;
                 position < 2;
                 ++position) {
                value +=
                    embedded[static_cast<size_t>(
                        position * kHidden +
                        input_start + input)] *
                    conv_[static_cast<size_t>(
                        (output * kInputsPerGroup +
                         input) *
                            2 +
                        position)];
            }
        }
        convolved[static_cast<size_t>(output)] =
            std::max(value, 0.0F);
    }

    std::array<float, kHidden> result{};
    for (int64_t output = 0; output < kHidden; ++output) {
        double value =
            decoder_bias_[static_cast<size_t>(output)];
        const float * weight =
            decoder_projection_.data() + output * kHidden;
        for (int64_t input = 0; input < kHidden; ++input) {
            value +=
                static_cast<double>(weight[input]) *
                static_cast<double>(
                    convolved[static_cast<size_t>(input)]);
        }
        result[static_cast<size_t>(output)] =
            static_cast<float>(value);
    }
    return result;
}

void KrokoTransducerDecoder::join_scores(
    const float * encoder_frame,
    const std::array<float, 512> & decoder_output,
    std::vector<float> & scores) const {
    const int64_t vocab = assets_->config.vocab_size;
    scores.resize(static_cast<size_t>(vocab));
    std::array<float, kHidden> activated{};
    for (int64_t hidden = 0; hidden < kHidden; ++hidden) {
        activated[static_cast<size_t>(hidden)] =
            std::tanh(
                encoder_frame[hidden] +
                decoder_output[static_cast<size_t>(hidden)]);
    }
#pragma omp parallel for schedule(static) if (vocab >= 256)
    for (int64_t token = 0; token < vocab; ++token) {
        double value =
            joiner_bias_[static_cast<size_t>(token)];
        const float * weight =
            joiner_projection_.data() + token * kHidden;
        for (int64_t hidden = 0; hidden < kHidden; ++hidden) {
            value +=
                static_cast<double>(weight[hidden]) *
                static_cast<double>(
                    activated[static_cast<size_t>(hidden)]);
        }
        scores[static_cast<size_t>(token)] =
            static_cast<float>(value);
    }
}

void KrokoTransducerDecoder::reset(int64_t frame_offset) {
    if (frame_offset < 0) {
        throw std::runtime_error(
            "Kroko decoder frame offset cannot be negative");
    }
    context_ = {
        -1,
        static_cast<int32_t>(assets_->config.blank_id),
    };
    decoder_output_ = predictor(context_);
    decoded_ = KrokoDecodedTokens{};
    decoded_frames_ = frame_offset;
    trailing_blank_frames_ = 0;
    impl_->predictor_cache.clear();
    impl_->hotword_graph =
        options_.hotwords.empty()
        ? nullptr
        : std::make_unique<HotwordGraph>(
              options_.hotwords,
              options_.hotwords_score);
    impl_->hypotheses.clear();
    BeamHypothesis initial;
    initial.context = context_;
    impl_->hypotheses.push_back(std::move(initial));
}

void KrokoTransducerDecoder::reset_segment(
    int64_t frame_offset) {
    if (frame_offset < 0) {
        throw std::runtime_error(
            "Kroko decoder frame offset cannot be negative");
    }
    decoded_ = KrokoDecodedTokens{};
    decoded_frames_ = frame_offset;
    trailing_blank_frames_ = 0;
    impl_->hotword_graph =
        options_.hotwords.empty()
        ? nullptr
        : std::make_unique<HotwordGraph>(
              options_.hotwords,
              options_.hotwords_score);
    if (options_.method ==
        KrokoDecodingMethod::GreedySearch) {
        decoder_output_ = predictor(context_);
        return;
    }

    std::vector<BeamHypothesis> retained;
    retained.reserve(impl_->hypotheses.size());
    for (auto hypothesis : impl_->hypotheses) {
        hypothesis.ids.clear();
        hypothesis.frame_indices.clear();
        hypothesis.context_state = 0;
        hypothesis.trailing_blank_frames = 0;
        const auto existing = std::find_if(
            retained.begin(),
            retained.end(),
            [&](const BeamHypothesis & value) {
                return value.context ==
                    hypothesis.context;
            });
        if (existing == retained.end()) {
            retained.push_back(std::move(hypothesis));
        } else {
            existing->log_prob = log_add(
                existing->log_prob,
                hypothesis.log_prob);
        }
    }
    if (retained.empty()) {
        BeamHypothesis initial;
        initial.context = context_;
        retained.push_back(std::move(initial));
    }
    impl_->hypotheses = std::move(retained);
}

void KrokoTransducerDecoder::append_greedy(
    const std::vector<float> & encoder_output,
    int64_t frames,
    int64_t hidden_size) {
    const int32_t blank =
        static_cast<int32_t>(assets_->config.blank_id);
    const int32_t unknown =
        static_cast<int32_t>(assets_->config.unk_id);
    for (int64_t frame = 0; frame < frames; ++frame) {
        join_scores(
            encoder_output.data() + frame * hidden_size,
            decoder_output_,
            joiner_scores_);
        joiner_scores_[static_cast<size_t>(blank)] -=
            options_.blank_penalty;
        int32_t token = 0;
        float best =
            -std::numeric_limits<float>::infinity();
        for (int32_t candidate = 0;
             candidate <
                 static_cast<int32_t>(joiner_scores_.size());
             ++candidate) {
            if (joiner_scores_[static_cast<size_t>(candidate)] >
                best) {
                best =
                    joiner_scores_[static_cast<size_t>(candidate)];
                token = candidate;
            }
        }
        if (token == blank || token == unknown) {
            ++trailing_blank_frames_;
            continue;
        }
        trailing_blank_frames_ = 0;
        decoded_.ids.push_back(token);
        decoded_.frame_indices.push_back(
            decoded_frames_ + frame);
        context_[0] = context_[1];
        context_[1] = token;
        decoder_output_ = predictor(context_);
    }
}

void KrokoTransducerDecoder::append_modified_beam(
    const std::vector<float> & encoder_output,
    int64_t frames,
    int64_t hidden_size) {
    struct Candidate {
        float score = 0.0F;
        int32_t hypothesis = 0;
        int32_t token = 0;
        int64_t order = 0;
    };

    const int32_t blank =
        static_cast<int32_t>(assets_->config.blank_id);
    const int32_t unknown =
        static_cast<int32_t>(assets_->config.unk_id);
    const int32_t vocab =
        static_cast<int32_t>(assets_->config.vocab_size);
    std::vector<std::vector<float>> all_scores;
    std::vector<Candidate> candidates;
    for (int64_t frame = 0; frame < frames; ++frame) {
        all_scores.assign(
            impl_->hypotheses.size(),
            std::vector<float>{});
        candidates.clear();
        candidates.reserve(
            impl_->hypotheses.size() *
            static_cast<size_t>(vocab));

        for (size_t hypothesis_index = 0;
             hypothesis_index < impl_->hypotheses.size();
             ++hypothesis_index) {
            const auto & hypothesis =
                impl_->hypotheses[hypothesis_index];
            const int32_t before_previous =
                hypothesis.context[0];
            const int32_t previous =
                hypothesis.context[1];
            const uint64_t cache_key =
                (static_cast<uint64_t>(
                     static_cast<uint32_t>(
                         before_previous + 1))
                << 32U) |
                static_cast<uint32_t>(previous + 1);
            if (impl_->predictor_cache.size() >=
                    kPredictorCacheEntries &&
                impl_->predictor_cache.count(cache_key) == 0) {
                impl_->predictor_cache.clear();
            }
            auto [cache, inserted] =
                impl_->predictor_cache.emplace(
                    cache_key,
                    std::array<float, 512>{});
            if (inserted) {
                cache->second = predictor(
                    {before_previous, previous});
            }
            auto & scores = all_scores[hypothesis_index];
            join_scores(
                encoder_output.data() +
                    frame * hidden_size,
                cache->second,
                scores);
            scores[static_cast<size_t>(blank)] -=
                options_.blank_penalty;
            const float maximum =
                *std::max_element(scores.begin(), scores.end());
            float normalizer = 0.0F;
            for (const float score : scores) {
                normalizer +=
                    std::exp(
                        score - maximum);
            }
            const float log_normalizer =
                maximum +
                std::log(normalizer);
            const float previous_log_prob =
                static_cast<float>(hypothesis.log_prob);
            for (int32_t token = 0; token < vocab; ++token) {
                candidates.push_back(Candidate{
                    previous_log_prob +
                        scores[static_cast<size_t>(token)] -
                        log_normalizer,
                    static_cast<int32_t>(hypothesis_index),
                    token,
                    static_cast<int64_t>(hypothesis_index) *
                            vocab +
                        token,
                });
            }
        }

        const size_t keep = std::min(
            candidates.size(),
            static_cast<size_t>(options_.max_active_paths));
        std::partial_sort(
            candidates.begin(),
            candidates.begin() +
                static_cast<std::ptrdiff_t>(keep),
            candidates.end(),
            [](const Candidate & lhs,
               const Candidate & rhs) {
                return lhs.score != rhs.score
                    ? lhs.score > rhs.score
                    : lhs.order < rhs.order;
            });

        std::vector<BeamHypothesis> next;
        next.reserve(keep);
        for (size_t index = 0; index < keep; ++index) {
            const Candidate & candidate = candidates[index];
            BeamHypothesis hypothesis =
                impl_->hypotheses[
                    static_cast<size_t>(candidate.hypothesis)];
            hypothesis.log_prob = candidate.score;
            if (candidate.token == blank ||
                candidate.token == unknown) {
                ++hypothesis.trailing_blank_frames;
            } else {
                hypothesis.ids.push_back(candidate.token);
                hypothesis.frame_indices.push_back(
                    decoded_frames_ + frame);
                hypothesis.context[0] =
                    hypothesis.context[1];
                hypothesis.context[1] =
                    candidate.token;
                hypothesis.trailing_blank_frames = 0;
                if (impl_->hotword_graph != nullptr) {
                    const auto [boost, state] =
                        impl_->hotword_graph->forward(
                            hypothesis.context_state,
                            candidate.token);
                    hypothesis.log_prob += boost;
                    hypothesis.context_state = state;
                }
            }
            const auto existing = std::find_if(
                next.begin(),
                next.end(),
                [&](const BeamHypothesis & value) {
                    return same_tokens(value, hypothesis);
                });
            if (existing == next.end()) {
                next.push_back(std::move(hypothesis));
            } else {
                existing->log_prob = log_add(
                    existing->log_prob,
                    hypothesis.log_prob);
            }
        }
        impl_->hypotheses = std::move(next);
    }

    const auto best = std::max_element(
        impl_->hypotheses.begin(),
        impl_->hypotheses.end(),
        [](const BeamHypothesis & lhs,
           const BeamHypothesis & rhs) {
            const double lhs_score =
                lhs.log_prob /
                static_cast<double>(lhs.ids.size() + 2);
            const double rhs_score =
                rhs.log_prob /
                static_cast<double>(rhs.ids.size() + 2);
            return lhs_score < rhs_score;
        });
    if (best == impl_->hypotheses.end()) {
        throw std::runtime_error(
            "Kroko modified beam search produced no hypotheses");
    }
    decoded_.ids = best->ids;
    decoded_.frame_indices = best->frame_indices;
    trailing_blank_frames_ =
        best->trailing_blank_frames;
}

const KrokoDecodedTokens &
KrokoTransducerDecoder::append(
    const std::vector<float> & encoder_output,
    int64_t frames,
    int64_t hidden_size) {
    if (frames <= 0 || hidden_size != kHidden ||
        static_cast<int64_t>(encoder_output.size()) <
            frames * hidden_size) {
        throw std::runtime_error(
            "Kroko transducer decoder received invalid encoder output");
    }
    const auto start = std::chrono::steady_clock::now();
    if (options_.method ==
        KrokoDecodingMethod::GreedySearch) {
        append_greedy(
            encoder_output, frames, hidden_size);
    } else {
        append_modified_beam(
            encoder_output, frames, hidden_size);
    }
    decoded_frames_ += frames;
    engine::debug::timing_log_scalar(
        "kroko_asr.decoder_ms",
        engine::debug::elapsed_ms(start));
    engine::debug::trace_log_scalar(
        "kroko_asr.decoder.tokens",
        decoded_.ids.size());
    engine::debug::trace_log_scalar(
        "kroko_asr.decoder.trailing_blank_frames",
        trailing_blank_frames_);
    return decoded_;
}

const KrokoDecodedTokens &
KrokoTransducerDecoder::decoded() const noexcept {
    return decoded_;
}

int64_t
KrokoTransducerDecoder::decoded_frames() const noexcept {
    return decoded_frames_;
}

int64_t
KrokoTransducerDecoder::trailing_blank_frames() const noexcept {
    return trailing_blank_frames_;
}

KrokoDecodedTokens KrokoTransducerDecoder::decode(
    const std::vector<float> & encoder_output,
    int64_t frames,
    int64_t hidden_size) {
    reset();
    append(encoder_output, frames, hidden_size);
    return decoded_;
}

}  // namespace engine::models::kroko_asr
