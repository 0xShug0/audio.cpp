#include "engine/models/sherpa_kws/keyword_decoder.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <memory>
#include <numeric>
#include <queue>
#include <stdexcept>
#include <unordered_map>
#include <utility>

namespace engine::models::sherpa_kws {
namespace {

double log_add(double lhs, double rhs) {
    if (lhs < rhs) std::swap(lhs, rhs);
    if (!std::isfinite(rhs)) return lhs;
    return lhs + std::log1p(std::exp(rhs - lhs));
}

struct ContextState {
    int32_t token = -1;
    float token_score = 0.0F;
    float node_score = 0.0F;
    float output_score = 0.0F;
    int level = 0;
    float threshold = 0.0F;
    bool is_end = false;
    std::string phrase;
    std::unordered_map<int32_t, std::unique_ptr<ContextState>> next;
    ContextState * fail = nullptr;
    ContextState * output = nullptr;
};

class ContextGraph {
public:
    explicit ContextGraph(const std::vector<KeywordDefinition> & keywords) {
        root_.fail = &root_;
        for (const auto & keyword : keywords) {
            auto * node = &root_;
            for (size_t i = 0; i < keyword.tokens.size(); ++i) {
                const int32_t token = keyword.tokens[i];
                auto [it, inserted] = node->next.emplace(token, nullptr);
                if (inserted) {
                    it->second = std::make_unique<ContextState>();
                    it->second->token = token;
                }
                auto * child = it->second.get();
                child->token_score = std::max(child->token_score, keyword.score);
                child->node_score = node->node_score + child->token_score;
                child->level = static_cast<int>(i + 1);
                if (i + 1 == keyword.tokens.size()) {
                    child->is_end = true;
                    child->output_score = child->node_score;
                    child->threshold = keyword.threshold;
                    child->phrase = keyword.phrase;
                }
                node = child;
            }
        }
        fill_failure_links();
    }

    ContextState * root() { return &root_; }

    std::pair<float, ContextState *> forward(ContextState * state, int32_t token) {
        ContextState * node = nullptr;
        float score = 0.0F;
        if (const auto it = state->next.find(token); it != state->next.end()) {
            node = it->second.get();
            score = node->token_score;
        } else {
            node = state->fail;
            while (node->next.count(token) == 0 && node->token != -1) node = node->fail;
            if (const auto it = node->next.find(token); it != node->next.end()) node = it->second.get();
            score = node->node_score - state->node_score;
        }
        return {score + node->output_score, node};
    }

    ContextState * matched(ContextState * state) const {
        if (state->is_end) return state;
        return state->output;
    }

private:
    void fill_failure_links() {
        std::queue<ContextState *> pending;
        for (auto & [token, child] : root_.next) {
            (void)token;
            child->fail = &root_;
            pending.push(child.get());
        }
        while (!pending.empty()) {
            ContextState * current = pending.front();
            pending.pop();
            for (auto & [token, child_ptr] : current->next) {
                ContextState * failure = current->fail;
                if (const auto it = failure->next.find(token); it != failure->next.end()) {
                    failure = it->second.get();
                } else {
                    failure = failure->fail;
                    while (failure->next.count(token) == 0 && failure->token != -1) failure = failure->fail;
                    if (const auto it = failure->next.find(token); it != failure->next.end()) failure = it->second.get();
                }
                ContextState * child = child_ptr.get();
                child->fail = failure;
                ContextState * output = failure;
                while (!output->is_end && output->token != -1) output = output->fail;
                child->output = output->is_end ? output : nullptr;
                if (child->output != nullptr) child->output_score += child->output->output_score;
                pending.push(child);
            }
        }
    }

    ContextState root_;
};

struct Hypothesis {
    std::vector<int32_t> tokens;
    std::vector<int64_t> timestamps;
    std::vector<float> probabilities;
    std::array<int32_t, 2> context{-1, 0};
    std::vector<float> decoder;
    double log_prob = 0.0;
    int trailing_blanks = 0;
    ContextState * state = nullptr;
};

bool same_tokens(const Hypothesis & lhs, const Hypothesis & rhs) {
    return lhs.tokens == rhs.tokens;
}

}  // namespace

struct KeywordDecoder::Impl {
    explicit Impl(std::shared_ptr<const SherpaKwsAssets> value)
        : assets(std::move(value)), scorer(assets) {}

    void initialize_hypotheses() {
        Hypothesis initial;
        initial.context = {-1, static_cast<int32_t>(assets->config.blank_id)};
        initial.tokens = {initial.context[0], initial.context[1]};
        initial.state = graph->root();
        hypotheses = {std::move(initial)};
    }

    std::shared_ptr<const SherpaKwsAssets> assets;
    TransducerScorer scorer;
    std::unique_ptr<ContextGraph> graph;
    std::vector<Hypothesis> hypotheses;
    int max_active_paths = 4;
    int trailing_blanks = 1;
    int64_t frame_offset = 0;
};

KeywordDecoder::KeywordDecoder(std::shared_ptr<const SherpaKwsAssets> assets)
    : impl_(std::make_unique<Impl>(std::move(assets))) {}
KeywordDecoder::~KeywordDecoder() = default;

void KeywordDecoder::configure(std::vector<KeywordDefinition> keywords,
                               int max_active_paths, int num_trailing_blanks) {
    if (keywords.empty()) throw std::runtime_error("sherpa KWS requires at least one keyword");
    if (max_active_paths < 1 || max_active_paths > 64 || num_trailing_blanks < 0) {
        throw std::runtime_error("invalid sherpa KWS decoder settings");
    }
    for (const auto & keyword : keywords) {
        if (keyword.tokens.empty() || keyword.phrase.empty() || !std::isfinite(keyword.score) ||
            !std::isfinite(keyword.threshold) || keyword.threshold < 0.0F || keyword.threshold > 1.0F) {
            throw std::runtime_error("invalid sherpa KWS keyword definition");
        }
    }
    impl_->graph = std::make_unique<ContextGraph>(keywords);
    impl_->max_active_paths = max_active_paths;
    impl_->trailing_blanks = num_trailing_blanks;
    reset();
}

void KeywordDecoder::reset() {
    impl_->frame_offset = 0;
    if (impl_->graph) impl_->initialize_hypotheses();
}

std::vector<KeywordDetection> KeywordDecoder::append(
    const std::vector<float> & encoder_output, int64_t frames, int64_t channels) {
    if (!impl_->graph || frames <= 0 || channels != impl_->assets->config.decoder_dim ||
        static_cast<int64_t>(encoder_output.size()) < frames * channels) {
        throw std::runtime_error("sherpa KWS decoder received invalid encoder output");
    }
    const int32_t blank = static_cast<int32_t>(impl_->assets->config.blank_id);
    const int32_t unknown = static_cast<int32_t>(impl_->assets->config.unk_id);
    const int32_t vocab = static_cast<int32_t>(impl_->assets->config.vocab_size);
    std::vector<KeywordDetection> detections;
    struct Candidate { double score; size_t hypothesis; int32_t token; float probability; };
    for (int64_t frame = 0; frame < frames; ++frame) {
        std::vector<Candidate> candidates;
        candidates.reserve(impl_->hypotheses.size() * static_cast<size_t>(vocab));
        for (size_t h = 0; h < impl_->hypotheses.size(); ++h) {
            auto & hypothesis = impl_->hypotheses[h];
            if (hypothesis.decoder.empty()) {
                hypothesis.decoder = impl_->scorer.predictor(hypothesis.context);
            }
            auto logits = impl_->scorer.score(encoder_output.data() + frame * channels,
                                               hypothesis.decoder);
            const float maximum = *std::max_element(logits.begin(), logits.end());
            double sum = 0.0;
            for (float value : logits) sum += std::exp(static_cast<double>(value - maximum));
            const double normalizer = maximum + std::log(sum);
            for (int32_t token = 0; token < vocab; ++token) {
                const double acoustic = static_cast<double>(logits[static_cast<size_t>(token)]) - normalizer;
                candidates.push_back({impl_->hypotheses[h].log_prob + acoustic, h, token,
                                      static_cast<float>(std::exp(acoustic))});
            }
        }
        const size_t keep = std::min(candidates.size(), static_cast<size_t>(impl_->max_active_paths));
        std::partial_sort(candidates.begin(), candidates.begin() + static_cast<std::ptrdiff_t>(keep), candidates.end(),
                          [](const Candidate & lhs, const Candidate & rhs) { return lhs.score > rhs.score; });
        std::vector<Hypothesis> next;
        for (size_t i = 0; i < keep; ++i) {
            const auto & candidate = candidates[i];
            Hypothesis hypothesis = impl_->hypotheses[candidate.hypothesis];
            if (candidate.token != blank && candidate.token != unknown) {
                hypothesis.decoder.clear();
                hypothesis.tokens.push_back(candidate.token);
                hypothesis.timestamps.push_back(impl_->frame_offset + frame);
                hypothesis.probabilities.push_back(candidate.probability);
                hypothesis.trailing_blanks = 0;
                const auto [context_score, state] = impl_->graph->forward(hypothesis.state, candidate.token);
                hypothesis.state = state;
                hypothesis.log_prob = candidate.score + context_score;
                hypothesis.context[0] = hypothesis.context[1];
                hypothesis.context[1] = candidate.token;
                if (state->token == -1) {
                    hypothesis.tokens = {-1, blank};
                    hypothesis.context = {-1, blank};
                    hypothesis.timestamps.clear();
                    hypothesis.probabilities.clear();
                }
            } else {
                ++hypothesis.trailing_blanks;
                hypothesis.log_prob = candidate.score;
            }
            auto existing = std::find_if(next.begin(), next.end(),
                                         [&](const Hypothesis & value) { return same_tokens(value, hypothesis); });
            if (existing == next.end()) next.push_back(std::move(hypothesis));
            else existing->log_prob = log_add(existing->log_prob, hypothesis.log_prob);
        }
        impl_->hypotheses = std::move(next);
        const auto best = std::max_element(impl_->hypotheses.begin(), impl_->hypotheses.end(),
                                           [](const Hypothesis & a, const Hypothesis & b) { return a.log_prob < b.log_prob; });
        ContextState * matched = impl_->graph->matched(best->state);
        if (matched != nullptr && static_cast<int>(best->probabilities.size()) >= matched->level) {
            const float confidence = std::accumulate(best->probabilities.begin(),
                                                      best->probabilities.begin() + matched->level, 0.0F) /
                                     static_cast<float>(matched->level);
            if (best->trailing_blanks > impl_->trailing_blanks && confidence >= matched->threshold) {
                detections.push_back({matched->phrase,
                                      *(best->timestamps.end() - matched->level),
                                      best->timestamps.back(), confidence});
                impl_->initialize_hypotheses();
            }
        }
    }
    impl_->frame_offset += frames;
    return detections;
}

}  // namespace engine::models::sherpa_kws
