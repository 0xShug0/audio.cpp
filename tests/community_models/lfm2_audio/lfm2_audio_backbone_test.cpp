#include "engine/community_models/lfm2_audio/backbone.h"
#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/runtime/errors.h"
#include "lfm2_audio_test_package.h"
#include "test_assert.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

namespace {

namespace lfm2 = engine::community_models::lfm2_audio;
using engine::test::require;
using engine::test::require_eq;
using lfm2_audio_test::BackboneShape;
using lfm2_audio_test::Tensor;
using lfm2_audio_test::TensorMap;
using lfm2_audio_test::require_throws_with;

constexpr int64_t kVocab = 40;
using Vector = std::vector<double>;

// LFM2 as transformers' modeling_lfm2 defines it, one position at a time in
// double precision. Tokens are recomputed from scratch at every step, so it
// has no cache for the runtime's prefill and decode paths to share bugs with.
class ReferenceLfm2 {
public:
    ReferenceLfm2(BackboneShape shape, const TensorMap & weights) : shape_(std::move(shape)), weights_(weights) {}

    [[nodiscard]] std::vector<Vector> embed(const lfm2::Lfm2Prompt & prompt, const lfm2::Lfm2AudioEmbeddings & audio) const {
        const auto & table = weight("token_embd.weight");
        std::vector<Vector> out;
        for (const int32_t id : prompt.input_ids) {
            out.push_back(row(table, id));
        }

        for (size_t i = 0; i < prompt.audio_positions.size(); ++i) {
            auto & slot = out[static_cast<size_t>(prompt.audio_positions[i])];
            for (int64_t c = 0; c < shape_.hidden; ++c) {
                slot[static_cast<size_t>(c)] = audio.values[i * static_cast<size_t>(shape_.hidden) + static_cast<size_t>(c)];
            }
        }

        return out;
    }

    [[nodiscard]] Vector logits(std::vector<Vector> xs) const {
        for (size_t layer = 0; layer < shape_.kv_heads.size(); ++layer) {
            const std::string p = "blk." + std::to_string(layer) + ".";
            if (shape_.kv_heads[layer] > 0) {
                attention(xs, p, shape_.kv_heads[layer]);
            } else {
                short_conv(xs, p);
            }

            for (auto & x : xs) {
                const auto n = rms_norm(x, weight(p + "ffn_norm.weight"));
                const auto gate = matvec(weight(p + "ffn_gate.weight"), n);
                const auto up = matvec(weight(p + "ffn_up.weight"), n);
                Vector h(gate.size());
                for (size_t i = 0; i < h.size(); ++i) {
                    h[i] = gate[i] / (1.0 + std::exp(-gate[i])) * up[i];
                }

                add(x, matvec(weight(p + "ffn_down.weight"), h));
            }
        }

        return matvec(weight("token_embd.weight"), rms_norm(xs.back(), weight("token_embd_norm.weight")));
    }

    // Greedy decoding with a check that no step is a near tie, where float
    // rounding alone could pick a different token.
    [[nodiscard]] std::vector<int32_t> greedy(
        const lfm2::Lfm2Prompt & prompt,
        const lfm2::Lfm2AudioEmbeddings & audio,
        int64_t max_new_tokens,
        const std::vector<int32_t> & stop_token_ids = {}) const {
        auto xs = embed(prompt, audio);
        std::vector<int32_t> out;

        for (int64_t step = 0; step < max_new_tokens; ++step) {
            const auto values = logits(xs);
            const auto token = argmax_with_margin(values);
            if (std::find(stop_token_ids.begin(), stop_token_ids.end(), token) != stop_token_ids.end()) {
                break;
            }

            out.push_back(token);
            xs.push_back(row(weight("token_embd.weight"), token));
        }

        return out;
    }

private:
    const Tensor & weight(const std::string & name) const { return weights_.at(name); }

    static Vector row(const Tensor & table, int64_t index) {
        const int64_t width = table.shape.back();
        return Vector(table.values.begin() + index * width, table.values.begin() + (index + 1) * width);
    }

    static Vector matvec(const Tensor & w, const Vector & x) {
        Vector out(static_cast<size_t>(w.shape[0]), 0.0);
        for (int64_t o = 0; o < w.shape[0]; ++o) {
            for (int64_t i = 0; i < w.shape[1]; ++i) {
                out[static_cast<size_t>(o)] += static_cast<double>(w.at(o, i)) * x[static_cast<size_t>(i)];
            }
        }

        return out;
    }

    static void add(Vector & x, const Vector & y) {
        for (size_t i = 0; i < x.size(); ++i) {
            x[i] += y[i];
        }
    }

    Vector rms_norm(const Vector & x, const Tensor & w, size_t offset = 0, size_t size = 0) const {
        size = size == 0 ? x.size() : size;
        double mean_square = 0.0;
        for (size_t i = 0; i < size; ++i) {
            mean_square += x[offset + i] * x[offset + i];
        }

        const double scale = 1.0 / std::sqrt(mean_square / static_cast<double>(size) + shape_.rms_eps);
        Vector out(size);
        for (size_t i = 0; i < size; ++i) {
            out[i] = x[offset + i] * scale * w.values[i];
        }

        return out;
    }

    // Lfm2ShortConv: in_proj -> B, C, x; causal depthwise conv over B * x;
    // gated by C; out_proj.
    void short_conv(std::vector<Vector> & xs, const std::string & p) const {
        const auto d = static_cast<size_t>(shape_.hidden);
        const int64_t k = shape_.kernel;
        std::vector<Vector> bx(xs.size(), Vector(d));
        std::vector<Vector> gate(xs.size(), Vector(d));
        for (size_t t = 0; t < xs.size(); ++t) {
            const auto bcx = matvec(weight(p + "shortconv.in_proj.weight"), rms_norm(xs[t], weight(p + "attn_norm.weight")));
            for (size_t c = 0; c < d; ++c) {
                bx[t][c] = bcx[c] * bcx[2 * d + c];
                gate[t][c] = bcx[d + c];
            }
        }

        const auto & kernel = weight(p + "shortconv.conv.weight");
        for (size_t t = 0; t < xs.size(); ++t) {
            Vector y(d, 0.0);
            for (size_t c = 0; c < d; ++c) {
                for (int64_t j = 0; j < k; ++j) {
                    const int64_t source = static_cast<int64_t>(t) - (k - 1) + j;
                    if (source >= 0) {
                        y[c] += kernel.at(static_cast<int64_t>(c), j) * bx[static_cast<size_t>(source)][c];
                    }
                }

                y[c] *= gate[t][c];
            }

            add(xs[t], matvec(weight(p + "shortconv.out_proj.weight"), y));
        }
    }

    // Lfm2Attention: per-head RMSNorm on q and k, NEOX RoPE, GQA, causal.
    void attention(std::vector<Vector> & xs, const std::string & p, int64_t kv_heads) const {
        const int64_t hd = shape_.head_dim();
        const int64_t heads = shape_.heads;
        std::vector<Vector> q(xs.size());
        std::vector<Vector> k(xs.size());
        std::vector<Vector> v(xs.size());
        for (size_t t = 0; t < xs.size(); ++t) {
            const auto n = rms_norm(xs[t], weight(p + "attn_norm.weight"));
            q[t] = head_norm_rope(matvec(weight(p + "attn_q.weight"), n), weight(p + "attn_q_norm.weight"), heads, t);
            k[t] = head_norm_rope(matvec(weight(p + "attn_k.weight"), n), weight(p + "attn_k_norm.weight"), kv_heads, t);
            v[t] = matvec(weight(p + "attn_v.weight"), n);
        }

        for (size_t t = 0; t < xs.size(); ++t) {
            Vector heads_out(static_cast<size_t>(heads * hd), 0.0);
            for (int64_t h = 0; h < heads; ++h) {
                const int64_t g = h / (heads / kv_heads);
                std::vector<double> scores(t + 1);
                double max_score = -std::numeric_limits<double>::infinity();
                for (size_t s = 0; s <= t; ++s) {
                    double dot = 0.0;
                    for (int64_t i = 0; i < hd; ++i) {
                        dot += q[t][static_cast<size_t>(h * hd + i)] * k[s][static_cast<size_t>(g * hd + i)];
                    }

                    scores[s] = dot / std::sqrt(static_cast<double>(hd));
                    max_score = std::max(max_score, scores[s]);
                }

                double total = 0.0;
                for (auto & score : scores) {
                    score = std::exp(score - max_score);
                    total += score;
                }

                for (size_t s = 0; s <= t; ++s) {
                    for (int64_t i = 0; i < hd; ++i) {
                        heads_out[static_cast<size_t>(h * hd + i)] += scores[s] / total * v[s][static_cast<size_t>(g * hd + i)];
                    }
                }
            }

            add(xs[t], matvec(weight(p + "attn_output.weight"), heads_out));
        }
    }

    Vector head_norm_rope(const Vector & x, const Tensor & norm, int64_t count, size_t position) const {
        const int64_t hd = shape_.head_dim();
        Vector out(x.size());
        for (int64_t h = 0; h < count; ++h) {
            const auto normed = rms_norm(x, norm, static_cast<size_t>(h * hd), static_cast<size_t>(hd));
            for (int64_t i = 0; i < hd / 2; ++i) {
                const double angle = static_cast<double>(position) *
                    std::pow(static_cast<double>(shape_.rope_theta), -2.0 * static_cast<double>(i) / static_cast<double>(hd));
                const double x0 = normed[static_cast<size_t>(i)];
                const double x1 = normed[static_cast<size_t>(i + hd / 2)];
                out[static_cast<size_t>(h * hd + i)] = x0 * std::cos(angle) - x1 * std::sin(angle);
                out[static_cast<size_t>(h * hd + i + hd / 2)] = x0 * std::sin(angle) + x1 * std::cos(angle);
            }
        }

        return out;
    }

    static int32_t argmax_with_margin(const Vector & values) {
        std::vector<size_t> order(values.size());
        for (size_t i = 0; i < order.size(); ++i) {
            order[i] = i;
        }

        std::partial_sort(order.begin(), order.begin() + 2, order.end(), [&](size_t a, size_t b) { return values[a] > values[b]; });
        require(values[order[0]] - values[order[1]] > 1e-3,
            "the reference has a near tie between tokens " + std::to_string(order[0]) + " and " +
                std::to_string(order[1]) + "; pick another seed");
        return static_cast<int32_t>(order[0]);
    }

    BackboneShape shape_;
    const TensorMap & weights_;
};

struct Fixture {
    BackboneShape shape;
    TensorMap weights;
    std::filesystem::path path;
    lfm2::Lfm2BackboneConfig config;
    engine::core::ExecutionContext execution{engine::core::BackendConfig{engine::core::BackendType::Cpu, 0, 2}};

    Fixture() : path(lfm2_audio_test::fresh_directory("audiocpp_lfm2_audio_backbone_test") / "backbone.gguf") {
        weights = lfm2_audio_test::backbone_tensors(shape, kVocab, lfm2_audio_test::random_fill(7));
        lfm2_audio_test::TextVocab vocab;
        for (int64_t i = 0; i < kVocab; ++i) {
            vocab.tokens.push_back("t" + std::to_string(i));
            vocab.types.push_back(1);
        }

        vocab.merges = {"t 1"};
        lfm2_audio_test::write_backbone(path, shape, vocab, weights);

        config.vocab_size = kVocab;
        config.hidden_size = shape.hidden;
        config.intermediate_size = shape.intermediate;
        config.num_attention_heads = shape.heads;
        config.head_dim = shape.head_dim();
        config.conv_kernel_size = shape.kernel;
        config.context_length = shape.context;
        config.kv_heads.assign(shape.kv_heads.begin(), shape.kv_heads.end());
        config.rms_norm_eps = shape.rms_eps;
        config.rope_theta = shape.rope_theta;
    }

    ~Fixture() { std::filesystem::remove_all(path.parent_path()); }

    [[nodiscard]] std::unique_ptr<lfm2::Lfm2BackboneRuntime> runtime() {
        return std::make_unique<lfm2::Lfm2BackboneRuntime>(engine::assets::open_tensor_source(path), config, execution);
    }

    [[nodiscard]] ReferenceLfm2 reference() const { return ReferenceLfm2(shape, weights); }
};

lfm2::Lfm2Prompt text_prompt(int64_t length, uint64_t seed) {
    lfm2_audio_test::Random random(seed);
    lfm2::Lfm2Prompt prompt;
    for (int64_t i = 0; i < length; ++i) {
        prompt.input_ids.push_back(static_cast<int32_t>((random.uniform(1.0f) + 1.0f) * 0.5f * (kVocab - 1)));
    }

    return prompt;
}

struct AudioPrompt {
    lfm2::Lfm2Prompt prompt;
    lfm2::Lfm2AudioEmbeddings audio;
};

// A prompt with `audio_tokens` audio rows starting at position 2, like the
// session's system and user turns around the audio.
AudioPrompt audio_prompt(int64_t length, int64_t audio_tokens, uint64_t seed) {
    AudioPrompt out{text_prompt(length, seed), {}};
    out.audio.tokens = audio_tokens;
    out.audio.hidden_size = 16;
    out.audio.values = lfm2_audio_test::Random(seed + 1).uniform(static_cast<size_t>(audio_tokens * 16), 1.0f);
    for (int64_t i = 0; i < audio_tokens; ++i) {
        out.prompt.audio_positions.push_back(static_cast<int32_t>(2 + i));
        out.prompt.input_ids[static_cast<size_t>(2 + i)] = 0;
    }

    return out;
}

std::string ids(const std::vector<int32_t> & values) {
    std::ostringstream out;
    for (size_t i = 0; i < values.size(); ++i) {
        out << (i == 0 ? "" : ",") << values[i];
    }

    return out.str();
}

void require_logits_close(const std::vector<float> & actual, const Vector & expected, const std::string & label) {
    require_eq(actual.size(), expected.size(), label + " logits size");
    double scale = 1.0;
    for (const double value : expected) {
        scale = std::max(scale, std::fabs(value));
    }

    for (size_t i = 0; i < actual.size(); ++i) {
        const double diff = std::fabs(static_cast<double>(actual[i]) - expected[i]);
        require(diff <= 1e-4 * scale,
            label + " logit " + std::to_string(i) + ": expected " + std::to_string(expected[i]) + ", got " +
                std::to_string(actual[i]));
    }
}

void test_prefill_matches_reference(Fixture & fixture) {
    auto runtime = fixture.runtime();
    const auto reference = fixture.reference();
    const lfm2::Lfm2AudioEmbeddings no_audio;
    const auto prompt = text_prompt(11, 1);
    const auto result = runtime->generate(prompt, no_audio, {1, {}});
    require_logits_close(result.prefill_logits, reference.logits(reference.embed(prompt, no_audio)), "text prefill");

    const auto with_audio = audio_prompt(12, 5, 2);
    const auto audio_result = runtime->generate(with_audio.prompt, with_audio.audio, {1, {}});
    require_logits_close(
        audio_result.prefill_logits, reference.logits(reference.embed(with_audio.prompt, with_audio.audio)), "audio prefill");
}

// Decode steps go through the conv tails and the KV cache instead of the
// whole sequence, so they must pick the same tokens as recomputing it.
void test_decode_matches_reference(Fixture & fixture) {
    auto runtime = fixture.runtime();
    const auto reference = fixture.reference();
    const auto c = audio_prompt(12, 5, 3);
    const auto result = runtime->generate(c.prompt, c.audio, {20, {}});
    require_eq(ids(result.tokens), ids(reference.greedy(c.prompt, c.audio, 20)), "greedy tokens");
}

void test_stops(Fixture & fixture) {
    auto runtime = fixture.runtime();
    const auto reference = fixture.reference();
    const auto c = audio_prompt(10, 3, 4);
    const auto expected = reference.greedy(c.prompt, c.audio, 12);

    // Stop at the first token that has not appeared before it.
    size_t stop_at = 1;
    while (stop_at < expected.size() &&
           std::find(expected.begin(), expected.begin() + static_cast<std::ptrdiff_t>(stop_at), expected[stop_at]) !=
               expected.begin() + static_cast<std::ptrdiff_t>(stop_at)) {
        ++stop_at;
    }

    require(stop_at < expected.size(), "the greedy sequence repeats one token; pick another seed");
    const auto stopped = runtime->generate(c.prompt, c.audio, {12, {expected[stop_at]}});
    require_eq(ids(stopped.tokens), ids({expected.begin(), expected.begin() + static_cast<std::ptrdiff_t>(stop_at)}), "stop token");
    require(stopped.stopped, "a stop token must be reported");

    const auto budget = runtime->generate(c.prompt, c.audio, {4, {}});
    require_eq(ids(budget.tokens), ids({expected.begin(), expected.begin() + 4}), "token budget");
    require(!budget.stopped, "running out of tokens must be reported");

    // A stop token as the very first choice is an empty, finished result.
    const auto immediate = runtime->generate(c.prompt, c.audio, {12, {expected[0]}});
    require(immediate.tokens.empty() && immediate.stopped, "an immediate stop token");
}

// The runtime keeps its prefill and decode graphs between requests; nothing
// of one request may leak into the next.
void test_requests_are_independent(Fixture & fixture) {
    auto runtime = fixture.runtime();
    const auto reference = fixture.reference();
    const lfm2::Lfm2AudioEmbeddings no_audio;
    const auto long_case = audio_prompt(16, 6, 5);
    const auto short_case = audio_prompt(9, 2, 6);
    const auto same_length = text_prompt(16, 7);
    const auto long_expected = ids(reference.greedy(long_case.prompt, long_case.audio, 24));
    const auto short_expected = ids(reference.greedy(short_case.prompt, short_case.audio, 6));
    const auto same_length_expected = ids(reference.greedy(same_length, no_audio, 10));
    const auto run = [&](const lfm2::Lfm2Prompt & prompt, const lfm2::Lfm2AudioEmbeddings & audio, int64_t tokens) {
        return ids(runtime->generate(prompt, audio, {tokens, {}}).tokens);
    };

    require_eq(run(short_case.prompt, short_case.audio, 6), short_expected, "short first");
    require_eq(run(long_case.prompt, long_case.audio, 24), long_expected, "long after short");
    require_eq(run(short_case.prompt, short_case.audio, 6), short_expected, "short after long");
    require_eq(run(same_length, no_audio, 10), same_length_expected, "text after audio");
    require_eq(run(long_case.prompt, long_case.audio, 24), long_expected, "long again");
}

void test_rejects_bad_requests(Fixture & fixture) {
    auto runtime = fixture.runtime();
    const lfm2::Lfm2AudioEmbeddings no_audio;
    const auto prompt = text_prompt(8, 8);
    const auto audio_case = audio_prompt(10, 3, 9);
    const auto & with_audio = audio_case.prompt;
    const auto & audio = audio_case.audio;

    require_throws_with([&] { (void)runtime->generate({}, no_audio, {4, {}}); }, "needs a prompt", "an empty prompt");
    require_throws_with([&] { (void)runtime->generate(prompt, no_audio, {0, {}}); }, "positive token budget", "a zero budget");
    require_throws_with(
        [&] { (void)runtime->generate(prompt, no_audio, {fixture.shape.context - 7, {}}); }, "-token context",
        "a request past the context");

    // Too long a request is the caller's to fix, so it is a CapacityError
    // (a client error in the server); the check must not overflow either.
    bool capacity_error = false;
    try {
        (void)runtime->generate(prompt, no_audio, {std::numeric_limits<int64_t>::max(), {}});
    } catch (const engine::runtime::CapacityError &) {
        capacity_error = true;
    }

    require(capacity_error, "an INT64_MAX token budget must be a CapacityError");
    require_throws_with([&] { (void)runtime->generate(with_audio, no_audio, {4, {}}); }, "audio embeddings",
        "missing audio embeddings");

    auto short_values = audio;
    short_values.values.pop_back();
    require_throws_with([&] { (void)runtime->generate(with_audio, short_values, {4, {}}); }, "audio embeddings",
        "truncated audio embeddings");

    auto bad_id = prompt;
    bad_id.input_ids[3] = static_cast<int32_t>(kVocab);
    require_throws_with([&] { (void)runtime->generate(bad_id, no_audio, {4, {}}); }, "outside the vocabulary",
        "a token id past the vocabulary");
    bad_id.input_ids[3] = -1;
    require_throws_with([&] { (void)runtime->generate(bad_id, no_audio, {4, {}}); }, "outside the vocabulary",
        "a negative token id");

    auto outside = with_audio;
    outside.audio_positions.back() = static_cast<int32_t>(outside.input_ids.size());
    require_throws_with([&] { (void)runtime->generate(outside, audio, {4, {}}); }, "audio positions",
        "an audio position past the prompt");
    auto repeated = with_audio;
    repeated.audio_positions[1] = repeated.audio_positions[0];
    require_throws_with([&] { (void)runtime->generate(repeated, audio, {4, {}}); }, "audio positions",
        "a repeated audio position");

    // A broken encoder shows up as NaN embeddings, which would reach the
    // logits, where argmax picks token 0 and decodes it to nothing.
    auto nan_audio = audio;
    nan_audio.values[0] = std::numeric_limits<float>::quiet_NaN();
    require_throws_with([&] { (void)runtime->generate(with_audio, nan_audio, {4, {}}); }, "non-finite audio embeddings",
        "NaN audio embeddings");

    // The runtime still works after rejected requests.
    const auto reference = fixture.reference();
    require_eq(ids(runtime->generate(with_audio, audio, {6, {}}).tokens), ids(reference.greedy(with_audio, audio, 6)),
        "a request after the rejected ones");
}

}  // namespace

int main() {
    try {
        Fixture fixture;
        test_prefill_matches_reference(fixture);
        test_decode_matches_reference(fixture);
        test_stops(fixture);
        test_requests_are_independent(fixture);
        test_rejects_bad_requests(fixture);
        std::cout << "lfm2_audio_backbone_test: PASS\n";
        return 0;
    } catch (const std::exception & error) {
        std::cerr << "lfm2_audio_backbone_test: " << error.what() << '\n';
        return 1;
    }
}
