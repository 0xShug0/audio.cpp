#include "engine/framework/debug/profiler.h"
// granite_speech5 C2 unit checks: decoder math, binding, generation, cache.
//
// Always-run (no checkpoint): budget/spec/binding constants, shared-module
// neutral defaults + single application (synthetic weights), block vs
// single vs full-recompute equivalence (synthetic), RoPE YaRN Q/K at
// 0/1/4095/4096/4097/16383 vs reference-derived divisors, A/B/A cache reset
// with an interleaved failed request (synthetic), budget-edge validation.
//
// Checkpoint-gated (GRANITE_SPEECH5_MODEL,
// plus GRANITE_SPEECH5_TRACES references; reports skipped checks otherwise):
// short-FP32 prefill/forced-step logit gates, embed-scale fixture, top-1
// margin rule, exact greedy sequences en/ja, text-only 4096 crossing, greedy
// wrapper parity, real A/B/A on text prompts.

#include "engine/community_models/granite_speech5/assets.h"
#include "decoder_parity.h"
#include "engine/community_models/granite_speech5/encoder.h"
#include "engine/community_models/granite_speech5/frontend.h"
#include "engine/community_models/granite_speech5/projector.h"
#include "engine/community_models/granite_speech5/prompt.h"
#include "engine/framework/audio/wav_reader.h"
#include "engine/framework/core/execution_context.h"
#include "engine/framework/modules/positional_modules.h"
#include "engine/framework/modules/transformers/causal_decoder_runtime.h"
#include "test_assert.h"

#include <ggml-backend.h>
#include <ggml.h>

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

namespace test = engine::test;
namespace g2b = engine::community_models::granite_speech5;
namespace modules = engine::modules;
namespace core = engine::core;

// Parity gates from reference.py PIPELINE (logits rows): last_pos_logits and
// forced_step{k} carry max_abs 2e-2 + rel_rms 1e-3; merged/scaled prefill
// carry rel_rms 1e-3 with no max_abs gate.
constexpr float kLogitsMaxAbs = 2e-2F;
constexpr float kLogitsRelRms = 1e-3F;
constexpr float kPrefillRelRms = 1e-3F;
constexpr float kTop1MarginGate = 2.0F * kLogitsMaxAbs;  // match where ref margin exceeds this

#ifdef AUDIOCPP_SOURCE_DIR
std::filesystem::path source_dir() {
    return std::filesystem::path(AUDIOCPP_SOURCE_DIR);
}
#else
std::filesystem::path source_dir() {
    return std::filesystem::current_path();
}
#endif

std::optional<std::filesystem::path> find_model_dir() {
    if (const char * env = std::getenv("GRANITE_SPEECH5_MODEL")) {
        const std::filesystem::path candidate(env);
        if (std::filesystem::exists(candidate / "model.safetensors")) {
            return candidate;
        }
    }

    return std::nullopt;
}

std::filesystem::path fixtures_dir() {
    const char * directory = std::getenv("GRANITE_SPEECH5_FIXTURES");
    if (directory == nullptr || directory[0] == '\0') {
        throw std::runtime_error("reference parity requires GRANITE_SPEECH5_FIXTURES");
    }
    return directory;
}

// --- tiny file readers ---

std::vector<float> read_f32_file(const std::filesystem::path & path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("C2 test cannot open file: " + path.string());
    }
    in.seekg(0, std::ios::end);
    const auto bytes = in.tellg();
    if (bytes < 0 || bytes % sizeof(float) != 0) {
        throw std::runtime_error("C2 test bad f32 size: " + path.string());
    }
    in.seekg(0, std::ios::beg);
    std::vector<float> values(static_cast<size_t>(bytes) / sizeof(float));
    in.read(reinterpret_cast<char *>(values.data()), bytes);
    return values;
}

std::string read_text_file(const std::filesystem::path & path) {
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("C2 test cannot open file: " + path.string());
    }
    return std::string((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
}

// Parse a flat JSON number list: "[1, 2.5, ...]".
std::vector<double> parse_number_list(const std::string & text) {
    std::vector<double> out;
    std::string token;
    bool in_list = false;
    for (char ch : text) {
        if (ch == '[') {
            if (in_list) {
                throw std::runtime_error("C2 test nested list");
            }
            in_list = true;
        } else if (ch == ']') {
            if (!in_list) {
                throw std::runtime_error("C2 test unmatched ']'");
            }
            in_list = false;
            if (!token.empty()) {
                out.push_back(std::stod(token));
                token.clear();
            }
        } else if (ch == ',') {
            if (!token.empty()) {
                out.push_back(std::stod(token));
                token.clear();
            }
        } else if (ch == '-' || ch == '+' || ch == '.' || ch == 'e' || ch == 'E' ||
                   (ch >= '0' && ch <= '9')) {
            token += ch;
        } else if (ch == ' ' || ch == '\t' || ch == '\n' || ch == '\r') {
            if (!token.empty()) {
                out.push_back(std::stod(token));
                token.clear();
            }
        }
    }
    if (in_list || !token.empty()) {
        throw std::runtime_error("C2 test truncated number list");
    }
    return out;
}

// Extract a top-level JSON scalar number by key: `"key": 1.5`.
double json_scalar(const std::string & text, const std::string & key) {
    const std::string needle = "\"" + key + "\"";
    const size_t at = text.find(needle);
    if (at == std::string::npos) {
        throw std::runtime_error("C2 test json has no key: " + key);
    }
    const size_t colon = text.find(':', at + needle.size());
    if (colon == std::string::npos) {
        throw std::runtime_error("C2 test json key has no value: " + key);
    }
    size_t begin = text.find_first_not_of(" \t\n\r", colon + 1);
    if (begin == std::string::npos) {
        throw std::runtime_error("C2 test json truncated value: " + key);
    }
    const size_t end = text.find_first_not_of("-+.eE0123456789", begin);
    return std::stod(text.substr(begin, end - begin));
}
std::vector<double> json_numbers(const std::string & text, const std::string & key) {
    const std::string needle = "\"" + key + "\"";
    const size_t at = text.find(needle);
    if (at == std::string::npos) {
        throw std::runtime_error("C2 test trace.json has no key: " + key);
    }
    const size_t open = text.find('[', at);
    if (open == std::string::npos) {
        throw std::runtime_error("C2 test trace.json key has no array: " + key);
    }
    int depth = 0;
    size_t close = open;
    for (; close < text.size(); ++close) {
        if (text[close] == '[') {
            ++depth;
        } else if (text[close] == ']') {
            if (--depth == 0) {
                break;
            }
        }
    }
    if (depth != 0) {
        throw std::runtime_error("C2 test trace.json truncated array: " + key);
    }
    return parse_number_list(text.substr(open, close - open + 1));
}

std::string json_string(const std::string & text, const std::string & key) {
    const std::string needle = "\"" + key + "\"";
    const size_t at = text.find(needle);
    if (at == std::string::npos) {
        throw std::runtime_error("C2 test trace.json has no key: " + key);
    }
    const size_t open = text.find('"', at + needle.size());
    if (open == std::string::npos) {
        throw std::runtime_error("C2 test trace.json key has no string: " + key);
    }
    std::string out;
    for (size_t i = open + 1; i < text.size(); ++i) {
        const char ch = text[i];
        if (ch == '"') {
            return out;
        }
        if (ch == '\\') {
            if (++i >= text.size()) {
                break;
            }
            const char esc = text[i];
            if (esc == 'n') {
                out += '\n';
            } else if (esc == 't') {
                out += '\t';
            } else if (esc == 'r') {
                out += '\r';
            } else if (esc == 'u') {
                if (i + 4 >= text.size()) {
                    throw std::runtime_error("C2 test bad \\u escape");
                }
                const unsigned code =
                    static_cast<unsigned>(std::stoul(text.substr(i + 1, 4), nullptr, 16));
                i += 4;
                if (code < 0x80) {
                    out += static_cast<char>(code);
                } else if (code < 0x800) {
                    out += static_cast<char>(0xC0 | (code >> 6));
                    out += static_cast<char>(0x80 | (code & 0x3F));
                } else {
                    out += static_cast<char>(0xE0 | (code >> 12));
                    out += static_cast<char>(0x80 | ((code >> 6) & 0x3F));
                    out += static_cast<char>(0x80 | (code & 0x3F));
                }
            } else {
                out += esc;
            }
        } else {
            out += ch;
        }
    }
    throw std::runtime_error("C2 test trace.json truncated string: " + key);
}

float max_abs_diff(const std::vector<float> & actual, const std::vector<float> & expected) {
    test::require_eq(actual.size(), expected.size(), "vector size");
    float worst = 0.0F;
    for (size_t i = 0; i < actual.size(); ++i) {
        if (!std::isfinite(actual[i]) || !std::isfinite(expected[i])) {
            throw std::runtime_error("C2 test non-finite values");
        }
        worst = std::max(worst, std::fabs(actual[i] - expected[i]));
    }
    return worst;
}

float rel_rms_diff(const std::vector<float> & actual, const std::vector<float> & expected) {
    test::require_eq(actual.size(), expected.size(), "vector size");
    double num = 0.0;
    double den = 0.0;
    for (size_t i = 0; i < actual.size(); ++i) {
        const double d = static_cast<double>(actual[i]) - static_cast<double>(expected[i]);
        num += d * d;
        den += static_cast<double>(expected[i]) * static_cast<double>(expected[i]);
    }
    return static_cast<float>(
        std::sqrt(num / actual.size()) / std::max(std::sqrt(den / actual.size()), 1e-12));
}

void require_logits_close(
    const std::vector<float> & actual,
    const std::vector<float> & expected,
    const std::string & label) {
    const float max_abs = max_abs_diff(actual, expected);
    const float rel_rms = rel_rms_diff(actual, expected);
    std::cout << "  gate [" << label << "]: max_abs=" << max_abs << " rel_rms=" << rel_rms << "\n";
    test::require(max_abs <= kLogitsMaxAbs, label + " max_abs gate");
    test::require(rel_rms <= kLogitsRelRms, label + " rel_rms gate");
}

void require_prefill_close(
    const std::vector<float> & actual,
    const std::vector<float> & expected,
    const std::string & label) {
    const float rel_rms = rel_rms_diff(actual, expected);
    std::cout << "  gate [" << label << "]: rel_rms=" << rel_rms << "\n";
    test::require(rel_rms <= kPrefillRelRms, label + " rel_rms gate");
}

int32_t top1(const std::vector<float> & logits) {
    int32_t best = 0;
    for (size_t i = 1; i < logits.size(); ++i) {
        if (logits[i] > logits[best]) {
            best = static_cast<int32_t>(i);
        }
    }
    return best;
}

float top_margin(const std::vector<float> & logits) {
    int32_t best = 0;
    int32_t second = -1;
    for (size_t i = 1; i < logits.size(); ++i) {
        if (logits[i] > logits[best]) {
            second = best;
            best = static_cast<int32_t>(i);
        } else if (second < 0 || logits[i] > logits[second]) {
            second = static_cast<int32_t>(i);
        }
    }
    return logits[best] - logits[second];
}

// --- always-run: budget + binding constants ---

void test_budget_rules() {
    bool threw = false;
    try {
        g2b::granite_speech5_require_generation_budget(0, 8);
    } catch (const std::runtime_error &) {
        threw = true;
    }
    test::require(threw, "empty prompt rejected");
    threw = false;
    try {
        g2b::granite_speech5_require_generation_budget(50, 0);
    } catch (const std::runtime_error &) {
        threw = true;
    }
    test::require(threw, "non-positive budget rejected");
    threw = false;
    try {
        g2b::granite_speech5_require_generation_budget(16380, 5);
    } catch (const std::runtime_error &) {
        threw = true;
    }
    test::require(threw, "over-cap request rejected");
    // Exact cap fits; small requests fit.
    g2b::granite_speech5_require_generation_budget(16380, 4);
    g2b::granite_speech5_require_generation_budget(51, 8);
}

void test_binding_values() {
    // C2 table verbatim: every constant the decoder binds.
    const auto spec = g2b::make_granite_speech5_greedy_decoder_spec();
    const auto & stack = spec.decoder.stack;
    test::require_eq(stack.hidden_size, int64_t{2048}, "hidden");
    test::require_eq(stack.num_attention_heads, int64_t{16}, "heads");
    test::require_eq(stack.num_key_value_heads, int64_t{4}, "kv heads");
    test::require_eq(stack.head_dim, int64_t{128}, "head dim");
    test::require_eq(stack.intermediate_size, int64_t{4096}, "intermediate");
    test::require_eq(stack.layers, g2b::kGraniteSpeech5DecoderLayers, "layers");
    test::require_close(stack.rms_norm_eps, 1e-5F, 0.0F, "rms eps");
    test::require_close(stack.rope_theta, 10000.0F, 0.0F, "rope theta");
    test::require(!stack.use_qk_norm, "no q/k norm");
    test::require(!spec.attention_bias, "bias-free");
    test::require(!spec.packed_qkv, "separate qkv");
    test::require_close(*stack.attention_scale, 0.0078125F, 0.0F, "attn scale");
    test::require_close(stack.residual_scale, 0.22F, 0.0F, "residual");
    test::require(stack.rope.has_value(), "composed rotary configuration");
    test::require_close(stack.rope->freq_scale, 0.25F, 0.0F, "yarn freq scale");
    test::require_close(stack.rope->ext_factor, 1.0F, 0.0F, "yarn ext");
    test::require_close(stack.rope->attn_factor, 1.0F, 0.0F, "yarn attn passthrough");
    test::require_close(stack.rope->beta_fast, 32.0F, 0.0F, "beta fast");
    test::require_close(stack.rope->beta_slow, 1.0F, 0.0F, "beta slow");
    test::require_eq(stack.rope->orig_context, int32_t{4096}, "yarn orig");
    test::require_close(spec.decoder.embedding_scale, 12.0F, 0.0F, "embed");
    test::require_close(spec.decoder.logit_divisor, 8.0F, 0.0F, "logits");
    test::require_eq(spec.decoder.logits_size, int64_t{100353}, "logits size");
    test::require_eq(spec.vocab_size, int64_t{100353}, "vocab");
    test::require_eq(spec.max_position_embeddings, int64_t{16384}, "limit");
    test::require(!spec.tie_word_embeddings, "untied head");
    test::require_eq(
        spec.token_embedding_tensor,
        std::string("language_model.model.embed_tokens.weight"),
        "embed tensor");
    test::require_eq(
        spec.lm_head_tensor, std::string("language_model.lm_head.weight"), "head tensor");
    test::require_eq(
        spec.final_norm_tensor, std::string("language_model.model.norm.weight"), "norm tensor");
    test::require_eq(spec.layer_prefix, std::string("language_model.model.layers"), "layer prefix");
    test::require(
        spec.eos_token_ids == std::vector<int64_t>{100257}, "eos");
    test::require(
        stack.runtime.attention.prefill_mode == modules::DecoderAttentionMode::FlashGrouped,
        "framework grouped-flash prefill");
    test::require(
        stack.runtime.attention.static_mode == modules::DecoderAttentionMode::FlashGrouped,
        "framework grouped-flash decode");
    test::require(
        stack.runtime.static_cache.update_mode ==
            modules::DecoderStaticCacheUpdateMode::DirectSetRows,
        "direct set rows");

    const auto causal = g2b::make_granite_speech5_causal_decoder_config();
    test::require_eq(causal.stack.hidden_size, int64_t{2048}, "causal hidden");
    test::require_close(*causal.stack.attention_scale, 0.0078125F, 0.0F, "causal attn");
    test::require_close(causal.embedding_scale, 12.0F, 0.0F, "causal embed");
    test::require_close(causal.logit_divisor, 8.0F, 0.0F, "causal divisor");
}

// --- always-run: synthetic tiny-model harness ---

std::vector<float> pattern_values(size_t count, float scale, float shift) {
    std::vector<float> out(count);
    for (size_t i = 0; i < count; ++i) {
        out[i] = scale * std::sin(static_cast<float>(i) * 0.17F + 0.3F) + shift;
    }
    return out;
}

struct SyntheticHarness {
    core::ExecutionContext execution{
        core::BackendConfig{core::BackendType::Cpu, 0, 8}};
    ggml_context * context = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    std::vector<core::TensorValue> owned;
    modules::CausalDecoderRuntimeWeights weights;

    int64_t hidden = 32;
    int64_t heads = 4;
    int64_t kv_heads = 2;
    int64_t head_dim = 8;
    int64_t intermediate = 64;
    int64_t layers = 2;
    int64_t vocab = 97;

    SyntheticHarness() {
        context = ggml_init({32 * 1024 * 1024, nullptr, true});
        if (!context) {
            throw std::runtime_error("C2 test synthetic context failed");
        }
        core::ModuleBuildContext build{context, "granite_c2_synthetic", core::BackendType::Cpu};
        auto tensor = [&](std::initializer_list<int64_t> shape) {
            auto value = core::make_tensor(
                build, GGML_TYPE_F32, core::TensorShape::from_dims(shape));
            owned.push_back(value);
            return value;
        };
        weights.token_embedding = tensor({vocab, hidden});
        weights.final_norm = {tensor({hidden}), std::nullopt};
        weights.lm_head = {{tensor({vocab, hidden}), std::nullopt}};
        for (int64_t layer = 0; layer < layers; ++layer) {
            modules::DecoderLayerWeights w;
            w.input_norm = {tensor({hidden}), std::nullopt};
            w.post_norm = {tensor({hidden}), std::nullopt};
            w.self_attention.q_weight = tensor({heads * head_dim, hidden});
            w.self_attention.k_weight = tensor({kv_heads * head_dim, hidden});
            w.self_attention.v_weight = tensor({kv_heads * head_dim, hidden});
            w.self_attention.out_weight = tensor({hidden, heads * head_dim});
            w.mlp.gate_proj = {tensor({intermediate, hidden}), std::nullopt};
            w.mlp.up_proj = {tensor({intermediate, hidden}), std::nullopt};
            w.mlp.down_proj = {tensor({hidden, intermediate}), std::nullopt};
            weights.stack.layers.push_back(w);
        }
        buffer = ggml_backend_alloc_ctx_tensors(context, execution.backend());
        if (!buffer) {
            throw std::runtime_error("C2 test synthetic weight buffer failed");
        }
        for (const auto & value : owned) {
            auto values = pattern_values(ggml_nelements(value.tensor), 0.04F, 0.0F);
            if (value.shape.rank == 1) {
                std::fill(values.begin(), values.end(), 1.0F);
            }
            core::write_tensor_f32(value, values);
        }
        // Pin the token table to ones so single-token decode steps append a
        // known row (used by the full-recompute equivalence below).
        core::write_tensor_f32(
            weights.token_embedding,
            std::vector<float>(static_cast<size_t>(vocab * hidden), 1.0F));
    }

    ~SyntheticHarness() {
        if (buffer) {
            ggml_backend_buffer_free(buffer);
        }
        if (context) {
            ggml_free(context);
        }
    }

    modules::CausalDecoderRuntimeConfig base_config() {
        modules::CausalDecoderRuntimeConfig config;
        auto & stack = config.decoder.stack;
        stack.hidden_size = hidden;
        stack.num_attention_heads = heads;
        stack.num_key_value_heads = kv_heads;
        stack.head_dim = head_dim;
        stack.intermediate_size = intermediate;
        stack.layers = layers;
        stack.use_qk_norm = false;
        stack.runtime.static_cache.update_mode =
            modules::DecoderStaticCacheUpdateMode::DirectSetRows;
        stack.runtime.attention.prefill_mode = modules::DecoderAttentionMode::ManualRepeat;
        stack.runtime.attention.static_mode = modules::DecoderAttentionMode::ManualRepeat;
        config.decoder.logits_size = vocab;
        config.prefill_graph_arena_bytes = 64 * 1024 * 1024;
        config.decode_graph_arena_bytes = 64 * 1024 * 1024;
        return config;
    }
};

void test_neutral_defaults() {
    // Omitting callers keep current behavior: defaults vs explicit-neutral
    // configs produce bitwise-identical logits on both the direct and block
    // prefill paths.
    SyntheticHarness harness;
    auto defaults = harness.base_config();
    auto explicit_neutral = harness.base_config();
    explicit_neutral.decoder.stack.attention_scale =
        1.0F / std::sqrt(static_cast<float>(harness.head_dim));
    explicit_neutral.decoder.stack.residual_scale = 1.0F;
    explicit_neutral.decoder.stack.rope = modules::RoPEConfig{harness.head_dim, GGML_ROPE_TYPE_NEOX, explicit_neutral.decoder.stack.rope_theta};
    explicit_neutral.decoder.stack.rope->freq_scale = 1.0F;
    explicit_neutral.decoder.stack.rope->ext_factor = 0.0F;
    explicit_neutral.decoder.stack.rope->attn_factor = 1.0F;
    explicit_neutral.decoder.stack.rope->beta_fast = 0.0F;
    explicit_neutral.decoder.stack.rope->beta_slow = 0.0F;
    explicit_neutral.decoder.stack.rope->orig_context = 0;
    explicit_neutral.decoder.embedding_scale = 1.0F;
    explicit_neutral.decoder.logit_divisor = 1.0F;
    modules::CausalDecoderRuntime run_default(harness.execution, defaults, harness.weights);
    modules::CausalDecoderRuntime run_explicit(harness.execution, explicit_neutral, harness.weights);
    const int64_t steps = 24;
    const auto embeddings = pattern_values(steps * harness.hidden, 0.3F, 0.0F);
    const auto direct_default = run_default.prefill_embeddings(embeddings, steps);
    const auto direct_explicit = run_explicit.prefill_embeddings(embeddings, steps);
    test::require(
        direct_default.logits == direct_explicit.logits, "default vs explicit-neutral direct logits");
    const auto block_default =
        run_default.prefill_embeddings_into_cache(embeddings, steps, 32, 8, 0);
    const auto block_explicit =
        run_explicit.prefill_embeddings_into_cache(embeddings, steps, 32, 8, 0);
    test::require(
        block_default.logits == block_explicit.logits, "default vs explicit-neutral block logits");
}

void test_single_application() {
    // Each opt-in scale is applied exactly once at its boundary: scaling the
    // input pre-graph vs the in-graph scale produce identical logits, and
    // non-default values actually change the output (i.e. are wired).
    SyntheticHarness harness;
    auto plain = harness.base_config();
    auto embed_only = harness.base_config();
    embed_only.decoder.embedding_scale = 12.0F;
    embed_only.decoder.logit_divisor = 8.0F;
    auto full_scaled = harness.base_config();
    full_scaled.decoder.embedding_scale = 12.0F;
    full_scaled.decoder.logit_divisor = 8.0F;
    full_scaled.decoder.stack.residual_scale = 0.22F;
    full_scaled.decoder.stack.attention_scale = 0.0078125F;
    modules::CausalDecoderRuntime run_plain(harness.execution, plain, harness.weights);
    modules::CausalDecoderRuntime run_embed(harness.execution, embed_only, harness.weights);
    modules::CausalDecoderRuntime run_full(harness.execution, full_scaled, harness.weights);
    const int64_t steps = 16;
    const auto embeddings = pattern_values(steps * harness.hidden, 0.3F, 0.0F);
    auto scaled_input = embeddings;
    for (float & value : scaled_input) {
        value *= 12.0F;
    }
    // Embed x12 in-graph == x12 pre-graph, then /8 on the logits.
    const auto via_graph =
        run_embed.prefill_embeddings_into_cache(embeddings, steps, 32, 8, 0).logits;
    auto via_host =
        run_plain.prefill_embeddings_into_cache(scaled_input, steps, 32, 8, 0).logits;
    for (float & value : via_host) {
        value /= 8.0F;
    }
    test::require(via_graph == via_host, "embed/divisor single application");
    // Non-default values must take effect (guard against dead fields).
    test::require(
        run_plain.prefill_embeddings_into_cache(embeddings, steps, 32, 8, 0).logits !=
            run_full.prefill_embeddings_into_cache(embeddings, steps, 32, 8, 0).logits,
        "scales change output");
}

void test_block_single_recompute() {
    // Block prefill (chunk 8) vs single-token stepping (chunk 1) vs
    // full-prefix direct recompute agree at the same positions.
    SyntheticHarness harness;
    auto config = harness.base_config();
    config.decoder.embedding_scale = 12.0F;
    config.decoder.logit_divisor = 8.0F;
    config.decoder.stack.residual_scale = 0.22F;
    config.decoder.stack.attention_scale = 0.0078125F;
    modules::CausalDecoderRuntime block(harness.execution, config, harness.weights);
    modules::CausalDecoderRuntime single(harness.execution, config, harness.weights);
    modules::CausalDecoderRuntime recompute(harness.execution, config, harness.weights);
    const int64_t steps = 24;
    const std::vector<float> embeddings = pattern_values(steps * harness.hidden, 0.25F, 0.1F);
    const auto block_out = block.prefill_embeddings_into_cache(embeddings, steps, 40, 8, 0);
    const auto single_out = single.prefill_embeddings_into_cache(embeddings, steps, 40, 1, 0);
    const float block_single = max_abs_diff(block_out.logits, single_out.logits);
    std::cout << "  block-vs-single max_abs=" << block_single << "\n";
    test::require(block_single <= 1e-5F, "block vs single");
    // Decode three forced steps on both, then compare against fresh direct
    // prefills of the growing prefix. The synthetic token table is all ones
    // (rank-1 fill), so every decoded token appends a ones row.
    std::vector<float> block_logits = block_out.logits;
    std::vector<float> single_logits = single_out.logits;
    for (int64_t k = 0; k < 3; ++k) {
        block_logits = block.decode_token(static_cast<int32_t>(k)).logits;
        single_logits = single.decode_token(static_cast<int32_t>(k)).logits;
        const float step_diff = max_abs_diff(block_logits, single_logits);
        std::cout << "  decode step " << k << " block-vs-single max_abs=" << step_diff << "\n";
        test::require(step_diff <= 1e-5F, "decode block vs single");
        std::vector<float> grown(embeddings);
        for (int64_t j = 0; j <= k; ++j) {
            grown.insert(grown.end(), static_cast<size_t>(harness.hidden), 1.0F);
        }
        const auto direct = recompute.prefill_embeddings(grown, steps + k + 1);
        const float direct_diff = max_abs_diff(block_logits, direct.logits);
        std::cout << "  decode step " << k << " vs full-recompute max_abs=" << direct_diff << "\n";
        test::require(direct_diff <= 1e-4F, "decode vs full recompute");
    }
}

void test_cache_reset_aba() {
    // A/B/A with a failed request between A and B: the failure must not
    // poison the cache, and re-prefilling A reproduces its logits exactly.
    SyntheticHarness harness;
    auto config = harness.base_config();
    config.decoder.embedding_scale = 12.0F;
    modules::CausalDecoderRuntime runtime(harness.execution, config, harness.weights);
    const int64_t steps = 16;
    const auto embeddings_a = pattern_values(steps * harness.hidden, 0.3F, 0.0F);
    const auto embeddings_b = pattern_values(steps * harness.hidden, 0.3F, 5.0F);
    const auto logits_a =
        runtime.prefill_embeddings_into_cache(embeddings_a, steps, 32, 8, 0).logits;
    bool threw = false;
    try {
        const auto bad = pattern_values(4 * harness.hidden, 0.1F, 0.0F);
        (void) runtime.prefill_embeddings_into_cache(bad, 8, 32, 8, 0);
    } catch (const std::runtime_error &) {
        threw = true;
    }
    test::require(threw, "shape-mismatched prefill rejected");
    const auto logits_b =
        runtime.prefill_embeddings_into_cache(embeddings_b, steps, 32, 8, 0).logits;
    test::require(logits_a != logits_b, "A/B logits differ");
    const auto logits_a2 =
        runtime.prefill_embeddings_into_cache(embeddings_a, steps, 32, 8, 0).logits;
    test::require(logits_a == logits_a2, "A/B/A cache reset exact");
}

// --- always-run: RoPE YaRN vs reference-derived divisors ---

void test_rope_yarn_positions() {
    // Reference-derived divisors: HF inv_freq buffer (pinned checkpoint,
    // external Python reference) + magnitude factor. For each probe position the
    // ggml parametric YaRN rotation must reproduce cos(p*inv)*m /
    // sin(p*inv)*m per pair on both Q-like (lower-hot) and K-like
    // (upper-hot) inputs through the shared RoPEModule.
    const auto fixture =
        read_text_file(fixtures_dir() / "rope_yarn_inv_freq.json");
    const auto inv_list = parse_number_list(fixture.substr(fixture.find('[')));
    test::require_eq(inv_list.size(), size_t{64}, "64 inverse freqs");
    // Magnitude comes from the fixture (HF attention_scaling buffer value),
    // asserted equal to the get_mscale formula 0.1*ln(4)+1, never hardcoded.
    const double magnitude = json_scalar(fixture, "attention_scaling");
    test::require(
        std::fabs(magnitude - (0.1 * std::log(4.0) + 1.0)) < 1e-12, "yarn magnitude formula");
    test::require(std::fabs(inv_list[0] - 1.0) < 1e-12, "inv[0] == 1");

    core::ExecutionContext execution(core::BackendConfig{core::BackendType::Cpu, 0, 8});
    ggml_context * context = ggml_init({64 * 1024 * 1024, nullptr, true});
    if (!context) {
        throw std::runtime_error("C2 test rope context failed");
    }
    struct ContextGuard {
        ggml_context * p;
        ~ContextGuard() {
            ggml_free(p);
        }
    } context_guard{context};
    core::ModuleBuildContext build{context, "granite_c2_rope", core::BackendType::Cpu};
    const int64_t heads = 64;
    const int64_t dim = 128;
    // Lower-hot (Q-like): head h carries 1.0 at lower-half index h.
    auto q_input = core::make_tensor(
        build, GGML_TYPE_F32, core::TensorShape::from_dims({1, 1, heads, dim}));
    auto k_input = core::make_tensor(
        build, GGML_TYPE_F32, core::TensorShape::from_dims({1, 1, heads, dim}));
    auto positions = core::make_tensor(build, GGML_TYPE_I32, core::TensorShape::from_dims({1}));
    modules::RoPEConfig rope;
    rope.dimensions = dim;
    rope.mode = GGML_ROPE_TYPE_NEOX;
    rope.theta = 10000.0F;
    rope.freq_scale = 0.25F;
    rope.ext_factor = 1.0F;
    // ggml derives the 0.1*ln(4)+1 magnitude from freq_scale when ext != 0;
    // passing it here as well would apply it twice.
    rope.attn_factor = 1.0F;
    rope.beta_fast = 32.0F;
    rope.beta_slow = 1.0F;
    rope.orig_context = 4096;
    auto q_out = modules::RoPEModule(rope).build(build, q_input, positions, nullptr);
    auto k_out = modules::RoPEModule(rope).build(build, k_input, positions, nullptr);
    ggml_set_output(q_out.tensor);
    ggml_set_output(k_out.tensor);
    ggml_cgraph * graph = ggml_new_graph_custom(context, 64, false);
    ggml_build_forward_expand(graph, q_out.tensor);
    ggml_build_forward_expand(graph, k_out.tensor);
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(context, execution.backend());
    if (!buffer) {
        throw std::runtime_error("C2 test rope buffer failed");
    }
    struct BufferGuard {
        ggml_backend_buffer_t p;
        ~BufferGuard() {
            ggml_backend_buffer_free(p);
        }
    } buffer_guard{buffer};

    std::vector<float> q_values(64 * 128, 0.0F);
    std::vector<float> k_values(64 * 128, 0.0F);
    for (int64_t h = 0; h < 64; ++h) {
        q_values[h * 128 + h] = 1.0F;          // lower-half hot
        k_values[h * 128 + 64 + h] = 1.0F;     // upper-half hot
    }
    ggml_backend_tensor_set(q_input.tensor, q_values.data(), 0, q_values.size() * sizeof(float));
    ggml_backend_tensor_set(k_input.tensor, k_values.data(), 0, k_values.size() * sizeof(float));

    const std::vector<int64_t> probe_positions = {0, 1, 4095, 4096, 4097, 16383};
    float worst = 0.0F;
    float worst_mag = 0.0F;
    for (int64_t pos : probe_positions) {
        // Float32 trig arguments carry ~1ulp of rounding at magnitude pos;
        // two valid implementations (HF float32 outer product vs ggml
        // iterative theta) agree only to about pos * 2e-7 on the angle. The
        // magnitude (cos^2 + sin^2) is argument-independent and checked
        // tightly instead.
        const float angle_tol = std::max(2e-6F, static_cast<float>(pos) * 2e-7F);
        const int32_t p = static_cast<int32_t>(pos);
        ggml_backend_tensor_set(positions.tensor, &p, 0, sizeof(int32_t));
        core::set_backend_threads(execution.backend(), 8);
        if (core::compute_backend_graph(execution.backend(), graph) != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("C2 test rope graph compute failed");
        }
        ggml_backend_synchronize(execution.backend());
        std::vector<float> q_got(64 * 128);
        std::vector<float> k_got(64 * 128);
        ggml_backend_tensor_get(q_out.tensor, q_got.data(), 0, q_got.size() * sizeof(float));
        ggml_backend_tensor_get(k_out.tensor, k_got.data(), 0, k_got.size() * sizeof(float));
        for (int64_t h = 0; h < 64; ++h) {
            const double angle = static_cast<double>(pos) * inv_list[static_cast<size_t>(h)];
            const float cos_e = static_cast<float>(std::cos(angle) * magnitude);
            const float sin_e = static_cast<float>(std::sin(angle) * magnitude);
            worst = std::max(worst, std::fabs(q_got[h * 128 + h] - cos_e));
            worst = std::max(worst, std::fabs(q_got[h * 128 + 64 + h] - sin_e));
            worst = std::max(worst, std::fabs(k_got[h * 128 + h] - -sin_e));
            worst = std::max(worst, std::fabs(k_got[h * 128 + 64 + h] - cos_e));
            const float mag_q = std::sqrt(
                q_got[h * 128 + h] * q_got[h * 128 + h] +
                q_got[h * 128 + 64 + h] * q_got[h * 128 + 64 + h]);
            const float mag_k = std::sqrt(
                k_got[h * 128 + h] * k_got[h * 128 + h] +
                k_got[h * 128 + 64 + h] * k_got[h * 128 + 64 + h]);
            worst_mag = std::max(worst_mag, std::fabs(mag_q - static_cast<float>(magnitude)));
            worst_mag = std::max(worst_mag, std::fabs(mag_k - static_cast<float>(magnitude)));
            test::require(
                std::fabs(q_got[h * 128 + h] - cos_e) <= angle_tol, "rope angle");
        }
        std::cout << "  rope pos=" << pos << " worst=" << worst << " worst_mag=" << worst_mag << "\n";
    }
    std::cout << "  rope worst over all positions=" << worst << " worst_mag=" << worst_mag << "\n";
    test::require(worst_mag <= 2e-6F, "rope yarn magnitude");
}

}  // namespace

namespace {

// --- checkpoint-gated: reference trace loading ---

struct ReferenceTrace {
    std::string json;
    std::vector<int32_t> input_ids;
    std::vector<int32_t> generated_ids;
    std::string text;
    std::string observed_stop;
};

ReferenceTrace load_reference_trace(
    const std::filesystem::path & trace_dir, const std::string & name) {
    const auto dir = trace_dir / (name + "-hf");
    ReferenceTrace out;
    out.json = read_text_file(dir / "trace.json");
    for (double id : json_numbers(out.json, "input_ids")) {
        out.input_ids.push_back(static_cast<int32_t>(id));
    }
    for (double id : json_numbers(out.json, "generated_ids")) {
        out.generated_ids.push_back(static_cast<int32_t>(id));
    }
    out.text = json_string(out.json, "text");
    out.observed_stop = json_string(out.json, "observed_stop");
    if (out.input_ids.empty() || out.generated_ids.empty()) {
        throw std::runtime_error("C2 test reference trace has empty id lists: " + name);
    }
    return out;
}

std::vector<float> load_stage(
    const std::filesystem::path & trace_dir,
    const std::string & name,
    const std::string & stage,
    size_t expect_elements) {
    const auto values = read_f32_file(trace_dir / (name + "-hf") / (stage + ".f32"));
    test::require_eq(values.size(), expect_elements, name + " " + stage + " size");
    return values;
}

// --- checkpoint-gated: full audio pipeline + decoder ---

struct AudioFixtureResult {
    engine::runtime::GreedyCausalDecoderRuntime::Prompt prompt;
    g2b::GraniteSpeech5Decoder::PrefillResult prefill;
    std::vector<std::vector<float>> forced;
    std::vector<int32_t> greedy;
    std::string text;
    std::vector<float> projected;
    std::shared_ptr<const engine::tokenizers::LlamaBpeTokenizer> tokenizer;
};

}  // namespace

namespace {

core::ExecutionContext make_cpu_execution() {
    return core::ExecutionContext(core::BackendConfig{core::BackendType::Cpu, 0, 8});
}

AudioFixtureResult run_audio_fixture(
    const std::filesystem::path & model_dir,
    const std::filesystem::path & wav_path,
    const std::vector<int32_t> & forced_continuation,
    int64_t max_new_tokens) {
    AudioFixtureResult out;
    const auto wav = engine::audio::read_wav_f32(wav_path);
    if (wav.samples.empty()) {
        throw std::runtime_error("C2 test audio has no samples");
    }
    engine::runtime::AudioBuffer buffer;
    buffer.sample_rate = wav.sample_rate;
    buffer.channels = wav.channels;
    buffer.samples = wav.samples;

    core::ExecutionContext execution = make_cpu_execution();
    const auto assets = g2b::load_granite_speech5_assets(model_dir);
    const g2b::GraniteSpeech5Frontend frontend(assets);
    const auto features = frontend.extract(buffer);
    g2b::GraniteSpeech5EncoderRuntime encoder(
        assets, execution, engine::assets::TensorStorageType::F32);
    const auto encoded = encoder.encode(features.stacked, features.encoder_steps);
    g2b::GraniteSpeech5ProjectorRuntime projector(
        assets, execution, engine::assets::TensorStorageType::F32);
    const auto projected = projector.project(encoded.last, features.encoder_steps);
    out.projected = projected.projected;

    out.tokenizer = g2b::load_granite_speech5_tokenizer(assets->resources);
    const g2b::GraniteSpeech5PromptBuilder builder(out.tokenizer);
    out.prompt = builder.build_decoder_prompt(
        "", projected.audio_placeholders, std::vector<float>(projected.projected));

    g2b::GraniteSpeech5Decoder decoder(assets->source, execution, true);
    out.prefill = decoder.prefill(out.prompt, max_new_tokens);
    const size_t n_steps =
        std::min<size_t>(8, forced_continuation.size());
    out.forced.reserve(n_steps);
    if (n_steps > 0) {
        out.forced.push_back(out.prefill.last_logits);
        for (size_t k = 1; k < n_steps; ++k) {
            out.forced.push_back(decoder.decode_token(forced_continuation[k - 1]));
        }
    }
    out.greedy = decoder.generate_greedy(out.prompt, max_new_tokens);
    out.text = g2b::granite_speech5_decode_generated(*out.tokenizer, out.greedy);
    return out;
}

void require_top1_margin_rule(
    const std::vector<float> & actual,
    const std::vector<float> & expected,
    const std::string & label) {
    const int32_t actual_top = top1(actual);
    const int32_t expected_top = top1(expected);
    const float margin = top_margin(expected);
    std::cout << "  margin [" << label << "]: ref_top=" << expected_top
              << " cpp_top=" << actual_top << " ref_margin=" << margin << "\n";
    if (margin > kTop1MarginGate) {
        test::require(actual_top == expected_top, label + " stable top-1");
    }
}

void test_audio_fixture_parity(
    const std::filesystem::path & model_dir,
    const std::filesystem::path & trace_dir,
    const std::string & name,
    const std::string & wav_relpath) {
    std::cout << "fixture [" << name << "]\n";
    const ReferenceTrace ref = load_reference_trace(trace_dir, name);
    const auto wav_path = source_dir() / wav_relpath;
    AudioFixtureResult pipe =
        run_audio_fixture(model_dir, wav_path, ref.generated_ids, 512);

    // Prompt parity: expanded IDs match the reference processor exactly.
    test::require(pipe.prompt.input_ids == ref.input_ids, name + " prompt ids");

    const size_t steps = ref.input_ids.size();
    const size_t width = 2048;
    const auto ref_merged = load_stage(trace_dir, name, "merged_prefill", steps * width);
    const auto ref_scaled = load_stage(trace_dir, name, "scaled_prefill", steps * width);
    const auto ref_last = load_stage(trace_dir, name, "last_pos_logits", 100353);
    require_prefill_close(pipe.prefill.merged, ref_merged, name + " merged_prefill");
    require_prefill_close(pipe.prefill.scaled, ref_scaled, name + " scaled_prefill");
    // Reference last_pos_logits.f32 is the raw lm_head output (pre-/8; the
    // reference hooks lm_head itself, verified bitwise as
    // last_pos_logits == 8 * forced_step0). The decoder carries post-scale
    // logits, so this stage gates the exact x8 inverse (exact in fp32).
    auto raw_logits = pipe.prefill.last_logits;
    for (float & value : raw_logits) {
        value *= 8.0F;
    }
    require_logits_close(raw_logits, ref_last, name + " last_pos_logits");
    // Internal consistency: forced step 0 is the prefill last-position
    // distribution (matches reference.py forced step construction).
    test::require(pipe.forced[0] == pipe.prefill.last_logits, name + " forced_step0 == last");

    // Embed-scale fixture: audio rows are substituted verbatim (exact, see
    // the injection-position check below); token rows flow through the same
    // lookup (covered by the merged gate above); the x12 is applied exactly
    // once (scaled gate above + tripwire below).
    // Audio positions from the prompt injection must hold the projector
    // outputs exactly.
    test::require_eq(
        pipe.prompt.injection.positions.size(),
        static_cast<size_t>(pipe.prompt.injection.tokens),
        "injection positions");
    for (size_t i = 0; i < pipe.prompt.injection.positions.size(); ++i) {
        const int32_t pos = pipe.prompt.injection.positions[i];
        for (size_t j = 0; j < width; ++j) {
            if (pipe.prefill.merged[pos * width + j] != pipe.projected[i * width + j]) {
                throw std::runtime_error(name + " merged audio row mismatch");
            }
        }
    }
    // Double-scale tripwire: scaled must be exactly 12x merged (not 144x).
    double scale_num = 0.0;
    double scale_den = 0.0;
    for (size_t i = 0; i < steps * width; ++i) {
        scale_num += static_cast<double>(pipe.prefill.scaled[i]) *
            static_cast<double>(pipe.prefill.merged[i]);
        scale_den += static_cast<double>(pipe.prefill.merged[i]) *
            static_cast<double>(pipe.prefill.merged[i]);
    }
    const double scale_ratio = scale_num / scale_den;
    std::cout << "  embed scale ratio=" << scale_ratio << "\n";
    test::require(std::fabs(scale_ratio - 12.0) < 1e-6, name + " embed scale ratio");

    // Forced cached steps: full-logit gates + stable-margin top-1 rule.
    test::require_eq(pipe.forced.size(), size_t{8}, name + " forced step count");
    for (size_t k = 0; k < 8; ++k) {
        const auto ref_forced =
            load_stage(trace_dir, name, "forced_step" + std::to_string(k), 100353);
        require_logits_close(pipe.forced[k], ref_forced, name + " forced_step" + std::to_string(k));
        require_top1_margin_rule(pipe.forced[k], ref_forced, name + " step" + std::to_string(k));
    }

    // Exact visible greedy sequences (reference preserves the observed stop;
    // the C++ runtime excludes stop IDs, so strip before comparing).
    auto ref_visible = g2b::granite_speech5_strip_observed_eos(ref.generated_ids);
    test::require(pipe.greedy == ref_visible, name + " greedy ids");
    test::require(pipe.text == ref.text, name + " visible text");
    std::cout << "  greedy text [" << name << "]: " << pipe.text << "\n";
}

void test_text_only_4096_crossing(const std::filesystem::path & model_dir) {
    std::cout << "text-only 4096 crossing\n";
    const auto ids_list = parse_number_list(read_text_file(fixtures_dir() / "text_only_4096_crossing_ids.json"));
    std::vector<int32_t> ids;
    ids.reserve(ids_list.size());
    for (double id : ids_list) {
        ids.push_back(static_cast<int32_t>(id));
    }
    test::require(ids.size() > 4096, "crossing prompt spans 4096");
    const auto ref_last =
        read_f32_file(fixtures_dir() / "text_only_4096_crossing_last_logits.f32");
    test::require_eq(ref_last.size(), size_t{100353}, "crossing ref logits");

    core::ExecutionContext execution = make_cpu_execution();
    const auto assets = g2b::load_granite_speech5_assets(model_dir);
    engine::runtime::GreedyCausalDecoderRuntime::Prompt prompt;
    prompt.input_ids = ids;
    {
        g2b::GraniteSpeech5Decoder decoder(assets->source, execution, true);
        // Block prefill at two chunk widths must agree with each other and
        // with the reference last-position logits.
        const auto out64 = decoder.prefill(prompt, 8, 64);
        const auto out512 = decoder.prefill(prompt, 8, 512);
        const float chunk_diff = max_abs_diff(out64.last_logits, out512.last_logits);
        std::cout << "  chunk64-vs-512 max_abs=" << chunk_diff << "\n";
        test::require(chunk_diff <= 1e-4F, "crossing chunk widths");
        require_logits_close(out64.last_logits, ref_last, "crossing last_pos_logits");
        require_top1_margin_rule(out64.last_logits, ref_last, "crossing top-1");
        std::cout << "  crossing top1 cpp=" << top1(out64.last_logits)
                  << " ref=" << top1(ref_last) << "\n";
        // Long-context single-token decode smoke: finite logits past 4096.
        const auto step = decoder.decode_token(ids.back());
        for (float value : step) {
            test::require(std::isfinite(value), "crossing decode finite");
        }
        test::require_eq(step.size(), size_t{100353}, "crossing decode size");
    }
}

void test_greedy_wrapper_parity(
    const std::filesystem::path & model_dir,
    const std::filesystem::path & trace_dir,
    const std::string & name,
    const std::string & wav_relpath) {
    // The brief-mandated generate path (GreedyCausalDecoderRuntime with
    // reuse_graphs=true) must reproduce the reference visible sequence and
    // the logits-capable decoder's greedy output.
    std::cout << "greedy wrapper [" << name << "]\n";
    const ReferenceTrace ref = load_reference_trace(trace_dir, name);
    core::ExecutionContext execution = make_cpu_execution();
    const auto assets = g2b::load_granite_speech5_assets(model_dir);
    const auto wav = engine::audio::read_wav_f32(source_dir() / wav_relpath);
    engine::runtime::AudioBuffer buffer;
    buffer.sample_rate = wav.sample_rate;
    buffer.channels = wav.channels;
    buffer.samples = wav.samples;
    const g2b::GraniteSpeech5Frontend frontend(assets);
    const auto features = frontend.extract(buffer);
    g2b::GraniteSpeech5EncoderRuntime encoder(
        assets, execution, engine::assets::TensorStorageType::F32);
    const auto encoded = encoder.encode(features.stacked, features.encoder_steps);
    g2b::GraniteSpeech5ProjectorRuntime projector(
        assets, execution, engine::assets::TensorStorageType::F32);
    const auto projected = projector.project(encoded.last, features.encoder_steps);
    const auto tokenizer = g2b::load_granite_speech5_tokenizer(assets->resources);
    const g2b::GraniteSpeech5PromptBuilder builder(tokenizer);
    auto prompt = builder.build_decoder_prompt(
        "", projected.audio_placeholders, std::vector<float>(projected.projected));
    std::vector<int32_t> wrapper_ids;
    {
        engine::runtime::GreedyCausalDecoderRuntime wrapper(
            assets->source, g2b::make_granite_speech5_greedy_decoder_spec(), execution,
            1024ull * 1024ull * 1024ull, 1024ull * 1024ull * 1024ull,
            4ull * 1024ull * 1024ull, engine::assets::TensorStorageType::F32);
        wrapper_ids = wrapper.generate(prompt, 512, true);
    }
    auto ref_visible = g2b::granite_speech5_strip_observed_eos(ref.generated_ids);
    test::require(wrapper_ids == ref_visible, name + " wrapper greedy ids");
    // Wrapper vs logits-decoder greedy loop on a text-only prompt (both
    // paths, same weights source, sequential scopes).
    std::vector<int32_t> text_ids(ref.input_ids.begin(), ref.input_ids.begin() + 32);
    engine::runtime::GreedyCausalDecoderRuntime::Prompt text_prompt;
    text_prompt.input_ids = text_ids;
    std::vector<int32_t> wrapper_text;
    {
        engine::runtime::GreedyCausalDecoderRuntime wrapper(
            assets->source, g2b::make_granite_speech5_greedy_decoder_spec(), execution,
            1024ull * 1024ull * 1024ull, 1024ull * 1024ull * 1024ull,
            4ull * 1024ull * 1024ull, engine::assets::TensorStorageType::F32);
        wrapper_text = wrapper.generate(text_prompt, 8, true);
    }
    std::vector<int32_t> decoder_text;
    {
        g2b::GraniteSpeech5Decoder decoder(assets->source, execution, true);
        decoder_text = decoder.generate_greedy(text_prompt, 8);
    }
    test::require(wrapper_text == decoder_text, name + " wrapper vs decoder text greedy");

    // Real A/B/A cache reset on text prompts with a failed budget request
    // between A and B.
    {
        g2b::GraniteSpeech5Decoder decoder(assets->source, execution, true);
        engine::runtime::GreedyCausalDecoderRuntime::Prompt prompt_a;
        prompt_a.input_ids = std::vector<int32_t>(text_ids.begin(), text_ids.begin() + 16);
        engine::runtime::GreedyCausalDecoderRuntime::Prompt prompt_b;
        prompt_b.input_ids = std::vector<int32_t>(text_ids.begin() + 16, text_ids.begin() + 32);
        const auto logits_a = decoder.prefill(prompt_a, 4).last_logits;
        bool threw = false;
        try {
            (void) decoder.generate_greedy(prompt_a, 16384);
        } catch (const std::runtime_error &) {
            threw = true;
        }
        test::require(threw, "over-budget greedy rejected");
        const auto logits_b = decoder.prefill(prompt_b, 4).last_logits;
        test::require(logits_a != logits_b, "A/B differ");
        const auto logits_a2 = decoder.prefill(prompt_a, 4).last_logits;
        test::require(logits_a == logits_a2, "A/B/A cache reset exact");
    }
}

}  // namespace

int main(int argc, char ** argv) {
    engine::debug::configure_logging({argc > 1 && std::string(argv[1]) == "--log"});
    const auto started = std::chrono::steady_clock::now();
    try {
        test_budget_rules();
        test_binding_values();
        test_neutral_defaults();
        test_single_application();
        test_block_single_recompute();
        test_cache_reset_aba();
        if (std::getenv("GRANITE_SPEECH5_FIXTURES")) {
            test_rope_yarn_positions();
        } else {
            std::cout << "NOTE: reference YaRN parity requires GRANITE_SPEECH5_FIXTURES\n";
        }
        if (const auto model_dir = find_model_dir()) {
            const char * traces = std::getenv("GRANITE_SPEECH5_TRACES");
            if (traces == nullptr) {
                std::cout << "NOTE: C2 checkpoint parity requires GRANITE_SPEECH5_TRACES\n";
            } else {
                const std::filesystem::path trace_dir(traces);
                test_audio_fixture_parity(
                    *model_dir,
                    trace_dir,
                    "librispeech-0000",
                    "assets/asr_validation/librispeech/librispeech_test_clean_6930-75918-0000.wav");
                test_audio_fixture_parity(
                    *model_dir,
                    trace_dir,
                    "ja",
                    "assets/asr_validation/common_voice_ja/common_voice_ja_20461197.wav");
                if (std::getenv("GRANITE_SPEECH5_FIXTURES")) test_text_only_4096_crossing(*model_dir);
                test_greedy_wrapper_parity(
                    *model_dir,
                    trace_dir,
                    "librispeech-0000",
                    "assets/asr_validation/librispeech/librispeech_test_clean_6930-75918-0000.wav");
                std::cout << "PASS: granite_speech5 C2 unit checks (checkpoint checks; reference parity requires explicit references)\n";
                engine::debug::timing_log_scalar("session.wall_ms", engine::debug::elapsed_ms(started));
        return 0;
            }
        } else {
            std::cout << "NOTE: C2 checkpoint parity skipped (set GRANITE_SPEECH5_MODEL)\n";
        }
        std::cout << "PASS: granite_speech5 C2 unit checks (synthetic)\n";
    } catch (const std::exception & error) {
        std::cerr << "FAIL: " << error.what() << "\n";
        return 1;
    }
    engine::debug::timing_log_scalar("session.wall_ms", engine::debug::elapsed_ms(started));
        return 0;
}
