#include "engine/models/qwen3_forced_aligner/nagisa_segmenter.h"

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/core/prepared_graph.h"
#include "engine/framework/debug/trace.h"
#include "engine/framework/io/json.h"
#include "engine/framework/modules/linear_module.h"
#include "engine/framework/modules/recurrent_modules.h"

#include <utf8proc.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdlib>
#include <stdexcept>
#include <unordered_map>

namespace engine::models::qwen3_forced_aligner {

struct NagisaSegmenter::Impl {
    core::ExecutionContext execution{{core::BackendType::Cpu, 0, 1}};
    core::BackendWeightStore store{execution.backend(), core::BackendType::Cpu, "qwen3_forced_aligner.nagisa.weights", 1024 * 1024};
    std::array<std::unordered_map<std::string, int64_t>, 3> vocab;
    std::array<assets::TensorDataF32, 4> embeddings;
    std::vector<float> transitions;
    modules::BidirectionalLSTMWeights lstm;
    modules::LinearWeights projection;
    core::TensorValue zero;
    int64_t window = 0, hidden = 0, input_width = 0, graph_length = 0;
    std::unique_ptr<core::PreparedGraph> graph;
    ggml_tensor * input = nullptr;
    ggml_tensor * output = nullptr;

    explicit Impl(const assets::ResourceBundle & resources) {
        if (!resources.has_file("nagisa_config") || !resources.has_tensor_source("nagisa_weights")) {
            throw std::runtime_error("Japanese alignment tokenizer assets are missing (nagisa.json and Nagisa weights). Download the updated Qwen3-ForcedAligner-0.6B GGUF from audio-cpp/audio.cpp-gguf, or repack your GGUF with the tokenizer assets as described in docs/models/qwen3_forced_aligner.md");
        }
        const auto config = resources.parse_json("nagisa_config");
        window = config.require("window_size").as_i64();
        hidden = config.require("hidden_size").as_i64();
        if (window <= 0 || window % 2 == 0 || hidden <= 0) {
            throw std::runtime_error("Invalid Nagisa segmenter dimensions");
        }
        const auto source = resources.open_tensor_source("nagisa_weights");
        const std::array<std::string, 4> names{"unigram", "bigram", "word", "chartype"};
        for (size_t i = 0; i < names.size(); ++i) {
            embeddings[i] = source->require_f32_tensor(names[i]);
            if (embeddings[i].shape.rank != 2) throw std::runtime_error("Nagisa embedding must be a matrix");
            if (i < vocab.size()) {
                for (const auto & item : config.require(names[i]).as_object()) {
                    const auto id = item.second.as_i64();
                    if (id < 0 || id >= embeddings[i].shape.dims[0]) throw std::runtime_error("Nagisa vocabulary ID outside embedding table");
                    vocab[i].emplace(item.first, id);
                }
            }
        }
        if (embeddings[3].shape.dims[0] != 7) throw std::runtime_error("Nagisa requires seven character types");
        input_width = window * (embeddings[0].shape.dims[1] + embeddings[1].shape.dims[1] + embeddings[3].shape.dims[1]) + 2 * embeddings[2].shape.dims[1];
        for (int direction = 0; direction < 2; ++direction) {
            const std::string prefix = direction == 0 ? "forward." : "reverse.";
            auto & weights = direction == 0 ? lstm.forward : lstm.reverse;
            weights.weight_ih = store.load_f32_tensor(*source, prefix + "weight_ih", {4 * hidden, input_width});
            weights.weight_hh = store.load_f32_tensor(*source, prefix + "weight_hh", {4 * hidden, hidden});
            weights.bias_ih = store.load_f32_tensor(*source, prefix + "bias_ih", {4 * hidden});
            weights.bias_hh = store.load_f32_tensor(*source, prefix + "bias_hh", {4 * hidden});
        }
        projection = {store.load_f32_tensor(*source, "projection.weight", {6, 2 * hidden}),
                      store.load_f32_tensor(*source, "projection.bias", {6})};
        transitions = source->require_f32("transitions", {6, 6});
        zero = store.make_f32(core::TensorShape::from_dims({1, hidden}), std::vector<float>(hidden, 0.0F));
        store.upload();
    }

    std::vector<std::string> segment(const std::string & text) {
        const auto start = std::chrono::steady_clock::now();
        // Nagisa: rstrip, NFKC, Turkish capital I replacement, then ASCII-space
        // to ideographic-space replacement. Preserve normalized case in words;
        // only the feature lookup uses lowercase characters.
        std::vector<int32_t> original;
        for (size_t offset = 0; offset < text.size();) {
            int32_t cp = 0;
            const auto n = utf8proc_iterate(reinterpret_cast<const uint8_t *>(text.data() + offset), text.size() - offset, &cp);
            if (n <= 0) throw std::runtime_error("Nagisa input contains invalid UTF-8");
            original.push_back(cp);
            offset += static_cast<size_t>(n);
        }
        while (!original.empty()) {
            const auto c = original.back();
            const auto category = utf8proc_category(c);
            if (category != UTF8PROC_CATEGORY_ZS && category != UTF8PROC_CATEGORY_ZL && category != UTF8PROC_CATEGORY_ZP && c != 0x85 && !(c >= 9 && c <= 13) && !(c >= 0x1c && c <= 0x1f)) break;
            original.pop_back();
        }
        std::string stripped;
        uint8_t encoded[4];
        for (auto cp : original) {
            const auto n = utf8proc_encode_char(cp, encoded);
            stripped.append(reinterpret_cast<char *>(encoded), n);
        }
        std::unique_ptr<uint8_t, decltype(&std::free)> normalized(utf8proc_NFKC(reinterpret_cast<const uint8_t *>(stripped.c_str())), &std::free);
        if (!normalized) throw std::runtime_error("Nagisa NFKC normalization failed");
        const std::string normalized_text(reinterpret_cast<char *>(normalized.get()));
        std::vector<std::string> surface, lower;
        std::vector<int64_t> types;
        for (size_t offset = 0; offset < normalized_text.size();) {
            int32_t cp = 0;
            const auto n = utf8proc_iterate(reinterpret_cast<const uint8_t *>(normalized_text.data() + offset), normalized_text.size() - offset, &cp);
            if (n <= 0) throw std::runtime_error("Nagisa normalization produced invalid UTF-8");
            offset += static_cast<size_t>(n);
            if (cp == 0x130) cp = 'I';
            if (cp == ' ') cp = 0x3000;
            auto size = utf8proc_encode_char(cp, encoded);
            surface.emplace_back(reinterpret_cast<char *>(encoded), size);
            cp = utf8proc_tolower(cp);
            size = utf8proc_encode_char(cp, encoded);
            lower.emplace_back(reinterpret_cast<char *>(encoded), size);
            types.push_back(cp >= 0x3040 && cp <= 0x309f ? 0 : cp >= 0x30a1 && cp <= 0x30fa ? 1 : cp >= 0x4e00 && cp <= 0x9fa5 ? 2 : cp >= 'a' && cp <= 'z' ? 3 : cp >= '0' && cp <= '9' ? 4 : 5);
        }
        const int64_t length = static_cast<int64_t>(surface.size());
        if (length == 0) return {};
        std::vector<float> features(static_cast<size_t>(length * input_width), 0.0F);
        for (int64_t i = 0; i < length; ++i) {
            auto dst = features.begin() + i * input_width;
            for (const size_t table : {size_t{0}, size_t{1}, size_t{3}}) {
                const int64_t width = embeddings[table].shape.dims[1];
                for (int64_t j = i - window / 2; j <= i + window / 2; ++j) {
                    int64_t id = table == 3 ? 6 : 1;
                    if (j >= 0 && j < length) {
                        if (table == 3) id = types[j];
                        else {
                            const auto key = table == 0 ? lower[j] : lower[j] + (j + 1 == length ? "<E>" : lower[j + 1]);
                            const auto found = vocab[table].find(key);
                            id = found == vocab[table].end() ? vocab[table].at("oov") : found->second;
                        }
                    }
                    std::copy_n(embeddings[table].values.begin() + id * width, width, dst);
                    dst += width;
                }
            }
            const auto width = embeddings[2].shape.dims[1];
            for (const bool ending : {false, true}) {
                std::string word;
                bool found_word = false;
                for (int64_t n = 0; n < 8; ++n) {
                    const auto j = ending ? i - n : i + n;
                    if (j < 0 || j >= length) break;
                    word = ending ? lower[j] + word : word + lower[j];
                    const auto found = vocab[2].find(word);
                    if (found == vocab[2].end()) continue;
                    found_word = true;
                    for (int64_t k = 0; k < width; ++k) dst[k] += embeddings[2].values[found->second * width + k];
                }
                if (!found_word) std::copy_n(embeddings[2].values.begin() + vocab[2].at("oov") * width, width, dst);
                dst += width;
            }
        }
        if (graph_length != length) {
            graph.reset();
            graph_length = 0;
            const size_t nodes = static_cast<size_t>(length) * 128 + 1024;
            const size_t bytes = ggml_graph_overhead_custom(nodes, false) + 3 * nodes * ggml_tensor_overhead();
            graph = std::make_unique<core::PreparedGraph>(execution, core::PreparedGraphConfig{"qwen3_forced_aligner.nagisa", bytes, nodes});
            auto & ctx = graph->build_context();
            input = ggml_new_tensor_2d(ctx.ggml, GGML_TYPE_F32, input_width, length);
            ggml_set_input(input);
            const auto x = core::wrap_tensor(input, core::TensorShape::from_dims({length, input_width}), GGML_TYPE_F32);
            const auto h = modules::BidirectionalLSTMModule({input_width, hidden}).build(ctx, x, zero, zero, zero, zero, lstm);
            output = modules::LinearModule({2 * hidden, 6, true}).build(ctx, h.sequence, projection).tensor;
            ggml_set_output(output);
            ggml_build_forward_expand(graph->graph(), output);
            debug::timing_log_context_reservation("qwen3_forced_aligner.nagisa.graph", ctx.ggml);
            graph->allocate();
            graph_length = length;
        }
        ggml_backend_tensor_set(input, features.data(), 0, features.size() * sizeof(float));
        graph->compute();
        std::vector<float> logits(static_cast<size_t>(length * 6));
        ggml_backend_tensor_get(output, logits.data(), 0, logits.size() * sizeof(float));
        // Upstream np_viterbi uses float64 scores, next/previous transition
        // indexing, start=4, end=5, and the first index on ties.
        std::array<double, 6> scores;
        scores.fill(-1e10);
        scores[4] = 0.0;
        std::vector<std::array<int, 6>> backpointers(length);
        for (int64_t i = 0; i < length; ++i) {
            std::array<double, 6> next;
            for (int tag = 0; tag < 6; ++tag) {
                int best = 0;
                for (int prev = 1; prev < 6; ++prev) {
                    if (scores[prev] + transitions[tag * 6 + prev] > scores[best] + transitions[tag * 6 + best]) best = prev;
                }
                backpointers[i][tag] = best;
                next[tag] = scores[best] + transitions[tag * 6 + best] + logits[i * 6 + tag];
            }
            scores = next;
        }
        int best = 0;
        for (int tag = 1; tag < 6; ++tag) if (scores[tag] + transitions[5 * 6 + tag] > scores[best] + transitions[5 * 6 + best]) best = tag;
        std::vector<int> tags(length);
        for (int64_t i = length; i-- > 0;) { tags[i] = best; best = backpointers[i][best]; }
        std::vector<std::string> words;
        std::string pending;
        for (int64_t i = 0; i < length; ++i) {
            if (tags[i] == 3 && !pending.empty()) { words.push_back(std::move(pending)); pending.clear(); }
            pending += surface[i];
            if (tags[i] == 2 || tags[i] == 3) { words.push_back(std::move(pending)); pending.clear(); }
        }
        if (!pending.empty()) words.push_back(std::move(pending));
        debug::timing_log_scalar("qwen3_forced_aligner.nagisa.wall_ms", std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count());
        return words;
    }
};

NagisaSegmenter::NagisaSegmenter(const assets::ResourceBundle & resources) : impl_(std::make_unique<Impl>(resources)) {}
NagisaSegmenter::~NagisaSegmenter() = default;
std::vector<std::string> NagisaSegmenter::segment(const std::string & text) { return impl_->segment(text); }

}  // namespace engine::models::qwen3_forced_aligner
