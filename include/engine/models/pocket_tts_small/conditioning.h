#pragma once

#include "engine/models/pocket_tts_small/assets.h"
#include "engine/models/pocket_tts_small/flow_lm.h"

#include "engine/framework/assets/tensor_source.h"

#include <cstddef>
#include <cstdint>
#include <ggml-backend.h>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace engine::models::pocket_tts_small {

struct Assets;
struct HostWeights;

struct TextConditionerConfig {
    int64_t hidden_size = 512;
};

struct TextConditioningResult {
    std::string prepared_text;
    std::vector<int32_t> tokens;
    std::vector<float> text_embeddings;
};

class TextConditioner {
public:
    explicit TextConditioner(TextConditionerConfig config = {});

    TextConditioningResult prepare(
        const Assets & manifest,
        const HostWeights & weights,
        const std::string & text,
        ggml_backend_t backend = nullptr,
        int threads = 1) const;

    // Splits a prompt the way the reference implementation does (pocket_tts_small
    // split_into_best_sentences): sentence boundaries found on the model's own
    // end-of-sentence tokens, sentences longer than max_tokens split again on
    // ",;:", then regrouped into chunks of at most max_tokens tokens.
    std::vector<std::string> split_into_sentence_chunks(
        const Assets & manifest,
        const std::string & text,
        int64_t max_tokens) const;

private:
    TextConditionerConfig config_;
    mutable std::optional<assets::TensorDataF32> projected_embeddings_;
};

struct Assets;
struct BackendWeights;

struct MimiPromptEmbedding {
    std::vector<float> values;
    int64_t frames = 0;
};

class MimiEncoder {
public:
    MimiEncoder();
    ~MimiEncoder();

    MimiPromptEmbedding encode_prompt_embedding(
        ggml_backend_t backend,
        int threads,
        const Assets & manifest,
        const BackendWeights & weights,
        const std::vector<float> & waveform,
        size_t graph_context_bytes) const;

private:
    struct Runtime;
    mutable std::unique_ptr<Runtime> runtime_;
};

struct Assets;
struct BackendWeights;

struct VoiceConditioningResult {
    VoiceConditioningPlan plan;
    FlowLMState acoustic_state;
};

class VoiceConditioner {
public:
    explicit VoiceConditioner(FlowLMConfig flow_config = {});

    VoiceConditioningResult prepare(
        const VoiceConditioningPlan & plan,
        const Assets & manifest,
        const BackendWeights & weights,
        ggml_backend_t backend,
        int threads,
        size_t mimi_encoder_graph_context_bytes,
        size_t flow_weights_view_context_bytes,
        size_t flow_step_graph_context_bytes) const;

private:
    FlowLM flow_lm_;
    MimiEncoder mimi_encoder_;
};

}  // namespace engine::models::pocket_tts_small
