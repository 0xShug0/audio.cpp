#pragma once

#include "engine/framework/assets/resource_bundle.h"
#include "engine/framework/audio/dsp.h"
#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/core/execution_context.h"
#include "engine/framework/modules/speech_encoders/whisper_embedding.h"
#include "engine/framework/modules/transformers/transformer_blocks.h"
#include "engine/framework/runtime/model.h"
#include "engine/framework/tokenizers/llama_bpe.h"

#include <memory>
#include <string>
#include <vector>

namespace engine::models::crisperwhisper {

struct CrisperWhisperAssets {
    assets::ResourceBundle resources;
    std::shared_ptr<const assets::TensorSource> source;
    std::shared_ptr<tokenizers::LlamaBpeTokenizer> tokenizer;
    modules::WhisperEmbeddingConfig encoder;
    int64_t decoder_layers = 0, decoder_heads = 0, decoder_ffn = 0;
    int64_t vocabulary_size = 0, max_positions = 0;
    int32_t eos = 0;
    std::vector<int32_t> suppress;
    std::vector<bool> control_tokens;
    std::vector<std::pair<int64_t, int64_t>> alignment_heads;

    std::vector<int32_t> prompt(const std::string & language, const std::string & mode,
                                const std::string & context = {}, const std::string & transcript = {}) const;
    std::string decode_text(const std::vector<int32_t> & tokens) const;
};

struct CrisperWhisperWeights {
    std::unique_ptr<core::BackendWeightStore> store;
    modules::WhisperEmbeddingWeights encoder;
    std::vector<modules::TransformerDecoderBlockWeights> decoder;
    core::TensorValue embedding, positions;
    modules::NormWeights decoder_norm;
};

struct CrisperWhisperWord {
    std::string text;
    double start = -1.0, end = -1.0;
};

struct CrisperWhisperDecodeResult {
    std::vector<int32_t> tokens;
    std::vector<float> attention;
    audio::WhisperLogMelFeatures features;
    std::vector<CrisperWhisperWord> words;
    bool reached_eos = false;
    double stop_probability = -1.0;
};

std::vector<CrisperWhisperWord> align_words(const CrisperWhisperAssets & assets,
    const CrisperWhisperDecodeResult & decoded, const std::string & language);
std::vector<CrisperWhisperWord> align_transcript(const std::string & text,
    const std::vector<CrisperWhisperWord> & hypothesis, double duration);

std::shared_ptr<const CrisperWhisperAssets> load_assets(const std::filesystem::path & path);
std::unique_ptr<CrisperWhisperWeights> load_weights(
    const CrisperWhisperAssets & assets, core::ExecutionContext & execution, assets::TensorStorageType type);

class CrisperWhisperEncoderDecoderRuntime {
public:
    CrisperWhisperEncoderDecoderRuntime(const CrisperWhisperAssets & assets,
        const CrisperWhisperWeights & weights, core::ExecutionContext & execution);
    ~CrisperWhisperEncoderDecoderRuntime();
    CrisperWhisperDecodeResult transcribe(const std::vector<float> & samples,
        const std::vector<int32_t> & prompt, int64_t max_tokens,
        const std::string & language, double recovery_tail_sec = 0.0,
        const std::vector<int32_t> & sibling_prompt = {});

private:
    struct Graphs;
    const CrisperWhisperAssets & assets_;
    const CrisperWhisperWeights & weights_;
    core::ExecutionContext & execution_;
    audio::WhisperLogMelExtractor frontend_;
    std::unique_ptr<Graphs> graphs_;
};

std::shared_ptr<runtime::IVoiceModelLoader> make_crisperwhisper_loader();

}  // namespace engine::models::crisperwhisper
