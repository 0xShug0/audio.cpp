#include "engine/community_models/kitten_tts2/decoder.h"
#include "engine/models/chatterbox/s3gen_inference.h"

#include <cmath>
#include <cstdlib>
#include <fstream>
#include <stdexcept>

namespace engine::community_models::kitten_tts2 {
namespace s3 = engine::models::chatterbox;
struct WaveformDecoder::Impl {
    const core::ExecutionContext & execution;
    std::shared_ptr<const s3::S3FlowEncoderWeights> encoder;
    std::shared_ptr<const s3::S3FlowDecoderWeights> decoder;
    s3::HiFTVocoderComponent vocoder;
    s3::S3GenSessionCache cache;
    Impl(std::shared_ptr<const assets::TensorSource> source, const core::ExecutionContext & context)
        : execution(context),
          encoder(s3::load_s3_flow_encoder_weights(*source, context, assets::TensorStorageType::F32)),
          decoder(s3::load_s3_flow_decoder_weights(*source, context, assets::TensorStorageType::F32)),
          vocoder(s3::HiFTVocoderComponent::load_from_source(source, context, assets::TensorStorageType::F32)),
          cache(context.config()) {
        if (!s3::s3_flow_decoder_is_meanflow(*decoder))
            throw std::runtime_error("Kitten TTS 2 requires the S3 meanflow decoder checkpoint");
    }
};
WaveformDecoder::WaveformDecoder(std::shared_ptr<const assets::TensorSource> source,
    const core::ExecutionContext & execution) : impl_(std::make_unique<Impl>(std::move(source), execution)) {}
WaveformDecoder::~WaveformDecoder() = default;

std::vector<float> WaveformDecoder::decode(const std::vector<int32_t> & codes,
    const std::vector<int64_t> & prompt_tokens, const std::vector<float> & prompt_mel,
    const std::vector<float> & embedding, uint64_t seed) {
    if (codes.empty()) throw std::runtime_error("Kitten TTS 2 generated no speech tokens");
    if (prompt_tokens.empty() || prompt_mel.empty() || prompt_mel.size() % 80 || embedding.size() != 192)
        throw std::runtime_error("invalid Kitten TTS 2 decoder conditioning");
    s3::EmbedReferenceOutputs ref;
    ref.prompt_tokens.assign(prompt_tokens.begin(), prompt_tokens.end());
    ref.prompt_token_count = ref.prompt_tokens.size();
    ref.prompt_feat = prompt_mel;
    ref.prompt_feat_frames = prompt_mel.size() / 80;
    ref.prompt_feat_dims = 80;
    ref.embedding = embedding;
    ref.embedding_size = 192;
    auto tokens = codes;
    tokens.insert(tokens.end(), 3, 4299); // Upstream S3 lookahead preserves the final syllable.
    auto result = s3::compute_s3gen_inference(impl_->cache, *impl_->encoder, *impl_->decoder,
        impl_->vocoder, ref, tokens, tokens.size(), 2, 0.0f, false, {}, seed, seed,
        impl_->execution.config());
    if (result.waveform.empty()) throw std::runtime_error("Kitten decoder returned empty audio");
    for (float value : result.waveform)
        if (!std::isfinite(value)) throw std::runtime_error("Kitten decoder returned non-finite audio");
    if (const char * trace = std::getenv("AUDIOCPP_KITTEN_TTS2_TRACE_DIR"); trace && *trace) {
        std::ofstream out(std::filesystem::path(trace) / "mel.f32", std::ios::binary);
        out.write(reinterpret_cast<const char *>(result.mel.data()), result.mel.size() * sizeof(float));
    }
    return result.waveform;
}
}
