#pragma once

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/core/execution_context.h"

#include <memory>
#include <vector>

namespace engine::models::zonos3 {

// ZONOS3's frozen semantic teacher (ssl-teacher.gguf, w2vbert_ssl): SeamlessM4T fbank frontend on 16 kHz audio,
// then a 16-layer Wav2Vec2-BERT conformer under emulated CUDA BF16 autocast. Output is layer 16.
class Zonos3SslTeacher {
public:
    Zonos3SslTeacher(std::shared_ptr<const assets::TensorSource> source, core::ExecutionContext & execution);
    ~Zonos3SslTeacher();
    Zonos3SslTeacher(const Zonos3SslTeacher &) = delete;
    Zonos3SslTeacher & operator=(const Zonos3SslTeacher &) = delete;

    // mono_44k: 44.1 kHz samples, at most 60 s. Returns row-major [frames][1024], stride-padding frames included.
    std::vector<float> encode(const std::vector<float> & mono_44k, int & frames) const;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace engine::models::zonos3
