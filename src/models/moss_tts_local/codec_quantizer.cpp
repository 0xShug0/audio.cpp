#include "engine/models/moss_tts_local/codec_quantizer.h"

#include "engine/framework/assets/tensor_source.h"

#include <algorithm>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>

namespace engine::models::moss_tts_local {
namespace {

// Opens every safetensors shard in the codec directory and resolves tensors by
// name across them (the codec ships as model-0000N-of-00003.safetensors + an
// index; a name-scan avoids parsing the index for the handful of tensors the
// dequantizer needs).
class CodecShards {
public:
    explicit CodecShards(const std::filesystem::path & codec_dir) {
        for (const auto & entry : std::filesystem::directory_iterator(codec_dir)) {
            if (entry.is_regular_file() && entry.path().extension() == ".safetensors") {
                sources_.push_back(assets::open_tensor_source(entry.path()));
            }
        }
        if (sources_.empty()) {
            throw std::runtime_error("MOSS codec has no safetensors shards: " + codec_dir.string());
        }
    }

    std::vector<float> require_f32(const std::string & name) const {
        for (const auto & source : sources_) {
            if (source->has_tensor(name)) {
                return source->require_f32(name);
            }
        }
        throw std::runtime_error("MOSS codec tensor not found: " + name);
    }

private:
    std::vector<std::shared_ptr<const assets::TensorSource>> sources_;
};

// Rebuilds a weight-normalized 1x1 conv weight from its parametrization
// (original0 = magnitude g per output channel, original1 = direction v), the
// PyTorch weight_norm(dim=0) reconstruction weight = g * v / ||v||.
std::vector<float> reconstruct_weight_norm(
    const std::vector<float> & g,
    const std::vector<float> & v,
    int64_t out_channels,
    int64_t in_channels) {
    std::vector<float> weight(static_cast<size_t>(out_channels * in_channels));
    for (int64_t o = 0; o < out_channels; ++o) {
        double norm = 0.0;
        for (int64_t k = 0; k < in_channels; ++k) {
            const double value = v[static_cast<size_t>(o * in_channels + k)];
            norm += value * value;
        }
        const float scale = static_cast<float>(g[static_cast<size_t>(o)] / std::sqrt(norm));
        for (int64_t k = 0; k < in_channels; ++k) {
            weight[static_cast<size_t>(o * in_channels + k)] =
                v[static_cast<size_t>(o * in_channels + k)] * scale;
        }
    }
    return weight;
}

std::vector<float> load_wn_conv_weight(
    const CodecShards & shards,
    const std::string & prefix,
    int64_t out_channels,
    int64_t in_channels) {
    const auto g = shards.require_f32(prefix + ".parametrizations.weight.original0");
    const auto v = shards.require_f32(prefix + ".parametrizations.weight.original1");
    return reconstruct_weight_norm(g, v, out_channels, in_channels);
}

}  // namespace

MossCodecDequantizer::MossCodecDequantizer(const std::filesystem::path & codec_dir, int64_t num_quantizers)
    : codebook_size_(1024), codebook_dim_(8), rvq_dim_(512), code_dim_(768), num_quantizers_(num_quantizers) {
    if (num_quantizers_ <= 0) {
        throw std::runtime_error("MOSS codec dequantizer requires a positive quantizer count");
    }
    CodecShards shards(codec_dir);

    codebooks_.reserve(static_cast<size_t>(num_quantizers_));
    for (int64_t index = 0; index < num_quantizers_; ++index) {
        const std::string prefix = "quantizer.quantizers." + std::to_string(index);
        Codebook codebook;
        codebook.table = shards.require_f32(prefix + ".codebook.weight");
        codebook.out_weight = load_wn_conv_weight(shards, prefix + ".out_proj", rvq_dim_, codebook_dim_);
        codebook.out_bias = shards.require_f32(prefix + ".out_proj.bias");
        codebooks_.push_back(std::move(codebook));
    }

    output_weight_ = load_wn_conv_weight(shards, "quantizer.output_proj", code_dim_, rvq_dim_);
    output_bias_ = shards.require_f32("quantizer.output_proj.bias");
}

std::vector<float> MossCodecDequantizer::decode(const std::vector<std::vector<int32_t>> & codes) const {
    if (static_cast<int64_t>(codes.size()) != num_quantizers_) {
        throw std::runtime_error("MOSS codec dequantizer got the wrong number of codebooks");
    }
    const int64_t steps = codes.empty() ? 0 : static_cast<int64_t>(codes.front().size());
    if (steps <= 0) {
        throw std::runtime_error("MOSS codec dequantizer requires a non-empty code sequence");
    }

    std::vector<float> latent(static_cast<size_t>(code_dim_ * steps));
    std::vector<double> residual(static_cast<size_t>(rvq_dim_));
    for (int64_t step = 0; step < steps; ++step) {
        std::fill(residual.begin(), residual.end(), 0.0);
        for (int64_t index = 0; index < num_quantizers_; ++index) {
            const auto & codebook = codebooks_[static_cast<size_t>(index)];
            const int64_t code = codes[static_cast<size_t>(index)][static_cast<size_t>(step)];
            if (code < 0 || code >= codebook_size_) {
                throw std::runtime_error("MOSS codec code index out of range");
            }
            const float * embedding = &codebook.table[static_cast<size_t>(code * codebook_dim_)];
            for (int64_t out = 0; out < rvq_dim_; ++out) {
                double sum = codebook.out_bias[static_cast<size_t>(out)];
                const float * row = &codebook.out_weight[static_cast<size_t>(out * codebook_dim_)];
                for (int64_t k = 0; k < codebook_dim_; ++k) {
                    sum += static_cast<double>(row[k]) * static_cast<double>(embedding[k]);
                }
                residual[static_cast<size_t>(out)] += sum;
            }
        }
        for (int64_t out = 0; out < code_dim_; ++out) {
            double sum = output_bias_[static_cast<size_t>(out)];
            const float * row = &output_weight_[static_cast<size_t>(out * rvq_dim_)];
            for (int64_t k = 0; k < rvq_dim_; ++k) {
                sum += static_cast<double>(row[k]) * residual[static_cast<size_t>(k)];
            }
            latent[static_cast<size_t>(out * steps + step)] = static_cast<float>(sum);
        }
    }
    return latent;
}

}  // namespace engine::models::moss_tts_local
