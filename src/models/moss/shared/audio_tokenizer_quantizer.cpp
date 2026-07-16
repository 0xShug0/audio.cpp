#include "engine/models/moss/shared/audio_tokenizer_quantizer.h"

#include "engine/framework/assets/tensor_source.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace engine::models::moss {
namespace {

// Rebuilds a weight-normalized 1x1 conv weight from its parametrization
// (original0 = magnitude g per output channel, original1 = direction v), the
// PyTorch weight_norm(dim=0) reconstruction weight = g * v / ||v||.
std::vector<float> reconstruct_weight_norm(
    const std::vector<float> & g,
    const std::vector<float> & v,
    int64_t out_channels,
    int64_t in_channels) {
    std::vector<float> weight(static_cast<size_t>(out_channels * in_channels));
#ifdef _OPENMP
#pragma omp parallel for if(out_channels * in_channels >= 4096)
#endif
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
    const assets::TensorSource & source,
    const std::string & prefix,
    int64_t out_channels,
    int64_t in_channels) {
    const auto g = source.require_f32(prefix + ".parametrizations.weight.original0");
    const auto v = source.require_f32(prefix + ".parametrizations.weight.original1");
    return reconstruct_weight_norm(g, v, out_channels, in_channels);
}

}  // namespace

MossAudioTokenizerQuantizer::MossAudioTokenizerQuantizer(
    const assets::TensorSource & source,
    int64_t num_quantizers,
    AudioTokenizerQuantizerConfig config)
    : codebook_size_(config.codebook_size),
      codebook_dim_(config.codebook_dim),
      rvq_dim_(config.rvq_dim),
      code_dim_(config.code_dim),
      num_quantizers_(num_quantizers) {
    if (num_quantizers_ <= 0) {
        throw std::runtime_error("MOSS codec dequantizer requires a positive quantizer count");
    }

    output_weight_ = load_wn_conv_weight(source, "quantizer.output_proj", code_dim_, rvq_dim_);
    output_bias_ = source.require_f32("quantizer.output_proj.bias");

    codebooks_.reserve(static_cast<size_t>(num_quantizers_));
    for (int64_t index = 0; index < num_quantizers_; ++index) {
        const std::string prefix = "quantizer.quantizers." + std::to_string(index);
        Codebook codebook;
        codebook.table = source.require_f32(prefix + ".codebook.weight");
        codebook.out_weight = load_wn_conv_weight(source, prefix + ".out_proj", rvq_dim_, codebook_dim_);
        codebook.out_bias = source.require_f32(prefix + ".out_proj.bias");
        std::vector<float> combined_bias(static_cast<size_t>(code_dim_));
        std::vector<float> combined_weight(static_cast<size_t>(code_dim_ * codebook_dim_));
#ifdef _OPENMP
#pragma omp parallel for if(code_dim_ >= 256)
#endif
        for (int64_t out = 0; out < code_dim_; ++out) {
            const float * output_row = &output_weight_[static_cast<size_t>(out * rvq_dim_)];
            float bias_sum = 0.0F;
            for (int64_t rvq = 0; rvq < rvq_dim_; ++rvq) {
                bias_sum += output_row[rvq] * codebook.out_bias[static_cast<size_t>(rvq)];
            }
            combined_bias[static_cast<size_t>(out)] = bias_sum;
            for (int64_t k = 0; k < codebook_dim_; ++k) {
                float sum = 0.0F;
                for (int64_t rvq = 0; rvq < rvq_dim_; ++rvq) {
                    sum += output_row[rvq] * codebook.out_weight[static_cast<size_t>(rvq * codebook_dim_ + k)];
                }
                combined_weight[static_cast<size_t>(out * codebook_dim_ + k)] = sum;
            }
        }
        codebook.latent_table.resize(static_cast<size_t>(codebook_size_ * code_dim_));
#ifdef _OPENMP
#pragma omp parallel for collapse(2) if(codebook_size_ * code_dim_ >= 4096)
#endif
        for (int64_t code = 0; code < codebook_size_; ++code) {
            for (int64_t out = 0; out < code_dim_; ++out) {
                const float * embedding = &codebook.table[static_cast<size_t>(code * codebook_dim_)];
                float sum = combined_bias[static_cast<size_t>(out)];
                const float * row = &combined_weight[static_cast<size_t>(out * codebook_dim_)];
                for (int64_t k = 0; k < codebook_dim_; ++k) {
                    sum += row[k] * embedding[k];
                }
                codebook.latent_table[static_cast<size_t>(code * code_dim_ + out)] = sum;
            }
        }
        codebook.in_weight = load_wn_conv_weight(source, prefix + ".in_proj", codebook_dim_, rvq_dim_);
        codebook.in_bias = source.require_f32(prefix + ".in_proj.bias");
        // Pre-normalize the codebook rows once (encode does L2-normalized nearest
        // search, matching the training LFQ; F.normalize uses eps=1e-12).
        codebook.table_normalized = codebook.table;
#ifdef _OPENMP
#pragma omp parallel for if(codebook_size_ * codebook_dim_ >= 4096)
#endif
        for (int64_t code = 0; code < codebook_size_; ++code) {
            float * row = &codebook.table_normalized[static_cast<size_t>(code * codebook_dim_)];
            double norm = 0.0;
            for (int64_t k = 0; k < codebook_dim_; ++k) {
                norm += static_cast<double>(row[k]) * static_cast<double>(row[k]);
            }
            const double scale = 1.0 / std::max(std::sqrt(norm), 1.0e-12);
            for (int64_t k = 0; k < codebook_dim_; ++k) {
                row[k] = static_cast<float>(row[k] * scale);
            }
        }
        codebooks_.push_back(std::move(codebook));
    }

    input_weight_ = load_wn_conv_weight(source, "quantizer.input_proj", rvq_dim_, code_dim_);
    input_bias_ = source.require_f32("quantizer.input_proj.bias");
}

std::vector<float> MossAudioTokenizerQuantizer::decode(const std::vector<std::vector<int32_t>> & codes) const {
    if (static_cast<int64_t>(codes.size()) != num_quantizers_) {
        throw std::runtime_error("MOSS codec dequantizer got the wrong number of codebooks");
    }
    const int64_t steps = codes.empty() ? 0 : static_cast<int64_t>(codes.front().size());
    if (steps <= 0) {
        throw std::runtime_error("MOSS codec dequantizer requires a non-empty code sequence");
    }
    for (int64_t step = 0; step < steps; ++step) {
        for (int64_t index = 0; index < num_quantizers_; ++index) {
            const int64_t code = codes[static_cast<size_t>(index)][static_cast<size_t>(step)];
            if (code < 0 || code >= codebook_size_) {
                throw std::runtime_error("MOSS codec code index out of range");
            }
        }
    }

    std::vector<float> latent(static_cast<size_t>(code_dim_ * steps));
#ifdef _OPENMP
#pragma omp parallel for if(steps * code_dim_ >= 4096)
#endif
    for (int64_t step = 0; step < steps; ++step) {
        for (int64_t out = 0; out < code_dim_; ++out) {
            float value = output_bias_[static_cast<size_t>(out)];
            for (int64_t index = 0; index < num_quantizers_; ++index) {
                const auto & codebook = codebooks_[static_cast<size_t>(index)];
                const int64_t code = codes[static_cast<size_t>(index)][static_cast<size_t>(step)];
                const float * decoded = &codebook.latent_table[static_cast<size_t>(code * code_dim_)];
                value += decoded[static_cast<size_t>(out)];
            }
            latent[static_cast<size_t>(out * steps + step)] = value;
        }
    }
    return latent;
}

std::vector<std::vector<int32_t>> MossAudioTokenizerQuantizer::encode(
    const std::vector<float> & hidden, int64_t frames) const {
    if (frames <= 0) {
        throw std::runtime_error("MOSS codec quantizer requires a non-empty encoder latent");
    }
    if (static_cast<int64_t>(hidden.size()) != frames * code_dim_) {
        throw std::runtime_error("MOSS codec quantizer got a mis-shaped encoder latent");
    }

    std::vector<std::vector<int32_t>> codes(
        static_cast<size_t>(num_quantizers_), std::vector<int32_t>(static_cast<size_t>(frames)));

    const auto encode_frame = [&](int64_t step, std::vector<double> & residual, std::vector<double> & encoding) {
        std::fill(residual.begin(), residual.end(), 0.0);
        std::fill(encoding.begin(), encoding.end(), 0.0);

        // input_proj: encoder latent [code_dim] -> rvq_dim (WNConv1d 1x1).
        const float * frame_hidden = &hidden[static_cast<size_t>(step * code_dim_)];
        for (int64_t out = 0; out < rvq_dim_; ++out) {
            double sum = input_bias_[static_cast<size_t>(out)];
            const float * row = &input_weight_[static_cast<size_t>(out * code_dim_)];
            for (int64_t k = 0; k < code_dim_; ++k) {
                sum += static_cast<double>(row[k]) * static_cast<double>(frame_hidden[k]);
            }
            residual[static_cast<size_t>(out)] = sum;
        }

        for (int64_t index = 0; index < num_quantizers_; ++index) {
            const auto & codebook = codebooks_[static_cast<size_t>(index)];

            // in_proj: residual [rvq_dim] -> codebook_dim, then L2-normalize.
            double enc_norm = 0.0;
            for (int64_t c = 0; c < codebook_dim_; ++c) {
                double sum = codebook.in_bias[static_cast<size_t>(c)];
                const float * row = &codebook.in_weight[static_cast<size_t>(c * rvq_dim_)];
                for (int64_t k = 0; k < rvq_dim_; ++k) {
                    sum += static_cast<double>(row[k]) * residual[static_cast<size_t>(k)];
                }
                encoding[static_cast<size_t>(c)] = sum;
                enc_norm += sum * sum;
            }
            const double enc_scale = 1.0 / std::max(std::sqrt(enc_norm), 1.0e-12);
            for (int64_t c = 0; c < codebook_dim_; ++c) {
                encoding[static_cast<size_t>(c)] *= enc_scale;
            }

            // Nearest code by cosine similarity (both sides L2-normalized), i.e.
            // argmax dot == argmin squared distance on the unit sphere.
            int32_t best_code = 0;
            double best_dot = -std::numeric_limits<double>::infinity();
            for (int64_t code = 0; code < codebook_size_; ++code) {
                const float * row = &codebook.table_normalized[static_cast<size_t>(code * codebook_dim_)];
                double dot = 0.0;
                for (int64_t c = 0; c < codebook_dim_; ++c) {
                    dot += static_cast<double>(row[c]) * encoding[static_cast<size_t>(c)];
                }
                if (dot > best_dot) {
                    best_dot = dot;
                    best_code = static_cast<int32_t>(code);
                }
            }
            codes[static_cast<size_t>(index)][static_cast<size_t>(step)] = best_code;

            // Subtract the residual contribution: out_proj(raw codebook row).
            const float * embedding = &codebook.table[static_cast<size_t>(best_code * codebook_dim_)];
            for (int64_t out = 0; out < rvq_dim_; ++out) {
                double sum = codebook.out_bias[static_cast<size_t>(out)];
                const float * row = &codebook.out_weight[static_cast<size_t>(out * codebook_dim_)];
                for (int64_t k = 0; k < codebook_dim_; ++k) {
                    sum += static_cast<double>(row[k]) * static_cast<double>(embedding[k]);
                }
                residual[static_cast<size_t>(out)] -= sum;
            }
        }
    };

#ifdef _OPENMP
    if (frames >= 8) {
#pragma omp parallel
        {
            std::vector<double> residual(static_cast<size_t>(rvq_dim_));
            std::vector<double> encoding(static_cast<size_t>(codebook_dim_));
#pragma omp for
            for (int64_t step = 0; step < frames; ++step) {
                encode_frame(step, residual, encoding);
            }
        }
    } else
#endif
    {
        std::vector<double> residual(static_cast<size_t>(rvq_dim_));
        std::vector<double> encoding(static_cast<size_t>(codebook_dim_));
        for (int64_t step = 0; step < frames; ++step) {
            encode_frame(step, residual, encoding);
        }
    }
    return codes;
}

}  // namespace engine::models::moss
