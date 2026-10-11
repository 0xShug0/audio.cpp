#include "engine/community_models/sanotts/neon_weights.h"
#include "engine/framework/core/backend.h"
#include "engine/framework/core/module.h"

#include <algorithm>
#include <future>
#include <stdexcept>

namespace engine::models::sanotts {

SanoTtsNeonWeights load_sanotts_neon_weights(const SanoTtsConfig &config, const SanoTtsWeightReader &reader,
                                             int threads) {
    const int hidden = static_cast<int>(config.acoustic_hidden);
    const int channels = static_cast<int>(config.dim);
    const int noise_channels = static_cast<int>(config.noise_channels);
    const int embedding_channels = hidden + 1 + noise_channels;
    const int mels = static_cast<int>(config.mels);
    const int ak = static_cast<int>(config.acoustic_kernel);
    const int ek = static_cast<int>(config.embed_kernel);
    if (hidden <= 0 || channels <= 0 || config.pw_hidden <= 0 || config.blocks <= 0 || mels <= 0 || ak <= 0 ||
        ak % 2 == 0 || ak > 33 || ek <= 0 || ek % 2 == 0 || ek > 33 || config.dw_kernel <= 0) {
        throw std::runtime_error("sanoTTS NEON weights: unsupported config");
    }
    auto read = [&](const std::string &name, size_t size) {
        auto values = reader(name);
        if (values.size() != size) {
            throw std::runtime_error("sanoTTS NEON weights: unexpected size for " + name);
        }
        return values;
    };
    SanoTtsNeonWeights result;
    result.projection = {read("acoustic.frame_input_proj.weight", static_cast<size_t>(hidden) * (hidden + 3)),
                         read("acoustic.frame_input_proj.bias", hidden)};
    for (int i = 0; i < config.acoustic_depth; ++i) {
        const std::string prefix = "acoustic.frame_blocks." + std::to_string(i);
        const size_t size = static_cast<size_t>(hidden) * hidden * ak;
        auto scale = reader(prefix + ".scale");
        if (scale.size() != 1 && scale.size() != static_cast<size_t>(hidden)) {
            throw std::runtime_error("sanoTTS NEON weights: unexpected scale size");
        }
        result.acoustic_blocks.push_back({read(prefix + ".net.0.weight", size), read(prefix + ".net.0.bias", hidden),
                                          read(prefix + ".net.2.weight", size), read(prefix + ".net.2.bias", hidden),
                                          std::move(scale)});
    }
    // acoustic.output (1x1, hidden -> mels) has no nonlinearity before decoder.embed, so fold it
    // in: embed(output(f)) = embed'(f) with one extra input row of ones (zero-padded like f) for
    // output's bias. noise_adapter takes the last rows, its bias joins embed's.
    {
        const auto wo = read("acoustic.output.weight", static_cast<size_t>(mels) * hidden);
        const auto bo = read("acoustic.output.bias", mels);
        const auto we = read("decoder.embed.weight", static_cast<size_t>(channels) * mels * ek);
        const auto be = read("decoder.embed.bias", channels);
        const auto wn = read("decoder.noise_adapter.weight", static_cast<size_t>(channels) * noise_channels * ek);
        const auto bn = read("decoder.noise_adapter.bias", channels);
        std::vector<float> w(static_cast<size_t>(channels) * embedding_channels * ek, 0.0F);
        std::vector<float> b(channels);
        const int h1 = hidden + 1; // folded rows + the bias row
        // Output channels are independent: split them across `threads` (same result for any count).
        auto fold_range = [&](int o_lo, int o_hi) {
            std::vector<double> acc(static_cast<size_t>(ek) * h1);
            for (int o = o_lo; o < o_hi; ++o) {
                // acc[k][c] = sum over m (in order) of embed[o][m][k] * output[m][c], c == hidden -> bias
                std::fill(acc.begin(), acc.end(), 0.0);
                for (int m = 0; m < mels; ++m) {
                    const float *wrow = wo.data() + static_cast<size_t>(m) * hidden;
                    for (int k = 0; k < ek; ++k) {
                        const double e = we[(static_cast<size_t>(o) * mels + m) * ek + k];
                        double *ak = acc.data() + static_cast<size_t>(k) * h1;
                        for (int c = 0; c < hidden; ++c) {
                            ak[c] += e * wrow[c];
                        }
                        ak[hidden] += e * bo[m];
                    }
                }
                for (int k = 0; k < ek; ++k) {
                    for (int c = 0; c < h1; ++c) {
                        w[(static_cast<size_t>(o) * embedding_channels + c) * ek + k] =
                            static_cast<float>(acc[static_cast<size_t>(k) * h1 + c]);
                    }
                    for (int c = 0; c < noise_channels; ++c) {
                        w[(static_cast<size_t>(o) * embedding_channels + h1 + c) * ek + k] =
                            wn[(static_cast<size_t>(o) * noise_channels + c) * ek + k];
                    }
                }
                b[o] = static_cast<float>(static_cast<double>(be[o]) + bn[o]);
            }
        };
        const int nt = std::clamp(threads, 1, channels);
        std::vector<std::future<void>> workers;
        for (int tid = 1; tid < nt; ++tid) {
            workers.push_back(
                std::async(std::launch::async, fold_range, channels * tid / nt, channels * (tid + 1) / nt));
        }
        fold_range(0, channels / nt);
        for (auto &worker : workers) {
            worker.get();
        }
        result.embedding = {std::move(w), std::move(b)};
    }

    result.input_norm = {read("decoder.norm.weight", channels), read("decoder.norm.bias", channels)};
    for (int i = 0; i < config.blocks; ++i) {
        const std::string prefix = "decoder.blocks." + std::to_string(i);
        neon::ConvNeXtWeights block;
        block.depthwise_weight = read(prefix + ".dwconv.weight", static_cast<size_t>(channels) * config.dw_kernel);
        block.depthwise_bias = read(prefix + ".dwconv.bias", channels);
        block.norm_weight = read(prefix + ".norm.weight", channels);
        block.norm_bias = read(prefix + ".norm.bias", channels);
        block.expand_weight = read(prefix + ".pwconv1.weight", static_cast<size_t>(config.pw_hidden) * channels);
        block.expand_bias = read(prefix + ".pwconv1.bias", config.pw_hidden);
        block.project_weight = read(prefix + ".pwconv2.weight", static_cast<size_t>(channels) * config.pw_hidden);
        block.project_bias = read(prefix + ".pwconv2.bias", channels);
        block.scale = read(prefix + ".gamma", channels);
        result.decoder_blocks.push_back(std::move(block));
    }
    result.final_norm = {read("decoder.final_norm.weight", channels), read("decoder.final_norm.bias", channels)};
    result.head = {read("decoder.head.weight", static_cast<size_t>(config.n_fft + 2) * channels),
                   read("decoder.head.bias", config.n_fft + 2)};
    return result;
}

SanoTtsNeonWeights load_sanotts_neon_weights(const SanoTtsConfig &config,
                                             const std::unordered_map<std::string, core::TensorValue> &tensors,
                                             int threads) {
    return load_sanotts_neon_weights(
        config,
        [&](const std::string &name) {
            const auto tensor = tensors.find(name);
            if (tensor == tensors.end()) {
                throw std::runtime_error("sanoTTS missing native weight: " + name);
            }
            return core::read_tensor_f32(tensor->second.tensor);
        },
        threads);
}

} // namespace engine::models::sanotts
