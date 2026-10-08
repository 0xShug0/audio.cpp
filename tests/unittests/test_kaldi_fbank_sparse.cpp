#include "engine/framework/audio/kaldi_fbank.h"

#include <cmath>
#include <iostream>
#include <random>
#include <stdexcept>

int main() {
    try {
        std::mt19937 generator(7);
        std::uniform_real_distribution<float> random(-0.8f, 0.8f);
        for (auto window : {engine::audio::KaldiFbankWindowType::Povey, engine::audio::KaldiFbankWindowType::Hamming}) {
            for (int length : {0, 399, 400, 560, 16000, 16013}) {
                std::vector<float> audio(length);
                for (auto & sample : audio) sample = random(generator);
                engine::audio::KaldiFbankOptions config;
                config.window_type = window;
                config.upscale_samples = true;
                for (int stacking : {1, 7}) {
                    config.lfr_m = stacking;
                    config.lfr_n = stacking == 1 ? 1 : 6;
                    config.apply_cmvn = true;
                    config.cmvn_shift.assign(80 * stacking, 0.25f);
                    config.cmvn_scale.assign(80 * stacking, 0.75f);
                    config.sparse_filterbank = false;
                    auto dense = engine::audio::extract_kaldi_fbank(audio, config);
                    config.sparse_filterbank = true;
                    auto sparse = engine::audio::extract_kaldi_fbank(audio, config);
                    if (dense.frames != sparse.frames || dense.feature_dim != sparse.feature_dim || dense.values != sparse.values)
                        throw std::runtime_error("Sparse Kaldi fbank changed the dense result");
                }
            }
        }
        std::cout << "Sparse Kaldi fbank matches dense features exactly\n";
        return 0;
    } catch (const std::exception & error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
