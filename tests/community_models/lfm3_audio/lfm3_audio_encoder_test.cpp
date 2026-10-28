// The LFM3-Audio ASR encoder (downsampler, LFM2 stack, projector) against a
// host fp64 reference on the synthetic package: odd and even frame counts,
// the c * 16 + f flatten order and graph reuse.
//
//   lfm3_audio_encoder_test [--backend cpu|cuda|metal|best] [--threads N]
//
// A requested backend that is not available exits 77.
#include "lfm3_audio_reference.h"
#include "lfm3_audio_test_package.h"
#include "test_assert.h"

#include "engine/community_models/lfm3_audio/assets.h"
#include "engine/community_models/lfm3_audio/encoder.h"
#include "engine/community_models/lfm3_audio/features.h"
#include "engine/framework/core/execution_context.h"

#include <cmath>
#include <iostream>
#include <string>
#include <vector>

namespace {

namespace lfm3 = engine::community_models::lfm3_audio;
namespace core = engine::core;
namespace ref = lfm3_audio_test;
using engine::test::require;
using engine::test::require_eq;
using ref::Values;

// Synthetic native mode: BF16 weights and activations rounded to BF16 by the CPU matmul.
constexpr double kSyntheticNativeTolerance = 3e-2;

// PyTorch Conv2d on x [cin][h][w] with weight [cout][cin / groups][k][k] (groups is 1 or cin) and bias.
Values conv2d(const Values & x, int64_t cin, int64_t h, int64_t w, const Values & weight, const Values & bias, int64_t cout, int64_t groups,
              int64_t k, int64_t stride, int64_t pad, int64_t * oh, int64_t * ow) {
    *oh = (h + 2 * pad - k) / stride + 1;
    *ow = (w + 2 * pad - k) / stride + 1;
    const int64_t per_group = cin / groups;
    Values y(static_cast<size_t>(cout * *oh * *ow));
    for (int64_t o = 0; o < cout; ++o) {
        const int64_t first = o / (cout / groups) * per_group;
        for (int64_t r = 0; r < *oh; ++r) {
            for (int64_t c = 0; c < *ow; ++c) {
                double sum = bias[static_cast<size_t>(o)];
                for (int64_t i = 0; i < per_group; ++i) {
                    for (int64_t a = 0; a < k; ++a) {
                        for (int64_t b = 0; b < k; ++b) {
                            const int64_t yy = r * stride - pad + a;
                            const int64_t xx = c * stride - pad + b;
                            if (yy < 0 || yy >= h || xx < 0 || xx >= w) continue;
                            sum += weight[static_cast<size_t>(((o * per_group + i) * k + a) * k + b)] *
                                   x[static_cast<size_t>(((first + i) * h + yy) * w + xx)];
                        }
                    }
                }
                y[static_cast<size_t>((o * *oh + r) * *ow + c)] = sum;
            }
        }
    }
    return y;
}

void silu(Values & x) {
    for (auto & v : x) v = v / (1.0 + std::exp(-v));
}

struct Reference {
    Values downsample, encoder, projector;
};

// mel is [n_mels][frames].
Reference reference_encoder(const engine::assets::TensorSource & source, const lfm3::Lfm3EncoderConfig & config, const std::vector<float> & mel,
                            int64_t frames) {
    auto read = [&](const std::string & name) { return ref::to_values(source.require_f32(name)); };
    const int64_t c = config.conv_channels;
    const std::string p = "encoder.downsample.";
    int64_t h = 0, w = 0, h2 = 0, w2 = 0;
    auto x = conv2d(ref::to_values(mel), 1, config.n_mels, frames, read(p + "conv_in.weight"), read(p + "conv_in.bias"), c, 1, 3, 2, 1, &h, &w);
    silu(x);
    for (const std::string stage : {"mid", "end"}) {
        x = conv2d(x, c, h, w, read(p + "conv_" + stage + ".weight"), read(p + "conv_" + stage + ".bias"), c, c, 3, 2, 1, &h2, &w2);
        x = conv2d(x, c, h2, w2, read(p + "lin_" + stage + ".weight"), read(p + "lin_" + stage + ".bias"), c, 1, 1, 1, 0, &h, &w);
        silu(x);
    }

    // flatten(1, 2) then a 1x1 Conv1d: token n reads feature c * 16 + f.
    const int64_t features = c * h;
    Values rows(static_cast<size_t>(w * features));
    for (int64_t n = 0; n < w; ++n) {
        for (int64_t i = 0; i < features; ++i) rows[static_cast<size_t>(n * features + i)] = x[static_cast<size_t>(i * w + n)];
    }

    const int64_t d = config.stack.hidden;
    const int64_t o = config.output_size;
    Reference out;
    out.downsample = ref::linear_bias(rows, read(p + "lin_out.weight"), read(p + "lin_out.bias"), features, d);
    out.encoder =
        ref::stack(config.stack, ref::read_ref_stack(source, "encoder.lfm.lfm.", config.stack), out.downsample, ref::positions_from(0, w));

    // LayerNorm (eps 1e-5) -> Linear -> exact GELU -> Linear.
    const auto gamma = read("mlp.model.0.weight");
    const auto beta = read("mlp.model.0.bias");
    Values y = out.encoder;
    for (int64_t n = 0; n < w; ++n) {
        double * row = y.data() + n * d;
        double mean = 0.0, var = 0.0;
        for (int64_t i = 0; i < d; ++i) mean += row[i];
        mean /= static_cast<double>(d);
        for (int64_t i = 0; i < d; ++i) var += (row[i] - mean) * (row[i] - mean);
        var /= static_cast<double>(d);
        for (int64_t i = 0; i < d; ++i) row[i] = (row[i] - mean) / std::sqrt(var + 1e-5) * gamma[static_cast<size_t>(i)] + beta[static_cast<size_t>(i)];
    }
    y = ref::linear_bias(y, read("mlp.model.1.weight"), read("mlp.model.1.bias"), d, o);
    for (auto & v : y) v = 0.5 * v * (1.0 + std::erf(v / std::sqrt(2.0)));
    out.projector = ref::linear_bias(y, read("mlp.model.3.weight"), read("mlp.model.3.bias"), o, o);
    return out;
}

lfm3::Lfm3Features random_features(int64_t frames, uint64_t seed) {
    lfm3_audio_test::Fill fill = lfm3_audio_test::random_fill(seed, 1.0f);
    return {fill("mel", static_cast<size_t>(128 * frames)), frames};
}

void test_synthetic(core::ExecutionContext & execution) {
    const auto package = lfm3_audio_test::write_package("lfm3_audio_encoder_test");
    const auto assets = lfm3::load_lfm3_audio_assets(package.root);
    const auto & config = assets->encoder;

    for (const auto & options : {lfm3::Lfm3WeightOptions{lfm3::Lfm3WeightType::F32, true}, lfm3::Lfm3WeightOptions{lfm3::Lfm3WeightType::Native, true}}) {
        const bool f32 = lfm3::lfm3_reference_precision(options);
        const double tolerance = f32 ? ref::reference_tolerance(execution) : kSyntheticNativeTolerance;
        lfm3::Lfm3AudioEncoderRuntime encoder(assets->model, config, execution, options);
        // 37: odd frames, 5 tokens (> window 4); 40: even; 8: one token; 300: 38 tokens, far past the window; 1: one frame.
        for (const int64_t frames : {37, 40, 8, 300, 1}) {
            const auto features = random_features(frames, static_cast<uint64_t>(frames));
            lfm3::Lfm3EncoderTaps taps;
            const auto out = encoder.encode(features, &taps);
            require_eq(out.tokens, lfm3::lfm3_audio_tokens(frames), "tokens");
            require_eq(out.hidden, config.output_size, "hidden");
            const auto expected = reference_encoder(*assets->model, config, features.values, frames);
            const std::string label = std::string(f32 ? "f32" : "native") + " T=" + std::to_string(frames);
            ref::report(label + " downsample", ref::rel_error(taps.downsample, expected.downsample), tolerance);
            ref::report(label + " encoder", ref::rel_error(taps.encoder, expected.encoder), tolerance);
            ref::report(label + " projector", ref::rel_error(out.values, expected.projector), tolerance);
        }
        require_eq(encoder.graphs_built(), int64_t{5}, "one graph per length");

        // The last length again reuses its graph.
        const auto features = random_features(1, 1);
        const auto again = encoder.encode(features);
        require_eq(encoder.graphs_built(), int64_t{5}, "a same-length call builds no graph");
        lfm3::Lfm3EncoderTaps taps;
        require(encoder.encode(features, &taps).values == again.values, "a reused graph gives the same values");
        (void)encoder.encode(random_features(37, 37));
        require_eq(encoder.graphs_built(), int64_t{6}, "a new length builds a graph");
    }

    lfm3::Lfm3AudioEncoderRuntime encoder(assets->model, config, execution, {});
    ref::require_throws([&] { (void)encoder.encode({std::vector<float>(127 * 4), 4}); }, "features of the wrong size");
}

}  // namespace

int main(int argc, char ** argv) {
    int status = 0;
    const auto execution = ref::open_backend(argc, argv, status);
    if (!execution) {
        return status;
    }

    try {
        test_synthetic(*execution);
        std::cout << "lfm3_audio_encoder_test: PASS\n";
        return 0;
    } catch (const std::exception & error) {
        std::cerr << "lfm3_audio_encoder_test: FAIL: " << error.what() << '\n';
        return 1;
    }
}
