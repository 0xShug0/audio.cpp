// The encoder runtime keeps its graph and compute buffer between chunks. On a
// synthetic mmproj, a sequence of chunks of the same and of different lengths
// must give exactly what a fresh runtime gives for each chunk alone.
#include "engine/community_models/lfm2_audio/audio_encoder.h"
#include "engine/framework/assets/tensor_source.h"
#include "lfm2_audio_test_package.h"
#include "test_assert.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <map>
#include <string>
#include <vector>

namespace {

namespace lfm2 = engine::community_models::lfm2_audio;
using engine::test::require;
using engine::test::require_eq;

lfm2::Lfm2FastConformerEncoderConfig config_for(const lfm2_audio_test::EncoderShape & shape) {
    lfm2::Lfm2FastConformerEncoderConfig config;
    config.n_mels = shape.n_mels;
    config.hidden_size = shape.hidden;
    config.num_layers = shape.layers;
    config.num_heads = shape.heads;
    config.intermediate_size = shape.intermediate;
    config.conv_kernel_size = shape.kernel;
    config.subsampling_channels = shape.channels;
    config.adapter_hidden_size = shape.adapter_hidden;
    config.output_size = shape.output;
    return config;
}

lfm2::Lfm2AudioFeatures random_features(int64_t n_mels, int64_t frames, uint64_t seed) {
    return {n_mels, frames, lfm2_audio_test::Random(seed).uniform(static_cast<size_t>(n_mels * frames), 2.0f)};
}

class Encoder {
public:
    Encoder(const std::filesystem::path & mmproj, const lfm2::Lfm2FastConformerEncoderConfig & config, bool cpu_repack = true)
        : execution_(engine::core::BackendConfig{engine::core::BackendType::Cpu, 0, 2}),
          runtime_(engine::assets::open_tensor_source(mmproj), config, execution_, cpu_repack) {}

    lfm2::Lfm2AudioEmbeddings encode(const lfm2::Lfm2AudioFeatures & features) { return runtime_.encode(features); }

private:
    engine::core::ExecutionContext execution_;
    lfm2::Lfm2FastConformerEncoderRuntime runtime_;
};

// Lengths repeat, shrink and grow. 57 and 64 frames both give 8 steps, so a
// graph kept for one must not serve the other. With this shape, from about 700
// frames on the allocator would give the position table's memory to later
// tensors if the graph did not keep it, so the repeats there check that it
// does. 3001 frames is a 30 s chunk, the longest whose buffer is kept; 3002
// frames gets a buffer of its own, freed after it.
void test_reuse_matches_fresh_runs(const std::filesystem::path & mmproj, const lfm2::Lfm2FastConformerEncoderConfig & config) {
    const std::vector<int64_t> lengths = {120, 120, 57, 64, 1000, 1000, 9, 300, 57, 3001, 3001, 64, 3002, 3002, 64, 3001, 120};
    Encoder reused(mmproj, config);
    for (size_t i = 0; i < lengths.size(); ++i) {
        const auto features = random_features(config.n_mels, lengths[i], 100 + i);
        const auto expected = Encoder(mmproj, config).encode(features);
        const auto actual = reused.encode(features);
        const auto label = "chunk " + std::to_string(i) + " (" + std::to_string(lengths[i]) + " frames)";
        require_eq(actual.tokens, (lengths[i] + 7) / 8, label + " tokens");
        require_eq(actual.hidden_size, config.output_size, label + " hidden size");
        require(actual.values == expected.values, label + " differs from a fresh runtime");
    }
}

// The same features twice through one graph, with other input in between.
void test_same_input_same_output(const std::filesystem::path & mmproj, const lfm2::Lfm2FastConformerEncoderConfig & config) {
    Encoder encoder(mmproj, config);
    const auto features = random_features(config.n_mels, 1000, 7);
    const auto first = encoder.encode(features);
    (void)encoder.encode(random_features(config.n_mels, 1000, 8));
    require(encoder.encode(features).values == first.values, "the same features after another chunk of their length");
    (void)encoder.encode(random_features(config.n_mels, 31, 9));
    require(encoder.encode(features).values == first.values, "the same features after a chunk of another length");
}

// With Q8_0 or Q4_0 linears, lfm2_audio.cpu_repack multiplies them with
// ggml's repacked CPU kernels where those take them. Those sum in another
// order, and the activations they quantize then round differently here and
// there, so the output moves slightly; a misread weight would move it by order
// 100%. A repacked runtime reuses its graph as exactly as a plain one.
void test_quantized_cpu_repack(const std::filesystem::path & dir) {
    lfm2_audio_test::EncoderShape shape;  // widths in whole 32-value blocks
    shape.hidden = 32;
    shape.intermediate = 64;
    shape.adapter_hidden = 32;
    shape.output = 32;
    const auto tensors = lfm2_audio_test::encoder_tensors(shape, lfm2_audio_test::random_fill(23, 0.2f));
    const auto config = config_for(shape);
    for (const ggml_type type : {GGML_TYPE_Q8_0, GGML_TYPE_Q4_0}) {
        // The linears and linear_pos; not the convolutions or position biases.
        std::map<std::string, ggml_type> types;
        for (const auto & [name, tensor] : tensors) {
            if (tensor.shape.size() == 2 && name.find("conv_dw") == std::string::npos && name.find("pos_bias") == std::string::npos) {
                types[name] = type;
            }
        }

        const std::string label = ggml_type_name(type);
        const auto mmproj = dir / ("mmproj-" + label + ".gguf");
        lfm2_audio_test::write_mmproj(mmproj, shape, tensors, "lfm2a", types);
        Encoder plain(mmproj, config, false);
        Encoder repacked(mmproj, config, true);
        const auto features = random_features(config.n_mels, 1000, 31);
        const auto off = plain.encode(features).values;
        const auto on = repacked.encode(features).values;
        require_eq(on.size(), off.size(), label + " repacked output size");
        double scale = 0.0;
        double diff = 0.0;
        for (size_t i = 0; i < off.size(); ++i) {
            scale = std::max(scale, std::fabs(static_cast<double>(off[i])));
            diff = std::max(diff, std::fabs(static_cast<double>(on[i]) - off[i]));
        }

        std::cout << label << " encoder output, repacked against plain: max difference " << diff / scale << " of the largest\n";
        require(scale > 0.0 && diff <= 1e-2 * scale, label + " repacked output differs by " + std::to_string(diff / scale) + " of the largest");
        (void)repacked.encode(random_features(config.n_mels, 300, 32));
        require(repacked.encode(features).values == on, label + " repacked, the same features after a chunk of another length");
    }
}

}  // namespace

int main() {
    try {
        const auto dir = lfm2_audio_test::fresh_directory("audiocpp_lfm2_audio_encoder_test");
        lfm2_audio_test::EncoderShape shape;
        shape.layers = 2;
        const auto mmproj = dir / "mmproj.gguf";
        lfm2_audio_test::write_mmproj(mmproj, shape, lfm2_audio_test::encoder_tensors(shape, lfm2_audio_test::random_fill(17, 0.2f)));
        const auto config = config_for(shape);

        test_reuse_matches_fresh_runs(mmproj, config);
        test_same_input_same_output(mmproj, config);
        test_quantized_cpu_repack(dir);
        std::filesystem::remove_all(dir);
        std::cout << "lfm2_audio_encoder_test: PASS\n";
        return 0;
    } catch (const std::exception & error) {
        std::cerr << "lfm2_audio_encoder_test: " << error.what() << '\n';
        return 1;
    }
}
