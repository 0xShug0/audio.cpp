// Liquid's Q4_0 packages store token_embd as Q6_K, which ggml's CUDA get_rows
// cannot gather, so on CUDA the backbone looks tokens up in an F16 copy. This
// runs one such GGUF on CUDA and on the CPU, which gathers Q6_K directly, and
// requires the same logits. Exits 77 (skip) without a CUDA device.
#include "engine/community_models/lfm2_audio/backbone.h"
#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/core/backend.h"
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

constexpr int64_t kVocab = 40;

lfm2::Lfm2BackboneConfig config_for(const lfm2_audio_test::BackboneShape & shape) {
    lfm2::Lfm2BackboneConfig config;
    config.vocab_size = kVocab;
    config.hidden_size = shape.hidden;
    config.intermediate_size = shape.intermediate;
    config.num_attention_heads = shape.heads;
    config.head_dim = shape.head_dim();
    config.conv_kernel_size = shape.kernel;
    config.context_length = shape.context;
    config.kv_heads.assign(shape.kv_heads.begin(), shape.kv_heads.end());
    config.rms_norm_eps = shape.rms_eps;
    config.rope_theta = shape.rope_theta;
    return config;
}

std::vector<float> prefill_logits(
    const std::filesystem::path & gguf,
    const lfm2::Lfm2BackboneConfig & config,
    engine::core::BackendType backend,
    const lfm2::Lfm2Prompt & prompt,
    const lfm2::Lfm2AudioEmbeddings & audio) {
    engine::core::ExecutionContext execution(engine::core::BackendConfig{backend, 0, 2});
    lfm2::Lfm2BackboneRuntime runtime(engine::assets::open_tensor_source(gguf), config, execution);
    return runtime.generate(prompt, audio, {1, {}}).prefill_logits;
}

}  // namespace

int main() {
    engine::core::ensure_backends_loaded();
    const auto reg = ggml_backend_reg_by_name("CUDA");
    if (reg == nullptr || ggml_backend_reg_dev_count(reg) == 0) {
        return 77;
    }

    try {
        lfm2_audio_test::BackboneShape shape;
        shape.hidden = 256;  // Q6_K rows are 256 wide
        shape.intermediate = 512;
        shape.heads = 4;
        auto weights = lfm2_audio_test::backbone_tensors(shape, kVocab, lfm2_audio_test::random_fill(21, 0.075f));
        std::map<std::string, ggml_type> types = {{"token_embd.weight", GGML_TYPE_Q6_K}};
        for (const auto & [name, tensor] : weights) {
            if (tensor.shape.size() == 2 && name != "token_embd.weight" && name.find("shortconv.conv") == std::string::npos) {
                types[name] = GGML_TYPE_Q4_0;
            }
        }

        lfm2_audio_test::TextVocab vocab;
        for (int64_t i = 0; i < kVocab; ++i) {
            vocab.tokens.push_back("t" + std::to_string(i));
            vocab.types.push_back(1);
        }

        vocab.merges = {"t 1"};
        const auto dir = lfm2_audio_test::fresh_directory("audiocpp_lfm2_audio_cuda_test");
        const auto gguf = dir / "backbone.gguf";
        lfm2_audio_test::write_backbone(gguf, shape, vocab, weights, {"en"}, types);

        lfm2::Lfm2Prompt prompt;
        lfm2_audio_test::Random random(32);
        for (int i = 0; i < 12; ++i) {
            prompt.input_ids.push_back(static_cast<int32_t>((random.uniform(1.0f) + 1.0f) * 0.5f * (kVocab - 1)));
        }

        lfm2::Lfm2AudioEmbeddings audio;
        audio.tokens = 4;
        audio.hidden_size = shape.hidden;
        audio.values = lfm2_audio_test::Random(33).uniform(static_cast<size_t>(4 * shape.hidden), 1.0f);
        for (int32_t i = 0; i < 4; ++i) {
            prompt.audio_positions.push_back(2 + i);
            prompt.input_ids[static_cast<size_t>(2 + i)] = 0;
        }

        const auto config = config_for(shape);
        const auto cpu = prefill_logits(gguf, config, engine::core::BackendType::Cpu, prompt, audio);
        const auto cuda = prefill_logits(gguf, config, engine::core::BackendType::Cuda, prompt, audio);
        std::filesystem::remove_all(dir);

        double scale = 1.0;
        double worst = 0.0;
        for (size_t i = 0; i < cpu.size(); ++i) {
            scale = std::max(scale, std::fabs(static_cast<double>(cpu[i])));
            worst = std::max(worst, std::fabs(static_cast<double>(cuda[i]) - cpu[i]));
        }

        // Each backend quantizes the activations for its quantized matmuls its
        // own way, about 2% of the largest logit apart from exact math, so up
        // to 4% between them; a wrong or missing row would be off by order 100%.
        std::cout << "largest logit difference " << worst << " of " << scale << "\n";
        require(worst <= 6e-2 * scale, "CUDA logits differ from the CPU's");
        require(std::max_element(cpu.begin(), cpu.end()) - cpu.begin() == std::max_element(cuda.begin(), cuda.end()) - cuda.begin(),
            "CUDA and the CPU pick different first tokens");
        std::cout << "lfm2_audio_cuda_test: PASS\n";
        return 0;
    } catch (const std::exception & error) {
        std::cerr << "lfm2_audio_cuda_test: " << error.what() << '\n';
        return 1;
    }
}
