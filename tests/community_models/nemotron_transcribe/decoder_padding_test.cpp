#include "engine/community_models/nemotron_transcribe/decoder.h"

#include <gguf.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iostream>
#include <stdexcept>

namespace {
using namespace engine;
using namespace engine::models::nemotron_transcribe;

void require(bool condition, const std::string & message) {
    if (!condition) throw std::runtime_error(message);
}

struct FixtureFiles {
    std::vector<std::filesystem::path> files;
    ~FixtureFiles() {
        for (const auto & file : files) { std::error_code error; std::filesystem::remove(file, error); }
    }
    std::filesystem::path add() {
        const auto tick = std::chrono::steady_clock::now().time_since_epoch().count();
        auto path = std::filesystem::temp_directory_path() /
            ("nemotron_decoder_padding_" + std::to_string(tick) + "_" + std::to_string(files.size()) + ".gguf");
        files.push_back(path);
        return path;
    }
};

DecoderConfig config() {
    DecoderConfig c;
    c.hidden_size = 128;
    c.vocab_size = 8;
    c.block_pattern = "E";
    c.experts = 4;
    c.experts_per_token = 2;
    c.expert_width = 64;
    c.shared_expert_width = 128;
    c.max_context = 32;
    c.prefill_batch_size = 3;
    return c;
}

// Build source F32 values, optionally append zero input columns, then quantize
// the routed banks directly from those values. No checkpoint or requantization.
void write_fixture(const std::filesystem::path & path, const DecoderConfig & c, ggml_type routed_type) {
    std::unique_ptr<ggml_context, decltype(&ggml_free)> ctx(ggml_init({1024 * 1024, nullptr, true}), ggml_free);
    std::unique_ptr<gguf_context, decltype(&gguf_free)> out(gguf_init_empty(), gguf_free);
    require(ctx && out, "fixture context allocation");
    if (c.expert_row_alignment != 1)
        gguf_set_val_u32(out.get(), "nemotron_transcribe.expert_row_alignment", static_cast<uint32_t>(c.expert_row_alignment));
    std::vector<std::vector<std::byte>> buffers;
    buffers.reserve(16);
    auto add = [&](const std::string & name, std::vector<int64_t> shape, std::vector<float> values, ggml_type type = GGML_TYPE_F32) {
        int64_t ne[4] = {1, 1, 1, 1};
        for (size_t i = 0; i < shape.size(); ++i) ne[i] = shape[shape.size() - 1 - i];
        auto * tensor = ggml_new_tensor(ctx.get(), type, static_cast<int>(shape.size()), ne);
        ggml_set_name(tensor, name.c_str());
        buffers.emplace_back(ggml_nbytes(tensor));
        auto & bytes = buffers.back();
        if (type == GGML_TYPE_F32) std::memcpy(bytes.data(), values.data(), bytes.size());
        else {
            const auto written = ggml_quantize_chunk(type, values.data(), bytes.data(), 0,
                static_cast<int64_t>(values.size()) / ne[0], ne[0], nullptr);
            require(written == bytes.size(), "fixture quantization byte count");
        }
        tensor->data = bytes.data();
        gguf_add_tensor(out.get(), tensor);
    };
    const auto h = c.hidden_size, f = c.expert_width, e = c.experts, shared = c.shared_expert_width;
    std::vector<float> embedding(c.vocab_size * h), head(c.vocab_size * h);
    for (int64_t token = 0; token < c.vocab_size; ++token) {
        for (int64_t j = 0; j < h; ++j) {
            embedding[token * h + j] = 0.1F + 0.02F * std::sin(static_cast<float>(token * h + j) * 0.19F);
            head[token * h + j] = (token == 0 ? 1.0F : -0.1F * static_cast<float>(token)) / static_cast<float>(h);
        }
    }
    add("token_embd.weight", {c.vocab_size, h}, embedding);
    add("output.weight", {c.vocab_size, h}, head);
    add("output_norm.weight", {h}, std::vector<float>(h, 1.0F));
    add("blk.0.attn_norm.weight", {h}, std::vector<float>(h, 1.0F));
    std::vector<float> router(e * h), bias(e);
    for (size_t i = 0; i < router.size(); ++i) router[i] = 0.02F * std::sin(static_cast<float>(i) * 0.33F);
    for (int64_t i = 0; i < e; ++i) bias[i] = 0.03F * static_cast<float>(i);
    add("blk.0.ffn_gate_inp.weight", {e, h}, router);
    add("blk.0.exp_probs_b.bias", {e}, bias);
    const auto ph = ((h + c.expert_row_alignment - 1) / c.expert_row_alignment) * c.expert_row_alignment;
    const auto pf = ((f + c.expert_row_alignment - 1) / c.expert_row_alignment) * c.expert_row_alignment;
    std::vector<float> up(e * f * ph), down(e * h * pf);
    for (int64_t expert = 0; expert < e; ++expert) {
        for (int64_t row = 0; row < f; ++row)
            for (int64_t col = 0; col < h; ++col)
                up[(expert * f + row) * ph + col] = 0.04F * std::sin(static_cast<float>((expert * f + row) * h + col) * 0.11F);
        for (int64_t row = 0; row < h; ++row)
            for (int64_t col = 0; col < f; ++col)
                down[(expert * h + row) * pf + col] = 0.04F * std::cos(static_cast<float>((expert * h + row) * f + col) * 0.13F);
    }
    add("blk.0.ffn_up_exps.weight", {e, f, ph}, up, routed_type);
    add("blk.0.ffn_down_exps.weight", {e, h, pf}, down, routed_type);
    std::vector<float> shared_up(shared * h), shared_down(h * shared);
    for (size_t i = 0; i < shared_up.size(); ++i) shared_up[i] = 0.01F * std::sin(static_cast<float>(i) * 0.17F);
    for (size_t i = 0; i < shared_down.size(); ++i) shared_down[i] = 0.01F * std::cos(static_cast<float>(i) * 0.23F);
    add("blk.0.ffn_up_shexp.weight", {shared, h}, shared_up);
    add("blk.0.ffn_down_shexp.weight", {h, shared}, shared_down);
    require(gguf_write_to_file(out.get(), path.string().c_str(), false), "write fixture GGUF");
}

float error(const std::vector<float> & a, const std::vector<float> & b) {
    require(a.size() == b.size(), "logit sizes");
    float maximum = 0;
    for (size_t i = 0; i < a.size(); ++i) maximum = std::max(maximum, std::abs(a[i] - b[i]));
    return maximum;
}

}  // namespace

int main() {
    try {
        FixtureFiles files;
        auto native = config();
        require(native.expert_row_alignment == 1, "native alignment default changed");
        auto padded = native;
        padded.expert_row_alignment = 256;
        const auto unpadded_path = files.add(), padded_path = files.add();
        write_fixture(unpadded_path, native, GGML_TYPE_F32);
        write_fixture(padded_path, padded, GGML_TYPE_F32);
        auto unpadded_source = assets::open_tensor_source(unpadded_path);
        auto padded_source = assets::open_tensor_source(padded_path);
        core::ExecutionContext execution({core::BackendType::Cpu, 0, 2});
        GenerationOptions options;
        options.max_new_tokens = 3;
        options.stop_token_ids.clear();
        options.return_last_logits = true;
        const std::vector<float> audio(native.hidden_size, 0.12F);
        NemotronHDecoderRuntime baseline(unpadded_source, execution, native);
        NemotronHDecoderRuntime zero_padded(padded_source, execution, padded);
        const auto expected = baseline.generate({1, 2}, audio, 1, {3}, options);
        const auto actual = zero_padded.generate({1, 2}, audio, 1, {3}, options);
        require(expected.token_ids == actual.token_ids && error(expected.last_logits, actual.last_logits) < 2e-6F,
            "F32 input-column padding changed decoder output");
        const auto repeat = zero_padded.generate({1, 2}, audio, 1, {3}, options);
        require(repeat.last_logits == actual.last_logits, "padded decoder repeated request changed output");
        for (auto type : {GGML_TYPE_Q4_K, GGML_TYPE_Q3_K, GGML_TYPE_Q2_K}) {
            const auto path = files.add();
            write_fixture(path, padded, type);
            auto source = assets::open_tensor_source(path);
            require(assets::ggml_type_for_tensor_dtype(source->require_metadata("blk.0.ffn_up_exps.weight").dtype) == type,
                "routed bank does not have requested quantized type");
            NemotronHDecoderRuntime quantized(source, execution, padded);
            NemotronHDecoderRuntime dequantized(source, execution, padded, assets::TensorStorageType::F32);
            const auto quant_result = quantized.generate({1, 2}, audio, 1, {3}, options);
            const auto f32_result = dequantized.generate({1, 2}, audio, 1, {3}, options);
            require(quant_result.token_ids == f32_result.token_ids, "quantized padded decode token parity");
            require(error(quant_result.last_logits, f32_result.last_logits) < 5e-3F, "quantized padded decode numerical parity");
            require(std::all_of(quant_result.last_logits.begin(), quant_result.last_logits.end(),
                [](float value) { return std::isfinite(value); }), "non-finite quantized padded logits");
            std::cout << ggml_type_name(type) << " max_abs=" << error(quant_result.last_logits, f32_result.last_logits) << '\n';
        }
        for (const auto & mismatch : {native, padded}) {
            bool rejected = false;
            try {
                NemotronHDecoderRuntime wrong(mismatch.expert_row_alignment == 1 ? padded_source : unpadded_source,
                    execution, mismatch);
            } catch (const std::runtime_error &) { rejected = true; }
            require(rejected, "padding metadata/config mismatch must fail closed");
        }
        for (auto alignment : {0, 32, 257}) {
            auto invalid = native;
            invalid.expert_row_alignment = alignment;
            bool rejected = false;
            try { NemotronHDecoderRuntime wrong(unpadded_source, execution, invalid); }
            catch (const std::runtime_error &) { rejected = true; }
            require(rejected, "unsupported expert alignment accepted");
        }
    } catch (const std::exception & exception) {
        std::cerr << "Nemotron decoder padding test failed: " << exception.what() << '\n';
        return 1;
    }
    std::cout << "Nemotron decoder padding test passed\n";
    return 0;
}
