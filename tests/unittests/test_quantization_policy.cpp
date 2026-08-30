#include "engine/framework/assets/tensor_source.h"

#include "engine/framework/io/safetensors.h"

#include "test_assert.h"

#include <cstddef>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

namespace {

using engine::assets::GgufConversionOptions;
using engine::assets::TensorStorageType;
using engine::test::require;
using engine::test::require_eq;

// Every test tensor is [4, 64]: rank 2, last dimension divisible by the Q8_0 block
// size of 32, so the packer's shape rule alone would quantize all of them. What each
// one is stored as is therefore decided by the name policy and nothing else.
constexpr int64_t kRows = 4;
constexpr int64_t kCols = 64;

// ggml_row_size(Q8_0, 64) = 2 blocks x sizeof(block_q8_0) = 2 x 34 = 68 bytes per row.
constexpr size_t kQ8Bytes = 4 * 68;
constexpr size_t kF16Bytes = 4 * 64 * 2;
constexpr size_t kF32Bytes = 4 * 64 * 4;

std::vector<unsigned char> f32_payload(int64_t elements) {
    std::vector<unsigned char> bytes(static_cast<size_t>(elements) * sizeof(float));
    for (int64_t index = 0; index < elements; ++index) {
        const float value = static_cast<float>(index % 17) * 0.25F - 2.0F;
        std::memcpy(bytes.data() + static_cast<size_t>(index) * sizeof(float), &value, sizeof(float));
    }
    return bytes;
}

engine::io::SafeTensorWriteEntry matrix(const std::string & name) {
    return {name, "F32", {kRows, kCols}, f32_payload(kRows * kCols)};
}

engine::io::SafeTensorWriteEntry vector_1d(const std::string & name) {
    return {name, "F32", {kCols}, f32_payload(kCols)};
}

engine::io::SafeTensorWriteEntry conv_kernel(const std::string & name) {
    return {name, "F32", {4, 4, 7}, f32_payload(4 * 4 * 7)};
}

// --- names ------------------------------------------------------------------
// Ordinary mid-stack weights. These MUST stay quantized: they have the rest of the
// network downstream to absorb the error, and protecting them would grow every
// package the project ships.
constexpr const char * kFfnWeight = "model.layers.0.mlp.down_proj.weight";
constexpr const char * kAttentionWeight = "model.layers.0.self_attn.q_proj.weight";
// A transformer stack that merely has "decoder" in its path. Deliberately NOT
// protected -- the RedAE decoder's own iSTFT head is what needs protecting, and a
// blanket "decoder" exclusion would move a whole 24-layer Qwen stack to F16.
constexpr const char * kCodecDecoderBackbone = "redae.decoder.qwen2.layers.0.mlp.down_proj.weight";
// Analysis side of an audio tokenizer. Its error is absorbed by the decoder that
// follows it, so it stays quantized while the matching decoder tensor does not.
constexpr const char * kTokenizerEncoder = "model.acoustic_tokenizer.encoder.stages.0.ffn.linear1.weight";

// Lookup tables. Pre-existing behaviour, demoted to F16 rather than quantized.
constexpr const char * kEmbedding = "model.embed_tokens.weight";
constexpr const char * kCodebook = "quantizer.quantizers.0.codebook.weight";

// Audible tensors, newly protected.
constexpr const char * kIstftHead = "redae.decoder.istft_head.out.weight";
constexpr const char * kVocoderLinear = "vocoder.resblocks.0.linear.weight";
constexpr const char * kNormProjection = "flow.transformer.layers.0.attention_norm.project_layer.weight";
constexpr const char * kAdaLnModulation = "flow.final_layer.adaLN_modulation.1.weight";
constexpr const char * kLmHead = "lm_head.weight";
constexpr const char * kMelHead = "gpt.mel_head.weight";
constexpr const char * kProjOut = "proj_out.weight";
constexpr const char * kTokenizerDecoder = "model.acoustic_tokenizer.decoder.stages.0.ffn.linear1.weight";

// Shapes the policy must not touch.
constexpr const char * kLayerNorm = "model.layers.0.input_layernorm.weight";
constexpr const char * kBias = "model.layers.0.mlp.down_proj.bias";
constexpr const char * kConvKernel = "vocoder.conv_pre.weight";

std::filesystem::path scratch_root() {
    return std::filesystem::temp_directory_path() / "audiocpp_quantization_policy_test";
}

std::filesystem::path write_source(const std::filesystem::path & root) {
    std::filesystem::create_directories(root);
    const std::vector<engine::io::SafeTensorWriteEntry> entries{
        matrix(kFfnWeight),
        matrix(kAttentionWeight),
        matrix(kCodecDecoderBackbone),
        matrix(kTokenizerEncoder),
        matrix(kEmbedding),
        matrix(kCodebook),
        matrix(kIstftHead),
        matrix(kVocoderLinear),
        matrix(kNormProjection),
        matrix(kAdaLnModulation),
        matrix(kLmHead),
        matrix(kMelHead),
        matrix(kProjOut),
        matrix(kTokenizerDecoder),
        vector_1d(kLayerNorm),
        vector_1d(kBias),
        conv_kernel(kConvKernel),
    };
    const auto path = root / "model.safetensors";
    engine::io::write_safetensors_file(path, entries);
    return path;
}

// Packs the fixture at Q8_0 with `options` and returns the result, opened for reading.
std::shared_ptr<const engine::assets::TensorSource> pack(
    const std::filesystem::path & root,
    const std::filesystem::path & source,
    const std::string & output_name,
    GgufConversionOptions options) {
    const auto output = root / output_name;
    engine::assets::convert_tensor_sources_to_gguf(
        {{source, ""}},
        output,
        TensorStorageType::Q8_0,
        /*overwrite=*/true,
        /*embed_sidecars=*/false,
        root,
        {},
        std::nullopt,
        std::move(options));
    return engine::assets::open_tensor_source(output);
}

void require_stored_as(
    const engine::assets::TensorSource & packed,
    const char * name,
    const std::string & dtype,
    size_t bytes) {
    require_eq(packed.require_metadata(name).dtype, dtype, std::string("dtype of ") + name);
    require_eq(packed.require_tensor_data(name).bytes.size(), bytes, std::string("byte size of ") + name);
}

// The name predicate on its own, with no conversion involved. This is the surface a
// contributor reads and adds to, so it is asserted directly as well as through the
// packer.
void test_audible_tensor_reasons() {
    require(!engine::assets::gguf_audible_tensor_rules().empty(), "the exclusion list is not empty");

    require(!engine::assets::gguf_audible_tensor_reason(kIstftHead).empty(), "istft head is audible");
    require(!engine::assets::gguf_audible_tensor_reason(kVocoderLinear).empty(), "vocoder linear is audible");
    require(!engine::assets::gguf_audible_tensor_reason(kNormProjection).empty(), "norm projection is audible");
    require(!engine::assets::gguf_audible_tensor_reason(kAdaLnModulation).empty(), "adaLN modulation is audible");
    require(!engine::assets::gguf_audible_tensor_reason(kLmHead).empty(), "lm head is audible");
    require(!engine::assets::gguf_audible_tensor_reason(kMelHead).empty(), "mel head is audible");
    require(!engine::assets::gguf_audible_tensor_reason(kProjOut).empty(), "proj_out is audible");
    require(!engine::assets::gguf_audible_tensor_reason(kTokenizerDecoder).empty(), "tokenizer decoder is audible");

    require(engine::assets::gguf_audible_tensor_reason(kFfnWeight).empty(), "an FFN weight is not audible");
    require(engine::assets::gguf_audible_tensor_reason(kAttentionWeight).empty(), "an attention weight is not audible");
    require(
        engine::assets::gguf_audible_tensor_reason(kCodecDecoderBackbone).empty(),
        "a decoder-side transformer stack is not audible");
    require(
        engine::assets::gguf_audible_tensor_reason(kTokenizerEncoder).empty(),
        "the analysis side of a tokenizer is not audible");

    // Matching is case-insensitive and covers the whole name, not just a prefix.
    require(
        !engine::assets::gguf_audible_tensor_reason("MODEL.DECODER.ISTFT_HEAD.OUT.WEIGHT").empty(),
        "matching is case-insensitive");
    // "shift" contains "hift"; the HiFT rules are anchored so it must not match.
    require(
        engine::assets::gguf_audible_tensor_reason("dit.audio_scale_shift_table.weight").empty(),
        "a scale-shift table is not mistaken for a HiFT vocoder");
}

// The behaviour that already shipped. If any of this changes, every published GGUF
// stops matching what the converter would now produce.
void test_existing_behaviour_is_unchanged(const engine::assets::TensorSource & packed) {
    require_stored_as(packed, kFfnWeight, "q8_0", kQ8Bytes);
    require_stored_as(packed, kAttentionWeight, "q8_0", kQ8Bytes);
    require_stored_as(packed, kCodecDecoderBackbone, "q8_0", kQ8Bytes);
    require_stored_as(packed, kTokenizerEncoder, "q8_0", kQ8Bytes);

    // Lookup tables were, and remain, demoted to F16 rather than quantized.
    require_stored_as(packed, kEmbedding, "f16", kF16Bytes);
    require_stored_as(packed, kCodebook, "f16", kF16Bytes);

    // The shape rule keeps rank-1 and rank-3 tensors at their source dtype. The new
    // name policy must not pull them down to F16 -- `vocoder.conv_pre.weight` matches
    // a vocoder rule by name and must still come out F32.
    require_stored_as(packed, kLayerNorm, "f32", kCols * 4);
    require_stored_as(packed, kBias, "f32", kCols * 4);
    require_stored_as(packed, kConvKernel, "f32", 4 * 4 * 7 * 4);
}

void test_audible_tensors_are_not_quantized(const engine::assets::TensorSource & packed) {
    require_stored_as(packed, kIstftHead, "f16", kF16Bytes);
    require_stored_as(packed, kVocoderLinear, "f16", kF16Bytes);
    require_stored_as(packed, kNormProjection, "f16", kF16Bytes);
    require_stored_as(packed, kAdaLnModulation, "f16", kF16Bytes);
    require_stored_as(packed, kLmHead, "f16", kF16Bytes);
    require_stored_as(packed, kMelHead, "f16", kF16Bytes);
    require_stored_as(packed, kProjOut, "f16", kF16Bytes);
    require_stored_as(packed, kTokenizerDecoder, "f16", kF16Bytes);
}

// A packager who knows better can still quantize one protected tensor by name.
void test_per_tensor_override_wins(
    const std::filesystem::path & root,
    const std::filesystem::path & source) {
    GgufConversionOptions options;
    options.type_overrides.push_back({kIstftHead, TensorStorageType::Q8_0});
    const auto packed = pack(root, source, "override.gguf", std::move(options));

    require_stored_as(*packed, kIstftHead, "q8_0", kQ8Bytes);
    // Only the named tensor moves; the rest of the policy still holds.
    require_stored_as(*packed, kVocoderLinear, "f16", kF16Bytes);
    require_stored_as(*packed, kLmHead, "f16", kF16Bytes);
    require_stored_as(*packed, kFfnWeight, "q8_0", kQ8Bytes);
}

// The whole-policy escape hatch reproduces the pre-policy output exactly.
void test_policy_can_be_disabled(
    const std::filesystem::path & root,
    const std::filesystem::path & source) {
    GgufConversionOptions options;
    options.quantize_audible_tensors = true;
    const auto packed = pack(root, source, "unprotected.gguf", std::move(options));

    require_stored_as(*packed, kIstftHead, "q8_0", kQ8Bytes);
    require_stored_as(*packed, kVocoderLinear, "q8_0", kQ8Bytes);
    require_stored_as(*packed, kNormProjection, "q8_0", kQ8Bytes);
    require_stored_as(*packed, kAdaLnModulation, "q8_0", kQ8Bytes);
    require_stored_as(*packed, kLmHead, "q8_0", kQ8Bytes);
    require_stored_as(*packed, kTokenizerDecoder, "q8_0", kQ8Bytes);

    // Disabling the audible-tensor policy must not disable the lookup-table rule.
    require_stored_as(*packed, kEmbedding, "f16", kF16Bytes);
    require_stored_as(*packed, kCodebook, "f16", kF16Bytes);
    require_stored_as(*packed, kConvKernel, "f32", 4 * 4 * 7 * 4);
}

// A 16-bit target quantizes nothing, so the policy must be inert there: every float
// weight still lands at BF16 and no tensor is pushed to F16 behind the packager's back.
void test_policy_is_inert_for_non_quantized_targets(
    const std::filesystem::path & root,
    const std::filesystem::path & source) {
    const auto output = root / "bf16.gguf";
    engine::assets::convert_tensor_sources_to_gguf(
        {{source, ""}},
        output,
        TensorStorageType::BF16,
        /*overwrite=*/true,
        /*embed_sidecars=*/false,
        root,
        {},
        std::nullopt,
        {});
    const auto packed = engine::assets::open_tensor_source(output);

    require_stored_as(*packed, kIstftHead, "bf16", kF16Bytes);
    require_stored_as(*packed, kLmHead, "bf16", kF16Bytes);
    require_stored_as(*packed, kFfnWeight, "bf16", kF16Bytes);
    require_stored_as(*packed, kConvKernel, "bf16", 4 * 4 * 7 * 2);
}

}  // namespace

int main() {
    try {
        const auto root = scratch_root();
        std::error_code cleanup_error;
        std::filesystem::remove_all(root, cleanup_error);
        const auto source = write_source(root);

        test_audible_tensor_reasons();

        const auto packed = pack(root, source, "policy.gguf", {});
        test_existing_behaviour_is_unchanged(*packed);
        test_audible_tensors_are_not_quantized(*packed);

        test_per_tensor_override_wins(root, source);
        test_policy_can_be_disabled(root, source);
        test_policy_is_inert_for_non_quantized_targets(root, source);

        std::filesystem::remove_all(root, cleanup_error);
        std::cout << "quantization_policy_test passed\n";
    } catch (const std::exception & ex) {
        std::cerr << "quantization_policy_test failed: " << ex.what() << "\n";
        return 1;
    }
    return 0;
}
