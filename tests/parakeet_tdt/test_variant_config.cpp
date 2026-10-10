#include "engine/community_models/parakeet_tdt/assets.h"
#include "engine/community_models/parakeet_tdt/graph_precision.h"
#include "engine/community_models/parakeet_tdt/word_timestamps.h"
#include "engine/framework/io/safetensors.h"
#include "engine/framework/model_spec/package.h"
#include "test_assert.h"

#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
void write(const std::filesystem::path & path, const std::string & data) {
    std::ofstream(path, std::ios::binary) << data;
}

std::string config(const std::string & encoder_extra = "", const std::string & root_extra = "") {
    return R"json({"model_type":"parakeet_tdt","vocab_size":3,"blank_token_id":2,"pad_token_id":0,
      "decoder_hidden_size":4,"num_decoder_layers":2,"max_symbols_per_step":10,"durations":[0,1,2,3,4],
      "encoder_config":{"hidden_size":8,"intermediate_size":16,"num_hidden_layers":1,"num_attention_heads":1,
        "conv_kernel_size":9,"subsampling_factor":8,"subsampling_conv_channels":8,
        "subsampling_conv_kernel_size":3,"subsampling_conv_stride":2,"max_position_embeddings":32)json"
        + encoder_extra + "}" + root_extra + "}";
}

void test_config(const std::filesystem::path & root) {
    const float value = 1.f;
    std::vector<unsigned char> bytes(sizeof(value));
    std::memcpy(bytes.data(), &value, sizeof(value));
    engine::io::write_safetensors_file(root / "model.safetensors", {{"dummy.weight", "F32", {1}, bytes}});
    write(root / "processor_config.json", R"json({"feature_extractor":{"sampling_rate":16000,
        "feature_size":128,"n_fft":512,"win_length":400,"hop_length":160,"preemphasis":0.97}})json");
    write(root / "tokenizer.json", R"json({"model":{"type":"BPE","vocab":{"<unk>":0,"hello":1,"<blank>":2},"merges":[]}})json");
    engine::model_spec::ScopedSpecOverride spec(
        std::filesystem::path(ENGINE_REPO_ROOT) / "model_specs/parakeet_tdt.json", root);
    write(root / "config.json", config());
    auto base = engine::community_models::parakeet_tdt::load_parakeet_assets(root);
    engine::test::require_close(base->config.encoder.batch_norm_variance_floor, 1e-5f, 0.f, "legacy BN floor");
    engine::test::require_close(base->config.encoder.subsampling_input_scale, 0.f, 0.f, "legacy scale sentinel");
    engine::test::require(!base->config.force_f32_matmul, "legacy matmul policy unchanged");
    engine::test::require(!base->config.cpu_f32_matmul_weights, "legacy CPU storage unchanged");
    engine::test::require(!base->config.token_duration_word_timestamps, "legacy word timing unchanged");

    write(root / "config.json", config(
        ",\"batch_norm_variance_floor\":0,\"subsampling_input_scale\":1", ",\"audiocpp_matmul_precision\":\"f32\""));
    auto phonon = engine::community_models::parakeet_tdt::load_parakeet_assets(root);
    engine::test::require_close(phonon->config.encoder.batch_norm_variance_floor, 0.f, 0.f, "reference BN floor");
    engine::test::require_close(phonon->config.encoder.subsampling_input_scale, 1.f, 0.f, "reference scale");
    engine::test::require(phonon->config.force_f32_matmul, "Phonon matmul policy");
    write(root / "config.json", config("", ",\"audiocpp_cpu_matmul_weight_type\":\"f32\""));
    auto quantized = engine::community_models::parakeet_tdt::load_parakeet_assets(root);
    engine::test::require(quantized->config.cpu_f32_matmul_weights, "Q8 CPU F32 compute storage");
    write(root / "config.json", config("", ",\"audiocpp_word_timestamp_mode\":\"token_duration\",\"audiocpp_punctuation_token_ids\":[1,0]"));
    auto timed = engine::community_models::parakeet_tdt::load_parakeet_assets(root);
    engine::test::require(timed->config.token_duration_word_timestamps, "Phonon token duration policy");
    engine::test::require_eq(timed->config.punctuation_token_ids.front(), int32_t(0), "punctuation ids sorted");

    for (const auto & bad : {config(",\"batch_norm_variance_floor\":-1"),
                            config(",\"subsampling_input_scale\":0"),
                            config(",\"subsampling_input_scale\":-1"),
                            config("", ",\"audiocpp_matmul_precision\":\"half\""),
                            config("", ",\"audiocpp_cpu_matmul_weight_type\":\"q8_0\""),
                            config("", ",\"audiocpp_word_timestamp_mode\":\"token_duration\""),
                            config("", ",\"audiocpp_word_timestamp_mode\":\"rounded\""),
                            config("", ",\"audiocpp_punctuation_token_ids\":[3]"),
                            config("", ",\"audiocpp_punctuation_token_ids\":[0.5]")}) {
        write(root / "config.json", bad);
        bool rejected = false;
        try { (void)engine::community_models::parakeet_tdt::load_parakeet_assets(root); }
        catch (const std::runtime_error &) { rejected = true; }
        engine::test::require(rejected, "invalid precision/normalization policy rejected");
    }
}

void test_graph_precision() {
    auto * ctx = ggml_init({1024 * 1024, nullptr, true});
    if (!ctx) throw std::runtime_error("test graph allocation failed");
    auto * graph = ggml_new_graph(ctx);
    auto * weight = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, 4);
    auto * input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, 2);
    auto * mm = ggml_mul_mat(ctx, weight, input);
    auto * activation = ggml_relu(ctx, mm);
    const auto activation_op = activation->op;
    ggml_build_forward_expand(graph, activation);
    engine::community_models::parakeet_tdt::configure_matmul_precision(graph, false);
    engine::test::require_eq(mm->op_params[0], int32_t(GGML_PREC_DEFAULT), "legacy graph policy unchanged");
    engine::community_models::parakeet_tdt::configure_matmul_precision(graph, true);
    engine::test::require_eq(mm->op_params[0], int32_t(GGML_PREC_F32), "Phonon graph accumulation");
    engine::test::require_eq(activation->op, activation_op, "non-matmul graph node unchanged");
    ggml_free(ctx);
}

void test_token_duration_timestamps() {
    using namespace engine::community_models::parakeet_tdt;
    auto words = token_duration_word_timestamps({
        {" Hi", 0, 10, false}, {" !", 10, 0, true}, {" there", 30, 4, false}}, 40);
    engine::test::require_eq(words.size(), size_t(2), "punctuation attached to previous word");
    engine::test::require(words[0].word == "Hi !", "preserve punctuation spacing");
    engine::test::require_eq(words[0].span.end_sample, int64_t(10), "zero duration is not stretched over silence");
    words = token_duration_word_timestamps({
        {" ", 3, 0, true}, {"word", 4, 2, false}, {" .", 6, 0, true}, {" next", 8, 4, false}}, 10);
    engine::test::require_eq(words[0].span.start_sample, int64_t(3), "bare marker retains its timing");
    engine::test::require_eq(words[0].span.end_sample, int64_t(6), "last token duration determines end");
    engine::test::require_eq(words[1].span.end_sample, int64_t(10), "last word clamped to actual audio length");
    engine::test::require(token_duration_word_timestamps({{" ", 0, 1, true}}, 10).empty(), "whitespace produces no word");
}
}  // namespace

int main() {
    const auto root = std::filesystem::temp_directory_path() /
        ("audiocpp-parakeet-variant-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    if (!std::filesystem::create_directory(root)) return 1;
    try {
        test_config(root);
        test_graph_precision();
        test_token_duration_timestamps();
        std::filesystem::remove_all(root);
        std::cout << "PASS: legacy defaults and Phonon variant policies\n";
        return 0;
    } catch (const std::exception & error) {
        std::filesystem::remove_all(root);
        std::cerr << error.what() << '\n';
        return 1;
    }
}
