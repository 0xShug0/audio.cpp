#include "engine/community_models/lfm2_audio/assets.h"
#include "test_assert.h"

#include <ggml.h>
#include <gguf.h>

#include <cstdint>
#include <filesystem>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using engine::community_models::lfm2_audio::load_lfm2_audio_assets;
using engine::community_models::lfm2_audio::load_lfm2_audio_components;
using engine::community_models::lfm2_audio::load_lfm2_audio_output_components;
using engine::test::require;
using engine::test::require_eq;

// A tensor name and its ggml ne dimensions (innermost first).
using TensorDims = std::pair<std::string, std::vector<int64_t>>;

struct GgufDeleter {
    void operator()(gguf_context * ctx) const noexcept { gguf_free(ctx); }
};

struct GgmlDeleter {
    void operator()(ggml_context * ctx) const noexcept { ggml_free(ctx); }
};

using GgufPtr = std::unique_ptr<gguf_context, GgufDeleter>;

GgufPtr new_gguf() {
    return GgufPtr(gguf_init_empty());
}

void set_str_array(gguf_context * ctx, const char * key, const std::vector<std::string> & values) {
    std::vector<const char *> data;
    for (const auto & value : values) {
        data.push_back(value.c_str());
    }

    gguf_set_arr_str(ctx, key, data.data(), data.size());
}

void write_gguf(const std::filesystem::path & path, gguf_context * ctx, const std::vector<TensorDims> & tensors) {
    std::unique_ptr<ggml_context, GgmlDeleter> tensor_ctx(ggml_init({1024 * 1024, nullptr, false}));
    for (const auto & [name, dims] : tensors) {
        auto * tensor = ggml_new_tensor(tensor_ctx.get(), GGML_TYPE_F32, static_cast<int>(dims.size()), dims.data());
        ggml_set_name(tensor, name.c_str());
        ggml_set_zero(tensor);
        gguf_add_tensor(ctx, tensor);
    }

    require(gguf_write_to_file(ctx, path.string().c_str(), false), "write " + path.string());
}

// An LFM2 backbone with 3 layers (short-conv, attention, short-conv), width 8
// and a 4-token vocabulary.
void write_backbone(
    const std::filesystem::path & path, const std::string & language = "en", bool merges = true, bool attention = true) {
    auto ctx = new_gguf();
    gguf_set_val_str(ctx.get(), "general.architecture", "lfm2");
    set_str_array(ctx.get(), "general.languages", {language});
    gguf_set_val_u32(ctx.get(), "lfm2.block_count", 3);
    gguf_set_val_u32(ctx.get(), "lfm2.embedding_length", 8);
    gguf_set_val_u32(ctx.get(), "lfm2.feed_forward_length", 16);
    gguf_set_val_u32(ctx.get(), "lfm2.attention.head_count", 2);
    gguf_set_val_u32(ctx.get(), "lfm2.context_length", 128);
    gguf_set_val_u32(ctx.get(), "lfm2.shortconv.l_cache", 3);
    gguf_set_val_f32(ctx.get(), "lfm2.attention.layer_norm_rms_epsilon", 1e-5f);
    gguf_set_val_f32(ctx.get(), "lfm2.rope.freq_base", 1e6f);
    const int32_t kv_heads[] = {0, attention ? 1 : 0, 0};
    gguf_set_arr_data(ctx.get(), "lfm2.attention.head_count_kv", GGUF_TYPE_INT32, kv_heads, 3);
    gguf_set_val_str(ctx.get(), "tokenizer.ggml.model", "gpt2");
    gguf_set_val_str(ctx.get(), "tokenizer.ggml.pre", "lfm2");
    set_str_array(ctx.get(), "tokenizer.ggml.tokens", {"<|im_end|>", "a", "b", "ab"});
    if (merges) {
        set_str_array(ctx.get(), "tokenizer.ggml.merges", {"a b"});
    }

    const int32_t token_types[] = {3, 1, 1, 1};
    gguf_set_arr_data(ctx.get(), "tokenizer.ggml.token_type", GGUF_TYPE_INT32, token_types, 4);
    write_gguf(path, ctx.get(), {{"token_embd.weight", {8, 4}}});
}

// An lfm2a mmproj whose adapter maps to width output_size.
void write_mmproj(
    const std::filesystem::path & path, int64_t output_size = 8, const char * projector = "lfm2a", uint32_t heads = 2) {
    auto ctx = new_gguf();
    gguf_set_val_str(ctx.get(), "general.architecture", "clip");
    gguf_set_val_str(ctx.get(), "clip.projector_type", projector);
    gguf_set_val_u32(ctx.get(), "clip.audio.num_mel_bins", 128);
    gguf_set_val_u32(ctx.get(), "clip.audio.embedding_length", 16);
    gguf_set_val_u32(ctx.get(), "clip.audio.block_count", 2);
    gguf_set_val_u32(ctx.get(), "clip.audio.attention.head_count", heads);
    gguf_set_val_f32(ctx.get(), "clip.audio.attention.layer_norm_epsilon", 1e-5f);
    write_gguf(path, ctx.get(), {
        {"a.blk.0.ffn_up.weight", {16, 64}},
        {"a.blk.0.conv_dw.weight", {9, 16}},
        {"a.conv1d.0.weight", {3, 3, 1, 32}},
        {"mm.a.mlp.1.weight", {16, 24}},
        {"mm.a.mlp.3.weight", {24, output_size}},
    });
}

void write_other(const std::filesystem::path & path, const char * architecture) {
    auto ctx = new_gguf();
    gguf_set_val_str(ctx.get(), "general.architecture", architecture);
    write_gguf(path, ctx.get(), {{"weight", {4}}});
}

std::filesystem::path fresh_directory(const std::string & name) {
    const auto root = std::filesystem::temp_directory_path() / name;
    std::filesystem::remove_all(root);
    std::filesystem::create_directories(root);
    return root;
}

template <typename Fn>
void require_throws(Fn && fn, const std::string & label) {
    bool threw = false;
    try {
        fn();
    } catch (const std::runtime_error &) {
        threw = true;
    }

    require(threw, label + " must throw");
}

template <typename Fn>
void require_throws_with(Fn && fn, const std::string & needle, const std::string & label) {
    try {
        fn();
    } catch (const std::runtime_error & error) {
        const std::string message = error.what();
        require(message.find(needle) != std::string::npos, label + " threw \"" + message + "\", expected \"" + needle + "\"");
        return;
    }

    require(false, label + " must throw");
}

// The layout of Liquid's GGUF repos: two quantizations of each component and
// the audio output files, which ASR does not use.
std::filesystem::path write_package(const std::string & name) {
    const auto root = fresh_directory(name);
    write_backbone(root / "Model-F16.gguf");
    write_backbone(root / "Model-Q8_0.gguf");
    write_mmproj(root / "mmproj-Model-F16.gguf");
    write_mmproj(root / "mmproj-Model-Q8_0.gguf");
    write_other(root / "vocoder-Model-F16.gguf", "vocoder");
    write_other(root / "tokenizer-Model-F16.gguf", "tokenizer");
    return root;
}

void test_reads_configs() {
    const auto root = fresh_directory("audiocpp_lfm2_audio_configs_test");
    write_backbone(root / "Model-F16.gguf", "ja");
    write_mmproj(root / "mmproj-Model-F16.gguf");
    const auto components = load_lfm2_audio_components(*load_lfm2_audio_assets(root), "", "");

    const auto & backbone = components->backbone;
    require_eq(backbone.num_layers(), 3, "backbone layers");
    require(!backbone.is_attention_layer(0) && backbone.is_attention_layer(1) && !backbone.is_attention_layer(2),
        "only layer 1 is an attention layer");
    require_eq(backbone.hidden_size, 8, "backbone hidden size");
    require_eq(backbone.intermediate_size, 16, "backbone intermediate size");
    require_eq(backbone.head_dim, 4, "backbone head dim");
    require_eq(backbone.conv_kernel_size, 3, "backbone conv kernel");
    require_eq(backbone.context_length, 128, "backbone context length");
    require_eq(backbone.vocab_size, 4, "backbone vocab size");

    const auto & encoder = components->encoder;
    require_eq(encoder.n_mels, 128, "encoder mel bins");
    require_eq(encoder.hidden_size, 16, "encoder hidden size");
    require_eq(encoder.num_layers, 2, "encoder layers");
    require_eq(encoder.num_heads, 2, "encoder heads");
    require_eq(encoder.intermediate_size, 64, "encoder FFN width");
    require_eq(encoder.conv_kernel_size, 9, "encoder conv kernel");
    require_eq(encoder.subsampling_channels, 32, "subsampling channels");
    require_eq(encoder.adapter_hidden_size, 24, "adapter hidden size");
    require_eq(encoder.output_size, 8, "adapter output size");

    const auto & vocabulary = components->vocabulary;
    require_eq(vocabulary.tokens.size(), size_t{4}, "tokens");
    require_eq(vocabulary.tokens[3], std::string("ab"), "token 3");
    require_eq(vocabulary.merges.size(), size_t{1}, "merges");
    require_eq(vocabulary.token_types[0], 3, "token 0 type");
    require_eq(vocabulary.pre_tokenizer, std::string("lfm2"), "pre-tokenizer");
    require_eq(components->languages.size(), size_t{1}, "languages");
    require_eq(components->languages[0], std::string("ja"), "language");
    std::filesystem::remove_all(root);
}

void test_selects_components() {
    const auto root = write_package("audiocpp_lfm2_audio_select_test");
    const auto assets = load_lfm2_audio_assets(root);
    require_throws([&] { (void)load_lfm2_audio_components(*assets, "", ""); }, "two backbones and no choice");

    const auto q8 = load_lfm2_audio_components(*assets, "Model-Q8_0.gguf", "");
    require_eq(q8->model_path, root / "Model-Q8_0.gguf", "chosen backbone");
    require_eq(q8->mmproj_path, root / "mmproj-Model-Q8_0.gguf", "paired mmproj");

    const auto mixed = load_lfm2_audio_components(*assets, "Model-F16.gguf", "mmproj-Model-Q8_0.gguf");
    require_eq(mixed->model_path, root / "Model-F16.gguf", "mixed backbone");
    require_eq(mixed->mmproj_path, root / "mmproj-Model-Q8_0.gguf", "mixed mmproj");

    // --model naming a backbone file makes it the default.
    const auto file_assets = load_lfm2_audio_assets(root / "Model-F16.gguf");
    require_eq(file_assets->model_root, root, "model root of a file");
    const auto from_file = load_lfm2_audio_components(*file_assets, "", "");
    require_eq(from_file->model_path, root / "Model-F16.gguf", "backbone from --model");
    require_eq(from_file->mmproj_path, root / "mmproj-Model-F16.gguf", "mmproj paired with --model");
    std::filesystem::remove_all(root);
}

void test_single_unpaired_mmproj() {
    const auto root = fresh_directory("audiocpp_lfm2_audio_unpaired_test");
    write_backbone(root / "Model-Q4_0.gguf");
    write_mmproj(root / "mmproj-Model-F16.gguf");
    const auto components = load_lfm2_audio_components(*load_lfm2_audio_assets(root), "", "");
    require_eq(components->mmproj_path, root / "mmproj-Model-F16.gguf", "the only mmproj");

    write_mmproj(root / "mmproj-Model-Q8_0.gguf");
    require_throws(
        [&] { (void)load_lfm2_audio_components(*load_lfm2_audio_assets(root), "", ""); },
        "two unpaired mmproj files and no choice");
    std::filesystem::remove_all(root);
}

void test_rejects_bad_choices() {
    const auto root = write_package("audiocpp_lfm2_audio_reject_test");
    const auto assets = load_lfm2_audio_assets(root);
    const auto absolute = (root / "Model-F16.gguf").string();
    require_throws([&] { (void)load_lfm2_audio_components(*assets, absolute, ""); }, "an absolute model_gguf");
    require_throws([&] { (void)load_lfm2_audio_components(*assets, "Model-Q4_0.gguf", ""); }, "a missing model_gguf");
    require_throws(
        [&] { (void)load_lfm2_audio_components(*assets, "mmproj-Model-F16.gguf", "mmproj-Model-F16.gguf"); },
        "an mmproj as the backbone");
    require_throws(
        [&] { (void)load_lfm2_audio_components(*assets, "Model-F16.gguf", "Model-Q8_0.gguf"); },
        "a backbone as the mmproj");
    require_throws(
        [&] { (void)load_lfm2_audio_components(*assets, "Model-F16.gguf", "vocoder-Model-F16.gguf"); },
        "a vocoder as the mmproj");
    std::filesystem::remove_all(root);

    const auto mismatch = fresh_directory("audiocpp_lfm2_audio_width_test");
    write_backbone(mismatch / "Model-F16.gguf");
    write_mmproj(mismatch / "mmproj-Model-F16.gguf", 12);
    require_throws_with(
        [&] { (void)load_lfm2_audio_components(*load_lfm2_audio_assets(mismatch), "", ""); }, "adapter output width",
        "an adapter narrower than the backbone");
    std::filesystem::remove_all(mismatch);

    // Without merges the tokenizer would quietly spell the prompt in bytes.
    const auto no_merges = fresh_directory("audiocpp_lfm2_audio_no_merges_test");
    write_backbone(no_merges / "Model-F16.gguf", "en", false);
    write_mmproj(no_merges / "mmproj-Model-F16.gguf");
    require_throws_with(
        [&] { (void)load_lfm2_audio_components(*load_lfm2_audio_assets(no_merges), "", ""); }, "tokenizer vocabulary",
        "a backbone without merges");
    std::filesystem::remove_all(no_merges);

    const auto no_heads = fresh_directory("audiocpp_lfm2_audio_no_heads_test");
    write_backbone(no_heads / "Model-F16.gguf");
    write_mmproj(no_heads / "mmproj-Model-F16.gguf", 8, "lfm2a", 0);
    require_throws_with(
        [&] { (void)load_lfm2_audio_components(*load_lfm2_audio_assets(no_heads), "", ""); }, "encoder head count",
        "an encoder with no attention heads");
    std::filesystem::remove_all(no_heads);
}

void test_file_names() {
    // Extensions are case-insensitive, like the framework's GGUF reader.
    const auto upper = fresh_directory("audiocpp_lfm2_audio_upper_test");
    write_backbone(upper / "Model-F16.GGUF");
    write_mmproj(upper / "mmproj-Model-F16.GGUF");
    const auto components = load_lfm2_audio_components(*load_lfm2_audio_assets(upper), "", "");
    require_eq(components->model_path, upper / "Model-F16.GGUF", "uppercase backbone");
    std::filesystem::remove_all(upper);

    // Component options name files inside the model directory.
    const auto root = write_package("audiocpp_lfm2_audio_escape_test");
    const auto nested = root / "nested";
    std::filesystem::create_directories(nested);
    write_mmproj(nested / "mmproj-Model-F16.gguf");
    const auto assets = load_lfm2_audio_assets(root);
    require_eq(load_lfm2_audio_components(*assets, "Model-F16.gguf", "nested/mmproj-Model-F16.gguf")->mmproj_path,
        root / "nested" / "mmproj-Model-F16.gguf", "a subdirectory");
    require_throws_with([&] { (void)load_lfm2_audio_components(*assets, "../Model-F16.gguf", ""); },
        "inside the model directory", "a path leaving the model directory");
    require_throws_with([&] { (void)load_lfm2_audio_components(*assets, "nested/../../Model-F16.gguf", ""); },
        "inside the model directory", "a path leaving it through a subdirectory");
    std::filesystem::remove_all(root);

    // A backbone without attention layers would leave the decoder no KV cache.
    const auto no_attention = fresh_directory("audiocpp_lfm2_audio_no_attention_test");
    write_backbone(no_attention / "Model-F16.gguf", "en", true, false);
    write_mmproj(no_attention / "mmproj-Model-F16.gguf");
    require_throws_with([&] { (void)load_lfm2_audio_components(*load_lfm2_audio_assets(no_attention), "", ""); },
        "no attention layer", "a backbone without attention layers");
    std::filesystem::remove_all(no_attention);
}

// The loader probes directories of other models, which must not look like an
// LFM2-Audio package.
void test_rejects_other_packages() {
    const auto root = fresh_directory("audiocpp_lfm2_audio_other_test");
    write_other(root / "model.gguf", "llama");
    write_mmproj(root / "mmproj-model.gguf", 8, "ultravox");
    require_throws_with([&] { (void)load_lfm2_audio_assets(root); }, "no backbone GGUF", "a llama package");

    write_backbone(root / "Model-F16.gguf");
    require_throws_with([&] { (void)load_lfm2_audio_assets(root); }, "no mmproj GGUF", "a package without an lfm2a mmproj");
    require_throws([&] { (void)load_lfm2_audio_assets(root / "model.gguf"); }, "a non-LFM2 --model file");
    require_throws([&] { (void)load_lfm2_audio_assets(root / "missing"); }, "a missing path");
    std::filesystem::remove_all(root);
}

// Packages installed before audio.cpp had LFM2-Audio TTS hold the backbone
// and mmproj only; TTS and S2S must say to reinstall rather than to choose a
// vocoder that is not there.
void test_missing_output_files() {
    const auto models = fresh_directory("audiocpp_lfm2_audio_old_package_test");
    const auto root = models / "LFM2.5-Audio-1.5B-GGUF";
    std::filesystem::create_directories(root);
    write_backbone(root / "LFM2.5-Audio-1.5B-Q8_0.gguf");
    write_mmproj(root / "mmproj-LFM2.5-Audio-1.5B-Q8_0.gguf");
    const auto assets = load_lfm2_audio_assets(root);
    const auto components = load_lfm2_audio_components(*assets, "", "");
    const auto output = [&] { (void)load_lfm2_audio_output_components(*assets, *components, "", ""); };
    const auto command = "python3 tools/model_manager_v2.py install lfm2_audio_1_5b_q8_0 --models-root " + models.string() + " --overwrite";
    require_throws_with(output, "need vocoder-LFM2.5-Audio-1.5B-Q8_0.gguf and tokenizer-LFM2.5-Audio-1.5B-Q8_0.gguf, which",
        "both files named");
    require_throws_with(output, command, "the package's reinstall command");

    // The directory given with a trailing separator.
    const auto slashed = load_lfm2_audio_assets(root.string() + "/");
    const auto slashed_components = load_lfm2_audio_components(*slashed, "", "");
    require_throws_with([&] { (void)load_lfm2_audio_output_components(*slashed, *slashed_components, "", ""); }, command,
        "the reinstall command for a directory ending in a separator");

    // A relative directory is printed absolute: the command runs from the
    // repository root, not from where the session was opened.
    const auto relative = std::filesystem::relative(root);
    const auto relative_assets = load_lfm2_audio_assets(relative);
    const auto relative_components = load_lfm2_audio_components(*relative_assets, "", "");
    require_throws_with([&] { (void)load_lfm2_audio_output_components(*relative_assets, *relative_components, "", ""); },
        "--models-root " + (std::filesystem::current_path() / relative).lexically_normal().parent_path().string() + " --overwrite",
        "the reinstall command for a relative directory");

    // A models directory whose path has a space is quoted.
    const auto spaced = fresh_directory("audiocpp lfm2_audio spaced package test");
    const auto spaced_root = spaced / "LFM2.5-Audio-1.5B-GGUF";
    std::filesystem::create_directories(spaced_root);
    write_backbone(spaced_root / "LFM2.5-Audio-1.5B-Q8_0.gguf");
    write_mmproj(spaced_root / "mmproj-LFM2.5-Audio-1.5B-Q8_0.gguf");
    const auto spaced_assets = load_lfm2_audio_assets(spaced_root);
    const auto spaced_components = load_lfm2_audio_components(*spaced_assets, "", "");
    require_throws_with([&] { (void)load_lfm2_audio_output_components(*spaced_assets, *spaced_components, "", ""); },
        "--models-root \"" + spaced.string() + "\" --overwrite", "the reinstall command for a path with a space");
    std::filesystem::remove_all(spaced);

    // A file of the pair still there: only the other is missing.
    write_other(root / "vocoder-LFM2.5-Audio-1.5B-Q8_0.gguf", "vocoder");
    require_throws_with(output, "need tokenizer-LFM2.5-Audio-1.5B-Q8_0.gguf, which", "only the missing file named");
    std::filesystem::remove_all(models);

    // Files no package lists get placeholders.
    const auto renamed = fresh_directory("audiocpp_lfm2_audio_renamed_test");
    write_backbone(renamed / "Model-F16.gguf");
    write_mmproj(renamed / "mmproj-Model-F16.gguf");
    const auto renamed_assets = load_lfm2_audio_assets(renamed);
    const auto renamed_components = load_lfm2_audio_components(*renamed_assets, "", "");
    require_throws_with([&] { (void)load_lfm2_audio_output_components(*renamed_assets, *renamed_components, "", ""); },
        "install <package> --models-root <models directory> --overwrite", "a renamed package");
    std::filesystem::remove_all(renamed);
}

}  // namespace

int main() {
    try {
        test_reads_configs();
        test_selects_components();
        test_single_unpaired_mmproj();
        test_rejects_bad_choices();
        test_file_names();
        test_rejects_other_packages();
        test_missing_output_files();
        std::cout << "lfm2_audio_assets_test: PASS\n";
        return 0;
    } catch (const std::exception & error) {
        std::cerr << "lfm2_audio_assets_test: " << error.what() << '\n';
        return 1;
    }
}
