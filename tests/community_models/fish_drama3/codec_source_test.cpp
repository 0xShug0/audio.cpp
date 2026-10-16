#include "engine/community_models/fish_drama3/assets.h"
#include "engine/framework/audio/wav_reader.h"
#include "engine/framework/audio/wav_writer.h"
#include "engine/framework/core/execution_context.h"
#include "engine/framework/debug/trace.h"
#include "engine/framework/io/safetensors.h"
#include "engine/framework/io/json.h"
#include "engine/framework/model_spec/package.h"
#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/modules/conv_modules.h"
#include "engine/framework/modules/activation_modules.h"
#include "engine/framework/modules/attention_modules.h"
#include "engine/framework/modules/weight_binding.h"
#include <ggml-alloc.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iostream>
#include <stdexcept>

namespace {
namespace assets = engine::assets;

template <typename T>
std::vector<unsigned char> bytes_for(const std::vector<T> & values) {
    std::vector<unsigned char> bytes(values.size() * sizeof(T));
    std::memcpy(bytes.data(), values.data(), bytes.size());
    return bytes;
}

template <typename F>
void require_failure(F action) {
    try { action(); } catch (const std::exception &) { return; }
    throw std::runtime_error("invalid codec source was accepted");
}

struct FixtureDirectory {
    std::filesystem::path path = std::filesystem::temp_directory_path() /
        ("audiocpp_drama3_codec_" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    FixtureDirectory() { std::filesystem::create_directories(path); }
    ~FixtureDirectory() { std::error_code error; std::filesystem::remove_all(path, error); }
};
}  // namespace

int main(int argc, char ** argv) {
    try {
        if (argc == 5 && std::string(argv[1]) == "--attention" && std::string(argv[4]) == "--log") {
            engine::debug::configure_logging({true, std::nullopt});
            const auto started = std::chrono::steady_clock::now();
            const auto source = assets::open_tensor_source(argv[2]);
            engine::core::BackendConfig backend;
            backend.type = engine::core::BackendType::Cuda;
            backend.threads = 8;
            engine::core::ExecutionContext execution(backend);
            std::unique_ptr<ggml_context, decltype(&ggml_free)> context(
                ggml_init({4 * 1024 * 1024, nullptr, true}), ggml_free);
            if (!context) throw std::runtime_error("attention graph context allocation failed");
            engine::core::BackendWeightStore store(execution.backend(), backend.type, "codec_attention", 1024 * 1024);
            const auto q = store.load_tensor(*source, "q", assets::TensorStorageType::F32, source->require_metadata("q").shape);
            const auto k = store.load_tensor(*source, "k", assets::TensorStorageType::F32, source->require_metadata("k").shape);
            const auto v = store.load_tensor(*source, "v", assets::TensorStorageType::F32, source->require_metadata("v").shape);
            const auto mask = store.load_tensor(*source, "mask", assets::TensorStorageType::F16, source->require_metadata("mask").shape);
            store.upload();
            auto * output = ggml_flash_attn_ext(context.get(), q.tensor, k.tensor, v.tensor, mask.tensor,
                1.0F / std::sqrt(static_cast<float>(q.tensor->ne[0])), 0.0F, 0.0F);
            ggml_flash_attn_ext_set_prec(output, GGML_PREC_F32);
            output = ggml_round_bf16(context.get(), output);
            ggml_set_output(output);
            engine::core::ModuleBuildContext ctx{context.get(), "codec.attention", backend.type};
            const auto explicit_context = engine::modules::ScaledDotProductAttentionModule({
                q.shape.at(3), engine::modules::ScaledDotProductAttentionLowering::Explicit, GGML_PREC_F32
            }).build(ctx, q, k, v, mask);
            auto * explicit_output = ggml_round_bf16(context.get(), explicit_context.tensor);
            ggml_set_output(explicit_output);
            auto * graph = ggml_new_graph_custom(context.get(), 128, false);
            ggml_build_forward_expand(graph, output);
            ggml_build_forward_expand(graph, explicit_output);
            std::unique_ptr<std::remove_pointer_t<ggml_gallocr_t>, decltype(&ggml_gallocr_free)> allocator(
                ggml_gallocr_new(ggml_backend_get_default_buffer_type(execution.backend())), ggml_gallocr_free);
            if (!allocator || !ggml_gallocr_alloc_graph(allocator.get(), graph))
                throw std::runtime_error("attention graph allocation failed");
            if (engine::core::compute_backend_graph(execution.backend(), graph) != GGML_STATUS_SUCCESS)
                throw std::runtime_error("attention graph execution failed");
            ggml_backend_synchronize(execution.backend());
            engine::io::write_safetensors_file(argv[3], {
                {"output", "F32", {output->ne[3], output->ne[2], output->ne[1], output->ne[0]},
                 bytes_for(engine::core::read_tensor_f32(output))},
                {"explicit", "F32", {explicit_output->ne[3], explicit_output->ne[2], explicit_output->ne[1], explicit_output->ne[0]},
                 bytes_for(engine::core::read_tensor_f32(explicit_output))}});
            engine::core::release_backend_graph_resources(execution.backend(), graph);
            engine::debug::timing_log_scalar("session.wall_ms", std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - started).count());
            return 0;
        }
        if (argc == 5 && std::string(argv[1]) == "--first-conv" && std::string(argv[4]) == "--log") {
            engine::debug::configure_logging({true, std::nullopt});
            const auto started = std::chrono::steady_clock::now();
            const auto source = assets::open_tensor_source(argv[2]);
            engine::core::BackendConfig backend;
            backend.type = engine::core::BackendType::Cuda;
            backend.threads = 8;
            engine::core::ExecutionContext execution(backend);
            std::unique_ptr<ggml_context, decltype(&ggml_free)> context(
                ggml_init({8 * 1024 * 1024, nullptr, true}), ggml_free);
            if (!context) throw std::runtime_error("first-conv graph context allocation failed");
            engine::core::ModuleBuildContext ctx{context.get(), "fish_codec.first_conv", backend.type};
            engine::core::BackendWeightStore store(execution.backend(), backend.type, "first_conv", 1024 * 1024);
            const auto weights = engine::modules::binding::conv1d_from_named_source(
                store, *source, "weight", "bias", assets::TensorStorageType::Native);
            const engine::modules::Snake1dWeights snake_weights{store.make_f32(
                engine::core::TensorShape::from_dims({64}), source->require_f32("snake_alpha")), std::nullopt};
            auto alpha = source->require_f32("snake_alpha");
            engine::core::round_f32_to_bf16_in_place(alpha);
            std::vector<float> inverse(alpha.size());
            for (size_t i = 0; i < alpha.size(); ++i) {
                inverse[i] = 1.0F / (alpha[i] + ggml_bf16_to_fp32(ggml_fp32_to_bf16(1.0e-9F)));
            }
            const auto snake_shape = engine::core::TensorShape::from_dims({1, 64, 1});
            const engine::modules::Snake1dWeights fused_weights{
                store.make_f32(snake_shape, std::move(alpha)),
                store.make_f32(snake_shape, std::move(inverse))};
            store.upload();
            const auto input_shape = source->require_metadata("input").shape;
            const int64_t frames = input_shape.at(2) - 6;
            const auto output_shape = engine::core::TensorShape::from_dims({1, 64, frames});
            const auto input = engine::core::make_tensor(ctx, GGML_TYPE_F32,
                engine::core::TensorShape::from_dims({1, 1, input_shape.at(2)}));
            ggml_set_input(input.tensor);
            auto current = engine::modules::Conv1dModule({1, 64, 7, 1, 0, 1, true}).build(ctx, input, weights);
            current = engine::core::wrap_tensor(ggml_cast(ctx.ggml, current.tensor, GGML_TYPE_BF16), output_shape, GGML_TYPE_BF16);
            const auto candidate = engine::modules::Conv1dModule({1, 64, 7, 1, 0, 1, true, true}).build(ctx, input, weights);
            auto snake = engine::modules::Snake1dModule({64}).build(ctx, candidate, snake_weights);
            snake = engine::core::wrap_tensor(ggml_round_bf16(ctx.ggml, snake.tensor), output_shape, GGML_TYPE_F32);
            auto fused_snake = engine::modules::Snake1dModule({64}).build(ctx, candidate, fused_weights);
            fused_snake = engine::core::wrap_tensor(ggml_round_bf16(ctx.ggml, fused_snake.tensor), output_shape, GGML_TYPE_F32);
            ggml_set_output(current.tensor);
            ggml_set_output(candidate.tensor);
            ggml_set_output(snake.tensor);
            ggml_set_output(fused_snake.tensor);
            auto * graph = ggml_new_graph_custom(ctx.ggml, 2048, false);
            ggml_build_forward_expand(graph, current.tensor);
            ggml_build_forward_expand(graph, candidate.tensor);
            ggml_build_forward_expand(graph, snake.tensor);
            ggml_build_forward_expand(graph, fused_snake.tensor);
            std::unique_ptr<std::remove_pointer_t<ggml_gallocr_t>, decltype(&ggml_gallocr_free)> allocator(
                ggml_gallocr_new(ggml_backend_get_default_buffer_type(execution.backend())), ggml_gallocr_free);
            if (!allocator || !ggml_gallocr_alloc_graph(allocator.get(), graph))
                throw std::runtime_error("first-conv graph allocation failed");
            engine::core::write_tensor_f32(input, source->require_f32("input"));
            if (engine::core::compute_backend_graph(execution.backend(), graph) != GGML_STATUS_SUCCESS)
                throw std::runtime_error("first-conv graph execution failed");
            ggml_backend_synchronize(execution.backend());
            engine::io::write_safetensors_file(argv[3], {
                {"current", "F32", {1, 64, frames}, bytes_for(engine::core::read_tensor_bf16(current.tensor))},
                {"bf16", "F32", {1, 64, frames}, bytes_for(engine::core::read_tensor_f32(candidate.tensor))},
                {"snake", "F32", {1, 64, frames}, bytes_for(engine::core::read_tensor_f32(snake.tensor))},
                {"fused_snake", "F32", {1, 64, frames}, bytes_for(engine::core::read_tensor_f32(fused_snake.tensor))}});
            engine::core::release_backend_graph_resources(execution.backend(), graph);
            engine::debug::timing_log_scalar("session.wall_ms", std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - started).count());
            return 0;
        }
        if (argc > 1 && std::string(argv[1]) == "--roundtrip") {
            if (argc < 8 || std::string(argv[7]) != "--log")
                throw std::runtime_error("usage: --roundtrip fish_audio|fish_drama3 cpu|cuda MODEL OUTPUT REFERENCES --log [--weight-type native|f32|f16] [--encoder-bf16]");
            engine::debug::configure_logging({true, std::nullopt});
            engine::core::BackendConfig backend;
            const std::string backend_name = argv[3];
            if (backend_name == "cpu") backend.type = engine::core::BackendType::Cpu;
            else if (backend_name == "cuda") backend.type = engine::core::BackendType::Cuda;
            else throw std::runtime_error("roundtrip backend must be cpu or cuda");
            backend.threads = 8;
            engine::core::ExecutionContext execution(backend);
            std::shared_ptr<const assets::TensorSource> source;
            engine::codecs::FishDacCodecConfig config;
            const std::string family = argv[2];
            if (family == "fish_audio") {
                const auto resources = engine::model_spec::load_resource_bundle(
                    argv[4], engine::model_spec::default_spec_path(family));
                source = resources.open_tensor_source("codec_weights");
                config.total_codebooks = engine::io::json::require_i64(
                    resources.parse_json("config").require("audio_decoder_config"), "num_codebooks");
            } else if (family == "fish_drama3") {
                const auto model = engine::models::fish_drama3::load_drama3_assets(argv[4]);
                source = model->codec_weights;
                config = model->codec;
            } else throw std::runtime_error("unsupported codec family");
            engine::codecs::FishDacCodecRuntimeOptions options;
            options.weight_context_bytes = 16 * 1024 * 1024;
            for (int i = 8; i < argc; ++i) {
                const std::string option = argv[i];
                if (option == "--encoder-bf16") {
                    options.encoder_bf16_compute = true;
                } else if (option == "--weight-type" && i + 1 < argc) {
                    const std::string type = argv[++i];
                    if (type != "native" && type != "f32" && type != "f16") throw std::runtime_error("roundtrip weight type must be native, f32, or f16");
                    options.matmul_weight_storage_type = options.conv_weight_storage_type = assets::parse_tensor_storage_type(type);
                } else {
                    throw std::runtime_error("unknown roundtrip option: " + option);
                }
            }
            std::filesystem::create_directories(argv[5]);
            std::vector<engine::io::SafeTensorWriteEntry> anchors;
            for (const char * name : {"encoder.block.0.conv.weight", "encoder.block.1.block.0.block.0.alpha",
                     "encoder.block.4.block.5.layers.0.attention.q_proj.weight",
                     "quantizer.pre_module.norm.weight", "quantizer.semantic_quantizer.quantizers.0.in_proj.weight",
                     "quantizer.semantic_quantizer.quantizers.0.codebook.weight",
                     "quantizer.quantizer.quantizers.0.codebook.weight"}) {
                anchors.push_back({name, "F32", source->require_metadata(name).shape, bytes_for(source->require_f32(name))});
            }
            for (int64_t book = 0; book < config.total_codebooks; ++book) {
                const std::string prefix = book == 0 ? "quantizer.semantic_quantizer.quantizers.0"
                    : "quantizer.quantizer.quantizers." + std::to_string(book - 1);
                for (const char * suffix : {"in_proj.weight", "in_proj.bias", "out_proj.weight", "out_proj.bias", "codebook.weight"}) {
                    const std::string name = prefix + "." + suffix;
                    if (book == 0 && (std::string(suffix) == "in_proj.weight" || std::string(suffix) == "codebook.weight")) continue;
                    if (book == 1 && std::string(suffix) == "codebook.weight") continue;
                    anchors.push_back({name, "F32", source->require_metadata(name).shape, bytes_for(source->require_f32(name))});
                }
            }
            engine::io::write_safetensors_file(std::filesystem::path(argv[5]) / "source-anchors.safetensors", anchors);
            const auto component = engine::codecs::FishDacCodecComponent::load_from_tensor_source(
                source, config, {}, execution.backend(), execution.backend_type(), options);
            engine::codecs::FishDacCodecRuntime codec(component, backend, backend.threads, options.graph_arena_bytes);
            const std::filesystem::path output = argv[5], references = argv[6];
            engine::audio::WavWriteOptions write_options;
            write_options.format = engine::audio::WavSampleFormat::Float32;
            for (int pass = 0; pass < 2; ++pass) {
                const auto started = std::chrono::steady_clock::now();
                const std::string name = "a";
                const std::string label = pass == 0 ? name : name + "-warm";
                const auto wav = engine::audio::read_wav_f32(references / ("input-" + name + ".wav"));
                if (wav.sample_rate != config.sample_rate || wav.channels != 1)
                    throw std::runtime_error("roundtrip requires mono input at the codec sample rate");
                const auto encode_started = std::chrono::steady_clock::now();
                const auto codes = codec.encode_codes({wav.sample_rate, wav.channels, wav.samples});
                const auto decode_started = std::chrono::steady_clock::now();
                const auto audio = codec.decode_codes(codes);
                const auto decode_finished = std::chrono::steady_clock::now();
                const auto directory = output / label;
                std::filesystem::create_directories(directory);
                engine::io::write_safetensors_file(directory / "codes.safetensors",
                    {{"codes", "I32", {codes.codebooks, codes.frames}, bytes_for(codes.codes)}});
                engine::audio::write_wav(directory / "roundtrip.wav", audio.sample_rate, audio.channels, audio.samples, write_options);
                const auto python = assets::open_tensor_source(references / (family + "-python-" + name + ".safetensors"));
                const auto data = python->require_tensor_data("codes");
                if (data.metadata.dtype != "I32" || data.metadata.shape != std::vector<int64_t>{codes.codebooks, codes.frames} ||
                    data.bytes.size() != codes.codes.size() * sizeof(int32_t))
                    throw std::runtime_error("Python reference codes must be I32 and match encoder dimensions");
                engine::codecs::FishDacCodes python_codes;
                python_codes.codebooks = codes.codebooks;
                python_codes.frames = codes.frames;
                python_codes.codes.resize(codes.codes.size());
                std::memcpy(python_codes.codes.data(), data.bytes.data(), data.bytes.size());
                const auto decoded = codec.decode_codes(python_codes);
                engine::audio::write_wav(directory / "python-codes.wav", decoded.sample_rate, decoded.channels, decoded.samples, write_options);
                engine::debug::trace_log_scalar("fish_codec.reference", label);
                engine::debug::timing_log_scalar("fish_codec.encode_ms", std::chrono::duration<double, std::milli>(decode_started - encode_started).count());
                engine::debug::timing_log_scalar("fish_codec.decode_ms", std::chrono::duration<double, std::milli>(decode_finished - decode_started).count());
                engine::debug::trace_log_i32("fish_codec.codes", {codes.codebooks, codes.frames}, codes.codes);
                engine::debug::timing_log_scalar("session.wall_ms", std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - started).count());
            }
            return 0;
        }
        FixtureDirectory directory;
        const std::string prefix = "decoder.transformer.layers.0.attention.";
        const std::string packed_name = prefix + "wqkv.weight";
        const char * projections[] = {"q_proj", "k_proj", "v_proj"};
        constexpr size_t width = 8;
        constexpr size_t count = width * width;
        std::vector<float> values(3 * count);
        for (size_t i = 0; i < values.size(); ++i) values[i] = (static_cast<float>(i) - 91) / 128;
        std::vector<ggml_fp16_t> half(values.size());
        std::vector<ggml_bf16_t> bfloat(values.size());
        ggml_fp32_to_fp16_row(values.data(), half.data(), values.size());
        ggml_fp32_to_bf16_row(values.data(), bfloat.data(), values.size());
        for (const auto & fixture : std::vector<std::pair<std::string, std::vector<unsigned char>>>{
            {"F32", bytes_for(values)}, {"F16", bytes_for(half)}, {"BF16", bytes_for(bfloat)}}) {
            const auto packed_path = directory.path / (fixture.first + "-packed.safetensors");
            const auto split_path = directory.path / (fixture.first + "-split.safetensors");
            engine::io::write_safetensors_file(packed_path, {{packed_name, fixture.first, {3 * width, width}, fixture.second}});
            std::vector<engine::io::SafeTensorWriteEntry> entries;
            const auto byte_count = fixture.second.size() / 3;
            for (size_t part = 0; part < 3; ++part) {
                const auto begin = fixture.second.begin() + part * byte_count;
                entries.push_back({prefix + projections[part] + ".weight", fixture.first, {width, width}, {begin, begin + byte_count}});
            }
            engine::io::write_safetensors_file(split_path, entries);
            const auto original = assets::open_tensor_source(packed_path);
            const auto adapted = engine::models::fish_drama3::make_drama3_codec_source(original);
            const auto split = assets::open_tensor_source(split_path);
            if (adapted->source_path() != original->source_path() || adapted->has_tensor("missing") || adapted->optional_f32("missing"))
                throw std::runtime_error("codec source delegation failed");
            if (adapted->require_tensor_data(packed_name).bytes != original->require_tensor_data(packed_name).bytes)
                throw std::runtime_error("packed source changed");
            for (const auto * projection : projections) {
                const auto name = prefix + projection + ".weight";
                const auto metadata = adapted->require_metadata(name);
                if (!adapted->has_tensor(name) || metadata.shape != std::vector<int64_t>{width, width} ||
                    metadata.dtype != split->require_metadata(name).dtype || metadata.name != name ||
                    adapted->require_tensor(name, assets::TensorStorageType::Native, {width, width}).bytes !=
                        split->require_tensor(name, assets::TensorStorageType::Native, {width, width}).bytes ||
                    adapted->require_f32(name) != split->require_f32(name)) {
                    throw std::runtime_error("codec slice differs from an independent split checkpoint");
                }
                require_failure([&] { (void) adapted->require_f32(name, std::vector<int64_t>{1, width}); });
            }
            const auto passthrough = engine::models::fish_drama3::make_drama3_codec_source(split);
            if (passthrough->tensors().size() != split->tensors().size()) throw std::runtime_error("split checkpoint changed");
            for (const auto & entry : entries) {
                if (passthrough->require_tensor_data(entry.name).bytes != split->require_tensor_data(entry.name).bytes)
                    throw std::runtime_error("split checkpoint bytes changed");
            }
            entries.push_back({packed_name, fixture.first, {3 * width, width}, fixture.second});
            const auto both_path = directory.path / (fixture.first + "-both.safetensors");
            engine::io::write_safetensors_file(both_path, entries);
            const auto both = engine::models::fish_drama3::make_drama3_codec_source(assets::open_tensor_source(both_path));
            if (both->tensors().size() != entries.size()) throw std::runtime_error("complete split weights were shadowed");
            entries.erase(entries.begin() + 1, entries.begin() + 3);
            const auto partial_path = directory.path / (fixture.first + "-partial.safetensors");
            engine::io::write_safetensors_file(partial_path, entries);
            require_failure([&] { (void) engine::models::fish_drama3::make_drama3_codec_source(assets::open_tensor_source(partial_path)); });
        }
        const auto invalid_path = directory.path / "invalid-shape.safetensors";
        engine::io::write_safetensors_file(invalid_path, {{packed_name, "F32", {3, width, width}, bytes_for(values)}});
        require_failure([&] { (void) engine::models::fish_drama3::make_drama3_codec_source(assets::open_tensor_source(invalid_path)); });
        require_failure([] { (void) engine::models::fish_drama3::make_drama3_codec_source(nullptr); });
        std::cout << "F32/F16/BF16 packed bytes and decoded values match split checkpoints; split passthrough and invalid inputs passed\n";
        // Optional real-package check needs no GPU and never loads backbone weights.
        for (int argument = 1; argument < argc; ++argument) {
            const auto model = engine::models::fish_drama3::load_drama3_assets(argv[argument]);
            const auto original = model->resources.open_tensor_source("codec_weights");
            const std::string suffix = ".attention.wqkv.weight";
            size_t checked = 0;
            for (const auto & tensor : original->tensors()) {
                if (tensor.name.size() < suffix.size() ||
                    tensor.name.compare(tensor.name.size() - suffix.size(), suffix.size(), suffix) != 0) continue;
                const auto packed = original->require_f32(tensor.name);
                const auto part_size = packed.size() / 3;
                const auto layer = tensor.name.substr(0, tensor.name.size() - suffix.size()) + ".attention.";
                for (size_t part = 0; part < 3; ++part) {
                    const auto name = layer + projections[part] + ".weight";
                    const auto values = model->codec_weights->require_f32(name);
                    if (values.size() != part_size || !std::equal(values.begin(), values.end(), packed.begin() + part * part_size))
                        throw std::runtime_error("real codec slice differs from the original packed checkpoint");
                    ++checked;
                }
            }
            if (checked == 0) throw std::runtime_error("real checkpoint has no packed codec weights");
            std::cout << "real_package=" << argv[argument] << " exact_QKV_slices=" << checked << '\n';
        }
    } catch (const std::exception & error) { std::cerr << error.what() << '\n'; return 1; }
}
