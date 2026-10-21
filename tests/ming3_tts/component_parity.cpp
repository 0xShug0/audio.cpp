#include "engine/framework/codecs/ming_audio_vae_runtime.h"
#include "engine/framework/modules/speech_decoders/ming_flow_runtime.h"
#include "engine/models/ming3_tts/generator.h"
#include "engine/models/ming3_tts/prompt.h"
#include "engine/framework/audio/wav_writer.h"
#include "engine/framework/audio/istft_graph.h"
#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/debug/trace.h"
#include "engine/framework/io/safetensors.h"
#include "engine/framework/modules/speaker_encoders/campplus_encoder.h"
#include "engine/framework/modules/activation_modules.h"
#include "engine/framework/modules/norm_modules.h"
#include "engine/framework/modules/primitive_modules.h"
#include "engine/framework/modules/positional_modules.h"
#include "engine/framework/modules/structural_modules.h"
#include "engine/framework/modules/weight_binding.h"

#include <ggml-alloc.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <utility>

int main(int argc, char ** argv) try {
    std::filesystem::path weights, config_path, reference, output;
    std::filesystem::path requests_path, tokenizer_dir;
    std::filesystem::path speaker_weights;
    std::filesystem::path generated_fixture;
    std::string request_name;
    bool log = false;
    bool flow = false;
    bool generate = false;
    bool reference_history = false;
    bool flow_unit = false;
    bool prompt_only = false;
    bool encode = false;
    bool istft_only = false;
    bool head_only = false;
    bool input_only = false;
    auto attention = engine::core::AttentionPreference::Auto;
    engine::core::BackendConfig backend;
    backend.type = engine::core::BackendType::Cuda;
    backend.threads = 8;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--log") log = true;
        else if (arg == "--flow") flow = true;
        else if (arg == "--generate") generate = true;
        else if (arg == "--reference-history") reference_history = true;
        else if (arg == "--flow-unit") flow_unit = true;
        else if (arg == "--prompt-only") prompt_only = true;
        else if (arg == "--encode") encode = true;
        else if (arg == "--istft-only") istft_only = true;
        else if (arg == "--head-only") head_only = true;
        else if (arg == "--input-only") input_only = true;
        else if (arg == "--generated-fixture" && i + 1 < argc) generated_fixture = argv[++i];
        else if (arg == "--speaker-weights" && i + 1 < argc) speaker_weights = argv[++i];
        else if (arg == "--requests" && i + 1 < argc) requests_path = argv[++i];
        else if (arg == "--tokenizer-dir" && i + 1 < argc) tokenizer_dir = argv[++i];
        else if (arg == "--request-name" && i + 1 < argc) request_name = argv[++i];
        else if (arg == "--attention" && i + 1 < argc) {
            attention = engine::core::parse_attention_preference(argv[++i], "--attention");
        }
        else if (arg == "--weights" && i + 1 < argc) weights = argv[++i];
        else if (arg == "--config" && i + 1 < argc) config_path = argv[++i];
        else if (arg == "--reference" && i + 1 < argc) reference = argv[++i];
        else if (arg == "--output" && i + 1 < argc) output = argv[++i];
        else throw std::runtime_error("unknown or incomplete argument: " + arg);
    }
    if (!log || weights.empty() || config_path.empty() || reference.empty()) {
        throw std::runtime_error("usage: ming3_tts_component_parity --weights <safetensors> --config <json> --reference <safetensors> --output <wav> --log");
    }
    if (!generated_fixture.empty() && !generate) throw std::runtime_error("--generated-fixture requires --generate");
    engine::debug::configure_logging({true, std::nullopt});
    const auto fixture = engine::assets::open_tensor_source(reference);
    if (fixture->has_tensor("sampler.timesteps")) {
        engine::debug::trace_log_f32("ming3_tts.reference.timesteps", {11}, fixture->require_f32("sampler.timesteps"));
    }
    if (fixture->has_tensor("decode.0.spectrum")) {
        const auto shape = fixture->require_metadata("decode.0.spectrum").shape;
        engine::debug::trace_log_f32("ming3_tts.reference.spectrum", shape,
            fixture->require_f32("decode.0.spectrum"));
        const auto envelope = fixture->require_f32("decode.0.envelope");
        engine::debug::trace_log_f32("ming3_tts.reference.envelope", {static_cast<int64_t>(envelope.size())}, envelope);
        for (const auto * key : {"input_embeddings", "hidden"}) {
            const auto name = std::string("decode.0.") + key;
            if (fixture->has_tensor(name)) {
                engine::debug::trace_log_f32(std::string("ming3_tts.reference.") + key,
                    fixture->require_metadata(name).shape, fixture->require_f32(name));
            }
        }
    }
    const auto config = engine::io::json::parse_file(config_path);
    const auto model_config = engine::models::ming3_tts::parse_ming_config(config);
    const auto latent_dim = config.require("audio_tokenizer_config").require("enc_kwargs").require("latent_dim").as_i64();
    engine::core::ExecutionContext execution(backend);
    if (input_only) {
        const auto shape = fixture->require_metadata("decode.0.latent").shape;
        const auto & codec = config.require("audio_tokenizer_config");
        const int64_t hidden = codec.require("dec_kwargs").require("backbone").require("hidden_size").as_i64();
        const auto & backbone = codec.require("dec_kwargs").require("backbone");
        const auto heads = backbone.require("num_attention_heads").as_i64();
        const auto head_dim = hidden / heads;
        const auto theta = backbone.require("rope_theta").as_f32();
        const int64_t frames = shape.at(1) * codec.require("patch_size").as_i64();
        engine::core::BackendWeightStore store(execution.backend(), execution.backend_type(), "ming3_tts.input.weights", 1024 * 1024);
        const auto source = engine::assets::open_tensor_source(weights);
        const auto projection = engine::modules::binding::linear_from_source(store, *source, "audio.decoder.fc1",
            engine::assets::TensorStorageType::Native, hidden, shape.at(2), true);
        const auto norm = engine::modules::binding::norm_weight_from_source(store, *source,
            "audio.decoder.decoder.layers.0.input_layernorm", hidden);
        const auto q_weight = engine::modules::binding::linear_from_source(store, *source,
            "audio.decoder.decoder.layers.0.self_attn.q_proj", engine::assets::TensorStorageType::Native,
            hidden, hidden, true);
        std::vector<float> frequencies(head_dim / 2);
        for (size_t i = 0; i < frequencies.size(); ++i) frequencies[i] = 1.0f / std::pow(theta, static_cast<float>(2 * i) / head_dim);
        auto rounded_frequencies = frequencies;
        engine::core::round_f32_to_bf16_in_place(rounded_frequencies);
        for (size_t i = 0; i < frequencies.size(); ++i) frequencies[i] /= rounded_frequencies[i];
        const auto rope_factors = store.make_f32(engine::core::TensorShape::from_dims({head_dim / 2}), frequencies);
        store.upload();
        std::unique_ptr<ggml_context, decltype(&ggml_free)> context(ggml_init({1024 * 1024, nullptr, true}), ggml_free);
        engine::core::ModuleBuildContext ctx{context.get(), "ming3_tts.input", execution.backend_type()};
        const auto input = engine::core::make_tensor(ctx, GGML_TYPE_F32,
            engine::core::TensorShape::from_dims({1, shape.at(1), shape.at(2)}));
        ggml_set_input(input.tensor);
        auto x = engine::modules::LinearModule({shape.at(2), hidden, true}).build(ctx, input, projection);
        x = engine::modules::Bf16RoundModule{}.build(ctx, x);
        x = engine::modules::TransposeModule({{0, 2, 1}, 3}).build(ctx, x);
        x = engine::modules::Interpolate1dModule({frames, engine::modules::Interpolate1dMode::Linear}).build(ctx, x);
        x = engine::modules::TransposeModule({{0, 2, 1}, 3}).build(ctx, x);
        x = engine::core::wrap_tensor(ggml_cont(ctx.ggml, x.tensor), x.shape, x.type);
        ggml_set_output(x.tensor);
        const auto normalized = engine::modules::RMSNormModule({hidden,
            codec.require("dec_kwargs").require("backbone").require("rms_norm_eps").as_f32(), true, false})
            .build(ctx, x, norm);
        ggml_set_output(normalized.tensor);
        const auto pixel_norm = engine::modules::PixelNormModule({2,
            codec.require("dec_kwargs").require("backbone").require("rms_norm_eps").as_f32()}).build(ctx, x);
        const auto explicit_norm = engine::core::wrap_tensor(ggml_mul(ctx.ggml, pixel_norm.tensor, norm.weight->tensor),
            x.shape, GGML_TYPE_F32);
        ggml_set_output(explicit_norm.tensor);
        const auto q_input = engine::core::make_tensor(ctx, GGML_TYPE_F32,
            engine::core::TensorShape::from_dims({1, frames, hidden}));
        ggml_set_input(q_input.tensor);
        engine::modules::LinearConfig q_config{hidden, hidden, true};
        const auto legacy_q = engine::modules::Bf16RoundModule{}.build(ctx,
            engine::modules::LinearModule(q_config).build(ctx, q_input, q_weight));
        const auto projected_input = engine::core::make_tensor(ctx, GGML_TYPE_F32,
            engine::core::TensorShape::from_dims({1, frames, hidden}));
        ggml_set_input(projected_input.tensor);
        const auto positions = engine::core::make_tensor(ctx, GGML_TYPE_I32, engine::core::TensorShape::from_dims({frames}));
        ggml_set_input(positions.tensor);
        const auto q_heads = engine::core::reshape_tensor(ctx, projected_input,
            engine::core::TensorShape::from_dims({1, frames, heads, head_dim}));
        engine::modules::RoPEConfig rope{head_dim, GGML_ROPE_TYPE_NEOX, theta};
        const auto fused_rotary = engine::modules::RoPEModule(rope).build(ctx, q_heads, positions, &rope_factors);
        rope.intermediate_round_type = GGML_TYPE_F32;
        const auto split_rotary = engine::modules::RoPEModule(rope).build(ctx, q_heads, positions, &rope_factors);
        const auto fused_q_rotary = engine::core::ensure_backend_addressable_layout(ctx,
            engine::modules::TransposeModule({{0, 2, 1, 3}, 4}).build(ctx, fused_rotary));
        const auto split_q_rotary = engine::core::ensure_backend_addressable_layout(ctx,
            engine::modules::TransposeModule({{0, 2, 1, 3}, 4}).build(ctx, split_rotary));
        const auto table_shape = engine::core::TensorShape::from_dims({1, 1, frames, head_dim / 2});
        const auto cosine = engine::core::make_tensor(ctx, GGML_TYPE_F32, table_shape);
        const auto sine = engine::core::make_tensor(ctx, GGML_TYPE_F32, table_shape);
        const auto host_cosine = engine::core::make_tensor(ctx, GGML_TYPE_F32, table_shape);
        const auto host_sine = engine::core::make_tensor(ctx, GGML_TYPE_F32, table_shape);
        for (const auto & value : {cosine, sine, host_cosine, host_sine}) ggml_set_input(value.tensor);
        const auto attention_q = engine::modules::TransposeModule({{0, 2, 1, 3}, 4}).build(ctx, q_heads);
        const engine::modules::RepeatModule broadcast({engine::core::TensorShape::from_dims({1, heads, frames, head_dim / 2})});
        const auto reference_table_rotary = engine::modules::SplitRoPEModule({head_dim}).build(ctx, attention_q,
            broadcast.build(ctx, cosine), broadcast.build(ctx, sine));
        const auto host_table_rotary = engine::modules::SplitRoPEModule({head_dim}).build(ctx, attention_q,
            broadcast.build(ctx, host_cosine), broadcast.build(ctx, host_sine));
        ggml_set_output(fused_q_rotary.tensor);
        ggml_set_output(split_q_rotary.tensor);
        ggml_set_output(reference_table_rotary.tensor);
        ggml_set_output(host_table_rotary.tensor);
        const auto gate_shape = fixture->require_metadata("decode.0.mlp_gate.output").shape;
        const auto mlp_shape = engine::core::TensorShape::from_dims({1, gate_shape.at(1), gate_shape.at(2)});
        const auto gate_input = engine::core::make_tensor(ctx, GGML_TYPE_F32, mlp_shape);
        const auto up_input = engine::core::make_tensor(ctx, GGML_TYPE_F32, mlp_shape);
        ggml_set_input(gate_input.tensor);
        ggml_set_input(up_input.tensor);
        const auto activated = engine::modules::Bf16RoundModule{}.build(ctx,
            engine::modules::SiluModule{}.build(ctx, gate_input));
        const auto gated = engine::modules::Bf16RoundModule{}.build(ctx,
            engine::modules::MulModule{}.build(ctx, activated, up_input));
        ggml_set_output(gated.tensor);
        ggml_set_output(legacy_q.tensor);
        auto * graph = ggml_new_graph_custom(context.get(), 256, false);
        ggml_build_forward_expand(graph, x.tensor);
        ggml_build_forward_expand(graph, normalized.tensor);
        ggml_build_forward_expand(graph, explicit_norm.tensor);
        ggml_build_forward_expand(graph, legacy_q.tensor);
        ggml_build_forward_expand(graph, gated.tensor);
        ggml_build_forward_expand(graph, fused_q_rotary.tensor);
        ggml_build_forward_expand(graph, split_q_rotary.tensor);
        ggml_build_forward_expand(graph, reference_table_rotary.tensor);
        ggml_build_forward_expand(graph, host_table_rotary.tensor);
        std::unique_ptr<ggml_gallocr, decltype(&ggml_gallocr_free)> allocator(
            ggml_gallocr_new(ggml_backend_get_default_buffer_type(execution.backend())), ggml_gallocr_free);
        if (!ggml_gallocr_reserve(allocator.get(), graph) || !ggml_gallocr_alloc_graph(allocator.get(), graph)) {
            throw std::runtime_error("Ming isolated input projection allocation failed");
        }
        engine::core::HostGraphPlan plan;
        engine::core::prepare_host_graph_plan(execution, graph, plan);
        engine::core::write_tensor_f32(input, fixture->require_f32("decode.0.latent"));
        engine::core::write_tensor_f32(q_input, fixture->require_f32("decode.0.qkv_input"));
        engine::core::write_tensor_f32(projected_input, fixture->require_f32("decode.0.q_projection"));
        engine::core::write_tensor_f32(gate_input, fixture->require_f32("decode.0.mlp_gate.output"));
        engine::core::write_tensor_f32(up_input, fixture->require_f32("decode.0.mlp_up.output"));
        std::vector<int32_t> position_ids(frames);
        for (int64_t i = 0; i < frames; ++i) position_ids[i] = static_cast<int32_t>(i);
        ggml_backend_tensor_set(positions.tensor, position_ids.data(), 0, position_ids.size() * sizeof(int32_t));
        auto reference_cosine = fixture->require_f32("decode.0.rotary.cos");
        auto reference_sine = fixture->require_f32("decode.0.rotary.sin");
        std::vector<float> compact_cosine(frames * head_dim / 2), compact_sine(compact_cosine.size());
        std::vector<float> host_cos(compact_cosine.size()), host_sin(compact_cosine.size());
        for (int64_t t = 0; t < frames; ++t) {
            for (int64_t d = 0; d < head_dim / 2; ++d) {
                const auto index = t * head_dim / 2 + d;
                compact_cosine[index] = reference_cosine.at(t * head_dim + d);
                compact_sine[index] = reference_sine.at(t * head_dim + d);
                const float angle = static_cast<float>(t) * rounded_frequencies[d];
                host_cos[index] = std::cos(angle);
                host_sin[index] = std::sin(angle);
            }
        }
        engine::core::write_tensor_f32(cosine, compact_cosine);
        engine::core::write_tensor_f32(sine, compact_sine);
        engine::core::write_tensor_f32(host_cosine, host_cos);
        engine::core::write_tensor_f32(host_sine, host_sin);
        const auto start = std::chrono::steady_clock::now();
        const auto status = engine::core::compute_graph(execution, graph, plan, "ming3_tts.input");
        const auto actual = engine::core::read_tensor_f32(x.tensor);
        auto actual_norm = engine::core::read_tensor_f32(normalized.tensor);
        auto actual_explicit_norm = engine::core::read_tensor_f32(explicit_norm.tensor);
        const auto actual_legacy_q = engine::core::read_tensor_f32(legacy_q.tensor);
        const auto actual_gated = engine::core::read_tensor_f32(gated.tensor);
        auto actual_fused_rotary = engine::core::read_tensor_f32(fused_q_rotary.tensor);
        auto actual_split_rotary = engine::core::read_tensor_f32(split_q_rotary.tensor);
        auto actual_reference_rotary = engine::core::read_tensor_f32(reference_table_rotary.tensor);
        auto actual_host_rotary = engine::core::read_tensor_f32(host_table_rotary.tensor);
        engine::core::release_backend_graph_resources(execution.backend(), graph);
        if (status != GGML_STATUS_SUCCESS) throw std::runtime_error("Ming isolated input projection failed");
        const auto expected = fixture->require_f32("decode.0.input_embeddings");
        if (actual.size() != expected.size()) throw std::runtime_error("Ming decoder input length mismatch");
        size_t mismatched = 0;
        double max_error = 0;
        for (size_t i = 0; i < actual.size(); ++i) {
            mismatched += actual[i] != expected[i];
            max_error = std::max(max_error, std::abs(static_cast<double>(actual[i]) - expected[i]));
        }
        engine::debug::timing_log_scalar("session.wall_ms", engine::debug::elapsed_ms(start));
        engine::debug::trace_log_scalar("ming3_tts.parity.input.different_elements", mismatched);
        engine::debug::trace_log_scalar("ming3_tts.parity.input.max_error", max_error);
        auto expected_norm = fixture->require_f32("decode.0.qkv_input");
        if (actual_norm.size() != expected_norm.size()) throw std::runtime_error("Ming normalized input length mismatch");
        const auto raw_actual_norm = actual_norm;
        const auto raw_expected_norm = expected_norm;
        max_error = 0;
        for (size_t i = 0; i < actual_norm.size(); ++i) {
            max_error = std::max(max_error, std::abs(static_cast<double>(actual_norm[i]) - expected_norm[i]));
        }
        engine::debug::trace_log_scalar("ming3_tts.parity.input_norm.max_error", max_error);
        engine::core::round_f32_to_bf16_in_place(actual_norm);
        engine::core::round_f32_to_bf16_in_place(expected_norm);
        size_t norm_mismatched = 0;
        for (size_t i = 0; i < actual_norm.size(); ++i) norm_mismatched += actual_norm[i] != expected_norm[i];
        engine::debug::trace_log_scalar("ming3_tts.parity.input_norm.bf16_different_elements", norm_mismatched);
        if (norm_mismatched) {
            const auto different = std::mismatch(actual_norm.begin(), actual_norm.end(), expected_norm.begin());
            const auto index = static_cast<size_t>(different.first - actual_norm.begin());
            engine::debug::trace_log_scalar("ming3_tts.parity.input_norm.first_index", index);
            engine::debug::trace_log_scalar("ming3_tts.parity.input_norm.first_actual", static_cast<double>(raw_actual_norm[index]));
            engine::debug::trace_log_scalar("ming3_tts.parity.input_norm.first_expected", static_cast<double>(raw_expected_norm[index]));
        }
        engine::core::round_f32_to_bf16_in_place(actual_explicit_norm);
        size_t explicit_mismatched = 0;
        for (size_t i = 0; i < actual_explicit_norm.size(); ++i) explicit_mismatched += actual_explicit_norm[i] != expected_norm[i];
        engine::debug::trace_log_scalar("ming3_tts.parity.explicit_norm.bf16_different_elements", explicit_mismatched);
        const auto expected_q = fixture->require_f32("decode.0.q_projection");
        if (actual_legacy_q.size() != expected_q.size()) {
            throw std::runtime_error("Ming Q-projection length mismatch");
        }
        size_t legacy_mismatched = 0;
        for (size_t i = 0; i < expected_q.size(); ++i) {
            legacy_mismatched += actual_legacy_q[i] != expected_q[i];
        }
        engine::debug::trace_log_scalar("ming3_tts.parity.q_projection.legacy_different_elements", legacy_mismatched);
        auto expected_rotary = fixture->require_f32("decode.0.q_rotary");
        engine::core::round_f32_to_bf16_in_place(expected_rotary);
        for (auto item : {std::make_pair("fused", &actual_fused_rotary), std::make_pair("split", &actual_split_rotary),
                std::make_pair("reference_table", &actual_reference_rotary), std::make_pair("host_table", &actual_host_rotary)}) {
            if (item.second->size() != expected_rotary.size()) throw std::runtime_error("Ming rotary shape mismatch");
            engine::core::round_f32_to_bf16_in_place(*item.second);
            size_t different = 0;
            double error = 0;
            for (size_t i = 0; i < expected_rotary.size(); ++i) {
                different += item.second->at(i) != expected_rotary[i];
                error = std::max(error, std::abs(static_cast<double>(item.second->at(i)) - expected_rotary[i]));
            }
            const auto prefix = std::string("ming3_tts.parity.rotary.") + item.first;
            engine::debug::trace_log_scalar(prefix + ".bf16_different_elements", different);
            engine::debug::trace_log_scalar(prefix + ".max_error", error);
        }
        const auto expected_gated = fixture->require_f32("decode.0.mlp_down.input");
        if (actual_gated.size() != expected_gated.size()) throw std::runtime_error("Ming MLP activation length mismatch");
        size_t activation_mismatched = 0;
        for (size_t i = 0; i < actual_gated.size(); ++i) activation_mismatched += actual_gated[i] != expected_gated[i];
        engine::debug::trace_log_scalar("ming3_tts.parity.mlp_activation.different_elements", activation_mismatched);
        return mismatched == 0 ? 0 : 1;
    }
    if (!speaker_weights.empty()) {
        engine::modules::CampplusEncoderConfig speaker_config;
        speaker_config.weight_layout = engine::modules::CampplusEncoderWeightLayout::Fused;
        speaker_config.normalize_partial_segment_by_full_length = true;
        auto encoder = engine::modules::CampplusEncoderComponent::load_from_tensor_source(
            engine::assets::open_tensor_source(speaker_weights), backend, speaker_config);
        const auto shape = fixture->require_metadata("speaker.features").shape;
        const auto start = std::chrono::steady_clock::now();
        const auto actual = encoder.embed_from_features(fixture->require_f32("speaker.features"), shape.at(0), shape.at(1));
        const auto expected = fixture->require_f32("speaker.embedding");
        engine::debug::timing_log_scalar("session.wall_ms", engine::debug::elapsed_ms(start));
        if (actual.embedding.size() != expected.size()) throw std::runtime_error("Ming speaker embedding dimension mismatch");
        double dot = 0, aa = 0, bb = 0, squared_error = 0;
        for (size_t i = 0; i < expected.size(); ++i) {
            const double a = actual.embedding[i], b = expected[i];
            dot += a * b; aa += a * a; bb += b * b; squared_error += (a - b) * (a - b);
        }
        const double cosine = dot / std::sqrt(aa * bb);
        engine::debug::trace_log_scalar("ming3_tts.parity.speaker.cosine", cosine);
        engine::debug::trace_log_scalar("ming3_tts.parity.speaker.rmse", std::sqrt(squared_error / expected.size()));
        return cosine > 0.99999 ? 0 : 1;
    }
    std::optional<engine::models::ming3_tts::MingPrompt> native_prompt;
    std::unique_ptr<engine::models::ming3_tts::MingPromptBuilder> prompt_builder;
    engine::models::ming3_tts::MingPromptOptions prompt_options;
    engine::models::ming3_tts::MingGenerationOptions generation_options;
    std::string prompt_reference_text;
    std::vector<std::string> fragments;
    std::vector<std::vector<float>> prompt_speakers;
    const auto prompt_latents = fixture->has_tensor("prompt.0.latents")
        ? fixture->require_f32("prompt.0.latents") : std::vector<float>{};
    for (size_t i = 0; fixture->has_tensor("prompt.0.speaker." + std::to_string(i)); ++i) {
        prompt_speakers.push_back(fixture->require_f32("prompt.0.speaker." + std::to_string(i)));
    }
    if (!requests_path.empty()) {
        if (tokenizer_dir.empty() || request_name.empty()) throw std::runtime_error("native prompts require --tokenizer-dir and --request-name");
        const auto requests = engine::io::json::parse_file(requests_path);
        std::optional<engine::io::json::Value> selected;
        for (const auto & item : requests.as_array()) {
            if (item.require("name").as_string() == request_name) selected = item;
        }
        if (!selected) throw std::runtime_error("request not found: " + request_name);
        generation_options.seed = engine::io::json::optional_i64(*selected, "seed", 1895);
        generation_options.max_tokens = engine::io::json::optional_i64(*selected, "max_decode_steps", 1000);
        generation_options.cfg = engine::io::json::optional_f32(*selected, "cfg", 2.0f);
        generation_options.sigma = engine::io::json::optional_f32(*selected, "sigma", 0.25f);
        generation_options.temperature = engine::io::json::optional_f32(*selected, "temperature", 0.0f);
        prompt_reference_text = engine::io::json::optional_string(*selected, "prompt_text", "");
        auto & options = prompt_options;
        options.template_name = selected->require("task").as_string();
        options.language = engine::io::json::optional_string(*selected, "language_id", "");
        options.instruct = engine::io::json::optional_string(*selected, "instruction", "");
        options.instruction_schema = engine::io::json::optional_string(*selected, "instruction_schema", "");
        if (const auto * controls = selected->find("controls")) {
            for (const auto & [key, value] : controls->as_object()) options.controls[key] = value.as_string();
        }
        prompt_builder = std::make_unique<engine::models::ming3_tts::MingPromptBuilder>(
            engine::tokenizers::LlamaBpeTokenizerSpec{{}, {}, tokenizer_dir / "tokenizer_config.json",
                tokenizer_dir / "tokenizer.json", engine::tokenizers::LlamaBpePreTokenizer::Qwen2});
        if (const auto * pieces = selected->find("fragments")) {
            for (const auto & piece : pieces->as_array()) fragments.push_back(piece.as_string());
            if (fragments.empty()) throw std::runtime_error("Ming fixture fragments must not be empty");
        } else {
            fragments.push_back(selected->require("text").as_string());
        }
        native_prompt = prompt_builder->build(fragments.front(), options, prompt_speakers.size(),
            prompt_reference_text, prompt_latents.size() / latent_dim);
        const auto expected_ids = fixture->require_tensor_data("prompt.0.ids");
        const std::vector<int64_t> actual_ids(native_prompt->ids.begin(), native_prompt->ids.end());
        if (expected_ids.metadata.dtype != "I64" || expected_ids.bytes.size() != actual_ids.size() * sizeof(int64_t) ||
            std::memcmp(expected_ids.bytes.data(), actual_ids.data(), expected_ids.bytes.size()) != 0) {
            engine::debug::trace_log_i32("ming3_tts.prompt.ids", {static_cast<int64_t>(native_prompt->ids.size())}, native_prompt->ids);
            throw std::runtime_error("native Ming prompt token IDs differ from the official reference");
        }
        engine::debug::trace_log_scalar("ming3_tts.prompt.ids_equal", true);
    }
    if (prompt_only) {
        if (!native_prompt) throw std::runtime_error("--prompt-only requires --requests");
        engine::models::ming3_tts::MingGenerator generator(engine::assets::open_tensor_source(weights), model_config, execution, attention);
        const auto start = std::chrono::steady_clock::now();
        const auto actual = generator.prepare_inputs(*native_prompt, prompt_speakers, prompt_latents);
        const auto expected = fixture->require_f32("prompt.0.embeddings");
        engine::debug::timing_log_scalar("session.wall_ms", engine::debug::elapsed_ms(start));
        const auto repeat_start = std::chrono::steady_clock::now();
        const auto repeated = generator.prepare_inputs(*native_prompt, prompt_speakers, prompt_latents);
        engine::debug::timing_log_scalar("ming3_tts.prompt.repeat_ms", engine::debug::elapsed_ms(repeat_start));
        if (repeated != actual) throw std::runtime_error("native Ming prompt changes on identical replay");
        engine::debug::trace_log_scalar("ming3_tts.prompt.repeat_equal", true);
        if (actual.size() != expected.size()) throw std::runtime_error("native Ming prompt embedding shape mismatch");
        size_t different = 0;
        double dot = 0, aa = 0, bb = 0, max_error = 0;
        for (size_t i = 0; i < actual.size(); ++i) {
            different += actual[i] != expected[i];
            dot += static_cast<double>(actual[i]) * expected[i];
            aa += static_cast<double>(actual[i]) * actual[i];
            bb += static_cast<double>(expected[i]) * expected[i];
            max_error = std::max(max_error, std::abs(static_cast<double>(actual[i]) - expected[i]));
        }
        engine::debug::trace_log_scalar("ming3_tts.prompt.embeddings.different_elements", different);
        engine::debug::trace_log_scalar("ming3_tts.prompt.embeddings.max_error", max_error);
        engine::debug::trace_log_scalar("ming3_tts.prompt.embeddings.cosine", dot / std::sqrt(aa * bb));
        if (different) throw std::runtime_error("native Ming prompt embeddings differ from the official reference");
        engine::debug::trace_log_scalar("ming3_tts.prompt.embeddings_equal", true);
        return 0;
    }
    if (flow_unit) {
        engine::modules::MingFlowRuntime sampler(engine::assets::open_tensor_source(weights), model_config.flow, execution);
        const auto start = std::chrono::steady_clock::now();
        if (fixture->has_tensor("flow.unit.velocity")) {
            const auto velocity = sampler.predict_velocity(fixture->require_f32("flow.0.noise"),
                fixture->require_f32("flow.0.c"), fixture->require_f32("flow.0.latent_history"), 0.0f);
            engine::debug::trace_log_f32("ming3_tts.reference.velocity", {2, 64}, fixture->require_f32("flow.unit.velocity"));
            engine::debug::trace_log_f32("ming3_tts.actual.velocity", {2, 64}, velocity);
            if (fixture->has_tensor("flow.unit.velocity_half")) {
                for (const auto * key : {"time_embedding", "time_projected"}) {
                    const auto name = std::string("flow.unit.") + key;
                    if (fixture->has_tensor(name)) engine::debug::trace_log_f32(std::string("ming3_tts.reference.") + key,
                        fixture->require_metadata(name).shape, fixture->require_f32(name));
                }
                for (const auto * key : {"input", "output", "normalized", "attention", "q", "k", "v", "q_rope", "k_rope"}) {
                    const auto name = std::string("flow.unit.block0.") + key;
                    if (fixture->has_tensor(name)) engine::debug::trace_log_f32(std::string("ming3_tts.reference.block0.") + key,
                        fixture->require_metadata(name).shape, fixture->require_f32(name));
                }
                const auto half = sampler.predict_velocity(fixture->require_f32("flow.0.noise"),
                    fixture->require_f32("flow.0.c"), fixture->require_f32("flow.0.latent_history"), 0.5f);
                engine::debug::trace_log_f32("ming3_tts.reference.velocity_half", {2, 64}, fixture->require_f32("flow.unit.velocity_half"));
                engine::debug::trace_log_f32("ming3_tts.actual.velocity_half", {2, 64}, half);
            }
        }
        const auto actual = sampler.sample(fixture->require_f32("flow.0.c"), fixture->require_f32("flow.0.latent_history"),
            fixture->require_f32("flow.0.noise"), fixture->require_f32("flow.0.sde_noise"));
        const auto expected = fixture->require_f32("flow.0.latent");
        engine::debug::timing_log_scalar("session.wall_ms", engine::debug::elapsed_ms(start));
        engine::debug::trace_log_f32("ming3_tts.reference.flow", {64}, expected);
        engine::debug::trace_log_f32("ming3_tts.actual.flow", {64}, actual);
        double dot = 0, aa = 0, bb = 0;
        for (size_t i = 0; i < actual.size(); ++i) {
            dot += actual[i] * expected.at(i); aa += actual[i] * actual[i]; bb += expected[i] * expected[i];
        }
        engine::debug::trace_log_scalar("ming3_tts.flow.cosine", dot / std::sqrt(aa * bb));
        return dot / std::sqrt(aa * bb) > 0.99999 ? 0 : 1;
    }
    std::unique_ptr<engine::modules::MingAudioDecoder> decoder;
    if (!istft_only && !head_only) decoder = std::make_unique<engine::modules::MingAudioDecoder>(
        engine::assets::open_tensor_source(weights), model_config.codec, execution, attention);
    std::unique_ptr<engine::modules::MingFlowRuntime> flow_runtime;
    if (flow) flow_runtime = std::make_unique<engine::modules::MingFlowRuntime>(
        engine::assets::open_tensor_source(weights), model_config.flow, execution);
    std::unique_ptr<engine::models::ming3_tts::MingGenerator> generator;
    if (generate) generator = std::make_unique<engine::models::ming3_tts::MingGenerator>(
        engine::assets::open_tensor_source(weights), model_config, execution, attention);
    const auto start = std::chrono::steady_clock::now();
    auto latent = fixture->require_f32("decode.0.latent");
    std::vector<std::vector<float>> generated_latents;
    if (encode) {
        engine::modules::MingAudioEncoder encoder(engine::assets::open_tensor_source(weights),
            model_config.codec, execution, attention);
        latent = encoder.encode(fixture->require_f32("encoder.waveform"), fixture->require_f32("encoder.noise"));
        const auto expected = fixture->require_f32("encoder.latent");
        if (latent.size() != expected.size()) throw std::runtime_error("Ming encoder latent length mismatch");
        double dot = 0, aa = 0, bb = 0;
        for (size_t i = 0; i < latent.size(); ++i) {
            dot += latent[i] * expected[i]; aa += latent[i] * latent[i]; bb += expected[i] * expected[i];
        }
        engine::debug::trace_log_scalar("ming3_tts.parity.encoder.cosine", dot / std::sqrt(aa * bb));
    } else if (generator) {
        uint64_t random_offset = 0;
        if (native_prompt) {
            std::vector<float> history;
            if (!prompt_latents.empty()) history.assign(prompt_latents.end() - latent_dim, prompt_latents.end());
            for (const auto & fragment : fragments) {
                const auto prompt = prompt_builder->build(fragment, prompt_options, prompt_speakers.size(),
                    prompt_reference_text, prompt_latents.size() / latent_dim);
                const auto embeddings = generator->prepare_inputs(prompt, prompt_speakers, prompt_latents);
                const auto prefix = "prompt." + std::to_string(generated_latents.size());
                const auto expected_ids = fixture->require_tensor_data(prefix + ".ids");
                const std::vector<int64_t> ids(prompt.ids.begin(), prompt.ids.end());
                if (expected_ids.metadata.dtype != "I64" || expected_ids.bytes.size() != ids.size() * sizeof(int64_t) ||
                    std::memcmp(expected_ids.bytes.data(), ids.data(), expected_ids.bytes.size()) != 0) {
                    throw std::runtime_error("native Ming token IDs differ at " + prefix);
                }
                if (embeddings != fixture->require_f32(prefix + ".embeddings")) {
                    throw std::runtime_error("native Ming embeddings differ at " + prefix);
                }
                engine::debug::trace_log_scalar("ming3_tts." + prefix + ".inputs_equal", true);
                generated_latents.push_back(generator->generate(embeddings, generation_options, random_offset, history));
            }
        } else {
            generated_latents.push_back(generator->generate(fixture->require_f32("prompt.0.embeddings"), {}, random_offset));
        }
    } else if (flow_runtime) {
        const auto dimensions = fixture->require_metadata("decode.0.latent").shape;
        auto history = fixture->require_f32("flow.0.latent_history");
        latent.clear();
        for (int64_t frame = 0; frame < dimensions.at(1); ++frame) {
            const auto prefix = "flow." + std::to_string(frame);
            if (reference_history) history = fixture->require_f32(prefix + ".latent_history");
            auto patch = flow_runtime->sample(fixture->require_f32(prefix + ".c"), history,
                fixture->require_f32(prefix + ".noise"), fixture->require_f32(prefix + ".sde_noise"));
            latent.insert(latent.end(), patch.begin(), patch.end());
            history = std::move(patch);
        }
    }
    std::vector<float> actual;
    if (istft_only || head_only) {
        const auto shape = fixture->require_metadata("decode.0.spectrum").shape;
        const auto hop = config.require("audio_tokenizer_config").require("dec_kwargs").require("output_dim").as_i64();
        engine::audio::HostLogMagnitudePhaseISTFT istft({shape.at(1), hop * 4, hop, hop * 4 + 2, 8});
        const auto source = engine::assets::open_tensor_source(weights);
        const auto window = source->require_f32("audio.decoder.head.istft.window");
        auto spectrum = fixture->require_f32("decode.0.spectrum");
        if (head_only) {
            const auto hidden_shape = fixture->require_metadata("decode.0.hidden").shape;
            const int64_t hidden = hidden_shape.at(2);
            engine::core::BackendWeightStore store(execution.backend(), execution.backend_type(), "ming3_tts.head.weights", 1024 * 1024);
            const auto projection = engine::modules::binding::linear_from_source(store, *source, "audio.decoder.head.out",
                engine::assets::TensorStorageType::Native, hop * 4 + 2, hidden, true);
            store.upload();
            std::unique_ptr<ggml_context, decltype(&ggml_free)> context(ggml_init({1024 * 1024, nullptr, true}), ggml_free);
            engine::core::ModuleBuildContext ctx{context.get(), "ming3_tts.head", execution.backend_type()};
            const auto input = engine::core::make_tensor(ctx, GGML_TYPE_F32,
                engine::core::TensorShape::from_dims({1, shape.at(1), hidden}));
            ggml_set_input(input.tensor);
            const engine::modules::Bf16RoundModule round;
            auto projected = engine::modules::LinearModule({hidden, hop * 4 + 2, false})
                .build(ctx, round.build(ctx, input), {projection.weight, std::nullopt});
            projected = engine::modules::AddModule(false, true).build(ctx, round.build(ctx, projected), *projection.bias);
            projected = round.build(ctx, projected);
            ggml_set_output(projected.tensor);
            auto * graph = ggml_new_graph_custom(context.get(), 256, false);
            ggml_build_forward_expand(graph, projected.tensor);
            std::unique_ptr<ggml_gallocr, decltype(&ggml_gallocr_free)> allocator(
                ggml_gallocr_new(ggml_backend_get_default_buffer_type(execution.backend())), ggml_gallocr_free);
            if (!ggml_gallocr_reserve(allocator.get(), graph) || !ggml_gallocr_alloc_graph(allocator.get(), graph)) {
                throw std::runtime_error("Ming isolated output projection allocation failed");
            }
            engine::core::HostGraphPlan plan;
            engine::core::prepare_host_graph_plan(execution, graph, plan);
            engine::core::write_tensor_f32(input, fixture->require_f32("decode.0.hidden"));
            const auto status = engine::core::compute_graph(execution, graph, plan, "ming3_tts.head");
            spectrum = engine::core::read_tensor_f32(projected.tensor);
            engine::core::release_backend_graph_resources(execution.backend(), graph);
            if (status != GGML_STATUS_SUCCESS) throw std::runtime_error("Ming isolated output projection failed");
            const auto expected_spectrum = fixture->require_f32("decode.0.spectrum");
            size_t mismatched = 0;
            for (size_t i = 0; i < spectrum.size(); ++i) mismatched += spectrum[i] != expected_spectrum.at(i);
            engine::debug::trace_log_scalar("ming3_tts.parity.head.different_elements", mismatched);
        }
        actual = istft.compute(spectrum, window).audio;
    } else if (generate) {
        std::vector<engine::io::SafeTensorWriteEntry> entries;
        for (size_t index = 0; index < generated_latents.size(); ++index) {
            const auto & generated = generated_latents[index];
            const auto audio = decoder->decode(generated);
            actual.insert(actual.end(), audio.begin(), audio.end());
            if (!generated_fixture.empty()) {
                const auto prefix = "decode." + std::to_string(index);
                engine::io::SafeTensorWriteEntry latent_entry{prefix + ".latent", "F32",
                    {1, static_cast<int64_t>(generated.size() / latent_dim), latent_dim}, std::vector<unsigned char>(generated.size() * sizeof(float))};
                std::memcpy(latent_entry.data.data(), generated.data(), latent_entry.data.size());
                entries.push_back(std::move(latent_entry));
                engine::io::SafeTensorWriteEntry audio_entry{prefix + ".waveform", "F32",
                    {1, static_cast<int64_t>(audio.size())}, std::vector<unsigned char>(audio.size() * sizeof(float))};
                std::memcpy(audio_entry.data.data(), audio.data(), audio_entry.data.size());
                entries.push_back(std::move(audio_entry));
            }
            engine::debug::trace_log_scalar("ming3_tts.generated.chunk_frames", generated.size() / latent_dim);
        }
        if (!generated_fixture.empty()) engine::io::write_safetensors_file(generated_fixture, entries);
    } else {
        actual = decoder->decode(latent);
    }
    engine::debug::timing_log_scalar("session.wall_ms", engine::debug::elapsed_ms(start));
    const auto expected = fixture->require_f32("decode.0.waveform");
    if (generate) {
        if (output.empty()) throw std::runtime_error("--generate requires --output");
        engine::audio::write_pcm16_wav(output, 44100, 1, actual);
        return 0;
    }
    if (actual.size() != expected.size()) throw std::runtime_error("Ming waveform length mismatch");
    double dot = 0, norm_a = 0, norm_b = 0, squared_error = 0;
    for (size_t i = 0; i < actual.size(); ++i) {
        const double a = actual[i], b = expected[i];
        dot += a * b; norm_a += a * a; norm_b += b * b;
        squared_error += (a - b) * (a - b);
    }
    const double cosine = dot / std::sqrt(norm_a * norm_b);
    engine::debug::trace_log_scalar("ming3_tts.parity.waveform.cosine", cosine);
    engine::debug::trace_log_scalar("ming3_tts.parity.waveform.rmse", std::sqrt(squared_error / actual.size()));
    std::cout.precision(10);
    std::cout << "waveform cosine=" << cosine << " samples=" << actual.size() << '\n';
    if (!output.empty()) {
        engine::audio::WavWriteOptions options;
        options.format = engine::audio::WavSampleFormat::Float32;
        engine::audio::write_wav(output, config.require("audio_tokenizer_config").require("sample_rate").as_i64(),
            1, actual, options);
    }
    return cosine > 0.99999 ? 0 : 1;
} catch (const std::exception & error) {
    std::cerr << "ming3_tts_component_parity: " << error.what() << '\n';
    return 1;
}
