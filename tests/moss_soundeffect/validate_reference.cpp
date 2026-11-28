// Manual production replay: native tokenizer/Qwen -> DiT diffusion -> DAC waveform, with aligned noise.
#include "engine/models/moss_soundeffect/assets.h"
#include "engine/models/moss_soundeffect/dit.h"
#include "engine/models/moss_soundeffect/vae.h"
#include "engine/framework/assets/tensor_source.h"
#include "engine/models/moss_soundeffect/text_encoder.h"
#include "engine/framework/core/execution_context.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/io/json.h"
#include "engine/framework/model_spec/package.h"
#include "engine/framework/sampling/flow_sampler_runtime.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {
namespace core = engine::core;
namespace debug = engine::debug;
namespace json = engine::io::json;
namespace modules = engine::modules;
namespace moss = engine::models::moss_soundeffect;
constexpr double kMinimumCosine = 0.99999;

struct Options {
    std::filesystem::path model, fixture, log;
    std::optional<std::filesystem::path> spec;
    core::BackendType backend = core::BackendType::Cuda;
    core::AttentionPreference attention = core::AttentionPreference::Auto;
    int device = 0;
    int threads = 8;
    bool sampling_check = false;
    double atol = 1e-3;
    double rtol = 1e-3;
};

// Every sample is checked without time alignment, gain normalization, or dropping quiet regions.
bool compare(const std::string & label, const std::vector<float> & actual,
             const std::vector<float> & expected, double atol, double rtol) {
    if (actual.empty() || actual.size() != expected.size())
        throw std::runtime_error(label + " tensor length mismatch or empty tensor");
    double squared_error = 0, dot = 0, actual_energy = 0, reference_energy = 0, max_abs = 0;
    size_t failures = 0;
    for (size_t i = 0; i < actual.size(); ++i) {
        const double a = actual[i], b = expected[i];
        if (!std::isfinite(a) || !std::isfinite(b))
            throw std::runtime_error(label + " contains non-finite samples");
        const double difference = std::abs(a - b);
        max_abs = std::max(max_abs, difference);
        squared_error += difference * difference;
        actual_energy += a * a;
        reference_energy += b * b;
        dot += a * b;
        if (difference > atol + rtol * std::abs(b)) ++failures;
    }
    const double denominator = std::sqrt(actual_energy) * std::sqrt(reference_energy);
    const double cosine = denominator > 0 ? dot / denominator : (actual_energy == reference_energy ? 1.0 : 0.0);
    debug::timing_log_scalar(label + ".samples", actual.size());
    debug::timing_log_scalar(label + ".max_abs_error", max_abs);
    debug::timing_log_scalar(label + ".rmse", std::sqrt(squared_error / actual.size()));
    debug::timing_log_scalar(label + ".cosine", cosine);
    debug::timing_log_scalar(label + ".failed_samples", failures);
    debug::timing_log_scalar(label + ".allclose_passed", failures == 0);
    debug::timing_log_scalar(label + ".passed", cosine > kMinimumCosine);
    return cosine > kMinimumCosine;
}

std::pair<std::vector<float>, bool> validate_text(
    const std::string & branch, const std::string & prompt, const moss::Assets & assets,
    const engine::assets::TensorSource & tokens, moss::TextEncoderRuntime & text_runtime, const std::vector<float> & expected,
    double atol, double rtol) {
    const auto prefix = "text." + branch;
    std::vector<int32_t> reference_ids(static_cast<size_t>(assets.text_length));
    std::vector<int32_t> mask(reference_ids.size());
    std::vector<int32_t> valid_count(1);
    const std::pair<const char *, std::vector<int32_t> *> fields[] = {
        {"ids", &reference_ids}, {"mask", &mask}, {"valid_count", &valid_count},
    };
    for (const auto & field : fields) {
        const auto shape = std::string(field.first) == "valid_count"
            ? std::vector<int64_t>{1} : std::vector<int64_t>{1, assets.text_length};
        const auto tensor = tokens.require_tensor_data(prefix + "." + field.first);
        if (tensor.metadata.shape != shape ||
            engine::assets::ggml_type_for_tensor_dtype(tensor.metadata.dtype) != GGML_TYPE_I32 ||
            tensor.bytes.size() != field.second->size() * sizeof(int32_t))
            throw std::runtime_error(prefix + "." + field.first + " must be an I32 token fixture");
        std::memcpy(field.second->data(), tensor.bytes.data(), tensor.bytes.size());
    }
    if (valid_count[0] < 0 || valid_count[0] > assets.text_length)
        throw std::runtime_error(prefix + " invalid reference token count");
    const auto count = static_cast<size_t>(valid_count[0]);
    for (size_t i = 0; i < mask.size(); ++i) {
        if (mask[i] != (i < count ? 1 : 0))
            throw std::runtime_error(prefix + " mask must be a valid prefix matching valid_count");
    }
    const auto ids = text_runtime.tokenize(prompt);
    const bool tokens_match = ids.size() == count && std::equal(ids.begin(), ids.end(), reference_ids.begin());
    debug::timing_log_scalar("parity." + prefix + ".valid_count", ids.size());
    debug::timing_log_scalar("parity." + prefix + ".expected_valid_count", count);
    debug::timing_log_scalar("parity." + prefix + ".tokens_exact", tokens_match);
    if (!tokens_match) throw std::runtime_error(prefix + " native token IDs/count differ from the official tokenizer");
    auto hidden = text_runtime.encode(prompt);
    const bool passed = compare("parity." + prefix + ".context", hidden, expected, atol, rtol);
    return {std::move(hidden), passed};
}

bool validate(const Options & options) {
    const auto manifest = json::parse_file(options.fixture / "manifest.json");
    const auto format = json::require_string(manifest, "format");
    const bool bf16 = json::require_string(manifest, "dtype") == "bfloat16";
    if ((format != "moss_soundeffect_v2.parity.v1" && format != "moss_soundeffect_v2.f32_parity.v1") ||
        (json::require_string(manifest, "dtype") != "float32" && !bf16) ||
        json::require_string(manifest, "attention") != (bf16 ? "official" : "sdpa_math") ||
        json::require_bool(manifest, "autocast") != bf16 || json::require_bool(manifest, "tf32") ||
        json::require_bool(manifest, "compiled_model"))
        throw std::runtime_error("fixture precision/attention contract is inconsistent");
    engine::model_spec::ScopedSpecOverride spec_override(options.spec, options.model);
    const auto assets = moss::load_assets(options.model);
    if (!bf16)
        throw std::runtime_error("the production runtime follows upstream BF16 inference; use a BF16 fixture");
    const auto & c = assets->dit;
    const int64_t frames = int64_t(assets->max_duration_sec) * assets->sample_rate / assets->vae.hop_length;
    const std::pair<const char *, int64_t> dimensions[] = {
        {"frames", frames}, {"latent_dim", c.latent_dim}, {"text_length", assets->text_length},
        {"text_dim", c.text_dim}, {"sample_rate", assets->sample_rate},
        {"max_duration_sec", assets->max_duration_sec}, {"hop_length", assets->vae.hop_length},
    };
    for (const auto & item : dimensions)
        if (json::require_i64(manifest, item.first) != item.second)
            throw std::runtime_error(std::string("fixture/model dimension mismatch: ") + item.first);
    const auto & transformer = manifest.require("transformer_config");
    const std::pair<const char *, int64_t> architecture[] = {
        {"dim", c.dim}, {"ffn_dim", c.ffn_dim}, {"num_heads", c.heads}, {"num_layers", c.layers},
        {"in_dim", c.latent_dim}, {"out_dim", c.latent_dim}, {"text_dim", c.text_dim}, {"freq_dim", c.freq_dim},
    };
    for (const auto & item : architecture)
        if (json::require_i64(transformer, item.first) != item.second)
            throw std::runtime_error(std::string("fixture/model architecture mismatch: ") + item.first);
    const auto & patch = transformer.require("patch_size").as_array();
    if (patch.size() != 1 || patch.front().as_i64() != c.patch_size ||
        json::require_f32(transformer, "eps") != c.eps)
        throw std::runtime_error("fixture/model patch or normalization epsilon mismatch");
    const auto fixture = engine::assets::open_tensor_source(options.fixture / "reference.safetensors");
    const auto positive = fixture->require_f32("context.positive", {1, assets->text_length, c.text_dim});
    const auto negative = fixture->require_f32("context.negative", {1, assets->text_length, c.text_dim});
    const auto noise = fixture->require_f32("initial_noise", {1, c.latent_dim, frames});
    const auto expected_latent = fixture->require_f32("final_latent", {1, c.latent_dim, frames});
    const auto steps = json::require_i64(manifest, "steps");
    const auto timesteps = fixture->require_f32("schedule.timesteps", {steps});
    const auto sigmas = fixture->require_f32("schedule.sigmas", {steps + 1});
    const auto full_samples = json::require_i64(manifest, "full_waveform_samples");
    const auto crop_samples = json::require_i64(manifest, "output_samples");
    const auto expected_waveform = fixture->require_f32("waveform", {1, 1, full_samples});
    const float guidance = json::require_f32(manifest, "guidance_scale");
    if (steps <= 0 || full_samples <= 0 || crop_samples <= 0 || crop_samples > full_samples ||
        !std::isfinite(guidance) || sigmas.back() != 0.0F)
        throw std::runtime_error("invalid reference schedule, guidance or waveform crop");
    for (size_t i = 0; i < timesteps.size(); ++i) {
        if (!std::isfinite(sigmas[i]) || !std::isfinite(timesteps[i]) || sigmas[i] <= sigmas[i + 1] ||
            std::abs(timesteps[i] - 1000.0F * sigmas[i]) > 1e-3F)
            throw std::runtime_error("reference schedule must descend in sigma-space with DiT timestep=1000*sigma");
    }
    debug::timing_log_scalar("parity.prompt", json::require_string(manifest, "formatted_prompt"));
    debug::timing_log_scalar("parity.steps", steps);
    debug::timing_log_scalar("parity.atol", options.atol);
    debug::timing_log_scalar("parity.rtol", options.rtol);
    debug::timing_log_scalar("parity.cosine_min_exclusive", kMinimumCosine);
    debug::timing_log_scalar("parity.native_text_encoder_tested", true);

    core::BackendConfig backend;
    backend.type = options.backend;
    backend.device = options.device;
    backend.threads = options.threads;
    core::ExecutionContext execution(backend);
    if (assets->text.hidden_size != c.text_dim)
        throw std::runtime_error("native Qwen hidden size must equal the DiT text input size");
    moss::TextEncoderRuntime text_runtime(execution, assets, engine::assets::TensorStorageType::Native);
    for (const auto & test : manifest.require("tokenizer_cases").as_array()) {
        const auto prompt = json::require_string(test, "text");
        const auto actual = text_runtime.tokenize(prompt);
        std::vector<int32_t> expected;
        for (const auto & token : test.require("ids").as_array()) expected.push_back(static_cast<int32_t>(token.as_i64()));
        if (actual != expected) {
            debug::timing_log_scalar("parity.tokenizer.failed_text", prompt);
            debug::trace_log_i32("parity.tokenizer.actual_ids", {static_cast<int64_t>(actual.size())}, actual);
            debug::trace_log_i32("parity.tokenizer.expected_ids", {static_cast<int64_t>(expected.size())}, expected);
            throw std::runtime_error("native tokenizer normalization differs from the official implementation");
        }
    }
    debug::timing_log_scalar("parity.tokenizer.cases_exact", true);
    auto native_positive = validate_text("positive", json::require_string(manifest, "formatted_prompt"),
        *assets, *fixture, text_runtime, positive, options.atol, options.rtol);
    auto native_negative = validate_text("negative", json::require_string(manifest, "negative_prompt"),
        *assets, *fixture, text_runtime, negative, options.atol, options.rtol);
    bool passed = native_positive.second && native_negative.second;
    // One session owns Qwen/DiT/VAE and reuses their graphs. Only noise/schedule are injected.
    moss::DitRuntime dit(execution, assets, engine::assets::TensorStorageType::Native, options.attention);
    moss::VaeRuntime vae(execution, assets);
    dit.prepare_context(native_positive.first, native_negative.first);
    if (options.sampling_check) {
        text_runtime.encode("A passing train rattles along the tracks, with a loud whistle and rhythmic wheels.");
        std::string long_prompt;
        for (int i = 0; i < 32; ++i) long_prompt += json::require_string(manifest, "formatted_prompt") + " ";
        debug::timing_log_scalar("sampling.text_context_max_tokens", text_runtime.tokenize(long_prompt).size());
        text_runtime.encode(long_prompt);
        const auto repeated_context = text_runtime.encode(json::require_string(manifest, "formatted_prompt"));
        bool matched = repeated_context == native_positive.first;
        debug::timing_log_scalar("sampling.text_context_reuse.exact_match", matched);
        for (const float scale : {4.0F, 1.0F, 4.0F}) {
            auto host = noise;
            constexpr int count = 4;
            constexpr float shift = 5.0F;
            for (int step = 0; step < count; ++step) {
                const float r = 1.0F - float(step) / count;
                const float nr = 1.0F - float(step + 1) / count;
                const float sigma = shift * r / (1.0F + (shift - 1.0F) * r);
                const float next = shift * nr / (1.0F + (shift - 1.0F) * nr);
                const auto prediction = scale == 1 ? dit.predict(host, sigma * 1000.0F) :
                                                     dit.predict_cfg(host, sigma * 1000.0F);
                std::vector<float> delta(host.size());
                for (size_t i = 0; i < host.size(); ++i) {
                    const float velocity = scale == 1 ? prediction[i] :
                        prediction[i + host.size()] + scale * (prediction[i] - prediction[i + host.size()]);
                    delta[i] = velocity * (next - sigma);
                }
                if (scale == 1) core::round_f32_to_bf16_in_place(delta);
                for (size_t i = 0; i < host.size(); ++i) host[i] += delta[i];
                if (scale == 1) core::round_f32_to_bf16_in_place(host);
            }
            const auto device = dit.sample(noise, count, scale, shift);
            const bool exact = device == host;
            debug::timing_log_scalar("sampling.cfg" + std::to_string(int(scale)) + ".exact_match", exact);
            matched = exact && matched;
        }
        const auto decoded = vae.decode(expected_latent);
        for (const size_t length : {size_t(4800), size_t(480000), size_t(1439999)}) {
            const auto prefix = vae.decode(expected_latent, length);
            const bool exact = prefix.samples.size() == length &&
                std::equal(prefix.samples.begin(), prefix.samples.end(), decoded.samples.begin());
            debug::timing_log_scalar("sampling.vae_prefix" + std::to_string(length) + ".exact_match", exact);
            matched = exact && matched;
        }
        return matched;
    }
    modules::FlowSamplerRuntimeConfig config;
    config.label = "moss_soundeffect.parity";
    config.latent_shape = {1, c.latent_dim, frames};
    config.branches = {{"velocity", 1.0F}};
    config.prediction_type = modules::FlowSamplerPredictionType::Velocity;
    config.update_rule = modules::FlowSamplerUpdateRule::Euler;
    for (int64_t i = 0; i < steps; ++i) {
        // Euler uses t_next - t: t MUST be sigma, not the 1000-scaled DiT timestep.
        config.schedule.push_back({i, sigmas[i], sigmas[i + 1], sigmas[i], sigmas[i + 1]});
    }
    modules::FlowSamplerSingleBranchDenoiserConfig denoiser;
    denoiser.label = "moss_soundeffect.parity.dit";
    denoiser.branch_name = "velocity";
    denoiser.sampler_mode = [](const modules::FlowSamplerStepState &) { return std::string("reference_replay"); };
    denoiser.predict = [&](const modules::FlowSamplerDenoiserInput & input) {
        // Replay the saved official 1000*sigma values, avoiding schedule regeneration drift.
        const float timestep = timesteps.at(static_cast<size_t>(input.state.schedule.index));
        auto pos = dit.predict(input.latent, timestep);
        if (guidance != 1.0F) {
            const auto neg = dit.predict(input.latent, timestep, true);
            for (size_t i = 0; i < pos.size(); ++i) pos[i] = neg[i] + guidance * (pos[i] - neg[i]);
        }
        return pos;
    };
    modules::FlowSamplerRuntime sampler(std::move(config),
        modules::make_flow_sampler_single_branch_denoiser(std::move(denoiser)));
    sampler.run_sequence(noise);
    passed = compare("parity.final_latent", sampler.latent(), expected_latent, options.atol, options.rtol) && passed;
    // Always continue through waveform comparison, even if the latent gate failed.
    const auto generated = vae.decode(sampler.latent());
    if (generated.sample_rate != assets->sample_rate || generated.channels != 1)
        throw std::runtime_error("native decoder returned incorrect sample rate or channels");
    passed = compare("parity.waveform.full", generated.samples, expected_waveform, options.atol, options.rtol) && passed;
    const std::vector<float> actual_crop(generated.samples.begin(), generated.samples.begin() + crop_samples);
    const std::vector<float> reference_crop(expected_waveform.begin(), expected_waveform.begin() + crop_samples);
    passed = compare("parity.waveform.crop", actual_crop, reference_crop, options.atol, options.rtol) && passed;
    // Same decoder, oracle final latent: separates DAC error from diffusion trajectory error.
    const auto oracle_decode = vae.decode(expected_latent);
    passed = compare("parity.vae_reference_latent", oracle_decode.samples, expected_waveform,
                     options.atol, options.rtol) && passed;
    // Replay the same trajectory with oracle text conditioning to isolate encoder drift.
    dit.prepare_context(positive, negative);
    sampler.run_sequence(noise);
    const auto oracle_conditioning = vae.decode(sampler.latent());
    passed = compare("parity.reference_context_waveform", oracle_conditioning.samples, expected_waveform,
                     options.atol, options.rtol) && passed;
    return passed;
}
}  // namespace

int main(int argc, char ** argv) {
    Options options;
    try {
        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            if (i + 1 >= argc) throw std::runtime_error("missing option value: " + arg);
            const std::string value = argv[++i];
            if (arg == "--model") options.model = value;
            else if (arg == "--fixture-dir") options.fixture = value;
            else if (arg == "--log") options.log = value;
            else if (arg == "--model-spec-override") options.spec = value;
            else if (arg == "--attention") options.attention = core::parse_attention_preference(value, "attention");
            else if (arg == "--backend") {
                if (value == "cuda") options.backend = core::BackendType::Cuda;
                else if (value == "vulkan") options.backend = core::BackendType::Vulkan;
                else if (value == "cpu") options.backend = core::BackendType::Cpu;
                else throw std::runtime_error("unsupported parity backend: " + value);
            }
            else if (arg == "--device") options.device = std::stoi(value);
            else if (arg == "--threads") options.threads = std::stoi(value);
            else if (arg == "--sampling-check") options.sampling_check = value == "true";
            else if (arg == "--atol") options.atol = std::stod(value);
            else if (arg == "--rtol") options.rtol = std::stod(value);
            else throw std::runtime_error("unknown option: " + arg);
        }
        if (options.model.empty() || options.fixture.empty() || options.log.empty() || options.threads <= 0 ||
            !std::isfinite(options.atol) || !std::isfinite(options.rtol) || options.atol < 0 || options.rtol < 0)
            throw std::runtime_error("usage: moss_soundeffect_parity --model <F32.gguf> --fixture-dir <dir> --log <file> "
                                     "[--model-spec-override <json>] [--attention auto|flash|eager] [--backend cuda|vulkan|cpu] [--device 0] [--threads 8] [--atol 0.001] [--rtol 0.001] [--sampling-check true]");
        if (options.log.has_parent_path()) std::filesystem::create_directories(options.log.parent_path());
        debug::configure_logging({true, options.log.string()});
    } catch (const std::exception & error) {
        std::cerr << error.what() << '\n';
        return 2;
    }
    const auto started = std::chrono::steady_clock::now();
    int result = 1;
    try {
        const bool passed = validate(options);
        debug::timing_log_scalar("parity.passed", passed);
        result = passed ? 0 : 1;
    } catch (const std::exception & error) {
        debug::log_message(debug::LogLevel::Error, "parity", error.what());
        debug::timing_log_scalar("parity.passed", false);
    }
    debug::timing_log_scalar("session.wall_ms", debug::elapsed_ms(started));
    return result;
}
