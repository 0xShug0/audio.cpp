#include "engine/models/kugelaudio/generator.h"
#include "engine/models/kugelaudio/scheduler.h"

#include "engine/framework/debug/trace.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/modules/lookup_modules.h"
#include "engine/framework/sampling/hf_sampler.h"
#include "engine/framework/sampling/noise.h"
#include "engine/framework/tokenizers/llama_bpe.h"

#include <ggml-alloc.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <random>
#include <stdexcept>

namespace engine::models::kugelaudio {

struct Generator::Impl {
    core::ExecutionContext & execution;
    ArConfig config;
    DiffusionConfig diffusion_config;
    core::BackendWeightStore weights;
    modules::CausalDecoderRuntimeWeights ar_weights;
    ConnectorWeights connector;
    std::unique_ptr<modules::CausalDecoderRuntime> positive, negative_prefill;
    bool batched_cfg = false;
    DiffusionHead diffusion;
    AcousticDecoder codec;
    SpeechScheduler scheduler;
    tokenizers::LlamaBpeTokenizer tokenizer;
    float scale, bias;
    std::unique_ptr<ggml_context, decltype(&ggml_free)> graph_context{nullptr, ggml_free};
    std::array<ggml_cgraph *, 2> graphs{};
    std::array<ggml_gallocr_t, 2> allocators{};
    std::array<core::HostGraphPlan, 2> plans;
    core::TensorValue ids, latent, token_embeddings, acoustic_embedding;

    Impl(const assets::TensorSource & source, const std::filesystem::path & tokenizer_dir,
         core::ExecutionContext & execution_, assets::TensorStorageType storage,
         ArConfig ar_config, DiffusionConfig diffusion_config_, CodecConfig codec_config)
        : execution(execution_), config(ar_config), diffusion_config(diffusion_config_),
          weights(execution.backend(), execution.backend_type(), "kugelaudio.ar.weights", 4 * 1024 * 1024),
          ar_weights(load_ar_weights(weights, source, storage, config)),
          connector(load_connector_weights(weights, source, storage, diffusion_config.latent_size, config.hidden_size)),
          diffusion(source, execution, storage, diffusion_config), codec(source, execution, storage, codec_config),
          tokenizer({{}, {}, tokenizer_dir / "tokenizer_config.json", tokenizer_dir / "tokenizer.json",
                     tokenizers::LlamaBpePreTokenizer::Qwen2}),
          scale(source.require_f32("model.speech_scaling_factor").at(0)),
          bias(source.require_f32("model.speech_bias_factor").at(0)) {
        weights.upload();
        positive = std::make_unique<modules::CausalDecoderRuntime>(execution, ar_runtime_config(config), ar_weights);
        auto negative_config = ar_runtime_config(config);
        negative_config.trace_name = "kugelaudio.ar.negative";
        negative_config.output_mode = modules::CausalDecoderOutputMode::Hidden;
        negative_prefill = std::make_unique<modules::CausalDecoderRuntime>(execution, negative_config, ar_weights);

        graph_context.reset(ggml_init({4 * 1024 * 1024, nullptr, true}));
        core::ModuleBuildContext ctx{graph_context.get(), "kugelaudio.conditioning", execution.backend_type()};
        ids = core::make_tensor(ctx, GGML_TYPE_I32, core::TensorShape::from_dims({256}));
        latent = core::make_tensor(ctx, GGML_TYPE_F32, core::TensorShape::from_dims({1, diffusion_config.latent_size}));
        ggml_set_input(ids.tensor);
        ggml_set_input(latent.tensor);
        token_embeddings = modules::EmbeddingModule({config.vocab_size, config.hidden_size}).build(ctx, ids, ar_weights.token_embedding);
        acoustic_embedding = build_connector(ctx, latent, connector, config.hidden_size);
        const std::array<ggml_tensor *, 2> outputs{token_embeddings.tensor, acoustic_embedding.tensor};
        for (size_t i = 0; i < graphs.size(); ++i) {
            ggml_set_output(outputs[i]);
            graphs[i] = ggml_new_graph_custom(graph_context.get(), 512, false);
            ggml_build_forward_expand(graphs[i], outputs[i]);
            core::validate_backend_graph_supported(execution.backend(), graphs[i], "KugelAudio conditioning");
            allocators[i] = ggml_gallocr_new(ggml_backend_get_default_buffer_type(execution.backend()));
            if (!ggml_gallocr_alloc_graph(allocators[i], graphs[i])) {
                throw std::runtime_error("KugelAudio conditioning graph allocation failed");
            }
            core::prepare_host_graph_plan(execution, graphs[i], plans[i]);
        }
    }

    ~Impl() {
        for (size_t i = 0; i < graphs.size(); ++i) {
            if (graphs[i]) {
                core::release_backend_graph_resources(execution.backend(), graphs[i], true);
            }
            plans[i].reset();
            ggml_gallocr_free(allocators[i]);
        }
    }

    std::vector<float> embed_tokens(const std::vector<int32_t> & tokens) {
        std::vector<float> output;
        output.reserve(tokens.size() * config.hidden_size);
        for (size_t offset = 0; offset < tokens.size(); offset += 256) {
            std::array<int32_t, 256> block{};
            const auto count = std::min<size_t>(256, tokens.size() - offset);
            std::copy_n(tokens.begin() + offset, count, block.begin());
            ggml_backend_tensor_set(ids.tensor, block.data(), 0, sizeof(block));
            if (core::compute_graph(execution, graphs[0], plans[0], "KugelAudio embeddings") != GGML_STATUS_SUCCESS) {
                throw std::runtime_error("KugelAudio embedding compute failed");
            }
            const auto values = core::read_tensor_f32(token_embeddings.tensor);
            output.insert(output.end(), values.begin(), values.begin() + count * config.hidden_size);
        }
        return output;
    }

    std::vector<float> connect(const std::vector<float> & values) {
        core::write_tensor_f32(latent, values);
        if (core::compute_graph(execution, graphs[1], plans[1], "KugelAudio connector") != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("KugelAudio connector compute failed");
        }
        return core::read_tensor_f32(acoustic_embedding.tensor);
    }

    std::vector<float> generate(const std::string & text, const VoiceFeatures & voice,
        const GenerationOptions & options, const std::function<void(const std::vector<float> &)> & on_audio) {
        if (options.max_tokens <= 0 || options.steps <= 0 || voice.mean.empty() ||
            voice.mean.size() % diffusion_config.latent_size != 0) {
            throw std::runtime_error("invalid KugelAudio generation dimensions");
        }
        const auto begin = std::chrono::steady_clock::now();
        std::mt19937 rng(options.seed);
        const auto voice_frames = voice.mean.size() / diffusion_config.latent_size;
        auto prompt = tokenizer.encode(" Transform the text provided by various speakers into speech output, utilizing the distinct voice of each respective speaker.\n", false);
        auto append = [&](const std::string & value) {
            auto encoded = tokenizer.encode(value, false);
            prompt.insert(prompt.end(), encoded.begin(), encoded.end());
        };
        append(" Voice input:\n");
        append(" Speaker 0:");
        const size_t voice_offset = prompt.size();
        prompt.insert(prompt.end(), voice_frames, kSpeechTokens[2]);
        append("\n");
        append(" Text input:\n");
        const auto first = text.find_first_not_of(" \t\r\n");
        if (first == std::string::npos) {
            throw std::runtime_error("KugelAudio requires non-empty text");
        }
        const auto cleaned = text.substr(first, text.find_last_not_of(" \t\r\n") - first + 1);
        append(" " + (cleaned.rfind("Speaker", 0) == 0 ? cleaned : "Speaker 0: " + cleaned) + "\n");
        append(" Speech output:\n");
        prompt.push_back(kSpeechTokens[0]);
        if (prompt.size() + options.max_tokens > static_cast<size_t>(config.max_position_embeddings)) {
            throw std::runtime_error("KugelAudio prompt and max_tokens exceed the context limit");
        }
        auto embeddings = embed_tokens(prompt);
        const auto voice_noise = sampling::generate_normal_noise(voice.mean.size(), rng());
        std::vector<float> frame(diffusion_config.latent_size);
        for (size_t i = 0; i < voice_frames; ++i) {
            for (size_t j = 0; j < frame.size(); ++j) {
                const auto index = i * frame.size() + j;
                frame[j] = (voice.mean[index] + voice.std * voice_noise[index] + bias) * scale;
            }
            const auto connected = connect(frame);
            std::copy(connected.begin(), connected.end(), embeddings.begin() + (voice_offset + i) * config.hidden_size);
        }
        const auto control_embeddings = embed_tokens({kSpeechTokens.begin(), kSpeechTokens.end()});
        const auto conditioning_ms = debug::elapsed_ms(begin);
        auto stage_start = std::chrono::steady_clock::now();
        const bool use_cfg = options.guidance_scale != 1;
        if (batched_cfg != use_cfg) {
            positive->release_runtime_graphs();
            negative_prefill->release_runtime_graphs();
            batched_cfg = use_cfg;
        }
        auto prefill = positive->prefill_embeddings(embeddings, prompt.size());
        modules::CausalDecoderStepResult current{std::move(prefill.logits), std::move(prefill.hidden)};
        if (use_cfg) {
            auto unconditional = negative_prefill->prefill_embeddings(
                std::vector<float>(control_embeddings.begin(), control_embeddings.begin() + config.hidden_size), 1);
            runtime::TransformerBatchedKVState state;
            state.batch_size = 2;
            state.current_end = prefill.state.current_end;
            state.current_end_by_batch = {prefill.state.current_end, unconditional.state.current_end};
            state.valid_steps_by_batch = {static_cast<int64_t>(prompt.size()), 1};
            state.layers.resize(prefill.state.layers.size());
            for (size_t layer = 0; layer < state.layers.size(); ++layer) {
                auto & packed = state.layers[layer];
                auto & conditional = prefill.state.layers[layer];
                const auto & unconditioned = unconditional.state.layers[layer];
                packed.valid_steps = conditional.valid_steps;
                packed.key = std::move(conditional.key);
                packed.value = std::move(conditional.value);
                packed.key.insert(packed.key.end(), unconditioned.key.begin(), unconditioned.key.end());
                packed.value.insert(packed.value.end(), unconditioned.value.begin(), unconditioned.value.end());
            }
            positive->start_decode_embeddings_batched(state, prompt.size() + options.max_tokens);
            current.hidden.insert(current.hidden.end(), unconditional.hidden.begin(), unconditional.hidden.end());
        } else {
            positive->start_decode_embeddings(prefill.state, prompt.size() + options.max_tokens);
        }
        prefill.state = {};
        const auto prefill_ms = debug::elapsed_ms(stage_start);
        double ar_ms = 0, diffusion_ms = 0, codec_ms = 0, connector_ms = 0;
        codec.reset();
        bool negative_has_audio = false;
        std::vector<float> next_embedding(control_embeddings.begin(), control_embeddings.begin() + config.hidden_size);
        std::vector<float> audio;
        sampling::HfSampler token_sampler;
        sampling::HfSamplerScratch scratch;
        sampling::HfSamplingOptions sampling_options;
        sampling_options.do_sample = options.do_sample && options.temperature > 0;
        sampling_options.temperature = options.temperature;
        for (int64_t step = 0; step < options.max_tokens; ++step) {
            const auto index = token_sampler.sample(current.logits, {}, sampling_options, scratch, rng, nullptr, "KugelAudio AR");
            const auto token = kSpeechTokens.at(index);
            if (token == kSpeechTokens[1] || token == kSpeechTokens[3]) {
                break;
            }
            if (token == kSpeechTokens[2]) {
                negative_has_audio = true;
                stage_start = std::chrono::steady_clock::now();
                auto sample = sampling::generate_normal_noise(frame.size() * (use_cfg ? 2 : 1), rng());
                scheduler.reset(options.steps);
                diffusion.begin_frame(current.hidden, scheduler.timesteps());
                for (size_t diffusion_step = 0; diffusion_step < scheduler.timesteps().size(); ++diffusion_step) {
                    auto input = sample;
                    if (use_cfg) {
                        std::copy_n(sample.begin(), frame.size(), input.begin() + frame.size());
                    }
                    auto prediction = diffusion.predict(input, diffusion_step);
                    if (use_cfg) {
                        for (size_t j = 0; j < frame.size(); ++j) {
                            const auto guided = prediction[j + frame.size()] + options.guidance_scale * (prediction[j] - prediction[j + frame.size()]);
                            prediction[j] = prediction[j + frame.size()] = guided;
                        }
                    }
                    scheduler.step(sample, prediction, sampling::generate_normal_noise(sample.size(), rng()));
                }
                sample.resize(frame.size());
                diffusion_ms += debug::elapsed_ms(stage_start);
                stage_start = std::chrono::steady_clock::now();
                next_embedding = connect(sample);
                connector_ms += debug::elapsed_ms(stage_start);
                for (size_t j = 0; j < frame.size(); ++j) {
                    frame[j] = sample[j] / scale - bias;
                }
                stage_start = std::chrono::steady_clock::now();
                auto chunk = codec.decode_frame(frame);
                codec_ms += debug::elapsed_ms(stage_start);
                audio.insert(audio.end(), chunk.begin(), chunk.end());
                if (on_audio) {
                    on_audio(chunk);
                }
            } else {
                next_embedding.assign(control_embeddings.begin() + index * config.hidden_size,
                                      control_embeddings.begin() + (index + 1) * config.hidden_size);
                if (use_cfg && negative_has_audio) {
                    // Keep the conditional history while starting a new negative segment.
                    stage_start = std::chrono::steady_clock::now();
                    auto state = positive->export_batched_decode_state();
                    const auto positive_elems = static_cast<size_t>(
                        state.valid_steps_by_batch[0] * config.kv_heads * config.head_dim);
                    for (auto & layer : state.layers) {
                        layer.key.resize(positive_elems);
                        layer.value.resize(positive_elems);
                        layer.valid_steps = state.valid_steps_by_batch[0];
                    }
                    state.current_end_by_batch[1] = 0;
                    state.valid_steps_by_batch[1] = 0;
                    positive->start_decode_embeddings_batched(state, prompt.size() + options.max_tokens);
                    ar_ms += debug::elapsed_ms(stage_start);
                }
                negative_has_audio = false;
            }
            if (step + 1 < options.max_tokens) {
                stage_start = std::chrono::steady_clock::now();
                if (use_cfg) {
                    auto paired = next_embedding;
                    paired.insert(paired.end(), next_embedding.begin(), next_embedding.end());
                    current = positive->decode_embeddings_batched(paired, 2);
                    current.logits.resize(kSpeechTokens.size());
                } else {
                    current = positive->decode_embedding(next_embedding);
                }
                ar_ms += debug::elapsed_ms(stage_start);
            }
        }
        if (audio.empty()) {
            throw std::runtime_error("KugelAudio generated no audio");
        }
        float peak = 0;
        for (const auto value : audio) {
            peak = std::max(peak, std::abs(value));
        }
        if (peak > 1 && !on_audio) {
            for (auto & value : audio) {
                value *= 0.95F / peak;
            }
        }
        debug::timing_log_scalar("kugelaudio.conditioning_ms", conditioning_ms);
        debug::timing_log_scalar("kugelaudio.prefill_ms", prefill_ms);
        debug::timing_log_scalar("kugelaudio.ar_ms", ar_ms);
        debug::timing_log_scalar("kugelaudio.diffusion_ms", diffusion_ms);
        debug::timing_log_scalar("kugelaudio.codec_ms", codec_ms);
        debug::timing_log_scalar("kugelaudio.connector_ms", connector_ms);
        debug::timing_log_scalar("kugelaudio.generation_ms", debug::elapsed_ms(begin));
        return audio;
    }
};

Generator::Generator(const assets::TensorSource & source, const std::filesystem::path & tokenizer_dir,
    core::ExecutionContext & execution, assets::TensorStorageType storage,
    ArConfig ar_config, DiffusionConfig diffusion_config, CodecConfig codec_config)
    : impl_(std::make_unique<Impl>(source, tokenizer_dir, execution, storage, ar_config, diffusion_config, codec_config)) {}
Generator::~Generator() = default;
std::vector<float> Generator::generate(const std::string & text, const VoiceFeatures & voice,
    const GenerationOptions & options, const std::function<void(const std::vector<float> &)> & on_audio) {
    return impl_->generate(text, voice, options, on_audio);
}

}  // namespace engine::models::kugelaudio
