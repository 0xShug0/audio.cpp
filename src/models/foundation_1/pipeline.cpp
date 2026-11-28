#include "engine/models/foundation_1/pipeline.h"

#include "engine/framework/core/prepared_graph.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/modules/linear_module.h"
#include "engine/framework/modules/transformers/dit_runtime.h"
#include "engine/framework/sampling/diffusion_math.h"
#include "engine/framework/sampling/flow_sampler_runtime.h"
#include "engine/framework/sampling/torch_random.h"
#include "engine/framework/tokenizers/sentencepiece.h"

#include <algorithm>
#include <chrono>
#include <cmath>

namespace engine::models::foundation_1 {
namespace {
constexpr double kHalfPi = 1.57079632679489661923;

// The generic sampler owns sequence execution; only the model's v-prediction
// DDIM update differs from its built-in rectified-flow updates.
class DdimUpdate final : public modules::FlowSamplerUpdateRuntime {
public:
    void update_latent(const modules::FlowSamplerUpdateInput & input) override {
        const auto & step = input.state.schedule;
        const float alpha = static_cast<float>(std::cos(kHalfPi * step.t));
        const float sigma = static_cast<float>(std::sin(kHalfPi * step.t));
        const float next_alpha = static_cast<float>(std::cos(kHalfPi * step.t_next));
        const float next_sigma = static_cast<float>(std::sin(kHalfPi * step.t_next));
        for (size_t i = 0; i < input.latent.size(); ++i) {
            const float clean = input.latent[i] * alpha - input.prediction[i] * sigma;
            const float noise = input.latent[i] * sigma + input.prediction[i] * alpha;
            input.latent[i] = clean * next_alpha + noise * next_sigma;
        }
    }
};
} // namespace

struct FoundationPipeline::Impl {
    std::shared_ptr<const FoundationAssets> assets;
    core::ExecutionContext & execution;
    std::shared_ptr<tokenizers::SentencePieceTokenizer> tokenizer;
    modules::T5EncoderRuntime text;
    modules::DitRuntime denoiser;
    codecs::OobleckAudioVaeRuntime codec;
    NumberConditioningWeights numbers;
    core::PreparedGraph number_graph;
    core::TensorValue start_input, duration_input, start_output, duration_output;

    Impl(std::shared_ptr<const FoundationAssets> source, core::ExecutionContext & context,
         assets::TensorStorageType storage)
        : assets(std::move(source)), execution(context),
          tokenizer(tokenizers::load_sentencepiece_tokenizer(assets->resources.require_file("tokenizer"))),
          text(context, assets->text_config,
              [a = assets, &context, storage] { return load_foundation_text_weights(a, context, storage); },
              [] { modules::T5EncoderRuntimeConfig c; c.label = "foundation_1.text";
                   c.sequence_padding_multiple = 128; return c; }()),
          denoiser(context, *assets->weights, [&] {
              modules::DitRuntimeConfig c; c.label = "foundation_1.dit";
              c.model = assets->dit_config; c.weight_type = storage;
              c.time_features = modules::DitTimeFeatures::CosSin;
              return c;
          }(), assets->dit_names),
          codec(assets->weights, context, assets->codec_config, [] {
              codecs::OobleckAudioVaeRuntimeOptions o;
              o.graph_arena_bytes = 32ull << 20; o.weight_context_bytes = 8ull << 20; return o;
          }()),
          numbers(load_foundation_number_weights(*assets, context)),
          number_graph(context, {"foundation_1.numbers", 1ull << 20, 128}) {
        auto & ctx = number_graph.build_context();
        start_input = core::make_tensor(ctx, GGML_TYPE_F32, core::TensorShape::from_dims({1, 257}));
        duration_input = core::make_tensor(ctx, GGML_TYPE_F32, core::TensorShape::from_dims({1, 257}));
        ggml_set_input(start_input.tensor); ggml_set_output(start_input.tensor);
        ggml_set_input(duration_input.tensor); ggml_set_output(duration_input.tensor);
        start_output = modules::LinearModule({257, 768, true}).build(ctx, start_input, numbers.start);
        duration_output = modules::LinearModule({257, 768, true}).build(ctx, duration_input, numbers.duration);
        ggml_set_output(start_output.tensor); ggml_set_output(duration_output.tensor);
        ggml_build_forward_expand(number_graph.graph(), start_output.tensor);
        ggml_build_forward_expand(number_graph.graph(), duration_output.tensor);
        number_graph.allocate();
    }

    runtime::AudioBuffer generate(const std::string & prompt, const GenerationOptions & options) {
        const auto started = std::chrono::steady_clock::now();
        std::vector<std::vector<int32_t>> ids;
        std::vector<std::string> prompts{prompt};
        if (options.guidance_scale != 1 && !options.negative_prompt.empty()) prompts.push_back(options.negative_prompt);
        for (const auto & value : prompts) {
            auto tokens = tokenizer->tokenize(value).token_ids;
            if (tokens.size() >= static_cast<size_t>(assets->max_text_tokens))
                tokens.resize(static_cast<size_t>(assets->max_text_tokens - 1));
            tokens.push_back(1); // T5 EOS, retained after truncation.
            ids.push_back(std::move(tokens));
        }
        auto embeddings = text.encode_batch(ids);
        std::vector<float> start_features(257), duration_features(257);
        const float normalized = options.duration_sec / 512.f;
        duration_features[0] = normalized;
        for (size_t i = 0; i < 128; ++i) {
            start_features[129 + i] = 1.f;
            const float angle = normalized * numbers.duration_frequencies[i] * static_cast<float>(4 * kHalfPi);
            duration_features[1 + i] = std::sin(angle);
            duration_features[129 + i] = std::cos(angle);
        }
        core::write_tensor_f32(start_input, start_features);
        core::write_tensor_f32(duration_input, duration_features);
        number_graph.compute();
        const auto start = core::read_tensor_f32(start_output.tensor);
        const auto duration = core::read_tensor_f32(duration_output.tensor);
        constexpr int64_t width = 768, channels = 64, ratio = 2048;
        const int64_t audio_frames = static_cast<int64_t>(std::ceil(options.duration_sec * assets->sample_rate));
        const int64_t frames = (audio_frames + ratio - 1) / ratio;
        modules::DitConditioning condition;
        condition.frames = frames; condition.branches = options.guidance_scale == 1 ? 1 : 2;
        condition.context_tokens = assets->max_text_tokens + 2;
        const auto branch_values = static_cast<size_t>(condition.context_tokens * width);
        condition.context.resize(static_cast<size_t>(condition.branches) * branch_values, 0);
        condition.global.reserve(static_cast<size_t>(condition.branches * width * 2));
        for (int64_t b = 0; b < condition.branches; ++b) {
            condition.global.insert(condition.global.end(), start.begin(), start.end());
            condition.global.insert(condition.global.end(), duration.begin(), duration.end());
            if (b && options.negative_prompt.empty()) continue;
            auto dst = condition.context.begin() + b * static_cast<int64_t>(branch_values);
            std::copy(embeddings[static_cast<size_t>(b)].begin(), embeddings[static_cast<size_t>(b)].end(), dst);
            std::copy(start.begin(), start.end(), dst + assets->max_text_tokens * width);
            std::copy(duration.begin(), duration.end(), dst + (assets->max_text_tokens + 1) * width);
        }
        denoiser.prepare_conditioning(std::move(condition));
        debug::timing_log_scalar("foundation_1.condition_ms", debug::elapsed_ms(started));
        const auto sampled_at = std::chrono::steady_clock::now();
        modules::FlowSamplerRuntimeConfig config;
        config.label = "foundation_1.ddim"; config.latent_shape = {1, channels, frames};
        config.branches = {{"v", 1}}; config.update_rule = modules::FlowSamplerUpdateRule::Custom;
        for (int64_t i = 0; i < options.num_steps; ++i)
            config.schedule.push_back({i, 1.f - float(i) / float(options.num_steps),
                1.f - float(i + 1) / float(options.num_steps), 0, 0});
        modules::FlowSamplerSingleBranchDenoiserConfig prediction;
        prediction.label = config.label; prediction.branch_name = "v";
        prediction.sampler_mode = [](const auto &) { return "ddim"; };
        prediction.predict = [&](const modules::FlowSamplerDenoiserInput & input) {
            if (options.guidance_scale == 1) return denoiser.predict(input.latent, input.state.schedule.t);
            const auto both = denoiser.predict_cfg(input.latent, input.state.schedule.t);
            const auto n = input.latent.size();
            return sampling::cfg_guidance(std::vector<float>(both.begin(), both.begin() + n),
                std::vector<float>(both.begin() + n, both.end()), options.guidance_scale);
        };
        modules::FlowSamplerRuntime sampler(config,
            modules::make_flow_sampler_single_branch_denoiser(std::move(prediction)), std::make_unique<DdimUpdate>());
        const auto policy = sampling::resolve_torch_cuda_sampling_policy(execution.backend_type(),
            execution.config().device, "foundation_1", "Foundation-1");
        auto noise = sampling::generate_torch_cuda_tensor_iterator_randn(
            static_cast<size_t>(channels * frames), options.seed, 0, policy);
        sampler.run_sequence(noise);
        debug::timing_log_scalar("foundation_1.sample_ms", debug::elapsed_ms(sampled_at));
        const auto decoded_at = std::chrono::steady_clock::now();
        auto audio = codec.decode(sampler.latent(), 1, frames).front();
        audio.samples.resize(static_cast<size_t>(audio_frames * audio.channels));
        debug::timing_log_scalar("foundation_1.decode_ms", debug::elapsed_ms(decoded_at));
        return audio;
    }
};

FoundationPipeline::FoundationPipeline(std::shared_ptr<const FoundationAssets> assets,
    core::ExecutionContext & execution, assets::TensorStorageType storage)
    : impl_(std::make_unique<Impl>(std::move(assets), execution, storage)) {}
FoundationPipeline::~FoundationPipeline() = default;
runtime::AudioBuffer FoundationPipeline::generate(const std::string & prompt, const GenerationOptions & options) {
    return impl_->generate(prompt, options);
}

} // namespace engine::models::foundation_1
