#include "engine/models/soulx_singer/session.h"
#include "engine/models/soulx_singer/runtime.h"
#include "engine/framework/runtime/session_base.h"
#include "engine/framework/runtime/spec_backed_model.h"
#include "engine/framework/runtime/options.h"
#include "engine/framework/runtime/errors.h"
#include "engine/framework/debug/profiler.h"
#include <chrono>
#include <algorithm>
#include <cmath>

namespace engine::models::soulx_singer {
namespace {
class Session : public runtime::RuntimeSessionBase, public runtime::IOfflineVoiceTaskSession {
  public:
    Session(runtime::TaskSpec task, const runtime::SessionOptions &options,
            std::shared_ptr<const ModelAssets> assets,
            std::shared_ptr<const model_spec::ModelContract> contract)
        : RuntimeSessionBase(options), task_(task), assets_(assets), contract_(std::move(contract)) {
        runtime::validate_spec_backed_session_options(options, *contract_, "soulx_singer", "SoulX-Singer");
        const auto storage = assets::parse_tensor_storage_type(
            runtime::find_option(options.options, {"soulx_singer.weight_type", "weight_type"})
                .value_or("native"));
        const auto attention = core::parse_attention_preference(
            runtime::find_option(options.options, {"soulx_singer.attention", "attention"})
                .value_or("auto"),
            "soulx_singer.attention");
        model_ =
            std::make_unique<SoulXSingerRuntime>(std::move(assets), execution_context(), storage, attention);
    }
    std::string family() const override { return "soulx_singer"; }
    runtime::VoiceTaskKind task_kind() const override { return task_.task; }
    runtime::RunMode run_mode() const override { return task_.mode; }
    void prepare(const runtime::SessionPreparationRequest &request) override {
        voice_ = request.voice;
        mark_prepared();
    }
    runtime::TaskResult run(const runtime::TaskRequest &request) override {
        require_prepared("SoulX-Singer run");
        try {
            runtime::validate_spec_backed_request_options(request.options, request.option_arrays, *contract_,
                                                         "SoulX-Singer");
        } catch (const std::runtime_error &error) {
            throw runtime::InvalidRequestError(error.what());
        }
        if (!request.option_arrays.empty())
            throw runtime::InvalidRequestError("SoulX-Singer does not accept list-valued request options");
        const auto &voice = request.voice ? request.voice : voice_;
        if (!voice || !voice->speaker || !voice->speaker->audio)
            throw runtime::InvalidRequestError("SoulX-Singer requires reference voice audio");
        GenerationOptions o;
        o.steps = runtime::parse_i64_option(request.options, {"num_steps"}).value_or(32);
        o.guidance = runtime::parse_finite_float_option(request.options, {"guidance_scale"}).value_or(3);
        o.pitch_shift = runtime::parse_int_option(request.options, {"pitch_shift"}).value_or(0);
        if (o.steps <= 0 || o.guidance < 0 || o.pitch_shift < -24 || o.pitch_shift > 24)
            throw runtime::InvalidRequestError(
                "SoulX-Singer requires num_steps>0, guidance_scale>=0 and pitch_shift in [-24,24]");
        const auto auto_shift = runtime::find_option(request.options, {"auto_pitch_shift"});
        o.auto_pitch_shift = auto_shift && runtime::parse_bool_option(*auto_shift, "auto_pitch_shift");
        const auto seed = runtime::parse_i64_option(request.options, {"seed"}).value_or(2026);
        if (seed < -1)
            throw runtime::InvalidRequestError("SoulX-Singer seed must be -1 or nonnegative");
        o.seed = seed < 0 ? runtime::random_u64_seed() : static_cast<uint64_t>(seed);
        const auto started = std::chrono::steady_clock::now();
        runtime::TaskResult result;
        result.audio_output = generate(request, *voice->speaker->audio, o);
        debug::timing_log_scalar("session.wall_ms", debug::elapsed_ms(started));
        return result;
    }

  protected:
    virtual runtime::AudioBuffer generate(const runtime::TaskRequest &, const runtime::AudioBuffer &,
                                           const GenerationOptions &) = 0;
    runtime::TaskSpec task_;
    std::shared_ptr<const ModelAssets> assets_;
    std::shared_ptr<const model_spec::ModelContract> contract_;
    std::unique_ptr<SoulXSingerRuntime> model_;
    std::optional<runtime::VoiceCondition> voice_;
};

class SvcSession final : public Session {
  public:
    using Session::Session;
  private:
    runtime::AudioBuffer generate(const runtime::TaskRequest &request, const runtime::AudioBuffer &voice,
                                  const GenerationOptions &options) override {
        if (!request.audio_input)
            throw runtime::InvalidRequestError("SoulX-Singer SVC requires source singing audio");
        return model_->convert(*request.audio_input, voice, options);
    }
};

class SvsSession final : public Session {
  public:
    using Session::Session;
  private:
    runtime::AudioBuffer generate(const runtime::TaskRequest &request,
                                          const runtime::AudioBuffer &voice,
                                          const GenerationOptions &options) override {
        const auto prompt_path = runtime::find_option(request.options, {"reference_metadata_file"});
        const auto target_path = runtime::find_option(request.options, {"target_metadata_file"});
        if (!prompt_path || !target_path)
            throw runtime::InvalidRequestError("SVS requires reference_metadata_file and target_metadata_file JSON files");
        const auto prompt_meta = io::json::parse_file(*prompt_path);
        const auto target_meta = io::json::parse_file(*target_path);
        if (prompt_meta.as_array().empty() || target_meta.as_array().empty())
            throw runtime::InvalidRequestError("SVS metadata must contain singing segments");
        const auto prompt_score = parse_score(prompt_meta.as_array().front(), assets_->phones);
        const auto control = runtime::find_option(request.options, {"control"}).value_or("melody");
        if (control != "score" && control != "melody")
            throw runtime::InvalidRequestError("SVS control must be score or melody");
        const bool melody = control == "melody";
        const auto &segments = target_meta.as_array();
        const auto final_ms = segments.back().require("time").as_array().at(1).as_number();
        if (!std::isfinite(final_ms) || final_ms <= 0)
            throw runtime::InvalidRequestError("SVS metadata has invalid end time");
        runtime::AudioBuffer output{24000, 1, std::vector<float>(static_cast<size_t>(final_ms * 24), 0)};
        uint64_t offset = 0;
        for (const auto &segment : segments) {
            const auto score = parse_score(segment, assets_->phones);
            int shift = options.pitch_shift;
            if (options.auto_pitch_shift && shift == 0) {
                const auto median = [](const auto &input) {
                    std::vector<double> values;
                    for (auto v : input)
                        if (v > 0)
                            values.push_back(v);
                    if (values.empty())
                        throw runtime::InvalidRequestError("Automatic pitch shift requires voiced notes");
                    std::sort(values.begin(), values.end());
                    return values[(values.size() - 1) / 2];
                };
                shift = static_cast<int>(
                    std::nearbyint(melody ? 12 * std::log2(median(prompt_score.f0) / median(score.f0))
                                          : median(prompt_score.pitches) - median(score.pitches)));
            }
            auto [prompt, target] = model_->encode_scores(prompt_score, score, shift, melody, voice);
            model_->prepare_conditioning(prompt, target, shift);
            auto noise =
                sampling::generate_torch_cuda_randn(static_cast<size_t>(target.frames * 128), options.seed,
                                                    sampling::TorchRandnPrecision::Float32, offset);
            offset += noise.size();
            const auto wave = model_->decode(model_->denoise(noise, options));
            const auto start_ms = segment.require("time").as_array().at(0).as_number();
            if (!std::isfinite(start_ms) || start_ms < 0 || start_ms >= final_ms)
                throw runtime::InvalidRequestError("SVS metadata has invalid segment start");
            const auto start = static_cast<size_t>(start_ms * 24);
            std::copy_n(wave.begin(), std::min(wave.size(), output.samples.size() - start),
                        output.samples.begin() + static_cast<std::ptrdiff_t>(start));
        }
        return output;
    }
};
} // namespace

std::unique_ptr<runtime::IVoiceTaskSession> make_svc_session(
    const runtime::TaskSpec &task, const runtime::SessionOptions &options,
    std::shared_ptr<const ModelAssets> assets, std::shared_ptr<const model_spec::ModelContract> contract) {
    return std::make_unique<SvcSession>(task, options, std::move(assets), std::move(contract));
}
std::unique_ptr<runtime::IVoiceTaskSession> make_svs_session(
    const runtime::TaskSpec &task, const runtime::SessionOptions &options,
    std::shared_ptr<const ModelAssets> assets, std::shared_ptr<const model_spec::ModelContract> contract) {
    return std::make_unique<SvsSession>(task, options, std::move(assets), std::move(contract));
}
} // namespace engine::models::soulx_singer
