#include "engine/models/mossformer2/session.h"
#include "engine/models/mossformer2/runtime.h"
#include "engine/framework/audio/conversion.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/runtime/session_base.h"
#include "engine/framework/runtime/spec_backed_model.h"
#include "engine/framework/runtime/options.h"

#include <algorithm>
#include <cmath>

namespace engine::models::mossformer2 {
namespace {

class Session final : public runtime::RuntimeSessionBase, public runtime::IOfflineVoiceTaskSession {
public:
    Session(runtime::TaskSpec task, runtime::SessionOptions options,
        std::shared_ptr<const MossFormer2Assets> assets, std::shared_ptr<const model_spec::ModelContract> contract)
        : RuntimeSessionBase(options), task_(task), assets_(std::move(assets)), contract_(std::move(contract)) {
        runtime::validate_spec_backed_session_options(options, *contract_, "mossformer2", "MossFormer2");
        if (task.task != runtime::VoiceTaskKind::SourceSeparation || task.mode != runtime::RunMode::Offline) {
            throw std::runtime_error("MossFormer2 supports offline source separation");
        }
        runtime_ = std::make_unique<MossFormer2GatedFSMNRuntime>(assets_, execution_context());
    }
    std::string family() const override { return "mossformer2"; }
    runtime::VoiceTaskKind task_kind() const override { return task_.task; }
    runtime::RunMode run_mode() const override { return task_.mode; }
    void prepare(const runtime::SessionPreparationRequest &) override { mark_prepared(); }

    runtime::TaskResult run(const runtime::TaskRequest & request) override {
        require_prepared("MossFormer2 run");
        const auto started = std::chrono::steady_clock::now();
        runtime::validate_spec_backed_request_options(request.options, request.option_arrays, *contract_, "MossFormer2");
        if (!request.audio_input) throw std::runtime_error("MossFormer2 requires input audio");
        const auto & audio = *request.audio_input;
        if (audio.sample_rate <= 0 || audio.channels <= 0 || audio.samples.empty() || audio.samples.size() % audio.channels) {
            throw std::runtime_error("MossFormer2 invalid input audio");
        }
        auto input = audio::convert_interleaved_audio_to_mono_torchaudio_sinc_hann_resampled(
            audio.samples, audio.sample_rate, audio.channels, assets_->config.sample_rate);
        if (input.size() < static_cast<size_t>(assets_->config.kernel)) throw std::runtime_error("MossFormer2 audio is shorter than the encoder kernel");
        const auto seconds = runtime::parse_finite_float_option(request.options, {"audio_chunk_duration_sec"}).value_or(2.0f);
        const auto overlap_seconds = runtime::parse_finite_float_option(request.options, {"audio_chunk_overlap_sec"}).value_or(0.5f);
        if (seconds < 0) throw std::runtime_error("MossFormer2 chunk duration must be non-negative");
        if (overlap_seconds < 0 || (seconds > 0 && overlap_seconds >= seconds)) {
            throw std::runtime_error("MossFormer2 chunk overlap must be non-negative and smaller than chunk duration");
        }
        const int64_t samples = input.size();
        const int64_t window = std::llrint(seconds * assets_->config.sample_rate);
        if (seconds > 0 && window < assets_->config.kernel) throw std::runtime_error("MossFormer2 chunk duration is shorter than the encoder kernel");
        double energy = 0;
        for (float value : input) energy += double(value) * value;
        const double rms = std::sqrt(energy / samples);
        std::vector<std::vector<float>> outputs;
        if (window == 0) {
            outputs = runtime_->separate(input);
        } else {
            const int64_t overlap = std::llrint(double(overlap_seconds) * assets_->config.sample_rate);
            const int64_t stride = window - overlap;
            if (stride <= 0) throw std::runtime_error("MossFormer2 chunk overlap leaves no hop samples");
            int64_t padded = samples;
            if (samples < window) padded = window;
            else if (samples < window + stride) padded = window + stride;
            else if ((samples - window) % stride) padded += samples - (samples - window) / stride * stride;
            input.resize(padded, 0.0f);
            if (samples <= window) {
                outputs = runtime_->separate(input);
            } else {
                outputs.assign(assets_->config.speakers, std::vector<float>(padded, 0.0f));
                const int64_t discard = (window - stride) / 2;
                std::vector<float> chunk(window);
                for (int64_t offset = 0; offset + window <= padded; offset += stride) {
                    std::copy_n(input.begin() + offset, window, chunk.begin());
                    auto separated = runtime_->separate(chunk);
                    const int64_t first = offset == 0 ? 0 : discard;
                    for (int speaker = 0; speaker < assets_->config.speakers; ++speaker) {
                        std::copy(separated[speaker].begin() + first, separated[speaker].end() - discard,
                            outputs[speaker].begin() + offset + first);
                    }
                }
            }
        }
        const auto option = runtime::find_option(request.options, {"normalize_output"});
        const bool normalize = !option || runtime::parse_bool_option(*option, "normalize_output");
        runtime::TaskResult result;
        for (size_t speaker = 0; speaker < outputs.size(); ++speaker) {
            auto & waveform = outputs[speaker];
            waveform.resize(samples);
            if (normalize) {
                double power = 0;
                for (float value : waveform) power += double(value) * value;
                if (power > 0) {
                    const float scale = rms / std::sqrt(power / waveform.size());
                    for (float & value : waveform) value *= scale;
                }
            }
            runtime::NamedAudioBuffer named;
            named.id = "speaker_" + std::to_string(speaker + 1);
            named.audio = runtime::AudioBuffer{assets_->config.sample_rate, 1, std::move(waveform)};
            result.named_audio_outputs.push_back(std::move(named));
        }
        debug::timing_log_scalar("session.wall_ms", debug::elapsed_ms(started));
        return result;
    }

private:
    runtime::TaskSpec task_;
    std::shared_ptr<const MossFormer2Assets> assets_;
    std::shared_ptr<const model_spec::ModelContract> contract_;
    std::unique_ptr<MossFormer2GatedFSMNRuntime> runtime_;
};
}  // namespace

std::shared_ptr<runtime::IVoiceModelLoader> make_mossformer2_loader() {
    runtime::SpecBackedVoiceModelConfig<MossFormer2Assets> config;
    config.family = "mossformer2";
    config.load_assets = load_mossformer2_assets;
    config.create_session = [](const runtime::TaskSpec & task, const runtime::SessionOptions & options,
        std::shared_ptr<const MossFormer2Assets> assets, std::shared_ptr<const model_spec::ModelContract> contract) {
        return std::make_unique<Session>(task, options, std::move(assets), std::move(contract));
    };
    return runtime::make_spec_backed_voice_loader(std::move(config));
}
}  // namespace engine::models::mossformer2
