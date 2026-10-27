#include "engine/models/magenta_realtime_2/session.h"

#include "engine/models/magenta_realtime_2/depthformer.h"
#include "engine/framework/conditioners/musiccoca_runtime.h"
#include "engine/models/magenta_realtime_2/spectrostream.h"
#include "engine/framework/audio/conversion.h"
#include "engine/framework/runtime/options.h"
#include "engine/framework/runtime/session_base.h"
#include "engine/framework/runtime/spec_backed_model.h"
#include "engine/framework/runtime/streaming_tts_session.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace engine::models::magenta_realtime_2 {
namespace {

constexpr const char * kFamily = "magenta_realtime_2";

class MagentaSession final : public runtime::RuntimeSessionBase,
                             public runtime::IOfflineVoiceTaskSession,
                             public runtime::StreamingTtsSessionBase {
public:
    MagentaSession(runtime::TaskSpec task, runtime::SessionOptions options,
                   std::shared_ptr<const MagentaAssets> assets,
                   std::shared_ptr<const model_spec::ModelContract> contract)
        : RuntimeSessionBase(options), task_(task), assets_(std::move(assets)), contract_(std::move(contract)) {
        runtime::validate_spec_backed_session_options(options, *contract_, kFamily, "Magenta RealTime 2");
        if (task.task != runtime::VoiceTaskKind::AudioGeneration) throw std::runtime_error("Magenta RealTime 2 supports the gen route");
        musiccoca_ = std::make_unique<conditioners::MusicCoCaRuntime>(execution_context(),
            assets_->resources.open_tensor_source("musiccoca"), assets_->resources.parse_json("musiccoca_config"),
            assets_->resources.require_file("musiccoca_tokenizer"));
        storage_ = engine::assets::parse_tensor_storage_type(runtime::find_option(options.options, {"magenta_realtime_2.weight_type"})
                                                               .value_or("native"));
    }

    std::string family() const override { return kFamily; }
    runtime::VoiceTaskKind task_kind() const override { return task_.task; }
    runtime::RunMode run_mode() const override { return task_.mode; }

    void prepare(const runtime::SessionPreparationRequest &) override {
        if (!decoder_) decoder_ = std::make_unique<DepthformerRuntime>(execution_context(), assets_, storage_);
        if (!codec_) codec_ = std::make_unique<SpectroStreamRuntime>(execution_context(), assets_);
        assets_->weights->release_storage();
        mark_prepared();
    }

    runtime::TaskResult run(const runtime::TaskRequest & request) override { return generate(request, false); }

private:
    runtime::TaskResult generate_stream(const runtime::TaskRequest & request) override { return generate(request, true); }
    void reset_stream_state() override {}

    runtime::TaskResult generate(const runtime::TaskRequest & request, bool streaming) {
        require_prepared("Magenta RealTime 2 generation");
        runtime::validate_spec_backed_request_options(request.options, request.option_arrays, *contract_, "Magenta RealTime 2");
        const auto start = std::chrono::steady_clock::now();
        const float duration = runtime::parse_positive_finite_float_option(request.options, {"duration_sec"}).value_or(10.0F);
        if (duration < 0.04F) throw std::runtime_error("Magenta duration_sec must be at least one 40 ms frame");
        const int64_t frames = static_cast<int64_t>(std::ceil(duration * 25.0F));
        MagentaSamplingOptions sampling;
        sampling.temperature = runtime::parse_finite_float_option(request.options, {"temperature"}).value_or(1.3F);
        sampling.top_k = runtime::parse_int_option(request.options, {"top_k"}).value_or(40);
        sampling.seed = runtime::parse_u32_option(request.options, {"seed"}).value_or(0);
        if (sampling.temperature < 0 || sampling.top_k < 0 || sampling.top_k > 1024)
            throw std::runtime_error("Magenta temperature must be nonnegative and top_k must be between 0 and 1024");
        const float scales[] = {
            runtime::parse_finite_float_option(request.options, {"guidance_scale"}).value_or(3.0F),
            runtime::parse_finite_float_option(request.options, {"note_guidance_scale"}).value_or(1.0F),
            runtime::parse_finite_float_option(request.options, {"drum_guidance_scale"}).value_or(1.0F)};
        for (auto scale : scales) if (scale < -1 || scale > 7) throw std::runtime_error("Magenta guidance scales must be between -1 and 7");
        std::vector<float> embedding;
        if (request.audio_input) {
            const auto & audio = *request.audio_input;
            embedding = musiccoca_->embed_audio(engine::audio::convert_interleaved_audio_to_mono_torchaudio_sinc_hann_resampled(
                audio.samples, audio.sample_rate, audio.channels, 16000));
        } else {
            if (!request.text_input || request.text_input->text.empty()) throw std::runtime_error("Magenta requires a text prompt or audio style input");
            embedding = musiccoca_->embed_text(request.text_input->text);
            const auto mapper = runtime::find_option(request.options, {"use_mapper"});
            if (!mapper || runtime::parse_bool_option(*mapper, "use_mapper")) embedding = musiccoca_->map_text(embedding);
        }
        auto style = musiccoca_->quantize(embedding);
        debug::trace_log_i32("magenta_realtime_2.style_codes", {12}, {style.begin(), style.end()});
        for (size_t i = 0; i < style.size(); ++i) style[i] += 7 + static_cast<int32_t>(i) * 1031;
        std::array<int32_t, 132> controls;
        for (int i = 0; i < 128; ++i) controls[static_cast<size_t>(i)] = i * 11 + 6;
        const int drums = runtime::parse_int_option(request.options, {"drums"}).value_or(-1);
        if (drums < -1 || drums > 1) throw std::runtime_error("Magenta drums must be -1, 0, or 1");
        controls[128] = 1408 + 7 + drums;
        controls[129] = 1417 + 7 + static_cast<int32_t>(std::nearbyint((static_cast<double>(scales[0]) + 1) / 0.2));
        controls[130] = 1464 + 7 + static_cast<int32_t>(std::nearbyint((static_cast<double>(scales[1]) + 1) / 0.2));
        controls[131] = 1511 + 7 + static_cast<int32_t>(std::nearbyint(static_cast<double>(scales[2]) + 1));
        std::vector<std::array<int32_t, 128>> piano;
        if (const auto path = runtime::find_option(request.options, {"pianoroll_file"})) {
            const auto data = io::json::parse_file(*path);
            if (data.as_array().size() != static_cast<size_t>(frames)) throw std::runtime_error("Magenta pianoroll must have one row per generated 25 Hz frame");
            for (const auto & row : data.as_array()) {
                if (row.as_array().size() != 128) throw std::runtime_error("Magenta pianoroll rows must have 128 pitch states");
                std::array<int32_t, 128> values;
                for (size_t pitch = 0; pitch < 128; ++pitch) {
                    const auto value = row.as_array()[pitch].as_i64();
                    if (value < -1 || value > 3) throw std::runtime_error("Magenta pitch states must be between -1 and 3");
                    values[pitch] = static_cast<int32_t>(value) + 7 + static_cast<int32_t>(pitch) * 11;
                }
                piano.push_back(values);
            }
        }
        decoder_->reset(sampling);
        codec_->reset();
        runtime::AudioBuffer audio{48000, 2, {}};
        audio.samples.reserve(static_cast<size_t>(frames) * 1920 * 2);
        std::vector<std::array<int32_t, 12>> pending;
        pending.reserve(4);
        bool first = true;
        double ar_ms = 0, codec_ms = 0;
        for (int64_t frame = 0; frame < frames; ++frame) {
            if (!piano.empty()) std::copy(piano[static_cast<size_t>(frame)].begin(), piano[static_cast<size_t>(frame)].end(), controls.begin());
            auto phase = std::chrono::steady_clock::now();
            decoder_->condition(style, controls);
            pending.push_back(decoder_->step());
            ar_ms += debug::elapsed_ms(phase);
            if (pending.size() != 4 && frame + 1 != frames) continue;
            phase = std::chrono::steady_clock::now();
            const size_t valid_samples = pending.size() * 1920 * 2;
            // Pad the final causal block to reuse the four-frame graph.
            pending.resize(4);
            auto decoded = codec_->decode(pending);
            decoded.resize(valid_samples);
            for (auto & value : decoded) {
                const float limited = std::clamp(value * 0.5F, -1.0F, 1.0F);
                value = std::nearbyint(32767.5F * limited - 0.5F) / 32768.0F;
            }
            codec_ms += debug::elapsed_ms(phase);
            if (first) {
                debug::timing_log_scalar("session.ttfa_ms", debug::elapsed_ms(start));
                first = false;
            }
            if (streaming && stream_event_sink()) {
                runtime::StreamEvent event;
                event.audio_output = runtime::AudioBuffer{48000, 2, decoded};
                event.is_final = frame + 1 == frames;
                stream_event_sink()(event);
            }
            audio.samples.insert(audio.samples.end(), decoded.begin(), decoded.end());
            pending.clear();
        }
        debug::timing_log_scalar("magenta_realtime_2.ar.total_ms", ar_ms);
        debug::timing_log_scalar("magenta_realtime_2.codec.total_ms", codec_ms);
        debug::timing_log_scalar("audio.duration_sec", static_cast<double>(audio.samples.size()) / 96000);
        debug::timing_log_scalar("session.wall_ms", debug::elapsed_ms(start));
        runtime::TaskResult result;
        result.audio_output = std::move(audio);
        return result;
    }

    runtime::TaskSpec task_;
    std::shared_ptr<const MagentaAssets> assets_;
    std::shared_ptr<const model_spec::ModelContract> contract_;
    engine::assets::TensorStorageType storage_ = engine::assets::TensorStorageType::Native;
    std::unique_ptr<conditioners::MusicCoCaRuntime> musiccoca_;
    std::unique_ptr<DepthformerRuntime> decoder_;
    std::unique_ptr<SpectroStreamRuntime> codec_;
};

}  // namespace

std::shared_ptr<runtime::IVoiceModelLoader> make_magenta_realtime_2_loader() {
    runtime::SpecBackedVoiceModelConfig<MagentaAssets> config;
    config.family = kFamily;
    config.load_assets = load_magenta_realtime_2_assets;
    config.create_session = [](const runtime::TaskSpec & task, const runtime::SessionOptions & options,
                               std::shared_ptr<const MagentaAssets> assets,
                               std::shared_ptr<const model_spec::ModelContract> contract) {
        return std::make_unique<MagentaSession>(task, options, std::move(assets), std::move(contract));
    };
    return runtime::make_spec_backed_voice_loader(std::move(config));
}

}  // namespace engine::models::magenta_realtime_2
