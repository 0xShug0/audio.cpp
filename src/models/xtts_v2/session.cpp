#include "engine/models/xtts_v2/session.h"

#include "engine/framework/runtime/spec_backed_model.h"
#include "engine/models/xtts_v2/audio_features.h"
#include "engine/models/xtts_v2/request.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace engine::models::xtts_v2 {
namespace {
constexpr const char * kFamily = "xtts_v2";

std::shared_ptr<const XttsV2Assets> require_assets(std::shared_ptr<const XttsV2Assets> assets) {
    if (!assets) throw std::runtime_error("XTTS v2 session requires assets");
    return assets;
}

std::shared_ptr<const engine::model_spec::ModelContract> require_contract(
    std::shared_ptr<const engine::model_spec::ModelContract> contract) {
    if (!contract) throw std::runtime_error("XTTS v2 session requires a model contract");
    return contract;
}

std::unique_ptr<runtime::IVoiceTaskSession> create_session(
    const runtime::TaskSpec & task, const runtime::SessionOptions & options,
    std::shared_ptr<const XttsV2Assets> assets,
    std::shared_ptr<const engine::model_spec::ModelContract> contract) {
    if (task.mode != runtime::RunMode::Offline ||
        (task.task != runtime::VoiceTaskKind::Tts && task.task != runtime::VoiceTaskKind::VoiceCloning))
        throw std::runtime_error("XTTS v2 supports offline TTS and voice cloning");
    return std::make_unique<XttsV2Session>(task, options, std::move(assets), std::move(contract));
}

XttsV2ConditioningLatent conditioning_for_reference(
    XttsV2ConditioningRuntime & runtime, const std::vector<float> & waveform) {
    constexpr size_t chunk = 4U * 22050U;
    constexpr size_t minimum = 22050U / 3U;
    XttsV2ConditioningLatent average; size_t used = 0;
    for (size_t offset = 0; offset < waveform.size(); offset += chunk) {
        const size_t end = std::min(waveform.size(), offset + chunk);
        if (end - offset < minimum) continue;
        const std::vector<float> samples(waveform.begin() + static_cast<std::ptrdiff_t>(offset),
                                         waveform.begin() + static_cast<std::ptrdiff_t>(end));
        auto latent = runtime.encode(compute_xtts_v2_conditioning_mel(samples, runtime.mel_stats()));
        if (average.values.empty()) { average = latent; std::fill(average.values.begin(), average.values.end(), 0.0F); }
        for (size_t i = 0; i < latent.values.size(); ++i) average.values[i] += latent.values[i];
        ++used;
    }
    if (used == 0) throw std::runtime_error("XTTS v2 reference audio must contain at least 0.33 seconds");
    for (float & value : average.values) value /= static_cast<float>(used);
    return average;
}

std::vector<float> scale_speed(const std::vector<float> & input, int64_t frames, float speed, int64_t & output_frames) {
    if (speed == 1.0F) { output_frames = frames; return input; }
    output_frames = std::max<int64_t>(1, static_cast<int64_t>(std::floor(static_cast<double>(frames) / speed)));
    std::vector<float> output(static_cast<size_t>(output_frames * 1024));
    const double scale = 1.0 / static_cast<double>(speed);
    for (int64_t t = 0; t < output_frames; ++t) {
        const double source = (static_cast<double>(t) + 0.5) / scale - 0.5;
        const int64_t left = std::max<int64_t>(0, std::min<int64_t>(frames - 1, static_cast<int64_t>(std::floor(source))));
        const int64_t right = std::min<int64_t>(frames - 1, left + 1);
        const float fraction = static_cast<float>(std::max(0.0, std::min(1.0, source - left)));
        for (int64_t c = 0; c < 1024; ++c) {
            const float a = input[static_cast<size_t>(left * 1024 + c)];
            const float b = input[static_cast<size_t>(right * 1024 + c)];
            output[static_cast<size_t>(t * 1024 + c)] = a + (b - a) * fraction;
        }
    }
    return output;
}
}  // namespace

XttsV2Session::XttsV2Session(runtime::TaskSpec task, runtime::SessionOptions options,
    std::shared_ptr<const XttsV2Assets> assets,
    std::shared_ptr<const engine::model_spec::ModelContract> contract)
    : RuntimeSessionBase(options), task_(task), assets_(require_assets(std::move(assets))),
      contract_(require_contract(std::move(contract))), tokenizer_(assets_->tokenizer_path) {
    runtime::validate_spec_backed_session_options(options, *contract_, kFamily, "XTTS v2");
    auto & execution = execution_context();
    conditioning_ = std::make_unique<XttsV2ConditioningRuntime>(*assets_, execution,
        512U * 1024U * 1024U, 512U * 1024U * 1024U,
        assets::TensorStorageType::Native, assets::TensorStorageType::Native);
    speaker_ = std::make_unique<XttsV2SpeakerEncoderRuntime>(*assets_, execution,
        128U * 1024U * 1024U, 512U * 1024U * 1024U,
        assets::TensorStorageType::Native, assets::TensorStorageType::Native);
    gpt_ = std::make_unique<XttsV2GptRuntime>(*assets_, execution,
        1536U * 1024U * 1024U, 1536U * 1024U * 1024U, assets::TensorStorageType::Native);
    decoder_ = std::make_unique<XttsV2DecoderRuntime>(*assets_, execution,
        128U * 1024U * 1024U, 1024U * 1024U * 1024U, assets::TensorStorageType::Native);
}
XttsV2Session::~XttsV2Session() = default;
std::string XttsV2Session::family() const { return kFamily; }
runtime::VoiceTaskKind XttsV2Session::task_kind() const { return task_.task; }
runtime::RunMode XttsV2Session::run_mode() const { return task_.mode; }
void XttsV2Session::prepare(const runtime::SessionPreparationRequest & request) {
    runtime::validate_spec_backed_request_options(request.options, *contract_, "XTTS v2"); mark_prepared();
}
runtime::TaskResult XttsV2Session::run(const runtime::TaskRequest & request) {
    require_prepared("XTTS v2 run"); runtime::validate_spec_backed_request_options(request.options, *contract_, "XTTS v2");
    const auto parsed = parse_xtts_v2_request(request);
    const auto reference = prepare_xtts_v2_reference(parsed.speaker_audio);
    const auto condition = conditioning_for_reference(*conditioning_, reference.waveform_22050);
    const auto speaker_mel = compute_xtts_v2_speaker_mel(reference.waveform_16000,
        assets_->speaker_encoder->require_f32("torch_spec.1.spectrogram.window", {400}),
        assets_->speaker_encoder->require_f32("torch_spec.1.mel_scale.fb", {257, 64}));
    const auto speaker_embedding = speaker_->encode(speaker_mel);
    const auto text = tokenizer_.encode(parsed.text, parsed.language);
    const auto generated = gpt_->generate(condition.values, text, parsed.generation);
    if (generated.codes.empty()) throw std::runtime_error("XTTS v2 generated no acoustic tokens");
    int64_t decoder_frames = static_cast<int64_t>(generated.codes.size());
    auto latents = scale_speed(generated.latents, decoder_frames, parsed.generation.speed, decoder_frames);
    runtime::TaskResult result; runtime::AudioBuffer audio; audio.sample_rate = 24000; audio.channels = 1;
    audio.samples = decoder_->decode(latents, decoder_frames, speaker_embedding); result.audio_output = std::move(audio); return result;
}
std::shared_ptr<runtime::IVoiceModelLoader> make_xtts_v2_loader() {
    runtime::SpecBackedVoiceModelConfig<XttsV2Assets> config;
    config.family = kFamily; config.load_assets = load_xtts_v2_assets; config.create_session = create_session;
    return runtime::make_spec_backed_voice_loader(std::move(config));
}

}  // namespace engine::models::xtts_v2
