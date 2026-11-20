#include "engine/models/voxmel_tts/session.h"
#include "engine/models/voxmel_tts/runtime.h"

#include "engine/framework/audio/conversion.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/runtime/errors.h"
#include "engine/framework/runtime/options.h"
#include "engine/framework/runtime/session_base.h"
#include "engine/framework/runtime/spec_backed_model.h"
#include "engine/framework/text/chunking.h"
#include "engine/framework/tokenizers/llama_bpe.h"

#include <algorithm>
#include <cmath>

namespace engine::models::voxmel_tts {
namespace {

class Session final : public runtime::RuntimeSessionBase, public runtime::IOfflineVoiceTaskSession {
public:
    Session(runtime::TaskSpec task, const runtime::SessionOptions & options,
        std::shared_ptr<const ModelAssets> assets, std::shared_ptr<const model_spec::ModelContract> contract)
        : RuntimeSessionBase(options), task_(task), assets_(std::move(assets)), contract_(std::move(contract)),
          tokenizer_({{}, {}, assets_->resources.require_file("tokenizer_config"),
              assets_->resources.require_file("tokenizer_json"), tokenizers::LlamaBpePreTokenizer::Qwen2}) {
        runtime::validate_spec_backed_session_options(options, *contract_, "voxmel_tts", "VoxMel");
        const auto storage = assets::parse_tensor_storage_type(
            runtime::find_option(options.options, {"voxmel_tts.weight_type"}).value_or("native"));
        const auto attention = core::parse_attention_preference(
            runtime::find_option(options.options, {"voxmel_tts.attention"}).value_or("auto"), "voxmel_tts.attention");
        runtime_ = std::make_unique<VoxMelRuntime>(assets_, execution_context(), storage, attention);
    }

    std::string family() const override { return "voxmel_tts"; }
    runtime::VoiceTaskKind task_kind() const override { return task_.task; }
    runtime::RunMode run_mode() const override { return task_.mode; }
    void prepare(const runtime::SessionPreparationRequest &) override { mark_prepared(); }

    runtime::TaskResult run(const runtime::TaskRequest & request) override {
        require_prepared("VoxMel run");
        runtime::validate_spec_backed_request_options(request.options, request.option_arrays, *contract_, "VoxMel");
        const auto started = std::chrono::steady_clock::now();
        if (!request.text_input || request.text_input->text.find_first_not_of(" \t\r\n") == std::string::npos)
            throw runtime::InvalidRequestError("VoxMel requires nonempty text");
        const auto * reference = request.voice && request.voice->speaker && request.voice->speaker->audio
            ? &*request.voice->speaker->audio : nullptr;
        const auto reference_text = runtime::find_option(request.options, {"reference_text"}).value_or("");
        if ((reference == nullptr) != reference_text.empty())
            throw runtime::InvalidRequestError("VoxMel reference audio and its matching reference_text must be supplied together");
        if (task_.task == runtime::VoiceTaskKind::VoiceCloning && !reference)
            throw runtime::InvalidRequestError("VoxMel cloning requires reference audio and reference_text");
        SynthesisOptions settings;
        settings.steps = runtime::parse_i64_option(request.options, {"num_inference_steps"}).value_or(32);
        settings.up_steps = runtime::parse_i64_option(request.options, {"residual_num_inference_steps"}).value_or(settings.steps);
        settings.text_guidance = runtime::parse_finite_float_option(request.options, {"guidance_scale"}).value_or(3.5f);
        settings.speaker_guidance = runtime::parse_finite_float_option(request.options, {"speaker_guidance_scale"}).value_or(3.0f);
        settings.seed = runtime::parse_u64_option(request.options, {"seed"}).value_or(runtime::random_u64_seed());
        const float minimum = runtime::parse_finite_float_option(request.options, {"min_duration_sec"}).value_or(0.3f);
        const float maximum = runtime::parse_finite_float_option(request.options, {"max_duration_sec"}).value_or(30.0f);
        if (!(minimum >= 0 && maximum > minimum && maximum <= 300) ||
            settings.steps < 1 || settings.steps > 256 || settings.up_steps < 1 || settings.up_steps > 256)
            throw runtime::InvalidRequestError("VoxMel requires 0 <= min_duration_sec < max_duration_sec <= 300 and 1..256 sampling steps");
        const double fps = static_cast<double>(assets_->sample_rate) / static_cast<double>(assets_->hop) / 2;
        settings.min_frames = std::max(int64_t{1}, static_cast<int64_t>(std::ceil(minimum * fps)));
        settings.max_frames = std::max(int64_t{1}, static_cast<int64_t>(std::ceil(maximum * fps)));
        std::vector<float> mel;
        std::vector<int32_t> reference_ids;
        if (reference) {
            auto mono = audio::mixdown_interleaved_to_mono_average(reference->samples, reference->channels);
            mono = audio::resample_mono_torchaudio_sinc_hann(mono, reference->sample_rate, 24000,
                audio::torchaudio_sinc_hann_float32_options());
            if (mono.size() <= 512 || !std::all_of(mono.begin(), mono.end(), [](float x) { return std::isfinite(x); }))
                throw runtime::InvalidRequestError("VoxMel reference must be finite and longer than 512 samples at 24 kHz");
            mel = reference_mel(*assets_, mono, static_cast<size_t>(execution_context().config().threads));
            reference_ids = tokenizer_.encode(reference_text, true);
        }
        const auto chunks = runtime::chunk_text_request(request,
            text::parse_text_chunk_size_override(request.options).value_or(400),
            text::parse_text_chunk_mode_override(request.options).value_or(text::TextChunkMode::Default));
        runtime::AudioBuffer merged{24000, 1, {}};
        for (size_t i = 0; i < chunks.size(); ++i) {
            if (!chunks[i].text_input || chunks[i].text_input->text.empty()) continue;
            auto text = chunks[i].text_input->text;
            // The reference separates Latin prompt/target text by one space;
            // Mandarin targets join the reference transcript without this space.
            const auto first = text.find_first_not_of(" \t\r\n");
            if (reference && first != std::string::npos) {
                const bool mandarin = text.compare(first, 3, u8"\u4e00") >= 0 && text.compare(first, 3, u8"\u9fff") <= 0;
                if (!mandarin) text = " " + text;
            }
            auto per_chunk = settings;
            per_chunk.seed += i;
            auto wave = runtime_->synthesize(tokenizer_.encode(text, true), reference_ids, mel, per_chunk);
            if (wave.empty() || !std::all_of(wave.begin(), wave.end(), [](float x) { return std::isfinite(x); }))
                throw std::runtime_error("VoxMel produced empty or non-finite audio");
            merged.samples.insert(merged.samples.end(), wave.begin(), wave.end());
        }
        debug::timing_log_scalar("session.wall_ms", debug::elapsed_ms(started));
        runtime::TaskResult result;
        result.audio_output = std::move(merged);
        return result;
    }

private:
    runtime::TaskSpec task_;
    std::shared_ptr<const ModelAssets> assets_;
    std::shared_ptr<const model_spec::ModelContract> contract_;
    tokenizers::LlamaBpeTokenizer tokenizer_;
    std::unique_ptr<VoxMelRuntime> runtime_;
};

} // namespace

std::shared_ptr<runtime::IVoiceModelLoader> make_voxmel_tts_loader() {
    runtime::SpecBackedVoiceModelConfig<ModelAssets> c;
    c.family = "voxmel_tts";
    c.load_assets = load_assets;
    c.create_session = [](const runtime::TaskSpec & t, const runtime::SessionOptions & o,
        std::shared_ptr<const ModelAssets> a, std::shared_ptr<const model_spec::ModelContract> contract) {
        return std::make_unique<Session>(t, o, std::move(a), std::move(contract));
    };
    return runtime::make_spec_backed_voice_loader(std::move(c));
}

} // namespace engine::models::voxmel_tts
