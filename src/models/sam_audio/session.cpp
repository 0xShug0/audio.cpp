#include "engine/models/sam_audio/session.h"
#include "engine/models/sam_audio/codec.h"
#include "engine/models/sam_audio/conditioner.h"
#include "engine/models/sam_audio/dit.h"
#include "engine/models/sam_audio/frontend.h"
#include "engine/models/sam_audio/vision.h"

#include "engine/framework/audio/conversion.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/runtime/options.h"
#include "engine/framework/runtime/session_base.h"
#include "engine/framework/runtime/spec_backed_model.h"
#include "engine/framework/sampling/noise.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>

namespace engine::models::sam_audio {
namespace {

struct SAMAudioAssets {
    assets::ResourceBundle resources;
    std::shared_ptr<const assets::TensorSource> tensors;
    DacVAEConfig codec;
    int sample_rate = 48000;
    int64_t video_channels = 1024;
};

class SAMAudioSession final : public runtime::RuntimeSessionBase, public runtime::IOfflineVoiceTaskSession {
public:
    SAMAudioSession(runtime::TaskSpec task, const runtime::SessionOptions & options,
                    std::shared_ptr<const SAMAudioAssets> assets,
                    std::shared_ptr<const model_spec::ModelContract> contract)
        : RuntimeSessionBase(options), task_(task), assets_(std::move(assets)), contract_(std::move(contract)) {
        runtime::validate_spec_backed_session_options(options, *contract_, "sam_audio", "SAM Audio");
        if (task.task != runtime::VoiceTaskKind::SpeechToSpeech || task.mode != runtime::RunMode::Offline)
            throw std::runtime_error("SAM Audio supports offline s2s separation");
        const auto bounded_option = runtime::find_option(options.options, {"sam_audio.memory_bounded"});
        const bool memory_bounded = bounded_option && runtime::parse_bool_option(*bounded_option, "sam_audio.memory_bounded");
        auto & execution = execution_context();
        encoder_ = std::make_unique<DacVAEEncoder>(assets_->tensors, execution, assets_->codec, memory_bounded);
        text_ = std::make_unique<T5TextEncoder>(assets_->tensors, execution,
            assets_->resources.require_file("t5_config"), assets_->resources.require_file("tokenizer"));
        denoiser_ = std::make_unique<DiTRuntime>(assets_->tensors, execution, assets_->resources.require_file("config"), memory_bounded);
        decoder_ = std::make_unique<DacVAEDecoder>(assets_->tensors, execution, assets_->codec, memory_bounded);
        assets_->tensors->release_storage();
    }

    std::string family() const override { return "sam_audio"; }
    runtime::VoiceTaskKind task_kind() const override { return task_.task; }
    runtime::RunMode run_mode() const override { return task_.mode; }

    void prepare(const runtime::SessionPreparationRequest & request) override {
        if (!request.audio || request.audio->sample_rate <= 0 || request.audio->channels <= 0)
            throw std::runtime_error("SAM Audio requires input audio");
        mark_prepared();
    }

    runtime::TaskResult run(const runtime::TaskRequest & request) override {
        require_prepared("SAM Audio run");
        runtime::validate_spec_backed_request_options(request.options, request.option_arrays, *contract_, "SAM Audio");
        if (!request.audio_input || !request.text_input)
            throw std::runtime_error("SAM Audio requires audio and a text description");
        const int steps = runtime::parse_int_option(request.options, {"num_inference_steps"}).value_or(16);
        const int64_t requested_seed = runtime::parse_i64_option(request.options, {"seed"}).value_or(42);
        const auto image_path = runtime::find_option(request.options, {"reference_image_path"});
        const auto video_path = runtime::find_option(request.options, {"reference_video_path"});
        if (image_path && video_path)
            throw std::runtime_error("SAM Audio accepts either reference_image_path or reference_video_path, not both");
        if (steps <= 0 || requested_seed < -1 || requested_seed > UINT32_MAX)
            throw std::runtime_error("SAM Audio requires positive steps and a seed from -1 through 4294967295");
        const uint32_t seed = requested_seed == -1 ? runtime::random_u32_seed() : static_cast<uint32_t>(requested_seed);
        const auto start = std::chrono::steady_clock::now();
        const auto & input = *request.audio_input;
        const auto mono = audio::convert_interleaved_audio_to_mono_torchaudio_sinc_hann_resampled(
            input.samples, input.sample_rate, input.channels, assets_->sample_rate);
        if (mono.empty()) throw std::runtime_error("SAM Audio input audio is empty");
        const auto encoded = encoder_->encode(mono);
        const auto text = text_->encode(request.text_input->text);
        DiTConditioning conditioning;
        const int64_t channels = assets_->codec.codebook_dim * 2;
        conditioning.frames = encoded.size() / assets_->codec.codebook_dim;
        conditioning.tokens = text.tokens.size();
        conditioning.text = text.features;
        conditioning.audio.resize(conditioning.frames * channels);
        for (int64_t t = 0; t < conditioning.frames; ++t)
            for (int64_t c = 0; c < channels; ++c)
                conditioning.audio[t * channels + c] = encoded[(c % assets_->codec.codebook_dim) * conditioning.frames + t];
        conditioning.video.assign(conditioning.frames * assets_->video_channels, 0.0f);
        if (image_path || video_path) {
            if (!vision_) {
                auto tensors = assets_->resources.open_tensor_source("weights");
                vision_ = std::make_unique<PECoreVisionEncoder>(tensors, execution_context());
                tensors->release_storage();
            }
            if (image_path) {
                const auto features = vision_->encode(load_reference_image(*image_path), 1);
                for (int64_t c = 0; c < assets_->video_channels; ++c)
                    std::fill_n(conditioning.video.begin() + c * conditioning.frames,
                                conditioning.frames, features.at(c));
            } else {
                const auto video = load_video_frames(*video_path);
                const int64_t hop = std::accumulate(assets_->codec.encoder_rates.begin(), assets_->codec.encoder_rates.end(),
                                                    int64_t{1}, std::multiplies<>());
                conditioning.video = vision_->encode_video(video, conditioning.frames, hop, assets_->sample_rate);
            }
        }
        conditioning.anchors.assign(conditioning.frames, 0);
        if (const auto anchors = runtime::find_option(request.options, {"anchors"})) {
            const auto spans = io::json::parse(*anchors);
            const int64_t hop = std::accumulate(assets_->codec.encoder_rates.begin(), assets_->codec.encoder_rates.end(),
                                               int64_t{1}, std::multiplies<>());
            for (const auto & span : spans.as_array()) {
                const auto & values = span.as_array();
                if (values.size() != 3) throw std::runtime_error("SAM Audio anchors must be [\"+\" or \"-\", start_seconds, end_seconds]");
                const auto & kind = values[0].as_string();
                const double start = values[1].as_number(), end = values[2].as_number();
                if ((kind != "+" && kind != "-") || !std::isfinite(start) || !std::isfinite(end) || start < 0 || end <= start)
                    throw std::runtime_error("SAM Audio anchor requires a +/- label and finite 0 <= start < end");
                const auto begin = static_cast<int64_t>(std::min<double>(conditioning.frames, std::ceil(start * assets_->sample_rate / hop)));
                const auto stop = static_cast<int64_t>(std::min<double>(conditioning.frames, std::ceil(end * assets_->sample_rate / hop)));
                std::fill(conditioning.anchors.begin() + begin, conditioning.anchors.begin() + stop, kind == "+" ? 1 : 2);
            }
        }
        const auto noise = sampling::generate_normal_noise(conditioning.audio.size(), seed);
        const auto generated = denoiser_->sample(conditioning, noise, steps);
        std::vector<float> latents(generated.size());
        for (int64_t t = 0; t < conditioning.frames; ++t)
            for (int64_t c = 0; c < channels; ++c)
                latents[c * conditioning.frames + t] = generated[t * channels + c];
        std::mt19937 rng(seed);
        std::vector<int32_t> message(32);
        for (auto & bit : message) bit = static_cast<int32_t>(rng() & 1U);
        const auto waveform = decoder_->decode(latents, 2, conditioning.frames, message);
        const size_t samples = waveform.size() / 2;
        runtime::TaskResult result;
        for (size_t i = 0; i < 2; ++i) {
            runtime::NamedAudioBuffer output;
            output.id = i == 0 ? "target" : "residual";
            output.audio = {assets_->sample_rate, 1,
                std::vector<float>(waveform.begin() + i * samples, waveform.begin() + (i + 1) * samples)};
            result.named_audio_outputs.push_back(std::move(output));
        }
        debug::timing_log_scalar("session.wall_ms", debug::elapsed_ms(start));
        return result;
    }

private:
    runtime::TaskSpec task_;
    std::shared_ptr<const SAMAudioAssets> assets_;
    std::shared_ptr<const model_spec::ModelContract> contract_;
    std::unique_ptr<DacVAEEncoder> encoder_;
    std::unique_ptr<T5TextEncoder> text_;
    std::unique_ptr<DiTRuntime> denoiser_;
    std::unique_ptr<DacVAEDecoder> decoder_;
    std::unique_ptr<PECoreVisionEncoder> vision_;
};

}  // namespace

std::shared_ptr<runtime::IVoiceModelLoader> make_sam_audio_loader() {
    runtime::SpecBackedVoiceModelConfig<SAMAudioAssets> config;
    config.family = "sam_audio";
    config.load_assets = [](const std::filesystem::path & path) {
        auto assets = std::make_shared<SAMAudioAssets>();
        assets->resources = model_spec::load_resource_bundle_for_family(path, "sam_audio");
        assets->tensors = assets->resources.open_tensor_source("weights");
        const auto json = assets->resources.parse_json("config");
        const auto & codec = json.require("audio_codec");
        assets->codec.encoder_dim = io::json::require_i64(codec, "encoder_dim");
        assets->codec.latent_dim = io::json::require_i64(codec, "latent_dim");
        assets->codec.codebook_dim = io::json::require_i64(codec, "codebook_dim");
        assets->codec.encoder_rates = io::json::number_array_as<int>(codec.require("encoder_rates"));
        assets->codec.decoder_rates = io::json::number_array_as<int>(codec.require("decoder_rates"));
        assets->sample_rate = io::json::require_i32(codec, "sample_rate");
        assets->video_channels = io::json::require_i64(json.require("vision_encoder"), "dim");
        return assets;
    };
    config.create_session = [](const runtime::TaskSpec & task, const runtime::SessionOptions & options,
                               std::shared_ptr<const SAMAudioAssets> assets,
                               std::shared_ptr<const model_spec::ModelContract> contract) {
        return std::make_unique<SAMAudioSession>(task, options, std::move(assets), std::move(contract));
    };
    return runtime::make_spec_backed_voice_loader(std::move(config));
}

}  // namespace engine::models::sam_audio
