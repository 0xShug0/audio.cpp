#include "engine/models/voxtral_tts/session.h"

#include "engine/models/voxtral_tts/acoustic.h"
#include "engine/models/voxtral_tts/ar.h"
#include "engine/models/voxtral_tts/codec.h"
#include "engine/framework/runtime/options.h"
#include "engine/framework/runtime/session_base.h"
#include "engine/framework/runtime/spec_backed_model.h"
#include "engine/framework/runtime/streaming_tts_session.h"
#include "engine/framework/sampling/torch_random.h"
#include "engine/framework/text/chunking.h"
#include "engine/framework/tokenizers/llama_bpe.h"

#include <algorithm>
#include <chrono>
#include <limits>
#include <numeric>

namespace engine::models::voxtral_tts {
namespace {

class VoxtralTtsSession final : public runtime::RuntimeSessionBase,
                                public runtime::IOfflineVoiceTaskSession,
                                public runtime::StreamingTtsSessionBase {
public:
    VoxtralTtsSession(runtime::TaskSpec task, const runtime::SessionOptions & options,
                     std::shared_ptr<const VoxtralTtsAssets> assets,
                     std::shared_ptr<const model_spec::ModelContract> contract)
        : RuntimeSessionBase(options), task_(task), assets_(std::move(assets)), contract_(std::move(contract)),
          tokenizer_({{}, {}, assets_->resources.require_file("tokenizer_config"),
                      assets_->resources.require_file("tokenizer_json"), tokenizers::LlamaBpePreTokenizer::Tekken}) {
        runtime::validate_spec_backed_session_options(options, *contract_, "voxtral_tts", "Voxtral TTS");
        if (task.task != runtime::VoiceTaskKind::Tts) {
            throw std::runtime_error("Voxtral TTS implements preset-voice TTS, not reference-audio cloning");
        }
        const auto storage = assets::parse_tensor_storage_type(
            runtime::find_option(options.options, {"voxtral_tts.weight_type"}).value_or("native"));
        const auto cache = assets::parse_tensor_storage_type(
            runtime::find_option(options.options, {"voxtral_tts.kv_cache_type"}).value_or("f16"));
        if (cache != assets::TensorStorageType::F32 && cache != assets::TensorStorageType::F16 &&
            cache != assets::TensorStorageType::BF16) {
            throw std::runtime_error("Voxtral TTS kv_cache_type must be f32, f16, or bf16");
        }
        ar_ = std::make_unique<VoxtralMinistralARRuntime>(*assets_, execution_context(), storage,
                                                         assets::ggml_type_for_tensor_storage(cache));
        acoustic_ = std::make_unique<VoxtralAcousticFlowRuntime>(*assets_, execution_context(), storage);
        codec_ = std::make_unique<VoxtralCausalCodecDecoderRuntime>(*assets_, execution_context(), storage);
        const auto acoustic_storage = assets::resolve_tensor_storage_type(
            *assets_->weights, "acoustic_transformer.input_projection.weight", storage);
        noise_precision_ = acoustic_storage == assets::TensorStorageType::F32
            ? sampling::TorchRandnPrecision::Float32 : sampling::TorchRandnPrecision::BFloat16;
        random_policy_ = sampling::resolve_torch_cuda_sampling_policy(
            execution_context().backend_type(), options.backend.device, "voxtral_tts", "Voxtral TTS");
        hop_ = assets_->config.patch_size * std::accumulate(assets_->config.codec_strides.begin(),
            assets_->config.codec_strides.end(), int64_t{1}, std::multiplies<int64_t>{});
        size_t voice_bytes = 0;
        for (const auto & [name, voice] : assets_->voices) {
            (void)name;
            voice_bytes += voice.size() * sizeof(float);
        }
        debug::timing_log_scalar("voxtral_tts.voices.host_mb", static_cast<double>(voice_bytes) / (1024 * 1024));
    }

    std::string family() const override { return "voxtral_tts"; }
    runtime::VoiceTaskKind task_kind() const override { return runtime::VoiceTaskKind::Tts; }
    runtime::RunMode run_mode() const override { return task_.mode; }

    void prepare(const runtime::SessionPreparationRequest & request) override {
        runtime::validate_spec_backed_request_options(request.options, request.option_arrays, *contract_, "Voxtral TTS");
        mark_prepared();
    }

    runtime::TaskResult run(const runtime::TaskRequest & request) override {
        if (task_.mode != runtime::RunMode::Offline) {
            throw std::runtime_error("Voxtral TTS run requires an offline session");
        }
        return synthesize(request, false);
    }

private:
    runtime::TaskResult generate_stream(const runtime::TaskRequest & request) override {
        return synthesize(request, true);
    }

    void reset_stream_state() override {}

    runtime::TaskResult synthesize(const runtime::TaskRequest & request, bool streaming) {
        require_prepared("Voxtral TTS synthesis");
        const auto begin = std::chrono::steady_clock::now();
        runtime::validate_spec_backed_request_options(request.options, request.option_arrays, *contract_, "Voxtral TTS");
        if (!request.text_input || request.text_input->text.empty()) {
            throw std::runtime_error("Voxtral TTS requires non-empty text");
        }
        auto voice_id = runtime::find_option(request.options, {"voice_id"}).value_or("casual_male");
        if (request.voice && request.voice->speaker) {
            if (request.voice->speaker->audio) {
                throw std::runtime_error("Voxtral TTS's released checkpoint does not contain voice-cloning encoder weights");
            }
            if (request.voice->speaker->cached_voice_id) {
                voice_id = *request.voice->speaker->cached_voice_id;
            }
        }
        const auto voice = assets_->voices.find(voice_id);
        if (voice == assets_->voices.end()) {
            throw std::runtime_error("unknown Voxtral TTS voice_id: " + voice_id);
        }
        const auto max_frames = runtime::parse_i64_option(request.options, {"max_tokens"}).value_or(2048);
        const auto steps = runtime::parse_int_option(request.options, {"num_inference_steps"}).value_or(7);
        const auto guidance = runtime::parse_finite_float_option(request.options, {"guidance_scale"}).value_or(1.2F);
        if (max_frames <= 0 || steps <= 0 || guidance < 0) {
            throw std::runtime_error("Voxtral TTS token limit, inference steps, or guidance is invalid");
        }
        auto seed = runtime::parse_u64_option(request.options, {"seed"}).value_or(runtime::random_u64_seed());
        const auto chunk_size = text::parse_text_chunk_size_override(request.options).value_or(300);
        const auto chunk_mode = text::parse_text_chunk_mode_override(request.options).value_or(text::TextChunkMode::Default);
        const auto chunks = runtime::chunk_text_request(request, chunk_size, chunk_mode);
        debug::trace_log_scalar("voxtral_tts.voice_id", voice_id);
        debug::trace_log_scalar("voxtral_tts.seed", seed);
        debug::trace_log_scalar("voxtral_tts.text_chunk_count", static_cast<int64_t>(chunks.size()));
        runtime::AudioBuffer audio{static_cast<int>(assets_->config.sample_rate), 1, {}};
        double ar_ms = 0, acoustic_ms = 0, codec_ms = 0;
        for (const auto & chunk : chunks) {
            const auto & config = assets_->config;
            std::vector<int32_t> prompt{1, 25};
            prompt.insert(prompt.end(), voice->second.size() / config.ar.hidden_size, 24);
            prompt.push_back(36);
            const auto text_tokens = tokenizer_.encode(chunk.text_input->text, false);
            prompt.insert(prompt.end(), text_tokens.begin(), text_tokens.end());
            prompt.insert(prompt.end(), {35, 25});
            auto output = ar_->prefill(prompt, voice->second, max_frames);
            std::vector<int32_t> codes;
            int64_t emitted_frames = 0;
            uint64_t noise_offset = 0;
            const auto flush = [&] {
                const auto frames = static_cast<int64_t>(codes.size()) / (config.acoustic_dim + 1);
                const auto count = frames - emitted_frames;
                if (count == 0) {
                    return;
                }
                const auto context = std::min(emitted_frames, kCodecContextFrames);
                const auto offset = (emitted_frames - context) * (config.acoustic_dim + 1);
                const auto codec_start = std::chrono::steady_clock::now();
                auto waveform = codec_->decode(std::vector<int32_t>(codes.begin() + offset, codes.end()), emitted_frames == 0);
                codec_ms += debug::elapsed_ms(codec_start);
                waveform.erase(waveform.begin(), waveform.begin() + context * hop_);
                audio.samples.insert(audio.samples.end(), waveform.begin(), waveform.end());
                if (streaming && stream_event_sink()) {
                    runtime::StreamEvent event;
                    event.audio_output = runtime::AudioBuffer{audio.sample_rate, 1, std::move(waveform)};
                    stream_event_sink()(event);
                }
                emitted_frames = frames;
            };
            for (int64_t frame = 0; frame < max_frames; ++frame) {
                output.logits[0] = -std::numeric_limits<float>::infinity();
                std::fill(output.logits.begin() + config.semantic_vocab_size + kAudioTokenOffset,
                          output.logits.end(), -std::numeric_limits<float>::infinity());
                const auto semantic = static_cast<int32_t>(std::distance(output.logits.begin(),
                    std::max_element(output.logits.begin(), output.logits.end())));
                if (semantic == kAudioEndToken) {
                    break;
                }
                const auto acoustic_start = std::chrono::steady_clock::now();
                auto noise = sampling::generate_torch_cuda_tensor_iterator_randn(
                    config.acoustic_dim, seed, noise_offset, random_policy_, noise_precision_);
                noise_offset += sampling::torch_cuda_tensor_iterator_offset_blocks(config.acoustic_dim, random_policy_);
                auto frame_codes = acoustic_->sample(output.hidden, noise, steps, guidance);
                frame_codes.insert(frame_codes.begin(), semantic);
                acoustic_ms += debug::elapsed_ms(acoustic_start);
                codes.insert(codes.end(), frame_codes.begin(), frame_codes.end());
                const auto target = emitted_frames == 0 ? kCodecInitialFrames : kCodecChunkFrames;
                if (frame + 1 - emitted_frames == target) {
                    flush();
                }
                if (frame + 1 < max_frames) {
                    const auto ar_start = std::chrono::steady_clock::now();
                    output = ar_->decode(frame_codes);
                    ar_ms += debug::elapsed_ms(ar_start);
                }
            }
            flush();
            ++seed;
        }
        if (audio.samples.empty()) {
            throw std::runtime_error("Voxtral TTS ended without generating an audio frame");
        }
        debug::timing_log_scalar("voxtral_tts.ar.decode_ms", ar_ms);
        debug::timing_log_scalar("voxtral_tts.acoustic.sample_ms", acoustic_ms);
        debug::timing_log_scalar("voxtral_tts.codec.decode_ms", codec_ms);
        debug::timing_log_scalar("session.wall_ms", debug::elapsed_ms(begin));
        runtime::TaskResult result;
        result.audio_output = std::move(audio);
        return result;
    }

    runtime::TaskSpec task_;
    std::shared_ptr<const VoxtralTtsAssets> assets_;
    std::shared_ptr<const model_spec::ModelContract> contract_;
    tokenizers::LlamaBpeTokenizer tokenizer_;
    std::unique_ptr<VoxtralMinistralARRuntime> ar_;
    std::unique_ptr<VoxtralAcousticFlowRuntime> acoustic_;
    std::unique_ptr<VoxtralCausalCodecDecoderRuntime> codec_;
    sampling::TorchCudaSamplingPolicy random_policy_;
    sampling::TorchRandnPrecision noise_precision_;
    int64_t hop_ = 0;
};

}  // namespace

std::shared_ptr<runtime::IVoiceModelLoader> make_voxtral_tts_loader() {
    runtime::SpecBackedVoiceModelConfig<VoxtralTtsAssets> config;
    config.family = "voxtral_tts";
    config.load_assets = load_assets;
    config.create_session = [](const auto & task, const auto & options, auto assets, auto contract) {
        return std::make_unique<VoxtralTtsSession>(task, options, std::move(assets), std::move(contract));
    };
    return runtime::make_spec_backed_voice_loader(std::move(config));
}

}  // namespace engine::models::voxtral_tts
