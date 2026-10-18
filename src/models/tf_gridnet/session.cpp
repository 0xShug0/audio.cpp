#include "engine/models/tf_gridnet/session.h"

#include "engine/framework/audio/chunking.h"
#include "engine/framework/audio/conversion.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/runtime/options.h"
#include "engine/framework/runtime/spec_backed_model.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace engine::models::tf_gridnet {
namespace {

// Match speaker lanes over the overlap using ESPnet's zero-mean SI-SNR objective.
std::vector<size_t> match_speakers(const std::vector<std::vector<float>> & previous,
    const std::vector<std::vector<float>> & current, int64_t overlap) {
    const size_t count = previous.size();
    std::vector<std::vector<double>> scores(count, std::vector<double>(count));
    for (size_t i = 0; i < count; ++i) {
        const auto a = previous[i].end() - overlap;
        const double mean_a = std::accumulate(a, previous[i].end(), 0.0) / overlap;
        for (size_t j = 0; j < count; ++j) {
            const double mean_b = std::accumulate(current[j].begin(), current[j].begin() + overlap, 0.0) / overlap;
            double dot = 0, aa = 0, bb = 0;
            for (int64_t t = 0; t < overlap; ++t) {
                const double x = a[t] - mean_a;
                const double y = current[j][t] - mean_b;
                dot += x * y;
                aa += x * x;
                bb += y * y;
            }
            constexpr double epsilon = std::numeric_limits<float>::epsilon();
            const double correlation_squared = std::clamp(dot * dot /
                (std::max(aa, epsilon * epsilon) * std::max(bb, epsilon * epsilon)), epsilon, 1.0 - epsilon);
            scores[i][j] = 10 * std::log10(correlation_squared / (1.0 - correlation_squared));
        }
    }
    std::vector<size_t> permutation(count);
    std::iota(permutation.begin(), permutation.end(), 0);
    auto best = permutation;
    double best_score = -std::numeric_limits<double>::infinity();
    do {
        double score = 0;
        for (size_t i = 0; i < count; ++i) score += scores[i][permutation[i]];
        if (score > best_score) {
            best_score = score;
            best = permutation;
        }
    } while (std::next_permutation(permutation.begin(), permutation.end()));
    return best;
}

}  // namespace

TFGridNetSession::TFGridNetSession(runtime::TaskSpec task, runtime::SessionOptions options,
    std::shared_ptr<const TFGridNetAssets> assets, std::shared_ptr<const model_spec::ModelContract> contract)
    : RuntimeSessionBase(options), task_(task), assets_(std::move(assets)), contract_(std::move(contract)) {
    runtime::validate_spec_backed_session_options(options, *contract_, "tf_gridnet", "TF-GridNet");
    if (task.task != runtime::VoiceTaskKind::SourceSeparation || task.mode != runtime::RunMode::Offline) {
        throw std::runtime_error("TF-GridNet supports offline source separation");
    }
    using Storage = assets::TensorStorageType;
    const auto storage = runtime::parse_tensor_storage_option(options.options, "tf_gridnet.weight_type", Storage::Native,
        {Storage::Native, Storage::F32, Storage::F16, Storage::BF16, Storage::Q8_0, Storage::Q4_0,
         Storage::Q4_1, Storage::Q5_0, Storage::Q5_1, Storage::Q2_K, Storage::Q3_K,
         Storage::Q4_K, Storage::Q5_K, Storage::Q6_K});
    runtime_ = std::make_unique<TFGridNetBiLSTMAttentionRuntime>(assets_, execution_context(), storage);
    assets_->tensors->release_storage();
}

TFGridNetSession::~TFGridNetSession() = default;
std::string TFGridNetSession::family() const { return "tf_gridnet"; }
runtime::VoiceTaskKind TFGridNetSession::task_kind() const { return task_.task; }
runtime::RunMode TFGridNetSession::run_mode() const { return task_.mode; }

void TFGridNetSession::prepare(const runtime::SessionPreparationRequest & request) {
    if (!request.audio || request.audio->sample_rate <= 0 || request.audio->channels <= 0 ||
        (assets_->config.microphones != 1 && request.audio->channels != assets_->config.microphones)) {
        throw std::runtime_error("TF-GridNet requires audio matching the checkpoint microphone count");
    }
    mark_prepared();
}

runtime::TaskResult TFGridNetSession::run(const runtime::TaskRequest & request) {
    require_prepared("TF-GridNet run");
    runtime::validate_spec_backed_request_options(request.options, request.option_arrays, *contract_, "TF-GridNet");
    const auto started = std::chrono::steady_clock::now();
    if (!request.audio_input) throw std::runtime_error("TF-GridNet requires input audio");
    const auto & audio = *request.audio_input;
    const auto & config = assets_->config;
    if (audio.sample_rate <= 0 || audio.channels <= 0 || audio.samples.size() % audio.channels ||
        (config.microphones != 1 && audio.channels != config.microphones)) {
        throw std::runtime_error("TF-GridNet input audio format does not match the checkpoint");
    }
    std::vector<float> input;
    if (config.microphones == 1) {
        input = audio::convert_interleaved_audio_to_mono_torchaudio_sinc_hann_resampled(
            audio.samples, audio.sample_rate, audio.channels, config.sample_rate);
    } else {
        std::vector<float> planar;
        for (int channel = 0; channel < config.microphones; ++channel) {
            auto lane = audio::extract_interleaved_channel(audio.samples, audio.channels, channel);
            lane = audio::convert_interleaved_audio_to_mono_torchaudio_sinc_hann_resampled(
                lane, audio.sample_rate, 1, config.sample_rate);
            planar.insert(planar.end(), lane.begin(), lane.end());
        }
        input = audio::interleave_planar_channels(planar, config.microphones, planar.size() / config.microphones);
    }
    const int64_t samples = input.size() / config.microphones;
    if (samples <= config.n_fft / 2) throw std::runtime_error("TF-GridNet input must exceed its STFT reflection pad");
    const float chunk_seconds = runtime::parse_finite_float_option(request.options, {"audio_chunk_duration_sec"}).value_or(2.4f);
    const float overlap_seconds = runtime::parse_finite_float_option(request.options, {"audio_chunk_overlap_sec"}).value_or(1.6f);
    const int64_t chunk = std::llrint(double(chunk_seconds) * config.sample_rate);
    const int64_t overlap = std::llrint(double(overlap_seconds) * config.sample_rate);
    if (chunk_seconds < 0 || overlap_seconds < 0 ||
        (chunk_seconds > 0 && (chunk <= config.n_fft / 2 || overlap <= 0 || overlap >= chunk))) {
        throw std::runtime_error("TF-GridNet requires 0 < overlap < chunk duration, or zero chunk duration for whole-file inference");
    }
    std::vector<std::vector<float>> separated;
    if (chunk == 0 || samples <= chunk) {
        separated = runtime_->separate(input);
    } else {
        const audio::AudioChunkSpec spec{chunk, chunk - overlap, audio::AudioChunkPadMode::Zero};
        std::vector<float> planar(static_cast<size_t>(chunk * config.microphones));
        for (const auto & span : audio::plan_audio_chunks(samples, spec)) {
            audio::copy_interleaved_chunk_to_planar(planar, input, config.microphones, samples, span, spec);
            auto current = runtime_->separate(audio::interleave_planar_channels(planar, config.microphones, chunk));
            if (separated.empty()) {
                separated = std::move(current);
            } else {
                const auto permutation = match_speakers(separated, current, overlap);
                for (size_t speaker = 0; speaker < separated.size(); ++speaker) {
                    auto & output = separated[speaker];
                    const auto & next = current[permutation[speaker]];
                    for (int64_t t = 0; t < overlap; ++t) {
                        output[span.output_start_sample + t] = (output[span.output_start_sample + t] + next[t]) * 0.5f;
                    }
                    output.insert(output.end(), next.begin() + overlap, next.begin() + span.valid_samples);
                }
            }
            if (span.output_start_sample + span.valid_samples == samples) break;
        }
    }
    const auto normalize_option = runtime::find_option(request.options, {"normalize_output"});
    const bool normalize = !normalize_option || runtime::parse_bool_option(*normalize_option, "normalize_output");
    runtime::TaskResult result;
    for (size_t speaker = 0; speaker < separated.size(); ++speaker) {
        if (normalize) {
            float peak = 0;
            for (float value : separated[speaker]) peak = std::max(peak, std::abs(value));
            if (peak > 0) {
                for (float & value : separated[speaker]) value *= 0.9f / peak;
            }
        }
        runtime::NamedAudioBuffer named;
        named.id = "speaker_" + std::to_string(speaker + 1);
        named.audio = runtime::AudioBuffer{config.sample_rate, 1, std::move(separated[speaker])};
        result.named_audio_outputs.push_back(std::move(named));
    }
    debug::timing_log_scalar("session.wall_ms", debug::elapsed_ms(started));
    return result;
}

std::shared_ptr<runtime::IVoiceModelLoader> make_tf_gridnet_loader() {
    runtime::SpecBackedVoiceModelConfig<TFGridNetAssets> config;
    config.family = "tf_gridnet";
    config.load_assets = load_tf_gridnet_assets;
    config.create_session = [](const runtime::TaskSpec & task, const runtime::SessionOptions & options,
        std::shared_ptr<const TFGridNetAssets> assets, std::shared_ptr<const model_spec::ModelContract> contract) {
        return std::make_unique<TFGridNetSession>(task, options, std::move(assets), std::move(contract));
    };
    return runtime::make_spec_backed_voice_loader(std::move(config));
}

}  // namespace engine::models::tf_gridnet
