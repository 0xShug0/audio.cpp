#include "engine/models/rvc/native_pipeline.h"

#include "engine/framework/audio/conversion.h"
#include "engine/framework/audio/resampling.h"
#include "engine/framework/audio/waveform_ops.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/debug/trace.h"
#include "engine/framework/sampling/torch_random.h"
#include "engine/models/rvc/hubert.h"
#include "engine/models/rvc/rmvpe.h"
#include "engine/models/rvc/synthesizer.h"

#include "retrieval_index.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <mutex>
#include <numeric>
#include <stdexcept>
#include <sstream>
#include <unordered_map>
#include <utility>

namespace engine::models::rvc {
namespace {

constexpr int kContentSampleRate = 16000;
constexpr int64_t kRvcInterChannels = 192;
constexpr float kRmvpeThreshold = 0.03F;
constexpr double kPi = 3.141592653589793238462643383279502884;

struct RvcCustomF0Point {
    float time_seconds = 0.0F;
    float frequency_hz = 0.0F;
};

float option_clamp(float value, float lo, float hi) {
    return std::max(lo, std::min(hi, value));
}

std::vector<RvcCustomF0Point> read_custom_f0_file(const std::string & path) {
    if (path.empty()) {
        return {};
    }
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("failed to open RVC pitch_path: " + path);
    }
    std::vector<RvcCustomF0Point> points;
    std::string line;
    while (std::getline(input, line)) {
        if (line.empty()) {
            continue;
        }
        std::replace(line.begin(), line.end(), ',', ' ');
        std::istringstream row(line);
        RvcCustomF0Point point;
        if (!(row >> point.time_seconds >> point.frequency_hz)) {
            throw std::runtime_error("invalid RVC pitch_path row: " + line);
        }
        points.push_back(point);
    }
    if (!std::is_sorted(points.begin(), points.end(), [](const auto & lhs, const auto & rhs) {
            return lhs.time_seconds < rhs.time_seconds;
        })) {
        throw std::runtime_error("RVC pitch_path times must be sorted ascending");
    }
    return points;
}

float interpolate_custom_f0(const std::vector<RvcCustomF0Point> & points, float frame_index) {
    const float time_index = frame_index / 100.0F;
    if (time_index <= points.front().time_seconds) {
        return points.front().frequency_hz;
    }
    if (time_index >= points.back().time_seconds) {
        return points.back().frequency_hz;
    }
    const auto right = std::upper_bound(
        points.begin(),
        points.end(),
        time_index,
        [](float value, const RvcCustomF0Point & point) {
            return value < point.time_seconds;
        });
    const auto left = right - 1;
    const float span = right->time_seconds - left->time_seconds;
    if (span <= 0.0F) {
        throw std::runtime_error("RVC pitch_path contains duplicate time points");
    }
    const float frac = (time_index - left->time_seconds) / span;
    return left->frequency_hz * (1.0F - frac) + right->frequency_hz * frac;
}

void apply_custom_f0(
    std::vector<float> & f0,
    const std::vector<RvcCustomF0Point> & points,
    int audio_pad_duration_sec) {
    if (points.empty()) {
        return;
    }
    const int64_t start = static_cast<int64_t>(audio_pad_duration_sec) * 100;
    const int64_t count = static_cast<int64_t>(std::llround(
        static_cast<double>((points.back().time_seconds - points.front().time_seconds) * 100.0F + 1.0F)));
    if (count <= 0) {
        throw std::runtime_error("RVC pitch_path time span is invalid");
    }
    for (int64_t i = 0; i < count && start + i < static_cast<int64_t>(f0.size()); ++i) {
        f0[static_cast<size_t>(start + i)] = interpolate_custom_f0(points, static_cast<float>(i));
    }
}

float squared_l2(const float * lhs, const float * rhs, int64_t dim) {
    float sum = 0.0F;
    for (int64_t i = 0; i < dim; ++i) {
        const float diff = lhs[i] - rhs[i];
        sum += diff * diff;
    }
    return sum;
}

void apply_retrieval_blend(
    const std::vector<float> & source_features,
    std::vector<float> & features,
    int64_t frames,
    int64_t dim,
    const RvcRetrievalIndex & index,
    float retrieval_blend,
    size_t threads) {
    if (retrieval_blend == 0.0F) {
        return;
    }
    if (index.dim != dim ||
        static_cast<int64_t>(source_features.size()) != frames * dim ||
        static_cast<int64_t>(features.size()) != frames * dim) {
        throw std::runtime_error("RVC retrieval feature shape mismatch");
    }
    constexpr size_t kNeighbors = 8;
    const int worker_count = static_cast<int>(
        std::max<size_t>(
            1,
            std::min<size_t>(
                {threads, static_cast<size_t>(frames), static_cast<size_t>(std::numeric_limits<int>::max())})));
#pragma omp parallel num_threads(worker_count) if(frames >= 8)
    {
        std::vector<float> blended_frame(static_cast<size_t>(dim));
        std::array<float, kNeighbors> nearest_dist {};
        std::array<int64_t, kNeighbors> nearest_index {};
#pragma omp for schedule(static)
        for (int64_t frame = 0; frame < frames; ++frame) {
            const auto * query = source_features.data() + static_cast<size_t>(frame * dim);
            int64_t best_list = 0;
            float best_centroid = std::numeric_limits<float>::infinity();
            for (int64_t list = 0; list < index.nlist; ++list) {
                const float dist = squared_l2(query, index.centroids.data() + static_cast<size_t>(list * dim), dim);
                if (dist < best_centroid) {
                    best_centroid = dist;
                    best_list = list;
                }
            }
            const int64_t offset = index.list_offsets[static_cast<size_t>(best_list)];
            const int64_t length = index.list_lengths[static_cast<size_t>(best_list)];
            size_t nearest_count = 0;
            for (int64_t row = 0; row < length; ++row) {
                const int64_t vector_index = offset + row;
                const float dist = squared_l2(query, index.vectors.data() + static_cast<size_t>(vector_index * dim), dim);
                if (nearest_count < kNeighbors) {
                    nearest_dist[nearest_count] = dist;
                    nearest_index[nearest_count] = vector_index;
                    ++nearest_count;
                    continue;
                }
                size_t worst_slot = 0;
                float worst_dist = nearest_dist[0];
                for (size_t slot = 1; slot < kNeighbors; ++slot) {
                    if (nearest_dist[slot] > worst_dist) {
                        worst_dist = nearest_dist[slot];
                        worst_slot = slot;
                    }
                }
                if (dist < worst_dist) {
                    nearest_dist[worst_slot] = dist;
                    nearest_index[worst_slot] = vector_index;
                }
            }
            if (nearest_count == 0) {
                throw std::runtime_error("RVC retrieval selected an empty IVF list");
            }
            std::fill(blended_frame.begin(), blended_frame.end(), 0.0F);
            float weight_sum = 0.0F;
            for (size_t slot = 0; slot < nearest_count; ++slot) {
                const float dist = nearest_dist[slot];
                const int64_t vector_index = nearest_index[slot];
                if (dist <= 0.0F) {
                    std::copy(
                        index.vectors.data() + static_cast<size_t>(vector_index * dim),
                        index.vectors.data() + static_cast<size_t>((vector_index + 1) * dim),
                        blended_frame.data());
                    weight_sum = -1.0F;
                    break;
                }
                const float inv = 1.0F / dist;
                const float weight = inv * inv;
                weight_sum += weight;
                const auto * src = index.vectors.data() + static_cast<size_t>(vector_index * dim);
                for (int64_t i = 0; i < dim; ++i) {
                    blended_frame[static_cast<size_t>(i)] += src[i] * weight;
                }
            }
            if (weight_sum > 0.0F) {
                for (int64_t i = 0; i < dim; ++i) {
                    blended_frame[static_cast<size_t>(i)] /= weight_sum;
                }
            }
            auto * feature = features.data() + static_cast<size_t>(frame * dim);
            for (int64_t i = 0; i < dim; ++i) {
                feature[i] =
                    blended_frame[static_cast<size_t>(i)] * retrieval_blend +
                    source_features[static_cast<size_t>(frame * dim + i)] * (1.0F - retrieval_blend);
            }
        }
    }
}

std::vector<float> mono_16k(const runtime::AudioBuffer & source) {
    if (source.sample_rate <= 0 || source.channels <= 0 || source.samples.empty()) {
        throw std::runtime_error("RVC source audio is invalid");
    }
    if (source.samples.size() % static_cast<size_t>(source.channels) != 0) {
        throw std::runtime_error("RVC source audio samples are not divisible by channel count");
    }
    return engine::audio::convert_interleaved_audio_to_mono_linear_resampled(
        source.samples,
        source.sample_rate,
        source.channels,
        kContentSampleRate);
}

void high_pass_48hz_in_place(std::vector<float> & samples) {
    if (samples.empty()) {
        return;
    }
    constexpr int order = 5;
    constexpr int padlen = 18;
    constexpr double b[order + 1] = {
        0.9699606451838447,
        -4.849803225919223,
        9.699606451838447,
        -9.699606451838447,
        4.849803225919223,
        -0.9699606451838447};
    constexpr double a[order + 1] = {
        1.0,
        -4.939001819168364,
        9.757863526739543,
        -9.639544849413458,
        4.761506797356209,
        -0.9408236532054606};
    constexpr double zi[order] = {
        -0.9699604796995847,
        3.8798419288925783,
        -5.819762908173043,
        3.879841948472456,
        -0.9699604894923387};
    if (samples.size() <= static_cast<size_t>(padlen)) {
        throw std::runtime_error("RVC high-pass filtfilt input is too short");
    }
    std::vector<double> working(samples.size() + static_cast<size_t>(2 * padlen), 0.0);
    const double first = samples.front();
    const double last = samples.back();
    for (int i = 0; i < padlen; ++i) {
        working[static_cast<size_t>(i)] = 2.0 * first - samples[static_cast<size_t>(padlen - i)];
    }
    for (size_t i = 0; i < samples.size(); ++i) {
        working[static_cast<size_t>(padlen) + i] = samples[i];
    }
    for (int i = 0; i < padlen; ++i) {
        working[static_cast<size_t>(padlen) + samples.size() + static_cast<size_t>(i)] =
            2.0 * last - samples[samples.size() - 2U - static_cast<size_t>(i)];
    }
    const auto lfilter = [&](std::vector<double> & values) {
        double state[order];
        for (int i = 0; i < order; ++i) {
            state[i] = zi[i] * values.front();
        }
        for (double & x : values) {
            const double y = b[0] * x + state[0];
            for (int i = 1; i < order; ++i) {
                state[i - 1] = b[i] * x + state[i] - a[i] * y;
            }
            state[order - 1] = b[order] * x - a[order] * y;
            x = y;
        }
    };
    lfilter(working);
    std::reverse(working.begin(), working.end());
    lfilter(working);
    std::reverse(working.begin(), working.end());
    for (size_t i = 0; i < samples.size(); ++i) {
        samples[i] = static_cast<float>(working[static_cast<size_t>(padlen) + i]);
    }
}

int coarse_pitch_bin(float f0_hz) {
    if (f0_hz <= 0.0F) {
        return 1;
    }
    const float f0_mel = 1127.0F * std::log(1.0F + f0_hz / 700.0F);
    const float f0_min = 1127.0F * std::log(1.0F + 50.0F / 700.0F);
    const float f0_max = 1127.0F * std::log(1.0F + 1100.0F / 700.0F);
    const float scaled = (f0_mel - f0_min) * 254.0F / (f0_max - f0_min) + 1.0F;
    return static_cast<int>(std::lrint(option_clamp(scaled, 1.0F, 255.0F)));
}

void median_filter_f0(std::vector<float> & f0, int radius) {
    if (radius <= 0 || f0.empty()) {
        return;
    }
    std::vector<float> filtered(f0.size(), 0.0F);
    std::vector<float> window;
    window.reserve(static_cast<size_t>(radius * 2 + 1));
    for (size_t i = 0; i < f0.size(); ++i) {
        window.clear();
        const size_t begin = i > static_cast<size_t>(radius) ? i - static_cast<size_t>(radius) : 0;
        const size_t end = std::min(f0.size(), i + static_cast<size_t>(radius) + 1);
        for (size_t j = begin; j < end; ++j) {
            window.push_back(f0[j]);
        }
        const auto mid = window.begin() + static_cast<std::ptrdiff_t>(window.size() / 2);
        std::nth_element(window.begin(), mid, window.end());
        filtered[i] = *mid;
    }
    f0 = std::move(filtered);
}

std::vector<float> frame_rms_linear(const std::vector<float> & audio, int sample_rate, int output_frames) {
    const int frame_length = (sample_rate / 2) * 2;
    const int hop = sample_rate / 2;
    if (frame_length <= 0 || hop <= 0 || audio.empty() || output_frames <= 0) {
        throw std::runtime_error("RVC RMS input is invalid");
    }
    const int64_t frame_count = std::max<int64_t>(1, (static_cast<int64_t>(audio.size()) + hop - 1) / hop);
    std::vector<float> rms(static_cast<size_t>(frame_count), 0.0F);
    for (int64_t frame = 0; frame < frame_count; ++frame) {
        const int64_t center = frame * hop;
        const int64_t begin = center - frame_length / 2;
        double sum = 0.0;
        for (int i = 0; i < frame_length; ++i) {
            const int64_t src = std::clamp<int64_t>(begin + i, 0, static_cast<int64_t>(audio.size()) - 1);
            const float value = audio[static_cast<size_t>(src)];
            sum += static_cast<double>(value) * static_cast<double>(value);
        }
        rms[static_cast<size_t>(frame)] = static_cast<float>(std::sqrt(sum / static_cast<double>(frame_length)));
    }
    std::vector<float> out(static_cast<size_t>(output_frames), 0.0F);
    if (output_frames == 1 || frame_count == 1) {
        std::fill(out.begin(), out.end(), rms.front());
        return out;
    }
    for (int i = 0; i < output_frames; ++i) {
        const float pos = static_cast<float>(i) * static_cast<float>(frame_count - 1) / static_cast<float>(output_frames - 1);
        const int64_t left = static_cast<int64_t>(std::floor(pos));
        const int64_t right = std::min<int64_t>(frame_count - 1, left + 1);
        const float frac = pos - static_cast<float>(left);
        out[static_cast<size_t>(i)] = rms[static_cast<size_t>(left)] * (1.0F - frac) + rms[static_cast<size_t>(right)] * frac;
    }
    return out;
}

void apply_rms_mix(
    const std::vector<float> & source_16k,
    std::vector<float> & converted,
    int target_sample_rate,
    float rms_mix_rate) {
    if (rms_mix_rate == 1.0F) {
        return;
    }
    const auto rms1 = frame_rms_linear(source_16k, kContentSampleRate, static_cast<int>(converted.size()));
    auto rms2 = frame_rms_linear(converted, target_sample_rate, static_cast<int>(converted.size()));
    for (size_t i = 0; i < converted.size(); ++i) {
        rms2[i] = std::max(rms2[i], 1.0e-6F);
        const float factor =
            std::pow(rms1[i], 1.0F - rms_mix_rate) *
            std::pow(rms2[i], rms_mix_rate - 1.0F);
        converted[i] *= factor;
    }
}

std::vector<int64_t> quiet_split_points(
    const std::vector<float> & audio,
    int split_query_sec,
    int split_center_sec,
    int split_threshold_sec) {
    constexpr int64_t window = 160;
    const int64_t t_query = static_cast<int64_t>(split_query_sec) * kContentSampleRate;
    const int64_t t_center = static_cast<int64_t>(split_center_sec) * kContentSampleRate;
    const int64_t t_max = static_cast<int64_t>(split_threshold_sec) * kContentSampleRate;
    if (static_cast<int64_t>(audio.size()) <= t_max) {
        return {};
    }
    if (t_query <= 0 || t_center <= 0 || t_max <= 0) {
        throw std::runtime_error("RVC chunk timing options must be positive");
    }
    const auto audio_pad = engine::audio::reflect_pad_samples(audio, window / 2, window / 2);
    std::vector<float> audio_sum(audio.size(), 0.0F);
    for (int64_t i = 0; i < window; ++i) {
        for (size_t t = 0; t < audio.size(); ++t) {
            audio_sum[t] += std::abs(audio_pad[static_cast<size_t>(i) + t]);
        }
    }
    std::vector<int64_t> splits;
    for (int64_t t = t_center; t < static_cast<int64_t>(audio.size()); t += t_center) {
        const int64_t begin = std::max<int64_t>(0, t - t_query);
        const int64_t end = std::min<int64_t>(static_cast<int64_t>(audio_sum.size()), t + t_query);
        if (begin >= end) {
            continue;
        }
        const auto min_it = std::min_element(audio_sum.begin() + begin, audio_sum.begin() + end);
        splits.push_back(t - t_query + static_cast<int64_t>(std::distance(audio_sum.begin() + begin, min_it)));
    }
    return splits;
}

RvcSynthesizerInput make_synthesizer_input(
    const RvcHubertFeatures & content,
    const RvcHubertFeatures * original_content,
    const std::vector<float> & f0,
    const RvcInferenceConfig & config,
    int sample_rate,
    int64_t hop_samples,
    int64_t target_frames,
    bool has_f0) {
    if (content.frames <= 0 || content.dim <= 0 || content.values.empty()) {
        throw std::runtime_error("RVC synthesizer input requires content features");
    }
    if (has_f0 && f0.empty()) {
        throw std::runtime_error("RVC synthesizer input requires f0");
    }
    const float semitone = std::pow(2.0F, static_cast<float>(config.semitone_shift) / 12.0F);
    const int64_t doubled_frames = content.frames * 2;
    const int64_t frames = std::min<int64_t>(
        target_frames,
        has_f0 ? std::min<int64_t>(doubled_frames, static_cast<int64_t>(f0.size())) : doubled_frames);
    if (frames <= 0) {
        throw std::runtime_error("RVC synthesizer input has no aligned frames");
    }
    if (original_content != nullptr &&
        (original_content->frames != content.frames || original_content->dim != content.dim ||
         original_content->values.size() != content.values.size())) {
        throw std::runtime_error("RVC unvoiced_protection feature shape mismatch");
    }
    RvcSynthesizerInput out;
    out.frames = frames;
    out.feature_dim = content.dim;
    out.speaker_id = config.speaker_id;
    out.features.resize(static_cast<size_t>(frames * content.dim));
    for (int64_t t = 0; t < frames; ++t) {
        const int64_t src_t = std::min(content.frames - 1, t / 2);
        const auto * src = content.values.data() + static_cast<size_t>(src_t * content.dim);
        auto * dst = out.features.data() + static_cast<size_t>(t * content.dim);
        std::copy(src, src + content.dim, dst);
    }
    if (!has_f0) {
        return out;
    }
    std::vector<float> shifted_f0(static_cast<size_t>(f0.size()), 0.0F);
    for (size_t i = 0; i < f0.size(); ++i) {
        shifted_f0[i] = f0[i] * semitone;
    }
    out.pitch.resize(static_cast<size_t>(frames), 1);
    out.pitchf.resize(static_cast<size_t>(frames), 0.0F);
    for (int64_t t = 0; t < frames; ++t) {
        const float shifted = shifted_f0[static_cast<size_t>(t)];
        out.pitchf[static_cast<size_t>(t)] = shifted;
        out.pitch[static_cast<size_t>(t)] = coarse_pitch_bin(shifted);
    }
    if (original_content != nullptr && config.unvoiced_protection < 0.5F) {
        for (int64_t t = 0; t < frames; ++t) {
            const float pitchff = out.pitchf[static_cast<size_t>(t)] < 1.0F ? config.unvoiced_protection : 1.0F;
            const int64_t src_t = std::min(original_content->frames - 1, t / 2);
            const auto * original = original_content->values.data() + static_cast<size_t>(src_t * content.dim);
            auto * dst = out.features.data() + static_cast<size_t>(t * content.dim);
            for (int64_t i = 0; i < content.dim; ++i) {
                dst[i] = dst[i] * pitchff + original[i] * (1.0F - pitchff);
            }
        }
    }
    const int64_t upsample = hop_samples;
    std::vector<float> cumulative(static_cast<size_t>(frames), 0.0F);
    double running = 0.0;
    for (int64_t t = 0; t < frames; ++t) {
        const float rad = std::fmod(
            std::max(0.0F, out.pitchf[static_cast<size_t>(t)]) / static_cast<float>(sample_rate),
            1.0F);
        running += static_cast<double>(rad);
        cumulative[static_cast<size_t>(t)] = static_cast<float>(running) * static_cast<float>(upsample);
    }
    out.sine_source.resize(static_cast<size_t>(frames * upsample), 0.0F);
    auto noise = engine::sampling::generate_torch_cuda_randn(
        out.sine_source.size(),
        1234,
        engine::sampling::TorchRandnPrecision::Float32,
        static_cast<uint64_t>(kRvcInterChannels * frames));
    double sine_cumsum = 0.0;
    float prev_tmp_mod = 0.0F;
    const int64_t output_frames = frames * upsample;
    for (int64_t i = 0; i < output_frames; ++i) {
        const int64_t source_frame = std::min<int64_t>(frames - 1, i / upsample);
        const float base_rad =
            std::fmod(
                std::max(0.0F, out.pitchf[static_cast<size_t>(source_frame)]) / static_cast<float>(sample_rate),
                1.0F);
        float interpolated = cumulative.front();
        if (output_frames > 1 && frames > 1) {
            const float pos =
                static_cast<float>(i) * static_cast<float>(frames - 1) /
                static_cast<float>(output_frames - 1);
            const int64_t left = static_cast<int64_t>(std::floor(pos));
            const int64_t right = std::min<int64_t>(frames - 1, left + 1);
            const float frac = pos - static_cast<float>(left);
            interpolated =
                cumulative[static_cast<size_t>(left)] * (1.0F - frac) +
                cumulative[static_cast<size_t>(right)] * frac;
        }
        const float tmp_mod = interpolated - std::floor(interpolated);
        const bool wrapped = i > 0 && (tmp_mod - prev_tmp_mod) < 0.0F;
        prev_tmp_mod = tmp_mod;
        sine_cumsum += static_cast<double>(base_rad + (wrapped ? -1.0F : 0.0F));
        const bool voiced = out.pitchf[static_cast<size_t>(source_frame)] > 0.0F;
        const float noise_amp = voiced ? 0.003F : (0.1F / 3.0F);
        const float sine = std::sin(static_cast<float>(sine_cumsum) * static_cast<float>(2.0 * kPi)) * 0.1F;
        out.sine_source[static_cast<size_t>(i)] =
            voiced ? (sine + noise_amp * noise[static_cast<size_t>(i)]) :
                     (noise_amp * noise[static_cast<size_t>(i)]);
        }
    return out;
}

}  // namespace

struct RvcNativePipeline::State {
    std::shared_ptr<const RvcAssets> assets;
    engine::core::BackendConfig backend;
    engine::assets::TensorStorageType storage_type = engine::assets::TensorStorageType::Native;
    RvcHubertEncoder hubert;
    RvcRmvpeF0Extractor rmvpe;
    std::unordered_map<std::string, std::unique_ptr<RvcSynthesizer>> synthesizers;
    std::unordered_map<std::string, std::unique_ptr<RvcRetrievalIndex>> retrieval_indices;
    std::mutex mutex;
};

RvcNativePipeline::RvcNativePipeline(
    std::shared_ptr<const RvcAssets> assets,
    engine::core::BackendConfig backend,
    engine::assets::TensorStorageType storage_type)
    : state_(std::make_shared<State>()) {
    if (assets == nullptr || assets->rmvpe == nullptr || assets->hubert == nullptr) {
        throw std::runtime_error("RVC native pipeline requires safetensors assets");
    }
    state_->assets = std::move(assets);
    state_->backend = std::move(backend);
    state_->storage_type = storage_type;
    state_->hubert = RvcHubertEncoder(state_->assets->hubert, state_->backend, state_->storage_type);
    state_->rmvpe = RvcRmvpeF0Extractor(
        state_->assets->rmvpe,
        state_->backend,
        state_->storage_type);
}

RvcNativePipeline::~RvcNativePipeline() = default;
RvcNativePipeline::RvcNativePipeline(RvcNativePipeline &&) noexcept = default;
RvcNativePipeline & RvcNativePipeline::operator=(RvcNativePipeline &&) noexcept = default;

runtime::AudioBuffer RvcNativePipeline::infer(
    const runtime::AudioBuffer & source,
    const RvcVoiceModel & voice,
    const RvcInferenceConfig & config,
    size_t threads) {
    if (state_ == nullptr) {
        throw std::runtime_error("RVC native pipeline is not initialized");
    }
    if (voice.has_f0 && config.pitch_extractor != "rmvpe") {
        throw std::runtime_error("RVC native inference currently supports only rmvpe pitch_extractor");
    }
    if (config.retrieval_blend < 0.0F || config.retrieval_blend > 1.0F) {
        throw std::runtime_error("RVC retrieval_blend must be in [0, 1]");
    }
    if (config.unvoiced_protection < 0.0F || config.unvoiced_protection > 1.0F) {
        throw std::runtime_error("RVC unvoiced_protection must be in [0, 1]");
    }
    if (config.speaker_id < 0 || config.speaker_id >= voice.speaker_count) {
        throw std::runtime_error("RVC speaker_id is outside the checkpoint speaker table");
    }
    if (config.audio_pad_duration_sec <= 0 || config.split_query_sec <= 0 || config.split_center_sec <= 0 ||
        config.split_threshold_sec <= 0) {
        throw std::runtime_error("RVC chunk timing options must be positive");
    }
    std::lock_guard<std::mutex> lock(state_->mutex);
    auto content_audio = mono_16k(source);
    high_pass_48hz_in_place(content_audio);
    engine::debug::trace_log_f32(
        "rvc.audio.filtered_16k",
        {static_cast<int64_t>(content_audio.size())},
        content_audio);
    constexpr int64_t rvc_hop_samples = 160;
    const int64_t content_pad_samples = kContentSampleRate * static_cast<int64_t>(config.audio_pad_duration_sec);
    const auto padded_audio = engine::audio::reflect_pad_samples(
        content_audio,
        content_pad_samples,
        content_pad_samples);
    engine::debug::trace_log_f32(
        "rvc.audio.padded_16k",
        {static_cast<int64_t>(padded_audio.size())},
        padded_audio);
    std::vector<float> f0;
    RvcInferenceConfig synth_config = config;
    if (voice.has_f0) {
        f0 = state_->rmvpe.infer_16k_mono(padded_audio, kRmvpeThreshold, threads);
        if (f0.empty()) {
            throw std::runtime_error("RVC RMVPE produced no f0 frames");
        }
        if (config.pitch_filter_radius > 2) {
            median_filter_f0(f0, 1);
        }
        const auto custom_f0 = read_custom_f0_file(config.pitch_path);
        if (!custom_f0.empty()) {
            const float semitone = std::pow(2.0F, static_cast<float>(config.semitone_shift) / 12.0F);
            for (auto & value : f0) {
                value *= semitone;
            }
            apply_custom_f0(f0, custom_f0, config.audio_pad_duration_sec);
            synth_config.semitone_shift = 0;
        }
        engine::debug::trace_log_f32(
            "rvc.f0.pitchf",
            {static_cast<int64_t>(f0.size())},
            f0);
    }
    auto synth_it = state_->synthesizers.find(voice.id);
    if (synth_it == state_->synthesizers.end()) {
        synth_it = state_->synthesizers.emplace(
            voice.id,
            std::make_unique<RvcSynthesizer>(
                voice.checkpoint,
                state_->backend,
                state_->storage_type,
                voice.sample_rate,
                voice.synthesizer_layout,
                voice.version == "v1",
                voice.has_f0)).first;
    }
    RvcRetrievalIndex * retrieval = nullptr;
    if (config.retrieval_blend != 0.0F) {
        const bool packaged_index = config.retrieval_index_path.empty();
        if (packaged_index && voice.index_vectors == nullptr) {
            throw std::runtime_error("RVC retrieval_blend requires retrieval_index_path for a user voice model");
        }
        const std::filesystem::path index_path = std::filesystem::path(config.retrieval_index_path);
        const std::string index_key = packaged_index
            ? voice.id + ":packaged_index_vectors"
            : std::filesystem::absolute(index_path).lexically_normal().string();
        auto index_it = state_->retrieval_indices.find(index_key);
        if (index_it == state_->retrieval_indices.end()) {
            const int64_t index_dim = voice.version == "v1" ? 256 : 768;
            auto loaded_index = packaged_index
                ? load_rvc_retrieval_index(voice.index_vectors, index_dim, voice.id + ":packaged_index_vectors")
                : load_rvc_retrieval_index(index_path, index_dim);
            index_it = state_->retrieval_indices.emplace(
                index_key,
                std::make_unique<RvcRetrievalIndex>(std::move(loaded_index))).first;
        }
        retrieval = index_it->second.get();
    }

    const auto splits = quiet_split_points(
        content_audio,
        config.split_query_sec,
        config.split_center_sec,
        config.split_threshold_sec);
    const int64_t target_pad_samples =
        static_cast<int64_t>(voice.sample_rate) * static_cast<int64_t>(config.audio_pad_duration_sec);
    const int64_t t_pad2 = 2 * content_pad_samples;
    std::vector<float> converted;
    int output_sample_rate = voice.sample_rate;
    int64_t segment_start = 0;
    const auto run_segment = [&](int64_t split_point, bool final_segment) {
        const int64_t segment_end = final_segment
            ? static_cast<int64_t>(padded_audio.size())
            : std::min<int64_t>(static_cast<int64_t>(padded_audio.size()), split_point + t_pad2 + rvc_hop_samples);
        if (segment_start >= segment_end) {
            throw std::runtime_error("RVC chunk produced an empty audio segment");
        }
        const std::vector<float> segment_audio(
            padded_audio.begin() + static_cast<std::ptrdiff_t>(segment_start),
            padded_audio.begin() + static_cast<std::ptrdiff_t>(segment_end));
        std::vector<float> segment_f0;
        int64_t target_frames = static_cast<int64_t>(segment_audio.size()) / rvc_hop_samples;
        if (voice.has_f0) {
            const int64_t f0_begin = segment_start / rvc_hop_samples;
            const int64_t f0_end = final_segment
                ? static_cast<int64_t>(f0.size())
                : std::min<int64_t>(static_cast<int64_t>(f0.size()), (split_point + t_pad2) / rvc_hop_samples);
            if (f0_begin >= f0_end) {
                throw std::runtime_error("RVC chunk produced an empty F0 segment");
            }
            segment_f0.assign(
                f0.begin() + static_cast<std::ptrdiff_t>(f0_begin),
                f0.begin() + static_cast<std::ptrdiff_t>(f0_end));
            target_frames = static_cast<int64_t>(segment_f0.size());
        }
        auto content = state_->hubert.encode_16k_mono(segment_audio, voice.version == "v1");
        if (content.frames <= 0 || (voice.version == "v1" && content.dim != 256) ||
            (voice.version != "v1" && content.dim != 768)) {
            throw std::runtime_error("RVC HuBERT produced invalid content feature shape");
        }
        engine::debug::trace_log_f32(
            "rvc.hubert.features",
            {1, content.frames, content.dim},
            content.values);
        auto original_content = content;
        if (retrieval != nullptr) {
            const auto retrieval_start = std::chrono::steady_clock::now();
            apply_retrieval_blend(
                original_content.values,
                content.values,
                content.frames,
                content.dim,
                *retrieval,
                config.retrieval_blend,
                threads);
            engine::debug::timing_log_scalar("rvc.retrieval_blend_ms", engine::debug::elapsed_ms(retrieval_start));
        }
        const RvcHubertFeatures * protect_source =
            (voice.has_f0 && synth_config.unvoiced_protection < 0.5F) ? &original_content : nullptr;
        auto synth_input = make_synthesizer_input(
            content,
            protect_source,
            segment_f0,
            synth_config,
            voice.sample_rate,
            voice.synthesizer_layout.hop_samples,
            target_frames,
            voice.has_f0);
        engine::debug::trace_log_f32(
            "rvc.synth.features",
            {1, synth_input.frames, synth_input.feature_dim},
            synth_input.features);
        if (voice.has_f0) {
            engine::debug::trace_log_i32(
                "rvc.synth.pitch",
                {1, synth_input.frames},
                synth_input.pitch);
            engine::debug::trace_log_f32(
                "rvc.synth.pitchf",
                {1, synth_input.frames},
                synth_input.pitchf);
            engine::debug::trace_log_f32(
                "rvc.synth.sine_source",
                {1, synth_input.frames * voice.synthesizer_layout.hop_samples, 1},
                synth_input.sine_source);
        }
        auto synth_output = synth_it->second->infer(synth_input);
        output_sample_rate = synth_output.sample_rate;
        engine::debug::trace_log_f32(
            "rvc.synth.raw_output",
            {static_cast<int64_t>(synth_output.audio.size())},
            synth_output.audio);
        auto segment_output = std::move(synth_output.audio);
        if (static_cast<int64_t>(segment_output.size()) <= 2 * target_pad_samples) {
            throw std::runtime_error("RVC synthesized audio is too short for Python padding crop");
        }
        converted.insert(
            converted.end(),
            segment_output.begin() + static_cast<std::ptrdiff_t>(target_pad_samples),
            segment_output.end() - static_cast<std::ptrdiff_t>(target_pad_samples));
    };
    for (const auto split : splits) {
        const int64_t aligned_split = (split / rvc_hop_samples) * rvc_hop_samples;
        run_segment(aligned_split, false);
        segment_start = aligned_split;
    }
    run_segment(static_cast<int64_t>(content_audio.size()), true);

    apply_rms_mix(content_audio, converted, output_sample_rate, config.rms_mix_rate);
    const int target_sr = config.output_sample_rate > 0 ? config.output_sample_rate : output_sample_rate;
    if (target_sr != output_sample_rate) {
        converted = engine::audio::resample_mono_linear(converted, output_sample_rate, target_sr);
    }
    return runtime::AudioBuffer{
        target_sr,
        1,
        std::move(converted),
    };
}

}  // namespace engine::models::rvc
