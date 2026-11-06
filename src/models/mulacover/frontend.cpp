#include "engine/models/mulacover/frontend.h"

#include "engine/framework/audio/dsp.h"
#include "engine/framework/audio/fft.h"
#include "engine/framework/audio/resampling.h"
#include "engine/framework/debug/trace.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <complex>
#include <limits>
#include <map>
#include <stdexcept>
#include <utility>

namespace engine::models::mulacover {
namespace {
constexpr double pi = 3.14159265358979323846264338327950288;
constexpr double sample_rate_hz = 22050.0;
constexpr size_t hop_length = 512;
constexpr int bins_per_octave = 36;
constexpr int bins = 288;

// filters._relative_bandwidth: endpoint differences and centered interior differences.
std::vector<double> relative_bandwidth(const std::vector<double> & frequencies) {
    std::vector<double> alpha(frequencies.size());
    for (size_t i = 0; i < frequencies.size(); ++i) {
        const size_t lo = i == 0 ? 0 : i-1;
        const size_t hi = i+1 == frequencies.size() ? i : i+1;
        const double exponent = 2 * (std::log2(frequencies[hi])-std::log2(frequencies[lo])) / (hi-lo);
        const double ratio = std::pow(2.0,exponent);
        alpha[i] = (ratio-1)/(ratio+1);
    }
    return alpha;
}
}  // namespace

class ChordFrontend::Impl {
    struct FFTWorkspace {
        std::shared_ptr<audio::RealFFTPlan> plan;
        std::vector<float> real, imaginary;
        std::vector<std::complex<float>> spectrum, imaginary_spectrum;
        std::vector<double> hann;
        explicit FFTWorkspace(size_t size)
            : plan(audio::get_real_fft_plan(size)), real(size), imaginary(size),
              spectrum(size/2+1), imaginary_spectrum(size/2+1), hann(size) {
            for (size_t i = 0; i < size; ++i) hann[i] = 0.5+0.5*std::cos(-pi+i*(2*pi/size));
        }
    };
    struct Coefficient {
        size_t bin;
        std::complex<float> value;
    };
    struct Band {
        int first = 0;
        bool pseudo = false;
        size_t fft_size = 0;
        size_t hop = 0;
        // Full CQT lengths at the sampling rate AFTER early downsampling.
        std::vector<double> lengths;
        std::vector<std::vector<Coefficient>> filters;
    };
    struct FilterBank {
        int early_downsamples = 0;
        std::vector<Band> bands; // Pseudo first, then full octaves from high to low.
    };
    struct Response {
        std::vector<float> values; // Row-major frames x band filters.
        size_t frames = 0;
    };
    struct TempoWorkspace {
        audio::AudioTensor mel_filters, magnitude, mel;
        std::vector<float> window, power, onset, normalized_onset;
        std::vector<double> ac_window, mean_tempogram, gaussian;
        std::vector<double> local_score, cumulative_score, selected_score, maxima;
        std::vector<int32_t> backlink;
        std::vector<uint8_t> beats;
        std::vector<int32_t> beat_frames;
    };
    std::map<int, TempoWorkspace> tempo_;
    std::map<size_t, FFTWorkspace> fft_;
    std::map<size_t, std::vector<double>> wavelet_windows_;
    std::map<int, FilterBank> banks_;
    std::vector<float> mono_;
    std::array<std::vector<float>, 10> octaves_;
    std::vector<Response> responses_;
    std::vector<float> pitches_, magnitudes_, sorted_magnitudes_;
    std::vector<float> magnitude_frame_, full_cqt_;
    std::vector<std::complex<double>> wavelet_;
    ChordFrontendOutput output_;

    double estimate_bpm(const std::vector<float> & y, int source_rate) {
        // Installed librosa beat.py: beat_track(y, sr), including its MEDIAN
        // onset aggregation. Tempo selection precedes DP and is not revised by it.
        constexpr int64_t n_fft = 2048;
        constexpr size_t mel_count = 128;
        constexpr size_t fft_bins = n_fft/2+1;
        auto & work = tempo_[source_rate];
        if (work.window.empty()) {
            const auto & hann = fft_.try_emplace(n_fft,n_fft).first->second.hann;
            work.window.assign(hann.begin(),hann.end());
            // filters.mel: float64 Slaney coordinates; cast triangular weights to
            // F32 BEFORE applying the float64 Slaney area normalization.
            // The framework's default builder uses F32 coordinates, so use its
            // existing custom-filter projection surface with these source weights.
            const double logstep = std::log(6.4)/27.0;
            const double nyquist = source_rate/2.0;
            const double mel_max = nyquist >= 1000
                ? 15+std::log(nyquist/1000)/logstep : nyquist/(200.0/3);
            std::array<double,mel_count+2> hz{};
            for (size_t i = 0; i < hz.size(); ++i) {
                const double mel = i*(mel_max/(mel_count+1));
                hz[i] = mel >= 15 ? 1000*std::exp(logstep*(mel-15)) : mel*(200.0/3);
            }
            work.mel_filters.shape = {static_cast<int64_t>(mel_count),static_cast<int64_t>(fft_bins)};
            work.mel_filters.values.resize(mel_count*fft_bins);
            for (size_t m = 0; m < mel_count; ++m) {
                const double area = 2/(hz[m+2]-hz[m]);
                for (size_t f = 0; f < fft_bins; ++f) {
                    const double frequency = f*(static_cast<double>(source_rate)/n_fft);
                    const double lower = (frequency-hz[m])/(hz[m+1]-hz[m]);
                    const double upper = (hz[m+2]-frequency)/(hz[m+2]-hz[m+1]);
                    const float weight = static_cast<float>(std::max(0.0,std::min(lower,upper)));
                    work.mel_filters.values[m*fft_bins+f] = static_cast<float>(weight*area);
                }
            }
        }
        const audio::STFTConfig config{n_fft,static_cast<int64_t>(hop_length),n_fft,
                                       true,audio::STFTPadMode::Constant};
        work.magnitude = audio::STFT().compute_magnitude(y,work.window,1,
            static_cast<int64_t>(y.size()),config,1);
        const size_t frames = static_cast<size_t>(work.magnitude.shape[2]);
        work.power.resize(work.magnitude.values.size());
        for (size_t i = 0; i < work.power.size(); ++i)
            work.power[i] = work.magnitude.values[i]*work.magnitude.values[i];
        work.mel = audio::MelFilterbank().compute_custom(work.power,1,fft_bins,
            static_cast<int64_t>(frames),work.mel_filters);

        // power_to_db(ref=1, amin=1e-10, top_db=80), global clipping over M x T.
        float max_db = -std::numeric_limits<float>::infinity();
        for (auto & value : work.mel.values) {
            value = 10.0f*std::log10(std::max(1e-10f,value));
            max_db = std::max(max_db,value);
        }
        for (auto & value : work.mel.values) value = std::max(value,max_db-80.0f);
        work.onset.assign(frames,0.0f);
        std::array<float,mel_count> flux{};
        // onset_strength_multi lag=1, max_size=1, detrend=False. The padded
        // difference t -> t+1 is shifted by 1+2048//(2*512)=3, then truncated.
        for (size_t t = 0; t+3 < frames; ++t) {
            for (size_t m = 0; m < mel_count; ++m)
                flux[m] = std::max(0.0f,work.mel.values[m*frames+t+1]-work.mel.values[m*frames+t]);
            std::sort(flux.begin(),flux.end());
            work.onset[t+3] = (flux[63]+flux[64])*0.5f;
        }
        if (std::none_of(work.onset.begin(),work.onset.end(),[](float value) { return value != 0; })) {
            // beat_track returns (0, []); upstream transcription rejects that BPM.
            work.beat_frames.clear();
            return 0;
        }

        // feature.tempo(ac_size=8): time_to_frames floors after time_to_samples.
        const size_t win_length = static_cast<size_t>(8LL*source_rate/hop_length);
        if (win_length == 0) throw std::invalid_argument("Tempo autocorrelation window must be positive");
        if (work.ac_window.empty()) {
            work.ac_window.resize(win_length);
            for (size_t i = 0; i < win_length; ++i)
                work.ac_window[i] = win_length == 1 ? 1 : 0.5+0.5*std::cos(-pi+i*(2*pi/win_length));
        }
        // scipy.fft.next_fast_len(2*win_length-1, real=True): real transforms
        // use 2/3/5-smooth lengths. Only FFT length planning happens here.
        const size_t required = 2*win_length-1;
        size_t fft_size = 1;
        while (fft_size < required) fft_size *= 2;
        for (size_t p2 = 1; p2 <= fft_size; p2 *= 2) {
            for (size_t p23 = p2; p23 <= fft_size; p23 *= 3) {
                size_t candidate = p23;
                while (candidate < required) candidate *= 5;
                fft_size = std::min(fft_size,candidate);
            }
        }
        auto & ac = fft_.try_emplace(fft_size,fft_size).first->second;
        work.mean_tempogram.assign(win_length,0.0);
        const int64_t pad = static_cast<int64_t>(win_length/2);
        for (size_t t = 0; t < frames; ++t) {
            std::fill(ac.real.begin(),ac.real.end(),0.0f);
            for (size_t i = 0; i < win_length; ++i) {
                const int64_t index = static_cast<int64_t>(t+i)-pad;
                float value;
                // np.pad(mode='linear_ramp', end_values=[0,0]) BEFORE windowing.
                if (index < 0)
                    value = static_cast<float>(work.onset.front()*static_cast<double>(index+pad)/pad);
                else if (static_cast<size_t>(index) >= frames)
                    value = static_cast<float>(work.onset.back()*static_cast<double>(
                        static_cast<int64_t>(frames)+pad-1-index)/pad);
                else value = work.onset[static_cast<size_t>(index)];
                ac.real[i] = static_cast<float>(value*work.ac_window[i]);
            }
            ac.plan->forward({fft_size},{sizeof(float)},{sizeof(std::complex<float>)},0,
                             ac.real.data(),ac.spectrum.data(),1.0f,1);
            for (auto & value : ac.spectrum) value = {std::norm(value),0};
            ac.plan->inverse({fft_size},{sizeof(std::complex<float>)},{sizeof(float)},0,
                             ac.spectrum.data(),ac.real.data(),1.0f/static_cast<float>(fft_size),1);
            // tempogram norm=inf, per column; then tempo aggregates across time.
            double norm = 0;
            for (size_t lag = 0; lag < win_length; ++lag)
                norm = std::max(norm,std::abs(static_cast<double>(ac.real[lag])));
            if (norm < std::numeric_limits<double>::min()) norm = 1;
            for (size_t lag = 0; lag < win_length; ++lag)
                work.mean_tempogram[lag] += ac.real[lag]/norm;
        }
        for (auto & value : work.mean_tempogram) value /= frames;
        // Default log-normal prior: start_bpm=120, std_bpm=1, max_tempo=320.
        // argmax is stable on ties (lowest lag first), including the excluded lags.
        size_t max_idx = 0;
        for (size_t lag = 1; lag < win_length; ++lag) {
            if (60.0*source_rate/(hop_length*lag) < 320) { max_idx = lag; break; }
        }
        size_t best_lag = 0;
        double best_score = -std::numeric_limits<double>::infinity();
        for (size_t lag = max_idx; lag < win_length; ++lag) {
            const double bpm = lag ? 60.0*source_rate/(hop_length*lag) : std::numeric_limits<double>::infinity();
            const double deviation = std::log2(bpm)-std::log2(120.0);
            const double score = std::log1p(1e6*work.mean_tempogram[lag])-0.5*deviation*deviation;
            if (score > best_score) { best_score = score; best_lag = lag; }
        }
        const double bpm = best_lag ? 60.0*source_rate/(hop_length*best_lag)
                                   : std::numeric_limits<double>::infinity();
        if (!std::isfinite(bpm) || bpm <= 0)
            throw std::runtime_error("Could not determine a positive BPM; provide bpm explicitly");

        // beat.py __beat_tracker: sample std(ddof=1), then static-tempo local
        // Gaussian scoring, tightness=100 DP, last-beat selection and trim=True.
        float mean = 0;
        for (float value : work.onset) mean += value;
        mean /= static_cast<float>(frames);
        float variance = 0;
        for (float value : work.onset) variance += (value-mean)*(value-mean);
        const float stddev = std::sqrt(variance/static_cast<float>(frames-1));
        work.normalized_onset.resize(frames);
        for (size_t i = 0; i < frames; ++i)
            work.normalized_onset[i] = work.onset[i]/(stddev+std::numeric_limits<float>::min());
        const auto round_even = [](double value) -> int64_t {
            const double lower = std::floor(value);
            return static_cast<int64_t>(lower+(value-lower > 0.5 ||
                (value-lower == 0.5 && std::fmod(lower,2.0) != 0)));
        };
        const int64_t period = round_even((static_cast<double>(source_rate)/hop_length)*60.0/bpm);
        const int64_t kernel = 2*period+1;
        work.gaussian.resize(static_cast<size_t>(kernel));
        for (int64_t k = 0; k < kernel; ++k) {
            const double position = (k-period)*32.0/period;
            work.gaussian[static_cast<size_t>(k)] = std::exp(-0.5*position*position);
        }
        work.local_score.assign(frames,0.0);
        for (int64_t i = 0; i < static_cast<int64_t>(frames); ++i) {
            // Preserve installed __beat_local_score's exclusive upper bound.
            for (int64_t k = std::max<int64_t>(0,i+kernel/2-static_cast<int64_t>(frames)+1);
                 k < std::min(i+kernel/2,kernel); ++k)
                work.local_score[i] += work.gaussian[k]*work.normalized_onset[i+kernel/2-k];
        }
        work.backlink.resize(frames);
        work.cumulative_score.resize(frames);
        const double first_threshold = 0.01*(*std::max_element(work.local_score.begin(),work.local_score.end()));
        bool first_beat = true;
        for (int64_t i = 0; i < static_cast<int64_t>(frames); ++i) {
            double best = -std::numeric_limits<double>::infinity();
            int32_t location = -1;
            for (int64_t loc = i-round_even(period/2.0); loc >= i-2*period; --loc) {
                if (loc < 0) break;
                const double delta = std::log(static_cast<double>(i-loc))-std::log(static_cast<double>(period));
                const double score = work.cumulative_score[loc]-100.0*delta*delta;
                if (score > best) { best = score; location = static_cast<int32_t>(loc); }
            }
            work.cumulative_score[i] = work.local_score[i]+(location >= 0 ? best : 0);
            if (first_beat && work.local_score[i] < first_threshold) work.backlink[i] = -1;
            else { work.backlink[i] = location; first_beat = false; }
        }
        work.maxima.clear();
        for (size_t i = 1; i < frames; ++i) {
            if (work.cumulative_score[i] > work.cumulative_score[i-1] &&
                (i+1 == frames || work.cumulative_score[i] >= work.cumulative_score[i+1]))
                work.maxima.push_back(work.cumulative_score[i]);
        }
        std::sort(work.maxima.begin(),work.maxima.end());
        const size_t middle = work.maxima.size()/2;
        const double tail_threshold = work.maxima.empty() ? 0 : 0.5*(work.maxima.size()%2
            ? work.maxima[middle] : (work.maxima[middle-1]+work.maxima[middle])*0.5);
        int64_t tail = static_cast<int64_t>(frames)-1;
        for (size_t i = frames; i-- > 1;) {
            if (work.cumulative_score[i] > work.cumulative_score[i-1] &&
                (i+1 == frames || work.cumulative_score[i] >= work.cumulative_score[i+1]) &&
                work.cumulative_score[i] >= tail_threshold) { tail = static_cast<int64_t>(i); break; }
        }
        work.beats.assign(frames,0);
        for (int64_t i = tail; i >= 0; i = work.backlink[i]) work.beats[i] = 1;
        work.selected_score.clear();
        for (size_t i = 0; i < frames; ++i)
            if (work.beats[i]) work.selected_score.push_back(work.local_score[i]);
        // __trim_beats uses np.hanning(5), full convolution, then the exact
        // [2:len(localscore)+2] slice (which can include two trailing values).
        constexpr std::array<double,5> trim_window{0,0.5,1,0.5,0};
        const size_t smooth_size = std::min(frames,work.selected_score.size()+2);
        double sum_squares = 0;
        for (size_t i = 0; i < smooth_size; ++i) {
            double smooth = 0;
            for (size_t k = 0; k < trim_window.size(); ++k) {
                const int64_t index = static_cast<int64_t>(i+2)-static_cast<int64_t>(k);
                if (index >= 0 && static_cast<size_t>(index) < work.selected_score.size())
                    smooth += trim_window[k]*work.selected_score[index];
            }
            sum_squares += smooth*smooth;
        }
        const double trim_threshold = 0.5*std::sqrt(sum_squares/smooth_size);
        size_t first = 0;
        while (first < frames && work.local_score[first] <= trim_threshold) work.beats[first++] = 0;
        if (first == frames) throw std::runtime_error("Beat trimming found no surviving local score");
        size_t last = frames;
        while (last > first && work.local_score[last-1] <= trim_threshold) work.beats[--last] = 0;
        work.beat_frames.clear();
        for (size_t i = 0; i < frames; ++i)
            if (work.beats[i]) work.beat_frames.push_back(static_cast<int32_t>(i));
        return bpm;
    }

    // librosa.resample(..., res_type="soxr_hq", fix=True), never linear fallback.
    std::vector<float> resample(const std::vector<float> & input, int source, int target, bool scale) {
        audio::SoxrResampleOptions options;
        options.profile = audio::SoxrResampleProfile::ExplicitFloat32Runtime;
        options.output_length_policy = audio::SoxrOutputLengthPolicy::ExactExpected;
        options.output_padding = 64;
        options.require_full_input = true;
        options.reject_empty_output = true;
        options.warning_context = "mulacover.chord.frontend";
        options.fallback_description = "no substitute is permitted; frontend will fail";
        auto result = audio::try_resample_mono_soxr(input,source,target,options);
        if (!result) throw std::runtime_error("MuLaCover Chord CQT requires working soxr_hq resampling");
        // Match librosa's ceil(n * ratio) evaluation order, including source-rate conversion.
        result->resize(static_cast<size_t>(std::ceil(input.size()*(static_cast<double>(target)/source))),0.0f);
        if (scale) {
            const double divisor = std::sqrt(static_cast<double>(target)/source);
            for (auto & value : *result) value = static_cast<float>(value/divisor);
        }
        return std::move(*result);
    }

    int estimate_tuning(const std::vector<float> & y) {
        // pitch.py: piptrack n_fft=2048, hop=512, periodic Hann, constant center padding.
        constexpr size_t n_fft = 2048;
        auto & fft = fft_.try_emplace(n_fft,n_fft).first->second;
        magnitude_frame_.resize(n_fft/2+1);
        pitches_.clear();
        magnitudes_.clear();
        const size_t frames = 1+y.size()/hop_length;
        for (size_t t = 0; t < frames; ++t) {
            for (size_t i = 0; i < n_fft; ++i) {
                const int64_t index = static_cast<int64_t>(t*hop_length+i)-static_cast<int64_t>(n_fft/2);
                fft.real[i] = index >= 0 && static_cast<size_t>(index) < y.size()
                    ? static_cast<float>(y[index]*fft.hann[i]) : 0.0f;
            }
            fft.plan->forward({n_fft},{sizeof(float)},{sizeof(std::complex<float>)},0,
                              fft.real.data(),fft.spectrum.data(),1.0f,1);
            float maximum = 0;
            for (size_t i = 0; i < magnitude_frame_.size(); ++i) {
                magnitude_frame_[i] = std::abs(fft.spectrum[i]);
                maximum = std::max(maximum,magnitude_frame_[i]);
            }
            const float threshold = 0.1f*maximum;
            for (size_t i = 1; i+1 < magnitude_frame_.size(); ++i) {
                const double frequency = i*sample_rate_hz/n_fft;
                if (frequency < 150 || frequency >= 4000) continue;
                const float center = magnitude_frame_[i];
                const float left = magnitude_frame_[i-1], right = magnitude_frame_[i+1];
                // localmax compares strictly to the previous bin, non-strictly to the next.
                if (!(center > threshold && center > left && center >= right)) continue;
                const double a = static_cast<double>(right+left)-2.0*center;
                const double b = static_cast<double>(right-left)/2.0;
                const float shift = std::abs(b) >= std::abs(a) ? 0.0f : static_cast<float>(-b/a);
                const float gradient = (right-left)*0.5f;
                const float skew = 0.5f*gradient*shift;
                pitches_.push_back(static_cast<float>((i+static_cast<double>(shift))*sample_rate_hz/n_fft));
                magnitudes_.push_back(center+skew);
            }
        }
        if (pitches_.empty()) {
            // pitch_tuning explicitly returns zero for an empty frequency set.
            debug::trace_log_scalar("mulacover.chord.frontend.tuning_empty",true);
            return 50;
        }
        sorted_magnitudes_ = magnitudes_;
        std::sort(sorted_magnitudes_.begin(),sorted_magnitudes_.end());
        const size_t middle = sorted_magnitudes_.size()/2;
        const float median = sorted_magnitudes_.size()%2 ? sorted_magnitudes_[middle]
            : (sorted_magnitudes_[middle-1]+sorted_magnitudes_[middle])*0.5f;
        std::array<size_t,100> histogram{};
        for (size_t i = 0; i < pitches_.size(); ++i) {
            if (magnitudes_[i] < median) continue;
            // hz_to_octs operates on the F32 piptrack output with A440/16=27.5.
            const float octaves = std::log2(pitches_[i]/27.5f);
            const float pitch = bins_per_octave*octaves;
            float residual = pitch-std::floor(pitch);
            if (residual >= 0.5f) residual -= 1.0f;
            // np.histogram uses linspace(-.5,.5,101), returning its LEFT edge.
            int bin = static_cast<int>(std::floor((static_cast<double>(residual)+0.5)*100));
            // Correct bin-edge rounding against the explicit double linspace edges.
            if (bin > 0 && residual < -0.5+0.01*bin) --bin;
            if (bin < 99 && residual >= -0.5+0.01*(bin+1)) ++bin;
            ++histogram.at(static_cast<size_t>(bin));
        }
        return static_cast<int>(std::max_element(histogram.begin(),histogram.end())-histogram.begin());
    }

    Band build_band(const std::vector<double> & freqs, const std::vector<double> & alpha,
                    double sr, double base_sr, int first, size_t hop, bool pseudo) {
        Band band;
        band.first = first;
        band.pseudo = pseudo;
        band.hop = hop;
        std::vector<double> lengths(freqs.size());
        for (size_t i = 0; i < freqs.size(); ++i) lengths[i] = (1.0/alpha[i])*sr/freqs[i];
        const size_t base_fft = static_cast<size_t>(std::pow(2.0,std::ceil(std::log2(
            *std::max_element(lengths.begin(),lengths.end())))));
        band.fft_size = pseudo ? std::max(base_fft,2*hop_length) : base_fft;
        auto & fft = fft_.try_emplace(band.fft_size,band.fft_size).first->second;
        band.filters.resize(freqs.size());
        band.lengths.resize(freqs.size());
        std::vector<float> magnitudes(band.fft_size/2+1), sorted(magnitudes.size());
        std::vector<std::complex<float>> coefficients(magnitudes.size());
        for (size_t row = 0; row < freqs.size(); ++row) {
            const double length = lengths[row];
            // Exactly np.arange(-ilen // 2, ilen // 2): not truncation of ilen.
            const int64_t begin = static_cast<int64_t>(std::floor(-length/2));
            const int64_t end = static_cast<int64_t>(std::floor(length/2));
            const size_t count = static_cast<size_t>(end-begin);
            const size_t pad = (base_fft-count)/2;
            auto inserted = wavelet_windows_.try_emplace(count);
            auto & window = inserted.first->second;
            if (inserted.second) {
                window.resize(count);
                for (size_t j = 0; j < count; ++j) window[j] = 0.5+0.5*std::cos(-pi+j*(2*pi/count));
            }
            double norm = 0;
            wavelet_.resize(count);
            for (size_t j = 0; j < count; ++j) {
                const double angle = (begin+static_cast<double>(j))*2*pi*freqs[row]/sr;
                wavelet_[j] = std::complex<double>(std::cos(angle),std::sin(angle))*window[j];
                norm += std::abs(wavelet_[j]);
            }
            std::fill(fft.real.begin(),fft.real.end(),0.0f);
            std::fill(fft.imaginary.begin(),fft.imaginary.end(),0.0f);
            for (size_t j = 0; j < count; ++j) {
                // wavelet casts normalized filters to complex64 before length/n_fft scaling.
                const float real = static_cast<float>(wavelet_[j].real()/norm);
                const float imag = static_cast<float>(wavelet_[j].imag()/norm);
                fft.real[pad+j] = static_cast<float>(real*(length/band.fft_size));
                fft.imaginary[pad+j] = static_cast<float>(imag*(length/band.fft_size));
            }
            // Complex FFT via two existing real plans: FFT(real)+i*FFT(imag).
            // Pseudo's additional FFT padding is at the right, AFTER wavelet centering.
            fft.plan->forward({band.fft_size},{sizeof(float)},{sizeof(std::complex<float>)},0,
                              fft.real.data(),fft.spectrum.data(),1.0f,1);
            fft.plan->forward({band.fft_size},{sizeof(float)},{sizeof(std::complex<float>)},0,
                              fft.imaginary.data(),fft.imaginary_spectrum.data(),1.0f,1);
            float norm_frequency = 0;
            for (size_t k = 0; k < coefficients.size(); ++k) {
                coefficients[k] = fft.spectrum[k]+std::complex<float>(
                    -fft.imaginary_spectrum[k].imag(),fft.imaginary_spectrum[k].real());
                magnitudes[k] = std::abs(coefficients[k]);
                norm_frequency += magnitudes[k];
            }
            sorted = magnitudes;
            std::sort(sorted.begin(),sorted.end());
            float cumulative = 0;
            float cutoff = sorted.front();
            for (float value : sorted) {
                cumulative += value/norm_frequency;
                if (cumulative >= 0.01f) { cutoff = value; break; }
            }
            const double compensation = std::sqrt(base_sr/sr);
            for (size_t k = 0; k < coefficients.size(); ++k) {
                if (magnitudes[k] < cutoff) continue;
                auto value = coefficients[k];
                if (pseudo) value = {magnitudes[k],0};
                else value = {static_cast<float>(value.real()*compensation),
                              static_cast<float>(value.imag()*compensation)};
                band.filters[row].push_back({k,value});
            }
            band.lengths[row] = (1.0/alpha[row])*base_sr/freqs[row];
        }
        return band;
    }

    FilterBank & filter_bank(int tuning_bin) {
        const auto found = banks_.find(tuning_bin);
        if (found != banks_.end()) return found->second;
        const double tuning = -0.5+0.01*tuning_bin;
        // F#0 is MIDI 18. librosa.note_to_hz = 440*2**((midi-69)/12).
        const double fmin = 440*std::pow(2.0,(18.0-69)/12)*std::pow(2.0,tuning/bins_per_octave);
        std::vector<double> frequencies(bins);
        for (int i = 0; i < bins; ++i) frequencies[i] = fmin*std::pow(2.0,static_cast<double>(i)/bins_per_octave);
        const auto alpha = relative_bandwidth(frequencies);
        int full = 0;
        for (int i = 0; i < bins; ++i) {
            const double length = (1.0/alpha[i])*sample_rate_hz/frequencies[i];
            if (std::pow(2.0,std::ceil(std::log2(length))) >= 2*hop_length) ++full;
        }
        FilterBank bank;
        if (full < bins) {
            // pseudo_cqt independently constructs its frequencies and bandwidths.
            std::vector<double> pseudo(bins-full);
            for (size_t i = 0; i < pseudo.size(); ++i)
                pseudo[i] = frequencies[full]*std::pow(2.0,static_cast<double>(i)/bins_per_octave);
            bank.bands.push_back(build_band(pseudo,relative_bandwidth(pseudo),sample_rate_hz,
                                           sample_rate_hz,full,hop_length,true));
        }
        if (full > 0) {
            // vqt uses interval_frequencies, tiling one octave rather than a single power.
            frequencies.resize(full);
            for (int i = 0; i < full; ++i) frequencies[i] = fmin*std::pow(2.0,i/bins_per_octave)
                *std::pow(2.0,static_cast<double>(i%bins_per_octave)/bins_per_octave);
            const auto full_alpha = relative_bandwidth(frequencies);
            double cutoff = 0;
            for (int i = 0; i < full; ++i)
                cutoff = std::max(cutoff,frequencies[i]*(1+0.5*1.50018310546875*full_alpha[i]));
            if (cutoff > sample_rate_hz/2) throw std::runtime_error("Chord CQT basis exceeds Nyquist");
            const int octave_count = (full+bins_per_octave-1)/bins_per_octave;
            bank.early_downsamples = std::min(std::max(0,static_cast<int>(std::ceil(
                std::log2((sample_rate_hz/2)/cutoff)))-2),std::max(0,9-octave_count+1));
            const double base_sr = sample_rate_hz/std::pow(2.0,bank.early_downsamples);
            double sr = base_sr;
            size_t hop = hop_length >> bank.early_downsamples;
            for (int end = full; end > 0; end -= bins_per_octave) {
                const int first = std::max(0,end-bins_per_octave);
                bank.bands.push_back(build_band({frequencies.begin()+first,frequencies.begin()+end},
                    {full_alpha.begin()+first,full_alpha.begin()+end},sr,base_sr,first,hop,false));
                if (hop%2 == 0) { hop /= 2; sr /= 2; }
            }
        }
        return banks_.emplace(tuning_bin,std::move(bank)).first->second;
    }

    void response(const std::vector<float> & y, const Band & band, Response & out) {
        auto & fft = fft_.try_emplace(band.fft_size,band.fft_size).first->second;
        out.frames = 1+y.size()/band.hop;
        out.values.resize(out.frames*band.filters.size());
        for (size_t t = 0; t < out.frames; ++t) {
            for (size_t i = 0; i < band.fft_size; ++i) {
                const int64_t index = static_cast<int64_t>(t*band.hop+i)-static_cast<int64_t>(band.fft_size/2);
                const float value = index >= 0 && static_cast<size_t>(index) < y.size() ? y[index] : 0;
                fft.real[i] = band.pseudo ? static_cast<float>(value*fft.hann[i]) : value;
            }
            fft.plan->forward({band.fft_size},{sizeof(float)},{sizeof(std::complex<float>)},0,
                              fft.real.data(),fft.spectrum.data(),1.0f,1);
            if (band.pseudo) {
                for (auto & value : fft.spectrum) value = {std::abs(value),0};
            }
            for (size_t row = 0; row < band.filters.size(); ++row) {
                std::complex<float> sum{};
                // CSR projection is not conjugated; sparse entries stay frequency-sorted.
                for (const auto & coefficient : band.filters[row])
                    sum += coefficient.value*fft.spectrum[coefficient.bin];
                const double divisor = std::sqrt(band.pseudo ? static_cast<double>(band.fft_size) : band.lengths[row]);
                const std::complex<float> normalized{static_cast<float>(sum.real()/divisor),
                                                      static_cast<float>(sum.imag()/divisor)};
                out.values[t*band.filters.size()+row] = std::abs(normalized);
            }
        }
    }

public:
    const ChordFrontendOutput & forward(const std::vector<float> & samples,
                                        int sample_rate, int channels, double bpm) {
        const auto started = std::chrono::steady_clock::now();
        if (sample_rate <= 0 || channels <= 0 || samples.empty() || samples.size()%channels)
            throw std::invalid_argument("Chord frontend requires nonempty interleaved PCM with positive rate/channels");
        if (!std::isfinite(bpm) || bpm < 0)
            throw std::invalid_argument("Chord frontend BPM must be zero (automatic) or positive and finite");
        mono_.resize(samples.size()/channels);
        for (size_t i = 0; i < mono_.size(); ++i) {
            float sum = 0;
            for (int channel = 0; channel < channels; ++channel) {
                const float value = samples[i*channels+channel];
                if (!std::isfinite(value)) throw std::invalid_argument("Chord frontend PCM contains nonfinite samples");
                sum += value;
            }
            mono_[i] = sum/static_cast<float>(channels);
        }
        // Upstream estimates tempo at the decoded source rate, before harmony.py
        // independently loads/resamples the audio for its 22050-Hz CQT.
        if (bpm == 0) {
            bpm = estimate_bpm(mono_,sample_rate);
            if (!std::isfinite(bpm) || bpm <= 0)
                throw std::runtime_error("Could not determine a positive BPM; provide bpm explicitly");
        }
        if (sample_rate != static_cast<int>(sample_rate_hz))
            mono_ = resample(mono_,sample_rate,static_cast<int>(sample_rate_hz),false);
        const int tuning_bin = estimate_tuning(mono_);
        auto & bank = filter_bank(tuning_bin);
        responses_.resize(bank.bands.size());
        size_t octave = 0;
        bool full_started = false;
        size_t frames = std::numeric_limits<size_t>::max();
        for (size_t i = 0; i < bank.bands.size(); ++i) {
            const auto & band = bank.bands[i];
            const std::vector<float> * y = &mono_;
            if (!band.pseudo) {
                if (!full_started) {
                    full_started = true;
                    if (bank.early_downsamples) {
                        const int factor = 1 << bank.early_downsamples;
                        if (mono_.size() < static_cast<size_t>(factor))
                            throw std::invalid_argument("Audio is too short for Chord CQT early downsampling");
                        octaves_[0] = resample(mono_,factor,1,true);
                    } else octaves_[0] = mono_;
                } else if (bank.bands[i-1].hop%2 == 0) {
                    octaves_.at(octave+1) = resample(octaves_[octave],2,1,true);
                    ++octave;
                }
                y = &octaves_[octave];
            }
            response(*y,band,responses_[i]);
            frames = std::min(frames,responses_[i].frames);
        }
        // __trim_stack trims every octave and the hybrid branches to the shortest time axis.
        full_cqt_.resize(frames*bins);
        for (size_t i = 0; i < bank.bands.size(); ++i) {
            const auto & band = bank.bands[i];
            for (size_t t = 0; t < frames; ++t)
                std::copy_n(responses_[i].values.begin()+t*band.filters.size(),band.filters.size(),
                            full_cqt_.begin()+t*bins+band.first);
        }
        output_.frames = static_cast<int64_t>(frames);
        output_.tuning = -0.5+0.01*tuning_bin;
        output_.bpm = bpm;
        output_.cqt.resize(frames*252);
        for (size_t t = 0; t < frames; ++t)
            std::copy_n(full_cqt_.begin()+t*bins+18,252,output_.cqt.begin()+t*252);
        debug::trace_log_scalar("mulacover.chord.frontend.bpm",output_.bpm);
        debug::trace_log_scalar("mulacover.chord.frontend.tuning",output_.tuning);
        debug::timing_log_scalar("mulacover.chord.frontend.wall_ms",
            std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-started).count());
        return output_;
    }
};

ChordFrontend::ChordFrontend() : impl_(std::make_unique<Impl>()) {}
ChordFrontend::~ChordFrontend() = default;
const ChordFrontendOutput & ChordFrontend::forward(const std::vector<float> & samples,
                                                   int sample_rate, int channels, double bpm) {
    return impl_->forward(samples,sample_rate,channels,bpm);
}

}  // namespace engine::models::mulacover
