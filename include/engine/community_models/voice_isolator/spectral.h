#pragma once

#include "engine/community_models/voice_isolator/config.h"

#include <complex>
#include <cstdint>
#include <memory>
#include <vector>

// The isolator's signal processing around the network: framing, band packing,
// the spectral correction its decoder encodes and synthesis. Spectra here use
// the framework STFT layout, bin-major [bins][frames][2] (re, im).
namespace engine::audio {
class RealFFTPlan;
}

namespace engine::models::voice_isolator {

// Periodic Hann window of n_fft samples (torch.hann_window).
const std::vector<float> & analysis_window(const Config & config);

// torch.stft(center=True) of a whole clip: reflect padding, or zeros for a clip
// no longer than half a window. Sets `frames`.
std::vector<float> clip_spectra(const Config & config, const std::vector<float> & samples, int64_t & frames);

// torch.istft(center=True, length=samples) of bin-major spectra.
std::vector<float> clip_waveform(
    const Config & config, const std::vector<float> & spectra, int64_t frames, int64_t samples);

// Spectra of stream frames [first, first + count): frame f covers samples
// [f * hop - n_fft / 2, f * hop + n_fft / 2), zeros where none is held. The held
// samples are [held_start, held_start + held_count).
std::vector<float> stream_spectra(
    const Config & config,
    const float * held,
    int64_t held_start,
    int64_t held_count,
    int64_t first,
    int64_t count);

// The encoder's input: band-major [bands][frames][2 * max_band_bins], each band's
// bins as interleaved (re, im) and zero-padded to the widest band.
void pack_bands(const BandLayout & layout, const float * spectra, int64_t frames, float * packed);

// The complex change the decoder encodes for spectra frame `frame`, from decoder
// frame `raw_frame` of raw [bands][raw_frames][4 * max_band_bins]: the spectrum
// times a bounded complex mask delta plus a residual bounded by the bin's
// magnitude. Writes that frame of `change`, laid out like `spectra`.
void spectral_change(
    const BandLayout & layout,
    float bound,
    const float * raw,
    int64_t raw_frames,
    int64_t raw_frame,
    const float * spectra,
    int64_t frames,
    int64_t frame,
    float * change);

// Overlap-adds a stream's synthesized frames and releases the change of each
// sample once every frame that touches it has been added.
class StreamSynthesizer {
public:
    explicit StreamSynthesizer(const Config & config);
    ~StreamSynthesizer();

    // Adds frames [first, first + count) of bin-major `change` [bins][count][2].
    void add(int64_t first, int64_t count, const float * change);
    // The change of samples [start, stop); forgets everything before `stop`.
    void take(int64_t start, int64_t stop, float * out);
    void reset();

private:
    int64_t n_fft_ = 0;
    int64_t hop_ = 0;
    const std::vector<float> * window_ = nullptr;
    std::shared_ptr<audio::RealFFTPlan> plan_;
    std::vector<float> sum_;
    std::vector<float> envelope_;
    std::vector<float> frames_;
    int64_t base_ = 0;
};

// Offline render windows: full windows of `chunk` samples overlapping by
// `overlap` with linear crossfades, the last one ending at the clip's end and
// fading in where the window before it fades out.
// Each window's weights are built when it is rendered, so a long clip never
// holds a chunk of weights per window.
struct RenderWindow {
    int64_t start = 0;
    int64_t length = 0;
    int64_t overlap = 0;
    // Where it starts fading in, silent before (-1: no fade-in); whether its last
    // `overlap` samples fade out.
    int64_t fade_in = -1;
    bool fade_out = false;

    std::vector<float> weights() const;
};

std::vector<RenderWindow> plan_render_windows(int64_t samples, int64_t chunk, int64_t overlap);

}  // namespace engine::models::voice_isolator
