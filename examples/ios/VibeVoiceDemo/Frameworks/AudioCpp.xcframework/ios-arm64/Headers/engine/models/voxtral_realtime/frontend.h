#pragma once

#include "engine/framework/audio/dsp.h"
#include "engine/models/voxtral_realtime/assets.h"
#include "engine/models/voxtral_realtime/types.h"

#include <cstdint>
#include <memory>
#include <vector>

namespace engine::models::voxtral_realtime {

struct VoxtralRealtimeFrontendStreamState {
    std::vector<float> cached_magnitude_frame;
    int64_t cached_freq_bins = 0;
    bool cached_frame_ready = false;
};

class VoxtralRealtimeFrontend {
public:
    explicit VoxtralRealtimeFrontend(std::shared_ptr<const VoxtralRealtimeAssets> assets);

    VoxtralRealtimeFeatures extract(const runtime::AudioBuffer & audio, bool first_chunk) const;
    // A steady chunk carries `steady_tokens` audio tokens of 80 ms each; batching them into one
    // extraction yields 8 * steady_tokens feature frames from a single STFT pass.
    VoxtralRealtimeFeatures extract_stream_chunk(
        const runtime::AudioBuffer & audio,
        bool first_chunk,
        VoxtralRealtimeFrontendStreamState & state,
        int64_t steady_tokens = 1) const;

    int64_t first_stream_chunk_samples() const;
    int64_t steady_stream_chunk_samples(int64_t steady_tokens = 1) const;
    int64_t first_stream_chunk_advance_samples() const;
    int64_t steady_stream_chunk_advance_samples(int64_t steady_tokens = 1) const;

private:
    std::shared_ptr<const VoxtralRealtimeAssets> assets_;
    engine::audio::SparseMelFilterbank mel_filterbank_;
};

}  // namespace engine::models::voxtral_realtime
