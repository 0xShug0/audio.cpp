#include "streaming_tempo.h"
#include "engine/framework/io/dynamic_library.h"

#include <cmath>
#include <limits>
#include <stdexcept>

namespace minitts::app {

class StreamingSpeechTempo::Impl {
public:
    using Create = void * (*)(int, int);
    using Destroy = void (*)(void *);
    using SetSpeed = void (*)(void *, float);
    using Write = int (*)(void *, const float *, int);
    using Read = int (*)(void *, float *, int);
    using Flush = int (*)(void *);
    using Available = int (*)(void *);

    Impl(int sample_rate, float speed) {
        if (sample_rate <= 0 || !std::isfinite(speed) || speed < 0.5F || speed > 2.0F)
            throw std::invalid_argument("speech tempo requires a positive sample rate and speed in [0.5, 2]");
        if (speed == 1.0F) return;
        library = engine::io::open_dynamic_library({"libsonic.so.0", "libsonic.so", "libsonic.dylib", "sonic.dll", "libsonic.dll"});
        if (!library) throw std::runtime_error("speaking_speed other than 1 requires libsonic");
        const auto create = reinterpret_cast<Create>(engine::io::dynamic_library_symbol(library, "sonicCreateStream"));
        destroy = reinterpret_cast<Destroy>(engine::io::dynamic_library_symbol(library, "sonicDestroyStream"));
        const auto set_speed = reinterpret_cast<SetSpeed>(engine::io::dynamic_library_symbol(library, "sonicSetSpeed"));
        write = reinterpret_cast<Write>(engine::io::dynamic_library_symbol(library, "sonicWriteFloatToStream"));
        read = reinterpret_cast<Read>(engine::io::dynamic_library_symbol(library, "sonicReadFloatFromStream"));
        flush = reinterpret_cast<Flush>(engine::io::dynamic_library_symbol(library, "sonicFlushStream"));
        available = reinterpret_cast<Available>(engine::io::dynamic_library_symbol(library, "sonicSamplesAvailable"));
        if (!create || !destroy || !set_speed || !write || !read || !flush || !available) {
            engine::io::close_dynamic_library(library);
            throw std::runtime_error("libsonic is missing streaming entry points");
        }
        stream = create(sample_rate, 1);
        if (!stream) {
            engine::io::close_dynamic_library(library);
            throw std::runtime_error("sonicCreateStream failed");
        }
        set_speed(stream, speed);
    }
    ~Impl() {
        if (stream) destroy(stream);
        engine::io::close_dynamic_library(library);
    }
    engine::io::DynamicLibraryHandle library = nullptr;
    void * stream = nullptr;
    Destroy destroy = nullptr;
    Write write = nullptr;
    Read read = nullptr;
    Flush flush = nullptr;
    Available available = nullptr;
    bool finished = false;
};

StreamingSpeechTempo::StreamingSpeechTempo(int sample_rate, float speed) : impl_(std::make_unique<Impl>(sample_rate, speed)) {}
StreamingSpeechTempo::~StreamingSpeechTempo() = default;

std::vector<float> StreamingSpeechTempo::process(const std::vector<float> & samples, bool final) {
    if (impl_->finished) throw std::runtime_error("speech tempo stream is already finalized");
    if (!impl_->stream) { impl_->finished = final; return samples; }
    if (samples.size() > static_cast<size_t>(std::numeric_limits<int>::max()))
        throw std::invalid_argument("speech tempo input chunk exceeds libsonic's sample-count limit");
    if (!samples.empty() && !impl_->write(impl_->stream, samples.data(), static_cast<int>(samples.size())))
        throw std::runtime_error("sonicWriteFloatToStream failed");
    if (final && !impl_->flush(impl_->stream)) throw std::runtime_error("sonicFlushStream failed");
    const int count = impl_->available(impl_->stream);
    if (count < 0) throw std::runtime_error("sonicSamplesAvailable returned an invalid count");
    std::vector<float> output(static_cast<size_t>(count));
    if (count && impl_->read(impl_->stream, output.data(), count) != count)
        throw std::runtime_error("sonicReadFloatFromStream returned an incomplete block");
    impl_->finished = final;
    return output;
}

} // namespace minitts::app
