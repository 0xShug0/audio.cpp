#include "engine/framework/vision/media/video.h"

#if !defined(__EMSCRIPTEN__)
#include "engine/framework/io/dynamic_library.h"

#include <condition_variable>
#include <deque>
#include <future>
#include <mutex>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace engine::io {
namespace {
constexpr double kDefaultFrameRate = 30.0;
constexpr int64_t kNoPts = std::numeric_limits<int64_t>::min();

struct AVRational {
    int num = 0;
    int den = 1;
};

struct AVCodec;
struct AVCodecContext;
struct AVCodecParameters;
struct AVDictionary;
struct AVInputFormat;
struct AVOutputFormat;
struct AVIOContext;
struct SwsContext;
struct SwsFilter;

struct AVPacket {
    void * buf = nullptr;
    int64_t pts = kNoPts;
    int64_t dts = kNoPts;
    uint8_t * data = nullptr;
    int size = 0;
    int stream_index = -1;
};

struct AVFrame {
    uint8_t * data[8] = {};
    int linesize[8] = {};
    uint8_t ** extended_data = nullptr;
    int width = 0;
    int height = 0;
    int nb_samples = 0;
    int format = -1;
    int key_frame = 0;
    int pict_type = 0;
    AVRational sample_aspect_ratio;
    int64_t pts = kNoPts;
};

struct AVStreamModern {
    const void * av_class = nullptr;
    int index = 0;
    int id = 0;
    AVCodecParameters * codecpar = nullptr;
    void * priv_data = nullptr;
    AVRational time_base;
};

struct AVFormatContextModern {
    const void * av_class = nullptr;
    AVInputFormat * iformat = nullptr;
    AVOutputFormat * oformat = nullptr;
    void * priv_data = nullptr;
    AVIOContext * pb = nullptr;
    int ctx_flags = 0;
    unsigned int nb_streams = 0;
    AVStreamModern ** streams = nullptr;
};

class LibAvApi {
    using Library = std::unique_ptr<std::remove_pointer_t<DynamicLibraryHandle>, decltype(&close_dynamic_library)>;
public:
    LibAvApi()
        : avformat_(engine::io::open_dynamic_library({
#ifdef _WIN32
              "avformat-62.dll", "avformat-61.dll", "avformat-60.dll", "avformat-59.dll",
              "avformat.dll",
#elif __APPLE__
              "libavformat.62.dylib", "libavformat.61.dylib", "libavformat.60.dylib",
              "libavformat.59.dylib", "libavformat.dylib",
#else
              "libavformat.so.62", "libavformat.so.61", "libavformat.so.60",
              "libavformat.so.59", "libavformat.so",
#endif
          }), close_dynamic_library),
          avcodec_(engine::io::open_dynamic_library({
#ifdef _WIN32
              "avcodec-62.dll", "avcodec-61.dll", "avcodec-60.dll", "avcodec-59.dll",
              "avcodec.dll",
#elif __APPLE__
              "libavcodec.62.dylib", "libavcodec.61.dylib", "libavcodec.60.dylib",
              "libavcodec.59.dylib", "libavcodec.dylib",
#else
              "libavcodec.so.62", "libavcodec.so.61", "libavcodec.so.60",
              "libavcodec.so.59", "libavcodec.so",
#endif
          }), close_dynamic_library),
          avutil_(engine::io::open_dynamic_library({
#ifdef _WIN32
              "avutil-60.dll", "avutil-59.dll", "avutil-58.dll", "avutil-57.dll",
              "avutil.dll",
#elif __APPLE__
              "libavutil.60.dylib", "libavutil.59.dylib", "libavutil.58.dylib",
              "libavutil.57.dylib", "libavutil.dylib",
#else
              "libavutil.so.60", "libavutil.so.59", "libavutil.so.58",
              "libavutil.so.57", "libavutil.so",
#endif
          }), close_dynamic_library),
          swscale_(engine::io::open_dynamic_library({
#ifdef _WIN32
              "swscale-9.dll", "swscale-8.dll", "swscale-7.dll", "swscale-6.dll",
              "swscale.dll",
#elif __APPLE__
              "libswscale.9.dylib", "libswscale.8.dylib", "libswscale.7.dylib",
              "libswscale.6.dylib", "libswscale.dylib",
#else
              "libswscale.so.9", "libswscale.so.8", "libswscale.so.7",
              "libswscale.so.6", "libswscale.so",
#endif
          }), close_dynamic_library) {
        if (!avformat_ || !avcodec_ || !avutil_ || !swscale_)
            throw std::runtime_error("Video decoding requires FFmpeg 6+ shared libraries: libavformat, libavcodec, libavutil and libswscale");

        avformat_version = symbol<unsigned (*)()>(avformat_.get(), "avformat_version");
        avformat_open_input = symbol<int (*)(AVFormatContextModern **, const char *, const AVInputFormat *, AVDictionary **)>(
            avformat_.get(), "avformat_open_input");
        avformat_find_stream_info = symbol<int (*)(AVFormatContextModern *, AVDictionary **)>(
            avformat_.get(), "avformat_find_stream_info");
        av_find_best_stream = symbol<int (*)(AVFormatContextModern *, int, int, int, const AVCodec **, int)>(
            avformat_.get(), "av_find_best_stream");
        av_read_frame = symbol<int (*)(AVFormatContextModern *, AVPacket *)>(avformat_.get(), "av_read_frame");
        avformat_close_input = symbol<void (*)(AVFormatContextModern **)>(avformat_.get(), "avformat_close_input");

        avcodec_alloc_context3 = symbol<AVCodecContext * (*)(const AVCodec *)>(avcodec_.get(), "avcodec_alloc_context3");
        avcodec_parameters_to_context = symbol<int (*)(AVCodecContext *, const AVCodecParameters *)>(
            avcodec_.get(), "avcodec_parameters_to_context");
        avcodec_open2 = symbol<int (*)(AVCodecContext *, const AVCodec *, AVDictionary **)>(avcodec_.get(), "avcodec_open2");
        avcodec_send_packet = symbol<int (*)(AVCodecContext *, const AVPacket *)>(avcodec_.get(), "avcodec_send_packet");
        avcodec_receive_frame = symbol<int (*)(AVCodecContext *, AVFrame *)>(avcodec_.get(), "avcodec_receive_frame");
        avcodec_free_context = symbol<void (*)(AVCodecContext **)>(avcodec_.get(), "avcodec_free_context");
        av_packet_alloc = symbol<AVPacket * (*)()>(avcodec_.get(), "av_packet_alloc");
        av_packet_free = symbol<void (*)(AVPacket **)>(avcodec_.get(), "av_packet_free");
        av_packet_unref = symbol<void (*)(AVPacket *)>(avcodec_.get(), "av_packet_unref");

        av_frame_alloc = symbol<AVFrame * (*)()>(avutil_.get(), "av_frame_alloc");
        av_frame_get_buffer = symbol<int (*)(AVFrame *, int)>(avutil_.get(), "av_frame_get_buffer");
        av_frame_free = symbol<void (*)(AVFrame **)>(avutil_.get(), "av_frame_free");
        av_frame_unref = symbol<void (*)(AVFrame *)>(avutil_.get(), "av_frame_unref");
        av_get_pix_fmt = symbol<int (*)(const char *)>(avutil_.get(), "av_get_pix_fmt");
        av_dict_set = symbol<int (*)(AVDictionary **, const char *, const char *, int)>(avutil_.get(), "av_dict_set");
        av_dict_free = symbol<void (*)(AVDictionary **)>(avutil_.get(), "av_dict_free");
        av_strerror = optional_symbol<int (*)(int, char *, size_t)>(avutil_.get(), "av_strerror");

        sws_getContext = symbol<SwsContext * (*)(
            int, int, int, int, int, int, int, SwsFilter *, SwsFilter *, const double *)>(
            swscale_.get(), "sws_getContext");
        sws_scale = symbol<int (*)(SwsContext *, const uint8_t * const [], const int [], int, int, uint8_t * const [], const int [])>(
            swscale_.get(), "sws_scale");
        sws_freeContext = symbol<void (*)(SwsContext *)>(swscale_.get(), "sws_freeContext");

        const unsigned version = avformat_version();
        const unsigned major = version >> 16U;
        if (major < 60U) {
            throw std::runtime_error(
                "Video decoding requires FFmpeg 6+ runtime libraries");
        }
    }


    unsigned (*avformat_version)() = nullptr;
    int (*avformat_open_input)(AVFormatContextModern **, const char *, const AVInputFormat *, AVDictionary **) = nullptr;
    int (*avformat_find_stream_info)(AVFormatContextModern *, AVDictionary **) = nullptr;
    int (*av_find_best_stream)(AVFormatContextModern *, int, int, int, const AVCodec **, int) = nullptr;
    int (*av_read_frame)(AVFormatContextModern *, AVPacket *) = nullptr;
    void (*avformat_close_input)(AVFormatContextModern **) = nullptr;

    AVCodecContext * (*avcodec_alloc_context3)(const AVCodec *) = nullptr;
    int (*avcodec_parameters_to_context)(AVCodecContext *, const AVCodecParameters *) = nullptr;
    int (*avcodec_open2)(AVCodecContext *, const AVCodec *, AVDictionary **) = nullptr;
    int (*avcodec_send_packet)(AVCodecContext *, const AVPacket *) = nullptr;
    int (*avcodec_receive_frame)(AVCodecContext *, AVFrame *) = nullptr;
    void (*avcodec_free_context)(AVCodecContext **) = nullptr;
    AVPacket * (*av_packet_alloc)() = nullptr;
    void (*av_packet_free)(AVPacket **) = nullptr;
    void (*av_packet_unref)(AVPacket *) = nullptr;

    AVFrame * (*av_frame_alloc)() = nullptr;
    int (*av_frame_get_buffer)(AVFrame *, int) = nullptr;
    void (*av_frame_free)(AVFrame **) = nullptr;
    void (*av_frame_unref)(AVFrame *) = nullptr;
    int (*av_get_pix_fmt)(const char *) = nullptr;
    int (*av_dict_set)(AVDictionary **, const char *, const char *, int) = nullptr;
    void (*av_dict_free)(AVDictionary **) = nullptr;
    int (*av_strerror)(int, char *, size_t) = nullptr;

    SwsContext * (*sws_getContext)(int, int, int, int, int, int, int, SwsFilter *, SwsFilter *, const double *) = nullptr;
    int (*sws_scale)(SwsContext *, const uint8_t * const [], const int [], int, int, uint8_t * const [], const int []) = nullptr;
    void (*sws_freeContext)(SwsContext *) = nullptr;

private:
    template <typename Fn>
    Fn symbol(engine::io::DynamicLibraryHandle library, const char * name) {
        auto * address = engine::io::dynamic_library_symbol(library, name);
        if (address == nullptr) {
            throw std::runtime_error(std::string("FFmpeg library is missing symbol ") + name);
        }
        return reinterpret_cast<Fn>(address);
    }

    template <typename Fn>
    Fn optional_symbol(engine::io::DynamicLibraryHandle library, const char * name) {
        return reinterpret_cast<Fn>(engine::io::dynamic_library_symbol(library, name));
    }

    Library avformat_{nullptr, close_dynamic_library};
    Library avcodec_{nullptr, close_dynamic_library};
    Library avutil_{nullptr, close_dynamic_library};
    Library swscale_{nullptr, close_dynamic_library};
};

class LibAvVideoDecoder {
public:
    LibAvVideoDecoder() {
        static const auto api = std::make_shared<const LibAvApi>();
        api_ = api;
    }

    ~LibAvVideoDecoder() {
        if (sws_context_ != nullptr) {
            api_->sws_freeContext(sws_context_);
            sws_context_ = nullptr;
        }
    }

    void decode(const std::filesystem::path & path, double max_duration_sec, const RgbVideoConsumer & consume) {
        AVFormatContextModern * format = nullptr;
        check(api_->avformat_open_input(&format, path.string().c_str(), nullptr, nullptr), "open video");
        std::unique_ptr<AVFormatContextModern, FormatDeleter> format_owner(format, FormatDeleter{api_.get()});
        check(api_->avformat_find_stream_info(format, nullptr), "read stream info");

        const AVCodec * codec = nullptr;
        const int stream_index = api_->av_find_best_stream(format, 0, -1, -1, &codec, 0);
        if (stream_index < 0 || codec == nullptr) {
            throw std::runtime_error("FFmpeg decoder found no video stream: " + path.string());
        }
        if (static_cast<unsigned int>(stream_index) >= format->nb_streams || format->streams == nullptr) {
            throw std::runtime_error("FFmpeg decoder returned an invalid video stream index");
        }
        AVStreamModern * stream = format->streams[stream_index];
        if (stream == nullptr || stream->codecpar == nullptr) {
            throw std::runtime_error("FFmpeg decoder found video stream without codec parameters");
        }

        AVCodecContext * codec_ctx = api_->avcodec_alloc_context3(codec);
        if (codec_ctx == nullptr) {
            throw std::runtime_error("FFmpeg decoder failed to allocate codec context");
        }
        std::unique_ptr<AVCodecContext, CodecContextDeleter> codec_owner(codec_ctx, CodecContextDeleter{api_.get()});
        check(api_->avcodec_parameters_to_context(codec_ctx, stream->codecpar), "copy codec parameters");
        AVDictionary * codec_options = nullptr;
        check(api_->av_dict_set(&codec_options, "threads", "auto", 0), "set decoder threads");
        const int open_status = api_->avcodec_open2(codec_ctx, codec, &codec_options);
        api_->av_dict_free(&codec_options);
        check(open_status, "open video decoder");

        AVPacket * packet = api_->av_packet_alloc();
        AVFrame * frame = api_->av_frame_alloc();
        if (packet == nullptr || frame == nullptr) {
            if (packet != nullptr) api_->av_packet_free(&packet);
            if (frame != nullptr) api_->av_frame_free(&frame);
            throw std::runtime_error("FFmpeg decoder failed to allocate packet/frame");
        }
        std::unique_ptr<AVPacket, PacketDeleter> packet_owner(packet, PacketDeleter{api_.get()});
        std::unique_ptr<AVFrame, FrameDeleter> frame_owner(frame, FrameDeleter{api_.get()});

        int64_t frame_count = 0;
        bool stopped_at_duration = false;
        while (api_->av_read_frame(format, packet) >= 0) {
            bool keep_decoding = true;
            if (packet->stream_index == stream_index) {
                keep_decoding = decode_packet(codec_ctx, packet, frame, stream->time_base, max_duration_sec, consume, frame_count);
            }
            api_->av_packet_unref(packet);
            if (!keep_decoding) {
                stopped_at_duration = true;
                break;
            }
        }
        if (!stopped_at_duration) {
            check(api_->avcodec_send_packet(codec_ctx, nullptr), "flush video decoder");
            drain_frames(codec_ctx, frame, stream->time_base, max_duration_sec, consume, frame_count);
        }

        if (frame_count == 0) {
            throw std::runtime_error("FFmpeg decoder produced no frames: " + path.string());
        }
    }

private:
    std::string error_string(int code) const {
        char buffer[256] = {};
        if (api_->av_strerror != nullptr && api_->av_strerror(code, buffer, sizeof(buffer)) == 0) {
            return buffer;
        }
        return std::to_string(code);
    }

    void check(int status, const char * action) const {
        if (status < 0) {
            throw std::runtime_error(
                std::string("FFmpeg failed to ") + action + ": " + error_string(status));
        }
    }

    double timestamp_seconds(const AVFrame & frame, AVRational time_base, int64_t frame_count) const {
        int64_t pts = frame.pts;
        if (pts == kNoPts) {
            pts = frame_count;
            time_base = {1, 30};
        }
        if (time_base.num <= 0 || time_base.den <= 0) {
            return static_cast<double>(frame_count) / kDefaultFrameRate;
        }
        return static_cast<double>(pts) * static_cast<double>(time_base.num) / static_cast<double>(time_base.den);
    }

    bool append_frame(
        const AVFrame & frame,
        AVRational time_base,
        double max_duration_sec,
        const RgbVideoConsumer & consume,
        int64_t & frame_count) const {
        if (frame.width <= 0 || frame.height <= 0 || frame.format < 0) {
            throw std::runtime_error("FFmpeg decoder produced an invalid video frame");
        }
        if (rgb24_ < 0) {
            rgb24_ = api_->av_get_pix_fmt("rgb24");
            if (rgb24_ < 0) {
                throw std::runtime_error("FFmpeg libavutil does not know rgb24 pixel format");
            }
        }
        if (sws_context_ == nullptr ||
            sws_width_ != frame.width ||
            sws_height_ != frame.height ||
            sws_format_ != frame.format) {
            if (sws_context_ != nullptr) {
                api_->sws_freeContext(sws_context_);
                sws_context_ = nullptr;
            }
            sws_context_ = api_->sws_getContext(
                frame.width, frame.height, frame.format,
                frame.width, frame.height, rgb24_,
                4, nullptr, nullptr, nullptr);
            if (sws_context_ == nullptr) {
                throw std::runtime_error("FFmpeg failed to create RGB24 scaler");
            }
            sws_width_ = frame.width;
            sws_height_ = frame.height;
            sws_format_ = frame.format;

        }

        const double timestamp = timestamp_seconds(frame, time_base, frame_count);
        if (max_duration_sec > 0.0 && timestamp > max_duration_sec) {
            return false;
        }
        RgbVideoFrame decoded;
        decoded.time = timestamp;
        AVFrame * rgb_frame = api_->av_frame_alloc();
        if (rgb_frame == nullptr) throw std::runtime_error("FFmpeg failed to allocate RGB frame");
        const auto api = api_;
        std::shared_ptr<AVFrame> owner(rgb_frame, [api](AVFrame * value) { api->av_frame_free(&value); });
        rgb_frame->width = frame.width;
        rgb_frame->height = frame.height;
        rgb_frame->format = rgb24_;
        // Retain FFmpeg's aligned, padded allocation through the returned frame.
        check(api_->av_frame_get_buffer(rgb_frame, 0), "allocate RGB frame buffer");
        api_->sws_scale(sws_context_, const_cast<const uint8_t * const *>(frame.data), frame.linesize,
                        0, frame.height, rgb_frame->data, rgb_frame->linesize);
        decoded.stride = rgb_frame->linesize[0];
        decoded.rgb = std::shared_ptr<const uint8_t>(owner, rgb_frame->data[0]);
        if (frame_count == 0) {
            width_ = frame.width;
            height_ = frame.height;
        } else if (width_ != frame.width || height_ != frame.height) {
            throw std::runtime_error("FFmpeg decoder produced changing frame sizes");
        }
        ++frame_count;
        return consume(std::move(decoded), width_, height_);
    }

    bool drain_frames(
        AVCodecContext * codec_ctx,
        AVFrame * frame,
        AVRational time_base,
        double max_duration_sec,
        const RgbVideoConsumer & consume,
        int64_t & frame_count) const {
        while (api_->avcodec_receive_frame(codec_ctx, frame) == 0) {
            const bool keep_decoding = append_frame(*frame, time_base, max_duration_sec, consume, frame_count);
            api_->av_frame_unref(frame);
            if (!keep_decoding) {
                return false;
            }
        }
        return true;
    }

    bool decode_packet(
        AVCodecContext * codec_ctx,
        AVPacket * packet,
        AVFrame * frame,
        AVRational time_base,
        double max_duration_sec,
        const RgbVideoConsumer & consume,
        int64_t & frame_count) const {
        const int send = api_->avcodec_send_packet(codec_ctx, packet);
        if (send < 0) {
            throw std::runtime_error("FFmpeg failed to send packet: " + error_string(send));
        }
        return drain_frames(codec_ctx, frame, time_base, max_duration_sec, consume, frame_count);
    }

    struct FormatDeleter {
        const LibAvApi * api = nullptr;
        void operator()(AVFormatContextModern * value) const {
            if (value != nullptr) {
                api->avformat_close_input(&value);
            }
        }
    };
    struct CodecContextDeleter {
        const LibAvApi * api = nullptr;
        void operator()(AVCodecContext * value) const {
            if (value != nullptr) {
                api->avcodec_free_context(&value);
            }
        }
    };
    struct PacketDeleter {
        const LibAvApi * api = nullptr;
        void operator()(AVPacket * value) const {
            if (value != nullptr) {
                api->av_packet_free(&value);
            }
        }
    };
    struct FrameDeleter {
        const LibAvApi * api = nullptr;
        void operator()(AVFrame * value) const {
            if (value != nullptr) {
                api->av_frame_free(&value);
            }
        }
    };
    std::shared_ptr<const LibAvApi> api_;

    mutable int64_t width_ = 0, height_ = 0;
    mutable SwsContext * sws_context_ = nullptr;
    mutable int sws_width_ = 0;
    mutable int sws_height_ = 0;
    mutable int sws_format_ = -1;
    mutable int rgb24_ = -1;
};

} // namespace

void read_video_rgb(const std::filesystem::path & path, const RgbVideoConsumer & consume,
                    double max_duration_sec, size_t prefetch_frames) {
    if (!std::filesystem::is_regular_file(path))
        throw std::runtime_error("Video path is not a regular file: " + path.string());
    if (prefetch_frames == 0) {
        LibAvVideoDecoder decoder;
        decoder.decode(path, max_duration_sec, consume);
        return;
    }
    struct PendingFrame {
        RgbVideoFrame frame;
        int64_t width, height;
    };
    std::mutex mutex;
    std::condition_variable ready;
    std::deque<PendingFrame> pending;
    bool done = false, cancelled = false;
    auto producer = std::async(std::launch::async, [&] {
        try {
            LibAvVideoDecoder decoder;
            decoder.decode(path, max_duration_sec, [&](RgbVideoFrame frame, int64_t width, int64_t height) {
                std::unique_lock<std::mutex> lock(mutex);
                ready.wait(lock, [&] { return cancelled || pending.size() < prefetch_frames; });
                if (cancelled) return false;
                pending.push_back({std::move(frame), width, height});
                ready.notify_all();
                return true;
            });
        } catch (...) {
            { std::lock_guard<std::mutex> lock(mutex); done = true; }
            ready.notify_all();
            throw;
        }
        { std::lock_guard<std::mutex> lock(mutex); done = true; }
        ready.notify_all();
    });
    try {
        while (true) {
            std::unique_lock<std::mutex> lock(mutex);
            ready.wait(lock, [&] { return done || !pending.empty(); });
            if (pending.empty()) break;
            auto frame = std::move(pending.front());
            pending.pop_front();
            ready.notify_all();
            lock.unlock();
            if (!consume(std::move(frame.frame), frame.width, frame.height)) break;
        }
    } catch (...) {
        { std::lock_guard<std::mutex> lock(mutex); cancelled = true; }
        ready.notify_all();
        producer.wait();
        throw;
    }
    { std::lock_guard<std::mutex> lock(mutex); cancelled = true; }
    ready.notify_all();
    producer.get();
}

RgbVideo load_video_rgb(const std::filesystem::path & path, double max_duration_sec) {
    RgbVideo out;
    read_video_rgb(path, [&](RgbVideoFrame frame, int64_t width, int64_t height) {
        out.width = width;
        out.height = height;
        out.frames.push_back(std::move(frame));
        return true;
    }, max_duration_sec);
    return out;
}
} // namespace engine::io
#else
#include <stdexcept>
namespace engine::io {
void read_video_rgb(const std::filesystem::path &, const RgbVideoConsumer &, double, size_t) {
    throw std::runtime_error("FFmpeg shared-library video decoding is unavailable in browser builds");
}
RgbVideo load_video_rgb(const std::filesystem::path &, double) {
    throw std::runtime_error("FFmpeg shared-library video decoding is unavailable in browser builds");
}
} // namespace engine::io
#endif
