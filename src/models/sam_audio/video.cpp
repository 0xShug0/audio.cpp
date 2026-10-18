#include "engine/models/sam_audio/frontend.h"
#include "engine/framework/io/dynamic_library.h"

#include <array>
#include <cmath>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>

namespace engine::models::sam_audio {
namespace {
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

class LibAvVideoDecoder {
public:
    LibAvVideoDecoder()
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
          })),
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
          })),
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
          })),
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
          })) {
        if (!available()) return;

        avformat_version = symbol<unsigned (*)()>(avformat_, "avformat_version");
        avformat_open_input = symbol<int (*)(AVFormatContextModern **, const char *, const AVInputFormat *, AVDictionary **)>(
            avformat_, "avformat_open_input");
        avformat_find_stream_info = symbol<int (*)(AVFormatContextModern *, AVDictionary **)>(
            avformat_, "avformat_find_stream_info");
        av_find_best_stream = symbol<int (*)(AVFormatContextModern *, int, int, int, const AVCodec **, int)>(
            avformat_, "av_find_best_stream");
        av_read_frame = symbol<int (*)(AVFormatContextModern *, AVPacket *)>(avformat_, "av_read_frame");
        avformat_close_input = symbol<void (*)(AVFormatContextModern **)>(avformat_, "avformat_close_input");

        avcodec_alloc_context3 = symbol<AVCodecContext * (*)(const AVCodec *)>(avcodec_, "avcodec_alloc_context3");
        avcodec_parameters_to_context = symbol<int (*)(AVCodecContext *, const AVCodecParameters *)>(
            avcodec_, "avcodec_parameters_to_context");
        avcodec_open2 = symbol<int (*)(AVCodecContext *, const AVCodec *, AVDictionary **)>(avcodec_, "avcodec_open2");
        avcodec_send_packet = symbol<int (*)(AVCodecContext *, const AVPacket *)>(avcodec_, "avcodec_send_packet");
        avcodec_receive_frame = symbol<int (*)(AVCodecContext *, AVFrame *)>(avcodec_, "avcodec_receive_frame");
        avcodec_free_context = symbol<void (*)(AVCodecContext **)>(avcodec_, "avcodec_free_context");
        av_packet_alloc = symbol<AVPacket * (*)()>(avcodec_, "av_packet_alloc");
        av_packet_free = symbol<void (*)(AVPacket **)>(avcodec_, "av_packet_free");
        av_packet_unref = symbol<void (*)(AVPacket *)>(avcodec_, "av_packet_unref");

        av_frame_alloc = symbol<AVFrame * (*)()>(avutil_, "av_frame_alloc");
        av_frame_free = symbol<void (*)(AVFrame **)>(avutil_, "av_frame_free");
        av_frame_unref = symbol<void (*)(AVFrame *)>(avutil_, "av_frame_unref");
        av_get_pix_fmt = symbol<int (*)(const char *)>(avutil_, "av_get_pix_fmt");
        av_dict_set = symbol<int (*)(AVDictionary **, const char *, const char *, int)>(avutil_, "av_dict_set");
        av_dict_free = symbol<void (*)(AVDictionary **)>(avutil_, "av_dict_free");
        av_strerror = optional_symbol<int (*)(int, char *, size_t)>(avutil_, "av_strerror");

        sws_getContext = symbol<SwsContext * (*)(
            int, int, int, int, int, int, int, SwsFilter *, SwsFilter *, const double *)>(
            swscale_, "sws_getContext");
        sws_scale = symbol<int (*)(SwsContext *, const uint8_t * const [], const int [], int, int, uint8_t * const [], const int [])>(
            swscale_, "sws_scale");
        sws_freeContext = symbol<void (*)(SwsContext *)>(swscale_, "sws_freeContext");

        const unsigned version = avformat_version();
        const unsigned major = version >> 16U;
        if (major < 59U) {
            throw std::runtime_error(
                "SAM Audio video decoding found libavformat, but the version is too old for "
                "the in-process decoder; install FFmpeg 5+ runtime libraries, or pre-decode the "
                "video into a frame directory and pass that directory as video");
        }
    }

    ~LibAvVideoDecoder() {
        if (sws_context_ != nullptr) {
            sws_freeContext(sws_context_);
            sws_context_ = nullptr;
        }
        engine::io::close_dynamic_library(swscale_);
        engine::io::close_dynamic_library(avutil_);
        engine::io::close_dynamic_library(avcodec_);
        engine::io::close_dynamic_library(avformat_);
    }

    bool available() const {
        return avformat_ != nullptr && avcodec_ != nullptr && avutil_ != nullptr && swscale_ != nullptr;
    }

    SourceVideo decode(const std::filesystem::path & path, double max_duration_sec) {
        if (!available()) {
            throw std::runtime_error(
                "SAM Audio video decoding requires FFmpeg/libav shared libraries, but one or "
                "more of libavformat, libavcodec, libavutil, and libswscale could not be loaded. "
                "Install FFmpeg runtime libraries to use reference_video_path");
        }

        AVFormatContextModern * format = nullptr;
        check(avformat_open_input(&format, path.string().c_str(), nullptr, nullptr), "open video");
        std::unique_ptr<AVFormatContextModern, FormatDeleter> format_owner(format, FormatDeleter{this});
        check(avformat_find_stream_info(format, nullptr), "read stream info");

        const AVCodec * codec = nullptr;
        const int stream_index = av_find_best_stream(format, 0, -1, -1, &codec, 0);
        if (stream_index < 0 || codec == nullptr) {
            throw std::runtime_error("SAM Audio FFmpeg decoder found no video stream: " + path.string());
        }
        if (static_cast<unsigned int>(stream_index) >= format->nb_streams || format->streams == nullptr) {
            throw std::runtime_error("SAM Audio FFmpeg decoder returned an invalid video stream index");
        }
        AVStreamModern * stream = format->streams[stream_index];
        if (stream == nullptr || stream->codecpar == nullptr) {
            throw std::runtime_error("SAM Audio FFmpeg decoder found video stream without codec parameters");
        }

        AVCodecContext * codec_ctx = avcodec_alloc_context3(codec);
        if (codec_ctx == nullptr) {
            throw std::runtime_error("SAM Audio FFmpeg decoder failed to allocate codec context");
        }
        std::unique_ptr<AVCodecContext, CodecContextDeleter> codec_owner(codec_ctx, CodecContextDeleter{this});
        check(avcodec_parameters_to_context(codec_ctx, stream->codecpar), "copy codec parameters");
        AVDictionary * codec_options = nullptr;
        check(av_dict_set(&codec_options, "threads", "auto", 0), "set decoder threads");
        const int open_status = avcodec_open2(codec_ctx, codec, &codec_options);
        av_dict_free(&codec_options);
        check(open_status, "open video decoder");

        AVPacket * packet = av_packet_alloc();
        AVFrame * frame = av_frame_alloc();
        if (packet == nullptr || frame == nullptr) {
            if (packet != nullptr) av_packet_free(&packet);
            if (frame != nullptr) av_frame_free(&frame);
            throw std::runtime_error("SAM Audio FFmpeg decoder failed to allocate packet/frame");
        }
        std::unique_ptr<AVPacket, PacketDeleter> packet_owner(packet, PacketDeleter{this});
        std::unique_ptr<AVFrame, FrameDeleter> frame_owner(frame, FrameDeleter{this});

        SourceVideo out;
        int64_t frame_count = 0;
        bool stopped_at_duration = false;
        while (av_read_frame(format, packet) >= 0) {
            bool keep_decoding = true;
            if (packet->stream_index == stream_index) {
                keep_decoding = decode_packet(codec_ctx, packet, frame, stream->time_base, max_duration_sec, out, frame_count);
            }
            av_packet_unref(packet);
            if (!keep_decoding) {
                stopped_at_duration = true;
                break;
            }
        }
        if (!stopped_at_duration) {
            check(avcodec_send_packet(codec_ctx, nullptr), "flush video decoder");
            drain_frames(codec_ctx, frame, stream->time_base, max_duration_sec, out, frame_count);
        }

        if (out.frames.empty()) {
            throw std::runtime_error("SAM Audio FFmpeg decoder produced no frames: " + path.string());
        }
        return out;
    }

private:
    template <typename Fn>
    Fn symbol(engine::io::DynamicLibraryHandle library, const char * name) {
        auto * address = engine::io::dynamic_library_symbol(library, name);
        if (address == nullptr) {
            throw std::runtime_error(std::string("SAM Audio FFmpeg library is missing symbol ") + name);
        }
        return reinterpret_cast<Fn>(address);
    }

    template <typename Fn>
    Fn optional_symbol(engine::io::DynamicLibraryHandle library, const char * name) {
        return reinterpret_cast<Fn>(engine::io::dynamic_library_symbol(library, name));
    }

    std::string error_string(int code) const {
        char buffer[256] = {};
        if (av_strerror != nullptr && av_strerror(code, buffer, sizeof(buffer)) == 0) {
            return buffer;
        }
        return std::to_string(code);
    }

    void check(int status, const char * action) const {
        if (status < 0) {
            throw std::runtime_error(
                std::string("SAM Audio FFmpeg failed to ") + action + ": " + error_string(status));
        }
    }

    double timestamp_seconds(const AVFrame & frame, AVRational time_base, int64_t /* frame_count */) const {
        int64_t pts = frame.pts;
        if (pts == kNoPts) {
            throw std::runtime_error("SAM Audio video frame has no presentation timestamp");
        }
        if (time_base.num <= 0 || time_base.den <= 0) {
            throw std::runtime_error("SAM Audio video has an invalid time base");
        }
        return static_cast<double>(pts) * static_cast<double>(time_base.num) / static_cast<double>(time_base.den);
    }

    bool append_frame(
        const AVFrame & frame,
        AVRational time_base,
        double max_duration_sec,
        SourceVideo & out,
        int64_t & frame_count) const {
        if (frame.width <= 0 || frame.height <= 0 || frame.format < 0) {
            throw std::runtime_error("SAM Audio FFmpeg decoder produced an invalid video frame");
        }
        if (rgb24_ < 0) {
            rgb24_ = av_get_pix_fmt("rgb24");
            if (rgb24_ < 0) {
                throw std::runtime_error("SAM Audio FFmpeg libavutil does not know rgb24 pixel format");
            }
        }
        if (sws_context_ == nullptr ||
            sws_width_ != frame.width ||
            sws_height_ != frame.height ||
            sws_format_ != frame.format) {
            if (sws_context_ != nullptr) {
                sws_freeContext(sws_context_);
                sws_context_ = nullptr;
            }
            sws_context_ = sws_getContext(
                frame.width, frame.height, frame.format,
                frame.width, frame.height, rgb24_,
                4, nullptr, nullptr, nullptr);
            if (sws_context_ == nullptr) {
                throw std::runtime_error("SAM Audio FFmpeg failed to create RGB24 scaler");
            }
            sws_width_ = frame.width;
            sws_height_ = frame.height;
            sws_format_ = frame.format;
        }

        const double timestamp = timestamp_seconds(frame, time_base, frame_count);
        if (max_duration_sec > 0.0 && timestamp > max_duration_sec) {
            return false;
        }
        DecodedFrame decoded;
        decoded.time = timestamp;
        decoded.rgb.resize(static_cast<size_t>(frame.width) * static_cast<size_t>(frame.height) * 3);
        uint8_t * dst_data[4] = {decoded.rgb.data(), nullptr, nullptr, nullptr};
        int dst_linesize[4] = {frame.width * 3, 0, 0, 0};
        sws_scale(
            sws_context_,
            const_cast<const uint8_t * const *>(frame.data),
            frame.linesize,
            0,
            frame.height,
            dst_data,
            dst_linesize);
        if (out.frames.empty()) {
            out.width = frame.width;
            out.height = frame.height;
        } else if (out.width != frame.width || out.height != frame.height) {
            throw std::runtime_error("SAM Audio FFmpeg decoder produced changing frame sizes");
        }
        out.frames.push_back(std::move(decoded));
        ++frame_count;
        return true;
    }

    bool drain_frames(
        AVCodecContext * codec_ctx,
        AVFrame * frame,
        AVRational time_base,
        double max_duration_sec,
        SourceVideo & out,
        int64_t & frame_count) const {
        while (avcodec_receive_frame(codec_ctx, frame) == 0) {
            const bool keep_decoding = append_frame(*frame, time_base, max_duration_sec, out, frame_count);
            av_frame_unref(frame);
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
        SourceVideo & out,
        int64_t & frame_count) const {
        const int send = avcodec_send_packet(codec_ctx, packet);
        if (send < 0) {
            throw std::runtime_error("SAM Audio FFmpeg failed to send packet: " + error_string(send));
        }
        return drain_frames(codec_ctx, frame, time_base, max_duration_sec, out, frame_count);
    }

    struct FormatDeleter {
        LibAvVideoDecoder * api = nullptr;
        void operator()(AVFormatContextModern * value) const {
            if (value != nullptr) {
                api->avformat_close_input(&value);
            }
        }
    };
    struct CodecContextDeleter {
        LibAvVideoDecoder * api = nullptr;
        void operator()(AVCodecContext * value) const {
            if (value != nullptr) {
                api->avcodec_free_context(&value);
            }
        }
    };
    struct PacketDeleter {
        LibAvVideoDecoder * api = nullptr;
        void operator()(AVPacket * value) const {
            if (value != nullptr) {
                api->av_packet_free(&value);
            }
        }
    };
    struct FrameDeleter {
        LibAvVideoDecoder * api = nullptr;
        void operator()(AVFrame * value) const {
            if (value != nullptr) {
                api->av_frame_free(&value);
            }
        }
    };
    struct SwsDeleter {
        const LibAvVideoDecoder * api = nullptr;
        void operator()(SwsContext * value) const {
            if (value != nullptr) {
                api->sws_freeContext(value);
            }
        }
    };

    engine::io::DynamicLibraryHandle avformat_ = nullptr;
    engine::io::DynamicLibraryHandle avcodec_ = nullptr;
    engine::io::DynamicLibraryHandle avutil_ = nullptr;
    engine::io::DynamicLibraryHandle swscale_ = nullptr;

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
    void (*av_frame_free)(AVFrame **) = nullptr;
    void (*av_frame_unref)(AVFrame *) = nullptr;
    int (*av_get_pix_fmt)(const char *) = nullptr;
    int (*av_dict_set)(AVDictionary **, const char *, const char *, int) = nullptr;
    void (*av_dict_free)(AVDictionary **) = nullptr;
    int (*av_strerror)(int, char *, size_t) = nullptr;

    SwsContext * (*sws_getContext)(int, int, int, int, int, int, int, SwsFilter *, SwsFilter *, const double *) = nullptr;
    int (*sws_scale)(SwsContext *, const uint8_t * const [], const int [], int, int, uint8_t * const [], const int []) = nullptr;
    void (*sws_freeContext)(SwsContext *) = nullptr;

    mutable SwsContext * sws_context_ = nullptr;
    mutable int sws_width_ = 0;
    mutable int sws_height_ = 0;
    mutable int sws_format_ = -1;
    mutable int rgb24_ = -1;
};


}  // namespace

SourceVideo load_video_frames(const std::filesystem::path & path) {
    LibAvVideoDecoder decoder;
    return decoder.decode(path, 0.0);
}

}  // namespace engine::models::sam_audio

