#include "engine/models/controlfoley/video_processing.h"

#include "engine/framework/debug/profiler.h"
#include "engine/framework/debug/trace.h"
#include "engine/framework/io/dynamic_library.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>

#define STB_IMAGE_IMPLEMENTATION
#include "../../../external/ggml/examples/stb_image.h"

namespace engine::models::controlfoley::video {
namespace {

constexpr double kDefaultFrameDirectoryFps = 30.0;
constexpr int64_t kNoPts = std::numeric_limits<int64_t>::min();

bool is_supported_image_file(const std::filesystem::path & path) {
    auto extension = path.extension().string();
    std::transform(extension.begin(), extension.end(), extension.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return extension == ".png" || extension == ".jpg" || extension == ".jpeg" ||
        extension == ".bmp" || extension == ".tga";
}

std::string frame_directory_help() {
    return "ControlFoley frame-directory fallback expects a directory containing "
           "decoded image frames (*.png/*.jpg/*.jpeg/*.bmp/*.tga) and optional "
           "timestamps.txt with one timestamp per frame";
}

std::vector<std::filesystem::path> list_frame_paths(const std::filesystem::path & path) {
    if (!std::filesystem::exists(path)) {
        throw std::runtime_error("ControlFoley video path does not exist: " + path.string());
    }
    if (!std::filesystem::is_directory(path)) {
        throw std::runtime_error(frame_directory_help() + ": " + path.string());
    }

    std::vector<std::filesystem::path> out;
    for (const auto & entry : std::filesystem::directory_iterator(path)) {
        if (entry.is_regular_file() && is_supported_image_file(entry.path())) {
            out.push_back(entry.path());
        }
    }
    std::sort(out.begin(), out.end());
    if (out.empty()) {
        throw std::runtime_error(frame_directory_help() + ": " + path.string());
    }
    return out;
}

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
                "ControlFoley video decoding found libavformat, but the version is too old for "
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
                "ControlFoley video decoding requires FFmpeg/libav shared libraries, but one or "
                "more of libavformat, libavcodec, libavutil, and libswscale could not be loaded. "
                "Install FFmpeg runtime libraries, or pre-decode the video into a frame directory "
                "with image frames and optional timestamps.txt, then pass that directory as video");
        }

        AVFormatContextModern * format = nullptr;
        check(avformat_open_input(&format, path.string().c_str(), nullptr, nullptr), "open video");
        std::unique_ptr<AVFormatContextModern, FormatDeleter> format_owner(format, FormatDeleter{this});
        check(avformat_find_stream_info(format, nullptr), "read stream info");

        const AVCodec * codec = nullptr;
        const int stream_index = av_find_best_stream(format, 0, -1, -1, &codec, 0);
        if (stream_index < 0 || codec == nullptr) {
            throw std::runtime_error("ControlFoley FFmpeg decoder found no video stream: " + path.string());
        }
        if (static_cast<unsigned int>(stream_index) >= format->nb_streams || format->streams == nullptr) {
            throw std::runtime_error("ControlFoley FFmpeg decoder returned an invalid video stream index");
        }
        AVStreamModern * stream = format->streams[stream_index];
        if (stream == nullptr || stream->codecpar == nullptr) {
            throw std::runtime_error("ControlFoley FFmpeg decoder found video stream without codec parameters");
        }

        AVCodecContext * codec_ctx = avcodec_alloc_context3(codec);
        if (codec_ctx == nullptr) {
            throw std::runtime_error("ControlFoley FFmpeg decoder failed to allocate codec context");
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
            throw std::runtime_error("ControlFoley FFmpeg decoder failed to allocate packet/frame");
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
            throw std::runtime_error("ControlFoley FFmpeg decoder produced no frames: " + path.string());
        }
        return out;
    }

private:
    template <typename Fn>
    Fn symbol(engine::io::DynamicLibraryHandle library, const char * name) {
        auto * address = engine::io::dynamic_library_symbol(library, name);
        if (address == nullptr) {
            throw std::runtime_error(std::string("ControlFoley FFmpeg library is missing symbol ") + name);
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
                std::string("ControlFoley FFmpeg failed to ") + action + ": " + error_string(status));
        }
    }

    double timestamp_seconds(const AVFrame & frame, AVRational time_base, int64_t frame_count) const {
        int64_t pts = frame.pts;
        if (pts == kNoPts) {
            pts = frame_count;
            time_base = {1, 30};
        }
        if (time_base.num <= 0 || time_base.den <= 0) {
            return static_cast<double>(frame_count) / kDefaultFrameDirectoryFps;
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
            throw std::runtime_error("ControlFoley FFmpeg decoder produced an invalid video frame");
        }
        if (rgb24_ < 0) {
            rgb24_ = av_get_pix_fmt("rgb24");
            if (rgb24_ < 0) {
                throw std::runtime_error("ControlFoley FFmpeg libavutil does not know rgb24 pixel format");
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
                throw std::runtime_error("ControlFoley FFmpeg failed to create RGB24 scaler");
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
            throw std::runtime_error("ControlFoley FFmpeg decoder produced changing frame sizes");
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
            throw std::runtime_error("ControlFoley FFmpeg failed to send packet: " + error_string(send));
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

std::vector<double> read_timestamps(
    const std::filesystem::path & directory,
    size_t frame_count) {
    const auto path = directory / "timestamps.txt";
    std::vector<double> out;
    if (std::filesystem::exists(path)) {
        std::ifstream stream(path);
        if (!stream) {
            throw std::runtime_error("ControlFoley failed to read video timestamps: " + path.string());
        }
        std::string line;
        while (std::getline(stream, line)) {
            const size_t split = line.find(',');
            if (split != std::string::npos) {
                line.resize(split);
            }
            if (line.empty()) {
                continue;
            }
            out.push_back(std::stod(line));
        }
        if (out.size() != frame_count) {
            throw std::runtime_error("ControlFoley video timestamps count does not match frame count: " + path.string());
        }
        return out;
    }

    out.resize(frame_count);
    for (size_t i = 0; i < frame_count; ++i) {
        out[i] = static_cast<double>(i) / kDefaultFrameDirectoryFps;
    }
    return out;
}

struct ImageBuffer {
    int64_t width = 0;
    int64_t height = 0;
    std::vector<uint8_t> rgb;
};

ImageBuffer load_rgb_image(const std::filesystem::path & path) {
    int width = 0;
    int height = 0;
    int channels = 0;
    unsigned char * data = stbi_load(path.string().c_str(), &width, &height, &channels, 3);
    if (data == nullptr || width <= 0 || height <= 0) {
        if (data != nullptr) {
            stbi_image_free(data);
        }
        throw std::runtime_error("ControlFoley failed to load video frame image: " + path.string());
    }
    ImageBuffer out;
    out.width = width;
    out.height = height;
    out.rgb.assign(data, data + static_cast<size_t>(width) * static_cast<size_t>(height) * 3);
    stbi_image_free(data);
    return out;
}

struct Contributor {
    int64_t index = 0;
    double weight = 0.0;
};

double cubic_weight(double x) {
    constexpr double a = -0.75;
    x = std::abs(x);
    if (x <= 1.0) {
        return ((a + 2.0) * x - (a + 3.0)) * x * x + 1.0;
    }
    if (x < 2.0) {
        return ((a * x - 5.0 * a) * x + 8.0 * a) * x - 4.0 * a;
    }
    return 0.0;
}

std::vector<std::array<Contributor, 4>> make_cubic_contributors(
    int64_t input_size,
    int64_t output_size) {
    if (input_size <= 0 || output_size <= 0) {
        throw std::runtime_error("ControlFoley video resize shape is invalid");
    }
    const double scale = static_cast<double>(input_size) / static_cast<double>(output_size);
    std::vector<std::array<Contributor, 4>> out(static_cast<size_t>(output_size));
    for (int64_t dst = 0; dst < output_size; ++dst) {
        const double center = (static_cast<double>(dst) + 0.5) * scale - 0.5;
        const int64_t start = static_cast<int64_t>(std::floor(center)) - 1;
        double sum = 0.0;
        for (int tap = 0; tap < 4; ++tap) {
            const int64_t src = std::clamp<int64_t>(start + tap, 0, input_size - 1);
            const double weight = cubic_weight(center - static_cast<double>(start + tap));
            out[static_cast<size_t>(dst)][static_cast<size_t>(tap)] = {src, weight};
            sum += weight;
        }
        if (sum != 0.0 && std::isfinite(sum)) {
            for (auto & contributor : out[static_cast<size_t>(dst)]) {
                contributor.weight /= sum;
            }
        }
    }
    return out;
}

ImageBuffer resize_rgb_bicubic(
    int64_t source_width,
    int64_t source_height,
    const std::vector<uint8_t> & source_rgb,
    int64_t width,
    int64_t height,
    const std::vector<std::array<Contributor, 4>> & x_contributors,
    const std::vector<std::array<Contributor, 4>> & y_contributors) {
    if (source_width == width && source_height == height) {
        ImageBuffer out;
        out.width = width;
        out.height = height;
        out.rgb = source_rgb;
        return out;
    }

    std::vector<float> temp(static_cast<size_t>(source_height * width * 3));
    for (int64_t y = 0; y < source_height; ++y) {
        for (int64_t x = 0; x < width; ++x) {
            const auto & contributors = x_contributors[static_cast<size_t>(x)];
            for (int64_t c = 0; c < 3; ++c) {
                double value = 0.0;
                for (const auto & contributor : contributors) {
                    const int64_t src = (y * source_width + contributor.index) * 3 + c;
                    value += static_cast<double>(source_rgb[static_cast<size_t>(src)]) * contributor.weight;
                }
                temp[static_cast<size_t>((y * width + x) * 3 + c)] = static_cast<float>(value);
            }
        }
    }

    ImageBuffer out;
    out.width = width;
    out.height = height;
    out.rgb.resize(static_cast<size_t>(height * width * 3));
    for (int64_t y = 0; y < height; ++y) {
        const auto & contributors = y_contributors[static_cast<size_t>(y)];
        for (int64_t x = 0; x < width; ++x) {
            for (int64_t c = 0; c < 3; ++c) {
                double value = 0.0;
                for (const auto & contributor : contributors) {
                    const int64_t src = (contributor.index * width + x) * 3 + c;
                    value += static_cast<double>(temp[static_cast<size_t>(src)]) * contributor.weight;
                }
                value = std::clamp(value, 0.0, 255.0);
                out.rgb[static_cast<size_t>((y * width + x) * 3 + c)] = static_cast<uint8_t>(std::lround(value));
            }
        }
    }
    return out;
}

struct ResizePlan {
    ResizePlan(
        int64_t source_width,
        int64_t source_height,
        int64_t size,
        bool center_crop)
        : source_width(source_width),
          source_height(source_height),
          target_size(size),
          center_crop(center_crop) {
        if (center_crop) {
            const double ratio = static_cast<double>(size) / static_cast<double>(std::min(source_width, source_height));
            resized_width = static_cast<int64_t>(std::llround(static_cast<double>(source_width) * ratio));
            resized_height = static_cast<int64_t>(std::llround(static_cast<double>(source_height) * ratio));
        } else {
            resized_width = size;
            resized_height = size;
        }
        x_contributors = make_cubic_contributors(source_width, resized_width);
        y_contributors = make_cubic_contributors(source_height, resized_height);
    }

    ImageBuffer resize(const std::vector<uint8_t> & source_rgb) const {
        auto resized = resize_rgb_bicubic(
            source_width,
            source_height,
            source_rgb,
            resized_width,
            resized_height,
            x_contributors,
            y_contributors);
        if (!center_crop) {
            return resized;
        }

        const int64_t crop_x = std::max<int64_t>(0, (resized.width - target_size) / 2);
        const int64_t crop_y = std::max<int64_t>(0, (resized.height - target_size) / 2);
        ImageBuffer out;
        out.width = target_size;
        out.height = target_size;
        out.rgb.resize(static_cast<size_t>(target_size * target_size * 3));
        for (int64_t y = 0; y < target_size; ++y) {
            const auto * src = resized.rgb.data() + static_cast<size_t>(((crop_y + y) * resized.width + crop_x) * 3);
            auto * dst = out.rgb.data() + static_cast<size_t>(y * target_size * 3);
            std::memcpy(dst, src, static_cast<size_t>(target_size * 3));
        }
        return out;
    }

    int64_t source_width = 0;
    int64_t source_height = 0;
    int64_t target_size = 0;
    int64_t resized_width = 0;
    int64_t resized_height = 0;
    bool center_crop = false;
    std::vector<std::array<Contributor, 4>> x_contributors;
    std::vector<std::array<Contributor, 4>> y_contributors;
};

SourceVideo load_frame_directory(const std::filesystem::path & path) {
    const auto frame_paths = list_frame_paths(path);
    const auto timestamps = read_timestamps(path, frame_paths.size());
    SourceVideo out;
    out.frames.reserve(frame_paths.size());
    for (size_t i = 0; i < frame_paths.size(); ++i) {
        auto image = load_rgb_image(frame_paths[i]);
        if (i == 0) {
            out.width = image.width;
            out.height = image.height;
        } else if (out.width != image.width || out.height != image.height) {
            throw std::runtime_error("ControlFoley video frame directory has inconsistent frame sizes");
        }
        DecodedFrame frame;
        frame.time = timestamps[i];
        frame.rgb = std::move(image.rgb);
        out.frames.push_back(std::move(frame));
    }
    return out;
}

}  // namespace

SourceVideo load_video_frames(const std::filesystem::path & path, double max_duration_sec) {
    if (!std::filesystem::exists(path)) {
        throw std::runtime_error("ControlFoley video path does not exist: " + path.string());
    }
    if (std::filesystem::is_directory(path)) {
        return load_frame_directory(path);
    }
    if (!std::filesystem::is_regular_file(path)) {
        throw std::runtime_error(
            "ControlFoley video path is neither a regular video file nor " + frame_directory_help() + ": " +
            path.string());
    }
    LibAvVideoDecoder decoder;
    return decoder.decode(path, max_duration_sec);
}

std::vector<int64_t> select_source_indices(
    const SourceVideo & video,
    float duration_sec,
    int64_t fps) {
    const int64_t expected_frames = static_cast<int64_t>(duration_sec * static_cast<float>(fps));
    if (expected_frames <= 0) {
        throw std::runtime_error("ControlFoley video is too short for the requested duration");
    }
    std::vector<int64_t> out;
    out.reserve(static_cast<size_t>(expected_frames));
    double next_timestamp = 0.0;
    const double frame_interval = 1.0 / static_cast<double>(fps);
    for (size_t index = 0; index < video.frames.size(); ++index) {
        while (video.frames[index].time >= next_timestamp && static_cast<int64_t>(out.size()) < expected_frames) {
            out.push_back(static_cast<int64_t>(index));
            next_timestamp += frame_interval;
        }
        if (static_cast<int64_t>(out.size()) >= expected_frames) {
            break;
        }
    }
    if (static_cast<int64_t>(out.size()) < expected_frames) {
        throw std::runtime_error("ControlFoley video is too short for the requested duration");
    }
    return out;
}

void write_normalized_frame(
    const std::vector<uint8_t> & rgb,
    int64_t written,
    int64_t size,
    const std::array<float, 3> & mean,
    const std::array<float, 3> & std,
    std::vector<float> & out) {
    for (int64_t y = 0; y < size; ++y) {
        for (int64_t x = 0; x < size; ++x) {
            for (int64_t c = 0; c < 3; ++c) {
                const int64_t src = (y * size + x) * 3 + c;
                const int64_t dst = ((written * 3 + c) * size + y) * size + x;
                const float value = static_cast<float>(rgb[static_cast<size_t>(src)]) / 255.0F;
                out[static_cast<size_t>(dst)] = (value - mean[static_cast<size_t>(c)]) / std[static_cast<size_t>(c)];
            }
        }
    }
}

ResizedFrameCache::ResizedFrameCache(
    const SourceVideo & video,
    int64_t size,
    bool center_crop)
    : video_(&video),
      size_(size),
      center_crop_(center_crop) {
    if (size_ <= 0) {
        throw std::runtime_error("ControlFoley video target size is invalid");
    }
}

std::vector<float> ResizedFrameCache::extract(
    float duration_sec,
    int64_t fps,
    const std::array<float, 3> & mean,
    const std::array<float, 3> & std,
    std::string_view trace_name) {
    std::vector<int64_t> indices;
    const double select_ms = engine::debug::measure_ms([&]() {
        indices = select_source_indices(*video_, duration_sec, fps);
    });
    std::vector<float> out(static_cast<size_t>(indices.size() * 3 * size_ * size_));
    const ResizePlan resize_plan(video_->width, video_->height, size_, center_crop_);
    double resize_ms = 0.0;
    double write_ms = 0.0;
    std::vector<int64_t> miss_indices;
    std::unordered_map<int64_t, size_t> planned_misses;
    planned_misses.reserve(indices.size());
    for (const auto index : indices) {
        if (cache_.find(index) == cache_.end() && planned_misses.find(index) == planned_misses.end()) {
            planned_misses.emplace(index, miss_indices.size());
            miss_indices.push_back(index);
        }
    }
    const int64_t cache_misses = static_cast<int64_t>(miss_indices.size());
    std::vector<ImageBuffer> resized_misses(miss_indices.size());
    resize_ms = engine::debug::measure_ms([&]() {
#ifdef _OPENMP
#pragma omp parallel for if(static_cast<int64_t>(miss_indices.size()) >= 8)
#endif
        for (int64_t miss = 0; miss < static_cast<int64_t>(miss_indices.size()); ++miss) {
            const int64_t index = miss_indices[static_cast<size_t>(miss)];
            resized_misses[static_cast<size_t>(miss)] =
                resize_plan.resize(video_->frames[static_cast<size_t>(index)].rgb);
        }
    });
    for (size_t miss = 0; miss < miss_indices.size(); ++miss) {
        cache_.emplace(miss_indices[miss], std::move(resized_misses[miss].rgb));
    }
    for (size_t written = 0; written < indices.size(); ++written) {
        const auto cached = cache_.find(indices[written]);
        if (cached == cache_.end()) {
            throw std::runtime_error("ControlFoley resized video frame cache miss after resize planning");
        }
        write_ms += engine::debug::measure_ms([&]() {
            write_normalized_frame(cached->second, static_cast<int64_t>(written), size_, mean, std, out);
        });
    }
    if (!trace_name.empty()) {
        const std::string prefix(trace_name);
        engine::debug::timing_log_scalar(prefix + ".select_ms", select_ms);
        engine::debug::timing_log_scalar(prefix + ".resize_ms", resize_ms);
        engine::debug::timing_log_scalar(prefix + ".write_ms", write_ms);
        engine::debug::timing_log_scalar(prefix + ".cache_misses", cache_misses);
    }
    return out;
}

std::vector<float> extract_video_frames(
    const SourceVideo & video,
    float duration_sec,
    int64_t fps,
    int64_t size,
    bool center_crop,
    const std::array<float, 3> & mean,
    const std::array<float, 3> & std) {
    ResizedFrameCache cache(video, size, center_crop);
    return cache.extract(duration_sec, fps, mean, std);
}

}  // namespace engine::models::controlfoley::video
