#include "engine/framework/audio/wav_reader.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void require(bool condition, const std::string & message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

template <typename T>
void write_le(std::ofstream & output, T value) {
    output.write(reinterpret_cast<const char *>(&value), sizeof(T));
    if (!output) {
        throw std::runtime_error("failed to write test WAV");
    }
}

void write_bytes(std::ofstream & output, const char * bytes, std::streamsize count) {
    output.write(bytes, count);
    if (!output) {
        throw std::runtime_error("failed to write test WAV");
    }
}

void write_pcm24_sample(std::ofstream & output, int32_t value) {
    const uint32_t bits = static_cast<uint32_t>(value) & 0x00FFFFFFu;
    const char bytes[3] = {
        static_cast<char>(bits & 0xFFu),
        static_cast<char>((bits >> 8) & 0xFFu),
        static_cast<char>((bits >> 16) & 0xFFu),
    };
    write_bytes(output, bytes, 3);
}

void write_pcm24_wav(
    const std::filesystem::path & path,
    int sample_rate,
    int channels,
    const std::vector<int32_t> & samples) {
    const uint32_t data_bytes = static_cast<uint32_t>(samples.size() * 3);
    const uint16_t block_align = static_cast<uint16_t>(channels * 3);
    const uint32_t byte_rate = static_cast<uint32_t>(sample_rate * block_align);

    std::ofstream output(path, std::ios::binary);
    if (!output) {
        throw std::runtime_error("failed to open test WAV: " + path.string());
    }
    write_bytes(output, "RIFF", 4);
    write_le<uint32_t>(output, 36u + data_bytes);
    write_bytes(output, "WAVE", 4);
    write_bytes(output, "fmt ", 4);
    write_le<uint32_t>(output, 16u);
    write_le<uint16_t>(output, 1u);
    write_le<uint16_t>(output, static_cast<uint16_t>(channels));
    write_le<uint32_t>(output, static_cast<uint32_t>(sample_rate));
    write_le<uint32_t>(output, byte_rate);
    write_le<uint16_t>(output, block_align);
    write_le<uint16_t>(output, 24u);
    write_bytes(output, "data", 4);
    write_le<uint32_t>(output, data_bytes);
    for (const int32_t sample : samples) {
        write_pcm24_sample(output, sample);
    }
}

// G.711 decode tables, all 256 codes each. Ground truth from outside this
// codebase: ffmpeg 9.0.1 decoding a 256-byte file of every code, matching the
// values implied by the ITU-T G.711 segment definitions.
//
// Every expected value in this file must come from an external decoder, never
// from the arithmetic in wav_reader.cpp. The two codings do not share a sign
// convention -- mu-law sets the top bit for negative samples, A-law for
// positive ones -- and a table derived from the implementation cannot see that.
constexpr int16_t kALawExpected[256] = {
     -5504,  -5248,  -6016,  -5760,  -4480,  -4224,  -4992,  -4736,
     -7552,  -7296,  -8064,  -7808,  -6528,  -6272,  -7040,  -6784,
     -2752,  -2624,  -3008,  -2880,  -2240,  -2112,  -2496,  -2368,
     -3776,  -3648,  -4032,  -3904,  -3264,  -3136,  -3520,  -3392,
    -22016, -20992, -24064, -23040, -17920, -16896, -19968, -18944,
    -30208, -29184, -32256, -31232, -26112, -25088, -28160, -27136,
    -11008, -10496, -12032, -11520,  -8960,  -8448,  -9984,  -9472,
    -15104, -14592, -16128, -15616, -13056, -12544, -14080, -13568,
      -344,   -328,   -376,   -360,   -280,   -264,   -312,   -296,
      -472,   -456,   -504,   -488,   -408,   -392,   -440,   -424,
       -88,    -72,   -120,   -104,    -24,     -8,    -56,    -40,
      -216,   -200,   -248,   -232,   -152,   -136,   -184,   -168,
     -1376,  -1312,  -1504,  -1440,  -1120,  -1056,  -1248,  -1184,
     -1888,  -1824,  -2016,  -1952,  -1632,  -1568,  -1760,  -1696,
      -688,   -656,   -752,   -720,   -560,   -528,   -624,   -592,
      -944,   -912,  -1008,   -976,   -816,   -784,   -880,   -848,
      5504,   5248,   6016,   5760,   4480,   4224,   4992,   4736,
      7552,   7296,   8064,   7808,   6528,   6272,   7040,   6784,
      2752,   2624,   3008,   2880,   2240,   2112,   2496,   2368,
      3776,   3648,   4032,   3904,   3264,   3136,   3520,   3392,
     22016,  20992,  24064,  23040,  17920,  16896,  19968,  18944,
     30208,  29184,  32256,  31232,  26112,  25088,  28160,  27136,
     11008,  10496,  12032,  11520,   8960,   8448,   9984,   9472,
     15104,  14592,  16128,  15616,  13056,  12544,  14080,  13568,
       344,    328,    376,    360,    280,    264,    312,    296,
       472,    456,    504,    488,    408,    392,    440,    424,
        88,     72,    120,    104,     24,      8,     56,     40,
       216,    200,    248,    232,    152,    136,    184,    168,
      1376,   1312,   1504,   1440,   1120,   1056,   1248,   1184,
      1888,   1824,   2016,   1952,   1632,   1568,   1760,   1696,
       688,    656,    752,    720,    560,    528,    624,    592,
       944,    912,   1008,    976,    816,    784,    880,    848,
};

constexpr int16_t kMuLawExpected[256] = {
    -32124, -31100, -30076, -29052, -28028, -27004, -25980, -24956,
    -23932, -22908, -21884, -20860, -19836, -18812, -17788, -16764,
    -15996, -15484, -14972, -14460, -13948, -13436, -12924, -12412,
    -11900, -11388, -10876, -10364,  -9852,  -9340,  -8828,  -8316,
     -7932,  -7676,  -7420,  -7164,  -6908,  -6652,  -6396,  -6140,
     -5884,  -5628,  -5372,  -5116,  -4860,  -4604,  -4348,  -4092,
     -3900,  -3772,  -3644,  -3516,  -3388,  -3260,  -3132,  -3004,
     -2876,  -2748,  -2620,  -2492,  -2364,  -2236,  -2108,  -1980,
     -1884,  -1820,  -1756,  -1692,  -1628,  -1564,  -1500,  -1436,
     -1372,  -1308,  -1244,  -1180,  -1116,  -1052,   -988,   -924,
      -876,   -844,   -812,   -780,   -748,   -716,   -684,   -652,
      -620,   -588,   -556,   -524,   -492,   -460,   -428,   -396,
      -372,   -356,   -340,   -324,   -308,   -292,   -276,   -260,
      -244,   -228,   -212,   -196,   -180,   -164,   -148,   -132,
      -120,   -112,   -104,    -96,    -88,    -80,    -72,    -64,
       -56,    -48,    -40,    -32,    -24,    -16,     -8,      0,
     32124,  31100,  30076,  29052,  28028,  27004,  25980,  24956,
     23932,  22908,  21884,  20860,  19836,  18812,  17788,  16764,
     15996,  15484,  14972,  14460,  13948,  13436,  12924,  12412,
     11900,  11388,  10876,  10364,   9852,   9340,   8828,   8316,
      7932,   7676,   7420,   7164,   6908,   6652,   6396,   6140,
      5884,   5628,   5372,   5116,   4860,   4604,   4348,   4092,
      3900,   3772,   3644,   3516,   3388,   3260,   3132,   3004,
      2876,   2748,   2620,   2492,   2364,   2236,   2108,   1980,
      1884,   1820,   1756,   1692,   1628,   1564,   1500,   1436,
      1372,   1308,   1244,   1180,   1116,   1052,    988,    924,
       876,    844,    812,    780,    748,    716,    684,    652,
       620,    588,    556,    524,    492,    460,    428,    396,
       372,    356,    340,    324,    308,    292,    276,    260,
       244,    228,    212,    196,    180,    164,    148,    132,
       120,    112,    104,     96,     88,     80,     72,     64,
        56,     48,     40,     32,     24,     16,      8,      0,
};

void require_near(float actual, float expected, const std::string & label) {
    if (std::fabs(actual - expected) > 1.0e-7F) {
        throw std::runtime_error(label + " mismatch");
    }
}

// Writes a fmt chunk of `format_tag`/`bits` plus a data chunk of raw bytes. When
// `extensible` is set the chunk is the 40-byte WAVEFORMATEXTENSIBLE layout and
// `format_tag` moves into the SubFormat GUID, exactly as encoders emit it for
// multichannel or channel-masked PCM.
void write_wav(
    const std::filesystem::path & path,
    uint16_t format_tag,
    uint16_t bits,
    int sample_rate,
    int channels,
    const std::vector<char> & payload,
    bool extensible = false,
    bool valid_guid_tail = true,
    uint16_t cb_size = 22) {
    const uint16_t block_align = static_cast<uint16_t>(channels * ((bits + 7) / 8));
    const uint32_t byte_rate = static_cast<uint32_t>(sample_rate) * block_align;
    const uint32_t data_bytes = static_cast<uint32_t>(payload.size());
    const uint32_t fmt_bytes = extensible ? 40u : 16u;

    std::ofstream output(path, std::ios::binary);
    if (!output) {
        throw std::runtime_error("failed to open test WAV: " + path.string());
    }
    write_bytes(output, "RIFF", 4);
    write_le<uint32_t>(output, 20u + fmt_bytes + data_bytes);
    write_bytes(output, "WAVE", 4);
    write_bytes(output, "fmt ", 4);
    write_le<uint32_t>(output, fmt_bytes);
    write_le<uint16_t>(output, extensible ? uint16_t{0xFFFE} : format_tag);
    write_le<uint16_t>(output, static_cast<uint16_t>(channels));
    write_le<uint32_t>(output, static_cast<uint32_t>(sample_rate));
    write_le<uint32_t>(output, byte_rate);
    write_le<uint16_t>(output, block_align);
    write_le<uint16_t>(output, bits);
    if (extensible) {
        write_le<uint16_t>(output, cb_size);           // cbSize
        write_le<uint16_t>(output, bits);              // wValidBitsPerSample
        write_le<uint32_t>(output, uint32_t{0x3});     // dwChannelMask
        write_le<uint16_t>(output, format_tag);        // SubFormat GUID, first field
        // Remainder of KSDATAFORMAT_SUBTYPE_*: 0000-0010-8000-00aa00389b71
        const char valid_tail[14] = {
            0x00, 0x00, 0x00, 0x00, 0x10, 0x00, static_cast<char>(0x80),
            0x00, 0x00, static_cast<char>(0xAA), 0x00, 0x38, static_cast<char>(0x9B),
            0x71,
        };
        const char foreign_tail[14] = {
            static_cast<char>(0xBE), static_cast<char>(0xEF), static_cast<char>(0xDE),
            static_cast<char>(0xAD), 0x42, 0x41, 0x44, 0x47, 0x55, 0x49, 0x44, 0x21,
            0x00, 0x00,
        };
        write_bytes(output, valid_guid_tail ? valid_tail : foreign_tail, 14);
    }
    write_bytes(output, "data", 4);
    write_le<uint32_t>(output, data_bytes);
    if (data_bytes > 0) {
        write_bytes(output, payload.data(), static_cast<std::streamsize>(data_bytes));
    }
}

template <typename T>
std::vector<char> to_bytes(const std::vector<T> & values) {
    std::vector<char> out(values.size() * sizeof(T));
    if (!values.empty()) {
        std::memcpy(out.data(), values.data(), out.size());
    }
    return out;
}

void require_near(float actual, float expected, float tolerance, const std::string & label) {
    if (!std::isfinite(actual)) {
        throw std::runtime_error(label + " is not finite");
    }
    if (std::fabs(actual - expected) > tolerance) {
        throw std::runtime_error(label + " mismatch");
    }
}

// Runs `body` and requires it to throw with `needle` in the message. A silent
// success here would mean the reader accepted something it cannot decode.
void require_throws_containing(
    const std::function<void()> & body, const std::string & needle, const std::string & label) {
    try {
        body();
    } catch (const std::exception & ex) {
        if (std::string(ex.what()).find(needle) == std::string::npos) {
            throw std::runtime_error(label + ": wrong message: " + ex.what());
        }
        return;
    }
    throw std::runtime_error(label + ": expected a throw, got none");
}

}  // namespace

int main() {
    try {
        const auto root = std::filesystem::temp_directory_path() / "audio_cpp_wav_reader_test";
        std::filesystem::remove_all(root);
        std::filesystem::create_directories(root);
        const auto path = root / "pcm24_stereo.wav";
        write_pcm24_wav(
            path,
            48000,
            2,
            {
                0,
                0x007FFFFF,
                -0x00800000,
                -1,
            });

        const auto wav = engine::audio::read_wav_f32(path);
        require(wav.sample_rate == 48000, "PCM24 sample rate mismatch");
        require(wav.channels == 2, "PCM24 channel count mismatch");
        require(wav.samples.size() == 4, "PCM24 sample count mismatch");
        require_near(wav.samples[0], 0.0F, "PCM24 zero");
        require_near(wav.samples[1], 8388607.0F / 8388608.0F, "PCM24 max positive");
        require_near(wav.samples[2], -1.0F, "PCM24 min negative");
        require_near(wav.samples[3], -1.0F / 8388608.0F, "PCM24 negative one");

        // --- WAVEFORMATEXTENSIBLE wrapping ordinary PCM16 ---------------------
        // The case that actually bites: the payload is plain PCM16, but the
        // format tag says 0xFFFE and the real tag lives in the SubFormat GUID.
        // A reader that stops at the tag rejects a file it can decode.
        {
            const auto ext = root / "extensible_pcm16.wav";
            write_wav(ext, 0x0001, 16, 44100, 2,
                      to_bytes<int16_t>({0, 16384, -16384, -32768}), true);
            const auto wav = engine::audio::read_wav_f32(ext);
            require(wav.sample_rate == 44100, "EXTENSIBLE PCM16 sample rate mismatch");
            require(wav.channels == 2, "EXTENSIBLE PCM16 channel count mismatch");
            require(wav.samples.size() == 4, "EXTENSIBLE PCM16 sample count mismatch");
            require_near(wav.samples[0], 0.0F, 1.0e-7F, "EXTENSIBLE PCM16 zero");
            require_near(wav.samples[1], 0.5F, 1.0e-7F, "EXTENSIBLE PCM16 half");
            require_near(wav.samples[2], -0.5F, 1.0e-7F, "EXTENSIBLE PCM16 negative half");
            require_near(wav.samples[3], -1.0F, 1.0e-7F, "EXTENSIBLE PCM16 full negative");
        }

        // --- WAVEFORMATEXTENSIBLE wrapping float32 ---------------------------
        // Proves the GUID is actually read rather than assumed to be PCM.
        {
            const auto ext = root / "extensible_f32.wav";
            write_wav(ext, 0x0003, 32, 48000, 1,
                      to_bytes<float>({0.0F, 0.25F, -0.75F}), true);
            const auto wav = engine::audio::read_wav_f32(ext);
            require(wav.channels == 1, "EXTENSIBLE float32 channel count mismatch");
            require(wav.samples.size() == 3, "EXTENSIBLE float32 sample count mismatch");
            require_near(wav.samples[1], 0.25F, 1.0e-7F, "EXTENSIBLE float32 quarter");
            require_near(wav.samples[2], -0.75F, 1.0e-7F, "EXTENSIBLE float32 negative");
        }

        // --- PCM8 is unsigned, biased by 128 ---------------------------------
        // The sign convention differs from every other PCM width, so a decoder
        // that treats it as signed silently inverts the waveform.
        {
            const auto path8 = root / "pcm8.wav";
            write_wav(path8, 0x0001, 8, 8000, 1,
                      std::vector<char>{static_cast<char>(128), static_cast<char>(255),
                                        static_cast<char>(0), static_cast<char>(64),
                                        static_cast<char>(1)});
            const auto wav = engine::audio::read_wav_f32(path8);
            require(wav.samples.size() == 5, "PCM8 sample count mismatch");
            // Frozen from `ffmpeg -i pcm8.wav -f f32le -`. Note the endpoints are
            // asymmetric: 255 is 127/128, not 1.0. ffmpeg agrees.
            const float expected8[5] = {0.0F, 0.9921875F, -1.0F, -0.5F, -0.9921875F};
            for (int i = 0; i < 5; ++i) {
                require_near(wav.samples[static_cast<size_t>(i)], expected8[i], 1.0e-7F,
                             "PCM8 sample " + std::to_string(i));
            }
        }

        // --- PCM32 -----------------------------------------------------------
        {
            const auto path32 = root / "pcm32.wav";
            write_wav(path32, 0x0001, 32, 96000, 1,
                      to_bytes<int32_t>({0, 1073741824, -2147483647 - 1, 2147483647, 1}));
            const auto wav = engine::audio::read_wav_f32(path32);
            require(wav.sample_rate == 96000, "PCM32 sample rate mismatch");
            require(wav.samples.size() == 5, "PCM32 sample count mismatch");
            // Frozen from ffmpeg. INT32_MAX lands on exactly 1.0 rather than
            // 2147483647/2^31, because float32's ULP at that magnitude is 256 and
            // the cast rounds up before the divide. ffmpeg does the same.
            const float expected32[5] = {0.0F, 0.5F, -1.0F, 1.0F, 4.656612873077393e-10F};
            for (int i = 0; i < 5; ++i) {
                require_near(wav.samples[static_cast<size_t>(i)], expected32[i], 1.0e-12F,
                             "PCM32 sample " + std::to_string(i));
            }
        }

        // --- float64 ---------------------------------------------------------
        {
            const auto path64 = root / "float64.wav";
            write_wav(path64, 0x0003, 64, 44100, 2,
                      to_bytes<double>({0.0, 0.125, -0.875, 1.0, 1.0 + 0x1p-30}));
            const auto wav = engine::audio::read_wav_f32(path64);
            require(wav.channels == 2, "float64 channel count mismatch");
            require(wav.samples.size() == 5, "float64 sample count mismatch");
            // Frozen from ffmpeg. The last one is deliberately not representable
            // in float32 and collapses to 1.0 on both sides.
            const float expected64[5] = {0.0F, 0.125F, -0.875F, 1.0F, 1.0F};
            for (int i = 0; i < 5; ++i) {
                require_near(wav.samples[static_cast<size_t>(i)], expected64[i], 1.0e-7F,
                             "float64 sample " + std::to_string(i));
            }
        }

        // --- G.711 mu-law and A-law, every code ---------------------------
        // One byte per code, decoded in one pass and compared against the
        // external tables above. A spot check cannot catch a whole-table sign
        // inversion; this can.
        {
            std::vector<char> codes(256);
            for (int i = 0; i < 256; ++i) {
                codes[static_cast<size_t>(i)] = static_cast<char>(i);
            }

            const auto path_mu = root / "mulaw_all.wav";
            write_wav(path_mu, 0x0007, 8, 8000, 1, codes);
            const auto mu = engine::audio::read_wav_f32(path_mu);
            require(mu.samples.size() == 256, "mu-law sample count mismatch");
            for (int i = 0; i < 256; ++i) {
                require_near(
                    mu.samples[static_cast<size_t>(i)],
                    static_cast<float>(kMuLawExpected[i]) / 32768.0F,
                    1.0e-7F,
                    "mu-law code " + std::to_string(i));
            }

            const auto path_a = root / "alaw_all.wav";
            write_wav(path_a, 0x0006, 8, 8000, 1, codes);
            const auto alaw = engine::audio::read_wav_f32(path_a);
            require(alaw.samples.size() == 256, "A-law sample count mismatch");
            for (int i = 0; i < 256; ++i) {
                require_near(
                    alaw.samples[static_cast<size_t>(i)],
                    static_cast<float>(kALawExpected[i]) / 32768.0F,
                    1.0e-7F,
                    "A-law code " + std::to_string(i));
            }
        }

        // --- Extensible headers are validated, not trusted -------------------
        // Only the first two bytes of the SubFormat GUID are the format tag. The
        // other fourteen are a fixed suffix; without checking them, any codec
        // whose GUID happens to start with 0x0001 decodes as PCM.
        {
            const auto bad_guid = root / "extensible_foreign_guid.wav";
            write_wav(bad_guid, 0x0001, 16, 44100, 1,
                      to_bytes<int16_t>({0, 16384}), true, /*valid_guid_tail=*/false);
            require_throws_containing(
                [&] { (void)engine::audio::read_wav_f32(bad_guid); },
                "KSDATAFORMAT_SUBTYPE", "foreign SubFormat GUID rejection");

            // WAVEFORMATEXTENSIBLE requires cbSize >= 22; anything less is an
            // internally inconsistent header.
            const auto bad_cb = root / "extensible_bad_cbsize.wav";
            write_wav(bad_cb, 0x0001, 16, 44100, 1,
                      to_bytes<int16_t>({0, 16384}), true, true, /*cb_size=*/0);
            require_throws_containing(
                [&] { (void)engine::audio::read_wav_f32(bad_cb); },
                "cbSize", "extensible cbSize rejection");
        }

        // --- Truncated sample data is an error, not a silent trim ------------
        // PCM24 already rejected a partial trailing sample; PCM32 and float64
        // divided and dropped it, which turns a truncated download into audio
        // that looks fine.
        {
            auto pcm32 = to_bytes<int32_t>({1073741824});
            pcm32.push_back(0x7F);
            const auto path32 = root / "pcm32_partial.wav";
            write_wav(path32, 0x0001, 32, 44100, 1, pcm32);
            require_throws_containing(
                [&] { (void)engine::audio::read_wav_f32(path32); },
                "malformed PCM32", "PCM32 partial sample rejection");

            auto f64 = to_bytes<double>({0.25});
            f64.push_back(static_cast<char>(0xAA));
            const auto path64 = root / "float64_partial.wav";
            write_wav(path64, 0x0003, 64, 44100, 1, f64);
            require_throws_containing(
                [&] { (void)engine::audio::read_wav_f32(path64); },
                "malformed float64", "float64 partial sample rejection");
        }

        // --- All three overloads agree, including on a missing final pad -----
        // RIFF pads odd-sized chunks, but writers routinely omit the byte when
        // the chunk ends the file, and one-byte-per-sample formats make odd data
        // chunks common. The path overload tolerated this because seeking past
        // EOF is legal on an ifstream; the in-memory overload did not, so the
        // same bytes parsed from a file and rejected from an upload buffer.
        {
            const auto odd = root / "pcm8_odd_no_pad.wav";
            write_wav(odd, 0x0001, 8, 8000, 1,
                      std::vector<char>{static_cast<char>(0), static_cast<char>(128),
                                        static_cast<char>(254)});
            std::ifstream in(odd, std::ios::binary);
            const std::string blob((std::istreambuf_iterator<char>(in)),
                                   std::istreambuf_iterator<char>());
            require(blob.size() % 2 == 1, "test file should end without the pad byte");

            const auto from_path = engine::audio::read_wav_f32(odd);
            const auto from_memory = engine::audio::read_wav_f32(std::string_view(blob));
            require(from_path.samples.size() == 3, "odd-size PCM8 sample count from path");
            require(from_memory.samples.size() == 3, "odd-size PCM8 sample count from memory");
            for (size_t i = 0; i < from_path.samples.size(); ++i) {
                require_near(from_memory.samples[i], from_path.samples[i], 0.0F,
                             "path and memory overloads disagree at " + std::to_string(i));
            }

            // The istream overload must not assume the stream began at offset 0,
            // which an absolute rewind after the header sniff would.
            std::istringstream prefixed(std::string("PREFIX!!") + blob, std::ios::binary);
            prefixed.seekg(8);
            const auto from_stream = engine::audio::read_wav_f32(prefixed);
            require(from_stream.samples.size() == 3, "offset istream sample count");
            require_near(from_stream.samples[0], from_path.samples[0], 0.0F,
                         "offset istream disagrees with path");
        }

        // --- A short fmt chunk is an error, not a read into the next chunk ----
        {
            const auto short_fmt = root / "short_fmt.wav";
            {
                std::ofstream out(short_fmt, std::ios::binary);
                const std::vector<char> payload = to_bytes<int16_t>({0, 16384});
                write_bytes(out, "RIFF", 4);
                write_le<uint32_t>(out, 20u + 14u + static_cast<uint32_t>(payload.size()));
                write_bytes(out, "WAVE", 4);
                write_bytes(out, "fmt ", 4);
                write_le<uint32_t>(out, 14u);          // one field short of the minimum
                write_le<uint16_t>(out, uint16_t{1});
                write_le<uint16_t>(out, uint16_t{1});
                write_le<uint32_t>(out, uint32_t{44100});
                write_le<uint32_t>(out, uint32_t{88200});
                write_le<uint16_t>(out, uint16_t{2});
                write_bytes(out, "data", 4);
                write_le<uint32_t>(out, static_cast<uint32_t>(payload.size()));
                write_bytes(out, payload.data(), static_cast<std::streamsize>(payload.size()));
            }
            require_throws_containing(
                [&] { (void)engine::audio::read_wav_f32(short_fmt); },
                "malformed WAV fmt chunk", "short fmt chunk rejection");
        }

        // --- Still rejects what it genuinely cannot decode -------------------
        // Widening the accepted set must not turn into accepting everything.
        {
            const auto path_bad = root / "unsupported.wav";
            write_wav(path_bad, 0x0011, 4, 8000, 1, std::vector<char>{0x01, 0x02});
            require_throws_containing(
                [&] { (void)engine::audio::read_wav_f32(path_bad); },
                "unsupported WAV encoding", "ADPCM rejection");

            const auto path_flac = root / "actually.flac";
            {
                std::ofstream output(path_flac, std::ios::binary);
                write_bytes(output, "fLaC\0\0\0\x22\0\0\0\0", 12);
            }
            require_throws_containing(
                [&] { (void)engine::audio::read_wav_f32(path_flac); },
                "FLAC", "FLAC container identification");
        }

        std::cout << "wav_reader_test passed\n";
    } catch (const std::exception & ex) {
        std::cerr << "wav_reader_test failed: " << ex.what() << "\n";
        return 1;
    }
    return 0;
}
