// LFM2.5-Audio TTS on Liquid's published GGUFs against liquid-audio 1.3.0
// (LFM2.5-Audio-1.5B revision c362a0625dfe45aa588dce5f0ada28a7e5707628,
// fp32), for "What is this obsession people have with books?" in the UK male
// voice:
// - the prompt ChatState builds;
// - the backbone output that feeds the depthformer, the first frame's
//   depthformer logits, the greedy frames, and the detokenizer's head output
//   and waveform for the reference frames;
// - end-to-end speech through the registry, transcribed back by the ASR task,
//   and streamed;
// - speech cut off at max_tokens, offline and streamed.
//
// --model is the directory of LiquidAI/LFM2.5-Audio-1.5B-GGUF (default
// models/LFM2.5-Audio-1.5B-GGUF, where the lfm2_audio_1_5b_* packages install);
// --model-gguf picks the backbone, and the vocoder- and tokenizer- files of
// the same name come with it. Skips with 125 when the files are not there.
#include "engine/community_models/lfm2_audio/assets.h"
#include "engine/community_models/lfm2_audio/backbone.h"
#include "engine/community_models/lfm2_audio/depthformer.h"
#include "engine/community_models/lfm2_audio/detokenizer.h"
#include "engine/community_models/lfm2_audio/session.h"
#include "engine/community_models/lfm2_audio/tokenizer.h"
#include "engine/community_models/lfm2_audio/tts.h"
#include "engine/framework/core/execution_context.h"
#include "engine/framework/io/filesystem.h"
#include "engine/framework/runtime/model.h"
#include "engine/framework/runtime/registry.h"
#include "engine/framework/runtime/session.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <functional>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#ifndef ENGINE_REPO_ROOT
#define ENGINE_REPO_ROOT "."
#endif

namespace {

namespace lfm2 = engine::community_models::lfm2_audio;

constexpr int kExitPass = 0;
constexpr int kExitFail = 1;
constexpr int kExitSkip = 125;

constexpr const char * kText = "What is this obsession people have with books?";
constexpr const char * kVoice = "uk_male";
constexpr const char * kSampledSeeds[] = {"1234", "1235", "1236"};

const std::vector<int32_t> kPromptIds = {1, 6, 24131, 708, 8173, 1199, 837, 10255, 523, 8146, 779, 6729, 6541, 8403, 523, 7, 708, 6,
                                         6423, 708, 3493, 856, 1033, 18619, 1746, 1519, 1052, 916, 5936, 540, 7, 708, 6, 64015, 708};

struct Point {
    int64_t index;
    float value;
};

// Backbone output after <|audio_start|>, the depthformer's input.
const Point kHiddenPoints[] = {{0, -0.218750f}, {17, 0.215646f}, {511, 0.034909f}, {1024, -0.006849f}, {1500, 0.079719f}, {2047, -0.123280f}};

// First frame: each codebook's greedy code and its logit.
struct CodebookTop {
    int32_t code;
    float logit;
};

const CodebookTop kFirstFrame[] = {
    {1049, 19.91094f}, {477, 15.77336f}, {1626, 20.17520f}, {142, 17.97590f},
    {1335, 17.54575f}, {555, 16.09092f}, {976, 17.17711f}, {1648, 17.43196f},
};

// generate_sequential's greedy frames; the next frame is end-of-audio.
const std::vector<std::vector<int32_t>> kFrames = {
    {1049, 477, 1626, 142, 1335, 555, 976, 1648},   {127, 1056, 1697, 290, 481, 1443, 976, 1744},
    {1880, 1056, 1178, 290, 1736, 1443, 666, 1744},  {1031, 1056, 1178, 290, 1736, 1443, 825, 1648},
    {1156, 1262, 437, 1999, 3, 945, 778, 1944},      {167, 38, 410, 346, 2015, 1074, 1625, 1796},
    {460, 792, 1503, 1121, 1980, 975, 437, 241},     {1251, 1278, 1952, 1908, 1653, 633, 1185, 1620},
    {330, 1837, 1245, 657, 1148, 338, 1140, 993},    {1237, 274, 51, 438, 1038, 1748, 386, 1211},
    {1273, 1265, 1385, 1384, 1894, 933, 1087, 2020}, {1134, 1816, 242, 1279, 1791, 137, 449, 625},
    {252, 2019, 1711, 30, 656, 700, 1579, 1599},     {1840, 1724, 190, 251, 1300, 1981, 1890, 1003},
    {1517, 1572, 2033, 105, 19, 468, 2014, 1925},    {985, 875, 402, 1323, 1951, 295, 493, 388},
    {1540, 1311, 579, 31, 734, 1341, 1951, 1135},    {1490, 1549, 402, 135, 582, 1298, 552, 1466},
    {711, 67, 1195, 1170, 1207, 28, 1516, 27},       {376, 1118, 2033, 1156, 774, 1511, 328, 1373},
    {1849, 1332, 550, 1922, 604, 319, 473, 1211},    {622, 1014, 1443, 245, 17, 1496, 1418, 471},
    {9, 1520, 1442, 1673, 1067, 1152, 1955, 1529},   {680, 1132, 1659, 1602, 1000, 246, 1989, 962},
    {571, 560, 2001, 241, 102, 1922, 1604, 306},     {619, 1663, 1559, 1276, 942, 2010, 1520, 114},
    {2033, 1651, 1321, 458, 647, 829, 1039, 1712},   {205, 921, 2001, 2040, 1837, 1170, 513, 1628},
    {1331, 1515, 1535, 1348, 420, 1273, 976, 1744},  {1743, 1056, 1559, 546, 481, 1443, 825, 1744},
    {356, 1056, 1559, 164, 267, 1443, 976, 1744},    {868, 818, 323, 290, 306, 1030, 1238, 1648},
};

// For each codebook of kFrames, how far the reference's best logit is above
// its runner-up.
const float kFrameGaps[][8] = {
    {8.7497f, 0.1983f, 3.7671f, 1.5620f, 0.6233f, 0.2551f, 1.4607f, 1.9044f},
    {5.9887f, 4.4497f, 1.1002f, 2.8774f, 0.6214f, 0.4310f, 0.0466f, 4.1133f},
    {6.2334f, 3.6723f, 0.0738f, 1.1140f, 0.9030f, 2.8178f, 1.5583f, 0.8674f},
    {1.0494f, 2.5581f, 3.3568f, 0.2945f, 1.2743f, 1.9717f, 1.6806f, 1.5090f},
    {1.5158f, 1.0875f, 0.3224f, 0.9291f, 0.1325f, 1.7659f, 0.2750f, 0.9055f},
    {3.9284f, 0.3964f, 0.2994f, 0.3338f, 1.9245f, 0.6460f, 0.6082f, 1.1388f},
    {5.7928f, 1.8549f, 0.8458f, 1.2390f, 0.8121f, 0.4359f, 0.9546f, 0.3444f},
    {0.1795f, 0.0528f, 0.4219f, 1.1724f, 0.0565f, 1.2440f, 2.3549f, 0.3909f},
    {3.2926f, 1.5977f, 3.2096f, 0.0778f, 0.2610f, 1.3376f, 0.0540f, 0.2369f},
    {0.4930f, 0.6255f, 1.1905f, 0.6213f, 0.5908f, 0.3480f, 0.6047f, 0.0497f},
    {0.9354f, 0.3649f, 0.3207f, 0.0172f, 0.1111f, 1.0805f, 0.3474f, 0.0274f},
    {3.0147f, 0.7361f, 0.3225f, 0.2281f, 1.1174f, 0.5528f, 1.0709f, 0.8812f},
    {2.0617f, 1.5264f, 3.7476f, 0.3078f, 0.6811f, 0.2404f, 0.3665f, 0.5121f},
    {3.2047f, 1.2930f, 1.3553f, 0.7308f, 0.0038f, 0.6648f, 0.3533f, 0.2499f},
    {2.2267f, 0.3904f, 0.0553f, 0.9773f, 0.8412f, 0.8955f, 0.9344f, 0.7625f},
    {2.3502f, 0.2537f, 1.4494f, 1.6078f, 0.1209f, 1.4134f, 0.2676f, 0.5587f},
    {1.5207f, 0.4524f, 0.4150f, 1.3518f, 2.6995f, 0.5586f, 0.1481f, 0.8708f},
    {1.0083f, 0.3082f, 0.7947f, 0.1522f, 0.3049f, 0.9793f, 0.7531f, 0.0563f},
    {0.9020f, 0.6105f, 0.7749f, 1.1957f, 0.4004f, 0.5589f, 0.2698f, 0.2589f},
    {3.7376f, 1.6121f, 0.4297f, 0.4866f, 1.0517f, 0.2149f, 0.4426f, 1.6593f},
    {4.4777f, 0.0626f, 0.7395f, 0.1156f, 0.2681f, 0.1772f, 2.7115f, 4.8412f},
    {3.3083f, 0.1107f, 0.7155f, 1.0171f, 0.3505f, 2.3678f, 0.1161f, 0.0090f},
    {2.3274f, 0.4537f, 0.0043f, 0.0585f, 1.0131f, 0.8113f, 1.2763f, 1.3800f},
    {2.1885f, 0.5628f, 0.1090f, 0.2182f, 0.6953f, 0.0006f, 0.3562f, 0.1547f},
    {2.1961f, 1.1076f, 0.1012f, 0.1799f, 0.8490f, 0.5019f, 0.4110f, 0.4656f},
    {3.2486f, 0.0501f, 0.1690f, 2.2923f, 0.2975f, 0.4103f, 1.0631f, 0.0793f},
    {5.0210f, 1.1487f, 1.6144f, 0.6327f, 0.3195f, 0.2209f, 1.1444f, 0.2733f},
    {4.7366f, 1.3775f, 0.1301f, 0.5289f, 0.6588f, 1.3056f, 0.8659f, 0.0615f},
    {7.1659f, 4.2903f, 1.9392f, 0.5934f, 3.1354f, 2.5204f, 0.2127f, 0.1226f},
    {0.4853f, 4.4536f, 1.1995f, 2.5165f, 3.1818f, 1.3993f, 1.7719f, 4.6741f},
    {0.7524f, 0.0676f, 0.4212f, 0.9022f, 2.0683f, 3.7320f, 0.5490f, 0.0294f},
    {0.8000f, 1.4172f, 0.3545f, 0.7020f, 0.5260f, 0.8285f, 0.0075f, 0.2785f},
};

// Detokenizer output for kFrames: head [row][column] and waveform samples.
struct HeadPoint {
    int64_t row;
    int64_t column;
    float value;
};

const HeadPoint kHeadPoints[] = {{0, 0, -1.36464f}, {10, 100, -5.50510f}, {50, 640, -1.56862f},
                                 {100, 641, -6.03843f}, {150, 900, -21.30030f}, {191, 1281, -0.13237f}};
const Point kWavePoints[] = {{1000, 0.000134f}, {12345, 0.028856f}, {30000, -0.001946f}, {45678, 0.052879f}, {60000, 0.000039f}};
constexpr size_t kWaveSamples = 61440;
constexpr double kWaveRms = 0.069406;

// A greedy frame may differ from the reference's only in codebooks where the
// two codes' logits are closer than this and so are the reference's top two;
// past such a near-tie the weights' rounding decides, so the frames after it
// are not compared.
constexpr float kNearTie = 0.05f;

std::filesystem::path repo_path(const std::string & relative) {
    return std::filesystem::path(ENGINE_REPO_ROOT) / relative;
}

std::string arg_value(int argc, char ** argv, const std::string & name, const std::string & fallback) {
    for (int i = 1; i + 1 < argc; ++i) {
        if (argv[i] == name) {
            return argv[i + 1];
        }
    }

    return fallback;
}

std::string join(const std::vector<int32_t> & ids) {
    std::ostringstream out;
    for (size_t i = 0; i < ids.size(); ++i) {
        out << (i == 0 ? "" : ",") << ids[i];
    }

    return out.str();
}

class Checks {
public:
    void expect(bool ok, const std::string & label, const std::string & detail = "") {
        std::cout << (ok ? "PASS " : "FAIL ") << label << (ok || detail.empty() ? "" : ": " + detail) << "\n";
        failures_ += ok ? 0 : 1;
    }

    void expect_close(double actual, double expected, double tolerance, const std::string & label) {
        std::ostringstream detail;
        detail << label << ": got " << actual << ", expected " << expected << " +- " << tolerance;
        expect(std::fabs(actual - expected) <= tolerance, detail.str());
    }

    [[nodiscard]] int failures() const { return failures_; }

private:
    int failures_ = 0;
};

void check_prompt(const lfm2::Lfm2AudioComponents & components, Checks & checks) {
    const lfm2::Lfm2TextTokenizer tokenizer(components.vocabulary);
    const auto prompt = lfm2::make_lfm2_tts_prompt(tokenizer, lfm2::lfm2_tts_system_prompt("en", kVoice), kText);
    checks.expect(prompt.input_ids == kPromptIds, "TTS prompt", "got " + join(prompt.input_ids));
    checks.expect(lfm2::lfm2_tts_system_prompt("en", "") == "Perform TTS. Use the US male voice.", "default English voice");
    checks.expect(lfm2::lfm2_tts_system_prompt("ja", "") == "Perform TTS in japanese.", "Japanese TTS prompt");
}

// F16 weights keep the backbone output within 0.003, the depthformer logits
// within 0.07, the head within 0.3 and the waveform within 0.01 of fp32 on
// every backend measured.
void check_stage_numbers(
    const lfm2::Lfm2AudioComponents & components,
    const lfm2::Lfm2AudioOutputComponents & output,
    const engine::core::BackendConfig & backend,
    Checks & checks) {
    engine::core::ExecutionContext execution(backend);
    const lfm2::Lfm2TextTokenizer tokenizer(components.vocabulary);
    lfm2::Lfm2BackboneRuntime backbone(
        components.model, components.backbone, execution, components.mmproj, output.depthformer.codebooks, output.depthformer.audio_vocab_size);
    lfm2::Lfm2DepthformerRuntime depthformer(output.vocoder, output.depthformer, execution);

    const auto prompt = lfm2::make_lfm2_tts_prompt(tokenizer, lfm2::lfm2_tts_system_prompt("en", kVoice), kText);
    const auto logits = backbone.start(prompt, {}, static_cast<int64_t>(kFrames.size()) + 2, lfm2::Lfm2DecodeCache::Speech);
    const auto first = static_cast<int32_t>(std::max_element(logits.begin(), logits.end()) - logits.begin());
    checks.expect(first == tokenizer.require_token_id("<|audio_start|>"), "speech starts with <|audio_start|>", std::to_string(first));

    auto hidden = backbone.step_text(first, lfm2::Lfm2StepOutput::Hidden);
    for (const auto & point : kHiddenPoints) {
        checks.expect_close(hidden[static_cast<size_t>(point.index)], point.value, 0.01, "backbone output [" + std::to_string(point.index) + "]");
    }

    std::vector<std::vector<float>> frame_logits;
    const auto greedy = [&](int64_t, std::vector<float> & values) {
        frame_logits.push_back(values);
        return static_cast<int32_t>(std::max_element(values.begin(), values.end()) - values.begin());
    };
    // Whether a frame that differs from the reference's differs only at
    // near-ties. Each codebook's step reads the code picked before it, so
    // after one codebook differs the later ones see other input. The frame
    // is run again with the reference's codes fed in instead: a codebook
    // differs when its best code is then not the reference's, and each one
    // that does must have the reference's code within kNearTie of the best,
    // with the reference's own top two logits as close.
    const auto differs_only_at_near_ties = [&](size_t frame, const std::vector<float> & frame_hidden, std::string & detail) {
        const auto & expected = kFrames[frame];
        std::vector<std::vector<float>> forced;
        (void)depthformer.frame(frame_hidden, [&](int64_t codebook, std::vector<float> & values) {
            forced.push_back(values);
            return expected[static_cast<size_t>(codebook)];
        });

        size_t differing = 0;
        bool near_ties = true;
        for (size_t codebook = 0; codebook < forced.size(); ++codebook) {
            const auto & values = forced[codebook];
            const auto best = static_cast<size_t>(std::max_element(values.begin(), values.end()) - values.begin());
            const auto reference = static_cast<size_t>(expected[codebook]);
            if (best == reference) {
                continue;
            }

            const float gap = values[best] - values[reference];
            const float reference_gap = kFrameGaps[frame][codebook];
            ++differing;
            near_ties = near_ties && gap < kNearTie && reference_gap < kNearTie;
            detail += (differing > 1 ? ", codebook " : "codebook ") + std::to_string(codebook) + " has " + std::to_string(best) +
                      " for " + std::to_string(reference) + " at a gap of " + std::to_string(gap) + ", the reference's " +
                      std::to_string(reference_gap);
        }

        if (differing == 0) {
            detail = "no codebook differs with the reference's codes fed in";
        }

        return differing > 0 && near_ties;
    };

    auto codes = depthformer.frame(hidden, greedy);
    for (size_t codebook = 0; codebook < std::size(kFirstFrame); ++codebook) {
        const auto & top = kFirstFrame[codebook];
        const auto & values = frame_logits[codebook];
        checks.expect(codes[codebook] == top.code, "first frame, codebook " + std::to_string(codebook), std::to_string(codes[codebook]));
        checks.expect_close(values[static_cast<size_t>(top.code)], top.logit, 0.15, "first frame, codebook " + std::to_string(codebook) + " top logit");
    }

    // Greedy frames, fed back one by one, until the reference's end.
    size_t compared = 0;
    for (size_t frame = 0; frame < kFrames.size(); ++frame) {
        if (frame > 0) {
            hidden = backbone.step_audio(codes, lfm2::Lfm2StepOutput::Hidden);
            frame_logits.clear();
            codes = depthformer.frame(hidden, greedy);
        }

        if (codes != kFrames[frame]) {
            std::string detail;
            const bool near_ties = differs_only_at_near_ties(frame, hidden, detail);
            checks.expect(near_ties, "greedy frame " + std::to_string(frame) + " differs only at near-ties (" + detail + ")", join(codes));
            break;
        }

        ++compared;
    }

    std::cout << "greedy frames identical: " << compared << " of " << kFrames.size() << "\n";
    if (compared == kFrames.size()) {
        hidden = backbone.step_audio(codes, lfm2::Lfm2StepOutput::Hidden);
        codes = depthformer.frame(hidden, greedy);
        checks.expect(codes.front() == output.depthformer.end_of_audio(), "end-of-audio after the last frame", join(codes));
    }

    lfm2::Lfm2DetokenizerRuntime detokenizer(output.detokenizer, output.vocoder, output.detokenizer_config, execution);
    const auto head = detokenizer.spectrum(kFrames);
    const int64_t width = output.detokenizer_config.output_size;
    for (const auto & point : kHeadPoints) {
        checks.expect_close(head[static_cast<size_t>(point.row * width + point.column)], point.value, 0.3,
            "detokenizer head [" + std::to_string(point.row) + "][" + std::to_string(point.column) + "]");
    }

    // Longer audio is decoded in chunks that overlap by the receptive field;
    // 8-frame chunks over 96 frames must give the one-pass result. On the
    // CPU, as GPU kernels change with the batch size: there F16 weights
    // differ by up to 0.3 between graph sizes, even over identical steps.
    // A chunk that missed context would be off by the values themselves.
    std::vector<std::vector<int32_t>> long_frames;
    for (int repeat = 0; repeat < 3; ++repeat) {
        long_frames.insert(long_frames.end(), kFrames.begin(), kFrames.end());
    }

    engine::core::ExecutionContext cpu({engine::core::BackendType::Cpu, 0, backend.threads});
    lfm2::Lfm2DetokenizerRuntime chunked(output.detokenizer, output.vocoder, output.detokenizer_config, cpu, 8);
    lfm2::Lfm2DetokenizerRuntime one_pass(output.detokenizer, output.vocoder, output.detokenizer_config, cpu, 1024);
    const auto chunked_head = chunked.spectrum(long_frames);
    const auto one_pass_head = one_pass.spectrum(long_frames);
    double chunk_difference = chunked_head.size() == one_pass_head.size() ? 0.0 : INFINITY;
    for (size_t i = 0; i < std::min(chunked_head.size(), one_pass_head.size()); ++i) {
        chunk_difference = std::max(chunk_difference, std::fabs(static_cast<double>(chunked_head[i]) - one_pass_head[i]));
    }

    checks.expect_close(chunk_difference, 0.0, 0.1, "chunked detokenizer against one pass, largest difference");

    // A stream decodes a frame at a time from the state the frames before it
    // left; that must give the one-pass rows too.
    one_pass.start_stream();
    std::vector<float> streamed_head;
    for (const auto & frame : long_frames) {
        const auto rows = one_pass.stream({frame});
        streamed_head.insert(streamed_head.end(), rows.begin(), rows.end());
    }

    double stream_difference = streamed_head.size() == one_pass_head.size() ? 0.0 : INFINITY;
    for (size_t i = 0; i < std::min(streamed_head.size(), one_pass_head.size()); ++i) {
        stream_difference = std::max(stream_difference, std::fabs(static_cast<double>(streamed_head[i]) - one_pass_head[i]));
    }

    checks.expect_close(stream_difference, 0.0, 0.1, "detokenizer stream against one pass, largest difference");

    const auto wave = detokenizer.decode(kFrames);
    checks.expect(wave.size() == kWaveSamples, "waveform length", std::to_string(wave.size()));
    if (wave.size() == kWaveSamples) {
        for (const auto & point : kWavePoints) {
            checks.expect_close(wave[static_cast<size_t>(point.index)], point.value, 0.01, "waveform [" + std::to_string(point.index) + "]");
        }

        double squares = 0.0;
        for (const float sample : wave) {
            squares += static_cast<double>(sample) * sample;
        }

        checks.expect_close(std::sqrt(squares / static_cast<double>(wave.size())), kWaveRms, 0.02 * kWaveRms, "waveform RMS");
    }
}

std::unique_ptr<engine::runtime::IVoiceTaskSession> open_session(
    engine::runtime::ILoadedVoiceModel & model,
    engine::runtime::VoiceTaskKind task,
    const std::string & model_gguf,
    const engine::core::BackendConfig & backend,
    engine::runtime::RunMode mode = engine::runtime::RunMode::Offline) {
    engine::runtime::SessionOptions options;
    options.backend = backend;
    options.options["lfm2_audio.model_gguf"] = model_gguf;
    auto session = model.create_task_session({task, mode}, options);
    if (auto * offline = dynamic_cast<engine::runtime::IOfflineVoiceTaskSession *>(session.get())) {
        offline->prepare({});
    }

    return session;
}

engine::runtime::TaskResult run(engine::runtime::IVoiceTaskSession & session, const engine::runtime::TaskRequest & request) {
    auto * offline = dynamic_cast<engine::runtime::IOfflineVoiceTaskSession *>(&session);
    if (offline == nullptr) {
        throw std::runtime_error("the session is not an IOfflineVoiceTaskSession");
    }

    return offline->run(request);
}

// What `call` writes to std::cerr, where the family's warnings go.
std::string stderr_of(const std::function<void()> & call) {
    std::ostringstream captured;
    auto * const previous = std::cerr.rdbuf(captured.rdbuf());
    try {
        call();
    } catch (...) {
        std::cerr.rdbuf(previous);
        throw;
    }

    std::cerr.rdbuf(previous);
    return captured.str();
}

double relative_rms(const std::vector<float> & actual, const std::vector<float> & expected, size_t count) {
    double difference = 0.0;
    double energy = 0.0;
    for (size_t i = 0; i < count; ++i) {
        difference += (static_cast<double>(actual[i]) - expected[i]) * (static_cast<double>(actual[i]) - expected[i]);
        energy += static_cast<double>(expected[i]) * expected[i];
    }

    return std::sqrt(difference / energy);
}

// Speech through the registry, greedy and sampled, and back through ASR.
void check_round_trip(
    const std::filesystem::path & model_dir,
    const std::filesystem::path & spec_override,
    const std::string & model_gguf,
    const engine::core::BackendConfig & backend,
    bool full_precision,
    Checks & checks) {
    auto registry = engine::runtime::make_default_registry();
    engine::runtime::ModelLoadRequest load_request;
    load_request.model_path = model_dir;
    load_request.model_spec_override = spec_override;
    load_request.family_hint = "lfm2_audio";
    auto model = registry.load(load_request);
    auto tts = open_session(*model, engine::runtime::VoiceTaskKind::Tts, model_gguf, backend);
    auto asr = open_session(*model, engine::runtime::VoiceTaskKind::Asr, model_gguf, backend);
    const auto & tts_session = dynamic_cast<const lfm2::Lfm2AudioTtsSession &>(*tts);

    const auto speech_request = [](const char * temperature, const char * seed = kSampledSeeds[0]) {
        engine::runtime::TaskRequest request;
        request.text_input = engine::runtime::Transcript{kText, "en"};
        request.voice = engine::runtime::VoiceCondition{engine::runtime::VoiceReference{std::nullopt, std::string(kVoice)}, std::nullopt};
        request.options["temperature"] = temperature;
        request.options["seed"] = seed;
        return request;
    };

    // The speech's length, whether it ended before max_tokens, and what the
    // ASR task hears in it.
    struct Take {
        double seconds = 0.0;
        bool ended = false;
        std::string heard;
    };

    const auto speak = [&](const engine::runtime::TaskRequest & request, const std::string & label, std::vector<float> * samples) {
        Take take;
        const auto speech = run(*tts, request);
        const bool has_audio = speech.audio_output.has_value() && speech.audio_output->sample_rate == 24000 && !speech.audio_output->samples.empty();
        checks.expect(has_audio, label + " speech is 24 kHz audio");
        if (!has_audio) {
            return take;
        }

        if (samples != nullptr) {
            *samples = speech.audio_output->samples;
        }

        take.seconds = static_cast<double>(speech.audio_output->samples.size()) / 24000.0;
        take.ended = !tts_session.reached_max_tokens();
        engine::runtime::TaskRequest transcribe;
        transcribe.audio_input = speech.audio_output;
        const auto text = run(*asr, transcribe);
        take.heard = text.text_output.has_value() ? text.text_output->text : "<no transcript>";
        std::cout << label << " speech: " << take.seconds << " s, heard \"" << take.heard << "\"\n";
        return take;
    };

    std::vector<float> greedy_speech;
    const auto greedy = speak(speech_request("0"), "greedy", &greedy_speech);
    checks.expect(greedy.seconds > 1.5 && greedy.seconds < 6.0, "greedy speech length", std::to_string(greedy.seconds));
    checks.expect(greedy.heard == kText, "greedy speech transcribes back", "got \"" + greedy.heard + "\"");
    checks.expect(greedy.ended, "greedy speech ends before max_tokens");

    // A sampled take says the text, but not always word for word: of seeds 1
    // to 40, 35 to 40 came back exactly with each quantized package and CPU
    // measured, with ggml's repacked kernels and without, and kernels that sum
    // in another order sample other takes. So two of three seeds have to come
    // back exactly, ending before max_tokens at a length of 1.5 to 6 s.
    size_t sampled_exact = 0;
    std::string sampled_heard;
    for (const char * seed : kSampledSeeds) {
        const auto take = speak(speech_request("0.8", seed), std::string("sampled (seed ") + seed + ")", nullptr);
        if (take.seconds > 1.5 && take.seconds < 6.0 && take.ended && take.heard == kText) {
            ++sampled_exact;
        }

        sampled_heard += std::string(sampled_heard.empty() ? "" : " | ") + seed + ": " + std::to_string(take.seconds) + " s" +
            (take.ended ? "" : ", cut at max_tokens") + ", \"" + take.heard + "\"";
    }

    checks.expect(sampled_exact >= 2, "sampled speech transcribes back, " + std::to_string(sampled_exact) + " of 3 seeds", sampled_heard);

    // A turn that reaches max_tokens keeps the speech it has, as liquid-audio
    // keeps what it generated at max_new_tokens, with a warning, and the next
    // text chunk still gets its turn. The greedy speech runs to 32 frames of
    // 1920 samples, so max_tokens=10 keeps up to 10 of them.
    auto cut_request = speech_request("0");
    cut_request.options["max_tokens"] = "10";
    engine::runtime::TaskResult cut;
    const auto cut_warnings = stderr_of([&] { cut = run(*tts, cut_request); });
    const std::vector<float> cut_speech = cut.audio_output.has_value() ? cut.audio_output->samples : std::vector<float>();
    checks.expect(!cut_speech.empty() && cut_speech.size() <= 10 * 1920 && cut_speech.size() % 1920 == 0,
        "max_tokens=10 keeps up to 10 frames of speech", std::to_string(cut_speech.size()) + " samples");
    checks.expect(tts_session.reached_max_tokens(), "the session reports speech cut off at max_tokens");
    checks.expect(cut_warnings.find("[warning][lfm2_audio] the speech reached max_tokens=10") != std::string::npos &&
                      cut_warnings.find("raise max_tokens") != std::string::npos,
        "a warning says what to raise", cut_warnings);

    auto two_chunks = cut_request;
    two_chunks.text_input->text = std::string(kText) + " " + kText;
    two_chunks.options["text_chunk_size"] = "50";
    two_chunks.options["max_tokens"] = "5";
    const auto chunk_warnings = stderr_of([&] { cut = run(*tts, two_chunks); });
    const size_t chunk_samples = cut.audio_output.has_value() ? cut.audio_output->samples.size() : 0;
    checks.expect(chunk_samples > 5 * 1920 && chunk_samples <= 10 * 1920, "two text chunks cut at 5 frames each",
        std::to_string(chunk_samples) + " samples");
    checks.expect(chunk_warnings.find("text chunk 1 of 2") != std::string::npos && chunk_warnings.find("text chunk 2 of 2") != std::string::npos,
        "each text chunk is cut on its own", chunk_warnings);

    const auto rejects = [&](const std::function<void(engine::runtime::TaskRequest &)> & edit, const std::string & needle, const std::string & label) {
        engine::runtime::TaskRequest request;
        request.text_input = engine::runtime::Transcript{kText, "en"};
        edit(request);
        std::string message = "<no error>";
        try {
            (void)run(*tts, request);
        } catch (const std::exception & error) {
            message = error.what();
        }

        checks.expect(message.find(needle) != std::string::npos, "rejects " + label, message);
    };

    rejects([](auto & r) { r.voice = engine::runtime::VoiceCondition{engine::runtime::VoiceReference{std::nullopt, std::string("robot")}, std::nullopt}; },
        "unknown LFM2-Audio voice", "an unknown voice");
    rejects([](auto & r) {
        r.voice = engine::runtime::VoiceCondition{engine::runtime::VoiceReference{engine::runtime::AudioBuffer{24000, 1, std::vector<float>(2400, 0.0f)}, std::nullopt}, std::nullopt};
    }, "does not clone", "reference audio");
    rejects([](auto & r) { r.text_input->language = "ja"; }, "speaks en", "another language");
    rejects([](auto & r) { r.options["audio_chunk_mode"] = "vad"; }, "does not take request option audio_chunk_mode", "an ASR option");
    rejects([](auto & r) { r.options["text_temperature"] = "0.7"; }, "does not take request option text_temperature", "an S2S option");
    rejects([](auto & r) { r.text_input->text = "  "; }, "requires text_input", "empty text");

    // Streaming, one frame per event by default: the events add up to the
    // offline speech, up to the detokenizer's arithmetic in graphs of other
    // sizes. The other sessions go first, so one set of weights is loaded.
    tts.reset();
    asr.reset();
    auto streaming_session = open_session(*model, engine::runtime::VoiceTaskKind::Tts, model_gguf, backend, engine::runtime::RunMode::Streaming);
    auto * streaming = dynamic_cast<engine::runtime::IStreamingVoiceTaskSession *>(streaming_session.get());
    checks.expect(streaming != nullptr, "the TTS session streams");
    if (streaming != nullptr && !greedy_speech.empty()) {
        streaming->start_stream(speech_request("0"));
        std::vector<float> streamed;
        size_t events = 0;
        while (auto event = streaming->next_stream_event()) {
            ++events;
            if (event->audio_output.has_value()) {
                streamed.insert(streamed.end(), event->audio_output->samples.begin(), event->audio_output->samples.end());
            }
        }

        const auto finished = streaming->finish_stream();
        checks.expect(finished.audio_output.has_value() && finished.audio_output->samples == streamed, "the stream's result is its events");
        // One event per frame, and one for the last 20 ms, which the ISTFT
        // completes once end-of-audio shows no frame follows.
        checks.expect(events == greedy_speech.size() / 1920 + 1, "one event per frame and one for the tail", std::to_string(events) + " events");
        checks.expect(streamed.size() == greedy_speech.size(), "streamed speech has the offline length", std::to_string(streamed.size()));
        if (streamed.size() == greedy_speech.size()) {
            double difference = 0.0;
            double energy = 0.0;
            for (size_t i = 0; i < streamed.size(); ++i) {
                difference += (static_cast<double>(streamed[i]) - greedy_speech[i]) * (static_cast<double>(streamed[i]) - greedy_speech[i]);
                energy += static_cast<double>(greedy_speech[i]) * greedy_speech[i];
            }

            // Measured with F16: 5.5e-4 on an x86 CPU, 1.7e-3 on the M3 Ultra's
            // CPU, 7.2e-4 on Metal, 5.4e-3 on CUDA, whose cuBLAS accumulates the
            // offline products in half precision. Quantized packages round
            // activations to 8 bits on the CPU and CUDA, and graphs of other
            // sizes round them differently: on CUDA 2.2e-2 with Q8_0 and 6.0e-2
            // with Q4_0. The stream's six-row Q4_0 matmuls run ggml-cuda's MMVQ
            // kernel, whose dot product makes the rounding error two to three
            // times larger than in the kernel offline decoding uses. They get
            // the S2S test's 0.3, five times the Q4_0 number on this text
            // (other texts reach 0.32 on CUDA, see the streaming docs) and
            // still under a stream that lost its context: 1.2, or 0.70 to 0.81
            // when it restarts every 5 frames.
            checks.expect_close(std::sqrt(difference / energy), 0.0, full_precision ? 0.02 : 0.3, "streamed against offline speech, relative RMS difference");
        }

        // Cut off at max_tokens, the stream is the offline cut speech. Its
        // frames are the first of the greedy stream's, decoded the same way,
        // so its samples are the same but for the last 20 ms, which the ISTFT
        // completes without the frame that came next.
        std::vector<float> streamed_cut;
        const auto stream_warnings = stderr_of([&] {
            streaming->start_stream(cut_request);
            while (auto event = streaming->next_stream_event()) {
                if (event->audio_output.has_value()) {
                    streamed_cut.insert(streamed_cut.end(), event->audio_output->samples.begin(), event->audio_output->samples.end());
                }
            }
        });

        const auto & streaming_tts = dynamic_cast<const lfm2::Lfm2AudioTtsSession &>(*streaming_session);
        checks.expect(streaming_tts.reached_max_tokens() && stream_warnings.find("reached max_tokens=10") != std::string::npos,
            "the cut stream warns and reports it", stream_warnings);
        const auto cut_finished = streaming->finish_stream();
        checks.expect(cut_finished.audio_output.has_value() && cut_finished.audio_output->samples == streamed_cut, "the cut stream's result is its events");
        checks.expect(streamed_cut.size() == cut_speech.size(), "the cut stream has the offline cut length", std::to_string(streamed_cut.size()));
        if (streamed_cut.size() == cut_speech.size() && streamed_cut.size() > 480 && streamed.size() >= streamed_cut.size()) {
            checks.expect_close(relative_rms(streamed_cut, streamed, streamed_cut.size() - 480), 0.0, 1e-3,
                "cut stream against the start of the greedy stream, relative RMS difference");
        }
    }
}

}  // namespace

int main(int argc, char ** argv) {
    const std::filesystem::path model_dir = arg_value(argc, argv, "--model", repo_path("models/LFM2.5-Audio-1.5B-GGUF").string());
    const std::string model_gguf = arg_value(argc, argv, "--model-gguf", "LFM2.5-Audio-1.5B-F16.gguf");
    const std::filesystem::path spec_override =
        arg_value(argc, argv, "--model-spec-override", repo_path("model_specs").string());
    const std::string backend_name = arg_value(argc, argv, "--backend", "cpu");
    const int threads = std::atoi(arg_value(argc, argv, "--threads", "4").c_str());
    if (backend_name != "cpu" && backend_name != "best") {
        std::cerr << "FAIL: --backend must be cpu or best\n";
        return kExitFail;
    }

    for (const auto & prefix : {"", "mmproj-", "vocoder-", "tokenizer-"}) {
        if (!engine::io::is_existing_file(model_dir / (std::string(prefix) + model_gguf))) {
            std::fprintf(
                stderr,
                "SKIP: test_lfm2_audio_tts needs %s and its mmproj-, vocoder- and tokenizer- files in '%s'.\n"
                "      python3 tools/model_manager_v2.py install lfm2_audio_1_5b_f16 --models-root models\n",
                model_gguf.c_str(),
                model_dir.string().c_str());
            return kExitSkip;
        }
    }

    engine::core::BackendConfig backend;
    backend.type = backend_name == "cpu" ? engine::core::BackendType::Cpu : engine::core::BackendType::BestAvailable;
    backend.threads = threads > 0 ? threads : 1;

    // Quantized weights move the stage numbers and the stream's arithmetic by
    // design; their speech is still checked.
    const bool full_precision = model_gguf.find("-F16.") != std::string::npos || model_gguf.find("-F32.") != std::string::npos;
    Checks checks;
    try {
        {
            const auto assets = lfm2::load_lfm2_audio_assets(model_dir);
            const auto components = lfm2::load_lfm2_audio_components(*assets, model_gguf, "");
            const auto output = lfm2::load_lfm2_audio_output_components(*assets, *components, "", "");
            check_prompt(*components, checks);
            if (full_precision) {
                check_stage_numbers(*components, *output, backend, checks);
            } else {
                std::cout << "SKIP stage numbers: " << model_gguf << " is quantized\n";
            }
        }

        check_round_trip(model_dir, spec_override, model_gguf, backend, full_precision, checks);
    } catch (const std::exception & error) {
        std::cerr << "FAIL: " << error.what() << "\n";
        return kExitFail;
    }

    std::cout << (checks.failures() == 0 ? "PASS" : "FAIL") << ": " << checks.failures() << " failed checks\n";
    return checks.failures() == 0 ? kExitPass : kExitFail;
}
