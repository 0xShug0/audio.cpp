// LFM3-Audio stage-by-stage parity against the fp32 HF fixtures (see lfm3_audio_model_test.h for the arguments and tolerances):
// - ASR on c_16k: features, encoder stages, prompt ids, prefill hidden and logits; the encoder stages also for
//   the odd-length syn_1s and syn_037s clips, and for all three fed the fixture's features instead of the host's;
// - the 418 s speech_16k prompt in one pass (5244 positions, across the sliding window and the 2048 / 4096 blocks);
// - the depth decoder on the fixture's inputs for the first 40 frames of the greedy TTS and S2S runs;
// - the codec on fixed codes (36 TTS frames, 200 random frames), offline and streamed in the fixture's chunk sizes.
// TTS reference waveforms and normal streaming generation are checked by the task tests.
#include "lfm3_audio_model_test.h"

#include "engine/community_models/lfm3_audio/codec.h"
#include "engine/community_models/lfm3_audio/depth.h"
#include "engine/community_models/lfm3_audio/frontend.h"
#include "engine/framework/core/execution_context.h"

#include <exception>
#include <tuple>

namespace {

using namespace lfm3_model_test;

// Features are computed on the host in every mode, so one tolerance holds everywhere.
constexpr double kFeaturesRel = 1e-4;
constexpr double kFeaturesMaxAbs = 2e-3;
// CPU F32 encoder stages inherit the host features' ~1e-5 error (up to 1.2e-5 at the projector on arm64);
// fed the fixture's features, 1e-5 holds.
constexpr double kCpuF32HostFeaturesRel = 3e-5;

void compare_shape(Checks & checks, const std::vector<int64_t> & got, const std::vector<int64_t> & want, const std::string & label) {
    if (got != want) {
        std::string g, w;
        for (const auto d : got) g += std::to_string(d) + " ";
        for (const auto d : want) w += std::to_string(d) + " ";
        checks.expect(false, label + " shape", "[" + g + "] vs [" + w + "]");
    }
}

void check_encoder_clip(Checks & checks, const Context & ctx, lfm3::Lfm3AudioFrontend & encoder,
                        lfm3::Lfm3BackboneRuntime & backbone, const lfm3::Lfm3Tokenizer & tokenizer, const std::string & clip) {
    const auto input = read_audio(ctx.fixture("audio/" + clip + "_16k.wav"));
    const auto features = lfm3::Lfm3FeatureExtractor{}.extract(input.samples, static_cast<size_t>(std::max(1, ctx.backend.threads)));
    std::vector<int64_t> shape;
    const auto mel = ctx.f32("feat/" + clip + "_mel", &shape);
    compare_shape(checks, {128, features.frames}, shape, clip + " features");
    const auto error = diff(features.values, mel);
    checks.expect(error.rel <= kFeaturesRel && error.max_abs <= kFeaturesMaxAbs, clip + " features");
    const auto audio = encoder.encode(features);
    const bool long_input = clip == "speech";
    const std::vector<std::tuple<std::string, std::string, const std::vector<float> *>> stages = {
        {"projector", "encoder.projector", &audio.values}};
    for (const auto & [stage, floor, got] : stages) {
        const auto want = ctx.f32("enc/" + clip + "_" + stage);
        const auto tolerance = ctx.profile == Profile::F32Cpu && !long_input ? kCpuF32HostFeaturesRel : ctx.tolerance(floor, false, long_input);
        checks.expect(got->size() == want.size(), clip + " " + stage + " size");
        if (got->size() == want.size()) checks.expect_rel(diff(*got, want), tolerance, clip + " " + stage);
    }
    if (clip != "c" && !long_input) return;
    const auto prompt = lfm3::make_lfm3_asr_prompt(tokenizer, lfm3::lfm3_asr_system_prompt(long_input ? "auto" : "en"), audio.tokens);
    checks.expect(prompt.ids == ctx.i32(long_input ? "prompt/asr_plain_speech_ids" : "prompt/asr_en_c_ids"), clip + " prompt ids");
    const auto logits = backbone.prefill(prompt, tokenizer.ids().audio, tokenizer.ids().audio_out, audio, 1,
                                         lfm3::Lfm3Output::Logits);
    const std::string prefix = long_input ? "lm/long_speech" : "lm/asr_en_c";
    const auto hidden = ctx.f32(prefix + (long_input ? "_final_norm_at_positions" : "_final_norm"), &shape);
    const std::vector<float> last(hidden.end() - shape[1], hidden.end());
    checks.expect_rel(diff(backbone.prefill(prompt, tokenizer.ids().audio, tokenizer.ids().audio_out, audio, 1, lfm3::Lfm3Output::Hidden), last), ctx.tolerance(long_input ? "backbone_long.final_norm_last" : "backbone.final_norm", false, long_input), clip + " prefill hidden");
    const auto want_logits = ctx.f32(prefix + "_last_logits");
    checks.expect_rel(diff(logits, want_logits), ctx.tolerance(long_input ? "backbone_long.last_logits" : "backbone.last_logits", false, long_input), clip + " prefill logits");
}

void check_encoder_fixture_features(Checks & checks, const Context & ctx, lfm3::Lfm3AudioFrontend & encoder, const std::string & clip) {
    std::vector<int64_t> shape;
    const auto mel = ctx.f32("feat/" + clip + "_mel", &shape);
    const auto out = encoder.encode({mel, shape[1]});
    const std::vector<std::tuple<std::string, std::string, const std::vector<float> *>> stages = {
        {"projector", "projector", &out.values}};
    for (const auto & [stage, label, got] : stages) {
        const auto want = ctx.f32("enc/" + clip + "_" + stage);
        const auto name = clip + " " + label + " from fixture features";
        if (got->size() != want.size()) {
            checks.expect(false, name, "size " + std::to_string(got->size()) + " vs " + std::to_string(want.size()));
            continue;
        }
        checks.expect_rel(diff(*got, want), ctx.tolerance("encoder." + stage), name);
    }
}

void check_depth(Checks & checks, const Context & ctx, lfm3::Lfm3DepthDecoderRuntime & depth, const std::string & tag) {
    std::vector<int64_t> hidden_shape, codes_shape, logits_shape;
    const auto hidden = ctx.f32("gen/" + tag + "_frame_hidden", &hidden_shape);
    const auto codes = ctx.i32("gen/" + tag + "_frame_pass_codes", &codes_shape);
    const auto masked = ctx.i32("gen/" + tag + "_frame_pass_masked");
    const auto want = ctx.f32("gen/" + tag + "_frame_pass_logits", &logits_shape);
    const auto frames = static_cast<size_t>(logits_shape[0]);
    const auto vocab = static_cast<size_t>(logits_shape[3]);
    const auto width = static_cast<size_t>(hidden_shape[1]);
    double num = 0.0, den = 0.0, max_abs = 0.0, worst = 0.0;
    size_t agree = 0, rows = 0;
    for (size_t f = 0; f < frames; ++f) {
        const std::vector<float> h(hidden.begin() + static_cast<std::ptrdiff_t>(f * width), hidden.begin() + static_cast<std::ptrdiff_t>((f + 1) * width));
        for (size_t p = 0; p < 3; ++p) {
            lfm3::Lfm3Codes in{};
            std::array<bool, 12> mask{};
            for (size_t k = 0; k < kCodebooks; ++k) {
                in[k] = codes[(f * 3 + p) * kCodebooks + k];
                mask[k] = masked[(f * 3 + p) * kCodebooks + k] != 0;
            }
            const auto got = depth.logits(h, in, mask);
            const float * ref = want.data() + (f * 3 + p) * kCodebooks * vocab;
            for (size_t i = 0; i < kCodebooks * vocab; ++i) {
                const double d = static_cast<double>(got[i]) - ref[i];
                num += d * d;
                den += static_cast<double>(ref[i]) * ref[i];
                max_abs = std::max(max_abs, std::fabs(d));
            }
            worst = std::max(worst, diff(got.data(), ref, kCodebooks * vocab).rel);
            for (size_t k = 0; k < kCodebooks; ++k) {
                const auto * g = got.data() + k * vocab;
                const auto * r = ref + k * vocab;
                agree += std::max_element(g, g + vocab) - g == std::max_element(r, r + vocab) - r ? 1 : 0;
                ++rows;
            }
        }
    }
    char extra[96];
    std::snprintf(extra, sizeof(extra), ", worst pass rel %.3e, argmax agrees %zu/%zu", worst, agree, rows);
    checks.expect_rel({std::sqrt(num / std::max(den, 1e-30)), max_abs}, ctx.tolerance("depth.logits"),
                      tag + " depth logits on fixture inputs (" + std::to_string(frames) + " frames x 3 passes)", extra);
}

void check_codec(Checks & checks, const Context & ctx, lfm3::Lfm3CodecDecoder & codec, const std::string & tag) {
    std::vector<int64_t> shape;
    const auto codes = ctx.i32("codec/" + tag + "_codes", &shape);  // [12, T]
    const auto count = static_cast<size_t>(shape[1]);
    std::vector<std::vector<int32_t>> frames(count, std::vector<int32_t>(kCodebooks));
    for (size_t f = 0; f < count; ++f) {
        for (size_t k = 0; k < kCodebooks; ++k) frames[f][k] = codes[k * count + f];
    }

    const auto stage = [&](const std::vector<float> & got, const std::string & fixture, const std::string & floor_key, const std::string & label) {
        const auto want = ctx.f32("codec/" + tag + "_" + fixture);
        if (got.size() < want.size()) {
            checks.expect(false, label, "size " + std::to_string(got.size()) + " vs " + std::to_string(want.size()));
            return;
        }
        checks.expect_rel(diff(got.data(), want.data(), want.size()), ctx.tolerance(floor_key), label);
    };
    codec.start_stream();
    const auto spectrum = codec.spectrum(frames);
    stage(spectrum, "lin_first", "codec.lin", "codec " + tag + " spectrum (first 48 rows)");

    const auto wave_floor = "codec." + tag + "_wave";
    const auto wave = [&](const std::vector<float> & got, const std::string & fixture, const std::string & label) {
        const auto want = ctx.f32("codec/" + tag + "_" + fixture);
        if (got.size() != want.size()) {
            checks.expect(false, label, "length " + std::to_string(got.size()) + " vs " + std::to_string(want.size()));
            return;
        }
        checks.expect_rel(diff(got, want), ctx.tolerance(wave_floor, true), label);
    };
    const auto offline = codec.decode(frames);
    wave(offline, "wave", "codec " + tag + " offline waveform");

    std::vector<int64_t> chunk_shape;
    const auto chunks = ctx.i32("codec/" + tag + "_stream_chunks", &chunk_shape);  // [n, 2]: frames, samples
    codec.start_stream();
    std::vector<float> streamed;
    size_t at = 0;
    for (int64_t c = 0; c < chunk_shape[0]; ++c) {
        const auto n = static_cast<size_t>(chunks[static_cast<size_t>(c) * 2]);
        const std::vector<std::vector<int32_t>> piece(frames.begin() + static_cast<std::ptrdiff_t>(at), frames.begin() + static_cast<std::ptrdiff_t>(at + n));
        const auto samples = codec.push(piece);
        streamed.insert(streamed.end(), samples.begin(), samples.end());
        at += n;
    }
    const auto rest = codec.finish();
    streamed.insert(streamed.end(), rest.begin(), rest.end());
    wave(streamed, "stream_wave", "codec " + tag + " streamed waveform");
}

}  // namespace

int main(int argc, char ** argv) {
    int exit_code = kExitPass;
    auto parsed = parse_context(argc, argv, "test_lfm3_audio_parity", exit_code);
    if (!parsed) {
        return exit_code;
    }
    auto & ctx = *parsed;

    Checks checks;
    try {
        const auto assets = lfm3::load_lfm3_audio_assets(ctx.model);
        engine::core::ExecutionContext execution(ctx.backend);
        {
            lfm3::Lfm3AudioFrontend encoder(lfm3::load_lfm3_encoder_weights(*assets, execution, ctx.weight_options()), assets->encoder, execution, ctx.weight_options());
            lfm3::Lfm3BackboneRuntime backbone(lfm3::load_lfm3_backbone_weights(*assets, execution, ctx.weight_options()), assets->backbone, assets->depth.codebooks,
                                             assets->depth.audio_vocab_size, execution, ctx.weight_options());
            const lfm3::Lfm3Tokenizer tokenizer(*assets);
            for (const char * clip : {"c", "syn_1s", "syn_037s"}) {
                check_encoder_clip(checks, ctx, encoder, backbone, tokenizer, clip);
                check_encoder_fixture_features(checks, ctx, encoder, clip);
            }
            if (!has_flag(argc, argv, "--skip-long")) check_encoder_clip(checks, ctx, encoder, backbone, tokenizer, "speech");
        }
        {
            lfm3::Lfm3DepthDecoderRuntime depth(lfm3::load_lfm3_depth_weights(*assets, execution, ctx.weight_options()), assets->depth, execution, ctx.weight_options());
            check_depth(checks, ctx, depth, "tts_fox_greedy");
            check_depth(checks, ctx, depth, "s2s_c_greedy");
        }
        lfm3::Lfm3CodecDecoder codec(lfm3::load_lfm3_codec_weights(*assets, execution, ctx.weight_options()), assets->codec, execution, ctx.weight_options());
        check_codec(checks, ctx, codec, "tts");
        check_codec(checks, ctx, codec, "rand200");
    } catch (const std::exception & error) {
        std::cerr << "FAIL: " << error.what() << "\n";
        return kExitFail;
    }

    return checks.finish("test_lfm3_audio_parity");
}
