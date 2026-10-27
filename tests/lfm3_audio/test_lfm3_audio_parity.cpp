// LFM3-Audio stage-by-stage parity against the fp32 HF fixtures (see lfm3_audio_model_test.h for the arguments and tolerances):
// - ASR on c_16k: features, encoder stages, prompt ids, prefill hidden and logits, greedy ids; the encoder stages also for
//   the odd-length syn_1s and syn_037s clips, and for all three fed the fixture's features instead of the host's;
// - the 418 s speech_16k prompt in one pass (5244 positions, across the sliding window and the 2048 / 4096 blocks);
// - the depth decoder on the fixture's inputs for the first 40 frames of the greedy TTS and S2S runs;
// - the codec on fixed codes (36 TTS frames, 200 random frames), offline and streamed in the fixture's chunk sizes.
// The greedy TTS and S2S slot sequences are checked by test_lfm3_audio_tts and test_lfm3_audio_s2s.
#include "lfm3_audio_model_test.h"

#include "engine/community_models/lfm3_audio/codec.h"
#include "engine/community_models/lfm3_audio/depth.h"
#include "engine/community_models/lfm3_audio/encoder.h"
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

void compare_stage(Checks & checks, const Context & ctx, const ScopedDumps & dumps, const std::string & name, const std::string & fixture,
                   double tolerance, const std::string & label) {
    if (!dumps.has(name)) {
        checks.expect(false, label, "no " + name + " dump");
        return;
    }
    std::vector<int64_t> got_shape, want_shape;
    const auto got = dumps.f32(name, &got_shape);
    const auto want = ctx.f32(fixture, &want_shape);
    compare_shape(checks, got_shape, want_shape, label);
    if (got_shape == want_shape) {
        checks.expect_rel(diff(got, want), tolerance, label);
    }
}

void compare_features(Checks & checks, const Context & ctx, const ScopedDumps & dumps, const std::string & clip) {
    std::vector<int64_t> got_shape, want_shape;
    const auto got = dumps.f32("features", &got_shape);
    const auto want = ctx.f32("feat/" + clip + "_mel", &want_shape);
    compare_shape(checks, got_shape, want_shape, clip + " features");
    if (got_shape == want_shape) {
        const auto d = diff(got, want);
        checks.expect(d.rel <= kFeaturesRel && d.max_abs <= kFeaturesMaxAbs, clip + " features",
                      "rel " + sci(d.rel) + " max_abs " + sci(d.max_abs) + " (tol " + sci(kFeaturesRel) + ", max_abs " + sci(kFeaturesMaxAbs) + ")");
    }
}

void compare_encoder(Checks & checks, const Context & ctx, const ScopedDumps & dumps, const std::string & clip) {
    compare_features(checks, ctx, dumps, clip);
    const bool long_input = clip == "speech";
    const auto tol = [&](const std::string & floor) {
        return ctx.profile == Profile::F32Cpu && !long_input ? kCpuF32HostFeaturesRel : ctx.tolerance(floor, false, long_input);
    };
    compare_stage(checks, ctx, dumps, "downsample", "enc/" + clip + "_downsample", tol("encoder.downsample"), clip + " encoder downsample");
    compare_stage(checks, ctx, dumps, "encoder", "enc/" + clip + "_final_norm", tol("encoder.final_norm"), clip + " encoder final norm");
    compare_stage(checks, ctx, dumps, "projector", "enc/" + clip + "_projector", tol("encoder.projector"), clip + " projector");
}

// The encoder alone on the fixture's features, so the host features' error is out of the comparison.
void check_encoder_fixture_features(Checks & checks, const Context & ctx, lfm3::Lfm3AudioEncoderRuntime & encoder, const std::string & clip) {
    std::vector<int64_t> shape;
    const auto mel = ctx.f32("feat/" + clip + "_mel", &shape);
    lfm3::Lfm3EncoderTaps taps;
    const auto out = encoder.encode({mel, shape[1]}, &taps);
    const std::vector<std::tuple<std::string, std::string, const std::vector<float> *>> stages = {
        {"downsample", "encoder downsample", &taps.downsample}, {"final_norm", "encoder final norm", &taps.encoder}, {"projector", "projector", &out.values}};
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

void compare_prompt(Checks & checks, const Context & ctx, const ScopedDumps & dumps, const std::string & fixture, const std::string & label) {
    const auto got = dumps.i32("prompt_ids");
    const auto want = ctx.i32(fixture);
    checks.expect(got == want, label + " prompt ids", std::to_string(got.size()) + " ids vs " + std::to_string(want.size()));
}

// `long_input` uses the last position's hidden floor (3.2%, three times the 13-position average).
void compare_prefill(Checks & checks, const Context & ctx, const ScopedDumps & dumps, const std::vector<float> & want_hidden,
                     const std::string & logits_fixture, bool long_input, const std::string & label) {
    const std::string floors = long_input ? "backbone_long." : "backbone.";
    const auto hidden = dumps.f32("prefill_hidden");
    if (hidden.size() == want_hidden.size()) {
        checks.expect_rel(diff(hidden, want_hidden), ctx.tolerance(floors + (long_input ? "final_norm_last" : "final_norm"), false, long_input),
                          label + " prefill hidden");
    } else {
        checks.expect(false, label + " prefill hidden", "size " + std::to_string(hidden.size()));
    }
    const auto logits = dumps.f32("prefill_logits");
    const auto want = ctx.f32(logits_fixture);
    if (logits.size() != want.size()) {
        checks.expect(false, label + " prefill logits", "size " + std::to_string(logits.size()));
        return;
    }
    const auto argmax = [](const std::vector<float> & v) { return std::max_element(v.begin(), v.end()) - v.begin(); };
    const auto got_top = argmax(logits), want_top = argmax(want);
    checks.expect_rel(diff(logits, want), ctx.tolerance(floors + "last_logits", false, long_input), label + " prefill logits",
                      ", argmax " + std::to_string(got_top) + " vs " + std::to_string(want_top));
    checks.expect_or_info(ctx.profile != Profile::Q4_K, got_top == want_top, label + " prefill argmax");
}

lfm3::Lfm3AsrSession & asr(Context & ctx, std::unique_ptr<lfm3::Lfm3AsrSession> & session) {
    if (session == nullptr) {
        session = ctx.open<lfm3::Lfm3AsrSession>(runtime::VoiceTaskKind::Asr, runtime::RunMode::Offline);
    }
    return *session;
}

void check_asr_c(Checks & checks, Context & ctx, std::unique_ptr<lfm3::Lfm3AsrSession> & session) {
    ScopedDumps dumps("parity_asr_c");
    asr(ctx, session).run(audio_request(read_audio(ctx.fixture("audio/c_16k.wav")), {{"language", "en"}}));
    dumps.stop();
    compare_encoder(checks, ctx, dumps, "c");
    compare_prompt(checks, ctx, dumps, "prompt/asr_en_c_ids", "asr_en_c");
    std::vector<int64_t> shape;
    const auto final_norm = ctx.f32("lm/asr_en_c_final_norm", &shape);
    const std::vector<float> last(final_norm.end() - shape[1], final_norm.end());
    compare_prefill(checks, ctx, dumps, last, "lm/asr_en_c_last_logits", false, "asr_en_c");

    std::vector<int32_t> tokens;
    for (const auto & slot : dumped_slots(dumps)) tokens.push_back(slot.token);
    const auto want = ctx.i32("gen/asr_en_c_greedy_ids");
    size_t first = 0;
    while (first < std::min(tokens.size(), want.size()) && tokens[first] == want[first]) ++first;
    checks.expect_or_info(ctx.strict_text(), tokens == want, "asr_en_c greedy ids",
                          std::to_string(tokens.size()) + " vs " + std::to_string(want.size()) + " ids" +
                              (tokens == want ? std::string() : ", first difference at " + std::to_string(first)));
}

void check_encoder_clip(Checks & checks, Context & ctx, std::unique_ptr<lfm3::Lfm3AsrSession> & session, const std::string & clip) {
    ScopedDumps dumps("parity_" + clip);
    asr(ctx, session).run(audio_request(read_audio(ctx.fixture("audio/" + clip + "_16k.wav")), {{"max_tokens", "1"}}));
    dumps.stop();
    compare_encoder(checks, ctx, dumps, clip);
}

void check_long_prefill(Checks & checks, Context & ctx) {
    auto session = ctx.open<lfm3::Lfm3AsrSession>(runtime::VoiceTaskKind::Asr, runtime::RunMode::Offline,
                                                  {{"lfm3_audio.max_pass_duration_sec", "420"}});
    ScopedDumps dumps("parity_long");
    session->run(audio_request(read_audio(ctx.fixture("audio/speech_16k.wav")), {{"audio_chunk_mode", "none"}, {"max_tokens", "1"}}));
    dumps.stop();
    compare_encoder(checks, ctx, dumps, "speech");
    compare_prompt(checks, ctx, dumps, "prompt/asr_plain_speech_ids", "long speech");
    const auto positions = ctx.i32("lm/long_speech_positions");
    const auto prompt = ctx.i32("prompt/asr_plain_speech_ids");
    if (positions.back() + 1 != static_cast<int32_t>(prompt.size())) {
        checks.expect(false, "long speech fixture has the last position", std::to_string(positions.back()));
        return;
    }
    std::vector<int64_t> shape;
    const auto rows = ctx.f32("lm/long_speech_final_norm_at_positions", &shape);
    const std::vector<float> last(rows.end() - shape[1], rows.end());
    compare_prefill(checks, ctx, dumps, last, "lm/long_speech_last_logits", true, "long speech");
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
    lfm3::Lfm3CodecTaps taps;
    codec.start_stream();
    const auto spectrum = codec.spectrum(frames, &taps);
    stage(taps.project_out, "project_out", "codec.project_out", "codec " + tag + " FSQ project_out");
    stage(taps.from_quantizer, "from_quantizer", "codec.from_quantizer", "codec " + tag + " from_quantizer");
    stage(taps.decoder, "dec_final_norm", "codec.dec_final_norm", "codec " + tag + " decoder final norm");
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
        {
            std::unique_ptr<lfm3::Lfm3AsrSession> session;
            check_asr_c(checks, ctx, session);
            check_encoder_clip(checks, ctx, session, "syn_1s");
            check_encoder_clip(checks, ctx, session, "syn_037s");
        }
        if (has_flag(argc, argv, "--skip-long")) {
            checks.info("long speech prefill skipped (--skip-long)");
        } else {
            check_long_prefill(checks, ctx);
        }
        ctx.loaded.reset();

        const auto assets = lfm3::load_lfm3_audio_assets(ctx.model);
        engine::core::ExecutionContext execution(ctx.backend);
        {
            lfm3::Lfm3AudioEncoderRuntime encoder(assets->model, assets->encoder, execution, ctx.weight_options());
            for (const char * clip : {"c", "syn_1s", "syn_037s"}) check_encoder_fixture_features(checks, ctx, encoder, clip);
        }
        {
            lfm3::Lfm3DepthDecoderRuntime depth(assets->model, assets->depth, execution, ctx.weight_options());
            check_depth(checks, ctx, depth, "tts_fox_greedy");
            check_depth(checks, ctx, depth, "s2s_c_greedy");
        }
        lfm3::Lfm3CodecDecoder codec(assets->codec_weights, assets->codec, execution, ctx.weight_options());
        check_codec(checks, ctx, codec, "tts");
        check_codec(checks, ctx, codec, "rand200");
    } catch (const std::exception & error) {
        std::cerr << "FAIL: " << error.what() << "\n";
        return kExitFail;
    }

    return checks.finish("test_lfm3_audio_parity");
}
