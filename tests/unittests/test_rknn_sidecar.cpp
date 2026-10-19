// The board side of the RKNN path: everything in src/framework/rknn/engine.cpp that needs
// librknnrt and an engine compiled for the build host's SoC. The plan reader and the status
// formatter live in rknn_engine_test; this one is registered only in an RKNN-enabled build and
// self-skips (77) when no engine was supplied, so CI runs it as `Not Run` and a board runs it
// for real:
//
//   cmake --build build --target rknn_sidecar_test
//   ctest --test-dir build -R '^rknn_sidecar_test$' --output-on-failure \
//       # with -DAUDIOCPP_RKNN_TEST_ENGINE=/path/to/this-board.engine.rknn at configure
//
// The engine is a configure-time path (the repo ships no compiled engines): point it at any
// engine built for this SoC, e.g. the sidecar artefacts a board deployment keeps beside the
// weights.
//
// The fake-engine section runs before that path is consulted: it needs only `librknnrt`, which
// every build that registers this test links, and it is the only board coverage the sidecars'
// own loader failure paths get — the validation chain behind `Engine::load` needs a real
// context of the right shape, so a fabricated image can only reach (and must be reported by)
// the driver refusal itself.
#include "engine/framework/rknn/engine.h"
#if defined(AUDIOCPP_TEST_HAS_MODEL_INFLECT_V2)
#include "engine/community_models/inflect_v2/rknn_decoder.h"
#endif
#if defined(AUDIOCPP_TEST_HAS_MODEL_PIPER_TTS)
#include "engine/community_models/piper_tts/rknn_decoder.h"
#endif
#if defined(AUDIOCPP_TEST_HAS_MODEL_COSYVOICE3)
#include "engine/models/cosyvoice3/rknn_flow.h"
#endif
#if defined(AUDIOCPP_TEST_HAS_MODEL_KOKORO_TTS)
#include "engine/models/kokoro_tts/rknn_generator.h"
#endif
#if defined(AUDIOCPP_TEST_HAS_MODEL_QWEN3_TTS)
#include "engine/models/qwen3_tts/rknn_speech_decoder.h"
#endif

#include "test_assert.h"

#include <filesystem>
#include <fstream>
#include <functional>
#include <string>
#include <vector>

namespace {

using engine::test::require;

// Every sidecar constructor must hand a refused image on with its own label: an operator
// reading `rknn_init failed with code -4` in a server log has to learn from the same line
// which stage's file is broken (the #989 fix this pins).
void require_driver_refusal(
    const std::function<void ()> & construct, const std::string & label) {
    std::string message;
    bool threw = false;
    try {
        construct();
    } catch (const std::runtime_error & e) {
        threw = true;
        message = e.what();
    }
    require(threw, label + ": a fabricated engine must throw");
    require(
        message.find(label + ": rknn_init failed with code -") == 0,
        "the refusal carries the label: " + message);
}

void test_fake_engine(const std::filesystem::path & dir) {
    const std::string fake = (dir / "fake.rknn").string();
    // The flow ladder reads the length it covers from the file name, so its fake image
    // carries the producer's name.
    const std::string fake_flow = (dir / "flow_estimator_seq900.rknn").string();
    for (const std::string & path : {fake, fake_flow}) {
        std::ofstream out(path, std::ios::binary | std::ios::trunc);
        out << "RKNN" << "not a compiled image";
    }
#if defined(AUDIOCPP_TEST_HAS_MODEL_PIPER_TTS)
    require_driver_refusal(
        [&] {
            engine::models::piper_tts::PiperTtsRknnDecoder(
                fake, 192, 256);
        },
        "Piper TTS decoder RKNN");
#endif
#if defined(AUDIOCPP_TEST_HAS_MODEL_INFLECT_V2)
    require_driver_refusal(
        [&] {
            engine::models::inflect_v2::InflectV2RknnDecoder(
                fake, 32, 512);
        },
        "Inflect v2 decoder RKNN");
#endif
#if defined(AUDIOCPP_TEST_HAS_MODEL_KOKORO_TTS)
    require_driver_refusal(
        [&] {
            engine::models::kokoro_tts::KokoroTtsRknnGenerator(fake, 76, 80, 100, 0);
        },
        "Kokoro generator RKNN");
#endif
#if defined(AUDIOCPP_TEST_HAS_MODEL_QWEN3_TTS)
    require_driver_refusal(
        [&] {
            engine::models::qwen3_tts::RknnSpeechDecoder(fake, "", 16, 64, 1024, 10000.0F);
        },
        "Qwen3 speech decoder RKNN");
#endif
#if defined(AUDIOCPP_TEST_HAS_MODEL_COSYVOICE3)
    // The flow ladder loads on demand, so its refusal surfaces on the first step, not on the
    // constructor -- and the constructor has a refusal of its own before the driver is ever
    // asked: a ladder named by nothing.
    require_driver_refusal(
        [&] {
            engine::models::cosyvoice3::CosyVoice3RknnFlow flow(
                {fake_flow}, 80);
            std::vector<float> zeros(static_cast<size_t>(2 * 80), 0.0F);
            (void)flow.run(
                zeros.data(), zeros.data(), zeros.data(), zeros.data(), 0.5F, 1);
        },
        "CosyVoice3 flow estimator RKNN");
#endif
}

}  // namespace

int main(int argc, char ** argv) {
    // The fake-engine section needs only the driver this build links, so it runs before the
    // real-engine guard decides the rest of the file applies to this host.
    const auto dir = std::filesystem::temp_directory_path() / "audiocpp-rknn-sidecar-test";
    std::filesystem::create_directories(dir);
    test_fake_engine(dir);
    std::filesystem::remove_all(dir);

    const std::filesystem::path engine_path = argc > 1 ? argv[1] : std::string();
    std::error_code ec;
    if (engine_path.empty() || !std::filesystem::exists(engine_path, ec)) {
        // No engine for this host: the driver path this test covers cannot run here.
        return 77;
    }

    // Residency pricing: the model-info-only init and RKNN_QUERY_MEM_SIZE must say a real
    // number for a real engine -- 0 is the "unknown" answer, and an engine the board's own
    // driver reports as 0 MiB resident would make every ladder accounting meaningless.
    const auto resident = engine::rknn::engine_memory_mib(engine_path.string(), "Sidecar test RKNN");
    engine::test::require(
        resident > 0,
        "the driver prices its own engine, got " + std::to_string(resident) + " MiB");

    // The same init caches the driver's self-report, which every later failure quotes.
    const auto report = engine::rknn::driver_report();
    engine::test::require(
        report.find("api ") != std::string::npos && report.find("driver ") != std::string::npos,
        "the driver report carries both versions: " + report);

    // And the full init path builds a real context for the same image.
    engine::rknn::Engine npu;
    npu.load(engine_path.string(), "Sidecar test RKNN");
    engine::test::require(npu.loaded(), "the driver built a context for its engine");
    engine::test::require(
        npu.input_count() > 0 && npu.output_count() > 0,
        "the context reports its graph's inputs and outputs");
    npu.release();
    return 0;
}
