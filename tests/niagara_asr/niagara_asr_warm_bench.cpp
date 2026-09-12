#include "../core/audio_task_warm_bench.h"

int main(int argc, char ** argv) {
    try {
        engine::tools::AudioTaskBenchConfig config;
        config.family = "niagara_asr";
        config.default_model = "models/Niagara-19M-Batch-EN-GGUF";
        config.task = engine::runtime::VoiceTaskKind::Asr;
        config.output_kind = engine::tools::AudioTaskOutputKind::Asr;
        return engine::tools::run_audio_task_warm_bench(argc, argv, config);
    } catch (const std::exception & ex) {
        std::cerr << "niagara_asr_warm_bench failed: " << ex.what() << "\n";
        return 1;
    }
}
