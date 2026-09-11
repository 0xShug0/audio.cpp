#include "../core/audio_task_warm_bench.h"

int main(int argc, char **argv) {
  return engine::tools::run_audio_task_warm_bench(
      argc, argv,
      {"fun_asr_nano", "models/Fun-ASR-Nano-2512-hf",
       engine::runtime::VoiceTaskKind::Asr,
       engine::tools::AudioTaskOutputKind::Asr});
}
