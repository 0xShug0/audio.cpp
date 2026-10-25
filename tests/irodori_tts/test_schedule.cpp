#include "engine/models/irodori_tts/schedule.h"
#include "engine/framework/debug/profiler.h"

#include <chrono>
#include <cmath>
#include <iostream>
#include <stdexcept>

int main() {
  engine::debug::configure_logging({true});
  const auto start = std::chrono::steady_clock::now();
  using namespace engine::models::irodori_tts;
  try {
    for (const int steps : {1, 3, 4, 8}) {
      const auto schedule = make_irodori_meanflow_schedule(steps);
      if (schedule.size() != static_cast<size_t>(steps) ||
          schedule.front().t != 1.0F || schedule.back().t_next != 0.0F) {
        throw std::runtime_error("MeanFlow schedule endpoints or length differ");
      }
      for (int i = 0; i < steps; ++i) {
        const auto &step = schedule[static_cast<size_t>(i)];
        if (step.index != i || !(step.t > step.t_next) ||
            (i > 0 && schedule[static_cast<size_t>(i - 1)].t_next != step.t)) {
          throw std::runtime_error("MeanFlow intervals are not contiguous and decreasing");
        }
      }
    }
    const auto mf = make_irodori_meanflow_schedule(4);
    for (size_t i = 0; i < mf.size(); ++i) {
      if (mf[i].t - mf[i].t_next != 0.25F) {
        throw std::runtime_error("MeanFlow default interval differs from Python");
      }
    }
    const auto rf = make_irodori_rf_schedule(2);
    if (make_irodori_meanflow_schedule(3)[2].t != 1.0F / 3.0F) {
      throw std::runtime_error("MeanFlow does not match linspace endpoint rounding");
    }
    if (rf[0].t != 0.999F || rf[0].t_next != 0.4995F || rf[1].t_next != 0.0F) {
      throw std::runtime_error("RF schedule changed");
    }
    for (const int steps : {0, -1}) {
      bool rejected = false;
      try {
        make_irodori_meanflow_schedule(steps);
      } catch (const engine::runtime::InvalidRequestError &) {
        rejected = true;
      }
      if (!rejected) throw std::runtime_error("MeanFlow accepted nonpositive steps");
    }
    engine::debug::timing_log_scalar("session.wall_ms", engine::debug::elapsed_ms(start));
    return 0;
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
