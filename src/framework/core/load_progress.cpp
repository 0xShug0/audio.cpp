#include "engine/framework/core/load_progress.h"

#include "engine/framework/debug/trace.h"

#include <algorithm>
#include <mutex>

namespace engine::core {

namespace {

struct LoadProgressState {
    bool active = false;
    uint64_t total_bytes = 0;
    uint64_t done_bytes = 0;
    double last_emitted = -1.0;
    std::string last_store;
};

LoadProgressState & load_progress_state() {
    static LoadProgressState state;
    return state;
}

std::mutex & load_progress_mutex() {
    static std::mutex mutex;
    return mutex;
}

// ~500 lines per load keeps the trace log tight while still reading as a
// continuous curve to anything tailing it.
constexpr double kEmitStep = 0.002;

}  // namespace

void begin_model_load(uint64_t total_bytes) {
    if (!engine::debug::trace_log_enabled()) {
        return;
    }
    std::lock_guard<std::mutex> lock(load_progress_mutex());
    auto & state = load_progress_state();
    state.active = true;
    state.total_bytes = total_bytes;
    state.done_bytes = 0;
    state.last_emitted = -1.0;
    state.last_store.clear();
}

void register_weight_bytes(uint64_t bytes) {
    // Trace-gated before the lock: with logging off the tracker is inert and
    // store uploads (thousands of tensors per model) must not pay for a mutex.
    if (!engine::debug::trace_log_enabled()) {
        return;
    }
    std::lock_guard<std::mutex> lock(load_progress_mutex());
    auto & state = load_progress_state();
    if (!state.active) {
        return;
    }
    // Stores register their upload budget one at a time, as each upload()
    // starts. Raise the denominator rather than letting the fraction run
    // past the seed when backend tensors expand past the on-disk bytes.
    state.total_bytes = std::max(state.total_bytes, state.done_bytes + bytes);
}

void add_uploaded_weight_bytes(uint64_t bytes) {
    if (!engine::debug::trace_log_enabled()) {
        return;
    }
    std::lock_guard<std::mutex> lock(load_progress_mutex());
    auto & state = load_progress_state();
    if (!state.active) {
        return;
    }
    state.done_bytes += bytes;
}

void emit_weight_upload_progress(const std::string & store_name) {
    if (!engine::debug::trace_log_enabled()) {
        return;
    }
    std::lock_guard<std::mutex> lock(load_progress_mutex());
    auto & state = load_progress_state();
    if (!state.active || state.total_bytes == 0) {
        return;
    }
    const double fraction = std::max(
        std::min(
            static_cast<double>(state.done_bytes) / static_cast<double>(state.total_bytes),
            1.0),
        state.last_emitted);
    if (state.last_emitted >= 0.0 && fraction - state.last_emitted < kEmitStep && fraction < 1.0) {
        return;
    }
    state.last_emitted = fraction;
    state.last_store = store_name;
    engine::debug::trace_log_scalar(store_name + ".weights.upload_progress", fraction);
}

void end_model_load() {
    std::lock_guard<std::mutex> lock(load_progress_mutex());
    auto & state = load_progress_state();
    if (!state.active) {
        return;
    }
    state.active = false;
    if (engine::debug::trace_log_enabled() && !state.last_store.empty() && state.last_emitted < 1.0) {
        state.last_emitted = 1.0;
        engine::debug::trace_log_scalar(state.last_store + ".weights.upload_progress", 1.0);
    }
}

}  // namespace engine::core
