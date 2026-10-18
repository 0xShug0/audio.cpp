#include "model_memory.h"

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/runtime/registry.h"

#include <algorithm>
#include <filesystem>
#include <functional>
#include <set>
#include <string>
#include <system_error>
#include <vector>

namespace minitts::server {

std::optional<size_t> estimate_model_memory_bytes(const ServerModelConfig & model) {
    size_t weights = 0;
    // What the NPU driver reports for the engines a plan declares, in total. Counted apart from
    // `weights` because the runtime factor below prices ggml graphs and KV state, while the driver's
    // number is already the allocation.
    size_t aperture = 0;
    std::error_code ec;
    // A compiled engine can be declared by a session option and also sit inside a checkpoint tree
    // that gets summed, so each path is counted once.
    std::set<std::filesystem::path> counted;
    // Checkpoint trees (safetensors / HF-style directories) are summed recursively
    // with hard limits so a pathological tree cannot stall the load path or blow the
    // counter; anything beyond the limits contributes 0, while the fixed floor
    // below keeps the estimate conservative even for a directory that reads as
    // empty.
    constexpr size_t kMaxDepth = 3;
    constexpr size_t kMaxFiles = 10000;
    size_t visited_files = 0;
    const auto add_file = [&](const std::filesystem::path & path) {
        if (!std::filesystem::is_regular_file(path, ec)) {
            return;
        }
        if (!counted.insert(std::filesystem::weakly_canonical(path, ec)).second) {
            return;
        }
        // A file that disappears or becomes unreadable mid-scan contributes 0
        // (file_size reports (uintmax_t)-1 with ec set) rather than wrapping the
        // counter.
        const auto size = std::filesystem::file_size(path, ec);
        if (ec) {
            return;
        }
        weights += static_cast<size_t>(size);
        ++visited_files;
    };
    // What the model's own resource plan reports for the compiled engines it will hold resident.
    // Counted apart from `weights`, and at 1.0 and not at the runtime factor, because a compiled
    // engine is resident at its own allocation: it builds no ggml graph and holds no KV state. Where
    // the target runtime cannot price an engine -- every build without an RKNN runtime, every path
    // that is not an engine image -- it is priced at its file size.
    const auto add_engine = [&](const std::filesystem::path & path, int64_t resident_mib) {
        if (!std::filesystem::is_regular_file(path, ec)) {
            return;
        }
        // A compiled engine can be declared by a session option and also sit inside a checkpoint tree
        // that gets summed, so each path is counted once.
        if (!counted.insert(std::filesystem::weakly_canonical(path, ec)).second) {
            return;
        }
        const auto size = [&]() -> size_t {
            if (resident_mib > 0) {
                return static_cast<size_t>(resident_mib) * (1024u * 1024u);
            }
            // A file that vanished mid-scan reports (uintmax_t)-1 with `ec` set; contribute 0
            // rather than wrapping the counter.
            const auto bytes = std::filesystem::file_size(path, ec);
            return ec ? 0 : static_cast<size_t>(bytes);
        }();
        aperture += size;
        ++visited_files;
    };
    const std::function<void(const std::filesystem::path &, size_t)> add_tree =
        [&](const std::filesystem::path & path, size_t depth) {
            if (std::filesystem::is_regular_file(path, ec)) {
                add_file(path);
            } else if (std::filesystem::is_directory(path, ec) && depth < kMaxDepth) {
                std::filesystem::directory_iterator it(path, ec), end;
                for (; it != end && visited_files < kMaxFiles; it.increment(ec)) {
                    add_tree(it->path(), depth + 1);
                }
            }
        };
    // Estimate only what the loader will actually read from model.path, so the
    // guard neither overestimates nor masks the loader's own error:
    //  - a single-file model contributes that file;
    //  - a model directory contributes the one GGUF it selects (model.gguf, or the
    //    sole *.gguf) -- a package holding several variants is loaded from just one;
    //  - a directory with no GGUF is a safetensors/HF checkpoint whose whole tree
    //    loads, so it is summed;
    //  - a directory with several GGUFs and no model.gguf is ambiguous: the loader
    //    rejects the spec-driven case with its own "contains N GGUF files" error
    //    and loads family-specific layouts instead, so the footprint is
    //    indeterminate and the caller skips the guard rather than answer 503.
    if (std::filesystem::is_regular_file(model.path, ec)) {
        add_file(model.path);
    } else if (std::filesystem::is_directory(model.path, ec)) {
        // One listing of the directory, mirroring the loader's own selection:
        // model.gguf wins, the sole *.gguf is used alone, and several GGUFs
        // without model.gguf are ambiguous.
        const auto ggufs = engine::assets::directory_gguf_files(model.path);
        std::optional<std::filesystem::path> selected;
        for (const auto & gguf : ggufs) {
            if (gguf.filename() == "model.gguf") {
                selected = gguf;
                break;
            }
        }
        if (!selected.has_value() && ggufs.size() == 1) {
            selected = ggufs.front();
        }
        if (selected.has_value()) {
            add_file(*selected);
        } else if (!ggufs.empty()) {
            return std::nullopt;
        } else {
            add_tree(model.path, 0);
        }
    }
    // Relative auxiliary paths resolve against the model directory when model.path
    // is a directory, and against the model file's parent when it is a file.
    const std::filesystem::path aux_base =
        std::filesystem::is_directory(model.path, ec) ? model.path : model.path.parent_path();
    for (const auto & option : model.session_options) {
        // A session option names an auxiliary artifact, so its value is scanned as a path. An
        // `exec_plan` value is JSON rather than a path, and the engines it declares are reported by
        // the model below: the scan sees a string that is no file and contributes nothing.
        std::filesystem::path aux(option.second);
        if (aux.is_relative()) {
            aux = aux_base / aux;
        }
        add_tree(aux, 0);
    }
    // The family's own resource plan: the compiled engines its execution plans declare, priced by
    // the target runtime. Priced apart from the weights because a family can cap the set its engines
    // hold resident at once, and that ceiling must not meet the runtime factor that prices ggml
    // graphs and KV state.
    const auto resources = engine::runtime::make_default_registry().model_resources(
        model.family, model.path, model.session_options);
    for (const auto & requirement : resources.engines) {
        add_engine(requirement.path, requirement.resident_mib);
    }
    if (resources.resident_cap_mib > 0) {
        aperture = std::min(
            aperture, static_cast<size_t>(resources.resident_cap_mib) * (1024u * 1024u));
    }
    // Weights plus a runtime factor for Metal/GPU buffers, activation graphs and
    // KV state, plus a fixed floor for per-model bookkeeping.
    constexpr double kRuntimeOverheadFactor = 1.5;
    constexpr size_t kFixedOverhead = 128ull * 1024 * 1024;
    return static_cast<size_t>(static_cast<double>(weights) * kRuntimeOverheadFactor) + kFixedOverhead +
        aperture;
}

}  // namespace minitts::server
