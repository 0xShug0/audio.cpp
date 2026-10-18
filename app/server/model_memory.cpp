#include "model_memory.h"

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/io/json.h"

#include <filesystem>
#include <functional>
#include <set>
#include <string>
#include <system_error>
#include <vector>

namespace minitts::server {

std::optional<size_t> estimate_model_memory_bytes(const ServerModelConfig & model) {
    size_t weights = 0;
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
    for (const auto & [key, value] : model.session_options) {
        // A compiled engine declared through an exec_plan is resident weight memory the loader
        // reads, and the option's value is JSON, so the path scan below sees nothing: walk the
        // artifact strings instead. A plan is declared per component role, and CosyVoice3's flow
        // ladder packs several engines into one comma-separated value.
        if (key.size() >= 10 && key.compare(key.size() - 10, 10, ".exec_plan") == 0) {
            const auto plan = [&]() -> engine::io::json::Value {
                try {
                    return engine::io::json::parse(value);
                } catch (const std::exception &) {
                    // The session reports a malformed plan itself; the guard stays silent.
                    return {};
                }
            }();
            if (!plan.is_object()) {
                continue;
            }
            for (const auto & [_, component] : plan.as_object()) {
                if (!component.is_object()) {
                    continue;
                }
                const auto & members = component.as_object();
                const auto artifacts = members.find("artifacts");
                if (artifacts == members.end() || !artifacts->second.is_object()) {
                    continue;
                }
                for (const auto & [_, artifact] : artifacts->second.as_object()) {
                    if (!artifact.is_string()) {
                        continue;
                    }
                    const std::string & declared = artifact.as_string();
                    size_t start = 0;
                    while (start <= declared.size()) {
                        const auto end = declared.find(',', start);
                        const auto stop = end == std::string::npos ? declared.size() : end;
                        if (stop > start) {
                            std::filesystem::path engine_path =
                                std::filesystem::path(declared.substr(start, stop - start));
                            if (engine_path.is_relative()) {
                                engine_path = aux_base / engine_path;
                            }
                            add_file(engine_path);
                        }
                        if (end == std::string::npos) {
                            break;
                        }
                        start = end + 1;
                    }
                }
            }
            continue;
        }
        std::filesystem::path aux(value);
        if (aux.is_relative()) {
            aux = aux_base / aux;
        }
        add_tree(aux, 0);
    }
    // Weights plus a runtime factor for Metal/GPU buffers, activation graphs and
    // KV state, plus a fixed floor for per-model bookkeeping.
    constexpr double kRuntimeOverheadFactor = 1.5;
    constexpr size_t kFixedOverhead = 128ull * 1024 * 1024;
    return static_cast<size_t>(static_cast<double>(weights) * kRuntimeOverheadFactor) + kFixedOverhead;
}

}  // namespace minitts::server
