// runtime::load_vad_model: the loader ASR VAD chunking uses for <family>.vad_model_path.

#include "engine/framework/runtime/registry.h"
#include "gguf.h"

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <optional>
#include <functional>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef ENGINE_REPO_ROOT
#define ENGINE_REPO_ROOT "."
#endif

namespace {

namespace rt = engine::runtime;
namespace fs = std::filesystem;

fs::path repo_path(const std::string & relative) {
    return fs::path(ENGINE_REPO_ROOT) / relative;
}

void require(bool condition, const std::string & message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

std::string family_of(const fs::path & path, const std::optional<std::string> & hint = std::nullopt) {
    rt::ModelLoadRequest request;
    request.model_path = path;
    request.family_hint = hint;
    return rt::load_vad_model(request)->metadata().family;
}

std::string error_of(const std::function<void()> & action) {
    try {
        action();
    } catch (const std::exception & error) {
        return error.what();
    }
    return {};
}

struct TempDir {
    fs::path path;
    explicit TempDir(const std::string & name) : path(fs::temp_directory_path() / name) {
        fs::remove_all(path);
        fs::create_directories(path);
    }
    ~TempDir() {
        std::error_code ec;
        fs::remove_all(path, ec);
    }
};

// A valid safetensors file with one F32 tensor named "x" (not Silero weights).
void write_tiny_safetensors(const fs::path & path) {
    std::string header = R"({"x":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}})";
    header.append((8 - header.size() % 8) % 8, ' ');
    std::ofstream out(path, std::ios::binary);
    const uint64_t size = header.size();
    out.write(reinterpret_cast<const char *>(&size), sizeof(size));
    out.write(header.data(), static_cast<std::streamsize>(header.size()));
    const float value = 0.0f;
    out.write(reinterpret_cast<const char *>(&value), sizeof(value));
}

const fs::path kSilero = repo_path("assets/framework/models/silero_vad/silero_vad_16k.safetensors");

void test_bundled_models() {
    require(family_of(repo_path("assets/framework/models/silero_vad")) == "silero_vad", "silero dir");
    require(family_of(kSilero) == "silero_vad", "silero file");
    require(family_of(repo_path("assets/framework/models/marblenet_vad")) == "marblenet_vad", "marblenet dir");
    require(family_of(repo_path("assets/framework/models/silero_vad"), "silero_vad") == "silero_vad", "hint");
}

void test_renamed_silero_file() {
    // The Silero-only loader accepted Silero weights under any file name.
    TempDir dir("audiocpp_vad_loader_renamed");
    fs::copy_file(kSilero, dir.path / "my_vad.safetensors");
    require(family_of(dir.path / "my_vad.safetensors") == "silero_vad", "renamed silero file");
}

void test_sibling_package_does_not_claim_file() {
    // A package whose loader resolves a file to its folder (pulsevad) sits next to the named
    // VAD file; the named file must still decide the family.
    TempDir dir("audiocpp_vad_loader_sibling");
    fs::copy_file(kSilero, dir.path / "silero_vad_16k.safetensors");
    fs::copy_file(kSilero, dir.path / "my_vad.safetensors");
    fs::copy_file(kSilero, dir.path / "pulsevad-f32.safetensors");
    require(family_of(dir.path / "silero_vad_16k.safetensors") == "silero_vad", "exact silero file next to package");
    require(family_of(dir.path / "my_vad.safetensors") == "silero_vad", "renamed silero file next to package");
}

void test_rejections() {
    const auto non_model = error_of([] { (void)family_of(repo_path("README.md")); });
    require(!non_model.empty(), "a non-model file was accepted");
    require(non_model.find("built VAD families") != std::string::npos, "error lists built VAD families: " + non_model);

    const auto non_vad_dir = error_of([] { (void)family_of(repo_path("docs")); });
    require(!non_vad_dir.empty(), "a non-VAD directory was accepted");

    const auto missing = error_of([] { (void)family_of(repo_path("assets/framework/models/does_not_exist")); });
    require(!missing.empty(), "a missing path was accepted");

    // Weights of a VAD family this binary lacks: say so instead of a Silero tensor error.
    TempDir dir("audiocpp_vad_loader_foreign");
    write_tiny_safetensors(dir.path / "other_vad.safetensors");
    const auto foreign = error_of([&] { (void)family_of(dir.path / "other_vad.safetensors"); });
    require(foreign.find("no Silero VAD tensors") != std::string::npos &&
                foreign.find("built VAD families") != std::string::npos,
            "foreign weights error: " + foreign);

    const auto bad_hint = error_of([] { (void)family_of(kSilero, std::string("not_a_vad_family")); });
    require(bad_hint.find("not built into this binary") != std::string::npos, "unknown hint error: " + bad_hint);
}

void test_standalone_embedded_vad() {
    if (!rt::make_default_registry().supports_family("pulsevad")) {
        std::cout << "[SKIP] standalone_embedded_vad: pulsevad is not built\n";
        return;
    }
    TempDir dir("audiocpp_vad_loader_standalone");
    std::ifstream spec(repo_path("model_specs/pulsevad.json"));
    require(spec.good(), "open PulseVAD spec fixture");
    const std::string json((std::istreambuf_iterator<char>(spec)), std::istreambuf_iterator<char>());
    const auto path = dir.path / "vad.gguf";
    auto * gguf = gguf_init_empty();
    gguf_set_val_u32(gguf, "audiocpp.model_spec.version", 1);
    gguf_set_val_str(gguf, "audiocpp.model_spec.family", "pulsevad");
    gguf_set_val_str(gguf, "audiocpp.model_spec.json", json.c_str());
    const bool written = gguf_write_to_file(gguf, path.string().c_str(), true);
    gguf_free(gguf);
    require(written, "write embedded VAD spec fixture");

    const auto previous = fs::current_path();
    fs::current_path(dir.path);
    const auto error = error_of([&] {
        require(family_of(path) == "pulsevad", "standalone embedded VAD family");
    });
    fs::current_path(previous);
    require(error.empty(), "standalone embedded VAD: " + error);
}

}  // namespace

int main() {
    const std::vector<std::pair<const char *, void (*)()>> tests = {
        {"bundled_models", test_bundled_models},
        {"renamed_silero_file", test_renamed_silero_file},
        {"sibling_package_does_not_claim_file", test_sibling_package_does_not_claim_file},
        {"rejections", test_rejections},
        {"standalone_embedded_vad", test_standalone_embedded_vad},
    };
    int failures = 0;
    for (const auto & [name, test] : tests) {
        try {
            test();
            std::cout << "[PASS] " << name << "\n";
        } catch (const std::exception & error) {
            std::cerr << "[FAIL] " << name << ": " << error.what() << "\n";
            ++failures;
        }
    }
    return failures == 0 ? 0 : 1;
}
