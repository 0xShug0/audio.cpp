// Token-level parity test for the zipvoice Chinese frontend (emilia mode):
// compares EmiliaTokenizer::encode against reference ids dumped from the
// upstream tokenizer by tests/zipvoice/zh_reference_ids.py.
//
// Usage: zipvoice_zh_tokens <reference_ids.json> <model-dir>
#include "engine/community_models/zipvoice/emilia_tokenizer.h"

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

std::string slurp(const std::string & path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open " + path);
    return std::string(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
}

// Minimal JSON parsing for the flat {"text": {"ids": [...], "tokens": [...]}}
// file (corpus keys contain no escaped quotes).
struct Case { std::string text; std::vector<int32_t> ids; };

std::vector<Case> parse_cases(const std::string & json) {
    std::vector<Case> cases;
    size_t pos = 0;
    while (true) {
        const auto ids_key = json.find("\"ids\"", pos);
        if (ids_key == std::string::npos) break;
        const auto colon = json.rfind(':', ids_key);
        const auto key_end = json.rfind('"', colon);
        const auto key_start = json.rfind('"', key_end - 1);
        if (colon == std::string::npos || key_end == std::string::npos ||
            key_start == std::string::npos) break;
        Case entry;
        entry.text = json.substr(key_start + 1, key_end - key_start - 1);
        const auto ids_open = json.find('[', ids_key);
        const auto ids_close = json.find(']', ids_open);
        std::string ids_raw = json.substr(ids_open + 1, ids_close - ids_open - 1);
        size_t start = 0;
        while (start < ids_raw.size()) {
            const auto comma = ids_raw.find(',', start);
            const auto piece = ids_raw.substr(start,
                comma == std::string::npos ? std::string::npos : comma - start);
            if (!piece.empty()) entry.ids.push_back(std::stoi(piece));
            if (comma == std::string::npos) break;
            start = comma + 1;
        }
        cases.push_back(std::move(entry));
        pos = ids_close;
    }
    return cases;
}

std::unordered_map<std::string, int32_t> load_vocab(const std::filesystem::path & dir) {
    std::ifstream in(dir / "tokens.txt");
    if (!in) throw std::runtime_error("cannot open tokens.txt in " + dir.string());
    std::unordered_map<std::string, int32_t> vocab;
    std::string line;
    while (std::getline(in, line)) {
        const auto tab = line.find('\t');
        if (tab == std::string::npos) continue;
        vocab.emplace(line.substr(0, tab), std::stoi(line.substr(tab + 1)));
    }
    return vocab;
}

}  // namespace

int main(int argc, char ** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: %s <reference_ids.json> <model-dir>\n", argv[0]);
        return 2;
    }
    using namespace engine::models::zipvoice;
    const std::filesystem::path model_dir = argv[2];
    const auto vocab = load_vocab(model_dir);
    EmiliaTokenizer tokenizer(
        EmiliaTokenizer::TablePaths::from_model_dir(model_dir), vocab,
        EmiliaTokenizer::EspeakConfig{
            "/opt/homebrew/lib/libespeak-ng.dylib", "", "en-us"});

    const auto cases = parse_cases(slurp(argv[1]));
    int failures = 0;
    for (const auto & c : cases) {
        const auto got = tokenizer.encode(c.text);
        bool ok = got.size() == c.ids.size();
        for (size_t i = 0; ok && i < got.size(); ++i) ok = got[i] == c.ids[i];
        if (!ok) {
            ++failures;
            std::printf("FAIL %-30s got=%zu want=%zu\n", c.text.c_str(), got.size(), c.ids.size());
            std::printf("  got :");
            for (int32_t v : got) std::printf(" %d", v);
            std::printf("\n  want:");
            for (int32_t v : c.ids) std::printf(" %d", v);
            std::printf("\n");
        } else {
            std::printf("OK   %s (%zu ids)\n", c.text.c_str(), got.size());
        }
    }
    std::printf("%d/%zu cases passed\n", static_cast<int>(cases.size() - failures), cases.size());
    return failures == 0 ? 0 : 1;
}
