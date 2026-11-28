#include "engine/models/irodori_tts/tokenizer_text.h"

#include "test_assert.h"

#include <cstdint>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using namespace engine::models::irodori_tts;
using engine::test::require;
namespace fs = std::filesystem;

// A small Unigram vocab. Single characters score -5, so a piece of four or more
// characters at -14 beats spelling it out, and is what the tokenizer picks
// unless the rare-piece split disables it.
const std::vector<std::pair<std::string, double>> & vocab() {
    static const std::vector<std::pair<std::string, double>> pieces = [] {
        std::vector<std::pair<std::string, double>> out = {
            {"<unk>", 0.0}, {"<s>", 0.0}, {"<pad>", 0.0}, {"\xE2\x96\x81", -5.0},
        };
        for (const char * ch : {"浦", "和", "レ", "ッ", "ズ", "瀬", "戸", "内", "海", "あ", "り", "が", "と", "う",
                                "野", "球", "場", "セ", "ブ", "ン", "イ"}) {
            out.emplace_back(ch, -5.0);
        }
        out.insert(out.end(), {
            {"浦和", -6.0},
            {"レッ", -6.0},
            {"浦和レッズ", -14.0},       // rare: disabled
            {"\xE2\x96\x81浦和レッズ", -14.0},  // rare, U+2581 not counted: disabled
            {"瀬戸内海", -14.0},         // in Python's KEEP_TOKENS: kept
            {"ありがとう", -14.0},       // hiragana only: kept
            {"野球場", -14.0},           // three characters: kept
            {"セブン", -6.0},
            {"イレブン", -6.0},
            {"セブンイレブン", -8.0},    // in Python's FORCE_SPLITS
        });
        return out;
    }();
    return pieces;
}

int32_t id_of(const std::string & piece) {
    const auto & pieces = vocab();
    for (size_t i = 0; i < pieces.size(); ++i) {
        if (pieces[i].first == piece) {
            return static_cast<int32_t>(i);
        }
    }
    throw std::runtime_error("piece missing from the test vocab: " + piece);
}

std::vector<int32_t> ids_of(const std::vector<std::string> & pieces) {
    std::vector<int32_t> ids = {id_of("<s>")};
    for (const auto & piece : pieces) {
        ids.push_back(id_of(piece));
    }
    return ids;
}

fs::path write_tokenizer() {
    const auto path = fs::temp_directory_path() /
        ("irodori-tokenizer-test-" + std::to_string(std::random_device{}()) + ".json");
    std::ofstream out(path, std::ios::binary);
    out << "{\"model\":{\"type\":\"Unigram\",\"unk_id\":0,\"vocab\":[";
    const auto & pieces = vocab();
    for (size_t i = 0; i < pieces.size(); ++i) {
        out << (i == 0 ? "" : ",") << "[\"" << pieces[i].first << "\"," << pieces[i].second << "]";
    }
    out << "]}}";
    return path;
}

IrodoriTextTokenizer make_tokenizer(const fs::path & path, const std::string & repo) {
    auto assets = std::make_shared<IrodoriTTSAssets>();
    assets->resources = engine::assets::ResourceBundle(path.parent_path());
    assets->resources.add_file("tokenizer_v4_json", path);
    assets->config.text_encoder_type = "pretrained";
    assets->config.text_tokenizer_repo = repo;
    assets->config.text_vocab_size = static_cast<int64_t>(vocab().size());
    return IrodoriTextTokenizer(assets);
}

void test_rare_pieces_are_split_in_the_text_only(const fs::path & path) {
    const auto tokenizer = make_tokenizer(path, "sbintuitions/modernbert-ja-310m");
    require(tokenizer.encode("浦和レッズ") == ids_of({"浦和", "レッ", "ズ"}), "rare piece split in the text");
    require(tokenizer.encode(" 浦和レッズ") == ids_of({"\xE2\x96\x81", "浦和", "レッ", "ズ"}),
            "rare piece with U+2581 split in the text");
    require(tokenizer.encode("瀬戸内海") == ids_of({"瀬戸内海"}), "KEEP_TOKENS piece kept");
    require(tokenizer.encode("ありがとう") == ids_of({"ありがとう"}), "hiragana-only piece kept");
    require(tokenizer.encode("野球場") == ids_of({"野球場"}), "three-character piece kept");
    require(tokenizer.encode("セブンイレブン") == ids_of({"セブン", "イレブン"}), "FORCE_SPLITS piece split");

    require(tokenizer.encode_caption("浦和レッズ") == ids_of({"浦和レッズ"}), "caption keeps the rare piece");
    require(tokenizer.encode_caption("セブンイレブン") == ids_of({"セブンイレブン"}),
            "caption keeps the FORCE_SPLITS piece");

    const auto text = tokenizer.encode_padded("浦和レッズ", 6);
    auto expected = ids_of({"浦和", "レッ", "ズ"});
    expected.insert(expected.end(), {id_of("<pad>"), id_of("<pad>")});
    require(text.token_ids == expected && text.mask == std::vector<uint8_t>({1, 1, 1, 1, 0, 0}),
            "padded text");
    const auto caption = tokenizer.encode_caption_padded("浦和レッズ", 3);
    require(caption.token_ids == ids_of({"浦和レッズ", "<pad>"}) && caption.mask == std::vector<uint8_t>({1, 1, 0}),
            "padded caption");
}

void test_other_tokenizers_keep_every_piece(const fs::path & path) {
    const auto tokenizer = make_tokenizer(path, "sbintuitions/sarashina2.2-0.5b");
    require(tokenizer.encode("浦和レッズ") == ids_of({"浦和レッズ"}), "other tokenizer keeps the rare piece");
    require(tokenizer.encode("セブンイレブン") == ids_of({"セブンイレブン"}),
            "other tokenizer keeps the FORCE_SPLITS piece");
}

}  // namespace

int main() {
    const auto path = write_tokenizer();
    try {
        test_rare_pieces_are_split_in_the_text_only(path);
        test_other_tokenizers_keep_every_piece(path);
        fs::remove(path);
        std::cout << "irodori_text_tokenizer_test passed\n";
        return 0;
    } catch (const std::exception & ex) {
        fs::remove(path);
        std::cerr << "irodori_text_tokenizer_test failed: " << ex.what() << "\n";
        return 1;
    }
}
