// ZONOS3 text normalization fallback, language codes, .npy voice files and phoneme-marker rejection.
#include "engine/community_models/zonos3/npy.h"
#include "engine/community_models/zonos3/textnorm.h"
#include "engine/community_models/zonos3/tokenizer.h"
#include "engine/framework/runtime/errors.h"

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace zonos3 = engine::models::zonos3;

namespace {

void expect(bool condition, const std::string & message) {
    if (!condition) throw std::runtime_error(message);
}

void expect_rejected(const std::function<void()> & action, const std::string & needle, const std::string & what) {
    try {
        action();
    } catch (const engine::runtime::InvalidRequestError & error) {
        expect(std::string(error.what()).find(needle) != std::string::npos,
               what + ": unexpected message: " + error.what());
        return;
    }
    throw std::runtime_error(what + " was accepted");
}

std::filesystem::path write_file(const std::string & name, const std::vector<std::byte> & bytes) {
    const auto path = std::filesystem::temp_directory_path() / name;
    std::ofstream out(path, std::ios::binary);
    out.write(reinterpret_cast<const char *>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    return path;
}

}  // namespace

int main() {
    try {
        // Whitespace runs holding a line break or tab become one space; plain space runs stay.
        expect(zonos3::zonos3_collapse_line_breaks("a\n\nb  c\t d") == "a b  c d", "line-break collapse");
        expect(zonos3::zonos3_collapse_line_breaks("x \x1c y") == "x y", "U+001C is Python whitespace");
        expect(zonos3::zonos3_normalize_text("keep\nme", "en_us", false) == "keep\nme", "disabled normalization");

        expect(zonos3::zonos3_language("en") == "en_us" && zonos3::zonos3_language("zh") == "cmn" &&
                   zonos3::zonos3_language("pt") == "pt_br" && zonos3::zonos3_language("ko") == "ko",
               "language aliases");
        expect_rejected([] { (void) zonos3::zonos3_language("ru"); },
                        "zonos3 language must be one of en_us, en_gb, fr_fr, de, es, it, pt_br, ja, cmn, ko", "language ru");

        std::vector<float> embedding(192);
        for (size_t i = 0; i < embedding.size(); ++i) embedding[i] = 0.25f * static_cast<float>(i) - 7.0f;
        const auto embedding_path = write_file("zonos3_text_test_embedding.npy", zonos3::speaker_embedding_npy(embedding));
        expect(zonos3::read_speaker_embedding_npy(embedding_path) == embedding, "speaker embedding round trip");

        std::vector<int32_t> codes(7 * 10);
        for (size_t i = 0; i < codes.size(); ++i) codes[i] = static_cast<int32_t>((i * 577) % 4096);
        const auto codes_path = write_file("zonos3_text_test_codes.npy", zonos3::reference_codes_npy(codes));
        expect(zonos3::read_reference_codes_npy(codes_path, 900) == codes, "reference codes round trip");
        const auto first = zonos3::read_reference_codes_npy(codes_path, 3);
        expect(first == std::vector<int32_t>(codes.begin(), codes.begin() + 30), "reference codes frame limit");

        // A [191] float file and a code of 4096 are refused.
        auto short_bytes = zonos3::speaker_embedding_npy(embedding);
        const size_t header_size = static_cast<size_t>(short_bytes[8]) | (static_cast<size_t>(short_bytes[9]) << 8);
        const std::string header(reinterpret_cast<const char *>(short_bytes.data()) + 10, header_size);
        const auto shape_at = header.find("(192,)");
        expect(shape_at != std::string::npos, "speaker header shape");
        short_bytes[10 + shape_at + 3] = static_cast<std::byte>('1');
        short_bytes.resize(short_bytes.size() - 4);
        const auto short_path = write_file("zonos3_text_test_short.npy", short_bytes);
        expect_rejected([&] { (void) zonos3::read_speaker_embedding_npy(short_path); },
                        "speaker embedding must have shape [192] or [1,192]", "[191] embedding");
        auto bad_codes = codes;
        bad_codes[13] = 4096;
        const auto bad_path = write_file("zonos3_text_test_bad_codes.npy", zonos3::reference_codes_npy(bad_codes));
        expect_rejected([&] { (void) zonos3::read_reference_codes_npy(bad_path, 900); },
                        "reference codes must be ordinary integer codec IDs 0..4095", "code 4096");

        const std::string tokenizer_json =
            R"({"model":{"type":"BPE","vocab":{"<|endoftext|>":0,"<|pad|>":1,"a":2,"b":3,"ab":4},"merges":[["a","b"]]},)"
            R"("added_tokens":[{"id":0,"content":"<|endoftext|>"},{"id":1,"content":"<|pad|>"}]})";
        const zonos3::Zonos3Tokenizer tokenizer(tokenizer_json);
        expect(tokenizer.text_ids("ab") == std::vector<int32_t>({2, 8, 3}), "minimal tokenizer");
        expect(tokenizer.text_ids("a<|pad|>b") == std::vector<int32_t>({2, 6, 5, 7, 3}), "added token splits literally");
        expect_rejected([&] { (void) tokenizer.text_ids("say \xEE\x80\x80h\xC9\x99\xEE\x80\x81"); },
                        "This checkpoint has no phoneme tokens", "phoneme group");
        expect(zonos3::Zonos3Tokenizer::split("Hi  there 12345!!\n") ==
                   std::vector<std::string>({"Hi", " ", " there ", "123", "45", "!!\n"}),
               "pre-tokenizer pieces");

        for (const auto & path : {embedding_path, codes_path, short_path, bad_path}) std::filesystem::remove(path);
        std::cout << "zonos3 text tests passed\n";
        return 0;
    } catch (const std::exception & error) {
        std::cerr << "zonos3_text_test: " << error.what() << "\n";
        return 1;
    }
}
