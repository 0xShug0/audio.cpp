#include "engine/framework/tokenizers/sentencepiece.h"

#include <iostream>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

std::filesystem::path test_asset_path(const std::string & relative) {
    return std::filesystem::path(ENGINE_TEST_ASSET_ROOT) / relative;
}

void require(bool condition, const std::string & message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

// Citrinet CTC decoding maps token ids straight into the tokenizer's piece
// table, so the table must be exactly id-aligned with the SentencePiece
// model. A table that drops or reorders entries (NeMo's sidecar vocab.txt
// omits <unk> at id 0) yields plausible-looking but wrong words instead of
// an error, which is why the runtime decodes through tokenizer.model.
void test_ctc_id_decode_round_trip() {
    const auto pieces = engine::tokenizers::load_sentencepiece_model(
        test_asset_path("tokenizers/tokenizer-1.model"));

    // The id-alignment contract the vocab.txt sidecar violated: the piece
    // table must include the model's <unk> entry at id 0, not start at the
    // first normal piece.
    require(!pieces.empty(), "tokenizer model contained no pieces");
    require(pieces[0].text == "<unk>", "expected piece 0 to be <unk>");
    require(
        pieces[0].type == engine::tokenizers::SentencePieceType::Unknown,
        "expected piece 0 to have type Unknown");

    const std::string phrase = "hello world how are you";
    const auto ids = engine::tokenizers::tokenize_sentencepiece(pieces, phrase);
    require(!ids.empty(), "tokenizer produced no ids for the test phrase");

    const auto decoded = engine::tokenizers::decode_sentencepiece(pieces, ids);
    require(
        decoded == phrase,
        "decode_sentencepiece round trip failed: got '" + decoded + "'");

    // Shifting every id by one simulates a piece table that lost its first
    // entry, the failure mode of NeMo's sidecar vocab.txt. The result must
    // be different text, not a silent reproduction of the phrase.
    std::vector<int32_t> shifted;
    shifted.reserve(ids.size());
    for (const int32_t id : ids) {
        if (static_cast<size_t>(id) + 1 < pieces.size()) {
            shifted.push_back(id + 1);
        }
    }
    require(shifted.size() == ids.size(), "shifted id set lost entries");
    const auto misaligned = engine::tokenizers::decode_sentencepiece(pieces, shifted);
    require(
        misaligned != phrase,
        "shifted ids unexpectedly decoded to the original phrase");
}

}  // namespace

int main() {
    try {
        test_ctc_id_decode_round_trip();
    } catch (const std::exception & ex) {
        std::cerr << "test_citrinet_tokenizer_decode failed: " << ex.what() << "\n";
        return 1;
    }
    std::cout << "test_citrinet_tokenizer_decode passed\n";
    return 0;
}
