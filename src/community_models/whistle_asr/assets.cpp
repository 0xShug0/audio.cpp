#include "engine/community_models/whistle_asr/assets.h"

#include "engine/framework/io/json.h"
#include "engine/framework/model_spec/package.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace engine::community_models::whistle_asr {
namespace {

std::vector<float> read_f32_file(const std::filesystem::path & path, size_t count) {
    if (std::filesystem::file_size(path) != count * sizeof(float)) {
        throw std::runtime_error("Whistle asset has the wrong byte length: " + path.string());
    }
    std::vector<float> values(count);
    std::ifstream stream(path, std::ios::binary);
    if (!stream.read(reinterpret_cast<char *>(values.data()),
            static_cast<std::streamsize>(values.size() * sizeof(float)))) {
        throw std::runtime_error("Cannot read Whistle asset: " + path.string());
    }
    if (!std::all_of(values.begin(), values.end(), [](float value) { return std::isfinite(value); })) {
        throw std::runtime_error("Whistle asset contains non-finite values: " + path.string());
    }
    return values;
}

std::array<uint16_t, 512> read_permutation(const std::filesystem::path & path) {
    const auto values = read_f32_file(path, 512);
    std::array<uint16_t, 512> permutation{};
    std::array<bool, 512> used{};
    for (size_t index = 0; index < values.size(); ++index) {
        const float value = values[index];
        if (value < 0 || value >= 512 || std::floor(value) != value ||
            used[static_cast<size_t>(value)]) {
            throw std::runtime_error("Invalid Whistle Hadamard permutation: " + path.string());
        }
        const auto at = static_cast<uint16_t>(value);
        permutation[index] = at;
        used[at] = true;
    }
    return permutation;
}

uint32_t read_u32(const std::vector<uint8_t> & bytes, size_t & offset) {
    if (offset > bytes.size() || bytes.size() - offset < 4) {
        throw std::runtime_error("Truncated Whistle tokenizer");
    }
    const uint32_t value = static_cast<uint32_t>(bytes[offset]) |
        static_cast<uint32_t>(bytes[offset + 1]) << 8 |
        static_cast<uint32_t>(bytes[offset + 2]) << 16 |
        static_cast<uint32_t>(bytes[offset + 3]) << 24;
    offset += 4;
    return value;
}

std::vector<WhistleToken> read_tokenizer(const std::filesystem::path & path) {
    if (std::filesystem::file_size(path) != 109231) {
        throw std::runtime_error("Whistle tokenizer has the wrong byte length");
    }
    std::ifstream stream(path, std::ios::binary);
    const std::vector<uint8_t> data(
        (std::istreambuf_iterator<char>(stream)), std::istreambuf_iterator<char>());
    if (data.size() != 109231) {
        throw std::runtime_error("Cannot read Whistle tokenizer");
    }
    size_t offset = 0;
    const auto count = read_u32(data, offset);
    const auto pad = read_u32(data, offset);
    const auto eos = read_u32(data, offset);
    const auto bos = read_u32(data, offset);
    const auto unk = read_u32(data, offset);
    if (count != 8199 || pad != 0 || eos != 1 || bos != 2 || unk != 3 ||
        offset + 4 > data.size() || data[offset] != 0 || data[offset + 1] != 1 ||
        data[offset + 2] != 0 || data[offset + 3] != 0) {
        throw std::runtime_error("Whistle tokenizer header is unsupported");
    }
    offset += 4;
    std::vector<WhistleToken> tokens;
    tokens.reserve(count);
    for (size_t index = 0; index < count; ++index) {
        const uint32_t bits = read_u32(data, offset);
        if (data.size() - offset < 3) {
            throw std::runtime_error("Truncated Whistle tokenizer token");
        }
        const uint8_t type = data[offset++];
        const uint16_t length = static_cast<uint16_t>(data[offset]) |
            static_cast<uint16_t>(data[offset + 1]) << 8;
        offset += 2;
        float score;
        std::memcpy(&score, &bits, sizeof(score));
        if (type > 4 || !std::isfinite(score) || length == 0 || data.size() - offset < length) {
            throw std::runtime_error("Invalid Whistle tokenizer token");
        }
        tokens.push_back({
            std::string(reinterpret_cast<const char *>(data.data() + offset), length), score, type});
        offset += length;
    }
    if (offset != data.size()) {
        throw std::runtime_error("Whistle tokenizer has trailing data");
    }
    return tokens;
}

std::string replace_invalid_utf8(const std::string & bytes) {
    constexpr char kReplacement[] = "\xEF\xBF\xBD";
    std::string text;
    text.reserve(bytes.size());
    for (size_t offset = 0; offset < bytes.size();) {
        const auto first = static_cast<unsigned char>(bytes[offset]);
        size_t length = 0;
        unsigned char second_low = 0;
        unsigned char second_high = 0;
        if (first <= 0x7F) {
            length = 1;
        } else if (first >= 0xC2 && first <= 0xDF) {
            length = 2;
            second_low = 0x80;
            second_high = 0xBF;
        } else if (first == 0xE0) {
            length = 3;
            second_low = 0xA0;
            second_high = 0xBF;
        } else if ((first >= 0xE1 && first <= 0xEC) || (first >= 0xEE && first <= 0xEF)) {
            length = 3;
            second_low = 0x80;
            second_high = 0xBF;
        } else if (first == 0xED) {
            length = 3;
            second_low = 0x80;
            second_high = 0x9F;
        } else if (first == 0xF0) {
            length = 4;
            second_low = 0x90;
            second_high = 0xBF;
        } else if (first >= 0xF1 && first <= 0xF3) {
            length = 4;
            second_low = 0x80;
            second_high = 0xBF;
        } else if (first == 0xF4) {
            length = 4;
            second_low = 0x80;
            second_high = 0x8F;
        }

        size_t valid_prefix = 1;
        while (valid_prefix < length && offset + valid_prefix < bytes.size()) {
            const auto byte = static_cast<unsigned char>(bytes[offset + valid_prefix]);
            const bool valid = valid_prefix == 1
                ? byte >= second_low && byte <= second_high
                : (byte & 0xC0U) == 0x80U;
            if (!valid) {
                break;
            }
            ++valid_prefix;
        }
        if (valid_prefix == length) {
            text.append(bytes, offset, length);
        } else {
            text += kReplacement;
        }
        offset += valid_prefix;
    }
    return text;
}

}  // namespace

std::string decode_whistle_tokens(
    const std::vector<WhistleToken> & pieces, const std::vector<int32_t> & ids) {
    std::string bytes;
    for (const int32_t id : ids) {
        if (id < 0 || static_cast<size_t>(id) >= pieces.size()) {
            throw std::out_of_range("Whistle token ID is outside the vocabulary");
        }
        const auto & piece = pieces[static_cast<size_t>(id)];
        if (piece.type == 1 || piece.type == 2) {
            continue;
        }
        if (piece.type == 4) {
            if (piece.text.size() != 6 || piece.text.substr(0, 3) != "<0x" ||
                piece.text.back() != '>' ||
                !std::isxdigit(static_cast<unsigned char>(piece.text[3])) ||
                !std::isxdigit(static_cast<unsigned char>(piece.text[4]))) {
                throw std::runtime_error("Invalid Whistle byte-fallback token");
            }
            const auto value = std::stoi(piece.text.substr(3, 2), nullptr, 16);
            bytes.push_back(static_cast<char>(value));
            continue;
        }
        bytes += piece.text;
    }
    constexpr const char * kSpaceMarker = "\xE2\x96\x81";
    std::string output = replace_invalid_utf8(bytes);
    for (size_t index = 0; (index = output.find(kSpaceMarker, index)) != std::string::npos;) {
        output.replace(index, 3, " ");
        ++index;
    }
    return output;
}

std::shared_ptr<const WhistleAssets> load_whistle_assets(const std::filesystem::path & model_path) {
    auto result = std::make_shared<WhistleAssets>();
    result->resources = engine::model_spec::load_resource_bundle_for_family(model_path, "whistle_asr");
    result->weights = result->resources.open_tensor_source("weights", assets::TensorSourceOptions{true});
    if (!result->weights || result->weights->tensors().size() != 116) {
        throw std::runtime_error("Whistle requires the complete 116-tensor checkpoint");
    }
    const auto config = result->resources.parse_json("config");
    if (engine::io::json::optional_string(config, "model_type", "") != "whistle" ||
        engine::io::json::optional_i64(config, "sample_rate", 0) != 16000 ||
        engine::io::json::optional_i64(config, "max_audio_seconds", 0) != 30) {
        throw std::runtime_error("Unsupported Whistle checkpoint configuration");
    }
    engine::assets::require_tensor_shape(*result->weights, "embedding/embedding", {8199, 512});
    engine::assets::require_tensor_shape(*result->weights, "encoder/mhc_phi_res", {8, 2048, 16});
    engine::assets::require_tensor_shape(*result->weights, "stack/mhc_phi_res", {8, 2048, 16});
    result->mel_filterbank =
        read_f32_file(result->resources.require_file("mel_filterbank"), 257 * 80);
    result->hadamard_permutations = {
        read_permutation(result->resources.require_file("hadamard_perm1")),
        read_permutation(result->resources.require_file("hadamard_perm2")),
    };
    result->tokenizer_pieces = read_tokenizer(result->resources.require_file("tokenizer_blob"));
    return result;
}

}  // namespace engine::community_models::whistle_asr
