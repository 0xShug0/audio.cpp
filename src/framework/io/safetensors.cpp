#include "engine/framework/io/safetensors.h"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <unordered_set>
#include <utility>

namespace engine::io {
namespace {

void skip_ws(const std::string & text, size_t & pos) {
    while (pos < text.size() && std::isspace(static_cast<unsigned char>(text[pos])) != 0) {
        ++pos;
    }
}

std::string parse_string(const std::string & text, size_t & pos) {
    skip_ws(text, pos);
    if (pos >= text.size() || text[pos] != '"') {
        throw std::runtime_error("expected json string");
    }
    ++pos;
    std::string out;
    while (pos < text.size()) {
        const char ch = text[pos++];
        if (ch == '\\') {
            if (pos >= text.size()) {
                throw std::runtime_error("invalid string escape");
            }
            out.push_back(text[pos++]);
            continue;
        }
        if (ch == '"') {
            return out;
        }
        out.push_back(ch);
    }
    throw std::runtime_error("unterminated json string");
}

int64_t parse_int64(const std::string & text, size_t & pos) {
    skip_ws(text, pos);
    size_t end = pos;
    while (end < text.size() && (std::isdigit(static_cast<unsigned char>(text[end])) != 0 || text[end] == '-')) {
        ++end;
    }
    if (end == pos) {
        throw std::runtime_error("expected integer");
    }
    const auto value = std::stoll(text.substr(pos, end - pos));
    pos = end;
    return value;
}

std::vector<int64_t> parse_int_array(const std::string & text, size_t & pos) {
    skip_ws(text, pos);
    if (pos >= text.size() || text[pos] != '[') {
        throw std::runtime_error("expected '['");
    }
    ++pos;
    std::vector<int64_t> values;
    while (true) {
        skip_ws(text, pos);
        if (pos < text.size() && text[pos] == ']') {
            ++pos;
            return values;
        }
        values.push_back(parse_int64(text, pos));
        skip_ws(text, pos);
        if (pos < text.size() && text[pos] == ',') {
            ++pos;
            continue;
        }
        if (pos < text.size() && text[pos] == ']') {
            ++pos;
            return values;
        }
        throw std::runtime_error("invalid integer array");
    }
}

void expect_char(const std::string & text, size_t & pos, char expected) {
    skip_ws(text, pos);
    if (pos >= text.size() || text[pos] != expected) {
        throw std::runtime_error(std::string("expected '") + expected + "'");
    }
    ++pos;
}

void skip_json_value(const std::string & text, size_t & pos) {
    skip_ws(text, pos);
    if (pos >= text.size()) {
        throw std::runtime_error("unexpected end of json while skipping value");
    }
    const char ch = text[pos];
    if (ch == '"') {
        (void)parse_string(text, pos);
        return;
    }
    if (ch == '{') {
        ++pos;
        while (true) {
            skip_ws(text, pos);
            if (pos < text.size() && text[pos] == '}') {
                ++pos;
                return;
            }
            (void)parse_string(text, pos);
            expect_char(text, pos, ':');
            skip_json_value(text, pos);
            skip_ws(text, pos);
            if (pos < text.size() && text[pos] == ',') {
                ++pos;
                continue;
            }
            if (pos < text.size() && text[pos] == '}') {
                ++pos;
                return;
            }
            throw std::runtime_error("invalid json object while skipping value");
        }
    }
    if (ch == '[') {
        ++pos;
        while (true) {
            skip_ws(text, pos);
            if (pos < text.size() && text[pos] == ']') {
                ++pos;
                return;
            }
            skip_json_value(text, pos);
            skip_ws(text, pos);
            if (pos < text.size() && text[pos] == ',') {
                ++pos;
                continue;
            }
            if (pos < text.size() && text[pos] == ']') {
                ++pos;
                return;
            }
            throw std::runtime_error("invalid json array while skipping value");
        }
    }
    if (std::isdigit(static_cast<unsigned char>(ch)) != 0 || ch == '-') {
        (void)parse_int64(text, pos);
        return;
    }
    if (text.compare(pos, 4, "true") == 0) {
        pos += 4;
        return;
    }
    if (text.compare(pos, 5, "false") == 0) {
        pos += 5;
        return;
    }
    if (text.compare(pos, 4, "null") == 0) {
        pos += 4;
        return;
    }
    throw std::runtime_error("unsupported json value while skipping");
}

std::unordered_map<std::string, std::string> parse_string_map(const std::string & text, size_t & pos) {
    std::unordered_map<std::string, std::string> values;
    expect_char(text, pos, '{');
    while (true) {
        skip_ws(text, pos);
        if (pos < text.size() && text[pos] == '}') {
            ++pos;
            return values;
        }
        const auto key = parse_string(text, pos);
        expect_char(text, pos, ':');
        const auto value = parse_string(text, pos);
        values.emplace(key, value);
        skip_ws(text, pos);
        if (pos < text.size() && text[pos] == ',') {
            ++pos;
            continue;
        }
        if (pos < text.size() && text[pos] == '}') {
            ++pos;
            return values;
        }
        throw std::runtime_error("invalid string map object");
    }
}

void write_u64_le(std::ostream & output, uint64_t value) {
    unsigned char bytes[8];
    for (int i = 0; i < 8; ++i) {
        bytes[i] = static_cast<unsigned char>((value >> (i * 8)) & 0xffU);
    }
    output.write(reinterpret_cast<const char *>(bytes), sizeof(bytes));
}

std::string escape_json_string(const std::string & value) {
    std::string out;
    out.reserve(value.size() + 8);
    constexpr char kHex[] = "0123456789abcdef";
    for (unsigned char ch : value) {
        switch (ch) {
        case '\\':
            out += "\\\\";
            break;
        case '"':
            out += "\\\"";
            break;
        case '\b':
            out += "\\b";
            break;
        case '\f':
            out += "\\f";
            break;
        case '\n':
            out += "\\n";
            break;
        case '\r':
            out += "\\r";
            break;
        case '\t':
            out += "\\t";
            break;
        default:
            if (ch < 0x20) {
                out += "\\u00";
                out.push_back(kHex[(ch >> 4) & 0x0f]);
                out.push_back(kHex[ch & 0x0f]);
            } else {
                out.push_back(static_cast<char>(ch));
            }
            break;
        }
    }
    return out;
}

std::string shape_to_json(const std::vector<int64_t> & shape) {
    std::ostringstream out;
    out << "[";
    for (size_t i = 0; i < shape.size(); ++i) {
        if (i != 0) {
            out << ",";
        }
        out << shape[i];
    }
    out << "]";
    return out.str();
}

}  // namespace

SafeTensorIndex load_safetensors_index(const std::filesystem::path & path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("failed to open safetensors file: " + path.string());
    }

    uint64_t header_len = 0;
    input.read(reinterpret_cast<char *>(&header_len), sizeof(header_len));
    if (!input) {
        throw std::runtime_error("failed to read safetensors header length: " + path.string());
    }
    std::string header(static_cast<size_t>(header_len), '\0');
    input.read(header.data(), static_cast<std::streamsize>(header.size()));
    if (!input) {
        throw std::runtime_error("failed to read safetensors header: " + path.string());
    }

    SafeTensorIndex index;
    index.source_path = std::filesystem::weakly_canonical(path);
    index.header_bytes = sizeof(header_len) + header.size();

    size_t pos = 0;
    expect_char(header, pos, '{');
    while (true) {
        skip_ws(header, pos);
        if (pos < header.size() && header[pos] == '}') {
            ++pos;
            break;
        }

        const auto tensor_name = parse_string(header, pos);
        expect_char(header, pos, ':');
        if (tensor_name == "__metadata__") {
            skip_ws(header, pos);
            if (pos < header.size() && header[pos] == '{') {
                index.metadata = parse_string_map(header, pos);
            } else {
                // Some valid safetensors writers, including MLX, emit a null
                // __metadata__ value instead of omitting the key.
                skip_json_value(header, pos);
            }
            skip_ws(header, pos);
            if (pos < header.size() && header[pos] == ',') {
                ++pos;
                continue;
            }
            if (pos < header.size() && header[pos] == '}') {
                ++pos;
                break;
            }
            throw std::runtime_error("invalid safetensors metadata object");
        }
        expect_char(header, pos, '{');

        SafeTensorInfo info;
        info.name = tensor_name;
        while (true) {
            const auto field = parse_string(header, pos);
            expect_char(header, pos, ':');
            if (field == "dtype") {
                info.dtype = parse_string(header, pos);
            } else if (field == "shape") {
                info.shape = parse_int_array(header, pos);
            } else if (field == "data_offsets") {
                const auto offsets = parse_int_array(header, pos);
                if (offsets.size() != 2) {
                    throw std::runtime_error("safetensors data_offsets must contain 2 entries");
                }
                info.data_begin = static_cast<size_t>(offsets[0]);
                info.data_end = static_cast<size_t>(offsets[1]);
            } else {
                throw std::runtime_error("unsupported safetensors header field: " + field);
            }

            skip_ws(header, pos);
            if (pos < header.size() && header[pos] == ',') {
                ++pos;
                continue;
            }
            if (pos < header.size() && header[pos] == '}') {
                ++pos;
                break;
            }
            throw std::runtime_error("invalid safetensors tensor object");
        }

        index.tensors.emplace(info.name, std::move(info));
        skip_ws(header, pos);
        if (pos < header.size() && header[pos] == ',') {
            ++pos;
            continue;
        }
        if (pos < header.size() && header[pos] == '}') {
            ++pos;
            break;
        }
    }

    return index;
}

void write_safetensors_file(
    const std::filesystem::path & path,
    const std::vector<SafeTensorWriteEntry> & entries) {
    if (entries.empty()) {
        throw std::runtime_error("safetensors writer requires at least one tensor");
    }

    std::unordered_set<std::string> names;
    std::vector<std::pair<size_t, size_t>> offsets;
    offsets.reserve(entries.size());
    size_t cursor = 0;
    for (const auto & entry : entries) {
        if (entry.name.empty() || entry.dtype.empty()) {
            throw std::runtime_error("safetensors writer received an unnamed tensor or dtype");
        }
        if (!names.insert(entry.name).second) {
            throw std::runtime_error("safetensors writer received duplicate tensor: " + entry.name);
        }
        for (int64_t dim : entry.shape) {
            if (dim < 0) {
                throw std::runtime_error("safetensors writer received negative shape dimension: " + entry.name);
            }
        }
        offsets.emplace_back(cursor, cursor + entry.data.size());
        cursor += entry.data.size();
    }

    std::ostringstream header_stream;
    header_stream << "{";
    for (size_t i = 0; i < entries.size(); ++i) {
        if (i != 0) {
            header_stream << ",";
        }
        header_stream << "\"" << escape_json_string(entries[i].name) << "\":{";
        header_stream << "\"dtype\":\"" << escape_json_string(entries[i].dtype) << "\",";
        header_stream << "\"shape\":" << shape_to_json(entries[i].shape) << ",";
        header_stream << "\"data_offsets\":[" << offsets[i].first << "," << offsets[i].second << "]";
        header_stream << "}";
    }
    header_stream << "}";

    std::string header = header_stream.str();
    header.append((8 - (header.size() % 8)) % 8, ' ');

    const auto parent = path.parent_path();
    if (!parent.empty()) {
        std::filesystem::create_directories(parent);
    }
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    if (!output) {
        throw std::runtime_error("failed to open safetensors output file: " + path.string());
    }
    write_u64_le(output, static_cast<uint64_t>(header.size()));
    output.write(header.data(), static_cast<std::streamsize>(header.size()));
    for (const auto & entry : entries) {
        output.write(
            reinterpret_cast<const char *>(entry.data.data()),
            static_cast<std::streamsize>(entry.data.size()));
    }
}

}  // namespace engine::io
