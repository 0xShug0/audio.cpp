#include "text.h"

#include <algorithm>

namespace dizione {

void append_utf8(std::string & out, char32_t code) {
    if (code < 0x80) {
        out.push_back(static_cast<char>(code));
    } else if (code < 0x800) {
        out.push_back(static_cast<char>(0xC0 | (code >> 6)));
        out.push_back(static_cast<char>(0x80 | (code & 0x3F)));
    } else if (code < 0x10000) {
        out.push_back(static_cast<char>(0xE0 | (code >> 12)));
        out.push_back(static_cast<char>(0x80 | ((code >> 6) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | (code & 0x3F)));
    } else {
        out.push_back(static_cast<char>(0xF0 | (code >> 18)));
        out.push_back(static_cast<char>(0x80 | ((code >> 12) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | ((code >> 6) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | (code & 0x3F)));
    }
}

std::string encode(StrView text) {
    std::string out;
    out.reserve(text.size());
    for (const char32_t code : text) {
        append_utf8(out, code);
    }
    return out;
}

Str decode(std::string_view utf8) {
    Str out;
    out.reserve(utf8.size());
    const auto * data = reinterpret_cast<const unsigned char *>(utf8.data());
    const size_t size = utf8.size();
    size_t i = 0;
    while (i < size) {
        const unsigned char lead = data[i];
        if (lead < 0x80) {
            out.push_back(lead);
            ++i;
            continue;
        }
        int extra = 0;
        char32_t code = 0;
        char32_t least = 0;
        if (lead >= 0xC2 && lead <= 0xDF) {
            extra = 1;
            code = lead & 0x1F;
            least = 0x80;
        } else if (lead >= 0xE0 && lead <= 0xEF) {
            extra = 2;
            code = lead & 0x0F;
            least = 0x800;
        } else if (lead >= 0xF0 && lead <= 0xF4) {
            extra = 3;
            code = lead & 0x07;
            least = 0x10000;
        }
        bool valid = extra != 0 && i + static_cast<size_t>(extra) < size;
        if (valid) {
            for (int k = 1; k <= extra; ++k) {
                const unsigned char next = data[i + static_cast<size_t>(k)];
                if ((next & 0xC0) != 0x80) {
                    valid = false;
                    break;
                }
                code = (code << 6) | (next & 0x3F);
            }
        }
        if (valid && (code < least || code > 0x10FFFF || (code >= 0xD800 && code <= 0xDFFF))) {
            valid = false;
        }
        if (!valid) {
            out.push_back(0xFFFD);
            ++i;
            continue;
        }
        out.push_back(code);
        i += static_cast<size_t>(extra) + 1;
    }
    return out;
}

Str S(std::string_view text) {
    bool ascii = true;
    for (const char c : text) {
        if (static_cast<unsigned char>(c) >= 0x80) {
            ascii = false;
            break;
        }
    }
    if (!ascii) {
        return decode(text);
    }
    Str out(text.size(), U'\0');
    for (size_t i = 0; i < text.size(); ++i) {
        out[i] = static_cast<unsigned char>(text[i]);
    }
    return out;
}

namespace py {

namespace {

template <typename Test>
bool all_nonempty(StrView text, Test test) {
    if (text.empty()) {
        return false;
    }
    for (const char32_t c : text) {
        if (!test(c)) {
            return false;
        }
    }
    return true;
}

bool in_set(StrView chars, char32_t c) {
    return chars.find(c) != StrView::npos;
}

std::string shown(StrView text) {
    return "'" + encode(text) + "'";
}

}  // namespace

bool isalpha(StrView text) { return all_nonempty(text, uni::is_alpha); }
bool isdigit(StrView text) { return all_nonempty(text, uni::is_digit); }
bool isalnum(StrView text) { return all_nonempty(text, uni::is_alnum); }
bool isspace(StrView text) { return all_nonempty(text, uni::is_space); }

bool is_ascii(StrView text) {
    for (const char32_t c : text) {
        if (c >= 0x80) {
            return false;
        }
    }
    return true;
}

bool isupper(StrView text) {
    bool cased = false;
    for (const char32_t c : text) {
        if (uni::is_lower(c) || uni::is_title(c)) {
            return false;
        }
        if (!cased && uni::is_upper(c)) {
            cased = true;
        }
    }
    return cased;
}

bool islower(StrView text) {
    bool cased = false;
    for (const char32_t c : text) {
        if (uni::is_upper(c) || uni::is_title(c)) {
            return false;
        }
        if (!cased && uni::is_lower(c)) {
            cased = true;
        }
    }
    return cased;
}

char32_t at(StrView text, long index) {
    const long size = static_cast<long>(text.size());
    if (index < 0) {
        index += size;
    }
    if (index < 0 || index >= size) {
        throw IndexError("string index out of range");
    }
    return text[static_cast<size_t>(index)];
}

Str slice(StrView text, long start, long stop) {
    const long size = static_cast<long>(text.size());
    if (start == kNone) {
        start = 0;
    } else if (start < 0) {
        start = std::max(0L, start + size);
    } else {
        start = std::min(start, size);
    }
    if (stop == kNone) {
        stop = size;
    } else if (stop < 0) {
        stop = std::max(0L, stop + size);
    } else {
        stop = std::min(stop, size);
    }
    if (stop <= start) {
        return Str();
    }
    return Str(text.substr(static_cast<size_t>(start), static_cast<size_t>(stop - start)));
}

Str lstrip(StrView text) {
    size_t first = 0;
    while (first < text.size() && uni::is_space(text[first])) {
        ++first;
    }
    return Str(text.substr(first));
}

Str rstrip(StrView text) {
    size_t last = text.size();
    while (last > 0 && uni::is_space(text[last - 1])) {
        --last;
    }
    return Str(text.substr(0, last));
}

Str strip(StrView text) {
    size_t first = 0;
    size_t last = text.size();
    while (first < last && uni::is_space(text[first])) {
        ++first;
    }
    while (last > first && uni::is_space(text[last - 1])) {
        --last;
    }
    return Str(text.substr(first, last - first));
}

Str lstrip(StrView text, StrView chars) {
    size_t first = 0;
    while (first < text.size() && in_set(chars, text[first])) {
        ++first;
    }
    return Str(text.substr(first));
}

Str rstrip(StrView text, StrView chars) {
    size_t last = text.size();
    while (last > 0 && in_set(chars, text[last - 1])) {
        --last;
    }
    return Str(text.substr(0, last));
}

Str strip(StrView text, StrView chars) {
    const Str left = lstrip(text, chars);
    return rstrip(left, chars);
}

std::vector<Str> split(StrView text) {
    std::vector<Str> out;
    size_t i = 0;
    const size_t size = text.size();
    while (i < size) {
        while (i < size && uni::is_space(text[i])) {
            ++i;
        }
        if (i >= size) {
            break;
        }
        const size_t start = i;
        while (i < size && !uni::is_space(text[i])) {
            ++i;
        }
        out.emplace_back(text.substr(start, i - start));
    }
    return out;
}

std::vector<Str> split(StrView text, StrView sep, long maxsplit) {
    if (sep.empty()) {
        throw ValueError("empty separator");
    }
    std::vector<Str> out;
    size_t start = 0;
    while (maxsplit < 0 || static_cast<long>(out.size()) < maxsplit) {
        const size_t found = text.find(sep, start);
        if (found == StrView::npos) {
            break;
        }
        out.emplace_back(text.substr(start, found - start));
        start = found + sep.size();
    }
    out.emplace_back(text.substr(start));
    return out;
}

std::vector<Str> rsplit(StrView text, StrView sep, long maxsplit) {
    if (sep.empty()) {
        throw ValueError("empty separator");
    }
    std::vector<Str> out;
    size_t end = text.size();
    while (maxsplit < 0 || static_cast<long>(out.size()) < maxsplit) {
        if (end < sep.size()) {
            break;
        }
        const size_t found = text.rfind(sep, end - sep.size());
        if (found == StrView::npos) {
            break;
        }
        out.emplace_back(text.substr(found + sep.size(), end - found - sep.size()));
        end = found;
    }
    out.emplace_back(text.substr(0, end));
    std::reverse(out.begin(), out.end());
    return out;
}

std::vector<Str> splitlines(StrView text) {
    std::vector<Str> out;
    size_t start = 0;
    size_t i = 0;
    const size_t size = text.size();
    while (i < size) {
        const char32_t c = text[i];
        const bool boundary = c == U'\n' || c == U'\r' || c == 0x0B || c == 0x0C || c == 0x1C || c == 0x1D ||
            c == 0x1E || c == 0x85 || c == 0x2028 || c == 0x2029;
        if (!boundary) {
            ++i;
            continue;
        }
        out.emplace_back(text.substr(start, i - start));
        if (c == U'\r' && i + 1 < size && text[i + 1] == U'\n') {
            ++i;
        }
        ++i;
        start = i;
    }
    if (start < size) {
        out.emplace_back(text.substr(start));
    }
    return out;
}

Str join(StrView sep, const std::vector<Str> & parts) {
    Str out;
    for (size_t i = 0; i < parts.size(); ++i) {
        if (i != 0) {
            out.append(sep);
        }
        out.append(parts[i]);
    }
    return out;
}

bool startswith(StrView text, StrView prefix) {
    return text.size() >= prefix.size() && text.compare(0, prefix.size(), prefix) == 0;
}

bool endswith(StrView text, StrView suffix) {
    return text.size() >= suffix.size() && text.compare(text.size() - suffix.size(), suffix.size(), suffix) == 0;
}

bool startswith_any(StrView text, std::initializer_list<StrView> prefixes) {
    for (const StrView prefix : prefixes) {
        if (startswith(text, prefix)) {
            return true;
        }
    }
    return false;
}

bool endswith_any(StrView text, std::initializer_list<StrView> suffixes) {
    for (const StrView suffix : suffixes) {
        if (endswith(text, suffix)) {
            return true;
        }
    }
    return false;
}

bool contains(StrView text, StrView part) {
    return text.find(part) != StrView::npos;
}

long find(StrView text, StrView part) {
    const size_t found = text.find(part);
    return found == StrView::npos ? -1 : static_cast<long>(found);
}

Str replace(StrView text, StrView old_part, StrView new_part, long count) {
    if (old_part.empty()) {
        throw ValueError("replace() with an empty pattern is not supported");
    }
    Str out;
    size_t start = 0;
    long done = 0;
    while (count < 0 || done < count) {
        const size_t found = text.find(old_part, start);
        if (found == StrView::npos) {
            break;
        }
        out.append(text.substr(start, found - start));
        out.append(new_part);
        start = found + old_part.size();
        ++done;
    }
    out.append(text.substr(start));
    return out;
}

Str ljust(StrView text, size_t width, char32_t fill) {
    Str out(text);
    if (out.size() < width) {
        out.append(width - out.size(), fill);
    }
    return out;
}

Str repeat(StrView text, long count) {
    Str out;
    for (long i = 0; i < count; ++i) {
        out.append(text);
    }
    return out;
}

Parts partition(StrView text, StrView sep) {
    const size_t found = text.find(sep);
    if (found == StrView::npos) {
        return {Str(text), Str(), Str()};
    }
    return {Str(text.substr(0, found)), Str(sep), Str(text.substr(found + sep.size()))};
}

Parts rpartition(StrView text, StrView sep) {
    const size_t found = text.rfind(sep);
    if (found == StrView::npos) {
        return {Str(), Str(), Str(text)};
    }
    return {Str(text.substr(0, found)), Str(sep), Str(text.substr(found + sep.size()))};
}

long long to_int(StrView text) {
    const Str trimmed = strip(text);
    size_t i = 0;
    bool negative = false;
    if (i < trimmed.size() && (trimmed[i] == U'+' || trimmed[i] == U'-')) {
        negative = trimmed[i] == U'-';
        ++i;
    }
    long long value = 0;
    size_t significant = 0;
    bool previous_digit = false;
    for (; i < trimmed.size(); ++i) {
        const char32_t c = trimmed[i];
        if (c == U'_') {
            if (!previous_digit) {
                throw ValueError("invalid literal for int() with base 10: " + shown(text));
            }
            previous_digit = false;
            continue;
        }
        const int digit = uni::decimal_value(c);
        if (digit < 0) {
            throw ValueError("invalid literal for int() with base 10: " + shown(text));
        }
        if (value != 0 || digit != 0) {
            if (significant >= 18) {
                throw Error("OverflowError", "number too long for the port: " + shown(text));
            }
            ++significant;
        }
        value = value * 10 + digit;
        previous_digit = true;
    }
    if (!previous_digit) {
        throw ValueError("invalid literal for int() with base 10: " + shown(text));
    }
    return negative ? -value : value;
}

Str from_int(long long value) {
    return S(std::to_string(value));
}

}  // namespace py

}  // namespace dizione
