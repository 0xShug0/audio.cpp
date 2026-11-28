#include "json.h"

#include "text.h"

#include <locale>
#include <sstream>

namespace dizione::json {

const Value * Value::find(std::string_view key) const {
    for (const auto & member : members) {
        if (member.first == key) {
            return &member.second;
        }
    }
    return nullptr;
}

const Value & Value::at(std::string_view key) const {
    const Value * found = find(key);
    if (found == nullptr) {
        throw KeyError(std::string(key));
    }
    return *found;
}

namespace {

class Reader {
public:
    explicit Reader(std::string_view text) : text_(text) {}

    Value document() {
        Value value = element();
        space();
        if (at_ != text_.size()) {
            fail("extra data");
        }
        return value;
    }

private:
    [[noreturn]] void fail(const char * what) const {
        throw ValueError(std::string("JSON: ") + what + " at byte " + std::to_string(at_));
    }

    void space() {
        while (at_ < text_.size()) {
            const char c = text_[at_];
            if (c != ' ' && c != '\n' && c != '\r' && c != '\t') {
                break;
            }
            ++at_;
        }
    }

    bool word(std::string_view expected) {
        if (text_.substr(at_, expected.size()) == expected) {
            at_ += expected.size();
            return true;
        }
        return false;
    }

    unsigned hex4() {
        if (at_ + 4 > text_.size()) {
            fail("incomplete \\u escape");
        }
        unsigned value = 0;
        for (int k = 0; k < 4; ++k) {
            const char c = text_[at_++];
            value <<= 4;
            if (c >= '0' && c <= '9') {
                value |= static_cast<unsigned>(c - '0');
            } else if (c >= 'a' && c <= 'f') {
                value |= static_cast<unsigned>(c - 'a' + 10);
            } else if (c >= 'A' && c <= 'F') {
                value |= static_cast<unsigned>(c - 'A' + 10);
            } else {
                fail("bad \\u escape");
            }
        }
        return value;
    }

    std::string string() {
        std::string out;
        ++at_;                                          // the opening quote
        for (;;) {
            if (at_ >= text_.size()) {
                fail("unterminated string");
            }
            const char c = text_[at_++];
            if (c == '"') {
                return out;
            }
            if (c != '\\') {
                out.push_back(c);
                continue;
            }
            if (at_ >= text_.size()) {
                fail("unterminated string");
            }
            const char e = text_[at_++];
            switch (e) {
                case '"': out.push_back('"'); break;
                case '\\': out.push_back('\\'); break;
                case '/': out.push_back('/'); break;
                case 'b': out.push_back('\b'); break;
                case 'f': out.push_back('\f'); break;
                case 'n': out.push_back('\n'); break;
                case 'r': out.push_back('\r'); break;
                case 't': out.push_back('\t'); break;
                case 'u': {
                    char32_t code = hex4();
                    if (code >= 0xD800 && code <= 0xDBFF && text_.substr(at_, 2) == "\\u") {
                        const size_t back = at_;
                        at_ += 2;
                        const unsigned low = hex4();
                        if (low >= 0xDC00 && low <= 0xDFFF) {
                            code = 0x10000 + ((code - 0xD800) << 10) + (low - 0xDC00);
                        } else {
                            at_ = back;
                        }
                    }
                    append_utf8(out, code);
                    break;
                }
                default: fail("bad escape");
            }
        }
    }

    Value number() {
        const size_t start = at_;
        bool whole = true;
        if (at_ < text_.size() && text_[at_] == '-') {
            ++at_;
        }
        while (at_ < text_.size()) {
            const char c = text_[at_];
            if (c >= '0' && c <= '9') {
                ++at_;
            } else if (c == '.' || c == 'e' || c == 'E' || c == '+' || c == '-') {
                whole = false;
                ++at_;
            } else {
                break;
            }
        }
        const std::string digits(text_.substr(start, at_ - start));
        if (digits.empty() || digits == "-") {
            fail("bad number");
        }
        Value value;
        // The "C" locale, whatever the program has set, and correctly rounded.
        std::istringstream stream(digits);
        stream.imbue(std::locale::classic());
        if (whole && digits.size() <= 18) {
            value.kind = Value::Int;
            stream >> value.integer;
            value.number = static_cast<double>(value.integer);
        } else {
            value.kind = Value::Float;
            stream >> value.number;
        }
        if (stream.fail()) {
            fail("bad number");
        }
        return value;
    }

    Value element() {
        space();
        if (at_ >= text_.size()) {
            fail("unexpected end");
        }
        Value value;
        const char c = text_[at_];
        if (c == '{') {
            ++at_;
            value.kind = Value::Object;
            space();
            if (at_ < text_.size() && text_[at_] == '}') {
                ++at_;
                return value;
            }
            for (;;) {
                space();
                if (at_ >= text_.size() || text_[at_] != '"') {
                    fail("expected a name");
                }
                std::string key = string();
                space();
                if (at_ >= text_.size() || text_[at_] != ':') {
                    fail("expected a colon");
                }
                ++at_;
                value.members.emplace_back(std::move(key), element());
                space();
                if (at_ < text_.size() && text_[at_] == ',') {
                    ++at_;
                    continue;
                }
                if (at_ < text_.size() && text_[at_] == '}') {
                    ++at_;
                    return value;
                }
                fail("expected a comma or a closing brace");
            }
        }
        if (c == '[') {
            ++at_;
            value.kind = Value::Array;
            space();
            if (at_ < text_.size() && text_[at_] == ']') {
                ++at_;
                return value;
            }
            for (;;) {
                value.items.push_back(element());
                space();
                if (at_ < text_.size() && text_[at_] == ',') {
                    ++at_;
                    continue;
                }
                if (at_ < text_.size() && text_[at_] == ']') {
                    ++at_;
                    return value;
                }
                fail("expected a comma or a closing bracket");
            }
        }
        if (c == '"') {
            value.kind = Value::String;
            value.string = string();
            return value;
        }
        if (word("true")) {
            value.kind = Value::Bool;
            value.boolean = true;
            return value;
        }
        if (word("false")) {
            value.kind = Value::Bool;
            return value;
        }
        if (word("null")) {
            return value;
        }
        return number();
    }

    std::string_view text_;
    size_t at_ = 0;
};

}  // namespace

Value parse(std::string_view text) {
    return Reader(text).document();
}

}  // namespace dizione::json
