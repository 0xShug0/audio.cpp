// A small JSON reader for dizione's data files (align.json, g2p.json,
// homographs.json). Numbers keep the distinction Python makes between whole
// numbers and floats.
//
// Part of the dizione C++ port. Standard library only.
#pragma once

#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace dizione::json {

struct Value {
    enum Kind { Null, Bool, Int, Float, String, Array, Object };
    Kind kind = Null;
    bool boolean = false;
    long long integer = 0;
    double number = 0.0;                    // the value of an Int or a Float
    std::string string;                     // UTF-8
    std::vector<Value> items;
    std::vector<std::pair<std::string, Value>> members;     // in the order written

    bool is_number() const { return kind == Int || kind == Float; }
    // The member of an object, or nullptr.
    const Value * find(std::string_view key) const;
    // The member of an object; raises KeyError when it is missing.
    const Value & at(std::string_view key) const;
};

// Raises ValueError for text that is not JSON.
Value parse(std::string_view text);

}  // namespace dizione::json
