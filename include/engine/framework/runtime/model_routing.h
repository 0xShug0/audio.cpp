#pragma once

#include <cstddef>
#include <stdexcept>
#include <string>

namespace engine::runtime {

template<class Key, class Impl>
struct ModelRouteEntry {
    Key key;
    const Impl * impl;
};

// A model-local, immutable routing table. Entries and implementations must
// outlive the table. Resolve at the load/session/request boundary and retain
// the returned implementation; graph execution must not repeat the lookup.
template<class Key, class Impl>
class ModelRouteTable {
public:
    using Entry = ModelRouteEntry<Key, Impl>;

    template<std::size_t N>
    ModelRouteTable(const char * label, const Entry (&entries)[N])
        : entries_(entries), size_(N), label_(label) {
        for (std::size_t i = 0; i < size_; ++i) {
            if (!entries_[i].impl)
                throw std::invalid_argument(label_ + ": null implementation registration");
            for (std::size_t j = 0; j < i; ++j) {
                if (entries_[i].key == entries_[j].key)
                    throw std::invalid_argument(label_ + ": duplicate implementation registration");
            }
        }
    }

    const Impl & require(const Key & key) const {
        for (std::size_t i = 0; i < size_; ++i) {
            if (entries_[i].key == key) return *entries_[i].impl;
        }
        throw std::invalid_argument(label_ + ": unregistered implementation key");
    }

private:
    const Entry * entries_;
    std::size_t size_;
    std::string label_;
};

} // namespace engine::runtime
