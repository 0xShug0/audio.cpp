#include "engine/community_models/tiny_voca/frontend.h"

#include "dizione/dizione.h"
#include "dizione/pyre.h"

#include <stdexcept>
#include <utility>

namespace engine::models::tiny_voca {
namespace {

using dizione::Str;
using dizione::StrView;
namespace py = dizione::py;
namespace pyre = dizione::pyre;

// The patterns of tiny-voca's speak_text (tiny_voca/infer.py), word for word.
const pyre::Regex & space() {
    static const pyre::Regex value(R"re([\s\U0000FEFF]+)re");
    return value;
}

const pyre::Regex & sentence_end() {
    static const pyre::Regex value(R"re((?:(?<=[.!?…])|(?<=[.!?…]["')\]]))\s+(?=["'(\[]?[A-Z0-9]))re");
    return value;
}

const pyre::Regex & not_an_end() {
    static const pyre::Regex value(R"re((?:\b(?:mr|mrs|ms|dr|prof|sr|jr|st|vs|etc|no|inc|ltd|co|e\.g|i\.e|a\.m|p\.m)|\b[A-Z])\.$)re",
                                   pyre::IGNORECASE | pyre::ASCII);
    return value;
}

bool has_alnum(StrView text) {
    for (const char32_t c : text) {
        if (dizione::uni::is_alnum(c)) {
            return true;
        }
    }
    return false;
}

dizione::Data phonemizer_data(const TinyVocaAssets & assets) {
    dizione::Data data;
    data.lexicon = assets.resources.read_text("lexicon");
    data.align = assets.resources.read_text("align");
    data.g2p = assets.resources.read_text("g2p");
    data.homographs = assets.resources.read_text("homographs");
    for (const auto & tensor : assets.g2p_weights->tensors()) {
        dizione::Tensor values;
        for (const int64_t size : tensor.shape) {
            values.shape.push_back(static_cast<size_t>(size));
        }
        values.values = assets.g2p_weights->require_f32(tensor.name, std::optional<std::vector<int64_t>>(tensor.shape));
        data.g2p_weights.emplace(tensor.name, std::move(values));
    }
    return data;
}

}  // namespace

struct TinyVocaFrontend::State {
    explicit State(dizione::Data data) : phonemizer(std::move(data)) {}
    dizione::Phonemizer phonemizer;
};

TinyVocaFrontend::TinyVocaFrontend(const TinyVocaAssets & assets) {
    if (assets.config.phonemizer_version != dizione::kVersion) {
        throw std::runtime_error("tiny-voca: the model was trained with another version of the phonemizer's tokens");
    }
    if (assets.config.symbols != dizione::symbols()) {
        throw std::runtime_error("tiny-voca: the phoneme inventory of the model is not the phonemizer's");
    }
    try {
        state_ = std::make_unique<State>(phonemizer_data(assets));
    } catch (const dizione::Error & error) {
        throw std::runtime_error(std::string("tiny-voca: cannot load the phonemizer data: ") + error.what());
    }
}

TinyVocaFrontend::~TinyVocaFrontend() = default;

std::optional<TinyVocaFrontend::Utterance> TinyVocaFrontend::phonemize(const std::string & text) {
    try {
        const dizione::Phonemized result = state_->phonemizer.phonemize(text);
        Utterance out;
        out.ids.assign(result.ids.begin(), result.ids.end());
        out.stress.assign(result.stress.begin(), result.stress.end());
        return out;
    } catch (const dizione::ValueError &) {
        return std::nullopt;                // an empty text, or nothing in it to pronounce
    } catch (const dizione::Error & error) {
        throw std::runtime_error(std::string("tiny-voca: the phonemizer could not read the text: ") + error.what());
    }
}

std::vector<std::string> TinyVocaFrontend::sentences(const std::string & text) {
    const Str flat = py::strip(space().sub(U" ", dizione::decode(text)));
    std::vector<std::string> parts;
    Str current;
    auto keep = [&](const Str & part) {
        if (has_alnum(part)) {
            parts.push_back(dizione::encode(part));
        }
    };
    for (const Str & piece : sentence_end().split(flat)) {
        current = current.empty() ? piece : current + U" " + piece;
        if (!not_an_end().search(current)) {
            keep(current);
            current.clear();
        }
    }
    if (!current.empty()) {
        keep(current);
    }
    return parts;
}

std::string TinyVocaFrontend::finished(const std::string & text) {
    static const pyre::Regex text_end(R"re([\s\U0000FEFF]+$)re");
    const Str body = text_end.sub(U"", dizione::decode(text));
    if (body.empty()) {
        return text;
    }
    const Str last = space().split(body).back();
    // these may stand after the last mark: She said "go!"
    const Str closing = {U'"', U'\'', 0x201D, 0x2019, U')', U']', 0xBB};
    bool marked = false;
    if (has_alnum(last)) {
        const Str core = py::rstrip(last, closing);
        const char32_t end = core.empty() ? U'\0' : core.back();
        marked = end == U'.' || end == U'!' || end == U'?' || end == 0x2026 || end == U',' || end == U';' || end == U':';
    } else {
        for (const char32_t c : last) {
            marked = marked || c == U'.' || c == U'!' || c == U'?' || c == 0x2026;
        }
        marked = marked || last == U"," || last == U";" || last == U":";
    }
    return marked ? text : dizione::encode(body) + ".";
}

std::vector<std::string> TinyVocaFrontend::shorter(const std::string & text) {
    static const pyre::Regex clause_end(R"re((?<=[,;:—])\s+)re");
    const Str whole = dizione::decode(text);
    std::vector<std::string> out;
    const std::vector<Str> pieces = clause_end.split(whole);
    if (pieces.size() > 1) {
        for (const Str & piece : pieces) {
            out.push_back(dizione::encode(piece));
        }
        return out;
    }
    const std::vector<Str> words = py::split(whole, U" ");
    if (words.size() < 2) {
        return out;
    }
    const size_t half = (words.size() + 1) / 2;
    out.push_back(dizione::encode(py::join(U" ", std::vector<Str>(words.begin(), words.begin() + static_cast<std::ptrdiff_t>(half)))));
    out.push_back(dizione::encode(py::join(U" ", std::vector<Str>(words.begin() + static_cast<std::ptrdiff_t>(half), words.end()))));
    return out;
}

}  // namespace engine::models::tiny_voca
