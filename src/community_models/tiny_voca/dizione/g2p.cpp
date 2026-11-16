#include "g2p.h"

#include <cmath>

namespace dizione {

namespace {

// A dot product summed in eight lanes, so that the order of the additions is
// fixed whatever the compiler does with the loop.
float dot(const float * a, const float * b, size_t size) {
    float lane[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    size_t i = 0;
    for (; i + 8 <= size; i += 8) {
        for (size_t k = 0; k < 8; ++k) {
            lane[k] += a[i + k] * b[i + k];
        }
    }
    float rest = 0;
    for (; i < size; ++i) {
        rest += a[i] * b[i];
    }
    return ((lane[0] + lane[1]) + (lane[2] + lane[3])) + ((lane[4] + lane[5]) + (lane[6] + lane[7])) + rest;
}

float sigmoid(float x) {
    return 1.0F / (1.0F + std::exp(-x));
}

// One direction of a GRU layer over `frames` rows of `width` values. The
// gates are in the order reset, update, new, as PyTorch stores them.
void gru(const std::vector<float> & x, size_t frames, size_t width, const Tensor & w_ih, const Tensor & w_hh, const Tensor & b_ih,
         const Tensor & b_hh, bool reverse, size_t hidden, float * out, size_t stride) {
    std::vector<float> inputs(frames * 3 * hidden);
    for (size_t t = 0; t < frames; ++t) {
        for (size_t g = 0; g < 3 * hidden; ++g) {
            inputs[t * 3 * hidden + g] = dot(&x[t * width], &w_ih.values[g * width], width) + b_ih.values[g];
        }
    }
    std::vector<float> state(hidden, 0.0F);
    std::vector<float> recurrent(3 * hidden);
    std::vector<float> next(hidden);
    for (size_t step = 0; step < frames; ++step) {
        const size_t t = reverse ? frames - 1 - step : step;
        for (size_t g = 0; g < 3 * hidden; ++g) {
            recurrent[g] = dot(&w_hh.values[g * hidden], state.data(), hidden) + b_hh.values[g];
        }
        const float * in = &inputs[t * 3 * hidden];
        for (size_t h = 0; h < hidden; ++h) {
            const float reset = sigmoid(in[h] + recurrent[h]);
            const float update = sigmoid(in[hidden + h] + recurrent[hidden + h]);
            const float candidate = std::tanh(in[2 * hidden + h] + reset * recurrent[2 * hidden + h]);
            next[h] = (1.0F - update) * candidate + update * state[h];
        }
        state = next;
        for (size_t h = 0; h < hidden; ++h) {
            out[t * stride + h] = state[h];
        }
    }
}

}  // namespace

G2p::G2p(const json::Value & settings, Tensors weights) : weights_(std::move(weights)) {
    const Str letters = decode(settings.at("letters").string);
    for (size_t index = 0; index < letters.size(); ++index) {
        letters_[letters[index]] = static_cast<int>(index) + 1;
    }
    for (const json::Value & symbol : settings.at("symbols").items) {
        symbols_.push_back(symbol.string);
    }
    slots_ = static_cast<long>(settings.at("slots").integer);
    layers_ = static_cast<long>(settings.at("layers").integer);
    hidden_ = static_cast<long>(settings.at("hidden").integer);
    if (settings.at("convs").integer != 0) {
        throw ValueError("this build reads unknown-word models without convolution blocks only");
    }
    if (slots_ < 1 || layers_ < 1 || hidden_ < 1) {
        throw ValueError("g2p.json: bad settings");
    }
    // Every tensor is checked once here, so the forward pass can index freely.
    const auto hidden = static_cast<size_t>(hidden_);
    const size_t width = tensor("embed__weight", {letters.size() + 1, 0}).shape[1];
    tensor("slot", {static_cast<size_t>(slots_), width});
    size_t input = width;
    for (long layer = 0; layer < layers_; ++layer) {
        for (const char * suffix : {"", "_reverse"}) {
            const std::string name = "l" + std::to_string(layer) + suffix;
            tensor("gru__weight_ih_" + name, {3 * hidden, input});
            tensor("gru__weight_hh_" + name, {3 * hidden, hidden});
            tensor("gru__bias_ih_" + name, {3 * hidden});
            tensor("gru__bias_hh_" + name, {3 * hidden});
        }
        input = 2 * hidden;
    }
    tensor("out__weight", {symbols_.size() + 1, 2 * hidden});
    tensor("out__bias", {symbols_.size() + 1});
}

// The tensor called `name`; a 0 in `shape` stands for any size.
const Tensor & G2p::tensor(const std::string & name, std::vector<size_t> shape) const {
    const auto found = weights_.find(name);
    if (found == weights_.end()) {
        throw KeyError(name);
    }
    const Tensor & value = found->second;
    bool fits = value.shape.size() == shape.size();
    size_t count = 1;
    for (size_t k = 0; fits && k < shape.size(); ++k) {
        fits = shape[k] == 0 || shape[k] == value.shape[k];
        count *= value.shape[k];
    }
    if (!fits || value.values.size() != count) {
        throw ValueError("unknown-word model: tensor " + name + " has an unexpected shape");
    }
    return value;
}

Str G2p::spellable(StrView word) const {
    Str kept;
    for (const char32_t c : uni::nfkd(py::lower(word))) {
        if (letters_.count(c) != 0) {
            kept.push_back(c);
        }
    }
    return py::strip(kept, U"'");
}

std::vector<std::string> G2p::predict(StrView word_in) const {
    const Str word = spellable(word_in);
    if (word.empty()) {
        return {};
    }
    const auto hidden = static_cast<size_t>(hidden_);
    const auto slots = static_cast<size_t>(slots_);
    const Tensor & embed = weights_.at("embed__weight");
    const Tensor & slot = weights_.at("slot");
    size_t width = embed.shape[1];
    const size_t frames = word.size() * slots;
    std::vector<float> x(frames * width);
    for (size_t i = 0; i < word.size(); ++i) {
        const auto id = static_cast<size_t>(letters_.at(word[i]));
        for (size_t s = 0; s < slots; ++s) {
            for (size_t k = 0; k < width; ++k) {
                x[(i * slots + s) * width + k] = embed.values[id * width + k] + slot.values[s * width + k];
            }
        }
    }
    for (long layer = 0; layer < layers_; ++layer) {
        std::vector<float> both(frames * 2 * hidden);
        for (int direction = 0; direction < 2; ++direction) {
            const std::string name = "l" + std::to_string(layer) + (direction == 1 ? "_reverse" : "");
            gru(x, frames, width, weights_.at("gru__weight_ih_" + name), weights_.at("gru__weight_hh_" + name),
                weights_.at("gru__bias_ih_" + name), weights_.at("gru__bias_hh_" + name), direction == 1, hidden,
                both.data() + (direction == 1 ? hidden : 0), 2 * hidden);
        }
        x = std::move(both);
        width = 2 * hidden;
    }
    const Tensor & out_weight = weights_.at("out__weight");
    const Tensor & out_bias = weights_.at("out__bias");
    const size_t labels = out_bias.values.size();
    std::vector<std::string> out;
    size_t previous = 0;
    std::vector<float> logits(labels);
    for (size_t t = 0; t < frames; ++t) {
        float highest = 0;
        for (size_t label = 0; label < labels; ++label) {
            logits[label] = dot(&x[t * width], &out_weight.values[label * width], width) + out_bias.values[label];
            if (label == 0 || logits[label] > highest) {
                highest = logits[label];
            }
        }
        // The log-probabilities, as the Python code forms them before it takes the largest.
        float total = 0;
        for (size_t label = 0; label < labels; ++label) {
            logits[label] -= highest;
            total += std::exp(logits[label]);
        }
        const float shift = std::log(total);
        size_t best = 0;
        for (size_t label = 0; label < labels; ++label) {
            logits[label] -= shift;
            if (logits[label] > logits[best]) {
                best = label;
            }
        }
        if (best != previous && best != 0) {
            out.push_back(symbols_.at(best - 1));
        }
        previous = best;
    }
    return out;
}

}  // namespace dizione
