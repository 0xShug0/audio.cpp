// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 AND BSD-3-Clause
/*
Adapted from SciPy rectangular_lsap, via the existing Nemotron alignment port.
Copyright (c) 2001-2002 Enthought, Inc.
Copyright (c) 2003, SciPy Developers.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

1. Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above
   copyright notice, this list of conditions and the following
   disclaimer in the documentation and/or other materials provided
   with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived
   from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/
#include "grouping_decode.h"
#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace engine::models::dettato::detail {
namespace {
std::vector<std::pair<int, int>>
linear_sum_assignment(const std::vector<double>& cost_in, int nr, int nc) {
    // Port of SciPy's scipy/optimize/rectangular_lsap/rectangular_lsap.cpp (Crouse's shortest
    // augmenting path; BSD 3-Clause, notice above).
    std::vector<std::pair<int, int>> out;
    if (nr < 0 || nc < 0 || cost_in.size() != static_cast<size_t>(nr) * nc)
        throw std::invalid_argument("linear_sum_assignment: cost size");
    if (nr == 0 || nc == 0) return out;
    const bool transpose = nc < nr;
    std::vector<double> cost(cost_in);
    if (transpose) {
        for (int i = 0; i < nr; ++i)
            for (int j = 0; j < nc; ++j)
                cost[static_cast<size_t>(j) * nr + i] = cost_in[static_cast<size_t>(i) * nc + j];
        std::swap(nr, nc);
    }
    for (double v : cost)
        if (std::isnan(v) || v == -std::numeric_limits<double>::infinity())
            throw std::invalid_argument("matrix contains invalid numeric entries");
    const double inf = std::numeric_limits<double>::infinity();
    std::vector<double> u(nr, 0.0), v(nc, 0.0), shortest(nc);
    std::vector<int> path(nc, -1), col4row(nr, -1), row4col(nc, -1), remaining(nc);
    std::vector<char> SR(nr), SC(nc);
    for (int cur = 0; cur < nr; ++cur) {
        double min_val = 0.0;
        int i = cur, sink = -1, num_remaining = nc;
        for (int it = 0; it < nc; ++it) remaining[it] = nc - it - 1;
        std::fill(SR.begin(), SR.end(), 0);
        std::fill(SC.begin(), SC.end(), 0);
        std::fill(shortest.begin(), shortest.end(), inf);
        while (sink == -1) {
            int index = -1;
            double lowest = inf;
            SR[i] = 1;
            for (int it = 0; it < num_remaining; ++it) {
                const int j = remaining[it];
                const double r = min_val + cost[static_cast<size_t>(i) * nc + j] - u[i] - v[j];
                if (r < shortest[j]) {
                    path[j] = i;
                    shortest[j] = r;
                }
                if (shortest[j] < lowest || (shortest[j] == lowest && row4col[j] == -1)) {
                    lowest = shortest[j];
                    index = it;
                }
            }
            min_val = lowest;
            if (min_val == inf)
                throw std::invalid_argument("cost matrix is infeasible");
            const int j = remaining[index];
            if (row4col[j] == -1)
                sink = j;
            else
                i = row4col[j];
            SC[j] = 1;
            remaining[index] = remaining[--num_remaining];
        }
        u[cur] += min_val;
        for (int r = 0; r < nr; ++r)
            if (SR[r] && r != cur)
                u[r] += min_val - shortest[col4row[r]];
        for (int c = 0; c < nc; ++c)
            if (SC[c])
                v[c] -= min_val - shortest[c];
        for (int j = sink;;) {
            const int r = path[j];
            row4col[j] = r;
            std::swap(col4row[r], j);
            if (r == cur)
                break;
        }
    }
    if (transpose) {
        // argsort of col4row: original rows ascending.
        std::vector<int> order(nr);
        for (int k = 0; k < nr; ++k) order[k] = k;
        std::sort(
            order.begin(), order.end(), [&](int a, int b) { return col4row[a] < col4row[b]; });
        for (int k : order) out.emplace_back(col4row[k], k);
    } else {
        for (int r = 0; r < nr; ++r) out.emplace_back(r, col4row[r]);
    }
    return out;
}

constexpr int embedding_channels = 32;
constexpr int instrument_count = 73;

// Nearest-neighbour-chain average linkage, followed by SciPy's stable merge
// ordering and distance-cut traversal. Unlike scanning every remaining pair
// after every merge, this requires quadratic work for a dense note window.
std::vector<int> cluster_window(const GroupingWindow & window, double threshold, float lambda) {
    const int n = static_cast<int>(window.notes.size());
    if (n == 1) return {0};
    std::vector<float> probabilities = window.probabilities;
    for (int i = 0; i < n; ++i) {
        float norm = 0;
        for (int c = 0; c < instrument_count; ++c) {
            const auto p = probabilities[i * instrument_count + c];
            norm += p * p;
        }
        norm = std::max(std::sqrt(norm), 1e-9f);
        for (int c = 0; c < instrument_count; ++c) probabilities[i * instrument_count + c] /= norm;
    }
    std::vector<double> distances(static_cast<size_t>(n) * n);
    for (int i = 0; i < n; ++i) for (int j = i + 1; j < n; ++j) {
        float e = 0, p = 0;
        for (int c = 0; c < embedding_channels; ++c)
            e += window.embeddings[i * embedding_channels + c] * window.embeddings[j * embedding_channels + c];
        for (int c = 0; c < instrument_count; ++c)
            p += probabilities[i * instrument_count + c] * probabilities[j * instrument_count + c];
        const float d = std::max(0.f, (1.f - e) + (lambda > 0 ? lambda * (1.f - p) : 0.f));
        distances[i * n + j] = distances[j * n + i] = d;
    }
    struct Merge { int left, right, size; double distance; };
    std::vector<Merge> merges;
    std::vector<int> size(n, 1), chain;
    for (int k = 0; k < n - 1; ++k) {
        if (chain.empty()) {
            int first = 0;
            while (!size[first]) ++first;
            chain.push_back(first);
        }
        int x, y;
        double minimum;
        while (true) {
            x = chain.back();
            y = chain.size() > 1 ? chain[chain.size() - 2] : -1;
            minimum = y >= 0 ? distances[x * n + y] : std::numeric_limits<double>::infinity();
            for (int i = 0; i < n; ++i) if (i != x && size[i] && distances[x * n + i] < minimum) {
                minimum = distances[x * n + i];
                y = i;
            }
            if (chain.size() > 1 && y == chain[chain.size() - 2]) break;
            chain.push_back(y);
        }
        chain.resize(chain.size() - 2);
        if (x > y) std::swap(x, y);
        const int nx = size[x], ny = size[y];
        merges.push_back({x, y, nx + ny, minimum});
        size[x] = 0; size[y] = nx + ny;
        for (int i = 0; i < n; ++i) if (i != y && size[i])
            distances[i * n + y] = distances[y * n + i] =
                (nx * distances[i * n + x] + ny * distances[i * n + y]) / (nx + ny);
    }
    std::stable_sort(merges.begin(), merges.end(), [](const auto & a, const auto & b) { return a.distance < b.distance; });
    std::vector<int> parent(2 * n - 1);
    std::iota(parent.begin(), parent.end(), 0);
    const auto root = [&](int x) {
        while (parent[x] != x) { parent[x] = parent[parent[x]]; x = parent[x]; }
        return x;
    };
    std::vector<double> maximum(n - 1);
    for (int k = 0; k < n - 1; ++k) {
        auto & m = merges[k];
        m.left = root(m.left); m.right = root(m.right);
        if (m.left > m.right) std::swap(m.left, m.right);
        parent[m.left] = parent[m.right] = n + k;
        maximum[k] = m.distance;
        if (m.left >= n) maximum[k] = std::max(maximum[k], maximum[m.left - n]);
        if (m.right >= n) maximum[k] = std::max(maximum[k], maximum[m.right - n]);
    }
    std::vector<int> labels(n, -1);
    int next = 0;
    std::function<void(int, int)> visit = [&](int node, int active) {
        const auto & m = merges[node - n];
        if (active < 0 && maximum[node - n] <= threshold) active = next++;
        if (m.left >= n) visit(m.left, active);
        if (m.right >= n) visit(m.right, active);
        if (m.left < n) labels[m.left] = active < 0 ? next++ : active;
        if (m.right < n) labels[m.right] = active < 0 ? next++ : active;
    };
    visit(2 * n - 2, -1);
    return labels;
}
} // namespace

std::vector<int> chain_grouping_windows(const std::vector<GroupingWindow> & windows,
    const std::vector<double> & onsets, double window_seconds, double threshold, float lambda) {
    if (!(window_seconds > 0)) throw std::invalid_argument("Dettato grouping window duration must be positive");
    std::vector<int> track(onsets.size(), -1), owner(onsets.size(), -1);
    std::vector<double> best(onsets.size(), -1);
    std::vector<size_t> order(windows.size());
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) { return windows[a].start < windows[b].start; });
    int global = 0;
    for (const auto wi : order) {
        const auto & w = windows[wi];
        if (w.embeddings.size() != w.notes.size() * embedding_channels ||
            w.probabilities.size() != w.notes.size() * instrument_count)
            throw std::invalid_argument("Dettato grouping window feature shape mismatch");
        if (w.notes.empty()) continue;
        for (const int id : w.notes) if (id < 0 || static_cast<size_t>(id) >= onsets.size())
            throw std::invalid_argument("Dettato grouping window note index out of range");
        const auto labels = cluster_window(w, threshold, lambda);
        const int count = *std::max_element(labels.begin(), labels.end()) + 1;
        std::vector<int> mapping(count, -1), previous;
        for (const int id : w.notes) if (owner[id] >= 0) previous.push_back(owner[id]);
        std::sort(previous.begin(), previous.end());
        previous.erase(std::unique(previous.begin(), previous.end()), previous.end());
        if (!previous.empty()) {
            std::vector<double> costs(static_cast<size_t>(count) * previous.size(), 0);
            for (size_t i = 0; i < w.notes.size(); ++i) if (owner[w.notes[i]] >= 0) {
                const auto col = std::lower_bound(previous.begin(), previous.end(), owner[w.notes[i]]) - previous.begin();
                costs[labels[i] * previous.size() + col] -= 1;
            }
            for (const auto & pair : linear_sum_assignment(costs, count, static_cast<int>(previous.size())))
                if (costs[pair.first * previous.size() + pair.second] < 0) mapping[pair.first] = previous[pair.second];
        }
        for (auto & id : mapping) if (id < 0) id = global++;
        for (size_t i = 0; i < w.notes.size(); ++i) {
            const int id = w.notes[i], assigned = mapping[labels[i]];
            owner[id] = assigned;
            const double weight = std::clamp(1. - std::abs((onsets[id] - w.start) / window_seconds - .5) * 2., .05, 1.);
            if (weight > best[id]) { track[id] = assigned; best[id] = weight; }
        }
    }
    return track;
}

std::vector<int> merge_grouping_tracklets(const std::vector<int> & tracklets,
    const std::vector<float> & embeddings, const std::vector<float> & probabilities,
    double threshold, double lambda) {
    if (embeddings.size() != tracklets.size() * embedding_channels || probabilities.size() != tracklets.size() * instrument_count)
        throw std::invalid_argument("Dettato song grouping feature shape mismatch");
    std::vector<int> ids;
    for (const auto id : tracklets) if (id >= 0) ids.push_back(id);
    std::sort(ids.begin(), ids.end()); ids.erase(std::unique(ids.begin(), ids.end()), ids.end());
    const int count = static_cast<int>(ids.size());
    if (count < 2) return tracklets;
    std::vector<double> sizes(count, 0), e(static_cast<size_t>(count) * embedding_channels), p(static_cast<size_t>(count) * instrument_count);
    std::vector<int> inverse(tracklets.size(), -1), labels(count);
    std::iota(labels.begin(), labels.end(), 0);
    for (size_t i = 0; i < tracklets.size(); ++i) if (tracklets[i] >= 0) {
        const int t = static_cast<int>(std::lower_bound(ids.begin(), ids.end(), tracklets[i]) - ids.begin());
        inverse[i] = t; sizes[t] += 1;
        double norm = 0;
        for (int c = 0; c < instrument_count; ++c) { const double x = probabilities[i * instrument_count + c]; norm += x * x; }
        norm = std::max(std::sqrt(norm), 1e-9);
        for (int c = 0; c < embedding_channels; ++c) e[t * embedding_channels + c] += embeddings[i * embedding_channels + c];
        for (int c = 0; c < instrument_count; ++c) p[t * instrument_count + c] += probabilities[i * instrument_count + c] / norm;
    }
    for (int t = 0; t < count; ++t) {
        for (int c = 0; c < embedding_channels; ++c) e[t * embedding_channels + c] /= sizes[t];
        for (int c = 0; c < instrument_count; ++c) p[t * instrument_count + c] /= sizes[t];
    }
    const auto distance = [&](int i, int j) {
        double ed = 0, pd = 0;
        for (int c = 0; c < embedding_channels; ++c) ed += e[i * embedding_channels + c] * e[j * embedding_channels + c];
        for (int c = 0; c < instrument_count; ++c) pd += p[i * instrument_count + c] * p[j * instrument_count + c];
        return 1. - ed + (lambda > 0 ? lambda * (1. - pd) : 0.);
    };
    const auto infinity = std::numeric_limits<double>::infinity();
    std::vector<double> distances(static_cast<size_t>(count) * count, infinity);
    std::vector<bool> alive(count, true), eligible(count, true);
    if (count > 600) {
        std::vector<int> largest(count); std::iota(largest.begin(), largest.end(), 0);
        std::stable_sort(largest.begin(), largest.end(), [&](int a, int b) { return sizes[a] > sizes[b]; });
        for (int i = 600; i < count; ++i) eligible[largest[i]] = false;
    }
    for (int i = 0; i < count; ++i) for (int j = i + 1; j < count; ++j)
        if (eligible[i] && eligible[j]) distances[i * count + j] = distances[j * count + i] = distance(i, j);
    while (true) {
        const auto found = std::min_element(distances.begin(), distances.end());
        if (!std::isfinite(*found) || *found > threshold) break;
        const int i = static_cast<int>((found - distances.begin()) / count), j = static_cast<int>((found - distances.begin()) % count);
        const double si = sizes[i], sj = sizes[j], total = si + sj;
        for (int c = 0; c < embedding_channels; ++c) e[i * embedding_channels + c] = (si * e[i * embedding_channels + c] + sj * e[j * embedding_channels + c]) / total;
        for (int c = 0; c < instrument_count; ++c) p[i * instrument_count + c] = (si * p[i * instrument_count + c] + sj * p[j * instrument_count + c]) / total;
        sizes[i] = total; alive[j] = false;
        for (auto & label : labels) if (label == j) label = i;
        for (int k = 0; k < count; ++k) {
            distances[j * count + k] = distances[k * count + j] = infinity;
            distances[i * count + k] = distances[k * count + i] = alive[k] && k != i ? distance(i, k) : infinity;
        }
    }
    auto result = inverse;
    for (auto & label : result) if (label >= 0) label = labels[label];
    return result;
}

GroupingDecoder::GroupingDecoder(const assets::TensorSource & source, const io::json::Value & config) {
    const auto & decode = config.require("decode");
    threshold_ = decode.require("tau").as_number(); merge_threshold_ = decode.require("tau2").as_number();
    lambda_ = decode.require("lam").as_number(); alpha_ = decode.require("alpha").as_number();
    absorb_ = decode.require("absorb").as_number(); absorb_fraction_ = decode.require("absorb_frac").as_number();
    minimum_notes_ = decode.require("min_notes").as_i64(); refine_ = decode.require("refine").as_i64();
    expected_credit_ = decode.require("ec").as_bool(); merge_ = decode.require("merge").as_bool();
    for (const auto & name : config.require("instruments").as_array()) names_.push_back(name.as_string());
    for (const auto & program : config.require("programs").as_array()) programs_.push_back(static_cast<int>(program.as_i64()));
    if (names_.size() != instrument_count || programs_.size() != instrument_count)
        throw std::invalid_argument("Dettato grouping instrument table must contain 73 entries");
    credit_ = source.require_f32("instrument_credit", {instrument_count, instrument_count});
    for (int i = 0; i < 3; ++i) {
        const auto prefix = "heads.th." + std::to_string(i * 2);
        const auto weight = source.require_f32(prefix + ".weight"), bias = source.require_f32(prefix + ".bias");
        DenseLayer layer;
        layer.input = i == 0 ? 182 : 176; layer.output = i == 2 ? instrument_count : 176;
        if (weight.size() != static_cast<size_t>(layer.input * layer.output) || bias.size() != static_cast<size_t>(layer.output))
            throw std::invalid_argument("Dettato track classifier shape mismatch");
        layer.weight.assign(weight.begin(), weight.end()); layer.bias.assign(bias.begin(), bias.end());
        head_.push_back(std::move(layer));
    }
}

std::vector<Track> GroupingDecoder::decode(const GroupingPrediction & prediction, const std::vector<Note> & notes,
    double time_ratio) const {
    const size_t n = notes.size();
    if (prediction.embeddings.size() != n * embedding_channels || prediction.probabilities.size() != n * instrument_count ||
        prediction.hidden.size() != n * 176) throw std::invalid_argument("Dettato grouping prediction shape mismatch");
    std::vector<double> onsets; onsets.reserve(n);
    for (const auto & note : notes) onsets.push_back(note.onset);
    auto track = chain_grouping_windows(prediction.windows, onsets, prediction.window_seconds, threshold_, static_cast<float>(lambda_));
    track = merge_grouping_tracklets(track, prediction.embeddings, prediction.probabilities, merge_threshold_, lambda_);
    std::vector<size_t> selected;
    std::vector<int> ids;
    for (size_t i = 0; i < n; ++i) if (track[i] >= 0) { selected.push_back(i); ids.push_back(track[i]); }
    if (selected.empty()) return {};
    std::sort(ids.begin(), ids.end()); ids.erase(std::unique(ids.begin(), ids.end()), ids.end());
    const int count = static_cast<int>(ids.size());
    std::vector<int> labels(selected.size());
    std::vector<double> e(selected.size() * embedding_channels), p(selected.size() * instrument_count);
    for (size_t i = 0; i < selected.size(); ++i) {
        labels[i] = static_cast<int>(std::lower_bound(ids.begin(), ids.end(), track[selected[i]]) - ids.begin());
        for (int c = 0; c < embedding_channels; ++c) e[i * embedding_channels + c] = prediction.embeddings[selected[i] * embedding_channels + c];
        for (int c = 0; c < instrument_count; ++c) p[i * instrument_count + c] = prediction.probabilities[selected[i] * instrument_count + c];
    }
    std::vector<int> sizes(count);
    std::vector<double> centres(static_cast<size_t>(count) * embedding_channels), mass(static_cast<size_t>(count) * instrument_count);
    const auto accumulate = [&] {
        std::fill(sizes.begin(), sizes.end(), 0); std::fill(centres.begin(), centres.end(), 0); std::fill(mass.begin(), mass.end(), 0);
        for (size_t i = 0; i < selected.size(); ++i) {
            const int t = labels[i]; ++sizes[t];
            for (int c = 0; c < embedding_channels; ++c) centres[t * embedding_channels + c] += e[i * embedding_channels + c];
            for (int c = 0; c < instrument_count; ++c) mass[t * instrument_count + c] += p[i * instrument_count + c];
        }
    };
    const auto normalize = [](std::vector<double> & values, int width) {
        for (size_t row = 0; row < values.size(); row += width) {
            double norm = 0;
            for (int c = 0; c < width; ++c) norm += values[row + c] * values[row + c];
            norm = std::max(std::sqrt(norm), 1e-9);
            for (int c = 0; c < width; ++c) values[row + c] /= norm;
        }
    };
    auto pn = p; normalize(pn, instrument_count);
    accumulate();
    for (int pass = 0; pass < refine_; ++pass) {
        std::vector<int> big;
        for (int t = 0; t < count; ++t) if (sizes[t] >= minimum_notes_) big.push_back(t);
        if (big.size() < 2) break;
        auto ce = centres, cp = mass; normalize(ce, embedding_channels); normalize(cp, instrument_count);
        for (size_t i = 0; i < selected.size(); ++i) {
            double best = -std::numeric_limits<double>::infinity(); int chosen = -1;
            for (const auto t : big) {
                double score = 0, probability = 0;
                for (int c = 0; c < embedding_channels; ++c) score += e[i * embedding_channels + c] * ce[t * embedding_channels + c];
                for (int c = 0; c < instrument_count; ++c) probability += pn[i * instrument_count + c] * cp[t * instrument_count + c];
                score += lambda_ > 0 ? lambda_ * probability : 0;
                if (score > best) { best = score; chosen = t; }
            }
            labels[i] = chosen;
        }
        accumulate();
    }
    std::vector<int> big;
    for (int t = 0; t < count; ++t) if (sizes[t] >= minimum_notes_) big.push_back(t);
    if (big.empty()) big.push_back(static_cast<int>(std::max_element(sizes.begin(), sizes.end()) - sizes.begin()));
    auto normalized_mass = mass; normalize(normalized_mass, instrument_count);
    for (int t = 0; t < count; ++t) if (sizes[t] && std::find(big.begin(), big.end(), t) == big.end()) {
        double best = -std::numeric_limits<double>::infinity(); int chosen = -1;
        for (const auto b : big) {
            double score = 0;
            for (int c = 0; c < instrument_count; ++c) score += normalized_mass[t * instrument_count + c] * normalized_mass[b * instrument_count + c];
            if (score > best) { best = score; chosen = b; }
        }
        for (auto & label : labels) if (label == t) label = chosen;
    }
    if (absorb_ > 0 && absorb_fraction_ > 0) {
        auto unit = e; normalize(unit, embedding_channels);
        while (true) {
            accumulate();
            std::vector<int> order;
            for (int t = 0; t < count; ++t) if (sizes[t]) order.push_back(t);
            if (order.size() < 2) break;
            std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return sizes[a] < sizes[b]; });
            // Absorption uses means of per-note unit embeddings, not normalized
            // mean directions; only the final note reassignment uses directions.
            std::fill(centres.begin(), centres.end(), 0);
            for (size_t i = 0; i < selected.size(); ++i) for (int c = 0; c < embedding_channels; ++c)
                centres[labels[i] * embedding_channels + c] += unit[i * embedding_channels + c];
            for (int t = 0; t < count; ++t) if (sizes[t]) for (int c = 0; c < embedding_channels; ++c)
                centres[t * embedding_channels + c] /= sizes[t];
            bool changed = false;
            for (const auto t : order) {
                const double share = static_cast<double>(sizes[t]) / selected.size();
                if (share >= absorb_fraction_) break;
                std::vector<int> larger;
                for (int b = 0; b < count; ++b) if (sizes[b] > sizes[t]) larger.push_back(b);
                bool dissolve = false;
                for (const auto b : larger) {
                    double dot = 0;
                    for (int c = 0; c < embedding_channels; ++c) dot += centres[t * embedding_channels + c] * centres[b * embedding_channels + c];
                    if (1. - dot < absorb_ * (1. - share / absorb_fraction_)) dissolve = true;
                }
                if (!dissolve) continue;
                auto directions = centres; normalize(directions, embedding_channels);
                for (size_t i = 0; i < selected.size(); ++i) if (labels[i] == t) {
                    double best = -std::numeric_limits<double>::infinity(); int chosen = -1;
                    for (const auto b : larger) {
                        double score = 0;
                        for (int c = 0; c < embedding_channels; ++c) score += unit[i * embedding_channels + c] * directions[b * embedding_channels + c];
                        if (score > best) { best = score; chosen = b; }
                    }
                    labels[i] = chosen;
                }
                changed = true; break;
            }
            if (!changed) break;
        }
    }
    std::vector<std::pair<int, std::vector<size_t>>> groups;
    for (int t = 0; t < count; ++t) {
        std::vector<size_t> members;
        for (size_t i = 0; i < selected.size(); ++i) if (labels[i] == t) members.push_back(selected[i]);
        if (members.empty()) continue;
        std::vector<double> logits(instrument_count, 0);
        for (const auto i : members) for (int c = 0; c < instrument_count; ++c) logits[c] += prediction.probabilities[i * instrument_count + c];
        const double total = std::accumulate(logits.begin(), logits.end(), 0.);
        for (auto & x : logits) x = std::log(std::max(x / total, 1e-9));
        if (alpha_ > 0) {
            std::vector<double> state(182, 0);
            double pitch_mean = 0, pitch_square = 0, low = std::numeric_limits<double>::infinity(), high = -low, polyphony = 0;
            for (const auto i : members) {
                for (int c = 0; c < 176; ++c) state[c] += prediction.hidden[i * 176 + c];
                const double pitch = notes[i].pitch / 64. - 1.;
                pitch_mean += pitch; pitch_square += pitch * pitch; low = std::min(low, pitch); high = std::max(high, pitch);
                if (members.size() <= 3000) {
                    int overlaps = 0;
                    for (const auto j : members) if (std::min(std::max(notes[i].offset, notes[i].onset + .02), std::max(notes[j].offset, notes[j].onset + .02)) > std::max(notes[i].onset, notes[j].onset) + .03) ++overlaps;
                    if (overlaps >= 2) polyphony += 1;
                }
            }
            for (int c = 0; c < 176; ++c) state[c] /= members.size();
            pitch_mean /= members.size();
            state[176] = pitch_mean; state[177] = std::sqrt(std::max(0., pitch_square / members.size() - pitch_mean * pitch_mean));
            state[178] = low; state[179] = high; state[180] = std::log1p(members.size()) / 5.; state[181] = polyphony / members.size();
            // Upstream deliberately evaluates this small track classifier in F64
            // on the host, outside its neural graph/autocast context.
            for (size_t l = 0; l < head_.size(); ++l) {
                const auto & layer = head_[l]; std::vector<double> next(layer.output);
                for (int row = 0; row < layer.output; ++row) {
                    double x = layer.bias[row];
                    for (int c = 0; c < layer.input; ++c) x += layer.weight[row * layer.input + c] * state[c];
                    next[row] = l + 1 < head_.size() ? .5 * x * (1. + std::erf(x / std::sqrt(2.))) : x;
                }
                state = std::move(next);
            }
            const double maximum = *std::max_element(state.begin(), state.end());
            double sum = 0; for (const auto x : state) sum += std::exp(x - maximum);
            const double log_sum = std::log(sum);
            for (int c = 0; c < instrument_count; ++c) logits[c] = alpha_ * (state[c] - maximum - log_sum) + (1. - alpha_) * logits[c];
        }
        int instrument;
        if (expected_credit_) {
            const double maximum = *std::max_element(logits.begin(), logits.end());
            double total_probability = 0;
            for (auto & x : logits) { x = std::exp(x - maximum); total_probability += x; }
            std::vector<double> score(instrument_count, 0);
            for (int row = 0; row < instrument_count; ++row) for (int c = 0; c < instrument_count; ++c)
                score[row] += credit_[row * instrument_count + c] * logits[c] / total_probability;
            instrument = static_cast<int>(std::max_element(score.begin(), score.end()) - score.begin());
        } else instrument = static_cast<int>(std::max_element(logits.begin(), logits.end()) - logits.begin());
        auto found = std::find_if(groups.begin(), groups.end(), [&](const auto & group) { return merge_ && group.first == instrument; });
        if (found == groups.end()) groups.emplace_back(instrument, std::move(members));
        else { found->second.insert(found->second.end(), members.begin(), members.end()); std::sort(found->second.begin(), found->second.end()); }
    }
    std::stable_sort(groups.begin(), groups.end(), [](const auto & a, const auto & b) { return a.second.size() > b.second.size(); });
    std::vector<Track> tracks;
    for (auto & group : groups) {
        auto & members = group.second;
        std::stable_sort(members.begin(), members.end(), [&](size_t a, size_t b) {
            return notes[a].pitch == notes[b].pitch ? notes[a].onset < notes[b].onset : notes[a].pitch < notes[b].pitch;
        });
        std::vector<size_t> kept;
        for (const auto i : members) if (kept.empty() || notes[i].pitch != notes[kept.back()].pitch || (notes[i].onset - notes[kept.back()].onset) * time_ratio > .03) kept.push_back(i);
        std::stable_sort(kept.begin(), kept.end(), [&](size_t a, size_t b) { return notes[a].onset < notes[b].onset; });
        Track out; out.name = names_[group.first]; out.program = programs_[group.first];
        for (const auto i : kept) {
            auto note = notes[i]; note.onset *= time_ratio; note.offset = std::max(note.offset * time_ratio, note.onset + .02);
            note.velocity = static_cast<int>(std::clamp(20. + 107. * note.confidence, 1., 127.)); out.notes.push_back(note);
        }
        tracks.push_back(std::move(out));
    }
    return tracks;
}
} // namespace engine::models::dettato::detail
