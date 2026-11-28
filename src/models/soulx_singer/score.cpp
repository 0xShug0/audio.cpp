#include "engine/models/soulx_singer/score.h"
#include "engine/framework/modules/lookup_modules.h"
#include "engine/framework/modules/primitive_modules.h"
#include "engine/framework/modules/weight_binding.h"
#include <algorithm>
#include <cmath>
#include <sstream>
#include <stdexcept>

namespace engine::models::soulx_singer {
SingingScore parse_score(const io::json::Value &meta, const std::map<std::string, int32_t> &phones) {
    std::vector<std::string> phonemes;
    std::vector<double> duration;
    std::vector<int32_t> pitches, types;
    {
        std::istringstream stream(meta.require("phoneme").as_string());
        std::string ph;
        while (stream >> ph)
            phonemes.push_back(ph == "<AP>" ? "<SP>" : ph);
    }
    {
        std::istringstream stream(meta.require("duration").as_string());
        double d;
        while (stream >> d)
            duration.push_back(d);
    }
    {
        std::istringstream stream(meta.require("note_pitch").as_string());
        int32_t n;
        while (stream >> n)
            pitches.push_back(n);
    }
    {
        std::istringstream stream(meta.require("note_type").as_string());
        int32_t n;
        while (stream >> n)
            types.push_back(n);
    }
    const auto count = phonemes.size();
    if (!count || duration.size() != count || pitches.size() != count || types.size() != count)
        throw std::invalid_argument("Singing score fields must have matching nonempty note counts");
    std::vector<std::string> ph;
    std::vector<double> dur;
    std::vector<int32_t> pitch, type;
    for (size_t i = 0; i < count; ++i) {
        if (!std::isfinite(duration[i]) || duration[i] < 0 || pitches[i] < 0 || pitches[i] > 255 ||
            types[i] < 0 || types[i] > 255)
            throw std::invalid_argument("Invalid singing note duration, pitch or type");
        const auto ty = phonemes[i] == "<SP>" ? 1 : types[i];
        if (!ph.empty() && ph.back() == "<SP>" && phonemes[i] == "<SP>" && pitch.back() == pitches[i] &&
            type.back() == ty)
            dur.back() += duration[i];
        else {
            ph.push_back(phonemes[i]);
            dur.push_back(duration[i]);
            pitch.push_back(pitches[i]);
            type.push_back(ty);
        }
    }
    // Preserve Python's compensated duration sum before truncating to frames.
    double total = 0, correction = 0;
    for (auto d : dur) {
        const auto next = total + d;
        correction += std::abs(total) >= std::abs(d) ? (total - next) + d : (d - next) + total;
        total = next;
    }
    const auto frames = static_cast<size_t>((total + correction) * 24000 / 480);
    if (!frames)
        throw std::invalid_argument("Singing score has no frames");
    SingingScore out;
    out.frame_notes.assign(frames, 0);
    out.phonemes.push_back(phones.at("<PAD>"));
    out.pitches.push_back(0);
    out.types.push_back(1);
    std::vector<size_t> starts, lengths;
    double elapsed = 0;
    for (size_t i = 0; i < ph.size(); ++i) {
        starts.push_back(std::min(static_cast<size_t>(std::nearbyint(elapsed * 24000 / 480)), frames - 1));
        out.phonemes.push_back(phones.at("<BOW>"));
        out.pitches.push_back(pitch[i]);
        out.types.push_back(type[i]);
        std::vector<std::string> items;
        if (ph[i].rfind("en_", 0) == 0) {
            std::istringstream stream(ph[i].substr(3));
            std::string part;
            while (std::getline(stream, part, '-'))
                items.push_back("en_" + part);
            items.push_back("<SEP>");
        } else
            items.push_back(ph[i]);
        lengths.push_back(items.size());
        for (const auto &item : items) {
            out.phonemes.push_back(phones.at(item));
            out.pitches.push_back(pitch[i]);
            out.types.push_back(type[i]);
        }
        out.phonemes.push_back(phones.at("<EOW>"));
        out.pitches.push_back(pitch[i]);
        out.types.push_back(type[i]);
        elapsed += dur[i];
    }
    int32_t note = 1;
    for (size_t n = 0; n < starts.size(); ++n) {
        auto i = starts[n];
        const auto j = lengths[n], next = n + 1 < starts.size() ? starts[n + 1] : frames;
        if (i >= frames || i + j > frames)
            break;
        while (i < frames && out.frame_notes[i] > 0)
            ++i;
        if (i >= frames || !next)
            throw std::invalid_argument("Singing note alignment exceeds frame capacity");
        out.frame_notes[i] = note;
        for (size_t k = i + 1; k + j < next; k += j)
            for (size_t q = 0; q < j; ++q)
                out.frame_notes[k + q] = note + static_cast<int32_t>(q) + 1;
        out.frame_notes[next - 1] = note + static_cast<int32_t>(j) + 1;
        note += static_cast<int32_t>(j) + 2;
    }
    if (const auto *value = meta.find("f0")) {
        std::istringstream stream(value->as_string());
        float x;
        while (stream >> x)
            out.f0.push_back(x);
        if (!out.f0.empty()) {
            const auto size = std::min(out.frame_notes.size(), out.f0.size());
            out.frame_notes.resize(size);
            out.f0.resize(size);
        }
    }
    return out;
}

ScoreConditioner::ScoreConditioner(const assets::TensorSource &source, core::ExecutionContext &e)
    : execution_(e), weights_(e.backend(), e.backend_type(), "soulx_singer.score.weights", 2 * 1024 * 1024) {
    const auto load = [&](const std::string &name) {
        return weights_.load_f32_tensor(source, name, source.require_metadata(name).shape);
    };
    text_ = load("note_text_encoder.weight");
    pitch_ = load("note_pitch_encoder.weight");
    type_ = load("note_type_encoder.weight");
    for (int layer = 0; source.has_tensor("preflow." + std::to_string(layer) + ".dwconv.weight"); ++layer) {
        const auto p = "preflow." + std::to_string(layer);
        modules::ConvNeXt1dWeights w;
        const auto kernel = source.require_metadata(p + ".dwconv.weight").shape.at(2);
        w.depthwise = modules::binding::depthwise_conv1d_from_source(
            weights_, source, p + ".dwconv", assets::TensorStorageType::F32, 512, kernel, true);
        w.norm = modules::binding::norm_from_source(weights_, source, p + ".norm", 512);
        w.expansion = modules::binding::linear_from_source(weights_, source, p + ".pwconv1",
                                                           assets::TensorStorageType::F32, 1024, 512, true);
        w.projection = modules::binding::linear_from_source(weights_, source, p + ".pwconv2",
                                                            assets::TensorStorageType::F32, 512, 1024, true);
        w.response_gamma = load(p + ".grn.gamma");
        w.response_beta = load(p + ".grn.beta");
        blocks_.push_back(std::move(w));
    }
    if (blocks_.empty())
        throw std::invalid_argument("Singing score encoder has no preflow blocks");
    weights_.upload();
}
std::vector<float> ScoreConditioner::encode(const SingingScore &score, int shift, bool melody) {
    const auto notes = score.phonemes.size(), frames = score.frame_notes.size();
    const auto key = std::make_pair(notes, frames);
    auto *slot = graphs_.find(key);
    if (!slot) {
        graphs_.put(key, Graph{});
        slot = graphs_.find(key);
    }
    auto &g = *slot;
    if (!g.owner) {
        g.owner = std::make_unique<core::PreparedGraph>(
            execution_, core::PreparedGraphConfig{"soulx_singer.score", 4 * 1024 * 1024, 2048});
        auto &ctx = g.owner->build_context();
        const auto input = [&](size_t size) {
            auto x = core::make_tensor(ctx, GGML_TYPE_I32,
                                       core::TensorShape::from_dims({static_cast<int64_t>(size)}));
            ggml_set_input(x.tensor);
            return x;
        };
        g.text = input(notes);
        g.pitch = input(notes);
        g.type = input(notes);
        g.frames = input(frames);
        auto x = modules::EmbeddingModule({3000, 512}).build(ctx, g.text, text_);
        x = modules::AddModule().build(ctx, x,
                                       modules::EmbeddingModule({256, 512}).build(ctx, g.pitch, pitch_));
        x = modules::AddModule().build(ctx, x,
                                       modules::EmbeddingModule({256, 512}).build(ctx, g.type, type_));
        x = core::reshape_tensor(ctx, x, core::TensorShape::from_dims({1, static_cast<int64_t>(notes), 512}));
        for (const auto &block : blocks_)
            x = modules::build_convnext1d(ctx, x, block);
        x = core::reshape_tensor(ctx, core::ensure_backend_addressable_layout(ctx, x),
                                 core::TensorShape::from_dims({static_cast<int64_t>(notes), 512}));
        g.output = modules::EmbeddingModule({static_cast<int64_t>(notes), 512}).build(ctx, g.frames, x);
        ggml_set_output(g.output.tensor);
        ggml_build_forward_expand(g.owner->graph(), g.output.tensor);
        g.owner->allocate();
    }
    auto pitches = score.pitches;
    for (auto &x : pitches)
        x = melody ? 0 : (x > 0 ? std::clamp(x + shift, 0, 255) : x);
    for (const auto &item :
         {std::pair<core::TensorValue, const std::vector<int32_t> *>{g.text, &score.phonemes},
          {g.pitch, &pitches},
          {g.type, &score.types},
          {g.frames, &score.frame_notes}})
        ggml_backend_tensor_set(item.first.tensor, item.second->data(), 0,
                                item.second->size() * sizeof(int32_t));
    g.owner->compute();
    return core::read_tensor_f32(g.output.tensor);
}
} // namespace engine::models::soulx_singer
