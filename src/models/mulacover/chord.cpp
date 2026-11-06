#include "engine/models/mulacover/chord.h"

#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/debug/trace.h"
#include "engine/framework/modules/activation_modules.h"
#include "engine/framework/modules/conv_modules.h"
#include "engine/framework/modules/linear_module.h"
#include "engine/framework/modules/norm_modules.h"
#include "engine/framework/modules/recurrent_modules.h"
#include "engine/framework/modules/structural_modules.h"
#include "engine/framework/modules/weight_binding.h"
#include "ggml-alloc.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>
#include <utility>

namespace engine::models::mulacover {
namespace {
using core::TensorShape;
using core::TensorValue;
constexpr std::array<int, 6> head_sizes{73, 13, 4, 4, 3, 3};
constexpr std::array<const char *, 10> conv_names{
    "conv1a", "conv1b", "conv1c", "conv2a", "conv2b", "conv2c",
    "conv3a", "conv3b", "conv4a", "conv4b"};
constexpr std::array<int, 10> channels{16, 16, 16, 32, 32, 32, 64, 64, 80, 80};

struct ChordTemplate {
    std::array<int, 6> factors;
    std::string label;
};

// complex_chord.py's categorical suffix representation, before root transposition.
std::array<int, 5> parse_suffix(std::string suffix) {
    static const std::map<std::string, std::array<int, 5>> types{
        {".", {0,0,0,0,0}}, {"maj", {1,0,0,0,0}}, {"min", {2,0,0,0,0}},
        {"sus4", {3,0,0,0,0}}, {"sus2", {4,0,0,0,0}}, {"dim", {5,0,0,0,0}},
        {"aug", {6,0,0,0,0}}, {"5", {7,0,0,0,0}}, {"1", {8,0,0,0,0}},
        {"", {8,0,0,0,0}}, {"maj6", {1,0,0,0,1}}, {"min6", {2,0,0,0,1}},
        {"7", {1,2,0,0,0}}, {"maj7", {1,1,0,0,0}}, {"min7", {2,2,0,0,0}},
        {"minmaj7", {2,1,0,0,0}}, {"dim7", {5,3,0,0,0}}, {"hdim7", {5,2,0,0,0}},
        {"9", {1,2,1,0,0}}, {"maj9", {1,1,1,0,0}}, {"min9", {2,2,1,0,0}},
        {"11", {1,2,1,1,0}}, {"min11", {2,2,1,1,0}},
        {"13", {1,2,1,1,1}}, {"maj13", {1,1,1,1,1}}, {"min13", {2,2,1,1,1}},
        {"N", {0,-2,-2,-2,-2}}, {"X", {-2,-2,-2,-2,-2}}};
    const auto bracket = suffix.find('(');
    const auto type = types.find(suffix.substr(0, bracket));
    if (type == types.end()) throw std::runtime_error("Unknown chord type: " + suffix);
    auto result = type->second;
    if (bracket == std::string::npos) return result;
    if (suffix.back() != ')') throw std::runtime_error("Invalid chord suffix: " + suffix);
    std::vector<std::string> additions;
    std::set<std::string> omissions;
    size_t begin = bracket + 1;
    while (begin < suffix.size() - 1) {
        auto end = suffix.find(',', begin);
        if (end == std::string::npos) end = suffix.size() - 1;
        auto note = suffix.substr(begin, end - begin);
        if (!note.empty() && note[0] == '*') omissions.insert(note.substr(1));
        else additions.push_back(note);
        begin = end + 1;
    }
    static const std::set<std::string> valid_omissions{"1","b3","3","b5","5","b7","7"};
    for (const auto & note : omissions) {
        if (!valid_omissions.count(note)) throw std::runtime_error("Invalid chord omission: " + note);
    }
    if (result[0] == 1 && omissions.erase("3")) result[0] = 7;
    else if (result[0] == 2 && omissions.erase("b3")) result[0] = 7;
    if (result[0] == 7 && omissions.erase("5")) result[0] = 8;
    for (const auto & note : {"1", "b3", "3", "b5", "5"}) {
        if (omissions.count(note)) result[0] = -2;
    }
    if (result[1] == 2 && omissions.erase("b7")) result[1] = 0;
    else if (result[1] == 1 && omissions.erase("7")) result[1] = 0;
    if (omissions.count("b7") || omissions.count("7")) result[1] = -2;
    static const std::map<std::string, std::pair<int,int>> decorations{
        {"7",{1,1}}, {"b7",{1,2}}, {"bb7",{1,3}}, {"2",{2,1}}, {"9",{2,1}},
        {"#9",{2,2}}, {"b9",{2,3}}, {"4",{3,1}}, {"11",{3,1}}, {"#11",{3,2}},
        {"13",{4,1}}, {"b13",{4,2}}, {"6",{4,1}}, {"b6",{4,2}}, {"bb6",{4,3}},
        {"#4",{0,-2}}, {"b5",{0,-2}}, {"5",{0,-2}}, {"#5",{0,-2}},
        {"b3",{0,-2}}, {"b2",{0,-2}}, {"3",{0,-2}}};
    for (const auto & note : additions) {
        if (note == "1") continue;
        if (note == "5" && result[0] == 8) { result[0] = 7; continue; }
        const auto it = decorations.find(note);
        if (it == decorations.end()) throw std::runtime_error("Unknown chord decoration: " + note);
        // Upstream assigns dec_type even after marking a conflicting decoration unknown.
        result[it->second.first] = it->second.second;
    }
    return result;
}

std::vector<ChordTemplate> load_vocabulary(const std::filesystem::path & path) {
    std::ifstream file(path);
    if (!file) throw std::runtime_error("Cannot open chord vocabulary: " + path.string());
    const std::array<std::string,12> roots{"C","C#","D","Eb","E","F","F#","G","Ab","A","Bb","B"};
    std::vector<ChordTemplate> result{{{0,-1,-1,-1,-1,-1}, "N"}};
    std::set<std::array<int,6>> seen;
    std::string name;
    while (std::getline(file, name)) {
        const auto first = name.find_first_not_of(" \t\r\n");
        if (first == std::string::npos) continue;
        name = name.substr(first, name.find_last_not_of(" \t\r\n") - first + 1);
        if (name.find(':') == std::string::npos) continue;
        if (name.substr(0,2) != "C:" || name.find(':',2) != std::string::npos)
            throw std::runtime_error("Unexpected chord template root: " + name);
        const auto suffix = name.substr(2);
        const auto slash = suffix.find('/');
        int bass = 0;
        if (slash != std::string::npos) {
            const auto degree = suffix.substr(slash + 1);
            const std::string scale = "1*2*34*5*6*78*9";
            if (degree.empty() || scale.find(degree.back()) == std::string::npos)
                throw std::runtime_error("Invalid chord bass: " + name);
            bass = static_cast<int>(scale.find(degree.back()));
            bass += static_cast<int>(std::count(degree.begin(), degree.end(), '#'));
            bass -= static_cast<int>(std::count(degree.begin(), degree.end(), 'b'));
            bass = (bass % 12 + 12) % 12;
        }
        const auto factors = parse_suffix(suffix.substr(0,slash));
        if (std::find(factors.begin(), factors.end(), -2) != factors.end()) continue;
        for (int shift = 0; shift < 12; ++shift) {
            std::array<int,6> chord{factors[0] > 0 ? (factors[0]-1)*12+1+shift : factors[0],
                (bass+shift)%12, factors[1], factors[2], factors[3], factors[4]};
            if (!seen.insert(chord).second) continue;
            bool valid = true;
            for (size_t i = 0; i < 6; ++i) valid = valid && chord[i] < head_sizes[i];
            if (valid) result.push_back({chord, roots[shift] + ":" + suffix});
        }
    }
    if (!file.eof()) throw std::runtime_error("Failed reading chord vocabulary: " + path.string());
    return result;
}
}  // namespace

class ChordNet::Impl {
    struct Graph {
        core::ExecutionContext & execution;
        ggml_context * ctx = nullptr;
        ggml_cgraph * graph = nullptr;
        ggml_gallocr_t allocator = nullptr;
        core::HostGraphPlan plan;
        ggml_tensor * input = nullptr;
        ggml_tensor * zero = nullptr;
        ggml_tensor * output = nullptr;
        explicit Graph(core::ExecutionContext & e) : execution(e) {}
        ~Graph() {
            release_workspace();
            if (ctx) ggml_free(ctx);
        }
        void allocate() {
            if (allocator) return;
            allocator = ggml_gallocr_new(ggml_backend_get_default_buffer_type(execution.backend()));
            if (!allocator || !ggml_gallocr_alloc_graph(allocator, graph))
                throw std::runtime_error("Cannot allocate ChordNet graph tensors");
            core::prepare_host_graph_plan(execution, graph, plan);
            debug::timing_log_scalar("mulacover.chord.graph_buffer_mb",
                static_cast<double>(ggml_gallocr_get_buffer_size(allocator, 0)) / (1024 * 1024));
        }
        void release_workspace() {
            plan.reset();
            if (!allocator) return;
            core::release_backend_graph_resources(execution.backend(), graph, true);
            ggml_gallocr_free(allocator);
            allocator = nullptr;
            for (auto * tensor = ggml_get_first_tensor(ctx); tensor; tensor = ggml_get_next_tensor(ctx, tensor)) {
                tensor->data = nullptr;
                tensor->buffer = nullptr;
                tensor->extra = nullptr;
            }
            debug::timing_log_scalar("mulacover.chord.graph_buffer_mb", 0.0);
        }
    };
    struct Fold {
        std::unique_ptr<core::BackendWeightStore> store;
        std::array<modules::Conv2dWeights,10> conv;
        modules::BidirectionalLSTMWeights lstm;
        modules::LinearWeights linear;
        std::unique_ptr<Graph> graph;
        int64_t frames = 0;
    };
    core::ExecutionContext & execution_;
    std::array<Fold,5> folds_;
    std::vector<ChordTemplate> vocabulary_;

    Graph & graph_for(Fold & fold, int64_t frames) {
        if (fold.graph && fold.frames == frames) return *fold.graph;
        fold.graph.reset();
        auto target = std::make_unique<Graph>(execution_);
        // LSTMSequence unrolls time. Size the metadata arena and node table from T.
        const size_t nodes = 8192 + static_cast<size_t>(frames) * 256;
        target->ctx = ggml_init({nodes * (ggml_tensor_overhead() + 128) +
            ggml_graph_overhead_custom(nodes, false), nullptr, true});
        if (!target->ctx) throw std::runtime_error("Cannot allocate ChordNet graph context");
        core::ModuleBuildContext ctx{target->ctx, "mulacover.chord", execution_.backend_type()};
        auto x = core::make_tensor(ctx, GGML_TYPE_F32, TensorShape::from_dims({1,1,frames,252}));
        target->input = x.tensor;
        ggml_set_input(target->input);
        for (size_t i = 0; i < 10; ++i) {
            modules::Conv2dConfig config;
            config.in_channels = i == 0 ? 1 : channels[i-1];
            config.out_channels = channels[i];
            config.kernel_height = config.kernel_width = 3;
            config.padding_height = 1;
            config.padding_width = i < 8 ? 1 : 0;
            x = modules::Conv2dModule(config).build(ctx, x, fold.conv[i]);
            x = modules::GroupNormModule({channels[i],channels[i],1e-5f,false,false}).build(ctx,x,{});
            x = modules::SeluModule().build(ctx,x);
            if (i == 2 || i == 5 || i == 7) {
                const int width = i == 7 ? 4 : 3;
                // Existing GGML max pool: NCHW physical [W,H,C,N], kernel (1,width).
                auto * pooled = ggml_pool_2d(ctx.ggml,x.tensor,GGML_OP_POOL_MAX,width,1,width,1,0,0);
                auto shape = x.shape;
                shape.dims[3] /= width;
                x = core::wrap_tensor(pooled,shape);
            }
        }
        // N,C,T,F -> N,T,C,F, contiguous flatten preserving channel-major frequency.
        x = modules::TransposeModule({{0,2,1,3},4}).build(ctx,x);
        x = core::ensure_backend_addressable_layout(ctx,x);
        x = core::reshape_tensor(ctx,x,TensorShape::from_dims({frames,240}));
        auto zero = core::make_tensor(ctx,GGML_TYPE_F32,TensorShape::from_dims({1,96}));
        target->zero = zero.tensor;
        ggml_set_input(target->zero);
        x = modules::BidirectionalLSTMModule({240,96}).build(ctx,x,zero,zero,zero,zero,fold.lstm).sequence;
        x = modules::LinearModule({192,100,true,GGML_PREC_DEFAULT}).build(ctx,x,fold.linear);
        TensorValue probabilities;
        int offset = 0;
        for (int size : head_sizes) {
            auto head = modules::SliceModule({1,offset,size}).build(ctx,x);
            head = modules::SoftmaxModule().build(ctx,head);
            probabilities = probabilities.valid() ? modules::ConcatModule({1}).build(ctx,probabilities,head) : head;
            offset += size;
        }
        target->output = core::ensure_backend_addressable_layout(ctx,probabilities).tensor;
        ggml_set_output(target->output);
        target->graph = ggml_new_graph_custom(ctx.ggml,nodes,false);
        ggml_build_forward_expand(target->graph,target->output);
        core::validate_backend_graph_supported(execution_.backend(),target->graph,"mulacover.chord");
        fold.frames = frames;
        fold.graph = std::move(target);
        return *fold.graph;
    }

public:
    Impl(core::ExecutionContext & execution,
         const std::array<std::shared_ptr<const assets::TensorSource>,5> & sources,
         const std::filesystem::path & vocabulary)
        : execution_(execution), vocabulary_(load_vocabulary(vocabulary)) {
        for (size_t i = 0; i < folds_.size(); ++i) {
            if (!sources[i]) throw std::invalid_argument("ChordNet requires all five tensor sources");
            auto & fold = folds_[i];
            fold.store = std::make_unique<core::BackendWeightStore>(execution.backend(),execution.backend_type(),
                "mulacover.chord.fold" + std::to_string(i),2*1024*1024);
            auto & store = *fold.store;
            const auto & source = *sources[i];
            for (size_t j = 0; j < 10; ++j) {
                fold.conv[j] = modules::binding::conv2d_from_source(store,source,
                    std::string("audio_feature_block.") + conv_names[j],assets::TensorStorageType::Native,
                    channels[j],j == 0 ? 1 : channels[j-1],3,3,true);
            }
            fold.lstm.forward = modules::binding::lstm_cell_from_source(
                store,source,"lstm1",0,240,96,assets::TensorStorageType::Native);
            fold.lstm.reverse = {
                store.load_tensor(source,"lstm1.weight_ih_l0_reverse",assets::TensorStorageType::Native,{384,240}),
                store.load_tensor(source,"lstm1.weight_hh_l0_reverse",assets::TensorStorageType::Native,{384,96}),
                store.load_f32_tensor(source,"lstm1.bias_ih_l0_reverse",{384}),
                store.load_f32_tensor(source,"lstm1.bias_hh_l0_reverse",{384})};
            fold.linear = modules::binding::linear_from_source(store,source,"final_fc1",
                assets::TensorStorageType::Native,100,192,true);
            // condition_linear is retained in upstream checkpoints but never executed.
            store.upload();
        }
    }

    std::vector<ChordEvent> forward(const std::vector<float> & cqt) {
        const auto started = std::chrono::steady_clock::now();
        if (cqt.size() % 252 != 0) throw std::invalid_argument("Chord CQT must have shape [T,252]");
        const size_t frames = cqt.size()/252;
        std::vector<ChordEvent> events;
        if (frames != 0) {
            std::vector<float> probabilities(frames*100,0.0f), prediction(frames*100);
            const std::array<float,96> zero{};
            for (auto & fold : folds_) {
                auto & graph = graph_for(fold,static_cast<int64_t>(frames));
                graph.allocate();
                ggml_backend_tensor_set(graph.input,cqt.data(),0,cqt.size()*sizeof(float));
                ggml_backend_tensor_set(graph.zero,zero.data(),0,sizeof(zero));
                if (core::compute_graph(execution_,graph.graph,graph.plan) != GGML_STATUS_SUCCESS)
                    throw std::runtime_error("ChordNet graph execution failed");
                ggml_backend_tensor_get(graph.output,prediction.data(),0,prediction.size()*sizeof(float));
                graph.release_workspace();
                for (size_t j = 0; j < prediction.size(); ++j) probabilities[j] += prediction[j];
            }
            // numpy.mean on five F32 predictions: sum in fold order, then divide.
            for (auto & p : probabilities) p /= 5.0f;
            const size_t count = vocabulary_.size();
            std::vector<float> scores(count), next(count);
            std::vector<size_t> previous(frames*count), decoded(frames);
            size_t best = 0;
            for (size_t frame = 0; frame < frames; ++frame) {
                const float different = frame ? scores[best]-30.0f : 0.0f;
                for (size_t chord = 0; chord < count; ++chord) {
                    const auto & factors = vocabulary_[chord].factors;
                    size_t offset = 0;
                    float observation = 0;
                    for (size_t head = 0; head < 6; ++head) {
                        const int index = factors[head] + (head == 1 ? 1 : 0);
                        if (index >= 0) observation += std::log(probabilities[frame*100+offset+index]);
                        offset += head_sizes[head];
                    }
                    if (frame == 0) {
                        next[chord] = (chord == 0 ? 0.0f : -std::numeric_limits<float>::infinity()) + observation;
                    } else {
                        const bool keep = scores[chord] > different;
                        previous[frame*count+chord] = keep ? chord : best;
                        next[chord] = std::max(different,scores[chord]) + observation;
                    }
                }
                scores.swap(next);
                best = static_cast<size_t>(std::max_element(scores.begin(),scores.end())-scores.begin());
            }
            decoded.back() = best;
            for (size_t frame = frames-1; frame > 0; --frame)
                decoded[frame-1] = previous[frame*count+decoded[frame]];
            size_t begin = 0;
            constexpr double frame_seconds = 512.0/22050.0;
            for (size_t frame = 0; frame < frames; ++frame) {
                if (frame+1 == frames || decoded[frame+1] != decoded[frame]) {
                    events.push_back({vocabulary_[decoded[frame]].label,begin*frame_seconds,(frame+1)*frame_seconds});
                    begin = frame+1;
                }
            }
        }
        debug::timing_log_scalar("mulacover.chord.wall_ms",
            std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-started).count());
        return events;
    }
};

ChordNet::ChordNet(core::ExecutionContext & execution,
                   const std::array<std::shared_ptr<const assets::TensorSource>,5> & sources,
                   const std::filesystem::path & vocabulary_path)
    : impl_(std::make_unique<Impl>(execution,sources,vocabulary_path)) {}
ChordNet::~ChordNet() = default;
std::vector<ChordEvent> ChordNet::forward(const std::vector<float> & cqt) { return impl_->forward(cqt); }

std::vector<ChordGridEvent> ChordNet::forward(const std::vector<float> & cqt, double bpm) {
    if (!std::isfinite(bpm) || bpm <= 0) throw std::invalid_argument("bpm must be positive and finite");
    const double seconds_per_beat = 60.0/bpm;
    // Explicit ties-to-even, independent of the process floating-point rounding mode.
    const auto round_even = [](double value) -> int64_t {
        const double lower = std::floor(value);
        const double fraction = value-lower;
        const double rounded = lower + (fraction > 0.5 || (fraction == 0.5 && std::fmod(lower,2.0) != 0.0));
        if (!std::isfinite(rounded) || rounded >= 9223372036854775808.0)
            throw std::overflow_error("Chord beat position exceeds int64");
        return static_cast<int64_t>(rounded);
    };
    std::vector<ChordGridEvent> result;
    for (const auto & event : forward(cqt))
        result.push_back({event.label,round_even(event.onset_seconds/seconds_per_beat),
                          round_even(event.offset_seconds/seconds_per_beat)});
    return result;
}

}  // namespace engine::models::mulacover
