#include "engine/community_models/latentsync/scrfd.h"

#include "engine/framework/io/json.h"
#include "engine/framework/modules/activation_modules.h"
#include "engine/framework/modules/primitive_modules.h"
#include "engine/framework/modules/structural_modules.h"
#include "engine/framework/modules/weight_binding.h"

#include <stdexcept>
#include <algorithm>
#include <unordered_map>

namespace engine::models::latentsync {

LatentSyncScrfdWeights load_scrfd_weights(core::BackendWeightStore & store,
    const assets::TensorSource & source, const std::filesystem::path & config_path,
    assets::TensorStorageType type) {
    const auto config = io::json::parse_file(config_path);
    LatentSyncScrfdWeights weights;
    weights.input = config.require("input").as_string();
    weights.input_shape = core::TensorShape::from_dims({1, 3, 512, 512});
    weights.negative_one = store.make_from_f32(core::TensorShape::from_dims({1, 1, 1, 1}), assets::TensorStorageType::F32, {-1.0F});
    for (const auto & item : config.require("nodes").as_array()) {
        LatentSyncScrfdNode node;
        node.op = item.require("op").as_string();
        node.output = item.require("output").as_string();
        for (const auto & input : item.require("inputs").as_array()) node.inputs.push_back(input.as_string());
        const auto & dimensions = item.require("shape").as_array();
        node.shape.rank = dimensions.size();
        if (node.shape.rank > core::kMaxTensorRank) throw std::runtime_error("Unsupported SCRFD tensor rank");
        for (size_t i = 0; i < dimensions.size(); ++i) node.shape.dims[i] = dimensions[i].as_i64();
        const auto prefix = "face.scrfd." + node.output;
        if (node.op == "Conv") {
            const auto & kernel = item.require("kernel").as_array();
            auto & c = node.convolution_config;
            c.out_channels = kernel[0].as_i64();
            c.in_channels = kernel[1].as_i64();
            c.kernel_height = kernel[2].as_i64();
            c.kernel_width = kernel[3].as_i64();
            c.stride_height = static_cast<int>(item.require("stride").as_array()[0].as_i64());
            c.stride_width = static_cast<int>(item.require("stride").as_array()[1].as_i64());
            c.padding_height = static_cast<int>(item.require("padding").as_array()[0].as_i64());
            c.padding_width = static_cast<int>(item.require("padding").as_array()[1].as_i64());
            c.dilation_height = static_cast<int>(item.require("dilation").as_array()[0].as_i64());
            c.dilation_width = static_cast<int>(item.require("dilation").as_array()[1].as_i64());
            node.convolution = modules::binding::conv2d_from_source(store, source, prefix, type,
                c.out_channels, c.in_channels, c.kernel_height, c.kernel_width, true);
        } else if (node.op == "Mul") {
            node.scale = store.load_tensor(source, prefix + ".scale", assets::TensorStorageType::F32, {1, 1, 1, 1});
        } else if (node.op == "Transpose") {
            const auto & axes = item.require("axes").as_array();
            for (size_t i = 0; i < axes.size(); ++i) node.axes[i] = static_cast<int>(axes[i].as_i64());
        }
        weights.nodes.push_back(std::move(node));
    }
    for (const auto & output : config.require("outputs").as_array()) weights.outputs.push_back(output.as_string());
    return weights;
}

std::vector<core::TensorValue> LatentSyncScrfdDetector::build(core::ModuleBuildContext & ctx,
    const core::TensorValue & input, const LatentSyncScrfdWeights & weights) const {
    using namespace modules;
    core::validate_shape(input, weights.input_shape, "LatentSync SCRFD input");
    std::unordered_map<std::string, core::TensorValue> values{{weights.input, input}};
    for (const auto & node : weights.nodes) {
        auto x = values.at(node.inputs.at(0));
        if (node.op == "Conv") {
            x = Conv2dModule(node.convolution_config).build(ctx, x, node.convolution);
        } else if (node.op == "Relu") {
            x = ReluModule().build(ctx, x);
        } else if (node.op == "Sigmoid") {
            x = SigmoidModule().build(ctx, x);
        } else if (node.op == "Add") {
            x = AddModule().build(ctx, x, values.at(node.inputs.at(1)));
        } else if (node.op == "Mul") {
            x = MulModule().build(ctx, x, RepeatModule({x.shape}).build(ctx, node.scale));
        } else if (node.op == "Transpose") {
            x = TransposeModule({node.axes, x.shape.rank}).build(ctx, x);
        } else if (node.op == "Reshape") {
            x = core::reshape_tensor(ctx, core::ensure_backend_addressable_layout(ctx, x), node.shape);
        } else if (node.op == "Resize") {
            x = NearestUpsample2dModule({node.shape.dims[2], node.shape.dims[3]}).build(ctx, x);
        } else if (node.op == "MaxPool" || node.op == "AveragePool") {
            const auto s = x.shape;
            if (s.dims[2] % 2 || s.dims[3] % 2) throw std::runtime_error("SCRFD pooling expects even spatial dimensions");
            // Gather each non-overlapping 2x2 neighborhood into the last dimension.
            x = core::reshape_tensor(ctx, x, core::TensorShape::from_dims({s.dims[0] * s.dims[1] * s.dims[2] / 2, 2, s.dims[3] / 2, 2}));
            x = TransposeModule({{0, 2, 1, 3}, 4}).build(ctx, x);
            x = core::reshape_tensor(ctx, core::ensure_backend_addressable_layout(ctx, x),
                core::TensorShape::from_dims({node.shape.dims[0] * node.shape.dims[1], node.shape.dims[2], node.shape.dims[3], 4}));
            if (node.op == "AveragePool") {
                x = ReduceMeanModule({3}).build(ctx, x);
            } else {
                auto maximum = SliceModule({3, 0, 1}).build(ctx, x);
                const auto minus_one = RepeatModule({maximum.shape}).build(ctx, weights.negative_one);
                for (int64_t i = 1; i < 4; ++i) {
                    const auto next = SliceModule({3, i, 1}).build(ctx, x);
                    const auto difference = AddModule().build(ctx, maximum, MulModule().build(ctx, next, minus_one));
                    maximum = AddModule().build(ctx, ReluModule().build(ctx, difference), next);
                }
                x = maximum;
            }
            x = core::reshape_tensor(ctx, core::ensure_backend_addressable_layout(ctx, x), node.shape);
        } else {
            throw std::runtime_error("Unsupported LatentSync SCRFD operation: " + node.op);
        }
        core::validate_shape(x, node.shape, "LatentSync SCRFD node output");
        values.emplace(node.output, x);
    }
    std::vector<core::TensorValue> outputs;
    for (const auto & name : weights.outputs) outputs.push_back(values.at(name));
    return outputs;
}

std::vector<LatentSyncFaceDetection> decode_scrfd_faces(
    const std::array<std::vector<float>, 9> & heads, float resize_scale,
    float score_threshold, float nms_threshold) {
    if (resize_scale <= 0) throw std::runtime_error("SCRFD resize scale must be positive");
    std::vector<LatentSyncFaceDetection> candidates;
    for (size_t level = 0; level < 3; ++level) {
        const int stride = 8 << level;
        const size_t width = 512 / stride;
        const size_t anchors = width * width * 2;
        if (heads[level].size() != anchors || heads[level + 3].size() != anchors * 4 ||
            heads[level + 6].size() != anchors * 10) {
            throw std::runtime_error("SCRFD output head size mismatch");
        }
        for (size_t i = 0; i < anchors; ++i) {
            if (heads[level][i] < score_threshold) continue;
            const float x = (i / 2 % width) * stride;
            const float y = (i / 2 / width) * stride;
            const auto * box = heads[level + 3].data() + i * 4;
            LatentSyncFaceDetection face;
            face.score = heads[level][i];
            face.box = {(x - box[0] * stride) / resize_scale, (y - box[1] * stride) / resize_scale,
                        (x + box[2] * stride) / resize_scale, (y + box[3] * stride) / resize_scale};
            for (size_t j = 0; j < 10; ++j) {
                face.landmarks[j] = ((j % 2 ? y : x) + heads[level + 6][i * 10 + j] * stride) / resize_scale;
            }
            candidates.push_back(face);
        }
    }
    std::stable_sort(candidates.begin(), candidates.end(), [](const auto & a, const auto & b) {
        return a.score > b.score;
    });
    std::vector<LatentSyncFaceDetection> faces;
    for (const auto & candidate : candidates) {
        bool suppressed = false;
        for (const auto & face : faces) {
            const auto & a = candidate.box;
            const auto & b = face.box;
            const float intersection = std::max(0.0F, std::min(a[2], b[2]) - std::max(a[0], b[0]) + 1) *
                                       std::max(0.0F, std::min(a[3], b[3]) - std::max(a[1], b[1]) + 1);
            const float area_a = (a[2] - a[0] + 1) * (a[3] - a[1] + 1);
            const float area_b = (b[2] - b[0] + 1) * (b[3] - b[1] + 1);
            if (intersection / (area_a + area_b - intersection) > nms_threshold) {
                suppressed = true;
                break;
            }
        }
        if (!suppressed) faces.push_back(candidate);
    }
    return faces;
}

}  // namespace engine::models::latentsync
