#include "engine/framework/neon/execution.h"
#include "operator_kernel.h"

#include <algorithm>
#include <limits>
#include <stdexcept>

namespace engine::neon {

bool available() noexcept {
    return detail::available();
}

struct ExecutionPlan::Impl {
    struct ValueState {
        int input_index = -1;
        std::shared_ptr<const detail::Kernel> kernel;
        std::vector<size_t> sources;
        std::vector<Shape> source_shapes;
        std::vector<DataType> source_types;
        DataType type = DataType::Float32;
        bool retained = false;
        Shape shape{1};
        int slot = -1;
    };
    struct Dispatch {
        const detail::Kernel *kernel = nullptr;
        std::vector<detail::BufferView> inputs;
        float *output = nullptr;
        Shape shape{1};
    };
    std::vector<ValueState> values;
    std::vector<std::vector<float>> buffers;
    std::vector<Dispatch> dispatch;
    std::vector<Shape> input_shapes;
    std::vector<DataType> input_types;
    size_t input_count = 0;
    size_t output_index = std::numeric_limits<size_t>::max();
    Layout output_layout = Layout::ChannelFirst;
    bool dirty = true;
    bool completed = false;
    bool copy_output = false;
    Tensor output{{1}, {}};
    detail::Workspace workspace;
    std::unique_ptr<detail::WorkerTeam> workers;

    void prepare(const std::vector<TensorView> &inputs, int threads) {
        bool changed = dirty || input_shapes.size() != inputs.size();
        for (size_t i = 0; !changed && i < inputs.size(); ++i) {
            changed = !(input_shapes[i] == inputs[i].shape) || input_types[i] != inputs[i].type;
        }
        if (changed) {
            dirty = true;
            input_shapes.clear();
            input_types.clear();
            for (const auto &input : inputs) {
                input_shapes.push_back(input.shape);
                input_types.push_back(input.type);
            }
            detail::WorkspaceRequirements requirements;
            std::vector<size_t> last_use(values.size());
            for (size_t i = 0; i < values.size(); ++i) {
                auto &value = values[i];
                last_use[i] = i;
                if (value.input_index >= 0) {
                    value.shape = inputs[value.input_index].shape;
                    value.type = inputs[value.input_index].type;
                    continue;
                }
                value.source_shapes.resize(value.sources.size(), Shape{1});
                value.source_types.resize(value.sources.size());
                for (size_t source = 0; source < value.sources.size(); ++source) {
                    value.source_shapes[source] = values[value.sources[source]].shape;
                    value.source_types[source] = values[value.sources[source]].type;
                }
                value.kernel->validate_types(value.source_types);
                value.shape = value.kernel->output_shape(value.source_shapes);
                value.shape.elements();
                const auto need = value.kernel->workspace(value.source_shapes);
                requirements.frames = std::max(requirements.frames, need.frames);
                requirements.channels = std::max(requirements.channels, need.channels);
                requirements.kernel_size = std::max(requirements.kernel_size, need.kernel_size);
                requirements.panel_rows = std::max(requirements.panel_rows, need.panel_rows);
                requirements.attention_rows = std::max(requirements.attention_rows, need.attention_rows);
                requirements.quantized_elements = std::max(requirements.quantized_elements, need.quantized_elements);
            }
            for (size_t i = 0; i < values.size(); ++i) {
                for (size_t source : values[i].sources) {
                    last_use[source] = std::max(last_use[source], i);
                }
            }
            last_use[output_index] = values.size();
            if (values[output_index].type != DataType::Float32) {
                throw std::invalid_argument("ARM plan output must be Float32");
            }
            for (size_t i = 0; i < values.size(); ++i) {
                if (values[i].retained) {
                    last_use[i] = values.size();
                }
            }
            output.shape = values[output_index].shape;
            output.shape.layout = output_layout;
            output.values.resize(output.shape.elements());
            const auto &result = values[output_index];
            const bool direct_frame =
                output_index + 1 == values.size() && result.kernel && result.kernel->supports_frame_output();
            copy_output = result.input_index >= 0 || (output_layout != result.shape.layout && !direct_frame);

            std::vector<bool> free(buffers.size(), true);
            std::vector<size_t> sizes;
            sizes.reserve(buffers.size());
            for (const auto &buffer : buffers) {
                sizes.push_back(buffer.size());
            }
            for (size_t i = 0; i < values.size(); ++i) {
                auto &value = values[i];
                value.slot = -1;
                if (!value.kernel) {
                    continue;
                }
                if (i == output_index && !copy_output) {
                    value.slot = -2;
                } else {
                    if (value.kernel->inplace() && !value.sources.empty()) {
                        const size_t source = value.sources.front();
                        if (values[source].slot >= 0 && last_use[source] == i && values[source].shape == value.shape) {
                            value.slot = values[source].slot;
                        }
                    }
                    if (value.slot < 0) {
                        size_t best = free.size();
                        const size_t required = value.shape.elements();
                        for (size_t slot = 0; slot < free.size(); ++slot) {
                            if (!free[slot]) {
                                continue;
                            }
                            if (best == free.size()) {
                                best = slot;
                            } else if (sizes[slot] >= required) {
                                if (sizes[best] < required || sizes[slot] < sizes[best]) {
                                    best = slot;
                                }
                            } else if (sizes[best] < required && sizes[slot] > sizes[best]) {
                                best = slot;
                            }
                        }
                        if (best == free.size()) {
                            buffers.emplace_back();
                            free.push_back(true);
                            sizes.push_back(0);
                        }
                        value.slot = static_cast<int>(best);
                    }
                    free[value.slot] = false;
                    sizes[value.slot] = std::max(sizes[value.slot], value.shape.elements());
                }
                for (size_t source : value.sources) {
                    if (last_use[source] == i && values[source].slot >= 0 && values[source].slot != value.slot) {
                        free[values[source].slot] = true;
                    }
                }
                if (last_use[i] == i && value.slot >= 0) {
                    free[value.slot] = true;
                }
            }
            for (size_t i = 0; i < buffers.size(); ++i) {
                detail::grow(buffers[i], sizes[i]);
            }
            workspace.prepare(requirements.frames, requirements.channels, requirements.kernel_size);
            workspace.quantized_input.resize(requirements.quantized_elements);
            workspace.quantized_scales.resize(requirements.quantized_elements / 32);
            panel_rows_ = requirements.panel_rows;
            attention_rows_ = requirements.attention_rows;
            dirty = false;
        }
        if (panel_rows_ > 0) {
            workspace.reserve_panels(panel_rows_, threads);
        }
        if (attention_rows_ > 0) {
            workspace.reserve_attention(attention_rows_, threads);
        }
        if (!workers || workers->size() != threads) {
            workers = std::make_unique<detail::WorkerTeam>(threads);
        }

        // Input storage can change without shape changes. Bind all views after buffer
        // growth so no dispatch retains a pointer invalidated by allocation.
        dispatch.resize(values.size() - input_count);
        size_t dispatch_index = 0;
        auto data = [&](size_t index) -> const float * {
            const auto &value = values[index];
            if (value.input_index >= 0) {
                return inputs[value.input_index].data;
            }
            return value.slot == -2 ? output.values.data() : buffers[value.slot].data();
        };
        for (size_t i = 0; i < values.size(); ++i) {
            const auto &value = values[i];
            if (!value.kernel) {
                continue;
            }
            auto &call = dispatch[dispatch_index++];
            call.kernel = value.kernel.get();
            call.output = const_cast<float *>(data(i));
            call.shape = value.shape;
            if (i == output_index && !copy_output) {
                call.shape.layout = output_layout;
            }
            call.inputs.resize(value.sources.size());
            for (size_t source = 0; source < value.sources.size(); ++source) {
                call.inputs[source] = {data(value.sources[source]), &values[value.sources[source]].shape};
                const auto &state = values[value.sources[source]];
                if (state.type == DataType::Int32) {
                    call.inputs[source].indices = inputs[state.input_index].indices;
                }
            }
            call.kernel->validate_values(call.inputs);
        }
        output_source_ = data(output_index);
        source_layout_ = values[output_index].shape.layout;
    }

    int panel_rows_ = 0;
    int attention_rows_ = 0;
    const float *output_source_ = nullptr;
    Layout source_layout_ = Layout::ChannelFirst;
};

ExecutionPlan::ExecutionPlan() : impl_(std::make_unique<Impl>()) {}
ExecutionPlan::~ExecutionPlan() = default;

ExecutionPlan::Value ExecutionPlan::input() {
    auto &p = *impl_;
    p.values.emplace_back();
    p.values.back().input_index = static_cast<int>(p.input_count++);
    p.dirty = true;
    return {p.values.size() - 1, impl_.get()};
}

ExecutionPlan::Value ExecutionPlan::add(const Operator &operation, std::initializer_list<Value> inputs) {
    auto &p = *impl_;
    Impl::ValueState value;
    value.kernel = operation.kernel_;
    for (const auto &input : inputs) {
        if (input.owner_ != impl_.get() || input.index_ >= p.values.size()) {
            throw std::invalid_argument("ARM plan value belongs to another plan");
        }
        value.sources.push_back(input.index_);
    }
    p.values.push_back(std::move(value));
    p.dirty = true;
    return {p.values.size() - 1, impl_.get()};
}

void ExecutionPlan::set_output(Value value, Layout layout) {
    if (value.owner_ != impl_.get() || value.index_ >= impl_->values.size()) {
        throw std::invalid_argument("ARM plan output belongs to another plan");
    }
    impl_->output_index = value.index_;
    impl_->output_layout = layout;
    impl_->dirty = true;
}

void ExecutionPlan::retain(Value value) {
    if (value.owner_ != impl_.get() || value.index_ >= impl_->values.size() || !impl_->values[value.index_].kernel) {
        throw std::invalid_argument("ARM retained value must be an owned operator result");
    }
    impl_->values[value.index_].retained = true;
    impl_->dirty = true;
}

TensorView ExecutionPlan::view(Value value) const {
    if (value.owner_ != impl_.get() || value.index_ >= impl_->values.size() || !impl_->completed || impl_->dirty ||
        !impl_->values[value.index_].retained) {
        throw std::invalid_argument("ARM result is not retained or has not completed");
    }
    const auto &state = impl_->values[value.index_];
    if (state.slot == -2) {
        return TensorView(impl_->output.values, impl_->output.shape);
    }
    return TensorView(impl_->buffers[state.slot].data(), state.shape.elements(), state.shape);
}

const Tensor &ExecutionPlan::run(const std::vector<TensorView> &inputs, int threads) {
    auto &p = *impl_;
    p.completed = false;
    if (threads <= 0 || inputs.size() != p.input_count || p.output_index >= p.values.size()) {
        throw std::invalid_argument("ARM plan input count, output or thread count invalid");
    }
    for (const auto &view : inputs) {
        if ((view.type == DataType::Float32 ? !view.data : view.type != DataType::Int32 || !view.indices) ||
            view.size != view.shape.elements()) {
            throw std::invalid_argument("ARM plan input size mismatch");
        }
    }
    p.prepare(inputs, threads);
    p.workers->run([&](detail::WorkerContext &worker) {
        for (const auto &call : p.dispatch) {
            call.kernel->compute(call.inputs, call.output, call.shape, p.workspace, worker);
            worker.synchronize();
        }
        if (p.copy_output) {
            int begin, end;
            detail::split_range(static_cast<int>(p.output.values.size()), worker.index, worker.count, begin, end);
            if (p.output.shape.layout == p.source_layout_) {
                std::copy(p.output_source_ + begin, p.output_source_ + end, p.output.values.data() + begin);
            } else {
                const int channels = p.output.shape.dimensions[0], frames = p.output.shape.dimensions[1];
                for (int index = begin; index < end; ++index) {
                    const int channel = p.output.shape.layout == Layout::FrameMajor ? index % channels : index / frames;
                    const int frame = p.output.shape.layout == Layout::FrameMajor ? index / channels : index % frames;
                    p.output.values[index] =
                        p.output_source_[p.source_layout_ == Layout::FrameMajor ? frame * channels + channel
                                                                                : channel * frames + frame];
                }
            }
        }
    });
    p.completed = true;
    return p.output;
}

} // namespace engine::neon
