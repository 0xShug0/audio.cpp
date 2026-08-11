#include "flow_impl.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

namespace engine::models::dots_tts::detail {

class SoarDiTRunnerImpl final : public SoarDiTRunner {
    struct StepGraphState {
        std::unique_ptr<ggml_context, GgmlContextDeleter> ggml;
        std::unique_ptr<std::remove_pointer_t<ggml_gallocr_t>, GgmlGallocrDeleter> gallocr;
        ggml_cgraph * graph = nullptr;
        core::HostGraphPlan plan;
        ggml_tensor * sequence = nullptr;
        ggml_tensor * positions = nullptr;
        ggml_tensor * slots = nullptr;
        ggml_tensor * modulation_indices = nullptr;
        ggml_tensor * guidance_scale = nullptr;
        ggml_tensor * mask = nullptr;
        ggml_tensor * output = nullptr;
        std::vector<int32_t> slot_values;
        int64_t capacity = 0;
        int64_t tail_len = 0;
        int64_t unit_len = 0;
        ggml_tensor * modulation_source_tensor = nullptr;
    };

public:
    SoarDiTRunnerImpl(std::shared_ptr<const DotFlowWeights> weights, int64_t call_count)
        : weights_(std::move(weights)), call_count_(call_count) {
        if (call_count_ <= 0) {
            throw std::runtime_error("DotTTS SOAR DiT cache requires positive call count");
        }
    }

    ~SoarDiTRunnerImpl() {
        release_runtime_graphs();
        release_cache();
    }

    void ensure_cache(int64_t capacity) {
        std::lock_guard<std::mutex> lock(mutex_);
        ensure_cache_locked(capacity);
    }

    int64_t valid_steps(int64_t ode_index) const {
        if (ode_index < 0 || ode_index >= static_cast<int64_t>(valid_steps_.size())) {
            throw std::runtime_error("DotTTS SOAR DiT cache call index is invalid");
        }
        return valid_steps_[static_cast<size_t>(ode_index)];
    }

    void prefill(
        int64_t ode_index,
        const std::vector<float> & branch_sequence,
        int64_t steps,
        const DotModulationOutput & modulations,
        DotsFlowRuntimeStats * runtime_stats) {
        std::lock_guard<std::mutex> lock(mutex_);
        const auto & config = weights_->config.dit;
        if (steps < 0 || static_cast<int64_t>(branch_sequence.size()) != 2 * steps * config.hidden_size) {
            throw std::runtime_error("DotTTS SOAR DiT prefill input shape mismatch");
        }
        if (ode_index < 0 || ode_index >= call_count_) {
            throw std::runtime_error("DotTTS SOAR DiT prefill call index is invalid");
        }
        if (steps == 0) {
            valid_steps_[static_cast<size_t>(ode_index)] = 0;
            return;
        }
        if (steps > capacity_) {
            throw std::runtime_error("DotTTS SOAR DiT prefill exceeds cache capacity");
        }
        const auto graph_start = Clock::now();
        ensure_prefill_graph_locked(steps, modulations);
        const double graph_ms = engine::debug::elapsed_ms(graph_start);

        const auto upload_start = Clock::now();
        ggml_backend_tensor_set(prefill_sequence_, branch_sequence.data(), 0, branch_sequence.size() * sizeof(float));
        const auto positions = build_position_range(0, steps);
        ggml_backend_tensor_set(prefill_positions_, positions.data(), 0, positions.size() * sizeof(int32_t));
        const int32_t rows[2] = {
            static_cast<int32_t>(2 * ode_index),
            static_cast<int32_t>(2 * ode_index + 1),
        };
        ggml_backend_tensor_set(prefill_modulation_indices_, rows, 0, sizeof(rows));
        prefill_slot_values_.resize(static_cast<size_t>(2 * steps));
        for (int64_t row = 0; row < steps; ++row) {
            prefill_slot_values_[static_cast<size_t>(row)] = static_cast<int32_t>((2 * ode_index) * capacity_ + row);
            prefill_slot_values_[static_cast<size_t>(steps + row)] = static_cast<int32_t>((2 * ode_index + 1) * capacity_ + row);
        }
        ggml_backend_tensor_set(prefill_slots_, prefill_slot_values_.data(), 0, prefill_slot_values_.size() * sizeof(int32_t));
        const auto mask = build_prefill_mask_values(steps);
        ggml_backend_tensor_set(prefill_mask_, mask.data(), 0, mask.size() * sizeof(ggml_fp16_t));
        const double upload_ms = engine::debug::elapsed_ms(upload_start);

        valid_steps_[static_cast<size_t>(ode_index)] = 0;
        const auto compute_start = Clock::now();
        if (core::compute_graph(*weights_->execution_context, prefill_graph_, prefill_plan_, "dots_tts.flow.soar.prefill") != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("DotTTS SOAR DiT prefill graph compute failed");
        }
        valid_steps_[static_cast<size_t>(ode_index)] = steps;
        const double compute_ms = engine::debug::elapsed_ms(compute_start);
        if (runtime_stats != nullptr) {
            runtime_stats->velocity_graph_ms += graph_ms;
            runtime_stats->velocity_input_upload_ms += upload_ms;
            runtime_stats->velocity_compute_ms += compute_ms;
        }
    }

    DotsVelocityOutput run_step(
        int64_t ode_index,
        const std::vector<float> & tail_sequence,
        int64_t persistent_len,
        int64_t unit_len,
        const DotModulationOutput & modulations,
        float guidance_scale,
        DotsFlowRuntimeStats * runtime_stats) {
        std::lock_guard<std::mutex> lock(mutex_);
        const auto & config = weights_->config;
        const int64_t tail_len = 2 * unit_len;
        if (ode_index < 0 || ode_index >= call_count_ ||
            persistent_len < 0 || unit_len <= 0 ||
            persistent_len != valid_steps_[static_cast<size_t>(ode_index)] ||
            persistent_len + unit_len > capacity_ ||
            static_cast<int64_t>(tail_sequence.size()) != 2 * tail_len * config.dit.hidden_size) {
            throw std::runtime_error("DotTTS SOAR DiT cached step input shape mismatch");
        }
        const auto graph_start = Clock::now();
        StepGraphState & step = ensure_step_graph_locked(ode_index, capacity_, tail_len, unit_len, modulations);
        const double graph_ms = engine::debug::elapsed_ms(graph_start);

        const auto upload_start = Clock::now();
        ggml_backend_tensor_set(step.sequence, tail_sequence.data(), 0, tail_sequence.size() * sizeof(float));
        const auto positions = build_position_range(persistent_len, tail_len);
        ggml_backend_tensor_set(step.positions, positions.data(), 0, positions.size() * sizeof(int32_t));
        step.slot_values.resize(static_cast<size_t>(2 * unit_len));
        for (int64_t row = 0; row < unit_len; ++row) {
            step.slot_values[static_cast<size_t>(2 * row)] = static_cast<int32_t>((2 * ode_index) * capacity_ + persistent_len + row);
            step.slot_values[static_cast<size_t>(2 * row + 1)] = static_cast<int32_t>((2 * ode_index + 1) * capacity_ + persistent_len + row);
        }
        ggml_backend_tensor_set(step.slots, step.slot_values.data(), 0, step.slot_values.size() * sizeof(int32_t));
        const int32_t rows[2] = {
            static_cast<int32_t>(2 * ode_index),
            static_cast<int32_t>(2 * ode_index + 1),
        };
        ggml_backend_tensor_set(step.modulation_indices, rows, 0, sizeof(rows));
        ggml_backend_tensor_set(step.guidance_scale, &guidance_scale, 0, sizeof(float));
        const auto mask = build_cached_update_mask_values(capacity_, persistent_len, unit_len);
        ggml_backend_tensor_set(step.mask, mask.data(), 0, mask.size() * sizeof(ggml_fp16_t));
        const double upload_ms = engine::debug::elapsed_ms(upload_start);

        const auto compute_start = Clock::now();
        if (core::compute_graph(*weights_->execution_context, step.graph, step.plan, "dots_tts.flow.soar.step") != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("DotTTS SOAR DiT cached step graph compute failed");
        }
        valid_steps_[static_cast<size_t>(ode_index)] += unit_len;
        const double compute_ms = engine::debug::elapsed_ms(compute_start);

        DotsVelocityOutput out;
        out.frames = config.patch_size;
        out.latent_dim = config.latent_dim;
        const auto read_start = Clock::now();
        out.values = core::read_tensor_f32(step.output);
        const double read_ms = engine::debug::elapsed_ms(read_start);
        if (runtime_stats != nullptr) {
            ++runtime_stats->velocity_calls;
            runtime_stats->velocity_graph_ms += graph_ms;
            runtime_stats->velocity_input_upload_ms += upload_ms;
            runtime_stats->velocity_compute_ms += compute_ms;
            runtime_stats->velocity_output_read_ms += read_ms;
        }
        return out;
    }

    void release_runtime_graphs() {
        if (prefill_graph_ != nullptr) {
            core::release_backend_graph_resources(weights_->execution_context->backend(), prefill_graph_);
        }
        for (auto & step : step_graphs_) {
            release_step_graph(step);
        }
        prefill_gallocr_.reset();
        prefill_ggml_.reset();
        prefill_plan_.reset();
        prefill_graph_ = nullptr;
        prefill_sequence_ = nullptr;
        prefill_positions_ = nullptr;
        prefill_modulation_indices_ = nullptr;
        prefill_slots_ = nullptr;
        prefill_mask_ = nullptr;
        prefill_slot_values_.clear();
        prefill_steps_ = 0;
        prefill_modulation_source_tensor_ = nullptr;
        step_graphs_.clear();
    }

private:
    void release_step_graph(StepGraphState & step) {
        if (step.graph != nullptr) {
            core::release_backend_graph_resources(weights_->execution_context->backend(), step.graph);
        }
        step.gallocr.reset();
        step.ggml.reset();
        step.plan.reset();
        step.graph = nullptr;
        step.sequence = nullptr;
        step.positions = nullptr;
        step.slots = nullptr;
        step.modulation_indices = nullptr;
        step.guidance_scale = nullptr;
        step.mask = nullptr;
        step.output = nullptr;
        step.slot_values.clear();
        step.capacity = 0;
        step.tail_len = 0;
        step.unit_len = 0;
        step.modulation_source_tensor = nullptr;
    }

    void ensure_cache_locked(int64_t capacity) {
        if (capacity <= 0) {
            throw std::runtime_error("DotTTS SOAR DiT cache requires positive capacity");
        }
        if (cache_ctx_ != nullptr && capacity_ >= capacity) {
            return;
        }
        release_runtime_graphs();
        release_cache();
        const auto & config = weights_->config.dit;
        const int64_t dim = head_dim(config);
        ggml_init_params params{kSmallGraphContextBytes, nullptr, true};
        cache_ctx_.reset(ggml_init(params));
        if (cache_ctx_ == nullptr) {
            throw std::runtime_error("failed to initialize DotTTS SOAR DiT cache context");
        }
        core::ModuleBuildContext cache_build_ctx{cache_ctx_.get(), "dots_tts.flow.soar.cache", weights_->execution_context->backend_type()};
        cache_keys_.clear();
        cache_values_.clear();
        cache_keys_.reserve(weights_->blocks.size());
        cache_values_.reserve(weights_->blocks.size());
        const ggml_type cache_type = cached_dit_kv_type(weights_->execution_context->backend_type());
        for (int64_t layer = 0; layer < config.num_layers; ++layer) {
            cache_keys_.push_back(core::make_tensor(
                cache_build_ctx,
                cache_type,
                core::TensorShape::from_dims({2 * call_count_, capacity, config.num_heads, dim})));
            cache_values_.push_back(core::make_tensor(
                cache_build_ctx,
                cache_type,
                core::TensorShape::from_dims({2 * call_count_, capacity, config.num_heads, dim})));
        }
        cache_buffer_ = ggml_backend_alloc_ctx_tensors(cache_ctx_.get(), weights_->execution_context->backend());
        if (cache_buffer_ == nullptr) {
            release_cache();
            throw std::runtime_error("failed to allocate DotTTS SOAR DiT cache memory");
        }
        capacity_ = capacity;
        valid_steps_.assign(static_cast<size_t>(call_count_), 0);
    }

    void release_cache() {
        cache_keys_.clear();
        cache_values_.clear();
        if (cache_buffer_ != nullptr) {
            ggml_backend_buffer_free(cache_buffer_);
            cache_buffer_ = nullptr;
        }
        cache_ctx_.reset();
        capacity_ = 0;
        valid_steps_.clear();
    }

    core::TensorValue select_modulation_rows(
        core::ModuleBuildContext & ctx,
        const DotModulationOutput & modulations,
        ggml_tensor ** index_tensor) const {
        if (!modulations.backend_value.valid() || 2 * call_count_ - 1 >= modulations.rows) {
            throw std::runtime_error("DotTTS SOAR DiT modulation rows are invalid");
        }
        auto indices = core::make_tensor(ctx, GGML_TYPE_I32, core::TensorShape::from_dims({2}));
        *index_tensor = indices.tensor;
        return core::wrap_tensor(
            ggml_get_rows(
                ctx.ggml,
                core::ensure_backend_addressable_layout(ctx, modulations.backend_value).tensor,
                indices.tensor),
            core::TensorShape::from_dims({2, modulations.width}),
            GGML_TYPE_F32);
    }

    void ensure_prefill_graph_locked(int64_t steps, const DotModulationOutput & modulations) {
        if (prefill_ggml_ != nullptr && prefill_steps_ == steps &&
            prefill_modulation_source_tensor_ == modulations.backend_value.tensor) {
            return;
        }
        if (prefill_graph_ != nullptr) {
            core::release_backend_graph_resources(weights_->execution_context->backend(), prefill_graph_);
        }
        prefill_gallocr_.reset();
        prefill_ggml_.reset();
        prefill_plan_.reset();
        prefill_graph_ = nullptr;
        prefill_sequence_ = nullptr;
        prefill_positions_ = nullptr;
        prefill_modulation_indices_ = nullptr;
        prefill_slots_ = nullptr;
        prefill_mask_ = nullptr;
        prefill_slot_values_.clear();
        prefill_steps_ = 0;
        prefill_modulation_source_tensor_ = nullptr;

        ggml_init_params params{kLargeGraphContextBytes, nullptr, true};
        prefill_ggml_.reset(ggml_init(params));
        if (prefill_ggml_ == nullptr) {
            throw std::runtime_error("failed to initialize DotTTS SOAR DiT prefill graph context");
        }
        const auto & config = weights_->config.dit;
        core::ModuleBuildContext build_ctx{prefill_ggml_.get(), "dots_tts.flow.soar.prefill", weights_->execution_context->backend_type()};
        auto sequence = core::make_tensor(build_ctx, GGML_TYPE_F32, core::TensorShape::from_dims({2, steps, config.hidden_size}));
        auto positions = core::make_tensor(build_ctx, GGML_TYPE_I32, core::TensorShape::from_dims({steps}));
        auto slots = core::make_tensor(build_ctx, GGML_TYPE_I32, core::TensorShape::from_dims({2 * steps}));
        auto mask = core::make_tensor(build_ctx, GGML_TYPE_F16, core::TensorShape::from_dims({1, 1, steps, steps}));
        prefill_sequence_ = sequence.tensor;
        prefill_positions_ = positions.tensor;
        prefill_slots_ = slots.tensor;
        prefill_mask_ = mask.tensor;
        auto modulation = select_modulation_rows(build_ctx, modulations, &prefill_modulation_indices_);
        auto x = modules::LinearModule({config.hidden_size, config.hidden_size, true}).build(build_ctx, sequence, weights_->input_layer);
        prefill_graph_ = ggml_new_graph_custom(prefill_ggml_.get(), 262144, false);
        for (size_t layer = 0; layer < weights_->blocks.size(); ++layer) {
            auto mods = modules::SliceModule({
                static_cast<int>(modulation.shape.rank - 1),
                static_cast<int64_t>(layer) * 6 * config.hidden_size,
                6 * config.hidden_size,
            }).build(build_ctx, modulation);
            auto block = dit_block_with_mods_and_cache(build_ctx, x, mods, positions, weights_->blocks[layer], config, mask, std::nullopt, std::nullopt, true);
            auto updated_key = set_dit_cache_flat_rows(build_ctx, cache_keys_.at(layer), block.key, slots);
            auto updated_value = set_dit_cache_flat_rows(build_ctx, cache_values_.at(layer), block.value, slots);
            ggml_build_forward_expand(prefill_graph_, updated_key.tensor);
            ggml_build_forward_expand(prefill_graph_, updated_value.tensor);
            x = block.output;
        }
        core::validate_backend_graph_supported(weights_->execution_context->backend(), prefill_graph_, "dots_tts.flow.soar.prefill");
        prefill_gallocr_.reset(ggml_gallocr_new(ggml_backend_get_default_buffer_type(weights_->execution_context->backend())));
        if (prefill_gallocr_ == nullptr ||
            !ggml_gallocr_reserve(prefill_gallocr_.get(), prefill_graph_) ||
            !ggml_gallocr_alloc_graph(prefill_gallocr_.get(), prefill_graph_)) {
            release_runtime_graphs();
            throw std::runtime_error("failed to allocate DotTTS SOAR DiT prefill graph memory");
        }
        core::prepare_host_graph_plan(*weights_->execution_context, prefill_graph_, prefill_plan_);
        prefill_steps_ = steps;
        prefill_modulation_source_tensor_ = modulations.backend_value.tensor;
    }

    StepGraphState & ensure_step_graph_locked(
        int64_t ode_index,
        int64_t capacity,
        int64_t tail_len,
        int64_t unit_len,
        const DotModulationOutput & modulations) {
        if (ode_index < 0 || ode_index >= call_count_) {
            throw std::runtime_error("DotTTS SOAR DiT cached step graph call index is invalid");
        }
        if (step_graphs_.empty()) {
            step_graphs_.resize(static_cast<size_t>(call_count_));
        }
        auto & step = step_graphs_.at(static_cast<size_t>(ode_index));
        if (step.ggml != nullptr && step.capacity == capacity && step.tail_len == tail_len &&
            step.unit_len == unit_len && step.modulation_source_tensor == modulations.backend_value.tensor) {
            return step;
        }
        release_step_graph(step);

        ggml_init_params params{kLargeGraphContextBytes, nullptr, true};
        step.ggml.reset(ggml_init(params));
        if (step.ggml == nullptr) {
            throw std::runtime_error("failed to initialize DotTTS SOAR DiT cached step graph context");
        }
        const auto & config = weights_->config.dit;
        const int64_t dim = head_dim(config);
        core::ModuleBuildContext build_ctx{step.ggml.get(), "dots_tts.flow.soar.step", weights_->execution_context->backend_type()};
        auto sequence = core::make_tensor(build_ctx, GGML_TYPE_F32, core::TensorShape::from_dims({2, tail_len, config.hidden_size}));
        auto positions = core::make_tensor(build_ctx, GGML_TYPE_I32, core::TensorShape::from_dims({tail_len}));
        auto slots = core::make_tensor(build_ctx, GGML_TYPE_I32, core::TensorShape::from_dims({2 * unit_len}));
        auto guidance = core::make_tensor(build_ctx, GGML_TYPE_F32, core::TensorShape::from_dims({1}));
        auto mask = core::make_tensor(build_ctx, GGML_TYPE_F16, core::TensorShape::from_dims({1, 1, tail_len, capacity + tail_len}));
        step.sequence = sequence.tensor;
        step.positions = positions.tensor;
        step.slots = slots.tensor;
        step.guidance_scale = guidance.tensor;
        step.mask = mask.tensor;
        auto modulation = select_modulation_rows(build_ctx, modulations, &step.modulation_indices);
        auto x = modules::LinearModule({config.hidden_size, config.hidden_size, true}).build(build_ctx, sequence, weights_->input_layer);
        step.graph = ggml_new_graph_custom(step.ggml.get(), 262144, false);
        for (size_t layer = 0; layer < weights_->blocks.size(); ++layer) {
            auto mods = modules::SliceModule({
                static_cast<int>(modulation.shape.rank - 1),
                static_cast<int64_t>(layer) * 6 * config.hidden_size,
                6 * config.hidden_size,
            }).build(build_ctx, modulation);
            auto block = dit_block_with_mods_and_cache(
                build_ctx,
                x,
                mods,
                positions,
                weights_->blocks[layer],
                config,
                mask,
                view_dit_cache_banks(build_ctx, cache_keys_.at(layer), 2 * ode_index, capacity, config.num_heads, dim),
                view_dit_cache_banks(build_ctx, cache_values_.at(layer), 2 * ode_index, capacity, config.num_heads, dim),
                true);
            auto updated_key = cache_keys_.at(layer);
            auto updated_value = cache_values_.at(layer);
            for (int64_t row = 0; row < unit_len; ++row) {
                auto slot = modules::SliceModule({0, 2 * row, 2}).build(build_ctx, slots);
                auto key_row = modules::SliceModule({1, row, 1}).build(build_ctx, block.key);
                auto value_row = modules::SliceModule({1, row, 1}).build(build_ctx, block.value);
                updated_key = set_dit_cache_flat_rows(build_ctx, updated_key, key_row, slot);
                updated_value = set_dit_cache_flat_rows(build_ctx, updated_value, value_row, slot);
            }
            ggml_build_forward_expand(step.graph, updated_key.tensor);
            ggml_build_forward_expand(step.graph, updated_value.tensor);
            x = block.output;
        }
        x = modules::SliceModule({1, unit_len + 1, weights_->config.patch_size}).build(build_ctx, x);
        auto final_mods = modules::SliceModule({
            static_cast<int>(modulation.shape.rank - 1),
            static_cast<int64_t>(weights_->blocks.size()) * 6 * config.hidden_size,
            2 * config.hidden_size,
        }).build(build_ctx, modulation);
        auto branch_output = final_projection_with_mods(build_ctx, x, final_mods, *weights_);
        auto cond = modules::SliceModule({0, 0, 1}).build(build_ctx, branch_output);
        auto uncond = modules::SliceModule({0, 1, 1}).build(build_ctx, branch_output);
        auto delta = core::wrap_tensor(
            ggml_sub(build_ctx.ggml, cond.tensor, uncond.tensor),
            cond.shape,
            GGML_TYPE_F32);
        auto guided_delta = core::wrap_tensor(
            ggml_mul(build_ctx.ggml, delta.tensor, guidance.tensor),
            delta.shape,
            GGML_TYPE_F32);
        auto output = modules::AddModule().build(build_ctx, cond, guided_delta);
        step.output = output.tensor;
        ggml_build_forward_expand(step.graph, step.output);
        core::validate_backend_graph_supported(weights_->execution_context->backend(), step.graph, "dots_tts.flow.soar.step");
        step.gallocr.reset(ggml_gallocr_new(ggml_backend_get_default_buffer_type(weights_->execution_context->backend())));
        if (step.gallocr == nullptr ||
            !ggml_gallocr_reserve(step.gallocr.get(), step.graph) ||
            !ggml_gallocr_alloc_graph(step.gallocr.get(), step.graph)) {
            release_runtime_graphs();
            throw std::runtime_error("failed to allocate DotTTS SOAR DiT cached step graph memory");
        }
        core::prepare_host_graph_plan(*weights_->execution_context, step.graph, step.plan);
        step.capacity = capacity;
        step.tail_len = tail_len;
        step.unit_len = unit_len;
        step.modulation_source_tensor = modulations.backend_value.tensor;
        return step;
    }

    std::shared_ptr<const DotFlowWeights> weights_;
    int64_t call_count_ = 0;
    mutable std::mutex mutex_;
    std::unique_ptr<ggml_context, GgmlContextDeleter> cache_ctx_;
    ggml_backend_buffer_t cache_buffer_ = nullptr;
    std::vector<core::TensorValue> cache_keys_;
    std::vector<core::TensorValue> cache_values_;
    int64_t capacity_ = 0;
    std::vector<int64_t> valid_steps_;

    std::unique_ptr<ggml_context, GgmlContextDeleter> prefill_ggml_;
    std::unique_ptr<std::remove_pointer_t<ggml_gallocr_t>, GgmlGallocrDeleter> prefill_gallocr_;
    ggml_cgraph * prefill_graph_ = nullptr;
    core::HostGraphPlan prefill_plan_;
    ggml_tensor * prefill_sequence_ = nullptr;
    ggml_tensor * prefill_positions_ = nullptr;
    ggml_tensor * prefill_modulation_indices_ = nullptr;
    ggml_tensor * prefill_slots_ = nullptr;
    ggml_tensor * prefill_mask_ = nullptr;
    std::vector<int32_t> prefill_slot_values_;
    int64_t prefill_steps_ = 0;
    ggml_tensor * prefill_modulation_source_tensor_ = nullptr;

    std::vector<StepGraphState> step_graphs_;
};



std::unique_ptr<SoarDiTRunner> make_soar_dit_runner(std::shared_ptr<const DotFlowWeights> weights, int64_t call_count) {
    return std::make_unique<SoarDiTRunnerImpl>(std::move(weights), call_count);
}

DotsLatentMatrix decode_next_soar(DotFlowSharedRuntime & runtime, const DotsFlowDecodeRequest & request, FlowDecodeCacheState * decode_state) {
    const auto & config = runtime.weights->config;
    const int64_t hidden_size = config.dit.hidden_size;
    const int64_t latent_patch_size = config.patch_size;
    const int64_t latent_dim = config.latent_dim;
    const int64_t total_len = request.fm_seq_len + latent_patch_size;
    if (request.sequence == nullptr || request.cfg_sequence == nullptr || request.speaker_condition == nullptr) {
        throw std::runtime_error("DotTTS SOAR flow decode request is missing borrowed inputs");
    }
    const auto & sequence = *request.sequence;
    const auto & cfg_sequence = *request.cfg_sequence;
    const auto & speaker_condition = *request.speaker_condition;
    if (request.fm_seq_len <= 0 || request.num_inference_steps <= 0 ||
        static_cast<int64_t>(sequence.size()) < request.fm_seq_len * hidden_size ||
        static_cast<int64_t>(cfg_sequence.size()) < request.fm_seq_len * hidden_size ||
        static_cast<int64_t>(speaker_condition.size()) != hidden_size) {
        throw std::runtime_error("DotTTS SOAR flow decode request shape mismatch");
    }
    const auto rng_policy = engine::sampling::resolve_torch_cuda_sampling_policy(
        runtime.weights->execution_context->backend_type(),
        runtime.weights->execution_context->config().device,
        "dots_tts.flow.rng",
        "DotTTS",
        engine::sampling::TorchCudaSamplingPolicyFailureMode::StrictCuda);
    std::vector<float> z = engine::sampling::generate_torch_cuda_tensor_iterator_randn(
        static_cast<size_t>(latent_patch_size * latent_dim),
        request.seed,
        request.rng_offset_blocks,
        rng_policy,
        engine::sampling::TorchRandnPrecision::BFloat16);
    const auto positions = build_decode_positions(total_len, request.fm_seq_len, latent_patch_size);
    const auto attention_mask = build_decode_mask(2, total_len, request.fm_seq_len, latent_patch_size, 1);

    const float dt = 1.0F / static_cast<float>(request.num_inference_steps);
    std::vector<float> call_times;
    call_times.reserve(static_cast<size_t>(request.num_inference_steps * 4));
    for (int64_t step = 0; step < request.num_inference_steps; ++step) {
        const float t = static_cast<float>(step) * dt;
        switch (request.ode_method) {
            case DotsOdeMethod::Euler:
                call_times.push_back(t);
                break;
            case DotsOdeMethod::Midpoint:
                call_times.push_back(t);
                call_times.push_back(t + 0.5F * dt);
                break;
            case DotsOdeMethod::Rk4:
                call_times.push_back(t);
                call_times.push_back(t + 0.5F * dt);
                call_times.push_back(t + 0.5F * dt);
                call_times.push_back(t + dt);
                break;
        }
    }
    std::vector<float> modulation_timesteps;
    std::vector<float> modulation_speaker_condition;
    modulation_timesteps.reserve(call_times.size() * 2);
    modulation_speaker_condition.reserve(call_times.size() * static_cast<size_t>(2 * hidden_size));
    for (float t : call_times) {
        modulation_timesteps.push_back(t);
        modulation_timesteps.push_back(t);
        modulation_speaker_condition.insert(modulation_speaker_condition.end(), speaker_condition.begin(), speaker_condition.end());
        modulation_speaker_condition.insert(modulation_speaker_condition.end(), static_cast<size_t>(hidden_size), 0.0F);
    }
    DotModulationOutput modulations;
        if (decode_state != nullptr &&
        decode_state->soar.has_value() &&
        decode_state->soar->num_inference_steps == request.num_inference_steps &&
        decode_state->soar->ode_method == request.ode_method &&
        !decode_state->soar->use_duration_embedding &&
        decode_state->soar->speaker_condition == speaker_condition &&
        decode_state->soar->output.rows == static_cast<int64_t>(modulation_timesteps.size())) {
        modulations = decode_state->soar->output;
    } else {
        modulations = run_modulation(runtime, modulation_timesteps, {}, modulation_speaker_condition, request.runtime_stats);
        if (decode_state != nullptr) {
            decode_state->meanflow.reset();
            decode_state->meanflow_dit.clear();
            decode_state->meanflow_dit_steps = 0;
            decode_state->meanflow_dit_capacity = 0;
            decode_state->soar_dit.reset();
            decode_state->soar_dit_calls = 0;
            decode_state->soar_dit_capacity = 0;
            decode_state->soar = FlowDecodeCacheState::CachedModulations{
                modulations,
                speaker_condition,
                request.num_inference_steps,
                request.ode_method,
                false,
            };
        }
    }
    int64_t modulation_call = 0;
    const int64_t hidden_patch_size = 1;
    const int64_t unit_len = hidden_patch_size + latent_patch_size;
    const int64_t prefix_len = request.fm_seq_len - hidden_patch_size;
    if (prefix_len < 0 || prefix_len % unit_len != 0) {
        throw std::runtime_error("DotTTS SOAR DiT expects unit-aligned finalized history");
    }
    const bool use_cached_dit = request.use_cached_dit && decode_state != nullptr && prefix_len >= unit_len;
    std::vector<float> persistent_sequence;
    int64_t persistent_len = 0;
    if (use_cached_dit) {
        if (request.cache_capacity < prefix_len) {
            throw std::runtime_error("DotTTS SOAR DiT cache capacity is smaller than the current prefix");
        }
        const int64_t capacity = resolve_dit_cache_capacity_tokens(request.fm_seq_len, unit_len);
        const int64_t call_count = static_cast<int64_t>(modulation_timesteps.size()) / 2;
        if (decode_state->soar_dit_calls != call_count ||
            decode_state->soar_dit_capacity < capacity ||
            decode_state->soar_dit == nullptr) {
            decode_state->soar_dit = make_soar_dit_runner(runtime.weights, call_count);
            decode_state->soar_dit->ensure_cache(capacity);
            decode_state->soar_dit_calls = call_count;
            decode_state->soar_dit_capacity = capacity;
        }
        persistent_len = prefix_len - unit_len;
        if (persistent_len < 0) {
            throw std::runtime_error("DotTTS SOAR DiT persistent prefix is invalid");
        }
        if (persistent_len > 0) {
            persistent_sequence.reserve(static_cast<size_t>(2 * persistent_len * hidden_size));
            persistent_sequence.insert(
                persistent_sequence.end(),
                sequence.begin(),
                sequence.begin() + static_cast<std::ptrdiff_t>(persistent_len * hidden_size));
            persistent_sequence.insert(
                persistent_sequence.end(),
                cfg_sequence.begin(),
                cfg_sequence.begin() + static_cast<std::ptrdiff_t>(persistent_len * hidden_size));
        }
        for (int64_t call = 0; call < call_count; ++call) {
            if (decode_state->soar_dit->valid_steps(call) != persistent_len) {
                decode_state->soar_dit->prefill(call, persistent_sequence, persistent_len, modulations, request.runtime_stats);
            }
        }
    }

    auto velocity_for = [&](const std::vector<float> & current_z) {
        auto z_projected = run_coordinate_projection(runtime, current_z, latent_patch_size);
        engine::core::round_f32_to_bf16_in_place(z_projected.values);
        if (use_cached_dit) {
            std::vector<float> tail_sequence;
            tail_sequence.reserve(static_cast<size_t>(4 * unit_len * hidden_size));
            tail_sequence.insert(
                tail_sequence.end(),
                sequence.begin() + static_cast<std::ptrdiff_t>(persistent_len * hidden_size),
                sequence.begin() + static_cast<std::ptrdiff_t>(request.fm_seq_len * hidden_size));
            tail_sequence.insert(tail_sequence.end(), z_projected.values.begin(), z_projected.values.end());
            tail_sequence.insert(
                tail_sequence.end(),
                cfg_sequence.begin() + static_cast<std::ptrdiff_t>(persistent_len * hidden_size),
                cfg_sequence.begin() + static_cast<std::ptrdiff_t>(request.fm_seq_len * hidden_size));
            tail_sequence.insert(tail_sequence.end(), z_projected.values.begin(), z_projected.values.end());
            auto velocity = decode_state->soar_dit->run_step(
                modulation_call,
                tail_sequence,
                persistent_len,
                unit_len,
                modulations,
                request.guidance_scale,
                request.runtime_stats);
            ++modulation_call;
            if (velocity.frames != latent_patch_size ||
                static_cast<int64_t>(velocity.values.size()) != latent_patch_size * latent_dim) {
                throw std::runtime_error("DotTTS SOAR cached velocity shape mismatch");
            }
            engine::core::round_f32_to_bf16_in_place(velocity.values);
            return velocity.values;
        }
        std::vector<float> branch_sequence;
        branch_sequence.reserve(static_cast<size_t>(2 * total_len * hidden_size));
        append_branch_sequence(branch_sequence, sequence, z_projected.values, request.fm_seq_len, hidden_size, latent_patch_size);
        append_branch_sequence(branch_sequence, cfg_sequence, z_projected.values, request.fm_seq_len, hidden_size, latent_patch_size);
        const int64_t modulation_row_start = modulation_call * 2;
        ++modulation_call;
        auto velocity = run_velocity(runtime, 
            branch_sequence,
            total_len,
            {},
            {},
            {},
            2,
            positions,
            attention_mask,
            &modulations,
            modulation_row_start,
            total_len - latent_patch_size,
            latent_patch_size,
            request.runtime_stats);
        engine::core::round_f32_to_bf16_in_place(velocity.values);
        auto combined = cfg_combine_velocity(velocity, latent_patch_size, request.guidance_scale);
        engine::core::round_f32_to_bf16_in_place(combined);
        return combined;
    };

    for (int64_t step = 0; step < request.num_inference_steps; ++step) {
        switch (request.ode_method) {
            case DotsOdeMethod::Euler: {
                auto k1 = velocity_for(z);
                add_scaled(z, k1, dt);
                break;
            }
            case DotsOdeMethod::Midpoint: {
                auto k1 = velocity_for(z);
                auto midpoint = sum_scaled(z, k1, 0.5F * dt);
                auto k2 = velocity_for(midpoint);
                add_scaled(z, k2, dt);
                break;
            }
            case DotsOdeMethod::Rk4: {
                auto k1 = velocity_for(z);
                auto k2 = velocity_for(sum_scaled(z, k1, 0.5F * dt));
                auto k3 = velocity_for(sum_scaled(z, k2, 0.5F * dt));
                auto k4 = velocity_for(sum_scaled(z, k3, dt));
                for (size_t i = 0; i < z.size(); ++i) {
                    z[i] += (dt / 6.0F) * (k1[i] + 2.0F * k2[i] + 2.0F * k3[i] + k4[i]);
                }
                break;
            }
        }
    }
    DotsLatentMatrix out;
    out.frames = latent_patch_size;
    out.dims = latent_dim;
    out.values = std::move(z);
    engine::core::round_f32_to_bf16_in_place(out.values);
    return out;
}


}  // namespace engine::models::dots_tts::detail
