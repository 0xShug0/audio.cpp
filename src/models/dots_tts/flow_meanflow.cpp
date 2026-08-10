#include "flow_impl.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

namespace engine::models::dots_tts::detail {

class MeanFlowDiTRunnerImpl final : public MeanFlowDiTRunner {
public:
    MeanFlowDiTRunnerImpl(std::shared_ptr<const DotFlowWeights> weights, int64_t ode_index)
        : weights_(std::move(weights)), ode_index_(ode_index) {}

    ~MeanFlowDiTRunnerImpl() {
        release_runtime_graphs();
        release_cache();
    }

    void ensure_cache(int64_t capacity) {
        std::lock_guard<std::mutex> lock(mutex_);
        ensure_cache_locked(capacity);
    }

    int64_t valid_steps() const noexcept {
        return cache_.valid_steps();
    }

    void prefill(
        const std::vector<float> & sequence,
        int64_t steps,
        const DotModulationOutput & modulations,
        DotsFlowRuntimeStats * runtime_stats) {
        std::lock_guard<std::mutex> lock(mutex_);
        const auto & config = weights_->config.dit;
        if (steps < 0 || static_cast<int64_t>(sequence.size()) != steps * config.hidden_size) {
            throw std::runtime_error("DotTTS MeanFlow DiT prefill input shape mismatch");
        }
        if (steps == 0) {
            cache_.retain_prefix(0);
            return;
        }
        if (steps > cache_.cache_steps()) {
            throw std::runtime_error("DotTTS MeanFlow DiT prefill exceeds cache capacity");
        }
        const auto graph_start = Clock::now();
        ensure_prefill_graph_locked(steps, modulations);
        const double graph_ms = engine::debug::elapsed_ms(graph_start);

        const auto upload_start = Clock::now();
        ggml_backend_tensor_set(prefill_sequence_, sequence.data(), 0, sequence.size() * sizeof(float));
        const auto positions = build_position_range(0, steps);
        ggml_backend_tensor_set(prefill_positions_, positions.data(), 0, positions.size() * sizeof(int32_t));
        const int32_t row = static_cast<int32_t>(ode_index_);
        ggml_backend_tensor_set(prefill_modulation_index_, &row, 0, sizeof(int32_t));
        const auto mask = build_prefill_mask_values(steps);
        ggml_backend_tensor_set(prefill_mask_, mask.data(), 0, mask.size() * sizeof(ggml_fp16_t));
        const double upload_ms = engine::debug::elapsed_ms(upload_start);

        cache_.retain_prefix(0);
        const auto compute_start = Clock::now();
        if (core::compute_graph(*weights_->execution_context, prefill_graph_, prefill_plan_, "dots_tts.flow.meanflow.prefill") != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("DotTTS MeanFlow DiT prefill graph compute failed");
        }
        cache_.advance_after_direct_append(steps);
        const double compute_ms = engine::debug::elapsed_ms(compute_start);
        if (runtime_stats != nullptr) {
            runtime_stats->velocity_graph_ms += graph_ms;
            runtime_stats->velocity_input_upload_ms += upload_ms;
            runtime_stats->velocity_compute_ms += compute_ms;
        }
    }

    DotsVelocityOutput run_step(
        const std::vector<float> & tail_sequence,
        int64_t persistent_len,
        int64_t unit_len,
        const DotModulationOutput & modulations,
        DotsFlowRuntimeStats * runtime_stats) {
        std::lock_guard<std::mutex> lock(mutex_);
        const auto & config = weights_->config;
        const int64_t tail_len = 2 * unit_len;
        if (persistent_len < 0 || unit_len <= 0 ||
            persistent_len != cache_.valid_steps() ||
            persistent_len + unit_len > cache_.cache_steps() ||
            static_cast<int64_t>(tail_sequence.size()) != tail_len * config.dit.hidden_size) {
            throw std::runtime_error("DotTTS MeanFlow DiT cached step input shape mismatch");
        }
        const auto graph_start = Clock::now();
        ensure_step_graph_locked(cache_.cache_steps(), tail_len, unit_len, modulations);
        const double graph_ms = engine::debug::elapsed_ms(graph_start);

        const auto upload_start = Clock::now();
        ggml_backend_tensor_set(step_sequence_, tail_sequence.data(), 0, tail_sequence.size() * sizeof(float));
        const auto positions = build_position_range(persistent_len, tail_len);
        ggml_backend_tensor_set(step_positions_, positions.data(), 0, positions.size() * sizeof(int32_t));
        step_slot_values_.resize(static_cast<size_t>(unit_len));
        for (int64_t i = 0; i < unit_len; ++i) {
            step_slot_values_[static_cast<size_t>(i)] = static_cast<int32_t>(persistent_len + i);
        }
        ggml_backend_tensor_set(step_slots_, step_slot_values_.data(), 0, step_slot_values_.size() * sizeof(int32_t));
        const int32_t modulation_row = static_cast<int32_t>(ode_index_);
        ggml_backend_tensor_set(step_modulation_index_, &modulation_row, 0, sizeof(int32_t));
        const auto mask = build_cached_update_mask_values(cache_.cache_steps(), persistent_len, unit_len);
        ggml_backend_tensor_set(step_mask_, mask.data(), 0, mask.size() * sizeof(ggml_fp16_t));
        const double upload_ms = engine::debug::elapsed_ms(upload_start);

        const auto compute_start = Clock::now();
        if (core::compute_graph(*weights_->execution_context, step_graph_, step_plan_, "dots_tts.flow.meanflow.step") != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("DotTTS MeanFlow DiT cached step graph compute failed");
        }
        cache_.advance_after_direct_append(unit_len);
        const double compute_ms = engine::debug::elapsed_ms(compute_start);

        DotsVelocityOutput out;
        out.frames = config.patch_size;
        out.latent_dim = config.latent_dim;
        const auto read_start = Clock::now();
        out.values = core::read_tensor_f32(step_output_);
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
        if (step_graph_ != nullptr) {
            core::release_backend_graph_resources(weights_->execution_context->backend(), step_graph_);
        }
        prefill_gallocr_.reset();
        step_gallocr_.reset();
        prefill_ggml_.reset();
        step_ggml_.reset();
        prefill_plan_.reset();
        step_plan_.reset();
        prefill_graph_ = nullptr;
        step_graph_ = nullptr;
        prefill_sequence_ = nullptr;
        prefill_positions_ = nullptr;
        prefill_modulation_index_ = nullptr;
        prefill_mask_ = nullptr;
        prefill_steps_ = 0;
        prefill_modulation_source_tensor_ = nullptr;
        step_sequence_ = nullptr;
        step_positions_ = nullptr;
        step_slots_ = nullptr;
        step_modulation_index_ = nullptr;
        step_mask_ = nullptr;
        step_output_ = nullptr;
        step_slot_values_.clear();
        step_capacity_ = 0;
        step_tail_len_ = 0;
        step_unit_len_ = 0;
        step_modulation_source_tensor_ = nullptr;
    }

private:
    void ensure_cache_locked(int64_t capacity) {
        if (capacity <= 0) {
            throw std::runtime_error("DotTTS MeanFlow DiT cache requires positive capacity");
        }
        if (cache_ctx_ != nullptr && cache_.cache_steps() >= capacity) {
            return;
        }
        release_runtime_graphs();
        release_cache();
        const auto & config = weights_->config.dit;
        const int64_t dim = head_dim(config);
        ggml_init_params params{kSmallGraphContextBytes, nullptr, true};
        cache_ctx_.reset(ggml_init(params));
        if (cache_ctx_ == nullptr) {
            throw std::runtime_error("failed to initialize DotTTS MeanFlow DiT cache context");
        }
        core::ModuleBuildContext cache_build_ctx{cache_ctx_.get(), "dots_tts.flow.meanflow.cache", weights_->execution_context->backend_type()};
        std::vector<core::TensorValue> keys;
        std::vector<core::TensorValue> values;
        keys.reserve(weights_->blocks.size());
        values.reserve(weights_->blocks.size());
        for (int64_t layer = 0; layer < config.num_layers; ++layer) {
            keys.push_back(core::make_tensor(
                cache_build_ctx,
                GGML_TYPE_F32,
                core::TensorShape::from_dims({1, capacity, config.num_heads, dim})));
            values.push_back(core::make_tensor(
                cache_build_ctx,
                GGML_TYPE_F32,
                core::TensorShape::from_dims({1, capacity, config.num_heads, dim})));
        }
        cache_buffer_ = ggml_backend_alloc_ctx_tensors(cache_ctx_.get(), weights_->execution_context->backend());
        if (cache_buffer_ == nullptr) {
            release_cache();
            throw std::runtime_error("failed to allocate DotTTS MeanFlow DiT cache memory");
        }
        cache_ = runtime::TransformerKVCache(capacity, config.num_heads * dim, std::move(keys), std::move(values));
    }

    void release_cache() {
        cache_ = runtime::TransformerKVCache();
        if (cache_buffer_ != nullptr) {
            ggml_backend_buffer_free(cache_buffer_);
            cache_buffer_ = nullptr;
        }
        cache_ctx_.reset();
    }

    core::TensorValue select_modulation_row(
        core::ModuleBuildContext & ctx,
        const DotModulationOutput & modulations,
        ggml_tensor ** index_tensor) const {
        if (!modulations.backend_value.valid() || ode_index_ < 0 || ode_index_ >= modulations.rows) {
            throw std::runtime_error("DotTTS MeanFlow DiT modulation row is invalid");
        }
        auto index = core::make_tensor(ctx, GGML_TYPE_I32, core::TensorShape::from_dims({1}));
        *index_tensor = index.tensor;
        return core::wrap_tensor(
            ggml_get_rows(
                ctx.ggml,
                core::ensure_backend_addressable_layout(ctx, modulations.backend_value).tensor,
                index.tensor),
            core::TensorShape::from_dims({1, modulations.width}),
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
        prefill_modulation_index_ = nullptr;
        prefill_mask_ = nullptr;
        prefill_steps_ = 0;
        prefill_modulation_source_tensor_ = nullptr;

        ggml_init_params params{kLargeGraphContextBytes, nullptr, true};
        prefill_ggml_.reset(ggml_init(params));
        if (prefill_ggml_ == nullptr) {
            throw std::runtime_error("failed to initialize DotTTS MeanFlow DiT prefill graph context");
        }
        const auto & config = weights_->config.dit;
        const int64_t dim = head_dim(config);
        core::ModuleBuildContext build_ctx{prefill_ggml_.get(), "dots_tts.flow.meanflow.prefill", weights_->execution_context->backend_type()};
        auto sequence = core::make_tensor(build_ctx, GGML_TYPE_F32, core::TensorShape::from_dims({1, steps, config.hidden_size}));
        auto positions = core::make_tensor(build_ctx, GGML_TYPE_I32, core::TensorShape::from_dims({steps}));
        auto mask = core::make_tensor(build_ctx, GGML_TYPE_F16, core::TensorShape::from_dims({1, 1, steps, steps}));
        prefill_sequence_ = sequence.tensor;
        prefill_positions_ = positions.tensor;
        prefill_mask_ = mask.tensor;
        auto modulation = select_modulation_row(build_ctx, modulations, &prefill_modulation_index_);
        auto x = modules::LinearModule({config.hidden_size, config.hidden_size, true}).build(build_ctx, sequence, weights_->input_layer);
        prefill_graph_ = ggml_new_graph_custom(prefill_ggml_.get(), 262144, false);
        for (size_t layer = 0; layer < weights_->blocks.size(); ++layer) {
            auto mods = modules::SliceModule({
                static_cast<int>(modulation.shape.rank - 1),
                static_cast<int64_t>(layer) * 6 * config.hidden_size,
                6 * config.hidden_size,
            }).build(build_ctx, modulation);
            auto block = dit_block_with_mods_and_cache(build_ctx, x, mods, positions, weights_->blocks[layer], config, mask);
            auto key_dest = runtime::view_transformer_kv_cache_steps(
                build_ctx,
                cache_.key_tensor(layer),
                0,
                steps,
                config.num_heads,
                dim,
                "DotTTS MeanFlow DiT prefill key");
            auto value_dest = runtime::view_transformer_kv_cache_steps(
                build_ctx,
                cache_.value_tensor(layer),
                0,
                steps,
                config.num_heads,
                dim,
                "DotTTS MeanFlow DiT prefill value");
            ggml_build_forward_expand(prefill_graph_, ggml_cpy(build_ctx.ggml, block.key.tensor, key_dest.tensor));
            ggml_build_forward_expand(prefill_graph_, ggml_cpy(build_ctx.ggml, block.value.tensor, value_dest.tensor));
            x = block.output;
        }
        core::validate_backend_graph_supported(weights_->execution_context->backend(), prefill_graph_, "dots_tts.flow.meanflow.prefill");
        prefill_gallocr_.reset(ggml_gallocr_new(ggml_backend_get_default_buffer_type(weights_->execution_context->backend())));
        if (prefill_gallocr_ == nullptr ||
            !ggml_gallocr_reserve(prefill_gallocr_.get(), prefill_graph_) ||
            !ggml_gallocr_alloc_graph(prefill_gallocr_.get(), prefill_graph_)) {
            release_runtime_graphs();
            throw std::runtime_error("failed to allocate DotTTS MeanFlow DiT prefill graph memory");
        }
        core::prepare_host_graph_plan(*weights_->execution_context, prefill_graph_, prefill_plan_);
        prefill_steps_ = steps;
        prefill_modulation_source_tensor_ = modulations.backend_value.tensor;
    }

    void ensure_step_graph_locked(
        int64_t capacity,
        int64_t tail_len,
        int64_t unit_len,
        const DotModulationOutput & modulations) {
        if (step_ggml_ != nullptr && step_capacity_ == capacity && step_tail_len_ == tail_len &&
            step_unit_len_ == unit_len && step_modulation_source_tensor_ == modulations.backend_value.tensor) {
            return;
        }
        if (step_graph_ != nullptr) {
            core::release_backend_graph_resources(weights_->execution_context->backend(), step_graph_);
        }
        step_gallocr_.reset();
        step_ggml_.reset();
        step_plan_.reset();
        step_graph_ = nullptr;
        step_sequence_ = nullptr;
        step_positions_ = nullptr;
        step_slots_ = nullptr;
        step_modulation_index_ = nullptr;
        step_mask_ = nullptr;
        step_output_ = nullptr;
        step_slot_values_.clear();
        step_capacity_ = 0;
        step_tail_len_ = 0;
        step_unit_len_ = 0;
        step_modulation_source_tensor_ = nullptr;

        ggml_init_params params{kLargeGraphContextBytes, nullptr, true};
        step_ggml_.reset(ggml_init(params));
        if (step_ggml_ == nullptr) {
            throw std::runtime_error("failed to initialize DotTTS MeanFlow DiT cached step graph context");
        }
        const auto & config = weights_->config.dit;
        core::ModuleBuildContext build_ctx{step_ggml_.get(), "dots_tts.flow.meanflow.step", weights_->execution_context->backend_type()};
        auto sequence = core::make_tensor(build_ctx, GGML_TYPE_F32, core::TensorShape::from_dims({1, tail_len, config.hidden_size}));
        auto positions = core::make_tensor(build_ctx, GGML_TYPE_I32, core::TensorShape::from_dims({tail_len}));
        auto slots = core::make_tensor(build_ctx, GGML_TYPE_I32, core::TensorShape::from_dims({unit_len}));
        auto mask = core::make_tensor(build_ctx, GGML_TYPE_F16, core::TensorShape::from_dims({1, 1, tail_len, capacity + tail_len}));
        step_sequence_ = sequence.tensor;
        step_positions_ = positions.tensor;
        step_slots_ = slots.tensor;
        step_mask_ = mask.tensor;
        auto modulation = select_modulation_row(build_ctx, modulations, &step_modulation_index_);
        auto x = modules::LinearModule({config.hidden_size, config.hidden_size, true}).build(build_ctx, sequence, weights_->input_layer);
        step_graph_ = ggml_new_graph_custom(step_ggml_.get(), 262144, false);
        const modules::FastKVSetRowsModule set_rows({modules::FastKVSetRowsMode::BackendViewOptimized});
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
                cache_.key_tensor(layer),
                cache_.value_tensor(layer));
            auto updated_key = cache_.key_tensor(layer);
            auto updated_value = cache_.value_tensor(layer);
            for (int64_t row = 0; row < unit_len; ++row) {
                auto slot = modules::SliceModule({0, row, 1}).build(build_ctx, slots);
                updated_key = set_rows.build(
                    build_ctx,
                    updated_key,
                    modules::SliceModule({1, row, 1}).build(build_ctx, block.key),
                    slot);
                updated_value = set_rows.build(
                    build_ctx,
                    updated_value,
                    modules::SliceModule({1, row, 1}).build(build_ctx, block.value),
                    slot);
            }
            ggml_build_forward_expand(step_graph_, updated_key.tensor);
            ggml_build_forward_expand(step_graph_, updated_value.tensor);
            x = block.output;
        }
        x = modules::SliceModule({1, unit_len + 1, weights_->config.patch_size}).build(build_ctx, x);
        auto final_mods = modules::SliceModule({
            static_cast<int>(modulation.shape.rank - 1),
            static_cast<int64_t>(weights_->blocks.size()) * 6 * config.hidden_size,
            2 * config.hidden_size,
        }).build(build_ctx, modulation);
        auto output = final_projection_with_mods(build_ctx, x, final_mods, *weights_);
        step_output_ = output.tensor;
        ggml_build_forward_expand(step_graph_, step_output_);
        core::validate_backend_graph_supported(weights_->execution_context->backend(), step_graph_, "dots_tts.flow.meanflow.step");
        step_gallocr_.reset(ggml_gallocr_new(ggml_backend_get_default_buffer_type(weights_->execution_context->backend())));
        if (step_gallocr_ == nullptr ||
            !ggml_gallocr_reserve(step_gallocr_.get(), step_graph_) ||
            !ggml_gallocr_alloc_graph(step_gallocr_.get(), step_graph_)) {
            release_runtime_graphs();
            throw std::runtime_error("failed to allocate DotTTS MeanFlow DiT cached step graph memory");
        }
        core::prepare_host_graph_plan(*weights_->execution_context, step_graph_, step_plan_);
        step_capacity_ = capacity;
        step_tail_len_ = tail_len;
        step_unit_len_ = unit_len;
        step_modulation_source_tensor_ = modulations.backend_value.tensor;
    }

    std::shared_ptr<const DotFlowWeights> weights_;
    int64_t ode_index_ = 0;
    mutable std::mutex mutex_;
    std::unique_ptr<ggml_context, GgmlContextDeleter> cache_ctx_;
    ggml_backend_buffer_t cache_buffer_ = nullptr;
    runtime::TransformerKVCache cache_;

    std::unique_ptr<ggml_context, GgmlContextDeleter> prefill_ggml_;
    std::unique_ptr<std::remove_pointer_t<ggml_gallocr_t>, GgmlGallocrDeleter> prefill_gallocr_;
    ggml_cgraph * prefill_graph_ = nullptr;
    core::HostGraphPlan prefill_plan_;
    ggml_tensor * prefill_sequence_ = nullptr;
    ggml_tensor * prefill_positions_ = nullptr;
    ggml_tensor * prefill_modulation_index_ = nullptr;
    ggml_tensor * prefill_mask_ = nullptr;
    int64_t prefill_steps_ = 0;
    ggml_tensor * prefill_modulation_source_tensor_ = nullptr;

    std::unique_ptr<ggml_context, GgmlContextDeleter> step_ggml_;
    std::unique_ptr<std::remove_pointer_t<ggml_gallocr_t>, GgmlGallocrDeleter> step_gallocr_;
    ggml_cgraph * step_graph_ = nullptr;
    core::HostGraphPlan step_plan_;
    ggml_tensor * step_sequence_ = nullptr;
    ggml_tensor * step_positions_ = nullptr;
    ggml_tensor * step_slots_ = nullptr;
    ggml_tensor * step_modulation_index_ = nullptr;
    ggml_tensor * step_mask_ = nullptr;
    ggml_tensor * step_output_ = nullptr;
    std::vector<int32_t> step_slot_values_;
    int64_t step_capacity_ = 0;
    int64_t step_tail_len_ = 0;
    int64_t step_unit_len_ = 0;
    ggml_tensor * step_modulation_source_tensor_ = nullptr;
};


std::unique_ptr<MeanFlowDiTRunner> make_meanflow_dit_runner(std::shared_ptr<const DotFlowWeights> weights, int64_t ode_index) {
    return std::make_unique<MeanFlowDiTRunnerImpl>(std::move(weights), ode_index);
}

DotsLatentMatrix decode_next_meanflow(DotFlowSharedRuntime & runtime, const DotsFlowDecodeRequest & request, FlowDecodeCacheState * decode_state) {
    const auto & config = runtime.weights->config;
    if (!config.meanflow.has_value() || !config.meanflow->enabled) {
        throw std::runtime_error("DotTTS MeanFlow decode requested for a non-MeanFlow checkpoint");
    }
    const int64_t hidden_size = config.dit.hidden_size;
    const int64_t latent_patch_size = config.patch_size;
    const int64_t latent_dim = config.latent_dim;
    const int64_t total_len = request.fm_seq_len + latent_patch_size;
    if (request.sequence == nullptr || request.speaker_condition == nullptr) {
        throw std::runtime_error("DotTTS MeanFlow decode request is missing borrowed inputs");
    }
    const auto & sequence = *request.sequence;
    const auto & speaker_condition = *request.speaker_condition;
    if (request.fm_seq_len <= 0 || request.num_inference_steps <= 0 ||
        static_cast<int64_t>(sequence.size()) < request.fm_seq_len * hidden_size ||
        static_cast<int64_t>(speaker_condition.size()) != hidden_size) {
        throw std::runtime_error("DotTTS MeanFlow decode request shape mismatch");
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
    const auto attention_mask = build_decode_mask(1, total_len, request.fm_seq_len, latent_patch_size, 1);
    const float dt = 1.0F / static_cast<float>(request.num_inference_steps);
    const bool use_duration_embedding = config.meanflow->use_duration_embedding;
    std::vector<float> modulation_timesteps;
    std::vector<float> modulation_durations;
    std::vector<float> modulation_speaker_condition;
    modulation_timesteps.reserve(static_cast<size_t>(request.num_inference_steps));
    if (use_duration_embedding) {
        modulation_durations.reserve(static_cast<size_t>(request.num_inference_steps));
    }
    modulation_speaker_condition.reserve(static_cast<size_t>(request.num_inference_steps * hidden_size));
    for (int64_t step = 0; step < request.num_inference_steps; ++step) {
        modulation_timesteps.push_back(static_cast<float>(step) * dt);
        if (use_duration_embedding) {
            modulation_durations.push_back(dt);
        }
        modulation_speaker_condition.insert(modulation_speaker_condition.end(), speaker_condition.begin(), speaker_condition.end());
    }
    DotModulationOutput modulations;
        if (decode_state != nullptr &&
        decode_state->meanflow.has_value() &&
        decode_state->meanflow->num_inference_steps == request.num_inference_steps &&
        decode_state->meanflow->ode_method == request.ode_method &&
        decode_state->meanflow->use_duration_embedding == use_duration_embedding &&
        decode_state->meanflow->speaker_condition == speaker_condition &&
        decode_state->meanflow->output.rows == static_cast<int64_t>(modulation_timesteps.size())) {
        modulations = decode_state->meanflow->output;
    } else {
        modulations = run_modulation(runtime, 
            modulation_timesteps,
            modulation_durations,
            modulation_speaker_condition,
            request.runtime_stats);
        if (decode_state != nullptr) {
            decode_state->soar.reset();
            decode_state->soar_dit.reset();
            decode_state->soar_dit_calls = 0;
            decode_state->soar_dit_capacity = 0;
            decode_state->meanflow_dit.clear();
            decode_state->meanflow_dit_steps = 0;
            decode_state->meanflow_dit_capacity = 0;
            decode_state->meanflow = FlowDecodeCacheState::CachedModulations{
                modulations,
                speaker_condition,
                request.num_inference_steps,
                request.ode_method,
                use_duration_embedding,
            };
        }
    }
    const int64_t hidden_patch_size = 1;
    const int64_t unit_len = hidden_patch_size + latent_patch_size;
    const int64_t prefix_len = request.fm_seq_len - hidden_patch_size;
    if (prefix_len < 0 || prefix_len % unit_len != 0) {
        throw std::runtime_error("DotTTS MeanFlow DiT expects unit-aligned finalized history");
    }
    const bool use_cached_dit = request.use_cached_dit && decode_state != nullptr && prefix_len >= unit_len;
    if (use_cached_dit) {
        if (request.cache_capacity < prefix_len) {
            throw std::runtime_error("DotTTS MeanFlow DiT cache capacity is smaller than the current prefix");
        }
        const int64_t capacity = resolve_dit_cache_capacity_tokens(request.fm_seq_len, unit_len);
        if (decode_state->meanflow_dit_steps != request.num_inference_steps ||
            decode_state->meanflow_dit_capacity < capacity ||
            static_cast<int64_t>(decode_state->meanflow_dit.size()) != request.num_inference_steps) {
            decode_state->meanflow_dit.clear();
            decode_state->meanflow_dit.reserve(static_cast<size_t>(request.num_inference_steps));
            for (int64_t step = 0; step < request.num_inference_steps; ++step) {
                decode_state->meanflow_dit.push_back(make_meanflow_dit_runner(runtime.weights, step));
                decode_state->meanflow_dit.back()->ensure_cache(capacity);
            }
            decode_state->meanflow_dit_steps = request.num_inference_steps;
            decode_state->meanflow_dit_capacity = capacity;
        }
        const int64_t persistent_len = prefix_len - unit_len;
        if (persistent_len < 0) {
            throw std::runtime_error("DotTTS MeanFlow DiT persistent prefix is invalid");
        }
        std::vector<float> persistent_sequence;
        if (persistent_len > 0) {
            persistent_sequence.assign(
                sequence.begin(),
                sequence.begin() + static_cast<std::ptrdiff_t>(persistent_len * hidden_size));
        }
        for (auto & runner : decode_state->meanflow_dit) {
            if (runner->valid_steps() != persistent_len) {
                runner->prefill(persistent_sequence, persistent_len, modulations, request.runtime_stats);
            }
        }
        for (int64_t step = 0; step < request.num_inference_steps; ++step) {
            auto z_projected = run_coordinate_projection(runtime, z, latent_patch_size);
            engine::core::round_f32_to_bf16_in_place(z_projected.values);
            std::vector<float> tail_sequence;
            tail_sequence.reserve(static_cast<size_t>(2 * unit_len * hidden_size));
            tail_sequence.insert(
                tail_sequence.end(),
                sequence.begin() + static_cast<std::ptrdiff_t>(persistent_len * hidden_size),
                sequence.begin() + static_cast<std::ptrdiff_t>(request.fm_seq_len * hidden_size));
            tail_sequence.insert(tail_sequence.end(), z_projected.values.begin(), z_projected.values.end());
            auto velocity = decode_state->meanflow_dit[static_cast<size_t>(step)]->run_step(
                tail_sequence,
                persistent_len,
                unit_len,
                modulations,
                request.runtime_stats);
            if (velocity.frames != latent_patch_size ||
                static_cast<int64_t>(velocity.values.size()) != latent_patch_size * latent_dim) {
                throw std::runtime_error("DotTTS MeanFlow cached velocity shape mismatch");
            }
            engine::core::round_f32_to_bf16_in_place(velocity.values);
            for (int64_t frame = 0; frame < latent_patch_size; ++frame) {
                for (int64_t dim = 0; dim < latent_dim; ++dim) {
                    const size_t out_index = static_cast<size_t>(frame * latent_dim + dim);
                    z[out_index] += velocity.values[out_index] * dt;
                }
            }
            engine::core::round_f32_to_bf16_in_place(z);
        }
    } else {
        for (int64_t step = 0; step < request.num_inference_steps; ++step) {
            auto z_projected = run_coordinate_projection(runtime, z, latent_patch_size);
            engine::core::round_f32_to_bf16_in_place(z_projected.values);
            std::vector<float> branch_sequence;
            branch_sequence.reserve(static_cast<size_t>(total_len * hidden_size));
            append_branch_sequence(branch_sequence, sequence, z_projected.values, request.fm_seq_len, hidden_size, latent_patch_size);
            auto velocity = run_velocity(runtime, 
                branch_sequence,
                total_len,
                {},
                {},
                {},
                1,
                positions,
                attention_mask,
                &modulations,
                step,
                total_len - latent_patch_size,
                latent_patch_size,
                request.runtime_stats);
            if (velocity.frames != latent_patch_size ||
                static_cast<int64_t>(velocity.values.size()) != latent_patch_size * latent_dim) {
                throw std::runtime_error("DotTTS MeanFlow velocity shape mismatch");
            }
            engine::core::round_f32_to_bf16_in_place(velocity.values);
            for (int64_t frame = 0; frame < latent_patch_size; ++frame) {
                for (int64_t dim = 0; dim < latent_dim; ++dim) {
                    const size_t out_index = static_cast<size_t>(frame * latent_dim + dim);
                    z[out_index] += velocity.values[out_index] * dt;
                }
            }
            engine::core::round_f32_to_bf16_in_place(z);
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
