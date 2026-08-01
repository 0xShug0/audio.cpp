#pragma once

#include "engine/models/ace_step/assets.h"
#include "engine/models/ace_step/tokenizer_text.h"
#include "engine/models/ace_step/types.h"
#include "engine/framework/core/execution_context.h"

#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace engine::models::ace_step {

class PlannerWeightsRuntime;
class PrefillGraph;
class DecodeGraph;
class CfgPrefillGraph;
class CfgDecodeGraph;

struct AceStepPlannerPreparedInput {
    std::string formatted_prompt;
    AceStepTokenizedText tokenized_prompt;
};

struct Phase1ConstraintTables {
    std::optional<int32_t> newline_token;
    std::optional<int32_t> backtick_token;
    std::vector<int32_t> forbidden_tokens;
    std::map<std::vector<int32_t>, std::vector<int32_t>> bpm_prefix_map;
    std::map<std::vector<int32_t>, std::vector<int32_t>> duration_prefix_map;
    std::map<std::vector<int32_t>, std::vector<int32_t>> keyscale_prefix_map;
    std::map<std::vector<int32_t>, std::vector<int32_t>> language_prefix_map;
    std::map<std::vector<int32_t>, std::vector<int32_t>> timesig_prefix_map;
};

class AceStepPlannerRuntime {
public:
    struct GenerationConfig {
        int64_t max_prompt_tokens = 4096;
        int64_t max_cot_tokens = 512;
        int64_t max_code_tokens = 4032;
    };

    AceStepPlannerRuntime(
        std::shared_ptr<const AceStepAssets> assets,
        core::ExecutionContext & execution);

    AceStepPlannerRuntime(
        std::shared_ptr<const AceStepAssets> assets,
        core::ExecutionContext & execution,
        assets::TensorStorageType weight_storage_type,
        size_t decode_graph_arena_bytes,
        size_t weight_context_bytes,
        GenerationConfig generation);
    ~AceStepPlannerRuntime();

    AceStepPlannerPreparedInput prepare_prompt(const AceStepRequest & request) const;
    AceStepPlan generate(const AceStepRequest & request, bool generate_audio_codes = true) const;
    void release_graph_workspace() const;
    std::string decode_tokens(const std::vector<int32_t> & token_ids) const;
    AceStepPlan parse_output(const std::string & output_text) const;

private:
    std::shared_ptr<const AceStepAssets> assets_;
    AceStepTextTokenizer tokenizer_;
    GenerationConfig generation_;
    std::shared_ptr<PlannerWeightsRuntime> weights_runtime_;
    std::unique_ptr<core::ExecutionContext> host_planner_prefill_execution_;
    std::shared_ptr<PlannerWeightsRuntime> planner_prefill_weights_runtime_;
    size_t decode_graph_arena_bytes_ = 0;
    mutable std::unique_ptr<PrefillGraph> prefill_graph_;
    mutable std::unique_ptr<DecodeGraph> decode_graph_;
    mutable std::unique_ptr<CfgPrefillGraph> cfg_prefill_graph_;
    mutable std::unique_ptr<CfgDecodeGraph> cfg_decode_graph_;
    std::vector<uint8_t> is_audio_code_token_;
    std::vector<int32_t> phase2_candidate_token_ids_;
    Phase1ConstraintTables phase1_constraints_;
};

}  // namespace engine::models::ace_step
