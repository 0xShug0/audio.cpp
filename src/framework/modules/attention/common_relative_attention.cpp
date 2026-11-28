#include "attention_internal.h"

namespace engine::modules::attention::internal {

namespace {

bool use_specialized_flash_attention(
    const RelativeAttentionConfig & config,
    const std::optional<core::TensorValue> & attention_mask,
    const std::optional<core::TensorValue> & projected_pos_emb) {
    // Keep the production path on the reference implementation until callers
    // explicitly opt in. The flash-attention path is only valid for offline
    // full-context attention with a precomputed projected positional embedding.
    return config.use_flash_attention &&
           config.left_context < 0 &&
           config.right_context < 0 &&
           attention_mask.has_value() &&
           projected_pos_emb.has_value();
}

}  // namespace

core::TensorValue build_global_relative_attention_impl(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & query,
    const core::TensorValue & key_value,
    const std::optional<core::TensorValue> & pos_emb,
    const RelativeAttentionConfig & config,
    const RelativeAttentionWeights & weights,
    const std::optional<core::TensorValue> & attention_mask,
    const std::optional<core::TensorValue> & query_keep_mask,
    const std::optional<core::TensorValue> & projected_pos_emb) {
    validate_sequence_input(query, config.hidden_size, "query");
    validate_sequence_input(key_value, config.hidden_size, "key_value");

    const int64_t head_dim = config.hidden_size / config.num_heads;
    const int64_t key_frames = key_value.shape.dims[1];
    const int64_t required_pos_frames = config.legacy_position_layout ? key_frames : 2 * key_frames - 1;
    if (projected_pos_emb.has_value()) {
        core::validate_rank_between(*projected_pos_emb, 3, 3, "projected_pos_emb");
        core::validate_last_dim(*projected_pos_emb, config.hidden_size, "projected_pos_emb");
        if (projected_pos_emb->shape.dims[1] != required_pos_frames) {
            throw std::runtime_error(config.legacy_position_layout ? "Legacy relative attention expects projected_pos_emb frames == key_frames" : "Relative attention expects projected_pos_emb frames == 2 * key_frames - 1");
        }
    } else {
        if (!pos_emb.has_value()) {
            throw std::runtime_error("Relative attention requires pos_emb when projected_pos_emb is not provided");
        }
        core::validate_rank_between(*pos_emb, 3, 3, "pos_emb");
        core::validate_last_dim(*pos_emb, config.hidden_size, "pos_emb");
        if (pos_emb->shape.dims[1] != required_pos_frames) {
            throw std::runtime_error(config.legacy_position_layout ? "Legacy relative attention expects pos_emb frames == key_frames" : "Relative attention expects pos_emb frames == 2 * key_frames - 1");
        }
    }

    const LinearModule q_proj({config.hidden_size, config.hidden_size, config.use_bias});
    const LinearModule k_proj({config.hidden_size, config.hidden_size, config.use_bias});
    const LinearModule v_proj({config.hidden_size, config.hidden_size, config.use_bias});
    const LinearModule qkv_proj({config.hidden_size, config.hidden_size * 3, config.use_bias});
    const LinearModule p_proj({config.hidden_size, config.hidden_size, false});
    const LinearModule out_proj({config.hidden_size, config.hidden_size, config.use_bias});
    const MatMulModule matmul;
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    core::TensorValue q;
    core::TensorValue k;
    core::TensorValue v;
    const bool can_fuse_qkv =
        weights.attention.qkv_weight.has_value() &&
        (!config.use_bias || weights.attention.qkv_bias.has_value()) &&
        query.tensor == key_value.tensor;
    if (can_fuse_qkv) {
        auto qkv = qkv_proj.build(ctx, query, LinearWeights{*weights.attention.qkv_weight, weights.attention.qkv_bias});
        q = SliceModule({2, 0, config.hidden_size}).build(ctx, qkv);
        k = SliceModule({2, config.hidden_size, config.hidden_size}).build(ctx, qkv);
        v = SliceModule({2, 2 * config.hidden_size, config.hidden_size}).build(ctx, qkv);
        q = ensure_contiguous_layout(ctx, q);
        k = ensure_contiguous_layout(ctx, k);
        v = ensure_contiguous_layout(ctx, v);
    } else {
        q = q_proj.build(ctx, query, LinearWeights{weights.attention.q_weight, weights.attention.q_bias});
        k = k_proj.build(ctx, key_value, LinearWeights{weights.attention.k_weight, weights.attention.k_bias});
        v = v_proj.build(ctx, key_value, LinearWeights{weights.attention.v_weight, weights.attention.v_bias});
    }
    auto p = projected_pos_emb.has_value()
        ? *projected_pos_emb
        : p_proj.build(ctx, *pos_emb, LinearWeights{weights.pos_weight, std::nullopt});

    q = reshape_heads(ctx, q, config.num_heads, head_dim);
    k = reshape_heads(ctx, k, config.num_heads, head_dim);
    v = reshape_heads(ctx, v, config.num_heads, head_dim);
    p = reshape_heads(ctx, p, config.num_heads, head_dim);

    auto q_heads = permute_tensor(ctx, q, {0, 2, 1, 3});
    auto k_heads = permute_tensor(ctx, k, {0, 2, 1, 3});
    auto v_heads = permute_tensor(ctx, v, {0, 2, 1, 3});
    auto p_heads = permute_tensor(ctx, p, {0, 2, 1, 3});
    if (p_heads.shape.dims[0] == 1 && q_heads.shape.dims[0] > 1) {
        p_heads = RepeatModule(RepeatConfig{core::TensorShape::from_dims(
            {q_heads.shape.dims[0], p_heads.shape.dims[1], p_heads.shape.dims[2], p_heads.shape.dims[3]})}).build(ctx, p_heads);
    }

    auto q_u = add_attention_bias(ctx, q_heads, weights.pos_bias_u, config.num_heads, head_dim);
    auto q_v = add_attention_bias(ctx, q_heads, weights.pos_bias_v, config.num_heads, head_dim);

    const bool specialized_flash = use_specialized_flash_attention(config, attention_mask, projected_pos_emb);
    std::optional<core::TensorValue> k_transposed;
    if (!specialized_flash) {
        k_transposed = permute_tensor(ctx, k_heads, {0, 1, 3, 2});
    }
    auto p_transposed = permute_tensor(ctx, p_heads, {0, 1, 3, 2});

    std::optional<core::TensorValue> matrix_ac;
    if (!specialized_flash) {
        matrix_ac = matmul.build(ctx, q_u, *k_transposed);
    }
    auto matrix_bd = matmul.build(ctx, q_v, p_transposed);
    matrix_bd = relative_shift(ctx, matrix_bd);
    matrix_bd = SliceModule({3, 0, key_frames}).build(ctx, matrix_bd);
    core::TensorValue context;
    if (specialized_flash) {
        auto q_flash = ensure_contiguous_layout(ctx, q_u);
        auto k_flash = ensure_contiguous_layout(ctx, k_heads);
        auto v_flash = ensure_contiguous_layout(ctx, v_heads);

        ggml_tensor * flash = ggml_flash_attn_ext_with_bias_mask(
            ctx.ggml,
            q_flash.tensor,
            k_flash.tensor,
            v_flash.tensor,
            ensure_contiguous_layout(ctx, matrix_bd).tensor,
            attention_mask->tensor,
            scale,
            0.0f,
            0.0f);
        ggml_flash_attn_ext_set_prec(flash, GGML_PREC_F32);
        context = core::wrap_tensor(
            flash,
            core::TensorShape::from_dims({query.shape.dims[0], query.shape.dims[1], config.num_heads, head_dim}),
            GGML_TYPE_F32);
    } else {
        auto scores = core::wrap_tensor(ggml_add(ctx.ggml, matrix_ac->tensor, matrix_bd.tensor), matrix_bd.shape, GGML_TYPE_F32);
        core::TensorValue attn;
        if (attention_mask.has_value()) {
            attn = core::wrap_tensor(
                ggml_soft_max_ext(ctx.ggml, ensure_contiguous_layout(ctx, scores).tensor, attention_mask->tensor, scale, 0.0f),
                scores.shape,
                GGML_TYPE_F32);
        } else {
        scores = core::wrap_tensor(ggml_scale(ctx.ggml, scores.tensor, scale), scores.shape, GGML_TYPE_F32);
        attn = core::wrap_tensor(ggml_soft_max(ctx.ggml, ensure_contiguous_layout(ctx, scores).tensor), scores.shape, GGML_TYPE_F32);
        }
        context = matmul.build(ctx, attn, v_heads);
    }
    if (!specialized_flash) {
        context = permute_tensor(ctx, context, {0, 2, 1, 3});
    }
    context = ensure_contiguous_layout(ctx, context);
    context = core::reshape_tensor(ctx, context, core::TensorShape::from_dims({query.shape.dims[0], query.shape.dims[1], config.hidden_size}));
    if (query_keep_mask.has_value()) {
        context = MaskingModule().build(ctx, context, *query_keep_mask);
    }
    return out_proj.build(ctx, context, LinearWeights{weights.attention.out_weight, weights.attention.out_bias});
}

core::TensorValue build_windowed_relative_attention_impl(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & query,
    const core::TensorValue & key_value,
    const core::TensorValue & pos_emb,
    const RelativeAttentionConfig & config,
    const RelativeAttentionWeights & weights,
    const std::optional<core::TensorValue> & attention_mask,
    const std::optional<core::TensorValue> & query_keep_mask) {
    if (!attention_mask.has_value()) {
        throw std::runtime_error("Local relative attention requires an explicit attention_mask");
    }

    validate_sequence_input(query, config.hidden_size, "query");
    validate_sequence_input(key_value, config.hidden_size, "key_value");
    core::validate_rank_between(pos_emb, 3, 3, "pos_emb");
    core::validate_last_dim(pos_emb, config.hidden_size, "pos_emb");

    const int64_t left = config.left_context;
    const int64_t right = config.right_context;
    const int64_t pos_frames = left + right + 1;
    const int64_t w = std::max(left, right);
    const int64_t local_frames = 2 * w + 1;
    const int64_t query_frames = query.shape.dims[1];
    const int64_t key_frames = key_value.shape.dims[1];
    if (key_frames < query_frames) {
        throw std::runtime_error("Local relative attention requires key/value frames >= query frames");
    }
    const int64_t cache_prefix = key_frames - query_frames;
    if (pos_emb.shape.dims[1] != pos_frames) {
        throw std::runtime_error("Local relative attention expects pos_emb frames == left_context + right_context + 1");
    }

    const LinearModule q_proj({config.hidden_size, config.hidden_size, config.use_bias});
    const LinearModule k_proj({config.hidden_size, config.hidden_size, config.use_bias});
    const LinearModule v_proj({config.hidden_size, config.hidden_size, config.use_bias});
    const LinearModule p_proj({config.hidden_size, config.hidden_size, false});
    const LinearModule out_proj({config.hidden_size, config.hidden_size, config.use_bias});
    const MulModule mul;
    const AddModule add;
    const int64_t head_dim = config.hidden_size / config.num_heads;
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    auto q = q_proj.build(ctx, query, LinearWeights{weights.attention.q_weight, weights.attention.q_bias});
    auto k = k_proj.build(ctx, key_value, LinearWeights{weights.attention.k_weight, weights.attention.k_bias});
    auto v = v_proj.build(ctx, key_value, LinearWeights{weights.attention.v_weight, weights.attention.v_bias});
    auto p = p_proj.build(ctx, pos_emb, LinearWeights{weights.pos_weight, std::nullopt});

    q = reshape_heads(ctx, q, config.num_heads, head_dim);
    k = reshape_heads(ctx, k, config.num_heads, head_dim);
    v = reshape_heads(ctx, v, config.num_heads, head_dim);
    p = reshape_heads(ctx, p, config.num_heads, head_dim);

    auto q_heads = permute_tensor(ctx, q, {0, 2, 1, 3});
    auto k_heads = permute_tensor(ctx, k, {0, 2, 1, 3});
    auto v_heads = permute_tensor(ctx, v, {0, 2, 1, 3});
    auto p_heads = permute_tensor(ctx, p, {0, 2, 1, 3});
    if (p_heads.shape.dims[0] == 1 && q_heads.shape.dims[0] > 1) {
        p_heads = RepeatModule(RepeatConfig{core::TensorShape::from_dims(
            {q_heads.shape.dims[0], p_heads.shape.dims[1], p_heads.shape.dims[2], p_heads.shape.dims[3]})}).build(ctx, p_heads);
    }

    auto q_u = add_attention_bias(ctx, q_heads, weights.pos_bias_u, config.num_heads, head_dim);
    auto q_v = add_attention_bias(ctx, q_heads, weights.pos_bias_v, config.num_heads, head_dim);

    std::vector<core::TensorValue> ac_columns;
    std::vector<core::TensorValue> value_columns;
    ac_columns.reserve(static_cast<size_t>(local_frames));
    value_columns.reserve(static_cast<size_t>(local_frames));

    for (int64_t rel = -w; rel <= w; ++rel) {
        const int64_t q_start = std::max<int64_t>(0, -cache_prefix - rel);
        const int64_t q_end = std::min<int64_t>(query_frames, query_frames - std::max<int64_t>(rel, 0));
        const int64_t overlap = q_end - q_start;
        core::TensorValue padded_ac;
        core::TensorValue padded_value;
        if (overlap <= 0) {
            auto q_template = SliceModule({2, 0, 1}).build(ctx, q_u);
            auto k_template = SliceModule({2, 0, 1}).build(ctx, k_heads);
            auto v_template = SliceModule({2, 0, 1}).build(ctx, v_heads);
            auto ac_prod = mul.build(ctx, q_template, k_template);
            core::TensorShape reduced_shape = ac_prod.shape;
            reduced_shape.dims[3] = 1;
            auto ac = core::wrap_tensor(ggml_sum_rows(ctx.ggml, ac_prod.tensor), reduced_shape, GGML_TYPE_F32);
            padded_ac = RepeatModule(RepeatConfig{core::TensorShape::from_dims(
                {ac.shape.dims[0], ac.shape.dims[1], query_frames, ac.shape.dims[3]})}).build(ctx, ac);
            padded_ac = core::wrap_tensor(ggml_scale(ctx.ggml, padded_ac.tensor, 0.0f), padded_ac.shape, GGML_TYPE_F32);
            auto value_column = core::reshape_tensor(
                ctx,
                ensure_contiguous_layout(ctx, v_template),
                core::TensorShape::from_dims({v_template.shape.dims[0], v_template.shape.dims[1], 1, head_dim}));
            padded_value = RepeatModule(RepeatConfig{core::TensorShape::from_dims(
                {value_column.shape.dims[0], value_column.shape.dims[1], query_frames, value_column.shape.dims[3]})}).build(ctx, value_column);
            padded_value = core::wrap_tensor(ggml_scale(ctx.ggml, padded_value.tensor, 0.0f), padded_value.shape, GGML_TYPE_F32);
            ac_columns.push_back(padded_ac);
            value_columns.push_back(padded_value);
            continue;
        }
        const int64_t k_start = cache_prefix + q_start + rel;

        auto q_u_slice = SliceModule({2, q_start, overlap}).build(ctx, q_u);
        auto k_slice = SliceModule({2, k_start, overlap}).build(ctx, k_heads);
        auto v_slice = SliceModule({2, k_start, overlap}).build(ctx, v_heads);

        auto ac_prod = mul.build(ctx, q_u_slice, k_slice);
        core::TensorShape reduced_shape = ac_prod.shape;
        reduced_shape.dims[3] = 1;
        auto ac = core::wrap_tensor(ggml_sum_rows(ctx.ggml, ac_prod.tensor), reduced_shape, GGML_TYPE_F32);

        auto value_column = core::reshape_tensor(
            ctx,
            ensure_contiguous_layout(ctx, v_slice),
            core::TensorShape::from_dims({v_slice.shape.dims[0], v_slice.shape.dims[1], overlap, head_dim}));
        auto zero_template = SliceModule({2, 0, 1}).build(ctx, ac);

        padded_ac = ac;
        padded_value = value_column;
        if (q_start > 0) {
            auto zeros = RepeatModule(RepeatConfig{core::TensorShape::from_dims(
                {zero_template.shape.dims[0], zero_template.shape.dims[1], q_start, 1})}).build(ctx, zero_template);
            zeros = core::wrap_tensor(ggml_scale(ctx.ggml, zeros.tensor, 0.0f), zeros.shape, GGML_TYPE_F32);
            auto value_zeros = RepeatModule(RepeatConfig{core::TensorShape::from_dims(
                {value_column.shape.dims[0], value_column.shape.dims[1], q_start, value_column.shape.dims[3]})}).build(ctx, SliceModule({2, 0, 1}).build(ctx, value_column));
            value_zeros = core::wrap_tensor(ggml_scale(ctx.ggml, value_zeros.tensor, 0.0f), value_zeros.shape, GGML_TYPE_F32);
            padded_ac = ConcatModule({2}).build(ctx, zeros, padded_ac);
            padded_value = ConcatModule({2}).build(ctx, value_zeros, padded_value);
        }
        const int64_t tail = query_frames - q_start - overlap;
        if (tail > 0) {
            auto zeros = RepeatModule(RepeatConfig{core::TensorShape::from_dims(
                {zero_template.shape.dims[0], zero_template.shape.dims[1], tail, 1})}).build(ctx, zero_template);
            zeros = core::wrap_tensor(ggml_scale(ctx.ggml, zeros.tensor, 0.0f), zeros.shape, GGML_TYPE_F32);
            auto value_zeros = RepeatModule(RepeatConfig{core::TensorShape::from_dims(
                {value_column.shape.dims[0], value_column.shape.dims[1], tail, value_column.shape.dims[3]})}).build(ctx, SliceModule({2, 0, 1}).build(ctx, value_column));
            value_zeros = core::wrap_tensor(ggml_scale(ctx.ggml, value_zeros.tensor, 0.0f), value_zeros.shape, GGML_TYPE_F32);
            padded_ac = ConcatModule({2}).build(ctx, padded_ac, zeros);
            padded_value = ConcatModule({2}).build(ctx, padded_value, value_zeros);
        }

        ac_columns.push_back(padded_ac);
        value_columns.push_back(padded_value);
    }

    std::vector<core::TensorValue> score_columns = ac_columns;
    for (int64_t pos_idx = 0; pos_idx < left; ++pos_idx) {
        auto p_slice = SliceModule({2, pos_idx, 1}).build(ctx, p_heads);
        auto p_repeat = RepeatModule(RepeatConfig{core::TensorShape::from_dims(
            {p_slice.shape.dims[0], p_slice.shape.dims[1], query_frames, p_slice.shape.dims[3]})}).build(ctx, p_slice);
        auto bd_prod = mul.build(ctx, q_v, p_repeat);
        core::TensorShape reduced_shape = bd_prod.shape;
        reduced_shape.dims[3] = 1;
        auto bd = core::wrap_tensor(ggml_sum_rows(ctx.ggml, bd_prod.tensor), reduced_shape, GGML_TYPE_F32);
        score_columns[static_cast<size_t>(pos_idx)] = add.build(ctx, score_columns[static_cast<size_t>(pos_idx)], bd);
    }
    for (int64_t pos_idx = left; pos_idx < pos_frames; ++pos_idx) {
        auto p_slice = SliceModule({2, pos_idx, 1}).build(ctx, p_heads);
        auto p_repeat = RepeatModule(RepeatConfig{core::TensorShape::from_dims(
            {p_slice.shape.dims[0], p_slice.shape.dims[1], query_frames, p_slice.shape.dims[3]})}).build(ctx, p_slice);
        auto bd_prod = mul.build(ctx, q_v, p_repeat);
        core::TensorShape reduced_shape = bd_prod.shape;
        reduced_shape.dims[3] = 1;
        auto bd = core::wrap_tensor(ggml_sum_rows(ctx.ggml, bd_prod.tensor), reduced_shape, GGML_TYPE_F32);
        const int64_t target = local_frames - (right + 1) + (pos_idx - left);
        score_columns[static_cast<size_t>(target)] = add.build(ctx, score_columns[static_cast<size_t>(target)], bd);
    }

    auto scores = concat_all(ctx, score_columns, 3);

    auto attn = core::wrap_tensor(
        ggml_soft_max_ext(ctx.ggml, ensure_contiguous_layout(ctx, scores).tensor, attention_mask->tensor, scale, 0.0f),
        scores.shape,
        GGML_TYPE_F32);

    std::vector<core::TensorValue> weighted_columns;
    weighted_columns.reserve(value_columns.size());
    for (size_t i = 0; i < value_columns.size(); ++i) {
        auto attn_col = SliceModule({3, static_cast<int64_t>(i), 1}).build(ctx, attn);
        auto attn_rep = RepeatModule(RepeatConfig{value_columns[i].shape}).build(ctx, attn_col);
        weighted_columns.push_back(mul.build(ctx, attn_rep, value_columns[i]));
    }
    auto context = add_all(ctx, weighted_columns);

    context = permute_tensor(ctx, context, {0, 2, 1, 3});
    context = ensure_contiguous_layout(ctx, context);
    context = core::reshape_tensor(ctx, context, core::TensorShape::from_dims({query.shape.dims[0], query_frames, config.hidden_size}));
    if (query_keep_mask.has_value()) {
        context = MaskingModule().build(ctx, context, *query_keep_mask);
    }
    return out_proj.build(ctx, context, LinearWeights{weights.attention.out_weight, weights.attention.out_bias});
}

}  // namespace engine::modules::attention::internal
