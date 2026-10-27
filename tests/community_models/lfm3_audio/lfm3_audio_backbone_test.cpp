// The LFM3-Audio backbone runtime (backbone.h). Synthetic: the tiny test
// package against the host fp64 reference (lfm3_audio_reference.h): chunked
// prefill with audio rows and replayed frames, taps, the tied head, decode
// steps against a longer prefill, frame embeddings, cache capacity reuse and
// growth, and the error paths.
//
//   lfm3_audio_backbone_test [--backend cpu|cuda|metal|best] [--threads N]
//
// A requested backend that is not available exits 77.

#include "lfm3_audio_reference.h"
#include "lfm3_audio_test_package.h"
#include "test_assert.h"

#include "engine/community_models/lfm3_audio/assets.h"
#include "engine/community_models/lfm3_audio/backbone.h"
#include "engine/framework/core/execution_context.h"
#include "engine/framework/runtime/errors.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <limits>
#include <memory>
#include <string>
#include <vector>

namespace {

namespace lfm3 = engine::community_models::lfm3_audio;
namespace core = engine::core;
namespace ref = lfm3_audio_test;
using engine::test::require;
using lfm3::Lfm3BackboneRuntime;
using lfm3::Lfm3Output;
using ref::mode_name;
using ref::native_mode;
using ref::reference_mode;
using ref::rel_error;
using ref::report;

// Native mode: BF16 weights as stored (the reference reads the same values), F16 caches and flash attention. A BF16
// matmul also rounds its input to BF16 (CPU vec_dot_type), about 0.4% per projection: ~1e-2 over the tiny stack.
constexpr double kNativeTolerance = 3e-2;

double tolerance_of(const core::ExecutionContext & execution, const lfm3::Lfm3WeightOptions & options) {
    return lfm3::lfm3_reference_precision(options) ? ref::reference_tolerance(execution) : kNativeTolerance;
}

struct Synthetic {
    ref::Package package;
    std::shared_ptr<const lfm3::Lfm3AudioAssets> assets;
    ref::RefStack reference;
    int32_t audio_id = 0, audio_out_id = 0;
    int64_t d = 0, vocab = 0, codebooks = 0, audio_vocab = 0;
};

Synthetic make_synthetic() {
    Synthetic s;
    s.package = ref::write_package("lfm3_audio_backbone_test");
    s.assets = lfm3::load_lfm3_audio_assets(s.package.root);
    s.reference = ref::read_ref_stack(*s.assets->model, "lfm.model.", s.assets->backbone.stack);
    s.audio_id = s.package.ids.at("<|audio|>");
    s.audio_out_id = s.package.ids.at("<|audio_out|>");
    s.d = s.assets->backbone.stack.hidden;
    s.vocab = s.assets->backbone.vocab_size;
    s.codebooks = s.assets->depth.codebooks;
    s.audio_vocab = s.assets->depth.audio_vocab_size;
    return s;
}

std::unique_ptr<Lfm3BackboneRuntime> make_runtime(const Synthetic & s, core::ExecutionContext & execution,
                                                  const lfm3::Lfm3WeightOptions & options) {
    return std::make_unique<Lfm3BackboneRuntime>(s.assets->model, s.assets->backbone, s.codebooks, s.audio_vocab, execution, options);
}

struct Case {
    lfm3::Lfm3Prompt prompt;
    lfm3::Lfm3AudioEmbeddings audio;
};

Case make_case(const Synthetic & s, int64_t steps, int64_t audio_at, int64_t audio_count, const std::vector<int64_t> & frames_at,
               uint64_t seed) {
    lfm2_audio_test::Random random(seed);
    Case c;
    for (int64_t i = 0; i < steps; ++i) {
        // Byte tokens only: the specials sit above them.
        c.prompt.ids.push_back(static_cast<int32_t>(std::abs(random.uniform(1.0f)) * 255.0f));
    }
    for (int64_t i = audio_at; i < audio_at + audio_count; ++i) {
        c.prompt.ids[static_cast<size_t>(i)] = s.audio_id;
    }
    for (const int64_t p : frames_at) {
        c.prompt.ids[static_cast<size_t>(p)] = s.audio_out_id;
        std::vector<int32_t> codes;
        for (int64_t k = 0; k < s.codebooks; ++k) {
            codes.push_back(static_cast<int32_t>(std::abs(random.uniform(1.0f)) * static_cast<float>(s.audio_vocab - 1)));
        }
        c.prompt.frames.push_back(codes);
    }
    c.prompt.frames.push_back(std::vector<int32_t>(static_cast<size_t>(s.codebooks), static_cast<int32_t>(s.audio_vocab - 1)));
    c.prompt.ids.back() = s.audio_out_id;  // the last position is the EOS frame
    c.audio.tokens = audio_count;
    c.audio.hidden = s.d;
    c.audio.values = random.uniform(static_cast<size_t>(audio_count * s.d), 0.3f);
    return c;
}

ref::Values host_frame(const Synthetic & s, const std::vector<int32_t> & codes) {
    const auto & table = s.package.model.at("codebook_embedding.weight").values;
    ref::Values out(static_cast<size_t>(s.d), 0.0);
    for (int64_t k = 0; k < s.codebooks; ++k) {
        const size_t row = static_cast<size_t>(k * s.audio_vocab + codes[static_cast<size_t>(k)]);
        for (int64_t j = 0; j < s.d; ++j) {
            out[static_cast<size_t>(j)] += table[row * static_cast<size_t>(s.d) + static_cast<size_t>(j)];
        }
    }
    return out;
}

ref::Values host_embeddings(const Synthetic & s, const Case & c) {
    const auto & table = s.package.model.at("lfm.model.embed_tokens.weight").values;
    ref::Values x;
    size_t audio = 0;
    size_t frame = 0;
    for (const int32_t id : c.prompt.ids) {
        if (id == s.audio_id) {
            const auto * row = c.audio.values.data() + (audio++) * static_cast<size_t>(s.d);
            x.insert(x.end(), row, row + s.d);
        } else if (id == s.audio_out_id) {
            const auto row = host_frame(s, c.prompt.frames[frame++]);
            x.insert(x.end(), row.begin(), row.end());
        } else {
            const auto * row = table.data() + static_cast<size_t>(id) * static_cast<size_t>(s.d);
            x.insert(x.end(), row, row + s.d);
        }
    }
    return x;
}

ref::Values row_of(const ref::Values & rows, int64_t row, int64_t d) {
    return ref::Values(rows.begin() + row * d, rows.begin() + (row + 1) * d);
}

ref::Values host_logits(const Synthetic & s, const ref::Values & hidden) {
    return ref::linear(hidden, ref::to_values(s.package.model.at("lfm.model.embed_tokens.weight").values), s.d, s.vocab);
}

// 600 positions = 3 prefill blocks; sliding window 4 and full attention reach across block boundaries.
void test_prefill_reference(const Synthetic & s, core::ExecutionContext & execution) {
    const int64_t steps = 600;
    const auto c = make_case(s, steps, 20, 30, {270, 271, 300, 520}, 3);
    const auto expected = ref::stack(s.assets->backbone.stack, s.reference, host_embeddings(s, c), ref::positions_from(0, steps));

    for (const auto & options : {reference_mode(), native_mode()}) {
        auto runtime = make_runtime(s, execution, options);
        Lfm3BackboneRuntime::Taps taps;
        taps.positions = {0, 3, 255, 256, 259, 511, 512, 599};
        const auto logits = runtime->prefill(c.prompt, s.audio_id, s.audio_out_id, c.audio, 5, Lfm3Output::Logits, &taps);
        require(runtime->capacity() == 768 && runtime->position() == steps, "capacity 605 rounded up to 256 = 768, position = prompt length");

        ref::Values expected_taps;
        for (const int64_t p : taps.positions) {
            const auto row = row_of(expected, p, s.d);
            expected_taps.insert(expected_taps.end(), row.begin(), row.end());
        }
        const double tol = tolerance_of(execution, options);
        report(std::string("prefill taps (8 positions, 3 blocks) ") + mode_name(options), ref::rel_error(taps.rows, expected_taps), tol);
        const auto last = row_of(expected, steps - 1, s.d);
        report(std::string("prefill last logits ") + mode_name(options), ref::rel_error(logits, host_logits(s, last)), tol);

        const auto hidden = runtime->prefill(c.prompt, s.audio_id, s.audio_out_id, c.audio, 5, Lfm3Output::Hidden);
        report(std::string("prefill last hidden ") + mode_name(options), ref::rel_error(hidden, last), tol);
        report(std::string("logits(hidden) == prefill logits ") + mode_name(options), rel_error(runtime->logits(hidden), logits), 0.0);
    }
}

// prefill(L) + one step per position == prefill(L + n) at every step, across the 256 boundary, with text steps and
// frame steps (the EOS frame last).
void test_steps_match_prefill(const Synthetic & s, core::ExecutionContext & execution) {
    const int64_t total = 270;
    const int64_t head = 250;
    const auto c = make_case(s, total, 5, 12, {252, 256, 257, 263}, 7);

    for (const auto & options : {reference_mode(), native_mode()}) {
        auto runtime = make_runtime(s, execution, options);
        Lfm3BackboneRuntime::Taps taps;
        for (int64_t p = head - 1; p < total; ++p) {
            taps.positions.push_back(p);
        }
        runtime->prefill(c.prompt, s.audio_id, s.audio_out_id, c.audio, 0, Lfm3Output::Hidden, &taps);

        lfm3::Lfm3Prompt prefix;
        prefix.ids.assign(c.prompt.ids.begin(), c.prompt.ids.begin() + head);
        size_t frame = 0;
        for (const int32_t id : prefix.ids) {
            frame += id == s.audio_out_id ? 1 : 0;
        }
        prefix.frames.assign(c.prompt.frames.begin(), c.prompt.frames.begin() + static_cast<std::ptrdiff_t>(frame));
        const auto first = runtime->prefill(prefix, s.audio_id, s.audio_out_id, c.audio, total - head, Lfm3Output::Hidden);
        const double tol = tolerance_of(execution, options);
        report(std::string("prefill(250) last vs prefill(270) row 249 ") + mode_name(options),
               rel_error(first, std::vector<float>(taps.rows.begin(), taps.rows.begin() + s.d)), tol);

        std::vector<float> stepped;
        std::vector<float> expected;
        double logits_rel = 0.0;
        for (int64_t p = head; p < total; ++p) {
            const int32_t id = c.prompt.ids[static_cast<size_t>(p)];
            const auto output = p % 3 == 0 ? Lfm3Output::Logits : Lfm3Output::Hidden;
            const auto out = id == s.audio_out_id ? runtime->step_frame(c.prompt.frames[frame++], output) : runtime->step_text(id, output);
            const std::vector<float> want(taps.rows.begin() + (p - head + 1) * s.d, taps.rows.begin() + (p - head + 2) * s.d);
            if (output == Lfm3Output::Logits) {
                logits_rel = std::max(logits_rel, rel_error(out, runtime->logits(want)));
            } else {
                stepped.insert(stepped.end(), out.begin(), out.end());
                expected.insert(expected.end(), want.begin(), want.end());
            }
        }
        require(runtime->position() == total, "position after the steps");
        report(std::string("steps hidden vs prefill(270) taps ") + mode_name(options), rel_error(stepped, expected), tol);
        report(std::string("steps logits vs prefill(270) taps (worst step) ") + mode_name(options), logits_rel, tol);
    }
}

void test_frame_embedding(const Synthetic & s, core::ExecutionContext & execution) {
    auto runtime = make_runtime(s, execution, reference_mode());
    const auto c = make_case(s, 8, 0, 0, {1, 2, 3}, 11);
    const auto got = runtime->embed_frames(c.prompt.frames);
    ref::Values want;
    for (const auto & codes : c.prompt.frames) {
        const auto row = host_frame(s, codes);
        want.insert(want.end(), row.begin(), row.end());
    }
    report("embed_frames vs host sum (incl. EOS frame)", ref::rel_error(got, want), 1e-6);
}

// The same capacity reuses the state, cleared for each request; a longer request grows it; output never depends
// on what ran before.
void test_capacity(const Synthetic & s, core::ExecutionContext & execution) {
    const auto small = make_case(s, 300, 4, 6, {290}, 13);
    const auto longer = make_case(s, 400, 4, 6, {350}, 17);
    const auto big = make_case(s, 700, 4, 6, {650}, 19);
    for (const auto & options : {reference_mode(), native_mode()}) {
        auto fresh = make_runtime(s, execution, options);
        const auto expected = fresh->prefill(small.prompt, s.audio_id, s.audio_out_id, small.audio, 100, Lfm3Output::Logits);
        const auto expected_step = fresh->step_text(7, Lfm3Output::Hidden);

        auto runtime = make_runtime(s, execution, options);
        runtime->prefill(longer.prompt, s.audio_id, s.audio_out_id, longer.audio, 50, Lfm3Output::Hidden);
        runtime->step_text(9, Lfm3Output::Hidden);
        require(runtime->capacity() == 512, "capacity 450 rounded up to 256 = 512");
        const auto reused = runtime->prefill(small.prompt, s.audio_id, s.audio_out_id, small.audio, 100, Lfm3Output::Logits);
        require(runtime->capacity() == 512 && reused == expected, std::string("reused state gives the fresh output (") + mode_name(options) + ")");
        require(runtime->step_text(7, Lfm3Output::Hidden) == expected_step, "reused state: first step equals the fresh one");

        runtime->prefill(big.prompt, s.audio_id, s.audio_out_id, big.audio, 100, Lfm3Output::Hidden);
        require(runtime->capacity() == 1024, "capacity grows to 800 rounded up to 256 = 1024");
        const auto shrunk = runtime->prefill(small.prompt, s.audio_id, s.audio_out_id, small.audio, 100, Lfm3Output::Logits);
        require(runtime->capacity() == 512 && shrunk == expected, "back to 512: the fresh output");
        std::printf("PASS capacity reuse, growth to 1024 and back give identical output (%s)\n", mode_name(options));
    }
}

void test_errors(const Synthetic & s, core::ExecutionContext & execution) {
    auto runtime = make_runtime(s, execution, reference_mode());
    const auto c = make_case(s, 256, 4, 6, {100}, 23);
    const auto prefill = [&](const lfm3::Lfm3Prompt & prompt, const lfm3::Lfm3AudioEmbeddings & audio, int64_t max_new) {
        return runtime->prefill(prompt, s.audio_id, s.audio_out_id, audio, max_new, Lfm3Output::Hidden);
    };
    ref::require_throws([&] { runtime->step_text(1, Lfm3Output::Hidden); }, "step before prefill");

    auto short_audio = c.audio;
    short_audio.tokens -= 1;
    short_audio.values.resize(static_cast<size_t>(short_audio.tokens * s.d));
    ref::require_throws([&] { prefill(c.prompt, short_audio, 1); }, "audio rows != <|audio|> slots");
    auto missing_frame = c.prompt;
    missing_frame.frames.pop_back();
    ref::require_throws([&] { prefill(missing_frame, c.audio, 1); }, "frames != <|audio_out|> slots");
    auto bad_code = c.prompt;
    bad_code.frames[0][3] = static_cast<int32_t>(s.audio_vocab);
    ref::require_throws([&] { prefill(bad_code, c.audio, 1); }, "code outside the codebook");
    auto bad_id = c.prompt;
    bad_id.ids[0] = static_cast<int32_t>(s.vocab);
    ref::require_throws([&] { prefill(bad_id, c.audio, 1); }, "id outside the vocabulary");
    auto nan_audio = c.audio;
    nan_audio.values[0] = std::nanf("");
    ref::require_throws([&] { prefill(c.prompt, nan_audio, 1); }, "non-finite audio rows");
    bool capacity_error = false;
    try {
        prefill(c.prompt, c.audio, 8192);
    } catch (const engine::runtime::CapacityError &) {
        capacity_error = true;
    }
    require(capacity_error, "prompt + max_new past 8192 is a CapacityError");
    capacity_error = false;
    try {
        prefill(c.prompt, c.audio, std::numeric_limits<int64_t>::max());
    } catch (const engine::runtime::CapacityError &) {
        capacity_error = true;
    }
    require(capacity_error, "max_new = INT64_MAX is a CapacityError, not an overflow");

    // 256 + 0 fills the cache exactly: the first step has no row left.
    prefill(c.prompt, c.audio, 0);
    require(runtime->capacity() == 256, "capacity 256");
    capacity_error = false;
    try {
        runtime->step_text(1, Lfm3Output::Hidden);
    } catch (const engine::runtime::CapacityError &) {
        capacity_error = true;
    }
    require(capacity_error, "a step past the capacity is a CapacityError");
    ref::require_throws([&] { runtime->step_frame({1, 2}, Lfm3Output::Hidden); }, "a frame needs one code per codebook");
    ref::require_throws([&] { runtime->logits({1.0f}); }, "logits() needs a full hidden state");
    std::printf("PASS error paths\n");
}

}  // namespace

int main(int argc, char ** argv) {
    int status = 0;
    const auto execution = ref::open_backend(argc, argv, status);
    if (!execution) {
        return status;
    }

    try {
        const auto start = std::chrono::steady_clock::now();
        const auto synthetic = make_synthetic();
        test_prefill_reference(synthetic, *execution);
        test_steps_match_prefill(synthetic, *execution);
        test_frame_embedding(synthetic, *execution);
        test_capacity(synthetic, *execution);
        test_errors(synthetic, *execution);
        std::printf("PASS: lfm3_audio backbone synthetic (%.1f s)\n",
                    std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count());
    } catch (const std::exception & error) {
        std::cerr << "FAIL: " << error.what() << "\n";
        return 1;
    }
    return 0;
}
