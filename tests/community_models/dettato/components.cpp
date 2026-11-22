#include "engine/community_models/dettato/note_repair.h"
#include "engine/community_models/dettato/drums.h"
#include "engine/community_models/dettato/frontend.h"
#include "engine/community_models/dettato/grouper.h"
#include "engine/community_models/dettato/transcriber.h"
#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/audio/conversion.h"
#include "engine/framework/audio/wav_reader.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/io/json.h"
#include "engine/community_models/dettato/pipeline.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <unordered_map>

int main(int argc, char **argv) {
    const auto start = std::chrono::steady_clock::now();
    try {
        std::unordered_map<std::string, std::string> options;
        for (int i = 1; i < argc; ++i) {
            const std::string key = argv[i];
            if (key == "--log")
                continue;
            if (i + 1 == argc)
                throw std::runtime_error("Missing option value");
            options[key] = argv[++i];
        }
        engine::debug::configure_logging({true, options.at("--log-file")});
        const auto assets = engine::community_models::dettato::load_dettato_assets(options.at("--model"));
        const auto reference = engine::assets::open_tensor_source(options.at("--reference"));
        const auto check = [&](const std::string &key, const std::vector<float> &actual) {
            const auto expected = reference->require_f32(key);
            if (actual.size() != expected.size())
                throw std::runtime_error(key + ": shape mismatch");
            double dot = 0, aa = 0, bb = 0;
            float maximum = 0;
            for (size_t i = 0; i < actual.size(); ++i) {
                if (!std::isfinite(actual[i]) || !std::isfinite(expected[i]))
                    throw std::runtime_error(key + ": nonfinite comparison value");
                dot += static_cast<double>(actual[i]) * expected[i];
                aa += static_cast<double>(actual[i]) * actual[i];
                bb += static_cast<double>(expected[i]) * expected[i];
                maximum = std::max(maximum, std::abs(actual[i] - expected[i]));
            }
            const double cosine = dot / std::sqrt(aa * bb);
            engine::debug::timing_log_scalar("dettato." + key + ".cosine", cosine);
            engine::debug::timing_log_scalar("dettato." + key + ".max_abs", maximum);
            if (!std::isfinite(cosine) || cosine < 0.99999) {
                engine::debug::trace_log_f32(key + ".actual", {static_cast<int64_t>(actual.size())}, actual);
                engine::debug::trace_log_f32(key + ".expected", {static_cast<int64_t>(expected.size())}, expected);
                throw std::runtime_error(key + ": cosine gate failed");
            }
        };
        if (options.count("--network")) {
            if (options.at("--network") == "drum-decode") {
                const auto cases = reference->require_i64_scalar("case_count");
                for (int64_t c = 0; c < cases; ++c) {
                    const auto key = "case." + std::to_string(c) + ".";
                    engine::community_models::dettato::detail::DrumProbabilities p;
                    p.router = reference->require_f32(key + "router").at(0);
                    p.kit = reference->require_f32(key + "kit"); p.kit_velocity = reference->require_f32(key + "kit_vel");
                    p.orchestral = reference->require_f32(key + "drum"); p.timpani = reference->require_f32(key + "timpani");
                    p.velocity = reference->require_f32(key + "vel");
                    const auto track = engine::community_models::dettato::detail::decode_drum_hits(p, assets->config.require("drums"));
                    if (track.notes.size() != static_cast<size_t>(reference->require_i64_scalar(key + "hit_count")))
                        throw std::runtime_error(key + "drum hit count differs from upstream");
                    if (!track.notes.empty()) {
                        const auto expected = reference->require_f32(key + "hits");
                        for (size_t i = 0; i < track.notes.size(); ++i) {
                            const auto & note = track.notes[i];
                            if (std::abs(note.onset - expected[i * 4]) > 1e-6 || std::abs(note.offset - expected[i * 4 + 1]) > 1e-6 ||
                                note.pitch != expected[i * 4 + 2] || note.velocity != expected[i * 4 + 3])
                                throw std::runtime_error(key + "drum hit differs from upstream at " + std::to_string(i));
                        }
                    }
                }
                engine::debug::timing_log_scalar("dettato.drums.decode.checked_cases", cases);
                engine::debug::timing_log_scalar("session.wall_ms", engine::debug::elapsed_ms(start));
                return 0;
            }
            if (options.at("--network") == "grouping-decode") {
                const auto cases = reference->require_i64_scalar("case_count");
                engine::community_models::dettato::detail::GroupingDecoder decoder(engine::community_models::dettato::detail::load_grouping_decoder_weights(*assets->grouper), assets->config.require("grouper"));
                for (int64_t c = 0; c < cases; ++c) {
                    const auto prefix = "case." + std::to_string(c) + ".";
                    const auto onset_f = reference->require_f32(prefix + "onsets");
                    std::vector<double> onsets(onset_f.begin(), onset_f.end());
                    const auto settings = reference->require_f32(prefix + "settings");
                    std::vector<engine::community_models::dettato::detail::GroupingWindow> windows;
                    for (int64_t w = 0; w < reference->require_i64_scalar(prefix + "window_count"); ++w) {
                        const auto key = prefix + "window." + std::to_string(w) + ".";
                        engine::community_models::dettato::detail::GroupingWindow window;
                        window.start = reference->require_f32(key + "start").at(0);
                        if (reference->require_i64_scalar(key + "note_count") > 0) {
                            for (const auto id : reference->require_f32(key + "ids")) window.notes.push_back(static_cast<int>(id));
                            window.embeddings = reference->require_f32(key + "embeddings");
                            window.probabilities = reference->require_f32(key + "probabilities");
                        }
                        windows.push_back(std::move(window));
                    }
                    const auto chain = engine::community_models::dettato::detail::chain_grouping_windows(windows, onsets, 24, settings[0], settings[1]);
                    const auto merged = engine::community_models::dettato::detail::merge_grouping_tracklets(chain,
                        reference->require_f32(prefix + "embeddings"), reference->require_f32(prefix + "probabilities"), settings[2], settings[1]);
                    for (const auto & result : {std::make_pair("chain", chain), std::make_pair("merged", merged)}) {
                        const auto expected = reference->require_f32(prefix + result.first);
                        if (result.second.size() != expected.size()) throw std::runtime_error(prefix + result.first + ": shape mismatch");
                        for (size_t i = 0; i < expected.size(); ++i) if (result.second[i] != expected[i])
                            throw std::runtime_error(prefix + result.first + ": differing note " + std::to_string(i) +
                                " actual=" + std::to_string(result.second[i]) + " expected=" + std::to_string(expected[i]));
                    }
                    engine::community_models::dettato::detail::GroupingPrediction prediction;
                    prediction.windows = windows;
                    prediction.embeddings = reference->require_f32(prefix + "embeddings");
                    prediction.probabilities = reference->require_f32(prefix + "probabilities");
                    prediction.hidden = reference->require_f32(prefix + "hidden");
                    const auto rows = reference->require_f32(prefix + "notes");
                    std::vector<engine::community_models::dettato::Note> notes;
                    for (size_t i = 0; i < onsets.size(); ++i) notes.push_back({rows[i * 4], rows[i * 4 + 1],
                        static_cast<int>(rows[i * 4 + 2]), 100, rows[i * 4 + 3], 0, false});
                    const auto tracks = decoder.decode(prediction, notes);
                    const auto track_count = reference->require_i64_scalar(prefix + "track_count");
                    if (tracks.size() != static_cast<size_t>(track_count)) throw std::runtime_error(prefix + "track count mismatch");
                    for (size_t t = 0; t < tracks.size(); ++t) {
                        const auto key = prefix + "track." + std::to_string(t) + ".";
                        const auto expected = reference->require_f32(key + "notes");
                        if (tracks[t].program != reference->require_i64_scalar(key + "program") || tracks[t].notes.size() * 4 != expected.size())
                            throw std::runtime_error(key + "instrument or note count mismatch");
                        for (size_t i = 0; i < tracks[t].notes.size(); ++i) {
                            const auto & note = tracks[t].notes[i];
                            if (std::abs(note.onset - expected[i * 4]) > 1e-6 || std::abs(note.offset - expected[i * 4 + 1]) > 1e-6 ||
                                note.pitch != expected[i * 4 + 2] || note.velocity != expected[i * 4 + 3])
                                throw std::runtime_error(key + "note mismatch at " + std::to_string(i));
                        }
                    }
                }
                engine::debug::timing_log_scalar("dettato.grouping_decode.checked_cases", cases);
                engine::debug::timing_log_scalar("session.wall_ms", engine::debug::elapsed_ms(start));
                return 0;
            }
            if(options.at("--network")=="grouping-features") {
                engine::community_models::dettato::detail::WindowTranscription transcription;
                const auto windows=engine::assets::open_tensor_source(options.at("--windows"));
                transcription.frames=windows->require_metadata("maps.0").shape[0];
                for(size_t i=0;i<6;++i) transcription.grouping_maps[i]=windows->require_f32("group.maps."+std::to_string(i));
                const auto h=windows->require_f32("group.hidden"),s=windows->require_f32("group.stem");
                transcription.grouping_hidden->resize(h.size());transcription.grouping_stem->resize(s.size());
                for(size_t i=0;i<h.size();++i) {(*transcription.grouping_hidden)[i]=ggml_fp32_to_fp16(h[i]);(*transcription.grouping_stem)[i]=ggml_fp32_to_fp16(s[i]);}
                const auto raw=reference->require_tensor_data("input.notes");
                if(raw.metadata.dtype!="F64") throw std::runtime_error("Grouping source note times must retain F64");
                std::vector<double> rows(raw.bytes.size()/sizeof(double));std::memcpy(rows.data(),raw.bytes.data(),raw.bytes.size());
                std::vector<engine::community_models::dettato::Note> notes;
                for(size_t i=0;i<rows.size();i+=3) notes.push_back({rows[i],rows[i+1],static_cast<int>(rows[i+2]),100,1,1,false});
                const auto gathered=engine::community_models::dettato::detail::collect_grouping_features(transcription,notes,{});
                check("notes",gathered.notes);
                std::vector<float> taps(gathered.taps.size());
                for(size_t i=0;i<taps.size();++i) taps[i]=ggml_fp16_to_fp32(gathered.taps[i]);
                check("taps",taps);
                engine::debug::timing_log_scalar("session.wall_ms",engine::debug::elapsed_ms(start));
                return 0;
            }
            if (options.at("--network") == "evidence-extract" || options.at("--network") == "repair") {
                engine::community_models::dettato::detail::WindowTranscription transcription;
                const auto windows = engine::assets::open_tensor_source(options.at("--windows"));
                transcription.frames = windows->require_metadata("maps.0").shape[0];
                for (size_t i = 0; i < 6; ++i)
                    transcription.maps[i] = windows->require_f32("maps." + std::to_string(i));
                const auto hidden = windows->require_f32("hidden");
                transcription.hidden->resize(hidden.size());
                for (size_t i = 0; i < hidden.size(); ++i)
                    (*transcription.hidden)[i] = ggml_fp32_to_fp16(hidden[i]);
                const auto raw_notes = reference->require_tensor_data("notes");
                if (raw_notes.metadata.dtype != "F64")
                    throw std::runtime_error("Evidence reference notes must retain F64 timing");
                std::vector<double> source_notes(raw_notes.bytes.size() / sizeof(double));
                std::memcpy(source_notes.data(), raw_notes.bytes.data(), raw_notes.bytes.size());
                std::vector<engine::community_models::dettato::Note> notes;
                for (size_t i = 0; i < source_notes.size(); i += 5)
                    notes.push_back({source_notes[i], source_notes[i + 1], static_cast<int>(source_notes[i + 2]), 100,
                                     static_cast<float>(source_notes[i + 3]), static_cast<int>(source_notes[i + 4]),
                                     false});
                const auto features = engine::community_models::dettato::detail::extract_note_evidence(transcription, notes, 8);
                check("evidence", features);
                if (features != engine::community_models::dettato::detail::extract_note_evidence(transcription, notes, 1))
                    throw std::runtime_error("Evidence gathering changed with thread count");
                if (options.at("--network") == "repair") {
                    engine::core::ExecutionContext execution({engine::core::BackendType::Cpu, 0, 8});
                    engine::community_models::dettato::detail::NoteRepairNetwork network(engine::community_models::dettato::detail::bind_note_repair_weights(*assets->note_repair, execution), execution);
                    const auto order = reference->require_f32("order");
                    std::vector<engine::community_models::dettato::Note> sorted_notes;
                    std::vector<float> sorted_evidence(features.size());
                    for (size_t i = 0; i < order.size(); ++i) {
                        const auto index = static_cast<size_t>(order[i]);
                        sorted_notes.push_back(notes[index]);
                        std::copy_n(features.begin() + index * 769, 769, sorted_evidence.begin() + i * 769);
                    }
                    const auto prepared =
                        engine::community_models::dettato::detail::make_note_repair_inputs(sorted_notes, sorted_evidence, 0);
                    check("prepared.on", prepared.onsets);
                    check("prepared.duration", prepared.durations);
                    check("prepared.continuous", prepared.continuous);
                    check("prepared.cosine", prepared.cosine);
                    check("prepared.sine", prepared.sine);
                    const auto raw = network.run(prepared);
                    const std::array<const char *, 5> head_names{"keep", "don", "dur", "ins.0", "ins.1"};
                    for (size_t i = 0; i < head_names.size(); ++i)
                        check(std::string("prepared.") + head_names[i], raw[i]);
                    const auto result = engine::community_models::dettato::detail::repair_notes(network, notes, features);
                    const auto expected = reference->require_f32("repaired");
                    const auto stats = reference->require_f32("stats");
                    engine::debug::timing_log_scalar("dettato.repair.actual_notes", result.notes.size());
                    engine::debug::timing_log_scalar("dettato.repair.actual_dropped", result.dropped);
                    engine::debug::timing_log_scalar("dettato.repair.actual_inserted", result.inserted);
                    engine::debug::timing_log_scalar("dettato.repair.expected_notes", expected.size() / 5);
                    engine::debug::timing_log_scalar("dettato.repair.expected_dropped", stats[0]);
                    engine::debug::timing_log_scalar("dettato.repair.expected_inserted", stats[1]);
                    if (result.notes.size() * 5 != expected.size() || result.dropped != static_cast<size_t>(stats[0]) ||
                        result.inserted != static_cast<size_t>(stats[1]))
                        throw std::runtime_error("Repaired note count differs from upstream");
                    for (size_t i = 0; i < result.notes.size(); ++i) {
                        const auto &note = result.notes[i];
                        if (std::abs(note.onset - expected[i * 5]) > 1e-6 ||
                            std::abs(note.offset - expected[i * 5 + 1]) > 1e-6 ||
                            note.pitch != static_cast<int>(expected[i * 5 + 2]) ||
                            std::abs(note.confidence - expected[i * 5 + 3]) > 1e-5 ||
                            note.inserted != (expected[i * 5 + 4] != 0))
                            throw std::runtime_error("Repaired note event differs from upstream");
                    }
                    engine::debug::timing_log_scalar("dettato.evidence.matched_repaired_notes", result.notes.size());
                    const std::vector<engine::community_models::dettato::Note> before_zero{
                        {-0.01, 0.03, 60, 100, 0.3f, 1, false}};
                    const auto untouched =
                        engine::community_models::dettato::detail::repair_notes(network, before_zero, std::vector<float>(769));
                    if (untouched.notes.size() != 1 || untouched.dropped != 0 || untouched.inserted != 0 ||
                        std::abs(untouched.notes[0].onset + 0.01) > 1e-12 ||
                        std::abs(untouched.notes[0].offset - 0.03) > 1e-12 || untouched.notes[0].confidence != 1 ||
                        untouched.notes[0].pitch != 60)
                        throw std::runtime_error("Out-of-window repair note was not preserved");
                }
                engine::debug::timing_log_scalar("session.wall_ms", engine::debug::elapsed_ms(start));
                return 0;
            }
            if (options.at("--network") == "decode") {
                std::array<std::vector<float>, 6> maps;
                for (size_t i = 0; i < maps.size(); ++i)
                    maps[i] = reference->require_f32("maps." + std::to_string(i));
                const auto frames = reference->require_metadata("maps.0").shape[0];
                const auto notes = engine::community_models::dettato::detail::decode_notes(maps, frames, 0.02, {});
                const auto expected = reference->require_f32("notes");
                if (expected.size() != notes.size() * 5)
                    throw std::runtime_error("Note decoder count differs from Python");
                for (size_t i = 0; i < notes.size(); ++i) {
                    const auto &note = notes[i];
                    if (std::abs(note.onset - expected[i * 5]) > 1e-6 ||
                        std::abs(note.offset - expected[i * 5 + 1]) > 1e-6 ||
                        note.pitch != static_cast<int>(expected[i * 5 + 2]) ||
                        std::abs(note.confidence - expected[i * 5 + 3]) > 1e-6 ||
                        note.voice != static_cast<int>(expected[i * 5 + 4]))
                        throw std::runtime_error("Note decoder event differs from Python");
                }
                engine::debug::timing_log_scalar("dettato.decode.matched_notes", notes.size());
                engine::debug::timing_log_scalar("session.wall_ms", engine::debug::elapsed_ms(start));
                return 0;
            }
            auto backend = engine::core::BackendType::Cpu;
            if (options.count("--backend")) {
                const auto name = options.at("--backend");
                if (name == "cuda")
                    backend = engine::core::BackendType::Cuda;
                else if (name == "vulkan")
                    backend = engine::core::BackendType::Vulkan;
                else if (name != "cpu")
                    throw std::runtime_error("Unsupported backend");
            }
            engine::core::ExecutionContext execution({backend, 0, 8});
            if (options.at("--network") == "pipeline") {
                const auto wav = engine::audio::read_wav_f32(std::filesystem::path(options.at("--audio")));
                const auto mono = engine::audio::mixdown_interleaved_to_mono_average(wav.samples, wav.channels);
                engine::community_models::dettato::DettatoPipeline pipeline(assets, execution,
                    options.count("--window-hop-sec") ? std::stoi(options.at("--window-hop-sec")) : 2);
                const int repeats = options.count("--repeats") ? std::stoi(options.at("--repeats")) : 2;
                for (int i = 0; i <= repeats; ++i) {
                    ggml_backend_synchronize(execution.backend());
                    const auto measured = std::chrono::steady_clock::now();
                    const auto result = pipeline.transcribe(mono, wav.sample_rate);
                    ggml_backend_synchronize(execution.backend());
                    const double ms = engine::debug::elapsed_ms(measured);
                    size_t notes = 0;
                    for (const auto & track : result.tracks) notes += track.notes.size();
                    engine::debug::timing_log_scalar("dettato.pipeline.run", i);
                    engine::debug::timing_log_scalar("dettato.pipeline.warmup", i == 0);
                    engine::debug::timing_log_scalar("session.wall_ms", ms);
                    engine::debug::timing_log_scalar("dettato.pipeline.rtf", ms / 1000 * wav.sample_rate / mono.size());
                    engine::debug::timing_log_scalar("dettato.pipeline.notes", notes);
                    engine::debug::timing_log_scalar("dettato.pipeline.tracks", result.tracks.size());
                }
                return 0;
            }
            if (options.at("--network") == "drum-expert") {
                const auto full = reference->require_f32("features"), short_input = reference->require_f32("short.features");
                bool router_mismatch = false;
                for (size_t k = 0; k < 3; ++k) {
                    engine::community_models::dettato::detail::OrchestralDrumExpert expert(engine::community_models::dettato::detail::bind_drum_adapter_weights(*assets->drums, k, assets->config.require("drums"), execution), assets->config.require("drums"), execution);
                    const auto collect = [](const auto & values) {
                        std::array<std::vector<float>, 5> result;
                        for (size_t i = 0; i < result.size(); ++i) result[i] = engine::core::read_tensor_f32(values[i].tensor);
                        return result;
                    };
                    const auto result = collect(expert.run(full));
                    const auto shorter = collect(expert.run(short_input));
                    for (size_t i = 0; i < result.size(); ++i) {
                        const auto key = "expert." + std::to_string(k) + "." + std::to_string(i);
                        check(key, result[i]); check("short." + key, shorter[i]);
                        if (i == 4 && (std::abs(result[i][0] - reference->require_f32(key)[0]) > 5e-3f ||
                            std::abs(shorter[i][0] - reference->require_f32("short." + key)[0]) > 5e-3f))
                            router_mismatch = true;
                    }
                    if (result != collect(expert.run(full))) throw std::runtime_error("Orchestral A/B/A reuse changed outputs");
                }
                if (router_mismatch) throw std::runtime_error("Orchestral router scalar differs from upstream");
                engine::debug::timing_log_scalar("session.wall_ms", engine::debug::elapsed_ms(start));
                return 0;
            }
            if (options.at("--network") == "drums") {
                engine::community_models::dettato::detail::DrumTranscriber network(engine::community_models::dettato::detail::bind_drum_weights(*assets->drums, assets->config.require("drums"), execution), assets->config.require("drums"), execution);
                const auto features = reference->require_f32("features");
                const auto result = network.probabilities(features);
                check("kit", result.kit); check("kit_vel", result.kit_velocity); check("drum", result.orchestral);
                check("timpani", result.timpani); check("vel", result.velocity);
                const auto expected_router = reference->require_f32("router")[0];
                engine::debug::timing_log_scalar("dettato.drums.router_abs_error", std::abs(result.router - expected_router));
                if (std::abs(result.router - expected_router) > 1e-5) throw std::runtime_error("Drum recording router differs from upstream");
                const auto audio = reference->require_f32("audio44");
                engine::community_models::dettato::detail::DrumFrontend frontend(engine::community_models::dettato::detail::load_drum_frontend(*assets->drums), 8);
                const auto output = network.transcribe_features(frontend.extract(audio));
                const auto expected = reference->require_f32("hits");
                if (output.track.notes.size() * 4 != expected.size() || output.model != "drum-kit") throw std::runtime_error("Drum track count/route differs from upstream");
                for (size_t i = 0; i < output.track.notes.size(); ++i) {
                    const auto & note = output.track.notes[i];
                    if (std::abs(note.onset - expected[i * 4]) > 1e-6 || std::abs(note.offset - expected[i * 4 + 1]) > 1e-6 ||
                        note.pitch != expected[i * 4 + 2] || note.velocity != expected[i * 4 + 3]) throw std::runtime_error("Drum track event differs from upstream");
                }
                const auto short_result = network.probabilities(reference->require_f32("short.features"));
                if (!std::isfinite(short_result.router)) throw std::runtime_error("Short drum request has a nonfinite router");
                const auto repeated = network.probabilities(features);
                if (result.kit != repeated.kit || result.kit_velocity != repeated.kit_velocity || result.orchestral != repeated.orchestral ||
                    result.timpani != repeated.timpani || result.velocity != repeated.velocity || result.router != repeated.router)
                    throw std::runtime_error("Drum probability A/B/A reuse changed outputs");
                engine::debug::timing_log_scalar("dettato.drums.matched_hits", output.track.notes.size());
                engine::debug::timing_log_scalar("session.wall_ms", engine::debug::elapsed_ms(start));
                return 0;
            }
            if (options.at("--network") == "drum-trunk") {
                const std::array<std::string, 2> prefixes{"drum_kit.", "orchestral.0.base."};
                for (size_t k = 0; k < prefixes.size(); ++k) {
                    engine::community_models::dettato::detail::DrumTrunk network(engine::community_models::dettato::detail::bind_drum_trunk_weights(*assets->drums, prefixes[k], assets->config.require("drums").require("drum_kit"), execution), assets->config.require("drums").require("drum_kit"), execution);
                    const auto a = reference->require_f32_tensor("input." + std::to_string(k));
                    const auto b = reference->require_f32_tensor("short.input." + std::to_string(k));
                    const auto collect = [](const auto & values) {
                        return std::array<std::vector<float>, 3>{engine::core::read_tensor_f32(values[0].tensor),
                            engine::core::read_tensor_f32(values[1].tensor), engine::core::read_tensor_f32(values[2].tensor)};
                    };
                    const auto result = collect(network.run(a.values));
                    const auto shorter = collect(network.run(b.values));
                    for (size_t i = 0; i < result.size(); ++i) {
                        check("trunk." + std::to_string(k) + "." + std::to_string(i), result[i]);
                        check("short.trunk." + std::to_string(k) + "." + std::to_string(i), shorter[i]);
                    }
                    if (result != collect(network.run(a.values))) throw std::runtime_error("Drum trunk A/B/A reuse changed outputs");
                    engine::community_models::dettato::detail::DrumEmbedder embedder(engine::community_models::dettato::detail::bind_drum_embedder_weights(*assets->drums, prefixes[k], assets->config.require("drums").require("drum_kit"), execution), assets->config.require("drums").require("drum_kit"), execution);
                    const auto composed = collect(network.run(embedder.run(reference->require_f32("mel"))));
                    for (size_t i = 0; i < composed.size(); ++i) check("trunk." + std::to_string(k) + "." + std::to_string(i), composed[i]);
                }
                engine::debug::timing_log_scalar("session.wall_ms", engine::debug::elapsed_ms(start));
                return 0;
            }
            if (options.at("--network") == "drum-embed") {
                const std::array<std::string, 2> prefixes{"drum_kit.", "orchestral.0.base."};
                for (size_t k = 0; k < prefixes.size(); ++k) {
                    engine::community_models::dettato::detail::DrumEmbedder network(engine::community_models::dettato::detail::bind_drum_embedder_weights(*assets->drums, prefixes[k], assets->config.require("drums").require("drum_kit"), execution), assets->config.require("drums").require("drum_kit"), execution);
                    const auto mel = reference->require_f32("mel");
                    const auto output = network.run(mel);
                    check("embed." + std::to_string(k), output);
                    const auto shorter = network.run(reference->require_f32("short.mel"));
                    check("short.embed." + std::to_string(k), shorter);
                    const auto repeated = network.run(mel);
                    if (output != repeated) throw std::runtime_error("Drum embedding A/B/A reuse changed outputs");
                }
                engine::debug::timing_log_scalar("session.wall_ms", engine::debug::elapsed_ms(start));
                return 0;
            }
            if (options.at("--network") == "grouping-windows") {
                engine::community_models::dettato::detail::GroupingFeatures features;
                features.notes = reference->require_f32("notes");
                const auto taps = reference->require_f32("taps");
                features.taps.resize(taps.size());
                for (size_t i = 0; i < taps.size(); ++i) features.taps[i] = ggml_fp32_to_fp16(taps[i]);
                engine::community_models::dettato::detail::WindowGrouper network(engine::community_models::dettato::detail::bind_grouper_weights(*assets->grouper, execution), assets->config.require("grouper"), execution);
                const auto audio = reference->require_f32("audio16");
                const auto prediction = network.predict(audio, features);
                const std::array<const std::vector<float> *, 3> actual{&prediction.probabilities, &prediction.embeddings, &prediction.hidden};
                const std::array<const char *, 3> names{"probs", "emb", "h"};
                for (size_t k = 0; k < names.size(); ++k) {
                    const auto expected = reference->require_f32(names[k]);
                    if (actual[k]->size() != expected.size()) throw std::runtime_error("Grouping-window prediction shape mismatch");
                    double dot = 0, aa = 0, bb = 0; float maximum = 0;
                    for (size_t i = 0; i < expected.size(); ++i) {
                        const double a = (*actual[k])[i], b = expected[i];
                        if (!std::isfinite(a)) throw std::runtime_error("Grouping-window prediction is nonfinite");
                        dot += a * b; aa += a * a; bb += b * b; maximum = std::max(maximum, static_cast<float>(std::abs(a - b)));
                    }
                    engine::debug::timing_log_scalar(std::string("dettato.grouping_windows.") + names[k] + ".cosine", dot / std::sqrt(aa * bb));
                    engine::debug::timing_log_scalar(std::string("dettato.grouping_windows.") + names[k] + ".max_abs", maximum);
                }
                if (prediction.windows.size() != static_cast<size_t>(reference->require_i64_scalar("window_count")))
                    throw std::runtime_error("Grouping-window count differs from upstream");
                std::vector<engine::community_models::dettato::Note> notes;
                for (size_t i = 0; i < features.notes.size(); i += 12) {
                    const auto * m = features.notes.data() + i;
                    notes.push_back({m[0], m[1], static_cast<int>(m[2]), 100, m[3] > .5f ? m[5] : m[4], 0, false});
                }
                engine::community_models::dettato::detail::GroupingDecoder decoder(engine::community_models::dettato::detail::load_grouping_decoder_weights(*assets->grouper), assets->config.require("grouper"));
                const auto tracks = decoder.decode(prediction, notes);
                if (tracks.size() != static_cast<size_t>(reference->require_i64_scalar("track_count")))
                    throw std::runtime_error("Grouping-window track count differs from upstream");
                for (size_t t = 0; t < tracks.size(); ++t) {
                    const auto key = "track." + std::to_string(t) + ".";
                    const auto expected = reference->require_f32(key + "notes");
                    if (tracks[t].program != reference->require_i64_scalar(key + "program") || tracks[t].notes.size() * 4 != expected.size())
                        throw std::runtime_error("Grouping-window instrument or note count differs from upstream");
                    for (size_t i = 0; i < tracks[t].notes.size(); ++i) {
                        const auto & note = tracks[t].notes[i];
                        if (std::abs(note.onset - expected[i * 4]) > 1e-6 || std::abs(note.offset - expected[i * 4 + 1]) > 1e-6 ||
                            note.pitch != expected[i * 4 + 2] || note.velocity != expected[i * 4 + 3])
                            throw std::runtime_error("Grouping-window final note differs from upstream");
                    }
                }
                auto few = features;
                // Three notes exercises upstream's duplicate-to-four path and a
                // different body shape, before returning to the original song.
                const size_t begin = features.notes.size() / 12 > 15 ? 12 : 0;
                const size_t count = std::min<size_t>(3, features.notes.size() / 12 - begin);
                few.notes.assign(features.notes.begin() + begin * 12, features.notes.begin() + (begin + count) * 12);
                few.taps.assign(features.taps.begin() + begin * 1104, features.taps.begin() + (begin + count) * 1104);
                const auto small = network.predict(audio, few);
                for (const auto x : small.probabilities) if (!std::isfinite(x)) throw std::runtime_error("Small-window probabilities are nonfinite");
                const auto repeated = network.predict(audio, features);
                if (prediction.probabilities != repeated.probabilities || prediction.embeddings != repeated.embeddings || prediction.hidden != repeated.hidden)
                    throw std::runtime_error("Grouping-window A/B/A reuse changed outputs");
                engine::debug::timing_log_scalar("dettato.grouping_windows.matched_tracks", tracks.size());
                engine::debug::timing_log_scalar("session.wall_ms", engine::debug::elapsed_ms(start));
                return 0;
            }
            if (options.at("--network") == "windows") {
                engine::community_models::dettato::detail::WindowTranscriber network(engine::community_models::dettato::detail::bind_transcriber_weights(*assets->transcriber, execution), execution,
                    options.count("--window-hop-sec") ? std::stoi(options.at("--window-hop-sec")) : 2);
                const auto audio = reference->require_f32("audio16");
                network.run_device(audio);
                const auto outputs = network.readback();
                for (size_t head = 0; head < 6; ++head) {
                    check("maps." + std::to_string(head), outputs.maps[head]);
                    check("group.maps." + std::to_string(head), outputs.grouping_maps[head]);
                }
                const std::array<const std::vector<ggml_fp16_t> *, 3> taps{outputs.hidden.get(), outputs.grouping_hidden.get(),
                                                                           outputs.grouping_stem.get()};
                const std::array<const char *, 3> names{"hidden", "group.hidden", "group.stem"};
                for (size_t i = 0; i < taps.size(); ++i) {
                    std::vector<float> values(taps[i]->size());
                    for (size_t j = 0; j < values.size(); ++j)
                        values[j] = ggml_fp16_to_fp32((*taps[i])[j]);
                    check(names[i], values);
                }
                // The component returns borrowed session buffers. Retain an
                // independent snapshot so reuse cannot make this check vacuous.
                const std::array<std::vector<ggml_fp16_t>, 3> snapshot{
                    *outputs.hidden, *outputs.grouping_hidden, *outputs.grouping_stem};
                network.run_device(reference->require_f32("short.audio16"));
                const auto shorter = network.readback();
                for (size_t head = 0; head < 6; ++head) {
                    check("short.maps." + std::to_string(head), shorter.maps[head]);
                    check("short.group.maps." + std::to_string(head), shorter.grouping_maps[head]);
                }
                const std::array<const std::vector<ggml_fp16_t> *, 3> short_taps{
                    shorter.hidden.get(), shorter.grouping_hidden.get(), shorter.grouping_stem.get()};
                for (size_t i = 0; i < short_taps.size(); ++i) {
                    std::vector<float> values(short_taps[i]->size());
                    ggml_fp16_to_fp32_row(short_taps[i]->data(), values.data(), static_cast<int64_t>(values.size()));
                    check(std::string("short.") + names[i], values);
                }
                network.run_device(audio);
                const auto repeated = network.readback();
                if (outputs.maps != repeated.maps || snapshot[0] != *repeated.hidden ||
                    snapshot[1] != *repeated.grouping_hidden || snapshot[2] != *repeated.grouping_stem)
                    throw std::runtime_error("Window accumulation reuse changed outputs");
                if (execution.backend_type() == engine::core::BackendType::Cuda) {
                    network.run_device(audio, false);
                    auto compact = network.readback();
                    const double duration = compact.frames * .02;
                    const std::vector<engine::community_models::dettato::Note> small{{0, .02, 21, 100, 1, 1, false}};
                    const std::vector<engine::community_models::dettato::Note> larger{
                        {0, std::min(duration, 2.0), 21, 100, 1, 1, false},
                        {std::max(0.0, duration - .06), duration, 108, 100, 1, 1, false},
                        {.01, std::min(duration, .31), 21, 100, 1, 1, false}};
                    for (const auto &selection : {small, larger, std::vector<engine::community_models::dettato::Note>{}, small}) {
                        network.readback_grouping(compact, selection);
                        if (*compact.hidden != snapshot[0])
                            throw std::runtime_error("Compact grouping overwrote evidence hidden states");
                        for (size_t i = 0; i < compact.grouping_rows.size(); ++i)
                            for (size_t c = 0; c < 368; ++c) {
                                const size_t source = static_cast<size_t>(compact.grouping_rows[i]) * 368 + c;
                                if ((*compact.grouping_hidden)[i * 368 + c] != snapshot[1][source] ||
                                    (*compact.grouping_stem)[i * 368 + c] != snapshot[2][source])
                                    throw std::runtime_error("Compact grouping differs from full-bank readback");
                            }
                    }
                    // Host feature storage changes size while the bounded device
                    // graph remains reusable; selected rows must remain correct.
                    network.run_device(reference->require_f32("short.audio16"), false);
                    compact = network.readback();
                    network.readback_grouping(compact, small);
                    network.run_device(audio, false);
                    compact = network.readback();
                    network.readback_grouping(compact, small);
                    if (*compact.hidden != snapshot[0])
                        throw std::runtime_error("Compact readback shape replacement changed evidence states");
                    for (size_t c = 0; c < 368; ++c)
                        if ((*compact.grouping_hidden)[c] != snapshot[1][c] ||
                            (*compact.grouping_stem)[c] != snapshot[2][c])
                            throw std::runtime_error("Compact readback shape replacement changed selected rows");
                }
                engine::debug::timing_log_scalar("session.wall_ms", engine::debug::elapsed_ms(start));
                return 0;
            }
            if (options.at("--network") == "grouper") {
                engine::community_models::dettato::detail::GrouperInputs inputs;
                for (float value : reference->require_f32("pitch"))
                    inputs.pitch.push_back(static_cast<int32_t>(value));
                inputs.scalars = reference->require_f32("scalars");
                inputs.harmonics = reference->require_f32("harmonics");
                inputs.profile = reference->require_f32("profile");
                inputs.lanes = reference->require_f32("lanes");
                inputs.taps = reference->require_f32("taps");
                inputs.cosine = reference->require_f32("cosine");
                inputs.sine = reference->require_f32("sine");
                engine::community_models::dettato::detail::GrouperNetwork network(engine::community_models::dettato::detail::bind_grouper_weights(*assets->grouper, execution), execution);
                const auto front_tensor=reference->require_f32_tensor("front.features");
                engine::community_models::dettato::detail::HarmonicFeatures front;
                front.batch=1;front.channels=9;front.frames=front_tensor.shape.dims[2];front.values=front_tensor.values;
                const auto notes=reference->require_f32("notes");
                const auto lanes=network.lane_features(front,notes);
                check("lanes",lanes);
                const auto outputs = network.run(inputs);
                const std::array<const char *, 4> names{"inst", "emb", "keep", "h"};
                for (size_t i = 0; i < names.size(); ++i)
                    check(names[i], outputs[i]);
                if (outputs != network.run(inputs))
                    throw std::runtime_error("Grouper same-input reuse changed output");
                if(lanes!=network.lane_features(front,notes)) throw std::runtime_error("Grouper lane/body shared-workspace reuse changed outputs");
                inputs.lanes=lanes;
                const auto composed=network.run(inputs);
                for(size_t i=0;i<names.size();++i) check(names[i],composed[i]);
                const auto power=reference->require_f32_tensor("front.power");
                front.power=power.values;front.extended_bins=power.shape.dims[1];
                const auto prior=assets->grouper->require_f32("hd.prior");
                const auto prepared=engine::community_models::dettato::detail::make_grouper_inputs(front,notes,inputs.taps,prior,lanes);
                check("scalars",prepared.scalars);check("harmonics",prepared.harmonics);check("profile",prepared.profile);
                check("cosine",prepared.cosine);check("sine",prepared.sine);
                const auto prepared_outputs=network.run(prepared);
                for(size_t i=0;i<names.size();++i) check(names[i],prepared_outputs[i]);
                engine::community_models::dettato::detail::HarmonicFrontend frontend(engine::community_models::dettato::detail::load_harmonic_frontend(*assets->grouper, "front.h"), 8);
                const auto native_front=frontend.extract(reference->require_f32("audio"),1,true);
                const auto native_lanes=network.lane_features(native_front,notes);
                const auto native_inputs=engine::community_models::dettato::detail::make_grouper_inputs(native_front,notes,inputs.taps,prior,native_lanes);
                const auto native_outputs=network.run(native_inputs);
                for(size_t i=0;i<names.size();++i) check(names[i],native_outputs[i]);
                if (execution.backend_type() == engine::core::BackendType::Cuda) {
                    for (const int64_t count : {1, 768, 767, 766, 1}) {
                        engine::community_models::dettato::detail::GrouperInputs varied;
                        varied.pitch.assign(count, 60);
                        varied.scalars.assign(count * 11, .1f);
                        varied.harmonics.assign(count * 150, .1f);
                        varied.profile.assign(count * 36, .1f);
                        varied.lanes.assign(count * 256, .1f);
                        varied.taps.assign(count * 1104, .1f);
                        varied.cosine.assign(count * 4 * 22, 1.f);
                        varied.sine.assign(count * 4 * 22, 0.f);
                        network.run(varied);
                        if (native_outputs != network.run(native_inputs))
                            throw std::runtime_error("Grouper shape eviction/reload changed output");
                    }
                }
                engine::debug::timing_log_scalar("session.wall_ms", engine::debug::elapsed_ms(start));
                return 0;
            }
            if (options.at("--network") == "evidence") {
                engine::community_models::dettato::detail::NoteRepairInputs inputs;
                for (float value : reference->require_f32("instrument"))
                    inputs.instrument.push_back(static_cast<int32_t>(value));
                for (float value : reference->require_f32("pitch"))
                    inputs.pitch.push_back(static_cast<int32_t>(value));
                for (float value : reference->require_f32("velocity_bin"))
                    inputs.velocity_bin.push_back(static_cast<int32_t>(value));
                inputs.continuous = reference->require_f32("continuous");
                inputs.evidence = reference->require_f32("evidence");
                inputs.cosine = reference->require_f32("cosine");
                inputs.sine = reference->require_f32("sine");
                engine::community_models::dettato::detail::NoteRepairNetwork network(engine::community_models::dettato::detail::bind_note_repair_weights(*assets->note_repair, execution), execution);
                const auto outputs = network.run(inputs);
                const std::array<const char *, 5> names{"keep", "don", "dur", "ins.0", "ins.1"};
                for (size_t i = 0; i < names.size(); ++i)
                    check(names[i], outputs[i]);
                if (outputs != network.run(inputs))
                    throw std::runtime_error("Evidence same-input reuse changed output");
                if (backend == engine::core::BackendType::Cuda) {
                    const auto resize_rows = [&](size_t count) {
                        auto small = inputs;
                        const auto repeat_rows = [&](auto &dst, const auto &src, size_t width) {
                            dst.resize(count * width);
                            for (size_t row = 0; row < count; ++row)
                                std::copy_n(src.data() + (row % inputs.instrument.size()) * width,
                                            width, dst.data() + row * width);
                        };
                        repeat_rows(small.instrument, inputs.instrument, 1);
                        repeat_rows(small.pitch, inputs.pitch, 1);
                        repeat_rows(small.velocity_bin, inputs.velocity_bin, 1);
                        repeat_rows(small.continuous, inputs.continuous, 110);
                        repeat_rows(small.evidence, inputs.evidence, 769);
                        repeat_rows(small.cosine, inputs.cosine, 14);
                        repeat_rows(small.sine, inputs.sine, 14);
                        return small;
                    };
                    auto changed = inputs;
                    changed.continuous[0] += 0.25f;
                    if (network.run(changed) == outputs)
                        throw std::runtime_error("Evidence cache reused stale input");
                    if (network.run(inputs) != outputs)
                        throw std::runtime_error("Evidence changed-input reuse changed restored output");
                    // Fill the bounded exact-shape cache and evict its original
                    // least-recently-used shape, then check the complete output.
                    for (size_t rows = 1; rows <= 33; ++rows) {
                        if (rows == inputs.instrument.size()) continue;
                        const auto result = network.run(resize_rows(rows));
                        for (const auto &head : result)
                            for (const float value : head)
                                if (!std::isfinite(value)) throw std::runtime_error("Evidence shape reuse is nonfinite");
                    }
                    if (network.run(inputs) != outputs)
                        throw std::runtime_error("Evidence cache eviction/reprepare changed output");
                    // An oversized graph exercises workspace-budget eviction;
                    // returning to the reference shape must still be exact.
                    network.run(resize_rows(1536));
                    if (network.run(inputs) != outputs)
                        throw std::runtime_error("Evidence workspace growth/reprepare changed output");
                }
                const auto source_notes = reference->require_f32("source_notes");
                std::vector<engine::community_models::dettato::Note> notes;
                for (size_t i = 0; i < source_notes.size(); i += 4)
                    notes.push_back({source_notes[i], source_notes[i + 1], static_cast<int>(source_notes[i + 2]), 100,
                                     source_notes[i + 3], 1, false});
                const auto prepared = engine::community_models::dettato::detail::make_note_repair_inputs(notes, inputs.evidence, 0);
                if (prepared.instrument != inputs.instrument || prepared.pitch != inputs.pitch ||
                    prepared.velocity_bin != inputs.velocity_bin)
                    throw std::runtime_error("Evidence note indices differ from upstream");
                check("continuous", prepared.continuous);
                check("cosine", prepared.cosine);
                check("sine", prepared.sine);
                const auto prepared_outputs = network.run(prepared);
                for (size_t i = 0; i < names.size(); ++i)
                    check(names[i], prepared_outputs[i]);
                if (backend == engine::core::BackendType::Cpu) {
                    const auto candidates =
                        engine::community_models::dettato::detail::decode_repair_candidates(prepared, network.run(prepared, true));
                    const auto expected = reference->require_f32("candidates");
                    if (expected.size() != candidates.size() * 5)
                        throw std::runtime_error("Repair candidate count differs");
                    for (size_t i = 0; i < candidates.size(); ++i)
                        if (std::abs(candidates[i].onset - expected[i * 5]) > 1e-5 ||
                            std::abs(candidates[i].offset - expected[i * 5 + 1]) > 1e-5 ||
                            candidates[i].pitch != static_cast<int>(expected[i * 5 + 2]) ||
                            std::abs(candidates[i].confidence - expected[i * 5 + 3]) > 1e-5 ||
                            candidates[i].inserted != (expected[i * 5 + 4] != 0))
                            throw std::runtime_error("Repair candidate differs from upstream");
                    engine::debug::timing_log_scalar("dettato.evidence.matched_candidates", candidates.size());
                }
                engine::debug::timing_log_scalar("session.wall_ms", engine::debug::elapsed_ms(start));
                return 0;
            }
            engine::community_models::dettato::detail::LaneTranscriber network(engine::community_models::dettato::detail::bind_transcriber_weights(*assets->transcriber, execution), execution);
            const auto input = reference->require_f32_tensor("input");
            const auto result = network.run(input.values, input.shape.dims[0], input.shape.dims[2]);
            check("stem", result.stem);
            check("hidden", result.hidden);
            check("logits", result.logits);
            const auto repeated = network.run(input.values, input.shape.dims[0], input.shape.dims[2]);
            if (result.logits != repeated.logits || result.hidden != repeated.hidden || result.stem != repeated.stem)
                throw std::runtime_error("Transcriber same-input reuse changed its output");
            const auto short_input = reference->require_f32_tensor("short.input");
            const auto shorter = network.run(short_input.values, short_input.shape.dims[0], short_input.shape.dims[2]);
            check("short.stem", shorter.stem);
            check("short.hidden", shorter.hidden);
            check("short.logits", shorter.logits);
            const auto restored = network.run(input.values, input.shape.dims[0], input.shape.dims[2]);
            check("logits", restored.logits);
            check("hidden", restored.hidden);
            engine::debug::timing_log_scalar("session.wall_ms", engine::debug::elapsed_ms(start));
            return 0;
        }
        const auto audio = reference->require_f32("audio16");
        engine::community_models::dettato::detail::HarmonicFrontend transcriber(engine::community_models::dettato::detail::load_harmonic_frontend(*assets->transcriber, "front"), 8);
        check("hcqt", transcriber.extract(audio, 1).values);
        engine::community_models::dettato::detail::HarmonicFrontend grouper(engine::community_models::dettato::detail::load_harmonic_frontend(*assets->grouper, "front.h"), 8);
        const auto features = grouper.extract(audio, 1, true);
        check("grouper_front", features.values);
        check("grouper_power", features.power);
        engine::community_models::dettato::detail::DrumFrontend drums(engine::community_models::dettato::detail::load_drum_frontend(*assets->drums), 8);
        check("drum_front", drums.extract(reference->require_f32("audio44")));
        engine::debug::timing_log_scalar("session.wall_ms", engine::debug::elapsed_ms(start));
        return 0;
    } catch (const std::exception &error) {
        engine::debug::timing_log_scalar("session.wall_ms", engine::debug::elapsed_ms(start));
        std::cerr << error.what() << '\n';
        return 1;
    }
}
