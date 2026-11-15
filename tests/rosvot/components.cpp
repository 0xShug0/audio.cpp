// Compile the private model implementation into this manual component test.
// This exercises production graphs without adding inference debug hooks.
#include "../../src/models/rosvot/runtime.cpp"
#include <iostream>
#include <cstring>
#include <iomanip>
#include <fstream>
#include "engine/framework/audio/wav_reader.h"
#include "engine/framework/audio/conversion.h"
#ifdef ENGINE_HAS_CUDA_RMVPE
#include "../../src/framework/modules/pitch_extractors/rmvpe_cuda_runtime.h"
#endif

int main(int argc, char ** argv) {
    try {
        std::unordered_map<std::string, std::string> args;
        for (int i = 1; i < argc; ++i) {
            const std::string key = argv[i];
            if (key == "--log") continue;
            if (i + 1 >= argc) throw std::runtime_error("Missing option value");
            args[key] = argv[++i];
        }
        engine::debug::configure_logging({true, args.at("--log-file")});
        engine::core::BackendConfig backend;
        const auto name = args.at("--backend");
        backend.type = name == "cuda" ? engine::core::BackendType::Cuda :
            name == "vulkan" ? engine::core::BackendType::Vulkan : engine::core::BackendType::Cpu;
        backend.threads = 8;
        engine::core::ExecutionContext execution(backend);
        auto assets = engine::models::rosvot::load_rosvot_assets(args.at("--model"));
#ifdef ENGINE_HAS_CUDA_RMVPE
        if (args.count("--native-cell")) {
            namespace c = engine::core;
            namespace m = engine::modules;
            constexpr int H = 256, T = 512;
            c::BackendWeightStore store(execution.backend(), backend.type, "test.rmvpe.cell.weights", 1ull<<20);
            auto w = m::binding::tensor_from_named_source(store, *assets->pitch_weights, "fc.0.gru.weight_hh_l0", engine::assets::TensorStorageType::Native);
            auto b = m::binding::tensor_from_named_source(store, *assets->pitch_weights, "fc.0.gru.bias_hh_l0", engine::assets::TensorStorageType::Native);
            store.upload();
            c::PreparedGraph owner(execution, {"test.rmvpe.cell", 4ull<<20, 1024, 1ull<<20});
            auto & ctx = owner.build_context();
            auto & persist = owner.persistent_build_context();
            auto x = c::make_tensor(persist, GGML_TYPE_F32, c::TensorShape::from_dims({T, 3*H}));
            auto h = c::make_tensor(persist, GGML_TYPE_F32, c::TensorShape::from_dims({1,H}));
            auto k = c::make_tensor(persist, GGML_TYPE_F32, c::TensorShape::from_dims({T,1}));
            auto seq = c::make_tensor(persist, GGML_TYPE_F32, c::TensorShape::from_dims({T,H}));
            ggml_set_input(x.tensor);ggml_set_input(h.tensor);ggml_set_input(k.tensor);
            auto xi = m::SliceModule({0,0,1}).build(ctx,x);
            auto hi = m::LinearModule({H,3*H,true}).build(ctx,h,{w,b});
            auto r = m::SigmoidModule().build(ctx,m::AddModule().build(ctx,
                m::SliceModule({1,0,H}).build(ctx,xi),m::SliceModule({1,0,H}).build(ctx,hi)));
            auto z = m::SigmoidModule().build(ctx,m::AddModule().build(ctx,
                m::SliceModule({1,H,H}).build(ctx,xi),m::SliceModule({1,H,H}).build(ctx,hi)));
            auto n = m::TanhModule().build(ctx,m::AddModule().build(ctx,
                m::SliceModule({1,2*H,H}).build(ctx,xi), m::MulModule().build(ctx,r,m::SliceModule({1,2*H,H}).build(ctx,hi))));
            auto diff = c::wrap_tensor(ggml_sub(ctx.ggml,h.tensor,n.tensor),h.shape);
            auto expected = m::AddModule().build(ctx,n,m::MulModule().build(ctx,z,diff));
            auto final_delta = c::wrap_tensor(ggml_sub(ctx.ggml,expected.tensor,h.tensor),h.shape);
            expected = m::AddModule().build(ctx,h,m::MulModule(true).build(ctx,final_delta,m::SliceModule({0,0,1}).build(ctx,k)));
            ggml_set_output(expected.tensor); ggml_set_output(hi.tensor);
            ggml_build_forward_expand(owner.graph(),expected.tensor);
            owner.allocate_persistent(); owner.allocate();
            std::vector<float> xv(T*3*H), hv(H), kv(T,0);
            for (size_t i=0;i<xv.size();++i) xv[i]=std::sin(float(i)*.13f)*.2f;
            for (int i=0;i<H;++i) hv[i]=std::cos(float(i)*.07f)*.1f;
            kv[0]=1;
            c::write_tensor_f32(x,xv);c::write_tensor_f32(h,hv);c::write_tensor_f32(k,kv);
            owner.compute();
            auto want=c::read_tensor_f32(expected.tensor);
            auto want_project=c::read_tensor_f32(hi.tensor);
            engine::debug::trace_log_f32("cell.expected",{H},want);
            engine::debug::trace_log_f32("cell.expected_projection",{3*H},want_project);
            m::detail::RmvpeCudaGru native(execution.backend(),backend.device,T,H,false,
                x.tensor,w.tensor,b.tensor,h.tensor,k.tensor,seq.tensor);
            native.compute();
            auto got=c::read_tensor_f32(h.tensor);
            engine::debug::trace_log_f32("cell.native",{H},got);
            auto sequence=c::read_tensor_f32(seq.tensor);
            size_t changed=0;double maximum=0;
            for(int i=0;i<H;++i) {changed+=want[i]!=got[i];maximum=std::max(maximum,std::abs(double(want[i])-got[i]));}
            engine::debug::log_message("cell differences="+std::to_string(changed)+" max="+std::to_string(maximum));
            for(int t=0;t<T;++t) for(int i=0;i<H;++i) if(sequence[t*H+i]!=want[i])
                throw std::runtime_error("Native GRU cell differs at frame " + std::to_string(t) + " channel " + std::to_string(i) + " expected " + std::to_string(want[i]) + " got " + std::to_string(sequence[t*H+i]));
            engine::debug::log_message("Native GRU cell exact parity passed");
            return 0;
        }
#endif
        if (args.count("--pitch-inputs")) {
            engine::modules::RmvpePitchExtractorConfig pitch_config;
            pitch_config.cuda_native = args.count("--cuda-native") && args.at("--cuda-native") == "true";
            engine::modules::RmvpePitchExtractorComponent pitch(assets->pitch_weights, backend,
                engine::assets::TensorStorageType::Native, pitch_config);
            const auto inputs = engine::io::json::parse_file(args.at("--pitch-inputs"));
            engine::io::json::Value::Array outputs;
            for (const auto & path : inputs.as_array()) {
                const auto wav = engine::audio::read_wav_f32(std::filesystem::path(path.as_string()));
                auto mono = engine::audio::mixdown_interleaved_to_mono_average(wav.samples, wav.channels);
                engine::audio::TorchaudioSincHannResampleOptions resampling;
                resampling.lowpass_filter_width = 128;
                if (wav.sample_rate != 16000) mono = engine::audio::resample_mono_torchaudio_sinc_hann(
                    mono, wav.sample_rate, 16000, resampling);
                const auto start = std::chrono::steady_clock::now();
                const auto values = pitch.infer_16k_mono(mono, 0.03f, 8);
                engine::debug::timing_log_scalar("session.wall_ms", engine::debug::elapsed_ms(start));
                engine::io::json::Value::Array f0;
                for (const auto value : values) f0.push_back(engine::io::json::Value::make_number(value));
                outputs.push_back(engine::io::json::Value::make_array(std::move(f0)));
            }
            const auto result = engine::io::json::Value::make_array(std::move(outputs));
            std::ofstream output(args.at("--pitch-output"));
            output << engine::io::json::stringify(result) << '\n';
            if (!output) throw std::runtime_error("Failed to write pitch validation output");
            if (args.count("--pitch-reference")) {
                const auto expected = engine::io::json::parse_file(args.at("--pitch-reference"));
                if (engine::io::json::stringify(expected) != engine::io::json::stringify(result))
                    throw std::runtime_error("RMVPE pitch output differs from unchanged component");
            }
            engine::debug::log_message("RMVPE pitch sequence passed");
            return 0;
        }
        auto ref = engine::assets::open_tensor_source(args.at("--reference"));
        auto read_int = [&](const std::string & key) {
            const auto data = ref->require_tensor_data(key);
            if (data.metadata.dtype != "I32" || data.bytes.size() % sizeof(int32_t))
                throw std::runtime_error("Expected I32 fixture tensor: " + key);
            std::vector<int32_t> out(data.bytes.size() / sizeof(int32_t));
            std::memcpy(out.data(), data.bytes.data(), data.bytes.size());
            return out;
        };
        auto controlled = [](engine::core::PreparedGraph & owner) {
            auto * graph = owner.graph();
            for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
                auto * node = ggml_graph_node(graph, i);
                if (node->op == GGML_OP_MUL_MAT) ggml_mul_mat_set_prec(node, GGML_PREC_F32);
            }
        };
        auto compare = [&](const std::string & name, const engine::core::TensorValue & tensor, const std::string & key, bool strict = true) {
            auto expected = ref->require_f32_tensor(key).values;
            std::vector<float> actual(static_cast<size_t>(tensor.shape.num_elements()));
            ggml_backend_tensor_get(tensor.tensor, actual.data(), 0, actual.size() * sizeof(float));
            if (actual.size() != expected.size()) throw std::runtime_error(name + " output shape mismatch");
            double dot = 0, aa = 0, bb = 0, max_error = 0;
            for (size_t i = 0; i < actual.size(); ++i) {
                dot += double(actual[i]) * expected[i]; aa += double(actual[i]) * actual[i];
                bb += double(expected[i]) * expected[i]; max_error = std::max(max_error, std::abs(double(actual[i]) - expected[i]));
            }
            const double cosine = dot / std::sqrt(aa * bb);
            std::ostringstream line; line << std::setprecision(14) << name << " cosine=" << cosine << " max_abs=" << max_error;
            engine::debug::log_message(line.str());
            const double minimum = name.find("pitch_logits") != std::string::npos && args.count("--min-pitch-cosine")
                ? std::stod(args.at("--min-pitch-cosine")) : 0.99999;
            if (!std::isfinite(cosine) || (strict && cosine < minimum)) throw std::runtime_error(name + " failed cosine gate");
        };
        namespace r = engine::models::rosvot;
        const auto mel = ref->require_f32_tensor("mel");
        const auto frames = mel.shape.dims[1];
        const auto pitch = read_int("pitch"), uv = read_int("uv"), words = read_int("word_boundary");
        r::detail::Network w(*assets->word_weights, assets->words, execution, "test.rosvot.words");
        r::detail::FrameGraph wg(execution, w, frames, true);
        controlled(wg.owner);
        wg.run(mel.values, pitch, uv);
        compare("word_logits", wg.boundary, "word_logits");
        wg.run(mel.values, pitch, uv);
        compare("word_logits_repeat", wg.boundary, "word_logits");
        r::detail::Network n(*assets->note_weights, assets->notes, execution, "test.rosvot.notes");
        r::detail::FrameGraph ng(execution, n, frames, false);
        controlled(ng.owner);
        ng.run(mel.values, pitch, uv, words);
        compare("features", ng.feature, "features");
        compare("boundary_logits", ng.boundary, "boundary_logits");
        compare("attention", ng.attention, "attention");
        ng.run(mel.values, pitch, uv, words);
        compare("features_repeat", ng.feature, "features");
        compare("boundary_repeat", ng.boundary, "boundary_logits");
        const auto bd = read_int("note_boundary");
        const auto count = int64_t(1) + std::accumulate(bd.begin(), bd.end(), int64_t(0));
        const auto attn = ref->require_f32_tensor("attention").values;
        std::vector<float> pooling(static_cast<size_t>(count * frames), 0), denom(static_cast<size_t>(count), 1e-5f);
        int64_t note = 0;
        for (int64_t t = 0; t < frames; ++t) { note += bd[static_cast<size_t>(t)]; denom[static_cast<size_t>(note)] += attn[static_cast<size_t>(t)]; }
        note = 0;
        for (int64_t t = 0; t < frames; ++t) { note += bd[static_cast<size_t>(t)]; pooling[static_cast<size_t>(note * frames + t)] = attn[static_cast<size_t>(t)]; }
        r::detail::PitchGraph pg(execution, n, ng.feature, count);
        ggml_backend_tensor_set(pg.pooling.tensor, pooling.data(), 0, pooling.size() * sizeof(float));
        controlled(pg.owner);
        ggml_backend_tensor_set(pg.denominator.tensor, denom.data(), 0, denom.size() * sizeof(float));
        pg.owner.compute();
        compare("propagated_pitch_logits", pg.logits, "note_logits");
        auto frozen = ref->require_f32_tensor("features").values;
        ggml_backend_tensor_set(ng.feature.tensor, frozen.data(), 0, frozen.size() * sizeof(float));
        ggml_backend_tensor_set(pg.pooling.tensor, pooling.data(), 0, pooling.size() * sizeof(float));
        ggml_backend_tensor_set(pg.denominator.tensor, denom.data(), 0, denom.size() * sizeof(float));
        pg.owner.compute();
        compare("isolated_pitch_logits", pg.logits, "note_logits");
        engine::debug::log_message("ROSVOT component parity passed");
    } catch (const std::exception & e) { std::cerr << e.what() << '\n'; return 1; }
    return 0;
}
