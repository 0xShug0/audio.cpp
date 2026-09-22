// ZipVoice parity harness against the moonlight fixture produced by the
// zipvoice-lite MLX port (zipvoice/mlx/fixtures/moonlight_distill_case.npz).
// The fixture stores the PyTorch reference inputs and outputs for the
// ZipVoice-Distill checkpoint:
//   tokens, prompt_tokens, prompt_features(+lens), x0, x1 (full sample),
//   text_condition (text encoder + duration prediction output).
// Usage: zipvoice_parity <fixture.npz> <model-path>
#include "engine/community_models/zipvoice/synthesize.h"
#include "engine/framework/audio/wav_writer.h"
#include "engine/framework/model_spec/package.h"

#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <fstream>
#include <filesystem>
#include <optional>
#include <cstdlib>
#include <string>
#include <vector>
#include <algorithm>

namespace {

// ---- minimal ZIP (STORED) + .npy reader -----------------------------------

struct NpyArray {
    std::string name;
    std::vector<int64_t> shape;
    char typechar = 'f';  // 'f' float32, 'i' int32
    std::vector<uint8_t> data;
};

uint16_t rd16(const uint8_t * p) { return uint16_t(p[0]) | (uint16_t(p[1]) << 8); }
uint32_t rd32(const uint8_t * p) {
    return uint32_t(p[0]) | (uint32_t(p[1]) << 8) | (uint32_t(p[2]) << 16) | (uint32_t(p[3]) << 24);
}

bool parse_npy(const uint8_t * raw, size_t size, NpyArray & arr) {
    if (size < 10 || std::memcmp(raw, "\x93NUMPY", 6) != 0) {
        std::fprintf(stderr, "bad npy magic\n");
        std::exit(2);
    }
    const uint16_t header_len = rd16(raw + 8);
    const std::string header(reinterpret_cast<const char *>(raw + 10), header_len);
    const size_t data_offset = 10 + header_len;
    // 'descr': '<f4' / '<i4' ; 'fortran_order': False ; 'shape': (a, b)
    const auto descr = header.find("'descr'");
    const auto descr_str = header.substr(descr, 30);
    if (descr_str.find("f4") != std::string::npos) arr.typechar = 'f';
    else if (descr_str.find("i4") != std::string::npos) arr.typechar = 'i';
    else {
        // skip bool/other dtypes (padding_mask etc.)
        return false;
    }
    const auto shape_pos = header.find("'shape'");
    size_t p = header.find('(', shape_pos) + 1;
    while (p < header.size() && header[p] != ')') {
        if (std::isdigit(static_cast<unsigned char>(header[p]))) {
            int64_t v = 0;
            while (std::isdigit(static_cast<unsigned char>(header[p]))) {
                v = v * 10 + (header[p] - '0');
                ++p;
            }
            arr.shape.push_back(v);
        } else {
            ++p;
        }
    }
    arr.data.assign(raw + data_offset, raw + size);
    return true;
}

std::vector<NpyArray> load_npz(const std::string & path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { std::fprintf(stderr, "cannot open %s\n", path.c_str()); std::exit(1); }
    std::vector<uint8_t> zip((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    // find EOCD
    size_t eocd = zip.size() - 22;
    while (eocd > 0 && rd32(&zip[eocd]) != 0x06054b50) --eocd;
    if (rd32(&zip[eocd]) != 0x06054b50) { std::fprintf(stderr, "bad npz\n"); std::exit(2); }
    const uint16_t entries = rd16(&zip[eocd + 10]);
    const uint32_t cd_offset = rd32(&zip[eocd + 16]);
    // collect (name, local offset) in directory order
    struct Entry { std::string name; uint32_t local; };
    std::vector<Entry> directory;
    uint32_t off = cd_offset;
    for (uint16_t i = 0; i < entries; ++i) {
        if (rd32(&zip[off]) != 0x02014b50) break;
        const uint16_t name_len = rd16(&zip[off + 28]);
        const uint16_t extra_len = rd16(&zip[off + 30]);
        const uint16_t comment_len = rd16(&zip[off + 32]);
        const uint32_t local = rd32(&zip[off + 42]);
        directory.push_back({std::string(reinterpret_cast<char *>(&zip[off + 46]), name_len), local});
        off += 46 + name_len + extra_len + comment_len;
    }
    std::vector<NpyArray> arrays;
    for (size_t i = 0; i < directory.size(); ++i) {
        const uint32_t local = directory[i].local;
        const std::string & name = directory[i].name;
        uint32_t csize = rd32(&zip[local + 18]);
        const size_t data_start = local + 30 + rd16(&zip[local + 26]) + rd16(&zip[local + 28]);
        if (csize == 0xFFFFFFFFu) {
            // ZIP64 sentinel: derive the size from the neighbouring entry
            // (STORED entries are laid out contiguously).
            const size_t next = (i + 1 < directory.size()) ? directory[i + 1].local : cd_offset;
            if (next < data_start) { std::fprintf(stderr, "bad npz entry %s\n", name.c_str()); std::exit(2); }
            csize = static_cast<uint32_t>(next - data_start);
        }
        if (name.size() > 4 && name.substr(name.size() - 4) == ".npy") {
            NpyArray arr;
            if (parse_npy(&zip[data_start], csize, arr)) {
                arr.name = name.substr(0, name.size() - 4);
                arrays.push_back(std::move(arr));
            }
        }
    }
    return arrays;
}

double cosine(const std::vector<float> & a, const std::vector<float> & b) {
    double dot = 0, na = 0, nb = 0;
    const size_t n = std::min(a.size(), b.size());
    for (size_t i = 0; i < n; ++i) {
        dot += double(a[i]) * double(b[i]);
        na += double(a[i]) * double(a[i]);
        nb += double(b[i]) * double(b[i]);
    }
    return dot / (std::sqrt(na) * std::sqrt(nb) + 1e-12);
}

double max_abs_diff(const std::vector<float> & a, const std::vector<float> & b) {
    double m = 0;
    const size_t n = std::min(a.size(), b.size());
    for (size_t i = 0; i < n; ++i) m = std::max(m, double(std::fabs(a[i] - b[i])));
    return m;
}

bool check(const std::string & name, const std::vector<float> & got,
           const std::vector<float> & want, float atol, float rtol) {
    bool ok = !got.empty() && got.size() == want.size();
    double squared = 0;
    for (size_t i = 0; i < std::min(got.size(), want.size()); ++i) {
        const double diff = std::abs(double(got[i]) - want[i]);
        squared += diff * diff;
        ok = ok && std::isfinite(got[i]) && std::isfinite(want[i]) &&
            diff <= atol + rtol * std::abs(want[i]);
    }
    std::printf("%-24s size=%zu/%zu cosine=%.9f maxdiff=%.6g rmse=%.6g %s\n",
                name.c_str(), got.size(), want.size(), cosine(got, want),
                max_abs_diff(got, want), std::sqrt(squared / std::max<size_t>(1, got.size())),
                ok ? "OK" : "FAIL");
    return ok;
}

}  // namespace

int main(int argc, char ** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: %s <fixture.npz> <model-path> [taps.npz] [vocos-path] [output.wav]\n", argv[0]);
        return 2;
    }
    const auto arrays = load_npz(argv[1]);
    // Arrays written by zipvoice/mlx/fixtures/build_fixture.py. Keyed by name:
    // tokens, prompt_tokens, prompt_features(+lens), text_condition, x0, x1.
    const auto find_arr = [&](const char * key) -> const NpyArray & {
        for (const auto & a : arrays) {
            if (a.name == key) return a;
        }
        std::fprintf(stderr, "fixture missing array '%s'\n", key);
        std::exit(2);
    };
    const auto & tokens_arr = find_arr("tokens");
    const auto & prompt_tokens_arr = find_arr("prompt_tokens");
    const auto & prompt_features = find_arr("prompt_features");
    const auto & prompt_features_lens = find_arr("prompt_features_lens");
    const auto & text_condition = find_arr("text_condition");
    const auto & x0 = find_arr("x0");
    const auto & x1 = find_arr("x1");

    std::vector<int32_t> tokens(tokens_arr.data.size() / 4);
    std::memcpy(tokens.data(), tokens_arr.data.data(), tokens_arr.data.size());
    std::vector<int32_t> prompt_tokens(prompt_tokens_arr.data.size() / 4);
    std::memcpy(prompt_tokens.data(), prompt_tokens_arr.data.data(), prompt_tokens_arr.data.size());
    int64_t prompt_len = 0;
    std::memcpy(&prompt_len, prompt_features_lens.data.data(), sizeof(int32_t));

    const auto to_floats = [](const NpyArray & a) {
        std::vector<float> v(a.data.size() / 4);
        std::memcpy(v.data(), a.data.data(), a.data.size());
        return v;
    };

    const std::string model = argv[2];
    std::optional<engine::assets::ResourceBundle> bundle;
    if (std::filesystem::path(model).extension() == ".gguf") {
        bundle = engine::model_spec::load_resource_bundle_for_family(model, "zipvoice");
    }
    const auto * resources = bundle ? &*bundle : nullptr;
    bool checks_ok = true;
    engine::models::zipvoice::ZipVoiceComputeDevice device;
    device.backend_type = engine::core::BackendType::Cpu;  // parity: bit-stable reference path
    device.threads = 8;

    // ---- stage 0: raw text encoder vs torch taps (when provided) ----
    if (argc > 3) {
        const auto taps = load_npz(argv[3]);
        const auto find_tap_opt = [&](const char * key) -> const NpyArray * {
            for (const auto & a : taps) {
                if (a.name == key) return &a;
            }
            return nullptr;
        };
        // torch embed is [1, S+1, feat]; row-major == [S+1, F] rows (optional:
        // the layer0 tap file lacks it)
        const NpyArray * embed_ref = find_tap_opt("embed");
        // reference runs the encoder on prompt_tokens + tokens (pad appended
        // inside the hook)
        std::vector<int32_t> cat_ids(prompt_tokens);
        cat_ids.insert(cat_ids.end(), tokens.begin(), tokens.end());
        engine::models::zipvoice::ZipVoiceLayerTaps got_taps;
        const auto got_embed = engine::models::zipvoice::zipvoice_text_encoder_raw(
            model, cat_ids, &got_taps, device, resources);
        if (embed_ref != nullptr) {
            const auto want_embed = to_floats(*embed_ref);
            checks_ok &= check("text_encoder_raw", got_embed, want_embed, 2e-6F, 2e-5F);
        }
        // per-layer outputs (torch (S+1, 1, C) rows) for bisecting
        for (size_t li = 0; li < got_taps.layers.size(); ++li) {
            const std::string name = "layer" + std::to_string(li);
            const NpyArray * ref = find_tap_opt(name.c_str());
            if (ref == nullptr) continue;
            const auto want = to_floats(*ref);
            const auto & got = got_taps.layers[li];
            checks_ok &= check(name, got, want, 3e-5F, 2e-5F);
        }
        const size_t stage_count = got_taps.stages.size();
        const char * got_names[] = {
            "layer_in", "attn_qtb", "attn_ktb", "attn_inproj", "attn_scores",
            "attn_pos", "attn_w", "ff1_inproj", "ff1_act", "ff1", "na",
            "na_gated", "na_gtb", "na_attnrep", "na_attended", "na_mul", "na_y2", "na_h", "sa1",
            "conv1", "ff2", "bypass_mid", "sa2", "conv2", "ff3", "norm"};
        if (stage_count != sizeof(got_names) / sizeof(got_names[0])) {
            throw std::runtime_error("text stage tap/name count mismatch");
        }
        for (size_t si = 0; si < stage_count; ++si) {
            const auto & got = got_taps.stages[si];
            const char * ref_key = nullptr;
            // map got -> ref by semantic name
            const char * gn = got_names[si];
            if (std::strcmp(gn, "layer_in") == 0) ref_key = "layer_in";
            else if (std::strcmp(gn, "attn_qtb") == 0) ref_key = "attn_qtb_ref";
            else if (std::strcmp(gn, "attn_ktb") == 0) ref_key = "attn_ktb_ref";
            else if (std::strcmp(gn, "attn_inproj") == 0) ref_key = "attn_inproj_ref";
            else if (std::strcmp(gn, "attn_scores") == 0) ref_key = "attn_scores_ref";
            else if (std::strcmp(gn, "attn_pos") == 0) ref_key = "attn_pos_ref";
            else if (std::strcmp(gn, "attn_w") == 0) ref_key = "self_attn_weights";
            else if (std::strcmp(gn, "ff1_inproj") == 0) ref_key = "ff1_inproj";
            else if (std::strcmp(gn, "ff1_act") == 0) ref_key = "ff1_act";
            else if (std::strcmp(gn, "ff1") == 0) ref_key = "feed_forward1";
            else if (std::strcmp(gn, "na") == 0) ref_key = "nonlin_attention";
            else if (std::strcmp(gn, "na_gated") == 0) ref_key = "na_gated_ref";
            else if (std::strcmp(gn, "na_gtb") == 0) ref_key = "na_gtb_ref";
            else if (std::strcmp(gn, "na_mul") == 0) ref_key = "na_mul_ref";
            else if (std::strcmp(gn, "na_y2") == 0) ref_key = "na_y2_ref";
            else if (std::strcmp(gn, "na_h") == 0) ref_key = "na_h_ref";
            else if (std::strcmp(gn, "na_attnrep") == 0) ref_key = "self_attn_weights";
            else if (std::strcmp(gn, "na_attended") == 0) ref_key = "na_attended_ref";
            else if (std::strcmp(gn, "sa1") == 0) ref_key = "self_attn1";
            else if (std::strcmp(gn, "conv1") == 0) ref_key = "conv_module1";
            else if (std::strcmp(gn, "ff2") == 0) ref_key = "feed_forward2";
            else if (std::strcmp(gn, "bypass_mid") == 0) ref_key = "bypass_mid";
            else if (std::strcmp(gn, "sa2") == 0) ref_key = "self_attn2";
            else if (std::strcmp(gn, "conv2") == 0) ref_key = "conv_module2";
            else if (std::strcmp(gn, "ff3") == 0) ref_key = "feed_forward3";
            else if (std::strcmp(gn, "norm") == 0) ref_key = "norm";
            if (ref_key == nullptr) continue;
            // torch attention weights are (H, B, T, T); got is [S*T? ] handled below
            if (getenv("ZV_DUMP") != nullptr) {
                const std::string path = std::string("/tmp/zv_mine_") + gn + ".bin";
                std::ofstream f(path, std::ios::binary);
                f.write(reinterpret_cast<const char *>(got.data()),
                        static_cast<std::streamsize>(got.size() * sizeof(float)));
            }
            const NpyArray * ref = find_tap_opt(ref_key);
            if (ref == nullptr) continue;
            auto want = to_floats(*ref);
            if ((std::strcmp(gn, "attn_qtb") == 0 || std::strcmp(gn, "attn_ktb") == 0) && ref->shape.size() == 3) {
                const int64_t D = ref->shape[0], T = ref->shape[1], H = ref->shape[2];
                auto reordered = want;
                for (int64_t d = 0; d < D; ++d)
                for (int64_t t = 0; t < T; ++t)
                for (int64_t h = 0; h < H; ++h)
                    reordered[d + D * (t + T * h)] = want[h + H * (t + T * d)];
                want = std::move(reordered);
            }
            if (std::strcmp(gn, "na_attended") == 0 && ref->shape.size() == 4) {
                const int64_t H = ref->shape[0], B = ref->shape[1], T = ref->shape[2], D = ref->shape[3];
                auto reordered = want;
                for (int64_t h = 0; h < H; ++h)
                for (int64_t b = 0; b < B; ++b)
                for (int64_t t = 0; t < T; ++t)
                for (int64_t d = 0; d < D; ++d)
                    reordered[t + T * (d + D * (h + H * b))] = want[d + D * (t + T * (b + B * h))];
                want = std::move(reordered);
            }
            if (std::strcmp(gn, "na_attnrep") == 0 && ref->shape.size() == 4 && ref->shape[1] == 1) {
                want.resize(static_cast<size_t>(ref->shape[2] * ref->shape[3]));
            }
            // attention weights need a 4D reorder: torch (H, B, tgt, src)
            // row-major vs ggml [src (ne0), tgt, H, B]
            if (std::strcmp(gn, "attn_w") == 0) {
                const int64_t H = ref->shape[0], B = ref->shape[1];
                const int64_t TT = ref->shape[2], TS = ref->shape[3];
                std::vector<float> reordered(want.size(), 0.0F);
                for (int64_t h = 0; h < H; ++h)
                for (int64_t b = 0; b < B; ++b)
                for (int64_t tt = 0; tt < TT; ++tt)
                for (int64_t ts = 0; ts < TS; ++ts) {
                    const size_t torch_flat = static_cast<size_t>(
                        h * B * TT * TS + b * TT * TS + tt * TS + ts);
                    const size_t mine_flat = static_cast<size_t>(
                        ts + tt * TS + h * TS * TT + b * TS * TT * H);
                    reordered[mine_flat] = want[torch_flat];
                }
                want = std::move(reordered);
            }
            if (got.size() != want.size()) {
                std::printf("  stage %-18s: size cpp=%zu ref=%zu SKIP\n", gn, got.size(), want.size());
                checks_ok = false;
                continue;
            }
            checks_ok &= check(gn, got, want, 3e-5F, 2e-5F);

        }
    }

    // ---- stage 1: text_condition ----
    const auto want_text = to_floats(text_condition);
    const auto got_text = engine::models::zipvoice::zipvoice_text_condition(
        model, tokens, prompt_tokens, prompt_len, 1.0F, device, resources);
    std::printf("text_condition: cpp=%zu ref=%zu cosine=%.6f maxdiff=%.4g\n",
                got_text.size(), want_text.size(),
                cosine(got_text, want_text), max_abs_diff(got_text, want_text));

    // ---- stage 2: full sample (distill, 8 steps, guidance 3.0, t_shift 0.5) ----
    const auto prompt_feats = to_floats(prompt_features);
    const auto want_x1 = to_floats(x1);
    const auto got_x1 = engine::models::zipvoice::zipvoice_sample(
        model, tokens, prompt_tokens, prompt_feats, prompt_len,
        to_floats(x0), 8, 3.0F, 0.5F, 1.0F, device, resources);
    std::printf("x1:            cpp=%zu ref=%zu cosine=%.6f maxdiff=%.4g\n",
                got_x1.size(), want_x1.size(),
                cosine(got_x1, want_x1), max_abs_diff(got_x1, want_x1));

    const bool text_ok = check("text_condition", got_text, want_text, 2e-6F, 2e-5F);
    const bool x1_ok = check("sample", got_x1, want_x1, 3e-4F, 2e-4F);
    for (const auto & array : arrays) {
        if (array.name.rfind("velocity_", 0) != 0) continue;
        int length = 0, step = 0;
        if (std::sscanf(array.name.c_str(), "velocity_%d_%d", &length, &step) != 2) return 2;
        auto noise = to_floats(x0);
        auto text = want_text;
        auto speech = to_floats(find_arr("speech_condition"));
        noise.resize(length * 100);
        text.resize(length * 100);
        speech.resize(length * 100);
        const auto velocity = engine::models::zipvoice::zipvoice_velocity(
            model, noise, text, speech, length, step == 0 ? 0.0F : 0.6F, 3.0F, device, 1, resources);
        checks_ok &= check(array.name, velocity, to_floats(array), 3e-4F, 2e-4F);
    }
    for (const auto & array : arrays) {
        if (array.name == "batch_velocity") {
            checks_ok &= check("batch_velocity", engine::models::zipvoice::zipvoice_velocity(
                model, to_floats(find_arr("batch_x")), to_floats(find_arr("batch_text")),
                to_floats(find_arr("batch_speech")), array.shape[1], 0.6F, 3.0F, device, 2, resources),
                to_floats(array), 3e-4F, 2e-4F);
        }
        if (array.name == "fbank_mel") {
            checks_ok &= check("fbank", engine::models::zipvoice::zipvoice_logmel(
                to_floats(find_arr("fbank_wav"))), to_floats(array), 5e-5F, 2e-5F);
        }
        if (array.name == "vocos_audio") {
            const std::string vocos = argc > 4 ? argv[4] : model;
            checks_ok &= check("vocos", engine::models::zipvoice::zipvoice_vocos_decode(
                vocos, to_floats(find_arr("vocos_mel"))), to_floats(array), 2e-4F, 2e-3F);
        }
    }
    std::printf("%s %s\n", text_ok ? "TEXT_CONDITION OK" : "TEXT_CONDITION FAIL",
                x1_ok ? "SAMPLE OK" : "SAMPLE FAIL");
    if (argc > 5 && text_ok && x1_ok && checks_ok) {
        std::vector<float> mel(got_x1.begin() + prompt_len * 100, got_x1.end());
        for (float & value : mel) value /= 0.1F;
        auto audio = engine::models::zipvoice::zipvoice_vocos_decode(argv[4], mel);
        double energy = 0;
        for (float value : audio) {
            if (!std::isfinite(value)) throw std::runtime_error("non-finite audio sample");
            energy += double(value) * value;
        }
        if (audio.empty() || energy / audio.size() < 1e-8) throw std::runtime_error("silent synthesis output");
        engine::audio::write_pcm16_wav(argv[5], 24000, 1, audio);
        std::printf("Wrote %s: %.3f seconds, RMS %.6f\n", argv[5], audio.size() / 24000.0,
                    std::sqrt(energy / audio.size()));
    }
    return (text_ok && x1_ok && checks_ok) ? 0 : 1;
}
