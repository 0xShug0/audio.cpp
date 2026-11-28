#include "engine/models/kugelaudio/reference_encoder.h"
#include "engine/framework/runtime/errors.h"
#include "test_assert.h"
#include <cmath>
#include <iostream>
#include <limits>
#include <string>
#include <utility>
#include <vector>

int main() {
    try {
        using engine::runtime::AudioBuffer;
        using engine::models::kugelaudio::prepare_reference_audio;
        auto check = [](bool condition) { engine::test::require(condition, "KugelAudio reference frontend"); };
        AudioBuffer mono{24000, 1, std::vector<float>(2400)};
        for (size_t i = 0; i < mono.samples.size(); ++i) mono.samples[i] = float(0.3 * std::sin(i * 0.03));
        const auto input = mono.samples;
        auto normalized = prepare_reference_audio(mono);
        check(mono.samples == input && normalized.size() == input.size());
        double energy = 0;
        for (float value : normalized) energy += double(value) * value;
        engine::test::require_close(float(std::sqrt(energy / normalized.size())), float(std::pow(10., -25. / 20.)), 1e-6F, "reference level");
        AudioBuffer stereo{24000, 2, {}};
        for (float sample : input) { stereo.samples.push_back(sample); stereo.samples.push_back(sample); }
        check(prepare_reference_audio(stereo) == normalized);
        AudioBuffer lower_rate = mono;
        lower_rate.sample_rate = 12000;
        check(prepare_reference_audio(lower_rate).size() == 2 * normalized.size());
        AudioBuffer impulse{24000, 1, std::vector<float>(24000, 0)};
        impulse.samples[100] = 1;
        for (float value : prepare_reference_audio(impulse)) check(std::isfinite(value) && std::abs(value) <= 1);
        auto reject = [&](const AudioBuffer & audio) {
            bool failed = false;
            try { (void)prepare_reference_audio(audio); } catch (const engine::runtime::InvalidRequestError &) { failed = true; }
            check(failed);
        };
        reject({24000, 1, {}});
        reject({0, 1, {0.1F}});
        reject({24000, 0, {0.1F}});
        reject({24000, 2, {0.1F}});
        reject({24000, 1, std::vector<float>(100, 0)});
        reject({24000, 1, {std::numeric_limits<float>::quiet_NaN()}});
        reject({24000, 1, {std::numeric_limits<float>::infinity()}});
        check(prepare_reference_audio({24000, 1, std::vector<float>(30 * 24000, 0.1F)}).size() == 30 * 24000);
        reject({24000, 1, std::vector<float>(30 * 24000 + 1, 0.1F)});
        namespace json = engine::io::json;
        using engine::models::kugelaudio::parse_reference_encoder_config;
        const auto encoder = json::parse(R"({"causal":true,"channels":1,"conv_bias":true,
            "disable_last_norm":true,"layernorm_elementwise_affine":true,"layernorm":"RMSNorm",
            "mixer_layer":"depthwise_conv","conv_norm":"none","pad_mode":"constant",
            "vae_dim":64,"encoder_n_filters":32,"layernorm_eps":0.00001,"fix_std":0.5,
            "encoder_ratios":[8,5,5,4,2,2],"encoder_depths":"3-3-3-3-3-3-8"})").as_object();
        const auto parsed = parse_reference_encoder_config(json::Value::make_object(encoder));
        check(parsed.latent_size == 64 && parsed.filters == 32 && parsed.depths.back() == 8 && parsed.fixed_std == 0.5F);
        auto array_depths = encoder;
        array_depths["encoder_depths"] = json::parse("[3,3,3,3,3,3,8]");
        check(parse_reference_encoder_config(json::Value::make_object(array_depths)).depths == parsed.depths);
        const std::vector<std::pair<std::string, json::Value>> invalid_config = {
            {"causal", json::Value::make_bool(false)},
            {"channels", json::Value::make_number(2)},
            {"encoder_depths", json::Value::make_string("3-3-3-3-3-3-8junk")},
            {"encoder_depths", json::parse("[64,64,3,3,3,3,8]")},
            {"encoder_ratios", json::parse("[8,5,5,4,2,0]")},
            {"encoder_ratios", json::parse("[8,5,5,4,2,1073741824]")},
            {"encoder_n_filters", json::Value::make_number(std::ldexp(1.0, 62))},
            {"layernorm_eps", json::Value::make_number(0)},
            {"fix_std", json::Value::make_number(-1)},
        };
        for (const auto & [key, value] : invalid_config) {
            auto invalid = encoder;
            invalid[key] = value;
            bool failed = false;
            try { (void)parse_reference_encoder_config(json::Value::make_object(std::move(invalid))); }
            catch (const std::exception &) { failed = true; }
            check(failed);
        }
        std::cout << "KugelAudio reference frontend/config tests passed\n";
        return 0;
    } catch (const std::exception & error) { std::cerr << error.what() << '\n'; return 1; }
}
