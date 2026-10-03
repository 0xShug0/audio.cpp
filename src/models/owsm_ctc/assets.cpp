#include "engine/models/owsm_ctc/model.h"

#include "engine/framework/io/json.h"
#include "engine/framework/model_spec/package.h"

#include <stdexcept>

namespace engine::models::owsm_ctc {
namespace {

int32_t model_id_from_sentencepiece(int32_t id) {
    if (id == 0) {
        return 1;
    }
    if (id >= 3 && id < 50000) {
        return id - 1;
    }
    throw std::runtime_error("OWSM-CTC v4 SentencePiece produced a reserved BOS or EOS token");
}

int32_t sentencepiece_id_from_model(int32_t id) {
    if (id == 1) {
        return 0;
    }
    if (id >= 2 && id <= 49998) {
        return id + 1;
    }
    return -1;
}

}  // namespace

int32_t OWSMCTCV4Assets::token_id(const std::string & token) const {
    if (token == "<blank>") {
        return config.blank_id;
    }
    for (const auto & piece : sentencepiece) {
        if (piece.text == token) {
            return model_id_from_sentencepiece(piece.id);
        }
    }
    throw std::runtime_error("OWSM-CTC v4 tokenizer is missing " + token);
}

std::string OWSMCTCV4Assets::decode_visible(const std::vector<int32_t> & ids) const {
    std::vector<int32_t> visible;
    visible.reserve(ids.size());
    for (const auto id : ids) {
        const auto sentencepiece_id = sentencepiece_id_from_model(id);
        if (sentencepiece_id < 0) {
            continue;
        }
        const auto & text = sentencepiece[static_cast<size_t>(sentencepiece_id)].text;
        if (text.size() >= 2 && text.front() == '<' && text.back() == '>') {
            continue;
        }
        visible.push_back(sentencepiece_id);
    }
    return tokenizers::decode_sentencepiece(sentencepiece, visible);
}

std::shared_ptr<const OWSMCTCV4Assets> load_owsm_ctc_assets(const std::filesystem::path & path) {
    auto out = std::make_shared<OWSMCTCV4Assets>();
    out->resources = model_spec::load_resource_bundle(path, model_spec::default_spec_path("owsm_ctc"));
    out->source = out->resources.open_tensor_source("weights");
    const auto config = out->resources.parse_json("config");
    auto & model = out->config;
    model.variant = io::json::require_string(config, "variant");
    model.prompt_hidden_size = io::json::require_i64(config, "prompt_hidden_size");
    model.prompt_num_heads = io::json::require_i64(config, "prompt_num_heads");
    model.prompt_layers = io::json::require_i64(config, "prompt_layers");
    model.prompt_intermediate_size = io::json::require_i64(config, "prompt_intermediate_size");
    model.interctc_layers = io::json::require_i64_array(config, "interctc_layers");
    model.cross_attention_layers = io::json::require_i64_array(config, "cross_attention_layers");
    model.hidden_size = io::json::require_i64(config, "hidden_size");
    model.num_heads = io::json::require_i64(config, "num_heads");
    model.encoder_layers = io::json::require_i64(config, "encoder_layers");
    model.intermediate_size = io::json::require_i64(config, "intermediate_size");
    model.vocabulary_size = io::json::require_i64(config, "vocabulary_size");
    model.max_audio_samples = io::json::require_i64(config, "max_audio_samples");
    model.frontend_frames = io::json::require_i64(config, "frontend_frames");
    model.encoder_frames = io::json::require_i64(config, "encoder_frames");
    if (model.variant != "medium" || model.hidden_size != 1024 || model.num_heads != 16 ||
        model.encoder_layers != 27 || model.intermediate_size != 4096 ||
        model.vocabulary_size != 50002 || model.max_audio_samples != 480000 ||
        model.frontend_frames != 3001 || model.encoder_frames != 374 ||
        model.prompt_hidden_size != 512 || model.prompt_num_heads != 8 || model.prompt_layers != 4 ||
        model.prompt_intermediate_size != 2048 ||
        model.interctc_layers != std::vector<int64_t>{6, 12, 15, 21} ||
        model.cross_attention_layers != std::vector<int64_t>{2, 5, 8, 11, 14, 17, 20, 23, 26}) {
        throw std::runtime_error("OWSM-CTC v4 config does not describe the supported 1B checkpoint");
    }

    out->sentencepiece = tokenizers::load_sentencepiece_model(out->resources.require_file("tokenizer"));
    if (out->sentencepiece.size() != 50000) {
        throw std::runtime_error("OWSM-CTC v4 tokenizer does not match the published 50,000-piece vocabulary");
    }

    audio::MelSpectrogramFrontendConfig frontend;
    frontend.sample_rate = 16000;
    frontend.n_fft = 512;
    frontend.hop_length = 160;
    frontend.win_length = 400;
    frontend.n_mels = 128;
    frontend.mel_fmax = 8000.0F;
    frontend.filterbank_normalization = audio::MelFilterbankNormalization::Slaney;
    frontend.stft_pad_mode = audio::STFTPadMode::Reflect;
    frontend.window = audio::MelHannWindow::Periodic;
    frontend.stft_center = true;
    frontend.waveform_padding = audio::MelWaveformPadding::None;
    frontend.filterbank_projection = audio::MelFilterbankProjection::DenseF32;
    frontend.spectrum_mode = audio::MelSpectrumMode::PowerBeforeProjection;
    frontend.log_floor = 1.0e-10;
    frontend.log_precision = audio::MelLogPrecision::F32;
    frontend.value_transform = audio::MelValueTransform::Ln;
    frontend.layout = audio::MelOutputLayout::TimeMajor;
    frontend.require_mono = true;
    frontend.minimum_samples = 400;
    out->frontend = audio::get_cached_mel_spectrogram_frontend(frontend);
    out->feature_mean = out->source->require_f32("normalize.mean", {128});
    out->feature_std = out->source->require_f32("normalize.std", {128});
    return out;
}

}  // namespace engine::models::owsm_ctc
