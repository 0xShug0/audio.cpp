#include "engine/community_models/coqui_speedy_speech/session.h"

#include "engine/community_models/coqui_speedy_speech/runtime.h"
#include "engine/community_models/inflect_v2/frontend.h"
#include "engine/framework/runtime/options.h"
#include "engine/framework/runtime/spec_backed_model.h"

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <stdexcept>
#include <unordered_map>
#include <utility>

namespace engine::community_models::coqui_speedy_speech {
namespace {
constexpr const char *kFamily = "coqui_speedy_speech";

std::filesystem::path
session_path(const engine::runtime::SessionOptions &options, const char *key) {
  const auto found = options.options.find(key);
  return found == options.options.end() ? std::filesystem::path{}
                                        : std::filesystem::path(found->second);
}

std::vector<std::string> symbols(std::string_view value) {
  std::vector<std::string> out;
  for (size_t i = 0; i < value.size();) {
    const unsigned char ch = static_cast<unsigned char>(value[i]);
    const size_t width = ch < 0x80             ? 1
                         : (ch & 0xe0) == 0xc0 ? 2
                         : (ch & 0xf0) == 0xe0 ? 3
                                               : 4;
    if (i + width > value.size())
      throw std::runtime_error("invalid UTF-8 in SpeedySpeech vocabulary");
    out.emplace_back(value.substr(i, width));
    i += width;
  }
  return out;
}

std::vector<int32_t> tokenize(const Config &config,
                              const std::string &phonemes) {
  auto alphabet = symbols(config.phonemes);
  std::sort(alphabet.begin(), alphabet.end());
  // Coqui's IPAPhonemes vocabulary always reserves PAD, EOS, and BOS in
  // that order.  enable_eos_bos_chars only controls whether EOS/BOS are
  // emitted; it does not remove them from the embedding vocabulary.
  std::vector<std::string> vocab{"_", "~", "^"};
  vocab.insert(vocab.end(), alphabet.begin(), alphabet.end());
  const auto punctuation = symbols(config.punctuations);
  vocab.insert(vocab.end(), punctuation.begin(), punctuation.end());
  if (vocab.size() != static_cast<size_t>(config.vocab_size))
    throw std::runtime_error("SpeedySpeech vocabulary size mismatch");
  std::unordered_map<std::string, int32_t> ids;
  for (size_t i = 0; i < vocab.size(); ++i)
    ids.emplace(vocab[i], static_cast<int32_t>(i));
  std::vector<int32_t> out;
  for (const auto &symbol : symbols(phonemes)) {
    const auto found = ids.find(symbol);
    if (found != ids.end())
      out.push_back(found->second); // Coqui discards OOV phonemes.
  }
  if (out.empty())
    throw std::runtime_error(
        "SpeedySpeech frontend produced no supported phonemes");
  return out;
}

std::string gruut_phonemize(
    const Assets &assets,
    const engine::models::inflect_v2::InflectV2Frontend &fallback,
    const std::string &text) {
  std::string normalized =
      engine::models::inflect_v2::InflectV2Frontend::normalize(text);
  std::string output;
  std::string word;
  const auto append_word = [&] {
    if (word.empty())
      return;
    std::transform(word.begin(), word.end(), word.begin(), [](unsigned char ch) {
      return static_cast<char>(std::tolower(ch));
    });
    auto found = assets.lexicon.find(word);
    std::string phonemes = found != assets.lexicon.end()
                               ? found->second
                               : fallback.phonemize(word);
    for (const auto mark : {std::string("ˈ"), std::string("ˌ")}) {
      for (size_t at; (at = phonemes.find(mark)) != std::string::npos;)
        phonemes.erase(at, mark.size());
    }
    if (!output.empty())
      output.push_back(' ');
    output += phonemes;
    word.clear();
  };
  for (size_t index = 0; index <= normalized.size(); ++index) {
    const char ch = index < normalized.size() ? normalized[index] : '\0';
    if ((ch >= 'A' && ch <= 'Z') || (ch >= 'a' && ch <= 'z') ||
        (ch == '\'' && !word.empty())) {
      word.push_back(ch);
      continue;
    }
    append_word();
    if (ch != '\0' && assets.config.punctuations.find(ch) != std::string::npos &&
        !std::isspace(static_cast<unsigned char>(ch)))
      output.push_back(ch);
  }
  return output;
}
} // namespace

Session::Session(
    engine::runtime::TaskSpec task, engine::runtime::SessionOptions options,
    std::shared_ptr<const Assets> assets,
    std::shared_ptr<const engine::model_spec::ModelContract> contract)
    : RuntimeSessionBase(options), task_(task), assets_(std::move(assets)),
      contract_(std::move(contract)) {
  if (!assets_ || !contract_)
    throw std::runtime_error("SpeedySpeech requires assets and model contract");
  if (task_.task != engine::runtime::VoiceTaskKind::Tts ||
      task_.mode != engine::runtime::RunMode::Offline)
    throw std::runtime_error("SpeedySpeech supports offline TTS only");
  engine::runtime::validate_spec_backed_session_options(
      options, *contract_, kFamily, "SpeedySpeech");
  runtime_ = std::make_unique<NativeRuntime>(assets_, options.backend);
}
Session::~Session() = default;
std::string Session::family() const { return kFamily; }
engine::runtime::VoiceTaskKind Session::task_kind() const { return task_.task; }
engine::runtime::RunMode Session::run_mode() const { return task_.mode; }
void Session::prepare(
    const engine::runtime::SessionPreparationRequest &request) {
  engine::runtime::validate_spec_backed_request_options(
      request.options, *contract_, "SpeedySpeech");
  mark_prepared();
}
engine::runtime::TaskResult
Session::run(const engine::runtime::TaskRequest &request) {
  require_prepared("SpeedySpeech run");
  if (!request.text_input || request.text_input->text.empty())
    throw std::runtime_error("SpeedySpeech requires text input");
  float rate = engine::runtime::parse_finite_float_option(request.options,
                                                          {"speaking_rate"})
                   .value_or(1.0F);
  if (request.voice && request.voice->style &&
      request.voice->style->speaking_rate)
    rate = *request.voice->style->speaking_rate;
  if (rate < 0.5F || rate > 2.0F)
    throw std::runtime_error(
        "SpeedySpeech speaking_rate must be between 0.5 and 2.0");
  engine::models::inflect_v2::InflectV2Frontend frontend(
      session_path(options(), "coqui_speedy_speech.espeak_library_path"),
      session_path(options(), "coqui_speedy_speech.espeak_data_path"));
  const auto phonemes = gruut_phonemize(*assets_, frontend, request.text_input->text);
  engine::runtime::TaskResult result;
  result.audio_output =
      runtime_->synthesize(tokenize(assets_->config, phonemes), rate);
  return result;
}

std::shared_ptr<engine::runtime::IVoiceModelLoader>
make_coqui_speedy_speech_loader() {
  engine::runtime::SpecBackedVoiceModelConfig<Assets> config;
  config.family = kFamily;
  config.load_assets = load_assets;
  config.create_session =
      [](const engine::runtime::TaskSpec &task,
         const engine::runtime::SessionOptions &options,
         std::shared_ptr<const Assets> assets,
         std::shared_ptr<const engine::model_spec::ModelContract> contract) {
        return std::make_unique<Session>(task, options, std::move(assets),
                                         std::move(contract));
      };
  return engine::runtime::make_spec_backed_voice_loader(std::move(config));
}

} // namespace engine::community_models::coqui_speedy_speech
