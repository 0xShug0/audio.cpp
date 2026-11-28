#include "engine/models/irodori_tts/tokenizer_text.h"

#include "engine/framework/io/json.h"
#include "engine/framework/tokenizers/llama_bpe.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace engine::models::irodori_tts {
namespace {

constexpr const char * kSentencePieceSpace = "\xE2\x96\x81";

std::shared_ptr<const IrodoriTTSAssets> require_assets(std::shared_ptr<const IrodoriTTSAssets> assets) {
    if (assets == nullptr) {
        throw std::runtime_error("Irodori-TTS text tokenizer requires assets");
    }
    return assets;
}

std::string normalize_text_for_llm_jp(std::string text) {
    text.insert(0, kSentencePieceSpace);
    size_t pos = 0;
    while ((pos = text.find(' ', pos)) != std::string::npos) {
        text.replace(pos, 1, kSentencePieceSpace);
        pos += 3;
    }
    return text;
}

std::string normalize_text_for_metaspace(std::string text) {
    size_t pos = 0;
    while ((pos = text.find(' ', pos)) != std::string::npos) {
        text.replace(pos, 1, kSentencePieceSpace);
        pos += 3;
    }
    return text;
}

std::vector<std::string> utf8_chars(const std::string & text) {
    std::vector<std::string> chars;
    for (size_t i = 0; i < text.size();) {
        const unsigned char ch = static_cast<unsigned char>(text[i]);
        size_t width = 1;
        if ((ch & 0x80U) == 0) {
            width = 1;
        } else if ((ch & 0xE0U) == 0xC0U) {
            width = 2;
        } else if ((ch & 0xF0U) == 0xE0U) {
            width = 3;
        } else if ((ch & 0xF8U) == 0xF0U) {
            width = 4;
        } else {
            throw std::runtime_error("Irodori-TTS tokenizer received invalid UTF-8");
        }
        if (i + width > text.size()) {
            throw std::runtime_error("Irodori-TTS tokenizer received truncated UTF-8");
        }
        chars.push_back(text.substr(i, width));
        i += width;
    }
    return chars;
}

std::string byte_token(unsigned char value) {
    constexpr char kHex[] = "0123456789ABCDEF";
    std::string out = "<0x";
    out.push_back(kHex[(value >> 4U) & 0x0FU]);
    out.push_back(kHex[value & 0x0FU]);
    out.push_back('>');
    return out;
}

// Python Irodori-TTS token_split.py (since 82c9980b). The modernbert-ja-310m
// tokenizer of the v4 Small checkpoints keeps many long phrases as one piece, and
// the model often misreads or skips them. Python encodes the text with those
// pieces disabled, so Unigram falls back to shorter ones: pieces of 4 or more
// characters (U+2581 not counted) with a score of -13 or less, except
// hiragana-only pieces and kKeepWhole. forced_splits() lists pieces above the
// threshold that are always split, and how. The caption keeps every piece. The
// rule and both tables are copied from token_split.py.
constexpr const char * kSplitTokenizerRepo = "sbintuitions/modernbert-ja-310m";
constexpr double kSplitScoreThreshold = -13.0;
constexpr size_t kSplitMinChars = 4;
constexpr double kDisabledScore = -1e9;

struct ForcedSplit {
    std::string word;
    std::vector<std::string> pieces;
};

const char * const kKeepWhole[] = {
    "心筋梗塞", "日常茶飯事", "瀬戸内海", "老若男女", "興味津々",
    "高田馬場", "アクリル板", "年モデル", "オフショア", "モスクワ",
    "テラス席", "缶バッジ", "ジュリー", "ッチャー", "ティーン", "トゥルー",
    "マリッジ", "リフィル", "リベット", "デパート", "カリウム", "タピオカ",
    "ボーイズ", "フレックス", "フリース", "ファブリック", "ユナイテッド",
    "ブルース", "ミスター", "タイムズ", "ソケット", "コスチューム",
    "プレビュー", "リーズナブル", "プロダクション", "ファイト",
    "アバター", "産婦人科", "行方不明", "直射日光",
};

const std::vector<ForcedSplit> & forced_splits() {
    static const std::vector<ForcedSplit> splits = {
        {"セブンイレブン", {"セブン", "イレブン"}},
        {"楽天市場", {"楽天", "市場"}},
        {"外壁塗装", {"外壁", "塗装"}},
        {"イベルメクチン", {"イベルメ", "クチン"}},
        {"ミノキシジル", {"ミノ", "キシ", "ジル"}},
        {"送料無料", {"送料", "無料"}},
        {"ヴィンテージ", {"ヴィン", "テージ"}},
        {"点セット", {"点", "セット"}},
        {"フォルダ", {"フォル", "ダ"}},
        {"消費者金融", {"消費者", "金融"}},
        {"代表取締役", {"代表", "取締役"}},
        {"本セット", {"本", "セット"}},
        {"厚生労働省", {"厚生", "労働", "省"}},
        {"オンラインカジノ", {"オンライン", "カジノ"}},
        {"当サイト", {"当", "サイト"}},
        {"プロペシア", {"プロ", "ペ", "シア"}},
        {"リノベーション", {"リノベ", "ー", "ション"}},
        {"メール便", {"メール", "便"}},
        {"不動産投資", {"不動産", "投資"}},
        {"ティック", {"ティ", "ック"}},
        {"フジテレビ", {"フジ", "テレビ"}},
        {"東日本大震災", {"東日本", "大震災"}},
        {"新品未使用", {"新品", "未使用"}},
        {"ネクスガード", {"ネ", "クス", "ガード"}},
        {"全身脱毛", {"全身", "脱毛"}},
        {"債務整理", {"債務", "整理"}},
        {"ス感染症", {"ス", "感染症"}},
        {"代表取締役社長", {"代表", "取締役", "社長"}},
        {"枚セット", {"枚", "セット"}},
        {"フィナステリド", {"フィ", "ナス", "テリ", "ド"}},
        {"クチコミ", {"クチ", "コミ"}},
        {"ブロガー", {"ブロ", "ガー"}},
        {"ルイヴィトン", {"ルイ", "ヴィ", "トン"}},
        {"転職エージェント", {"転職", "エージェント"}},
        {"数量限定", {"数量", "限定"}},
        {"ホワイトニング", {"ホワイト", "ニング"}},
        {"無料ダウンロード", {"無料", "ダウンロード"}},
        {"スーパーコピー", {"スーパー", "コピー"}},
        {"一般社団法人", {"一般", "社", "団", "法人"}},
        {"日本語版", {"日本語", "版"}},
        {"ペット用品", {"ペット", "用品"}},
        {"投資信託", {"投資", "信託"}},
        {"脱毛サロン", {"脱毛", "サロン"}},
        {"東京都港区", {"東京都", "港区"}},
        {"注文住宅", {"注文", "住宅"}},
        {"マイナビ", {"マイ", "ナビ"}},
        {"行政書士", {"行政", "書", "士"}},
        {"無料相談", {"無料", "相談"}},
        {"市区町村", {"市", "区", "町", "村"}},
        {"ザガーロ", {"ザ", "ガー", "ロ"}},
        {"リハビリテーショ", {"リハビリ", "テーショ"}},
        {"株主優待", {"株主", "優待"}},
        {"オンラインストア", {"オンライン", "ストア"}},
        {"プレゼンテーショ", {"プレゼン", "テーショ"}},
        {"個人事業主", {"個人", "事業主"}},
        {"社会福祉法人", {"社会", "福祉", "法人"}},
        {"カルティエ", {"カル", "ティエ"}},
        {"イオンモール", {"イオン", "モール"}},
        {"ピーター", {"ピー", "ター"}},
        {"銀行振込", {"銀行", "振込"}},
        {"スタッフ一同", {"スタッフ", "一同"}},
        {"即日融資", {"即日", "融資"}},
        {"プラセンタ", {"プラ", "センタ"}},
        {"大学大学院", {"大学", "大学院"}},
        {"モンクレール", {"モン", "ク", "レール"}},
        {"賃貸物件", {"賃貸", "物件"}},
        {"協同組合", {"協同", "組合"}},
        {"年以上前", {"年", "以上", "前"}},
        {"土日祝日", {"土日", "祝日"}},
        {"ウェイト", {"ウェ", "イト"}},
    };
    return splits;
}

std::string without_metaspace(std::string piece) {
    size_t pos = 0;
    while ((pos = piece.find(kSentencePieceSpace, pos)) != std::string::npos) {
        piece.erase(pos, 3);
    }
    return piece;
}

// Python's _HIRAGANA_ONLY, [ぁ-ゖゝゞー]+.
bool hiragana_only(const std::string & text) {
    const auto chars = utf8_chars(text);
    return !chars.empty() && std::all_of(chars.begin(), chars.end(), [](const std::string & ch) {
        if (ch.size() != 3) {
            return false;
        }
        const uint32_t cp = ((static_cast<unsigned char>(ch[0]) & 0x0FU) << 12U) |
            ((static_cast<unsigned char>(ch[1]) & 0x3FU) << 6U) |
            (static_cast<unsigned char>(ch[2]) & 0x3FU);
        return (cp >= 0x3041U && cp <= 0x3096U) || cp == 0x309DU || cp == 0x309EU || cp == 0x30FCU;
    });
}

bool disabled_for_split(const std::string & piece, double score) {
    if (score > kSplitScoreThreshold) {
        return false;
    }
    const auto text = without_metaspace(piece);
    if (utf8_chars(text).size() < kSplitMinChars || hiragana_only(text)) {
        return false;
    }
    if (std::find(std::begin(kKeepWhole), std::end(kKeepWhole), text) != std::end(kKeepWhole)) {
        return false;
    }
    const auto & splits = forced_splits();
    return std::none_of(splits.begin(), splits.end(), [&](const ForcedSplit & split) {
        return split.word == text;
    });
}

IrodoriTokenizedText padded(std::vector<int32_t> ids, int64_t max_length, int32_t pad) {
    if (max_length <= 0) {
        throw std::runtime_error("Irodori-TTS tokenizer max_length must be positive");
    }
    if (ids.size() > static_cast<size_t>(max_length)) {
        ids.resize(static_cast<size_t>(max_length));
    }
    IrodoriTokenizedText out;
    out.token_ids = std::move(ids);
    out.mask.assign(out.token_ids.size(), 1);
    while (out.token_ids.size() < static_cast<size_t>(max_length)) {
        out.token_ids.push_back(pad);
        out.mask.push_back(0);
    }
    return out;
}

}  // namespace

struct IrodoriTextTokenizer::Impl {
    explicit Impl(std::shared_ptr<const IrodoriTTSAssets> input_assets)
        : assets(std::move(input_assets)) {
        if (assets->config.use_t5gemma2_text_encoder()) {
            load_gemma_bpe();
            return;
        }
        const bool use_v4_tokenizer =
            assets->config.use_pretrained_text_encoder() &&
            assets->resources.has_file("tokenizer_v4_json");
        uses_metaspace = use_v4_tokenizer;
        const auto tokenizer_id =
            use_v4_tokenizer ? "tokenizer_v4_json" : "tokenizer_json";
        const auto root = assets->resources.parse_json(tokenizer_id);
        const auto & model = root.require("model");
        if (engine::io::json::require_string(model, "type") != "Unigram") {
            throw std::runtime_error("Irodori-TTS tokenizer expects HF Unigram tokenizer.json");
        }
        unk = static_cast<int32_t>(engine::io::json::optional_i64(model, "unk_id", 0));
        const auto & vocab_json = model.require("vocab").as_array();
        pieces.reserve(vocab_json.size());
        scores.reserve(vocab_json.size());
        int32_t id = 0;
        for (const auto & item : vocab_json) {
            const auto & pair = item.as_array();
            if (pair.size() != 2 || !pair[0].is_string() || !pair[1].is_number()) {
                throw std::runtime_error("Irodori-TTS tokenizer vocab entry must be [piece, score]");
            }
            pieces.push_back(pair[0].as_string());
            scores.push_back(pair[1].as_number());
            token_to_id.emplace(pieces.back(), id);
            ++id;
        }
        auto bos_it = token_to_id.find("<s>");
        auto pad_it = token_to_id.find("<PAD|LLM-jp>");
        if (pad_it == token_to_id.end()) {
            pad_it = token_to_id.find("<pad>");
        }
        if (bos_it == token_to_id.end() || pad_it == token_to_id.end()) {
            throw std::runtime_error("Irodori-TTS tokenizer missing BOS or PAD token");
        }
        bos = bos_it->second;
        pad = pad_it->second;
        if (auto eos_it = token_to_id.find("</s>"); eos_it != token_to_id.end()) {
            eos = eos_it->second;
        }
        if (assets->config.text_vocab_size != static_cast<int64_t>(pieces.size())) {
            throw std::runtime_error("Irodori-TTS text_vocab_size does not match tokenizer vocab size");
        }
        if (assets->config.text_tokenizer_repo == kSplitTokenizerRepo) {
            enable_rare_piece_split();
        }
    }

    void enable_rare_piece_split() {
        split_scores = scores;
        for (size_t id = 0; id < pieces.size(); ++id) {
            if (disabled_for_split(pieces[id], scores[id])) {
                split_scores[id] = kDisabledScore;
            }
        }
        for (const auto & split : forced_splits()) {
            const auto word_it = token_to_id.find(split.word);
            if (word_it == token_to_id.end()) {
                continue;
            }
            std::vector<int32_t> ids;
            std::string spelled;
            for (const auto & piece : split.pieces) {
                if (const auto it = token_to_id.find(piece); it != token_to_id.end()) {
                    ids.push_back(it->second);
                    spelled += piece;
                    continue;
                }
                for (const int32_t id : encode_piece_sequence(normalize_text_for_metaspace(piece), split_scores)) {
                    ids.push_back(id);
                    spelled += pieces[static_cast<size_t>(id)];
                }
            }
            // Like Python, keep a split only when its pieces make up the whole word.
            if (spelled == split.word) {
                forced.emplace(word_it->second, std::move(ids));
            }
        }
    }

    // v4 Large: the Gemma 3 BPE tokenizer of T5Gemma 2. Like Hugging Face's
    // tokenizer.json, spaces become U+2581 and there is no dummy prefix.
    void load_gemma_bpe() {
        bpe = std::make_shared<engine::tokenizers::LlamaBpeTokenizer>(engine::tokenizers::LlamaBpeTokenizerSpec{
            {},
            {},
            {},
            assets->resources.require_file("tokenizer_v4_json"),
            engine::tokenizers::LlamaBpePreTokenizer::Gemma4,
            {},
            kSentencePieceSpace});
        const auto bos_id = bpe->find_token_id("<bos>");
        const auto pad_id = bpe->find_token_id("<pad>");
        if (!bos_id.has_value() || !pad_id.has_value()) {
            throw std::runtime_error("Irodori-TTS Gemma tokenizer missing <bos> or <pad>");
        }
        bos = *bos_id;
        pad = *pad_id;
        if (const auto unk_id = bpe->find_token_id("<unk>"); unk_id.has_value()) {
            unk = *unk_id;
        }
        if (const auto eos_id = bpe->find_token_id("<eos>"); eos_id.has_value()) {
            eos = *eos_id;
        }
        bpe_vocab_size = assets->config.text_vocab_size;
    }

    std::vector<int32_t> encode_piece_sequence(
        const std::string & normalized, const std::vector<double> & piece_scores) const {
        const auto chars = utf8_chars(normalized);
        const size_t n = chars.size();
        std::vector<std::string> suffix(n + 1);
        for (size_t i = n; i-- > 0;) {
            suffix[i] = chars[i] + suffix[i + 1];
        }

        constexpr double kNegInf = -std::numeric_limits<double>::infinity();
        std::vector<double> best(n + 1, kNegInf);
        std::vector<size_t> next(n + 1, 0);
        std::vector<int32_t> best_id(n + 1, unk);
        best[n] = 0.0;
        for (size_t i = n; i-- > 0;) {
            auto byte_fallback = [&]() {
                double score = 0.0;
                std::vector<int32_t> ids;
                for (const unsigned char byte : chars[i]) {
                    const auto it = token_to_id.find(byte_token(byte));
                    if (it == token_to_id.end()) {
                        ids.push_back(unk);
                    } else {
                        ids.push_back(it->second);
                    }
                    score += piece_scores[static_cast<size_t>(ids.back())];
                }
                return std::pair<double, std::vector<int32_t>>(score, std::move(ids));
            };

            for (size_t j = i + 1; j <= n; ++j) {
                const size_t byte_len = suffix[i].size() - suffix[j].size();
                const std::string piece = suffix[i].substr(0, byte_len);
                const auto it = token_to_id.find(piece);
                if (it == token_to_id.end()) {
                    continue;
                }
                const double candidate = piece_scores[static_cast<size_t>(it->second)] + best[j];
                if (candidate > best[i]) {
                    best[i] = candidate;
                    next[i] = j;
                    best_id[i] = it->second;
                }
            }
            if (best[i] == kNegInf) {
                const auto fallback = byte_fallback();
                best[i] = fallback.first + best[i + 1];
                next[i] = i + 1;
                best_id[i] = fallback.second.empty() ? unk : fallback.second.front();
            }
        }

        std::vector<int32_t> ids;
        for (size_t i = 0; i < n;) {
            if (next[i] == i + 1 && token_to_id.find(chars[i]) == token_to_id.end()) {
                for (const unsigned char byte : chars[i]) {
                    const auto it = token_to_id.find(byte_token(byte));
                    ids.push_back(it == token_to_id.end() ? unk : it->second);
                }
                ++i;
                continue;
            }
            ids.push_back(best_id[i]);
            i = next[i];
        }
        return ids;
    }

    std::vector<int32_t> encode(const std::string & text, bool split_rare_pieces) const {
        if (bpe != nullptr) {
            auto ids = bpe->encode(text, true);
            if (assets->config.text_add_bos) {
                ids.insert(ids.begin(), bos);
            }
            return ids;
        }
        const bool split = split_rare_pieces && !split_scores.empty();
        const auto ids = encode_piece_sequence(
            uses_metaspace ? normalize_text_for_metaspace(text) : normalize_text_for_llm_jp(text),
            split ? split_scores : scores);
        std::vector<int32_t> out;
        out.reserve(ids.size() + 1);
        if (assets->config.text_add_bos) {
            out.push_back(bos);
        }
        for (const int32_t id : ids) {
            const auto it = split ? forced.find(id) : forced.end();
            if (it == forced.end()) {
                out.push_back(id);
            } else {
                out.insert(out.end(), it->second.begin(), it->second.end());
            }
        }
        return out;
    }

    std::shared_ptr<const IrodoriTTSAssets> assets;
    std::vector<std::string> pieces;
    std::vector<double> scores;
    // The scores with the rare pieces disabled, and the forced splits by piece
    // id; both empty unless the tokenizer is kSplitTokenizerRepo.
    std::vector<double> split_scores;
    std::unordered_map<int32_t, std::vector<int32_t>> forced;
    std::unordered_map<std::string, int32_t> token_to_id;
    std::shared_ptr<engine::tokenizers::LlamaBpeTokenizer> bpe;
    int64_t bpe_vocab_size = 0;
    int32_t bos = 1;
    int32_t eos = -1;
    int32_t pad = 4;
    int32_t unk = 0;
    bool uses_metaspace = false;
};

IrodoriTextTokenizer::IrodoriTextTokenizer(std::shared_ptr<const IrodoriTTSAssets> assets)
    : impl_(std::make_shared<Impl>(require_assets(std::move(assets)))) {}

std::vector<int32_t> IrodoriTextTokenizer::encode(const std::string & text) const {
    return impl_->encode(text, true);
}

std::vector<int32_t> IrodoriTextTokenizer::encode_caption(const std::string & caption) const {
    return impl_->encode(caption, false);
}

IrodoriTokenizedText IrodoriTextTokenizer::encode_padded(const std::string & text, int64_t max_length) const {
    return padded(encode(text), max_length, impl_->pad);
}

IrodoriTokenizedText IrodoriTextTokenizer::encode_caption_padded(
    const std::string & caption, int64_t max_length) const {
    return padded(encode_caption(caption), max_length, impl_->pad);
}

int32_t IrodoriTextTokenizer::bos_id() const noexcept {
    return impl_->bos;
}

int32_t IrodoriTextTokenizer::pad_id() const noexcept {
    return impl_->pad;
}

int32_t IrodoriTextTokenizer::unk_id() const noexcept {
    return impl_->unk;
}

int64_t IrodoriTextTokenizer::vocab_size() const noexcept {
    if (impl_->bpe != nullptr) {
        return impl_->bpe_vocab_size;
    }
    return static_cast<int64_t>(impl_->pieces.size());
}

}  // namespace engine::models::irodori_tts
