#include "engine/framework/text/unicode_normalization.h"
#include "engine/models/supertonic/tokenizer_text.h"

#include "test_assert.h"

#include <cstdint>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_set>
#include <vector>

namespace {

struct Case {
    uint32_t input;
    std::vector<uint32_t> expected;
};

struct NfkdCase {
    uint32_t input;
    std::vector<uint32_t> expected;
};

struct TextCase {
    const char * language;
    const char * text;
};

std::vector<uint32_t> utf8_to_codepoints(std::string_view text) {
    std::vector<uint32_t> out;
    for (size_t i = 0; i < text.size();) {
        const auto c = static_cast<unsigned char>(text[i]);
        uint32_t codepoint = 0;
        size_t width = 0;
        if ((c & 0x80U) == 0U) {
            codepoint = c;
            width = 1;
        } else if ((c & 0xE0U) == 0xC0U && i + 1 < text.size()) {
            codepoint = static_cast<uint32_t>(c & 0x1FU) << 6U;
            codepoint |= static_cast<uint32_t>(static_cast<unsigned char>(text[i + 1]) & 0x3FU);
            width = 2;
        } else if ((c & 0xF0U) == 0xE0U && i + 2 < text.size()) {
            codepoint = static_cast<uint32_t>(c & 0x0FU) << 12U;
            codepoint |= static_cast<uint32_t>(static_cast<unsigned char>(text[i + 1]) & 0x3FU) << 6U;
            codepoint |= static_cast<uint32_t>(static_cast<unsigned char>(text[i + 2]) & 0x3FU);
            width = 3;
        } else if ((c & 0xF8U) == 0xF0U && i + 3 < text.size()) {
            codepoint = static_cast<uint32_t>(c & 0x07U) << 18U;
            codepoint |= static_cast<uint32_t>(static_cast<unsigned char>(text[i + 1]) & 0x3FU) << 12U;
            codepoint |= static_cast<uint32_t>(static_cast<unsigned char>(text[i + 2]) & 0x3FU) << 6U;
            codepoint |= static_cast<uint32_t>(static_cast<unsigned char>(text[i + 3]) & 0x3FU);
            width = 4;
        } else {
            throw std::runtime_error("invalid UTF-8 in unicode normalization test");
        }
        out.push_back(codepoint);
        i += width;
    }
    return out;
}

const std::vector<Case> & issue_cases() {
    static const std::vector<Case> cases = {
        {0x00C0, {0x0041, 0x0300}}, {0x00C1, {0x0041, 0x0301}}, {0x00C2, {0x0041, 0x0302}},
        {0x00C3, {0x0041, 0x0303}}, {0x00C4, {0x0041, 0x0308}}, {0x00C5, {0x0041, 0x030A}},
        {0x00C7, {0x0043, 0x0327}}, {0x00C8, {0x0045, 0x0300}}, {0x00C9, {0x0045, 0x0301}},
        {0x00CA, {0x0045, 0x0302}}, {0x00CB, {0x0045, 0x0308}}, {0x00CC, {0x0049, 0x0300}},
        {0x00CD, {0x0049, 0x0301}}, {0x00CE, {0x0049, 0x0302}}, {0x00CF, {0x0049, 0x0308}},
        {0x00D1, {0x004E, 0x0303}}, {0x00D2, {0x004F, 0x0300}}, {0x00D3, {0x004F, 0x0301}},
        {0x00D4, {0x004F, 0x0302}}, {0x00D5, {0x004F, 0x0303}}, {0x00D6, {0x004F, 0x0308}},
        {0x00D9, {0x0055, 0x0300}}, {0x00DA, {0x0055, 0x0301}}, {0x00DB, {0x0055, 0x0302}},
        {0x00DC, {0x0055, 0x0308}}, {0x00DD, {0x0059, 0x0301}}, {0x00E0, {0x0061, 0x0300}},
        {0x00E1, {0x0061, 0x0301}}, {0x00E2, {0x0061, 0x0302}}, {0x00E3, {0x0061, 0x0303}},
        {0x00E4, {0x0061, 0x0308}}, {0x00E5, {0x0061, 0x030A}}, {0x00E7, {0x0063, 0x0327}},
        {0x00E8, {0x0065, 0x0300}}, {0x00E9, {0x0065, 0x0301}}, {0x00EA, {0x0065, 0x0302}},
        {0x00EB, {0x0065, 0x0308}}, {0x00EC, {0x0069, 0x0300}}, {0x00ED, {0x0069, 0x0301}},
        {0x00EE, {0x0069, 0x0302}}, {0x00EF, {0x0069, 0x0308}}, {0x00F1, {0x006E, 0x0303}},
        {0x00F2, {0x006F, 0x0300}}, {0x00F3, {0x006F, 0x0301}}, {0x00F4, {0x006F, 0x0302}},
        {0x00F5, {0x006F, 0x0303}}, {0x00F6, {0x006F, 0x0308}}, {0x00F9, {0x0075, 0x0300}},
        {0x00FA, {0x0075, 0x0301}}, {0x00FB, {0x0075, 0x0302}}, {0x00FC, {0x0075, 0x0308}},
        {0x00FD, {0x0079, 0x0301}}, {0x00FF, {0x0079, 0x0308}}, {0x0100, {0x0041, 0x0304}},
        {0x0101, {0x0061, 0x0304}}, {0x0102, {0x0041, 0x0306}}, {0x0103, {0x0061, 0x0306}},
        {0x0104, {0x0041, 0x0328}}, {0x0105, {0x0061, 0x0328}}, {0x0106, {0x0043, 0x0301}},
        {0x0107, {0x0063, 0x0301}}, {0x010C, {0x0043, 0x030C}}, {0x010D, {0x0063, 0x030C}},
        {0x010E, {0x0044, 0x030C}}, {0x010F, {0x0064, 0x030C}}, {0x0112, {0x0045, 0x0304}},
        {0x0113, {0x0065, 0x0304}}, {0x0116, {0x0045, 0x0307}}, {0x0117, {0x0065, 0x0307}},
        {0x0118, {0x0045, 0x0328}}, {0x0119, {0x0065, 0x0328}}, {0x011A, {0x0045, 0x030C}},
        {0x011B, {0x0065, 0x030C}}, {0x011E, {0x0047, 0x0306}}, {0x011F, {0x0067, 0x0306}},
        {0x0122, {0x0047, 0x0327}}, {0x0123, {0x0067, 0x0327}}, {0x012A, {0x0049, 0x0304}},
        {0x012B, {0x0069, 0x0304}}, {0x012E, {0x0049, 0x0328}}, {0x012F, {0x0069, 0x0328}},
        {0x0130, {0x0049, 0x0307}}, {0x0136, {0x004B, 0x0327}}, {0x0137, {0x006B, 0x0327}},
        {0x0139, {0x004C, 0x0301}}, {0x013A, {0x006C, 0x0301}}, {0x013B, {0x004C, 0x0327}},
        {0x013C, {0x006C, 0x0327}}, {0x013D, {0x004C, 0x030C}}, {0x013E, {0x006C, 0x030C}},
        {0x0143, {0x004E, 0x0301}}, {0x0144, {0x006E, 0x0301}}, {0x0145, {0x004E, 0x0327}},
        {0x0146, {0x006E, 0x0327}}, {0x0147, {0x004E, 0x030C}}, {0x0148, {0x006E, 0x030C}},
        {0x0150, {0x004F, 0x030B}}, {0x0151, {0x006F, 0x030B}}, {0x0154, {0x0052, 0x0301}},
        {0x0155, {0x0072, 0x0301}}, {0x0158, {0x0052, 0x030C}}, {0x0159, {0x0072, 0x030C}},
        {0x015A, {0x0053, 0x0301}}, {0x015B, {0x0073, 0x0301}}, {0x015E, {0x0053, 0x0327}},
        {0x015F, {0x0073, 0x0327}}, {0x0160, {0x0053, 0x030C}}, {0x0161, {0x0073, 0x030C}},
        {0x0164, {0x0054, 0x030C}}, {0x0165, {0x0074, 0x030C}}, {0x016A, {0x0055, 0x0304}},
        {0x016B, {0x0075, 0x0304}}, {0x016E, {0x0055, 0x030A}}, {0x016F, {0x0075, 0x030A}},
        {0x0170, {0x0055, 0x030B}}, {0x0171, {0x0075, 0x030B}}, {0x0172, {0x0055, 0x0328}},
        {0x0173, {0x0075, 0x0328}}, {0x0178, {0x0059, 0x0308}}, {0x0179, {0x005A, 0x0301}},
        {0x017A, {0x007A, 0x0301}}, {0x017B, {0x005A, 0x0307}}, {0x017C, {0x007A, 0x0307}},
        {0x017D, {0x005A, 0x030C}}, {0x017E, {0x007A, 0x030C}}, {0x0218, {0x0053, 0x0326}},
        {0x0219, {0x0073, 0x0326}}, {0x021A, {0x0054, 0x0326}}, {0x021B, {0x0074, 0x0326}},
        {0x0386, {0x0391, 0x0301}}, {0x0388, {0x0395, 0x0301}}, {0x0389, {0x0397, 0x0301}},
        {0x038A, {0x0399, 0x0301}}, {0x038C, {0x039F, 0x0301}}, {0x038E, {0x03A5, 0x0301}},
        {0x038F, {0x03A9, 0x0301}}, {0x0390, {0x03B9, 0x0308, 0x0301}},
        {0x03AA, {0x0399, 0x0308}}, {0x03AB, {0x03A5, 0x0308}}, {0x03AC, {0x03B1, 0x0301}},
        {0x03AD, {0x03B5, 0x0301}}, {0x03AE, {0x03B7, 0x0301}}, {0x03AF, {0x03B9, 0x0301}},
        {0x03B0, {0x03C5, 0x0308, 0x0301}}, {0x03CA, {0x03B9, 0x0308}}, {0x03CB, {0x03C5, 0x0308}},
        {0x03CC, {0x03BF, 0x0301}}, {0x03CD, {0x03C5, 0x0301}}, {0x03CE, {0x03C9, 0x0301}},
        {0x0401, {0x0415, 0x0308}}, {0x0407, {0x0406, 0x0308}}, {0x040E, {0x0423, 0x0306}},
        {0x0419, {0x0418, 0x0306}}, {0x0439, {0x0438, 0x0306}}, {0x0451, {0x0435, 0x0308}},
        {0x0457, {0x0456, 0x0308}}, {0x045E, {0x0443, 0x0306}},
    };
    return cases;
}

const std::vector<TextCase> & supertonic_issue_text_cases() {
    static const std::vector<TextCase> cases = {
        {"en", "Hello there."},
        {"en", "你好"},
        {"en", "你好。"},
        {"en", "你好！"},
        {"en", "你好，世界"},
        {"en", "你好!"},
        {"ja", "こんにちは / コンニチハ / ガガガ"},
        {"ja", "日本語"},
        {"ja", "はい、そうです"},
        {"ja", "こんにちは！"},
        {"ko", "안녕하세요"},
        {"ko", "안녕하세요"},
        {"ko", "안녕하세요！"},
        {"vi", "Xin chào bạn"},
        {"vi", "Xin chao ban"},
    };
    return cases;
}

const std::vector<TextCase> & supertonic_supported_language_text_cases() {
    static const std::vector<TextCase> cases = {
        {"en", "The careful narrator paused at the red signal, then described the quiet station clearly."},
        {"ko", "오늘 아침 연구실에서 팀원들은 새 음성 모델의 발음과 쉼표 뒤 멈춤을 차분히 확인했습니다."},
        {"ja", "今朝の研究室で、担当者は新しい音声モデルの発音と句読点の間を丁寧に確認しました。"},
        {"ar", "في صباح هادئ، شرح المهندس نتيجة الاختبار بوضوح ثم توقف قليلا عند كل فاصلة."},
        {"bg", "Тази сутрин екипът провери гласа, паузите и ясното произношение в новия модел."},
        {"cs", "Ráno tým pečlivě ověřil hlas, krátké pauzy po čárkách a srozumitelnou výslovnost."},
        {"da", "I morges testede holdet stemmen, pauserne efter kommaer og en tydelig rolig udtale."},
        {"de", "Am Morgen prüfte das Team die Stimme, kurze Pausen nach Kommas und klare Aussprache."},
        {"el", "Το πρωί η ομάδα έλεγξε τη φωνή, τις παύσεις μετά τα κόμματα και την καθαρή άρθρωση."},
        {"es", "Esta mañana el equipo revisó la voz, las pausas tras las comas y una pronunciación clara."},
        {"et", "Täna hommikul kontrollis meeskond häält, komajärgseid pause ja selget hääldust."},
        {"fi", "Tänä aamuna ryhmä tarkisti äänen, pilkkujen jälkeiset tauot ja selkeän ääntämyksen."},
        {"fr", "Ce matin, l’équipe a vérifié la voix, les pauses après les virgules et la diction claire."},
        {"hi", "आज सुबह दल ने आवाज, अल्पविराम के बाद ठहराव और स्पष्ट उच्चारण को ध्यान से परखा।"},
        {"hr", "Jutros je tim provjerio glas, kratke pauze nakon zareza i jasno izgovorene riječi."},
        {"hu", "Ma reggel a csapat ellenőrizte a hangot, a vessző utáni szüneteket és a tiszta kiejtést."},
        {"id", "Pagi ini tim memeriksa suara, jeda setelah koma, dan pengucapan yang jelas serta tenang."},
        {"it", "Stamattina il gruppo ha controllato la voce, le pause dopo le virgole e una dizione chiara."},
        {"lt", "Šį rytą komanda tikrino balsą, pauzes po kablelių ir aiškų, ramų tarimą."},
        {"lv", "Šorīt komanda pārbaudīja balsi, pauzes pēc komatiem un skaidru, mierīgu izrunu."},
        {"nl", "Vanmorgen controleerde het team de stem, pauzes na komma’s en een heldere uitspraak."},
        {"pl", "Dziś rano zespół sprawdził głos, pauzy po przecinkach oraz wyraźną wymowę."},
        {"pt", "Nesta manhã, a equipe revisou a voz, as pausas após vírgulas e a pronúncia clara."},
        {"ro", "În această dimineață, echipa a verificat vocea, pauzele după virgule și dicția clară."},
        {"ru", "Сегодня утром команда проверила голос, паузы после запятых и ясное произношение."},
        {"sk", "Dnes ráno tím skontroloval hlas, pauzy po čiarkach a jasnú, pokojnú výslovnosť."},
        {"sl", "Danes zjutraj je ekipa preverila glas, premore po vejicah in jasno izgovorjavo."},
        {"sv", "I morse granskade teamet rösten, pauser efter kommatecken och ett tydligt uttal."},
        {"tr", "Bu sabah ekip sesi, virgüllerden sonraki durakları ve açık telaffuzu dikkatle inceledi."},
        {"uk", "Сьогодні вранці команда перевірила голос, паузи після ком і чітку вимову."},
        {"vi", "Sáng nay nhóm kiểm tra giọng đọc, quãng ngừng sau dấu phẩy và cách phát âm rõ ràng."},
        {"na", "A calm speaker tests punctuation, short pauses, and steady rhythm across an unknown language tag."},
    };
    return cases;
}

const std::vector<TextCase> & supertonic_failure_class_text_cases() {
    static const std::vector<TextCase> cases = {
        {"ja", "こんにちは，世界！本当に大丈夫？"},
        {"ko", "안녕하세요，세계！정말 괜찮아요？"},
        {"na", "你好，世界！真的可以吗？"},
        {"cs", "Příliš žluťoučký kůň úpěl ďábelské ódy."},
        {"hu", "Árvíztűrő tükörfúrógép ellenőrzi a hosszú ő és ű hangokat."},
        {"pl", "Zażółć gęślą jaźń, później sprawdź ś, ź, ż oraz ą i ę."},
        {"ro", "Încălzirea verifică ă, â, î, ș și ț într-o propoziție clară."},
        {"tr", "Çağrı öğlen ışıklı köşede şüpheli küçük üçgeni inceledi."},
        {"lt", "Šį rytą ąžuolas, ėriukas, įrankis, ūdra ir žirgas skambėjo aiškiai."},
        {"lv", "Šorīt ābele, čiekurs, ģimene, ķirsis, ļaudis un žogs skanēja skaidri."},
        {"vi", "Tiếng Việt có ạ, ậ, ệ, ộ, ữ, ỳ và dấu hỏi ngã rõ ràng."},
        {"ar", "اللغة العربية تختبر الحروف والنقاط والفواصل بوضوح."},
        {"bg", "Българският текст проверява кирилица, ударения и ясна пунктуация."},
        {"el", "Τα ελληνικά ελέγχουν τόνους, διαλυτικά και καθαρή στίξη."},
        {"hi", "हिंदी वाक्य देवनागरी मात्राओं और स्पष्ट विरामों को परखता है।"},
        {"ru", "Русский текст проверяет кириллицу, запятые и ясное произношение."},
        {"uk", "Український текст перевіряє ї, є, ґ, апостроф і чіткі паузи."},
    };
    return cases;
}

const std::vector<NfkdCase> & supertonic_failure_class_nfkd_cases() {
    static const std::vector<NfkdCase> cases = {
        {0xFF01, {0x0021}},
        {0xFF0C, {0x002C}},
        {0xFF1F, {0x003F}},
        {0x00E1, {0x0061, 0x0301}},
        {0x00E9, {0x0065, 0x0301}},
        {0x010D, {0x0063, 0x030C}},
        {0x010F, {0x0064, 0x030C}},
        {0x0159, {0x0072, 0x030C}},
        {0x016F, {0x0075, 0x030A}},
        {0x017E, {0x007A, 0x030C}},
        {0x0151, {0x006F, 0x030B}},
        {0x0171, {0x0075, 0x030B}},
        {0x0105, {0x0061, 0x0328}},
        {0x0119, {0x0065, 0x0328}},
        {0x015B, {0x0073, 0x0301}},
        {0x017A, {0x007A, 0x0301}},
        {0x0219, {0x0073, 0x0326}},
        {0x021B, {0x0074, 0x0326}},
        {0x011F, {0x0067, 0x0306}},
        {0x0131, {0x0131}},
        {0x0161, {0x0073, 0x030C}},
        {0x012F, {0x0069, 0x0328}},
        {0x016B, {0x0075, 0x0304}},
        {0x0101, {0x0061, 0x0304}},
        {0x0123, {0x0067, 0x0327}},
        {0x0137, {0x006B, 0x0327}},
        {0x013C, {0x006C, 0x0327}},
        {0x1EA1, {0x0061, 0x0323}},
        {0x1EAD, {0x0061, 0x0323, 0x0302}},
        {0x1EC7, {0x0065, 0x0323, 0x0302}},
        {0x1ED9, {0x006F, 0x0323, 0x0302}},
        {0x1EEF, {0x0075, 0x031B, 0x0303}},
        {0x1EF3, {0x0079, 0x0300}},
        {0x30AC, {0x30AB, 0x3099}},
        {0x3071, {0x306F, 0x309A}},
        {0xAC00, {0x1100, 0x1161}},
        {0xAC01, {0x1100, 0x1161, 0x11A8}},
        {0xC548, {0x110B, 0x1161, 0x11AB}},
        {0xB155, {0x1102, 0x1167, 0x11BC}},
        {0xD558, {0x1112, 0x1161}},
        {0xC138, {0x1109, 0x1166}},
        {0xC694, {0x110B, 0x116D}},
        {0x0457, {0x0456, 0x0308}},
        {0x0454, {0x0454}},
        {0x0491, {0x0491}},
    };
    return cases;
}

void require_eq_u32_vector(
    const std::vector<uint32_t> & actual,
    const std::vector<uint32_t> & expected,
    const std::string & label) {
    engine::test::require_eq(actual.size(), expected.size(), label + " size");
    for (size_t i = 0; i < expected.size(); ++i) {
        engine::test::require_eq(actual[i], expected[i], label + " value " + std::to_string(i));
    }
}

void test_issue_character_decompositions() {
    for (const auto & test_case : issue_cases()) {
        std::vector<uint32_t> out;
        const bool decomposed = engine::text::append_known_unicode_decomposition(test_case.input, out);
        engine::test::require(decomposed, "expected decomposition");
        require_eq_u32_vector(out, test_case.expected, "direct decomposition");
        require_eq_u32_vector(
            engine::text::decompose_known_unicode_codepoints({test_case.input}),
            test_case.expected,
            "vector decomposition");
    }
}

void test_mixed_sequence_and_passthrough() {
    const std::vector<uint32_t> input{0x0078, 0x0439, 0x0020, 0x03B0, 0x4E2D, 0x00E9};
    const std::vector<uint32_t> expected{
        0x0078,
        0x0438, 0x0306,
        0x0020,
        0x03C5, 0x0308, 0x0301,
        0x4E2D,
        0x0065, 0x0301,
    };
    require_eq_u32_vector(
        engine::text::decompose_known_unicode_codepoints(input),
        expected,
        "mixed sequence");

    std::vector<uint32_t> out;
    engine::test::require(!engine::text::append_known_unicode_decomposition(0x2603, out), "snowman passthrough");
    engine::test::require(out.empty(), "snowman append output");
    require_eq_u32_vector(
        engine::text::decompose_known_unicode_codepoints({0x2603}),
        {0x2603},
        "snowman vector passthrough");
}

void test_nfkd_extended_coverage() {
    require_eq_u32_vector(
        engine::text::normalize_nfkd_codepoints({0x1EA1, 0x0020, 0xFF01, 0xFF0C, 0xFF1F}),
        {0x0061, 0x0323, 0x0020, 0x0021, 0x002C, 0x003F},
        "vietnamese and fullwidth nfkd");
    require_eq_u32_vector(
        engine::text::normalize_nfkd_codepoints({0xAC01}),
        {0x1100, 0x1161, 0x11A8},
        "hangul nfkd");
    require_eq_u32_vector(
        engine::text::normalize_nfkd_codepoints({0x30AC, 0x3071}),
        {0x30AB, 0x3099, 0x306F, 0x309A},
        "kana nfkd");

    std::vector<uint32_t> out;
    engine::test::require(engine::text::append_nfkd_decomposition(0xFB03, out), "ligature nfkd");
    require_eq_u32_vector(out, {0x0066, 0x0066, 0x0069}, "ligature decomposition");

    out.clear();
    engine::test::require(!engine::text::append_nfkd_decomposition(0x2603, out), "snowman nfkd passthrough");
    engine::test::require(out.empty(), "snowman nfkd append output");
}

void test_supertonic_failure_class_nfkd_mappings() {
    for (const auto & test_case : supertonic_failure_class_nfkd_cases()) {
        require_eq_u32_vector(
            engine::text::normalize_nfkd_codepoints({test_case.input}),
            test_case.expected,
            "Supertonic failure class NFKD " + std::to_string(test_case.input));
    }
}

std::shared_ptr<const engine::models::supertonic::SupertonicAssets> make_synthetic_supertonic_assets() {
    std::unordered_set<uint32_t> codepoints;
    for (uint32_t codepoint = 0x20; codepoint <= 0x7E; ++codepoint) {
        codepoints.insert(codepoint);
    }

    const auto add_text = [&](const TextCase & text_case) {
        const std::string wrapped = "<" + std::string(text_case.language) + ">" + text_case.text +
                                    "</" + text_case.language + ">";
        for (const uint32_t codepoint : engine::text::normalize_nfkd_codepoints(utf8_to_codepoints(wrapped))) {
            codepoints.insert(codepoint);
        }
    };
    for (const auto & text_case : supertonic_issue_text_cases()) {
        add_text(text_case);
    }
    for (const auto & text_case : supertonic_supported_language_text_cases()) {
        add_text(text_case);
    }
    for (const auto & text_case : supertonic_failure_class_text_cases()) {
        add_text(text_case);
    }
    for (const auto & test_case : supertonic_failure_class_nfkd_cases()) {
        for (const uint32_t codepoint : test_case.expected) {
            codepoints.insert(codepoint);
        }
    }

    auto assets = std::make_shared<engine::models::supertonic::SupertonicAssets>();
    int64_t token = 1;
    for (const uint32_t codepoint : codepoints) {
        assets->unicode_indexer.emplace(codepoint, token++);
    }
    return assets;
}

void test_supertonic_tokenizer_regression_inputs() {
    const auto assets = make_synthetic_supertonic_assets();
    const engine::models::supertonic::SupertonicTextTokenizer tokenizer(assets);
    for (const auto & test_case : supertonic_failure_class_nfkd_cases()) {
        const bool unchanged = test_case.expected.size() == 1 && test_case.expected.front() == test_case.input;
        if (!unchanged) {
            engine::test::require(
                assets->unicode_indexer.find(test_case.input) == assets->unicode_indexer.end(),
                "synthetic Supertonic indexer should not contain raw failing codepoint " +
                    std::to_string(test_case.input));
        }
        for (const uint32_t codepoint : test_case.expected) {
            engine::test::require(
                assets->unicode_indexer.find(codepoint) != assets->unicode_indexer.end(),
                "synthetic Supertonic indexer missing expected normalized codepoint " + std::to_string(codepoint));
        }
    }

    for (const auto & text_case : supertonic_issue_text_cases()) {
        const auto encoded = tokenizer.encode(text_case.text, text_case.language);
        engine::test::require(encoded.length > 0, std::string("Supertonic issue input encoded for ") + text_case.language);
        engine::test::require_eq(
            encoded.ids.size(),
            encoded.mask.size(),
            std::string("Supertonic issue input mask size for ") + text_case.language);
    }
    for (const auto & text_case : supertonic_supported_language_text_cases()) {
        const auto encoded = tokenizer.encode(text_case.text, text_case.language);
        engine::test::require(
            encoded.length > 0,
            std::string("Supertonic supported language input encoded for ") + text_case.language);
        engine::test::require_eq(
            encoded.ids.size(),
            encoded.mask.size(),
            std::string("Supertonic supported language input mask size for ") + text_case.language);
    }
    for (const auto & text_case : supertonic_failure_class_text_cases()) {
        const auto encoded = tokenizer.encode(text_case.text, text_case.language);
        engine::test::require(
            encoded.length > 0,
            std::string("Supertonic failure class input encoded for ") + text_case.language);
        engine::test::require_eq(
            encoded.ids.size(),
            encoded.mask.size(),
            std::string("Supertonic failure class input mask size for ") + text_case.language);
    }
}

}  // namespace

int main() {
    try {
        test_issue_character_decompositions();
        test_mixed_sequence_and_passthrough();
        test_nfkd_extended_coverage();
        test_supertonic_failure_class_nfkd_mappings();
        test_supertonic_tokenizer_regression_inputs();
        std::cout << "unicode_normalization_test passed\n";
        return 0;
    } catch (const std::exception & ex) {
        std::cerr << "unicode_normalization_test failed: " << ex.what() << "\n";
        return 1;
    }
}
