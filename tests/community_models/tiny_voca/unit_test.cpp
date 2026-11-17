#include "engine/community_models/tiny_voca/frontend.h"
#include "engine/community_models/tiny_voca/graph_cache.h"
#include "engine/community_models/tiny_voca/session.h"
#include "engine/community_models/tiny_voca/source_filter.h"
#include "engine/framework/runtime/errors.h"

#include "dizione/dizione.h"
#include "dizione/normalize.h"

#include "../../unittests/test_assert.h"

#include <cmath>
#include <cstdint>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <unordered_set>
#include <vector>

namespace {

using engine::models::tiny_voca::TinyVocaConfig;
using engine::models::tiny_voca::TinyVocaFrontend;
using engine::models::tiny_voca::TinyVocaSourceFilter;
using engine::test::require;
using engine::test::require_eq;

// The words the dictionary is taken to know in the text-rule cases.
const char * const kWords =
    "a about am and at away border call century cheaper cost costs ended example file for great help here i if in including is it "
    "let lives me meet met miles name near now of on or order out paid range ran report see she shipped shop smith tax that the "
    "time to today true use value version war was way we world write";

std::string said(const std::vector<dizione::normalize::Token> & tokens) {
    std::string out;
    for (const auto & token : tokens) {
        if (!out.empty()) {
            out += ' ';
        }
        if (token.punct) {
            out += dizione::encode(token.mark);
            continue;
        }
        const std::string text = dizione::encode(token.text);
        out += token.attach ? "+" : "";
        out += token.spell ? "[" + text + "]" : token.phones.has_value() ? "{" + text + "}" : text;
    }
    return out;
}

std::string joined(const std::vector<std::string> & parts) {
    std::string out;
    for (const std::string & part : parts) {
        out += "<" + part + ">";
    }
    return out;
}

void text_rules() {
    std::unordered_set<dizione::Str> words;
    std::istringstream list(kWords);
    for (std::string word; list >> word;) {
        words.insert(dizione::decode(word));
    }
    dizione::normalize::Normalizer normalizer([&](const dizione::Str & word) { return words.count(word) != 0; });
    const struct {
        const char * text;
        const char * expected;
    } cases[] = {
        {"It costs $3.14, including tax.", "it costs three dollars and fourteen cents , including tax ."},
        {"She paid $1,250.50 for it.", "she paid one thousand two hundred fifty dollars and fifty cents for it ."},
        {"Meet me at 3:45 PM or 10:30am.", "meet me at three forty five [pm] or ten thirty [am] ."},
        {"On 12/09/2026 we met.", "on december ninth twenty twenty six we met ."},
        {"Call 555-0142 now.", "call five five five , oh one four two now ."},
        {"Version 2.4.1 is out.", "version two point four point one is out ."},
        {"The file is report_2026.pdf.", "the file is report underscore twenty twenty six dot [pdf] ."},
        {"Write to help@shop.example.co.uk today.", "write to help at shop dot example dot co dot [uk] today ."},
        {"It is 20 \302\260C and 5 km away.", "it is twenty degrees celsius and five kilometers away ."},
        {"I ran 3/4 of the way, 1\302\275 miles.", "i ran three quarters of the way , one and a half miles ."},
        {"Let a=10, b=20, see if b>a.", "let [a] equals ten , [b] equals twenty , see if [b] is greater than [a] ."},
        {"67 > 66 is true.", "sixty seven is greater than sixty six is true ."},
        {"Dr. Smith lives on Elm St. near the U.S. border.", "doctor smith lives on elm street near the [us] border ."},
        {"The 1990s and the 21st century.", "the nineteen nineties and the twenty first century ."},
        {"Order #A7-4419 shipped.", "order number [a] seven , four four one nine shipped ."},
        {"A range of 10-15 or 2019-2021.", "a range of ten to fifteen or two thousand nineteen to two thousand twenty one ."},
        {"lol that is gr8 :) <3", "[lol] that is great"},
        {"Use {HH AH0 L OW1} here, or {name}.", "use {} here , or ( name ) ."},
        {"iOS 17.2 on the iPhone.", "[ios] seventeen point two on the [i] +phone ."},
        {"Sooo goood, Zzzz.", "sooo goood , {zzzz} ."},
        {"k", "okay"},
        {"the value of k", "the value of [k]"},
        {"50% of 1,000,000 is 500,000.", "fifty percent of one million is five hundred thousand ."},
        {"Henry VIII and World War II.", "henry the eighth and world war two ."},
        {"caf\303\251 r\303\251sum\303\251 na\303\257ve", "cafe {} naive"},
        {"It ended 5-10 \342\202\254 cheaper.", "it ended five to ten euros cheaper ."},
        {"2:03:45 was the time.", "two hours , three minutes and forty five seconds was the time ."},
        {"I'm in, y'all: it's 5 o'clock!", "i'm in , y'all : it's five o'clock !"},
        {"The CPU hit 95.5% at 3.2 GHz, then -4 \302\260F.",
         "the [cpu] hit ninety five point five percent at three point two gigahertz , then minus four degrees fahrenheit ."},
    };
    for (const auto & item : cases) {
        require_eq(said(normalizer.tokens(dizione::decode(item.text))), std::string(item.expected), std::string("text rules: ") + item.text);
    }
}

void numbers() {
    namespace normalize = dizione::normalize;
    const struct {
        long long number;
        const char * cardinal;
        const char * ordinal;
    } spoken[] = {
        {0LL, "zero", "zeroth"},
        {7LL, "seven", "seventh"},
        {13LL, "thirteen", "thirteenth"},
        {21LL, "twenty one", "twenty first"},
        {100LL, "one hundred", "one hundredth"},
        {101LL, "one hundred one", "one hundred first"},
        {1234LL, "one thousand two hundred thirty four", "one thousand two hundred thirty fourth"},
        {1000000LL, "one million", "one millionth"},
        {20000000000LL, "twenty billion", "twenty billionth"},
    };
    for (const auto & item : spoken) {
        require_eq(dizione::encode(normalize::cardinal(item.number)), std::string(item.cardinal), "cardinal");
        require_eq(dizione::encode(normalize::ordinal(item.number)), std::string(item.ordinal), "ordinal");
    }
    const struct {
        long long number;
        const char * year;
        const char * pairs;
    } years[] = {
        {1984LL, "nineteen eighty four", "nineteen eighty four"},
        {1905LL, "nineteen oh five", "nineteen oh five"},
        {1900LL, "nineteen hundred", "nineteen hundred"},
        {2000LL, "two thousand", "two thousand"},
        {2005LL, "two thousand five", "twenty oh five"},
        {2026LL, "twenty twenty six", "twenty twenty six"},
        {720LL, "seven hundred twenty", "seven twenty"},
        {4070LL, "four thousand seventy", "forty seventy"},
    };
    for (const auto & item : years) {
        require_eq(dizione::encode(normalize::year(item.number)), std::string(item.year), "year");
        require_eq(dizione::encode(normalize::pairs(item.number)), std::string(item.pairs), "pairs");
    }
    const struct {
        const char * text;
        const char * expected;
    } written[] = {
        {"1,234", "one thousand two hundred thirty four"},
        {"3.14", "three point one four"},
        {"007", "zero zero seven"},
        {"123456789012345678901234",
         "one two three four five six seven eight nine zero one two three four five six seven eight nine zero one two three four"},
        {"0.5", "zero point five"},
        {"10", "ten"},
    };
    for (const auto & item : written) {
        require_eq(dizione::encode(normalize::number_words(dizione::decode(item.text))), std::string(item.expected), "number as written");
    }
    require_eq(dizione::encode(normalize::duration(2, 3, 45)), std::string("two hours, three minutes and forty five seconds"), "duration");
    require_eq(dizione::encode(normalize::duration(0, 1, 1)), std::string("one minute and one second"), "duration");
    require_eq(dizione::encode(normalize::duration(1, 0, 0)), std::string("one hour"), "duration");
}

void sentences() {
    const std::string text =
        "Dr. Lee met Mr. A. B. Stone at 5 p.m. He said \"go!\" Then they left. No. 5 was open, e.g. on Friday. And then? Yes";
    require_eq(
        joined(TinyVocaFrontend::sentences(TinyVocaFrontend::finished(text))),
        joined({"Dr. Lee met Mr. A. B. Stone at 5 p.m. He said \"go!\"", "Then they left.", "No. 5 was open, e.g. on Friday.", "And then?",
                "Yes."}),
        "sentences");
    const struct {
        const char * text;
        const char * expected;
    } ends[] = {
        {"I will see you tomorrow", "I will see you tomorrow."},
        {"see you :)", "see you :)."},
        {"Wait,", "Wait,"},
        {"She said \"go!\"", "She said \"go!\""},
        {"ok.  ", "ok.  "},
        {"", ""},
        {"one\357\273\277", "one."},
    };
    for (const auto & item : ends) {
        require_eq(TinyVocaFrontend::finished(item.text), std::string(item.expected), "full stop at the end");
    }
    require_eq(joined(TinyVocaFrontend::shorter("one, two; three")), joined({"one,", "two;", "three"}), "cut at marks");
    require_eq(
        joined(TinyVocaFrontend::shorter("alpha beta gamma delta epsilon")), joined({"alpha beta gamma", "delta epsilon"}), "cut between words");
    require_eq(
        joined(TinyVocaFrontend::shorter("first: second \342\200\224 third")), joined({"first:", "second \342\200\224", "third"}), "cut at a dash");
    require_eq(joined(TinyVocaFrontend::shorter("a b")), joined({"a", "b"}), "two words");
    require(TinyVocaFrontend::shorter("word").empty(), "one word cannot be cut");
}

void dictionary() {
    const dizione::Lexicon lexicon(
        "zebra\tZ IY1 B R AH0\r\napple\tAE1 P AH0 L\nbanana\tB AH0 N AE1 N AH0\napple\tAE1 P L\na line without a tab\n\napp\tAE1 P\n"
        "caf\303\251\tK AE0 F EY1");
    require(!lexicon.empty(), "dictionary has words");
    require_eq(std::string(lexicon.find("apple").value()), std::string("AE1 P L"), "of two lines for a word the later wins");
    require_eq(std::string(lexicon.find("zebra").value()), std::string("Z IY1 B R AH0"), "a carriage return is not part of the entry");
    require_eq(std::string(lexicon.find("app").value()), std::string("AE1 P"), "a word that begins another");
    require_eq(std::string(lexicon.find("caf\303\251").value()), std::string("K AE0 F EY1"), "the last line needs no line end");
    for (const char * missing : {"ap", "apples", "banan", "bananas", "", "a line without a tab", "zebr", "zzz", "A"}) {
        require(!lexicon.contains(missing), std::string("not in the dictionary: ") + missing);
    }
    require(dizione::Lexicon("").empty() && dizione::Lexicon("no tabs here\n").empty(), "empty dictionaries");
}

// With every weight of the head at zero the two filters pass their sources
// unchanged, so the head returns the pulse train plus the noise: this checks
// the framing, the transforms and the overlap-add.
void waveform_head() {
    TinyVocaConfig config;
    config.channels = 4;
    config.synth_fft = 512;
    config.synth_hop = 128;
    config.noise_fft = 256;
    config.noise_hop = 64;
    config.cep_causal = 80;
    config.cep_anti = 16;
    config.noise_cep = 32;
    const int64_t harmonic_outputs = config.cep_causal + config.cep_anti + config.hop / config.synth_hop;
    const int64_t noise_outputs = config.noise_cep + config.hop / config.noise_hop;
    const TinyVocaSourceFilter head(
        config,
        std::vector<float>(static_cast<size_t>(harmonic_outputs * config.channels), 0.0F),
        std::vector<float>(static_cast<size_t>(harmonic_outputs), 0.0F),
        std::vector<float>(static_cast<size_t>(noise_outputs * config.channels), 0.0F),
        std::vector<float>(static_cast<size_t>(noise_outputs), 0.0F));

    const int64_t frames = 40;
    const int64_t samples = frames * config.hop;
    const std::vector<float> f0(static_cast<size_t>(frames), 100.0F);
    const std::vector<float> pulses = head.pulse_train(f0);
    require_eq(static_cast<int64_t>(pulses.size()), samples, "pulse train length");
    // 100 Hz at 24 kHz: the wave comes back every 240 samples, with unit power.
    const int64_t period = config.sample_rate / 100;
    double power = 0.0;
    float largest = 0.0F;
    for (int64_t n = 0; n < 42 * period; ++n) {
        power += static_cast<double>(pulses[static_cast<size_t>(n)]) * pulses[static_cast<size_t>(n)];
    }
    for (int64_t n = 0; n + period < samples; ++n) {
        largest = std::max(largest, std::fabs(pulses[static_cast<size_t>(n)] - pulses[static_cast<size_t>(n + period)]));
    }
    require(largest < 1.0e-3F, "the pulse train of a steady pitch is periodic");
    engine::test::require_close(static_cast<float>(power / static_cast<double>(42 * period)), 1.0F, 0.02F, "pulse train power");

    std::vector<float> noise(static_cast<size_t>(samples));
    uint32_t state = 12345;
    for (float & value : noise) {
        state = state * 1664525U + 1013904223U;
        value = static_cast<float>(state >> 8) / 8388608.0F - 1.0F;
    }
    const std::vector<float> hidden(static_cast<size_t>(frames * config.channels), 0.0F);
    const std::vector<float> out = head.synthesize(hidden, f0, noise, frames);
    require_eq(static_cast<int64_t>(out.size()), samples, "waveform length");
    float apart = 0.0F;
    for (int64_t n = 0; n < samples; ++n) {
        apart = std::max(apart, std::fabs(out[static_cast<size_t>(n)] - (pulses[static_cast<size_t>(n)] + noise[static_cast<size_t>(n)])));
    }
    require(apart < 1.0e-3F, "flat filters return the sources");

    bool rejected = false;
    try {
        (void)head.synthesize(hidden, f0, std::vector<float>(10), frames);
    } catch (const std::exception &) {
        rejected = true;
    }
    require(rejected, "too little noise is an error");
}

// The graph cache keeps the graphs used last, within a number and a size.
void graph_cache() {
    using engine::models::tiny_voca::GraphCache;
    using engine::models::tiny_voca::GraphKey;
    GraphCache<int> cache(3, 100);
    auto keep = [&](int stack, int64_t length, size_t bytes) {
        return cache.put(GraphKey{stack, length}, std::make_unique<int>(static_cast<int>(length)), bytes) == nullptr;
    };
    require(cache.find({0, 10}) == nullptr, "an empty cache has nothing");
    require(keep(0, 10, 40), "a graph within the limits is kept");
    require(keep(1, 10, 40), "the same length of another stack is kept as its own graph");
    require(cache.find({0, 10}) != nullptr && *cache.find({0, 10}) == 10, "a kept graph is found");
    require(cache.find({2, 10}) == nullptr && cache.find({0, 11}) == nullptr, "another stack or length is not found");
    // 40 + 40 + 30 is over 100 bytes: the graph used longest ago goes, which is stack 1.
    require(keep(0, 20, 30), "a third graph is kept");
    require(cache.find({1, 10}) == nullptr, "the graph used longest ago went for the size limit");
    require(cache.find({0, 10}) != nullptr && cache.find({0, 20}) != nullptr, "the graphs used last stay");
    require_eq(cache.stats().kept_bytes, size_t{70}, "kept bytes");
    // The number limit: three small graphs fill the cache and a fourth pushes out the oldest, {0, 10}.
    require(keep(2, 5, 1) && keep(2, 6, 1), "small graphs are kept");
    require_eq(cache.stats().kept_graphs, size_t{3}, "kept graphs");
    require(cache.find({0, 10}) == nullptr && cache.find({0, 20}) != nullptr, "the oldest went for the number limit");
    // A graph larger than the size limit is handed back and nothing is dropped for it.
    require(!keep(0, 3000, 101), "a graph over the size limit is not kept");
    require_eq(cache.stats().kept_graphs, size_t{3}, "kept graphs after a refusal");
    // One graph as large as the limit takes the place of all the others.
    require(keep(0, 2000, 100), "a graph of the full size is kept");
    require_eq(cache.stats().kept_graphs, size_t{1}, "kept graphs after a large one");
    require_eq(cache.stats().kept_bytes, size_t{100}, "kept bytes after a large one");
    const auto stats = cache.stats();
    require_eq(stats.hits, uint64_t{5}, "hits");
    require_eq(stats.misses, uint64_t{5}, "misses");
    GraphCache<int> off(0, 100);
    require(off.put({0, 1}, std::make_unique<int>(1), 1) != nullptr && off.find({0, 1}) == nullptr, "no graphs allowed keeps nothing");
}

// English and no language pass; any other language is an invalid request,
// whether it comes with the text (the command-line tool's --language, the
// speech endpoint's "language") or as the request option (the server's
// options.language).
void language() {
    using engine::models::tiny_voca::require_english;
    using engine::runtime::TaskRequest;
    using engine::runtime::Transcript;
    auto refused = [](const TaskRequest & request) {
        try {
            require_english(request);
        } catch (const engine::runtime::InvalidRequestError & error) {
            require(std::string(error.what()).find("English only") != std::string::npos, "the error says what the model reads");
            return true;
        }
        return false;
    };
    auto with_text = [](const std::string & name) {
        TaskRequest request;
        request.text_input = Transcript{"Hello.", name};
        return request;
    };
    auto with_option = [](const std::string & name) {
        TaskRequest request;
        request.text_input = Transcript{"Hello.", ""};
        request.options["language"] = name;
        return request;
    };
    for (const char * name : {"", "en", "EN", "en-us", "en-US", "en_us", "English", "english"}) {
        require(!refused(with_text(name)), std::string("accepted with the text: ") + name);
        require(!refused(with_option(name)), std::string("accepted as an option: ") + name);
    }
    for (const char * name : {"ja", "de", "en-gb", "zh", "Japanese", "auto", " "}) {
        require(refused(with_text(name)), std::string("refused with the text: ") + name);
        require(refused(with_option(name)), std::string("refused as an option: ") + name);
    }
    // The command-line tool sets both; one wrong value is enough.
    TaskRequest both = with_text("en");
    both.options["language"] = "ja";
    require(refused(both), "an English text language does not hide another language in the options");
}

}  // namespace

int main() {
    try {
        text_rules();
        numbers();
        sentences();
        dictionary();
        waveform_head();
        graph_cache();
        language();
        std::cout << "tiny_voca unit test passed\n";
        return 0;
    } catch (const std::exception & error) {
        std::cerr << "tiny_voca unit test failed: " << error.what() << "\n";
        return 1;
    }
}
