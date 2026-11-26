// Reads one command per stdin line and prints the result, for check_frontend.py to compare against reverb_tts.
#include "engine/community_models/reverb/frontend.h"
#include <cstdio>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>
using namespace engine::community_models::reverb;
static std::vector<float> read_f32(const std::string & path) {
    std::ifstream f(path, std::ios::binary); std::vector<float> v;
    float x; while (f.read(reinterpret_cast<char *>(&x), 4)) v.push_back(x); return v;
}
static void write_f32(const std::string & path, const std::vector<float> & v) {
    std::ofstream f(path, std::ios::binary); f.write(reinterpret_cast<const char *>(v.data()), v.size() * 4);
}
int main() {
    std::string line;
    while (std::getline(std::cin, line)) {
        std::istringstream in(line); std::string cmd; in >> cmd;
        if (cmd == "estimate" || cmd == "split" || cmd == "prepare") {
            // estimate <ref_frames>\t<text>\t<ref_text>
            int64_t frames; in >> frames; std::string rest; std::getline(in, rest);
            auto tab = rest.find('\t', 1); std::string text = rest.substr(1, tab - 1), ref = rest.substr(tab + 1);
            if (cmd == "estimate") std::cout << estimate_frames(text, ref, frames) << "\n";
            else if (cmd == "prepare") std::cout << prepare_text(text) << "\n";
            else { for (auto & p : split_text(text, ref, frames)) std::cout << p << "\x1f"; std::cout << "\n"; }
        } else if (cmd == "resample") {
            std::string src, dst; int a, b; in >> src >> a >> b >> dst; write_f32(dst, resample_kaiser_best(read_f32(src), a, b));
            std::cout << "ok\n";
        } else if (cmd == "trim") {
            std::string src, dst; int sr; in >> src >> sr >> dst; write_f32(dst, trim_silence(read_f32(src), sr)); std::cout << "ok\n";
        } else if (cmd == "speech") {
            std::string src; int sr; in >> src >> sr; std::printf("%.10f\n", speech_length(read_f32(src), sr));
        } else if (cmd == "quiet") {
            std::string src, dst; in >> src >> dst; write_f32(dst, raise_quiet(read_f32(src))); std::cout << "ok\n";
        }
        std::cout.flush();
    }
}
