#include "engine/framework/audio/espeak_data.h"
#include <iostream>
int main(int argc, char ** argv) try {
    if (argc != 3) { std::cerr << "Usage: audiocpp_espeak_pack <espeak-ng-data> <output.bin>\n"; return 1; }
    engine::audio::pack_espeak_data(argv[1], argv[2]);
    return 0;
} catch (const std::exception & e) { std::cerr << e.what() << '\n'; return 1; }
