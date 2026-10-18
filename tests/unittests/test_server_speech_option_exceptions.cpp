#include "speech_option_exceptions.h"

#include <cmath>
#include <iostream>
#include <limits>

int main() {
    using minitts::server::is_neutral_speech_option;
    if (!is_neutral_speech_option("speed", 1.0)) return 1;
    for (double value : {0.0, -1.0, 0.5, 1.5, std::nextafter(1.0, 2.0),
                         std::numeric_limits<double>::infinity(),
                         std::numeric_limits<double>::quiet_NaN()}) {
        if (is_neutral_speech_option("speed", value)) return 1;
    }
    if (is_neutral_speech_option("temperature", 1.0)) return 1;
    std::cout << "speech option exceptions passed\n";
}
