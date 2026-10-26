#pragma once

#include <iosfwd>

namespace minitts::app {

const char * build_version();
const char * build_git_sha();
const char * build_git_date();

void print_build_info(std::ostream & out);
void print_build_info_summary(std::ostream & out);

}  // namespace minitts::app
