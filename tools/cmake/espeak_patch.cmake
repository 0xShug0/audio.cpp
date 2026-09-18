# Upstream 1.52 fetches Sonic even when USE_LIBSONIC=OFF. Respect that option
# so a phonemizer-only build has no unnecessary transitive network dependency.
set(path "${SOURCE_DIR}/cmake/deps.cmake")
file(READ "${path}" contents)
string(REPLACE "else()\n  FetchContent_Declare(sonic-git" "elseif(USE_LIBSONIC)\n  FetchContent_Declare(sonic-git" contents "${contents}")
file(WRITE "${path}" "${contents}")
set(path "${SOURCE_DIR}/src/CMakeLists.txt")
file(READ "${path}" contents)
if(NOT contents MATCHES "ARCHIVE_OUTPUT_NAME espeak-ng-tool")
    string(APPEND contents "\n# Avoid colliding with the static espeak-ng library on MSVC.\nset_target_properties(espeak-ng-bin PROPERTIES ARCHIVE_OUTPUT_NAME espeak-ng-tool)\n")
    file(WRITE "${path}" "${contents}")
endif()
