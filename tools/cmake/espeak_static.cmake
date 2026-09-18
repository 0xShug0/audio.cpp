# Keep upstream options and target names out of audio.cpp's CMake scope.
include(ExternalProject)
if(POLICY CMP0135)
    cmake_policy(SET CMP0135 NEW)
endif()
if(CMAKE_CROSSCOMPILING)
    message(FATAL_ERROR "Static eSpeak data compilation currently requires a native build")
endif()
set(_espeak_root "${CMAKE_BINARY_DIR}/_deps/espeak-static")
set(_espeak_source "${_espeak_root}/source")
set(_espeak_build "${_espeak_root}/build")
set(_espeak_lib "${_espeak_build}/lib/${CMAKE_STATIC_LIBRARY_PREFIX}espeak-ng${CMAKE_STATIC_LIBRARY_SUFFIX}")
set(_espeak_ucd "${_espeak_build}/lib/${CMAKE_STATIC_LIBRARY_PREFIX}ucd${CMAKE_STATIC_LIBRARY_SUFFIX}")
ExternalProject_Add(audiocpp_espeak_source
    URL https://codeload.github.com/espeak-ng/espeak-ng/tar.gz/refs/tags/1.52.0
    URL_HASH SHA256=bb4338102ff3b49a81423da8a1a158b420124b055b60fa76cfb4b18677130a23
    DOWNLOAD_DIR "${_espeak_root}/download"
    DOWNLOAD_NAME espeak-ng-1.52.0.tar.gz
    SOURCE_DIR "${_espeak_source}"
    BINARY_DIR "${_espeak_build}"
    PATCH_COMMAND ${CMAKE_COMMAND} -DSOURCE_DIR=<SOURCE_DIR>
        -P "${CMAKE_CURRENT_LIST_DIR}/espeak_patch.cmake"
    CMAKE_ARGS
        -DCMAKE_BUILD_TYPE=Release
        -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
        -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
        -DCMAKE_POLICY_VERSION_MINIMUM=3.5
        -DCMAKE_INSTALL_PREFIX=<BINARY_DIR>/install
        -DCMAKE_ARCHIVE_OUTPUT_DIRECTORY=<BINARY_DIR>/lib
        -DCMAKE_ARCHIVE_OUTPUT_DIRECTORY_RELEASE=<BINARY_DIR>/lib
        -DBUILD_SHARED_LIBS=OFF -DBUILD_TESTING=OFF
        -DUSE_ASYNC=OFF -DUSE_LIBSONIC=OFF -DUSE_LIBPCAUDIO=OFF
        -DUSE_MBROLA=OFF -DUSE_SPEECHPLAYER=OFF -DESPEAK_BUILD_MANPAGES=OFF
    BUILD_COMMAND ${CMAKE_COMMAND} --build <BINARY_DIR> --config Release --target data
    INSTALL_COMMAND ""
    LOG_BUILD ON
    BUILD_BYPRODUCTS "${_espeak_lib}" "${_espeak_ucd}")
ExternalProject_Add_StepDependencies(audiocpp_espeak_source patch "${CMAKE_CURRENT_LIST_DIR}/espeak_patch.cmake")
add_library(audiocpp_espeak_static STATIC IMPORTED)
set_target_properties(audiocpp_espeak_static PROPERTIES IMPORTED_LOCATION "${_espeak_lib}")
add_library(audiocpp_espeak_ucd STATIC IMPORTED)
set_target_properties(audiocpp_espeak_ucd PROPERTIES IMPORTED_LOCATION "${_espeak_ucd}")
add_dependencies(audiocpp_espeak_static audiocpp_espeak_source)
add_dependencies(engine_core audiocpp_espeak_source)
target_include_directories(engine_core PRIVATE "${_espeak_source}/src/include")
target_compile_definitions(engine_core PRIVATE AUDIOCPP_STATIC_ESPEAK=1 LIBESPEAK_NG_EXPORT=1)
target_link_libraries(engine_runtime PRIVATE audiocpp_espeak_static audiocpp_espeak_ucd)
if(UNIX)
    target_link_libraries(engine_runtime PRIVATE m)
endif()
# Data stays separate and relocatable. These are not executable model weights.
add_executable(audiocpp_espeak_pack tools/espeak_data_pack.cpp src/framework/audio/espeak_data.cpp)
target_include_directories(audiocpp_espeak_pack PRIVATE "${PROJECT_SOURCE_DIR}/include")
target_link_libraries(audiocpp_espeak_pack PRIVATE ggml-base)
set(_espeak_package "${_espeak_root}/espeak-ng-data.bin")
add_custom_command(OUTPUT "${_espeak_package}"
    COMMAND $<TARGET_FILE:audiocpp_espeak_pack> "${_espeak_build}/espeak-ng-data" "${_espeak_package}"
    DEPENDS audiocpp_espeak_pack audiocpp_espeak_source
    VERBATIM)
add_custom_target(audiocpp_espeak_package DEPENDS "${_espeak_package}")
# One staging target shared by CLI/server/probes. Run on every requested build,
# even when only the data package changed and executables do not need relinking.
set(_espeak_bin "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}")
if(CMAKE_CONFIGURATION_TYPES)
    string(APPEND _espeak_bin "/$<CONFIG>")
endif()
add_custom_target(audiocpp_espeak_stage
    COMMAND ${CMAKE_COMMAND} -E make_directory "${_espeak_bin}/licenses/espeak-ng"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "${_espeak_package}" "${_espeak_bin}/espeak-ng-data.bin"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "${_espeak_source}/COPYING"
        "${_espeak_root}/download/espeak-ng-1.52.0.tar.gz"
        "${PROJECT_SOURCE_DIR}/tools/cmake/espeak_patch.cmake"
        "${PROJECT_SOURCE_DIR}/tools/cmake/espeak_static.cmake"
        "${PROJECT_SOURCE_DIR}/docs/espeak_phonemizer.md"
        "${_espeak_bin}/licenses/espeak-ng/"
    DEPENDS audiocpp_espeak_package
    VERBATIM)
function(audiocpp_stage_espeak target)
    add_dependencies(${target} audiocpp_espeak_stage)
endfunction()
