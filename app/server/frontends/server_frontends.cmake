option(AUDIOCPP_BUILD_SERVER_FRONTENDS
    "Build optional in-process audiocpp_server frontend adapters"
    OFF)
set(AUDIOCPP_SERVER_FRONTEND_MODULES ""
    CACHE STRING "Semicolon-separated optional audiocpp_server frontend modules to build")
if (NOT AUDIOCPP_BUILD_SERVER_FRONTENDS AND AUDIOCPP_SERVER_FRONTEND_MODULES)
    message(FATAL_ERROR
        "AUDIOCPP_SERVER_FRONTEND_MODULES requires AUDIOCPP_BUILD_SERVER_FRONTENDS=ON")
endif()

set(AUDIOCPP_SERVER_FRONTENDS_SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}")

function(audiocpp_configure_server_frontends AUDIOCPP_SERVER_TARGET)
    set(AUDIOCPP_SERVER_FRONTEND_DECLARATIONS "")
    set(AUDIOCPP_SERVER_FRONTEND_REGISTRATIONS "")

    if (AUDIOCPP_BUILD_SERVER_FRONTENDS)
        set(AUDIOCPP_SERVER_FRONTEND_SOURCES "")
        set(AUDIOCPP_SERVER_FRONTEND_INCLUDE_DIRS "")
        set(AUDIOCPP_SERVER_FRONTEND_LIBRARIES "")
        foreach(AUDIOCPP_SERVER_FRONTEND_MODULE IN LISTS AUDIOCPP_SERVER_FRONTEND_MODULES)
            if (AUDIOCPP_SERVER_FRONTEND_MODULE STREQUAL "audio_decode")
                string(APPEND AUDIOCPP_SERVER_FRONTEND_DECLARATIONS
                    "void register_audio_decode_module(ServerFrontendRegistry & registry);\n")
                string(APPEND AUDIOCPP_SERVER_FRONTEND_REGISTRATIONS
                    "    register_audio_decode_module(registry);\n")
                list(APPEND AUDIOCPP_SERVER_FRONTEND_SOURCES
                    "${AUDIOCPP_SERVER_FRONTENDS_SOURCE_DIR}/audio_decode.cpp")
                list(APPEND AUDIOCPP_SERVER_FRONTEND_INCLUDE_DIRS
                    "${PROJECT_SOURCE_DIR}/external/miniaudio")
            elseif (AUDIOCPP_SERVER_FRONTEND_MODULE STREQUAL "mp3_encode")
                find_path(AUDIOCPP_LAME_INCLUDE_DIR
                    NAMES lame/lame.h
                    HINTS "$ENV{CONDA_PREFIX}"
                    PATH_SUFFIXES include)
                find_library(AUDIOCPP_LAME_LIBRARY
                    NAMES mp3lame lame
                    HINTS "$ENV{CONDA_PREFIX}"
                    PATH_SUFFIXES lib)
                if (NOT AUDIOCPP_LAME_INCLUDE_DIR OR NOT AUDIOCPP_LAME_LIBRARY)
                    message(FATAL_ERROR
                        "AUDIOCPP_SERVER_FRONTEND_MODULES=mp3_encode requires libmp3lame headers and library "
                        "(install libmp3lame-dev or build from an environment that provides lame/lame.h)")
                endif()
                string(APPEND AUDIOCPP_SERVER_FRONTEND_DECLARATIONS
                    "void register_mp3_encode_module(ServerFrontendRegistry & registry);\n")
                string(APPEND AUDIOCPP_SERVER_FRONTEND_REGISTRATIONS
                    "    register_mp3_encode_module(registry);\n")
                list(APPEND AUDIOCPP_SERVER_FRONTEND_SOURCES
                    "${AUDIOCPP_SERVER_FRONTENDS_SOURCE_DIR}/mp3_encode.cpp")
                list(APPEND AUDIOCPP_SERVER_FRONTEND_INCLUDE_DIRS
                    "${AUDIOCPP_LAME_INCLUDE_DIR}")
                list(APPEND AUDIOCPP_SERVER_FRONTEND_LIBRARIES
                    "${AUDIOCPP_LAME_LIBRARY}")
            else()
                message(FATAL_ERROR
                    "Unknown AUDIOCPP_SERVER_FRONTEND_MODULES entry: ${AUDIOCPP_SERVER_FRONTEND_MODULE}")
            endif()
        endforeach()

        if (AUDIOCPP_SERVER_FRONTEND_SOURCES)
            list(REMOVE_DUPLICATES AUDIOCPP_SERVER_FRONTEND_INCLUDE_DIRS)
            list(REMOVE_DUPLICATES AUDIOCPP_SERVER_FRONTEND_LIBRARIES)
            add_library(audiocpp_server_frontends OBJECT
                ${AUDIOCPP_SERVER_FRONTEND_SOURCES}
            )
            target_include_directories(audiocpp_server_frontends PRIVATE
                ${AUDIOCPP_SERVER_FRONTEND_INCLUDE_DIRS}
            )
            target_link_libraries(audiocpp_server_frontends PUBLIC engine_runtime)
            target_sources(${AUDIOCPP_SERVER_TARGET} PRIVATE $<TARGET_OBJECTS:audiocpp_server_frontends>)
            target_link_libraries(${AUDIOCPP_SERVER_TARGET} PRIVATE
                audiocpp_server_frontends
                ${AUDIOCPP_SERVER_FRONTEND_LIBRARIES})
        endif()
    endif()

    file(WRITE
        "${CMAKE_CURRENT_BINARY_DIR}/generated/server_frontend_module_declarations.inc"
        "${AUDIOCPP_SERVER_FRONTEND_DECLARATIONS}")
    file(WRITE
        "${CMAKE_CURRENT_BINARY_DIR}/generated/server_frontend_module_registrations.inc"
        "${AUDIOCPP_SERVER_FRONTEND_REGISTRATIONS}")
endfunction()
