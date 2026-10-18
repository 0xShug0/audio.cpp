set(TARGET audiocpp_cpp_httplib)

find_package(Threads REQUIRED)

add_library(${TARGET} STATIC httplib.cpp httplib.h)
target_include_directories(${TARGET} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_features(${TARGET} PUBLIC cxx_std_17)
target_link_libraries(${TARGET} PRIVATE Threads::Threads)

if (MSVC)
    target_compile_options(${TARGET} PRIVATE /w)
else()
    target_compile_options(${TARGET} PRIVATE -w)
endif()

# Match llama.cpp's TLS arrangement: cpp-httplib is vendored and BoringSSL is
# built with the opt-in native manager, so Hugging Face HTTPS does not require
# libcurl or system OpenSSL development packages. Packagers may explicitly use
# system OpenSSL instead.
if (NOT AUDIOCPP_USE_SYSTEM_OPENSSL)
    set(AUDIOCPP_BORINGSSL_VERSION "0.20260813.0" CACHE STRING
        "Pinned BoringSSL release used by the native package manager")
    set(AUDIOCPP_BORINGSSL_SHA256
        "37e23cb9a5fa54f01b07cadd653cebc1d1b235945439a6c334fea58ea47b5b0a"
        CACHE STRING "SHA256 of the pinned BoringSSL source archive")
    set(AUDIOCPP_BORINGSSL_ARCHIVE "" CACHE FILEPATH
        "Optional pre-fetched BoringSSL archive for offline/reproducible builds")

    if (AUDIOCPP_BORINGSSL_ARCHIVE)
        set(AUDIOCPP_BORINGSSL_URL "${AUDIOCPP_BORINGSSL_ARCHIVE}")
    else()
        set(AUDIOCPP_BORINGSSL_URL
            "https://github.com/google/boringssl/archive/refs/tags/${AUDIOCPP_BORINGSSL_VERSION}.tar.gz")
    endif()

    set(OPENSSL_NO_ASM ON CACHE BOOL "Disable BoringSSL assembly generation" FORCE)
    set(FIPS OFF CACHE BOOL "Disable BoringSSL FIPS build" FORCE)

    include(FetchContent)
    FetchContent_Declare(audiocpp_boringssl
        URL "${AUDIOCPP_BORINGSSL_URL}"
        URL_HASH "SHA256=${AUDIOCPP_BORINGSSL_SHA256}"
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    )

    set(AUDIOCPP_SAVED_BUILD_SHARED_LIBS ${BUILD_SHARED_LIBS})
    set(AUDIOCPP_SAVED_BUILD_TESTING ${BUILD_TESTING})
    set(BUILD_SHARED_LIBS OFF)
    set(BUILD_TESTING OFF)
    FetchContent_MakeAvailable(audiocpp_boringssl)
    set(BUILD_SHARED_LIBS ${AUDIOCPP_SAVED_BUILD_SHARED_LIBS})
    set(BUILD_TESTING ${AUDIOCPP_SAVED_BUILD_TESTING})

    # Some packagers (notably Nix) inject a global -Werror. Do not let the
    # parent project's warning policy turn compatibility warnings inside the
    # pinned third-party TLS implementation into audio.cpp build failures.
    # Discover targets recursively instead of naming only crypto/ssl: Nix's
    # default all-target build also compiles auxiliary libraries such as
    # decrepit, and BoringSSL may add further subdirectory targets over time.
    function(audiocpp_relax_boringssl_warnings AUDIOCPP_DIRECTORY)
        get_property(AUDIOCPP_DIRECTORY_TARGETS
            DIRECTORY "${AUDIOCPP_DIRECTORY}" PROPERTY BUILDSYSTEM_TARGETS)
        foreach(AUDIOCPP_BORINGSSL_TARGET IN LISTS AUDIOCPP_DIRECTORY_TARGETS)
            get_target_property(AUDIOCPP_BORINGSSL_TARGET_TYPE
                ${AUDIOCPP_BORINGSSL_TARGET} TYPE)
            if (NOT AUDIOCPP_BORINGSSL_TARGET_TYPE STREQUAL "INTERFACE_LIBRARY"
                    AND NOT AUDIOCPP_BORINGSSL_TARGET_TYPE STREQUAL "UTILITY")
                set_property(TARGET ${AUDIOCPP_BORINGSSL_TARGET}
                    PROPERTY COMPILE_WARNING_AS_ERROR OFF)
                if (MSVC)
                    target_compile_options(${AUDIOCPP_BORINGSSL_TARGET} PRIVATE /WX-)
                else()
                    target_compile_options(${AUDIOCPP_BORINGSSL_TARGET} PRIVATE
                        -Wno-error
                        -Wno-unused-parameter
                        # BoringSSL intentionally uses GNU compound literals
                        # in its vectorized C++ implementation. The parent
                        # enables -pedantic-errors globally, so keep that
                        # policy out of this pinned third-party dependency.
                        -Wno-pedantic
                        -Wno-error=pedantic)
                endif()
            endif()
        endforeach()

        get_property(AUDIOCPP_BORINGSSL_SUBDIRECTORIES
            DIRECTORY "${AUDIOCPP_DIRECTORY}" PROPERTY SUBDIRECTORIES)
        foreach(AUDIOCPP_BORINGSSL_SUBDIRECTORY
                IN LISTS AUDIOCPP_BORINGSSL_SUBDIRECTORIES)
            audiocpp_relax_boringssl_warnings(
                "${AUDIOCPP_BORINGSSL_SUBDIRECTORY}")
        endforeach()
    endfunction()
    audiocpp_relax_boringssl_warnings("${audiocpp_boringssl_SOURCE_DIR}")

    target_compile_definitions(${TARGET} PUBLIC CPPHTTPLIB_OPENSSL_SUPPORT)
    target_link_libraries(${TARGET} PUBLIC ssl crypto)
else()
    find_package(OpenSSL REQUIRED)
    target_compile_definitions(${TARGET} PUBLIC CPPHTTPLIB_OPENSSL_SUPPORT)
    target_link_libraries(${TARGET} PUBLIC OpenSSL::SSL OpenSSL::Crypto)
endif()

if (WIN32)
    target_compile_definitions(${TARGET} PUBLIC _WIN32_WINNT=0x0A00)
    target_link_libraries(${TARGET} PUBLIC ws2_32 crypt32)
elseif (APPLE)
    find_library(AUDIOCPP_CORE_FOUNDATION_FRAMEWORK CoreFoundation REQUIRED)
    find_library(AUDIOCPP_SECURITY_FRAMEWORK Security REQUIRED)
    target_link_libraries(${TARGET} PUBLIC
        ${AUDIOCPP_CORE_FOUNDATION_FRAMEWORK}
        ${AUDIOCPP_SECURITY_FRAMEWORK}
    )
endif()
