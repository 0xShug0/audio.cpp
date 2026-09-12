/* audio.cpp only needs miniaudio's in-memory decoding APIs (WAV/MP3/FLAC), so
 * the playback/capture device backends, the high level engine, and the
 * threading primitives that only those pieces need are compiled out. This
 * keeps the vendored build free of platform audio-driver dependencies
 * (ALSA, CoreAudio, WASAPI, ...) across the CI matrix. */
#define MA_NO_DEVICE_IO
#define MA_NO_ENCODING
#define MA_NO_ENGINE
#define MA_NO_NODE_GRAPH
#define MA_NO_RESOURCE_MANAGER
#define MA_NO_THREADING
#define MA_NO_GENERATION

#define MINIAUDIO_IMPLEMENTATION
#include "miniaudio.h"
